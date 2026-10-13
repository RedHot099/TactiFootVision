"""G-combo's side of the frozen test (xg-detector 06): its 10-model ensemble probability for every index shot of the 172
held-out matches, with its bundles and ensemble.json untouched. GPU for the unpaired shots only (inference).

    ./run.sh python scripts/xgd_gcombo_test.py      # -> results/xgd_gcombo_test/{predictions.csv, unpaired_members.csv, checks.json, complete.json}

Population: every index v1.2 shot (penalties included; the test leaves them out) of the plan's evaluation and gate
matches, read from label-free index columns. Each shot gets its probability from one source:

* paired: the 2,754 paired shots of the evaluation matches, p = sigmoid(frame_logit_0) of xg-vendor's evaluation table
  (results/xgv_eval_primary, labels absent). Byte check: the table's sha256 equals its completion record and its
  metadata, the record names this ensemble.json and these ten bundles. Cross-check: `ensemble_probability` on the ten
  stored member logits of the same folder's scores.parquet reproduces it.
* gate: the 825 gate shots, `ensemble_probability` of the ten stored gate predictions (each file checked against its
  completion record and p = sigmoid(logit / T)), cross-checked against scores.parquet's gate rows.
* unpaired: the other index shots of the evaluation matches (285), scored here by the ten bundles (`g_bundle.g_outputs`,
  the t0 frame, batch 8, eval mode), then `ensemble_probability`.

Path check: in the same process each bundle also rescores the gate shots in their stored row order; the logits must
reproduce the stored gate predictions within `PATH_TOLERANCE`. Nothing here reads a label: the index is read without
label columns, the gate files without their goal column, and everything runs inside the held-out label seal. The
evaluation table and scores.parquet are G-combo outputs that xg-vendor stored next to StatsBomb columns: they are read
first, outside `no_statsbomb_reads()`, with only their key, logit, role and status columns; the gate files, the bundles
and the frames are read inside the guard. No new model reads them.
"""

import argparse
import gc
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from run_xgc_g import frame_fingerprint, own_sources
from xgd_approach1 import check_gpu_free
from xgd_approach1_report import G_COMBO, G_NAME
from xgpoc.bundle import sha256_file
from xgpoc.detector import approach1 as a1
from xgpoc.detector import records, seal
from xgpoc.detector.guard import no_statsbomb_reads
from xgpoc.eval.calibration import sigmoid
from xgpoc.models.g_bundle import ensemble_probability, g_outputs, load_g
from xgpoc.paths import INDEX

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "results/xgv_eval_primary"
TABLE = "eval_table_labels_absent.parquet"
ROLES = ("evaluation", "gate")
PATH_TOLERANCE = 1e-4            # max |logit difference| of the gate rescoring in the stored order (replays give ~2e-7)
CROSS_TOLERANCE = 1e-6           # max |p difference| against xg-vendor's scores.parquet


def held_out_shots(index: Path, plan: pd.DataFrame) -> pd.DataFrame:
    """shot_id, game, role, is_penalty, frame_path of every index shot of the plan's evaluation and gate matches, sorted by
    shot_id; label-free columns only."""
    shots = pd.read_parquet(index, columns=["shot_id", "game", "is_penalty", "frame_path"])
    shots["role"] = shots.game.map(plan.set_index("game").role)
    out = shots[shots.role.isin(ROLES)].sort_values("shot_id").reset_index(drop=True)
    out["is_penalty"] = out.is_penalty.astype(bool)
    return out


def read_ensemble(root: Path = G_COMBO) -> dict:
    """ensemble.json with every member's bundle checked against its sha256 and its run's completion record."""
    ensemble = json.loads((root / "ensemble.json").read_text())
    if sorted((m["fold"], m["seed"]) for m in ensemble["members"]) != [(k, s) for k in range(5) for s in (0, 1)]:
        raise ValueError("G-combo's ensemble.json does not list the ten fold models")
    for m in ensemble["members"]:
        run = root / f"fold{m['fold']}" / f"seed{m['seed']}"
        if json.loads((run / "complete.json").read_text())["files"]["bundle.pt"] != m["sha256"]:
            raise ValueError(f"{run}: completion record and ensemble.json name different bundles")
    ensemble["members"] = sorted(ensemble["members"], key=lambda m: (m["fold"], m["seed"]))
    return ensemble


def key(m: dict) -> str:
    return f"f{m['fold']}s{m['seed']}"


def paired_predictions(population: pd.DataFrame, ensemble: dict, ensemble_sha: str) -> tuple[pd.Series, dict]:
    """shot_id -> p of the evaluation table's paired shots, after the byte checks, and the cross-check record."""
    record = json.loads((EVAL / "complete.json").read_text())
    meta = json.loads((EVAL / TABLE.replace(".parquet", ".json")).read_text())
    digest = sha256_file(EVAL / TABLE)
    if not digest == record["files"][TABLE] == meta["table_sha256"] or meta["labels_state"] != "absent":
        raise ValueError(f"{EVAL / TABLE} is not the recorded label-free table")
    if record["identity"]["ensemble_json_sha256"] != ensemble_sha or meta["sources"]["ensemble.json"] != ensemble_sha:
        raise ValueError("the evaluation table was not made with this ensemble.json")
    if record["identity"]["bundles"] != {key(m): m["sha256"] for m in ensemble["members"]}:
        raise ValueError("the evaluation table was not made with these ten bundles")
    table = pd.read_parquet(EVAL / TABLE, columns=["soccernet_shot_id", "frame_logit_0", "frame_status"])
    if table.soccernet_shot_id.duplicated().any() or (table.frame_status != "ok").any() or not np.isfinite(table.frame_logit_0).all():
        raise ValueError("the evaluation table has duplicate, unscored or non-finite paired shots")
    roles = population.set_index("shot_id").role.reindex(table.soccernet_shot_id)
    if (roles != "evaluation").any():
        raise ValueError(f"{int((roles != 'evaluation').sum())} paired shots are not index shots of the evaluation matches")
    p = pd.Series(sigmoid(table.frame_logit_0.to_numpy()), index=table.soccernet_shot_id.to_numpy())
    scores = cross_scores(population, ensemble, "primary")
    gap = float(np.max(np.abs(scores.loc[p.index].to_numpy() - p.to_numpy())))
    if gap > CROSS_TOLERANCE:
        raise ValueError(f"the evaluation table's paired p differ from its member logits by {gap:.2e}")
    return p, {"shots": len(p), "table_sha256": digest, "max_abs_dp_vs_member_logits": gap}


def cross_scores(population: pd.DataFrame, ensemble: dict, role: str) -> pd.Series:
    """shot_id -> `ensemble_probability` of the ten stored member logits of xg-vendor's scores.parquet (rows of `role`)."""
    record = json.loads((EVAL / "complete.json").read_text())
    if sha256_file(EVAL / "scores.parquet") != record["files"]["scores.parquet"]:
        raise ValueError(f"{EVAL / 'scores.parquet'} does not match its completion record")
    columns = [f"xg_{key(m)}_0" for m in ensemble["members"]]
    scores = pd.read_parquet(EVAL / "scores.parquet", columns=["key", "role", *columns])
    scores = scores[scores.role == role].set_index("key")
    penalty = population.set_index("shot_id").is_penalty.reindex(scores.index)
    if penalty.isna().any():
        raise ValueError(f"scores.parquet has {role} shots outside the population")
    p = ensemble_probability([scores[c].to_numpy() for c in columns], [m["temperature"] for m in ensemble["members"]],
                             ensemble["calibration"], penalty.to_numpy(bool))
    return pd.Series(p, index=scores.index)


def stored_gate(population: pd.DataFrame, ensemble: dict) -> tuple[pd.DataFrame, dict]:
    """(shot_id x member stored gate logits in the stored row order, the check record); goal columns never read."""
    gate_ids = set(population.shot_id[population.role == "gate"])
    logits, order = {}, None
    for m in ensemble["members"]:
        path = G_COMBO / f"fold{m['fold']}" / f"seed{m['seed']}" / f"{G_NAME}_gate_predictions.csv"
        records.verified_file(path)
        frame = pd.read_csv(path, usecols=["shot_id", "logit", "p"])
        if order is None:
            order = frame.shot_id.tolist()
        if frame.shot_id.tolist() != order or set(order) != gate_ids or len(order) != len(gate_ids):
            raise ValueError(f"{path}: the gate rows are not the plan's gate shots in one shared order")
        if (gap := float(np.max(np.abs(sigmoid(frame.logit / m["temperature"]) - frame.p)))) > 1e-6:
            raise ValueError(f"{path}: p is not sigmoid(logit / T) (max gap {gap:.2e})")
        logits[key(m)] = frame.logit.to_numpy()
    return pd.DataFrame(logits, index=pd.Index(order, name="shot_id")), {"shots": len(order)}


def score_members(ensemble: dict, frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """name -> shot_id x member logits of each frame table, every bundle loaded once and run over the tables in order."""
    out = {name: {} for name in frames}
    for m in ensemble["members"]:
        path = G_COMBO / f"fold{m['fold']}" / f"seed{m['seed']}" / "bundle.pt"
        if sha256_file(path) != m["sha256"]:
            raise ValueError(f"{path} is not the bundle ensemble.json lists")
        bundle = load_g(path)
        for name, rows in frames.items():
            z, _, _ = g_outputs(bundle.model, rows.reset_index(drop=True).assign(goal=False), bundle.spec,
                                batch_size=bundle.recipe["cfg"]["batch_size"])
            out[name][key(m)] = z
        print(f"scored {key(m)}", flush=True)
        del bundle
        gc.collect()
        torch.cuda.empty_cache()
    return {name: pd.DataFrame(values, index=pd.Index(frames[name].shot_id, name="shot_id")) for name, values in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--out", type=Path, default=ROOT / "results/xgd_gcombo_test")
    args = parser.parse_args()
    if (args.out / "complete.json").exists():
        raise SystemExit(f"{args.out} is complete; G-combo's test predictions are written once")
    check_gpu_free()

    with seal.development_labels_only(a1.PLAN_DIR / "plan.csv", args.index):
        plan = a1.verify_plan(args.index)
        population = held_out_shots(args.index, plan)
        ensemble = read_ensemble()
        ensemble_sha = sha256_file(G_COMBO / "ensemble.json")
        paired, paired_check = paired_predictions(population, ensemble, ensemble_sha)
        gate_reference = cross_scores(population, ensemble, "gate")      # xg-vendor's outputs: read before the guard
        eval_hashes = {"eval_table": paired_check["table_sha256"], "eval_complete_json": sha256_file(EVAL / "complete.json"),
                       "scores_parquet": sha256_file(EVAL / "scores.parquet")}
        with no_statsbomb_reads():                                         # G-combo's files, models and frames
            gate_logits, gate_check = stored_gate(population, ensemble)
            unpaired = population[(population.role == "evaluation") & ~population.shot_id.isin(paired.index)]
            frames = population.set_index("shot_id")
            sources = own_sources()
            scored = score_members(ensemble, {"gate": frames.loc[gate_logits.index].reset_index(),
                                              "unpaired": unpaired.reset_index(drop=True)})
            if own_sources() != sources:
                raise SystemExit("source files changed while scoring")
            path_gap = float(np.max(np.abs(scored["gate"].to_numpy() - gate_logits.loc[scored["gate"].index].to_numpy())))
            if path_gap > PATH_TOLERANCE:
                raise SystemExit(f"the gate rescoring differs from the stored gate logits by {path_gap:.2e}")
            temperatures = [m["temperature"] for m in ensemble["members"]]

            def probability(logits: pd.DataFrame) -> pd.Series:
                penalty = frames.is_penalty.loc[logits.index].to_numpy(bool)
                return pd.Series(ensemble_probability([logits[key(m)].to_numpy() for m in ensemble["members"]], temperatures,
                                                      ensemble["calibration"], penalty), index=logits.index)

            gate = probability(gate_logits)
            gate_cross = float(np.max(np.abs(gate_reference.loc[gate.index] - gate)))
            if gate_cross > CROSS_TOLERANCE:
                raise SystemExit(f"the stored gate predictions' ensemble differs from scores.parquet by {gate_cross:.2e}")
            parts = [pd.DataFrame({"p": paired, "source": "paired"}), pd.DataFrame({"p": gate, "source": "gate"}),
                     pd.DataFrame({"p": probability(scored["unpaired"]), "source": "unpaired"})]
            predictions = pd.concat(parts).rename_axis("shot_id").reset_index()
            predictions = population[["shot_id", "game", "role", "is_penalty"]].merge(predictions, on="shot_id", how="left",
                                                                                       validate="one_to_one")
            if predictions.p.isna().any() or len(predictions) != len(population) or not predictions.p.between(0, 1).all():
                raise SystemExit("the G-combo test predictions do not cover the held-out shots exactly once")
            non_penalty = predictions[~predictions.is_penalty]
            checks = {
                "population": {"shots": len(predictions), "matches": int(predictions.game.nunique()),
                               "by_source": predictions.source.value_counts().to_dict(),
                               "non_penalty": {"shots": len(non_penalty), "matches": int(non_penalty.game.nunique()),
                                               "by_source": non_penalty.source.value_counts().to_dict(),
                                               "by_role": non_penalty.role.value_counts().to_dict()}},
                "paired": paired_check,
                "gate": {**gate_check, "max_abs_dp_vs_scores_parquet": gate_cross},
                "path_check": {"rule": "each bundle rescored the gate shots in their stored order in this process",
                               "max_abs_logit_vs_stored": path_gap, "tolerance": PATH_TOLERANCE},
                "unpaired": {"shots": len(unpaired), "batch_size": 8, "frame": "frame_path (t0)", "order": "shot_id"},
            }
            args.out.mkdir(parents=True, exist_ok=True)
            predictions.to_csv(args.out / "predictions.csv", index=False, float_format="%.17g")
            scored["unpaired"].reset_index().to_csv(args.out / "unpaired_members.csv", index=False, float_format="%.9g")
            (args.out / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
            inputs = {"index": sha256_file(args.index), "plan": a1.PLAN_SHA256, "ensemble_json": ensemble_sha,
                      "bundles": {key(m): m["sha256"] for m in ensemble["members"]},
                      **eval_hashes,
                      "gate_predictions": {key(m): sha256_file(G_COMBO / f"fold{m['fold']}" / f"seed{m['seed']}" / f"{G_NAME}_gate_predictions.csv")
                                           for m in ensemble["members"]},
                      "unpaired_frames": frame_fingerprint(unpaired)}
            record = {"inputs": inputs, "code": records.code_hashes(Path(__file__)), "sources": sources, "git": records.git_state(),
                      "finished": datetime.now(UTC).isoformat(timespec="seconds"), "files": records.output_hashes(args.out)}
            (args.out / "complete.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
