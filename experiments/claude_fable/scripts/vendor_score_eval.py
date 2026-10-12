"""Score the primary evaluation shots and the gate shots with the retrained G-combo and write the label-free evaluation
table (xg-vendor ticket 11; the scoring stage is approved GPU work, decision D2: scoring only, no training).

    ./run.sh python scripts/vendor_score_eval.py score --gpu-approved    # GPU: ten fold models x two offsets + DINOv2
    ./run.sh python scripts/vendor_score_eval.py replay                  # GPU: a sample again in identical batch order
    ./run.sh python scripts/vendor_score_eval.py build [--wait-minutes 60]   # CPU: ensemble, gate check, join, table

Stage `score`. Checks that no other job holds the GPU, that the split plan is the approved one and that every bundle is the
one `results/xgv_g_combo/ensemble.json` lists. The rows are the 2,754 paired shots (the frozen pairing; penalties included)
and the 825 gate shots; the frames are index v1.2 `frame_path` and `frame_path_-400` through the dataset the models were
trained on. A fold model that is done is kept in `members/<fold>s<seed>.npz`, so an interrupted run resumes at the next
model. The pooled embeddings of the seed-0 models of folds 0 and 1 and the frozen DINOv2 embeddings are written for the
evaluation shots at offset 0.

Stage `replay`. Reloads every bundle in a fresh process and scores the first 64 scored rows of each role and offset again
(the same batches as the first of the original run); logits, probabilities (<= 1e-5) and the stored embeddings must agree.

Stage `build`. Averages the ten models, applies the single ensemble calibration, checks the gate shots against the stored
gate predictions of the fold models, joins vendor xG and the frozen comparator rungs (waits for its completion record),
writes `eval_table_labels_absent.*` (kind confirmatory, labels absent, validated, hashed), `scores.parquet` and the
completion record. No label is read or joined; no metric is computed on the evaluation shots.
"""

import argparse
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow
import torch
from vendor_retrain_g import approved_split, check_gpu_free
from xgpoc.bundle import code_version, read_bundle, sha256_file
from xgpoc.eval.calibration import sigmoid
from xgpoc.models.g_bundle import build_g, ensemble_probability, load_g
from xgpoc.paths import INDEX
from xgpoc.probe.scoring import SharedScorer
from xgpoc.vendor import eval_scoring as es
from xgpoc.vendor import evaltable as et
from xgpoc.vendor import split_check
from xgpoc.vendor.statsbomb_fetch import file_sha256

ROOT = Path(__file__).resolve().parents[1]   # experiments/claude_fable
OUT = ROOT / "results/xgv_eval_primary"
G_COMBO = ROOT / "results/xgv_g_combo"
PAIRING = ROOT / "results/xgv_pairing"
COMPARATOR = ROOT / "results/xgv_comparator"
TABLE_NAME = "eval_table_labels_absent"
REPLAY_TOL = 1e-5   # probability scale, as the bundles' own replay
GATE_TOL = 1e-6     # ensemble probability on the gate shots against the stored gate predictions of the fold models
NUM_WORKERS = 4
SOURCE_FILES = (Path(__file__).resolve(), ROOT / "xgpoc/vendor/eval_scoring.py", ROOT / "xgpoc/vendor/evaltable.py")
VERSION = "xgv_eval_primary_v1"


def json_text(value) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


# ------------------------------------------------------------------ inputs
def read_ensemble(g_root: Path) -> dict:
    """ensemble.json, after checking its plan and that each listed bundle file has the recorded hash."""
    ensemble = json.loads((g_root / "ensemble.json").read_text())
    if ensemble["plan"]["sha256"] != split_check.PLAN_SHA256:
        raise SystemExit("the G-combo was trained on another split plan than the approved one")
    if sorted((m["fold"], m["seed"]) for m in ensemble["members"]) != sorted(es.MEMBERS):
        raise SystemExit("ensemble.json does not list the ten fold models")
    for member in ensemble["members"]:
        bundle = g_root / f"fold{member['fold']}" / f"seed{member['seed']}" / "bundle.pt"
        if sha256_file(bundle) != member["sha256"]:
            raise SystemExit(f"{bundle} changed since ensemble.json")
    return ensemble


def read_pairing(root: Path) -> tuple[pd.DataFrame, dict]:
    """The pair table and the pairing's identity, after checking pairs.parquet against the completion record."""
    record = json.loads((root / "complete.json").read_text())
    if file_sha256(root / "pairs.parquet") != record["files"]["pairs.parquet"]:
        raise SystemExit("pairs.parquet differs from the pairing's completion record")
    return pd.read_parquet(root / "pairs.parquet"), {"complete_json_sha256": file_sha256(root / "complete.json"),
                                                      "pairs_sha256": record["files"]["pairs.parquet"]}


def inputs(args) -> dict:
    """Everything the stages share: shots, ensemble, pairing identity, gate order."""
    _, gate, plan, _ = approved_split(args.index, False, args.out)   # verifies the frozen plan; the pool is not used
    pairs, pairing = read_pairing(args.pairing)
    index = pd.read_parquet(args.index)
    shots = es.shots_frame(index, pairs, gate)
    return {"shots": shots, "pairs": pairs, "pairing": pairing, "plan": plan, "ensemble": read_ensemble(args.g_root)}


def member_dir(args) -> Path:
    return args.out / "members"


def bundle_path(args, fold: int, seed: int) -> Path:
    return args.g_root / f"fold{fold}" / f"seed{seed}" / "bundle.pt"


def save_atomic(path: Path, **arrays) -> None:
    stage = path.with_name(path.stem + ".partial.npz")
    np.savez(stage, **arrays)
    stage.rename(path)


# ------------------------------------------------------------------ stage score
def score_member(args, shots: pd.DataFrame, fold: int, seed: int, bundle_sha: str) -> None:
    path = member_dir(args) / f"{es.member_name(fold, seed)}.npz"
    if path.exists() and str(np.load(path)["bundle_sha256"]) == bundle_sha:
        print(f"{path.name} done", flush=True)
        return
    started = time.time()
    loaded = load_g(bundle_path(args, fold, seed))
    batch = loaded.recipe["cfg"]["batch_size"]
    keep_embedding = seed == 0 and fold in es.EMBEDDING_FOLDS
    xg, on_target = np.full((2, len(shots)), np.nan), np.full((2, len(shots)), np.nan)
    for i, name in enumerate(es.OFFSETS):
        for role in es.scoring_roles(shots):   # separate calls: the gate shots keep the batches of the retrain
            rows = shots.role.eq(role).to_numpy()
            wanted = keep_embedding and name == "0" and role != es.GATE
            z, o, e = es.score_rows(loaded.model, loaded.spec, shots[rows], name, batch_size=batch, num_workers=NUM_WORKERS, embedding=wanted)
            xg[i, rows], on_target[i, rows] = z, o
            if wanted:
                np.save(args.out / f"emb_g_fold{fold}.npy", e)
    save_atomic(path, xg=xg, on_target=on_target, bundle_sha256=np.str_(bundle_sha))
    del loaded
    torch.cuda.empty_cache()
    print(f"{path.name} scored in {(time.time() - started) / 60:.1f} min", flush=True)


def score_dino(args, shots: pd.DataFrame, spec: dict) -> None:
    path = args.out / "emb_dinov2.npy"
    if path.exists():
        print(f"{path.name} done", flush=True)
        return
    pretrained, _ = build_g(spec)   # the untouched pretrained network: the frozen encoder
    scorer = SharedScorer(pretrained.eval(), spec["model"]["n_blocks"], "cuda")
    rows = es.embedding_rows(shots)
    embedding = es.frozen_embeddings(scorer, spec, rows, "0", num_workers=NUM_WORKERS)
    np.save(path, embedding)
    print(f"{path.name}: {embedding.shape}", flush=True)


def stage_score(args) -> None:
    run_score(args, inputs)


def run_score(args, read_inputs) -> None:
    """The score stage for the rows `read_inputs(args)` returns ({"shots", "ensemble", ...}); ticket 13 passes its own."""
    if not args.gpu_approved:
        raise SystemExit("this stage uses the GPU: pass --gpu-approved (decision D2, scoring only)")
    check_gpu_free()
    args.out.mkdir(parents=True, exist_ok=True)
    member_dir(args).mkdir(exist_ok=True)
    state = read_inputs(args)
    shots = state["shots"]
    es.embedding_rows(shots)   # refuses two evaluation roles before any GPU time is spent
    stored = args.out / "shots.parquet"
    if stored.exists():
        if not pd.read_parquet(stored).equals(shots):
            raise SystemExit(f"{stored} differs from the rows and frames found now; refusing to continue a changed run")
    else:
        shots.to_parquet(stored, index=False)
    print(f"{len(shots)} rows ({shots.role.value_counts().to_dict()}); frame status offset 0 "
          f"{shots.status_0.value_counts().to_dict()}, -400 {shots.status_m400.value_counts().to_dict()}", flush=True)
    for member in sorted(state["ensemble"]["members"], key=lambda m: (m["fold"], m["seed"])):
        score_member(args, shots, member["fold"], member["seed"], member["sha256"])
    score_dino(args, shots, read_bundle(bundle_path(args, 0, 0))["spec"])
    if not es.frames_unchanged(shots):
        raise SystemExit("input frames changed while scoring")


# ------------------------------------------------------------------ stage replay
def stage_replay(args) -> None:
    check_gpu_free()
    shots = pd.read_parquet(args.out / "shots.parquet")
    ensemble = read_ensemble(args.g_root)
    rows = {}
    for role in es.scoring_roles(shots):
        for name in es.OFFSETS:
            sub = shots[shots.role.eq(role) & es.scored_mask(shots, name)].head(es.REPLAY_ROWS)
            rows[role, name] = sub
    worst = {}
    for member in sorted(ensemble["members"], key=lambda m: (m["fold"], m["seed"])):
        fold, seed, t = member["fold"], member["seed"], member["temperature"]
        stored = np.load(member_dir(args) / f"{es.member_name(fold, seed)}.npz")
        loaded = load_g(bundle_path(args, fold, seed))
        if abs(loaded.calibration["temperature"] - t) > 1e-12:
            raise SystemExit(f"fold {fold} seed {seed}: the bundle's temperature differs from ensemble.json")
        embeddings = np.load(args.out / f"emb_g_fold{fold}.npy", mmap_mode="r") if seed == 0 and fold in es.EMBEDDING_FOLDS else None
        dlogit = dp = demb = 0.0
        for (role, name), sub in rows.items():
            i = list(es.OFFSETS).index(name)
            positions = sub.index.to_numpy()
            z, o, e = es.score_rows(loaded.model, loaded.spec, sub, name, batch_size=loaded.recipe["cfg"]["batch_size"],
                                    num_workers=NUM_WORKERS, embedding=embeddings is not None and role != es.GATE and name == "0")
            dlogit = max(dlogit, float(np.abs(z - stored["xg"][i, positions]).max()), float(np.abs(o - stored["on_target"][i, positions]).max()))
            dp = max(dp, float(np.abs(sigmoid(z / t) - sigmoid(stored["xg"][i, positions] / t)).max()))
            if e is not None:   # the first rows of the evaluation role are the first rows of the embedding matrix
                demb = float(np.abs(e.astype(np.float32) - embeddings[: len(e)].astype(np.float32)).max())
        worst[es.member_name(fold, seed)] = {"max_abs_dlogit": dlogit, "max_abs_dp": dp, "max_abs_d_embedding": demb}
        print(es.member_name(fold, seed), worst[es.member_name(fold, seed)], flush=True)
        del loaded
        torch.cuda.empty_cache()
    spec = read_bundle(bundle_path(args, 0, 0))["spec"]
    pretrained, _ = build_g(spec)
    scorer = SharedScorer(pretrained.eval(), spec["model"]["n_blocks"], "cuda")
    sub = rows[es.scoring_roles(shots)[0], "0"]   # the first role holds the first rows of the embedding matrices
    fresh = es.frozen_embeddings(scorer, spec, sub, "0", num_workers=NUM_WORKERS).astype(np.float32)
    dino = float(np.abs(fresh - np.load(args.out / "emb_dinov2.npy", mmap_mode="r")[: len(fresh)].astype(np.float32)).max())
    result = {"rows_per_role_offset": es.REPLAY_ROWS, "members": worst, "dinov2_max_abs_d_embedding": dino,
              "max_abs_dp": max(v["max_abs_dp"] for v in worst.values()), "tolerance_dp": REPLAY_TOL,
              "passed": bool(max(v["max_abs_dp"] for v in worst.values()) <= REPLAY_TOL)}
    (args.out / "replay.json").write_text(json_text(result))
    print(json_text(result), flush=True)
    if not result["passed"]:
        raise SystemExit(f"replay error exceeds {REPLAY_TOL:g}")


# ------------------------------------------------------------------ stage build
def load_members(args, ensemble: dict) -> tuple[dict, dict]:
    members = {}
    for m in ensemble["members"]:
        data = np.load(member_dir(args) / f"{es.member_name(m['fold'], m['seed'])}.npz")
        if str(data["bundle_sha256"]) != m["sha256"]:
            raise SystemExit(f"{m['fold']}/{m['seed']}: scored with another bundle than ensemble.json lists")
        members[m["fold"], m["seed"]] = {"xg": data["xg"], "on_target": data["on_target"]}
    return members, {(m["fold"], m["seed"]): m["temperature"] for m in ensemble["members"]}


def check_gate(args, shots: pd.DataFrame, scores: pd.DataFrame, members: dict, temperatures: dict, calibration: dict) -> dict:
    """The gate shots, scored here, against the stored gate predictions of the fold models (same rows, same batch order):
    per-model logits and the ensemble probability computed from those stored logits with the same calibration."""
    name = json.loads((args.g_root / "fold0/seed0/complete.json").read_text())["identity"]["name"]
    gate = shots[shots.role.eq(es.GATE)]
    positions = gate.index.to_numpy()
    stored = {}
    for fold, seed in es.MEMBERS:
        csv = pd.read_csv(args.g_root / f"fold{fold}" / f"seed{seed}" / f"{name}_gate_predictions.csv").drop_duplicates("shot_id")
        stored[fold, seed] = csv.set_index("shot_id").logit.reindex(gate.key).to_numpy()
    if any(np.isnan(v).any() for v in stored.values()):
        raise SystemExit("a gate shot has no stored gate prediction")
    dlogit = max(float(np.nanmax(np.abs(members[k]["xg"][0, positions] - stored[k]))) for k in es.MEMBERS)
    reference = ensemble_probability([stored[k] for k in sorted(es.MEMBERS)], [temperatures[k] for k in sorted(es.MEMBERS)], calibration,
                                     gate.is_penalty.to_numpy(bool))
    mine = scores.p_0.to_numpy()[positions]
    if np.isnan(mine).any():
        raise SystemExit("a gate shot has no offset-0 score")
    result = {"gate_shots": len(gate), "max_abs_dlogit_vs_stored": dlogit, "max_abs_dp_ensemble_vs_stored": float(np.abs(mine - reference).max()),
              "tolerance_dp": GATE_TOL}
    result["passed"] = bool(result["max_abs_dp_ensemble_vs_stored"] <= GATE_TOL)
    return result


def wait_for_comparator(root: Path, minutes: float) -> dict:
    """The comparator's completion record once it exists and verifies (waits up to `minutes`)."""
    deadline = time.time() + minutes * 60
    while not (root / "complete.json").exists():
        if time.time() > deadline:
            raise SystemExit(f"{root / 'complete.json'} does not exist (waited {minutes:g} min)")
        print("waiting for the frozen comparator ...", flush=True)
        time.sleep(120)
    time.sleep(5)   # the folder is swapped in as a whole; let a rename in progress finish
    record = json.loads((root / "complete.json").read_text())
    present = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()} - {"complete.json"}
    if present != set(record["files"]):
        raise SystemExit(f"{root} holds files that differ from its completion record")
    for name, expected in record["files"].items():
        if file_sha256(root / name) != expected:
            raise SystemExit(f"{root / name} differs from the comparator's completion record")
    return record


def embedding_sets(args, n_rows: dict) -> dict:
    sets = {}
    for column, file, model in (("emb_g_fold0", "emb_g_fold0.npy", "retrained G-combo, fold 0 seed 0 (pre-specified), offset 0"),
                                ("emb_g_fold1", "emb_g_fold1.npy", "retrained G-combo, fold 1 seed 0 (sensitivity), offset 0"),
                                ("emb_dinov2", "emb_dinov2.npy", "frozen DINOv2-L, CLS + mean patch token, offset 0")):
        array = np.load(args.out / file, mmap_mode="r")
        if array.shape[0] != n_rows["offset0"] or array.dtype != np.float16:
            raise SystemExit(f"{file}: {array.shape} {array.dtype}, expected {n_rows['offset0']} float16 rows")
        sets[column] = {"file": file, "sha256": sha256_file(args.out / file), "dim": int(array.shape[1]), "rows": int(array.shape[0]),
                        "model": model + ", float16, rows of the shots with a scored offset-0 frame in table order"}
    return sets


def stage_build(args) -> None:
    if (args.out / "complete.json").exists():
        raise SystemExit(f"{args.out} is complete; delete it to rebuild")
    shots = pd.read_parquet(args.out / "shots.parquet")
    ensemble = read_ensemble(args.g_root)
    replay = json.loads((args.out / "replay.json").read_text())
    if not replay["passed"]:
        raise SystemExit("the replay did not pass")
    members, temperatures = load_members(args, ensemble)
    calibration = json.loads((args.g_root / "ensemble.json").read_text())["calibration"]
    scores = es.assemble_scores(shots, members, temperatures, calibration)
    gate_check = check_gate(args, shots, scores, members, temperatures, calibration)
    print("gate check", gate_check, flush=True)
    if not gate_check["passed"]:
        raise SystemExit("the gate shots do not reproduce the stored gate predictions within tolerance")

    pairs, pairing = read_pairing(args.pairing)
    if not es.frames_unchanged(shots):
        raise SystemExit("input frames changed since scoring")
    comparator_record = wait_for_comparator(args.comparator, args.wait_minutes)
    if comparator_record["identity"]["inputs"]["pairing"]["complete_json_sha256"] != pairing["complete_json_sha256"]:
        raise SystemExit("the comparator was built on another pairing")
    comparator = pd.read_parquet(args.comparator / "predictions.parquet")
    table = es.build_eval_table(shots, scores, pairs, comparator)
    n_rows = {"offset0": int(es.scored_mask(shots[shots.role.eq(es.PRIMARY)], "0").sum())}
    sets = embedding_sets(args, n_rows)
    fingerprint = es.shots_fingerprint(shots)
    sources = {"pairs.parquet": pairing["pairs_sha256"], "pairing_complete.json": pairing["complete_json_sha256"],
               "comparator_complete.json": file_sha256(args.comparator / "complete.json"),
               "comparator_predictions.parquet": comparator_record["files"]["predictions.parquet"],
               "ensemble.json": sha256_file(args.g_root / "ensemble.json"), "split_plan": split_check.PLAN_SHA256,
               "frames": fingerprint["sha256"], "index": sha256_file(args.index)}
    notes = ("confirmatory table, labels absent; primary population = the 2,754 paired shots of the frozen pairing, penalties included "
             "(is_penalty is a label column and is joined later; the frame logits of index-flagged penalties are the routed penalty rate)",
             "frame logits: all ten retrained fold models, temperature-scaled logits averaged, the single ensemble calibration of "
             "results/xgv_g_combo/ensemble.json; frames are index v1.2 frame_path and frame_path_-400",
             "vendor xG is StatsBomb's xG unclipped; comparator rungs are the frozen predictions of results/xgv_comparator")
    meta = et.write_table(table, args.out, TABLE_NAME, kind="confirmatory", embedding_sets=sets, notes=notes, sources=sources)
    loaded, loaded_meta = et.load_table(args.out, TABLE_NAME)   # reads and validates
    scores.to_parquet(args.out / "scores.parquet", index=False, compression="zstd")

    files = {str(p.relative_to(args.out)): sha256_file(p) for p in sorted(args.out.rglob("*")) if p.is_file() and p.name != "complete.json"}
    bundles = {es.member_name(m["fold"], m["seed"]): m["sha256"] for m in ensemble["members"]}
    record = {"completed_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
              "identity": {"version": VERSION, "table_kind": "confirmatory", "schema_version": et.SCHEMA_VERSION, "bundles": bundles,
                           "plan_sha256": split_check.PLAN_SHA256, "pairing": pairing, "comparator_complete_json_sha256": sources["comparator_complete.json"],
                           "comparator_predictions_sha256": sources["comparator_predictions.parquet"],
                           "ensemble_json_sha256": sources["ensemble.json"], "frame_fingerprint": fingerprint, "index_sha256": sources["index"],
                           "sources": {str(p.relative_to(ROOT)): sha256_file(p) for p in SOURCE_FILES}, "code": code_version(ROOT)},
              "summary": {"rows": meta["rows"], "by_population_split": meta["by_population_split"], "labels_state": loaded_meta["labels_state"],
                          "label_free_table_sha256": meta["table_sha256"], "frame_status": loaded.frame_status.value_counts().to_dict(),
                          "gate_shots": gate_check["gate_shots"], "gate_check": gate_check,
                          "replay": {k: v for k, v in replay.items() if k != "members"},
                          "embedding_rows": {c: s["rows"] for c, s in sets.items()}},
              "runtime": {"python": platform.python_version(), "pandas": pd.__version__, "pyarrow": pyarrow.__version__, "torch": torch.__version__},
              "files": files}
    (args.out / "complete.json").write_text(json_text(record))
    print(f"wrote {args.out}; label-free table sha256 {meta['table_sha256']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("score", "replay", "build"))
    parser.add_argument("--gpu-approved", action="store_true", help="required by the scoring stage")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--g-root", type=Path, default=G_COMBO)
    parser.add_argument("--pairing", type=Path, default=PAIRING)
    parser.add_argument("--comparator", type=Path, default=COMPARATOR)
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--wait-minutes", type=float, default=60)
    args = parser.parse_args()
    {"score": stage_score, "replay": stage_replay, "build": stage_build}[args.stage](args)


if __name__ == "__main__":
    main()
