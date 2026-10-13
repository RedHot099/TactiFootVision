"""Descriptive splits of audit 3 after the choice (pitch-keypoints 04): where the shot location came from, and the gate
by whether the pitch model saw frames of the match.

    ./run.sh python scripts/pk_audit3_diag.py [--audit results/xgd_audit3] [--out results/xgd_audit3_diag]

Not part of the frozen choice or gate, and no configuration is added: it splits the per-shot errors audit 3 already
computed (`--audit`/per_shot.parquet, checked against its completion record), with the audit's match draws.

    source       per audited configuration, the errors by where the shot location came from: the ball of t0, of the
                 -400 or of the -800 ms frame, or the flagged shooter (`audit.source_masks`, `audit.group_errors`).
    calibration  council member S's condition: the gate's three quantities (shot-location coverage and median error,
                 goalkeeper coverage) per configuration, split by whether Calibration-2023 frames of the shot's match
                 were in the pitch model's training export (`results/pk_dataset/labelled.csv`: source calib, export
                 split train, not excluded; matched to the split plan's game names by league, season, date and teams,
                 `exclusions.game_key`). Groups: `calib_train` (the model trained on frames of the match), `calib_val`
                 (frames only in its validation export), `other` (the match name parses, no Calibration frame).
                 GSR games carry no names, so a match whose GSR frames were in training cannot be resolved and counts
                 as `other`.

Writes diag.csv (split, group, config, quantity, stat, value, lo, hi; quantity `population`, stat `share`: the group's
share of the population), calibration_games.csv (each population match and its group) and complete.json, last. Like
the audit it reads StatsBomb-derived data and is on no model or feature path.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from xgpoc.bundle import sha256_file
from xgpoc.detector import audit, records
from xgpoc.detector import extract3 as e3
from xgpoc.pitchkp import exclusions as X

ROOT = Path(__file__).resolve().parents[1]
LABELLED = ROOT / "results/pk_dataset/labelled.csv"
GROUPS = ("calib_train", "calib_val", "other")


def verified(path: Path) -> str:
    try:
        return records.verified_file(path)
    except ValueError as e:
        raise SystemExit(str(e)) from None


def calibration_groups(games: pd.Series, labelled: pd.DataFrame) -> pd.Series:
    """Per population match name (split-plan format): `calib_train`, `calib_val` or `other` (module docstring)."""
    calib = labelled[(labelled.source == "calib") & labelled.exclusion.isna()]
    keys = {split: {X.game_key(str(g), X.CALIB_GAME) for g in calib.game[calib.export_split == split]} for split in ("train", "val")}
    out = {}
    for game in sorted(set(games)):
        key = X.game_key(game, X.PLAN_GAME)
        out[game] = "calib_train" if key in keys["train"] else "calib_val" if key in keys["val"] else "other"
    return pd.Series(out, name="group").rename_axis("game")


def gate_split(per_shot: pd.DataFrame, groups: dict[str, np.ndarray], config: str, d: audit.Draws) -> list[dict]:
    """The gate's quantities inside each group of one configuration's rows (population order): shot-location coverage
    (share of the group's shots with a location), median shot-location error, goalkeeper coverage; and the group's
    share of the population."""
    t = per_shot[per_shot.config == config].reset_index(drop=True)
    everyone = np.ones(len(t), bool)
    located, gk = t.shot_ok.to_numpy() == 1, t.gk_ok.to_numpy() == 1
    rows = []
    for name, mask in groups.items():
        cells = {("population", "share"): audit.share(mask, everyone, d),
                 ("population", "matches"): (float(t.game[mask].nunique()), np.nan, np.nan),
                 ("shot_location", "coverage"): audit.share(located, mask, d),
                 ("shot_location", "n"): (float((mask & located).sum()), np.nan, np.nan),
                 ("shot_location", "median"): audit.quantiles(t.shot_err.to_numpy(float), mask & located, (0.5,), d)[0],
                 ("gk_any", "coverage"): audit.share(gk, mask, d)}
        rows += [{"split": "calibration", "group": name, "config": config, "quantity": q, "stat": st, "value": v[0],
                  "lo": v[1], "hi": v[2]} for (q, st), v in cells.items()]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--audit", type=Path, default=ROOT / "results/xgd_audit3")
    parser.add_argument("--labelled", type=Path, default=LABELLED)
    parser.add_argument("--out", type=Path, default=ROOT / "results/xgd_audit3_diag")
    args = parser.parse_args()
    if (args.out / "complete.json").exists():
        raise SystemExit(f"{args.out} is complete; use a new --out")
    started = datetime.now(UTC)

    inputs = {"per_shot": {"path": str(args.audit / "per_shot.parquet"), "sha256": verified(args.audit / "per_shot.parquet")},
              "labelled": {"path": str(args.labelled), "sha256": sha256_file(args.labelled),
                           "columns": ["source", "export_split", "game", "exclusion"]}}
    per_shot = pd.read_parquet(args.audit / "per_shot.parquet")
    params = json.loads((args.audit / "complete.json").read_text())["parameters"]
    population = per_shot[per_shot.config == per_shot.config.iloc[0]].reset_index(drop=True)
    draws = audit.draws(population.game.to_numpy(), params["n_boot"], params["seed"])
    labelled = pd.read_csv(args.labelled, usecols=["source", "export_split", "game", "exclusion"])
    by_game = calibration_groups(population.game, labelled)
    group = population.game.map(by_game).to_numpy()
    masks = {g: group == g for g in GROUPS}

    rows = []
    configs = sorted(per_shot.config.unique())
    for name in configs:
        rows += audit.group_errors(per_shot, audit.source_masks(per_shot, name), name, "source", draws)
        rows += gate_split(per_shot, masks, name, draws)

    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out / "diag.csv", index=False, float_format="%.6g")
    by_game.reset_index().to_csv(args.out / "calibration_games.csv", index=False)
    complete = {"inputs": inputs, "audit_complete_json_sha256": sha256_file(args.audit / "complete.json"),
                "parameters": {"n_boot": params["n_boot"], "seed": params["seed"], "sources": audit.SOURCE_GROUPS,
                               "groups": list(GROUPS)},
                "population_shots": len(population),
                "calibration_matches": {g: int((by_game == g).sum()) for g in GROUPS},
                "calibration_shots": {g: int(m.sum()) for g, m in masks.items()},
                "configs": configs, "declaration_sha256": e3.AUDIT_3_DECLARATION_SHA256,
                "code": records.code_hashes(Path(__file__)), "git": records.git_state(),
                "run": {"started": started.isoformat(timespec="seconds"), "finished": datetime.now(UTC).isoformat(timespec="seconds")},
                "files": records.output_hashes(args.out)}
    (args.out / "complete.json").write_text(json.dumps(complete, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"matches": complete["calibration_matches"], "shots": complete["calibration_shots"]}))


if __name__ == "__main__":
    main()
