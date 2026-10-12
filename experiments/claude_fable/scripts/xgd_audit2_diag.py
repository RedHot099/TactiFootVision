"""Descriptive diagnostics of audit 2 after the choice: where the position error comes from (xg-detector 09).

    ./run.sh python scripts/xgd_audit2_diag.py [--audit results/xgd_audit2] [--extract cache/xgd_extract2] \
        [--out results/xgd_audit2_diag]

Not part of the frozen choice or gate, and no configuration is added: it splits the per-shot errors that audit 2
already computed (`--audit`/per_shot.parquet, checked against its completion record), with the audit's match draws.

    refinement   ticket 08 left this comparison to ticket 09. D3 and D4 run the same keypoint fits as D1 and D2 with
                 the line refinement on; a t0 homography is "refined" when the refined configuration's matrix differs
                 from the unrefined one's, "kept" when the refinement fell back to the keypoint fit, "no_fit" without
                 one (`audit.refinement_groups`, which refuses shots whose t0 keypoints differ between the runs). For
                 each group: both configurations' errors on the same shots.
    source       for every audited configuration, the errors by where the shot location came from: the t0 ball, the
                 ball of the -400 or -800 ms frame (used only when t0 has no ball on the pitch), or the flagged shooter.

Writes diag.csv (split, group, config, quantity, stat, value, lo, hi; `share` is the group's share of the population)
and complete.json, last. Like the audit it reads StatsBomb-derived data and is on no model or feature path.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from xgpoc.bundle import sha256_file
from xgpoc.detector import audit, records
from xgpoc.detector import extract as ex

ROOT = Path(__file__).resolve().parents[1]


def verified(path: Path) -> str:
    """sha256 of a file after checking it against its folder's completion record."""
    digest = sha256_file(path)
    if json.loads((path.parent / "complete.json").read_text())["files"].get(path.name) != digest:
        raise SystemExit(f"{path} does not match its completion record")
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--audit", type=Path, default=ROOT / "results/xgd_audit2")
    parser.add_argument("--extract", type=Path, default=ROOT / "cache/xgd_extract2")
    parser.add_argument("--out", type=Path, default=ROOT / "results/xgd_audit2_diag")
    args = parser.parse_args()
    if (args.out / "complete.json").exists():
        raise SystemExit(f"{args.out} is complete; use a new --out")
    started = datetime.now(UTC)

    inputs = {"per_shot": {"path": str(args.audit / "per_shot.parquet"), "sha256": verified(args.audit / "per_shot.parquet")}}
    per_shot = pd.read_parquet(args.audit / "per_shot.parquet")
    record = json.loads((args.audit / "complete.json").read_text())
    params = record["parameters"]
    population = per_shot[per_shot.config == per_shot.config.iloc[0]]
    draws = audit.draws(population.game.to_numpy(), params["n_boot"], params["seed"])

    rows, refined_fits = [], {}
    for refined, plain in audit.REFINEMENT_PAIRS.items():
        frames = {}
        for name in (plain, refined):
            path = args.extract / name / "frames.parquet"
            inputs[f"{name}_frames"] = {"path": str(path), "sha256": verified(path)}
            frames[name] = pd.read_parquet(path, columns=["shot_id", "offset_ms", "kp_xy", "homography"])
        groups = audit.refinement_groups(frames[plain], frames[refined]).reindex(population.shot_id).to_numpy()
        masks = {g: groups == g for g in ("refined", "kept", "no_fit")}
        refined_fits[refined] = {g: int(m.sum()) for g, m in masks.items()}
        for name in (plain, refined):
            rows += audit.group_errors(per_shot, masks, name, f"refinement_{refined}", draws)
    for name in sorted(per_shot.config.unique()):
        rows += audit.group_errors(per_shot, audit.source_masks(per_shot, name), name, "source", draws)

    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out / "diag.csv", index=False, float_format="%.6g")
    complete = {"inputs": inputs, "audit_complete_json_sha256": sha256_file(args.audit / "complete.json"),
                "parameters": {"n_boot": params["n_boot"], "seed": params["seed"], "pairs": audit.REFINEMENT_PAIRS,
                               "sources": audit.SOURCE_GROUPS},
                "population_shots": len(population), "refinement_groups": refined_fits,
                "configs": sorted(per_shot.config.unique()), "declaration_sha256": ex.AUDIT_2_DECLARATION_SHA256,
                "code": records.code_hashes(Path(__file__)), "git": records.git_state(),
                "run": {"started": started.isoformat(timespec="seconds"), "finished": datetime.now(UTC).isoformat(timespec="seconds")},
                "files": records.output_hashes(args.out)}
    (args.out / "complete.json").write_text(json.dumps(complete, indent=2, sort_keys=True) + "\n")
    print(json.dumps(refined_fits))


if __name__ == "__main__":
    main()
