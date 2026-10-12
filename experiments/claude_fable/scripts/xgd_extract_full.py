"""Run the chosen extractor configuration on every index shot at t0 and -400 ms (xg-detector tickets 02 and 09, story 11).

    audit 2 (ticket 09, the defaults; the features approach 1 uses whatever the gate says):
    ./run.sh python scripts/xgd_extract_full.py [--out cache/xgd_full2] [--report results/xgd_full2] [--limit N] [--device cuda]

    audit 1 (ticket 02): --choice results/xgd_audit/choice.json --out cache/xgd_full --report results/xgd_full \
        --reference cache/xgd_extract

Which configuration: `chosen` in the audit's `choice.json` (default audit 2's), read first, here and nowhere else; it
holds a name, no StatsBomb value. Everything after it (models, frames, features, hashes, coverage) runs inside
`no_statsbomb_reads()`. The index is read for the frame paths and the game (`ex.frame_table`, then shot_id and game
for the roles): no label column. The pairing is never read.

Each shot's t0 and -400 ms frames are analysed with teams, each giving a feature row (`offset_ms`); with the chosen
configuration's ball fallback, t0 takes the ball of -400 and then of -800 ms, -400 that of -800 ms. -800 ms is analysed
only for that fallback, without teams. No frame after t0 is read (`ex.extract_study`).

Writes <out>/<config>/frames.parquet, features.parquet and complete.json (last; a complete folder is refused), and
<report>/coverage_by_role.csv with its own complete.json: coverage by split role (xg-vendor split plan: training,
gate, evaluation) and offset. The t0 rows of the shots in the audit cache (`--reference`/<config>) are compared
with it; the record keeps the number of shots whose features differ, per column.
"""

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")   # SigLIP comes from the local Hugging Face cache, never downloaded

import cv2
import pandas as pd
from tqdm import tqdm
from xgpoc.bundle import canonical_digest, sha256_file
from xgpoc.detector import extract as ex
from xgpoc.detector import records
from xgpoc.detector.features import COLUMNS
from xgpoc.detector.guard import no_statsbomb_reads
from xgpoc.paths import INDEX

ROOT = Path(__file__).resolve().parents[1]
CHOICE = ROOT / "results/xgd_audit2/choice.json"
PLAN = ROOT / "results/xgv_split_plan/plan.csv"


def chosen_config(choice: Path) -> str:
    """The configuration the audit chose; the only read of the audit's outputs in the detector's model path."""
    return str(json.loads(choice.read_text())["chosen"])


def reference_differences(features: pd.DataFrame, reference: Path) -> dict | None:
    """Shots in both runs whose t0 features differ between the audit cache and the full run, per column (None
    without an audit cache)."""
    if not reference.is_file():
        return None
    ref = pd.read_parquet(reference).set_index("shot_id").sort_index()
    new = features[features.offset_ms == 0].set_index("shot_id")
    common = ref.index.intersection(new.index)
    ref, new = ref.loc[common], new.loc[common]
    differs = ex.feature_differences(ref, new, COLUMNS, 1e-6)
    return {"shots": len(ref), "shots_differing": int(differs.any(axis=1).sum()),
            "columns": {str(c): int(n) for c, n in differs.sum().items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=ROOT / "cache/xgd_full2")
    parser.add_argument("--report", type=Path, default=ROOT / "results/xgd_full2")
    parser.add_argument("--choice", type=Path, default=CHOICE)
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--reference", type=Path, default=ROOT / "cache/xgd_extract2")
    parser.add_argument("--limit", type=int, help="first N shots only (smoke runs)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    name = chosen_config(args.choice)
    choice_input = {"path": str(args.choice), "sha256": sha256_file(args.choice), "chosen": name}
    config = ex.CONFIGS[name]
    folder = args.out / name
    for done in (folder / "complete.json", args.report / "complete.json"):
        if done.exists():
            raise SystemExit(f"{done} exists; use a new --out / --report")

    with no_statsbomb_reads():
        import tactifoot_vision as tv

        shot_games = pd.read_parquet(args.index, columns=["shot_id", "game"]).sort_values("shot_id", kind="stable")
        shot_ids = shot_games.shot_id.tolist()[: args.limit]
        shot_games = shot_games[shot_games.shot_id.isin(shot_ids)]
        shots = ex.frame_table(args.index, shot_ids)
        identity = {"models": records.model_hashes(ex.DETECTOR_WEIGHTS, ex.KEYPOINT_WEIGHTS, ex.SIGLIP),
                    "code": records.code_hashes(Path(__file__)), "git": records.git_state(),
                    "libraries": records.library_versions(),
                    "inputs": {"index": {"path": str(args.index), "sha256": sha256_file(args.index),
                                         "columns": ["shot_id", "game", *ex.FRAME_COLUMNS.values()]},
                               "shots": {"rule": "every index shot", "count": len(shot_ids),
                                         "digest": canonical_digest(shot_ids), "limit": args.limit},
                               "choice": choice_input}}
        embedder = tv.teams.SigLIPEmbedder(model_name=ex.SIGLIP, device=args.device)
        analyzer, ball_only = ex.build_analyzers(config, embedder, device=args.device)
        started, clock = datetime.now(UTC), time.perf_counter()
        result = ex.extract_study(shots, config, analyzer, ball_only, cv2.imread,
                                  progress=lambda rows: tqdm(rows, desc=name, mininterval=60))
        seconds = time.perf_counter() - clock
        finished = datetime.now(UTC)

        folder.mkdir(parents=True, exist_ok=True)
        ex.frames_table(result.frames).to_parquet(folder / "frames.parquet", index=False)
        features = pd.DataFrame(result.features)
        features.to_parquet(folder / "features.parquet", index=False)
        summary = {str(offset): ex.coverage(rows) for offset, rows in features.groupby("offset_ms", sort=False)}
        summary["frames_analysed"] = len(result.frames)
        summary["frames_by_offset"] = {str(k): int(v) for k, v in pd.Series([r["offset_ms"] for r in result.frames]).value_counts().items()}
        reference = reference_differences(features, args.reference / name / "features.parquet")
        record = {"config": ex.settings(config), "identity": identity,
                  "run": {"offsets_ms": list(ex.STUDY_OFFSETS), "fallback_offsets_ms": list(ex.FALLBACK_OFFSETS),
                          "started": started.isoformat(timespec="seconds"), "finished": finished.isoformat(timespec="seconds"),
                          "seconds": round(seconds, 1), "ms_per_shot": round(1000 * seconds / max(len(shots), 1), 1),
                          "device": args.device},
                  "coverage": summary, "audit_cache_t0_differences": reference, "files": records.output_hashes(folder)}
        (folder / "complete.json").write_text(ex.json_text(record))

        plan = pd.read_csv(args.plan, usecols=["game", "role"])
        table = ex.coverage_by_role(features, shot_games, plan)
        args.report.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.report / "coverage_by_role.csv", index=False, float_format="%.4f")
        report_record = {"config": name, "cache": {"path": str(folder), "complete_json_sha256": sha256_file(folder / "complete.json")},
                         "plan": {"path": str(args.plan), "sha256": sha256_file(args.plan), "columns": ["game", "role"]},
                         "code": identity["code"], "git": identity["git"], "files": records.output_hashes(args.report)}
        (args.report / "complete.json").write_text(ex.json_text(report_record))
    print(table.to_string(index=False))
    print(f"done: {len(shot_ids)} shots, {name}, {seconds / 60:.1f} min ({1000 * seconds / max(len(shots), 1):.0f} ms per shot); "
          f"audit-cache t0 differences: {reference and reference['shots_differing']}")


if __name__ == "__main__":
    main()
