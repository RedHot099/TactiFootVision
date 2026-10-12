"""Run the four declared extractor configurations on the t0 frames of the paired shots (xg-detector ticket 01).

    ./run.sh python scripts/xgd_extract.py [--out cache/xgd_extract] [--configs C1 C2 C3 C4] [--limit N] [--device cuda]

Which shots: the 2,754 SoccerNet shots paired with a StatsBomb shot (`results/xgv_pairing/pairs.parquet`, columns
`shot_id` and `paired` only), because the audit (ticket 02) compares their detected positions with the freeze frames.
That list is read first, here and nowhere else. Everything after it (models, frames, features, hashes) runs inside
`no_statsbomb_reads()`, so no model or feature input can come from StatsBomb data or the pairing. The index is read
for the frame paths only (`FRAME_COLUMNS`): no label column.

Configurations with the same detector settings share one pass (C1 with C2, C3 with C4): t0 is analysed once and the
-400 / -800 ms frames only for the ball fallback of C2 / C4, when t0 has no ball on the pitch.

Writes <out>/<config>/frames.parquet (one row per analysed frame: detections, keypoints, homography), features.parquet
(one row per shot) and complete.json, last. A configuration whose complete.json exists is refused.
"""

import argparse
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
from xgpoc.detector.guard import no_statsbomb_reads
from xgpoc.paths import INDEX

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ROOT / "results/xgv_pairing/pairs.parquet"


def paired_shot_ids(pairs: Path) -> list[str]:
    """The paired SoccerNet shot ids, sorted; the only read of the pairing in the detector code."""
    table = pd.read_parquet(pairs, columns=["shot_id", "paired"])
    return sorted(table.shot_id[table.paired.astype(bool)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=ROOT / "cache/xgd_extract")
    parser.add_argument("--configs", nargs="+", default=list(ex.AUDIT_1), choices=list(ex.CONFIGS))
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--pairs", type=Path, default=PAIRS)
    parser.add_argument("--limit", type=int, help="first N shots only (smoke runs)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    configs = [ex.CONFIGS[name] for name in args.configs]
    for config in configs:
        if (args.out / config.name / "complete.json").exists():
            raise SystemExit(f"{args.out / config.name} is complete; use a new --out")

    shot_ids = paired_shot_ids(args.pairs)[: args.limit]
    shots_input = {"source": str(args.pairs), "source_sha256": sha256_file(args.pairs), "rule": "pairs.paired",
                   "count": len(shot_ids), "digest": canonical_digest(shot_ids), "limit": args.limit}

    with no_statsbomb_reads():
        import tactifoot_vision as tv

        shots = ex.frame_table(args.index, shot_ids)
        identity = {"models": records.model_hashes(ex.DETECTOR_WEIGHTS, ex.KEYPOINT_WEIGHTS, ex.SIGLIP),
                    "code": records.code_hashes(Path(__file__)), "git": records.git_state(),
                    "libraries": records.library_versions(),
                    "inputs": {"index": {"path": str(args.index), "sha256": sha256_file(args.index),
                                         "columns": ["shot_id", *ex.FRAME_COLUMNS.values()]}, "shots": shots_input}}
        embedder = tv.teams.SigLIPEmbedder(model_name=ex.SIGLIP, device=args.device)
        groups: dict[tuple, list[ex.ExtractorConfig]] = {}
        for config in configs:
            groups.setdefault(config.analysis, []).append(config)
        for group in groups.values():
            names = "+".join(c.name for c in group)
            analyzer, ball_only = ex.build_analyzers(group[0], embedder, device=args.device)
            started, clock = datetime.now(UTC), time.perf_counter()
            results = ex.extract_detector(shots, group, analyzer, ball_only, cv2.imread,
                                          progress=lambda rows, names=names: tqdm(rows, desc=names, mininterval=30))
            seconds = time.perf_counter() - clock
            finished = datetime.now(UTC)
            for config in group:
                result = results[config.name]
                folder = args.out / config.name
                folder.mkdir(parents=True, exist_ok=True)
                ex.frames_table(result.frames).to_parquet(folder / "frames.parquet", index=False)
                features = pd.DataFrame(result.features)
                features.to_parquet(folder / "features.parquet", index=False)
                summary = ex.coverage(features) | {"frames_analysed": len(result.frames),
                                                "fallback_frames": int(sum(r["offset_ms"] != 0 for r in result.frames))}
                record = {"config": ex.settings(config), "identity": identity,
                          "run": {"pass": names, "started": started.isoformat(timespec="seconds"),
                                  "finished": finished.isoformat(timespec="seconds"), "pass_seconds": round(seconds, 1),
                                  "pass_ms_per_shot": round(1000 * seconds / max(len(shots), 1), 1), "device": args.device},
                          "coverage": summary, "files": records.output_hashes(folder)}
                (folder / "complete.json").write_text(ex.json_text(record))
                print(config.name, {k: v for k, v in summary.items()}, flush=True)
            print(f"pass {names}: {seconds / 60:.1f} min, {1000 * seconds / max(len(shots), 1):.0f} ms per shot", flush=True)
            del analyzer, ball_only
    print(f"done: {len(shot_ids)} shots, configurations {' '.join(args.configs)} in {args.out}")


if __name__ == "__main__":
    main()
