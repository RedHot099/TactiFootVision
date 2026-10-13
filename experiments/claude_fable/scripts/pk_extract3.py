"""Run audit 3's declared configurations E1-E3 on the t0 frames of the paired shots (pitch-keypoints 04).

    ./run.sh python scripts/pk_extract3.py [--out cache/xgd_extract3] [--limit N] [--device cuda]

The shots are xg-detector's: the 2,754 SoccerNet shots paired with a StatsBomb shot (`xgd_extract.paired_shot_ids`, the
pairing's `shot_id` and `paired` columns only), read first; everything after it runs inside `no_statsbomb_reads()`. The
index is read for the frame paths only. The configurations must be the committed declaration
(`research/pitchkp_audit3_configs.json`, `extract3.AUDIT_3_DECLARATION_SHA256`), and the pitch model its sha256.

One pass serves all three (`extract3.extract`): t0 is analysed with teams, the -400 / -800 ms frames without teams and
only when a configuration's source order reaches them. Each configuration gets <out>/<config>/frames.parquet (every
frame the pass analysed for its shots), features.parquet (one t0 row per shot) and complete.json, last. A configuration
whose complete.json exists is refused.
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
from xgd_extract import PAIRS, paired_shot_ids
from xgpoc.bundle import canonical_digest, sha256_file
from xgpoc.detector import extract as ex
from xgpoc.detector import extract3 as e3
from xgpoc.detector import records
from xgpoc.detector.guard import no_statsbomb_reads
from xgpoc.paths import INDEX

ROOT = Path(__file__).resolve().parents[1]


def check_declaration() -> dict:
    digest = sha256_file(e3.AUDIT_3_DECLARATION) if e3.AUDIT_3_DECLARATION.is_file() else None
    if e3.AUDIT_3_DECLARATION_SHA256 is None or digest != e3.AUDIT_3_DECLARATION_SHA256:
        raise SystemExit(f"{e3.AUDIT_3_DECLARATION} is not the committed E1-E3 declaration")
    declared = e3.declaration()["configs"]
    if {n: tuple(c["source_ms"]) for n, c in declared.items()} != {n: c.source for n, c in e3.CONFIGS.items()}:
        raise SystemExit("extract3.CONFIGS differ from the declaration")
    return {"path": str(e3.AUDIT_3_DECLARATION), "sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=ROOT / "cache/xgd_extract3")
    parser.add_argument("--configs", nargs="+", default=list(e3.AUDIT_3), choices=list(e3.CONFIGS))
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--pairs", type=Path, default=PAIRS)
    parser.add_argument("--limit", type=int, help="first N shots only (smoke runs)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    configs = [e3.CONFIGS[name] for name in args.configs]
    for config in configs:
        if (args.out / config.name / "complete.json").exists():
            raise SystemExit(f"{args.out / config.name} is complete; use a new --out")
    declaration = check_declaration()
    if sha256_file(e3.KEYPOINT_WEIGHTS) != e3.KEYPOINT_SHA256:
        raise SystemExit(f"{e3.KEYPOINT_WEIGHTS} is not the declared pitch model")

    shot_ids = paired_shot_ids(args.pairs)[: args.limit]
    shots_input = {"source": str(args.pairs), "source_sha256": sha256_file(args.pairs), "rule": "pairs.paired",
                   "count": len(shot_ids), "digest": canonical_digest(shot_ids), "limit": args.limit}

    with no_statsbomb_reads():
        import tactifoot_vision as tv

        shots = ex.frame_table(args.index, shot_ids)
        identity = {"models": records.model_hashes(ex.DETECTOR_WEIGHTS, e3.KEYPOINT_WEIGHTS, ex.SIGLIP),
                    "code": records.code_hashes(Path(__file__)), "git": records.git_state(),
                    "libraries": records.library_versions(), "declaration": declaration,
                    "inputs": {"index": {"path": str(args.index), "sha256": sha256_file(args.index),
                                         "columns": ["shot_id", *ex.FRAME_COLUMNS.values()]}, "shots": shots_input}}
        embedder = tv.teams.SigLIPEmbedder(model_name=ex.SIGLIP, device=args.device)
        analyzer, ball_only = e3.build_analyzers(embedder, device=args.device)
        started, clock = datetime.now(UTC), time.perf_counter()
        out = e3.extract(shots, configs, analyzer, ball_only, cv2.imread,
                         progress=lambda rows: tqdm(rows, desc="+".join(args.configs), mininterval=30))
        seconds, finished = time.perf_counter() - clock, datetime.now(UTC)
        frames = ex.frames_table(out.frames)
        for config in configs:
            folder = args.out / config.name
            folder.mkdir(parents=True, exist_ok=True)
            frames.to_parquet(folder / "frames.parquet", index=False)
            features = pd.DataFrame(out.features[config.name])
            features.to_parquet(folder / "features.parquet", index=False)
            summary = ex.coverage(features) | {"frames_analysed": len(frames),
                                               "earlier_frames": int((frames.offset_ms != 0).sum())}
            record = {"config": e3.settings(config), "identity": identity,
                      "run": {"pass": "+".join(args.configs), "started": started.isoformat(timespec="seconds"),
                              "finished": finished.isoformat(timespec="seconds"), "pass_seconds": round(seconds, 1),
                              "pass_ms_per_shot": round(1000 * seconds / max(len(shots), 1), 1), "device": args.device},
                      "coverage": summary, "files": records.output_hashes(folder)}
            (folder / "complete.json").write_text(ex.json_text(record))
            print(config.name, summary, flush=True)
    print(f"done: {len(shot_ids)} shots, {seconds / 60:.1f} min, {1000 * seconds / max(len(shots), 1):.0f} ms per shot")


if __name__ == "__main__":
    main()
