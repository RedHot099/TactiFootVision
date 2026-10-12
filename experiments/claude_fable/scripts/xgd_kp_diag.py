"""Diagnostic after the audit: C4 with the pitch keypoints of a stretched input instead of the letterbox (xg-detector 02).

    ./run.sh python scripts/xgd_kp_diag.py [--config C4] [--out results/xgd_kp_diag] [--device cpu]

Not part of the frozen config choice or gate, and not a fifth configuration: it measures how much of the audited
position error comes from the keypoint input. The keypoint model was trained on frames stretched to 640 x 640 (the
Roboflow export; `xgpoc/features/freeze_frame.py` stretched for this reason), while the package, and so C1-C4, feed it
a letterboxed frame. Here each cached frame of the configuration is resized to 640 x 640, its keypoints are scaled back
to frame pixels, the homography is fitted with C4's settings (`ex.COMMON`: instance 0.1, keypoint 0.5, RANSAC 10 m),
and the cached detection boxes are placed again (people at the bottom centre, the ball at the centre, as the package's
`FrameAnalyzer`). Detections, teams and everything else are the cache's; the features are replayed
(`ex.replay_features`) and compared with StatsBomb by the audit's own functions, with the audit's draws.

A fallback frame that the letterbox run never analysed (t0 had a ball there) has no detections in the cache. When the
stretched homography loses t0's ball, such a frame counts as one without a ball; the record gives the number of shots.

Writes per_shot.parquet, metrics.csv (configuration name `<config>s`), keypoints.parquet (the new keypoints and
homographies per frame) and complete.json, last. Like the audit, this script reads StatsBomb data and is not on any
model or feature path.
"""

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from tqdm import tqdm
from xgpoc.bundle import sha256_file
from xgpoc.detector import audit, records
from xgpoc.detector import extract as ex
from xgpoc.paths import STATSBOMB

from tactifoot_vision.pitch import HomographyEstimator, frame_to_pitch

ROOT = Path(__file__).resolve().parents[1]
PAIRS = ROOT / "results/xgv_pairing/pairs.parquet"
SHOTS = STATSBOMB / "tables/shots.parquet"
SIZE = 640


def stretched_keypoints(model, image: np.ndarray) -> sv.KeyPoints:
    """The keypoint model on the frame resized to SIZE x SIZE, coordinates scaled back to frame pixels."""
    h, w = image.shape[:2]
    keypoints = model.predict(cv2.resize(image, (SIZE, SIZE)))
    if len(keypoints):
        keypoints.xy[..., 0] *= w / SIZE
        keypoints.xy[..., 1] *= h / SIZE
    return keypoints


def anchors(xyxy, bottom: bool) -> np.ndarray:
    boxes = np.asarray(xyxy, dtype=float).reshape(-1, 4)
    y = boxes[:, 3] if bottom else (boxes[:, 1] + boxes[:, 3]) / 2
    return np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2, y])


def placed(xyxy, matrix, bottom: bool) -> list[float]:
    points = anchors(xyxy, bottom)
    if matrix is None:
        return [float("nan")] * (2 * len(points))
    return [float(v) for v in frame_to_pitch(points, matrix).ravel()]


def restretched_row(row: dict, keypoints: sv.KeyPoints, matrix) -> dict:
    """The cache row with the new keypoints, homography and pitch positions; detections and teams unchanged."""
    has = len(keypoints) > 0
    return row | {"homography": [float(v) for v in np.asarray(matrix).ravel()] if matrix is not None else None,
                  "kp_xy": [float(v) for v in keypoints.xy[0].ravel()] if has else None,
                  "kp_conf": [float(v) for v in keypoints.confidence[0]] if has and keypoints.confidence is not None else None,
                  "kp_instance_conf": float(keypoints.data["box_confidence"][0]) if has else np.nan,
                  "det_pitch_xy": placed(row["det_xyxy"], matrix, bottom=True),
                  "ball_pitch_xy": placed(row["ball_xyxy"], matrix, bottom=False)}


def with_unanalysed_fallbacks(frames: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Frames plus an unreadable placeholder for every fallback offset a shot lacks, and the number of shots padded."""
    have = set(zip(frames.shot_id, frames.offset_ms, strict=True))
    pads = [{"shot_id": s, "offset_ms": o, "frame_path": None, "readable": False}
            for s in frames.shot_id.unique() for o in ex.FALLBACK_OFFSETS if (s, o) not in have]
    return pd.concat([frames, pd.DataFrame(pads)], ignore_index=True), len({p["shot_id"] for p in pads})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="C4", choices=list(ex.CONFIGS))
    parser.add_argument("--extract", type=Path, default=ROOT / "cache/xgd_extract")
    parser.add_argument("--out", type=Path, default=ROOT / "results/xgd_kp_diag")
    parser.add_argument("--pairs", type=Path, default=PAIRS)
    parser.add_argument("--shots", type=Path, default=SHOTS)
    parser.add_argument("--n-boot", type=int, default=audit.N_BOOT)
    parser.add_argument("--seed", type=int, default=audit.SEED)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if (args.out / "complete.json").exists():
        raise SystemExit(f"{args.out} is complete; use a new --out")
    import tactifoot_vision as tv

    config = ex.CONFIGS[args.config]
    folder = args.extract / config.name
    frames = pd.read_parquet(folder / "frames.parquet", columns=audit.FRAME_READ_COLUMNS)
    model = tv.load_model("yolo_pose", ex.KEYPOINT_WEIGHTS, conf=ex.COMMON["keypoint_conf"], device=args.device)
    estimator = HomographyEstimator(min_confidence=ex.COMMON["homography_min_confidence"],
                                    ransac_threshold=ex.COMMON["homography_ransac_threshold"])
    started, clock = datetime.now(UTC), time.perf_counter()
    rows = []
    for row in tqdm(frames.to_dict("records"), desc="keypoints", mininterval=30):
        if not row["readable"]:
            rows.append(row)
            continue
        keypoints = stretched_keypoints(model, cv2.imread(row["frame_path"]))
        rows.append(restretched_row(row, keypoints, estimator.fit(keypoints)))
    seconds = time.perf_counter() - clock
    new_frames, padded = with_unanalysed_fallbacks(pd.DataFrame(rows))

    pop = audit.read_population(args.pairs)
    sb = audit.truth(pop, audit.read_freeze_frames(args.shots, pop.sb_event_id))
    name = f"{config.name}s"
    replayed = ex.replay_features(new_frames, config)
    per_shot = audit.compare(pop, sb, audit.detected(replayed, new_frames), name)
    metrics = audit.metrics(per_shot, audit.draws(pop.game.to_numpy(), args.n_boot, args.seed))

    args.out.mkdir(parents=True, exist_ok=True)
    per_shot.to_parquet(args.out / "per_shot.parquet", index=False)
    metrics.to_csv(args.out / "metrics.csv", index=False, float_format="%.6g")
    new_frames[["shot_id", "offset_ms", "readable", "homography", "kp_xy", "kp_conf", "kp_instance_conf"]].to_parquet(
        args.out / "keypoints.parquet", index=False)

    def value(quantity, stat):
        rows = metrics[(metrics.quantity == quantity) & (metrics.subset == "all") & (metrics.stat == stat)]
        return float(rows.value.iloc[0])
    figures = {"shot_location_coverage": value("shot_location", "coverage"), "shot_location_median_m": value("shot_location", "median"),
               "gk_information_coverage": value("gk_any", "coverage"), "homography_t0": float(replayed.homography_ok.mean())}
    record = {"purpose": "diagnostic: keypoint input stretched vs letterbox; not the config choice or the gate",
              "config": ex.settings(config) | {"keypoint_input": f"stretched {SIZE}x{SIZE}"},
              "inputs": {"cache": {"path": str(folder), "complete_json_sha256": sha256_file(folder / "complete.json")},
                         "pairs": {"path": str(args.pairs), "sha256": sha256_file(args.pairs), "columns": audit.PAIR_COLUMNS},
                         "statsbomb_shots": {"path": str(args.shots), "sha256": sha256_file(args.shots), "columns": audit.SHOT_COLUMNS},
                         "keypoint_weights": {"path": str(ex.KEYPOINT_WEIGHTS), "sha256": sha256_file(ex.KEYPOINT_WEIGHTS)}},
              "parameters": {"n_boot": args.n_boot, "seed": args.seed, "device": args.device},
              "frames": len(frames), "shots_with_unanalysed_fallback_frames": padded, "figures": figures,
              "code": records.code_hashes(Path(__file__)), "git": records.git_state(),
              "run": {"started": started.isoformat(timespec="seconds"), "keypoint_seconds": round(seconds, 1),
                      "finished": datetime.now(UTC).isoformat(timespec="seconds")},
              "files": records.output_hashes(args.out)}
    (args.out / "complete.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(name, figures, f"padded fallback shots {padded}")


if __name__ == "__main__":
    main()
