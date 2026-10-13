"""The pitch keypoint model against its own training labels (xg-detector 08): vertex definitions and keypoint input.

    ./run.sh python scripts/xgd_kp_labels.py [--data /home/kuba/projects/tactifoot_vision/data/keypoints]
                                             [--out results/xgd_homography_dev/labels] [--device cpu]

The labels are the Roboflow "football-field-detection-f07vi" v15 export the model was trained on (train, valid and
test; 640 x 640 images stretched from 16:9 frames, keypoints in YOLO pose format with visibility 0 or 2). No SoccerNet
frame, StatsBomb data or goal label is read.

1. **Vertex definitions** (`vertex_check.csv`). For every labelled image and every visible keypoint, a homography is
   fitted on the image's other visible keypoints (at least 5, all-points least squares, kept when they fit the package
   pitch within 1 m) and the held-out keypoint is mapped to the pitch. Per vertex: the median pitch position of the
   label against the package vertex (`SoccerPitch().vertices`) and against the research extractor's old table
   (`xgpoc.features.pitch`), which put 10/11/18/19 at the goal-box width. Keypoint order and left/right mirroring
   show here as metres-large offsets; the far touchline is checked to be y = 0 (vertex 0 above vertex 5 in the image).
2. **Keypoint input** (`keypoint_error.csv`). The model on each labelled image, as trained (the 640 x 640 image) and as
   the letterbox sees a frame (the image resized back to 1280 x 720 and letterboxed by Ultralytics), through the
   package (`tv.load_model("yolo_pose", ..., resize=...)`). Per image: the median distance between predicted and
   labelled visible keypoints, in 1280 x 720 pixels, and the share of visible keypoints predicted with confidence
   >= 0.5.

Writes vertex_check.csv, keypoint_error.csv, summary.json and complete.json, last.
"""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from xgpoc.bundle import sha256_file
from xgpoc.detector import extract as ex
from xgpoc.detector import records
from xgpoc.features.pitch import pitch_vertices

from tactifoot_vision.pitch import SoccerPitch

ROOT = Path(__file__).resolve().parents[1]
DATA = ex.MAIN_CHECKOUT / "data/keypoints"
SPLITS = ("train", "valid", "test")
FRAME = (1280, 720)                       # SoccerNet frames of the study; the labels' 16:9 source
FIT_LIMIT_M = 1.0                         # a leave-one-out fit counts when the other keypoints fit within 1 m
MIN_OTHERS = 5


def read_labels(data: Path) -> pd.DataFrame:
    """One row per labelled image: split, image path and the (32, 3) keypoints (x, y in 640-px units, visibility)."""
    rows = []
    for split in SPLITS:
        for image in sorted((data / split / "images").glob("*.jpg")):
            label = data / split / "labels" / f"{image.stem}.txt"
            values = np.array(label.read_text().split()[:5 + 96], dtype=float)
            keypoints = values[5:].reshape(32, 3) * [640, 640, 1]
            rows.append({"split": split, "image": str(image), "keypoints": keypoints})
    return pd.DataFrame(rows)


def held_out_positions(labels: pd.DataFrame, vertices: np.ndarray) -> pd.DataFrame:
    """Pitch position of every visible labelled keypoint under a homography fitted on the image's other keypoints."""
    rows = []
    for split, image, keypoints in labels[["split", "image", "keypoints"]].itertuples(index=False):
        visible = np.flatnonzero(keypoints[:, 2] > 0)
        for k in visible:
            others = visible[visible != k]
            if len(others) < MIN_OTHERS:
                continue
            matrix, _ = cv2.findHomography(keypoints[others, :2], vertices[others], 0)
            if matrix is None:
                continue
            fitted = cv2.perspectiveTransform(keypoints[others, None, :2], matrix).reshape(-1, 2)
            if np.linalg.norm(fitted - vertices[others], axis=1).max() > FIT_LIMIT_M:
                continue
            x, y = cv2.perspectiveTransform(keypoints[None, [k], :2], matrix).reshape(2)
            rows.append({"split": split, "image": image, "vertex": int(k), "x": x, "y": y})
    return pd.DataFrame(rows)


def vertex_check(held_out: pd.DataFrame, package: np.ndarray, old: np.ndarray) -> pd.DataFrame:
    """Per vertex: count and median labelled position, offsets from the package vertex and from the old table."""
    table = held_out.groupby("vertex").agg(n=("x", "size"), label_x=("x", "median"), label_y=("y", "median"))
    table["package_x"], table["package_y"] = package[table.index, 0], package[table.index, 1]
    table["old_x"], table["old_y"] = old[table.index, 0], old[table.index, 1]
    for name in ("package", "old"):
        table[f"{name}_offset_m"] = np.hypot(table.label_x - table[f"{name}_x"], table.label_y - table[f"{name}_y"])
    return table.reset_index()


def keypoint_errors(labels: pd.DataFrame, weights: Path, device: str) -> pd.DataFrame:
    """Per image and input: median distance (1280 x 720 px) of predicted to labelled visible keypoints, and recall."""
    import tactifoot_vision as tv

    models = {"stretch": tv.load_model("yolo_pose", weights, conf=0.1, resize="stretch", device=device),
              "letterbox": tv.load_model("yolo_pose", weights, conf=0.1, device=device)}
    to_frame = np.array([FRAME[0] / 640, FRAME[1] / 640])
    rows = []
    for split, path, keypoints in labels[["split", "image", "keypoints"]].itertuples(index=False):
        frame = cv2.resize(cv2.imread(path), FRAME)       # the 16:9 frame the labelled image was stretched from
        truth = keypoints[:, :2] * to_frame
        visible = keypoints[:, 2] > 0
        for name, model in models.items():
            predicted = model(frame)
            row = {"split": split, "image": path, "input": name, "detected": len(predicted) > 0}
            if len(predicted):
                error = np.linalg.norm(predicted.xy[0] - truth, axis=1)[visible]
                confident = predicted.confidence[0][visible] >= 0.5
                row |= {"median_px": float(np.median(error)), "confident_share": float(confident.mean()),
                        "confident_median_px": float(np.median(error[confident])) if confident.any() else np.nan}
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=ROOT / "results/xgd_homography_dev/labels")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if (args.out / "complete.json").exists():
        raise SystemExit(f"{args.out} is complete; use a new --out")
    started, clock = datetime.now(UTC), time.perf_counter()
    labels = read_labels(args.data)
    package, old = SoccerPitch().vertices.astype(float), pitch_vertices(105.0, 68.0).astype(float)
    held_out = held_out_positions(labels, package)
    vertices = vertex_check(held_out, package, old)
    errors = keypoint_errors(labels, ex.KEYPOINT_WEIGHTS, args.device)

    corners = np.stack(labels.keypoints.to_numpy())[:, [0, 5]]          # (images, 2, 3): vertices 0 and 5
    both = (corners[:, 0, 2] > 0) & (corners[:, 1, 2] > 0)
    by_input = errors[errors.detected].groupby("input")
    summary = {
        "images": int(len(labels)), "held_out_keypoints": int(len(held_out)),
        "vertex_offset_m": {"package_median": float(vertices.package_offset_m.median()),
                            "package_max": float(vertices.package_offset_m.max()),
                            "package_max_vertex": int(vertices.vertex[vertices.package_offset_m.idxmax()]),
                            "old_10_11_18_19": [float(v) for v in vertices.set_index("vertex").old_offset_m[[10, 11, 18, 19]]],
                            "package_10_11_18_19": [float(v) for v in vertices.set_index("vertex").package_offset_m[[10, 11, 18, 19]]]},
        "far_touchline_on_top": {"images_with_0_and_5": int(both.sum()),
                                 "vertex_0_above_5": int((corners[both, 0, 1] < corners[both, 1, 1]).sum())},
        "keypoint_error_px_1280x720": {name: {"images": int(len(g)), "median_of_image_medians": float(g.median_px.median()),
                                              "mean_of_image_medians": float(g.median_px.mean()),
                                              "confident_share_median": float(g.confident_share.median())}
                                       for name, g in by_input},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    vertices.to_csv(args.out / "vertex_check.csv", index=False, float_format="%.4f")
    errors.to_csv(args.out / "keypoint_error.csv", index=False, float_format="%.4f")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    record = {"purpose": "xg-detector 08: the keypoint model against its own training labels (vertex definitions, "
                         "stretched vs letterboxed input)",
              "inputs": {"labels": str(args.data), "keypoint_weights": {"path": str(ex.KEYPOINT_WEIGHTS),
                                                                       "sha256": sha256_file(ex.KEYPOINT_WEIGHTS)}},
              "parameters": {"frame": FRAME, "fit_limit_m": FIT_LIMIT_M, "min_others": MIN_OTHERS, "device": args.device},
              "summary": summary, "code": records.code_hashes(Path(__file__)), "git": records.git_state(),
              "run": {"started": started.isoformat(timespec="seconds"), "seconds": round(time.perf_counter() - clock, 1)},
              "files": records.output_hashes(args.out)}
    (args.out / "complete.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
