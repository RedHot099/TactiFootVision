"""Label-free development loop of the homography fix on training-match shot frames (xg-detector 08).

    ./run.sh python scripts/xgd_homography_dev.py [--n 1500] [--seed 0] [--device cuda]
                                                  [--cache cache/xgd_homography_dev] [--out results/xgd_homography_dev]

Frames: `--n` random shots of the split plan's training matches (`xgpoc.detector.homography_dev`), t0 frame. The gate
and evaluation matches are never read. Folds 0-2 are `dev`, folds 3-4 `confirm`.

1. Keypoints (GPU, cached in `<cache>/keypoints.parquet` with its completion record; reused when complete): the
   package's pitch model on every frame with the letterbox input and with `stretch=True`, instance threshold 0.1.
2. Variants (CPU): `VARIANTS` below through `HomographyEstimator.fit(keypoints, frame, boxes)`; `C4` is the homography
   of ticket 02's full C4 run (`cache/xgd_full/C4`, before this ticket). The people boxes come from that run's t0
   detections (the C4 detector), for the line mask and the checks without lines.
3. Scores (`xgpoc.detector.lines`): line agreement and recall at 5 px, players on the pitch, the height ratio.
   `summary.csv` per variant and subset (dev, confirm, all), `per_frame.parquet`, and overlays of the projected
   markings on `--overlays` frames (`overlays/`, one image per frame with every variant in `OVERLAY_VARIANTS`).

Everything runs inside `no_statsbomb_reads()`; the index is read for shot_id, game and frame_path only (no label).
Writes `<out>/complete.json` last.
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
from tqdm import tqdm
from xgpoc.bundle import canonical_digest, sha256_file
from xgpoc.detector import extract as ex
from xgpoc.detector import homography_dev as dev
from xgpoc.detector import lines, records
from xgpoc.detector.guard import no_statsbomb_reads
from xgpoc.paths import INDEX

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "results/xgv_split_plan/plan.csv"
FULL_C4 = ROOT / "cache/xgd_full/C4"

V = dev.Variant
VARIANTS = (
    V("letterbox", "letterbox"),                           # the package input of C1-C4, with the degenerate-fit fix
    V("letterbox_refined", "letterbox", refine_with_lines=True),
    V("stretch", "stretch"),
    V("stretch_kp03", "stretch", min_confidence=0.3),
    V("stretch_refined", "stretch", refine_with_lines=True),
    V("stretch_kp03_refined", "stretch", min_confidence=0.3, refine_with_lines=True),
)
OVERLAY_VARIANTS = ("C4", "stretch_kp03", "stretch_kp03_refined")
OVERLAY_COLORS = {"C4": (0, 0, 255), "stretch_kp03": (255, 0, 255), "stretch_kp03_refined": (255, 255, 0)}


def keypoint_cache(frames: pd.DataFrame, folder: Path, device: str | None) -> pd.DataFrame:
    """The first keypoint instance per frame and input, computed once (GPU) and reused when the cache is complete."""
    complete = folder / "complete.json"
    sample = canonical_digest(sorted(frames.shot_id))
    if complete.exists():
        record = json.loads(complete.read_text())
        if record["sample_digest"] != sample:
            raise SystemExit(f"{folder} holds keypoints of another sample; use a new --cache")
        return pd.read_parquet(folder / "keypoints.parquet")
    import tactifoot_vision as tv

    models = {name: tv.load_model("yolo_pose", ex.KEYPOINT_WEIGHTS, conf=dev.INSTANCE_CONF, stretch=name == "stretch",
                                  device=device) for name in ("letterbox", "stretch")}
    started, clock = datetime.now(UTC), time.perf_counter()
    rows = []
    for shot_id, path in tqdm(list(zip(frames.shot_id, frames.frame_path, strict=True)), desc="keypoints", mininterval=30):
        image = cv2.imread(path)
        for name, model in models.items():
            rows.append({"shot_id": shot_id, "input": name} | dev.keypoint_row(model(image)))
    table = pd.DataFrame(rows)
    folder.mkdir(parents=True, exist_ok=True)
    table.to_parquet(folder / "keypoints.parquet", index=False)
    record = {"purpose": "xg-detector 08: pitch keypoints of training-match shot frames, letterbox and stretched input",
              "sample_digest": sample, "frames": len(frames), "instance_conf": dev.INSTANCE_CONF, "device": device,
              "keypoint_weights": {"path": str(ex.KEYPOINT_WEIGHTS), "sha256": sha256_file(ex.KEYPOINT_WEIGHTS)},
              "code": records.code_hashes(Path(__file__)), "git": records.git_state(),
              "libraries": records.library_versions(),
              "run": {"started": started.isoformat(timespec="seconds"), "seconds": round(time.perf_counter() - clock, 1)},
              "files": records.output_hashes(folder)}
    complete.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return table


def c4_frames(shot_ids: list[str]) -> pd.DataFrame:
    """t0 rows of ticket 02's full C4 run: its homography and the people boxes and classes."""
    table = pd.read_parquet(FULL_C4 / "frames.parquet", columns=["shot_id", "offset_ms", "homography", "det_xyxy", "det_class"])
    table = table[(table.offset_ms == 0) & table.shot_id.isin(shot_ids)].set_index("shot_id")
    missing = sorted(set(shot_ids) - set(table.index))
    if missing:
        raise ValueError(f"{len(missing)} sampled shots have no t0 row in {FULL_C4}, e.g. {missing[0]}")
    return table


def scores(view: lines.LineView, matrix, boxes: np.ndarray, players: np.ndarray) -> dict:
    return {"has_homography": matrix is not None} | lines.score(view, matrix) | {
        "players_on_pitch": lines.players_on_pitch(matrix, players), "height_ratio": lines.height_ratio(matrix, players)}


def overlay(image: np.ndarray, matrices: dict, line_view: lines.LineView, scored: dict) -> np.ndarray:
    """The frame with the line mask in green and each variant's projected markings in its colour, one panel each."""
    panels = []
    for name, matrix in matrices.items():
        panel = image.copy()
        panel[line_view.lines] = (0, 200, 0)
        panel = lines.draw(panel, matrix, OVERLAY_COLORS[name])
        agreement = scored[name]["agreement"]
        label = f"{name}: " + ("no homography" if matrix is None else f"agreement {agreement:.2f}")
        cv2.putText(panel, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        panels.append(cv2.resize(panel, (960, 540)))
    return np.vstack(panels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overlays", type=int, default=30)
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--plan", type=Path, default=PLAN)
    parser.add_argument("--cache", type=Path, default=ROOT / "cache/xgd_homography_dev")
    parser.add_argument("--out", type=Path, default=ROOT / "results/xgd_homography_dev")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if (args.out / "complete.json").exists():
        raise SystemExit(f"{args.out} is complete; use a new --out")

    with no_statsbomb_reads():
        started, clock = datetime.now(UTC), time.perf_counter()
        plan = pd.read_csv(args.plan)
        index = pd.read_parquet(args.index, columns=["shot_id", "game", "frame_path"])
        frames = dev.sample_training_frames(plan, index, args.n, args.seed)
        keypoints = keypoint_cache(frames, args.cache, args.device).set_index(["shot_id", "input"])
        c4 = c4_frames(list(frames.shot_id))
        estimators = {v.name: (v, v.estimator()) for v in VARIANTS}
        overlay_ids = set(frames.shot_id.sample(min(args.overlays, len(frames)), random_state=args.seed))

        rows = []
        (args.out / "overlays").mkdir(parents=True, exist_ok=True)
        for shot in tqdm(frames.itertuples(index=False), total=len(frames), desc="variants", mininterval=30):
            image = cv2.imread(shot.frame_path)
            line_view = lines.LineView.of(image)
            row = c4.loc[shot.shot_id]
            boxes = np.asarray(row.det_xyxy, dtype=float).reshape(-1, 4)
            players = boxes[np.asarray(row.det_class, dtype=str) == "player"]
            homography = row.homography
            matrices = {"C4": np.asarray(homography, dtype=float).reshape(3, 3) if ex._listed(homography) else None}
            for name, (variant, estimator) in estimators.items():   # noqa: B007 (variant names the cached input)
                found = dev.keypoints_from_row(keypoints.loc[(shot.shot_id, variant.keypoint_input)])
                matrices[name] = estimator.fit(found, image, boxes)
            scored = {name: scores(line_view, matrix, boxes, players) for name, matrix in matrices.items()}
            for name, values in scored.items():
                rows.append({"shot_id": shot.shot_id, "game": shot.game, "fold": shot.fold, "subset": shot.subset,
                             "variant": name} | values)
            if shot.shot_id in overlay_ids:
                shown = {name: matrices[name] for name in OVERLAY_VARIANTS}
                cv2.imwrite(str(args.out / "overlays" / f"{shot.shot_id}.jpg"), overlay(image, shown, line_view, scored),
                            [cv2.IMWRITE_JPEG_QUALITY, 85])
        per_frame = pd.DataFrame(rows)
        summary = dev.summarise(per_frame)
        per_frame.to_parquet(args.out / "per_frame.parquet", index=False)
        summary.to_csv(args.out / "summary.csv", index=False, float_format="%.4f")
        frames.drop(columns=["frame_path"]).to_csv(args.out / "frames.csv", index=False)
        record = {"purpose": "xg-detector 08: label-free line agreement of homography variants on training-match frames",
                  "variants": {v.name: v.settings() for v in VARIANTS} | {"C4": "cache/xgd_full/C4 t0 homography"},
                  "parameters": {"n": args.n, "seed": args.seed, "tol_px": lines.TOL, "min_points": lines.MIN_POINTS,
                                 "good": dev.GOOD, "dev_folds": dev.DEV_FOLDS, "confirm_folds": dev.CONFIRM_FOLDS},
                  "inputs": {"index": {"path": str(args.index), "sha256": sha256_file(args.index),
                                       "columns": ["shot_id", "game", "frame_path"]},
                             "plan": {"path": str(args.plan), "sha256": sha256_file(args.plan)},
                             "c4": {"path": str(FULL_C4), "complete_json_sha256": sha256_file(FULL_C4 / "complete.json")},
                             "keypoints": {"path": str(args.cache),
                                           "complete_json_sha256": sha256_file(args.cache / "complete.json")}},
                  "sample": {"frames": len(frames), "games": int(frames.game.nunique()),
                             "subsets": frames.subset.value_counts().to_dict(),
                             "digest": canonical_digest(sorted(frames.shot_id))},
                  "code": records.code_hashes(Path(__file__)), "git": records.git_state(),
                  "run": {"started": started.isoformat(timespec="seconds"), "seconds": round(time.perf_counter() - clock, 1),
                          "finished": datetime.now(UTC).isoformat(timespec="seconds")},
                  "files": records.output_hashes(args.out)}
        (args.out / "complete.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
