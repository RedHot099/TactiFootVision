"""Homography evaluation in metres on SoccerNet GSR, and the current pitch model's baseline (pitch-keypoints 02).

    ./run.sh python scripts/pitchkp_gsr_eval.py --split valid [--methods D1 D2 D3 D4] [--workers 12] [--device cuda]
                                                [--data ../../data/SoccerNetGS] [--cache cache/pitchkp_gsr_eval/<split>]
                                                [--out results/pitchkp_gsr_eval/<split>] [--clips N]

Frames: every 5th frame of each clip of the split (frames 1, 6, 11, ...; `gsr_eval.keeps_frame`), resized to
1280 x 720 (`gsr_eval.EVAL_SIZE`), the size of the shot frames on which xg-detector 08 chose D1-D4.

1. Keypoints (GPU, cached in `<cache>/keypoints.parquet` with its completion record; reused when complete): the
   current pitch model (xg-detector's D1-D4 weights, checked by sha256 against `research/xg_detector_audit2_configs.json`)
   with the stretched input and instance threshold 0.1, first instance per frame.
2. Methods (CPU, one process per clip): D1-D4 of the configs file (keypoint threshold 0.5 / 0.3, line refinement off /
   on, RANSAC 10 m) through `HomographyEstimator.fit(keypoints, frame, boxes)`; the boxes are GSR's person boxes (the
   audit used the C4 detector's). A refined method's frame counts as refined when its matrix differs from the same
   threshold's keypoint fit.
3. Scores (`xgpoc.pitchkp.gsr_eval`): per-frame median error in metres of the persons' box feet against GSR's pitch
   positions, ticket 08's line agreement (`xgpoc.detector.lines`), and the summary with clip-bootstrap 95% intervals.
   `convention.csv`: the feet-fit check of GSR's coordinate convention and the floor of the feet metric.

Writes `<out>/per_frame.parquet`, `summary.csv`, `convention.csv` and `complete.json` (last). Reads no StatsBomb data
(`no_statsbomb_reads()`).
"""

import argparse
import dataclasses
import json
import multiprocessing
import os
import time
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")      # one thread per worker process
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from xgpoc.bundle import canonical_digest, sha256_file
from xgpoc.detector import extract as ex
from xgpoc.detector import homography_dev as dev
from xgpoc.detector import lines, records
from xgpoc.detector.guard import no_statsbomb_reads
from xgpoc.pitchkp import gsr_eval as ge

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "research/xg_detector_audit2_configs.json"
DATA = records.CHECKOUT / "data/SoccerNetGS"


def methods_from_configs(path: Path, names: list[str]) -> dict[str, dev.Variant]:
    """D1-D4 as `Variant`s: the configs file's keypoint threshold and refinement, its RANSAC limit, stretched input."""
    configs = json.loads(Path(path).read_text())
    ransac = configs["common"]["homography"]["ransac_threshold_m"]
    methods = {}
    for name in names:
        c = configs["configs"][name]
        if c["keypoint_input"] != "stretch":
            raise ValueError(f"{name}: only the stretched keypoint input is evaluated, got {c['keypoint_input']}")
        methods[name] = dev.Variant(name, "stretch", min_confidence=c["keypoint_conf"],
                                    refine_with_lines=c["refine_with_lines"], ransac_threshold=ransac)
    return methods


def eval_frame(path: str, size: tuple[int, int] = ge.EVAL_SIZE) -> np.ndarray:
    """The frame at `path` (BGR) resized to `size` (w, h) with area interpolation."""
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(path)
    return image if image.shape[1::-1] == size else cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def keypoint_cache(frames: pd.DataFrame, folder: Path, weights: Path, device: str | None) -> pd.DataFrame:
    """First keypoint instance per sampled frame (stretched input), computed once on the GPU and reused when complete."""
    complete = folder / "complete.json"
    weights_sha = sha256_file(weights)
    sample = canonical_digest(sorted(f"{c}/{f}" for c, f in zip(frames["clip"], frames.frame, strict=True)))
    if complete.exists():
        record = json.loads(complete.read_text())
        if record["sample_digest"] != sample or record["keypoint_weights"]["sha256"] != weights_sha:
            raise SystemExit(f"{folder} holds keypoints of another sample or model; use a new --cache")
        return pd.read_parquet(folder / "keypoints.parquet")
    import tactifoot_vision as tv

    model = tv.load_model("yolo_pose", weights, conf=dev.INSTANCE_CONF, stretch=True, device=device)
    started, clock = datetime.now(UTC), time.perf_counter()
    rows = [{"clip": clip, "frame": frame} | dev.keypoint_row(model(eval_frame(path)))
            for clip, frame, path in tqdm(list(zip(frames["clip"], frames.frame, frames.path, strict=True)),
                                          desc="keypoints", mininterval=30)]
    table = pd.DataFrame(rows)
    folder.mkdir(parents=True, exist_ok=True)
    table.to_parquet(folder / "keypoints.parquet", index=False)
    record = {"purpose": "pitch-keypoints 02: the current pitch model's keypoints on GSR frames (stretched input, 720p)",
              "sample_digest": sample, "frames": len(frames), "instance_conf": dev.INSTANCE_CONF, "device": device,
              "eval_size": list(ge.EVAL_SIZE), "keypoint_weights": {"path": str(weights), "sha256": weights_sha},
              "git": records.git_state(), "libraries": records.library_versions(),
              "run": {"started": started.isoformat(timespec="seconds"),
                      "finished": datetime.now(UTC).isoformat(timespec="seconds"),
                      "seconds": round(time.perf_counter() - clock, 1)},
              "files": records.output_hashes(folder)}
    complete.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return table


def evaluate_clip(task: tuple) -> tuple[list[dict], pd.DataFrame]:
    """Per-frame rows of every method on one clip's scored frames (and a row per unscored frame), and its convention
    check."""
    cv2.setNumThreads(1)
    split, folder, keypoints, methods = task
    clip = ge.read_clip(folder, ge.keeps_frame)
    estimators = {name: (m, m.estimator()) for name, m in methods.items()}
    unrefined = {name: dataclasses.replace(m, name=f"{name}_start", refine_with_lines=False).estimator()
                 for name, m in methods.items() if m.refine_with_lines}
    width, height = ge.EVAL_SIZE
    rows = []
    for frame in clip.frames.itertuples(index=False):
        scale = (width / frame.width, height / frame.height)
        persons = clip.persons[clip.persons.frame == frame.frame]
        truth_rows = persons[ge.on_pitch(persons)]
        base = {"split": split, "clip": clip.name, "frame": frame.frame, "persons": len(truth_rows),
                "scored": len(truth_rows) > 0}
        if not base["scored"]:
            rows += [base | {"method": name} for name in methods]
            continue
        image = eval_frame(frame.path)
        view = lines.LineView.of(image)
        boxes = persons[["x1", "y1", "x2", "y2"]].to_numpy(float) * np.tile(scale, 2)
        feet, truth = ge.feet(truth_rows, scale), truth_rows[["pitch_x", "pitch_y"]].to_numpy(float)
        found = dev.keypoints_from_row(keypoints.loc[frame.frame])
        for name, (method, estimator) in estimators.items():
            matrix = estimator.fit(found, image, boxes)
            errors = ge.person_errors(matrix, feet, truth, (height, width))
            row = base | {"method": name, "has_homography": matrix is not None,
                          "keypoints": int((found.confidence[0] >= method.min_confidence).sum()) if found is not None else 0,
                          "error_m": float(np.median(errors)) if errors is not None else np.nan,
                          "error_mean_m": float(errors.mean()) if errors is not None else np.nan,
                          "capped": int((errors >= ge.CAP_M).sum()) if errors is not None else 0,
                          "agreement": lines.score(view, matrix)["agreement"],
                          "homography": None if matrix is None else [float(v) for v in np.ravel(matrix)]}
            if name in unrefined:
                start = unrefined[name].fit(found)
                row["refined"] = float(matrix is not None and start is not None and not np.allclose(matrix, start))
            rows.append(row)
    return rows, ge.convention_check(clip)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", required=True, choices=["valid", "test"])
    parser.add_argument("--methods", nargs="+", default=["D1", "D2", "D3", "D4"])
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--configs", type=Path, default=CONFIGS)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--clips", type=int, default=None, help="first N clips only (smoke runs)")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--draws", type=int, default=ge.BOOTSTRAP_DRAWS)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    cache = args.cache or ROOT / f"cache/pitchkp_gsr_eval/{args.split}"
    out = args.out or ROOT / f"results/pitchkp_gsr_eval/{args.split}"
    if (out / "complete.json").exists():
        raise SystemExit(f"{out} is complete; use a new --out")

    configs = json.loads(args.configs.read_text())
    weights = ex.KEYPOINT_WEIGHTS
    if sha256_file(weights) != configs["common"]["keypoints"]["weights_sha256"]:
        raise SystemExit(f"{weights} is not the pitch model of {args.configs}")
    methods = methods_from_configs(args.configs, args.methods)

    with no_statsbomb_reads():
        started, clock = datetime.now(UTC), time.perf_counter()
        folders = sorted((args.data / args.split).glob("SNGS-*"))[:args.clips]
        if not folders:
            raise SystemExit(f"no clips in {args.data / args.split}")
        frames = pd.concat([ge.read_clip(f, ge.keeps_frame).frames.assign(clip=f.name) for f in folders],
                           ignore_index=True)
        keypoints = keypoint_cache(frames, cache, weights, args.device)
        gpu_done = datetime.now(UTC)
        tasks = [(args.split, folder, keypoints[keypoints["clip"] == folder.name].set_index("frame"), methods)
                 for folder in folders]
        rows, convention = [], []
        with multiprocessing.get_context("fork").Pool(args.workers) as pool:
            for clip_rows, clip_convention in tqdm(pool.imap(evaluate_clip, tasks), total=len(tasks), desc="clips"):
                rows += clip_rows
                convention.append(clip_convention)
        per_frame = pd.DataFrame(rows)
        per_frame["scored"] = per_frame.scored.astype(bool)
        summary = ge.summarise(per_frame, draws=args.draws)
        convention = pd.concat(convention, ignore_index=True)

        out.mkdir(parents=True, exist_ok=True)
        per_frame.to_parquet(out / "per_frame.parquet", index=False)
        summary.to_csv(out / "summary.csv", index=False, float_format="%.4f")
        convention.to_csv(out / "convention.csv", index=False, float_format="%.4f")
        line_columns = [c for c in ge.CONVENTION_LINES if c in convention]
        pitchkp = sorted((ROOT / "xgpoc/pitchkp").glob("*.py"))
        record = {"purpose": "pitch-keypoints 02: homography error in metres of D1-D4 (the current pitch model) on GSR",
                  "split": args.split,
                  "methods": {name: m.settings() for name, m in methods.items()},
                  "truth": "GSR bbox_pitch bottom-middle (centre origin -> corner origin) of every person but the ball "
                           "within 3 m of the pitch, against the projected bottom-middle of its bbox_image",
                  "denominators": {"availability, under_*m, lines_good, refined": "scored frames (>= 1 person on the "
                                   "pitch with a pitch position; frames without GSR's camera have none)",
                                   "median_m, mean_m, p90_m": "scored frames with a homography; per-person errors "
                                   "capped at cap_m (also feet beyond the horizon)"},
                  "parameters": {"frame_step": ge.FRAME_STEP, "eval_size": list(ge.EVAL_SIZE),
                                 "on_pitch_margin_m": ge.ON_PITCH_MARGIN_M, "cap_m": ge.CAP_M,
                                 "thresholds_m": list(ge.THRESHOLDS_M), "lines_good": ge.LINES_GOOD,
                                 "line_tol_px": lines.TOL, "draws": args.draws, "seed": ge.BOOTSTRAP_SEED,
                                 "boxes": "GSR bbox_image of every person (not the ball), for the refinement's line mask"},
                  "inputs": {"configs": {"path": str(args.configs), "sha256": sha256_file(args.configs)},
                             "keypoint_weights": {"path": str(weights), "sha256": sha256_file(weights)},
                             "keypoints": {"path": str(cache), "complete_json_sha256": sha256_file(cache / "complete.json")},
                             "labels": {f.name: sha256_file(f / "Labels-GameState.json") for f in folders}},
                  "sample": {"clips": len(folders), "frames": len(frames),
                             "scored_frames": int(per_frame.drop_duplicates(["clip", "frame"]).scored.sum())},
                  "convention": {"frames": len(convention),
                                 "feet_loo_median_m": float(convention.feet_loo_median_m.median()),
                                 "line_residual_median_m": {c: float(convention[c].median()) for c in line_columns}},
                  "code": records.code_hashes(Path(__file__)) | {
                      "pitchkp": {str(p.relative_to(records.CHECKOUT)): sha256_file(p) for p in pitchkp}},
                  "git": records.git_state(), "libraries": records.library_versions(),
                  "run": {"started": started.isoformat(timespec="seconds"), "keypoints_done": gpu_done.isoformat(
                      timespec="seconds"), "finished": datetime.now(UTC).isoformat(timespec="seconds"),
                          "seconds": round(time.perf_counter() - clock, 1), "workers": args.workers},
                  "files": records.output_hashes(out)}
        (out / "complete.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    wide = summary.pivot_table(index="method", columns="metric", values="value", sort=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(wide.round(3).to_string())


if __name__ == "__main__":
    main()
