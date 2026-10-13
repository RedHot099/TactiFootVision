"""Ticket 08's label-free loop on a new pitch keypoint model (pitch-keypoints 03).

    ./run.sh python scripts/pk_label_free_loop.py --weights W --input letterbox|stretch [--imgsz N]
        --methods-file M.json --out results/pk_label_free/<name> [--n 1500] [--seed 0] [--device cuda]

The frames, scores and summary of `scripts/xgd_homography_dev.py` (xg-detector 08): the t0 frames of `--n` random shots
of the split plan's training matches (seed 0: the same 1,500 frames; folds 0-2 `dev`, 3-4 `confirm`), the line
agreement, players on the pitch, the height ratio and mirror images, summarised by `homography_dev.summarise`. The
model's keypoints (first instance, GPU) are cached under `<cache>/<model key>/`. Each method of `--methods-file`
(`xgpoc.pitchkp.evaluator.Method` fields) runs `HomographyEstimator.fit` on them; `mask="detector"` takes the people
boxes of ticket 02's full C4 run at t0 (the C4 detector), as the loop does. The current model's `stretch` and
`stretch_refined` rows of the loop's own run (`--reference`, the same frames) are copied in as `current_kp05` and
`current_kp05_refined`, its keypoint-only and line-refined fits at keypoint threshold 0.5 (D1 and D3).

`keypoint_residuals.csv`: on frames where `--residual-method`'s lines agree >= 0.9, the pixel distance of each
confident keypoint from its landmark under that homography (as the loop measured the current model: 48 px median).

Everything runs inside `no_statsbomb_reads()`; the index is read for shot_id, game and frame_path only (no label).
"""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pitchkp_gsr_eval as gsr
import xgd_homography_dev as loop
from tqdm import tqdm
from xgpoc.bundle import canonical_digest, sha256_file
from xgpoc.detector import homography_dev as dev
from xgpoc.detector import lines, records
from xgpoc.detector.guard import no_statsbomb_reads
from xgpoc.paths import INDEX
from xgpoc.pitchkp import evaluator as ev

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_VARIANTS = {"stretch": "current_kp05", "stretch_refined": "current_kp05_refined"}


def keypoint_table(frames: pd.DataFrame, model: ev.KeypointModel, folder: Path, device: str | None) -> pd.DataFrame:
    """The model's first keypoint instance per shot frame (as read from disk), cached."""
    identity = {"shots": canonical_digest(sorted(frames.shot_id)), "n": len(frames),
                "model": model.settings() | {"sha256": sha256_file(model.weights)}}

    def compute() -> pd.DataFrame:
        predict = model.load(device)
        return pd.DataFrame([{"shot_id": s} | dev.keypoint_row(predict(cv2.imread(p)))
                             for s, p in tqdm(list(zip(frames.shot_id, frames.frame_path, strict=True)),
                                              desc="keypoints", mininterval=30)])
    return gsr.cached_table(folder, identity, compute, "pitch-keypoints 03: keypoints of a pitch model on ticket 08's "
                                                       "training-match shot frames")


def reference_rows(reference: Path, shot_ids: set[str]) -> pd.DataFrame:
    """The current model's keypoint-only and refined rows of the loop's own run, renamed; the same frames."""
    records.verified_file(reference / "per_frame.parquet")
    rows = pd.read_parquet(reference / "per_frame.parquet")
    rows = rows[rows.variant.isin(REFERENCE_VARIANTS)].assign(variant=lambda t: t.variant.map(REFERENCE_VARIANTS))
    if set(rows.shot_id) != shot_ids:
        raise SystemExit(f"{reference} was run on other frames")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--input", default="letterbox", choices=ev.INPUTS)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--instance-conf", type=float, default=dev.INSTANCE_CONF)
    parser.add_argument("--backend", default="yolo_pose", choices=ev.BACKENDS)
    parser.add_argument("--methods-file", type=Path, required=True)
    parser.add_argument("--residual-method", default=None, help="default: the first refined method")
    parser.add_argument("--n", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--plan", type=Path, default=loop.PLAN)
    parser.add_argument("--reference", type=Path, default=ROOT / "results/xgd_homography_dev")
    parser.add_argument("--cache", type=Path, default=ROOT / "cache/pk_label_free")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if (args.out / "complete.json").exists():
        raise SystemExit(f"{args.out} is complete; use a new --out")
    model = ev.KeypointModel(args.weights, args.input, args.imgsz, args.instance_conf, args.backend)
    methods = ev.load_methods(args.methods_file)
    if any(m.source != "keypoints" or m.mask == "gt" for m in methods.values()):
        raise SystemExit("shot frames have no labels or ground-truth boxes: keypoint methods with mask detector/none")
    refined = [n for n, m in methods.items() if m.refine_with_lines]
    residual_method = args.residual_method or (refined[0] if refined else None)
    folder = args.cache / (f"{sha256_file(model.weights)[:12]}_{model.input}_{model.imgsz or 'train'}"
                           f"_i{model.instance_conf:g}_n{args.n}_s{args.seed}")

    with no_statsbomb_reads():
        started, clock = datetime.now(UTC), time.perf_counter()
        plan = pd.read_csv(args.plan)
        index = pd.read_parquet(args.index, columns=["shot_id", "game", "frame_path"])
        frames = dev.sample_training_frames(plan, index, args.n, args.seed)
        keypoints = keypoint_table(frames, model, folder, args.device).set_index("shot_id")
        gpu_done = datetime.now(UTC)
        c4 = loop.c4_frames(list(frames.shot_id))
        estimators = {name: m.estimator() for name, m in methods.items()}
        rows, residuals = [], []
        for shot in tqdm(frames.itertuples(index=False), total=len(frames), desc="methods", mininterval=30):
            image = cv2.imread(shot.frame_path)
            view = lines.LineView.of(image)
            row = c4.loc[shot.shot_id]
            boxes = np.asarray(row.det_xyxy, dtype=float).reshape(-1, 4)
            players = boxes[np.asarray(row.det_class, dtype=str) == "player"]
            found = ev.keypoints_from_row(keypoints.loc[shot.shot_id], model.instance_conf)
            matrices = {name: estimators[name].fit(found, image, boxes if m.mask == "detector" else None)
                        for name, m in methods.items()}
            scored = {name: loop.scores(view, matrix, boxes, players) for name, matrix in matrices.items()}
            rows += [{"shot_id": shot.shot_id, "game": shot.game, "fold": shot.fold, "subset": shot.subset,
                      "variant": name} | values for name, values in scored.items()]
            if residual_method and scored[residual_method]["agreement"] >= loop.CONFIRMED:
                residuals += [{"shot_id": shot.shot_id, "vertex": int(v), "px": d}
                              for v, d in lines.keypoint_residuals(found, matrices[residual_method])]
        per_frame = pd.concat([pd.DataFrame(rows), reference_rows(args.reference, set(frames.shot_id))],
                              ignore_index=True)
        summary = dev.summarise(per_frame)
        args.out.mkdir(parents=True, exist_ok=True)
        per_frame.to_parquet(args.out / "per_frame.parquet", index=False)
        summary.to_csv(args.out / "summary.csv", index=False, float_format="%.4f")
        residual_table = pd.DataFrame(residuals, columns=["shot_id", "vertex", "px"])
        residual_table.to_csv(args.out / "keypoint_residuals.csv", index=False, float_format="%.3f")
        record = {
            "purpose": "pitch-keypoints 03: ticket 08's label-free loop on a new pitch keypoint model",
            "model": model.settings() | {"sha256": sha256_file(model.weights)},
            "methods": {name: m.settings() for name, m in methods.items()},
            "reference_variants": REFERENCE_VARIANTS, "residual_method": residual_method,
            "residuals": {"frames": int(residual_table.shot_id.nunique()),
                          "median_px": float(residual_table.px.median()) if len(residual_table) else None},
            "parameters": {"n": args.n, "seed": args.seed, "tol_px": lines.TOL, "good": dev.GOOD,
                           "confirmed_agreement": loop.CONFIRMED},
            "inputs": {"index": {"path": str(args.index), "sha256": sha256_file(args.index),
                                 "columns": ["shot_id", "game", "frame_path"]},
                       "plan": {"path": str(args.plan), "sha256": sha256_file(args.plan)},
                       "c4": {"path": str(loop.FULL_C4),
                              "complete_json_sha256": sha256_file(loop.FULL_C4 / "complete.json")},
                       "reference": {"path": str(args.reference),
                                     "complete_json_sha256": sha256_file(args.reference / "complete.json")},
                       "keypoints": {"path": str(folder),
                                     "complete_json_sha256": sha256_file(folder / "complete.json")}},
            "sample": {"frames": len(frames), "games": int(frames.game.nunique()),
                       "subsets": frames.subset.value_counts().to_dict(),
                       "digest": canonical_digest(sorted(frames.shot_id))},
            "code": records.code_hashes(Path(__file__)), "git": records.git_state(),
            "run": {"started": started.isoformat(timespec="seconds"), "gpu_done": gpu_done.isoformat(timespec="seconds"),
                    "finished": datetime.now(UTC).isoformat(timespec="seconds"),
                    "seconds": round(time.perf_counter() - clock, 1)},
            "files": records.output_hashes(args.out)}
        (args.out / "complete.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary[summary.subset == "all"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
