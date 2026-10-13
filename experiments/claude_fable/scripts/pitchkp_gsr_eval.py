"""Homography evaluation in metres on SoccerNet GSR, and keypoint error in pixels, for a pitch keypoint model
(pitch-keypoints 02; ticket 03 runs its candidates through it).

    ./run.sh python scripts/pitchkp_gsr_eval.py --split valid|test|calib_test [--device cuda] [--workers 14]
        [--weights W --input stretch|letterbox --imgsz N]        the keypoint model (default: the current model)
        [--methods-file M.json]                                  methods (default: the baseline, `baseline_methods`)
        [--reference-method D3 [--reference-run DIR]]            paired differences against this method
        [--out DIR] [--cache DIR] [--clips N] [--draws 2000]

GSR splits (`valid`, `test`): every 5th frame of each clip (1, 6, 11, ...), resized to 1280 x 720
(`gsr_eval.EVAL_SIZE`), the size of the shot frames on which xg-detector 08 chose D1-D4.
1. GPU, cached under `<cache>/<split>/` with completion records and reused when complete: the model's keypoints
   (first instance), and when a method uses the detector mask, the package detector's people boxes (xg-detector's
   C4/D3 detector: `common.detector` of `research/xg_detector_audit2_configs.json`, sha256-checked).
2. CPU, one process per clip (`xgpoc.pitchkp.evaluator.evaluate_clip`): every method's homography, the persons' errors in
   metres, ticket 08's line agreement; the label oracle uses ticket 01's kept label homographies (`--labels`).
3. Outputs: `per_frame.parquet`; `summary.csv` (all and near-goal, clip-bootstrap 95% intervals); `paired.csv`
   (method - reference); `per_game.csv`, `per_clip.csv`; `keypoint_errors.parquet` and `keypoints_summary.csv` (px at
   720p against ticket 01's kept labels); `convention.csv`; `complete.json` (last).

`calib_test`: Camera Calibration 2023 test images with a kept label (no pitch positions): the keypoint error only,
bootstrap over games.

Reads no StatsBomb data (`no_statsbomb_reads()`).
"""

import argparse
import json
import multiprocessing
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")      # one thread per worker process
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from tqdm import tqdm
from xgpoc.bundle import canonical_digest, sha256_file
from xgpoc.detector import extract as ex
from xgpoc.detector import homography_dev as dev
from xgpoc.detector import lines, records
from xgpoc.detector.guard import no_statsbomb_reads
from xgpoc.pitchkp import evaluator as ev
from xgpoc.pitchkp import gsr_eval as ge

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "research/xg_detector_audit2_configs.json"
DATA = records.CHECKOUT / "data/SoccerNetGS"
LABELS = ROOT / "results/pk_dataset/fits.parquet"
SPLITS = ("valid", "test", "calib_test")


def cached_table(folder: Path, identity: dict, compute: Callable[[], pd.DataFrame], purpose: str) -> pd.DataFrame:
    """`folder/table.parquet`, computed once and reused while its record's `identity` matches."""
    complete = folder / "complete.json"
    if complete.exists():
        if json.loads(complete.read_text())["identity"] != identity:
            raise SystemExit(f"{folder} holds another sample or model; use a new --cache")
        return pd.read_parquet(folder / "table.parquet")
    started, clock = datetime.now(UTC), time.perf_counter()
    table = compute()
    folder.mkdir(parents=True, exist_ok=True)
    table.to_parquet(folder / "table.parquet", index=False)
    record = {"purpose": purpose, "identity": identity, "git": records.git_state(),
              "libraries": records.library_versions(),
              "run": {"started": started.isoformat(timespec="seconds"),
                      "finished": datetime.now(UTC).isoformat(timespec="seconds"),
                      "seconds": round(time.perf_counter() - clock, 1)},
              "files": records.output_hashes(folder)}
    complete.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return table


def frame_digest(frames: pd.DataFrame) -> str:
    return canonical_digest(sorted(f"{g}/{f}" for g, f in zip(frames["group"], frames["frame"], strict=True)))


def keypoint_table(frames: pd.DataFrame, model: ev.KeypointModel, folder: Path, device: str | None) -> pd.DataFrame:
    """The model's first keypoint instance per frame (`frames`: group, frame, path), at 720p."""
    identity = {"frames": frame_digest(frames), "n": len(frames), "model": model.settings() | {
        "sha256": sha256_file(model.weights)}, "eval_size": list(ge.EVAL_SIZE)}

    def compute() -> pd.DataFrame:
        predict = model.load(device)
        return pd.DataFrame([{"group": g, "frame": f} | dev.keypoint_row(predict(ev.eval_frame(p)))
                             for g, f, p in tqdm(list(zip(frames["group"], frames["frame"], frames["path"], strict=True)),
                                                 desc="keypoints", mininterval=30)])
    return cached_table(folder, identity, compute, "pitch-keypoints 02: keypoints of a pitch model at 720p")


def detector_table(frames: pd.DataFrame, configs: dict, folder: Path, device: str | None) -> pd.DataFrame:
    """The package detector's people boxes (720p xyxy, every class but the ball) per frame, at the configs file's
    detector settings (xg-detector's C4/D3)."""
    spec = configs["common"]["detector"]
    weights = ex.DETECTOR_WEIGHTS
    if sha256_file(weights) != spec["weights_sha256"]:
        raise SystemExit(f"{weights} is not the detector of the configs file")
    identity = {"frames": frame_digest(frames), "n": len(frames), "eval_size": list(ge.EVAL_SIZE),
                "detector": {k: spec[k] for k in ("weights_sha256", "imgsz", "conf", "iou", "class_conf", "stretch")}}

    def compute() -> pd.DataFrame:
        import tactifoot_vision as tv

        detector = tv.load_model("yolo", weights, conf=spec["conf"], iou=spec["iou"], imgsz=spec["imgsz"],
                                 class_conf=spec["class_conf"], stretch=spec["stretch"], device=device)
        rows = []
        for g, f, p in tqdm(list(zip(frames["group"], frames["frame"], frames["path"], strict=True)), desc="detector",
                            mininterval=30):
            found = detector.predict(ev.eval_frame(p))
            people = found[found.data["class_name"] != "ball"]
            rows.append({"group": g, "frame": f, "det_xyxy": [float(v) for v in people.xyxy.ravel()],
                         "det_class": [str(c) for c in people.data["class_name"]]})
        return pd.DataFrame(rows)
    return cached_table(folder, identity, compute, "pitch-keypoints 02: the package detector's people boxes at 720p")


def label_table(path: Path, source: str, split: str) -> pd.DataFrame:
    """Ticket 01's kept labels of one source and split: group, frame, key, image, H (720p frame -> corner metres),
    xy (32 x 2, 720p) and visible."""
    fits = pd.read_parquet(path, columns=["key", "source", "split", "group", "frame", "image", "H", "xy", "visible",
                                          "reject"])
    kept = fits[(fits.source == source) & (fits.split == split) & (fits.reject == "")].reset_index(drop=True)
    if kept.empty:
        raise SystemExit(f"no kept {source}/{split} labels in {path}")
    return kept


def vertex_rows(split: str, keypoints: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """`gsr_eval.vertex_errors` of every labelled frame that has keypoints, with split, group and frame."""
    index = keypoints.set_index(["group", "frame"])
    parts = []
    for label in labels.itertuples(index=False):
        if (label.group, label.frame) not in index.index:
            continue
        found = dev.keypoints_from_row(index.loc[(label.group, label.frame)])
        xy, conf = (None, None) if found is None else (found.xy[0], found.confidence[0])
        parts.append(ge.vertex_errors(xy, conf, np.asarray(label.xy).reshape(-1, 2), np.asarray(label.visible))
                     .assign(split=split, group=label.group, frame=label.frame))
    return pd.concat(parts, ignore_index=True)


def gsr_frames(folders: list[Path]) -> pd.DataFrame:
    return pd.concat([ge.read_clip(f).frames.assign(group=f.name) for f in folders], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument("--weights", type=Path, default=None, help="default: the current model (configs file)")
    parser.add_argument("--input", default="stretch", choices=ev.INPUTS)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--instance-conf", type=float, default=dev.INSTANCE_CONF)
    parser.add_argument("--methods-file", type=Path, default=None, help="default: baseline_methods(configs)")
    parser.add_argument("--reference-method", default="D3")
    parser.add_argument("--reference-run", type=Path, default=None, help="a complete run of the same split")
    parser.add_argument("--configs", type=Path, default=CONFIGS)
    parser.add_argument("--labels", type=Path, default=LABELS)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--calib-data", type=Path, default=records.CHECKOUT / "data/SoccerNetCalibration")
    parser.add_argument("--cache", type=Path, default=ROOT / "cache/pitchkp_gsr_eval")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--clips", type=int, default=None, help="first N clips only (smoke runs)")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--draws", type=int, default=ge.BOOTSTRAP_DRAWS)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    out = args.out or ROOT / f"results/pitchkp_gsr_eval/{args.split}"
    if (out / "complete.json").exists():
        raise SystemExit(f"{out} is complete; use a new --out")

    configs = json.loads(args.configs.read_text())
    if args.weights is None and sha256_file(ex.KEYPOINT_WEIGHTS) != configs["common"]["keypoints"]["weights_sha256"]:
        raise SystemExit(f"{ex.KEYPOINT_WEIGHTS} is not the pitch model of {args.configs}")
    model = ev.KeypointModel(args.weights or ex.KEYPOINT_WEIGHTS, args.input, args.imgsz, args.instance_conf)
    methods = ev.load_methods(args.methods_file) if args.methods_file else ev.baseline_methods(args.configs)
    model_key = f"keypoints_{sha256_file(model.weights)[:12]}_{model.input}_{model.imgsz or 'train'}"

    with no_statsbomb_reads():
        started, clock = datetime.now(UTC), time.perf_counter()
        outputs: dict[str, pd.DataFrame] = {}
        record: dict = {"split": args.split, "model": model.settings() | {"sha256": sha256_file(model.weights)},
                        "inputs": {"labels": {"path": str(args.labels), "sha256": sha256_file(args.labels)}}}
        if args.split == "calib_test":
            labels = label_table(args.labels, "calib", "test")
            frames = labels.rename(columns={"image": "path"})[["group", "frame", "path"]].assign(frame=labels.key)
            keypoints = keypoint_table(frames, model, args.cache / args.split / model_key, args.device)
            gpu_done = datetime.now(UTC)
            outputs["keypoint_errors"] = vertex_rows(args.split, keypoints, labels.assign(frame=labels.key))
        else:
            folders = sorted((args.data / args.split).glob("SNGS-*"))[:args.clips]
            if not folders:
                raise SystemExit(f"no clips in {args.data / args.split}")
            frames = gsr_frames(folders)
            labels = label_table(args.labels, "gsr", args.split)
            labels = labels[labels.group.isin([f.name for f in folders])]
            keypoints = keypoint_table(frames, model, args.cache / args.split / model_key, args.device)
            detections = None
            if any(m.mask == "detector" for m in methods.values()):
                detections = detector_table(frames, configs, args.cache / args.split / "detector", args.device)
            gpu_done = datetime.now(UTC)
            label_h = {(r.group, r.frame): np.asarray(r.H, dtype=float).reshape(3, 3)
                       for r in labels.itertuples(index=False)}
            tasks = []
            for folder in folders:
                mine = detections[detections.group == folder.name] if detections is not None else None
                tasks.append(ev.ClipTask(
                    args.split, folder, keypoints[keypoints.group == folder.name].set_index("frame"), methods,
                    detections={} if mine is None else {f: np.asarray(b, dtype=float).reshape(-1, 4)
                                                        for f, b in zip(mine.frame, mine.det_xyxy, strict=True)},
                    labels={f: h for (g, f), h in label_h.items() if g == folder.name}))
            rows, convention = [], []
            with multiprocessing.get_context("fork").Pool(args.workers) as pool:
                for clip_rows, clip_convention in tqdm(pool.imap(ev.evaluate_clip, tasks), total=len(tasks),
                                                       desc="clips"):
                    rows += clip_rows
                    convention.append(clip_convention)
            per_frame = pd.DataFrame(rows)
            per_frame["scored"] = per_frame.scored.astype(bool)
            convention = pd.concat(convention, ignore_index=True)
            outputs |= {"per_frame": per_frame, "convention": convention,
                        "summary": ge.summarise(per_frame, draws=args.draws),
                        "per_game": ge.per_group(per_frame, "game"), "per_clip": ge.per_group(per_frame, "clip"),
                        "keypoint_errors": vertex_rows(args.split, keypoints, labels)}
            reference = per_frame[per_frame.method == args.reference_method]
            if args.reference_run is not None:
                reference_file = args.reference_run / "per_frame.parquet"
                records.verified_file(reference_file)
                reference = pd.read_parquet(reference_file)
                reference = reference[reference.method == args.reference_method]
                record["inputs"]["reference_run"] = {"path": str(args.reference_run),
                                                     "complete_json_sha256": sha256_file(args.reference_run /
                                                                                         "complete.json")}
            if not reference.empty:
                outputs["paired"] = ge.paired(per_frame, reference, draws=args.draws)
            line_columns = [c for c in ge.CONVENTION_LINES if c in convention]
            record |= {
                "methods": {name: m.settings() for name, m in methods.items()},
                "reference_method": args.reference_method if not reference.empty else None,
                "truth": "GSR bbox_pitch bottom-middle (centre origin -> corner origin) of every person but the ball "
                         "within 3 m of the pitch, against the projected bottom-middle of its bbox_image",
                "denominators": {"availability, under_*m, lines_good, refined": "scored frames (>= 1 person on the "
                                 "pitch with a pitch position; frames without GSR's camera have none)",
                                 "median_m, mean_m, p90_m": "scored frames with a homography; per-person errors "
                                 "capped at cap_m (also feet beyond the horizon)",
                                 "near_goal stratum": f"scored frames whose persons' median x is within "
                                 f"{ge.NEAR_GOAL_M} m of a goal line"},
                "sample": {"clips": len(folders), "frames": len(frames), "labelled_frames": len(labels),
                           "scored_frames": int(per_frame.drop_duplicates(["clip", "frame"]).scored.sum())},
                "convention": {"frames": len(convention),
                               "feet_loo_median_m": float(convention.feet_loo_median_m.median()),
                               "line_residual_median_m": {c: float(convention[c].median()) for c in line_columns}}}
            record["inputs"] |= {"configs": {"path": str(args.configs), "sha256": sha256_file(args.configs)},
                                 "gsr_labels": {f.name: sha256_file(f / "Labels-GameState.json") for f in folders}}
            if detections is not None:
                record["inputs"]["detector"] = json.loads((args.cache / args.split / "detector/complete.json")
                                                          .read_text())["identity"]["detector"]
        outputs["keypoints_summary"] = ge.summarise_keypoints(outputs["keypoint_errors"], draws=args.draws)

        out.mkdir(parents=True, exist_ok=True)
        for name, table in outputs.items():
            if name in ("per_frame", "keypoint_errors"):
                table.to_parquet(out / f"{name}.parquet", index=False)
            else:
                table.to_csv(out / f"{name}.csv", index=False, float_format="%.4f")
        pitchkp = sorted((ROOT / "xgpoc/pitchkp").glob("*.py"))
        record |= {
            "purpose": "pitch-keypoints 02: homography error in metres and keypoint error in px of a pitch model",
            "parameters": {"frame_step": ge.FRAME_STEP, "eval_size": list(ge.EVAL_SIZE),
                           "on_pitch_margin_m": ge.ON_PITCH_MARGIN_M, "cap_m": ge.CAP_M,
                           "near_goal_m": ge.NEAR_GOAL_M, "thresholds_m": list(ge.THRESHOLDS_M),
                           "thresholds_px": list(ge.THRESHOLDS_PX), "keypoint_conf": ge.KEYPOINT_CONF,
                           "lines_good": ge.LINES_GOOD, "line_tol_px": lines.TOL, "draws": args.draws,
                           "seed": ge.BOOTSTRAP_SEED},
            "caches": {p.parent.name: sha256_file(p) for p in sorted((args.cache / args.split).glob("*/complete.json"))},
            "code": records.code_hashes(Path(__file__)) | {
                "pitchkp": {str(p.relative_to(records.CHECKOUT)): sha256_file(p) for p in pitchkp}},
            "git": records.git_state(), "libraries": records.library_versions(),
            "run": {"started": started.isoformat(timespec="seconds"), "gpu_done": gpu_done.isoformat(timespec="seconds"),
                    "finished": datetime.now(UTC).isoformat(timespec="seconds"),
                    "seconds": round(time.perf_counter() - clock, 1), "workers": args.workers},
            "files": records.output_hashes(out)}
        (out / "complete.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        if "summary" in outputs:
            for stratum, part in outputs["summary"].groupby("stratum", sort=False):
                print(stratum)
                print(part.pivot_table(index="method", columns="metric", values="value", sort=False).round(3).to_string())
        print(outputs["keypoints_summary"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
