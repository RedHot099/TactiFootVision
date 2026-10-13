"""The package's square input against the keypoints ticket 03 evaluated (pitch-keypoints 04).

    ./run.sh python scripts/pk_square_input_check.py [--cache cache/pk_label_free/<key>] [--out results/pk_square_input]
        [--device cuda]

Ticket 03 ran its pitch model through the evaluator's own square canvas (`SquareInput` at 5fc946b: the frame padded to
a centred grey square, fed to the package's letterbox input at 960, the outputs moved back). The package now does this
itself: `tv.load_model("yolo_pose", W, conf=0.1, imgsz=960, resize="square")`. This script runs the package input on
the frames of ticket 03's label-free keypoint cache (ticket 08's 1,500 training-match t0 shot frames, as read from
disk) and compares its first instance with the cached one: instance found or not, keypoint positions (px), keypoint and
instance confidences. Identical inputs give identical outputs, so the expected differences are 0.

Writes `per_frame.parquet` (the differences per frame) and `complete.json` (the maxima, inputs and run), last. Runs
inside `no_statsbomb_reads()`; the index is read for shot_id and frame_path only.
"""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from xgpoc.bundle import sha256_file
from xgpoc.detector import homography_dev as dev
from xgpoc.detector import records
from xgpoc.detector.guard import no_statsbomb_reads
from xgpoc.paths import INDEX

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache/pk_label_free/5be9954a0682_square_960_i0.1_n1500_s0"


def differences(cached: dict, new: dict) -> dict:
    """How one frame's new first instance differs from the cached one (`dev.keypoint_row` rows)."""
    has_a, has_b = cached["kp_xy"] is not None, new["kp_xy"] is not None
    out = {"instance_cached": has_a, "instance_new": has_b, "xy_px": np.nan, "conf": np.nan, "instance_conf": np.nan}
    if has_a and has_b:
        xy_a, xy_b = np.asarray(cached["kp_xy"], float).reshape(-1, 2), np.asarray(new["kp_xy"], float).reshape(-1, 2)
        out |= {"xy_px": float(np.hypot(*(xy_a - xy_b).T).max()),
                "conf": float(np.abs(np.asarray(cached["kp_conf"], float) - np.asarray(new["kp_conf"], float)).max()),
                "instance_conf": abs(float(cached["kp_instance_conf"]) - float(new["kp_instance_conf"]))}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--out", type=Path, default=ROOT / "results/pk_square_input")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if (args.out / "complete.json").exists():
        raise SystemExit(f"{args.out} is complete; use a new --out")
    cache_record = json.loads((args.cache / "complete.json").read_text())
    records.verified_file(args.cache / "table.parquet")
    spec = cache_record["identity"]["model"]
    if spec["input"] != "square":
        raise SystemExit(f"{args.cache} holds the {spec['input']} input, not the square canvas")
    weights = (ROOT / spec["weights"]).resolve() if not Path(spec["weights"]).is_absolute() else Path(spec["weights"])
    if sha256_file(weights) != spec["sha256"]:
        raise SystemExit(f"{weights} is not the cached model")

    with no_statsbomb_reads():
        import tactifoot_vision as tv

        started, clock = datetime.now(UTC), time.perf_counter()
        cached = pd.read_parquet(args.cache / "table.parquet").set_index("shot_id")
        paths = pd.read_parquet(args.index, columns=["shot_id", "frame_path"]).set_index("shot_id").frame_path
        model = tv.load_model("yolo_pose", weights, conf=spec["instance_conf"], imgsz=spec["imgsz"], resize="square",
                              device=args.device)
        rows = []
        for shot_id in tqdm(cached.index, desc="square input", mininterval=30):
            new = dev.keypoint_row(model(cv2.imread(paths[shot_id])))
            old = cached.loc[shot_id].to_dict()
            old["kp_xy"] = None if old["kp_xy"] is None or isinstance(old["kp_xy"], float) else old["kp_xy"]
            rows.append({"shot_id": shot_id} | differences(old, new))
        table = pd.DataFrame(rows)
        args.out.mkdir(parents=True, exist_ok=True)
        table.to_parquet(args.out / "per_frame.parquet", index=False)
        both = table.instance_cached & table.instance_new
        summary = {"frames": len(table), "instance_cached": int(table.instance_cached.sum()),
                   "instance_new": int(table.instance_new.sum()),
                   "instance_mismatch": int((table.instance_cached != table.instance_new).sum()),
                   "max_xy_px": float(table.xy_px[both].max()), "max_conf": float(table.conf[both].max()),
                   "max_instance_conf": float(table.instance_conf[both].max())}
        record = {"purpose": "pitch-keypoints 04: the package's resize='square' against ticket 03's evaluated keypoints",
                  "model": spec, "package_call": f"tv.load_model('yolo_pose', W, conf={spec['instance_conf']}, "
                                                 f"imgsz={spec['imgsz']}, resize='square')",
                  "inputs": {"cache": {"path": str(args.cache), "complete_json_sha256": sha256_file(args.cache / "complete.json")},
                             "index": {"path": str(args.index), "sha256": sha256_file(args.index),
                                       "columns": ["shot_id", "frame_path"]}},
                  "summary": summary, "code": records.code_hashes(Path(__file__)), "git": records.git_state(),
                  "libraries": records.library_versions(),
                  "run": {"started": started.isoformat(timespec="seconds"),
                          "finished": datetime.now(UTC).isoformat(timespec="seconds"),
                          "seconds": round(time.perf_counter() - clock, 1), "device": args.device},
                  "files": records.output_hashes(args.out)}
        (args.out / "complete.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
