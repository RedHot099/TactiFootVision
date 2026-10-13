"""Copy a chosen pitch keypoint checkpoint to `models/` with its manifest (pitch-keypoints 03).

    ./run.sh python scripts/pk_publish_model.py --run <runs>/<name> --weights best.pt|weights/best.pt
        --out models/<file>.pt --backend yolo_pose|heatmap --input letterbox [--imgsz 960] [--note TEXT]
        [--settings research/pitchkp_methods/<chosen>.json]

Writes `<out>` and `<out>.json`: sha256 of the weights, the dataset (ticket 01's folder, its `complete.json` sha256 and
labels digest), the run's training configuration (`args.yaml` of Ultralytics or `args.json` of the heatmap script),
the run's sessions (wall time) and results, the inference settings and homography method the evaluation chose, and the
git commit. `models/` is untracked; the manifest is the record.
"""

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import yaml
from xgpoc.bundle import sha256_file
from xgpoc.detector import records
from xgpoc.pitchkp import evaluator as ev
from xgpoc.pitchkp import training as T

ROOT = Path(__file__).resolve().parents[1]
DATASET_RECORD = ROOT / "results/pk_dataset/complete.json"


def training_config(run: Path) -> dict:
    if (run / "args.yaml").exists():
        return yaml.safe_load((run / "args.yaml").read_text())
    return json.loads((run / "args.json").read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--weights", required=True, help="checkpoint path inside the run folder")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", choices=ev.BACKENDS, required=True)
    parser.add_argument("--input", choices=ev.INPUTS, default="letterbox")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--settings", type=Path, default=None, help="the chosen homography method (JSON)")
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    source = args.run / args.weights
    if args.out.exists():
        raise SystemExit(f"{args.out} exists")
    ev.KeypointModel(source, args.input, args.imgsz, backend=args.backend)        # checks the combination
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, args.out)
    dataset = json.loads(DATASET_RECORD.read_text())
    sessions = [json.loads(line) for line in (args.run / "sessions.jsonl").read_text().splitlines()]
    manifest = {
        "purpose": "pitch-keypoints 03: the chosen pitch keypoint model",
        "weights": {"file": args.out.name, "sha256": sha256_file(args.out), "source": str(source)},
        "inference": {"backend": args.backend, "input": args.input, "imgsz": args.imgsz,
                      "homography": json.loads(args.settings.read_text()) if args.settings else None},
        "data": {"dataset": str(T.DATASET), "record": str(DATASET_RECORD), "record_sha256": sha256_file(DATASET_RECORD),
                 "labels_digest": dataset["dataset"]["labels_digest"], "training_lists": str(args.run / "data")},
        "training": {"run": str(args.run), "config": training_config(args.run),
                     "gpu_seconds": round(sum(s.get("seconds", 0) for s in sessions if s["event"] == "end"), 1),
                     "sessions": sessions},
        "git": records.git_state(), "note": args.note,
        "written": datetime.now(UTC).isoformat(timespec="seconds")}
    Path(f"{args.out}.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(manifest["weights"]))


if __name__ == "__main__":
    main()
