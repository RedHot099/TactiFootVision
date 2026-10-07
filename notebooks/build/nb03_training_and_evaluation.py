"""03_training_and_evaluation.ipynb: `tv.models` training and `tv.evaluation` metrics."""

from common import Notebook


def notebook() -> Notebook:
    nb = Notebook("03_training_and_evaluation")
    nb.md("""
# Training and evaluation with `tv.models` and `tv.evaluation`

Three backends sit behind one `Model` interface:

| name | library | task |
|---|---|---|
| `yolo` | Ultralytics YOLO | player, goalkeeper, referee and ball boxes |
| `rfdetr` | RF-DETR | the same boxes, transformer detector |
| `yolo_pose` | Ultralytics YOLO-pose | the 32 pitch landmarks |

All three train with the same `train` call and are scored by the same
evaluation code, so their numbers compare directly.

The training runs here are **short demos**: a few epochs on a subset, enough
to show the interface in a few minutes. The checkpoints in `models/` were
trained on the full datasets for far longer (200 epochs for the YOLO models,
about 30 for RF-DETR), and they are the ones the other notebooks use.

Part of the [TactiFoot Vision notebooks](README.md).
""")
    nb.setup(
        imports="""
import logging
import os
import warnings
from pathlib import Path

import pandas as pd

import tactifoot_vision as tv
""",
        body="""
detection = tv.load_dataset(DATA / "datasets" / "football_yolo")
pitch_data = tv.load_dataset(DATA / "keypoints")
train_set = detection.subset({"train": 300, "valid": 60}, seed=0)  # demo-sized
print(train_set)
""",
    )
    nb.md("""
## Backends and settings

Models are looked up by name in a registry. `available_models` lists them, and
`tv.load_model(name, weights, ...)` creates one.
""")
    nb.code("""
tv.models.available_models()
""")
    nb.md("""
`TrainConfig` holds the settings every backend understands, with their
defaults. Anything else passed to it is a backend option (`mosaic` for
Ultralytics, `grad_accum_steps` for RF-DETR). Training checks the backend
options before it creates the run folder: an unknown name, an option that
would move the run folder (`project`, `name`, ...) or the backend's own name
for a `TrainConfig` field (Ultralytics `batch`) is refused with the field to
use instead. The rest go to the backend unchanged.
""")
    nb.code("""
fields = tv.TrainConfig.model_fields
pd.DataFrame({
    "default": {name: field.default for name, field in fields.items()},
    "description": {name: field.description or "" for name, field in fields.items()},
})
""")
    nb.code("""
print(tv.TrainConfig(epochs=50, mosaic=0.0))
print("backend options:", tv.TrainConfig(epochs=50, mosaic=0.0).backend_options)
""")
    nb.md("""
One config serves all three demo runs below: few epochs, everything written
under this notebook's output folder. `exist_ok=True` reuses the run folders
when the notebook runs again instead of numbering new ones.
""")
    nb.code("""
config = tv.TrainConfig(epochs=20, batch_size=16, output_dir=OUT / "runs", exist_ok=True)
""")

    nb.md("""
## Training: one call for every backend

`tv.train(name, dataset, config, weights=..., **overrides)` exports the dataset
in the format the backend needs (YOLO folder or COCO folder) into the run
folder, trains, and returns a `TrainResult`. Keyword overrides replace single
config fields. The trainers print a lot, so `%%capture` keeps their logs out of
the notebook (`yolo_log.show()` prints them).

The detectors train on the 300-image subset, a few of which are shown first.
""")
    nb.code("""
tv.viz.show_samples(train_set, "train", n=3)
""")
    nb.code("""
%%capture yolo_log
yolo_run = tv.train("yolo", train_set, config, weights="yolo11n.pt", name="yolo11n")
""")
    nb.md("""
A `TrainResult` holds the best checkpoint, the run folder, the per-epoch
history and the validation metrics of the best checkpoint. Metric keys are the
same for every backend.
""")
    nb.code("""
print("weights:", os.path.relpath(yolo_run.weights))
print("metrics:", {k: round(v, 3) for k, v in yolo_run.metrics.items()})
yolo_run.history.head(3)
""")
    nb.code("""
tv.viz.plot_training(yolo_run)
""")
    nb.md("""
RF-DETR trains through the same interface. Here the model is created first and
trained with `Model.train`, the method form of the same call; afterwards the
object holds the trained weights. `weights=None` starts from the COCO-pretrained
checkpoint. `grad_accum_steps` is an RF-DETR option.
""")
    nb.code("""
%%capture rfdetr_log
# RF-DETR suggests optimize_for_inference() on its first prediction and torch warns
# about a meshgrid argument inside RF-DETR; both are benign here.
logging.getLogger("rfdetr.detr").addFilter(lambda record: "not optimized for inference" not in record.getMessage())
warnings.filterwarnings("ignore", message="torch.meshgrid")
rfdetr = tv.load_model("rfdetr", size="base")
rfdetr_run = rfdetr.train(train_set, config, epochs=3, batch_size=4, grad_accum_steps=4, name="rfdetr")
""")
    nb.code("""
print("metrics:", {k: round(v, 3) for k, v in rfdetr_run.metrics.items()})
tv.viz.plot_training(rfdetr_run)
""")
    nb.md("""
RF-DETR ignores options it does not know, so a typo would go unnoticed. The
backend checks them and fails before the run folder is created.
""")
    nb.code("""
try:
    rfdetr.train(train_set, config, epoch=5)
except ValueError as error:
    print("ValueError:", error)
""")
    nb.md("""
The pitch keypoint model is a pose model. It trains on the pose dataset with
the same call; the dataset's `flip_idx` goes into the exported `data.yaml`, so
Ultralytics' own flip augmentation renames landmarks correctly.
""")
    nb.code("""
%%capture pose_log
pose_run = tv.train("yolo_pose", pitch_data, config, weights="yolov8n-pose.pt", epochs=40, name="pitch_pose")
""")
    nb.code("""
tv.viz.plot_training(pose_run)
""")

    nb.md("""
## Evaluation

`tv.evaluate(model, dataset, split="valid")` (or `model.evaluate(dataset)`)
picks the metrics from the model's task. Detectors get COCO-style mAP computed
by one shared implementation, with classes matched by name, so YOLO and
RF-DETR are scored identically. `TrainResult.load()` rebuilds a trained model.
""")
    nb.code("""
detectors = {
    "YOLO11n (demo run)": yolo_run.load(),
    "RF-DETR base (demo run)": rfdetr,
    "YOLO11m (models/)": tv.load_model("yolo", MODELS / "football_yolo11m.pt"),
    "RF-DETR base (models/)": tv.load_model("rfdetr", MODELS / "football_rfdetr_base.pth"),
}
detection_metrics = {name: tv.evaluate(model, detection) for name, model in detectors.items()}
pd.DataFrame({name: m.as_dict() for name, m in detection_metrics.items()}).T.round(3)
""")
    nb.md("""
`DetectionMetrics` also has a per-class table. The ball, small and often
blurred, is the hardest class.
""")
    nb.code("""
detection_metrics["RF-DETR base (models/)"].per_class.round(3)
""")
    nb.md("""
`plot_metrics` draws one model's per-class AP, or compares several side by side
when given a dict.
""")
    nb.code("""
tv.viz.plot_metrics(detection_metrics)
""")
    nb.md("""
Keypoint models get the mean pixel error of labelled landmarks and PCK, the
share of landmarks closer than a fraction of the image diagonal (5% by
default). Evaluation options pass through, here a stricter `pck_threshold`.
""")
    nb.code("""
pitch_models = {
    "YOLOv8n-pose (demo run)": pose_run.load(),
    "YOLOv8n-pose (models/)": tv.load_model("yolo_pose", MODELS / "pitch_yolov8n_pose.pt"),
}
pitch_metrics = {name: model.evaluate(pitch_data, pck_threshold=0.02) for name, model in pitch_models.items()}
pd.DataFrame({name: m.as_dict() for name, m in pitch_metrics.items()}).T.round(3)
""")
    nb.code("""
tv.viz.plot_metrics(pitch_metrics["YOLOv8n-pose (models/)"])
""")
    return nb
