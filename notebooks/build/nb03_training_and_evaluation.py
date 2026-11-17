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
| `lines` | the package (torch) | points on the painted pitch markings, for the homography |

All train with the same `train` call; the detectors are scored by the same
evaluation code, so their numbers compare directly.

The training runs here are **short demos**: a few epochs on a subset, enough
to show the interface in a few minutes. The checkpoints in `models/` were
trained on the full datasets for far longer (200 epochs for the YOLO models,
about 30 for RF-DETR, 20 for the line model), and they are the ones the other
notebooks use.

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
line_data = tv.load_pitch_lines(DATA / "SoccerNetCalibration").subset({"train": 400, "valid": 100}, seed=0)
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
One config serves the detector demo runs below: few epochs, everything written
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
The pitch line model learns heatmaps of the 20 SoccerNet markings from
SoccerNet's line labels: `tv.load_pitch_lines(calibration, gamestate)` reads
the Camera Calibration and Game State folders into a `lines` dataset (the
shipped label rule keeps the main-side and along-the-pitch training images).
It trains with the same call; a run folder that holds a checkpoint resumes
from it, and `manifest.json` records the recipe, the data and the versions.
The checkpoint in `models/` is the declared recipe (20 epochs, batch 12) on
the full set.
""")
    nb.code("""
%%capture lines_log
lines_run = tv.train("lines", line_data, epochs=2, batch_size=12, output_dir=config.output_dir, name="pitch_lines")
""")
    nb.code("""
print(lines_run.metrics)
tv.viz.plot_training(lines_run)
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
    "YOLO11l (models/)": tv.load_model("yolo", MODELS / "detector_yolo11l.pt"),
    "RF-DETR sample (models/)": tv.load_model("rfdetr", MODELS / "rfdetr_smoketest_sample.pth"),
}
detection_metrics = {name: tv.evaluate(model, detection) for name, model in detectors.items()}
pd.DataFrame({name: m.as_dict() for name, m in detection_metrics.items()}).T.round(3)
""")
    nb.md("""
`DetectionMetrics` also has a per-class table. The ball, small and often
blurred, is the hardest class.
""")
    nb.code("""
detection_metrics["RF-DETR sample (models/)"].per_class.round(3)
""")
    nb.md("""
`plot_metrics` draws one model's per-class AP, or compares several side by side
when given a dict.
""")
    nb.code("""
tv.viz.plot_metrics(detection_metrics)
""")
    return nb
