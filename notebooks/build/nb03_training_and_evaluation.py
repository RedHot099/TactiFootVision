"""03_training_and_evaluation.ipynb: `tv.models` training and `tv.evaluation` metrics."""

from common import (
    DEMO_DETECTOR,
    DETECTOR_EPOCHS,
    DETECTOR_IMGSZ,
    DETECTOR_WEIGHTS,
    PITCH_TRAINING_DATASET,
    REPO_URL,
    RFDETR_GUARD,
    Notebook,
)

from tactifoot_vision.data.demo import ATTRIBUTION as DEMO_ATTRIBUTION


def notebook() -> Notebook:
    nb = Notebook("03_training_and_evaluation")
    nb.md(f"""
# Training and evaluation with `tv.models` and `tv.evaluation`

{nb.badge}

Three backends sit behind one `Model` interface:

| name | library | task |
|---|---|---|
| `yolo` | Ultralytics YOLO | player, goalkeeper, referee and ball boxes |
| `rfdetr` | RF-DETR | the same boxes, transformer detector |
| `yolo_pose` | Ultralytics YOLO-pose | the 32 pitch landmarks |

All three train with the same `train` call and are scored by the same
evaluation code, so their numbers compare directly.

It runs on open data, the same from a checkout and on Google Colab:

* the SoccerTrack v2 demo match from `tv.data.download_demo` (about 300 MB,
  CC BY 4.0), on which a player detector trains for a few minutes, as in the
  [overview]({REPO_URL}/notebooks/00_overview.ipynb);
* a pitch keypoint dataset rendered here from the package's synthetic pitch,
  on which a short pitch keypoint model trains.

Both runs are short demonstrations of the interface, not models for real
matches. **On Colab**, pick a GPU first (Runtime, Change runtime type, T4
GPU); the notebook then takes about ten minutes. The RF-DETR cells run only
where the `rfdetr` extra is installed, which Colab's runtime is not (see
below).

The demo data needs your own free Hugging Face token, as a Colab secret
named `HF_TOKEN` (or the environment variable of that name, or a saved
`hf auth login`). The [overview]({REPO_URL}/notebooks/00_overview.ipynb)
lists the three steps to get one.

Part of the [TactiFoot Vision notebooks]({REPO_URL}/notebooks/README.md);
datasets are introduced in [01_data]({REPO_URL}/notebooks/01_data.ipynb).
""")
    nb.setup(
        imports="""
import logging
import os
import warnings
from pathlib import Path

import pandas as pd

import tactifoot_vision as tv
"""
    )

    nb.md("""
## Demo frames

`download_demo` fetches the demo match once; a second call does nothing. Its
detection dataset holds frames of one static camera that sees the whole pitch
through a fisheye lens, 4096x1080 pixels, so a player is about 25x60 pixels.
The demo labels only the 22 players: the dataset declares the package's four
detection classes, and `goalkeeper`, `referee` and `ball` stay empty.
""")
    nb.code("""
demo = tv.data.download_demo(DATA / "demo")
dataset = tv.load_dataset(demo.detect_yaml)
print(dataset)
tv.viz.show_samples(dataset, "train", n=2, cols=1)
""")

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
A config can serve several runs. This one is used by the RF-DETR and pitch
keypoint runs below: everything goes under this notebook's output folder, and
`exist_ok=True` reuses the run folders when the notebook runs again instead of
numbering new ones.
""")
    nb.code("""
config = tv.TrainConfig(batch_size=16, output_dir=OUT / "runs", exist_ok=True)
""")

    nb.md(f"""
## Training: one call for every backend

`tv.train(name, dataset, config, weights=..., **overrides)` exports the dataset
in the format the backend needs (YOLO folder or COCO folder) into the run
folder, trains, and returns a `TrainResult`. Keyword overrides replace single
config fields, and without a config they are the whole of it.

The detector is the overview's: the COCO weights `{DETECTOR_WEIGHTS}` that
Ultralytics publishes, trained for {DETECTOR_EPOCHS} epochs on the demo frames
at {DETECTOR_IMGSZ} pixels on the long side, because the players are small;
about five minutes on a Colab T4. While it runs, one bar shows the epoch, the
training loss and the validation mAP50, and the loss and metric curves under
it are redrawn after every epoch (`progress=False` gives the backend's own
log instead).
""")
    nb.code(DEMO_DETECTOR, live_log=True)
    nb.md("""
A `TrainResult` holds the best checkpoint, the run folder, the per-epoch
history and the validation metrics of the best checkpoint. Metric keys are the
same for every backend.
""")
    nb.code("""
print("weights:", os.path.relpath(training.weights))
print("metrics:", {k: round(v, 3) for k, v in training.metrics.items()})
training.history.head(3)
""")
    nb.md("""
`plot_training` draws the curves of a finished run from its history: the
figure the progress bar showed, for a run trained with `progress=False`, by
another backend or loaded later.
""")
    nb.code("""
tv.viz.plot_training(training)
""")
    nb.md("""
A misspelt option or a backend name for a `TrainConfig` field fails at once,
before a run folder exists:
""")
    nb.code("""
detector = training.load()
for option in ({"epoch": 5}, {"batch": 8}):
    try:
        detector.train(dataset, config, **option)
    except ValueError as error:
        print("ValueError:", error)
""")

    nb.md("""
### RF-DETR

RF-DETR trains through the same interface. Its cells run only where the
`rfdetr` extra is installed: the extra pins numpy and transformers below the
versions Colab ships, so on Colab they print one line and skip, and a local
environment with the extra (`pip install "tactifoot-vision[rfdetr]"`, or
`uv sync --extra rfdetr` in a checkout) runs them. The saved outputs of this
notebook come from such a run.
""")
    nb.code(RFDETR_GUARD)
    nb.md("""
Here the model is created first and trained with `Model.train`, the method
form of the same call; afterwards the object holds the trained weights.
`weights=None` starts from the COCO-pretrained checkpoint, and
`grad_accum_steps` is an RF-DETR option. One epoch on a hundred demo frames
keeps it to a few minutes: enough to show the interface, far too short to
find players this small. RF-DETR's own log is long, so `%%capture` keeps it
out of the notebook (`rfdetr_log.show()` prints it).
""")
    nb.code("""
%%capture rfdetr_log
if has_rfdetr:
    # RF-DETR suggests optimize_for_inference() on its first prediction and torch warns
    # about a meshgrid argument inside RF-DETR; both are benign here.
    logging.getLogger("rfdetr.detr").addFilter(lambda record: "not optimized for inference" not in record.getMessage())
    warnings.filterwarnings("ignore", message="torch.meshgrid")
    rfdetr = tv.load_model("rfdetr", size="base")
    rfdetr_run = rfdetr.train(dataset.subset({"train": 100, "valid": 20}, seed=0), config,
                              epochs=1, batch_size=4, grad_accum_steps=4, name="rfdetr")
""")
    nb.code("""
if has_rfdetr:
    print("metrics:", {k: round(v, 3) for k, v in rfdetr_run.metrics.items()})
    display(tv.viz.plot_training(rfdetr_run))
""")

    nb.md(f"""
### A pitch keypoint model

The pitch keypoint model is a pose model: one `pitch` object per image whose
32 keypoints are the landmarks of `tv.SoccerPitch`. Its training data here is
rendered from the package's synthetic pitch, as in
[01_data]({REPO_URL}/notebooks/01_data.ipynb), larger and harder:
`broadcast_homographies` draws random views of a main broadcast camera,
`synthetic_pitch_clip` renders the markings through them with noise and
player-sized boxes in front of the lines, and `Dataset.from_clips(...,
task="pose")` projects the landmarks into every frame. The split counts
clips, so the test clip shares no view with training. Rendering takes
seconds on a CPU.

This is a demonstration of the API: a model that finds the pitch in real
footage needs real labelled frames, which the synthetic pitch does not
replace.
""")
    nb.code(f"""
{PITCH_TRAINING_DATASET.strip()}
print(pitch_data)
tv.viz.show_samples(pitch_data, "train", n=3)
""")
    nb.md("""
It trains with the same call and the shared config; the dataset's `flip_idx`
goes into the exported `data.yaml`, so Ultralytics' own flip augmentation
renames landmarks correctly. Thirty epochs from the COCO pose weights take a
few minutes on a T4.
""")
    nb.code(
        """
pose_run = tv.train("yolo_pose", pitch_data, config, weights="yolov8n-pose.pt", epochs=30, name="pitch_pose")
""",
        live_log=True,
    )

    nb.md("""
## Evaluation

`tv.evaluate(model, dataset, split="valid")` (or `model.evaluate(dataset)`)
picks the metrics from the model's task. Detectors get COCO-style mAP computed
by one shared implementation, with classes matched by name, so YOLO and
RF-DETR are scored identically. `TrainResult.load()` rebuilds a trained model.
""")
    nb.code("""
detectors = {"YOLO11s (overview recipe)": detector}
if has_rfdetr:
    detectors["RF-DETR base (one epoch)"] = rfdetr
detection_metrics = {name: tv.evaluate(model, dataset) for name, model in detectors.items()}
pd.DataFrame({name: m.as_dict() for name, m in detection_metrics.items()}).T.round(3)
""")
    nb.md("""
`DetectionMetrics` also has a per-class table. Here only `player` has
instances; on a dataset with all four classes the ball, small and often
blurred, is the hardest one.
""")
    nb.code("""
detection_metrics["YOLO11s (overview recipe)"].per_class.round(3)
""")
    nb.md("""
`plot_metrics` draws one model's per-class AP, or compares several side by side
when given a dict.
""")
    nb.code("""
tv.viz.plot_metrics(detection_metrics)
""")
    nb.md("""
Keypoint models get the mean pixel error of labelled landmarks, PCK (the
share of landmarks closer than a fraction of the image diagonal, 5% by
default) and the share of images where the model found the pitch at all. The
model is scored on the test clip, which took no part in training or in
picking the best epoch. A low confidence threshold keeps the uncertain
predictions of a short run in the score. Evaluation options pass through,
here a stricter `pck_threshold`.
""")
    nb.code("""
pitch_model = pose_run.load(conf=0.05)
pitch_metrics = {
    f"PCK at {threshold:.0%}": tv.evaluate(pitch_model, pitch_data, "test", pck_threshold=threshold)
    for threshold in (0.05, 0.02)
}
pd.DataFrame({name: m.as_dict() for name, m in pitch_metrics.items()}).T.round(3)
""")
    nb.md("""
For a keypoint model `plot_metrics` draws the mean error of each landmark
(the index into `tv.SoccerPitch().vertices`); a landmark the test clip never
shows has no bar.
""")
    nb.code("""
tv.viz.plot_metrics(pitch_metrics["PCK at 5%"])
""")

    nb.md(f"""
## Sources

{DEMO_ATTRIBUTION}

The pitch keypoint frames are rendered by `tactifoot_vision`.
""")
    return nb
