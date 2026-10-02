# TactiFoot Vision

Football video analysis as an importable Python package: prepare and augment
annotated datasets, train detection and pitch-keypoint models, run a tracking +
pitch-projection + team-classification pipeline on match video, and visualise
the results. The whole pipeline takes a few lines:

```python
import tactifoot_vision as tv

ds = tv.load_dataset("data/datasets/football_yolo")                     # data preparation
aug = tv.augment.Compose([tv.augment.HorizontalFlip(), tv.augment.ColorJitter()])
train_ds = tv.augment.augment_dataset(ds, aug, "outputs/aug")            # augmentation
run = tv.train("yolo", train_ds, weights="yolo11n.pt", epochs=30)        # training
print(run.load().evaluate(ds))                                           # evaluation

pipeline = tv.Pipeline(
    detector=run.load(conf=0.3),
    keypoint_model=tv.load_model("yolo_pose", "models/pitch_yolov8n_pose.pt"),
    team_classifier=tv.teams.TeamClassifier("siglip"),
)
result = pipeline.run("match.mp4", max_frames=500)                       # inference
result.to_csv("outputs/tracks.csv")
tv.viz.render_video(result, "match.mp4", "outputs/annotated.mp4")        # visualisation
```

The end-to-end walkthrough lives in
[`notebooks/tactifoot_pipeline.ipynb`](notebooks/tactifoot_pipeline.ipynb):

```bash
uv run --with jupyterlab jupyter lab notebooks/tactifoot_pipeline.ipynb
```

(or open it in VS Code with the project's `.venv` kernel).

## Install

```bash
uv sync                      # creates .venv with the package installed in editable mode
uv run python -c "import tactifoot_vision as tv; print(tv.__version__)"
```

or `pip install -e .` into any Python 3.12+ environment. The `tactifoot`
command is installed alongside (`tactifoot --help`). The SAM2 tracker needs
`uv sync --extra sam2` plus the SAM2 repository (see below).

## Modules

| Area | Module | What you get |
|---|---|---|
| Data preparation | `tv.data` | `load_dataset` (YOLO detection/pose, COCO), `Dataset.subset/resplit/merge/summary`, `to_yolo`/`to_coco` conversion, `VideoReader`, `extract_frames`, `load_statsbomb` |
| Augmentation | `tv.augment` | `Compose`, `OneOf`, `HorizontalFlip` (keypoint-aware), `RandomAffine`, `ColorJitter`, blur/noise/JPEG transforms, `augment_dataset` |
| Training | `tv.models`, `tv.train` | one `Model` interface for `yolo`, `yolo_pose` and `rfdetr`: `load_model`, `train`, `predict`, `evaluate` |
| Evaluation | `tv.evaluation` | backend-independent mAP and keypoint metrics, StatsBomb 360 comparison |
| Inference | `tv.tracking`, `tv.teams`, `tv.pitch`, `tv.pipeline` | ByteTrack / SAM2 trackers, SigLIP / ResNet team clustering, homography to pitch coordinates, `Pipeline.run` → `PipelineResult` |
| Visualisation | `tv.viz` | frame annotation, 2D pitch radar and overlay, video rendering, notebook plots |
| Configuration | `tv.config` | YAML pipeline configs (`configs/*.yaml`) |

Backends are pluggable: models, trackers and team embedders are looked up by
name in registries, so `tv.load_model("rfdetr", ...)` and
`tv.load_model("yolo", ...)` return objects with the same methods, and a YAML
config can switch between them. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
for conventions and the design.

## Command line

```bash
tactifoot run --config configs/pipeline.yaml --video data/videos/match.mp4 --output-dir outputs/match
tactifoot train yolo --data data/datasets/football_yolo --weights yolo11n.pt --epochs 50
tactifoot evaluate rfdetr --weights models/football_rfdetr_base.pth --data data/datasets/football_yolo
tactifoot info
```

`run` writes `result.pkl`, `tracks.csv`, StatsBomb-style `freeze_frames.csv`
and `annotated.mp4`.

## Local data layout

Datasets, weights and videos are not tracked by git. The notebook, configs and
examples expect them under the repository root:

```
data/datasets/football_yolo/      YOLO detection dataset (ball, goalkeeper, player, referee)
data/keypoints/                   YOLO-pose pitch dataset (32 landmarks, flip_idx)
data/statsbomb/                   StatsBomb 360 *_events.json + *_360.json
data/videos/*.mp4                 match footage
models/football_yolo11m.pt        trained detector
models/pitch_yolov8n_pose.pt      trained pitch keypoint model
models/football_rfdetr_base.pth   trained RF-DETR detector
external/segment-anything-2-real-time/   optional, for the SAM2 tracker
```

Symlinks work fine for all of these. Pretrained starting weights for training
(`yolo11n.pt`, `yolov8n-pose.pt`, RF-DETR's COCO checkpoints) are downloaded on
first use to `~/.cache/tactifoot_vision/`.

## Development

```bash
uv run pytest                 # fast tests on synthetic data
uv run pytest -m model        # tests that load real weights / use the GPU
uv run ruff check src tests && uv run ruff format --check src tests
```

The `examples/` scripts run each functional area on the local data; the notebook
chains them.
