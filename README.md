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
result = pipeline.run("match.mp4", end=500)                       # inference
result.to_csv("outputs/tracks.csv")
tv.viz.render_video(result, "match.mp4", "outputs/annotated.mp4")        # visualisation
```

Executed walkthroughs live in [`notebooks/`](notebooks/README.md):
[`00_overview`](notebooks/00_overview.ipynb) runs the whole workflow, and one
notebook per area shows everything its module can do.

```bash
uv run --with jupyterlab jupyter lab notebooks/
```

(or open them in VS Code with the project's `.venv` kernel).

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
| CLI mode | `tv.run_file`, `tactifoot` | run files (`configs/*.yaml`) and the command line |

Backends are pluggable: models, trackers and team embedders are looked up by
name in registries, so `tv.load_model("rfdetr", ...)` and
`tv.load_model("yolo", ...)` return objects with the same methods, and a run
file can switch between them by name. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
for conventions and the design.

## Two modes: notebook and command line

The package is used in two independent ways
([ADR 0001](docs/adr/0001-notebook-and-cli-are-independent-front-ends.md)):

* **Notebook mode**: plain Python, as above. Build the objects you want and
  call them; there is no config object.
* **CLI mode**: the `tactifoot` command. It translates its arguments into the
  same calls and declares no defaults of its own, so an option you leave out
  keeps the package default.

```bash
tactifoot run configs/pipeline.yaml --video data/videos/match.mp4 --output-dir outputs/match \
    --end 500 --set detector.conf=0.4 --set render.annotator.style=video_game
tactifoot train yolo --data data/datasets/football_yolo --weights yolo11n.pt --epochs 50 --set mosaic=0.0
tactifoot evaluate rfdetr --weights models/football_rfdetr_base.pth --data data/datasets/football_yolo --set max_images=50
tactifoot info
```

`run` writes the **run folder**: `result.pkl`, `tracks.csv`, StatsBomb-style
`freeze_frames.csv` (all three via `PipelineResult.export`) and
`annotated.mp4` (skip it with `--no-video`). Its flags are the run inputs:
`--start`, `--end`, `--stride` (`Pipeline.run`) and `--period`,
`--period-start` (`PipelineResult.export`). `train` has one flag per
`TrainConfig` field (`tactifoot train --help` lists them with their defaults);
backend options go through `--set`. `evaluate` passes `--split` and `--set`
keys to `model.evaluate`.

### Run files

A run file says how to process video: models, tracker, teams, homography and
rendering. Each section is the keyword arguments of one package call, and
backend options sit next to `type`:

```yaml
detector: {type: yolo, weights: ../models/football_yolo11m.pt, conf: 0.3, iou: 0.5}   # tv.load_model
keypoints: {type: yolo_pose, weights: ../models/pitch_yolov8n_pose.pt}                 # null: none
tracker: {type: bytetrack, lost_track_buffer: 30}     # tv.tracking.create_tracker; null: none
teams: {embedder: siglip}                             # tv.teams.TeamClassifier; null: none
pitch: {length: 120, width: 80}                       # tv.SoccerPitch
homography: {smoothing_window: 5}                     # tv.pitch.HomographyEstimator
pipeline: {include_classes: [player, goalkeeper]}     # remaining tv.Pipeline arguments
render:                                               # tv.viz.render_video arguments
  annotator: {style: video_game}                      # tv.viz.FrameAnnotator
  radar: {draw_ids: true}                             # tv.viz.PitchRadar; null: none
```

Keys are checked against the signatures when the file is loaded (a typo fails
with the list of valid keys); values are checked by the constructors before the
first frame. Paths starting with `./` or `../` are relative to the run file.
`--set dotted.key=value` overrides one value for one run; the value is YAML
(`null`, `true`, `0.4`, `[a, b]`) and missing keys are created. In Python,
`tv.run_file.load("configs/pipeline.yaml", ["tracker=null"]).build_pipeline()`
does the same as the CLI. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md#run-files-and-the-cli-tvrun_file-tactifoot)
for every section and [`examples/06_cli.sh`](examples/06_cli.sh) for a runnable tour.

## Local data layout

Datasets, weights and videos are not tracked by git. The notebooks, configs and
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
uv run ruff check src tests examples notebooks/build && uv run ruff format --check src tests examples notebooks/build
uv run python notebooks/build/build.py --execute   # regenerate and run the notebooks
```

The `examples/` scripts run each functional area on the local data. The
notebooks are generated by `notebooks/build/`; edit the generator modules,
not the `.ipynb` files.
