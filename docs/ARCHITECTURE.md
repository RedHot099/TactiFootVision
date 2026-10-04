# TactiFoot Vision architecture

`tactifoot_vision` is an importable Python package (think `cv2` or
`supervision`): one short import, discoverable submodules, and the same
interface across interchangeable backends.

```python
import tactifoot_vision as tv
```

## Two modes, one core

The package has two independent front-ends
([ADR 0001](adr/0001-notebook-and-cli-are-independent-front-ends.md)):

* **Notebook mode**: plain Python imports. No run file and no config object
  sit between the user and the classes.
* **CLI mode**: the `tactifoot` command. `cli.py` and `run_file.py` only
  translate text into calls of the same functions and classes.

Only the constructors, functions and `TrainConfig` hold default values and
validate values. The CLI declares no defaults (`argparse.SUPPRESS`; a missing
option is not passed), `tactifoot train` generates its flags from
`TrainConfig.model_fields`, the `run` flags come from the signatures of
`Pipeline.run` and `PipelineResult.export`, and run file keys are checked
against the target signatures. A new field, parameter or changed default
reaches both modes without editing `cli.py` or `run_file.py`.

## Functional areas → modules

| Area | Module | Main entry points |
|---|---|---|
| Data preparation | `tv.data` | `load_dataset`, `Dataset` (`subset`, `resplit`, `merge`, `with_split`, `summary`, `to_yolo`, `to_coco`), `VideoReader`, `extract_frames`, `load_statsbomb` |
| Data augmentation | `tv.augment` | `Compose`, `OneOf`, geometric and photometric transforms, `augment_dataset` |
| Model training | `tv.models` (+ `tv.train`) | `load_model`, `train`, `Model.train`, `TrainConfig`, `TrainResult` |
| Model inference | `tv.models`, `tv.tracking`, `tv.teams`, `tv.pitch`, `tv.pipeline` | `Model.predict`, `create_tracker`, `TeamClassifier`, `HomographyEstimator`, `Pipeline.run` → `PipelineResult` |
| Evaluation | `tv.evaluation` (+ `Model.evaluate`) | `evaluate`, `evaluate_detector`, `evaluate_keypoints`, `compare_with_statsbomb` |
| Results visualisation | `tv.viz` | `FrameAnnotator`, `PitchRadar`, `render_video`, `draw_annotations`, `show`, plotting helpers |
| CLI mode | `tv.run_file`, `tactifoot` | `run_file.load` → `RunFile` (`sections`, `run`, `build_pipeline`, `render_video`); `tactifoot run/train/evaluate/info` |

Swappable component families sit behind one abstract base class and a
name registry (`tactifoot_vision/registry.py`):

| Family | Base class | Registry | Backends |
|---|---|---|---|
| Models | `models.base.Model` | `MODELS` | `yolo` (detection), `yolo_pose` (pitch keypoints), `rfdetr` (detection) |
| Trackers | `tracking.base.Tracker` | `TRACKERS` | `bytetrack`, `sam2` |
| Team embedders | `teams.embedders.Embedder` | `EMBEDDERS` | `resnet`, `siglip` |

Adding a backend = subclass + `@REGISTRY.register("name")`; nothing else changes.

## Layout

```
src/tactifoot_vision/
  __init__.py         lazy submodules + flat shortcuts (tv.load_dataset, tv.train, tv.Pipeline, ...)
  registry.py         Registry[T]
  utils.py            setup_logging, resolve_device, ensure_dir
  run_file.py         run files: load (+ overrides, key checks), RunFile.run / build_pipeline / render_video
  cli.py              `tactifoot` command (run / train / evaluate / info)
  data/               annotations.py (Task, Annotations), dataset.py (Sample, Dataset),
                      yolo.py, coco.py (format IO), video.py, statsbomb.py
  augment/            base.py (Transform, Compose, OneOf), geometric.py, photometric.py,
                      dataset.py (augment_dataset)
  models/             base.py (Model, TrainConfig, TrainResult, MODELS, load_model, train),
                      ultralytics.py (YOLODetector, YOLOPoseModel), rfdetr.py (RFDETRDetector)
  evaluation/         detection.py, keypoints.py, statsbomb.py, __init__ (evaluate dispatcher)
  pitch/              pitch.py (SoccerPitch), homography.py (HomographyEstimator)
  tracking/           base.py (Tracker, TRACKERS), bytetrack.py, sam2.py, ball.py
  teams/              crops.py, embedders.py, classifier.py
  pipeline/           pipeline.py (Pipeline), result.py (FrameResult, PipelineResult)
  viz/                annotate.py, radar.py, video.py, plots.py
tests/                pytest; model/GPU tests are marked `model` and skipped by default
notebooks/            executed notebooks: overview + one per functional area
  build/              their generator (one module per notebook, build.py entry point)
examples/             one script per functional area
configs/              example run files
```

## Conventions

* **Images** are `np.ndarray` `uint8` BGR `H×W×3` (OpenCV convention) everywhere.
  Convert to RGB only at the matplotlib boundary (`viz.show`).
* **Boxes** are `xyxy` float pixels. **Detections** are `sv.Detections`;
  every producer fills `data["class_name"]` so consumers never rely on class-id maps.
* **Keypoints** from models are `sv.KeyPoints` (`xy (M,K,2)`, `confidence (M,K)`),
  instances sorted by confidence. Dataset keypoints are `(N,K,3)` arrays in
  `Annotations` with YOLO/COCO visibility (0/1/2).
* **Pitch coordinates** start in a corner, with `x` along the length and `y`
  across. The default pitch is 105×68 m; `SoccerPitch(120, 80)` gives StatsBomb units.
* **Heavy imports** (`torch`, `ultralytics`, `rfdetr`, `transformers`, `umap`,
  `matplotlib`) happen inside functions/methods, never at module top level of a
  subpackage `__init__`. Importing `tactifoot_vision.models` must not import
  ultralytics or rfdetr.
* **Logging** goes through `logging.getLogger(__name__)`, never `print`. Users
  opt in with `tv.setup_logging("INFO")`.
* **Errors** are `ValueError` / `FileNotFoundError` / `RuntimeError` with a
  message that says what to change. Don't swallow exceptions or return `None`
  for failures the caller cannot see.
* **Outputs** go to an explicit path argument, and the function returns the path it wrote.
* **Style** is Python ≥ 3.12 (`X | None`, `list[int]`, PEP 695 generics), no
  `from __future__ import annotations`, clean `ruff check` + `ruff format`,
  docstrings on public API, comments only where the *why* is not obvious.
* **Tests** run on synthetic data by default. Anything needing real weights,
  downloads or a GPU gets `@pytest.mark.model`.

## Contracts (read the code, it is the spec)

* `data/annotations.py`: `Task`, `Annotations` (pixel boxes, class ids, keypoints, flip_idx)
* `data/dataset.py`: `Sample`, `Dataset` (splits `train`/`valid`/`test`)
* `models/base.py`: `Model`, `TrainConfig`, `TrainResult`, `MODELS`, `load_model`, `train`
* `tracking/base.py`: `Tracker`, `TRACKERS`, `create_tracker`
* `pitch/`: `SoccerPitch` (32 landmark vertices in dataset order), `HomographyEstimator`
* `pipeline/result.py`: `FrameResult`, `PipelineResult` (tables, freeze frames, save/load)

## Module specifications

### Data preparation (`tv.data`)

* `load_dataset(path)` / `Dataset.load`: YOLO `data.yaml` (detection or pose,
  picked by `kpt_shape`) or a Roboflow/RF-DETR COCO folder.
* YOLO IO follows Ultralytics path rules (`path:` root, `../` prefixes of
  Roboflow exports, `images/` ↔ `labels/`); detection labels given as polygons
  become their bounding boxes. Writing clips boxes to the image and writes
  `0 0 0` for unlabelled or out-of-frame keypoints (Ultralytics rejects
  out-of-range values).
* COCO IO writes category ids `0..C-1` in class order with a non-`"none"`
  supercategory (RF-DETR drops `"none"` categories from its class names);
  image and annotation ids start at 1 (pycocotools treats id 0 as "no match").
* Exports link images (`symlink` default, `hardlink`, `copy`) and write labels
  fresh, so subsets, merges and augmented datasets export the same way.
  Images sharing a stem inside a split get a numeric suffix (labels are keyed
  by stem).
* Writers never touch the source: they refuse an output folder that contains
  source images or lies inside a source image folder, and only clear folders
  they created earlier (marked with `.tactifoot-export`). Every argument (the
  link mode, the source images' existence) is checked before anything is deleted.
* The marker records the folders an export manages and every file it writes,
  and a re-export deletes exactly those: the previous export's split folders
  (a split the new export drops disappears with them, so it is not loaded
  back) and files such as `data.yaml` (switching between YOLO and COCO leaves
  no trace of the old format). A recorded folder holding any file the marker
  does not list (augmented images, a user's notes) is refused with the file's
  name, so re-exporting into the same folder works for every link mode and
  loses nothing else.
* `Dataset.with_split(split, samples)` returns a copy with one split replaced.
* `VideoReader(path)`: `fps`, `width`, `height`, `frame_count`, `duration`,
  `frames(start=0, end=None, stride=1)` → `(index, frame)` pairs, `read(index)`.
  `frames` checks the range when called (`start >= 0`, `end >= start`,
  `stride >= 1`); `read` refuses negative indices.
* `extract_frames(video, out_dir, start=0, end=None, stride=25)` → image paths.
  Frame ranges use `start`, `end` (exclusive, `None` = to the end) and
  `stride` everywhere: `VideoReader.frames`, `extract_frames`, `Pipeline.run`.
* `Dataset.resplit(train, valid, test)` rounds the split boundaries
  cumulatively and gives the remainder to the last non-zero split, so a zero
  fraction stays empty. `subset` takes integer counts (NumPy integers too) or
  fractions in `[0, 1]`.
* The YOLO and COCO readers refuse class ids outside the dataset's classes,
  naming the label or annotation file.
* `load_statsbomb(path)` → DataFrame of StatsBomb 360 freeze-frame objects merged with event timing.

### Data augmentation (`tv.augment`)

* `Transform(p=1.0)`: `__call__(image, annotations, rng=None) -> (image, annotations)`;
  applies itself with probability `p`. Subclasses implement `apply`.
* `Compose(transforms)`, `OneOf(transforms)`.
* Geometric (move boxes and keypoints consistently): `HorizontalFlip`
  (uses `annotations.flip_idx`, errors if keypoints lack it), `RandomAffine`
  (rotation, scale, translation; boxes from transformed corners, clipped;
  objects with too little area left are dropped; keypoints leaving the frame
  become visibility 0).
* Photometric (pixels only): `ColorJitter`, `GaussianBlur`, `MotionBlur`,
  `GaussianNoise`, `JpegCompression`.
* `augment_dataset(dataset, transform, out_dir, copies=1, splits=("train",), seed=0)`
  → new `Dataset` with originals plus `copies` augmented variants per image
  written to `out_dir`; other splits untouched (never augment validation data).
  `out_dir` must not be a `to_yolo` / `to_coco` export folder, which a later
  export would clear.

### Training and inference behind one interface (`tv.models`)

* `load_model(name, weights, conf=..., device=..., **options)`; `available_models()`.
* `Model.predict(image)` / `model(image)`; `Model.train(dataset, config=None, **overrides) -> TrainResult`;
  `Model.evaluate(dataset, split="valid")`; `train(name_or_model, dataset, weights=..., **overrides)`.
* `yolo` / `yolo_pose`: Ultralytics; training exports the dataset with `to_yolo`.
  `yolo_pose.predict` returns `sv.KeyPoints` of all instances sorted by confidence.
* `rfdetr`: RF-DETR (`size` option: nano/small/medium/base/large); training
  exports with `to_coco`; class names come from the checkpoint.
* `Model.train` picks the run folder (`output_dir/name`, then `name2`, ... unless
  `exist_ok=True`) and hands it to the backend, which exports the dataset into it.
  An `exist_ok=True` re-run deletes the previous run's `results.csv` /
  `log.txt` first, so history and metrics describe that run only.
* `TrainConfig.backend_options` holds the keyword arguments that are not
  `TrainConfig` fields; they go verbatim to the backend.
* `TrainResult.metrics` uses the same keys for every backend (`map50_95`, `map50`,
  plus `map75`, `precision`, `recall`, `pose_*` when reported);
  `TrainResult.history` is a per-epoch DataFrame (1-based `epoch`, backend
  column names); `TrainResult.load()` rebuilds the model.
* Backend options are checked before the run folder is created
  (`Model._check_options`: Ultralytics' known arguments, RF-DETR's train
  config and device), so a typo costs no run folder. Options that would move
  the run folder (`output_dir`, `project`, `name`, `dataset_dir`, `data`) are
  refused, and so are backend names for `TrainConfig` fields (Ultralytics
  `batch`, `lr0`; RF-DETR `num_workers`, `resolution`,
  `early_stopping_patience`), with the field to use instead.
* `train("rfdetr", ..., device=...)` loads the model on the training device.
  A bare Ultralytics checkpoint name is downloaded only if Ultralytics
  publishes it; any other missing file is "Checkpoint not found".

### Evaluation (`tv.evaluation`)

* `evaluate(model, dataset, split="valid")` dispatches on `model.task`:
  `evaluate_detector` → `DetectionMetrics` (mAP50-95, mAP50, mAP75 and the
  per-class table `per_class`, via `supervision.metrics`), `evaluate_keypoints` → `KeypointMetrics`
  (mean pixel error, PCK at a fraction of the image diagonal).
  Same numbers for every backend, so YOLO and RF-DETR compare fairly.
* `compare_with_statsbomb(freeze_frames, statsbomb, period=1)` → merged table
  with the nearest detected player or goalkeeper per StatsBomb object and its
  distance (`euclidean_distance`); the ball, referees and other classes never
  match. The match comes from the processed frame closest in time to the
  event (`detected_frame_id`) when both tables carry `timestamp_seconds`;
  otherwise it is the minimum over every frame of that second. Build the
  pipeline with `SoccerPitch(120, 80)` so both sides use StatsBomb units. A
  `PipelineResult` is accepted too and goes through `to_freeze_frames(period)`,
  like the exported CSV, so both routes agree.
* `evaluate` refuses options its evaluator does not take.

### Inference (`tv.tracking`, `tv.teams`, `tv.pipeline`)

* `ByteTrackTracker` (supervision ByteTrack), `SAM2Tracker` (segment-anything-2-real-time
  camera predictor with detector-driven re-seeding of new objects). Detections
  overlapping a tracked box never become new tracks; a track whose mask is
  empty on a re-seed frame loses its id.
* `clean_ball_path(positions, max_jump)` drops ball positions further than
  `max_jump` pitch units per elapsed frame from the last accepted one
  (the pipeline's `ball_max_speed` is the same limit in units per second).
* `TeamClassifier(embedder="siglip" | "resnet" | Embedder, n_teams=2, reducer="umap" | None)`:
  `fit(crops)`, `predict(crops)`, `fit_predict(crops)`; `extract_crops(frame, boxes, scale=...)`.
  Fits on at most `max_fit_samples` embeddings and skips UMAP below 30 crops.
* `Pipeline(detector, keypoint_model=None, tracker="bytetrack", team_classifier=None, pitch=SoccerPitch(), ...)`
  `.run(video, start=0, end=None, stride=1) -> PipelineResult`. The range is
  checked before any model runs; an empty range is an error.
  `PipelineResult.export(out_dir, *, period=1, period_start=None)` writes the
  run folder's data files (`result.pkl`, `tracks.csv`, `freeze_frames.csv`,
  with headers even for an empty result).
  `to_freeze_frames(period, period_start=None)`: `period_start=None` is the
  period's kick-off on StatsBomb's clock (0', 45', 90', 105'); `minute` /
  `second` follow that match clock, `timestamp` is that match clock too
  (StatsBomb's event timestamp restarts every period), and `visible_area` is
  StatsBomb's flat `[x1, y1, x2, y2, ...]` polygon (`None` when part of the
  frame lies beyond the horizon).
  With `keep_masks=True` the tracker's segmentation masks (SAM2) are kept in
  `FrameResult.masks` as `ObjectMasks`: each mask cropped to the pixels it
  covers, so memory grows with the people's area, not the frame area.
  Teams are assigned after tracking by majority vote over each track's crops
  (a bounded random sample of `team_samples_per_track` crops per track), so
  every frame of a track carries the same team id. Goalkeepers join the team
  whose players are nearest on the pitch; referees get `NO_TEAM`.
* `HomographyEstimator` averages the last `smoothing_window` fits (3 by default:
  half the position jitter of no smoothing, ~2 px more line lag on pans) and
  restarts the average after a gap. `used_indices` lists the confident
  landmarks given to RANSAC in the current frame's fit (not only its
  inliers), `None` when the frame gave no fit. Pitch
  vertices 10/11/18/19 sit where the penalty arc meets the box line, matching
  how the keypoint dataset labels them. Circles and arcs use
  `SoccerPitch.circle_radii` (9.15 m scaled along each axis), the same
  ellipse the radar draws, so on any `length × width` those vertices lie
  exactly on the drawn arcs.

### Results visualisation (`tv.viz`)

* `FrameAnnotator(style="standard" | "video_game", ...)`: `annotate(frame, frame_result)`.
  `draw_masks=True` draws `FrameResult.masks` and warns once when a result has none.
* `PitchRadar(pitch, ...)`: `draw(frame_result)` → top-down pitch image;
  `draw_points(xy, colors=None, radius=None, image=None)` draws arbitrary pitch
  points (e.g. StatsBomb positions);
  `tv.viz.overlay(frame, image, position=..., width_fraction=..., alpha=...)` pastes it onto a frame;
  `tv.viz.check_overlay(...)` checks those settings without a frame.
* Colours (`ColorLike`) are hex strings, `sv.Color`s or BGR tuples or lists
  (what YAML gives).
* `draw_annotations(image, annotations, class_names=None)` draws dataset ground truth.
* `render_video(result, source, output, annotator=None, radar=None,
  overlay_position="bottom-center", overlay_width_fraction=0.25,
  overlay_alpha=0.8, overlay_padding=10, fps=None, progress=True)` → output
  path; `source=None` uses `result.video_path`. Checks its settings and
  refuses an `output` that is the source before opening anything, checks that
  the source video (header and decoded frame size) and the drawing pitch
  match the result, raises `RuntimeError` when the source lacks a result
  frame, and deletes the partial output when rendering fails.
* `show(images, titles=None, cols=...)`, `show_samples(dataset, split, n)`,
  `show_augmentations(dataset, transform, n)`, `plot_training(train_result)`,
  `plot_heatmap(result, team=None, smoothing=1.5)` (0: no blur), `plot_tracks(result)`, `plot_metrics(metrics)`,
  `plot_distance_histogram(distances)`.

### Run files and the CLI (`tv.run_file`, `tactifoot`)

A **run file** is YAML that says how to process video; it holds no run inputs
(video, output folder, frame range, period), so one file serves many matches.
Each section maps onto exactly one package call:

| Section | Call | Notes |
|---|---|---|
| `detector` (required) | `tv.models.load_model(type, weights, **rest)` | keys checked against `MODELS[type]` |
| `keypoints` | same as `detector` | absent or `null`: no keypoint model |
| `tracker` | `tv.tracking.create_tracker(type, **rest)` | absent: `Pipeline`'s default; `null`: no tracking |
| `teams` | `TeamClassifier(**section)` | absent or `null`: no teams; `{}`: defaults; keys checked against `TeamClassifier` plus the chosen embedder's constructor |
| `pitch` | `SoccerPitch(**section)` | |
| `homography` | `HomographyEstimator(pitch, **section)` | gets the pitch built from `pitch` |
| `pipeline` | `Pipeline(**section)` | the scalar arguments (`include_classes`, `keep_masks`, ...) |
| `render.annotator` | `FrameAnnotator(pitch=result.pitch, **section)` | |
| `render.radar` | `PitchRadar(pitch=result.pitch, **section)` | `null`: no radar |
| other `render` keys | `tv.viz.render_video(...)` keyword arguments | `overlay_position`, `fps`, ... |

Backend options sit flat next to `type`:

```yaml
detector: {type: rfdetr, weights: ../models/football_rfdetr_base.pth, conf: 0.5, size: base}
tracker: {type: bytetrack, lost_track_buffer: 30}
render: {annotator: {style: video_game}, radar: null}
```

* `tv.run_file.load(path, overrides=())` checks every key against the target's
  signature and fails with the section, the bad key and the valid keys (an
  unknown section, `type` or embedder fails too; malformed YAML is a one-line
  `ValueError`). Loading imports no torch, ultralytics, rfdetr, transformers
  or sam2.
* `RunFile.run(video, output_dir, *, render=True, **run_inputs)` is one run:
  it builds the pipeline and the render objects (constructors and
  `render_video` settings check every value before the first frame), runs
  the pipeline, exports the data files and renders `annotated.mp4` unless
  `render=False`. `run_inputs` are the `Pipeline.run` and
  `PipelineResult.export` arguments; it returns the `PipelineResult`.
  `RunFile.render_video(result, source, output)` re-renders a saved result.
* A key left out is not passed, so the package default applies.
* Paths: string values starting with `./` or `../` (at any depth, lists
  included) are relative to the run file. Absolute paths and bare names such
  as `yolo11n.pt` pass through.
* **Overrides** `dotted.key=value` replace one value for one run. The value is
  YAML (`null`, `true`, `0.4`, `[player, goalkeeper]`), missing mappings are
  created, and `./` / `../` paths in the value are relative to the current
  directory. Overrides are applied before the key check.

```
tactifoot [--log-level LEVEL] run RUN_FILE --video V --output-dir D
          [--start N] [--end N] [--stride N] [--period N] [--period-start S]
          [--no-video] [--set KEY=VALUE ...]
tactifoot train MODEL --data D [--weights W] [--epochs N --batch-size N ... (one flag per TrainConfig field)]
          [--set KEY=VALUE ...]
tactifoot evaluate MODEL --weights W --data D [--split S] [--set KEY=VALUE ...]
tactifoot info
```

* `run`: `run_file.load(run_file, overrides).run(video, output_dir,
  render=not --no-video, **run inputs)`. User errors (bad values, malformed
  YAML, a missing optional dependency) are reported in one line;
  `--log-level DEBUG` keeps the traceback.
* `train`: `--set` keys are `TrainConfig` fields or backend options
  (`mosaic=0.0`, `grad_accum_steps=4`); setting a key by flag and `--set` is an error.
* `evaluate`: `--set model.KEY=VALUE` goes to `load_model` (`model.device=cuda:1`,
  `model.imgsz=1280`, RF-DETR `model.resolution=...`), `--split` and the other
  `--set` keys to `model.evaluate`.
