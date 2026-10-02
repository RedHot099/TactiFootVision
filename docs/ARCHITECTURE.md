# TactiFoot Vision architecture

`tactifoot_vision` is an importable Python package (think `cv2` or
`supervision`): one short import, discoverable submodules, and the same
interface across interchangeable backends.

```python
import tactifoot_vision as tv
```

## Functional areas → modules

| Area | Module | Main entry points |
|---|---|---|
| Data preparation | `tv.data` | `load_dataset`, `Dataset` (`subset`, `resplit`, `merge`, `summary`, `to_yolo`, `to_coco`), `VideoReader`, `extract_frames`, `load_statsbomb` |
| Data augmentation | `tv.augment` | `Compose`, `OneOf`, geometric and photometric transforms, `augment_dataset` |
| Model training | `tv.models` (+ `tv.train`) | `load_model`, `train`, `Model.train`, `TrainConfig`, `TrainResult` |
| Model inference | `tv.models`, `tv.tracking`, `tv.teams`, `tv.pitch`, `tv.pipeline` | `Model.predict`, `create_tracker`, `TeamClassifier`, `HomographyEstimator`, `Pipeline.run` → `PipelineResult` |
| Evaluation | `tv.evaluation` (+ `Model.evaluate`) | `evaluate`, `evaluate_detector`, `evaluate_keypoints`, `compare_with_statsbomb` |
| Results visualisation | `tv.viz` | `FrameAnnotator`, `PitchRadar`, `render_video`, `show`, plotting helpers |
| Configuration | `tv.config` | `PipelineConfig`, `load_config` (YAML), `Pipeline.from_config` |

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
  config.py           PipelineConfig (+ sub-configs), load_config
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
notebooks/            end-to-end pipeline notebook
examples/             one script per functional area (used to verify the notebook flow)
configs/              example YAML configs
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
  they created earlier (marked with `.tactifoot-export`).
* `VideoReader(path)`: `fps`, `width`, `height`, `frame_count`, `duration`,
  `frames(start=0, end=None, stride=1)` → `(index, frame)` pairs, `read(index)`.
* `extract_frames(video, out_dir, every=25, start=0, end=None, max_frames=None)` → image paths.
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
* `TrainResult.metrics` uses the same keys for every backend (`map50_95`, `map50`,
  plus `map75`, `precision`, `recall`, `pose_*` when reported);
  `TrainResult.history` is a per-epoch DataFrame (1-based `epoch`, backend
  column names); `TrainResult.load()` rebuilds the model.
* Unknown training options fail loudly (Ultralytics validates them itself;
  the RF-DETR backend checks them because RF-DETR would ignore them).

### Evaluation (`tv.evaluation`)

* `evaluate(model, dataset, split="valid")` dispatches on `model.task`:
  `evaluate_detector` → `DetectionMetrics` (mAP50-95, mAP50, mAP75, per-class AP
  table, via `supervision.metrics`), `evaluate_keypoints` → `KeypointMetrics`
  (mean pixel error, PCK at a fraction of the image diagonal).
  Same numbers for every backend, so YOLO and RF-DETR compare fairly.
* `compare_with_statsbomb(freeze_frames, statsbomb, period=1)` → merged table
  with the nearest detected object per StatsBomb object and its distance
  (`euclidean_distance`). Build the pipeline with `SoccerPitch(120, 80)` so both
  sides use StatsBomb units.

### Inference (`tv.tracking`, `tv.teams`, `tv.pipeline`)

* `ByteTrackTracker` (supervision ByteTrack), `SAM2Tracker` (segment-anything-2-real-time
  camera predictor with detector-driven re-seeding of new objects).
* `clean_ball_path(positions, max_jump)` drops ball positions further than
  `max_jump` pitch units per elapsed frame from the last accepted one
  (the pipeline's `ball_max_speed` is the same limit in units per second).
* `TeamClassifier(embedder="siglip" | "resnet" | Embedder, n_teams=2, reducer="umap" | None)`:
  `fit(crops)`, `predict(crops)`; `extract_crops(frame, boxes, scale=...)`.
  Fits on at most `max_fit_samples` embeddings and skips UMAP below 30 crops.
* `Pipeline(detector, keypoint_model=None, tracker="bytetrack", team_classifier=None, pitch=SoccerPitch(), ...)`
  `.run(video, start=0, max_frames=None, stride=1) -> PipelineResult`;
  `Pipeline.from_config(yaml_or_config)`.
  Teams are assigned after tracking by majority vote over each track's crops
  (a bounded random sample of `team_samples_per_track` crops per track), so
  every frame of a track carries the same team id. Goalkeepers join the team
  whose players are nearest on the pitch; referees get `NO_TEAM`.
* `HomographyEstimator` averages the last `smoothing_window` fits (3 by default:
  half the position jitter of no smoothing, ~2 px more line lag on pans) and
  restarts the average after a gap. Pitch vertices 10/11/18/19 sit where the
  penalty arc meets the box line, matching how the keypoint dataset labels them.

### Results visualisation (`tv.viz`)

* `FrameAnnotator(style="standard" | "video_game", ...)`: `annotate(frame, frame_result)`.
* `PitchRadar(pitch, ...)`: `draw(frame_result)` → top-down pitch image;
  `tv.viz.overlay(frame, image, position=..., width_fraction=..., alpha=...)` pastes it onto a frame.
* `render_video(result, source=None, output, annotator=None, radar=None)` → output path;
  checks that the source video and the drawing pitch match the result.
* `show(images, titles=None, cols=...)`, `show_samples(dataset, split, n)`,
  `show_augmentations(dataset, transform, n)`, `plot_training(train_result)`,
  `plot_heatmap(result, team=None)`, `plot_tracks(result)`, `plot_metrics(metrics)`,
  `plot_distance_histogram(distances)`.
