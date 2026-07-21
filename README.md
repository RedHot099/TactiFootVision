# TactiFoot Vision

TactiFoot Vision is a Python toolkit for football video analysis. The production
runtime uses a domain-oriented `src/` package with notebook-friendly APIs for
detection, tracking, pitch projection, team assignment, export and experiments.

## Install

```bash
uv sync --group dev
```

## Python API

```python
from tactifoot_vision.detection import RFDETRDetectionModel, YOLODetectionModel
from tactifoot_vision.tracking import ByteTrackTracker
from tactifoot_vision.pipeline import InferencePipeline

model = YOLODetectionModel.from_weights("models/yolo11m.pt")
detector = model.as_detector(confidence=0.3)
pipeline = InferencePipeline(detector=detector, tracker=ByteTrackTracker(frame_rate=25))

result = pipeline.run_video("input.mp4", max_frames=300)
result.to_csv("results/pipeline.csv")
result.to_mot("results/tracks.txt")
```

RF-DETR uses the same detector interface:

```python
from tactifoot_vision.detection import RFDETRDetectionModel

detector = RFDETRDetectionModel.from_weights("models/rfdetr_smoketest_sample.pth").as_detector()
```

## CLI

```bash
uv run tactifoot infer --config configs/pipeline/fake_bytetrack.yaml
uv run tactifoot track images --config configs/pipeline/yolo_bytetrack_smoke.yaml --max-frames 3
uv run tactifoot dataset convert soccernet-tracking --input data/soccernet/tracking/extracted/train --output results/coco_smoke --max-sequences 1
uv run tactifoot experiment detection-tracking --config configs/experiments/soccernet_detection_tracking.yaml
uv run tactifoot experiment team-classification --config configs/experiments/team_classification_smoke.yaml
```

## Detection Smoke

Run one-image detection without tracking:

```bash
uv run tactifoot detect image --config configs/pipeline/yolo_model_smoke.yaml --input data/soccernet_dummy/img1/frame_0001.jpg
```

Run opt-in model smoke tests:

```bash
uv run pytest tests/model/test_detection_model_smoke.py -m model -v
```

## Tracking Smoke

ByteTrack can be used directly from Python:

```python
from tactifoot_vision.detection import YOLODetectionModel
from tactifoot_vision.pipeline import InferencePipeline
from tactifoot_vision.tracking import ByteTrackTracker

detector = YOLODetectionModel.from_weights("models/yolo11m.pt").as_detector()
pipeline = InferencePipeline(detector=detector, tracker=ByteTrackTracker(frame_rate=25))
result = pipeline.run_video("data/soccernet_dummy/img1", max_frames=3)
```

SAM2 uses the same tracker contract:

```python
from tactifoot_vision.config import SAM2Config
from tactifoot_vision.tracking import SAM2Tracker

tracker = SAM2Tracker(
    SAM2Config(
        checkpoint="external/segment-anything-2-real-time/checkpoints/sam2.1_hiera_tiny.pt",
        model_config_path="external/segment-anything-2-real-time/sam2/configs/sam2.1/sam2.1_hiera_t.yaml",
        device="auto",
        max_side=768,
        max_objects=32,
    )
)
```

Run a tracking CLI smoke:

```bash
uv run tactifoot track images --config configs/pipeline/yolo_bytetrack_smoke.yaml --max-frames 3
```

Run the opt-in SAM2 model smoke:

```bash
uv run pytest tests/model/test_sam2_tracker_smoke.py -m "model and sam2" -v
```

BoTSORT is intentionally disabled until a stable production adapter is selected.

## Projection And Team Assignment

Pitch projection composes a keypoint detector and a homography projector:

```python
from tactifoot_vision.keypoints import YOLOPoseKeypointModel
from tactifoot_vision.projection import PitchProjector

keypoints = YOLOPoseKeypointModel.from_weights("models/yolo_pose.pt")
projector = PitchProjector(keypoint_detector=keypoints)
```

Team assignment uses the same Python/YAML config surface:

```python
from tactifoot_vision.config import TeamAssignmentConfig
from tactifoot_vision.team_assignment import TeamAssigner

assigner = TeamAssigner.from_config(TeamAssignmentConfig(clusters=2))
assigner.fit(crops)
team_ids = assigner.predict(crops)
```

Run tracking evaluation from exported predictions:

```bash
uv run tactifoot evaluate tracking --pred results/pipeline.csv --gt data/.../gt/gt.txt --output results/metrics.json
uv run tactifoot evaluate tracking --pred results/mot.txt --gt data/.../gt/gt.txt
```

Pipeline CSV predictions are auto-detected and shifted from 0-based frames to
MOT/SoccerNet 1-based frames when evaluated against MOT ground truth; pass
`--pred-frame-offset` to override that behavior.

StatsBomb360 support is limited to evaluation helpers for already-normalized
projection tables. Native StatsBomb360 export is not implemented; use pipeline
CSV for generic projection output.

Legacy scripts and configs are archived under `legacy/` for reference. New
development should target `src/tactifoot_vision`.
