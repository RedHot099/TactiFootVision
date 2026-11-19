"""Model inference: single-frame predictions and the full video pipeline.

Run from the repository root: ``uv run python examples/04_inference.py``
"""

import logging
from collections import Counter
from pathlib import Path

import supervision as sv

import tactifoot_vision as tv
from tactifoot_vision.enums import (
    DetectionKey,
    EmbedderType,
    ModelType,
    Resize,
    TrackerType,
)

OUT = Path("outputs/examples/04_inference")
VIDEO = Path("data/videos/broadcast_60s.mp4")
tv.setup_logging(logging.INFO)

video = tv.VideoReader(VIDEO)
frame = video.read(250)

# Every detector answers the same call with sv.Detections + class names.
detectors = {
    ModelType.YOLO: tv.load_model(
        ModelType.YOLO, "models/detector_yolo11l.pt", conf=0.3
    ),
    ModelType.RFDETR: tv.load_model(
        ModelType.RFDETR, "models/rfdetr_smoketest_sample.pth", conf=0.5
    ),
}
for name, model in detectors.items():
    detections = model(frame)
    print(
        f"{name:7s} {len(detections):3d} objects",
        dict(Counter(detections.data[DetectionKey.CLASS_NAME])),
    )

# Pitch keypoints -> homography -> pitch coordinates of the players' feet.
# This pitch model learnt on frames padded to a square, so it sees them that
# way here; the estimator's settings are the ones it was tuned with (the line
# refinement leaves out the people's boxes).
pitch_model = tv.load_model(
    ModelType.YOLO_POSE, "models/pitch_yolo11m_pose.pt", conf=0.1, resize=Resize.SQUARE
)
keypoints = pitch_model(frame)
people = detectors[ModelType.YOLO](frame)
estimator = tv.pitch.HomographyEstimator(
    min_confidence=0.3, ransac_threshold=2.0, refine_with_lines=True
)
homography = estimator.update(keypoints, frame, people.xyxy)
feet = people.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
print(
    "first players on the pitch (m):",
    tv.pitch.frame_to_pitch(feet, homography)[:3].round(1),
)

# The full pipeline: detection -> tracking -> homography -> teams.
pipeline = tv.Pipeline(
    detector=detectors[ModelType.YOLO],
    keypoint_model=pitch_model,
    homography=estimator,
    tracker=TrackerType.BYTETRACK,
    team_classifier=tv.teams.TeamClassifier(embedder=EmbedderType.SIGLIP),
)
result = pipeline.run(VIDEO, end=250)
print(result.to_dataframe().head())
print("tracks:", len(result.track_ids))

# result.pkl, tracks.csv, freeze_frames.csv, carriers.csv, passes.csv
result.export(OUT, period=1)

# Swapping a component is plain Python: here RF-DETR replaces YOLO.
rfdetr_pipeline = tv.Pipeline(
    detector=detectors[ModelType.RFDETR],
    keypoint_model=pitch_model,
    homography=estimator,
    tracker=TrackerType.BYTETRACK,
)
rfdetr_result = rfdetr_pipeline.run(VIDEO, end=50)
print(
    "RF-DETR pipeline:",
    len(rfdetr_result),
    "frames,",
    len(rfdetr_result.track_ids),
    "tracks",
)
print("results in", OUT)
