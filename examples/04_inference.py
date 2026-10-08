"""Model inference: single-frame predictions and the full video pipeline.

Run from the repository root: ``uv run python examples/04_inference.py``
"""

from collections import Counter
from pathlib import Path

import supervision as sv

import tactifoot_vision as tv

OUT = Path("outputs/examples/04_inference")
VIDEO = Path("data/videos/broadcast_60s.mp4")
tv.setup_logging("INFO")

video = tv.VideoReader(VIDEO)
frame = video.read(250)

# Every detector answers the same call with sv.Detections + class names.
detectors = {
    "yolo": tv.load_model("yolo", "models/football_yolo11m.pt", conf=0.3),
    "rfdetr": tv.load_model("rfdetr", "models/football_rfdetr_base.pth", conf=0.5),
}
for name, model in detectors.items():
    detections = model(frame)
    print(
        f"{name:7s} {len(detections):3d} objects",
        dict(Counter(detections.data["class_name"])),
    )

# Pitch keypoints -> homography -> pitch coordinates of the players' feet.
pitch_model = tv.load_model("yolo_pose", "models/pitch_yolov8n_pose.pt")
keypoints = pitch_model(frame)
homography = tv.pitch.HomographyEstimator().update(keypoints)
feet = detectors["yolo"](frame).get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
print(
    "first players on the pitch (m):",
    tv.pitch.frame_to_pitch(feet, homography)[:3].round(1),
)

# The full pipeline: detection -> tracking -> homography -> teams.
pipeline = tv.Pipeline(
    detector=detectors["yolo"],
    keypoint_model=pitch_model,
    tracker="bytetrack",
    team_classifier=tv.teams.TeamClassifier(embedder="siglip"),
)
result = pipeline.run(VIDEO, end=250)
print(result.to_dataframe().head())
print("tracks:", len(result.track_ids))

result.export(OUT, period=1)  # result.pkl, tracks.csv, freeze_frames.csv

# Swapping a component is plain Python: here RF-DETR replaces YOLO.
rfdetr_pipeline = tv.Pipeline(
    detector=detectors["rfdetr"],
    keypoint_model=pitch_model,
    tracker="bytetrack",
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
