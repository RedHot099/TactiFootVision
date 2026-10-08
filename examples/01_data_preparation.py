"""Data preparation: load, inspect, subset, convert and extract frames.

Run from the repository root: ``uv run python examples/01_data_preparation.py``
"""

from pathlib import Path

import tactifoot_vision as tv

OUT = Path("outputs/examples/01_data")
tv.setup_logging("INFO")

# Detection dataset in Ultralytics YOLO layout (a Roboflow export).
detection = tv.load_dataset("data/datasets/football_yolo")
print(detection)
print(detection.summary())

# Pitch-keypoint dataset: pose task with 32 landmarks and their flip permutation.
keypoints = tv.load_dataset("data/keypoints")
print(keypoints)
image, labels = keypoints.read("train", 0)
print(
    "image",
    image.shape,
    "keypoints",
    labels.keypoints.shape,
    "flip_idx",
    labels.flip_idx[:6],
)

# A small reproducible subset, exported in both formats the trainers use.
small = detection.subset({"train": 60, "valid": 20}, seed=0)
print("YOLO export:", small.to_yolo(OUT / "small_yolo"))
print("COCO export:", small.to_coco(OUT / "small_coco"))
assert tv.load_dataset(OUT / "small_coco").summary().equals(small.summary())

# Frames from match footage, e.g. to annotate a new dataset.
video = tv.VideoReader("data/videos/broadcast_60s.mp4")
print(video)
frames = tv.data.extract_frames(video.path, OUT / "frames", end=1500, stride=250)
print("extracted", [p.name for p in frames])

# StatsBomb 360 freeze frames for positional comparisons.
statsbomb = tv.data.load_statsbomb("data/statsbomb")
print(statsbomb[["period", "minute", "second", "type", "pitch_location"]].head())

tv.viz.show_samples(detection, "train", n=6).savefig(OUT / "detection_samples.png")
tv.viz.show_samples(keypoints, "train", n=3).savefig(OUT / "keypoint_samples.png")
print("figures in", OUT)
