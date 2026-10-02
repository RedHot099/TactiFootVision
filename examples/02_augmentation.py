"""Data augmentation: preview transforms and grow the training split offline.

Run from the repository root: ``uv run python examples/02_augmentation.py``
"""

from pathlib import Path

import tactifoot_vision as tv
from tactifoot_vision import augment as A

OUT = Path("outputs/examples/02_augment")
tv.setup_logging("INFO")

# Broadcast-style augmentations: camera zoom/pan, lighting, motion blur, compression.
transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomAffine(degrees=3, translate=0.05, scale=(0.8, 1.2), p=0.7),
        A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.3, p=0.8),
        A.OneOf([A.MotionBlur(), A.GaussianBlur(), A.GaussianNoise()], p=0.4),
        A.JpegCompression(p=0.3),
    ]
)
print(transform)

detection = tv.load_dataset("data/datasets/football_yolo").subset(
    {"train": 40, "valid": 10}
)
keypoints = tv.load_dataset("data/keypoints").subset({"train": 40})

# The same transform handles boxes and pitch keypoints (flip_idx swaps landmarks).
OUT.mkdir(parents=True, exist_ok=True)
tv.viz.show_augmentations(detection, transform, n=3).savefig(
    OUT / "detection_augmentations.png"
)
tv.viz.show_augmentations(keypoints, transform, n=3).savefig(
    OUT / "keypoint_augmentations.png"
)

augmented = A.augment_dataset(detection, transform, OUT / "detection", copies=2, seed=0)
print(augmented)
print(augmented.summary())
assert len(augmented["train"]) == 3 * len(detection["train"])
assert augmented["valid"] == detection["valid"]  # validation data stays real

pose_augmented = A.augment_dataset(keypoints, transform, OUT / "keypoints", copies=1)
print(pose_augmented)
print("figures in", OUT)
