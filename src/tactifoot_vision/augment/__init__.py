"""Data augmentation for detection and pitch-keypoint datasets.

    import numpy as np
    import tactifoot_vision as tv

    transform = tv.augment.Compose([
        tv.augment.HorizontalFlip(),
        tv.augment.RandomAffine(),
        tv.augment.OneOf([tv.augment.GaussianBlur(), tv.augment.MotionBlur()]),
        tv.augment.ColorJitter(),
    ])
    image, annotations = transform(image, annotations, rng=np.random.default_rng(0))
    bigger = tv.augment.augment_dataset(dataset, transform, "outputs/aug", copies=2)

Geometric transforms move boxes and keypoints with the pixels; photometric
transforms only change pixel values. There are no vertical flips on purpose:
broadcast football footage is never upside down.
"""

from tactifoot_vision.augment.base import Compose, OneOf, Transform
from tactifoot_vision.augment.dataset import augment_dataset
from tactifoot_vision.augment.geometric import HorizontalFlip, RandomAffine
from tactifoot_vision.augment.photometric import (
    ColorJitter,
    GaussianBlur,
    GaussianNoise,
    ImageOnlyTransform,
    JpegCompression,
    MotionBlur,
)

__all__ = [
    "ColorJitter",
    "Compose",
    "GaussianBlur",
    "GaussianNoise",
    "HorizontalFlip",
    "ImageOnlyTransform",
    "JpegCompression",
    "MotionBlur",
    "OneOf",
    "RandomAffine",
    "Transform",
    "augment_dataset",
]
