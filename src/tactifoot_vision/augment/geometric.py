"""Geometric transforms: pixels, boxes and keypoints move together."""

from dataclasses import replace
from numbers import Real

import cv2
import numpy as np

from tactifoot_vision.augment.base import Transform, _as_range
from tactifoot_vision.data.annotations import NOT_LABELLED, Annotations

# Boxes thinner than this after clipping carry no trainable signal.
_MIN_BOX_SIZE = 1.0


class HorizontalFlip(Transform):
    """Mirror the image left-right.

    Keypoints are reordered with ``annotations.flip_idx`` so left/right
    landmarks swap names (new keypoint ``i`` is the mirrored old keypoint
    ``flip_idx[i]``, as in Ultralytics). Keypoint annotations without a
    ``flip_idx`` raise ``ValueError``: flipping them would silently mislabel
    the pitch landmarks.
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p)

    def apply(
        self, image: np.ndarray, annotations: Annotations, rng: np.random.Generator
    ) -> tuple[np.ndarray, Annotations]:
        width = image.shape[1]
        boxes = annotations.boxes.copy()
        boxes[:, [0, 2]] = width - annotations.boxes[:, [2, 0]]
        keypoints = annotations.keypoints
        if keypoints is not None:
            keypoints = keypoints[:, _checked_flip_idx(annotations)]  # a copy
            keypoints[..., 0] = width - keypoints[..., 0]
        return cv2.flip(image, 1), replace(
            annotations, boxes=boxes, keypoints=keypoints
        )


def _checked_flip_idx(annotations: Annotations) -> list[int]:
    flip_idx = annotations.flip_idx
    if flip_idx is None:
        raise ValueError(
            "HorizontalFlip needs annotations.flip_idx to flip keypoints; set "
            "flip_idx in the dataset (e.g. data.yaml) or leave HorizontalFlip out"
        )
    if len(flip_idx) != annotations.num_keypoints:
        raise ValueError(
            f"flip_idx has {len(flip_idx)} entries but objects have "
            f"{annotations.num_keypoints} keypoints"
        )
    return list(flip_idx)


class RandomAffine(Transform):
    """Random rotation and scale about the image centre, then a random shift.

    Args:
        degrees: maximum rotation ``d`` (angle drawn from ``(-d, d)``) or an
            explicit ``(min, max)`` range. Positive angles rotate
            counter-clockwise, as in ``cv2.getRotationMatrix2D``.
        translate: maximum shift as a fraction of the image width / height.
        scale: ``(min, max)`` zoom factor.
        p: probability of applying the transform.
        min_area_ratio: drop an object when its clipped box keeps less than
            this fraction of its transformed box area (mostly out of frame).
        border_value: BGR fill for pixels that come from outside the image.

    Boxes become the bounding box of their four transformed corners, clipped
    to the image; objects left with a box under one pixel wide or tall are
    dropped too. Keypoints that leave the image become not labelled
    (visibility 0); unlabelled keypoints stay unlabelled.
    """

    def __init__(
        self,
        degrees: float | tuple[float, float] = 5.0,
        translate: float = 0.1,
        scale: tuple[float, float] = (0.8, 1.2),
        p: float = 0.5,
        min_area_ratio: float = 0.3,
        border_value: tuple[int, int, int] = (114, 114, 114),
    ) -> None:
        super().__init__(p)
        if isinstance(degrees, Real):
            if degrees < 0:
                raise ValueError(f"degrees must be >= 0, got {degrees}")
            self.degrees: float | tuple[float, float] = float(degrees)
            self._angle_range = (-float(degrees), float(degrees))
        else:
            self.degrees = self._angle_range = _as_range(degrees, "degrees")
        if not 0.0 <= translate < 1.0:
            raise ValueError(f"translate must be in [0, 1), got {translate}")
        self.translate = float(translate)
        self.scale = _as_range(scale, "scale")
        if self.scale[0] <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        if not 0.0 <= min_area_ratio <= 1.0:
            raise ValueError(f"min_area_ratio must be in [0, 1], got {min_area_ratio}")
        self.min_area_ratio = float(min_area_ratio)
        self.border_value = tuple(int(v) for v in border_value)

    def apply(
        self, image: np.ndarray, annotations: Annotations, rng: np.random.Generator
    ) -> tuple[np.ndarray, Annotations]:
        height, width = image.shape[:2]
        matrix = self._sample_matrix(width, height, rng)
        # Labels use continuous coordinates (pixel i spans [i, i+1]) while cv2
        # puts pixel centres on integers; shift by half a pixel so pixels and
        # labels move identically.
        pixel_matrix = matrix.copy()
        pixel_matrix[:, 2] += 0.5 * matrix[:, :2].sum(axis=1) - 0.5
        warped = cv2.warpAffine(
            image,
            pixel_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.border_value,
        )

        corners = annotations.boxes[:, [[0, 1], [2, 1], [2, 3], [0, 3]]]
        moved_corners = _affine(corners, matrix)
        moved = np.concatenate(
            [moved_corners.min(axis=1), moved_corners.max(axis=1)], axis=1
        )
        clipped = moved.clip(0, [width, height, width, height])
        keep = (clipped[:, 2:] - clipped[:, :2] >= _MIN_BOX_SIZE).all(axis=1) & (
            _area(clipped) >= self.min_area_ratio * _area(moved)
        )

        keypoints = annotations.keypoints
        if keypoints is not None:
            keypoints = keypoints.copy()
            keypoints[..., :2] = _affine(keypoints[..., :2], matrix)
            x, y = keypoints[..., 0], keypoints[..., 1]
            outside = (x < 0) | (x > width) | (y < 0) | (y > height)
            keypoints[outside, 2] = NOT_LABELLED

        moved_annotations = replace(annotations, boxes=clipped, keypoints=keypoints)
        return warped, moved_annotations.select(keep)

    def _sample_matrix(
        self, width: int, height: int, rng: np.random.Generator
    ) -> np.ndarray:
        """``2 x 3`` affine matrix in continuous pixel coordinates."""
        angle = rng.uniform(*self._angle_range)
        zoom = rng.uniform(*self.scale)
        shift = rng.uniform(-self.translate, self.translate, size=2) * (width, height)
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, zoom)
        matrix[:, 2] += shift
        return matrix


def _affine(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a ``2 x 3`` matrix to ``(..., 2)`` points (same as ``cv2.transform``)."""
    return points @ matrix[:, :2].T + matrix[:, 2]


def _area(boxes: np.ndarray) -> np.ndarray:
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
