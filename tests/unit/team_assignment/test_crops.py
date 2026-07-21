import numpy as np

from tactifoot_vision.domain import BBox
from tactifoot_vision.team_assignment.crops import crop_bbox, crop_sample
from tactifoot_vision.team_assignment.opencv_masks import opencv_mask_crop


def test_crop_bbox_returns_expected_shape() -> None:
    image = np.zeros((10, 20, 3), dtype=np.uint8)

    crop = crop_bbox(image, BBox(2, 3, 12, 9), ratio=1.0)

    assert crop.shape == (6, 10, 3)


def test_invalid_bbox_returns_empty_crop() -> None:
    image = np.zeros((10, 20, 3), dtype=np.uint8)

    crop = crop_bbox(image, BBox(5, 5, 5, 5), ratio=1.0)

    assert crop.size == 0


def test_crop_sample_wraps_metadata() -> None:
    image = np.zeros((10, 20, 3), dtype=np.uint8)

    sample = crop_sample(
        frame_index=3,
        track_id=10,
        class_name="player",
        image=image,
        bbox=BBox(2, 3, 12, 9),
    )

    assert sample.frame_index == 3
    assert sample.track_id == 10
    assert sample.image.shape == (6, 10, 3)


def test_opencv_mask_crop_does_not_crash() -> None:
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    image[3:8, 4:12] = 255

    crop = opencv_mask_crop(image, BBox(2, 2, 14, 9))

    assert crop is not None
    assert crop.ndim == 3
