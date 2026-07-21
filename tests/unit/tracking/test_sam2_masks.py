import numpy as np

from tactifoot_vision.tracking.sam2_masks import (
    filter_segments_by_distance,
    postprocess_masks,
    tight_bbox_from_masks,
)


def test_filter_segments_keeps_largest_component() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[1:3, 1:3] = True
    mask[10:16, 10:16] = True

    result = filter_segments_by_distance(mask, distance_threshold=1.0)

    assert not result[1:3, 1:3].any()
    assert result[10:16, 10:16].all()


def test_filter_segments_keeps_nearby_components() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[10:14, 10:14] = True
    mask[10:12, 15:17] = True

    result = filter_segments_by_distance(mask, distance_threshold=6.0)

    assert result[10:14, 10:14].all()
    assert result[10:12, 15:17].all()


def test_postprocess_open_removes_isolated_noise() -> None:
    masks = np.zeros((1, 9, 9), dtype=bool)
    masks[0, 4, 4] = True

    result = postprocess_masks(masks, open_kernel_size=3, close_kernel_size=0)

    assert not result.any()


def test_postprocess_close_fills_small_gap() -> None:
    masks = np.zeros((1, 9, 9), dtype=bool)
    masks[0, 3:6, 3] = True
    masks[0, 3:6, 5] = True

    result = postprocess_masks(masks, open_kernel_size=0, close_kernel_size=3)

    assert result[0, 4, 4]


def test_tight_bbox_from_masks_returns_expected_coordinates() -> None:
    masks = np.zeros((1, 10, 10), dtype=bool)
    masks[0, 2:6, 3:8] = True

    result = tight_bbox_from_masks(masks, min_mask_area=1.0)

    assert result.tolist() == [[3.0, 2.0, 8.0, 6.0]]


def test_tight_bbox_falls_back_to_pixel_extent_for_small_artifact() -> None:
    masks = np.zeros((1, 10, 10), dtype=bool)
    masks[0, 4, 5] = True

    result = tight_bbox_from_masks(masks, min_mask_area=100.0)

    assert result.tolist() == [[5.0, 4.0, 5.0, 4.0]]


def test_tight_bbox_empty_mask_returns_zero_box() -> None:
    masks = np.zeros((1, 10, 10), dtype=bool)

    result = tight_bbox_from_masks(masks, min_mask_area=1.0)

    assert result.tolist() == [[0.0, 0.0, 0.0, 0.0]]
