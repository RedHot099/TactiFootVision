import numpy as np

from tactifoot_vision.team_assignment.embeddings import ColorHistogramEmbedder


def test_color_histogram_embedder_returns_expected_shape() -> None:
    crops = [
        np.full((8, 8, 3), (255, 0, 0), dtype=np.uint8),
        np.full((8, 8, 3), (0, 255, 0), dtype=np.uint8),
    ]

    features = ColorHistogramEmbedder(bins=4).embed(crops)

    assert features.shape == (2, 12)


def test_color_histogram_empty_crops_returns_empty_matrix() -> None:
    features = ColorHistogramEmbedder(bins=4).embed([])

    assert features.shape == (0, 12)
