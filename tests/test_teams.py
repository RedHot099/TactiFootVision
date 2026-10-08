from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
import pytest

from tactifoot_vision.teams import (
    EMBEDDERS,
    Embedder,
    TeamClassifier,
    extract_crops,
)
from tactifoot_vision.teams.embedders import SigLIPEmbedder

RED, BLUE = (0, 0, 220), (220, 0, 0)  # BGR


class MeanColorEmbedder(Embedder):
    """Mean BGR colour in [0, 1]: enough to separate two kit colours."""

    name = "mean_color"

    def embed(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, 3), dtype=np.float32)
        return np.stack([c.reshape(-1, 3).mean(axis=0) / 255 for c in crops]).astype(
            np.float32
        )


def _crops(color, n, seed) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    crops = []
    for _ in range(n):
        crop = np.full((20, 10, 3), color, dtype=np.float64)
        crop += rng.normal(0, 25, crop.shape)
        crops.append(np.clip(crop, 0, 255).astype(np.uint8))
    return crops


# ---------------------------------------------------------------------- crops
def test_extract_crops_geometry():
    frame = (
        np.arange(100 * 200 * 3, dtype=np.uint32).reshape(100, 200, 3).astype(np.uint8)
    )
    boxes = np.array(
        [
            [20, 10, 40, 60],  # 20 x 50
            [0, 0, 1, 50],  # degenerate
            [300, 300, 350, 350],  # outside the frame
            [190, 90, 230, 130],  # partly outside: clipped
        ]
    )
    crops = extract_crops(frame, boxes, scale=0.6)
    assert crops[1] is None and crops[2] is None
    assert crops[0].shape == (30, 12, 3)  # 0.6 of the box around its centre
    np.testing.assert_array_equal(crops[0], frame[20:50, 24:36])
    np.testing.assert_array_equal(crops[3], frame[98:100, 198:200])
    crops[0][:] = 0
    assert frame[20:50, 24:36].any()  # crops are copies
    full = extract_crops(frame, boxes[:1], scale=1.0)[0]
    centre = extract_crops(frame, boxes[:1], scale=1.0, center_ratio=0.5)[0]
    assert full.shape == (50, 20, 3) and centre.shape == (25, 10, 3)
    np.testing.assert_array_equal(centre, full[12:37, 5:15])
    with pytest.raises(ValueError):
        extract_crops(frame, boxes, scale=0)


# ----------------------------------------------------------------- classifier
def test_classifier_splits_two_kits_with_deterministic_numbering():
    crops = _crops(RED, 30, 0) + _crops(BLUE, 20, 1)
    classifier = TeamClassifier(MeanColorEmbedder(), reducer=None).fit(crops)
    assert classifier.is_fitted
    labels = classifier.predict(crops)
    assert len(set(labels[:30])) == 1 and len(set(labels[30:])) == 1
    assert labels[0] != labels[-1]
    # teams are ordered by the first centre coordinate (mean blue channel here)
    assert labels[0] == 0 and labels[-1] == 1
    new = classifier.predict(_crops(BLUE, 5, 2) + _crops(RED, 5, 3))
    assert list(new) == [1] * 5 + [0] * 5
    again = TeamClassifier(MeanColorEmbedder(), reducer=None).fit_predict(crops[::-1])
    np.testing.assert_array_equal(again, labels[::-1])


def test_classifier_with_umap():
    crops = _crops(RED, 25, 0) + _crops(BLUE, 25, 1)
    classifier = TeamClassifier(MeanColorEmbedder(), umap_neighbors=10)
    labels = classifier.fit_predict(crops)
    assert len(set(labels[:25])) == 1 and len(set(labels[25:])) == 1
    assert labels[0] != labels[-1]
    np.testing.assert_array_equal(
        TeamClassifier(MeanColorEmbedder(), umap_neighbors=10).fit_predict(crops),
        labels,
    )


def test_classifier_embedding_api_matches_crop_api():
    crops = _crops(RED, 10, 0) + _crops(BLUE, 10, 1)
    classifier = TeamClassifier(MeanColorEmbedder(), reducer=None)
    embeddings = classifier.embed(crops)
    assert embeddings.shape == (20, 3) and embeddings.dtype == np.float32
    labels = classifier.fit_embeddings(embeddings).predict_embeddings(embeddings)
    np.testing.assert_array_equal(labels, classifier.predict(crops))
    assert classifier.predict_embeddings(np.zeros((0, 3))).shape == (0,)


def test_classifier_validation():
    with pytest.raises(RuntimeError, match="not fitted"):
        TeamClassifier(MeanColorEmbedder(), reducer=None).predict(_crops(RED, 2, 0))
    with pytest.raises(ValueError, match="at least 2"):
        TeamClassifier(MeanColorEmbedder(), reducer=None).fit(_crops(RED, 1, 0))
    with pytest.raises(ValueError, match="embedder given by name"):
        TeamClassifier(MeanColorEmbedder(), color_hist_bins=8)
    with pytest.raises(ValueError, match="reducer"):
        TeamClassifier(MeanColorEmbedder(), reducer="pca")
    with pytest.raises(ValueError, match="n_teams must be >= 2"):
        TeamClassifier(MeanColorEmbedder(), n_teams=1)
    with pytest.raises(ValueError, match="Unknown embedder"):
        TeamClassifier("nope")
    assert {"resnet", "siglip"} <= set(EMBEDDERS.names())


# ------------------------------------------------------------- real embedders
@pytest.mark.model
@pytest.mark.parametrize(
    ("name", "options", "dim"),
    [
        ("resnet", {}, 512),
        ("siglip", {}, 768),
        (
            "siglip",
            {"color_hist_bins": 8, "color_space": "hsv", "pooling": "cls"},
            768 + 24,
        ),
    ],
)
def test_real_embedders(name, options, dim):
    embedder = EMBEDDERS.create(name, **options)
    crops = _crops(RED, 3, 0) + _crops(BLUE, 2, 1) + [np.zeros((7, 3, 3), np.uint8)]
    features = embedder.embed(crops)
    assert features.shape == (6, dim) and features.dtype == np.float32
    assert np.isfinite(features).all()
    assert embedder.embed([]).shape == (0, dim)
    labels = TeamClassifier(embedder, reducer=None).fit_predict(crops[:5])
    assert len(set(labels[:3])) == 1 and len(set(labels[3:5])) == 1


def test_fit_uses_at_most_max_fit_samples():
    rng = np.random.default_rng(0)
    embeddings = np.concatenate(
        [rng.normal(0, 0.1, (60, 4)), rng.normal(5, 0.1, (60, 4))]
    )
    classifier = TeamClassifier(
        embedder=MeanColorEmbedder(), reducer=None, max_fit_samples=20
    )
    classifier.fit_embeddings(embeddings)
    assert classifier._kmeans.labels_.shape == (20,)
    teams = classifier.predict_embeddings(embeddings)
    assert (
        len(set(teams[:60])) == 1
        and len(set(teams[60:])) == 1
        and teams[0] != teams[-1]
    )


def test_umap_is_skipped_for_a_handful_of_crops():
    crops = _crops(RED, 3, seed=1) + _crops(BLUE, 3, seed=2)
    classifier = TeamClassifier(embedder=MeanColorEmbedder(), reducer="umap").fit(crops)
    teams = classifier.predict(crops)
    assert classifier._umap is None
    assert (
        len(set(teams[:3])) == 1 and len(set(teams[3:])) == 1 and teams[0] != teams[3]
    )


class _RecordingSigLIP:
    """Stands in for the SigLIP vision tower and keeps the pixels it was given."""

    config = SimpleNamespace(hidden_size=4)

    def __call__(self, pixel_values):
        import torch

        self.pixel_values = pixel_values
        hidden = torch.ones(len(pixel_values), 3, 4)
        return SimpleNamespace(last_hidden_state=hidden, pooler_output=hidden[:, 0])


@pytest.mark.parametrize("height", [1, 3])
def test_siglip_reads_flat_crops_as_channels_last(height):
    from transformers import SiglipImageProcessor

    embedder = object.__new__(SigLIPEmbedder)
    embedder.batch_size, embedder.pooling, embedder.color_hist_bins = 8, "mean", 0
    embedder.device, embedder._dim = "cpu", 4
    embedder._processor = SiglipImageProcessor()  # default config, no download
    embedder._model = _RecordingSigLIP()
    red = np.zeros((height, 40, 3), np.uint8)
    red[..., 2] = 255  # BGR
    assert embedder.embed([red]).shape == (1, 4)
    pixels = embedder._model.pixel_values
    assert pixels.shape == (1, 3, 224, 224)
    # SigLIP normalises to [-1, 1]: red channel full, green and blue empty.
    means = pixels[0].mean(dim=(1, 2)).tolist()
    assert means == pytest.approx([1.0, -1.0, -1.0], abs=1e-3)


# ------------------------------------------------------- review round 3
def test_a_fit_cap_below_n_teams_fails_before_the_embedder_is_built(monkeypatch):
    monkeypatch.setitem(
        EMBEDDERS._factories, "siglip", lambda **_: pytest.fail("embedder built")
    )
    with pytest.raises(ValueError, match="max_fit_samples=2 .* n_teams=3"):
        TeamClassifier("siglip", n_teams=3, max_fit_samples=2)
    with pytest.raises(ValueError, match="max_fit_samples"):
        TeamClassifier(MeanColorEmbedder(), max_fit_samples=0)


def test_a_fit_cap_of_exactly_n_teams_fits():
    classifier = TeamClassifier(MeanColorEmbedder(), reducer=None, max_fit_samples=2)
    embeddings = np.array([[0.0, 0.0], [1.0, 1.0], [0.1, 0.0], [0.9, 1.0]])
    classifier.fit_embeddings(embeddings)
    assert classifier._kmeans.labels_.shape == (2,)
    assert classifier.predict_embeddings(embeddings).shape == (4,)
