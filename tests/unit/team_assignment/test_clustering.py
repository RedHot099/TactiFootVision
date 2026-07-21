import numpy as np
import pytest

from tactifoot_vision.domain import AdapterUnavailable
from tactifoot_vision.team_assignment.clustering import (
    CMeansClusterer,
    DBSCANClusterer,
    KMeansClusterer,
)


def test_kmeans_two_color_fixture_assigns_two_clusters() -> None:
    features = np.array([[0.0], [0.1], [10.0], [10.1]], dtype=np.float32)

    labels = KMeansClusterer(clusters=2, random_state=0).fit_predict(features)

    assert len(set(labels.tolist())) == 2


def test_dbscan_handles_noise_labels() -> None:
    features = np.array([[0.0], [0.1], [10.0]], dtype=np.float32)

    labels = DBSCANClusterer(eps=0.2, min_samples=2).fit_predict(features)

    assert -1 in labels


def test_dbscan_predict_is_explicitly_unsupported() -> None:
    clusterer = DBSCANClusterer(eps=0.2, min_samples=2)
    clusterer.fit_predict(np.array([[0.0], [0.1], [10.0]], dtype=np.float32))

    with pytest.raises(RuntimeError, match="cannot predict unseen samples"):
        clusterer.predict(np.array([[0.0]], dtype=np.float32))


def test_cmeans_unavailable_is_explicit() -> None:
    with pytest.raises(AdapterUnavailable):
        CMeansClusterer()
