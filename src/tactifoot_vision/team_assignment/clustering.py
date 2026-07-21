import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import DBSCAN, KMeans

from tactifoot_vision.domain import AdapterUnavailable


class KMeansClusterer:
    def __init__(self, *, clusters: int = 2, random_state: int = 0) -> None:
        self._model = KMeans(n_clusters=clusters, n_init=10, random_state=random_state)

    def fit_predict(self, features: NDArray[np.float32]) -> NDArray[np.int_]:
        return np.asarray(self._model.fit_predict(features), dtype=np.int_)

    def predict(self, features: NDArray[np.float32]) -> NDArray[np.int_]:
        return np.asarray(self._model.predict(features), dtype=np.int_)


class DBSCANClusterer:
    def __init__(self, *, eps: float = 0.5, min_samples: int = 2) -> None:
        self._model = DBSCAN(eps=eps, min_samples=min_samples)
        self._labels: NDArray[np.int_] | None = None

    def fit_predict(self, features: NDArray[np.float32]) -> NDArray[np.int_]:
        self._labels = np.asarray(self._model.fit_predict(features), dtype=np.int_)
        return self._labels

    def predict(self, features: NDArray[np.float32]) -> NDArray[np.int_]:
        _ = features
        raise RuntimeError(
            "DBSCANClusterer cannot predict unseen samples; use fit_predict()."
        )


class CMeansClusterer:
    def __init__(self, *args: object, **kwargs: object) -> None:
        _ = args, kwargs
        raise AdapterUnavailable(
            "CMeans clustering requires an optional dependency that is not installed."
        )
