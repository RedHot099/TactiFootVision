import numpy as np
from numpy.typing import NDArray


class IdentityReducer:
    def fit_transform(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        return features

    def transform(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        return features


class UMAPReducer:
    def __init__(
        self,
        *,
        components: int = 3,
        neighbors: int = 15,
        min_dist: float = 0.1,
        random_state: int | None = 0,
    ) -> None:
        import umap

        self._model = umap.UMAP(
            n_components=components,
            n_neighbors=neighbors,
            min_dist=min_dist,
            random_state=random_state,
        )

    def fit_transform(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        if len(features) == 0:
            return features
        return np.asarray(self._model.fit_transform(features), dtype=np.float32)

    def transform(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        if len(features) == 0:
            return features
        return np.asarray(self._model.transform(features), dtype=np.float32)
