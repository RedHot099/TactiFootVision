from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class Embedder(Protocol):
    def embed(self, crops: list[NDArray[np.uint8]]) -> NDArray[np.float32]:
        raise NotImplementedError


class Reducer(Protocol):
    def fit_transform(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        raise NotImplementedError

    def transform(self, features: NDArray[np.float32]) -> NDArray[np.float32]:
        raise NotImplementedError


class Clusterer(Protocol):
    def fit_predict(self, features: NDArray[np.float32]) -> NDArray[np.int_]:
        raise NotImplementedError

    def predict(self, features: NDArray[np.float32]) -> NDArray[np.int_]:
        raise NotImplementedError
