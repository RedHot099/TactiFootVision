import numpy as np
from numpy.typing import NDArray


def clustering_purity(predicted: NDArray[np.int_], expected: NDArray[np.int_]) -> float:
    if len(predicted) == 0 or len(predicted) != len(expected):
        return 0.0
    correct = 0
    for cluster in np.unique(predicted):
        mask = predicted == cluster
        labels, counts = np.unique(expected[mask], return_counts=True)
        _ = labels
        correct += int(counts.max()) if len(counts) else 0
    return correct / len(predicted)
