"""Post-hoc calibration helpers shared by the trainer and the ensemble builders.

Three things the pre-v1.2 pipeline got wrong are handled here: a temperature alone
cannot move the mean prediction (Platt adds the offset), the validation split has a
higher goal rate than the rest of the data (prior shift), and penalties are a known
input condition with a known conversion rate (routing) rather than something an image
model should guess from the picture.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.asarray(z, dtype=float)))


class PlattScaler:
    """p = sigmoid(a * z + b), fitted on validation logits without regularisation."""

    def __init__(self, a: float = 1.0, b: float = 0.0):
        self.a, self.b = a, b

    def fit(self, z: np.ndarray, y: np.ndarray) -> "PlattScaler":
        lr = LogisticRegression(C=1e6, max_iter=2000).fit(np.asarray(z, float).reshape(-1, 1), np.asarray(y).astype(int))
        self.a, self.b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        return self

    def __call__(self, z: np.ndarray) -> np.ndarray:
        return sigmoid(self.a * np.asarray(z, float) + self.b)


def prior_shift(p: np.ndarray, from_rate: float, to_rate: float) -> np.ndarray:
    """Re-express probabilities calibrated under base rate `from_rate` for base rate
    `to_rate` (known-prior correction; a constant shift in logit space)."""
    return sigmoid(logit(p) + np.log(to_rate / (1 - to_rate)) - np.log(from_rate / (1 - from_rate)))


def penalty_rate(n_goals: int, n_penalties: int, prior_mean: float = 0.76, prior_strength: float = 40.0) -> float:
    """Posterior mean conversion of penalties: Beta prior centred on the literature rate
    (~0.76 across top leagues) with the weight of `prior_strength` observations, updated
    with the training penalties."""
    a, b = prior_mean * prior_strength, (1 - prior_mean) * prior_strength
    return float((a + n_goals) / (a + b + n_penalties))


def route_penalties(p: np.ndarray, is_penalty: np.ndarray, p_pen: float) -> np.ndarray:
    """Penalties get the dedicated rate; everything else keeps the model's estimate."""
    return np.where(np.asarray(is_penalty, bool), p_pen, np.asarray(p, float))
