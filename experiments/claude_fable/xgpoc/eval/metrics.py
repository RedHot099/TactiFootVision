"""Shared evaluation for all PoCs: probabilistic quality of xG predictions."""

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def evaluate(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    base = np.full_like(p, y.mean())
    out = {
        "n": int(len(y)),
        "goal_rate": float(y.mean()),
        "brier": float(brier_score_loss(y, p)),
        "brier_baseline": float(brier_score_loss(y, base)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "log_loss_baseline": float(log_loss(y, base, labels=[0, 1])),
        "ece": expected_calibration_error(y, p),
    }
    out["brier_skill_score"] = 1 - out["brier"] / out["brier_baseline"]
    out["auc"] = float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else float("nan")
    return out
