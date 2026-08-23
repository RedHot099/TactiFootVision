import numpy as np
from xgpoc.eval.metrics import evaluate


def test_perfect_and_baseline():
    y = np.array([0, 0, 1, 0, 1, 0, 0, 0, 0, 1])
    perfect = evaluate(y, y.astype(float))
    assert perfect["brier"] < 1e-6 and perfect["auc"] == 1.0
    base = evaluate(y, np.full(10, y.mean()))
    assert abs(base["brier_skill_score"]) < 1e-9
