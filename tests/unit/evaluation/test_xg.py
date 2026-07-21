import pytest

from tactifoot_vision.evaluation.xg import (
    aggregate_mae,
    brier_score,
    expected_calibration_error,
    log_loss,
)


def test_xg_probability_metrics() -> None:
    y_true = [1.0, 0.0]
    y_prob = [0.8, 0.2]

    assert brier_score(y_true, y_prob) == pytest.approx(0.04)
    assert log_loss(y_true, y_prob) == pytest.approx(0.223143, rel=1e-5)
    assert expected_calibration_error(y_true, y_prob, bins=2) == pytest.approx(0.2)
    assert aggregate_mae([1.0, 3.0], [1.5, 2.0]) == pytest.approx(0.75)


def test_xg_metrics_require_equal_lengths() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        brier_score([1.0], [0.5, 0.6])
