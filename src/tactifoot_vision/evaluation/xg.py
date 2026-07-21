import math
from collections.abc import Sequence


def brier_score(y_true: Sequence[float], y_prob: Sequence[float]) -> float:
    _validate_equal_lengths(y_true, y_prob)
    if not y_true:
        return 0.0
    total = sum(
        (float(prob) - float(true)) ** 2
        for true, prob in zip(y_true, y_prob, strict=True)
    )
    return float(total / len(y_true))


def log_loss(
    y_true: Sequence[float], y_prob: Sequence[float], *, eps: float = 1e-15
) -> float:
    _validate_equal_lengths(y_true, y_prob)
    if not y_true:
        return 0.0
    total = 0.0
    for true, prob in zip(y_true, y_prob, strict=True):
        clipped = min(max(float(prob), eps), 1.0 - eps)
        total += float(true) * math.log(clipped) + (1.0 - float(true)) * math.log(
            1.0 - clipped
        )
    return float(-total / len(y_true))


def expected_calibration_error(
    y_true: Sequence[float], y_prob: Sequence[float], *, bins: int = 10
) -> float:
    _validate_equal_lengths(y_true, y_prob)
    if bins <= 0:
        raise ValueError("bins must be positive.")
    if not y_true:
        return 0.0
    total = 0.0
    for bin_index in range(bins):
        start = bin_index / bins
        end = (bin_index + 1) / bins
        indexes = [
            index
            for index, prob in enumerate(y_prob)
            if start <= float(prob) < end
            or (bin_index == bins - 1 and float(prob) == 1.0)
        ]
        if not indexes:
            continue
        accuracy = sum(float(y_true[index]) for index in indexes) / len(indexes)
        confidence = sum(float(y_prob[index]) for index in indexes) / len(indexes)
        total += (len(indexes) / len(y_true)) * abs(accuracy - confidence)
    return float(total)


def aggregate_mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    _validate_equal_lengths(actual, predicted)
    if not actual:
        return 0.0
    total = sum(
        abs(float(left) - float(right))
        for left, right in zip(actual, predicted, strict=True)
    )
    return float(total / len(actual))


def _validate_equal_lengths(left: Sequence[float], right: Sequence[float]) -> None:
    if len(left) != len(right):
        raise ValueError("Metric inputs must have equal lengths.")
