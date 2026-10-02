"""Ball path post-processing."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def clean_ball_path(positions: np.ndarray, max_jump: float) -> np.ndarray:
    """Drop ball positions that move implausibly fast.

    Walks the path in order and keeps a position only if it lies within
    ``max_jump * gap`` of the last kept one, where ``gap`` is the number of
    rows (frames) between the two; rejected positions become NaN. The first
    finite position is always kept. Scaling by the gap matters: the ball is
    often undetected for a second or more while it travels far, and a fixed
    threshold would then reject every later position.

    Args:
        positions: ``(T, 2)`` positions, one row per frame, NaN where the ball is missing.
        max_jump: largest plausible move per row (e.g. metres per processed frame).

    Returns:
        A cleaned ``(T, 2)`` float copy of ``positions``.
    """
    if max_jump <= 0:
        raise ValueError("max_jump must be positive")
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(f"positions must have shape (T, 2), got {positions.shape}")
    cleaned = np.full_like(positions, np.nan)
    last: np.ndarray | None = None
    last_row = 0
    for row, point in enumerate(positions):
        if not np.isfinite(point).all():
            continue
        if last is not None and np.linalg.norm(point - last) > max_jump * (
            row - last_row
        ):
            continue
        cleaned[row] = point
        last, last_row = point, row
    logger.debug(
        "Ball path cleaning kept %d of %d positions",
        np.isfinite(cleaned).all(axis=1).sum(),
        np.isfinite(positions).all(axis=1).sum(),
    )
    return cleaned
