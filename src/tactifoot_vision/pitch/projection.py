"""Mapping points through a homography, minding the horizon."""

import numpy as np


def project_points(
    points: np.ndarray, homography: np.ndarray, front: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Map ``(N, 2)`` points with ``homography``, flagging those beyond the horizon.

    A perspective view has a horizon: points on its far side have no image on
    the other plane and would wrap around to absurd coordinates. ``front`` is a
    point known to lie on the visible side (e.g. the frame centre for a
    frame->pitch homography). Returns ``(xy, valid)``: the mapped points and
    a mask of those on the same side as ``front``; invalid rows of ``xy`` are
    meaningless.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    homography = np.asarray(homography, dtype=float)
    side = np.sign((homography @ np.array([front[0], front[1], 1.0]))[2])
    projected = np.column_stack([points, np.ones(len(points))]) @ homography.T
    valid = projected[:, 2] * side > 1e-9
    xy = projected[:, :2] / np.where(valid, projected[:, 2], 1.0)[:, None]
    return xy, valid
