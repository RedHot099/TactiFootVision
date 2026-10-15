"""Frame <-> pitch homography from detected pitch keypoints."""

import logging
from collections import deque
from typing import TYPE_CHECKING

import cv2
import numpy as np
import supervision as sv

from tactifoot_vision.pitch.pitch import SoccerPitch

if TYPE_CHECKING:
    from tactifoot_vision.pitch.lines import LineFit

logger = logging.getLogger(__name__)

_GOOD_INLIER_SHARE = 0.85


def frame_to_pitch(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Map ``(N, 2)`` frame pixels to pitch coordinates with a frame->pitch homography."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return cv2.perspectiveTransform(points.reshape(-1, 1, 2), homography).reshape(-1, 2)


def pitch_to_frame(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Map ``(N, 2)`` pitch coordinates to frame pixels (inverse of :func:`frame_to_pitch`)."""
    return frame_to_pitch(points, np.linalg.inv(homography))


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


def visible_area(
    homography: np.ndarray | None, frame_size: tuple[int, int]
) -> np.ndarray | None:
    """The part of the pitch a frame shows: its outline through a frame->pitch homography.

    ``frame_size`` is the frame's ``(width, height)``. Returns the ``(4, 2)``
    pitch positions of the frame's corners, clockwise on screen from the
    top-left one, taken at the pixel edges (``(0, 0)`` to ``(width, height)``)
    as :func:`~tactifoot_vision.viz.warp_to_pitch` warps them. ``None`` when
    part of the frame lies beyond the horizon, where the outline has no finite
    pitch position, or without a homography.
    """
    if homography is None:
        return None
    width, height = frame_size
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], float)
    outline, in_front = project_points(corners, homography, (width / 2, height / 2))
    return outline if in_front.all() else None


class HomographyEstimator:
    """Fits a frame->pitch homography per frame and smooths it over time.

    Keypoints below ``min_confidence`` are ignored; at least four are needed.
    Averaging the last ``smoothing_window`` fits steadies pitch positions but lags
    behind camera pans: on broadcast footage 3 frames cut position jitter in half
    versus no smoothing while adding about 2 px of line misalignment.
    When a frame yields no fit, the last good matrix is reused for up to
    ``max_age`` updates (processed frames; ``None`` = forever), then dropped.
    After such a gap the first new fit starts a fresh average, so matrices from
    before a camera cut are never blended with the new view.

    ``ransac_threshold`` is the RANSAC reprojection limit in pitch units (metres
    on the default pitch): keypoints further than this from the fitted model are
    ignored. The loose default favours smooth positions over outlier rejection.

    ``refine_with_lines`` (off by default) then fits each frame's matrix to the
    painted lines of the frame passed to :meth:`update`: the pitch markings are
    pulled onto a white-line mask from which the people ``boxes`` are cut,
    starting from the previous returned matrix and from the keypoint one, and
    the start whose fit has more inliers wins. Fewer than ``min_line_points``
    visible model points (close-ups, lines out of view) keep the keypoint
    matrix. :attr:`line_fit` tells how the current frame's fit went; a model
    point within ``line_inlier_px`` pixels of a painted line is an inlier.
    """

    def __init__(
        self,
        pitch: SoccerPitch | None = None,
        *,
        min_confidence: float = 0.5,
        smoothing_window: int = 3,
        ransac_threshold: float = 10.0,
        max_age: int | None = 50,
        refine_with_lines: bool = False,
        min_line_points: int = 80,
        line_inlier_px: float = 3.0,
    ) -> None:
        if not 0 <= min_confidence <= 1:
            raise ValueError(f"min_confidence must be in [0, 1], got {min_confidence}")
        if smoothing_window < 1:
            raise ValueError(f"smoothing_window must be >= 1, got {smoothing_window}")
        if ransac_threshold <= 0:
            raise ValueError(f"ransac_threshold must be > 0, got {ransac_threshold}")
        if max_age is not None and max_age < 0:
            raise ValueError(f"max_age must be None or >= 0, got {max_age}")
        if min_line_points < 4:
            raise ValueError(f"min_line_points must be >= 4, got {min_line_points}")
        if line_inlier_px <= 0:
            raise ValueError(f"line_inlier_px must be > 0, got {line_inlier_px}")
        self.pitch = pitch or SoccerPitch()
        self.min_confidence = min_confidence
        self.ransac_threshold = ransac_threshold
        self.max_age = max_age
        self.refine_with_lines = refine_with_lines
        self.min_line_points = min_line_points
        self.line_inlier_px = line_inlier_px
        self._model_points = None
        if refine_with_lines:
            # lines.py projects with this module's functions: import it late.
            from tactifoot_vision.pitch.lines import model_points

            self._model_points = model_points(self.pitch)
        self._history: deque[np.ndarray] = deque(maxlen=smoothing_window)
        # The keypoint fits' average, and the matrix returned (refined or not).
        self._keypoint_matrix: np.ndarray | None = None
        self.matrix: np.ndarray | None = None
        # How the current frame's line refinement went; None when it was off,
        # had no matrix to start from, or fell back to the keypoint matrix.
        self.line_fit: LineFit | None = None
        # Confident keypoints given to RANSAC in the current frame's fit (not
        # only its inliers); None when this frame gave no fit.
        self.used_indices: np.ndarray | None = None
        self._age = 0

    def reset(self) -> None:
        self._history.clear()
        self._keypoint_matrix = None
        self.matrix = None
        self.used_indices = None
        self.line_fit = None
        self._age = 0

    def update(
        self,
        keypoints: sv.KeyPoints | None,
        frame: np.ndarray | None = None,
        boxes: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Fit on this frame's keypoints (first instance) and return the current matrix.

        With ``refine_with_lines``, ``frame`` (BGR) is required and the matrix
        is fitted to its painted lines, ignoring the people ``boxes`` (xyxy).
        """
        if self.refine_with_lines and frame is None:
            raise ValueError("refine_with_lines needs the frame passed to update")
        previous = self.matrix
        fitted = self._fit(keypoints)
        if fitted is not None:
            if self._age:
                self._history.clear()
            self._history.append(fitted)
            self._keypoint_matrix = np.mean(self._history, axis=0)
            self._age = 0
        elif self._keypoint_matrix is not None:
            self._age += 1
            if self.max_age is not None and self._age > self.max_age:
                self.reset()
                previous = None
        self.matrix = self._keypoint_matrix
        self.line_fit = None
        if self.refine_with_lines and self.matrix is not None:
            self._refine(frame, previous, boxes)
        return self.matrix

    def _refine(
        self, frame: np.ndarray, previous: np.ndarray | None, boxes: np.ndarray | None
    ) -> None:
        """Replace :attr:`matrix` with the better line fit from ``previous`` or from it."""
        from tactifoot_vision.pitch.lines import line_masks, refine_homography

        line, seen = line_masks(frame, boxes)
        best = None
        for start in (previous, self.matrix):
            if start is None:
                continue
            refined = refine_homography(
                start,
                self._model_points,
                line,
                seen,
                min_points=self.min_line_points,
                inlier_px=self.line_inlier_px,
            )
            if refined is not None and (
                best is None or refined[1].inlier_share > best[1].inlier_share
            ):
                best = refined
            # A fit this good from the previous frame needs no second try.
            if best is not None and best[1].inlier_share > _GOOD_INLIER_SHARE:
                break
        if best is not None:
            self.matrix, self.line_fit = best

    def _fit(self, keypoints: sv.KeyPoints | None) -> np.ndarray | None:
        self.used_indices = None
        vertices = self.pitch.vertices
        if keypoints is None or len(keypoints) == 0 or keypoints.confidence is None:
            return None
        xy, confidence = keypoints.xy[0], keypoints.confidence[0]
        if len(xy) != len(vertices):
            raise ValueError(
                f"The keypoint model predicts {len(xy)} keypoints but the pitch has "
                f"{len(vertices)} landmarks; use a pitch-landmark model"
            )
        indices = np.flatnonzero(confidence >= self.min_confidence)
        if len(indices) < 4:
            return None
        matrix, _ = cv2.findHomography(
            xy[indices].astype(np.float32),
            vertices[indices],
            cv2.RANSAC,
            self.ransac_threshold,
        )
        if matrix is None or not np.isfinite(matrix).all() or abs(matrix[2, 2]) < 1e-12:
            return None
        self.used_indices = indices
        return matrix / matrix[2, 2]
