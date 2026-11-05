"""Fisheye undistortion of frame points, for a calibrated static camera."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class FisheyeUndistortion:
    """Moves frame points to where a fisheye camera's undistorted image shows them.

    The model is OpenCV's ``cv2.fisheye``: ``camera`` is the 3x3 matrix of
    the image as recorded, ``distortion`` its four coefficients, and
    ``new_camera`` the 3x3 matrix of the undistorted image, the one a
    homography of that image is fitted on. :meth:`load` reads the three from
    an ``.npz`` with the keys ``K``, ``D`` and ``Knew`` (as
    :func:`~tactifoot_vision.data.download_demo` writes them).
    """

    camera: np.ndarray
    distortion: np.ndarray
    new_camera: np.ndarray

    def __post_init__(self) -> None:
        camera = np.asarray(self.camera, dtype=float)
        distortion = np.asarray(self.distortion, dtype=float).reshape(-1)
        new_camera = np.asarray(self.new_camera, dtype=float)
        if camera.shape != (3, 3) or new_camera.shape != (3, 3):
            raise ValueError(
                "camera and new_camera must be 3x3 matrices, got shapes "
                f"{camera.shape} and {new_camera.shape}"
            )
        if distortion.shape != (4,):
            raise ValueError(
                f"distortion must hold the 4 fisheye coefficients, got {distortion.shape}"
            )
        object.__setattr__(self, "camera", camera)
        object.__setattr__(self, "distortion", distortion)
        object.__setattr__(self, "new_camera", new_camera)

    @classmethod
    def load(cls, path: str | Path) -> "FisheyeUndistortion":
        """Read ``K``, ``D`` and ``Knew`` from an ``.npz``."""
        with np.load(path) as arrays:
            try:
                return cls(arrays["K"], arrays["D"], arrays["Knew"])
            except KeyError as error:
                raise ValueError(
                    f"{path} has no array {error}; needs K, D and Knew"
                ) from None

    def points(self, xy: np.ndarray) -> np.ndarray:
        """``(N, 2)`` frame points -> their ``(N, 2)`` undistorted-image points."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 1, 2)
        if len(xy) == 0:
            return np.zeros((0, 2))
        moved = cv2.fisheye.undistortPoints(
            xy, self.camera, self.distortion, P=self.new_camera
        )
        return moved.reshape(-1, 2)
