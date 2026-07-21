from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from sskit.camera import image_to_ground, load_camera, normalize, unnormalize, world_to_image


@dataclass(frozen=True)
class CameraCalibration:
    camera_matrix: list[list[float]]
    dist_poly: list[float]
    undist_poly: list[float]


def load_camera_sidecar(scene_dir: Path) -> CameraCalibration:
    camera_matrix, dist_poly, undist_poly = load_camera(scene_dir)
    return CameraCalibration(
        camera_matrix=camera_matrix.tolist(),
        dist_poly=dist_poly.tolist(),
        undist_poly=undist_poly.tolist(),
    )


def camera_from_image_record(
    image_record: Mapping[str, object],
    *,
    image_path: Path | None = None,
) -> CameraCalibration:
    camera_matrix = image_record.get("camera_matrix")
    dist_poly = image_record.get("dist_poly")
    undist_poly = image_record.get("undist_poly")
    if camera_matrix is not None and dist_poly is not None and undist_poly is not None:
        return CameraCalibration(
            camera_matrix=[[float(v) for v in row] for row in camera_matrix],  # type: ignore[arg-type]
            dist_poly=[float(v) for v in dist_poly],  # type: ignore[arg-type]
            undist_poly=[float(v) for v in undist_poly],  # type: ignore[arg-type]
        )

    if image_path is None:
        raise ValueError("Missing embedded camera info and no image_path provided.")
    return load_camera_sidecar(image_path.parent)


def pitch_points_to_image(
    world_points: Sequence[Sequence[float]] | np.ndarray,
    camera_matrix: Sequence[Sequence[float]],
    dist_poly: Sequence[float],
    *,
    image_shape: tuple[int, int, int] | None = None,
) -> np.ndarray:
    pts = np.asarray(world_points, dtype=np.float32)
    projected = np.asarray(world_to_image(camera_matrix, dist_poly, pts), dtype=np.float32)
    if image_shape is None:
        return projected
    return np.asarray(unnormalize(projected, image_shape), dtype=np.float32)


def image_points_to_pitch(
    image_points: Sequence[Sequence[float]] | np.ndarray,
    camera_matrix: Sequence[Sequence[float]],
    undist_poly: Sequence[float],
    *,
    image_shape: tuple[int, int, int] | None = None,
) -> np.ndarray:
    pts = np.asarray(image_points, dtype=np.float32)
    if image_shape is not None:
        pts = np.asarray(normalize(pts, image_shape), dtype=np.float32)
    projected = np.asarray(image_to_ground(camera_matrix, undist_poly, pts), dtype=np.float32)
    if projected.ndim == 1:
        projected = projected.reshape(1, -1)
    return projected


def read_camera_files(scene_dir: Path) -> dict[str, object]:
    calibration = load_camera_sidecar(scene_dir)
    lens_path = scene_dir / "lens.json"
    lens = {}
    if lens_path.is_file():
        with lens_path.open("r", encoding="utf-8") as handle:
            lens = json.load(handle)
    return {
        "camera_matrix": calibration.camera_matrix,
        "dist_poly": calibration.dist_poly,
        "undist_poly": calibration.undist_poly,
        "lens": lens,
    }
