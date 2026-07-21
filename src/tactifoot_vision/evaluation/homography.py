import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tactifoot_vision.datasets.soccernet_gsr import (
    GsrAthleteAnnotation,
    SoccerNetGsrLabels,
)
from tactifoot_vision.enums import HomographyStatus

LOCSIM_TAU_METERS = 5.0


@dataclass(frozen=True, slots=True)
class HomographyRecord:
    sequence: str
    frame: int
    method: str
    status: str
    homography_3x3: tuple[tuple[float, float, float], ...] | None = None
    runtime_ms: float | None = None
    inliers: int | None = None
    source_artifact: str | None = None
    failure_reason: str | None = None

    @classmethod
    def available(
        cls,
        *,
        sequence: str,
        frame: int,
        method: str,
        homography_3x3: object,
        runtime_ms: float | None = None,
        inliers: int | None = None,
        source_artifact: str | None = None,
    ) -> "HomographyRecord":
        matrix = validate_homography_matrix(homography_3x3)
        return cls(
            sequence=sequence,
            frame=int(frame),
            method=method,
            status=HomographyStatus.AVAILABLE.value,
            homography_3x3=_matrix_tuple(matrix),
            runtime_ms=runtime_ms,
            inliers=inliers,
            source_artifact=source_artifact,
        )

    @classmethod
    def unavailable(
        cls,
        *,
        sequence: str,
        frame: int,
        method: str,
        failure_reason: str,
        runtime_ms: float | None = None,
        source_artifact: str | None = None,
    ) -> "HomographyRecord":
        return cls(
            sequence=sequence,
            frame=int(frame),
            method=method,
            status=HomographyStatus.UNAVAILABLE.value,
            runtime_ms=runtime_ms,
            source_artifact=source_artifact,
            failure_reason=failure_reason,
        )

    @property
    def matrix(self) -> NDArray[np.float64] | None:
        if self.homography_3x3 is None:
            return None
        return validate_homography_matrix(self.homography_3x3)

    @property
    def is_available(self) -> bool:
        if (
            self.status != HomographyStatus.AVAILABLE.value
            or self.homography_3x3 is None
        ):
            return False
        try:
            validate_homography_matrix(self.homography_3x3)
        except ValueError:
            return False
        return True


@dataclass(frozen=True, slots=True)
class ProjectionRecord:
    sequence: str
    frame: int
    track_id: int
    role: str | None
    method: str
    image_x: float
    image_y: float
    pitch_x_pred: float
    pitch_y_pred: float
    pitch_x_gt: float
    pitch_y_gt: float
    error_m: float


def validate_homography_matrix(matrix: object) -> NDArray[np.float64]:
    try:
        array = np.asarray(matrix, dtype=np.float64)
    except ValueError as exc:
        if not isinstance(matrix, np.ndarray):
            raise exc
        array = np.vstack(
            [np.asarray(row, dtype=np.float64) for row in matrix.tolist()]
        )
    if array.shape != (3, 3):
        raise ValueError(f"homography_3x3 must have shape (3, 3), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("homography_3x3 must contain only finite values")
    return array


def project_gsr_athletes(
    labels: SoccerNetGsrLabels, homographies: Sequence[HomographyRecord]
) -> tuple[ProjectionRecord, ...]:
    athletes_by_frame: dict[int, tuple[GsrAthleteAnnotation, ...]] = {
        frame.frame: labels.athletes_for_frame(frame.frame) for frame in labels.frames
    }
    records: list[ProjectionRecord] = []
    for homography in homographies:
        if homography.sequence != labels.sequence or not homography.is_available:
            continue
        matrix = homography.matrix
        if matrix is None:
            continue
        athletes = athletes_by_frame.get(homography.frame, ())
        for athlete in athletes:
            image_point = athlete.image_bottom_middle
            pitch_gt = athlete.pitch_bottom_middle
            if image_point is None or pitch_gt is None:
                continue
            pitch_pred = apply_homography_to_point(image_point, matrix)
            error = math.dist(pitch_pred, pitch_gt)
            records.append(
                ProjectionRecord(
                    sequence=labels.sequence,
                    frame=homography.frame,
                    track_id=athlete.track_id,
                    role=athlete.role,
                    method=homography.method,
                    image_x=image_point[0],
                    image_y=image_point[1],
                    pitch_x_pred=pitch_pred[0],
                    pitch_y_pred=pitch_pred[1],
                    pitch_x_gt=pitch_gt[0],
                    pitch_y_gt=pitch_gt[1],
                    error_m=error,
                )
            )
    return tuple(records)


def apply_homography_to_point(
    point: tuple[float, float], matrix: NDArray[np.float64]
) -> tuple[float, float]:
    homogeneous = matrix @ np.array([point[0], point[1], 1.0], dtype=np.float64)
    if homogeneous[2] == 0.0:
        raise ValueError("homography maps point to infinity")
    return float(homogeneous[0] / homogeneous[2]), float(
        homogeneous[1] / homogeneous[2]
    )


def summarize_homography_metrics(
    projections: Sequence[ProjectionRecord],
    homographies: Sequence[HomographyRecord] = (),
    *,
    expected_frames: Mapping[str, set[int]] | None = None,
) -> dict[str, dict[str, float]]:
    methods = sorted(
        {projection.method for projection in projections}
        | {homography.method for homography in homographies}
    )
    return {
        method: _summarize_method_metrics(
            method,
            projections,
            homographies,
            expected_frames=expected_frames,
            tau=LOCSIM_TAU_METERS,
        )
        for method in methods
    }


def summarize_metrics_by_sequence(
    projections: Sequence[ProjectionRecord],
    homographies: Sequence[HomographyRecord] = (),
    *,
    expected_frames: Mapping[str, set[int]] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    sequences = sorted(
        {projection.sequence for projection in projections}
        | {homography.sequence for homography in homographies}
    )
    output: dict[str, dict[str, dict[str, float]]] = {}
    for sequence in sequences:
        sequence_projections = [
            projection for projection in projections if projection.sequence == sequence
        ]
        sequence_homographies = [
            homography for homography in homographies if homography.sequence == sequence
        ]
        frames = None
        if expected_frames is not None:
            frames = {sequence: expected_frames.get(sequence, set())}
        output[sequence] = summarize_homography_metrics(
            sequence_projections,
            sequence_homographies,
            expected_frames=frames,
        )
    return output


def _summarize_method_metrics(
    method: str,
    projections: Sequence[ProjectionRecord],
    homographies: Sequence[HomographyRecord],
    *,
    expected_frames: Mapping[str, set[int]] | None,
    tau: float,
) -> dict[str, float]:
    method_projections = [
        projection for projection in projections if projection.method == method
    ]
    errors = np.array(
        [projection.error_m for projection in method_projections], dtype=np.float64
    )
    return {
        "median_error_m": _percentile(errors, 50.0),
        "mean_error_m": float(np.mean(errors)) if errors.size else 0.0,
        "p90_error_m": _percentile(errors, 90.0),
        "success@1m": _success_rate(errors, 1.0),
        "success@2m": _success_rate(errors, 2.0),
        "success@5m": _success_rate(errors, 5.0),
        "availability": _availability(method, homographies, expected_frames),
        "locsim_tau5": _locsim(errors, tau),
        "temporal_jitter": _temporal_jitter(method_projections),
        "projections": float(len(method_projections)),
    }


def _availability(
    method: str,
    homographies: Sequence[HomographyRecord],
    expected_frames: Mapping[str, set[int]] | None,
) -> float:
    method_homographies = [
        homography for homography in homographies if homography.method == method
    ]
    if expected_frames is None:
        return (
            1.0
            if any(homography.is_available for homography in method_homographies)
            else 0.0
        )
    expected = {
        (sequence, frame)
        for sequence, frames in expected_frames.items()
        for frame in frames
    }
    if not expected:
        return 0.0
    available = {
        (homography.sequence, homography.frame)
        for homography in method_homographies
        if homography.is_available
        and (homography.sequence, homography.frame) in expected
    }
    return len(available) / len(expected)


def _percentile(errors: NDArray[np.float64], q: float) -> float:
    return float(np.percentile(errors, q)) if errors.size else 0.0


def _success_rate(errors: NDArray[np.float64], threshold: float) -> float:
    return float(np.mean(errors <= threshold)) if errors.size else 0.0


def _locsim(errors: NDArray[np.float64], tau: float) -> float:
    if not errors.size:
        return 0.0
    values = np.exp(math.log(0.05) * np.square(errors) / (tau * tau))
    return float(np.mean(values))


def _temporal_jitter(projections: Sequence[ProjectionRecord]) -> float:
    points_by_track: dict[tuple[str, int], list[ProjectionRecord]] = {}
    for projection in projections:
        points_by_track.setdefault(
            (projection.sequence, projection.track_id), []
        ).append(projection)
    deltas: list[float] = []
    for track_points in points_by_track.values():
        ordered = sorted(track_points, key=lambda projection: projection.frame)
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if current.frame != previous.frame + 1:
                continue
            deltas.append(
                math.dist(
                    (previous.pitch_x_pred, previous.pitch_y_pred),
                    (current.pitch_x_pred, current.pitch_y_pred),
                )
            )
    return float(np.median(np.asarray(deltas, dtype=np.float64))) if deltas else 0.0


def _matrix_tuple(
    matrix: NDArray[np.float64],
) -> tuple[tuple[float, float, float], ...]:
    return tuple((float(row[0]), float(row[1]), float(row[2])) for row in matrix)


def projection_to_dict(record: ProjectionRecord) -> dict[str, object]:
    return {
        "sequence": record.sequence,
        "frame": record.frame,
        "track_id": record.track_id,
        "role": record.role,
        "method": record.method,
        "image_x": record.image_x,
        "image_y": record.image_y,
        "pitch_x_pred": record.pitch_x_pred,
        "pitch_y_pred": record.pitch_y_pred,
        "pitch_x_gt": record.pitch_x_gt,
        "pitch_y_gt": record.pitch_y_gt,
        "error_m": record.error_m,
    }


def homography_to_dict(record: HomographyRecord) -> dict[str, object]:
    return {
        "sequence": record.sequence,
        "frame": record.frame,
        "method": record.method,
        "status": record.status,
        "homography_3x3": record.homography_3x3,
        "runtime_ms": record.runtime_ms,
        "inliers": record.inliers,
        "source_artifact": record.source_artifact,
        "failure_reason": record.failure_reason,
    }


def homography_from_mapping(
    row: Mapping[str, Any],
    *,
    allowed_frames: Mapping[str, set[int]] | None = None,
) -> HomographyRecord:
    sequence = str(row["sequence"])
    frame = int(row["frame"])
    if allowed_frames is not None and frame not in allowed_frames.get(sequence, set()):
        raise ValueError(f"Frame {sequence}/{frame} is not in the allowed frame set")
    method = str(row["method"])
    status = str(row["status"])
    homography = row.get("homography_3x3")
    if status == HomographyStatus.AVAILABLE.value:
        return HomographyRecord.available(
            sequence=sequence,
            frame=frame,
            method=method,
            homography_3x3=_decode_homography(homography),
            runtime_ms=_optional_float(row.get("runtime_ms")),
            inliers=_optional_int(row.get("inliers")),
            source_artifact=_optional_str(row.get("source_artifact")),
        )
    return HomographyRecord.unavailable(
        sequence=sequence,
        frame=frame,
        method=method,
        failure_reason=_optional_str(row.get("failure_reason")) or "unavailable",
        runtime_ms=_optional_float(row.get("runtime_ms")),
        source_artifact=_optional_str(row.get("source_artifact")),
    )


def _decode_homography(value: object) -> object:
    if isinstance(value, str):
        import json

        return json.loads(value)
    return value


def _optional_str(value: object) -> str | None:
    return None if value is None or _is_nan(value) else str(value)


def _optional_float(value: object) -> float | None:
    if value is None or _is_nan(value):
        return None
    return float(str(value))


def _optional_int(value: object) -> int | None:
    if value is None or _is_nan(value):
        return None
    return int(float(str(value)))


def _is_nan(value: object) -> bool:
    return isinstance(value, float) and math.isnan(value)
