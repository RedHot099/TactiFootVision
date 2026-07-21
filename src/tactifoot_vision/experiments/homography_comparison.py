import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from tactifoot_vision.config import ExperimentConfig
from tactifoot_vision.config.factories import build_projector
from tactifoot_vision.datasets import (
    GsrAthleteAnnotation,
    GsrFrame,
    SoccerNetGsrLabels,
    iter_gsr_sequence_dirs,
    read_gsr_labels,
)
from tactifoot_vision.domain import (
    AdapterUnavailable,
    BBox,
    ConfigurationError,
    ExperimentReport,
    ExportArtifact,
    Frame,
    ModelArtifactNotFound,
    Track,
    TrackSet,
)
from tactifoot_vision.enums import HomographyMethod
from tactifoot_vision.evaluation.homography import (
    HomographyRecord,
    ProjectionRecord,
    project_gsr_athletes,
    summarize_homography_metrics,
    summarize_metrics_by_sequence,
)
from tactifoot_vision.export.homography import (
    read_homographies,
    write_homographies_parquet,
    write_metrics_json,
    write_projections_parquet,
)


class HomographyComparisonRunner:
    def run(self, config: ExperimentConfig) -> ExperimentReport:
        if config.soccernet_root is None:
            raise ValueError("homography comparison requires soccernet_root")
        sequence_dirs = iter_gsr_sequence_dirs(
            config.soccernet_root, split=config.homography_comparison.split
        )
        if config.sequence_names is not None:
            wanted = set(config.sequence_names)
            sequence_dirs = [path for path in sequence_dirs if path.name in wanted]
        if config.max_sequences is not None:
            sequence_dirs = sequence_dirs[: config.max_sequences]
        sequence_entries = [
            (path, _limit_frames(read_gsr_labels(path), config.max_frames))
            for path in sequence_dirs
        ]
        labels_by_sequence = [labels for _, labels in sequence_entries]
        expected_frames = {
            labels.sequence: {frame.frame for frame in labels.frames}
            for labels in labels_by_sequence
        }
        homographies = _compute_internal_homographies(
            sequence_entries,
            methods=config.homography_comparison.methods,
            config=config,
        )
        for artifact in config.homography_comparison.external_homographies:
            homographies.extend(
                read_homographies(artifact, allowed_frames=expected_frames)
            )
        projections: list[ProjectionRecord] = []
        for labels in labels_by_sequence:
            sequence_homographies = [
                record for record in homographies if record.sequence == labels.sequence
            ]
            projections.extend(project_gsr_athletes(labels, sequence_homographies))
        metrics = {
            "per_method": summarize_homography_metrics(
                projections,
                homographies,
                expected_frames=expected_frames,
            ),
            "per_sequence": summarize_metrics_by_sequence(
                projections,
                homographies,
                expected_frames=expected_frames,
            ),
        }
        metrics["confidence_intervals"] = _bootstrap_confidence_intervals(
            projections,
            iterations=config.homography_comparison.confidence_iterations,
        )
        metrics["ranking"] = _rank_methods(
            metrics["per_method"], config.homography_comparison.ranking_weights
        )
        artifacts = _write_artifacts(
            output_dir=config.output_dir,
            homographies=homographies,
            projections=projections,
            metrics=metrics,
        )
        return ExperimentReport(
            config.name, tuple(artifacts), _flat_report_metrics(metrics)
        )


def _compute_internal_homographies(
    sequence_entries: list[tuple[Path, SoccerNetGsrLabels]],
    *,
    methods: tuple[HomographyMethod, ...],
    config: ExperimentConfig,
) -> list[HomographyRecord]:
    homographies: list[HomographyRecord] = []
    for method in methods:
        if method == HomographyMethod.CURRENT_YOLOPOSE_7PT:
            homographies.extend(_current_yolopose_baseline(sequence_entries, config))
        elif method == HomographyMethod.ORACLE_GSR_LINES_RANSAC:
            labels_by_sequence = [labels for _, labels in sequence_entries]
            homographies.extend(_oracle_from_gsr_athletes(labels_by_sequence))
    return homographies


def _current_yolopose_baseline(
    sequence_entries: list[tuple[Path, SoccerNetGsrLabels]], config: ExperimentConfig
) -> list[HomographyRecord]:
    if not config.pipeline.projection.enabled:
        return _unavailable_current_records(
            [labels for _, labels in sequence_entries],
            "pipeline.projection.enabled is false",
        )
    records: list[HomographyRecord] = []
    for sequence_dir, labels in sequence_entries:
        try:
            projector = build_projector(config.pipeline)
        except (
            AdapterUnavailable,
            ConfigurationError,
            FileNotFoundError,
            ModelArtifactNotFound,
            ValueError,
        ) as exc:
            records.extend(_unavailable_current_records([labels], str(exc)))
            continue
        if projector is None:
            records.extend(
                _unavailable_current_records([labels], "PitchProjector is disabled")
            )
            continue
        for frame in labels.frames:
            image_path = _frame_image_path(sequence_dir, frame)
            if image_path is None:
                records.append(
                    _unavailable_current_record(
                        labels.sequence,
                        frame.frame,
                        "frame image not found",
                    )
                )
                continue
            import cv2

            image = cv2.imread(str(image_path))
            if image is None:
                records.append(
                    _unavailable_current_record(
                        labels.sequence,
                        frame.frame,
                        "frame image could not be read",
                    )
                )
                continue
            start = time.perf_counter()
            projection = projector.project(
                frame=Frame(
                    index=frame.frame,
                    image=np.asarray(image, dtype=np.uint8),
                    path=image_path,
                ),
                keypoints=None,
                tracks=_tracks_from_athletes(labels.athletes_for_frame(frame.frame)),
            )
            runtime_ms = (time.perf_counter() - start) * 1000.0
            if projection.homography is None:
                records.append(
                    HomographyRecord.unavailable(
                        sequence=labels.sequence,
                        frame=frame.frame,
                        method=HomographyMethod.CURRENT_YOLOPOSE_7PT.value,
                        failure_reason=f"PitchProjector returned {projection.status}",
                        runtime_ms=runtime_ms,
                    )
                )
                continue
            records.append(
                HomographyRecord.available(
                    sequence=labels.sequence,
                    frame=frame.frame,
                    method=HomographyMethod.CURRENT_YOLOPOSE_7PT.value,
                    homography_3x3=projection.homography,
                    runtime_ms=runtime_ms,
                    inliers=None,
                )
            )
    return records


def _unavailable_current_records(
    labels_by_sequence: list[SoccerNetGsrLabels], failure_reason: str
) -> list[HomographyRecord]:
    records: list[HomographyRecord] = []
    for labels in labels_by_sequence:
        for frame in labels.frames:
            records.append(
                _unavailable_current_record(
                    labels.sequence, frame.frame, failure_reason
                )
            )
    return records


def _unavailable_current_record(
    sequence: str, frame: int, failure_reason: str
) -> HomographyRecord:
    return HomographyRecord.unavailable(
        sequence=sequence,
        frame=frame,
        method=HomographyMethod.CURRENT_YOLOPOSE_7PT.value,
        failure_reason=failure_reason,
    )


def _frame_image_path(sequence_dir: Path, frame: GsrFrame) -> Path | None:
    candidates = []
    if frame.file_name is not None:
        candidates.extend(
            [
                sequence_dir / frame.file_name,
                sequence_dir / "img1" / frame.file_name,
                sequence_dir / "images" / frame.file_name,
            ]
        )
    candidates.extend(
        [
            sequence_dir / f"{frame.frame:06d}.jpg",
            sequence_dir / "img1" / f"{frame.frame:06d}.jpg",
            sequence_dir / "images" / f"{frame.frame:06d}.jpg",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _tracks_from_athletes(athletes: tuple[GsrAthleteAnnotation, ...]) -> TrackSet:
    tracks: list[Track] = []
    for athlete in athletes:
        if athlete.bbox_image is None:
            continue
        bbox = athlete.bbox_image
        role = athlete.role or "player"
        tracks.append(
            Track(
                track_id=athlete.track_id,
                bbox=BBox(bbox.x, bbox.y, bbox.x + bbox.w, bbox.y + bbox.h),
                class_id=2,
                class_name=role,
                data={"jersey": athlete.jersey, "team": athlete.team},
            )
        )
    return TrackSet(tuple(tracks))


def _oracle_from_gsr_athletes(
    labels_by_sequence: list[SoccerNetGsrLabels],
) -> list[HomographyRecord]:
    records: list[HomographyRecord] = []
    for labels in labels_by_sequence:
        for frame in labels.frames:
            start = time.perf_counter()
            athletes = labels.athletes_for_frame(frame.frame)
            source: list[tuple[float, float]] = []
            target: list[tuple[float, float]] = []
            for athlete in athletes:
                image_point = athlete.image_bottom_middle
                pitch_point = athlete.pitch_bottom_middle
                if image_point is None or pitch_point is None:
                    continue
                source.append(image_point)
                target.append(pitch_point)
            matrix, inliers = _estimate_homography(source, target)
            runtime_ms = (time.perf_counter() - start) * 1000.0
            if matrix is None:
                records.append(
                    HomographyRecord.unavailable(
                        sequence=labels.sequence,
                        frame=frame.frame,
                        method=HomographyMethod.ORACLE_GSR_LINES_RANSAC.value,
                        failure_reason="fewer than four valid GT image/pitch correspondences",
                        runtime_ms=runtime_ms,
                    )
                )
                continue
            records.append(
                HomographyRecord.available(
                    sequence=labels.sequence,
                    frame=frame.frame,
                    method=HomographyMethod.ORACLE_GSR_LINES_RANSAC.value,
                    homography_3x3=matrix,
                    runtime_ms=runtime_ms,
                    inliers=inliers,
                )
            )
    return records


def _estimate_homography(
    source: list[tuple[float, float]], target: list[tuple[float, float]]
) -> tuple[np.ndarray | None, int | None]:
    if len(source) < 4 or len(target) < 4:
        return None, None
    import cv2

    matrix, inlier_mask = cv2.findHomography(
        np.asarray(source, dtype=np.float32),
        np.asarray(target, dtype=np.float32),
        cv2.RANSAC,
    )
    if matrix is None:
        return None, None
    inliers = int(inlier_mask.sum()) if inlier_mask is not None else None
    return np.asarray(matrix, dtype=np.float64), inliers


def _write_artifacts(
    *,
    output_dir: Path,
    homographies: list[HomographyRecord],
    projections: list[ProjectionRecord],
    metrics: dict[str, object],
) -> list[ExportArtifact]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = [
        write_homographies_parquet(homographies, output_dir / "homographies.parquet"),
        write_projections_parquet(projections, output_dir / "projections.parquet"),
        write_metrics_json(metrics, output_dir / "metrics.json"),
    ]
    artifacts.append(_write_report_markdown(metrics, output_dir / "report.md"))
    artifacts.append(_write_ranking_csv(metrics, output_dir / "ranking.csv"))
    failure_dir = output_dir / "failure_cases"
    failure_dir.mkdir(parents=True, exist_ok=True)
    return artifacts


def _write_report_markdown(metrics: dict[str, object], path: Path) -> ExportArtifact:
    ranking = metrics.get("ranking", [])
    lines = [
        "# Homography Comparison",
        "",
        "Ranking uses normalized median error, p90 error, success@2m, availability, and temporal jitter.",
        "",
        "| Rank | Method | Score | Median Error | P90 Error | Success@2m | Availability |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    if isinstance(ranking, list):
        for index, item in enumerate(ranking, start=1):
            if not isinstance(item, dict):
                continue
            lines.append(
                "| "
                f"{index} | {item['method']} | {item['score']:.6f} | "
                f"{item['median_error_m']:.6f} | {item['p90_error_m']:.6f} | "
                f"{item['success@2m']:.6f} | {item['availability']:.6f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rows = len(ranking) if isinstance(ranking, list) else None
    return ExportArtifact(path=path, format="homography_report_markdown", rows=rows)


def _write_ranking_csv(metrics: dict[str, object], path: Path) -> ExportArtifact:
    import csv

    ranking = metrics.get("ranking", [])
    rows = (
        [row for row in ranking if isinstance(row, dict)]
        if isinstance(ranking, list)
        else []
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_ranking_columns())
        writer.writeheader()
        writer.writerows(rows)
    return ExportArtifact(path=path, format="homography_ranking_csv", rows=len(rows))


def _rank_methods(
    per_method: object, weights: dict[str, float]
) -> list[dict[str, float | str]]:
    if not isinstance(per_method, dict):
        return []
    normalized = _normalize_metrics(per_method)
    ranking: list[dict[str, float | str]] = []
    for method, metrics in per_method.items():
        if not isinstance(metrics, dict):
            continue
        projections = float(metrics.get("projections", 0.0))
        availability = float(metrics.get("availability", 0.0))
        score = _ranking_score(method, metrics, normalized, weights)
        ranking.append(
            {
                "method": str(method),
                "score": float(score),
                "median_error_m": float(metrics.get("median_error_m", 0.0)),
                "p90_error_m": float(metrics.get("p90_error_m", 0.0)),
                "success@2m": float(metrics.get("success@2m", 0.0)),
                "availability": availability,
                "temporal_jitter": float(metrics.get("temporal_jitter", 0.0)),
                "rankable": float(projections > 0.0 and availability > 0.0),
            }
        )
    return sorted(
        ranking,
        key=lambda row: (
            1.0 - _as_float(row["rankable"]),
            _as_float(row["score"]),
            str(row["method"]),
        ),
    )


def _ranking_score(
    method: object,
    metrics: dict[object, object],
    normalized: dict[object, dict[str, float]],
    weights: dict[str, float],
) -> float:
    projections = _as_float(metrics.get("projections", 0.0))
    availability = _as_float(metrics.get("availability", 0.0))
    if projections <= 0.0 or availability <= 0.0:
        return 1.0
    return float(
        weights.get("median_error_m", 0.0) * normalized[method]["median_error_m"]
        + weights.get("p90_error_m", 0.0) * normalized[method]["p90_error_m"]
        + weights.get("success@2m", 0.0)
        * (1.0 - _as_float(metrics.get("success@2m", 0.0)))
        + weights.get("availability", 0.0) * (1.0 - availability)
        + weights.get("temporal_jitter", 0.0) * normalized[method]["temporal_jitter"]
    )


def _normalize_metrics(
    per_method: dict[object, object],
) -> dict[object, dict[str, float]]:
    metrics_to_normalize = ("median_error_m", "p90_error_m", "temporal_jitter")
    values_by_metric: dict[str, list[float]] = defaultdict(list)
    for metrics in per_method.values():
        if not isinstance(metrics, dict):
            continue
        for name in metrics_to_normalize:
            values_by_metric[name].append(_as_float(metrics.get(name, 0.0)))
    min_max = {
        name: (min(values), max(values)) if values else (0.0, 0.0)
        for name, values in values_by_metric.items()
    }
    normalized: dict[object, dict[str, float]] = {}
    for method, metrics in per_method.items():
        normalized[method] = {}
        if not isinstance(metrics, dict):
            continue
        for name in metrics_to_normalize:
            low, high = min_max[name]
            value = _as_float(metrics.get(name, 0.0))
            normalized[method][name] = (
                0.0 if high == low else (value - low) / (high - low)
            )
    return normalized


def _as_float(value: object) -> float:
    return float(str(value))


def _bootstrap_confidence_intervals(
    projections: list[ProjectionRecord], *, iterations: int
) -> dict[str, dict[str, dict[str, float]]]:
    if iterations <= 0:
        return {}
    errors_by_method_sequence: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for projection in projections:
        errors_by_method_sequence[projection.method][projection.sequence].append(
            projection.error_m
        )
    rng = np.random.default_rng(0)
    output: dict[str, dict[str, dict[str, float]]] = {}
    for method, errors_by_sequence in errors_by_method_sequence.items():
        sequences = list(errors_by_sequence)
        if not sequences:
            continue
        samples: list[float] = []
        for _ in range(iterations):
            sampled_sequences = rng.choice(sequences, size=len(sequences), replace=True)
            sampled_errors = [
                error
                for sequence in sampled_sequences
                for error in errors_by_sequence[str(sequence)]
            ]
            if sampled_errors:
                samples.append(float(statistics.median(sampled_errors)))
        if not samples:
            continue
        output[method] = {
            "median_error_m": {
                "low": float(np.percentile(samples, 2.5)),
                "high": float(np.percentile(samples, 97.5)),
            }
        }
    return output


def _limit_frames(
    labels: SoccerNetGsrLabels, max_frames: int | None
) -> SoccerNetGsrLabels:
    if max_frames is None:
        return labels
    kept_frames = labels.frames[:max_frames]
    kept_frame_numbers = {frame.frame for frame in kept_frames}
    return SoccerNetGsrLabels(
        sequence=labels.sequence,
        version=labels.version,
        frames=kept_frames,
        athletes=tuple(
            athlete
            for athlete in labels.athletes
            if athlete.frame in kept_frame_numbers
        ),
        lines=tuple(line for line in labels.lines if line.frame in kept_frame_numbers),
    )


def _flat_report_metrics(metrics: dict[str, object]) -> dict[str, float]:
    ranking = metrics.get("ranking", [])
    if not isinstance(ranking, list) or not ranking:
        return {}
    top = ranking[0]
    if not isinstance(top, dict):
        return {}
    return {
        "best_score": float(top["score"]),
        "best_median_error_m": float(top["median_error_m"]),
        "best_p90_error_m": float(top["p90_error_m"]),
    }


def _ranking_columns() -> list[str]:
    return [
        "method",
        "score",
        "median_error_m",
        "p90_error_m",
        "success@2m",
        "availability",
        "temporal_jitter",
        "rankable",
    ]
