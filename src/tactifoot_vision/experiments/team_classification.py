import numpy as np
import pandas as pd
from numpy.typing import NDArray

from tactifoot_vision.config import ExperimentConfig, TeamAssignmentConfig
from tactifoot_vision.config.factories import build_pipeline
from tactifoot_vision.domain import (
    AdapterUnavailable,
    BBox,
    ExperimentReport,
    ExportArtifact,
    Track,
)
from tactifoot_vision.enums import TeamAssignmentCropMethod
from tactifoot_vision.io import read_frames
from tactifoot_vision.team_assignment import TeamAssigner
from tactifoot_vision.team_assignment.crops import crop_bbox
from tactifoot_vision.team_assignment.metrics import clustering_purity


class TeamClassificationExperimentRunner:
    def run(self, config: ExperimentConfig) -> ExperimentReport:
        if config.pipeline.paths.input is None:
            raise ValueError(
                "Team classification experiment requires pipeline.paths.input"
            )
        pipeline_config = config.pipeline.model_copy(deep=True)
        pipeline_config.team_assignment.enabled = False
        pipeline = build_pipeline(pipeline_config)
        frames = list(read_frames(config.pipeline.paths.input))
        if config.max_frames is not None:
            frames = frames[: config.max_frames]
        result = pipeline.run(frames)
        frame_by_index = {frame.index: frame for frame in frames}
        crops: list[NDArray[np.uint8]] = []
        crop_keys: list[tuple[int, int]] = []
        has_crop_by_key: dict[tuple[int, int], bool] = {}
        true_label_by_key: dict[tuple[int, int], int | None] = {}
        rows: list[dict[str, object]] = []
        team_config = config.pipeline.team_assignment
        for frame_result in result.frames:
            frame = frame_by_index[frame_result.frame_index]
            for track in frame_result.tracks:
                if track.class_name not in {"player", "goalkeeper"}:
                    continue
                key = (frame_result.frame_index, track.track_id)
                crop = _extract_crop(frame.image, track.bbox, team_config)
                has_crop = _is_valid_crop(crop)
                has_crop_by_key[key] = has_crop
                true_label = _team_label_from_track(track)
                true_label_by_key[key] = true_label
                if crop is None or not has_crop:
                    continue
                crops.append(crop)
                crop_keys.append(key)
        assigner = TeamAssigner.from_config(team_config)
        labels_by_key: dict[tuple[int, int], int] = {}
        metrics: dict[str, float] = {
            "frames": float(len(result.frames)),
            "samples": 0.0,
            "valid_crops": 0.0,
            "teams": 0.0,
        }
        if crops:
            labels = assigner.fit_predict(crops)
            labels_by_key = {
                key: int(label) for key, label in zip(crop_keys, labels, strict=True)
            }
            metrics["samples"] = float(len(labels))
            metrics["valid_crops"] = float(len(crops))
            metrics["teams"] = float(len(set(labels.tolist())))
            labeled_keys = [
                key for key in crop_keys if true_label_by_key.get(key) is not None
            ]
            if labeled_keys:
                metrics["labeled_samples"] = float(len(labeled_keys))
                metrics["purity"] = clustering_purity(
                    np.asarray(
                        [labels_by_key[key] for key in labeled_keys], dtype=np.int_
                    ),
                    np.asarray(
                        [true_label_by_key[key] for key in labeled_keys],
                        dtype=np.int_,
                    ),
                )
        for frame_result in result.frames:
            for track in frame_result.tracks:
                key = (frame_result.frame_index, track.track_id)
                team_id = labels_by_key.get(key)
                rows.append(
                    {
                        "frame": frame_result.frame_index,
                        "track_id": track.track_id,
                        "team_id": team_id if team_id is not None else track.team_id,
                        "true_team_id": true_label_by_key.get(key),
                        "has_crop": has_crop_by_key.get(key, False),
                        "class_name": track.class_name,
                        "crop_method": team_config.crop_method.value,
                        "embedding": team_config.embedding.value,
                        "reducer": team_config.reducer.value,
                        "clusterer": team_config.clusterer.value,
                        "crop_ratio": team_config.crop_ratio,
                    }
                )
        output_dir = config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "team_classification_metrics.csv"
        assignments_path = output_dir / "team_classification_assignments.csv"
        pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
        pd.DataFrame(rows).to_csv(assignments_path, index=False)
        return ExperimentReport(
            name=config.name,
            artifacts=(
                ExportArtifact(metrics_path, "team_classification_metrics", 1),
                ExportArtifact(
                    assignments_path,
                    "team_classification_assignments",
                    len(rows),
                ),
            ),
            metrics=metrics,
        )


def _extract_crop(
    image: NDArray[np.uint8],
    bbox: BBox,
    config: TeamAssignmentConfig,
) -> NDArray[np.uint8] | None:
    if config.crop_method == TeamAssignmentCropMethod.CENTER:
        return crop_bbox(image, bbox, ratio=config.crop_ratio)
    if config.crop_method == TeamAssignmentCropMethod.OPENCV_MASK:
        from tactifoot_vision.team_assignment.opencv_masks import opencv_mask_crop

        return opencv_mask_crop(image, bbox)
    if config.crop_method == TeamAssignmentCropMethod.SAM2_MASK:
        raise AdapterUnavailable(
            "team_assignment.crop_method=sam2_mask requires a configured SAM2 cropper; "
            "the experiment runner does not silently fall back to center crops."
        )
    raise ValueError(f"Unsupported team assignment crop method: {config.crop_method}")


def _team_label_from_track(track: Track) -> int | None:
    value = track.data.get("team_label", track.data.get("team_id"))
    return int(value) if value is not None else None


def _is_valid_crop(crop: NDArray[np.uint8] | None) -> bool:
    return crop is not None and crop.size > 0 and crop.ndim == 3 and crop.shape[2] == 3
