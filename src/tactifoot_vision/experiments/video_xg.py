import json
from pathlib import Path

from tactifoot_vision.ball import LinearBallTrajectoryReconstructor
from tactifoot_vision.config import ExperimentConfig
from tactifoot_vision.config.factories import build_pipeline
from tactifoot_vision.datasets import iter_sequence_dirs
from tactifoot_vision.domain import ExperimentReport, ExportArtifact, PipelineResult
from tactifoot_vision.enums import ShotDetectorKind
from tactifoot_vision.export.xg import write_xg_shots_csv, write_xg_summary_json
from tactifoot_vision.shots import (
    KinematicShotDetector,
    MetadataShotDetector,
    is_shot_like_action,
    read_soccernet_action_metadata,
)
from tactifoot_vision.shots.interfaces import ShotDetector
from tactifoot_vision.shots.soccernet import SoccerNetActionMetadata
from tactifoot_vision.xg import GeometryXgEstimator, VideoXgEstimator, VideoXgSummary


class VideoXgExperimentRunner:
    def run(self, config: ExperimentConfig) -> ExperimentReport:
        artifacts: list[ExportArtifact] = []
        summaries: list[VideoXgSummary] = []
        if config.soccernet_root is not None:
            sequence_dirs = iter_sequence_dirs(config.soccernet_root)
            if config.sequence_names is not None:
                wanted = set(config.sequence_names)
                sequence_dirs = [
                    sequence for sequence in sequence_dirs if sequence.name in wanted
                ]
            if config.max_sequences is not None:
                sequence_dirs = sequence_dirs[: config.max_sequences]
            for sequence_dir in sequence_dirs:
                pipeline_config = config.pipeline.model_copy(deep=True)
                pipeline_config.paths.input = sequence_dir / "img1"
                result = build_pipeline(pipeline_config).run_video(
                    sequence_dir / "img1", max_frames=config.max_frames
                )
                metadata = read_soccernet_action_metadata(sequence_dir)
                summary = self._summarize_result(
                    result,
                    config=config,
                    metadata=metadata,
                    group_id=_group_id(metadata, config),
                )
                summaries.append(summary)
                sequence_output = config.output_dir / sequence_dir.name
                artifacts.extend(
                    _write_summary_artifacts(config, summary, sequence_output)
                )
        else:
            if config.pipeline.paths.input is None:
                raise ValueError("Experiment pipeline.paths.input is required")
            result = build_pipeline(config.pipeline).run_video(
                config.pipeline.paths.input,
                max_frames=config.max_frames,
            )
            summary = self._summarize_result(
                result,
                config=config,
                metadata=None,
                group_id=config.name,
            )
            summaries.append(summary)
            artifacts.extend(
                _write_summary_artifacts(config, summary, config.output_dir)
            )
        aggregate = _aggregate_summaries(summaries)
        if config.video_xg.write_summary_json:
            aggregate_path = config.output_dir / "video_xg_summary.json"
            aggregate_path.parent.mkdir(parents=True, exist_ok=True)
            aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
            artifacts.append(
                ExportArtifact(
                    path=aggregate_path,
                    format="video_xg_aggregate_json",
                    rows=len(summaries),
                )
            )
        return ExperimentReport(config.name, tuple(artifacts), aggregate)

    def _summarize_result(
        self,
        result: PipelineResult,
        *,
        config: ExperimentConfig,
        metadata: SoccerNetActionMetadata | None,
        group_id: str | None,
    ) -> VideoXgSummary:
        video_xg = config.video_xg
        estimator = VideoXgEstimator(
            ball_reconstructor=LinearBallTrajectoryReconstructor(
                max_speed_pixels_per_frame=video_xg.ball.max_speed_pixels_per_frame
            ),
            shot_detector=_build_shot_detector(config, metadata),
            xg_estimator=GeometryXgEstimator(penalty_xg=video_xg.xg.penalty_xg),
            image_width=video_xg.xg.image_width,
            image_height=video_xg.xg.image_height,
            attacking_goal_x=video_xg.xg.attacking_goal_x,
        )
        return estimator.run(result, group_id=group_id)


def _build_shot_detector(
    config: ExperimentConfig, metadata: SoccerNetActionMetadata | None
) -> ShotDetector:
    shot_config = config.video_xg.shots
    if shot_config.use_soccernet_metadata and metadata is not None:
        action_frames = (
            (metadata.action_frame,)
            if is_shot_like_action(metadata.action_class)
            else ()
        )
        return MetadataShotDetector(
            action_frames=action_frames,
            action_class=metadata.action_class,
            window_before=shot_config.window_before,
            window_after=shot_config.window_after,
        )
    if shot_config.kind == ShotDetectorKind.METADATA:
        return MetadataShotDetector(
            action_frames=(),
            window_before=shot_config.window_before,
            window_after=shot_config.window_after,
        )
    return KinematicShotDetector(
        window_before=shot_config.window_before,
        window_after=shot_config.window_after,
        max_candidates=shot_config.max_candidates,
        min_speed_pixels_per_frame=shot_config.min_speed_pixels_per_frame,
    )


def _group_id(metadata: SoccerNetActionMetadata, config: ExperimentConfig) -> str:
    if config.video_xg.group_by_game_id and metadata.game_id:
        return metadata.game_id
    return metadata.sequence_name


def _write_summary_artifacts(
    config: ExperimentConfig, summary: VideoXgSummary, output_dir: Path
) -> list[ExportArtifact]:
    path = Path(output_dir)
    artifacts = []
    if config.video_xg.write_shots_csv:
        artifacts.append(write_xg_shots_csv(summary, path / "shots.csv"))
    if config.video_xg.write_summary_json:
        artifacts.append(write_xg_summary_json(summary, path / "xg_summary.json"))
    return artifacts


def _aggregate_summaries(summaries: list[VideoXgSummary]) -> dict[str, float]:
    total_shots = sum(summary.shot_count for summary in summaries)
    total_xg = sum(summary.total_xg for summary in summaries)
    return {
        "sequences": float(len(summaries)),
        "shots": float(total_shots),
        "total_xg": float(total_xg),
        "mean_xg_per_shot": float(total_xg / total_shots) if total_shots else 0.0,
    }
