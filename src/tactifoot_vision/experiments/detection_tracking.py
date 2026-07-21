from tactifoot_vision.config import ExperimentConfig
from tactifoot_vision.config.factories import build_pipeline
from tactifoot_vision.domain import ExperimentReport
from tactifoot_vision.evaluation import summarize_tracking


class DetectionTrackingExperimentRunner:
    def run(self, config: ExperimentConfig) -> ExperimentReport:
        if config.soccernet_root is not None:
            from tactifoot_vision.experiments.soccernet_detection_tracking import (
                SoccerNetDetectionTrackingExperimentRunner,
            )

            return SoccerNetDetectionTrackingExperimentRunner().run(config)
        if config.pipeline.paths.input is None:
            raise ValueError("Experiment pipeline.paths.input is required")
        pipeline = build_pipeline(config.pipeline)
        result = pipeline.run_video(
            config.pipeline.paths.input, max_frames=config.max_frames
        )
        artifacts = []
        if config.pipeline.export.pipeline_csv is not None:
            artifacts.append(result.to_csv(config.pipeline.export.pipeline_csv))
        if config.pipeline.export.mot is not None:
            artifacts.append(result.to_mot(config.pipeline.export.mot))
        return ExperimentReport(
            name=config.name,
            artifacts=tuple(artifacts),
            metrics=summarize_tracking(result),
        )
