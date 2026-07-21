import json

from tactifoot_vision.config import ExperimentConfig
from tactifoot_vision.config.factories import build_pipeline
from tactifoot_vision.datasets import iter_sequence_dirs
from tactifoot_vision.domain import ExperimentReport, ExportArtifact

COUNT_KEYS = {"tp", "fp", "fn", "matches", "frames_evaluated", "id_switches"}


class SoccerNetDetectionTrackingExperimentRunner:
    def run(self, config: ExperimentConfig) -> ExperimentReport:
        if config.soccernet_root is None:
            raise ValueError("soccernet_root is required")
        sequence_dirs = iter_sequence_dirs(config.soccernet_root)
        if config.sequence_names is not None:
            wanted = set(config.sequence_names)
            sequence_dirs = [
                sequence for sequence in sequence_dirs if sequence.name in wanted
            ]
        if config.max_sequences is not None:
            sequence_dirs = sequence_dirs[: config.max_sequences]
        artifacts: list[ExportArtifact] = []
        metrics_by_sequence: dict[str, dict[str, float]] = {}
        for sequence_dir in sequence_dirs:
            pipeline_config = config.pipeline.model_copy(deep=True)
            pipeline_config.paths.input = sequence_dir / "img1"
            pipeline = build_pipeline(pipeline_config)
            result = pipeline.run_video(
                sequence_dir / "img1", max_frames=config.max_frames
            )
            sequence_output = config.output_dir / sequence_dir.name
            sequence_output.mkdir(parents=True, exist_ok=True)
            csv_artifact = result.to_csv(sequence_output / "pipeline.csv")
            artifacts.append(csv_artifact)
            mot_artifact = result.to_mot(sequence_output / "mot.txt")
            if config.write_mot:
                artifacts.append(mot_artifact)
            from tactifoot_vision.evaluation.mot import evaluate_tracking_files

            metrics_by_sequence[sequence_dir.name] = evaluate_tracking_files(
                mot_artifact.path,
                sequence_dir / "gt" / "gt.txt",
                iou_threshold=config.iou_threshold,
            )
        aggregate = _aggregate(metrics_by_sequence)
        if config.write_metrics_json:
            metrics_path = config.output_dir / "metrics.json"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(
                json.dumps(metrics_by_sequence, indent=2), encoding="utf-8"
            )
            artifacts.append(
                ExportArtifact(metrics_path, "metrics_json", len(metrics_by_sequence))
            )
        return ExperimentReport(config.name, tuple(artifacts), aggregate)


def _aggregate(metrics_by_sequence: dict[str, dict[str, float]]) -> dict[str, float]:
    if not metrics_by_sequence:
        return {"sequences": 0.0}
    aggregate = {"sequences": float(len(metrics_by_sequence))}
    for key in COUNT_KEYS:
        aggregate[key] = float(
            sum(metrics.get(key, 0.0) for metrics in metrics_by_sequence.values())
        )
    tp = aggregate["tp"]
    fp = aggregate["fp"]
    fn = aggregate["fn"]
    matches = aggregate["matches"]
    id_switches = aggregate["id_switches"]
    aggregate["precision"] = tp / (tp + fp) if tp + fp else 0.0
    aggregate["recall"] = tp / (tp + fn) if tp + fn else 0.0
    aggregate["f1"] = (
        2
        * aggregate["precision"]
        * aggregate["recall"]
        / (aggregate["precision"] + aggregate["recall"])
        if aggregate["precision"] + aggregate["recall"]
        else 0.0
    )
    aggregate["id_switch_rate"] = id_switches / matches if matches else 0.0
    iou_weight = sum(
        metrics.get("matches", 0.0) for metrics in metrics_by_sequence.values()
    )
    aggregate["mean_iou"] = (
        sum(
            metrics.get("mean_iou", 0.0) * metrics.get("matches", 0.0)
            for metrics in metrics_by_sequence.values()
        )
        / iou_weight
        if iou_weight
        else 0.0
    )
    thresholds = {
        metrics.get("iou_threshold") for metrics in metrics_by_sequence.values()
    }
    if len(thresholds) == 1:
        threshold = next(iter(thresholds))
        if threshold is not None:
            aggregate["iou_threshold"] = float(threshold)
    for key in ("avg_track_length", "median_track_length", "max_track_length"):
        values = [
            metrics[key] for metrics in metrics_by_sequence.values() if key in metrics
        ]
        if values:
            aggregate[f"macro_{key}"] = float(sum(values) / len(values))
    return aggregate
