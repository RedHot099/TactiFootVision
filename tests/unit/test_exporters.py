import numpy as np
import pandas as pd
import pytest

from tactifoot_vision.detection import FakeDetector
from tactifoot_vision.domain import Frame, PipelineResult
from tactifoot_vision.export import StatsBombExporter
from tactifoot_vision.export.timing import ExportManifestWriter
from tactifoot_vision.pipeline import InferencePipeline
from tactifoot_vision.tracking import FakeTracker


def test_pipeline_csv_and_mot_export_are_stateless(tmp_path) -> None:
    frames = [
        Frame(
            index=i,
            image=np.zeros((20, 20, 3), dtype=np.uint8),
            timestamp_seconds=i / 25,
        )
        for i in range(3)
    ]
    result = InferencePipeline(detector=FakeDetector(), tracker=FakeTracker()).run(
        frames
    )

    csv_artifact = result.to_csv(tmp_path / "pipeline.csv")
    mot_artifact = result.to_mot(tmp_path / "tracks.txt")

    df = pd.read_csv(csv_artifact.path)
    assert csv_artifact.rows == 6
    assert mot_artifact.rows == 6
    assert sorted(df["frame"].unique().tolist()) == [0, 1, 2]
    assert (tmp_path / "tracks.txt").read_text(encoding="utf-8").count("\n") == 6


def test_export_manifest_writer(tmp_path) -> None:
    artifact = ExportManifestWriter().write(
        artifacts=(),
        metrics={"frames": 1.0},
        path=tmp_path / "manifest.json",
    )

    assert artifact.path.is_file()
    assert artifact.format == "manifest_json"


def test_statsbomb_exporter_is_explicitly_not_native_export(tmp_path) -> None:
    with pytest.raises(NotImplementedError, match="Native StatsBomb360 export"):
        StatsBombExporter().write(PipelineResult(()), tmp_path / "statsbomb.csv")
