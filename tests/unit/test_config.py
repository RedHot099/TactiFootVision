import pytest

from tactifoot_vision.config import PipelineConfig, build_pipeline, load_pipeline_config
from tactifoot_vision.domain import ConfigurationError
from tactifoot_vision.enums import DetectionBackend, TrackingBackend


def test_python_and_yaml_config_build_same_pipeline_type() -> None:
    python_pipeline = build_pipeline(PipelineConfig())
    yaml_pipeline = build_pipeline(
        load_pipeline_config("configs/pipeline/fake_bytetrack.yaml")
    )

    assert type(python_pipeline).__name__ == type(yaml_pipeline).__name__


def test_yaml_backend_values_parse_to_enums() -> None:
    config = load_pipeline_config("configs/pipeline/fake_bytetrack.yaml")

    assert config.detection.backend is DetectionBackend.FAKE
    assert config.tracking.backend is TrackingBackend.FAKE


def test_build_pipeline_rejects_unfitted_team_assignment_config() -> None:
    config = PipelineConfig()
    config.team_assignment.enabled = True

    with pytest.raises(ConfigurationError, match="fitted TeamAssigner"):
        build_pipeline(config)
