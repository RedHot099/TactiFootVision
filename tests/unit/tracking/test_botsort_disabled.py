import pytest

from tactifoot_vision.config import PipelineConfig, TrackingConfig
from tactifoot_vision.config.factories import build_tracker
from tactifoot_vision.domain import AdapterUnavailable
from tactifoot_vision.enums import TrackingBackend
from tactifoot_vision.tracking import BoTSORTTracker


def test_botsort_tracker_is_explicitly_unavailable() -> None:
    with pytest.raises(AdapterUnavailable, match="BoTSORT tracking is disabled"):
        BoTSORTTracker()


def test_botsort_factory_is_explicitly_unavailable() -> None:
    config = PipelineConfig(tracking=TrackingConfig(backend=TrackingBackend.BOTSORT))

    with pytest.raises(AdapterUnavailable, match="BoTSORT tracking is disabled"):
        build_tracker(config)
