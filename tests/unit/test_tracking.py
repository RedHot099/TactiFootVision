from pathlib import Path

import pytest

from tactifoot_vision.config import SAM2Config
from tactifoot_vision.domain import ModelArtifactNotFound
from tactifoot_vision.tracking import SAM2Tracker


def test_sam2_requires_checkpoint_and_config(tmp_path: Path) -> None:
    config = SAM2Config(
        checkpoint=tmp_path / "missing.pt",
        model_config_path=tmp_path / "missing.yaml",
    )

    with pytest.raises(ModelArtifactNotFound):
        SAM2Tracker(config)
