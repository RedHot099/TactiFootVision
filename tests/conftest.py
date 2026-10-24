"""Fixtures shared by several test modules."""

from pathlib import Path

import pytest

from tactifoot_vision.pipeline import POSTPROCESSES

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


@pytest.fixture
def postprocess_registry(monkeypatch):
    """``POSTPROCESSES``, with whatever a test registers removed afterwards."""
    monkeypatch.setattr(POSTPROCESSES, "_factories", dict(POSTPROCESSES._factories))
    return POSTPROCESSES


# ----------------------------------------- checkpoints of the -m model tests
# The files in models/ on omarchy; each trained one has a .json sidecar.
@pytest.fixture(scope="session")
def detector_weights() -> Path:
    """The football detector: YOLO11l at 1280, ball/goalkeeper/player/referee."""
    return MODELS_DIR / "detector_yolo11l_1280_dd2c.pt"


@pytest.fixture(scope="session")
def pitch_weights() -> Path:
    """The pitch keypoint model: YOLO11m pose at 960, 32 keypoints."""
    return MODELS_DIR / "pitch_yolo11m_pose_sn960.pt"


@pytest.fixture(scope="session")
def rfdetr_coco_weights() -> Path:
    """RF-DETR base with the COCO classes."""
    return MODELS_DIR / "rfdetr_base.pth"


@pytest.fixture(scope="session")
def rfdetr_sample_weights() -> Path:
    """RF-DETR base fine-tuned for one epoch on football_yolo_sample100: football
    classes and a recorded size, but no accuracy to speak of."""
    return MODELS_DIR / "rfdetr_smoketest_sample.pth"


@pytest.fixture(scope="session")
def rfdetr_football_weights() -> Path:
    """RF-DETR base fine-tuned on football_yolo, accurate enough to score."""
    path = MODELS_DIR / "football_rfdetr_base.pth"
    if not path.is_file():
        pytest.skip(f"no fine-tuned football RF-DETR checkpoint at {path}")
    return path
