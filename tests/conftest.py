"""Fixtures shared by several test modules."""

import pytest

from tactifoot_vision.pipeline import POSTPROCESSES


@pytest.fixture
def postprocess_registry(monkeypatch):
    """``POSTPROCESSES``, with whatever a test registers removed afterwards."""
    monkeypatch.setattr(POSTPROCESSES, "_factories", dict(POSTPROCESSES._factories))
    return POSTPROCESSES
