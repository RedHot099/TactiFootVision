"""The notebook builder (``notebooks/build/build.py``) without running a kernel."""

import importlib.util
from pathlib import Path

import nbclient.exceptions
import nbformat
import pytest

BUILD_DIR = Path(__file__).resolve().parents[1] / "notebooks" / "build"


@pytest.fixture
def build(monkeypatch):
    monkeypatch.syspath_prepend(str(BUILD_DIR))
    spec = importlib.util.spec_from_file_location(
        "notebook_build", BUILD_DIR / "build.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "error",
    [
        nbclient.exceptions.CellExecutionError("cell failed", "", "ValueError"),
        nbclient.exceptions.CellTimeoutError("cell timed out"),
        nbclient.exceptions.DeadKernelError("kernel died"),
    ],
    ids=type,
)
def test_a_failed_timed_out_or_dead_notebook_is_reported_not_raised(
    build, monkeypatch, error
):
    node = nbformat.v4.new_notebook()
    node.cells = [nbformat.v4.new_code_cell("1")]

    class Client:
        def __init__(self, nb, **kwargs):
            self.nb = nb

        def execute(self):
            self.nb.cells[0].outputs = [
                nbformat.v4.new_output("stream", text="partial")
            ]
            raise error

    monkeypatch.setattr(build.nbclient, "NotebookClient", Client)
    ok, seconds, message = build._execute(node)
    assert not ok and seconds >= 0
    assert type(error).__name__ in message
    assert node.cells[0].outputs[0].text == "partial"  # kept for saving
