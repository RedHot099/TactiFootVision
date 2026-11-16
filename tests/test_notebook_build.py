"""The notebook builder (``notebooks/build/build.py``) without running a kernel."""

import ast
import importlib.util
import re
import textwrap
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


def test_what_leaks_past_a_capture_magic_is_not_saved(build):
    stream = nbformat.v4.new_output("stream", name="stderr", text="cc: warning\n")
    result = nbformat.v4.new_output("execute_result", data={"text/plain": "1"})
    node = nbformat.v4.new_notebook()
    node.cells = [
        nbformat.v4.new_code_cell("%%capture log\ntrain()", outputs=[stream, result]),
        nbformat.v4.new_code_cell("print(1)", outputs=[stream]),
    ]
    build._drop_unsaved_streams(node)
    assert [o.output_type for o in node.cells[0].outputs] == ["execute_result"]
    assert node.cells[1].outputs == [stream]  # an ordinary cell keeps its output


def test_a_live_log_shows_while_running_and_is_not_saved(build):
    import common

    notebook = common.Notebook("demo")
    notebook.code("train()", live_log=True)
    notebook.code("print(1)")
    node = notebook.to_node()
    assert not node.cells[0].source.startswith("%%capture")
    stream = nbformat.v4.new_output("stream", name="stderr", text="epoch 1/10\n")
    result = nbformat.v4.new_output("execute_result", data={"text/plain": "1"})
    widget = nbformat.v4.new_output(
        "display_data", data={"application/vnd.jupyter.widget-view+json": {}}
    )
    plot = nbformat.v4.new_output("display_data", data={"image/png": "iVBOR"})
    node.cells[0].outputs = [stream, widget, plot, result]
    node.cells[1].outputs = [stream]
    build._drop_unsaved_streams(node)
    kept = node.cells[0].outputs
    assert kept == [plot, result]  # the plot's last state stays
    assert node.cells[1].outputs == [stream]


def _without_magics(source: str) -> str:
    """A code cell as plain Python: IPython magics become ``pass``."""
    return "\n".join(
        line.replace(line.lstrip(), "pass") if line.lstrip()[:1] == "%" else line
        for line in source.splitlines()
    )


def _visualisation_keywords() -> set[str]:
    """The keyword arguments the visualisation notebook's code cells pass."""
    from nb05_visualisation import notebook

    sources = [
        _without_magics(cell.source)
        for cell in notebook().cells
        if cell.cell_type == "code"
    ]
    return {
        keyword.arg
        for source in sources
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        for keyword in node.keywords
    }


def test_the_visualisation_notebook_shows_the_ball_carrier_ring(build):
    assert {"ball_carrier", "ball_carriers"} <= _visualisation_keywords()


def test_the_visualisation_notebook_shows_the_velocity_arrows(build):
    assert {"velocity_seconds", "velocities"} <= _visualisation_keywords()


def _overview_code() -> list[str]:
    from nb00_overview import notebook

    return [cell.source for cell in notebook().cells if cell.cell_type == "code"]


@pytest.mark.parametrize("installed", [True, False])
def test_the_overview_notebook_installs_the_package_only_when_it_is_missing(
    build, monkeypatch, installed
):
    import common
    from IPython.core.inputtransformer2 import TransformerManager

    install = textwrap.dedent(common.INSTALL).strip()
    assert _overview_code()[0].startswith(install)  # before the package is imported
    calls = []

    class Shell:
        def run_line_magic(self, name, line):
            calls.append((name, line))

    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: (
            None
            if name == "tactifoot_vision" and not installed
            else real_find_spec(name)
        ),
    )
    exec(TransformerManager().transform_cell(install), {"get_ipython": Shell})

    if installed:
        assert calls == []
    else:  # the core from PyPI: an extra would downgrade Colab's numpy
        assert calls == [("pip", "install -q tactifoot-vision")]


def test_the_overview_notebook_needs_no_local_data_or_weights(build):
    names = {
        node.id
        for source in _overview_code()
        for node in ast.walk(ast.parse(_without_magics(source)))
        if isinstance(node, ast.Name)
    }
    assert "demo" in names  # it runs on what download_demo fetched
    assert not {"MODELS", "VIDEO"} & names


def _colab_ready_names() -> list[str]:
    import common

    return sorted(common.COLAB_READY)


def test_the_colab_links_open_each_ready_notebook_at_one_release():
    # `master` may use API newer than the package the first cell installs, so
    # the badge opens the notebook as it was at a release (never an rc).
    import common
    from nb00_overview import notebook as overview

    assert re.fullmatch(r"v\d+\.\d+\.\d+", common.RELEASE_TAG)
    assert "00_overview" in common.COLAB_READY
    root = BUILD_DIR.parents[1]
    for name in _colab_ready_names():
        url = common.Notebook(name).colab_url
        assert url == (
            "https://colab.research.google.com/github/RedHot099/TactiFootVision"
            f"/blob/{common.RELEASE_TAG}/notebooks/{name}.ipynb"
        )
        assert (root / "notebooks" / f"{name}.ipynb").is_file()
        built = nbformat.read(root / "notebooks" / f"{name}.ipynb", as_version=4)
        assert f"]({url})" in built.cells[0].source, name
    # the README pages link the overview
    url = overview().colab_url
    for page in ("README.md", "notebooks/README.md"):
        assert f"]({url})" in (root / page).read_text(), page
    assert f"]({url})" in overview().cells[0].source


def test_only_ready_notebooks_have_a_badge_and_a_standalone_first_cell():
    import common

    for name in ("00_overview", "01_data"):
        nb = common.Notebook(name)
        nb.setup()
        ready = name in common.COLAB_READY
        assert nb.colab_ready is ready
        assert ("find_spec" in nb.cells[0].source) is ready


def test_a_colab_ready_notebook_reads_no_local_data_or_weights():
    # 01 to 05 still read local data until they are made Colab-ready.
    forbidden = ("models/", "data/datasets", "data/keypoints", "external/")
    for name in _colab_ready_names():
        source = (BUILD_DIR / f"nb{name}.py").read_text()
        assert not [f for f in forbidden if f in source], name
