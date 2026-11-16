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


NOTEBOOKS = sorted(path.stem[2:] for path in BUILD_DIR.glob("nb*.py"))


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


def test_no_saved_cell_keeps_a_widget(build):
    import common

    notebook = common.Notebook("demo")
    notebook.code("download()")
    node = notebook.to_node()
    stream = nbformat.v4.new_output("stream", name="stdout", text="credit\n")
    widget = nbformat.v4.new_output(
        "display_data", data={"application/vnd.jupyter.widget-view+json": {}}
    )
    node.cells[0].outputs = [widget, stream]
    build._drop_unsaved_streams(node)
    assert node.cells[0].outputs == [stream]


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


def _generated(name: str) -> list[nbformat.NotebookNode]:
    """The cells the builder writes for notebook ``name``."""
    module = importlib.import_module(f"nb{name}")
    return module.notebook().cells


def _code_nodes(cells: list[nbformat.NotebookNode]) -> list[ast.AST]:
    return [
        node
        for cell in cells
        if cell.cell_type == "code"
        for node in ast.walk(ast.parse(_without_magics(cell.source)))
    ]


def _called(cells: list[nbformat.NotebookNode]) -> set[str]:
    """Names of the functions the code cells call (``tv.data.f(...)`` gives ``f``)."""
    return {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in _code_nodes(cells)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute | ast.Name)
    }


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_a_notebook_runs_on_what_it_downloads(build, name):
    # The standalone first cell defines no MODELS or VIDEO, and the maintainer's
    # datasets, weights and checkouts are not on Colab under any path.
    nodes = _code_nodes(_generated(name))
    names = {node.id for node in nodes if isinstance(node, ast.Name)}
    strings = {
        node.value
        for node in nodes
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "download_demo" in _called(_generated(name))
    assert not {"MODELS", "VIDEO"} & names
    assert not {"datasets", "keypoints", "models", "external"} & strings


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_a_notebook_tells_where_the_token_steps_are(build, name):
    import common

    title = _generated(name)[0].source
    assert "HF_TOKEN" in title
    if name != "00_overview":  # the overview lists the steps itself
        overview = common.Notebook("00_overview")
        assert f"]({overview.colab_url})" in title or "00_overview.ipynb)" in title


def test_the_colab_links_open_each_notebook_at_one_release():
    # `master` may use API newer than the package the first cell installs, so
    # the badge opens the notebook as it was at a release (never an rc).
    import common
    from nb00_overview import notebook as overview

    assert re.fullmatch(r"v\d+\.\d+\.\d+", common.RELEASE_TAG)
    assert len(NOTEBOOKS) == 6
    root = BUILD_DIR.parents[1]
    for name in NOTEBOOKS:
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


def test_the_readme_opens_with_its_badges(build):
    from nb00_overview import notebook as overview

    lines = (BUILD_DIR.parents[1] / "README.md").read_text().splitlines()
    # one paragraph, one badge per line, first thing on the page
    badges = "\n".join(lines[: lines.index("")])
    expected = [
        overview().colab_url,
        "https://img.shields.io/pypi/v/tactifoot-vision",
        "https://img.shields.io/pypi/pyversions/tactifoot-vision",
        "https://img.shields.io/pypi/l/tactifoot-vision",
        "actions/workflows/ci.yml/badge.svg",
        "https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json",
    ]
    positions = [badges.find(part) for part in expected]
    assert -1 not in positions and positions == sorted(positions), badges
    quick_start = lines.index("## Quick start")
    later = "\n".join(lines[quick_start:])
    assert "colab-badge.svg" not in later  # the badge moved up, not copied


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_every_notebook_installs_the_package_when_it_is_missing(build, name):
    import common

    install = textwrap.dedent(common.INSTALL).strip()
    first_code = next(c for c in _generated(name) if c.cell_type == "code")
    assert first_code.source.startswith(install)


def test_a_notebook_reads_no_local_data_or_weights():
    forbidden = ("models/", "data/datasets", "data/keypoints", "external/")
    for name in NOTEBOOKS:
        source = (BUILD_DIR / f"nb{name}.py").read_text()
        assert not [f for f in forbidden if f in source], name


def test_the_augmentation_notebook_shows_every_transform(build):
    import inspect

    from tactifoot_vision import augment

    transforms = {
        name
        for name in augment.__all__
        if inspect.isclass(cls := getattr(augment, name))
        and issubclass(cls, augment.Transform)
        and not inspect.isabstract(cls)
    }
    called = _called(_generated("02_augmentation"))
    assert {"HorizontalFlip", "RandomAffine", "Compose", "OneOf"} <= transforms
    assert transforms | {"augment_dataset", "show_augmentations"} <= called


def _string_arguments(cells: list[nbformat.NotebookNode], function: str) -> set[str]:
    """First arguments of the calls to ``function`` that are string literals."""
    return {
        node.args[0].value
        for node in _code_nodes(cells)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute | ast.Name)
        and (node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id)
        == function
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_a_notebook_trains_the_demo_detector_with_the_shared_recipe(build, name):
    # One recipe for the overview's detector, so the notebooks cannot drift apart.
    import common

    cells = _generated(name)
    if "yolo" not in _string_arguments(cells, "train"):
        pytest.skip(f"{name} trains no detector")
    recipe = textwrap.dedent(common.DEMO_DETECTOR).strip()
    trains = [c for c in cells if c.cell_type == "code" and '"yolo"' in c.source]
    assert [c.source for c in trains] == [recipe]
    assert trains[0].metadata["tags"] == [common.LIVE_LOG]


def test_the_training_notebook_trains_and_scores_every_backend(build):
    cells = _generated("03_training_and_evaluation")
    assert {"yolo", "yolo_pose"} <= _string_arguments(cells, "train")
    assert "rfdetr" in _string_arguments(cells, "load_model")
    called = _called(cells)
    assert {"evaluate", "plot_metrics", "plot_training", "from_clips"} <= called
    assert "broadcast_homographies" in called and "synthetic_pitch_clip" in called


def _rfdetr_guard(cells: list[nbformat.NotebookNode]) -> nbformat.NotebookNode:
    (guard,) = [c for c in cells if 'require_extra("rfdetr")' in c.source]
    return guard


def _used_outside(
    cells: list[nbformat.NotebookNode], flag: str, word: str
) -> list[str]:
    """Code cells that use ``word`` (in a name or as a string) other than
    in the body of an ``if`` on ``flag``."""
    outside = []
    for cell in cells:
        tree = ast.parse(_without_magics(cell.source))
        guarded = {
            id(node)
            for statement in ast.walk(tree)
            if isinstance(statement, ast.If)
            and flag
            in {n.id for n in ast.walk(statement.test) if isinstance(n, ast.Name)}
            for part in statement.body  # not the else branch
            for node in ast.walk(part)
        }
        uses = [
            node
            for node in ast.walk(tree)
            if (isinstance(node, ast.Name) and word in node.id and node.id != flag)
            or (isinstance(node, ast.Constant) and node.value == word)
        ]
        if not all(id(node) in guarded for node in uses):
            outside.append(cell.source)
    return outside


RFDETR_NOTEBOOKS = ["03_training_and_evaluation", "04_inference"]


@pytest.mark.parametrize("name", RFDETR_NOTEBOOKS)
def test_the_rfdetr_cells_run_only_where_the_extra_imports(build, name):
    # Colab's runtime has no rfdetr extra: every cell that uses RF-DETR sits
    # under the guard's flag, so the notebook still runs top to bottom there.
    import common

    cells = [c for c in _generated(name) if c.cell_type == "code"]
    guard = _rfdetr_guard(cells)
    assert guard.source == textwrap.dedent(common.RFDETR_GUARD).strip()
    assert "has_rfdetr = True" in guard.source
    later = cells[cells.index(guard) + 1 :]
    assert later and not _used_outside(later, "has_rfdetr", "rfdetr")


@pytest.mark.parametrize("name", RFDETR_NOTEBOOKS)
def test_without_the_rfdetr_extra_one_line_says_how_to_run_its_cells(
    build, monkeypatch, capsys, name
):
    import sys

    import tactifoot_vision as tv

    monkeypatch.setitem(sys.modules, "rfdetr", None)  # import fails: not installed
    guard = _rfdetr_guard(_generated(name))
    namespace = {"tv": tv}
    exec(guard.source, namespace)

    assert namespace["has_rfdetr"] is False
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert 'pip install "tactifoot-vision[rfdetr]"' in lines[0]


SAM2_NOTEBOOKS = ["04_inference", "05_visualisation"]


def _sam2_cells(name: str) -> tuple[str, str, list[str]]:
    """The SAM2 install cell, the guard after it (as the builder shares them)
    and the code cells after the guard."""
    import common

    sources = [c.source for c in _generated(name) if c.cell_type == "code"]
    install = textwrap.dedent(common.SAM2_INSTALL).strip()
    guard = textwrap.dedent(common.SAM2_GUARD).strip()
    assert sources.count(install) == 1 and sources.count(guard) == 1, name
    assert sources.index(install) < sources.index(guard)
    return install, guard, sources[sources.index(guard) + 1 :]


@pytest.mark.parametrize("name", SAM2_NOTEBOOKS)
def test_the_sam2_cells_run_only_where_sam2_is_ready(build, name):
    # Colab has no SAM2: every later cell that uses it sits under the guard's
    # flag, so the notebook runs top to bottom without it.
    *_, later = _sam2_cells(name)
    later = [nbformat.v4.new_code_cell(source) for source in later]
    assert later and not _used_outside(later, "has_sam2", "sam2")


def _run_cell(source: str, namespace: dict) -> None:
    """Run a code cell's source as IPython would, magics included."""
    from IPython.core.inputtransformer2 import TransformerManager

    exec(TransformerManager().transform_cell(source), namespace)


class _Shell:
    """What a cell's line magics reach: ``%pip`` is recorded, not run."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def run_line_magic(self, name: str, line: str) -> None:
        self.calls.append((name, line))


@pytest.mark.parametrize("cuda", [False, True])
def test_without_sam2_one_line_says_how_to_enable_it(
    build, monkeypatch, capsys, tmp_path, cuda
):
    import torch

    import tactifoot_vision as tv

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    install, guard, _ = _sam2_cells("04_inference")
    shell = _Shell()
    namespace = {"tv": tv, "DATA": tmp_path / "data", "get_ipython": lambda: shell}
    _run_cell(install, namespace)  # off by default: does nothing
    _run_cell(guard, namespace)

    assert shell.calls == [] and not (tmp_path / "data").exists()
    assert namespace["has_sam2"] is False
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert "INSTALL_SAM2 = True" in lines[0]


def test_the_sam2_install_cell_fetches_what_the_tracker_loads(
    build, monkeypatch, capsys, tmp_path
):
    # On: the sam2 extra from PyPI, a clone of the real-time SAM2 repository
    # and its tiny checkpoint, where the guard looks; a second run fetches
    # nothing. The network is mocked.
    import subprocess

    import torch

    import tactifoot_vision as tv

    clones, downloads = [], []

    def clone(command, check):
        clones.append(command)
        package = Path(command[-1]) / "sam2"
        (package / "configs" / "sam2.1").mkdir(parents=True)
        (package / "build_sam.py").touch()
        (package / "configs" / "sam2.1" / "sam2.1_hiera_t.yaml").touch()
        (Path(command[-1]) / "checkpoints").mkdir()

    def download(url, path, progress=True):
        downloads.append(url)
        Path(path).write_bytes(b"weights")

    monkeypatch.setattr(subprocess, "run", clone)
    monkeypatch.setattr(torch.hub, "download_url_to_file", download)
    real_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: None if name == "hydra" else real_find_spec(name),
    )
    install, *_ = _sam2_cells("04_inference")
    assert "INSTALL_SAM2 = False" in install  # off unless the reader turns it on
    switched_on = install.replace("INSTALL_SAM2 = False", "INSTALL_SAM2 = True")
    shell = _Shell()
    namespace = {"tv": tv, "DATA": tmp_path / "data", "get_ipython": lambda: shell}
    _run_cell(switched_on, namespace)

    [(magic, line)] = shell.calls
    assert magic == "pip" and '"tactifoot-vision[sam2]"' in line
    [command] = clones
    assert command[:2] == ["git", "clone"]
    assert "https://github.com/Gy920/segment-anything-2-real-time" in command
    assert downloads == [
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
    ]
    assert namespace["SAM2_CHECKPOINT"].read_bytes() == b"weights"
    assert namespace["SAM2_CONFIG"].is_file()
    assert namespace["SAM2_REPO"].is_relative_to(tmp_path / "data")
    assert "SAM2 installed" in capsys.readouterr().out

    shell.calls.clear()
    monkeypatch.setattr(importlib.util, "find_spec", real_find_spec)
    _run_cell(switched_on, namespace)
    assert len(clones) == 1 and len(downloads) == 1


def _markdown(cells: list[nbformat.NotebookNode]) -> str:
    return "\n".join(c.source for c in cells if c.cell_type == "markdown")


def _keywords(cells: list[nbformat.NotebookNode]) -> set[str]:
    return {
        keyword.arg
        for node in _code_nodes(cells)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
    }


def test_the_inference_notebook_shows_every_stage_on_the_demo(build):
    cells = _generated("04_inference")
    called = _called(cells)
    attributes = {n.attr for n in _code_nodes(cells) if isinstance(n, ast.Attribute)}
    # pitch geometry: the demo camera's calibration, and a keypoint model
    # trained and run on synthetic frames
    assert {"FisheyeUndistortion", "frame_to_pitch", "pitch_to_frame"} <= attributes
    assert "yolo_pose" in _string_arguments(cells, "train")
    assert {"synthetic_pitch_clip", "evaluate_homography"} <= called
    # trackers scored against the clip's truth, teams, the pipeline's outputs
    assert {"create_tracker", "evaluate_tracking", "fit_predict"} <= called
    assert {"to_freeze_frames", "export", "save", "load", "track_ball"} <= called
    assert {"fixed_homography", "undistortion", "keypoint_model"} <= _keywords(cells)
    assert {"keep_masks"} <= _keywords(cells)


def test_the_visualisation_notebook_shows_every_annotator_style_and_flag(build):
    import inspect

    from tactifoot_vision.enums import AnnotatorStyle
    from tactifoot_vision.viz import FrameAnnotator

    cells = _generated("05_visualisation")
    options = set(inspect.signature(FrameAnnotator).parameters) - {"pitch"}
    assert options <= _keywords(cells)
    assert {s.value for s in AnnotatorStyle} <= _strings(cells)
    assert {
        "PitchRadar",
        "draw_points",
        "overlay",
        "render_video",
        "plot_heatmap",
        "plot_tracks",
        "plot_distance_histogram",
        "draw_pitch",
        "show",
    } <= _called(cells)
    # the pipeline on the demo clip, as in 04_inference
    assert {"fixed_homography", "undistortion"} <= _keywords(cells)


def _strings(cells: list[nbformat.NotebookNode]) -> set[str]:
    return {
        node.value
        for node in _code_nodes(cells)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


def _function(call: ast.Call) -> str | None:
    func = call.func
    return func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)


def _draws_pitch_lines(annotator: ast.Call) -> bool:
    return not any(
        keyword.arg == "draw_pitch_lines"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is False
        for keyword in annotator.keywords
    )


def test_the_visualisation_notebook_draws_pitch_lines_only_on_the_undistorted_frame(
    build,
):
    # The demo homography holds in the undistorted image: on the fisheye frame
    # as recorded its straight lines would miss the curved painted ones.
    nodes = _code_nodes(_generated("05_visualisation"))
    calls = [node for node in nodes if isinstance(node, ast.Call)]
    annotators = [call for call in calls if _function(call) == "FrameAnnotator"]
    named = {
        target.id: node.value
        for node in nodes
        if isinstance(node, ast.Assign)
        and any(node.value is annotator for annotator in annotators)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    drawn_on = []
    for call in calls:
        if _function(call) == "annotate":
            receiver = call.func.value
            annotator = (
                named[receiver.id] if isinstance(receiver, ast.Name) else receiver
            )
            assert _function(annotator) == "FrameAnnotator", ast.unparse(call)
            if _draws_pitch_lines(annotator):
                drawn_on.append(ast.unparse(call.args[0]))
        if _function(call) == "render_video":
            (annotator,) = [k.value for k in call.keywords if k.arg == "annotator"]
            assert not _draws_pitch_lines(annotator), ast.unparse(call)
    assert drawn_on and set(drawn_on) == {"undistorted"}
