"""Shared pieces of the notebook generators: cell helpers and the setup cell."""

import textwrap
from pathlib import Path

import nbformat

NOTEBOOK_DIR = Path(__file__).resolve().parents[1]

BASE_IMPORTS = """
import os
from pathlib import Path

import tactifoot_vision as tv
"""


# First lines of a notebook that also runs outside a checkout: on a fresh Colab
# runtime the package is missing and comes from PyPI (the core, no extras: the
# rfdetr extra would downgrade Colab's numpy and transformers); in a checkout
# it is installed already and nothing happens.
INSTALL = """
import importlib.util

if importlib.util.find_spec("tactifoot_vision") is None:  # e.g. a fresh Colab runtime
    %pip install -q tactifoot-vision
"""

# The release the Colab badges open: the notebooks as they were at this tag, the
# package their first cell installs; `master` may be ahead of it. A release
# moves it to its own tag (never an rc).
RELEASE_TAG = "v0.3.0"
COLAB_READY = frozenset({"00_overview"})  # notebooks that run on Colab, by name
_COLAB_BASE = "https://colab.research.google.com/github/RedHot099/TactiFootVision/blob"
_COLAB_BADGE = "https://colab.research.google.com/assets/colab-badge.svg"

KERNELSPEC = {"display_name": "Python 3", "language": "python", "name": "python3"}


LIVE_LOG = "live-log"  # tag of a cell whose printed text is not saved


class Notebook:
    """Cells of one notebook, added top to bottom."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.cells: list[nbformat.NotebookNode] = []

    @property
    def colab_ready(self) -> bool:
        return self.name in COLAB_READY

    @property
    def colab_url(self) -> str:
        return f"{_COLAB_BASE}/{RELEASE_TAG}/notebooks/{self.name}.ipynb"

    @property
    def badge(self) -> str:
        """The "Open in Colab" badge for the title cell of a Colab-ready notebook."""
        return f"[![Open In Colab]({_COLAB_BADGE})]({self.colab_url})"

    @property
    def path(self) -> Path:
        return NOTEBOOK_DIR / f"{self.name}.ipynb"

    def md(self, text: str) -> None:
        self.cells.append(nbformat.v4.new_markdown_cell(_clean(text)))

    def code(self, text: str, live_log: bool = False) -> None:
        """A code cell. A ``live_log`` cell shows what it prints while it runs,
        a trainer's progress, and the builder saves it without that text."""
        cell = nbformat.v4.new_code_cell(_clean(text))
        if live_log:
            cell.metadata["tags"] = [LIVE_LOG]
        self.cells.append(cell)

    def setup(
        self,
        imports: str = BASE_IMPORTS,
        body: str = "",
        standalone: bool | None = None,
    ) -> None:
        """The first code cell: imports, repository paths, logging and figure format.

        ``imports`` replaces the default import block (keep ``os``, ``Path`` and
        ``tv`` in it); ``body`` adds notebook-specific lines at the end. A
        Colab-ready notebook (``standalone`` by default) also runs outside a
        checkout: the cell installs the package when it is missing, and no
        path points at local data or weights.
        """
        standalone = self.colab_ready if standalone is None else standalone
        parts = [
            INSTALL if standalone else "",
            imports,
            setup_code(self.name, standalone),
            body,
        ]
        self.code("\n\n".join(_clean(part) for part in parts if part))

    def to_node(self) -> nbformat.NotebookNode:
        node = nbformat.v4.new_notebook()
        node.cells = self.cells
        for number, cell in enumerate(node.cells):
            cell.id = f"cell-{number:03d}"  # stable ids keep rebuild diffs small
        node.metadata = {"kernelspec": KERNELSPEC, "language_info": {"name": "python"}}
        return node


_CHECKOUT_PATHS = """
# The repository root, found by walking up from the working directory and kept
# relative, so printed paths do not depend on where the repository lives.
_root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").is_file())
ROOT = Path(os.path.relpath(_root))
DATA, MODELS = ROOT / "data", ROOT / "models"
OUT = ROOT / "outputs" / "notebooks" / "{name}"
VIDEO = DATA / "videos" / "broadcast_60s.mp4"
"""

_STANDALONE_PATHS = """
# The repository root when run from a checkout, else the working directory
# (/content on Colab); kept relative, so printed paths are the same everywhere.
_here = [Path.cwd(), *Path.cwd().parents]
_root = next((p for p in _here if (p / "pyproject.toml").is_file()), Path.cwd())
ROOT = Path(os.path.relpath(_root))
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "notebooks" / "{name}"
"""

_DISPLAY = """
tv.setup_logging("WARNING")
# Most figures are video frames: JPEG keeps the saved notebook small.
%config InlineBackend.figure_formats = ["jpeg"]
%config InlineBackend.print_figure_kwargs = {"bbox_inches": "tight", "pil_kwargs": {"quality": 85}}
"""


def setup_code(name: str, standalone: bool) -> str:
    paths = _STANDALONE_PATHS if standalone else _CHECKOUT_PATHS
    return _clean(paths.format(name=name)) + "\n\n" + _clean(_DISPLAY)


def _clean(text: str) -> str:
    return textwrap.dedent(text).strip()
