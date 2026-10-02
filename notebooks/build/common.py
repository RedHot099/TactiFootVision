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


KERNELSPEC = {"display_name": "Python 3", "language": "python", "name": "python3"}


class Notebook:
    """Cells of one notebook, added top to bottom."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.cells: list[nbformat.NotebookNode] = []

    @property
    def path(self) -> Path:
        return NOTEBOOK_DIR / f"{self.name}.ipynb"

    def md(self, text: str) -> None:
        self.cells.append(nbformat.v4.new_markdown_cell(_clean(text)))

    def code(self, text: str) -> None:
        self.cells.append(nbformat.v4.new_code_cell(_clean(text)))

    def setup(self, imports: str = BASE_IMPORTS, body: str = "") -> None:
        """The first code cell: imports, repository paths, logging and figure format.

        ``imports`` replaces the default import block (keep ``os``, ``Path`` and
        ``tv`` in it); ``body`` adds notebook-specific lines at the end.
        """
        parts = [imports, setup_code(self.name), body]
        self.code("\n\n".join(_clean(part) for part in parts if part))

    def to_node(self) -> nbformat.NotebookNode:
        node = nbformat.v4.new_notebook()
        node.cells = self.cells
        node.metadata = {"kernelspec": KERNELSPEC, "language_info": {"name": "python"}}
        return node


def setup_code(name: str) -> str:
    return _clean(f"""
        # The repository root, found by walking up from the working directory and kept
        # relative, so printed paths do not depend on where the repository lives.
        _root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").is_file())
        ROOT = Path(os.path.relpath(_root))
        DATA, MODELS = ROOT / "data", ROOT / "models"
        OUT = ROOT / "outputs" / "notebooks" / "{name}"
        VIDEO = DATA / "videos" / "broadcast_60s.mp4"

        tv.setup_logging("WARNING")
        # Most figures are video frames: JPEG keeps the saved notebook small.
        %config InlineBackend.figure_formats = ["jpeg"]
        %config InlineBackend.print_figure_kwargs = {{"bbox_inches": "tight", "pil_kwargs": {{"quality": 85}}}}
    """)


def _clean(text: str) -> str:
    return textwrap.dedent(text).strip()
