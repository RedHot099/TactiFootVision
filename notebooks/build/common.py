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
# Notebooks that run on Colab, by name.
COLAB_READY = frozenset(
    {
        "00_overview",
        "01_data",
        "02_augmentation",
        "03_training_and_evaluation",
        "04_inference",
        "05_visualisation",
    }
)
_COLAB_BASE = "https://colab.research.google.com/github/RedHot099/TactiFootVision/blob"
_COLAB_BADGE = "https://colab.research.google.com/assets/colab-badge.svg"

# Links from one notebook to another, or to the docs, open them on GitHub.
REPO_URL = "https://github.com/RedHot099/TactiFootVision/blob/master"

# StatsBomb's terms ask for their logo wherever an analysis of their data is
# shown: the one in their open-data repository, pinned to a commit.
STATSBOMB_LOGO = (
    "https://raw.githubusercontent.com/statsbomb/open-data/"
    "3bfc16d8b40965104692c45e378d8f014888e90b/img/"
    "SB%20-%20Icon%20Lockup%20-%20Colour%20positive.png"
)

# The small pitch keypoint dataset the notebooks share, rendered in seconds on
# a CPU: four clips of eight random broadcast views of the synthetic pitch,
# split per clip. Defines `size` (the frame size) and `pitch_data`.
SYNTHETIC_PITCH_DATASET = """
size = (640, 360)
clips = [
    tv.data.synthetic_pitch_clip(
        # File names differ between clips, so an export keeps them.
        OUT / "pitch_views" / f"clip{seed}" / f"clip{seed}_%06d.jpg",
        tv.data.broadcast_homographies(8, size, seed=seed),
        size,
        line_width=0.25,  # metres: the real 12 cm would be under a pixel here
        seed=seed,
    )
    for seed in range(4)
]
pitch_data = tv.data.Dataset.from_clips(clips, train=0.75, valid=0.25, task="pose")
"""

# The pitch keypoint set a short `yolo_pose` run learns from within minutes on
# a T4: six clips of 40 broadcast views with noise and occluding boxes, split
# per clip into 160 training, 40 validation and 40 test frames. The model test
# `test_yolo_pose_learns_pitch_keypoints_from_synthetic_broadcast_views` trains
# on the same recipe. Defines `size` and `pitch_data`.
PITCH_TRAINING_DATASET = """
size = (640, 360)
clips = [
    tv.data.synthetic_pitch_clip(
        OUT / "pitch_views" / f"clip{seed}" / f"clip{seed}_%06d.jpg",
        tv.data.broadcast_homographies(40, size, seed=seed),
        size,
        line_width=0.25,  # metres: the real 12 cm would be under a pixel here
        noise=4,
        occluders=3,  # player-sized boxes over the markings
        seed=seed,
    )
    for seed in range(6)
]
pitch_data = tv.data.Dataset.from_clips(clips, train=0.66, valid=0.17, test=0.17, task="pose")
"""

# The demo detector, trained the same way wherever a notebook needs one. Of
# three tried on the demo set for 10 epochs (YOLO11n at 640 and at 1280,
# YOLO11s at 1920), only this one finds all 22 players of the 4096-pixel
# panorama: MOTA 0.63 on the demo clip against 0.49 and 0.28, for under a
# minute of training on an RTX 4070 Ti SUPER, about five on a Colab T4.
DETECTOR_WEIGHTS, DETECTOR_IMGSZ, DETECTOR_EPOCHS = "yolo11s.pt", 1920, 10
# The training cell (a live-log cell): trains on `dataset`, the demo detection
# set, and leaves the `TrainResult` in `training`.
DEMO_DETECTOR = f"""
training = tv.train("yolo", dataset, weights="{DETECTOR_WEIGHTS}", epochs={DETECTOR_EPOCHS}, batch_size=16,
                    imgsz={DETECTOR_IMGSZ}, rect=True, output_dir=OUT / "runs", name="detector", exist_ok=True)
"""

# The RF-DETR guard: its cells run only where the `rfdetr` extra imports,
# which Colab's runtime does not have. Defines `has_rfdetr`.
RFDETR_GUARD = """
try:
    tv.utils.require_extra("rfdetr")
    has_rfdetr = True
except ImportError as error:
    has_rfdetr = False
    print(f"Skipping the RF-DETR cells: {error} in a local environment (on Colab it would downgrade numpy and transformers)")
"""

# SAM2 is not on PyPI: the SAM2 tracker imports it from a clone of the
# real-time SAM2 repository and loads a checkpoint from the clone's
# `checkpoints/`. The notebooks keep the clone in DATA, next to the demo, and
# install it only when the reader asks: about a minute on Colab (a 60 MB
# shallow clone, the 156 MB checkpoint and the `sam2` extra's two packages).
# Defines SAM2_REPO, SAM2_CHECKPOINT and SAM2_CONFIG.
SAM2_INSTALL = """
INSTALL_SAM2 = False  # True installs SAM2 here, in about a minute

SAM2_REPO = DATA / "segment-anything-2-real-time"
SAM2_CHECKPOINT = SAM2_REPO / "checkpoints" / "sam2.1_hiera_tiny.pt"
SAM2_CONFIG = SAM2_REPO / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_t.yaml"

if INSTALL_SAM2:
    import importlib.util
    import subprocess
    import time

    import torch

    started = time.perf_counter()
    if importlib.util.find_spec("hydra") is None or importlib.util.find_spec("iopath") is None:
        %pip install -q "tactifoot-vision[sam2]"
    if not (SAM2_REPO / "sam2").is_dir():
        subprocess.run(
            ["git", "clone", "-q", "--depth", "1", "https://github.com/Gy920/segment-anything-2-real-time", str(SAM2_REPO)],
            check=True,
        )
    if not SAM2_CHECKPOINT.is_file():
        torch.hub.download_url_to_file(
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt", str(SAM2_CHECKPOINT),
            progress=False,
        )
    print(f"SAM2 installed in {SAM2_REPO} ({time.perf_counter() - started:.0f} s)")
"""

# The SAM2 guard, after SAM2_INSTALL: SAM2 cells run only where the clone, its
# checkpoint and the `sam2` extra are present, on a GPU (the real-time
# predictor moves frames to CUDA). Defines `has_sam2` and, with it,
# `sam2_tracker`.
SAM2_GUARD = """
import importlib.util
import warnings

import torch

if not SAM2_CHECKPOINT.is_file() or not all(importlib.util.find_spec(m) for m in ("hydra", "iopath")):
    sam2_missing = "SAM2 is not installed: set INSTALL_SAM2 = True in the cell above, then run it and this one again"
elif not torch.cuda.is_available():
    sam2_missing = "SAM2 runs on a GPU only (Runtime, Change runtime type, T4 GPU)"
else:
    sam2_missing = None
has_sam2 = sam2_missing is None
if has_sam2:
    # The clone's optional compiled extension is not built: SAM2 warns and
    # uses its Python fallback, which is fine here.
    warnings.filterwarnings("ignore", message="Falling back to the Python connected-components")
    sam2_tracker = tv.tracking.create_tracker("sam2", checkpoint=SAM2_CHECKPOINT, config=SAM2_CONFIG)
    print("SAM2 tracker ready:", type(sam2_tracker).__name__)
else:
    print(f"Skipping the SAM2 cells: {sam2_missing}")
"""

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
