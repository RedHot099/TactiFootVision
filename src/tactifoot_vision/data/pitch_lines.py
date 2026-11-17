"""Pitch line data: SoccerNet's line annotations as a :data:`~tactifoot_vision.enums.Task.LINES` dataset, and the network's targets.

Two sources carry points along the painted markings, named by
:class:`~tactifoot_vision.enums.PitchMarking`:

* **SoccerNet Camera Calibration** (``calibration-2023``): ``<split>/<n>.jpg``
  with ``<n>.json`` (``{marking: [{"x", "y"}, ...]}``, fractions of the image).
* **SoccerNet Game State Reconstruction** (GSR): ``<split>/SNGS-*/`` with
  ``Labels-GameState.json``, whose ``pitch`` annotations give each frame's
  lines the same way. GSR draws them on key frames and interpolates linearly
  in between, so a frame that is not a key frame weighs ``interpolated_weight``.

Goal posts, crossbars and "Line unknown" are left out (off the pitch plane).

**The label rule** (``rule``): SoccerNet names lines in pitch coordinates,
so a view from the far side looks like a main-camera view of the other half
with every name turned, which no image can tell apart. The shipped rule
``"soccernet-v2"`` keeps, of Calibration's ``train`` split, the 8,316 images
whose line fit is consistent and that look from the main side (7,585) or
along the pitch (731), and none of the xG study's held-out games: the
training set of the package's line network (``pitch_lines_v2.csv`` beside
this module, each image's view). Other splits and GSR are not filtered.

:func:`render_line_targets` turns a sample's lines into the heatmaps the
line network learns: per marking, ``exp(-d^2 / 2 sigma^2)`` of the distance
``d`` (output cells) to the polyline through its points.
"""

import csv
import json
from importlib import resources
from pathlib import Path

import numpy as np

from tactifoot_vision.data._files import image_size
from tactifoot_vision.data.annotations import Annotations
from tactifoot_vision.data.dataset import Dataset, Sample
from tactifoot_vision.enums import LineRule, PitchMarking, PitchView, Split, Task

_RULE_FILE = "pitch_lines_v2.csv"
_MARKINGS = {m.value for m in PitchMarking}
# A GSR frame is a key frame when some line's points do not move linearly
# through it (second difference in time above this, in 1080p pixels).
KEY_FRAME_PX = 0.01
_SHIFT = 4  # cv2 fixed-point bits for sub-cell drawing


def load_pitch_lines(
    calibration: str | Path | None = None,
    gamestate: str | Path | None = None,
    *,
    splits: tuple[str, ...] = (Split.TRAIN, Split.VALID),
    rule: LineRule | str = LineRule.SOCCERNET_V2,
    interpolated_weight: float = 0.5,
) -> Dataset:
    """A ``lines`` dataset from SoccerNet Calibration and/or GSR folders (the folders holding ``train``, ``valid``, ...).

    Each sample's :class:`Annotations` holds ``lines`` (marking name ->
    ``(P, 2)`` pixel points), ``weight`` and ``view`` (``"main"`` or
    ``"along"``: which classes a horizontal flip swaps). GSR samples are the
    frames whose image is on disk (SoccerNet's GSR ``train`` download keeps
    every 5th). See the module for ``rule``.
    """
    if calibration is None and gamestate is None:
        raise ValueError("Give a SoccerNet Calibration folder, a GSR folder, or both")
    rule = LineRule.parse(rule, "rule")
    if not 0 <= interpolated_weight <= 1:
        raise ValueError(
            f"interpolated_weight must be in [0, 1], got {interpolated_weight}"
        )
    keep = _rule_table() if rule is LineRule.SOCCERNET_V2 else None
    out: dict[str, list[Sample]] = {}
    for split in splits:
        split = Split.parse(split)
        samples: list[Sample] = []
        if gamestate is not None and (Path(gamestate) / _gsr_folder(split)).is_dir():
            samples += gamestate_lines(
                Path(gamestate) / _gsr_folder(split),
                interpolated_weight=interpolated_weight,
            )
        if (
            calibration is not None
            and (Path(calibration) / _calib_folder(split)).is_dir()
        ):
            samples += calibration_lines(
                Path(calibration) / _calib_folder(split),
                keep=keep if split == Split.TRAIN else None,
            )
        out[split] = samples
    return Dataset(
        task=Task.LINES,
        class_names=[m.value for m in PitchMarking],
        splits=out,
        name="soccernet_lines",
    )


def calibration_lines(
    folder: str | Path, keep: dict[str, str] | None = None
) -> list[Sample]:
    """The images of one SoccerNet Calibration split folder with at least one marking.

    ``keep`` (stem -> view) limits them to its images, in its order;
    otherwise every image, by name.
    """
    folder = Path(folder)
    samples = []
    if keep is None:
        labels = sorted(p for p in folder.glob("*.json") if p.stem.isdigit())
    else:
        labels = [folder / f"{stem}.json" for stem in keep]
    for label in labels:
        image = label.with_suffix(".jpg")
        if not label.exists() or not image.exists():
            continue
        width, height = image_size(image)
        lines = _lines(json.loads(label.read_text()), width, height)
        if not lines:
            continue
        view = (
            keep.get(label.stem, PitchView.MAIN) if keep is not None else PitchView.MAIN
        )
        samples.append(Sample(image, width, height, _annotations(lines, 1.0, view)))
    if keep is not None and not samples:
        raise FileNotFoundError(
            f"{folder} holds none of the images the label rule keeps"
        )
    return samples


def gamestate_lines(
    folder: str | Path, interpolated_weight: float = 0.5
) -> list[Sample]:
    """The frames on disk of every GSR clip in ``folder`` that carry lines (weight 1 on key frames)."""
    samples = []
    for labels in sorted(Path(folder).glob("SNGS-*/Labels-GameState.json")):
        samples += _gsr_clip(labels, interpolated_weight)
    return samples


def render_line_targets(
    lines: dict[str, np.ndarray],
    out_size: tuple[int, int],
    stride: int,
    sigma: float = 1.0,
) -> np.ndarray:
    """``(20, H, W)`` float32 heatmaps at ``out_size`` (width, height) of ``lines`` given in input pixels.

    Output cell ``i`` covers input pixels ``[stride i, stride (i + 1))``, so a
    point at input pixel ``u`` sits at output coordinate ``u / stride - 0.5``.
    """
    import cv2

    from tactifoot_vision.pitch.markings import INDEX

    w, h = out_size
    target = np.zeros((len(INDEX), h, w), np.float32)
    for name, pts in lines.items():
        k = INDEX.get(PitchMarking(name)) if name in _MARKINGS else None
        if k is None or len(pts) < 2:
            continue
        o = np.asarray(pts, float) / stride - 0.5
        canvas = np.full((h, w), 255, np.uint8)
        poly = np.round(o * (1 << _SHIFT)).astype(np.int32)
        cv2.polylines(canvas, [poly], False, 0, 1, cv2.LINE_8, _SHIFT)
        d = np.minimum(cv2.distanceTransform(canvas, cv2.DIST_L2, 3), 100.0)
        target[k] = np.exp(-(d**2) / (2 * sigma**2)) * (d < 4 * sigma)
    return target


def _rule_table() -> dict[str, str]:
    with resources.files("tactifoot_vision.data").joinpath(_RULE_FILE).open() as f:
        return {row["stem"]: PitchView(row["view"]) for row in csv.DictReader(f)}


def _calib_folder(split: str) -> str:
    return str(split)


def _gsr_folder(split: str) -> str:
    return str(split)


def _lines(raw: dict, width: int, height: int) -> dict[str, np.ndarray]:
    """Marking name -> ``(P, 2)`` pixel points (fractions times the image size), planar markings only."""
    out = {}
    for name, points in raw.items():
        if name not in _MARKINGS or not points:
            continue
        out[name] = np.array(
            [[p["x"] * width, p["y"] * height] for p in points], np.float32
        )
    return out


def _annotations(
    lines: dict[str, np.ndarray], weight: float, view: PitchView
) -> Annotations:
    return Annotations(
        np.zeros((0, 4)), np.zeros(0), lines=lines, weight=weight, view=view
    )


def _gsr_clip(labels_path: Path, interpolated_weight: float) -> list[Sample]:
    labels = json.loads(labels_path.read_text())
    n = int(labels["info"]["seq_length"])
    index_of = {
        str(i["image_id"]): int(Path(i["file_name"]).stem) - 1 for i in labels["images"]
    }
    labelled = np.zeros(n, bool)
    for image in labels["images"]:
        labelled[index_of[str(image["image_id"])]] = (
            image.get("is_labeled", True) is not False
        )
    fractions: list[dict[str, np.ndarray]] = [{} for _ in range(n)]
    for a in labels["annotations"]:
        t = index_of[str(a["image_id"])]
        if labelled[t] and a.get("supercategory") == "pitch":
            fractions[t] = {
                k: np.array([[p["x"], p["y"]] for p in v], float)
                for k, v in a["lines"].items()
                if k in _MARKINGS and len(v)
            }
    key = _key_frames(fractions, labelled)
    folder = labels_path.parent / "img1"
    samples = []
    for t in range(n):
        image = folder / f"{t + 1:06d}.jpg"
        if not fractions[t] or not image.exists():
            continue
        width, height = image_size(image)
        lines = {
            k: (v * (width, height)).astype(np.float32) for k, v in fractions[t].items()
        }
        weight = 1.0 if key[t] else interpolated_weight
        samples.append(
            Sample(image, width, height, _annotations(lines, weight, PitchView.MAIN))
        )
    return samples


def _key_frames(lines: list[dict], labelled: np.ndarray) -> np.ndarray:
    """Frames whose line labels were drawn, not interpolated (the first and last labelled frames count)."""
    n = len(lines)
    key = np.zeros(n, bool)
    labelled_idx = np.flatnonzero(labelled)
    if len(labelled_idx):
        key[labelled_idx[0]] = key[labelled_idx[-1]] = True
    scale = np.array([1920.0, 1080.0])
    for t in range(1, n - 1):
        a, b, c = lines[t - 1], lines[t], lines[t + 1]
        for name, points in b.items():
            if name in a and name in c and len(a[name]) == len(points) == len(c[name]):
                if (
                    np.abs((a[name] - 2 * points + c[name]) * scale).max()
                    > KEY_FRAME_PX
                ):
                    key[t] = True
                    break
    return key
