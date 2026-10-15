"""SoccerNet readers: a Tracking sequence folder as a labelled clip."""

import configparser
from pathlib import Path

import numpy as np
import supervision as sv

from tactifoot_vision.data.labelled_clip import LabelledClip, LabelledFrame, Truth

BALL = "ball"
# First word of a gameinfo.ini tracklet label -> package class name. "other"
# objects (none of the package classes) are left out of the clip.
_TRACKLET_CLASSES = {
    "player": "player",
    "goalkeeper": "goalkeeper",
    "goalkeepers": "goalkeeper",
    "referee": "referee",
    "ball": BALL,
    "other": None,
}


def read_tracking(folder: Path) -> LabelledClip:
    """Read a SoccerNet Tracking sequence folder (MOT format).

    ``gt/gt.txt`` rows are ``frame, track id, x, y, w, h, ...`` with 1-based
    frames: MOT frame ``n`` is image ``img1/<n:06d>.jpg``, which the pipeline
    reads as frame ``n - 1``. ``gameinfo.ini`` names each track's class.
    """
    info = _read_ini(folder / "seqinfo.ini")
    try:
        fps = float(info["frameRate"])
        length = int(info["seqLength"])
        width, height = int(info["imWidth"]), int(info["imHeight"])
    except (KeyError, ValueError):
        raise ValueError(
            f"{folder / 'seqinfo.ini'}: [Sequence] needs frameRate, seqLength, "
            "imWidth and imHeight"
        ) from None
    classes = _tracklet_classes(folder / "gameinfo.ini")
    rows = _read_gt(folder / "gt" / "gt.txt")
    frame_numbers, track_ids = rows[:, 0].astype(int), rows[:, 1].astype(int)
    unknown = sorted(set(track_ids) - set(classes))
    if unknown:
        raise ValueError(
            f"{folder / 'gt' / 'gt.txt'} has boxes of track {unknown[0]}, which "
            f"{folder / 'gameinfo.ini'} does not list"
        )
    outside = frame_numbers[(frame_numbers < 1) | (frame_numbers > length)]
    if len(outside):
        raise ValueError(
            f"{folder / 'gt' / 'gt.txt'} has a box in frame {outside[0]}, outside "
            f"the sequence's frames 1..{length} (seqLength={length})"
        )

    names = np.array([classes[t] or "" for t in track_ids], dtype=object)
    xyxy = rows[:, 2:6].copy()
    xyxy[:, 2:] += xyxy[:, :2]
    kept = names != ""
    frames = []
    for index in range(length):
        here = kept & (frame_numbers == index + 1)
        ball = here & (names == BALL)
        people = here & (names != BALL)
        frames.append(
            LabelledFrame(
                index=index,
                detections=_detections(xyxy[people], track_ids[people], names[people]),
                ball=_detections(xyxy[ball], track_ids[ball], names[ball]),
            )
        )
    image_dir = info.get("imDir", "img1")
    extension = info.get("imExt", ".jpg")
    # OpenCV (FFmpeg) reads an image sequence at 25 fps whatever frameRate
    # says; SoccerNet sequences are 25 fps, so the two agree.
    return LabelledClip(
        name=info.get("name", folder.name),
        video_path=folder / image_dir / f"%06d{extension}",
        fps=fps,
        width=width,
        height=height,
        frames=tuple(frames),
        truth=frozenset({Truth.IDENTITIES, Truth.BOXES, Truth.BALL}),
    )


def _read_ini(path: Path) -> configparser.SectionProxy:
    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read_string(path.read_text(), source=str(path))
    except configparser.Error as error:
        raise ValueError(f"{path} is not a valid ini file: {error}") from None
    if not parser.has_section("Sequence"):
        raise ValueError(f"{path} has no [Sequence] section")
    return parser["Sequence"]


def _tracklet_classes(path: Path) -> dict[int, str | None]:
    """Track id -> package class name (``None``: left out) from ``trackletID_<id>`` keys."""
    classes = {}
    for key, value in _read_ini(path).items():
        if not key.startswith("trackletid_"):
            continue
        number = key.removeprefix("trackletid_")
        if not number.isdigit():
            raise ValueError(f"{path}: {key} does not end in a track id")
        label = value.split(";")[0].strip()
        word = label.split()[0].lower() if label else ""
        if word not in _TRACKLET_CLASSES:
            raise ValueError(
                f"{path}: {key} has the unknown class {label!r}; known: "
                f"{', '.join(_TRACKLET_CLASSES)}"
            )
        classes[int(number)] = _TRACKLET_CLASSES[word]
    return classes


def _read_gt(path: Path) -> np.ndarray:
    """``(N, 6)`` float rows ``frame, track id, x, y, w, h``."""
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        return np.zeros((0, 6))
    try:
        return np.loadtxt(lines, delimiter=",", usecols=range(6), ndmin=2)
    except ValueError as error:
        raise ValueError(f"{path} is not a MOT ground-truth file: {error}") from None


def _detections(
    xyxy: np.ndarray, track_ids: np.ndarray, names: np.ndarray
) -> sv.Detections:
    return sv.Detections(
        xyxy=xyxy.reshape(-1, 4),
        tracker_id=track_ids.astype(int),
        data={"class_name": names.astype(str)},
    )
