"""Labelled clips: a continuous video clip with per-frame truth, and the reader entry point."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np
import supervision as sv


class Truth(StrEnum):
    """A kind of truth a labelled clip carries; each source provides some of them."""

    IDENTITIES = "identities"  # track ids that follow one object through the clip
    BOXES = "boxes"  # frame-pixel boxes with class names
    BALL = "ball"  # the ball's box
    PITCH_POSITIONS = "pitch_positions"
    CALIBRATION = "calibration"  # a frame -> pitch homography per frame
    TEAMS = "teams"  # each person's team, or none (referees)


@dataclass(frozen=True)
class LabelledFrame:
    """The truth of one frame, shaped like a :class:`~tactifoot_vision.pipeline.FrameResult`.

    ``detections`` holds the people (players, goalkeepers, referees) and
    ``ball`` the ball, both as ``sv.Detections`` with ``xyxy`` frame pixels,
    ``tracker_id`` (the source's track ids) and ``data["class_name"]``. A
    source with pitch positions puts them in ``data["pitch_xy"]``, ``(N, 2)``
    ``SoccerPitch`` coordinates (NaN where unknown), as in a ``FrameResult``.
    A source with teams puts them in ``data["team_id"]``, the same key and
    ``(N,)`` int dtype as a ``FrameResult``'s: team ids from 0, and
    ``NO_TEAM`` (-1) for people who have no team, such as referees. Unlike a
    ``FrameResult``'s, truth ``NO_TEAM`` never means an unknown team.
    """

    index: int  # the frame's FrameResult.index when the pipeline runs on the clip
    detections: sv.Detections
    ball: sv.Detections
    homography: np.ndarray | None = None  # 3x3 frame -> pitch


@dataclass(frozen=True)
class LabelledClip:
    """A video clip with per-frame truth, read with :func:`load_labelled_clip`.

    Unlike a :class:`~tactifoot_vision.data.Dataset` of independent images, a
    labelled clip is continuous: ``frames`` holds every frame of the clip in
    order, unlabelled ones as empty frames. ``video_path`` is what the clip's
    frames are read from, in place (for an image sequence, an OpenCV pattern
    such as ``img1/%06d.jpg``). ``truth`` says which kinds of truth the source
    provides; what it lacks is absent, not empty.
    """

    name: str
    video_path: Path
    fps: float
    width: int
    height: int
    frames: tuple[LabelledFrame, ...] = field(repr=False)
    truth: frozenset[Truth]

    def __post_init__(self) -> None:
        # Frozen, so the index built here cannot go stale.
        object.__setattr__(self, "frames", tuple(self.frames))
        object.__setattr__(self, "truth", frozenset(Truth(t) for t in self.truth))
        object.__setattr__(self, "_by_index", {f.index: f for f in self.frames})

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> Iterator[LabelledFrame]:
        return iter(self.frames)

    def frame(self, index: int) -> LabelledFrame:
        """The truth of the frame with this ``FrameResult.index``."""
        try:
            return self._by_index[index]
        except KeyError:
            raise IndexError(f"Frame {index} is not in clip {self.name}") from None


_TRACKING_FILES = ("gt/gt.txt", "gameinfo.ini", "seqinfo.ini")
_EXPECTED = f"a SoccerNet Tracking sequence folder ({', '.join(_TRACKING_FILES)})"


def load_labelled_clip(path: str | Path) -> LabelledClip:
    """Load a labelled clip folder, detecting its layout.

    Known layouts: a SoccerNet Tracking sequence folder (``gt/gt.txt``,
    ``gameinfo.ini``, ``seqinfo.ini``). Video frames are referenced in place,
    never copied or re-encoded. A folder in no known layout, or one missing a
    file of its layout, is a ``ValueError`` naming the path and the layout.
    """
    from tactifoot_vision.data import soccernet  # it imports this module

    path = Path(path)
    if path.is_dir() and any((path / name).is_file() for name in _TRACKING_FILES):
        missing = [name for name in _TRACKING_FILES if not (path / name).is_file()]
        if missing:
            raise ValueError(f"{path / missing[0]} not found; expected {_EXPECTED}")
        return soccernet.read_tracking(path)
    raise ValueError(
        f"{path} is not a labelled clip in a known layout; expected {_EXPECTED}"
    )
