"""Fixed vocabularies of the package, as ``StrEnum``s (ADR 0004).

A member equals its string (``Role.GOALKEEPER == "goalkeeper"``), so every
function that takes one also takes the plain string, and run files, the CLI
and saved results keep their strings. Functions turn a value into its member
on entry with :meth:`Vocabulary.parse`, which names the parameter and the
allowed values when the value is unknown.

This module imports nothing from the rest of the package, so any module can
import it.
"""

from enum import StrEnum
from typing import Self


class Vocabulary(StrEnum):
    """A closed set of strings; the base of every enum of this module."""

    @classmethod
    def parse(cls, value: str, name: str | None = None) -> Self:
        """``cls(value)``, with an error that names ``name`` and the allowed values."""
        try:
            return cls(value)
        except ValueError:
            allowed = ", ".join(cls)
            raise ValueError(
                f"Unknown {name or cls.__name__} {value!r}. Available: {allowed}"
            ) from None

    def __repr__(self) -> str:
        # Messages, logs and printed lists show the plain string a user would type.
        return repr(self.value)

    def __reduce_ex__(self, protocol: int) -> tuple[type[str], tuple[str]]:
        # A member pickles as its plain string (a key of a saved result's
        # sv.Detections.data, say), so no saved file needs this module to load.
        return str, (self.value,)


class Task(Vocabulary):
    """What a dataset or model is about."""

    DETECT = "detect"  # bounding boxes with classes (players, ball, ...)
    POSE = "pose"  # boxes plus a fixed set of keypoints per object (pitch landmarks)


class Truth(Vocabulary):
    """A kind of truth a labelled clip carries; each source provides some of them."""

    IDENTITIES = "identities"  # track ids that follow one object through the clip
    BOXES = "boxes"  # frame-pixel boxes with class names
    BALL = "ball"  # the ball's box
    PITCH_POSITIONS = "pitch_positions"
    CALIBRATION = "calibration"  # a frame -> pitch homography per frame
    TEAMS = "teams"  # each person's team, or none (referees)


class Role(Vocabulary):
    """A person's **Role** on the pitch; a class name outside the set is ``OTHER``."""

    PLAYER = "player"
    GOALKEEPER = "goalkeeper"
    REFEREE = "referee"
    OTHER = "other"

    @classmethod
    def of(cls, class_name: str) -> Self:
        """The role of a **Class name**: itself when it is a role, else ``OTHER``."""
        return cls(class_name) if class_name in cls else cls.OTHER


class ObjectKind(Vocabulary):
    """What a row of the tracks table is: a person or the ball."""

    PERSON = "person"
    BALL = "ball"


class DetectionKey(Vocabulary):
    """A key of ``sv.Detections.data`` that the package writes or reads."""

    CLASS_NAME = "class_name"  # the detector's **Class name**
    PITCH_XY = "pitch_xy"
    TEAM_ID = "team_id"
    INTERPOLATED = "interpolated"
    UNCERTAINTY = "uncertainty"
    ORIGINAL_TRACKER_ID = "original_tracker_id"
    JERSEY = "jersey"


class Split(Vocabulary):
    """A dataset split."""

    TRAIN = "train"
    VALID = "valid"
    TEST = "test"

    @classmethod
    def parse(cls, value: str, name: str | None = "split") -> Self:
        """The split ``value`` names, aliases (``val``, ``validation``) included."""
        return super().parse(SPLIT_ALIASES.get(value, value), name)


SPLIT_ALIASES = {"val": Split.VALID, "validation": Split.VALID}


class AnnotatorStyle(Vocabulary):
    """How ``FrameAnnotator`` draws people."""

    STANDARD = "standard"
    VIDEO_GAME = "video_game"


class SpaceModel(Vocabulary):
    """A space-control model."""

    NEAREST_PLAYER = "nearest_player"
    PITCH_CONTROL = "pitch_control"


class OverlayPosition(Vocabulary):
    """Where an overlay sits on a frame."""

    CENTER = "center"
    TOP_LEFT = "top-left"
    TOP_CENTER = "top-center"
    TOP_RIGHT = "top-right"
    CENTER_LEFT = "center-left"
    CENTER_RIGHT = "center-right"
    BOTTOM_LEFT = "bottom-left"
    BOTTOM_CENTER = "bottom-center"
    BOTTOM_RIGHT = "bottom-right"


class LinkMode(Vocabulary):
    """How a dataset export places its images."""

    SYMLINK = "symlink"
    HARDLINK = "hardlink"
    COPY = "copy"


class Pooling(Vocabulary):
    """How the SigLIP embedder pools its patch tokens."""

    MEAN = "mean"
    CLS = "cls"


class ColorSpace(Vocabulary):
    """The colour space of the SigLIP embedder's colour histogram."""

    RGB = "rgb"
    HSV = "hsv"


class Reducer(Vocabulary):
    """The team classifier's dimensionality reducer (``None``: none)."""

    UMAP = "umap"


class ModelType(Vocabulary):
    """A built-in model backend of ``MODELS``."""

    YOLO = "yolo"
    YOLO_POSE = "yolo_pose"
    RFDETR = "rfdetr"


class TrackerType(Vocabulary):
    """A built-in tracker of ``TRACKERS``."""

    BYTETRACK = "bytetrack"
    SAM2 = "sam2"


class EmbedderType(Vocabulary):
    """A built-in team embedder of ``EMBEDDERS``."""

    SIGLIP = "siglip"
    RESNET = "resnet"


class PostProcessName(Vocabulary):
    """A built-in post-process of ``POSTPROCESSES``."""

    SMOOTH_HOMOGRAPHIES = "smooth_homographies"
    JOIN_TRACKS = "join_tracks"
    TRACK_BALL = "track_ball"
    ASSIGN_ROLES = "assign_roles"


class RunFileSection(Vocabulary):
    """A top-level section of a **Run file**."""

    DETECTOR = "detector"
    KEYPOINTS = "keypoints"
    TRACKER = "tracker"
    TEAMS = "teams"
    PITCH = "pitch"
    HOMOGRAPHY = "homography"
    BALL_SCORER = "ball_scorer"
    PIPELINE = "pipeline"
    RENDER = "render"
    POSTPROCESS = "postprocess"
