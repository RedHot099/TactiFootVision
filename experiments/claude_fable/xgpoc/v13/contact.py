"""Contact-interval contract and pre-shot frame admission.

All times are milliseconds relative to the shot anchor t0: the labelled position_ms,
which sits at clip time 4000 ms inside every 8 s clip. Nominal frame offsets
(-1600 .. 0) name the ffmpeg seek target in the half-match video; the frame actually
decoded can sit on another instant, so admission prefers a measured time whenever the
frame has been located inside the clip. A nominal offset <= 0 is therefore not, on its
own, proof that a frame precedes the contact.

Rule: with a known contact interval the input frame must lie strictly before the start
of the interval. With an unknown interval the nominal t0 frame is kept, exactly as the
baseline does, and the record is flagged rather than dropped or shifted. Never pick a
later frame silently.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
import pandas as pd

ANCHOR_MS = 4000
MATCH_SIZE = (112, 63)


class ContactSource(str, Enum):
    NONE = "none"
    MODEL_REVIEW = "model_review"
    HUMAN = "human"


class AdmissionStatus(str, Enum):
    SAFE = "safe"          # known interval, admitted frame lies before its start
    NOMINAL = "nominal"    # unknown interval, nominal t0 frame kept as the baseline does
    UNSAFE = "unsafe"      # known interval, no available frame precedes it


@dataclass(frozen=True)
class ContactInterval:
    lower_ms: int | None
    upper_ms: int | None
    source: ContactSource = ContactSource.NONE
    confidence: str = ""

    def __post_init__(self):
        if (self.lower_ms is None) != (self.upper_ms is None):
            raise ValueError("contact interval needs both bounds or neither")
        if self.lower_ms is not None and self.lower_ms > self.upper_ms:
            raise ValueError("contact interval lower bound above upper bound")
        if self.known and self.source is ContactSource.NONE:
            raise ValueError("a contact interval without a source is not evidence")

    @property
    def known(self) -> bool:
        return self.lower_ms is not None

    @classmethod
    def unknown(cls, source: ContactSource = ContactSource.NONE, confidence: str = "") -> "ContactInterval":
        return cls(None, None, source, confidence)

    @classmethod
    def from_review(cls, row: Mapping, source: ContactSource = ContactSource.MODEL_REVIEW) -> "ContactInterval":
        """Row of combined_review.csv. Uncertain status, missing bounds or a timing
        inconsistency all mean unknown; the reviewer's confidence is carried along."""
        confidence = str(row.get("confidence", "") or "")
        lower, upper = row.get("contact_lower_ms"), row.get("contact_upper_ms")
        consistent = row.get("timing_consistent", True)
        if pd.isna(lower) or pd.isna(upper) or str(row.get("t0_status", "uncertain")) == "uncertain" \
                or not (consistent is True or str(consistent).lower() == "true"):
            return cls.unknown(source, confidence)
        return cls(int(lower), int(upper), source, confidence)


@dataclass(frozen=True)
class DecodedFrame:
    path: str
    nominal_ms: int
    measured_ms: int | None = None

    def __post_init__(self):
        if self.nominal_ms > 0:
            raise ValueError("frames after the nominal shot instant are never inputs")

    @property
    def measured(self) -> bool:
        return self.measured_ms is not None

    @property
    def time_ms(self) -> int:
        return self.nominal_ms if self.measured_ms is None else self.measured_ms


@dataclass(frozen=True)
class Admission:
    status: AdmissionStatus
    frame: DecodedFrame | None
    gap_ms: int | None          # contact start minus frame time; positive = frame precedes contact
    measured: bool              # the decision rests on a measured frame time
    rejected: tuple[str, ...]   # frames at or after the contact start

    def to_record(self) -> dict:
        return {"admission": self.status.value, "frame_path": self.frame.path if self.frame else "",
                "frame_nominal_ms": self.frame.nominal_ms if self.frame else None,
                "frame_time_ms": self.frame.time_ms if self.frame else None,
                "gap_ms": self.gap_ms, "timing_measured": self.measured, "n_rejected": len(self.rejected)}


def admit(frames: Iterable[DecodedFrame], contact: ContactInterval) -> Admission:
    frames = sorted(frames, key=lambda f: f.time_ms)
    if not frames:
        raise ValueError("no candidate frames")
    if len({f.nominal_ms for f in frames}) != len(frames):
        raise ValueError("duplicate nominal offsets")
    if not contact.known:
        t0 = [f for f in frames if f.nominal_ms == 0]
        if not t0:
            raise ValueError("unknown contact keeps the nominal t0 frame, which is missing")
        return Admission(AdmissionStatus.NOMINAL, t0[0], None, t0[0].measured, ())
    rejected = tuple(f.path for f in frames if f.time_ms >= contact.lower_ms)
    safe = [f for f in frames if f.time_ms < contact.lower_ms]
    if not safe:
        return Admission(AdmissionStatus.UNSAFE, None, contact.lower_ms - frames[0].time_ms,
                         all(f.measured for f in frames), rejected)
    chosen = safe[-1]
    return Admission(AdmissionStatus.SAFE, chosen, contact.lower_ms - chosen.time_ms, chosen.measured, rejected)


@dataclass(frozen=True)
class FrameMatch:
    offset_ms: int          # clip instant of the best match, relative to the anchor
    distance: float         # mean absolute pixel difference of the best match
    runner_up: float        # same for the second-best clip frame
    fps: float

    @property
    def margin(self) -> float:
        return self.runner_up - self.distance

    def measured_offset(self, min_margin: float = 0.25) -> int | None:
        """The measured time, or None when neighbouring clip frames are too similar
        (static scene, duplicated frames) to tell which one the JPEG came from."""
        return self.offset_ms if self.margin >= min_margin else None


def _small(image: np.ndarray) -> np.ndarray:
    return cv2.resize(image, MATCH_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)


def locate_frames_in_clip(frame_paths: Iterable[str], clip_path: str, anchor_ms: int = ANCHOR_MS,
                          window_ms: tuple[int, int] = (-2400, 1200)) -> list[FrameMatch]:
    """Find the clip frames that the stored JPEGs were decoded from, by exhaustive image
    comparison over `window_ms` around the anchor. Sequential decoding is used on
    purpose: seeking is what made the nominal offsets untrustworthy in the first place.
    Timestamps come from the container, so variable frame rate clips are honoured."""
    references = []
    for frame_path in frame_paths:
        image = cv2.imread(frame_path)
        if image is None:
            raise FileNotFoundError(frame_path)
        references.append(_small(image))
    capture = cv2.VideoCapture(clip_path)
    if not capture.isOpened():
        raise FileNotFoundError(clip_path)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    candidates: list[list[tuple[float, float]]] = [[] for _ in references]
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            t = float(capture.get(cv2.CAP_PROP_POS_MSEC)) - anchor_ms
            if window_ms[0] <= t <= window_ms[1]:
                small = _small(frame)
                for reference, found in zip(references, candidates):
                    found.append((float(np.abs(small - reference).mean()), t))
    finally:
        capture.release()
    matches = []
    for found in candidates:
        if len(found) < 2:
            raise ValueError(f"fewer than two clip frames inside {window_ms} for {clip_path}")
        found.sort()
        (best, t), (second, _) = found[0], found[1]
        matches.append(FrameMatch(int(round(t)), best, second, fps))
    return matches


def locate_in_clip(frame_path: str, clip_path: str, anchor_ms: int = ANCHOR_MS,
                   window_ms: tuple[int, int] = (-2400, 1200)) -> FrameMatch:
    return locate_frames_in_clip([frame_path], clip_path, anchor_ms, window_ms)[0]
