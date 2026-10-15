"""The ``track_ball`` post-process: choose the real ball among the ball candidates.

The pipeline's ball is the most confident candidate of each frame, which is
often a spare ball behind the goal, a painted spot or a player box. Across
the whole video, :func:`track_ball` drops candidates that cannot be the live
ball, follows the ball with a constant-velocity gate on the pitch, removes
the isolated outliers that are left and fills short gaps.

Distances are in metres and times in seconds whatever the pitch units: a
pitch of ``length x width`` units is taken as a standard 105 x 68 m pitch, as
:class:`~tactifoot_vision.pitch.SoccerPitch` scales its markings. Positions
are projections onto the pitch plane, so an airborne ball appears displaced
away from the camera; ball height is not modelled.
"""

import copy
import dataclasses
import logging

import numpy as np
import supervision as sv

from tactifoot_vision.pipeline.postprocess import POSTPROCESSES
from tactifoot_vision.pipeline.result import FrameResult, PipelineResult
from tactifoot_vision.pitch.homography import pitch_to_frame
from tactifoot_vision.pitch.pitch import SoccerPitch, pitch_markings

logger = logging.getLogger(__name__)

_STD_SIZE = np.array([105.0, 68.0])  # metres of the pitch every pitch unit maps to
_CONFIDENCE_WEIGHT = (
    2.0  # metres of distance to the prediction a unit of confidence is worth
)
_NEIGHBOURS = 4  # kept positions per side an outlier is judged against
_EPS = 1e-6  # seconds: timestamps are index / fps, so equal spans may differ in the last bit


@dataclasses.dataclass(frozen=True)
class _Options:
    min_confidence: float
    restart_confidence: float
    restart_after: float
    gate: float
    gate_speed: float
    max_gap: float
    static_radius: float
    static_window: float
    static_share: float
    spot_radius: float
    max_person_iou: float
    isolation: float

    def __post_init__(self) -> None:
        for name in ("min_confidence", "restart_confidence"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        for name in ("static_share", "max_person_iou"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        for name in ("restart_after", "spot_radius", "isolation"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        for name in ("gate", "gate_speed", "max_gap", "static_radius", "static_window"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    def elapsed(self, dt: float) -> float:
        """Seconds the ball counts as moving: past ``max_gap`` it is taken as stopped."""
        return min(dt, self.max_gap)

    def radius(self, dt: float) -> float:
        """Gate radius in metres ``dt`` seconds after the last ball."""
        return self.gate + self.gate_speed * self.elapsed(dt)


@POSTPROCESSES.register("track_ball")
def track_ball(
    result: PipelineResult,
    min_confidence: float = 0.12,
    restart_confidence: float = 0.45,
    restart_after: float = 0.25,
    gate: float = 1.5,
    gate_speed: float = 32.5,
    max_gap: float = 0.5,
    static_radius: float = 0.7,
    static_window: float = 3.2,
    static_share: float = 0.75,
    spot_radius: float = 0.6,
    max_person_iou: float = 0.5,
    isolation: float = 0.15,
) -> PipelineResult:
    """Copy of ``result`` whose ``ball`` is the live ball chosen among the candidates.

    1. Candidates below ``min_confidence``, without a pitch position, within
       ``spot_radius`` m of a painted spot of ``result.pitch``, or whose box
       matches a person's (IoU above ``max_person_iou``: a detector that
       scores every box in every class reports players as balls too) are
       dropped. So is a static candidate: one with a candidate within
       ``static_radius`` m in at least ``static_share`` of the frames within
       ``static_window / 2`` s of it. Only windows spanning at least half of
       ``static_window`` count, so a clip shorter than that keeps them. A
       ball at rest that long (before a set piece) is dropped too, and so
       is the real ball on a spot.
    2. The ball is followed frame by frame: the next position is predicted
       at constant velocity, and the candidate nearest the prediction (a
       more confident one may be slightly farther) is taken if it lies
       within ``gate + gate_speed * dt`` m, ``dt`` being the time since the
       last ball, at most ``max_gap`` (the prediction stops there too).
    3. With nothing in the gate, the most confident candidate of at least
       ``restart_confidence`` (re)starts the track, once the ball has been
       missing for more than ``restart_after`` s (or at the first ball).
    4. Outliers are removed: a ball far (by the same gate) from the median
       of the balls before it and from the median of those after it, and a
       ball less confident than ``restart_confidence`` with no other ball
       within ``isolation`` s.
    5. Gaps of up to ``max_gap`` s are filled by linear interpolation on the
       pitch, in frames that have a homography: the box is centred on the
       projected position with its size interpolated, ``confidence`` is NaN
       and ``data["interpolated"]`` is ``True`` (``False`` on measured balls).

    Frames with no ball chosen get an empty ``ball``; the candidates are
    left as they are. The tables export the flag as ``interpolated``. Run
    this after any post-process that changes the homographies (such as one
    calling :func:`~tactifoot_vision.pipeline.reproject`): those re-project
    the chosen ball, they do not choose it again.

    Args:
        result: the result to track the ball in (its frames need ``ball_candidates``).
        min_confidence: candidates below this confidence are ignored.
        restart_confidence: confidence a candidate needs to start the track.
        restart_after: seconds the ball must be missing before a restart.
        gate: gate radius in metres right after the last ball.
        gate_speed: metres per second the gate widens while the ball is missing.
        max_gap: longest gap in seconds that is filled; the gate stops widening there.
        static_radius: metres within which a candidate counts as not moving.
        static_window: seconds around a frame the static test looks at.
        static_share: share of that window a static candidate is seen in.
        spot_radius: metres around a painted spot within which candidates are dropped.
        max_person_iou: candidates overlapping a person box more than this are dropped.
        isolation: seconds within which a weak ball needs another ball.
    """
    options = _Options(
        min_confidence=min_confidence,
        restart_confidence=restart_confidence,
        restart_after=restart_after,
        gate=gate,
        gate_speed=gate_speed,
        max_gap=max_gap,
        static_radius=static_radius,
        static_window=static_window,
        static_share=static_share,
        spot_radius=spot_radius,
        max_person_iou=max_person_iou,
        isolation=isolation,
    )
    frames = result.frames
    times = np.array([f.timestamp for f in frames], dtype=float)
    if np.any(np.diff(times) <= 0):
        raise ValueError("track_ball needs frames in order of increasing timestamps")
    scale = _STD_SIZE / [result.pitch.length, result.pitch.width]
    positions = [_candidate_xy(f) * scale for f in frames]
    confidences = [_confidence(f.ball_candidates) for f in frames]
    usable = _usable(
        frames, positions, confidences, times, result.pitch, scale, options
    )
    chosen = _follow(positions, confidences, usable, times, options)
    kept = _remove_outliers(positions, confidences, chosen, times, options)
    balls = [
        _chosen_ball(frame, choice if keep else None)
        for frame, choice, keep in zip(frames, chosen, kept, strict=True)
    ]
    filled = _fill_gaps(frames, balls, kept, times, options.max_gap)
    logger.info(
        "Ball tracking chose a ball in %d of %d frames and filled %d more",
        int(kept.sum()),
        len(frames),
        filled,
    )
    return dataclasses.replace(
        result,
        frames=[
            dataclasses.replace(frame, ball=ball)
            for frame, ball in zip(frames, balls, strict=True)
        ],
    )


def _candidate_xy(frame: FrameResult) -> np.ndarray:
    candidates = frame.ball_candidates
    xy = candidates.data.get("pitch_xy")
    if xy is None:
        return np.full((len(candidates), 2), np.nan)
    return np.asarray(xy, dtype=float).reshape(-1, 2)


def _usable(
    frames: list[FrameResult],
    positions: list[np.ndarray],
    confidences: list[np.ndarray],
    times: np.ndarray,
    pitch: SoccerPitch,
    scale: np.ndarray,
    options: _Options,
) -> list[np.ndarray]:
    """Per frame, which candidates may be the live ball (step 1)."""
    _, spots = pitch_markings(pitch)
    spots = spots * scale
    usable = []
    statics = _static(positions, times, options)
    for frame, xy, confidence, static in zip(
        frames, positions, confidences, statics, strict=True
    ):
        keep = np.isfinite(xy).all(axis=1) & ~static
        keep &= confidence >= options.min_confidence
        if len(xy):
            to_spot = np.linalg.norm(xy[:, None] - spots[None], axis=2).min(axis=1)
            keep &= ~(to_spot <= options.spot_radius)
        if len(xy) and len(frame.detections):
            iou = sv.box_iou_batch(frame.ball_candidates.xyxy, frame.detections.xyxy)
            keep &= iou.max(axis=1) <= options.max_person_iou
        usable.append(keep)
    return usable


def _static(
    positions: list[np.ndarray], times: np.ndarray, options: _Options
) -> list[np.ndarray]:
    """Per frame, which candidates stay in place over the window around their frame."""
    counts = np.array([len(xy) for xy in positions])
    # every candidate in one array: frame i's are rows starts[i]:starts[i + 1]
    starts = np.concatenate([[0], np.cumsum(counts)])
    stacked = np.concatenate(
        [xy.reshape(-1, 2) for xy in positions] or [np.zeros((0, 2))]
    )
    half = options.static_window / 2
    first = np.searchsorted(times, times - half - _EPS, side="left")
    last = np.searchsorted(times, times + half + _EPS, side="right") - 1
    static = []
    for i, xy in enumerate(positions):
        lo, hi = first[i], last[i]
        flags = np.zeros(len(xy), dtype=bool)
        if len(xy) and times[hi] - times[lo] >= half - _EPS:
            near = np.linalg.norm(
                xy[:, None] - stacked[starts[lo] : starts[hi + 1]][None], axis=2
            )
            seen = counts[lo : hi + 1] > 0  # each frame with candidates counts once
            blocks = starts[lo : hi + 1][seen] - starts[lo]
            hits = np.logical_or.reduceat(near < options.static_radius, blocks, axis=1)
            flags = hits.sum(axis=1) >= options.static_share * (hi - lo + 1)
        static.append(flags)
    return static


def _follow(
    positions: list[np.ndarray],
    confidences: list[np.ndarray],
    usable: list[np.ndarray],
    times: np.ndarray,
    options: _Options,
) -> list[int | None]:
    """The candidate followed in each frame, ``None`` for none (steps 2 and 3)."""
    chosen: list[int | None] = []
    last: np.ndarray | None = None
    last_time = 0.0
    velocity = np.zeros(2)
    for xy, confidence, keep, time in zip(
        positions, confidences, usable, times, strict=True
    ):
        indices = np.flatnonzero(keep)
        choice = None
        if last is not None:
            dt = time - last_time
            predicted = last + velocity * options.elapsed(dt)
            choice = _in_gate(xy, confidence, indices, predicted, dt, options)
        restarted = choice is None and (
            last is None or time - last_time > options.restart_after + _EPS
        )
        if restarted:
            choice = _most_confident(confidence, indices, options.restart_confidence)
        chosen.append(choice)
        if choice is None:
            continue
        # The prototype damps the velocity after a gap; averaging the old and
        # new velocity does the same, and a restart forgets the old one.
        if restarted:
            velocity = np.zeros(2)
        else:
            velocity = (velocity + (xy[choice] - last) / (time - last_time)) / 2
        last, last_time = xy[choice], time
    return chosen


def _in_gate(
    xy: np.ndarray,
    confidence: np.ndarray,
    indices: np.ndarray,
    predicted: np.ndarray,
    dt: float,
    options: _Options,
) -> int | None:
    """The candidate near ``predicted``, a more confident one may be a little farther."""
    if len(indices) == 0:
        return None
    distance = np.linalg.norm(xy[indices] - predicted, axis=1)
    best = int(np.argmin(distance - _CONFIDENCE_WEIGHT * confidence[indices]))
    return int(indices[best]) if distance[best] < options.radius(dt) else None


def _most_confident(
    confidence: np.ndarray, indices: np.ndarray, threshold: float
) -> int | None:
    strong = indices[confidence[indices] >= threshold]
    return int(strong[np.argmax(confidence[strong])]) if len(strong) else None


def _confidence(candidates: sv.Detections) -> np.ndarray:
    if candidates.confidence is None:
        return np.ones(len(candidates))
    return np.asarray(candidates.confidence, dtype=float)


def _remove_outliers(
    positions: list[np.ndarray],
    confidences: list[np.ndarray],
    chosen: list[int | None],
    times: np.ndarray,
    options: _Options,
) -> np.ndarray:
    """Which frames keep their followed ball (step 4)."""
    kept = np.array([choice is not None for choice in chosen])
    xy = np.full((len(chosen), 2), np.nan)
    confidence = np.zeros(len(chosen))
    for i in np.flatnonzero(kept):
        xy[i] = positions[i][chosen[i]]
        confidence[i] = confidences[i][chosen[i]]
    for _ in range(2):  # a second pass catches what the first one's outliers hid
        _drop_far(kept, xy, times, options)
    _drop_lone(kept, confidence, times, options)
    return kept


def _drop_far(
    kept: np.ndarray, xy: np.ndarray, times: np.ndarray, options: _Options
) -> None:
    """Unkeep balls far from their neighbours on every side that has at least two."""
    rows = np.flatnonzero(kept)
    for j, i in enumerate(rows):
        sides = (rows[max(0, j - _NEIGHBOURS) : j], rows[j + 1 : j + 1 + _NEIGHBOURS])
        judged = [side for side in sides if len(side) >= 2]
        if judged and all(
            _far(xy[i], times[i], xy[s], times[s], options) for s in judged
        ):
            kept[i] = False


def _drop_lone(
    kept: np.ndarray, confidence: np.ndarray, times: np.ndarray, options: _Options
) -> None:
    """Unkeep weak balls with no other ball within ``isolation`` seconds."""
    rows = np.flatnonzero(kept)
    gaps = np.diff(times[rows])
    before = np.concatenate([[np.inf], gaps])
    after = np.concatenate([gaps, [np.inf]])
    lone = np.minimum(before, after) > options.isolation + _EPS
    kept[rows[lone & (confidence[rows] < options.restart_confidence)]] = False


def _far(
    xy: np.ndarray,
    time: float,
    side_xy: np.ndarray,
    side_times: np.ndarray,
    options: _Options,
) -> bool:
    """Whether ``xy`` lies outside the gate around the median of its neighbours on one side."""
    distance = np.linalg.norm(xy - np.median(side_xy, axis=0))
    dt = float(np.median(np.abs(side_times - time)))
    return bool(distance > options.radius(dt))


def _chosen_ball(frame: FrameResult, choice: int | None) -> sv.Detections:
    """A copy of the chosen candidate, flagged as measured (no rows for none)."""
    rows = [] if choice is None else [choice]
    ball = copy.deepcopy(frame.ball_candidates[np.array(rows, dtype=int)])
    ball.data["interpolated"] = np.zeros(len(ball), dtype=bool)
    return ball


def _fill_gaps(
    frames: list[FrameResult],
    balls: list[sv.Detections],
    kept: np.ndarray,
    times: np.ndarray,
    max_gap: float,
) -> int:
    """Fill gaps of up to ``max_gap`` s between kept balls in place; returns how many (step 5)."""
    filled = 0
    rows = np.flatnonzero(kept)
    for a, b in zip(rows[:-1], rows[1:], strict=True):
        if times[b] - times[a] > max_gap + _EPS:
            continue
        for k in range(a + 1, b):
            if frames[k].homography is not None:
                share = (times[k] - times[a]) / (times[b] - times[a])
                balls[k] = _between(balls[a], balls[b], share, frames[k].homography)
                filled += 1
    return filled


def _between(
    start: sv.Detections, end: sv.Detections, share: float, homography: np.ndarray
) -> sv.Detections:
    """The ball ``share`` of the way from ``start`` to ``end``, flagged as interpolated."""
    pitch_xy = (1 - share) * start.data["pitch_xy"][0] + share * end.data["pitch_xy"][0]
    centre = pitch_to_frame(pitch_xy[None], homography)[0]
    size = (1 - share) * (start.xyxy[0, 2:] - start.xyxy[0, :2]) + share * (
        end.xyxy[0, 2:] - end.xyxy[0, :2]
    )
    ball = copy.deepcopy(start)
    ball.xyxy = np.concatenate([centre - size / 2, centre + size / 2])[None].astype(
        start.xyxy.dtype
    )
    if ball.confidence is not None:
        ball.confidence = np.full(1, np.nan, dtype=ball.confidence.dtype)
    ball.data["pitch_xy"] = pitch_xy[None].astype(start.data["pitch_xy"].dtype)
    ball.data["interpolated"] = np.ones(1, dtype=bool)
    return ball
