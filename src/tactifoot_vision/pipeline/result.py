"""What the inference pipeline produces: per-frame tracks, teams and pitch positions."""

import bisect
import json
import math
import pickle
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import supervision as sv

from tactifoot_vision.pitch.homography import frame_to_pitch, visible_area
from tactifoot_vision.pitch.lines import LineFit
from tactifoot_vision.pitch.pitch import SoccerPitch

NO_TEAM = -1
UNTRACKED = -1  # track id / player id of a detection without a tracker
BALL_PLAYER_ID = -99  # player id of the ball in freeze frames
PERSON_ANCHOR = sv.Position.BOTTOM_CENTER  # where a person's pitch position is taken
BALL_ANCHOR = sv.Position.CENTER
# StatsBomb's match clock: the minute at which each period kicks off.
_PERIOD_KICK_OFF_MINUTE = {1: 0, 2: 45, 3: 90, 4: 105}
# A person's role: their class name when it is one of these, else "other".
ROLES = ("player", "goalkeeper", "referee", "other")
_FREEZE_FRAME_COLUMNS = [
    "frame_id", "period", "timestamp", "timestamp_seconds", "minute", "second",
    "homography_matrix", "visible_area", "player_id", "type", "class_name",
    "location", "frame_bbox", "confidence", "keeper", "team_id", "team_name",
    "interpolated",
]  # fmt: skip
_EPS = 1e-6  # seconds: timestamps are index / fps, so equal spans may differ in the last bit
_TRACK_COLUMNS = [
    "frame", "time", "object", "track_id", "class_name", "team_id", "team_name",
    "confidence", "x1", "y1", "x2", "y2", "pitch_x", "pitch_y", "interpolated",
]  # fmt: skip


@dataclass
class ObjectMasks:
    """Segmentation masks of a frame's people, each stored inside its own bounding region.

    A full-frame boolean mask costs width x height bytes per object (2 MB at
    1080p); a crop costs only the area the object covers. ``crops[i]`` is the
    mask of detection ``i`` with its top-left pixel at ``origins[i]`` (x, y).
    """

    origins: np.ndarray  # (N, 2) int x, y
    crops: list[np.ndarray]  # N bool arrays of shape (h_i, w_i)

    @classmethod
    def from_dense(cls, masks: np.ndarray) -> "ObjectMasks":
        """Crop ``(N, H, W)`` boolean masks to the pixels each one covers."""
        origins = np.zeros((len(masks), 2), dtype=int)
        crops = []
        for i, mask in enumerate(masks):
            rows, cols = (
                np.flatnonzero(mask.any(axis=1)),
                np.flatnonzero(mask.any(axis=0)),
            )
            if len(rows) == 0:
                crops.append(np.zeros((0, 0), dtype=bool))
                continue
            origins[i] = cols[0], rows[0]
            crops.append(mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1].copy())
        return cls(origins=origins, crops=crops)

    def to_dense(self, width: int, height: int) -> np.ndarray:
        """``(N, height, width)`` boolean masks at their full-frame position."""
        dense = np.zeros((len(self), height, width), dtype=bool)
        for mask, (region, crop) in zip(
            dense, self.regions(width, height), strict=True
        ):
            mask[region] = crop
        return dense

    def regions(
        self, width: int, height: int
    ) -> Iterator[tuple[tuple[slice, slice], np.ndarray]]:
        """``((rows, cols), crop)`` per object, clipped to a ``width x height`` frame.

        ``frame[rows, cols]`` has the shape of ``crop``; a crop reaching past
        the frame (or with its origin outside it) is cut to the visible part,
        which may be empty.
        """
        for (x, y), crop in zip(self.origins, self.crops, strict=True):
            h, w = crop.shape
            x0, y0 = min(max(x, 0), width), min(max(y, 0), height)
            x1, y1 = max(min(x + w, width), x0), max(min(y + h, height), y0)
            region = (slice(y0, y1), slice(x0, x1))
            yield region, crop[y0 - y : y1 - y, x0 - x : x1 - x]

    def __len__(self) -> int:
        return len(self.crops)


@dataclass
class FrameResult:
    """Everything the pipeline knows about one video frame.

    ``detections`` holds tracked people (players, goalkeepers, referees),
    ``ball_candidates`` every ball detection of the frame (best confidence
    first) and ``ball`` the chosen ball among them: the best candidate, until
    a post-process chooses another. The tables export ``ball`` only. All three
    are ``sv.Detections`` whose ``data`` carries:

    * ``class_name`` – ``(N,)`` str
    * ``pitch_xy`` – ``(N, 2)`` float pitch coordinates (NaN without homography);
      people are anchored at the bottom-centre of the box, the ball at its centre
    * ``team_id`` – ``(N,)`` int, ``NO_TEAM`` (-1) when unknown (people only)
    * ``interpolated`` – ``(N,)`` bool, set on ``ball`` by
      :func:`~tactifoot_vision.pipeline.track_ball`: ``True`` for a position
      filled between measured ones (missing means ``False``)
    """

    index: int  # frame index in the source video
    timestamp: float  # seconds since the start of the video
    detections: sv.Detections
    ball: sv.Detections
    ball_candidates: sv.Detections = field(kw_only=True)
    keypoints: sv.KeyPoints | None = None  # pitch keypoints in frame pixels
    homography: np.ndarray | None = None  # 3x3 frame -> pitch
    masks: ObjectMasks | None = None  # one per person (Pipeline(keep_masks=True))
    # How the homography fits the painted lines (HomographyEstimator's
    # refine_with_lines); None when it was not refined on them.
    line_fit: LineFit | None = None

    @property
    def pitch_xy(self) -> np.ndarray:
        return _data(self.detections, "pitch_xy", (len(self.detections), 2), np.nan)

    @property
    def team_ids(self) -> np.ndarray:
        return _data(
            self.detections, "team_id", (len(self.detections),), NO_TEAM
        ).astype(int)

    @property
    def ball_xy(self) -> np.ndarray:
        return _data(self.ball, "pitch_xy", (len(self.ball), 2), np.nan)


def project_detections(
    detections: sv.Detections, anchor: sv.Position, homography: np.ndarray | None
) -> np.ndarray:
    """``(N, 2)`` pitch positions of the detections' ``anchor`` (NaN without a homography)."""
    if homography is None or len(detections) == 0:
        return np.full((len(detections), 2), np.nan, dtype=np.float32)
    return frame_to_pitch(detections.get_anchors_coordinates(anchor), homography)


def _data(
    detections: sv.Detections, key: str, shape: tuple[int, ...], fill: float
) -> np.ndarray:
    value = detections.data.get(key)
    return np.asarray(value) if value is not None else np.full(shape, fill)


@dataclass
class AppliedPostProcess:
    """One post-process a result went through: its registered name and every option."""

    name: str
    options: dict[str, Any]


@dataclass
class PipelineResult:
    """Output of :meth:`Pipeline.run`: one :class:`FrameResult` per processed frame.

    ``frames`` are in video order, as :meth:`Pipeline.run` makes them, and
    every post-process keeps that order. ``postprocesses`` lists the
    post-processes applied since, in order
    (see :mod:`tactifoot_vision.pipeline.postprocess`).
    ``track_embeddings`` holds the mean of the team embeddings the pass
    sampled for each track (float16), by the tracker's own id
    (``original_tracker_id`` once tracks are renamed), and
    ``track_embedding_samples`` how many embeddings each mean is over, so a
    post-process can compare kits without the video.
    """

    frames: list[FrameResult]
    fps: float
    frame_size: tuple[int, int]  # (width, height) of the source video
    class_names: list[str]
    pitch: SoccerPitch = field(default_factory=SoccerPitch)
    video_path: Path | None = None
    postprocesses: list[AppliedPostProcess] = field(default_factory=list)
    track_embeddings: dict[int, np.ndarray] = field(default_factory=dict)
    track_embedding_samples: dict[int, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, i: int) -> FrameResult:
        return self.frames[i]

    def __iter__(self) -> Iterator[FrameResult]:
        return iter(self.frames)

    def frame(self, video_index: int) -> FrameResult:
        """Look up the result of a frame by its index in the source video."""
        return self.frames[self._position(video_index)]

    def window(self, video_index: int, seconds: float) -> list[FrameResult]:
        """The results of the frames before video frame ``video_index``, oldest first.

        A frame belongs to the window when its ``timestamp`` is at most
        ``seconds`` before that of ``video_index``; the frame itself does
        not. The window covers the same time whatever the stride, and is
        shorter, or empty, at the start of the result. This is the
        ``history`` a trail drawing takes (ADR 0003).
        """
        position = self._position(video_index)
        now = self.frames[position].timestamp
        start = position
        while start > 0 and now - self.frames[start - 1].timestamp <= seconds + _EPS:
            start -= 1
        return self.frames[start:position]

    def _position(self, video_index: int) -> int:
        """Where the frame ``video_index`` is in ``frames``."""
        position = bisect.bisect_left(self.frames, video_index, key=lambda f: f.index)
        if position == len(self.frames) or self.frames[position].index != video_index:
            if any(f.index == video_index for f in self.frames):
                raise ValueError(
                    "PipelineResult needs frames in video order to look one up"
                )
            raise KeyError(f"Frame {video_index} was not processed")
        return position

    @property
    def track_ids(self) -> list[int]:
        ids: set[int] = set()
        for f in self.frames:
            if f.detections.tracker_id is not None:
                ids.update(int(t) for t in f.detections.tracker_id)
        return sorted(ids)

    # ---------------------------------------------------------------- tables
    def to_dataframe(self, team_names: Sequence[str] | None = None) -> pd.DataFrame:
        """One row per object per frame.

        Columns: ``frame, time, object ("person" | "ball"), track_id, class_name,
        team_id, team_name, confidence, x1, y1, x2, y2, pitch_x, pitch_y,
        interpolated``. ``track_id`` is ``UNTRACKED`` (-1) without a tracker
        and for the ball; ``team_name`` is ``team_names[team_id]`` (``None``
        without names or a team); ``interpolated`` marks a ball position
        filled by ``track_ball``.
        """
        return self._tracks_table(self._check_team_names(team_names))

    def _tracks_table(self, names: list[str] | None) -> pd.DataFrame:
        """:meth:`to_dataframe` with ``names`` already checked."""
        rows: list[dict] = []
        for f in self.frames:
            for kind, detections, pitch_xy, team_ids in _objects(f):
                for obj in _object_rows(detections, pitch_xy, team_ids, names):
                    x1, y1, x2, y2 = (float(v) for v in obj["xyxy"])
                    rows.append({
                        "frame": f.index, "time": f.timestamp, "object": kind,
                        "track_id": obj["track_id"], "class_name": obj["class_name"],
                        "team_id": obj["team_id"], "team_name": obj["team_name"],
                        "confidence": obj["confidence"],
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "pitch_x": float(obj["pitch_xy"][0]),
                        "pitch_y": float(obj["pitch_xy"][1]),
                        "interpolated": obj["interpolated"],
                    })  # fmt: skip
        return pd.DataFrame(rows, columns=_TRACK_COLUMNS)

    def to_csv(self, path: str | Path, team_names: Sequence[str] | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe(team_names).to_csv(path, index=False)
        return path

    def to_parquet(
        self, path: str | Path, team_names: Sequence[str] | None = None
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe(team_names).to_parquet(path, index=False)
        return path

    def to_mot(self, path: str | Path) -> Path:
        """Write the people and the ball as MOT text (MOTChallenge, SoccerNet Tracking).

        One row per object per frame: ``frame, track id, x, y, w, h,
        confidence, -1, -1, -1``, with 1-based frames (video frame ``i`` is
        MOT frame ``i + 1``) and confidence -1 when there is none (an
        interpolated ball's is NaN). MOT text has no class column, so the
        ball is one track whose id follows the people's largest. Track ids
        must not be negative.
        """
        ball_id = max(self.track_ids, default=0) + 1
        lines = []
        for f in self.frames:
            people, ball = f.detections, f.ball
            if len(people) and people.tracker_id is None:
                raise ValueError(
                    "to_mot needs the people's track ids; the result was made "
                    "without a tracker"
                )
            if len(people) and (people.tracker_id < 0).any():
                raise ValueError(
                    f"to_mot needs track ids of 0 or more; frame {f.index} has "
                    f"{int(people.tracker_id.min())}"
                )
            ball_ids = np.full(len(ball), ball_id)
            for detections, ids in ((people, people.tracker_id), (ball, ball_ids)):
                for i in range(len(detections)):
                    x1, y1, x2, y2 = (float(v) for v in detections.xyxy[i])
                    score = _float(detections.confidence, i)
                    finite = score is not None and math.isfinite(score)
                    confidence = f"{score:g}" if finite else "-1"
                    lines.append(
                        f"{f.index + 1},{int(ids[i])},{x1:.2f},{y1:.2f},"
                        f"{x2 - x1:.2f},{y2 - y1:.2f},{confidence},-1,-1,-1"
                    )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(line + "\n" for line in lines))
        return path

    def to_freeze_frames(
        self,
        period: int = 1,
        period_start: float | None = None,
        team_names: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """StatsBomb-360-like table: one row per object with JSON-encoded geometry.

        Columns match what :func:`tactifoot_vision.evaluation.compare_with_statsbomb`
        expects (``period, minute, second, location, type, player_id, ...``),
        and an empty result gives an empty table with the same columns.

        ``period_start`` is the match clock (seconds) at the first video frame.
        ``None`` means the video starts at the period's kick-off on StatsBomb's
        clock: minute 0, 45, 90 or 105 for periods 1 to 4 (other periods need
        an explicit ``period_start``).

        ``minute`` and ``second`` follow StatsBomb's match clock (the second
        half starts at minute 45). ``timestamp`` (``HH:MM:SS.mmm``) and
        ``timestamp_seconds`` are that match clock too, unlike StatsBomb's
        event ``timestamp``, which restarts every period. ``type`` is
        ``player``, ``goalkeeper``, ``referee``, ``other`` or ``ball``;
        ``player_id`` is the track id (``UNTRACKED`` without a tracker,
        ``BALL_PLAYER_ID`` for the ball), and ``interpolated`` marks a ball
        position filled by ``track_ball``. ``team_name`` is
        ``team_names[team_id]``, as in :meth:`to_dataframe`: the object's own
        team, unlike the ``team_name`` of
        :func:`~tactifoot_vision.data.load_statsbomb`, which is the team of
        the event. ``visible_area`` is the frame outline on the pitch as
        StatsBomb's flat ``[x1, y1, x2, y2, ...]`` polygon (``None`` without a
        homography, or when part of the frame lies beyond the horizon, where
        the outline has no finite pitch position).
        """
        return self._freeze_frames_table(
            period,
            _period_start(period, period_start),
            self._check_team_names(team_names),
        )

    def _freeze_frames_table(
        self, period: int, period_start: float, names: list[str] | None
    ) -> pd.DataFrame:
        """:meth:`to_freeze_frames` with ``period_start`` and ``names`` resolved."""
        rows: list[dict] = []
        for f in self.frames:
            t = period_start + f.timestamp
            area = visible_area(f.homography, self.frame_size)
            base = {
                "frame_id": f.index,
                "period": period,
                "timestamp": _format_clock(t),
                "timestamp_seconds": t,
                "minute": math.floor(t / 60),
                "second": math.floor(t % 60),
                "homography_matrix": _json(f.homography),
                "visible_area": _json(None if area is None else area.reshape(-1)),
            }
            for kind, detections, pitch_xy, team_ids in _objects(f):
                for obj in _object_rows(detections, pitch_xy, team_ids, names):
                    name = obj["class_name"]
                    if kind == "ball":
                        kind_type, player_id = "ball", BALL_PLAYER_ID
                    else:
                        kind_type = name if name in ROLES else "other"
                        player_id = obj["track_id"]
                    rows.append(base | {
                        "player_id": player_id, "type": kind_type, "class_name": name,
                        "location": _json(obj["pitch_xy"]),
                        "frame_bbox": _json(obj["xyxy"].round().astype(int)),
                        "confidence": obj["confidence"],
                        "keeper": name == "goalkeeper", "team_id": obj["team_id"],
                        "team_name": obj["team_name"],
                        "interpolated": obj["interpolated"],
                    })  # fmt: skip
        return pd.DataFrame(rows, columns=_FREEZE_FRAME_COLUMNS)

    def _check_team_names(self, team_names: Sequence[str] | None) -> list[str] | None:
        """``team_names`` as a list, checked to name every team of the result."""
        if team_names is None:
            return None
        if isinstance(team_names, str) or not isinstance(
            team_names, Sequence | np.ndarray
        ):
            raise ValueError(
                "team_names must be a list of names, one per team id from 0, "
                f"got {team_names!r}"
            )
        names = list(team_names)
        for name in names:
            if not isinstance(name, str) or not name:
                # YAML reads [1860, Bayern] or [Yes, No] as numbers and booleans.
                raise ValueError(
                    f"team_names holds an empty or non-text name: {names}; "
                    'quote names such as "1860" or "Yes" in YAML'
                )
            if names.count(name) > 1:
                raise ValueError(f"team_names holds {name!r} twice: {names}")
        teams = {int(t) for f in self.frames for t in f.team_ids if t != NO_TEAM}
        if missing := sorted(t for t in teams if not 0 <= t < len(names)):
            raise ValueError(
                f"team_names {names} has no name for team {missing[0]}; "
                "give one name per team id from 0"
            )
        return names

    # ----------------------------------------------------------- persistence
    def export(
        self,
        out_dir: str | Path,
        *,
        period: int = 1,
        period_start: float | None = None,
        team_names: Sequence[str] | None = None,
    ) -> Path:
        """Write the run folder's data files to ``out_dir`` (created if missing).

        ``result.pkl`` (:meth:`save`), ``tracks.csv`` (:meth:`to_dataframe`)
        and ``freeze_frames.csv`` (:meth:`to_freeze_frames` with ``period``
        and ``period_start``; ``None`` is the period's kick-off).
        ``team_names`` (one per team id from 0, e.g. ``["Home", "Away"]``)
        fill the tables' ``team_name`` column. ``period`` and ``team_names``
        are checked before anything is written. Render the annotated video
        with :func:`tactifoot_vision.viz.render_video`, or write the whole run
        folder with :meth:`tactifoot_vision.run_file.RunFile.run`.
        """
        start = _period_start(period, period_start)
        names = self._check_team_names(team_names)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.save(out_dir / "result.pkl")
        self._tracks_table(names).to_csv(out_dir / "tracks.csv", index=False)
        self._freeze_frames_table(period, start, names).to_csv(
            out_dir / "freeze_frames.csv", index=False
        )
        return out_dir

    def save(self, path: str | Path) -> Path:
        """Pickle the whole result (reload with :meth:`load`)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PipelineResult":
        """Read a result written by :meth:`save` (pickle: only load files you trust).

        A result saved before frames kept their ``ball_candidates``, or
        before results kept their ``track_embeddings``, is refused.
        """
        with Path(path).open("rb") as fh:
            result = pickle.load(fh)
        if not isinstance(result, cls):
            raise TypeError(f"{path} does not contain a PipelineResult")
        if not all(hasattr(frame, "ball_candidates") for frame in result.frames):
            raise ValueError(
                f"{path} was saved before results kept the ball candidates; "
                "run the pipeline again to post-process it"
            )
        if not hasattr(result, "track_embedding_samples"):
            raise ValueError(
                f"{path} was saved before results kept the team embeddings; "
                "run the pipeline again to post-process it"
            )
        return result


def _period_start(period: int, period_start: float | None) -> float:
    """``period_start``, or the period's kick-off on StatsBomb's clock when ``None``."""
    if period_start is not None:
        return period_start
    if period not in _PERIOD_KICK_OFF_MINUTE:
        raise ValueError(
            f"Period {period} has no StatsBomb kick-off time; pass period_start"
        )
    return _PERIOD_KICK_OFF_MINUTE[period] * 60.0


def _objects(
    f: FrameResult,
) -> Iterator[tuple[str, sv.Detections, np.ndarray, np.ndarray | None]]:
    """``(kind, detections, pitch_xy, team_ids)`` of a frame's people and its ball."""
    yield "person", f.detections, f.pitch_xy, f.team_ids
    yield "ball", f.ball, f.ball_xy, None


def _object_rows(
    detections: sv.Detections,
    pitch_xy: np.ndarray,
    team_ids: np.ndarray | None,
    team_names: list[str] | None,
) -> Iterator[dict]:
    """The fields every table row of an object shares, one dict per detection."""
    names = detections.data.get("class_name", [])
    interpolated = detections.data.get("interpolated", [])
    for i in range(len(detections)):
        team = int(team_ids[i]) if team_ids is not None else NO_TEAM
        yield {
            "track_id": int(detections.tracker_id[i])
            if detections.tracker_id is not None
            else UNTRACKED,
            "class_name": str(names[i]) if len(names) else "",
            "team_id": team,
            "team_name": team_names[team] if team_names and team != NO_TEAM else None,
            "confidence": _float(detections.confidence, i),
            "xyxy": detections.xyxy[i],
            "pitch_xy": pitch_xy[i],
            "interpolated": bool(interpolated[i]) if len(interpolated) else False,
        }


def _float(values: np.ndarray | None, i: int) -> float | None:
    return None if values is None else float(values[i])


def _json(value: np.ndarray | None) -> str | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if not np.isfinite(array).all():
        return None
    return json.dumps(array.tolist())


def _format_clock(seconds: float) -> str:
    whole, millis = divmod(round(seconds * 1000), 1000)
    return f"{whole // 3600:02d}:{whole % 3600 // 60:02d}:{whole % 60:02d}.{millis:03d}"
