"""What the inference pipeline produces: per-frame tracks, teams and pitch positions."""

import json
import math
import pickle
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import supervision as sv

from tactifoot_vision.pitch.homography import project_points
from tactifoot_vision.pitch.pitch import SoccerPitch

NO_TEAM = -1
UNTRACKED = -1  # track id / player id of a detection without a tracker
BALL_PLAYER_ID = -99  # player id of the ball in freeze frames
# StatsBomb's match clock: the minute at which each period kicks off.
_PERIOD_KICK_OFF_MINUTE = {1: 0, 2: 45, 3: 90, 4: 105}
_FREEZE_FRAME_TYPES = ("player", "goalkeeper", "referee")  # other people: "other"
_FREEZE_FRAME_COLUMNS = [
    "frame_id", "period", "timestamp", "timestamp_seconds", "minute", "second",
    "homography_matrix", "visible_area", "player_id", "type", "class_name",
    "location", "frame_bbox", "confidence", "keeper", "team_id",
]  # fmt: skip
_TRACK_COLUMNS = [
    "frame", "time", "object", "track_id", "class_name", "team_id",
    "confidence", "x1", "y1", "x2", "y2", "pitch_x", "pitch_y",
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

    ``detections`` holds tracked people (players, goalkeepers, referees) and
    ``ball`` the ball; both are ``sv.Detections`` whose ``data`` carries:

    * ``class_name`` – ``(N,)`` str
    * ``pitch_xy`` – ``(N, 2)`` float pitch coordinates (NaN without homography);
      people are anchored at the bottom-centre of the box, the ball at its centre
    * ``team_id`` – ``(N,)`` int, ``NO_TEAM`` (-1) when unknown (people only)
    """

    index: int  # frame index in the source video
    timestamp: float  # seconds since the start of the video
    detections: sv.Detections
    ball: sv.Detections
    keypoints: sv.KeyPoints | None = None  # pitch keypoints in frame pixels
    homography: np.ndarray | None = None  # 3x3 frame -> pitch
    masks: ObjectMasks | None = None  # one per person (Pipeline(keep_masks=True))

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


def _data(
    detections: sv.Detections, key: str, shape: tuple[int, ...], fill: float
) -> np.ndarray:
    value = detections.data.get(key)
    return np.asarray(value) if value is not None else np.full(shape, fill)


@dataclass
class PipelineResult:
    """Output of :meth:`Pipeline.run`: one :class:`FrameResult` per processed frame."""

    frames: list[FrameResult]
    fps: float
    frame_size: tuple[int, int]  # (width, height) of the source video
    class_names: list[str]
    pitch: SoccerPitch = field(default_factory=SoccerPitch)
    video_path: Path | None = None

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, i: int) -> FrameResult:
        return self.frames[i]

    def __iter__(self) -> Iterator[FrameResult]:
        return iter(self.frames)

    def frame(self, video_index: int) -> FrameResult:
        """Look up the result of a frame by its index in the source video."""
        for result in self.frames:
            if result.index == video_index:
                return result
        raise KeyError(f"Frame {video_index} was not processed")

    @property
    def track_ids(self) -> list[int]:
        ids: set[int] = set()
        for f in self.frames:
            if f.detections.tracker_id is not None:
                ids.update(int(t) for t in f.detections.tracker_id)
        return sorted(ids)

    # ---------------------------------------------------------------- tables
    def to_dataframe(self) -> pd.DataFrame:
        """One row per object per frame.

        Columns: ``frame, time, object ("person" | "ball"), track_id, class_name,
        team_id, confidence, x1, y1, x2, y2, pitch_x, pitch_y``. ``track_id`` is
        ``UNTRACKED`` (-1) without a tracker and for the ball.
        """
        rows: list[dict] = []
        for f in self.frames:
            for kind, detections, pitch_xy, team_ids in _objects(f):
                for obj in _object_rows(detections, pitch_xy, team_ids):
                    x1, y1, x2, y2 = (float(v) for v in obj["xyxy"])
                    rows.append({
                        "frame": f.index, "time": f.timestamp, "object": kind,
                        "track_id": obj["track_id"], "class_name": obj["class_name"],
                        "team_id": obj["team_id"], "confidence": obj["confidence"],
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "pitch_x": float(obj["pitch_xy"][0]),
                        "pitch_y": float(obj["pitch_xy"][1]),
                    })  # fmt: skip
        return pd.DataFrame(rows, columns=_TRACK_COLUMNS)

    def to_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(path, index=False)
        return path

    def to_parquet(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False)
        return path

    def to_freeze_frames(
        self, period: int = 1, period_start: float | None = None
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
        ``BALL_PLAYER_ID`` for the ball). ``visible_area`` is the frame
        outline on the pitch as StatsBomb's flat ``[x1, y1, x2, y2, ...]``
        polygon (``None`` without a homography, or when part of the frame lies
        beyond the horizon, where the outline has no finite pitch position).
        """
        if period_start is None:
            if period not in _PERIOD_KICK_OFF_MINUTE:
                raise ValueError(
                    f"Period {period} has no StatsBomb kick-off time; pass period_start"
                )
            period_start = _PERIOD_KICK_OFF_MINUTE[period] * 60.0
        width, height = self.frame_size
        corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], float)
        centre = (width / 2, height / 2)
        rows: list[dict] = []
        for f in self.frames:
            t = period_start + f.timestamp
            visible_area = None
            if f.homography is not None:
                outline, in_front = project_points(corners, f.homography, centre)
                visible_area = outline.reshape(-1) if in_front.all() else None
            base = {
                "frame_id": f.index,
                "period": period,
                "timestamp": _format_clock(t),
                "timestamp_seconds": t,
                "minute": math.floor(t / 60),
                "second": math.floor(t % 60),
                "homography_matrix": _json(f.homography),
                "visible_area": _json(visible_area),
            }
            for kind, detections, pitch_xy, team_ids in _objects(f):
                for obj in _object_rows(detections, pitch_xy, team_ids):
                    name = obj["class_name"]
                    if kind == "ball":
                        kind_type, player_id = "ball", BALL_PLAYER_ID
                    else:
                        kind_type = name if name in _FREEZE_FRAME_TYPES else "other"
                        player_id = obj["track_id"]
                    rows.append(base | {
                        "player_id": player_id, "type": kind_type, "class_name": name,
                        "location": _json(obj["pitch_xy"]),
                        "frame_bbox": _json(obj["xyxy"].round().astype(int)),
                        "confidence": obj["confidence"],
                        "keeper": name == "goalkeeper", "team_id": obj["team_id"],
                    })  # fmt: skip
        return pd.DataFrame(rows, columns=_FREEZE_FRAME_COLUMNS)

    # ----------------------------------------------------------- persistence
    def export(
        self,
        out_dir: str | Path,
        *,
        period: int = 1,
        period_start: float | None = None,
    ) -> Path:
        """Write the run folder's data files to ``out_dir`` (created if missing).

        ``result.pkl`` (:meth:`save`), ``tracks.csv`` (:meth:`to_csv`) and
        ``freeze_frames.csv`` (:meth:`to_freeze_frames` with ``period`` and
        ``period_start``; ``None`` is the period's kick-off). Render the
        annotated video with :func:`tactifoot_vision.viz.render_video`, or
        write the whole run folder with :meth:`tactifoot_vision.run_file.RunFile.run`.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.save(out_dir / "result.pkl")
        self.to_csv(out_dir / "tracks.csv")
        self.to_freeze_frames(period, period_start).to_csv(
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
        """Read a result written by :meth:`save` (pickle: only load files you trust)."""
        with Path(path).open("rb") as fh:
            result = pickle.load(fh)
        if not isinstance(result, cls):
            raise TypeError(f"{path} does not contain a PipelineResult")
        return result


def _objects(
    f: FrameResult,
) -> Iterator[tuple[str, sv.Detections, np.ndarray, np.ndarray | None]]:
    """``(kind, detections, pitch_xy, team_ids)`` of a frame's people and its ball."""
    yield "person", f.detections, f.pitch_xy, f.team_ids
    yield "ball", f.ball, f.ball_xy, None


def _object_rows(
    detections: sv.Detections, pitch_xy: np.ndarray, team_ids: np.ndarray | None
) -> Iterator[dict]:
    """The fields every table row of an object shares, one dict per detection."""
    names = detections.data.get("class_name", [])
    for i in range(len(detections)):
        yield {
            "track_id": int(detections.tracker_id[i])
            if detections.tracker_id is not None
            else UNTRACKED,
            "class_name": str(names[i]) if len(names) else "",
            "team_id": int(team_ids[i]) if team_ids is not None else NO_TEAM,
            "confidence": _float(detections.confidence, i),
            "xyxy": detections.xyxy[i],
            "pitch_xy": pitch_xy[i],
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
