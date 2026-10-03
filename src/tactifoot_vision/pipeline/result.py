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

from tactifoot_vision.pitch.homography import frame_to_pitch
from tactifoot_vision.pitch.pitch import SoccerPitch

NO_TEAM = -1


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
        for mask, (x, y), crop in zip(dense, self.origins, self.crops, strict=True):
            crop = crop[: height - y, : width - x]  # a crop may reach past the frame
            mask[y : y + crop.shape[0], x : x + crop.shape[1]] = crop
        return dense

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
        team_id, confidence, x1, y1, x2, y2, pitch_x, pitch_y``.
        """
        rows: list[dict] = []
        for f in self.frames:
            rows.extend(_rows(f, f.detections, "person", f.pitch_xy, f.team_ids))
            rows.extend(_rows(f, f.ball, "ball", f.ball_xy, None))
        columns = [
            "frame", "time", "object", "track_id", "class_name", "team_id",
            "confidence", "x1", "y1", "x2", "y2", "pitch_x", "pitch_y",
        ]  # fmt: skip
        return pd.DataFrame(rows, columns=columns)

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
        self, period: int = 1, period_start: float = 0.0
    ) -> pd.DataFrame:
        """StatsBomb-360-like table: one row per object with JSON-encoded geometry.

        Columns match what :func:`tactifoot_vision.evaluation.compare_with_statsbomb`
        expects (``period, minute, second, location, type, player_id, ...``).
        ``period_start`` is the match clock (seconds) at the first video frame.

        ``minute`` and ``second`` follow StatsBomb's match clock (the second
        half starts at minute 45). ``timestamp`` (``HH:MM:SS.mmm``) and
        ``timestamp_seconds`` are that match clock too, unlike StatsBomb's
        event ``timestamp``, which restarts every period. ``visible_area`` is
        the frame outline on the pitch as StatsBomb's flat
        ``[x1, y1, x2, y2, ...]`` polygon (``None`` without a homography).
        """
        corners = np.array(
            [
                [0, 0],
                [self.frame_size[0], 0],
                list(self.frame_size),
                [0, self.frame_size[1]],
            ],
            dtype=np.float32,
        )
        rows: list[dict] = []
        for f in self.frames:
            t = period_start + f.timestamp
            base = {
                "frame_id": f.index,
                "period": period,
                "timestamp": _format_clock(t),
                "timestamp_seconds": t,
                "minute": math.floor(t / 60),
                "second": math.floor(t % 60),
                "homography_matrix": _json(f.homography),
                "visible_area": _json(
                    frame_to_pitch(corners, f.homography).reshape(-1)
                    if f.homography is not None
                    else None
                ),
            }
            names = f.detections.data.get("class_name", [])
            for i in range(len(f.detections)):
                name = str(names[i]) if len(names) else ""
                rows.append(base | {
                    "player_id": int(f.detections.tracker_id[i]) if f.detections.tracker_id is not None else -1,
                    "type": name if name in {"player", "goalkeeper", "referee"} else "other",
                    "class_name": name,
                    "location": _json(f.pitch_xy[i]),
                    "frame_bbox": _json(f.detections.xyxy[i].round().astype(int)),
                    "confidence": _float(f.detections.confidence, i),
                    "keeper": name == "goalkeeper",
                    "team_id": int(f.team_ids[i]),
                })  # fmt: skip
            for i in range(len(f.ball)):
                rows.append(base | {
                    "player_id": -99, "type": "ball", "class_name": "ball",
                    "location": _json(f.ball_xy[i]),
                    "frame_bbox": _json(f.ball.xyxy[i].round().astype(int)),
                    "confidence": _float(f.ball.confidence, i),
                    "keeper": False, "team_id": NO_TEAM,
                })  # fmt: skip
        return pd.DataFrame(rows)

    # ----------------------------------------------------------- persistence
    def export(
        self, out_dir: str | Path, *, period: int = 1, period_start: float = 0.0
    ) -> Path:
        """Write the run folder's data files to ``out_dir`` (created if missing).

        ``result.pkl`` (:meth:`save`), ``tracks.csv`` (:meth:`to_csv`) and
        ``freeze_frames.csv`` (:meth:`to_freeze_frames` with ``period`` and
        ``period_start``). Render the annotated video with
        :func:`tactifoot_vision.viz.render_video`.
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


def _rows(
    f: FrameResult,
    detections: sv.Detections,
    kind: str,
    pitch_xy: np.ndarray,
    team_ids: np.ndarray | None,
) -> list[dict]:
    names = detections.data.get("class_name", [])
    rows = []
    for i in range(len(detections)):
        x1, y1, x2, y2 = (float(v) for v in detections.xyxy[i])
        rows.append({
            "frame": f.index, "time": f.timestamp, "object": kind,
            "track_id": int(detections.tracker_id[i]) if detections.tracker_id is not None else -1,
            "class_name": str(names[i]) if len(names) else "",
            "team_id": int(team_ids[i]) if team_ids is not None else NO_TEAM,
            "confidence": _float(detections.confidence, i),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "pitch_x": float(pitch_xy[i, 0]), "pitch_y": float(pitch_xy[i, 1]),
        })  # fmt: skip
    return rows


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
