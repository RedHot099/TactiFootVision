from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


MATCHES_ROOT = Path("/home/kuba/projects/ball-vision/data/20232024")
STATS_FULL_ROOT = Path("results/project/raw/statsbomb_full")

RESULTS_ROOT = Path("results/project")
PIPELINE_ROOT = RESULTS_ROOT / "raw"
MODELS: list[str] = ["yolov8m", "yolo11m", "yolo12m", "rfdetr_base"]

# Pipeline coordinates currently come from inference configs that use default geometry 100x100.
# StatsBomb coordinates are in a 120x80 pitch convention.
PIPE_PITCH_LENGTH = 120.0
PIPE_PITCH_WIDTH = 80.0
SB_PITCH_LENGTH = 120.0
SB_PITCH_WIDTH = 80.0

MAX_MATCH_DISTANCE_M = None  # set to a float to enable thresholding
TIME_WINDOW_S = 0.5


@dataclass(frozen=True)
class ClipWindow:
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return max(0.0, float(self.end_s) - float(self.start_s))


def _read_clip_window_seconds(match_dir: Path) -> Optional[ClipWindow]:
    llc_path = match_dir / MATCH_CLIP_META_NAME
    if not llc_path.is_file():
        return None
    text = llc_path.read_text(encoding="utf-8", errors="ignore")
    start_match = re.search(r"start:\s*([0-9]+(?:\\.[0-9]+)?)", text)
    end_match = re.search(r"end:\s*([0-9]+(?:\\.[0-9]+)?)", text)
    if not start_match or not end_match:
        return None
    start_s = float(start_match.group(1))
    end_s = float(end_match.group(1))
    if end_s <= start_s:
        return None
    return ClipWindow(start_s=start_s, end_s=end_s)


def _pseudo_event_timestamp_s(index: int, total: int, clip: ClipWindow) -> float:
    if total <= 1:
        return float(clip.start_s)
    alpha = float(index) / float(total - 1)
    return float(clip.start_s + alpha * clip.duration_s)


def _to_minute_second(timestamp_s: float) -> Tuple[int, int]:
    minute = int(np.floor(timestamp_s / 60.0))
    second = int(np.floor(timestamp_s % 60.0))
    return minute, second


def _scale_pipe_to_sb(points_xy: np.ndarray) -> np.ndarray:
    if points_xy.size == 0:
        return points_xy
    # Pipeline is re-run with pitch dims 120x80, so this is now identity.
    return points_xy.astype(np.float32).copy()


def _timestamp_to_seconds(ts: Any) -> Optional[float]:
    if ts is None or (isinstance(ts, float) and not np.isfinite(ts)):
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    if not isinstance(ts, str) or not ts:
        return None
    try:
        hms, ms = ts.split(".", 1) if "." in ts else (ts, "0")
        h, m, s = hms.split(":")
        return int(h) * 3600.0 + int(m) * 60.0 + float(s) + float(f"0.{ms}")
    except Exception:
        return None


def _parse_json_xy(value: Any) -> Optional[Tuple[float, float]]:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    if isinstance(value, np.ndarray) and value.shape == (2,):
        try:
            return float(value[0]), float(value[1])
        except Exception:
            return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except Exception:
            return None
        if isinstance(parsed, list) and len(parsed) == 2:
            return float(parsed[0]), float(parsed[1])
    return None


def _extract_sb_points(freeze_frame: Any) -> np.ndarray:
    if freeze_frame is None:
        return np.empty((0, 2), dtype=np.float32)
    # parquet stores numpy array of dicts
    items: Iterable[Any]
    if isinstance(freeze_frame, np.ndarray):
        items = freeze_frame.tolist()
    elif isinstance(freeze_frame, list):
        items = freeze_frame
    else:
        return np.empty((0, 2), dtype=np.float32)

    pts: list[tuple[float, float]] = []
    for obj in items:
        if not isinstance(obj, dict):
            continue
        loc = obj.get("location")
        xy = _parse_json_xy(loc)
        if xy is None:
            continue
        pts.append(xy)
    if not pts:
        return np.empty((0, 2), dtype=np.float32)
    arr = np.array(pts, dtype=np.float32)
    mask = np.isfinite(arr).all(axis=1)
    arr = arr[mask]
    return arr


def _extract_pipeline_points(df_group: pd.DataFrame) -> np.ndarray:
    if df_group.empty:
        return np.empty((0, 2), dtype=np.float32)
    if "player_id" in df_group.columns:
        # Collapse multiple frames per-second to one point per tracked object.
        # Keep the latest record within this (minute, second) bucket.
        df_group = (
            df_group.sort_values("frame_id", kind="mergesort")
            .groupby("player_id", as_index=False, sort=False)
            .tail(1)
        )
    pts: list[tuple[float, float]] = []
    for v in df_group["location"].tolist():
        xy = _parse_json_xy(v)
        if xy is None:
            continue
        pts.append(xy)
    if not pts:
        return np.empty((0, 2), dtype=np.float32)
    arr = np.array(pts, dtype=np.float32)
    mask = np.isfinite(arr).all(axis=1)
    arr = arr[mask]
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    return _scale_pipe_to_sb(arr)


def _best_distances_trimmed(
    det_xy: np.ndarray, sb_xy: np.ndarray, sb_count: int
) -> np.ndarray:
    """
    For each detection pick the nearest StatsBomb point (no 1-1 constraint),
    then if detections exceed StatsBomb count keep the best `sb_count` distances.
    Finally apply `MAX_MATCH_DISTANCE_M` threshold.
    """
    if det_xy.size == 0 or sb_xy.size == 0:
        return np.empty((0,), dtype=np.float32)
    diffs = det_xy[:, None, :] - sb_xy[None, :, :]
    dist_matrix = np.sqrt((diffs**2).sum(axis=2))
    nearest = np.min(dist_matrix, axis=1)
    nearest = nearest[np.isfinite(nearest)]
    if nearest.size == 0:
        return np.empty((0,), dtype=np.float32)
    if MAX_MATCH_DISTANCE_M is not None:
        nearest = nearest[nearest <= float(MAX_MATCH_DISTANCE_M)]
        if nearest.size == 0:
            return np.empty((0,), dtype=np.float32)
    if det_xy.shape[0] > sb_count and sb_count > 0:
        nearest.sort()
        return nearest[:sb_count].astype(np.float32)
    return nearest.astype(np.float32)


def _pipeline_path(model: str, match_name: str) -> Path:
    return (
        PIPELINE_ROOT
        / model
        / "inference"
        / match_name
        / f"{model}_first_5_pipelinedata_p1.csv"
    )


def evaluate_match(model: str, match_dir: Path) -> pd.DataFrame:
    stats_path = STATS_FULL_ROOT / f"{match_dir.name}.parquet"
    if not stats_path.is_file():
        raise FileNotFoundError(f"Missing statsbomb_full parquet: {stats_path}")

    df_sb = pd.read_parquet(stats_path).reset_index(drop=True)
    required = {"period", "minute", "second", "freeze_frame"}
    missing = required - set(df_sb.columns)
    if missing:
        raise ValueError(f"statsbomb_full missing columns {sorted(missing)}: {stats_path}")
    total_events = len(df_sb)
    if "timestamp_seconds" not in df_sb.columns:
        df_sb["timestamp_seconds"] = df_sb["timestamp"].apply(_timestamp_to_seconds)

    pipe_path = _pipeline_path(model, match_dir.name)
    if not pipe_path.is_file():
        raise FileNotFoundError(f"Missing pipeline CSV: {pipe_path}")

    df_pipe = pd.read_csv(pipe_path, low_memory=False)
    # keep only objects that should correspond to freeze_frame players (ignore ball)
    df_pipe = df_pipe[df_pipe["type"].isin(["player", "goalkeeper", "referee"])].copy()
    df_pipe["minute"] = pd.to_numeric(df_pipe["minute"], errors="coerce").fillna(-1).astype(int)
    df_pipe["second"] = pd.to_numeric(df_pipe["second"], errors="coerce").fillna(-1).astype(int)
    df_pipe["timestamp_seconds"] = pd.to_numeric(
        df_pipe.get("timestamp_seconds"), errors="coerce"
    )
    df_pipe["period"] = pd.to_numeric(df_pipe.get("period"), errors="coerce").fillna(-1).astype(int)
    pipe_groups = df_pipe.groupby(["period", "minute", "second"], sort=False)

    rows: list[Dict[str, Any]] = []
    for idx in range(total_events):
        minute = int(df_sb.loc[idx, "minute"])
        second = int(df_sb.loc[idx, "second"])
        period = int(df_sb.loc[idx, "period"]) if pd.notna(df_sb.loc[idx, "period"]) else None
        event_ts = df_sb.loc[idx, "timestamp_seconds"]
        sb_points = _extract_sb_points(df_sb.loc[idx, "freeze_frame"])
        sb_count = int(len(sb_points))

        pipe_pts = np.empty((0, 2), dtype=np.float32)
        if period is not None and event_ts is not None and np.isfinite(event_ts):
            window = df_pipe[
                (df_pipe["period"] == period)
                & df_pipe["timestamp_seconds"].notna()
                & (df_pipe["timestamp_seconds"] >= float(event_ts) - TIME_WINDOW_S)
                & (df_pipe["timestamp_seconds"] <= float(event_ts) + TIME_WINDOW_S)
            ].copy()
            if not window.empty:
                window["abs_dt"] = (window["timestamp_seconds"] - float(event_ts)).abs()
                window = (
                    window.sort_values(["abs_dt", "frame_id"], kind="mergesort")
                    .groupby("player_id", as_index=False, sort=False)
                    .head(1)
                )
                pipe_pts = _extract_pipeline_points(window)
        if pipe_pts.size == 0:
            key = (period if period is not None else -1, minute, second)
            if key in pipe_groups.groups:
                pipe_pts = _extract_pipeline_points(pipe_groups.get_group(key))

        dists = _best_distances_trimmed(pipe_pts, sb_points, sb_count)
        rows.append(
            {
                "model": model,
                "match": match_dir.name,
                "match_id": int(df_sb.loc[idx, "match_id"])
                if "match_id" in df_sb.columns and pd.notna(df_sb.loc[idx, "match_id"])
                else None,
                "event_idx": int(idx),
                "period": period,
                "timestamp": df_sb.loc[idx, "timestamp"] if "timestamp" in df_sb.columns else None,
                "timestamp_seconds": float(event_ts) if event_ts is not None and np.isfinite(event_ts) else None,
                "minute": int(minute),
                "second": int(second),
                "sb_count": sb_count,
                "det_count": int(len(pipe_pts)),
                "matched_count": int(len(dists)),
                "coverage": float(len(dists) / sb_count) if sb_count else None,
                "mean_dist": float(np.mean(dists)) if len(dists) else None,
                "median_dist": float(np.median(dists)) if len(dists) else None,
                "p95_dist": float(np.quantile(dists, 0.95)) if len(dists) else None,
                "abs_count_diff": int(abs(len(pipe_pts) - sb_count)),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    if not MATCHES_ROOT.is_dir():
        raise FileNotFoundError(f"Missing MATCHES_ROOT: {MATCHES_ROOT}")

    match_dirs = sorted([p for p in MATCHES_ROOT.iterdir() if p.is_dir()])
    if not match_dirs:
        raise RuntimeError(f"No match dirs found under: {MATCHES_ROOT}")

    out_root = RESULTS_ROOT / "numeric" / "statsbomb360_eval"
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[Dict[str, Any]] = []
    for model in MODELS:
        model_out = out_root / model
        model_out.mkdir(parents=True, exist_ok=True)
        for match_dir in match_dirs:
            df = evaluate_match(model, match_dir)
            per_match_path = model_out / f"{match_dir.name}.parquet"
            df.to_parquet(per_match_path, index=False)

            summary_rows.append(
                {
                    "model": model,
                    "match": match_dir.name,
                    "events": int(len(df)),
                    "mean_mean_dist": float(df["mean_dist"].dropna().mean())
                    if df["mean_dist"].notna().any()
                    else None,
                    "mean_median_dist": float(df["median_dist"].dropna().mean())
                    if df["median_dist"].notna().any()
                    else None,
                    "mean_coverage": float(df["coverage"].dropna().mean())
                    if df["coverage"].notna().any()
                    else None,
                    "mean_abs_count_diff": float(df["abs_count_diff"].mean())
                    if not df.empty
                    else None,
                }
            )

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(out_root / "summary.csv", index=False)


if __name__ == "__main__":
    main()
