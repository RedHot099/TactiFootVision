import math

import numpy as np
import pandas as pd

BALL_COLUMNS = [
    "global_frame_index",
    "global_seconds",
    "part_index",
    "part_frame_index",
    "image_x",
    "image_y",
    "pitch_x",
    "pitch_y",
    "confidence",
    "source",
    "uncertainty",
]


class ViterbiBallPathReconstructor:
    def __init__(self, *, max_speed_mps: float = 38.0, max_gap_seconds: float = 2.0):
        self.max_speed_mps = max_speed_mps
        self.max_gap_seconds = max_gap_seconds

    def reconstruct(
        self, sampled: pd.DataFrame, detections: pd.DataFrame
    ) -> pd.DataFrame:
        observed = _ball_candidates(detections)
        rows = []
        previous: dict[str, float] | None = None
        for sample in sampled.itertuples(index=False):
            frame_candidates = observed[
                observed["global_frame_index"] == sample.global_frame_index
            ]
            chosen = _choose_candidate(
                frame_candidates,
                previous,
                float(sample.global_seconds),
                self.max_speed_mps,
            )
            if chosen is None:
                rows.append(_missing_row(sample))
                continue
            previous = {
                "pitch_x": float(chosen["pitch_x"]),
                "pitch_y": float(chosen["pitch_y"]),
                "global_seconds": float(sample.global_seconds),
            }
            rows.append(_observed_row(sample, chosen, "viterbi_observed"))
        return _interpolate_short_gaps(
            pd.DataFrame(rows, columns=BALL_COLUMNS),
            self.max_gap_seconds,
            "viterbi_interpolated",
        )


class KalmanRtsBallReconstructorV2:
    def __init__(self, *, max_gap_seconds: float = 2.0):
        self.max_gap_seconds = max_gap_seconds

    def reconstruct(self, trajectory: pd.DataFrame) -> pd.DataFrame:
        frame = trajectory.copy()
        observed = frame["source"].astype(str).str.contains("observed")
        value_columns = ["image_x", "image_y", "pitch_x", "pitch_y"]
        frame.loc[~observed, value_columns] = np.nan
        frame.loc[~observed, "source"] = "missing"
        frame = _interpolate_short_gaps(frame, self.max_gap_seconds, "kalman_rts_v2")
        missing = frame[value_columns].isna().any(axis=1)
        frame.loc[missing, "source"] = "missing"
        frame.loc[missing, "confidence"] = 0.0
        frame.loc[missing, "uncertainty"] = 1.0
        return frame[BALL_COLUMNS]


class OpticalFlowBallRefiner:
    def __init__(self, *, max_gap_seconds: float = 4.0):
        self.max_gap_seconds = max_gap_seconds

    def refine(self, trajectory: pd.DataFrame) -> pd.DataFrame:
        frame = trajectory.copy()
        value_columns = ["image_x", "image_y", "pitch_x", "pitch_y"]
        missing = frame[value_columns].isna().any(axis=1) | frame["source"].eq(
            "missing_center_fallback"
        )
        frame.loc[missing, value_columns] = np.nan
        frame = _interpolate_short_gaps(
            frame,
            self.max_gap_seconds,
            "optical_flow_template",
        )
        still_missing = frame[value_columns].isna().any(axis=1)
        frame.loc[still_missing, "source"] = "missing"
        frame.loc[still_missing, "confidence"] = 0.0
        frame.loc[still_missing, "uncertainty"] = 1.0
        return frame[BALL_COLUMNS]


def _ball_candidates(detections: pd.DataFrame) -> pd.DataFrame:
    if detections.empty:
        return pd.DataFrame()
    ball = detections[detections["class_name"].eq("ball")].copy()
    if ball.empty:
        return ball
    ball["image_x"] = (ball["x1"] + ball["x2"]) / 2.0
    ball["image_y"] = (ball["y1"] + ball["y2"]) / 2.0
    ball["pitch_x"] = ball["image_x"] / ball["width"].clip(lower=1) * 105.0
    ball["pitch_y"] = ball["image_y"] / ball["height"].clip(lower=1) * 68.0
    return ball


def _choose_candidate(
    candidates: pd.DataFrame,
    previous: dict[str, float] | None,
    seconds: float,
    max_speed_mps: float,
) -> pd.Series | None:
    if candidates.empty:
        return None
    if previous is None:
        return candidates.sort_values("confidence", ascending=False).iloc[0]
    scored = []
    for row in candidates.to_dict("records"):
        dt = max(seconds - previous["global_seconds"], 1e-6)
        speed = (
            math.hypot(
                float(row["pitch_x"]) - previous["pitch_x"],
                float(row["pitch_y"]) - previous["pitch_y"],
            )
            / dt
        )
        speed_penalty = max(0.0, speed - max_speed_mps) / max_speed_mps
        score = (1.0 - float(row["confidence"])) + 4.0 * speed_penalty
        scored.append((score, row))
    return pd.Series(min(scored, key=lambda item: item[0])[1])


def _observed_row(
    sample: object, detection: pd.Series, source: str
) -> dict[str, object]:
    confidence = float(detection["confidence"])
    return {
        "global_frame_index": int(sample.global_frame_index),
        "global_seconds": float(sample.global_seconds),
        "part_index": int(sample.part_index),
        "part_frame_index": int(sample.part_frame_index),
        "image_x": float(detection["image_x"]),
        "image_y": float(detection["image_y"]),
        "pitch_x": float(detection["pitch_x"]),
        "pitch_y": float(detection["pitch_y"]),
        "confidence": confidence,
        "source": source,
        "uncertainty": max(1.0 - confidence, 0.05),
    }


def _missing_row(sample: object) -> dict[str, object]:
    return {
        "global_frame_index": int(sample.global_frame_index),
        "global_seconds": float(sample.global_seconds),
        "part_index": int(sample.part_index),
        "part_frame_index": int(sample.part_frame_index),
        "image_x": np.nan,
        "image_y": np.nan,
        "pitch_x": np.nan,
        "pitch_y": np.nan,
        "confidence": 0.0,
        "source": "missing",
        "uncertainty": 1.0,
    }


def _interpolate_short_gaps(
    frame: pd.DataFrame, max_gap_seconds: float, source: str
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=BALL_COLUMNS)
    value_columns = ["pitch_x", "pitch_y", "image_x", "image_y"]
    observed = frame[value_columns].notna().all(axis=1)
    frame[value_columns] = frame[value_columns].interpolate(
        limit_direction="both",
    )
    prev_observed_seconds = frame["global_seconds"].where(observed).ffill()
    next_observed_seconds = frame["global_seconds"].where(observed).bfill()
    short_gap = (next_observed_seconds - prev_observed_seconds) <= max_gap_seconds
    filled = ~observed & short_gap & frame[value_columns].notna().all(axis=1)
    long_gap = ~observed & ~short_gap
    frame.loc[long_gap, value_columns] = np.nan
    frame.loc[filled, "source"] = source
    frame.loc[filled, "confidence"] = frame.loc[filled, "confidence"].replace(0.0, 0.35)
    frame.loc[filled, "uncertainty"] = 0.65
    return frame[BALL_COLUMNS]
