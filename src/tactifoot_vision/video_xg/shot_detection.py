import math
from typing import Any

import numpy as np
import pandas as pd


class VideoShotCandidateGenerator:
    def __init__(
        self,
        *,
        contact_distance_m: float = 2.5,
        min_candidate_confidence: float = 0.25,
        temporal_nms_seconds: float = 8.0,
        max_candidates: int = 80,
    ) -> None:
        self.contact_distance_m = contact_distance_m
        self.min_candidate_confidence = min_candidate_confidence
        self.temporal_nms_seconds = temporal_nms_seconds
        self.max_candidates = max_candidates

    def generate(
        self, ball_trajectory: pd.DataFrame, tracks: pd.DataFrame
    ) -> pd.DataFrame:
        if ball_trajectory.empty:
            return _empty_candidates()
        frame = ball_trajectory.sort_values("global_seconds").copy()
        frame["prev_pitch_x"] = frame["pitch_x"].shift(1)
        frame["prev_pitch_y"] = frame["pitch_y"].shift(1)
        frame["prev_seconds"] = frame["global_seconds"].shift(1)
        frame["next_pitch_x"] = frame["pitch_x"].shift(-1)
        frame["next_pitch_y"] = frame["pitch_y"].shift(-1)
        frame["next_seconds"] = frame["global_seconds"].shift(-1)
        frame["speed_after"] = frame.apply(_speed_after, axis=1)
        frame["speed_before"] = frame.apply(_speed_before, axis=1)
        frame["acceleration"] = (frame["speed_after"] - frame["speed_before"]).fillna(
            0.0
        )
        speed_scale = max(float(frame["speed_after"].quantile(0.95) or 1.0), 1.0)
        accel_scale = max(float(frame["acceleration"].quantile(0.95) or 1.0), 1.0)
        rows: list[dict[str, Any]] = []
        for row in frame.itertuples(index=False):
            nearest = _nearest_player_distance(row, tracks)
            goal_x = 105.0 if row.pitch_x >= 52.5 else 0.0
            direction = _direction_to_goal(row, goal_x)
            distance_to_goal = math.hypot(goal_x - row.pitch_x, 34.0 - row.pitch_y)
            contact_score = (
                0.0
                if nearest is None
                else max(0.0, 1.0 - nearest / self.contact_distance_m)
            )
            speed_score = min(max(row.speed_after / speed_scale, 0.0), 1.0)
            acceleration_score = min(max(row.acceleration / accel_scale, 0.0), 1.0)
            direction_score = max(0.0, direction)
            distance_score = max(0.0, 1.0 - distance_to_goal / 35.0)
            conditions = [
                speed_score >= 0.5,
                acceleration_score >= 0.5,
                contact_score > 0.0,
                direction_score > 0.0,
                distance_score > 0.0,
            ]
            if sum(conditions) < 3:
                continue
            score = (
                0.30 * speed_score
                + 0.20 * acceleration_score
                + 0.20 * contact_score
                + 0.20 * direction_score
                + 0.10 * distance_score
            )
            if score < self.min_candidate_confidence:
                continue
            rows.append(
                {
                    "shot_id": f"video-shot-{len(rows) + 1:04d}",
                    "global_frame_index": int(row.global_frame_index),
                    "global_seconds": float(row.global_seconds),
                    "part_index": int(row.part_index),
                    "part_frame_index": int(row.part_frame_index),
                    "score": float(score),
                    "confidence": float(score),
                    "source": "contact_kinematic",
                    "nearest_player_distance": nearest,
                    "ball_speed": float(row.speed_after),
                    "ball_direction_to_goal": float(direction),
                }
            )
        candidates = pd.DataFrame(rows)
        if candidates.empty:
            return _empty_candidates()
        return TemporalShotRanker(
            temporal_nms_seconds=self.temporal_nms_seconds,
            max_candidates=self.max_candidates,
        ).rank(candidates)


class ContactKinematicShotDetector(VideoShotCandidateGenerator):
    pass


class TemporalShotRanker:
    def __init__(self, *, temporal_nms_seconds: float = 8.0, max_candidates: int = 80):
        self.temporal_nms_seconds = temporal_nms_seconds
        self.max_candidates = max_candidates

    def rank(self, candidates: pd.DataFrame) -> pd.DataFrame:
        if candidates.empty:
            return _empty_candidates()
        selected: list[dict[str, Any]] = []
        for row in candidates.sort_values("score", ascending=False).to_dict("records"):
            if any(
                abs(float(row["global_seconds"]) - float(other["global_seconds"]))
                <= self.temporal_nms_seconds
                for other in selected
            ):
                continue
            selected.append(row)
            if len(selected) >= self.max_candidates:
                break
        return (
            pd.DataFrame(selected).sort_values("global_seconds").reset_index(drop=True)
        )


def _speed_after(row: pd.Series) -> float:
    if pd.isna(row["next_seconds"]) or row["next_seconds"] <= row["global_seconds"]:
        return 0.0
    return float(
        math.hypot(
            row["next_pitch_x"] - row["pitch_x"],
            row["next_pitch_y"] - row["pitch_y"],
        )
        / (row["next_seconds"] - row["global_seconds"])
    )


def _speed_before(row: pd.Series) -> float:
    if pd.isna(row["prev_seconds"]) or row["global_seconds"] <= row["prev_seconds"]:
        return 0.0
    return float(
        math.hypot(
            row["pitch_x"] - row["prev_pitch_x"],
            row["pitch_y"] - row["prev_pitch_y"],
        )
        / (row["global_seconds"] - row["prev_seconds"])
    )


def _nearest_player_distance(row: object, tracks: pd.DataFrame) -> float | None:
    if tracks.empty:
        return None
    same_frame = tracks[
        (tracks["global_frame_index"] == row.global_frame_index)
        & (tracks["class_name"].isin(["player", "goalkeeper"]))
    ]
    if same_frame.empty:
        return None
    distances = np.hypot(
        same_frame["pitch_x"] - row.pitch_x, same_frame["pitch_y"] - row.pitch_y
    )
    return float(distances.min())


def _direction_to_goal(row: object, goal_x: float) -> float:
    if pd.isna(row.next_pitch_x) or pd.isna(row.next_pitch_y):
        return 0.0
    vx = float(row.next_pitch_x - row.pitch_x)
    vy = float(row.next_pitch_y - row.pitch_y)
    gx = float(goal_x - row.pitch_x)
    gy = float(34.0 - row.pitch_y)
    denom = math.hypot(vx, vy) * math.hypot(gx, gy)
    if denom <= 0.0:
        return 0.0
    return float((vx * gx + vy * gy) / denom)


def _empty_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "shot_id",
            "global_frame_index",
            "global_seconds",
            "part_index",
            "part_frame_index",
            "score",
            "confidence",
            "source",
            "nearest_player_distance",
            "ball_speed",
            "ball_direction_to_goal",
        ]
    )
