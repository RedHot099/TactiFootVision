import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class ThresholdSelection:
    threshold: float
    candidates: pd.DataFrame
    metrics: dict[str, float]
    composite_score: float


class ShotDirectionResolver:
    def __init__(self, *, post_window_seconds: float = 1.0) -> None:
        self.post_window_seconds = post_window_seconds

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()
        resolved = frame.sort_values("global_seconds").reset_index(drop=True).copy()
        times = resolved["global_seconds"].to_numpy(dtype=float)
        pitch_x = resolved["pitch_x"].to_numpy(dtype=float)
        pitch_y = resolved["pitch_y"].to_numpy(dtype=float)
        goal_x_values: list[float] = []
        direction_values: list[float] = []
        progress_values: list[float] = []
        distance_values: list[float] = []
        for index, seconds in enumerate(times):
            future_index = _future_index(
                times, index, seconds + self.post_window_seconds
            )
            now_x = float(pitch_x[index])
            now_y = float(pitch_y[index])
            future_x = float(pitch_x[future_index])
            future_y = float(pitch_y[future_index])
            progress_left = _distance(now_x, now_y, 0.0, 34.0) - _distance(
                future_x, future_y, 0.0, 34.0
            )
            progress_right = _distance(now_x, now_y, 105.0, 34.0) - _distance(
                future_x, future_y, 105.0, 34.0
            )
            if abs(progress_left - progress_right) < 1e-6:
                goal_x = 105.0 if now_x <= 52.5 else 0.0
            else:
                goal_x = 105.0 if progress_right > progress_left else 0.0
            direction = _direction(now_x, now_y, future_x, future_y, goal_x, 34.0)
            goal_progress = progress_right if goal_x == 105.0 else progress_left
            goal_x_values.append(goal_x)
            direction_values.append(direction)
            progress_values.append(goal_progress)
            distance_values.append(_distance(now_x, now_y, goal_x, 34.0))
        resolved["resolved_goal_x"] = goal_x_values
        resolved["resolved_goal_y"] = 34.0
        resolved["direction_to_resolved_goal"] = direction_values
        resolved["goal_progress_m"] = progress_values
        resolved["distance_to_resolved_goal"] = distance_values
        return resolved


class ShotWindowFeatureExtractor:
    def __init__(
        self,
        *,
        contact_pre_window_seconds: float = 0.5,
        post_shot_window_seconds: float = 1.0,
        long_shot_distance_m: float = 40.0,
    ) -> None:
        self.contact_pre_window_seconds = contact_pre_window_seconds
        self.post_shot_window_seconds = post_shot_window_seconds
        self.long_shot_distance_m = long_shot_distance_m

    def transform(self, candidate_features: pd.DataFrame) -> pd.DataFrame:
        if candidate_features.empty:
            return candidate_features.copy()
        frame = ShotDirectionResolver(
            post_window_seconds=self.post_shot_window_seconds
        ).transform(candidate_features)
        frame = frame.sort_values("global_seconds").reset_index(drop=True)
        times = frame["global_seconds"].to_numpy(dtype=float)
        speed = _series(frame, "ball_speed")
        acceleration = _series(frame, "ball_acceleration")
        contact = _series(frame, "contact_score")
        nearest = _series(frame, "nearest_player_distance", fill=np.nan)
        direction = _series(frame, "direction_to_resolved_goal")
        distance_to_goal = _series(frame, "distance_to_resolved_goal")
        goal_progress = _series(frame, "goal_progress_m")

        aggregates: dict[str, list[float | bool]] = {
            "pre_contact_score": [],
            "pre_contact_min_distance": [],
            "post_recontact_score": [],
            "post_goal_progress_m": [],
            "post_direction_consistency": [],
            "post_speed_mean": [],
            "post_speed_max": [],
            "tracks_observed_in_window": [],
            "contact_acceleration_order_score": [],
            "shot_zone_score": [],
            "long_shot_exception": [],
            "recontact_penalty": [],
        }
        for suffix in ("0_5", "1_0", "2_0"):
            aggregates[f"speed_max_{suffix}s"] = []
            aggregates[f"speed_mean_{suffix}s"] = []
            aggregates[f"acceleration_max_{suffix}s"] = []
            aggregates[f"contact_max_{suffix}s"] = []
            aggregates[f"direction_mean_{suffix}s"] = []

        for index, seconds in enumerate(times):
            pre_start = int(
                np.searchsorted(
                    times, seconds - self.contact_pre_window_seconds, "left"
                )
            )
            pre_end = index + 1
            post_end = int(
                np.searchsorted(times, seconds + self.post_shot_window_seconds, "right")
            )
            post_start = index
            pre_contact = contact[pre_start:pre_end]
            pre_nearest = nearest[pre_start:pre_end]
            post_contact = contact[min(index + 1, len(contact)) : post_end]
            post_direction = direction[post_start:post_end]
            post_speed = speed[post_start:post_end]

            pre_contact_score = _safe_max(pre_contact)
            post_recontact_score = _safe_max(post_contact)
            tracks_observed = bool(np.isfinite(pre_nearest).any())
            pre_min_distance = _safe_min(pre_nearest, default=np.nan)
            post_progress = _safe_max(goal_progress[post_start:post_end])
            direction_consistency = max(_safe_mean(post_direction), 0.0)
            post_speed_mean = _safe_mean(post_speed)
            post_speed_max = _safe_max(post_speed)
            acceleration_score = max(float(acceleration[index]), 0.0)
            if not tracks_observed:
                contact_order = 0.35 if acceleration_score > 0.0 else 0.0
            else:
                contact_order = pre_contact_score if acceleration_score > 0.0 else 0.0
            distance = float(distance_to_goal[index])
            long_exception = bool(
                distance > self.long_shot_distance_m
                and post_progress >= 3.0
                and post_speed_max >= max(float(np.nanpercentile(speed, 80)), 1.0)
            )
            zone_score = max(0.0, 1.0 - distance / self.long_shot_distance_m)
            if long_exception:
                zone_score = max(zone_score, 0.55)
            recontact_penalty = (
                0.25
                if post_recontact_score >= 0.35
                and (post_speed_mean < post_speed_max * 0.6 or post_speed_max < 18.0)
                else 0.0
            )

            aggregates["pre_contact_score"].append(pre_contact_score)
            aggregates["pre_contact_min_distance"].append(pre_min_distance)
            aggregates["post_recontact_score"].append(post_recontact_score)
            aggregates["post_goal_progress_m"].append(post_progress)
            aggregates["post_direction_consistency"].append(direction_consistency)
            aggregates["post_speed_mean"].append(post_speed_mean)
            aggregates["post_speed_max"].append(post_speed_max)
            aggregates["tracks_observed_in_window"].append(tracks_observed)
            aggregates["contact_acceleration_order_score"].append(contact_order)
            aggregates["shot_zone_score"].append(zone_score)
            aggregates["long_shot_exception"].append(long_exception)
            aggregates["recontact_penalty"].append(recontact_penalty)

            for seconds_window, suffix in ((0.5, "0_5"), (1.0, "1_0"), (2.0, "2_0")):
                start = int(np.searchsorted(times, seconds - seconds_window, "left"))
                end = int(np.searchsorted(times, seconds + seconds_window, "right"))
                aggregates[f"speed_max_{suffix}s"].append(_safe_max(speed[start:end]))
                aggregates[f"speed_mean_{suffix}s"].append(_safe_mean(speed[start:end]))
                aggregates[f"acceleration_max_{suffix}s"].append(
                    _safe_max(acceleration[start:end])
                )
                aggregates[f"contact_max_{suffix}s"].append(
                    _safe_max(contact[start:end])
                )
                aggregates[f"direction_mean_{suffix}s"].append(
                    _safe_mean(direction[start:end])
                )

        for column, values in aggregates.items():
            frame[column] = values
        return frame


class ShotPatternScorer:
    def __init__(self, *, long_shot_distance_m: float = 40.0) -> None:
        self.long_shot_distance_m = long_shot_distance_m

    def score(self, window_features: pd.DataFrame) -> pd.DataFrame:
        if window_features.empty:
            return window_features.copy()
        frame = window_features.copy()
        speed_scale = max(float(frame["ball_speed"].quantile(0.95) or 1.0), 1.0)
        acc_scale = max(
            float(frame["ball_acceleration"].clip(lower=0.0).quantile(0.95) or 1.0),
            1.0,
        )
        progress_scale = max(
            float(frame["post_goal_progress_m"].clip(lower=0.0).quantile(0.9) or 1.0),
            1.0,
        )
        frame["speed_pattern_score"] = (frame["ball_speed"] / speed_scale).clip(
            0.0, 1.0
        )
        frame["acceleration_pattern_score"] = (
            frame["ball_acceleration"].clip(lower=0.0) / acc_scale
        ).clip(0.0, 1.0)
        frame["direction_pattern_score"] = frame["direction_to_resolved_goal"].clip(
            lower=0.0, upper=1.0
        )
        frame["post_flight_score"] = 0.65 * (
            frame["post_goal_progress_m"].clip(lower=0.0) / progress_scale
        ).clip(0.0, 1.0) + 0.35 * frame["post_direction_consistency"].clip(0.0, 1.0)
        frame["contact_pattern_score"] = frame["contact_acceleration_order_score"].clip(
            0.0, 1.0
        )
        frame["raw_pattern_score"] = (
            0.22 * frame["speed_pattern_score"]
            + 0.18 * frame["acceleration_pattern_score"]
            + 0.18 * frame["contact_pattern_score"]
            + 0.18 * frame["direction_pattern_score"]
            + 0.16 * frame["post_flight_score"]
            + 0.08 * frame["shot_zone_score"].clip(0.0, 1.0)
        )
        frame["veto_multiplier"] = frame.apply(self._veto_multiplier, axis=1)
        frame["pattern_score"] = (
            frame["raw_pattern_score"]
            * frame["veto_multiplier"]
            * (1.0 - frame["recontact_penalty"].clip(0.0, 0.8))
        ).clip(0.0, 1.0)
        frame["precision_gate"] = frame.apply(self._precision_gate, axis=1)
        return frame

    def _veto_multiplier(self, row: pd.Series) -> float:
        multiplier = 1.0
        if (
            row["direction_to_resolved_goal"] < -0.25
            and row["post_goal_progress_m"] <= 0.0
        ):
            multiplier *= 0.45
        if (
            bool(row["tracks_observed_in_window"])
            and row["pre_contact_score"] <= 0.05
            and row["speed_pattern_score"] < 0.75
        ):
            multiplier *= 0.65
        if row["post_goal_progress_m"] <= 0.0 and row["direction_pattern_score"] < 0.1:
            multiplier *= 0.55
        if row["distance_to_resolved_goal"] > self.long_shot_distance_m and not bool(
            row["long_shot_exception"]
        ):
            multiplier *= 0.70
        return multiplier

    def _precision_gate(self, row: pd.Series) -> bool:
        if (
            row["direction_to_resolved_goal"] < -0.2
            and row["post_goal_progress_m"] <= 0.0
        ):
            return False
        has_contact = row["pre_contact_score"] > 0.05
        if bool(row["tracks_observed_in_window"]) and not has_contact:
            return bool(row["long_shot_exception"])
        if row["distance_to_resolved_goal"] > self.long_shot_distance_m:
            return bool(row["long_shot_exception"])
        return row["post_flight_score"] >= 0.05 or row["speed_pattern_score"] >= 0.55


class AdaptiveShotNms:
    def __init__(
        self,
        *,
        temporal_nms_seconds: float = 8.0,
        max_candidates: int = 80,
        max_candidates_per_half: int = 25,
        min_rebound_gap_seconds: float = 1.2,
    ) -> None:
        self.temporal_nms_seconds = temporal_nms_seconds
        self.max_candidates = max_candidates
        self.max_candidates_per_half = max_candidates_per_half
        self.min_rebound_gap_seconds = min_rebound_gap_seconds

    def select(
        self, scored: pd.DataFrame, *, score_column: str = "score"
    ) -> pd.DataFrame:
        if scored.empty:
            return scored.copy()
        selected: list[pd.Series] = []
        for _, row in scored.sort_values(score_column, ascending=False).iterrows():
            if self._suppressed(row, selected):
                continue
            selected.append(row)
            if len(selected) >= self.max_candidates:
                break
        if not selected:
            return scored.head(0).copy()
        frame = pd.DataFrame(selected)
        frame = self._cap_per_half(frame, score_column)
        return frame.sort_values("global_seconds").reset_index(drop=True)

    def _suppressed(self, row: pd.Series, selected: list[pd.Series]) -> bool:
        for other in selected:
            gap = abs(float(row["global_seconds"]) - float(other["global_seconds"]))
            if gap > self.temporal_nms_seconds:
                continue
            if self._rebound_allowed(row, other, gap):
                continue
            return True
        return False

    def _rebound_allowed(self, row: pd.Series, other: pd.Series, gap: float) -> bool:
        if gap < self.min_rebound_gap_seconds:
            return False
        row_contact = float(row.get("pre_contact_score", 0.0))
        other_contact = float(other.get("pre_contact_score", 0.0))
        row_progress = float(row.get("post_goal_progress_m", 0.0))
        other_progress = float(other.get("post_goal_progress_m", 0.0))
        return (
            row_contact >= 0.25
            and other_contact >= 0.25
            and row_progress >= 1.0
            and other_progress >= 1.0
        )

    def _cap_per_half(self, frame: pd.DataFrame, score_column: str) -> pd.DataFrame:
        if "part_index" not in frame.columns or self.max_candidates_per_half <= 0:
            return frame.head(self.max_candidates)
        capped = (
            frame.sort_values(score_column, ascending=False)
            .groupby("part_index", sort=False, group_keys=False)
            .head(self.max_candidates_per_half)
        )
        return capped.head(self.max_candidates)


class SoftCompositeThresholdSelector:
    def __init__(
        self,
        *,
        recall_floor_hit2: float = 0.78,
        min_hit1: float = 0.45,
        target_max_false_positives: int = 30,
    ) -> None:
        self.recall_floor_hit2 = recall_floor_hit2
        self.min_hit1 = min_hit1
        self.target_max_false_positives = target_max_false_positives

    def select(
        self,
        scored: pd.DataFrame,
        reference: pd.DataFrame,
        build_candidates: Callable[[pd.DataFrame], pd.DataFrame],
    ) -> ThresholdSelection:
        if scored.empty:
            empty = build_candidates(scored)
            return ThresholdSelection(
                1.0, empty, _candidate_metrics(empty, reference), 0.0
            )
        thresholds = _threshold_grid(scored["score"])
        best: ThresholdSelection | None = None
        for threshold in thresholds:
            filtered = scored[
                (scored["score"] >= threshold)
                & (scored["precision_gate"] | (scored["score"] >= 0.68))
            ]
            candidates = build_candidates(filtered)
            metrics = _candidate_metrics(candidates, reference)
            composite = self._composite(metrics)
            if best is None or composite > best.composite_score:
                best = ThresholdSelection(threshold, candidates, metrics, composite)
        if best is None:
            empty = build_candidates(scored.head(0))
            return ThresholdSelection(
                1.0, empty, _candidate_metrics(empty, reference), 0.0
            )
        return best

    def _composite(self, metrics: dict[str, float]) -> float:
        hit2 = metrics["hit@2s"]
        hit1 = metrics["hit@1s"]
        precision = metrics["precision@2s"]
        false_positives = metrics["false_positives"]
        fp_reduction = max(0.0, 1.0 - false_positives / 61.0)
        recall_gate = (
            1.0
            if hit2 >= self.recall_floor_hit2
            else max(0.0, hit2 / self.recall_floor_hit2)
        )
        hit1_gate = 1.0 if hit1 >= self.min_hit1 else max(0.0, hit1 / self.min_hit1)
        fp_target_bonus = (
            0.05 if false_positives <= self.target_max_false_positives else 0.0
        )
        return (
            recall_gate
            * hit1_gate
            * (0.40 * precision + 0.30 * fp_reduction + 0.20 * hit2 + 0.10 * hit1)
            + fp_target_bonus
        )


def _candidate_metrics(
    candidates: pd.DataFrame, reference: pd.DataFrame
) -> dict[str, float]:
    if reference.empty:
        return {
            "hit@0.5s": 0.0,
            "hit@1s": 0.0,
            "hit@2s": 0.0,
            "precision@2s": 0.0,
            "temporal_mae_seconds": 0.0,
            "false_positives": float(len(candidates)),
        }
    if candidates.empty:
        return {
            "hit@0.5s": 0.0,
            "hit@1s": 0.0,
            "hit@2s": 0.0,
            "precision@2s": 0.0,
            "temporal_mae_seconds": 0.0,
            "false_positives": 0.0,
        }
    errors = []
    matched_candidates: set[str] = set()
    for ref in reference.itertuples(index=False):
        nearest_idx = (
            (candidates["global_seconds"] - float(ref.reference_seconds)).abs().idxmin()
        )
        nearest = candidates.loc[nearest_idx]
        error = float(nearest["global_seconds"] - float(ref.reference_seconds))
        errors.append(error)
        if abs(error) <= 2.0:
            matched_candidates.add(str(nearest["shot_id"]))
    abs_errors = [abs(value) for value in errors]
    return {
        "hit@0.5s": sum(value <= 0.5 for value in abs_errors) / len(reference),
        "hit@1s": sum(value <= 1.0 for value in abs_errors) / len(reference),
        "hit@2s": sum(value <= 2.0 for value in abs_errors) / len(reference),
        "precision@2s": len(matched_candidates) / len(candidates)
        if len(candidates)
        else 0.0,
        "temporal_mae_seconds": float(np.mean(abs_errors)) if abs_errors else 0.0,
        "false_positives": float(max(len(candidates) - len(matched_candidates), 0)),
    }


def _threshold_grid(scores: pd.Series) -> list[float]:
    values = scores.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if values.empty:
        return [1.0]
    quantiles = values.quantile(np.linspace(0.05, 0.95, 19)).tolist()
    fixed = np.linspace(0.15, 0.85, 15).tolist()
    return sorted({round(float(value), 6) for value in [*quantiles, *fixed]})


def _future_index(times: np.ndarray, index: int, target_seconds: float) -> int:
    future = int(np.searchsorted(times, target_seconds, side="right")) - 1
    return min(max(future, index), len(times) - 1)


def _distance(x: float, y: float, goal_x: float, goal_y: float) -> float:
    return math.hypot(goal_x - x, goal_y - y)


def _direction(
    x: float,
    y: float,
    future_x: float,
    future_y: float,
    goal_x: float,
    goal_y: float,
) -> float:
    vx = future_x - x
    vy = future_y - y
    gx = goal_x - x
    gy = goal_y - y
    denom = math.hypot(vx, vy) * math.hypot(gx, gy)
    if denom <= 0.0:
        return 0.0
    return float((vx * gx + vy * gy) / denom)


def _series(frame: pd.DataFrame, column: str, *, fill: float = 0.0) -> np.ndarray:
    if column not in frame.columns:
        return np.full(len(frame), fill, dtype=float)
    return (
        frame[column]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(fill)
        .to_numpy(dtype=float)
    )


def _safe_max(values: np.ndarray, *, default: float = 0.0) -> float:
    if len(values) == 0:
        return default
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return default
    return float(np.max(finite))


def _safe_min(values: np.ndarray, *, default: float = 0.0) -> float:
    if len(values) == 0:
        return default
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return default
    return float(np.min(finite))


def _safe_mean(values: np.ndarray, *, default: float = 0.0) -> float:
    if len(values) == 0:
        return default
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return default
    return float(np.mean(finite))
