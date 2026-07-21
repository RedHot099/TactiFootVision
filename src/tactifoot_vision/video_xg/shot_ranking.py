import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from tactifoot_vision.video_xg.shot_detection import TemporalShotRanker
from tactifoot_vision.video_xg.shot_quality import (
    AdaptiveShotNms,
    ShotPatternScorer,
    ShotWindowFeatureExtractor,
    SoftCompositeThresholdSelector,
)

CANDIDATE_COLUMNS = [
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


@dataclass(frozen=True, slots=True)
class ShotRankingMetrics:
    hit_05: float
    hit_1: float
    hit_2: float
    precision_2: float
    temporal_mae_seconds: float
    false_positives: float


class ShotCandidateFeatureExtractor:
    def transform(self, ball: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
        if ball.empty:
            return pd.DataFrame()
        frame = ball.sort_values("global_seconds").copy()
        track_index = _track_index(tracks)
        frame["prev_pitch_x"] = frame["pitch_x"].shift(1)
        frame["prev_pitch_y"] = frame["pitch_y"].shift(1)
        frame["prev_seconds"] = frame["global_seconds"].shift(1)
        frame["next_pitch_x"] = frame["pitch_x"].shift(-1)
        frame["next_pitch_y"] = frame["pitch_y"].shift(-1)
        frame["next_seconds"] = frame["global_seconds"].shift(-1)
        frame["ball_speed"] = frame.apply(_speed_after, axis=1)
        frame["ball_speed_before"] = frame.apply(_speed_before, axis=1)
        frame["ball_acceleration"] = (
            frame["ball_speed"] - frame["ball_speed_before"]
        ).fillna(0.0)
        frame["nearest_player_distance"] = [
            _nearest_player_distance(row, track_index)
            for row in frame.itertuples(index=False)
        ]
        frame["goal_x"] = np.where(frame["pitch_x"] >= 52.5, 105.0, 0.0)
        frame["distance_to_goal"] = np.hypot(
            frame["goal_x"] - frame["pitch_x"], 34.0 - frame["pitch_y"]
        )
        frame["ball_direction_to_goal"] = frame.apply(_direction_to_goal, axis=1)
        frame["contact_score"] = frame["nearest_player_distance"].apply(
            lambda value: 0.0 if pd.isna(value) else max(0.0, 1.0 - float(value) / 2.5)
        )
        frame["distance_score"] = (1.0 - frame["distance_to_goal"] / 35.0).clip(
            lower=0.0,
            upper=1.0,
        )
        return frame


class RuleSweepShotRanker:
    def __init__(self, *, temporal_nms_seconds: float = 8.0, max_candidates: int = 80):
        self.temporal_nms_seconds = temporal_nms_seconds
        self.max_candidates = max_candidates

    def rank(
        self,
        candidate_features: pd.DataFrame,
        reference: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if candidate_features.empty:
            return _empty_candidates()
        variants = [
            {"speed": 0.45, "acc": 0.35, "contact": 0.0, "direction": 0.0},
            {"speed": 0.35, "acc": 0.25, "contact": 0.0, "direction": -0.2},
            {"speed": 0.25, "acc": 0.20, "contact": 0.0, "direction": -0.5},
        ]
        best: pd.DataFrame | None = None
        best_score = -1.0
        for variant in variants:
            ranked = _rule_candidates(
                candidate_features,
                variant,
                self.temporal_nms_seconds,
                self.max_candidates,
                source="rule_sweep",
            )
            score = (
                evaluate_shot_candidates(ranked, reference).hit_2
                if reference is not None
                else float(len(ranked))
            )
            if score > best_score:
                best = ranked
                best_score = score
        return best if best is not None else _empty_candidates()


class LearnedTemporalShotRanker:
    def __init__(self, *, temporal_nms_seconds: float = 8.0, max_candidates: int = 80):
        self.temporal_nms_seconds = temporal_nms_seconds
        self.max_candidates = max_candidates

    def rank(
        self, candidate_features: pd.DataFrame, reference: pd.DataFrame
    ) -> pd.DataFrame:
        if candidate_features.empty or reference.empty:
            return _empty_candidates()
        features = _model_columns(candidate_features)
        labels = _labels(candidate_features, reference, tolerance_seconds=2.0)
        if len(set(labels.tolist())) < 2:
            return _rule_candidates(
                candidate_features,
                {"speed": 0.35, "acc": 0.25, "contact": 0.0, "direction": -0.2},
                self.temporal_nms_seconds,
                self.max_candidates,
                source="learned_temporal_fallback",
            )
        model = HistGradientBoostingClassifier(max_iter=30, random_state=0)
        model.fit(features, labels)
        scores = model.predict_proba(features)[:, 1]
        frame = candidate_features.copy()
        frame["score"] = scores
        frame = frame[frame["score"] > 0.05]
        return _candidates_from_scored(
            frame,
            self.temporal_nms_seconds,
            self.max_candidates,
            source="learned_temporal",
        )


class HighRecallCascadeShotRanker:
    def __init__(
        self,
        *,
        temporal_nms_seconds: float = 8.0,
        max_candidates: int = 80,
        max_candidates_per_half: int = 25,
        contact_pre_window_seconds: float = 0.5,
        post_shot_window_seconds: float = 1.0,
        long_shot_distance_m: float = 40.0,
    ) -> None:
        self.temporal_nms_seconds = temporal_nms_seconds
        self.max_candidates = max_candidates
        self.max_candidates_per_half = max_candidates_per_half
        self.contact_pre_window_seconds = contact_pre_window_seconds
        self.post_shot_window_seconds = post_shot_window_seconds
        self.long_shot_distance_m = long_shot_distance_m

    def rank(self, candidate_features: pd.DataFrame) -> pd.DataFrame:
        scored = _quality_scored_features(
            candidate_features,
            contact_pre_window_seconds=self.contact_pre_window_seconds,
            post_shot_window_seconds=self.post_shot_window_seconds,
            long_shot_distance_m=self.long_shot_distance_m,
        )
        if scored.empty:
            return _empty_candidates()
        filtered = scored[
            (scored["score"] >= 0.24)
            & (scored["precision_gate"] | (scored["score"] >= 0.62))
        ]
        return _candidates_from_scored_adaptive(
            filtered,
            self.temporal_nms_seconds,
            self.max_candidates,
            self.max_candidates_per_half,
            source="high_recall_cascade",
        )


class HardNegativeCalibratedShotRanker:
    def __init__(
        self,
        *,
        temporal_nms_seconds: float = 8.0,
        max_candidates: int = 80,
        max_candidates_per_half: int = 25,
        recall_floor_hit2: float = 0.78,
        min_hit1: float = 0.45,
        target_max_false_positives: int = 30,
        contact_pre_window_seconds: float = 0.5,
        post_shot_window_seconds: float = 1.0,
        long_shot_distance_m: float = 40.0,
    ) -> None:
        self.temporal_nms_seconds = temporal_nms_seconds
        self.max_candidates = max_candidates
        self.max_candidates_per_half = max_candidates_per_half
        self.recall_floor_hit2 = recall_floor_hit2
        self.min_hit1 = min_hit1
        self.target_max_false_positives = target_max_false_positives
        self.contact_pre_window_seconds = contact_pre_window_seconds
        self.post_shot_window_seconds = post_shot_window_seconds
        self.long_shot_distance_m = long_shot_distance_m

    def rank(
        self,
        candidate_features: pd.DataFrame,
        reference: pd.DataFrame,
        seed_candidates: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        scored = _quality_scored_features(
            candidate_features,
            contact_pre_window_seconds=self.contact_pre_window_seconds,
            post_shot_window_seconds=self.post_shot_window_seconds,
            long_shot_distance_m=self.long_shot_distance_m,
        )
        if scored.empty:
            return _empty_candidates()
        if reference.empty:
            return HighRecallCascadeShotRanker(
                temporal_nms_seconds=self.temporal_nms_seconds,
                max_candidates=self.max_candidates,
                max_candidates_per_half=self.max_candidates_per_half,
                contact_pre_window_seconds=self.contact_pre_window_seconds,
                post_shot_window_seconds=self.post_shot_window_seconds,
                long_shot_distance_m=self.long_shot_distance_m,
            ).rank(candidate_features)
        labels = _labels(scored, reference, tolerance_seconds=2.0)
        if len(set(labels.tolist())) >= 2:
            model = HistGradientBoostingClassifier(max_iter=60, random_state=0)
            weights = _hard_negative_weights(scored, labels)
            model.fit(_model_columns(scored), labels, sample_weight=weights)
            probabilities = model.predict_proba(_model_columns(scored))[:, 1]
            scored = scored.copy()
            scored["model_probability"] = probabilities
            scored["score"] = (
                0.55 * scored["model_probability"] + 0.45 * scored["pattern_score"]
            ).clip(0.0, 1.0)
        if seed_candidates is not None and not seed_candidates.empty:
            seed_scored = _score_seed_candidates(seed_candidates, scored)
            selection = SoftCompositeThresholdSelector(
                recall_floor_hit2=self.recall_floor_hit2,
                min_hit1=self.min_hit1,
                target_max_false_positives=self.target_max_false_positives,
            ).select(
                seed_scored,
                reference,
                lambda frame: _candidate_columns_from_seed_scores(
                    frame, "hard_negative_calibrated"
                ),
            )
            return selection.candidates
        selection = SoftCompositeThresholdSelector(
            recall_floor_hit2=self.recall_floor_hit2,
            min_hit1=self.min_hit1,
            target_max_false_positives=self.target_max_false_positives,
        ).select(
            scored,
            reference,
            lambda frame: _candidates_from_scored_adaptive(
                frame,
                self.temporal_nms_seconds,
                self.max_candidates,
                self.max_candidates_per_half,
                source="hard_negative_calibrated",
            ),
        )
        return selection.candidates


class WindowedTemporalShotRanker:
    def __init__(
        self,
        *,
        temporal_nms_seconds: float = 8.0,
        max_candidates: int = 80,
        max_candidates_per_half: int = 25,
        recall_floor_hit2: float = 0.78,
        min_hit1: float = 0.45,
        target_max_false_positives: int = 30,
        contact_pre_window_seconds: float = 0.5,
        post_shot_window_seconds: float = 1.0,
        long_shot_distance_m: float = 40.0,
    ) -> None:
        self.temporal_nms_seconds = temporal_nms_seconds
        self.max_candidates = max_candidates
        self.max_candidates_per_half = max_candidates_per_half
        self.recall_floor_hit2 = recall_floor_hit2
        self.min_hit1 = min_hit1
        self.target_max_false_positives = target_max_false_positives
        self.contact_pre_window_seconds = contact_pre_window_seconds
        self.post_shot_window_seconds = post_shot_window_seconds
        self.long_shot_distance_m = long_shot_distance_m

    def rank(
        self,
        candidate_features: pd.DataFrame,
        reference: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        scored = _quality_scored_features(
            candidate_features,
            contact_pre_window_seconds=self.contact_pre_window_seconds,
            post_shot_window_seconds=self.post_shot_window_seconds,
            long_shot_distance_m=self.long_shot_distance_m,
        )
        if scored.empty:
            return _empty_candidates()
        if reference is None or reference.empty:
            filtered = scored[
                (scored["score"] >= 0.34)
                & (scored["precision_gate"] | (scored["score"] >= 0.68))
            ]
            return _candidates_from_scored_adaptive(
                filtered,
                self.temporal_nms_seconds,
                self.max_candidates,
                self.max_candidates_per_half,
                source="windowed_temporal",
            )
        selection = SoftCompositeThresholdSelector(
            recall_floor_hit2=self.recall_floor_hit2,
            min_hit1=self.min_hit1,
            target_max_false_positives=self.target_max_false_positives,
        ).select(
            scored,
            reference,
            lambda frame: _candidates_from_scored_adaptive(
                frame,
                self.temporal_nms_seconds,
                self.max_candidates,
                self.max_candidates_per_half,
                source="windowed_temporal",
            ),
        )
        return selection.candidates


class DenseContactRefiner:
    def __init__(
        self, *, window_before_seconds: float = 1.5, window_after_seconds: float = 1.0
    ):
        self.window_before_seconds = window_before_seconds
        self.window_after_seconds = window_after_seconds

    def refine(
        self, candidates: pd.DataFrame, candidate_features: pd.DataFrame
    ) -> pd.DataFrame:
        if candidates.empty or candidate_features.empty:
            return _empty_candidates()
        rows: list[dict[str, Any]] = []
        for candidate in candidates.itertuples(index=False):
            window = candidate_features[
                candidate_features["global_seconds"].between(
                    float(candidate.global_seconds) - self.window_before_seconds,
                    float(candidate.global_seconds) + self.window_after_seconds,
                )
            ].copy()
            if window.empty:
                rows.append(candidate._asdict())
                continue
            accel_scale = max(float(window["ball_acceleration"].abs().max()), 1.0)
            window["refine_score"] = (
                window["ball_acceleration"].clip(lower=0.0) / accel_scale
            ) * (window["contact_score"] + 0.1)
            best = window.sort_values("refine_score", ascending=False).iloc[0]
            row = candidate._asdict()
            row.update(
                {
                    "global_frame_index": int(best["global_frame_index"]),
                    "global_seconds": float(best["global_seconds"]),
                    "part_index": int(best["part_index"]),
                    "part_frame_index": int(best["part_frame_index"]),
                    "score": float(max(row["score"], best["refine_score"])),
                    "confidence": float(max(row["confidence"], best["refine_score"])),
                    "source": "dense_local_refinement",
                    "nearest_player_distance": best["nearest_player_distance"],
                    "ball_speed": float(best["ball_speed"]),
                    "ball_direction_to_goal": float(best["ball_direction_to_goal"]),
                }
            )
            rows.append(row)
        return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)


def evaluate_shot_candidates(
    candidates: pd.DataFrame, reference: pd.DataFrame | None
) -> ShotRankingMetrics:
    if reference is None or reference.empty:
        return ShotRankingMetrics(0.0, 0.0, 0.0, 0.0, 0.0, float(len(candidates)))
    if candidates.empty:
        return ShotRankingMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    errors = []
    matched_candidates = set()
    for ref in reference.itertuples(index=False):
        nearest_idx = (
            (candidates["global_seconds"] - ref.reference_seconds).abs().idxmin()
        )
        nearest = candidates.loc[nearest_idx]
        errors.append(float(nearest["global_seconds"] - ref.reference_seconds))
        if abs(errors[-1]) <= 2.0:
            matched_candidates.add(str(nearest["shot_id"]))
    abs_errors = [abs(value) for value in errors]
    return ShotRankingMetrics(
        hit_05=sum(value <= 0.5 for value in abs_errors) / len(reference),
        hit_1=sum(value <= 1.0 for value in abs_errors) / len(reference),
        hit_2=sum(value <= 2.0 for value in abs_errors) / len(reference),
        precision_2=len(matched_candidates) / len(candidates)
        if len(candidates)
        else 0.0,
        temporal_mae_seconds=float(np.mean(abs_errors)) if abs_errors else 0.0,
        false_positives=float(max(len(candidates) - len(matched_candidates), 0)),
    )


def _rule_candidates(
    frame: pd.DataFrame,
    thresholds: dict[str, float],
    temporal_nms_seconds: float,
    max_candidates: int,
    *,
    source: str,
) -> pd.DataFrame:
    scored = frame.copy()
    speed_scale = max(float(scored["ball_speed"].quantile(0.95) or 1.0), 1.0)
    acc_scale = max(float(scored["ball_acceleration"].quantile(0.95) or 1.0), 1.0)
    scored["speed_score"] = (scored["ball_speed"] / speed_scale).clip(0.0, 1.0)
    scored["acc_score"] = (scored["ball_acceleration"] / acc_scale).clip(0.0, 1.0)
    scored["direction_score"] = scored["ball_direction_to_goal"].clip(-1.0, 1.0)
    mask = (
        (scored["speed_score"] >= thresholds["speed"])
        & (scored["acc_score"] >= thresholds["acc"])
        & (scored["contact_score"] >= thresholds["contact"])
        & (scored["direction_score"] >= thresholds["direction"])
    )
    scored = scored[mask].copy()
    scored["score"] = (
        0.35 * scored["speed_score"]
        + 0.25 * scored["acc_score"]
        + 0.20 * scored["contact_score"]
        + 0.15 * scored["direction_score"].clip(lower=0.0)
        + 0.05 * scored["distance_score"]
    )
    return _candidates_from_scored(
        scored, temporal_nms_seconds, max_candidates, source=source
    )


def _candidates_from_scored(
    scored: pd.DataFrame,
    temporal_nms_seconds: float,
    max_candidates: int,
    *,
    source: str,
) -> pd.DataFrame:
    if scored.empty:
        return _empty_candidates()
    rows = []
    for index, row in enumerate(
        scored.sort_values("score", ascending=False).to_dict("records"), start=1
    ):
        rows.append(
            {
                "shot_id": f"video-shot-{index:04d}",
                "global_frame_index": int(row["global_frame_index"]),
                "global_seconds": float(row["global_seconds"]),
                "part_index": int(row["part_index"]),
                "part_frame_index": int(row["part_frame_index"]),
                "score": float(row["score"]),
                "confidence": float(row["score"]),
                "source": source,
                "nearest_player_distance": row.get("nearest_player_distance"),
                "ball_speed": float(row["ball_speed"]),
                "ball_direction_to_goal": float(row["ball_direction_to_goal"]),
            }
        )
    ranked = TemporalShotRanker(
        temporal_nms_seconds=temporal_nms_seconds,
        max_candidates=max_candidates,
    ).rank(pd.DataFrame(rows))
    return ranked[CANDIDATE_COLUMNS] if not ranked.empty else _empty_candidates()


def _candidates_from_scored_adaptive(
    scored: pd.DataFrame,
    temporal_nms_seconds: float,
    max_candidates: int,
    max_candidates_per_half: int,
    *,
    source: str,
) -> pd.DataFrame:
    if scored.empty:
        return _empty_candidates()
    selected = AdaptiveShotNms(
        temporal_nms_seconds=temporal_nms_seconds,
        max_candidates=max_candidates,
        max_candidates_per_half=max_candidates_per_half,
    ).select(scored, score_column="score")
    if selected.empty:
        return _empty_candidates()
    rows = []
    for index, row in enumerate(selected.to_dict("records"), start=1):
        rows.append(
            {
                "shot_id": f"video-shot-{index:04d}",
                "global_frame_index": int(row["global_frame_index"]),
                "global_seconds": float(row["global_seconds"]),
                "part_index": int(row["part_index"]),
                "part_frame_index": int(row["part_frame_index"]),
                "score": float(row["score"]),
                "confidence": float(row["score"]),
                "source": source,
                "nearest_player_distance": row.get("nearest_player_distance"),
                "ball_speed": float(row["ball_speed"]),
                "ball_direction_to_goal": float(
                    row.get("direction_to_resolved_goal", row["ball_direction_to_goal"])
                ),
            }
        )
    return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)


def _score_seed_candidates(
    seed_candidates: pd.DataFrame, scored: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for seed in seed_candidates.itertuples(index=False):
        nearest_idx = (
            (scored["global_seconds"] - float(seed.global_seconds)).abs().idxmin()
        )
        quality = scored.loc[nearest_idx]
        row = seed._asdict()
        row["score"] = float(quality["score"])
        row["confidence"] = float(quality["score"])
        row["pattern_score"] = float(quality.get("pattern_score", quality["score"]))
        row["model_probability"] = float(
            quality.get("model_probability", quality["score"])
        )
        row["precision_gate"] = True
        row["direction_to_resolved_goal"] = float(
            quality.get("direction_to_resolved_goal", row["ball_direction_to_goal"])
        )
        row["post_goal_progress_m"] = float(quality.get("post_goal_progress_m", 0.0))
        rows.append(row)
    if not rows:
        return seed_candidates.head(0).copy()
    return pd.DataFrame(rows)


def _candidate_columns_from_seed_scores(
    frame: pd.DataFrame, source: str
) -> pd.DataFrame:
    if frame.empty:
        return _empty_candidates()
    selected = frame.sort_values("score", ascending=False).head(len(frame)).copy()
    selected = selected.sort_values("global_seconds").reset_index(drop=True)
    rows = []
    for index, row in enumerate(selected.to_dict("records"), start=1):
        rows.append(
            {
                "shot_id": f"video-shot-{index:04d}",
                "global_frame_index": int(row["global_frame_index"]),
                "global_seconds": float(row["global_seconds"]),
                "part_index": int(row["part_index"]),
                "part_frame_index": int(row["part_frame_index"]),
                "score": float(row["score"]),
                "confidence": float(row["score"]),
                "source": source,
                "nearest_player_distance": row.get("nearest_player_distance"),
                "ball_speed": float(row["ball_speed"]),
                "ball_direction_to_goal": float(
                    row.get("direction_to_resolved_goal", row["ball_direction_to_goal"])
                ),
            }
        )
    return pd.DataFrame(rows, columns=CANDIDATE_COLUMNS)


def _model_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "ball_speed",
        "ball_acceleration",
        "contact_score",
        "ball_direction_to_goal",
        "distance_to_goal",
        "distance_score",
        "direction_to_resolved_goal",
        "goal_progress_m",
        "distance_to_resolved_goal",
        "pre_contact_score",
        "post_recontact_score",
        "post_goal_progress_m",
        "post_direction_consistency",
        "post_speed_mean",
        "post_speed_max",
        "contact_acceleration_order_score",
        "shot_zone_score",
        "speed_max_0_5s",
        "speed_mean_0_5s",
        "acceleration_max_0_5s",
        "contact_max_0_5s",
        "direction_mean_0_5s",
        "speed_max_1_0s",
        "speed_mean_1_0s",
        "acceleration_max_1_0s",
        "contact_max_1_0s",
        "direction_mean_1_0s",
        "speed_max_2_0s",
        "speed_mean_2_0s",
        "acceleration_max_2_0s",
        "contact_max_2_0s",
        "direction_mean_2_0s",
        "pattern_score",
    ]
    model_frame = frame.copy()
    for column in columns:
        if column not in model_frame.columns:
            model_frame[column] = 0.0
    return model_frame[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _quality_scored_features(
    candidate_features: pd.DataFrame,
    *,
    contact_pre_window_seconds: float,
    post_shot_window_seconds: float,
    long_shot_distance_m: float,
) -> pd.DataFrame:
    window_features = ShotWindowFeatureExtractor(
        contact_pre_window_seconds=contact_pre_window_seconds,
        post_shot_window_seconds=post_shot_window_seconds,
        long_shot_distance_m=long_shot_distance_m,
    ).transform(candidate_features)
    scored = ShotPatternScorer(long_shot_distance_m=long_shot_distance_m).score(
        window_features
    )
    if scored.empty:
        return scored
    scored = scored.copy()
    scored["score"] = scored["pattern_score"]
    scored["confidence"] = scored["score"]
    return scored


def _hard_negative_weights(frame: pd.DataFrame, labels: pd.Series) -> np.ndarray:
    pattern = frame["pattern_score"].to_numpy(dtype=float)
    weights = np.full(len(frame), 0.55, dtype=float)
    weights[labels.to_numpy(dtype=bool)] = 2.0
    hard_negative = (~labels.to_numpy(dtype=bool)) & (
        pattern >= np.quantile(pattern, 0.75)
    )
    weights[hard_negative] = 1.5
    return weights


def _labels(
    frame: pd.DataFrame, reference: pd.DataFrame, tolerance_seconds: float
) -> pd.Series:
    labels = []
    reference_seconds = reference["reference_seconds"].to_numpy(dtype=float)
    for row in frame.itertuples(index=False):
        labels.append(
            bool(
                np.any(
                    np.abs(reference_seconds - row.global_seconds) <= tolerance_seconds
                )
            )
        )
    return pd.Series(labels, dtype=int)


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


def _track_index(tracks: pd.DataFrame) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    if tracks.empty:
        return {}
    valid = tracks[
        tracks["class_name"].isin(["player", "goalkeeper"])
        & tracks[["pitch_x", "pitch_y"]].notna().all(axis=1)
    ]
    return {
        int(frame_index): (
            group["pitch_x"].to_numpy(dtype=float),
            group["pitch_y"].to_numpy(dtype=float),
        )
        for frame_index, group in valid.groupby("global_frame_index", sort=False)
    }


def _nearest_player_distance(
    row: object, track_index: dict[int, tuple[np.ndarray, np.ndarray]]
) -> float | None:
    positions = track_index.get(int(row.global_frame_index))
    if positions is None:
        return None
    pitch_x, pitch_y = positions
    if len(pitch_x) == 0:
        return None
    distances = np.hypot(pitch_x - row.pitch_x, pitch_y - row.pitch_y)
    return float(distances.min())


def _direction_to_goal(row: pd.Series) -> float:
    if pd.isna(row["next_pitch_x"]) or pd.isna(row["next_pitch_y"]):
        return 0.0
    goal_x = 105.0 if row["pitch_x"] >= 52.5 else 0.0
    vx = float(row["next_pitch_x"] - row["pitch_x"])
    vy = float(row["next_pitch_y"] - row["pitch_y"])
    gx = float(goal_x - row["pitch_x"])
    gy = float(34.0 - row["pitch_y"])
    denom = math.hypot(vx, vy) * math.hypot(gx, gy)
    if denom <= 0.0:
        return 0.0
    return float((vx * gx + vy * gy) / denom)


def _empty_candidates() -> pd.DataFrame:
    return pd.DataFrame(columns=CANDIDATE_COLUMNS)
