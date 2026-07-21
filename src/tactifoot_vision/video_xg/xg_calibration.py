import importlib.resources
import json
import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tactifoot_vision.enums import XgModelKind
from tactifoot_vision.video_xg.estimators import (
    VideoFreezeContextXgEstimator,
    VideoGeometryXgEstimator,
    VideoKinematicContextXgEstimator,
)
from tactifoot_vision.video_xg.results import VideoShotFeatures

FEATURE_COLUMNS = [
    "shot_x",
    "shot_y",
    "goalkeeper_distance",
    "nearest_player_distance",
    "defender_count_in_cone",
    "ball_speed",
    "ball_direction_to_goal",
    "shot_confidence",
]
NEURAL_FEATURE_COLUMNS = [
    "shot_x",
    "shot_y",
    "distance_to_goal",
    "shot_angle_degrees",
    "centrality",
    "goalkeeper_distance",
    "nearest_player_distance",
    "defender_count_in_cone",
    "ball_speed",
    "ball_direction_to_goal",
    "shot_confidence",
    "projection_confidence",
]
DATABALLPY_FOOT_XG_PARAMS: dict[str, Any] = {
    "standard_scaler": {
        "mean": {"dist": 18.610252422407328, "angle": 22.34941514934875},
        "var": {"dist": 52.86317816375148, "angle": 185.6593693612403},
    },
    "logreg": {
        "coefs": {"dist": -0.6108847292678112, "angle": 0.31690681366731754},
        "intercept": -2.431934835054635,
    },
}


class FormulaCoefficientCalibrator:
    def predict(self, features: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
        if features.empty:
            return _empty_predictions()
        base = _feature_matrix(features)
        if reference.empty:
            values = _base_predictions(features, XgModelKind.VIDEO_KINEMATIC_CONTEXT)
        else:
            train = features.merge(
                reference[["shot_id", "reference_xg"]], on="shot_id", how="inner"
            )
            if len(train) < 2:
                values = _base_predictions(
                    features, XgModelKind.VIDEO_KINEMATIC_CONTEXT
                )
            else:
                model = Ridge(alpha=0.1)
                model.fit(_feature_matrix(train), train["reference_xg"].clip(0.0, 1.0))
                values = model.predict(base)
        return _prediction_frame(features, values, "coefficient_fit")


class IsotonicXgCalibrator:
    def calibrate(
        self, predictions: pd.DataFrame, reference: pd.DataFrame
    ) -> pd.DataFrame:
        if predictions.empty:
            return _empty_predictions()
        rows = []
        for method, group in predictions.groupby("method"):
            joined = group.merge(
                reference[["shot_id", "reference_xg"]], on="shot_id", how="inner"
            )
            if len(joined) >= 3 and joined["xg"].nunique() >= 2:
                model = IsotonicRegression(
                    out_of_bounds="clip", y_min=0.001, y_max=0.999
                )
                model.fit(joined["xg"], joined["reference_xg"].clip(0.001, 0.999))
                calibrated = model.predict(group["xg"])
            else:
                scale = (
                    joined["reference_xg"].sum() / max(joined["xg"].sum(), 1e-9)
                    if len(joined)
                    else 1.0
                )
                calibrated = group["xg"] * scale
            for row, xg in zip(group.itertuples(index=False), calibrated, strict=True):
                rows.append(
                    {
                        "shot_id": row.shot_id,
                        "frame_index": int(row.frame_index),
                        "method": f"{method}_isotonic_platt",
                        "xg": float(np.clip(xg, 0.001, 0.999)),
                    }
                )
        return pd.DataFrame(rows)


class NeuralVideoXgCalibrator:
    def __init__(
        self,
        *,
        hidden_layer_sizes: tuple[int, ...] = (12, 6),
        random_state: int = 42,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.random_state = random_state

    def predict(self, features: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
        if features.empty:
            return _empty_predictions()
        train = _training_frame(features, reference)
        if len(train) < 4 or train["reference_xg"].nunique() < 2:
            values = _base_predictions(features, XgModelKind.VIDEO_KINEMATIC_CONTEXT)
            return _prediction_frame(features, values, "neural_video_xg")
        model = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation="relu",
                solver="lbfgs",
                alpha=0.001,
                max_iter=2000,
                random_state=self.random_state,
            ),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(
                _neural_feature_matrix(train),
                train["reference_xg"].clip(0.001, 0.999),
            )
        values = model.predict(_neural_feature_matrix(features))
        return _prediction_frame(features, values, "neural_video_xg")


class DataBallPySimpleXgBaseline:
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        if features.empty:
            return _empty_predictions()
        params = _load_databallpy_foot_xg_params()
        values = []
        for row in features.itertuples(index=False):
            distance = _row_distance_to_goal(row)
            angle = _row_shot_angle_degrees(row)
            values.append(_databallpy_logreg_xg(distance, angle, params))
        return _prediction_frame(features, np.asarray(values), "databallpy_simple_xg")


class QualityAwareXgEnsemble:
    def predict(
        self,
        predictions: pd.DataFrame,
        features: pd.DataFrame,
        *,
        reference_total_xg: float | None = None,
    ) -> pd.DataFrame:
        if predictions.empty:
            return _empty_predictions()
        pivot = predictions.pivot_table(
            index=["shot_id", "frame_index"],
            columns="method",
            values="xg",
            aggfunc="first",
        ).reset_index()
        frame = pivot.merge(features, on=["shot_id", "frame_index"], how="left")
        geometry = frame.get(XgModelKind.VIDEO_GEOMETRY.value, 0.0)
        freeze = frame.get(XgModelKind.VIDEO_FREEZE_CONTEXT.value, geometry)
        kinetic = frame.get(XgModelKind.VIDEO_KINEMATIC_CONTEXT.value, freeze)
        quality = _quality_weight(frame)
        values = (1.0 - quality) * geometry + quality * (0.35 * freeze + 0.65 * kinetic)
        if reference_total_xg is not None and values.sum() > 0.0:
            values = values * min(reference_total_xg / values.sum(), 3.0)
        rows = []
        for row, value in zip(frame.itertuples(index=False), values, strict=True):
            rows.append(
                {
                    "shot_id": row.shot_id,
                    "frame_index": int(row.frame_index),
                    "method": "quality_aware_ensemble",
                    "xg": float(np.clip(value, 0.001, 0.999)),
                }
            )
        return pd.DataFrame(rows)


def _base_predictions(features: pd.DataFrame, model_kind: XgModelKind) -> np.ndarray:
    estimator = {
        XgModelKind.VIDEO_GEOMETRY: VideoGeometryXgEstimator(),
        XgModelKind.VIDEO_FREEZE_CONTEXT: VideoFreezeContextXgEstimator(),
        XgModelKind.VIDEO_KINEMATIC_CONTEXT: VideoKinematicContextXgEstimator(),
    }[model_kind]
    values = []
    for row in features.itertuples(index=False):
        values.append(estimator.predict(_features_from_row(row)).xg)
    return np.asarray(values, dtype=float)


def _features_from_row(row: object) -> VideoShotFeatures:
    return VideoShotFeatures(
        shot_id=str(row.shot_id),
        frame_index=int(row.frame_index),
        shot_x=float(row.shot_x),
        shot_y=float(row.shot_y),
        goal_x=float(getattr(row, "goal_x", 105.0)),
        goal_y=float(getattr(row, "goal_y", 34.0)),
        nearest_player_distance=_optional_float(
            getattr(row, "nearest_player_distance", None)
        ),
        goalkeeper_distance=_optional_float(getattr(row, "goalkeeper_distance", None)),
        defender_count_in_cone=int(getattr(row, "defender_count_in_cone", 0) or 0),
        ball_speed=_optional_float(getattr(row, "ball_speed", None)),
        ball_direction_to_goal=_optional_float(
            getattr(row, "ball_direction_to_goal", None)
        ),
        shot_confidence=float(getattr(row, "shot_confidence", 1.0) or 1.0),
    )


def _feature_matrix(features: pd.DataFrame) -> pd.DataFrame:
    frame = features.copy()
    for column in FEATURE_COLUMNS:
        if column not in frame.columns:
            frame[column] = 0.0
    return frame[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _training_frame(features: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    if reference.empty:
        return pd.DataFrame(columns=[*features.columns, "reference_xg"])
    return features.merge(
        reference[["shot_id", "reference_xg"]],
        on="shot_id",
        how="inner",
    )


def _neural_feature_matrix(features: pd.DataFrame) -> pd.DataFrame:
    frame = features.copy()
    frame["distance_to_goal"] = [
        _row_distance_to_goal(row) for row in frame.itertuples(index=False)
    ]
    frame["shot_angle_degrees"] = [
        _row_shot_angle_degrees(row) for row in frame.itertuples(index=False)
    ]
    frame["centrality"] = [
        _row_centrality(row) for row in frame.itertuples(index=False)
    ]
    for column in NEURAL_FEATURE_COLUMNS:
        if column not in frame.columns:
            frame[column] = 0.0
    return frame[NEURAL_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _row_distance_to_goal(row: object) -> float:
    goal_x = float(getattr(row, "goal_x", 105.0))
    goal_y = float(getattr(row, "goal_y", 34.0))
    return math.hypot(goal_x - float(row.shot_x), goal_y - float(row.shot_y))


def _row_centrality(row: object) -> float:
    goal_y = float(getattr(row, "goal_y", 34.0))
    return max(0.0, 1.0 - abs(float(row.shot_y) - goal_y) / max(abs(goal_y), 1e-9))


def _row_shot_angle_degrees(row: object) -> float:
    shot_x = float(row.shot_x)
    shot_y = float(row.shot_y)
    goal_x = float(getattr(row, "goal_x", 105.0))
    goal_y = float(getattr(row, "goal_y", 34.0))
    post_top = np.array([goal_x, goal_y - 3.66])
    post_bottom = np.array([goal_x, goal_y + 3.66])
    shot = np.array([shot_x, shot_y])
    top_vector = post_top - shot
    bottom_vector = post_bottom - shot
    denominator = np.linalg.norm(top_vector) * np.linalg.norm(bottom_vector)
    if denominator <= 1e-9:
        return 180.0
    cosine = float(np.dot(top_vector, bottom_vector) / denominator)
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def _load_databallpy_foot_xg_params() -> dict[str, Any]:
    try:
        params_path = importlib.resources.files("databallpy").joinpath(
            "models/xg_params.json"
        )
        with params_path.open("r", encoding="utf-8") as file:
            params = json.load(file)
        foot = params.get("xg_by_foot")
        if isinstance(foot, dict):
            return foot
    except (FileNotFoundError, ModuleNotFoundError, json.JSONDecodeError):
        pass
    return DATABALLPY_FOOT_XG_PARAMS


def _databallpy_logreg_xg(
    distance: float, angle_degrees: float, params: dict[str, Any]
) -> float:
    scaler = params["standard_scaler"]
    logreg = params["logreg"]
    mean = scaler["mean"]
    var = scaler["var"]
    coefs = logreg["coefs"]
    dist_scaled = (distance - float(mean["dist"])) / math.sqrt(float(var["dist"]))
    angle_scaled = (angle_degrees - float(mean["angle"])) / math.sqrt(
        float(var["angle"])
    )
    logit = (
        float(logreg["intercept"])
        + float(coefs["dist"]) * dist_scaled
        + float(coefs["angle"]) * angle_scaled
    )
    return 1.0 / (1.0 + math.exp(-logit))


def _prediction_frame(
    features: pd.DataFrame, values: np.ndarray, method: str
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "shot_id": features["shot_id"].to_numpy(),
            "frame_index": features["frame_index"].astype(int).to_numpy(),
            "method": method,
            "xg": np.clip(values, 0.001, 0.999),
        }
    )


def _quality_weight(features: pd.DataFrame) -> pd.Series:
    source = features.get("feature_source", pd.Series("", index=features.index)).astype(
        str
    )
    source_weight = np.where(source.str.contains("missing"), 0.2, 0.8)
    confidence = features.get(
        "shot_confidence", pd.Series(0.5, index=features.index)
    ).fillna(0.5)
    projection = features.get(
        "projection_confidence", pd.Series(0.25, index=features.index)
    ).fillna(0.25)
    return pd.Series(
        np.clip(0.4 * source_weight + 0.3 * confidence + 0.3 * projection, 0.0, 1.0),
        index=features.index,
    )


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, int | float | np.integer | np.floating | str):
        return float(value)
    return float(str(value))


def _empty_predictions() -> pd.DataFrame:
    return pd.DataFrame(columns=["shot_id", "frame_index", "method", "xg"])
