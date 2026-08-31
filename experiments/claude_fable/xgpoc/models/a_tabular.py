"""PoC A: engineered positional features -> logistic regression / gradient boosting.

The classical xG recipe. Reference point every other PoC has to beat.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from xgpoc.features.tabular import FEATURE_NAMES


def _xy(df: pd.DataFrame, tab: pd.DataFrame):
    m = df[["shot_id", "goal"]].merge(tab, on="shot_id", how="inner")
    return m[FEATURE_NAMES].to_numpy(np.float32), m["goal"].to_numpy(bool), m["shot_id"]


def fit_predict(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame, tab: pd.DataFrame, kind: str = "gbm"):
    xtr, ytr, _ = _xy(pd.concat([train, valid]), tab)  # sklearn models pick hyperparameters internally; valid folds in
    xte, yte, ids = _xy(test, tab)
    if kind == "logreg":
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    else:
        model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
                                               early_stopping=True, validation_fraction=0.15, random_state=0)
    model.fit(xtr, ytr)
    p = model.predict_proba(xte)[:, 1]
    return pd.DataFrame({"shot_id": ids.values, "goal": yte, "p": p}), model
