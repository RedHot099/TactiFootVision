"""The arms of xg-stack, their fold-wise design matrices and the two model classes (ticket 01, council C5).

Arms (`ARMS`)
    R    z_g_combo                                          reference
    G    R + the G inputs: dist and angle as natural cubic splines (4 df each), the G linear columns   confirmatory
    GS   G + the SoccerNet flags                            confirmatory
    S    R + the SoccerNet flags                            descriptive
    ALL  GS + the remaining E3 columns at t0 and -400 ms, linear    descriptive
Design (`Preprocessor`, fitted on the training folds only)
    1. a missing input value becomes the training-fold mean of that column (the indicators are inputs of their own);
    2. each spline column becomes its 4-column natural cubic spline basis (`natural_spline_basis`) with knots at the
       0, 25, 50, 75 and 100% quantiles of the training fold's non-missing values (two boundary and three interior
       knots: 4 df without the intercept, R's ns(x, df = 4));
    3. every design column is standardised with the training fold's mean and standard deviation (a constant column
       keeps scale 1).
    The design depends on the inputs and the folds only, never on y, so `fold_designs` builds it once per arm and
    every outcome vector reuses it.
Models
    `logistic`: LogisticRegression(l2, C = 1.0, lbfgs, max_iter = 5000) on the standardised design; z_g_combo enters
    linearly, no interactions. `gbm` (the sensitivity of G and GS): HistGradientBoostingClassifier with C5's settings,
    monotone increasing in z_g_combo, on the raw inputs with native NaN (`GBM_COLUMNS`, no spline, no imputation).
    Each fold's model is fitted on the other four folds and scores its own fold (`out_of_fold`, `out_of_fold_gbm`).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from xgpoc.stack import inputs as si

Z = "z_g_combo"
SPLINE_DF = 4
KNOT_QUANTILES = (0.0, 0.25, 0.5, 0.75, 1.0)
LOGISTIC = {"penalty": "l2", "C": 1.0, "solver": "lbfgs", "max_iter": 5000}
GBM = {"max_depth": 2, "learning_rate": 0.05, "max_iter": 300, "min_samples_leaf": 40, "l2_regularization": 1.0,
       "early_stopping": False, "random_state": 0}

G_SPLINE = [si.col("dist", 0), si.col("angle", 0)]
G_LINEAR = [c for c in si.base_columns(0) if c not in G_SPLINE]
ALL_EXTRA = si.extra_columns(0) + si.base_columns(-400) + si.extra_columns(-400)


@dataclass(frozen=True)
class Arm:
    """An arm's inputs: spline columns (natural cubic splines) and linear columns, z_g_combo first."""

    name: str
    status: str
    linear: tuple[str, ...]
    spline: tuple[str, ...] = ()

    @property
    def inputs(self) -> list[str]:
        return [*self.linear, *self.spline]

    @property
    def columns(self) -> list[str]:
        """The design columns in matrix order: the linear columns, then each spline column's basis."""
        return [*self.linear, *[f"{c}_ns{k}" for c in self.spline for k in range(1, SPLINE_DF + 1)]]


ARMS = {arm.name: arm for arm in (
    Arm("R", "reference", (Z,)),
    Arm("G", "confirmatory", (Z, *G_LINEAR), tuple(G_SPLINE)),
    Arm("GS", "confirmatory", (Z, *G_LINEAR, *si.S_FLAGS), tuple(G_SPLINE)),
    Arm("S", "descriptive", (Z, *si.S_FLAGS)),
    Arm("ALL", "descriptive", (Z, *G_LINEAR, *si.S_FLAGS, *ALL_EXTRA), tuple(G_SPLINE)),
)}
CONFIRMATORY = ("G", "GS")
DESCRIPTIVE = ("S", "ALL")
GBM_ARMS = ("G", "GS")
INDICATORS = frozenset(k for o in si.OFFSETS for k in si.missing_groups(o))
GBM_COLUMNS = {name: [Z, *G_SPLINE, *[c for c in G_LINEAR if c not in INDICATORS],
                      *(si.S_FLAGS if name == "GS" else [])] for name in GBM_ARMS}


# ------------------------------------------------------------------ natural cubic spline
def natural_spline_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """(n, K - 1) natural cubic spline basis with K knots, without the intercept (Hastie, Tibshirani & Friedman, eq.
    5.4-5.5): x, then d_k(x) - d_{K-1}(x) with d_k(x) = ((x - xi_k)+^3 - (x - xi_K)+^3) / (xi_K - xi_k). Linear beyond
    the boundary knots."""
    x, knots = np.asarray(x, float), np.asarray(knots, float)
    if len(knots) < 3 or np.any(np.diff(knots) <= 0):
        raise ValueError(f"knots must be at least 3 strictly increasing values, got {knots}")
    last = knots[-1]

    def d(k: float) -> np.ndarray:
        return (np.clip(x - k, 0, None) ** 3 - np.clip(x - last, 0, None) ** 3) / (last - k)
    penultimate = d(knots[-2])
    return np.column_stack([x, *[d(k) - penultimate for k in knots[:-2]]])


# ------------------------------------------------------------------ design
@dataclass(frozen=True)
class Preprocessor:
    """Imputation means, spline knots and standardisation of one arm, fitted on training rows (module docstring)."""

    arm: Arm
    means: dict[str, float]
    knots: dict[str, np.ndarray]
    centre: np.ndarray
    scale: np.ndarray

    @staticmethod
    def _expand(arm: Arm, inputs: pd.DataFrame, means: dict[str, float], knots: dict[str, np.ndarray]) -> np.ndarray:
        filled = inputs[arm.inputs].astype(float).fillna(means)
        blocks = [filled[list(arm.linear)].to_numpy()]
        blocks += [natural_spline_basis(filled[c].to_numpy(), knots[c]) for c in arm.spline]
        return np.hstack(blocks)

    @classmethod
    def fit(cls, arm: Arm, train: pd.DataFrame) -> "Preprocessor":
        values = train[arm.inputs].astype(float)
        if values.isna().all().any():
            raise ValueError(f"{arm.name}: a column is missing on every training row")
        means = {c: float(m) for c, m in values.mean(skipna=True).items()}
        knots = {c: np.quantile(values[c].dropna().to_numpy(), KNOT_QUANTILES) for c in arm.spline}
        expanded = cls._expand(arm, train, means, knots)
        centre, scale = expanded.mean(axis=0), expanded.std(axis=0)
        return cls(arm, means, knots, centre, np.where(scale > 1e-12, scale, 1.0))

    def __call__(self, table: pd.DataFrame) -> np.ndarray:
        return (self._expand(self.arm, table, self.means, self.knots) - self.centre) / self.scale


@dataclass(frozen=True)
class FoldDesign:
    """One fold's training and scored rows (positions in the population) and their design matrices."""

    fold: int
    train: np.ndarray
    test: np.ndarray
    x_train: np.ndarray
    x_test: np.ndarray


def fold_designs(table: pd.DataFrame, folds: np.ndarray, arm: Arm) -> list[FoldDesign]:
    """Per fold in sorted order: the `Preprocessor` fitted on the other folds' rows, applied to both."""
    out = []
    for k in np.unique(folds):
        train, test = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        pre = Preprocessor.fit(arm, table.iloc[train])
        out.append(FoldDesign(int(k), train, test, pre(table.iloc[train]), pre(table.iloc[test])))
    return out


# ------------------------------------------------------------------ models
def logistic() -> LogisticRegression:
    return LogisticRegression(**LOGISTIC)


def fitted_logistic(x: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """The logistic model fitted on (x, y); raises when lbfgs stops at max_iter."""
    model = logistic().fit(x, np.asarray(y).astype(int))
    if model.n_iter_[0] >= LOGISTIC["max_iter"]:
        raise RuntimeError("lbfgs did not converge")
    return model


def out_of_fold(designs: list[FoldDesign], y: np.ndarray) -> np.ndarray:
    """(n,) out-of-fold goal probabilities of the logistic model, each fold scored by the model of the other folds."""
    y = np.asarray(y)
    p = np.full(len(y), np.nan)
    for d in designs:
        p[d.test] = fitted_logistic(d.x_train, y[d.train]).predict_proba(d.x_test)[:, 1]
    if np.isnan(p).any():
        raise ValueError("the folds do not cover every shot")
    return p


def gbm(columns: list[str]) -> HistGradientBoostingClassifier:
    """C5's GBM, monotone increasing in z_g_combo (which must be the first column)."""
    if columns[0] != Z:
        raise ValueError(f"the GBM's first column must be {Z}")
    return HistGradientBoostingClassifier(**GBM, monotonic_cst=[1] + [0] * (len(columns) - 1))


def out_of_fold_gbm(table: pd.DataFrame, folds: np.ndarray, y: np.ndarray, arm: str) -> np.ndarray:
    """(n,) out-of-fold probabilities of the GBM sensitivity of `arm` (native NaN, no preprocessing)."""
    columns = GBM_COLUMNS[arm]
    x, y = table[columns].to_numpy(float), np.asarray(y).astype(int)
    p = np.full(len(y), np.nan)
    for k in np.unique(folds):
        train, test = folds != k, folds == k
        p[test] = gbm(columns).fit(x[train], y[train]).predict_proba(x[test])[:, 1]
    if np.isnan(p).any():
        raise ValueError("the folds do not cover every shot")
    return p


def reference_fit(table: pd.DataFrame, y: np.ndarray) -> tuple[float, float]:
    """(a, b) of R fitted on every shot, on the raw z_g_combo scale: p_R = sigmoid(a + b z) (the controls' truth)."""
    arm = ARMS["R"]
    pre = Preprocessor.fit(arm, table)
    model = fitted_logistic(pre(table), y)
    w, c = float(model.coef_[0, 0]), float(model.intercept_[0])
    b = w / float(pre.scale[0])
    return c - b * float(pre.centre[0]), b
