"""The arms of xg-stack, their fold-wise design matrices and the two model classes (ticket 01; councils C5 and C6).

Candidate designs of G (council C6, `CANDIDATES`, listed before any simulation)
    b    parsimonious G: dist as a natural cubic spline with 3 df; angle (wrapped to [0, π]) linear; one shot-location
         missing indicator (shot_missing@0); one no-ball indicator (ball_none@0). Model `FIXED` (C5's penalty).
    c    C5's G columns (dist and angle splines with 4 df, the C5 linear columns). Model `INNER_CV`.
    bc   (b)'s columns with model `INNER_CV`.
    `spec(candidate, table)` builds every arm of a candidate:
        R    z_g_combo                                        reference
        G    R + the candidate's G                            confirmatory
        GS   G + the SoccerNet flags                          confirmatory
        S    R + the SoccerNet flags                          descriptive
        Q    G + the quality covariates (`QUALITY`: the C5 G linear columns (b) leaves out)    descriptive, bound only
        ALL  GS + the quality covariates + the remaining E3 columns at t0 and -400 ms, linear    descriptive
    and drops every linear input whose population variance is 0 (`constant_columns`, the frozen zero-variance rule;
    `check_no_constant` asserts it afterwards). GS is therefore G plus the S flags with non-zero variance.
Models (`Model`)
    The L2 logistic regression on the standardised design, fitted by Newton's method (`logistic_fit`): minimise the
    summed log-loss + 0.5 Σ λ_j w_j² with an unpenalised intercept and λ_j = 1 / C.
    FIXED      C = 1 on every column, z_g_combo included: C5's LogisticRegression(l2, C = 1.0, lbfgs).
    INNER_CV   C chosen inside each outer training fold by inner match-grouped CV over `C_GRID`: the inner folds are the
               other four host folds, each inner design is preprocessed on its own inner training rows, the summed
               inner held-out log-loss picks C (ties: the smaller C), and the model is refitted on the whole outer
               training fold. z_g_combo is unpenalised (λ = 0) in every arm, R included, so R is the unpenalised fit
               and every arm nests it. An arm whose only column is z needs no CV.
Design (`Preprocessor`, fitted on the training rows only)
    1. a missing input value becomes the training mean of its column (the indicators are inputs of their own);
    2. each spline column becomes its natural cubic spline basis (`natural_spline_basis`) with df columns and knots at
       the quantiles 0, 1/df, ..., 1 of the training rows' non-missing values (df = 4: 0/25/50/75/100%; df = 3:
       0/33/67/100%; R's ns(x, df));
    3. every design column is standardised with the training mean and standard deviation (scale 1 if constant).
    The design depends on the inputs and the folds only, never on y, so `fold_designs` builds it once per arm (with
    the inner designs for `INNER_CV`) and every outcome vector reuses it.
GBM sensitivity of G and GS: HistGradientBoostingClassifier with C5's settings, monotone increasing in z_g_combo, on
    the arm's raw inputs without the missing indicators (native NaN), never a claim.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from xgpoc.eval.calibration import sigmoid
from xgpoc.stack import inputs as si

Z = "z_g_combo"
C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
GBM = {"max_depth": 2, "learning_rate": 0.05, "max_iter": 300, "min_samples_leaf": 40, "l2_regularization": 1.0,
       "early_stopping": False, "random_state": 0}
NEWTON_ITERATIONS = 100
NEWTON_TOLERANCE = 1e-10           # largest change of a linear predictor at convergence
ZERO_VARIANCE = 0.0                # a column whose population variance is <= this is dropped

C5_SPLINE = (si.col("dist", 0), si.col("angle", 0))
C5_LINEAR = tuple(c for c in si.base_columns(0) if c not in C5_SPLINE)
B_SPLINE = (si.col("dist", 0),)
B_LINEAR = (si.col("angle", 0), si.col("shot_missing", 0), si.col("ball_none", 0))
QUALITY = tuple(c for c in C5_LINEAR if c not in B_LINEAR)
ALL_EXTRA = (*si.extra_columns(0), *si.base_columns(-400), *si.extra_columns(-400))
INDICATORS = frozenset(k for o in si.OFFSETS for k in si.missing_groups(o))
ARM_NAMES = ("R", "G", "GS", "S", "Q", "ALL")
CONFIRMATORY = ("G", "GS")
DESCRIPTIVE = ("S", "Q", "ALL")
GBM_ARMS = ("G", "GS")


@dataclass(frozen=True)
class Model:
    """The logistic model's penalty (module docstring): the C values to choose from and whether z is unpenalised."""

    name: str
    grid: tuple[float, ...]
    z_free: bool

    @property
    def inner_cv(self) -> bool:
        return len(self.grid) > 1

    def penalties(self, n_columns: int, c: float) -> np.ndarray:
        """λ per design column (z_g_combo first) at C = `c`."""
        lam = np.full(n_columns, 1.0 / c)
        if self.z_free:
            lam[0] = 0.0
        return lam


FIXED = Model("fixed", (1.0,), False)
INNER_CV = Model("inner_cv", C_GRID, True)


@dataclass(frozen=True)
class Candidate:
    """A candidate design of G (module docstring)."""

    name: str
    linear: tuple[str, ...]
    spline: tuple[str, ...]
    df: int
    model: Model
    description: str


CANDIDATES = {c.name: c for c in (
    Candidate("b", B_LINEAR, B_SPLINE, 3, FIXED, "parsimonious G: dist natural spline 3 df, angle linear (wrapped), one "
              "shot-location missing indicator, one no-ball indicator; C5's fixed penalty (C = 1 on every column)"),
    Candidate("c", C5_LINEAR, C5_SPLINE, 4, INNER_CV, "C5's G columns; C chosen by inner match-grouped CV from the grid, "
              "z unpenalised in every arm"),
    Candidate("bc", B_LINEAR, B_SPLINE, 3, INNER_CV, "(b)'s columns with (c)'s penalty choice"),
)}
SIMPLICITY = ("b", "c", "bc")      # the tie order of the selection: the simpler design first


@dataclass(frozen=True)
class Arm:
    """An arm's inputs: linear columns (z_g_combo first) and spline columns with `df` basis columns each."""

    name: str
    status: str
    linear: tuple[str, ...]
    spline: tuple[str, ...] = ()
    df: int = 4

    @property
    def inputs(self) -> list[str]:
        return [*self.linear, *self.spline]

    @property
    def columns(self) -> list[str]:
        """The design columns in matrix order: the linear columns, then each spline column's basis."""
        return [*self.linear, *[f"{c}_ns{k}" for c in self.spline for k in range(1, self.df + 1)]]

    @property
    def knot_quantiles(self) -> tuple[float, ...]:
        return tuple(float(q) for q in np.linspace(0.0, 1.0, self.df + 1))


@dataclass(frozen=True)
class Spec:
    """A candidate's arms after the zero-variance rule, the model, the dropped columns and the GBM inputs."""

    candidate: Candidate
    arms: dict[str, Arm]
    dropped: dict[str, list[str]]
    gbm_columns: dict[str, list[str]] = field(default_factory=dict)

    @property
    def model(self) -> Model:
        return self.candidate.model


def unique(columns) -> tuple[str, ...]:
    return tuple(dict.fromkeys(columns))


def constant_columns(table: pd.DataFrame, columns) -> list[str]:
    """The frozen zero-variance rule: the columns whose population variance over their non-missing values is <=
    `ZERO_VARIANCE` (or that have no non-missing value)."""
    values = table[list(columns)].astype(float)
    variance = values.var(ddof=0, skipna=True)
    return [c for c in columns if not np.isfinite(variance[c]) or variance[c] <= ZERO_VARIANCE]


def check_no_constant(table: pd.DataFrame, arm: Arm) -> None:
    """Raise unless every input of `arm` has non-zero population variance."""
    if constant := constant_columns(table, arm.inputs):
        raise ValueError(f"arm {arm.name}: constant input columns {constant}")


def spec(candidate: str | Candidate, table: pd.DataFrame) -> Spec:
    """Every arm of `candidate` (module docstring) with the constant linear inputs dropped."""
    cand = CANDIDATES[candidate] if isinstance(candidate, str) else candidate
    g = (Z, *cand.linear)
    linear = {"R": (Z,), "G": g, "GS": (*g, *si.S_FLAGS), "S": (Z, *si.S_FLAGS), "Q": (*g, *QUALITY),
              "ALL": (*g, *si.S_FLAGS, *QUALITY, *ALL_EXTRA)}
    status = {"R": "reference"} | dict.fromkeys(CONFIRMATORY, "confirmatory") | dict.fromkeys(DESCRIPTIVE, "descriptive")
    arms, dropped = {}, {}
    for name in ARM_NAMES:
        columns = [c for c in unique(linear[name]) if c not in cand.spline]
        dropped[name] = constant_columns(table, columns)
        spline = () if name in ("R", "S") else cand.spline
        arms[name] = Arm(name, status[name], tuple(c for c in columns if c not in dropped[name]), spline, cand.df)
        check_no_constant(table, arms[name])
    gbm = {name: [c for c in arms[name].inputs if c not in INDICATORS] for name in GBM_ARMS}
    return Spec(cand, arms, dropped, gbm)


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
        knots = {c: np.quantile(values[c].dropna().to_numpy(), arm.knot_quantiles) for c in arm.spline}
        expanded = cls._expand(arm, train, means, knots)
        centre, scale = expanded.mean(axis=0), expanded.std(axis=0)
        return cls(arm, means, knots, centre, np.where(scale > 1e-12, scale, 1.0))

    def __call__(self, table: pd.DataFrame) -> np.ndarray:
        return (self._expand(self.arm, table, self.means, self.knots) - self.centre) / self.scale


@dataclass(frozen=True)
class FoldDesign:
    """One fold's training and scored rows (positions in the population), their design matrices, and for `INNER_CV`
    the inner designs of the training rows (one per other host fold)."""

    fold: int
    train: np.ndarray
    test: np.ndarray
    x_train: np.ndarray
    x_test: np.ndarray
    inner: tuple["FoldDesign", ...] = ()


def _design(table: pd.DataFrame, arm: Arm, fold: int, train: np.ndarray, test: np.ndarray) -> FoldDesign:
    pre = Preprocessor.fit(arm, table.iloc[train])
    return FoldDesign(int(fold), train, test, pre(table.iloc[train]), pre(table.iloc[test]))


def fold_designs(table: pd.DataFrame, folds: np.ndarray, arm: Arm, inner: bool = False) -> list[FoldDesign]:
    """Per fold in sorted order: the `Preprocessor` fitted on the other folds' rows, applied to both; with `inner`, the
    same inside the training rows for each other fold."""
    out = []
    for k in np.unique(folds):
        train, test = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        design = _design(table, arm, k, train, test)
        if inner:
            nested = tuple(_design(table, arm, j, train[folds[train] != j], train[folds[train] == j])
                           for j in np.unique(folds[train]))
            design = FoldDesign(design.fold, train, test, design.x_train, design.x_test, nested)
        out.append(design)
    return out


def all_designs(table: pd.DataFrame, folds: np.ndarray, spec_: Spec, names=ARM_NAMES) -> dict[str, list[FoldDesign]]:
    return {name: fold_designs(table, folds, spec_.arms[name], spec_.model.inner_cv and len(spec_.arms[name].columns) > 1)
            for name in names}


# ------------------------------------------------------------------ models
def _objective(x1: np.ndarray, y: np.ndarray, beta: np.ndarray, lam: np.ndarray) -> float:
    eta = x1 @ beta
    return float(np.sum(np.logaddexp(0.0, eta) - y * eta) + 0.5 * np.sum(lam * beta**2))


def logistic_fit(x: np.ndarray, y: np.ndarray, penalties: np.ndarray, start: np.ndarray | None = None) -> np.ndarray:
    """(1 + p,) intercept and coefficients minimising the summed log-loss + 0.5 Σ λ_j w_j² (intercept unpenalised), by
    Newton's method with step halving; raises when it does not converge."""
    x1 = np.column_stack([np.ones(len(x)), x])
    y = np.asarray(y, float)
    lam = np.r_[0.0, np.asarray(penalties, float)]
    beta = np.zeros(x1.shape[1]) if start is None else np.array(start, float)
    current = _objective(x1, y, beta, lam)
    for _ in range(NEWTON_ITERATIONS):
        q = sigmoid(x1 @ beta)
        gradient = x1.T @ (q - y) + lam * beta
        hessian = (x1.T * (q * (1 - q))) @ x1 + np.diag(lam)
        step = np.linalg.solve(hessian, gradient)
        t = 1.0
        while (trial := _objective(x1, y, beta - t * step, lam)) > current + 1e-12 and t > 1e-8:
            t /= 2
        beta, current = beta - t * step, trial
        if np.abs(x1 @ (t * step)).max() < NEWTON_TOLERANCE:
            return beta
    raise RuntimeError("the logistic Newton fit did not converge")


def predict(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    return sigmoid(beta[0] + x @ beta[1:])


def mean_log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def chosen_c(design: FoldDesign, y: np.ndarray, model: Model) -> float:
    """The C of `model` for one outer fold: the grid value with the smallest summed inner held-out log-loss (ties: the
    smaller C); the single C of a fixed model, or of an arm with z alone."""
    if not model.inner_cv or not design.inner:
        return model.grid[-1]
    losses = []
    for c in model.grid:
        total = 0.0
        for d in design.inner:
            beta = logistic_fit(d.x_train, y[d.train], model.penalties(d.x_train.shape[1], c))
            total += mean_log_loss(y[d.test], predict(beta, d.x_test)) * len(d.test)
        losses.append(total)
    return float(model.grid[int(np.argmin(losses))])


def out_of_fold(designs: list[FoldDesign], y: np.ndarray, model: Model, chosen: list | None = None) -> np.ndarray:
    """(n,) out-of-fold goal probabilities, each fold scored by the model fitted on the other folds (with its C chosen
    inside them for `INNER_CV`); the chosen C per fold is appended to `chosen` when given."""
    y = np.asarray(y, float)
    p = np.full(len(y), np.nan)
    for d in designs:
        c = chosen_c(d, y, model)
        beta = logistic_fit(d.x_train, y[d.train], model.penalties(d.x_train.shape[1], c))
        p[d.test] = predict(beta, d.x_test)
        if chosen is not None:
            chosen.append(c)
    if np.isnan(p).any():
        raise ValueError("the folds do not cover every shot")
    return p


def gbm(columns: list[str]) -> HistGradientBoostingClassifier:
    """C5's GBM, monotone increasing in z_g_combo (which must be the first column)."""
    if columns[0] != Z:
        raise ValueError(f"the GBM's first column must be {Z}")
    return HistGradientBoostingClassifier(**GBM, monotonic_cst=[1] + [0] * (len(columns) - 1))


def out_of_fold_gbm(table: pd.DataFrame, folds: np.ndarray, y: np.ndarray, columns: list[str]) -> np.ndarray:
    """(n,) out-of-fold probabilities of the GBM on `columns` (native NaN, no preprocessing)."""
    x, y = table[columns].to_numpy(float), np.asarray(y).astype(int)
    p = np.full(len(y), np.nan)
    for k in np.unique(folds):
        train, test = folds != k, folds == k
        p[test] = gbm(columns).fit(x[train], y[train]).predict_proba(x[test])[:, 1]
    if np.isnan(p).any():
        raise ValueError("the folds do not cover every shot")
    return p


def reference_fit(table: pd.DataFrame, y: np.ndarray, model: Model) -> tuple[float, float]:
    """(a, b) of R under `model` fitted on every shot, on the raw z_g_combo scale: p_R = sigmoid(a + b z) (the
    controls' truth). For `INNER_CV` R is the unpenalised fit."""
    arm = Arm("R", "reference", (Z,))
    pre = Preprocessor.fit(arm, table)
    beta = logistic_fit(pre(table), y, model.penalties(1, model.grid[-1]))
    b = beta[1] / float(pre.scale[0])
    return float(beta[0] - b * float(pre.centre[0])), float(b)


def unpenalised_reference(table: pd.DataFrame, y: np.ndarray) -> tuple[float, float]:
    """(a, b) of the unpenalised logistic regression of y on [1, z] over every shot (the selection's common truth)."""
    beta = logistic_fit(table[[Z]].to_numpy(float), y, np.zeros(1))
    return float(beta[0]), float(beta[1])
