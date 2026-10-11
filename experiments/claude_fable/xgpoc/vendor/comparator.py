"""The positional comparator: three rungs of the information ladder trained once outside the overlap (xg-vendor ticket 06).

Rungs (features: `comparator_features.RUNG_FEATURES`)
    dist_angle     logistic regression on cubic B-splines of distance and angle
    event          histogram gradient boosting on the event features
    freeze_frame   histogram gradient boosting on the event plus freeze-frame features (the "full comparator", K1)
    constant       the training pool's goal rate (no model)

Training pool: the non-penalty shots of periods 1-4 in the StatsBomb matches that no SoccerNet match is paired with
(`training_pool`). `training_pool` refuses evaluation matches that are not excluded, and `predict` refuses a training
match among the evaluation shots (`check_disjoint`). Penalties are no model input: an evaluation penalty gets the pool's
penalty conversion rate in every rung (`prediction_source` "penalty_rate").

Tuning: match-grouped K-fold cross-validation on the pool (`match_folds`, the same folds for every rung and setting),
pooled out-of-fold log-loss. The spline rung searches knots x C (`SPLINE_GRID`). The boosted rungs search leaves x
minimum leaf size x L2 at learning rate 0.03 (`BOOST_GRID`) with early stopping off; each fit is evaluated at every
boosting iteration up to the cap (`staged_predict_proba`), so the iteration count is chosen on the full curve, not on
a short list. The prototype searched leaves {15, 31} x iterations {150, 300, 600} at learning rate 0.03 and, by
sklearn's default for more than 10,000 rows, stopped early on an internal 10% split; its event model sat at the grid
edge. A first full search on the pool (learning rate {0.03, 0.1}, leaves {4, 8, 15, 31, 63}, L2 {0, 1, 10}, caps
1500 / 500; kept in `results/xgv_comparator_grid1`) chose learning rate 0.03, 4 leaves and L2 10 for both boosted rungs,
all three at an end of their range, and 12 knots, the top of the spline range. The frozen grids extend past those ends:
smaller trees, stronger L2, more knots, a longer iteration cap. Learning rate 0.1 lost by 1e-4 everywhere and is dropped;
0.03 is kept as the one rate (a lower rate trades for more iterations at several times the cost). Tuning reads pool
labels only, never an evaluation shot's.
Every estimator has `random_state` = `SEED`; nothing else is random (GroupKFold is deterministic). Every fit runs
on one thread, so the frozen models and the prediction hash do not depend on the number of parallel jobs.

The selected setting is refitted on the whole pool. Its pooled cross-validated log-loss is the comparator's pool
log-loss for kill rule K1 (`k1`): within `K1_MARGIN` of the vendor's on the same shots. That value is the minimum over
the grid, so it is optimistic by the selection; with 36,000 shots the bias is far below the margin.
"""

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
from threadpoolctl import threadpool_limits

from xgpoc.eval.calibration import EPS, logit, sigmoid
from xgpoc.eval.metrics import evaluate
from xgpoc.vendor.comparator_features import CATEGORICAL, RUNG_FEATURES, check_features

SEED = 0
N_FOLDS = 5
K1_MARGIN = 0.01
PERIODS = (1, 2, 3, 4)
PENALTY = "Penalty"
RUNGS = ("dist_angle", "event", "freeze_frame")
PREDICTION_RUNGS = ("constant", *RUNGS)
EVAL_POPULATIONS = {"recovered_goal": "recovered_goal", "secondary": "statsbomb_only"}   # sb_unpaired population -> evaluation table


@dataclass(frozen=True)
class SplineConfig:
    n_knots: int
    C: float


@dataclass(frozen=True)
class BoostConfig:
    learning_rate: float
    max_leaf_nodes: int
    min_samples_leaf: int
    l2_regularization: float
    max_iter: int   # the cap of the iteration curve; the chosen count is `RungFit.iterations`


SPLINE_GRID = tuple(SplineConfig(k, c) for k in (3, 4, 5, 6, 8, 10, 12, 16, 20, 25) for c in (0.01, 0.1, 1.0, 10.0, 100.0))
LEARNING_RATE, ITERATION_CAP = 0.03, 3000
BOOST_GRID = tuple(BoostConfig(LEARNING_RATE, leaves, leaf, l2, ITERATION_CAP) for leaves in (2, 3, 4, 6, 8, 15)
                   for leaf in (20, 50, 100, 200, 400) for l2 in (0.0, 1.0, 10.0, 30.0, 100.0))
RUNG_GRIDS = {"dist_angle": SPLINE_GRID, "event": BOOST_GRID, "freeze_frame": BOOST_GRID}


class OverlapError(ValueError):
    """An evaluation match would enter training."""


class K1Failed(RuntimeError):
    """Kill rule K1: the comparator is more than K1_MARGIN worse than the vendor on the pool. `report` is attached by the caller."""

    def __init__(self, message: str, report: dict | None = None):
        super().__init__(message)
        self.report = report


# ------------------------------------------------------------------ populations
def check_disjoint(train_matches, evaluation_matches) -> None:
    """Raise `OverlapError` when a match is in both collections."""
    both = set(np.asarray(train_matches).tolist()) & set(np.asarray(evaluation_matches).tolist())
    if both:
        raise OverlapError(f"{len(both)} evaluation matches are in the training data, e.g. {sorted(both)[0]}")


def training_pool(features: pd.DataFrame, shots: pd.DataFrame, excluded_matches, evaluation_matches) -> pd.DataFrame:
    """The training rows: `features` rows of non-penalty shots in periods 1-4 outside `excluded_matches`, with the label
    `goal` and the vendor's `statsbomb_xg` (for K1 and the pool metrics only) from `shots`. Raises `OverlapError` when
    an evaluation match is not among the excluded ones."""
    excluded = set(excluded_matches)
    if stray := set(evaluation_matches) - excluded:
        raise OverlapError(f"{len(stray)} evaluation matches are not excluded from training, e.g. {sorted(stray)[0]}")
    labels = shots.set_index("event_id")[["outcome", "statsbomb_xg"]]
    keep = ~features.match_id.isin(excluded) & features.shot_type.ne(PENALTY) & features.period.isin(PERIODS)
    pool = features[keep].join(labels, on="event_id")
    pool["goal"] = pool.pop("outcome").eq("Goal").astype(int)
    return pool.reset_index(drop=True)


def penalty_rate(shots: pd.DataFrame, excluded_matches) -> dict:
    """Conversion of the penalties in periods 1-4 outside `excluded_matches` (shoot-outs are period 5)."""
    pens = shots[shots.shot_type.eq(PENALTY) & shots.period.isin(PERIODS) & ~shots.match_id.isin(set(excluded_matches))]
    if pens.empty:
        raise ValueError("no penalty outside the excluded matches")
    goals = int(pens.outcome.eq("Goal").sum())
    return {"penalties": int(len(pens)), "goals": goals, "rate": goals / len(pens)}


def evaluation_shots(pairs: pd.DataFrame, sb_unpaired: pd.DataFrame) -> pd.DataFrame:
    """Every StatsBomb shot an evaluation population holds, by event id: the paired shots ("primary", with their
    SoccerNet shot id), the recovered goals ("recovered_goal") and the StatsBomb-only shots ("statsbomb_only", the
    pairing's secondary population); the populations of the evaluation table. Possible mispairs and the unpaired
    StatsBomb penalties are in no population, as in the pairing. Columns: statsbomb_event_id, statsbomb_match_id, population, soccernet_shot_id."""
    paired = pairs[pairs.paired]
    primary = pd.DataFrame({"statsbomb_event_id": paired.sb_event_id, "statsbomb_match_id": paired.sb_match_id.astype(int),
                            "population": "primary", "soccernet_shot_id": paired.shot_id})
    rest = sb_unpaired[sb_unpaired.population.isin(list(EVAL_POPULATIONS))]
    others = pd.DataFrame({"statsbomb_event_id": rest.event_id, "statsbomb_match_id": rest.match_id.astype(int),
                           "population": rest.population.map(EVAL_POPULATIONS), "soccernet_shot_id": None})
    out = pd.concat([primary, others], ignore_index=True)
    if out.statsbomb_event_id.isna().any() or out.statsbomb_event_id.duplicated().any():
        raise ValueError("evaluation shots need distinct StatsBomb event ids")
    return out


# ------------------------------------------------------------------ design and folds
def categories_of(pool: pd.DataFrame, features) -> dict[str, list[str]]:
    """The categories of each categorical feature as seen in the training pool (sorted); others map to missing."""
    return {c: sorted(pool[c].astype(str).unique()) for c in features if c in CATEGORICAL}


def design_matrix(table: pd.DataFrame, features, categories: dict[str, list[str]]) -> np.ndarray:
    """Float matrix of `features`; a categorical feature becomes its code in `categories` (NaN when unseen)."""
    columns = []
    for c in features:
        if c in categories:
            codes = pd.Categorical(table[c].astype(str), categories=categories[c]).codes.astype(float)
            columns.append(np.where(codes < 0, np.nan, codes))
        else:
            columns.append(table[c].to_numpy(float))
    return np.column_stack(columns)


def match_folds(match_ids, n_folds: int = N_FOLDS) -> np.ndarray:
    """Fold of every row: whole matches per fold (GroupKFold, deterministic for the same rows)."""
    match_ids = np.asarray(match_ids)
    folds = np.full(len(match_ids), -1)
    for k, (_, held) in enumerate(GroupKFold(n_folds).split(match_ids, groups=match_ids)):
        folds[held] = k
    return folds


def log_loss_sum(y: np.ndarray, p: np.ndarray) -> float:
    """Summed (not mean) log-loss, so fold sums add up to the pooled value."""
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def make_estimator(config, categorical_mask: np.ndarray, iterations: int | None = None):
    """Unfitted estimator of a setting: spline logistic pipeline or HGB with `iterations` (default: the cap)."""
    if isinstance(config, SplineConfig):
        return make_pipeline(SplineTransformer(n_knots=config.n_knots, degree=3), LogisticRegression(C=config.C, max_iter=5000))
    return HistGradientBoostingClassifier(learning_rate=config.learning_rate, max_iter=iterations or config.max_iter,
                                          max_leaf_nodes=config.max_leaf_nodes, min_samples_leaf=config.min_samples_leaf,
                                          l2_regularization=config.l2_regularization, categorical_features=categorical_mask,
                                          early_stopping=False, random_state=SEED)


def _fold_curve(config, X: np.ndarray, y: np.ndarray, folds: np.ndarray, k: int, mask: np.ndarray) -> np.ndarray:
    """Summed held-out log-loss of fold k: one value for a spline setting, one per boosting iteration otherwise."""
    with threadpool_limits(1):
        train, held = folds != k, folds == k
        model = make_estimator(config, mask).fit(X[train], y[train])
        if isinstance(config, SplineConfig):
            return np.array([log_loss_sum(y[held], model.predict_proba(X[held])[:, 1])])
        return np.array([log_loss_sum(y[held], p[:, 1]) for p in model.staged_predict_proba(X[held])])


# ------------------------------------------------------------------ fitted rung
@dataclass
class RungModel:
    """A frozen rung: its features, the pool's categories and the estimator refitted on the whole pool."""

    name: str
    features: list[str]
    categories: dict[str, list[str]]
    estimator: object

    def logit(self, table: pd.DataFrame) -> np.ndarray:
        check_features(self.features)
        return self.estimator.decision_function(design_matrix(table, self.features, self.categories))


@dataclass
class RungFit:
    """A tuned rung: the frozen model, the chosen setting and the evidence of the choice."""

    model: RungModel
    config: SplineConfig | BoostConfig
    iterations: int | None            # chosen boosting iterations (None for the spline rung)
    cv_log_loss: float                # pooled out-of-fold log-loss of the chosen setting
    oof_logit: np.ndarray             # out-of-fold logits of the chosen setting, pool order
    grid: pd.DataFrame                # every setting: its best iteration and log-loss
    grid_edges: list[str] = field(default_factory=list)   # chosen values at an end of their grid range

    def summary(self) -> dict:
        return {"features": self.model.features, "config": asdict(self.config), "iterations": self.iterations,
                "cv_log_loss": self.cv_log_loss, "settings_searched": len(self.grid), "grid_edges": self.grid_edges}


def grid_edges(config, iterations: int | None, grid) -> list[str]:
    """The chosen values that sit at an end of their range in `grid`, and the iteration count when it hit the cap."""
    edges = []
    for name, value in asdict(config).items():
        if name == "max_iter":
            continue
        values = sorted({getattr(g, name) for g in grid})
        if len(values) > 1 and value in (values[0], values[-1]):
            edges.append(f"{name}={value} ({'low' if value == values[0] else 'high'} end of {values})")
    if iterations is not None and iterations == config.max_iter:
        edges.append(f"iterations={iterations} (cap of the curve)")
    return edges


def fit_rung(name: str, pool: pd.DataFrame, folds: np.ndarray, *, grid=None, n_jobs: int = 1) -> RungFit:
    """Tune rung `name` by match-grouped CV on `pool` (columns: its features, `goal`), refit on all of `pool`.
    `grid` defaults to `RUNG_GRIDS[name]`; `n_jobs` parallel fits of one thread each."""
    features = RUNG_FEATURES[name]
    check_features(features)
    grid = RUNG_GRIDS[name] if grid is None else grid
    if not grid:
        raise ValueError("empty grid")
    categories = categories_of(pool, features)
    X, y = design_matrix(pool, features, categories), pool.goal.to_numpy(int)
    mask = np.array([c in categories for c in features])
    n_folds = int(folds.max()) + 1
    jobs = [(config, k) for config in grid for k in range(n_folds)]
    curves = Parallel(n_jobs=n_jobs)(delayed(_fold_curve)(config, X, y, folds, k, mask) for config, k in jobs)
    rows, best = [], None
    for i, config in enumerate(grid):
        curve = np.sum(curves[i * n_folds:(i + 1) * n_folds], axis=0) / len(y)
        j = int(np.argmin(curve))
        iterations = None if isinstance(config, SplineConfig) else j + 1
        rows.append(asdict(config) | {"iterations": iterations, "cv_log_loss": float(curve[j])})
        if best is None or curve[j] < best[2]:
            best = (config, iterations, float(curve[j]))
    config, iterations, cv_ll = best
    oof = np.zeros(len(y))
    with threadpool_limits(1):
        for k in range(n_folds):
            m = make_estimator(config, mask, iterations).fit(X[folds != k], y[folds != k])
            oof[folds == k] = m.decision_function(X[folds == k])
        estimator = make_estimator(config, mask, iterations).fit(X, y)
    return RungFit(RungModel(name, list(features), categories, estimator), config, iterations, cv_ll, oof, pd.DataFrame(rows),
                   grid_edges(config, iterations, grid))


# ------------------------------------------------------------------ metrics, K1, predictions
def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """n, goals, log-loss, Brier, AUC (NaN for one class) and mean prediction minus goal rate (`xgpoc.eval.metrics`)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = evaluate(y, p)
    return {"n": m["n"], "goals": int(y.sum()), "log_loss": m["log_loss"], "brier": m["brier"], "auc": m["auc"],
            "mean_p_minus_rate": float(np.clip(p, EPS, 1 - EPS).mean() - y.mean())}


def pool_cv_table(pool: pd.DataFrame, fits: dict[str, RungFit], folds: np.ndarray, matches: pd.DataFrame) -> pd.DataFrame:
    """Per pool shot: keys, competition and season, fold, label, vendor xG and every rung's out-of-fold probability
    (the constant rung is the goal rate of the other folds)."""
    out = pool[["event_id", "match_id", "goal", "statsbomb_xg"]].copy()
    out = out.merge(matches[["match_id", "competition", "season"]], on="match_id", how="left", validate="many_to_one")
    out["fold"] = folds
    rate = pd.Series(folds).map({k: pool.goal[folds != k].mean() for k in np.unique(folds)}).to_numpy()
    out["p_constant"] = rate
    for name, fit in fits.items():
        out[f"p_{name}"] = sigmoid(fit.oof_logit)
    return out


def pool_metrics(cv: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Metrics of the vendor and every rung on the pool (or per group of `by`), with each rung's log-loss minus the vendor's."""
    rungs = ["vendor", *[c[2:] for c in cv.columns if c.startswith("p_")]]
    rows = []
    for key, d in (cv.groupby(by) if by else [((), cv)]):
        key = key if isinstance(key, tuple) else (key,)
        vendor_ll = metrics(d.goal, d.statsbomb_xg)["log_loss"]
        for rung in rungs:
            m = metrics(d.goal, d.statsbomb_xg if rung == "vendor" else d[f"p_{rung}"])
            rows.append(dict(zip(by or [], key, strict=True)) | {"rung": rung} | m | {"log_loss_minus_vendor": m["log_loss"] - vendor_ll})
    return pd.DataFrame(rows)


def k1(comparator_log_loss: float, vendor_log_loss: float, margin: float = K1_MARGIN) -> dict:
    """Kill rule K1: passed when the comparator's pool log-loss exceeds the vendor's by at most `margin`."""
    gap = comparator_log_loss - vendor_log_loss
    return {"comparator_cv_log_loss": comparator_log_loss, "vendor_log_loss": vendor_log_loss, "gap": gap, "margin": margin,
            "passed": bool(gap <= margin)}


def check_k1(result: dict, report: dict | None = None) -> None:
    """Raise `K1Failed` (carrying `report`) when K1 did not pass."""
    if not result["passed"]:
        raise K1Failed(f"K1 fires: comparator pool CV log-loss {result['comparator_cv_log_loss']:.5f} is {result['gap']:+.5f} "
                       f"against the vendor's {result['vendor_log_loss']:.5f} (margin {result['margin']})", report)


def predict(models: dict[str, RungModel], features: pd.DataFrame, shots: pd.DataFrame, constant: float, penalty: float,
            train_matches) -> pd.DataFrame:
    """Label-free predictions for the evaluation `shots` (`evaluation_shots`): keys, `prediction_source` and per rung
    `p_<rung>` and `logit_<rung>`. Penalties get the `penalty` rate in every rung, every other shot the rung models
    (`constant` for the constant rung). Raises when an evaluation match is a training match or a shot has no features."""
    check_disjoint(train_matches, shots.statsbomb_match_id)
    f = features.set_index("event_id").reindex(shots.statsbomb_event_id)
    if f.match_id.isna().any():
        raise ValueError(f"{int(f.match_id.isna().sum())} evaluation shots have no features")
    if not np.array_equal(f.match_id.to_numpy(int), shots.statsbomb_match_id.to_numpy(int)):
        raise ValueError("an evaluation shot's match disagrees with its StatsBomb match")
    is_pen = f.shot_type.eq(PENALTY).to_numpy()
    out = shots.reset_index(drop=True).copy()
    out["prediction_source"] = np.where(is_pen, "penalty_rate", "model")
    f = f.reset_index()
    for rung in PREDICTION_RUNGS:
        z = np.full(len(out), float(logit(constant))) if rung == "constant" else models[rung].logit(f)
        z = np.where(is_pen, float(logit(penalty)), z)
        out[f"p_{rung}"], out[f"logit_{rung}"] = sigmoid(z), z
    return out
