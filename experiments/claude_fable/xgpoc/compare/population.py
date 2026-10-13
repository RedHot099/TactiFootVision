"""The analysed populations of xg-compare (ticket 02): shots, labels, stacked draws and the main scale, plus the
per-shot outcomes, level indicators and floors that the tested (`xgpoc.compare.analysis`) and the descriptive
(`xgpoc.compare.describe`) parts share.

Population (`build_population`)
    rows of the shot table (`xgpoc.compare.table`) with the label file joined by key, the stacked match draws of the
    whole table (`stats.stacked_counts`) and, per method of `METHODS`, its raw logit and its main-scale
    recalibration refitted in every row of the stack on the population's own match folds (`stats.recalibrate`). A
    population is the primary or the combined population of the paper, or one scan half of the primary shots, so the
    recalibration is refitted separately inside each population, half and draw. The post-shot categories
    (`categories.CATEGORIES` with role post_shot) are derived from the labels here.
Scales (`Population.z`)
    "main" the main-scale logits (W, n); "raw" the stored logits (n,); "rank" the within-population percentile of the
    raw prediction in every row (W, n), in percentile points.
"""

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from xgpoc.compare import categories as cat
from xgpoc.compare import protocol as pr
from xgpoc.compare import stats as st
from xgpoc.eval.calibration import sigmoid
from xgpoc.vendor import ladder as ld
from xgpoc.vendor.draws import TableDraws

METHODS = {"V": "vendor", "P": "positions", "E": "events", "D": "dist_angle", "F": "frame", "F-400": "frame_m400", "A1": "a1"}
PAPER_RUNGS = {"V": "vendor", "P": "freeze_frame", "E": "event", "D": "dist_angle", "F": "frame"}
SCALES = ("main", "raw", "rank")
PRIMARY_ONLY_METHODS = ("A1",)   # protocol: A1 where the shot is in the xg-detector test predictions (primary shots only)


@dataclass
class Population:
    """One analysed population on the stacked draws (module docstring). `counts` (W, M) are the stacked match counts,
    `match` (n,) each shot's column in them and `weights` (W, n) the shot weights, counts[:, match]; `base` names the
    population whose categories apply (a scan half is "primary")."""

    name: str
    base: str
    rows: pd.DataFrame
    counts: np.ndarray
    match: np.ndarray
    folds: np.ndarray
    raw: dict[str, np.ndarray]
    coefficients: dict[str, np.ndarray]
    weights: np.ndarray = field(init=False, repr=False)
    _scaled: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.weights = self.counts[:, self.match]

    @property
    def y(self) -> np.ndarray:
        return self.rows.goal_sn.to_numpy(float)

    @property
    def methods(self) -> list[str]:
        return list(self.raw)

    def z(self, method: str, scale: str = "main") -> np.ndarray:
        """The method's logit on a scale (module docstring): (W, n) for main and rank, (n,) for raw."""
        if scale == "raw":
            return self.raw[method]
        if scale not in SCALES:
            raise ValueError(f"unknown scale {scale!r}")
        key = (method, scale)
        if key not in self._scaled:
            self._scaled[key] = (st.main_scale(self.raw[method], self.coefficients[method], self.folds) if scale == "main"
                                 else st.percentile_ranks(self.raw[method], self.weights))
        return self._scaled[key]

    def p(self, method: str, scale: str = "main") -> np.ndarray:
        """The method's probability on the main or raw scale."""
        if scale == "rank":
            raise ValueError("the rank scale has no probability")
        return sigmoid(self.z(method, scale))

    def categories_defined(self, category: str) -> bool:
        return self.base in cat.BY_NAME[category].populations

    def levels_of(self, category: str) -> np.ndarray:
        """(n,) level names of a category (None where undefined)."""
        return self.rows[cat.BY_NAME[category].column].to_numpy(object)


def post_shot_columns(labels: pd.DataFrame) -> pd.DataFrame:
    """The post-shot category columns from the label file (on target: SoccerNet, undefined outside the primary
    shots; StatsBomb outcome)."""
    return pd.DataFrame({cat.BY_NAME["on_target_sn"].column: cat.flag(labels.on_target_sn.astype(object)),
                         cat.BY_NAME["sb_outcome"].column: cat.text(labels.sb_outcome)}, index=labels.index)


def raw_logits(rows: pd.DataFrame) -> dict[str, np.ndarray]:
    """The raw logit of every method the rows have. A method missing on some rows is left out of the population when
    it is a primary-only method (`PRIMARY_ONLY_METHODS`: A1 in the combined population) and raises otherwise."""
    out = {}
    for method, column in METHODS.items():
        z = rows[f"z_{column}"].to_numpy(float)
        finite = np.isfinite(z)
        if finite.all():
            out[method] = z
        elif finite.any() and method not in PRIMARY_ONLY_METHODS:
            raise ValueError(f"method {method} has a logit on {int(finite.sum())} of {len(z)} rows; a population needs all or none")
    return out


def build_population(table: pd.DataFrame, labels: pd.DataFrame, mask: np.ndarray, draws: TableDraws, name: str,
                     base: str) -> Population:
    """The population of the rows in `mask` (module docstring); `draws` are the whole table's shared draws."""
    if not labels.key.equals(table.key):
        raise ValueError("the label file's keys must be the table's, in order")
    joined = pd.concat([table, labels.drop(columns="key"), post_shot_columns(labels)], axis=1)
    rows = joined[np.asarray(mask, bool)].reset_index(drop=True)
    y = rows.goal_sn.to_numpy(float)
    matches = rows.match.to_numpy()
    boot = ld.draws_for(draws, matches)
    counts = st.stacked_counts(boot)
    folds = ld.match_folds(matches, pr.N_FOLDS, pr.FOLD_SEED)
    weights = counts[:, boot.match]
    raw = raw_logits(rows)
    coefficients = {m: st.recalibrate(z, y, folds, weights) for m, z in raw.items()}
    return Population(name, base, rows, counts, boot.match, folds, raw, coefficients)


def paper_coefficients_match(population: Population, paper: dict, tolerance: float = 2e-6) -> dict[str, float]:
    """Largest absolute difference per rung between the population's observed recalibration (row 0) and the
    paper's stored per-fold coefficients ({rung: [[a, b] per fold]}, rounded to 6 decimals); raises above
    `tolerance`."""
    out = {}
    for method, rung in PAPER_RUNGS.items():
        out[method] = float(np.abs(population.coefficients[method][0] - np.asarray(paper[rung], float)).max())
        if out[method] > tolerance:
            raise ValueError(f"{method}: the main scale departs from the paper's recalibration by {out[method]:.2e}")
    return out


# ------------------------------------------------------------------ per-shot outcomes
def gap(pop: Population, a: str, b: str, scale: str = "main") -> np.ndarray:
    """z_a - z_b on a scale."""
    return pop.z(a, scale) - pop.z(b, scale)


def log_loss_difference(pop: Population, a: str, b: str, scale: str = "main") -> np.ndarray:
    """Per-shot LL(a) - LL(b) of the probabilities on the main or raw scale."""
    y = pop.y
    return ld.per_shot_log_loss(y, pop.p(a, scale)) - ld.per_shot_log_loss(y, pop.p(b, scale))


def mean_vendor_probability(pop: Population, inside: np.ndarray) -> float:
    """The mean main-scale vendor probability of the shots in `inside`, observed sample (the base of `stats.xg_points`)."""
    return float(sigmoid(pop.z("V")[0][inside]).mean())


def scales_for(label_free: bool) -> tuple[str, ...]:
    """The scales a run reports: the main scale always, the raw scale only when label-free parts run."""
    return ("main", "raw") if label_free else ("main",)


# ------------------------------------------------------------------ levels and floors
def membership(pop: Population, category: str, level: str, reference: tuple[str, ...] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(inside, defined) of a level: defined is every shot where the category is defined, or only the shots in
    `level` or `reference` when a reference set is given."""
    values = pop.levels_of(category)
    defined = pd.notna(values) if reference is None else np.isin(values, (level, *reference))
    return (values == level) & defined, defined.astype(bool)


def level_matrices(pop: Population, levels: list[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    """(inside, defined) (n, L) of (category, level) pairs."""
    pairs = [membership(pop, c, level) for c, level in levels]
    return np.column_stack([p[0] for p in pairs]), np.column_stack([p[1] for p in pairs])


def level_floors(pop: Population, inside: np.ndarray, defined: np.ndarray) -> pd.DataFrame:
    """Per level column of (n, L) masks: shots in and out, matches in, goals and non-goals in, and whether the
    category floor and the label floor hold (`protocol.FLOORS`)."""
    y = pop.y
    n_in, n_out = inside.sum(0), (defined & ~inside).sum(0)
    matches_in = np.array([len(np.unique(pop.match[inside[:, j]])) for j in range(inside.shape[1])])
    goals = (y @ inside).astype(int)
    return pd.DataFrame({"n_in": n_in, "n_out": n_out, "matches_in": matches_in, "goals_in": goals, "non_goals_in": n_in - goals,
                         "meets_category_floor": (n_in >= pr.FLOORS["category_in"]) & (n_out >= pr.FLOORS["category_out"])
                         & (matches_in >= pr.FLOORS["category_matches_in"]),
                         "meets_label_floor": (goals >= pr.FLOORS["label_goals"]) & (n_in - goals >= pr.FLOORS["label_non_goals"])})


def floors(pop: Population, inside: np.ndarray, defined: np.ndarray) -> dict:
    """`level_floors` of one level as a dict of Python scalars."""
    row = level_floors(pop, inside[:, None], defined[:, None]).iloc[0]
    return {k: (bool(v) if isinstance(v, np.bool_) else int(v)) for k, v in row.items()}


# ------------------------------------------------------------------ records
def flat(prefix: str, summary: dict, keep_nan_draws: bool = True) -> dict:
    """A summary's keys prefixed (`prefix_point`, ...); `nan_draws` dropped unless `keep_nan_draws`."""
    return {f"{prefix}_{k}": v for k, v in summary.items() if keep_nan_draws or k != "nan_draws"}


def json_text(obj) -> str:
    """Sorted, indented JSON with numpy scalars as Python ones and non-finite floats as null."""
    def plain(value):
        if isinstance(value, dict):
            return {str(k): plain(v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [plain(v) for v in value]
        if isinstance(value, np.generic):
            value = value.item()
        return None if isinstance(value, float) and not np.isfinite(value) else value
    return json.dumps(plain(obj), indent=2, sort_keys=True, allow_nan=False) + "\n"
