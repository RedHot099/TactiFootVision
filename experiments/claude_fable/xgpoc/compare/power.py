"""Expected power of the frozen xg-compare design (ticket 01, council C4). Label-free.

Basis
    the primary shots of the shot table (`xgpoc.compare.table`). The main scale is the paper's frozen cross-fitted
    Platt recalibration: per shot, z' = a_k + b_k z with (a_k, b_k) the stored coefficients of the shot's match fold
    (`results/xgv_confirmatory/ladder/coefficients.json` and `folds.csv`, primary population). The simulation reads no label. The
    gap is g = z'_F - z'_V and the control is z'_D. The simulation holds the recalibration fixed across draws and halves;
    the analysis refits it inside each draw and half, so the simulated intervals are somewhat narrower (ticket 08 of
    xg-vendor measured about 8% on H2) and the power somewhat optimistic.
One simulated study (generator `default_rng([seed, study])`)
    1. permute the rows of the whole category block across the primary shots (one permutation for every category
       column, so sizes and the joint structure of the categories are kept, and their link to the gaps is broken);
    2. draw labels y ~ Bernoulli(sigmoid(z'_V)) for the log-loss items (the recalibrated vendor as the truth);
    3. plant the effects of a scenario and run the family (BH and BY over the 14 items) and the scan (discovery,
       frozen candidates, one-sided validation with Holm) with the analysis's statistics and the 2,000 shared match
       draws.
Planted effects (`protocol.Item.injection`)
    gap items       0.15 logit added to g inside the level (item 2: -0.15), which moves the adjusted effect by exactly 0.15
    log-loss items  0.005 added to the per-shot LL(F) - LL(V) inside the level; scenario "<id>:logit" instead shifts
                    z'_F by 0.15 logit inside the level and recomputes the log-loss difference
    item 8          the statistic is the whole population's, so the category permutation cannot null it. The study's
                    estimate is 0.10 (or 0 under the null) plus a bootstrap deviation; its draws are that estimate plus
                    the centred bootstrap deviations of the real statistic. Only the deviations are kept; the point
                    estimate is never returned
    item 9          the headers-vs-foot contrast on the permuted body labels, shifted by -0.10
    scan level      0.20 logit added to g inside one scan level (scenario "scan:<category>=<level>")
Scenarios: "null" (nothing planted), "all" (every family item planted), "item:<id>" (one item), "item:<id>:logit" for
items 10 and 16, and "scan:<category>=<level>" for every scan level that meets the floors in the study.
Floors are checked per study on the permuted categories (`protocol.FLOORS`). An item below its floor is not tested,
and BH runs over the items that are.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from xgpoc.compare import categories as cat
from xgpoc.compare import protocol as pr
from xgpoc.eval.calibration import sigmoid
from xgpoc.vendor import ladder as ld

N_SIMS = 1000
SIM_SEED = 0
ALPHA = 0.05
MAIN_SCALE = {"V": ("vendor", "z_vendor"), "P": ("freeze_frame", "z_positions"), "D": ("dist_angle", "z_dist_angle"),
              "F": ("frame", "z_frame")}   # method: (the paper's rung, the table's raw logit)


# ------------------------------------------------------------------ basis
@dataclass
class Basis:
    """Arrays over the primary shots. `weights` (1 + B, M): row 0 weighs every match once (the observed sample), the
    other rows are the shared draws' match counts; `match` is each shot's column in them."""

    z: dict[str, np.ndarray]       # main-scale logits: "F", "V", "D", "P"
    match: np.ndarray
    half: np.ndarray
    categories: pd.DataFrame       # cat_<name> columns of the pre-shot categories
    weights: np.ndarray

    @property
    def gap(self) -> np.ndarray:
        return self.z["F"] - self.z["V"]


def frozen_main_scale(rows: pd.DataFrame, coefficients: dict, folds: pd.DataFrame) -> dict[str, np.ndarray]:
    """Main-scale logits of V, P, D, F from the paper's stored per-fold Platt coefficients of the primary population
    ({rung: [[a, b] per fold in fold order]}) and its match folds (match, fold)."""
    fold = rows.match.map(folds.set_index("match").fold)
    if fold.isna().any():
        raise ValueError("a shot's match has no fold")
    k = fold.to_numpy(int)
    out = {}
    for name, (rung, column) in MAIN_SCALE.items():
        ab = np.asarray(coefficients[rung], float)
        out[name] = ab[k, 0] + ab[k, 1] * rows[column].to_numpy(float)
    return out


def basis_from_table(table: pd.DataFrame, coefficients: dict, folds: pd.DataFrame, n_boot: int = pr.N_BOOT,
                     seed: int = pr.BOOT_SEED) -> Basis:
    """The basis of the primary shots; the draws are those of the whole table (`ladder.bootstrap` over its matches)."""
    draws = ld.bootstrap(table.match.to_numpy(), n_boot, seed)
    rows = table[table.in_primary].reset_index(drop=True)
    boot = ld.draws_for(draws, rows.match.to_numpy())
    weights = np.vstack([np.ones(len(boot.keys)), boot.counts]).astype(float)
    return Basis(frozen_main_scale(rows, coefficients, folds), boot.match, rows.scan_half.to_numpy(),
                 rows[[c.column for c in cat.PRE_SHOT]].reset_index(drop=True), weights)


# ------------------------------------------------------------------ statistics
class AdjustedEffects:
    """The coefficient of each level indicator in the weighted OLS fit y ~ 1 + indicator + control over the level's
    defined shots, per weight row (match-count weights; row 0 is the observed sample). The design sums are fixed at
    construction; calling with an outcome y (n,) returns (point (L,), draws (B, L)). Linear in y."""

    def __init__(self, indicator: np.ndarray, defined: np.ndarray, control: np.ndarray, match: np.ndarray, weights: np.ndarray):
        n_matches = weights.shape[1]
        self.onehot = np.zeros((n_matches, len(match)))
        self.onehot[match, np.arange(len(match))] = 1.0
        self.weights = weights
        self.d = defined.astype(float)
        self.i = (indicator & defined).astype(float)
        self.dz = self.d * control[:, None]
        self.a11, self.a12, self.a13, self.a23, self.a33 = (
            weights @ (self.onehot @ q) for q in (self.d, self.i, self.dz, self.i * control[:, None], self.dz * control[:, None]))
        per_match_in = self.onehot @ self.i
        self.n_in = self.i.sum(0)
        self.n_out = self.d.sum(0) - self.n_in
        self.matches_in = (per_match_in > 0).sum(0)

    def __call__(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hy = self.onehot * y[None, :]
        r1, r2, r3 = (self.weights @ (hy @ q) for q in (self.d, self.i, self.dz))
        a11, a12, a13, a22, a23, a33 = self.a11, self.a12, self.a13, self.a12, self.a23, self.a33
        det = a11 * (a22 * a33 - a23**2) - a12 * (a12 * a33 - a23 * a13) + a13 * (a12 * a23 - a22 * a13)
        num = a11 * (r2 * a33 - a23 * r3) - r1 * (a12 * a33 - a23 * a13) + a13 * (a12 * r3 - r2 * a13)
        with np.errstate(invalid="ignore", divide="ignore"):
            b = np.where(det > 0, num / det, np.nan)
        return b[0], b[1:]


def p_values(draws: np.ndarray, sides: Sequence[str]) -> np.ndarray:
    """Bootstrap p-value per column of `draws` (B, L) as the protocol defines it: the share of draws on the wrong side
    of 0 (`ladder.superiority_p` for one side); two-sided doubles the smaller tail. NaN where a column has a NaN draw."""
    draws = np.atleast_2d(draws)
    below, above = (draws <= 0).mean(0), (draws >= 0).mean(0)
    out = np.array([below[j] if s == "greater" else above[j] if s == "less" else min(1.0, 2 * min(below[j], above[j]))
                    for j, s in enumerate(sides)])
    return np.where(np.isnan(draws).any(0), np.nan, out)


def bh(p: np.ndarray, q: float = pr.Q_BH, dependent: bool = False) -> np.ndarray:
    """Benjamini-Hochberg rejections at `q` over the finite p-values (Benjamini-Yekutieli when `dependent`)."""
    p = np.asarray(p, float)
    tested = np.flatnonzero(np.isfinite(p))
    reject = np.zeros(len(p), bool)
    m = len(tested)
    if not m:
        return reject
    level = q / np.sum(1.0 / np.arange(1, m + 1)) if dependent else q
    order = tested[np.argsort(p[tested], kind="stable")]
    passed = np.flatnonzero(p[order] <= level * np.arange(1, m + 1) / m)
    if len(passed):
        reject[order[: passed[-1] + 1]] = True
    return reject


def holm(p: np.ndarray, alpha: float = pr.HOLM_ALPHA) -> np.ndarray:
    """Holm's step-down rejections at `alpha` over the finite p-values."""
    p = np.asarray(p, float)
    tested = np.flatnonzero(np.isfinite(p))
    reject = np.zeros(len(p), bool)
    order = tested[np.argsort(p[tested], kind="stable")]
    for rank, j in enumerate(order):
        if p[j] > alpha / (len(order) - rank):
            break
        reject[j] = True
    return reject


def weighted_ranks(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Midranks (W, n) of `x` in each resampled sample (shot i present weights[w, i] times); ties share their midrank."""
    order = np.argsort(x, kind="stable")
    ordered = x[order]
    new = np.r_[True, ordered[1:] != ordered[:-1]]
    w = weights[:, order]
    group_weight = np.add.reduceat(w, np.flatnonzero(new), axis=1)
    rank = np.cumsum(group_weight, axis=1) - group_weight + (group_weight + 1) / 2
    out = np.empty_like(w)
    out[:, order] = rank[:, np.cumsum(new) - 1]
    return out


def weighted_pearson(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> np.ndarray:
    total = weights.sum(1, keepdims=True)
    ca = a - (weights * a).sum(1, keepdims=True) / total
    cb = b - (weights * b).sum(1, keepdims=True) / total
    return (weights * ca * cb).sum(1) / np.sqrt((weights * ca**2).sum(1) * (weights * cb**2).sum(1))


def spearman(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Spearman correlation of a and b in each resampled sample, from the weights (W, n)."""
    return weighted_pearson(weighted_ranks(a, weights), weighted_ranks(b, weights), weights)


def correlation_difference_deviations(basis: Basis) -> np.ndarray:
    """Item 8's bootstrap deviations: draws of Spearman(F, D) - Spearman(F, V) minus the observed value (module
    docstring). The observed value itself is not returned."""
    w = basis.weights[:, basis.match]
    stat = spearman(basis.z["F"], basis.z["D"], w) - spearman(basis.z["F"], basis.z["V"], w)
    return stat[1:] - stat[0]


def agreement_contrast(z: dict[str, np.ndarray], level: np.ndarray, reference: np.ndarray, match: np.ndarray,
                       weights: np.ndarray) -> tuple[float, np.ndarray]:
    """Item 9's (point, draws): [Spearman(F, V) - Spearman(P, V)] in `level` minus the same in `reference`."""
    def within(mask):
        w = weights[:, match[mask]]
        return spearman(z["F"][mask], z["V"][mask], w) - spearman(z["P"][mask], z["V"][mask], w)
    stat = within(level) - within(reference)
    return float(stat[0]), stat[1:]


# ------------------------------------------------------------------ one study
def indicators(categories: np.ndarray, columns: list[str], levels: Sequence[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    """(in, defined) boolean matrices (n, L) of (category, level) pairs over a category block (n, C) of level names."""
    index = {c: j for j, c in enumerate(columns)}
    inside = np.column_stack([categories[:, index[cat.PREFIX + c]] == level for c, level in levels])
    defined = np.column_stack([pd.notna(categories[:, index[cat.PREFIX + c]]) for c, _ in levels])
    return inside, defined


def meets(effects: AdjustedEffects) -> np.ndarray:
    """Per level: the category floor (shots in, shots out, matches in) holds."""
    return ((effects.n_in >= pr.FLOORS["category_in"]) & (effects.n_out >= pr.FLOORS["category_out"])
          & (effects.matches_in >= pr.FLOORS["category_matches_in"]))


def half_floor(effects: AdjustedEffects) -> np.ndarray:
    """Per level: the scan floor of one half (shots in and out)."""
    return (effects.n_in >= pr.FLOORS["scan_half_in"]) & (effects.n_out >= pr.FLOORS["scan_half_out"])


class Study:
    """The permuted categories and simulated labels of one study, with the fixed designs of the family and the scan."""

    def __init__(self, basis: Basis, deviations: np.ndarray, rng: np.random.Generator):
        n = len(basis.match)
        block = basis.categories.to_numpy()[rng.permutation(n)]
        columns = list(basis.categories.columns)
        self.basis = basis
        self.y = (rng.random(n) < sigmoid(basis.z["V"])).astype(float)
        self.e8 = float(deviations[rng.integers(len(deviations))])
        self.deviations = deviations
        self.effect_items = [item for item in pr.FAMILY if item.kind in ("gap", "log_loss")]
        inside, defined = indicators(block, columns, [(i.category, i.level) for i in self.effect_items])
        self.item_in = inside
        self.family = AdjustedEffects(inside, defined, basis.z["D"], basis.match, basis.weights)
        body = block[:, columns.index(cat.BY_NAME["body"].column)]
        self.head, self.foot = body == "Head", np.isin(body, cat.FOOT)
        self.levels = pr.scan_levels()
        scan_in, scan_defined = indicators(block, columns, self.levels)
        self.scan_in = scan_in
        whole = AdjustedEffects(scan_in, scan_defined, basis.z["D"], basis.match, basis.weights[:1])
        self.halves = {h: AdjustedEffects(scan_in, scan_defined & (basis.half == h)[:, None], basis.z["D"], basis.match, basis.weights)
                       for h in pr.HALVES}
        self.scannable = meets(whole) & half_floor(self.halves["discovery"]) & half_floor(self.halves["validation"])
        self.p_f = sigmoid(basis.z["F"])
        self.ll_v = ld.per_shot_log_loss(self.y, sigmoid(basis.z["V"]))
        self.item9 = agreement_contrast(basis.z, self.head, self.foot, basis.match, basis.weights)

    def run_family(self, planted: Sequence[str], logit_shift: str | None = None) -> pd.DataFrame:
        """One row per family item: point, p, tested, BH and BY rejection, interesting (module docstring)."""
        gap, p_f = self.basis.gap.copy(), self.p_f
        d_extra = np.zeros(len(gap))
        for j, item in enumerate(self.effect_items):
            if item.id in planted and item.kind == "gap":
                gap += item.injection * self.item_in[:, j]
            elif item.id in planted and item.kind == "log_loss":
                d_extra += item.injection * self.item_in[:, j]
            if logit_shift == item.id:
                p_f = sigmoid(self.basis.z["F"] + pr.THRESHOLDS["family_logit"] * self.item_in[:, j])
        d = ld.per_shot_log_loss(self.y, p_f) - self.ll_v + d_extra
        gap_point, gap_draws = self.family(gap)
        ll_point, ll_draws = self.family(d)
        rows = {}
        for j, item in enumerate(self.effect_items):
            point, draws = (gap_point[j], gap_draws[:, j]) if item.kind == "gap" else (ll_point[j], ll_draws[:, j])
            tested = bool(meets(self.family)[j])
            if item.kind == "log_loss":
                inside = self.item_in[:, j]
                tested &= bool(self.y[inside].sum() >= pr.FLOORS["label_goals"] and (1 - self.y[inside]).sum() >= pr.FLOORS["label_non_goals"])
            rows[item.id] = (point, draws, tested)
        shift8 = pr.BY_ID["8"].injection if "8" in planted else 0.0
        rows["8"] = (shift8 + self.e8, shift8 + self.e8 + self.deviations, True)
        point9, draws9 = self.item9
        shift9 = pr.BY_ID["9"].injection if "9" in planted else 0.0
        rows["9"] = (point9 + shift9, draws9 + shift9, bool(self.head.sum() >= pr.FLOORS["category_in"]))
        items = list(pr.FAMILY)
        p = np.array([p_values(rows[i.id][1][:, None], [i.side])[0] if rows[i.id][2] else np.nan for i in items])
        out = pd.DataFrame({"item": [i.id for i in items], "point": [rows[i.id][0] for i in items], "p": p,
                            "tested": [rows[i.id][2] for i in items], "reject_bh": bh(p), "reject_by": bh(p, dependent=True)})
        out["interesting"] = [interesting(i, pt) for i, pt in zip(items, out.point, strict=True)]
        return out

    def run_scan(self, planted_items: Sequence[str] = (), planted_level: int | None = None) -> np.ndarray:
        """Replicated scan levels (boolean per scan level) with the planted family items and/or one planted scan level."""
        gap = self.basis.gap.copy()
        for j, item in enumerate(self.effect_items):
            if item.id in planted_items and item.kind == "gap":
                gap += item.injection * self.item_in[:, j]
        if planted_level is not None:
            gap += pr.THRESHOLDS["scan_logit"] * self.scan_in[:, planted_level]
        d_point, d_draws = self.halves["discovery"](gap)
        p_discovery = p_values(d_draws, ["two-sided"] * len(self.levels))
        candidate = self.scannable & (np.abs(d_point) >= pr.THRESHOLDS["scan_logit"]) & (p_discovery < pr.SCAN_ALPHA)
        replicated = np.zeros(len(self.levels), bool)
        if candidate.any():
            _, v_draws = self.halves["validation"](gap)
            sides = ["greater" if x > 0 else "less" for x in d_point]
            p_validation = np.where(candidate, p_values(v_draws, sides), np.nan)
            replicated = holm(p_validation)
        return replicated


def interesting(item: pr.Item, point: float) -> bool:
    """The point estimate reaches the item's threshold, in the tested direction for a directional item."""
    if item.side == "greater":
        return point >= item.threshold
    if item.side == "less":
        return point <= -item.threshold
    return abs(point) >= item.threshold


# ------------------------------------------------------------------ the simulation
def scenarios() -> list[tuple[str, tuple[str, ...], str | None]]:
    """(name, planted items, logit-shift item) of the family scenarios."""
    out = [("null", (), None), ("all", pr.ALL_ITEMS, None)]
    out += [(f"item:{i.id}", (i.id,), None) for i in pr.FAMILY]
    out += [(f"item:{i.id}:logit", (), i.id) for i in pr.FAMILY if i.kind == "log_loss"]
    return out


def simulate(basis: Basis, studies: Sequence[int], seed: int = SIM_SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(family, scan, scan_levels) records of the given studies; see the module docstring."""
    deviations = correlation_difference_deviations(basis)
    levels = pr.scan_levels()
    family, scan, scan_levels = [], [], []
    for study in studies:
        s = Study(basis, deviations, np.random.default_rng([seed, study]))
        for name, planted, shift in scenarios():
            family.append(s.run_family(planted, shift).assign(study=study, scenario=name))
        for name, planted in (("null", ()), ("all", pr.ALL_ITEMS)):
            replicated = s.run_scan(planted)
            planted_levels = {(i.category, i.level) for i in pr.FAMILY if i.id in planted and i.kind == "gap"}
            on_planted = np.array([lv in planted_levels for lv in levels])
            scan.append({"study": study, "scenario": name, "levels_meeting_floors": int(s.scannable.sum()), "replicated": int(replicated.sum()),
                         "replicated_planted": int((replicated & on_planted).sum()), "replicated_other": int((replicated & ~on_planted).sum())})
        for j, (category, level) in enumerate(levels):
            if s.scannable[j]:
                replicated = s.run_scan(planted_level=j)
                scan_levels.append({"study": study, "category": category, "level": level, "replicated": bool(replicated[j]),
                                    "replicated_other": int(replicated.sum() - replicated[j])})
    return pd.concat(family, ignore_index=True), pd.DataFrame(scan), pd.DataFrame(scan_levels)


def summarise(family: pd.DataFrame, scan: pd.DataFrame, scan_levels: pd.DataFrame) -> dict:
    """The power record: per item, the rejection rates under the null, alone and with all items planted; the scan's
    expected replications; the per-level scan power."""
    null = family[family.scenario.eq("null")]
    items = {}
    for item in pr.FAMILY:
        alone, together = family[family.scenario.eq(f"item:{item.id}") & family.item.eq(item.id)], family[family.scenario.eq("all") & family.item.eq(item.id)]
        at_null = null[null.item.eq(item.id)]
        z_alpha = 1.959964 if item.side == "two-sided" else 1.644854
        sd = float(np.nanstd(at_null.point, ddof=1))
        record = {"label": item.label, "kind": item.kind, "side": item.side, "planted": item.injection,
                  "null_raw_rejection": float((at_null.p <= ALPHA).mean()), "null_bh_rejection": float(at_null.reject_bh.mean()),
                  "power_bh_alone": float(alone.reject_bh.mean()), "power_by_alone": float(alone.reject_by.mean()),
                  "power_raw_alone": float((alone.p <= ALPHA).mean()),
                  "power_bh_and_threshold_alone": float((alone.reject_bh & alone.interesting).mean()),
                  "power_bh_all": float(together.reject_bh.mean()), "power_by_all": float(together.reject_by.mean()),
                  "tested_share": float(alone.tested.mean()), "null_sd": sd, "mde80_unadjusted": (z_alpha + 0.841621) * sd}
        logit = family[family.scenario.eq(f"item:{item.id}:logit") & family.item.eq(item.id)]
        if len(logit):
            record["power_bh_logit_shift"] = float(logit.reject_bh.mean())
            record["mean_point_logit_shift"] = float(logit.point.mean())
        items[item.id] = record
    any_null = null.groupby("study").reject_bh.any()
    out = {"items": items, "family_null_any_bh_rejection": float(any_null.mean()),
           "family_null_any_by_rejection": float(null.groupby("study").reject_by.any().mean()),
           "n_studies": int(family.study.nunique())}
    out["scan"] = {name: {"mean_replicated": float(s.replicated.mean()), "mean_replicated_planted": float(s.replicated_planted.mean()),
                          "mean_replicated_other": float(s.replicated_other.mean()), "any_replicated": float(s.replicated.gt(0).mean()),
                          "mean_levels_meeting_floors": float(s.levels_meeting_floors.mean())}
                   for name, s in scan.groupby("scenario")}
    if len(scan_levels):
        n = out["n_studies"]
        per_level = scan_levels.groupby(["category", "level"], sort=False).agg(replicated=("replicated", "sum"), studies=("replicated", "size"))
        power = per_level.replicated / n
        out["scan_levels"] = {"planted": pr.THRESHOLDS["scan_logit"], "power": "share of all studies in which the level is replicated "
                              "(a study where the level misses a floor counts as not replicated)",
                              "median_power": float(power.median()), "min_power": float(power.min()), "max_power": float(power.max()),
                              "levels": [{"category": c, "level": lv, "power": float(power[(c, lv)]), "meets_floors_share": float(r.studies / n)}
                                         for (c, lv), r in per_level.iterrows()]}
    return out
