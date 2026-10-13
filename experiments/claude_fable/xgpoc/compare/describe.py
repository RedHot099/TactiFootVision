"""The descriptive part of the xg-compare analysis (ticket 02, council C4): relative valuation of every level,
the multivariable fit, agreement, who is right, the largest disagreements and the raw levels. Nothing here is
tested; intervals are 95% match-bootstrap percentile intervals on the stacked draws (`xgpoc.compare.stats`), and a
level below a floor is kept with its n and flagged.

Levels (`level_table`)
    every level of every category defined in the population: shots and matches inside, the floors, and per method
    pair of `PAIRS` the unadjusted contrast and the adjusted effect (control z_D) on the main scale; for F - V also
    the raw and rank scales and xG points. Post-shot levels carry only the F - V and (F-400) - V pairs and the flag
    `post_contact` (C4: F and F-400 side by side, gaps only).
Multivariable (`multivariable`)
    g ~ 1 + z_D + the indicators of the family's gap items, over the shots where all of them are defined: the spec's
    Analysis 2 regression of the gap on the category indicators, descriptive.
Agreement (`agreement`)
    per level (and overall) and pair of methods: Spearman of the raw probabilities and Pearson of the main-scale
    logits; jointly bootstrapped differences rho(F, V) - rho(P, V) and rho(F, D) - rho(F, V) (Spearman).
Who is right (`who_is_right`)
    per level (and overall) and method on the main scale: O, E, O - E, O / E, log-loss, the log-loss difference
    against V and the AUC. The AUC needs both goal classes at the label floor; the other metrics are flagged below
    it. Post-shot levels get only the AUC (no calibration inside outcome-defined categories).
Largest disagreements (`disagreements`)
    the 5% of primary shots with the largest and the 5% with the smallest observed F - V gap on the main scale, listed
    by shot ID with their categories, and per tail the goal rate against each method's mean probability and the
    category profile (share of each level in the tail against the population).
"""

import numpy as np
import pandas as pd

from xgpoc.compare import categories as cat
from xgpoc.compare import protocol as pr
from xgpoc.compare import stats as st
from xgpoc.compare.population import (
    Population,
    flat,
    gap,
    level_floors,
    level_matrices,
    log_loss_difference,
    mean_vendor_probability,
)

PAIRS = (("F", "V"), ("F-400", "V"), ("A1", "V"), ("P", "V"), ("E", "V"), ("D", "V"), ("F", "D"), ("F", "P"))
POST_SHOT_PAIRS = (("F", "V"), ("F-400", "V"))
AGREEMENT_DIFFERENCES = ((("F", "V"), ("P", "V")), (("F", "D"), ("F", "V")))
MIN_AGREEMENT_SHOTS = 20       # smaller levels have no correlation row
TAIL_SHARE = 0.05
UNDEFINED = "undefined"   # the profile's name for a shot where the category is undefined


def category_levels(pop: Population) -> list[tuple[cat.Category, str]]:
    """(category, level) of every category defined in the population, in dictionary order."""
    return [(c, level) for c in cat.CATEGORIES if pop.categories_defined(c.name) for level in c.levels]


def _matrices(pop: Population, levels: list[tuple[cat.Category, str]]) -> tuple[np.ndarray, np.ndarray]:
    return level_matrices(pop, [(c.name, level) for c, level in levels])


def _summaries(stacked: np.ndarray, prefix: str) -> list[dict]:
    """Per column of a stacked (W, L) array: {prefix_point, prefix_lo, prefix_hi}."""
    out = []
    for j in range(stacked.shape[1]):
        s = st.summarise(stacked[:, j])
        out.append({f"{prefix}_point": s["point"], f"{prefix}_lo": s["lo"], f"{prefix}_hi": s["hi"]})
    return out


def level_table(pop: Population, label_free: bool = True) -> pd.DataFrame:
    """Relative valuation of every level (module docstring); one row per level and pair."""
    levels = category_levels(pop)
    inside, defined = _matrices(pop, levels)
    base = pd.DataFrame({"population": pop.name, "category": [c.name for c, _ in levels], "level": [lv for _, lv in levels],
                         "role": [c.role for c, _ in levels], "group": [c.group for c, _ in levels]})
    base = pd.concat([base, level_floors(pop, inside, defined)], axis=1)
    base["post_contact"] = base.role.eq("post_shot")
    p_bar = np.array([mean_vendor_probability(pop, inside[:, j]) if inside[:, j].any() else np.nan for j in range(len(levels))])
    parts = []
    for a, b in PAIRS:
        if a not in pop.methods:
            continue
        keep = np.array([(a, b) in POST_SHOT_PAIRS or c.role != "post_shot" for c, _ in levels])
        scales = ("main", "raw", "rank") if (a, b) == ("F", "V") and label_free else ("main",)
        frame = base[keep].assign(pair=f"{a}-{b}").reset_index(drop=True)
        for scale in scales:
            adjusted, unadjusted = st.level_effects(gap(pop, a, b, scale), pop.z("D", scale), inside[:, keep], defined[:, keep], pop.weights)
            frame = pd.concat([frame, pd.DataFrame(_summaries(adjusted, f"{scale}_adjusted")),
                               pd.DataFrame(_summaries(unadjusted, f"{scale}_unadjusted"))], axis=1)
        if (a, b) == ("F", "V"):
            frame["mean_vendor_p_in"] = p_bar[keep]
            frame["xg_points"] = [st.xg_points(e, p) if np.isfinite(p) else np.nan for e, p in zip(frame.main_adjusted_point, p_bar[keep], strict=True)]
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def multivariable(pop: Population) -> pd.DataFrame:
    """The multivariable fit of the F - V gap (module docstring); one row per coefficient."""
    items = [i for i in pr.FAMILY if i.kind == "gap" and pop.categories_defined(i.category)]
    values = {i.id: pop.levels_of(i.category) for i in items}
    keep = np.all([pd.notna(v) for v in values.values()], axis=0)
    w = pop.weights[:, keep]
    zd = pop.z("D")[:, keep]
    indicators = np.column_stack([values[i.id][keep] == i.level for i in items]).astype(float)
    x = np.concatenate([np.ones(zd.shape + (1,)), zd[:, :, None], np.broadcast_to(indicators, (len(w),) + indicators.shape)], axis=2)
    beta = st.ols(gap(pop, "F", "V")[:, keep], x, w)
    names = ["intercept", "z_D", *[f"item {i.id}: {i.category}={i.level}" for i in items]]
    rows = [{"population": pop.name, "term": name, "n": int(keep.sum()), **st.summarise(beta[:, j], "two-sided")} for j, name in enumerate(names)]
    return pd.DataFrame(rows)


def _correlations(pop: Population, mask: np.ndarray, methods: list[str]) -> dict[tuple[str, str, str], np.ndarray]:
    """Stacked (W,) correlations of every method pair inside `mask`: Spearman of the raw probabilities (the
    within-mask ranks, `power.weighted_ranks` through `stats.spearman`) and Pearson of the main-scale logits."""
    w = pop.weights[:, mask]
    ranks = {m: st.percentile_ranks(pop.raw[m][mask], w) for m in methods}
    main = {m: pop.z(m)[:, mask] for m in methods}
    out = {}
    for i, a in enumerate(methods):
        for b in methods[i + 1:]:
            out[("spearman", a, b)] = st.pearson(ranks[a], ranks[b], w)
            out[("pearson", a, b)] = st.pearson(main[a], main[b], w)
    return out


def _pair(corr: dict, kind: str, a: str, b: str) -> np.ndarray:
    return corr[(kind, a, b)] if (kind, a, b) in corr else corr[(kind, b, a)]


def agreement(pop: Population) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(correlations, differences): per level (and "overall") and pair, Spearman and Pearson with intervals; per
    level the jointly bootstrapped differences of `AGREEMENT_DIFFERENCES` (Spearman). Levels with fewer than
    `MIN_AGREEMENT_SHOTS` shots have no row; the category floor is flagged."""
    methods = pop.methods
    levels = category_levels(pop)
    inside, defined = _matrices(pop, levels)
    floors = level_floors(pop, inside, defined)
    groups = [("overall", "all", np.ones(len(pop.y), bool), True)]
    groups += [(c.name, level, inside[:, j], bool(floors.meets_category_floor[j])) for j, (c, level) in enumerate(levels)
               if inside[:, j].sum() >= MIN_AGREEMENT_SHOTS]
    rows, differences = [], []
    for category, level, mask, meets in groups:
        corr = _correlations(pop, mask, methods)
        common = {"population": pop.name, "category": category, "level": level, "n": int(mask.sum()), "meets_category_floor": meets}
        for (kind, a, b), stacked in corr.items():
            rows.append(common | {"kind": kind, "a": a, "b": b, **st.summarise(stacked)})
        for (a1, b1), (a2, b2) in AGREEMENT_DIFFERENCES:
            d = _pair(corr, "spearman", a1, b1) - _pair(corr, "spearman", a2, b2)
            differences.append(common | {"difference": f"rho({a1},{b1}) - rho({a2},{b2})", **st.summarise(d, "two-sided")})
    return pd.DataFrame(rows), pd.DataFrame(differences)


def who_is_right(pop: Population) -> pd.DataFrame:
    """Label metrics per level (and overall) and method (module docstring); one row per level and method."""
    levels = category_levels(pop)
    inside, defined = _matrices(pop, levels)
    floors = level_floors(pop, inside, defined)
    groups = [("overall", "all", "category", np.ones(len(pop.y), bool), True, True)]
    groups += [(c.name, level, c.role, inside[:, j], bool(floors.meets_category_floor[j]), bool(floors.meets_label_floor[j]))
               for j, (c, level) in enumerate(levels) if inside[:, j].any()]
    y = pop.y
    rows = []
    for category, level, role, mask, meets_category, meets_label in groups:
        w = pop.weights[:, mask]
        post = role == "post_shot"
        ll_v = None
        for method in ["V", *[m for m in pop.methods if m != "V"]]:
            metrics = st.label_metrics(y[mask], pop.p(method)[:, mask], w, auc=meets_label)
            row = {"population": pop.name, "category": category, "level": level, "role": role, "method": method,
                   "n": int(mask.sum()), "goals": int(y[mask].sum()), "meets_category_floor": meets_category,
                   "meets_label_floor": meets_label, "post_contact": post}
            names = ("auc",) if post else ("observed", "expected", "o_minus_e", "o_over_e", "log_loss", "auc")
            for name in names:
                if name in metrics:
                    row.update(flat(name, st.summarise(metrics[name]), keep_nan_draws=False))
            if not post:
                ll = metrics["log_loss"]
                ll_v = ll if method == "V" else ll_v
                if method != "V":
                    row.update(flat("delta_ll_vs_V", st.summarise(ll - ll_v, "two-sided"), keep_nan_draws=False))
            rows.append(row)
    return pd.DataFrame(rows)


def disagreements(pop: Population, share: float = TAIL_SHARE) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(shots, tails, profile) of the largest observed F - V disagreements (module docstring). `shots` lists the
    shots of both tails by ID with their gap, main-scale probabilities, goal, StatsBomb outcome and categories."""
    g = gap(pop, "F", "V")[0]
    n_tail = int(round(share * len(g)))
    order = np.argsort(g, kind="stable")
    tails = {"F_above_V": order[::-1][:n_tail], "F_below_V": order[:n_tail]}
    category_columns = [c.column for c in cat.CATEGORIES if pop.categories_defined(c.name)]
    probs = {m: pop.p(m)[0] for m in pop.methods}
    shots = []
    for tail, idx in tails.items():
        part = pop.rows.iloc[idx][["key", "soccernet_shot_id", "statsbomb_event_id", "match", "goal_sn", "sb_outcome", *category_columns]]
        part = part.assign(direction=tail, rank=np.arange(1, len(idx) + 1), gap_F_minus_V=g[idx], **{f"p_{m}": probs[m][idx] for m in pop.methods})
        shots.append(part)
    shots = pd.concat(shots, ignore_index=True)
    y, w = pop.y, pop.weights
    summary, profile = [], []
    for tail, idx in [*tails.items(), ("population", np.arange(len(g)))]:
        mask = np.zeros(len(g), bool)
        mask[idx] = True
        rate = (w[:, mask] @ y[mask]) / w[:, mask].sum(1)
        row = {"direction": tail, "n": int(mask.sum()), "goals": int(y[mask].sum()), "mean_gap": float(g[mask].mean()),
               **flat("goal_rate", st.summarise(rate), keep_nan_draws=False)}
        for m in pop.methods:
            row[f"mean_p_{m}"] = float(probs[m][mask].mean())
        summary.append(row)
        for column in category_columns:
            shares = pop.rows[column][mask].fillna(UNDEFINED).value_counts(normalize=True)
            for level, value in shares.items():
                profile.append({"direction": tail, "category": column.removeprefix(cat.PREFIX), "level": level, "share": float(value)})
    profile = pd.DataFrame(profile)
    population = profile[profile.direction.eq("population")].set_index(["category", "level"]).share
    profile["population_share"] = [population.get((c, lv), 0.0) for c, lv in zip(profile.category, profile.level, strict=True)]
    profile["lift"] = profile.share / profile.population_share
    return shots, pd.DataFrame(summary), profile[~profile.direction.eq("population")].reset_index(drop=True)


def raw_levels(pop: Population) -> pd.DataFrame:
    """Per method, once for context: the goal rate, the mean raw and main-scale probabilities and their log-loss."""
    y = pop.y
    rows = []
    for m in pop.methods:
        raw, main = pop.p(m, "raw"), pop.p(m)[0]
        rows.append({"population": pop.name, "method": m, "n": len(y), "goals": int(y.sum()), "goal_rate": float(y.mean()),
                     "mean_p_raw": float(raw.mean()), "mean_p_main": float(main.mean()),
                     "log_loss_raw": float(st.label_metrics(y, raw, pop.weights[:1], auc=False)["log_loss"][0]),
                     "log_loss_main": float(st.label_metrics(y, main, pop.weights[:1], auc=False)["log_loss"][0]),
                     "ll_main_minus_V": float(log_loss_difference(pop, m, "V")[0].mean())})
    return pd.DataFrame(rows)
