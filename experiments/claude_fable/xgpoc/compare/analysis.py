"""The tested part of the xg-compare analysis (ticket 02, council C4): the pre-declared family, the items outside it
and the open scan, on the populations of `xgpoc.compare.population`. The descriptive part is
`xgpoc.compare.describe`.

Family (`family`)
    every item of `protocol.FAMILY` on its tested statistic (columns effect_*): the adjusted effect of gap items on
    g = z_F - z_V (main scale) with control z_D and of log-loss items on LL(F) - LL(V) of the main-scale
    probabilities with control z_D; item 8 Spearman(F, D) - Spearman(F, V) of the raw probabilities; item 9
    [Spearman(F, V) - Spearman(P, V)] in headers minus in foot shots (`power.agreement_contrast`). Beside each gap
    and log-loss item: the unadjusted contrast, the D - V and P - V references, the raw and rank sensitivities, the
    item's level sensitivity (`SENSITIVITY_LEVELS`: item 2 at 3 s) and every logit effect in xG points at the level's
    mean vendor probability. p-values have the item's side. BH and BY q-values run over the items that meet their
    floors (as the power simulation of ticket 01 does); `reading` places each item against its threshold.
Outside the family (`item_1`, `item_13`, `level_item`)
    items 1 and 13 regress the F - V and (F-400) - V gaps of non-goal shots on the on-target indicator with the
    paper's H4 controls (the vendor logit on the same scale, StatsBomb body-part and shot-type indicators against
    their most frequent level); item 6b and the negative control (first against second half) are level effects.
Scan (`scan_discovery`, `freeze_candidates`, `scan_validation`)
    F - V adjusted effects of every scan level on the discovery half; candidates (|effect| >= 0.20, two-sided p <
    0.05, the floors met in both halves) go to a file whose SHA-256 `scan_validation` checks before it builds the
    validation half; there, a one-sided p-value in the discovered direction and Holm at 0.05.
"""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from xgpoc.bundle import sha256_file
from xgpoc.compare import categories as cat
from xgpoc.compare import power as pw
from xgpoc.compare import protocol as pr
from xgpoc.compare import stats as st
from xgpoc.compare.population import (
    Population,
    flat,
    floors,
    gap,
    json_text,
    level_matrices,
    log_loss_difference,
    mean_vendor_probability,
    membership,
    scales_for,
)

REFERENCES = (("D", "V"), ("P", "V"))
SENSITIVITY_LEVELS = {"2": ("rebound_3s", "yes")}       # protocol: item 2's 3 s window
ON_TARGET = ("Saved", "Saved to Post")                  # item 1 (protocol, outside_family)
OFF_TARGET = ("Off T", "Post", "Wayward")
APART = ("Blocked", "Saved Off Target")
H4_CONTROLS = ("sb_body", "sb_stype")
ON_TARGET_SIDE = "greater"                              # items 1 and 13: the owner's statement, F - V larger on target
NEGATIVE_CONTROL = ("half", "first", ("first", "second"))
OPEN_GOAL = ("open_goal", "yes")
XG_BOUNDS = ("point", "lo", "hi")


def outcome(pop: Population, kind: str, a: str, b: str, scale: str) -> np.ndarray:
    """The per-shot outcome of an effect kind: the gap or the log-loss difference of a - b."""
    return gap(pop, a, b, scale) if kind == "gap" else log_loss_difference(pop, a, b, scale)


def effect(pop: Population, kind: str, a: str, b: str, scale: str, inside: np.ndarray, defined: np.ndarray,
           side: str | None = None) -> tuple[dict, dict]:
    """(adjusted, unadjusted) summaries of one level: the outcome of `kind` for a - b on `scale`, control z_D on the
    same scale."""
    adjusted, unadjusted = st.level_effects(outcome(pop, kind, a, b, scale), pop.z("D", scale), inside[:, None], defined[:, None],
                                            pop.weights)
    return st.summarise(adjusted[:, 0], side), st.summarise(unadjusted[:, 0])


def xg_points(prefix: str, summary: dict, p_bar: float) -> dict:
    """The point and interval of a main-scale logit summary in xG points at `p_bar` (`stats.xg_points`)."""
    return {f"{prefix}_xg_points_{k}": st.xg_points(summary[k], p_bar) for k in XG_BOUNDS}


def gap_record(pop: Population, prefix: str, kind: str, a: str, b: str, inside: np.ndarray, defined: np.ndarray, side: str,
               p_bar: float) -> dict:
    """The adjusted main-scale effect of a - b with its xG points for gap kinds."""
    adjusted = effect(pop, kind, a, b, "main", inside, defined, side)[0]
    return flat(prefix, adjusted) | (xg_points(prefix, adjusted, p_bar) if kind == "gap" else {})


# ------------------------------------------------------------------ the family
def reading(tested: bool, q_bh: float, point: float, lo: float, hi: float, side: str, threshold: float) -> str:
    """The item against its threshold. Below a floor: descriptive. Otherwise BH-significant at or beyond the
    threshold in the tested direction, BH-significant but smaller, or not significant with the 95% interval inside
    the threshold in the tested direction (no effect of interesting size) or reaching it (inconclusive)."""
    if not tested:
        return "below floor: descriptive"
    beyond = point >= threshold if side == "greater" else point <= -threshold if side == "less" else abs(point) >= threshold
    if q_bh <= pr.Q_BH:
        return "BH-significant, at or beyond the threshold" if beyond else "BH-significant, below the threshold"
    inside = hi < threshold if side == "greater" else lo > -threshold if side == "less" else (lo > -threshold and hi < threshold)
    return "not significant, interval inside the threshold" if inside else "not significant, inconclusive"


def effect_item(pop: Population, item: pr.Item, label_free: bool) -> dict:
    """One gap or log-loss item (module docstring)."""
    inside, defined = membership(pop, item.category, item.level)
    row = floors(pop, inside, defined)
    row["tested"] = row["meets_category_floor"] and (item.kind != "log_loss" or row["meets_label_floor"])
    p_bar = mean_vendor_probability(pop, inside)
    adjusted, unadjusted = effect(pop, item.kind, "F", "V", "main", inside, defined, item.side)
    row.update(mean_vendor_p_in=p_bar, **flat("effect", adjusted), **flat("unadjusted", unadjusted))
    if item.kind == "gap":
        row.update(xg_points("effect", adjusted, p_bar))
    for a, b in REFERENCES:
        row.update(gap_record(pop, f"ref_{a}{b}", item.kind, a, b, inside, defined, "two-sided", p_bar))
    if item.id in SENSITIVITY_LEVELS:
        s_inside, s_defined = membership(pop, *SENSITIVITY_LEVELS[item.id])
        row.update(n_in_sensitivity=int(s_inside.sum()),
                   **gap_record(pop, "sensitivity", item.kind, "F", "V", s_inside, s_defined, item.side, mean_vendor_probability(pop, s_inside)))
    if label_free:
        row.update(flat("raw", effect(pop, item.kind, "F", "V", "raw", inside, defined, item.side)[0]))
        if item.kind == "gap":
            row.update(flat("rank", effect(pop, item.kind, "F", "V", "rank", inside, defined, item.side)[0]))
    return row


def correlation_item(pop: Population) -> dict:
    """Item 8: Spearman(F, D) - Spearman(F, V) of the raw probabilities; Pearson of the main-scale and of the raw
    logits as sensitivities."""
    w = pop.weights
    fd, fv = pw.spearman(pop.raw["F"], pop.raw["D"], w), pw.spearman(pop.raw["F"], pop.raw["V"], w)
    row = {"n_in": len(pop.y), "matches_in": int(len(np.unique(pop.match))), "tested": True,
           **flat("effect", st.summarise(fd - fv, pr.BY_ID["8"].side)), "spearman_FD": float(fd[0]), "spearman_FV": float(fv[0])}
    for scale in ("main", "raw"):
        z = {m: pop.z(m, scale) for m in ("F", "D", "V")}
        pearson = st.pearson(z["F"], z["D"], w) - st.pearson(z["F"], z["V"], w)
        row.update(flat(f"pearson_{scale}", st.summarise(pearson, pr.BY_ID["8"].side)))
    return row


def agreement_contrast_item(pop: Population) -> dict:
    """Item 9 on the raw logits (Spearman of the probabilities), tested when headers and foot shots each meet the
    category floor (headers also in 20 matches)."""
    body = pop.levels_of("body")
    head, foot = body == "Head", np.isin(body, cat.FOOT)
    point, draws = pw.agreement_contrast(pop.raw, head, foot, pop.match, pop.counts)
    counts = floors(pop, head, head | foot)
    tested = counts["n_in"] >= pr.FLOORS["category_in"] and counts["n_out"] >= pr.FLOORS["category_in"] \
        and counts["matches_in"] >= pr.FLOORS["category_matches_in"]
    return {"n_in": counts["n_in"], "n_out": counts["n_out"], "matches_in": counts["matches_in"], "tested": bool(tested),
            **flat("effect", st.summarise(np.r_[point, draws], pr.BY_ID["9"].side))}


def family(pop: Population, label_free: bool = True) -> pd.DataFrame:
    """One row per family item (module docstring). With `label_free` False (the dry run on permuted labels) items 8
    and 9 and the raw and rank sensitivities are left out: they read no label, so a label permutation cannot null
    them, and computing them would show real results before the run."""
    rows = []
    for item in pr.FAMILY:
        base = {"item": item.id, "label": item.label, "kind": item.kind, "side": item.side, "threshold": item.threshold}
        if item.kind in ("gap", "log_loss") and not pop.categories_defined(item.category):
            rows.append(base | {"tested": False, "status": f"not defined in the {pop.base} population"})
        elif item.kind in ("gap", "log_loss"):
            rows.append(base | effect_item(pop, item, label_free) | {"status": "run"})
        elif not label_free:
            rows.append(base | {"tested": False, "status": "label-free: not run on permuted labels"})
        elif item.kind == "correlation_difference":
            rows.append(base | correlation_item(pop) | {"status": "run"})
        else:
            rows.append(base | agreement_contrast_item(pop) | {"status": "run"})
    out = pd.DataFrame(rows)
    p = np.where(out.tested.astype(bool), out.effect_p, np.nan)
    out["q_bh"], out["q_by"] = st.q_values(p), st.q_values(p, dependent=True)
    out["reject_bh"], out["reject_by"] = out.q_bh <= pr.Q_BH, out.q_by <= pr.Q_BH
    out["interesting"] = [bool(t) and pw.interesting(pr.BY_ID[i], pt) for i, t, pt in zip(out.item, out.tested, out.effect_point, strict=True)]
    out["reading"] = [reading(bool(r.tested), r.q_bh, r.effect_point, r.effect_lo, r.effect_hi, r.side, r.threshold) if r.status == "run"
                      else r.status for r in out.itertuples()]
    return out


# ------------------------------------------------------------------ outside the family
def dummies(values: pd.Series) -> tuple[list[np.ndarray], list[str]]:
    """Indicators of every level but the most frequent (ties: first by name), the rule of `xgpoc.vendor.leakage.design`
    (which builds them inside its own design matrix, so it is not reusable here)."""
    counts = values.value_counts()
    ranked = sorted(counts.index, key=lambda level: (-counts[level], level))
    levels = sorted(ranked[1:])
    return [values.eq(level).to_numpy(float) for level in levels], [f"{values.name}={level}" for level in levels]


def h4_regression(pop: Population, keep: np.ndarray, on_target: np.ndarray, method: str, scale: str) -> dict:
    """The coefficient of the on-target indicator in OLS fits of z_method - z_V over the rows in `keep`: with the H4
    controls (z_V on the same scale, body part and shot type) and without them; main-scale effects also in xG
    points at the rows' mean vendor probability."""
    shape = pop.weights.shape
    w = pop.weights[:, keep]
    y = np.broadcast_to(gap(pop, method, "V", scale), shape)[:, keep]
    zv = np.broadcast_to(pop.z("V", scale), shape)[:, keep]
    rows = pop.rows[keep]
    fixed, names = [], []
    for control in H4_CONTROLS:
        columns, labels = dummies(rows[control].reset_index(drop=True))
        fixed += columns
        names += labels
    on = on_target[keep].astype(float)
    base = np.column_stack([np.ones(len(on)), on])
    controls = np.column_stack(fixed) if fixed else np.empty((len(on), 0))
    x = np.concatenate([np.broadcast_to(base, (len(w),) + base.shape), zv[:, :, None],
                        np.broadcast_to(controls, (len(w),) + controls.shape)], axis=2)
    adjusted = st.summarise(st.ols(y, x, w)[:, 1], ON_TARGET_SIDE)
    unadjusted = st.summarise(st.ols(y, base, w)[:, 1], ON_TARGET_SIDE)
    out = {"controls": ["vendor logit", *names], **flat("adjusted", adjusted), **flat("unadjusted", unadjusted)}
    if scale == "main":
        out.update(xg_points("adjusted", adjusted, mean_vendor_probability(pop, keep)))
    return out


def weighted_group_mean(values: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(W,) weighted mean of `values` (W, n) over the shots in `mask`."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return (weights[:, mask] * values[:, mask]).sum(1) / weights[:, mask].sum(1)


def on_target_regressions(pop: Population, keep: np.ndarray, on: np.ndarray, label_free: bool) -> dict:
    """`h4_regression` for F and F-400 on every reported scale."""
    return {f"{method}|{scale}": h4_regression(pop, keep, on, method, scale) for scale in scales_for(label_free) for method in ("F", "F-400")}


def item_1(pop: Population, label_free: bool = True) -> dict:
    """Item 1: non-goal shots from outside the box, on target (saved) against off target (off target, post, wayward)
    on the StatsBomb outcome; F and F-400 side by side; Blocked and Saved Off Target apart (module docstring)."""
    sb_outcome = pop.rows.sb_outcome.to_numpy(object)
    base = (pop.y == 0) & (pop.levels_of("outside_box") == "yes")
    keep = base & np.isin(sb_outcome, ON_TARGET + OFF_TARGET)
    on = np.isin(sb_outcome, ON_TARGET)
    groups = {o: base & (sb_outcome == o) for o in (*ON_TARGET, *OFF_TARGET, *APART)}
    out = {"population": pop.name, "n_on_target": int((keep & on).sum()), "n_off_target": int((keep & ~on).sum()),
           "matches": int(len(np.unique(pop.match[keep]))), "n_apart": {o: int(groups[o].sum()) for o in APART},
           "side": ON_TARGET_SIDE, "regressions": on_target_regressions(pop, keep, on, label_free), "group_means": {}}
    w = pop.weights
    for method in ("F", "F-400"):
        g = np.broadcast_to(gap(pop, method, "V"), w.shape)
        out["group_means"][method] = {o: {"n": int(mask.sum()), **st.summarise(weighted_group_mean(g, w, mask))}
                                      for o, mask in groups.items() if mask.any()}
    return out


def item_13(pop: Population, label_free: bool = True) -> dict:
    """Item 13: non-goal shots, SoccerNet on target (blocked shots count) against the rest, with the H4 controls; F
    and F-400; the raw-scale F row re-expresses the paper's H4 coefficient."""
    on = pop.rows.on_target_sn.astype("boolean")
    keep = (pop.y == 0) & on.notna().to_numpy()
    on = on.fillna(False).to_numpy(bool)
    return {"population": pop.name, "n_non_goals": int(keep.sum()), "n_on_target": int((keep & on).sum()),
            "matches": int(len(np.unique(pop.match[keep]))), "side": ON_TARGET_SIDE,
            "regressions": on_target_regressions(pop, keep, on, label_free)}


def level_item(pop: Population, category: str, level: str, reference: tuple[str, ...] | None = None,
               label_free: bool = True) -> dict:
    """A descriptive level effect (item 6b, the negative control): F - V adjusted and unadjusted with xG points, the
    D - V and P - V references and, if `label_free`, the raw scale; two-sided p-values."""
    inside, defined = membership(pop, category, level, reference)
    p_bar = mean_vendor_probability(pop, inside)
    adjusted, unadjusted = effect(pop, "gap", "F", "V", "main", inside, defined, "two-sided")
    row = {"population": pop.name, "category": category, "level": level, **floors(pop, inside, defined),
           **flat("main", adjusted), **xg_points("main", adjusted, p_bar), **flat("unadjusted", unadjusted)}
    for a, b in REFERENCES:
        row.update(gap_record(pop, f"ref_{a}{b}", "gap", a, b, inside, defined, "two-sided", p_bar))
    if label_free:
        row.update(flat("raw", effect(pop, "gap", "F", "V", "raw", inside, defined, "two-sided")[0]))
    row["within_threshold"] = abs(row["main_point"]) < pr.THRESHOLDS["family_logit"]
    return row


# ------------------------------------------------------------------ the open scan
def scan_discovery(pop: Population, coverage: pd.DataFrame) -> pd.DataFrame:
    """Per scan level on the discovery half: the label-free floors (`table.coverage` of the primary population:
    the category floor on all primary shots and the scan floor in each half), the adjusted F - V effect with its xG
    points, its two-sided p-value and whether it is a candidate."""
    levels = pr.scan_levels()
    inside, defined = level_matrices(pop, levels)
    adjusted, unadjusted = st.level_effects(gap(pop, "F", "V"), pop.z("D"), inside, defined, pop.weights)
    primary = coverage[coverage.population.eq("primary")].set_index(["category", "level"])
    rows = []
    for j, (category, level) in enumerate(levels):
        s = st.summarise(adjusted[:, j], "two-sided")
        p_bar = mean_vendor_probability(pop, inside[:, j]) if inside[:, j].any() else np.nan
        rows.append({"category": category, "level": level, "meets_floors": bool(primary.loc[(category, level), "meets_scan_floor"]),
                     "n_in": int(inside[:, j].sum()), "n_out": int((defined[:, j] & ~inside[:, j]).sum()), **s,
                     **xg_points("adjusted", s, p_bar), "unadjusted": float(unadjusted[0, j])})
    out = pd.DataFrame(rows)
    out["candidate"] = out.meets_floors & (out.point.abs() >= pr.THRESHOLDS["scan_logit"]) & (out.p < pr.SCAN_ALPHA)
    return out


def freeze_candidates(discovery: pd.DataFrame, path: Path, context: dict) -> str:
    """Write the candidates (category, level, direction, estimand, discovery numbers) and `context` to `path`; return
    its SHA-256. Refuses to overwrite: the candidates are frozen once."""
    path = Path(path)
    if path.exists():
        raise ValueError(f"{path} exists; the candidates are frozen once")
    candidates = discovery[discovery.candidate]
    record = {"rule": pr.design()["scan"]["candidate"],
              "estimand": "coefficient of the level indicator in the weighted OLS fit g ~ 1 + indicator + z_D over the validation "
                          "half's defined shots, g = z_F - z_V on the main scale refitted inside the validation half and in each "
                          "of its draws",
              "test": f"one-sided bootstrap p-value in the discovered direction; Holm at {pr.HOLM_ALPHA} across the candidates",
              "candidates": [{"category": r.category, "level": r.level, "direction": "greater" if r.point > 0 else "less",
                              "discovery": {"point": r.point, "lo": r.lo, "hi": r.hi, "p": r.p, "n_in": r.n_in}}
                             for r in candidates.itertuples()],
              "levels_meeting_floors": int(discovery.meets_floors.sum()), **context}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_text(record))
    return sha256_file(path)


def scan_validation(make_validation: Callable[[], Population], path: Path, sha256: str) -> pd.DataFrame:
    """The validation of the frozen candidates: checks the file's SHA-256, then builds the validation half
    (`make_validation()`, called only here) and tests each candidate one-sided in its direction, Holm across them."""
    if sha256_file(path) != sha256:
        raise ValueError(f"{path} is not the frozen candidate file")
    candidates = json.loads(Path(path).read_text())["candidates"]
    if not candidates:
        return pd.DataFrame(columns=["category", "level", "direction", "point", "lo", "hi", "p", "replicated", "status"])
    pop = make_validation()
    inside, defined = level_matrices(pop, [(c["category"], c["level"]) for c in candidates])
    adjusted, _ = st.level_effects(gap(pop, "F", "V"), pop.z("D"), inside, defined, pop.weights)
    rows = []
    for j, c in enumerate(candidates):
        s = st.summarise(adjusted[:, j], c["direction"])
        rows.append({"category": c["category"], "level": c["level"], "direction": c["direction"], "n_in": int(inside[:, j].sum()),
                     "discovery_point": c["discovery"]["point"], **s, **xg_points("adjusted", s, mean_vendor_probability(pop, inside[:, j]))})
    out = pd.DataFrame(rows)
    out["replicated"] = pw.holm(out.p.to_numpy(float))
    out["status"] = np.where(out.replicated, "replicated exploratory", "hypothesis-generating")
    return out
