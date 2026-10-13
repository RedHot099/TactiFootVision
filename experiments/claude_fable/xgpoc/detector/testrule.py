"""The frozen test rule of the xg-detector study (ticket 06, council C2): its parameters and its arithmetic.

The parameters live in `research/xg_detector_test_rule.json` next to the rule's text (`xg_detector_test_rule.md`); this
module reads them and implements every computation the single test (`scripts/xgd_test.py`, ticket 07) makes after the
predictions exist. The power computation of the rule (`scripts/xgd_test_power.py`) uses the same functions on
development data.

    ensemble_probability   both arms: member logits temperature-scaled, averaged in logit space, Platt fitted on the
                           pool's out-of-fold ensemble, prior correction, penalties routed (`g_bundle.ensemble_probability`)
    shot_log_loss          per-shot log-loss with the rule's fixed probability clip
    match_weights          the paired match bootstrap: (draws, shots) weights, each shot weighted by how often its match
                           was drawn (`match_draw_counts`, the rule's draws and seed)
    paired_delta           ΔLL = variant - baseline, shot-weighted, with the percentile intervals of the rule
    reading                success / equivalent within δ / success and equivalent / inconclusive / worse, the smallest
                           equivalence margin
    expected_power         normal-approximation probabilities of each reading at an assumed Δ and standard error
"""

import json
from pathlib import Path
from statistics import NormalDist

import numpy as np

from xgpoc.models.g_bundle import ensemble_probability
from xgpoc.probe.readability import match_draw_counts

EXPERIMENT = Path(__file__).resolve().parents[2]
RULE_JSON = EXPERIMENT / "research/xg_detector_test_rule.json"
RULE_MD = EXPERIMENT / "research/xg_detector_test_rule.md"
LABELS = ("success and equivalent", "success", "equivalent within δ", "worse", "inconclusive")

__all__ = ["ensemble_probability", "expected_power", "load_rule", "match_weights", "paired_delta", "reading",
           "shot_log_loss"]


def load_rule(path: Path = RULE_JSON) -> dict:
    """The rule's parameters, refused unless they hold the fields this module reads with the values council C2 fixed."""
    rule = json.loads(Path(path).read_text())
    fixed = {("variant",): "A1", ("bootstrap", "draws"): 5000, ("success", "interval"): 0.95,
             ("equivalence", "margin"): 0.005, ("equivalence", "interval"): 0.90, ("population", "matches"): 172}
    for keys, value in fixed.items():
        found = rule
        for k in keys:
            found = found[k]
        if found != value:
            raise ValueError(f"rule field {'.'.join(keys)} is {found!r}, council C2 fixed {value!r}")
    if tuple(rule["labels"]) != LABELS:
        raise ValueError(f"rule labels {rule['labels']} differ from {LABELS}")
    return rule


def shot_log_loss(y: np.ndarray, p: np.ndarray, clip: float) -> np.ndarray:
    """-(y log p + (1 - y) log(1 - p)) per shot, p clipped to [clip, 1 - clip]."""
    y = np.asarray(y, float)
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("labels must be 0 or 1")
    p = np.clip(np.asarray(p, float), clip, 1 - clip)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def match_weights(games: np.ndarray, draws: int, seed: int) -> np.ndarray:
    """(draws, shots) float weights: each shot weighted by the number of times its match is drawn in each bootstrap
    replicate (`match_draw_counts` over the sorted unique matches)."""
    games = np.asarray(games)
    keys, counts = match_draw_counts(games, draws, seed)
    return counts[:, np.searchsorted(keys, games)].astype(float)


def _percentiles(boot: np.ndarray, level: float) -> tuple[float, float]:
    tail = 100 * (1 - level) / 2
    return float(np.percentile(boot, tail)), float(np.percentile(boot, 100 - tail))


def paired_delta(y: np.ndarray, p_variant: np.ndarray, p_base: np.ndarray, games: np.ndarray, rule: dict) -> dict:
    """ΔLL = mean over shots of LL(variant) - LL(baseline), with the paired match bootstrap of the rule: each replicate is
    the shot-weighted mean of the per-shot differences, shots weighted by their match's draw count. Returns the point,
    the success interval (95%), the equivalence interval (90%, from the same draws) and the replicates' SD."""
    clip = rule["metric"]["clip"]
    d = shot_log_loss(y, p_variant, clip) - shot_log_loss(y, p_base, clip)
    weights = match_weights(games, rule["bootstrap"]["draws"], rule["bootstrap"]["seed"])
    boot = (weights * d).sum(axis=1) / weights.sum(axis=1)
    lo95, hi95 = _percentiles(boot, rule["success"]["interval"])
    lo90, hi90 = _percentiles(boot, rule["equivalence"]["interval"])
    return {"shots": int(len(d)), "matches": int(len(np.unique(games))), "delta_ll": float(d.mean()),
            "ll_variant": float(shot_log_loss(y, p_variant, clip).mean()), "ll_base": float(shot_log_loss(y, p_base, clip).mean()),
            "lo95": lo95, "hi95": hi95, "lo90": lo90, "hi90": hi90, "boot_sd": float(boot.std(ddof=1))}


def reading(delta: dict, rule: dict) -> dict:
    """The pre-registered reading of one paired comparison.

    success      the upper bound of the 95% interval is below 0 (the only confirmatory criterion);
    worse        the lower bound of the 95% interval is above 0;
    equivalent   TOST at alpha 0.05: the 90% interval lies wholly inside [-δ, +δ] (descriptive, never a success).
    The label takes the first that applies of: success and equivalent, success, worse, equivalent within δ,
    inconclusive; `equivalent` is reported as its own field in every case. The smallest equivalence margin is the
    smallest δ' whose [-δ', +δ'] contains the 90% interval."""
    margin = rule["equivalence"]["margin"]
    success, worse = delta["hi95"] < 0, delta["lo95"] > 0
    equivalent = -margin < delta["lo90"] and delta["hi90"] < margin
    if success:
        label = "success and equivalent" if equivalent else "success"
    elif worse:
        label = "worse"
    else:
        label = "equivalent within δ" if equivalent else "inconclusive"
    return {"label": label, "success": bool(success), "worse": bool(worse), "equivalent": bool(equivalent),
            "margin": margin, "smallest_equivalence_margin": float(max(abs(delta["lo90"]), abs(delta["hi90"])))}


def expected_power(delta: float, se: float, rule: dict) -> dict:
    """Normal approximation of the test's readings when the true ΔLL is `delta` and its standard error `se`:
    P(success) = P(upper 95% bound < 0), P(worse), P(equivalent within δ), and the expected interval half-widths and
    smallest equivalence margin. The test's intervals are percentile intervals of the match bootstrap; the normal
    approximation is the planning figure."""
    n = NormalDist()
    z95 = n.inv_cdf(1 - (1 - rule["success"]["interval"]) / 2)
    z90 = n.inv_cdf(1 - (1 - rule["equivalence"]["interval"]) / 2)
    margin = rule["equivalence"]["margin"]
    equivalent = max(0.0, n.cdf((margin - delta) / se - z90) - n.cdf((-margin - delta) / se + z90))
    return {"delta": delta, "se": se, "p_success": n.cdf(-delta / se - z95), "p_worse": n.cdf(delta / se - z95),
            "p_equivalent": equivalent, "half_width_95": z95 * se, "half_width_90": z90 * se,
            "expected_smallest_margin": abs(delta) + z90 * se}
