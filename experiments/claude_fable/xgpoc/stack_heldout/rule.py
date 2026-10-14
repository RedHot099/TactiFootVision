"""The frozen rule of the xg-stack test (ticket 04, council C7): its parameters and its arithmetic.

The parameters live in `research/xg_stack_test_rule.json` next to the rule's text (`xg_stack_test_rule.md`). This
module checks them against the values C7 fixed (`load_rule`) and implements every computation of the test:

    delta              ΔLL = variant - base, shot-weighted, with percentile intervals of the paired match bootstrap at
                       every level the rule names (the success ladder and the 90% interval of the TOST)
    reading            the label (`LABELS`, in precedence order) at the success level in force, the equivalence flags at
                       δ = 0.002 (the label's) and δ = 0.005 (comparability), the practical-bar flag and the smallest
                       equivalence margin
    sentence           the report sentence written in advance for the label, with the numbers filled in
    null_gate          the synthetic-null false-success check on the written test predictions (labels unread): its
                       rates per ladder level and the success level in force
    power_scenarios    the planning power of the readings under C7's standard-error and effect scenarios

The bootstrap draws are `testrule.match_weights` (A1's draws: `readability.match_draw_counts` over the sorted unique
matches), so one (draws, shots) weight matrix serves the test, its descriptive readings and the gate.
"""

import json
from pathlib import Path

import numpy as np

from xgpoc.bundle import canonical_digest
from xgpoc.detector.testrule import expected_power, match_weights, shot_log_loss
from xgpoc.eval.calibration import logit, sigmoid

EXPERIMENT = Path(__file__).resolve().parents[2]
RULE_JSON = EXPERIMENT / "research/xg_stack_test_rule.json"
RULE_MD = EXPERIMENT / "research/xg_stack_test_rule.md"
LABELS = ("success and equivalent", "success", "worse", "equivalent within δ", "inconclusive")   # precedence order
DEFAULT_LEVEL = 0.95
LADDER = (0.975, 0.99, 0.995)              # C7 Q5: the levels the gate may move success to, in order
LEVELS = (DEFAULT_LEVEL, *LADDER)
TOST_INTERVAL = 0.90
MARGIN = 0.002                             # the label's δ (C7 Q3: the practical bar)
ALSO_MARGIN = 0.005                        # A1's δ, reported for comparability
PRACTICAL = -0.002
BEFORE_SCORING = "before_scoring"          # the block written after the gate and the power, before the freeze

FIXED = {("confirmatory",): "G", ("reference",): "R", ("bootstrap", "draws"): 5000, ("bootstrap", "seed"): 0,
         ("metric", "clip"): 1e-6, ("success", "default_interval"): DEFAULT_LEVEL, ("success", "ladder"): list(LADDER),
         ("equivalence", "margin"): MARGIN, ("equivalence", "also_margin"): ALSO_MARGIN,
         ("equivalence", "interval"): TOST_INTERVAL, ("practical_bar",): PRACTICAL, ("population", "matches"): 172,
         ("population", "expected_shots"): 3819, ("gate", "max_false_success"): 0.05,
         ("gate", "random_intercept_sd"): 0.25, ("gate", "replicates"): 5000, ("gate", "seed"): 301}


def _field(rule: dict, keys: tuple[str, ...]):
    found = rule
    for k in keys:
        found = found[k]
    return found


def load_rule(path: Path = RULE_JSON) -> dict:
    """The rule's parameters, refused unless they hold the values council C7 fixed (`FIXED`) and the labels and
    sentences of `LABELS`."""
    rule = json.loads(Path(path).read_text())
    for keys, value in FIXED.items():
        if (found := _field(rule, keys)) != value:
            raise ValueError(f"rule field {'.'.join(keys)} is {found!r}, council C7 fixed {value!r}")
    if tuple(rule["labels"]) != LABELS or set(rule["sentences"]) != set(LABELS):
        raise ValueError(f"rule labels or sentences differ from {LABELS}")
    return rule


def parameters_digest(rule: dict) -> str:
    """sha256 of the rule's parameters without the block written after the gate and the power (`BEFORE_SCORING`):
    the preparation's record pins the rule by this digest, since that block is filled in after it."""
    return canonical_digest({k: v for k, v in rule.items() if k != BEFORE_SCORING})


def level_in_force(rule: dict) -> float:
    """The success level the gate fixed, refused when the rule does not state one of `LEVELS`."""
    level = rule.get(BEFORE_SCORING, {}).get("level_in_force")
    if level not in LEVELS:
        raise ValueError(f"the rule states no success level in force of {LEVELS} (found {level!r})")
    return float(level)


def key(level: float) -> str:
    return f"{level:g}"


def _interval(boot: np.ndarray, level: float) -> list[float]:
    tail = 100 * (1 - level) / 2
    return [float(np.percentile(boot, tail)), float(np.percentile(boot, 100 - tail))]


def delta(y: np.ndarray, p_variant: np.ndarray, p_base: np.ndarray, weights: np.ndarray, clip: float,
          mask: np.ndarray | None = None) -> dict:
    """ΔLL = mean over the shots of `mask` (all by default) of LL(variant) - LL(base), p clipped at `clip`, with the
    paired match bootstrap of `weights` ((draws, shots), `bootstrap_weights`): each replicate is the weighted mean of
    the per-shot differences. Intervals (percentile) at every level of `LEVELS` and the TOST's 90%, keyed by `key`."""
    ll_variant, ll_base = shot_log_loss(y, p_variant, clip), shot_log_loss(y, p_base, clip)
    d = ll_variant - ll_base
    if mask is not None:
        weights, d, ll_variant, ll_base = weights[:, mask], d[mask], ll_variant[mask], ll_base[mask]
    boot = (weights @ d) / weights.sum(axis=1)
    if not np.isfinite(boot).all():
        raise ValueError("a bootstrap draw holds none of the shots")
    return {"shots": int(len(d)), "delta_ll": float(d.mean()), "ll_variant": float(ll_variant.mean()),
            "ll_base": float(ll_base.mean()), "boot_sd": float(boot.std(ddof=1)), "draws": int(len(boot)),
            "intervals": {key(level): _interval(boot, level) for level in (*LEVELS, TOST_INTERVAL)}}


def bootstrap_weights(games: np.ndarray, rule: dict) -> np.ndarray:
    """(draws, shots) weights of the rule's paired match bootstrap (A1's `testrule.match_weights`)."""
    return match_weights(np.asarray(games), rule["bootstrap"]["draws"], rule["bootstrap"]["seed"])


def reading(result: dict, level: float) -> dict:
    """The pre-registered reading of one paired comparison at the success level `level`.

    success      the upper bound of the `level` interval is below 0 (the only confirmatory criterion);
    worse        the lower bound of the same interval is above 0;
    equivalent   TOST at α = 0.05: the 90% interval lies wholly inside the open (-δ, +δ), δ = `MARGIN` for the label
                 and `ALSO_MARGIN` reported beside it (descriptive, never a success);
    practical    success with the point ΔLL at or below `PRACTICAL`.
    The label is the first of `LABELS` that applies. The smallest equivalence margin is max(|lo90|, |hi90|)."""
    lo, hi = result["intervals"][key(level)]
    lo90, hi90 = result["intervals"][key(TOST_INTERVAL)]
    success, worse = hi < 0, lo > 0
    equivalent = {key(m): bool(-m < lo90 and hi90 < m) for m in (MARGIN, ALSO_MARGIN)}
    if success:
        label = "success and equivalent" if equivalent[key(MARGIN)] else "success"
    elif worse:
        label = "worse"
    else:
        label = "equivalent within δ" if equivalent[key(MARGIN)] else "inconclusive"
    return {"label": label, "level": level, "success": bool(success), "worse": bool(worse), "equivalent": equivalent,
            "practical": bool(success and result["delta_ll"] <= PRACTICAL),
            "smallest_equivalence_margin": float(max(abs(lo90), abs(hi90)))}


def signed(x: float, digits: int = 5) -> str:
    """A signed number with a true minus, as the reports write it."""
    return f"{x:+.{digits}f}".replace("-", "−")


def sentence(rule: dict, result: dict, read: dict) -> str:
    """The report sentence of `read`'s label (rule `sentences`), followed by the practical clause when it applies
    (rule `practical_clauses`: below or at the bar for a success), with the numbers of `result` filled in."""
    lo, hi = result["intervals"][key(read["level"])]
    lo90, hi90 = result["intervals"][key(TOST_INTERVAL)]
    values = {"delta": signed(result["delta_ll"]), "lo": signed(lo), "hi": signed(hi), "lo90": signed(lo90),
              "hi90": signed(hi90), "level": f"{100 * read['level']:g}", "margin": f"{MARGIN:g}",
              "smallest": f"{read['smallest_equivalence_margin']:.5f}"}
    text = rule["sentences"][read["label"]].format(**values)
    if read["success"]:
        text += " " + rule["practical_clauses"]["at_bar" if read["practical"] else "below_bar"].format(**values)
    return text


# ------------------------------------------------------------------ the synthetic-null gate (labels unread)
def choose_level(rates: dict[str, float], ceiling: float) -> float | None:
    """The first of `LEVELS` (0.95, then the ladder) whose false-success rate is <= `ceiling`; None when none is."""
    return next((level for level in LEVELS if rates[key(level)] <= ceiling), None)


def null_gate(p_r: np.ndarray, p_g: np.ndarray, games: np.ndarray, weights: np.ndarray, rule: dict,
              chunk: int = 100) -> dict:
    """C7 Q5's false-success check on the written test predictions, with no label: `replicates` outcome vectors
    y* ~ Bernoulli(sigmoid(logit(p_R) + u_m)), u_m ~ N(0, sd²) a match random intercept drawn per replicate before the
    outcomes, from one `default_rng(seed)`. For each y*, ΔLL(G - R) and its bootstrap with the test's own draws
    (`weights`); a false success at level L is an upper bound of the L interval below 0. The level in force is 0.95 if
    its rate is <= `max_false_success`, else the first ladder level whose rate is; None when none is (the test does
    not run)."""
    gate, clip = rule["gate"], rule["metric"]["clip"]
    n, seed, sd, ceiling = gate["replicates"], gate["seed"], gate["random_intercept_sd"], gate["max_false_success"]
    matches, index = np.unique(np.asarray(games), return_inverse=True)
    eta = logit(np.clip(p_r, clip, 1 - clip))
    d_goal = shot_log_loss(np.ones(len(p_r)), p_g, clip) - shot_log_loss(np.ones(len(p_r)), p_r, clip)
    d_miss = shot_log_loss(np.zeros(len(p_r)), p_g, clip) - shot_log_loss(np.zeros(len(p_r)), p_r, clip)
    total = weights.sum(axis=1)
    tails = {level: 100 - 100 * (1 - level) / 2 for level in LEVELS}
    rng = np.random.default_rng(seed)
    upper, lower95, points = {level: [] for level in LEVELS}, [], []
    for start in range(0, n, chunk):
        block = []
        for _ in range(min(chunk, n - start)):
            u = rng.normal(0.0, sd, len(matches))[index]
            y = rng.random(len(eta)) < sigmoid(eta + u)
            block.append(np.where(y, d_goal, d_miss))
        d = np.column_stack(block)
        boot = (weights @ d) / total[:, None]
        points.extend(d.mean(axis=0))
        for level, q in tails.items():
            upper[level].extend(np.percentile(boot, q, axis=0))
        lower95.extend(np.percentile(boot, 100 * (1 - DEFAULT_LEVEL) / 2, axis=0))
    rates = {key(level): float(np.mean(np.asarray(upper[level]) < 0)) for level in LEVELS}
    chosen = choose_level(rates, ceiling)
    return {"replicates": n, "seed": seed, "random_intercept_sd": sd, "max_false_success": ceiling,
            "false_success_rate": rates, "mc_se": {k: float(np.sqrt(r * (1 - r) / n)) for k, r in rates.items()},
            "false_worse_rate_95": float(np.mean(np.asarray(lower95) > 0)),
            "mean_null_delta": float(np.mean(points)), "level_in_force": chosen, "passed": chosen is not None}


# ------------------------------------------------------------------ power (planning, no test label)
def power_scenarios(dev: dict, test: dict, a1_test_se: float, recovery: float) -> dict:
    """C7 Q3's power scenarios, normal approximation (`testrule.expected_power`) at every success level of `LEVELS`
    and both TOST margins.

    Standard errors: the development bootstrap SD of G (`dev["boot_sd"]`) scaled by sqrt(N_dev / N_test) shots (the
    headline) and by sqrt(M_dev / M_test) matches, and A1's test SE. Effects: the development point (the headline), the
    practical bar shrunk by C6's recovery (control 4: the share of a planted -0.002 that G recovered), and the bar."""
    ses = {"dev_sd_shot_scaled": dev["boot_sd"] * float(np.sqrt(dev["shots"] / test["shots"])),
           "dev_sd_match_scaled": dev["boot_sd"] * float(np.sqrt(dev["matches"] / test["matches"])),
           "a1_test_se": a1_test_se}
    effects = {"development_point": dev["delta_ll"], "practical_recovered": PRACTICAL * recovery, "practical": PRACTICAL}
    table = []
    for se_name, se in ses.items():
        for effect_name, effect in effects.items():
            row = {"se": se_name, "se_value": se, "effect": effect_name, "delta": effect}
            for level in LEVELS:
                read = expected_power(effect, se, {"success": {"interval": level},
                                                   "equivalence": {"interval": TOST_INTERVAL, "margin": MARGIN}})
                row[f"p_success_{key(level)}"] = read["p_success"]
                row[f"p_worse_{key(level)}"] = read["p_worse"]
            for margin in (MARGIN, ALSO_MARGIN):
                row[f"p_equivalent_{key(margin)}"] = expected_power(
                    effect, se, {"success": {"interval": DEFAULT_LEVEL},
                                 "equivalence": {"interval": TOST_INTERVAL, "margin": margin}})["p_equivalent"]
            table.append(row)
    return {"standard_errors": ses, "effects": effects, "recovery": recovery, "scenarios": table,
            "headline": {"se": "dev_sd_shot_scaled", "effect": "development_point"}}
