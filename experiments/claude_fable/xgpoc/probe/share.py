"""Post-contact share, known-answer test and verdict of the direction probe (xg-credibility 10).

Pure functions of the readability curves D of ticket 09 (`xgpoc.probe.readability`), implementing the estimator, the
second positive-control condition, the known-answer test and the verdict rules of the approved protocol
`research/xgc_direction_probe_protocol.md` (sha256 `readability.PROTOCOL_SHA256`). Every number is the protocol's;
nothing here is tuned. All functions are vectorised over leading axes, so the same call gives the point estimate
(D of shape (30,)) and the bootstrap draws (D of shape (B, 30), the shared match draws of ticket 09).

Terms (protocol, "Estymator udziału klatek po kontakcie"; D, P and Delta as in `readability`):
  S_r(o)     (D_r(o) - P_r(o)) / Delta_r                         normalised curve
  U_r        S_r(0)                                              post-contact share of the anchor frames
  t50_r      first crossing of S_r = 0.5 in DENSE, linear interpolation; "<= -400" when S_r(-400) >= 0.5 (coded
             -400), "> +400" when S_r stays below 0.5 up to +400 (coded +440, the protocol's percentile code)
  KA         the known-answer test: for a random half of the shots (`default_rng(1).random(n) < 0.5` over the shots
             sorted by shot_id) the frame at offset o is replaced by the frame at o + 200 where the grid has it
  A          mean over o in KA of S_KA(o) - S(o)
  A_pred     1/2 * mean over o in KA of S(o + 200) - S(o)

Gates on the reading F (protocol, "Kontrole"): positive control (lower 95% bound of Delta > 0 and half-width of the U
interval <= 0.15), known answer (lower bound of A > 0 and 0.5 <= A / A_pred <= 1.5, point ratio), placebo and
negative control (`readability.placebo_gate`, `readability.negative_gate`). Verdict rules in order: 1 inconclusive
(an F gate fails), 2 post-contact leakage (lower bound of U_F > 0.10), 3 post-contact leakage learned only by G (G or X passes its own positive
control, its U lower bound > 0.10 and the lower bound of U_r - U_F > 0), 4 no evidence of post-contact leakage.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from xgpoc.probe.readability import (
    DENSE,
    GRID,
    OFFSET_INDEX,
    POST,
    PRE,
    interval,
    jump,
    trend,
)

KA_MOVE_MS = 200
KA = tuple(range(-400, 201, 40))            # protocol: the known-answer window, 16 offsets
KA_SEED = 1
U_POST_CONTACT_LEAKAGE = 0.10                               # verdict threshold on the lower bound of U
HALF_WIDTH_LIMIT = 0.15                     # positive control: half-width of the 95% U interval
KA_RATIO_RANGE = (0.5, 1.5)
T50_BELOW = -400.0                          # percentile codes of "<= -400" and "> +400"
T50_ABOVE = 440.0
G_RULE_READINGS = ("G", "X")                # readings of rule 3
ZERO = OFFSET_INDEX[0]

# offsets with a +200 partner on the grid; the estimator (PRE, 0, POST) and the KA window must all have one
PARTNER = {OFFSET_INDEX[o]: OFFSET_INDEX[o + KA_MOVE_MS] for o in GRID if o + KA_MOVE_MS in OFFSET_INDEX}
NO_PARTNER = tuple(o for o in GRID if OFFSET_INDEX[o] not in PARTNER)     # +240..+360, +1000: unchanged, unused
assert all(OFFSET_INDEX[o] in PARTNER for o in (*PRE, 0, *POST, *KA))
assert NO_PARTNER == (240, 280, 320, 360, 1000)

VERDICTS = {1: "inconclusive", 2: "post_contact_leakage", 3: "post_contact_leakage_learned_only_by_g",
            4: "no_evidence_of_post_contact_leakage"}
F_GATES = ("positive_control", "known_answer", "placebo", "negative_control")


# ---------------------------------------------------------------- estimator

def normalised(d: np.ndarray) -> np.ndarray:
    """S(o) = (D(o) - P(o)) / Delta over the grid, (..., 30)."""
    return (d - trend(d)) / jump(d)[..., None]


def share(d: np.ndarray) -> np.ndarray:
    """U = S(0), (...)."""
    return normalised(d)[..., ZERO]


def t50(s: np.ndarray) -> np.ndarray:
    """First crossing of S = 0.5 in DENSE with linear interpolation, (...); -400 / +440 codes as in the protocol,
    NaN where S is NaN (a skipped draw)."""
    dense = np.array(DENSE, float)
    sd = np.asarray(s, float)[..., [OFFSET_INDEX[o] for o in DENSE]]
    above = sd >= 0.5
    j = above.argmax(-1)
    prev = np.maximum(j - 1, 0)
    s_prev = np.take_along_axis(sd, prev[..., None], -1)[..., 0]
    s_j = np.take_along_axis(sd, j[..., None], -1)[..., 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        cross = dense[prev] + (0.5 - s_prev) / (s_j - s_prev) * (dense[j] - dense[prev])
    out = np.where(j == 0, T50_BELOW, cross)
    out = np.where(above.any(-1), out, T50_ABOVE)
    return np.where(np.isnan(sd).any(-1), np.nan, out)


def t50_label(value: float) -> str:
    """The protocol's reading of a t50 point value."""
    if np.isnan(value):
        return "nan"
    if value == T50_BELOW:
        return "<= -400"
    if value == T50_ABOVE:
        return "> +400"
    return f"{value:+.0f}"


# ---------------------------------------------------------------- known-answer test

def ka_half(n: int, seed: int = KA_SEED) -> np.ndarray:
    """The moved half: `default_rng(seed).random(n) < 0.5` over the n shots sorted by shot_id."""
    return np.random.default_rng(seed).random(n) < 0.5


def move_anchor(values: np.ndarray, moved: np.ndarray) -> np.ndarray:
    """Per-shot (n, 30, ...) values with, for the `moved` rows (anchor moved by +200 ms), every offset that has a +200 partner replaced by the
    partner's value; offsets without a partner stay unchanged. Works for logits and for embedding refs."""
    out = np.array(values, copy=True)
    rows = np.flatnonzero(moved)
    for j, partner in PARTNER.items():
        out[rows, j] = values[rows, partner]
    return out


@dataclass(frozen=True)
class KnownAnswer:
    """A, A_pred and the anchor-only change and its prediction, each (...)."""

    a: np.ndarray
    a_pred: np.ndarray
    anchor: np.ndarray
    anchor_pred: np.ndarray


def known_answer(d: np.ndarray, d_ka: np.ndarray) -> KnownAnswer:
    """KA statistics from the D of the original pipeline and the D of the pipeline on moved anchors (same shape)."""
    s, s_ka = normalised(d), normalised(d_ka)
    ka = [OFFSET_INDEX[o] for o in KA]
    ka_partner = [PARTNER[j] for j in ka]
    return KnownAnswer(a=(s_ka[..., ka] - s[..., ka]).mean(-1),
                       a_pred=0.5 * (s[..., ka_partner] - s[..., ka]).mean(-1),
                       anchor=s_ka[..., ZERO] - s[..., ZERO],
                       anchor_pred=0.5 * (s[..., PARTNER[ZERO]] - s[..., ZERO]))


# ---------------------------------------------------------------- summaries and gates

def estimate(point: float, draws: np.ndarray) -> dict:
    """Point value with its 95% percentile interval and the number of skipped (NaN) draws."""
    lo, hi, skipped = interval(np.asarray(draws, float))
    return {"value": float(point), "lo": float(lo), "hi": float(hi), "skipped_draws": skipped}


def half_width(est: Mapping) -> float:
    """Half-width of an `estimate`'s 95% interval."""
    return (est["hi"] - est["lo"]) / 2


def flat(name: str, est: Mapping) -> dict:
    """An `estimate` as CSV columns: name, name_lo, name_hi."""
    return {name: est["value"], f"{name}_lo": est["lo"], f"{name}_hi": est["hi"]}


def reading_summary(d: np.ndarray, d_draws: np.ndarray) -> dict:
    """Delta, U (with its half-width) and t50 of one reading and stratum, from D point (30,) and draws (B, 30)."""
    u = estimate(float(share(d)), share(d_draws))
    t = estimate(float(t50(normalised(d))), t50(normalised(d_draws)))
    return {"delta": estimate(float(jump(d)), jump(d_draws)), "u": u, "u_half_width": half_width(u),
            "t50": t | {"label": t50_label(t["value"])}}


def positive_control(summary: Mapping) -> dict:
    """Both conditions of the positive control on a `reading_summary`."""
    lower = summary["delta"]["lo"] > 0
    narrow = summary["u_half_width"] <= HALF_WIDTH_LIMIT
    return {"delta_lower_positive": bool(lower), "u_half_width": summary["u_half_width"],
            "u_half_width_ok": bool(narrow), "passed": bool(lower and narrow)}


def known_answer_gate(point: KnownAnswer, draws: KnownAnswer) -> dict:
    """A, A_pred, A / A_pred (point ratio decides; its interval is descriptive) and the anchor-only change."""
    a = estimate(float(point.a), draws.a)
    ratio = float(point.a / point.a_pred)
    lo, hi = KA_RATIO_RANGE
    detected = a["lo"] > 0
    in_range = bool(lo <= ratio <= hi)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio_draws = draws.a / draws.a_pred
    return {"A": a, "A_pred": estimate(float(point.a_pred), draws.a_pred), "ratio": ratio,
            "ratio_interval": estimate(ratio, ratio_draws), "A_lower_positive": bool(detected),
            "ratio_in_range": in_range, "passed": bool(detected and in_range),
            "anchor_change": estimate(float(point.anchor), draws.anchor),
            "anchor_change_pred": estimate(float(point.anchor_pred), draws.anchor_pred)}


def verdict(f_gates: Mapping[str, bool], u_f: Mapping, g_rule: Mapping[str, Mapping]) -> dict:
    """The protocol's verdict, rules applied in order.

    `f_gates`: pass/fail of the four F gates (`F_GATES`); `u_f`: `estimate` of U_F; `g_rule[r]` for r in G, X: the
    reading's own positive control (`positive_passed`), the `estimate` of U_r (`u`) and of U_r - U_F (`diff`)."""
    if set(f_gates) != set(F_GATES):
        raise ValueError(f"expected the gates {F_GATES}, got {sorted(f_gates)}")
    rule3 = {r: {"positive_passed": bool(v["positive_passed"]), "u_lower_above": bool(v["u"]["lo"] > U_POST_CONTACT_LEAKAGE),
                 "diff_lower_positive": bool(v["diff"]["lo"] > 0)} for r, v in g_rule.items()}
    for v in rule3.values():
        v["met"] = all(v.values())
    failed = [g for g in F_GATES if not f_gates[g]]
    if failed:
        rule = 1
    elif u_f["lo"] > U_POST_CONTACT_LEAKAGE:
        rule = 2
    elif any(v["met"] for v in rule3.values()):
        rule = 3
    else:
        rule = 4
    out = {"rule": rule, "verdict": VERDICTS[rule], "failed_gates": failed, "u_f_lower_above": bool(u_f["lo"] > U_POST_CONTACT_LEAKAGE),
           "rule3": rule3}
    if rule == 4:
        out["sentence"] = ("udział powyżej 0,10 wykluczony" if u_f["hi"] <= U_POST_CONTACT_LEAKAGE
                           else f"udział do {u_f['hi']:.2f} nie wykluczony".replace(".", ","))
    return out


def paired_difference(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> dict:
    """`estimate` of a - b from (point, draws) pairs on the same bootstrap draws."""
    return estimate(float(a[0] - b[0]), a[1] - b[1])

