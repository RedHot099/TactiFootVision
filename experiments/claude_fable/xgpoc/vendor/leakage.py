"""H4 leakage diagnostic (xg-vendor ticket 09): does the frame read the ball after the contact?

A pure function of the evaluation table with labels joined (`xgpoc.vendor.evaltable`). Descriptive, not a test. The
logic is the prototype's (`s2_ladder.leakage`), with the fixed definitions of the spec:

Population  primary rows (the SoccerNet on-target label exists only there) that are SoccerNet non-goals (decision D4),
            not penalties, with frame_status "ok" and a finite frame logit at offset 0.
on_target   SoccerNet "Shots on target", which counts blocked shots (decision D4).
gap         logit(frame) - logit(vendor): frame_logit_0 minus the logit of pred_vendor_xg (clipped to [1e-6, 1 - 1e-6]).
offset      G(0) - G(-400 ms): the frame logit at the anchor minus at -400 ms on the same shot; `offset_logits` picks the
            calibrated columns (frame_logit_*) or the uncalibrated mean logits (frame_raw_logit_*), whose scale only
            differs by the Platt slope. Shots without both logits drop out of the offset specs.
Controls    vendor logit, body part and shot type (StatsBomb), as indicators against the most frequent level; levels
            absent from the rows drop out.

Parts (`SPECS`)
    on_target   OLS of gap on on_target with the controls ("gap_on_target"), and without them (reference).
    offset      the mean offset contrast, and OLS of the contrast on on_target with and without the controls.
    residual    the timing-residual arm: OLS of gap on the timing residual (seconds, clipped to +-1 s) and, separately,
                on early/late indicators (anchor more than 40 ms before/after the aligned StatsBomb time), with and
                without the controls, on all rows and restricted to |residual| <= 200 ms. It runs only when the
                ticket-04 verdict record says "independent" (`xgpoc.vendor.timing_verdict`), or under the development
                override on a development table.

Intervals: weighted OLS in every shared match-bootstrap draw (`xgpoc.vendor.draws`), 95% percentile intervals.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from xgpoc.eval.calibration import logit
from xgpoc.probe.readability import interval
from xgpoc.vendor.draws import TableDraws
from xgpoc.vendor.evaltable import OK
from xgpoc.vendor.timing_verdict import (
    TimingVerdict,
    TimingVerdictError,
    residual_arm_permission,
)

CONTROLS = ("sb_body_part", "sb_shot_type")
OFFSET_LOGITS = {"calibrated": ("frame_logit_0", "frame_logit_m400"), "raw": ("frame_raw_logit_0", "frame_raw_logit_m400")}
RESIDUAL_CLIP_MS = 1000.0
WITHIN_MS = 200.0
EARLY_LATE_MS = 40.0
RESTRICTIONS = ("none", "abs_residual_le_200ms")


@dataclass(frozen=True)
class Spec:
    name: str
    part: str                      # on_target | offset | residual
    outcome: str                   # gap | offset_contrast
    regressors: tuple[str, ...]
    controls: bool
    restriction: str = "none"


SPECS = (
    Spec("gap_on_target", "on_target", "gap", ("on_target",), True),
    Spec("gap_on_target_unadjusted", "on_target", "gap", ("on_target",), False),
    Spec("offset_mean", "offset", "offset_contrast", (), False),
    Spec("offset_on_target", "offset", "offset_contrast", ("on_target",), True),
    Spec("offset_on_target_unadjusted", "offset", "offset_contrast", ("on_target",), False),
    *(Spec(f"gap_{name}{'' if controls else '_unadjusted'}{'' if r == 'none' else '_within_200ms'}", "residual", "gap", regressors, controls, r)
      for r in RESTRICTIONS for name, regressors in (("residual_slope", ("residual_s",)), ("early_late", ("early", "late")))
      for controls in (True, False)),
)


def leakage_frame(table: pd.DataFrame, offset_logits: str = "calibrated") -> pd.DataFrame:
    """The leakage population (module docstring) with its derived columns; `row` is the table position."""
    if "goal_sn" not in table.columns:
        raise ValueError("the leakage diagnostic needs a table with labels joined")
    at0, atm400 = OFFSET_LOGITS[offset_logits]
    keep = (table.population.eq("primary") & ~table.goal_sn.astype(bool) & ~table.is_penalty.astype(bool)
            & table.frame_status.eq(OK) & np.isfinite(table.frame_logit_0.to_numpy(float)))
    t = table[keep.to_numpy()]
    residual = t.timing_residual_ms.to_numpy(float)
    vendor = logit(t.pred_vendor_xg.to_numpy(float))
    return pd.DataFrame({
        "row": np.flatnonzero(keep.to_numpy()), "match": t.match.to_numpy(), "split": t.split.to_numpy(),
        "on_target": t.on_target_sn.astype(bool).to_numpy(float), "vendor_logit": vendor, "gap": t.frame_logit_0.to_numpy(float) - vendor,
        "offset_contrast": t[at0].to_numpy(float) - t[atm400].to_numpy(float),
        "residual_ms": residual, "residual_s": np.clip(residual, -RESIDUAL_CLIP_MS, RESIDUAL_CLIP_MS) / 1000,
        "early": (residual < -EARLY_LATE_MS).astype(float), "late": (residual > EARLY_LATE_MS).astype(float),
        **{c: t[c].to_numpy() for c in CONTROLS}})


def design(frame: pd.DataFrame, spec: Spec) -> tuple[np.ndarray, list[str]]:
    """Intercept, the spec's regressors and (if asked) the controls: vendor logit and category indicators."""
    columns, names = [np.ones(len(frame))], ["intercept"]
    for name in spec.regressors:
        columns.append(frame[name].to_numpy(float))
        names.append(name)
    if spec.controls:
        columns.append(frame.vendor_logit.to_numpy(float))
        names.append("vendor_logit")
        for control in CONTROLS:
            counts = frame[control].value_counts()
            ranked = sorted(counts.index, key=lambda level: (-counts[level], level))
            for level in sorted(ranked[1:]):   # the most frequent level (ties: first by name) is the reference
                columns.append(frame[control].eq(level).to_numpy(float))
                names.append(f"{control}={level}")
    return np.column_stack(columns), names


def ols_with_draws(y: np.ndarray, x: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and their (B, p) weighted-OLS draws (minimum-norm when a draw loses a category)."""
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    p = x.shape[1]
    xtwx = (weights @ (x[:, :, None] * x[:, None, :]).reshape(len(x), p * p)).reshape(-1, p, p)
    xtwy = weights @ (x * y[:, None])
    return beta, np.einsum("bij,bj->bi", np.linalg.pinv(xtwx, hermitian=True), xtwy)


def run_spec(frame: pd.DataFrame, spec: Spec, subset: str, draws: TableDraws) -> list[dict]:
    """Coefficient rows of one spec on one subset of the leakage frame."""
    use = np.isfinite(frame[spec.outcome].to_numpy(float))
    if spec.restriction == "abs_residual_le_200ms":
        use &= np.abs(frame.residual_ms.to_numpy(float)) <= WITHIN_MS
    f = frame[use]
    if len(f) == 0:
        return []
    x, names = design(f, spec)
    beta, beta_draws = ols_with_draws(f[spec.outcome].to_numpy(float), x, draws.weights(f.row.to_numpy()))
    lo, hi, skipped = interval(beta_draws)
    common = {"spec": spec.name, "part": spec.part, "outcome": spec.outcome, "subset": subset, "restriction": spec.restriction,
              "controls": spec.controls, "n": len(f), "matches": int(f.match.nunique()),
              "on_target_shots": int(f.on_target.sum()), "skipped_draws": skipped}
    return [common | {"regressor": name, "of_interest": name in spec.regressors or (not spec.regressors and name == "intercept"),
                      "coef": float(beta[i]), "lo": float(lo[i]), "hi": float(hi[i])} for i, name in enumerate(names)]


def timing_residual_arm(frame: pd.DataFrame, draws: TableDraws, subsets: dict[str, np.ndarray], table_kind: str,
                        verdict: TimingVerdict | None, development_override: bool = False) -> tuple[list[dict], str]:
    """Coefficient rows of the residual specs and the reason they may run; raises `TimingVerdictError` unless the
    verdict is "independent" (or the development override applies to a development table)."""
    reason = residual_arm_permission(verdict, table_kind, development_override)
    rows = []
    for subset, mask in subsets.items():
        rows += [r for spec in SPECS if spec.part == "residual" for r in run_spec(frame[mask[frame.row]], spec, subset, draws)]
    return rows, reason


@dataclass
class LeakageResults:
    """Coefficient rows of every spec and subset, shot counts per subset, and whether the residual arm ran and why."""

    coefficients: pd.DataFrame
    counts: dict
    residual_arm: dict    # {"ran": bool, "reason": str}


def leakage_table(table: pd.DataFrame, draws: TableDraws, *, offset_logits: str = "calibrated", subsets: dict[str, np.ndarray] | None = None,
                  verdict: TimingVerdict | None = None, development_override: bool = False) -> LeakageResults:
    """Every part of H4 (module docstring). The residual arm is skipped, with the reason, when the verdict does not
    allow it; asking for the development override on a confirmatory table raises."""
    kinds = set(table.table_kind)
    if len(kinds) != 1:
        raise ValueError("the table mixes table kinds")
    kind = kinds.pop()
    at0, atm400 = OFFSET_LOGITS[offset_logits]
    if table[[at0, atm400]].isna().all().any():
        raise ValueError(f"the {offset_logits} offset logits ({at0}, {atm400}) are unavailable in this table")
    frame = leakage_frame(table, offset_logits)
    subsets = subsets or {"all": np.ones(len(table), bool)}
    rows, counts = [], {}
    for subset, mask in subsets.items():
        f = frame[mask[frame.row]]
        rows += [r for spec in SPECS if spec.part != "residual" for r in run_spec(f, spec, subset, draws)]
        counts[subset] = {"non_goal_shots": len(f), "on_target": int(f.on_target.sum()), "matches": int(f.match.nunique()),
                          "with_offset_contrast": int(np.isfinite(f.offset_contrast).sum()),
                          "abs_residual_le_200ms": int((f.residual_ms.abs() <= WITHIN_MS).sum()),
                          "late_gt_40ms": int(f.late.sum()), "early_lt_minus_40ms": int(f.early.sum()),
                          "late_gt_200ms": int((f.residual_ms > WITHIN_MS).sum())}
    try:
        arm_rows, reason = timing_residual_arm(frame, draws, subsets, kind, verdict, development_override)
        rows += arm_rows
        arm = {"ran": True, "reason": reason}
    except TimingVerdictError as refusal:
        if development_override:
            raise
        arm = {"ran": False, "reason": str(refusal)}
    return LeakageResults(pd.DataFrame(rows), counts, arm)
