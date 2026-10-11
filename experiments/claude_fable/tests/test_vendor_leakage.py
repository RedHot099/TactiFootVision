"""H4 leakage diagnostic and the timing-verdict gate (xg-vendor ticket 09, seam 2): planted effects are recovered,
the residual arm refuses to run without an allowing verdict."""

import json

import numpy as np
import pandas as pd
import pytest
from xgpoc.eval.calibration import logit, sigmoid
from xgpoc.vendor import leakage as lk
from xgpoc.vendor import timing_verdict as tv
from xgpoc.vendor.draws import table_draws

RULE = "c" * 64
ON_TARGET_EFFECT, RESIDUAL_SLOPE, OFFSET_EFFECT = 0.5, 0.8, 0.4


def make_table(n_matches=40, per_match=30, kind="development", seed=0) -> pd.DataFrame:
    """Label-joined rows with planted effects: gap = 0.5 on_target + 0.3 vendor logit + 0.2 header + 0.8 residual_s + noise;
    offset contrast = 0.1 + 0.4 on_target + noise. Goals, a penalty and an unscored frame carry absurd values and must drop out."""
    rng = np.random.default_rng(seed)
    n = n_matches * per_match
    on_target = rng.random(n) < 0.45
    head = rng.random(n) < 0.15
    vendor_logit = rng.normal(-2.2, 0.8, n)
    residual_ms = np.where(rng.random(n) < 0.9, rng.normal(0, 60, n), rng.normal(0, 700, n))
    gap = ON_TARGET_EFFECT * on_target + 0.3 * vendor_logit + 0.2 * head + RESIDUAL_SLOPE * np.clip(residual_ms, -1000, 1000) / 1000 + 0.3 * rng.normal(size=n)
    raw0 = rng.normal(-2, 1, n)
    contrast = 0.1 + OFFSET_EFFECT * on_target + 0.2 * rng.normal(size=n)
    table = pd.DataFrame({
        "key": [f"s{i}" for i in range(n)], "match": np.repeat([f"g{i:02d}" for i in range(n_matches)], per_match),
        "population": "primary", "split": np.where(np.arange(n) < n // 4, "test", "train"),
        "sb_body_part": np.where(head, "Head", np.where(rng.random(n) < 0.6, "Right Foot", "Left Foot")),
        "sb_shot_type": np.where(rng.random(n) < 0.05, "Free Kick", "Open Play"),
        "pred_vendor_xg": sigmoid(vendor_logit), "frame_logit_0": vendor_logit + gap, "frame_logit_m400": np.nan,
        "frame_raw_logit_0": raw0, "frame_raw_logit_m400": raw0 - contrast, "frame_status": "ok",
        "timing_residual_ms": residual_ms, "table_kind": kind,
        "goal_sn": False, "on_target_sn": pd.array(on_target, dtype="boolean"), "is_penalty": False})
    absurd = [0, 1, 2]
    table.loc[absurd, "frame_logit_0"] = 50.0
    table.loc[absurd, "frame_raw_logit_0"] = 50.0
    table.loc[0, "goal_sn"] = True
    table.loc[1, "is_penalty"] = True
    table.loc[2, "frame_status"] = "missing_frame"
    return table


def verdict(name: str) -> tv.TimingVerdict:
    return tv.parse_timing_verdict({"schema": tv.SCHEMA, "verdict": name, "rule_sha256": RULE, "decided_at": "2026-10-20T10:00:00+00:00"})


def coefficient(coefs: pd.DataFrame, spec: str, regressor: str, subset: str = "all") -> pd.Series:
    row = coefs[coefs.spec.eq(spec) & coefs.regressor.eq(regressor) & coefs.subset.eq(subset)]
    assert len(row) == 1
    return row.iloc[0]


def test_planted_on_target_coefficient_is_recovered():
    table = make_table()
    res = lk.leakage_table(table, table_draws(table, n_boot=300), offset_logits="raw")
    c = coefficient(res.coefficients, "gap_on_target", "on_target")
    assert c.coef == pytest.approx(ON_TARGET_EFFECT, abs=0.05)
    assert c.lo < ON_TARGET_EFFECT < c.hi and c.lo > 0.3
    assert c.n == len(table) - 3                                         # goal, penalty and unscored frame dropped
    assert coefficient(res.coefficients, "gap_on_target", "vendor_logit").coef == pytest.approx(0.3, abs=0.05)
    assert coefficient(res.coefficients, "gap_on_target", "sb_body_part=Head").coef == pytest.approx(0.2, abs=0.1)
    regressors = set(res.coefficients[res.coefficients.spec.eq("gap_on_target_unadjusted")].regressor)
    assert regressors == {"intercept", "on_target"}
    assert res.counts["all"]["non_goal_shots"] == len(table) - 3


def test_offset_contrast_recovers_planted_difference():
    table = make_table()
    res = lk.leakage_table(table, table_draws(table, n_boot=200), offset_logits="raw")
    assert coefficient(res.coefficients, "offset_on_target_unadjusted", "on_target").coef == pytest.approx(OFFSET_EFFECT, abs=0.04)
    assert coefficient(res.coefficients, "offset_on_target", "on_target").coef == pytest.approx(OFFSET_EFFECT, abs=0.04)
    kept = table.iloc[3:]
    mean = coefficient(res.coefficients, "offset_mean", "intercept")
    assert mean.of_interest and mean.coef == pytest.approx((kept.frame_raw_logit_0 - kept.frame_raw_logit_m400).mean())


def test_unavailable_offset_logits_are_refused():
    table = make_table()
    with pytest.raises(ValueError, match="unavailable"):
        lk.leakage_table(table, table_draws(table, n_boot=10), offset_logits="calibrated")


def test_residual_arm_refuses_without_an_allowing_verdict():
    table = make_table()
    draws = table_draws(table, n_boot=50)
    frame = lk.leakage_frame(table, "raw")
    subsets = {"all": np.ones(len(table), bool)}
    for refused in (None, verdict("shared"), verdict("unresolved")):
        with pytest.raises(tv.TimingVerdictError, match="needs the verdict 'independent'"):
            lk.timing_residual_arm(frame, draws, subsets, "development", refused)
        res = lk.leakage_table(table, draws, offset_logits="raw", verdict=refused)
        assert not res.residual_arm["ran"] and "independent" in res.residual_arm["reason"]
        assert not res.coefficients.part.eq("residual").any()
    rows, reason = lk.timing_residual_arm(frame, draws, subsets, "confirmatory", verdict("independent"))
    assert rows and "independent" in reason


def test_development_override_only_on_development_tables():
    draws_table = make_table()
    draws = table_draws(draws_table, n_boot=50)
    res = lk.leakage_table(draws_table, draws, offset_logits="raw", verdict=verdict("shared"), development_override=True)
    assert res.residual_arm["ran"] and "development override" in res.residual_arm["reason"]
    confirmatory = make_table(kind="confirmatory")
    with pytest.raises(tv.TimingVerdictError, match="confirmatory"):
        lk.leakage_table(confirmatory, draws, offset_logits="raw", development_override=True)


def test_residual_slope_recovered_and_restricted():
    table = make_table()
    res = lk.leakage_table(table, table_draws(table, n_boot=300), offset_logits="raw", verdict=verdict("independent"))
    full = coefficient(res.coefficients, "gap_residual_slope", "residual_s")
    assert full.coef == pytest.approx(RESIDUAL_SLOPE, abs=0.15) and full.lo > 0
    within = coefficient(res.coefficients, "gap_residual_slope_within_200ms", "residual_s")
    assert within.restriction == "abs_residual_le_200ms"
    kept = table.iloc[3:]
    assert within.n == int((kept.timing_residual_ms.abs() <= 200).sum()) < full.n
    assert within.hi - within.lo > full.hi - full.lo                   # the restriction leaves a narrow residual range
    early_late = res.coefficients[res.coefficients.spec.eq("gap_early_late")]
    assert {"early", "late"} <= set(early_late.regressor)


def test_subsets_use_their_own_rows():
    table = make_table()
    subsets = {"all": np.ones(len(table), bool), "test": table.split.eq("test").to_numpy()}
    res = lk.leakage_table(table, table_draws(table, n_boot=50), offset_logits="raw", subsets=subsets)
    assert coefficient(res.coefficients, "gap_on_target", "on_target", "test").n == int(table.split.eq("test").sum()) - 3
    assert res.counts["test"]["non_goal_shots"] < res.counts["all"]["non_goal_shots"]


def test_bootstrap_ols_draws_match_row_duplication():
    rng = np.random.default_rng(4)
    x = np.column_stack([np.ones(8), rng.normal(size=8), np.r_[np.ones(2), np.zeros(6)]])
    y = rng.normal(size=8)
    w = np.array([[1, 2, 0, 0, 1, 3, 1, 1], [0, 0, 1, 2, 1, 1, 2, 1]], float)   # the second draw loses the indicator
    _, draws = lk.ols_with_draws(y, x, w)
    dup = np.repeat(np.arange(8), w[0].astype(int))
    assert np.allclose(draws[0], np.linalg.lstsq(x[dup], y[dup], rcond=None)[0])
    dup = np.repeat(np.arange(8), w[1].astype(int))
    assert np.allclose(draws[1], np.linalg.lstsq(x[dup], y[dup], rcond=None)[0])
    assert draws[1, 2] == pytest.approx(0.0, abs=1e-9)


def test_leakage_needs_labels():
    table = make_table().drop(columns=["goal_sn"])
    with pytest.raises(ValueError, match="labels"):
        lk.leakage_frame(table)


# ------------------------------------------------------------------ verdict record
def test_verdict_record_validation(tmp_path):
    good = {"schema": tv.SCHEMA, "verdict": "independent", "rule_sha256": RULE, "statistics": {"substitutions_within_200ms": 0.12}}
    path = tmp_path / "verdict.json"
    path.write_text(json.dumps(good))
    v = tv.load_timing_verdict(path, expected_rule_sha256=RULE)
    assert v.verdict == "independent" and v.record["statistics"] == good["statistics"]
    for broken, message in (({**good, "verdict": "maybe"}, "unknown"), ({**good, "rule_sha256": "abc"}, "sha256"),
                            ({**good, "schema": "other"}, "schema"), ({k: good[k] for k in ("schema", "verdict")}, "lacks")):
        with pytest.raises(tv.TimingVerdictError, match=message):
            tv.parse_timing_verdict(broken)
    with pytest.raises(tv.TimingVerdictError, match="another rule"):
        tv.parse_timing_verdict(good, expected_rule_sha256="d" * 64)


def test_vendor_logit_is_clipped():
    table = make_table()
    table.loc[5, "pred_vendor_xg"] = 0.0
    frame = lk.leakage_frame(table, "raw")
    assert frame.loc[frame.row.eq(5), "vendor_logit"].iloc[0] == pytest.approx(logit(np.array([0.0]))[0])
