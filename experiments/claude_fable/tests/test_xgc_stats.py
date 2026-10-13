"""The xg-compare analysis statistics (ticket 02): against direct computations, on synthetic data with planted
effects, and on permuted labels."""

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from xgpoc.compare import analysis as an
from xgpoc.compare import categories as cat
from xgpoc.compare import describe as ds
from xgpoc.compare import pipeline as pl
from xgpoc.compare import population as po
from xgpoc.compare import power as pw
from xgpoc.compare import protocol as pr
from xgpoc.compare import stats as st
from xgpoc.compare import table as tb
from xgpoc.eval.calibration import logit, sigmoid
from xgpoc.vendor import ladder as ld
from xgpoc.vendor import refit

N_BOOT = 200


def repeated(weights_row: np.ndarray, *arrays: np.ndarray) -> list[np.ndarray]:
    """The arrays with every shot repeated as often as its weight (the resampled data set)."""
    repeat = weights_row.astype(int)
    return [np.repeat(a, repeat, axis=0) for a in arrays]


def synthetic(n_matches: int = 60, per_match: int = 25, seed: int = 0, header: float = 0.0, saved: float = 0.0,
              n_boot: int = N_BOOT) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    """A shot table, its label file and the shared draws. The truth is t ~ N(-2, 1) plus a match effect; V, P, E and
    D read t with noise; F reads t with noise plus `header` logit on headers and `saved` logit on saved non-goal
    shots from outside the box. Categories are drawn at random, so no other level has an effect."""
    rng = np.random.default_rng(seed)
    n = n_matches * per_match
    match_id = np.repeat(np.arange(n_matches), per_match)
    truth = rng.normal(-2, 1, n) + rng.normal(0, 0.2, n_matches)[match_id]
    goal = rng.random(n) < sigmoid(truth)
    table = pd.DataFrame({"key": [f"s{i}" for i in range(n)], "match": [f"m{m:03d}" for m in match_id]})
    table["soccernet_shot_id"], table["statsbomb_event_id"] = "sn_" + table.key, "sb_" + table.key
    table["in_primary"] = True
    table["scan_half"] = np.where(match_id % 2 == 0, "discovery", "validation")
    for c in cat.PRE_SHOT:
        levels = np.array(c.levels, dtype=object)
        table[c.column] = rng.choice(levels, n, p=np.r_[0.25, np.full(len(levels) - 1, 0.75 / (len(levels) - 1))])
    table[cat.BY_NAME["half"].column] = rng.choice(np.array(["first", "second"], dtype=object), n)
    table["sb_body"] = table.cat_body
    table["sb_stype"] = table.cat_shot_type
    outcome = np.where(goal, "Goal", rng.choice(np.array(["Saved", "Off T", "Blocked", "Wayward"], dtype=object), n))
    labels = pd.DataFrame({"key": table.key, "goal_sn": goal, "on_target_sn": pd.array(np.isin(outcome, ["Goal", "Saved", "Blocked"]), "boolean"),
                           "sb_outcome": outcome})
    is_saved = (outcome == "Saved") & (table.cat_outside_box.to_numpy() == "yes")
    z = {"vendor": truth + rng.normal(0, 0.3, n), "positions": truth + rng.normal(0, 0.3, n), "events": truth + rng.normal(0, 0.5, n),
         "dist_angle": 0.7 * truth + rng.normal(0, 0.6, n),
         "frame": truth + rng.normal(0, 0.4, n) + header * (table.cat_body.to_numpy() == "Head") + saved * is_saved}
    z["frame_m400"] = z["frame"] - saved * is_saved + rng.normal(0, 0.1, n)
    for name, values in z.items():
        table[f"z_{name}"] = values
    table["z_a1"] = np.nan
    return table, labels, ld.bootstrap(table.match.to_numpy(), n_boot)


def population(table, labels, draws, name="primary"):
    return po.build_population(table, labels, np.ones(len(table), bool), draws, name, "primary")


@pytest.fixture(scope="module")
def planted():
    return population(*synthetic(header=0.6, saved=0.8))


# ------------------------------------------------------------------ the main scale
def test_main_scale_is_the_paper_crossfit_in_row_0_and_the_per_draw_refit_after():
    table, labels, draws = synthetic(n_matches=30, per_match=20, n_boot=20)
    pop = population(table, labels, draws)
    z, y = pop.raw["V"], pop.y
    oof, _ = ld.crossfit(z, y, pop.folds)
    np.testing.assert_allclose(sigmoid(pop.z("V")[0]), oof, atol=1e-10)
    _, start = ld.crossfit(z, y, pop.folds)
    expected = refit.crossfit_draws(z, y, pop.folds, pop.weights[1:], start)
    np.testing.assert_allclose(sigmoid(pop.z("V")[1:]), expected, atol=1e-10)


def test_paper_coefficients_match_refuses_a_different_recalibration():
    pop = population(*synthetic(n_matches=30, per_match=20, n_boot=5))
    paper = {rung: pop.coefficients[m][0].round(6).tolist() for m, rung in po.PAPER_RUNGS.items()}
    assert max(po.paper_coefficients_match(pop, paper).values()) < 1e-6
    paper["vendor"][0][1] += 0.01
    with pytest.raises(ValueError, match="departs"):
        po.paper_coefficients_match(pop, paper)


def test_a_method_with_some_missing_logits_is_refused_and_one_missing_everywhere_is_left_out():
    table, labels, draws = synthetic(n_matches=10, per_match=10, n_boot=5)
    assert "A1" not in population(table, labels, draws).methods
    table.loc[:49, "z_a1"] = 0.0                                                 # A1 on part of the rows: left out
    assert "A1" not in population(table, labels, draws).methods
    table.loc[3, "z_events"] = np.nan
    with pytest.raises(ValueError, match="all or none"):
        population(table, labels, draws)


def test_percentile_ranks_are_midrank_percentiles_of_the_resampled_shots():
    rng = np.random.default_rng(1)
    x = rng.normal(size=50)
    x[4] = x[9]
    weights = np.vstack([np.ones(50), rng.integers(0, 3, 50)]).astype(float)
    got = st.percentile_ranks(x, weights)
    np.testing.assert_allclose(got[0], 100 * (rankdata(x) - 0.5) / 50)
    assert np.average(got[1], weights=weights[1]) == pytest.approx(50.0)


# ------------------------------------------------------------------ effects and OLS
def test_level_effects_equal_least_squares_on_the_resampled_rows_with_per_draw_outcome_and_control():
    rng = np.random.default_rng(3)
    n, w_rows = 200, 4
    weights = rng.integers(0, 3, (w_rows, n)).astype(float)
    weights[0] = 1
    y, control = rng.normal(size=(w_rows, n)), rng.normal(size=(w_rows, n))
    inside = rng.random((n, 2)) < 0.3
    defined = np.column_stack([np.ones(n, bool), rng.random(n) < 0.8])
    adjusted, unadjusted = st.level_effects(y, control, inside, defined, weights)
    for level in range(2):
        for row in range(w_rows):
            keep = defined[:, level]
            x = np.column_stack([np.ones(keep.sum()), inside[keep, level], control[row, keep]])
            xr, yr = repeated(weights[row, keep], x, y[row, keep])
            assert adjusted[row, level] == pytest.approx(np.linalg.lstsq(xr, yr, rcond=None)[0][1], abs=1e-9)
            i = xr[:, 1] == 1
            assert unadjusted[row, level] == pytest.approx(yr[i].mean() - yr[~i].mean(), abs=1e-12)


def test_level_effects_are_nan_without_weight_inside():
    inside = np.zeros((10, 1), bool)
    inside[:3] = True
    weights = np.ones((2, 10))
    weights[1, :3] = 0
    adjusted, unadjusted = st.level_effects(np.arange(10.0), np.ones(10) * np.arange(10) % 3, inside, np.ones((10, 1), bool), weights)
    assert np.isfinite(adjusted[0, 0]) and np.isnan(adjusted[1, 0]) and np.isnan(unadjusted[1, 0])


def test_level_effects_are_nan_for_a_singular_fit():
    inside = np.zeros((10, 1), bool)
    inside[:5] = True
    control = inside[:, 0].astype(float)                                       # collinear with the indicator
    adjusted, unadjusted = st.level_effects(np.arange(10.0), control, inside, np.ones((10, 1), bool), np.ones((1, 10)))
    assert np.isnan(adjusted[0, 0]) and unadjusted[0, 0] == pytest.approx(2.0 - 7.0)


def test_ols_with_one_design_for_every_row_equals_the_per_row_design():
    rng = np.random.default_rng(5)
    x = np.column_stack([np.ones(60), rng.normal(size=60)])
    weights = np.vstack([np.ones(60), rng.integers(0, 3, 60)]).astype(float)
    y = rng.normal(size=60)
    np.testing.assert_allclose(st.ols(y, x, weights), st.ols(y, np.broadcast_to(x, (2, 60, 2)), weights), atol=1e-12)


def test_ols_equals_least_squares_on_the_resampled_rows():
    rng = np.random.default_rng(4)
    n = 120
    weights = np.vstack([np.ones(n), rng.integers(0, 3, (2, n))]).astype(float)
    x = np.concatenate([np.ones((3, n, 1)), rng.normal(size=(3, n, 2))], axis=2)
    y = rng.normal(size=(3, n))
    beta = st.ols(y, x, weights)
    for row in range(3):
        xr, yr = repeated(weights[row], x[row], y[row])
        np.testing.assert_allclose(beta[row], np.linalg.lstsq(xr, yr, rcond=None)[0], atol=1e-10)


# ------------------------------------------------------------------ summaries and multiplicity
def test_summarise_gives_the_point_the_interval_and_the_protocol_p_value():
    stacked = np.r_[0.3, np.full(90, 0.2), np.full(10, -0.1)]
    s = st.summarise(stacked, "greater")
    assert s["point"] == 0.3 and s["p"] == pytest.approx(0.10) and s["nan_draws"] == 0
    assert st.summarise(stacked, "two-sided")["p"] == pytest.approx(0.20)
    with_nan = st.summarise(np.r_[0.3, np.nan, np.full(99, 0.2)], "less")
    assert with_nan["nan_draws"] == 1 and np.isnan(with_nan["p"]) and with_nan["lo"] == pytest.approx(0.2)


def test_q_values_reject_exactly_what_bh_and_by_reject():
    rng = np.random.default_rng(6)
    for _ in range(200):
        p = rng.random(14) ** 3
        p[rng.random(14) < 0.2] = np.nan
        assert np.array_equal(st.q_values(p) <= 0.05, pw.bh(p)), p
        assert np.array_equal(st.q_values(p, dependent=True) <= 0.05, pw.bh(p, dependent=True)), p


def test_q_values_on_a_known_example():
    p = np.array([0.01, 0.04, 0.03, 0.5])
    np.testing.assert_allclose(st.q_values(p), [0.04, 0.16 / 3, 0.16 / 3, 0.5])
    by = st.q_values(p, dependent=True)
    assert np.all(by >= st.q_values(p)) and by[0] == pytest.approx(0.04 * (1 + 1 / 2 + 1 / 3 + 1 / 4))


def test_xg_points_at_the_category_mean():
    assert st.xg_points(0.0, 0.2) == 0.0
    assert st.xg_points(np.log(2), 0.2) == pytest.approx(100 * (sigmoid(logit(0.2) + np.log(2)) - 0.2))
    assert st.xg_points(0.15, 0.1) == pytest.approx(1.433, abs=0.001)


# ------------------------------------------------------------------ label metrics and correlations
def test_label_metrics_equal_direct_computations_on_the_resampled_shots():
    rng = np.random.default_rng(7)
    n = 300
    y = (rng.random(n) < 0.2).astype(float)
    p = np.clip(rng.random((3, n)) * 0.5, 0.01, 0.99)
    p[:, :5] = p[0, 5]                                      # ties
    weights = np.vstack([np.ones(n), rng.integers(0, 3, (2, n))]).astype(float)
    metrics = st.label_metrics(y, p, weights)
    for row in range(3):
        yr, pr_ = repeated(weights[row], y, p[row])
        assert metrics["observed"][row] == yr.sum() and metrics["expected"][row] == pytest.approx(pr_.sum())
        assert metrics["o_over_e"][row] == pytest.approx(yr.sum() / pr_.sum())
        assert metrics["log_loss"][row] == pytest.approx(-np.mean(yr * np.log(pr_) + (1 - yr) * np.log(1 - pr_)))
        assert metrics["auc"][row] == pytest.approx(roc_auc_score(yr, pr_))


def test_label_metrics_auc_is_nan_without_a_goal():
    metrics = st.label_metrics(np.zeros(5), np.full(5, 0.1), np.ones((1, 5)))
    assert np.isnan(metrics["auc"][0]) and metrics["o_over_e"][0] == 0.0


def test_correlation_difference_is_bootstrapped_jointly_and_finds_a_planted_difference():
    rng = np.random.default_rng(8)
    n_matches, per = 50, 30
    match = np.repeat(np.arange(n_matches), per)
    d = rng.normal(size=len(match))
    f = d + rng.normal(0, 0.5, len(match))                  # F tracks D closely
    v = d + rng.normal(0, 1.5, len(match))                  # and V loosely
    counts = st.stacked_counts(ld.bootstrap(np.arange(n_matches), 300))
    w = counts[:, match]
    diff = pw.spearman(f, d, w) - pw.spearman(f, v, w)
    assert diff[0] == pytest.approx(spearmanr(f, d).statistic - spearmanr(f, v).statistic, abs=1e-12)
    s = st.summarise(diff, "greater")
    assert s["lo"] > 0 and s["p"] < 0.01


# ------------------------------------------------------------------ the family on synthetic data
def test_a_planted_header_effect_is_recovered_and_the_null_items_are_not_rejected(planted):
    fam = an.family(planted).set_index("item")
    header = fam.loc["3"]
    assert header.raw_lo < 0.6 < header.raw_hi                     # the raw gap carries the planted 0.6 exactly
    assert header.effect_point > 0.3 and header.reject_bh and header.reading.startswith("BH-significant")
    nulls = fam.loc[["4", "5", "6a", "7", "11", "12", "14", "15"]]
    assert not nulls.reject_by.any() and nulls.reject_bh.sum() <= 1
    assert (fam.q_by >= fam.q_bh).all()


def test_floors_and_unknown_categories_are_reported(planted):
    fam = an.family(planted).set_index("item")
    assert fam.loc["3"].n_in == int((planted.levels_of("body") == "Head").sum())
    combined = dataclasses.replace(planted, base="combined")
    assert an.family(combined).set_index("item").loc["15"].status == "not defined in the combined population"


def test_item_1_finds_a_planted_on_target_gap_and_the_offset_view_does_not(planted):
    out = an.item_1(planted)
    f, f400 = out["regressions"]["F|raw"], out["regressions"]["F-400|raw"]
    assert f["adjusted_lo"] > 0.5 and f["adjusted_p"] < 0.01
    assert abs(f400["adjusted_point"]) < 0.15
    assert out["n_apart"]["Blocked"] > 0 and "Blocked" in out["group_means"]["F"]


def test_item_13_regresses_the_gap_on_soccernet_on_target_among_non_goals(planted):
    out = an.item_13(planted)
    assert out["n_non_goals"] == int((planted.y == 0).sum())
    assert set(out["regressions"]) == {"F|main", "F-400|main", "F|raw", "F-400|raw"}
    assert out["regressions"]["F|raw"]["adjusted_point"] > 0                    # saved shots carry the planted gap
    assert "adjusted_xg_points_point" in out["regressions"]["F|main"] and "adjusted_xg_points_point" not in out["regressions"]["F|raw"]


def test_item_2_carries_its_3_s_sensitivity_and_xg_points_beside_every_logit_effect(planted):
    row = an.family(planted).set_index("item").loc["2"]
    assert row.n_in_sensitivity == int((planted.levels_of("rebound_3s") == "yes").sum())
    assert np.isfinite([row.sensitivity_point, row.effect_xg_points_point, row.ref_DV_xg_points_point]).all()


def test_level_item_negative_control_stays_within_the_threshold(planted):
    row = an.level_item(planted, *an.NEGATIVE_CONTROL)
    assert row["n_in"] + row["n_out"] == len(planted.y) and row["within_threshold"]


# ------------------------------------------------------------------ permuted labels
def test_permute_within_matches_keeps_each_match_s_labels():
    table, labels, _ = synthetic(n_matches=10, per_match=8, n_boot=1)
    permuted = st.permute_within_matches(labels, table.match.to_numpy(), seed=3)
    assert permuted.key.equals(labels.key) and not permuted.goal_sn.equals(labels.goal_sn)
    for m, rows in table.groupby("match").groups.items():
        assert sorted(permuted.loc[rows, "sb_outcome"]) == sorted(labels.loc[rows, "sb_outcome"]), m


def test_on_permuted_labels_the_family_finds_about_nothing():
    """A planted frame effect lives in the raw logits, but on permuted labels the main scale loses it: the
    recalibration slopes collapse, and refitting them in every draw keeps the p-values honest."""
    table, labels, draws = synthetic(header=0.6, seed=11)
    rejected = []
    for seed in range(12):
        pop = population(table, st.permute_within_matches(labels, table.match.to_numpy(), seed), draws)
        fam = an.family(pop, label_free=False)
        assert set(fam[fam.status.ne("run")].item) == {"8", "9"}
        rejected.append(int(fam.reject_bh.sum()))
    assert sum(r > 0 for r in rejected) <= 2


# ------------------------------------------------------------------ the scan
def test_scan_freezes_candidates_and_validates_only_the_frozen_file(tmp_path, planted):
    table, labels, draws = synthetic(header=0.6, saved=0.8)
    halves = {h: (lambda h=h: po.build_population(table, labels, table.scan_half.eq(h).to_numpy(), draws, h, "primary")) for h in pr.HALVES}
    coverage = tb.coverage(table)
    discovery = an.scan_discovery(halves["discovery"](), coverage)
    head = discovery.set_index(["category", "level"]).loc[("body", "Head")]
    assert head.candidate and head.point > 0.2
    path = tmp_path / "candidates.json"
    sha = an.freeze_candidates(discovery, path, {"population": "synthetic"})
    with pytest.raises(ValueError, match="frozen once"):
        an.freeze_candidates(discovery, path, {})
    called = []
    with pytest.raises(ValueError, match="not the frozen"):
        an.scan_validation(lambda: called.append(1), path, "0" * 64)
    assert not called
    validation = an.scan_validation(halves["validation"], path, sha).set_index(["category", "level"])
    assert validation.loc[("body", "Head")].replicated and validation.loc[("body", "Head")].direction == "greater"
    assert len(validation) == len(json.loads(path.read_text())["candidates"])


# ------------------------------------------------------------------ the descriptive part
def test_level_table_and_who_is_right_on_synthetic_data(planted):
    levels = ds.level_table(planted)
    fv = levels[levels.pair.eq("F-V")].set_index(["category", "level"])
    assert fv.loc[("body", "Head")].raw_unadjusted_lo < 0.6 < fv.loc[("body", "Head")].raw_unadjusted_hi
    assert set(levels[levels.post_contact].pair) == {"F-V", "F-400-V"}
    right = ds.who_is_right(planted)
    overall = right[right.category.eq("overall")].set_index("method")
    assert overall.loc["V"].o_over_e_point == pytest.approx(1.0, abs=0.05)     # recalibrated: E close to O
    assert np.isnan(overall.loc["V"].get("delta_ll_vs_V_point", np.nan))
    assert overall.loc["F"].delta_ll_vs_V_point > 0                              # F is the noisier reader
    post = right[right.role.eq("post_shot")]
    assert post.filter(like="o_over_e").isna().all().all()


def test_agreement_matches_scipy_on_the_observed_sample(planted):
    corr, differences = ds.agreement(planted)
    row = corr[corr.category.eq("overall") & corr.kind.eq("spearman") & corr.a.eq("V") & corr.b.eq("F")].iloc[0]
    assert row.point == pytest.approx(spearmanr(planted.raw["V"], planted.raw["F"]).statistic, abs=1e-12)
    assert set(differences.difference) == {"rho(F,V) - rho(P,V)", "rho(F,D) - rho(F,V)"}


def test_multivariable_raw_levels_and_the_descriptive_scan(planted):
    fit = ds.multivariable(planted).set_index("term")
    assert fit.loc["item 3: body=Head"].lo > 0 and fit.loc["item 3: body=Head"].point > 0.3      # the planted header gap
    raw = ds.raw_levels(planted).set_index("method")
    assert raw.loc["V"].mean_p_main == pytest.approx(planted.y.mean(), abs=0.01) and raw.loc["V"].ll_main_minus_V == 0
    descriptive = pl.scan_descriptive(ds.level_table(planted))
    assert set(descriptive.pair) == {"F-D", "F-P"} and len(descriptive) == 2 * len(pr.scan_levels())


def test_disagreement_tails_are_five_percent_each_way(planted):
    shots, tails, profile = ds.disagreements(planted)
    n_tail = round(0.05 * len(planted.y))
    assert (shots.direction.value_counts() == n_tail).all()
    above = shots[shots.direction.eq("F_above_V")].gap_F_minus_V
    below = shots[shots.direction.eq("F_below_V")].gap_F_minus_V
    assert above.min() > below.max()
    lift = profile[profile.direction.eq("F_above_V") & profile.category.eq("body") & profile.level.eq("Head")].lift.iloc[0]
    assert lift > 1.5                                                            # the planted header effect
    assert set(tails.direction) == {"F_above_V", "F_below_V", "population"}


# ------------------------------------------------------------------ the whole run
@pytest.mark.parametrize("dry_run", [True, False])
def test_the_pipeline_runs_end_to_end_and_the_dry_run_reads_nothing_label_free(tmp_path, dry_run):
    table, labels, draws = synthetic(header=0.6, n_boot=50)
    table["in_primary"] = np.arange(len(table)) % 5 != 0                         # combined adds non-primary shots
    table["z_a1"] = np.where(table.in_primary, table.z_frame, np.nan)            # A1 on primary shots only
    summary = pl.run(table, labels, tb.coverage(table), draws, tmp_path, dry_run=dry_run, paper_coefficients=None,
                     context={"table_sha256": "t"}, log=lambda _: None)
    written = {p.name for p in tmp_path.iterdir()}
    label_free = {"agreement.csv", "agreement_differences.csv", "disagreement_shots.csv"}
    assert {"family.csv", "outside_family.json", "levels.csv", "who_is_right.csv", pl.CANDIDATES, "scan_validation.csv"} <= written
    assert (label_free <= written) != dry_run
    frozen = json.loads((tmp_path / pl.CANDIDATES).read_text())
    assert frozen["dry_run"] == dry_run and frozen["table_sha256"] == "t"
    assert summary["notice"] == "Data: StatsBomb." and ("agreement_overall" in summary) != dry_run
    assert "A1" in summary["who_is_right_overall"]
    if not dry_run:
        assert summary["family"]["3"]["reading"].startswith("BH-significant")
