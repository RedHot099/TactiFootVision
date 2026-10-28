"""The xg-dino-info protocol (ticket 11, C3) at its entry points on synthetic populations: the redundancy contract and
its verdicts, Holm families, the Xa allowlist and the label bar, leak estimands, subgroups, the masking test, frozen
fold models, region overlap, probe targets, the control truths and the run script's code identity."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import xgdino_info_protocol as script  # noqa: E402
from xgpoc.compare.stats import permute_within_matches  # noqa: E402
from xgpoc.dino import ablation as ab  # noqa: E402
from xgpoc.dino import attributes as at  # noqa: E402
from xgpoc.dino import info_controls as ic  # noqa: E402
from xgpoc.dino import info_protocol as ip  # noqa: E402
from xgpoc.dino import regions as rg  # noqa: E402
from xgpoc.eval.calibration import sigmoid  # noqa: E402
from xgpoc.stack import fit as sf  # noqa: E402
from xgpoc.stack import stats as ss  # noqa: E402
from xgs_fixtures import standardised_distance, synthetic_table  # noqa: E402

DRAWS = 200
BINARY = {"ball_ok", "ball_fallback", "shot_from_ball", "ball_own", "ball_above", *at.FLAG_WINDOWS, "sn_team_unknown"}


def synthetic_attributes(table: pd.DataFrame, seed: int = 0, signal: np.ndarray | None = None) -> pd.DataFrame:
    """Every Xa column as noise of its kind (ball geometry and motion partly missing); with `signal`, box_players@0
    carries it."""
    rng = np.random.default_rng(seed)
    n = len(table)
    out = {}
    gaps = {o: rng.random(n) < 0.3 for o in ("0", "-400", "motion")}
    for c in at.COLUMNS:
        name = c.split("@")[0]
        out[c] = (rng.random(n) < 0.2).astype(float) if name in BINARY else rng.normal(0, 1, n)
        if name in ("ball_above", "ball_height_rel", "ball_dist_rel"):
            out[c][gaps[c.split("@")[1]]] = np.nan
        elif name.startswith("motion"):
            out[c][gaps["motion"]] = np.nan
    out["sn_team_unknown"] = np.zeros(n)
    frame = pd.DataFrame(out, index=table.index)
    if signal is not None:
        frame["box_players@0"] = signal
    return frame


def population(seed: int = 50, n_matches: int = 100, per_match: int = 30, carried: bool = True):
    """A table where z holds an image-specific signal h; with `carried`, box_players@0 = h + small noise."""
    table = synthetic_table(n_matches=n_matches, per_match=per_match, seed=seed)
    rng = np.random.default_rng(seed + 1)
    h = rng.normal(0, 1, len(table))
    table[sf.Z] = -2.0 + h + 0.1 * rng.normal(0, 1, len(table))
    signal = h + 0.05 * rng.normal(0, 1, len(table)) if carried else None
    attributes = synthetic_attributes(table, seed + 2, signal)
    y = (rng.random(len(table)) < sigmoid(-1.8 + 1.2 * h + 0.8 * standardised_distance(table))).astype(int)
    return table, attributes, y


def subsets(table: pd.DataFrame) -> dict[str, np.ndarray]:
    games = table.game.to_numpy()
    first = np.isin(games, np.unique(games)[:25])
    return {"no_v3": first, "unseen": np.random.default_rng(3).random(len(table)) < 0.5}


@pytest.fixture(scope="module")
def carried_run():
    table, attributes, y = population()
    design, columns = ip.design_table(table, attributes)
    prepared = ip.prepare(design, columns, subsets(table), blocks=("Xc", *ip.FAMILY), draws=DRAWS)
    p, _ = ip.fit_arms(prepared, y)
    return design, prepared, y, p, ip.redundancy(y, p, prepared.block.weights, prepared.masks)


def test_verdicts_follow_c32():
    assert ip.verdict(True, -0.001, 0.0001, 0.002, 0.004) == "explains most"
    assert ip.verdict(True, -0.001, -0.0001, 0.0001, 0.004) == "explains part"
    assert ip.verdict(False, -0.001, 0.001, 0.002, 0.004) == "inconclusive"      # the gain must be Holm-rejected
    assert ip.verdict(True, 0.0001, 0.001, 0.002, 0.004) == "inconclusive"       # ... and its upper bound below 0
    assert ip.verdict(False, 0.001, -0.01, -0.01, -0.0001) == "does not explain"
    assert ip.verdict(True, -0.001, -0.01, -0.0001, 0.0001) == "inconclusive"


def test_an_attribute_carrying_the_image_signal_explains_most_and_noise_does_not_explain(carried_run):
    _, _, _, _, result = carried_run
    blocks = result["blocks"]
    assert blocks["Xc"]["G"]["lo95"] > 0
    assert blocks["crowding"]["verdict"] == "explains most" and blocks["Xc"]["verdict"] == "explains most"
    assert blocks["crowding"]["adjustment"] == "Holm over the content family" and blocks["Xc"]["adjustment"] == "single test"
    for noise in ("shooter", "ball_t0", "motion", "flags"):
        assert blocks[noise]["verdict"] == "does not explain", noise
    ratio = blocks["crowding"]["ratio_R_over_G"]
    assert ratio["reportable"] and 0.7 < ratio["point"] < 1.3 and ratio["draws_G_le_0"] == 0
    assert blocks["crowding"]["C-BX"]["wording"] != "DINOv2 adds"
    assert blocks["shooter"]["C-BX"]["wording"] == "DINOv2 adds"
    assert result["wording"] == ip.WORDING


def test_statistics_are_means_of_per_shot_vectors(carried_run):
    _, prepared, y, p, result = carried_run
    ell = ip.losses(y, p)
    r = result["blocks"]["crowding"]
    g = ell["B"].mean() - ell["C"].mean()
    gx = ell["B+crowding"].mean() - ell["C+crowding"].mean()
    assert r["G"]["delta"] == pytest.approx(g) and r["R"]["delta"] == pytest.approx(g - gx)
    assert r["part"]["delta"] == pytest.approx(g - gx - 0.25 * g)
    assert r["gain"]["draws"] == DRAWS


def test_leak_estimands_and_table(carried_run):
    _, prepared, y, _, result = carried_run
    assert set(result["estimands"]) == {"no_v3", "unseen", "unseen_standardised"}
    assert result["estimands"]["no_v3"]["crowding"]["verdict"] in ip.VERDICTS
    leak = result["leak"]["crowding"]
    assert set(leak) == set(ip.LEAK_STATISTICS)
    assert set(leak["R"]["heterogeneity"]) == set(ab.HETEROGENEITY) and leak["R"]["verdict"] in ("decidable", "not decidable")
    # a planted heterogeneity on the no_v3 shots makes R not decidable
    v = {"R": np.where(prepared.masks["no_v3"], 0.05, 0.0), "gain": np.zeros(len(y)), "C-BX": np.zeros(len(y))}
    planted = ip.leak({"X": v}, prepared.block.weights, prepared.masks, {"X": {k: {"delta": 0.01} for k in v}})
    assert planted["X"]["R"]["verdict"] == "not decidable" and "v3-no_v3" in planted["X"]["R"]["heterogeneity_excluding_0"]


def test_permuted_labels_give_no_explains_verdict():
    table, attributes, y = population(seed=60, n_matches=80, per_match=25)
    y = permute_within_matches(pd.DataFrame({"goal": y}), table.game.to_numpy(), 0).goal.to_numpy()
    design, columns = ip.design_table(table, attributes)
    prepared = ip.prepare(design, columns, blocks=("Xc", *ip.FAMILY), draws=DRAWS)
    result = ip.redundancy(y, ip.fit_arms(prepared, y)[0], prepared.block.weights)
    assert not [b for b, r in result["blocks"].items() if r["verdict"] in ("explains most", "explains part")]


def test_holm_family_sensitivity_and_descriptive_blocks():
    def read(p):
        return {"gain": {"p_one_sided": p, "upper_one_sided_95": -1.0}, "most": {"lo90": 1.0},
                "part": {"lo90": 1.0, "upper_one_sided_95": 2.0},
                "R": {}, "G": {}}
    p = {"crowding": 0.001, "shooter": 0.02, "ball_t0": 0.03, "ball_m400": 0.5, "motion": 0.5, "flags": 0.009,
         "flags_le0": 0.0095, "Xc": 0.04, "Xq": 0.04}
    decided = ip.decide({b: read(v) for b, v in p.items()})
    assert decided["crowding"]["gain_rejected"] and decided["flags"]["gain_rejected"]
    assert not decided["shooter"]["gain_rejected"]                      # 0.02 > 0.05 / 4
    assert decided["flags_le0"]["gain_rejected"] and "flags_le0 for flags" in decided["flags_le0"]["adjustment"]
    assert decided["Xc"]["gain_rejected"] and decided["Xq"]["adjustment"] == "unadjusted, descriptive"
    assert decided["crowding"]["verdict"] == "explains most" and decided["shooter"]["verdict"] == "inconclusive"


def test_new_arms_hold_only_allowlisted_columns_and_no_label():
    table, attributes, _ = population(seed=70, n_matches=20, per_match=10)
    design, columns = ip.design_table(table, attributes)
    arms, _ = ip.arm_set(design, columns)
    for name, arm in arms.items():
        if "+" in name:
            assert set(arm.inputs) - {sf.Z} <= set(ab.GEOMETRY) | ip.XA_ALLOWLIST
            assert (sf.Z in arm.inputs) == name.startswith("C+")
    assert set(arms["B"].inputs) <= set(ab.GEOMETRY)
    assert "xa_sn_direct_fk_le0" in arms["B+flags_le0"].inputs and "xa_sn_direct_fk" not in arms["B+flags_le0"].inputs
    assert "xa_sn_team_unknown" not in arms["B+Xc"].inputs
    missing = [c for c in columns["ball_t0"] if "_missing" in c]
    assert len(missing) == 1                                   # identical indicators kept once
    assert not set(columns["Xc"]) & {ip.xa(c) for c in at.BLOCKS["quality"]}
    for planted in ("header_gpt", "kp_instance_conf@0"):
        bad = dict(columns, crowding=[*columns["crowding"], planted])
        frame = design.assign(**{planted: np.random.default_rng(0).random(len(design))})
        with pytest.raises(ValueError):
            ip.arm_set(frame, bad, ("crowding",))
    with pytest.raises(ValueError):
        ip.design_table(table, attributes.assign(header_caption=1.0))


def test_subgroups_weighted_contrast_holm_and_min_support():
    rng = np.random.default_rng(5)
    n = 3000
    games = np.repeat(np.arange(100), 30)
    idx = pd.Index([f"s{i}" for i in range(n)])
    y = (rng.random(n) < 0.15).astype(int)
    set_piece = rng.random(n) < 0.3
    d = np.where(set_piece, -0.05, 0.0) + 0.01 * rng.normal(0, 1, n)
    head = pd.Series(np.where(rng.random(n) < 0.3, 1.0, 0.0), index=idx).where(rng.random(n) < 0.4)
    keys = pd.DataFrame({"set_piece": ip.side(set_piece, np.ones(n, bool)),
                         "header_gpt": ip.side(head.to_numpy() == 1, head.notna().to_numpy()),
                         "tiny": ip.side(np.arange(n) < 50, np.ones(n, bool))}, index=idx)
    weight = pd.Series(np.where(rng.random(n) < 0.5, 3.0, 1.0), index=idx)[head.notna()]
    out = ip.subgroups(y, d, ss.bootstrap_weights(games, DRAWS, 0), keys, weight, 0.13)["keys"]
    assert out["set_piece"]["holm_rejected"] and out["set_piece"]["contrast_in_minus_out"]["delta"] == pytest.approx(-0.05, abs=0.003)
    m = keys.header_gpt.to_numpy() == "in"
    w = weight.reindex(idx).fillna(0).to_numpy()
    assert out["header_gpt"]["sides"]["in"]["mean_d"]["delta"] == pytest.approx(np.average(d[m], weights=w[m]))
    assert out["header_gpt"]["weighted"] and not out["header_gpt"]["holm_rejected"]
    assert not out["tiny"]["sides"]["in"]["readable"] and "contrast_in_minus_out" not in out["tiny"]
    s = out["set_piece"]["sides"]["in"]
    assert s["standardised"] == pytest.approx(0.13 * s["goals"] + 0.87 * s["non_goals"])


def test_tercile_sides_and_timing_bands():
    table = pd.DataFrame({"dist@0": np.arange(9.0)})
    b = ip.tercile_boundaries(table, pd.DataFrame({"box_players@0": np.arange(9.0), "px_per_m@0": np.arange(9.0)}))
    q1, q2 = b["distance"]
    x = table["dist@0"].to_numpy()
    s = ip.side(x > q2, (x <= q1) | (x > q2))
    assert list(s) == ["out", "out", "out", None, None, None, "in", "in", "in"]
    band = ip.timing_band(np.r_[np.zeros(300), np.full(50, 100.0), np.full(150, 300.0), np.nan])
    assert (band == "50to200").sum() == 0 and (band == "le50").sum() == 350 and (band == "gt200").sum() == 150
    assert band[-1] == "missing"


def test_region_test_takes_the_larger_p_and_holm_over_the_family():
    rng = np.random.default_rng(6)
    n = 2000
    games = np.repeat(np.arange(100), 20)
    w = ss.bootstrap_weights(games, DRAWS, 0)
    base = rng.random(n)
    worse = base + 0.05 + 0.01 * rng.normal(0, 1, n)
    on = np.ones(n, bool)
    clear = ip.region_test(base, worse, base, w, on)
    assert clear["p"] == pytest.approx(1 / (DRAWS + 1)) and clear["shots"] == n
    placebo_too = ip.region_test(base, worse, worse + 0.0001 * rng.normal(0, 1, n), w, on)
    assert placebo_too["p"] > 0.05 and placebo_too["vs_unmasked"]["p_degradation"] < 0.01
    tests = {r: clear for r in ip.MASK_FAMILY} | {"goal": placebo_too}
    holm = ip.masking_family(tests)
    assert holm == {r: r != "goal" for r in ip.MASK_FAMILY}
    agree = ip.fill_agreement({"mean": holm, "blur": holm | {"shooter": False}})
    assert agree["shooter"] == "inconclusive (fills disagree)" and agree["goal"] == "no detectable sensitivity"
    assert agree["ball"] == "sensitive to the intervention"


def test_frozen_fold_models_reproduce_out_of_fold_and_read_a_masked_z(carried_run):
    design, prepared, y, p, _ = carried_run
    block = prepared.block
    fits = ip.fold_fits(design, block.arms["C"], block.models["C"], y)
    np.testing.assert_allclose(ip.frozen_predict(fits, design), p["C"], atol=1e-10)
    np.testing.assert_allclose(ip.refit(design, block.arms["C"], block.models["C"], y), p["C"], atol=1e-10)
    masked = ip.frozen_predict(fits, ip.with_z(design, np.full(len(design), design[sf.Z].mean())))
    assert not np.allclose(masked, p["C"])


def test_region_overlap_matrix():
    def mask(rows, cols):
        m = rg.empty()
        m[rows, cols] = True
        return rg.pack(m)
    rows = [{"shot_id": "a", "region": "shooter", "kind": "region", "mask": mask(slice(0, 2), slice(0, 2))},
            {"shot_id": "a", "region": "ball", "kind": "region", "mask": mask(slice(0, 1), slice(0, 2))},
            {"shot_id": "a", "region": "ball", "kind": "placebo", "mask": mask(slice(5, 6), slice(0, 2))},
            {"shot_id": "b", "region": "shooter", "kind": "region", "mask": mask(slice(0, 1), slice(0, 1))}]
    out = ip.region_overlap(pd.DataFrame(rows)).set_index(["a", "b"])
    assert out.loc[("ball", "shooter")].share_of_a_in_b == 1.0 and out.loc[("shooter", "ball")].share_of_a_in_b == 0.5
    assert out.loc[("shooter", "shooter")].shots == 2 and out.loc[("ball", "shooter+ball")].jaccard == 0.5


def test_probe_targets_kinds_families_and_sources():
    table, attributes, _ = population(seed=80, n_matches=10, per_match=10)
    gpt = pd.Series([1.0, 0.0, np.nan], index=table.index[:3])
    caption = pd.Series(["header", "foot", "unknown"], index=table.index[:3])
    frozen = pd.DataFrame({"p_B": 0.2, "p_C": 0.3}, index=table.index)
    targets, meta = ip.probe_targets(attributes, gpt, caption, frozen)
    meta = meta.set_index("target")
    assert meta.loc["header_gpt", "source"] == "gpt" and meta.loc["header_caption", "source"] == "caption"
    assert meta.loc["header_caption", "present"] == 2 and meta.loc["set_piece", "kind"] == "binary"
    assert meta.loc["box_players@0", "kind"] == "continuous" and meta.loc["sn_direct_fk_le0", "family"] == "flags"
    assert targets.z_C_minus_z_B.iloc[0] == pytest.approx(np.log(0.3 / 0.7) - np.log(0.2 / 0.8))


def test_control_truths_hit_their_oracle_targets():
    table, attributes, _ = population(seed=90, n_matches=60, per_match=25, carried=False)
    rng = np.random.default_rng(191)
    attributes["box_players@0"] = table[sf.Z].to_numpy() + rng.normal(0, 1.0, len(table))   # Xc carries part of z
    design, columns = ip.design_table(table, attributes)
    prepared = ip.prepare(design, columns, blocks=("Xc",), draws=20)
    truths = ic.Truths.build(prepared, design)
    a, b = truths.solve(None)
    null = truths.oracle(a, b)
    assert a == 0 and null["G"] == pytest.approx(ic.G_TARGET, abs=1e-6) and null["R"] <= 1e-9
    a, b = truths.solve(0.25)
    plant = truths.oracle(a, b)
    assert a > 0 and plant["G"] == pytest.approx(ic.G_TARGET, abs=1e-5) and plant["R"] / plant["G"] == pytest.approx(0.25, abs=1e-3)
    p = truths.sampler(a, b)(np.random.default_rng(0))
    assert abs(p.mean() - ic.GOAL_RATE) < 0.02
    row = ic.replicate_row(prepared, (np.random.default_rng(1).random(len(p)) < p).astype(int), ("Xc",))
    assert row["Xc_verdict"] in ip.VERDICTS and "family_false_part" not in row


def test_code_identity_covers_the_closure_and_ignores_the_pin():
    code = script.code_hashes()
    assert {"xgpoc/dino/info_protocol.py", "xgpoc/dino/info_controls.py", "xgpoc/dino/ablation.py",
            "xgpoc/stack/fit.py"} <= set(code["files"])
    assert any(k.startswith("scripts/xgdino_info_protocol.py") for k in code["files"])
    assert "PROTOCOL_SHA256: str | None = " in Path(script.__file__).read_text()


def test_the_boundary_gate_turns_does_not_explain_into_inconclusive():
    read = {"gain": {"p_one_sided": 0.5, "upper_one_sided_95": 0.001}, "most": {"lo90": -0.01},
            "part": {"lo90": -0.01, "upper_one_sided_95": -0.001}}
    assert ip.decide({"Xc": read})["Xc"]["verdict"] == "does not explain"
    gated = ip.decide({"Xc": read}, not_explain_available=False)["Xc"]
    assert gated["verdict"] == "inconclusive" and gated["raw_verdict"] == "does not explain" and gated["gate"]
    rates = {"does not explain": {"rate": 0.65}}
    gate = ip.boundary_gate({"scenarios": {"plant25": {"blocks": {"Xc": {"rates": rates}}}}})
    assert not gate["available"] and gate["rate"] == 0.65
