"""Masked re-embedding of xg-dino-info (ticket 14): variants of a shot (regions, placebos, shooter+ball, keep-only),
the two fills, the overlay left alone, two-seed own-fold scoring equal to each seed's full forward averaged as
`z_g_combo`, the gate rule, and the run plan's phases. CPU, random weights, a ViT-S on small inputs."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from xgpoc.dino import masking as mk
from xgpoc.dino import regions as rg
from xgpoc.models.d_foundation import FinetuneViTXG

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import xgdino_mask_embed as script  # noqa: E402

N_TRAINED = 2


def box(r0, r1, c0, c1) -> np.ndarray:
    m = rg.empty()
    m[r0:r1, c0:c1] = True
    return m


def stored_for(shooter=True, ball=True, goal=True) -> dict:
    out = {}
    if shooter:
        out[("shooter", "region", 0)] = (box(8, 12, 10, 12), 0, 0)
        for k in range(rg.N_PLACEBO):
            out[("shooter", "placebo", k)] = (rg.shift(box(8, 12, 10, 12), 5 + k, 0), 5 + k, 0)
    if ball:
        out[("ball", "region", 0)] = (box(11, 13, 11, 13), 0, 0)
    if goal:
        out[("goal", "region", 0)] = (box(5, 8, 28, 32), 0, 0)
    out[("players", "region", 0)] = (box(8, 13, 10, 13), 0, 0)
    out[("background", "region", 0)] = (~box(5, 13, 10, 32), 0, 0)
    return out


def test_variants_hold_regions_placebos_union_and_keep_only():
    allowed = box(6, 16, 0, 32) & ~box(8, 13, 9, 14)
    variants, union = mk.shot_variants("s1", stored_for(), allowed)
    names = [v.name for v in variants]
    assert names[0] == "unmasked" and variants[0].mask is None
    assert {"shooter", "ball", "goal", "players", "background", "shooter+ball"} <= set(names)
    assert [n for n in names if n.startswith("shooter|placebo")] == ["shooter|placebo0", "shooter|placebo1", "shooter|placebo2"]
    u = next(v for v in variants if v.name == "shooter+ball")
    assert (u.mask == (box(8, 12, 10, 12) | box(11, 13, 11, 13))).all() and union["present"]
    shifts, n = rg.placebos(u.mask, allowed, rg.rng_for("s1", "shooter+ball"))       # the protocol's draw
    placebos = [v for v in variants if v.region == "shooter+ball" and v.kind == "placebo"]
    assert [(v.dx, v.dy) for v in placebos] == shifts and union["n_feasible"] == n > 0
    for v in placebos:
        assert (v.mask == rg.shift(u.mask, v.dx, v.dy)).all() and not (v.mask & ~allowed).any()
    keep = {v.name: v for v in variants if v.kind == "keep"}
    assert (keep["keep_shooter_ball"].mask == ~u.mask).all() and keep["keep_shooter_ball"].parts == "shooter+ball"
    assert (keep["keep_goal"].mask == ~box(5, 8, 28, 32)).all()


def test_absent_regions_get_no_variant_and_keep_only_uses_the_present_ones():
    variants, union = mk.shot_variants("s2", stored_for(ball=False, goal=False), rg.empty())
    names = {v.name for v in variants}
    assert "ball" not in names and "goal" not in names and "keep_goal" not in names and "shooter+ball" not in names
    assert not union["present"] and union["placebo_status"] == "region_absent"
    keep = next(v for v in variants if v.name == "keep_shooter_ball")
    assert keep.parts == "shooter" and (keep.mask == ~box(8, 12, 10, 12)).all()
    _, union = mk.shot_variants("s3", stored_for(), rg.empty())                       # nowhere to put a placebo
    assert union["placebo_status"] == "no_valid_placebo"


def test_fills_replace_only_the_masked_pixels_outside_the_overlay():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (252, 448, 3), dtype=np.uint8)
    overlay = np.zeros((252, 448), bool)
    overlay[:28] = True
    img[overlay] = 114
    mask = box(0, 4, 0, 4)                                                            # two patch rows under the overlay
    sources = mk.fill_sources(img)
    mean = mk.apply(img, mask, sources["mean"], overlay)
    pixels = mk.patch_pixels(mask)
    assert pixels.shape == (252, 448) and pixels.sum() == 16 * 14 * 14
    assert (mean[overlay] == 114).all()                                               # overlay grey kept
    assert (mean[pixels & ~overlay] == img.reshape(-1, 3).mean(0).round().astype(np.uint8)).all()
    assert (mean[~pixels] == img[~pixels]).all()
    blur = mk.apply(img, mask, sources["blur"], overlay)
    assert (blur[pixels & ~overlay] == sources["blur"][pixels & ~overlay]).all() and (blur[~pixels] == img[~pixels]).all()
    assert mk.apply(img, None, None, overlay) is img


def model_with_seed(tail_seed: int | None) -> FinetuneViTXG:
    torch.manual_seed(0)
    model = FinetuneViTXG("dinov2_s", n_blocks=N_TRAINED, n_out=2, pretrained=False).eval()
    if tail_seed is not None:
        generator = torch.Generator().manual_seed(tail_seed)
        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad:
                    p.add_(0.02 * torch.randn(p.shape, generator=generator))
    return model


def test_fold_scorer_averages_both_seeds_of_the_own_fold_as_z_g_combo():
    images = torch.randn(3, 3, 56, 98, generator=torch.Generator().manual_seed(1))
    scorer = mk.FoldScorer(model_with_seed(None), N_TRAINED, "cpu")
    models = {(0, 0): model_with_seed(1), (0, 1): model_with_seed(2), (1, 0): model_with_seed(3), (1, 1): model_with_seed(4)}
    temps = {(0, 0): 1.3, (0, 1): 0.8, (1, 0): 1.1, (1, 1): 0.9}
    for (fold, seed), model in models.items():
        scorer.add(fold, seed, temps[(fold, seed)], model)
    with pytest.raises(ValueError, match="registered twice"):
        scorer.add(0, 1, 1.0, model_with_seed(5))
    assert scorer.seeds == {0: [0, 1], 1: [0, 1]}
    out = scorer.score(images, 1)
    with torch.no_grad():
        raw = [models[(1, s)](image=images)[:, 0].double().numpy() for s in (0, 1)]
    np.testing.assert_allclose(out["logit"], np.stack(raw, 1), atol=1e-5)
    np.testing.assert_allclose(out["z"], (raw[0] / temps[(1, 0)] + raw[1] / temps[(1, 1)]) / 2, atol=1e-5)
    assert out["pool"].shape == (3, 2, 5 * 384)
    with pytest.raises(ValueError, match="no model for fold 2"):
        scorer.score(images, 2)


def test_gate_needs_both_the_max_and_the_mean_bound_and_lists_the_outliers():
    ids = [f"s{i}" for i in range(1000)]
    reference = pd.Series(np.linspace(-3, 0, 1000), index=ids)
    table = pd.DataFrame({"shot_id": ids, "fold": 0, "row": range(1000), "kind": "unmasked", "z": reference.to_numpy() + 5e-8})
    table.loc[3, "z"] += 1.5e-3
    gate = script.gate_record(table, reference)
    assert gate["passed"] and gate["max_abs_delta"] == pytest.approx(1.5e-3, abs=1e-7)
    assert [r["shot_id"] for r in gate["listed"]] == ["s3"] and gate["counts_above"]["1e-06"] == 1
    worse = table.copy()
    worse.loc[3, "z"] += 1e-3                                                         # max over 2e-3
    assert not script.gate_record(worse, reference)["passed"]
    noisy = table.copy()
    noisy["z"] += 2e-5                                                                # mean over 1e-5, max fine
    assert not script.gate_record(noisy, reference)["passed"]
    assert not script.gate_record(table, reference.drop("s7"))["passed"]              # a shot without reference


def test_plan_runs_the_gate_first_then_the_mean_fill_then_blur_in_whole_shots_of_one_fold():
    shots = pd.DataFrame({"shot_id": ["b", "a", "c"], "fold": [1, 0, 0], "jpeg": ["b.jpg", "a.jpg", "c.jpg"]})
    stored = {"a": stored_for(), "b": stored_for(ball=False), "c": {}}
    allowed = {s: box(6, 16, 0, 32) & ~box(8, 13, 9, 14) for s in "abc"}
    plan, unions = script.build_plan(shots, stored, allowed, gate_shots=2, chunk_shots=1)
    assert plan.row.tolist() == list(range(len(plan)))
    assert plan.phase.tolist() == sorted(plan.phase, key=["gate", "mean", "blur"].index)
    gate = plan[plan.phase.eq("gate")]
    assert gate.variant.eq("unmasked").all() and gate.fill.eq("none").all() and len(gate) == 3
    assert gate.chunk.tolist() == [0, 0, 1]                                           # fold 0 (a, c) then fold 1 (b)
    for _, rows in plan.groupby("chunk"):
        assert rows.fold.nunique() == 1 and rows.phase.nunique() == 1
        assert rows.row.diff().dropna().eq(1).all()
    masked = plan[plan.phase.ne("gate")]
    assert masked.groupby("variant").fill.apply(set).eq({"mean", "blur"}).all()
    assert not plan.shot_id.eq("c").loc[plan.phase.ne("gate")].any()                 # no region -> unmasked only
    assert set(unions.shot_id) == {"a", "b", "c"} and unions.set_index("shot_id").present.to_dict() == {"a": True, "b": False, "c": False}
