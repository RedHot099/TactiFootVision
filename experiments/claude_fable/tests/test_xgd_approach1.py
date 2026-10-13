"""Approach 1 (xg-detector 03): A1 vector and standardiser, A2 raster, the dataset's augmentations, the branches, the
plan check and the development comparison."""

import ast
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from xgpoc.bundle import trained_keys
from xgpoc.data.datasets import ImageDataset
from xgpoc.detector import approach1 as a1
from xgpoc.detector import sanitation as sn
from xgpoc.detector.features import FLAGS, GEOMETRY

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import xgd_approach1_report as report  # noqa: E402


def sanitised_rows(shot_ids, offsets=a1.OFFSETS, **values) -> pd.DataFrame:
    rows = []
    for i, shot_id in enumerate(shot_ids):
        for offset in offsets:
            row = sn.missing_row() | {"shot_id": shot_id, "offset_ms": offset, "aligned": False, "pts_x": [], "pts_y": [],
                                      "pts_channel": []}
            row |= {k: (v[i] if isinstance(v, list) else v) for k, v in values.items()}
            rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ A1
def test_the_a1_vector_has_both_offsets_and_no_label():
    rows = sanitised_rows(["a", "b"], dist=[10.0, np.nan], ball_ok=[1.0, 0.0], ball_source=[1.0, np.nan], n_valid=[12.0, 0.0])
    rows["goal"], rows["on_target"] = 1, 1
    table = a1.a1_table(rows)
    assert table.shape == (2, 2 * len(a1.A1_COLUMNS)) and len(a1.A1_COLUMNS) == 38
    assert not [c for c in table.columns if c.split("@")[0] in ("goal", "on_target", "label", "is_penalty")]
    assert (table.loc["a", "dist@0"], table.loc["a", "dist_missing@0"], table.loc["b", "dist_missing@-400"]) == (10.0, 0.0, 1.0)
    assert table.loc["a", ["ball_source_0@0", "ball_source_1@0", "ball_source_2@0"]].tolist() == [0.0, 1.0, 0.0]
    assert table.loc["b", ["ball_source_0@0", "ball_source_1@0", "ball_source_2@0"]].tolist() == [0.0, 0.0, 0.0]
    assert set(GEOMETRY + FLAGS + sn.INDICATORS + sn.COUNTS) <= {c.split("@")[0] for c in table.columns}


def test_the_standardiser_is_fitted_on_the_training_rows_only_and_imputes_the_mean():
    rows = sanitised_rows(["a", "b", "c", "d"], dist=[10.0, 20.0, np.nan, 1000.0], n_valid=[10.0, 14.0, 0.0, 5.0])
    table = a1.a1_table(rows)
    s = a1.Standardiser.fit(table.loc[["a", "b", "c"]])                       # "d" is the held-out shot
    assert (s.mean["dist@0"], s.std["dist@0"]) == (15.0, 5.0)
    assert s.std["gk_depth@0"] == 1.0 and s.mean["gk_depth@0"] == 0.0          # all missing: no NaN, no division by 0
    x = pd.DataFrame(s(table), index=table.index, columns=table.columns)
    assert x.loc["c", "dist@0"] == 0.0 and x.loc["c", "dist_missing@0"] == 1.0
    assert x.loc["d", "dist@0"] == pytest.approx(197.0)                        # transformed, never refitted
    assert x.loc["a", "ball_ok@0"] == table.loc["a", "ball_ok@0"]              # binary columns pass unchanged
    vectors = a1.geometry_inputs(table, s)
    assert vectors["a"].dtype == np.float32 and vectors["a"].shape == (76,)


# ------------------------------------------------------------------ A2
POINTS = [a1.FramePoints(np.array([[94.0, 30.0], [100.0, 34.0], [30.0, 10.0]]), np.array([sn.ATTACKER, sn.DEFENDER, sn.DEFENDER]),
                         True, True),
          a1.FramePoints(np.zeros((0, 2)), np.zeros(0, int), False, True)]


def test_the_raster_draws_points_in_their_channel_and_marks_presence():
    r = a1.rasterise(POINTS)
    assert r.shape == (a1.RASTER_CHANNELS, 72, 64) == (16, 72, 64)
    row, col = np.unravel_index(np.argmax(r[sn.ATTACKER]), r[sn.ATTACKER].shape)
    assert (abs(a1._XC[col] - 94.0), abs(a1._YC[row] - 30.0)) == (0.5, 0.5)   # a cell next to the point
    assert r[sn.DEFENDER].max() == pytest.approx(np.exp(-0.25), rel=1e-5)     # one point, cell centres 0.5 m away
    assert r[sn.DEFENDER].sum() == pytest.approx(2 * np.pi, rel=0.01)          # the own-half point is not drawn
    assert (r[5].min(), r[6].min()) == (1.0, 1.0)                               # t0: aligned, homography
    assert (r[7:12].sum(), r[12].max(), r[13].min()) == (0.0, 0.0, 1.0)         # -400: no points, homography
    assert r[-2, 0, 0] == -1.0 and r[-2, 0, -1] == 1.0 and r[-1, -1, 0] == 1.0


def test_a_mirrored_raster_is_the_raster_flipped_across_the_long_axis():
    plain, mirrored = a1.rasterise(POINTS), a1.rasterise(POINTS, mirrored=True)
    np.testing.assert_allclose(mirrored[:-2], plain[:-2, ::-1, :], atol=1e-6)
    np.testing.assert_array_equal(mirrored[-2:], plain[-2:])


@pytest.fixture
def frame_file(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "frame.jpg"
    cv2.imwrite(str(path), rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8))
    return str(path)


def _shots(frame_file, n=3):
    return pd.DataFrame({"shot_id": [f"s{i}" for i in range(n)], "frame_path": frame_file, "goal": [True, False, False][:n],
                         "on_target": [1.0, 0.0, 1.0][:n]})


@pytest.mark.parametrize("train", [False, True])
def test_the_dataset_makes_g_combos_image_augmentations(frame_file, train):
    shots = _shots(frame_file)
    kw = {"size_wh": (448, 252), "mask": "overlay", "train": train, "aux_col": "on_target",
          **({"rrc_scale": (0.85, 1.0)} if train else {})}
    points = {s: POINTS for s in shots.shot_id}
    ours, theirs = a1.DetectorDataset(shots, variant="A2", points=points, **kw), ImageDataset(shots, **kw)
    for i in range(len(shots)):
        np.random.seed(i)
        a = ours[i]
        np.random.seed(i)
        b = theirs[i]
        assert torch.equal(a["image"], b["image"]) and torch.equal(a["y"], b["y"]) and torch.equal(a["y_aux"], b["y_aux"])
        assert a["raster"].shape == (16, 72, 64) and a["has_raster"] == 1.0


@pytest.mark.parametrize(("draw", "mirrored"), [(0.1, True), (0.9, False)])
def test_a_flipped_image_gets_a_mirrored_raster(frame_file, monkeypatch, draw, mirrored):
    shots = _shots(frame_file, 1)
    kw = {"size_wh": (448, 252), "train": True, "rrc_scale": (0.85, 1.0)}
    ours, theirs = a1.DetectorDataset(shots, variant="A2", points={"s0": POINTS}, **kw), ImageDataset(shots, **kw)
    monkeypatch.setattr(np.random, "rand", lambda: draw)                    # the flip draw (< 0.5 flips)
    np.random.seed(0)
    item = ours[0]
    np.random.seed(0)
    assert torch.equal(item["image"], theirs[0]["image"])                   # flipped exactly when G-combo's would be
    np.testing.assert_array_equal(item["raster"].numpy(), a1.rasterise(POINTS, mirrored=mirrored))


def test_the_dataset_refuses_missing_inputs_and_other_frames(frame_file):
    shots = _shots(frame_file)
    with pytest.raises(ValueError, match="missing"):
        a1.DetectorDataset(shots, variant="A1", geometry={"s0": np.zeros(76, np.float32)})
    with pytest.raises(ValueError, match="t0 frame"):
        a1.DetectorDataset(shots, variant="A1", geometry={}, path_col="frame_path_-400")
    with pytest.raises(ValueError, match="variant"):
        a1.DetectorDataset(shots, variant="A3", geometry={})


# ------------------------------------------------------------------ branches and model
def test_a_shot_without_any_raster_gets_the_learned_missing_vector():
    branch = a1.RasterBranch().eval()
    raster = torch.from_numpy(np.stack([a1.rasterise(POINTS)] * 2))
    e = branch(raster, torch.tensor([1.0, 0.0]))
    assert e.shape == (2, 64)
    assert torch.equal(e[1], branch.missing.detach()) and not torch.equal(e[0], branch.missing.detach())
    assert sum(p.numel() for p in branch.parameters()) < 200_000             # a small CNN


@pytest.mark.parametrize("variant", a1.VARIANTS)
def test_the_branch_enters_g_combos_head_and_is_stored_in_the_bundle(variant):
    torch.manual_seed(0)
    model = a1.DetectorG(variant, backbone="dinov2_s", n_blocks=1, pretrained=False).eval()
    inputs = {"geometry": torch.zeros(2, 76)} if variant == "A1" else {
        "raster": torch.from_numpy(np.stack([a1.rasterise(POINTS)] * 2)), "has_raster": torch.ones(2)}
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 252, 448), **inputs)
    assert out.shape == (2, 2)
    assert model.head.head[2].in_features == 384 * 5 + 64                     # z (CLS + 4 queries) and the branch
    keys = trained_keys(model)
    assert any(k.startswith("branch.") for k in keys) and any(k.startswith("head.aux_norm") for k in keys)


def test_the_spec_carries_the_standardiser_and_rebuilds_the_variant():
    table = a1.a1_table(sanitised_rows(["a", "b"], dist=[10.0, 20.0]))
    s = a1.Standardiser.fit(table)
    spec = a1.spec("A1", "dinov2_l", 4, "overlay", s)
    assert a1.standardiser_of(json.loads(json.dumps(spec))) == s
    assert spec["input"]["a1_columns"] == a1.A1_COLUMNS
    with pytest.raises(ValueError):
        a1.spec("A2", "dinov2_l", 4, "overlay", s)
    with pytest.raises(ValueError):
        a1.spec("A1", "dinov2_l", 4, "overlay", None)
    with pytest.raises(ValueError, match="approach-1"):
        a1.build({"kind": "finetune_vit_xg"})


# ------------------------------------------------------------------ plan and pool
@pytest.mark.skipif(not (a1.PLAN_DIR / "plan.csv").exists(), reason="needs the frozen split plan")
def test_the_plan_check_refuses_any_other_plan(tmp_path):
    from xgpoc.paths import INDEX
    from xgpoc.vendor import split_check

    assert a1.PLAN_SHA256 == split_check.PLAN_SHA256
    if not INDEX.exists():
        pytest.skip("needs the index")
    assert a1.verify_plan(INDEX).role.value_counts().to_dict() == a1.PLAN_COUNTS
    other = tmp_path / "plan"
    shutil.copytree(a1.PLAN_DIR, other)
    plan = pd.read_csv(other / "plan.csv")
    plan.loc[0, "fold"] = (plan.loc[0, "fold"] + 1) % 5
    plan.to_csv(other / "plan.csv", index=False)
    with pytest.raises(ValueError, match="changed since its completion record"):
        a1.verify_plan(INDEX, other)
    record = json.loads((other / "complete.json").read_text())
    record["files"]["plan.csv"] = a1.sha256_file(other / "plan.csv")
    (other / "complete.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="not the approved one"):
        a1.verify_plan(INDEX, other)


def test_the_pool_is_the_training_matches_with_their_fold():
    plan = pd.DataFrame({"game": ["g1", "g2", "g3"], "role": ["training", "gate", "evaluation"], "fold": [3, -1, -1]})
    shots = pd.DataFrame({"shot_id": ["a", "b", "c"], "game": ["g1", "g2", "g3"]})
    assert a1.training_pool(shots, plan)[["shot_id", "fold"]].values.tolist() == [["a", 3]]
    with pytest.raises(ValueError, match="not in the plan"):
        a1.training_pool(pd.DataFrame({"shot_id": ["x"], "game": ["g9"]}), plan)


# ------------------------------------------------------------------ the development comparison
def test_the_paired_delta_and_its_interval():
    y = np.array([1.0, 0.0, 0.0, 1.0])
    p = np.array([0.7, 0.2, 0.1, 0.6])
    weights = np.ones((3, 4))
    assert report.paired_delta(y, p, p, weights) == {"delta_ll": 0.0, "lo": 0.0, "hi": 0.0}
    better = report.paired_delta(y, np.array([0.8, 0.1, 0.1, 0.7]), p, weights)
    assert better["delta_ll"] < 0 and better["lo"] == pytest.approx(better["delta_ll"])


@pytest.mark.parametrize(("delta", "crosses", "noise"), [(-0.0025, True, True), (-0.002, True, True), (-0.004, True, False),
                                                          (-0.0019, False, False), (0.001, False, False)])
def test_the_a3_trigger_and_the_noise_flag(delta, crosses, noise):
    assert report.trigger(delta) == {"crosses": crosses, "noise_compatible": noise}


def test_the_out_of_fold_check_refuses_gaps_wrong_folds_labels_and_temperatures():
    pool = pd.DataFrame({"shot_id": ["a", "b"], "goal": [True, False], "fold": [0, 1]})
    good = pd.DataFrame({"shot_id": ["a", "b"], "goal": [1, 0], "logit": [0.0, 1.0], "fold": [0, 1]})
    good["p"] = 1 / (1 + np.exp(-good.logit / 2.0))
    report.check_out_of_fold(good, pool, {0: 2.0, 1: 2.0})
    for broken, message in ((good.iloc[:1], "cover"), (good.assign(fold=[1, 1]), "outside their plan fold"),
                            (good.assign(goal=[0, 0]), "labels"), (good.assign(p=good.p + 0.01), "temperature|sigmoid")):
        with pytest.raises(ValueError, match=message):
            report.check_out_of_fold(broken, pool, {0: 2.0, 1: 2.0})


# ------------------------------------------------------------------ no StatsBomb on the approach-1 paths
@pytest.mark.parametrize(("script", "readers"), [
    ("xgd_sanitise.py", {"verified_file", "read_parquet", "replay_features", "sanitised_table", "read_text"}),
    ("xgd_approach1.py", {"development_pool", "verified_file", "read_parquet", "variant_inputs", "replay", "run_trial",
                          "own_sources", "frame_fingerprint"}),
    ("xgd_approach1_report.py", {"verify_plan", "read_parquet", "out_of_fold", "variant_predictions", "training_pool"}),
])
def test_every_read_of_the_approach_1_scripts_is_inside_the_guard(script, readers):
    main = next(n for n in ast.parse((ROOT / "scripts" / script).read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    guarded = [n for n in main.body if isinstance(n, ast.With) and any(
        getattr(i.context_expr.func, "id", "") == "no_statsbomb_reads" for i in n.items if isinstance(i.context_expr, ast.Call))]
    assert len(guarded) == 1
    inside = {id(n) for n in ast.walk(guarded[0])}
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Call) and ast.unparse(n.func).split(".")[-1] in readers]
    assert {ast.unparse(c.func).split(".")[-1] for c in calls} == readers
    assert all(id(c) in inside for c in calls), [ast.unparse(c) for c in calls if id(c) not in inside]


# ------------------------------------------------------------------ the runner's and the sanitation script's refusals
def test_frame_points_rebuilds_both_offsets_in_order():
    rows = sanitised_rows(["a"])
    rows.loc[rows.offset_ms == 0, ["aligned", "homography_ok"]] = [True, 1.0]
    rows.at[int(np.flatnonzero(rows.offset_ms == 0)[0]), "pts_x"] = [94.0, 100.0]
    rows.at[int(np.flatnonzero(rows.offset_ms == 0)[0]), "pts_y"] = [30.0, 34.0]
    rows.at[int(np.flatnonzero(rows.offset_ms == 0)[0]), "pts_channel"] = [sn.ATTACKER, sn.BALL]
    t0, before = a1.frame_points(rows)["a"]
    assert t0.xy.tolist() == [[94.0, 30.0], [100.0, 34.0]] and t0.channel.tolist() == [sn.ATTACKER, sn.BALL]
    assert (t0.aligned, t0.homography, before.aligned, before.homography, len(before.xy)) == (True, True, False, False, 0)


def test_the_runner_refuses_a_sanitised_table_without_both_offsets_of_every_pool_shot():
    import xgd_approach1 as runner

    pool = pd.DataFrame({"shot_id": ["a", "b"]})
    table, points = runner.variant_inputs(sanitised_rows(["a", "b", "z"]), pool, "A1")
    assert table.index.tolist() == ["a", "b"] and points is None
    assert set(runner.variant_inputs(sanitised_rows(["a", "b"]), pool, "A2")[1]) == {"a", "b"}
    for broken in (sanitised_rows(["a"]), sanitised_rows(["a", "b"], offsets=(0,)), sanitised_rows(["a", "b"], offsets=(0, -800))):
        with pytest.raises(ValueError, match="sanitised table"):
            runner.variant_inputs(broken, pool, "A1")


def test_the_runner_refuses_to_start_while_another_job_holds_the_gpu(monkeypatch):
    import subprocess

    import xgd_approach1 as runner

    def smi(listing):
        return lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=listing)
    monkeypatch.setattr(subprocess, "run", smi("2891, 139\n3388, 159\n"))       # desktop and test contexts only
    runner.check_gpu_free()
    monkeypatch.setattr(subprocess, "run", smi("2891, 139\n4242, 14000\n"))
    with pytest.raises(SystemExit, match="another job"):
        runner.check_gpu_free()


def test_the_sanitation_script_reports_label_free_coverage_per_offset():
    import xgd_sanitise

    rows = sanitised_rows(["a", "b"], shot_ok=[1.0, 0.0], ff_def_cone=[0.0, np.nan], n_valid=[10.0, 0.0])
    rows["aligned"] = rows.shot_ok.astype(bool)
    out = xgd_sanitise.coverage(rows)
    assert set(out) == {"0", "-400"}
    assert (out["0"]["shot_ok"], out["0"]["def_cone"], out["0"]["n_valid"], out["0"]["aligned"]) == (0.5, 0.5, 5.0, 0.5)
