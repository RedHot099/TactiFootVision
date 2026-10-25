"""Layout xgt of `scripts/score_offset_frames.py` (xg-temporal ticket 01b): g_combo only, every shot scored by the two
models of the xgv fold that held out its match (no test group), embeddings of the kept seeds outside --out, the
spot check against z_g_combo, the role and plan checks, and the g_xgv folder contract of `xgpoc.temporal.inputs`.
CPU, ViT-S with random weights, on the dummy world of `tests/test_offset_scores.py`."""

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from xgpoc.bundle import sha256_file
from xgpoc.models import g_bundle
from xgpoc.models.d_foundation import FinetuneViTXG
from xgpoc.probe.table import TABLE_COLUMNS, build_plan, reference_check
from xgpoc.temporal import inputs as ti

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_offset_frames as builder  # noqa: E402
import score_offset_frames as scorer  # noqa: E402
from test_offset_frames import write_clip  # noqa: E402
from test_offset_scores import N_TRAINED, tiny_bundles  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SHOTS = {  # shot -> (game, official split, penalty, clip frames, cut at)
    "a1": ("A", "train", False, 200, None), "a2": ("A", "train", False, 200, 80), "ap": ("A", "train", True, 200, None),
    "b1": ("B", "valid", False, 200, None), "t1": ("T", "test", False, 200, None)}
XGV_FOLD = {"A": 0, "B": 1, "T": 0}      # the xgv folds cover a match of the official test split
GRID = list(ti.GRID)


def temperature(fold: int, seed: int) -> float:
    return 1.1 + 0.2 * seed + 0.05 * fold          # as `tiny_bundles` writes them


def xgt_args(world, out: Path, emb: Path, *extra: str) -> list[str]:
    files, tmp = world["files"], world["tmp"]
    return ["--layout", "xgt", "--out", str(out), "--embeddings-dir", str(emb), "--index", str(files["index"]),
            "--offset-frames", str(files["frames"]), "--g-combo", str(tmp / "g_combo"), "--folds", str(files["folds"]),
            "--overlay", str(files["overlay"]), "--protocol", str(files["protocol"]), "--reference", str(files["oof"]),
            "--shots-csv", str(files["shots_csv"]), "--expect-models", "4", "--batch-size", "3", "--chunk-shots", "1",
            "--workers", "0", "--device", "cpu", *extra]


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Dummy index, offset frames, xgv folds (game T of the official test split in fold 0), tiny fold models named
    under the plan, the plan's roles (patched `verify_plan`), z_g_combo from the models on the index JPEG, and one
    finished xgt run with embedding seeds 0 and 1."""
    tmp = tmp_path_factory.mktemp("xgt_xgv")
    with pytest.MonkeyPatch.context() as mp:
        weights = tmp / "model.safetensors"
        weights.write_bytes(b"stand-in for the pretrained checkpoint")
        mp.setattr(g_bundle, "pretrained_weight_file", lambda backbone: weights)

        def build(*, backbone, n_blocks, n_out, pretrained):
            torch.manual_seed(0)
            return FinetuneViTXG(backbone, n_blocks=n_blocks, n_out=n_out, pretrained=False)

        mp.setattr(g_bundle, "FinetuneViTXG", build)
        mp.setattr(scorer, "own_sources", lambda: {"xgpoc/probe/scoring.py": "fixed", "xgpoc/probe/table.py": "fixed"})
        rows = []
        for i, (shot, (game, split, penalty, frames, cut_at)) in enumerate(SHOTS.items()):
            clip = write_clip(tmp / f"{shot}.mp4", frames=frames, seed=30 + i, cut_at=cut_at)
            capture = cv2.VideoCapture(str(clip))
            for _ in range(101):
                ok, frame = capture.read()
            assert ok
            jpeg = tmp / f"{shot}.jpg"
            cv2.imwrite(str(jpeg), frame)
            rows.append(dict(shot_id=shot, frame_path=str(jpeg), game=game, split=split, label="Shots off target",
                             goal=False, is_penalty=penalty, half=1, position_ms=90000, clip_path=str(clip)))
        index = tmp / "index.parquet"
        pd.DataFrame(rows).to_parquet(index)
        protocol = tmp / "protocol.md"
        protocol.write_text("approved protocol")
        frames_dir = tmp / "offset_frames"
        builder.build(index, frames_dir, "model_input_png", 2, protocol=protocol)
        folds = tmp / "folds.csv"
        pd.DataFrame({"game": list(XGV_FOLD), "home": list(XGV_FOLD), "fold": list(XGV_FOLD.values())}).to_csv(folds, index=False)
        files = {"index": index, "folds": folds, "overlay": ROOT / "configs/overlay_mask.png", "frames": frames_dir,
                 "protocol": protocol}
        spec = g_bundle.g_spec("dinov2_s", N_TRAINED, 2, "overlay")
        shots = pd.read_csv(frames_dir / "shots.csv")
        tiny_bundles(tmp / "g_combo", "g_combo", spec, files, shots)
        for record_file in sorted((tmp / "g_combo").glob("fold*/seed*/complete.json")):     # trained under the xgv plan
            record = json.loads(record_file.read_text())
            record["identity"]["plan"] = {"sha256": scorer.a1.PLAN_SHA256, "seed": 0}
            record_file.write_text(json.dumps(record))
        roles = pd.DataFrame({"game": ["A", "B", "T", "E"], "role": ["training", "training", "training", "evaluation"],
                              "fold": [0, 1, 0, np.nan]})                  # the plan's roles and training folds
        mp.setattr(scorer.a1, "verify_plan", lambda index, folds: roles)

        z = {}                                    # z_g_combo: mean logit / T of the fold's two models on the index JPEG
        jpeg = pd.read_parquet(index).set_index("shot_id").frame_path
        for shot, (game, *_rest) in SHOTS.items():
            fold = XGV_FOLD[game]
            df = pd.DataFrame({"shot_id": [shot], "goal": 0, "frame_path": [jpeg[shot]]})
            logits = [g_bundle.g_outputs(g_bundle.load_g(tmp / "g_combo" / f"fold{fold}" / f"seed{s}" / "bundle.pt").model, df, spec,
                                         device="cpu", batch_size=1, num_workers=0)[0][0] / temperature(fold, s) for s in (0, 1)]
            z[shot] = float(np.mean(logits))
        oof_dir = tmp / "oof"
        oof_dir.mkdir()
        oof = oof_dir / "oof.csv"
        pd.DataFrame({"shot_id": list(z), "fold": 0, "z_g_combo": list(z.values()), "goal": 1}).to_csv(oof, index=False)
        (oof_dir / "complete.json").write_text(json.dumps({"files": {"oof.csv": sha256_file(oof)}}))
        shots_csv = tmp / "extraction_shots.csv"
        pd.DataFrame({"shot_id": [s for s, v in SHOTS.items() if not v[2]], "game": [v[0] for v in SHOTS.values() if not v[2]]}).to_csv(
            shots_csv, index=False)
        files |= {"oof": oof, "shots_csv": shots_csv}
        world = {"tmp": tmp, "files": files, "spec": spec, "shots": shots, "frames": pd.read_csv(frames_dir / "frames.csv"),
                 "z": pd.Series(z), "out": tmp / "out", "emb": tmp / "emb", "roles": roles}
        world["argv"] = xgt_args(world, world["out"], world["emb"], "--embed-seeds", "0", "1")
        assert scorer.run(scorer.parse_args(world["argv"])) == 0
        yield world


def reference_outputs(world, fold: int, seed: int, frames: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.DataFrame({"shot_id": frames.shot_id.to_numpy(), "goal": 0,
                       "frame_path": [str(world["files"]["frames"] / p) for p in frames.frame_path]})
    model = g_bundle.load_g(world["tmp"] / "g_combo" / f"fold{fold}" / f"seed{seed}" / "bundle.pt").model
    return g_bundle.g_outputs(model, df, world["spec"], device="cpu", batch_size=4, num_workers=0)


# --- end to end ------------------------------------------------------------------------------------------------------------

def test_every_shot_is_scored_by_its_xgv_fold_models_with_both_seed_embeddings(world):
    table = pd.read_parquet(world["out"] / "predictions.parquet")
    assert list(table.columns) == TABLE_COLUMNS and set(table.encoder) == {"g_combo"}
    assert sorted(table.shot_id.unique()) == ["a1", "a2", "b1", "t1"] and len(table) == 4 * len(GRID)
    assert sorted(table.offset_ms.unique().tolist()) == GRID
    assert (table.fold_source == table.game.map(XGV_FOLD)).all() and (table.n_models == 2).all()
    assert (table.seed_source == "0,1").all() and table.oof.all()
    scored = table.frame_status.eq("ok")
    assert (table.embedding_ref[scored] >= 0).all()           # t1 too: no test group in the xgv folds
    seed0 = np.load(world["emb"] / "embeddings_g_combo.npy", mmap_mode="r")
    seed1 = np.load(world["emb"] / "embeddings_g_combo_seed1.npy", mmap_mode="r")
    assert seed0.shape == seed1.shape == (int(scored.sum()), 5 * 384) and seed0.dtype == np.float16
    frames = world["frames"]
    for shot in ("a1", "t1"):
        fold = XGV_FOLD[SHOTS[shot][0]]
        mine = frames[frames.shot_id.eq(shot) & frames.offset_ms.isin([-1600, 0, 400]) & frames.status.eq("ok")]
        rows = table.set_index(["shot_id", "offset_ms"]).loc[[(shot, o) for o in mine.offset_ms]]
        outputs = {s: reference_outputs(world, fold, s, mine) for s in (0, 1)}
        np.testing.assert_allclose(rows.xg_logit, np.mean([outputs[s][0] / temperature(fold, s) for s in (0, 1)], 0), atol=1e-4)
        np.testing.assert_allclose(rows.on_target_logit, np.mean([outputs[s][1] for s in (0, 1)], 0), atol=1e-4)
        refs = rows.embedding_ref.to_numpy()
        np.testing.assert_allclose(seed0[refs], outputs[0][2], rtol=2e-3, atol=2e-3)
        np.testing.assert_allclose(seed1[refs], outputs[1][2], rtol=2e-3, atol=2e-3)
        assert not np.allclose(seed0[refs], seed1[refs], atol=1e-2)


def test_out_holds_the_table_and_record_only_and_names_the_arrays_by_path_and_hash(world):
    out, emb = world["out"], world["emb"]
    assert sorted(p.name for p in out.iterdir()) == ["complete.json", "predictions.parquet"]
    assert sorted(p.name for p in emb.iterdir()) == ["embeddings_g_combo.npy", "embeddings_g_combo_seed1.npy"]
    record = json.loads((out / "complete.json").read_text())
    assert set(record["files"]) == {"predictions.parquet"}
    assert {n: (e["seed"], e["encoder"]) for n, e in record["embeddings"].items()} == {
        "g_combo": (0, "g_combo"), "g_combo_seed1": (1, "g_combo")}
    for entry in record["embeddings"].values():
        assert sha256_file(Path(entry["path"])) == entry["sha256"] and entry["dtype"] == "float16"
    identity = record["identity"]
    assert identity["version"] == scorer.XGT_VERSION and identity["xgt"]["embedding_seeds"] == [0, 1]
    assert identity["xgt"]["roles"] == {"training": 4} and identity["xgt"]["reference"]["columns"] == ["shot_id", "z_g_combo"]
    assert identity["xgt"]["shots_csv"]["sha256"] == sha256_file(world["files"]["shots_csv"])
    assert all(m["plan_sha256"] == scorer.a1.PLAN_SHA256 for m in identity["bundles"]["g_combo"])
    assert identity["plan"]["full"] is False and identity["plan"]["offsets_ms"] == GRID


def test_spot_check_against_z_g_combo_passes_on_the_jpeg_and_describes_the_png(world):
    check = json.loads((world["out"] / "complete.json").read_text())["spot_check"]
    control = check["index_jpeg_control"]
    assert check["pipeline_passed"] and control["passed"] and control["shots"] == 4 and control["max_abs_dlogit"] < 1e-3
    png = check["offset0_vs_z_g_combo"]
    assert png["shots"] == 4 and png["missing_reference"] == 0 and png["mean_abs_dlogit"] > 0 and -1 <= png["pearson_r"] <= 1
    assert check["input_gap"]["mean_abs_rgb_units"] > 0
    assert check["roles"] == {"matches": 3, "shots": 4, "match_roles": {"training": 3}, "fold_source_is_the_xgv_fold": True,
                              "passed": True}
    assert scorer.run(scorer.parse_args([*world["argv"], "--spot-check"])) == 0


def test_table_check_finds_a_held_out_match_and_a_foreign_fold(world):
    table = pd.read_parquet(world["out"] / "predictions.parquet")
    assert scorer.xgt_table_check(table, world["files"]["index"], world["files"]["folds"])["passed"]
    held_out = scorer.xgt_table_check(table.assign(game=table.game.replace("B", "E")), world["files"]["index"], world["files"]["folds"])
    assert not held_out["passed"] and held_out["match_roles"] == {"training": 2, "evaluation": 1}
    foreign = scorer.xgt_table_check(table.assign(fold_source=(1 - table.fold_source).astype(np.int8)), world["files"]["index"],
                                     world["files"]["folds"])
    assert not foreign["passed"] and not foreign["fold_source_is_the_xgv_fold"]


def test_the_folder_is_the_g_xgv_source_of_xg_temporal(world):
    source = ti.Source("g_xgv", world["out"], "g_combo", True)
    files = ti.verify_source(source)
    frames = ti.read_frames(files, ["a1", "a2", "b1", "t1"], ti.Window().offsets)
    base = pd.DataFrame({"shot_id": ["a1", "a2", "b1", "t1"], "game": ["A", "A", "B", "T"], "fold": [0, 0, 1, 0],
                         "z_g_combo": 0.0}).set_index("shot_id")
    population = ti.population_from(base, frames, ti.Window(), source)
    assert population.counts["shots"] == 4 and population.counts["matches"] == 3
    with pytest.raises(ValueError, match="fold model that held out"):
        ti.population_from(base.assign(fold=1), frames, ti.Window(), source)


def test_rerun_verifies_the_arrays_outside_out(world, tmp_path, monkeypatch):
    out = tmp_path / "copy"
    shutil.copytree(world["out"], out)
    argv = [*world["argv"]]
    argv[argv.index("--out") + 1] = str(out)
    monkeypatch.setattr(scorer, "ModelBank", lambda *a, **k: pytest.fail("rescored a finished output"))
    assert scorer.run(scorer.parse_args(argv)) == 0
    emb = tmp_path / "emb"
    shutil.copytree(world["emb"], emb)
    record = json.loads((out / "complete.json").read_text())
    for entry in record["embeddings"].values():
        entry["path"] = str(emb / Path(entry["path"]).name)
    (out / "complete.json").write_text(json.dumps(record))
    argv[argv.index("--embeddings-dir") + 1] = str(emb)
    assert all(Path(e["path"]).is_absolute() for e in json.loads((world["out"] / "complete.json").read_text())["embeddings"].values())
    with pytest.raises(ValueError, match="identity changed"):          # the folder is part of the identity
        scorer.run(scorer.parse_args(argv))


def test_a_changed_array_fails_verification(world, tmp_path):
    out, emb = tmp_path / "out", tmp_path / "emb"
    shutil.copytree(world["out"], out)
    shutil.copytree(world["emb"], emb)
    record = json.loads((out / "complete.json").read_text())
    for entry in record["embeddings"].values():
        entry["path"] = str(emb / Path(entry["path"]).name)
    (out / "complete.json").write_text(json.dumps(record))
    scorer.verify_complete(out, record["identity"])
    array = emb / "embeddings_g_combo_seed1.npy"
    array.write_bytes(array.read_bytes()[:-1] + b"\1")
    with pytest.raises(ValueError, match="changed since the run completed"):
        scorer.verify_complete(out, record["identity"])


# --- checks before any scoring -------------------------------------------------------------------------------------------------

def test_check_only_prints_the_plan_and_writes_nothing(world, tmp_path, capsys):
    out, emb = tmp_path / "out", tmp_path / "emb"
    assert scorer.run(scorer.parse_args(xgt_args(world, out, emb, "--check-only"))) == 0
    assert not out.exists() and not emb.exists()
    text = capsys.readouterr().out
    summary = json.loads(text[text.index("{"):])
    plan = build_plan(world["shots"], world["frames"], pd.read_csv(world["files"]["folds"]), world["files"]["frames"], 1, ti.GRID,
                      test_group=False)
    assert summary["layout"] == "xgt" and summary["encoders"] == ["g_combo"] and summary["shots"] == 4
    assert summary["frames_to_score"] == int(plan.has_frame.sum()) and summary["tail_passes"] == 2 * summary["frames_to_score"]
    assert summary["shots_per_group"] == {"fold0": 3, "fold1": 1} and summary["official_split"]["test"] == 1
    assert list(summary["embedding_arrays_gib"]) == ["g_combo"] and len(summary["bundles"]["g_combo"]) == 4
    assert summary["xgt"]["roles"] == {"training": 4} and summary["folds_sha256"] == sha256_file(world["files"]["folds"])


def test_a_shot_outside_the_training_role_or_a_bundle_of_another_plan_is_refused(world, tmp_path, monkeypatch):
    out, emb = tmp_path / "out", tmp_path / "emb"
    roles = world["roles"].assign(role=lambda d: np.where(d.game.eq("B"), "evaluation", d.role))
    monkeypatch.setattr(scorer.a1, "verify_plan", lambda index, folds: roles)
    with pytest.raises(ValueError, match="outside the plan's training role"):
        scorer.run(scorer.parse_args(xgt_args(world, out, emb, "--check-only")))
    monkeypatch.setattr(scorer.a1, "verify_plan", lambda index, folds: world["roles"])
    monkeypatch.setattr(scorer.a1, "PLAN_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="not trained under the xgv split plan"):
        scorer.run(scorer.parse_args(xgt_args(world, out, emb, "--check-only")))


def test_a_subset_is_not_written_to_the_default_folders_and_seed_0_is_required(world, tmp_path):
    with pytest.raises(SystemExit, match="must not be written"):
        scorer.run(scorer.parse_args(xgt_args(world, scorer.XGT_OUT, tmp_path / "emb")))
    with pytest.raises(SystemExit, match="must not be written"):
        scorer.run(scorer.parse_args(xgt_args(world, tmp_path / "out", scorer.XGT_EMBEDDINGS)))
    with pytest.raises(SystemExit):
        scorer.parse_args(["--layout", "xgt", "--embed-seeds", "1"])
    args = scorer.parse_args(["--layout", "xgt"])
    assert (args.out, args.embeddings_dir, args.shots_csv, args.offsets) == (
        scorer.XGT_OUT, scorer.XGT_EMBEDDINGS, scorer.XGT_SHOTS, GRID)
    assert args.g_combo == ROOT / "results/xgv_g_combo" and args.folds == scorer.a1.FOLDS and args.embed_seeds == (0,)
    xgc = scorer.parse_args([])
    assert xgc.layout == "xgc" and xgc.out == scorer.OUT and xgc.embeddings_dir == scorer.OUT


def test_reference_is_read_with_shot_id_and_z_g_combo_only(world, monkeypatch):
    seen = []
    real = pd.read_csv
    monkeypatch.setattr(scorer.pd, "read_csv", lambda path, **k: seen.append(k.get("usecols")) or real(path, **k))
    z, record = scorer.xgt_reference(world["files"]["oof"])
    assert seen == [["shot_id", "z_g_combo"]] and z.name == "z_g_combo" and record["columns"] == ["shot_id", "z_g_combo"]
    oof = world["files"]["oof"]
    copy = oof.parent.parent / "oof_changed"
    copy.mkdir(exist_ok=True)
    shutil.copy(oof, copy / "oof.csv")
    shutil.copy(oof.parent / "complete.json", copy / "complete.json")
    (copy / "oof.csv").write_text((copy / "oof.csv").read_text() + "x9,0,0.0,1\n")
    with pytest.raises(ValueError, match="completion record"):
        scorer.xgt_reference(copy / "oof.csv")


# --- units ---------------------------------------------------------------------------------------------------------------------

def test_plan_without_test_group_puts_every_shot_in_its_fold(world):
    folds = pd.read_csv(world["files"]["folds"])
    plan = build_plan(world["shots"], world["frames"], folds, world["files"]["frames"], 1, ti.GRID, test_group=False)
    groups = plan.drop_duplicates("shot_id").set_index("shot_id").group
    assert groups.to_dict() == {"a1": "fold0", "a2": "fold0", "t1": "fold0", "b1": "fold1"}
    assert (plan.emb_row_g[plan.has_frame] >= 0).all()
    with pytest.raises(ValueError, match="games have no fold"):
        build_plan(world["shots"], world["frames"], folds[folds.game.ne("T")], world["files"]["frames"], 1, ti.GRID, test_group=False)
    with pytest.raises(ValueError, match="test game"):                       # the xgc layout keeps its rule
        build_plan(world["shots"], world["frames"], folds, world["files"]["frames"], 1, ti.GRID)


def test_reference_check_reports_mean_max_and_correlation():
    table = pd.DataFrame({"shot_id": ["a", "b", "c", "d"], "encoder": "g_combo", "offset_ms": [0, 0, 0, -200],
                          "frame_status": ["ok", "ok", "ok", "ok"], "xg_logit": [1.0, 2.0, 3.0, 9.0]})
    exact = reference_check(table, pd.Series({"a": 1.0, "b": 2.0, "c": 3.0}))
    assert exact["passed"] and exact["shots"] == 3 and exact["max_abs_dlogit"] == 0 and exact["pearson_r"] == pytest.approx(1)
    off = reference_check(table, pd.Series({"a": 1.1, "b": 2.0, "c": 2.9}))
    assert not off["passed"] and off["max_abs_dlogit"] == pytest.approx(0.1) and off["mean_abs_dlogit"] == pytest.approx(0.2 / 3)
    missing = reference_check(table, pd.Series({"a": 1.0, "b": 2.0}))
    assert not missing["passed"] and missing["missing_reference"] == 1
    json.dumps(reference_check(table.iloc[:0], pd.Series(dtype=float)), allow_nan=False)
