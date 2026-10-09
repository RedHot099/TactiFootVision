"""The per-shot prediction table: plan, out-of-fold rule, resume, completion record, guards, and the end-to-end smoke
on a dummy index (tiny clips -> offset frames -> tiny fold models -> table). CPU, ViT-S with random weights."""

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from xgpoc.bundle import make_provenance, save_bundle, sha256_file
from xgpoc.data.offset_frames import GRID_MS
from xgpoc.models import g_bundle
from xgpoc.models.d_foundation import FinetuneViTXG
from xgpoc.probe.table import (
    ENCODERS,
    TABLE_COLUMNS,
    TEST_GROUP,
    build_plan,
    chunk_table,
    spot_check,
    stored_reference,
    validate_table,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_offset_frames as builder  # noqa: E402
import score_offset_frames as scorer  # noqa: E402
from test_offset_frames import write_clip  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
N_TRAINED = 2
SHOTS = {  # shot -> (game, split, goal, penalty, clip frames)
    "a1": ("A", "train", False, False, 200), "a2": ("A", "train", True, False, 200), "ap": ("A", "train", True, True, 200),
    "b1": ("B", "valid", False, False, 200), "b2": ("B", "valid", True, False, 120),
    "t1": ("T", "test", False, False, 200), "t2": ("T", "test", True, False, 200)}
GAME_FOLD = {"A": 0, "B": 1}
CHECKED_OFFSETS = (-1600, -40, 0, 600, 800)   # the reference forward path is slow on CPU; the table is checked at these
NAMES = {"g_combo": "tiny_combo", "g_combo_no_head": "tiny_control"}


# --- fixtures: dummy index -> offset frames -> tiny fold models --------------------------------------------------------

def tiny_bundles(root: Path, encoder: str, spec: dict, files: dict, shots: pd.DataFrame) -> None:
    aux = scorer.AUX_WEIGHT[encoder]
    name = NAMES[encoder]
    provenance = make_provenance({"train.py": "aa"}, index_sha256=sha256_file(files["index"]), folds_sha256=sha256_file(files["folds"]),
                                 overlay_sha256=sha256_file(files["overlay"]))
    jpeg = pd.read_parquet(files["index"]).set_index("shot_id").frame_path    # the frames the models "were trained on"
    members = []
    for fold in (0, 1):
        for seed in (0, 1):
            run = root / f"fold{fold}" / f"seed{seed}"
            run.mkdir(parents=True)
            torch.manual_seed(0)
            model, _ = g_bundle.build_g(spec)
            generator = torch.Generator().manual_seed(1000 * fold + 10 * seed + (1 if aux else 2))
            with torch.no_grad():
                for p in model.parameters():
                    if p.requires_grad:
                        p.add_(0.03 * torch.randn(p.shape, generator=generator))
            model.eval()
            temperature = 1.1 + 0.2 * seed + 0.05 * fold
            save_bundle(run / "bundle.pt", model, spec=spec, backbone=g_bundle.g_identity(spec), commit={"sha": "x", "dirty": False},
                        calibration={"temperature": temperature, "penalty_rate": 0.5},
                        recipe={"cfg": {"aux_weight": aux, "batch_size": 4}, "fold": fold, "seed": seed, "best_epoch": 0},
                        provenance=provenance)
            written = ["bundle.pt"]
            for role, part in (("valid", shots[shots.game.map(GAME_FOLD).eq(fold)]), ("test", shots[shots.split.eq("test")])):
                df = pd.DataFrame({"shot_id": part.shot_id.to_numpy(), "goal": 0, "frame_path": [jpeg[s] for s in part.shot_id]})
                xg, _, _ = g_bundle.g_outputs(model, df, spec, device="cpu", batch_size=4, num_workers=0)
                pd.DataFrame({"shot_id": df.shot_id, "goal": 0, "logit": xg, "p": 1 / (1 + np.exp(-xg / temperature))}).to_csv(
                    run / f"{name}_{role}_predictions.csv", index=False)
                written.append(f"{name}_{role}_predictions.csv")
            record = {"identity": {"fold": fold, "seed": seed, "name": name, "index": {"sha256": sha256_file(files["index"])},
                                   "folds": {"sha256": sha256_file(files["folds"])},
                                   "overlay_sha256": sha256_file(files["overlay"])},
                      "code_sha256": {"train.py": "aa"}, "files": {f: sha256_file(run / f) for f in written}}
            (run / "complete.json").write_text(json.dumps(record))
            members.append({"fold": fold, "seed": seed, "sha256": sha256_file(run / "bundle.pt"), "temperature": temperature})
    pd.DataFrame(members)[["fold", "seed", "sha256"]].to_csv(root / "manifest.tsv", sep="\t", index=False)
    (root / "ensemble.json").write_text(json.dumps({"name": name, "members": members}))
    (root / "gate.json").write_text(json.dumps({"passed": True if encoder == "g_combo" else None}))


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Dummy index, offset frames (model-input storage), folds file and tiny fold models of both encoders; the shared
    scoring run (`out`) is done once. Patches are active for the whole module."""
    tmp = tmp_path_factory.mktemp("offset_scores")
    with pytest.MonkeyPatch.context() as mp:
        weights = tmp / "model.safetensors"
        weights.write_bytes(b"stand-in for the pretrained checkpoint")
        mp.setattr(g_bundle, "pretrained_weight_file", lambda backbone: weights)

        def build(*, backbone, n_blocks, n_out, pretrained):
            torch.manual_seed(0)   # the same frozen weights every time, as a checkpoint would give
            return FinetuneViTXG(backbone, n_blocks=n_blocks, n_out=n_out, pretrained=False)

        mp.setattr(g_bundle, "FinetuneViTXG", build)
        mp.setattr(scorer, "own_sources", lambda: {"xgpoc/probe/scoring.py": "fixed", "xgpoc/probe/table.py": "fixed"})
        rows = []
        for i, (shot, (game, split, goal, penalty, frames)) in enumerate(SHOTS.items()):
            clip = write_clip(tmp / f"{shot}.mp4", frames=frames, seed=10 + i, cut_at=80 if shot == "a2" else None)
            capture = cv2.VideoCapture(str(clip))
            for _ in range(101):
                ok, frame = capture.read()
            assert ok
            jpeg = tmp / f"{shot}.jpg"
            cv2.imwrite(str(jpeg), frame)
            rows.append(dict(shot_id=shot, frame_path=str(jpeg), game=game, split=split, label="Shots on target" if i % 2 else "Shots off target",
                             goal=goal, is_penalty=penalty, half=1 + i % 2, position_ms=90000, clip_path=str(clip)))
        index = tmp / "index.parquet"
        pd.DataFrame(rows).to_parquet(index)
        protocol = tmp / "protocol.md"
        protocol.write_text("approved protocol")
        frames_dir = tmp / "offset_frames"
        builder.build(index, frames_dir, "model_input_png", 2, protocol=protocol)
        folds = tmp / "folds.csv"
        pd.DataFrame({"game": ["A", "B"], "home": ["A", "B"], "fold": [0, 1]}).to_csv(folds, index=False)
        files = {"index": index, "folds": folds, "overlay": ROOT / "configs/overlay_mask.png", "frames": frames_dir,
                 "protocol": protocol}
        spec = g_bundle.g_spec("dinov2_s", N_TRAINED, 2, "overlay")
        shots, frames = pd.read_csv(frames_dir / "shots.csv"), pd.read_csv(frames_dir / "frames.csv")
        for encoder in NAMES:
            tiny_bundles(tmp / encoder, encoder, spec, files, shots)
        out = tmp / "out"
        argv = common_args(files, tmp, out)
        assert scorer.run(scorer.parse_args(argv)) == 0
        yield {"tmp": tmp, "files": files, "spec": spec, "out": out, "argv": argv, "shots": shots, "frames": frames}


def common_args(files: dict, tmp: Path, out: Path, *extra: str) -> list[str]:
    return ["--out", str(out), "--index", str(files["index"]), "--offset-frames", str(files["frames"]),
            "--g-combo", str(tmp / "g_combo"), "--head-free", str(tmp / "g_combo_no_head"), "--folds", str(files["folds"]),
            "--overlay", str(files["overlay"]), "--protocol", str(files["protocol"]), "--expect-models", "4", "--batch-size", "3",
            "--chunk-shots", "1", "--workers", "0", "--device", "cpu", *extra]


def reference_scores(world, encoder: str, model_keys) -> tuple[pd.DataFrame, dict]:
    """Raw xG logit, on-target logit and embedding of the listed (fold, seed) models on every stored frame, through the
    one forward path of the bundles (`g_outputs`)."""
    frames = world["frames"]
    frames = frames[frames.status.isin(["ok", "anchor_off_tolerance"]) & frames.offset_ms.isin(CHECKED_OFFSETS)]
    df = pd.DataFrame({"shot_id": frames.shot_id.to_numpy(), "offset_ms": frames.offset_ms.to_numpy(), "goal": 0,
                       "frame_path": [str(world["files"]["frames"] / p) for p in frames.frame_path]})
    out = {}
    for fold, seed in model_keys:
        loaded = g_bundle.load_g(world["tmp"] / encoder / f"fold{fold}" / f"seed{seed}" / "bundle.pt")
        xg, on_target, embedding = g_bundle.g_outputs(loaded.model, df, world["spec"], device="cpu", batch_size=4, num_workers=0)
        out[(fold, seed)] = (xg, on_target, embedding)
    return df, out


# --- end to end ------------------------------------------------------------------------------------------------------------

def test_table_follows_the_out_of_fold_rule_and_matches_the_bundles_forward_path(world):
    table = pd.read_parquet(world["out"] / "predictions.parquet")
    population = [s for s, v in SHOTS.items() if not v[3]]
    assert list(table.columns) == TABLE_COLUMNS and len(table) == len(population) * 30 * 3
    assert "ap" not in set(table.shot_id)                                       # the penalty is not scored
    assert set(table.encoder) == set(ENCODERS) and table.oof.all()
    for encoder in ("g_combo", "g_combo_no_head"):
        df, ref = reference_scores(world, encoder, [(f, s) for f in (0, 1) for s in (0, 1)])
        temperatures = {k: 1.1 + 0.2 * k[1] + 0.05 * k[0] for k in ref}
        part = table[table.encoder.eq(encoder)].set_index(["shot_id", "offset_ms"])
        for row in df[~df.shot_id.map(lambda shot: SHOTS[shot][3])].itertuples():
            r = part.loc[(row.shot_id, row.offset_ms)]
            test = SHOTS[row.shot_id][1] == "test"
            keys = list(ref) if test else [(GAME_FOLD[SHOTS[row.shot_id][0]], s) for s in (0, 1)]   # never the held-in fold
            i = row.Index
            raw = np.mean([ref[k][0][i] for k in keys])
            assert r.xg_logit_raw == pytest.approx(raw, abs=1e-4)
            assert r.xg_logit == pytest.approx(np.mean([ref[k][0][i] / temperatures[k] for k in keys]), abs=1e-4)
            assert r.n_models == len(keys) and r.fold_source == (-1 if test else keys[0][0])
            assert r.seed_source == "0,1" and r.oof
            if encoder == "g_combo":
                assert r.on_target_logit == pytest.approx(np.mean([ref[k][1][i] for k in keys]), abs=1e-4)
            else:
                assert np.isnan(r.on_target_logit)
            if test:
                assert r.embedding_ref == -1
            else:
                embedding = np.load(world["out"] / f"embeddings_{encoder}.npy", mmap_mode="r")[r.embedding_ref]
                assert embedding.dtype == np.float16
                np.testing.assert_allclose(embedding, ref[(keys[0][0], 0)][2][i], rtol=2e-3, atol=2e-3)   # seed-0 model of the fold


def test_missing_frames_stay_in_the_table_and_the_frozen_encoder_has_its_own_layout(world):
    table = pd.read_parquet(world["out"] / "predictions.parquet")
    short = table[table.shot_id.eq("b2") & table.encoder.eq("g_combo")].set_index("offset_ms")
    assert short.frame_status[[800, 1000]].tolist() == ["missing", "missing"] and not short.complete_case.any()
    assert short.loc[[800, 1000], ["xg_logit", "xg_logit_raw", "on_target_logit"]].isna().all().all()
    assert (short.loc[[800, 1000], "embedding_ref"] == -1).all() and short.decoded_relative_ms[[800, 1000]].isna().all()
    assert short.xg_logit.drop([800, 1000]).notna().all()
    frozen = table[table.encoder.eq("dinov2_frozen")]
    assert frozen.xg_logit.isna().all() and frozen.on_target_logit.isna().all()
    assert (frozen.fold_source == -1).all() and (frozen.n_models == 0).all() and (frozen.seed_source == "").all()
    refs = frozen.embedding_ref[frozen.embedding_ref >= 0]
    assert sorted(refs) == list(range(len(refs))) and len(refs) == (frozen.frame_status == "ok").sum()
    embeddings = np.load(world["out"] / "embeddings_dinov2_frozen.npy", mmap_mode="r")
    assert embeddings.shape == (len(refs), 2 * 384) and embeddings.dtype == np.float16 and np.isfinite(embeddings).all()
    complete = table[table.encoder.eq("g_combo")].groupby("shot_id").complete_case.first()
    assert complete.to_dict() == {"a1": True, "a2": True, "b1": True, "b2": False, "t1": True, "t2": True}


def test_cut_flag_and_decoded_time_come_from_the_offset_frames(world):
    table = pd.read_parquet(world["out"] / "predictions.parquet")
    one = table[table.encoder.eq("g_combo")].merge(world["frames"], on=["shot_id", "offset_ms"])
    ok = one[one.frame_status.eq("ok")]
    assert (ok.decoded_relative_ms_x == ok.decoded_relative_ms_y).all()
    assert ok.cut_between_anchor_x.astype(bool).eq(ok.cut_between_anchor_y.astype(bool)).all()
    flagged = ok[ok.cut_between_anchor_x.astype(bool)]
    assert set(flagged.shot_id) == {"a2"} and flagged.offset_ms.max() < -800 and len(flagged) == 4   # the cut at -800 ms: -1600..-1000


def test_completion_record_holds_hashes_provenance_protocol_and_the_spot_check(world):
    out, files = world["out"], world["files"]
    record = json.loads((out / "complete.json").read_text())
    assert record["identity"]["protocol"]["sha256"] == sha256_file(files["protocol"])
    assert record["identity"]["protocol"]["approval"]["commit"] == "41dfd12"
    assert record["identity"]["index_sha256"] == sha256_file(files["index"])
    assert record["identity"]["folds_sha256"] == sha256_file(files["folds"])
    assert record["identity"]["offset_frames"]["complete_json_sha256"] == sha256_file(files["frames"] / "complete.json")
    assert [m["sha256"] for m in record["identity"]["bundles"]["g_combo"]] == [
        sha256_file(world["tmp"] / "g_combo" / f"fold{f}" / f"seed{s}" / "bundle.pt") for f in (0, 1) for s in (0, 1)]
    assert record["identity"]["bundles"]["g_combo"][0]["data"]["index_sha256"] == sha256_file(files["index"])
    assert set(record["files"]) == {"predictions.parquet", *(f"embeddings_{e}.npy" for e in ENCODERS)}
    assert all(sha256_file(out / name) == h for name, h in record["files"].items())
    check = record["spot_check"]
    assert check["pipeline_passed"] and check["index_jpeg_control"]["g_combo"]["test"]["shots"] == 2
    assert check["index_jpeg_control"]["g_combo_no_head"]["train_valid"]["max_abs_dlogit"] < 1e-3
    assert check["index_jpeg_control"]["shots"] == 6 and check["input_gap"]["mean_abs_rgb_units"] > 0
    assert set(check["offset0_vs_stored"]) == {"g_combo", "g_combo_no_head", "passed"} and "offset0_passed" in check
    assert record["identity"]["plan"]["full"] is True                          # every shot and offset of the dummy index
    assert not (out / "chunks").exists() and not (out / "identity.json").exists()
    assert record["summary"]["rows"] == 6 * 30 * 3 and record["summary"]["complete_case_shots"] == 5


def test_rerun_verifies_instead_of_rescoring_and_detects_changes(world, tmp_path, monkeypatch):
    out = tmp_path / "copy"
    shutil.copytree(world["out"], out)
    argv = common_args(world["files"], world["tmp"], out)
    monkeypatch.setattr(scorer, "ModelBank", lambda *a, **k: pytest.fail("rescored a finished output"))
    assert scorer.run(scorer.parse_args(argv)) == 0
    with pytest.raises(ValueError, match="identity changed"):
        scorer.run(scorer.parse_args([*argv, "--batch-size", "5"]))
    (out / "embeddings_dinov2_frozen.npy").write_bytes((out / "embeddings_dinov2_frozen.npy").read_bytes()[:-1] + b"\0")
    with pytest.raises(ValueError, match="changed since the run completed"):
        scorer.run(scorer.parse_args(argv))
    (out / "stray.txt").write_text("x")
    with pytest.raises(ValueError, match="differ from its completion record"):
        scorer.run(scorer.parse_args(argv))


def test_interrupted_run_resumes_to_the_same_table_and_refuses_a_changed_identity(world, tmp_path):
    out = tmp_path / "partial"
    argv = common_args(world["files"], world["tmp"], out)
    assert scorer.run(scorer.parse_args([*argv, "--max-chunks", "2"])) == 0
    committed = sorted((out / "chunks").glob("chunk_*.json"))
    assert len(committed) == 2 and not (out / "complete.json").exists()
    with pytest.raises(ValueError, match="identity changed"):
        scorer.run(scorer.parse_args([*argv, "--batch-size", "5"]))
    damaged = (out / "chunks" / "chunk_00001.parquet")
    damaged.write_bytes(damaged.read_bytes() + b"torn")                          # a write that did not finish
    assert scorer.run(scorer.parse_args(argv)) == 0
    expected = pd.read_parquet(world["out"] / "predictions.parquet")
    pd.testing.assert_frame_equal(pd.read_parquet(out / "predictions.parquet"), expected)
    for encoder in ENCODERS:
        a, b = (np.load(d / f"embeddings_{encoder}.npy") for d in (out, world["out"]))
        np.testing.assert_array_equal(a, b)


def test_resume_does_not_redo_committed_chunks(world, tmp_path, monkeypatch):
    out = tmp_path / "half"
    argv = common_args(world["files"], world["tmp"], out)
    scorer.run(scorer.parse_args([*argv, "--max-chunks", "3"]))
    calls = []
    real = scorer.score_chunk
    monkeypatch.setattr(scorer, "score_chunk", lambda rows, *a, **k: calls.append(int(rows.chunk.iloc[0])) or real(rows, *a, **k))
    scorer.run(scorer.parse_args(argv))
    assert calls[:3] == [3, 4, 5] and len(calls) == 3 + 3        # 6 chunks of one shot, 0-2 were committed; then the 3 groups' control


def test_a_foreign_directory_and_a_foreign_identity_are_not_continued(world, tmp_path):
    out = tmp_path / "foreign"
    out.mkdir()
    (out / "something.txt").write_text("x")
    with pytest.raises(ValueError, match="no run identity"):
        scorer.run(scorer.parse_args(common_args(world["files"], world["tmp"], out)))


# --- guards ----------------------------------------------------------------------------------------------------------------

def test_gpu_needs_approval_and_a_free_gpu(world, tmp_path, monkeypatch):
    argv = common_args(world["files"], world["tmp"], tmp_path / "gpu")
    argv[argv.index("--device") + 1] = "cuda"
    with pytest.raises(SystemExit, match="gpu-approved"):
        scorer.run(scorer.parse_args(argv))
    monkeypatch.setattr(scorer, "check_gpu_free", lambda: (_ for _ in ()).throw(SystemExit("another python process is using the GPU")))
    with pytest.raises(SystemExit, match="another python process"):
        scorer.run(scorer.parse_args([*argv, "--gpu-approved"]))
    assert not (tmp_path / "gpu").exists()


def test_stop_rule_of_the_offset_frames_stops_the_scoring(world, tmp_path, monkeypatch):
    clips = [world["tmp"] / "a1.mp4", tmp_path / "broken.mp4"]
    clips[1].write_bytes(b"not a video")
    index = tmp_path / "index.parquet"
    pd.DataFrame([dict(shot_id=f"s{i}", game="A", split="train", label="Shots on target", goal=False, is_penalty=False, half=1,
                       position_ms=90000, clip_path=str(c)) for i, c in enumerate(clips)]).to_parquet(index)
    monkeypatch.setattr(builder, "MAX_FAILED_FRACTION", 1.0)
    builder.build(index, tmp_path / "frames", "model_input_png", 2, protocol=world["files"]["protocol"])
    with pytest.raises(SystemExit, match="stop rule"):
        scorer.load_offset_frames(tmp_path / "frames", world["files"]["protocol"], index)


def test_offset_frames_must_match_protocol_index_and_files(world, tmp_path):
    frames_dir, protocol, index = world["files"]["frames"], world["files"]["protocol"], world["files"]["index"]
    scorer.load_offset_frames(frames_dir, protocol, index)
    other = tmp_path / "protocol.md"
    other.write_text("a different protocol")
    with pytest.raises(ValueError, match="another protocol"):
        scorer.load_offset_frames(frames_dir, other, index)
    other_index = tmp_path / "index.parquet"
    shutil.copy(index, other_index)
    with pytest.raises(ValueError, match="another index"):
        scorer.load_offset_frames(frames_dir, protocol, other_index)
    copy = tmp_path / "frames"
    shutil.copytree(frames_dir, copy)
    (copy / "frames.csv").write_text((copy / "frames.csv").read_text() + "\n")
    with pytest.raises(ValueError, match="changed since"):
        scorer.load_offset_frames(copy, protocol, index)


def test_bundle_checks_catch_swapped_changed_and_foreign_models(world, tmp_path):
    tmp, files = world["tmp"], world["files"]
    folds = files["folds"]
    ok = scorer.verify_bundles(tmp / "g_combo", "g_combo", files["index"], folds, files["overlay"])
    assert [(m["fold"], m["seed"]) for m in ok] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    with pytest.raises(ValueError, match="auxiliary weight"):
        scorer.verify_bundles(tmp / "g_combo", "g_combo_no_head", files["index"], folds, files["overlay"])
    with pytest.raises(ValueError, match="reproduction gate"):
        scorer.verify_bundles(tmp / "g_combo_no_head", "g_combo", files["index"], folds, files["overlay"])
    with pytest.raises(ValueError, match="reproduction gate"):
        scorer.verify_bundles(_gate_failed(tmp, tmp_path), "g_combo", files["index"], folds, files["overlay"])
    other_folds = tmp_path / "other_folds.csv"
    pd.DataFrame({"game": ["A", "B"], "home": ["A", "B"], "fold": [1, 0]}).to_csv(other_folds, index=False)
    with pytest.raises(Exception, match="folds_sha256"):
        scorer.verify_bundles(tmp / "g_combo", "g_combo", files["index"], other_folds, files["overlay"])
    copy = tmp_path / "g_combo"
    shutil.copytree(tmp / "g_combo", copy)
    data = (copy / "fold0" / "seed0" / "bundle.pt").read_bytes()
    (copy / "fold0" / "seed0" / "bundle.pt").write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    with pytest.raises(ValueError, match="changed since it was written"):
        scorer.verify_bundles(copy, "g_combo", files["index"], folds, files["overlay"])


def _gate_failed(tmp: Path, tmp_path: Path) -> Path:
    copy = tmp_path / "gate_failed"
    shutil.copytree(tmp / "g_combo", copy)
    (copy / "gate.json").write_text(json.dumps({"passed": False}))
    return copy


def test_layout_check_wants_the_folds_of_the_folds_file_and_one_model_per_fold_and_seed(world):
    tmp, files = world["tmp"], world["files"]
    models = {e: scorer.verify_bundles(tmp / e, e, files["index"], files["folds"], files["overlay"]) for e in NAMES}
    folds = pd.read_csv(files["folds"])
    spec, backbone, n_seeds = scorer.check_layout(models, folds, 4)
    assert n_seeds == 2 and spec == world["spec"] and backbone["sha256"]
    with pytest.raises(ValueError, match="expected 10"):
        scorer.check_layout(models, folds, 10)
    with pytest.raises(ValueError, match="folds differ"):
        scorer.check_layout(models, pd.DataFrame({"game": ["A", "B", "C"], "fold": [0, 1, 2]}), 4)
    short = {**models, "g_combo": [m for m in models["g_combo"] if m["seed"] == 0]}
    with pytest.raises(ValueError, match="expected 4"):
        scorer.check_layout(short, folds, 4)
    mixed = {**models, "g_combo": [{**m, "backbone": {**m["backbone"], "sha256": "00"}} if m["fold"] == 1 else m for m in models["g_combo"]]}
    with pytest.raises(ValueError, match="share one architecture"):
        scorer.check_layout(mixed, folds, 4)


def test_spot_check_passes_at_batch_noise_and_fails_beyond_it(world):
    table = pd.read_parquet(world["out"] / "predictions.parquet")
    roots = {e: world["tmp"] / e for e in NAMES}
    for encoder, root in roots.items():                  # a table that carries exactly the stored predictions at offset 0
        pool, test, _ = stored_reference(root)
        zero = table.encoder.eq(encoder) & table.offset_ms.eq(0) & table.frame_status.eq("ok")
        rows = table[zero]
        table.loc[zero, "xg_logit_raw"] = np.where(rows.split.eq("test"), test.reindex(rows.shot_id), pool.reindex(rows.shot_id)).astype(np.float32)
    assert spot_check(table, roots)["passed"]
    off = table.copy()
    mask = off.encoder.eq("g_combo") & off.offset_ms.eq(0) & off.shot_id.eq("t1")
    off.loc[mask, "xg_logit_raw"] += 2e-3
    report = spot_check(off, roots)
    assert not report["passed"] and not report["g_combo"]["passed"] and report["g_combo_no_head"]["passed"]
    assert report["g_combo"]["test"]["max_abs_dlogit"] == pytest.approx(2e-3, abs=1e-4)
    assert spot_check(off, roots, tolerance=5e-3)["passed"]
    assert not spot_check(table[table.offset_ms.ne(0)], roots)["passed"]        # nothing compared is not a pass
    assert scorer.run(scorer.parse_args([*world["argv"], "--spot-check"])) == 0


# --- plan and table units ----------------------------------------------------------------------------------------------------

def test_plan_orders_folds_then_test_assigns_embedding_rows_from_the_plan_alone(world):
    shots, frames, folds = world["shots"], world["frames"], pd.read_csv(world["files"]["folds"])
    plan = build_plan(shots, frames, folds, world["files"]["frames"], chunk_shots=1)
    assert plan.group.drop_duplicates().tolist() == ["fold0", "fold1", TEST_GROUP]
    assert plan.drop_duplicates("shot_id").shot_id.tolist() == ["a1", "a2", "b1", "b2", "t1", "t2"]    # penalty "ap" left out
    assert plan.groupby("shot_id").offset_ms.apply(list).map(lambda o: o == list(GRID_MS)).all()
    assert plan.chunk.tolist() == sorted(plan.chunk) and plan.chunk.nunique() == 6
    assert plan.groupby("chunk").group.nunique().eq(1).all()
    big = build_plan(shots, frames, folds, world["files"]["frames"], chunk_shots=2)
    assert big.drop_duplicates("chunk").group.tolist() == ["fold0", "fold1", TEST_GROUP]
    frozen, g = plan.emb_row_frozen[plan.has_frame], plan.emb_row_g[plan.emb_row_g >= 0]
    assert frozen.tolist() == list(range(len(frozen))) and g.tolist() == list(range(len(g)))
    assert (plan.emb_row_g[plan.group.eq(TEST_GROUP)] == -1).all() and (plan.emb_row_g[~plan.has_frame] == -1).all()
    assert plan.has_frame.sum() == 6 * 30 - 2 and plan.frame_file[~plan.has_frame].eq("").all()
    with pytest.raises(ValueError, match="no fold"):
        build_plan(shots, frames, folds[folds.game.ne("B")], world["files"]["frames"], 1)
    with pytest.raises(ValueError, match="test game"):
        build_plan(shots, frames, pd.concat([folds, pd.DataFrame({"game": ["T"], "home": ["T"], "fold": [0]})]), world["files"]["frames"], 1)
    with pytest.raises(ValueError, match="every grid offset"):
        build_plan(shots, frames[frames.offset_ms.ne(0)], folds, world["files"]["frames"], 1)


def test_validate_table_rejects_what_the_contract_forbids(world):
    shots, frames, folds = world["shots"], world["frames"], pd.read_csv(world["files"]["folds"])
    plan = build_plan(shots, frames, folds, world["files"]["frames"], 1)
    table = pd.read_parquet(world["out"] / "predictions.parquet")
    validate_table(table, plan, 4, 2)
    first_scored = table.index[table.encoder.eq("g_combo") & table.frame_status.eq("ok")][0]
    in_fold0 = table.encoder.eq("g_combo") & table.fold_source.eq(0)
    broken = {
        "a missing row": ("table has", table.iloc[1:]),
        "a duplicate": ("duplicate", pd.concat([table.iloc[:-1], table.iloc[[0]]], ignore_index=True)),
        "a NaN logit on a scored frame": ("finite exactly", table.assign(xg_logit=table.xg_logit.where(table.index != first_scored, np.nan))),
        "a logit on the frozen encoder": ("must be empty", table.assign(xg_logit=np.where(table.encoder.eq("dinov2_frozen"), 1.0, table.xg_logit))),
        "an embedding on a test shot": ("embedding rows", table.assign(embedding_ref=np.where(
            table.encoder.eq("g_combo") & table.split.eq("test"), 0, table.embedding_ref))),
        "a gap in the embedding rows": ("without gaps", table.assign(embedding_ref=np.where(
            table.embedding_ref.eq(1) & table.encoder.eq("dinov2_frozen"), 7, table.embedding_ref))),
        "a not-out-of-fold row": ("not out-of-fold", table.assign(oof=False)),
        "a test shot scored by two models": ("all 4 models", table.assign(n_models=np.where(
            table.split.eq("test") & table.encoder.eq("g_combo"), 2, table.n_models).astype(np.int8))),
        "a shot scored by the fold that holds it in": ("another fold", table.assign(fold_source=np.where(in_fold0, 1, table.fold_source).astype(np.int8))),
        "another match on a row": ("follow the plan", table.assign(game=np.where(table.index == first_scored, "Z", table.game))),
        "another column order": ("columns", table[list(reversed(TABLE_COLUMNS))]),
    }
    for what, (message, bad) in broken.items():
        with pytest.raises(ValueError, match=message):
            validate_table(bad, plan, 4, 2)
            pytest.fail(f"accepted {what}")


def test_chunk_table_fills_nan_and_minus_one_where_there_is_no_frame(world):
    plan = build_plan(world["shots"], world["frames"], pd.read_csv(world["files"]["folds"]), world["files"]["frames"], 1)
    rows = plan[plan.shot_id.eq("b2")]
    n = int(rows.has_frame.sum())
    values = {e: {"xg_logit": np.ones(n), "xg_logit_raw": np.ones(n), "on_target_logit": np.ones(n) if e == "g_combo" else None,
                  "fold_source": 1, "seed_source": "0,1", "n_models": 2} for e in ("g_combo", "g_combo_no_head")}
    values["dinov2_frozen"] = {"fold_source": -1, "seed_source": "", "n_models": 0}
    table = chunk_table(rows, values)
    assert len(table) == 90 and table.encoder.tolist()[:3] == list(ENCODERS)      # rows interleave: offset, then encoder
    assert table[table.encoder.eq("g_combo")].xg_logit.notna().sum() == n
    assert (table.embedding_ref[table.encoder.eq("g_combo") & table.frame_status.eq("missing")] == -1).all()
    assert table.dtypes["offset_ms"] == np.int16 and table.dtypes["xg_logit"] == np.float32


def test_disk_floor_stops_a_start_that_would_not_leave_five_gib(world, tmp_path, monkeypatch):
    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(scorer.shutil, "disk_usage", lambda p: type(usage)(usage.total, usage.total - 1, 1024))
    with pytest.raises(OSError, match="5 GiB"):
        scorer.run(scorer.parse_args(common_args(world["files"], world["tmp"], tmp_path / "full")))
    assert not (tmp_path / "full" / "identity.json").exists()


# --- review fixes: GPU never touched by checks, subsets, missing frames, unusable references --------------------------------

def test_check_only_and_verification_need_no_gpu_approval(world, tmp_path, monkeypatch):
    monkeypatch.setattr(scorer, "check_gpu_free", lambda: pytest.fail("the GPU guard ran for a check that scores nothing"))
    cuda = [*common_args(world["files"], world["tmp"], tmp_path / "unused"), "--check-only"]
    cuda[cuda.index("--device") + 1] = "cuda"
    assert scorer.run(scorer.parse_args(cuda)) == 0
    assert not (tmp_path / "unused").exists()
    finished = tmp_path / "finished"
    shutil.copytree(world["out"], finished)
    argv = common_args(world["files"], world["tmp"], finished)
    argv[argv.index("--device") + 1] = "cuda"                                  # scored on cpu, verified without a GPU
    assert scorer.run(scorer.parse_args(argv)) == 0


def test_a_smoke_subset_is_not_written_to_the_default_output(world):
    argv = common_args(world["files"], world["tmp"], scorer.OUT)
    with pytest.raises(SystemExit, match="smoke subset"):
        scorer.run(scorer.parse_args([*argv, "--offsets", "0"]))


def test_missing_or_resized_offset_frame_is_found_before_any_scoring(world, tmp_path):
    plan = build_plan(world["shots"], world["frames"], pd.read_csv(world["files"]["folds"]), world["files"]["frames"], 1)
    scorer.check_frame_files(plan)
    first = Path(plan[plan.has_frame].frame_file.iloc[0])
    data = first.read_bytes()
    try:
        first.write_bytes(data + b"x")
        with pytest.raises(ValueError, match="missing or have another size"):
            scorer.check_frame_files(plan)
        first.unlink()
        with pytest.raises(ValueError, match="missing or have another size"):
            scorer.check_frame_files(plan)
    finally:
        first.write_bytes(data)


def test_unusable_reference_is_reported_not_crashed(world):
    table = pd.read_parquet(world["out"] / "predictions.parquet").assign(shot_id=lambda d: "unknown_" + d.shot_id)
    report = spot_check(table, {"g_combo": world["tmp"] / "g_combo"})
    assert report["passed"] is False and report["g_combo"]["train_valid"]["max_abs_dlogit"] is None
    json.dumps(report, allow_nan=False)


def test_resume_redoes_a_chunk_whose_embeddings_were_torn(world, tmp_path):
    out = tmp_path / "torn"
    argv = common_args(world["files"], world["tmp"], out)
    scorer.run(scorer.parse_args([*argv, "--max-chunks", "2"]))
    array = np.load(out / "embeddings_g_combo.npy", mmap_mode="r+")
    array[0, :4] = 1.0                                                          # chunk 0's embedding rows, not its marker
    array.flush()
    del array
    scorer.run(scorer.parse_args(argv))
    expected = np.load(world["out"] / "embeddings_g_combo.npy")
    np.testing.assert_array_equal(np.load(out / "embeddings_g_combo.npy"), expected)
