"""Per-layer embeddings (xg-dino 05): the last block's normed output is the existing frozen embedding, the tail kinds are
each fold model's own full forward, blocks before the tail are shared, fold routing picks the shot's own fold model,
and the run script's plan, resume markers and StatsBomb guard. CPU, random weights, a ViT-S on small inputs."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from xgpoc.detector.guard import StatsBombRead, no_statsbomb_reads
from xgpoc.dino import layers
from xgpoc.models.d_foundation import FinetuneViTXG
from xgpoc.probe.scoring import SharedScorer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import xgdino_layers as script  # noqa: E402

N_TRAINED = 2


def model_with_seed(trunk_seed: int, tail_seed: int | None) -> FinetuneViTXG:
    """Same frozen weights for the same `trunk_seed`; `tail_seed` perturbs the trained part like fine-tuning would."""
    torch.manual_seed(trunk_seed)
    model = FinetuneViTXG("dinov2_s", n_blocks=N_TRAINED, n_out=2, pretrained=False).eval()
    if tail_seed is not None:
        generator = torch.Generator().manual_seed(tail_seed)
        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad:
                    p.add_(0.02 * torch.randn(p.shape, generator=generator))
    return model


@pytest.fixture(scope="module")
def images():
    return torch.randn(3, 3, 56, 98, generator=torch.Generator().manual_seed(0))


def encoder_with(folds_and_seeds):
    encoder = layers.LayerEncoder(model_with_seed(0, None), N_TRAINED, "cpu")
    models = {}
    for fold, tail_seed in folds_and_seeds:
        models[fold] = model_with_seed(0, tail_seed)
        encoder.add(fold, 0, models[fold])
    return encoder, models


def test_last_block_normed_output_is_the_existing_frozen_embedding(images):
    encoder, _ = encoder_with([])
    out = encoder.encode(images)
    vit = encoder.scorer.vit
    assert out["dino_blocks"].shape == (3, len(vit.blocks), 2, 384) and set(out) == {"dino_blocks", "dino_final"}
    _, frozen = SharedScorer(model_with_seed(0, None), N_TRAINED, "cpu").score(images, [], frozen=True)
    np.testing.assert_allclose(out["dino_final"].reshape(3, 768), frozen, atol=1e-6)
    with torch.no_grad():
        tokens = vit.forward_features(images)
        last_cls = vit.norm(torch.from_numpy(out["dino_blocks"][:, -1, 0]))
    np.testing.assert_allclose(out["dino_final"][:, 0], tokens[:, 0].numpy(), atol=1e-5)
    np.testing.assert_allclose(last_cls.numpy(), out["dino_final"][:, 0], atol=1e-5)  # CLS before the norm, then normed


def test_tail_kinds_are_each_fold_models_full_forward_and_the_trunk_is_shared(images):
    encoder, models = encoder_with([(0, 1), (1, 2)])
    for fold, model in models.items():
        gap = layers.full_forward_gap(encoder, fold, model, images)
        assert gap["shared_blocks"] == 0.0
        assert max(gap.values()) < 1e-5
    out = encoder.encode(images)
    assert out["g_blocks"].shape == (3, 2, N_TRAINED, 2, 384) and out["g_pool"].shape == (3, 2, 5 * 384)
    member = encoder.scorer.members[("g_combo", 1, 0)]
    (score,), _ = encoder.scorer.score(images, [member], embed=[member], frozen=False)
    np.testing.assert_allclose(out["g_pool"][:, 1], score.embedding, atol=1e-6)    # the existing pooled-embedding path
    assert not np.allclose(out["g_pool"][:, 0], out["g_pool"][:, 1])                 # the tails really differ


def test_a_model_on_another_trunk_is_refused():
    encoder, _ = encoder_with([])
    with pytest.raises(ValueError, match="frozen trunk differs"):
        encoder.add(0, 0, model_with_seed(1, 1))
    encoder.add(0, 0, model_with_seed(0, 1))
    with pytest.raises(ValueError, match="registered twice"):
        encoder.add(0, 1, model_with_seed(0, 2))


def test_fold_routing_takes_the_model_of_the_shots_own_fold(images):
    encoder, models = encoder_with([(3, 1), (1, 2), (0, 3)])        # registration order is not fold order
    out = encoder.encode(images)
    shot_folds = np.array([1, 3, 0])
    picked = layers.own_fold(out["g_pool"], np.arange(3), shot_folds, encoder.folds)
    pooled = []
    for i, fold in enumerate(shot_folds):
        hook = models[fold].head.head.register_forward_hook(lambda _m, x, _o: pooled.append(x[0]))
        with torch.no_grad():
            models[fold](image=images[i:i + 1])
        hook.remove()
        np.testing.assert_allclose(picked[i], pooled[-1][0].numpy(), atol=1e-5)
    with pytest.raises(ValueError, match="training rows"):
        layers.own_fold(out["g_pool"], np.arange(3), np.array([1, layers.NO_FOLD, 0]), encoder.folds)
    with pytest.raises(ValueError, match="no model"):
        layers.own_fold(out["g_pool"], np.arange(3), np.array([1, 2, 0]), encoder.folds)


# --- run script ----------------------------------------------------------------------------------------------------------

def offset_inputs():
    shots = pd.DataFrame({"shot_id": ["t1", "t0", "p0", "pen"], "game": ["g1", "g0", "gp", "g0"],
                          "split": ["train", "valid", "test", "train"], "is_penalty": [False, False, False, True]})
    frames = pd.DataFrame([{"shot_id": s, "offset_ms": o, "status": "missing" if (s, o) == ("t1", -400) else "ok",
                            "decoded_relative_ms": float(o), "timing_error_ms": 0.0, "cut_between_anchor": False,
                            "frame_path": None if (s, o) == ("t1", -400) else f"frames/{s}/{o}.png", "file_sha256": "x",
                            "file_bytes": 1} for s in shots.shot_id for o in (-1600, -400, 0)])
    folds = pd.DataFrame({"game": ["g0", "g1"], "fold": [4, 2]})
    return shots, frames, folds


def test_plan_orders_training_then_paired_and_routes_folds():
    shots, frames, folds = offset_inputs()
    plan = script.build_plan(pd.DataFrame({"shot_id": ["t1", "t0"]}), ["p0"], shots, frames, folds, Path("/f"), chunk_rows=4)
    assert plan.shot_id.tolist() == ["t0", "t0", "t1", "t1", "p0", "p0"]
    assert plan.offset_ms.tolist() == [0, -400] * 3
    assert plan.fold.tolist() == [4, 4, 2, 2, layers.NO_FOLD, layers.NO_FOLD]
    assert plan.row.tolist() == [0, 1, 2, -1, 3, 4]
    assert plan.chunk.tolist() == [0, 0, 0, 0, 1, 1]
    assert plan.frame_file.iloc[0] == "/f/frames/t0/0.png"
    with pytest.raises(ValueError, match="both training and paired"):
        script.build_plan(pd.DataFrame({"shot_id": ["t0"]}), ["t0"], shots, frames, folds, Path("/f"), 4)
    with pytest.raises(ValueError, match="penalty"):
        script.build_plan(pd.DataFrame({"shot_id": ["t0"]}), ["pen"], shots, frames, folds, Path("/f"), 4)


def test_a_chunk_counts_only_while_its_slices_hash_as_recorded(tmp_path):
    shots, frames, folds = offset_inputs()
    plan = script.build_plan(pd.DataFrame({"shot_id": ["t1", "t0"]}), ["p0"], shots, frames, folds, Path("/f"), chunk_rows=4)
    arrays = script.open_arrays(tmp_path, int(plan.has_frame.sum()), {"g_pool": (2, 3)})
    rows = plan[plan.chunk.eq(0)]
    assert not script.committed(tmp_path, 0, rows, arrays)
    arrays["g_pool"][:3] = 1.0
    script.write_atomic(script.marker_file(tmp_path, 0), script.json_text(script.marker(rows, arrays)).encode())
    assert script.committed(tmp_path, 0, rows, arrays)
    arrays["g_pool"][2, 0, 0] = 2.0
    assert not script.committed(tmp_path, 0, rows, arrays)
    arrays["g_pool"][3] = 5.0                                          # another chunk's rows do not matter
    arrays["g_pool"][2, 0, 0] = 1.0
    assert script.committed(tmp_path, 0, rows, arrays)


def test_the_paired_list_is_refused_inside_the_statsbomb_guard(tmp_path):
    folder = tmp_path / "xgv_eval_primary"
    folder.mkdir()
    pd.DataFrame({"key": ["a", "b", "c"], "role": ["primary", "primary", "gate"],
                  "is_penalty": [False, True, False], "sb_x": [1, 2, 3]}).to_parquet(folder / "shots.parquet")
    (folder / "complete.json").write_text(f'{{"files": {{"shots.parquet": "{script.sha256_file(folder / "shots.parquet")}"}}}}')
    assert script.read_paired(folder / "shots.parquet") == ["a"]
    with no_statsbomb_reads(), pytest.raises(StatsBombRead):
        script.read_paired(folder / "shots.parquet")
