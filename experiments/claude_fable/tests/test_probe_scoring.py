"""Shared-trunk scoring: the split into trunk and tails is the full forward, the frozen encoder is what the protocol
says, and the stored frames become the same model input as the dataset the models were trained with. CPU, random
weights, a ViT-S so the tests stay fast."""

import hashlib

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from xgpoc.data.datasets import ImageDataset
from xgpoc.models import g_bundle
from xgpoc.models.d_foundation import FinetuneViTXG
from xgpoc.probe.scoring import (
    FrameDataset,
    SharedScorer,
    check_same_trunk,
    ensemble_logits,
    trunk_features,
    trunk_state,
)

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


def write_frames(tmp_path, n=3, size=(448, 252)):
    rng = np.random.default_rng(0)
    paths = []
    for i in range(n):
        path = tmp_path / f"f{i}.png"
        cv2.imwrite(str(path), rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))
        paths.append(path)
    return paths


def test_trunk_then_tails_reproduce_the_full_forward_of_each_model(tmp_path):
    scorer = SharedScorer(model_with_seed(0, None), N_TRAINED, "cpu")
    models = [model_with_seed(0, 1), model_with_seed(0, 2)]
    members = [scorer.add("g_combo", 0, seed, 1.0, m) for seed, m in enumerate(models)]
    x = torch.randn(3, 3, 252, 448)
    scores, _ = scorer.score(x, members, embed=members[:1])
    with torch.no_grad():
        for model, score in zip(models, scores, strict=True):
            full = model(image=x).numpy()
            np.testing.assert_allclose(score.xg, full[:, 0], atol=1e-6)
            np.testing.assert_allclose(score.on_target, full[:, 1], atol=1e-6)
    assert scores[0].embedding.shape == (3, 384 * 5) and scores[1].embedding is None
    assert not np.allclose(scores[0].xg, scores[1].xg)  # the tails really differ


def test_scores_and_embeddings_equal_g_outputs_on_stored_frames(tmp_path):
    """The single forward path of the bundles (`g_outputs`) and the shared-trunk path agree on the same PNG files."""
    paths = write_frames(tmp_path)
    scorer = SharedScorer(model_with_seed(0, None), N_TRAINED, "cpu")
    model = model_with_seed(0, 5)
    member = scorer.add("g_combo", 0, 0, 1.0, model)
    spec = g_bundle.g_spec("dinov2_s", N_TRAINED, 2, "overlay")
    xg, on_target, embedding = g_bundle.g_outputs(model, pd.DataFrame({"frame_path": [str(p) for p in paths], "goal": 0}), spec,
                                                  device="cpu", batch_size=2, num_workers=0)
    shas = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    images = torch.stack(list(FrameDataset(paths, shas, (448, 252), "overlay")))
    (score,), _ = scorer.score(images, [member], embed=[member])
    np.testing.assert_allclose(score.xg, xg, atol=1e-5)
    np.testing.assert_allclose(score.on_target, on_target, atol=1e-5)
    np.testing.assert_allclose(score.embedding, embedding, atol=1e-5)


def test_frozen_embedding_is_cls_plus_mean_patch_token_of_the_last_layer():
    pretrained = model_with_seed(0, None)
    scorer = SharedScorer(pretrained, N_TRAINED, "cpu")
    x = torch.randn(2, 3, 252, 448)
    _, frozen = scorer.score(x, [], frozen=True)
    with torch.no_grad():
        tokens = model_with_seed(0, None).backbone.forward_features(x)
    expected = torch.cat([tokens[:, 0], tokens[:, 1:].mean(1)], 1).numpy()
    assert frozen.shape == (2, 2 * 384)
    np.testing.assert_allclose(frozen, expected, atol=1e-6)


def test_a_model_on_other_frozen_weights_is_refused():
    scorer = SharedScorer(model_with_seed(0, None), N_TRAINED, "cpu")
    with pytest.raises(ValueError, match="differs from the shared trunk"):
        scorer.add("g_combo", 0, 0, 1.0, model_with_seed(1, 3))
    reference = trunk_state(model_with_seed(0, None).backbone, N_TRAINED)
    check_same_trunk(reference, model_with_seed(0, 4).backbone, N_TRAINED)  # a different tail is fine
    assert not any(k.startswith(("blocks.10.", "blocks.11.", "norm.")) for k in reference)
    assert "blocks.9.attn.qkv.weight" in reference


def test_trunk_is_the_first_blocks_of_forward_features():
    model = model_with_seed(0, None)
    x = torch.randn(1, 3, 252, 448)
    with torch.no_grad():
        features = trunk_features(model.backbone, x, N_TRAINED)
        rest = model.backbone.norm(model.backbone.blocks[-N_TRAINED:](features))
        np.testing.assert_array_equal(rest.numpy(), model.backbone.forward_features(x).numpy())


def test_ensemble_logits_average_the_scaled_and_the_raw_logits_separately():
    class Score:
        def __init__(self, xg, ot):
            self.xg, self.on_target = np.array(xg, np.float32), np.array(ot, np.float32)

    class Member:
        def __init__(self, t):
            self.temperature = t

    scaled, raw, on_target = ensemble_logits([Score([1.0, 2.0], [0.0, 1.0]), Score([3.0, 4.0], [2.0, 3.0])],
                                             [Member(1.0), Member(2.0)])
    np.testing.assert_allclose(scaled, [(1 / 1 + 3 / 2) / 2, (2 / 1 + 4 / 2) / 2])
    np.testing.assert_allclose(raw, [2.0, 3.0])
    np.testing.assert_allclose(on_target, [1.0, 2.0])


@pytest.mark.parametrize("size", [(448, 252), (640, 360)])
def test_frame_dataset_is_the_dataset_the_models_were_trained_with(tmp_path, size):
    paths = write_frames(tmp_path, n=2, size=size)
    shas = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    ours = FrameDataset(paths, shas, (448, 252), "overlay")
    theirs = ImageDataset(pd.DataFrame({"frame_path": [str(p) for p in paths], "goal": 0}), size_wh=(448, 252), mask="overlay")
    for i in range(2):
        assert torch.equal(ours[i], theirs[i]["image"])


def test_frame_dataset_refuses_a_file_that_changed(tmp_path):
    (path,) = write_frames(tmp_path, n=1)
    dataset = FrameDataset([path], [hashlib.sha256(path.read_bytes()).hexdigest()], (448, 252), "overlay")
    dataset[0]
    cv2.imwrite(str(path), np.zeros((252, 448, 3), np.uint8))
    with pytest.raises(ValueError, match="recorded sha256"):
        dataset[0]
    with pytest.raises(ValueError, match="one sha256 per frame"):
        FrameDataset([path], [], (448, 252), "overlay")
