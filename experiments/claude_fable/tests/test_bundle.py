"""Trained-parameter bundles: save, load, replay on a tiny model that has the structure of
FinetuneViTXG (frozen pretrained trunk, last block and head trainable, the trainable block
registered a second time under another name)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from xgpoc.bundle import (
    BundleError,
    backbone_identity,
    canonical_digest,
    check_data_files,
    load_bundle,
    make_provenance,
    read_bundle,
    save_bundle,
    sha256_file,
    trained_keys,
)
from xgpoc.models.g_bundle import ensemble_probability, g_spec

SPEC = {"kind": "tiny", "dim": 6}


class Tiny(nn.Module):
    def __init__(self, dim: int, weights):
        super().__init__()
        self.backbone = nn.ModuleDict({"blocks": nn.Sequential(nn.Linear(4, dim), nn.Linear(dim, dim)),
                                       "norm": nn.LayerNorm(dim)})
        self.backbone.load_state_dict(torch.load(weights, weights_only=True))
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.trainable_blocks = self.backbone["blocks"][-1:]  # alias, as in FinetuneViTXG
        for p in list(self.trainable_blocks.parameters()) + list(self.backbone["norm"].parameters()):
            p.requires_grad = True
        self.head = nn.Linear(dim, 1)

    def forward(self, x):
        return self.head(self.backbone["norm"](self.backbone["blocks"](x))).squeeze(-1)


@pytest.fixture
def pretrained(tmp_path):
    torch.manual_seed(0)
    path = tmp_path / "pretrained.pt"
    trunk = nn.ModuleDict({"blocks": nn.Sequential(nn.Linear(4, 6), nn.Linear(6, 6)), "norm": nn.LayerNorm(6)})
    torch.save(trunk.state_dict(), path)
    return path


def factory_for(weights):
    return lambda spec: (Tiny(spec["dim"], weights), weights)


def trained(weights):
    torch.manual_seed(1)
    model = Tiny(6, weights)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
    x, y = torch.randn(32, 4), torch.rand(32)
    for _ in range(20):
        opt.zero_grad()
        nn.functional.mse_loss(model(x), y).backward()
        opt.step()
    return model.eval(), x


def provenance(**data):
    return make_provenance({"train.py": "aa", "model.py": "bb"},
                           **{"index_sha256": "i", "folds_sha256": "f", "overlay_sha256": "o", **data})


def save(tmp_path, model, weights, name="b.pt"):
    return save_bundle(tmp_path / name, model, spec=SPEC, backbone=backbone_identity("tiny", weights, repo="r"),
                       calibration={"temperature": 1.3}, recipe={"lr": 0.1}, commit={"sha": "abc", "dirty": False},
                       provenance=provenance())


def test_save_load_replay_is_exact(tmp_path, pretrained):
    model, x = trained(pretrained)
    expected = model(x).detach()
    info = save(tmp_path, model, pretrained)
    loaded = load_bundle(tmp_path / "b.pt", factory_for(pretrained))
    assert torch.equal(loaded.model(x).detach(), expected)
    assert loaded.calibration == {"temperature": 1.3} and loaded.commit["sha"] == "abc"
    assert loaded.backbone["file"] == "pretrained.pt" and loaded.backbone["repo"] == "r"
    assert loaded.provenance == provenance()
    assert loaded.backbone["sha256"] == sha256_file(pretrained)
    assert not loaded.model.training
    assert info["sha256"] == sha256_file(tmp_path / "b.pt")


def test_bundle_holds_only_trained_tensors(tmp_path, pretrained):
    model, _ = trained(pretrained)
    keys = trained_keys(model)
    assert not any(k.startswith("trainable_blocks") for k in keys)  # the alias is not stored twice
    assert not any(k.startswith("backbone.blocks.0") for k in keys)  # frozen trunk comes from the pretrained file
    info = save(tmp_path, model, pretrained)
    assert info["n_tensors"] == len(keys) == 6  # last block, norm and head: weight and bias each
    n_trained = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert info["n_params"] == n_trained < sum(p.numel() for p in model.parameters())


def test_changed_pretrained_weights_are_refused(tmp_path, pretrained):
    model, _ = trained(pretrained)
    save(tmp_path, model, pretrained)
    other = tmp_path / "other.pt"
    state = torch.load(pretrained, weights_only=True)
    state["blocks.0.weight"] = state["blocks.0.weight"] + 1e-3
    torch.save(state, other)
    with pytest.raises(BundleError, match="differs"):
        load_bundle(tmp_path / "b.pt", factory_for(other))


def test_bundle_must_match_the_trainable_set(tmp_path, pretrained):
    model, _ = trained(pretrained)
    save(tmp_path, model, pretrained)

    def frozen_head(spec):
        m, w = factory_for(pretrained)(spec)
        for p in m.head.parameters():
            p.requires_grad = False
        return m, w

    with pytest.raises(BundleError, match="trains"):
        load_bundle(tmp_path / "b.pt", frozen_head)

    def extra_trainable(spec):
        m, w = factory_for(pretrained)(spec)
        m.extra = nn.Linear(2, 2)  # trainable tensors the bundle does not have
        return m, w

    with pytest.raises(BundleError, match="trains"):
        load_bundle(tmp_path / "b.pt", extra_trainable)


def test_foreign_file_is_refused(tmp_path, pretrained):
    torch.save({"state_dict": {}}, tmp_path / "old.pt")
    with pytest.raises(BundleError, match="not a"):
        load_bundle(tmp_path / "old.pt", factory_for(pretrained))


def test_ensemble_probability_follows_the_deployed_steps():
    rng = np.random.default_rng(0)
    z1, z2 = rng.normal(-2, 1, 50), rng.normal(-2, 1, 50)
    pen = np.zeros(50, bool)
    pen[:3] = True
    cal = {"platt": {"a": 1.1, "b": -0.2}, "oof_rate": 0.12, "pooled_rate": 0.13, "penalty_rate": 0.7}
    p = ensemble_probability([z1, z2], [1.5, 1.2], cal, pen)
    z = (z1 / 1.5 + z2 / 1.2) / 2
    q = 1 / (1 + np.exp(-(1.1 * z - 0.2)))
    q = 1 / (1 + np.exp(-(np.log(q / (1 - q)) + np.log(0.13 / 0.87) - np.log(0.12 / 0.88))))
    np.testing.assert_allclose(p[3:], q[3:], atol=1e-9)
    assert (p[:3] == 0.7).all()


def test_g_spec_records_the_input_geometry():
    spec = g_spec("dinov2_l", 4, 2, "overlay")
    assert spec["model"] == {"backbone": "dinov2_l", "n_blocks": 4, "n_out": 2}
    assert spec["input"] == {"size_wh": [448, 252], "mask": "overlay", "frame_col": "frame_path"}


def test_provenance_is_exposed_and_checked_against_files(tmp_path, pretrained):
    model, _ = trained(pretrained)
    save(tmp_path, model, pretrained)
    prov = load_bundle(tmp_path / "b.pt", factory_for(pretrained)).provenance
    assert prov["code"]["digest"] == canonical_digest({"train.py": "aa", "model.py": "bb"})
    folds, overlay = tmp_path / "folds.csv", tmp_path / "mask.png"
    folds.write_text("game,fold\n")
    overlay.write_bytes(b"png")
    good = provenance(folds_sha256=sha256_file(folds), overlay_sha256=sha256_file(overlay))
    check_data_files(good, folds_sha256=folds, overlay_sha256=overlay)
    folds.write_text("game,fold\nx,1\n")  # another folds file
    with pytest.raises(BundleError, match="folds_sha256"):
        check_data_files(good, folds_sha256=folds)
    with pytest.raises(KeyError):
        check_data_files(good, other=folds)


def _tamper(path, edit):
    payload = torch.load(path, weights_only=True)
    edit(payload)
    torch.save(payload, path)


def test_malformed_bundles_raise_bundle_error(tmp_path, pretrained):
    model, _ = trained(pretrained)
    save(tmp_path, model, pretrained)
    path = tmp_path / "b.pt"
    factory = factory_for(pretrained)
    torch.save([1, 2, 3], tmp_path / "list.pt")
    with pytest.raises(BundleError, match="not a dict"):
        load_bundle(tmp_path / "list.pt", factory)
    (tmp_path / "junk.pt").write_bytes(b"not a pickle")
    with pytest.raises(BundleError, match="cannot be read"):
        read_bundle(tmp_path / "junk.pt")

    def rename(p):  # a trained tensor under a name the rebuilt model does not train
        p["params"]["head.renamed"] = p["params"].pop("head.weight")

    def reshape(p):  # right name, wrong shape
        p["params"]["head.weight"] = torch.zeros(3, 6)

    def bad_digest(p):
        p["provenance"]["code"]["files"]["train.py"] = "changed"

    def drop_provenance(p):
        del p["provenance"]

    for edit, match in ((rename, "trains"), (reshape, "do not fit"), (bad_digest, "digest"), (drop_provenance, "lacks")):
        fresh = tmp_path / f"{edit.__name__}.pt"
        fresh.write_bytes(path.read_bytes())
        _tamper(fresh, edit)
        with pytest.raises(BundleError, match=match):
            load_bundle(fresh, factory)


def _synthetic_runs(rng):
    """(fold, seed) -> member frames as report_complexity.load_runs builds them, plus labels."""
    from xgpoc.eval.calibration import logit

    valid_ids = {f: [f"v{f}_{i}" for i in range(40)] for f in (0, 1)}
    test_ids = [f"t{i}" for i in range(60)]
    temps = {(f, s): 1.0 + 0.2 * f + 0.1 * s for f in (0, 1) for s in (0, 1)}
    raw, runs = {}, {}
    for (f, s), t in temps.items():
        raw[f, s] = {"valid": rng.normal(-2, 1, 40), "test": rng.normal(-2, 1, 60)}
        runs[f, s] = {role: pd.DataFrame({"shot_id": ids, "z": logit(1 / (1 + np.exp(-raw[f, s][role] / t)))})
                      for role, ids in (("valid", valid_ids[f]), ("test", test_ids))}
    ids = [i for f in (0, 1) for i in valid_ids[f]] + test_ids
    labels = pd.Series(rng.random(len(ids)) < 0.15, index=ids)
    is_pen = pd.Series(rng.random(len(ids)) < 0.05, index=ids)
    return runs, raw, temps, labels, is_pen, test_ids


def test_ensemble_probability_equals_the_archived_deployment():
    """The shipped scoring path and report_complexity.deploy (which reproduces the archived G10)
    give the same probabilities from the same member logits."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from report_complexity import configurations, deploy, mean_logit
    from xgpoc.models.g_bundle import fit_ensemble_calibration

    runs, raw, temps, labels, is_pen, test_ids = _synthetic_runs(np.random.default_rng(3))
    keys = configurations(runs)["g10"][0][1]
    pooled, p_pen = 0.14, 0.7
    expected = deploy(runs, keys, labels, is_pen, pooled, p_pen)
    oof = mean_logit([runs[k]["valid"] for k in keys])
    cal = fit_ensemble_calibration(oof.to_numpy(), labels.loc[oof.index].to_numpy(), pooled, p_pen)
    got = ensemble_probability([raw[k]["test"] for k in keys], [temps[k] for k in keys], cal,
                               is_pen.loc[test_ids].to_numpy())
    np.testing.assert_allclose(got, expected.loc[test_ids].to_numpy(), atol=1e-9)
    assert is_pen.loc[test_ids].any() and (got[is_pen.loc[test_ids].to_numpy()] == p_pen).all()


@pytest.fixture
def small_g(tmp_path, monkeypatch):
    """The real FinetuneViTXG layout on a ViT-S with random weights; a stand-in weight file
    plays the pretrained checkpoint (only its hash matters to the bundle)."""
    from xgpoc.models import g_bundle
    from xgpoc.models.d_foundation import FinetuneViTXG

    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"stand-in for the pretrained checkpoint")
    monkeypatch.setattr(g_bundle, "pretrained_weight_file", lambda backbone: weights)

    def build(*, backbone, n_blocks, n_out, pretrained):
        assert pretrained
        torch.manual_seed(0)  # same frozen weights every time, as a checkpoint would give
        return FinetuneViTXG(backbone, n_blocks=n_blocks, n_out=n_out, pretrained=False)

    monkeypatch.setattr(g_bundle, "FinetuneViTXG", build)
    return g_bundle, weights


def test_g_layout_trains_only_blocks_norm_and_head(small_g):
    g_bundle, _ = small_g
    spec = g_spec("dinov2_s", 2, 2, "overlay")
    model, _ = g_bundle.build_g(spec)
    keys = trained_keys(model)
    assert not list(model.buffers())
    assert any(k.startswith("trainable_blocks.") for k in model.state_dict())  # the alias exists ...
    assert not any(k.startswith("trainable_blocks.") for k in keys)  # ... and is not stored
    trained_prefixes = ("backbone.blocks.10.", "backbone.blocks.11.", "backbone.norm.", "head.")  # dinov2_s: 12 blocks, last 2
    assert keys and all(k.startswith(trained_prefixes) for k in keys)
    assert all(any(k.startswith(f"backbone.blocks.{i}.") for k in keys) for i in (10, 11))
    assert not any(k.startswith("backbone.blocks.9.") or k.startswith("backbone.patch_embed") for k in keys)
    assert sum(v.numel() for k, v in model.state_dict().items() if k in set(keys)) == sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    with pytest.raises(ValueError, match="not a G bundle"):
        g_bundle.build_g({**spec, "kind": "something_else"})


def test_g_outputs_replay_and_embedding(small_g, tmp_path):
    import cv2

    g_bundle, weights = small_g
    spec = g_spec("dinov2_s", 2, 2, "overlay")
    model, _ = g_bundle.build_g(spec)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(0.02 * torch.randn_like(p))
    rng = np.random.default_rng(0)
    paths = []
    for i in range(3):
        path = tmp_path / f"f{i}.jpg"
        cv2.imwrite(str(path), rng.integers(0, 255, (360, 640, 3), dtype=np.uint8))
        paths.append(str(path))
    df = pd.DataFrame({"shot_id": list("abc"), "frame_path": paths, "goal": [0, 1, 0]})
    save_bundle(tmp_path / "g.pt", model, spec=spec, backbone=g_bundle.g_identity(spec), commit={"sha": "x", "dirty": False},
                calibration={"temperature": 1.0}, recipe={"cfg": {"batch_size": 2}}, provenance=provenance())
    expected = g_bundle.g_outputs(model, df, spec, device="cpu", batch_size=2, num_workers=0)
    loaded = g_bundle.load_g(tmp_path / "g.pt")
    xg, on_target, embedding = g_bundle.g_outputs(loaded.model, df, spec, device="cpu", batch_size=2, num_workers=0)
    for a, b in zip((xg, on_target, embedding), expected, strict=True):
        np.testing.assert_array_equal(a, b)
    assert xg.shape == on_target.shape == (3,) and embedding.shape == (3, 384 * 5)
    # the embedding is exactly what the head MLP turns into the two logits
    with torch.no_grad():
        both = loaded.model.head.head(torch.from_numpy(embedding)).numpy()
    np.testing.assert_allclose(both, np.stack([xg, on_target], 1), atol=1e-6)
    with pytest.raises(ValueError, match="two-output"):
        g_bundle.g_outputs(loaded.model, df, {**spec, "model": {**spec["model"], "n_out": 1}}, device="cpu")
    weights.write_bytes(b"a different checkpoint")
    with pytest.raises(BundleError, match="differs"):
        g_bundle.load_g(tmp_path / "g.pt")
