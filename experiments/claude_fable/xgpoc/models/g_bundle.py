"""G (fine-tuned DINOv2) specifics for bundles: rebuild from a spec, score frames, combine
fold models into the deployed ensemble probability.

`g_outputs` is the one forward path for replay and for scoring new inputs: it runs the same
fp32 eval-mode forward over an `ImageDataset` as the training-time prediction, so a replay in
an identical batch order reproduces the stored predictions, and it also returns the on-target
logit and the pooled embedding.

Pooled embedding. `AttentionPoolHead` builds z = concat(CLS token, the n_queries attention-pooled
query vectors), each a LayerNorm-ed ViT-L token of width 1024, so z has 5 x 1024 = 5120 features
(float32). z is exactly the input of `model.head.head`, the MLP (LayerNorm, Linear 256, GELU,
Linear 2) that produces both logits: the xG logit and the on-target logit are functions of z and
nothing else. It is captured with a forward hook on that MLP, taken after the model's own
LayerNorm on the tokens and before the MLP's first LayerNorm.

Selection note. Epoch selection and the temperature of a fold model used the shots of its
held-out fold, and the ensemble Platt step is fitted on the pooled out-of-fold predictions, so
out-of-fold scores of training-pool shots are mildly optimistic. The same holds at every frame
offset, so contrasts between offsets are not biased by it; test-split scores are not affected."""

from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from huggingface_hub import hf_hub_download
from torch import nn
from torch.utils.data import DataLoader

from xgpoc.bundle import LoadedBundle, backbone_identity, load_bundle
from xgpoc.data.datasets import ImageDataset
from xgpoc.eval.calibration import PlattScaler, logit, prior_shift, route_penalties
from xgpoc.models.d_foundation import BACKBONES, FinetuneViTXG

KIND = "finetune_vit_xg"
EVAL_BATCH = 8  # the training-time evaluation batch of ViT-L; numerics depend on batch composition


WEIGHT_FILE = "model.safetensors"


def hub_repo(backbone: str) -> str:
    return timm.models.get_pretrained_cfg(BACKBONES[backbone][0]).hf_hub_id


def pretrained_weight_file(backbone: str) -> Path:
    """The cached hub file timm loads for this backbone (offline: it must be present). The
    returned path is the snapshot entry `<repo cache>/snapshots/<revision>/model.safetensors`."""
    return Path(hf_hub_download(hub_repo(backbone), WEIGHT_FILE, local_files_only=True))


def g_spec(backbone: str, n_blocks: int, n_out: int, mask: str) -> dict:
    return {"kind": KIND, "model": {"backbone": backbone, "n_blocks": n_blocks, "n_out": n_out},
            "input": {"size_wh": list(BACKBONES[backbone][1]), "mask": mask, "frame_col": "frame_path"}}


def g_identity(spec: dict) -> dict:
    name = spec["model"]["backbone"]
    weights = pretrained_weight_file(name)
    return backbone_identity(name, weights, repo=hub_repo(name), revision=weights.parent.name)


def build_g(spec: dict) -> tuple[nn.Module, Path]:
    if spec["kind"] != KIND:
        raise ValueError(f"not a G bundle: {spec['kind']}")
    return FinetuneViTXG(**spec["model"], pretrained=True), pretrained_weight_file(spec["model"]["backbone"])


def load_g(path: Path) -> LoadedBundle:
    return load_bundle(path, build_g)


def g_dataset(df: pd.DataFrame, spec: dict) -> ImageDataset:
    i = spec["input"]
    return ImageDataset(df, size_wh=tuple(i["size_wh"]), mask=i["mask"], path_col=i["frame_col"])


def g_outputs(model: nn.Module, df: pd.DataFrame, spec: dict, device: str = "cuda", batch_size: int = EVAL_BATCH,
              num_workers: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(xG logit [N], on-target logit [N], pooled embedding [N, 5120]) for every row of `df`, in
    row order, uncalibrated, float32. Embedding: see the module docstring. Use the batch size
    the model was trained/evaluated with (bundle recipe `cfg.batch_size`) for bit-level replay."""
    if spec["model"]["n_out"] != 2:
        raise ValueError("g_outputs needs the two-output head (xG and on-target logits)")
    ds = g_dataset(df, spec)
    if len(ds) != len(df):
        raise ValueError(f"{len(df) - len(ds)} rows without a usable frame")
    model.to(device).eval()
    captured: list[torch.Tensor] = []
    handle = model.head.head.register_forward_hook(lambda _m, inputs, _out: captured.append(inputs[0].detach().float().cpu()))
    logits = []
    try:
        with torch.no_grad():
            for batch in DataLoader(ds, batch_size, shuffle=False, num_workers=num_workers):
                logits.append(model(image=batch["image"].to(device, non_blocking=True)).float().cpu())
    finally:
        handle.remove()
    out = torch.cat(logits).numpy() if logits else np.zeros((0, 2), np.float32)
    emb = torch.cat(captured).numpy() if captured else np.zeros((0, 0), np.float32)
    return out[:, 0], out[:, 1], emb


def fit_ensemble_calibration(oof_logits: np.ndarray, y: np.ndarray, pooled_rate: float, penalty_rate: float) -> dict:
    """Calibration of the ten-model ensemble, fitted on out-of-fold logits only: Platt, the
    out-of-fold goal rate, the pooled rate it is shifted to, the penalty rate."""
    platt = PlattScaler().fit(oof_logits, y)
    return {"platt": {"a": platt.a, "b": platt.b}, "oof_rate": float(np.mean(y)), "pooled_rate": float(pooled_rate),
            "penalty_rate": float(penalty_rate), "n_oof": len(oof_logits)}


def ensemble_probability(logits: list[np.ndarray], temperatures: list[float], calibration: dict,
                         is_penalty: np.ndarray) -> np.ndarray:
    """Deployed G-combo probability from member goal logits, same steps as the reported
    model: temperature-scaled logits averaged, Platt fitted on the out-of-fold ensemble,
    prior correction to the pooled rate, penalties routed to their own rate."""
    z = np.mean([logit(1 / (1 + np.exp(-np.asarray(lg, float) / t))) for lg, t in zip(logits, temperatures, strict=True)],
                axis=0)
    p = PlattScaler(**calibration["platt"])(z)
    p = prior_shift(p, calibration["oof_rate"], calibration["pooled_rate"])
    return route_penalties(p, np.asarray(is_penalty, bool), calibration["penalty_rate"])

