"""Fit one G-combo model and score named frame sets: the part of a fine-tuning run that
`scripts/run_xgc_g.py` (fold models) and `scripts/run_xgc_transfer.py` (transfer arms) share.

The caller owns everything around it: which shots go where, the guards against changed code or
frames, the files written and the completion record. The fit uses the recipe's `kw` dict (see
`run_v12.STAGES`) and a `TrainConfig`; epoch selection and the temperature come from
`train_torch` on the `select` frames.
"""

import gc
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from xgpoc.data.datasets import ImageDataset
from xgpoc.models.common import (
    TrainConfig,
    _predict_logits,
    n_trainable,
    seed_all,
    train_torch,
)
from xgpoc.models.d_foundation import FinetuneViTXG


@dataclass
class GFit:
    model: nn.Module  # the selected-epoch EMA weights; the caller deletes it when done
    fit: dict  # train_torch's result: history, best_epoch, valid_logits, temperature
    logits: dict[str, np.ndarray]  # per scored frame set, in the row order of the set
    compute: dict  # train_minutes, epochs_run, trainable_parameters, peak_gpu_gib


def fit_g(train: pd.DataFrame, select: pd.DataFrame, scored: dict[str, pd.DataFrame], cfg: TrainConfig, recipe_kw: dict,
          spec: dict, *, aux_col: str) -> GFit:
    """Seed, build the model of `spec`, train on `train`, select the epoch on `select`, and return
    the logits of every frame set in `scored`. Raises ValueError if no epoch could be selected;
    the model is freed on any failure."""
    started = time.time()
    seed_all(cfg.seed)
    model = FinetuneViTXG(recipe_kw["backbone"], n_blocks=recipe_kw["n_blocks"], n_out=2)
    try:
        input_args = {"size_wh": tuple(spec["input"]["size_wh"]), "mask": spec["input"]["mask"]}
        fit = train_torch(model, ImageDataset(train, train=True, **input_args, rrc_scale=tuple(recipe_kw["rrc"]), aux_col=aux_col),
                          ImageDataset(select, **input_args), cfg, backbone_lr_mult=recipe_kw["backbone_lr_mult"])
        if fit["best_epoch"] < 0:
            raise ValueError("no epoch was selected (non-finite select-set score); refusing to go on")
        logits = {name: _predict_logits(model, DataLoader(ImageDataset(df, **input_args), cfg.batch_size,
                                                          num_workers=cfg.num_workers), cfg.device)
                  for name, df in scored.items()}
        compute = {"train_minutes": (time.time() - started) / 60, "epochs_run": len(fit["history"]),
                   "trainable_parameters": n_trainable(model),
                   "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None}
        return GFit(model, fit, logits, compute)
    except BaseException:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        raise
