"""Shared training loop, calibration and result writing for all PoCs."""

import contextlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from xgpoc.eval.metrics import evaluate


@dataclass
class TrainConfig:
    epochs: int = 10               # length of the one-cycle LR schedule *and* the epoch cap
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-2
    patience: int = 4              # early stopping on the selection score; >= epochs runs the whole cycle
    max_minutes: float = 5.0       # hard wall-clock cap per PoC
    num_workers: int = 4
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True
    warmup_epochs: float = 1.0     # linear warm-up at the start of the cycle
    select_on: str = "calibrated"  # "calibrated": valid log-loss after temperature scaling; "raw"; "auc"
    ema_decay: float = 0.0         # exponential moving average of the trainable weights; 0 = off
    aux_weight: float = 0.0        # weight of the extra logits' BCE (multi-task); needs "y_aux" in batches
    mixup_alpha: float = 0.0       # mixup on the "image" input; 0 = off
    llrd: float = 1.0              # layer-wise LR decay across unfrozen backbone blocks, top to bottom; 1 = flat


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


class TemperatureScaler:
    """Post-hoc calibration on validation logits (single scalar, fit by LBFGS)."""

    def __init__(self):
        self.t = 1.0

    def fit(self, logits: np.ndarray, y: np.ndarray) -> "TemperatureScaler":
        lg, yy = torch.tensor(logits, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        log_t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(lg / log_t.exp().clamp(1e-3, 1e3), yy)
            loss.backward()
            return loss
        opt.step(closure)
        t = float(log_t.exp())
        # near-constant logits (e.g. an EMA copy evaluated after its first few steps) make the
        # likelihood flat in T and LBFGS can run off to 0 or infinity; clamp to a sane range
        self.t = float(np.clip(t, 0.05, 20.0)) if np.isfinite(t) else 1.0
        return self

    def __call__(self, logits: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-logits / self.t))


class _Ema:
    """Exponential moving average of the trainable parameters, evaluated in place of the
    raw weights: it smooths the last steps of a short, high-LR cycle, which is exactly
    where the fine-tuning runs were unstable."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay, self.n = decay, 0
        self.shadow = {n: p.detach().clone().float() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        # warm-up: the average forgets the random initialisation quickly instead of
        # carrying it through the first epochs (decay ramps from 0.1 towards `decay`)
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].lerp_(p.detach().float(), 1 - d)

    @contextlib.contextmanager
    def applied(self, model: nn.Module):
        backup = {n: p.detach().clone() for n, p in model.named_parameters() if n in self.shadow}
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in self.shadow:
                    p.copy_(self.shadow[n].to(p.dtype))
        try:
            yield
        finally:
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if n in backup:
                        p.copy_(backup[n])

    def state(self, model: nn.Module) -> dict:
        with self.applied(model):
            return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _main_logit(out: torch.Tensor) -> torch.Tensor:
    return out[:, 0] if out.dim() == 2 else out


def _predict_logits(model: nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for batch in loader:
            x = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k not in ("y", "y_aux")}
            out.append(_main_logit(model(**x)).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def _param_groups(model: nn.Module, params: list, cfg: TrainConfig, backbone_lr_mult: float | None):
    """Head at cfg.lr; pretrained backbone at a fraction of it, optionally decaying
    block by block from the head downwards (layer-wise LR decay)."""
    if backbone_lr_mult is None:
        return [{"params": params, "lr": cfg.lr}], cfg.lr
    if hasattr(model, "backbone_param_groups"):
        bb_groups = model.backbone_param_groups()
    else:
        bb_groups = [model.backbone_parameters()]
    groups, max_lr, bb_ids = [], [], set()
    for depth, plist in enumerate(bb_groups):
        plist = [p for p in plist if p.requires_grad]
        if not plist:
            continue
        lr = cfg.lr * backbone_lr_mult * (cfg.llrd ** depth)
        groups.append({"params": plist, "lr": lr})
        max_lr.append(lr)
        bb_ids.update(id(p) for p in plist)
    rest = [p for p in params if id(p) not in bb_ids]
    groups.append({"params": rest, "lr": cfg.lr})
    max_lr.append(cfg.lr)
    return groups, max_lr


def _loss(crit: nn.Module, out: torch.Tensor, y: torch.Tensor, y_aux: torch.Tensor | None,
          aux_weight: float, lam: float, perm: torch.Tensor | None) -> torch.Tensor:
    def mixed(o, t):
        return crit(o, t) if perm is None else lam * crit(o, t) + (1 - lam) * crit(o, t[perm])
    loss = mixed(_main_logit(out), y)
    if out.dim() == 2 and out.size(1) > 1 and y_aux is not None and aux_weight > 0:
        loss = loss + aux_weight * mixed(out[:, 1:], y_aux.reshape(len(y), -1))
    return loss


def train_torch(model: nn.Module, train_ds: Dataset, valid_ds: Dataset, cfg: TrainConfig,
                pos_weight: float | None = None, backbone_lr_mult: float | None = None) -> dict:
    """BCE trainer: one-cycle LR over `cfg.epochs`, optional EMA / mixup / multi-task,
    epoch selection on the *calibrated* validation log-loss.

    The cycle length equals the epoch cap on purpose: with early stopping cutting a long
    nominal cycle short, the earlier runs never left the warm-up phase. Selecting on the
    raw log-loss was the second defect - the logit scale swings between epochs of a
    fine-tuning run while the ranking is stable, so the epoch chosen by the raw loss was
    mostly the best-calibrated one, not the best model; the temperature is refitted for
    every epoch (milliseconds) and the calibrated loss is compared instead."""
    seed_all(cfg.seed)
    dev = cfg.device
    model.to(dev)
    tl = DataLoader(train_ds, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
                    drop_last=len(train_ds) > cfg.batch_size, pin_memory=True)
    vl = DataLoader(valid_ds, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    params = [p for p in model.parameters() if p.requires_grad]
    groups, max_lr = _param_groups(model, params, cfg, backbone_lr_mult)
    opt = torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
    pct = min(0.5, max(0.02, cfg.warmup_epochs / max(cfg.epochs, 1)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr, total_steps=max(1, cfg.epochs * len(tl)), pct_start=pct)
    scaler = torch.amp.GradScaler(enabled=cfg.amp and dev == "cuda")
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=dev) if pos_weight else None)
    yv = np.array([valid_ds[i]["y"] for i in range(len(valid_ds))], dtype=float)
    ema = _Ema(model, cfg.ema_decay) if cfg.ema_decay > 0 else None
    best, best_state, best_epoch, bad, history, t0 = float("inf"), None, -1, 0, [], time.time()
    for ep in range(cfg.epochs):
        model.train()
        tot, n = 0.0, 0
        for batch in tl:
            y = batch["y"].float().to(dev)
            y_aux = batch["y_aux"].float().to(dev) if "y_aux" in batch else None
            x = {k: v.to(dev, non_blocking=True) for k, v in batch.items() if k not in ("y", "y_aux")}
            lam, perm = 1.0, None
            if cfg.mixup_alpha > 0 and "image" in x and len(y) > 1:
                lam = float(np.random.beta(cfg.mixup_alpha, cfg.mixup_alpha))
                perm = torch.randperm(len(y), device=dev)
                x["image"] = lam * x["image"] + (1 - lam) * x["image"][perm]
            with torch.autocast(device_type="cuda", enabled=scaler.is_enabled()):
                out = model(**x).float()
            loss = _loss(crit, out, y, y_aux, cfg.aux_weight, lam, perm)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema is not None:
                ema.update(model)
            tot += loss.item() * len(y)
            n += len(y)
        with (ema.applied(model) if ema is not None else contextlib.nullcontext()):
            vlog = _predict_logits(model, vl, dev)
        temp = TemperatureScaler().fit(vlog, yv)
        vm_raw = evaluate(yv, np.nan_to_num(1 / (1 + np.exp(-vlog)), nan=float(yv.mean())))
        vm_cal = evaluate(yv, np.nan_to_num(temp(vlog), nan=float(yv.mean())))
        score = {"calibrated": vm_cal["log_loss"], "raw": vm_raw["log_loss"], "auc": -vm_raw["auc"]}[cfg.select_on]
        history.append({"epoch": ep, "train_loss": tot / max(n, 1), "valid_log_loss": vm_raw["log_loss"],
                        "valid_log_loss_cal": vm_cal["log_loss"], "valid_auc": vm_raw["auc"],
                        "temperature": temp.t, "lr": float(sched.get_last_lr()[-1]), "minutes": (time.time() - t0) / 60})
        print(f"ep {ep} train {tot / max(n, 1):.4f} valid logloss {vm_raw['log_loss']:.4f} cal {vm_cal['log_loss']:.4f} "
              f"auc {vm_raw['auc']:.3f} T {temp.t:.2f} ({history[-1]['minutes']:.1f} min)")
        if score < best - 1e-4:
            best, bad, best_epoch = score, 0, ep
            best_state = ema.state(model) if ema is not None else {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= cfg.patience or (time.time() - t0) / 60 > cfg.max_minutes:
            break
    if best_state:
        model.load_state_dict(best_state)
    vlog = _predict_logits(model, vl, dev)
    return {"history": history, "best_epoch": best_epoch, "valid_logits": vlog, "valid_y": yv,
            "temperature": TemperatureScaler().fit(vlog, yv)}


def predict_torch(model: nn.Module, ds: Dataset | list[Dataset], cfg: TrainConfig,
                  temperature: TemperatureScaler) -> np.ndarray:
    """Several datasets = test-time augmentation: logits are averaged before the
    temperature, so calibration stays defined on the same scale it was fitted on.
    Every view must yield rows in the same order."""
    views = ds if isinstance(ds, list) else [ds]
    logits = [_predict_logits(model, DataLoader(v, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers), cfg.device)
              for v in views]
    return temperature(np.mean(logits, axis=0))


def fit_temperature(model: nn.Module, views: list[Dataset], cfg: TrainConfig) -> TemperatureScaler:
    """Refit calibration on the validation set *through the same TTA* used at test
    time; a temperature fitted on single-view logits is wrong for averaged ones."""
    y = np.array([views[0][i]["y"] for i in range(len(views[0]))], dtype=float)
    logits = [_predict_logits(model, DataLoader(v, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers), cfg.device)
              for v in views]
    return TemperatureScaler().fit(np.mean(logits, axis=0), y)


def write_result(out_dir: Path, name: str, test_df: pd.DataFrame, p_test: np.ndarray, extra: dict | None = None,
                 valid_df: pd.DataFrame | None = None, p_valid: np.ndarray | None = None) -> dict:
    """Validation predictions are written alongside the test ones so that any later
    ensembling can pick its weights without ever touching the test set."""
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate(test_df["goal"].to_numpy(), p_test)
    metrics.update({"model": name, **(extra or {})})
    pd.DataFrame({"shot_id": test_df["shot_id"], "goal": test_df["goal"], "p": p_test}).to_csv(out_dir / f"{name}_test_predictions.csv", index=False)
    if valid_df is not None and p_valid is not None:
        pd.DataFrame({"shot_id": valid_df["shot_id"], "goal": valid_df["goal"], "p": p_valid}).to_csv(out_dir / f"{name}_valid_predictions.csv", index=False)
    (out_dir / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(name, {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)})
    return metrics


def n_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def config_dict(cfg) -> dict:
    return asdict(cfg)
