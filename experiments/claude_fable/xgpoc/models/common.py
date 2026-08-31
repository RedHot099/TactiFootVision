"""Shared training loop, calibration and result writing for all PoCs."""

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
    epochs: int = 10
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-2
    patience: int = 4
    max_minutes: float = 5.0       # hard wall-clock cap per PoC ("small tries" for now)
    num_workers: int = 4
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True


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
        opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)
        def closure():
            opt.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(lg / log_t.exp(), yy)
            loss.backward()
            return loss
        opt.step(closure)
        self.t = float(log_t.exp())
        return self

    def __call__(self, logits: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-logits / self.t))


def _predict_logits(model: nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for batch in loader:
            x = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != "y"}
            out.append(model(**x).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def train_torch(model: nn.Module, train_ds: Dataset, valid_ds: Dataset, cfg: TrainConfig,
                pos_weight: float | None = None, backbone_lr_mult: float | None = None) -> dict:
    """Minimal BCE trainer with early stopping on validation log-loss and a wall-clock cap.

    `backbone_lr_mult` splits the parameters into a pretrained group (whatever
    `model.backbone_parameters()` returns) and the rest, giving the pretrained one a
    fraction of the head's learning rate - fine-tuning a foundation model at the head's
    LR destroys the representation we came for."""
    seed_all(cfg.seed)
    dev = cfg.device
    model.to(dev)
    tl = DataLoader(train_ds, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=len(train_ds) > cfg.batch_size, pin_memory=True)
    vl = DataLoader(valid_ds, cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    params = [p for p in model.parameters() if p.requires_grad]
    if backbone_lr_mult is not None:
        bb = model.backbone_parameters()
        bb_ids = {id(p) for p in bb}
        rest = [p for p in params if id(p) not in bb_ids]
        groups = [{"params": bb, "lr": cfg.lr * backbone_lr_mult}, {"params": rest, "lr": cfg.lr}]
        max_lr = [cfg.lr * backbone_lr_mult, cfg.lr]
    else:
        groups, max_lr = [{"params": params, "lr": cfg.lr}], cfg.lr
    opt = torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr, total_steps=max(1, cfg.epochs * len(tl)), pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=cfg.amp and dev == "cuda")
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=dev) if pos_weight else None)
    yv = np.array([valid_ds[i]["y"] for i in range(len(valid_ds))], dtype=float)
    best, best_state, bad, history, t0 = float("inf"), None, 0, [], time.time()
    for ep in range(cfg.epochs):
        model.train()
        tot, n = 0.0, 0
        for batch in tl:
            y = batch["y"].float().to(dev)
            x = {k: v.to(dev, non_blocking=True) for k, v in batch.items() if k != "y"}
            with torch.autocast(device_type="cuda", enabled=scaler.is_enabled()):
                loss = crit(model(**x).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(y)
            n += len(y)
        vlog = _predict_logits(model, vl, dev)
        vm = evaluate(yv, 1 / (1 + np.exp(-vlog)))
        history.append({"epoch": ep, "train_loss": tot / max(n, 1), "valid_log_loss": vm["log_loss"], "valid_auc": vm["auc"], "minutes": (time.time() - t0) / 60})
        print(f"ep {ep} train {tot / max(n, 1):.4f} valid logloss {vm['log_loss']:.4f} auc {vm['auc']:.3f} ({history[-1]['minutes']:.1f} min)")
        if vm["log_loss"] < best - 1e-4:
            best, bad = vm["log_loss"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= cfg.patience or (time.time() - t0) / 60 > cfg.max_minutes:
            break
    if best_state:
        model.load_state_dict(best_state)
    vlog = _predict_logits(model, vl, dev)
    return {"history": history, "valid_logits": vlog, "valid_y": yv, "temperature": TemperatureScaler().fit(vlog, yv)}


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
