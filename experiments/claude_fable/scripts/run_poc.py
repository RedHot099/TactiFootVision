"""Train and evaluate one PoC on an index.

    ./run.sh python scripts/run_poc.py --index cache/dummy/index.parquet --poc a b c d e \
        --cache cache/dummy --out results/dummy --max-minutes 2

Positional cache (PoC a, b, e) must exist under <cache>/positional; DINOv2 features
(PoC d, e) are computed on first use into <cache>/features_<backbone>.npz.

Training-time options shared by the image models (c, g): `--path-cols` draws the input
frame from several pre-shot offsets, `--rrc` enables the aspect-preserving random crop,
`--mt` adds the "shot on target" auxiliary head, `--mixup` / `--ema` / `--llrd` pass
straight to the trainer. Evaluation always uses the t0 frame.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from xgpoc.data.datasets import FeatureDataset, ImageDataset, TokenDataset
from xgpoc.data.index import load_index
from xgpoc.features.cache import load_cache
from xgpoc.models.common import (TrainConfig, config_dict, fit_temperature, n_trainable, predict_torch,
                                 seed_all, train_torch, write_result)

ON_TARGET = "on_target"


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Derived training targets. `on_target` is an *outcome* (like `goal`), never an input."""
    if "label" in df:
        df = df.assign(**{ON_TARGET: (df["label"] == "Shots on target").astype(float)})
    return df


def run_a(tr, va, te, cache, out, cfg):
    from xgpoc.models.a_tabular import fit_predict
    tab, _, _ = load_cache(cache / "positional")
    res = {}
    for kind in ("logreg", "gbm"):
        pred, _ = fit_predict(tr, va, te, tab, kind)
        res[kind] = write_result(out, f"a_tabular_{kind}", pred, pred["p"].to_numpy(), {"n_train": int(len(tr) + len(va))})
    return res


def run_b(tr, va, te, cache, out, cfg):
    from xgpoc.models.b_set_transformer import SetTransformerXG
    seed_all(cfg.seed)
    _, tokens, _ = load_cache(cache / "positional")
    model = SetTransformerXG()
    fit = train_torch(model, TokenDataset(tr, tokens, train=True), TokenDataset(va, tokens), cfg)
    ds = TokenDataset(te, tokens)
    return write_result(out, "b_set_transformer", ds.df, predict_torch(model, ds, cfg, fit["temperature"]),
                        {"params": n_trainable(model), "history": fit["history"], "temperature": fit["temperature"].t})


def _train_kw(path_cols, rrc, mt):
    kw = {}
    if path_cols:
        kw["path_cols"] = tuple(path_cols)
    if rrc:
        kw["rrc_scale"] = tuple(rrc)
    if mt:
        kw["aux_col"] = ON_TARGET
    return kw


def run_c(tr, va, te, cache, out, cfg, backbone="convnext_tiny.fb_in22k_ft_in1k", size_wh=(448, 256),
          path_col="frame_path", tag="", mask="none", tta=False, path_cols=None, rrc=None, mt=False, lr=5e-5):
    from xgpoc.models.c_cnn_e2e import CnnXG
    seed_all(cfg.seed)
    model = CnnXG(backbone, n_out=2 if mt else 1)
    bs = 16 if size_wh[0] <= 512 else 8
    cfg = TrainConfig(**{**config_dict(cfg), "lr": lr, "batch_size": min(cfg.batch_size, bs)})
    kw = dict(size_wh=size_wh, path_col=path_col, mask=mask)
    fit = train_torch(model, ImageDataset(tr, train=True, **kw, **_train_kw(path_cols, rrc, mt)), ImageDataset(va, **kw), cfg)
    views = lambda d: [ImageDataset(d, **kw)] + ([ImageDataset(d, flip=True, **kw)] if tta else [])  # noqa: E731
    temp = fit_temperature(model, views(va), cfg) if tta else fit["temperature"]
    te_v, va_v = views(te), views(va)
    return write_result(out, "c_cnn_e2e" + tag, te_v[0].df, predict_torch(model, te_v, cfg, temp),
                        {"params": n_trainable(model), "backbone": backbone, "size": list(size_wh), "path_col": path_col,
                         "mask": mask, "tta": tta, "path_cols": list(path_cols or ()), "rrc": list(rrc or ()), "mt": mt,
                         "cfg": config_dict(cfg), "best_epoch": fit["best_epoch"], "history": fit["history"],
                         "temperature": temp.t},
                        valid_df=va_v[0].df, p_valid=predict_torch(model, va_v, cfg, temp))


def run_f(tr, va, te, cache, out, cfg, backbone="dinov2_s", offsets=(-1600, -1200, -800, -400, 0)):
    from xgpoc.models.d_foundation import extract_patch_features
    from xgpoc.models.f_temporal import TemporalFeatureDataset, TemporalXG
    seed_all(cfg.seed)
    df_all = pd.concat([tr, va, te])
    feats = {}
    for o in offsets:
        col = "frame_path" if o == 0 else f"frame_path_{o:+d}"
        feats[o] = extract_patch_features(df_all, cache / f"features_{backbone}_off{o}.npz", backbone,
                                          device=cfg.device, path_col=col)
    dim = next(iter(feats[0].values())).shape[-1]
    seed_all(cfg.seed)
    model = TemporalXG(dim, n_frames=len(offsets))
    cfg = TrainConfig(**{**config_dict(cfg), "lr": 1e-4})
    fit = train_torch(model, TemporalFeatureDataset(tr, feats), TemporalFeatureDataset(va, feats), cfg)
    ds = TemporalFeatureDataset(te, feats)
    return write_result(out, f"f_temporal_{backbone}", ds.df, predict_torch(model, ds, cfg, fit["temperature"]),
                        {"params": n_trainable(model), "backbone": backbone, "offsets": list(offsets),
                         "n_train": len(TemporalFeatureDataset(tr, feats)), "history": fit["history"], "temperature": fit["temperature"].t})


def _features(df_all, cache, backbone, cfg, mask="none", flip=False):
    from xgpoc.models.d_foundation import extract_patch_features
    name = f"features_{backbone}" + ("" if mask == "none" else f"_{mask}") + ("_flip" if flip else "")
    return extract_patch_features(df_all, cache / f"{name}.npz", backbone, device=cfg.device, mask=mask, flip=flip)


def _n_prefix(backbone: str) -> int:
    return 5 if "reg" in backbone or "dinov3" in backbone else 1


def run_d(tr, va, te, cache, out, cfg, backbone="dinov2_s", mask="none", tta=False, tag="", mt=False, lr=1e-4):
    from xgpoc.models.d_foundation import AttentionPoolHead
    seed_all(cfg.seed)
    df_all = pd.concat([tr, va, te])
    feats = _features(df_all, cache, backbone, cfg, mask=mask)
    alt = _features(df_all, cache, backbone, cfg, mask=mask, flip=True) if tta else None
    dim = next(iter(feats.values())).shape[-1]
    seed_all(cfg.seed)
    model = AttentionPoolHead(dim, n_out=2 if mt else 1, n_prefix=_n_prefix(backbone))
    cfg = TrainConfig(**{**config_dict(cfg), "lr": lr})
    fit = train_torch(model, FeatureDataset(tr, feats, alt=alt, train=True, aux_col=ON_TARGET if mt else None),
                      FeatureDataset(va, feats), cfg)
    views = lambda d: [FeatureDataset(d, feats)] + ([FeatureDataset(d, feats, alt=alt)] if tta else [])  # noqa: E731
    temp = fit_temperature(model, views(va), cfg) if tta else fit["temperature"]
    te_v, va_v = views(te), views(va)
    return write_result(out, f"d_frozen_{backbone}{tag}", te_v[0].df, predict_torch(model, te_v, cfg, temp),
                        {"params": n_trainable(model), "backbone": backbone, "mask": mask, "tta": tta, "mt": mt,
                         "cfg": config_dict(cfg), "best_epoch": fit["best_epoch"], "history": fit["history"],
                         "temperature": temp.t},
                        valid_df=va_v[0].df, p_valid=predict_torch(model, va_v, cfg, temp))


def run_g(tr, va, te, cache, out, cfg, backbone="dinov2_b", n_blocks=4, mask="none", tta=False, tag="",
          backbone_lr_mult=0.1, lr=3e-4, path_cols=None, rrc=None, mt=False):
    """PoC G: PoC D with the top blocks of the backbone unfrozen (see FinetuneViTXG)."""
    from xgpoc.models.d_foundation import BACKBONES, FinetuneViTXG
    seed_all(cfg.seed)
    model = FinetuneViTXG(backbone, n_blocks=n_blocks, n_out=2 if mt else 1)
    size_wh = BACKBONES[backbone][1]
    cfg = TrainConfig(**{**config_dict(cfg), "lr": lr, "batch_size": min(cfg.batch_size, 8 if "_l" in backbone else 16)})
    kw = dict(size_wh=size_wh, mask=mask)
    fit = train_torch(model, ImageDataset(tr, train=True, **kw, **_train_kw(path_cols, rrc, mt)), ImageDataset(va, **kw), cfg,
                      backbone_lr_mult=backbone_lr_mult)
    views = lambda d: [ImageDataset(d, **kw)] + ([ImageDataset(d, flip=True, **kw)] if tta else [])  # noqa: E731
    temp = fit_temperature(model, views(va), cfg) if tta else fit["temperature"]
    te_v, va_v = views(te), views(va)
    return write_result(out, f"g_finetune_{backbone}_b{n_blocks}{tag}", te_v[0].df,
                        predict_torch(model, te_v, cfg, temp),
                        {"params": n_trainable(model), "backbone": backbone, "n_blocks": n_blocks, "mask": mask,
                         "tta": tta, "backbone_lr_mult": backbone_lr_mult, "lr": lr, "path_cols": list(path_cols or ()),
                         "rrc": list(rrc or ()), "mt": mt, "cfg": config_dict(cfg), "best_epoch": fit["best_epoch"],
                         "history": fit["history"], "temperature": temp.t},
                        valid_df=va_v[0].df, p_valid=predict_torch(model, va_v, cfg, temp))


def run_e(tr, va, te, cache, out, cfg, backbone="dinov2_s"):
    from xgpoc.models.d_foundation import BACKBONES
    from xgpoc.models.e_hybrid import HybridXG
    seed_all(cfg.seed)
    feats = _features(pd.concat([tr, va, te]), cache, backbone, cfg)
    _, _, boxes = load_cache(cache / "positional")
    w, h = BACKBONES[backbone][1]
    dim = next(iter(feats.values())).shape[-1]
    seed_all(cfg.seed)
    model = HybridXG(dim, grid_hw=(h // 14, w // 14) if "dinov2" in backbone else (h // 16, w // 16))
    cfg = TrainConfig(**{**config_dict(cfg), "lr": 1e-4})
    fit = train_torch(model, FeatureDataset(tr, feats, boxes), FeatureDataset(va, feats, boxes), cfg)
    ds = FeatureDataset(te, feats, boxes)
    return write_result(out, f"e_hybrid_{backbone}", ds.df, predict_torch(model, ds, cfg, fit["temperature"]),
                        {"params": n_trainable(model), "backbone": backbone, "history": fit["history"], "temperature": fit["temperature"].t})


RUNNERS = {"a": run_a, "b": run_b, "c": run_c, "d": run_d, "e": run_e, "f": run_f, "g": run_g}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--poc", nargs="+", default=list(RUNNERS))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=None, help="default: run the whole cycle")
    ap.add_argument("--max-minutes", type=float, default=5.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.0)
    ap.add_argument("--mixup", type=float, default=0.0)
    ap.add_argument("--llrd", type=float, default=1.0)
    ap.add_argument("--aux-weight", type=float, default=0.5)
    ap.add_argument("--select-on", default="calibrated", choices=["calibrated", "raw", "auc"])
    ap.add_argument("--img-size", default=None, help="WxH for poc c, e.g. 672x384")
    ap.add_argument("--frame-col", default="frame_path", help="input frame column for poc c")
    ap.add_argument("--path-cols", nargs="*", default=None, help="training frames drawn at random from these columns (c, g)")
    ap.add_argument("--rrc", nargs=2, type=float, default=None, metavar=("LO", "HI"), help="random crop area range (c, g)")
    ap.add_argument("--mt", action="store_true", help="multi-task with the on-target head (c, d, g)")
    ap.add_argument("--mask", default="none", choices=["none", "overlay", "overlay_only", "graphics_only"],
                    help="blank the broadcast overlay (c, d, g)")
    ap.add_argument("--tta", action="store_true", help="horizontal-flip test-time augmentation (c, d, g)")
    ap.add_argument("--backbone", default=None, help="backbone override for poc d and g")
    ap.add_argument("--n-blocks", type=int, default=4, help="unfrozen top blocks for poc g")
    ap.add_argument("--lr", type=float, default=None, help="head LR override (c, d, g)")
    ap.add_argument("--lr-mult", type=float, default=0.02, help="backbone LR multiplier for poc g")
    ap.add_argument("--tag", default="", help="suffix for the result name")
    a = ap.parse_args()
    df = add_targets(load_index(Path(a.index)))
    tr, va, te = (df[df["split"] == s].reset_index(drop=True) for s in ("train", "valid", "test"))
    print(f"shots train/valid/test = {len(tr)}/{len(va)}/{len(te)}, goal rate {df['goal'].mean():.3f}")
    cfg = TrainConfig(epochs=a.epochs, patience=a.patience if a.patience is not None else a.epochs, max_minutes=a.max_minutes,
                      batch_size=a.batch_size, seed=a.seed, warmup_epochs=a.warmup, ema_decay=a.ema, mixup_alpha=a.mixup,
                      llrd=a.llrd, aux_weight=a.aux_weight if a.mt else 0.0, select_on=a.select_on)
    summary = {}
    for p in a.poc:
        kw = {}
        if p in ("c", "d", "g"):
            if a.mask != "none":
                kw["mask"] = a.mask
            if a.tta:
                kw["tta"] = True
            if a.mt:
                kw["mt"] = True
            if a.lr is not None:
                kw["lr"] = a.lr
            if p in ("d", "g") and a.backbone:
                kw["backbone"] = a.backbone
            if p == "g":
                kw["n_blocks"] = a.n_blocks
                kw["backbone_lr_mult"] = a.lr_mult
            if p in ("c", "g"):
                if a.path_cols:
                    kw["path_cols"] = a.path_cols
                if a.rrc:
                    kw["rrc"] = a.rrc
            auto_tag = ("_" + a.mask if a.mask != "none" else "") + ("_tta" if a.tta else "")
            if p == "c" and a.img_size:
                w, h = a.img_size.split("x")
                kw["size_wh"] = (int(w), int(h))
                auto_tag = "_" + a.img_size + auto_tag
            if p == "c" and a.frame_col != "frame_path":
                kw["path_col"] = a.frame_col
                auto_tag += "_" + a.frame_col.replace("frame_path_", "off")
            kw["tag"] = auto_tag + a.tag
        summary[p] = RUNNERS[p](tr, va, te, Path(a.cache), Path(a.out), cfg, **kw)
    rows = []
    for f in sorted(Path(a.out).glob("*_metrics.json")):
        m = json.loads(f.read_text())
        rows.append({k: m[k] for k in ("model", "n", "brier", "brier_skill_score", "log_loss", "auc", "ece")})
    table = pd.DataFrame(rows)
    table.to_csv(Path(a.out) / "summary.csv", index=False)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
