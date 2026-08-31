"""Train and evaluate one PoC on an index.

    ./run.sh python scripts/run_poc.py --index cache/dummy/index.parquet --poc a b c d e \
        --cache cache/dummy --out results/dummy --max-minutes 2

Positional cache (PoC a, b, e) must exist under <cache>/positional; DINOv2 features
(PoC d, e) are computed on first use into <cache>/features_<backbone>.npz.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from xgpoc.data.datasets import FeatureDataset, ImageDataset, TokenDataset
from xgpoc.data.index import load_index
from xgpoc.features.cache import load_cache
from xgpoc.models.common import (TrainConfig, config_dict, fit_temperature, n_trainable, predict_torch,
                                 train_torch, write_result)


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
    _, tokens, _ = load_cache(cache / "positional")
    model = SetTransformerXG()
    fit = train_torch(model, TokenDataset(tr, tokens, train=True), TokenDataset(va, tokens), cfg)
    ds = TokenDataset(te, tokens)
    return write_result(out, "b_set_transformer", ds.df, predict_torch(model, ds, cfg, fit["temperature"]),
                        {"params": n_trainable(model), "history": fit["history"], "temperature": fit["temperature"].t})


def run_c(tr, va, te, cache, out, cfg, backbone="convnext_tiny.fb_in22k_ft_in1k", size_wh=(448, 256),
          path_col="frame_path", tag="", mask="none", tta=False):
    from xgpoc.models.c_cnn_e2e import CnnXG
    model = CnnXG(backbone)
    bs = 16 if size_wh[0] <= 512 else 8
    cfg = TrainConfig(**{**config_dict(cfg), "lr": 5e-5, "batch_size": min(cfg.batch_size, bs)})
    kw = dict(size_wh=size_wh, path_col=path_col, mask=mask)
    fit = train_torch(model, ImageDataset(tr, train=True, **kw), ImageDataset(va, **kw), cfg)
    views = lambda d: [ImageDataset(d, **kw)] + ([ImageDataset(d, flip=True, **kw)] if tta else [])  # noqa: E731
    temp = fit_temperature(model, views(va), cfg) if tta else fit["temperature"]
    te_v, va_v = views(te), views(va)
    return write_result(out, "c_cnn_e2e" + tag, te_v[0].df, predict_torch(model, te_v, cfg, temp),
                        {"params": n_trainable(model), "backbone": backbone, "size": list(size_wh), "path_col": path_col,
                         "mask": mask, "tta": tta, "history": fit["history"], "temperature": temp.t},
                        valid_df=va_v[0].df, p_valid=predict_torch(model, va_v, cfg, temp))


def run_f(tr, va, te, cache, out, cfg, backbone="dinov2_s", offsets=(-1600, -1200, -800, -400, 0)):
    from xgpoc.models.d_foundation import extract_patch_features
    from xgpoc.models.f_temporal import TemporalFeatureDataset, TemporalXG
    df_all = pd.concat([tr, va, te])
    feats = {}
    for o in offsets:
        col = "frame_path" if o == 0 else f"frame_path_{o:+d}"
        feats[o] = extract_patch_features(df_all, cache / f"features_{backbone}_off{o}.npz", backbone,
                                          device=cfg.device, path_col=col)
    dim = next(iter(feats[0].values())).shape[-1]
    model = TemporalXG(dim, n_frames=len(offsets))
    cfg = TrainConfig(**{**config_dict(cfg), "epochs": max(cfg.epochs, 30), "lr": 1e-4})
    fit = train_torch(model, TemporalFeatureDataset(tr, feats), TemporalFeatureDataset(va, feats), cfg)
    ds = TemporalFeatureDataset(te, feats)
    return write_result(out, f"f_temporal_{backbone}", ds.df, predict_torch(model, ds, cfg, fit["temperature"]),
                        {"params": n_trainable(model), "backbone": backbone, "offsets": list(offsets),
                         "n_train": len(TemporalFeatureDataset(tr, feats)), "history": fit["history"], "temperature": fit["temperature"].t})


def _features(df_all, cache, backbone, cfg, mask="none", flip=False):
    from xgpoc.models.d_foundation import extract_patch_features
    name = f"features_{backbone}" + ("" if mask == "none" else f"_{mask}") + ("_flip" if flip else "")
    return extract_patch_features(df_all, cache / f"{name}.npz", backbone, device=cfg.device, mask=mask, flip=flip)


def run_d(tr, va, te, cache, out, cfg, backbone="dinov2_s", mask="none", tta=False, tag=""):
    from xgpoc.models.d_foundation import AttentionPoolHead
    df_all = pd.concat([tr, va, te])
    feats = _features(df_all, cache, backbone, cfg, mask=mask)
    alt = _features(df_all, cache, backbone, cfg, mask=mask, flip=True) if tta else None
    dim = next(iter(feats.values())).shape[-1]
    model = AttentionPoolHead(dim)
    cfg = TrainConfig(**{**config_dict(cfg), "epochs": max(cfg.epochs, 30), "lr": 1e-4})
    fit = train_torch(model, FeatureDataset(tr, feats, alt=alt, train=True), FeatureDataset(va, feats), cfg)
    views = lambda d: [FeatureDataset(d, feats)] + ([FeatureDataset(d, feats, alt=alt)] if tta else [])  # noqa: E731
    temp = fit_temperature(model, views(va), cfg) if tta else fit["temperature"]
    te_v, va_v = views(te), views(va)
    return write_result(out, f"d_frozen_{backbone}{tag}", te_v[0].df, predict_torch(model, te_v, cfg, temp),
                        {"params": n_trainable(model), "backbone": backbone, "mask": mask, "tta": tta,
                         "history": fit["history"], "temperature": temp.t},
                        valid_df=va_v[0].df, p_valid=predict_torch(model, va_v, cfg, temp))


def run_g(tr, va, te, cache, out, cfg, backbone="dinov2_b", n_blocks=4, mask="none", tta=False, tag="",
          backbone_lr_mult=0.1, lr=3e-4):
    """PoC G: PoC D with the top blocks of the backbone unfrozen (see FinetuneViTXG)."""
    from xgpoc.models.d_foundation import BACKBONES, FinetuneViTXG
    model = FinetuneViTXG(backbone, n_blocks=n_blocks)
    size_wh = BACKBONES[backbone][1]
    cfg = TrainConfig(**{**config_dict(cfg), "lr": lr, "batch_size": min(cfg.batch_size, 8 if "_l" in backbone else 16)})
    kw = dict(size_wh=size_wh, mask=mask)
    fit = train_torch(model, ImageDataset(tr, train=True, **kw), ImageDataset(va, **kw), cfg,
                      backbone_lr_mult=backbone_lr_mult)
    views = lambda d: [ImageDataset(d, **kw)] + ([ImageDataset(d, flip=True, **kw)] if tta else [])  # noqa: E731
    temp = fit_temperature(model, views(va), cfg) if tta else fit["temperature"]
    te_v, va_v = views(te), views(va)
    return write_result(out, f"g_finetune_{backbone}_b{n_blocks}{tag}", te_v[0].df,
                        predict_torch(model, te_v, cfg, temp),
                        {"params": n_trainable(model), "backbone": backbone, "n_blocks": n_blocks, "mask": mask,
                         "tta": tta, "backbone_lr_mult": backbone_lr_mult, "lr": lr, "history": fit["history"],
                         "temperature": temp.t},
                        valid_df=va_v[0].df, p_valid=predict_torch(model, va_v, cfg, temp))


def run_e(tr, va, te, cache, out, cfg, backbone="dinov2_s"):
    from xgpoc.models.d_foundation import BACKBONES
    from xgpoc.models.e_hybrid import HybridXG
    feats = _features(pd.concat([tr, va, te]), cache, backbone, cfg)
    _, _, boxes = load_cache(cache / "positional")
    w, h = BACKBONES[backbone][1]
    dim = next(iter(feats.values())).shape[-1]
    model = HybridXG(dim, grid_hw=(h // 14, w // 14) if "dinov2" in backbone else (h // 16, w // 16))
    cfg = TrainConfig(**{**config_dict(cfg), "epochs": max(cfg.epochs, 30), "lr": 1e-4})
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
    ap.add_argument("--max-minutes", type=float, default=5.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--img-size", default=None, help="WxH for poc c, e.g. 672x384")
    ap.add_argument("--frame-col", default="frame_path", help="input frame column for poc c")
    ap.add_argument("--mask", default="none", choices=["none", "overlay", "overlay_only", "graphics_only"],
                    help="blank the broadcast overlay (c, d, g)")
    ap.add_argument("--tta", action="store_true", help="horizontal-flip test-time augmentation (c, d, g)")
    ap.add_argument("--backbone", default=None, help="backbone override for poc d and g")
    ap.add_argument("--n-blocks", type=int, default=4, help="unfrozen top blocks for poc g")
    a = ap.parse_args()
    df = load_index(Path(a.index))
    tr, va, te = (df[df["split"] == s].reset_index(drop=True) for s in ("train", "valid", "test"))
    print(f"shots train/valid/test = {len(tr)}/{len(va)}/{len(te)}, goal rate {df['goal'].mean():.3f}")
    cfg = TrainConfig(epochs=a.epochs, max_minutes=a.max_minutes, batch_size=a.batch_size, seed=a.seed)
    summary = {}
    for p in a.poc:
        kw = {}
        if p in ("c", "d", "g"):
            if a.mask != "none":
                kw["mask"] = a.mask
            if a.tta:
                kw["tta"] = True
            if p in ("d", "g") and a.backbone:
                kw["backbone"] = a.backbone
            if p == "g":
                kw["n_blocks"] = a.n_blocks
            if p in ("d", "g") and (a.mask != "none" or a.tta):
                kw["tag"] = ("_" + a.mask if a.mask != "none" else "") + ("_tta" if a.tta else "")
        if p == "c" and (a.img_size or a.frame_col != "frame_path"):
            if a.img_size:
                w, h = a.img_size.split("x")
                kw["size_wh"] = (int(w), int(h))
            kw["path_col"] = a.frame_col
            kw["tag"] = ("_" + a.img_size if a.img_size else "") + ("_" + a.frame_col.replace("frame_path_", "off") if a.frame_col != "frame_path" else "")
        if p == "c" and (a.mask != "none" or a.tta):
            kw["tag"] = kw.get("tag", "") + ("_" + a.mask if a.mask != "none" else "") + ("_tta" if a.tta else "")
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
