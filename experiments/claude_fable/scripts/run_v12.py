"""Post-review baseline on index v1.2 (official split): the fixed trainer with the same
members as the pre-merge ensemble - frozen D-L (masked), C (masked), G-L/4 (masked) and
the positional GBM - 3 seeds each, then the ensemble with prior shift and penalty routing.

    ./run.sh python scripts/run_v12.py                 # everything still missing
    ./run.sh python scripts/run_v12.py --stages dl_mask g_l4 --seeds 3

Sequential on one GPU, resumable at (stage, seed): a finished run leaves its metrics
json and is skipped. Extra stages (`--extra name=...`) are read from STAGES too, so the
variant screening of the regularisation round uses the same queue.
"""

import argparse
import subprocess
import time
from pathlib import Path

import pandas as pd

import run_poc as rp
from xgpoc.data.index import load_index
from xgpoc.models.common import TrainConfig
from xgpoc.paths import INDEX

CACHE = Path("cache/full_v1")
OUT = Path("results/v12")

HEAD = dict(epochs=25, patience=25, ema_decay=0.999, max_minutes=15, batch_size=32, warmup_epochs=2.0)
CNN = dict(epochs=8, patience=8, ema_decay=0.999, max_minutes=25, batch_size=32, warmup_epochs=1.0)
FT = dict(epochs=5, patience=5, ema_decay=0.998, max_minutes=45, batch_size=16, warmup_epochs=1.0)
G_KW = dict(backbone="dinov2_l", n_blocks=4, mask="overlay", lr=3e-4, backbone_lr_mult=0.02)
PRE = ("frame_path", "frame_path_-400", "frame_path_-800")

# stage -> (runner, model name, TrainConfig overrides, runner kwargs)
STAGES = {
    "dl_mask": ("d", "d_frozen_dinov2_l_overlay", HEAD, dict(backbone="dinov2_l", mask="overlay", tag="_overlay")),
    "c_mask": ("c", "c_cnn_e2e_overlay", CNN, dict(mask="overlay", tag="_overlay")),
    "g_l4": ("g", "g_finetune_dinov2_l_b4_overlay", FT, dict(**G_KW, tag="_overlay")),
    # --- regularisation screening (one change at a time against g_l4) ---
    "g_rrc": ("g", "g_finetune_dinov2_l_b4_overlay_rrc", FT, dict(**G_KW, tag="_overlay_rrc", rrc=(0.85, 1.0))),
    "g_time": ("g", "g_finetune_dinov2_l_b4_overlay_time", FT, dict(**G_KW, tag="_overlay_time", path_cols=PRE)),
    "g_mt": ("g", "g_finetune_dinov2_l_b4_overlay_mt", {**FT, "aux_weight": 0.5}, dict(**G_KW, tag="_overlay_mt", mt=True)),
    "g_mixup": ("g", "g_finetune_dinov2_l_b4_overlay_mixup", {**FT, "mixup_alpha": 0.2}, dict(**G_KW, tag="_overlay_mixup")),
    "g_llrd": ("g", "g_finetune_dinov2_l_b8_overlay_llrd", {**FT, "llrd": 0.7}, dict(**{**G_KW, "n_blocks": 8, "backbone_lr_mult": 0.05}, tag="_overlay_llrd")),
    "dl_mt": ("d", "d_frozen_dinov2_l_overlay_mt", {**HEAD, "aux_weight": 0.5}, dict(backbone="dinov2_l", mask="overlay", tag="_overlay_mt", mt=True)),
    # --- combinations of the screening winners (RRC and the on-target head; the temporal
    #     offset draw and the multi-task head on the frozen features did not help) ---
    "g_combo": ("g", "g_finetune_dinov2_l_b4_overlay_combo", {**FT, "aux_weight": 0.5},
                dict(**G_KW, tag="_overlay_combo", rrc=(0.85, 1.0), mt=True)),
    "g_combo_mix": ("g", "g_finetune_dinov2_l_b4_overlay_combo_mix", {**FT, "aux_weight": 0.5, "mixup_alpha": 0.2},
                    dict(**G_KW, tag="_overlay_combo_mix", rrc=(0.85, 1.0), mt=True)),
    "g_combo_llrd": ("g", "g_finetune_dinov2_l_b8_overlay_combo_llrd", {**FT, "aux_weight": 0.5, "llrd": 0.7},
                     dict(**{**G_KW, "n_blocks": 8, "backbone_lr_mult": 0.05}, tag="_overlay_combo_llrd", rrc=(0.85, 1.0), mt=True)),
    "c_combo": ("c", "c_cnn_e2e_overlay_combo", {**CNN, "aux_weight": 0.5}, dict(mask="overlay", tag="_overlay_combo", rrc=(0.85, 1.0), mt=True)),
    # --- backbone check on the frozen head (features cached by scripts/extract_features.py) ---
    "dl3_mask": ("d", "d_frozen_dinov3_l_overlay", HEAD, dict(backbone="dinov3_l", mask="overlay", tag="_overlay")),
    "dlreg_mask": ("d", "d_frozen_dinov2_l_reg_overlay", HEAD, dict(backbone="dinov2_l_reg", mask="overlay", tag="_overlay")),
}


def done(out: Path, name: str) -> bool:
    return (out / f"{name}_metrics.json").exists()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="*", default=["gbm", "dl_mask", "c_mask", "g_l4"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--index", default=str(INDEX))
    a = ap.parse_args()
    df = rp.add_targets(load_index(Path(a.index)))
    data = [df[df.split == s].reset_index(drop=True) for s in ("train", "valid", "test")]
    print(f"index {a.index}: train/valid/test = {[len(d) for d in data]}", flush=True)

    for st in a.stages:
        if st == "gbm":
            out = a.out / "gbm" / "seed0"
            if not done(out, "a_tabular_gbm_sel"):
                subprocess.run(["./run.sh", "python", "scripts/build_positional_member.py", "--index", a.index, "--out", str(out)], check=True)
            continue
        runner, name, cfg_over, kw = STAGES[st]
        for s in range(a.seeds):
            out = a.out / st / f"seed{s}"
            if done(out, name):
                print(f"skip {out}/{name}")
                continue
            print(f"=== {out}/{name} ({time.strftime('%H:%M')}) ===", flush=True)
            t0 = time.time()
            rp.RUNNERS[runner](*data, CACHE, out, TrainConfig(seed=s, **cfg_over), **kw)
            print(f"--- {name} seed {s} done in {(time.time() - t0) / 60:.1f} min", flush=True)

    rows = []
    for f in sorted(a.out.glob("**/*_metrics.json")):
        m = pd.read_json(f, typ="series")
        rows.append({"run": str(f.parent.relative_to(a.out)), "model": m["model"], "auc": m["auc"],
                     "log_loss": m["log_loss"], "bss": m["brier_skill_score"], "ece": m["ece"],
                     "best_epoch": m.get("best_epoch", None)})
    if rows:
        pd.set_option("display.width", 250)
        print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(f"V12 QUEUE DONE ({time.strftime('%H:%M')})", flush=True)


if __name__ == "__main__":
    main()
