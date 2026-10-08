"""Render comparison figures: shot frame + per-model xG predictions (test set).

Models shown as seed-ensembles. Single-hue bars (identity is on the axis, colour
carries no information); base rate marked; outcome stated in the title text.

    ./run.sh python scripts/make_situation_figures.py --out results_reports/figures
"""

import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK = "#1a1a2e"
MUTED = "#6b7280"
BAR = "#5b8db8"
GRID = "#e5e7eb"

# preset -> {label: list of prediction csv globs}; several globs = pooled seed ensemble
PRESETS = {
    "architektury": {
        "A pozycyjny (GBM)": ["results/big_v1/seed*/a_tabular_gbm_test_predictions.csv"],
        "B Set Transformer": ["results/big_v1/seed*/b_set_transformer_test_predictions.csv"],
        "C ConvNeXt 448": ["results/big_v1/c8/seed*/c_cnn_e2e_test_predictions.csv"],
        "D DINOv2-L": ["results/night_v1/dl/seed*/d_frozen_dinov2_l_test_predictions.csv"],
        "E hybryda": ["results/big_v1/seed*/e_hybrid_dinov2_s_test_predictions.csv"],
        "F temporalny": ["results/night_v1/f/seed*/f_temporal_dinov2_s_test_predictions.csv"],
        "Ensemble C+D-L": ["results/big_v1/c8/seed*/c_cnn_e2e_test_predictions.csv",
                           "results/night_v1/dl/seed*/d_frozen_dinov2_l_test_predictions.csv"],
        "GBM po poprawkach": ["results/before_after/GBM_po_poprawkach_test_predictions.csv"],
        "D po poprawkach": ["results/before_after/D_po_poprawkach_(+pen)_test_predictions.csv"],
    },
    # after the pre-merge pass: everything below is trained on masked frames, so no
    # model in this preset can read the scoreboard
    "finalne": {
        "A pozycyjny (GBM)": ["results/ensemble/gbm_test_predictions.csv"],
        "C ConvNeXt + maska": ["results/ensemble/c_mask_test_predictions.csv"],
        "D DINOv2-L zamrozony + maska": ["results/ensemble/dl_mask_test_predictions.csv"],
        "G fine-tune DINOv2-L/4": ["results/ensemble/g_final__dinov2_l_b4_lr0.0003_m0.02_test_predictions.csv"],
        "G fine-tune DINOv2-L/12": ["results/ensemble/g_final__dinov2_l_b12_lr0.0001_m0.02_test_predictions.csv"],
        "Stary ensemble C+D-L": ["results/ensemble/OLD_ensemble_C_DL_test_predictions.csv"],
        "ENSEMBLE finalny": ["results/ensemble/best_ensemble_test_predictions.csv"],
    },
    # index v1.2, CV-trained final models (results_reports/v12_quality_REPORT.md); use with --index index_v1_2.parquet
    "v12": {
        "A pozycyjny (GBM)": ["results/v12_cv_ensemble/gbm_test_predictions.csv"],
        "C ConvNeXt-T": ["results/v12_cv_ensemble/c_combo_test_predictions.csv"],
        "D DINOv2-L zamrozony": ["results/v12_cv_ensemble/dl_mask_test_predictions.csv"],
        "G DINOv2-L dostrajany": ["results/v12_cv_ensemble/g_combo_test_predictions.csv"],
        "G + mixup": ["results/v12_cv_ensemble/g_combo_mix_test_predictions.csv"],
        "ENSEMBLE": ["results/v12_cv_ensemble/best_ensemble_test_predictions.csv"],
    },
}
MODELS = PRESETS["architektury"]


def ensemble(globs: list[str]) -> pd.Series:
    frames = []
    for g in globs:
        for f in sorted(glob.glob(g)):
            frames.append(pd.read_csv(f).drop_duplicates("shot_id").set_index("shot_id")["p"])
    return pd.concat(frames, axis=1).mean(axis=1)


def pick_situations(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Return (shot_id, situation description) pairs covering distinct cases.

    Situations are ranked by the last model in the preset, which is the reference
    ensemble; the preset therefore has to keep it last."""
    REF = list(MODELS)[-1]
    picks = []
    d = df.copy()
    d["spread"] = d[list(MODELS)].max(axis=1) - d[list(MODELS)].min(axis=1)
    def take(mask, sort_col, asc, desc_txt):
        cand = d[mask].sort_values(sort_col, ascending=asc)
        if len(cand):
            r = cand.iloc[0]
            picks.append((r.name, desc_txt))
            d.drop(index=r.name, inplace=True)
    open_play = ~d.is_penalty.astype(bool) & d.has_ball.astype(bool)  # penalties are routed, so they would win every ranking
    take(d.goal & (d.dist_to_goal < 10) & open_play, REF, False, "Gol z bliska (dobitka/sytuacja sam na sam)")
    take(d.goal & (d.dist_to_goal > 22) & open_play, REF, True, "Gol z dystansu (niska szansa a priori)")
    take(~d.goal & (d.dist_to_goal < 10) & open_play, REF, False, "Zmarnowana wielka okazja (strzał z pola bramkowego)")
    take(~d.goal & (d.dist_to_goal > 25) & open_play, REF, True, "Strzał z dużego dystansu (typowe niskie xG)")
    take(d.is_penalty.astype(bool), REF, False, "Rzut karny")
    take(d.spread > 0, "spread", False, "Największa niezgoda między modelami")
    return picks


def render(row: pd.Series, desc: str, out: Path, base_rate: float):
    fig = plt.figure(figsize=(14.5, 4.6), dpi=150, facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=0.30, left=0.02, right=0.97, top=0.82, bottom=0.12)
    ax_img = fig.add_subplot(gs[0])
    img = plt.imread(row["frame_path"])
    ax_img.imshow(img)
    ax_img.axis("off")
    outcome = "GOL" if row["goal"] else "BRAK GOLA"
    fig.suptitle(f"{desc}  —  wynik: {outcome}", x=0.02, ha="left", fontsize=13, fontweight="bold", color=INK)
    meta = f"{row['game_short']}  |  {row['half']}. połowa, {int(row['position_ms'] // 60000)}'  |  dystans wg homografii: {row['dist_to_goal']:.0f} m"
    fig.text(0.02, 0.855, meta, fontsize=9.5, color=MUTED)

    ax = fig.add_subplot(gs[1])
    labels = list(MODELS)
    vals = [row[m] for m in labels]
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, height=0.62, color=BAR, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + 0.012, yi, f"{v:.2f}", va="center", fontsize=9.5, color=INK)
    ax.axvline(base_rate, color=MUTED, lw=1, ls="--", zorder=2)
    ax.text(base_rate + 0.005, len(labels) - 0.25, f"baza {base_rate:.2f}", fontsize=8, color=MUTED)
    ax.set_yticks(y, labels, fontsize=10, color=INK)
    ax.set_xlim(0, max(0.65, max(vals) + 0.1))
    ax.set_xlabel("przewidywane xG (prawdopodobieństwo gola)", fontsize=9.5, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    fig.savefig(out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results_reports/figures"))
    ap.add_argument("--index", default="/home/kuba/projects/football_data/xg_shots_full/soccernet_720p/index_v1_1.parquet")
    ap.add_argument("--preset", default="architektury", choices=list(PRESETS))
    a = ap.parse_args()
    global MODELS
    MODELS = PRESETS[a.preset]
    a.out.mkdir(parents=True, exist_ok=True)
    idx = pd.read_parquet(a.index)
    test = idx[idx.split == "test"].set_index("shot_id")
    tab = pd.read_parquet("cache/full_v1/positional/tabular.parquet").drop_duplicates("shot_id").set_index("shot_id")
    df = test.join(tab[["dist_to_goal", "has_ball"]], how="inner")
    for name, globs in MODELS.items():
        df[name] = ensemble(globs)
    df = df.dropna(subset=list(MODELS))
    df["game_short"] = df.game.str.split("/").str[-1].str.slice(19)
    base_rate = float(df.goal.mean())
    for i, (sid, desc) in enumerate(pick_situations(df), 1):
        f = a.out / f"sytuacja_{i}.png"
        render(df.loc[sid], desc, f, base_rate)
        print(f, "<-", desc, "| goal:", bool(df.loc[sid, "goal"]))


if __name__ == "__main__":
    main()
