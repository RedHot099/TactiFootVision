"""Figures of the xg-compare report (xg-compare ticket 03; CPU, no frame read).

    ./run.sh --no-sync python scripts/xgc_report_figures.py

From the run of ticket 02 (results/xgc_compare/analysis, each file checked against its completion record):

- `family_forest.png`: the pre-declared family on the primary population. Gap items: the adjusted F - V effect
  (control z_D, main scale) with its 95% interval, beside the D - V and P - V references, against the 0.15-logit
  threshold band. Correlation items 8 and 9 and log-loss items 10 and 16 on their own axes and thresholds.
- `agreement_heatmap.png`: Spearman correlations of the raw probabilities between every pair of methods, primary
  population, overall.

Both carry "Data: StatsBomb." Writes results_reports/xg_compare_2026-10/figures (or `--out`) with figures.json:
input hashes, code hashes (`xgpoc.compare.records.code_hashes`), git state and output hashes. The figures are
deterministic in their inputs, so a rerun overwrites them.
"""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.ticker
import numpy as np
import pandas as pd
from report_complexity import GRID, INK, INK2, MUTED, SURFACE, _axes_style
from xgc_analysis import RUN_OUT, verified
from xgpoc.bundle import sha256_file
from xgpoc.compare import pipeline as pl
from xgpoc.compare import protocol as pr
from xgpoc.compare import records as xr
from xgpoc.detector import records

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "results_reports/xg_compare_2026-10/figures"
FAMILY_FILE, AGREEMENT_FILE = RUN_OUT / "family.csv", RUN_OUT / "agreement.csv"

# Categorical slots 1-3 of the dataviz reference palette (validated light: adjacent CVD dE >= 9.2; slot 3 is below
# 3:1 contrast, so the legend and the report's tables carry it), the threshold band one step off the surface, and
# the blue sequential ramp 100..700 for the heatmap.
SERIES = {"F-V": "#2a78d6", "D-V": "#eb6834", "P-V": "#1baf7a"}
SERIES_PREFIX = {"F-V": "effect", "D-V": "ref_DV", "P-V": "ref_PV"}    # column prefixes of family.csv
SERIES_LABEL = {"F-V": "F − V (obraz − StatsBomb)", "D-V": "D − V (odległość i kąt − StatsBomb)",
                "P-V": "P − V (pozycje − StatsBomb)"}
BAND = "#f0efec"
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ITEM_LABEL = {"2": "dobitka do 5 s", "3": "strzał głową", "4": "po dośrodkowaniu", "5": "≥ 2 obrońców w stożku",
              "6a": "bramkarz ≥ 4 jardy od linii", "7": "rzut wolny bezpośredni", "11": "pod presją",
              "12": "sam na sam", "14": "spoza pola karnego", "15": "bramkarz niewykryty na klatce",
              "8": "ρ(F, D) − ρ(F, V)", "9": "[ρ(F, V) − ρ(P, V)]: głowa − noga",
              "10": "ciasny kadr: LL(F) − LL(V)", "16": "duża reszta czasu: LL(F) − LL(V)"}
METHODS = ["V", "P", "E", "D", "F", "F-400", "A1"]
METHOD_LABEL = {"V": "V StatsBomb", "P": "P pozycje", "E": "E zdarzenia", "D": "D odległość i kąt", "F": "F obraz",
                "F-400": "F−400", "A1": "A1 obraz + detektor"}
PANELS = (("gap", "Luka obrazu wobec StatsBomb ponad geometrię (logit, efekt skorygowany o z_D)", ("F-V", "D-V", "P-V")),
          ("correlation", "Zgodność (różnica korelacji Spearmana)", ("F-V",)),
          ("log_loss", "Kto trafniejszy (różnica log-loss na strzał)", ("F-V", "D-V", "P-V")))
KINDS = {"gap": ("gap",), "correlation": ("correlation_difference", "correlation_contrast"), "log_loss": ("log_loss",)}
THRESHOLD = {"gap": pr.THRESHOLDS["family_logit"], "correlation": pr.THRESHOLDS["correlation"],
             "log_loss": pr.THRESHOLDS["log_loss"]}


def minus(text: str) -> str:
    return text.replace("-", "−")


def tier(row) -> str:
    """BH and BY rejections of a family item, for its axis label."""
    return " · BH, BY" if row.reject_by else " · BH" if row.reject_bh else ""


def series_columns(name: str) -> tuple[str, str, str]:
    prefix = SERIES_PREFIX[name]
    return f"{prefix}_point", f"{prefix}_lo", f"{prefix}_hi"


def forest(family: pd.DataFrame, out: Path) -> None:
    """Three stacked panels, one per item kind, each on its own axis and threshold (module docstring)."""
    groups = {panel: family[family.kind.isin(KINDS[panel])] for panel, _, _ in PANELS}
    heights = [len(groups[p]) + 1.2 for p, _, _ in PANELS]
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 9.6), gridspec_kw={"height_ratios": heights, "hspace": 0.62},
                             facecolor=SURFACE)
    for ax, (panel, title, series) in zip(axes, PANELS, strict=True):
        rows = groups[panel].iloc[::-1].reset_index(drop=True)
        _axes_style(ax)
        ax.grid(axis="y", visible=False)
        threshold = THRESHOLD[panel]
        ax.axvspan(-threshold, threshold, color=BAND, zorder=0, lw=0)
        ax.axvline(0, color="#c3c2b7", lw=1, zorder=1)
        offsets = np.linspace(0.22, -0.22, len(series)) if len(series) > 1 else [0.0]
        for name, dy in zip(series, offsets, strict=True):
            point, lo, hi = series_columns(name)
            if point not in rows or rows[point].isna().all():
                continue
            y = np.arange(len(rows)) + dy
            ax.hlines(y, rows[lo], rows[hi], color=SERIES[name], lw=2, zorder=2)
            ax.scatter(rows[point], y, s=40, color=SERIES[name], edgecolor=SURFACE, linewidth=2, zorder=3,
                       label=SERIES_LABEL[name])
        ax.set_yticks(np.arange(len(rows)), [f"{r.item}  {ITEM_LABEL[r.item]}{tier(r)}" for r in rows.itertuples()],
                      fontsize=9, color=INK)
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.set_title(title, fontsize=10, loc="left", color=INK, pad=8)
        ax.set_xlabel(minus(f"pasmo progu: ±{threshold:g}"), fontsize=8.5, color=MUTED)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: minus(f"{v:+g}") if v else "0"))
    fig.subplots_adjust(top=0.9, bottom=0.07)
    draws = f"{pr.N_BOOT:,d}".replace(",", " ")
    fig.suptitle(f"Rodzina z góry zadeklarowana: populacja główna, 95% CI z {draws} losowań meczów", fontsize=10.5,
                 color=INK, x=0.02, ha="left", y=0.985)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0.02, 0.96), ncol=3,
               labelcolor=INK2, handletextpad=0.3, columnspacing=1.2)
    fig.text(0.02, 0.01, f"Badanie eksploracyjne: etykiety znane przed jego rozpoczęciem. BH/BY: odrzucenie przy q = {pr.Q_BH:g}. "
             f"{pl.NOTICE}", fontsize=8, color=INK2)
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def heatmap(agreement: pd.DataFrame, out: Path) -> None:
    """The overall Spearman matrix with each value in its cell (the cell text doubles as the table view)."""
    rows = agreement[agreement.population.eq("primary") & agreement.category.eq("overall") & agreement.kind.eq("spearman")]
    value = {frozenset((r.a, r.b)): r.point for r in rows.itertuples()}
    methods = [m for m in METHODS if m in set(rows.a) | set(rows.b)]
    k = len(methods)
    matrix = np.full((k, k), np.nan)
    for i, a in enumerate(methods):
        for j, b in enumerate(methods[: i + 1]):
            matrix[i, j] = 1.0 if a == b else value[frozenset((a, b))]
    low = 0.6
    cmap = LinearSegmentedColormap.from_list("blue", RAMP)
    cmap.set_bad(SURFACE)
    fig, ax = plt.subplots(figsize=(7.4, 6.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    image = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=low, vmax=1.0)
    for i in range(k):
        for j in range(i + 1):
            v = matrix[i, j]
            dark = (v - low) / (1 - low) > 0.5
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color=SURFACE if dark else INK)
    for x in np.arange(0.5, k - 1, 1.0):   # the 2px surface gap between cells
        ax.axhline(x, color=SURFACE, lw=2)
        ax.axvline(x, color=SURFACE, lw=2)
    ax.set_xticks(range(k), [METHOD_LABEL[m] for m in methods], rotation=35, ha="right", fontsize=9, color=INK)
    ax.set_yticks(range(k), [METHOD_LABEL[m] for m in methods], fontsize=9, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=INK2, labelsize=8, length=0)
    shots = f"{int(rows.n.iloc[0]):,d}".replace(",", " ")
    ax.set_title(f"Zgodność metod: korelacja Spearmana prognoz, populacja główna ({shots} strzałów)", fontsize=10,
                 loc="left", color=INK, pad=10)
    fig.text(0.01, -0.06, f"Skala barw od {low:g}; przekątna = 1. {pl.NOTICE}", fontsize=8, color=INK2)
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=FIGURES)
    out = parser.parse_args().out
    inputs = {str(p.relative_to(ROOT)): verified(p) for p in (FAMILY_FILE, AGREEMENT_FILE)}
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID})
    forest(pd.read_csv(FAMILY_FILE, dtype={"item": str}), out / "family_forest.png")
    heatmap(pd.read_csv(AGREEMENT_FILE), out / "agreement_heatmap.png")
    record = {"notice": pl.NOTICE, "inputs": inputs, "code": xr.code_hashes(Path(__file__)), "git": records.git_state(),
              "files": {p.name: sha256_file(p) for p in sorted(out.glob("*.png"))}}
    (out / "figures.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record["files"], indent=2))


if __name__ == "__main__":
    main()
