"""Draw the two figures of the xg-vendor paper draft from the confirmatory result files (xg-vendor 17).

    ./run.sh python scripts/vendor_paper_figures.py [--out results_reports/xg_vendor_2026-10/paper/figures]

- `fig_ladder.pdf`: H1 and H2. Left, the recalibrated log-loss of every rung with its 95% interval, primary and
  combined population. Right, the H2 contrast and frame + vendor minus vendor with 95% and 90% intervals and the
  pre-registered margin.
- `fig_probes.pdf`: H3. Every probe target for G-combo fold 0, fold 1 and frozen DINOv2 with 95% intervals and the
  pre-registered threshold; R2 and AUC targets in separate panels.

Every plotted value is read from `results/xgv_confirmatory` (relative to `--root`); nothing is typed in. The output is
vector PDF without a creation date, so the same inputs give the same bytes and `check_vendor_paper.py` can redraw
the figures and compare.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

EXP = Path(__file__).resolve().parents[1]
CONFIRMATORY = Path("results/xgv_confirmatory")
OUT = Path("results_reports/xg_vendor_2026-10/paper/figures")
FIGURES = ("fig_ladder.pdf", "fig_probes.pdf")

# Reference palette of the data-viz method: the first three categorical slots (pass all-pairs CVD checks) and ink.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

RUNGS = [
    ("constant", "constant"),
    ("dist_angle", "distance + angle"),
    ("event", "events"),
    ("freeze_frame", "positions (freeze frame)"),
    ("vendor", "vendor xG"),
    ("frame", "frame (G-combo)"),
    ("frame+positions", "frame + positions"),
    ("frame+vendor", "frame + vendor"),
]
H2_ROWS = [
    ("primary", "h2", "frame + positions $-$ positions\nprimary (confirmatory)"),
    ("combined", "h2", "frame + positions $-$ positions\ncombined (robustness)"),
    ("primary", "frame_plus_vendor", "frame + vendor $-$ vendor\nprimary (descriptive)"),
    ("combined", "frame_plus_vendor", "frame + vendor $-$ vendor\ncombined (descriptive)"),
]
TARGETS = {
    "dist": "distance to goal *",
    "angle": "goal angle",
    "ff_def_cone": "defenders in cone",
    "gk_depth": "keeper depth",
    "gk_dist_goal": "keeper to goal centre",
    "gk_off_line_abs": "keeper lateral offset",
    "ff_opp_3m": "opponents near shooter",
    "def_cone_ge1": "$\\geq$1 defender in cone *",
    "def_cone_ge2": "$\\geq$2 defenders in cone",
    "head": "header",
}
ENCODERS = [
    ("emb_g_fold0", "G-combo fold 0", BLUE, "o", True),
    ("emb_g_fold1", "G-combo fold 1", AQUA, "s", True),
    ("emb_dinov2", "frozen DINOv2", ORANGE, "D", False),
]
STYLE = {
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.edgecolor": INK_2,
    "axes.labelcolor": INK,
    "xtick.color": INK_2,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
}


def ladder_data(root: Path) -> pd.DataFrame:
    """Recalibrated log-loss with 95% interval for every rung, both populations, in ladder order."""
    m = pd.read_csv(root / CONFIRMATORY / "ladder/metrics.csv")
    m = m[(m.variant == "recalibrated") & m.population.isin(["primary", "combined"])]
    order = {rung: i for i, (rung, _) in enumerate(RUNGS)}
    m = m[m.rung.isin(order)].assign(order=lambda d: d.rung.map(order))
    if len(m) != 2 * len(RUNGS):
        raise ValueError(f"metrics.csv: expected {2 * len(RUNGS)} recalibrated rung rows, found {len(m)}")
    return m.sort_values(["population", "order"])[["population", "rung", "order", "log_loss", "log_loss_lo", "log_loss_hi"]]


def h2_data(root: Path) -> tuple[list[dict], float]:
    """The four S5 rows (difference, 95% and 90% intervals) and the margin δ, from h2.json."""
    sets = json.loads((root / CONFIRMATORY / "ladder/h2.json").read_text())["sets"]
    rows = [sets[population][key] for population, key, _ in H2_ROWS]
    margins = {r["margin"] for r in rows}
    if len(margins) != 1:
        raise ValueError(f"h2.json: rows disagree on the margin: {sorted(margins)}")
    return rows, margins.pop()


def probe_data(root: Path) -> pd.DataFrame:
    """Every probe target × encoder on all shots, with value, 95% interval, metric and threshold."""
    p = pd.read_csv(root / CONFIRMATORY / "probes/probes.csv")
    p = p[(p.subset == "all") & p.target.isin(TARGETS) & p.encoder.isin([e for e, *_ in ENCODERS])]
    expected = {(e, t) for e, *_ in ENCODERS for t in TARGETS}
    found = list(zip(p.encoder, p.target, strict=True))
    if len(found) != len(expected) or set(found) != expected:
        missing = sorted(expected - set(found))
        raise ValueError(f"probes.csv: needs one row per encoder and target; missing {missing}, {len(found)} rows found")
    if (p.groupby("target").metric.nunique() != 1).any():
        raise ValueError("probes.csv: a target has different metrics across encoders")
    return p[["encoder", "target", "metric", "value", "lo", "hi", "threshold"]]


def draw_ladder(root: Path, path: Path) -> None:
    data = ladder_data(root)
    rows, delta = h2_data(root)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.9, 2.35), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.85})
    n = len(RUNGS)
    for population, colour, marker, shift, filled in (("primary", BLUE, "o", -0.14, True),
                                                      ("combined", ORANGE, "D", 0.14, False)):
        d = data[data.population == population]
        y = n - 1 - d.order.to_numpy() + shift
        ax.errorbar(d.log_loss, y, xerr=[d.log_loss - d.log_loss_lo, d.log_loss_hi - d.log_loss], fmt=marker,
                    ms=4, color=colour, mfc=colour if filled else "white", mew=1.2, elinewidth=1.2, capsize=0,
                    label=f"{population} population")
    ax.axhspan(-0.5, 2.5, color=GRID, alpha=0.35, lw=0, zorder=0)
    ax.set_yticks(range(n), [label for _, label in reversed(RUNGS)])
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel("log-loss, recalibrated (lower is better)")
    ax.set_title("(a) H1: information ladder", loc="left")
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.2), ncol=2, frameon=False, handletextpad=0.3)

    m = len(rows)
    y = [m - 1 - i for i in range(m)]
    bx.axvspan(-delta, delta, color=GRID, alpha=0.45, lw=0, zorder=0)
    bx.axvline(0, color=INK_2, lw=0.6)
    for yi, r in zip(y, rows, strict=True):
        bx.plot([r["lo95"], r["hi95"]], [yi, yi], color=BLUE, lw=1.0, solid_capstyle="round")
        bx.plot([r["lo90"], r["hi90"]], [yi, yi], color=BLUE, lw=3.2, solid_capstyle="round")
        bx.plot(r["diff"], yi, "o", ms=4.5, color=BLUE, mec="white", mew=0.8)
    bx.set_yticks(y, [label for *_, label in H2_ROWS])
    bx.set_ylim(-0.6, m - 0.4)
    bx.set_xlim(-1.4 * delta, 1.4 * delta)
    bx.set_xlabel(r"$\Delta$ log-loss (negative: the frame helps)")
    bx.set_title(r"(b) H2: what the frame adds (band: $\pm\delta$)", loc="left")
    bx.grid(axis="x", color=GRID, lw=0.5)
    save(fig, path)


def draw_probes(root: Path, path: Path) -> None:
    data = probe_data(root)
    metric_of = data.groupby("target").metric.first()
    panels = [(metric, [t for t in TARGETS if metric_of[t] == metric]) for metric in ("R2", "AUC")]
    heights = [len(targets) for _, targets in panels]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.45), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.75})
    for ax, (metric, targets), size in zip(axes, panels, heights, strict=True):
        for k, (encoder, label, colour, marker, filled) in enumerate(ENCODERS):
            d = data[data.encoder == encoder].set_index("target").loc[targets]
            y = [size - 1 - i + 0.22 * (1 - k) for i in range(size)]
            ax.errorbar(d.value, y, xerr=[d.value - d.lo, d.hi - d.value], fmt=marker, ms=3.5, color=colour,
                        mfc=colour if filled else "white", mew=1.0, elinewidth=1.0, capsize=0, label=label)
        thresholds = data[data.encoder == ENCODERS[0][0]].set_index("target").loc[targets].threshold
        for i, t in enumerate(thresholds):
            yi = size - 1 - i
            ax.plot([t, t], [yi - 0.38, yi + 0.38], color=INK, lw=1.4, solid_capstyle="butt")
        ax.set_yticks(range(size), [TARGETS[t] for t in reversed(targets)])
        ax.set_ylim(-0.6, size - 0.4)
        ax.set_xlim(left=min(ax.get_xlim()[0], min(thresholds) - 0.02))
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_xlabel(f"probe {'$R^2$' if metric == 'R2' else 'AUC'} (out-of-fold)")
    axes[0].set_title("(a) continuous targets, $R^2$", loc="left")
    axes[1].set_title("(b) binary targets, AUC", loc="left")
    axes[0].legend(loc="upper left", bbox_to_anchor=(-0.02, -0.2), ncol=3, frameon=False, handletextpad=0.3,
                   columnspacing=1.0)
    save(fig, path)


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.02, metadata={"CreationDate": None})
    plt.close(fig)


def build(out: Path, root: Path = EXP) -> list[Path]:
    """Draw both figures into `out` from the result files under `root`; returns their paths."""
    with plt.rc_context(STYLE):
        draw_ladder(root, out / FIGURES[0])
        draw_probes(root, out / FIGURES[1])
    return [out / name for name in FIGURES]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=EXP / OUT)
    ap.add_argument("--root", type=Path, default=EXP, help="base of the result paths")
    args = ap.parse_args(argv)
    for path in build(args.out, args.root):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
