"""Charts for the xG summary, all on the 2,721 StatsBomb-paired shots (Data: StatsBomb).

Run: uv run python reports/research_summaries/figures_xg.py
Writes SVGs to outputs/research_summaries/xg/figures/ and the numbers to numbers.json.
"""

import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sources import OUT, XG_DINO_MASK, XG_DINO_MASK_SHA256, paired_shots

FIG = OUT / "xg/figures"
# Model identity colours, fixed across every chart (reference palette slots 1 and 3;
# the vendor benchmark is neutral ink, not a series hue).
MODELS = {
    "p_video": ("Video only", "#2a78d6"),
    "p_geometry": ("Video + geometry", "#1baf7a"),
    "p_vendor": ("StatsBomb xG", "#52514e"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BOOT = 2000

plt.rcParams.update(
    {
        "font.family": ["Noto Sans", "DejaVu Sans"],
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
    }
)


def log_loss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def match_bootstrap(df: pd.DataFrame, columns: list[str], seed: int = 0) -> dict:
    """Mean per-shot log-loss difference vs video only, resampling matches."""
    rng = np.random.default_rng(seed)
    y = df.goal.to_numpy(float)
    base = log_loss(df.p_video.to_numpy(), y)
    groups = df.groupby("match").indices
    keys = list(groups)
    out = {}
    for c in columns:
        diff = log_loss(df[c].to_numpy(), y) - base
        sums = np.array([diff[groups[k]].sum() for k in keys])
        counts = np.array([len(groups[k]) for k in keys])
        draws = rng.integers(0, len(keys), size=(BOOT, len(keys)))
        stats = sums[draws].sum(1) / counts[draws].sum(1)
        out[c] = (float(diff.mean()), *np.percentile(stats, [2.5, 97.5]).tolist())
    return out


def gap_chart(df: pd.DataFrame, numbers: dict) -> None:
    deltas = match_bootstrap(df, ["p_detector", "p_geometry", "p_vendor"])
    vendor = deltas["p_vendor"][0]
    rows = [
        ("p_detector", "Video + detector\n(players, GK)", "#86b6ef"),
        (
            "p_geometry",
            "Video + geometry\n(+ shooter-feet location)",
            MODELS["p_geometry"][1],
        ),
        ("p_vendor", "StatsBomb xG\n(manual event data)", MODELS["p_vendor"][1]),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 2.1))
    for i, (c, _label, color) in enumerate(rows):
        m, lo, hi = deltas[c]
        ax.barh(i, -m * 1000, color=color, height=0.55)
        ax.plot([-lo * 1000, -hi * 1000], [i, i], color=INK, lw=1.2)
        share = m / vendor
        text = f"{-m * 1000:.1f}" + (
            "" if c == "p_vendor" else f"  ({share:.0%} of gap)"
        )
        ax.text(-lo * 1000 + 0.6, i, text, va="center", color=INK, fontsize=8.5)
        numbers[c] = {"delta": m, "lo": lo, "hi": hi, "gap_share": share}
    ax.set_yticks(range(len(rows)), [r[1] for r in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 40)
    ax.set_xlabel(
        "Log-loss reduction vs video-only model (×10⁻³), 95% match-bootstrap CI"
    )
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "gap.svg")
    plt.close(fig)


def recalibrated(df: pd.DataFrame, folds: int = 5) -> pd.DataFrame:
    """Every model Platt-recalibrated alike, cross-fitted over match folds (the reports' main scale)."""
    from sklearn.linear_model import LogisticRegression

    out = df.copy()
    matches = df.match.unique()
    fold = dict(
        zip(
            matches,
            np.random.default_rng(0).permutation(len(matches)) % folds,
            strict=True,
        )
    )
    f = df.match.map(fold).to_numpy()
    for c in MODELS:
        z = np.log(
            np.clip(df[c], 1e-6, 1 - 1e-6) / np.clip(1 - df[c], 1e-6, 1)
        ).to_numpy()[:, None]
        p = np.empty(len(df))
        for k in range(folds):
            model = LogisticRegression(C=1e6).fit(z[f != k], df.goal[f != k])
            p[f == k] = model.predict_proba(z[f == k])[:, 1]
        out[c] = p
    return out


def reliability_chart(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(3.3, 3.1))
    ax.plot([0, 0.6], [0, 0.6], color=GRID, lw=1.2, zorder=0)
    y = df.goal.to_numpy(float)
    for c, (label, color) in MODELS.items():
        p = df[c].to_numpy()
        bins = np.quantile(p, np.linspace(0, 1, 11))
        idx = np.clip(np.digitize(p, bins[1:-1]), 0, 9)
        xs = [p[idx == b].mean() for b in range(10)]
        ys = [y[idx == b].mean() for b in range(10)]
        ax.plot(
            xs,
            ys,
            color=color,
            lw=2,
            marker="o",
            ms=4.5,
            label=label,
            mec="white",
            mew=0.8,
        )
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 0.6)
    ax.set_xlabel("Predicted xG (decile mean)")
    ax.set_ylabel("Observed goal rate")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "reliability.svg")
    plt.close(fig)


CATEGORIES = [
    ("Angle < 15°", lambda d: d.cat_angle == "00_15"),
    ("Outside the box", lambda d: d.cat_outside_box == "yes"),
    ("Header", lambda d: d.cat_body == "Head"),
    ("GK ≥ 4 yd off line", lambda d: d.cat_gk_depth == "04_plus"),
    ("Under pressure", lambda d: d.cat_under_pressure == "yes"),
]


def category_chart(df: pd.DataFrame, numbers: dict) -> None:
    """Expected ÷ actual goals per situation on a log axis: an arrow from video only to video + geometry,
    StatsBomb as a reference tick, and the range that chance alone would give for that many goals."""
    fig, ax = plt.subplots(figsize=(6.4, 2.35))
    numbers["categories"] = {}
    video, geometry = MODELS["p_video"][1], MODELS["p_geometry"][1]
    for i, (name, select) in enumerate(CATEGORIES):
        sub = df[select(df)]
        goals = sub.goal.sum()
        ratio = {c: float(sub[c].sum() / goals) for c in MODELS}
        # A calibrated model's E/A has a log-sd of about 1/sqrt(goals) (Poisson goals, clustering ignored).
        half = 1.96 / np.sqrt(goals)
        numbers["categories"][name] = {
            "shots": len(sub),
            "goals": int(goals),
            **ratio,
            "chance_range": [float(np.exp(-half)), float(np.exp(half))],
        }
        ax.add_patch(
            plt.Rectangle(
                (np.exp(-half), i - 0.32),
                np.exp(half) - np.exp(-half),
                0.64,
                color="#ecebe7",
                lw=0,
                zorder=0,
            )
        )
        ax.annotate(
            "",
            xy=(ratio["p_geometry"], i),
            xytext=(ratio["p_video"], i),
            arrowprops={
                "arrowstyle": "-|>,head_width=0.25,head_length=0.5",
                "color": geometry,
                "lw": 1.6,
                "shrinkA": 4,
                "shrinkB": 0,
            },
            zorder=2,
        )
        ax.scatter(
            ratio["p_video"], i, color=video, s=36, zorder=3, edgecolor="white", lw=0.8
        )
        ax.plot(
            [ratio["p_vendor"]] * 2,
            [i - 0.24, i + 0.24],
            color=INK,
            lw=2,
            zorder=3,
            solid_capstyle="butt",
        )
        ax.text(
            3.05,
            i,
            f"{len(sub):,} shots · {int(goals)} goals",
            va="center",
            fontsize=7.5,
            color=MUTED,
        )
        if i == 0:
            for c, dy in (("p_video", -0.42), ("p_geometry", -0.42)):
                ax.text(
                    ratio[c],
                    i + dy,
                    f"{100 * (ratio[c] - 1):+.0f}%",
                    ha="center",
                    fontsize=7.5,
                    color=INK,
                )
    ax.axvline(1, color=MUTED, lw=0.8, zorder=1)
    ax.set_xscale("log")
    ticks = [2 / 3, 0.8, 1, 1.25, 1.5, 2, 2.5]
    ax.set_xticks(ticks, ["−33%", "−20%", "0", "+25%", "+50%", "+100%", "+150%"])
    ax.minorticks_off()
    ax.set_xlim(0.6, 2.9)
    ax.set_ylim(len(CATEGORIES) - 0.5, -0.75)
    ax.set_yticks(range(len(CATEGORIES)), [c[0] for c in CATEGORIES], fontsize=8.5)
    ax.set_xlabel("Expected vs actual goals (log scale; right of 0 = over-predicts)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [
        plt.Line2D(
            [], [], color=video, marker="o", lw=0, markersize=6, label="Video only"
        ),
        plt.Line2D(
            [],
            [],
            color=geometry,
            marker=">",
            lw=1.6,
            markersize=5,
            label="Video + geometry",
        ),
        plt.Line2D(
            [],
            [],
            color=INK,
            marker="|",
            lw=0,
            markersize=9,
            markeredgewidth=2,
            label="StatsBomb xG",
        ),
        plt.Rectangle((0, 0), 1, 1, color="#ecebe7", label="range expected by chance"),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=7.8,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        handlelength=1.6,
        columnspacing=1.4,
    )
    fig.tight_layout()
    fig.savefig(FIG / "categories.svg")
    plt.close(fig)


# xg-dino-info, refit estimand, mean fill, all shots: share of the C−B gain left after masking
MASKS = [
    ("Shooter", "shooter"),
    ("Ball", "ball"),
    ("Goalkeeper", "goalkeeper"),
    ("Goal", "goal"),
    ("Background", "background"),
    ("All players", "players"),
    ("Shooter + ball only", "keep_shooter_ball"),
]


def masking_chart(numbers: dict) -> None:
    """How much of the image's gain over geometry survives masking one region of the frame."""
    raw = XG_DINO_MASK.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == XG_DINO_MASK_SHA256, (
        "gain_left.csv changed"
    )
    gl = pd.read_csv(XG_DINO_MASK)
    gl = gl[(gl.estimand == "refit") & (gl["fill"] == "mean")].set_index("region")
    fig, ax = plt.subplots(figsize=(3.3, 1.9))
    numbers["masking"] = {}
    for i, (_label, region) in enumerate(MASKS):
        r = gl.loc[region]
        numbers["masking"][region] = [r.all_point, r.all_lo95, r.all_hi95]
        color = MODELS["p_video"][1] if i < 5 else INK
        ax.plot([r.all_lo95, r.all_hi95], [i, i], color=color, lw=2)
        ax.scatter(
            r.all_point, i, color=color, s=30, zorder=3, edgecolor="white", lw=0.8
        )
        ax.text(1.12, i, f"{r.all_point:.2f}", va="center", fontsize=7.5, color=INK)
    ax.axhline(4.5, color=GRID, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.axvline(1, color=MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.set_yticks(range(len(MASKS)), [m[0] for m in MASKS], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(-0.1, 1.22)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xlabel("Share of the image's gain that remains", fontsize=8)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "masking.svg")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    df = paired_shots()
    y = df.goal.to_numpy(float)
    numbers = {
        "shots": len(df),
        "goals": int(y.sum()),
        "matches": int(df.match.nunique()),
        "log_loss": {
            c: float(log_loss(df[c].to_numpy(), y).mean())
            for c in [*MODELS, "p_detector"]
        },
    }
    gap_chart(df, numbers)
    calibrated = recalibrated(df)
    reliability_chart(calibrated)
    category_chart(calibrated, numbers)
    masking_chart(numbers)
    (OUT / "xg/numbers.json").write_text(json.dumps(numbers, indent=1))
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
