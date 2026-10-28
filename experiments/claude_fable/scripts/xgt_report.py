"""Report readings of xg-temporal (ticket 05; CPU, seconds, no label read): the planted-effect recovery and power of the
three label-free control attempts of ticket 03, a cautious sample-size extrapolation, and two small figures. The phase
closed after the third positive-control miss (owner decision 2026-10-28), so nothing here touches a label.

    CUDA_VISIBLE_DEVICES= ./run.sh --no-sync python scripts/xgt_report.py
        # -> results/xgt_report/{readings.json, complete.json}
        #    results_reports/xg_temporal/figures/{recovery_power.png, null_rejections.png, figures.json}

Inputs, each checked against its folder's completion record:
- `results/xgt/controls/{label_free, label_free_remedy, label_free_c10}/controls.json`: the first attempt (T = S + 96
  motion columns, one shared C), the C9 Q2 remedy (separate motion-block C) and the C10 design (8 vs_start columns);
- `results/xgd_a1_ensemble/oof.csv`, only the columns `shot_id` and `z_g_combo` (the model's out-of-fold logit, not a
  label): the expected goal count sum(sigmoid(z_g_combo)) of the main population P_G;
- `results/xgt/controls/label_free_c10/population.csv`: the shot list of P_G;
- `results/xgt/controls/label_free_c10/complete.json`: the C10 prediction written before the run, copied as is.

Recovery share = mean recovered T-S ΔLL / oracle T-S ΔLL of the plant, per attempt and planted effect.

Sample-size extrapolation (descriptive, an extrapolation, not a result). For a plant with mean recovered T-S m, its SD
s over the 200 replicates and its Holm power p at N = 7,010 shots: z = |m| / s, c = z - Phi^-1(p) (the test's own
relation of power and z at that point), and the shots needed for 80% power N80 = N ((c + Phi^-1(0.8)) / z)^2. It
assumes the recovery share stays as at N and s shrinks as 1 / sqrt(N). With more data the shrinkage cost falls, so the
recovery share should rise and N80 is then an upper figure for the in-block plant; for the diffuse plant it says only
that the linear head cannot reach it on any realistic sample.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from report_complexity import GRID, INK, INK2, MUTED, SURFACE, _axes_style
from scipy.stats import norm
from xgpoc.bundle import sha256_file
from xgpoc.detector import records
from xgpoc.detector.guard import no_statsbomb_reads
from xgpoc.temporal import records as temporal_records

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONTROLS = ROOT / "results/xgt/controls"
ATTEMPTS = {"first": CONTROLS / "label_free/controls.json",
            "remedy": CONTROLS / "label_free_remedy/controls.json",
            "c10": CONTROLS / "label_free_c10/controls.json"}
LABELS = {"first": "1: 96 kolumn, wspólne C", "remedy": "2: 96 kolumn, osobne C (C9 Q2)",
          "c10": "3: 8 kolumn vs_start (C10)"}
MARKERS = {"first": "s", "remedy": "^", "c10": "o"}
# Categorical slots 1-3 of the dataviz reference palette, as in scripts/xgs_report.py.
COLOUR = {"first": "#eb6834", "remedy": "#1baf7a", "c10": "#2a78d6"}
C10_RECORD = CONTROLS / "label_free_c10/complete.json"   # the C10 prediction, written in ticket 03 before the run
OOF = ROOT / "results/xgd_a1_ensemble/oof.csv"
POPULATION = CONTROLS / "label_free_c10/population.csv"
OUT = ROOT / "results/xgt_report"
FIGURES = ROOT / "results_reports/xg_temporal/figures"
TARGETS = (-0.002, -0.003, -0.004)
KEY = {-0.002: "m2", -0.003: "m3", -0.004: "m4"}          # JSON keys without dots, so the number check can address them
POWER = 0.8
GATE = (-0.003, -0.001)
NULL_LIMIT = 0.09


def plant(controls: dict, target: float) -> dict:
    """The positive-control record of `controls` at planted oracle T-S `target` (the gate at -0.002, reports else)."""
    if target == -0.002:
        return controls["gates"]["positive"]
    return controls["reports"]["positive_-0.004" if target == -0.004 else "curve_-0.003"]


def row(record: dict) -> dict:
    oracle, mean = record["oracle_T-S"], record["mean_recovered_delta_T-S"]
    return {"oracle_T-S": oracle, "mean_recovered_T-S": mean, "sd_recovered_T-S": record["sd_recovered_delta_T-S"],
            "recovery_share": mean / oracle, "power_T-S": record["power_T-S"],
            "power_T-S_mc_se": record["power_T-S_mc_se"], "oracle_power": record["oracle_power"],
            "mean_recovered_T-R": record["mean_recovered_delta_T-R"], "power_T-R": record["power_T-R"]}


def extrapolate(r: dict, shots: int, matches: int) -> dict:
    """N80 of the module docstring for one plant row."""
    z = abs(r["mean_recovered_T-S"]) / r["sd_recovered_T-S"]
    c = z - norm.ppf(r["power_T-S"])
    factor = ((c + norm.ppf(POWER)) / z) ** 2
    return {"z": z, "c": c, "factor": factor, "shots_80": shots * factor, "matches_80": matches * factor}


def recovery_power(rows: dict, out: Path) -> None:
    """Left: mean recovered T-S against the planted oracle T-S, with the identity and the gate. Right: Holm power."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 3.9), facecolor=SURFACE)
    x = -np.array(TARGETS) * 1000
    for ax in (left, right):
        _axes_style(ax)
        ax.set_xticks(x, [f"−{v:.0f}" for v in x])
        ax.set_xlabel("zasiany efekt: wyrocznia T−S (×10⁻³ LL)", fontsize=8.5, color=INK2)
        ax.tick_params(labelsize=8, colors=INK2)
    left.plot(x, x, color=MUTED, lw=1, ls="--", zorder=1)
    left.text(x[-1], x[-1] - 0.25, "pełne odzyskanie", fontsize=7.5, color=MUTED, ha="right", va="top")
    left.fill_between([x[0] - 0.12, x[0] + 0.12], -GATE[1] * 1000, -GATE[0] * 1000, color=GRID, zorder=0)
    left.text(x[0] + 0.16, -GATE[0] * 1000, "bramka kontroli\n[−3; −1] przy −2", fontsize=7.5, color=INK2, va="top")
    right.axhline(POWER, color=INK2, lw=1, ls=":", zorder=1)
    right.text(x[0], POWER + 0.02, "moc 0.8", fontsize=7.5, color=INK2, va="bottom")
    for name, attempt in rows.items():
        rec = [-attempt[t]["mean_recovered_T-S"] * 1000 for t in TARGETS]
        pw = [attempt[t]["power_T-S"] for t in TARGETS]
        style = {"color": COLOUR[name], "marker": MARKERS[name], "lw": 2, "markersize": 7,
                 "markeredgecolor": SURFACE, "markeredgewidth": 1.5}
        left.plot(x, rec, label=LABELS[name], **style)
        right.plot(x, pw, **style)
        va = {"first": "bottom", "remedy": "top"}.get(name, "center")
        left.text(x[-1] + 0.08, rec[-1], f"{rec[-1]:.2f}", fontsize=7.5, color=INK2, va=va)
        right.text(x[-1] + 0.08, pw[-1], f"{pw[-1]:.2f}", fontsize=7.5, color=INK2, va="center")
    oracle = [rows["c10"][t]["oracle_power"] for t in TARGETS]
    right.plot(x, oracle, color=MUTED, lw=1, ls="--", marker="D", markersize=5, label="wyrocznia (próba 3)")
    left.set_ylabel("średnie odzyskane −(T−S) (×10⁻³ LL)", fontsize=8.5, color=INK2)
    right.set_ylabel("moc Holma dla T−S", fontsize=8.5, color=INK2)
    left.set_ylim(0, 4.3)
    right.set_ylim(0, 1.05)
    left.set_xlim(x[0] - 0.2, x[-1] + 0.5)
    right.set_xlim(x[0] - 0.2, x[-1] + 0.5)
    left.set_title("Odzyskany efekt ruchu", fontsize=10, loc="left", color=INK)
    right.set_title("Moc wykrycia", fontsize=10, loc="left", color=INK)
    handles, names = left.get_legend_handles_labels()
    handles.append(right.get_lines()[-1])
    names.append("wyrocznia, moc (próba 3)")
    fig.legend(handles, names, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.02), ncol=2,
               labelcolor=INK2)
    fig.text(0.01, -0.13, "200 replik na punkt, wyniki syntetyczne bez etykiet; 7 010 strzałów, 328 meczów.",
             fontsize=7.5, color=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def null_rejections(nulls: dict, out: Path) -> None:
    """Holm rejection rate under the plain nulls (either contrast) and the rich nulls (T-S), with the 0.09 limit."""
    fig, ax = plt.subplots(figsize=(6.4, 3.0), facecolor=SURFACE)
    _axes_style(ax)
    ax.grid(axis="x", visible=False)
    groups = ("plain", "rich")
    width = 0.26
    for i, name in enumerate(nulls):
        xs = np.arange(len(groups)) + (i - 1) * (width + 0.02)
        vals = [nulls[name][g]["rate"] for g in groups]
        ax.bar(xs, vals, width, color=COLOUR[name], label=LABELS[name], edgecolor=SURFACE, linewidth=2)
        for xv, v in zip(xs, vals, strict=True):
            ax.text(xv, v + 0.004, f"{v:.3f}".rstrip("0").rstrip(".") if v else "0", fontsize=7.5, color=INK2,
                    ha="center", va="bottom")
    ax.axhline(NULL_LIMIT, color=INK2, lw=1, ls=":")
    ax.text(1.45, NULL_LIMIT + 0.003, "limit 0.09", fontsize=7.5, color=INK2, ha="right", va="bottom")
    ax.set_xticks(range(len(groups)), ["zera proste: Holm, T−R lub T−S", "zera bogate: Holm, T−S"], fontsize=8.5,
                  color=INK)
    ax.set_ylabel("udział odrzuceń (200 replik)", fontsize=8.5, color=INK2)
    ax.tick_params(labelsize=8, colors=INK2)
    ax.set_ylim(0, 0.14)
    ax.set_title("Kontrole zerowe: odsetek fałszywych odrzuceń", fontsize=10, loc="left", color=INK)
    fig.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.0), ncol=2, labelcolor=INK2)
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--figures", type=Path, default=FIGURES)
    args = parser.parse_args()
    with no_statsbomb_reads():
        inputs = {str(p.relative_to(ROOT)): records.verified_file(p) for p in (*ATTEMPTS.values(), OOF, POPULATION)}
        inputs[str(C10_RECORD.relative_to(ROOT))] = sha256_file(C10_RECORD)
        prediction = json.loads(C10_RECORD.read_text())["settings"]["c10"]["prediction"]
        controls = {name: json.loads(path.read_text()) for name, path in ATTEMPTS.items()}
        population = pd.read_csv(POPULATION, usecols=["shot_id", "game"])
        oof = pd.read_csv(OOF, usecols=["shot_id", "z_g_combo"]).set_index("shot_id")
        if not population["shot_id"].isin(oof.index).all():
            raise SystemExit("a shot of P_G has no z_g_combo")
        z = oof.loc[population["shot_id"], "z_g_combo"].to_numpy()
        shots, matches = len(population), int(population["game"].nunique())
        counts = controls["c10"]["window"], controls["c10"]["k"]
        rows = {name: {t: row(plant(c, t)) for t in TARGETS} for name, c in controls.items()}
        diffuse = row(controls["c10"]["reports"]["positive_c9_diffuse_-0.002"])
        pc01 = {name: row(c["reports"]["positive_pc01_-0.002"]) for name, c in controls.items()}
        nulls = {name: {"plain": c["gates"]["plain_nulls"]["holm_rejection_any"],
                        "rich": c["gates"]["rich_nulls"]["holm_rejection"]["T-S"]} for name, c in controls.items()}
        readings = {
            "label": "label-free readings of the xg-temporal controls (ticket 05); no outcome read",
            "population": {"shots": shots, "matches": matches, "shots_per_match": shots / matches,
                           "expected_goals": float((1 / (1 + np.exp(-z))).sum()),
                           "expected_goal_rate": float((1 / (1 + np.exp(-z))).mean()),
                           "window": counts[0], "k": counts[1]},
            "attempts": {name: {KEY[t]: r for t, r in attempt.items()} for name, attempt in rows.items()},
            "recovery_share_range_m2": {"min": min(a[-0.002]["recovery_share"] for a in rows.values()),
                                               "max": max(a[-0.002]["recovery_share"] for a in rows.values())},
            "max_power_T-S": max(a[t]["power_T-S"] for a in rows.values() for t in TARGETS),
            "c9_diffuse_c10": diffuse,
            "pc01_m2": pc01,
            "nulls": nulls,
            "prediction_c10": {"in_block_m2": prediction["positive_-0.002_mean_recovered_T-S"],
                               "c9_diffuse_m2": prediction["positive_c9_diffuse_-0.002_mean_recovered_T-S"],
                               "written": prediction["written"]},
            "extrapolation": {
                "rule": "N80 = N ((c + Phi^-1(0.8)) / z)^2, z = |mean| / sd over replicates, c = z - Phi^-1(power); "
                        "recovery share held as at N, SD as 1 / sqrt(N); an extrapolation, not a result",
                "c10": {KEY[t]: extrapolate(rows["c10"][t], shots, matches) for t in TARGETS},
                "c10_c9_diffuse": extrapolate(diffuse, shots, matches),
                "remedy_m4": extrapolate(rows["remedy"][-0.004], shots, matches),
            },
        }
        code, git = temporal_records.code_hashes(Path(__file__)), records.git_state()
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "readings.json").write_text(json.dumps(readings, indent=2) + "\n")
        record = {"inputs": inputs, "code": code, "git": git, "finished": datetime.now(UTC).isoformat(timespec="seconds"),
                  "files": records.output_hashes(args.out)}
        (args.out / "complete.json").write_text(json.dumps(record, indent=2) + "\n")

        args.figures.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID})
        recovery_power(rows, args.figures / "recovery_power.png")
        null_rejections(nulls, args.figures / "null_rejections.png")
        figures = {"inputs": inputs, "code": code, "git": git,
                   "files": {p.name: sha256_file(p) for p in sorted(args.figures.glob("*.png"))}}
        (args.figures / "figures.json").write_text(json.dumps(figures, indent=2, sort_keys=True) + "\n")
    print(json.dumps(readings, indent=2))


if __name__ == "__main__":
    main()
