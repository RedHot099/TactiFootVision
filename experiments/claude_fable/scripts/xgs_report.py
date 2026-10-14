"""Report readings of xg-stack (ticket 03; CPU, seconds, no label read): the recomputed power on the 172 matches and
the arms' forest plot. Descriptive, added after the real run; nothing here is in the frozen protocol.

    ./run.sh --no-sync python scripts/xgs_report.py
        # -> results/xgs_stack/report/{power.json, complete.json}
        #    results_reports/xg_stack_2026-10/figures/{arms_forest.png, figures.json}

Inputs, each checked against its folder's completion record:
- `results/xgs_stack/real/summary.json`, the single real run (ticket 02): each arm's ΔLL against R, its 95% interval
  and its match-bootstrap SD (2,000 draws over the 328 training matches);
- `results/xgs_stack/controls/controls.json`, the final controls (C6): the planted-effect recovery and the smallest
  detectable effect, drawn on the figure;
- `results/xgd_test_power/power.json`: the size of the 172 held-out matches' non-penalty population (3,819 shots),
  counted there from label-free index columns. No shot or label of those matches is read here.

Power on the 172 matches (C5 Q6 asks for it beside any recommendation). As `scripts/xgd_test_power.py` does for A1,
the confirmatory arm's bootstrap SD is scaled to the test population by sqrt(N_dev / N_test) shots, and the test
rule's success reading (upper bound of the two-sided interval below 0) is read off the normal approximation of
`testrule.expected_power`, for one variant (95% interval) and for two variants under Bonferroni (97.5%). The true
ΔLL is set to the arm's observed point estimate and to the practical bar −0.002. The approximation ignores that a
test arm would be fitted on all 328 matches, and that the stacker's expected gain is below the planted effect
(control 4 recovered about 80% of it), so it is a planning figure only. The test rule's one-sided level (0.025, or
0.0125 under Bonferroni) is stricter than the development read-out's one-sided p < 0.05 with Holm.

Everything runs inside `no_statsbomb_reads()`. The outputs are deterministic in their inputs, so a rerun overwrites
them (only the completion records' time stamp and git state change).
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import matplotlib.ticker
import numpy as np
from report_complexity import GRID, INK, INK2, MUTED, SURFACE, _axes_style
from xgpoc.bundle import sha256_file
from xgpoc.detector import records, testrule
from xgpoc.detector.guard import no_statsbomb_reads
from xgpoc.stack import records as stack_records

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/xgs_stack/real/summary.json"
CONTROLS = ROOT / "results/xgs_stack/controls/controls.json"
TEST_POWER_FILE = ROOT / "results/xgd_test_power/power.json"
OUT = ROOT / "results/xgs_stack/report"
FIGURES = ROOT / "results_reports/xg_stack_2026-10/figures"

CONFIRMATORY = ("G", "GS")
PRACTICAL = -0.002                                        # C5's practical bar (the A3 trigger)
INTERVALS = {"one_variant": 0.95, "two_variants_bonferroni": 0.975}

# Categorical slots 1-3 of the dataviz reference palette, as in scripts/xgc_report_figures.py.
COLOUR = {"confirmatory": "#2a78d6", "descriptive": "#eb6834", "gbm": "#1baf7a"}
ROWS = (("G", "logistic", "confirmatory", "G: geometria E3"),
        ("GS", "logistic", "confirmatory", "GS: geometria E3 + flagi SoccerNet"),
        ("S", "logistic", "descriptive", "S: flagi SoccerNet"),
        ("Q", "logistic", "descriptive", "Q: G + jakość detekcji"),
        ("ALL", "logistic", "descriptive", "ALL: wszystkie wejścia E3"),
        ("G", "gbm", "gbm", "G, GBM (wrażliwość)"),
        ("GS", "gbm", "gbm", "GS, GBM (wrażliwość)"))
LEGEND = {"confirmatory": "potwierdzające (Holm)", "descriptive": "opisowe", "gbm": "GBM, wrażliwość"}


def minus(text: str) -> str:
    return text.replace("-", "−")


def power_on_test(deltas: dict, n_dev: int, n_test: int, rule: dict) -> dict:
    """Per confirmatory arm: the scaled SE and P(success) at the observed point and at the practical bar, for each
    interval of `INTERVALS` (module docstring)."""
    scale = float(np.sqrt(n_dev / n_test))
    out = {}
    for arm in CONFIRMATORY:
        se = deltas[arm]["boot_sd"] * scale
        at = {}
        for name, value in (("observed", deltas[arm]["delta_ll"]), ("practical", PRACTICAL)):
            at[name] = {"delta": value}
            for label, level in INTERVALS.items():
                level_rule = {**rule, "success": {**rule["success"], "interval": level}}
                at[name][label] = testrule.expected_power(value, se, level_rule)["p_success"]
        out[arm] = {"se_dev": deltas[arm]["boot_sd"], "se_test": se, "at": at}
    return {"scale": scale, "arms": out}


def arm_rows(summary: dict) -> list[dict]:
    """The figure's rows: each arm's ΔLL against R with its 95% interval, logistic arms then the GBM sensitivity."""
    rows = []
    for arm, model, kind, label in ROWS:
        delta = summary["deltas"][arm] if model == "logistic" else summary["gbm"][arm]["delta_vs_R"]
        rows.append({"arm": arm, "model": model, "kind": kind, "label": label, "delta_ll": delta["delta_ll"],
                     "lo95": delta["lo95"], "hi95": delta["hi95"]})
    return rows


def forest(rows: list[dict], smallest_detectable: float, power: float, draws: int, out: Path) -> None:
    """ΔLL of each arm against R with its 95% interval, the 0 line, the practical bar and the smallest detectable
    effect of the power curve."""
    rows = rows[::-1]
    fig, ax = plt.subplots(figsize=(8.6, 4.4), facecolor=SURFACE)
    _axes_style(ax)
    ax.grid(axis="y", visible=False)
    ax.axvline(0, color="#c3c2b7", lw=1, zorder=1)
    ax.axvline(PRACTICAL, color=INK2, lw=1, ls="--", zorder=1)
    ax.axvline(smallest_detectable, color=MUTED, lw=1, ls=":", zorder=1)
    for i, r in enumerate(rows):
        colour = COLOUR[r["kind"]]
        ax.hlines(i, r["lo95"], r["hi95"], color=colour, lw=2, zorder=2)
        ax.scatter(r["delta_ll"], i, s=40, color=colour, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.set_yticks(range(len(rows)), [r["label"] for r in rows], fontsize=9, color=INK)
    header = len(rows) + 0.2                              # a strip above the rows for the line labels
    ax.set_ylim(-0.7, header + 0.5)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: minus(f"{v:+.3f}") if v else "0"))
    ax.tick_params(axis="x", labelsize=8.5, colors=INK2)
    ax.text(PRACTICAL, header, minus(f" próg praktyczny {PRACTICAL:+.3f}"), fontsize=8, color=INK2, va="center")
    ax.text(smallest_detectable, header, minus(f"moc {power:g} przy {smallest_detectable:+.4f} "), fontsize=8, color=MUTED,
            va="center", ha="right")
    handles = [plt.Line2D([], [], color=COLOUR[k], marker="o", lw=2, markersize=6) for k in LEGEND]
    fig.legend(handles, LEGEND.values(), frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0.0, 0.0),
               ncol=3, labelcolor=INK2)
    ax.set_xlabel("ΔLL wobec R (ujemne = lepiej), log-loss na strzał", fontsize=9, color=INK2)
    shown = f"{draws:,d}".replace(",", " ")
    ax.set_title(f"Ramiona wobec R: 95% przedziały z {shown} losowań meczów treningowych", fontsize=10,
                 loc="left", color=INK, pad=10)
    fig.text(0.01, -0.12, "Wyniki rozwojowe, poza foldem. Bez danych StatsBomb i bez 172 meczów odłożonych.", fontsize=8,
             color=INK2)
    fig.savefig(out, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--figures", type=Path, default=FIGURES)
    args = parser.parse_args()
    with no_statsbomb_reads():
        inputs = {str(p.relative_to(ROOT)): records.verified_file(p) for p in (SUMMARY, CONTROLS, TEST_POWER_FILE)}
        summary, controls = json.loads(SUMMARY.read_text()), json.loads(CONTROLS.read_text())
        size = json.loads(TEST_POWER_FILE.read_text())["test_population"]
        n_dev = summary["scores"]["R"]["n"]
        rule = testrule.load_rule()
        if size["matches"] != rule["population"]["matches"] or size["shots"] != rule["population"]["expected_shots"]:
            raise SystemExit(f"test population {size} differs from the test rule's")
        result = {
            "rule": "bootstrap SD of the real run scaled by sqrt(N_dev / N_test) shots; P(success) = P(upper bound of "
                    "the two-sided interval < 0), normal approximation (testrule.expected_power); descriptive, added "
                    "after the run",
            "development": {"shots": n_dev, "draws": summary["deltas"]["G"]["draws"]},
            "test_population": size,
            **power_on_test(summary["deltas"], n_dev, size["shots"], rule),
        }
        code, git = stack_records.code_hashes(Path(__file__)), records.git_state()
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "power.json").write_text(json.dumps(result, indent=2) + "\n")
        record = {"inputs": inputs, "code": code, "git": git, "finished": datetime.now(UTC).isoformat(timespec="seconds"),
                  "files": records.output_hashes(args.out)}
        (args.out / "complete.json").write_text(json.dumps(record, indent=2) + "\n")

        args.figures.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": GRID})
        curve = controls["reports"]["power_curve"]
        forest(arm_rows(summary), curve["smallest_detectable"], curve["power"], summary["deltas"]["G"]["draws"],
               args.figures / "arms_forest.png")
        figures = {"inputs": inputs, "code": code, "git": git,
                   "files": {p.name: sha256_file(p) for p in sorted(args.figures.glob("*.png"))}}
        (args.figures / "figures.json").write_text(json.dumps(figures, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
