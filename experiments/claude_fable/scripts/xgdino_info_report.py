"""Report readings of xg-dino-info (ticket 16; CPU, seconds, no label and no StatsBomb read): the few ratios the Polish
report `results_reports/xg_dino_2026-10/info.md` states that no earlier output holds, and its two figures.

    ./run.sh --no-sync python scripts/xgdino_info_report.py
        # -> results/xgdino_info_report/{derived.json, complete.json}
        #    results_reports/xg_dino_2026-10/figures/{info_redundancy.png, info_gain_left.png, info_figures.json}

Inputs, each checked against its folder's completion record:
- `results/xgdino_ablation/real/summary.json` (xg-dino 02): B+Q - B as a share of C - B (the known baseline of C3.1);
- `results/xgdino_info_redundancy/verdicts.csv` (ticket 13): R/G per block with its 95% interval -> figure;
- `results/xgdino_info_mask/gain_left.csv` (ticket 15): the C - B gain left after masking, refit, mean fill, all
  shots -> figure.

Descriptive, added after the runs; nothing here is in the pinned protocol 39a547b5. Deterministic in its inputs.
"""

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xgpoc.bundle import sha256_file
from xgpoc.detector import records
from xgpoc.detector.guard import no_statsbomb_reads
from xgpoc.stack import records as stack_records

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "xgdino_info_report"
FIGURES = ROOT / "results_reports" / "xg_dino_2026-10" / "figures"
ABLATION = RESULTS / "xgdino_ablation" / "real" / "summary.json"
VERDICTS = RESULTS / "xgdino_info_redundancy" / "verdicts.csv"
GAIN_LEFT = RESULTS / "xgdino_info_mask" / "gain_left.csv"
# block: Polish label, in reading order (content blocks, then the descriptive quality-like ones)
BLOCKS = {"Xc": "Xc, cała treść", "crowding": "tłok", "shooter": "strzelec", "ball_t0": "piłka t0",
          "ball_m400": "piłka −400 ms", "motion": "ruch", "flags": "flagi", "Xq": "Xq, jakość (opisowo)",
          "Q": "Q z xg-dino (opisowo)"}
REGIONS = {"shooter": "strzelec", "goalkeeper": "bramkarz", "ball": "piłka", "shooter+ball": "strzelec + piłka",
           "goal": "bramka", "background": "tło", "players": "wszyscy zawodnicy",
           "keep_persons": "zostają tylko osoby", "keep_goal": "zostaje tylko bramka",
           "keep_shooter_ball": "zostaje strzelec + piłka"}
INK, MUTED, MARK = "#222222", "#888888", "#2a6fb0"


def interval_plot(path: Path, labels: list[str], points: list[float], lo: list[float], hi: list[float], *,
                  xlabel: str, refs: dict[float, str], title: str) -> None:
    """Horizontal point-and-interval chart, one series, reference lines in grey."""
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(labels) + 1.2), dpi=110)
    ys = list(range(len(labels)))[::-1]
    for x, name in refs.items():
        ax.axvline(x, color=MUTED, lw=1, ls="--" if x else "-", zorder=0)
        ax.text(x, len(labels) - 0.35, name, color=MUTED, fontsize=8, ha="center", va="bottom")
    ax.hlines(ys, lo, hi, color=MARK, lw=2)
    ax.plot(points, ys, "o", color=MARK, ms=6)
    ax.set_yticks(ys, labels, color=INK, fontsize=9)
    ax.set_xlabel(xlabel, color=INK, fontsize=9)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.set_ylim(-0.7, len(labels) + 0.2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.tick_params(axis="y", labelcolor=INK)
    ax.grid(axis="x", color="#e6e6e6", lw=0.8)
    fig.tight_layout()
    fig.savefig(path, metadata={"Software": None})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--figures", type=Path, default=FIGURES)
    args = parser.parse_args()
    with no_statsbomb_reads():
        inputs = {str(p.relative_to(ROOT)): records.verified_file(p) for p in (ABLATION, VERDICTS, GAIN_LEFT)}
        contrasts = json.loads(ABLATION.read_text())["blocks"]["primary"]["contrasts"]
        bq, cb = contrasts["BQ-B"]["log_loss"]["delta"], contrasts["C-B"]["log_loss"]["delta"]
        with VERDICTS.open(newline="") as f:
            verdicts = {r["block"]: r for r in csv.DictReader(f)}
        with GAIN_LEFT.open(newline="") as f:
            left = {r["region"]: r for r in csv.DictReader(f) if r["estimand"] == "refit" and r["fill"] == "mean"}
        derived = {
            "rule": "descriptive ratios read from the tickets' outputs after the runs (ticket 16)",
            "bq_minus_b_share_of_c_minus_b": bq / cb,
        }
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "derived.json").write_text(json.dumps(derived, indent=2) + "\n")

        args.figures.mkdir(parents=True, exist_ok=True)
        rows = [verdicts[b] for b in BLOCKS]
        interval_plot(args.figures / "info_redundancy.png", list(BLOCKS.values()),
                      [float(r["R_over_G"]) for r in rows], [float(r["R_over_G_lo95"]) for r in rows],
                      [float(r["R_over_G_hi95"]) for r in rows], xlabel="R/G (95% przedział z bootstrapu meczów)",
                      refs={0.0: "0", 0.25: "0.25", 0.5: "0.5"},
                      title="Ile zysku C−B pokrywa blok X (redundancja predykcyjna)")
        regs = [left[r] for r in REGIONS]
        interval_plot(args.figures / "info_gain_left.png", list(REGIONS.values()),
                      [float(r["all_point"]) for r in regs], [float(r["all_lo95"]) for r in regs],
                      [float(r["all_hi95"]) for r in regs], xlabel="część zysku C−B, która zostaje (refit, wypełnienie średnią)",
                      refs={0.0: "0", 1.0: "1"}, title="Zysk C−B po zamaskowaniu regionu")
        code, git = stack_records.code_hashes(Path(__file__)), records.git_state()
        figures = {"code": code, "git": git, "inputs": inputs,
                   "files": {p.name: sha256_file(p) for p in sorted(args.figures.glob("info_*.png"))}}
        (args.figures / "info_figures.json").write_text(json.dumps(figures, indent=2, sort_keys=True) + "\n")
        record = {"inputs": inputs, "code": code, "git": git, "finished": datetime.now(UTC).isoformat(timespec="seconds"),
                  "files": records.output_hashes(args.out)}
        (args.out / "complete.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
