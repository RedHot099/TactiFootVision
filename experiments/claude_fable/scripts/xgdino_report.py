"""Report readings of xg-dino (ticket 08; CPU, seconds, no label and no StatsBomb read): a few counts and ratios the
Polish report states that no earlier output holds, and the report's figures copied from the tickets' outputs.

    ./run.sh --no-sync python scripts/xgdino_report.py
        # -> results/xgdino_report/{derived.json, complete.json}
        #    results_reports/xg_dino_2026-10/figures/{*.png, figures.json}

Inputs, each data file checked against its folder's completion record:
- `results/xgdino_probes_layers/encoder_differences.csv` (ticket 06): G-combo minus frozen DINOv2, embedding-only
  probe, per block 20-23, offset and target, with 95% match-bootstrap intervals. Counted here: targets whose interval
  lies above 0 per block at offset 0, and the largest |difference| at block 20;
- `results/xgdino_timing/timing.json` (ticket 03): the ratio of B to A per shot and the homography stage's share of B
  (fp32, batch 1, medians);
- `results/xgdino_probes_last/coverage.csv` (ticket 04): shots whose goalkeeper target is the E3 proxy (no
  goalkeeper detected).

Descriptive, added after the runs; nothing here is in the frozen protocol. Deterministic in its inputs.
"""

import argparse
import csv
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from xgpoc.bundle import sha256_file
from xgpoc.detector import records
from xgpoc.detector.guard import no_statsbomb_reads
from xgpoc.stack import records as stack_records

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "xgdino_report"
FIGURES = ROOT / "results_reports" / "xg_dino_2026-10" / "figures"
DIFFERENCES = RESULTS / "xgdino_probes_layers" / "encoder_differences.csv"
TIMING = RESULTS / "xgdino_timing" / "timing.json"
COVERAGE = RESULTS / "xgdino_probes_last" / "coverage.csv"
# report figure name: source figure (each at most 200 KB)
FIGURE_SOURCES = {
    "ablation_contrasts.png": RESULTS / "xgdino_ablation" / "real" / "figures" / "contrasts_forest.png",
    "ablation_reliability.png": RESULTS / "xgdino_ablation" / "real" / "figures" / "reliability.png",
    "timing.png": RESULTS / "xgdino_timing" / "timing.png",
    "layers_location.png": RESULTS / "xgdino_probes_layers" / "figures" / "location.png",
    "layers_goalkeeper.png": RESULTS / "xgdino_probes_layers" / "figures" / "goalkeeper.png",
    "agreement_errors.png": RESULTS / "xgdino_agreement" / "figures" / "errors.png",
}
MAX_FIGURE_BYTES = 200 * 1024


def encoder_differences(rows: list[dict]) -> dict:
    """Per block at offset 0: targets where G-combo's embedding-only probe beats frozen DINOv2 (95% lower bound > 0)."""
    emb = [r for r in rows if r["probe"] == "embedding" and r["offset_ms"] == "0"]
    out = {"targets": len({r["target"] for r in emb})}
    for block in sorted({r["block"] for r in emb}, key=int):
        here = [r for r in emb if r["block"] == block]
        out[f"block_{block}"] = {
            "g_better": sum(float(r["lo"]) > 0 for r in here),
            "dino_better": sum(float(r["hi"]) < 0 for r in here),
            "max_abs_point": max(abs(float(r["point"])) for r in here),
        }
    return out


def timing_ratios(timing: dict) -> dict:
    head = timing["headline_ms_per_shot"]["fp32_b1"]
    stages = timing["configs"]["B_fp32"]["stages_ms_per_shot"]
    b = timing["configs"]["B_fp32"]["ms_per_shot"]["median"]
    return {
        "b_over_a_10_tails": head["B"] / head["A_10_tails"],
        "b_over_a_1_tail": head["B"] / head["A_1_tail"],
        "homography_share_of_b": stages["homography"]["median"] / b,
        "homography_teams_share_of_b": (stages["homography"]["median"] + stages["teams"]["median"]) / b,
    }


def goalkeeper_proxy(rows: list[dict]) -> dict:
    row = next(r for r in rows if r["target"] == "gk_depth_m")
    present, proxy = int(row["present"]), int(float(row["gk_proxy"]))
    return {"present": present, "proxy": proxy, "proxy_share": proxy / present}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--figures", type=Path, default=FIGURES)
    args = parser.parse_args()
    with no_statsbomb_reads():
        inputs = {str(p.relative_to(ROOT)): records.verified_file(p) for p in (DIFFERENCES, TIMING, COVERAGE)}
        with DIFFERENCES.open(newline="") as f:
            differences = encoder_differences(list(csv.DictReader(f)))
        with COVERAGE.open(newline="") as f:
            proxy = goalkeeper_proxy(list(csv.DictReader(f)))
        derived = {
            "rule": "descriptive counts and ratios read from the tickets' outputs after the runs (ticket 08)",
            "encoder_differences_offset_0": differences,
            "timing_fp32_b1": timing_ratios(json.loads(TIMING.read_text())),
            "goalkeeper_proxy": proxy,
        }
        code, git = stack_records.code_hashes(Path(__file__)), records.git_state()
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "derived.json").write_text(json.dumps(derived, indent=2) + "\n")
        record = {"inputs": inputs, "code": code, "git": git, "finished": datetime.now(UTC).isoformat(timespec="seconds"),
                  "files": records.output_hashes(args.out)}
        (args.out / "complete.json").write_text(json.dumps(record, indent=2) + "\n")

        args.figures.mkdir(parents=True, exist_ok=True)
        sources = {}
        for name, src in FIGURE_SOURCES.items():
            if src.stat().st_size > MAX_FIGURE_BYTES:
                raise SystemExit(f"{src} is larger than {MAX_FIGURE_BYTES} bytes")
            shutil.copyfile(src, args.figures / name)
            sources[name] = {"source": str(src.relative_to(ROOT)), "sha256": sha256_file(src)}
        figures = {"code": code, "git": git, "sources": sources,
                   "files": {p.name: sha256_file(p) for p in sorted(args.figures.glob("*.png"))}}
        (args.figures / "figures.json").write_text(json.dumps(figures, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
