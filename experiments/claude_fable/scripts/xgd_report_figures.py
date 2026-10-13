"""Figures of the final xg-detector report that no result file states directly (xg-detector 07; CPU, seconds).

    ./run.sh python scripts/xgd_report_figures.py      # -> results/xgd_report_figures/{figures.json, complete.json}

Only one figure is needed: how far the pitch keypoint model's keypoints lie from the landmarks on SoccerNet frames.
Ticket 08 wrote every residual to `results/xgd_homography_dev/keypoint_residuals.csv` (training-match t0 frames whose
refined homography the painted lines confirm, agreement >= 0.9; one row per frame, input and detected landmark) and
quoted the median in its Comments. The report's number check reads medians from a file, so this script states them:
per keypoint input, the median residual in pixels at 1280x720, the share within 10 and 20 px, and the counts. The
residual file is checked against its completion record first. Nothing else is read: no label, no StatsBomb data.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from xgpoc.detector import records

ROOT = Path(__file__).resolve().parents[1]
RESIDUALS = ROOT / "results/xgd_homography_dev/keypoint_residuals.csv"
OUT = ROOT / "results/xgd_report_figures"


def keypoint_figures(residuals: pd.DataFrame) -> dict:
    """Per keypoint input: residual count, frames, median px, share within 10 and 20 px."""
    out = {}
    for name, rows in residuals.groupby("input", sort=True):
        out[str(name)] = {"residuals": int(len(rows)), "frames": int(rows.shot_id.nunique()),
                          "median_px": float(rows.px.median()), "share_lt10": float((rows.px < 10).mean()),
                          "share_lt20": float((rows.px < 20).mean())}
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    try:
        digest = records.verified_file(RESIDUALS)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    figures = {"keypoint_residuals": keypoint_figures(pd.read_csv(RESIDUALS, usecols=["shot_id", "input", "px"])),
               "source": {"path": str(RESIDUALS.relative_to(ROOT)), "sha256": digest}}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "figures.json").write_text(json.dumps(figures, indent=2) + "\n")
    record = {"purpose": "xg-detector 07: report figures not stated directly by a result file",
              "inputs": {figures["source"]["path"]: digest}, "code": records.code_hashes(Path(__file__)),
              "git": records.git_state(), "finished": datetime.now(UTC).isoformat(timespec="seconds"),
              "files": records.output_hashes(args.out)}
    (args.out / "complete.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(figures, indent=2))


if __name__ == "__main__":
    main()
