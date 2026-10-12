"""Read every number of the xg-detector audit note back from its source (xg-detector 02).

    ./run.sh python scripts/check_xgd_audit.py [results_reports/xg_detector_2026-10/AUDIT.md]

The checks of `scripts/check_vendor_report.py` on the note and the `claims.toml` next to it: each tagged number is
rendered from its result file and must match the text, every decimal number, percentage and integer of two or more
digits must be tagged (dates, ticket numbers, code and the `[report].untagged` phrases excepted), and every source
file is listed with its SHA-256 in the note's source table and agrees with its folder's completion record. See the
docstrings of that script and of `scripts/check_reviewer_report.py` for the tag and claim scheme.

Exit code 0 when every check passes, 1 otherwise (each failure is printed).
"""

import argparse
import sys
from pathlib import Path

import check_vendor_report as cvr

REPORT = cvr.EXP / "results_reports" / "xg_detector_2026-10" / "AUDIT.md"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("report", type=Path, nargs="?", default=REPORT)
    ap.add_argument("--claims", type=Path, help="default: claims.toml next to the report")
    ap.add_argument("--root", type=Path, default=cvr.EXP, help="base of the claim paths")
    args = ap.parse_args(argv)
    claims = ["--claims", str(args.claims)] if args.claims else []
    return cvr.main([str(args.report), *claims, "--root", str(args.root)])


if __name__ == "__main__":
    sys.exit(main())
