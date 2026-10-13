"""Read every number of the xg-detector audit note and final report back from its source (xg-detector 02, 09, 07).

    ./run.sh python scripts/check_xgd_audit.py                     # both documents of results_reports/xg_detector_2026-10
    ./run.sh python scripts/check_xgd_audit.py REPORT.md [--claims C]   # one document

The checks of `scripts/check_vendor_report.py` on each document and its claims file (`AUDIT.md` with `claims.toml`,
`REPORT.md` with `report_claims.toml`; a document given on the command line takes `claims.toml` next to it unless
`--claims` says otherwise): each tagged number is rendered from its result file and must match the text, every
decimal number, percentage and integer of two or more digits must be tagged (dates, ticket numbers, code and the
`[report].untagged` phrases excepted), and every source file is listed with its SHA-256 in the document's source
table and agrees with its folder's completion record. In the report, development numbers (folders with
`dev = true`) must stand under a heading with "wyniki rozwojowe" and the test's numbers (`confirmatory = true`) under
a heading with "wynik testu" and none with the development label. See the docstrings of that script and of
`scripts/check_reviewer_report.py` for the tag and claim scheme.

Exit code 0 when every check passes, 1 otherwise (each failure is printed).
"""

import argparse
import sys
from pathlib import Path

import check_vendor_report as cvr

FOLDER = cvr.EXP / "results_reports" / "xg_detector_2026-10"
AUDIT = FOLDER / "AUDIT.md"
REPORT = FOLDER / "REPORT.md"
DOCUMENTS = {AUDIT: FOLDER / "claims.toml", REPORT: FOLDER / "report_claims.toml"}     # document: its claims file


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("report", type=Path, nargs="?", help="default: both documents (DOCUMENTS)")
    ap.add_argument("--claims", type=Path, help="default: claims.toml next to the report")
    ap.add_argument("--root", type=Path, default=cvr.EXP, help="base of the claim paths")
    args = ap.parse_args(argv)
    if args.report is None:
        if args.claims:
            ap.error("--claims needs a report")
        return max(cvr.main([str(report), "--claims", str(claims), "--root", str(args.root)]) for report, claims in DOCUMENTS.items())
    claims = ["--claims", str(args.claims)] if args.claims else []
    return cvr.main([str(args.report), *claims, "--root", str(args.root)])


if __name__ == "__main__":
    sys.exit(main())
