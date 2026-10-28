"""Read every number of the detector audit notes, the xg-detector final report, the pitch-keypoints report, the
xg-compare report, the xg-stack report and the xg-stack test report back from its source (xg-detector 02, 09, 07;
pitch-keypoints 04, 06; xg-compare 03; xg-stack 03, 04).

    ./run.sh python scripts/check_xgd_audit.py                     # every document of DOCUMENTS
    ./run.sh python scripts/check_xgd_audit.py REPORT.md [--claims C]   # one document

The checks of `scripts/check_vendor_report.py` on each document and its claims file (`AUDIT.md` with `claims.toml`,
`REPORT.md` with `report_claims.toml`, audit 3's `results_reports/pitch_keypoints_2026-10/AUDIT3.md` with the
`claims.toml` next to it, that folder's phase report `REPORT.md` with its `report_claims.toml`, the xg-compare report
`results_reports/xg_compare_2026-10/REPORT.md` and the xg-stack report `results_reports/xg_stack_2026-10/REPORT.md`,
each with the `claims.toml` next to it, the xg-stack test report `TEST.md` beside it with its `test_claims.toml`, and the
sweep report `results_reports/sweep_2026_10/REPORT.md` and the xg-dino report
`results_reports/xg_dino_2026-10/results.md` (xg-dino 08), each with the `claims.toml` next to it, and the
xg-dino-info report `info.md` beside it with its `info_claims.toml` (xg-dino 16); a document given on the command line takes `claims.toml` next to it unless
`--claims` says otherwise): each tagged number is rendered from
its result file and must match the text, every decimal number, percentage and integer of two or more digits must be tagged (dates, ticket numbers, code and the
`[report].untagged` phrases excepted), and every source file is listed with its SHA-256 in the document's source
table and agrees with its folder's completion record. In the report, development numbers (folders with
`dev = true`) must stand under a heading with "wyniki rozwojowe" and the test's numbers (`confirmatory = true`) under
a heading with "wynik testu" and none with the development label; the pitch-keypoints report uses the same labels for
the choice on the validation splits and the single scoring on the test splits. See the docstrings of that script and of
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
AUDIT3 = cvr.EXP / "results_reports" / "pitch_keypoints_2026-10" / "AUDIT3.md"
PK_REPORT = AUDIT3.with_name("REPORT.md")
XGC_REPORT = cvr.EXP / "results_reports" / "xg_compare_2026-10" / "REPORT.md"
XGS_REPORT = cvr.EXP / "results_reports" / "xg_stack_2026-10" / "REPORT.md"
XGS_TEST = XGS_REPORT.with_name("TEST.md")
SWEEP_REPORT = cvr.EXP / "results_reports" / "sweep_2026_10" / "REPORT.md"
XGDINO_REPORT = cvr.EXP / "results_reports" / "xg_dino_2026-10" / "results.md"
XGDINO_INFO = XGDINO_REPORT.with_name("info.md")
DOCUMENTS = {AUDIT: FOLDER / "claims.toml", REPORT: FOLDER / "report_claims.toml",       # document: its claims file
             AUDIT3: AUDIT3.with_name("claims.toml"), PK_REPORT: PK_REPORT.with_name("report_claims.toml"),
             XGC_REPORT: XGC_REPORT.with_name("claims.toml"), XGS_REPORT: XGS_REPORT.with_name("claims.toml"),
             XGS_TEST: XGS_TEST.with_name("test_claims.toml"), SWEEP_REPORT: SWEEP_REPORT.with_name("claims.toml"),
             XGDINO_REPORT: XGDINO_REPORT.with_name("claims.toml"), XGDINO_INFO: XGDINO_INFO.with_name("info_claims.toml")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("report", type=Path, nargs="?", help="default: every document of DOCUMENTS")
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
