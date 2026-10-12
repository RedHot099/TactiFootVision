"""Read every number of the xg-vendor report back from its source (xg-vendor 16).

    ./run.sh python scripts/check_vendor_report.py [results_reports/xg_vendor_2026-10/REPORT.md]

Runs every check of `scripts/check_reviewer_report.py` on the report and the `claims.toml` next
to it: each tagged number is rendered from its result file and must match the text, every
decimal number or percentage must be tagged, every source file is listed with its SHA-256 and
agrees with its folder's completion record, and a development claim stands under a heading with
the development label. See that script's docstring for the tag and claim scheme. Two rules
are added for this report:

- Confirmatory and development numbers stay apart. A folder of `claims.toml` with
  `confirmatory = true` holds results of the confirmatory run. Each of its claims must stand
  under a heading containing `[report].confirmatory_label` and under no heading containing the
  development label.
- Integers count too. Every integer of two or more digits (a number with space-separated
  thousands counts as one) must be tagged. Skipped: code spans, link targets, comments, fenced
  blocks, section numbers, dates, years, clock times, ticket references and the literal phrases
  listed in `[report].untagged` (fixed design constants, each named there).

Exit code 0 when every check passes, 1 otherwise (each failure is printed).
"""

import argparse
import re
import sys
import tomllib
from pathlib import Path

import check_reviewer_report as crr

EXP = crr.EXP
REPORT = EXP / "results_reports" / "xg_vendor_2026-10" / "REPORT.md"
# Integers that are not results: dates, seasons, years, clock times, ticket references.
NOT_RESULTS = re.compile(
    r"\d{4}-\d{2}-\d{2}(?:T[\d:.]+Z?)?|\d{4}/\d{2,4}|\b20\d{2}\b|\b\d{1,2}:\d{2}(?::\d{2})?\b"
    r"|\b(?:ticket\w*|xg-vendor)\s+\d{2}(?:\s*[–-]\s*\d{2})?",
    re.IGNORECASE,
)
# A two-or-more-digit integer, or a number with space-separated thousands, not part of a word or decimal
# (a full stop or comma after it ends the sentence, it does not make a decimal).
INTEGER = re.compile(r"(?<![\w.,])\d{1,3}(?: \d{3})+(?!\w|[.,]\d)|(?<![\w.,])\d{2,}(?!\w|[.,]\d)")


def tag_spans_blanked(line: str, claims: dict[str, crr.Claim]) -> str:
    """`line` with each tag and the claim text it closes replaced by spaces."""
    for tag in reversed(list(crr.TAG.finditer(line))):
        claim = claims.get(tag.group(1))
        start = tag.start()
        if claim is not None and line[: tag.start()].endswith(claim.text):
            start -= len(claim.text)
        line = crr.blank_span(line, start, tag.end())
    return line


def blank_phrases(line: str, phrases: list[str]) -> str:
    """`line` with each allowed phrase blanked, as a whole only: "400 ms" does not hide the tail of "2400 ms" or "1 400 ms"."""
    for phrase in phrases:
        line = re.sub(rf"(?<![\w.,])(?<!\d ){re.escape(phrase)}", lambda m: " " * len(m.group()), line)
    return line


def untagged_integers(lines: list[str], claims: dict[str, crr.Claim], untagged: list[str]) -> list[str]:
    """Integers of two or more digits left outside tags, code, links, comments, dates and the allowed phrases."""
    found, fenced = [], False
    for no, line in enumerate(lines, 1):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        line = tag_spans_blanked(line, claims)
        for pattern in (crr.COMMENT, crr.CODE_SPAN, crr.LINK_TARGET, crr.SECTION, NOT_RESULTS):
            line = pattern.sub(lambda m: " " * len(m.group()), line)
        line = blank_phrases(line, untagged)
        found += [f"REPORT:{no}: integer {m.group()!r} is not tied to a source" for m in INTEGER.finditer(line)]
    return found


def confirmatory_claims(config: dict) -> set[str]:
    """Ids of the claims read from a folder marked `confirmatory = true`."""
    folders = config.get("folders", {})
    out = set()
    for cid, spec in config.get("claims", {}).items():
        alias, sep, _ = str(spec.get("src", "")).partition(":")
        if sep and folders.get(alias, {}).get("confirmatory", False):
            out.add(cid)
    return out


def misplaced(cid: str, titles: list[str], *, confirmatory: bool, dev: bool, confirmatory_label: str,
              dev_label: str) -> list[str]:
    """Why claim `cid` may not stand under headings `titles` (labels match case-insensitively); empty when it may.

    A confirmatory claim needs a heading with the confirmatory label and none with the development label; a
    development claim needs a heading with the development label.
    """
    lowered = [t.lower() for t in titles]
    in_conf = any(confirmatory_label.lower() in t for t in lowered)
    in_dev = any(dev_label.lower() in t for t in lowered)
    out = []
    if confirmatory and in_dev:
        out.append(f"{cid} is a confirmatory result under a '{dev_label}' heading")
    if confirmatory and not in_conf:
        out.append(f"{cid} is a confirmatory result outside a '{confirmatory_label}' heading")
    if dev and not in_dev:
        out.append(f"{cid} is a development result outside a '{dev_label}' heading")
    return out


def separation(lines: list[str], confirmatory: set[str], dev_label: str, confirmatory_label: str) -> list[str]:
    """A confirmatory claim outside a confirmatory heading, or under a development heading."""
    errors, headings = [], []
    for no, line in enumerate(lines, 1):
        if heading := re.match(r"^(#+)\s+(.*)", line):
            level = len(heading.group(1))
            headings = [h for h in headings if h[0] < level] + [(level, heading.group(2))]
        titles = [h for _, h in headings]
        for tag in crr.TAG.finditer(line):
            errors += [f"REPORT:{no}: {e}" for e in misplaced(tag.group(1), titles, confirmatory=tag.group(1) in confirmatory,
                                                               dev=False, confirmatory_label=confirmatory_label,
                                                               dev_label=dev_label)]
    return errors


def check(report: Path, claims_file: Path, root: Path) -> list[str]:
    """Every failure of the report: the reviewer-report checks plus separation and integers."""
    errors = crr.check(report, claims_file, root)
    claims, config, _ = crr.load_claims(claims_file, root)   # load errors are already in `errors`
    settings = config.get("report", {})
    dev_label = settings.get("dev_label", "wyniki rozwojowe")
    confirmatory_label = settings.get("confirmatory_label", "wyniki konfirmacyjne")
    lines = report.read_text().split("\n")
    errors += separation(lines, confirmatory_claims(config), dev_label, confirmatory_label)
    errors += untagged_integers(lines, claims, list(settings.get("untagged", [])))
    return errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("report", type=Path, nargs="?", default=REPORT)
    ap.add_argument("--claims", type=Path, help="default: claims.toml next to the report")
    ap.add_argument("--root", type=Path, default=EXP, help="base of the claim paths")
    args = ap.parse_args(argv)
    claims_file = args.claims or args.report.with_name("claims.toml")
    try:
        errors = check(args.report, claims_file, args.root)
    except (OSError, tomllib.TOMLDecodeError) as e:
        errors = [f"cannot read the report or its claims: {e}"]
    for e in errors:
        print(f"FAIL {e}")
    n_tags = len(crr.TAG.findall(args.report.read_text())) if args.report.is_file() else 0
    print(f"{'FAILED' if errors else 'OK'}: {n_tags} tagged numbers, {len(errors)} failures")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
