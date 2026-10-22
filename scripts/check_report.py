"""Read every number of a research report back from its result file (detector-data 06).

    uv run python scripts/check_report.py results_reports/detector_data_2026-10/REPORT.md [--claims C] [--root R]

A port of the number check of the earlier phase reports (research/xg-detector, `scripts/check_reviewer_report.py` and
`scripts/check_vendor_report.py`): the same tag scheme and the same rules, so a report reads and checks the same way.

How a number is tied to its source:

- In the report a number, or a number with its interval, is followed by an HTML comment that does not render:
  `77.9% [72.7; 82.4]<!--n:t.c.placed-->`.
- `claims.toml` (next to the report unless `--claims`) gives each tag a source, a locator and a format. The checker
  reads the value(s), renders the expected text and requires the text right before the tag to end with it, with no
  digit, sign or decimal point glued in front (a dropped minus sign or digit fails).
- Sources: `src = "alias:name"` resolves through `[folders]` (a folder `path` is absolute or relative to `--root`, the
  repository by default); `src = "path"` alone is relative to `--root` or absolute.
- Locators (exactly one): `keys` (dotted JSON paths; list indices allowed; a key may itself contain "/", e.g.
  `rows.gsr/1280.map50`), `cols` (CSV columns of the one row `row` picks, or `agg = "mean" | "max" | "min"` over the
  rows it picks, or `agg = "count"` with `cols = []`), `quote` + `values` (the text file contains the quote and the
  quote contains each value) and `listdir` (the number of directory entries matching the glob, e.g. `"*"`).
- Options: `scale` multiplies numbers, `abs` takes absolute values. `fmt.format(*values)` renders them; "-" becomes
  the minus sign "−" and the "," thousands separator a space.

Further checks:

- Every decimal number or percentage, and every integer of two or more digits ("99 478" counts as one), outside a
  tagged span fails. Skipped: code spans, link targets, comments, fenced blocks, section numbers, dates, years, clock
  times, ticket references ("ticket 02", "tickety 01 i 03"), (m)AP50 / AP50-95 names and the phrases of `[report].untagged` (design constants).
- The source table, rows "| `path` | `sha256` | ...", must list every claim's file. Each listed file is hashed and
  compared with the table (a directory hashes as the sorted, newline-joined names of its entries) and, where its
  folder has a `complete.json` with a `files` entry for it, with that record.
- A claim from a folder with `dev = true` (the choice on development data) must stand under a heading containing
  `[report].dev_label`; one from a folder with `confirmatory = true` (the single test read) under a heading
  containing `[report].confirmatory_label` and under none with the development label.
- Every claim must be used.

Exit code 0 when every check passes, 1 otherwise (each failure is printed).
"""

import argparse
import csv
import hashlib
import json
import re
import string
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MINUS = "\N{MINUS SIGN}"
TAG = re.compile(r"<!--n:([\w.\-/]+)-->")
# A decimal number or a percentage, not glued to a word ("v1.2" is a name).
NUMBER = re.compile(rf"(?<![\w.,])[+{MINUS}-]?\d+(?:[.,]\d+)+\s?%?|(?<![\w.,])\d+\s?%")
# An integer of two or more digits, or one with space-separated thousands, not part of a word or a decimal.
INTEGER = re.compile(
    r"(?<![\w.,])\d{1,3}(?: \d{3})+(?!\w|[.,]\d)|(?<![\w.,])\d{2,}(?!\w|[.,]\d)"
)
CODE_SPAN = re.compile(r"`[^`]*`")
LINK_TARGET = re.compile(r"\]\([^)]*\)")
COMMENT = re.compile(r"<!--.*?-->")
SECTION = re.compile(
    r"^#+\s+\d+(?:\.\d+)*|\b(?:sekcj\w*|section)\s+\d+(?:\.\d+)*", re.IGNORECASE
)
# Integers that are not results: dates, years, clock times, ticket references, metric names.
NOT_RESULTS = re.compile(
    r"\d{4}-\d{2}-\d{2}(?:T[\d:.]+Z?)?|\b20\d{2}\b|\b\d{1,2}:\d{2}(?::\d{2})?\b"
    r"|\b(?:ticke\w*|detector-data|ball-detection)\s+\d{2}(?:\s*(?:[–-]|,|i)\s*\d{2})*"
    r"|\bm?AP50(?:-95)?\b|\b50-95\b",
    re.IGNORECASE,
)
SOURCE_ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*`([0-9a-f]{64})`\s*\|")
NOT_BEFORE = set(f"0123456789+-{MINUS}.,")
CLAIM_KEYS = {
    "src",
    "fmt",
    "cols",
    "row",
    "agg",
    "keys",
    "quote",
    "values",
    "listdir",
    "scale",
    "abs",
}
LOCATORS = ("cols", "keys", "quote", "listdir")


class ClaimError(ValueError):
    """A claim that cannot be read from its source."""


@dataclass(frozen=True)
class Claim:
    id: str
    file: Path
    dev: bool
    confirmatory: bool
    text: str


def sha256_path(path: Path) -> str:
    """SHA-256 of a file, or of a directory's sorted entry names joined by newlines."""
    h = hashlib.sha256()
    if path.is_dir():
        h.update("\n".join(sorted(p.name for p in path.iterdir())).encode())
        return h.hexdigest()
    with path.open("rb") as f:
        while block := f.read(2**20):
            h.update(block)
    return h.hexdigest()


def render(fmt: str, values: list) -> str:
    """`fmt.format(*values)` with the report's number style: "−" for minus, space for thousands."""
    fields = list(string.Formatter().parse(fmt))
    if sum(name is not None for _, name, _, _ in fields) != len(values):
        raise ClaimError(f"format {fmt!r} does not take {len(values)} values")
    out, it = [], iter(values)
    for literal, name, spec, _ in fields:
        out.append(literal)
        if name is None:
            continue
        if name:
            raise ClaimError(f"format {fmt!r}: use automatic fields {{}} only")
        value = next(it)
        if isinstance(value, str):
            out.append(format(value, spec))
            continue
        if spec.endswith("d"):
            if value != int(value):
                raise ClaimError(f"{value} is not an integer for format {spec!r}")
            value = int(value)
        out.append(format(value, spec).replace("-", MINUS).replace(",", " "))
    return "".join(out)


def json_path(data, path: str):
    """The value at a dotted path; a key containing dots is not supported, one containing "/" is."""
    for part in path.split("."):
        try:
            data = data[int(part)] if isinstance(data, list) else data[part]
        except (KeyError, IndexError, ValueError, TypeError) as e:
            raise ClaimError(f"no {path!r} ({part!r} missing)") from e
    return data


def csv_values(path: Path, spec: dict) -> list:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    wanted = {
        k: [str(v) for v in (vs if isinstance(vs, list) else [vs])]
        for k, vs in spec.get("row", {}).items()
    }
    for column in [*wanted, *spec["cols"]]:
        if rows and column not in rows[0]:
            raise ClaimError(f"{path.name} has no column {column!r}")
    hits = [r for r in rows if all(r[k] in vs for k, vs in wanted.items())]
    agg = spec.get("agg")
    if agg == "count":
        if spec["cols"]:
            raise ClaimError(f"{path.name}: agg = count takes cols = []")
        return [len(hits)]
    if not hits:
        raise ClaimError(f"{path.name}: no row matches {wanted}")
    if agg is None:
        if len(hits) > 1:
            raise ClaimError(
                f"{path.name}: {len(hits)} rows match {wanted}; narrow `row` or set agg"
            )
        return [float(hits[0][c]) for c in spec["cols"]]
    reducers = {"mean": lambda xs: sum(xs) / len(xs), "max": max, "min": min}
    if agg not in reducers:
        raise ClaimError(f"unknown agg {agg!r}")
    return [reducers[agg]([float(r[c]) for r in hits]) for c in spec["cols"]]


def quoted_values(path: Path, quote: str, values: list) -> list:
    """`values` after checking that the source contains `quote` and the quote contains each value."""
    if quote not in path.read_text():
        raise ClaimError(f"{path.name} does not contain {quote!r}")
    in_quote = [
        float(n.replace(",", "")) for n in re.findall(r"-?\d+(?:[.,]\d+)*", quote)
    ]
    for v in values:
        if not any(abs(abs(v) - abs(q)) < 1e-12 for q in in_quote):
            raise ClaimError(f"value {v} is not a number of the quote {quote!r}")
    return list(values)


def raw_values(path: Path, spec: dict) -> list:
    kinds = [k for k in LOCATORS if k in spec]
    if len(kinds) != 1:
        raise ClaimError(f"needs exactly one of {', '.join(LOCATORS)} (has {kinds})")
    kind = kinds[0]
    if kind == "listdir":
        if not path.is_dir():
            raise ClaimError(f"source {path} is not a directory")
        return [sum(1 for _ in path.glob(spec["listdir"]))]
    if not path.is_file():
        raise ClaimError(f"source {path} does not exist")
    if kind == "cols":
        return csv_values(path, spec)
    if kind == "keys":
        data = json.loads(path.read_text())
        return [json_path(data, k) for k in spec["keys"]]
    return quoted_values(path, spec["quote"], spec["values"])


def transform(value, spec: dict):
    if isinstance(value, bool) or not isinstance(value, int | float):
        return value
    value *= spec.get("scale", 1)
    return abs(value) if spec.get("abs") else value


def claims_from_config(config: dict, root: Path) -> tuple[dict[str, Claim], list[str]]:
    """Every claim of a parsed claims file rendered from its source, and the claims that failed."""
    folders = config.get("folders", {})
    claims, errors = {}, []
    for cid, spec in config.get("claims", {}).items():
        if missing := [k for k in ("src", "fmt") if k not in spec]:
            errors.append(f"claim {cid}: missing {', '.join(missing)}")
            continue
        if unknown := sorted(set(spec) - CLAIM_KEYS):
            errors.append(f"claim {cid}: unknown keys {', '.join(unknown)}")
            continue
        alias, sep, name = spec["src"].partition(":")
        if sep and "path" not in folders.get(alias, {}):
            errors.append(f"claim {cid}: unknown folder {alias!r}")
            continue
        folder = folders.get(alias, {}) if sep else {}
        file = root / folder["path"] / name if sep else root / spec["src"]
        try:
            text = render(
                spec["fmt"], [transform(v, spec) for v in raw_values(file, spec)]
            )
        except ClaimError as e:
            errors.append(f"claim {cid}: {e}")
            continue
        claims[cid] = Claim(
            cid,
            file.resolve(),
            folder.get("dev", False),
            folder.get("confirmatory", False),
            text,
        )
    return claims, errors


def blank_span(line: str, start: int, end: int) -> str:
    """`line` with characters start..end replaced by spaces (positions stay put)."""
    return line[:start] + " " * (end - start) + line[end:]


def blank(line: str, pattern: re.Pattern) -> str:
    return pattern.sub(lambda m: " " * len(m.group()), line)


def blank_phrases(line: str, phrases: list[str]) -> str:
    """`line` with each allowed phrase blanked as a whole: "640" does not hide the tail of "1 640" or "6400"."""
    for phrase in phrases:
        line = re.sub(
            rf"(?<![\w.,])(?<!\d ){re.escape(phrase)}(?!\w|[.,]\d)",
            lambda m: " " * len(m.group()),
            line,
        )
    return line


def misplaced(
    claim: Claim, titles: list[str], dev_label: str, confirmatory_label: str
) -> list[str]:
    lowered = [t.lower() for t in titles]
    in_dev = any(dev_label.lower() in t for t in lowered)
    in_conf = any(confirmatory_label.lower() in t for t in lowered)
    out = []
    if claim.dev and not in_dev:
        out.append(
            f"{claim.id} is a development result outside a '{dev_label}' heading"
        )
    if claim.confirmatory and in_dev:
        out.append(f"{claim.id} is a test result under a '{dev_label}' heading")
    if claim.confirmatory and not in_conf:
        out.append(
            f"{claim.id} is a test result outside a '{confirmatory_label}' heading"
        )
    return out


def check_report(
    lines: list[str], claims: dict[str, Claim], config: dict
) -> tuple[list[str], set[str]]:
    """Tags, their rendered text, their headings and untied numbers; returns failures and the tags used."""
    report = config.get("report", {})
    dev_label = report.get("dev_label", "wyniki rozwojowe")
    confirmatory_label = report.get("confirmatory_label", "wynik testu")
    untagged = report.get("untagged", [])
    errors, used, headings, fenced = [], set(), [], False
    for no, line in enumerate(lines, 1):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        if heading := re.match(r"^(#+)\s+(.*)", line):
            level = len(heading.group(1))
            headings = [h for h in headings if h[0] < level] + [
                (level, heading.group(2))
            ]
        titles = [h for _, h in headings]
        for tag in reversed(list(TAG.finditer(line))):
            cid = tag.group(1)
            used.add(cid)
            claim = claims.get(cid)
            if claim is None:
                errors.append(f"REPORT:{no}: unknown or unreadable claim {cid!r}")
                line = blank_span(line, tag.start(), tag.end())
                continue
            start = tag.start() - len(claim.text)
            if not line[: tag.start()].endswith(claim.text) or (
                start > 0 and line[start - 1] in NOT_BEFORE
            ):
                found = line[max(0, start - 3) : tag.start()].strip()
                errors.append(
                    f"REPORT:{no}: {cid} expects {claim.text!r}, text has {found!r}"
                )
                line = blank_span(line, tag.start(), tag.end())
                continue
            errors += [
                f"REPORT:{no}: {e}"
                for e in misplaced(claim, titles, dev_label, confirmatory_label)
            ]
            line = blank_span(line, start, tag.end())
        for pattern in (COMMENT, CODE_SPAN, LINK_TARGET, SECTION, NOT_RESULTS):
            line = blank(line, pattern)
        line = blank_phrases(line, untagged)
        errors += [
            f"REPORT:{no}: number {m.group().strip()!r} is not tied to a source"
            for m in NUMBER.finditer(line)
        ]
        errors += [
            f"REPORT:{no}: integer {m.group()!r} is not tied to a source"
            for m in INTEGER.finditer(line)
        ]
    return errors, used


def check_sources(lines: list[str], claims: dict[str, Claim], root: Path) -> list[str]:
    errors, listed = [], {}
    for no, line in enumerate(lines, 1):
        if row := SOURCE_ROW.match(line):
            listed[(root / row.group(1)).resolve()] = (no, row.group(1), row.group(2))
    for path, (no, name, digest) in listed.items():
        if not path.exists():
            errors.append(f"REPORT:{no}: source {name} does not exist")
            continue
        actual = sha256_path(path)
        if actual != digest:
            errors.append(
                f"REPORT:{no}: source {name} has sha256 {actual}, table says {digest}"
            )
        record = path.parent / "complete.json"
        if path.is_file() and record.is_file() and record != path:
            recorded = json.loads(record.read_text()).get("files", {}).get(path.name)
            if recorded is not None and recorded != actual:
                errors.append(
                    f"REPORT:{no}: {name} differs from its completion record {record}"
                )
    errors += [
        f"source table does not list {file}"
        for file in sorted({c.file for c in claims.values()} - set(listed))
    ]
    return errors


def check(report: Path, claims_file: Path, root: Path) -> list[str]:
    """Every failure of the report against its claims and their sources."""
    config = tomllib.loads(claims_file.read_text())
    claims, errors = claims_from_config(config, root)
    lines = report.read_text().split("\n")
    report_errors, used = check_report(lines, claims, config)
    errors += report_errors
    errors += [f"claim {cid} is not used" for cid in sorted(set(claims) - used)]
    errors += check_sources(lines, claims, root)
    return errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("report", type=Path)
    ap.add_argument(
        "--claims", type=Path, help="default: claims.toml next to the report"
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="base of relative claim paths (default: the repository)",
    )
    args = ap.parse_args(argv)
    errors = check(
        args.report, args.claims or args.report.with_name("claims.toml"), args.root
    )
    for e in errors:
        print(f"FAIL {e}")
    n_tags = len(TAG.findall(args.report.read_text()))
    print(
        f"{'FAILED' if errors else 'OK'}: {n_tags} tagged numbers, {len(errors)} failures"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
