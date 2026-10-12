"""Read every number of the reviewer report and reply back from its source (xg-credibility 14).

    ./run.sh python scripts/check_reviewer_report.py results_reports/reviewer_response_2026-10/REPORT.md \
        --reply results_reports/reviewer_response_2026-10/reply.md

Exit code 0 when every check passes, 1 otherwise (each failure is printed). The report and
reply paths are arguments, so the check runs on drafts with any file name. The claims file
defaults to `claims.toml` next to the report; paths in it are relative to the experiment
directory (`--root`, default: the parent of this script's folder).

How a number is tied to its source:

- In the report, a number, or a number with its interval, is followed by a tag, an HTML
  comment that does not render: `+0.0041 [+0.0020; +0.0061]<!--n:tr.gap-->`.
- `claims.toml` gives each tag a source file, a locator and a format. The checker reads the
  value(s), renders the expected text and requires the text right before the tag to end with
  it. The character before the rendered text must not be a digit, sign or decimal point, so
  a dropped minus sign or digit is caught.
- Locators: `cols` (CSV columns, one row chosen by `row`, or several averaged with
  `agg = "mean"`, or the number of rows `row` matches with `agg = "count"` and
  `cols = []`), `keys` (dotted JSON paths, list indices allowed), `quote` + `values`
  (the source text must contain the quote and the quote each value; used for protocol
  constants) and `git` (the
  SHA-256 of the source file as committed at that revision).
- Options: `scale` multiplies numbers, `abs` takes absolute values, `labels` maps raw
  values (strings, booleans) to report words, `dev` overrides the folder's flag.
- The rendering is `fmt.format(*values)` with every number post-processed: "-" becomes the
  minus sign "−" and the "," thousands separator becomes a space.

Further checks:

- Every decimal number or percentage outside a tagged span fails. Integers (dates, offsets
  in ms, draw counts, ticket numbers) are only checked where they are tagged. Code spans,
  link targets, comments, fenced blocks and section numbers ("## 2.1", "sekcja 2.6") are
  skipped.
- The reply is pasted into an email, so it carries no tags. `[reply].claims` lists its
  claims in reading order; each rendered text must occur after the previous one, and every
  decimal number or percentage of the reply must lie inside one of these matches.
- The report's source table, rows of the form "| `path` | `sha256` | ...", must list every
  claim's file. Each listed file is hashed and compared with the table and, where the
  folder has a `complete.json` with a `files` entry for it, with that record.
- A claim from data seen before (`dev`) must sit under a heading that contains the
  development label (`[report].dev_label`); a reply that uses one must contain the label.
- Every claim must be used by the report or the reply.
"""

import argparse
import csv
import hashlib
import json
import re
import string
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
MINUS = "\N{MINUS SIGN}"
TAG = re.compile(r"<!--n:([\w.\-]+)-->")
# A decimal number (dot or comma) or a percentage, not glued to a word ("v1.2" is a name).
NUMBER = re.compile(rf"(?<![\w.,])[+{MINUS}-]?\d+(?:[.,]\d+)+\s?%?|(?<![\w.,])\d+\s?%")
CODE_SPAN = re.compile(r"`[^`]*`")
LINK_TARGET = re.compile(r"\]\([^)]*\)")
COMMENT = re.compile(r"<!--.*?-->")
# Section numbers are not results: "## 2.1 Title", "sekcja 2.6", "section 2.6".
SECTION = re.compile(r"^#+\s+\d+(?:\.\d+)*|\b(?:sekcj\w*|section)\s+\d+(?:\.\d+)*", re.IGNORECASE)
SOURCE_ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*`([0-9a-f]{64})`\s*\|")
NOT_BEFORE = set(f"0123456789+-{MINUS}.,")
CLAIM_KEYS = {"src", "fmt", "cols", "row", "agg", "keys", "quote", "values", "git", "scale", "abs", "labels", "dev"}


class ClaimError(ValueError):
    """A claim that cannot be read from its source."""


@dataclass(frozen=True)
class Claim:
    id: str
    file: Path
    dev: bool
    text: str


def sha256_file(path: Path) -> str:
    """SHA-256 of a file (kept here so the check does not import torch via `xgpoc.bundle`)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(2**20):
            h.update(block)
    return h.hexdigest()


def render(fmt: str, values: list) -> str:
    """`fmt.format(*values)` with the report's number style: "−" for minus, space for thousands."""
    out = []
    fields = list(string.Formatter().parse(fmt))
    if sum(name is not None for _, name, _, _ in fields) != len(values):
        raise ClaimError(f"format {fmt!r} does not take {len(values)} values")
    it = iter(values)
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
    for part in path.split("."):
        try:
            data = data[int(part)] if isinstance(data, list) else data[part]
        except (KeyError, IndexError, ValueError, TypeError) as e:
            raise ClaimError(f"no {path!r} ({part!r} missing)") from e
    return data


def csv_values(path: Path, spec: dict) -> list:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    wanted = {k: [str(v) for v in (vs if isinstance(vs, list) else [vs])] for k, vs in spec.get("row", {}).items()}
    for column in [*wanted, *spec["cols"]]:
        if rows and column not in rows[0]:
            raise ClaimError(f"{path.name} has no column {column!r}")
    hits = [r for r in rows if all(r[k] in vs for k, vs in wanted.items())]
    if spec.get("agg") == "count":
        if spec["cols"]:
            raise ClaimError(f"{path.name}: agg = count takes cols = []")
        if absent := [f"{k}={v}" for k, vs in wanted.items() for v in vs if v not in {r[k] for r in rows}]:
            raise ClaimError(f"{path.name}: no row with {absent}")   # a misspelt value is not a count of 0
        return [len(hits)]
    if not hits:
        raise ClaimError(f"{path.name}: no row matches {wanted}")
    for k, vs in wanted.items():
        if absent := [v for v in vs if v not in {r[k] for r in hits}]:
            raise ClaimError(f"{path.name}: no row with {k} in {absent}")
    if len(hits) > 1 and spec.get("agg") != "mean":
        raise ClaimError(f"{path.name}: {len(hits)} rows match {wanted}; narrow `row` or set agg")
    if len(hits) == 1:
        return [as_number(hits[0][c]) for c in spec["cols"]]
    try:
        return [sum(float(r[c]) for r in hits) / len(hits) for c in spec["cols"]]
    except ValueError as e:
        raise ClaimError(f"{path.name}: cannot average non-numeric {spec['cols']}") from e


def as_number(cell: str) -> float | str:
    """A CSV cell as a float, or the text itself (labels such as "> +400")."""
    try:
        return float(cell)
    except ValueError:
        return cell


def git_blob_sha256(path: Path, revision: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(path.parent), "show", f"{revision}:./{path.name}"],
        capture_output=True,
        check=False,
    )
    if done.returncode:
        raise ClaimError(f"git show {revision}:{path.name} failed: {done.stderr.decode().strip()}")
    return hashlib.sha256(done.stdout).hexdigest()


def quoted_values(path: Path, quote: str, values: list) -> list:
    """`values` after checking that the source contains `quote` and the quote contains each value."""
    if quote not in path.read_text():
        raise ClaimError(f"{path.name} does not contain {quote!r}")
    in_quote = [float(n.replace(",", ".")) for n in re.findall(r"\d+(?:[.,]\d+)?", quote)]
    for v in values:
        if not any(abs(v - q) < 1e-12 for q in in_quote):
            raise ClaimError(f"value {v} is not a number of the quote {quote!r}")
    return list(values)


def raw_values(path: Path, spec: dict) -> list:
    kinds = [k for k in ("cols", "keys", "quote", "git") if k in spec]
    if len(kinds) != 1:
        raise ClaimError(f"needs exactly one of cols, keys, quote, git (has {kinds})")
    if not path.is_file():
        raise ClaimError(f"source {path} does not exist")
    kind = kinds[0]
    if kind == "cols":
        return csv_values(path, spec)
    if kind == "keys":
        data = json.loads(path.read_text())
        return [json_path(data, k) for k in spec["keys"]]
    if kind == "quote":
        return quoted_values(path, spec["quote"], spec["values"])
    return [git_blob_sha256(path, spec["git"])]


def transform(value, spec: dict):
    if "labels" in spec:
        key = str(value).lower() if isinstance(value, bool) else str(value)
        if key not in spec["labels"]:
            raise ClaimError(f"no label for value {key!r}")
        return spec["labels"][key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        return value
    value *= spec.get("scale", 1)
    return abs(value) if spec.get("abs") else value


def load_claims(path: Path, root: Path) -> tuple[dict[str, Claim], dict, list[str]]:
    """Every claim rendered from its source, the parsed file, and the claims that failed."""
    config = tomllib.loads(path.read_text())
    claims, errors = claims_from_config(config, root)
    return claims, config, errors


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
            values = [transform(v, spec) for v in raw_values(file, spec)]
            text = render(spec["fmt"], values)
        except ClaimError as e:
            errors.append(f"claim {cid}: {e}")
            continue
        claims[cid] = Claim(cid, file.resolve(), spec.get("dev", folder.get("dev", False)), text)
    return claims, errors


def blank_span(line: str, start: int, end: int) -> str:
    """`line` with characters start..end replaced by spaces (positions stay put)."""
    return line[:start] + " " * (end - start) + line[end:]


def glued_before(text: str, start: int) -> bool:
    """True when the character before `start` would extend the number (digit, sign, point)."""
    return start > 0 and text[start - 1] in NOT_BEFORE


def untied_numbers(lines: list[str], label: str) -> list[str]:
    """Decimal numbers and percentages left in `lines` outside code, links and comments."""
    found, fenced = [], False
    for no, line in enumerate(lines, 1):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        for pattern in (COMMENT, CODE_SPAN, LINK_TARGET, SECTION):
            line = pattern.sub(lambda m: " " * len(m.group()), line)
        found += [f"{label}:{no}: number {m.group().strip()!r} is not tied to a source" for m in NUMBER.finditer(line)]
    return found


def check_report(lines: list[str], claims: dict[str, Claim], dev_label: str) -> tuple[list[str], set[str]]:
    """Tags, their rendered text and the dev label; returns failures and the tags used."""
    errors, used, headings = [], set(), []
    out = []
    for no, line in enumerate(lines, 1):
        if heading := re.match(r"^(#+)\s+(.*)", line):
            level = len(heading.group(1))
            headings = [h for h in headings if h[0] < level] + [(level, heading.group(2))]
        for tag in reversed(list(TAG.finditer(line))):
            cid = tag.group(1)
            claim = claims.get(cid)
            used.add(cid)
            if claim is None:
                errors.append(f"REPORT:{no}: unknown or unreadable claim {cid!r}")
                continue
            start = tag.start() - len(claim.text)
            before = line[: tag.start()]
            if not before.endswith(claim.text) or glued_before(line, start):
                found = before[-len(claim.text) - 3 :].strip()
                errors.append(f"REPORT:{no}: {cid} expects {claim.text!r}, text has {found!r}")
                continue
            if claim.dev and not any(dev_label in h for _, h in headings):
                errors.append(f"REPORT:{no}: {cid} is a development result outside a '{dev_label}' heading")
            line = blank_span(line, start, tag.end())
        out.append(line)
    return errors + untied_numbers(out, "REPORT"), used


def occurrences(text: str, needle: str) -> list[int]:
    starts, i = [], text.find(needle)
    while i >= 0:
        end = i + len(needle)
        if not glued_before(text, i) and (end == len(text) or not text[end].isdigit()):
            starts.append(i)
        i = text.find(needle, i + 1)
    return starts


def check_reply(text: str, ids: list[str], claims: dict[str, Claim], dev_label: str) -> list[str]:
    """The listed claims occur in the reply in list order; no other decimal or percentage does.

    Each claim is searched from the end of the previous match, so two numbers swapped between
    claims, or a short claim found inside a later one ("1.2" in "1.2%"), fail.
    """
    errors = []
    if dev_label not in text and any(claims[c].dev for c in ids if c in claims):
        errors.append(f"reply: uses development results but never says '{dev_label}'")
    remaining, cursor = text, 0
    for cid in ids:
        claim = claims.get(cid)
        if claim is None:
            errors.append(f"reply: unknown or unreadable claim {cid!r}")
            continue
        start = next((s for s in occurrences(text, claim.text) if s >= cursor), None)
        if start is None:
            errors.append(f"reply: {cid} expects {claim.text!r} after position {cursor}, not found")
            continue
        cursor = start + len(claim.text)
        remaining = blank_span(remaining, start, cursor)
    return errors + untied_numbers(remaining.split("\n"), "reply")


def check_sources(lines: list[str], claims: dict[str, Claim], root: Path) -> list[str]:
    errors, listed = [], {}
    for no, line in enumerate(lines, 1):
        if row := SOURCE_ROW.match(line):
            listed[(root / row.group(1)).resolve()] = (no, row.group(1), row.group(2))
    for file, (no, name, digest) in listed.items():
        if not file.is_file():
            errors.append(f"REPORT:{no}: source {name} does not exist")
            continue
        actual = sha256_file(file)
        if actual != digest:
            errors.append(f"REPORT:{no}: source {name} has sha256 {actual}, table says {digest}")
        record = file.parent / "complete.json"
        if record.is_file() and record != file:
            recorded = json.loads(record.read_text()).get("files", {}).get(file.name)
            if recorded is not None and recorded != actual:
                errors.append(f"REPORT:{no}: {name} differs from its completion record {record.name}")
    for file in sorted({c.file for c in claims.values()} - set(listed)):
        errors.append(f"source table does not list {file.relative_to(root.resolve())}")
    return errors


def check(report: Path, claims_file: Path, root: Path, reply: Path | None = None) -> list[str]:
    """Every failure of the report (and reply) against the claims and their sources."""
    claims, config, errors = load_claims(claims_file, root)
    dev_label = config.get("report", {}).get("dev_label", "wyniki rozwojowe")
    lines = report.read_text().split("\n")
    report_errors, used = check_report(lines, claims, dev_label)
    errors += report_errors
    reply_ids = config.get("reply", {}).get("claims", [])
    used |= set(reply_ids)  # reply claims count as used even when only the report is checked
    if reply is not None:
        errors += check_reply(reply.read_text(), reply_ids, claims, dev_label)
    errors += [f"claim {cid} is not used" for cid in sorted(set(claims) - used)]
    errors += check_sources(lines, claims, root)
    return errors


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("report", type=Path)
    ap.add_argument("--reply", type=Path)
    ap.add_argument("--claims", type=Path, help="default: claims.toml next to the report")
    ap.add_argument("--root", type=Path, default=EXP, help="base of the claim paths")
    args = ap.parse_args(argv)
    claims_file = args.claims or args.report.with_name("claims.toml")
    errors = check(args.report, claims_file, args.root, args.reply)
    for e in errors:
        print(f"FAIL {e}")
    n_tags = len(TAG.findall(args.report.read_text()))
    print(f"{'FAILED' if errors else 'OK'}: {n_tags} tagged numbers, {len(errors)} failures")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
