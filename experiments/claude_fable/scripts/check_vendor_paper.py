r"""Read every number of the xg-vendor paper draft back from the number-checked report (xg-vendor 17).

    ./run.sh python scripts/check_vendor_paper.py            # check; exit 0 when everything passes
    ./run.sh python scripts/check_vendor_paper.py --write    # rewrite numbers.tex and the figures, then check

The paper (LaTeX, `results_reports/xg_vendor_2026-10/paper/`) has no typed number. Each number is a macro
`\val{id}`, and `numbers.tex` defines every id. This script writes `numbers.tex` from the `claims.toml` next to the
paper:

- A paper claim names a claim of the report's `claims.toml` (`from`, default: the paper claim's own id). It may
  override only the rendering: `fmt` (English number style) and `labels` (English words for raw values). The source
  file, the locator, the scale and the confirmatory or development flag stay the report's. Every number of the paper
  is therefore a number of the report, read from the same result file.
- The rendered text becomes LaTeX that works in text and in math mode: signs, ±, δ and ≥ go into `\ensuremath`,
  % is escaped, and the thousands separator becomes a comma.

The check fails (exit 1, each failure printed) when:

- the report's own number check (`check_vendor_report.py`) fails; it also checks every source file's SHA-256
  against the report's source table and the completion records;
- a paper claim cannot be read, names an unknown report claim, overrides anything but `fmt` and `labels`, or is
  not used; or `\val` names an id that has no claim;
- `numbers.tex` differs from what the claims render now;
- a figure differs from what `vendor_paper_figures.py` draws from the result files now;
- the body of a .tex file (after `\begin{document}`) has a decimal number, a percentage or an integer of two or
  more digits outside `\val`. Skipped: comments, `\label`/`\ref`/`\cite`/`\url`/file arguments, layout lengths,
  table column specs and column spans, dates, years, clock times, and the phrases in `[paper].untagged`;
- a `\val` whose text has a space stands inside inline math, where the space would be lost;
- a confirmatory claim stands outside a heading that contains `[paper].confirmatory_label`, or under a heading that
  contains `[paper].dev_label`; or a development claim stands outside a heading with the development label.
  Headings are `\section`, `\subsection`, `\subsubsection` and `\paragraph` (matched case-insensitively). The
  abstract counts as a heading whose title is its own text, so it may carry confirmatory numbers only if it says so.
"""

import argparse
import re
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import NamedTuple

import check_reviewer_report as crr
import check_vendor_report as cvr
import vendor_paper_figures as figures

EXP = crr.EXP
PAPER = EXP / "results_reports" / "xg_vendor_2026-10" / "paper"
NUMBERS = "numbers.tex"
OVERRIDES = {"from", "fmt", "labels"}
VAL = re.compile(r"\\val\{([^{}]*)\}")
HEADING = re.compile(r"\\(section|subsection|subsubsection|paragraph)\*?(?:\[[^\]]*\])?\{")
LEVELS = {"section": 1, "subsection": 2, "subsubsection": 3, "paragraph": 4}
TEX_COMMENT = re.compile(r"(?<!\\)%.*$")
# Arguments that are names, not results: labels, references, citations, files, URLs, environments, packages.
NAME_ARGS = re.compile(
    r"\\(?:label|ref|eqref|cref|Cref|autoref|cite[a-z]*|url|href|input|include|includegraphics|bibliography"
    r"|bibliographystyle|usepackage|begin|end|IfFileExists)\*?(?:\[[^\]]*\])*\{[^{}]*\}"
)
# Layout: lengths, column specs after \begin{tabular}, spans of \multicolumn and \cmidrule, \arraystretch.
LAYOUT = re.compile(
    r"[-+]?\d*\.?\d+\s*(?:pt|mm|cm|em|ex|in|bp)\b|[-+]?\d*\.?\d+\s*\\(?:linewidth|columnwidth|textwidth|hsize)"
    r"|(?<=\\begin\{tabular\})\{[^{}]*\}|(?<=\\begin\{tabular\*\})\{[^{}]*\}\{[^{}]*\}"
    r"|\\multi(?:column|row)\{\d+\}|\\cmidrule(?:\([^)]*\))?\{\d+-\d+\}|\\renewcommand\{\\arraystretch\}\{[\d.]+\}"
    r"|\\setcounter\{[^{}]*\}\{\d+\}"
)
LATEX_CHARS = {
    "\N{MINUS SIGN}": "\\ensuremath{-}", "±": "\\ensuremath{\\pm}", "δ": "\\ensuremath{\\delta}", "≥": "\\ensuremath{\\geq}",
    "≤": "\\ensuremath{\\leq}", "%": "\\%",
}
THOUSANDS = re.compile(r"(?<=\d) (?=\d{3}(?!\d))")


class PaperError(ValueError):
    """A paper claims file that cannot be used."""


def latex(text: str) -> str:
    """A rendered claim as LaTeX that works in text and math mode: signs and symbols in \\ensuremath, % escaped,
    comma thousands separator."""
    text = THOUSANDS.sub(",", text)
    text = re.sub(r"(?<![\w$])\+(?=\d)", r"\\ensuremath{+}", text)
    return "".join(LATEX_CHARS.get(ch, ch) for ch in text)


def paper_config(paper_claims: Path, root: Path) -> tuple[dict, dict, Path]:
    """The parsed paper claims, the parsed report claims and the report path."""
    paper = tomllib.loads(paper_claims.read_text())
    report = root / paper.get("paper", {}).get("report", "")
    if not report.is_file():
        raise PaperError(f"[paper].report {report} is not a file")
    return paper, tomllib.loads(report.with_name("claims.toml").read_text()), report


def merged_specs(paper: dict, report: dict) -> tuple[dict, dict[str, str], list[str]]:
    """Report claim specs with the paper's rendering overrides, keyed by paper id; also paper id -> report id."""
    specs, origin, errors = {}, {}, []
    for pid, override in paper.get("claims", {}).items():
        if unknown := sorted(set(override) - OVERRIDES):
            errors.append(f"paper claim {pid}: may override only fmt and labels, not {', '.join(unknown)}")
            continue
        rid = override.get("from", pid)
        if rid not in report.get("claims", {}):
            errors.append(f"paper claim {pid}: no report claim {rid!r}")
            continue
        specs[pid] = {**report["claims"][rid], **{k: v for k, v in override.items() if k != "from"}}
        origin[pid] = rid
    return specs, origin, errors


def numbers_tex(claims: dict[str, crr.Claim]) -> str:
    """The generated numbers.tex: the \\val macro and one definition per claim, in claim order."""
    head = [
        "% Generated by scripts/check_vendor_paper.py --write from claims.toml; do not edit.",
        "% \\val{id} prints the number of claim id; an unknown id stops the LaTeX run.",
        "\\providecommand{\\defval}[2]{\\expandafter\\def\\csname xgv:#1\\endcsname{#2}}",
        "\\providecommand{\\val}{}",
        "\\DeclareRobustCommand{\\val}[1]{\\ifcsname xgv:#1\\endcsname\\csname xgv:#1\\endcsname"
        "\\else\\PackageError{numbers}{No number with id #1}{}\\fi}",
    ]
    return "\n".join(head + [f"\\defval{{{c.id}}}{{{latex(c.text)}}}" for c in claims.values()]) + "\n"


def braced(text: str, start: int) -> str:
    """The content of the brace group that opens at text[start] (nested braces allowed)."""
    depth = 0
    for i in range(start, len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return text[start + 1 : i]
    return text[start + 1 :]


def body_lines(text: str) -> list[tuple[int, str]]:
    """(line number, line without comment) of every line after \\begin{document}."""
    lines = text.split("\n")
    start = next((i for i, line in enumerate(lines) if "\\begin{document}" in line), -1) + 1
    return [(no, TEX_COMMENT.sub("", line)) for no, line in enumerate(lines[start:], start + 1)]


def untagged_numbers(name: str, text: str, untagged: list[str]) -> list[str]:
    """Numbers in the body of a .tex file that do not come from \\val."""
    found = []
    for no, line in body_lines(text):
        line = VAL.sub(" ", line)
        for pattern in (LAYOUT, NAME_ARGS, cvr.NOT_RESULTS):
            line = pattern.sub(" ", line)
        line = cvr.blank_phrases(line, untagged)
        line = line.replace("\\%", "%")
        hits = [m.group().strip() for m in crr.NUMBER.finditer(line)]
        hits += [m.group() for m in cvr.INTEGER.finditer(crr.NUMBER.sub(" ", line))]
        found += [f"{name}:{no}: number {h!r} is not a \\val" for h in hits]
    return found


def headed_lines(text: str):
    """(line number, line, heading titles in force) for each body line.

    Titles are read from the whole body, so a title may span lines and nest braces; a heading counts from its own
    line on. The abstract is a heading whose title is its own text, in force from \\begin{abstract} to
    \\end{abstract}.
    """
    lines = body_lines(text)
    body = "\n".join(line for _, line in lines)
    starts = [0]
    for _, line in lines:
        starts.append(starts[-1] + len(line) + 1)
    events: dict[int, list[tuple[int, str]]] = {}   # line index -> headings that open there
    for m in HEADING.finditer(body):
        index = next(i for i in range(len(lines)) if starts[i + 1] > m.start())
        events.setdefault(index, []).append((LEVELS[m.group(1)], braced(body, m.end() - 1)))
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, re.DOTALL)
    stack: list[tuple[int, str]] = []
    for i, (no, line) in enumerate(lines):
        if "\\begin{abstract}" in line and abstract:
            stack = [(1, abstract.group(1))]
        for level, title in events.get(i, []):
            stack = [h for h in stack if h[0] < level] + [(level, title)]
        yield no, line, [title for _, title in stack]
        if "\\end{abstract}" in line:
            stack = []


def separation(name: str, text: str, confirmatory: set[str], dev: set[str], conf_label: str, dev_label: str) -> list[str]:
    """Confirmatory claims under confirmatory headings only; development claims under development headings only."""
    errors = []
    for no, line, titles in headed_lines(text):
        for m in VAL.finditer(line):
            pid = m.group(1)
            errors += [f"{name}:{no}: {e}" for e in cvr.misplaced(pid, titles, confirmatory=pid in confirmatory,
                                                                    dev=pid in dev, confirmatory_label=conf_label,
                                                                    dev_label=dev_label)]
    return errors


def spaced_in_math(name: str, text: str, claims: dict[str, crr.Claim]) -> list[str]:
    """A \\val inside inline math whose text has a space: math mode would drop the space ("0.711[0.678, ...")."""
    errors = []
    for no, line in body_lines(text):
        for span in re.findall(r"(?<!\\)\$([^$]*)\$", line):
            errors += [f"{name}:{no}: \\val{{{m.group(1)}}} has spaces and stands in math mode; move it out of $...$"
                       for m in VAL.finditer(span) if m.group(1) in claims and " " in claims[m.group(1)].text]
    return errors


def stale_figures(paper_dir: Path, root: Path) -> list[str]:
    """Figures in paper_dir/figures that differ from a fresh drawing from the result files."""
    errors = []
    with tempfile.TemporaryDirectory() as tmp:
        for fresh in figures.build(Path(tmp), root):
            kept = paper_dir / "figures" / fresh.name
            if not kept.is_file() or kept.read_bytes() != fresh.read_bytes():
                errors.append(f"figures/{fresh.name} is not what vendor_paper_figures.py draws now; run --write")
    return errors


class PaperClaims(NamedTuple):
    paper_config: dict      # the paper's claims.toml
    report_config: dict     # the report's claims.toml
    report_path: Path
    claims: dict[str, crr.Claim]
    origin: dict[str, str]  # paper id -> report id
    errors: list[str]       # claims that could not be read


def render_paper_claims(paper_dir: Path, root: Path) -> PaperClaims:
    """Every paper claim rendered from the report claim it names, with the paper's overrides."""
    paper, report_config, report = paper_config(paper_dir / "claims.toml", root)
    specs, origin, errors = merged_specs(paper, report_config)
    claims, claim_errors = crr.claims_from_config({"folders": report_config.get("folders", {}), "claims": specs}, root)
    return PaperClaims(paper, report_config, report, claims, origin, errors + [f"paper {e}" for e in claim_errors])


def check(paper_dir: Path, root: Path = EXP, *, with_figures: bool = True) -> list[str]:
    """Every failure of the paper draft against its claims, the report and the result files."""
    r = render_paper_claims(paper_dir, root)
    claims, errors = r.claims, list(r.errors)
    errors += [f"report: {e}" for e in cvr.check(r.report_path, r.report_path.with_name("claims.toml"), root)]
    confirmatory = {pid for pid, rid in r.origin.items() if rid in cvr.confirmatory_claims(r.report_config)}
    dev = {pid for pid, c in claims.items() if c.dev}
    settings = r.paper_config.get("paper", {})
    numbers = paper_dir / NUMBERS
    if not numbers.is_file() or numbers.read_text() != numbers_tex(claims):
        errors.append(f"{NUMBERS} is not what the claims render now; run --write")
    used: set[str] = set()
    for name in settings.get("tex", []):
        text = (paper_dir / name).read_text()
        for no, line in body_lines(text):
            for m in VAL.finditer(line):
                used.add(m.group(1))
                if m.group(1) not in claims:
                    errors.append(f"{name}:{no}: \\val{{{m.group(1)}}} has no readable claim")
        errors += untagged_numbers(name, text, list(settings.get("untagged", [])))
        errors += spaced_in_math(name, text, claims)
        errors += separation(name, text, confirmatory, dev, settings.get("confirmatory_label", "confirmatory results"),
                             settings.get("dev_label", "development results"))
    errors += [f"paper claim {pid} is not used" for pid in sorted(set(r.origin) - used)]
    if with_figures:
        errors += stale_figures(paper_dir, root)
    return errors


def write(paper_dir: Path, root: Path = EXP) -> None:
    """Rewrite numbers.tex and the figures from the claims and result files."""
    (paper_dir / NUMBERS).write_text(numbers_tex(render_paper_claims(paper_dir, root).claims))
    figures.build(paper_dir / "figures", root)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paper", type=Path, nargs="?", default=PAPER, help="the paper folder")
    ap.add_argument("--root", type=Path, default=EXP, help="base of the claim and result paths")
    ap.add_argument("--write", action="store_true", help="rewrite numbers.tex and the figures first")
    args = ap.parse_args(argv)
    n_claims = 0
    try:
        if args.write:
            write(args.paper, args.root)
        errors = check(args.paper, args.root)
        n_claims = len(render_paper_claims(args.paper, args.root).claims)
    except (OSError, ValueError, KeyError) as e:   # unreadable files, bad TOML (a ValueError), missing result rows
        errors = [f"cannot read the paper, its claims or the report: {e}"]
    for e in errors:
        print(f"FAIL {e}")
    print(f"{'FAILED' if errors else 'OK'}: {n_claims} paper numbers read back, {len(errors)} failures")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
