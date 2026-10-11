"""The timing-source verdict record (written by xg-vendor ticket 04, obeyed by the H4 residual arm of ticket 09).

The record is a JSON object:

    {"schema": "xgv-timing-verdict-1",
     "verdict": "shared" | "independent" | "unresolved",
     "rule_sha256": sha256 hex of the rule document fixed before the comparison ran,
     ... any further fields (rule document path, decision time, statistics, code version) are kept verbatim}

Only "independent" allows an analysis that reads the timing residual as evidence about the contact. A development
override exists for exercising the residual arm before ticket 04 has written its record; it is refused on a
confirmatory table.
"""

import json
from dataclasses import dataclass
from pathlib import Path

SCHEMA = "xgv-timing-verdict-1"
VERDICTS = ("shared", "independent", "unresolved")
ALLOWING = "independent"
REQUIRED_FIELDS = ("schema", "verdict", "rule_sha256")


class TimingVerdictError(ValueError):
    """The verdict record is malformed, or it does not allow the timing-residual arm."""


@dataclass(frozen=True)
class TimingVerdict:
    """A validated verdict record; `record` keeps every field of the file."""

    verdict: str
    rule_sha256: str
    record: dict


def is_sha256(text) -> bool:
    return isinstance(text, str) and len(text) == 64 and set(text) <= set("0123456789abcdef")


def parse_timing_verdict(record: dict, expected_rule_sha256: str | None = None) -> TimingVerdict:
    """Validate a verdict record (module docstring); `expected_rule_sha256` pins the rule document when given."""
    missing = [f for f in REQUIRED_FIELDS if f not in record]
    if missing:
        raise TimingVerdictError(f"timing verdict record lacks {missing}")
    if record["schema"] != SCHEMA:
        raise TimingVerdictError(f"timing verdict schema {record['schema']!r}, expected {SCHEMA}")
    if record["verdict"] not in VERDICTS:
        raise TimingVerdictError(f"unknown timing verdict {record['verdict']!r}; expected one of {VERDICTS}")
    if not is_sha256(record["rule_sha256"]):
        raise TimingVerdictError("rule_sha256 is not a sha256 hex digest")
    if expected_rule_sha256 is not None and record["rule_sha256"] != expected_rule_sha256:
        raise TimingVerdictError("the verdict was reached under another rule document than the expected one")
    return TimingVerdict(record["verdict"], record["rule_sha256"], dict(record))


def load_timing_verdict(path: Path, expected_rule_sha256: str | None = None) -> TimingVerdict:
    """Read and validate the verdict record at `path` (`parse_timing_verdict`)."""
    return parse_timing_verdict(json.loads(Path(path).read_text()), expected_rule_sha256)


def residual_arm_permission(verdict: TimingVerdict | None, table_kind: str, development_override: bool = False) -> str:
    """Why the timing-residual arm may run; raises `TimingVerdictError` when it may not."""
    if verdict is not None and verdict.verdict == ALLOWING:
        return f"timing verdict {ALLOWING!r} (rule {verdict.rule_sha256[:12]})"
    if development_override:
        if table_kind != "development":
            raise TimingVerdictError("the development override is refused on a confirmatory table")
        found = "no verdict record" if verdict is None else f"verdict {verdict.verdict!r}"
        return f"development override ({found})"
    found = "there is no timing verdict record" if verdict is None else f"the timing verdict is {verdict.verdict!r}"
    raise TimingVerdictError(f"the timing-residual arm needs the verdict {ALLOWING!r}; {found}")
