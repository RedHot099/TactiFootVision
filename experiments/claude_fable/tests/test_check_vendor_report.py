"""The xg-vendor report number check (xg-vendor 16) on a small tree whose answers are known, and on the real report."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_reviewer_report import ClaimError, load_claims  # noqa: E402
from check_vendor_report import EXP, REPORT, check, main  # noqa: E402

CLAIMS = """
[report]
dev_label = "wyniki rozwojowe"
confirmatory_label = "wyniki konfirmacyjne"
untagged = ["400 ms"]

[folders]
cf = { path = "conf", confirmatory = true }
dv = { path = "dev", dev = true }
gate = { path = "gate" }

[claims]
"h2" = { src = "cf:h2.json", keys = ["diff", "lo", "hi"], fmt = "{:+.4f} [{:+.4f}; {:+.4f}]" }
"n" = { src = "cf:h2.json", keys = ["n"], fmt = "{:,d}" }
"dev_h2" = { src = "dv:h2.json", keys = ["diff"], fmt = "{:+.4f}" }
"reasons" = { src = "gate:reasons.csv", row = { reason = "dead_ball" }, cols = [], agg = "count", fmt = "{:d}" }
"all_reasons" = { src = "gate:reasons.csv", cols = [], agg = "count", fmt = "{:d}" }
"""

REPORT_TEXT = """# Raport, ticket 16, 2026-10-12

Wstęp: `kod 135`, offset 400 ms, sezon 2015/16, 11:57:06 UTC, ticket 03.

## 1. H2 (wyniki konfirmacyjne)

Δ = −0.0004 [−0.0026; +0.0018]<!--n:h2--> na 2 721<!--n:n--> strzałach.

## 2. Wejścia

Martwa piłka: 2<!--n:reasons--> z 3<!--n:all_reasons-->.

## 3. Kontekst (wyniki rozwojowe)

Deweloperskie: −0.0007<!--n:dev_h2-->.

## Źródła

"""


def write_tree(root: Path, report: str = REPORT_TEXT, claims: str = CLAIMS) -> Path:
    for folder in ("conf", "dev", "gate"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    (root / "conf/h2.json").write_text(json.dumps({"diff": -0.00037, "lo": -0.00262, "hi": 0.00183, "n": 2721}))
    (root / "dev/h2.json").write_text(json.dumps({"diff": -0.00069}))
    (root / "gate/reasons.csv").write_text("shot,reason\na,dead_ball\nb,dead_ball\nc,pass\n")
    rows = "\n".join(f"| `{p}` | `{hashlib.sha256((root / p).read_bytes()).hexdigest()}` | x |"
                     for p in ("conf/h2.json", "dev/h2.json", "gate/reasons.csv"))
    path = root / "REPORT.md"
    path.write_text(report + rows + "\n")
    (root / "claims.toml").write_text(claims)
    return path


def test_a_consistent_report_passes(tmp_path):
    report = write_tree(tmp_path)
    assert check(report, tmp_path / "claims.toml", tmp_path) == []


def test_a_changed_number_fails(tmp_path):
    report = write_tree(tmp_path, REPORT_TEXT.replace("+0.0018]", "+0.0019]"))
    errors = check(report, tmp_path / "claims.toml", tmp_path)
    assert any("h2 expects" in e for e in errors)


def test_a_confirmatory_number_under_a_development_heading_fails(tmp_path):
    moved = REPORT_TEXT.replace("Deweloperskie: −0.0007<!--n:dev_h2-->.",
                                "Deweloperskie: −0.0007<!--n:dev_h2-->, a konfirmacyjnie 2 721<!--n:n-->.")
    errors = check(write_tree(tmp_path, moved), tmp_path / "claims.toml", tmp_path)
    assert any("confirmatory result under a 'wyniki rozwojowe' heading" in e for e in errors)


def test_a_confirmatory_number_outside_a_confirmatory_heading_fails(tmp_path):
    moved = REPORT_TEXT.replace("Martwa piłka:", "Strzały 2 721<!--n:n-->. Martwa piłka:")
    errors = check(write_tree(tmp_path, moved), tmp_path / "claims.toml", tmp_path)
    assert any("outside a 'wyniki konfirmacyjne' heading" in e for e in errors)


def test_a_development_number_outside_a_development_heading_fails(tmp_path):
    moved = REPORT_TEXT.replace("na 2 721<!--n:n--> strzałach.", "na 2 721<!--n:n--> strzałach, rozwojowo −0.0007<!--n:dev_h2-->.")
    errors = check(write_tree(tmp_path, moved), tmp_path / "claims.toml", tmp_path)
    assert any("development result outside" in e for e in errors)


@pytest.mark.parametrize("text", ["na 2 721 strzałach", "w 37 meczach", "offset 300 ms", "było ich 37", "razem 2 721, czyli",
                                  "offset 2400 ms", "offset 1 400 ms"])
def test_an_untagged_integer_fails(tmp_path, text):
    errors = check(write_tree(tmp_path, REPORT_TEXT + text + ".\n\n"), tmp_path / "claims.toml", tmp_path)
    assert any("integer" in e and "is not tied to a source" in e for e in errors)


def test_dates_tickets_code_and_allowed_phrases_are_not_results(tmp_path):
    # the fixture's introduction holds a date, a season, a clock time, a ticket number, code and "400 ms"
    report = write_tree(tmp_path)
    assert not [e for e in check(report, tmp_path / "claims.toml", tmp_path) if "integer" in e]


def test_row_counts(tmp_path):
    write_tree(tmp_path)
    claims, _, errors = load_claims(tmp_path / "claims.toml", tmp_path)
    assert errors == []
    assert (claims["reasons"].text, claims["all_reasons"].text) == ("2", "3")


def test_a_row_count_with_columns_is_refused(tmp_path):
    write_tree(tmp_path, claims=CLAIMS.replace('cols = [], agg = "count"', 'cols = ["shot"], agg = "count"', 1))
    _, _, errors = load_claims(tmp_path / "claims.toml", tmp_path)
    assert any("agg = count takes cols = []" in e for e in errors)
    assert issubclass(ClaimError, ValueError)


def test_a_misspelt_row_value_is_not_a_count_of_zero(tmp_path):
    write_tree(tmp_path, claims=CLAIMS.replace('reason = "dead_ball"', 'reason = "dead_bal"'))
    _, _, errors = load_claims(tmp_path / "claims.toml", tmp_path)
    assert any("no row with ['reason=dead_bal']" in e for e in errors)


def test_main_exit_codes(tmp_path, capsys):
    report = write_tree(tmp_path)
    assert main([str(report), "--root", str(tmp_path)]) == 0
    report.write_text(report.read_text().replace("2<!--n:reasons-->", "4<!--n:reasons-->"))
    assert main([str(report), "--root", str(tmp_path)]) == 1
    assert "FAILED" in capsys.readouterr().out


@pytest.mark.skipif(not (EXP / "results/xgv_confirmatory/complete.json").is_file(), reason="needs the confirmatory results")
def test_the_real_report_passes():
    assert check(REPORT, REPORT.with_name("claims.toml"), EXP) == []
