"""scripts/check_report.py: the number check of the detector-data report (detector-data 06)."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import check_report as cr  # noqa: E402

CLAIMS = """
[report]
untagged = ["640"]

[folders]
dev = { path = "dev", dev = true }
test = { path = "test", confirmatory = true }
other = { path = "other" }

[claims]
"v.map" = { src = "dev:det.json", keys = ["rows.gsr/1280.map50", "rows.gsr/1280.map50_95"], fmt = "{:.3f} / {:.3f}" }
"t.placed" = { src = "test:placement.json", keys = ["placed.value", "placed.lo", "placed.hi"], scale = 100, fmt = "{:.1f}% [{:.1f}; {:.1f}]" }
"o.best" = { src = "other:results.csv", cols = ["map"], agg = "max", fmt = "{:.3f}" }
"o.epochs" = { src = "other:results.csv", cols = [], agg = "count", fmt = "{:d}" }
"o.images" = { src = "other:images", listdir = "*.jpg", fmt = "{:,d}" }
"o.diff" = { src = "other:paired.txt", quote = "far -0.1248 [-0.1622 -0.092 ]", values = [-0.1248, -0.1622, -0.092], scale = 100, fmt = "{:+.1f} pp [{:+.1f}; {:+.1f}]" }
"""

BODY = """# Raport

## 1. Wybór (wyniki rozwojowe)

GSR valid 0.760 / 0.465<!--n:v.map-->, przy wejściu 640 px.

## 2. Odczyt (wynik testu)

Umieszczona 77.9% [72.7; 82.4]<!--n:t.placed-->.

## 3. Reszta

Najlepsza epoka 0.487<!--n:o.best--> z 20<!--n:o.epochs-->, obrazów 1 012<!--n:o.images-->, różnica −12.5 pp [−16.2; −9.2]<!--n:o.diff-->.
"""


@pytest.fixture
def report(tmp_path: Path) -> Path:
    (tmp_path / "dev").mkdir()
    (tmp_path / "dev" / "det.json").write_text(
        json.dumps({"rows": {"gsr/1280": {"map50": 0.7596, "map50_95": 0.46503}}})
    )
    (tmp_path / "test").mkdir()
    (tmp_path / "test" / "placement.json").write_text(
        json.dumps({"placed": {"value": 0.7794, "lo": 0.7268, "hi": 0.8243}})
    )
    (tmp_path / "other").mkdir()
    rows = "\n".join(f"{e},{0.4872 if e == 13 else 0.47}" for e in range(1, 21))
    (tmp_path / "other" / "results.csv").write_text("epoch,map\n" + rows + "\n")
    (tmp_path / "other" / "paired.txt").write_text(
        "placed 0.1436 [0.1028 0.1854]\nfar -0.1248 [-0.1622 -0.092 ]\n"
    )
    images = tmp_path / "other" / "images"
    images.mkdir()
    for i in range(1012):
        (images / f"{i:04d}.jpg").touch()
    (images / "notes.txt").touch()
    (tmp_path / "claims.toml").write_text(CLAIMS)
    files = [
        "dev/det.json",
        "test/placement.json",
        "other/results.csv",
        "other/paired.txt",
        "other/images",
    ]
    table = "\n".join(f"| `{f}` | `{cr.sha256_path(tmp_path / f)}` | |" for f in files)
    path = tmp_path / "REPORT.md"
    path.write_text(
        BODY + "\n## Źródła\n\n| plik | skrót | |\n|---|---|---|\n" + table + "\n"
    )
    return path


def errors(report: Path) -> list[str]:
    return cr.check(report, report.with_name("claims.toml"), report.parent)


def test_a_report_whose_numbers_match_their_files_passes(report: Path) -> None:
    assert errors(report) == []


def test_a_number_that_differs_from_its_file_fails(report: Path) -> None:
    report.write_text(report.read_text().replace("0.760 / 0.465", "0.760 / 0.466"))
    assert any("v.map expects '0.760 / 0.465'" in e for e in errors(report))


def test_a_dropped_minus_sign_fails(report: Path) -> None:
    report.write_text(report.read_text().replace("−12.5 pp", "12.5 pp"))
    assert any("o.diff expects" in e for e in errors(report))


def test_an_untagged_decimal_or_integer_fails(report: Path) -> None:
    report.write_text(
        report.read_text().replace(
            "## 3. Reszta\n",
            "## 3. Reszta\n\nOkoło 0.5 i 1 300 klatek, w tickecie 05.\n",
        )
    )
    found = errors(report)
    assert any("number '0.5'" in e for e in found)
    assert any("integer '1 300'" in e for e in found)
    assert not any("'05'" in e for e in found)


def test_development_and_test_numbers_stay_under_their_headings(report: Path) -> None:
    text = report.read_text()
    report.write_text(
        text.replace("## 1. Wybór (wyniki rozwojowe)", "## 1. Wybór").replace(
            "## 2. Odczyt (wynik testu)", "## 2. Odczyt (wyniki rozwojowe)"
        )
    )
    found = errors(report)
    assert any("v.map is a development result outside" in e for e in found)
    assert any("t.placed is a test result under" in e for e in found)


def test_a_changed_source_file_or_unlisted_source_fails(report: Path) -> None:
    (report.parent / "other" / "paired.txt").write_text(
        "placed 0.1436 [0.1028 0.1854]\nfar -0.1248 [-0.1622 -0.092 ]\nextra\n"
    )
    text = report.read_text()
    report.write_text(
        "\n".join(line for line in text.split("\n") if "dev/det.json" not in line)
    )
    found = errors(report)
    assert any("paired.txt has sha256" in e for e in found)
    assert any("does not list" in e and "det.json" in e for e in found)


def test_a_directory_hashes_as_its_sorted_entry_names(tmp_path: Path) -> None:
    (tmp_path / "b").touch()
    (tmp_path / "a").touch()
    assert cr.sha256_path(tmp_path) == hashlib.sha256(b"a\nb").hexdigest()
