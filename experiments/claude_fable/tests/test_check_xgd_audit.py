"""The number check of the xg-detector audit note (xg-detector 02): the real note, and a changed copy."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_vendor_report import EXP, check  # noqa: E402
from check_xgd_audit import REPORT, main  # noqa: E402
from test_check_vendor_report import write_tree  # noqa: E402

READY = REPORT.is_file() and (EXP / "results/xgd_audit/complete.json").is_file() and (EXP / "results/xgd_full/complete.json").is_file()
needs_results = pytest.mark.skipif(not READY, reason="needs the audit note and the xgd_audit / xgd_full results")


@needs_results
def test_the_real_audit_note_passes():
    assert check(REPORT, REPORT.with_name("claims.toml"), EXP) == []
    assert main([]) == 0


@needs_results
def test_a_changed_number_in_the_note_fails(tmp_path, capsys):
    text = REPORT.read_text()
    tag = "<!--n:au.c4.median-->"
    assert tag in text
    head, tail = text.split(tag, 1)
    changed = head[:-1] + ("0" if head[-1] != "0" else "1") + tag + tail      # the last digit before the tag
    copy = tmp_path / "AUDIT.md"
    copy.write_text(changed)
    errors = check(copy, REPORT.with_name("claims.toml"), EXP)
    assert any("au.c4.median expects" in e for e in errors)
    assert main([str(copy), "--claims", str(REPORT.with_name("claims.toml"))]) == 1
    assert "FAILED" in capsys.readouterr().out


@needs_results
def test_an_untagged_number_in_the_note_fails(tmp_path):
    copy = tmp_path / "AUDIT.md"
    copy.write_text(REPORT.read_text() + "\nDodatkowo 12.5 m bez źródła.\n")
    errors = check(copy, REPORT.with_name("claims.toml"), EXP)
    assert any("'12.5 m' is not tied to a source" in e or "12.5" in e for e in errors)


def test_the_note_check_on_a_planted_tree(tmp_path, capsys):
    report = write_tree(tmp_path)                       # a consistent report with its claims and source table
    assert main([str(report), "--root", str(tmp_path)]) == 0
    report.write_text(report.read_text().replace("2<!--n:reasons-->", "4<!--n:reasons-->"))
    assert main([str(report), "--root", str(tmp_path)]) == 1
    assert "FAILED" in capsys.readouterr().out


def test_without_a_path_the_note_check_reads_the_audit_note(monkeypatch):
    import check_xgd_audit

    seen = []
    monkeypatch.setattr(check_xgd_audit.cvr, "main", lambda argv: seen.append(argv) or 0)
    assert main(["--root", "/x"]) == 0 and seen[-1] == [str(REPORT), "--root", "/x"]
    assert main(["other.md", "--claims", "c.toml"]) == 0 and seen[-1] == ["other.md", "--claims", "c.toml", "--root", str(EXP)]
