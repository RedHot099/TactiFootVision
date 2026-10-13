"""The number check of the xg-detector audit note (xg-detector 02 and 09) and final report (xg-detector 07): the real
documents, and changed copies."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_vendor_report import EXP, check  # noqa: E402
from check_xgd_audit import DOCUMENTS, FINAL, REPORT, main  # noqa: E402
from test_check_vendor_report import write_tree  # noqa: E402

RESULTS = ("results/xgd_audit", "results/xgd_full", "results/xgd_audit2", "results/xgd_audit2_diag", "results/xgd_full2")
READY = REPORT.is_file() and all((EXP / folder / "complete.json").is_file() for folder in RESULTS)
needs_results = pytest.mark.skipif(not READY, reason="needs the audit note and the results of both audits")


@needs_results
def test_the_real_audit_note_passes():
    assert check(REPORT, REPORT.with_name("claims.toml"), EXP) == []
    assert main([]) == 0


@needs_results
@pytest.mark.parametrize("claim", ["au.c4.median", "a2.d3.median", "dg.src.d3.ball_m400.med", "fu2.evaluation.t0.shot_ok"])
def test_a_changed_number_in_the_note_fails(tmp_path, capsys, claim):
    text = REPORT.read_text()
    tag = f"<!--n:{claim}-->"
    assert tag in text
    head, tail = text.split(tag, 1)
    changed = head[:-1] + ("0" if head[-1] != "0" else "1") + tag + tail      # the last digit before the tag
    copy = tmp_path / "AUDIT.md"
    copy.write_text(changed)
    errors = check(copy, REPORT.with_name("claims.toml"), EXP)
    assert any(f"{claim} expects" in e for e in errors)
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


def test_without_a_path_the_check_reads_both_documents(monkeypatch):
    import check_xgd_audit

    seen = []
    monkeypatch.setattr(check_xgd_audit.cvr, "main", lambda argv: seen.append(argv) or 0)
    assert main(["--root", "/x"]) == 0
    assert seen == [[str(REPORT), "--claims", str(REPORT.with_name("claims.toml")), "--root", "/x"],
                    [str(FINAL), "--claims", str(FINAL.with_name("report_claims.toml")), "--root", "/x"]]
    assert main(["other.md", "--claims", "c.toml"]) == 0 and seen[-1] == ["other.md", "--claims", "c.toml", "--root", str(EXP)]


def test_without_a_path_one_failing_document_fails_the_check(monkeypatch):
    import check_xgd_audit

    seen = []
    monkeypatch.setattr(check_xgd_audit.cvr, "main", lambda argv: seen.append(argv) or int(argv[0] == str(REPORT)))
    assert main([]) == 1 and len(seen) == 2        # the second document is still checked


def test_claims_without_a_report_is_refused():
    with pytest.raises(SystemExit):
        main(["--claims", "c.toml"])


# --- the final report (xg-detector 07) ---

FINAL_CLAIMS = FINAL.with_name("report_claims.toml")
FINAL_RESULTS = ("results/xgd_approach1", "results/xgd_a1_ensemble", "results/xgd_test_power", "results/xgd_test",
                 "results/xgd_report_figures")
FINAL_READY = READY and FINAL.is_file() and all((EXP / folder / "complete.json").is_file() for folder in FINAL_RESULTS)
needs_final = pytest.mark.skipif(not FINAL_READY, reason="needs the final report, the test and the development results")


@needs_final
def test_the_real_final_report_passes():
    assert dict(DOCUMENTS)[FINAL] == FINAL_CLAIMS
    assert check(FINAL, FINAL_CLAIMS, EXP) == []


@needs_final
@pytest.mark.parametrize("claim", ["ts.d95", "ts.margin", "ts.a1.ll", "ap.a1.d", "pw.p0", "a2.d3.median", "rf.stretch.med"])
def test_a_changed_number_in_the_final_report_fails(tmp_path, claim):
    text = FINAL.read_text()
    tag = f"<!--n:{claim}-->"
    assert tag in text
    head, tail = text.split(tag, 1)
    copy = tmp_path / "REPORT.md"
    copy.write_text(head[:-1] + ("0" if head[-1] != "0" else "1") + tag + tail)
    assert any(f"{claim} expects" in e for e in check(copy, FINAL_CLAIMS, EXP))


@needs_final
def test_a_changed_reading_label_in_the_final_report_fails(tmp_path):
    copy = tmp_path / "REPORT.md"
    copy.write_text(FINAL.read_text().replace("**equivalent within δ<!--n:ts.label-->**", "**success<!--n:ts.label-->**", 1))
    assert any("ts.label expects" in e for e in check(copy, FINAL_CLAIMS, EXP))


@needs_final
def test_test_numbers_under_a_development_heading_fail(tmp_path):
    copy = tmp_path / "REPORT.md"
    copy.write_text(FINAL.read_text().replace("### 6.3 Wynik testu", "### 6.3 Wyniki rozwojowe", 1))
    errors = check(copy, FINAL_CLAIMS, EXP)
    assert any("ts.d is a confirmatory result under a 'wyniki rozwojowe' heading" in e for e in errors)


@needs_final
def test_development_numbers_outside_a_development_heading_fail(tmp_path):
    copy = tmp_path / "REPORT.md"
    copy.write_text(FINAL.read_text().replace("## 5. Podejście 1: wyniki rozwojowe", "## 5. Podejście 1", 1)
                    .replace("(wyniki rozwojowe, ticket 03)", "(ticket 03)", 1))
    assert any("ap.a1.d is a development result outside a 'wyniki rozwojowe' heading" in e
               for e in check(copy, FINAL_CLAIMS, EXP))


@needs_final
def test_an_untagged_number_in_the_final_report_fails(tmp_path):
    copy = tmp_path / "REPORT.md"
    copy.write_text(FINAL.read_text() + "\nDodatkowo ΔLL −0.0031 bez źródła i 1234 strzały.\n")
    errors = check(copy, FINAL_CLAIMS, EXP)
    assert any("'−0.0031' is not tied to a source" in e for e in errors)
    assert any("integer '1234'" in e for e in errors)
