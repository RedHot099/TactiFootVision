"""The number check of the xg-detector audit note (xg-detector 02 and 09) and final report (xg-detector 07), audit 3
(pitch-keypoints 04), the pitch-keypoints report (pitch-keypoints 06), the xg-compare report (xg-compare 03) and the
xg-stack report (xg-stack 03): the real documents, and changed copies."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_vendor_report import EXP, check  # noqa: E402
from check_xgd_audit import (  # noqa: E402
    AUDIT,
    AUDIT3,
    DOCUMENTS,
    PK_REPORT,
    REPORT,
    XGC_REPORT,
    XGS_REPORT,
    main,
)
from test_check_vendor_report import write_tree  # noqa: E402

RESULTS = ("results/xgd_audit", "results/xgd_full", "results/xgd_audit2", "results/xgd_audit2_diag", "results/xgd_full2")
READY = AUDIT.is_file() and all((EXP / folder / "complete.json").is_file() for folder in RESULTS)
needs_results = pytest.mark.skipif(not READY, reason="needs the audit note and the results of both audits")


def changed_copy(document: Path, tmp_path: Path, claim: str) -> Path:
    """A copy of `document` in `tmp_path` with the last digit before `claim`'s tag changed."""
    text = document.read_text()
    tag = f"<!--n:{claim}-->"
    assert tag in text
    head, tail = text.split(tag, 1)
    copy = tmp_path / document.name
    copy.write_text(head[:-1] + ("0" if head[-1] != "0" else "1") + tag + tail)
    return copy


@needs_results
def test_the_real_audit_note_passes():
    assert check(AUDIT, AUDIT.with_name("claims.toml"), EXP) == []
    assert main([]) == 0


@needs_results
@pytest.mark.parametrize("claim", ["au.c4.median", "a2.d3.median", "dg.src.d3.ball_m400.med", "fu2.evaluation.t0.shot_ok"])
def test_a_changed_number_in_the_note_fails(tmp_path, capsys, claim):
    copy = changed_copy(AUDIT, tmp_path, claim)
    errors = check(copy, AUDIT.with_name("claims.toml"), EXP)
    assert any(f"{claim} expects" in e for e in errors)
    assert main([str(copy), "--claims", str(AUDIT.with_name("claims.toml"))]) == 1
    assert "FAILED" in capsys.readouterr().out


@needs_results
def test_an_untagged_number_in_the_note_fails(tmp_path):
    copy = tmp_path / "AUDIT.md"
    copy.write_text(AUDIT.read_text() + "\nDodatkowo 12.5 m bez źródła.\n")
    errors = check(copy, AUDIT.with_name("claims.toml"), EXP)
    assert any("'12.5 m' is not tied to a source" in e or "12.5" in e for e in errors)


def test_the_note_check_on_a_planted_tree(tmp_path, capsys):
    report = write_tree(tmp_path)                       # a consistent report with its claims and source table
    assert main([str(report), "--root", str(tmp_path)]) == 0
    report.write_text(report.read_text().replace("2<!--n:reasons-->", "4<!--n:reasons-->"))
    assert main([str(report), "--root", str(tmp_path)]) == 1
    assert "FAILED" in capsys.readouterr().out


def test_without_a_path_the_check_reads_every_document(monkeypatch):
    import check_xgd_audit

    seen = []
    monkeypatch.setattr(check_xgd_audit.cvr, "main", lambda argv: seen.append(argv) or 0)
    assert main(["--root", "/x"]) == 0
    assert seen == [[str(AUDIT), "--claims", str(AUDIT.with_name("claims.toml")), "--root", "/x"],
                    [str(REPORT), "--claims", str(REPORT.with_name("report_claims.toml")), "--root", "/x"],
                    [str(AUDIT3), "--claims", str(AUDIT3.with_name("claims.toml")), "--root", "/x"],
                    [str(PK_REPORT), "--claims", str(PK_REPORT.with_name("report_claims.toml")), "--root", "/x"],
                    [str(XGC_REPORT), "--claims", str(XGC_REPORT.with_name("claims.toml")), "--root", "/x"],
                    [str(XGS_REPORT), "--claims", str(XGS_REPORT.with_name("claims.toml")), "--root", "/x"]]
    assert main(["other.md", "--claims", "c.toml"]) == 0 and seen[-1] == ["other.md", "--claims", "c.toml", "--root", str(EXP)]


def test_without_a_path_one_failing_document_fails_the_check(monkeypatch):
    import check_xgd_audit

    seen = []
    monkeypatch.setattr(check_xgd_audit.cvr, "main", lambda argv: seen.append(argv) or int(argv[0] == str(AUDIT)))
    assert main([]) == 1 and len(seen) == len(DOCUMENTS)        # the other documents are still checked


def test_claims_without_a_report_is_refused():
    with pytest.raises(SystemExit):
        main(["--claims", "c.toml"])


# --- the final report (xg-detector 07) ---

FINAL_CLAIMS = REPORT.with_name("report_claims.toml")
FINAL_RESULTS = ("results/xgd_approach1", "results/xgd_a1_ensemble", "results/xgd_test_power", "results/xgd_test",
                 "results/xgd_report_figures")
FINAL_READY = READY and REPORT.is_file() and all((EXP / folder / "complete.json").is_file() for folder in FINAL_RESULTS)
needs_final = pytest.mark.skipif(not FINAL_READY, reason="needs the final report, the test and the development results")


@needs_final
def test_the_real_final_report_passes():
    assert DOCUMENTS[REPORT] == FINAL_CLAIMS
    assert check(REPORT, FINAL_CLAIMS, EXP) == []


@needs_final
@pytest.mark.parametrize("claim", ["ts.d95", "ts.margin", "ts.a1.ll", "ap.a1.d", "pw.p0", "a2.d3.median", "rf.stretch.med"])
def test_a_changed_number_in_the_final_report_fails(tmp_path, claim):
    assert any(f"{claim} expects" in e for e in check(changed_copy(REPORT, tmp_path, claim), FINAL_CLAIMS, EXP))


@needs_final
def test_a_changed_reading_label_in_the_final_report_fails(tmp_path):
    copy = tmp_path / "REPORT.md"
    copy.write_text(REPORT.read_text().replace("**equivalent within δ<!--n:ts.label-->**", "**success<!--n:ts.label-->**", 1))
    assert any("ts.label expects" in e for e in check(copy, FINAL_CLAIMS, EXP))


@needs_final
def test_test_numbers_under_a_development_heading_fail(tmp_path):
    copy = tmp_path / "REPORT.md"
    copy.write_text(REPORT.read_text().replace("### 6.3 Wynik testu", "### 6.3 Wyniki rozwojowe", 1))
    errors = check(copy, FINAL_CLAIMS, EXP)
    assert any("ts.d is a confirmatory result under a 'wyniki rozwojowe' heading" in e for e in errors)


@needs_final
def test_development_numbers_outside_a_development_heading_fail(tmp_path):
    copy = tmp_path / "REPORT.md"
    copy.write_text(REPORT.read_text().replace("## 5. Podejście 1: wyniki rozwojowe", "## 5. Podejście 1", 1)
                    .replace("(wyniki rozwojowe, ticket 03)", "(ticket 03)", 1))
    assert any("ap.a1.d is a development result outside a 'wyniki rozwojowe' heading" in e
               for e in check(copy, FINAL_CLAIMS, EXP))


@needs_final
def test_an_untagged_number_in_the_final_report_fails(tmp_path):
    copy = tmp_path / "REPORT.md"
    copy.write_text(REPORT.read_text() + "\nDodatkowo ΔLL −0.0031 bez źródła i 1234 strzały.\n")
    errors = check(copy, FINAL_CLAIMS, EXP)
    assert any("'−0.0031' is not tied to a source" in e for e in errors)
    assert any("integer '1234'" in e for e in errors)


# --- audit 3 (pitch-keypoints 04) ---

AUDIT3_RESULTS = ("results/xgd_audit3", "results/xgd_audit3_diag", "results/pk_ball_evidence", "results/pk_square_input",
                  "results/xgd_full3")
AUDIT3_READY = AUDIT3.is_file() and all((EXP / folder / "complete.json").is_file() for folder in AUDIT3_RESULTS)
needs_audit3 = pytest.mark.skipif(not AUDIT3_READY, reason="needs the audit-3 note and its results")


@needs_audit3
def test_the_real_audit_3_note_passes():
    assert check(AUDIT3, AUDIT3.with_name("claims.toml"), EXP) == []


@needs_audit3
@pytest.mark.parametrize("claim", ["a3.g.med", "a3.e1.med", "ev.to400", "dg3.e3.ball_m400.med", "fu3.all.t0.shot_ok"])
def test_a_changed_number_in_the_audit_3_note_fails(tmp_path, claim):
    copy = changed_copy(AUDIT3, tmp_path, claim)
    assert any(f"{claim} expects" in e for e in check(copy, AUDIT3.with_name("claims.toml"), EXP))


# --- the pitch-keypoints report (pitch-keypoints 06) ---

PK_CLAIMS = PK_REPORT.with_name("report_claims.toml")
PK_RESULTS = ("results/pk_dataset", "results/pitchkp_gsr_eval/valid", "results/pitchkp_gsr_eval/test",
              "results/pk_candidates/y11m_960_e18_square/test", "results/pk_candidates/y11m_960_e18_square/retune_valid",
              "results/pk_label_free/y11m_960_e18_square")
PK_READY = AUDIT3_READY and PK_REPORT.is_file() and all((EXP / folder / "complete.json").is_file() for folder in PK_RESULTS)
needs_pk_report = pytest.mark.skipif(not PK_READY, reason="needs the pitch-keypoints report and the phase's results")


@needs_pk_report
def test_the_real_pitch_keypoints_report_passes():
    assert DOCUMENTS[PK_REPORT] == PK_CLAIMS
    assert check(PK_REPORT, PK_CLAIMS, EXP) == []


@needs_pk_report
@pytest.mark.parametrize("claim", ["ct.med", "ct.avail.lo", "bt.d3.med", "rt.paired", "ds.excluded", "a3.e3.med2ci",
                                   "c3.budget", "lf.res.px"])
def test_a_changed_number_in_the_pitch_keypoints_report_fails(tmp_path, claim):
    assert any(f"{claim} expects" in e for e in check(changed_copy(PK_REPORT, tmp_path, claim), PK_CLAIMS, EXP))


@needs_pk_report
def test_test_numbers_outside_a_test_heading_fail(tmp_path):
    copy = tmp_path / "REPORT.md"
    copy.write_text(PK_REPORT.read_text().replace("### 6.1 Wynik testu wobec celu C3", "### 6.1 Wobec celu C3", 1)
                    .replace("## 6. Wynik testu na GSR i Calibration", "## 6. GSR i Calibration", 1))
    assert any("ct.lt1 is a confirmatory result outside a 'wynik testu' heading" in e for e in check(copy, PK_CLAIMS, EXP))


@needs_pk_report
def test_validation_numbers_outside_a_development_heading_fail(tmp_path):
    copy = tmp_path / "REPORT.md"
    copy.write_text(PK_REPORT.read_text().replace("(wyniki rozwojowe, ticket 03)", "(ticket 03)", 1))
    assert any("rt.med is a development result outside a 'wyniki rozwojowe' heading" in e
               for e in check(copy, PK_CLAIMS, EXP))


@needs_pk_report
def test_an_untagged_number_in_the_pitch_keypoints_report_fails(tmp_path):
    copy = tmp_path / "REPORT.md"
    copy.write_text(PK_REPORT.read_text() + "\nDodatkowo mediana 0.41 m i 7261 klatek bez źródła.\n")
    errors = check(copy, PK_CLAIMS, EXP)
    assert any("'0.41' is not tied to a source" in e for e in errors)
    assert any("integer '7261'" in e for e in errors)


XGC_CLAIMS = XGC_REPORT.with_name("claims.toml")
XGC_RESULTS = ("results/xgc_compare/analysis", "results/xgc_compare/recovery")
XGC_READY = XGC_REPORT.is_file() and all((EXP / folder / "complete.json").is_file() for folder in XGC_RESULTS)
needs_xgc_report = pytest.mark.skipif(not XGC_READY, reason="needs the xg-compare report and the phase's results")


@needs_xgc_report
def test_the_real_xg_compare_report_passes():
    assert DOCUMENTS[XGC_REPORT] == XGC_CLAIMS
    assert check(XGC_REPORT, XGC_CLAIMS, EXP) == []


@needs_xgc_report
@pytest.mark.parametrize("claim", ["f.3.eff", "r.14.share", "w.6a.oe.F", "ag.VF", "sv.7.v", "pw.null.bh", "o1.F", "dt.a.rate"])
def test_a_changed_number_in_the_xg_compare_report_fails(tmp_path, claim):
    assert any(f"{claim} expects" in e for e in check(changed_copy(XGC_REPORT, tmp_path, claim), XGC_CLAIMS, EXP))


@needs_xgc_report
def test_an_untagged_number_in_the_xg_compare_report_fails(tmp_path):
    copy = tmp_path / "REPORT.md"
    copy.write_text(XGC_REPORT.read_text() + "\nDodatkowo efekt +0.31 na 512 strzałach bez źródła.\n")
    errors = check(copy, XGC_CLAIMS, EXP)
    assert any("'+0.31' is not tied to a source" in e for e in errors)
    assert any("integer '512'" in e for e in errors)


# --- the xg-stack report (xg-stack 03) ---

XGS_CLAIMS = XGS_REPORT.with_name("claims.toml")
XGS_RESULTS = ("results/xgs_stack/real", "results/xgs_stack/controls", "results/xgs_stack/c5_controls",
               "results/xgs_stack/selection", "results/xgs_stack/report")
XGS_READY = XGS_REPORT.is_file() and all((EXP / folder / "complete.json").is_file() for folder in XGS_RESULTS)
needs_xgs_report = pytest.mark.skipif(not XGS_READY, reason="needs the xg-stack report and the phase's results")


@needs_xgs_report
def test_the_real_xg_stack_report_passes():
    assert DOCUMENTS[XGS_REPORT] == XGS_CLAIMS
    assert check(XGS_REPORT, XGS_CLAIMS, EXP) == []


@needs_xgs_report
@pytest.mark.parametrize("claim", ["rl.G.d", "rl.G.p", "c6.c4.power", "c6.mde", "c5.c4.rec", "cat.box.oe.G", "pt.G.prac1"])
def test_a_changed_number_in_the_xg_stack_report_fails(tmp_path, claim):
    assert any(f"{claim} expects" in e for e in check(changed_copy(XGS_REPORT, tmp_path, claim), XGS_CLAIMS, EXP))


@needs_xgs_report
def test_xg_stack_numbers_outside_a_development_heading_fail(tmp_path):
    copy = tmp_path / "REPORT.md"
    text = XGS_REPORT.read_text()
    title = text.split("\n", 1)[0]
    copy.write_text(text.replace(title, title.replace(" (wyniki rozwojowe)", ""), 1))
    assert any("rl.G.d is a development result outside a 'wyniki rozwojowe' heading" in e
               for e in check(copy, XGS_CLAIMS, EXP))


@needs_xgs_report
def test_an_untagged_number_in_the_xg_stack_report_fails(tmp_path):
    copy = tmp_path / "REPORT.md"
    copy.write_text(XGS_REPORT.read_text() + "\nDodatkowo ΔLL −0.0031 na 4096 strzałach bez źródła.\n")
    errors = check(copy, XGS_CLAIMS, EXP)
    assert any("'−0.0031' is not tied to a source" in e for e in errors)
    assert any("integer '4096'" in e for e in errors)
