"""The xg-vendor paper number check and its figures (xg-vendor 17) on a small tree with known answers, and on the real paper."""

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import vendor_paper_figures as figures  # noqa: E402
from check_vendor_paper import (  # noqa: E402
    EXP,
    NUMBERS,
    PAPER,
    check,
    latex,
    main,
    numbers_tex,
    render_paper_claims,
    write,
)

REPORT_CLAIMS = """
[report]
dev_label = "wyniki rozwojowe"
confirmatory_label = "wyniki konfirmacyjne"

[folders]
cf = { path = "conf", confirmatory = true }
dv = { path = "dev", dev = true }
gate = { path = "gate" }

[claims]
"h2" = { src = "cf:h2.json", keys = ["diff", "lo", "hi"], fmt = "{:+.4f} [{:+.4f}; {:+.4f}]" }
"n" = { src = "cf:h2.json", keys = ["n"], fmt = "{:,d}" }
"dev_h2" = { src = "dv:h2.json", keys = ["diff"], fmt = "{:+.4f}" }
"gate_pass" = { src = "gate:gate.json", keys = ["passed"], labels = { true = "przechodzi", false = "nie przechodzi" }, fmt = "{}" }
"""

REPORT_TEXT = """# Raport

## 1. H2 (wyniki konfirmacyjne)

Δ = −0.0004 [−0.0026; +0.0018]<!--n:h2--> na 2 721<!--n:n--> strzałach.

## 2. Wejścia

Bramka przechodzi<!--n:gate_pass-->.

## 3. Kontekst (wyniki rozwojowe)

Deweloperskie: −0.0007<!--n:dev_h2-->.

## Źródła

"""

PAPER_CLAIMS = """
[paper]
report = "REPORT.md"
tex = ["main.tex"]
confirmatory_label = "confirmatory results"
dev_label = "development results"
untagged = ['$-400$\\,ms']

[claims]
"h2" = { fmt = "{:+.4f} [{:+.4f}, {:+.4f}]" }
"n" = {}
"dev_h2" = {}
"gate" = { from = "gate_pass", labels = { true = "passed", false = "failed" } }
"""

MAIN_TEX = r"""\documentclass{article}
\def\scale{0.75}
\input{numbers}
\begin{document}
\begin{abstract}
Our confirmatory results: the frame changes log-loss by \val{h2}.
\end{abstract}
\section{Data}
\label{sec:data}
The gate \val{gate}.
\section{Confirmatory results}
On \val{n} shots.
\section{Context (development results)}
Earlier: \val{dev_h2}.
\end{document}
"""


def write_tree(root: Path, *, report: str = REPORT_TEXT, tex: str = MAIN_TEX, paper_claims: str = PAPER_CLAIMS) -> Path:
    """A report with its sources and a paper folder whose numbers.tex is current; returns the paper folder."""
    for folder in ("conf", "dev", "gate", "paper"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    (root / "conf/h2.json").write_text(json.dumps({"diff": -0.00037, "lo": -0.00262, "hi": 0.00183, "n": 2721}))
    (root / "dev/h2.json").write_text(json.dumps({"diff": -0.00069}))
    (root / "gate/gate.json").write_text(json.dumps({"passed": True}))
    rows = "\n".join(f"| `{p}` | `{hashlib.sha256((root / p).read_bytes()).hexdigest()}` | x |"
                     for p in ("conf/h2.json", "dev/h2.json", "gate/gate.json"))
    (root / "REPORT.md").write_text(report + rows + "\n")
    (root / "claims.toml").write_text(REPORT_CLAIMS)
    paper = root / "paper"
    (paper / "claims.toml").write_text(paper_claims)
    (paper / "main.tex").write_text(tex)
    (paper / NUMBERS).write_text(numbers_tex(render_paper_claims(paper, root).claims))
    return paper


def errors_of(tmp_path, **kw) -> list[str]:
    return check(write_tree(tmp_path, **kw), tmp_path, with_figures=False)


def test_a_consistent_paper_passes(tmp_path):
    assert errors_of(tmp_path) == []


def test_latex_rendering():
    assert latex("−0.0004 [−0.0026, +0.0018]") == r"\ensuremath{-}0.0004 [\ensuremath{-}0.0026, \ensuremath{+}0.0018]"
    assert latex("2 721 shots, 95%") == r"2,721 shots, 95\%"
    assert latex("±200 ms at δ") == r"\ensuremath{\pm}200 ms at \ensuremath{\delta}"


def test_numbers_tex_uses_the_paper_format_and_labels(tmp_path):
    text = (write_tree(tmp_path) / NUMBERS).read_text()
    assert r"\defval{h2}{\ensuremath{-}0.0004 [\ensuremath{-}0.0026, \ensuremath{+}0.0018]}" in text
    assert r"\defval{n}{2,721}" in text
    assert r"\defval{gate}{passed}" in text


def test_a_changed_result_makes_numbers_tex_stale(tmp_path):
    paper = write_tree(tmp_path)
    (tmp_path / "conf/h2.json").write_text(json.dumps({"diff": -0.00047, "lo": -0.00262, "hi": 0.00183, "n": 2721}))
    errors = check(paper, tmp_path, with_figures=False)
    assert any(e.startswith(NUMBERS) for e in errors)
    assert any(e.startswith("report:") for e in errors)   # the report's own check fails too


def test_write_refreshes_numbers_tex(tmp_path, monkeypatch):
    paper = write_tree(tmp_path)
    (paper / NUMBERS).write_text("stale\n")
    monkeypatch.setattr(figures, "build", lambda out, root: [])
    write(paper, tmp_path)
    assert check(paper, tmp_path, with_figures=False) == []


def test_a_failing_report_check_fails_the_paper(tmp_path):
    errors = errors_of(tmp_path, report=REPORT_TEXT.replace("+0.0018]", "+0.0019]"))
    assert any(e.startswith("report:") and "h2 expects" in e for e in errors)


@pytest.mark.parametrize("body", ["It is 0.05 here.", "On 135 matches.", "About 12% of shots.", "On 2 754 shots."])
def test_an_untagged_number_in_the_body_fails(tmp_path, body):
    errors = errors_of(tmp_path, tex=MAIN_TEX.replace("On \\val{n} shots.", f"On \\val{{n}} shots. {body}"))
    assert any("is not a \\val" in e for e in errors), errors


@pytest.mark.parametrize("decoy", [
    "on 2026-10-12 at 11:57 UTC in 2027",
    "see \\cref{sec:data} and \\cite{x2018}",
    "% 3.14 in a comment",
    "\\includegraphics[width=0.48\\linewidth]{figures/fig_2.pdf}",
    "\\begin{tabular}{@{}l@{}}\\multicolumn{2}{c}{x}\\cmidrule(lr){2-3}\\end{tabular}",
    "\\vspace{-12pt}\\setlength{\\tabcolsep}{4pt}",
    "frames at $-400$\\,ms",
])
def test_names_layout_dates_comments_and_untagged_phrases_are_not_results(tmp_path, decoy):
    # The preamble's 0.75 is skipped in every case; each decoy is added to the body on its own.
    assert errors_of(tmp_path, tex=MAIN_TEX.replace("The gate \\val{gate}.", f"The gate \\val{{gate}} {decoy}")) == []


def test_an_untagged_phrase_does_not_hide_a_number_glued_before_it(tmp_path):
    errors = errors_of(tmp_path, tex=MAIN_TEX.replace("The gate \\val{gate}.", "The gate \\val{gate}, 1 $-400$\\,ms."))
    assert any("'400'" in e for e in errors), errors


def test_a_heading_title_over_two_lines_or_with_a_short_title_counts(tmp_path):
    tex = MAIN_TEX.replace("\\section{Confirmatory results}", "\\section[Short]{Long title,\nconfirmatory results}")
    assert errors_of(tmp_path, tex=tex) == []
    tex = MAIN_TEX.replace("\\section{Context (development results)}", "\\section{Context\n(development results)}")
    assert errors_of(tmp_path, tex=tex) == []


def test_a_confirmatory_number_outside_a_confirmatory_heading_fails(tmp_path):
    errors = errors_of(tmp_path, tex=MAIN_TEX.replace("The gate \\val{gate}", "The gate \\val{gate} and \\val{n}"))
    assert any("n is a confirmatory result outside" in e for e in errors)


def test_a_confirmatory_number_under_a_development_heading_fails(tmp_path):
    errors = errors_of(tmp_path, tex=MAIN_TEX.replace("Earlier: \\val{dev_h2}.", "Earlier: \\val{dev_h2}, \\val{n}."))
    assert any("n is a confirmatory result under" in e for e in errors)


def test_an_abstract_that_does_not_say_confirmatory_cannot_carry_confirmatory_numbers(tmp_path):
    errors = errors_of(tmp_path, tex=MAIN_TEX.replace("Our confirmatory results:", "Our results:"))
    assert any("h2 is a confirmatory result outside" in e for e in errors)


def test_a_development_number_outside_a_development_heading_fails(tmp_path):
    errors = errors_of(tmp_path, tex=MAIN_TEX.replace("On \\val{n} shots.", "On \\val{n} shots, \\val{dev_h2}."))
    assert any("dev_h2 is a development result outside" in e for e in errors)


def test_an_unknown_val_and_an_unused_claim_fail(tmp_path):
    errors = errors_of(tmp_path, tex=MAIN_TEX.replace("Earlier: \\val{dev_h2}.", "Earlier: \\val{dev_h3}."))
    assert any("\\val{dev_h3} has no readable claim" in e for e in errors)
    assert any("paper claim dev_h2 is not used" in e for e in errors)


def test_a_paper_claim_may_override_only_the_rendering(tmp_path):
    claims = PAPER_CLAIMS.replace('"n" = {}', '"n" = { src = "cf:other.json" }')
    errors = errors_of(tmp_path, paper_claims=claims)
    assert any("may override only fmt and labels, not src" in e for e in errors)


def test_a_paper_claim_must_name_a_report_claim(tmp_path):
    errors = errors_of(tmp_path, paper_claims=PAPER_CLAIMS.replace('from = "gate_pass"', 'from = "gate_ok"'))
    assert any("no report claim 'gate_ok'" in e for e in errors)


def test_a_spaced_number_inside_math_fails(tmp_path):
    errors = errors_of(tmp_path, tex=MAIN_TEX.replace("changes log-loss by \\val{h2}", "changes log-loss by $\\Delta = \\val{h2}$"))
    assert any("h2} has spaces and stands in math mode" in e for e in errors)


def test_main_reports_a_missing_report(tmp_path, capsys):
    paper = write_tree(tmp_path)
    (paper / "claims.toml").write_text(PAPER_CLAIMS.replace('report = "REPORT.md"', 'report = "NOPE.md"'))
    assert main([str(paper), "--root", str(tmp_path)]) == 1
    assert "is not a file" in capsys.readouterr().out


# Figures --------------------------------------------------------------------------------------------------------


def write_results(root: Path, shift: float = 0.0) -> None:
    """Minimal confirmatory result files with every row the figures read."""
    base = root / figures.CONFIRMATORY
    (base / "ladder").mkdir(parents=True, exist_ok=True)
    (base / "probes").mkdir(parents=True, exist_ok=True)
    rows = [{"population": pop, "variant": "recalibrated", "rung": rung, "log_loss": 0.33 + 0.01 * i + shift,
             "log_loss_lo": 0.30 + 0.01 * i, "log_loss_hi": 0.36 + 0.01 * i}
            for pop in ("primary", "combined") for i, (rung, _) in enumerate(figures.RUNGS)]
    rows.append({"population": "primary", "variant": "raw", "rung": "vendor", "log_loss": 0.4, "log_loss_lo": 0.3,
                 "log_loss_hi": 0.5})
    pd.DataFrame(rows).to_csv(base / "ladder/metrics.csv", index=False)
    entry = {"diff": -0.0004, "lo95": -0.0026, "hi95": 0.0018, "lo90": -0.0022, "hi90": 0.0015, "margin": 0.005}
    sets = {pop: {"h2": entry, "frame_plus_vendor": entry} for pop in ("primary", "combined")}
    (base / "ladder/h2.json").write_text(json.dumps({"sets": sets}))
    probes = [{"encoder": enc, "subset": "all", "target": t, "metric": "AUC" if t in ("def_cone_ge1", "def_cone_ge2", "head") else "R2",
               "value": 0.6, "lo": 0.55, "hi": 0.65, "threshold": 0.5}
              for enc, *_ in figures.ENCODERS for t in [*figures.TARGETS, "dist_shuffled"]]
    pd.DataFrame(probes).to_csv(base / "probes/probes.csv", index=False)


def test_figures_are_drawn_from_the_result_files_and_are_byte_stable(tmp_path):
    write_results(tmp_path)
    first = [p.read_bytes() for p in figures.build(tmp_path / "a", tmp_path)]
    second = [p.read_bytes() for p in figures.build(tmp_path / "b", tmp_path)]
    assert first == second and all(b.startswith(b"%PDF") for b in first)
    write_results(tmp_path, shift=0.02)
    third = [p.read_bytes() for p in figures.build(tmp_path / "c", tmp_path)]
    assert third[0] != first[0] and third[1] == first[1]   # only the ladder reads metrics.csv


def test_figures_refuse_missing_rows(tmp_path):
    write_results(tmp_path)
    path = tmp_path / figures.CONFIRMATORY / "ladder/metrics.csv"
    m = pd.read_csv(path)
    m[m.rung != "vendor"].to_csv(path, index=False)
    with pytest.raises(ValueError, match="recalibrated rung rows"):
        figures.build(tmp_path / "out", tmp_path)


def test_figures_refuse_a_probe_row_set_with_the_right_count_but_wrong_rows(tmp_path):
    write_results(tmp_path)
    path = tmp_path / figures.CONFIRMATORY / "probes/probes.csv"
    p = pd.read_csv(path)
    p.loc[(p.encoder == "emb_dinov2") & (p.target == "head"), "target"] = "angle"   # head missing, angle twice
    p.to_csv(path, index=False)
    with pytest.raises(ValueError, match="one row per encoder and target"):
        figures.build(tmp_path / "out", tmp_path)


def test_main_reports_missing_result_rows_without_a_traceback(tmp_path, capsys):
    paper = write_tree(tmp_path)
    write_results(tmp_path)
    path = tmp_path / figures.CONFIRMATORY / "ladder/metrics.csv"
    pd.read_csv(path).query("rung != 'vendor'").to_csv(path, index=False)
    assert main([str(paper), "--root", str(tmp_path)]) == 1
    assert "recalibrated rung rows" in capsys.readouterr().out


def test_stale_figures_fail_the_check(tmp_path):
    paper = write_tree(tmp_path)
    write_results(tmp_path)
    figures.build(paper / "figures", tmp_path)
    assert check(paper, tmp_path) == []
    write_results(tmp_path, shift=0.02)
    assert any("fig_ladder.pdf is not what" in e for e in check(paper, tmp_path))


# The real paper ------------------------------------------------------------------------------------------------


@pytest.mark.skipif(not (EXP / "results/xgv_confirmatory/complete.json").is_file(), reason="needs the confirmatory results")
def test_the_real_paper_passes():
    assert check(PAPER, EXP) == []
