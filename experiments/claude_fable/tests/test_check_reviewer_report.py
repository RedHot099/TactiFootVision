"""The reviewer-report number check (xg-credibility 14) on a small tree whose answers are known."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_reviewer_report import (  # noqa: E402
    EXP,
    ClaimError,
    check,
    load_claims,
    main,
    render,
)

CLAIMS = """
[report]
dev_label = "wyniki rozwojowe"

[folders]
r = { path = "res", dev = true }
p = { path = "proto", dev = false }

[reply]
claims = ["gap", "share"]

[claims]
"gap" = { src = "r:gaps.csv", row = { group = "pooled" }, cols = ["gap", "lo", "hi"], fmt = "{:+.4f} [{:+.4f}; {:+.4f}]" }
"share" = { src = "r:gaps.csv", row = { group = "pooled" }, cols = ["rel"], scale = 100, fmt = "{:.1f}%" }
"neg" = { src = "r:gaps.csv", row = { group = "league" }, cols = ["gap"], fmt = "{:.3f}" }
"label" = { src = "r:gaps.csv", row = { group = "league" }, cols = ["label"], fmt = "{}" }
"mean" = { src = "r:gaps.csv", row = { group = ["pooled", "league"] }, agg = "mean", cols = ["gap"], fmt = "{:.4f}" }
"n" = { src = "r:summary.json", keys = ["population.shots"], fmt = "{:,d}" }
"verdict" = { src = "r:summary.json", keys = ["verdict"], labels = { inconclusive = "niekonkluzywny" }, fmt = "{}" }
"gate" = { src = "r:summary.json", keys = ["gates.1"], labels = { true = "przechodzi", false = "nie przechodzi" }, fmt = "{}" }
"threshold" = { src = "p:protocol.txt", quote = "próg 0,10", values = [0.10], fmt = "{:.2f}" }
"""

REPORT = """# Raport

Wstęp bez liczb dziesiętnych, wersja v1.2, `0.123` w kodzie, 2 000 losowań, ticket 03.

## Protokół

Próg przecieku 0.10<!--n:threshold-->.

## 2.1 Transfer (wyniki rozwojowe)

Luka +0.0041 [+0.0020; +0.0061]<!--n:gap-->, czyli 1.2%<!--n:share-->. Liga: −0.012<!--n:neg-->, t50 > +400<!--n:label-->.
Średnia −0.0039<!--n:mean-->, strzały 9 307<!--n:n-->. Werdykt: niekonkluzywny<!--n:verdict-->, bramka nie przechodzi<!--n:gate-->.
Zob. sekcja 2.1 i [wykres](../res/plot_2.5.png).

## Źródła

| Plik | SHA-256 |
|---|---|
{rows}
"""

REPLY = """Dzień dobry,
luka wynosi +0.0041 [+0.0020; +0.0061] log-loss (1.2%). Wszystkie liczby to wyniki rozwojowe.
"""


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_rows(root: Path, names: list[str]) -> str:
    return "\n".join(f"| `{n}` | `{sha(root / n)}` |" for n in names)


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A root with a CSV, a JSON, a protocol, a completion record, claims, report and reply."""
    (tmp_path / "res").mkdir()
    (tmp_path / "proto").mkdir()
    (tmp_path / "res/gaps.csv").write_text(
        "group,gap,lo,hi,rel,label\npooled,0.0041,0.002,0.0061,0.0118,x\nleague,-0.0119,-0.02,0.01,0.01,> +400\n"
    )
    summary = {"population": {"shots": 9307}, "verdict": "inconclusive", "gates": [True, False]}
    (tmp_path / "res/summary.json").write_text(json.dumps(summary))
    (tmp_path / "proto/protocol.txt").write_text("Reguła: próg 0,10 na dolnej granicy.\n")
    files = {n: sha(tmp_path / "res" / n) for n in ("gaps.csv", "summary.json")}
    (tmp_path / "res/complete.json").write_text(json.dumps({"files": files}))
    out = tmp_path / "out"
    out.mkdir()
    (out / "claims.toml").write_text(CLAIMS)
    write_report(tmp_path, REPORT)
    (out / "reply.txt").write_text(REPLY)
    return tmp_path


def write_report(root: Path, template: str, names=("res/gaps.csv", "res/summary.json", "proto/protocol.txt")) -> Path:
    path = root / "out/REPORT.draft.txt"
    path.write_text(template.replace("{rows}", source_rows(root, list(names))))
    return path


def run(root: Path, reply: bool = True) -> list[str]:
    out = root / "out"
    return check(out / "REPORT.draft.txt", out / "claims.toml", root, out / "reply.txt" if reply else None)


def test_render_uses_the_report_number_style():
    assert render("{:+.4f} [{:+.4f}; {:+.4f}]", [0.0041, -0.002, 0.0061]) == "+0.0041 [−0.0020; +0.0061]"
    assert render("{:,d}", [9307.0]) == "9 307"
    assert render("{:.1f}–{:.1f}", [112.55, 275.25]) == "112.5–275.2"
    assert render("{:.12}", ["23c8bd25934a1fd7"]) == "23c8bd25934a"


@pytest.mark.parametrize(
    ("fmt", "values"),
    [("{:.2f} {:.2f}", [1.0]), ("{0:.2f}", [1.0]), ("{:d}", [2.5])],
)
def test_render_refuses_bad_formats(fmt, values):
    with pytest.raises(ClaimError):
        render(fmt, values)


def test_a_report_that_matches_its_sources_passes(tree):
    assert run(tree) == []


def test_the_report_alone_passes_without_flagging_reply_claims(tree):
    assert run(tree, reply=False) == []


def test_every_locator_reads_its_value(tree):
    claims, _, errors = load_claims(tree / "out/claims.toml", tree)
    assert errors == []
    texts = {cid: c.text for cid, c in claims.items()}
    assert texts == {
        "gap": "+0.0041 [+0.0020; +0.0061]",
        "share": "1.2%",
        "neg": "−0.012",
        "label": "> +400",
        "mean": "−0.0039",
        "n": "9 307",
        "verdict": "niekonkluzywny",
        "gate": "nie przechodzi",
        "threshold": "0.10",
    }
    assert claims["gap"].dev and not claims["threshold"].dev


def test_a_changed_number_fails(tree):
    write_report(tree, REPORT.replace("+0.0041 [", "+0.0042 ["))
    errors = run(tree)
    assert any("gap expects '+0.0041 [+0.0020; +0.0061]'" in e for e in errors)


def test_a_dropped_or_added_sign_fails(tree):
    write_report(tree, REPORT.replace("−0.012<!--n:neg-->", "0.012<!--n:neg-->"))
    assert any("neg expects" in e for e in run(tree))
    write_report(tree, REPORT.replace("1.2%<!--n:share-->", "−1.2%<!--n:share-->"))
    assert any("share expects" in e for e in run(tree))


def test_an_untagged_decimal_or_percentage_fails(tree):
    write_report(tree, REPORT.replace("Wstęp bez", "Wstęp 0.5 i 7% bez"))
    errors = run(tree)
    assert any("'0.5' is not tied" in e for e in errors)
    assert any("'7%' is not tied" in e for e in errors)


def test_an_unknown_tag_fails(tree):
    write_report(tree, REPORT.replace("<!--n:gap-->", "<!--n:nope-->"))
    errors = run(tree)
    assert any("unknown or unreadable claim 'nope'" in e for e in errors)


def test_a_development_result_outside_the_labelled_section_fails(tree):
    write_report(tree, REPORT.replace("## Protokół\n", "## Protokół\n\nStrzały 9 307<!--n:n-->.\n"))
    assert any("n is a development result outside" in e for e in run(tree))


def test_an_unused_claim_fails(tree):
    write_report(tree, REPORT.replace(" Werdykt: niekonkluzywny<!--n:verdict-->,", ""))
    assert "claim verdict is not used" in run(tree)


def test_a_changed_source_file_fails_against_the_table(tree):
    write_report(tree, REPORT)
    path = tree / "proto/protocol.txt"
    path.write_text(path.read_text() + "dopisek\n")
    errors = run(tree)
    assert any("source proto/protocol.txt has sha256" in e for e in errors)


def test_an_unlisted_source_fails(tree):
    write_report(tree, REPORT, names=("res/gaps.csv", "res/summary.json"))
    assert "source table does not list proto/protocol.txt" in run(tree)


def test_a_source_that_disagrees_with_its_completion_record_fails(tree):
    record = tree / "res/complete.json"
    record.write_text(json.dumps({"files": {"gaps.csv": "0" * 64}}))
    write_report(tree, REPORT)
    assert any("gaps.csv differs from its completion record" in e for e in run(tree))


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ('src = "r:gaps.csv", row = { group = "none" }, cols = ["gap"]', "no row matches"),
        ('src = "r:gaps.csv", cols = ["gap"]', "2 rows match"),
        ('src = "r:gaps.csv", row = { group = "pooled" }, cols = ["missing"]', "no column 'missing'"),
        ('src = "r:summary.json", keys = ["population.matches"]', "no 'population.matches'"),
        ('src = "p:protocol.txt", quote = "próg 0,20", values = [0.2]', "does not contain"),
        ('src = "r:summary.json", keys = ["verdict"], labels = { leakage = "przeciek" }', "no label"),
        ('src = "r:gaps.csv", keys = ["x"], cols = ["gap"]', "exactly one of"),
        ('src = "q:gaps.csv", cols = ["gap"]', "unknown folder"),
        ('src = "r:absent.csv", cols = ["gap"]', "does not exist"),
        ('cols = ["gap"]', "missing src"),
        ('src = "p:protocol.txt", quote = "próg 0,10", values = [0.2]', "not a number of the quote"),
        ('src = "r:gaps.csv", row = { group = ["pooled", "nope"] }, agg = "mean", cols = ["gap"]', "no row with group in ['nope']"),
        ('src = "r:gaps.csv", row = { group = "pooled" }, cols = ["gap"], scael = 100', "unknown keys scael"),
    ],
)
def test_unreadable_claims_are_reported(tree, spec, message):
    path = tree / "out/claims.toml"
    path.write_text(CLAIMS + f'"bad" = {{ {spec}, fmt = "{{}}" }}\n')
    _, _, errors = load_claims(path, tree)
    assert len(errors) == 1 and message in errors[0], errors


def git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True)


def test_git_claim_hashes_the_file_as_committed(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "t@example.com")
    git(tmp_path, "config", "user.name", "t")
    (tmp_path / "proto").mkdir()
    path = tmp_path / "proto/protocol.txt"
    path.write_bytes(b"wersja 1\n")
    git(tmp_path, "add", "proto/protocol.txt")
    git(tmp_path, "commit", "-q", "-m", "v1")
    path.write_bytes(b"wersja 2\n")
    claims = tmp_path / "claims.toml"
    claims.write_text(
        '[claims]\n"ok" = { src = "proto/protocol.txt", git = "HEAD", fmt = "{}" }\n'
        '"bad" = { src = "proto/protocol.txt", git = "no-such-rev", fmt = "{}" }\n'
    )
    loaded, _, errors = load_claims(claims, tmp_path)
    assert loaded["ok"].text == hashlib.sha256(b"wersja 1\n").hexdigest()
    assert len(errors) == 1 and "git show no-such-rev" in errors[0]


def test_reply_numbers_must_come_from_listed_claims(tree):
    write_report(tree, REPORT)
    reply = tree / "out/reply.txt"
    reply.write_text(REPLY.replace("(1.2%)", "(1.2%, czyli 0.004)"))
    assert any("reply:2: number '0.004' is not tied" in e for e in run(tree))
    reply.write_text(REPLY.replace("+0.0041 [", "+0.0042 ["))
    assert any("reply: gap expects" in e for e in run(tree))
    reply.write_text(REPLY.replace("Wszystkie liczby to wyniki rozwojowe.", ""))
    assert any("never says 'wyniki rozwojowe'" in e for e in run(tree))


def test_reply_claims_are_bound_to_their_order(tree):
    reply = tree / "out/reply.txt"
    reply.write_text("Najpierw 1.2%, potem +0.0041 [+0.0020; +0.0061]. Wszystkie liczby to wyniki rozwojowe.\n")
    errors = run(tree)
    assert any("reply: share expects '1.2%' after position" in e for e in errors)
    assert any("'1.2%' is not tied" in e for e in errors)


def test_main_finds_claims_next_to_the_report_and_sets_the_exit_code(tree, capsys):
    report = write_report(tree, REPORT)
    reply = tree / "out/reply.txt"
    assert main([str(report), "--reply", str(reply), "--root", str(tree)]) == 0
    assert "OK: 9 tagged numbers" in capsys.readouterr().out
    reply.write_text(REPLY + "Dodatkowo 3.5%.\n")
    assert main([str(report), "--reply", str(reply), "--root", str(tree)]) == 1


REAL = EXP / "results_reports/reviewer_response_2026-10"


@pytest.mark.skipif(
    not (EXP / "results/xgc_g_combo/gate.json").is_file(), reason="git-ignored result folders are not present"
)
def test_the_reviewer_report_matches_its_sources():
    report = next((p for p in (REAL / "REPORT.md", REAL / "REPORT.draft.txt") if p.is_file()), None)
    reply = next((p for p in (REAL / "reply.md", REAL / "reply.draft.txt") if p.is_file()), None)
    if report is None or reply is None:
        pytest.skip("reviewer report or reply not written yet")
    assert check(report, REAL / "claims.toml", EXP, reply) == []
