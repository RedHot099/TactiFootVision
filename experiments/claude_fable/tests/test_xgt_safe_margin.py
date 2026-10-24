"""Safe-margin check of o_max = -600 ms (xg-temporal 02): the Clopper-Pearson pass rule, the worst-case reading of
the two AI runs, agreement statistics, the seeded sample and the answer-file checks."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import xgt_safe_margin as sm  # noqa: E402


def answers(rows: dict[str, tuple[str, str, str]]) -> pd.DataFrame:
    """Answer frame indexed by sheet id; tuple order -800, -600, -400."""
    return pd.DataFrame([{"sheet_id": k, "struck_at_m800": v[0], "struck_at_m600": v[1], "struck_at_m400": v[2]}
                         for k, v in rows.items()]).set_index("sheet_id")


@pytest.mark.parametrize(("violations", "upper", "ok"), [(0, 0.0247, True), (1, 0.0388, True), (2, 0.0516, False)])
def test_clopper_pearson_rule_at_120(violations, upper, ok):
    assert sm.cp_upper(violations, 120) == pytest.approx(upper, abs=5e-4)
    assert sm.passes(violations, 120) is ok


def test_clopper_pearson_edges():
    assert sm.cp_upper(0, 28) == pytest.approx(0.102, abs=1e-3)  # the 28 existing contacts are not enough
    assert sm.cp_upper(120, 120) == 1.0
    assert sm.cp_upper(0, 120) == pytest.approx(1 - 0.05 ** (1 / 120))
    with pytest.raises(ValueError):
        sm.cp_upper(3, 2)


@pytest.mark.parametrize(("m800", "m600", "m400", "struck_600"), [
    ("no", "no", "no", False),
    ("no", "no", "yes", False),          # struck between -600 and -400: no violation
    ("unknown", "no", "yes", False),     # `no` at -600 settles -800 too
    ("no", "unknown", "no", True),       # unknown = earliest strike
    ("no", "yes", "yes", True),
    ("yes", "no", "no", True),           # inconsistent: the earlier yes wins
])
def test_worst_case_per_run(m800, m600, m400, struck_600):
    assert sm.struck_worst({-800: m800, -600: m600, -400: m400}, -600) is struck_600


def test_inconsistent_answers_are_flagged():
    assert sm.inconsistent({-800: "yes", -600: "no", -400: "yes"})
    assert sm.inconsistent({-800: "no", -600: "yes", -400: "no"})
    assert not sm.inconsistent({-800: "unknown", -600: "no", -400: "yes"})


def test_disagreement_and_unknown_count_as_violations():
    a = answers({"s1": ("no", "no", "no"), "s2": ("no", "no", "yes"), "s3": ("no", "no", "no"),
                 "s4": ("no", "unknown", "yes"), "s5": ("no", "no", "no")})
    b = answers({"s1": ("no", "no", "no"), "s2": ("no", "yes", "yes"), "s3": ("no", "no", "no"),
                 "s4": ("no", "no", "yes"), "s5": ("no", "no", "unknown")})
    table, result = sm.evaluate_runs(a, b)
    assert table.set_index("sheet_id").violation_m600.to_dict() == {
        "s1": False, "s2": True, "s3": False, "s4": True, "s5": False}
    assert result["violations"] == 2
    assert result["per_offset"]["-600"]["disagreements"] == 2
    assert result["per_offset"]["-400"]["worst_case_struck"] == 3  # s2, s4, and s5's unknown
    assert result["per_offset"]["-800"]["worst_case_struck"] == 0


def test_verdict_follows_the_bound():
    clean = {f"s{i:03d}": ("no", "no", "no") for i in range(120)}
    one, two = dict(clean), dict(clean)
    one["s000"] = ("no", "yes", "yes")
    two["s000"] = two["s001"] = ("no", "unknown", "yes")
    assert sm.evaluate_runs(answers(clean), answers(clean))[1]["verdict"] == "pass"
    assert sm.evaluate_runs(answers(one), answers(clean))[1]["verdict"] == "pass"
    stop = sm.evaluate_runs(answers(two), answers(clean))[1]
    assert stop["verdict"] == "stop" and stop["violations"] == 2 and "owner" in stop["next"]


def test_cohen_kappa():
    assert sm.cohen_kappa(["yes", "no", "no", "unknown"], ["yes", "no", "no", "unknown"]) == pytest.approx(1.0)
    # 2x2 textbook case: po = 0.7, pe = 0.5*0.6 + 0.5*0.4 = 0.5 -> kappa 0.4
    a = ["yes"] * 5 + ["no"] * 5
    b = ["yes"] * 4 + ["no"] + ["yes"] * 2 + ["no"] * 3
    assert sm.cohen_kappa(a, b) == pytest.approx(0.4)
    assert sm.cohen_kappa(["no"] * 4, ["no"] * 4) is None


def test_sample_is_seeded_and_batched():
    ids = [f"shot{i:04d}" for i in range(500)]
    games = [f"g{i % 40}" for i in range(500)]
    s = sm.draw_sample(ids, games)
    again = sm.draw_sample(ids[::-1], games[::-1])  # input order does not matter
    assert s.equals(again)
    assert len(s) == 120 and s.shot_id.is_unique
    assert s.batch.value_counts().to_dict() == {1: 30, 2: 30, 3: 30, 4: 30}
    assert s.sheet_id.tolist() == [f"s{i:03d}" for i in range(1, 121)]
    assert not s.equals(sm.draw_sample(ids, games, seed=1))


def test_answer_file_checks(tmp_path):
    good = tmp_path / "a.csv"
    good.write_text("sheet_id,struck_at_m800,struck_at_m600,struck_at_m400,note\ns1,no,No,yes,cross first\n"
                    "s2,no,no,unknown,\n")
    read = sm.read_answers([good], ["s1", "s2"])
    assert read.loc["s1", "struck_at_m600"] == "no"
    bad = tmp_path / "b.csv"
    bad.write_text("sheet_id,struck_at_m800,struck_at_m600,struck_at_m400\ns1,no,maybe,yes\ns2,no,no,no\n")
    with pytest.raises(SystemExit, match="answers outside"):
        sm.read_answers([bad], ["s1", "s2"])
    with pytest.raises(SystemExit, match="cover 2 of 3"):
        sm.read_answers([good], ["s1", "s2", "s3"])


def test_no_frame_after_the_anchor_is_decoded():
    with pytest.raises(ValueError, match="<= 0"):
        sm.decode_until_anchor(Path("unused.mp4"), (-600, 0, 40))
    assert max(sm.SHEET_OFFSETS) == 0 and set(sm.QUESTIONS) <= set(sm.SHEET_OFFSETS)
