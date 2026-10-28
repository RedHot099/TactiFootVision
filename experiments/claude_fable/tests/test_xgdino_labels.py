"""xgpoc.dino.labels on synthetic data (ticket 10)."""

import numpy as np
import pandas as pd
import pytest
from xgpoc.dino import labels as lb
from xgpoc.stack import inputs as si


def comments(*rows):
    return [lb.Comment(h, p, t, team) for h, p, t, team in rows]


def shots(*rows):
    return pd.DataFrame(rows, columns=["shot_id", "half", "position_ms"]).set_index("shot_id")


def test_text_rules():
    assert lb.comment_label("his header goes wide") == "header"
    assert lb.comment_label("he volleys it over the bar") == "foot"
    assert lb.comment_label("a fine shot from outside the box") == "unknown"      # no header words is not foot
    assert lb.comment_label("his shot is headed clear by the defender") == "unknown"
    assert lb.is_shot_like("A shot from distance") and not lb.is_shot_like("The referee blows his whistle")
    assert not lb.is_shot_like("The ball is headed away by the defender")


def test_match_window_half_and_one_to_one():
    s = shots(("a", 1, 100_000), ("b", 2, 100_000))
    c = comments((1, 112_000, "his header goes wide", None), (2, 200_000, "his header goes wide", None))
    out = lb.match_game(s, {"a": "home", "b": "home"}, c)
    assert out.label["a"] == "header" and out.reason["a"] == "matched"
    assert out.reason["b"] == "no_comment"                                        # 100 s after: outside the window
    # one comment, two candidate shots: both rejected
    s2 = shots(("a", 1, 100_000), ("b", 1, 120_000))
    out2 = lb.match_game(s2, {"a": None, "b": None}, comments((1, 130_000, "his header goes wide", None)))
    assert set(out2.reason) == {"ambiguous"} and set(out2.label) == {"unknown"}


def test_team_and_conflicting_comments():
    s = shots(("a", 1, 100_000))
    away = comments((1, 110_000, "his header goes wide", "away"))
    assert lb.match_game(s, {"a": "home"}, away).reason["a"] == "no_comment"
    both = comments((1, 105_000, "his header goes wide", None), (1, 110_000, "he volleys it wide", None))
    assert lb.match_game(s, {"a": "home"}, both).reason["a"] == "conflict"


def test_outcome_cannot_change_the_match():
    s = shots(("a", 1, 100_000))
    base = lb.match_game(s, {"a": "home"}, comments((1, 110_000, "his header goes wide", None)))
    goal = lb.match_game(s, {"a": "home"}, comments((1, 110_000, "Goal! His header flies in", None)))
    assert base.label["a"] == goal.label["a"] == "header"
    assert "goal" not in {c.lower() for c in base.columns}


def test_labels_barred_from_designs():
    for name in ("header_caption", "header_gpt", "x_caption"):
        with pytest.raises(ValueError):
            lb.assert_not_design(["dist@0", name])
    lb.assert_not_design(si.E3_COLUMNS)
    lb.assert_not_design(si.S_FLAGS)
    assert lb.SOURCES == ("caption", "gpt")


def test_sample_is_stratified_seeded_and_weighted():
    rng = np.random.default_rng(0)
    n = 6000
    t = pd.DataFrame({"goal": rng.random(n) < 0.12, "caption_label": rng.choice(lb.LABELS, n, p=[0.08, 0.01, 0.91]),
                      "dist": rng.uniform(2, 40, n)}, index=[f"s{i}" for i in range(n)])
    a, b = lb.select_sample(t), lb.select_sample(t)
    assert a.equals(b) and len(a) == lb.SAMPLE_SIZE and a.index.is_unique
    sizes = t.assign(k=t.goal.astype(int).astype(str) + "|" + t.caption_label + "|" + lb.distance_band(t.dist)).k.value_counts()
    assert (a.stratum.value_counts() <= sizes.loc[a.stratum.value_counts().index]).all()
    assert (a.groupby("stratum").weight.sum().round() == sizes.loc[a.stratum.unique()].reindex(a.groupby("stratum").weight.sum().index)).all()
    assert lb.select_sample(t, seed=1).index.tolist() != a.index.tolist()


def test_allocation_water_filling():
    assert lb.sample_allocation({"a": 5, "b": 100, "c": 100}, 50) == {"a": 5, "b": 23, "c": 22}
    assert sum(lb.sample_allocation({"a": 3, "b": 4}, 100).values()) == 7


def test_anon_ids_and_blanking(tmp_path):
    ids = lb.anon_ids(["x1", "x2", "x3"])
    assert ids.is_unique and ids.str.match(r"^s\d{5}$").all() and not ids.index.isin(ids.to_list()).any()
    image = np.full((100, 200, 3), 200, np.uint8)
    out = lb.blank_scoreboard(image)
    assert (out[:15] == 0).all() and (out[15:] == 200).all() and (image == 200).all()


def test_reserved_names_refused():
    lb.check_reserved("england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley")
    for bad in ("a/CON/b", "a/b./c", "a/b:c"):
        with pytest.raises(ValueError):
            lb.check_reserved(bad)


def test_gpt_answers_join_and_validation(tmp_path):
    (tmp_path / "part1.csv").write_text("s00001,head\ns00002,foot\n")
    (tmp_path / "part2.csv").write_text("s00003,unclear\n")
    answers = lb.read_answers(sorted(tmp_path.glob("part*.csv")))
    key = pd.DataFrame({"anon_id": ["s00001", "s00002", "s00003"], "shot_id": ["a", "b", "c"]})
    sample = pd.DataFrame(index=pd.Index(["a", "b", "c"], name="shot_id"))
    g = lb.gpt_labels(answers, key, sample)
    assert g.header_gpt["a"] == 1 and g.header_gpt["b"] == 0 and np.isnan(g.header_gpt["c"]) and set(g.source) == {"gpt"}
    (tmp_path / "bad.csv").write_text("s00001,head\ns00001,foot\n")
    with pytest.raises(ValueError):
        lb.read_answers([tmp_path / "bad.csv"])
    (tmp_path / "bad2.csv").write_text("s00001,arm\n")
    with pytest.raises(ValueError):
        lb.read_answers([tmp_path / "bad2.csv"])
    with pytest.raises(ValueError):
        lb.gpt_labels(answers.iloc[:2], key, sample)


def test_agreement_weighted_by_outcome():
    d = pd.DataFrame({
        "outcome": [0, 0, 0, 0, 0, 1, 1, 1],
        "caption_label": ["header", "header", "foot", "unknown", "unknown", "header", "unknown", "unknown"],
        "gpt_answer": ["head", "foot", "foot", "head", "unclear", "head", "foot", "head"],
        "weight": [1, 3, 1, 2, 5, 2, 1, 1]})
    r = lb.agreement(d)
    ng, g = r["no_goal"], r["goal"]
    assert ng["caption_header"]["precision"] == pytest.approx(1 / 4) and ng["caption_header"]["precision_unweighted"] == 0.5
    assert ng["gpt_head"]["recall"] == pytest.approx(1 / 3) and ng["gpt_head"]["n"] == 2        # unclear dropped
    assert ng["agreement_on_caption_labelled"]["weighted"] == pytest.approx(2 / 5)
    assert ng["caption_unknown"]["gpt_head_n"] == 1 and ng["caption_unknown"]["gpt_foot_n"] == 0
    assert g["share_of_gpt_head_with_caption_header"]["weighted"] == pytest.approx(2 / 3)
    assert g["caption_unknown"]["gpt_head_share"] == pytest.approx(1 / 2)
    assert r["answers_by_outcome"]["no_goal"]["unclear"] == 1 and "noisy" in r["note"]
