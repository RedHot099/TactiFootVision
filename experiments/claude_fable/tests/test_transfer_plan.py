"""Test seam 2: the leave-one-league-out split plan on a small fixture index."""

import json
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd
import pytest
from xgpoc.transfer.plan import (
    CHAMPIONS_LEAGUE,
    HALF_SETS,
    ROLE_WEIGHTS,
    _round_robin,
    build_plan,
    half_sets,
    lookalike_pairs,
    match_table,
    normalise_club,
    parse_match,
    training_pool,
)
from xgpoc.v13.protocol import digest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

CLUBS = {
    "england_epl": ["Arsenal", "Chelsea", "Burnley", "Watford", "Swansea", "Newcastle Utd"],
    "spain_laliga": ["Barcelona", "Getafe", "Eibar", "Malaga", "Osasuna", "Betis"],
    "italy_serie-a": ["Juventus", "Torino", "Lazio", "Sassuolo", "Cagliari", "Udinese"],
}
LEAGUES = tuple(CLUBS)
SEASONS = ("2015-2016", "2016-2017")
UCL = [("Chelsea", "Ajax"), ("Barcelona", "Celtic"), ("Juventus", "Basel"), ("Ajax", "Celtic"),
       ("Basel", "Benfica"), ("Arsenal", "Barcelona"), ("Benfica", "Ajax")]
N = 20  # the fixture's pools hold 33 matches
ROOT = Path(__file__).resolve().parents[1]
REAL_PLAN = ROOT / "research" / "xgc_transfer_plan"


def game(league, season, k, home, away):
    return f"{league}/{season}/{season[:4]}-{1 + k // 28:02d}-{1 + k % 28:02d} - 18-00 {home} 1 - 0 {away}"


def fixture_index(ucl=UCL):
    games = []
    for league, clubs in CLUBS.items():
        pairs = list(combinations(clubs, 2))
        for s, season in enumerate(SEASONS):  # 7 matches per season: odd, so half-sets cannot split evenly
            games += [game(league, season, k, *pairs[7 * s + k]) for k in range(7)]
    games += [game(CHAMPIONS_LEAGUE, "2016-2017", k, home, away) for k, (home, away) in enumerate(ucl)]
    return pd.DataFrame({"shot_id": [f"{g}#{i}" for g in games for i in range(3)],
                         "game": [g for g in games for _ in range(3)],
                         "goal": [i == 0 for _ in games for i in range(3)]})


def plan_clubs(plan, mask):
    matches = match_table(fixture_index()).set_index("game")
    games = plan.game[mask]
    return set(matches.home[games]) | set(matches.away[games])


@pytest.fixture(scope="module")
def plan():
    return build_plan(fixture_index(), n_matches=N, leagues=LEAGUES)


def test_match_identifier_parses_with_the_anchored_pattern():
    assert parse_match("england_epl/2014-2015/2015-02-21 - 18-00 Chelsea 1 - 1 Burnley") == (
        "england_epl", "2014-2015", "Chelsea", "Burnley")
    assert parse_match("germany_bundesliga/2016-2017/2016-09-10 - 16-30 1. FSV Mainz 05 4 - 4 Hoffenheim")[2:] == (
        "1. FSV Mainz 05", "Hoffenheim")
    assert parse_match("germany_bundesliga/2016-2017/2016-09-10 - 16-30 Hoffenheim 4 - 4 1. FSV Mainz 05")[2:] == (
        "Hoffenheim", "1. FSV Mainz 05")
    with pytest.raises(ValueError, match="identifier"):
        parse_match("england_epl/Chelsea 1 - 1 Burnley")


def test_club_names_are_folded_then_aliased():
    assert normalise_club("  Atlético   Madrid ") == "Atletico Madrid"
    aliases = {"Newcastle United": "Newcastle Utd"}
    assert normalise_club("Newcastle  United", aliases) == "Newcastle Utd"
    assert normalise_club("Newcastle Utd", aliases) == "Newcastle Utd"


def test_spelling_variant_must_be_aliased_and_then_keeps_its_ucl_match_out():
    index = fixture_index(UCL + [("Newcastle United", "Benfica")])
    variant = game(CHAMPIONS_LEAGUE, "2016-2017", len(UCL), "Newcastle United", "Benfica")
    with pytest.raises(ValueError, match="may be one club"):
        build_plan(index, n_matches=N, leagues=LEAGUES)
    aliases = {"Newcastle United": "Newcastle Utd"}
    assert variant not in training_pool(match_table(index, aliases), "england_epl")
    plan = build_plan(index, n_matches=N, leagues=LEAGUES, aliases=aliases)
    assert variant not in set(plan.game[plan.league == "england_epl"])
    # the failure the alias prevents: read as two clubs, the variant's match joins the pool
    assert variant in training_pool(match_table(index), "england_epl")


def test_training_clubs_never_meet_the_held_out_clubs(plan):
    for league in LEAGUES:
        held_out = set(CLUBS[league])
        training = (plan.league == league) & (plan.role != "test")
        assert not plan_clubs(plan, training & (plan.arm == "without")) & held_out
        assert not plan_clubs(plan, training & (plan.source == "pool")) & held_out
        # in "with", the held-out clubs come only from the league's other half-set
        assert set(plan.game[training & (plan.source == "league")]) <= set(plan.game[plan.role == "test"])


def test_champions_league_matches_with_held_out_clubs_are_left_out(plan):
    matches = match_table(fixture_index())
    ucl = matches[matches.league == CHAMPIONS_LEAGUE]
    for league in LEAGUES:
        clubs = set(CLUBS[league])
        involved = set(ucl.game[ucl.home.isin(clubs) | ucl.away.isin(clubs)])
        assert involved, "fixture must have a Champions League match per league"
        assert not involved & set(plan.game[plan.league == league])
        assert not involved & set(training_pool(matches, league))
        assert set(ucl.game) - involved <= set(training_pool(matches, league))


def test_both_arms_have_the_same_matches_per_role(plan):
    counts = plan.groupby(["league", "half_set", "role", "arm"]).size().unstack("arm")
    assert (counts["with"] == counts["without"]).all()
    training = plan[plan.role != "test"].groupby(["league", "half_set", "arm"]).size()
    assert (training == N).all()
    assert set(plan.role) == {"train", "select", "calibration", "test"}
    assert not plan.duplicated(["league", "half_set", "arm", "game"]).any()


def test_with_arm_holds_the_other_half_set_and_a_prefix_of_the_without_pool(plan):
    for (league, h), rows in plan.groupby(["league", "half_set"]):
        other = set(rows.game[rows.role == "test"])
        test_other = set(plan.game[(plan.league == league) & (plan.half_set != h) & (plan.role == "test")])
        with_arm, without = rows[rows.arm == "with"], rows[rows.arm == "without"]
        assert set(with_arm.game[(with_arm.source == "league") & (with_arm.role != "test")]) == test_other
        assert not other & test_other
        pool_roles = with_arm[with_arm.source == "pool"].set_index("game").role
        assert pool_roles.equals(without.set_index("game").role.loc[pool_roles.index])


def test_every_held_out_match_is_tested_exactly_once(plan):
    matches = match_table(fixture_index())
    for league in LEAGUES:
        for arm in ("without", "with"):
            tested = plan.game[(plan.league == league) & (plan.arm == arm) & (plan.role == "test")]
            assert tested.is_unique
            assert set(tested) == set(matches.game[matches.league == league])


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("league", LEAGUES)
def test_half_sets_are_balanced_within_every_season(league, seed):
    matches = match_table(fixture_index())
    halves = half_sets(matches, league, seed=seed)
    assert set(halves.index) == set(matches.game[matches.league == league])
    season = matches.set_index("game").season[halves.index]
    counts = pd.crosstab(season, halves)
    assert list(counts.columns) == list(HALF_SETS)
    assert (counts.A - counts.B).abs().max() <= 1
    assert abs((halves == "A").sum() - (halves == "B").sum()) <= 1


def test_plan_is_identical_for_a_seed_and_changes_with_it(plan):
    assert plan.equals(build_plan(fixture_index().sample(frac=1, random_state=3), n_matches=N, leagues=LEAGUES))
    assert not plan.equals(build_plan(fixture_index(), seed=1, n_matches=N, leagues=LEAGUES))


def test_lookalike_guard_catches_abbreviations_but_not_nicknames():
    pairs = lookalike_pairs(["Man Utd", "Manchester United", "PSG", "Paris Saint-Germain", "Spurs",
                             "Tottenham Hotspur", "Bayer Leverkusen", "Bayern Munich", "Dyn. Kiev", "D. Zagreb"])
    assert pairs == [("Man Utd", "Manchester United"), ("PSG", "Paris Saint-Germain")]  # Spurs: documented limit


def test_round_robin_skips_a_zero_quota():
    assert _round_robin(3, {"train": 0, "select": 3, "calibration": 0}) == ["select"] * 3
    assert _round_robin(20, ROLE_WEIGHTS).count("calibration") == ROLE_WEIGHTS["calibration"]


def test_odd_half_set_with_a_zero_quota_keeps_roles_equal():
    # N = 13 with a 7-match other half-set: the 6 pool matches already hold both select
    # slots of the 13-match sequence, so the half-set's select quota is zero
    plan = build_plan(fixture_index(), n_matches=13, leagues=LEAGUES)
    counts = plan.groupby(["league", "half_set", "role", "arm"]).size().unstack("arm")
    assert (counts["with"] == counts["without"]).all()
    league_rows = plan[(plan.arm == "with") & (plan.role != "test") & (plan.source == "league")]
    assert league_rows.groupby(["league", "half_set"]).size().eq(7).all()
    assert not (league_rows.role == "select").any()


def test_other_half_set_larger_than_the_arm_is_refused():
    with pytest.raises(ValueError, match="exceeds"):
        build_plan(fixture_index(), n_matches=6, leagues=LEAGUES)


@pytest.mark.skipif(not (REAL_PLAN / "manifest.json").exists(), reason="real plan not written")
def test_committed_real_plan_matches_its_manifest_and_current_code():
    manifest = json.loads((REAL_PLAN / "manifest.json").read_text())
    assert digest(REAL_PLAN / "plan.csv") == manifest["plan_sha256"]
    assert {p: digest(ROOT / p) for p in manifest["sources"]} == manifest["sources"]


def test_too_small_pool_is_refused():
    with pytest.raises(ValueError, match="fewer than"):
        build_plan(fixture_index(), n_matches=34, leagues=LEAGUES)


def test_a_written_plan_is_not_replaced_by_a_different_one(tmp_path):
    from make_transfer_plan import write_plan_files

    write_plan_files(tmp_path, {"plan.csv": "a"}, {"plan_sha256": "1", "sources": {"x": "old"}})
    write_plan_files(tmp_path, {"plan.csv": "a"}, {"plan_sha256": "1", "sources": {"x": "new"}})
    assert json.loads((tmp_path / "manifest.json").read_text())["sources"] == {"x": "new"}
    with pytest.raises(SystemExit, match="--replace"):
        write_plan_files(tmp_path, {"plan.csv": "b"}, {"plan_sha256": "2"})
    assert (tmp_path / "plan.csv").read_text() == "a"
    write_plan_files(tmp_path, {"plan.csv": "b"}, {"plan_sha256": "2"}, replace=True)
    assert (tmp_path / "plan.csv").read_text() == "b"
