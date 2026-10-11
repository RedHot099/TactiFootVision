"""Pairing of SoccerNet with StatsBomb (xg-vendor ticket 02, test seam 1) on small fixtures with known answers.

The fixture world has one match, Alpha 2 - 1 Beta (Premier League, 2015-08-16), with a planted clock offset per half:
SoccerNet time = StatsBomb time + OFFSET[half] (+ a few ms of jitter). Every test adds the shots it needs.
"""

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import vendor_pair as script  # noqa: E402
from xgpoc.data.soccernet_labels import parse_labels  # noqa: E402
from xgpoc.paths import INDEX, LABELS_ROOT, STATSBOMB  # noqa: E402
from xgpoc.vendor import match_pairing as mp  # noqa: E402
from xgpoc.vendor import pairing, pairing_inputs, pairing_report  # noqa: E402
from xgpoc.vendor import shot_pairing as sp  # noqa: E402

GAME = "england_epl/2015-2016/2015-08-16 - 18-00 Alpha 2 - 1 Beta"
OFFSET = {1: 7300, 2: -2100}
MATCH_ID = 11
ANCHORS = {1: [120_000, 400_000, 700_000, 1_000_000, 1_500_000], 2: [100_000, 350_000, 800_000, 1_200_000, 1_700_000]}
SB_COLUMNS = ["match_id", "event_id", "index", "period", "timestamp_ms", "team", "player", "play_pattern", "x", "y", "under_pressure",
              "statsbomb_xg", "outcome", "body_part", "shot_type", "technique", "first_time", "n_freeze_frame"]


def sb_shot(event_id, period, ms, team="Alpha", outcome="Saved", shot_type="Open Play", match_id=MATCH_ID):
    return {"match_id": match_id, "event_id": event_id, "index": 0, "period": period, "timestamp_ms": ms, "team": team, "player": "P",
            "play_pattern": "Regular Play", "x": 100.0, "y": 40.0, "under_pressure": False, "statsbomb_xg": 0.1, "outcome": outcome,
            "body_part": "Right Foot", "shot_type": shot_type, "technique": "Normal", "first_time": False, "n_freeze_frame": 3}


def sb_match(match_id=MATCH_ID, home="Alpha", away="Beta", score=(2, 1), date="2015-08-16", competition="Premier League"):
    return {"match_id": match_id, "match_date": date, "kick_off": "17:00:00.000", "competition": competition, "season": "2015/2016",
            "home_team": home, "away_team": away, "home_score": score[0], "away_score": score[1]}


def labels_record(game=GAME, home="Alpha", away="Beta", score="2 - 1", date="16/08/2015 - 18:00", annotations=()):
    return {"UrlLocal": game + "/", "gameDate": date, "gameHomeTeam": home, "gameAwayTeam": away, "gameScore": score,
            "annotations": list(annotations)}


def annotation(half, position_ms, label="Shots off target", team="home"):
    return {"gameTime": f"{half} - 00:00", "label": label, "position": str(position_ms), "team": team, "visibility": "visible"}


class World:
    """Builds the four inputs of `pair_sources`. `shot` adds a SoccerNet shot, `sb` a StatsBomb shot."""

    def __init__(self, anchors=True, game=GAME):
        self.game, self.sn, self.sb_rows, self.annotations, self.dropped = game, [], [], [], []
        self.own_goals = []
        if anchors:
            for half, times in ANCHORS.items():
                for k, ms in enumerate(times):
                    self.pair(half, ms, jitter=(-15, 0, 12, 5, -8)[k], team="home")

    def shot(self, half, position_ms, team="home", goal=False, label="Shots off target", penalty=False, split="train"):
        self.sn.append({"shot_id": f"{self.game}__{half}__{position_ms}", "game": self.game, "split": split, "goal": goal,
                        "is_penalty": penalty, "label": label, "half": half, "position_ms": position_ms})
        self.annotations.append(annotation(half, position_ms, label, team))
        return self.sn[-1]["shot_id"]

    def sb(self, event_id, period, ms, team="Alpha", outcome="Saved", shot_type="Open Play"):
        self.sb_rows.append(sb_shot(event_id, period, ms, team, outcome, shot_type))

    def pair(self, half, sb_ms, jitter=0, team="home", goal=False, outcome="Saved", **kwargs):
        """A SoccerNet shot and its StatsBomb shot, the planted offset apart (plus `jitter`)."""
        event = f"e{half}_{sb_ms}"
        self.sb(event, half, sb_ms, "Alpha" if team == "home" else "Beta", "Goal" if goal else outcome)
        self.shot(half, sb_ms + OFFSET[half] + jitter, team, goal, **kwargs)
        return event

    def drop_goal(self, half, label_ms, team="home"):
        """A Goal label without a shot annotation: attempts anchor it 1000 ms before the label."""
        self.annotations.append(annotation(half, label_ms, "Goal", team))
        self.dropped.append({"game": self.game, "half": half, "position_ms": label_ms - 1000, "label": "Goal",
                             "status": "goal_without_shot_annotation", "shot_id": ""})

    def inputs(self, matches=None, records=None):
        index = pd.DataFrame(self.sn)
        attempts = pd.concat([index.assign(status="ok"), pd.DataFrame(self.dropped, columns=["game", "half", "position_ms", "label", "status", "shot_id"])],
                             ignore_index=True)
        shots = pd.DataFrame(self.sb_rows, columns=SB_COLUMNS)
        shots["index"] = np.arange(len(shots))
        context = pd.DataFrame([{"match_id": MATCH_ID, "event_id": f"og{k}", "type": "Own Goal For", "period": p, "timestamp_ms": ms,
                                 "team": "Alpha"} for k, (p, ms) in enumerate(self.own_goals)],
                               columns=["match_id", "event_id", "type", "period", "timestamp_ms", "team"])
        records = records or [labels_record(self.game, annotations=self.annotations)]
        return index, records, attempts, {"matches": pd.DataFrame(matches or [sb_match()]), "shots": shots, "context": context}

    def run(self):
        return pairing.pair_sources(*self.inputs())


def row(frame, **where):
    selected = frame
    for column, value in where.items():
        selected = selected[selected[column] == value]
    assert len(selected) == 1, f"{len(selected)} rows for {where}"
    return selected.iloc[0]


# ------------------------------------------------------------------------------------------------ match pairing
def test_truncated_score_in_folder_name_pairs_by_labels_score():
    game = "england_epl/2015-2016/2015-08-16 - 18-00 Alpha 1 - 0 Beta"   # the folder says 1 - 0, Labels-v2 says 10 - 2
    games = mp.sn_games([labels_record(game, score="10 - 2")])
    assert (games.home_score[0], games.away_score[0]) == (10, 2)
    match_map = mp.pair_matches(games, pd.DataFrame([sb_match(score=(10, 2)), sb_match(match_id=12, score=(1, 0))]))
    assert match_map.sb_match_id[0] == MATCH_ID and match_map.reason[0] == "paired"


def test_match_rule_reasons():
    games = mp.sn_games([labels_record()])
    cases = {
        "paired": [sb_match()],
        "teams+date match but score differs": [sb_match(score=(3, 1))],
        "StatsBomb has matches that day, none with these teams": [sb_match(home="Gamma", away="Delta")],
        "no StatsBomb match within 1 day": [sb_match(date="2015-08-20")],
        "ambiguous: two date+score candidates pass the name rule": [sb_match(), sb_match(match_id=12, date="2015-08-17")],
    }
    for reason, matches in cases.items():
        assert mp.pair_matches(games, pd.DataFrame(matches)).reason[0] == reason
    other_competition = mp.pair_matches(games, pd.DataFrame([sb_match(competition="La Liga")]))
    assert other_competition.reason[0] == "no StatsBomb match within 1 day"
    next_day = mp.pair_matches(games, pd.DataFrame([sb_match(date="2015-08-17")]))
    assert next_day.reason[0] == "paired"


def test_reused_statsbomb_match_fails():
    twin = "england_epl/2015-2016/2015-08-16 - 18-00 Alpha 2 - 1 Beta twin"
    games = mp.sn_games([labels_record(), labels_record(twin)])
    with pytest.raises(ValueError, match="more than one SoccerNet match"):
        mp.pair_matches(games, pd.DataFrame([sb_match()]))


def test_alias_list_covers_clubs_and_a_missing_alias_is_found(monkeypatch):
    games = mp.sn_games([labels_record(home="Ath Bilbao", away="Beta")])
    matches = pd.DataFrame([sb_match(home="Athletic Club", away="Beta")])
    matches["season"] = "2015/2016"
    matches = pd.concat([matches] * mp.MIN_SB_SEASON_MATCHES, ignore_index=True)
    assert mp.unmatched_clubs(games, matches)["sn_without_sb"] == []
    monkeypatch.setattr(mp, "ALIAS", {})
    assert mp.unmatched_clubs(games, matches)["sn_without_sb"] == [("Premier League 2015/2016", "Ath Bilbao")]
    assert mp.unmatched_clubs(games, matches)["sb_without_sn"] == [("Premier League", "Athletic Club")]


@pytest.mark.skipif(not (INDEX.exists() and LABELS_ROOT.exists() and (STATSBOMB / "complete.json").exists()), reason="needs the real data")
def test_alias_list_on_the_real_clubs():
    """Every SoccerNet club of a season the open data cover has a StatsBomb club; the StatsBomb clubs without a
    SoccerNet club in any season of the competition are the ones reviewed on 2026-10-11 (no SoccerNet game with them)."""
    tables, _ = pairing_inputs.read_statsbomb(STATSBOMB)
    records, _ = pairing_inputs.read_labels(LABELS_ROOT)
    found = mp.unmatched_clubs(mp.sn_games(records), tables["matches"])
    assert found["sn_without_sb"] == []
    assert found["sb_without_sn"] == [("1. Bundesliga", "Hannover 96"), ("Ligue 1", "Gazélec Ajaccio"), ("Ligue 1", "Troyes")]


# ------------------------------------------------------------------------------------------------ shot pairing
def test_planted_offsets_are_recovered_per_half():
    result = World().run()
    offsets = result.half_offsets.set_index("half").offset_ms
    assert offsets[1] == pytest.approx(OFFSET[1], abs=15) and offsets[2] == pytest.approx(OFFSET[2], abs=15)
    assert result.pairs.paired.all() and len(result.pairs) == 10
    assert result.pairs.residual_ms.abs().max() <= 30
    assert set(result.half_offsets.offset_source) == {"refit"}


def test_two_shots_competing_for_one_statsbomb_shot_pair_one_to_one():
    world = World()
    world.sb("target", 1, 2_000_000)
    near = world.shot(1, 2_000_000 + OFFSET[1] + 300)
    far = world.shot(1, 2_000_000 + OFFSET[1] + 1_100)
    result = world.run()
    pairs = result.pairs.set_index("shot_id")
    assert pairs.loc[near, "paired"] and pairs.loc[near, "sb_event_id"] == "target"
    assert not pairs.loc[far, "paired"]
    paired = result.pairs[result.pairs.paired]
    assert paired.sb_event_id.is_unique and paired.shot_id.is_unique
    assert result.unpaired_sn.shot_id.tolist() == [far]


def test_shot_outside_the_window_stays_unpaired():
    world = World()
    world.sb("lonely", 1, 2_000_000)
    inside = world.shot(1, 2_000_000 + OFFSET[1] + 2_400)
    result = world.run()
    assert result.pairs.set_index("shot_id").loc[inside, "paired"]
    world = World()
    world.sb("lonely", 1, 2_000_000)
    outside = world.shot(1, 2_000_000 + OFFSET[1] + 2_600)
    result = world.run()
    assert not result.pairs.set_index("shot_id").loc[outside, "paired"]
    unpaired = row(result.unpaired_sn, shot_id=outside)
    assert unpaired.nearest_sb_ms == pytest.approx(2_600, abs=30) and unpaired.nearest_sb_event_id == "lonely"
    assert "lonely" in set(result.sb_unpaired.event_id)


def test_window_sensitivity_table_and_k0():
    world = World()
    world.sb("late", 1, 2_000_000)
    world.shot(1, 2_000_000 + OFFSET[1] + 700)
    result = world.run()
    sensitivity = result.report["window_sensitivity"]
    assert {v["window_ms"] for v in sensitivity.values()} == {250, 500, 1000, 2500, 5000}
    assert sensitivity["offset_w250"]["pairs"] == 10 and sensitivity["offset_w1000"]["pairs"] == 11
    assert sensitivity["offset_w2500"]["same_pairs_as_main"] == 11
    assert sensitivity["no_offset_w250"]["pairs"] == 0
    assert result.report["k0"]["passed"] and result.report["shot_pairing"]["paired"] == 11
    pairing_report.check_k0(result.report)


def test_k0_fails_when_too_few_shots_pair():
    world = World()
    for k in range(3):
        world.shot(1, 3_000_000 + 100_000 * k)     # no StatsBomb shot anywhere near
    result = world.run()
    assert result.report["shot_pairing"]["paired_rate"] == pytest.approx(10 / 13, abs=1e-4)
    with pytest.raises(pairing_report.K0Failed, match="paired"):
        pairing_report.check_k0(result.report)


def test_k0_fails_on_goal_disagreement():
    world = World(anchors=False)
    for half, times in ANCHORS.items():
        for k, ms in enumerate(times):
            world.pair(half, ms, goal=k == 0, team="home")
    result = world.run()
    pairing_report.check_k0(result.report)   # all goals agree
    world = World(anchors=False)
    for half, times in ANCHORS.items():
        for k, ms in enumerate(times):
            event = f"e{half}_{ms}"
            world.sb(event, half, ms, outcome="Goal" if k < 2 else "Saved")
            world.shot(half, ms + OFFSET[half], goal=False)
    with pytest.raises(pairing_report.K0Failed, match="goal agreement"):
        pairing_report.check_k0(world.run().report)


def test_team_side_and_goal_label_disagreements_are_listed_with_reasons():
    world = World()
    flipped = "flipped"
    world.sb(flipped, 1, 2_000_000, team="Alpha")                         # StatsBomb: Alpha, the home team
    world.shot(1, 2_000_000 + OFFSET[1] + 20, team="away")                # SoccerNet: the away team
    world.pair(2, 2_100_000, team="home", goal=True)
    scored = world.run()
    assert scored.report["shot_pairing"]["team_disagree"] == 1
    team_list = scored.report["disagreements"]["team_side"]
    assert [d["sb_event_id"] for d in team_list] == [flipped] and "tight pair" in team_list[0]["reason"]
    assert scored.report["disagreements"]["goal_label"] == []

    world = World()
    world.pair(1, 2_000_000, goal=True)
    # the same shot is a goal in StatsBomb only: change the SoccerNet flag
    world.sn[-1]["goal"] = False
    result = world.run()
    labels = result.report["disagreements"]["goal_label"]
    assert len(labels) == 1 and labels[0]["sb_goal"] is True and labels[0]["goal"] is False
    assert "no SoccerNet Goal label" in labels[0]["reason"]
    assert result.report["shot_pairing"]["goal_disagreements"] == {"sn_goal_sb_not": 0, "sb_goal_sn_not": 1}


def test_conflicting_team_annotations_make_the_team_ambiguous():
    world = World()
    shot_id = world.pair(1, 2_000_000, team="home")
    world.annotations.append(annotation(1, 2_000_000 + OFFSET[1], "Shots off target", "away"))
    result = world.run()
    pair = row(result.pairs, sb_event_id=shot_id)
    assert pair.team == sp.AMBIGUOUS_TEAM and pd.isna(pair.team_agree)
    assert result.report["shot_pairing"]["team_ambiguous"] == 1 and result.report["shot_pairing"]["team_disagree"] == 0


def far_half(world, shift_ms):
    """Half 1 with four shots (irregularly spaced) whose clocks are `shift_ms` apart: the planted offset."""
    for k, ms in enumerate([2_000_000, 2_071_000, 2_233_000, 2_410_000]):
        world.sb(f"far{k}", 1, ms)
        world.shot(1, ms + shift_ms)


def test_a_planted_half_offset_of_seventy_seconds_is_recovered():
    world = World(anchors=False)
    far_half(world, 70_000)
    for ms in ANCHORS[2]:
        world.pair(2, ms)
    result = world.run()
    half = result.half_offsets.set_index("half")
    assert half.offset_ms[1] == 70_000 and half.n_pairs[1] == 4 and half.offset_source[1] == "refit"
    assert result.pairs[result.pairs.half == 1].paired.all()
    assert half.offset_ms[2] == pytest.approx(OFFSET[2], abs=15)


def test_a_half_the_offset_search_cannot_align_is_reported_not_guessed():
    """Half 1 has its shots 700 s apart on the two clocks, beyond the 120 s search: no offset, nothing paired, and the
    StatsBomb shots of the half are kept out of the secondary population."""
    world = World(anchors=False)
    far_half(world, 700_000)
    for ms in ANCHORS[2]:
        world.pair(2, ms)
    result = world.run()
    offsets = result.half_offsets.set_index("half")
    assert np.isnan(offsets.offset_ms[1]) and offsets.offset_source[1] == "none"
    assert not result.pairs[result.pairs.half == 1].paired.any()
    assert set(result.sb_unpaired[result.sb_unpaired.period == 1].population) == {"unaligned_half"}
    assert result.report["shot_pairing"]["unpaired_in_halves_without_offset"] == 4
    assert result.unpaired_sn.nearest_sb_event_id.isna().all() and result.unpaired_sn.nearest_sb_ms.isna().all()


# ------------------------------------------------------------------------------------------------ recovery
def test_dropped_goal_is_recovered_with_its_lag():
    world = World()
    world.sb("goal", 1, 2_000_000, outcome="Goal")
    world.drop_goal(1, 2_000_000 + OFFSET[1] + 600)   # label 600 ms after the StatsBomb goal; anchor 400 ms before it
    result = world.run()
    recovered = row(result.recovered_goals, sb_event_id="goal")
    assert recovered.recovered and recovered.reason == "recovered" and recovered.team_agree
    assert recovered.lag_ms == pytest.approx(-400, abs=20) and recovered.label_lag_ms == pytest.approx(600, abs=20)
    assert recovered.anchor_ms == 2_000_000 + OFFSET[1] - 400
    assert row(result.sb_unpaired, event_id="goal").population == "recovered_goal"
    target = row(result.cut_targets, sb_event_id="goal")
    assert target.kind == "recovered_goal" and target.target_position_ms == pytest.approx(2_000_000 + OFFSET[1], abs=20)
    assert result.report["sb_side"]["recovery"]["recovered"] == 1


def test_dropped_goals_are_recovered_one_to_one():
    world = World()
    world.sb("goal", 1, 2_000_000, outcome="Goal")
    world.drop_goal(1, 2_000_000 + OFFSET[1] + 500)
    world.drop_goal(1, 2_000_000 + OFFSET[1] + 3_000)
    result = world.run()
    table = result.recovered_goals.sort_values("goal_label_ms")
    assert table.recovered.tolist() == [True, False]
    assert table.sb_event_id.dropna().tolist() == ["goal"]
    assert table.reason.tolist() == ["recovered", "candidate taken by another dropped goal"]


def test_dropped_goal_reasons_own_goal_and_no_candidate():
    world = World()
    world.drop_goal(1, 2_000_000 + OFFSET[1])
    world.own_goals.append((1, 2_000_300))
    world.drop_goal(2, 2_200_000 + OFFSET[2])
    result = world.run()
    reasons = result.recovered_goals.set_index("half").reason
    assert reasons[1] == "own goal in StatsBomb" and reasons[2] == "no unpaired StatsBomb goal in the window"
    assert result.report["sb_side"]["own_goals_in_paired_matches"] == 1


def test_recovery_window_edges():
    world = World()
    for k, lag in enumerate([-3_000, -3_100, 8_000, 8_100]):   # lag = anchor - offset - StatsBomb time
        world.sb(f"g{k}", 1, 2_000_000 + 100_000 * k, outcome="Goal")
        world.drop_goal(1, 2_000_000 + 100_000 * k + OFFSET[1] + lag + 1000)
    got = world.run().recovered_goals.sort_values("goal_label_ms").recovered.tolist()
    assert got == [True, False, True, False]


def test_goal_without_a_goal_annotation_is_an_error():
    world = World()
    world.dropped.append({"game": GAME, "half": 1, "position_ms": 555_000, "label": "Goal", "status": "goal_without_shot_annotation",
                          "shot_id": ""})
    with pytest.raises(ValueError, match="no Goal annotation"):
        world.run()


# ------------------------------------------------------------------------------------------------ secondary population
def test_statsbomb_only_shot_near_an_unpaired_soccernet_shot_is_flagged():
    world = World()
    world.shot(1, 2_000_000 + OFFSET[1])                       # SoccerNet shot with no StatsBomb partner in 2.5 s
    world.sb("near", 1, 2_000_000 + 6_000, outcome="Blocked")  # 6 s after it on the StatsBomb clock: mispair candidate
    world.sb("far", 1, 2_000_000 + 30_000, outcome="Blocked")
    result = world.run()
    near, far = row(result.sb_unpaired, event_id="near"), row(result.sb_unpaired, event_id="far")
    assert near.population == "possible_mispair" and "possible mispair" in near.reason and near.outcome == "Blocked"
    assert far.population == "secondary" and far.reason == "not annotated by SoccerNet: outcome Blocked"
    assert result.secondary.event_id.tolist() == ["far"]
    assert "near" not in set(result.cut_targets.sb_event_id)
    assert near.nearest_unpaired_sn_ms == pytest.approx(6_000, abs=30)


def test_secondary_population_excludes_penalties_extra_time_and_paired_shots():
    world = World()
    world.sb("pen", 1, 2_200_000, outcome="Goal", shot_type="Penalty")
    world.sb("extra", 3, 100_000, outcome="Saved")
    world.sb("plain", 2, 2_300_000, outcome="Off T")
    result = world.run()
    populations = result.sb_unpaired.set_index("event_id").population.to_dict()
    assert populations == {"pen": "penalty", "extra": "extra_time", "plain": "secondary"}
    assert (result.sb_unpaired.outcome.notna() & result.sb_unpaired.reason.str.len().gt(0)).all()
    assert result.secondary.event_id.tolist() == ["plain"] and result.secondary.anchor_convention.eq(pairing_report.ANCHOR_CONVENTION).all()
    # every StatsBomb shot of the paired match is paired or listed once
    assert result.report["sb_side"]["paired"] + result.report["sb_side"]["unpaired"] == len(world.sb_rows)
    assert set(result.pairs.sb_event_id.dropna()).isdisjoint(result.sb_unpaired.event_id)


def test_secondary_goal_without_soccernet_goal_is_flagged():
    world = World()
    world.sb("lost-goal", 2, 2_300_000, outcome="Goal")
    result = world.run()
    lost = row(result.secondary, event_id="lost-goal")
    assert lost.sb_goal_without_sn_goal and result.report["sb_side"]["secondary"]["goals_without_sn_goal"] == 1


def test_cut_targets_and_halves_to_cut():
    world = World()
    world.sb("a", 1, 2_000_000)
    world.sb("b", 2, 2_300_000)
    world.sb("goal", 2, 2_500_000, outcome="Goal")
    world.drop_goal(2, 2_500_000 + OFFSET[2] + 700)
    result = world.run()
    targets = result.cut_targets.set_index("sb_event_id")
    offsets = result.half_offsets.set_index("half").offset_ms
    assert targets.loc["a", "target_position_ms"] == round(2_000_000 + offsets[1])
    assert targets.loc["b", "target_position_ms"] == round(2_300_000 + offsets[2])
    assert targets.loc["goal", "kind"] == "recovered_goal" and targets.usable.all()
    cuts = pairing.halves_to_cut(result.cut_targets).set_index("half")
    assert cuts.secondary_shots.to_dict() == {1: 1, 2: 1} and cuts.recovered_goals.to_dict() == {1: 0, 2: 1}
    assert result.report["sb_side"]["cut_targets"]["halves"] == 2


def test_unpaired_matches_get_a_reason_and_no_pairs():
    world = World()
    other = "england_epl/2015-2016/2015-09-01 - 18-00 Gamma 0 - 0 Delta"
    index, records, attempts, tables = world.inputs()
    records.append(labels_record(other, "Gamma", "Delta", "0 - 0", "01/09/2015 - 18:00"))
    index = pd.concat([index, pd.DataFrame([{"shot_id": "x", "game": other, "split": "test", "goal": False, "is_penalty": False,
                                             "label": "Shots off target", "half": 1, "position_ms": 5000}])], ignore_index=True)
    result = pairing.pair_sources(index, records, attempts, tables)
    assert result.match_map.set_index("game").reason.to_dict() == {
        GAME: "paired", other: "no StatsBomb match within 1 day"}
    assert set(result.pairs.game) == {GAME}


def test_index_game_without_labels_record_is_an_error():
    index, records, attempts, tables = World().inputs()
    with pytest.raises(ValueError, match="no Labels-v2 record"):
        pairing.pair_sources(index.assign(game="elsewhere/2015-2016/x"), records, attempts, tables)


# ------------------------------------------------------------------------------------------------ half-level algorithms
def test_assign_prefers_good_residuals_to_one_more_pair():
    """SN 1000 matches SB 1000 exactly. SN 0 could still pair with SB 1000 (1000 ms) or SN 1000 with SB 3400 (2400 ms): a
    rectangular assignment would take that second pair; the dummy "unpaired" column costing the window does not."""
    pairs = sp.assign(np.array([0, 1000]), np.array([1000, 3400]), 0.0, 2500)
    assert pairs == [(1, 0, 0.0)]
    assert sp.assign(np.array([1000, 3400]), np.array([1000]), 0.0, 2500) == [(0, 0, 0.0)]


def test_offset_is_refitted_from_the_halfs_own_pairs():
    """Differences 4700 ... 5200: the densest 400 ms window holds the first five (median 4900); the refit takes the
    median of all six pairs (4950)."""
    world = World(anchors=False)
    for ms, difference in zip([137_000, 311_000, 702_000, 1_103_000, 1_529_000, 2_011_000], [4700, 4800, 4900, 5000, 5100, 5200], strict=True):
        world.pair(1, ms, jitter=difference - OFFSET[1])
    half = world.run().half_offsets.iloc[0]
    assert half.initial_offset_ms == 4900 and half.offset_ms == 4950 and half.n_pairs == 6 and half.offset_source == "refit"


def test_goal_backoff_is_the_one_the_collector_uses():
    assert inspect.signature(parse_labels).parameters["goal_backoff_ms"].default == pairing.GOAL_BACKOFF_MS


def test_densest_window_offset_ignores_unrelated_pairs():
    sb = np.array([10_000, 50_000, 90_000, 200_000])
    sn = sb + 4_321 + np.array([3, -4, 5, 0])
    assert sp.densest_window_offset(sn, sb) == pytest.approx(4_321, abs=5)
    assert np.isnan(sp.densest_window_offset(np.array([0]), np.array([500_000])))


# ------------------------------------------------------------------------------------------------ inputs and script
def write_statsbomb_version(root: Path, tables: dict) -> None:
    (root / "tables").mkdir(parents=True)
    files = {}
    for name, frame in tables.items():
        path = root / "tables" / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        files[f"tables/{name}.parquet"] = script.file_sha256(path)
    record = {"identity": {"open_data": {"repo": "statsbomb/open-data", "sha": "a" * 40}}, "files": files}
    (root / "complete.json").write_text(json.dumps(record))


def full_tables(world):
    _, _, _, tables = world.inputs()
    return tables


def test_statsbomb_tables_are_used_only_when_they_match_the_completion_record(tmp_path):
    tables = full_tables(World())
    root = tmp_path / "sb"
    with pytest.raises(ValueError, match="no completion record"):
        pairing_inputs.read_statsbomb(root)
    write_statsbomb_version(root, tables)
    loaded, provenance = pairing_inputs.read_statsbomb(root)
    assert len(loaded["shots"]) == len(tables["shots"]) and provenance["open_data"]["sha"] == "a" * 40
    assert set(provenance["tables"]) == {"tables/matches.parquet", "tables/shots.parquet", "tables/context.parquet"}
    tables["shots"].assign(outcome="Goal").to_parquet(root / "tables/shots.parquet", index=False)
    with pytest.raises(ValueError, match="differs from the hash"):
        pairing_inputs.read_statsbomb(root)


def test_labels_are_read_from_json_files_with_a_digest(tmp_path):
    root = tmp_path / "labels"
    (root / GAME).mkdir(parents=True)
    (root / GAME / "Labels-v2.json").write_text(json.dumps(labels_record()))
    records, digest = pairing_inputs.read_labels(root)
    assert mp.game_id(records[0]) == GAME and len(digest) == 64
    (root / GAME / "Labels-v2.json").write_text(json.dumps(labels_record(score="3 - 1")))
    assert pairing_inputs.read_labels(root)[1] != digest
    with pytest.raises(FileNotFoundError):
        pairing_inputs.read_labels(tmp_path / "empty")


@pytest.fixture
def script_world(tmp_path, monkeypatch):
    """The script on the fixture world: paths pointed at temporary files."""
    world = World()
    index, records, attempts, tables = world.inputs()
    paths = {"INDEX": tmp_path / "index.parquet", "ATTEMPTS": tmp_path / "attempts.parquet", "LABELS_ROOT": tmp_path / "labels",
             "STATSBOMB": tmp_path / "sb"}
    index.to_parquet(paths["INDEX"])
    attempts.to_parquet(paths["ATTEMPTS"])
    (paths["LABELS_ROOT"] / GAME).mkdir(parents=True)
    (paths["LABELS_ROOT"] / GAME / "Labels-v2.json").write_text(json.dumps(records[0]))
    write_statsbomb_version(paths["STATSBOMB"], tables)
    for name, path in paths.items():
        monkeypatch.setattr(script, name, path)
    return world, paths, tmp_path


def test_script_publishes_verifies_and_refuses_changes(script_world):
    world, paths, tmp_path = script_world
    out = tmp_path / "out"
    script.main(["--out", str(out)])
    record = json.loads((out / "complete.json").read_text())
    assert set(record["files"]) >= {"match_map.csv", "pairs.parquet", "half_offsets.csv", "recovered_goals.csv", "sb_unpaired.parquet",
                                    "secondary.parquet", "unpaired_sn.parquet", "cut_targets.parquet", "halves_to_cut.csv",
                                    "pairing_report.json"}
    inputs = record["identity"]["inputs"]
    assert inputs["statsbomb"]["complete_json_sha256"] == script.file_sha256(paths["STATSBOMB"] / "complete.json")
    assert inputs["index"]["sha256"] == script.file_sha256(paths["INDEX"]) and record["summary"]["pairs"] == 10
    assert not out.with_name("out.partial").exists()
    script.main(["--out", str(out)])                                        # verifies, does not rebuild
    (out / "match_map.csv").write_text("changed")
    with pytest.raises(ValueError, match="Output changed"):
        script.main(["--out", str(out)])


def test_script_stops_on_k0_without_publishing(script_world):
    world, paths, tmp_path = script_world
    for k in range(5):
        world.shot(1, 3_000_000 + 100_000 * k)
    index, records, attempts, _ = world.inputs()
    index.to_parquet(paths["INDEX"])
    attempts.to_parquet(paths["ATTEMPTS"])
    (paths["LABELS_ROOT"] / GAME / "Labels-v2.json").write_text(json.dumps(records[0]))
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="STOPPED"):
        script.main(["--out", str(out)])
    assert not out.exists()
    failed = json.loads((tmp_path / "out_k0_failed.json").read_text())
    assert failed["k0"]["passed"] is False


def test_script_does_not_replace_a_folder_it_did_not_write(script_world):
    world, paths, tmp_path = script_world
    out = tmp_path / "precious"
    out.mkdir()
    (out / "notes.txt").write_text("keep")
    with pytest.raises(SystemExit, match="refusing"):
        script.main(["--out", str(out)])
    assert (out / "notes.txt").read_text() == "keep"


def test_script_refuses_changed_inputs_in_a_finished_folder(script_world):
    world, paths, tmp_path = script_world
    out = tmp_path / "out"
    script.main(["--out", str(out)])
    index = pd.read_parquet(paths["INDEX"])
    index.assign(split="valid").to_parquet(paths["INDEX"])
    with pytest.raises(ValueError, match="Inputs or sources changed"):
        script.main(["--out", str(out)])


# ------------------------------------------------------------------------------------------------ real data
@pytest.mark.skipif(not (INDEX.exists() and LABELS_ROOT.exists() and (STATSBOMB / "complete.json").exists()), reason="needs the real data")
def test_real_data_widening_the_offset_search_changes_three_halves(monkeypatch):
    """The prototype's 15 s search and the 120 s search give the same offset for 267 of 270 halves. Manchester City-Chelsea
    half 2 (+73.6 s) had none; Chelsea-Southampton half 1 (-63.0 s) and Betis-Barcelona half 1 (-17.3 s) had a spurious
    dense window near zero (2 and 5 pairs against 4 and 8 pairs with residuals of tens of ms)."""
    tables, _ = pairing_inputs.read_statsbomb(STATSBOMB)
    records, _ = pairing_inputs.read_labels(LABELS_ROOT)
    inputs = (pd.read_parquet(INDEX), records, pd.read_parquet(script.ATTEMPTS), tables)
    wide = pairing.pair_sources(*inputs).half_offsets.set_index(["game", "half"])
    monkeypatch.setattr(sp, "OFFSET_SEARCH_MS", 15_000)
    narrow = pairing.pair_sources(*inputs).half_offsets.set_index(["game", "half"])
    differing = wide.index[~(wide.offset_ms.eq(narrow.offset_ms) | (wide.offset_ms.isna() & narrow.offset_ms.isna()))]
    teams_and_half = sorted((g.rsplit("/", 1)[1].split(" ", 3)[3], h) for g, h in differing)   # "Home h - a Away", half
    assert teams_and_half == [("Betis 1 - 1 Barcelona", 1), ("Chelsea 1 - 3 Southampton", 1), ("Manchester City 3 - 0 Chelsea", 2)]
    assert narrow.offset_ms.isna().sum() == 1 and wide.offset_ms.isna().sum() == 0
    assert sorted(wide.offset_ms[differing].tolist()) == [-63034.5, -17308.0, 73550.0]


@pytest.mark.skipif(not (INDEX.exists() and LABELS_ROOT.exists() and (STATSBOMB / "complete.json").exists()), reason="needs the real data")
@pytest.mark.parametrize(("search_ms", "pairs", "goal_agree", "unpaired_sb", "team_disagree"), [
    (15_000, 2741, "2728/2741", 888, 20),      # the prototype's search: its counts are reproduced
    (120_000, 2754, "2741/2754", 875, 19),     # the frozen search
])
def test_real_data_counts(monkeypatch, search_ms, pairs, goal_agree, unpaired_sb, team_disagree):
    monkeypatch.setattr(sp, "OFFSET_SEARCH_MS", search_ms)
    tables, _ = pairing_inputs.read_statsbomb(STATSBOMB)
    records, _ = pairing_inputs.read_labels(LABELS_ROOT)
    result = pairing.pair_sources(pd.read_parquet(INDEX), records, pd.read_parquet(script.ATTEMPTS), tables)
    report = result.report
    assert report["match_pairing"]["paired"] == 135 and report["match_pairing"]["min_name_sim_paired"] >= 0.9
    shot = report["shot_pairing"]
    assert (shot["index_shots"], shot["paired"], shot["goal_agree"]) == (3039, pairs, goal_agree)
    assert report["sb_side"]["recovery"]["recovered"] == 50 and report["sb_side"]["recovery"]["dropped_goals"] == 74
    assert shot["unpaired_no_sb_shot_within_10s"] == 249 and report["k0"]["passed"]
    assert len(result.sb_unpaired) == unpaired_sb and result.pairs.sb_event_id.dropna().is_unique
    assert len(report["disagreements"]["team_side"]) == team_disagree and len(report["disagreements"]["goal_label"]) == 13
