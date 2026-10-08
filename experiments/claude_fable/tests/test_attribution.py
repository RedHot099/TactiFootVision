"""Goal attribution rules on a fixture game with known rebound chains.

Fixture (home team H, away team A, one period, times in seconds):
  chain 1   H shots at 10, 13, 16 (links 3 s), goal at 17 -> goal chain of three
  lone      A shot at 30 (no chain mate), goal at 31
  chain 2   H shots at 50, 53; goal at 54 is cut off from a later H shot at 56 (the goal
            ends the chain), so the goal chain is 50, 53
  broken    H shots at 70 and 73 with an A clearance at 71: possession changed, no chain
  far       H shots at 90 and 100: more than 6 s apart, no chain
  miss      H shots at 120, 122 without a goal: a chain, but no chain goal
"""

import json

import numpy as np
import pandas as pd
import pytest
from xgpoc.data.soccernet_labels import parse_labels
from xgpoc.eval.attribution import (
    attribute,
    bound_labels,
    chain_shot_keys,
    read_events,
    recount,
    relabel,
    source_events,
)

GAME = "league/season/game"
SHOTS = {  # second -> team
    10: "home", 13: "home", 16: "home", 30: "away", 50: "home", 53: "home", 56: "home",
    70: "home", 73: "home", 90: "home", 100: "home", 120: "home", 122: "home",
}


def _ann(sec: int, label: str, team: str) -> dict:
    return {"gameTime": f"1 - {sec // 60:02d}:{sec % 60:02d}", "label": label, "position": str(sec * 1000), "team": team}


@pytest.fixture
def labels_root(tmp_path):
    ann = [_ann(s, "Shots on target", t) for s, t in SHOTS.items()]
    ann += [_ann(17, "Goal", "home"), _ann(31, "Goal", "away"), _ann(54, "Goal", "home"), _ann(71, "Clearance", "away")]
    path = tmp_path / GAME
    path.mkdir(parents=True)
    (path / "Labels-v2.json").write_text(json.dumps({"annotations": ann[::-1]}))  # file order must not matter
    return tmp_path


@pytest.fixture
def attr(labels_root):
    return attribute(source_events([GAME], labels_root))


@pytest.fixture
def index():
    """Index rows for every shot except the one at 56 s (a shot the index does not hold)."""
    secs = [s for s in SHOTS if s != 56]
    return pd.DataFrame({"game": GAME, "half": 1, "position_ms": [s * 1000 for s in secs],
                         "shot_id": [f"s{s}" for s in secs], "split": "test"})


def goals_of(index, labels):
    return sorted(int(ms) // 1000 for ms in index.position_ms[labels])


def test_chains_follow_possession_and_gap(attr):
    sizes = attr.shots.assign(sec=attr.shots.position_ms // 1000).set_index("sec").chain_size.to_dict()
    assert {s: sizes[s] for s in (10, 13, 16)} == {10: 3, 13: 3, 16: 3}
    assert sizes[30] == 1                                   # lone shot
    assert sizes[50] == sizes[53] == 2 and sizes[56] == 1   # the goal at 54 ends the chain
    assert sizes[70] == sizes[73] == 1                      # opponent clearance in between
    assert sizes[90] == sizes[100] == 1                     # 10 s apart
    assert sizes[120] == sizes[122] == 2                    # a chain without a goal


def test_goal_members_are_chain_shots_up_to_the_goal(attr):
    goals = attr.goals.set_index(attr.goals.goal_ms // 1000)
    assert goals.nearest.to_dict() == {17: 16000, 31: 30000, 54: 53000}
    assert goals.members[17] == (10000, 13000, 16000)
    assert goals.members[31] == (30000,)
    assert goals.members[54] == (50000, 53000)


def test_rules_give_expected_labels(attr, index):
    assert goals_of(index, relabel(index, attr, "nearest")[0]) == [16, 30, 53]
    assert goals_of(index, relabel(index, attr, "earliest")[0]) == [10, 30, 50]
    assert goals_of(index, relabel(index, attr, "last")[0]) == [16, 30, 53]
    goal, keep = relabel(index, attr, "exclude")
    assert goals_of(index, goal) == [30]
    assert sorted(index.position_ms[~keep] // 1000) == [10, 13, 16, 50, 53, 120, 122]


def test_unknown_rule_is_rejected(attr, index):
    with pytest.raises(ValueError):
        relabel(index, attr, "median")


def test_bounds_pick_extreme_shot_of_each_chain(attr, index):
    p = pd.Series(0.1, index=index.position_ms // 1000)
    p[10], p[13], p[16], p[50], p[53] = 0.05, 0.4, 0.2, 0.3, 0.02
    pv = p.to_numpy()
    # worst case = lowest probability gets the goal, best case = highest; the lone goal stays
    assert goals_of(index, bound_labels(index, attr, pv, "worst")) == [10, 30, 53]
    assert goals_of(index, bound_labels(index, attr, pv, "best")) == [13, 30, 50]
    with pytest.raises(ValueError):
        bound_labels(index, attr, pv, "middle")


def test_bound_matches_brute_force_log_loss(attr, index):
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.6, len(index))

    def loss(goal):
        return float(-np.sum(np.where(goal, np.log(p), np.log(1 - p))))

    chain_rows = {k: [np.flatnonzero(index.position_ms == s * 1000)[0] for s in secs]
                  for k, secs in {"c1": (10, 13, 16), "c2": (50, 53)}.items()}
    base = relabel(index, attr, "nearest")[0].copy()
    for bound, pick in (("worst", max), ("best", min)):
        got = bound_labels(index, attr, p, bound)
        trials = []
        for r1 in chain_rows["c1"]:
            for r2 in chain_rows["c2"]:
                g = base.copy()
                g[chain_rows["c1"] + chain_rows["c2"]] = False
                g[[r1, r2]] = True
                trials.append(loss(g))
        assert loss(got) == pytest.approx(pick(trials))


def test_bound_on_difference_uses_reference_model(attr, index):
    ens = pd.Series(0.1, index=index.position_ms // 1000)
    g10 = ens.copy()
    ens[13], g10[13] = 0.5, 0.1   # largest ens advantage in chain 1: shot 13
    ens[10], g10[10] = 0.05, 0.4  # smallest: shot 10
    ens[53], g10[53] = 0.3, 0.1
    # ens advantage on goal j is logit(ens_j) - logit(g10_j): best case for ens maximises it
    assert goals_of(index, bound_labels(index, attr, ens.to_numpy(), "best", g10.to_numpy()))[0] == 13
    assert goals_of(index, bound_labels(index, attr, ens.to_numpy(), "worst", g10.to_numpy()))[0] == 10


def test_goal_credited_to_shot_missing_from_index_is_lost(attr, index):
    sub = index[index.position_ms != 10_000].reset_index(drop=True)  # drop the earliest shot of chain 1
    assert goals_of(sub, relabel(sub, attr, "earliest")[0]) == [30, 50]
    # the bounds fall back on the chain shots the index does hold
    p = np.full(len(sub), 0.1)
    assert goals_of(sub, bound_labels(sub, attr, p, "worst"))[0] in (13, 16)


def test_recount(attr, index):
    r = recount(index, attr).iloc[0]
    assert (r.shots, r.chain_shots, r.chains, r.goals, r.chain_goals, r.goals_moved_by_earliest) == (12, 7, 3, 3, 2, 2)
    assert len(chain_shot_keys(attr)) == 7  # the shot at 56 s follows the goal, so it is no chain shot


def test_nearest_rule_reproduces_parser(labels_root, attr):
    """The same fixture through the production parser (xgpoc.data.soccernet_labels)."""
    parsed = parse_labels(labels_root / GAME / "Labels-v2.json")
    parser_goals = sorted(e.position_ms // 1000 for e in parsed if e.source == "shot" and e.goal)
    all_shots = pd.DataFrame({"game": GAME, "half": 1, "position_ms": [s * 1000 for s in SHOTS]})
    assert goals_of(all_shots, relabel(all_shots, attr, "nearest")[0]) == parser_goals


def test_events_are_stable_sorted_by_time(labels_root):
    ev = read_events(labels_root / GAME / "Labels-v2.json")
    assert ev.position_ms.is_monotonic_increasing
