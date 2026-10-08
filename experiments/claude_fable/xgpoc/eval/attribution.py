"""Goal attribution rules and rebound chains, recomputed from the source annotations.

The index labels a shot `goal` by one rule (index v1.2): the goal annotation belongs to
the nearest preceding shot of the same team within the dataset protocol's 6 s goal window
(`xgpoc.data.soccernet_labels.parse_labels`). In a rebound chain that choice decides which
of several shots counts as the goal, and the annotation noise is of the order of the gaps
inside the chain. This module rebuilds the rule from the Labels-v2 events so that
evaluation labels can be recomputed under the alternatives and bounded.

Rebound chain (GLOSSARY, xG): consecutive shots by the same team in the same period, each
within `GOAL_WINDOW_MS` of the previous one, with no change of possession in between.
SoccerNet v2 has no possession events, so "no change of possession" is operational: two
consecutive same-team shots are linked unless, strictly between them, there is an event of
the opponent (any label), or a stoppage that hands the ball over (`BREAK_LABELS`: Goal,
Kick-off, Foul, Offside). A Goal annotated at the first shot's own millisecond also ends
the chain, since the goal belongs to that shot. Ball out of play, throw-ins, corners,
indirect and direct free kicks and clearances of the same team do not break a chain: the
team keeps the ball. On the 500 games, of 399 same-team shot pairs 6 s apart or less, 395 are
linked and 4 are cut by a Goal annotation; no opponent event ever falls inside such a pair,
so the opponent clause is a guard for other data, not an active filter here.

Rules (one goal annotation -> one shot of its chain, chain members at or before the goal):
  nearest   the nearest preceding same-team shot (the index v1.2 rule)
  earliest  the first shot of the chain
  last      the last shot of the chain at or before the goal. Because a Goal ends the chain
            and the nearest shot is the latest same-team shot before the goal, this is
            the same shot as `nearest` by construction; it is kept as a named rule so the
            sensitivity table shows the identity instead of assuming it
  exclude   nearest labels, with every shot of every chain of two or more left out
Bounds (`bound_labels`): each chain goal goes to the chain shot that maximises ("worst")
or minimises ("best") the model's log-loss.

Only evaluation labels are recomputed. Models are trained on the nearest-rule labels.
"""

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from xgpoc.data.soccernet_labels import GOAL_LABEL, SHOT_LABELS
from xgpoc.eval.calibration import logit
from xgpoc.paths import LABELS_ROOT

GOAL_WINDOW_MS = 6000
BREAK_LABELS = frozenset({GOAL_LABEL, "Kick-off", "Foul", "Offside"})
TEAMS = frozenset({"home", "away"})
RULES = ("nearest", "earliest", "last", "exclude")
KEY = ["game", "half", "position_ms"]


class Attribution(NamedTuple):
    """shots: one row per source shot annotation (game, half, position_ms, team, chain_id,
    chain_size). goals: one row per goal annotation (game, half, goal_ms, team, nearest,
    members); `nearest` is the position_ms of its nearest rule shot (NA when none), and
    `members` the sorted position_ms of the chain shots at or before the goal, including
    `nearest`. A goal is a chain goal when it has two or more members."""

    shots: pd.DataFrame
    goals: pd.DataFrame


# ---------------------------------------------------------------- source events

def read_events(labels_path) -> pd.DataFrame:
    """Events of one Labels-v2.json: half, position_ms, label, team; stable-sorted by time."""
    ann = json.loads(Path(labels_path).read_text())["annotations"]
    ev = pd.DataFrame({"half": [int(a["gameTime"].split(" - ")[0]) for a in ann],
                       "position_ms": [int(a["position"]) for a in ann],
                       "label": [a["label"] for a in ann],
                       "team": [a.get("team", "not applicable") for a in ann]})
    return ev.sort_values(["half", "position_ms"], kind="stable").reset_index(drop=True)


def source_events(games, root=LABELS_ROOT) -> pd.DataFrame:
    """Events of every game (path relative to `root`, as in the index `game` column)."""
    return pd.concat([read_events(Path(root) / g / "Labels-v2.json").assign(game=g) for g in sorted(set(games))],
                     ignore_index=True)


# ---------------------------------------------------------------- chains and goals

def _possession_changes(ev: pd.DataFrame, half: int, team: str, prev_ms: int, next_ms: int) -> bool:
    """Is there a change of possession for `team` strictly between its shots at prev_ms < next_ms?"""
    sel = ev[(ev.half == half) & (((ev.position_ms > prev_ms) & (ev.position_ms < next_ms))
                                  | ((ev.position_ms == prev_ms) & (ev.label == GOAL_LABEL)))]
    opponent = sel.team.isin(TEAMS) & (sel.team != team)
    return bool((opponent | sel.label.isin(BREAK_LABELS)).any())


def _attribute_game(ev: pd.DataFrame, window_ms: int) -> tuple[list[dict], list[dict]]:
    shots = ev[ev.label.isin(SHOT_LABELS)].reset_index(drop=True)
    chain_of = np.zeros(len(shots), dtype=int)
    next_chain = 0
    for _, grp in shots.groupby(["half", "team"], sort=False):
        prev = None
        for i, row in grp.iterrows():  # order of the stable time sort
            linked = (prev is not None and 0 <= row.position_ms - prev.position_ms <= window_ms
                      and not _possession_changes(ev, row.half, row.team, prev.position_ms, row.position_ms))
            if not linked:
                next_chain += 1
            chain_of[i] = next_chain
            prev = row
    shots = shots.assign(chain_id=chain_of)
    size = shots.groupby("chain_id").chain_id.transform("size")
    shot_rows = [{"half": r.half, "position_ms": r.position_ms, "team": r.team, "chain_id": r.chain_id, "chain_size": s}
                 for r, s in zip(shots.itertuples(), size, strict=True)]

    credited: set[int] = set()
    goal_rows = []
    for g in ev[ev.label == GOAL_LABEL].itertuples():
        cands = shots[(shots.half == g.half) & (shots.team == g.team) & ((g.position_ms - shots.position_ms).between(0, window_ms))
                      & ~shots.index.isin(credited)]
        row = {"half": g.half, "goal_ms": g.position_ms, "team": g.team, "nearest": pd.NA, "members": ()}
        if len(cands):
            n = cands.position_ms.idxmax()  # first of the maxima, as max() in parse_labels
            credited.add(n)
            mem = shots[(shots.chain_id == shots.chain_id[n]) & (shots.position_ms <= g.position_ms)]
            row |= {"nearest": int(shots.position_ms[n]), "members": tuple(sorted(int(x) for x in mem.position_ms))}
        goal_rows.append(row)
    return shot_rows, goal_rows


def attribute(events: pd.DataFrame, window_ms: int = GOAL_WINDOW_MS) -> Attribution:
    """Chains and goal candidates of every game in `events` (columns of `source_events`)."""
    shot_rows, goal_rows = [], []
    for game, ev in events.groupby("game", sort=True):
        s, g = _attribute_game(ev.reset_index(drop=True), window_ms)
        shot_rows += [r | {"game": game} for r in s]
        goal_rows += [r | {"game": game} for r in g]
    shots = pd.DataFrame(shot_rows)
    goals = pd.DataFrame(goal_rows, columns=["game", "half", "goal_ms", "team", "nearest", "members"])
    shots["chain_id"] = shots.game + "#" + shots.chain_id.astype(str)
    return Attribution(shots[[*KEY, "team", "chain_id", "chain_size"]], goals)


# ---------------------------------------------------------------- labels

def _goal_keys(goals: pd.DataFrame, rule: str) -> set[tuple]:
    """(game, half, position_ms) of the shot every goal is credited to under `rule`."""
    pick = {"nearest": lambda g: g.nearest, "exclude": lambda g: g.nearest,
            "earliest": lambda g: g.members[0] if g.members else pd.NA,
            "last": lambda g: g.members[-1] if g.members else pd.NA}[rule]
    out = set()
    for g in goals.itertuples():
        pos = pick(g)
        if not pd.isna(pos):
            out.add((g.game, g.half, int(pos)))
    return out


def _keys(game: pd.Series, half: pd.Series, position_ms: pd.Series) -> list[tuple]:
    """(game, half, position_ms) keys, the identity of a shot in the index and the source."""
    return list(zip(game, half.astype(int), position_ms.astype(int), strict=True))


def _row_keys(index: pd.DataFrame) -> list[tuple]:
    return _keys(index.game, index.half, index.position_ms)


def chain_shot_keys(attr: Attribution) -> set[tuple]:
    """(game, half, position_ms) of every shot in a chain of two or more."""
    s = attr.shots[attr.shots.chain_size >= 2]
    return set(_keys(s.game, s.half, s.position_ms))


def relabel(index: pd.DataFrame, attr: Attribution, rule: str) -> tuple[np.ndarray, np.ndarray]:
    """(goal, keep) per index row under `rule`. `goal` is True for the shot the goal is
    credited to; `keep` is False only under "exclude", for every shot of a rebound chain
    (and `goal` is False there too).
    A goal credited to a shot the index does not hold has no row and is lost, as under the
    index's own rule."""
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}, expected one of {RULES}")
    keys = _row_keys(index)
    credited = _goal_keys(attr.goals, rule)
    goal = np.array([k in credited for k in keys])
    keep = np.ones(len(keys), dtype=bool)
    if rule == "exclude":
        chain = chain_shot_keys(attr)
        keep = np.array([k not in chain for k in keys])
        goal &= keep
    return goal, keep


def bound_labels(index: pd.DataFrame, attr: Attribution, p: np.ndarray, bound: str,
                 p_ref: np.ndarray | None = None) -> np.ndarray:
    """Goal labels per index row where every chain goal goes to the chain shot (held by the
    index) that makes the model's log-loss worst ("worst") or best ("best"); other goals
    keep the nearest rule. `p` are the model's probabilities per index row.

    With the other chain shots labelled 0, assigning the goal to shot j gives the chain a
    log-loss of const - logit(p_j) (clipped as in the report's log-loss), so worst is the
    smallest p and best the largest. With `p_ref` the objective is the log-loss of p minus
    that of p_ref, so the bound is on the difference between two models:
    const + logit(p_ref_j) - logit(p_j). Ties go to the earliest shot."""
    if bound not in ("worst", "best"):
        raise ValueError(f"unknown bound {bound!r}, expected 'worst' or 'best'")
    pos = {k: i for i, k in enumerate(_row_keys(index))}
    score = -logit(p) if p_ref is None else logit(p_ref) - logit(p)  # loss(goal at j) up to a constant per chain
    goal = np.zeros(len(index), dtype=bool)
    for g in attr.goals.itertuples():
        if pd.isna(g.nearest):
            continue
        cands = [pos[k] for m in g.members if (k := (g.game, g.half, m)) in pos] if len(g.members) >= 2 else []
        if not cands:
            cands = [pos[k]] if (k := (g.game, g.half, int(g.nearest))) in pos else []
        if cands:
            s = score[cands]
            goal[cands[int(np.argmax(s) if bound == "worst" else np.argmin(s))]] = True
    return goal


def recount(index: pd.DataFrame, attr: Attribution) -> pd.DataFrame:
    """Per split: index shots, shots in chains, chains, goals, chain goals, and the goals
    that move to another shot under earliest-in-chain."""
    keys = _row_keys(index)
    chain = chain_shot_keys(attr)
    in_chain = np.array([k in chain for k in keys])
    near, _ = relabel(index, attr, "nearest")
    early, _ = relabel(index, attr, "earliest")
    chain_goals = attr.goals[attr.goals.members.map(len) >= 2]
    chain_goal_keys = set(_keys(chain_goals.game, chain_goals.half, chain_goals.nearest))
    is_chain_goal = np.array([k in chain_goal_keys for k in keys])
    chains = attr.shots[attr.shots.chain_size >= 2].drop_duplicates("chain_id")
    split_of = index.drop_duplicates("game").set_index("game").split
    chain_split = chains.game.map(split_of)
    rows = []
    for split in sorted(index.split.unique()):
        m = (index.split == split).to_numpy()
        rows.append({"split": split, "shots": int(m.sum()), "chain_shots": int((m & in_chain).sum()),
                     "chains": int((chain_split == split).sum()), "goals": int((m & near).sum()),
                     "chain_goals": int((m & is_chain_goal).sum()), "goals_moved_by_earliest": int((m & early & ~near).sum())})
    return pd.DataFrame(rows)
