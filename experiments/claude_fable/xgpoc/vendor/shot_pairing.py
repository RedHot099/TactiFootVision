"""Shot-level pairing of SoccerNet and StatsBomb shots inside one half: offsets, assignment, goal recovery.

Everything here works on millisecond arrays of one half; `xgpoc.vendor.pairing` loops over halves.

    half offset   SoccerNet minus StatsBomb time of one half. Start: the densest `DENSE_MS` window among all
                   pairwise differences within `OFFSET_SEARCH_MS` (median inside it). Then refitted from the half's
                   own pairs (median difference), `OFFSET_REFITS` times, re-pairing in between.
    assignment     one-to-one, minimal total |residual| plus the window for every unpaired SoccerNet shot (a dummy
                   "unpaired" column per row, so the solver never trades good residuals for one more pair). Only
                   pairs with |residual| <= window are kept. Time only; team is not used (decision D8).
    recovery       a SoccerNet Goal label without a shot annotation takes a still unpaired StatsBomb goal whose lag
                   (label minus offset-corrected StatsBomb time) is in [RECOVER_MS], one-to-one by minimal |lag|.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

WINDOW_MS = 2500
OFFSET_SEARCH_MS = 120_000   # was 15 s in the prototype; one half is 73.6 s off (owner decision 2026-10-11)
DENSE_MS = 400
OFFSET_REFITS = 2
RECOVER_MS = (-3000, 8000)
BIG = 1e12
MISPAIR_MS = 10_000
AMBIGUOUS_TEAM = "ambiguous"   # Labels-v2 annotations with one time and label but opposite teams


def densest_window_offset(sn_ms: np.ndarray, sb_ms: np.ndarray, search_ms: float | None = None) -> float:
    """Offset SN - SB: median of the densest `DENSE_MS` window of the pairwise differences within `search_ms`
    (default `OFFSET_SEARCH_MS`, read at call time). NaN when there are no differences that small. Deterministic;
    robust to the many unrelated SN x SB pairs."""
    search_ms = OFFSET_SEARCH_MS if search_ms is None else search_ms
    diff = (np.asarray(sn_ms)[:, None] - np.asarray(sb_ms)[None, :]).ravel()
    diff = np.sort(diff[np.abs(diff) <= search_ms])
    if len(diff) == 0:
        return float("nan")
    ends = np.searchsorted(diff, diff + DENSE_MS, side="right")
    start = int(np.argmax(ends - np.arange(len(diff))))
    return float(np.median(diff[start:ends[start]]))


def assign(sn_ms: np.ndarray, sb_ms: np.ndarray, offset: float, window: float) -> list[tuple[int, int, float]]:
    """One-to-one pairs (sn index, sb index, residual) with |residual| <= window; residual = SN - SB - offset."""
    sn_ms, sb_ms = np.asarray(sn_ms), np.asarray(sb_ms)
    n = len(sn_ms)
    if n == 0 or len(sb_ms) == 0:
        return []
    residual = sn_ms[:, None] - sb_ms[None, :] - offset
    cost = np.hstack([np.where(np.abs(residual) <= window, np.abs(residual), BIG), np.where(np.eye(n, dtype=bool), window, BIG)])
    rows, cols = linear_sum_assignment(cost)
    pairs = [(int(i), int(j), float(residual[i, j])) for i, j in zip(rows, cols, strict=True)
             if j < len(sb_ms) and abs(residual[i, j]) <= window]
    if len({j for _, j, _ in pairs}) != len(pairs) or len({i for i, _, _ in pairs}) != len(pairs):
        raise AssertionError("Assignment is not one-to-one")
    return pairs


def recover(goal_ms: np.ndarray, sb_goal_ms: np.ndarray, offset: float) -> tuple[list[int | None], np.ndarray]:
    """One-to-one recovery of dropped goal labels by unpaired StatsBomb goals of the same half.

    Returns (chosen, lag): `chosen[i]` is the index into `sb_goal_ms` taken by label i (or None); `lag[i, j]` is
    label i minus offset minus StatsBomb goal j (the whole matrix, NaN offset gives NaN). A label with no candidate
    in `RECOVER_MS` keeps None; `lag` shows whether any candidate existed."""
    goal_ms, sb_goal_ms = np.asarray(goal_ms, dtype=float), np.asarray(sb_goal_ms, dtype=float)
    lag = goal_ms[:, None] - offset - sb_goal_ms[None, :]
    chosen: list[int | None] = [None] * len(goal_ms)
    if lag.size == 0:
        return chosen, lag
    ok = (lag >= RECOVER_MS[0]) & (lag <= RECOVER_MS[1])
    if ok.any():
        n = len(goal_ms)
        dummy = max(map(abs, RECOVER_MS)) + 1
        cost = np.hstack([np.where(ok, np.abs(lag), BIG), np.where(np.eye(n, dtype=bool), dummy, BIG)])
        for i, j in zip(*linear_sum_assignment(cost), strict=True):
            if j < lag.shape[1] and ok[i, j]:
                chosen[int(i)] = int(j)
    taken = [j for j in chosen if j is not None]
    if len(set(taken)) != len(taken):
        raise AssertionError("Recovery is not one-to-one")
    return chosen, lag


class Halves:
    """The shots of both sources grouped by (game, half), as row positions in the two frames."""

    def __init__(self, sn: pd.DataFrame, sb: pd.DataFrame):
        self.sn, self.sb = sn, sb
        self.sn_rows = {k: np.asarray(v) for k, v in sn.groupby(["game", "half"]).indices.items()}
        self.sb_rows = {k: np.asarray(v) for k, v in sb.groupby(["game", "period"]).indices.items()}
        self.sn_ms = sn.position_ms.to_numpy()
        self.sb_ms = sb.timestamp_ms.to_numpy()
        self.keys = sorted(set(self.sn_rows) | {k for k in self.sb_rows if k[1] in (1, 2)})

    def of(self, key) -> tuple[np.ndarray, np.ndarray]:
        empty = np.empty(0, dtype=int)
        return self.sn_rows.get(key, empty), self.sb_rows.get(key, empty)


def initial_offsets(halves: Halves) -> dict:
    offsets = {}
    for key in halves.keys:
        sn_rows, sb_rows = halves.of(key)
        has_both = len(sn_rows) and len(sb_rows)
        offsets[key] = densest_window_offset(halves.sn_ms[sn_rows], halves.sb_ms[sb_rows]) if has_both else np.nan
    return offsets


def run_pairing(halves: Halves, offsets: dict, window: float, use_offset: bool = True, same_side: bool = False) -> pd.DataFrame:
    """Pair every half. Columns: sn_row, sb_row (positions in the two frames), residual_ms, offset_ms, game, half.
    A half without an offset is paired with offset 0. `same_side` pairs home with home and away with away only."""
    rows = []
    sn_side = halves.sn.team.to_numpy()
    sb_side = halves.sb.side.to_numpy()
    for key in halves.keys:
        sn_rows, sb_rows = halves.of(key)
        if not len(sn_rows) or not len(sb_rows):
            continue
        offset = offsets[key] if use_offset else 0.0
        offset = 0.0 if np.isnan(offset) else offset
        if same_side:
            groups = [(sn_rows[sn_side[sn_rows] == side], sb_rows[sb_side[sb_rows] == side]) for side in ("home", "away")]
        else:
            groups = [(sn_rows, sb_rows)]
        for sn_part, sb_part in groups:
            for i, j, residual in assign(halves.sn_ms[sn_part], halves.sb_ms[sb_part], offset, window):
                rows.append((sn_part[i], sb_part[j], residual, offset, key[0], key[1]))
    return pd.DataFrame(rows, columns=["sn_row", "sb_row", "residual_ms", "offset_ms", "game", "half"])


def fit(halves: Halves, window: float, refits: int) -> tuple[dict, dict, pd.DataFrame, set]:
    """(initial offsets, final offsets, final pairs, refitted halves). Each refit sets the half's offset to the median
    SN - SB difference of its own pairs and pairs again; a half without pairs keeps its offset."""
    initial = initial_offsets(halves)
    offsets, refitted = dict(initial), set()
    pairs = run_pairing(halves, offsets, window)
    for _ in range(refits):
        diff = halves.sn_ms[pairs.sn_row] - halves.sb_ms[pairs.sb_row]
        for key, median in pd.Series(diff).groupby([pairs.game, pairs.half]).median().items():
            offsets[key] = float(median)
            refitted.add(key)
        pairs = run_pairing(halves, offsets, window)
    return initial, offsets, pairs, refitted
