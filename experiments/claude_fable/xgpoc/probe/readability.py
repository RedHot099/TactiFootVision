"""Readability curves, linear probes, xG event study and controls of the direction probe (xg-credibility 09).

Pure functions of the per-shot prediction table (`xgpoc.probe.table`, test seam 1) and the index, implementing the
approved protocol `research/xgc_direction_probe_protocol.md` (sha256 `PROTOCOL_SHA256`). Every number below is the
protocol's; nothing here is tuned. The estimator of the post-contact share (S, U, t50), the known-answer test and the
verdict are ticket 10; they are functions of the curves D computed here (`trend`, `jump`, the per-match sums and the
shared bootstrap counts), so ticket 10 reuses this module instead of recomputing.

Terms (protocol, "Metryka czytelności"):
  reading r     F (linear probe on dinov2_frozen), G (g_combo on-target logit), X (g_combo xG logit),
                X0 (g_combo_no_head xG logit), Gp / G0p (linear probes on the G embeddings)
  z_r(i, o)     the reading's logit of shot i at offset o
  centring      z minus the mean of z over the population shots of the same match with that reading, per offset;
                in a stratum the match mean still comes from the whole population of the match
  D_r(o)        mean centred logit over on-target minus over off-target shots (of the stratum)
  P_r(o)        OLS line through (o, D_r(o)) for o in PRE
  Delta_r       mean of D_r over POST minus P_r(0)              (positive-control jump)
  pi(o)         (D_F(o) - P'_F(o)) / Delta_F, P' the line through PLACEBO_FIT, o in PLACEBO_TEST

The xG readings use `xg_logit_raw` (mean raw logit of the scoring models, i.e. averaged before any calibration).

Linear probe (protocol, "Sondy liniowe"): fixed in time, one probe per fold trained once on the PRE and POST frames
of its training shots (9 rows per shot, same label) and applied unchanged at all 30 offsets; StandardScaler on the
training rows, then `LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000)`, no class weights. C is chosen
once per encoder on the grid `C_GRID` by the mean over folds of the held-out log-loss on the same kind of rows (PRE and
POST frames, true labels); ties go to the smaller C.

Bootstrap: `match_draw_counts` reproduces the draw sequence of `scripts/report_complexity.game_draws` (and
`scripts/significance.py`): keys `np.unique` of the population's matches, `default_rng(0)`, each draw
`rng.choice(M, M, replace=True)`; a match drawn k times has weight k. Probes are not refitted in draws; every statistic
is recomputed per draw from per-match sums. A draw whose stratum lacks on- or off-target shots is NaN (skipped).
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from xgpoc.data.offset_frames import EPS_MS, GRID_MS, TOLERANCE_MS
from xgpoc.probe.table import FRAME_STATUSES_WITH_FRAME

PROTOCOL_SHA256 = "23c8bd25934a1fd7b63fe6513f9dd425899522641a19b068313981de7b0775a0"
GRID = tuple(GRID_MS)
OFFSET_INDEX = {o: i for i, o in enumerate(GRID)}
PRE = (-1600, -1400, -1200, -1000, -800, -600)
DENSE = tuple(range(-400, 401, 40))
POST = (400, 600, 800)
PROBE_TRAIN = PRE + POST
PLACEBO_FIT = (-1600, -1400, -1200, -1000)
PLACEBO_TEST = (-800, -600)
C_GRID = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
MAX_ITER = 5000
N_BOOT = 2000
BOOT_SEED = 0
NEG_SEED = 2
N_INNER = 5
TIME_TOLERANCE_MS = TOLERANCE_MS   # protocol: 20 ms; the same rule (with EPS_MS) as the offset-frames build
EXCLUSION_STOP = 0.05
PLACEBO_LIMIT = 0.10       # |pi| above this with a CI excluding 0 fails the placebo
NEG_LIMIT = 0.10           # |Delta_neg| or |D_neg(0)| above this fraction of Delta_F with a CI excluding 0 fails
ON_TARGET = "Shots on target"
STRATA = ("all", "rebound_chain", "test_split", "no_cut")


class Reading(NamedTuple):
    encoder: str
    source: str      # a table column, or "probe" for a linear probe on the encoder's embeddings


READINGS = {"F": Reading("dinov2_frozen", "probe"), "G": Reading("g_combo", "on_target_logit"),
            "X": Reading("g_combo", "xg_logit_raw"), "X0": Reading("g_combo_no_head", "xg_logit_raw"),
            "Gp": Reading("g_combo", "probe"), "G0p": Reading("g_combo_no_head", "probe")}
AUC_READINGS = ("F", "G", "Gp", "G0p")     # protocol: AUC for every reading except X and X0, uncentred logits
EVENT_READINGS = ("X", "X0")
POSITIVE_CONTROL_READINGS = ("F", "G", "X")
assert all(o in OFFSET_INDEX for o in PRE + DENSE + POST + PLACEBO_FIT + PLACEBO_TEST)


# ---------------------------------------------------------------- population

def probe_shots(table: pd.DataFrame, index: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The shots of the probe and the exclusions per offset.

    Returns (shots, exclusions). `shots`: one row per non-penalty shot of the table that is a complete case, sorted by
    shot_id, with game, half, position_ms, split, goal, y (1 = on target), in_population (complete case without a goal)
    and no_cut (no frame of the 30 has a cut flag). Complete case (protocol, "Populacja"): a frame at every grid offset
    whose decoded time is within `TIME_TOLERANCE_MS` of the nominal offset. Goal shots are kept (not in the population)
    so that ticket 10 can re-score them under other attribution rules.
    `exclusions`: per offset, the shots of the protocol population (non-penalty, no goal) without a usable frame there,
    plus a final row "any" with the shots excluded at one or more offsets."""
    ix = index.drop_duplicates("shot_id")
    ix = ix[~ix.is_penalty.astype(bool)]
    frames = table.drop_duplicates(["shot_id", "offset_ms"])
    if set(frames.offset_ms.unique()) != set(GRID):
        raise ValueError("the table does not hold exactly the protocol grid")
    unknown = set(frames.shot_id) - set(ix.shot_id)
    if unknown:
        raise ValueError(f"{len(unknown)} table shots are not non-penalty index shots, e.g. {sorted(unknown)[0]}")
    usable = (frames.frame_status.isin(FRAME_STATUSES_WITH_FRAME)
              & ((frames.decoded_relative_ms - frames.offset_ms).abs() <= TIME_TOLERANCE_MS + EPS_MS))
    frames = frames.assign(usable=usable.to_numpy(), cut=frames.cut_between_anchor.fillna(False).astype(bool).to_numpy())
    per_shot = frames.groupby("shot_id").agg(n=("usable", "size"), usable=("usable", "sum"), cut=("cut", "any"))
    complete = per_shot.index[(per_shot.n == len(GRID)) & (per_shot.usable == len(GRID))]

    shots = ix[ix.shot_id.isin(complete)].sort_values("shot_id").reset_index(drop=True)
    shots = pd.DataFrame({"shot_id": shots.shot_id, "game": shots.game, "half": shots.half.astype(int),
                          "position_ms": shots.position_ms.astype(int), "split": shots.split,
                          "goal": shots.goal.astype(bool), "y": shots.label.eq(ON_TARGET).astype(np.int8)})
    shots["in_population"] = ~shots.goal
    shots["no_cut"] = ~shots.shot_id.map(per_shot.cut).astype(bool)

    population = ix[~ix.goal.astype(bool)]
    in_table = frames[frames.shot_id.isin(population.shot_id)]
    if in_table.shot_id.nunique() != len(population):
        raise ValueError("population shots are missing from the table")
    rows = []
    for o in GRID:
        at = in_table[in_table.offset_ms == o]
        rows.append({"offset_ms": o, "population": len(population), "excluded": int((~at.usable).sum())})
    excluded = len(population) - int(shots.in_population.sum())
    rows.append({"offset_ms": "any", "population": len(population), "excluded": excluded})
    exclusions = pd.DataFrame(rows)
    exclusions["fraction"] = exclusions.excluded / exclusions.population
    return shots, exclusions


def stop_rule_triggered(exclusions: pd.DataFrame) -> bool:
    """Protocol: more than 5% of the population excluded stops the analysis."""
    return bool(exclusions.loc[exclusions.offset_ms.eq("any"), "fraction"].iloc[0] > EXCLUSION_STOP)


def pivot_offsets(table: pd.DataFrame, shot_ids: Sequence[str], encoder: str, column: str) -> np.ndarray:
    """(n_shots, 30) matrix of `column` for `encoder`, rows in `shot_ids` order, columns in `GRID` order."""
    part = table[table.encoder == encoder]
    wide = part.pivot(index="shot_id", columns="offset_ms", values=column)
    return wide.reindex(index=list(shot_ids), columns=list(GRID)).to_numpy()


def table_reading(table: pd.DataFrame, shots: pd.DataFrame, name: str) -> np.ndarray:
    """(n, 30) logits of a table reading (G, X, X0); NaN must not occur for complete cases."""
    reading = READINGS[name]
    if reading.source == "probe":
        raise ValueError(f"{name} is a probe reading")
    z = pivot_offsets(table, shots.shot_id, reading.encoder, reading.source).astype(np.float64)
    if not np.isfinite(z).all():
        raise ValueError(f"reading {name}: non-finite logits on complete-case shots")
    return z


# ---------------------------------------------------------------- folds

def outer_folds(shots: pd.DataFrame, folds: pd.DataFrame) -> np.ndarray:
    """Home-team fold of each shot's match for train/valid shots, -1 for test shots (scored by every fold probe)."""
    fold_of = folds.drop_duplicates("game").set_index("game").fold
    test = shots.split.eq("test").to_numpy()
    if shots.game[test].isin(fold_of.index).any():
        raise ValueError("a test match is in the folds file")
    fold = shots.game.map(fold_of)
    if fold[~test].isna().any():
        raise ValueError("a train/valid match has no fold")
    return np.where(test, -1, fold.fillna(-1)).astype(int)


def inner_folds(games: pd.Series, fold_matches: Sequence[str]) -> np.ndarray:
    """Protocol (Gp, G0p): the matches of outer fold k (`fold_matches`, every match of the fold in the folds file, so
    matches without an analysed shot keep their position) sorted lexicographically, inner fold = position mod 5."""
    pos = {g: i % N_INNER for i, g in enumerate(sorted(set(fold_matches)))}
    if not set(games).issubset(pos):
        raise ValueError("a shot's match is not in its outer fold")
    return games.map(pos).to_numpy().astype(int)


# ---------------------------------------------------------------- linear probes

@dataclass
class ProbeResult:
    """Out-of-fold logits of fixed-in-time probes for every C of the grid.

    logits     (n_C, n_shots, 30) float64; shots with fold -1 get the mean over the fold probes
    cv_loss    (n_folds, n_C) mean held-out log-loss on the PRE and POST rows of the population shots of each fold
    n_iter     (n_folds, n_C) lbfgs iterations (== MAX_ITER means not converged)
    c_grid     the C values"""

    logits: np.ndarray
    cv_loss: np.ndarray
    n_iter: np.ndarray
    c_grid: tuple[float, ...]

    def selected(self) -> int:
        """Index of the C with the smallest mean held-out log-loss; ties go to the smaller C (grid is ascending)."""
        mean = self.cv_loss.mean(0)
        if not np.isfinite(mean).all():
            raise ValueError("non-finite cross-validated log-loss (a fold without population shots?)")
        return int(np.argmin(mean))


def fit_probe(x: np.ndarray, y: np.ndarray, c_grid: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scaler + L2 logistic regression per C on training rows. Returns the probes folded into raw-feature space,
    (weights (d, n_C), bias (n_C,), n_iter (n_C,)), so that logit = x @ weights + bias."""
    scaler = StandardScaler().fit(x)
    xs = scaler.transform(x)
    weights, bias, n_iter = [], [], []
    for c in c_grid:
        model = LogisticRegression(penalty="l2", C=c, solver="lbfgs", max_iter=MAX_ITER).fit(xs, y)
        w = model.coef_[0] / scaler.scale_
        weights.append(w)
        bias.append(model.intercept_[0] - scaler.mean_ @ w)
        n_iter.append(int(model.n_iter_[0]))
    return np.stack(weights, 1), np.array(bias), np.array(n_iter)


def probe_readout(load: Callable[[np.ndarray, int], np.ndarray], y: np.ndarray, train: np.ndarray, fold: np.ndarray,
                  c_grid: Sequence[float] = C_GRID) -> ProbeResult:
    """Fixed-in-time probes, one per fold value >= 0, for every C.

    `load(shot_rows, offset_index)` returns the features (len(shot_rows), d) of those shots at one grid offset.
    `train`: shots allowed as training rows (the population); `fold`: fold per shot, -1 for shots scored by the mean of
    all fold probes. The probe of fold k trains on the PRE+POST frames of `train` shots with fold >= 0 and != k, and
    scores every shot with fold k at all 30 offsets."""
    c_grid = tuple(c_grid)
    fold_ids = sorted(int(f) for f in np.unique(fold) if f >= 0)
    if len(fold_ids) < 2:
        raise ValueError("a probe needs at least two folds")
    n = len(y)
    train_shots = np.flatnonzero(train & (fold >= 0))
    train_x = np.concatenate([load(train_shots, OFFSET_INDEX[o]).astype(np.float32) for o in PROBE_TRAIN])
    train_y = np.tile(y[train_shots], len(PROBE_TRAIN))
    train_fold = np.tile(fold[train_shots], len(PROBE_TRAIN))

    probes = []   # per fold: (weights, bias)
    n_iter = np.zeros((len(fold_ids), len(c_grid)), int)
    for k, f in enumerate(fold_ids):
        rows = train_fold != f
        if len(np.unique(train_y[rows])) < 2:
            raise ValueError(f"fold {f}: training rows hold one class only")
        w, b, n_iter[k] = fit_probe(train_x[rows], train_y[rows], c_grid)
        probes.append((w, b))
    del train_x

    logits = np.full((len(c_grid), n, len(GRID)), np.nan)
    weights = np.concatenate([w for w, _ in probes], 1)            # (d, n_folds * n_C)
    bias = np.concatenate([b for _, b in probes])
    every = np.arange(n)
    for j in range(len(GRID)):
        z = (load(every, j).astype(np.float64) @ weights + bias).reshape(n, len(fold_ids), len(c_grid))
        for k, f in enumerate(fold_ids):
            own = fold == f
            logits[:, own, j] = z[own, k, :].T
        pooled = fold < 0
        if pooled.any():
            logits[:, pooled, j] = z[pooled].mean(1).T

    cv_loss = np.zeros((len(fold_ids), len(c_grid)))
    cols = [OFFSET_INDEX[o] for o in PROBE_TRAIN]
    for k, f in enumerate(fold_ids):
        held = train & (fold == f)
        z = logits[:, held][:, :, cols]                             # (n_C, shots, 9)
        yy = y[held][None, :, None]
        cv_loss[k] = (np.logaddexp(0, z) - yy * z).mean((1, 2))     # log-loss of sigmoid(z)
    return ProbeResult(logits, cv_loss, n_iter, c_grid)


def embedding_loader(embeddings: np.ndarray, refs: np.ndarray) -> Callable[[np.ndarray, int], np.ndarray]:
    """`load` for `probe_readout` over a (memory-mapped) embedding file and an (n_shots, 30) matrix of row refs.
    Rows are read in ascending order (sequential on the memory map) and returned in the asked order."""
    def load(shot_rows: np.ndarray, j: int) -> np.ndarray:
        r = refs[shot_rows, j]
        if (r < 0).any():
            raise ValueError("a shot has no embedding at this offset")
        order = np.argsort(r, kind="stable")
        out = np.empty((len(r), embeddings.shape[1]), np.float32)
        out[order] = embeddings[r[order]]
        return out
    return load


def nested_probe_readout(load: Callable[[np.ndarray, int], np.ndarray], shots: pd.DataFrame, fold: np.ndarray,
                         fold_matches: dict[int, Sequence[str]], c_grid: Sequence[float] = C_GRID) -> ProbeResult:
    """Gp / G0p: within each outer fold k, inner folds over its matches (`inner_folds` on `fold_matches[k]`), probes
    trained and scored on the shots of fold k only (their embeddings come from the fold-k model). Shots with fold -1 (test) stay NaN.
    cv_loss stacks the inner folds of every outer fold, so the selected C is one per encoder."""
    n = len(shots)
    logits = np.full((len(c_grid), n, len(GRID)), np.nan)
    losses, iters = [], []
    for k in sorted(int(f) for f in np.unique(fold) if f >= 0):
        rows = np.flatnonzero(fold == k)
        sub = shots.iloc[rows]

        def load_k(shot_rows: np.ndarray, j: int, rows: np.ndarray = rows) -> np.ndarray:
            return load(rows[shot_rows], j)

        res = probe_readout(load_k, sub.y.to_numpy(), sub.in_population.to_numpy(), inner_folds(sub.game, fold_matches[k]),
                            c_grid)
        logits[:, rows] = res.logits
        losses.append(res.cv_loss)
        iters.append(res.n_iter)
    return ProbeResult(logits, np.concatenate(losses), np.concatenate(iters), tuple(c_grid))


def permute_within_match(shots: pd.DataFrame, seed: int = NEG_SEED) -> np.ndarray:
    """Negative-control labels (protocol): `default_rng(seed)`, matches in sorted order, in each match a permutation of
    the labels of its shots sorted by shot_id. `shots` must hold only the population (shot_id, game, y)."""
    rng = np.random.default_rng(seed)
    y = shots.y.to_numpy()
    out = y.copy()
    for _, part in shots.assign(_row=np.arange(len(shots))).sort_values(["game", "shot_id"]).groupby("game", sort=True):
        rows = part._row.to_numpy()
        out[rows] = rng.permutation(y[rows])
    return out


# ---------------------------------------------------------------- bootstrap and curves

def match_draw_counts(games: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> tuple[np.ndarray, np.ndarray]:
    """(keys, counts): `keys` = np.unique(games); counts (n_boot, M), how often each match is drawn. Same sequence as
    `report_complexity.game_draws`."""
    keys = np.unique(games)
    rng = np.random.default_rng(seed)
    counts = np.empty((n_boot, len(keys)), np.int32)
    for b in range(n_boot):
        counts[b] = np.bincount(rng.choice(len(keys), len(keys), replace=True), minlength=len(keys))
    return keys, counts


class MatchSums(NamedTuple):
    """Per-match sums of centred logits for one reading and stratum: on/off (M, O), n_on/n_off (M,)."""

    on: np.ndarray
    off: np.ndarray
    n_on: np.ndarray
    n_off: np.ndarray


def match_sums(z: np.ndarray, y: np.ndarray, match: np.ndarray, n_matches: int, centre: np.ndarray,
               member: np.ndarray) -> MatchSums:
    """Centre z (n, O) within match over the `centre` shots, then sum over the `member` shots (a subset of `centre`).
    `match`: index of each shot's match in the bootstrap keys."""
    if (member & ~centre).any():
        raise ValueError("stratum shots must be inside the centring population")
    zc_rows = np.flatnonzero(centre)
    if not np.isfinite(z[zc_rows]).all():
        raise ValueError("non-finite logits in the centring population")
    count = np.bincount(match[zc_rows], minlength=n_matches).astype(float)
    total = np.zeros((n_matches, z.shape[1]))
    np.add.at(total, match[zc_rows], z[zc_rows])
    mean = np.divide(total, count[:, None], out=np.zeros_like(total), where=count[:, None] > 0)
    rows = np.flatnonzero(member)
    zc = z[rows] - mean[match[rows]]
    on = y[rows] == 1
    s_on = np.zeros_like(total)
    s_off = np.zeros_like(total)
    np.add.at(s_on, match[rows[on]], zc[on])
    np.add.at(s_off, match[rows[~on]], zc[~on])
    return MatchSums(s_on, s_off, np.bincount(match[rows[on]], minlength=n_matches).astype(float),
                     np.bincount(match[rows[~on]], minlength=n_matches).astype(float))


def gap(sums: MatchSums, counts: np.ndarray | None = None) -> np.ndarray:
    """D(o): (O,) without counts, (B, O) with bootstrap counts (B, M). NaN where a draw has no on- or off-target shot."""
    w = np.ones((1, len(sums.n_on))) if counts is None else np.asarray(counts, float)
    n_on, n_off = w @ sums.n_on, w @ sums.n_off
    with np.errstate(invalid="ignore", divide="ignore"):
        d = (w @ sums.on) / n_on[:, None] - (w @ sums.off) / n_off[:, None]
    d[(n_on == 0) | (n_off == 0)] = np.nan
    return d[0] if counts is None else d


def line(d: np.ndarray, fit: Sequence[int] = PRE) -> tuple[np.ndarray, np.ndarray]:
    """OLS intercept and slope (per ms) through (o, D(o)) for o in `fit`, over the last axis of d (..., 30)."""
    x = np.array(fit, float)
    yv = d[..., [OFFSET_INDEX[o] for o in fit]]
    xc = x - x.mean()
    b = (yv * xc).sum(-1) / (xc ** 2).sum()
    return yv.mean(-1) - b * x.mean(), b


def trend(d: np.ndarray, fit: Sequence[int] = PRE) -> np.ndarray:
    """P(o) on the whole grid, (..., 30)."""
    a, b = line(d, fit)
    return a[..., None] + b[..., None] * np.array(GRID, float)


def jump(d: np.ndarray) -> np.ndarray:
    """Delta = mean D over POST - P(0) (positive-control jump), (...)."""
    return d[..., [OFFSET_INDEX[o] for o in POST]].mean(-1) - trend(d)[..., OFFSET_INDEX[0]]


def placebo(d: np.ndarray) -> np.ndarray:
    """pi(o) for o in PLACEBO_TEST, (..., 2): residual from the PLACEBO_FIT line in units of Delta."""
    p = trend(d, PLACEBO_FIT)
    cols = [OFFSET_INDEX[o] for o in PLACEBO_TEST]
    return (d[..., cols] - p[..., cols]) / jump(d)[..., None]


def plateau_slope(d: np.ndarray) -> np.ndarray:
    """(D(+800) - D(+400)) / Delta (reported; plateau decay inflates U)."""
    return (d[..., OFFSET_INDEX[800]] - d[..., OFFSET_INDEX[400]]) / jump(d)


def weighted_auc(score: np.ndarray, y: np.ndarray, match: np.ndarray, counts: np.ndarray | None = None,
                 chunk: int = 250) -> np.ndarray:
    """AUC of `score` for y, each shot weighted by its match's bootstrap count (ties count one half).
    Returns a scalar array without counts, (B,) with counts; NaN where a draw lacks a class."""
    order = np.argsort(score, kind="stable")
    s, pos, m = score[order], y[order] == 1, match[order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    w_all = np.ones((1, int(match.max()) + 1)) if counts is None else np.asarray(counts, float)
    out = []
    for lo in range(0, len(w_all), chunk):
        w = w_all[lo:lo + chunk][:, m]
        wp, wn = w * pos, w * ~pos
        gp, gn = np.add.reduceat(wp, starts, 1), np.add.reduceat(wn, starts, 1)
        below = np.cumsum(gn, 1) - gn
        den = wp.sum(1) * wn.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append(np.where(den > 0, (gp * (below + 0.5 * gn)).sum(1) / den, np.nan))
    res = np.concatenate(out)
    return res[0] if counts is None else res


# ---------------------------------------------------------------- summaries

def interval(draws: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """95% percentile interval over the draws axis (0), skipping NaN draws; returns (lo, hi, skipped draws)."""
    draws = np.asarray(draws, float)
    keep = ~np.isnan(draws.reshape(len(draws), -1)).any(1)
    if not keep.any():
        nan = np.full(draws.shape[1:], np.nan)
        return nan, nan, int(len(draws))
    lo, hi = np.percentile(draws[keep], [2.5, 97.5], axis=0)
    return lo, hi, int((~keep).sum())


def excludes_zero(lo: float, hi: float) -> bool:
    return bool(lo > 0 or hi < 0)


def placebo_gate(point: np.ndarray, draws: np.ndarray) -> dict:
    """Fails when for one of the two offsets |pi| > PLACEBO_LIMIT and its 95% interval excludes 0."""
    lo, hi, skipped = interval(draws)
    rows = {}
    failed = False
    for k, o in enumerate(PLACEBO_TEST):
        bad = abs(point[k]) > PLACEBO_LIMIT and excludes_zero(lo[k], hi[k])
        failed |= bad
        rows[f"pi_{o}"] = {"value": float(point[k]), "lo": float(lo[k]), "hi": float(hi[k]), "fails": bool(bad)}
    return rows | {"passed": not failed, "skipped_draws": skipped}


def negative_gate(stats: dict[str, tuple[float, np.ndarray]], delta_f: float) -> dict:
    """Fails when for Delta_neg or D_neg(0) |value| > NEG_LIMIT * Delta_F and its 95% interval excludes 0.
    With Delta_F <= 0 the magnitude condition always holds; the positive control fails first in that case."""
    rows = {}
    failed = False
    for name, (value, draws) in stats.items():
        lo, hi, skipped = interval(draws)
        bad = abs(value) > NEG_LIMIT * delta_f and excludes_zero(float(lo), float(hi))
        failed |= bad
        rows[name] = {"value": float(value), "lo": float(lo), "hi": float(hi), "limit": NEG_LIMIT * delta_f,
                      "fails": bool(bad), "skipped_draws": skipped}
    return rows | {"passed": not failed}


def positive_jump(point: float, draws: np.ndarray) -> dict:
    """First condition of the positive control: lower 95% bound of Delta > 0. The second condition (half-width of the
    U interval <= 0.15) needs the estimator U and is evaluated by ticket 10."""
    lo, hi, skipped = interval(draws)
    return {"delta": float(point), "lo": float(lo), "hi": float(hi), "delta_lower_positive": bool(lo > 0),
            "skipped_draws": skipped}


def curve_frame(reading: str, stratum: str, point: np.ndarray, draws: np.ndarray) -> pd.DataFrame:
    """One row per offset: D, its interval, the PRE trend P and the trend-relative gap D - P with their intervals."""
    p_point, p_draws = trend(point), trend(draws)
    d_lo, d_hi, skipped = interval(draws)
    p_lo, p_hi, _ = interval(p_draws)
    r_lo, r_hi, _ = interval(draws - p_draws)
    return pd.DataFrame({"reading": reading, "stratum": stratum, "offset_ms": GRID, "D": point, "D_lo": d_lo,
                         "D_hi": d_hi, "P": p_point, "P_lo": p_lo, "P_hi": p_hi, "D_minus_P": point - p_point,
                         "D_minus_P_lo": r_lo, "D_minus_P_hi": r_hi, "skipped_draws": skipped})


# ---------------------------------------------------------------- the analysis

def strata_masks(shots: pd.DataFrame) -> dict[str, np.ndarray]:
    """Population masks of the protocol's strata; `shots` needs in_population, split, no_cut and rebound_chain (every
    shot of a ticket-03 rebound chain of two or more shots)."""
    pop = shots.in_population.to_numpy(bool)
    return {"all": pop, "rebound_chain": pop & shots.rebound_chain.to_numpy(bool),
            "test_split": pop & shots.split.eq("test").to_numpy(), "no_cut": pop & shots.no_cut.to_numpy(bool)}


@dataclass
class Curves:
    """D per (reading, stratum): point (30,), draws (B, 30), and the per-match sums they come from."""

    point: dict[tuple[str, str], np.ndarray]
    draws: dict[tuple[str, str], np.ndarray]
    sums: dict[tuple[str, str], MatchSums]


def match_positions(shots: pd.DataFrame, keys: np.ndarray) -> np.ndarray:
    """Index of each shot's match in the bootstrap keys; -1 for a match outside the keys, which only shots outside the
    population may have (e.g. a goal shot in a match whose population shots were all excluded)."""
    games = shots.game.to_numpy()
    pos = np.minimum(np.searchsorted(keys, games), len(keys) - 1)
    pos = np.where(keys[pos] == games, pos, -1)
    if (pos[shots.in_population.to_numpy(bool)] < 0).any():
        raise ValueError("a population shot's match is not among the bootstrap keys")
    return pos


def readability_curves(readings: dict[str, tuple[np.ndarray, np.ndarray]], shots: pd.DataFrame, keys: np.ndarray,
                       counts: np.ndarray, strata: dict[str, Sequence[str]] | None = None) -> Curves:
    """D curves for every reading and stratum on the shared bootstrap counts.

    `readings[name] = (z (n, 30), y (n,))` over the rows of `shots` (y lets the negative control use permuted labels);
    shots without the reading are NaN. The centring population of a reading: population shots with a finite logit.
    `strata[name]` limits the strata of a reading (default: all of `STRATA`)."""
    masks = strata_masks(shots)
    match = match_positions(shots, keys)
    out = Curves({}, {}, {})
    for name, (z, y) in readings.items():
        centre = masks["all"] & np.isfinite(z).all(1)
        for stratum in (strata or {}).get(name, STRATA):
            sums = match_sums(z, y, match, len(keys), centre, masks[stratum] & centre)
            out.sums[name, stratum] = sums
            out.point[name, stratum] = gap(sums)
            out.draws[name, stratum] = gap(sums, counts)
    return out


def auc_table(readings: dict[str, tuple[np.ndarray, np.ndarray]], shots: pd.DataFrame, keys: np.ndarray,
              counts: np.ndarray, strata: dict[str, Sequence[str]] | None = None) -> pd.DataFrame:
    """On-target AUC by offset on uncentred logits, per reading and stratum, with match-bootstrap intervals."""
    masks = strata_masks(shots)
    match = match_positions(shots, keys)
    rows = []
    for name, (z, y) in readings.items():
        has = np.isfinite(z).all(1)
        for stratum in (strata or {}).get(name, STRATA):
            m = masks[stratum] & has
            if not m.any():
                continue
            for j, o in enumerate(GRID):
                point = float(weighted_auc(z[m, j], y[m], match[m]))
                lo, hi, skipped = interval(weighted_auc(z[m, j], y[m], match[m], counts))
                rows.append({"reading": name, "stratum": stratum, "offset_ms": o, "auc": point, "auc_lo": float(lo),
                             "auc_hi": float(hi), "skipped_draws": skipped})
    return pd.DataFrame(rows)


def curve_table(curves: Curves) -> pd.DataFrame:
    return pd.concat([curve_frame(r, s, curves.point[r, s], curves.draws[r, s]) for r, s in curves.point],
                     ignore_index=True)


def jump_table(curves: Curves) -> pd.DataFrame:
    """Delta and the plateau slope per reading and stratum; `delta_lower_positive` is the first positive-control
    condition. `role` marks the rows the protocol's gates read (F, G, X in the main population); the protocol reports
    the plateau slope for F only, the other rows are descriptive."""
    rows = []
    for (r, s), point in curves.point.items():
        draws = curves.draws[r, s]
        pj = positive_jump(float(jump(point)), jump(draws))
        lo, hi, _ = interval(plateau_slope(draws))
        role = "gate_input" if s == "all" and r in POSITIVE_CONTROL_READINGS else "descriptive"
        rows.append({"reading": r, "stratum": s, "role": role, **pj, "plateau_slope": float(plateau_slope(point)),
                     "plateau_slope_lo": float(lo), "plateau_slope_hi": float(hi)})
    return pd.DataFrame(rows)


def control_record(curves: Curves, negative: str = "F_neg") -> dict:
    """Placebo and negative control (both full gates) and the first positive-control condition, main population."""
    d_f, d_f_draws = curves.point["F", "all"], curves.draws["F", "all"]
    delta_f = float(jump(d_f))
    d_neg, d_neg_draws = curves.point[negative, "all"], curves.draws[negative, "all"]
    zero = OFFSET_INDEX[0]
    return {"placebo": placebo_gate(placebo(d_f), placebo(d_f_draws)),
            "negative": negative_gate({"delta_neg": (float(jump(d_neg)), jump(d_neg_draws)),
                                       "d_neg_0": (float(d_neg[zero]), d_neg_draws[:, zero])}, delta_f),
            "positive_jump": {r: positive_jump(float(jump(curves.point[r, "all"])), jump(curves.draws[r, "all"]))
                              for r in POSITIVE_CONTROL_READINGS if (r, "all") in curves.point}}
