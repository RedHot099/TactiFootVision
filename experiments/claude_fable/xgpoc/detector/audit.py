"""Audit of detected positions against StatsBomb freeze frames, config choice and gate (xg-detector tickets 02 and 09;
spec, Audit and story 13-15). Diagnostic only: nothing here is an input of any model.

Audit 1 (ticket 02) ran on C1-C4; audit 2 (ticket 09), the last one, runs on D1-D4 as declared by ticket 08
(`research/xg_detector_audit2_configs.json`, whose sha256 the script checks), with audit 1's choice C4 audited next to
them as a reference: same population, same draws, never a candidate of the choice.

This module and `scripts/xgd_audit.py` are the only detector code that reads StatsBomb data or the pairing. The model
and feature paths stay behind `guard.no_statsbomb_reads`, whose pattern also refuses this audit's outputs
(`results/xgd_audit`), because they hold StatsBomb positions. No goal or outcome column is read anywhere: every table
read names its columns (`PAIR_COLUMNS`, `SHOT_COLUMNS`, `FRAME_READ_COLUMNS`), none of which carries an outcome, and the
config choice and the gate see only coverages and position errors.

Population: the paired non-penalty shots (`paired` and not `is_penalty`), at the frame of the SoccerNet anchor (t0).

Truth, in aligned metres (`conventions.statsbomb_to_metres`; StatsBomb always attacks x = 120, which is the aligned
frame): the shot location (`sb_x`, `sb_y`); from the freeze frame the opponents (`teammate` false), the goalkeeper
being the first opponent whose position is "Goalkeeper" and the outfield opponents the others, as in xg-vendor's
probe targets. `features.geometry` on them gives the defenders-in-cone count, with the detector's own definitions.
A shot without a freeze frame has no goalkeeper or cone truth.

Detected, per configuration: the t0 feature rows replayed from the cached frames (`extract.replay_features`; the
script checks them against the cached feature table): shot location, goalkeeper (detected, or the proxy), cone count
and flags. `kp_conf` is the keypoint instance confidence of the frame whose homography placed the shot location (t0,
or the fallback frame whose ball was used); `kp_conf_t0` that of t0, whose homography placed the players.

Quantities, each with its coverage (share of the whole population that has the detected quantity):
    shot_location     error |detected - StatsBomb| in metres, on every covered shot
    gk_any            goalkeeper position error, detected goalkeeper or proxy, on covered shots with a StatsBomb GK
    gk_detected       the same for a detected goalkeeper only
    gk_proxy          ... for the proxy only, and split by `gk_proxy_same_team`: gk_proxy_same (the proxy is in the
    gk_proxy_same/    shooter's kit cluster) and gk_proxy_other; a proxy whose team is unknown is in neither
    gk_proxy_other
    def_cone          defenders-in-cone count: exact agreement, agreement of >= 1 and >= 2, mean difference
                      (detected - StatsBomb), on covered shots with a freeze frame; confusion tables separately
    shot_constant     baselines that ignore the frame: one constant guess for every shot, the coordinate-wise median of
    gk_constant       the StatsBomb positions over the whole population (shot location; goalkeeper, over the shots with
                      a StatsBomb goalkeeper), scored on exactly the shots compared for shot_location / gk_any. An
                      in-sample oracle guess: a detector worse than it carries no position information.

Subsets (`SUBSETS`, each a part of the population): all; ball / flag, the shot location from a ball (its own or a
fallback ball) or from the shooter flagged `shooter_by_goal` (owner decision Q23); fallback, the ball of an earlier
frame; kp05, every homography behind the compared positions has instance confidence >= 0.5 (Q25: the location's frame,
and t0 too for the goalkeeper and the cone).

Statistics: coverage, n (shots compared), median, p90, lt1/lt2/lt5 (share of compared shots under 1/2/5 m); for the
shot location also the median signed errors along x and y (dx, dy) and `median_mirror_y`, the median error after
mirroring the detected y (a convention check: a sign error in y would make it the smaller one); for the shot location
and the goalkeeper (gk_any) the Pearson and the Spearman correlations of the detected with the StatsBomb x and y (corr_x,
corr_y, rank_corr_x, rank_corr_y; Pearson is dominated by the few positions a bad homography throws far off the pitch,
which the ranks are not); for the shot location `off_pitch`, the share of compared shots placed outside the pitch
(x > 105, y < 0 or y > 68 in aligned metres); for the constant baselines the constant itself (const_x, const_y, no
interval). Quantiles are the
inverted CDF (`np.quantile(method="inverted_cdf")`). Intervals: 95% percentile intervals of a match bootstrap
(`N_BOOT` draws, seed `SEED`, over the population's matches, the same draws for every configuration and subset).

The errors include the annotation-timing noise of both sources (about 1 s, in which players move 3-6 m). `timing_table`
breaks the errors down by xg-vendor's timing residual, descriptively; nothing is selected on it.

Config choice (frozen in the spec), among the candidate configurations only (audit 2: D1-D4, not the reference C4):
among the configurations whose shot-location coverage is >= 70% of the population,
the lowest median shot-location error; ties go to the higher coverage, then to the configuration name. When none
reaches 70%, the same rule over all configurations chooses the configuration for approach 1, and the gate is STOP.

Gate (frozen, spec story 15), on the chosen configuration's point estimates: median shot-location error <= 3 m with
coverage >= 70%, and goalkeeper information (detected or proxy) for >= 50% of the population. PASS or STOP.
"""

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from xgpoc.detector.conventions import LENGTH, WIDTH, statsbomb_to_metres
from xgpoc.detector.extract import feature_differences
from xgpoc.detector.features import geometry
from xgpoc.paths import STATSBOMB

EXPERIMENT = Path(__file__).resolve().parents[2]
PAIRS = EXPERIMENT / "results/xgv_pairing/pairs.parquet"
SHOTS = STATSBOMB / "tables/shots.parquet"
PAIR_COLUMNS = ["shot_id", "game", "paired", "is_penalty", "sb_event_id", "sb_x", "sb_y", "residual_ms"]
SHOT_COLUMNS = ["event_id", "freeze_frame_json"]
FRAME_READ_COLUMNS = ["shot_id", "offset_ms", "frame_path", "readable", "homography", "kp_xy", "kp_conf", "kp_instance_conf",
                      "det_class", "det_conf", "det_xyxy", "det_pitch_xy", "det_team", "ball_conf", "ball_xyxy",
                      "ball_pitch_xy"]
GOALKEEPER_POSITION = "Goalkeeper"

N_BOOT, SEED = 5000, 0
THRESHOLDS_M = (1.0, 2.0, 5.0)
KP_STRICT = 0.5
COVERAGE_MIN, MEDIAN_MAX_M, GK_COVERAGE_MIN = 0.70, 3.0, 0.50
TIMING_BINS_MS = (0.0, 50.0, 200.0, np.inf)

GK_KINDS = {"gk_any": "gk_ok == 1", "gk_detected": "gk_detected == 1", "gk_proxy": "gk_proxy == 1",
            "gk_proxy_same": "gk_proxy == 1 and gk_proxy_same_team == 1",
            "gk_proxy_other": "gk_proxy == 1 and gk_proxy_same_team == 0"}
SUBSETS = ("all", "ball", "flag", "fallback", "kp05")


# ------------------------------------------------------------------ inputs
def population(pairs: pd.DataFrame) -> pd.DataFrame:
    """The paired non-penalty shots, sorted by shot id, with the `PAIR_COLUMNS` (no other column is accepted)."""
    extra = sorted(set(pairs.columns) - set(PAIR_COLUMNS))
    if extra:
        raise ValueError(f"the audit reads only {PAIR_COLUMNS}; got also {extra}")
    keep = pairs.paired.astype(bool) & ~pairs.is_penalty.astype(bool)
    out = pairs[keep].sort_values("shot_id").reset_index(drop=True)
    if out.shot_id.duplicated().any() or out.sb_event_id.duplicated().any():
        raise ValueError("a population shot or StatsBomb event appears twice")
    return out


def read_population(path) -> pd.DataFrame:
    return population(pd.read_parquet(path, columns=PAIR_COLUMNS))


def read_freeze_frames(path, event_ids) -> pd.Series:
    """Freeze-frame JSON by StatsBomb event id, for the given events (each must exist)."""
    shots = pd.read_parquet(path, columns=SHOT_COLUMNS).set_index("event_id").freeze_frame_json
    missing = sorted(set(event_ids) - set(shots.index))
    if missing:
        raise ValueError(f"{len(missing)} paired events are not in {path}, e.g. {missing[0]}")
    return shots.loc[list(event_ids)]


def truth(pop: pd.DataFrame, freeze_frames: pd.Series) -> pd.DataFrame:
    """StatsBomb positions in aligned metres, one row per population shot (index shot_id)."""
    rows = []
    for shot in pop.itertuples(index=False):
        location = statsbomb_to_metres([shot.sb_x, shot.sb_y])
        frame = freeze_frames.get(shot.sb_event_id)
        players = json.loads(frame) if isinstance(frame, str) else []
        row = {"shot_id": shot.shot_id, "sb_shot_x": location[0], "sb_shot_y": location[1], "sb_has_ff": float(bool(players)),
               "sb_gk_x": np.nan, "sb_gk_y": np.nan, "cone_sb": np.nan}
        if players:
            opponents = [p for p in players if not p["teammate"]]
            is_gk = np.array([(p.get("position") or {}).get("name") == GOALKEEPER_POSITION for p in opponents], bool)
            xy = statsbomb_to_metres(np.array([p["location"][:2] for p in opponents], float).reshape(-1, 2))
            gk, outfield = (xy[is_gk][0] if is_gk.any() else None), xy[~is_gk]
            g = geometry(location, defenders_xy=outfield, opponents_xy=xy, goalkeeper_xy=gk)
            row["cone_sb"] = g["ff_def_cone"]
            if gk is not None:
                row["sb_gk_x"], row["sb_gk_y"] = gk
        rows.append(row)
    return pd.DataFrame(rows).set_index("shot_id")


def detected(features: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """Detected positions and flags of the t0 feature rows (index shot_id), with the keypoint instance confidences."""
    conf = frames.set_index(["shot_id", "offset_ms"]).kp_instance_conf
    t0 = features[features.offset_ms == 0].set_index("shot_id")
    source = t0.ball_offset_ms.fillna(0).astype(int)
    keys = list(zip(t0.index, source, strict=True))
    out = pd.DataFrame({
        "shot_ok": t0.shot_ok, "shot_from_ball": t0.shot_from_ball, "shooter_by_goal": t0.shooter_by_goal,
        "ball_fallback": t0.ball_fallback, "ball_offset_ms": t0.ball_offset_ms, "homography_ok": t0.homography_ok,
        "gk_ok": t0.gk_ok, "gk_detected": t0.gk_detected, "gk_proxy": t0.gk_proxy, "gk_proxy_same_team": t0.gk_proxy_same_team,
        "det_shot_x": t0.shot_x, "det_shot_y": t0.shot_y, "det_gk_x": t0.gk_x, "det_gk_y": t0.gk_y, "cone_det": t0.ff_def_cone,
        "kp_conf": conf.reindex(keys).to_numpy(), "kp_conf_t0": conf.reindex([(s, 0) for s in t0.index]).to_numpy()},
        index=t0.index)
    return out


def compare(pop: pd.DataFrame, sb: pd.DataFrame, det: pd.DataFrame, config: str) -> pd.DataFrame:
    """One row per population shot: detected and StatsBomb positions, errors in metres, flags. A shot without a
    detected row (not extracted) counts as not covered."""
    d = det.reindex(pop.shot_id)
    for flag in ("shot_ok", "shot_from_ball", "shooter_by_goal", "ball_fallback", "homography_ok", "gk_ok", "gk_detected",
                 "gk_proxy"):
        d[flag] = d[flag].fillna(0.0)
    t = sb.reindex(pop.shot_id)
    out = pd.DataFrame({"config": config, "shot_id": pop.shot_id.to_numpy(), "game": pop.game.to_numpy(),
                        "residual_ms": pop.residual_ms.to_numpy()})
    for frame in (d, t):
        for column in frame.columns:
            out[column] = frame[column].to_numpy()
    dx, dy = out.det_shot_x - out.sb_shot_x, out.det_shot_y - out.sb_shot_y
    out["shot_dx"], out["shot_dy"], out["shot_err"] = dx, dy, np.hypot(dx, dy)
    out["shot_err_mirror_y"] = np.hypot(dx, (WIDTH - out.det_shot_y) - out.sb_shot_y)
    out["gk_err"] = np.hypot(out.det_gk_x - out.sb_gk_x, out.det_gk_y - out.sb_gk_y)
    return out


# ------------------------------------------------------------------ bootstrap
class Draws(NamedTuple):
    """Match-bootstrap counts (B, M) over the population's sorted matches `keys`; `match` is each row's position."""

    keys: np.ndarray
    counts: np.ndarray
    match: np.ndarray

    def weights(self, mask: np.ndarray) -> np.ndarray:
        return self.counts[:, self.match[mask]]


def draws(games: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> Draws:
    """The match draws of `xgpoc.probe.readability.match_draw_counts` (rows of one match share its count)."""
    from xgpoc.probe.readability import match_draw_counts

    keys, counts = match_draw_counts(np.asarray(games), n_boot, seed)
    return Draws(keys, counts, np.searchsorted(keys, games))


def _interval(boot: np.ndarray) -> tuple[float, float]:
    boot = boot[np.isfinite(boot)]
    return (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))) if len(boot) else (np.nan, np.nan)


def share(num: np.ndarray, den: np.ndarray, d: Draws) -> tuple[float, float, float]:
    """Share of the `den` rows that are also `num` rows, with its interval."""
    point = num[den].sum() / den.sum() if den.sum() else np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = d.weights(num & den).sum(axis=1) / d.weights(den).sum(axis=1)
    return (float(point), *_interval(boot))


def quantiles(values: np.ndarray, mask: np.ndarray, qs, d: Draws) -> list[tuple[float, float, float]]:
    """Inverted-CDF quantiles of `values[mask]` with their intervals (each draw weighs a row by its match's count)."""
    v = values[mask]
    if not len(v):
        return [(np.nan, np.nan, np.nan) for _ in qs]
    order = np.argsort(v, kind="stable")
    v = v[order]
    cum = np.cumsum(d.weights(mask)[:, order], axis=1)
    total = cum[:, -1]
    out = []
    for q in qs:
        point = float(np.quantile(v, q, method="inverted_cdf"))
        idx = np.minimum((cum < q * total[:, None] - 1e-9).sum(axis=1), len(v) - 1)
        boot = np.where(total > 0, v[idx], np.nan)
        out.append((point, *_interval(boot)))
    return out


def _weighted_pearson(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Pearson correlation of the columns x, y ((n,) or (B, n)) with row weights w (B, n), one per draw."""
    with np.errstate(invalid="ignore", divide="ignore"):       # a constant column has no correlation (NaN)
        total = w.sum(axis=1, keepdims=True)
        cx, cy = x - (w * x).sum(axis=1, keepdims=True) / total, y - (w * y).sum(axis=1, keepdims=True) / total
        return (w * cx * cy).sum(axis=1) / np.sqrt((w * cx**2).sum(axis=1) * (w * cy**2).sum(axis=1))


def weighted_ranks(values: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Mid-ranks (B, n) of `values` in each draw's resample, a row repeated as often as its weight: the copies of tied
    values share the mean of their positions. With unit weights these are `scipy.stats.rankdata` ranks."""
    order = np.argsort(values, kind="stable")
    v = values[order]
    start = np.r_[0, np.flatnonzero(np.diff(v) != 0) + 1]          # first sorted position of each run of equal values
    group = np.cumsum(np.r_[False, np.diff(v) != 0])
    end = np.r_[start[1:], len(v)]                                 # one past the last position of each run
    cum = np.concatenate([np.zeros((len(w), 1)), np.cumsum(w[:, order], axis=1)], axis=1)
    below, upto = cum[:, start[group]], cum[:, end[group]]
    ranks = np.empty_like(cum[:, 1:])
    ranks[:, order] = below + (upto - below + 1) / 2
    return ranks


def correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray, d: Draws, rank: bool = False) -> tuple[float, float, float]:
    """Pearson (or with `rank` Spearman) correlation of `a[mask]` with `b[mask]`, with its interval: each draw weighs a
    row by its match's count, and Spearman ranks each draw's resample afresh (`weighted_ranks`)."""
    x, y = a[mask], b[mask]
    if len(x) < 2:
        return (np.nan, np.nan, np.nan)
    w, ones = d.weights(mask).astype(float), np.ones((1, len(x)))
    if rank:
        point = _weighted_pearson(weighted_ranks(x, ones), weighted_ranks(y, ones), ones)[0]
        boot = _weighted_pearson(weighted_ranks(x, w), weighted_ranks(y, w), w)
    else:
        point, boot = _weighted_pearson(x, y, ones)[0], _weighted_pearson(x, y, w)
    return (float(point), *_interval(boot))


def constant_guess(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """The coordinate-wise inverted-CDF median of the finite (x, y) rows: the constant baselines' guess."""
    finite = np.isfinite(x) & np.isfinite(y)
    return float(np.quantile(x[finite], 0.5, method="inverted_cdf")), float(np.quantile(y[finite], 0.5, method="inverted_cdf"))


def mean(values: np.ndarray, mask: np.ndarray, d: Draws) -> tuple[float, float, float]:
    v = values[mask]
    point = float(v.mean()) if len(v) else np.nan
    w = d.weights(mask)
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = (w * v).sum(axis=1) / w.sum(axis=1)
    return (point, *_interval(boot))


# ------------------------------------------------------------------ metrics
def subset_masks(table: pd.DataFrame, players: bool) -> dict[str, np.ndarray]:
    """The `SUBSETS` of a configuration's rows; `players`: the compared positions include t0's players."""
    kp = table.kp_conf.to_numpy(float) >= KP_STRICT
    if players:
        kp &= table.kp_conf_t0.to_numpy(float) >= KP_STRICT
    return {"all": np.ones(len(table), bool), "ball": table.shot_from_ball.to_numpy() == 1,
            "flag": table.shooter_by_goal.to_numpy() == 1, "fallback": table.ball_fallback.to_numpy() == 1, "kp05": kp}


def _rows(config, quantity, subset, stats: dict, n: int) -> list[dict]:
    rows = [{"config": config, "quantity": quantity, "subset": subset, "stat": "n", "value": float(n), "lo": np.nan, "hi": np.nan}]
    return rows + [{"config": config, "quantity": quantity, "subset": subset, "stat": k, "value": v[0], "lo": v[1], "hi": v[2]}
                   for k, v in stats.items()]


def error_stats(err: np.ndarray, compared: np.ndarray, d: Draws) -> dict:
    stats = dict(zip(("median", "p90"), quantiles(err, compared, (0.5, 0.9), d), strict=True))
    for t in THRESHOLDS_M:
        stats[f"lt{t:g}"] = share(err < t, compared, d)
    return stats


def _correlations(detected: list[np.ndarray], statsbomb: list[np.ndarray], mask: np.ndarray, d: Draws) -> dict:
    """corr_x, corr_y (Pearson) and rank_corr_x, rank_corr_y (Spearman) of detected with StatsBomb positions."""
    out = {}
    for prefix, rank in (("corr", False), ("rank_corr", True)):
        for axis, det, sb in zip(("x", "y"), detected, statsbomb, strict=True):
            out[f"{prefix}_{axis}"] = correlation(det, sb, mask, d, rank=rank)
    return out


def _constant(table: pd.DataFrame, prefix: str) -> tuple[np.ndarray, dict]:
    """The constant baseline of `sb_<prefix>_x/y`: each row's error and the constant as statistics."""
    sx, sy = table[f"sb_{prefix}_x"].to_numpy(float), table[f"sb_{prefix}_y"].to_numpy(float)
    cx, cy = constant_guess(sx, sy)
    return np.hypot(sx - cx, sy - cy), {"const_x": (cx, np.nan, np.nan), "const_y": (cy, np.nan, np.nan)}


def config_metrics(table: pd.DataFrame, d: Draws) -> list[dict]:
    """Every quantity x subset x statistic of one configuration's per-shot rows (the whole population, in its order)."""
    config = table.config.iloc[0]
    everyone = np.ones(len(table), bool)
    shot_const_err, shot_const = _constant(table, "shot")
    gk_const_err, gk_const = _constant(table, "gk")
    rows = []
    for subset, mask in subset_masks(table, players=False).items():
        has = mask & (table.shot_ok.to_numpy() == 1)
        coverage = {"coverage": share(has, everyone, d)}
        stats = coverage | error_stats(table.shot_err.to_numpy(float), has, d)
        extra = [table.shot_dx.to_numpy(float), table.shot_dy.to_numpy(float), table.shot_err_mirror_y.to_numpy(float)]
        for name, values in zip(("median_dx", "median_dy", "median_mirror_y"), extra, strict=True):
            stats[name] = quantiles(values, has, (0.5,), d)[0]
        det_xy = [table[f"det_shot_{axis}"].to_numpy(float) for axis in ("x", "y")]
        sb_xy = [table[f"sb_shot_{axis}"].to_numpy(float) for axis in ("x", "y")]
        stats |= _correlations(det_xy, sb_xy, has, d)
        with np.errstate(invalid="ignore"):
            stats["off_pitch"] = share((det_xy[0] > LENGTH) | (det_xy[1] < 0) | (det_xy[1] > WIDTH), has, d)
        rows += _rows(config, "shot_location", subset, stats, int(has.sum()))
        rows += _rows(config, "shot_constant", subset, coverage | error_stats(shot_const_err, has, d) | shot_const, int(has.sum()))
    has_sb_gk = table.sb_gk_x.notna().to_numpy()
    cone_det, cone_sb = table.cone_det.to_numpy(float), table.cone_sb.to_numpy(float)
    for subset, mask in subset_masks(table, players=True).items():
        for kind, rule in GK_KINDS.items():
            has = mask & table.eval(rule).to_numpy(bool)
            compared = has & has_sb_gk
            coverage = {"coverage": share(has, everyone, d)}
            stats = coverage | error_stats(table.gk_err.to_numpy(float), compared, d)
            if kind == "gk_any":
                stats |= _correlations([table[f"det_gk_{axis}"].to_numpy(float) for axis in ("x", "y")],
                                       [table[f"sb_gk_{axis}"].to_numpy(float) for axis in ("x", "y")], compared, d)
                rows += _rows(config, "gk_constant", subset, coverage | error_stats(gk_const_err, compared, d) | gk_const,
                              int(compared.sum()))
            rows += _rows(config, kind, subset, stats, int(compared.sum()))
        has = mask & np.isfinite(cone_det)
        compared = has & np.isfinite(cone_sb)
        stats = {"coverage": share(has, everyone, d), "exact": share(cone_det == cone_sb, compared, d),
                 "ge1_agree": share((cone_det >= 1) == (cone_sb >= 1), compared, d),
                 "ge2_agree": share((cone_det >= 2) == (cone_sb >= 2), compared, d),
                 "mean_diff": mean(cone_det - cone_sb, compared, d)}
        rows += _rows(config, "def_cone", subset, stats, int(compared.sum()))
    return rows


def metrics(per_shot: pd.DataFrame, d: Draws) -> pd.DataFrame:
    """`config_metrics` of every configuration (per-shot rows of each in the order of the draws' population)."""
    return pd.DataFrame([row for _, table in per_shot.groupby("config", sort=True) for row in config_metrics(table, d)])


def cone_confusion(per_shot: pd.DataFrame) -> pd.DataFrame:
    """Counts of (detected, StatsBomb) for >= 1, >= 2 and the exact count (4 meaning 4 or more), per configuration,
    on compared shots."""
    rows = []
    both = per_shot[per_shot.cone_det.notna() & per_shot.cone_sb.notna()]
    for config, table in both.groupby("config", sort=True):
        for name, det, sb in (("ge1", table.cone_det >= 1, table.cone_sb >= 1), ("ge2", table.cone_det >= 2, table.cone_sb >= 2),
                              ("exact", table.cone_det.clip(upper=4), table.cone_sb.clip(upper=4))):
            counts = pd.crosstab(det.astype(int), sb.astype(int))
            rows += [{"config": config, "table": name, "detected": int(i), "statsbomb": int(j), "shots": int(counts.loc[i, j])}
                     for i in counts.index for j in counts.columns]
    return pd.DataFrame(rows)


def timing_table(per_shot: pd.DataFrame) -> pd.DataFrame:
    """Median errors by |timing residual| (xg-vendor's per-shot residual after the half offset), descriptive only.
    Every paired shot has a residual; a missing one is refused rather than dropped from the bins."""
    if per_shot.residual_ms.isna().any():
        raise ValueError(f"{int(per_shot.residual_ms.isna().sum())} rows have no timing residual")
    bins = pd.cut(per_shot.residual_ms.abs(), TIMING_BINS_MS, right=True, include_lowest=True)
    rows = []
    for (config, interval), table in per_shot.groupby(["config", bins], sort=True, observed=False):
        located, gk = table[table.shot_ok == 1], table[(table.gk_ok == 1) & table.sb_gk_x.notna()]
        rows.append({"config": config, "residual_ms": f"{interval.left:g}-{interval.right:g}", "shots": len(table),
                     "located": len(located), "shot_err_median": located.shot_err.median(),
                     "gk_compared": len(gk), "gk_err_median": gk.gk_err.median()})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ choice and gate
def choose(candidates: pd.DataFrame) -> dict:
    """The frozen config choice from `candidates` (columns config, coverage, median: shot location, point estimates).

    Among coverage >= `COVERAGE_MIN`: the lowest median, ties to the higher coverage, then the name. Without an
    eligible configuration, the same order over all of them (the approach-1 choice; the gate is then STOP).
    """
    table = candidates[["config", "coverage", "median"]].sort_values("config")
    eligible = table[table.coverage >= COVERAGE_MIN]
    pool = eligible if len(eligible) else table
    ranked = pool[pool["median"].notna()].sort_values(["median", "coverage", "config"], ascending=[True, False, True])
    if not len(ranked):
        raise ValueError("no configuration has a shot-location error")
    best = ranked.iloc[0]
    return {"rule": f"lowest median shot-location error among coverage >= {COVERAGE_MIN:g}; ties to higher coverage",
            "candidates": table.to_dict("records"), "eligible": eligible.config.tolist(), "chosen": str(best.config),
            "chosen_from": "eligible" if len(eligible) else "all configurations (none reaches the coverage)"}


def gate(coverage: float, median: float, gk_coverage: float) -> dict:
    """The frozen gate for approach 2 on the chosen configuration's point estimates: PASS or STOP with the checks."""
    checks = {"shot_location_coverage": {"value": coverage, "min": COVERAGE_MIN, "ok": bool(coverage >= COVERAGE_MIN)},
              "shot_location_median_m": {"value": median, "max": MEDIAN_MAX_M, "ok": bool(median <= MEDIAN_MAX_M)},
              "gk_information_coverage": {"value": gk_coverage, "min": GK_COVERAGE_MIN, "ok": bool(gk_coverage >= GK_COVERAGE_MIN)}}
    return {"result": "PASS" if all(c["ok"] for c in checks.values()) else "STOP", "checks": checks}


def stat(table: pd.DataFrame, quantity: str, name: str, subset: str = "all") -> pd.Series:
    """One statistic of a metrics table (`metrics`) per configuration."""
    rows = table[(table.quantity == quantity) & (table.subset == subset) & (table.stat == name)]
    return rows.set_index("config").value


def decide(table: pd.DataFrame, candidates: list[str] | None = None) -> dict:
    """Config choice and gate from a metrics table (`metrics`): reads only coverage and median rows of subset all,
    of the `candidates` (default: every configuration in the table; a reference configuration is never one).

    Eligibility and the gate use the point estimates; the intervals are reported, not used."""
    names = sorted(set(table.config)) if candidates is None else list(candidates)

    def stat_all(quantity, name):
        values = stat(table, quantity, name)
        missing = sorted(set(names) - set(values.index))
        if missing:
            raise ValueError(f"the metrics have no {quantity} {name} for {missing}")
        return values.loc[names]
    candidates = pd.DataFrame({"coverage": stat_all("shot_location", "coverage"), "median": stat_all("shot_location", "median")})
    choice = choose(candidates.reset_index())
    c = choice["chosen"]
    result = gate(float(candidates.coverage[c]), float(candidates["median"][c]), float(stat_all("gk_any", "coverage")[c]))
    return choice | {"gate": {"config": c} | result,
                     "approach2": "continues" if result["result"] == "PASS" else "closes with the audit report (ticket 05)"}


def metric_differences(new: pd.DataFrame, old: pd.DataFrame, configs) -> dict:
    """How the rows of `configs` in a metrics table `new` compare with the same rows of an earlier run `old` (both as
    read back from metrics.csv): rows of `old` compared, missing from `new`, and differing in value or interval
    (NaN equals NaN). The audit-1 reference C4 must reproduce audit 1's rows exactly."""
    keys = ["config", "quantity", "subset", "stat"]
    before, after = old[old.config.isin(list(configs))].set_index(keys), new.set_index(keys)
    present = before.index.isin(after.index)
    differs = feature_differences(before[present], after.loc[before.index[present]], ["value", "lo", "hi"], 0.0).any(axis=1)
    return {"rows": len(before), "missing": int((~present).sum()), "differing": int(differs.sum())}


def check_replay(cached: pd.DataFrame, replayed: pd.DataFrame, tolerance: float = 1e-9) -> dict[str, int]:
    """Columns where the replayed t0 features differ from the cached feature table, with the number of shots (only
    columns that differ; empty when identical)."""
    a = cached.set_index("shot_id").sort_index()
    b = replayed[replayed.offset_ms == 0].set_index("shot_id").sort_index()
    if not a.index.equals(b.index):
        return {"shot_id": int(len(a.index.symmetric_difference(b.index)))}
    counts = feature_differences(a, b, a.columns, tolerance).sum()
    return {str(c): int(n) for c, n in counts.items() if n}
