"""ball-detection 18: how many bad positions the ball tracker passes on, shorter gaps and rejection by an
uncertainty score (risk-coverage), CPU only.

    uv run python scripts/ball_reliability.py read      # games 2+3 of GSR valid: variants, scores, curves, D2 gate
    uv run python scripts/ball_reliability.py holdout   # D1: the one read of game 4, only with a frozen chosen.json

Tracks: ticket 04's dump (``ball/04/dump25``) with the 45 px post-filter, ticket 04's camera motion and the
package's ``track_ball`` with ``ball/04/tune/chosen.json``; only ``max_gap`` changes (``MAX_GAPS``). ``max_gap`` 2 s
is phase 1's candidate (ticket 09's tracks). ``max_gap`` also caps how far the gate widens, so every variant is its
own tracker run (``ball_temporal.run_tracks``).

Frames: rule 02's denominator (``detector_bench.placement_rows``, frozen homographies, box centre), as in ticket 17.
A position is a frame with a ball and a homography; bad = error > 5 m (and > 2 m); well placed = <= 5 m of all
denominator frames.

Uncertainty score: a logistic regression of P(error > 5 m) on features known at inference (``FEATURES``, from the
tracks, the camera motion and ticket 04's carried homographies; ``position_features``); baseline: the same on the gap
length alone. Alternating: fitted on game A's positions, scored on game B, (A, B) = (2, 3) and (3, 2); the
out-of-game scores of both games read together. A position is rejected when its score is above a threshold chosen on
the fitting game only (``choose``: rejected positions >= 50% bad, well placed down <= 2 pp vs phase 1, then the
lowest > 5 m share among passed; ties: more coverage). D2's gate (owner, 2026-10-27) on the pooled out-of-game read:
(a) below the curve of ``max_gap`` alone at the point's coverage, (b) >= 50% of rejected positions bad, (c) well
placed down <= 2 pp vs phase 1. Met: the rule on games 2+3 together -> ``ball/18/chosen.json`` and D1's one read of
game 4 (``holdout``). Not met: curves only, nothing changes, game 4 stays unread. Game 5 is never tracked here.
"""

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ball_errors as be  # noqa: E402
import ball_scorecard as bs  # noqa: E402
import ball_temporal as bt  # noqa: E402
import detector_bench as bench  # noqa: E402

import tactifoot_vision as tv  # noqa: E402

logger = logging.getLogger("ball_reliability")

OUT = be.ROOT / "ball/18"
SETTING = bt.OUT / "tune/chosen.json"  # ticket 04's tracker setting (max_gap 2 s)
REFERENCE = (
    be.ROOT / "ball/09/track_valid/tracks.parquet"
)  # phase 1's candidate (with the filter)
REFERENCE_NO_FILTER = (
    be.ROOT / "ball/04/track/tracks.parquet"
)  # 04's own tracks of games 2+3
HOLDOUT_DUMP = be.ROOT / "ball/10/dump25_holdout"
HOLDOUT_MOTION = be.ROOT / "ball/10/motion_holdout"
GAMES = ("2", "3")
MAX_GAPS = (0.25, 0.5, 1.0, 2.0)
PHASE1_GAP = 2.0
FAR = bench.PLACEMENT["far_m"]  # 5 m
NEAR = 2.0
FPS = bt.FPS
FEATURES = (
    "interpolated",
    "gap_s",
    "to_measure_s",
    "log_speed",
    "speed_missing",
    "conf",
    "log_motion",
    "motion_break",
    "log_size",
)
BASELINE = ("gap_s",)
TAUS = tuple(np.round(np.arange(0.20, 0.951, 0.01), 2))
MIN_BAD_SHARE = 0.5  # D2 (b): rejected positions with error > 5 m
MAX_LOSS_PP = 2.0  # D2 (c): well placed lost vs phase 1
LEVELS = (
    0.0,
    0.025,
    0.05,
    0.10,
    0.20,
    0.30,
)  # descriptive shares of rejected positions
REJECT_SHARE = (
    0.10  # the registered model-vs-baseline comparison: 10% of positions rejected
)
CENTRE = np.array([960.0, 540.0])


# ------------------------------------------------------------------ features
def gap_structure(status: np.ndarray) -> pd.DataFrame:
    """Per frame of one clip's track (``status``: measured / interpolated / none, frame order) the measured frames
    before and after an interpolated run (``before`` / ``after`` indices, -1 if none), the gap length in s (end to
    end; 0 for a measured ball, NaN without a ball) and the time to the nearest measurement in s."""
    status = np.asarray(status)
    n = len(status)
    before = np.full(n, -1)
    after = np.full(n, -1)
    last = -1
    for i in range(n):
        if status[i] == "measured":
            last = i
        before[i] = last
    last = -1
    for i in range(n - 1, -1, -1):
        if status[i] == "measured":
            last = i
        after[i] = last
    gap = np.full(n, np.nan)
    to_measure = np.full(n, np.nan)
    measured = status == "measured"
    gap[measured] = 0.0
    to_measure[measured] = 0.0
    idx = np.arange(n)
    interp = status == "interpolated"
    both = interp & (before >= 0) & (after >= 0)
    gap[both] = (after[both] - before[both]) / FPS
    near = np.where(
        before >= 0, idx - before, np.inf
    )  # one side may be missing at a clip end
    near = np.minimum(near, np.where(after >= 0, after - idx, np.inf))
    to_measure[interp] = near[interp] / FPS
    one = interp & ~both
    gap[one] = to_measure[one]
    return pd.DataFrame(
        {"before": before, "after": after, "gap_s": gap, "to_measure_s": to_measure}
    )


def project(centres: np.ndarray, homographies: list) -> np.ndarray:
    """Each frame's point through its own homography (NaN without one or without a point)."""
    out = np.full((len(centres), 2), np.nan)
    for i, (point, matrix) in enumerate(zip(centres, homographies, strict=True)):
        if matrix is not None and np.isfinite(point).all():
            out[i] = tv.pitch.frame_to_pitch(point[None], matrix)[0]
    return out


def camera_shift(motion: np.ndarray, frames: int) -> np.ndarray:
    """Per frame t (0-based) the image centre's shift in px under the motion t -> t + 1 (the last frame: the motion
    into it); NaN at a motion break."""
    shift = np.full(frames, np.nan)
    for t in range(frames):
        m = motion[min(t, len(motion) - 1)] if len(motion) else np.full((3, 3), np.nan)
        if np.isnan(m).any():
            continue
        p = m @ np.array([*CENTRE, 1.0])
        shift[t] = float(np.hypot(*(p[:2] / p[2] - CENTRE)))
    return shift


def position_features(
    track: pd.DataFrame, homographies: list, motion: np.ndarray
) -> pd.DataFrame:
    """``FEATURES`` for every frame of one clip's track (``ball_temporal.chosen_balls`` columns, frames 1..N in
    order): status, gap, time to a measurement, implied speed, confidence, camera motion and size."""
    track = track.sort_values("frame").reset_index(drop=True)
    n = len(track)
    if not np.array_equal(track.frame.to_numpy(), np.arange(1, n + 1)):
        raise ValueError("a track must hold every frame 1..N")
    status = track.status.to_numpy()
    boxes = track[bt.COLUMNS].to_numpy(float)
    gaps = gap_structure(status)
    pitch = project(be.centres(boxes), homographies)
    conf = track.conf.to_numpy(float)
    has = status != "none"
    speed = np.full(n, np.nan)
    end_conf = conf.copy()
    for i in np.flatnonzero(status == "interpolated"):
        a, b = gaps.before[i], gaps.after[i]
        ends = [e for e in (a, b) if e >= 0]
        end_conf[i] = min(conf[e] for e in ends) if ends else np.nan
        if a >= 0 and b >= 0 and np.isfinite(pitch[[a, b]]).all():
            speed[i] = np.hypot(*(pitch[b] - pitch[a])) / ((b - a) / FPS)
    for i in np.flatnonzero(status == "measured"):
        jumps = [
            np.hypot(*(pitch[j] - pitch[i])) * FPS
            for j in (i - 1, i + 1)
            if 0 <= j < n and has[j] and np.isfinite(pitch[[i, j]]).all()
        ]
        if jumps:
            speed[i] = max(jumps)
    shift = camera_shift(motion, n)
    out = pd.DataFrame(
        {
            "frame": track.frame.to_numpy(),
            "status": status,
            "interpolated": (status == "interpolated").astype(float),
            "gap_s": gaps.gap_s,
            "to_measure_s": gaps.to_measure_s,
            "log_speed": np.log1p(np.nan_to_num(speed, nan=0.0)),
            "speed_missing": (~np.isfinite(speed)).astype(float),
            "conf": np.nan_to_num(end_conf, nan=0.0),
            "log_motion": np.log1p(np.nan_to_num(shift, nan=0.0)),
            "motion_break": (~np.isfinite(shift)).astype(float),
            "log_size": np.log(np.maximum(be.diameter(boxes), 1.0)),
        }
    )
    out.loc[~has, list(FEATURES)] = np.nan
    return out


# ------------------------------------------------------------------ the score
class Score:
    """Standardised logistic regression of P(error > 5 m) on ``columns``."""

    def __init__(self, columns=FEATURES) -> None:
        self.columns = list(columns)

    def fit(self, table: pd.DataFrame, bad: np.ndarray) -> "Score":
        x = table[self.columns].to_numpy(float)
        self.mean = x.mean(axis=0)
        self.scale = np.where(x.std(axis=0) > 0, x.std(axis=0), 1.0)
        self.model = LogisticRegression(C=1.0, max_iter=1000).fit(
            (x - self.mean) / self.scale, np.asarray(bad, int)
        )
        return self

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        x = table[self.columns].to_numpy(float)
        return self.model.predict_proba((x - self.mean) / self.scale)[:, 1]

    def frozen(self) -> dict:
        return {
            "columns": self.columns,
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "coef": self.model.coef_[0].tolist(),
            "intercept": float(self.model.intercept_[0]),
        }


def frozen_probability(frozen: dict, table: pd.DataFrame) -> np.ndarray:
    """``Score.predict`` from ``Score.frozen()``'s record."""
    x = (table[frozen["columns"]].to_numpy(float) - frozen["mean"]) / np.asarray(
        frozen["scale"]
    )
    return 1 / (1 + np.exp(-(x @ np.asarray(frozen["coef"]) + frozen["intercept"])))


# ------------------------------------------------------------------ rejection and the curve
def apply_rejection(
    error: np.ndarray, score: np.ndarray, tau: float | None
) -> np.ndarray:
    """``error`` with the positions whose ``score`` > ``tau`` removed (NaN); ``tau`` None: nothing removed."""
    error = np.asarray(error, float).copy()
    if tau is not None:
        error[np.isfinite(error) & (np.asarray(score, float) > tau)] = np.nan
    return error


def operating(error: np.ndarray, kept: np.ndarray, reference: np.ndarray) -> dict:
    """Numbers of one operating point on denominator frames: ``error`` before rejection (NaN: no position),
    ``kept`` after it, ``reference`` = phase 1's errors on the same frames."""
    error, kept, reference = (np.asarray(a, float) for a in (error, kept, reference))
    n = len(error)
    placed = np.isfinite(kept)
    rejected = np.isfinite(error) & ~placed
    k = int(placed.sum())
    return {
        "frames": n,
        "coverage": float(placed.mean()),
        "far_of_passed": float((kept[placed] > FAR).mean()) if k else np.nan,
        "near_of_passed": float((kept[placed] > NEAR).mean()) if k else np.nan,
        "far_of_all": float((kept > FAR).mean()),
        "well_placed": float((kept <= FAR).mean()),
        "loss_pp": 100 * float((reference <= FAR).mean() - (kept <= FAR).mean()),
        "rejected": int(rejected.sum()),
        "rejected_bad_share": float((error[rejected] > FAR).mean())
        if rejected.any()
        else np.nan,
    }


def curve(
    error: np.ndarray,
    score: np.ndarray,
    weights: np.ndarray | None = None,
    limit: float = FAR,
) -> pd.DataFrame:
    """Risk-coverage curve: rejecting positions from the highest score down, one point per distinct score (ties
    leave together) plus the no-rejection point. ``coverage`` = passed positions / all frames, ``risk`` = share of
    passed with error > ``limit``, ``rejected_share`` = rejected / positions, ``rejected_bad_share`` = rejected with
    error > ``limit`` / rejected, ``good_lost_of_all`` = rejected within ``limit`` / all frames (``weights``:
    per-frame weights)."""
    error = np.asarray(error, float)
    w = np.ones(len(error)) if weights is None else np.asarray(weights, float)
    placed = np.isfinite(error)
    s, e, wp = np.asarray(score, float)[placed], error[placed], w[placed]
    order = np.argsort(-s, kind="stable")
    s, bad, wp = s[order], (e[order] > limit) * wp[order], wp[order]
    total, positions, bads = w.sum(), wp.sum(), bad.sum()
    ends = (
        np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
        if len(s)
        else np.array([], int)
    )
    cw, cb = np.cumsum(wp), np.cumsum(bad)
    removed = np.r_[0.0, cw[ends]]
    removed_bad = np.r_[0.0, cb[ends]]
    kept = positions - removed
    with np.errstate(invalid="ignore", divide="ignore"):
        return pd.DataFrame(
            {
                "threshold": np.r_[np.inf, s[ends]],
                "coverage": kept / total,
                "risk": (bads - removed_bad) / kept,
                "rejected_share": removed / positions,
                "bad_of_all": (bads - removed_bad) / total,
                "rejected_bad_share": removed_bad / removed,
                "good_lost_of_all": (removed - removed_bad) / total,
            }
        )


def at(table: pd.DataFrame, x: str, value: float, y: str = "risk") -> float:
    """Linear interpolation of ``y`` at ``x`` = ``value`` along a curve (NaN outside its range)."""
    t = table[[x, y]].dropna().sort_values(x)
    if t.empty or value < t[x].iloc[0] - 1e-12 or value > t[x].iloc[-1] + 1e-12:
        return np.nan
    return float(np.interp(value, t[x].to_numpy(), t[y].to_numpy()))


def aurc(table: pd.DataFrame) -> float:
    """Mean risk over the share of positions kept (0..1), trapezoid on the curve's points."""
    t = table.dropna(subset=["risk"]).sort_values("rejected_share")
    kept = 1 - t.rejected_share.to_numpy()
    return float(np.trapezoid(t.risk.to_numpy()[::-1], kept[::-1])) / max(
        kept.max() - kept.min(), 1e-12
    )


def gap_curve(points: pd.DataFrame, coverage: float) -> float:
    """The ``max_gap``-alone curve (``points``: coverage, far_of_passed per variant) at ``coverage``; NaN outside."""
    return at(points.rename(columns={"far_of_passed": "risk"}), "coverage", coverage)


# ------------------------------------------------------------------ the rule
def choose(candidates: list[dict]) -> dict | None:
    """Ticket 18's rule on the fitting game: admissible = (no rejection or >= 50% of rejected positions bad) and
    well placed down <= 2 pp vs phase 1; the lowest > 5 m share among passed, ties: more coverage, then order."""
    ok = [
        c
        for c in candidates
        if c["loss_pp"] <= MAX_LOSS_PP + 1e-9
        and (c["rejected"] == 0 or c["rejected_bad_share"] >= MIN_BAD_SHARE)
    ]
    if not ok:
        return None
    return min(ok, key=lambda c: (c["far_of_passed"], -c["coverage"]))


def gate(point: dict, gap_points: pd.DataFrame) -> dict:
    """D2 on the pooled out-of-game point: (a) below the ``max_gap`` curve at its coverage, (b) >= 50% of rejected
    positions bad, (c) well placed down <= 2 pp vs phase 1."""
    line = gap_curve(gap_points, point["coverage"])
    a = bool(np.isfinite(line) and point["far_of_passed"] < line)
    b = bool(point["rejected"] > 0 and point["rejected_bad_share"] >= MIN_BAD_SHARE)
    c = bool(point["loss_pp"] <= MAX_LOSS_PP + 1e-9)
    return {
        "gap_curve_at_coverage": line,
        "a_below_gap_curve": a,
        "b_rejected_bad": b,
        "c_loss": c,
        "met": a and b and c,
    }


def rule_candidates(table: pd.DataFrame, score_column: str, taus=TAUS) -> list[dict]:
    """Every (variant, tau) of ``table`` (one row per variant x denominator frame: ``error_m``, phase 1's
    ``reference_m`` and the score) and every variant without rejection, as ``operating`` numbers."""
    out = []
    for gap, part in table.groupby("max_gap", sort=True):
        error = part.error_m.to_numpy(float)
        reference = part.reference_m.to_numpy(float)
        score = part[score_column].to_numpy(float)
        for tau in (None, *taus):
            kept = apply_rejection(error, score, tau)
            out.append(
                {"max_gap": float(gap), "tau": tau} | operating(error, kept, reference)
            )
    return out


# ------------------------------------------------------------------ data
def variant_tracks(
    clips: list[str],
    split: str,
    dump: Path,
    motion_dir: Path,
    gaps=MAX_GAPS,
    limit: float | None = bt.POST_FILTER,
    workers: int = 12,
) -> dict[float, pd.DataFrame]:
    setting = json.loads(SETTING.read_text())
    settings = [setting | {"max_gap": g} for g in gaps]
    _, tables = bt.run_tracks(clips, settings, workers, dump, split, motion_dir, limit)
    return dict(zip(gaps, tables, strict=True))


def clip_features(
    tracks: pd.DataFrame, split: str, motion_dir: Path, frozen: dict
) -> pd.DataFrame:
    parts = []
    for clip, track in tracks.groupby("clip", sort=True):
        frames = int(track.frame.max())
        homographies = bt.clip_homographies(clip, frames, frozen, motion_dir)
        motion = bt.load_motion(clip, motion_dir)
        found = position_features(track.drop(columns=["clip"]), homographies, motion)
        found.insert(0, "clip", clip)
        parts.append(found)
    return pd.concat(parts, ignore_index=True)


def placed_rows(
    tracks: pd.DataFrame, split: str, frozen: dict, games=None
) -> pd.DataFrame:
    rows = bench.placement_rows(bt.table_top_ball(tracks), frozen, split)
    if games is not None:
        rows = rows[rows.game.isin(games)].reset_index(drop=True)
    return rows


def variant_table(
    tracks: dict[float, pd.DataFrame], split: str, motion_dir: Path, games=None
) -> pd.DataFrame:
    """One row per variant x denominator frame: the placement error and the position's features."""
    frozen = bench.frozen_homographies(split)
    parts = []
    for gap, table in tracks.items():
        rows = placed_rows(table, split, frozen, games)
        features = clip_features(table, split, motion_dir, frozen)
        rows = rows.merge(
            features, on=["clip", "frame"], how="left", validate="one_to_one"
        )
        rows.insert(0, "max_gap", gap)
        parts.append(rows)
    return pd.concat(parts, ignore_index=True)


def strip(table: pd.DataFrame) -> pd.DataFrame:
    return table.drop(columns=["seconds"]).reset_index(drop=True)


def controls(clips: list[str], tracks: pd.DataFrame, workers: int) -> dict:
    """max_gap 2 s without rejection = ticket 09's tracks (filter) and ticket 04's (no filter), whole tables."""
    ref09 = pd.read_parquet(REFERENCE)
    ref09 = ref09[ref09["clip"].isin(clips)]
    ref04 = pd.read_parquet(REFERENCE_NO_FILTER)
    nofilter = variant_tracks(
        clips, "valid", bt.DUMP, bt.MOTION_DIR, (PHASE1_GAP,), None, workers
    )
    key = ["clip", "frame"]

    def same(a: pd.DataFrame, b: pd.DataFrame) -> bool:
        a = strip(a).sort_values(key).reset_index(drop=True)
        b = strip(b).sort_values(key).reset_index(drop=True)
        return bool(a.equals(b))

    return {
        "tracks_09_identical": same(tracks, ref09),
        "tracks_04_identical": same(
            nofilter[PHASE1_GAP], ref04[ref04["clip"].isin(clips)]
        ),
        "rows": int(len(tracks)),
    }


# ------------------------------------------------------------------ bootstrap
def clip_weights(draws: bs.Draws, clips: pd.Series) -> np.ndarray:
    """``weights[d, i]`` = how often draw ``d`` takes row ``i``'s clip."""
    return draws.weights[:, draws.codes(clips)]


def point_draws(table: pd.DataFrame, kept: np.ndarray, draws: bs.Draws) -> dict:
    """Clip-bootstrap draws of the operating point's shares (``table``: error_m, reference_m, clip)."""
    error = table.error_m.to_numpy(float)
    reference = table.reference_m.to_numpy(float)
    placed = np.isfinite(kept)
    rejected = np.isfinite(error) & ~placed
    ones = np.ones(len(table))
    clip = table["clip"]
    flags = {
        "coverage": (placed, ones),
        "far_of_passed": (kept > FAR, placed),
        "near_of_passed": (kept > NEAR, placed),
        "far_of_all": (kept > FAR, ones),
        "well_placed": (kept <= FAR, ones),
        "rejected_bad_share": ((error > FAR) & rejected, rejected),
    }
    found = {
        name: bs.ratio_draws(
            draws, draws.per_clip(clip, num), draws.per_clip(clip, den)
        )
        for name, (num, den) in flags.items()
    }
    found["reference_well_placed"] = bs.ratio_draws(
        draws, draws.per_clip(clip, reference <= FAR), draws.per_clip(clip, ones)
    )
    found["reference_far_of_passed"] = bs.ratio_draws(
        draws,
        draws.per_clip(clip, reference > FAR),
        draws.per_clip(clip, np.isfinite(reference)),
    )
    found["reference_far_of_all"] = bs.ratio_draws(
        draws, draws.per_clip(clip, reference > FAR), draws.per_clip(clip, ones)
    )
    return found


def paired_point(found: dict) -> dict:
    out = bs.summary(found)
    for name in ("well_placed", "far_of_passed", "far_of_all"):
        v, d = found[name]
        r, rd = found[f"reference_{name}"]
        out[f"{name}_minus_phase1"] = {"value": v - r, "ci": bs.interval(d - rd)}
    return out


def risk_at_reject(error, score, weights, share=REJECT_SHARE) -> float:
    c = curve(error, score, weights)
    return at(c, "rejected_share", share)


def compare_scores(table: pd.DataFrame, draws: bs.Draws) -> dict:
    """Model vs baseline on max_gap 2 s, out of game: > 5 m among passed at 10% of positions rejected, paired."""
    error = table.error_m.to_numpy(float)
    model, base = table.p_model.to_numpy(float), table.p_baseline.to_numpy(float)
    weights = clip_weights(draws, table["clip"])
    m = np.array([risk_at_reject(error, model, w) for w in weights])
    b = np.array([risk_at_reject(error, base, w) for w in weights])
    pm, pb = risk_at_reject(error, model, None), risk_at_reject(error, base, None)
    return {
        "reject_share": REJECT_SHARE,
        "model": {"value": pm, "ci": bs.interval(m)},
        "baseline": {"value": pb, "ci": bs.interval(b)},
        "difference": {"value": pm - pb, "ci": bs.interval(m - b)},
        "model_beats_baseline": bool(bs.interval(m - b)[1] < 0),
    }


def calibration(p: np.ndarray, bad: np.ndarray) -> dict:
    p, bad = np.asarray(p, float), np.asarray(bad, float)
    eps = 1e-12
    bins = np.minimum((p * 10).astype(int), 9)
    table = [
        {
            "bin": f"{b / 10:.1f}-{(b + 1) / 10:.1f}",
            "n": int((bins == b).sum()),
            "mean_p": float(p[bins == b].mean()),
            "observed": float(bad[bins == b].mean()),
        }
        for b in range(10)
        if (bins == b).any()
    ]
    return {
        "log_loss": float(
            -np.mean(bad * np.log(p + eps) + (1 - bad) * np.log(1 - p + eps))
        ),
        "brier": float(np.mean((p - bad) ** 2)),
        "base_rate": float(bad.mean()),
        "bins": table,
    }


# ------------------------------------------------------------------ read
def fit_scores(table: pd.DataFrame, fit_games, score_games) -> pd.DataFrame:
    """Per variant: model and baseline fitted on ``fit_games``' positions, probabilities on ``score_games``' rows."""
    parts = []
    for _, part in table.groupby("max_gap", sort=True):
        fit = part[part.game.isin(fit_games) & np.isfinite(part.error_m)]
        bad = (fit.error_m > FAR).to_numpy()
        target = part[part.game.isin(score_games)].copy()
        model, base = Score(FEATURES).fit(fit, bad), Score(BASELINE).fit(fit, bad)
        has = np.isfinite(target.error_m)
        target["p_model"] = np.nan
        target["p_baseline"] = np.nan
        target.loc[has, "p_model"] = model.predict(target[has])
        target.loc[has, "p_baseline"] = base.predict(target[has])
        parts.append(target)
    return pd.concat(parts, ignore_index=True)


def with_reference(table: pd.DataFrame) -> pd.DataFrame:
    ref = table[table.max_gap == PHASE1_GAP][["clip", "frame", "error_m"]].rename(
        columns={"error_m": "reference_m"}
    )
    return table.merge(ref, on=["clip", "frame"], how="left", validate="many_to_one")


def apply_choice(scored: pd.DataFrame, choice: dict, column: str) -> pd.DataFrame:
    part = scored[scored.max_gap == choice["max_gap"]].copy()
    part["kept_m"] = apply_rejection(
        part.error_m.to_numpy(float), part[column].to_numpy(float), choice["tau"]
    )
    return part


def read(args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    clips = bt.games_clips(GAMES)
    tracks = variant_tracks(
        clips, "valid", bt.DUMP, bt.MOTION_DIR, workers=args.workers
    )
    for gap, t in tracks.items():
        t.to_parquet(OUT / f"tracks_gap{gap:g}.parquet", index=False)
    checks = controls(clips, tracks[PHASE1_GAP], args.workers)
    table = with_reference(variant_table(tracks, "valid", bt.MOTION_DIR, GAMES))
    phase1 = table[table.max_gap == PHASE1_GAP]
    checks["phase1_well_placed"] = float((phase1.error_m <= FAR).mean())
    checks["phase1_far_of_placed"] = float(
        (phase1.error_m[np.isfinite(phase1.error_m)] > FAR).mean()
    )
    print(json.dumps(checks, indent=2))

    directions, out_of_game, chosen = {}, [], {}
    for fit_game, score_game in (("2", "3"), ("3", "2")):
        insample = fit_scores(table, (fit_game,), (fit_game,))
        found = {}
        for name, column in (("model", "p_model"), ("baseline", "p_baseline")):
            found[name] = choose(rule_candidates(insample, column))
        gaps_only = choose(rule_candidates(insample, "p_model", taus=()))
        found["max_gap_only"] = gaps_only
        scored = fit_scores(table, (fit_game,), (score_game,))
        scored["direction"] = f"{fit_game}->{score_game}"
        out_of_game.append(scored)
        chosen[score_game] = found
        directions[f"{fit_game}->{score_game}"] = {"chosen_on_fit_game": found}
    pooled = pd.concat(out_of_game, ignore_index=True)
    pooled.to_parquet(OUT / "scored_out_of_game.parquet", index=False)
    draws = bs.Draws(clips)

    # max_gap alone (pooled out-of-game frames = all frames of games 2+3)
    gap_points = pd.DataFrame(
        [
            {"max_gap": g}
            | operating(
                p.error_m.to_numpy(float),
                p.error_m.to_numpy(float),
                p.reference_m.to_numpy(float),
            )
            for g, p in pooled.groupby("max_gap", sort=True)
        ]
    )
    game_points = {
        game: pd.DataFrame(
            [
                {"max_gap": g}
                | operating(
                    p.error_m.to_numpy(float),
                    p.error_m.to_numpy(float),
                    p.reference_m.to_numpy(float),
                )
                for g, p in q.groupby("max_gap", sort=True)
            ]
        )
        for game, q in pooled.groupby("game")
    }
    # curves per variant and score, > 5 m and > 2 m
    curves = []
    for g, p in pooled.groupby("max_gap", sort=True):
        for name, column in (("model", "p_model"), ("baseline", "p_baseline")):
            for limit in (FAR, NEAR):
                c = curve(
                    p.error_m.to_numpy(float), p[column].to_numpy(float), None, limit
                )
                c.insert(0, "limit_m", limit)
                c.insert(0, "score", name)
                c.insert(0, "max_gap", g)
                curves.append(c)
    curves = pd.concat(curves, ignore_index=True)
    curves.to_csv(OUT / "curves.csv", index=False)
    levels = []  # descriptive: the out-of-game curves at fixed shares of rejected positions (nothing chosen)
    for (g, name), c in curves[curves.limit_m == FAR].groupby(["max_gap", "score"]):
        near = curves[
            (curves.max_gap == g) & (curves.score == name) & (curves.limit_m == NEAR)
        ]
        for share in LEVELS:
            levels.append(
                {"max_gap": g, "score": name, "rejected_share": share}
                | {
                    y: at(c, "rejected_share", share, y)
                    for y in (
                        "coverage",
                        "risk",
                        "bad_of_all",
                        "rejected_bad_share",
                        "good_lost_of_all",
                    )
                }
                | {"risk_2m": at(near, "rejected_share", share)}
            )
    levels = pd.DataFrame(levels)
    levels.to_csv(OUT / "levels.csv", index=False)
    equal = []
    for g in MAX_GAPS:
        for name in ("model", "baseline"):
            for limit in (FAR, NEAR):
                c = curves[
                    (curves.max_gap == g)
                    & (curves.score == name)
                    & (curves.limit_m == limit)
                ]
                for _, point in gap_points.iterrows():
                    equal.append(
                        {
                            "max_gap": g,
                            "score": name,
                            "limit_m": limit,
                            "at_coverage_of_gap": point.max_gap,
                            "coverage": point.coverage,
                            "risk": at(c, "coverage", point.coverage),
                        }
                    )
    equal = pd.DataFrame(equal)
    equal.to_csv(OUT / "equal_coverage.csv", index=False)
    summary = []
    for g, p in pooled.groupby("max_gap", sort=True):
        has = np.isfinite(p.error_m)
        for name, column in (("model", "p_model"), ("baseline", "p_baseline")):
            summary.append(
                {"max_gap": g, "score": name}
                | {
                    "aurc": aurc(
                        curve(p.error_m.to_numpy(float), p[column].to_numpy(float))
                    )
                }
                | {
                    k: v
                    for k, v in calibration(
                        p[column][has], p.error_m[has] > FAR
                    ).items()
                    if k != "bins"
                }
            )
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "score_summary.csv", index=False)
    p2 = pooled[pooled.max_gap == PHASE1_GAP]
    has = np.isfinite(p2.error_m)
    calib = {
        name: calibration(p2[column][has], p2.error_m[has] > FAR)
        for name, column in (("model", "p_model"), ("baseline", "p_baseline"))
    }
    comparison = compare_scores(p2, draws)

    # the pooled out-of-game operating points of the rule (each game with the other game's choice)
    points = {}
    for name, column in (
        ("model", "p_model"),
        ("baseline", "p_baseline"),
        ("max_gap_only", "p_model"),
    ):
        parts = []
        for score_game, found in chosen.items():
            choice = found[name]
            if choice is None:
                break
            part = apply_choice(pooled[pooled.game == score_game], choice, column)
            parts.append(part)
        if len(parts) != len(chosen):
            points[name] = {"choice": None}
            continue
        part = pd.concat(parts, ignore_index=True)
        numbers = operating(
            part.error_m.to_numpy(float),
            part.kept_m.to_numpy(float),
            part.reference_m.to_numpy(float),
        )
        per_direction = {
            g: operating(
                q.error_m.to_numpy(float),
                q.kept_m.to_numpy(float),
                q.reference_m.to_numpy(float),
            )
            for g, q in part.groupby("game")
        }
        points[name] = {
            "choices": {g: chosen[g][name] for g in chosen},
            "pooled": numbers,
            "gate": gate(numbers, gap_points),
            "per_scored_game": per_direction,
            "gate_per_scored_game": {
                g: gate(q, game_points[g]) for g, q in per_direction.items()
            },
            "draws": paired_point(
                point_draws(part, part.kept_m.to_numpy(float), draws)
            ),
        }
    met = points["model"].get("gate", {}).get("met", False)
    coefficients = {}  # descriptive: the model of each fitting game on phase 1's tracks (standardised features)
    for game in GAMES:
        fit = phase1[(phase1.game == game) & np.isfinite(phase1.error_m)]
        coefficients[game] = (
            Score(FEATURES).fit(fit, (fit.error_m > FAR).to_numpy()).frozen()
        )

    frozen = None
    if met:
        insample = fit_scores(table, GAMES, GAMES)
        choice = choose(rule_candidates(insample, "p_model"))
        if choice is not None and choice["tau"] is not None:
            part = table[table.max_gap == choice["max_gap"]]
            fit = part[np.isfinite(part.error_m)]
            score = Score(FEATURES).fit(fit, (fit.error_m > FAR).to_numpy())
            setting = json.loads(SETTING.read_text()) | {"max_gap": choice["max_gap"]}
            frozen = {
                "created": datetime.now(UTC).isoformat(timespec="seconds"),
                "ticket": "ball-detection 18",
                "tracker": "tactifoot_vision.pipeline.track_ball",
                "setting": setting,
                "post_filter_long_side_px": bt.POST_FILTER,
                "reject_when_probability_above": choice["tau"],
                "score": score.frozen(),
                "fitted_on": "games 2+3 of GSR valid, every position of the variant",
                "rule_on_games_2_3": choice,
            }
            (OUT / "chosen.json").write_text(
                json.dumps(frozen, indent=2, default=float) + "\n"
            )

    record = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "games": list(GAMES),
        "clips": len(clips),
        "max_gaps": list(MAX_GAPS),
        "features": list(FEATURES),
        "baseline": list(BASELINE),
        "taus": [float(t) for t in TAUS],
        "bootstrap": bs.BOOTSTRAP,
        "controls": checks,
        "max_gap_points": gap_points.to_dict("records"),
        "max_gap_points_per_game": {
            g: t.to_dict("records") for g, t in game_points.items()
        },
        "levels": levels.to_dict("records"),
        "directions": directions,
        "points": points,
        "model_vs_baseline": comparison,
        "calibration_max_gap_2": calib,
        "coefficients_max_gap_2": coefficients,
        "score_summary": summary.to_dict("records"),
        "gate_met": met,
        "frozen": frozen,
    }
    (OUT / "read.json").write_text(json.dumps(record, indent=2, default=float) + "\n")
    print(
        json.dumps(
            {
                k: record[k]
                for k in (
                    "controls",
                    "max_gap_points",
                    "points",
                    "model_vs_baseline",
                    "gate_met",
                )
            },
            indent=2,
            default=float,
        )
    )


# ------------------------------------------------------------------ D1: game 4
def holdout(args: argparse.Namespace) -> None:
    """The one read of game 4 (D1): phase 1's candidate vs the frozen operating point, paired."""
    path = OUT / "chosen.json"
    if not path.exists():
        raise SystemExit("no frozen operating point: game 4 stays unread (D1)")
    target = OUT / "holdout"
    if (target / "read.json").exists():
        raise SystemExit("game 4 was read once already (D1)")
    frozen = json.loads(path.read_text())
    gap = float(frozen["setting"]["max_gap"])
    clips = [c.name for c in bt.clip_dirs("holdout")]
    gaps = tuple(sorted({PHASE1_GAP, gap}))
    tracks = variant_tracks(
        clips, "holdout", HOLDOUT_DUMP, HOLDOUT_MOTION, gaps, workers=args.workers
    )
    table = with_reference(variant_table(tracks, "holdout", HOLDOUT_MOTION))
    part = table[table.max_gap == gap].copy()
    has = np.isfinite(part.error_m)
    part["p"] = np.nan
    part.loc[has, "p"] = frozen_probability(frozen["score"], part[has])
    part["kept_m"] = apply_rejection(
        part.error_m.to_numpy(float),
        part.p.to_numpy(float),
        frozen["reject_when_probability_above"],
    )
    target.mkdir(parents=True, exist_ok=True)
    part.to_parquet(target / "rows.parquet", index=False)
    draws = bs.Draws(clips)
    record = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "note": "limited validation: dd2c and the keypoint model saw game 4",
        "clips": len(clips),
        "frozen": str(path),
        "point": operating(
            part.error_m.to_numpy(float),
            part.kept_m.to_numpy(float),
            part.reference_m.to_numpy(float),
        ),
        "draws": paired_point(point_draws(part, part.kept_m.to_numpy(float), draws)),
        "calibration": calibration(part.p[has], part.error_m[has] > FAR),
    }
    (target / "read.json").write_text(
        json.dumps(record, indent=2, default=float) + "\n"
    )
    print(json.dumps(record, indent=2, default=float))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name, func in (("read", read), ("holdout", holdout)):
        p = commands.add_parser(name)
        p.add_argument("--workers", type=int, default=12)
        p.set_defaults(func=func)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
