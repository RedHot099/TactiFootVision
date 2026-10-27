"""ball-detection 19: does the ball location improve xG over the shooter's feet? Out-of-fold comparison on the 7,062
non-penalty shots of the 328 development matches, the frozen xg-stack procedure (C6), one input changed per arm.

    XG=/home/kuba/projects/tactifoot_vision/.claude/worktrees/research-xg-detector/experiments/claude_fable
    $XG/run.sh --no-sync python $PWD/scripts/ball_xg_compare.py read     # controls, then the arms (CPU)
    uv run python scripts/ball_xg_compare.py clips                        # feasibility of arm T (CPU, no xG read)

``read`` runs in the xG experiment's environment (``run.sh``: xg-detector's ``src`` and ``xgpoc`` on the path), because
it replays the xG caches with the frozen xG code. It first runs the controls and stops unless each reproduces its
record exactly: the frozen E3 read-out (``results/xgs_stack/real``), arm R (``results/xgs_stack_dd2c``) and arm F
(``results/xgs_stack_feet_t0_flags``, whose replay also reproduces ``cache/xgd_full4_feet_t0/E3/features.parquet``).

Location arms (the E3 table the stack reads; everything else is the frozen procedure, every arm refitted in the folds):

    R    E3's rule (ball -400, -800, 0 ms; shot location = the ball's projection), cache xgd_full4/E3; reference row
    F    ``location.FEET_T0`` (frozen), cache xgd_full4_feet_t0/E3
    C    F + the ball-feet relation columns of F's t0 frame (``RELATION``), added to the stack arm ALL only
    S2   F with the shooter picked by ticket 13's second-stage ball (ticket 16's candidates): the FEET_T0 replay on
         frames with the S2 ball, the ball-source columns kept from F (``FLAG_PATTERN``), so only the shooter differs
    S2_rule  descriptive: the FEET_T0 replay on the S2 frames with its own ball-source columns (as ticket 16)

Primary measure: ΔLL of the stack arm ALL against F's, paired match bootstrap (2,000 draws, seed 0). Family {C, S2, T}
(T: ticket 20), Holm; "adds value" = Holm at α 0.025 on the one-sided p (the upper bound of the two-sided 95%),
"equivalent" = TOST within ±0.002 with Holm at α 0.05, else "inconclusive"; without T its p is 1 (conservative).

Registration: ticket 19's comment of 2026-10-27 13:55.
"""

import argparse
import json
import logging
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ball_errors as be  # noqa: E402

logger = logging.getLogger("ball_xg_compare")

XG = Path(
    "/home/kuba/projects/tactifoot_vision/.claude/worktrees/research-xg-detector/experiments/claude_fable"
)
OUT = be.ROOT / "ball/19"
INDEX = Path(
    "/home/kuba/projects/football_data/xg_shots_full/soccernet_720p/index_v1_2.parquet"
)
REFERENCES = {
    "frozen": "results/xgs_stack/real/summary.json",
    "R": "results/xgs_stack_dd2c/summary.json",
    "F": "results/xgs_stack_feet_t0_flags/summary.json",
}
CACHES = {"R": "cache/xgd_full4/E3", "F": "cache/xgd_full4_feet_t0/E3"}
LOCATION_ARMS = ("R", "F", "C", "S2", "S2_rule")
FAMILY = ("C", "S2", "T")
PRIMARY = "ALL"
STACK_ARMS = ("R", "G", "GS", "S", "Q", "ALL")
ALPHA_IMPROVE, ALPHA_EQUIV, MARGIN = 0.025, 0.05, 0.002
FLAG_PATTERN = re.compile(r"^(ball_(none|own|m\d+)|shot_from_ball)@-?\d+$")
RELATION = [
    "bf_dist",
    "bf_dist_missing",
    "bf_height",
    "bf_lateral",
    "bf_gap",
    "bf_conf",
    "bf_missing",
    "bf_src_own",
    "bf_src_m400",
    "bf_src_m800",
    "bf_src_none",
]
SOURCES = ("own", "m400", "m800")  # FEET_T0's order at t0: 0, -400, -800 ms
DIST_BANDS = ((0, 11), (11, 18), (18, 25), (25, np.inf))
HIGH_BALL = 0.6  # bf_height at or above: "ball high" (head proxy)
PITCH_MARGIN, LENGTH, WIDTH = 3.0, 105.0, 68.0  # the sanitation's on-pitch band


# ------------------------------------------------------------------ pure parts
def on_pitch(xy) -> bool:
    """Finite and at most 3 m outside the 105 x 68 m pitch (``sanitation.on_pitch`` for one point)."""
    x, y = (float(v) for v in np.asarray(xy, float).ravel()[:2])
    return (
        np.isfinite(x)
        and np.isfinite(y)
        and -PITCH_MARGIN <= x <= LENGTH + PITCH_MARGIN
        and -PITCH_MARGIN <= y <= WIDTH + PITCH_MARGIN
    )


def relation_missing() -> dict[str, float]:
    """The relation row of a frame without a ball or a shooter picked by it."""
    out = dict.fromkeys(RELATION, np.nan)
    out |= {"bf_dist_missing": 1.0, "bf_missing": 1.0, "bf_src_none": 1.0}
    out |= {f"bf_src_{s}": 0.0 for s in SOURCES}
    return out


def relation(
    shooter_box, ball_box, feet_xy, ball_xy, ball_conf: float, source: int
) -> dict[str, float]:
    """The ball-feet relation columns (registration): image boxes xyxy, pitch points (m), the ball's detector
    confidence and its source (0 own frame, 1 -400 ms, 2 -800 ms)."""
    sx1, sy1, sx2, sy2 = (float(v) for v in shooter_box)
    bx1, by1, bx2, by2 = (float(v) for v in ball_box)
    h = max(sy2 - sy1, 1.0)
    cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
    dx = max(sx1 - cx, 0.0, cx - sx2)
    dy = max(sy1 - cy, 0.0, cy - sy2)
    placed = on_pitch(feet_xy) and on_pitch(ball_xy)
    dist = (
        float(np.hypot(*(np.asarray(feet_xy, float) - np.asarray(ball_xy, float))))
        if placed
        else np.nan
    )
    out = {
        "bf_dist": dist,
        "bf_dist_missing": float(not placed),
        "bf_height": (sy2 - cy) / h,
        "bf_lateral": abs(cx - (sx1 + sx2) / 2) / h,
        "bf_gap": float(np.hypot(dx, dy)) / h,
        "bf_conf": float(ball_conf),
        "bf_missing": 0.0,
        "bf_src_none": 0.0,
    }
    return out | {f"bf_src_{s}": float(source == i) for i, s in enumerate(SOURCES)}


def flag_columns(columns) -> list[str]:
    """The ball-source columns of an E3 stack table (the one-hot ``ball_<level>@o`` and ``shot_from_ball@o``)."""
    return [c for c in columns if FLAG_PATTERN.match(c)]


def keep_flags(table: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """``table`` with its ball-source columns replaced by ``reference``'s (same shots, same order)."""
    if not table.index.equals(reference.index):
        raise ValueError("the two tables do not hold the same shots in the same order")
    out = table.copy()
    flags = flag_columns(table.columns)
    out[flags] = reference[flags].to_numpy()
    return out


def p_improve(boot: np.ndarray) -> float:
    """One-sided p of ΔLL < 0 from the bootstrap draws (C5's (1 + #{Δ_b >= 0}) / (B + 1))."""
    boot = boot[np.isfinite(boot)]
    return float((1 + np.sum(boot >= 0)) / (len(boot) + 1))


def p_equivalent(boot: np.ndarray, margin: float = MARGIN) -> float:
    """TOST p within ±margin from the bootstrap draws: the larger of the two one-sided p."""
    boot = boot[np.isfinite(boot)]
    upper = (1 + np.sum(boot >= margin)) / (len(boot) + 1)
    lower = (1 + np.sum(boot <= -margin)) / (len(boot) + 1)
    return float(max(upper, lower))


def holm(p: dict[str, float], alpha: float) -> dict[str, bool]:
    """Holm's step-down rejections at ``alpha`` (ties in input order); an arm without a p (None) has p = 1."""
    values = {k: (1.0 if v is None else float(v)) for k, v in p.items()}
    order = sorted(values, key=lambda k: values[k])
    rejected, m = dict.fromkeys(values, False), len(values)
    for i, k in enumerate(order):
        if values[k] > alpha / (m - i):
            break
        rejected[k] = True
    return rejected


def verdicts(
    p_imp: dict[str, float | None], p_eq: dict[str, float | None], points: dict
) -> dict[str, str]:
    """D3 per family arm: "adds value" (Holm-rejected improvement, point < 0), "equivalent" (Holm-rejected TOST),
    else "inconclusive"; "not run" for an arm without its p."""
    better, same = holm(p_imp, ALPHA_IMPROVE), holm(p_eq, ALPHA_EQUIV)
    out = {}
    for arm in p_imp:
        if p_imp[arm] is None:
            out[arm] = "not run"
        elif better[arm] and points[arm] < 0:
            out[arm] = "adds value"
        elif same[arm]:
            out[arm] = "equivalent"
        else:
            out[arm] = "inconclusive"
    return out


def interval(boot: np.ndarray, level: float) -> tuple[float, float]:
    boot = boot[np.isfinite(boot)]
    tail = 100 * (1 - level) / 2
    return float(np.percentile(boot, tail)), float(np.percentile(boot, 100 - tail))


def dist_band(dist: np.ndarray) -> np.ndarray:
    """Band label of each ``dist@0`` (m): "<11", "11-18", "18-25", ">=25", "missing"."""
    dist = np.asarray(dist, float)
    out = np.full(len(dist), "missing", dtype=object)
    for lo, hi in DIST_BANDS:
        name = (
            f"<{hi:g}"
            if lo == 0
            else (f">={lo:g}" if hi == np.inf else f"{lo:g}-{hi:g}")
        )
        out[(dist >= lo) & (dist < hi)] = name
    return out


def height_band(height: np.ndarray) -> np.ndarray:
    height = np.asarray(height, float)
    out = np.full(len(height), "missing", dtype=object)
    out[height >= HIGH_BALL] = "high"
    out[height < HIGH_BALL] = "low"
    return out


def content_scores(hsv: np.ndarray) -> np.ndarray:
    """(n - 1,) PySceneDetect ContentDetector score between consecutive frames: the mean over H, S and V of the mean
    absolute difference (``hsv``: (n, h, w, 3) uint8)."""
    a = hsv.astype(np.int16)
    return np.abs(np.diff(a, axis=0)).mean(axis=(1, 2)).mean(axis=1)


def t0_index(position_ms: int, fps: float = 25.0, span_s: float = 4.0) -> int:
    """The clip frame of the shot annotation: the clips start at max(t0 - 4 s, 0) (``collect_soccernet.cut_clip``)."""
    start = max(position_ms / 1000 - span_s, 0.0)
    return int(round((position_ms / 1000 - start) * fps))


# ------------------------------------------------------------------ read (xG environment)
def _xg_imports():
    sys.path.insert(0, str(XG / "scripts"))
    import ball_xg as bx
    import xgs_detector_rerun as rerun
    from xgpoc.detector import location as loc
    from xgpoc.detector.guard import no_statsbomb_reads
    from xgpoc.stack import analysis as sa
    from xgpoc.stack import fit as sf
    from xgpoc.stack import inputs as si
    from xgpoc.stack import protocol as pr
    from xgpoc.stack import stats as ss
    from xgpoc.stack_location import inputs as sl
    from xgs_run import PROTOCOL_SHA256

    return {
        "bx": bx,
        "rerun": rerun,
        "loc": loc,
        "guard": no_statsbomb_reads,
        "sa": sa,
        "sf": sf,
        "si": si,
        "pr": pr,
        "ss": ss,
        "sl": sl,
        "sha": PROTOCOL_SHA256,
    }


def design_check(m, spec, frozen: dict) -> list[str]:
    """Raise unless the spec's design is the protocol's up to columns the frozen zero-variance rule drops in this
    table only; return those columns (per arm, flattened)."""
    ours = json.loads(json.dumps(m["pr"].design(spec)))
    theirs = frozen["design"]
    same = {k for k in theirs if k not in ("arms", "gbm_columns")}
    if any(ours[k] != theirs[k] for k in same) or set(ours) != set(theirs):
        raise SystemExit("the design differs from the protocol's outside the arms")
    extra = set()
    for name, arm in theirs["arms"].items():
        mine = ours["arms"][name]
        more = [
            c
            for c in mine["dropped_by_zero_variance"]
            if c not in arm["dropped_by_zero_variance"]
        ]
        if (
            [c for c in arm["linear"] if c not in more] != mine["linear"]
            or (mine["spline"], mine["df"], mine["status"])
            != (arm["spline"], arm["df"], arm["status"])
            or set(arm["dropped_by_zero_variance"]) | set(more)
            != set(mine["dropped_by_zero_variance"])
        ):
            raise SystemExit(
                f"arm {name}: the design differs from the protocol's beyond zero-variance drops"
            )
        extra |= set(more)
    for name, cols in theirs["gbm_columns"].items():
        if [c for c in cols if c not in extra] != ours["gbm_columns"][name]:
            raise SystemExit(
                f"GBM {name}: the columns differ from the protocol's beyond zero-variance drops"
            )
    return sorted(extra)


def relation_table(m, frames: pd.DataFrame) -> pd.DataFrame:
    """``RELATION`` at t0 per shot (index shot_id, sorted) from F's filled t0 frame and its feet situation."""
    loc = m["loc"]
    rows = {}
    for shot_id, offset, _, filled, _ in loc.replay(frames, loc.FEET_T0.source):
        if offset != 0:
            continue
        row = relation_missing()
        if filled is not None:
            s = loc.feet_situation(filled)
            if s.ball is not None and s.shooter is not None and not s.shooter_by_goal:
                finite = np.isfinite(filled.ball_xy.astype(float)).all(axis=1)
                row = relation(
                    filled.detections.xyxy[s.shooter],
                    filled.ball.xyxy[finite][0],
                    s.xy[s.shooter],
                    s.ball,
                    float(filled.ball.confidence[finite][0]),
                    s.ball_source,
                )
        rows[shot_id] = row
    table = pd.DataFrame.from_dict(rows, orient="index")[RELATION]
    table.index.name = "shot_id"
    return table.sort_index()


def with_columns(m, spec, arm: str, columns: list[str], table: pd.DataFrame):
    """``spec`` with ``columns`` appended to the linear inputs of ``arm`` (constant ones dropped by the frozen rule)."""
    sf = m["sf"]
    constant = sf.constant_columns(table, columns)
    old = spec.arms[arm]
    linear = (*old.linear, *[c for c in columns if c not in constant])
    arms = dict(spec.arms) | {arm: sf.Arm(arm, old.status, linear, old.spline, old.df)}
    sf.check_no_constant(table, arms[arm])
    dropped = dict(spec.dropped) | {arm: [*spec.dropped[arm], *constant]}
    return sf.Spec(spec.candidate, arms, dropped, spec.gbm_columns)


def scores_row(m, y, p) -> dict:
    return json.loads(json.dumps(m["ss"].scores(y, p), default=float))


def read(args: argparse.Namespace) -> None:
    m = _xg_imports()
    bx, loc, sa, sf, si, sl, ss, rerun = (
        m[k] for k in ("bx", "loc", "sa", "sf", "si", "sl", "ss", "rerun")
    )
    started = datetime.now(UTC)
    args.out.mkdir(parents=True, exist_ok=True)
    frozen = m["pr"].load(XG / m["pr"].PROTOCOL, m["sha"])
    results, controls = {}, {}

    with m["guard"]():
        # ---------------- controls (stop unless exact)
        reference = si.build_population()
        y = si.load_outcomes(reference).aligned(reference)
        weights = ss.bootstrap_weights(reference.games)
        again = rerun.read_out(reference, y, frozen)
        expected = json.loads((XG / REFERENCES["frozen"]).read_text())
        controls["frozen_e3"] = rerun.summary_of(again) == expected
        if not controls["frozen_e3"]:
            raise SystemExit(
                "the frozen E3 read-out does not reproduce results/xgs_stack/real"
            )
        logger.info("control: frozen E3 read-out reproduced")

        populations = {}
        for arm, rule in (("R", loc.E3), ("F", loc.FEET_T0)):
            pop = sl.build_population(
                rule, e3=XG / CACHES[arm]
            )  # F: the replay must equal the cache exactly
            if (
                not pop.table.index.equals(reference.table.index)
                or (pop.folds != reference.folds).any()
            ):
                raise SystemExit(
                    f"arm {arm}: shots or folds differ from the reference population"
                )
            spec = sf.spec(frozen["design"]["chosen"], pop.table)
            extra = design_check(m, spec, frozen)
            result = sa.analyse(pop.table, pop.folds, y, weights, spec)
            expected = json.loads((XG / REFERENCES[arm]).read_text())
            controls[arm] = {
                "summary_reproduced": rerun.summary_of(result) == expected,
                "dropped_beyond_protocol": extra,
                "ALL_minus_Rstack": result["deltas"]["ALL"]["delta_ll"],
            }
            if not controls[arm]["summary_reproduced"]:
                raise SystemExit(f"arm {arm} does not reproduce {REFERENCES[arm]}")
            logger.info(
                "control: arm %s reproduced, ALL - R_stack %.4f",
                arm,
                controls[arm]["ALL_minus_Rstack"],
            )
            populations[arm], results[arm] = pop, result
        controls["F_replays_feet_cache"] = True  # sl.e3_table raises otherwise

        # ---------------- C: F + the relation columns in ALL
        f_table = populations["F"].table
        ids = list(f_table.index)
        cache_frames = pd.read_parquet(
            XG / CACHES["R"] / "frames.parquet", filters=[("shot_id", "in", ids)]
        )
        rel = relation_table(m, cache_frames)
        c_table = pd.concat([f_table, rel], axis=1)
        c_spec = with_columns(
            m, sf.spec(frozen["design"]["chosen"], c_table), "ALL", RELATION, c_table
        )
        results["C"] = sa.analyse(c_table, populations["F"].folds, y, weights, c_spec)
        logger.info(
            "arm C fitted (relation columns kept: %s)",
            [c for c in RELATION if c in c_spec.arms["ALL"].linear],
        )

        # ---------------- S2: the second-stage ball picks the shooter
        shots = bx.read_population()
        if sorted(shots.index) != ids:
            raise SystemExit("ticket 16's population is not the stack's")
        frames = bx.read_frames(ids)
        candidates = pd.read_parquet(bx.OUT / "candidates.parquet")
        s2_frames = bx.arm_frames(frames, bx.choose(candidates, "S2"))
        s2_features = loc.feature_rows(s2_frames, loc.FEET_T0)
        s2_e3, _ = sl.e3_table(s2_frames, s2_features, loc.FEET_T0)
        s2_rule = f_table.copy()
        s2_rule[si.E3_COLUMNS] = s2_e3.loc[ids, si.E3_COLUMNS].to_numpy()
        s2 = keep_flags(s2_rule, f_table)
        tables = {"S2": s2, "S2_rule": s2_rule}
        for arm, table in tables.items():
            spec = sf.spec(frozen["design"]["chosen"], table)
            controls[f"{arm}_dropped_beyond_protocol"] = design_check(m, spec, frozen)
            results[arm] = sa.analyse(table, populations["F"].folds, y, weights, spec)
            logger.info("arm %s fitted", arm)

    # ---------------- read-out
    games = reference.games
    f_p = results["F"]["probabilities"]
    ll = {
        arm: {
            s: ss.shot_log_loss(y, results[arm]["probabilities"][s], ss.CLIP)
            for s in STACK_ARMS
        }
        for arm in LOCATION_ARMS
    }
    boots, paired = {}, {}
    for arm in LOCATION_ARMS:
        if arm == "F":
            continue
        for s in STACK_ARMS:
            d = ll[arm][s] - ss.shot_log_loss(y, f_p[s], ss.CLIP)
            boot = ss.boot_means(weights, d)
            if s == PRIMARY:
                boots[arm] = boot
            paired[f"{arm}:{s}"] = {
                "delta_ll": float(d.mean()),
                "ci95": interval(boot, 0.95),
                "ci90": interval(boot, 0.90),
                "p_improve": p_improve(boot),
                "p_tost": p_equivalent(boot),
            }
    p_imp = {
        a: (paired[f"{a}:{PRIMARY}"]["p_improve"] if a != "T" else None) for a in FAMILY
    }
    p_eq = {
        a: (paired[f"{a}:{PRIMARY}"]["p_tost"] if a != "T" else None) for a in FAMILY
    }
    points = {
        a: (paired[f"{a}:{PRIMARY}"]["delta_ll"] if a != "T" else np.nan)
        for a in FAMILY
    }
    decision = verdicts(p_imp, p_eq, points)
    holm_levels = {
        a: {f"ci_{k}": interval(boots[a], 1 - ALPHA_IMPROVE * 2 / k) for k in (1, 2, 3)}
        | {
            f"ci_tost_{k}": interval(boots[a], 1 - 2 * ALPHA_EQUIV / k)
            for k in (1, 2, 3)
        }
        for a in ("C", "S2")
    }

    scores = {
        arm: {s: scores_row(m, y, results[arm]["probabilities"][s]) for s in STACK_ARMS}
        for arm in LOCATION_ARMS
    }

    # breakdowns (descriptive): distance band of F, ball-height proxy, F's own t0 ball source
    f_dist = f_table[si.col("dist", 0)].to_numpy(float)
    groups = {
        "dist_band": dist_band(f_dist),
        "ball_height": height_band(rel.bf_height.to_numpy(float)),
        "f_t0_ball_source": np.select(
            [rel[f"bf_src_{s}"].to_numpy() == 1 for s in SOURCES], list(SOURCES), "none"
        ),
    }
    breakdown = []
    for name, labels in groups.items():
        for level in sorted(set(labels)):
            mask = labels == level
            row = {
                "split": name,
                "level": level,
                "shots": int(mask.sum()),
                "goals": int(y[mask].sum()),
                "ll_F": float(ll["F"][PRIMARY][mask].mean()),
            }
            for arm in ("R", "C", "S2", "S2_rule"):
                d = ll[arm][PRIMARY] - ll["F"][PRIMARY]
                lo, hi = interval(ss.boot_means(weights, d, mask), 0.95)
                row |= {
                    f"{arm}": float(d[mask].mean()),
                    f"{arm}_lo": lo,
                    f"{arm}_hi": hi,
                }
            breakdown.append(row)
    breakdown = pd.DataFrame(breakdown)
    breakdown.to_csv(args.out / "breakdown.csv", index=False)

    # how much the t0 shot location moved (S2 vs F; aligned shot_x and distance to goal)
    dx = (s2[si.col("shot_x", 0)] - f_table[si.col("shot_x", 0)]).abs()
    dd = (s2[si.col("dist", 0)] - f_table[si.col("dist", 0)]).abs()
    both = s2[si.col("shot_missing", 0)].eq(0) & f_table[si.col("shot_missing", 0)].eq(
        0
    )
    changed = both & ((dx > 0.01) | (dd > 0.01))
    location = {
        "S2_vs_F_location_changed_share": float(changed.sum() / both.sum()),
        "S2_vs_F_dist_change_median_when_changed_m": float(dd[changed].median()),
        "S2_vs_F_dist_change_gt5m_share": float((both & (dd > 5)).sum() / both.sum()),
        "relation_missing_share": float(rel.bf_missing.mean()),
        "relation_dist_median_m": float(rel.bf_dist.median()),
        "ball_high_share_among_with_ball": float(
            (rel.bf_height >= HIGH_BALL).sum() / (rel.bf_missing == 0).sum()
        ),
    }

    probs = pd.DataFrame(
        {
            f"{arm}:{s}": results[arm]["probabilities"][s]
            for arm in LOCATION_ARMS
            for s in STACK_ARMS
        },
        index=f_table.index,
    )
    probs.insert(0, "game", games)
    probs.to_csv(args.out / "probabilities.csv")
    rel.to_parquet(args.out / "relation.parquet")
    record = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "started": started.isoformat(timespec="seconds"),
        "ticket": "ball-detection 19",
        "registration": "ticket 19 comment 2026-10-27 13:55",
        "population": {
            "shots": len(ids),
            "matches": int(pd.Series(games).nunique()),
            "goals": int(y.sum()),
        },
        "protocol_sha256": m["sha"],
        "controls": controls,
        "primary_stack_arm": PRIMARY,
        "paired_vs_F": paired,
        "family": {
            "p_improve": p_imp,
            "p_tost": p_eq,
            "verdict_with_pT_1": decision,
            "alpha_improve": ALPHA_IMPROVE,
            "alpha_equiv": ALPHA_EQUIV,
            "margin": MARGIN,
        },
        "holm_level_intervals": holm_levels,
        "scores": scores,
        "deltas_vs_Rstack": {
            arm: json.loads(json.dumps(results[arm]["deltas"], default=float))
            for arm in LOCATION_ARMS
        },
        "chosen_c_ALL": {arm: results[arm]["chosen_c"]["ALL"] for arm in LOCATION_ARMS},
        "c_relation_columns": [c for c in RELATION if c in c_spec.arms["ALL"].linear],
        "location": location,
    }
    (args.out / "read.json").write_text(
        json.dumps(record, indent=2, default=float) + "\n"
    )
    print(
        json.dumps(
            {k: record[k] for k in ("controls", "family", "location")},
            indent=2,
            default=float,
        )
    )
    print(breakdown.to_string())


# ------------------------------------------------------------------ clips (feasibility of T, CPU)
def _clip_worker(job: tuple) -> dict:
    import cv2

    shot_id, clip, position_ms, frame_paths, check = job
    cap = cv2.VideoCapture(clip)
    out = {
        "shot_id": shot_id,
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "n_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    t0 = t0_index(position_ms)
    out["t0_index"] = t0
    last = t0 + (3 if check else 0)
    small, grey = [], {}
    for i in range(last + 1):
        ok, frame = cap.read()
        if not ok:
            break
        if i >= t0 - 50:
            small.append(
                cv2.cvtColor(
                    cv2.resize(frame, (256, 144), interpolation=cv2.INTER_AREA),
                    cv2.COLOR_BGR2HSV,
                )
            )
        if check and (abs(i - t0) <= 3 or abs(i - (t0 - 10)) <= 3):
            grey[i] = cv2.cvtColor(
                cv2.resize(frame, (256, 144), interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2GRAY,
            )
    cap.release()
    out["frames_read"] = i + 1 if small else 0
    window = (
        np.stack(small[: min(len(small), 51)])
        if small
        else np.zeros((0, 144, 256, 3), np.uint8)
    )
    out["frames_in_window"] = len(window)
    score = content_scores(window) if len(window) > 1 else np.zeros(0)
    out["max_score_2s"] = float(score.max()) if len(score) else np.nan
    out["max_score_1s"] = float(score[-25:].max()) if len(score) else np.nan
    # the cut nearest t0 within the window, in frames before t0 (score index k: frame t0-50+k -> t0-49+k)
    cuts = np.flatnonzero(score >= 27.0)
    out["n_cuts_2s"] = int(len(cuts))
    out["last_cut_frames_before_t0"] = int(50 - (cuts[-1] + 1)) if len(cuts) else -1
    if check:
        for name, centre in (("+0", t0), ("-400", t0 - 10)):
            ref = cv2.imread(frame_paths[name], cv2.IMREAD_GRAYSCALE)
            if ref is None:
                continue
            ref = cv2.resize(ref, (256, 144), interpolation=cv2.INTER_AREA).astype(
                np.int16
            )
            errs = {
                k - centre: float(np.abs(grey[k].astype(np.int16) - ref).mean())
                for k in grey
                if abs(k - centre) <= 3
            }
            best = min(errs, key=errs.get)
            out[f"align_{name}_best_shift"] = best
            out[f"align_{name}_err_best"] = errs[best]
            out[f"align_{name}_err_0"] = errs.get(0, np.nan)
    return out


def clips(args: argparse.Namespace) -> None:
    from multiprocessing import Pool

    import cv2

    index = pd.read_parquet(
        INDEX,
        columns=[
            "shot_id",
            "game",
            "is_penalty",
            "clip_path",
            "position_ms",
            "frame_path",
            "frame_path_-400",
        ],
    )
    plan = pd.read_csv(XG / "results/xgv_split_plan/plan.csv", usecols=["game", "role"])
    role = index.game.map(plan.set_index("game").role)
    shots = (
        index[(role == "training") & ~index.is_penalty.astype(bool)]
        .set_index("shot_id")
        .sort_index()
    )
    if args.limit:
        shots = shots.iloc[: args.limit]
    rng = np.random.default_rng(0)
    check = set(rng.choice(shots.index.to_numpy(), min(200, len(shots)), replace=False))
    jobs = [
        (
            sid,
            r.clip_path,
            int(r.position_ms),
            {"+0": r.frame_path, "-400": r["frame_path_-400"]},
            sid in check,
        )
        for sid, r in shots.iterrows()
    ]
    cv2.setNumThreads(1)
    with Pool(args.workers) as pool:
        rows = pool.map(_clip_worker, jobs, chunksize=16)
    table = pd.DataFrame(rows).set_index("shot_id")
    table["game"] = shots.game
    table["position_ms"] = shots.position_ms
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out / "clips.parquet")

    # a look at a few flagged cuts: frames before / after the nearest cut, and the t0 frame
    sheet = args.out / "cuts_gallery"
    sheet.mkdir(exist_ok=True)
    flagged = table.index[table.last_cut_frames_before_t0 >= 0].to_numpy()
    for k, sid in enumerate(rng.choice(flagged, min(12, len(flagged)), replace=False)):
        r = table.loc[sid]
        cut = int(r.t0_index - r.last_cut_frames_before_t0)
        cap = cv2.VideoCapture(shots.at[sid, "clip_path"])
        tiles = {}
        for i in range(int(r.t0_index) + 1):
            ok, frame = cap.read()
            if not ok:
                break
            if i in (cut - 1, cut, int(r.t0_index)):
                tiles[i] = cv2.resize(frame, (426, 240))
        cap.release()
        row = np.hstack(
            [
                tiles.get(i, np.zeros((240, 426, 3), np.uint8))
                for i in (cut - 1, cut, int(r.t0_index))
            ]
        )
        cv2.imwrite(
            str(sheet / f"{k:02d}_{int(r.last_cut_frames_before_t0):02d}f.jpg"), row
        )

    summary = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "shots": len(table),
        "fps": {str(k): int(v) for k, v in table.fps.round(3).value_counts().items()},
        "n_frames": {
            str(k): int(v) for k, v in table.n_frames.value_counts().head(5).items()
        },
        "clip_start_clipped(t0<4s)": int((table.position_ms < 4000).sum()),
        "frames_in_window_51": float((table.frames_in_window == 51).mean()),
        "cut_2s_share": float((table.n_cuts_2s > 0).mean()),
        "cut_1s_share": float((table.max_score_1s >= 27.0).mean()),
        "cut_within_10f_share": float(
            (
                (table.last_cut_frames_before_t0 >= 0)
                & (table.last_cut_frames_before_t0 < 10)
            ).mean()
        ),
        "score_quantiles_2s": {
            str(q): float(table.max_score_2s.quantile(q))
            for q in (0.5, 0.9, 0.95, 0.99)
        },
    }
    for name in ("+0", "-400"):
        col = f"align_{name}_best_shift"
        if col in table:
            summary[f"align_{name}"] = {
                str(k): int(v)
                for k, v in table[col].dropna().astype(int).value_counts().items()
            }
            summary[f"align_{name}_err_best_median"] = float(
                table[f"align_{name}_err_best"].median()
            )
    (args.out / "clips.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    global OUT
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("read")
    p.set_defaults(func=read, out=OUT / "xg")
    p = sub.add_parser("clips")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=clips, out=OUT / "clips")
    for p in sub.choices.values():
        p.add_argument("--out", type=Path, default=p.get_default("out"))
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
