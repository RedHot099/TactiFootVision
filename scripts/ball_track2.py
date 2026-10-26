"""ball-detection 12: four fixes to the ball tracking, each on a switch, on games 2+3 of GSR valid (CPU).

    uv run python scripts/ball_track2.py check    # every fix off -> the tracks of tickets 09 and 04, identical
    uv run python scripts/ball_track2.py tune     # the registered grid, scored on the off-benchmark frames only
    uv run python scripts/ball_track2.py read     # each fix alone and all together vs ticket 09's tracks, paired

``track_ball2`` is a copy of the package's ``track_ball`` (``tactifoot_vision.pipeline.ball``; nothing in ``src/``
changes): the unchanged steps are the package's own private helpers, the changed ones (``_usable``, ``_follow``,
``_fill_gaps``, ``_between``) are copied here with a switch per fix (``Fixes``). With every switch off it gives the
package's tracks; ``check`` asserts that on the phase 1 candidate's dumps. The fixes:

1. ``image_fill``: a gap a -> b is filled in the pixels of each frame k: (1 - s) M_ak(centre a) + s M_bk(centre b),
   M the camera motion of ``ball_temporal motion`` composed between the frames and s the share of the time, the box
   size as before; the pitch position is the point through frame k's homography. A motion break in the gap falls back
   to the package's linear interpolation on the pitch.
2. ``corridor`` (m): in every frame of a gap, the usable candidate within ``corridor`` m of the interpolated path
   (nearest, a more confident one slightly farther, as in the gate) becomes a measured ball; one pass along the path
   a -> b, then the rest of the gap is filled between the measured balls.
3. ``restart_bounds``: ``_follow`` keeps which balls (re)started the track. A restart starts a new segment and a gap
   between two segments is neither filled nor searched (a restart lies outside the gate: another object).
4. ``spot_gate`` (m) / ``spot_window`` (s): a candidate within ``spot_radius`` of a painted spot is no longer dropped.
   It may be chosen through the gate when it lies within ``spot_gate`` m of the predicted position (fits the track)
   but cannot restart the track, unless it moves: not static over ``spot_window`` s (the package's static test with
   ``static_radius`` and ``static_share``), then it is an ordinary candidate. ``spot_window`` None: only the gate.

``tune`` runs the grid registered in ticket 12 (``GRID``) on the 39 clips of games 2+3 and scores every setting on
the frames that are not benchmark frames (frame - 1 not a multiple of 5; exactly one truth ball with ``bbox_pitch``,
on the pitch): well placed when the chosen box centre, through the frame's homography carried by the camera motion
(``ball_temporal.clip_homographies``), lies within 5 m of the truth; no ball or no homography is a miss. The best
setting of each row (ties: the first in grid order) goes to ``tune/selected.json``. ``read`` tracks each row's
setting and reads it once on the placement frames of games 2+3 (the frozen homographies of their clips only, as in
ticket 11) against ticket 09's tracks: ``detector_bench.placement_rows``, ``ball_scorecard.Draws`` /
``placement_draws`` / ``paired`` / ``barrier``. A row wins when the lower bound of its paired gain in well placed is
> 0 and the barrier holds; ``chosen.json`` is the winning row with the largest point gain (ties: fewer fixes), every
fix off when none wins. Games 4 and 5 and GSR test are never tracked here.
"""

import argparse
import copy
import dataclasses
import json
import logging
import multiprocessing
import sys
import time
from datetime import UTC, datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ball_errors as be  # noqa: E402
import ball_projection as bp  # noqa: E402
import ball_scorecard as bs  # noqa: E402
import ball_temporal as bt  # noqa: E402
import detector_bench as bench  # noqa: E402

from tactifoot_vision.enums import DetectionKey  # noqa: E402
from tactifoot_vision.pipeline import PipelineResult  # noqa: E402
from tactifoot_vision.pipeline import ball as pb  # noqa: E402
from tactifoot_vision.pipeline.postprocess import metres_per_unit  # noqa: E402
from tactifoot_vision.pitch.homography import frame_to_pitch  # noqa: E402
from tactifoot_vision.pitch.pitch import pitch_markings  # noqa: E402

OUT = be.ROOT / "ball/12"
REFERENCE = (
    be.ROOT / "ball/09/track_valid"
)  # the phase 1 candidate: 04's setting + the 45 px filter
REFERENCE_READ = be.ROOT / "ball/09/read_valid/read.json"
TRACK04 = be.ROOT / "ball/04/track"  # 04's own tracks of games 2+3, without the filter
SETTING = bt.OUT / "tune/chosen.json"
GAMES = bt.SELECTION  # games 2+3 only
FAR = bench.PLACEMENT["far_m"]

logger = logging.getLogger("ball_track2")


@dataclasses.dataclass(frozen=True)
class Fixes:
    """The switches of the four fixes; the defaults are the package's tracker."""

    image_fill: bool = False  # 1
    corridor: float | None = None  # 2: metres around the interpolated path
    restart_bounds: bool = False  # 3
    spot_gate: float | None = (
        None  # 4: metres around the prediction a spot candidate may be followed in
    )
    spot_window: float | None = (
        None  # 4: seconds a spot candidate must not stay in place to count as moving
    )

    def __post_init__(self) -> None:
        for name in ("corridor", "spot_gate", "spot_window"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive or None")
        if self.spot_window is not None and self.spot_gate is None:
            raise ValueError("spot_window needs spot_gate (fix 4 on)")


OFF = Fixes()


# ------------------------------------------------------------------ the tracker
def track_ball2(
    result: PipelineResult,
    motion: np.ndarray | None,
    fixes: Fixes = OFF,
    min_confidence: float = 0.12,
    restart_confidence: float = 0.45,
    restart_after: float = 0.25,
    gate: float = 1.5,
    gate_speed: float = 32.5,
    max_gap: float = 0.5,
    static_radius: float = 0.7,
    static_window: float = 3.2,
    static_share: float = 0.75,
    spot_radius: float = 0.6,
    max_person_iou: float = 0.5,
    isolation: float = 0.15,
) -> tuple[PipelineResult, dict]:
    """``track_ball`` with ``fixes``: the tracked copy of ``result`` and counts of what the fixes did. ``motion[t]``
    maps frame t's pixels to frame t + 1's (NaN at a break; ``ball_temporal motion``), needed by ``image_fill``."""
    options = pb._Options(
        min_confidence=min_confidence,
        restart_confidence=restart_confidence,
        restart_after=restart_after,
        gate=gate,
        gate_speed=gate_speed,
        max_gap=max_gap,
        static_radius=static_radius,
        static_window=static_window,
        static_share=static_share,
        spot_radius=spot_radius,
        max_person_iou=max_person_iou,
        isolation=isolation,
    )
    frames = result.frames
    if fixes.image_fill and (motion is None or len(motion) != len(frames) - 1):
        raise ValueError("image_fill needs one motion matrix per pair of frames")
    times = np.array([f.timestamp for f in frames], dtype=float)
    if np.any(np.diff(times) <= 0):
        raise ValueError("track_ball needs frames in order of increasing timestamps")
    scale = metres_per_unit(result.pitch)
    positions = [pb._candidate_xy(f) * scale for f in frames]
    confidences = [pb._confidence(f.ball_candidates) for f in frames]
    usable, spot_only = _usable(
        frames, positions, confidences, times, result.pitch, scale, options, fixes
    )
    chosen, restarts = _follow(
        positions, confidences, usable, spot_only, times, options, fixes
    )
    kept = pb._remove_outliers(positions, confidences, chosen, times, options)
    balls = [
        pb._chosen_ball(frame, choice if keep else None)
        for frame, choice, keep in zip(frames, chosen, kept, strict=True)
    ]
    segment = np.cumsum(
        restarts
    )  # frames of one segment share the count of restarts up to them
    stats = _fill_gaps(
        frames,
        balls,
        kept,
        times,
        options,
        fixes,
        segment,
        motion,
        positions,
        confidences,
        usable,
        spot_only,
        scale,
    )
    stats |= {
        "frames": len(frames),
        "kept": int(kept.sum()),
        "restarts": int(np.sum(restarts)),
        "spot_chosen": int(
            sum(
                c is not None and k and bool(o[c])
                for c, k, o in zip(chosen, kept, spot_only, strict=True)
            )
        ),
    }
    tracked = dataclasses.replace(
        result,
        frames=[
            dataclasses.replace(frame, ball=ball)
            for frame, ball in zip(frames, balls, strict=True)
        ],
    )
    return tracked, stats


def _usable(
    frames,
    positions: list[np.ndarray],
    confidences: list[np.ndarray],
    times: np.ndarray,
    pitch,
    scale: np.ndarray,
    options,
    fixes: Fixes,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per frame ``(usable, spot_only)``: the candidates that may be the live ball (the package's step 1) and, with
    fix 4, the spot candidates that may be followed through the gate only."""
    _, spots = pitch_markings(pitch)
    spots = spots * scale
    statics = pb._static(positions, times, options)
    spot_static = None
    if fixes.spot_gate is not None and fixes.spot_window is not None:
        spot_static = pb._static(
            positions,
            times,
            dataclasses.replace(options, static_window=fixes.spot_window),
        )
    usable, spot_only = [], []
    for i, (frame, xy, confidence, static) in enumerate(
        zip(frames, positions, confidences, statics, strict=True)
    ):
        keep = np.isfinite(xy).all(axis=1) & ~static
        keep &= confidence >= options.min_confidence
        spot = np.zeros(len(xy), dtype=bool)
        if len(xy):
            to_spot = np.linalg.norm(xy[:, None] - spots[None], axis=2).min(axis=1)
            spot = to_spot <= options.spot_radius
        if len(xy) and len(frame.detections):
            iou = sv.box_iou_batch(frame.ball_candidates.xyxy, frame.detections.xyxy)
            keep &= iou.max(axis=1) <= options.max_person_iou
        if fixes.spot_gate is None:
            usable.append(keep & ~spot)
            spot_only.append(np.zeros(len(xy), dtype=bool))
            continue
        free = ~spot if spot_static is None else ~spot | ~spot_static[i]
        usable.append(keep & free)
        spot_only.append(keep & ~free)
    return usable, spot_only


def _follow(
    positions: list[np.ndarray],
    confidences: list[np.ndarray],
    usable: list[np.ndarray],
    spot_only: list[np.ndarray],
    times: np.ndarray,
    options,
    fixes: Fixes,
) -> tuple[list[int | None], np.ndarray]:
    """The candidate followed in each frame (``None`` for none) and which frames' ball (re)started the track."""
    chosen: list[int | None] = []
    restarts = np.zeros(len(times), dtype=int)
    last: np.ndarray | None = None
    last_time = 0.0
    velocity = np.zeros(2)
    for i, (xy, confidence, keep, only, time_) in enumerate(
        zip(positions, confidences, usable, spot_only, times, strict=True)
    ):
        indices = np.flatnonzero(keep)
        choice = None
        if last is not None:
            dt = time_ - last_time
            predicted = last + velocity * options.elapsed(dt)
            choice = _in_gate(
                xy,
                confidence,
                indices,
                np.flatnonzero(only),
                predicted,
                dt,
                options,
                fixes,
            )
        restarted = choice is None and (
            last is None or time_ - last_time > options.restart_after + pb._EPS
        )
        if restarted:
            choice = pb._most_confident(confidence, indices, options.restart_confidence)
        chosen.append(choice)
        if choice is None:
            continue
        if restarted:
            velocity = np.zeros(2)
            restarts[i] = 1
        else:
            velocity = (velocity + (xy[choice] - last) / (time_ - last_time)) / 2
        last, last_time = xy[choice], time_
    return chosen, restarts


def _in_gate(xy, confidence, indices, only, predicted, dt, options, fixes: Fixes):
    """The package's gate; with fix 4 the spot candidates compete too, within ``spot_gate`` m of the prediction."""
    if not len(only):
        return pb._in_gate(xy, confidence, indices, predicted, dt, options)
    return _nearest(
        xy, confidence, indices, only, predicted, options.radius(dt), fixes.spot_gate
    )


def _nearest(xy, confidence, indices, only, centre, radius, spot_radius):
    """Of ``indices`` within ``radius`` m of ``centre`` and ``only`` within min(``radius``, ``spot_radius``), the one
    with the lowest distance - ``_CONFIDENCE_WEIGHT`` * confidence; None when there is none."""
    both = np.concatenate([indices, only]).astype(int)
    limit = np.concatenate(
        [np.full(len(indices), radius), np.full(len(only), min(radius, spot_radius))]
    )
    distance = np.linalg.norm(xy[both] - centre, axis=1)
    inside = distance < limit
    if not inside.any():
        return None
    score = distance[inside] - pb._CONFIDENCE_WEIGHT * confidence[both[inside]]
    return int(both[inside][np.argmin(score)])


def _carry(motion: np.ndarray, a: int, b: int) -> list[np.ndarray] | None:
    """``F[k - a]`` = the camera motion frame a -> frame k for k = a .. b, None when the motion breaks in between."""
    steps = motion[a:b]
    if np.isnan(steps).any():
        return None
    carried = [np.eye(3)]
    for step in steps:
        carried.append(step @ carried[-1])
    return carried


def _apply(matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
    p = matrix @ np.array([point[0], point[1], 1.0])
    return p[:2] / p[2]


def _between_image(
    start: sv.Detections,
    end: sv.Detections,
    share: float,
    homography: np.ndarray,
    start_to_k: np.ndarray,
    end_to_k: np.ndarray,
) -> sv.Detections:
    """The ball ``share`` of the way from ``start`` to ``end`` in frame k's pixels (fix 1), flagged as interpolated:
    both centres carried to frame k by the camera motion, the box size interpolated as in the package."""
    centres = [(d.xyxy[0, :2] + d.xyxy[0, 2:]) / 2 for d in (start, end)]
    centre = (1 - share) * _apply(start_to_k, centres[0]) + share * _apply(
        end_to_k, centres[1]
    )
    size = (1 - share) * (start.xyxy[0, 2:] - start.xyxy[0, :2]) + share * (
        end.xyxy[0, 2:] - end.xyxy[0, :2]
    )
    ball = copy.deepcopy(start)
    ball.xyxy = np.concatenate([centre - size / 2, centre + size / 2])[None].astype(
        start.xyxy.dtype
    )
    if ball.confidence is not None:
        ball.confidence = np.full(1, np.nan, dtype=ball.confidence.dtype)
    ball.data[DetectionKey.PITCH_XY] = frame_to_pitch(centre[None], homography).astype(
        start.data[DetectionKey.PITCH_XY].dtype
    )
    ball.data[DetectionKey.INTERPOLATED] = np.ones(1, dtype=bool)
    return ball


def _interpolate(
    balls: list[sv.Detections],
    frames,
    times: np.ndarray,
    x: int,
    y: int,
    k: int,
    carried: list[np.ndarray] | None,
    origin: int,
) -> sv.Detections:
    """The ball of frame k between the balls of frames x and y: on the pitch as in the package, or in frame k's
    pixels (fix 1) when ``carried`` holds the camera motion from frame ``origin`` to every frame up to y."""
    share = (times[k] - times[x]) / (times[y] - times[x])
    if carried is None:
        return pb._between(balls[x], balls[y], share, frames[k].homography)
    to_k = carried[k - origin]
    return _between_image(
        balls[x],
        balls[y],
        share,
        frames[k].homography,
        to_k @ np.linalg.inv(carried[x - origin]),
        to_k @ np.linalg.inv(carried[y - origin]),
    )


def _fill_gaps(
    frames,
    balls: list[sv.Detections],
    kept: np.ndarray,
    times: np.ndarray,
    options,
    fixes: Fixes,
    segment: np.ndarray,
    motion: np.ndarray | None,
    positions: list[np.ndarray],
    confidences: list[np.ndarray],
    usable: list[np.ndarray],
    spot_only: list[np.ndarray],
    scale: np.ndarray,
) -> dict:
    """Fill the gaps of up to ``max_gap`` s between kept balls in place (the package's step 5, with fixes 1-3)."""
    stats = {
        "filled": 0,
        "image_filled": 0,
        "pitch_fallback_filled": 0,
        "corridor_measured": 0,
        "gaps_across_restart": 0,
    }
    rows = np.flatnonzero(kept)
    for a, b in zip(rows[:-1], rows[1:], strict=True):
        if b == a + 1 or times[b] - times[a] > options.max_gap + pb._EPS:
            continue
        if fixes.restart_bounds and segment[a] != segment[b]:
            stats["gaps_across_restart"] += 1
            continue
        carried = _carry(motion, a, b) if fixes.image_fill else None

        anchors = [a]
        if fixes.corridor is not None:
            for k in range(a + 1, b):
                if frames[k].homography is None or not len(positions[k]):
                    continue
                path = (
                    _interpolate(balls, frames, times, a, b, k, carried, a).data[
                        DetectionKey.PITCH_XY
                    ][0]
                    * scale
                )
                pick = _nearest(
                    positions[k],
                    confidences[k],
                    np.flatnonzero(usable[k]),
                    np.flatnonzero(spot_only[k]),
                    path,
                    fixes.corridor,
                    np.inf if fixes.spot_gate is None else fixes.spot_gate,
                )
                if pick is not None:
                    balls[k] = pb._chosen_ball(frames[k], pick)
                    anchors.append(k)
                    stats["corridor_measured"] += 1
        anchors.append(b)
        for x, y in zip(anchors[:-1], anchors[1:], strict=True):
            for k in range(x + 1, y):
                if frames[k].homography is not None:
                    balls[k] = _interpolate(balls, frames, times, x, y, k, carried, a)
                    stats["filled"] += 1
                    key = "pitch_fallback_filled" if carried is None else "image_filled"
                    if fixes.image_fill:
                        stats[key] += 1
    return stats


# ------------------------------------------------------------------ the grid (registered 2026-10-26 10:25)
CORRIDORS = (1.0, 2.0, 3.0, 5.0)
SPOT_GATES = (1.0, 2.0, 4.0)
SPOT_WINDOWS = (0.4, 1.0, None)  # None: no "moves" exception, the gate only
ROWS = ("off", "image_fill", "corridor", "restart_bounds", "spots", "all")
FIX_COUNT = {
    "off": 0,
    "image_fill": 1,
    "corridor": 1,
    "restart_bounds": 1,
    "spots": 1,
    "all": 4,
}


def grid(row: str) -> list[Fixes]:
    """The registered settings of a row, in tie-break order."""
    spots = list(product(SPOT_GATES, SPOT_WINDOWS))
    return {
        "off": [OFF],
        "image_fill": [Fixes(image_fill=True)],
        "corridor": [Fixes(corridor=c) for c in CORRIDORS],
        "restart_bounds": [Fixes(restart_bounds=True)],
        "spots": [Fixes(spot_gate=g, spot_window=w) for g, w in spots],
        "all": [Fixes(True, c, True, g, w) for c, (g, w) in product(CORRIDORS, spots)],
    }[row]


def fixes_name(fixes: Fixes) -> str:
    parts = []
    if fixes.image_fill:
        parts.append("image")
    if fixes.corridor is not None:
        parts.append(f"corridor{fixes.corridor:g}")
    if fixes.restart_bounds:
        parts.append("bounds")
    if fixes.spot_gate is not None:
        window = "off" if fixes.spot_window is None else f"{fixes.spot_window:g}"
        parts.append(f"spot{fixes.spot_gate:g}w{window}")
    return "+".join(parts) or "off"


# ------------------------------------------------------------------ tracking clips
def _clip_worker(job: tuple) -> dict:
    """One clip: dd2c's top-1 table and, per fixes, the tracks (with the chosen centre's pitch position in corner
    metres through the frame's carried homography: ``pitch_x`` / ``pitch_y``), the tracker's seconds and counts."""
    logging.getLogger("tactifoot_vision").setLevel(logging.WARNING)
    clip, setting, all_fixes, folder, motion_dir, limit = job
    candidates = bt.clip_candidates(clip, folder)
    frames = len(bt.clip_frames(bench.GSR / "valid" / clip))
    homographies = bt.clip_homographies(clip, frames, bt._FROZEN, motion_dir)
    motion = bt.load_motion(clip, motion_dir)
    result = bt.clip_result(bt.drop_long_balls(candidates, limit), homographies)
    out = {"clip": clip, "top1": bt.top_balls(candidates, frames), "tracks": []}
    for fixes in all_fixes:
        clock = time.perf_counter()
        tracked, stats = track_ball2(result, motion, fixes, **setting)
        seconds = time.perf_counter() - clock
        table = bt.chosen_balls(tracked)
        table["pitch_x"], table["pitch_y"] = pitch_points(table, homographies).T
        out["tracks"].append((table, seconds, stats))
    out["top1"].insert(0, "clip", clip)
    for table, _, _ in out["tracks"]:
        table.insert(0, "clip", clip)
    return out


def pitch_points(table: pd.DataFrame, homographies) -> np.ndarray:
    """``(frames, 2)``: each chosen box centre through its frame's homography (NaN without a ball or homography)."""
    found = np.full((len(table), 2), np.nan)
    centres = be.centres(table[bt.COLUMNS].to_numpy(float))
    for i, (status, h) in enumerate(zip(table.status, homographies, strict=True)):
        if status != "none" and h is not None:
            found[i] = frame_to_pitch(centres[i][None], h)[0]
    return found


def run(
    clips: list[str],
    all_fixes: list[Fixes],
    workers: int,
    setting: dict,
    limit: float | None = bt.POST_FILTER,
    folder: Path = bt.DUMP,
    motion_dir: Path = bt.MOTION_DIR,
) -> tuple[pd.DataFrame, list[dict]]:
    """``(top-1 table, per fixes {tracks, ms_per_frame, stats})`` over ``clips`` of GSR valid."""
    jobs = [(c, setting, all_fixes, folder, motion_dir, limit) for c in clips]
    with multiprocessing.Pool(
        workers, initializer=bt._init_worker, initargs=("valid",)
    ) as pool:
        per_clip = pool.map(_clip_worker, jobs)
    top1 = pd.concat([c["top1"] for c in per_clip], ignore_index=True)
    found = []
    for i, fixes in enumerate(all_fixes):
        tables = [c["tracks"][i] for c in per_clip]
        tracks = pd.concat([t for t, _, _ in tables], ignore_index=True)
        stats = {}
        for _, _, s in tables:
            for key, value in s.items():
                stats[key] = stats.get(key, 0) + value
        found.append(
            {
                "fixes": fixes,
                "tracks": tracks,
                "ms_per_frame": 1000 * sum(s for _, s, _ in tables) / len(tracks),
                "stats": stats,
            }
        )
    return top1, found


def clips_of_games() -> list[str]:
    clips = bt.games_clips(GAMES)
    if len(clips) != 39:
        raise SystemExit(f"games 2+3 have 39 clips, found {len(clips)}")
    return clips


def setting_04() -> dict:
    return json.loads(SETTING.read_text())


# ------------------------------------------------------------------ check
TRACK_COLUMNS = ["clip", "frame", "status", *bt.COLUMNS, "conf"]


def same_tracks(mine: pd.DataFrame, theirs: pd.DataFrame) -> dict:
    """Whether two tracks tables agree exactly on ``TRACK_COLUMNS`` (NaN = NaN), and in how many frames they do not."""
    a = mine[TRACK_COLUMNS].reset_index(drop=True)
    b = theirs[TRACK_COLUMNS].reset_index(drop=True)
    if len(a) != len(b) or not a[["clip", "frame"]].equals(b[["clip", "frame"]]):
        return {"identical": False, "frames": len(a), "differ": None}
    differ = ~(
        (a.status == b.status).to_numpy()
        & np.all(
            [
                (a[c].to_numpy() == b[c].to_numpy())
                | (np.isnan(a[c].to_numpy()) & np.isnan(b[c].to_numpy()))
                for c in [*bt.COLUMNS, "conf"]
            ],
            axis=0,
        )
    )
    return {
        "identical": not differ.any(),
        "frames": len(a),
        "differ": int(differ.sum()),
    }


def of_clips(table: pd.DataFrame, clips: list[str]) -> pd.DataFrame:
    keep = table["clip"].astype(str).isin(clips).to_numpy()
    out = table[keep].copy()
    out["clip"] = out["clip"].astype(str)
    return out.reset_index(drop=True)


def check(args: argparse.Namespace) -> None:
    """Every fix off on games 2+3: with the 45 px filter = ticket 09's tracks, without it = ticket 04's."""
    clips = clips_of_games()
    setting = setting_04()
    record = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "clips": len(clips),
    }
    for name, limit, reference in (
        ("09_filter45", bt.POST_FILTER, REFERENCE),
        ("04_no_filter", None, TRACK04),
    ):
        top1, (found,) = run(clips, [OFF], args.workers, setting, limit)
        theirs = of_clips(pd.read_parquet(reference / "tracks.parquet"), clips)
        their_top = of_clips(pd.read_parquet(reference / "top1.parquet"), clips)
        record[name] = {
            "reference": str(reference),
            "tracks": same_tracks(found["tracks"], theirs),
            "top1": same_tracks(top1, their_top),
            "ms_per_frame": found["ms_per_frame"],
        }
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "check.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    if not all(
        record[k]["tracks"]["identical"] for k in ("09_filter45", "04_no_filter")
    ):
        raise SystemExit("fixes off do not give the reference tracks")


# ------------------------------------------------------------------ tune (off-benchmark frames)
def offbench_truth(clips: list[str]) -> pd.DataFrame:
    """The off-benchmark frames of ``clips`` with exactly one truth ball with a pitch position, on the pitch: its
    box, the centre rule's radius and its pitch position in corner metres."""
    rows = []
    for clip in clips:
        balls, _ = be.clip_labels(bench.GSR / "valid" / clip / "Labels-GameState.json")
        balls = balls[np.isfinite(balls.pitch_x)]
        single = balls.groupby("frame").filter(lambda g: len(g) == 1)
        on = (single.pitch_x.abs() <= bench.HALF_LENGTH) & (
            single.pitch_y.abs() <= bench.HALF_WIDTH
        )
        for ball in single[on].itertuples():
            if (ball.frame - 1) % bench.STRIDE == 0:
                continue
            rows.append(
                {
                    "clip": clip,
                    "frame": ball.frame,
                    "truth_x": ball.pitch_x + bench.HALF_LENGTH,
                    "truth_y": ball.pitch_y + bench.HALF_WIDTH,
                }
                | dict(zip(("tx1", "ty1", "tx2", "ty2"), ball.box, strict=True))
            )
    table = pd.DataFrame(rows)
    table["radius"] = be.radius(table[["tx1", "ty1", "tx2", "ty2"]].to_numpy(float))
    return table


def offbench_score(tracks: pd.DataFrame, truth: pd.DataFrame) -> dict:
    """The tuning score: well placed (<= 5 m on the carried homography) of the off-benchmark truth frames, with the
    > 5 m share and ticket 04's image hit rate (centre within r of the truth's) beside it."""
    joined = truth.merge(
        tracks[["clip", "frame", "status", *bt.COLUMNS, "pitch_x", "pitch_y"]],
        on=["clip", "frame"],
        how="left",
        validate="1:1",
    )
    error = np.hypot(joined.pitch_x - joined.truth_x, joined.pitch_y - joined.truth_y)
    centre = be.centres(joined[bt.COLUMNS].to_numpy(float))
    true = be.centres(joined[["tx1", "ty1", "tx2", "ty2"]].to_numpy(float))
    hit = np.linalg.norm(centre - true, axis=1) <= joined.radius.to_numpy()
    interpolated = (joined.status == "interpolated").to_numpy()
    return {
        "offbench_frames": len(joined),
        "offbench_well": float((error <= FAR).mean()),
        "offbench_far": float((error > FAR).mean()),
        "image_hit": float(hit.mean()),
        "interpolated_share": float(interpolated.mean()),
        "interpolated_well": float((error[interpolated] <= FAR).mean())
        if interpolated.any()
        else np.nan,
        "none_share": float((joined.status == "none").mean()),
    }


def tune(args: argparse.Namespace) -> None:
    """The registered grid on games 2+3, scored on the off-benchmark frames; the best setting per row."""
    clips = clips_of_games()
    entries = [(row, fixes) for row in ROWS for fixes in grid(row)]
    unique = list(dict.fromkeys(fixes for _, fixes in entries))
    _, found = run(clips, unique, args.workers, setting_04())
    by_fixes = {f["fixes"]: f for f in found}
    truth = offbench_truth(clips)
    lines = []
    for row, fixes in entries:
        f = by_fixes[fixes]
        lines.append(
            {"row": row, "setting": fixes_name(fixes)}
            | dataclasses.asdict(fixes)
            | offbench_score(f["tracks"], truth)
            | {"ms_per_frame": f["ms_per_frame"]}
            | f["stats"]
        )
    table = pd.DataFrame(lines)
    selected = {}
    for row in ROWS:
        part = table[table.row == row]
        best = part.loc[part.offbench_well.idxmax()]  # the first on a tie: grid order
        selected[row] = {
            "setting": best.setting,
            "fixes": dataclasses.asdict(grid(row)[list(part.index).index(best.name)]),
            "offbench_well": float(best.offbench_well),
        }
    out = args.out / "tune"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "grid.csv", index=False)
    (out / "grid.md").write_text(be.markdown(table) + "\n")
    (out / "selected.json").write_text(
        json.dumps(
            {
                "created": datetime.now(UTC).isoformat(timespec="seconds"),
                "score": "well placed (<= 5 m, carried homography) of the off-benchmark frames of games 2+3",
                "offbench_frames": len(truth),
                "rows": selected,
            },
            indent=2,
        )
        + "\n"
    )
    print(be.markdown(table))
    print(json.dumps(selected, indent=2))


# ------------------------------------------------------------------ read (placement frames, once per row)
def placement(table: pd.DataFrame, homographies: dict) -> pd.DataFrame:
    rows = bench.placement_rows(bt.table_top_ball(table), homographies, "valid")
    return rows[rows.game.isin(GAMES)].reset_index(drop=True)


def transitions(
    rows: pd.DataFrame,
    ref_rows: pd.DataFrame,
    tracks: pd.DataFrame,
    reference: pd.DataFrame,
) -> list[dict]:
    """Per (reference status, variant status) on the placement frames: frames, well placed of each and the change
    in pp of all frames."""
    keys = ["clip", "frame"]
    status = rows[keys].merge(tracks[[*keys, "status"]], how="left")["status"]
    ref_status = rows[keys].merge(reference[[*keys, "status"]], how="left")["status"]
    well, ref_well = rows.error_m <= FAR, ref_rows.error_m <= FAR
    out = []
    for (r, v), part in pd.DataFrame({"r": ref_status, "v": status}).groupby(
        ["r", "v"]
    ):
        index = part.index
        out.append(
            {
                "reference": r,
                "variant": v,
                "frames": len(index),
                "reference_well": int(ref_well[index].sum()),
                "variant_well": int(well[index].sum()),
                "variant_far": int((rows.error_m[index] > FAR).sum()),
                "reference_far": int((ref_rows.error_m[index] > FAR).sum()),
                "gain_pp_of_all": 100
                * float(well[index].sum() - ref_well[index].sum())
                / len(rows),
            }
        )
    return out


def read(args: argparse.Namespace) -> None:
    """Each row's selected setting, tracked and read once on the placement frames of games 2+3 vs ticket 09's tracks."""
    clips = clips_of_games()
    keep = set(clips)
    homographies = {
        key: m
        for key, m in bench.frozen_homographies("valid").items()
        if key[0] in keep
    }
    selected = json.loads((args.out / "tune/selected.json").read_text())["rows"]
    all_fixes = [Fixes(**selected[row]["fixes"]) for row in ROWS]
    setting = setting_04()
    top1, found = run(clips, all_fixes, args.workers, setting)
    reference = of_clips(pd.read_parquet(REFERENCE / "tracks.parquet"), clips)
    draws = bs.Draws(clips)
    ref_rows, top_rows = (
        placement(reference, homographies),
        placement(top1, homographies),
    )
    ref_numbers, top_numbers = bp.numbers(ref_rows, draws), bp.numbers(top_rows, draws)
    reference_selection = json.loads(REFERENCE_READ.read_text())["groups"]["selection"]
    out = args.out
    (out / "tracks").mkdir(parents=True, exist_ok=True)
    record = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "reference": str(REFERENCE),
        "setting_04": setting,
        "games": list(GAMES),
        "clips": len(clips),
        "homographies": bench.homography_file("valid"),
        "bootstrap": bs.BOOTSTRAP,
        "barrier_pp": bs.BARRIER_PP,
        "reference_numbers": {
            "track": bs.summary(ref_numbers),
            "top1": bs.summary(top_numbers),
            "status": bt.status_rows(ref_rows, top_rows, reference),
        },
        "rows": {},
    }
    keys = ["clip", "frame"]
    for row, f in zip(ROWS, found, strict=True):
        tracks = f["tracks"]
        tracks.drop(columns=["pitch_x", "pitch_y"]).to_parquet(
            out / "tracks" / f"{row}.parquet", index=False
        )
        rows = placement(tracks, homographies)
        if not rows[keys].equals(ref_rows[keys]):
            raise SystemExit(f"{row}: other placement frames than the reference's")
        numbers = bp.numbers(rows, draws)
        gain = bs.paired(numbers, ref_numbers)
        barrier = bs.barrier(rows, ref_rows, draws)
        entry = {
            "setting": fixes_name(f["fixes"]),
            "fixes": dataclasses.asdict(f["fixes"]),
            "track": bs.summary(numbers),
            "vs_reference": gain,
            "vs_top1": bs.paired(numbers, top_numbers),
            "barrier": barrier,
            "wins": bool(gain["well_placed"]["ci"][0] > 0 and barrier["met"]),
            "status": bt.status_rows(rows, top_rows, tracks),
            "transitions": transitions(rows, ref_rows, tracks, reference),
            "same_as_reference": same_tracks(tracks, reference),
            "cpu_ms_per_frame": f["ms_per_frame"],
            "stats": f["stats"],
        }
        if row == "off":
            entry["check_vs_09_read"] = bp.centre_check(
                {
                    "rows": rows,
                    "top1_rows": top_rows,
                    "track": numbers,
                    "top1": top_numbers,
                },
                tracks,
                reference_selection,
            )
        rows.to_parquet(out / "tracks" / f"placement_{row}.parquet", index=False)
        record["rows"][row] = entry
    record["chosen"] = choose(record["rows"])
    chosen = record["rows"][record["chosen"]]
    (out / "chosen.json").write_text(
        json.dumps(
            {
                "ticket": "ball-detection 12",
                "created": record["created"],
                "row": record["chosen"],
                "won": chosen["wins"],
                "setting": setting,
                "fixes": chosen["fixes"],
                "post_filter_long_side_px": bt.POST_FILTER,
                "gain_vs_phase1_pp": 100
                * chosen["vs_reference"]["well_placed"]["difference"],
                "gain_ci_pp": [
                    100 * v for v in chosen["vs_reference"]["well_placed"]["ci"]
                ],
                "note": "track_ball2 (scripts/ball_track2.py) with these fixes and ball/04/tune/chosen.json; "
                "every fix off = the phase 1 tracker (no row won)"
                if not chosen["wins"]
                else "track_ball2 (scripts/ball_track2.py) with these fixes and ball/04/tune/chosen.json",
            },
            indent=2,
        )
        + "\n"
    )
    (out / "read.json").write_text(json.dumps(record, indent=2, default=float) + "\n")
    text = markdown(record)
    (out / "read.md").write_text(text)
    print(text)


def choose(rows: dict) -> str:
    """The winning row with the largest point gain (ties: fewer fixes), ``off`` when none wins."""
    winners = [r for r in ROWS if r != "off" and rows[r]["wins"]]
    if not winners:
        return "off"
    return max(
        winners,
        key=lambda r: (
            rows[r]["vs_reference"]["well_placed"]["difference"],
            -FIX_COUNT[r],
        ),
    )


def pct(entry: dict) -> str:
    low, high = (100 * v for v in entry["ci"])
    return f"{100 * entry['value']:.2f}% [{low:.1f}; {high:.1f}]"


def pp(entry: dict) -> str:
    low, high = (100 * v for v in entry["ci"])
    return f"{100 * entry['difference']:+.2f} [{low:+.2f}; {high:+.2f}]"


def markdown(record: dict) -> str:
    ref = record["reference_numbers"]
    lines = [
        "# ball-detection 12: tracking fixes vs the phase 1 candidate, games 2+3 of GSR valid",
        "",
        f"Reference: ticket 09's tracks, well placed {pct(ref['track']['well_placed'])}; top-1 dd2c "
        f"{pct(ref['top1']['well_placed'])}. Chosen: **{record['chosen']}**.",
        "",
        "| row | setting | well placed | gain vs phase 1, pp [95%] | > 5 m of all: pp [95%] | barrier | wins | "
        "vs top-1, pp | > 5 m of placed | frames changed | CPU ms/frame |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row, e in record["rows"].items():
        b = e["barrier"]
        low, high = (100 * v for v in b["ci"])
        lines.append(
            f"| {row} | {e['setting']} | {pct(e['track']['well_placed'])} | "
            f"**{pp(e['vs_reference']['well_placed'])}** | {100 * b['difference']:+.2f} [{low:+.2f}; {high:+.2f}] | "
            f"{'ok' if b['met'] else 'NO'} | {'yes' if e['wins'] else 'no'} | {pp(e['vs_top1']['well_placed'])} | "
            f"{100 * e['track']['far_of_placed']['value']:.2f}% | {e['same_as_reference']['differ']} | "
            f"{e['cpu_ms_per_frame']:.3f} |"
        )
    lines += [
        "",
        "| row | status | frames (share) | well | > 5 m | top-1 well | gain vs top-1, pp of all |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in ref["status"]:
        lines.append(
            f"| reference | {s['status']} | {s['frames']} ({100 * s['share']:.1f}%) | {100 * s['track_well']:.1f}% | "
            f"{100 * s['track_far']:.1f}% | {100 * s['top1_well']:.1f}% | {s['gain_pp_of_all']:+.2f} |"
        )
    for row, e in record["rows"].items():
        for s in e["status"]:
            lines.append(
                f"| {row} | {s['status']} | {s['frames']} ({100 * s['share']:.1f}%) | {100 * s['track_well']:.1f}% | "
                f"{100 * s['track_far']:.1f}% | {100 * s['top1_well']:.1f}% | {s['gain_pp_of_all']:+.2f} |"
            )
    lines += [
        "",
        "Placement frames whose status changed (reference -> variant): frames, well placed and > 5 m of each.",
        "",
        "| row | reference -> variant | frames | well: ref / variant | > 5 m: ref / variant | pp of all |",
        "|---|---|---|---|---|---|",
    ]
    for row, e in record["rows"].items():
        for t in e["transitions"]:
            if (
                t["reference"] == t["variant"]
                and t["reference_well"] == t["variant_well"]
            ):
                continue
            lines.append(
                f"| {row} | {t['reference']} -> {t['variant']} | {t['frames']} | {t['reference_well']} / "
                f"{t['variant_well']} | {t['reference_far']} / {t['variant_far']} | {t['gain_pp_of_all']:+.2f} |"
            )
    lines += [
        "",
        "| row | " + " | ".join(record["rows"]["all"]["stats"]) + " |",
        "|---|" + "---|" * len(record["rows"]["all"]["stats"]),
    ]
    for row, e in record["rows"].items():
        lines.append(
            f"| {row} | " + " | ".join(str(v) for v in e["stats"].values()) + " |"
        )
    off = record["rows"]["off"]
    lines += [
        "",
        f"Control: every fix off = ticket 09's tracks ({'identical' if off['same_as_reference']['identical'] else 'DIFFER'}); "
        f"its read = ticket 09's selection card "
        f"({'identical' if off['check_vs_09_read']['identical'] else 'DIFFERS'}, "
        f"{off['check_vs_09_read']['numbers']} numbers). Bootstrap {bs.BOOTSTRAP['draws']} draws over clips, seed "
        f"{bs.BOOTSTRAP['seed']}; barrier: > 5 m of all placement frames up by at most {bs.BARRIER_PP:g} pp.",
        "",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------ commands
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_ in (
        ("check", "every fix off -> tickets 09's and 04's tracks"),
        ("tune", "the registered grid on the off-benchmark frames of games 2+3"),
        (
            "read",
            "each row's selected setting vs ticket 09, placement frames of games 2+3",
        ),
    ):
        c = commands.add_parser(name, help=help_)
        c.add_argument("--out", type=Path, default=OUT)
        c.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    {"check": check, "tune": tune, "read": read}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
