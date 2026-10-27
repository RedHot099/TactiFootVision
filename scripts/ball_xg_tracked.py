"""ball-detection 20: arm T of ticket 19's xG comparison, the t0 ball tracked through the shot clip before the shot,
plus (D4, descriptive) phase 1's and ticket 13's single-frame balls on the shot frames of the 172 held-out matches.

    XG=/home/kuba/projects/tactifoot_vision/.claude/worktrees/research-xg-detector/experiments/claude_fable
    $XG/run.sh --no-sync python $PWD/scripts/ball_xg_tracked.py dump --shard 0 --shards 2   # dd2c + keypoints (GPU)
    $XG/run.sh --no-sync python $PWD/scripts/ball_xg_tracked.py track                       # motion, H, track_ball (CPU)
    $XG/run.sh --no-sync python $PWD/scripts/ball_xg_tracked.py read                        # arm T in the family (CPU)
    $XG/run.sh --no-sync python $PWD/scripts/ball_xg_tracked.py detect172                   # P1 / S2 on the 172 (GPU)
    $XG/run.sh --no-sync python $PWD/scripts/ball_xg_tracked.py read172                     # vs StatsBomb (CPU)

Everything runs in the xG experiment's environment (``run.sh``), as tickets 16 and 19 do: the homographies must be the
xG cache's (pitch keypoint model ``sn960`` with ``extract3.COMMON``'s settings) and the read replays the cache with the
frozen xG code.

Arm T (causal: only frames up to t0). The population is ticket 19's (7,062 non-penalty shots, 328 development matches).
Per shot clip (8 s, 720p; t0 = clip frame 100, ``ball_xg_compare.t0_index``) the window is the 2 s before t0 (frames
k = -50..0 relative to t0), cut at the last scene cut before t0 found by ticket 19 (``ball/19/clips``: the window starts
at the first frame after the cut). Clips not at 25 fps (ticket 19: 115) and clips whose frames 100 / 90 do not match
the shot's ``+0.jpg`` / ``-400.jpg`` (``aligned``) get no T: the shot keeps F's frames. On the window:

    dump    dd2c 1280 (phase 1's row: imgsz 1280, NMS IoU 0.7, every candidate >= 0.01; ``ball_temporal.dump_predictor``)
            on every window frame, the frames k = 0 / -10 / -20 replaced by the shot's ``+0`` / ``-400`` / ``-800``
            jpgs (the images the xG cache analysed, so the t0 ball is found on xG's own frame); the pitch keypoint model
            on every 5th frame (k multiple of 5) that has no cached frame row
    track   the camera motion between consecutive window frames (``ball_temporal.frame_motion`` at 960 x 540, 04's
            working size), anchor homographies = the xG cache's at k = 0 / -10 / -20 (when the cache has the row) and
            ``HomographyEstimator.fit`` (``extract3.COMMON``) of the dumped keypoints elsewhere, carried to every frame
            as in 04 (``ball_temporal.propagate``, reach 10 frames); the 45 px post-filter (at 1080p) and the package's
            ``track_ball`` with ticket 04's setting (``ball/04/tune/chosen.json``)
    read    T's frames = F's cached frames with the ball of the frames at 0 / -400 / -800 ms swapped for the tracked
            ball of k = 0 / -10 / -20 (measured or interpolated; placed by the cached homography of that frame, as
            ``ball_xg.swap_ball``); a frame without a tracked ball keeps its cached ball (F's). Then F's construction
            (``FEET_T0`` replay, ``stack_location.e3_table``) and the frozen stack procedure C6 refitted in the folds.

Family {C, S2, T} vs F (ticket 19's registration): ΔLL of the stack arm ALL, paired match bootstrap (2,000, seed 0),
Holm over the three arms; C's and S2's probabilities are ticket 19's (``ball/19/xg/probabilities.csv``), F is refitted
and must reproduce them exactly.

Registration: ticket 20's comment of 2026-10-27 (before any T log-loss read).
"""

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ball_errors as be  # noqa: E402
import ball_xg_compare as bc  # noqa: E402

logger = logging.getLogger("ball_xg_tracked")

XG = bc.XG
OUT = be.ROOT / "ball/20"
CLIPS19 = be.ROOT / "ball/19/clips/clips.parquet"
READ19 = be.ROOT / "ball/19/xg"
SETTING = be.ROOT / "ball/04/tune/chosen.json"
F_CACHE = XG / "cache/xgd_full4_feet_t0/E3"
FPS = 25.0
WINDOW = 50  # frames before t0 (2 s at 25 fps)
ANCHOR_EVERY = 5
CACHED = {
    0: 0,
    -10: -400,
    -20: -800,
}  # window frame k -> the xG cache's offset_ms (the jpg frames)
JPG = {0: "frame_path", -10: "frame_path_-400", -20: "frame_path_-800"}
MOTION_SIZE = (960, 540)  # 04's working size (1080p at scale 0.5)
FRAME_SIZE = (1280, 720)
PEOPLE_CONF = 0.3  # the pipeline's people threshold (``detector_bench.OPERATING``)
THUMB = (256, 144)
CUT_SCORE = 27.0  # ticket 19's cut threshold (PySceneDetect ContentDetector's default)
ALIGN_TOLERANCE, ALIGN_MAX = (
    0.5,
    6.0,
)  # grey levels: shift 0 within 0.5 of the best shift, and at most 6
CONTROL_EVERY = 50  # every 50th shot of the population also refits the cached t0 homography (a control)
ALIGN_SHIFTS = {
    0: (-2, -1, 0),
    -10: (-2, -1, 0, 1, 2),
}  # shifts tried around k = 0 (never past t0) and k = -10
STATUSES = ("ok", "irregular_fps", "misaligned", "unreadable")
INDEX_COLUMNS = [
    "shot_id",
    "game",
    "is_penalty",
    "clip_path",
    "position_ms",
    *JPG.values(),
]  # no label


# ------------------------------------------------------------------ pure parts
def window_start(last_cut_frames_before_t0: int) -> int:
    """First window frame k (<= 0): the frame after the last cut before t0 (ticket 19's ``last_cut_frames_before_t0``,
    -1 without a cut in the 2 s), else -WINDOW."""
    last = int(last_cut_frames_before_t0)
    return -last if last >= 0 else -WINDOW


def anchor_frames(start: int, cached: set[int]) -> list[int]:
    """Window frames k in [start, 0] that get a keypoint fit: multiples of ``ANCHOR_EVERY`` without a cached row."""
    return [k for k in range(start, 1) if k % ANCHOR_EVERY == 0 and k not in cached]


def is_aligned(errors: dict[int, float]) -> bool:
    """Shift 0 is (within ``ALIGN_TOLERANCE``) the best match of the jpg among the shifted clip frames, and close."""
    if 0 not in errors or not np.isfinite(errors[0]):
        return False
    best = min(v for v in errors.values() if np.isfinite(v))
    return errors[0] <= best + ALIGN_TOLERANCE and errors[0] <= ALIGN_MAX


def tracked_ball(track: pd.DataFrame, k: int) -> pd.Series | None:
    """The tracked ball of window frame ``k`` (``x1..y2``, ``conf``; conf NaN when interpolated), None without one."""
    rows = track[(track.k == k) & (track.status != "none")]
    if not len(rows):
        return None
    row = rows.iloc[0]
    return pd.Series({c: float(row[c]) for c in ("x1", "y1", "x2", "y2", "conf")})


def t_frames(
    frames: pd.DataFrame, tracks: pd.DataFrame, project
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(T's frames table, per cached row the ball's source: "tracked_measured" / "tracked_interpolated" / "cached").
    ``frames``: the xG cache rows (shot_id, offset_ms, ...); ``tracks``: per shot with T and window frame k
    (shot_id, k, status, x1..y2, conf). A row gets the tracked ball of its frame k when there is one, else stays."""
    import ball_xg as bx

    by_shot = {s: g for s, g in tracks.groupby("shot_id")} if len(tracks) else {}
    offset_k = {v: k for k, v in CACHED.items()}
    rows, sources = [], []
    for row in frames.to_dict("records"):
        source = "cached"
        track = by_shot.get(row["shot_id"])
        k = offset_k.get(int(row["offset_ms"]))
        ball = tracked_ball(track, k) if track is not None and k is not None else None
        if ball is not None and row["readable"]:
            status = track.loc[track.k == k, "status"].iloc[0]
            row = bx.swap_ball(row, ball, project)
            source = f"tracked_{status}"
        rows.append(row)
        sources.append(
            {
                "shot_id": row["shot_id"],
                "offset_ms": int(row["offset_ms"]),
                "source": source,
            }
        )
    return pd.DataFrame(rows, columns=frames.columns), pd.DataFrame(sources)


def homography_gap(a: np.ndarray, b: np.ndarray) -> float:
    """Median pitch distance (m) between two frame -> pitch homographies over a grid of a 720p frame's lower two
    thirds (``ball_temporal.anchor_agreement``'s grid scaled to 720p)."""
    from tactifoot_vision.pitch.homography import frame_to_pitch

    xs, ys = np.meshgrid(np.linspace(133, 1147, 8), np.linspace(267, 693, 5))
    grid = np.c_[xs.ravel(), ys.ravel()]
    return float(
        np.median(
            np.linalg.norm(frame_to_pitch(grid, a) - frame_to_pitch(grid, b), axis=1)
        )
    )


def top1_matches(candidates: pd.DataFrame, cached_box) -> float:
    """Whether the dump's top-1 ball >= 0.15 at t0 is the cached t0 ball (centres within 2 px; both absent = 1;
    NaN when unknown): the dump's row (NMS IoU 0.7) against the xG analyser's (0.5) on the same jpg."""
    balls = candidates[
        (candidates.k == 0) & (candidates.class_id == 0) & (candidates.conf >= 0.15)
    ]
    box = np.asarray(cached_box if cached_box is not None else [], float).ravel()
    if not len(balls) or box.size < 4:
        return float(not len(balls) and box.size < 4)
    top = balls.loc[balls.conf.idxmax(), ["x1", "y1", "x2", "y2"]].to_numpy(float)
    gap = np.hypot(
        (top[0] + top[2] - box[0] - box[2]) / 2, (top[1] + top[3] - box[1] - box[3]) / 2
    )
    return float(gap <= 2.0)


def window_result(
    candidates: pd.DataFrame, ks: list[int], homographies: list, post_filter: float
):
    """The pipeline's result of one window as the dump sees it (``ball_temporal.clip_result`` for a window of 720p
    frames ``ks``): per frame the ball candidates after the post-filter (most confident first), the people >=
    ``PEOPLE_CONF``, the homography, positions projected (``reproject``). Frame i is window frame ``ks[i]`` at i / FPS."""
    import copy

    import ball_temporal as bt

    from tactifoot_vision.pipeline import FrameResult, PipelineResult, reproject

    table = bt.drop_long_balls(candidates, post_filter, FRAME_SIZE[1])
    by_k = {int(k): g for k, g in table.groupby("k")}
    empty = table.iloc[:0]
    frames = []
    for i, k in enumerate(ks):
        rows = by_k.get(k, empty)
        balls = bt.detections(rows[rows.class_id == 0])
        people = bt.detections(rows[(rows.class_id > 0) & (rows.conf >= PEOPLE_CONF)])
        frames.append(
            FrameResult(
                i, i / FPS, people, copy.deepcopy(balls[:1]), ball_candidates=balls
            )
        )
    result = PipelineResult(frames, FPS, FRAME_SIZE, list(bt.CLASSES))
    return reproject(result, homographies)


def track_table(result, ks: list[int]) -> pd.DataFrame:
    """``ball_temporal.chosen_balls`` of a tracked window with the window frame k instead of the GSR frame."""
    import ball_temporal as bt

    table = bt.chosen_balls(result).drop(columns="frame")
    table.insert(0, "k", ks)
    return table


# ------------------------------------------------------------------ inputs
def population(limit: int | None = None, role: str = "training") -> pd.DataFrame:
    """The non-penalty shots of the plan's ``role`` matches with the label-free columns, index shot_id, sorted."""
    import ball_xg as bx
    from xgpoc.paths import INDEX

    index = pd.read_parquet(INDEX, columns=INDEX_COLUMNS)
    plan = pd.read_csv(bx.PLAN, usecols=["game", "role"])
    shots = index[
        index.game.map(plan.set_index("game").role).eq(role)
        & ~index.is_penalty.astype(bool)
    ]
    shots = shots.drop(columns="is_penalty").set_index("shot_id").sort_index()
    return shots.iloc[:limit] if limit else shots


def plan_shots(limit: int | None = None) -> pd.DataFrame:
    """The population with ticket 19's clip facts (fps, t0 frame, last cut) and the window start."""
    shots = population(limit)
    clips = pd.read_parquet(CLIPS19)
    shots = shots.join(
        clips[["fps", "n_frames", "t0_index", "last_cut_frames_before_t0"]], how="left"
    )
    if shots.fps.isna().any():
        raise SystemExit("ticket 19's clip table does not cover the population")
    shots["start"] = shots.last_cut_frames_before_t0.map(window_start)
    return shots


def cached_rows(ids: list[str]) -> pd.DataFrame:
    return pd.read_parquet(F_CACHE / "frames.parquet", filters=[("shot_id", "in", ids)])


def _grey_thumb(image: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(
        cv2.resize(image, THUMB, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY
    ).astype(np.int16)


def decode_window(
    shot: pd.Series, first: int, last: int = 0
) -> dict[int, np.ndarray] | None:
    """Clip frames k in [first, last] (BGR; past the clip's end absent), None when the clip ends before t0."""
    import cv2

    t0 = int(shot.t0_index)
    cap = cv2.VideoCapture(str(shot.clip_path))
    out = {}
    for i in range(t0 + last + 1):
        ok, frame = cap.read()
        if not ok:
            break
        if i - t0 >= first:
            out[i - t0] = frame
    cap.release()
    return out if 0 in out and min(out) <= first else None


def first_cut_after(frames: dict[int, np.ndarray]) -> int:
    """The last frame k >= 0 before the first scene cut after t0 (ticket 19's detector: PySceneDetect's content score
    on HSV 256 x 144 thumbnails >= ``CUT_SCORE``), the last frame there is without one."""
    import cv2

    ks = sorted(k for k in frames if k >= 0)
    ks = ks[
        : next((i for i, k in enumerate(ks) if k != i), len(ks))
    ]  # consecutive from t0
    hsv = np.stack(
        [
            cv2.cvtColor(
                cv2.resize(frames[k], THUMB, interpolation=cv2.INTER_AREA),
                cv2.COLOR_BGR2HSV,
            )
            for k in ks
        ]
    )
    cuts = np.flatnonzero(bc.content_scores(hsv) >= CUT_SCORE) if len(ks) > 1 else []
    return int(cuts[0]) if len(cuts) else ks[-1]


def load_shot(shot_id: str, shot: pd.Series, after: int = 0) -> dict:
    """The window of one shot: status, alignment errors, the frames (jpgs swapped in at k 0 / -10 / -20). ``after``
    (the two-sided variant): frames after t0 up to ``after`` too, the window ends before the first cut after t0."""
    import cv2

    out = {
        "shot_id": shot_id,
        "start": int(shot.start),
        "end": 0,
        "status": "ok",
        "frames": None,
    }
    if abs(float(shot.fps) - FPS) > 0.01:
        out["status"] = "irregular_fps"
        return out
    first = min(int(shot.start), -12)
    frames = decode_window(shot, first, after)
    if frames is None:
        out["status"] = "unreadable"
        return out
    aligned = True
    for centre, shifts in ALIGN_SHIFTS.items():
        jpg = cv2.imread(str(shot[JPG[centre]]))
        if jpg is None:
            aligned = False
            continue
        ref = _grey_thumb(jpg)
        errors = {
            s: float(np.abs(_grey_thumb(frames[centre + s]) - ref).mean())
            for s in shifts
        }
        out[f"align_{centre}_err0"] = errors[0]
        aligned &= is_aligned(errors)
    if not aligned:
        out["status"] = "misaligned"
        return out
    if after:
        out["end"] = first_cut_after(frames)
    window = {k: frames[k] for k in range(int(shot.start), out["end"] + 1)}
    for k, column in JPG.items():
        if k in window:
            image = (
                cv2.imread(str(shot[column])) if isinstance(shot[column], str) else None
            )
            if image is not None and image.shape == window[k].shape:
                window[k] = image
            out[f"jpg_{k}"] = image is not None
    out["frames"] = window
    return out


# ------------------------------------------------------------------ dump (GPU)
def prefetch(pool, function, jobs: list[tuple], depth: int):
    """``function(*job)`` of every job, in order, at most ``depth`` jobs ahead (``Executor.map`` submits them all,
    which holds every decoded window in memory)."""
    from collections import deque

    pending = deque()
    for job in jobs:
        pending.append(pool.submit(function, *job))
        if len(pending) > depth:
            yield pending.popleft().result()
    while pending:
        yield pending.popleft().result()


def dump(args: argparse.Namespace) -> None:
    import ball_temporal as bt
    import detector_bench as bench
    import torch
    import torchvision  # noqa: F401  (loaded, ultralytics' NMS is torchvision's: same boxes, 22 ms less per call)
    from xgpoc.detector import extract3 as x3

    import tactifoot_vision as tv

    shots = plan_shots(args.limit)
    ids = list(shots.index)
    rows = cached_rows(ids)
    have = rows.groupby("shot_id").offset_ms.apply(set).to_dict()
    weights = x3.DETECTORS["dd2c_1280"].weights
    if bench.sha256_file(weights) != bench.sha256_file(bt.WEIGHTS):
        raise SystemExit("the xG dd2c weights are not phase 1's")
    predict, names, settings = bt.dump_predictor(bt.ROW, bt.WEIGHTS, args.device)
    if list(names) != list(bt.CLASSES):
        raise SystemExit(f"dd2c's classes {names} are not {bt.CLASSES}")
    c = x3.COMMON
    keypoints = tv.load_model(
        "yolo_pose",
        x3.KEYPOINT_WEIGHTS,
        conf=c["keypoint_conf"],
        imgsz=c["keypoint_imgsz"],
        resize=c["keypoint_resize"],
        device=args.device,
    )
    parts = args.out / ("dump_after" if args.after else "dump")
    parts.mkdir(parents=True, exist_ok=True)
    after = WINDOW if args.after else 0
    chunks = [ids[i : i + args.chunk] for i in range(0, len(ids), args.chunk)]
    det_s, kp_s, n_det, n_kp = 0.0, 0.0, 0, 0
    clock_all = time.perf_counter()
    with ThreadPoolExecutor(args.readers) as pool:
        for n, chunk in enumerate(chunks):
            if (
                n % args.shards != args.shard
                or (parts / f"meta_{n:04d}.parquet").is_file()
            ):
                continue
            cand_tables, kp_rows, meta = [], [], []
            cached_of = {
                s: {k for k, o in CACHED.items() if o in have.get(s, set())}
                for s in chunk
            }
            loads = [(s, shots.loc[s], after) for s in chunk]
            for item in prefetch(pool, load_shot, loads, args.readers + 1):
                frames = item.pop("frames")
                meta.append(item)
                if frames is None:
                    continue
                sid = item["shot_id"]
                for k, image in frames.items():
                    if after and k <= 0:  # the causal dump has them
                        continue
                    torch.cuda.synchronize()
                    clock = time.perf_counter()
                    found = predict(image)
                    torch.cuda.synchronize()
                    det_s += time.perf_counter() - clock
                    n_det += 1
                    table = bt.dump_table(found, k, image.shape[0], None)
                    table = table[
                        (table.class_id == 0) | (table.conf >= PEOPLE_CONF)
                    ].rename(columns={"frame": "k"})
                    table.insert(0, "shot_id", sid)
                    cand_tables.append(table)
                check = (
                    [0]
                    if ids.index(sid) % CONTROL_EVERY == 0
                    and 0 in cached_of[sid]
                    and not after
                    else []
                )
                anchors = (
                    list(range(ANCHOR_EVERY, item["end"] + 1, ANCHOR_EVERY))
                    if after
                    else anchor_frames(item["start"], cached_of[sid])
                )
                for k in anchors + check:
                    clock = time.perf_counter()
                    kp = keypoints.predict(frames[k])
                    torch.cuda.synchronize()
                    kp_s += time.perf_counter() - clock
                    n_kp += 1
                    row = {
                        "shot_id": sid,
                        "k": k,
                        "xy": None,
                        "conf": None,
                        "control": k in check,
                    }
                    if len(kp):
                        row["xy"] = np.asarray(kp.xy[0], float).ravel().tolist()
                        row["conf"] = (
                            np.asarray(kp.confidence[0], float).tolist()
                            if kp.confidence is not None
                            else None
                        )
                    kp_rows.append(row)
            if cand_tables:
                pd.concat(cand_tables, ignore_index=True).to_parquet(
                    parts / f"cand_{n:04d}.parquet", index=False
                )
            pd.DataFrame(
                kp_rows, columns=["shot_id", "k", "xy", "conf", "control"]
            ).to_parquet(parts / f"kp_{n:04d}.parquet", index=False)
            pd.DataFrame(meta).to_parquet(parts / f"meta_{n:04d}.parquet", index=False)
            logger.info(
                "shard %d: chunk %d / %d, %d frames %.1f ms, %d keypoint frames %.1f ms, %.0f s",
                args.shard,
                n + 1,
                len(chunks),
                n_det,
                1000 * det_s / max(n_det, 1),
                n_kp,
                1000 * kp_s / max(n_kp, 1),
                time.perf_counter() - clock_all,
            )
    record = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "shard": args.shard,
        "shards": args.shards,
        "detector": {
            "weights": str(bt.WEIGHTS),
            "sha256": bench.sha256_file(bt.WEIGHTS),
        }
        | settings
        | {"conf": bt.DUMP_CONF},
        "keypoints": {
            "weights": str(x3.KEYPOINT_WEIGHTS),
            "sha256": bench.sha256_file(x3.KEYPOINT_WEIGHTS),
        },
        "frames": n_det,
        "keypoint_frames": n_kp,
        "ms_detector": 1000 * det_s / max(n_det, 1),
        "ms_keypoints": 1000 * kp_s / max(n_kp, 1),
        "seconds": time.perf_counter() - clock_all,
    }
    (parts / f"record_shard{args.shard}.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print(json.dumps(record, indent=2))


def read_dump(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def stack(pattern):
        files = sorted(folder.glob(pattern))
        return (
            pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
            if files
            else pd.DataFrame()
        )

    return stack("cand_*.parquet"), stack("kp_*.parquet"), stack("meta_*.parquet")


# ------------------------------------------------------------------ track (CPU)
_STATE: dict = {}


def _init_track(out: Path, limit: int | None) -> None:
    import cv2
    from xgpoc.detector import extract3 as x3

    import tactifoot_vision as tv

    cv2.setNumThreads(1)
    logging.getLogger("tactifoot_vision").setLevel(logging.WARNING)
    c = x3.COMMON
    _STATE["estimator"] = tv.pitch.HomographyEstimator(
        min_confidence=c["homography_min_confidence"],
        ransac_threshold=c["homography_ransac_threshold"],
        refine_with_lines=c["refine_with_lines"],
    )
    _STATE["setting"] = json.loads(SETTING.read_text())
    _STATE["shots"] = plan_shots(limit)


def _track_worker(job: tuple) -> tuple[pd.DataFrame, dict]:
    import ball_temporal as bt
    import cv2
    import supervision as sv

    from tactifoot_vision.pipeline import track_ball

    sid, candidates, kps, cached_h, cached_ball, end = job
    shot = _STATE["shots"].loc[sid]
    clock = time.perf_counter()
    item = load_shot(sid, shot, WINDOW if end is not None else 0)
    frames = item["frames"]
    ks = list(range(int(shot.start), (end or 0) + 1))
    if frames is None or sorted(frames) != ks:
        raise RuntimeError(
            f"{sid}: the window does not decode as in the dump ({item['status']})"
        )
    people = {
        int(k): g[["x1", "y1", "x2", "y2"]].to_numpy(float)
        for k, g in candidates[
            (candidates.class_id > 0) & (candidates.conf >= PEOPLE_CONF)
        ].groupby("k")
    }
    scale = MOTION_SIZE[0] / FRAME_SIZE[0]
    motion = np.full((len(ks) - 1, 3, 3), np.nan)
    previous = None
    for i, k in enumerate(ks):
        grey = cv2.cvtColor(
            cv2.resize(frames[k], MOTION_SIZE, interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY,
        )
        if previous is not None:
            mask = bt.people_mask(
                previous.shape,
                people.get(ks[i - 1], np.zeros((0, 4))),
                scale,
                bt.MOTION["people_margin_px"],
            )
            matrix, _ = bt.frame_motion(previous, grey, mask)
            if matrix is not None:
                motion[i - 1] = bt.to_full(matrix, scale)
        previous = grey
    anchors, fitted = {}, 0
    for k, h in cached_h.items():
        if k in frames and h is not None:
            anchors[ks.index(k)] = np.asarray(h, float).reshape(3, 3)
    control_gap = np.nan
    for row in kps.itertuples():
        if row.xy is None or row.k not in frames:
            continue
        xy = np.asarray(row.xy, np.float32).reshape(1, -1, 2)
        conf = np.asarray(row.conf, np.float32)[None] if row.conf is not None else None
        matrix = _STATE["estimator"].fit(
            sv.KeyPoints(xy=xy, confidence=conf),
            frames[row.k],
            people.get(int(row.k), np.zeros((0, 4))),
        )
        if row.control:  # the cached t0 homography refitted from these keypoints
            cached = cached_h.get(0)
            if matrix is not None and cached is not None:
                control_gap = homography_gap(
                    matrix, np.asarray(cached, float).reshape(3, 3)
                )
            continue
        if matrix is not None:
            anchors[ks.index(int(row.k))] = matrix
            fitted += 1
    homographies = bt.propagate(motion, anchors, bt.REACH)
    result = window_result(candidates, ks, homographies, bt.POST_FILTER)
    table = track_table(track_ball(result, **_STATE["setting"]), ks)
    table.insert(0, "shot_id", sid)
    table["has_h"] = [h is not None for h in homographies]
    info = {
        "shot_id": sid,
        "window": len(ks),
        "anchors": len(anchors),
        "fitted": fitted,
        "control_h_gap_m": control_gap,
        "t0_top1_matches_cache": top1_matches(candidates, cached_ball),
        "motion_breaks": int(np.isnan(motion[:, 0, 0]).sum()),
        "seconds": time.perf_counter() - clock,
    }
    return table, info


def track(args: argparse.Namespace) -> None:
    from multiprocessing import Pool

    candidates, kps, meta = read_dump(args.out / "dump")
    ends, name = {}, ""
    if args.two_sided:  # the offline variant: the causal dump plus the frames after t0
        more_c, more_k, more_meta = read_dump(args.out / "dump_after")
        if not more_meta.set_index("shot_id").status.equals(
            meta.set_index("shot_id").status
        ):
            raise SystemExit("the two dumps disagree on the shots' status")
        candidates = pd.concat([candidates, more_c], ignore_index=True)
        kps = pd.concat([kps, more_k], ignore_index=True)
        kps = kps[~kps.control.astype(bool)]
        ends = more_meta.set_index("shot_id").end.to_dict()
        name = "_two_sided"
    ok = meta[meta.status == "ok"].shot_id.tolist()
    rows = cached_rows(ok)
    rows = rows[rows.offset_ms.isin(CACHED.values())]
    offset_k = {v: k for k, v in CACHED.items()}
    cached_h = {s: {} for s in ok}
    cached_ball = {}
    for r in rows.itertuples():
        h = r.homography if r.readable else None
        h = None if h is None or (isinstance(h, float) and np.isnan(h)) else h
        cached_h[r.shot_id][offset_k[int(r.offset_ms)]] = h
        if int(r.offset_ms) == 0 and r.readable:
            cached_ball[r.shot_id] = r.ball_xyxy
    by_shot_c = dict(tuple(candidates.groupby("shot_id")))
    by_shot_k = dict(tuple(kps.groupby("shot_id"))) if len(kps) else {}
    empty_c, empty_k = (
        candidates.iloc[:0],
        kps.iloc[:0]
        if len(kps)
        else pd.DataFrame(columns=["k", "xy", "conf", "control"]),
    )
    jobs = [
        (
            s,
            by_shot_c.get(s, empty_c),
            by_shot_k.get(s, empty_k),
            cached_h[s],
            cached_ball.get(s),
            int(ends[s]) if args.two_sided else None,
        )
        for s in ok
    ]
    with Pool(
        args.workers, initializer=_init_track, initargs=(args.out, args.limit)
    ) as pool:
        found = pool.map(_track_worker, jobs, chunksize=8)
    tracks = pd.concat([t for t, _ in found], ignore_index=True)
    info = pd.DataFrame([i for _, i in found])
    tracks.to_parquet(args.out / f"tracks{name}.parquet", index=False)
    info.to_parquet(args.out / f"track_info{name}.parquet", index=False)
    t0 = tracks[tracks.k == 0]
    summary = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "setting": json.loads(SETTING.read_text()),
        "shots": len(meta),
        "status": meta.status.value_counts().to_dict(),
        "tracked": len(ok),
        "t0_status": t0.status.value_counts().to_dict(),
        "t0_has_h": float(t0.has_h.mean()),
        "window_frames_median": float(info.window.median()),
        "anchors_median": float(info.anchors.median()),
        "motion_breaks_share_of_pairs": float(
            info.motion_breaks.sum() / max((info.window - 1).sum(), 1)
        ),
        "cpu_seconds": float(info.seconds.sum()),
        "control_h_gap_m": {
            "shots": int(info.control_h_gap_m.notna().sum()),
            "median": float(info.control_h_gap_m.median()),
            "p90": float(info.control_h_gap_m.quantile(0.9)),
            "max": float(info.control_h_gap_m.max()),
        },
        "t0_top1_matches_cache": float(info.t0_top1_matches_cache.mean()),
    }
    (args.out / f"track{name}.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )
    print(json.dumps(summary, indent=2, default=str))


# ------------------------------------------------------------------ read (CPU)
def read(args: argparse.Namespace) -> None:
    m = bc._xg_imports()
    loc, sa, sf, si, sl, ss = (m[k] for k in ("loc", "sa", "sf", "si", "sl", "ss"))
    from tactifoot_vision.pitch.homography import frame_to_pitch

    started = datetime.now(UTC)
    # --two-sided: the offline variant's tracks in place of T's (arms named T / T_flagsF), descriptive, no verdict
    name = "_two_sided" if args.two_sided else ""
    out = args.out / f"xg{name}"
    out.mkdir(parents=True, exist_ok=True)
    frozen = m["pr"].load(XG / m["pr"].PROTOCOL, m["sha"])
    probs19 = pd.read_csv(
        READ19 / "probabilities.csv", index_col=0, float_precision="round_trip"
    )  # exact floats (the default parser may differ in the last bit)
    read19 = json.loads((READ19 / "read.json").read_text())
    _, _, meta = read_dump(args.out / "dump")
    tracks = pd.read_parquet(args.out / f"tracks{name}.parquet")
    controls = {}

    with m["guard"]():
        reference = si.build_population()
        y = si.load_outcomes(reference).aligned(reference)
        weights = ss.bootstrap_weights(reference.games)
        pop = sl.build_population(loc.FEET_T0, e3=XG / bc.CACHES["F"])
        if (
            not pop.table.index.equals(reference.table.index)
            or (pop.folds != reference.folds).any()
        ):
            raise SystemExit("F: shots or folds differ from the reference population")
        spec = sf.spec(frozen["design"]["chosen"], pop.table)
        bc.design_check(m, spec, frozen)
        results = {"F": sa.analyse(pop.table, pop.folds, y, weights, spec)}
        expected = json.loads((XG / bc.REFERENCES["F"]).read_text())
        controls["F_summary_reproduced"] = (
            m["rerun"].summary_of(results["F"]) == expected
        )
        f_ids = list(pop.table.index)
        if list(probs19.index) != f_ids:
            raise SystemExit("ticket 19's probabilities are not on the same shots")
        controls["F_probabilities_equal_19"] = bool(
            np.array_equal(
                results["F"]["probabilities"]["ALL"], probs19["F:ALL"].to_numpy()
            )
        )
        if not (
            controls["F_summary_reproduced"] and controls["F_probabilities_equal_19"]
        ):
            raise SystemExit(f"F does not reproduce: {controls}")
        logger.info("control: F reproduced (summary and ticket 19's probabilities)")

        f_table = pop.table
        frames = cached_rows(f_ids)
        if set(meta.shot_id) != set(f_ids):
            raise SystemExit("the dump does not cover the population")
        swapped, sources = t_frames(frames, tracks, frame_to_pitch)
        features = loc.feature_rows(swapped, loc.FEET_T0)
        t_e3, _ = sl.e3_table(swapped, features, loc.FEET_T0)
        t_table = f_table.copy()
        t_table[si.E3_COLUMNS] = t_e3.loc[f_ids, si.E3_COLUMNS].to_numpy()
        tables = {"T": t_table, "T_flagsF": bc.keep_flags(t_table, f_table)}
        for arm, table in tables.items():
            spec = sf.spec(frozen["design"]["chosen"], table)
            controls[f"{arm}_dropped_beyond_protocol"] = bc.design_check(
                m, spec, frozen
            )
            results[arm] = sa.analyse(table, pop.folds, y, weights, spec)
            logger.info("arm %s fitted", arm)

    games = reference.games
    probs = {
        f"{a}:{s}": results[a]["probabilities"][s]
        for a in results
        for s in bc.STACK_ARMS
    }
    for arm in ("R", "C", "S2", "S2_rule"):
        for s in bc.STACK_ARMS:
            probs[f"{arm}:{s}"] = probs19[f"{arm}:{s}"].to_numpy()
    arms = ("R", "C", "S2", "S2_rule", "T", "T_flagsF")
    f_ll = {s: ss.shot_log_loss(y, probs[f"F:{s}"], ss.CLIP) for s in bc.STACK_ARMS}
    ll = {
        a: {s: ss.shot_log_loss(y, probs[f"{a}:{s}"], ss.CLIP) for s in bc.STACK_ARMS}
        for a in arms
    }
    paired, boots = {}, {}
    for a in arms:
        for s in bc.STACK_ARMS:
            d = ll[a][s] - f_ll[s]
            boot = ss.boot_means(weights, d)
            if s == bc.PRIMARY:
                boots[a] = boot
            paired[f"{a}:{s}"] = {
                "delta_ll": float(d.mean()),
                "ci95": bc.interval(boot, 0.95),
                "ci90": bc.interval(boot, 0.90),
                "p_improve": bc.p_improve(boot),
                "p_tost": bc.p_equivalent(boot),
            }
    for a in ("C", "S2"):
        mine, theirs = (
            paired[f"{a}:{bc.PRIMARY}"],
            read19["paired_vs_F"][f"{a}:{bc.PRIMARY}"],
        )
        controls[f"{a}_equals_19"] = all(
            np.allclose(mine[k], theirs[k], rtol=0, atol=1e-12)
            for k in ("delta_ll", "ci95", "p_improve", "p_tost")
        )
    if not (controls["C_equals_19"] and controls["S2_equals_19"]):
        raise SystemExit(f"C / S2 do not reproduce ticket 19: {controls}")
    p_imp = {a: paired[f"{a}:{bc.PRIMARY}"]["p_improve"] for a in bc.FAMILY}
    p_eq = {a: paired[f"{a}:{bc.PRIMARY}"]["p_tost"] for a in bc.FAMILY}
    points = {a: paired[f"{a}:{bc.PRIMARY}"]["delta_ll"] for a in bc.FAMILY}
    decision = bc.verdicts(p_imp, p_eq, points)
    holm_levels = {
        a: {
            f"ci_{k}": bc.interval(boots[a], 1 - bc.ALPHA_IMPROVE * 2 / k)
            for k in (1, 2, 3)
        }
        | {
            f"ci_tost_{k}": bc.interval(boots[a], 1 - 2 * bc.ALPHA_EQUIV / k)
            for k in (1, 2, 3)
        }
        for a in bc.FAMILY
    }
    scores = {
        a: {s: bc.scores_row(m, y, probs[f"{a}:{s}"]) for s in bc.STACK_ARMS}
        for a in ("F", *arms)
    }

    # breakdowns (descriptive): ticket 19's three, plus T's own t0 ball status
    rel = pd.read_parquet(READ19 / "relation.parquet").reindex(f_ids)
    t0_source = (
        sources[sources.offset_ms == 0].set_index("shot_id").source.reindex(f_ids)
    )
    status = meta.set_index("shot_id").status.reindex(f_ids)
    t0_kind = np.where(
        status != "ok",
        "no T (" + status.astype(str) + ")",
        t0_source.fillna("cached").to_numpy(),
    )
    t0_kind = np.where(
        (status == "ok") & (t0_kind == "cached"),
        "tracker: no ball (F's ball kept)",
        t0_kind,
    )
    groups = {
        "dist_band": bc.dist_band(f_table[si.col("dist", 0)].to_numpy(float)),
        "ball_height": bc.height_band(rel.bf_height.to_numpy(float)),
        "f_t0_ball_source": np.select(
            [rel[f"bf_src_{s}"].to_numpy() == 1 for s in bc.SOURCES],
            list(bc.SOURCES),
            "none",
        ),
        "t_t0_ball": np.asarray(t0_kind, dtype=object),
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
                "ll_F": float(f_ll[bc.PRIMARY][mask].mean()),
            }
            for a in ("C", "S2", "T", "T_flagsF"):
                d = ll[a][bc.PRIMARY] - f_ll[bc.PRIMARY]
                lo, hi = bc.interval(ss.boot_means(weights, d, mask), 0.95)
                row |= {a: float(d[mask].mean()), f"{a}_lo": lo, f"{a}_hi": hi}
            breakdown.append(row)
    breakdown = pd.DataFrame(breakdown)
    breakdown.to_csv(out / "breakdown.csv", index=False)

    # what T changed: the t0 ball (vs the cached one), the shot location (vs F)
    cached_t0 = frames[frames.offset_ms == 0].set_index("shot_id").reindex(f_ids)
    t0_swapped = swapped[swapped.offset_ms == 0].set_index("shot_id").reindex(f_ids)

    def centre(boxes):
        b = [
            np.asarray(v, float).reshape(-1)[:4]
            if v is not None and len(np.asarray(v).reshape(-1)) >= 4
            else np.full(4, np.nan)
            for v in boxes
        ]
        b = np.array(b)
        return np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2]

    tracked = t0_source.fillna("cached").str.startswith("tracked")
    cached_none = cached_t0.ball_conf.map(
        lambda v: not isinstance(v, list | np.ndarray) or len(v) == 0
    )
    moved_px = np.hypot(*(centre(t0_swapped.ball_xyxy) - centre(cached_t0.ball_xyxy)).T)
    dx = (t_table[si.col("shot_x", 0)] - f_table[si.col("shot_x", 0)]).abs()
    dd = (t_table[si.col("dist", 0)] - f_table[si.col("dist", 0)]).abs()
    both = t_table[si.col("shot_missing", 0)].eq(0) & f_table[
        si.col("shot_missing", 0)
    ].eq(0)
    changed = both & ((dx > 0.01) | (dd > 0.01))
    location = {
        "t0_ball_kind": pd.Series(t0_kind).value_counts().to_dict(),
        "t0_ball_new_where_F_had_none": int((tracked & cached_none).sum()),
        "t0_ball_moved_over_3px_share_of_tracked": float(
            (moved_px > 3)[tracked.to_numpy() & ~cached_none.to_numpy()].mean()
        ),
        "T_vs_F_location_changed_share": float(changed.sum() / both.sum()),
        "T_vs_F_dist_change_median_when_changed_m": float(dd[changed].median()),
        "T_vs_F_dist_change_gt5m_share": float((both & (dd > 5)).sum() / both.sum()),
        "row_sources": sources.groupby(["offset_ms", "source"])
        .size()
        .rename("rows")
        .reset_index()
        .to_dict("records"),
    }
    pd.DataFrame(
        {
            f"{a}:{s}": results[a]["probabilities"][s]
            for a in ("T", "T_flagsF")
            for s in bc.STACK_ARMS
        },
        index=f_table.index,
    ).assign(game=games).to_csv(out / "probabilities.csv")
    sources.to_parquet(out / "row_sources.parquet", index=False)
    record = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "started": started.isoformat(timespec="seconds"),
        "ticket": "ball-detection 20",
        "registration": "ticket 20 comment 2026-10-27",
        "population": {
            "shots": len(f_ids),
            "matches": int(pd.Series(games).nunique()),
            "goals": int(y.sum()),
        },
        "protocol_sha256": m["sha"],
        "controls": controls,
        "primary_stack_arm": bc.PRIMARY,
        "paired_vs_F": paired,
        "family": {
            "p_improve": p_imp,
            "p_tost": p_eq,
            "verdict": decision,
            "alpha_improve": bc.ALPHA_IMPROVE,
            "alpha_equiv": bc.ALPHA_EQUIV,
            "margin": bc.MARGIN,
            "variant": "two-sided +-2 s, offline, descriptive: no verdict"
            if args.two_sided
            else "causal (primary)",
        },
        "holm_level_intervals": holm_levels,
        "scores": scores,
        "chosen_c_ALL": {a: results[a]["chosen_c"]["ALL"] for a in results},
        "location": location,
    }
    (out / "read.json").write_text(json.dumps(record, indent=2, default=float) + "\n")
    print(
        json.dumps(
            {k: record[k] for k in ("controls", "family", "location")},
            indent=2,
            default=float,
        )
    )
    print(
        json.dumps(
            {a: paired[f"{a}:{bc.PRIMARY}"] for a in arms}, indent=2, default=float
        )
    )
    print(breakdown.to_string())


# ------------------------------------------------------------------ the 172 (D4, descriptive)
def detect172(args: argparse.Namespace) -> None:
    """Ticket 16's detection (``ball_xg.detect``: dd2c >= 0.01 + the second stage on the 0 / -400 / -800 ms jpgs, and
    the -800 ms frames an arm needs) on the non-penalty shots of the 172 evaluation matches, unchanged: only its
    population and output folder are swapped."""
    import ball_xg as bx

    bx.OUT = args.out / "172"
    bx.read_population = lambda limit=None: population(limit, "evaluation")
    bx.detect(args)


def read172(args: argparse.Namespace) -> None:
    """X0 / P1 / S2's t0 ball and their feet_t0 shot location against StatsBomb on ticket 17's 2,721 paired shots."""
    import ball_accuracy as ba
    import ball_scorecard as bs
    import ball_xg as bx
    from xgpoc.detector import location as loc

    folder = args.out / "172"
    shots = ba.shot_errors(ba.shot_rows()).set_index("shot_id")
    candidates = pd.read_parquet(folder / "candidates.parquet")
    ids = sorted(shots.index)
    if args.limit:  # smoke: the paired shots the smoke detection covered
        ids = [s for s in ids if s in set(candidates.shot_id)][: args.limit]
        shots = shots.loc[ids]
    frames = pd.read_parquet(
        bx.CACHE / "frames.parquet", filters=[("shot_id", "in", ids)]
    )
    extra = folder / "extra_frames.parquet"
    if extra.is_file():
        e = pd.read_parquet(extra)
        frames = pd.concat([frames, e[e.shot_id.isin(ids)]], ignore_index=True)
    if frames.duplicated(["shot_id", "offset_ms"]).any():
        raise SystemExit("a shot frame is twice in the frames table")
    target = shots.loc[ids, ["sb_mx", "sb_my"]].to_numpy(float)
    rows, controls, per = {}, {}, {}
    for arm in bx.ARMS:
        chosen = bx.choose(candidates, arm)
        af = bx.arm_frames(frames, chosen)
        t0 = af[af.offset_ms == 0].set_index("shot_id").reindex(ids)
        on_pitch = bx.has_homography(t0.fillna({"readable": False})).to_numpy()
        ball = np.full((len(ids), 2), np.nan)
        for i, xy in enumerate(t0.ball_pitch_xy):
            xy = np.asarray(xy if xy is not None else [], float)
            if xy.size >= 2:
                ball[i] = xy.reshape(-1, 2)[0]
        ball[~on_pitch] = np.nan
        feats = loc.feature_rows(af, loc.FEET_T0)
        f0 = feats[feats.offset_ms == 0].set_index("shot_id").reindex(ids)
        ok = f0.shot_ok == 1
        place = np.c_[f0.shot_x.where(ok), f0.shot_y.where(ok)]
        per[arm] = pd.DataFrame(
            {
                "ball_error_m": ba.aligned_error(ball, target),
                "location_error_m": np.hypot(*(place - target).T),
            },
            index=ids,
        )
        if arm == "X0":
            controls["X0_ball_equals_17"] = bool(
                np.allclose(
                    ball,
                    shots.loc[ids, ["ball_x", "ball_y"]].to_numpy(float),
                    equal_nan=True,
                    atol=1e-6,
                )
            )
            controls["X0_location_equals_feet_t0"] = bool(
                np.allclose(
                    place,
                    shots.loc[ids, ["feet_x", "feet_y"]].to_numpy(float),
                    equal_nan=True,
                    atol=1e-6,
                )
            )
    draws = bs.Draws(shots.loc[ids, "game"])
    homography = shots.loc[ids, "homography"].to_numpy(bool)
    for arm in bx.ARMS:
        for what in ("ball", "location"):
            frame = pd.DataFrame(
                {
                    "clip": shots.loc[ids, "game"].to_numpy(),
                    "error_m": per[arm][f"{what}_error_m"].to_numpy(),
                    "homography": homography,
                }
            )
            rows[
                f"{arm} {'piłka t0' if what == 'ball' else 'miejsce strzału feet_t0'}"
            ] = bs.summary(ba.measure_draws(frame, draws))
    moved = {}
    for arm in ("P1", "S2"):
        a, b = per[arm].location_error_m, per["X0"].location_error_m
        both = a.notna() & b.notna()
        moved[arm] = {
            "location_differs_share": float((np.abs(a - b)[both] > 1e-6).mean()),
            "median_error_change_m_when_differs": float(
                (a - b)[both & (np.abs(a - b) > 1e-6)].median()
            ),
        }
    record = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "ticket": "ball-detection 20 (D4)",
        "population": {"shots": len(ids), "matches": int(shots.game.nunique())},
        "controls": controls,
        "rows": rows,
        "location_vs_X0": moved,
    }
    (folder / "read172.json").write_text(
        json.dumps(record, indent=2, default=ba._json) + "\n"
    )
    pd.concat({a: per[a] for a in bx.ARMS}, axis=1).to_parquet(
        folder / "errors172.parquet"
    )
    lines = ba.summary_table(rows, False) + [""] + ba.summary_table(rows, True)
    (folder / "read172.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps({"controls": controls, "location_vs_X0": moved}, indent=2))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("dump")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--chunk", type=int, default=100)
    p.add_argument("--readers", type=int, default=3)
    p.add_argument(
        "--after", action="store_true", help="the two-sided variant's frames after t0"
    )
    p.add_argument("--device", default="cuda")
    p.set_defaults(func=dump)
    p = sub.add_parser("track")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument(
        "--two-sided", action="store_true", help="the offline variant (+-2 s)"
    )
    p.set_defaults(func=track)
    p = sub.add_parser("read")
    p.add_argument(
        "--two-sided",
        action="store_true",
        help="the offline variant's tracks (descriptive)",
    )
    p.set_defaults(func=read)
    p = sub.add_parser("detect172")
    p.add_argument("--device", default="cuda")
    p.add_argument("--chunk", type=int, default=2000)
    p.set_defaults(func=detect172)
    p = sub.add_parser("read172")
    p.set_defaults(func=read172)
    for p in sub.choices.values():
        p.add_argument(
            "--out",
            type=Path,
            default=OUT,
            help="results folder (smoke runs elsewhere)",
        )
        p.add_argument(
            "--limit",
            type=int,
            default=None,
            help="first N population shots only (smoke)",
        )
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
