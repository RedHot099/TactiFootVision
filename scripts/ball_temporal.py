"""ball-detection 04: how many ball misses tracking the ball through time recovers (offline, GSR valid at 25 fps).

    uv run python scripts/ball_temporal.py dump                  # dd2c candidates on all 43,500 frames (GPU, once)
    uv run python scripts/ball_temporal.py motion                # frame-to-frame camera motion of every clip (CPU)
    uv run python scripts/ball_temporal.py tune                  # sensitivity grid, games 2+3, off-benchmark frames
    uv run python scripts/ball_temporal.py track --games 2 3     # the chosen setting -> tracks + scorecard dumps
    uv run python scripts/ball_temporal.py breakdown             # the gain by ticket 01's categories (after the card)

The tracker is the package's ``track_ball`` (``tactifoot_vision.pipeline``, ported from the prototype
``.scratch/ball-tracking/prototype/ball_track.py``: static and spot candidates and ball boxes on a person dropped, a
constant-velocity gate in pitch metres, restarts only on confident candidates, isolated outliers removed, short gaps
filled and flagged ``interpolated``), run unchanged on a ``PipelineResult`` built per clip from the dump. Nothing in
the package changes (ball-detection decision 6).

Gate space: pitch metres. The frozen homographies (``kp03_r2_refined``, 720p pixels -> corner metres) exist only on
every 5th frame, so every frame gets one by camera-motion compensation: frame-to-frame image homographies (`motion`:
corners away from person boxes, pyramidal Lucas-Kanade with a forward-backward check, RANSAC) carry the frozen
homographies of the nearest anchors before and after a frame (at most ``REACH`` frames away, never across a motion
break) to it, blended linearly between the two (``propagate``). On an anchor frame the frozen homography is used
unchanged, so the tracked ball is placed exactly as the scorecard places it.

`dump` writes every candidate of every class >= ``DUMP_CONF`` of dd2c in the pipeline's row (1280: native frames,
imgsz 1280, NMS IoU 0.7, ``max_det`` 300) for every frame of the 58 clips. `tune` runs a stage of ``GRIDS`` (or ``AROUND``) on the
selection split (games 2 and 3) and scores it on the frames that are not benchmark frames (frame - 1 not a multiple
of 5; one truth ball with a pitch position on the pitch): a frame is hit when the chosen ball's centre is within
r = max(5 px, truth diameter) of the truth ball's (ticket 01's centre rule), so that choosing the setting never reads
the benchmark frames of the primary comparison; stage 2 writes the best setting of stages 1-2 to
``tune/chosen.json``. `track` runs one setting on the clips of ``--games`` and writes the
tracks and three scorecard dumps in ticket 01's format (`detector_bench.py dump`): ``track`` (the chosen ball is the
single ball candidate >= 0.15 of each frame; every other ball candidate is capped below 0.15, the tail kept for AP;
other classes and Ekstraklasa dev as in dd2c's ticket 01 dump), ``measured`` (the same without the interpolated balls)
and ``top1`` (dd2c's own top-1 ball encoded the same way, the reference for recall at r px).
"""

import argparse
import copy
import json
import logging
import multiprocessing
import queue
import sys
import threading
import time
from datetime import UTC, datetime
from itertools import product
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ball_errors as be  # noqa: E402
import detector_bench as bench  # noqa: E402

import tactifoot_vision as tv  # noqa: E402
from tactifoot_vision.pipeline import (  # noqa: E402
    FrameResult,
    PipelineResult,
    reproject,
    track_ball,
)

OUT = be.ROOT / "ball/04"
DUMP = OUT / "dump25"
MOTION_DIR = OUT / "motion"
WEIGHTS = bench.CHECKOUT / "models/detector_yolo11l_1280_dd2c.pt"
ROW = "1280"
DUMP_CONF = 0.01
FPS = 25.0
COLUMNS = ["x1", "y1", "x2", "y2"]
CLASSES = ("ball", "goalkeeper", "player", "referee")
# the pipeline's people threshold: the person boxes track_ball checks against
PEOPLE_CONF = bench.OPERATING["conf"]
SELECTION = be.SELECTION_GAMES
# Camera motion between consecutive frames, estimated at half resolution.
MOTION = {
    "scale": 0.5,
    "corners": 1500,
    "quality": 0.005,
    "min_distance": 7,
    "forward_backward_px": 1.0,
    "ransac_px": 1.0,
    "min_inliers": 40,
    "min_inlier_share": 0.3,
    "people_margin_px": 4,
}
REACH = 10  # frames a frozen homography is carried by the camera motion
ANCHOR_SCALE = np.diag(
    [2 / 3, 2 / 3, 1.0]
)  # 1080p -> the frozen homographies' 720p pixels
# The package defaults of track_ball, except min_confidence: the dump's floor, so that weak candidates can be chosen.
BASE = {
    "min_confidence": DUMP_CONF,
    "restart_confidence": 0.45,
    "restart_after": 0.25,
    "gate": 1.5,
    "gate_speed": 32.5,
    "max_gap": 0.5,
    "static_window": 3.2,
}
# Sensitivity: the gate (both terms scaled), the longest filled gap, the restart confidence and the static test's
# window (a live ball at rest before a set piece is dropped by a short one; ``STATIC_OFF`` s can never span half of
# its window in a 30 s clip, so it turns the test off). Stage 1: around the package defaults, min_confidence on its own
# at the base; stage 2: around stage 1's best, which lay on the edge of the grid in four parameters.
STATIC_OFF = 1000.0
GRIDS = {
    1: {
        "gate_scale": (0.5, 1.0, 1.5),
        "max_gap": (0.2, 0.5, 1.0),
        "restart_confidence": (0.25, 0.45, 0.6),
        "static_window": (3.2, 10.0, 20.0, STATIC_OFF),
        "min_confidence": (DUMP_CONF,),
    },
    2: {
        "gate_scale": (1.5, 2.0, 3.0),
        "max_gap": (1.0, 1.5, 2.0),
        "restart_confidence": (0.15, 0.25, 0.35),
        "static_window": (20.0, STATIC_OFF),
        "min_confidence": (0.01, 0.05),
    },
}
MIN_CONFIDENCES = (0.01, 0.05, 0.12)
# Stage 3: one parameter at a time around the chosen setting (``tune/chosen.json``, the best hit rate of stages 1-2).
AROUND = {
    "gate_scale": (0.5, 1.0, 1.5, 2.0, 3.0),
    "max_gap": (0.2, 0.5, 1.0, 1.5, 2.0),
    "restart_confidence": (0.15, 0.25, 0.35, 0.45, 0.6),
    "static_window": (3.2, 10.0, 20.0, STATIC_OFF),
    "min_confidence": (0.01, 0.05, 0.12),
}
# ball candidates that are not the chosen ball stay below the operating point
CAP = np.float32(0.1499)
CHOSEN = np.float32(bench.OPERATING["class_conf"]["ball"])

logger = logging.getLogger("ball_temporal")


# ------------------------------------------------------------------ dump
def clip_dirs() -> list[Path]:
    return sorted(
        p.parent for p in (bench.GSR / "valid").glob("*/Labels-GameState.json")
    )


def clip_game(clip_dir: Path) -> str:
    return str(
        json.loads((clip_dir / "Labels-GameState.json").read_text())["info"]["game_id"]
    )


def clip_frames(clip_dir: Path) -> list[Path]:
    """Every frame of a GSR clip in order (``img1/000001.jpg`` ...)."""
    return sorted((clip_dir / "img1").glob("*.jpg"))


def read_ahead(paths: list[Path], flags: int = cv2.IMREAD_COLOR, depth: int = 16):
    """``(path, image)`` of every path, read by a background thread ``depth`` images ahead."""
    images: queue.Queue = queue.Queue(depth)

    def reader() -> None:
        for path in paths:
            images.put((path, cv2.imread(str(path), flags)))
        images.put(None)

    threading.Thread(target=reader, daemon=True).start()
    while (item := images.get()) is not None:
        yield item


def dump(args: argparse.Namespace) -> None:
    """Every candidate of every class >= ``DUMP_CONF`` on every frame of every GSR valid clip, row 1280 (native
    frames, imgsz 1280, NMS IoU 0.7, the bench's ``max_det``), one parquet per clip, then ``candidates.parquet``."""
    from ultralytics.cfg import DEFAULT_CFG_DICT

    settings = bench.ROWS[ROW]
    model = tv.load_model(
        "yolo",
        args.weights,
        conf=DUMP_CONF,
        iou=settings["iou"],
        imgsz=settings["imgsz"],
        device=args.device,
    )
    parts = args.out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    clips = clip_dirs()
    timings, frames = [], []
    for clip_dir in clips:
        clip = clip_dir.name
        paths = clip_frames(clip_dir)
        frames += [
            {"clip": clip, "game": clip_game(clip_dir), "frame": int(p.stem)}
            for p in paths
        ]
        part = parts / f"{clip}.parquet"
        if part.is_file():
            continue
        clock = time.perf_counter()
        tables = []
        for path, image in read_ahead(paths):
            start = time.perf_counter()
            found = model.predict(image)
            timings.append(time.perf_counter() - start)
            table = pd.DataFrame(
                np.asarray(found.xyxy, np.float32).reshape(-1, 4), columns=COLUMNS
            )
            table["conf"] = np.asarray(found.confidence, np.float32)
            table["class_id"] = np.asarray(found.class_id, np.int8)
            table.insert(0, "frame", np.int16(int(path.stem)))
            tables.append(table)
        frame = pd.concat(tables, ignore_index=True)
        frame.insert(0, "clip", clip)
        frame.to_parquet(part, index=False)
        logger.info(
            "%s: %d frames in %.0f s", clip, len(paths), time.perf_counter() - clock
        )
    candidates = pd.concat(
        [pd.read_parquet(parts / f"{c.name}.parquet") for c in clips], ignore_index=True
    )
    candidates["clip"] = candidates["clip"].astype("category")
    candidates.to_parquet(args.out / "candidates.parquet", index=False)
    pd.DataFrame(frames).to_parquet(args.out / "frames.parquet", index=False)
    record = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "ticket": "ball-detection 04",
        "weights": str(args.weights),
        "weights_sha256": bench.sha256_file(args.weights),
        "conf": DUMP_CONF,
        "row": ROW,
        "settings": settings,
        "max_det": DEFAULT_CFG_DICT["max_det"],
        "agnostic_nms": DEFAULT_CFG_DICT["agnostic_nms"],
        "half": DEFAULT_CFG_DICT["half"],
        "classes": dict(enumerate(model.class_names)),
        "clips": len(clips),
        "frames": len(frames),
        "candidates": len(candidates),
        "boxes": "xyxy in the native 1920x1080 frame pixels; frame = 1-based GSR frame (all frames, 25 fps)",
        "gpu": bench.gpu_name(args.device),
        "ms_per_frame": round(1000 * float(np.mean(timings[bench.DUMP_WARMUP :])), 2)
        if len(timings) > bench.DUMP_WARMUP
        else None,
        "ms_per_frame_note": "model call per frame (inference + NMS), batch 1, image read in a separate thread",
    }
    (args.out / "dump.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


def clip_candidates(clip: str, folder: Path = DUMP) -> pd.DataFrame:
    return pd.read_parquet(folder / "parts" / f"{clip}.parquet")


# ------------------------------------------------------------------ camera motion
def people_mask(
    shape: tuple[int, int], boxes: np.ndarray, scale: float, margin: int
) -> np.ndarray:
    """uint8 mask of ``shape`` (rows, columns): 255 except inside the (full-resolution) person ``boxes``."""
    mask = np.full(shape, 255, np.uint8)
    for x1, y1, x2, y2 in np.asarray(boxes, float).reshape(-1, 4) * scale:
        mask[
            max(int(y1) - margin, 0) : max(int(np.ceil(y2)) + margin, 0),
            max(int(x1) - margin, 0) : max(int(np.ceil(x2)) + margin, 0),
        ] = 0
    return mask


def frame_motion(
    previous: np.ndarray, current: np.ndarray, mask: np.ndarray | None
) -> tuple[np.ndarray | None, int]:
    """Image homography ``previous`` -> ``current`` (grey images of one size) and its RANSAC inliers; ``None`` when the
    corners do not agree on one (a cut, a blank frame)."""
    corners = cv2.goodFeaturesToTrack(
        previous,
        MOTION["corners"],
        MOTION["quality"],
        MOTION["min_distance"],
        mask=mask,
    )
    if corners is None or len(corners) < MOTION["min_inliers"]:
        return None, 0
    ahead, ok, _ = cv2.calcOpticalFlowPyrLK(previous, current, corners, None)
    back, ok_back, _ = cv2.calcOpticalFlowPyrLK(current, previous, ahead, None)
    good = (
        ok.ravel().astype(bool)
        & ok_back.ravel().astype(bool)
        & (
            np.linalg.norm((back - corners).reshape(-1, 2), axis=1)
            < MOTION["forward_backward_px"]
        )
    )
    if good.sum() < MOTION["min_inliers"]:
        return None, int(good.sum())
    matrix, inliers = cv2.findHomography(
        corners[good], ahead[good], cv2.RANSAC, MOTION["ransac_px"]
    )
    count = int(inliers.sum()) if inliers is not None else 0
    if (
        matrix is None
        or count < MOTION["min_inliers"]
        or count < MOTION["min_inlier_share"] * good.sum()
    ):
        return None, count
    return matrix, count


def to_full(matrix: np.ndarray, scale: float) -> np.ndarray:
    """A homography between images reduced by ``scale``, in full-resolution pixels."""
    reduce = np.diag([scale, scale, 1.0])
    return np.linalg.inv(reduce) @ matrix @ reduce


def clip_motion(clip_dir: Path, candidates: pd.DataFrame) -> dict:
    """``motion[t]``: full-resolution image homography of frame t + 1 -> frame t + 2 (NaN at a break), the RANSAC
    inliers and the seconds taken. Corners on the dump's person boxes (>= ``PEOPLE_CONF``) are not used."""
    clock = time.perf_counter()
    scale = MOTION["scale"]
    people = candidates[(candidates.class_id > 0) & (candidates.conf >= PEOPLE_CONF)]
    boxes = {f: g[COLUMNS].to_numpy(float) for f, g in people.groupby("frame")}
    paths = clip_frames(clip_dir)
    motion = np.full((len(paths) - 1, 3, 3), np.nan)
    inliers = np.zeros(len(paths) - 1, int)
    previous = None
    for t, (_, grey) in enumerate(read_ahead(paths, cv2.IMREAD_REDUCED_GRAYSCALE_2)):
        if previous is not None:
            mask = people_mask(
                previous.shape,
                boxes.get(t, np.zeros((0, 4))),
                scale,
                MOTION["people_margin_px"],
            )
            matrix, inliers[t - 1] = frame_motion(previous, grey, mask)
            if matrix is not None:
                motion[t - 1] = to_full(matrix, scale)
        previous = grey
    return {
        "motion": motion,
        "inliers": inliers,
        "seconds": time.perf_counter() - clock,
    }


def anchor_agreement(
    motion: np.ndarray, anchors: dict[int, np.ndarray]
) -> tuple[float, float]:
    """Median pitch distance (m) between the frozen homography of anchor ``k + STRIDE`` and that of anchor ``k``
    carried there by the motion (``carried``) or taken as it is (``still``), over a grid of the frame's lower
    two thirds and every consecutive anchor pair without a break: the frozen homographies' own frame-to-frame noise
    plus the motion's error, against the camera's movement."""
    xs, ys = np.meshgrid(np.linspace(200, 1720, 8), np.linspace(400, 1040, 5))
    grid = np.c_[xs.ravel(), ys.ravel()]
    carried, still = [], []
    for k in sorted(anchors):
        b = k + bench.STRIDE
        if b not in anchors or np.isnan(motion[k:b]).any():
            continue
        steps = np.eye(3)
        for t in range(k, b):
            steps = motion[t] @ steps
        target = tv.pitch.frame_to_pitch(grid, anchors[b])
        for matrix, found in (
            (anchors[k] @ np.linalg.inv(steps), carried),
            (anchors[k], still),
        ):
            found.append(
                np.median(
                    np.linalg.norm(
                        tv.pitch.frame_to_pitch(grid, matrix) - target, axis=1
                    )
                )
            )
    return (
        (float(np.median(carried)), float(np.median(still)))
        if carried
        else (np.nan, np.nan)
    )


def clip_anchors(clip: str, frozen: dict, frames: int = 750) -> dict[int, np.ndarray]:
    """0-based frame -> the frozen homography of that frame in 1080p pixels."""
    return {
        f - 1: np.asarray(h, float) @ ANCHOR_SCALE
        for (c, f), h in frozen.items()
        if c == clip and f <= frames
    }


def _motion_worker(clip_dir: Path) -> dict:
    out = MOTION_DIR / f"{clip_dir.name}.npz"
    if not out.is_file():
        found = clip_motion(clip_dir, clip_candidates(clip_dir.name))
        np.savez_compressed(out, motion=found["motion"], inliers=found["inliers"])
    saved = np.load(out)
    carried, still = anchor_agreement(
        saved["motion"], clip_anchors(clip_dir.name, _FROZEN)
    )
    return {
        "clip": clip_dir.name,
        "game": clip_game(clip_dir),
        "breaks": int(np.isnan(saved["motion"][:, 0, 0]).sum()),
        "median_inliers": float(np.median(saved["inliers"])),
        "anchor_gap_carried_m": carried,
        "anchor_gap_still_m": still,
    }


def motion(args: argparse.Namespace) -> None:
    MOTION_DIR.mkdir(parents=True, exist_ok=True)
    clips = [c for c in clip_dirs() if (DUMP / "parts" / f"{c.name}.parquet").is_file()]
    with multiprocessing.Pool(args.workers, initializer=_init_worker) as pool:
        rows = pool.map(_motion_worker, clips)
    table = pd.DataFrame(rows)
    table.to_csv(MOTION_DIR / "summary.csv", index=False)
    (MOTION_DIR / "settings.json").write_text(
        json.dumps(MOTION | {"reach": REACH}, indent=2) + "\n"
    )
    print(table.to_string(index=False))
    print(
        f"breaks {table.breaks.sum()} of {len(clips) * 749} frame pairs; anchor gap carried / still (median of "
        f"clips) {table.anchor_gap_carried_m.median():.2f} / {table.anchor_gap_still_m.median():.2f} m"
    )


def load_motion(clip: str) -> np.ndarray:
    return np.load(MOTION_DIR / f"{clip}.npz")["motion"]


# ------------------------------------------------------------------ homographies
def _normalise(matrix: np.ndarray) -> np.ndarray:
    return matrix / matrix[2, 2]


def propagate(
    motion: np.ndarray, anchors: dict[int, np.ndarray], reach: int = REACH
) -> list[np.ndarray | None]:
    """A frame -> pitch homography for every frame of a clip (0-based ``t``; ``motion[t]`` maps frame t's pixels to
    frame t + 1's, NaN at a break) from the ``anchors`` (``t -> homography``): an anchor's own frame keeps it; any other
    frame takes the nearest anchor at most ``reach`` frames before and after it, carried by the motion in between, and
    blends the two linearly by distance (one side alone when the other is missing); ``None`` without either."""
    frames = len(motion) + 1
    out: list[np.ndarray | None] = []
    for t in range(frames):
        if t in anchors:
            out.append(_normalise(np.asarray(anchors[t], float)))
            continue
        before = after = None
        carried = np.eye(3)  # frame a -> frame t
        for a in range(t - 1, max(t - reach, 0) - 1, -1):
            if np.isnan(motion[a]).any():
                break
            carried = carried @ motion[a]
            if a in anchors:
                before = (t - a, _normalise(anchors[a] @ np.linalg.inv(carried)))
                break
        carried = np.eye(3)  # frame t -> frame b
        for b in range(t + 1, min(t + reach, frames - 1) + 1):
            if np.isnan(motion[b - 1]).any():
                break
            carried = motion[b - 1] @ carried
            if b in anchors:
                after = (b - t, _normalise(anchors[b] @ carried))
                break
        if before and after:
            w = before[0] / (before[0] + after[0])
            out.append(_normalise((1 - w) * before[1] + w * after[1]))
        else:
            side = before or after
            out.append(None if side is None else side[1])
    return out


def clip_homographies(clip: str, frames: int, frozen: dict) -> list[np.ndarray | None]:
    """Every frame's 1080p frame -> corner-metres homography: the frozen 720p ones carried by the camera motion."""
    return propagate(load_motion(clip), clip_anchors(clip, frozen, frames))


# ------------------------------------------------------------------ tracking
def detections(table: pd.DataFrame) -> sv.Detections:
    order = np.argsort(-table["conf"].to_numpy(), kind="stable")
    table = table.iloc[order]
    ids = table["class_id"].to_numpy(int)
    return sv.Detections(
        xyxy=table[COLUMNS].to_numpy(np.float32).reshape(-1, 4),
        confidence=table["conf"].to_numpy(np.float32),
        class_id=ids,
        data={"class_name": np.array(CLASSES, dtype=object)[ids].astype(str)},
    )


def clip_result(
    candidates: pd.DataFrame,
    homographies: list[np.ndarray | None],
    frame_size=(1920, 1080),
) -> PipelineResult:
    """The pipeline's result of one clip as the dump sees it: per frame the ball candidates (every ball box of the
    dump, most confident first, ``ball`` the most confident), the people >= ``PEOPLE_CONF`` and the homography, with
    pitch positions projected (``reproject``). Frame ``t`` (0-based) is GSR frame ``t + 1`` at ``t / FPS`` s."""
    by_frame = {int(f): g for f, g in candidates.groupby("frame")}
    empty = candidates.iloc[:0]
    frames = []
    for t in range(len(homographies)):
        table = by_frame.get(t + 1, empty)
        balls = detections(table[table.class_id == 0])
        people = detections(table[(table.class_id > 0) & (table.conf >= PEOPLE_CONF)])
        frames.append(
            FrameResult(
                t, t / FPS, people, copy.deepcopy(balls[:1]), ball_candidates=balls
            )
        )
    result = PipelineResult(frames, FPS, frame_size, list(CLASSES))
    return reproject(result, homographies)


def chosen_balls(result: PipelineResult) -> pd.DataFrame:
    """One row per frame of a tracked result: GSR ``frame``, ``status`` (measured / interpolated / none), the box and
    the detector's confidence (NaN when interpolated)."""
    rows = []
    for frame in result.frames:
        ball = frame.ball
        row = (
            {"frame": frame.index + 1, "status": "none"}
            | dict.fromkeys(COLUMNS, np.nan)
            | {"conf": np.nan}
        )
        if len(ball):
            interpolated = bool(np.asarray(ball.data.get("interpolated", [False]))[0])
            row |= dict(zip(COLUMNS, ball.xyxy[0].astype(float), strict=True))
            row["status"] = "interpolated" if interpolated else "measured"
            row["conf"] = float(ball.confidence[0]) if not interpolated else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def top_balls(
    candidates: pd.DataFrame, frames: int, threshold: float = float(CHOSEN)
) -> pd.DataFrame:
    """dd2c's top-1 ball (most confident >= ``threshold``) per frame, in ``chosen_balls``' columns."""
    balls = candidates[(candidates.class_id == 0) & (candidates.conf >= threshold)]
    best = balls.loc[balls.groupby("frame")["conf"].idxmax()].set_index("frame")
    table = pd.DataFrame({"frame": np.arange(1, frames + 1)})
    table = table.join(best[COLUMNS + ["conf"]].astype(float), on="frame")
    table.insert(1, "status", np.where(table.conf.notna(), "measured", "none"))
    return table


def settings_grid(stage: int) -> list[dict]:
    """The sensitivity settings of a stage (``GRIDS``); stage 1 also scans ``min_confidence`` alone at the base,
    stage 3 is ``AROUND`` the chosen setting."""
    if stage == 3:
        return around(json.loads((OUT / "tune/chosen.json").read_text()))
    grid = [BASE | {"min_confidence": m} for m in MIN_CONFIDENCES] if stage == 1 else []
    levels = GRIDS[stage]
    for scale, gap, restart, window, floor in product(*levels.values()):
        grid.append(
            BASE
            | {
                "gate": BASE["gate"] * scale,
                "gate_speed": BASE["gate_speed"] * scale,
                "max_gap": gap,
                "restart_confidence": restart,
                "static_window": window,
                "min_confidence": floor,
            }
        )
    unique = {json.dumps(s, sort_keys=True): s for s in grid}
    return list(unique.values())


def around(chosen: dict) -> list[dict]:
    """``chosen`` with one parameter of ``AROUND`` changed at a time (``chosen`` itself first)."""
    grid = [chosen]
    for name, levels in AROUND.items():
        for level in levels:
            if name == "gate_scale":
                change = {
                    "gate": BASE["gate"] * level,
                    "gate_speed": BASE["gate_speed"] * level,
                }
            else:
                change = {name: level}
            grid.append(chosen | change)
    unique = {json.dumps(s, sort_keys=True): s for s in grid}
    return list(unique.values())


def setting_name(setting: dict) -> str:
    return (
        f"min{setting['min_confidence']:g}_gate{setting['gate']:g}+{setting['gate_speed']:g}_"
        f"gap{setting['max_gap']:g}_restart{setting['restart_confidence']:g}_"
        f"static{'off' if setting['static_window'] >= STATIC_OFF else format(setting['static_window'], 'g')}"
    )


def _track_worker(job: tuple[str, list[dict]]) -> list[pd.DataFrame]:
    """Tracks of one clip for every setting (``clip`` column added), the top-1 table first."""
    logging.getLogger("tactifoot_vision").setLevel(logging.WARNING)
    clip, settings = job
    candidates = clip_candidates(clip)
    frames = len(clip_frames(bench.GSR / "valid" / clip))
    result = clip_result(candidates, clip_homographies(clip, frames, _FROZEN))
    tables = [top_balls(candidates, frames)]
    for setting in settings:
        clock = time.perf_counter()
        table = chosen_balls(track_ball(result, **setting))
        table["seconds"] = time.perf_counter() - clock
        tables.append(table)
    for table in tables:
        table.insert(0, "clip", clip)
    return tables


_FROZEN: dict = {}


def _init_worker() -> None:
    _FROZEN.update(bench.frozen_homographies("valid"))


def run_tracks(
    clips: list[str], settings: list[dict], workers: int
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """``(top-1 table, one tracks table per setting)`` over ``clips``."""
    with multiprocessing.Pool(workers, initializer=_init_worker) as pool:
        per_clip = pool.map(_track_worker, [(c, settings) for c in clips])
    stacked = [
        pd.concat(tables, ignore_index=True) for tables in zip(*per_clip, strict=True)
    ]
    return stacked[0], stacked[1:]


def games_clips(games) -> list[str]:
    return [c.name for c in clip_dirs() if clip_game(c) in games]


# ------------------------------------------------------------------ tuning score
def truth_frames(clips: list[str]) -> pd.DataFrame:
    """Per frame of ``clips`` with exactly one truth ball with a pitch position, lying on the pitch (the placement
    denominator's rule, every frame): the truth box, its centre-rule radius and ``bench`` (a benchmark frame)."""
    rows = []
    for clip in clips:
        balls, _ = be.clip_labels(bench.GSR / "valid" / clip / "Labels-GameState.json")
        balls = balls[np.isfinite(balls.pitch_x)]
        single = balls.groupby("frame").filter(lambda g: len(g) == 1)
        on = (single.pitch_x.abs() <= bench.HALF_LENGTH) & (
            single.pitch_y.abs() <= bench.HALF_WIDTH
        )
        for ball in single[on].itertuples():
            rows.append(
                {"clip": clip, "frame": ball.frame}
                | dict(zip(("tx1", "ty1", "tx2", "ty2"), ball.box, strict=True))
            )
    table = pd.DataFrame(rows)
    table["radius"] = be.radius(table[["tx1", "ty1", "tx2", "ty2"]].to_numpy(float))
    table["bench"] = (table.frame - 1) % bench.STRIDE == 0
    return table


def image_scores(tracks: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """``truth`` with ``hit`` (the chosen ball's centre within r of the truth's) and ``chosen`` (a ball chosen)."""
    joined = truth.merge(tracks, on=["clip", "frame"], how="left")
    chosen = joined[COLUMNS].notna().all(axis=1).to_numpy()
    centre = be.centres(joined[COLUMNS].to_numpy(float))
    true = be.centres(joined[["tx1", "ty1", "tx2", "ty2"]].to_numpy(float))
    joined["chosen"] = chosen
    joined["hit"] = chosen & (
        np.linalg.norm(centre - true, axis=1) <= joined.radius.to_numpy()
    )
    return joined


def score_summary(scored: pd.DataFrame) -> dict:
    chosen = scored.chosen.to_numpy()
    hit = scored.hit.to_numpy()
    interpolated = (scored.status == "interpolated").to_numpy()
    return {
        "frames": len(scored),
        "hit": hit.mean(),
        "chosen": chosen.mean(),
        "wrong_of_chosen": (chosen & ~hit).sum() / max(chosen.sum(), 1),
        "interpolated": interpolated.mean(),
        "hit_interpolated": (hit & interpolated).sum() / max(interpolated.sum(), 1),
    }


def best_setting(folder: Path) -> dict:
    """The setting of stages 1 and 2 with the highest hit rate (the first one on a tie)."""
    table = pd.concat(
        [
            pd.read_csv(folder / f"sensitivity_stage{stage}.csv").iloc[1:]
            for stage in GRIDS
        ],
        ignore_index=True,
    )
    best = table.loc[table.hit.idxmax()]
    return {key: float(best[key]) for key in BASE}


def tune(args: argparse.Namespace) -> None:
    """The sensitivity grid on the selection split, scored on its off-benchmark frames only."""
    clips = games_clips(SELECTION)
    settings = settings_grid(args.stage)
    top1, tracked = run_tracks(clips, settings, args.workers)
    truth = truth_frames(clips)
    off = truth[~truth.bench].reset_index(drop=True)
    rows = [
        {"setting": "top-1 dd2c (>= 0.15)"} | score_summary(image_scores(top1, off))
    ]
    for setting, tracks in zip(settings, tracked, strict=True):
        rows.append(
            {"setting": setting_name(setting)}
            | setting
            | score_summary(image_scores(tracks, off))
            | {
                "ms_per_frame": 1000
                * tracks.groupby("clip").seconds.first().sum()
                / len(tracks)
            }
        )
    table = pd.DataFrame(rows)
    out = OUT / "tune"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"sensitivity_stage{args.stage}.csv", index=False)
    if args.stage == 2:
        (out / "chosen.json").write_text(json.dumps(best_setting(out), indent=2) + "\n")
    (out / f"sensitivity_stage{args.stage}.md").write_text(be.markdown(table) + "\n")
    print(be.markdown(table))


# ------------------------------------------------------------------ scorecard dumps
def encode(tracks: pd.DataFrame, reference: Path, out: Path, note: dict, games) -> None:
    """A scorecard dump (ticket 01's format, row 1280) of dd2c's ticket 01 dump with the ball of ``tracks`` as the
    single ball candidate >= 0.15 of each tracked benchmark frame: a measured ball keeps its box and confidence (at
    least 0.15), an interpolated one is added at 0.15, every other ball candidate is capped at ``CAP``. Other classes,
    untracked clips (outside ``games``) and Ekstraklasa dev are dd2c's."""
    record, images, candidates = bench.read_dump(reference)
    candidates = candidates[candidates["row"] == ROW].copy()
    candidates["row"] = candidates["row"].cat.remove_unused_categories()
    gsr_images = images[(images.set == "gsr") & images.game.isin(games)]
    tracked = gsr_images.merge(tracks, on=["clip", "frame"], how="left", validate="1:1")
    if tracked.status.isna().any():
        raise SystemExit("tracks miss benchmark frames of the tracked games")
    is_ball = (
        (candidates["set"] == "gsr")
        & (candidates.class_id == 0)
        & candidates.image.isin(gsr_images.image)
    )
    candidates.loc[is_ball, "conf"] = np.minimum(candidates.loc[is_ball, "conf"], CAP)
    added = []
    balls = candidates[is_ball]
    by_image = {
        image: group.index.to_numpy() for image, group in balls.groupby("image")
    }
    for row in tracked[tracked.status != "none"].itertuples():
        box = np.array([[row.x1, row.y1, row.x2, row.y2]], np.float32)
        index = by_image.get(row.image, np.zeros(0, int))
        if row.status == "measured" and len(index):
            iou = sv.box_iou_batch(
                box, candidates.loc[index, COLUMNS].to_numpy(np.float32)
            )[0]
            if iou.max() >= 0.9:
                candidates.loc[index[np.argmax(iou)], "conf"] = max(
                    np.float32(row.conf), CHOSEN
                )
                continue
            logger.warning(
                "%s: chosen ball not in the reference dump (IoU %.2f), added",
                row.image,
                iou.max(),
            )
        added.append(
            {"set": "gsr", "row": ROW, "image": row.image}
            | dict(zip(COLUMNS, box[0], strict=True))
            | {
                "conf": CHOSEN
                if row.status == "interpolated"
                else max(np.float32(row.conf), CHOSEN),
                "class_id": 0,
            }
        )
    if added:
        extra = pd.DataFrame(added)
        extra["class_name"] = "ball"
        candidates = pd.concat([candidates, extra], ignore_index=True)
    for column in ("set", "row", "class_name"):
        candidates[column] = candidates[column].astype("category")
    for column in COLUMNS + ["conf"]:
        candidates[column] = candidates[column].astype(np.float32)
    candidates["class_id"] = candidates["class_id"].astype(np.int8)
    out.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(out / "candidates.parquet", index=False)
    images.to_parquet(out / "images.parquet", index=False)
    record = record | {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "rows": {ROW: record["rows"][ROW]},
        "seconds": {
            k: v for k, v in record["seconds"].items() if k.endswith(f"/{ROW}")
        },
        "candidates": len(candidates),
        "source_dump": str(reference),
        "ball-detection 04": note
        | {"tracked_games": list(games), "added_balls": len(added)},
    }
    if "ms_per_frame" in record:  # ticket 01's dump has wall seconds only
        record["ms_per_frame"] = {
            k: v for k, v in record["ms_per_frame"].items() if k.endswith(f"/{ROW}")
        }
    (out / "dump.json").write_text(json.dumps(record, indent=2, default=float) + "\n")


def track(args: argparse.Namespace) -> None:
    """The chosen setting on the clips of ``--games``: tracks and the three scorecard dumps."""
    setting = json.loads(args.setting.read_text()) if args.setting else BASE
    clips = games_clips(args.games)
    clock = time.perf_counter()
    top1, (tracks,) = run_tracks(clips, [setting], args.workers)
    seconds = time.perf_counter() - clock
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    tracks.to_parquet(out / "tracks.parquet", index=False)
    top1.to_parquet(out / "top1.parquet", index=False)
    note = {
        "setting": setting,
        "tracker": "tactifoot_vision.pipeline.track_ball on dump25 (dd2c 1280, conf >= 0.01, every frame)",
        "cpu_ms_per_frame": 1000
        * tracks.groupby("clip").seconds.first().sum()
        / len(tracks),
        "wall_seconds": seconds,
        "encoding": "the chosen ball is the single ball candidate >= 0.15 per frame; other balls capped at 0.1499",
    }
    measured = tracks.copy()
    measured.loc[measured.status == "interpolated", COLUMNS + ["conf"]] = np.nan
    measured.loc[measured.status == "interpolated", "status"] = "none"
    for name, table, extra in (
        ("track", tracks, {}),
        ("measured", measured, {"variant": "interpolated balls removed"}),
        ("top1", top1, {"variant": "dd2c's top-1 ball, same encoding"}),
    ):
        encode(table, args.reference, out / f"dump_{name}", note | extra, args.games)
    counts = tracks.status.value_counts().to_dict()
    print(json.dumps(note | {"status": counts}, indent=2, default=float))


# ------------------------------------------------------------------ breakdown
def breakdown(args: argparse.Namespace) -> None:
    """Gain of tracking over top-1 on the selection split's placement frames, by ticket 01's categories."""
    folder = args.out
    track_rows = pd.read_parquet(args.card / "placement_selection_1280.parquet")
    top_rows = pd.read_parquet(args.reference_card / "placement_selection_1280.parquet")
    keys = ["clip", "frame"]
    if not track_rows[keys].equals(top_rows[keys]):
        raise SystemExit("the two cards have other placement frames")
    far = bench.PLACEMENT["far_m"]
    table = track_rows[keys + ["game", "homography", "near_goal"]].copy()
    table["track_well"] = (track_rows.error_m <= far).to_numpy()
    table["top_well"] = (top_rows.error_m <= far).to_numpy()
    table["track_far"] = (track_rows.error_m > far).to_numpy()
    table["top_far"] = (top_rows.error_m > far).to_numpy()
    tracks = pd.read_parquet(folder / "tracks.parquet")
    top1 = pd.read_parquet(folder / "top1.parquet")
    table = table.merge(
        tracks[keys + ["status", "conf"] + COLUMNS], on=keys, how="left"
    )
    top = top1[keys + COLUMNS].rename(columns={c: f"top_{c}" for c in COLUMNS})
    table = table.merge(top, on=keys, how="left")
    same = np.isclose(
        table[COLUMNS].to_numpy(float),
        table[[f"top_{c}" for c in COLUMNS]].to_numpy(float),
    ).all(1)
    table["source"] = np.select(
        [
            table.status == "none",
            table.status == "interpolated",
            same,
            table.conf < float(CHOSEN),
        ],
        [
            "no ball",
            "interpolated (gap fill)",
            "measured, = top-1",
            "measured, below 0.15",
        ],
        "measured >= 0.15, other than top-1",
    )
    # ticket 01's per-ball features (the frame's single truth ball) and its candidate level under dd2c
    truth = pd.read_parquet(args.truth)
    _, _, found = be.load_found(args.reference_dump)
    truth = be.candidate_columns(truth, found[("gsr", ROW)])
    single = truth.groupby("image").filter(lambda g: len(g) == 1)
    features = ["size", "blur", "occlusion", "in_air", "camera", "candidate"]
    table["image"] = table["clip"] + "/" + table.frame.map("{:06d}".format)
    table = table.merge(single[["image", *features]], on="image", how="left")
    table["near_goal"] = np.where(table.near_goal, "<=30 m", ">30 m")
    table["homography"] = np.where(table.homography, "frozen H", "no H")
    rows = []
    total = len(table)
    for feature in ["source", "homography", "near_goal", *features]:
        for level, part in table.groupby(feature, observed=True, dropna=False):
            rows.append(
                {
                    "feature": feature,
                    "level": str(level),
                    "frames": len(part),
                    "top1_well": part.top_well.mean(),
                    "track_well": part.track_well.mean(),
                    "recovered": int((part.track_well & ~part.top_well).sum()),
                    "lost": int((~part.track_well & part.top_well).sum()),
                    "gain_pp_of_all": 100
                    * (part.track_well.sum() - part.top_well.sum())
                    / total,
                    "far_top1": int(part.top_far.sum()),
                    "far_track": int(part.track_far.sum()),
                }
            )
    result = pd.DataFrame(rows)
    # where the recovered and the lost frames come from: the tracker's source x ticket 01's candidate level
    table["change"] = np.select(
        [table.track_well & ~table.top_well, ~table.track_well & table.top_well],
        ["recovered", "lost"],
        "same",
    )
    cross = pd.crosstab([table.change, table.source], table.candidate, margins=True)
    cross.to_csv(folder / "breakdown_cross.csv")
    table.to_parquet(folder / "breakdown_frames.parquet", index=False)
    result.to_csv(folder / "breakdown.csv", index=False)
    flat = cross.reset_index().rename(columns={"change": "change", "source": "source"})
    (folder / "breakdown.md").write_text(
        be.markdown(result) + "\n\n" + be.markdown(flat) + "\n"
    )
    print(be.markdown(result))
    print(cross.to_string())


# ------------------------------------------------------------------ commands
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    d = commands.add_parser(
        "dump", help="dd2c candidates on every GSR valid frame (GPU)"
    )
    d.add_argument("--weights", type=Path, default=WEIGHTS)
    d.add_argument("--out", type=Path, default=DUMP)
    d.add_argument("--device", default=None)
    m = commands.add_parser(
        "motion", help="frame-to-frame camera motion of every dumped clip (CPU)"
    )
    t = commands.add_parser(
        "tune", help="sensitivity grid on games 2+3, off-benchmark frames"
    )
    t.add_argument("--stage", type=int, choices=[*GRIDS, 3], default=1)
    k = commands.add_parser("track", help="one setting -> tracks and scorecard dumps")
    k.add_argument("--games", nargs="+", default=list(SELECTION))
    k.add_argument(
        "--setting", type=Path, help="JSON of track_ball options (default: BASE)"
    )
    k.add_argument("--out", type=Path, default=OUT / "track")
    k.add_argument("--reference", type=Path, default=be.DUMP)
    b = commands.add_parser("breakdown", help="the gain by ticket 01's categories")
    b.add_argument("--out", type=Path, default=OUT / "track")
    b.add_argument("--card", type=Path, default=OUT / "card_track")
    b.add_argument("--reference-card", type=Path, default=be.ROOT / "ball/02/dd2c")
    b.add_argument("--reference-dump", type=Path, default=be.DUMP)
    b.add_argument(
        "--truth",
        type=Path,
        default=be.ROOT / "ball/02/truth/balls_gsr_games_23.parquet",
    )
    for command in (m, t, k):
        command.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    tv.setup_logging("WARNING")
    {
        "dump": dump,
        "motion": motion,
        "tune": tune,
        "track": track,
        "breakdown": breakdown,
    }[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
