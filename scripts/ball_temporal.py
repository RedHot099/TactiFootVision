"""ball-detection 04: how many ball misses tracking the ball through time recovers (offline, GSR valid at 25 fps).

    uv run python scripts/ball_temporal.py dump                  # dd2c candidates on all 43,500 frames (GPU, once)
    uv run python scripts/ball_temporal.py dump --variant tiles2x2 --post-filter --out DIR  # ticket 08's variant
    uv run python scripts/ball_temporal.py motion                # frame-to-frame camera motion of every clip (CPU)
    uv run python scripts/ball_temporal.py tune                  # sensitivity grid, games 2+3, off-benchmark frames
    uv run python scripts/ball_temporal.py track --games 2 3     # the chosen setting -> tracks + scorecard dumps
    uv run python scripts/ball_temporal.py breakdown             # the gain by ticket 01's categories (after the card)
    uv run python scripts/ball_temporal.py read --tracks DIR     # placement of tracks vs top-1, paired (ticket 09)
    uv run python scripts/ball_temporal.py dump --split test --out DIR  # ticket 09: GSR test at 25 fps (SNMOT frames)
    uv run python scripts/ball_temporal.py dump --split holdout --out DIR  # ticket 10: game 4 of GSR train at 25 fps
    uv run python scripts/ball_temporal.py track --fixes ball/12/chosen.json ...  # ticket 12's tracker (ball_track2)

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
imgsz 1280, NMS IoU 0.7, ``max_det`` 300) for every frame of the 58 clips; ``--variant tiles2x2`` dumps ticket 03's
tiles instead (``ball_inference.Tiled`` unchanged, its candidates >= ``DUMP_CONF`` kept: NMS and the ball merge keep a
candidate only against more confident ones, so these are the candidates a ``DUMP_CONF`` call would give) and
``--post-filter`` drops the ball candidates longer than ``RULE["post_filter_long_side_px"]`` (ticket 08). `track
--dump` tracks another 25 fps dump (the camera motion stays ``motion``'s: tiles take their people from the same full
frame 1280 call); the scorecard dumps then take the reference dump's single row. `tune` runs a stage of ``GRIDS`` (or ``AROUND``) on the
selection split (games 2 and 3) and scores it on the frames that are not benchmark frames (frame - 1 not a multiple
of 5; one truth ball with a pitch position on the pitch): a frame is hit when the chosen ball's centre is within
r = max(5 px, truth diameter) of the truth ball's (ticket 01's centre rule), so that choosing the setting never reads
the benchmark frames of the primary comparison; stage 2 writes the best setting of stages 1-2 to
``tune/chosen.json``. `track` runs one setting on the clips of ``--games`` and writes the
tracks and three scorecard dumps in ticket 01's format (`detector_bench.py dump`): ``track`` (the chosen ball is the
single ball candidate >= 0.15 of each frame; every other ball candidate is capped below 0.15, the tail kept for AP;
other classes and Ekstraklasa dev as in dd2c's ticket 01 dump), ``measured`` (the same without the interpolated balls)
and ``top1`` (dd2c's own top-1 ball encoded the same way, the reference for recall at r px).

Ticket 09 (the final candidate's "second read" of GSR test): ``--split test`` dumps, moves and tracks the GSR test
clips. Local GSR test holds only every 5th frame, so the 25 fps frames come from SoccerNet Tracking test
(``FULL_FRAMES``: the same clips, SNMOT-NNN = SNGS-NNN; ``check_frames`` asserts that every local GSR frame is byte for
byte the SNMOT one); labels, games and the frozen test homographies are GSR test's. ``track --post-filter`` drops the
ball candidates longer than ``RULE["post_filter_long_side_px"]`` before ``track_ball`` (the top-1 reference keeps the
unfiltered dump). `read` places the tracked ball and dd2c's top-1 ball (>= 0.15) on the split's placement frames
(``detector_bench.placement_rows``: the scorecard's denominator, frames without a homography count as misses) and
reads both with the scorecard's clip-bootstrap draws (``ball_scorecard.Draws``, ``placement_draws``, ``paired``), plus
the measured / interpolated / no-ball breakdown and phase 2's barrier (``ball_scorecard.barrier``: > 5 m of all
placement frames, paired); no scorecard dump is needed, so it reads GSR test too.

Ticket 10 (phase 2's holdout): ``--split holdout`` dumps, moves, tracks and reads game 4 of GSR train
(``detector_bench.HOLDOUT_FILE``'s 18 clips) like GSR test: 25 fps frames from SoccerNet Tracking train (byte-checked
against the local stride-5 GSR train frames), its frozen ``kp03_r2_refined`` homographies.
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
import ball_inference as bi  # noqa: E402
import ball_scorecard as bs  # noqa: E402
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
# GSR test and train are downloaded at stride 5 only (data/SoccerNetGS/fetch_log.jsonl); their 25 fps frames are
# SoccerNet Tracking's test / train sequences, the same clips (ball-detection spec, "Protokół oceny"; checked by
# ``check_frames``). Keyed by the GSR split folder: GSR test (ticket 09), the holdout's GSR train clips (ticket 10).
FULL_FRAMES = {folder: bench.TRACKING / folder for folder in ("test", "train")}
POST_FILTER = bs.RULE["post_filter_long_side_px"]
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
def clip_dirs(split: str = "valid") -> list[Path]:
    """The GSR clip folders (labels) of the read split ``split`` (the holdout: its listed GSR train clips)."""
    return [p.parent for p in bench.split_label_files(split)]


def frames_dir(clip_dir: Path) -> Path:
    """The folder holding every frame of a GSR clip: the clip itself, or for a split in ``FULL_FRAMES`` the SoccerNet
    Tracking sequence of the same clip."""
    full = FULL_FRAMES.get(clip_dir.parent.name)
    return clip_dir if full is None else full / clip_dir.name.replace("SNGS-", "SNMOT-")


def clip_game(clip_dir: Path) -> str:
    return str(
        json.loads((clip_dir / "Labels-GameState.json").read_text())["info"]["game_id"]
    )


def clip_frames(clip_dir: Path) -> list[Path]:
    """Every frame of a GSR clip in order (``img1/000001.jpg`` ...), from ``frames_dir``."""
    return sorted((frames_dir(clip_dir) / "img1").glob("*.jpg"))


def check_frames(clip_dir: Path) -> int:
    """The clip's local GSR frames that are byte for byte the 25 fps source's (raises on the first that is not or is
    missing there); returns how many were compared."""
    source = frames_dir(clip_dir)
    local = sorted((clip_dir / "img1").glob("*.jpg"))
    for path in local:
        twin = source / "img1" / path.name
        if not twin.is_file() or twin.read_bytes() != path.read_bytes():
            raise SystemExit(f"{path} is not the 25 fps source's {twin}")
    return len(local)


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


def dump_predictor(variant: str, weights: Path, device: str | None):
    """``(predict, class_names, row settings)`` of a 25 fps dump: dd2c's row 1280 at ``DUMP_CONF``, or ticket 03's
    tiles (``ball_inference``, dd2c's weights, candidates down to its 0.001 floor)."""
    if variant == ROW:
        settings = bench.ROWS[ROW]
        model = tv.load_model(
            "yolo",
            weights,
            conf=DUMP_CONF,
            iou=settings["iou"],
            imgsz=settings["imgsz"],
            device=device,
        )
        return model.predict, model.class_names, settings
    if weights != bi.WEIGHTS:
        raise SystemExit(f"{variant} is dumped with {bi.WEIGHTS} only")
    tiled = bi.predictor(variant, False, device)
    return tiled, tiled.full.class_names, bi.variant_settings(variant, False)


def dump_table(found: sv.Detections, frame: int, height: int, limit: float | None):
    """One frame's dump rows: the candidates >= ``DUMP_CONF``, without the ball candidates whose long side at 1080p
    is over ``limit`` (None: no filter)."""
    found = found[found.confidence >= DUMP_CONF]
    table = pd.DataFrame(
        np.asarray(found.xyxy, np.float32).reshape(-1, 4), columns=COLUMNS
    )
    table["conf"] = np.asarray(found.confidence, np.float32)
    table["class_id"] = np.asarray(found.class_id, np.int8)
    table = drop_long_balls(table, limit, height)
    table.insert(0, "frame", np.int16(frame))
    return table


def drop_long_balls(table: pd.DataFrame, limit: float | None, height: int = 1080):
    """``table`` without the ball candidates whose long side at 1080p (``height``: the frame's) is over ``limit``
    (ticket 03's post-filter; None: unchanged)."""
    if limit is None:
        return table
    side = np.maximum(table.x2 - table.x1, table.y2 - table.y1) * 1080 / height
    return table[~((table.class_id == 0) & (side > limit))].reset_index(drop=True)


def dump(args: argparse.Namespace) -> None:
    """Every candidate of every class >= ``DUMP_CONF`` on every frame of every GSR valid clip, row 1280 (native
    frames, imgsz 1280, NMS IoU 0.7, the bench's ``max_det``) or ``--variant``, one parquet per clip, then
    ``candidates.parquet``."""
    from ultralytics.cfg import DEFAULT_CFG_DICT

    predict, class_names, settings = dump_predictor(
        args.variant, args.weights, args.device
    )
    limit = POST_FILTER if args.post_filter else None
    row = args.variant + (f"+pf{limit}" if limit is not None else "")
    parts = args.out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    clips = clip_dirs(args.split)
    folder = bench.gsr_folder(args.split)
    checked = sum(check_frames(c) for c in clips) if folder in FULL_FRAMES else None
    timings, frames = [], []
    for clip_dir in clips:
        clip = clip_dir.name
        paths = clip_frames(clip_dir)
        game = clip_game(clip_dir)  # parses the clip's labels: once per clip
        frames += [{"clip": clip, "game": game, "frame": int(p.stem)} for p in paths]
        part = parts / f"{clip}.parquet"
        if part.is_file():
            continue
        clock = time.perf_counter()
        tables = []
        for path, image in read_ahead(paths):
            start = time.perf_counter()
            found = predict(image)
            timings.append(time.perf_counter() - start)
            tables.append(dump_table(found, int(path.stem), image.shape[0], limit))
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
        "ticket": args.ticket,
        "split": args.split,
        "frames_source": str(FULL_FRAMES.get(folder, bench.GSR / folder)),
        "gsr_frames_checked_identical": checked,
        "weights": str(args.weights),
        "weights_sha256": bench.sha256_file(args.weights),
        "conf": DUMP_CONF,
        "row": row,
        "settings": settings,
        "post_filter_long_side_px": limit,
        "max_det": DEFAULT_CFG_DICT["max_det"],
        "agnostic_nms": DEFAULT_CFG_DICT["agnostic_nms"],
        "half": DEFAULT_CFG_DICT["half"],
        "classes": dict(enumerate(class_names)),
        "clips": len(clips),
        "frames": len(frames),
        "candidates": len(candidates),
        "boxes": "xyxy in the native 1920x1080 frame pixels; frame = 1-based GSR frame (all frames, 25 fps)",
        "gpu": bench.gpu_name(args.device),
        "ms_per_frame": round(1000 * float(np.mean(timings[bench.DUMP_WARMUP :])), 2)
        if len(timings) > bench.DUMP_WARMUP
        else None,
        "ms_per_frame_note": "model call(s) per frame (inference + NMS; tiles: every tile, the full frame and the "
        "merge), batch 1, image read in a separate thread",
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


def _motion_worker(job: tuple[Path, Path, Path]) -> dict:
    clip_dir, folder, motion_dir = job
    out = motion_dir / f"{clip_dir.name}.npz"
    if not out.is_file():
        found = clip_motion(clip_dir, clip_candidates(clip_dir.name, folder))
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
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    clips = [
        c
        for c in clip_dirs(args.split)
        if (args.dump / "parts" / f"{c.name}.parquet").is_file()
    ]
    with multiprocessing.Pool(
        args.workers, initializer=_init_worker, initargs=(args.split,)
    ) as pool:
        rows = pool.map(_motion_worker, [(c, args.dump, out) for c in clips])
    table = pd.DataFrame(rows)
    table.to_csv(out / "summary.csv", index=False)
    (out / "settings.json").write_text(
        json.dumps(MOTION | {"reach": REACH}, indent=2) + "\n"
    )
    print(table.to_string(index=False))
    print(
        f"breaks {table.breaks.sum()} of {len(clips) * 749} frame pairs; anchor gap carried / still (median of "
        f"clips) {table.anchor_gap_carried_m.median():.2f} / {table.anchor_gap_still_m.median():.2f} m"
    )


def load_motion(clip: str, folder: Path = MOTION_DIR) -> np.ndarray:
    return np.load(folder / f"{clip}.npz")["motion"]


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


def clip_homographies(
    clip: str, frames: int, frozen: dict, motion_dir: Path = MOTION_DIR
) -> list[np.ndarray | None]:
    """Every frame's 1080p frame -> corner-metres homography: the frozen 720p ones carried by the camera motion."""
    return propagate(load_motion(clip, motion_dir), clip_anchors(clip, frozen, frames))


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


def _track_worker(job: tuple) -> list[pd.DataFrame]:
    """Tracks of one clip of the dump ``folder`` for every setting (``clip`` column added), the top-1 table first.
    ``job`` = (clip, settings, folder, split, motion folder, post-filter limit, fixes); the tracker sees the
    candidates without the balls over the limit, the top-1 table the dump's own. ``fixes`` (a dict of
    ``ball_track2.Fixes``) tracks with ticket 12's ``track_ball2`` instead of the package's ``track_ball``."""
    logging.getLogger("tactifoot_vision").setLevel(logging.WARNING)
    clip, settings, folder, split, motion_dir, limit, fixes = job
    candidates = clip_candidates(clip, folder)
    frames = len(clip_frames(bench.GSR / bench.gsr_folder(split) / clip))
    homographies = clip_homographies(clip, frames, _FROZEN, motion_dir)
    result = clip_result(drop_long_balls(candidates, limit), homographies)
    tables = [top_balls(candidates, frames)]
    if fixes is not None:  # ticket 12's tracker
        import ball_track2

        moves = load_motion(clip, motion_dir)
    for setting in settings:
        clock = time.perf_counter()
        if fixes is None:
            table = chosen_balls(track_ball(result, **setting))
        else:
            tracked, _ = ball_track2.track_ball2(
                result, moves, ball_track2.Fixes(**fixes), **setting
            )
            table = chosen_balls(tracked)
        table["seconds"] = time.perf_counter() - clock
        tables.append(table)
    for table in tables:
        table.insert(0, "clip", clip)
    return tables


_FROZEN: dict = {}


def _init_worker(split: str = "valid") -> None:
    _FROZEN.update(bench.frozen_homographies(split))


def run_tracks(
    clips: list[str],
    settings: list[dict],
    workers: int,
    folder: Path = DUMP,
    split: str = "valid",
    motion_dir: Path = MOTION_DIR,
    limit: float | None = None,
    fixes: dict | None = None,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """``(top-1 table, one tracks table per setting)`` over ``clips`` of the 25 fps dump ``folder`` (the tracker
    without the ball candidates over the post-filter ``limit``; ``fixes``: ticket 12's tracker)."""
    jobs = [(c, settings, folder, split, motion_dir, limit, fixes) for c in clips]
    with multiprocessing.Pool(
        workers, initializer=_init_worker, initargs=(split,)
    ) as pool:
        per_clip = pool.map(_track_worker, jobs)
    stacked = [
        pd.concat(tables, ignore_index=True) for tables in zip(*per_clip, strict=True)
    ]
    return stacked[0], stacked[1:]


def games_clips(games, split: str = "valid") -> list[str]:
    return [c.name for c in clip_dirs(split) if clip_game(c) in games]


def split_games(split: str) -> list[str]:
    return sorted({clip_game(c) for c in clip_dirs(split)}, key=int)


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
    """A scorecard dump (ticket 01's format) of the ``reference`` dump's row 1280 (or its only row; dd2c's ticket 01
    dump by default) with the ball of ``tracks`` as the single ball candidate >= 0.15 of each tracked benchmark
    frame: a measured ball keeps its box and confidence (at least 0.15), an interpolated one is added at 0.15, every
    other ball candidate is capped at ``CAP``. Other classes, untracked clips (outside ``games``) and Ekstraklasa dev
    are the reference's."""
    record, images, candidates = bench.read_dump(reference)
    row_name = ROW if ROW in record["rows"] else None
    if row_name is None:
        if len(record["rows"]) != 1:
            raise SystemExit(f"{reference}: no row {ROW} and not a single row")
        (row_name,) = record["rows"]
    candidates = candidates[candidates["row"] == row_name].copy()
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
            {"set": "gsr", "row": row_name, "image": row.image}
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
        "rows": {row_name: record["rows"][row_name]},
        "seconds": {
            k: v for k, v in record["seconds"].items() if k.endswith(f"/{row_name}")
        },
        "candidates": len(candidates),
        "source_dump": str(reference),
        "ball-detection 04": note
        | {"tracked_games": list(games), "added_balls": len(added)},
    }
    if "ms_per_frame" in record:  # ticket 01's dump has wall seconds only
        record["ms_per_frame"] = {
            k: v
            for k, v in record["ms_per_frame"].items()
            if k.endswith(f"/{row_name}")
        }
    (out / "dump.json").write_text(json.dumps(record, indent=2, default=float) + "\n")


def track(args: argparse.Namespace) -> None:
    """The chosen setting on the clips of ``--games``: tracks and the three scorecard dumps."""
    setting = json.loads(args.setting.read_text()) if args.setting else BASE
    if args.games == ["all"]:
        args.games = split_games(args.split)
    clips = games_clips(args.games, args.split)
    limit = POST_FILTER if args.post_filter else None
    fixes = None
    if args.fixes:
        found = json.loads(args.fixes.read_text())
        fixes = found.get("fixes", found)
    clock = time.perf_counter()
    top1, (tracks,) = run_tracks(
        clips, [setting], args.workers, args.dump, args.split, args.motion, limit, fixes
    )
    seconds = time.perf_counter() - clock
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    tracks.to_parquet(out / "tracks.parquet", index=False)
    top1.to_parquet(out / "top1.parquet", index=False)
    note = {
        "setting": setting,
        "tracker": (
            "tactifoot_vision.pipeline.track_ball"
            if fixes is None
            else f"ball_track2.track_ball2 with fixes {fixes} ({args.fixes})"
        )
        + f" on {args.dump} (conf >= {DUMP_CONF}, every frame)",
        "split": args.split,
        "clips": len(clips),
        "motion": str(args.motion),
        "post_filter_long_side_px": limit,
        "cpu_ms_per_frame": 1000
        * tracks.groupby("clip").seconds.first().sum()
        / len(tracks),
        "wall_seconds": seconds,
        "encoding": "the chosen ball is the single ball candidate >= 0.15 per frame; other balls capped at 0.1499",
    }
    (out / "track.json").write_text(json.dumps(note, indent=2, default=float) + "\n")
    counts = tracks.status.value_counts().to_dict()
    if args.split != "valid" or args.no_card_dumps:  # `read` places them without a dump
        print(json.dumps(note | {"status": counts}, indent=2, default=float))
        return
    measured = tracks.copy()
    measured.loc[measured.status == "interpolated", COLUMNS + ["conf"]] = np.nan
    measured.loc[measured.status == "interpolated", "status"] = "none"
    for name, table, extra in (
        ("track", tracks, {}),
        ("measured", measured, {"variant": "interpolated balls removed"}),
        ("top1", top1, {"variant": "dd2c's top-1 ball, same encoding"}),
    ):
        encode(table, args.reference, out / f"dump_{name}", note | extra, args.games)
    print(json.dumps(note | {"status": counts}, indent=2, default=float))


# ------------------------------------------------------------------ read (ticket 09)
# The reads of `read` per split: the GSR valid groups of the scorecard (game 5 only inside the full split, ticket
# 09), GSR test whole, the holdout (game 4, ticket 10) whole.
READS = {
    "valid": {"selection": SELECTION, "full": ("2", "3", "5")},
    "test": None,
    "holdout": None,
}


def table_top_ball(table: pd.DataFrame):
    """``detector_bench.placement_rows``' ``top_ball`` from a tracks (or top-1) table: the frame's ball centre in
    the homographies' ``PLACEMENT["input_size"]`` pixels (boxes in native 1920 x 1080), None without a ball."""
    scale = 1920 / bench.PLACEMENT["input_size"][0]
    balls = table[table.status != "none"]
    centres = {
        (c, int(f)): ((x1 + x2) / 2 / scale, (y1 + y2) / 2 / scale)
        for c, f, x1, y1, x2, y2 in zip(
            balls["clip"],
            balls.frame,
            balls.x1,
            balls.y1,
            balls.x2,
            balls.y2,
            strict=True,
        )
    }

    def top_ball(clip: str, frame: int, path: Path):
        centre = centres.get((clip, frame))
        return None if centre is None else np.asarray(centre)

    return top_ball


def status_rows(
    rows: pd.DataFrame, top_rows: pd.DataFrame, tracks: pd.DataFrame
) -> list[dict]:
    """Per tracking status on the placement frames (measured / interpolated / none): frames, share, well placed and
    > 5 m (of the status' frames) of tracking and top-1, and the gain in pp of all frames."""
    status = rows[["clip", "frame"]].merge(
        tracks[["clip", "frame", "status"]], on=["clip", "frame"], how="left"
    )["status"]
    if status.isna().any():
        raise SystemExit("tracks miss placement frames")
    far = bench.PLACEMENT["far_m"]
    well, top_well = rows.error_m <= far, top_rows.error_m <= far
    out = []
    for level in ("measured", "interpolated", "none"):
        part = (status == level).to_numpy()
        out.append(
            {
                "status": level,
                "frames": int(part.sum()),
                "share": float(part.mean()),
                "track_well": float(well[part].mean()) if part.any() else np.nan,
                "track_far": float((rows.error_m[part] > far).mean())
                if part.any()
                else np.nan,
                "top1_well": float(top_well[part].mean()) if part.any() else np.nan,
                "gain_pp_of_all": 100
                * float(well[part].sum() - top_well[part].sum())
                / len(rows),
            }
        )
    return out


def read_placement(
    tracks: pd.DataFrame, top1: pd.DataFrame, split: str, groups: dict | None = None
) -> dict:
    """Tracking vs top-1 on the placement frames of ``split`` per group of games (``READS``; None: all clips):
    the scorecard's placement numbers with clip-bootstrap intervals, the paired differences and the status rows."""
    homographies = bench.frozen_homographies(split)
    rows = bench.placement_rows(table_top_ball(tracks), homographies, split)
    top_rows = bench.placement_rows(table_top_ball(top1), homographies, split)
    keys = ["clip", "frame"]
    if not rows[keys].equals(top_rows[keys]):
        raise SystemExit("the two placements have other frames")
    clip_game_of = {c.name: clip_game(c) for c in clip_dirs(split)}
    groups = groups or {split: tuple(sorted(set(clip_game_of.values()), key=int))}
    out = {}
    for name, games in groups.items():
        keep = rows.game.isin(games).to_numpy()
        draws = bs.Draws([c for c, g in clip_game_of.items() if g in games])
        group_rows = rows[keep].reset_index(drop=True)
        group_top = top_rows[keep].reset_index(drop=True)
        found = bs.placement_draws(group_rows, draws)
        reference = bs.placement_draws(group_top, draws)
        out[name] = {
            "games": list(games),
            "counts": bs.placement_counts(group_rows),
            "track": bs.summary(found),
            "top1": bs.summary(reference),
            "track_vs_top1": bs.paired(found, reference),
            "status": status_rows(group_rows, group_top, tracks),
            "barrier": bs.barrier(group_rows, group_top, draws),
            "rows": group_rows,
            "top1_rows": group_top,
        }
    return out


def read_markdown(found: dict, title: str) -> str:
    def pct(entry: dict) -> str:
        low, high = (100 * v for v in entry["ci"])
        return f"{100 * entry['value']:.2f}% [{low:.1f}; {high:.1f}]"

    def pp(entry: dict) -> str:
        low, high = (100 * v for v in entry["ci"])
        return f"{100 * entry['difference']:+.2f} [{low:+.2f}; {high:+.2f}]"

    lines = [
        f"# {title}",
        "",
        "| group | frames (no H) | ceiling | top-1 well placed | tracking well placed | paired gain, pp [95%] | "
        "well with H: top-1 / tracking | > 5 m of placed: top-1 / tracking (pp) | placed: top-1 / tracking |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, g in found.items():
        t, r, d = g["track"], g["top1"], g["track_vs_top1"]
        lines.append(
            f"| {name} | {g['counts']['frames']} ({g['counts']['without_homography']}) | "
            f"{100 * t['ceiling']['value']:.1f}% | {pct(r['well_placed'])} | {pct(t['well_placed'])} | "
            f"**{pp(d['well_placed'])}** | {100 * r['well_placed_with_homography']['value']:.2f}% / "
            f"{100 * t['well_placed_with_homography']['value']:.2f}% ({pp(d['well_placed_with_homography'])}) | "
            f"{100 * r['far_of_placed']['value']:.2f}% / {100 * t['far_of_placed']['value']:.2f}% "
            f"({pp(d['far_of_placed'])}) | {100 * r['placed']['value']:.2f}% / {100 * t['placed']['value']:.2f}% |"
        )
    lines += [
        "",
        "| group | status | frames (share) | tracking well | tracking > 5 m | top-1 well | gain, pp of all |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, g in found.items():
        for s in g["status"]:
            lines.append(
                f"| {name} | {s['status']} | {s['frames']} ({100 * s['share']:.1f}%) | "
                f"{100 * s['track_well']:.1f}% | {100 * s['track_far']:.1f}% | {100 * s['top1_well']:.1f}% | "
                f"{s['gain_pp_of_all']:+.2f} |"
            )
    if all("barrier" in g for g in found.values()):
        lines += [
            "",
            "| group | > 5 m of all frames: reference / tracking | paired, pp [95%] | barrier (<= +1 pp) |",
            "|---|---|---|---|",
        ]
        for name, g in found.items():
            b = g["barrier"]
            low, high = (100 * v for v in b["ci"])
            lines.append(
                f"| {name} | {100 * b['reference']:.2f}% / {100 * b['variant']:.2f}% | "
                f"{100 * b['difference']:+.2f} [{low:+.2f}; {high:+.2f}] | {'met' if b['met'] else 'NOT met'} |"
            )
    gate = bs.RULE["gate_1280"]
    lines += [
        "",
        f"Gates (GSR valid values, orientation only): {100 * gate['satisfactory']:.2f}% / "
        f"{100 * gate['ambitious']:.2f}%. Bootstrap {bs.BOOTSTRAP['draws']} draws over clips, seed "
        f"{bs.BOOTSTRAP['seed']}.",
        "",
    ]
    return "\n".join(lines)


def read(args: argparse.Namespace) -> None:
    """Tracking (``--tracks``: `track`'s tracks.parquet) vs its top1.parquet (or ``--reference``, another tracks
    table in the same columns) on the placement frames."""
    tracks = pd.read_parquet(args.tracks / "tracks.parquet")
    top1 = pd.read_parquet(args.reference or args.tracks / "top1.parquet")
    found = read_placement(tracks, top1, args.split, READS[args.split])
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    for name, g in found.items():
        g.pop("rows").to_parquet(out / f"placement_{name}_track.parquet", index=False)
        g.pop("top1_rows").to_parquet(
            out / f"placement_{name}_top1.parquet", index=False
        )
    record = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "tracks": str(args.tracks),
        "reference": str(args.reference or args.tracks / "top1.parquet"),
        "split": args.split,
        "homographies": bench.homography_file(args.split),
        "bootstrap": bs.BOOTSTRAP,
        "groups": found,
    }
    (out / "read.json").write_text(json.dumps(record, indent=2, default=float) + "\n")
    text = read_markdown(
        found, args.title or f"{args.tracks.name}: tracking vs top-1 ({args.split})"
    )
    (out / "read.md").write_text(text)
    print(text)


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
    d.add_argument("--variant", choices=[ROW, "tiles2x2", "tiles3x2"], default=ROW)
    d.add_argument(
        "--post-filter",
        action="store_true",
        help="drop ball candidates longer than 45 px at 1080p (ticket 03's post-filter)",
    )
    d.add_argument("--out", type=Path, default=DUMP)
    d.add_argument("--split", choices=bench.READ_SPLITS, default="valid")
    d.add_argument("--device", default=None)
    d.add_argument(
        "--ticket", default="ball-detection 04", help="the dump record's ticket"
    )
    m = commands.add_parser(
        "motion", help="frame-to-frame camera motion of every dumped clip (CPU)"
    )
    m.add_argument("--dump", type=Path, default=DUMP, help="the 25 fps dump (people)")
    m.add_argument("--out", type=Path, default=MOTION_DIR)
    t = commands.add_parser(
        "tune", help="sensitivity grid on games 2+3, off-benchmark frames"
    )
    t.add_argument("--stage", type=int, choices=[*GRIDS, 3], default=1)
    k = commands.add_parser("track", help="one setting -> tracks and scorecard dumps")
    k.add_argument("--games", nargs="+", default=list(SELECTION), help="games, or all")
    k.add_argument(
        "--setting", type=Path, help="JSON of track_ball options (default: BASE)"
    )
    k.add_argument("--out", type=Path, default=OUT / "track")
    k.add_argument("--dump", type=Path, default=DUMP, help="the 25 fps dump to track")
    k.add_argument("--reference", type=Path, default=be.DUMP)
    k.add_argument("--motion", type=Path, default=MOTION_DIR, help="`motion`'s folder")
    k.add_argument(
        "--post-filter",
        action="store_true",
        help="drop ball candidates longer than 45 px at 1080p before tracking",
    )
    k.add_argument(
        "--fixes",
        type=Path,
        help="JSON of ball_track2.Fixes (or ticket 12's chosen.json): track with track_ball2",
    )
    k.add_argument(
        "--no-card-dumps",
        action="store_true",
        help="tracks only, no scorecard dumps (always so off GSR valid)",
    )
    r = commands.add_parser("read", help="placement of tracks vs top-1, paired")
    r.add_argument("--tracks", type=Path, required=True, help="`track`'s --out")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--title", default=None)
    r.add_argument(
        "--reference", type=Path, help="tracks parquet to pair with (default: top-1)"
    )
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
    for command in (m, k):
        command.add_argument("--split", choices=bench.READ_SPLITS, default="valid")
    r.add_argument("--split", choices=bench.READ_SPLITS, default="valid")
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
        "read": read,
        "breakdown": breakdown,
    }[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
