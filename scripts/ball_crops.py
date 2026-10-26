"""ball-detection 13: a second stage on crops around dd2c's ball candidates (rescorer + centre regressor).

    uv run python scripts/ball_crops.py oracle      # step 0, the gate (CPU): a perfect rescorer on games 2+3
    uv run python scripts/ball_crops.py view        # training / calibration images (game 4, GSR valid, test out)
    uv run python scripts/ball_crops.py candidates  # dd2c's ball candidates >= 0.01 on the view (GPU)
    uv run python scripts/ball_crops.py crops       # labelled samples and their 96 x 96 crops (CPU)
    uv run python scripts/ball_crops.py train       # ResNet18, two heads (GPU)
    uv run python scripts/ball_crops.py fuse        # the logistic fusion on the calibration candidates
    uv run python scripts/ball_crops.py rescore     # the rescored 25 fps dumps of GSR valid (GPU)
    uv run python scripts/ball_crops.py cost        # dd2c + second stage, ms per frame (GPU)
    uv run python scripts/ball_crops.py tune        # tracker thresholds on the fused scale (off-benchmark frames)
    uv run python scripts/ball_crops.py read        # rows (i) / (ii) vs phase 1 on games 2+3

Council round 2 (``.scratch/ball-detection/council/round2-13-decision.md``) fixes the design. Step 0: the oracle
rescorer gives every ball candidate of the 25 fps dump (``ball/04/dump25``, dd2c 1280, >= 0.01) that lies on a truth
ball (its centre within r = max(3 px, half the truth box's diagonal) of the nearest truth ball's centre)
``ORACLE_TRUE`` and every other one ``ORACLE_FALSE``; ``ball_temporal``'s unchanged tracker (ticket 04's setting, the
45 px post-filter, ticket 04's camera motion) tracks it and the read places the box centre on games 2+3 against phase
1's candidate (ticket 09's tracks); under ``GATE_PP`` the ticket stops.

The model: a square crop of max(32 px at 1080p, 3 x the candidate box's long side) around the candidate's centre,
resampled to 96 x 96 (RGB), into an ImageNet ResNet18 with a classification head and a centre-offset head (in crop
sides). Training data: SoccerNet-v3, TeamTrack and game 6 of GSR train; game 9's clips (every 5th frame) calibrate
the fusion sigmoid(a logit(p_crop) + b logit(p_dd2c) + c) and stop the training. Game 4, GSR valid and GSR test are
never in the view (``check_view``). The fused confidence replaces dd2c's in a dump of ``ball/04/dump25``'s format, so
``ball_temporal.py track --dump`` tracks it unchanged; the regressed rows also move the box by the offset.

Only games 2+3 are read: as in ticket 11 the frozen homographies are kept for their clips only and every other
clip's rows are dropped before any number.
"""

import argparse
import json
import logging
import multiprocessing
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ball_errors as be  # noqa: E402
import ball_projection as bp  # noqa: E402
import ball_scorecard as bs  # noqa: E402
import ball_temporal as bt  # noqa: E402
import detector_bench as bench  # noqa: E402

import tactifoot_vision as tv  # noqa: E402

OUT = be.ROOT / "ball/13"
SETTING = bt.OUT / "tune/chosen.json"
REFERENCE_TRACKS = be.ROOT / "ball/09/track_valid"
REFERENCE_READ = be.ROOT / "ball/09/read_valid/read.json"
GAMES = bt.SELECTION
ORACLE_TRUE, ORACLE_FALSE = np.float32(0.99), np.float32(0.001)
MIN_LABEL_RADIUS = 3.0  # px at 1080p (council decision 3)
GATE_PP = 3.0  # step 0: the oracle's point gain under which the ticket stops
FAR = bench.PLACEMENT["far_m"]

logger = logging.getLogger("ball_crops")


# ------------------------------------------------------------------ labels
def label_radius(xyxy: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Council decision 3: r = max(3 px, half the diagonal) of each truth box; ``scale``: native pixels per 1080p
    pixel (``image_scale``)."""
    xyxy = np.asarray(xyxy, float).reshape(-1, 4)
    half = np.hypot(xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]) / 2
    return np.maximum(MIN_LABEL_RADIUS * scale, half)


def image_scale(width: int, height: int) -> float:
    """Native pixels per 1080p-equivalent pixel (the long side against 1920, ``ball_view``'s scale)."""
    return max(width, height) / 1920


def truth_balls(labels: Path) -> dict[int, np.ndarray]:
    """GSR frame -> the truth ball boxes (xyxy, 1080p) of a clip's annotated frames."""
    balls, _ = be.clip_labels(labels)
    return {
        int(f): np.array(g.box.tolist(), float).reshape(-1, 4)
        for f, g in balls.groupby("frame")
    }


def match(
    boxes: np.ndarray, truth: np.ndarray | None, scale: float = 1.0
) -> dict[str, np.ndarray]:
    """Per candidate box: ``d`` (px) from its centre to the nearest truth ball's centre, that truth's ``r`` and its
    index (inf / nan / -1 without truth)."""
    n = len(boxes)
    if truth is None or not len(truth):
        return {
            "d": np.full(n, np.inf),
            "r": np.full(n, np.nan),
            "nearest": np.full(n, -1),
        }
    gap = np.linalg.norm(be.centres(boxes)[:, None] - be.centres(truth)[None], axis=2)
    nearest = gap.argmin(1)
    return {
        "d": gap[np.arange(n), nearest],
        "r": label_radius(truth, scale)[nearest],
        "nearest": nearest,
    }


def label_of(m: dict[str, np.ndarray]) -> np.ndarray:
    """Council decision 3: 1 when d <= r, 0 when d >= 2 r (or no truth ball), -1 in the band between (ignored)."""
    d = m["d"]
    r = np.nan_to_num(m["r"], nan=0.0)
    out = np.full(len(d), -1, np.int8)
    out[~np.isfinite(d) | (d >= 2 * r)] = 0
    out[np.isfinite(d) & (d <= r)] = 1
    return out


# ------------------------------------------------------------------ rescored dumps
def rescore_clip(
    candidates: pd.DataFrame, conf_of_balls, move_of_balls=None
) -> pd.DataFrame:
    """One clip's dump part with every ball candidate's confidence replaced by ``conf_of_balls(frame, boxes)`` (and
    its box moved by ``move_of_balls(frame, boxes)`` -> new xyxy, None: unchanged); other classes untouched."""
    out = candidates.copy()
    is_ball = (out.class_id == 0).to_numpy()
    for frame, index in out[is_ball].groupby("frame").groups.items():
        boxes = out.loc[index, bt.COLUMNS].to_numpy(float)
        out.loc[index, "conf"] = np.asarray(
            conf_of_balls(int(frame), boxes), np.float32
        )
        if move_of_balls is not None:
            out.loc[index, bt.COLUMNS] = np.asarray(
                move_of_balls(int(frame), boxes), np.float32
            )
    out["conf"] = out["conf"].astype(np.float32)
    for column in bt.COLUMNS:
        out[column] = out[column].astype(np.float32)
    return out


def write_dump(folder: Path, parts: dict[str, pd.DataFrame], record: dict) -> None:
    """A dump in ``ball/04/dump25``'s format: ``parts/<clip>.parquet``, ``candidates.parquet``, ``frames.parquet``
    (the clips' rows of the source's) and ``dump.json`` (the source's record updated with ``record``)."""
    (folder / "parts").mkdir(parents=True, exist_ok=True)
    for clip, part in parts.items():
        part.to_parquet(folder / "parts" / f"{clip}.parquet", index=False)
    candidates = pd.concat(list(parts.values()), ignore_index=True)
    candidates["clip"] = candidates["clip"].astype("category")
    candidates.to_parquet(folder / "candidates.parquet", index=False)
    frames = pd.read_parquet(bt.DUMP / "frames.parquet")
    frames[frames["clip"].isin(parts)].to_parquet(
        folder / "frames.parquet", index=False
    )
    source = json.loads((bt.DUMP / "dump.json").read_text())
    full = (
        source
        | {
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "source_dump": str(bt.DUMP),
            "clips": len(parts),
            "frames": int(frames["clip"].isin(parts).sum()),
            "candidates": len(candidates),
        }
        | record
    )
    (folder / "dump.json").write_text(json.dumps(full, indent=2, default=float) + "\n")


def oracle_part(clip: str, centre: bool) -> pd.DataFrame:
    """Step 0's oracle on one clip of ``ball/04/dump25``: true candidates ``ORACLE_TRUE``, others ``ORACLE_FALSE``;
    ``centre``: true candidates also moved onto their truth ball's centre (size kept)."""
    truth = truth_balls(bench.GSR / "valid" / clip / "Labels-GameState.json")

    def conf(frame: int, boxes: np.ndarray) -> np.ndarray:
        m = match(boxes, truth.get(frame))
        return np.where(m["d"] <= m["r"], ORACLE_TRUE, ORACLE_FALSE)

    def move(frame: int, boxes: np.ndarray) -> np.ndarray:
        m = match(boxes, truth.get(frame))
        out = boxes.copy()
        true = m["d"] <= m["r"]
        if true.any():
            shift = be.centres(truth[frame][m["nearest"][true]]) - be.centres(
                boxes[true]
            )
            out[true] += np.tile(shift, 2)
        return out

    return rescore_clip(bt.clip_candidates(clip), conf, move if centre else None)


# ------------------------------------------------------------------ the read (games 2+3)
def selection_homographies() -> tuple[list[str], dict]:
    clips = bt.games_clips(GAMES)
    keep = set(clips)
    frozen = bench.frozen_homographies("valid")
    return clips, {key: m for key, m in frozen.items() if key[0] in keep}


def read_rows(tracks: pd.DataFrame, homographies: dict) -> pd.DataFrame:
    """Placement rows of games 2+3 with the box centre of ``tracks``' ball (ticket 11's ``placement``, k = 0)."""
    return bp.placement(tracks, bp.CENTRE, homographies)


def compare(variant: pd.DataFrame, reference: pd.DataFrame, draws: bs.Draws) -> dict:
    """The variant vs the reference (placement rows of the same frames): numbers, paired gain, the ticket's verdict."""
    keys = ["clip", "frame"]
    if not variant[keys].equals(reference[keys]):
        raise SystemExit("the two placements have other frames")
    found, base = bp.numbers(variant, draws), bp.numbers(reference, draws)
    gain = bs.paired(found, base)
    return {
        "variant": bs.summary(found),
        "reference": bs.summary(base),
        "gain": gain,
        "verdict": bp.verdict(gain),
    }


def candidate_reach(
    rows: pd.DataFrame, homographies: dict, folder: Path = bt.DUMP
) -> dict:
    """Of the denominator frames whose reference ball is > 5 m away (``far``) or not placed (``none``; with a
    homography): the share with some dd2c ball candidate (>= 0.01, after the post-filter) placed <= 5 m."""
    scale = 1920 / bench.PLACEMENT["input_size"][0]
    out = {}
    with_h = rows[rows.homography]
    kinds = {
        "far": with_h[with_h.error_m > FAR],
        "none": with_h[~np.isfinite(with_h.error_m)],
    }
    cache: dict[str, pd.DataFrame] = {}
    for kind, part in kinds.items():
        reach = []
        for row in part.itertuples():
            if row.clip not in cache:
                c = bt.drop_long_balls(
                    bt.clip_candidates(row.clip, folder), bt.POST_FILTER
                )
                cache[row.clip] = c[c.class_id == 0]
            balls = cache[row.clip]
            balls = balls[balls.frame == row.frame]
            if not len(balls):
                reach.append(False)
                continue
            xy = be.centres(balls[bt.COLUMNS].to_numpy(float)) / scale
            pitch = tv.pitch.frame_to_pitch(
                xy.astype(np.float32), homographies[(row.clip, row.frame)]
            )
            gap = np.hypot(pitch[:, 0] - row.truth_x, pitch[:, 1] - row.truth_y)
            reach.append(bool((gap <= FAR).any()))
        out[kind] = {
            "frames": len(part),
            "with_candidate_within_5m": float(np.mean(reach)) if reach else np.nan,
        }
    return out


def pct(entry: dict) -> str:
    low, high = (100 * v for v in entry["ci"])
    return f"{100 * entry['value']:.2f}% [{low:.1f}; {high:.1f}]"


def pp(entry: dict) -> str:
    low, high = (100 * v for v in entry["ci"])
    return f"{100 * entry['difference']:+.2f} [{low:+.2f}; {high:+.2f}]"


def comparison_lines(found: dict[str, dict]) -> list[str]:
    lines = [
        "| row | reference well placed | variant well placed | paired gain, pp [95%] | > 5 m of all: reference / "
        "variant (pp) | barrier | wins | > 5 m of placed: variant |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, c in found.items():
        r, v, g, w = c["reference"], c["variant"], c["gain"], c["verdict"]
        lines.append(
            f"| {name} | {pct(r['well_placed'])} | {pct(v['well_placed'])} | **{pp(g['well_placed'])}** | "
            f"{100 * r['far_of_all']['value']:.2f}% / {100 * v['far_of_all']['value']:.2f}% "
            f"({pp(g['far_of_all'])}) | {'ok' if w['barrier'] else 'NO'} | {'yes' if w['wins'] else 'no'} | "
            f"{100 * v['far_of_placed']['value']:.2f}% |"
        )
    return lines


# ------------------------------------------------------------------ step 0
def oracle(args: argparse.Namespace) -> None:
    out = args.out
    setting = json.loads(SETTING.read_text())
    clips, homographies = selection_homographies()
    draws = bs.Draws(clips)
    reference = pd.read_parquet(REFERENCE_TRACKS / "tracks.parquet")
    top1 = pd.read_parquet(REFERENCE_TRACKS / "top1.parquet")
    reference_rows = read_rows(reference, homographies)
    check = bp.centre_check(
        {
            "rows": reference_rows,
            "top1_rows": read_rows(top1, homographies),
            "track": bp.numbers(reference_rows, draws),
            "top1": bp.numbers(read_rows(top1, homographies), draws),
        },
        reference,
        json.loads(REFERENCE_READ.read_text())["groups"]["selection"],
    )
    found, single = {}, {}
    for name, centre in (("oracle", False), ("oracle_centre", True)):
        folder = out / name / "dump25"
        if not (folder / "dump.json").is_file():
            parts = {clip: oracle_part(clip, centre) for clip in clips}
            write_dump(
                folder,
                parts,
                {
                    "ticket": "ball-detection 13",
                    "row": f"1280 oracle{' + truth centre' if centre else ''}",
                    "rescored": f"ball candidates: {ORACLE_TRUE:g} when the centre is within max(3 px, half the "
                    f"truth diagonal) of a truth ball's centre, else {ORACLE_FALSE:g}"
                    + ("; true ones moved onto the truth centre" if centre else ""),
                },
            )
        oracle_top1, (tracks,) = bt.run_tracks(
            clips,
            [setting],
            args.workers,
            folder,
            "valid",
            bt.MOTION_DIR,
            bt.POST_FILTER,
        )
        tracks.to_parquet(out / name / "tracks.parquet", index=False)
        oracle_top1.to_parquet(out / name / "top1.parquet", index=False)
        found[f"{name} vs phase 1"] = compare(
            read_rows(tracks, homographies), reference_rows, draws
        )
        single[f"{name} top-1 vs dd2c top-1"] = compare(
            read_rows(oracle_top1, homographies), read_rows(top1, homographies), draws
        )
    reach = candidate_reach(reference_rows, homographies)
    gain_pp = 100 * found["oracle vs phase 1"]["gain"]["well_placed"]["difference"]
    record = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "games": list(GAMES),
        "clips": len(clips),
        "setting": setting,
        "reference_tracks": str(REFERENCE_TRACKS),
        "reference_check": check,
        "bootstrap": bs.BOOTSTRAP,
        "gate_pp": GATE_PP,
        "oracle_gain_pp": gain_pp,
        "gate_passed": bool(gain_pp >= GATE_PP),
        "tracked": found,
        "single_frame": single,
        "candidate_reach": reach,
    }
    (out / "oracle.json").write_text(json.dumps(record, indent=2, default=float) + "\n")
    lines = [
        "# ball-detection 13, step 0: oracle rescorer on games 2+3 of GSR valid",
        "",
        f"Reference: phase 1's candidate (ticket 09's tracks); check vs ticket 09's card: "
        f"{'identical' if check['identical'] else 'DIFFERS'} ({check['numbers']} numbers). Gate: point gain >= "
        f"{GATE_PP:g} pp -> {'PASSED' if record['gate_passed'] else 'NOT passed'} ({gain_pp:+.2f} pp).",
        "",
        *comparison_lines(found),
        "",
        "Single frame (top-1 >= 0.15):",
        "",
        *comparison_lines(single),
        "",
        "Denominator frames (with a homography) where phase 1's ball is > 5 m / not placed: share with a dd2c ball "
        "candidate (>= 0.01, post-filtered) placed <= 5 m.",
        "",
        "| phase 1 | frames | with a candidate <= 5 m |",
        "|---|---|---|",
        *(
            f"| {k} | {v['frames']} | {100 * v['with_candidate_within_5m']:.1f}% |"
            for k, v in reach.items()
        ),
        "",
        f"Bootstrap {bs.BOOTSTRAP['draws']} draws over clips, seed {bs.BOOTSTRAP['seed']}.",
        "",
    ]
    text = "\n".join(lines)
    (out / "oracle.md").write_text(text)
    print(text)


# ------------------------------------------------------------------ the training view
TRAIN_SETS = ("soccernet_v3", "teamtrack")  # every image of run 2c's view of these sets
GSR_GAMES = {
    "6": "train",
    "9": "calibration",
}  # GSR train games (council decision 4); game 4 is the holdout
CALIBRATION_STRIDE = (
    bench.STRIDE
)  # calibration frames: every 5th frame of game 9's clips
POOL = be.ROOT / "pool_index.parquet"
POOL_GSR = be.ROOT / "pool_index_gsr.parquet"
RUN2C_LIST = be.ROOT / "runs/run2c/train.txt"


def label_file(image: str) -> str:
    """Ultralytics' images/ -> labels/ rule."""
    return image.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"


def build_view() -> pd.DataFrame:
    """Every image of the second stage's data: ``role`` train (SoccerNet-v3 and TeamTrack, every image of run 2c's
    view; GSR train game 6, every frame) or calibration (GSR train game 9, every ``CALIBRATION_STRIDE``-th frame),
    with its label file, size and provenance. Only images with at least one truth ball (decision 3: frames with a
    full ball annotation)."""
    listed = set(RUN2C_LIST.read_text().split())
    pool = pd.read_parquet(POOL)
    pool = pool[pool.set.isin(TRAIN_SETS) & pool.image.isin(listed)]
    rows = pd.DataFrame(
        {
            "image": pool.image,
            "label": pool.image.map(label_file),
            "source": pool.set,
            "gsr_split": None,
            "game": pool.game.astype(object),
            "clip": pool["clip"].astype(object),
            "frame": pool.frame.astype("Int64"),
            "width": pool.width,
            "height": pool.height,
            "n_ball": pool.n_ball,
            "role": "train",
        }
    )
    gsr = pd.read_parquet(POOL_GSR)
    gsr = gsr[
        gsr.gsr_split.eq("train") & gsr.game_id.isin([f"gsr:{g}" for g in GSR_GAMES])
    ]
    role = gsr.game_id.str.removeprefix("gsr:").map(GSR_GAMES)
    keep = (role == "train") | ((gsr.frame - 1) % CALIBRATION_STRIDE == 0)
    gsr, role = gsr[keep], role[keep]
    rows = pd.concat(
        [
            rows,
            pd.DataFrame(
                {
                    "image": gsr.image,
                    "label": gsr.label,
                    "source": "soccernet_gsr",
                    "gsr_split": gsr.gsr_split,
                    "game": gsr.game_id,
                    "clip": gsr["clip"],
                    "frame": gsr.frame.astype("Int64"),
                    "width": gsr.width,
                    "height": gsr.height,
                    "n_ball": gsr.n_ball,
                    "role": role,
                }
            ),
        ],
        ignore_index=True,
    )
    rows = rows[rows.n_ball > 0].reset_index(drop=True)
    check_view(rows)
    return rows


def excluded_clips() -> set[str]:
    """The clips no training or calibration image may come from: game 4 (the holdout list) and every GSR valid and
    test clip."""
    clips = set(bench.holdout_clips())
    for split in ("valid", "test"):
        clips |= {p.parent.name for p in bench.split_label_files(split)}
    return clips


def check_view(view: pd.DataFrame) -> None:
    """Raises when the view holds an image of game 4, of GSR valid or test, or a GSR image outside GSR train."""
    gsr = view[view.source == "soccernet_gsr"]
    if not gsr.gsr_split.eq("train").all():
        raise SystemExit("a GSR image outside GSR train")
    if gsr.game.eq(f"gsr:{bench.HOLDOUT['game']}").any():
        raise SystemExit("a game 4 image in the view")
    names = view.image.map(lambda p: Path(p).name.split("_")[0])
    bad = set(gsr["clip"]) & excluded_clips() | (
        set(names[view.source == "soccernet_gsr"]) & excluded_clips()
    )
    if bad:
        raise SystemExit(f"excluded clips in the view: {sorted(bad)[:5]}")
    if (
        view[view.source != "soccernet_gsr"]
        .image.str.contains("SNGS-|SNMOT-|soccernet_gsr")
        .any()
    ):
        raise SystemExit("a GSR image listed under another source")


def view_command(args: argparse.Namespace) -> None:
    view = build_view()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    view.to_parquet(out / "view.parquet", index=False)
    summary = view.groupby(["role", "source"]).agg(
        images=("image", "size"), balls=("n_ball", "sum")
    )
    record = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "ticket": "ball-detection 13",
        "rule": "SoccerNet-v3 + TeamTrack (every image of run 2c's view) and GSR train game 6 (every frame) train; "
        f"GSR train game 9 every {CALIBRATION_STRIDE}th frame calibrates the fusion and stops the training; only "
        "images with >= 1 truth ball; game 4, GSR valid and GSR test out (check_view)",
        "excluded_clips": len(excluded_clips()),
        "per_role_source": {
            f"{r}/{s}": v for (r, s), v in summary.to_dict("index").items()
        },
        "gsr_clips": {
            role: sorted(
                view[(view.source == "soccernet_gsr") & (view.role == role)][
                    "clip"
                ].unique()
            )
            for role in ("train", "calibration")
        },
    }
    (out / "view.json").write_text(json.dumps(record, indent=2, default=int) + "\n")
    print(summary.to_string())


# ------------------------------------------------------------------ dd2c candidates on the view (GPU)
def candidates_command(args: argparse.Namespace) -> None:
    """dd2c's ball candidates (>= ``bt.DUMP_CONF``, row 1280) on every view image, one parquet per chunk."""

    view = pd.read_parquet(args.view)
    settings = bench.ROWS[bt.ROW]
    model = tv.load_model(
        "yolo",
        bt.WEIGHTS,
        conf=bt.DUMP_CONF,
        iou=settings["iou"],
        imgsz=settings["imgsz"],
        device=args.device,
    )
    parts = args.out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    timings = []
    for start in range(0, len(view), args.chunk):
        path = parts / f"{start:07d}.parquet"
        if path.is_file():
            continue
        chunk = view.iloc[start : start + args.chunk]
        tables = []
        for image_path, image in bt.read_ahead([Path(p) for p in chunk.image]):
            if image is None:
                raise SystemExit(f"cannot read {image_path}")
            clock = time.perf_counter()
            found = model.predict(image)
            timings.append(time.perf_counter() - clock)
            found = found[(found.class_id == 0) & (found.confidence >= bt.DUMP_CONF)]
            table = pd.DataFrame(
                np.asarray(found.xyxy, np.float32).reshape(-1, 4), columns=bt.COLUMNS
            )
            table["conf"] = np.asarray(found.confidence, np.float32)
            table.insert(0, "image", str(image_path))
            tables.append(table)
        pd.concat(tables, ignore_index=True).to_parquet(path, index=False)
        logger.info("%d / %d images", min(start + args.chunk, len(view)), len(view))
    found = pd.concat(
        [pd.read_parquet(p) for p in sorted(parts.glob("*.parquet"))], ignore_index=True
    )
    found.to_parquet(args.out / "candidates.parquet", index=False)
    record = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "weights": str(bt.WEIGHTS),
        "weights_sha256": bench.sha256_file(bt.WEIGHTS),
        "row": bt.ROW,
        "settings": settings,
        "conf": bt.DUMP_CONF,
        "classes": "ball only",
        "images": len(view),
        "candidates": len(found),
        "boxes": "xyxy in native image pixels",
        "ms_per_frame": round(1000 * float(np.mean(timings[bench.DUMP_WARMUP :])), 2)
        if len(timings) > 20
        else None,
    }
    (args.out / "candidates.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


# ------------------------------------------------------------------ crop samples (CPU)
CROP = {"min_side_1080p": 32.0, "box_factor": 3.0, "size": 96}  # council decision 2
NEGATIVES_PER_POSITIVE = (
    3  # decision 3: 1:3, half of them the highest-scored false candidates
)
HARD_SHARE = 0.5
CHAIN = {"frames": 3, "min_px": 16.0, "box_factor": 2.0, "every": 12, "cap": 10}
JITTER = {
    "radius_share": 1.0,
    "log2_scale": 0.3,
}  # truth positives: centre within r, size x 2^U(-0.3, 0.3)
SEED = 0


def crop_side(boxes: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Decision 2: the square crop's side in native pixels, max(32 px at 1080p, 3 x the box's long side)."""
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    long = np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
    return np.maximum(CROP["min_side_1080p"] * scale, CROP["box_factor"] * long)


def long_side_1080p(boxes: np.ndarray, scale: float) -> np.ndarray:
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    return np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]) / scale


def extract(image: np.ndarray, centres: np.ndarray, sides: np.ndarray) -> np.ndarray:
    """``(n, size, size, 3)`` uint8 BGR crops: squares of ``sides`` around ``centres`` (native pixels), resampled
    bilinearly to ``CROP["size"]``, outside the image black."""
    size = CROP["size"]
    out = np.zeros((len(centres), size, size, 3), np.uint8)
    for i, ((cx, cy), side) in enumerate(zip(centres, sides, strict=True)):
        k = size / side
        matrix = np.array(
            [[k, 0, size / 2 - k * cx], [0, k, size / 2 - k * cy]], np.float64
        )
        out[i] = cv2.warpAffine(
            image,
            matrix,
            (size, size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    return out


def yolo_balls(label: str, width: int, height: int) -> np.ndarray:
    """The ball boxes (class 0) of a YOLO label file, xyxy in native pixels."""
    path = Path(label)
    rows = np.loadtxt(path, ndmin=2) if path.stat().st_size else np.zeros((0, 5))
    rows = rows[rows[:, 0] == 0]
    cx, cy = rows[:, 1] * width, rows[:, 2] * height
    w, h = rows[:, 3] * width, rows[:, 4] * height
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)


def image_samples(
    item, candidates: pd.DataFrame, rng: np.random.Generator
) -> list[dict]:
    """The candidate rows of one view image (after the 45 px post-filter; label by decision 3, offset target for
    positives) and, for training images, one jittered truth positive per truth ball."""
    scale = image_scale(item.width, item.height)
    truth = yolo_balls(item.label, item.width, item.height)
    boxes = candidates[bt.COLUMNS].to_numpy(float).reshape(-1, 4)
    keep = long_side_1080p(boxes, scale) <= bt.POST_FILTER
    boxes, conf = boxes[keep], candidates.conf.to_numpy(float)[keep]
    m = match(boxes, truth, scale)
    labels = label_of(m)
    rows = []
    centres = be.centres(boxes)
    sides = crop_side(boxes, scale)
    for i in range(len(boxes)):
        target = (
            (be.centres(truth[m["nearest"][i]])[0] - centres[i]) / sides[i]
            if labels[i] == 1
            else (np.nan, np.nan)
        )
        rows.append(
            {
                "kind": "candidate",
                "cx": centres[i, 0],
                "cy": centres[i, 1],
                "side": sides[i],
                "long_px": long_side_1080p(boxes[i], scale)[0],
                "conf": conf[i],
                "label": int(labels[i]),
                "d_over_r": m["d"][i] / m["r"][i] if np.isfinite(m["d"][i]) else np.inf,
                "dx": target[0],
                "dy": target[1],
            }
        )
    if item.role == "train":
        radius = label_radius(truth, scale)
        for box, r in zip(truth, radius, strict=True):
            angle, reach = rng.uniform(0, 2 * np.pi), r * np.sqrt(rng.uniform())
            shift = (
                reach
                * JITTER["radius_share"]
                * np.array([np.cos(angle), np.sin(angle)])
            )
            factor = 2 ** rng.uniform(-JITTER["log2_scale"], JITTER["log2_scale"])
            centre = be.centres(box)[0] + shift
            half = (box[2:] - box[:2]) * factor / 2
            fake = np.r_[centre - half, centre + half]
            side = crop_side(fake, scale)[0]
            rows.append(
                {
                    "kind": "truth",
                    "cx": centre[0],
                    "cy": centre[1],
                    "side": side,
                    "long_px": long_side_1080p(fake, scale)[0],
                    "conf": np.nan,
                    "label": 1,
                    "d_over_r": reach / r,
                    "dx": -shift[0] / side,
                    "dy": -shift[1] / side,
                }
            )
    return rows


def chain_ids(negatives: pd.DataFrame) -> np.ndarray:
    """The same mistake over consecutive frames of a GSR clip: a negative joins the chain whose last member is at
    most ``CHAIN["frames"]`` frames before it and within max(16 px, 2 x its box's long side); else it opens one."""
    ids = np.full(len(negatives), -1)
    order = np.argsort(negatives.frame.to_numpy(), kind="stable")
    frames = negatives.frame.to_numpy()
    xy = negatives[["cx", "cy"]].to_numpy(float)
    reach = np.maximum(
        CHAIN["min_px"],
        CHAIN["box_factor"] * negatives.side.to_numpy() / CROP["box_factor"],
    )
    chains: list[list] = []  # open chains: [last frame, x, y, id]
    opened = 0
    for i in order:
        best, gap = None, np.inf
        for chain in chains:
            if 0 < frames[i] - chain[0] <= CHAIN["frames"]:
                g = np.hypot(*(xy[i] - chain[1:3]))
                if g <= reach[i] and g < gap:
                    best, gap = chain, g
        if best is None:
            best = [frames[i], *xy[i], opened]
            chains.append(best)
            opened += 1
        best[0], best[1], best[2] = frames[i], *xy[i]
        ids[i] = best[3]
        chains = [c for c in chains if frames[i] - c[0] <= CHAIN["frames"]]
    return ids


def thin_chains(negatives: pd.DataFrame) -> pd.DataFrame:
    """GSR negatives with at most every ``CHAIN["every"]``-th member of a chain, ``CHAIN["cap"]`` per chain."""
    kept = []
    for _, clip in negatives.groupby("clip", sort=False):
        clip = clip.copy()
        clip["chain"] = chain_ids(clip)
        clip = clip.sort_values(["chain", "frame"])
        position = clip.groupby("chain").cumcount()
        kept.append(
            clip[
                (position % CHAIN["every"] == 0)
                & (position < CHAIN["every"] * CHAIN["cap"])
            ]
        )
    return pd.concat(kept) if kept else negatives.iloc[:0]


def sample_training(table: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Decision 3 on the training rows: every positive; per source up to ``NEGATIVES_PER_POSITIVE`` negatives per
    positive, ``HARD_SHARE`` of them the most confident (GSR's thinned along chains first), the rest at random. When
    a source has fewer negatives than that, all are kept and weighted (``weight``) so that the source's negatives
    weigh ``NEGATIVES_PER_POSITIVE`` times its positives in the loss."""
    out = []
    for source, part in table.groupby("source", sort=False):
        positives = part[part.label == 1]
        negatives = part[(part.label == 0) & (part.kind == "candidate")]
        if source == "soccernet_gsr":
            negatives = thin_chains(negatives)
        want = min(len(negatives), NEGATIVES_PER_POSITIVE * len(positives))
        hard = negatives.sort_values("conf", ascending=False).iloc[
            : int(want * HARD_SHARE)
        ]
        rest = negatives.drop(hard.index)
        easy = rest.iloc[rng.permutation(len(rest))[: want - len(hard)]]
        weight = max(1.0, NEGATIVES_PER_POSITIVE * len(positives) / max(want, 1))
        out += [
            positives.assign(weight=1.0),
            hard.assign(hard=True, weight=weight),
            easy.assign(hard=False, weight=weight),
        ]
    sampled = pd.concat(out)
    sampled["hard"] = sampled["hard"].fillna(False).astype(bool)
    return sampled.sort_index().reset_index(drop=True)


def _crop_worker(job: tuple) -> None:
    path, rows, store, shape = job
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"cannot read {path}")
    crops = np.lib.format.open_memmap(store, mode="r+", dtype=np.uint8, shape=shape)
    index = rows.index.to_numpy()
    crops[index] = extract(
        image, rows[["cx", "cy"]].to_numpy(float), rows.side.to_numpy(float)
    )
    crops.flush()


def write_crops(table: pd.DataFrame, store: Path, workers: int) -> None:
    """Every row's crop into the ``.npy`` ``store`` (row i = table row i), images read in parallel."""
    shape = (len(table), CROP["size"], CROP["size"], 3)
    np.lib.format.open_memmap(store, mode="w+", dtype=np.uint8, shape=shape).flush()
    jobs = [
        (image, rows, store, shape)
        for image, rows in table.groupby("image", sort=False)
    ]
    with multiprocessing.Pool(workers) as pool:
        for done, _ in enumerate(pool.imap_unordered(_crop_worker, jobs, chunksize=16)):
            if done % 5000 == 0:
                logger.info("crops: %d / %d images", done, len(jobs))


def crops_command(args: argparse.Namespace) -> None:
    """The training samples (decision 3) and the calibration rows (natural proportion) with their crops."""
    rng = np.random.default_rng(SEED)
    view = pd.read_parquet(args.view)
    check_view(view)
    found = pd.read_parquet(args.candidates)
    by_image = dict(tuple(found.groupby("image", sort=False)))
    empty = found.iloc[:0]
    rows = []
    for item in view.itertuples():
        for row in image_samples(item, by_image.get(item.image, empty), rng):
            rows.append(
                row
                | {
                    "image": item.image,
                    "source": item.source,
                    "role": item.role,
                    "clip": item.clip,
                    "frame": item.frame,
                }
            )
    table = pd.DataFrame(rows)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out / "all_rows.parquet", index=False)
    train = sample_training(table[table.role == "train"], rng)
    calibration = table[
        (table.role == "calibration") & (table.kind == "candidate")
    ].reset_index(drop=True)
    record = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "crop": CROP,
        "chain": CHAIN,
        "jitter": JITTER,
        "negatives_per_positive": NEGATIVES_PER_POSITIVE,
        "hard_share": HARD_SHARE,
        "seed": SEED,
        "sets": {},
    }
    for name, part in (("train", train), ("calibration", calibration)):
        part.to_parquet(out / f"{name}.parquet", index=False)
        write_crops(part, out / f"{name}_crops.npy", args.workers)
        counts = part.groupby(["source", "kind", "label"]).size()
        record["sets"][name] = {
            "rows": len(part),
            "by_source_kind_label": {
                "/".join(map(str, k)): int(v) for k, v in counts.items()
            },
        }
    (out / "crops.json").write_text(json.dumps(record, indent=2, default=float) + "\n")
    print(json.dumps(record, indent=2, default=float))


# ------------------------------------------------------------------ the model (GPU)
TRAIN = {
    "batch": 256,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "max_epochs": 12,
    "patience": 3,
    "offset_weight": 5.0,
    "smooth_l1_beta": 0.05,
    "colour_jitter": 0.2,
    "workers": 8,
    "seed": SEED,
}
MEAN = np.array([0.485, 0.456, 0.406], np.float32)  # ImageNet, RGB
STD = np.array([0.229, 0.224, 0.225], np.float32)
P_CLIP = 1e-6


def crop_net():
    """Decision 2: ResNet18 (ImageNet weights) with a classification head (1 logit) and a centre-offset head (dx,
    dy in crop sides)."""
    import torch
    import torchvision

    class CropNet(torch.nn.Module):
        def __init__(self, pretrained: bool = True):
            super().__init__()
            weights = (
                torchvision.models.ResNet18_Weights.IMAGENET1K_V1
                if pretrained
                else None
            )
            self.backbone = torchvision.models.resnet18(weights=weights)
            self.backbone.fc = torch.nn.Identity()
            self.cls = torch.nn.Linear(512, 1)
            self.offset = torch.nn.Linear(512, 2)

        def forward(self, x):
            features = self.backbone(x)
            return self.cls(features).squeeze(1), self.offset(features)

    return CropNet


def to_tensor(crops: np.ndarray, device):
    """uint8 BGR ``(n, s, s, 3)`` crops -> normalised RGB float ``(n, 3, s, s)`` on ``device``."""
    import torch

    x = torch.from_numpy(np.ascontiguousarray(crops[..., ::-1])).to(device)
    x = x.permute(0, 3, 1, 2).float().div_(255)
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def load_net(weights: Path, device):
    import torch

    net = crop_net()(pretrained=False)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    net.load_state_dict(state["model"])
    return net.to(device).eval()


def predict_crops(net, crops: np.ndarray, device, batch: int = 512):
    """``(p_crop, offsets)`` of uint8 BGR crops."""
    import torch

    probs, offsets = [], []
    with (
        torch.inference_mode(),
        torch.autocast("cuda", dtype=torch.float16, enabled=device != "cpu"),
    ):
        for start in range(0, len(crops), batch):
            logit, offset = net(to_tensor(crops[start : start + batch], device))
            probs.append(torch.sigmoid(logit.float()).cpu().numpy())
            offsets.append(offset.float().cpu().numpy())
    if not probs:
        return np.zeros(0, np.float32), np.zeros((0, 2), np.float32)
    return np.concatenate(probs), np.concatenate(offsets)


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, P_CLIP, 1 - P_CLIP)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def train_command(args: argparse.Namespace) -> None:
    """Decision 5: BCE + ``offset_weight`` x SmoothL1 (offsets in crop sides, positives only), AdamW, cosine over
    ``max_epochs``; the epoch with the lowest log loss of the fused confidence (the fusion fitted on the calibration
    candidates, labels 0 / 1, band out) is kept (early stopping after ``patience`` epochs without a better one); the
    raw p_crop's log loss is a poor criterion because training weighs negatives 3:1 and the calibration rows are
    mostly positive."""
    import torch

    torch.manual_seed(SEED)
    device = args.device or "cuda"
    folder = args.crops
    table = pd.read_parquet(folder / "train.parquet")
    crops = np.load(folder / "train_crops.npy", mmap_mode="r")
    calibration = pd.read_parquet(folder / "calibration.parquet")
    calibration_crops = np.load(folder / "calibration_crops.npy", mmap_mode="r")
    scored = (calibration.label >= 0).to_numpy()
    y_cal = calibration.label.to_numpy()[scored]
    conf_cal = calibration.conf.to_numpy(float)[scored]
    labels = table.label.to_numpy(np.float32)
    weights = table.weight.to_numpy(np.float32)
    targets = table[["dx", "dy"]].to_numpy(np.float32)

    class Crops(torch.utils.data.Dataset):
        def __len__(self):
            return len(table)

        def __getitem__(self, i):
            crop = np.array(crops[i])
            target = targets[i].copy()
            rng = np.random.default_rng((SEED, i, torch.initial_seed() % 2**31))
            if rng.uniform() < 0.5:
                crop = crop[:, ::-1]
                target[0] = -target[0]
            j = TRAIN["colour_jitter"]
            crop = np.clip(
                crop * rng.uniform(1 - j, 1 + j) + rng.uniform(-j, j) * 255 * 0.25,
                0,
                255,
            )
            return (
                crop.astype(np.uint8),
                labels[i],
                np.nan_to_num(target),
                weights[i],
            )

    loader = torch.utils.data.DataLoader(
        Crops(),
        batch_size=TRAIN["batch"],
        shuffle=True,
        num_workers=TRAIN["workers"],
        drop_last=True,
        persistent_workers=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    net = crop_net()(pretrained=True).to(device)
    optimiser = torch.optim.AdamW(
        net.parameters(), lr=TRAIN["lr"], weight_decay=TRAIN["weight_decay"]
    )
    steps = TRAIN["max_epochs"] * len(loader)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser,
        max_lr=TRAIN["lr"],
        total_steps=steps,
        pct_start=0.05,
        anneal_strategy="cos",
    )
    scaler = torch.amp.GradScaler()
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    smooth = torch.nn.SmoothL1Loss(beta=TRAIN["smooth_l1_beta"])
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    history, best, stale = [], np.inf, 0
    clock = time.perf_counter()
    for epoch in range(TRAIN["max_epochs"]):
        net.train()
        sums = np.zeros(3)
        for step, (x, y, t, w) in enumerate(loader):
            x = to_tensor(x.numpy(), device)
            y, t, w = y.to(device), t.to(device), w.to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                logit, offset = net(x)
                loss_cls = (bce(logit.float(), y) * w).sum() / w.sum()
                positive = y > 0.5
                loss_off = (
                    smooth(offset.float()[positive], t[positive])
                    if positive.any()
                    else logit.sum() * 0
                )
                loss = loss_cls + TRAIN["offset_weight"] * loss_off
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            schedule.step()
            sums += [loss.item(), loss_cls.item(), loss_off.item()]
            if step % 200 == 0:
                logger.info(
                    "epoch %d step %d / %d loss %.4f",
                    epoch,
                    step,
                    len(loader),
                    loss.item(),
                )
        net.eval()
        p, offsets = predict_crops(net, calibration_crops, device)
        positives = (calibration.label == 1).to_numpy()
        offset_error = np.abs(
            offsets[positives] - calibration.loc[positives, ["dx", "dy"]].to_numpy()
        ).mean()
        entry = {
            "epoch": epoch,
            "train_loss": sums[0] / len(loader),
            "train_bce": sums[1] / len(loader),
            "train_smooth_l1": sums[2] / len(loader),
            "calibration_log_loss_raw": log_loss(p[scored], y_cal),
            "calibration_log_loss": log_loss(
                fused(fit_fusion(p[scored], conf_cal, y_cal), p[scored], conf_cal),
                y_cal,
            ),
            "calibration_offset_mae_sides": float(offset_error),
            "seconds": time.perf_counter() - clock,
        }
        history.append(entry)
        logger.info("%s", entry)
        torch.save(
            {"model": net.state_dict(), "epoch": epoch}, out / f"epoch{epoch}.pt"
        )
        if entry["calibration_log_loss"] < best:
            best, stale = entry["calibration_log_loss"], 0
            torch.save({"model": net.state_dict(), "epoch": epoch}, out / "best.pt")
        else:
            stale += 1
            if stale >= TRAIN["patience"]:
                break
        (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    record = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "train": TRAIN,
        "crop": CROP,
        "rows": len(table),
        "batches_per_epoch": len(loader),
        "history": history,
        "best_epoch": int(
            min(history, key=lambda e: e["calibration_log_loss"])["epoch"]
        ),
        "weights": str(out / "best.pt"),
        "weights_sha256": bench.sha256_file(out / "best.pt"),
        "gpu": bench.gpu_name(device),
        "gpu_hours": (time.perf_counter() - clock) / 3600,
    }
    (out / "train.json").write_text(json.dumps(record, indent=2, default=float) + "\n")
    print(json.dumps(record, indent=2, default=float))


# ------------------------------------------------------------------ fusion (decision 5)
def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), P_CLIP, 1 - P_CLIP)
    return np.log(p / (1 - p))


def fused(fusion: dict, p_crop, p_dd2c) -> np.ndarray:
    """sigmoid(a logit(p_crop) + b logit(p_dd2c) + c)."""
    z = fusion["a"] * logit(p_crop) + fusion["b"] * logit(p_dd2c) + fusion["c"]
    return 1 / (1 + np.exp(-z))


def fit_fusion(p_crop: np.ndarray, p_dd2c: np.ndarray, y: np.ndarray) -> dict:
    """The logistic fusion fitted by maximum likelihood (no penalty) at the calibration rows' natural proportion."""
    from sklearn.linear_model import LogisticRegression

    x = np.c_[logit(p_crop), logit(p_dd2c)]
    model = LogisticRegression(penalty=None, max_iter=1000).fit(x, y)
    return {
        "a": float(model.coef_[0, 0]),
        "b": float(model.coef_[0, 1]),
        "c": float(model.intercept_[0]),
    }


def fuse_command(args: argparse.Namespace) -> None:
    """The fusion on the calibration candidates (game 9, natural proportion, band out), frozen in ``fusion.json``."""
    device = args.device or "cuda"
    calibration = pd.read_parquet(args.crops / "calibration.parquet")
    crops = np.load(args.crops / "calibration_crops.npy", mmap_mode="r")
    net = load_net(args.weights, device)
    p, offsets = predict_crops(net, crops, device)
    scored = (calibration.label >= 0).to_numpy()
    y = calibration.label.to_numpy()[scored]
    conf = calibration.conf.to_numpy(float)
    fusion = fit_fusion(p[scored], conf[scored], y)
    from sklearn.linear_model import LogisticRegression

    dd2c_only = LogisticRegression(penalty=None).fit(logit(conf[scored])[:, None], y)
    positive = (calibration.label == 1).to_numpy()
    target = calibration.loc[positive, ["dx", "dy"]].to_numpy(float)
    sides = calibration.side.to_numpy(float)[positive]
    record = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "weights": str(args.weights),
        "weights_sha256": bench.sha256_file(args.weights),
        "fusion": fusion,
        "calibration_rows": int(scored.sum()),
        "calibration_positives": int(y.sum()),
        "band_rows_ignored": int((~scored).sum()),
        "log_loss": {
            "dd2c_conf_raw": log_loss(conf[scored], y),
            "dd2c_conf_recalibrated": log_loss(
                dd2c_only.predict_proba(logit(conf[scored])[:, None])[:, 1], y
            ),
            "crop_raw": log_loss(p[scored], y),
            "fused": log_loss(fused(fusion, p[scored], conf[scored]), y),
        },
        "offset_mae_px": {
            "regressor": float(
                np.abs((offsets[positive] - target) * sides[:, None]).mean()
            ),
            "unmoved": float(np.abs(target * sides[:, None]).mean()),
        },
    }
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    calibration.assign(p_crop=p, off_x=offsets[:, 0], off_y=offsets[:, 1]).to_parquet(
        out / "calibration_scores.parquet", index=False
    )
    (out / "fusion.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


# ------------------------------------------------------------------ the rescored 25 fps dumps of GSR valid (GPU)
VARIANTS = ("rescore", "regress")  # rows (i) and (ii) of decision 6
# + "move": dd2c's confidences with the regressed boxes (the regressor alone, descriptive)
DUMPS = {
    "rescore": {"conf": True, "move": False},
    "regress": {"conf": True, "move": True},
    "move": {"conf": False, "move": True},
}


class SecondStage:
    """dd2c's ball candidates of one frame -> fused confidences and regressed boxes."""

    def __init__(self, weights: Path, fusion: dict, device):
        self.net = load_net(weights, device)
        self.fusion = fusion
        self.device = device

    def __call__(self, image: np.ndarray, boxes: np.ndarray, conf: np.ndarray):
        """``(fused conf, moved boxes, p_crop, offsets)`` of a frame's ball ``boxes`` (native pixels)."""
        scale = image_scale(image.shape[1], image.shape[0])
        centres = be.centres(boxes)
        sides = crop_side(boxes, scale)
        p, offsets = predict_crops(
            self.net, extract(image, centres, sides), self.device
        )
        moved = boxes + np.tile(offsets * sides[:, None], 2)
        return fused(self.fusion, p, conf), moved, p, offsets


def rescore_command(args: argparse.Namespace) -> None:
    """Every ball candidate of ``ball/04/dump25`` (all 58 GSR valid clips, all frames) rescored: ``dump25_rescore``
    (fused conf, boxes kept) and ``dump25_regress`` (fused conf, boxes moved by the offset head), plus the crop
    scores; other classes as in the source dump."""
    import torch

    fusion = json.loads(args.fusion.read_text())
    stage = SecondStage(
        Path(fusion["weights"]), fusion["fusion"], args.device or "cuda"
    )
    clips = [c for c in bt.clip_dirs("valid")]
    out = args.out
    scores_dir = out / "crop_scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    timings = []
    for clip_dir in clips:
        clip = clip_dir.name
        path = scores_dir / f"{clip}.parquet"
        if path.is_file():
            continue
        candidates = bt.clip_candidates(clip)
        balls = candidates[candidates.class_id == 0]
        by_frame = {int(f): g for f, g in balls.groupby("frame")}
        rows = []
        for image_path, image in bt.read_ahead(bt.clip_frames(clip_dir)):
            frame = int(image_path.stem)
            group = by_frame.get(frame)
            if group is None:
                continue
            torch.cuda.synchronize()
            clock = time.perf_counter()
            conf, moved, p, offsets = stage(
                image, group[bt.COLUMNS].to_numpy(float), group.conf.to_numpy(float)
            )
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - clock)
            rows.append(
                pd.DataFrame(
                    {
                        "index": group.index.to_numpy(),
                        "frame": frame,
                        "fused": conf,
                        "p_crop": p,
                        "off_x": offsets[:, 0],
                        "off_y": offsets[:, 1],
                    }
                ).join(pd.DataFrame(moved, columns=[f"m{c}" for c in bt.COLUMNS]))
            )
        pd.concat(rows, ignore_index=True).to_parquet(path, index=False)
        logger.info("%s rescored", clip)
    for variant, change in DUMPS.items():
        if (out / f"dump25_{variant}" / "dump.json").is_file():
            continue
        parts = {}
        for clip_dir in clips:
            clip = clip_dir.name
            scores = pd.read_parquet(scores_dir / f"{clip}.parquet").set_index("index")
            part = bt.clip_candidates(clip)
            if change["conf"]:
                part.loc[scores.index, "conf"] = scores.fused.astype(np.float32)
            if change["move"]:
                for c in bt.COLUMNS:
                    part.loc[scores.index, c] = scores[f"m{c}"].astype(np.float32)
            parts[clip] = part.astype(
                {"conf": np.float32, **dict.fromkeys(bt.COLUMNS, np.float32)}
            )
        write_dump(
            out / f"dump25_{variant}",
            parts,
            {
                "ticket": "ball-detection 13",
                "row": f"1280 + second stage ({variant})",
                "second_stage": {
                    "weights": fusion["weights"],
                    "weights_sha256": fusion["weights_sha256"],
                    "fusion": fusion["fusion"],
                    "crop": CROP,
                    "conf": "fused" if change["conf"] else "dd2c's",
                    "boxes": "moved by the offset head" if change["move"] else "dd2c's",
                },
                "second_stage_ms_per_frame": round(1000 * float(np.mean(timings)), 2)
                if timings
                else None,
            },
        )
    print(
        f"second stage {1000 * np.mean(timings) if timings else float('nan'):.2f} ms per frame with candidates"
    )


def cost_command(args: argparse.Namespace) -> None:
    """dd2c 1280 + the second stage per frame on one GSR valid clip (batch 1, CUDA synchronised, image decode out)."""
    import torch

    fusion = json.loads(args.fusion.read_text())
    device = args.device or "cuda"
    stage = SecondStage(Path(fusion["weights"]), fusion["fusion"], device)
    settings = bench.ROWS[bt.ROW]
    model = tv.load_model(
        "yolo",
        bt.WEIGHTS,
        conf=bt.DUMP_CONF,
        iou=settings["iou"],
        imgsz=settings["imgsz"],
        device=device,
    )
    clip_dir = bt.clip_dirs("valid")[0]
    detector, second = [], []
    for _, image in bt.read_ahead(bt.clip_frames(clip_dir)[: args.frames]):
        torch.cuda.synchronize()
        clock = time.perf_counter()
        found = model.predict(image)
        torch.cuda.synchronize()
        middle = time.perf_counter()
        balls = found[found.class_id == 0]
        if len(balls):
            stage(
                image,
                np.asarray(balls.xyxy, float),
                np.asarray(balls.confidence, float),
            )
        torch.cuda.synchronize()
        detector.append(middle - clock)
        second.append(time.perf_counter() - middle)
    warm = bench.DUMP_WARMUP
    record = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "clip": clip_dir.name,
        "frames": len(detector) - warm,
        "dd2c_ms_per_frame": 1000 * float(np.mean(detector[warm:])),
        "second_stage_ms_per_frame": 1000 * float(np.mean(second[warm:])),
        "total_ms_per_frame": 1000 * float(np.mean(np.add(detector, second)[warm:])),
        "limit_ms": 40.0,
        "gpu": bench.gpu_name(device),
        "note": "batch 1, CUDA synchronised, image decode and the dump write out; the second stage = crops (CPU) + "
        "ResNet18 fp16 + fusion",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cost.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


# ------------------------------------------------------------------ the read (games 2+3, CPU)
def retention(table: pd.DataFrame, column: str, keep: float = 0.99) -> dict:
    """The threshold on ``column`` keeping ``keep`` of the true candidates and the share of false ones still above it."""
    true = table.loc[table.label == 1, column].to_numpy(float)
    false = table.loc[table.label == 0, column].to_numpy(float)
    threshold = float(np.quantile(true, 1 - keep))
    return {
        "threshold": threshold,
        "true_kept": float((true >= threshold).mean()),
        "false_kept": float((false >= threshold).mean()),
        "true": len(true),
        "false": len(false),
    }


def candidate_labels(clips: list[str], folders: dict[str, Path]) -> pd.DataFrame:
    """The post-filtered ball candidates of the benchmark frames of ``clips`` with decision 3's label (against the
    unmoved dd2c box) and each folder's confidence."""
    rows = []
    for clip in clips:
        truth = truth_balls(bench.GSR / "valid" / clip / "Labels-GameState.json")
        base = bt.clip_candidates(clip)
        long = np.maximum(
            base.x2 - base.x1, base.y2 - base.y1
        )  # GSR valid: 1080p frames
        base = base[
            (base.class_id == 0)
            & (long <= bt.POST_FILTER)
            & ((base.frame - 1) % bench.STRIDE == 0)
        ]
        table = base[["frame", "conf"]].rename(columns={"conf": "dd2c"}).copy()
        table["label"] = -1
        for frame, index in base.groupby("frame").groups.items():
            table.loc[index, "label"] = label_of(
                match(
                    base.loc[index, bt.COLUMNS].to_numpy(float), truth.get(int(frame))
                )
            )
        for name, folder in folders.items():
            other = bt.clip_candidates(clip, folder)
            table[name] = other.loc[table.index, "conf"].to_numpy(float)
        table.insert(0, "clip", clip)
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


# Registered 2026-10-26 10:49 (before any read of the second stage on games 2+3): the tracker's two confidence
# thresholds on the fused scale, chosen on the off-benchmark frames of games 2+3 with ticket 04's image metric.
TUNE_GRID = {
    "min_confidence": (0.05, 0.1, 0.2, 0.3, 0.5, 0.7),
    "restart_confidence": (0.15, 0.3, 0.5, 0.7, 0.85, 0.95),
}
SINGLE_GRID = (0.15, 0.3, 0.5, 0.7, 0.85, 0.95)


def tune_settings(base: dict) -> list[dict]:
    """``base`` (ticket 04's setting) with every ``TUNE_GRID`` pair, restart >= min, in grid order."""
    return [
        base | {"min_confidence": m, "restart_confidence": r}
        for m in TUNE_GRID["min_confidence"]
        for r in TUNE_GRID["restart_confidence"]
        if r >= m
    ]


def filtered_top1(clips: list[str], folder: Path, threshold: float) -> pd.DataFrame:
    """Per frame the most confident post-filtered ball candidate >= ``threshold`` of the dump ``folder``."""
    tables = []
    for clip in clips:
        candidates = bt.drop_long_balls(
            bt.clip_candidates(clip, folder), bt.POST_FILTER
        )
        tables.append(bt.top_balls(candidates, 750, threshold).assign(clip=clip))
    return pd.concat(tables, ignore_index=True)


def first_best(table: pd.DataFrame) -> int:
    """The row of the highest hit rate, the first one on a tie."""
    return int(np.flatnonzero(table.hit.to_numpy() == table.hit.max())[0])


def tune_command(args: argparse.Namespace) -> None:
    """``TUNE_GRID`` on the rescored dump (i) and ``SINGLE_GRID`` for the single-frame row, scored on the
    off-benchmark frames of games 2+3 only (ticket 04's centre rule)."""
    base = json.loads(SETTING.read_text())
    clips = bt.games_clips(GAMES)
    folder = args.dumps / "dump25_rescore"
    truth = bt.truth_frames(clips)
    off = truth[~truth.bench].reset_index(drop=True)
    settings = tune_settings(base)
    _, tracked = bt.run_tracks(
        clips, settings, args.workers, folder, "valid", bt.MOTION_DIR, bt.POST_FILTER
    )
    table = pd.DataFrame(
        [
            {
                "min_confidence": s["min_confidence"],
                "restart_confidence": s["restart_confidence"],
            }
            | bt.score_summary(bt.image_scores(t, off))
            for s, t in zip(settings, tracked, strict=True)
        ]
    )
    single = pd.DataFrame(
        [
            {"threshold": t}
            | bt.score_summary(bt.image_scores(filtered_top1(clips, folder, t), off))
            for t in SINGLE_GRID
        ]
    )
    chosen = settings[first_best(table)]
    threshold = float(single.threshold.iloc[first_best(single)])
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "sensitivity.csv", index=False)
    single.to_csv(out / "single.csv", index=False)
    (out / "chosen.json").write_text(json.dumps(chosen, indent=2) + "\n")
    (out / "single_threshold.json").write_text(
        json.dumps({"threshold": threshold}) + "\n"
    )
    text = (
        f"# ball-detection 13: tracker thresholds on the fused scale (games 2+3, off-benchmark frames, "
        f"{len(off)} frames)\n\n{be.markdown(table)}\n\nChosen: {json.dumps(chosen)}\n\n"
        f"{be.markdown(single)}\n\nSingle-frame threshold: {threshold:g}\n"
    )
    (out / "tune.md").write_text(text)
    print(text)


def read_command(args: argparse.Namespace) -> None:
    """Rows (i) rescoring and (ii) rescoring + regressor with the tuned thresholds (``tune``), the council's literal
    row (fusion + ticket 04's setting), all with the 45 px filter, vs phase 1's candidate on games 2+3; (ii) vs (i);
    single-frame rows; false candidates kept at 99% of the true ones."""
    base = json.loads(SETTING.read_text())
    tuned = json.loads((args.tune / "chosen.json").read_text())
    threshold = json.loads((args.tune / "single_threshold.json").read_text())[
        "threshold"
    ]
    clips, homographies = selection_homographies()
    draws = bs.Draws(clips)
    reference = pd.read_parquet(REFERENCE_TRACKS / "tracks.parquet")
    reference_rows = read_rows(reference, homographies)
    dd2c_top1_rows = read_rows(
        pd.read_parquet(REFERENCE_TRACKS / "top1.parquet"), homographies
    )
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    runs = {
        "(i) rescore": ("rescore", tuned),
        "(ii) regress": ("regress", tuned),
        "council literal: rescore + 04 thresholds": ("rescore", base),
        "council literal: regress + 04 thresholds": ("regress", base),
        "regressor alone: dd2c conf + moved boxes, 04 setting": ("move", base),
    }
    tracked, single, rows = {}, {}, {}
    for name, (variant, setting) in runs.items():
        folder = args.dumps / f"dump25_{variant}"
        _, (tracks,) = bt.run_tracks(
            clips,
            [setting],
            args.workers,
            folder,
            "valid",
            bt.MOTION_DIR,
            bt.POST_FILTER,
        )
        if name.startswith("("):
            slug = name.split(" ")[0].strip("()")
        elif name.startswith("council"):
            slug = f"literal_{variant}"
        else:
            slug = "regressor_alone"
        (out / f"track_{slug}").mkdir(parents=True, exist_ok=True)
        tracks.to_parquet(out / f"track_{slug}" / "tracks.parquet", index=False)
        (out / f"track_{slug}" / "setting.json").write_text(
            json.dumps(setting, indent=2) + "\n"
        )
        rows[name] = read_rows(tracks, homographies)
        tracked[f"{name} vs phase 1"] = compare(rows[name], reference_rows, draws)
    for variant in VARIANTS:
        for t in sorted({threshold, 0.15}):
            top1 = filtered_top1(clips, args.dumps / f"dump25_{variant}", t)
            single[f"{variant} top-1 >= {t:g} (post-filtered) vs dd2c top-1"] = compare(
                read_rows(top1, homographies), dd2c_top1_rows, draws
            )
    tracked["(ii) regress vs (i) rescore"] = compare(
        rows["(ii) regress"], rows["(i) rescore"], draws
    )
    regressor_in = bool(
        tracked["(ii) regress vs (i) rescore"]["gain"]["well_placed"]["difference"] > 0
    )
    primary = "(ii) regress vs phase 1" if regressor_in else "(i) rescore vs phase 1"
    labelled = candidate_labels(clips, {"fused": args.dumps / "dump25_rescore"})
    kept = {name: retention(labelled, name) for name in ("dd2c", "fused")}
    record = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "games": list(GAMES),
        "clips": len(clips),
        "setting_04": base,
        "setting_tuned": tuned,
        "single_threshold": threshold,
        "reference_tracks": str(REFERENCE_TRACKS),
        "bootstrap": bs.BOOTSTRAP,
        "regressor_in": regressor_in,
        "primary": primary,
        "primary_wins": tracked[primary]["verdict"]["wins"],
        "tracked": tracked,
        "single_frame": single,
        "retention_99": kept,
    }
    (out / "read.json").write_text(json.dumps(record, indent=2, default=float) + "\n")
    lines = [
        "# ball-detection 13: second stage on crops, games 2+3 of GSR valid",
        "",
        f"Tuned thresholds (fused scale, off-benchmark frames of games 2+3): min_confidence "
        f"{tuned['min_confidence']:g}, restart_confidence {tuned['restart_confidence']:g}; single-frame threshold "
        f"{threshold:g}. Primary row: {primary} (regressor in: {regressor_in}, (ii) > (i) at the point) -> "
        f"{'WINS' if record['primary_wins'] else 'does not win'}.",
        "",
        *comparison_lines(tracked),
        "",
        "Single frame:",
        "",
        *comparison_lines(single),
        "",
        "Ball candidates of the benchmark frames (post-filtered; labels of decision 3, band out): false ones kept at "
        "the threshold keeping 99% of the true ones.",
        "",
        "| confidence | threshold | true kept | false kept | true / false |",
        "|---|---|---|---|---|",
        *(
            f"| {k} | {v['threshold']:.4f} | {100 * v['true_kept']:.1f}% | {100 * v['false_kept']:.1f}% | "
            f"{v['true']} / {v['false']} |"
            for k, v in kept.items()
        ),
        "",
        f"Bootstrap {bs.BOOTSTRAP['draws']} draws over clips, seed {bs.BOOTSTRAP['seed']}.",
        "",
    ]
    text = "\n".join(lines)
    (out / "read.md").write_text(text)
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    o = commands.add_parser(
        "oracle", help="step 0: the oracle rescorer on games 2+3 (CPU)"
    )
    o.add_argument("--out", type=Path, default=OUT / "oracle")
    o.add_argument("--workers", type=int, default=12)
    v = commands.add_parser("view", help="the training / calibration view (CPU)")
    v.add_argument("--out", type=Path, default=OUT / "view")
    c = commands.add_parser(
        "candidates", help="dd2c's ball candidates on the view (GPU)"
    )
    c.add_argument("--view", type=Path, default=OUT / "view/view.parquet")
    c.add_argument("--out", type=Path, default=OUT / "candidates")
    c.add_argument("--chunk", type=int, default=2000)
    c.add_argument("--device", default=None)
    k = commands.add_parser(
        "crops", help="training samples and calibration rows with crops (CPU)"
    )
    k.add_argument("--view", type=Path, default=OUT / "view/view.parquet")
    k.add_argument(
        "--candidates", type=Path, default=OUT / "candidates/candidates.parquet"
    )
    k.add_argument("--out", type=Path, default=OUT / "crops")
    k.add_argument("--workers", type=int, default=12)
    t = commands.add_parser("train", help="the crop model (GPU)")
    t.add_argument("--crops", type=Path, default=OUT / "crops")
    t.add_argument("--out", type=Path, default=OUT / "model")
    f = commands.add_parser("fuse", help="the logistic fusion on the calibration rows")
    f.add_argument("--crops", type=Path, default=OUT / "crops")
    f.add_argument("--weights", type=Path, default=OUT / "model/best.pt")
    f.add_argument("--out", type=Path, default=OUT / "fusion")
    r = commands.add_parser(
        "rescore", help="the rescored 25 fps dumps of GSR valid (GPU)"
    )
    r.add_argument("--fusion", type=Path, default=OUT / "fusion/fusion.json")
    r.add_argument("--out", type=Path, default=OUT)
    q = commands.add_parser("cost", help="dd2c + second stage ms per frame (GPU)")
    q.add_argument("--fusion", type=Path, default=OUT / "fusion/fusion.json")
    q.add_argument("--frames", type=int, default=300)
    q.add_argument("--out", type=Path, default=OUT / "cost")
    e = commands.add_parser(
        "read", help="rows (i) and (ii) vs phase 1 on games 2+3 (CPU)"
    )
    e.add_argument("--dumps", type=Path, default=OUT)
    e.add_argument("--tune", type=Path, default=OUT / "tune")
    e.add_argument("--out", type=Path, default=OUT / "read")
    e.add_argument("--workers", type=int, default=12)
    u = commands.add_parser(
        "tune", help="tracker thresholds on the fused scale (games 2+3, off-benchmark)"
    )
    u.add_argument("--dumps", type=Path, default=OUT)
    u.add_argument("--out", type=Path, default=OUT / "tune")
    u.add_argument("--workers", type=int, default=12)
    for command in (t, f, r, q):
        command.add_argument("--device", default=None)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    tv.setup_logging("WARNING")
    {
        "oracle": oracle,
        "view": view_command,
        "candidates": candidates_command,
        "crops": crops_command,
        "train": train_command,
        "fuse": fuse_command,
        "rescore": rescore_command,
        "cost": cost_command,
        "read": read_command,
        "tune": tune_command,
    }[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
