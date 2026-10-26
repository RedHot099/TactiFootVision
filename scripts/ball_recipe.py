"""ball-detection 14: the repaired fine-tune recipe, arm A (control) and arm B (A + referee hard negatives).

    uv run python scripts/ball_recipe.py negatives   # B's hard negatives: crops around referees in black (CPU)
    uv run python scripts/ball_recipe.py build       # runs/bd14_{A,B}: lists, data.yaml, class masks, val list
    uv run python scripts/detector_train.py train --run bd14_A      # then bd14_B (see runs/bd14_*/RUN.md)
    uv run python scripts/ball_temporal.py dump --weights W --out ball/14/<arm>/dump25    # 25 fps GSR valid (GPU)
    uv run python scripts/ball_scorecard.py card --weights W --out ball/14/<arm>/card     # single frame, guards
    uv run python scripts/ball_recipe.py read        # games 2+3: pipeline rows, single frame, referees, table

Both arms fine-tune dd2c (sha256 05d706cf...) for one epoch on the repaired recipe (`RUNS["bd14_*"]` in
scripts/detector_train.py: 1280, mosaic to the end (`close_mosaic` 0), `warmup_epochs` 0, lr0 0.0005, EMA); the
checkpoint is the last EMA after the epoch (`last.pt`), never chosen by a score.

* **A:** run 2c's training list without game 4 of GSR train (the phase 2 holdout, ``bench.HOLDOUT_FILE``): every
  run 2c line whose image is not a game 4 frame, in run 2c's order (``check_a`` compares it with run 2c's
  `train.txt` filtered the same way).
* **B:** A's list plus one line per hard negative, nothing else. A hard negative is a crop of an image of A's view
  around a referee box (the sources' own referee labels; linesmen are labelled referee) that is
  - **in black:** the median HSV value of the box's torso (``TORSO``: the middle half of its width, 15-50% of its
    height) < ``DARK_V`` and the median saturation < ``DARK_S`` (OpenCV's 0-255 scales; a pixel statistic of an
    existing label, no AI labelling), box height >= ``MIN_HEIGHT`` px at 1080p; or
  - **under a dd2c false ball:** one of dd2c's ball candidates >= ``FP_CONF`` (ticket 13's candidates on its view) is
    a negative under ticket 13's label rule (centre >= 2 r from every truth ball) and has its centre in the box.
  Screened: every image of A's view whose source annotates referees (SoccerNet-v3, Soccana, Roboflow players,
  football_yolo's training clips, GSR train games 6 and 9 at one frame per second, ``GSR_STRIDE``). Per selected box
  ``CROPS_PER_BOX`` crops with the image's aspect ratio, width ``CROP_FRACTION`` of the image's (seeded per box), that
  hold the whole referee box and no part of any truth ball box (``TRIES`` draws, else the crop is skipped). Labels:
  the image's non-ball boxes clipped to the crop, kept when the clipped part has >= ``KEEP_AREA`` of the box's area and
  sides >= 2 px (Ultralytics' own rule for boxes cut by an augmentation); its source's annotated classes. So a crop
  is background for the ball class and a normal image for the people classes.

Validation during training (log only, not used for anything): every 4th benchmark frame of games 2 and 3 of GSR
valid plus every Ekstraklasa dev image (game 5 is not in it).

`read` reads games 2+3 only (ticket 11's read: frozen homographies kept for their clips, box centre placed): both
arms' 25 fps dumps tracked with ticket 04's setting (`ball/04/tune/chosen.json`), the 45 px post-filter and ticket
04's camera motion (``ball_temporal.run_tracks`` unchanged), against each other (the primary B - A), against phase
1's candidate (ticket 09's tracks: the ticket's rule, ``ball_projection.verdict``) and per game; the single-frame rows
(top-1 >= 0.15 of the 25 fps dump against dd2c's top-1); false balls on referees on games 2 and 3 (all annotated
frames); the guards from the cards.
"""

import argparse
import hashlib
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ball_crops as bc  # noqa: E402
import ball_errors as be  # noqa: E402
import ball_projection as bp  # noqa: E402
import ball_scorecard as bs  # noqa: E402
import ball_temporal as bt  # noqa: E402
import detector_bench as bench  # noqa: E402
import detector_train as dt  # noqa: E402

OUT = be.ROOT / "ball/14"
NEGATIVES = OUT / "negatives"
ARMS = ("bd14_A", "bd14_B")
RUN2C_LIST = dt.run_dir("run2c") / "train.txt"
# Referees in black (2026-10-26, set on SoccerNet-v3 training crops before any training or read).
DARK_V, DARK_S = 80.0, 120.0
TORSO = (0.25, 0.75, 0.15, 0.50)  # x from, x to, y from, y to: fractions of the box
MIN_HEIGHT = 20.0  # px at 1080p
FP_CONF = 0.05  # ticket 04's min_confidence: the weakest candidate the tracker uses
FP_CANDIDATES = be.ROOT / "ball/13/candidates/candidates.parquet"
FP_VIEW = be.ROOT / "ball/13/view/view.parquet"
GSR_STRIDE = 25  # GSR train frames screened: one per second (25 fps)
CROPS_PER_BOX = 2
CROP_FRACTION = (0.5, 0.8)
CROP_MARGIN = 1.1  # the crop's side is at least this times the referee box's
TRIES = 50
KEEP_AREA = 0.1
SEED = 0
REFEREE = dt.CLASSES.index("referee")
BALL = dt.CLASSES.index("ball")
JPEG_QUALITY = 95
GAMES = bt.SELECTION

logger = logging.getLogger("ball_recipe")


# ------------------------------------------------------------------ view A
def holdout_names() -> set[str]:
    """Game 4's GSR clip names and their SoccerNet Tracking twins (SNGS-060 ... and SNMOT-060 ...)."""
    clips = set(bench.holdout_clips())
    return clips | {c.replace("SNGS-", "SNMOT-") for c in clips}


def is_holdout(view: pd.DataFrame, names: set[str] | None = None) -> pd.Series:
    """Rows of game 4: the GSR game, a holdout clip, or a holdout clip's name in the image file name."""
    names = holdout_names() if names is None else names
    stems = view.image.map(lambda p: Path(p).name.split("_")[0])
    return (
        view.game.eq(f"gsr:{bench.HOLDOUT['game']}").fillna(False)
        | view["clip"].astype(object).isin(names)
        | stems.isin(names)
    )


def view_a(pool: pd.DataFrame, names: set[str] | None = None) -> pd.DataFrame:
    """Run 2c's view (``detector_train.build_view``) without game 4."""
    view = dt.build_view(pool, "bd14_A")
    return view[~is_holdout(view, names)].sort_values("image").reset_index(drop=True)


def check_a(lines: list[str], run2c: list[str], names: set[str]) -> None:
    """A's list must be run 2c's list without its game 4 lines (same multiset; both sorted)."""
    stems = {line: Path(line).name.split("_")[0] for line in set(run2c)}
    expected = [line for line in run2c if stems[line] not in names]
    if sorted(lines) != sorted(expected):
        raise SystemExit("A's list is not run 2c's list without game 4")


# ------------------------------------------------------------------ negatives: selection
def read_label(image: str) -> np.ndarray:
    """A YOLO label file of ``image`` (class, cx, cy, w, h normalised), (0, 5) when empty."""
    path = Path(bc.label_file(image))
    rows = np.loadtxt(path, ndmin=2) if path.stat().st_size else np.zeros((0, 5))
    return rows.reshape(-1, 5)


def to_xyxy(rows: np.ndarray, width: int, height: int) -> np.ndarray:
    cx, cy, w, h = (rows[:, i] for i in range(1, 5))
    return np.c_[
        (cx - w / 2) * width,
        (cy - h / 2) * height,
        (cx + w / 2) * width,
        (cy + h / 2) * height,
    ]


def torso_colour(hsv: np.ndarray, box: np.ndarray) -> tuple[float, float] | None:
    """Median HSV value and saturation of the box's torso (``TORSO``), None when it has no pixel."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    tx1, tx2 = int(max(x1 + TORSO[0] * w, 0)), int(x1 + TORSO[1] * w)
    ty1, ty2 = int(max(y1 + TORSO[2] * h, 0)), int(y1 + TORSO[3] * h)
    patch = hsv[ty1:ty2, tx1:tx2].reshape(-1, 3)
    if not len(patch):
        return None
    return float(np.median(patch[:, 2])), float(np.median(patch[:, 1]))


def is_dark(value: float, saturation: float, height_1080: float) -> bool:
    return value < DARK_V and saturation < DARK_S and height_1080 >= MIN_HEIGHT


def screened(view: pd.DataFrame) -> pd.DataFrame:
    """The images of A's view screened for referees in black: sources annotating referees, with a referee box; GSR
    train one frame per ``GSR_STRIDE``."""
    annotates = view.annotated_classes.str.split(",").map(lambda a: "referee" in a)
    gsr = view.source.eq("soccernet_gsr")
    keep = annotates & (view.n_referee > 0)
    keep &= ~gsr | ((view.frame.astype("Int64") - 1) % GSR_STRIDE == 0).fillna(False)
    return view[keep]


def referee_boxes(row) -> list[dict]:
    """Every referee box of one image with its torso colour."""
    labels = read_label(row.image)
    image = cv2.imread(row.image)
    if image is None:
        raise FileNotFoundError(row.image)
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    boxes = to_xyxy(labels, width, height)
    out = []
    for index in np.flatnonzero(labels[:, 0].astype(int) == REFEREE):
        colour = torso_colour(hsv, boxes[index])
        if colour is None:
            continue
        height_1080 = (boxes[index, 3] - boxes[index, 1]) / bc.image_scale(
            width, height
        )
        out.append(
            {
                "image": row.image,
                "source": row.source,
                "label_row": int(index),
                **dict(zip(("x1", "y1", "x2", "y2"), boxes[index], strict=True)),
                "width": width,
                "height": height,
                "height_1080": float(height_1080),
                "value": colour[0],
                "saturation": colour[1],
                "dark": is_dark(*colour, height_1080),
            }
        )
    return out


def false_ball_boxes(
    candidates: pd.DataFrame, view: pd.DataFrame, a_images: set[str]
) -> pd.DataFrame:
    """(image, label_row) of the referee boxes holding the centre of a dd2c ball candidate >= ``FP_CONF`` that is a
    negative under ticket 13's label rule (``ball_crops.match`` / ``label_of``), on images of A's view."""
    meta = view.set_index("image")
    rows = []
    strong = candidates[(candidates.conf >= FP_CONF) & candidates.image.isin(a_images)]
    for image, group in strong.groupby("image"):
        m = meta.loc[image]
        labels = read_label(image)
        boxes = to_xyxy(labels, m.width, m.height)
        classes = labels[:, 0].astype(int)
        scale = bc.image_scale(m.width, m.height)
        found = group[["x1", "y1", "x2", "y2"]].to_numpy(float)
        negative = bc.label_of(bc.match(found, boxes[classes == BALL], scale)) == 0
        centres = be.centres(found[negative])
        for index in np.flatnonzero(classes == REFEREE):
            x1, y1, x2, y2 = boxes[index]
            inside = (
                (centres[:, 0] >= x1)
                & (centres[:, 0] <= x2)
                & (centres[:, 1] >= y1)
                & (centres[:, 1] <= y2)
            )
            if inside.any():
                rows.append({"image": image, "label_row": int(index)})
    return pd.DataFrame(rows, columns=["image", "label_row"])


# ------------------------------------------------------------------ negatives: crops
def box_seed(image: str, label_row: int, k: int) -> list[int]:
    digest = hashlib.sha256(f"{image}|{label_row}|{k}".encode()).digest()
    return [SEED, int.from_bytes(digest[:8], "little")]


def crop_window(
    box: np.ndarray,
    balls: np.ndarray,
    width: int,
    height: int,
    rng: np.random.Generator,
) -> tuple[int, int, int, int] | None:
    """A crop (x0, y0, w, h) with the image's aspect ratio and width ``CROP_FRACTION`` of the image's, raised to
    ``CROP_MARGIN`` times the box, that holds the whole box and touches no ball box; None after ``TRIES`` draws."""
    x1, y1, x2, y2 = box
    need = max(CROP_MARGIN * (x2 - x1) / width, CROP_MARGIN * (y2 - y1) / height)
    for _ in range(TRIES):
        fraction = max(rng.uniform(*CROP_FRACTION), need)
        if fraction > 1:
            return None
        w, h = int(round(fraction * width)), int(round(fraction * height))
        low_x, high_x = max(0, int(np.ceil(x2)) - w), min(int(np.floor(x1)), width - w)
        low_y, high_y = max(0, int(np.ceil(y2)) - h), min(int(np.floor(y1)), height - h)
        if low_x > high_x or low_y > high_y:
            continue
        x0 = int(rng.integers(low_x, high_x + 1))
        y0 = int(rng.integers(low_y, high_y + 1))
        if len(balls):
            touch = (
                (balls[:, 0] < x0 + w)
                & (balls[:, 2] > x0)
                & (balls[:, 1] < y0 + h)
                & (balls[:, 3] > y0)
            )
            if touch.any():
                continue
        return x0, y0, w, h
    return None


def crop_labels(
    classes: np.ndarray, boxes: np.ndarray, window: tuple[int, int, int, int]
) -> np.ndarray:
    """YOLO rows of the non-ball boxes clipped to ``window``: kept when the clipped part has >= ``KEEP_AREA`` of the
    box's area and both sides >= 2 px."""
    x0, y0, w, h = window
    keep = classes != BALL
    classes, boxes = classes[keep], boxes[keep]
    clipped = np.c_[
        np.clip(boxes[:, 0] - x0, 0, w),
        np.clip(boxes[:, 1] - y0, 0, h),
        np.clip(boxes[:, 2] - x0, 0, w),
        np.clip(boxes[:, 3] - y0, 0, h),
    ]
    cw, ch = clipped[:, 2] - clipped[:, 0], clipped[:, 3] - clipped[:, 1]
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    ok = (cw >= 2) & (ch >= 2) & (cw * ch >= KEEP_AREA * np.maximum(area, 1e-9))
    clipped, classes, cw, ch = clipped[ok], classes[ok], cw[ok], ch[ok]
    return np.c_[
        classes,
        (clipped[:, 0] + cw / 2) / w,
        (clipped[:, 1] + ch / 2) / h,
        cw / w,
        ch / h,
    ]


def crop_name(image: str, label_row: int, k: int, source: str) -> str:
    key = hashlib.sha256(image.encode()).hexdigest()[:16]
    return f"{source}_{key}_r{label_row}_k{k}"


def make_crops(group: pd.DataFrame, folder: Path) -> list[dict]:
    """The crops of the selected referee boxes of one image (``group``), written under ``folder``."""
    image_path = group.image.iloc[0]
    labels = read_label(image_path)
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    classes = labels[:, 0].astype(int)
    boxes = to_xyxy(labels, width, height)
    balls = boxes[classes == BALL]
    out = []
    for row in group.itertuples():
        box = boxes[row.label_row]
        for k in range(CROPS_PER_BOX):
            rng = np.random.default_rng(box_seed(image_path, row.label_row, k))
            window = crop_window(box, balls, width, height, rng)
            name = crop_name(image_path, row.label_row, k, row.source)
            record = {
                "image": image_path,
                "source": row.source,
                "label_row": row.label_row,
                "k": k,
                "crop": None,
            }
            if window is None:
                out.append(record)
                continue
            x0, y0, w, h = window
            rows = crop_labels(classes, boxes, window)
            crop = folder / "images" / f"{name}.jpg"
            cv2.imwrite(
                str(crop),
                image[y0 : y0 + h, x0 : x0 + w],
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )
            lines = [
                f"{int(c)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                for c, cx, cy, bw, bh in rows
            ]
            (folder / "labels" / f"{name}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else "")
            )
            out.append(
                record
                | {
                    "crop": str(crop),
                    "x0": x0,
                    "y0": y0,
                    "w": w,
                    "h": h,
                    "boxes": len(rows),
                    "referees": int((rows[:, 0] == REFEREE).sum()),
                }
            )
    return out


def negatives(args: argparse.Namespace) -> None:
    pool = pd.read_parquet(dt.POOL)
    view = view_a(pool)
    a_images = set(view.image)
    screen = screened(view)
    logger.info("screening %d images for referees in black", len(screen))
    with ThreadPoolExecutor(args.workers) as workers:
        found = [
            b for bs_ in workers.map(referee_boxes, screen.itertuples()) for b in bs_
        ]
    boxes = pd.DataFrame(found)
    fp = false_ball_boxes(
        pd.read_parquet(FP_CANDIDATES), pd.read_parquet(FP_VIEW), a_images
    )
    fp_extra = fp.merge(boxes[["image", "label_row"]], how="left", indicator=True)
    missing = fp_extra[fp_extra._merge == "left_only"]
    if len(
        missing
    ):  # referee boxes under a false ball on images outside the screen (e.g. GSR off the stride)
        meta = view.set_index("image")
        rows = [
            meta.loc[image].to_dict() | {"image": image}
            for image in sorted(set(missing.image))
        ]
        with ThreadPoolExecutor(args.workers) as workers:
            more = [
                b
                for bs_ in workers.map(referee_boxes, pd.DataFrame(rows).itertuples())
                for b in bs_
            ]
        boxes = pd.concat([boxes, pd.DataFrame(more)], ignore_index=True)
    boxes = boxes.drop_duplicates(["image", "label_row"]).reset_index(drop=True)
    keys = set(zip(fp.image, fp.label_row, strict=True))
    boxes["false_ball"] = [
        (i, r) in keys for i, r in zip(boxes.image, boxes.label_row, strict=True)
    ]
    boxes["selected"] = boxes.dark | boxes.false_ball
    selected = boxes[boxes.selected].sort_values(["image", "label_row"])
    folder = args.out
    for sub in ("images", "labels"):
        (folder / sub).mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(args.workers) as workers:
        crops = [
            c
            for cs in workers.map(
                lambda g: make_crops(g[1], folder), selected.groupby("image")
            )
            for c in cs
        ]
    table = pd.DataFrame(crops)
    table = table.merge(
        boxes[["image", "label_row", "dark", "false_ball", "value", "saturation"]],
        on=["image", "label_row"],
    )
    boxes.to_parquet(folder / "referee_boxes.parquet", index=False)
    table.to_parquet(folder / "negatives.parquet", index=False)
    made = table[table.crop.notna()]
    record = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "ticket": "ball-detection 14",
        "rule": {
            "dark": f"torso {TORSO} median HSV value < {DARK_V}, saturation < {DARK_S}, height >= {MIN_HEIGHT} px "
            "at 1080p",
            "false_ball": f"a dd2c ball candidate >= {FP_CONF} of {FP_CANDIDATES} that is a negative under ticket "
            "13's label rule has its centre in the referee box",
            "screened": f"A's view images whose source annotates referees, with a referee box; GSR train every "
            f"{GSR_STRIDE}th frame (games 6 and 9)",
            "crop": f"{CROPS_PER_BOX} per box, image aspect, width fraction U{CROP_FRACTION} (>= {CROP_MARGIN} x "
            f"box), whole box in, no ball box touched, {TRIES} draws, seed {SEED} + sha256(image|row|k)",
            "labels": f"non-ball boxes clipped to the crop, kept with >= {KEEP_AREA} of their area and sides >= 2 px",
        },
        "screened_images": int(len(screen)),
        "referee_boxes": int(len(boxes)),
        "selected_boxes": {
            "all": int(boxes.selected.sum()),
            "dark": int(boxes.dark.sum()),
            "false_ball": int(boxes.false_ball.sum()),
            "both": int((boxes.dark & boxes.false_ball).sum()),
        },
        "crops": int(len(made)),
        "crops_skipped": int(table.crop.isna().sum()),
        "per_source": {
            source: {
                "selected_boxes": int(part.selected.sum()),
                "dark": int(part.dark.sum()),
                "false_ball": int(part.false_ball.sum()),
                "source_images": int(part[part.selected].image.nunique()),
                "crops": int((made.source == source).sum()),
            }
            for source, part in boxes.groupby("source")
        },
    }
    (folder / "negatives.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


# ------------------------------------------------------------------ build
def val_list() -> list[str]:
    """Every 4th benchmark frame of games 2 and 3 (sorted, like run 2c's every 4th) plus every Ekstraklasa dev image."""
    clips = set(bt.games_clips(GAMES))
    frames = sorted(
        str(p)
        for p in dt.BENCH_VALID.iterdir()
        if p.suffix == ".jpg" and p.name.split("_")[0] in clips
    )
    extra = sorted(str(p) for p in dt.EKSTRAKLASA_DEV.iterdir() if p.suffix == ".jpg")
    return frames[:: dt.VAL_EVERY] + extra


def check_view(
    images: list[str], names: set[str], splits: tuple[str, ...] = ("valid", "test")
) -> None:
    """Raises when a listed image is a game 4 frame or a frame of the GSR ``splits`` (by clip name in the file
    name)."""
    banned = set(names)
    for split in splits:
        banned |= {p.parent.name for p in bench.split_label_files(split)}
    banned |= {b.replace("SNGS-", "SNMOT-") for b in banned}
    bad = {Path(i).name.split("_")[0] for i in images} & banned
    if bad:
        raise SystemExit(f"excluded clips in the list: {sorted(bad)[:5]}")


def arm_lists(a_lines: list[str], crops: list[str]) -> dict[str, list[str]]:
    return {"bd14_A": sorted(a_lines), "bd14_B": sorted(a_lines + crops)}


def write_arm(
    run: str, lines: list[str], classes: pd.DataFrame, val: list[str], extra: dict
) -> dict:
    spec, out = dt.RUNS[run], dt.run_dir(run)
    (out / "smoke_data").mkdir(parents=True, exist_ok=True)
    label_check = {
        "train": dt.check_label_paths(sorted(set(lines))),
        "val": dt.check_label_paths(val),
    }
    (out / "train.txt").write_text("\n".join(lines) + "\n")
    (out / "val.txt").write_text("\n".join(val) + "\n")
    dt.write_yaml(out / "data.yaml", "train.txt", "val.txt")
    classes.reset_index(drop=True).to_parquet(
        out / "annotated_classes.parquet", index=False
    )
    rng = np.random.default_rng(dt.SMOKE["seed"])
    pick = sorted(
        rng.choice(lines, min(dt.SMOKE["train_images"], len(lines)), replace=False)
    )
    (out / "smoke_data/train.txt").write_text("\n".join(pick) + "\n")
    (out / "smoke_data/val.txt").write_text(
        "\n".join(val[:: dt.SMOKE["val_every"] // dt.VAL_EVERY]) + "\n"
    )
    dt.write_yaml(out / "smoke_data/data.yaml", "train.txt", "val.txt")
    per_source = classes.assign(
        lines=classes.image.map(pd.Series(lines).value_counts())
    )
    record = {
        "title": spec["title"],
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "builder": "scripts/ball_recipe.py",
        "pool_index": str(dt.POOL),
        "rule": {
            "base_view": "run 2c's view (detector_train.build_view) without game 4 of GSR train "
            f"({bench.HOLDOUT_FILE.name}: clips, SNMOT twins)",
            "excluded_games": sorted(dt.BENCH_GAMES) + [f"gsr:{bench.HOLDOUT['game']}"],
            "held_out_splits": spec["holdout"],
            "oversample_by_set": spec["oversample"],
            "class_mask": "per image, the source's annotated_classes; classes outside it get no cls loss",
        }
        | extra,
        "init": str(spec["init"]),
        "init_sha256": bench.sha256_file(Path(spec["init"])),
        "after": spec["after"],
        "settings": spec["settings"],
        "train": {
            "images": int(len(classes)),
            "lines_per_epoch": len(lines),
            "list_sha256": dt.sha256_lines(lines),
            "per_source_lines": {
                s: int(v) for s, v in per_source.groupby("source").lines.sum().items()
            },
        },
        "val": {
            "rule": f"every {dt.VAL_EVERY}th GSR valid benchmark frame of games 2 and 3, then every Ekstraklasa dev "
            "image (training log only)",
            "images": len(val),
            "list_sha256": dt.sha256_lines(val),
        },
        "label_paths": label_check,
        "files": {
            "data_yaml": str(out / "data.yaml"),
            "annotated_classes": str(out / "annotated_classes.parquet"),
            "smoke_data_yaml": str(out / "smoke_data/data.yaml"),
        },
    }
    dt.view_json(run).write_text(json.dumps(record, indent=2, default=str) + "\n")
    return record


def build(args: argparse.Namespace) -> None:
    pool = pd.read_parquet(dt.POOL)
    names = holdout_names()
    view = view_a(pool, names)
    a_lines = dt.train_list(view, dt.RUNS["bd14_A"]["oversample"])
    check_a(a_lines, RUN2C_LIST.read_text().split(), names)
    table = pd.read_parquet(args.negatives / "negatives.parquet")
    made = table[table.crop.notna()]
    if not set(made.image) <= set(view.image):
        raise SystemExit("a negative comes from an image outside A's view")
    sources = view.set_index("image").annotated_classes
    crops = sorted(made.crop)
    lists = arm_lists(a_lines, crops)
    for lines in lists.values():
        check_view(lines, names)
    val = val_list()
    check_view(val, names, ("test",))
    selection = set(bt.games_clips(GAMES))
    gsr_val = {
        Path(v).name.split("_")[0] for v in val if Path(v).name.startswith("SNGS-")
    }
    if not gsr_val <= selection:
        raise SystemExit("a validation frame outside games 2 and 3")
    base = view[["image", "source", "annotated_classes"]]
    crop_rows = pd.DataFrame(
        {
            "image": made.crop.to_numpy(),
            "source": made.source.to_numpy(),
            "annotated_classes": made.image.map(sources).to_numpy(),
        }
    )
    negatives_record = json.loads((args.negatives / "negatives.json").read_text())
    for run, lines in lists.items():
        classes = base if run == "bd14_A" else pd.concat([base, crop_rows])
        extra = {"negatives": "none"}
        if run == "bd14_B":
            extra = {
                "negatives": negatives_record["rule"]
                | {
                    "folder": str(args.negatives),
                    "crops": len(crops),
                    "selected_boxes": negatives_record["selected_boxes"],
                    "per_source": negatives_record["per_source"],
                }
            }
        record = write_arm(run, lines, classes, val, extra)
        logger.info(
            "%s: %d lines, sha %s",
            run,
            record["train"]["lines_per_epoch"],
            record["train"]["list_sha256"][:12],
        )
    a = (dt.run_dir("bd14_A") / "train.txt").read_text().split()
    b = (dt.run_dir("bd14_B") / "train.txt").read_text().split()
    difference = (
        pd.Series(b).value_counts().sub(pd.Series(a).value_counts(), fill_value=0)
    )
    if not (
        difference.ge(0).all() and set(difference[difference > 0].index) == set(crops)
    ):
        raise SystemExit("B is not A plus the negatives")


# ------------------------------------------------------------------ read: referees
def referee_false_balls(
    candidates: pd.DataFrame, clip: str, conf: float = bs.BALL_CONF
) -> pd.DataFrame:
    """Per annotated frame of a GSR valid clip: ball candidates >= ``conf`` that are false (centre farther than the
    centre rule's r = max(5 px, diameter) from every truth ball) with their centre in a referee box: ``on_referee``
    (count), and ``false`` (all false balls)."""
    balls, people = be.clip_labels(bench.GSR / "valid" / clip / "Labels-GameState.json")
    truth = (
        {
            int(f): np.array(g.box.tolist(), float).reshape(-1, 4)
            for f, g in balls.groupby("frame")
        }
        if len(balls)
        else {}
    )
    found = candidates[(candidates.class_id == BALL) & (candidates.conf >= conf)]
    by_frame = dict(tuple(found.groupby("frame")))
    rows = []
    for frame, (boxes, _, roles) in sorted(people.items()):
        part = by_frame.get(frame)
        if part is None or not len(part):
            rows.append({"clip": clip, "frame": frame, "false": 0, "on_referee": 0})
            continue
        xy = be.centres(part[bt.COLUMNS].to_numpy(float))
        true = truth.get(frame)
        if true is not None and len(true):
            gap = np.linalg.norm(xy[:, None] - be.centres(true)[None], axis=2)
            false = ~(gap <= be.radius(true)[None]).any(axis=1)
        else:
            false = np.ones(len(xy), bool)
        refs = boxes[roles == "referee"]
        inside = (
            (
                (xy[:, None, 0] >= refs[None, :, 0])
                & (xy[:, None, 0] <= refs[None, :, 2])
                & (xy[:, None, 1] >= refs[None, :, 1])
                & (xy[:, None, 1] <= refs[None, :, 3])
            ).any(axis=1)
            if len(refs)
            else np.zeros(len(xy), bool)
        )
        rows.append(
            {
                "clip": clip,
                "frame": frame,
                "false": int(false.sum()),
                "on_referee": int((false & inside).sum()),
            }
        )
    return pd.DataFrame(rows)


def tracked_as_candidates(tracks: pd.DataFrame, clip: str) -> pd.DataFrame:
    """The chosen balls of ``tracks`` (measured and interpolated) as ball candidates at conf 1."""
    part = tracks[(tracks["clip"] == clip) & (tracks.status != "none")]
    return part[["frame", *bt.COLUMNS]].assign(class_id=BALL, conf=1.0)


def referee_rows(dumps: dict[str, Path], tracks: dict[str, pd.DataFrame]) -> list[dict]:
    """False balls on referees per game (2, 3) and detector: raw candidates >= 0.15, after the 45 px filter, and the
    tracked ball (per annotated frame)."""
    out = []
    for game in GAMES:
        clips = bt.games_clips((game,))
        for name, folder in dumps.items():
            tables = {"raw >= 0.15": [], "filtered >= 0.15": [], "tracked": []}
            for clip in clips:
                candidates = bt.clip_candidates(clip, folder)
                tables["raw >= 0.15"].append(referee_false_balls(candidates, clip))
                tables["filtered >= 0.15"].append(
                    referee_false_balls(
                        bt.drop_long_balls(candidates, bt.POST_FILTER), clip
                    )
                )
                tables["tracked"].append(
                    referee_false_balls(tracked_as_candidates(tracks[name], clip), clip)
                )
            for kind, parts in tables.items():
                table = pd.concat(parts, ignore_index=True)
                out.append(
                    {
                        "game": game,
                        "detector": name,
                        "balls": kind,
                        "frames": len(table),
                        "on_referee_per_frame": float(table.on_referee.mean()),
                        "frames_with_on_referee": float((table.on_referee > 0).mean()),
                        "false_per_frame": float(table["false"].mean()),
                    }
                )
    return out


# ------------------------------------------------------------------ read
def selection_rows(
    tracks: pd.DataFrame, homographies: dict, clips: list[str]
) -> pd.DataFrame:
    rows = bc.read_rows(tracks, homographies)
    return rows[rows["clip"].isin(clips)].reset_index(drop=True)


def comparisons(
    rows: dict[str, pd.DataFrame], pairs: dict[str, tuple[str, str]], clips: list[str]
) -> dict:
    """``bc.compare`` of each (variant, reference) pair over ``clips`` (the rows of other clips dropped)."""
    draws = bs.Draws(clips)
    out = {}
    for name, (variant, reference) in pairs.items():
        keep = rows[variant]["clip"].isin(clips).to_numpy()
        keep_r = rows[reference]["clip"].isin(clips).to_numpy()
        out[name] = bc.compare(
            rows[variant][keep].reset_index(drop=True),
            rows[reference][keep_r].reset_index(drop=True),
            draws,
        )
    return out


def card_guards(card: Path) -> dict:
    """The guards (vs the card's reference) and the single-frame row of a card (`ball_scorecard.py card`)."""
    found = json.loads((card / "scorecard.json").read_text())
    selection = found["cards"]["selection/1280"]
    eks = found["cards"]["ekstraklasa/1280"]
    return {
        "card": str(card),
        "weights_sha256": found["weights_sha256"],
        "reference": found["reference"],
        "people_map50_95": selection["detection"]["people_map50_95"]["value"],
        "people_map50_95_diff": selection["detection_vs_reference"]["people_map50_95"][
            "difference"
        ],
        "eks_map50_95": eks["detection"]["map50_95"]["value"],
        "eks_map50_95_diff": eks["detection_vs_reference"]["map50_95"]["difference"],
        "eks_ball_ap50": eks["detection"]["ball_ap50"]["value"],
        "ball_ap50": selection["detection"]["ball_ap50"]["value"],
        "ball_ap50_diff": selection["detection_vs_reference"]["ball_ap50"][
            "difference"
        ],
        "false_balls_per_frame": selection["hits"]["false_balls_per_frame"]["value"],
        "well_placed": selection["placement"]["well_placed"],
        "well_placed_vs_reference": selection["placement_vs_reference"]["well_placed"],
        "guard_people": bool(found["rule"]["guard_people"]),
        "guard_ekstraklasa": bool(found["rule"]["guard_ekstraklasa"]),
        "ms_per_frame": found["cost"]["ms_per_frame"].get("gsr/1280"),
    }


def read(args: argparse.Namespace) -> None:
    setting = json.loads(bc.SETTING.read_text())
    clips, homographies = bc.selection_homographies()
    dumps = {"dd2c": bt.DUMP} | {arm: args.root / arm / "dump25" for arm in ("A", "B")}
    tracks = {"dd2c": pd.read_parquet(bc.REFERENCE_TRACKS / "tracks.parquet")}
    top1 = {"dd2c": pd.read_parquet(bc.REFERENCE_TRACKS / "top1.parquet")}
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    for arm in ("A", "B"):
        top, (found,) = bt.run_tracks(
            clips,
            [setting],
            args.workers,
            dumps[arm],
            "valid",
            bt.MOTION_DIR,
            bt.POST_FILTER,
        )
        (out / f"track_{arm}").mkdir(parents=True, exist_ok=True)
        found.to_parquet(out / f"track_{arm}/tracks.parquet", index=False)
        top.to_parquet(out / f"track_{arm}/top1.parquet", index=False)
        tracks[arm], top1[arm] = found, top
    rows = {
        f"{k} pipeline": selection_rows(v, homographies, clips)
        for k, v in tracks.items()
    }
    rows |= {
        f"{k} top-1": selection_rows(v, homographies, clips) for k, v in top1.items()
    }
    pipeline = {
        "B - A (primary)": ("B pipeline", "A pipeline"),
        "A - phase 1": ("A pipeline", "dd2c pipeline"),
        "B - phase 1": ("B pipeline", "dd2c pipeline"),
    }
    single = {
        "B - A single frame": ("B top-1", "A top-1"),
        "A - dd2c top-1": ("A top-1", "dd2c top-1"),
        "B - dd2c top-1": ("B top-1", "dd2c top-1"),
    }
    found = {
        "games 2+3": comparisons(rows, pipeline | single, clips),
    } | {
        f"game {g}": comparisons(rows, pipeline | single, bt.games_clips((g,)))
        for g in GAMES
    }
    primary = found["games 2+3"]["B - A (primary)"]
    b_better = bool(primary["gain"]["well_placed"]["ci"][0] > 0)
    arm = "B" if b_better else "A"
    winner = found["games 2+3"][f"{arm} - phase 1"]
    guards = {a: card_guards(args.root / a / "card") for a in ("A", "B")}
    b_vs_a_card = args.root / "B_vs_A_card"
    if (b_vs_a_card / "scorecard.json").is_file():
        guards["B vs A"] = card_guards(b_vs_a_card)
    guard_ok = guards[arm]["guard_people"] and guards[arm]["guard_ekstraklasa"]
    referees = referee_rows(dumps, tracks)
    verdict = {
        "primary": "B - A, well placed, games 2+3, pipeline (45 px filter + track_ball 04, box centre)",
        "primary_difference": primary["gain"]["well_placed"]["difference"],
        "primary_ci": primary["gain"]["well_placed"]["ci"],
        "b_better": b_better,
        "recipe_for_15": arm,
        "winning_arm": arm,
        "winning_arm_vs_phase1": winner["gain"]["well_placed"],
        "winning_arm_rule_11": winner["verdict"],
        "guards_vs_dd2c": {
            "people": guards[arm]["guard_people"],
            "ekstraklasa": guards[arm]["guard_ekstraklasa"],
        },
        "ticket_wins": bool(winner["verdict"]["wins"] and guard_ok),
    }
    record = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "games": list(GAMES),
        "clips": len(clips),
        "setting": setting,
        "reference_tracks": str(bc.REFERENCE_TRACKS),
        "dumps": {k: str(v) for k, v in dumps.items()},
        "bootstrap": bs.BOOTSTRAP,
        "verdict": verdict,
        "comparisons": found,
        "guards": guards,
        "referees": referees,
    }
    (out / "read.json").write_text(json.dumps(record, indent=2, default=float) + "\n")
    table_rows(found).to_csv(out / "table.csv", index=False)
    pd.DataFrame(referees).to_csv(out / "referees.csv", index=False)
    text = markdown(record)
    (out / "read.md").write_text(text)
    print(text)


def table_rows(found: dict) -> pd.DataFrame:
    """One row per (group, comparison): well placed of both sides, the paired gain and > 5 m of all, with CIs."""
    rows = []
    for group, comparisons_ in found.items():
        for name, c in comparisons_.items():
            gain, far = c["gain"]["well_placed"], c["gain"]["far_of_all"]
            rows.append(
                {
                    "group": group,
                    "comparison": name,
                    "reference_well_placed": c["reference"]["well_placed"]["value"],
                    "variant_well_placed": c["variant"]["well_placed"]["value"],
                    "gain_pp": 100 * gain["difference"],
                    "gain_ci_low_pp": 100 * gain["ci"][0],
                    "gain_ci_high_pp": 100 * gain["ci"][1],
                    "far_of_all_pp": 100 * far["difference"],
                    "far_ci_low_pp": 100 * far["ci"][0],
                    "far_ci_high_pp": 100 * far["ci"][1],
                    "barrier": c["verdict"]["barrier"],
                    "wins_rule_11": c["verdict"]["wins"],
                }
            )
    return pd.DataFrame(rows)


def markdown(record: dict) -> str:
    v = record["verdict"]
    low, high = (100 * x for x in v["primary_ci"])
    lines = [
        "# ball-detection 14: repaired fine-tune, arms A and B (games 2+3 of GSR valid)",
        "",
        f"Primary B - A (pipeline): {100 * v['primary_difference']:+.2f} pp [{low:+.2f}; {high:+.2f}] -> "
        f"B better: {v['b_better']}; recipe for 15: **{v['recipe_for_15']}**. Winning arm vs phase 1: "
        f"{100 * v['winning_arm_vs_phase1']['difference']:+.2f} pp "
        f"[{100 * v['winning_arm_vs_phase1']['ci'][0]:+.2f}; {100 * v['winning_arm_vs_phase1']['ci'][1]:+.2f}], "
        f"rule 11: {v['winning_arm_rule_11']}, guards vs dd2c: {v['guards_vs_dd2c']} -> "
        f"**ticket {'WINS' if v['ticket_wins'] else 'does not win'}**.",
        "",
    ]
    for group, found in record["comparisons"].items():
        lines += [f"## {group}", "", *bc.comparison_lines(found), ""]
    lines += [
        "## Guards and cards (row 1280, games 2+3; Ekstraklasa dev)",
        "",
        "| card | people mAP50-95 (diff) | Eks. mAP50-95 (diff) | Eks. ball AP50 | ball AP50 (diff) | false balls / "
        "frame | single frame well placed vs reference, pp | guards | ms/frame |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, g in record["guards"].items():
        w = g["well_placed_vs_reference"]
        lines.append(
            f"| {name} | {g['people_map50_95']:.4f} ({g['people_map50_95_diff']:+.4f}) | {g['eks_map50_95']:.4f} "
            f"({g['eks_map50_95_diff']:+.4f}) | {g['eks_ball_ap50']:.3f} | {g['ball_ap50']:.3f} "
            f"({g['ball_ap50_diff']:+.3f}) | {g['false_balls_per_frame']:.3f} | {100 * w['difference']:+.2f} "
            f"[{100 * w['ci'][0]:+.2f}; {100 * w['ci'][1]:+.2f}] | people {g['guard_people']}, Eks. "
            f"{g['guard_ekstraklasa']} | {g['ms_per_frame']} |"
        )
    lines += [
        "",
        "## False balls on referees (every annotated frame, 25 fps)",
        "",
        "| game | detector | balls | frames | on referee / frame | frames with one | false / frame |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in record["referees"]:
        lines.append(
            f"| {r['game']} | {r['detector']} | {r['balls']} | {r['frames']} | {r['on_referee_per_frame']:.4f} | "
            f"{100 * r['frames_with_on_referee']:.2f}% | {r['false_per_frame']:.3f} |"
        )
    lines += [
        "",
        f"Bootstrap {bs.BOOTSTRAP['draws']} draws over clips, seed {bs.BOOTSTRAP['seed']}. Placement: box centre, "
        "frozen homographies (games 2+3 only). Barrier: > 5 m of all denominator frames up by <= "
        f"{bp.BARRIER_PP:g} pp.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(required=True)
    n = commands.add_parser("negatives", help="B's hard negatives (CPU)")
    n.add_argument("--out", type=Path, default=NEGATIVES)
    n.add_argument("--workers", type=int, default=24)
    n.set_defaults(handler=negatives)
    b = commands.add_parser("build", help="the lists of both arms")
    b.add_argument("--negatives", type=Path, default=NEGATIVES)
    b.set_defaults(handler=build)
    r = commands.add_parser("read", help="games 2+3: pipeline, single frame, referees")
    r.add_argument("--root", type=Path, default=OUT)
    r.add_argument("--out", type=Path, default=OUT / "read")
    r.add_argument("--workers", type=int, default=12)
    r.set_defaults(handler=read)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args.handler(args)


if __name__ == "__main__":
    main()
