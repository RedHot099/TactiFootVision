"""ball-detection 03: dd2c's ball without training: input size, SAHI-style tiles, flip TTA, the 45 px post-filter.

    uv run python scripts/ball_inference.py dump --variant 1920 --games all --out DIR       # one variant's dump (GPU)
    uv run python scripts/ball_inference.py dump --variant tiles3x2 [--augment] --out DIR   # games 2+3 by default
    uv run python scripts/ball_inference.py postfilter --dump DUMP --row 1280 --out DIR     # drop long ball boxes
    uv run python scripts/ball_inference.py table --card NAME=CARD_DIR[@ROW] ... --out DIR      # the ticket's table

`dump` writes a prediction dump in the format of `detector_bench.py dump` (`candidates.parquet`, `images.parquet`,
`dump.json`: every candidate of every class at conf >= 0.001, boxes in the images' pixels, rows, `max_det`, classes
and their mapping, GPU, ms per frame) with one row, the variant, so that `ball_scorecard.py card --dump` scores it.
Variants (ticket 03 registration, 2026-10-24 21:27), all with dd2c's weights unchanged:

- `1600`, `1920`: one model call on the native frame at that imgsz, NMS IoU 0.7 (1920 on a 1080p frame: no resize);
- `tiles2x2`, `tiles3x2`: the frame cut in columns x rows tiles at native resolution (`tile_grid`: overlap >=
  `MIN_OVERLAP_1080` px at 1080p, tile sides rounded up to the stride, imgsz = the tile's long side so a tile is not
  resized), plus the full frame in the 1280 row. Balls only are merged (`merge_tiles`): a tile's ball touching an
  inner tile edge is dropped (the overlap puts that ball whole in the neighbouring tile), the rest are shifted to the
  frame, joined with the full frame's balls and reduced by greedy NMS on IoS (keep the most confident, drop every
  other ball with intersection over the smaller box >= `MERGE_IOS`); people come from the full frame alone;
- `--augment`: Ultralytics' test-time augmentation (`augment=True`: scales 1 / 0.83 / 0.67, left-right flip at
  0.83) at every model call of the variant; the row is named `<variant>+tta`.

ms per frame is the wall time of all model calls of a frame (tiles, full frame) and the merge, batch 1, image read
excluded, the first `bench.DUMP_WARMUP` images of each set left out.

`postfilter` writes a dump without the ball candidates whose long side, in 1080p pixels (long side x 1080 / image
height), exceeds `RULE["post_filter_long_side_px"]` (45 px, from the truth only: ticket 02); people unchanged; the
row is renamed `<row>+pf45`. `table` collects scorecards (`ball_scorecard.py card --out`) into the ticket's table and
decision 9's question: how much of the gain over dd2c 1280 falls on balls < 12 px.
"""

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ball_errors as be  # noqa: E402
import ball_scorecard as bs  # noqa: E402
import detector_bench as bench  # noqa: E402

import tactifoot_vision as tv  # noqa: E402

OUT = be.ROOT / "ball/03"
REFERENCE = be.ROOT / "ball/02/dd2c"  # dd2c's dump and card (rows 1280, e3), ticket 02
WEIGHTS = bench.CHECKOUT / "models/detector_yolo11l_1280_dd2c.pt"
NMS_IOU = 0.7
SINGLE = {"1600": 1600, "1920": 1920}  # one call on the native frame at this imgsz
TILES = {"tiles2x2": (2, 2), "tiles3x2": (3, 2)}  # columns x rows
FULL_FRAME = bench.ROWS["1280"]
MIN_OVERLAP_1080 = (
    96  # px at 1080p: > 2 x 43 px, the longest truth ball side of the selection split
)
TILE_STRIDE = 32  # tile sides are multiples of the model stride, so imgsz = long side means no resize
EDGE_PX = 2.0  # a tile ball within this of an inner tile edge is cut by it
MERGE_IOS = (
    0.5  # SAHI's default match metric (intersection over the smaller box) and threshold
)
BALL_MAX = 300  # merged ball candidates per frame, as max_det
GAMES = {"selection": bs.SPLITS["selection"], "all": bs.SPLITS["full"]}
SMALL = ("<8", "8-12")  # ticket 01's size bins under 12 px (mean box side at 1080p)

logger = logging.getLogger("ball_inference")


# ------------------------------------------------------------------ tiles
def tile_grid(
    width: int, height: int, columns: int, rows: int
) -> tuple[list[tuple[int, int, int, int]], int]:
    """``(tiles, imgsz)``: ``columns x rows`` tiles ``(x0, y0, x1, y1)`` covering the image, neighbours overlapping by
    at least ``MIN_OVERLAP_1080`` px scaled by ``height / 1080``, each side rounded up to ``TILE_STRIDE``, spread
    evenly from 0 to the image side; ``imgsz`` is the tiles' long side."""
    overlap = MIN_OVERLAP_1080 * height / 1080

    def side(full: int, n: int) -> int:
        size = math.ceil((full + (n - 1) * overlap) / n / TILE_STRIDE) * TILE_STRIDE
        return min(size, full) if n > 1 else full

    def starts(full: int, n: int, size: int) -> list[int]:
        return [round(i * (full - size) / (n - 1)) if n > 1 else 0 for i in range(n)]

    w, h = side(width, columns), side(height, rows)
    tiles = [
        (x, y, x + w, y + h)
        for y in starts(height, rows, h)
        for x in starts(width, columns, w)
    ]
    return tiles, max(w, h)


def inner_edge_cut(
    xyxy: np.ndarray, tile: tuple[int, int, int, int], width: int, height: int
) -> np.ndarray:
    """Per box in the tile's pixels: touches (within ``EDGE_PX``) a tile edge that is not the image border."""
    x0, y0, x1, y1 = tile
    w, h = x1 - x0, y1 - y0
    cut = np.zeros(len(xyxy), bool)
    if x0 > 0:
        cut |= xyxy[:, 0] <= EDGE_PX
    if y0 > 0:
        cut |= xyxy[:, 1] <= EDGE_PX
    if x1 < width:
        cut |= xyxy[:, 2] >= w - EDGE_PX
    if y1 < height:
        cut |= xyxy[:, 3] >= h - EDGE_PX
    return cut


def ios_matrix(xyxy: np.ndarray) -> np.ndarray:
    """Intersection over the smaller box's area of every pair of boxes."""
    low = np.maximum(xyxy[:, None, :2], xyxy[None, :, :2])
    high = np.minimum(xyxy[:, None, 2:], xyxy[None, :, 2:])
    inter = np.clip(high - low, 0, None).prod(axis=2)
    area = np.clip(xyxy[:, 2:] - xyxy[:, :2], 0, None).prod(axis=1)
    smaller = np.minimum(area[:, None], area[None, :])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(smaller > 0, inter / smaller, 0.0)


def greedy_ios_nms(xyxy: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    """Indices kept by greedy NMS on IoS >= ``MERGE_IOS``, most confident first, at most ``BALL_MAX``."""
    order = np.argsort(-confidence, kind="stable")
    ios = ios_matrix(xyxy[order])
    suppressed = np.zeros(len(order), bool)
    kept = []
    for i in range(len(order)):
        if suppressed[i]:
            continue
        kept.append(order[i])
        if len(kept) == BALL_MAX:
            break
        suppressed |= ios[i] >= MERGE_IOS
    return np.asarray(kept, int)


def merge_tiles(
    full: sv.Detections,
    tiles: list[tuple[tuple[int, int, int, int], sv.Detections]],
    width: int,
    height: int,
    ball: str = "ball",
) -> sv.Detections:
    """The full frame's people and the merged balls of the full frame and the tiles (boxes in the frame's pixels)."""
    parts = [full[full.data["class_name"] == ball]]
    for tile, found in tiles:
        balls = found[found.data["class_name"] == ball]
        balls = balls[~inner_edge_cut(balls.xyxy, tile, width, height)]
        balls.xyxy = balls.xyxy + np.asarray(tile[:2] * 2, balls.xyxy.dtype)
        parts.append(balls)
    balls = sv.Detections.merge(parts)
    if len(balls):
        balls = balls[greedy_ios_nms(balls.xyxy, balls.confidence)]
    return sv.Detections.merge([full[full.data["class_name"] != ball], balls])


# ------------------------------------------------------------------ predictors
def load(imgsz: int, augment: bool, device: str | None):
    """dd2c at ``imgsz`` and NMS IoU 0.7, every candidate down to the dump floor; ``augment`` turns on Ultralytics'
    test-time augmentation for every call."""
    model = tv.load_model(
        "yolo", WEIGHTS, conf=bench.EVAL_CONF, iou=NMS_IOU, imgsz=imgsz, device=device
    )
    if augment:
        model._yolo.overrides["augment"] = True
    return model


@dataclass
class Single:
    """One call on the native frame."""

    model: object

    def __call__(self, image: np.ndarray) -> sv.Detections:
        return self.model.predict(image)


class Tiled:
    """The full frame in the 1280 row plus ``columns x rows`` native-resolution tiles, merged by ``merge_tiles``."""

    def __init__(self, columns: int, rows: int, augment: bool, device: str | None):
        self.columns, self.rows, self.augment, self.device = (
            columns,
            rows,
            augment,
            device,
        )
        self.full = load(FULL_FRAME["imgsz"], augment, device)
        self.models: dict[int, object] = {}

    def grid(self, width: int, height: int):
        return tile_grid(width, height, self.columns, self.rows)

    def __call__(self, image: np.ndarray) -> sv.Detections:
        height, width = image.shape[:2]
        tiles, imgsz = self.grid(width, height)
        if imgsz not in self.models:
            self.models[imgsz] = load(imgsz, self.augment, self.device)
        model = self.models[imgsz]
        found = [
            (t, model.predict(np.ascontiguousarray(image[t[1] : t[3], t[0] : t[2]])))
            for t in tiles
        ]
        return merge_tiles(self.full.predict(image), found, width, height)


def variant_settings(variant: str, augment: bool) -> dict:
    """The dump record's row settings of a variant."""
    if variant in SINGLE:
        settings = {"imgsz": SINGLE[variant], "iou": NMS_IOU, "input_size": None}
    else:
        columns, rows = TILES[variant]
        settings = dict(FULL_FRAME) | {
            "tiles": {
                "columns": columns,
                "rows": rows,
                "grid_1920x1080": tile_grid(1920, 1080, columns, rows),
                "grid_1280x720": tile_grid(1280, 720, columns, rows),
                "tile_nms_iou": NMS_IOU,
                "min_overlap_1080_px": MIN_OVERLAP_1080,
                "inner_edge_px": EDGE_PX,
                "merge": f"balls of the full frame and of the tiles (inner-edge balls dropped): greedy NMS on "
                f"IoS >= {MERGE_IOS}, most confident kept, at most {BALL_MAX}; people from the full frame",
            }
        }
    return settings | {"augment": augment}


def predictor(variant: str, augment: bool, device: str | None):
    if variant in SINGLE:
        return Single(load(SINGLE[variant], augment, device))
    return Tiled(*TILES[variant], augment, device)


# ------------------------------------------------------------------ dump
def dump(args: argparse.Namespace) -> None:
    """One variant's prediction dump in `detector_bench.py dump`'s format (one row, named after the variant)."""
    from ultralytics.cfg import DEFAULT_CFG_DICT

    row = args.variant + ("+tta" if args.augment else "")
    games = GAMES[args.games]
    sets = bench.evaluation_sets("valid", list(bench.DUMP_SETS))
    sets["gsr"] = (bs.subset(sets["gsr"][0], games), "valid")
    predict = predictor(args.variant, args.augment, args.device)
    images, candidates, seconds, ms_per_frame = [], [], {}, {}
    for name, (dataset, split) in sets.items():
        clock = time.perf_counter()
        timings = []
        for sample in dataset[split]:
            image = sample.read_image()
            start = time.perf_counter()
            found = predict(image)
            timings.append(time.perf_counter() - start)
            key = bench.image_key(name, sample)
            images.append(
                {"set": name}
                | key
                | {
                    "path": str(sample.image_path),
                    "width": sample.width,
                    "height": sample.height,
                }
            )
            table = pd.DataFrame(
                np.asarray(found.xyxy, np.float32).reshape(-1, 4),
                columns=bench.CANDIDATE_COLUMNS,
            )
            table["conf"] = np.asarray(found.confidence, np.float32)
            table["class_id"] = np.asarray(found.class_id, np.int8)
            table.insert(0, "image", key["image"])
            table.insert(0, "row", row)
            table.insert(0, "set", name)
            candidates.append(table)
        seconds[f"{name}/{row}"] = round(time.perf_counter() - clock, 1)
        ms_per_frame[f"{name}/{row}"] = round(
            1000 * float(np.mean(timings[bench.DUMP_WARMUP :] or timings)), 2
        )
        logger.info(
            "%s/%s dumped in %.0f s (%.1f ms/frame)",
            name,
            row,
            seconds[f"{name}/{row}"],
            ms_per_frame[f"{name}/{row}"],
        )
    model = predict.model if isinstance(predict, Single) else predict.full
    class_names = model.class_names
    frame = pd.concat(candidates, ignore_index=True)
    frame["class_name"] = pd.Categorical.from_codes(frame.class_id, class_names)
    for column in ("set", "row"):
        frame[column] = frame[column].astype("category")
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.out / "candidates.parquet", index=False)
    pd.DataFrame(images).to_parquet(args.out / "images.parquet", index=False)
    record = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "ticket": args.ticket,
        "variant": args.variant,
        "weights": str(WEIGHTS),
        "weights_sha256": bench.sha256_file(WEIGHTS),
        "conf": bench.EVAL_CONF,
        "rows": {row: variant_settings(args.variant, args.augment)},
        "max_det": DEFAULT_CFG_DICT["max_det"],
        "agnostic_nms": DEFAULT_CFG_DICT["agnostic_nms"],
        "half": DEFAULT_CFG_DICT["half"],
        "classes": dict(enumerate(class_names)),
        "games": list(games),
        "sets": {
            name: {
                "split": split,
                "images": len(dataset[split]),
                "resolutions": sorted(
                    {f"{s.width}x{s.height}" for s in dataset[split]}
                ),
                "class_map": bench.class_map(class_names, dataset.class_names),
            }
            for name, (dataset, split) in sets.items()
        },
        "boxes": "xyxy in the image's own pixels (images.parquet width x height)",
        "image_key": "GSR: <clip>/<frame:06d> (1-based frame, every STRIDE-th); Ekstraklasa dev: the file stem",
        "seconds": seconds,
        "gpu": bench.gpu_name(args.device),
        "ms_per_frame": ms_per_frame,
        "ms_per_frame_note": f"mean wall time of all model calls of a frame (tiles and full frame, inference at "
        f"conf {bench.EVAL_CONF}, NMS) and the ball merge, batch 1, image read excluded, the first "
        f"{bench.DUMP_WARMUP} images of each set left out",
        "candidates": len(frame),
    }
    (args.out / "dump.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


# ------------------------------------------------------------------ post-filter
def long_ball(candidates: pd.DataFrame, heights: pd.Series, limit: float) -> np.ndarray:
    """Ball candidates whose long side, in 1080p pixels, is over ``limit``."""
    side = np.maximum(
        candidates.x2 - candidates.x1, candidates.y2 - candidates.y1
    ).to_numpy()
    scale = 1080 / heights.to_numpy(float)
    return (candidates.class_name.astype(str).to_numpy() == "ball") & (
        side * scale > limit
    )


def postfilter(args: argparse.Namespace) -> None:
    """A dump of ``--row`` without the ball candidates longer than ``RULE["post_filter_long_side_px"]`` (1080p px)."""
    limit = bs.RULE["post_filter_long_side_px"]
    record, images, candidates = bench.read_dump(args.dump)
    candidates = candidates[candidates.row == args.row].copy()
    heights = candidates[["set", "image"]].merge(
        images[["set", "image", "height"]], on=["set", "image"], how="left"
    )["height"]
    drop = long_ball(candidates, heights, limit)
    row = f"{args.row}+pf{limit}"
    kept = candidates[~drop].copy()
    kept["row"] = pd.Categorical([row] * len(kept))
    args.out.mkdir(parents=True, exist_ok=True)
    kept.to_parquet(args.out / "candidates.parquet", index=False)
    images.to_parquet(args.out / "images.parquet", index=False)
    dropped = candidates[drop]
    record = record | {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "rows": {row: record["rows"][args.row] | {"post_filter_long_side_px": limit}},
        "post_filter": {
            "source": str(args.dump),
            "row": args.row,
            "long_side_1080_px_over": limit,
            "dropped": {
                s: int((dropped.set == s).sum()) for s in candidates.set.unique()
            },
            "dropped_at_0.15": {
                s: int(((dropped.set == s) & (dropped.conf >= bs.BALL_CONF)).sum())
                for s in candidates.set.unique()
            },
        },
        "seconds": {
            k.replace(f"/{args.row}", f"/{row}"): v
            for k, v in record["seconds"].items()
            if k.endswith(f"/{args.row}")
        },
        "ms_per_frame": {
            k.replace(f"/{args.row}", f"/{row}"): v
            for k, v in record.get("ms_per_frame", {}).items()
            if k.endswith(f"/{args.row}")
        },
        "candidates": len(kept),
    }
    (args.out / "dump.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record["post_filter"], indent=2))


# ------------------------------------------------------------------ table
def size_gain(variant: Path, row: str, reference: Path) -> dict:
    """Decision 9: the net gain in well placed frames (gained - lost) and in ball hits at 0.15 over dd2c 1280 on the
    selection split, and the share of it on balls < 12 px."""
    rows = pd.read_parquet(variant / f"placement_selection_{row}.parquet")
    base = pd.read_parquet(reference / "placement_selection_1280.parquet")
    truth = pd.read_parquet(bs.TRUTH / "balls_gsr_games_23.parquet")
    joined = rows.merge(base, on=["clip", "frame"], suffixes=("", "_ref")).merge(
        truth[["clip", "frame", "size"]], on=["clip", "frame"], how="left"
    )
    if len(joined) != len(rows) or joined["size"].isna().any():
        raise SystemExit("placement frames do not join one truth ball each")
    far = bench.PLACEMENT["far_m"]
    gain = (joined.error_m <= far).astype(int) - (joined.error_m_ref <= far).astype(int)
    small = joined["size"].astype(str).isin(SMALL)
    net, net_small = int(gain.sum()), int(gain[small].sum())
    out = {
        "frames": len(joined),
        "small_frames_share": float(small.mean()),
        "well_placed_net_frames": net,
        "well_placed_net_small": net_small,
        "well_placed_small_share": net_small / net if net else np.nan,
        "gain_per_frame_small": float(gain[small].mean()),
        "gain_per_frame_large": float(gain[~small].mean()),
    }
    cards = [
        json.loads((p / "scorecard.json").read_text())["cards"][f"selection/{r}"][
            "bins"
        ]
        for p, r in ((variant, row), (reference, "1280"))
    ]
    for rule in ("iou", "centre"):
        hits = {}
        for b, b_ref in zip(cards[0], cards[1], strict=True):
            if b["feature"] != "size":
                continue
            hits[b["level"]] = b["balls"] * (
                b[f"recall_{rule}"] - b_ref[f"recall_{rule}"]
            )
        total = sum(hits.values())
        out[f"hits_{rule}_net"] = round(total, 1)
        out[f"hits_{rule}_net_by_size"] = {k: round(v, 1) for k, v in hits.items()}
        out[f"hits_{rule}_small_share"] = (
            sum(hits[s] for s in SMALL) / total if total else np.nan
        )
    out["concentrated"] = bool(
        out["well_placed_small_share"] >= 0.5
        and out["gain_per_frame_small"] > out["gain_per_frame_large"]
    )
    return out


def card_row(name: str, folder: Path, reference: Path, row: str | None) -> dict:
    """The ticket's table row of one card (selection split and Ekstraklasa dev; ``row`` None: the card's only row)."""
    result = json.loads((folder / "scorecard.json").read_text())
    if row is None:
        (row,) = {k.split("/")[1] for k in result["cards"]}
    sel, eks = (
        result["cards"][f"selection/{row}"],
        result["cards"][f"ekstraklasa/{row}"],
    )
    size = {b["level"]: b for b in sel["bins"] if b["feature"] == "size"}
    pv, dv = sel["placement_vs_reference"], sel["detection_vs_reference"]
    ms = result["cost"]["ms_per_frame"]
    record = {
        "variant": name,
        "row": row,
        "ball_ap50": sel["detection"]["ball_ap50"]["value"],
        "ball_ap50_diff": dv["ball_ap50"]["difference"],
        "ball_ap50_diff_ci": dv["ball_ap50"]["ci"],
        "well_placed": sel["placement"]["well_placed"]["value"],
        "well_placed_ci": sel["placement"]["well_placed"]["ci"],
        "well_placed_diff": pv["well_placed"]["difference"],
        "well_placed_diff_ci": pv["well_placed"]["ci"],
        "well_with_h": sel["placement"]["well_placed_with_homography"]["value"],
        "far_of_placed": sel["placement"]["far_of_placed"]["value"],
        "far_diff": pv["far_of_placed"]["difference"],
        "far_diff_ci": pv["far_of_placed"]["ci"],
        "recall_iou_lt8": size["<8"]["recall_iou"],
        "recall_iou_8_12": size["8-12"]["recall_iou"],
        "recall_centre_lt8": size["<8"]["recall_centre"],
        "recall_centre_8_12": size["8-12"]["recall_centre"],
        "false_balls_per_frame": sel["hits"]["false_balls_per_frame"]["value"],
        "eks_ball_ap50": eks["detection"]["ball_ap50"]["value"],
        "eks_ball_ap50_ci": eks["detection"]["ball_ap50"]["ci"],
        "eks_map50_95": eks["detection"]["map50_95"]["value"],
        "eks_map50_95_diff": eks["detection_vs_reference"]["map50_95"]["difference"],
        "people_map50_95": sel["detection"]["people_map50_95"]["value"],
        "people_diff": dv["people_map50_95"]["difference"],
        "people_diff_ci": dv["people_map50_95"]["ci"],
        "ms_per_frame": ms.get(f"gsr/{row}"),
        "ms_per_frame_eks": ms.get(f"ekstraklasa/{row}"),
        "rule": result["rule"],
    }
    if folder.resolve() != reference.resolve():
        record["size_gain"] = size_gain(folder, row, reference)
    return record


def markdown(records: list[dict]) -> str:
    def pp(v, ci=None):
        text = f"{100 * v:+.2f}"
        return text + (f" [{100 * ci[0]:+.2f}; {100 * ci[1]:+.2f}]" if ci else "")

    lines = [
        "# ball-detection 03: variants on the selection split (games 2+3), paired with dd2c 1280",
        "",
        "| variant | ball AP50 (Δ) | well placed | Δ well placed, pp [95%] | > 5 m of placed (Δ pp) | recall IoU / "
        "centre < 8 px | recall IoU / centre 8-12 px | false balls / frame | Ekstraklasa ball AP50 [95%] | "
        "Ekstraklasa mAP50-95 (Δ) | people mAP50-95 (Δ) | ms/frame |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        lines.append(
            f"| {r['variant']} | {r['ball_ap50']:.3f} ({r['ball_ap50_diff']:+.3f}) | "
            f"{100 * r['well_placed']:.2f}% | {pp(r['well_placed_diff'], r['well_placed_diff_ci'])} | "
            f"{100 * r['far_of_placed']:.2f}% ({pp(r['far_diff'])}) | "
            f"{r['recall_iou_lt8']:.3f} / {r['recall_centre_lt8']:.3f} | "
            f"{r['recall_iou_8_12']:.3f} / {r['recall_centre_8_12']:.3f} | {r['false_balls_per_frame']:.3f} | "
            f"{r['eks_ball_ap50']:.3f} [{r['eks_ball_ap50_ci'][0]:.2f}; {r['eks_ball_ap50_ci'][1]:.2f}] | "
            f"{r['eks_map50_95']:.3f} ({r['eks_map50_95_diff']:+.3f}) | "
            f"{r['people_map50_95']:.3f} ({r['people_diff']:+.3f}) | {r['ms_per_frame']} |"
        )
    lines += [
        "",
        "Decision 9: net gain over dd2c 1280 (gained - lost) and its share on balls < 12 px",
        "",
        "| variant | well placed net frames | of it < 12 px | share | gain/frame < 12 / >= 12 px | hits IoU net "
        "(< 12 share) | hits centre net (< 12 share) | concentrated |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        g = r.get("size_gain")
        if not g:
            continue
        lines.append(
            f"| {r['variant']} | {g['well_placed_net_frames']} | {g['well_placed_net_small']} | "
            f"{g['well_placed_small_share']:.0%} | {g['gain_per_frame_small']:+.4f} / "
            f"{g['gain_per_frame_large']:+.4f} | {g['hits_iou_net']:+.0f} ({g['hits_iou_small_share']:.0%}) | "
            f"{g['hits_centre_net']:+.0f} ({g['hits_centre_small_share']:.0%}) | {g['concentrated']} |"
        )
    small = records[0].get("size_gain") or next(
        (r["size_gain"] for r in records if "size_gain" in r), {}
    )
    if small:
        lines.append(
            f"\nFrames with a ball < 12 px: {small['small_frames_share']:.1%} of {small['frames']}."
        )
    return "\n".join(lines) + "\n"


def table(args: argparse.Namespace) -> None:
    records = []
    for card in args.card:
        name, folder = card.split("=", 1)
        folder, _, row = folder.partition("@")
        records.append(card_row(name, Path(folder), args.reference, row or None))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "variants.json").write_text(
        json.dumps(records, indent=2, default=float) + "\n"
    )
    flat = pd.DataFrame([{k: v for k, v in r.items() if k != "rule"} for r in records])
    flat.to_csv(args.out / "variants.csv", index=False)
    text = markdown(records)
    (args.out / "variants.md").write_text(text)
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    d = commands.add_parser("dump", help="one variant's prediction dump (GPU)")
    d.add_argument("--variant", choices=[*SINGLE, *TILES], required=True)
    d.add_argument("--augment", action="store_true", help="Ultralytics TTA")
    d.add_argument(
        "--games",
        choices=list(GAMES),
        default="selection",
        help="GSR valid games: selection (2+3) or all (only the primary variant, for a confirm read)",
    )
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--device", default=None)
    d.add_argument(
        "--ticket", default="ball-detection 03", help="the dump record's ticket"
    )
    p = commands.add_parser("postfilter", help="drop ball candidates longer than 45 px")
    p.add_argument("--dump", type=Path, required=True)
    p.add_argument("--row", required=True)
    p.add_argument("--out", type=Path, required=True)
    t = commands.add_parser("table", help="the ticket's table from scorecards")
    t.add_argument("--card", nargs="+", required=True, help="NAME=CARD_DIR[@ROW]")
    t.add_argument("--reference", type=Path, default=REFERENCE)
    t.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    tv.setup_logging("INFO")
    {"dump": dump, "postfilter": postfilter, "table": table}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
