"""ball-detection 01: where dd2c misses the ball and where it sees a ball that is not there.

    uv run python scripts/detector_bench.py dump --weights W --out DUMP   # the candidates (GPU, once)
    uv run python scripts/ball_errors.py check --dump DUMP --out OUT       # the dump against the bench's result files
    uv run python scripts/ball_errors.py analyse --dump DUMP --out OUT     # per-ball features, recall tables, curve
    uv run python scripts/ball_errors.py gallery --dump DUMP --out OUT     # 50 seeded false-ball frames, crop sheets
    uv run python scripts/ball_errors.py taxonomy --out OUT                # shares of the gallery's classes

`check` scores the dump as `detector_bench.py evaluate` and `placement` score the weights and compares the numbers
with `final/candidate_valid/detection_valid.json` and `placement_frames.parquet` (whole GSR valid: the one read of
game 5 this ticket makes). Everything else reads only the selection split (GSR valid games 2 and 3) and Ekstraklasa
dev.

A truth ball is hit at a confidence threshold by one of two rules, ball predictions taken greedily, most confident
first, each by the free truth ball it fits best: `iou` (IoU >= 0.5, `evaluate`'s rule) or `centre` (centre distance
<= r = max(5 px, ball diameter), diameter = the mean of the box's sides). Per-ball features are proxies computed from
the GSR labels and the image (GSR has no visibility attributes): size, blur (image speed from the ball's track at
25 fps, box elongation, Laplacian variance), occlusion (cover by person boxes), contact (pitch distance to the nearest
feet), in the air (far from every foot and off a straight pitch path), position (camera distance, near goal, off
pitch) and the best ball candidate within r.
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parent))

import detector_bench as bench  # noqa: E402

import tactifoot_vision as tv  # noqa: E402

ROOT = Path("/mnt/storage/tactifoot_research/detector_data")
OUT = ROOT / "ball/01"
DUMP = OUT / "dump"
DETECTION_VALID = ROOT / "final/candidate_valid/detection_valid.json"
PLACEMENT_FRAMES = ROOT / "runs/compare/placement_run2c_epoch1/placement_frames.parquet"
SELECTION_GAMES = ("2", "3")  # spec "Protokół oceny": game 5 is the confirmation split
ROWS = bench.DUMP_ROWS
OPERATING = bench.OPERATING["class_conf"]["ball"]
THRESHOLDS = (0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7)
MIN_RADIUS = 5.0
SIZE_EDGES = (0, 8, 12, 16, 24, math.inf)
SIZE_LABELS = ("<8", "8-12", "12-16", "16-24", ">=24")
BLUR_SPEED = (
    1.0  # ball diameters per frame: a streak of >= half a diameter at a 1/50 s shutter
)
ELONGATED = 1.5  # long over short box side
OCCLUDED = 0.5  # share of the ball box under person boxes
PERSON_IOU = 0.5  # a ball candidate on a labelled person's box
CONTACT_M = 1.0  # pitch distance to the nearest feet
FREE_M = 2.0  # in the air: every foot farther than this ...
AIR_SPAN = 10  # ... and the ball this far (frames) off the straight path between frames f - span and f + span
AIR_DEVIATION_M = 2.0
CAMERA_THIRD = (
    2 * bench.HALF_WIDTH / 3
)  # pitch y bands; the main camera is on the +y side (balls grow with y)
BOOTSTRAP = {"draws": 2000, "seed": 0}
SAMPLE = {"per_source": 25, "seed": 0}
FALSE_CLASSES = (
    "spare_ball",
    "penalty_spot_line",
    "head",
    "boot",
    "crowd",
    "person_box",
    "true_ball",
    "other",
)
FEATURES = (
    "size",
    "blur",
    "elongated",
    "sharpness",
    "occlusion",
    "contact",
    "in_air",
    "camera",
    "near_goal",
    "off_pitch",
    "candidate",
)
CROSSES = (
    ("size", "blur"),
    ("size", "candidate"),
    ("size", "camera"),
    ("size", "occlusion"),
    ("blur", "candidate"),
    ("camera", "candidate"),
)

logger = logging.getLogger("ball_errors")


# ------------------------------------------------------------------ matching
def diameter(xyxy: np.ndarray) -> np.ndarray:
    """Ball diameter in pixels: the mean of the box's width and height."""
    xyxy = np.asarray(xyxy, float).reshape(-1, 4)
    return ((xyxy[:, 2] - xyxy[:, 0]) + (xyxy[:, 3] - xyxy[:, 1])) / 2


def radius(xyxy: np.ndarray) -> np.ndarray:
    """The centre rule's radius of each truth ball: max(``MIN_RADIUS``, diameter)."""
    return np.maximum(MIN_RADIUS, diameter(xyxy))


def centres(xyxy: np.ndarray) -> np.ndarray:
    xyxy = np.asarray(xyxy, float).reshape(-1, 4)
    return (xyxy[:, :2] + xyxy[:, 2:]) / 2


def greedy_match(conf: np.ndarray, affinity: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Matched prediction per truth (-1: none). Predictions, most confident first, take the free truth of highest
    ``affinity`` among those ``ok`` for them (``tv.evaluation.evaluate_detector``'s greedy matching)."""
    match = np.full(ok.shape[1], -1)
    if not ok.size:
        return match
    for p in np.argsort(-np.asarray(conf), kind="stable"):
        candidates = np.where(ok[p] & (match < 0), affinity[p], -np.inf)
        best = int(np.argmax(candidates))
        if np.isfinite(candidates[best]):
            match[best] = p
    return match


def match(
    pred_xyxy: np.ndarray, conf: np.ndarray, truth_xyxy: np.ndarray, rule: str
) -> np.ndarray:
    """Matched prediction per truth ball under ``rule``: ``iou`` (IoU >= 0.5) or ``centre`` (distance <= r)."""
    pred_xyxy = np.asarray(pred_xyxy, float).reshape(-1, 4)
    truth_xyxy = np.asarray(truth_xyxy, float).reshape(-1, 4)
    if rule == "iou":
        iou = sv.box_iou_batch(pred_xyxy, truth_xyxy)
        return greedy_match(conf, iou, iou >= bench.MATCH_IOU)
    if rule == "centre":
        distance = np.linalg.norm(
            centres(pred_xyxy)[:, None] - centres(truth_xyxy)[None], axis=2
        )
        return greedy_match(conf, -distance, distance <= radius(truth_xyxy)[None])
    raise ValueError(f"unknown rule {rule!r}")


def hits(
    pred_xyxy: np.ndarray,
    conf: np.ndarray,
    truth_xyxy: np.ndarray,
    thresholds=THRESHOLDS,
) -> dict[str, np.ndarray]:
    """``{"<rule>@<t>": hit per truth}`` for both rules at every threshold."""
    conf = np.asarray(conf, float)
    found = {}
    for t in thresholds:
        keep = conf >= t
        for rule in ("iou", "centre"):
            found[f"{rule}@{t:g}"] = (
                match(pred_xyxy[keep], conf[keep], truth_xyxy, rule) >= 0
            )
    return found


def false_balls(
    pred_xyxy: np.ndarray, conf: np.ndarray, truth_xyxy: np.ndarray
) -> np.ndarray:
    """Indices of the ball predictions >= ``OPERATING`` that no truth ball takes under the centre rule."""
    conf = np.asarray(conf, float)
    kept = np.flatnonzero(conf >= OPERATING)
    taken = match(np.asarray(pred_xyxy)[kept], conf[kept], truth_xyxy, "centre")
    return np.setdiff1d(kept, kept[taken[taken >= 0]])


# ------------------------------------------------------------------ features
def size_bin(d: np.ndarray) -> pd.Categorical:
    return pd.cut(np.asarray(d, float), SIZE_EDGES, right=False, labels=SIZE_LABELS)


def coverage(box: np.ndarray, others: np.ndarray, samples: int = 16) -> float:
    """Share of ``box`` (xyxy) inside the union of ``others``, on a ``samples`` x ``samples`` grid."""
    others = np.asarray(others, float).reshape(-1, 4)
    if not len(others):
        return 0.0
    x1, y1, x2, y2 = np.asarray(box, float)
    steps = (np.arange(samples) + 0.5) / samples
    xs, ys = np.meshgrid(x1 + steps * (x2 - x1), y1 + steps * (y2 - y1))
    inside = (
        (xs[..., None] >= others[:, 0])
        & (xs[..., None] <= others[:, 2])
        & (ys[..., None] >= others[:, 1])
        & (ys[..., None] <= others[:, 3])
    )
    return float(inside.any(axis=2).mean())


def laplacian_variance(image: np.ndarray, box: np.ndarray) -> float:
    """Variance of the grey Laplacian inside ``box`` (lower: blurrier or flatter)."""
    h, w = image.shape[:2]
    x1, y1 = max(int(np.floor(box[0])), 0), max(int(np.floor(box[1])), 0)
    x2, y2 = min(int(np.ceil(box[2])), w), min(int(np.ceil(box[3])), h)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return np.nan
    grey = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def track_motion(balls: pd.DataFrame) -> pd.DataFrame:
    """Per ball annotation (``clip``, ``track_id``, ``frame``, ``cx``, ``cy``, ``pitch_x``, ``pitch_y``): image speed
    (px per frame, central difference over frames f +- 1, one-sided at a gap), pitch speed (m/s at 25 fps) and the
    pitch distance from the midpoint of frames f +- ``AIR_SPAN`` (``deviation_m``, NaN without both)."""
    table = balls.sort_values(["clip", "track_id", "frame"]).copy()
    groups = table.groupby(["clip", "track_id"], sort=False)

    def at(offset: int, columns: list[str]) -> pd.DataFrame:
        moved = groups[["frame", *columns]].shift(-offset)
        ok = (moved["frame"] - table["frame"]) == offset
        return moved[columns].where(ok)

    for columns, name, rate in (
        (["cx", "cy"], "speed_px", 1.0),
        (["pitch_x", "pitch_y"], "pitch_speed", 25.0),
    ):
        after, before = at(1, columns), at(-1, columns)
        span = after.notna().all(axis=1).astype(int) + before.notna().all(axis=1)
        ahead = after.where(after.notna().all(axis=1), table[columns], axis=0)
        behind = before.where(before.notna().all(axis=1), table[columns], axis=0)
        distance = np.hypot(*(ahead - behind).to_numpy(float).T)
        table[name] = distance / span.replace(0, np.nan) * rate
    pitch = ["pitch_x", "pitch_y"]
    far_after, far_before = at(AIR_SPAN, pitch), at(-AIR_SPAN, pitch)
    middle = (far_after + far_before) / 2
    table["deviation_m"] = np.hypot(*(table[pitch] - middle).to_numpy().T)
    return table


def clip_labels(
    path: Path,
) -> tuple[pd.DataFrame, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Every ball annotation of a GSR clip (all frames, 25 fps; pitch in centre metres, NaN without ``bbox_pitch``) and
    per frame the person boxes, pitch feet and roles of every non-ball object (``other`` included)."""
    labels = json.loads(path.read_text())
    frames = {i["image_id"]: int(Path(i["file_name"]).stem) for i in labels["images"]}
    unlabelled = {
        i["image_id"] for i in labels["images"] if i.get("is_labeled") is False
    }
    ball_ids = {c["id"] for c in labels["categories"] if c["name"] == "ball"}
    balls, persons = [], {}
    for item in labels["annotations"]:
        if item.get("supercategory") != "object" or item["image_id"] in unlabelled:
            continue
        box, pitch = item["bbox_image"], item.get("bbox_pitch") or {}
        x, y, w, h = (float(box[k]) for k in ("x_center", "y_center", "w", "h"))
        xy = (
            pitch.get("x_bottom_middle", np.nan),
            pitch.get("y_bottom_middle", np.nan),
        )
        frame = frames[item["image_id"]]
        if item.get("category_id") in ball_ids:
            balls.append(
                {
                    "order": len(balls),
                    "frame": frame,
                    "track_id": int(item["track_id"]),
                    "cx": x,
                    "cy": y,
                    "box": (x - w / 2, y - h / 2, x + w / 2, y + h / 2),
                    "pitch_x": float(xy[0]),
                    "pitch_y": float(xy[1]),
                }
            )
        else:
            persons.setdefault(frame, []).append(
                (
                    (x - w / 2, y - h / 2, x + w / 2, y + h / 2, *map(float, xy)),
                    item["attributes"]["role"],
                )
            )
    table = pd.DataFrame(balls)
    table["clip"] = path.parent.name
    people = {}
    for f, found in persons.items():
        values = np.asarray([v for v, _ in found], float)
        people[f] = (values[:, :4], values[:, 4:], np.asarray([r for _, r in found]))
    return table, people


def gsr_balls(
    dataset: tv.data.Dataset, games=SELECTION_GAMES, split: str = "valid"
) -> pd.DataFrame:
    """One row per truth ball of the scored GSR frames of ``games`` (the bench's boxes) with its features;
    ``split``: the GSR split folder and ``dataset`` split of the frames (valid; train for ticket 10's holdout)."""
    samples = [s for s in dataset[split] if s.meta["game"] in games]
    rows = []
    for clip in sorted({s.meta["sequence"] for s in samples}):
        balls, people = clip_labels(bench.GSR / split / clip / "Labels-GameState.json")
        balls = track_motion(balls).sort_values("order")
        by_frame = {f: g for f, g in balls.groupby("frame")}
        for sample in (s for s in samples if s.meta["sequence"] == clip):
            truth = sample.annotations.boxes[sample.annotations.class_ids == 0]
            if not len(truth):
                continue
            frame = int(sample.meta["frame"])
            raw = by_frame[frame]
            raw_boxes = np.array(raw["box"].tolist(), np.float32)
            raw_boxes[:, [0, 2]] = raw_boxes[:, [0, 2]].clip(0, sample.width)
            raw_boxes[:, [1, 3]] = raw_boxes[:, [1, 3]].clip(0, sample.height)
            kept = (raw_boxes[:, 2] > raw_boxes[:, 0]) & (
                raw_boxes[:, 3] > raw_boxes[:, 1]
            )
            raw = raw[kept]
            if not np.array_equal(raw_boxes[kept], truth):
                raise SystemExit(
                    f"{clip} frame {frame}: label balls differ from the bench's"
                )
            image = sample.read_image()
            person_boxes, feet, _ = people.get(
                frame, (np.zeros((0, 4)), np.zeros((0, 2)), np.zeros(0, str))
            )
            for index, (box, (_, ball)) in enumerate(
                zip(truth, raw.iterrows(), strict=True)
            ):
                ball_xy = np.array([ball.pitch_x, ball.pitch_y])
                distances = np.linalg.norm(feet - ball_xy, axis=1)
                distances = distances[np.isfinite(distances)]
                rows.append(
                    {
                        "clip": clip,
                        "game": sample.meta["game"],
                        "frame": frame,
                        "image": f"{clip}/{frame:06d}",
                        "ball": index,
                        "x1": box[0],
                        "y1": box[1],
                        "x2": box[2],
                        "y2": box[3],
                        "track_id": ball.track_id,
                        "speed_px": ball.speed_px,
                        "pitch_speed": ball.pitch_speed,
                        "deviation_m": ball.deviation_m,
                        "pitch_x": ball.pitch_x,
                        "pitch_y": ball.pitch_y,
                        "feet_m": distances.min() if len(distances) else np.nan,
                        "coverage": coverage(box, person_boxes),
                        "laplacian": laplacian_variance(image, box),
                    }
                )
    return add_box_features(pd.DataFrame(rows))


def ekstraklasa_balls(dataset: tv.data.Dataset) -> pd.DataFrame:
    """One row per truth ball of Ekstraklasa dev (no tracks, no pitch positions); sizes scaled to 1080p."""
    rows = []
    for sample in dataset["valid"]:
        ids, boxes = sample.annotations.class_ids, sample.annotations.boxes
        truth, person_boxes = boxes[ids == 0], boxes[ids != 0]
        if not len(truth):
            continue
        image = sample.read_image()
        key = bench.image_key("ekstraklasa", sample)
        for index, box in enumerate(truth):
            rows.append(
                {"clip": key["clip"], "image": key["image"], "ball": index}
                | dict(zip(("x1", "y1", "x2", "y2"), box, strict=True))
                | {
                    "coverage": coverage(box, person_boxes),
                    "laplacian": laplacian_variance(image, box),
                    "scale_1080p": 1080 / sample.height,
                }
            )
    return add_box_features(pd.DataFrame(rows))


def add_box_features(table: pd.DataFrame) -> pd.DataFrame:
    """Size (1080p pixels), elongation and the categorical levels of every feature the table has."""
    xyxy = table[["x1", "y1", "x2", "y2"]].to_numpy(float)
    w, h = xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]
    scale = table["scale_1080p"].to_numpy(float) if "scale_1080p" in table else 1.0
    table["w"], table["h"] = w * scale, h * scale
    table["diameter"] = diameter(xyxy) * scale
    table["elongation"] = np.maximum(w, h) / np.minimum(w, h)
    table["size"] = size_bin(table["diameter"])
    table["elongated"] = np.where(table.elongation >= ELONGATED, ">=1.5", "<1.5")
    quartiles = table.laplacian.quantile([0.25, 0.5, 0.75]).to_numpy()
    table["sharpness"] = pd.cut(
        table.laplacian,
        [-np.inf, *quartiles, np.inf],
        labels=("Q1 (blurriest)", "Q2", "Q3", "Q4 (sharpest)"),
    )
    table["occlusion"] = pd.Categorical(
        np.select(
            [table.coverage <= 0, table.coverage < OCCLUDED], ["none", "<0.5"], ">=0.5"
        ),
        categories=("none", "<0.5", ">=0.5"),
    )
    if "speed_px" not in table:
        return table
    table["speed_d"] = table.speed_px / table.diameter
    table["blur"] = np.select(
        [table.speed_d.isna(), table.speed_d >= BLUR_SPEED],
        ["unknown", "blur"],
        "sharp",
    )
    table["contact"] = np.select(
        [table.feet_m.isna(), table.feet_m < CONTACT_M, table.feet_m < FREE_M],
        ["unknown", "<1 m", "1-2 m"],
        ">=2 m",
    )
    table["in_air"] = np.select(
        [
            table.feet_m.isna() | table.deviation_m.isna(),
            (table.feet_m >= FREE_M) & (table.deviation_m >= AIR_DEVIATION_M),
        ],
        ["unknown", "in air"],
        "ground",
    )
    table["camera"] = np.select(
        [
            table.pitch_y.isna(),
            table.pitch_y < -CAMERA_THIRD / 2,
            table.pitch_y <= CAMERA_THIRD / 2,
        ],
        ["unknown", "far third", "middle third"],
        "near third",
    )
    off = (table.pitch_x.abs() > bench.HALF_LENGTH) | (
        table.pitch_y.abs() > bench.HALF_WIDTH
    )
    table["off_pitch"] = np.select(
        [table.pitch_x.isna(), off], ["unknown", "off"], "on"
    )
    goal = bench.HALF_LENGTH - table.pitch_x.abs() <= bench.PLACEMENT["near_goal_m"]
    table["near_goal"] = np.select(
        [table.pitch_x.isna(), goal], ["unknown", "<=30 m"], ">30 m"
    )
    return table


def candidate_columns(
    truth: pd.DataFrame, found: dict[str, sv.Detections]
) -> pd.DataFrame:
    """The truth table with the best ball candidate within r (``best_conf``, ``candidate`` level) and the hit columns
    of both rules at every threshold, for one dump row."""
    table = truth.copy()
    best = np.zeros(len(table))
    hit_columns: dict[str, np.ndarray] = {}
    positions = {
        image: group.index.to_numpy() for image, group in table.groupby("image")
    }
    for image, index in positions.items():
        rows = table.index.get_indexer(index)
        boxes = table.loc[index, ["x1", "y1", "x2", "y2"]].to_numpy(float)
        balls = ball_predictions(found.get(image))
        distance = np.linalg.norm(
            centres(balls.xyxy)[:, None] - centres(boxes)[None], axis=2
        )
        near = np.where(distance <= radius(boxes)[None], balls.confidence[:, None], 0.0)
        best[rows] = near.max(axis=0) if len(balls) else 0.0
        for key, value in hits(balls.xyxy, balls.confidence, boxes).items():
            hit_columns.setdefault(key, np.zeros(len(table), bool))[rows] = value
    for key, value in hit_columns.items():
        table[key] = value
    table["best_conf"] = best
    table["candidate"] = np.select(
        [best <= 0, best < OPERATING], ["none", "<0.15"], ">=0.15"
    )
    return table


def ball_predictions(detections: sv.Detections | None) -> sv.Detections:
    if detections is None or not len(detections):
        return sv.Detections.empty()
    return detections[detections.data["class_name"] == "ball"]


def selection_people(
    clips: list[str],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``image -> (person boxes, roles)`` of every frame of the GSR valid ``clips``."""
    people = {}
    for clip in clips:
        _, persons = clip_labels(bench.GSR / "valid" / clip / "Labels-GameState.json")
        for frame, (boxes, _, roles) in persons.items():
            people[f"{clip}/{frame:06d}"] = (boxes, roles)
    return people


def candidate_kind(
    box: np.ndarray,
    truth_boxes: np.ndarray,
    person_boxes: np.ndarray,
    roles: np.ndarray,
) -> tuple[str, float]:
    """What a ball candidate sits on, from the labels alone: ``true ball`` (within r of a truth ball), ``person box
    (<role>)`` (IoU >= ``PERSON_IOU`` with a labelled person) or ``other``; and its best IoU with a person box."""
    box = np.asarray(box, float).reshape(1, 4)
    person_iou = (
        sv.box_iou_batch(box, np.asarray(person_boxes, float).reshape(-1, 4))[0]
        if len(person_boxes)
        else np.zeros(0)
    )
    best = float(person_iou.max()) if len(person_iou) else 0.0
    if len(truth_boxes):
        distance = np.linalg.norm(centres(truth_boxes) - centres(box), axis=1)
        if (distance <= radius(truth_boxes)).any():
            return "true ball", best
    if best >= PERSON_IOU:
        return f"person box ({roles[int(np.argmax(person_iou))]})", best
    return "other", best


def false_ball_table(
    images: list[str],
    truth: pd.DataFrame,
    found: dict[str, sv.Detections],
    people: dict[str, tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Every false ball >= ``OPERATING`` (unmatched under the centre rule) on ``images``, with ``candidate_kind``."""
    boxes_by_image = {
        image: group[["x1", "y1", "x2", "y2"]].to_numpy(float)
        for image, group in truth.groupby("image")
    }
    rows = []
    for image in images:
        balls = ball_predictions(found.get(image))
        boxes = boxes_by_image.get(image, np.zeros((0, 4)))
        person_boxes, roles = people.get(image, (np.zeros((0, 4)), np.zeros(0, str)))
        for p in false_balls(balls.xyxy, balls.confidence, boxes):
            kind, person_iou = candidate_kind(balls.xyxy[p], boxes, person_boxes, roles)
            nearest = (
                float(
                    np.min(
                        np.linalg.norm(centres(boxes) - centres(balls.xyxy[p]), axis=1)
                    )
                )
                if len(boxes)
                else np.nan
            )
            rows.append(
                {
                    "image": image,
                    "conf": float(balls.confidence[p]),
                    "truth_px": nearest,
                    "kind": kind,
                    "person_iou": person_iou,
                }
                | dict(
                    zip(("x1", "y1", "x2", "y2"), balls.xyxy[p].tolist(), strict=True)
                )
            )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ tables
def clip_bootstrap(
    clips: np.ndarray, numerator: np.ndarray, denominator: np.ndarray
) -> tuple[float, float]:
    """95% clip-bootstrap percentile interval of sum(numerator) / sum(denominator)."""
    codes, _ = pd.factorize(np.asarray(clips), sort=True)
    if not len(codes):
        return (np.nan, np.nan)
    n = np.bincount(codes, np.asarray(numerator, float), minlength=codes.max() + 1)
    d = np.bincount(codes, np.asarray(denominator, float), minlength=codes.max() + 1)
    rng = np.random.default_rng(BOOTSTRAP["seed"])
    draws = rng.integers(0, len(n), (BOOTSTRAP["draws"], len(n)))
    with np.errstate(invalid="ignore", divide="ignore"):
        ratios = n[draws].sum(axis=1) / d[draws].sum(axis=1)
    if np.isnan(ratios).all():
        return (np.nan, np.nan)
    low, high = np.nanquantile(ratios, [0.025, 0.975])
    return float(low), float(high)


def wilson(successes: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval of a proportion."""
    if not n:
        return (np.nan, np.nan)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def interval(table: pd.DataFrame, numerator, denominator, clustered: bool):
    if clustered:
        return clip_bootstrap(table["clip"].to_numpy(), numerator, denominator)
    return wilson(float(np.sum(numerator)), float(np.sum(denominator)))


def recall_table(
    table: pd.DataFrame, features=FEATURES, t: float = OPERATING, clustered: bool = True
) -> pd.DataFrame:
    """Per feature level: balls, share of balls, recall under both rules and the level's share of each rule's misses,
    with 95% intervals (clip bootstrap, or Wilson when ``clustered`` is false)."""
    records = []
    for feature in features:
        if feature not in table:
            continue
        levels = table[feature]
        for level in (
            levels.cat.categories if hasattr(levels, "cat") else sorted(levels.unique())
        ):
            inside = (levels == level).to_numpy()
            record = {
                "feature": feature,
                "level": str(level),
                "balls": int(inside.sum()),
                "share": inside.mean(),
            }
            for rule in ("iou", "centre"):
                hit = table[f"{rule}@{t:g}"].to_numpy()
                record[f"recall_{rule}"] = (
                    hit[inside].mean() if inside.any() else np.nan
                )
                record[f"recall_{rule}_ci"] = interval(
                    table, hit & inside, inside, clustered
                )
                record[f"miss_share_{rule}"] = (
                    (~hit & inside).sum() / (~hit).sum() if (~hit).any() else np.nan
                )
                record[f"miss_share_{rule}_ci"] = interval(
                    table, ~hit & inside, ~hit, clustered
                )
            records.append(record)
    return pd.DataFrame(records)


def cross_table(
    table: pd.DataFrame, rows: str, columns: str, t: float = OPERATING
) -> pd.DataFrame:
    """Balls and recall under both rules in every cell of ``rows`` x ``columns``."""
    cells = table.groupby([rows, columns], observed=True)
    return pd.DataFrame(
        {
            "balls": cells.size(),
            "recall_iou": cells[f"iou@{t:g}"].mean(),
            "recall_centre": cells[f"centre@{t:g}"].mean(),
        }
    ).reset_index()


def recall_curve(table: pd.DataFrame) -> pd.DataFrame:
    """Recall at every threshold under both rules: all balls and every size bin."""
    records = []
    groups = [("all", table)] + [
        (str(level), group) for level, group in table.groupby("size", observed=True)
    ]
    for name, group in groups:
        for t in THRESHOLDS:
            records.append(
                {
                    "group": name,
                    "conf": t,
                    "balls": len(group),
                    "recall_iou": group[f"iou@{t:g}"].mean(),
                    "recall_centre": group[f"centre@{t:g}"].mean(),
                }
            )
    return pd.DataFrame(records)


def bad_box(table: pd.DataFrame, t: float = OPERATING) -> dict:
    """Misses under IoU 0.5 that the centre rule hits ("bad box only"), as a share of the IoU misses."""
    iou, centre = table[f"iou@{t:g}"].to_numpy(), table[f"centre@{t:g}"].to_numpy()
    only = ~iou & centre
    return {
        "iou_misses": int((~iou).sum()),
        "bad_box_only": int(only.sum()),
        "share": float(only.sum() / (~iou).sum()) if (~iou).any() else np.nan,
        "centre_hit_but_iou_miss_of_all_balls": float(only.mean()),
    }


def miss_shares(
    table: pd.DataFrame, t: float = OPERATING, clustered: bool = True
) -> dict:
    """Decision inputs: shares of the misses (both rules) by blur proxies and by ball size < 12 / < 8 px."""
    flags = {"size<12": table.diameter < 12, "size<8": table.diameter < 8}
    if "speed_d" in table:
        flags |= {
            "blur (speed >= 1 d/frame, primary)": table.speed_d >= BLUR_SPEED,
            "speed >= 0.5 d/frame": table.speed_d >= 0.5,
            "speed >= 2 d/frame": table.speed_d >= 2,
            "blur or elongated": (table.speed_d >= BLUR_SPEED)
            | (table.elongation >= ELONGATED),
        }
    flags |= {
        "elongated >= 1.5": table.elongation >= ELONGATED,
        "Laplacian Q1 (blurriest quarter)": table.sharpness == "Q1 (blurriest)",
    }
    shares = {}
    for rule in ("iou", "centre"):
        miss = ~table[f"{rule}@{t:g}"].to_numpy()
        for name, flag in flags.items():
            flag = flag.to_numpy(bool)
            low, high = interval(table, miss & flag, miss, clustered)
            shares[f"{rule}: {name}"] = {
                "misses": int(miss.sum()),
                "in_flag": int((miss & flag).sum()),
                "share": float((miss & flag).sum() / miss.sum()),
                "ci": [low, high],
                "share_of_all_balls": float(flag.mean()),
            }
    return shares


def markdown(table: pd.DataFrame) -> str:
    """A table with numbers rounded and intervals as [low, high]."""
    shown = table.copy()
    for column in shown.columns:
        if column.endswith("_ci"):
            shown[column] = shown[column].map(lambda c: f"[{c[0]:.3f}, {c[1]:.3f}]")
        elif shown[column].dtype.kind == "f":
            shown[column] = shown[column].round(3)
    lines = [list(map(str, shown.columns))] + [
        list(map(str, r)) for r in shown.itertuples(index=False)
    ]
    lines.insert(1, ["---"] * len(shown.columns))
    return "\n".join("| " + " | ".join(line) + " |" for line in lines)


# ------------------------------------------------------------------ commands
def load_found(folder: Path) -> tuple[dict, pd.DataFrame, dict[tuple[str, str], dict]]:
    record, images, candidates = bench.read_dump(folder)
    found = {
        (set_name, row): bench.dump_detections(group)
        for (set_name, row), group in candidates.groupby(["set", "row"], observed=True)
    }
    return record, images, found


def check(args: argparse.Namespace) -> None:
    """The dump against the result files of the weights: detection metrics of every set and row, and placement."""
    record, _, found = load_found(args.dump)
    frozen = json.loads(DETECTION_VALID.read_text())
    if record["weights_sha256"] != frozen["weights_sha256"]:
        raise SystemExit(
            "the dump's weights are not the weights of detection_valid.json"
        )
    sets = bench.evaluation_sets("valid", list(bench.DUMP_SETS))
    report: dict = {"weights_sha256": record["weights_sha256"], "detection": {}}
    worst = 0.0
    for (set_name, row), predictions in sorted(found.items()):
        dataset, split = sets[set_name]
        if frozen["rows"][f"{set_name}/{row}"]["settings"] != record["rows"][row]:
            raise SystemExit(
                f"{set_name}/{row}: dump settings differ from detection_valid.json"
            )
        scored = bench.score_dump(dataset, split, predictions, set_name)
        reference = {
            c["class_name"]: c for c in frozen["rows"][f"{set_name}/{row}"]["per_class"]
        }
        differences = {
            f"{name}/{metric}": scored["per_class"][name][metric]
            - reference[name][metric]
            for name in reference
            for metric in ("map50", "map50_95", "precision", "recall")
        }
        differences["map50_95"] = (
            scored["map50_95"] - frozen["rows"][f"{set_name}/{row}"]["map50_95"]
        )
        worst = max(worst, max(abs(v) for v in differences.values()))
        report["detection"][f"{set_name}/{row}"] = {
            "ball_dump": scored["per_class"]["ball"],
            "ball_reference": {
                k: reference["ball"][k]
                for k in ("map50", "map50_95", "precision", "recall")
            },
            "max_abs_difference": max(abs(v) for v in differences.values()),
        }
    report["detection_max_abs_difference"] = worst

    placement_row = "e3"
    settings = record["rows"][placement_row]
    if (settings["imgsz"], settings["iou"], tuple(settings["input_size"])) != (
        bench.PLACEMENT["imgsz"],
        bench.PLACEMENT["iou"],
        tuple(bench.PLACEMENT["input_size"]),
    ):
        raise SystemExit("the e3 row of the dump is not the placement setting")
    size = bench.PLACEMENT["input_size"]
    top_ball = bench.dump_top_ball(
        found[("gsr", placement_row)], (1920 / size[0], 1080 / size[1])
    )
    rows = bench.placement_rows(top_ball, bench.frozen_homographies("valid"), "valid")
    reference = pd.read_parquet(PLACEMENT_FRAMES)
    joined = rows.merge(reference, on=["clip", "frame"], suffixes=("", "_ref"))
    both = np.isfinite(joined.error_m) & np.isfinite(joined.error_m_ref)
    report["placement"] = {
        "dump": placement_summary(rows),
        "reference": placement_summary(reference),
        "frames": len(rows),
        "frames_reference": len(reference),
        "same_placed_flag": float(
            (np.isfinite(joined.error_m) == np.isfinite(joined.error_m_ref)).mean()
        ),
        "max_abs_error_difference_m": float(
            np.abs(joined.error_m[both] - joined.error_m_ref[both]).max()
        ),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def placement_summary(rows: pd.DataFrame) -> dict:
    stats = bench.placement_statistics(rows)
    error = rows.error_m.to_numpy(float)
    return {
        "placed": stats["placed"],
        "far_of_placed": stats["far"],
        "well_placed": float((error <= bench.PLACEMENT["far_m"]).mean()),
        "well_placed_with_homography": float(
            (error[rows.homography.to_numpy()] <= bench.PLACEMENT["far_m"]).mean()
        ),
    }


def analyse(args: argparse.Namespace) -> None:
    """Features, recall tables, miss shares, bad-box share and the recall curve for both rows."""
    _, images, found = load_found(args.dump)
    gsr = bench.gsr_split("valid")
    bench.check_manifest(gsr, "valid")
    out = args.out / "tables"
    out.mkdir(parents=True, exist_ok=True)
    truths = {
        "gsr": gsr_balls(gsr),
        "ekstraklasa": ekstraklasa_balls(bench.ekstraklasa_dev()),
    }
    summary: dict = {
        "selection_games": SELECTION_GAMES,
        "settings": {
            "operating_conf": OPERATING,
            "min_radius_px": MIN_RADIUS,
            "blur_speed_diameters_per_frame": BLUR_SPEED,
            "elongated": ELONGATED,
            "occluded": OCCLUDED,
            "contact_m": CONTACT_M,
            "in_air": f"every foot >= {FREE_M} m and >= {AIR_DEVIATION_M} m off the midpoint of "
            f"frames f +- {AIR_SPAN}",
            "bootstrap": BOOTSTRAP,
        },
    }
    report = [
        "# ball-detection 01: recall tables (agent proxies, see ball_errors.py)\n"
    ]
    for set_name, truth in truths.items():
        clustered = set_name == "gsr"
        for row in ROWS:
            table = candidate_columns(truth, found[(set_name, row)])
            name = f"{set_name}_{row}"
            table.to_parquet(out / f"balls_{name}.parquet", index=False)
            recall = recall_table(table, clustered=clustered)
            recall.to_csv(out / f"recall_{name}.csv", index=False)
            curve = recall_curve(table)
            curve.to_csv(out / f"curve_{name}.csv", index=False)
            summary[name] = {
                "balls": len(table),
                "recall_iou": float(table[f"iou@{OPERATING:g}"].mean()),
                "recall_centre": float(table[f"centre@{OPERATING:g}"].mean()),
                "bad_box": bad_box(table),
                "miss_shares": miss_shares(table, clustered=clustered),
                "candidate_levels_of_iou_misses": table.loc[
                    ~table[f"iou@{OPERATING:g}"], "candidate"
                ]
                .value_counts()
                .to_dict(),
            }
            report += [f"\n## {name}: {len(table)} balls\n", markdown(recall), ""]
            for rows_feature, columns_feature in CROSSES:
                if {rows_feature, columns_feature} <= set(table.columns):
                    cross = cross_table(table, rows_feature, columns_feature)
                    cross.to_csv(
                        out / f"cross_{name}_{rows_feature}_x_{columns_feature}.csv",
                        index=False,
                    )
                    report += [
                        f"\n### {name}: {rows_feature} x {columns_feature}\n",
                        markdown(cross),
                        "",
                    ]
            report += ["\n### recall vs conf\n", markdown(curve), ""]
        if set_name == "gsr":
            selection = images[
                (images.set == "gsr") & images.game.isin(SELECTION_GAMES)
            ]
            people = selection_people(sorted(selection["clip"].unique()))
            for row in ROWS:
                falses = false_ball_table(
                    selection.image.tolist(), truth, found[("gsr", row)], people
                )
                falses.to_parquet(out / f"false_balls_gsr_{row}.parquet", index=False)
                summary[f"gsr_{row}"]["false_balls"] = {
                    "count": len(falses),
                    "frames": int(falses.image.nunique()),
                    "frames_scored": len(selection),
                    "on_frames_without_truth_ball": int(falses.truth_px.isna().sum()),
                    "kinds": falses.kind.value_counts().to_dict(),
                    "kinds_by_game": falses.merge(
                        selection[["image", "game"]], on="image"
                    )
                    .groupby("game")
                    .kind.value_counts()
                    .unstack(fill_value=0)
                    .to_dict(orient="index"),
                }
            summary["placement_far_e3"] = placement_far(
                truth, found[("gsr", "e3")], people
            )
    (out / "tables.md").write_text("\n".join(report) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    curve_figure(
        pd.read_csv(out / "curve_gsr_1280.csv"),
        args.out / "recall_vs_conf_gsr_1280.png",
    )
    print(json.dumps(summary, indent=2, default=str))


def placement_far(
    truth: pd.DataFrame,
    found: dict[str, sv.Detections],
    people: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict:
    """What the placed (e3 top-1) ball is on the selection frames placed > 5 m off: ``candidate_kind`` counts."""
    placement = pd.read_parquet(PLACEMENT_FRAMES)
    placement = placement[placement.game.isin(SELECTION_GAMES)]
    far = placement[placement.error_m > bench.PLACEMENT["far_m"]]
    kinds = []
    for clip, frame in zip(far["clip"], far["frame"], strict=True):
        image = f"{clip}/{frame:06d}"
        balls = ball_predictions(found.get(image))
        balls = balls[balls.confidence >= bench.PLACEMENT["ball_conf"]]
        box = balls.xyxy[int(np.argmax(balls.confidence))]
        boxes = truth.loc[truth.image == image, ["x1", "y1", "x2", "y2"]].to_numpy(
            float
        )
        person_boxes, roles = people.get(image, (np.zeros((0, 4)), np.zeros(0, str)))
        kinds.append(candidate_kind(box, boxes, person_boxes, roles)[0])
    table = far.assign(kind=kinds)
    return {
        "frames": len(placement),
        "placed": int(np.isfinite(placement.error_m).sum()),
        "far": len(far),
        "kinds": table.kind.value_counts().to_dict(),
        "kinds_by_game": table.groupby("game")
        .kind.value_counts()
        .unstack(fill_value=0)
        .to_dict(orient="index"),
    }


def curve_figure(curve: pd.DataFrame, path: Path) -> None:
    """Recall against the ball threshold, both rules, all balls and every size bin (small, for results.md)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ramp = {
        "<8": "#86b6ef",
        "8-12": "#5598e7",
        "12-16": "#2a78d6",
        "16-24": "#1c5cab",
        ">=24": "#104281",
    }
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True, dpi=110)
    for axis, rule, title in zip(
        axes, ("iou", "centre"), ("IoU >= 0.5", "centre <= r"), strict=True
    ):
        for group, colour in [*ramp.items(), ("all", "#1f1f1e")]:
            part = curve[curve.group == group]
            axis.plot(
                part.conf,
                part[f"recall_{rule}"],
                color=colour,
                linewidth=2.6 if group == "all" else 2,
                label=f"{group} px" if group != "all" else "all balls",
            )
        axis.axvline(OPERATING, color="#9a9a96", linewidth=1, linestyle=":")
        axis.set_xscale("log")
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("ball conf threshold")
        axis.grid(alpha=0.25, linewidth=0.6)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    axes[0].set_ylabel("recall (GSR valid games 2+3, row 1280)")
    axes[0].set_ylim(0, 1)
    axes[1].legend(
        fontsize=8,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        title="ball size",
    )
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


# ------------------------------------------------------------------ gallery
def gallery(args: argparse.Namespace) -> None:
    """50 seeded false-ball frames of games 2+3: 25 whose placed top-1 ball (e3) is > 5 m off, 25 with a false ball
    >= 0.15 (row 1280, centre rule); one crop sheet per 5 frames. Also a blur-proxy check sheet."""
    _, images, found = load_found(args.dump)
    tables = args.out / "tables"
    placement = pd.read_parquet(PLACEMENT_FRAMES)
    far = placement[
        placement.game.isin(SELECTION_GAMES)
        & (placement.error_m > bench.PLACEMENT["far_m"])
    ]
    falses = pd.read_parquet(tables / "false_balls_gsr_1280.parquet")
    rng = np.random.default_rng(SAMPLE["seed"])
    chosen_far = far.iloc[
        np.sort(rng.choice(len(far), SAMPLE["per_source"], replace=False))
    ]
    picked = []
    size = bench.PLACEMENT["input_size"]
    scale = (1920 / size[0], 1080 / size[1])
    for _, frame in chosen_far.iterrows():
        image = f"{frame['clip']}/{frame['frame']:06d}"
        balls = ball_predictions(found[("gsr", "e3")].get(image))
        balls = balls[balls.confidence >= bench.PLACEMENT["ball_conf"]]
        best = int(np.argmax(balls.confidence))
        picked.append(
            {
                "source": "placement>5m (e3)",
                "image": image,
                "conf": float(balls.confidence[best]),
                "error_m": float(frame.error_m),
            }
            | dict(
                zip(("x1", "y1", "x2", "y2"), balls.xyxy[best].tolist(), strict=True)
            )
        )
    taken = {p["image"] for p in picked}
    strongest = (
        falses[~falses.image.isin(taken)]
        .sort_values("conf", ascending=False)
        .drop_duplicates("image")
        .sort_values("image", ignore_index=True)
    )
    chosen = strongest.iloc[
        np.sort(rng.choice(len(strongest), SAMPLE["per_source"], replace=False))
    ]
    for _, false in chosen.iterrows():
        picked.append(
            {
                "source": "false ball>=0.15 (1280)",
                "image": false.image,
                "conf": false.conf,
            }
            | {k: false[k] for k in ("x1", "y1", "x2", "y2")}
        )
    sample = pd.DataFrame(picked)
    sample.insert(0, "id", range(1, len(sample) + 1))
    truth = pd.read_parquet(tables / "balls_gsr_1280.parquet")
    folder = args.out / "gallery"
    folder.mkdir(parents=True, exist_ok=True)
    paths = images.set_index("image").path
    sample["class"] = ""
    sample["note"] = ""
    sample.to_csv(folder / "false_balls.csv", index=False)
    tiles = [
        tile(
            cv2.imread(paths[r.image]),
            np.array([r.x1, r.y1, r.x2, r.y2]),
            truth.loc[truth.image == r.image, ["x1", "y1", "x2", "y2"]].to_numpy(float),
            f"#{r.id} {r.source} conf {r.conf:.2f} {r.image}",
        )
        for r in sample.itertuples()
    ]
    for start in range(0, len(tiles), 5):
        cv2.imwrite(
            str(folder / f"sheet_{start // 5 + 1:02d}.jpg"),
            np.vstack(tiles[start : start + 5]),
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        )
    blur_check(truth, paths, folder)
    other_false_balls(falses, paths, folder)
    logger.info("gallery: %s (%d frames, scale %s)", folder, len(sample), scale)


def tile(
    image: np.ndarray, box: np.ndarray, truth: np.ndarray, title: str
) -> np.ndarray:
    """Context (the frame at 768 x 432, red ring: candidate, green: truth balls) beside a 3x crop of 144 px."""
    context = cv2.resize(image, (768, 432), interpolation=cv2.INTER_AREA)
    factor = 768 / image.shape[1]
    for boxes, colour in ((truth, (60, 200, 60)), (box[None], (40, 40, 230))):
        for centre in centres(boxes):
            cv2.circle(context, tuple(int(v) for v in centre * factor), 14, colour, 2)
    cx, cy = centres(box[None])[0]
    half = 72
    x0 = int(np.clip(cx - half, 0, image.shape[1] - 2 * half))
    y0 = int(np.clip(cy - half, 0, image.shape[0] - 2 * half))
    crop = cv2.resize(
        image[y0 : y0 + 2 * half, x0 : x0 + 2 * half],
        (432, 432),
        interpolation=cv2.INTER_CUBIC,
    )
    for boxes, colour in ((truth, (60, 200, 60)), (box[None], (40, 40, 230))):
        for b in boxes:
            corners = (np.asarray(b) - [x0, y0, x0, y0]) * 3 + [-6, -6, 6, 6]
            cv2.rectangle(
                crop,
                tuple(map(int, corners[:2])),
                tuple(map(int, corners[2:])),
                colour,
                1,
            )
    sheet = np.hstack([context, crop])
    banner = np.full((28, sheet.shape[1], 3), 255, np.uint8)
    cv2.putText(
        banner,
        title,
        (6, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([banner, sheet])


def crop_grid(boxes: pd.DataFrame, paths: pd.Series, label: str) -> np.ndarray:
    """Two rows of 12 crops (48 px around each box centre, 3x) with ``label`` (a column) printed on each."""
    crops = []
    for r in boxes.itertuples():
        image = cv2.imread(paths[r.image])
        cx, cy = (r.x1 + r.x2) / 2, (r.y1 + r.y2) / 2
        x0 = int(np.clip(cx - 24, 0, image.shape[1] - 48))
        y0 = int(np.clip(cy - 24, 0, image.shape[0] - 48))
        crop = cv2.resize(
            image[y0 : y0 + 48, x0 : x0 + 48], (144, 144), interpolation=cv2.INTER_CUBIC
        )
        text = f"{getattr(r, label):.2f}"
        cv2.putText(
            crop, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1
        )
        crops.append(crop)
    return np.vstack([np.hstack(crops[i : i + 12]) for i in (0, 12)])


def blur_check(truth: pd.DataFrame, paths: pd.Series, folder: Path) -> None:
    """Crops of 24 seeded balls at speed >= 1 diameter/frame and 24 at < 0.3, to eyeball the blur proxy."""
    rng = np.random.default_rng(SAMPLE["seed"])
    rows = []
    for flag in (truth.speed_d >= BLUR_SPEED, truth.speed_d < 0.3):
        part = truth[flag.to_numpy()]
        rows.append(
            crop_grid(
                part.iloc[np.sort(rng.choice(len(part), 24, replace=False))],
                paths,
                "speed_d",
            )
        )
    cv2.imwrite(
        str(folder / "blur_check.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92]
    )


def other_false_balls(falses: pd.DataFrame, paths: pd.Series, folder: Path) -> None:
    """Crops of 24 seeded false balls of kind ``other`` (not on a truth ball or a person box): the ball-sized false
    balls that the 50-frame sample, one strongest false ball per frame, under-represents."""
    rng = np.random.default_rng(SAMPLE["seed"])
    other = falses[falses.kind == "other"].reset_index(drop=True)
    chosen = other.iloc[np.sort(rng.choice(len(other), 24, replace=False))]
    chosen.to_csv(folder / "other_false_balls.csv", index=False)
    cv2.imwrite(
        str(folder / "other_false_balls.jpg"),
        crop_grid(chosen, paths, "conf"),
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )


def taxonomy(args: argparse.Namespace) -> None:
    """Shares (Wilson 95%) of the agent's classes (agent proxy, not human labels): the 50-frame sample per source and
    together, and the supplementary 24 ``other``-kind false balls."""
    folder = args.out / "gallery"
    labels = pd.read_csv(folder / "false_balls_labelled.csv")
    supplement = pd.read_csv(folder / "other_false_balls_labelled.csv").assign(
        source="supplement: other-kind false ball>=0.15 (1280)"
    )
    unknown = set(labels["class"]) | set(supplement["class"])
    unknown -= set(FALSE_CLASSES)
    if unknown:
        raise SystemExit(f"unknown classes {sorted(unknown)}; known: {FALSE_CLASSES}")
    records = []
    groups = [("sample of 50", labels), *labels.groupby("source")]
    for source, group in [*groups, (supplement.source[0], supplement)]:
        for name in FALSE_CLASSES:
            k = int((group["class"] == name).sum())
            low, high = wilson(k, len(group))
            records.append(
                {
                    "source": source,
                    "class": name,
                    "count": k,
                    "of": len(group),
                    "share": k / len(group),
                }
                | {"ci_low": low, "ci_high": high}
            )
    table = pd.DataFrame(records)
    table.to_csv(args.out / "tables/false_ball_taxonomy.csv", index=False)
    print(markdown(table))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "analyse", "gallery", "taxonomy"):
        command = commands.add_parser(name)
        command.add_argument("--out", type=Path, default=OUT)
        if name != "taxonomy":
            command.add_argument("--dump", type=Path, default=DUMP)
    args = parser.parse_args()
    tv.setup_logging("INFO")
    {"check": check, "analyse": analyse, "gallery": gallery, "taxonomy": taxonomy}[
        args.command
    ](args)


if __name__ == "__main__":
    sys.exit(main())
