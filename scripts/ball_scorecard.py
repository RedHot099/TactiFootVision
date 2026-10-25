"""ball-detection 02: the ball scorecard every later ticket of the line is read with, from weights or a dump.

    uv run python scripts/ball_scorecard.py card --weights W --out OUT [--confirm]  # dump W (GPU), then score it
    uv run python scripts/ball_scorecard.py card --dump DUMP --out OUT [--confirm]  # score a dump (ticket 01 format)
    uv run python scripts/ball_scorecard.py agree OUT_A OUT_B                       # max |difference| of two cards
    uv run python scripts/ball_scorecard.py references                              # reference rows + file check

`card` scores every row of a prediction dump (`detector_bench.py dump`: every candidate of every class at conf
>= 0.001, boxes in the images' pixels; tiles, flip TTA or a tracker write the same files) and writes
`scorecard.json` and `scorecard.md`. With `--weights` it dumps the weights first (rows `--rows`, default 1280 and e3)
into `OUT/dump`. Per GSR valid split (`selection` = games 2 and 3, always; `confirm` = game 5 and `full` = all
three only with `--confirm`, the confirm read of a ticket winner or the dd2c reference row) and per row:

- detection (`detector_bench.image_scores`, exactly `evaluate`'s numbers): ball AP50 / AP50-95, ball precision /
  recall at the operating point (ball conf 0.15, IoU >= 0.5), people mAP50-95 (mean AP50-95 of goalkeeper, player,
  referee: the no-regression barrier), all-class mAP;
- ball hits under both rules of ticket 01 (`ball_errors.match`: IoU >= 0.5, centre distance <= max(5 px, diameter))
  at 0.15 and at 0.001, false balls >= 0.15 (unmatched under the centre rule), and recall in ticket 01's bins
  (`ball_errors.gsr_balls` features; truth tables cached in `TRUTH`);
- placement on the frozen homographies (`detector_bench.placement_rows`; the most confident ball >= 0.15 of the row,
  its centre through the frame's homography): frames (stride-5 frames with exactly one ball annotation on the pitch:
  6,404 on the full split, 426 of them without a homography), ceiling, placed, well placed (<= 5 m) among all frames
  (the main measure) and among frames with a homography, median error, > 5 m among placed; all frames and near goal
  (truth <= 30 m from a goal line).

Plus Ekstraklasa dev per row (ball AP50 / AP50-95, mAP50-95; its 3 clips make the interval a direction only) and the
cost (`ms_per_frame` of the dump record: the model call per image on this GPU; a dump without it gets the dump's wall
seconds per image, image read included, flagged as such).

Intervals: 95% percentile clip bootstrap, 2,000 draws, seed 0 (`np.random.default_rng(0).integers(0, clips,
(2000, clips))` over the split's clips in name order); the same draws for every number of a split, so that a paired
difference (variant minus the reference, frame-paired: both on the same frames and draws) is the percentile interval
of the per-draw differences. The reference is dd2c's ticket 01 dump (`--reference`); a row is paired with the
reference row of the same name, else with a single-row reference's only row (e.g. tiles + filter against dd2c 1280 +
filter), else with the reference's 1280 row (e.g. a 1920 variant against dd2c 1280).
Bootstrap APs are supervision's AP on the drawn images (`resampled_ap`), equal to `evaluate`'s on the original set.

`references` dumps and cards the reference rows (dd2c from its weights, run 1 best, the base model) and checks
them against their result files (`REFERENCES`); for run 1 and the base model the whole-valid numbers needed for that
check are compared and only their differences kept (game 5 is read only for dd2c, spec "Protokół oceny").
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ball_errors as be  # noqa: E402
import detector_bench as bench  # noqa: E402

import tactifoot_vision as tv  # noqa: E402

ROOT = be.ROOT
OUT = ROOT / "ball/02"
TRUTH = OUT / "truth"
DD2C_DUMP = (
    be.DUMP
)  # ticket 01: dd2c, rows 1280 and e3, checked against the result files
SPLITS = {"selection": ("2", "3"), "confirm": ("5",), "full": ("2", "3", "5")}
PEOPLE = ("goalkeeper", "player", "referee")
BALL_CONF = bench.OPERATING["class_conf"]["ball"]
BIN_FEATURES = tuple(
    f for f in be.FEATURES if f != "candidate"
)  # truth properties only
BOOTSTRAP = be.BOOTSTRAP
IOU_LEVELS = 10
# The selection rule (ticket 02 comment, 2026-10-24), fixed after the dd2c 1280 row and before any variant.
RULE = {
    "written": "2026-10-24",
    "primary": "paired gain over dd2c in well placed (all frames), row 1280, selection split: 95% CI low > 0",
    # dd2c 1280 well placed on the full GSR valid (0.6018, ticket 02) + 0.08 (ambitious + 0.13)
    "gate_1280": {"satisfactory": 0.6818, "ambitious": 0.7318},
    "gate_e3": {"satisfactory": 0.70, "ambitious": 0.75},
    "ball_ap50_min": 0.45,
    "people_map50_95_drop_max": 0.01,
    "ekstraklasa_map50_95_drop_max": 0.02,
    "ekstraklasa_ball_ap50_direction": 0.40,
    "post_filter_long_side_px": 45,  # ticket 03's registered variant: drop ball candidates longer (1080p px)
}
CHECKOUT = bench.CHECKOUT
REFERENCES = {
    "dd2c": {
        "weights": CHECKOUT / "models/detector_yolo11l_1280_dd2c.pt",
        "detection": ROOT / "final/candidate_valid/detection_valid.json",
        "ekstraklasa": ROOT / "final/candidate_valid/detection_valid.json",
        "placement": ROOT / "runs/compare/placement_run2c_epoch1/placement.json",
    },
    "run1_best": {
        "weights": ROOT / "runs/run1/yolo11l_1280/weights/best.pt",
        "detection": ROOT / "runs/run1/eval/detection_valid.json",
        "ekstraklasa": ROOT / "runs/run1/eval/ekstraklasa_dev/detection_valid.json",
        "placement": ROOT / "runs/run1/eval/placement/placement.json",
    },
    "base": {
        "weights": CHECKOUT
        / "results/project/raw/yolo11m/training/detect_yolo11m/weights/best.pt",
        "detection": ROOT / "baseline/detection_valid.json",
        "ekstraklasa": ROOT / "baseline/ekstraklasa_dev/detection_valid.json",
        "placement": ROOT / "baseline/placement/placement.json",
    },
}

logger = logging.getLogger("ball_scorecard")


# ------------------------------------------------------------------ dump
def check_dump(record: dict) -> None:
    """The dump settings a card depends on: candidates down to ``EVAL_CONF``, ``max_det`` and the class mapping
    recorded, and the known rows with their benchmark settings (NMS IoU 0.7 at 1280, 0.5 at e3)."""
    if record["conf"] > bench.EVAL_CONF:
        raise SystemExit(
            f"dump conf {record['conf']} > {bench.EVAL_CONF}: AP needs the tail"
        )
    for key in ("max_det", "classes"):
        if key not in record:
            raise SystemExit(f"the dump record has no {key!r}")
    for row, settings in record["rows"].items():
        if row in bench.ROWS and _settings(settings) != _settings(bench.ROWS[row]):
            raise SystemExit(
                f"dump row {row} {settings} is not the bench's {bench.ROWS[row]}"
            )


def _settings(settings: dict) -> tuple:
    size = settings.get("input_size")
    return settings["imgsz"], settings["iou"], tuple(size) if size else None


def dump_class_map(record: dict, set_name: str, dataset_classes: list[str]) -> dict:
    """The dump's model class -> benchmark class mapping (by name; recorded since ticket 02, derived for older
    dumps)."""
    recorded = record["sets"].get(set_name, {}).get("class_map")
    return recorded or bench.class_map(
        list(record["classes"].values()), dataset_classes
    )


def cost(record: dict, images: pd.DataFrame) -> dict:
    """ms per frame of every set and row of a dump: the model call (``ms_per_frame``), else wall seconds per image."""
    if "ms_per_frame" in record:
        return {
            "ms_per_frame": record["ms_per_frame"],
            "gpu": record.get("gpu"),
            "kind": "model call per image (resize, inference, NMS), batch 1",
        }
    counts = images.groupby("set").size()
    return {
        "ms_per_frame": {
            key: round(1000 * s / counts[key.split("/")[0]], 2)
            for key, s in record["seconds"].items()
        },
        "gpu": None,
        "kind": "dump wall time per image (image read included): no model-call timing in this dump",
    }


# ------------------------------------------------------------------ bootstrap
class Draws:
    """The clip-bootstrap draws of one split: ``weights[d, c]`` = how often draw ``d`` takes clip ``c`` (clips in
    name order), from ``ball_errors.BOOTSTRAP``'s generator."""

    def __init__(self, clips) -> None:
        self.clips = np.array(sorted(set(clips)))
        n = len(self.clips)
        rng = np.random.default_rng(BOOTSTRAP["seed"])
        picks = rng.integers(0, n, (BOOTSTRAP["draws"], n))
        self.weights = np.zeros((BOOTSTRAP["draws"], n), np.int64)
        np.add.at(self.weights, (np.arange(BOOTSTRAP["draws"])[:, None], picks), 1)

    def codes(self, clips) -> np.ndarray:
        clips = np.asarray(clips)
        codes = np.minimum(np.searchsorted(self.clips, clips), len(self.clips) - 1)
        if len(clips) and not np.array_equal(self.clips[codes], clips):
            raise ValueError("clips outside the split's draws")
        return codes

    def per_clip(self, clips, values) -> np.ndarray:
        return np.bincount(
            self.codes(clips), np.asarray(values, float), minlength=len(self.clips)
        )


def interval(values: np.ndarray) -> list[float]:
    values = np.asarray(values, float)
    if np.isnan(values).all():
        return [np.nan, np.nan]
    return np.nanquantile(values, [0.025, 0.975]).tolist()


def ratio_draws(draws: Draws, numerator: np.ndarray, denominator: np.ndarray):
    """Point value and per-draw values of sum(numerator) / sum(denominator) (per-clip sums)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        point = numerator.sum() / denominator.sum() if denominator.sum() else np.nan
        return float(point), (draws.weights @ numerator) / (draws.weights @ denominator)


def resampled_ap(
    scores: bench.ImageScores, keep: np.ndarray, draws: Draws
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per class with targets on ``keep``: ``(point AP per IoU level, per-draw APs (draws x levels), per-draw
    flag that the draw has targets)``.

    A draw repeats each image as often as its clip is drawn, copies next to each other in supervision's
    confidence order of the original images, and scores them as supervision does (``_average_precisions_per_class``:
    101-point AP of the cumulative precision / recall), so the point value is `evaluate`'s AP."""
    chosen = [
        (s, clip)
        for s, clip, k in zip(scores.stats, scores.keys["clip"], keep, strict=True)
        if k and s is not None
    ]
    matches, confidence, class_ids, target_ids = (
        np.concatenate([s[i] for s, _ in chosen]) for i in range(4)
    )
    image_clip = draws.codes([c for _, c in chosen])
    pred_clip = np.repeat(image_clip, [len(s[1]) for s, _ in chosen])
    target_clip = np.repeat(image_clip, [len(s[3]) for s, _ in chosen])
    order = np.argsort(
        -confidence
    )  # supervision's (unstable) order, for its exact point value
    matches, class_ids, pred_clip = matches[order], class_ids[order], pred_clip[order]
    ones = np.ones((1, len(draws.clips)), np.int64)
    result = {}
    for class_id in np.unique(target_ids):
        is_class = class_ids == class_id
        true = np.bincount(
            target_clip[target_ids == class_id], minlength=len(draws.clips)
        )
        aps = [
            weighted_ap(matches[is_class], pred_clip[is_class], true, w)
            for w in np.vstack([ones, draws.weights])
        ]
        has = np.vstack([ones, draws.weights]) @ true > 0
        result[int(class_id)] = (aps[0], np.asarray(aps[1:]), has[1:])
    return result


def weighted_ap(
    matches: np.ndarray, clip: np.ndarray, true: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """supervision's AP per IoU level of predictions (``matches`` sorted by confidence) repeated ``weights[clip]``
    times against ``weights @ true`` targets."""
    total = int(weights @ true)
    repeated = np.repeat(np.arange(len(matches)), weights[clip])
    if not total or not len(repeated):
        return np.zeros(matches.shape[1] if matches.ndim == 2 else IOU_LEVELS)
    hits = matches[repeated]
    tp = hits.cumsum(0)
    fp = (1 - hits).cumsum(0)
    recall = tp / (total + 1e-16)
    precision = tp / (tp + fp)
    levels = np.linspace(0, 1, 101)
    ap = np.zeros(hits.shape[1])
    for level in range(hits.shape[1]):
        first = np.searchsorted(recall[:, level], levels, side="left")
        inside = first < len(recall)
        values = np.where(
            inside, precision[np.minimum(first, len(recall) - 1), level], 0
        )
        ap[level] = (1 / 101 * values).sum()
    return ap


def detection_draws(scores: bench.ImageScores, keep: np.ndarray, draws: Draws) -> dict:
    """Point and per-draw values of the card's detection numbers on the images ``keep``."""
    names = scores.class_names
    aps = resampled_ap(scores, keep, draws)
    found: dict[str, tuple[float, np.ndarray]] = {}
    ball = names.index("ball")
    if ball in aps:
        point, draw, _ = aps[ball]
        found["ball_ap50"] = (float(point[0]), draw[:, 0])
        found["ball_ap50_95"] = (float(point.mean()), draw.mean(axis=1))
    people = [names.index(n) for n in PEOPLE if names.index(n) in aps]
    if people:
        found["people_map50_95"] = (
            float(np.mean([aps[c][0].mean() for c in people])),
            np.mean([aps[c][1].mean(axis=1) for c in people], axis=0),
        )
    # supervision's mAP: the mean over the classes that have targets in the (drawn) images
    point = np.mean([aps[c][0].mean() for c in aps])
    stacked = np.array([aps[c][1].mean(axis=1) for c in aps])
    has = np.array([aps[c][2] for c in aps])
    with np.errstate(invalid="ignore"):
        found["map50_95"] = (float(point), (stacked * has).sum(0) / has.sum(0))
    clips = scores.keys["clip"][keep]
    counts = scores.counts[keep][:, ball]
    tp, fp, fn = (draws.per_clip(clips, counts[:, i]) for i in range(3))
    found["ball_precision"] = ratio_draws(draws, tp, tp + fp)
    found["ball_recall"] = ratio_draws(draws, tp, tp + fn)
    return found


def summary(found: dict[str, tuple[float, np.ndarray]]) -> dict:
    return {k: {"value": v, "ci": interval(d)} for k, (v, d) in found.items()}


def paired(
    variant: dict[str, tuple[float, np.ndarray]],
    reference: dict[str, tuple[float, np.ndarray]],
) -> dict:
    """Variant minus reference for every shared number: point and the interval of the per-draw differences."""
    return {
        k: {
            "variant": variant[k][0],
            "reference": reference[k][0],
            "difference": variant[k][0] - reference[k][0],
            "ci": interval(variant[k][1] - reference[k][1]),
        }
        for k in variant
        if k in reference
    }


# ------------------------------------------------------------------ placement
def split_top_ball(found: dict[str, sv.Detections], scale, clips: set[str]):
    """``dump_top_ball`` asked only on the split's clips (other games are not read)."""
    top_ball = bench.dump_top_ball(found, scale)

    def top(clip: str, frame: int, path: Path):
        return top_ball(clip, frame, path) if clip in clips else None

    return top


def placement_table(
    found: dict[str, sv.Detections], images: pd.DataFrame, homographies: dict, games
) -> pd.DataFrame:
    """`placement_rows` of the frames of ``games`` with the dump row's top ball (boxes in the images' pixels,
    scaled to the homographies' ``PLACEMENT["input_size"]``)."""
    gsr = images[images.set == "gsr"]
    sizes = {(int(w), int(h)) for w, h in zip(gsr.width, gsr.height, strict=True)}
    if len(sizes) != 1:
        raise SystemExit(f"GSR images of several sizes {sizes}")
    (width, height), size = sizes.pop(), bench.PLACEMENT["input_size"]
    clips = set(gsr.loc[gsr.game.isin(games), "clip"])
    rows = bench.placement_rows(
        split_top_ball(found, (width / size[0], height / size[1]), clips),
        homographies,
        "valid",
    )
    return rows[rows.game.isin(games)].reset_index(drop=True)


def placement_draws(rows: pd.DataFrame, draws: Draws) -> dict:
    """Point and per-draw values of the placement numbers of ``rows`` (all frames, then near goal)."""
    found = {}
    for prefix, part in (("", rows), ("near_goal_", rows[rows.near_goal])):
        error = part.error_m.to_numpy(float)
        homography = part.homography.to_numpy(bool)
        placed = np.isfinite(error)
        well = error <= bench.PLACEMENT["far_m"]
        far = error > bench.PLACEMENT["far_m"]
        ones = np.ones(len(part))
        clip = part["clip"]
        flags = {
            "well_placed": (well, ones),
            "well_placed_with_homography": (well & homography, homography),
            "placed": (placed, ones),
            "far_of_placed": (far, placed),
            "ceiling": (homography, ones),
        }
        for name, (numerator, denominator) in flags.items():
            found[prefix + name] = ratio_draws(
                draws,
                draws.per_clip(clip, numerator),
                draws.per_clip(clip, denominator),
            )
        codes = draws.codes(clip[placed])
        errors = error[placed]
        medians = np.array(
            [
                np.median(np.repeat(errors, w[codes])) if w[codes].sum() else np.nan
                for w in draws.weights
            ]
        )
        found[prefix + "median_m"] = (float(np.median(errors)), medians)
    return found


def placement_counts(rows: pd.DataFrame) -> dict:
    return {
        "frames": len(rows),
        "without_homography": int((~rows.homography).sum()),
        "near_goal_frames": int(rows.near_goal.sum()),
        "clips": int(rows["clip"].nunique()),
    }


# ------------------------------------------------------------------ hits
def truth_table(gsr: tv.data.Dataset, games) -> pd.DataFrame:
    """``ball_errors.gsr_balls`` of ``games`` (per truth ball: box and ticket 01's features), cached in ``TRUTH``."""
    path = TRUTH / f"balls_gsr_games_{''.join(games)}.parquet"
    if path.is_file():
        return pd.read_parquet(path)
    table = be.gsr_balls(gsr, games)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    return pd.read_parquet(path)


def hit_draws(table: pd.DataFrame, draws: Draws) -> dict:
    """Ball recall under both rules at the operating point and at the dump's floor (point and per-draw)."""
    found = {}
    ones = draws.per_clip(table["clip"], np.ones(len(table)))
    for rule in ("iou", "centre"):
        for t in (BALL_CONF, be.THRESHOLDS[0]):
            hit = draws.per_clip(table["clip"], table[f"{rule}@{t:g}"])
            found[f"recall_{rule}@{t:g}"] = ratio_draws(draws, hit, ones)
    return found


def false_ball_draws(
    found: dict[str, sv.Detections],
    keys: pd.DataFrame,
    truth: pd.DataFrame,
    draws: Draws,
) -> dict:
    """False balls >= the operating point (``ball_errors.false_balls``: no truth ball within r) per frame and as a
    share of the kept balls (1 - precision under the centre rule)."""
    boxes = {
        image: group[["x1", "y1", "x2", "y2"]].to_numpy(float)
        for image, group in truth.groupby("image")
    }
    false, kept = [], []
    for image in keys["image"]:
        balls = be.ball_predictions(found.get(image))
        false.append(
            len(
                be.false_balls(
                    balls.xyxy, balls.confidence, boxes.get(image, np.zeros((0, 4)))
                )
            )
        )
        kept.append(int((balls.confidence >= BALL_CONF).sum()))
    clip = keys["clip"]
    false_sum = draws.per_clip(clip, false)
    return {
        "false_balls_per_frame": ratio_draws(
            draws, false_sum, draws.per_clip(clip, np.ones(len(keys)))
        ),
        "false_share_of_kept_balls": ratio_draws(
            draws, false_sum, draws.per_clip(clip, kept)
        ),
    }


def bins(table: pd.DataFrame) -> list[dict]:
    """Recall at the operating point in ticket 01's bins (both rules, clip-bootstrap intervals)."""
    recall = be.recall_table(table, features=BIN_FEATURES)
    columns = ["feature", "level", "balls", "share"] + [
        f"recall_{rule}{suffix}" for rule in ("iou", "centre") for suffix in ("", "_ci")
    ]
    records = recall[columns].to_dict(orient="records")
    for record in records:
        for key in ("recall_iou_ci", "recall_centre_ci"):
            record[key] = list(record[key])
    return records


# ------------------------------------------------------------------ card
class Model:
    """One dump read for the card: its record, images and predictions per (set, row)."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.record, self.images, self.found = be.load_found(folder)
        check_dump(self.record)

    @property
    def rows(self) -> list[str]:
        return list(self.record["rows"])


def pair_row(row: str, reference: Model) -> str:
    """The reference row ``row`` is paired with: the same name, else the reference's only row, else its 1280 row."""
    if row in reference.rows:
        return row
    return reference.rows[0] if len(reference.rows) == 1 else "1280"


def subset(dataset: tv.data.Dataset, games) -> tv.data.Dataset:
    return dataset.with_split(
        "valid", [s for s in dataset["valid"] if s.meta["game"] in games]
    )


def score_model(
    model: Model, sets: dict, splits: dict, homographies: dict, truths: dict
) -> dict:
    """Point and per-draw values of every card number: ``{(split, row): {section: {name: (value, draws)}}}``,
    plus the bins and counts that are not bootstrapped."""
    out = {}
    for row in model.rows:
        for set_name, (dataset, draws_by_split) in sets.items():
            mapping = dump_class_map(model.record, set_name, dataset.class_names)
            unmapped = sorted(k for k, v in mapping.items() if v is None)
            if unmapped:
                logger.info("%s: classes %s are not scored", set_name, unmapped)
            found = model.found[(set_name, row)]
            scores = bench.image_scores(dataset, "valid", found, set_name)
            for split, draws in draws_by_split.items():
                keep = (
                    scores.keys["game"].isin(splits[split]).to_numpy()
                    if set_name == "gsr"
                    else np.ones(len(scores.keys), bool)
                )
                exact = bench.summarise_scores(scores, keep)
                numbers = {"detection": detection_draws(scores, keep, draws)}
                check_point(numbers["detection"], exact)
                entry = {"numbers": numbers, "exact": exact}
                if set_name == "gsr":
                    truth = truths[split]
                    table = be.candidate_columns(truth, found)
                    numbers["hits"] = hit_draws(table, draws)
                    numbers["hits"] |= false_ball_draws(
                        found, scores.keys[keep].reset_index(drop=True), truth, draws
                    )
                    rows = placement_table(
                        found, model.images, homographies, splits[split]
                    )
                    numbers["placement"] = placement_draws(rows, draws)
                    entry |= {
                        "bins": bins(table),
                        "placement_counts": placement_counts(rows),
                        "placement_rows": rows,
                    }
                key = (split if set_name == "gsr" else "ekstraklasa", row)
                out[key] = entry
    return out


def check_point(found: dict, exact: dict) -> None:
    """The resampling AP of the original images is `evaluate`'s (supervision's) AP."""
    pairs = {
        "ball_ap50": exact["per_class"]["ball"]["map50"],
        "ball_ap50_95": exact["per_class"]["ball"]["map50_95"],
        "map50_95": exact["map50_95"],
    }
    for key, value in pairs.items():
        if key in found and not np.isclose(
            found[key][0], value, atol=1e-9, equal_nan=True
        ):
            raise AssertionError(f"{key}: resampling point {found[key][0]} != {value}")


def card(
    model: Model, reference: Model, splits: list[str], out: Path, primary_row: str
) -> dict:
    """Score ``model`` (and ``reference`` for the paired differences) on ``splits`` and write the card."""
    gsr = bench.gsr_split("valid")
    bench.check_manifest(gsr, "valid")
    games = sorted({g for s in splits for g in SPLITS[s]})
    gsr = subset(gsr, games)
    ekstraklasa = bench.ekstraklasa_dev()
    split_games = {s: SPLITS[s] for s in splits}
    clips = {
        s: sorted({x.meta["sequence"] for x in gsr["valid"] if x.meta["game"] in g})
        for s, g in split_games.items()
    }
    eks_clips = sorted(
        {bench.image_key("ekstraklasa", x)["clip"] for x in ekstraklasa["valid"]}
    )
    sets = {
        "gsr": (gsr, {s: Draws(clips[s]) for s in splits}),
        "ekstraklasa": (ekstraklasa, {"ekstraklasa": Draws(eks_clips)}),
    }
    homographies = bench.frozen_homographies("valid")
    truths = {s: truth_table(gsr, SPLITS[s]) for s in splits}
    scored = score_model(model, sets, split_games, homographies, truths)
    if reference.folder.resolve() == model.folder.resolve():
        reference_scored = scored
    else:
        reference_scored = score_model(
            reference, sets, split_games, homographies, truths
        )

    result: dict = {
        "dump": str(model.folder),
        "weights": model.record["weights"],
        "weights_sha256": model.record["weights_sha256"],
        "rows": model.record["rows"],
        "max_det": model.record["max_det"],
        "classes": model.record["classes"],
        "class_map": {
            name: dump_class_map(model.record, name, d.class_names)
            for name, (d, _) in sets.items()
        },
        "reference": {
            "dump": str(reference.folder),
            "weights_sha256": reference.record["weights_sha256"],
        },
        "splits": {
            s: {"games": list(SPLITS[s]), "clips": len(clips[s])} for s in splits
        },
        "bootstrap": BOOTSTRAP,
        "cost": cost(model.record, model.images),
        "primary_row": primary_row,
        "cards": {},
    }
    out.mkdir(parents=True, exist_ok=True)
    for (split, row), entry in scored.items():
        against = (split, pair_row(row, reference))
        reference_entry = reference_scored[against]
        if split != "ekstraklasa" and not entry["placement_rows"][
            ["clip", "frame"]
        ].equals(reference_entry["placement_rows"][["clip", "frame"]]):
            raise SystemExit(f"{split}/{row}: the reference has other placement frames")
        sections = {}
        for section, numbers in entry["numbers"].items():
            sections[section] = summary(numbers)
            sections[f"{section}_vs_reference"] = paired(
                numbers, reference_entry["numbers"][section]
            )
        exact = entry["exact"]
        sections["detection_per_class"] = exact["per_class"]
        sections["paired_with"] = f"reference {against[0]}/{against[1]}"
        if split != "ekstraklasa":
            sections["bins"] = entry["bins"]
            sections["placement_counts"] = entry["placement_counts"]
            entry["placement_rows"].to_parquet(
                out / f"placement_{split}_{row}.parquet", index=False
            )
        result["cards"][f"{split}/{row}"] = sections
    result["rule"] = verdict(result, primary_row)
    (out / "scorecard.json").write_text(
        json.dumps(result, indent=2, default=float) + "\n"
    )
    (out / "scorecard.md").write_text(markdown(result))
    return result


def verdict(result: dict, primary_row: str) -> dict:
    """The selection rule's reads of this card (``RULE``): primary and guards on the selection split, confirm on
    game 5, the absolute gates on the full split; ``None`` where the card lacks the split."""
    cards = result["cards"]
    out: dict = {"rule": RULE, "primary_row": primary_row}
    selection = cards.get(f"selection/{primary_row}")
    if selection:
        gain = selection["placement_vs_reference"]["well_placed"]
        people = selection["detection_vs_reference"].get("people_map50_95", {})
        out["primary"] = {
            "difference": gain["difference"],
            "ci": gain["ci"],
            "success": gain["ci"][0] > 0,
        }
        out["guard_people"] = (
            people.get("difference", np.nan) >= -RULE["people_map50_95_drop_max"]
        )
    eks = cards.get(f"ekstraklasa/{primary_row}")
    if eks:
        diff = eks["detection_vs_reference"]["map50_95"]["difference"]
        out["guard_ekstraklasa"] = diff >= -RULE["ekstraklasa_map50_95_drop_max"]
        out["ekstraklasa_ball_ap50_direction"] = (
            eks["detection"]["ball_ap50"]["value"]
            >= RULE["ekstraklasa_ball_ap50_direction"]
        )
    confirm = cards.get(f"confirm/{primary_row}")
    if confirm:
        gain = confirm["placement_vs_reference"]["well_placed"]
        out["confirm"] = {
            "difference": gain["difference"],
            "ci": gain["ci"],
            "confirmed": gain["ci"][0] > 0,
        }
    full = cards.get(f"full/{primary_row}")
    if full:
        out["full_ball_ap50_min"] = (
            full["detection"]["ball_ap50"]["value"] >= RULE["ball_ap50_min"]
        )
        well = full["placement"]["well_placed"]["value"]
        gate = RULE["gate_1280"] if primary_row != "e3" else RULE["gate_e3"]
        if gate:
            out["gate"] = {k: well >= v for k, v in gate.items()} | {
                "well_placed": well
            }
    full_e3 = cards.get("full/e3")
    if full_e3:
        well = full_e3["placement"]["well_placed"]["value"]
        out["gate_e3"] = {k: well >= v for k, v in RULE["gate_e3"].items()} | {
            "well_placed": well
        }
    return out


# ------------------------------------------------------------------ report
def _ci(entry: dict, scale: float = 1.0, digits: int = 3) -> str:
    low, high = (scale * v for v in entry["ci"])
    return f"[{low:.{digits}f}; {high:.{digits}f}]"


def markdown(result: dict) -> str:
    lines = [
        f"# Ball scorecard: {Path(result['weights']).name} ({result['weights_sha256'][:8]})",
        "",
        f"Dump `{result['dump']}`; paired against `{result['reference']['dump']}`; "
        f"bootstrap {result['bootstrap']['draws']} draws over clips, seed {result['bootstrap']['seed']}.",
        "",
        "| split/row | ball AP50 | AP50-95 | P / R @0.15 | recall centre | people mAP50-95 | well placed | "
        "well (with H) | placed | median m | > 5 m | near goal well | ms/frame |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    ms = result["cost"]["ms_per_frame"]
    for key, c in result["cards"].items():
        split, row = key.split("/")
        d = c["detection"]
        cost_key = f"{'ekstraklasa' if split == 'ekstraklasa' else 'gsr'}/{row}"
        if split == "ekstraklasa":
            lines.append(
                f"| {key} | {d['ball_ap50']['value']:.3f} {_ci(d['ball_ap50'])} | "
                f"{d['ball_ap50_95']['value']:.3f} | {d['ball_precision']['value']:.3f} / "
                f"{d['ball_recall']['value']:.3f} | | mAP50-95 {d['map50_95']['value']:.3f} "
                f"{_ci(d['map50_95'])} | | | | | | | {ms.get(cost_key, '')} |"
            )
            continue
        p, h = c["placement"], c["hits"]
        lines.append(
            f"| {key} | {d['ball_ap50']['value']:.3f} {_ci(d['ball_ap50'])} | {d['ball_ap50_95']['value']:.3f} | "
            f"{d['ball_precision']['value']:.3f} / {d['ball_recall']['value']:.3f} | "
            f"{h['recall_centre@0.15']['value']:.3f} | {d['people_map50_95']['value']:.3f} | "
            f"{100 * p['well_placed']['value']:.2f}% {_ci(p['well_placed'], 100, 1)} | "
            f"{100 * p['well_placed_with_homography']['value']:.2f}% | {100 * p['placed']['value']:.2f}% | "
            f"{p['median_m']['value']:.2f} | {100 * p['far_of_placed']['value']:.2f}% | "
            f"{100 * p['near_goal_well_placed']['value']:.2f}% | {ms.get(cost_key, '')} |"
        )
    lines += [
        "",
        "Paired with the reference (variant - reference, 95% clip-bootstrap interval):",
        "",
        "| split/row | paired with | well placed (pp) | well with H (pp) | > 5 m (pp) | ball AP50 | people mAP50-95 | "
        "Ekstraklasa mAP50-95 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, c in result["cards"].items():
        d = c["detection_vs_reference"]
        if key.startswith("ekstraklasa"):
            lines.append(
                f"| {key} | {c['paired_with']} | | | | {d['ball_ap50']['difference']:+.3f} {_ci(d['ball_ap50'])} | "
                f"| {d['map50_95']['difference']:+.3f} {_ci(d['map50_95'])} |"
            )
            continue
        p = c["placement_vs_reference"]
        lines.append(
            f"| {key} | {c['paired_with']} | {100 * p['well_placed']['difference']:+.2f} "
            f"{_ci(p['well_placed'], 100, 2)} | {100 * p['well_placed_with_homography']['difference']:+.2f} | "
            f"{100 * p['far_of_placed']['difference']:+.2f} | {d['ball_ap50']['difference']:+.3f} "
            f"{_ci(d['ball_ap50'])} | {d['people_map50_95']['difference']:+.3f} {_ci(d['people_map50_95'])} | |"
        )
    for key, c in result["cards"].items():
        if "placement_counts" in c:
            counts = c["placement_counts"]
            lines.append(
                f"\n{key}: {counts['frames']} frames ({counts['without_homography']} without homography, "
                f"ceiling {100 * c['placement']['ceiling']['value']:.1f}%), {counts['clips']} clips"
            )
    size_bins = [
        (key, b)
        for key, c in result["cards"].items()
        for b in c.get("bins", [])
        if b["feature"] == "size"
    ]
    if size_bins:
        lines += [
            "",
            "| split/row | ball size | balls | recall IoU | recall centre |",
            "|---|---|---|---|---|",
        ]
        lines += [
            f"| {key} | {b['level']} | {b['balls']} | {b['recall_iou']:.3f} | {b['recall_centre']:.3f} |"
            for key, b in size_bins
        ]
    lines += ["", "Rule reads: `" + json.dumps(result["rule"], default=float) + "`", ""]
    return "\n".join(lines)


# ------------------------------------------------------------------ agree
def leaves(value, path: str = "") -> dict[str, float]:
    """Every number of a card by path; cost, paths and settings left out."""
    if isinstance(value, dict):
        found = {}
        for key, item in value.items():
            if key in ("cost", "dump", "weights", "reference", "rule"):
                continue
            found |= leaves(item, f"{path}/{key}")
        return found
    if isinstance(value, list):
        found = {}
        for i, item in enumerate(value):
            name = item.get("level", i) if isinstance(item, dict) else i
            prefix = f"{item.get('feature')}=" if isinstance(item, dict) else ""
            found |= leaves(item, f"{path}/{prefix}{name}")
        return found
    if isinstance(value, bool) or not isinstance(value, int | float):
        return {}
    return {path: float(value)}


def agree(a: dict, b: dict) -> dict:
    """The largest absolute difference between two cards' numbers (NaN on both sides counts as equal)."""
    first, second = leaves(a), leaves(b)
    shared = sorted(set(first) & set(second))
    differences = {
        k: 0.0
        if np.isnan(first[k]) and np.isnan(second[k])
        else abs(first[k] - second[k])
        for k in shared
    }
    worst = max(differences, key=lambda k: np.nan_to_num(differences[k], nan=np.inf))
    return {
        "numbers": len(shared),
        "only_in_one": sorted(set(first) ^ set(second)),
        "max_abs_difference": differences[worst],
        "worst": worst,
        "over_1e-3": sorted(k for k, v in differences.items() if not v <= 1e-3),
    }


# ------------------------------------------------------------------ references
def file_rows(path: Path) -> dict:
    return json.loads(path.read_text())["rows"]


def check_against_files(name: str, model: Model, files: dict) -> dict:
    """The dump's whole-valid detection numbers (every set and row in the files) and its e3 placement against the
    model's result files: only the differences are kept."""
    frozen = file_rows(files["detection"]) | file_rows(files["ekstraklasa"])
    sha = {
        json.loads(files[k].read_text())["weights_sha256"]
        for k in ("detection", "ekstraklasa", "placement")
    }
    if sha != {model.record["weights_sha256"]}:
        raise SystemExit(f"{name}: result files of other weights {sha}")
    sets = bench.evaluation_sets("valid", list(bench.DUMP_SETS))
    differences = {}
    for (set_name, row), found in sorted(model.found.items()):
        reference = frozen.get(f"{set_name}/{row}")
        if reference is None:
            continue
        if _settings(reference["settings"]) != _settings(model.record["rows"][row]):
            raise SystemExit(
                f"{name} {set_name}/{row}: settings differ from the result file"
            )
        dataset, split = sets[set_name]
        scored = bench.score_dump(dataset, split, found, set_name)
        per_class = {c["class_name"]: c for c in reference["per_class"]}
        worst = abs(scored["map50_95"] - reference["map50_95"])
        for cls, metrics in per_class.items():
            for metric in ("map50", "map50_95", "precision", "recall"):
                worst = max(
                    worst, abs(scored["per_class"][cls][metric] - metrics[metric])
                )
        differences[f"{set_name}/{row}"] = worst
    placement = json.loads(files["placement"].read_text())
    size = bench.PLACEMENT["input_size"]
    rows = bench.placement_rows(
        bench.dump_top_ball(
            model.found[("gsr", "e3")], (1920 / size[0], 1080 / size[1])
        ),
        bench.frozen_homographies("valid"),
        "valid",
    )
    stats = bench.placement_statistics(rows)
    differences["placement/e3"] = max(
        abs(stats[k] - placement["all"][k]["value"])
        for k in ("frames", "placed", "median_m", "far", "lt5")
    )
    return differences


def references(args: argparse.Namespace) -> None:
    """Dump (GPU) and card every reference row, check them against their result files, write the table."""
    checks, cards = {}, {}
    for name, files in REFERENCES.items():
        folder = args.out / name
        if not (folder / "dump/dump.json").is_file():
            dump_weights(
                files["weights"], folder / "dump", list(bench.DUMP_ROWS), args.device
            )
        model = Model(folder / "dump")
        if model.record["weights_sha256"] != bench.sha256_file(files["weights"]):
            raise SystemExit(f"{folder} holds a dump of other weights")
        checks[name] = check_against_files(name, model, files)
        splits = ["selection", "confirm", "full"] if name == "dd2c" else ["selection"]
        cards[name] = card(model, Model(args.reference), splits, folder, "1280")
    direct = direct_placement_check(args.out / "dd2c")
    (args.out / "reference_check.json").write_text(
        json.dumps({"files": checks, "direct_placement_1280": direct}, indent=2) + "\n"
    )
    table = reference_table(cards)
    table.to_csv(args.out / "reference_rows.csv", index=False)
    (args.out / "reference_rows.md").write_text(reference_markdown(table, checks))
    print(json.dumps({"files": checks, "direct_placement_1280": direct}, indent=2))
    print(reference_markdown(table, checks))


def direct_placement_check(folder: Path) -> dict | None:
    """dd2c's 1280 placement through the weights (`detector_bench.py placement --row 1280 --out
    <folder>/placement_direct_1280`: the model at conf 0.15 on each native frame) against its card's, from the dump."""
    path = folder / "placement_direct_1280/placement_frames.parquet"
    if not path.is_file():
        return None
    direct = pd.read_parquet(path)
    rows = pd.read_parquet(folder / "placement_full_1280.parquet")
    joined = rows.merge(direct, on=["clip", "frame"], suffixes=("", "_direct"))
    placed = np.isfinite(joined.error_m), np.isfinite(joined.error_m_direct)
    both = placed[0] & placed[1]
    far = bench.PLACEMENT["far_m"]
    return {
        "frames": len(rows),
        "frames_direct": len(direct),
        "frames_joined": len(joined),
        "same_placed_flag": float((placed[0] == placed[1]).mean()),
        "max_abs_error_difference_m": float(
            np.abs(joined.error_m[both] - joined.error_m_direct[both]).max()
        ),
        "well_placed_card": float((rows.error_m <= far).mean()),
        "well_placed_direct": float((direct.error_m <= far).mean()),
    }


def reference_table(cards: dict[str, dict]) -> pd.DataFrame:
    records = []
    for name, result in cards.items():
        for key, c in result["cards"].items():
            split, row = key.split("/")
            d = c["detection"]
            record = {
                "model": name,
                "split": split,
                "row": row,
                "ball_ap50": d["ball_ap50"]["value"],
                "ball_ap50_95": d["ball_ap50_95"]["value"],
                "ball_precision": d["ball_precision"]["value"],
                "ball_recall": d["ball_recall"]["value"],
                "map50_95": d["map50_95"]["value"],
                "ms_per_frame": result["cost"]["ms_per_frame"].get(
                    f"{'ekstraklasa' if split == 'ekstraklasa' else 'gsr'}/{row}"
                ),
            }
            if split != "ekstraklasa":
                p, h = c["placement"], c["hits"]
                pr = c["placement_vs_reference"]["well_placed"]
                record |= {
                    "people_map50_95": d["people_map50_95"]["value"],
                    "recall_centre": h["recall_centre@0.15"]["value"],
                    "frames": c["placement_counts"]["frames"],
                    "ceiling": p["ceiling"]["value"],
                    "placed": p["placed"]["value"],
                    "well_placed": p["well_placed"]["value"],
                    "well_placed_ci": p["well_placed"]["ci"],
                    "well_placed_with_homography": p["well_placed_with_homography"][
                        "value"
                    ],
                    "median_m": p["median_m"]["value"],
                    "far_of_placed": p["far_of_placed"]["value"],
                    "near_goal_well_placed": p["near_goal_well_placed"]["value"],
                    "well_placed_vs_dd2c": pr["difference"],
                    "well_placed_vs_dd2c_ci": pr["ci"],
                }
            records.append(record)
    return pd.DataFrame(records)


def reference_markdown(table: pd.DataFrame, checks: dict) -> str:
    def pct(v):
        return "" if pd.isna(v) else f"{100 * v:.2f}%"

    lines = [
        "# ball-detection 02: reference rows",
        "",
        "| model | split | row | ball AP50 | AP50-95 | P / R | people mAP50-95 | recall centre | well placed "
        "[95%] | well (with H) | placed | median m | > 5 m | near goal well | vs dd2c (pp) [95%] | ms/frame |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in table.itertuples():
        gsr = r.split != "ekstraklasa"
        well = (
            f"{pct(r.well_placed)} [{100 * r.well_placed_ci[0]:.1f}; {100 * r.well_placed_ci[1]:.1f}]"
            if gsr
            else ""
        )
        versus = (
            f"{100 * r.well_placed_vs_dd2c:+.2f} [{100 * r.well_placed_vs_dd2c_ci[0]:+.2f}; "
            f"{100 * r.well_placed_vs_dd2c_ci[1]:+.2f}]"
            if gsr
            else ""
        )
        people = f"{r.people_map50_95:.3f}" if gsr else f"(all {r.map50_95:.3f})"
        lines.append(
            f"| {r.model} | {r.split} | {r.row} | {r.ball_ap50:.3f} | {r.ball_ap50_95:.3f} | "
            f"{r.ball_precision:.3f} / {r.ball_recall:.3f} | {people} | "
            f"{f'{r.recall_centre:.3f}' if gsr else ''} | {well} | {pct(r.well_placed_with_homography) if gsr else ''} | "
            f"{pct(r.placed) if gsr else ''} | {f'{r.median_m:.2f}' if gsr else ''} | "
            f"{pct(r.far_of_placed) if gsr else ''} | {pct(r.near_goal_well_placed) if gsr else ''} | {versus} | "
            f"{r.ms_per_frame} |"
        )
    lines += [
        "",
        "Check against the result files (max |difference|, whole GSR valid and Ekstraklasa dev):",
        "",
        "| model | " + " | ".join(next(iter(checks.values()))) + " |",
        "|---|" + "---|" * len(next(iter(checks.values()))),
    ]
    for name, check in checks.items():
        lines.append(
            f"| {name} | " + " | ".join(f"{v:.2g}" for v in check.values()) + " |"
        )
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ commands
def dump_weights(weights: Path, out: Path, rows: list[str], device: str | None) -> None:
    bench.dump(SimpleNamespace(weights=weights, out=out, rows=rows, device=device))


def card_command(args: argparse.Namespace) -> None:
    if (args.weights is None) == (args.dump is None):
        raise SystemExit("give --weights or --dump")
    folder = args.dump
    if args.weights is not None:
        folder = args.out / "dump"
        dump_weights(args.weights, folder, args.rows, args.device)
    splits = ["selection", "confirm", "full"] if args.confirm else ["selection"]
    result = card(
        Model(folder), Model(args.reference), splits, args.out, args.primary_row
    )
    print((args.out / "scorecard.md").read_text())
    print(json.dumps(result["rule"], indent=2, default=float))


def agree_command(args: argparse.Namespace) -> None:
    a, b = (json.loads((p / "scorecard.json").read_text()) for p in (args.a, args.b))
    report = agree(a, b)
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    c = commands.add_parser("card", help="the scorecard of weights or a dump")
    c.add_argument("--weights", type=Path)
    c.add_argument("--dump", type=Path)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument(
        "--rows", nargs="*", choices=list(bench.ROWS), default=list(bench.DUMP_ROWS)
    )
    c.add_argument(
        "--confirm",
        action="store_true",
        help="also read game 5 (confirm split) and the full GSR valid: a ticket winner's confirm read only",
    )
    c.add_argument("--primary-row", default="1280", help="the row the rule's reads use")
    a = commands.add_parser("agree", help="max |difference| of two cards' numbers")
    a.add_argument("a", type=Path)
    a.add_argument("b", type=Path)
    a.add_argument("--out", type=Path)
    r = commands.add_parser(
        "references", help="reference rows and their check against result files"
    )
    r.add_argument("--out", type=Path, default=OUT)
    for command in (c, r):
        command.add_argument("--reference", type=Path, default=DD2C_DUMP)
        command.add_argument("--device", default=None)
    args = parser.parse_args()
    tv.setup_logging("INFO")
    {"card": card_command, "agree": agree_command, "references": references}[
        args.command
    ](args)


if __name__ == "__main__":
    sys.exit(main())
