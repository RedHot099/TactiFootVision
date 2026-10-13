"""Homography evaluation in metres on SoccerNet GSR (pitch-keypoints 02).

Truth: GSR's pitch position of every person, `bbox_pitch` bottom-middle, in centre-origin metres (x along the length,
growing to the right of the main camera; y across, growing towards the camera). `SoccerPitch` has its origin in the far
left corner with the same axis directions, so a GSR position is `(x + 52.5, y + 34)` there (`to_corner`;
`convention_check` verifies it on the annotated touchlines, goal lines and halfway line).

A method's frame -> pitch homography maps the bottom-middle of each person's `bbox_image` to the pitch. Per person, the
error is the distance in metres to the truth; feet beyond the homography's horizon, and errors above `CAP_M` (the pitch
diagonal), count as `CAP_M`, so one absurd homography cannot dominate a mean. Only persons on the pitch count:
|x| <= 52.5 + 3 and |y| <= 34 + 3 in GSR coordinates (`ON_PITCH_MARGIN_M`). Persons are every object but the ball.

Frames: every `FRAME_STEP`-th frame of each clip (`keeps_frame`), the frames ticket 01 keeps of GSR train and test.

Per frame: the median error over its persons (`error_m`). A frame is *scored* when it has at least one person on the
pitch with a pitch position (frames without the camera annotation have none). Per method and split (`summarise`):
- `availability`: share of scored frames with a homography;
- `median_m`, `mean_m`, `p90_m`: of the per-frame errors over scored frames with a homography;
- `under_1m`, `under_2m`, `under_5m`: share of **all** scored frames whose error is below the threshold (a frame
  without a homography is not under), so a method cannot look good by refusing hard frames;
- `agreement_median` (scorable frames) and `lines_good` (share of all scored frames with agreement >= 0.5): ticket 08's
  label-free line agreement (`xgpoc.detector.lines`), at 1280 x 720;
- `refined`: share of scored frames whose refinement replaced the keypoint fit (methods with the line refinement).
Every statistic has a 95% interval from a bootstrap that resamples clips (`BOOTSTRAP_DRAWS` draws, seed 0), over all
scored frames and over the *near-goal* stratum (frames whose persons' median x is within `NEAR_GOAL_M` of a goal line,
where shots are taken). `per_group` gives the same statistics per game or clip (no interval), `paired` the difference of
each method from a reference method with the same clip draws for both.

Keypoint error (`vertex_errors`, `keypoint_statistics`), against ticket 01's 32-vertex labels at 720p: for every vertex
the label sees (in the frame, in front of the camera), the distance in pixels of the model's keypoint when its confidence
is >= `KEYPOINT_CONF`; `recall` = share of seen vertices predicted that confidently, `px_*` over those, and `spurious` =
share of confident predictions whose vertex the label does not see.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from tactifoot_vision.pitch.homography import project_points

HALF_LENGTH, HALF_WIDTH = 52.5, 34.0   # GSR's pitch, 105 x 68 m like `SoccerPitch()`
ON_PITCH_MARGIN_M = 3.0
CAP_M = float(np.hypot(105.0, 68.0))  # 125.1 m, the pitch diagonal
THRESHOLDS_M = (1.0, 2.0, 5.0)
LINES_GOOD = 0.5                       # ticket 08's usable line agreement
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 2000, 0
EVAL_SIZE = (1280, 720)                # frames are scored at 720p, the size of the shot frames D1-D4 were chosen on
NEAR_GOAL_M = 30.0                     # a frame is near a goal when its persons' median x is this close to a goal line
KEYPOINT_CONF = 0.5                    # keypoints the px error counts (D1/D3's homography threshold)
THRESHOLDS_PX = (5.0, 10.0, 20.0)
FRAME_STEP = 5                         # every 5th frame of a clip: 1, 6, 11, ... (ticket 01's subsample, a test ties them)
BALL = "ball"

# GSR line name -> (axis, coordinate in corner-origin metres) of the straight markings `convention_check` uses.
CONVENTION_LINES = {"Side line top": (1, 0.0), "Side line bottom": (1, 68.0), "Side line left": (0, 0.0),
                    "Side line right": (0, 105.0), "Middle line": (0, 52.5)}


@dataclass(frozen=True)
class Clip:
    """One GSR clip: its sampled frames and its persons (all frames).

    `frames`: frame (1-based), path, width, height. `persons`: frame, category, x1, y1, x2, y2 (`bbox_image` pixels)
    and pitch_x, pitch_y (`bbox_pitch` bottom-middle, corner origin; NaN without the camera annotation).
    `lines`: frame -> {GSR line name: (N, 2) normalised image points}. `game`: GSR's `info.game_id` (an index).
    """

    name: str
    frames: pd.DataFrame
    persons: pd.DataFrame
    lines: dict[int, dict[str, np.ndarray]]
    game: str = ""


def keeps_frame(frame: int) -> bool:
    """Whether 1-based GSR frame number `frame` is scored (frames 1, 6, 11, ...)."""
    return (frame - 1) % FRAME_STEP == 0


def to_corner(x, y):
    """GSR centre-origin metres -> `SoccerPitch` corner-origin metres."""
    return np.asarray(x, dtype=float) + HALF_LENGTH, np.asarray(y, dtype=float) + HALF_WIDTH


def read_clip(folder: Path, keep_frame=keeps_frame) -> Clip:
    """Read `folder/Labels-GameState.json`: the frames `keep_frame(frame)` selects (default: the scored frames), their
    persons and pitch lines."""
    folder = Path(folder)
    labels = json.loads((folder / "Labels-GameState.json").read_text())
    categories = {c["id"]: c["name"] for c in labels["categories"]}
    im_dir = labels["info"].get("im_dir", "img1")
    frames, frame_by_image_id = [], {}
    for image in labels["images"]:
        frame = int(Path(image["file_name"]).stem)
        if keep_frame(frame):
            frame_by_image_id[image["image_id"]] = frame
            frames.append({"frame": frame, "path": str(folder / im_dir / image["file_name"]),
                           "width": int(image["width"]), "height": int(image["height"])})
    persons, lines = [], {}
    for item in labels["annotations"]:
        frame = frame_by_image_id.get(item["image_id"])
        if frame is None:
            continue
        if item["supercategory"] == "pitch":
            lines[frame] = {name: np.array([[p["x"], p["y"]] for p in points], dtype=float)
                            for name, points in item["lines"].items()}
            continue
        category = categories[item["category_id"]]
        if item["supercategory"] != "object" or category == BALL:
            continue
        box, pitch = item["bbox_image"], item.get("bbox_pitch")
        x, y = to_corner(pitch["x_bottom_middle"], pitch["y_bottom_middle"]) if pitch else (np.nan, np.nan)
        persons.append({"frame": frame, "category": category, "x1": box["x"], "y1": box["y"],
                        "x2": box["x"] + box["w"], "y2": box["y"] + box["h"], "pitch_x": float(x), "pitch_y": float(y)})
    columns = ["frame", "category", "x1", "y1", "x2", "y2", "pitch_x", "pitch_y"]
    return Clip(folder.name, pd.DataFrame(frames).sort_values("frame", ignore_index=True),
                pd.DataFrame(persons, columns=columns), lines, str(labels["info"].get("game_id", "")))


def on_pitch(persons: pd.DataFrame) -> pd.Series:
    """Persons with a pitch position within `ON_PITCH_MARGIN_M` of the pitch."""
    return ((persons.pitch_x - HALF_LENGTH).abs() <= HALF_LENGTH + ON_PITCH_MARGIN_M) & \
           ((persons.pitch_y - HALF_WIDTH).abs() <= HALF_WIDTH + ON_PITCH_MARGIN_M)


def feet(persons: pd.DataFrame, scale: tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
    """`(N, 2)` bottom-middle of the persons' boxes, in pixels times `scale` (x, y)."""
    return np.column_stack([(persons.x1 + persons.x2) / 2 * scale[0], persons.y2 * scale[1]]).astype(float)


def person_errors(homography: np.ndarray | None, feet_px: np.ndarray, truth: np.ndarray,
                  shape: tuple[int, int]) -> np.ndarray | None:
    """Metres between the projected `feet_px` and `truth` (both `(N, 2)`), capped at `CAP_M`; None without a homography.

    `homography` maps frame pixels of a frame of `shape` (h, w) to corner-origin pitch metres. Feet on the far side of
    its horizon (the side the bottom-centre of the frame is not on) have no pitch position and get `CAP_M`."""
    if homography is None:
        return None
    h, w = shape
    xy, valid = project_points(feet_px, np.asarray(homography, dtype=float), (w / 2, h - 1))
    distance = np.linalg.norm(xy - np.asarray(truth, dtype=float), axis=1)
    return np.where(valid & np.isfinite(distance), np.minimum(distance, CAP_M), CAP_M)


# ------------------------------------------------------------------------------------------------- aggregation
def near_goal(median_x) -> np.ndarray:
    """Frames whose persons' median x (corner metres) is within `NEAR_GOAL_M` of either goal line."""
    x = np.asarray(median_x, dtype=float)
    return np.minimum(x, 2 * HALF_LENGTH - x) <= NEAR_GOAL_M


def frame_statistics(frames: pd.DataFrame) -> dict[str, float]:
    """The summary statistics of the module docstring over the scored frames of one method.

    `frames` columns: has_homography, error_m (NaN without a homography), agreement, and optionally refined."""
    has = frames.has_homography.to_numpy(bool)
    error = frames.error_m.to_numpy(float)[has]
    n = len(frames)
    stats = {"frames": float(n), "availability": has.mean() if n else np.nan,
             "median_m": float(np.median(error)) if len(error) else np.nan,
             "mean_m": float(error.mean()) if len(error) else np.nan,
             "p90_m": float(np.quantile(error, 0.9)) if len(error) else np.nan}
    for t in THRESHOLDS_M:
        stats[f"under_{t:g}m"] = float((error < t).sum() / n) if n else np.nan
    agreement = frames.agreement.to_numpy(float)
    stats["agreement_median"] = float(np.nanmedian(agreement)) if np.isfinite(agreement).any() else np.nan
    stats["lines_good"] = float(((agreement >= LINES_GOOD) & has).sum() / n) if n else np.nan
    if "refined" in frames and frames.refined.notna().any():
        stats["refined"] = float((frames.refined.to_numpy(float) == 1).sum() / n)
    return stats


def bootstrap(groups, statistic, draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED) -> pd.DataFrame:
    """`draws` values of `statistic(row positions)` (a dict per draw): each draw takes as many groups as there are,
    with replacement, and passes the positions of all their rows. `groups` holds each row's group (clip or game)."""
    rng = np.random.default_rng(seed)
    codes, _ = pd.factorize(pd.Series(np.asarray(groups)), sort=True)
    positions = [np.flatnonzero(codes == g) for g in range(codes.max() + 1)]
    rows = []
    for _ in range(draws):
        chosen = rng.integers(0, len(positions), len(positions))
        rows.append(statistic(np.concatenate([positions[i] for i in chosen])))
    return pd.DataFrame(rows)


def interval_rows(point: dict, draws_table: pd.DataFrame, labels: dict, exact=("frames",)) -> list[dict]:
    """Long rows `labels | metric, value, ci_low, ci_high` (95% percentile interval; none for `exact` metrics)."""
    rows = []
    for metric, value in point.items():
        low, high = ((np.nan, np.nan) if metric in exact or draws_table[metric].isna().all()
                     else np.nanquantile(draws_table[metric], [0.025, 0.975]))
        rows.append(labels | {"metric": metric, "value": value, "ci_low": low, "ci_high": high})
    return rows


def strata(frames: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """("all", frames) and ("near_goal", its near-goal frames) when the `near_goal` column exists."""
    out = [("all", frames)]
    if "near_goal" in frames:
        out.append(("near_goal", frames[frames.near_goal.astype(bool)]))
    return out


def summarise(per_frame: pd.DataFrame, draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED) -> pd.DataFrame:
    """Long table: split, method, stratum, metric, value, ci_low, ci_high (95%, clip bootstrap) over the scored frames."""
    rows = []
    scored = per_frame[per_frame.scored]
    for (split, method), frames in scored.groupby(["split", "method"], sort=False):
        for stratum, part in strata(frames):
            part = part.reset_index(drop=True)
            if part.empty:
                continue
            table = bootstrap(part["clip"], lambda i, part=part: frame_statistics(part.take(i)), draws, seed)
            labels = {"split": split, "method": method, "stratum": stratum}
            rows += interval_rows(frame_statistics(part), table, labels)
            rows.append(labels | {"metric": "clips", "value": float(part["clip"].nunique()), "ci_low": np.nan,
                                  "ci_high": np.nan})
    return pd.DataFrame(rows)


def per_group(per_frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Long table: split, method, `column` (game or clip), metric, value over each group's scored frames."""
    rows = []
    scored = per_frame[per_frame.scored]
    for (split, method, group), frames in scored.groupby(["split", "method", column], sort=True):
        rows += [{"split": split, "method": method, column: group, "metric": metric, "value": value}
                 for metric, value in frame_statistics(frames).items()]
    return pd.DataFrame(rows)


PAIRED_METRICS = ("availability", "median_m", "p90_m", "under_1m", "under_2m", "under_5m")


def paired(per_frame: pd.DataFrame, reference: pd.DataFrame, draws: int = BOOTSTRAP_DRAWS,
           seed: int = BOOTSTRAP_SEED) -> pd.DataFrame:
    """Each method of `per_frame` minus the reference method (`reference`: one method's per-frame rows of the same
    frames, possibly from another run), over the scored frames both have, all and near goal: the differences of the
    `PAIRED_METRICS` (method - reference) and `median_diff_m`, the median per-frame error difference over frames where
    both have a homography. Intervals use the same clip draws for both methods."""
    reference = reference[reference.scored]
    if reference.method.nunique() != 1:
        raise ValueError(f"the reference must be one method, got {sorted(reference.method.unique())}")
    name = reference.method.iloc[0]
    rows = []
    for (split, method), frames in per_frame[per_frame.scored].groupby(["split", "method"], sort=False):
        if method == name:
            continue
        both = frames.merge(reference, on=["clip", "frame"], suffixes=("", "_ref"), validate="one_to_one")
        if len(both) != len(frames) or len(both) != len(reference):
            raise ValueError(f"{method} and the reference {name} do not score the same frames")
        mine = both[["has_homography", "error_m", "agreement"]]
        theirs = both[["has_homography_ref", "error_m_ref", "agreement_ref"]].set_axis(mine.columns, axis=1)

        def difference(i, mine=mine, theirs=theirs):
            a, b = frame_statistics(mine.take(i)), frame_statistics(theirs.take(i))
            d = mine.error_m.to_numpy()[i] - theirs.error_m.to_numpy()[i]
            return {m: a[m] - b[m] for m in PAIRED_METRICS} | {
                "median_diff_m": float(np.nanmedian(d)) if np.isfinite(d).any() else np.nan}

        for stratum, part in strata(both):
            index = np.flatnonzero(both.index.isin(part.index))
            if not len(index):
                continue
            table = bootstrap(both["clip"].to_numpy()[index], lambda i, index=index: difference(index[i]), draws, seed)
            rows += interval_rows(difference(index), table, {"split": split, "method": method, "reference": name,
                                                             "stratum": stratum, "frames": len(index)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------- keypoint error
def vertex_errors(xy: np.ndarray | None, confidence: np.ndarray | None, label_xy: np.ndarray,
                  label_seen: np.ndarray) -> pd.DataFrame:
    """One row per vertex: seen (by the label), conf (the model's, 0 without an instance) and px (distance of the
    predicted keypoint from the label, NaN where the label does not see the vertex). All at the same pixel scale."""
    label_xy, seen = np.asarray(label_xy, dtype=float).reshape(-1, 2), np.asarray(label_seen, dtype=bool)
    if xy is None:
        xy, confidence = np.full_like(label_xy, np.nan), np.zeros(len(label_xy))
    px = np.where(seen, np.linalg.norm(np.asarray(xy, dtype=float).reshape(-1, 2) - label_xy, axis=1), np.nan)
    return pd.DataFrame({"vertex": np.arange(len(label_xy)), "seen": seen,
                         "conf": np.asarray(confidence, dtype=float), "px": px})


def keypoint_statistics(rows: pd.DataFrame, min_confidence: float = KEYPOINT_CONF) -> dict[str, float]:
    """`vertex_errors` rows of many frames -> frames, seen vertices, recall, px median/mean/p90, shares under
    `THRESHOLDS_PX` (of confidently predicted seen vertices) and spurious (share of confident predictions not seen)."""
    seen, confident = rows.seen.to_numpy(bool), rows.conf.to_numpy(float) >= min_confidence
    px = rows.px.to_numpy(float)[seen & confident]
    stats = {"frames": float(rows[["group", "frame"]].drop_duplicates().shape[0]) if "group" in rows else np.nan,
             "seen": float(seen.sum()), "recall": float(confident[seen].mean()) if seen.any() else np.nan,
             "px_median": float(np.median(px)) if len(px) else np.nan,
             "px_mean": float(px.mean()) if len(px) else np.nan,
             "px_p90": float(np.quantile(px, 0.9)) if len(px) else np.nan}
    for t in THRESHOLDS_PX:
        stats[f"under_{t:g}px"] = float((px < t).mean()) if len(px) else np.nan
    stats["spurious"] = float((confident & ~seen).sum() / confident.sum()) if confident.any() else np.nan
    return stats


def summarise_keypoints(rows: pd.DataFrame, draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED,
                        min_confidence: float = KEYPOINT_CONF) -> pd.DataFrame:
    """Long table: split, metric, value, ci_low, ci_high of `keypoint_statistics`, bootstrap over `group` (clip/game)."""
    out = []
    for split, part in rows.groupby("split", sort=False):
        part = part.reset_index(drop=True)
        table = bootstrap(part["group"], lambda i, part=part: keypoint_statistics(part.take(i), min_confidence),
                          draws, seed)
        out += interval_rows(keypoint_statistics(part, min_confidence), table,
                             {"split": split, "min_confidence": min_confidence}, exact=("frames", "seen"))
        out.append({"split": split, "min_confidence": min_confidence, "metric": "groups",
                    "value": float(part["group"].nunique()), "ci_low": np.nan, "ci_high": np.nan})
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------------- convention check
def fit_on_feet(feet_px: np.ndarray, truth: np.ndarray) -> np.ndarray | None:
    """Least-squares frame -> pitch homography of persons' feet to their GSR positions (>= 4 persons)."""
    if len(feet_px) < 4:
        return None
    matrix, _ = cv2.findHomography(feet_px.astype(np.float32), truth.astype(np.float32), 0)
    return None if matrix is None or not np.isfinite(matrix).all() else matrix


def convention_check(clip: Clip, min_persons: int = 8) -> pd.DataFrame:
    """Per frame with >= `min_persons` persons on the pitch: a homography fitted on the feet maps GSR's annotated
    straight lines (`CONVENTION_LINES`) to their `SoccerPitch` coordinates (residual metres along the line's normal),
    and its leave-one-out error on the feet (the floor of the feet metric: box feet vs the truth they came from).

    With a wrong origin, axis direction or left/right, the touchlines and goal lines land metres off."""
    rows = []
    persons = clip.persons[on_pitch(clip.persons)]
    for frame in clip.frames.itertuples(index=False):
        group = persons[persons.frame == frame.frame]
        if len(group) < min_persons:
            continue
        feet_px, truth = feet(group), group[["pitch_x", "pitch_y"]].to_numpy(float)
        matrix = fit_on_feet(feet_px, truth)
        if matrix is None:
            continue
        loo = []
        for i in range(len(group)):
            keep = np.arange(len(group)) != i
            held = fit_on_feet(feet_px[keep], truth[keep])
            if held is not None:
                loo.append(person_errors(held, feet_px[i:i + 1], truth[i:i + 1], (frame.height, frame.width))[0])
        row = {"clip": clip.name, "frame": frame.frame, "persons": len(group),
               "feet_loo_median_m": float(np.median(loo)) if loo else np.nan}
        for name, (axis, value) in CONVENTION_LINES.items():
            points = clip.lines.get(frame.frame, {}).get(name)
            if points is None or len(points) == 0:
                continue
            pixels = points * [frame.width, frame.height]
            projected = cv2.perspectiveTransform(pixels.reshape(-1, 1, 2), matrix).reshape(-1, 2)
            row[name] = float(np.median(np.abs(projected[:, axis] - value)))
        rows.append(row)
    return pd.DataFrame(rows)
