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
Every statistic has a 95% interval from a bootstrap that resamples clips (`BOOTSTRAP_DRAWS` draws, seed 0).
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
    `lines`: frame -> {GSR line name: (N, 2) normalised image points}.
    """

    name: str
    frames: pd.DataFrame
    persons: pd.DataFrame
    lines: dict[int, dict[str, np.ndarray]]


def keeps_frame(frame: int) -> bool:
    """Whether 1-based GSR frame number `frame` is scored (frames 1, 6, 11, ...)."""
    return (frame - 1) % FRAME_STEP == 0


def to_corner(x, y):
    """GSR centre-origin metres -> `SoccerPitch` corner-origin metres."""
    return np.asarray(x, dtype=float) + HALF_LENGTH, np.asarray(y, dtype=float) + HALF_WIDTH


def read_clip(folder: Path, keep_frame=lambda frame: True) -> Clip:
    """Read `folder/Labels-GameState.json`: the frames `keep_frame(frame)` selects, their persons and pitch lines."""
    folder = Path(folder)
    labels = json.loads((folder / "Labels-GameState.json").read_text())
    categories = {c["id"]: c["name"] for c in labels["categories"]}
    im_dir = labels["info"].get("im_dir", "img1")
    frames, frame_of = [], {}
    for image in labels["images"]:
        frame = int(Path(image["file_name"]).stem)
        if keep_frame(frame):
            frame_of[image["image_id"]] = frame
            frames.append({"frame": frame, "path": str(folder / im_dir / image["file_name"]),
                           "width": int(image["width"]), "height": int(image["height"])})
    persons, lines = [], {}
    for item in labels["annotations"]:
        frame = frame_of.get(item["image_id"])
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
                pd.DataFrame(persons, columns=columns), lines)


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


def bootstrap(frames: pd.DataFrame, statistic, draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED) -> pd.DataFrame:
    """`draws` values of `statistic(frames of resampled clips)` (a dict per draw), clips drawn with replacement."""
    rng = np.random.default_rng(seed)
    by_clip = [group for _, group in frames.groupby("clip", sort=True)]
    rows = []
    for _ in range(draws):
        chosen = rng.integers(0, len(by_clip), len(by_clip))
        rows.append(statistic(pd.concat([by_clip[i] for i in chosen], ignore_index=True)))
    return pd.DataFrame(rows)


def summarise(per_frame: pd.DataFrame, draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED) -> pd.DataFrame:
    """Long table: split, method, metric, value, ci_low, ci_high (95%, clip bootstrap) over the scored frames."""
    rows = []
    scored = per_frame[per_frame.scored]
    for (split, method), frames in scored.groupby(["split", "method"], sort=False):
        point = frame_statistics(frames)
        draws_table = bootstrap(frames, frame_statistics, draws, seed)
        for metric, value in point.items():
            low, high = (np.nan, np.nan) if metric == "frames" else np.nanquantile(draws_table[metric], [0.025, 0.975])
            rows.append({"split": split, "method": method, "metric": metric, "value": value,
                         "ci_low": low, "ci_high": high})
    rows += [{"split": split, "method": method, "metric": "clips", "value": float(frames["clip"].nunique()),
              "ci_low": np.nan, "ci_high": np.nan}
             for (split, method), frames in scored.groupby(["split", "method"], sort=False)]
    return pd.DataFrame(rows)


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
