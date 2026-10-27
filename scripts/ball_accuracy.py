"""ball-detection 17: how accurately the frozen ball predictions are placed on the pitch (descriptive, CPU only).

    uv run python scripts/ball_accuracy.py read     # GSR valid (games 2+3, whole valid) and GSR test (ticket 09's)
    uv run python scripts/ball_accuracy.py shots    # the "shot moment" row on the 172 xG matches (StatsBomb target)

Predictions (all frozen, nothing chosen here; the placed point is the box centre, as in 09 and 11):

    top1      dd2c 1280's most confident ball >= 0.15, no filter (ticket 09's ``top1.parquet``)
    phase1    phase 1's candidate: ticket 09's tracks (45 px post-filter, ``track_ball`` with ticket 04's setting)
    s2_frame  ticket 13's second stage on a single frame: the most confident fused + regressed ball >= 0.15 after the
              45 px post-filter (``ball_crops.filtered_top1`` on ``ball/13/dump25_regress``; GSR valid only)
    s2_track  ticket 13's row (ii): the same dump tracked with ``ball/13/tune/chosen.json`` (GSR valid only; on games
              2+3 it must equal ``ball/13/read/track_ii``)

Frames: the scorecard's denominator (``detector_bench.placement_rows``: every 5th frame with one truth ball on the
pitch; no homography = no position). Numbers per group (games 2+3, whole GSR valid as in 09, GSR test): the share of
all frames within 0.5 / 1 / 2 / 5 m, the same among frames with a position, p50 / p90 / p95 of the error among frames
with a position, the share without a position and without a homography; 95% clip-bootstrap intervals with the
scorecard's draws (``ball_scorecard.Draws``, 2,000, seed 0). Breakdowns (point values, no tests): tracking status,
cover of the truth ball by person boxes and "in the air" (ticket 01's definitions, ``ball_errors``), ball size
(ticket 01's bins, 1080p pixels), zone (penalty area or not, by the truth) and the shot moment.

Reference agreement (the first result): the truth ball box (``bbox_image``) projected at its centre and at its bottom
middle through the frame's frozen homography vs GSR's own ``bbox_pitch`` (bottom middle), frame by frame. It mixes
homography error, the annotation convention and the ball's height; it is not subtracted from any prediction error.

Shot moments in GSR: every GSR clip is 30 s around one SoccerNet action (``info.action_class``,
``info.action_position`` and ``info.clip_start`` in ms of the half), so the action's frame is
(action_position - clip_start) / 40 ms + 1. Frames within ``MOMENT_FRAMES`` of a shot action (``SHOT_ACTIONS``) or a
goal are their own levels. Other actions inside a clip are unknown (GSR does not name the match, so the match's
Labels-v2 are not mapped).

`shots` (D4, descriptive): the non-penalty shots of the 135 evaluation matches paired with StatsBomb (xG compare's
``primary`` population). The t0 ball of xG today (X0: dd2c's top-1 >= 0.15 in the xG cache ``xgd_full4/E3``, the
shot's own frame, centre through the frame's homography) against StatsBomb's shot location (``statsbomb_to_metres``,
the ball aligned to the goal nearer to it, the xG convention); the frozen ``feet_t0`` shot location for context.
StatsBomb is the evaluation target only. Phase 1 and ticket 13 need new detections on these frames (the cache keeps
only the top-1 ball, ticket 16's dump covers the 328 development matches only): GPU, not done here.
"""

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ball_errors as be  # noqa: E402
import ball_scorecard as bs  # noqa: E402
import detector_bench as bench  # noqa: E402

import tactifoot_vision as tv  # noqa: E402

logger = logging.getLogger("ball_accuracy")

OUT = be.ROOT / "ball/17"
B09 = be.ROOT / "ball/09"
B13 = be.ROOT / "ball/13"
THRESHOLDS = (0.5, 1.0, 2.0, 5.0)
QUANTILES = (0.5, 0.9, 0.95)
FAR = bench.PLACEMENT["far_m"]
GROUPS = {
    "valid": {"games_2_3": ("2", "3"), "valid_all": ("2", "3", "5")},
    "test": {"test": None},
}
INPUT_SCALE = (
    1920 / bench.PLACEMENT["input_size"][0]
)  # native pixels per homography pixel
PENALTY_DEPTH, PENALTY_HALF_WIDTH = 16.5, 20.16
SHOT_ACTIONS = ("Shots on target", "Shots off target", "Penalty")
GOAL_ACTION = "Goal"
MOMENT_FRAMES = 25  # +- 1 s at 25 fps
FEATURES = ("status", "occlusion", "in_air", "size", "zone", "moment")
ACTION_MS_PER_FRAME = 40.0


# ------------------------------------------------------------------ measures
def measures(error: np.ndarray, homography: np.ndarray) -> dict[str, float]:
    """Point values of one predictor on denominator frames (``error``: NaN without a position)."""
    error = np.asarray(error, float)
    placed = np.isfinite(error)
    n, k = len(error), int(placed.sum())
    found = {
        "frames": float(n),
        "no_homography": float((~np.asarray(homography, bool)).mean()) if n else np.nan,
        "no_position": float((~placed).mean()) if n else np.nan,
    }
    for t in THRESHOLDS:
        within = error <= t
        found[f"le{t:g}_all"] = float(within.mean()) if n else np.nan
        found[f"le{t:g}_placed"] = float(within[placed].mean()) if k else np.nan
    found["far_of_placed"] = float((error[placed] > FAR).mean()) if k else np.nan
    found["far_of_all"] = float((error > FAR).mean()) if n else np.nan
    for q in QUANTILES:
        found[f"p{round(100 * q)}_m"] = (
            float(np.quantile(error[placed], q)) if k else np.nan
        )
    return found


def measure_draws(
    rows: pd.DataFrame, draws: bs.Draws, error_column: str = "error_m"
) -> dict[str, tuple[float, np.ndarray]]:
    """Point and per-draw values of ``measures`` (the scorecard's clip draws; a quantile per draw is the quantile of
    the errors repeated by the draw's clip weights, as ``ball_scorecard.placement_draws``' median)."""
    error = rows[error_column].to_numpy(float)
    homography = rows.homography.to_numpy(bool)
    placed = np.isfinite(error)
    clip = rows["clip"]
    ones = np.ones(len(rows))
    flags = {
        "no_homography": (~homography, ones),
        "no_position": (~placed, ones),
        "far_of_placed": (error > FAR, placed),
        "far_of_all": (error > FAR, ones),
    }
    for t in THRESHOLDS:
        flags[f"le{t:g}_all"] = (error <= t, ones)
        flags[f"le{t:g}_placed"] = (error <= t, placed)
    found = {
        name: bs.ratio_draws(
            draws, draws.per_clip(clip, numerator), draws.per_clip(clip, denominator)
        )
        for name, (numerator, denominator) in flags.items()
    }
    codes = draws.codes(clip[placed])
    errors = error[placed]
    per_draw = np.full((len(draws.weights), len(QUANTILES)), np.nan)
    for d, w in enumerate(draws.weights):
        repeats = w[codes]
        if repeats.sum():
            per_draw[d] = np.quantile(np.repeat(errors, repeats), QUANTILES)
    point = measures(error, homography)
    for i, q in enumerate(QUANTILES):
        name = f"p{round(100 * q)}_m"
        found[name] = (point[name], per_draw[:, i])
    return found


# ------------------------------------------------------------------ truth: features and reference agreement
def zone(truth_x: np.ndarray, truth_y: np.ndarray) -> np.ndarray:
    """Penalty area (either end, corner metres) or the rest, by the truth position."""
    x, y = np.asarray(truth_x, float), np.asarray(truth_y, float)
    depth = np.minimum(x, 2 * bench.HALF_LENGTH - x)
    inside = (depth <= PENALTY_DEPTH) & (
        np.abs(y - bench.HALF_WIDTH) <= PENALTY_HALF_WIDTH
    )
    return np.where(inside, "penalty area", "rest")


def action_frame(info: dict) -> float:
    """The 1-based frame of the clip's SoccerNet action."""
    return (
        int(info["action_position"]) - int(info["clip_start"])
    ) / ACTION_MS_PER_FRAME + 1


def moment(frame: np.ndarray, action: str, at: float) -> np.ndarray:
    """``shot`` / ``goal`` within ``MOMENT_FRAMES`` of the clip's shot or goal action, else ``other``."""
    near = np.abs(np.asarray(frame, float) - at) <= MOMENT_FRAMES
    level = (
        "shot" if action in SHOT_ACTIONS else "goal" if action == GOAL_ACTION else None
    )
    return np.where(near & (level is not None), level or "other", "other")


def ball_categories(table: pd.DataFrame) -> pd.DataFrame:
    """Ticket 01's size, occlusion and in-the-air levels (``ball_errors.add_box_features``' rules and constants) of
    rows with ``x1..y2`` (1080p pixels), ``coverage``, ``feet_m`` and ``deviation_m``."""
    out = table.copy()
    out["diameter"] = be.diameter(out[["x1", "y1", "x2", "y2"]].to_numpy(float))
    out["size"] = be.size_bin(out.diameter).astype(str)
    out["occlusion"] = np.select(
        [out.coverage <= 0, out.coverage < be.OCCLUDED], ["none", "<0.5"], ">=0.5"
    )
    out["in_air"] = np.select(
        [
            out.feet_m.isna() | out.deviation_m.isna(),
            (out.feet_m >= be.FREE_M) & (out.deviation_m >= be.AIR_DEVIATION_M),
        ],
        ["unknown", "in air"],
        "ground",
    )
    return out


def clip_truth(labels_path: Path, frames: set[int]) -> pd.DataFrame:
    """Per wanted frame the single truth ball with a pitch position: its box (native pixels, clipped to the image),
    cover by person boxes, distance to the nearest feet and the deviation of ticket 01 (``ball_errors.clip_labels``,
    ``track_motion``, ``coverage``), plus the clip's action."""
    info = json.loads(labels_path.read_text())["info"]
    balls, people = be.clip_labels(labels_path)
    balls = be.track_motion(balls)
    balls = balls[balls.frame.isin(frames) & balls.pitch_x.notna()]
    if balls.frame.duplicated().any():
        raise SystemExit(f"{labels_path}: several pitch balls on a denominator frame")
    rows = []
    for ball in balls.itertuples():
        box = np.clip(np.asarray(ball.box, float), 0, [1920, 1080, 1920, 1080])
        boxes, feet, _ = people.get(
            ball.frame, (np.zeros((0, 4)), np.zeros((0, 2)), np.zeros(0, str))
        )
        distances = np.linalg.norm(feet - [ball.pitch_x, ball.pitch_y], axis=1)
        distances = distances[np.isfinite(distances)]
        rows.append(
            {
                "frame": int(ball.frame),
                "x1": box[0],
                "y1": box[1],
                "x2": box[2],
                "y2": box[3],
                "coverage": be.coverage(box, boxes),
                "feet_m": distances.min() if len(distances) else np.nan,
                "deviation_m": ball.deviation_m,
            }
        )
    table = pd.DataFrame(rows)
    table["clip"] = labels_path.parent.name
    table["action"] = info["action_class"]
    table["action_frame"] = action_frame(info)
    return table


def truth_rows(split: str, frames: pd.DataFrame) -> pd.DataFrame:
    """``frames`` (denominator rows: clip, frame, truth_x, truth_y) with ticket 01's ball levels, zone and moment."""
    tables = []
    for path in bench.split_label_files(split):
        wanted = set(frames.frame[frames["clip"] == path.parent.name].astype(int))
        if wanted:
            tables.append(clip_truth(path, wanted))
    truth = ball_categories(pd.concat(tables, ignore_index=True))
    table = frames.merge(truth, on=["clip", "frame"], how="left", validate="1:1")
    if table.x1.isna().any():
        raise SystemExit("denominator frames without their truth ball")
    table["zone"] = zone(table.truth_x, table.truth_y)
    table["moment"] = "other"
    for (action, at), index in table.groupby(["action", "action_frame"]).groups.items():
        table.loc[index, "moment"] = moment(table.loc[index, "frame"], action, at)
    return table


def reference_errors(table: pd.DataFrame, homographies: dict) -> pd.DataFrame:
    """Per row (``clip``, ``frame``, box ``x1..y2`` native pixels, ``truth_x``/``truth_y`` from ``bbox_pitch``): the
    truth box's centre and bottom middle through the frame's frozen homography vs ``bbox_pitch`` (NaN without one)."""
    out = table.copy()
    for name in ("centre", "bottom"):
        out[f"ref_{name}_m"] = np.nan
    cx = (out.x1 + out.x2).to_numpy(float) / 2 / INPUT_SCALE
    points = {
        "centre": np.c_[cx, (out.y1 + out.y2).to_numpy(float) / 2 / INPUT_SCALE],
        "bottom": np.c_[cx, out.y2.to_numpy(float) / INPUT_SCALE],
    }
    truth = out[["truth_x", "truth_y"]].to_numpy(float)
    for i, key in enumerate(zip(out["clip"], out.frame.astype(int), strict=True)):
        matrix = homographies.get(key)
        if matrix is None:
            continue
        for name, xy in points.items():
            placed = tv.pitch.frame_to_pitch(
                np.asarray(xy[i : i + 1], np.float32), matrix
            )[0]
            out.iloc[i, out.columns.get_loc(f"ref_{name}_m")] = float(
                np.hypot(*(placed.astype(float) - truth[i]))
            )
    return out


# ------------------------------------------------------------------ predictions
def predictions(split: str, workers: int) -> dict[str, pd.DataFrame]:
    """The frozen per-frame balls (``chosen_balls``' columns + clip) of every predictor available on ``split``."""
    import ball_crops as bc  # noqa: PLC0415  (pulls the tracker; not needed by `shots`)
    import ball_temporal as bt  # noqa: PLC0415

    folder = B09 / ("track_valid" if split == "valid" else "track_test")
    found = {
        "top1": pd.read_parquet(folder / "top1.parquet"),
        "phase1": pd.read_parquet(folder / "tracks.parquet"),
    }
    if split != "valid":
        return found
    clips = [c.name for c in bt.clip_dirs(split)]
    dump = B13 / "dump25_regress"
    threshold = json.loads((B13 / "tune/single_threshold.json").read_text())[
        "threshold"
    ]
    found["s2_frame"] = bc.filtered_top1(clips, dump, threshold)
    setting = json.loads((B13 / "tune/chosen.json").read_text())
    _, (tracks,) = bt.run_tracks(
        clips, [setting], workers, dump, split, bt.MOTION_DIR, bt.POST_FILTER
    )
    found["s2_track"] = tracks
    return found


def check_s2_track(tracks: pd.DataFrame) -> dict:
    """Games 2+3 of the re-tracked row (ii) against ticket 13's frozen tracks."""
    frozen = pd.read_parquet(B13 / "read/track_ii/tracks.parquet")
    keys = ["clip", "frame"]
    mine = tracks[tracks["clip"].isin(set(frozen["clip"]))]
    mine = mine.sort_values(keys).reset_index(drop=True)[frozen.columns]
    frozen = frozen.sort_values(keys).reset_index(drop=True)
    same = mine.drop(columns="seconds", errors="ignore").equals(
        frozen.drop(columns="seconds", errors="ignore")
    )
    return {"frames": len(frozen), "identical": bool(same)}


def placement(table: pd.DataFrame, homographies: dict, split: str) -> pd.DataFrame:
    """``detector_bench.placement_rows`` with the table's ball centre, plus the predictor's status per frame."""
    import ball_temporal as bt  # noqa: PLC0415

    rows = bench.placement_rows(bt.table_top_ball(table), homographies, split)
    status = rows[["clip", "frame"]].merge(
        table[["clip", "frame", "status"]], on=["clip", "frame"], how="left"
    )["status"]
    if status.isna().any():
        raise SystemExit("a predictor misses denominator frames")
    rows["status"] = status.to_numpy()
    return rows


# ------------------------------------------------------------------ read
def group_summary(rows: pd.DataFrame, clips: list[str]) -> dict:
    draws = bs.Draws(clips)
    return bs.summary(measure_draws(rows, draws))


def breakdown(rows: pd.DataFrame, features: pd.DataFrame, predictor: str) -> list[dict]:
    """Point measures per level of every ``FEATURES`` column (descriptive)."""
    table = rows.merge(
        features[["clip", "frame", *[f for f in FEATURES if f != "status"]]],
        on=["clip", "frame"],
        how="left",
        validate="1:1",
    )
    out = []
    for feature in FEATURES:
        for level, part in table.groupby(feature, sort=True):
            out.append(
                {"predictor": predictor, "feature": feature, "level": str(level)}
                | measures(part.error_m.to_numpy(float), part.homography.to_numpy(bool))
                | {"share_of_frames": len(part) / len(table)}
            )
    return out


def reference_breakdown(features: pd.DataFrame) -> list[dict]:
    out = []
    for point in ("centre", "bottom"):
        for feature in [f for f in FEATURES if f != "status"]:
            for level, part in features.groupby(feature, sort=True):
                out.append(
                    {"point": point, "feature": feature, "level": str(level)}
                    | measures(
                        part[f"ref_{point}_m"].to_numpy(float),
                        part.homography.to_numpy(bool),
                    )
                )
    return out


def control_09(found: dict, split: str) -> dict:
    """Well placed (<= 5 m of all) and > 5 m of placed of top1 and phase1 against ticket 09's read, value and
    interval to the last digit."""
    record = json.loads((B09 / f"read_{split}/read.json").read_text())["groups"]
    names = {"games_2_3": "selection", "valid_all": "full", "test": "test"}
    out = {}
    for group, theirs in names.items():
        if group not in found:
            continue
        for mine, key in (("phase1", "track"), ("top1", "top1")):
            ours = found[group]["predictors"][mine]["summary"]
            for a, b in (
                ("le5_all", "well_placed"),
                ("far_of_placed", "far_of_placed"),
            ):
                out[f"{group}/{mine}/{a}"] = ours[a] == record[theirs][key][b]
    return out


def read(args: argparse.Namespace) -> None:
    import ball_temporal as bt  # noqa: PLC0415

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    record = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "ticket": "ball-detection 17",
        "registration": "ticket 17 comment 2026-10-27 13:54",
        "bootstrap": bs.BOOTSTRAP,
        "thresholds_m": THRESHOLDS,
        "quantiles": QUANTILES,
        "splits": {},
    }
    breakdown_rows, reference_rows, summary_rows = [], [], []
    for split in args.splits:
        homographies = bench.frozen_homographies(split)
        tables = predictions(split, args.workers)
        rows = {name: placement(t, homographies, split) for name, t in tables.items()}
        frames = rows["top1"][
            ["clip", "game", "frame", "truth_x", "truth_y", "homography"]
        ]
        features = reference_errors(truth_rows(split, frames), homographies)
        features.to_parquet(out / f"truth_{split}.parquet", index=False)
        clip_game = {c.name: bt.clip_game(c) for c in bt.clip_dirs(split)}
        found = {}
        for group, games in GROUPS[split].items():
            games = games or tuple(sorted(set(clip_game.values()), key=int))
            clips = [c for c, g in clip_game.items() if g in games]
            keep = frames.game.isin(games).to_numpy()
            part = features[keep].reset_index(drop=True)
            entry = {
                "games": list(games),
                "clips": len(clips),
                "frames": int(keep.sum()),
                "reference": {
                    point: group_summary(
                        part.assign(error_m=part[f"ref_{point}_m"]), clips
                    )
                    for point in ("centre", "bottom")
                },
                "predictors": {},
            }
            for point in ("centre", "bottom"):
                summary_rows.append(
                    {"split": split, "group": group, "row": f"reference {point}"}
                    | {k: v["value"] for k, v in entry["reference"][point].items()}
                )
            for name, table in rows.items():
                group_rows = table[keep].reset_index(drop=True)
                group_rows.to_parquet(
                    out / f"placement_{group}_{name}.parquet", index=False
                )
                entry["predictors"][name] = {
                    "summary": group_summary(group_rows, clips),
                    "status": group_rows.status.value_counts().to_dict(),
                }
                summary_rows.append(
                    {"split": split, "group": group, "row": name}
                    | {
                        k: v["value"]
                        for k, v in entry["predictors"][name]["summary"].items()
                    }
                )
                breakdown_rows += [
                    {"split": split, "group": group} | r
                    for r in breakdown(group_rows, part, name)
                ]
            reference_rows += [
                {"split": split, "group": group} | r for r in reference_breakdown(part)
            ]
            found[group] = entry
        clips_actions = (
            features.groupby("clip")[["action", "action_frame"]]
            .first()
            .action.value_counts()
        )
        record["splits"][split] = {
            "homographies": bench.homography_file(split),
            "groups": found,
            "control_09": control_09(found, split),
            "clip_actions": clips_actions.to_dict(),
        }
        if "s2_track" in tables:
            record["splits"][split]["control_s2_track_vs_13"] = check_s2_track(
                tables["s2_track"]
            )
        if "games_2_3" in found:
            record["splits"][split]["control_categories_vs_01"] = control_categories(
                features[features.game.isin(("2", "3"))]
            )
    pd.DataFrame(summary_rows).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(breakdown_rows).to_csv(out / "breakdown.csv", index=False)
    pd.DataFrame(reference_rows).to_csv(out / "reference_breakdown.csv", index=False)
    (out / "read.json").write_text(json.dumps(record, indent=2, default=_json) + "\n")
    text = markdown(record, pd.DataFrame(breakdown_rows), pd.DataFrame(reference_rows))
    (out / "read.md").write_text(text)
    print(text)


def _json(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return float(value)


def control_categories(features: pd.DataFrame) -> dict:
    """This ticket's size / occlusion / in-air levels against ticket 01's cached truth table of games 2+3 (single-ball
    frames)."""
    cached = pd.read_parquet(bs.TRUTH / "balls_gsr_games_23.parquet")
    cached = cached.groupby("image").filter(lambda g: len(g) == 1)
    mine = features.assign(
        image=features["clip"] + "/" + features.frame.map("{:06d}".format)
    )
    both = mine.merge(
        cached[["image", "size", "occlusion", "in_air"]],
        on="image",
        suffixes=("", "_01"),
    )
    return {"frames": len(both)} | {
        f: float((both[f].astype(str) == both[f"{f}_01"].astype(str)).mean())
        for f in ("size", "occlusion", "in_air")
    }


# ------------------------------------------------------------------ markdown
NAMES = {
    "top1": "top-1 dd2c ≥ 0,15",
    "phase1": "kandydat fazy 1",
    "s2_frame": "drugi stopień 13, pojedyncza klatka",
    "s2_track": "drugi stopień 13, śledzony (ii)",
    "reference centre": "zgodność referencji: środek ramki prawdy",
    "reference bottom": "zgodność referencji: dół ramki prawdy",
}


def _pct(entry: dict) -> str:
    low, high = (100 * v for v in entry["ci"])
    return f"{100 * entry['value']:.1f} [{low:.1f}; {high:.1f}]".replace(".", ",")


def _m(entry: dict) -> str:
    low, high = entry["ci"]
    return f"{entry['value']:.2f} [{low:.2f}; {high:.2f}]".replace(".", ",")


def summary_table(entries: dict[str, dict], placed: bool) -> list[str]:
    suffix = "placed" if placed else "all"
    head = "wśród klatek z pozycją" if placed else "wśród wszystkich klatek"
    lines = [
        f"| wiersz ({head}) | ≤ 0,5 m | ≤ 1 m | ≤ 2 m | ≤ 5 m | p50 m | p90 m | p95 m | bez pozycji | bez H |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in entries.items():
        lines.append(
            f"| {NAMES.get(name, name)} | "
            + " | ".join(_pct(s[f"le{t:g}_{suffix}"]) for t in THRESHOLDS)
            + f" | {_m(s['p50_m'])} | {_m(s['p90_m'])} | {_m(s['p95_m'])} | {_pct(s['no_position'])} | "
            f"{_pct(s['no_homography'])} |"
        )
    return lines


def breakdown_table(table: pd.DataFrame, value_column: str = "predictor") -> list[str]:
    lines = [
        f"| {value_column} | cecha | poziom | klatki | bez pozycji | ≤ 0,5 m | ≤ 1 m | ≤ 2 m | ≤ 5 m | p50 m | p90 m | "
        "> 5 m wśród z pozycją |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in table.itertuples():
        values = [
            100 * getattr(r, f"le{t:g}_all".replace(".", "_")) for t in THRESHOLDS
        ]
        lines.append(
            f"| {NAMES.get(getattr(r, value_column), getattr(r, value_column))} | {r.feature} | {r.level} | "
            f"{int(r.frames)} | {100 * r.no_position:.1f} | "
            + " | ".join(f"{v:.1f}" for v in values)
            + f" | {r.p50_m:.2f} | {r.p90_m:.2f} | {100 * r.far_of_placed:.1f} |"
        )
    return [
        line.replace(".", ",") if not line.startswith("|---") else line
        for line in lines
    ]


def markdown(record: dict, breakdowns: pd.DataFrame, references: pd.DataFrame) -> str:
    lines = ["# ball-detection 17: dokładność lokalizacji piłki (odczyt opisowy)", ""]
    breakdowns = breakdowns.rename(columns=lambda c: c.replace(".", "_"))
    references = references.rename(columns=lambda c: c.replace(".", "_"))
    for split, s in record["splits"].items():
        for group, g in s["groups"].items():
            lines += [
                f"## {split} / {group} (gry {', '.join(g['games'])}; {g['clips']} klipów, {g['frames']} klatek)",
                "",
            ]
            entries = {
                f"reference {p}": g["reference"][p] for p in ("centre", "bottom")
            }
            entries |= {n: p["summary"] for n, p in g["predictors"].items()}
            lines += (
                summary_table(entries, False)
                + [""]
                + summary_table(entries, True)
                + [""]
            )
            part = breakdowns[(breakdowns.split == split) & (breakdowns.group == group)]
            lines += breakdown_table(part) + [""]
            ref = references[(references.split == split) & (references.group == group)]
            lines += breakdown_table(ref, "point") + [""]
        lines += [
            f"Kontrole ({split}): 09 {json.dumps(s['control_09'])}",
            f"akcje klipów: {json.dumps(s['clip_actions'], ensure_ascii=False)}",
        ]
        for key in ("control_s2_track_vs_13", "control_categories_vs_01"):
            if key in s:
                lines.append(f"{key}: {json.dumps(s[key])}")
        lines.append("")
    lines.append(
        f"Bootstrap {bs.BOOTSTRAP['draws']} losowań po klipach, seed {bs.BOOTSTRAP['seed']}. Rozbicia: wartości punktowe."
    )
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ shots (the 172, StatsBomb as target)
def aligned_error(xy: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Distance of pitch points (corner metres, NaN rows allowed) to aligned ``target`` points, each point aligned to
    the goal nearer to it (``xgpoc.detector.conventions``)."""
    from xgpoc.detector.conventions import align, attacks_right  # noqa: PLC0415

    out = np.full(len(xy), np.nan)
    for i, point in enumerate(np.asarray(xy, float)):
        if np.isfinite(point).all():
            out[i] = float(np.hypot(*(align(point, attacks_right(point)) - target[i])))
    return out


def shot_rows() -> pd.DataFrame:
    """Per paired non-penalty shot of the evaluation matches: game, StatsBomb location (aligned metres), the X0 t0
    ball's pitch point and box, the feet_t0 shot location."""
    import ball_xg as bx  # noqa: PLC0415

    sys.path.insert(0, str(bx.XG))
    from xgpoc.detector.conventions import statsbomb_to_metres  # noqa: PLC0415
    from xgpoc.paths import INDEX  # noqa: PLC0415

    index = pd.read_parquet(INDEX, columns=["shot_id", "game", "is_penalty"])
    plan = pd.read_csv(bx.PLAN, usecols=["game", "role"])
    index = index[
        index.game.map(plan.set_index("game").role).eq("evaluation")
        & ~index.is_penalty.astype(bool)
    ]
    sb = pd.read_parquet(
        bx.XG / "results/xgc_compare/table/shots.parquet",
        columns=["soccernet_shot_id", "population", "sb_x", "sb_y"],
    )
    sb = sb[(sb.population == "primary") & sb.sb_x.notna()]
    shots = index.merge(sb, left_on="shot_id", right_on="soccernet_shot_id")
    if shots.shot_id.duplicated().any():
        raise SystemExit("a shot is paired twice")
    target = statsbomb_to_metres(shots[["sb_x", "sb_y"]].to_numpy(float))
    shots["sb_mx"], shots["sb_my"] = target[:, 0], target[:, 1]
    ids = shots.shot_id.tolist()
    frames = pd.read_parquet(
        bx.CACHE / "frames.parquet",
        columns=[
            "shot_id",
            "offset_ms",
            "readable",
            "homography",
            "ball_xyxy",
            "ball_pitch_xy",
            "frame_path",
        ],
        filters=[("shot_id", "in", ids), ("offset_ms", "==", 0)],
    ).set_index("shot_id")
    feet = pd.read_parquet(
        bx.FEET_CACHE / "features.parquet",
        columns=["shot_id", "offset_ms", "shot_ok", "shot_x", "shot_y"],
        filters=[("shot_id", "in", ids), ("offset_ms", "==", 0)],
    ).set_index("shot_id")
    shots = shots.set_index("shot_id")
    frames = frames.reindex(shots.index)
    on_pitch = bx.has_homography(frames.fillna({"readable": False}))
    ball_xy = np.full((len(shots), 2), np.nan)
    box = np.full((len(shots), 4), np.nan)
    for i, (xy, b) in enumerate(
        zip(frames.ball_pitch_xy, frames.ball_xyxy, strict=True)
    ):
        xy, b = (
            np.asarray(xy if xy is not None else [], float),
            np.asarray(b if b is not None else [], float),
        )
        if xy.size >= 2 and b.size >= 4:
            ball_xy[i], box[i] = xy.reshape(-1, 2)[0], b.reshape(-1, 4)[0]
    ball_xy[~on_pitch.to_numpy()] = np.nan
    shots["homography"] = on_pitch.to_numpy()
    shots["ball_x"], shots["ball_y"] = ball_xy[:, 0], ball_xy[:, 1]
    shots[["bx1", "by1", "bx2", "by2"]] = box
    shots["frame_path"] = frames.frame_path
    feet = feet.reindex(shots.index)
    ok = feet.shot_ok == 1
    shots["feet_x"], shots["feet_y"] = feet.shot_x.where(ok), feet.shot_y.where(ok)
    return shots.reset_index()


def shot_errors(shots: pd.DataFrame) -> pd.DataFrame:
    """Ball t0 (aligned to its nearer goal) and feet_t0 (already aligned) errors against StatsBomb, metres."""
    out = shots.copy()
    target = out[["sb_mx", "sb_my"]].to_numpy(float)
    out["ball_error_m"] = aligned_error(
        out[["ball_x", "ball_y"]].to_numpy(float), target
    )
    out["feet_error_m"] = np.hypot(out.feet_x - out.sb_mx, out.feet_y - out.sb_my)
    return out


def shots(args: argparse.Namespace) -> None:
    from PIL import Image  # noqa: PLC0415

    table = shot_errors(shot_rows())
    heights = np.array(
        [
            Image.open(p).size[1] if isinstance(p, str) else np.nan
            for p in table.frame_path
        ],
        float,
    )
    long_1080 = (
        np.maximum(table.bx2 - table.bx1, table.by2 - table.by1) * 1080 / heights
    )
    table["ball_long_1080p"] = long_1080
    draws = bs.Draws(table.game)
    found = {}
    for name in ("ball", "feet"):
        rows = pd.DataFrame(
            {
                "clip": table.game,
                "error_m": table[f"{name}_error_m"],
                "homography": table.homography.to_numpy(bool),
            }
        )
        found[name] = bs.summary(measure_draws(rows, draws))
    with_ball = table.ball_error_m.notna()
    record = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "ticket": "ball-detection 17",
        "population": {
            "shots": len(table),
            "matches": int(table.game.nunique()),
            "rule": "non-penalty shots of the plan's evaluation matches, StatsBomb primary pairs",
        },
        "bootstrap": {"over": "matches"} | bs.BOOTSTRAP,
        "x0_t0_ball": found["ball"],
        "feet_t0_location": found["feet"],
        "x0_t0_ball_long_side_over_45px": float((long_1080[with_ball] > 45).mean()),
        "x0_t0_ball_long_side_over_45px_count": int((long_1080[with_ball] > 45).sum()),
        "not_done": "P1 and S2: new detections on the 172's frames (GPU); the cache holds the top-1 ball only",
    }
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    table.drop(columns=["sb_x", "sb_y", "sb_mx", "sb_my"]).to_parquet(
        out / "shots172.parquet", index=False
    )
    (out / "shots172.json").write_text(
        json.dumps(record, indent=2, default=_json) + "\n"
    )
    lines = summary_table(
        {"X0 piłka t0": found["ball"], "feet_t0": found["feet"]}, False
    )
    lines += [""] + summary_table(
        {"X0 piłka t0": found["ball"], "feet_t0": found["feet"]}, True
    )
    text = (
        "\n".join(lines)
        + "\n\n"
        + json.dumps(
            {
                k: record[k]
                for k in (
                    "population",
                    "x0_t0_ball_long_side_over_45px",
                    "x0_t0_ball_long_side_over_45px_count",
                )
            }
        )
        + "\n"
    )
    (out / "shots172.md").write_text(text)
    print(text)


# ------------------------------------------------------------------ commands
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    r = commands.add_parser("read", help="GSR valid and GSR test accuracy report")
    r.add_argument("--splits", nargs="+", choices=list(GROUPS), default=list(GROUPS))
    r.add_argument("--workers", type=int, default=12)
    r.add_argument("--out", type=Path, default=OUT)
    s = commands.add_parser(
        "shots", help="the shot-moment row on the 172 (StatsBomb as target)"
    )
    s.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    tv.setup_logging("WARNING")
    {"read": read, "shots": shots}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
