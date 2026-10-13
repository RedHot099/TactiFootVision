"""GSR's ball as truth for the shot-location source of audit 3 (pitch-keypoints 04, council C3 Q3).

GSR clips are 30 s around a SoccerNet action (`info.action_class`, `info.action_position` in ms of the half,
`info.clip_start`); frame f (1-based, 25 fps) shows time `clip_start + 40 (f - 1)`. The shot clips (`SHOT_CLASSES`,
the classes of the xG index's shots) are anchored like the index's shots: their t0 is the latest frame at or before the
action time (`anchor_frame`), and offsets are whole frames from it (-400 ms = 10 frames earlier).

The ball's truth is GSR's `bbox_pitch` bottom-middle (`gsr_eval.to_corner`, corner metres) and its `bbox_image` box.
A frame with two ball annotations has no truth (`read_ball` drops it).

`timing_rows`: per shot clip, from the truth alone, where the ball is at t0, -400 and -800 ms relative to where the
ball's track shows the kick (`kick.kick` on the truth's track over the kick window `WINDOW`), and how far it moves
from -800 to -400 and from -400 to t0. This says which of the index's frames shows the ball nearest the spot the shot
was struck from, with no detector involved.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from xgpoc.detector import kick
from xgpoc.pitchkp.gsr_eval import to_corner

SHOT_CLASSES = ("Shots on target", "Shots off target")
FRAME_MS = 40
OFFSETS_MS = (0, -400, -800)
WINDOW = (-25, 7)                 # kick window in frames from t0: -1000 ms .. +280 ms (council C3: t0 - 1 s .. t0 + 0.3 s)


def anchor_frame(info: dict) -> int:
    """The 1-based frame of t0: the latest frame at or before the action time."""
    return int(np.floor((int(info["action_position"]) - int(info["clip_start"])) / FRAME_MS)) + 1


def read_ball(folder: Path) -> tuple[dict, pd.DataFrame]:
    """(`info`, ball truth per frame: frame, x, y (corner metres), x1, y1, x2, y2 (image pixels)) of one clip; frames
    with more than one ball annotation are left out."""
    labels = json.loads((Path(folder) / "Labels-GameState.json").read_text())
    ball_ids = {c["id"] for c in labels["categories"] if c["name"] == "ball"}
    frames = {image["image_id"]: int(Path(image["file_name"]).stem) for image in labels["images"]}
    rows = []
    for item in labels["annotations"]:
        if item.get("category_id") in ball_ids and item.get("bbox_pitch") and item["image_id"] in frames:
            p, b = item["bbox_pitch"], item["bbox_image"]
            x, y = to_corner(p["x_bottom_middle"], p["y_bottom_middle"])
            rows.append({"frame": frames[item["image_id"]], "x": float(x), "y": float(y), "x1": b["x"], "y1": b["y"],
                         "x2": b["x"] + b["w"], "y2": b["y"] + b["h"]})
    table = pd.DataFrame(rows, columns=["frame", "x", "y", "x1", "y1", "x2", "y2"])
    table = table[~table.frame.duplicated(keep=False)].sort_values("frame", ignore_index=True)
    return labels["info"], table


def track(ball: pd.DataFrame, frames: np.ndarray) -> np.ndarray:
    """`(len(frames), 2)` truth positions at the given frames, NaN where there is none."""
    by_frame = ball.set_index("frame")[["x", "y"]]
    return by_frame.reindex(frames).to_numpy(float)


def window_frames(t0: int) -> np.ndarray:
    return np.arange(t0 + WINDOW[0], t0 + WINDOW[1] + 1)


def timing_row(info: dict, ball: pd.DataFrame) -> dict:
    """The truth-only timing of one shot clip (module docstring)."""
    t0 = anchor_frame(info)
    frames = window_frames(t0)
    index, location = kick.kick(track(ball, frames))
    at = {o: track(ball, np.array([t0 + o // FRAME_MS]))[0] for o in OFFSETS_MS}
    row = {"clip": info["name"], "action_class": info["action_class"], "t0_frame": t0,
           "kick_ms": np.nan if index is None else float((frames[index] - t0) * FRAME_MS)}
    for o in OFFSETS_MS:
        row[f"to_kick_{abs(o)}"] = float(np.hypot(*(at[o] - location))) if index is not None else np.nan
    row["move_800_400"] = float(np.hypot(*(at[-400] - at[-800])))
    row["move_400_0"] = float(np.hypot(*(at[0] - at[-400])))
    return row
