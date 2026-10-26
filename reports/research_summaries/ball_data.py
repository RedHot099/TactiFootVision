"""Per-frame ball data of one GSR test clip: truth, the detector's top-1 and the tracked ball.

Everything is read from the ball-detection phase outputs (ticket 09, GSR test at 25 fps); nothing is
recomputed.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BALL = Path("/mnt/storage/tactifoot_research/detector_data/ball")
GSR = Path("/home/kuba/projects/tactifoot_vision/data/SoccerNetGS")
# GSR test keeps every 5th frame; the same clips at 25 fps are SoccerNet Tracking test (SNMOT-NNN = SNGS-NNN,
# byte-identical frames checked by the phase).
FULL_FRAMES = Path(
    "/home/kuba/projects/tactifoot_vision/data/soccernet/tracking/extracted/test"
)
BD = Path(
    "/home/kuba/projects/tactifoot_vision/.claude/worktrees/research-ball-detection"
)
WELL_M = 5.0


@dataclass
class ClipBalls:
    clip: str
    frames: (
        pd.DataFrame
    )  # one row per frame: truth / top1 / track image centres and the track status
    placement: (
        pd.DataFrame
    )  # every 5th frame: truth and both picks on the pitch, errors in metres

    def image(self, frame: int) -> Path:
        return (
            FULL_FRAMES
            / self.clip.replace("SNGS-", "SNMOT-")
            / "img1"
            / f"{frame:06d}.jpg"
        )


def _truth(clip: str) -> pd.DataFrame:
    labels = json.loads((GSR / "test" / clip / "Labels-GameState.json").read_text())
    frame_of = {im["image_id"]: int(im["file_name"][:-4]) for im in labels["images"]}
    rows = [
        (
            frame_of[a["image_id"]],
            a["bbox_image"]["x_center"],
            a["bbox_image"]["y_center"],
            a["bbox_image"]["w"],
        )
        for a in labels["annotations"]
        if a["category_id"] == 4
    ]
    return pd.DataFrame(
        rows, columns=["frame", "truth_u", "truth_v", "truth_w"]
    ).drop_duplicates("frame")


def _top1(clip: str) -> pd.DataFrame:
    """The phase's reference pick (ticket 09): dd2c 1280's most confident ball >= 0.15, no filter."""
    top = pd.read_parquet(
        BALL / "09/track_test/top1.parquet", filters=[("clip", "==", clip)]
    )
    top = top[top.status == "measured"]
    return pd.DataFrame(
        {
            "frame": top.frame,
            "top_u": (top.x1 + top.x2) / 2,
            "top_v": (top.y1 + top.y2) / 2,
            "top_w": np.maximum(top.x2 - top.x1, top.y2 - top.y1),
        }
    )


def _track(clip: str) -> pd.DataFrame:
    tracks = pd.read_parquet(
        BALL / "09/track_test/tracks.parquet", filters=[("clip", "==", clip)]
    )
    return pd.DataFrame(
        {
            "frame": tracks.frame,
            "status": tracks.status,
            "track_u": (tracks.x1 + tracks.x2) / 2,
            "track_v": (tracks.y1 + tracks.y2) / 2,
        }
    )


def load(clip: str) -> ClipBalls:
    frames = pd.DataFrame({"frame": np.arange(1, 751)})
    for part in (_truth(clip), _top1(clip), _track(clip)):
        frames = frames.merge(part, on="frame", how="left")
    track = pd.read_parquet(BALL / "09/read_test/placement_test_track.parquet")
    top = pd.read_parquet(BALL / "09/read_test/placement_test_top1.parquet")
    keys = ["clip", "frame", "truth_x", "truth_y", "homography"]
    placement = track[track["clip"] == clip][[*keys, "x", "y", "error_m"]].merge(
        top[top["clip"] == clip][["frame", "x", "y", "error_m"]],
        on="frame",
        suffixes=("_track", "_top"),
    )
    return ClipBalls(
        clip, frames, placement.sort_values("frame").reset_index(drop=True)
    )


def phase_numbers() -> dict:
    """The phase's headline numbers, read from the ticket 09 records (not retyped)."""
    return {
        "test": json.loads((BALL / "09/read_test/read.json").read_text()),
        "valid": json.loads((BALL / "09/read_valid/read.json").read_text()),
    }
