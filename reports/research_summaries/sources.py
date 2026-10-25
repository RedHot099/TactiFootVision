"""Where the research summaries read their inputs (results stay in the research worktrees)."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/kuba/projects/tactifoot_vision")
XD = ROOT / ".claude/worktrees/research-xg-detector/experiments/claude_fable"
SHOTS = Path("/home/kuba/projects/football_data/xg_shots_full/soccernet_720p")
DETECTOR = ROOT / "models/detector_yolo11l_1280_dd2c.pt"
KEYPOINTS = ROOT / "models/pitch_yolo11m_pose_sn960.pt"
OUT = Path(__file__).resolve().parents[2] / "outputs/research_summaries"
# The shot (SoccerNet anchor) sits 4000 ms into every 8 s clip.
CLIP_ANCHOR_MS = 4000


def paired_shots() -> pd.DataFrame:
    """The 2,721 StatsBomb-paired primary shots with every model's xG and the outcome.

    p_video = G-combo (frame only), p_geometry = G-combo stacked with detector
    geometry (feet_t0 rule, ALL arm), p_vendor = StatsBomb xG. Data: StatsBomb.
    """
    sb = pd.read_parquet(XD / "results/xgc_compare/table/shots.parquet")
    labels = pd.read_parquet(XD / "results/xgc_compare/table/labels.parquet")
    sb = sb[sb.population == "primary"].merge(labels, on="key")
    base = pd.read_csv(XD / "results/sweep_2026_10/predictions.csv")
    feet = pd.read_csv(XD / "results/sweep_feet_2026_10/predictions.csv")
    preds = base[["shot_id", "p_g_combo", "p_stack_ALL_new"]].rename(
        columns={"p_g_combo": "p_video", "p_stack_ALL_new": "p_detector"}
    )
    preds = preds.merge(
        feet[["shot_id", "p_stack_ALL_new"]].rename(
            columns={"p_stack_ALL_new": "p_geometry"}
        ),
        on="shot_id",
    )
    df = sb.merge(preds, left_on="soccernet_shot_id", right_on="shot_id")
    df["p_vendor"] = 1 / (1 + np.exp(-df.z_vendor.astype(float)))
    df["goal"] = df.goal_sn.astype(int)
    df["clip"] = [SHOTS / "clips" / f"{s}.mp4" for s in df.shot_id]
    return df
