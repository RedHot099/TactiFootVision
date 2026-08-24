"""Build the SoccerTrack v2 sample index from files already downloaded on the codex branch.

Usage: ./run.sh python scripts/build_sample_soccertrack.py [--out DIR]

Only trusted attempts (SHOT / HEADER / FREE KICK anchors) get frames; weak records
(goal_without_candidate) are listed in attempts_weak.csv for accounting.
"""

import argparse
from pathlib import Path

import cv2
import pandas as pd

from xgpoc.data.frames import extract_shot_frames, grass_ratio
from xgpoc.data.index import split_by_game
from xgpoc.data.soccertrack_bas import parse_bas

CODEX_DATA = Path("/home/kuba/projects/football_data/xg_shots_full/soccertrack_v2")
OFFSETS = (-800, -400, 0, 400, 800)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/home/kuba/projects/football_data/xg_shots_full/samples/soccertrack"))
    a = ap.parse_args()
    frames_dir = a.out / "frames"
    rows, weak = [], []
    for bas in sorted((CODEX_DATA / "bas").glob("*_12_class_events.json")):
        match_id = bas.name.split("_")[0]
        for at in parse_bas(bas):
            rec = dict(match_id=match_id, half=at.half, position_ms=at.position_ms, video_ms=at.video_ms,
                       label=at.label, team=at.team, goal=at.goal, trusted=at.trusted, reason=at.reason)
            if not at.trusted:
                weak.append(rec)
                continue
            video = CODEX_DATA / "videos" / match_id / f"{match_id}_{'1st' if at.half == 1 else '2nd'}.mp4"
            if not video.exists():
                rec["reason"] = "missing_video_half"
                weak.append(rec)
                continue
            shot_id = f"soccertrack__{match_id}__{at.half}__{at.position_ms}"
            got = extract_shot_frames(video, at.video_ms, shot_id, frames_dir, offsets_ms=OFFSETS)
            if 0 not in got:
                rec["reason"] = "frame_extraction_failed"
                weak.append(rec)
                continue
            img = cv2.imread(str(got[0]))
            rows.append(dict(shot_id=shot_id, source="soccertrack", game=match_id, split="",
                             goal=at.goal, frame_path=str(got[0]), is_penalty=False,
                             visibility="visible", label=at.label, half=at.half,
                             position_ms=at.position_ms, grass_ratio=grass_ratio(img),
                             **{f"frame_path_{o:+d}": str(p) for o, p in got.items() if o != 0}))
    df = pd.DataFrame(rows)
    df["split"] = split_by_game(df["game"])
    a.out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out / "index.parquet", index=False)
    pd.DataFrame(weak).to_csv(a.out / "attempts_weak.csv", index=False)
    print(f"trusted shots with frames: {len(df)}  goals: {int(df['goal'].sum())}  "
          f"weak/excluded: {len(weak)}  matches: {df['game'].nunique()}")
    print(df.groupby("label")["goal"].agg(["size", "mean"]).round(3))
    print("splits:", df["split"].value_counts().to_dict())
    print("->", a.out / "index.parquet")


if __name__ == "__main__":
    main()
