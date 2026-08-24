"""Build the SoccerNet sample index from locally available 224p halves + Labels-v2.json.

Usage: ./run.sh python scripts/build_sample_soccernet.py [--videos DIR] [--labels DIR] [--out DIR]

Trusted records are `Shots on/off target` annotations with visibility == visible;
their goal label comes from pairing with a same-team `Goal` within 6 s (parser in
xgpoc.data.soccernet_labels). `goal_only` events are NOT anchored (per the codex-branch
audit a synthetic goal-minus-1s frame can show the ball already in the net); they go
to attempts_weak.csv.
"""

import argparse
from pathlib import Path

import cv2
import pandas as pd

from xgpoc.data.frames import extract_shot_frames, grass_ratio
from xgpoc.data.index import split_by_game
from xgpoc.data.soccernet_labels import parse_labels

OFFSETS = (-800, -400, 0, 400, 800)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=Path, default=Path("/home/kuba/projects/tactifoot_vision/data/soccernet_xg_smoke_224p"))
    ap.add_argument("--labels", type=Path, default=Path("/home/kuba/projects/tactifoot_vision/data/soccernet_as"))
    ap.add_argument("--out", type=Path, default=Path("/home/kuba/projects/tactifoot_vision/data/xg_shots_sample/claude_fable/soccernet"))
    a = ap.parse_args()
    frames_dir = a.out / "frames"
    rows, weak = [], []
    games = sorted({p.parent for p in a.videos.rglob("*_224p.mkv")})
    for game_dir in games:
        rel = game_dir.relative_to(a.videos)
        labels = a.labels / rel / "Labels-v2.json"
        if not labels.exists():
            print("no labels for", rel)
            continue
        for ev in parse_labels(labels):
            rec = dict(game=str(rel), half=ev.half, position_ms=ev.position_ms, label=ev.label,
                       team=ev.team, goal=ev.goal, visibility=ev.visibility, source_kind=ev.source)
            if ev.source != "shot":
                rec["reason"] = "goal_without_shot_annotation"
                weak.append(rec)
                continue
            if ev.visibility != "visible":
                rec["reason"] = "not_shown"
                weak.append(rec)
                continue
            video = game_dir / f"{ev.half}_224p.mkv"
            if not video.exists():
                rec["reason"] = "missing_video_half"
                weak.append(rec)
                continue
            shot_id = "soccernet__" + str(rel).replace("/", "_") + f"__{ev.half}__{ev.position_ms}"
            got = extract_shot_frames(video, ev.position_ms, shot_id, frames_dir, offsets_ms=OFFSETS)
            if 0 not in got:
                rec["reason"] = "frame_extraction_failed"
                weak.append(rec)
                continue
            img = cv2.imread(str(got[0]))
            rows.append(dict(shot_id=shot_id, source="soccernet", game=str(rel), split="",
                             goal=ev.goal, frame_path=str(got[0]), is_penalty=ev.is_penalty,
                             visibility=ev.visibility, label=ev.label, half=ev.half,
                             position_ms=ev.position_ms, grass_ratio=grass_ratio(img),
                             **{f"frame_path_{o:+d}": str(p) for o, p in got.items() if o != 0}))
    df = pd.DataFrame(rows)
    df["split"] = split_by_game(df["game"])
    a.out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out / "index.parquet", index=False)
    pd.DataFrame(weak).to_csv(a.out / "attempts_weak.csv", index=False)
    print(f"trusted shots with frames: {len(df)}  goals: {int(df['goal'].sum())}  "
          f"weak/excluded: {len(weak)}  games: {df['game'].nunique()}")
    print(df.groupby("label")["goal"].agg(["size", "mean"]).round(3))
    print("grass_ratio quartiles:", df["grass_ratio"].quantile([0.25, 0.5, 0.75]).round(2).to_dict())
    print("->", a.out / "index.parquet")


if __name__ == "__main__":
    main()
