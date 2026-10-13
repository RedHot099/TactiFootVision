"""GSR evidence for audit 3's shot-location sources (pitch-keypoints 04, council C3 Q3). No StatsBomb data.

    ./run.sh python scripts/pk_ball_evidence.py [--out results/pk_ball_evidence] [--device cuda]

Three parts, all on SoccerNet GSR (`xgpoc.pitchkp.gsr_ball` for the conventions); GSR test is never read.

1. **Truth timing** (`timing.csv`), GSR train and valid shot clips (`gsr_ball.SHOT_CLASSES`, the xG index's classes,
   anchored like its shots): from GSR's ball positions alone, how far the ball at t0, -400 and -800 ms lies from the
   ball where its track shows the kick (`kick.kick` over t0 - 1 s .. t0 + 0.28 s), and how far it moves from -800 to
   -400 and from -400 to t0. Annotations only (GSR train keeps every 5th image, but its labels cover every frame).
2. **The pipeline on the shot clips** (`shot_frames.parquet`, `shot_sources.csv`), GSR valid shot clips: every frame of
   the kick window at 720p through audit 3's ball-only analyzer (`extract3.build_analyzers`: C4's detector, the
   SoccerNet pitch model on its square input, keypoint threshold 0.3, RANSAC 2 m, line refinement with the detector's
   people boxes). Per frame the placed ball (the most confident ball, its box centre through the frame's homography,
   as `FrameAnalyzer` does) against the truth of that frame. Per clip and source rule (`extract3.CONFIGS`' orders
   over the t0 / -400 / -800 ms frames, and the kick rule on the placed ball's track): the offset used and the error
   against the truth's kick location of part 1.
3. **Ball placement on GSR valid** (`ball_frames.parquet`, `ball.csv`): every 5th frame of every GSR valid clip (the
   frames of the homography evaluation), the detector's most confident ball placed by ticket 03's cached homography
   of the chosen method (`kp03_r2_refined`, GSR valid retune run), against the truth: share of frames with a truth
   ball on the pitch where a ball is placed, the error, and the share placed > 5 m away (another object, or a ball in
   the air); near goal (truth ball within 30 m of a goal line) separately; clip-bootstrap intervals (2,000 draws).

Writes the files above, `summary.json` and `complete.json` (last). Runs inside `no_statsbomb_reads()`.
"""

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pandas as pd
import supervision as sv
from tqdm import tqdm
from xgpoc.bundle import sha256_file
from xgpoc.detector import extract3 as e3
from xgpoc.detector import kick, records
from xgpoc.detector.guard import no_statsbomb_reads
from xgpoc.pitchkp import evaluator as ev
from xgpoc.pitchkp import gsr_ball as gb
from xgpoc.pitchkp import gsr_eval as ge

from tactifoot_vision.pitch import frame_to_pitch

ROOT = Path(__file__).resolve().parents[1]
GSR = records.CHECKOUT / "data/SoccerNetGS"
RETUNE = ROOT / "results/pk_candidates/y11m_960_e18_square/retune_valid"
METHOD = "kp03_r2_refined"
FAR_M = 5.0


def clips(split: str) -> list[Path]:
    return sorted(p.parent for p in (GSR / split).glob("*/Labels-GameState.json"))


def placed_ball(result) -> np.ndarray:
    """The ball a feature row would use from this frame: the first ball with a pitch position, or NaN."""
    xy = result.ball_xy.astype(float) if result is not None else np.zeros((0, 2))
    finite = np.isfinite(xy).all(axis=1)
    return xy[finite][0] if finite.any() else np.full(2, np.nan)


def distance(a, b) -> float:
    return float(np.hypot(*(np.asarray(a, float) - np.asarray(b, float))))


def shot_clip_rows(folder: Path, analyzer) -> tuple[list[dict], list[dict]]:
    """Part 2 for one GSR valid shot clip: per-frame rows and per-rule rows."""
    info, ball = gb.read_ball(folder)
    t0 = gb.anchor_frame(info)
    frames = gb.window_frames(t0)
    truth = gb.track(ball, frames)
    kick_index, kick_truth = kick.kick(truth)
    results, frame_rows = {}, []
    for frame, true_xy in zip(frames, truth, strict=True):
        result = analyzer(ev.eval_frame(folder / "img1" / f"{frame:06d}.jpg"))
        offset = int((frame - t0) * gb.FRAME_MS)
        results[offset] = result
        xy = placed_ball(result)
        frame_rows.append({"clip": info["name"], "offset_ms": offset, "truth_x": true_xy[0], "truth_y": true_xy[1],
                           "x": xy[0], "y": xy[1], "homography": result.homography is not None,
                           "error_m": distance(xy, true_xy)})
    rows = []
    base = {"clip": info["name"], "truth_kick_ms": np.nan if kick_index is None else float((frames[kick_index] - t0) * gb.FRAME_MS)}
    for config in e3.CONFIGS.values():
        picked = e3.chosen_ball(results, config.order(0))
        xy = placed_ball(results[picked[1]]) if picked else np.full(2, np.nan)
        rows.append(base | {"rule": config.name, "source_ms": np.nan if picked is None else float(picked[1]),
                            "to_truth_kick_m": distance(xy, kick_truth) if kick_index is not None else np.nan})
    track = np.array([placed_ball(results[int((f - t0) * gb.FRAME_MS)]) for f in frames])
    index, location = kick.kick(track)
    rows.append(base | {"rule": "kick", "source_ms": np.nan if index is None else float((frames[index] - t0) * gb.FRAME_MS),
                        "to_truth_kick_m": distance(location, kick_truth) if index is not None and kick_index is not None
                        else np.nan})
    return frame_rows, rows


def ball_rows(folder: Path, detector, homographies: pd.DataFrame) -> list[dict]:
    """Part 3 for one GSR valid clip: every scored frame with a truth ball."""
    info, ball = gb.read_ball(folder)
    truth = ball.set_index("frame")
    clip = ge.read_clip(folder)
    rows = []
    for frame in clip.frames.itertuples(index=False):
        if frame.frame not in truth.index:
            continue
        true_xy = truth.loc[frame.frame, ["x", "y"]].to_numpy(float)
        if not ((abs(true_xy[0] - ge.HALF_LENGTH) <= ge.HALF_LENGTH) and (abs(true_xy[1] - ge.HALF_WIDTH) <= ge.HALF_WIDTH)):
            continue
        matrix = homographies.get((clip.name, frame.frame))
        detections = detector(ev.eval_frame(frame.path))
        balls = detections[detections.data["class_name"] == "ball"]
        xy = np.full(2, np.nan)
        if len(balls) and matrix is not None:
            best = balls[[int(np.argmax(balls.confidence))]]
            xy = frame_to_pitch(best.get_anchors_coordinates(sv.Position.CENTER), matrix)[0].astype(float)
        rows.append({"clip": clip.name, "game": clip.game, "frame": frame.frame, "truth_x": true_xy[0], "truth_y": true_xy[1],
                     "near_goal": bool(ge.near_goal(true_xy[0])), "homography": matrix is not None, "detected": len(balls) > 0,
                     "x": xy[0], "y": xy[1], "error_m": distance(xy, true_xy)})
    return rows


def ball_statistics(table: pd.DataFrame) -> dict:
    placed = np.isfinite(table.error_m.to_numpy(float))
    error = table.error_m.to_numpy(float)[placed]
    n = len(table)
    out = {"frames": float(n), "placed": placed.mean() if n else np.nan,
           "median_m": float(np.median(error)) if len(error) else np.nan,
           "p90_m": float(np.quantile(error, 0.9)) if len(error) else np.nan}
    for t in (1.0, 2.0, FAR_M):
        out[f"lt{t:g}"] = float((error < t).mean()) if len(error) else np.nan
    out["far"] = float((error > FAR_M).mean()) if len(error) else np.nan
    return out


def ball_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stratum, part in (("all", table), ("near_goal", table[table.near_goal])):
        part = part.reset_index(drop=True)
        point = ball_statistics(part)
        draws = ge.bootstrap(part["clip"].to_numpy(), lambda idx, part=part: ball_statistics(part.iloc[idx]))
        rows += ge.interval_rows(point, draws, {"stratum": stratum})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=ROOT / "results/pk_ball_evidence")
    parser.add_argument("--device", default=None)
    parser.add_argument("--clips", type=int, default=None, help="first N clips of part 3 (smoke runs)")
    args = parser.parse_args()
    if (args.out / "complete.json").exists():
        raise SystemExit(f"{args.out} is complete; use a new --out")
    records.verified_file(RETUNE / "per_frame.parquet")
    if sha256_file(e3.KEYPOINT_WEIGHTS) != e3.KEYPOINT_SHA256:
        raise SystemExit(f"{e3.KEYPOINT_WEIGHTS} is not the chosen pitch model")

    with no_statsbomb_reads():
        started, clock = datetime.now(UTC), time.perf_counter()
        shot_clips = {split: [c for c in clips(split) if gb.read_ball(c)[0]["action_class"] in gb.SHOT_CLASSES]
                      for split in ("train", "valid")}
        timing = pd.DataFrame([{"split": split} | gb.timing_row(*gb.read_ball(c))
                               for split, folders in shot_clips.items() for c in folders])

        ball_only = e3.ball_analyzer(args.device)
        frame_rows, source_rows = [], []
        for folder in tqdm(shot_clips["valid"], desc="shot clips"):
            f, s = shot_clip_rows(folder, ball_only)
            frame_rows += f
            source_rows += s
        shot_frames, sources = pd.DataFrame(frame_rows), pd.DataFrame(source_rows)

        retune = pd.read_parquet(RETUNE / "per_frame.parquet", columns=["clip", "frame", "method", "homography"])
        retune = retune[(retune.method == METHOD) & retune.homography.notna()]
        homographies = {(c, int(f)): np.asarray(h, float).reshape(3, 3)
                        for c, f, h in zip(retune["clip"], retune["frame"], retune["homography"], strict=True)}
        detector = ball_only.detector
        ball = pd.DataFrame([row for folder in tqdm(clips("valid")[: args.clips], desc="ball", mininterval=30)
                             for row in ball_rows(folder, detector, homographies)])
        gpu_done = datetime.now(UTC)

        args.out.mkdir(parents=True, exist_ok=True)
        timing.to_csv(args.out / "timing.csv", index=False, float_format="%.4f")
        shot_frames.to_parquet(args.out / "shot_frames.parquet", index=False)
        sources.to_csv(args.out / "shot_sources.csv", index=False, float_format="%.4f")
        ball.to_parquet(args.out / "ball_frames.parquet", index=False)
        ball_table = ball_summary(ball)
        ball_table.to_csv(args.out / "ball.csv", index=False, float_format="%.4f")
        summary = summarise(timing, shot_frames, sources)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        record = {"purpose": "pitch-keypoints 04: GSR evidence for audit 3's shot-location sources (no StatsBomb data)",
                  "settings": e3.COMMON | {"keypoint_weights": str(e3.KEYPOINT_WEIGHTS), "keypoint_sha256": e3.KEYPOINT_SHA256},
                  "kick": {"span": kick.SPAN, "v_shot": kick.V_SHOT, "dv_kick": kick.DV_KICK, "max_gap": kick.MAX_GAP,
                           "window_frames": list(gb.WINDOW)},
                  "inputs": {"gsr": str(GSR), "shot_clips": {k: [c.name for c in v] for k, v in shot_clips.items()},
                             "retune": {"path": str(RETUNE), "method": METHOD,
                                        "complete_json_sha256": sha256_file(RETUNE / "complete.json")}},
                  "code": records.code_hashes(Path(__file__)), "git": records.git_state(),
                  "libraries": records.library_versions(),
                  "run": {"started": started.isoformat(timespec="seconds"), "gpu_done": gpu_done.isoformat(timespec="seconds"),
                          "finished": datetime.now(UTC).isoformat(timespec="seconds"),
                          "seconds": round(time.perf_counter() - clock, 1), "device": args.device},
                  "files": records.output_hashes(args.out)}
        (args.out / "complete.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2))
    print(ball_table.round(3).to_string(index=False))


def summarise(timing: pd.DataFrame, shot_frames: pd.DataFrame, sources: pd.DataFrame) -> dict:
    """The medians the declaration quotes."""
    with_kick = timing[timing.kick_ms.notna()]
    out = {"timing": {"clips": len(timing), "by_split": timing.split.value_counts().to_dict(), "with_kick": len(with_kick),
                      "kick_ms_median": float(with_kick.kick_ms.median()),
                      **{f"to_kick_{o}_median_m": float(with_kick[f"to_kick_{o}"].median()) for o in (0, 400, 800)},
                      **{f"to_kick_{o}_n": int(with_kick[f"to_kick_{o}"].notna().sum()) for o in (0, 400, 800)},
                      "move_800_400_median_m": float(timing.move_800_400.median()),
                      "move_400_0_median_m": float(timing.move_400_0.median()),
                      "move_n": int((timing.move_800_400.notna() & timing.move_400_0.notna()).sum())}}
    frames = shot_frames.assign(has_truth=shot_frames.truth_x.notna(), placed=shot_frames.x.notna())
    out["shot_frames"] = {"frames": len(frames), "clips": int(frames["clip"].nunique()),
                          "truth": float(frames.has_truth.mean()), "placed": float(frames.placed.mean()),
                          "placed_where_truth": float(frames.placed[frames.has_truth].mean()),
                          "median_error_m": float(frames.error_m.median()),
                          "far_share": float((frames.error_m.dropna() > FAR_M).mean())}
    for offset in gb.OFFSETS_MS:
        at = frames[frames.offset_ms == offset]
        out["shot_frames"][f"at_{abs(offset)}"] = {"placed": int(at.placed.sum()), "of": len(at),
                                                   "median_error_m": float(at.error_m.median())}
    out["sources"] = {rule: {"clips": len(part), "located": int(part.source_ms.notna().sum()),
                             "compared": int(part.to_truth_kick_m.notna().sum()),
                             "median_to_truth_kick_m": float(part.to_truth_kick_m.median()),
                             "within_5m": int((part.to_truth_kick_m <= FAR_M).sum())}
                      for rule, part in sources.groupby("rule", sort=True)}
    return out


if __name__ == "__main__":
    main()
