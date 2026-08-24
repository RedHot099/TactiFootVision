"""Production collector: SoccerNet 720p shot windows, game by game.

For each game (official splits via getListGames): download each half's 720p mkv,
extract JPEG frames at fixed offsets around every trusted shot (Labels-v2 parser),
optionally re-encode an 8 s window clip, record the raw asset (bytes, sha256, streams)
and delete the mkv. Fully resumable: a game with an existing shard parquet is skipped.

    ./run.sh python scripts/collect_soccernet.py --out DIR [--games N] [--splits train valid test]
        [--clips] [--keep-video] [--password ...]

Peak disk = one half (~1 GB) + outputs. Frames-only output for all 500 games ≈ 3 GB.
"""

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

import cv2
import pandas as pd

from xgpoc.data.frames import extract_shot_frames, grass_ratio
from xgpoc.data.soccernet_labels import parse_labels

OFFSETS = (-1600, -1200, -800, -400, 0, 400, 800)
LABELS_ROOT = Path("/home/kuba/projects/football_data/xg_shots_full/soccernet_labels")
MIN_FREE_GB = 5.0       # hard floor requested by the user
DOWNLOAD_HEADROOM_GB = 3.0  # a 720p half is ~1 GB; leave margin above the floor


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def wait_for_space(path: Path, need_gb: float, max_wait_s: int = 3600) -> bool:
    """Block until enough space is free (something else may release it); False = give up."""
    t0 = time.time()
    while free_gb(path) < need_gb:
        if time.time() - t0 > max_wait_s:
            return False
        print(f"low disk ({free_gb(path):.1f} GB free < {need_gb:.1f} GB needed), waiting 60 s...", flush=True)
        time.sleep(60)
    return True


def sha256(path: Path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def probe(path: Path) -> dict:
    r = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", "-show_format", str(path)],
                       capture_output=True, text=True)
    j = json.loads(r.stdout or "{}")
    s = next((s for s in j.get("streams", []) if s.get("codec_type") == "video"), {})
    return {"width": s.get("width"), "height": s.get("height"), "fps": s.get("avg_frame_rate"),
            "duration_s": float(j.get("format", {}).get("duration", 0) or 0)}


def cut_clip(video: Path, center_ms: int, out: Path, span_s: float = 4.0) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    start = max(center_ms / 1000 - span_s, 0)
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{start:.3f}", "-i", str(video), "-t", f"{2 * span_s:.3f}",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-an", str(out)]
    return subprocess.run(cmd, capture_output=True).returncode == 0 and out.exists()


def collect_game(dl, game: str, split: str, out: Path, clips: bool, keep_video: bool) -> tuple[list[dict], list[dict]]:
    labels = LABELS_ROOT / game / "Labels-v2.json"
    if not labels.exists():
        return [], [{"game": game, "component": "labels", "error": "labels_missing"}]
    events = parse_labels(labels)
    rows, assets = [], []
    frames_dir = out / "frames"
    for half in (1, 2):
        half_events = [e for e in events if e.half == half and e.source == "shot" and e.visibility == "visible"]
        weak = [e for e in events if e.half == half and not (e.source == "shot" and e.visibility == "visible")]
        video = out / "raw" / game / f"{half}_720p.mkv"
        if not half_events:
            continue
        for attempt in range(2):
            if video.exists() and probe(video)["duration_s"] >= 2400:
                break  # complete file (a half is at least 40 min)
            video.unlink(missing_ok=True)  # partial file from an interrupted run
            if not wait_for_space(out, MIN_FREE_GB + DOWNLOAD_HEADROOM_GB):
                assets.append({"game": game, "half": half, "component": "720p", "error": "disk_full_gave_up"})
                break
            try:
                dl.LocalDirectory = str(out / "raw")
                dl.downloadGame(game, files=[f"{half}_720p.mkv"], spl=split, verbose=False)
            except Exception as e:  # noqa: BLE001 - stalled/failed download, retry once
                print(f"download error {game} half {half} (attempt {attempt}): {type(e).__name__}", flush=True)
                if attempt == 1:
                    assets.append({"game": game, "half": half, "component": "720p", "error": f"{type(e).__name__}: {e}"[:200]})
        if not video.exists() or (meta := probe(video))["duration_s"] < 2400:
            video.unlink(missing_ok=True)
            assets.append({"game": game, "half": half, "component": "720p", "error": "download_incomplete"})
            continue
        assets.append({"game": game, "half": half, "component": "720p", "bytes": video.stat().st_size,
                       "sha256": sha256(video), "error": "", **meta})
        for ev in half_events:
            shot_id = "soccernet__" + game.replace("/", "_").replace(" ", "-") + f"__{half}__{ev.position_ms}"
            got = extract_shot_frames(video, ev.position_ms, shot_id, frames_dir, offsets_ms=OFFSETS)
            row = dict(shot_id=shot_id, source="soccernet", game=game, split=split, goal=ev.goal,
                       frame_path=str(got.get(0, "")), is_penalty=ev.is_penalty, visibility=ev.visibility,
                       label=ev.label, half=half, position_ms=ev.position_ms,
                       status="ok" if 0 in got else "frame_extraction_failed")
            if 0 in got:
                row["grass_ratio"] = grass_ratio(cv2.imread(str(got[0])))
                row.update({f"frame_path_{o:+d}": str(p) for o, p in got.items() if o != 0})
                if clips:
                    clip = out / "clips" / f"{shot_id}.mp4"
                    row["clip_path"] = str(clip) if cut_clip(video, ev.position_ms, clip) else ""
            rows.append(row)
        for ev in weak:
            rows.append(dict(shot_id="", source="soccernet", game=game, split=split, goal=ev.goal, frame_path="",
                             is_penalty=ev.is_penalty, visibility=ev.visibility, label=ev.label, half=half,
                             position_ms=ev.position_ms,
                             status="goal_without_shot_annotation" if ev.source != "shot" else "not_shown"))
        if not keep_video:
            video.unlink()
    return rows, assets


def main():
    from SoccerNet.Downloader import SoccerNetDownloader
    from SoccerNet.utils import getListGames

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--games", type=int, default=None, help="limit total games (pilot)")
    ap.add_argument("--clips", action="store_true")
    ap.add_argument("--keep-video", action="store_true")
    ap.add_argument("--password", default=os.environ.get("SOCCERNET_PASSWORD"))
    a = ap.parse_args()
    if not a.password:
        raise SystemExit("need --password or SOCCERNET_PASSWORD")
    socket.setdefaulttimeout(120)  # a stalled read raises instead of hanging forever
    dl = SoccerNetDownloader(LocalDirectory=str(a.out / "raw"))
    dl.password = a.password
    todo = [(g, s) for s in a.splits for g in getListGames(s, task="spotting")]
    if a.games:
        todo = todo[:a.games]
    shards = a.out / "shards"
    shards.mkdir(parents=True, exist_ok=True)
    done = 0
    for i, (game, split) in enumerate(todo):
        shard = shards / (game.replace("/", "__").replace(" ", "-") + ".parquet")
        if shard.exists():
            done += 1
            continue
        rows, assets = collect_game(dl, game, split, a.out, a.clips, a.keep_video)
        pd.DataFrame(rows).to_parquet(shard, index=False)
        reg = a.out / "raw_assets.jsonl"
        with reg.open("a") as f:
            for rec in assets:
                f.write(json.dumps(rec) + "\n")
        ok = sum(1 for r in rows if r.get("status") == "ok")
        print(f"[{i + 1}/{len(todo)}] {game} ({split}): {ok} shots ok / {len(rows)} events")
    all_rows = pd.concat([pd.read_parquet(p) for p in shards.glob("*.parquet")], ignore_index=True)
    trusted = all_rows[all_rows["status"] == "ok"].reset_index(drop=True)
    trusted.to_parquet(a.out / "index.parquet", index=False)
    all_rows.to_parquet(a.out / "attempts_all.parquet", index=False)
    print(f"index: {len(trusted)} trusted shots, {int(trusted['goal'].sum())} goals, "
          f"{trusted['game'].nunique()} games (previously done shards: {done})")
    print("->", a.out / "index.parquet")


if __name__ == "__main__":
    main()
