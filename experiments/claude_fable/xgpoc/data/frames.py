"""Extract frames around shot timestamps from SoccerNet half-match videos.

Uses ffmpeg fast seek (-ss before -i) which is accurate to the nearest keyframe plus
decode; for 720p SoccerNet mkv this is well within the ~1 s label precision. Each shot
yields one JPEG per offset (e.g. -400, 0, +400 ms) so we can later pick the best frame
and test whether a short clip helps.
"""

import subprocess
from pathlib import Path

import cv2
import numpy as np

DEFAULT_OFFSETS_MS = (-800, -400, 0, 400, 800)


def frame_path(out_dir: Path, shot_id: str, offset_ms: int) -> Path:
    return out_dir / shot_id / f"{offset_ms:+d}.jpg"


def extract_frame(video: Path, position_ms: int, out: Path, height: int | None = None, quality: int = 3) -> bool:
    out.parent.mkdir(parents=True, exist_ok=True)
    vf = ["-vf", f"scale=-2:{height}"] if height else []
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{max(position_ms, 0) / 1000:.3f}", "-i", str(video),
           "-frames:v", "1", *vf, "-q:v", str(quality), str(out)]
    res = subprocess.run(cmd, capture_output=True)
    return res.returncode == 0 and out.exists()


def extract_shot_frames(video: Path, position_ms: int, shot_id: str, out_dir: Path,
                        offsets_ms=DEFAULT_OFFSETS_MS, height: int | None = None) -> dict[int, Path]:
    done = {}
    for off in offsets_ms:
        p = frame_path(out_dir, shot_id, off)
        if p.exists() or extract_frame(video, position_ms + off, p, height=height):
            done[off] = p
    return done


def grass_ratio(img_bgr: np.ndarray) -> float:
    """Fraction of green pixels; a cheap filter for replays/close-ups/crowd shots."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 40, 40), (90, 255, 255))
    return float(mask.mean() / 255.0)
