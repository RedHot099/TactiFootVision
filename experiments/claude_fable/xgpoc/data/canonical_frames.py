"""Native clip frames with checked timestamps, lossless storage and explicit quality measurements."""

import hashlib
import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from xgpoc.data.datasets import load_image, to_tensor


@dataclass(frozen=True)
class VideoTimeline:
    time_base: str
    pts: tuple[int, ...]
    width: int
    height: int
    codec: str
    pixel_format: str

    def milliseconds(self, index: int) -> float:
        return float(self.pts[index] * Fraction(self.time_base) * 1000)

    def relative_ms(self, index: int, anchor_ms: float) -> float:
        return round(float((self.pts[index] - self.pts[0]) * Fraction(self.time_base) * 1000) - anchor_ms, 6)


@dataclass(frozen=True)
class DecodedFrame:
    index: int
    pts: int
    relative_ms: float
    bgr: np.ndarray
    previous_change: float | None


def probe_video(path: Path) -> VideoTimeline:
    raw = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames", "-show_streams",
        "-show_entries", "frame=best_effort_timestamp:stream=time_base,width,height,codec_name,pix_fmt",
        "-of", "json", str(path),
    ], stderr=subprocess.PIPE)
    data = json.loads(raw)
    if len(data.get("streams", [])) != 1 or not data.get("frames"):
        raise ValueError("Expected one decodable video stream")
    stream = data["streams"][0]
    if any("best_effort_timestamp" not in f for f in data["frames"]):
        raise ValueError("Every frame needs a timestamp")
    pts = tuple(int(f["best_effort_timestamp"]) for f in data["frames"])
    if any(b <= a for a, b in zip(pts, pts[1:])) or Fraction(stream["time_base"]) <= 0:
        raise ValueError("Video timestamps must strictly increase")
    return VideoTimeline(stream["time_base"], pts, stream["width"], stream["height"],
                         stream["codec_name"], stream["pix_fmt"])


def extract_inputs(path: Path, lower_ms: float | None, anchor_ms: float = 4000) -> tuple[VideoTimeline, dict[str, DecodedFrame]]:
    if not np.isfinite(anchor_ms) or anchor_ms < 0 or lower_ms is not None and not np.isfinite(lower_ms):
        raise ValueError("Frame boundaries must be finite; anchor must be nonnegative")
    timeline = probe_video(path)
    selected = {}
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    count, cv_first, previous = 0, None, None
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot decode {path}")
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            if count >= len(timeline.pts) or bgr.shape != (timeline.height, timeline.width, 3):
                raise ValueError("Decoder shape or frame count differs from ffprobe")
            cv_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            if cv_first is None:
                cv_first = cv_ms
            relative_ms = timeline.relative_ms(count, anchor_ms)
            if not np.isfinite(cv_ms) or abs((cv_ms - cv_first - anchor_ms) - relative_ms) > .5:
                raise ValueError("OpenCV and ffprobe frame timestamps disagree")
            if relative_ms <= 0:
                thumb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA)
                change = float(np.abs(thumb.astype(float) - previous).mean() / 255) if previous is not None else None
                previous = thumb.astype(float)
                frame = DecodedFrame(count, timeline.pts[count], relative_ms, bgr, change)
                selected["t0"] = frame
                if lower_ms is not None and relative_ms < lower_ms:
                    selected["precontact"] = frame
            count += 1
    finally:
        capture.release()
    if count != len(timeline.pts):
        raise ValueError("Incomplete decode: frame count differs from ffprobe")
    if timeline.relative_ms(count - 1, anchor_ms) < 0:
        raise ValueError("Clip ends before the annotation anchor")
    return timeline, selected


def array_digest(array: np.ndarray) -> str:
    prefix = f"{array.dtype}:{array.shape}:".encode()
    return hashlib.sha256(prefix + np.ascontiguousarray(array).tobytes()).hexdigest()


def model_input_digest(path: Path) -> str:
    return array_digest(to_tensor(load_image(str(path), (448, 252), "overlay")).numpy())


def write_png(path: Path, bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise OSError(f"Cannot write {path}")
    if not np.array_equal(cv2.imread(str(path)), bgr):
        raise ValueError("PNG roundtrip changed pixels")


def image_quality(frame: DecodedFrame) -> dict:
    gray = cv2.cvtColor(cv2.resize(frame.bgr, (448, 252), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return {"luminance_mean": float(gray.mean()), "dark_pixel_fraction": float((gray <= 8).mean()),
            "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            "previous_frame_change": frame.previous_change,
            "near_black_review": bool(gray.mean() < 5 and (gray <= 8).mean() > .99)}


def decode_ffmpeg_frame(path: Path, index: int, width: int, height: int) -> np.ndarray:
    raw = subprocess.check_output([
        "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-vf", f"select=eq(n\\,{index})",
        "-frames:v", "1", "-threads", "1", "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1",
    ], stderr=subprocess.PIPE)
    if len(raw) != width * height * 3:
        raise ValueError("FFmpeg did not return exactly one native frame")
    return np.frombuffer(raw, np.uint8).reshape(height, width, 3)


def encode_jpeg_control(png: Path, out: Path) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(png), "-frames:v", "1", "-threads", "1",
                    "-pix_fmt", "yuvj420p", "-q:v", "3", str(out)], check=True, capture_output=True)


def runtime_identity() -> dict:
    return {"opencv": cv2.__version__, "numpy": np.__version__,
            "opencv_build_sha256": hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest(),
            **{tool: subprocess.check_output([tool, "-version"], text=True).splitlines()[0]
               for tool in ("ffmpeg", "ffprobe")}}
