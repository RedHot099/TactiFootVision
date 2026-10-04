"""Reading video frames and turning a video into an image folder."""

import logging
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoReader:
    """Frame access for a video file.

    >>> video = VideoReader("match.mp4")
    >>> for index, frame in video.frames(start=100, end=200, stride=5): ...
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Video not found: {self.path}")
        capture = self._open()
        try:
            self.fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not self.fps:
                logger.warning("%s reports no frame rate; assuming 25 fps", self.path)
                self.fps = 25.0
            self.width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()

    def _open(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV cannot open {self.path}")
        return capture

    @property
    def size(self) -> tuple[int, int]:
        """``(width, height)`` in pixels."""
        return self.width, self.height

    @property
    def duration(self) -> float:
        """Length in seconds."""
        return self.frame_count / self.fps

    def __len__(self) -> int:
        return self.frame_count

    def __iter__(self) -> Iterator[np.ndarray]:
        for _, frame in self.frames():
            yield frame

    def frames(
        self, start: int = 0, end: int | None = None, stride: int = 1
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(frame_index, bgr_frame)`` for ``start <= index < end`` every ``stride`` frames.

        The range is checked when this is called (``ValueError``), not on the
        first frame; ``end == start`` is an empty range.
        """
        check_frame_range(start, end, stride)
        return self._frames(start, end, stride)

    def _frames(
        self, start: int, end: int | None, stride: int
    ) -> Iterator[tuple[int, np.ndarray]]:
        capture = self._open()
        try:
            if start > 0:
                capture.set(cv2.CAP_PROP_POS_FRAMES, start)
            index = start
            while end is None or index < end:
                if (index - start) % stride == 0:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    yield index, frame
                elif not capture.grab():  # skip without decoding
                    break
                index += 1
        finally:
            capture.release()

    def read(self, index: int) -> np.ndarray:
        """Return a single frame by index."""
        if index >= 0:
            for _, frame in self.frames(start=index, end=index + 1):
                return frame
        raise IndexError(
            f"Frame {index} is out of range for {self.path} ({self.frame_count} frames)"
        )

    def __repr__(self) -> str:
        return (
            f"VideoReader({str(self.path)!r}, {self.width}x{self.height}, "
            f"{self.fps:.2f} fps, {self.frame_count} frames)"
        )


def check_frame_range(start: int, end: int | None, stride: int) -> None:
    """Raise ``ValueError`` unless ``0 <= start <= end`` (``end=None``: to the end) and ``stride >= 1``."""
    if start < 0:
        raise ValueError(f"start must be >= 0, got {start}")
    if end is not None and end < start:
        raise ValueError(f"end ({end}) must be >= start ({start})")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")


def extract_frames(
    video: str | Path,
    out_dir: str | Path,
    start: int = 0,
    end: int | None = None,
    stride: int = 25,
    image_format: str = "jpg",
) -> list[Path]:
    """Save the frames ``start <= index < end`` every ``stride`` frames; return the paths.

    Files are named ``<video stem>_<index>.<format>``; ``end=None`` runs to the
    end of the video. Handy for building a new dataset to annotate from match
    footage.
    """
    reader = VideoReader(video)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, frame in reader.frames(start=start, end=end, stride=stride):
        path = out / f"{reader.path.stem}_{index:06d}.{image_format}"
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"Could not write {path}")
        paths.append(path)
    logger.info("Extracted %d frames from %s to %s", len(paths), reader.path, out)
    return paths
