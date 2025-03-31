# tactifoot_vision/data/video_loader.py
import logging
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import supervision as sv

logger = logging.getLogger(__name__)


class VideoLoader:
    """Handles loading video frames and information."""

    def __init__(self, source_path: Path):
        """
        Initializes the VideoLoader.

        Args:
            source_path: Path to the video file.

        Raises:
            FileNotFoundError: If the source video file does not exist.
        """
        if not source_path.exists():
            logger.error(f"Video source file not found: {source_path}")
            raise FileNotFoundError(f"Video source file not found: {source_path}")
        self.source_path = source_path
        try:
            self.video_info = sv.VideoInfo.from_video_path(str(source_path))
            logger.info(
                f"Loaded video info for: {source_path.name} "
                f"({self.video_info.width}x{self.video_info.height}, "
                f"{self.video_info.fps:.2f} FPS, "
                f"{self.video_info.total_frames} frames)"
            )
        except Exception as e:
            logger.exception(f"Failed to get video info from {source_path}")
            raise RuntimeError(f"Could not load video info: {e}") from e

    def frame_generator(
        self, stride: int = 1, start: int = 0, end: Optional[int] = None
    ) -> Generator[np.ndarray, None, None]:
        """
        Creates a generator that yields video frames.

        Args:
            stride: Process every nth frame. Defaults to 1 (all frames).
            start: Starting frame index. Defaults to 0.
            end: Ending frame index (exclusive). Defaults to None (end of video).

        Yields:
            np.ndarray: The next video frame in BGR format.
        """
        try:
            frame_gen = sv.get_video_frames_generator(
                source_path=str(self.source_path),
                stride=stride,
                start=start,
                end=end,
            )
            logger.info(
                f"Starting frame generation from {self.source_path.name} "
                f"(stride={stride}, start={start}, end={end or 'EOF'})"
            )
            yield from frame_gen
            logger.info(f"Finished frame generation for {self.source_path.name}")
        except Exception as e:
            logger.exception(f"Error during frame generation from {self.source_path}")
            # Depending on desired behavior, you might want to re-raise,
            # log and continue, or handle differently.
            raise RuntimeError(f"Frame generation failed: {e}") from e

    def get_info(self) -> sv.VideoInfo:
        """Returns the video information."""
        return self.video_info
