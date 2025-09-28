from pathlib import Path
from typing import Iterator

import numpy as np
import supervision as sv


class VideoLoader:
    def __init__(self, video_path: Path):
        self.video_path = Path(video_path)
        if not self.video_path.is_file():
            raise FileNotFoundError(f"Video not found: {self.video_path}")

    def get_info(self) -> sv.VideoInfo:
        return sv.VideoInfo.from_video_path(str(self.video_path))

    def frame_generator(self, stride: int = 1) -> Iterator[np.ndarray]:
        gen = sv.get_video_frames_generator(str(self.video_path), stride=stride)
        for frame in gen:
            yield frame

