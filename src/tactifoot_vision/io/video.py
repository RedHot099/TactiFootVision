from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from tactifoot_vision.domain import Frame


class ImageSequenceReader:
    def __init__(self, directory: str | Path, *, fps: float = 25.0) -> None:
        self.directory = Path(directory)
        self.fps = fps
        if not self.directory.is_dir():
            raise FileNotFoundError(
                f"Image sequence directory not found: {self.directory}"
            )

    def __iter__(self) -> Iterator[Frame]:
        paths = sorted(
            path
            for path in self.directory.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
        for index, path in enumerate(paths):
            image = cv2.imread(str(path))
            if image is None:
                continue
            frame_image = np.asarray(image, dtype=np.uint8)
            yield Frame(
                index=index,
                image=frame_image,
                timestamp_seconds=index / self.fps,
                path=path,
            )


class VideoReader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Video file not found: {self.path}")

    def __iter__(self) -> Iterator[Frame]:
        capture = cv2.VideoCapture(str(self.path))
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        index = 0
        try:
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                frame_image = np.asarray(image, dtype=np.uint8)
                yield Frame(
                    index=index,
                    image=frame_image,
                    timestamp_seconds=index / fps,
                    path=self.path,
                )
                index += 1
        finally:
            capture.release()


def read_frames(path: str | Path) -> Iterator[Frame]:
    source = Path(path)
    if source.is_dir():
        return iter(ImageSequenceReader(source))
    return iter(VideoReader(source))
