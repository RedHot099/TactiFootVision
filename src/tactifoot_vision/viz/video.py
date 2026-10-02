"""Writing an annotated video from a pipeline result."""

import logging
from math import gcd
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from tactifoot_vision.data.video import VideoReader
from tactifoot_vision.pipeline.result import PipelineResult
from tactifoot_vision.viz.annotate import FrameAnnotator
from tactifoot_vision.viz.radar import PitchRadar, overlay

logger = logging.getLogger(__name__)


def render_video(
    result: PipelineResult,
    source: str | Path | None,
    output: str | Path,
    annotator: FrameAnnotator | None = None,
    radar: PitchRadar | Literal[False] | None = None,
    overlay_position: str = "bottom-center",
    overlay_width_fraction: float = 0.25,
    overlay_alpha: float = 0.8,
    fps: float | None = None,
    progress: bool = True,
) -> Path:
    """Re-read ``source`` and write the frames in ``result`` annotated to an mp4.

    Only processed frames are written (matched by :attr:`FrameResult.index`),
    so a result computed with ``stride=k`` plays at ``source fps / k`` and
    keeps real-time speed (for evenly spaced frames; a warning says otherwise).

    Args:
        result: pipeline output for ``source``.
        source: the video the result was computed on; ``None`` uses
            ``result.video_path``. Its frame size must match the result.
        output: ``.mp4`` path to write (parent folders are created).
        annotator: frame drawing; default ``FrameAnnotator(pitch=result.pitch)``.
        radar: pitch overlay; default ``PitchRadar(pitch=result.pitch)``,
            ``False`` for none.
        overlay_position, overlay_width_fraction, overlay_alpha: see :func:`overlay`.
        fps: output frame rate; default derived from the source and the stride.
        progress: show a progress bar.

    Returns:
        The written path.
    """
    if not result.frames:
        raise ValueError("The result has no frames to render")
    if source is None:
        if result.video_path is None:
            raise ValueError("The result has no video_path; pass the source video")
        source = result.video_path
    reader = VideoReader(source)
    if reader.size != tuple(result.frame_size):
        raise ValueError(
            f"{reader.path} is {reader.size[0]}x{reader.size[1]} but the result was "
            f"computed on {result.frame_size[0]}x{result.frame_size[1]} frames"
        )
    if abs(reader.fps - result.fps) > 1e-3:
        logger.warning(
            "%s runs at %.2f fps, the result at %.2f fps: is it the same video?",
            reader.path,
            reader.fps,
            result.fps,
        )
    by_index = {f.index: f for f in result.frames}
    indices = sorted(by_index)
    step = _stride(indices)
    if len(set(np.diff(indices))) > 1:
        logger.warning(
            "Frames are unevenly spaced; the video will not play in real time"
        )
    if fps is None:
        fps = reader.fps / step
    annotator = annotator or FrameAnnotator(pitch=result.pitch)
    if radar is None:
        radar = PitchRadar(pitch=result.pitch)
    for part in (annotator, radar):
        if part and part.pitch != result.pitch:
            raise ValueError(
                f"{type(part).__name__} draws {part.pitch} but the result uses "
                f"{result.pitch}; pass pitch=result.pitch"
            )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, reader.size
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV cannot write {output}")

    bar = None
    if progress:
        from tqdm.auto import tqdm

        bar = tqdm(total=len(indices), desc="Rendering", unit="frame")
    written = 0
    try:
        for index, frame in reader.frames(
            start=indices[0], end=indices[-1] + 1, stride=step
        ):
            frame_result = by_index.get(index)
            if frame_result is None:
                continue
            image = annotator.annotate(frame, frame_result)
            if radar:
                image = overlay(
                    image,
                    radar.draw(frame_result),
                    position=overlay_position,
                    width_fraction=overlay_width_fraction,
                    alpha=overlay_alpha,
                )
            writer.write(image)
            written += 1
            if bar is not None:
                bar.update()
    finally:
        writer.release()
        if bar is not None:
            bar.close()
    if written < len(indices):
        logger.warning(
            "%s ended early: wrote %d of %d result frames",
            reader.path,
            written,
            len(indices),
        )
    logger.info("Wrote %d frames at %.2f fps to %s", written, fps, output)
    return output


def _stride(indices: list[int]) -> int:
    """Common step between processed frame indices (1 for a single frame)."""
    steps = np.diff(indices)
    return max(1, gcd(*(int(s) for s in steps))) if len(steps) else 1
