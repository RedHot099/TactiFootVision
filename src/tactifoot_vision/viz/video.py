"""Writing an annotated video from a pipeline result."""

import inspect
import logging
from math import gcd
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from tactifoot_vision.data.video import VideoReader
from tactifoot_vision.pipeline.result import PipelineResult
from tactifoot_vision.viz.annotate import FrameAnnotator
from tactifoot_vision.viz.radar import PitchRadar, check_overlay, overlay

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
    overlay_padding: int = 10,
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
            ``result.video_path``. Its frame size must match the result and it
            must hold every frame of the result (``RuntimeError`` otherwise).
        output: ``.mp4`` path to write (parent folders are created).
        annotator: frame drawing; default ``FrameAnnotator(pitch=result.pitch)``.
        radar: pitch overlay; default ``PitchRadar(pitch=result.pitch)``,
            ``False`` for none.
        overlay_position, overlay_width_fraction, overlay_alpha, overlay_padding:
            see :func:`overlay`.
        fps: output frame rate; default derived from the source and the stride.
        progress: show a progress bar.

    Returns:
        The written path. Settings are checked before ``output`` is opened, an
        ``output`` that is the source file (same path, symlink or hard link)
        is refused, and a failed render deletes the partial ``output``.
    """
    if not result.frames:
        raise ValueError("The result has no frames to render")
    if source is None:
        if result.video_path is None:
            raise ValueError("The result has no video_path; pass the source video")
        source = result.video_path
    output = Path(output)
    if _same_file(output, Path(source)):
        raise ValueError(f"output {output} is the source video; pick another path")
    check_render_options(
        overlay_position=overlay_position,
        overlay_width_fraction=overlay_width_fraction,
        overlay_alpha=overlay_alpha,
        overlay_padding=overlay_padding,
        fps=fps,
    )
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
    written: set[int] = set()
    complete = False
    try:
        for index, frame in reader.frames(
            start=indices[0], end=indices[-1] + 1, stride=step
        ):
            if not written and frame.shape[:2] != reader.size[::-1]:
                raise ValueError(
                    f"{reader.path} decodes to {frame.shape[1]}x{frame.shape[0]} frames "
                    f"but its header says {reader.size[0]}x{reader.size[1]} (rotation "
                    "metadata?); re-encode the video without rotation and rerun the pipeline"
                )
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
                    padding=overlay_padding,
                )
            writer.write(image)
            written.add(index)
            if bar is not None:
                bar.update()
        if not written:
            raise RuntimeError(
                f"{reader.path} has no frame of the result (first index {indices[0]})"
            )
        if len(written) < len(indices):
            missing = next(i for i in indices if i not in written)
            raise RuntimeError(
                f"{reader.path} ended early: frame {missing} is missing "
                f"({len(written)} of {len(indices)} result frames were rendered)"
            )
        complete = True
    finally:
        writer.release()
        if bar is not None:
            bar.close()
        if not complete:
            output.unlink(missing_ok=True)  # never leave a partial video behind
    logger.info("Wrote %d frames at %.2f fps to %s", len(written), fps, output)
    return output


def check_render_options(**options: Any) -> None:
    """Raise ``ValueError`` for a :func:`render_video` setting it would reject.

    Takes any :func:`render_video` keyword arguments (a name it does not take
    is a ``ValueError``) and checks the values it knows: the overlay settings
    and ``fps``. The others pass unchecked, so a new ``render_video``
    parameter needs no change here. A run file uses this to fail before the
    first frame instead of after the whole run.
    """
    valid = inspect.signature(render_video).parameters
    unknown = [name for name in options if name not in valid]
    if unknown:
        raise ValueError(
            f"render_video has no argument {unknown[0]!r}; valid: {', '.join(valid)}"
        )
    check_overlay(
        options.get("overlay_position"),
        options.get("overlay_width_fraction"),
        options.get("overlay_alpha"),
        options.get("overlay_padding"),
    )
    fps = options.get("fps")
    if fps is not None and fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")


def _same_file(output: Path, source: Path) -> bool:
    """Whether writing ``output`` would overwrite ``source``.

    Compares file identity, so a symlink or a hard link to the source counts,
    not only the same path.
    """
    return output.exists() and source.exists() and output.samefile(source)


def _stride(indices: list[int]) -> int:
    """Common step between processed frame indices (1 for a single frame)."""
    steps = np.diff(indices)
    return max(1, gcd(*(int(s) for s in steps))) if len(steps) else 1
