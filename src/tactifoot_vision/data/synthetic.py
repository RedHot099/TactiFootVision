"""Synthetic labelled clip: pitch markings rendered through known homographies."""

import logging
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from tactifoot_vision.data.labelled_clip import LabelledClip, LabelledFrame, Truth
from tactifoot_vision.pitch.pitch import (
    _STD_LENGTH,
    _STD_WIDTH,
    SoccerPitch,
    pitch_markings,
)

logger = logging.getLogger(__name__)

_GRASS = ((45, 135, 50), (38, 118, 42))  # BGR mowing stripes
_LINE = (235, 235, 235)
_STAND = (95, 90, 85)  # beyond the grass and beyond the horizon
_STRIPES = 20  # along the length
_MARGIN_M = 5.0  # grass around the pitch
_SUPERSAMPLE = 3  # rendered at 3x and averaged down, so thin lines stay smooth
_SUBPIXEL = 4  # cv2 drawing shift: 4 fractional bits


def synthetic_pitch_clip(
    path: str | Path,
    homographies: Sequence[np.ndarray],
    frame_size: tuple[int, int] = (1280, 720),
    *,
    pitch: SoccerPitch | None = None,
    fps: float = 25.0,
    line_width: float = 0.12,
    noise: float = 0.0,
    occluders: int = 0,
    seed: int = 0,
) -> LabelledClip:
    """Render ``pitch``'s markings through known homographies into a video and its labelled clip.

    Frame ``i`` shows the pitch's painted lines, circle, arcs and spots on
    striped grass as ``homographies[i]`` maps it, so a static camera repeats
    one matrix and a pan changes it from frame to frame. Beyond the grass
    around the pitch, and beyond the horizon, the frame is a flat stand colour.
    The returned clip carries those homographies as its calibration and
    nothing else, so :func:`~tactifoot_vision.evaluation.evaluate_homography`
    scores any estimator against exact truth.

    Args:
        path: the video file to write (MPEG-4); the clip is named after its stem.
        homographies: one 3x3 frame -> pitch homography per frame, in the
            units of ``pitch``, taking frame coordinates where pixel ``i``
            covers ``[i, i + 1)``. The frame centre must show the pitch: it
            tells which side of the horizon is the pitch, so a centre in the
            sky renders the pitch mirrored into it.
        frame_size: ``(width, height)`` in pixels, both even (the encoder
            drops an odd last row or column).
        pitch: the pitch drawn and its units; the default 105 x 68 m.
        fps: frame rate of the video.
        line_width: width of the painted lines in metres (0.12 on a real
            pitch); widen it for small frames, where real lines are sub-pixel.
        noise: standard deviation of the Gaussian noise added to every
            pixel, in grey levels.
        occluders: number of player-sized boxes of random colour drawn over
            each frame at random places.
        seed: seeds the noise and the occluders; the same arguments give the
            same video.
    """
    path = Path(path)
    pitch = pitch or SoccerPitch()
    homographies = [np.array(h, dtype=float) for h in homographies]
    _check_arguments(homographies, frame_size, fps, line_width, noise, occluders)
    texture, pitch_to_texture = _pitch_texture(pitch, line_width)
    rng = np.random.default_rng(seed)

    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = frame_size
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV cannot write {path}")
    try:
        for homography in homographies:
            image = _render_frame(texture, pitch_to_texture @ homography, frame_size)
            for _ in range(occluders):
                _draw_occluder(image, rng)
            if noise:
                image = np.clip(image + rng.normal(0, noise, image.shape), 0, 255)
            writer.write(np.round(image).astype(np.uint8))
    finally:
        writer.release()

    empty = sv.Detections.empty()
    clip = LabelledClip(
        name=path.stem,
        video_path=path,
        fps=float(fps),
        width=width,
        height=height,
        frames=tuple(
            LabelledFrame(index, empty, empty, homography=h)
            for index, h in enumerate(homographies)
        ),
        truth=frozenset({Truth.CALIBRATION}),
    )
    logger.info("Wrote synthetic clip %s: %d frames", path, len(homographies))
    return clip


def _check_arguments(
    homographies: list[np.ndarray],
    frame_size: tuple[int, int],
    fps: float,
    line_width: float,
    noise: float,
    occluders: int,
) -> None:
    if not homographies:
        raise ValueError("A synthetic clip needs at least one homography")
    for index, h in enumerate(homographies):
        if h.shape != (3, 3) or not np.isfinite(h).all():
            raise ValueError(
                f"Homography {index} must be a finite 3x3 matrix, got shape {h.shape}"
            )
        if np.linalg.matrix_rank(h) < 3:
            raise ValueError(f"Homography {index} is not invertible")
    width, height = frame_size
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError(
            f"frame_size must be positive and even (the encoder drops an odd "
            f"last row or column), got {width} x {height}"
        )
    for index, h in enumerate(homographies):
        # The frame centre tells which side of the horizon is the pitch.
        if abs(h[2] @ [width / 2, height / 2, 1]) < 1e-12 * np.abs(h).max():
            raise ValueError(
                f"Homography {index} puts the frame centre on the horizon; "
                "the frame centre must show the pitch"
            )
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    if line_width <= 0:
        raise ValueError(f"line_width must be > 0, got {line_width}")
    if noise < 0:
        raise ValueError(f"noise must be >= 0, got {noise}")
    if occluders < 0:
        raise ValueError(f"occluders must be >= 0, got {occluders}")


def _pitch_texture(
    pitch: SoccerPitch, line_width: float
) -> tuple[np.ndarray, np.ndarray]:
    """Top-down image of the pitch on grass, and its pitch -> texture pixel homography.

    Texture pixels have their centres at integer coordinates (cv2's convention)
    and are square in metres, so lines keep ``line_width`` on any pitch units.
    """
    # At least 3 pixels per line width so the warp does not lose thin lines,
    # up to 40 px/m: a line narrower than 7.5 cm is drawn about 1-3 px wide.
    px_per_m = min(40.0, max(10.0, 3 / line_width))
    margin = _MARGIN_M * px_per_m
    scale = px_per_m * np.array([_STD_LENGTH / pitch.length, _STD_WIDTH / pitch.width])
    pitch_to_texture = np.array(
        [[scale[0], 0, margin - 0.5], [0, scale[1], margin - 0.5], [0, 0, 1]]
    )
    size = np.round(
        (np.array([_STD_LENGTH, _STD_WIDTH]) + 2 * _MARGIN_M) * px_per_m
    ).astype(int)
    stripe = np.floor(
        (np.arange(size[0]) + 0.5 - margin) / (size[0] - 2 * margin) * _STRIPES
    )
    texture = np.empty((size[1], size[0], 3), dtype=np.uint8)
    texture[:] = np.where((stripe % 2 == 0)[:, None], _GRASS[0], _GRASS[1])[None]

    def to_texture(points: np.ndarray) -> np.ndarray:
        xy = np.asarray(points, dtype=float) * scale + margin - 0.5
        return np.round(xy * (1 << _SUBPIXEL)).astype(np.int32)

    lines, spots = pitch_markings(pitch)
    thickness = max(1, round(line_width * px_per_m))
    cv2.polylines(
        texture, [to_texture(line) for line in lines], False, _LINE, thickness,
        cv2.LINE_AA, _SUBPIXEL,
    )  # fmt: skip
    for spot in to_texture(spots):
        cv2.circle(
            texture,
            tuple(spot),
            thickness << _SUBPIXEL,
            _LINE,
            -1,
            cv2.LINE_AA,
            _SUBPIXEL,
        )
    return texture, pitch_to_texture


def _render_frame(
    texture: np.ndarray, frame_to_texture: np.ndarray, frame_size: tuple[int, int]
) -> np.ndarray:
    """The frame of ``frame_size`` that shows ``texture`` through ``frame_to_texture``, as float BGR."""
    width, height = frame_size
    s = _SUPERSAMPLE
    # Supersampled pixel j (centre at j in cv2) covers frame coordinates
    # [j / s, (j + 1) / s): its centre is at (j + 0.5) / s.
    to_frame = np.array([[1 / s, 0, 0.5 / s], [0, 1 / s, 0.5 / s], [0, 0, 1]])
    mapping = frame_to_texture @ to_frame
    image = cv2.warpPerspective(
        texture, mapping, (width * s, height * s),
        flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=_STAND,
    )  # fmt: skip
    # Pixels beyond the horizon would mirror the pitch into the sky: keep the
    # side whose w has the sign of the frame centre's.
    centre = frame_to_texture[2] @ [width / 2, height / 2, 1]
    a, b, c = mapping[2] * np.sign(centre)
    in_front = a * np.arange(width * s) + b * np.arange(height * s)[:, None] + c > 1e-9
    image[~in_front] = _STAND
    return cv2.resize(image, frame_size, interpolation=cv2.INTER_AREA).astype(float)


def _draw_occluder(image: np.ndarray, rng: np.random.Generator) -> None:
    """A player-sized box of random colour at a random place."""
    height, width = image.shape[:2]
    box_w = rng.uniform(0.02, 0.05) * width
    box_h = box_w * rng.uniform(2, 3)
    x, y = rng.uniform(0, width), rng.uniform(0, height)
    color = tuple(float(c) for c in rng.integers(0, 256, 3))
    corners = np.round(np.array([[x, y], [x + box_w, y + box_h]])).astype(int)
    cv2.rectangle(image, tuple(corners[0]), tuple(corners[1]), color, -1)
