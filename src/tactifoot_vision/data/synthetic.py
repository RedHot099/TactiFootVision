"""Synthetic labelled clip: pitch markings rendered through known homographies, and broadcast camera views to render."""

import logging
from collections.abc import Iterator, Sequence
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from tactifoot_vision.data.labelled_clip import LabelledClip, LabelledFrame, Truth
from tactifoot_vision.data.video import FRAME_NUMBER, sequence_images
from tactifoot_vision.pitch.pitch import (
    _STD_LENGTH,
    _STD_WIDTH,
    SoccerPitch,
    pitch_markings,
)
from tactifoot_vision.pitch.projection import project_points

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
        path: the video file to write (MPEG-4), named after its stem, or an
            OpenCV image-sequence pattern such as ``views/%06d.png``, named
            after its folder: image ``n`` is frame ``n``, so
            :meth:`~tactifoot_vision.data.Dataset.from_clips` can reference
            the frames in place. A folder that already holds images of the
            pattern numbered beyond the new frames is refused; images of
            the new frames' numbers are overwritten.
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
    sequence = _is_sequence(path)
    if sequence:
        _check_no_later_images(path, len(homographies))
    texture, pitch_to_texture = _pitch_texture(pitch, line_width)
    rng = np.random.default_rng(seed)

    def images() -> Iterator[np.ndarray]:
        for homography in homographies:
            image = _render_frame(texture, pitch_to_texture @ homography, frame_size)
            for _ in range(occluders):
                _draw_occluder(image, rng)
            if noise:
                image = np.clip(image + rng.normal(0, noise, image.shape), 0, 255)
            yield np.round(image).astype(np.uint8)

    path.parent.mkdir(parents=True, exist_ok=True)
    if sequence:
        _write_images(path, images())
    else:
        _write_video(path, images(), fps, frame_size)

    width, height = frame_size
    empty = sv.Detections.empty()
    clip = LabelledClip(
        name=path.parent.name if sequence else path.stem,
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


def broadcast_homographies(
    count: int, frame_size: tuple[int, int] = (1280, 720), *, seed: int = 0
) -> list[np.ndarray]:
    """Frame -> pitch homographies of ``count`` random views of a broadcast camera.

    Each view is a main broadcast camera's, drawn independently: it stands
    10 to 35 m behind the touchline ``y = 68`` and 15 to 30 m above the
    ground, within 10 m of the halfway line, pans and tilts to aim the frame
    centre at a point of the pitch, zooms between a horizontal field of view
    of 15 and 60 degrees, and rolls by at most 2 degrees. That touchline is
    nearest the bottom of the frame and ``x`` grows to the right, as in the
    pitch keypoint dataset. The frame centre always shows the pitch, as
    :func:`synthetic_pitch_clip` needs, and so does the middle of the frame a
    quarter of the way down: like an operator's, a view towards the far side
    is zoomed in rather than showing mostly stands, and the centre stays
    short of the far touchline. The same arguments give the same views.

    Args:
        count: how many views.
        frame_size: ``(width, height)`` of the frames, in pixels.
        seed: seeds the views.

    Returns:
        ``count`` 3x3 frame -> pitch homographies onto the default
        ``SoccerPitch`` (105 x 68 m), frame pixel ``i`` covering ``[i, i + 1)``.
    """
    width, height = frame_size
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    if width <= 0 or height <= 0:
        raise ValueError(f"frame_size must be positive, got {width} x {height}")
    rng = np.random.default_rng(seed)
    homographies: list[np.ndarray] = []
    while len(homographies) < count:
        camera = np.array(
            [
                _STD_LENGTH / 2 + rng.uniform(-10, 10),
                _STD_WIDTH + rng.uniform(10, 35),
                -rng.uniform(15, 30),  # z points down, into the ground
            ]
        )
        target = np.array(
            [rng.uniform(0, _STD_LENGTH), rng.uniform(0, _STD_WIDTH), 0.0]
        )
        field_of_view = np.radians(rng.uniform(15, 60))
        roll = np.radians(rng.uniform(-2, 2))
        homography = _view(camera, target, field_of_view, roll, frame_size)
        # An operator zooms in on the far side rather than film the stands.
        upper, in_front = project_points(
            [[width / 2, height / 4]], homography, (width / 2, height / 2)
        )
        on_pitch = (upper[0] >= 0) & (upper[0] <= [_STD_LENGTH, _STD_WIDTH])
        if in_front[0] and on_pitch.all():
            homographies.append(homography)
    return homographies


def _view(
    camera: np.ndarray,
    target: np.ndarray,
    field_of_view: float,
    roll: float,
    frame_size: tuple[int, int],
) -> np.ndarray:
    """Frame -> pitch homography of a pinhole camera at ``camera`` aimed at ``target``.

    Pitch space is ``(x, y, z)`` with ``z`` down, so it is right-handed and
    the camera's image axes (right, down, forward) need no mirror.
    """
    forward = (target - camera) / np.linalg.norm(target - camera)
    right = np.cross([0.0, 0.0, 1.0], forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    right, down = (
        np.cos(roll) * right + np.sin(roll) * down,
        np.cos(roll) * down - np.sin(roll) * right,
    )
    rotation = np.stack([right, down, forward])
    width, height = frame_size
    focal = width / 2 / np.tan(field_of_view / 2)
    intrinsics = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    # Ground points (x, y, 0, 1) -> frame: the x, y and translation columns.
    pitch_to_frame = (
        intrinsics @ rotation @ np.column_stack([[1.0, 0, 0], [0, 1.0, 0], -camera])
    )
    frame_to_pitch = np.linalg.inv(pitch_to_frame)
    # Scaled so the frame centre, which shows target, has w = 1.
    return frame_to_pitch / (frame_to_pitch[2] @ [width / 2, height / 2, 1])


def _is_sequence(path: Path) -> bool:
    """Whether ``path`` is an image-sequence pattern rather than a video file."""
    try:
        sequence_images(path)
    except ValueError:
        return False
    return True


def _check_no_later_images(pattern: Path, count: int) -> None:
    """Refuse a folder holding images of ``pattern`` the new frames would not overwrite."""
    stale = [path for number, path in sequence_images(pattern) if number >= count]
    if stale:
        raise ValueError(
            f"{stale[0]} is beyond the {count} frames to write to {pattern}; "
            "remove it, or write to another folder, so the sequence holds only "
            "the new frames"
        )


def _write_images(pattern: Path, images: Iterator[np.ndarray]) -> None:
    """Image ``n`` of ``images`` to ``pattern`` numbered ``n``."""
    prefix, suffix = FRAME_NUMBER.split(pattern.name)  # one number: _is_sequence
    number = pattern.name[len(prefix) : len(pattern.name) - len(suffix)]
    for index, image in enumerate(images):
        name = prefix + number % index + suffix
        if not cv2.imwrite(str(pattern.parent / name), image):
            raise RuntimeError(f"OpenCV cannot write {pattern.parent / name}")


def _write_video(
    path: Path, images: Iterator[np.ndarray], fps: float, frame_size: tuple[int, int]
) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV cannot write {path}")
    try:
        for image in images:
            writer.write(image)
    finally:
        writer.release()


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
