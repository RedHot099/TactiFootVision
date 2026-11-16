"""In-memory index of an annotated image dataset (detection or pose)."""

import logging
import numbers
import random
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from tactifoot_vision.data._files import clipped_box
from tactifoot_vision.data.annotations import VISIBLE, Annotations, Task
from tactifoot_vision.data.labelled_clip import LabelledClip, Truth
from tactifoot_vision.data.video import sequence_images
from tactifoot_vision.enums import DetectionKey, LinkMode, ObjectKind, Role, Split
from tactifoot_vision.pitch.homography import frame_to_pitch
from tactifoot_vision.pitch.pitch import SoccerPitch
from tactifoot_vision.pitch.projection import project_points

logger = logging.getLogger(__name__)

# The classes of a dataset built from labelled clips: the package vocabulary.
_CLIP_CLASSES = tuple(
    kind.value for kind in (Role.PLAYER, Role.GOALKEEPER, Role.REFEREE, ObjectKind.BALL)
)
# The one class of a pitch keypoint dataset from labelled clips.
_PITCH_CLASS = "pitch"
# OpenCV's image-sequence reader only finds a first image numbered 0 to 4.
_LAST_FIRST_IMAGE = 4


def check_flip_idx(
    flip_idx: tuple[int, ...] | None, num_keypoints: int | None, source: str = ""
) -> None:
    """Refuse a ``flip_idx`` that does not have one entry per keypoint.

    Checked when a dataset is loaded, so a bad ``data.yaml`` fails there and
    not in :class:`~tactifoot_vision.augment.HorizontalFlip` after an
    augmentation run has written some images.
    """
    if flip_idx is None or num_keypoints is None or len(flip_idx) == num_keypoints:
        return
    where = f"{source}: " if source else ""
    raise ValueError(
        f"{where}flip_idx has {len(flip_idx)} entries but there are "
        f"{num_keypoints} keypoints; it needs one entry per keypoint"
    )


@dataclass(frozen=True)
class Sample:
    """One image on disk with its labels (pixel coordinates)."""

    image_path: Path
    width: int
    height: int
    annotations: Annotations

    def read_image(self) -> np.ndarray:
        """Load the image as a BGR ``uint8`` array (OpenCV convention)."""
        image = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image {self.image_path}")
        return image


@dataclass
class Dataset:
    """Annotated images grouped into ``train`` / ``valid`` / ``test`` splits.

    A ``Dataset`` is a lightweight index: images stay on disk, labels live in
    memory. All operations return new datasets, and ``to_yolo`` / ``to_coco``
    write whatever format a training backend needs.

    Load one with :func:`load_dataset` (YOLO ``data.yaml`` or COCO folder).
    """

    task: Task
    class_names: list[str]
    splits: dict[str, list[Sample]] = field(default_factory=dict)
    num_keypoints: int | None = None
    flip_idx: tuple[int, ...] | None = None
    name: str = "dataset"

    def __post_init__(self) -> None:
        self.task = Task(self.task)
        self.splits = {Split.parse(k): list(v) for k, v in self.splits.items()}
        if self.task is Task.POSE and self.num_keypoints is None:
            raise ValueError("Pose datasets need num_keypoints")
        check_flip_idx(self.flip_idx, self.num_keypoints)

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: str | Path) -> "Dataset":
        """Load a YOLO (``data.yaml`` or its folder) or COCO (split folders) dataset."""
        path = Path(path)
        if path.is_file() and path.suffix in {".yaml", ".yml"}:
            return cls.from_yolo(path)
        if path.is_dir():
            if (path / "data.yaml").is_file():
                return cls.from_yolo(path / "data.yaml")
            if any(
                (path / s / "_annotations.coco.json").is_file()
                for s in ("train", "valid", "val", "test")
            ):
                return cls.from_coco(path)
        raise FileNotFoundError(
            f"{path} is neither a YOLO data.yaml (or its folder) nor a COCO dataset folder"
        )

    @classmethod
    def from_yolo(cls, data_yaml: str | Path) -> "Dataset":
        from tactifoot_vision.data import yolo

        return yolo.read_dataset(Path(data_yaml))

    @classmethod
    def from_coco(
        cls, root: str | Path, flip_idx: tuple[int, ...] | None = None
    ) -> "Dataset":
        from tactifoot_vision.data import coco

        return coco.read_dataset(Path(root), flip_idx=flip_idx)

    @classmethod
    def from_clips(
        cls,
        clips: LabelledClip | Iterable[LabelledClip],
        train: float = 0.8,
        valid: float = 0.2,
        test: float = 0.0,
        seed: int = 0,
        task: Task | str = Task.DETECT,
    ) -> "Dataset":
        """Detection or pitch keypoint dataset of the frames of labelled clips, split per clip.

        The fractions count clips, not frames, and are rounded as in
        :meth:`resplit`: every frame of a clip lands in the clip's split, so
        adjacent frames never sit on both sides of a split. Each labelled
        frame becomes a sample whose image is that frame's file of the
        clip's image sequence, referenced in place.

        ``task="detect"`` labels the clips' box truth. A frame without a box
        is left out, since a clip cannot tell an empty frame from an
        unlabelled one.

        ``task="pose"`` labels the clips' calibration truth: each calibrated
        frame becomes one ``pitch`` object whose 32 keypoints are the
        :class:`~tactifoot_vision.pitch.SoccerPitch` vertices mapped into
        the frame through the inverse of its frame -> pitch homography
        (which must use the default pitch's units, as every reader gives
        it). A vertex outside the frame, or beyond the horizon (on the far
        side from the frame centre), is not labelled (``0 0 0``). The box
        holds the part of the frame that shows the pitch; ``flip_idx`` is
        the pitch's. A frame without calibration is left out, and so is one
        that shows no pitch. A clip without calibration truth is refused.

        For both tasks the images must be one per frame, numbered without a gap, the
        lowest-numbered being frame 0 as :class:`VideoReader` reads them; its
        number must be 0 to 4, the first numbers OpenCV looks for. The
        SoccerNet readers number from 1 (image ``n`` is frame ``n - 1``), so
        a SoccerNet folder that also holds an image 0 is refused rather than
        read one frame off; :func:`synthetic_pitch_clip` writes such a
        sequence when given a pattern. Detection boxes are clipped to the
        image. A clip without a labelled frame still takes a clip's share of its
        split, with a warning. Detection classes are ``player``,
        ``goalkeeper``, ``referee`` and ``ball``. A clip without the task's
        truth, one with another detection class, one read from a video file
        (its frames have no file of their own; extract them first), a
        missing or extra image and the same clip passed twice are
        ``ValueError``, as is no clip at all.
        """
        task = Task.parse(task, "task")
        clips = [clips] if isinstance(clips, LabelledClip) else list(clips)
        if not clips:
            raise ValueError("A dataset from clips needs at least one clip")
        seen: set[Path] = set()
        for clip in clips:
            # Resolved, so one sequence reached by two paths is caught too.
            images = clip.video_path.parent.resolve() / clip.video_path.name
            if images in seen:
                raise ValueError(
                    f"Clip {clip.name} ({clip.video_path}) is given twice; its "
                    "frames could land in two splits"
                )
            seen.add(images)
        pitch = SoccerPitch()
        make_samples = (
            _clip_samples
            if task is Task.DETECT
            else partial(_pitch_samples, pitch=pitch)
        )
        pairs = [(clip, make_samples(clip)) for clip in clips]
        for clip, samples in pairs:
            if not samples:
                logger.warning(
                    "Clip %s has no frame with %s; it makes no sample but "
                    "still counts as a clip in the split fractions",
                    clip.name,
                    "a box" if task is Task.DETECT else "the pitch in view",
                )
        splits = {
            split: [sample for _, samples in members for sample in samples]
            for split, members in _split_by_fractions(
                pairs, train, valid, test, seed
            ).items()
        }
        if task is Task.DETECT:
            return cls(task=task, class_names=list(_CLIP_CLASSES), splits=splits)
        return cls(
            task=task,
            class_names=[_PITCH_CLASS],
            splits=splits,
            num_keypoints=pitch.num_keypoints,
            flip_idx=pitch.flip_idx,
        )

    # ------------------------------------------------------------------- access
    def __getitem__(self, split: Split | str) -> list[Sample]:
        """The samples of ``split`` (a new list; empty if the split is missing)."""
        return list(self.splits.get(Split.parse(split), []))

    def __len__(self) -> int:
        return sum(len(samples) for samples in self.splits.values())

    def __iter__(self) -> Iterator[Sample]:
        for split in self.split_names:
            yield from self.splits[split]

    @property
    def split_names(self) -> list[str]:
        return [s.value for s in Split if self.splits.get(s)]

    def read(self, split: Split | str, index: int) -> tuple[np.ndarray, Annotations]:
        """Return ``(image_bgr, annotations)`` of one sample."""
        sample = self.splits.get(Split.parse(split), [])[index]
        return sample.read_image(), sample.annotations.copy()

    def summary(self) -> pd.DataFrame:
        """Images, objects and per-class object counts for every split."""
        rows = []
        for split in self.split_names:
            samples = self.splits[split]
            class_ids = np.concatenate(
                [s.annotations.class_ids for s in samples] or [np.zeros(0, np.int64)]
            )
            counts = np.bincount(class_ids, minlength=len(self.class_names))
            rows.append(
                {"split": split, "images": len(samples), "objects": int(class_ids.size)}
                | {name: int(counts[i]) for i, name in enumerate(self.class_names)}
            )
        return pd.DataFrame(rows).set_index("split") if rows else pd.DataFrame()

    def __repr__(self) -> str:
        sizes = ", ".join(f"{s}={len(self.splits[s])}" for s in self.split_names)
        kp = f", keypoints={self.num_keypoints}" if self.num_keypoints else ""
        return (
            f"Dataset(name={self.name!r}, task={self.task.value}, "
            f"classes={self.class_names}{kp}, {sizes or 'empty'})"
        )

    # ------------------------------------------------------ derived datasets
    def with_split(self, split: Split | str, samples: list[Sample]) -> "Dataset":
        """Return a copy where ``split`` holds ``samples``."""
        return replace(self, splits={**self.splits, Split.parse(split): list(samples)})

    def subset(
        self, size: int | float | Mapping[Split | str, int | float], seed: int = 0
    ) -> "Dataset":
        """Randomly keep ``size`` images per split.

        ``size`` is an image count (an integer, NumPy integers included), a
        fraction (a float in ``[0, 1]``) or a mapping
        ``{split: count_or_fraction}``; splits missing from the mapping are
        kept whole.
        """
        rng = random.Random(seed)
        sizes = (
            {Split.parse(k): v for k, v in size.items()}
            if isinstance(size, Mapping)
            else {}
        )
        splits = {}
        for split, samples in self.splits.items():
            want = sizes.get(split) if isinstance(size, Mapping) else size
            if want is None:
                splits[split] = list(samples)
                continue
            count = _subset_count(want, len(samples))
            splits[split] = rng.sample(samples, min(count, len(samples)))
        return replace(self, splits=splits)

    def resplit(
        self, train: float = 0.8, valid: float = 0.2, test: float = 0.0, seed: int = 0
    ) -> "Dataset":
        """Pool every image and split again with the given fractions.

        Split boundaries are rounded cumulatively and the rounding remainder
        goes to the last split with a non-zero fraction, so a split asked to
        be empty (``test=0``) stays empty.
        """
        splits = _split_by_fractions(list(self), train, valid, test, seed)
        return replace(self, splits=splits)

    def merge(self, other: "Dataset") -> "Dataset":
        """Concatenate the splits of two datasets with identical classes.

        Pose datasets must also share their keypoint layout: the keypoint
        count and ``flip_idx``. An export writes one ``flip_idx`` for every
        sample, so a different mapping would change what the other dataset's
        keypoints mean once reloaded.
        """
        if other.task != self.task or other.class_names != self.class_names:
            raise ValueError(
                "Can only merge datasets with the same task and class names"
            )
        if other.num_keypoints != self.num_keypoints:
            raise ValueError(
                "Can only merge datasets with the same keypoint layout, got "
                f"{self.num_keypoints} and {other.num_keypoints} keypoints"
            )
        if other.flip_idx != self.flip_idx:
            raise ValueError(
                "Can only merge datasets with the same keypoint layout, got flip_idx "
                f"{self.flip_idx} and {other.flip_idx}"
            )
        splits = {s: self[s] + other[s] for s in Split if self[s] or other[s]}
        return replace(self, splits=splits)

    # ----------------------------------------------------------------- export
    def to_yolo(
        self, out_dir: str | Path, link: LinkMode | str = LinkMode.SYMLINK
    ) -> Path:
        """Write the dataset in Ultralytics YOLO layout and return its ``data.yaml``."""
        from tactifoot_vision.data import yolo

        return yolo.write_dataset(self, Path(out_dir), link=link)

    def to_coco(
        self, out_dir: str | Path, link: LinkMode | str = LinkMode.SYMLINK
    ) -> Path:
        """Write the dataset in Roboflow/RF-DETR COCO layout and return its root folder.

        Layout: ``<out_dir>/<split>/_annotations.coco.json`` next to the images.
        """
        from tactifoot_vision.data import coco

        return coco.write_dataset(self, Path(out_dir), link=link)


def _clip_samples(clip: LabelledClip) -> list[Sample]:
    """One sample per frame of ``clip`` with a box, its image the frame's file in place.

    Boxes are clipped to the image, as both exporters write them, and a box
    with nothing left inside it is dropped.
    """
    if Truth.BOXES not in clip.truth:
        raise ValueError(
            f"Clip {clip.name} has no {Truth.BOXES.value} truth, so it makes no "
            "detection samples"
        )
    samples = []
    for frame, image in zip(clip.frames, _frame_images(clip), strict=True):
        objects = [d for d in (frame.detections, frame.ball) if len(d)]
        names = [
            str(n)
            for d in objects
            for n in d.data.get(DetectionKey.CLASS_NAME, [""] * len(d))
        ]
        unknown = sorted(set(names) - set(_CLIP_CLASSES))
        if unknown:
            raise ValueError(
                f"Frame {frame.index} of clip {clip.name} has class {unknown[0]!r}; "
                f"a detection dataset from clips has classes {_CLIP_CLASSES}"
            )
        boxes, class_ids = [], []
        for box, name in zip(
            (box for d in objects for box in d.xyxy), names, strict=True
        ):
            box = clipped_box(box, clip.width, clip.height)
            if box is not None:
                boxes.append(box)
                class_ids.append(_CLIP_CLASSES.index(name))
        if boxes:
            annotations = Annotations(np.array(boxes), class_ids)
            samples.append(Sample(image, clip.width, clip.height, annotations))
    return samples


def _pitch_samples(clip: LabelledClip, pitch: SoccerPitch) -> list[Sample]:
    """One ``pitch`` sample per calibrated frame of ``clip`` that shows the pitch."""
    if Truth.CALIBRATION not in clip.truth:
        raise ValueError(
            f"Clip {clip.name} has no {Truth.CALIBRATION.value} truth, so it makes "
            "no pitch keypoint samples"
        )
    frame_size = (clip.width, clip.height)
    centre = (clip.width / 2, clip.height / 2)
    samples = []
    for frame, image in zip(clip.frames, _frame_images(clip), strict=True):
        if frame.homography is None:
            continue
        box = _pitch_box(frame.homography, frame_size, pitch)
        if box is None:
            continue
        # The frame centre's pitch point lies on the visible side of the horizon.
        front = frame_to_pitch(np.array([centre]), frame.homography)[0]
        xy, in_front = project_points(
            pitch.vertices, np.linalg.inv(frame.homography), tuple(front)
        )
        shown = in_front & ((xy >= 0) & (xy <= frame_size)).all(axis=1)
        keypoints = np.zeros((pitch.num_keypoints, 3), np.float32)
        keypoints[shown, :2] = xy[shown]
        keypoints[shown, 2] = VISIBLE
        annotations = Annotations(
            box[None], [0], keypoints=keypoints[None], flip_idx=pitch.flip_idx
        )
        samples.append(Sample(image, clip.width, clip.height, annotations))
    return samples


def _pitch_box(
    homography: np.ndarray, frame_size: tuple[int, int], pitch: SoccerPitch
) -> np.ndarray | None:
    """``xyxy`` bounds of the part of the frame that shows ``pitch``, or ``None`` when none does.

    The pitch rectangle is cut at the horizon, mapped into the frame and cut
    to the frame; all three are convex, so the cuts are exact.
    """
    width, height = frame_size
    to_frame = np.linalg.inv(homography)
    centre = homography @ [width / 2, height / 2, 1]
    # w of a pitch point in the frame, positive on the frame centre's side.
    w = to_frame[2] * np.sign(to_frame[2] @ (centre / centre[2]))
    corners = np.array(
        [[0, 0], [pitch.length, 0], [pitch.length, pitch.width], [0, pitch.width]]
    )
    # Just short of the horizon, where frame coordinates grow without bound.
    margin = 1e-6 * (w @ np.append(centre[:2] / centre[2], 1))
    polygon = _cut(corners, w[0], w[1], w[2] - margin)
    if len(polygon) == 0:
        return None
    projected = np.column_stack([polygon, np.ones(len(polygon))]) @ to_frame.T
    polygon = projected[:, :2] / projected[:, 2:]
    for a, b, c in ((1, 0, 0), (-1, 0, width), (0, 1, 0), (0, -1, height)):
        polygon = _cut(polygon, a, b, c)
        if len(polygon) == 0:
            return None
    return np.concatenate([polygon.min(axis=0), polygon.max(axis=0)])


def _cut(polygon: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """The part of a convex ``(N, 2)`` polygon where ``a x + b y + c >= 0`` (Sutherland-Hodgman)."""
    values = polygon @ [a, b] + c
    kept = []
    for i, (point, value) in enumerate(zip(polygon, values, strict=True)):
        following, next_value = (
            polygon[(i + 1) % len(polygon)],
            values[(i + 1) % len(values)],
        )
        if value >= 0:
            kept.append(point)
        if (value >= 0) != (next_value >= 0):
            kept.append(point + value / (value - next_value) * (following - point))
    return np.array(kept).reshape(-1, 2)


def _frame_images(clip: LabelledClip) -> list[Path]:
    """The image file of each frame of ``clip``, as :class:`VideoReader` reads them.

    The lowest-numbered image is frame 0, and OpenCV only opens a sequence
    whose first image is numbered 0 to 4. The images must be exactly one
    per frame, numbered without a gap: a missing or extra image would move
    the labels onto other frames.
    """
    pattern = clip.video_path
    try:
        images = sequence_images(pattern)
    except ValueError:
        raise ValueError(
            f"Clip {clip.name} is read from {pattern}, not an image sequence, so "
            "its frames cannot be referenced in place; extract them first"
        ) from None
    numbers = [number for number, _ in images]
    first = numbers[0] if numbers else 0
    if first > _LAST_FIRST_IMAGE:
        raise ValueError(
            f"Clip {clip.name}: the first image matching {pattern} is "
            f"{images[0][1].name}, numbered {first}; OpenCV (VideoReader, the "
            f"pipeline) only opens a sequence starting at 0 to {_LAST_FIRST_IMAGE}"
        )
    if numbers != list(range(first, first + len(clip.frames))):
        found = f" ({images[0][1].name} to {images[-1][1].name})" if images else ""
        raise ValueError(
            f"Clip {clip.name} has {len(clip.frames)} frames but {pattern.parent} "
            f"holds {len(images)} images matching {pattern.name}{found}; it needs "
            "one image per frame, numbered without a gap"
        )
    return [path for _, path in images]


def _split_by_fractions[T](
    items: list[T], train: float, valid: float, test: float, seed: int
) -> dict[str, list[T]]:
    """Shuffle ``items`` and cut them into the non-empty splits of these fractions.

    Boundaries are rounded cumulatively and the remainder goes to the last
    split with a non-zero fraction, so a zero fraction stays empty.
    """
    fractions = {Split.TRAIN: train, Split.VALID: valid, Split.TEST: test}
    if any(f < 0 for f in fractions.values()):
        raise ValueError(f"Split fractions must not be negative, got {fractions}")
    total = train + valid + test
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split fractions must sum to 1, got {total}")
    pool = list(items)
    random.Random(seed).shuffle(pool)
    last = max(i for i, f in enumerate(fractions.values()) if f > 0)
    splits, begin, cumulative = {}, 0, 0.0
    for position, (name, fraction) in enumerate(fractions.items()):
        cumulative += fraction
        end = len(pool) if position >= last else round(len(pool) * cumulative)
        splits[name] = pool[begin:end]
        begin = end
    return {k: v for k, v in splits.items() if v}


def _subset_count(want: object, available: int) -> int:
    """Images to keep for a ``subset`` size: an integer count or a fraction in ``[0, 1]``."""
    if isinstance(want, numbers.Integral) and not isinstance(want, bool):
        if want < 0:
            raise ValueError(f"A subset count must be >= 0, got {want}")
        return int(want)
    if isinstance(want, numbers.Real) and not isinstance(want, bool):
        if not 0 <= want <= 1:
            raise ValueError(
                f"A subset fraction must be in [0, 1], got {want}; pass an int for a count"
            )
        return round(available * float(want))
    raise ValueError(f"A subset size must be a number, got {want!r}")


def load_dataset(path: str | Path) -> Dataset:
    """Load a YOLO (``data.yaml`` or its folder) or COCO dataset. See :class:`Dataset`."""
    return Dataset.load(path)
