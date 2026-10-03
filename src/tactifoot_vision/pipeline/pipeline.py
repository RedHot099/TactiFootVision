"""The inference pipeline: detection -> tracking -> pitch projection -> teams."""

import logging
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import supervision as sv
from tqdm.auto import tqdm

from tactifoot_vision.data.annotations import Task
from tactifoot_vision.data.video import VideoReader
from tactifoot_vision.models.base import Model
from tactifoot_vision.pipeline.result import (
    NO_TEAM,
    FrameResult,
    ObjectMasks,
    PipelineResult,
)
from tactifoot_vision.pitch.homography import HomographyEstimator, frame_to_pitch
from tactifoot_vision.pitch.pitch import SoccerPitch
from tactifoot_vision.teams.classifier import TeamClassifier
from tactifoot_vision.teams.crops import extract_crops
from tactifoot_vision.tracking.ball import clean_ball_path
from tactifoot_vision.tracking.base import Tracker, create_tracker

logger = logging.getLogger(__name__)

GOALKEEPER = "goalkeeper"


class Pipeline:
    """Turns a match video into per-frame tracks, pitch positions and teams.

    Per frame: detect, split off the ball (best-confidence ``ball_class``
    detection), track the people, detect pitch keypoints, update the
    homography and project everyone onto the pitch (people at the bottom
    centre of their box, the ball at its centre). After the pass, every track
    gets one team by majority vote over its crops, and ball positions that
    jump implausibly far are removed.

    Args:
        detector: a detection :class:`Model` (``data["class_name"]`` filled).
        keypoint_model: a pose :class:`Model` predicting the pitch landmarks;
            without it there is no homography and ``pitch_xy`` is NaN.
        tracker: a :class:`Tracker`, a registered tracker name, or ``None``
            (no tracking; each detection is then classified on its own).
        team_classifier: a :class:`TeamClassifier`. An unfitted one is fitted on
            this video's player crops; a fitted one is reused as is, so give each
            match a new classifier. ``None`` skips teams.
        pitch: pitch model; defaults to ``homography.pitch`` or ``SoccerPitch()``.
        homography: homography estimator; one is created when omitted.
        ball_class: detector class name of the ball.
        include_classes: people classes to keep (the ball is always kept); ``None`` keeps all.
        team_classes: classes that belong to a team. ``"goalkeeper"`` joins the
            team whose players are nearest on the pitch; the others are
            clustered by kit. Remaining classes (referees) get ``NO_TEAM``.
        team_sample_stride: crop tracks for the team vote every this many
            processed frames (plus the first frame of every track).
        team_samples_per_track: votes kept per track (a uniform random sample of
            its crops), which bounds memory on full matches.
        ball_max_speed: fastest plausible ball, in pitch units per second (m/s
            for the default pitch); faster jumps between kept ball positions
            are outliers whose ``pitch_xy`` becomes NaN (see
            :func:`~tactifoot_vision.tracking.clean_ball_path`). ``None`` keeps all.
        crop_scale, crop_center_ratio: crop geometry passed to
            :func:`~tactifoot_vision.teams.extract_crops`.
        keep_masks: keep the tracker's segmentation masks (SAM2) in
            :attr:`FrameResult.masks` so ``FrameAnnotator(draw_masks=True)`` can
            draw them. Each mask is cropped to the pixels it covers
            (:class:`ObjectMasks`), so memory grows with the people's on-screen
            area instead of the frame area: on a wide 1080p broadcast view about
            1 kB per person per frame (~40 MB per minute at 25 fps with 23
            people), where full-frame masks would take 2 MB per person per frame.
            Close-ups cost more. Off by default.
    """

    def __init__(
        self,
        detector: Model,
        keypoint_model: Model | None = None,
        tracker: Tracker | str | None = "bytetrack",
        team_classifier: TeamClassifier | None = None,
        pitch: SoccerPitch | None = None,
        homography: HomographyEstimator | None = None,
        ball_class: str = "ball",
        include_classes: Sequence[str] | None = None,
        team_classes: Sequence[str] = ("player", GOALKEEPER),
        team_sample_stride: int = 10,
        team_samples_per_track: int = 20,
        ball_max_speed: float | None = 40.0,
        crop_scale: float = 0.8,
        crop_center_ratio: float = 1.0,
        keep_masks: bool = False,
    ) -> None:
        if detector.task != Task.DETECT:
            raise ValueError(
                f"detector must be a detection model, got a {detector.task} model"
            )
        if keypoint_model is not None and keypoint_model.task != Task.POSE:
            raise ValueError(
                f"keypoint_model must be a pose model, got a {keypoint_model.task} model"
            )
        if team_sample_stride < 1:
            raise ValueError("team_sample_stride must be >= 1")
        if ball_max_speed is not None and ball_max_speed <= 0:
            raise ValueError("ball_max_speed must be positive or None")
        if pitch is not None and homography is not None and homography.pitch != pitch:
            raise ValueError(
                f"pitch {pitch} differs from the homography estimator's {homography.pitch}"
            )
        self.detector = detector
        self.keypoint_model = keypoint_model
        self.tracker = create_tracker(tracker) if isinstance(tracker, str) else tracker
        self.team_classifier = team_classifier
        self.pitch = pitch or (homography.pitch if homography else SoccerPitch())
        self.homography = homography or HomographyEstimator(self.pitch)
        self.ball_class = ball_class
        self.include_classes = (
            None if include_classes is None else tuple(include_classes)
        )
        self.team_classes = tuple(team_classes)
        self.team_sample_stride = team_sample_stride
        self.team_samples_per_track = team_samples_per_track
        missing = {ball_class, *self.team_classes} - set(detector.class_names)
        if missing:
            logger.warning(
                "The detector has no class %s (it has %s); check ball_class / team_classes",
                ", ".join(sorted(missing)),
                ", ".join(detector.class_names),
            )
        self.ball_max_speed = ball_max_speed
        self.crop_scale = crop_scale
        self.crop_center_ratio = crop_center_ratio
        self.keep_masks = keep_masks

    def run(
        self,
        video: str | Path | VideoReader,
        start: int = 0,
        end: int | None = None,
        stride: int = 1,
        progress: bool = True,
    ) -> PipelineResult:
        """Process the frames ``start <= index < end`` every ``stride`` frames.

        ``end=None`` runs to the end of the video, like :meth:`VideoReader.frames`.
        """
        reader = video if isinstance(video, VideoReader) else VideoReader(video)
        total = len(
            range(start, min(end or reader.frame_count, reader.frame_count), stride)
        )
        if self.tracker is not None:
            self.tracker.reset(fps=reader.fps / stride)
        self.homography.reset()
        samples = _TeamSamples(per_key=self.team_samples_per_track)
        frames: list[FrameResult] = []
        keys: list[np.ndarray] = []
        for position, (index, frame) in enumerate(
            tqdm(
                reader.frames(start=start, end=end, stride=stride),
                total=total or None,
                disable=not progress,
                desc="Pipeline",
                unit="frame",
            )
        ):
            result = self._process_frame(index, frame, reader.fps)
            frame_keys = _object_keys(result.detections, samples)
            if self.team_classifier is not None:
                self._sample_crops(
                    frame, result.detections, frame_keys, position, samples
                )
            frames.append(result)
            keys.append(frame_keys)
        logger.info("Processed %d frames of %s", len(frames), reader.path)
        if len(frames) < total:
            logger.warning(
                "%s ended after %d of %d expected frames (unreadable or truncated video?)",
                reader.path,
                len(frames),
                total,
            )
        if self.team_classifier is not None:
            self._assign_teams(frames, keys, samples)
        if self.ball_max_speed is not None:
            self._clean_ball(frames, max_jump=self.ball_max_speed * stride / reader.fps)
        return PipelineResult(
            frames=frames,
            fps=reader.fps,
            frame_size=reader.size,
            class_names=list(self.detector.class_names),
            pitch=self.pitch,
            video_path=reader.path,
        )

    # -------------------------------------------------------------- per frame
    def _process_frame(self, index: int, frame: np.ndarray, fps: float) -> FrameResult:
        detections = self.detector.predict(frame)
        if "class_name" not in detections.data:
            if len(detections):
                raise ValueError(
                    f"{type(self.detector).__name__} must fill data['class_name']"
                )
            detections.data["class_name"] = np.zeros(0, dtype=str)
        is_ball = detections.data["class_name"] == self.ball_class
        ball = detections[is_ball]
        if len(ball) > 1:
            ball = ball[[int(np.argmax(ball.confidence))]]
        people = detections[~is_ball]
        if self.include_classes is not None:
            people = people[np.isin(people.data["class_name"], self.include_classes)]
        if self.tracker is not None:
            people = self.tracker.update(people, frame)
        # Full-frame masks (SAM2) cost ~2 MB per object per 1080p frame: keep crops only.
        masks = None
        if self.keep_masks and people.mask is not None:
            masks = ObjectMasks.from_dense(people.mask)
        people.mask = None

        keypoints = None
        matrix = None
        if self.keypoint_model is not None:
            keypoints = self.keypoint_model.predict(frame)
            matrix = self.homography.update(keypoints)
        people.data["pitch_xy"] = _project(people, sv.Position.BOTTOM_CENTER, matrix)
        ball.data["pitch_xy"] = _project(ball, sv.Position.CENTER, matrix)
        people.data["team_id"] = np.full(len(people), NO_TEAM, dtype=int)
        return FrameResult(
            index=index,
            timestamp=index / fps,
            detections=people,
            ball=ball,
            keypoints=keypoints,
            homography=matrix,
            masks=masks,
        )

    def _sample_crops(
        self,
        frame: np.ndarray,
        people: sv.Detections,
        keys: np.ndarray,
        position: int,
        samples: "_TeamSamples",
    ) -> None:
        """Embed crops of team-class people on sampled frames and on each track's first frame."""
        on_stride = position % self.team_sample_stride == 0
        candidates = [
            i
            for i, (key, name) in enumerate(
                zip(keys, people.data["class_name"], strict=True)
            )
            if name in self.team_classes and (on_stride or key not in samples.seen)
        ]
        crops = extract_crops(
            frame,
            people.xyxy[candidates],
            scale=self.crop_scale,
            center_ratio=self.crop_center_ratio,
        )
        # Ask the sample only for valid crops: accepting reserves a slot and marks
        # the track as seen, which a crop that is never embedded must not do.
        valid = [
            (i, crop)
            for i, crop in zip(candidates, crops, strict=True)
            if crop is not None and samples.accepts(int(keys[i]))
        ]
        if not valid:
            return
        embeddings = self.team_classifier.embed([crop for _, crop in valid])
        for (i, _), embedding in zip(valid, embeddings, strict=True):
            samples.add(int(keys[i]), embedding)

    # ------------------------------------------------------------ after pass
    def _assign_teams(
        self, frames: list[FrameResult], keys: list[np.ndarray], samples: "_TeamSamples"
    ) -> None:
        """Majority vote per track; goalkeepers join the nearest team; write ``team_id``."""
        classifier = self.team_classifier
        track_class = _majority_classes(frames, keys)
        embeddings = samples.embeddings()
        sample_classes = np.array([track_class[k] for k in samples.keys], dtype=str)
        is_player = np.isin(sample_classes, self.team_classes) & (
            sample_classes != GOALKEEPER
        )
        if classifier.is_fitted:
            logger.info("Using the already fitted team classifier")
        else:
            if is_player.sum() < classifier.n_teams:
                logger.warning(
                    "Only %d player crops; not enough to fit the team classifier, "
                    "all team ids stay NO_TEAM",
                    is_player.sum(),
                )
                return
            classifier.fit_embeddings(embeddings[is_player])
        labels = classifier.predict_embeddings(embeddings)
        votes: dict[int, Counter[int]] = defaultdict(Counter)
        for key, label in zip(samples.keys, labels, strict=True):
            votes[key][int(label)] += 1
        teams = {
            key: votes[key].most_common(1)[0][0]
            for key, cls in track_class.items()
            if cls in self.team_classes and cls != GOALKEEPER and key in votes
        }
        if GOALKEEPER in self.team_classes:
            goalkeepers = [k for k, cls in track_class.items() if cls == GOALKEEPER]
            teams |= _goalkeeper_teams(goalkeepers, frames, keys, teams, votes)
        for frame, frame_keys in zip(frames, keys, strict=True):
            frame.detections.data["team_id"] = np.array(
                [teams.get(int(k), NO_TEAM) for k in frame_keys], dtype=int
            )
        assigned = sum(team != NO_TEAM for team in teams.values())
        logger.info("Assigned teams to %d of %d tracks", assigned, len(track_class))

    @staticmethod
    def _clean_ball(frames: list[FrameResult], max_jump: float) -> None:
        raw = np.array(
            [f.ball_xy[0] if len(f.ball) else (np.nan, np.nan) for f in frames],
            dtype=float,
        ).reshape(-1, 2)
        cleaned = clean_ball_path(raw, max_jump)
        dropped = 0
        for frame, before, after in zip(frames, raw, cleaned, strict=True):
            if np.isfinite(before).all() and not np.isfinite(after).all():
                frame.ball.data["pitch_xy"] = np.full((1, 2), np.nan)
                dropped += 1
        logger.info(
            "Ball cleaning removed %d of %d pitch positions",
            dropped,
            int(np.isfinite(raw).all(axis=1).sum()),
        )

    def __repr__(self) -> str:
        return (
            f"Pipeline(detector={self.detector!r}, keypoint_model={self.keypoint_model!r}, "
            f"tracker={type(self.tracker).__name__ if self.tracker else None}, "
            f"team_classifier={self.team_classifier!r})"
        )


@dataclass
class _TeamSamples:
    """Crop embeddings collected during the pass, at most ``per_key`` per track.

    Reservoir sampling keeps a uniform sample of each track's crops, so memory
    stays bounded on full matches; :meth:`accepts` is asked before embedding so
    rejected crops are never embedded.
    """

    per_key: int
    by_key: dict[int, list[np.ndarray]] = field(default_factory=dict)
    offered: Counter[int] = field(default_factory=Counter)
    seen: set[int] = field(default_factory=set)
    next_untracked: int = -1  # untracked detections get unique negative keys
    _pending: dict[int, int] = field(default_factory=dict)  # key -> slot to fill
    _rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))

    def accepts(self, key: int) -> bool:
        """Whether the next crop of ``key`` enters the sample (reserves its slot)."""
        self.seen.add(key)
        self.offered[key] += 1
        stored = len(self.by_key.get(key, ()))
        if stored < self.per_key:
            self._pending[key] = stored
            return True
        slot = int(self._rng.integers(self.offered[key]))
        if slot < self.per_key:
            self._pending[key] = slot
            return True
        return False

    def add(self, key: int, embedding: np.ndarray) -> None:
        rows = self.by_key.setdefault(key, [])
        slot = self._pending.pop(key, len(rows))
        value = np.asarray(embedding, dtype=np.float16)
        if slot < len(rows):
            rows[slot] = value
        else:
            rows.append(value)

    @property
    def keys(self) -> list[int]:
        return [key for key, rows in self.by_key.items() for _ in rows]

    def embeddings(self) -> np.ndarray:
        rows = [row for rows in self.by_key.values() for row in rows]
        return (
            np.stack(rows).astype(np.float32) if rows else np.zeros((0, 0), np.float32)
        )


def _object_keys(detections: sv.Detections, samples: _TeamSamples) -> np.ndarray:
    """Track ids, or fresh negative ids when there is no tracker."""
    if detections.tracker_id is not None:
        return np.asarray(detections.tracker_id, dtype=int)
    first = samples.next_untracked
    samples.next_untracked -= len(detections)
    return np.arange(first, first - len(detections), -1, dtype=int)


def _project(
    detections: sv.Detections, anchor: sv.Position, matrix: np.ndarray | None
) -> np.ndarray:
    if matrix is None or len(detections) == 0:
        return np.full((len(detections), 2), np.nan, dtype=np.float32)
    return frame_to_pitch(detections.get_anchors_coordinates(anchor), matrix)


def _majority_classes(
    frames: list[FrameResult], keys: list[np.ndarray]
) -> dict[int, str]:
    """Most frequent detector class of every track (detections may flicker between classes)."""
    counts: dict[int, Counter[str]] = defaultdict(Counter)
    for frame, frame_keys in zip(frames, keys, strict=True):
        for key, name in zip(
            frame_keys, frame.detections.data["class_name"], strict=True
        ):
            counts[int(key)][str(name)] += 1
    return {key: count.most_common(1)[0][0] for key, count in counts.items()}


def _goalkeeper_teams(
    goalkeepers: list[int],
    frames: list[FrameResult],
    keys: list[np.ndarray],
    teams: dict[int, int],
    votes: dict[int, Counter[int]],
) -> dict[int, int]:
    """Team of each goalkeeper track: the team whose players are nearest on the pitch.

    Compares the goalkeeper's mean pitch position with each team's mean player
    position over the frames the goalkeeper is visible in (one pass over the
    frames). Without pitch positions the goalkeeper keeps its own crop vote
    (or ``NO_TEAM``).
    """
    wanted = set(goalkeepers)
    own: dict[int, list[np.ndarray]] = defaultdict(list)
    team_sums: dict[int, dict[int, np.ndarray]] = defaultdict(
        dict
    )  # gk -> team -> (x, y, n)
    for frame, frame_keys in zip(frames, keys, strict=True):
        present = [(i, int(k)) for i, k in enumerate(frame_keys) if int(k) in wanted]
        if not present:
            continue
        xy = frame.pitch_xy
        frame_sums: dict[int, np.ndarray] = {}
        for key, point in zip(frame_keys, xy, strict=True):
            team = teams.get(int(key))
            if team is not None and np.isfinite(point).all():
                frame_sums[team] = frame_sums.get(team, np.zeros(3)) + (*point, 1.0)
        for i, goalkeeper in present:
            own[goalkeeper].append(xy[i])
            sums = team_sums[goalkeeper]
            for team, total in frame_sums.items():
                sums[team] = sums.get(team, np.zeros(3)) + total

    result: dict[int, int] = {}
    for goalkeeper in goalkeepers:
        position = _nanmean(own[goalkeeper])
        means = {t: total[:2] / total[2] for t, total in team_sums[goalkeeper].items()}
        if position is not None and means:
            result[goalkeeper] = min(
                means, key=lambda t: np.linalg.norm(means[t] - position)
            )
        elif goalkeeper in votes:
            result[goalkeeper] = votes[goalkeeper].most_common(1)[0][0]
    return result


def _nanmean(points: list[np.ndarray]) -> np.ndarray | None:
    """Mean of the finite points, ``None`` when there are none."""
    if not points:
        return None
    array = np.asarray(points, dtype=float).reshape(-1, 2)
    array = array[np.isfinite(array).all(axis=1)]
    return array.mean(axis=0) if len(array) else None
