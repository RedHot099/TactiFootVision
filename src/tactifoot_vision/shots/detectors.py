import math
from collections.abc import Sequence

from tactifoot_vision.ball import BallTrajectory, BallTrajectoryPoint
from tactifoot_vision.domain import FrameResult
from tactifoot_vision.enums import ShotDetectorKind, ShotOutcome
from tactifoot_vision.shots.results import ShotCandidate, ShotWindow


class MetadataShotDetector:
    def __init__(
        self,
        *,
        action_frames: int | Sequence[int],
        action_class: str | None = None,
        window_before: int = 64,
        window_after: int = 32,
    ) -> None:
        self.action_frames: tuple[int, ...]
        if isinstance(action_frames, int):
            self.action_frames = (action_frames,)
        else:
            self.action_frames = tuple(int(frame) for frame in action_frames)
        self.action_class = action_class
        self.window_before = window_before
        self.window_after = window_after

    def detect(
        self,
        ball_trajectory: BallTrajectory,
        frame_results: Sequence[FrameResult] = (),
    ) -> tuple[ShotCandidate, ...]:
        _ = frame_results
        if not ball_trajectory.points:
            return ()
        min_frame = min(ball_trajectory.frame_indexes)
        max_frame = max(ball_trajectory.frame_indexes)
        outcome = outcome_from_action_class(self.action_class)
        candidates = []
        for action_frame in self.action_frames:
            frame_index = min(max(action_frame, min_frame), max_frame)
            candidates.append(
                ShotCandidate(
                    frame_index=frame_index,
                    window=_window(
                        frame_index,
                        min_frame=min_frame,
                        max_frame=max_frame,
                        before=self.window_before,
                        after=self.window_after,
                    ),
                    confidence=1.0,
                    detector_kind=ShotDetectorKind.METADATA,
                    outcome=outcome,
                    data={"action_class": self.action_class},
                )
            )
        return tuple(candidates)


class KinematicShotDetector:
    def __init__(
        self,
        *,
        window_before: int = 64,
        window_after: int = 32,
        max_candidates: int = 1,
        min_speed_pixels_per_frame: float = 0.0,
        suppression_frames: int | None = None,
    ) -> None:
        self.window_before = window_before
        self.window_after = window_after
        self.max_candidates = max_candidates
        self.min_speed_pixels_per_frame = min_speed_pixels_per_frame
        self.suppression_frames = suppression_frames

    def detect(
        self,
        ball_trajectory: BallTrajectory,
        frame_results: Sequence[FrameResult] = (),
    ) -> tuple[ShotCandidate, ...]:
        _ = frame_results
        if self.max_candidates <= 0:
            return ()
        points = ball_trajectory.available_points()
        if len(points) < 2 or not ball_trajectory.points:
            return ()
        speeds = _speeds(points)
        if not speeds:
            return ()
        max_speed = max(speed for _, speed in speeds)
        if max_speed <= 0.0:
            return ()
        min_frame = min(ball_trajectory.frame_indexes)
        max_frame = max(ball_trajectory.frame_indexes)
        suppression = self.suppression_frames
        if suppression is None:
            suppression = max(1, (self.window_before + self.window_after) // 2)
        selected: list[tuple[int, float]] = []
        for frame_index, speed in sorted(
            speeds, key=lambda item: item[1], reverse=True
        ):
            if speed < self.min_speed_pixels_per_frame:
                continue
            if any(
                abs(frame_index - other_frame) <= suppression
                for other_frame, _ in selected
            ):
                continue
            selected.append((frame_index, speed))
            if len(selected) >= self.max_candidates:
                break
        return tuple(
            ShotCandidate(
                frame_index=frame_index,
                window=_window(
                    frame_index,
                    min_frame=min_frame,
                    max_frame=max_frame,
                    before=self.window_before,
                    after=self.window_after,
                ),
                confidence=speed / max_speed,
                detector_kind=ShotDetectorKind.KINEMATIC,
                data={"speed_pixels_per_frame": speed},
            )
            for frame_index, speed in sorted(selected)
        )


def outcome_from_action_class(action_class: str | None) -> ShotOutcome:
    normalized = (action_class or "").strip().lower()
    if normalized == "goal":
        return ShotOutcome.GOAL
    if normalized == "shots on target":
        return ShotOutcome.ON_TARGET
    if normalized == "shots off target":
        return ShotOutcome.OFF_TARGET
    if normalized == "penalty":
        return ShotOutcome.PENALTY
    return ShotOutcome.UNKNOWN


def is_shot_like_action(action_class: str | None) -> bool:
    return outcome_from_action_class(action_class) != ShotOutcome.UNKNOWN


def _window(
    frame_index: int,
    *,
    min_frame: int,
    max_frame: int,
    before: int,
    after: int,
) -> ShotWindow:
    return ShotWindow(
        start_frame=max(min_frame, frame_index - before),
        end_frame=min(max_frame, frame_index + after),
    )


def _speeds(points: tuple[BallTrajectoryPoint, ...]) -> list[tuple[int, float]]:
    speeds: list[tuple[int, float]] = []
    for previous, current in zip(points, points[1:], strict=False):
        if (
            previous.image_x is None
            or previous.image_y is None
            or current.image_x is None
            or current.image_y is None
        ):
            continue
        frame_gap = current.frame_index - previous.frame_index
        if frame_gap <= 0:
            continue
        distance = math.hypot(
            current.image_x - previous.image_x,
            current.image_y - previous.image_y,
        )
        speeds.append((current.frame_index, distance / frame_gap))
    return speeds
