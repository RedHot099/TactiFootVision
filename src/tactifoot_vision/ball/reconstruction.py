import math

from tactifoot_vision.ball.results import BallTrajectory, BallTrajectoryPoint
from tactifoot_vision.domain import FrameResult, PipelineResult, Track
from tactifoot_vision.enums import BallTrajectorySource


class LinearBallTrajectoryReconstructor:
    def __init__(self, *, max_speed_pixels_per_frame: float | None = None) -> None:
        self.max_speed_pixels_per_frame = max_speed_pixels_per_frame

    def reconstruct(self, result: PipelineResult) -> BallTrajectory:
        frame_indexes = tuple(frame.frame_index for frame in result.frames)
        if not frame_indexes:
            return BallTrajectory(())
        observations = self._filter_outliers(self._collect_observations(result.frames))
        if not observations:
            return BallTrajectory(
                tuple(
                    BallTrajectoryPoint(
                        frame_index=frame_index,
                        image_x=None,
                        image_y=None,
                        confidence=0.0,
                        source=BallTrajectorySource.MISSING,
                    )
                    for frame_index in frame_indexes
                )
            )
        by_frame = {point.frame_index: point for point in observations}
        points = [
            by_frame.get(frame_index)
            or self._reconstruct_missing(frame_index, observations)
            for frame_index in frame_indexes
        ]
        return BallTrajectory(tuple(points))

    def _collect_observations(
        self, frames: tuple[FrameResult, ...]
    ) -> list[BallTrajectoryPoint]:
        observations: list[BallTrajectoryPoint] = []
        for frame in frames:
            ball = _best_ball_track(tuple(frame.tracks))
            if ball is None:
                continue
            image_x = (ball.bbox.x1 + ball.bbox.x2) / 2.0
            image_y = (ball.bbox.y1 + ball.bbox.y2) / 2.0
            pitch_x = None
            pitch_y = None
            if frame.projection is not None and frame.projection.ball is not None:
                pitch_x = frame.projection.ball.x
                pitch_y = frame.projection.ball.y
            observations.append(
                BallTrajectoryPoint(
                    frame_index=frame.frame_index,
                    image_x=image_x,
                    image_y=image_y,
                    pitch_x=pitch_x,
                    pitch_y=pitch_y,
                    confidence=ball.confidence if ball.confidence is not None else 1.0,
                    source=BallTrajectorySource.OBSERVED,
                    uncertainty=0.0,
                )
            )
        return observations

    def _filter_outliers(
        self, observations: list[BallTrajectoryPoint]
    ) -> list[BallTrajectoryPoint]:
        max_speed = self.max_speed_pixels_per_frame
        if max_speed is None or len(observations) < 3:
            return observations
        kept = [observations[0]]
        for previous, current, following in zip(
            observations, observations[1:], observations[2:], strict=False
        ):
            previous_speed = _speed(previous, current)
            following_speed = _speed(current, following)
            bridge_speed = _speed(previous, following)
            is_spike = (
                previous_speed is not None
                and following_speed is not None
                and bridge_speed is not None
                and previous_speed > max_speed
                and following_speed > max_speed
                and bridge_speed <= max_speed
            )
            if not is_spike:
                kept.append(current)
        kept.append(observations[-1])
        return kept

    def _reconstruct_missing(
        self, frame_index: int, observations: list[BallTrajectoryPoint]
    ) -> BallTrajectoryPoint:
        previous = None
        following = None
        for observation in observations:
            if observation.frame_index < frame_index:
                previous = observation
            elif observation.frame_index > frame_index:
                following = observation
                break
        if previous is not None and following is not None:
            return _interpolate(frame_index, previous, following)
        nearest = previous or following
        if nearest is None:
            return BallTrajectoryPoint(
                frame_index=frame_index,
                image_x=None,
                image_y=None,
                confidence=0.0,
                source=BallTrajectorySource.MISSING,
            )
        frame_gap = abs(frame_index - nearest.frame_index)
        return BallTrajectoryPoint(
            frame_index=frame_index,
            image_x=nearest.image_x,
            image_y=nearest.image_y,
            pitch_x=nearest.pitch_x,
            pitch_y=nearest.pitch_y,
            confidence=nearest.confidence * 0.25,
            source=BallTrajectorySource.EXTRAPOLATED,
            uncertainty=float(frame_gap),
        )


def _best_ball_track(tracks: tuple[Track, ...]) -> Track | None:
    balls = [track for track in tracks if track.class_name == "ball"]
    if not balls:
        return None
    return max(balls, key=lambda track: track.confidence or 0.0)


def _interpolate(
    frame_index: int,
    previous: BallTrajectoryPoint,
    following: BallTrajectoryPoint,
) -> BallTrajectoryPoint:
    gap = following.frame_index - previous.frame_index
    ratio = (frame_index - previous.frame_index) / gap
    image_x = _lerp_optional(previous.image_x, following.image_x, ratio)
    image_y = _lerp_optional(previous.image_y, following.image_y, ratio)
    pitch_x = _lerp_optional(previous.pitch_x, following.pitch_x, ratio)
    pitch_y = _lerp_optional(previous.pitch_y, following.pitch_y, ratio)
    nearest_gap = min(
        frame_index - previous.frame_index, following.frame_index - frame_index
    )
    confidence_decay = max(0.1, 1.0 - (nearest_gap / (gap + 1)))
    return BallTrajectoryPoint(
        frame_index=frame_index,
        image_x=image_x,
        image_y=image_y,
        pitch_x=pitch_x,
        pitch_y=pitch_y,
        confidence=min(previous.confidence, following.confidence) * confidence_decay,
        source=BallTrajectorySource.INTERPOLATED,
        uncertainty=float(nearest_gap),
    )


def _lerp_optional(
    start: float | None, end: float | None, ratio: float
) -> float | None:
    if start is None or end is None:
        return None
    return start + (end - start) * ratio


def _speed(start: BallTrajectoryPoint, end: BallTrajectoryPoint) -> float | None:
    if (
        start.image_x is None
        or start.image_y is None
        or end.image_x is None
        or end.image_y is None
    ):
        return None
    frame_gap = end.frame_index - start.frame_index
    if frame_gap <= 0:
        return None
    distance = math.hypot(end.image_x - start.image_x, end.image_y - start.image_y)
    return distance / frame_gap
