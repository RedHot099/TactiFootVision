import math
from collections.abc import Sequence

from tactifoot_vision.ball import BallTrajectory, BallTrajectoryPoint
from tactifoot_vision.domain import FrameResult, PitchPoint
from tactifoot_vision.enums import ShotOutcome
from tactifoot_vision.shots import ShotCandidate
from tactifoot_vision.xg.results import XgShotFeatures

DEFAULT_PITCH_LENGTH = 105.0
DEFAULT_PITCH_WIDTH = 68.0
GOAL_WIDTH = 7.32


def build_shot_features(
    *,
    candidate: ShotCandidate,
    trajectory: BallTrajectory,
    frame_result: FrameResult | None,
    image_width: int = 1920,
    image_height: int = 1080,
    pitch_length: float = DEFAULT_PITCH_LENGTH,
    pitch_width: float = DEFAULT_PITCH_WIDTH,
    attacking_goal_x: float | None = None,
) -> XgShotFeatures:
    point = trajectory.point_at(candidate.frame_index)
    shot_x, shot_y = _pitch_coordinates(
        point,
        image_width=image_width,
        image_height=image_height,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
    )
    goal_x = _goal_x(shot_x, pitch_length, attacking_goal_x)
    goal_y = pitch_width / 2.0
    distance = math.hypot(goal_x - shot_x, goal_y - shot_y)
    angle = _shot_angle(
        shot_x=shot_x,
        shot_y=shot_y,
        goal_x=goal_x,
        goal_y=goal_y,
    )
    projected_players = _projected_players(frame_result)
    nearest = _nearest_distance(shot_x, shot_y, projected_players)
    goalkeeper = _nearest_distance(
        shot_x,
        shot_y,
        [
            point
            for class_name, point in projected_players
            if class_name == "goalkeeper"
        ],
    )
    defenders_in_cone = _count_in_goal_cone(
        shot_x=shot_x,
        shot_y=shot_y,
        goal_x=goal_x,
        goal_y=goal_y,
        players=[
            point
            for class_name, point in projected_players
            if class_name != "goalkeeper"
        ],
    )
    return XgShotFeatures(
        shot_x=shot_x,
        shot_y=shot_y,
        distance_to_goal=distance,
        angle_to_goal=angle,
        centrality=max(0.0, 1.0 - abs(shot_y - goal_y) / goal_y),
        ball_speed=_ball_speed(trajectory, candidate.frame_index),
        nearest_player_distance=nearest,
        goalkeeper_distance=goalkeeper,
        defender_count_in_cone=defenders_in_cone,
        is_penalty=candidate.outcome == ShotOutcome.PENALTY,
    )


def _pitch_coordinates(
    point: BallTrajectoryPoint | None,
    *,
    image_width: int,
    image_height: int,
    pitch_length: float,
    pitch_width: float,
) -> tuple[float, float]:
    if point is None:
        return pitch_length / 2.0, pitch_width / 2.0
    if point.pitch_x is not None and point.pitch_y is not None:
        return point.pitch_x, point.pitch_y
    if point.image_x is None or point.image_y is None:
        return pitch_length / 2.0, pitch_width / 2.0
    return (
        min(max(point.image_x / image_width, 0.0), 1.0) * pitch_length,
        min(max(point.image_y / image_height, 0.0), 1.0) * pitch_width,
    )


def _goal_x(
    shot_x: float, pitch_length: float, attacking_goal_x: float | None
) -> float:
    if attacking_goal_x is not None:
        return attacking_goal_x
    return 0.0 if shot_x < pitch_length / 2.0 else pitch_length


def _shot_angle(*, shot_x: float, shot_y: float, goal_x: float, goal_y: float) -> float:
    post_top_y = goal_y - GOAL_WIDTH / 2.0
    post_bottom_y = goal_y + GOAL_WIDTH / 2.0
    distance_top = math.hypot(goal_x - shot_x, post_top_y - shot_y)
    distance_bottom = math.hypot(goal_x - shot_x, post_bottom_y - shot_y)
    if distance_top == 0.0 or distance_bottom == 0.0:
        return math.pi
    value = min(max(GOAL_WIDTH / max(distance_top * distance_bottom, 1e-9), -1.0), 1.0)
    return math.asin(value)


def _projected_players(
    frame_result: FrameResult | None,
) -> list[tuple[str, PitchPoint]]:
    if frame_result is None or frame_result.projection is None:
        return []
    projected = frame_result.projection.points_by_track_id
    players = []
    for track in frame_result.tracks:
        if track.class_name == "ball":
            continue
        point = projected.get(track.track_id)
        if point is not None:
            players.append((track.class_name, point))
    return players


def _nearest_distance(
    shot_x: float,
    shot_y: float,
    players: Sequence[tuple[str, PitchPoint]] | Sequence[PitchPoint],
) -> float | None:
    distances = []
    for player in players:
        point = player[1] if isinstance(player, tuple) else player
        distances.append(math.hypot(point.x - shot_x, point.y - shot_y))
    return min(distances) if distances else None


def _count_in_goal_cone(
    *,
    shot_x: float,
    shot_y: float,
    goal_x: float,
    goal_y: float,
    players: Sequence[PitchPoint],
) -> int:
    dx = goal_x - shot_x
    dy = goal_y - shot_y
    length_sq = dx * dx + dy * dy
    if length_sq == 0.0:
        return 0
    count = 0
    for player in players:
        px = player.x - shot_x
        py = player.y - shot_y
        t = (px * dx + py * dy) / length_sq
        if not 0.0 < t < 1.0:
            continue
        projected_x = shot_x + t * dx
        projected_y = shot_y + t * dy
        if math.hypot(player.x - projected_x, player.y - projected_y) <= 2.5:
            count += 1
    return count


def _ball_speed(trajectory: BallTrajectory, frame_index: int) -> float | None:
    by_frame = trajectory.by_frame()
    current = by_frame.get(frame_index)
    if current is None or current.image_x is None or current.image_y is None:
        return None
    previous_frames = [
        point
        for point in trajectory.points
        if point.frame_index < frame_index
        and point.image_x is not None
        and point.image_y is not None
    ]
    if not previous_frames:
        return None
    previous = previous_frames[-1]
    previous_x = previous.image_x
    previous_y = previous.image_y
    if previous_x is None or previous_y is None:
        return None
    frame_gap = frame_index - previous.frame_index
    if frame_gap <= 0:
        return None
    return (
        math.hypot(
            current.image_x - previous_x,
            current.image_y - previous_y,
        )
        / frame_gap
    )
