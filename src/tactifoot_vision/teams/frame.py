"""Single-frame team mode: teams of one frame's people from that frame alone."""

import logging

import numpy as np
import supervision as sv

from tactifoot_vision.teams.classifier import NO_TEAM, TeamClassifier
from tactifoot_vision.teams.crops import extract_crops

logger = logging.getLogger(__name__)


def assign_frame_teams(
    frame: np.ndarray,
    detections: sv.Detections,
    classifier: TeamClassifier,
    *,
    positions: np.ndarray | None = None,
    player_class: str = "player",
    goalkeeper_class: str = "goalkeeper",
    crop_scale: float = 0.8,
    crop_center_ratio: float = 1.0,
) -> np.ndarray:
    """Team id of every detection, from this frame alone.

    For frames that are not part of a video run (one frame per shot, say),
    where :class:`~tactifoot_vision.pipeline.Pipeline`'s vote over a track is
    not available. The classifier is fitted again on this frame's
    ``player_class`` crops (about 15 on a broadcast frame), so it must cluster
    the embeddings directly: ``TeamClassifier(reducer=None)``.

    Goalkeepers join the team whose players' mean position is nearest, using
    ``positions`` (``(N, 2)``, e.g. pitch metres; rows may be NaN) where the
    goalkeeper and at least one team have one, else the bottom centres of the
    boxes in frame pixels. Every other class (referees) gets ``NO_TEAM``, and
    so does everyone when fewer than ``classifier.n_teams`` players have a
    valid crop. Team numbers are only comparable within one frame.
    """
    if classifier.reducer is not None:
        raise ValueError(
            "Single-frame teams cluster about 15 crops, too few for "
            f"reducer={classifier.reducer!r}; use TeamClassifier(reducer=None)"
        )
    if positions is not None and np.shape(positions) != (len(detections), 2):
        raise ValueError(
            f"positions must be ({len(detections)}, 2), got {np.shape(positions)}"
        )
    names = detections.data.get("class_name", np.zeros(len(detections), dtype=str))
    teams = np.full(len(detections), NO_TEAM, dtype=int)
    players = np.flatnonzero(names == player_class)
    crops = extract_crops(
        frame,
        detections.xyxy[players],
        scale=crop_scale,
        center_ratio=crop_center_ratio,
    )
    valid = [
        (i, crop) for i, crop in zip(players, crops, strict=True) if crop is not None
    ]
    if len(valid) < classifier.n_teams:
        logger.info(
            "Only %d player crops in the frame; no teams (need %d)",
            len(valid),
            classifier.n_teams,
        )
        return teams
    labels = classifier.fit_predict([crop for _, crop in valid])
    teams[[i for i, _ in valid]] = labels
    player_teams = teams.copy()
    pixels = detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    for i in np.flatnonzero(names == goalkeeper_class):
        if positions is not None:
            teams[i] = _nearest_team(np.asarray(positions, float), player_teams, i)
        if teams[i] == NO_TEAM:
            teams[i] = _nearest_team(pixels.astype(float), player_teams, i)
    return teams


def _nearest_team(points: np.ndarray, teams: np.ndarray, goalkeeper: int) -> int:
    """Team whose finite player points have the mean nearest ``points[goalkeeper]``."""
    if not np.isfinite(points[goalkeeper]).all():
        return NO_TEAM
    finite = np.isfinite(points).all(axis=1)
    best, best_distance = NO_TEAM, np.inf
    for team in np.unique(teams[teams != NO_TEAM]):
        members = (teams == team) & finite
        if members.any():
            distance = np.linalg.norm(points[members].mean(axis=0) - points[goalkeeper])
            if distance < best_distance:
                best, best_distance = int(team), distance
    return best
