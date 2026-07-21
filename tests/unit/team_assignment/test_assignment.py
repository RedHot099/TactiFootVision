import numpy as np
import pytest

from tactifoot_vision.config import TeamAssignmentConfig
from tactifoot_vision.domain import BBox, Frame, Track, TrackSet
from tactifoot_vision.team_assignment import TeamAssigner


class SequenceClusterer:
    def fit_predict(self, features):
        return np.arange(len(features), dtype=np.int_)

    def predict(self, features):
        return np.arange(len(features), dtype=np.int_)


def test_team_assigner_requires_fit_before_predict() -> None:
    with pytest.raises(RuntimeError):
        TeamAssigner().predict([np.zeros((8, 8, 3), dtype=np.uint8)])


def test_team_assigner_from_config_assigns_two_teams() -> None:
    assigner = TeamAssigner.from_config(TeamAssignmentConfig(clusters=2))
    crops = [
        np.full((8, 8, 3), (255, 0, 0), dtype=np.uint8),
        np.full((8, 8, 3), (0, 255, 0), dtype=np.uint8),
    ]

    labels = assigner.fit(crops).predict(crops)

    assert len(labels) == 2
    assert len(set(labels.tolist())) == 2
    assert assigner.is_fitted


def test_assign_tracks_preserves_ids_and_adds_team_ids() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[:, :10] = (255, 0, 0)
    image[:, 10:] = (0, 255, 0)
    tracks = TrackSet(
        (
            Track(
                1,
                BBox(0, 0, 8, 10),
                2,
                "player",
                confidence=0.7,
                data={"source": "test"},
            ),
            Track(2, BBox(12, 0, 19, 10), 2, "player"),
            Track(3, BBox(5, 5, 7, 7), 0, "ball", team_id=99),
            Track(4, BBox(8, 8, 10, 10), 3, "referee"),
        )
    )
    assigner = TeamAssigner.from_config(TeamAssignmentConfig(clusters=2))
    assigner.fit([image[:, :10], image[:, 10:]])

    assigned = assigner.assign_tracks(Frame(index=0, image=image), tracks)

    assert [track.track_id for track in assigned] == [1, 2, 3, 4]
    assert assigned.tracks[0].confidence == 0.7
    assert assigned.tracks[0].data == {"source": "test"}
    assert assigned.tracks[0].team_id is not None
    assert assigned.tracks[1].team_id is not None
    assert assigned.tracks[2].team_id == 99
    assert assigned.tracks[3].team_id is None


def test_fit_empty_crops_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="at least one valid crop"):
        TeamAssigner().fit([])


def test_predict_empty_crops_after_fit_returns_empty_labels() -> None:
    assigner = TeamAssigner(clusterer=SequenceClusterer())
    assigner.fit([np.full((8, 8, 3), 255, dtype=np.uint8)])

    labels = assigner.predict([])

    assert labels.shape == (0,)


def test_assign_tracks_ignores_invalid_player_crop() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    tracks = TrackSet(
        (
            Track(1, BBox(5, 5, 5, 5), 2, "player", team_id=7),
            Track(2, BBox(0, 0, 10, 10), 2, "player"),
        )
    )
    assigner = TeamAssigner(clusterer=SequenceClusterer())
    assigner.fit([np.full((8, 8, 3), 255, dtype=np.uint8)])

    assigned = assigner.assign_tracks(Frame(index=0, image=image), tracks)

    assert assigned.tracks[0].team_id == 7
    assert assigned.tracks[1].team_id == 0


def test_assign_tracks_returns_original_track_set_when_no_valid_crops() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    tracks = TrackSet(
        (
            Track(1, BBox(5, 5, 5, 5), 2, "player", team_id=7),
            Track(2, BBox(1, 1, 2, 2), 0, "ball"),
        )
    )
    assigner = TeamAssigner(clusterer=SequenceClusterer())
    assigner.fit([np.full((8, 8, 3), 255, dtype=np.uint8)])

    assigned = assigner.assign_tracks(Frame(index=0, image=image), tracks)

    assert assigned is tracks
