import numpy as np
from numpy.typing import NDArray

from tactifoot_vision.config import TeamAssignmentConfig
from tactifoot_vision.domain import Frame, Track, TrackSet
from tactifoot_vision.enums import (
    TeamAssignmentClusterer,
    TeamAssignmentEmbedding,
    TeamAssignmentReducer,
)
from tactifoot_vision.team_assignment.crops import crop_bbox

EMPTY_FIT_ERROR = "TeamAssigner requires at least one valid crop to fit."


class TeamAssigner:
    def __init__(
        self,
        *,
        embedder: object | None = None,
        reducer: object | None = None,
        clusterer: object | None = None,
        crop_ratio: float = 0.6,
    ) -> None:
        self.embedder = embedder or _default_embedder()
        self.reducer = reducer or _default_reducer()
        self.clusterer = clusterer or _default_clusterer()
        self.crop_ratio = crop_ratio
        self._fitted = False

    @classmethod
    def from_config(cls, config: TeamAssignmentConfig) -> "TeamAssigner":
        return cls(
            embedder=_embedder_from_config(config),
            reducer=_reducer_from_config(config),
            clusterer=_clusterer_from_config(config),
            crop_ratio=config.crop_ratio,
        )

    def fit(self, crops: list[NDArray[np.uint8]]) -> "TeamAssigner":
        if not crops:
            raise ValueError(EMPTY_FIT_ERROR)
        features = self.reducer.fit_transform(self.embedder.embed(crops))
        if len(features) == 0:
            raise ValueError(EMPTY_FIT_ERROR)
        labels = self.clusterer.fit_predict(features)
        _ensure_label_count(labels, len(crops))
        self._fitted = True
        return self

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit_predict(self, crops: list[NDArray[np.uint8]]) -> NDArray[np.int_]:
        if not crops:
            raise ValueError(EMPTY_FIT_ERROR)
        features = self.reducer.fit_transform(self.embedder.embed(crops))
        if len(features) == 0:
            raise ValueError(EMPTY_FIT_ERROR)
        labels = self.clusterer.fit_predict(features)
        _ensure_label_count(labels, len(crops))
        self._fitted = True
        return labels

    def predict(self, crops: list[NDArray[np.uint8]]) -> NDArray[np.int_]:
        if not self._fitted:
            raise RuntimeError("TeamAssigner must be fitted before predict().")
        if not crops:
            return np.empty((0,), dtype=np.int_)
        features = self.reducer.transform(self.embedder.embed(crops))
        labels = self.clusterer.predict(features)
        _ensure_label_count(labels, len(crops))
        return labels

    def assign_tracks(self, frame: Frame, tracks: TrackSet) -> TrackSet:
        eligible_tracks = [
            track for track in tracks if track.class_name in {"player", "goalkeeper"}
        ]
        crop_pairs = []
        for track in eligible_tracks:
            crop = crop_bbox(frame.image, track.bbox, ratio=self.crop_ratio)
            if _is_valid_crop(crop):
                crop_pairs.append((track, crop))
        if not crop_pairs:
            return tracks
        valid_tracks = [track for track, _crop in crop_pairs]
        crops = [crop for _track, crop in crop_pairs]
        labels = self.predict(crops)
        labels_by_track_id = {
            track.track_id: int(label)
            for track, label in zip(valid_tracks, labels, strict=True)
        }
        assigned = []
        for track in tracks:
            team_id = labels_by_track_id.get(track.track_id)
            assigned.append(
                Track(
                    track_id=track.track_id,
                    bbox=track.bbox,
                    class_id=track.class_id,
                    class_name=track.class_name,
                    confidence=track.confidence,
                    team_id=team_id if team_id is not None else track.team_id,
                    data=track.data,
                )
            )
        return TrackSet(tuple(assigned))


def _embedder_from_config(config: TeamAssignmentConfig) -> object:
    if config.embedding == TeamAssignmentEmbedding.RESNET:
        from tactifoot_vision.team_assignment.embeddings import ResNetEmbedder

        return ResNetEmbedder(device=config.device, batch_size=config.batch_size)
    if config.embedding == TeamAssignmentEmbedding.SIGLIP:
        from tactifoot_vision.team_assignment.embeddings import SigLIPEmbedder

        return SigLIPEmbedder(device=config.device, batch_size=config.batch_size)
    return _default_embedder()


def _reducer_from_config(config: TeamAssignmentConfig) -> object:
    if config.reducer == TeamAssignmentReducer.UMAP:
        from tactifoot_vision.team_assignment.reducers import UMAPReducer

        return UMAPReducer(random_state=config.random_state)
    return _default_reducer()


def _clusterer_from_config(config: TeamAssignmentConfig) -> object:
    if config.clusterer == TeamAssignmentClusterer.DBSCAN:
        from tactifoot_vision.team_assignment.clustering import DBSCANClusterer

        return DBSCANClusterer()
    if config.clusterer == TeamAssignmentClusterer.CMEANS:
        from tactifoot_vision.team_assignment.clustering import CMeansClusterer

        return CMeansClusterer()
    from tactifoot_vision.team_assignment.clustering import KMeansClusterer

    return KMeansClusterer(clusters=config.clusters, random_state=config.random_state)


def _default_embedder() -> object:
    from tactifoot_vision.team_assignment.embeddings import ColorHistogramEmbedder

    return ColorHistogramEmbedder()


def _default_reducer() -> object:
    from tactifoot_vision.team_assignment.reducers import IdentityReducer

    return IdentityReducer()


def _default_clusterer() -> object:
    from tactifoot_vision.team_assignment.clustering import KMeansClusterer

    return KMeansClusterer(clusters=2)


def _is_valid_crop(crop: NDArray[np.uint8]) -> bool:
    return crop.size > 0 and crop.ndim == 3 and crop.shape[2] == 3


def _ensure_label_count(labels: NDArray[np.int_], expected: int) -> None:
    if len(labels) != expected:
        raise RuntimeError(
            "TeamAssigner produced labels that do not match valid crops."
        )
