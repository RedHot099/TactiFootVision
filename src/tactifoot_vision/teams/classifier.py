"""Unsupervised team classification: embed crops, reduce, cluster with k-means."""

import logging
from collections.abc import Sequence
from typing import Any, Literal, Self

import numpy as np

from tactifoot_vision.teams.embedders import EMBEDDERS, Embedder

logger = logging.getLogger(__name__)

# Below this many crops UMAP fails or returns unstable layouts; cluster directly.
MIN_UMAP_SAMPLES = 30


class TeamClassifier:
    """Splits player crops into ``n_teams`` teams without labels.

    Crops are embedded (``"siglip"``, ``"resnet"`` or any :class:`Embedder`),
    optionally reduced with UMAP, and clustered with k-means. Team numbers are
    deterministic for the same data and ``seed``: the two largest clusters are
    teams 0 and 1 (ordered by the first coordinate of their centres), further
    clusters follow by size.

    >>> classifier = TeamClassifier("siglip").fit(player_crops)
    >>> teams = classifier.predict(crops)

    Args:
        embedder: registry name or an :class:`Embedder` instance.
        n_teams: number of clusters.
        reducer: ``"umap"`` or ``None`` (cluster the raw embeddings).
        umap_components, umap_neighbors, umap_min_dist: UMAP settings.
        seed: random state of UMAP, k-means and the fit subsample.
        max_fit_samples: fit on at most this many embeddings (a seeded random
            subset), which keeps UMAP fast on full matches; ``None`` uses all.
            At least ``n_teams``.
        device: torch device for an embedder created by name.
        **embedder_options: passed to an embedder created by name, e.g.
            ``color_hist_bins=16`` for SigLIP.
    """

    def __init__(
        self,
        embedder: str | Embedder = "siglip",
        n_teams: int = 2,
        reducer: Literal["umap"] | None = "umap",
        umap_components: int = 3,
        umap_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        seed: int = 0,
        max_fit_samples: int | None = 5000,
        device: str | None = None,
        **embedder_options: Any,
    ) -> None:
        if n_teams < 2:
            raise ValueError("n_teams must be >= 2")
        if reducer not in ("umap", None):
            raise ValueError(f"reducer must be 'umap' or None, got {reducer!r}")
        if max_fit_samples is not None and max_fit_samples < n_teams:
            raise ValueError(
                f"max_fit_samples={max_fit_samples} cannot fit n_teams={n_teams} "
                "clusters; use at least n_teams, or None for all embeddings"
            )
        if isinstance(embedder, str):
            embedder = EMBEDDERS.create(embedder, device=device, **embedder_options)
        elif embedder_options or device is not None:
            raise ValueError(
                "device and embedder options only apply to an embedder given by name"
            )
        self.embedder: Embedder = embedder
        self.n_teams = n_teams
        self.reducer = reducer
        self.umap_components = umap_components
        self.umap_neighbors = umap_neighbors
        self.umap_min_dist = umap_min_dist
        self.seed = seed
        self.max_fit_samples = max_fit_samples
        self._umap: Any = None
        self._kmeans: Any = None
        self._team_of_cluster: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        return self._kmeans is not None

    # ------------------------------------------------------------------ crops
    def embed(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """``(N, D)`` embeddings of ``N`` BGR crops (see :meth:`fit_embeddings`)."""
        return self.embedder.embed(list(crops))

    def fit(self, crops: Sequence[np.ndarray]) -> Self:
        """Learn the teams from player crops (at least ``n_teams`` of them)."""
        return self.fit_embeddings(self.embed(crops))

    def predict(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Team index ``0..n_teams-1`` per crop."""
        return self.predict_embeddings(self.embed(crops))

    def fit_predict(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        embeddings = self.embed(crops)
        return self.fit_embeddings(embeddings).predict_embeddings(embeddings)

    # ------------------------------------------------------------- embeddings
    def fit_embeddings(self, embeddings: np.ndarray) -> Self:
        """Like :meth:`fit` on precomputed :meth:`embed` output (lets callers embed once)."""
        from sklearn.cluster import KMeans

        features = np.asarray(embeddings, dtype=np.float32)
        if features.ndim != 2 or len(features) < self.n_teams:
            raise ValueError(
                f"Need at least {self.n_teams} embeddings to fit, got shape {features.shape}"
            )
        if self.max_fit_samples is not None and len(features) > self.max_fit_samples:
            rng = np.random.default_rng(self.seed)
            features = features[
                rng.choice(len(features), self.max_fit_samples, replace=False)
            ]
        self._umap = None
        if self.reducer == "umap":
            if len(features) >= MIN_UMAP_SAMPLES:
                features = self._fit_umap(features)
            else:
                logger.info("Only %d crops: clustering without UMAP", len(features))
        kmeans = KMeans(n_clusters=self.n_teams, n_init=10, random_state=self.seed)
        labels = kmeans.fit_predict(features)
        by_size = np.argsort(
            -np.bincount(labels, minlength=self.n_teams), kind="stable"
        )
        teams = sorted(by_size[:2], key=lambda c: kmeans.cluster_centers_[c, 0])
        order = [*teams, *by_size[2:]]
        self._team_of_cluster = np.empty(self.n_teams, dtype=int)
        self._team_of_cluster[order] = np.arange(self.n_teams)
        self._kmeans = kmeans
        logger.info(
            "Fitted %d teams on %d crops (sizes %s)",
            self.n_teams,
            len(features),
            np.bincount(self._team_of_cluster[labels], minlength=self.n_teams).tolist(),
        )
        return self

    def predict_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Like :meth:`predict` on precomputed :meth:`embed` output."""
        if self._kmeans is None or self._team_of_cluster is None:
            raise RuntimeError("TeamClassifier is not fitted; call fit() first")
        features = np.asarray(embeddings, dtype=np.float32)
        if len(features) == 0:
            return np.zeros(0, dtype=int)
        if self._umap is not None:
            features = self._umap.transform(features)
        return self._team_of_cluster[self._kmeans.predict(features)]

    def _fit_umap(self, features: np.ndarray) -> np.ndarray:
        import umap

        self._umap = umap.UMAP(
            n_components=self.umap_components,
            n_neighbors=min(self.umap_neighbors, len(features) - 1),
            min_dist=self.umap_min_dist,
            random_state=self.seed,
            n_jobs=1,  # implied by random_state; set to silence UMAP's warning
        )
        return self._umap.fit_transform(features)

    def __repr__(self) -> str:
        return (
            f"TeamClassifier(embedder={type(self.embedder).__name__}, n_teams={self.n_teams}, "
            f"reducer={self.reducer!r}, fitted={self.is_fitted})"
        )
