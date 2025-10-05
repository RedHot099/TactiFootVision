from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

import warnings

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from torchvision.models import ResNet18_Weights, resnet18


_DEFAULT_KMEANS_ARGS: Dict[str, Any] = {
    "n_clusters": 2,
    "n_init": 10,
    "random_state": 0,
}

_SIGLIP_DEFAULTS: Dict[str, Any] = {
    "model_name": "google/siglip-base-patch16-224",
    "batch_size": 32,
    "pooling": "mean",
    "use_umap": True,
    "umap_components": 3,
    "umap_neighbors": 15,
    "umap_min_dist": 0.1,
    "umap_metric": "euclidean",
    "umap_random_state": None,
    "color_space": "rgb",
    "color_hist_bins": 0,
    "color_hist_weight": 0.0,
}


class BaseTeamClassifier(ABC):
    """Shared helpers for team-clustering backends."""

    def __init__(
        self,
        *,
        device: Optional[str] = None,
        kmeans_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._kmeans_kwargs = {**_DEFAULT_KMEANS_ARGS, **(kmeans_kwargs or {})}
        self._kmeans: Optional[KMeans] = None
        self._cluster_remap: Dict[int, int] = {}
        self._feature_dim: Optional[int] = None

    @property
    def is_fitted(self) -> bool:
        return self._kmeans is not None

    def fit(self, crops: List[np.ndarray]) -> None:
        features = self._extract_features(crops)
        min_required = self._min_samples_required()
        if features.shape[0] < min_required:
            raise ValueError(
                f"Need at least {min_required} crops to fit team classifier"
            )

        features = self._prepare_features_for_fit(features)
        kmeans = KMeans(**self._kmeans_kwargs)
        kmeans.fit(features)

        centers = kmeans.cluster_centers_
        labels = getattr(kmeans, "labels_", None)
        if kmeans.n_clusters <= 1:
            order = [0]
        elif labels is not None and labels.size > 0:
            counts = np.bincount(labels, minlength=kmeans.n_clusters)
            sorted_by_count = list(np.argsort(counts)[::-1])
            top_two = sorted_by_count[:2]
            top_two_sorted = sorted(top_two, key=lambda idx: centers[idx, 0])
            remaining = sorted_by_count[2:]
            order = [int(idx) for idx in top_two_sorted + remaining]
        else:
            order = sorted(range(kmeans.n_clusters), key=lambda idx: centers[idx, 0])

        self._cluster_remap = {int(orig): int(rank) for rank, orig in enumerate(order)}
        self._kmeans = kmeans
        if self._feature_dim is None:
            self._feature_dim = features.shape[1]

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("TeamClassifier must be fitted before calling predict")
        if not crops:
            return np.empty((0,), dtype=int)

        features = self._extract_features(crops)
        if features.size == 0:
            return np.empty((0,), dtype=int)

        features = self._prepare_features_for_predict(features)
        raw_preds = self._kmeans.predict(features)
        remapped = np.vectorize(lambda idx: self._cluster_remap.get(int(idx), -1))(
            raw_preds
        )
        return remapped.astype(int)

    def _min_samples_required(self) -> int:
        n_clusters = int(
            self._kmeans_kwargs.get("n_clusters", _DEFAULT_KMEANS_ARGS["n_clusters"])
        )
        return max(2, n_clusters)

    def _prepare_features_for_fit(self, features: np.ndarray) -> np.ndarray:
        if self._feature_dim is None and features.size:
            self._feature_dim = features.shape[1]
        return features

    def _prepare_features_for_predict(self, features: np.ndarray) -> np.ndarray:
        return features

    @abstractmethod
    def _extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """Subclasses must provide embedding extraction."""


class ResnetTeamClassifier(BaseTeamClassifier):
    """Team clustering powered by torchvision ResNet embeddings."""

    def __init__(
        self,
        *,
        device: Optional[str] = None,
        model_name: str = "resnet18",
        kmeans_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(device=device, kmeans_kwargs=kmeans_kwargs)
        self._init_resnet_backend(model_name)

    def _init_resnet_backend(self, model_name: str) -> None:
        normalized = (model_name or "").lower()
        if normalized != "resnet18":
            raise ValueError("Currently only resnet18 embeddings are supported")
        weights = ResNet18_Weights.DEFAULT
        backbone = resnet18(weights=weights)
        backbone.fc = torch.nn.Identity()
        self.model = backbone.to(self.device)
        self.model.eval()
        self.preprocess = weights.transforms()
        with torch.inference_mode():
            dummy = torch.zeros(1, 3, 224, 224, device=self.device)
            self._feature_dim = int(self.model(dummy).shape[1])

    def _extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        tensors = []
        for crop in crops:
            if crop is None or crop.size == 0:
                continue
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            tensor = self.preprocess(pil)
            tensors.append(tensor)
        if not tensors:
            return np.empty((0, self._feature_dim or 0), dtype=np.float32)
        batch = torch.stack(tensors).to(self.device)
        with torch.inference_mode():
            feats = self.model(batch)
        return feats.detach().cpu().numpy().astype(np.float32)


class SiglipTeamClassifier(BaseTeamClassifier):
    """Team clustering powered by SigLIP embeddings and optional UMAP."""

    def __init__(
        self,
        *,
        device: Optional[str] = None,
        model_name: str = "google/siglip-base-patch16-224",
        siglip_config: Optional[Dict[str, Any]] = None,
        kmeans_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        config = {**_SIGLIP_DEFAULTS, **(siglip_config or {})}
        if model_name:
            config["model_name"] = model_name
        super().__init__(device=device, kmeans_kwargs=kmeans_kwargs)
        self._init_siglip_backend(config)

    def _init_siglip_backend(self, config: Dict[str, Any]) -> None:
        try:
            from transformers import SiglipVisionModel
        except ImportError as exc:  # pragma: no cover - runtime guard
            raise ImportError(
                "SigLIP team classification requires the 'transformers' package."
            ) from exc

        processor_cls = None
        try:  # pragma: no cover - runtime guard
            from transformers import SiglipImageProcessor as _ProcessorCls

            processor_cls = _ProcessorCls
        except ImportError:
            try:
                from transformers import AutoImageProcessor as _ProcessorCls

                processor_cls = _ProcessorCls
            except ImportError as proc_exc:
                raise ImportError(
                    "SigLIP image preprocessing requires either 'SiglipImageProcessor' or 'AutoImageProcessor' from transformers."
                ) from proc_exc
        assert processor_cls is not None  # narrow type checkers

        self._siglip_config = config
        pooling = config.get("pooling", "mean").lower()
        if pooling not in {"mean", "cls"}:
            raise ValueError("SigLIP pooling must be either 'mean' or 'cls'")
        self._siglip_pooling = pooling
        self._siglip_batch_size = int(config.get("batch_size", 32))
        self._siglip_processor = processor_cls.from_pretrained(config["model_name"])
        self._siglip_model = SiglipVisionModel.from_pretrained(config["model_name"]).to(
            self.device
        )
        self._siglip_model.eval()

        hidden_size = getattr(self._siglip_model.config, "hidden_size", None)
        if hidden_size is None:
            vision_cfg = getattr(self._siglip_model.config, "vision_config", None)
            hidden_size = getattr(vision_cfg, "hidden_size", None)
        self._siglip_embedding_dim = int(hidden_size) if hidden_size else None

        color_space = config.get("color_space", "rgb").lower()
        if color_space not in {"rgb", "hsv"}:
            raise ValueError("color_space must be one of {'rgb', 'hsv'}")
        self._color_space = color_space
        self._color_hist_bins = int(config.get("color_hist_bins", 0))
        if self._color_hist_bins < 0 or self._color_hist_bins > 256:
            raise ValueError("color_hist_bins must be between 0 and 256")
        self._color_hist_weight = float(config.get("color_hist_weight", 0.0))
        self._color_feature_dim = (
            3 * self._color_hist_bins if self._color_hist_bins > 0 else 0
        )

        self._use_umap = bool(config.get("use_umap", True))
        self._umap_components = int(config.get("umap_components", 3))
        self._umap_neighbors = int(config.get("umap_neighbors", 15))
        self._umap_min_dist = float(config.get("umap_min_dist", 0.1))
        self._umap_metric = config.get("umap_metric", "euclidean")
        self._umap_random_state = config.get("umap_random_state", None)
        self._umap_model = None

        base_dim = (self._siglip_embedding_dim or 0) + self._color_feature_dim
        self._feature_dim = self._umap_components if self._use_umap else base_dim

    def _extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        valid_rgb: List[np.ndarray] = []
        hist_features: List[np.ndarray] = []
        for crop in crops:
            if crop is None or crop.size == 0:
                continue
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            valid_rgb.append(rgb)
            if self._color_feature_dim > 0:
                if self._color_space == "rgb":
                    color_source = rgb
                else:
                    color_source = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                hist_features.append(self._compute_color_hist(color_source))
        if not valid_rgb:
            expected_dim = (
                self._feature_dim
                if self._feature_dim is not None
                else (self._siglip_embedding_dim or 0) + self._color_feature_dim
            )
            return np.empty((0, expected_dim), dtype=np.float32)

        embeddings = self._run_siglip_inference(valid_rgb)
        if self._siglip_embedding_dim is None and embeddings.size:
            self._siglip_embedding_dim = embeddings.shape[1]

        features = embeddings
        if hist_features:
            hist_array = np.stack(hist_features).astype(np.float32)
            if self._color_hist_weight != 1.0:
                hist_array *= self._color_hist_weight
            features = np.concatenate([embeddings, hist_array], axis=1)

        if self._use_umap and self._umap_model is None:
            self._init_umap_model()
        if not self._use_umap and features.size:
            self._feature_dim = features.shape[1]
        return features.astype(np.float32)

    def _prepare_features_for_fit(self, features: np.ndarray) -> np.ndarray:
        if not self._use_umap:
            if self._feature_dim is None and features.size:
                self._feature_dim = features.shape[1]
            return features
        if self._umap_model is None:
            self._init_umap_model()
        transformed = self._umap_model.fit_transform(features)
        self._feature_dim = transformed.shape[1]
        return transformed

    def _prepare_features_for_predict(self, features: np.ndarray) -> np.ndarray:
        if self._use_umap and self._umap_model is not None:
            return self._umap_model.transform(features)
        return features

    def _run_siglip_inference(self, images_rgb: List[np.ndarray]) -> np.ndarray:
        batches: List[np.ndarray] = []
        pil_images = [Image.fromarray(img) for img in images_rgb]
        for start in range(0, len(pil_images), self._siglip_batch_size):
            batch_imgs = pil_images[start : start + self._siglip_batch_size]
            inputs = self._siglip_processor(images=batch_imgs, return_tensors="pt").to(
                self.device
            )
            with torch.inference_mode():
                outputs = self._siglip_model(**inputs)
                if self._siglip_pooling == "cls":
                    embeddings = outputs.last_hidden_state[:, 0, :]
                else:
                    embeddings = outputs.last_hidden_state.mean(dim=1)
            batches.append(embeddings.detach().cpu().numpy())
        if not batches:
            expected_dim = self._siglip_embedding_dim or 0
            return np.empty((0, expected_dim), dtype=np.float32)
        return np.concatenate(batches, axis=0).astype(np.float32)

    def _compute_color_hist(self, image: np.ndarray) -> np.ndarray:
        hist_parts: List[np.ndarray] = []
        for channel in cv2.split(image):
            hist = cv2.calcHist([channel], [0], None, [self._color_hist_bins], [0, 256])
            hist = cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
            hist_parts.append(hist.flatten())
        return np.concatenate(hist_parts).astype(np.float32)

    def _init_umap_model(self) -> None:
        if not self._use_umap:
            self._umap_model = None
            return
        try:
            import umap
        except ImportError as exc:  # pragma: no cover - runtime guard
            raise ImportError(
                "SigLIP team classification with dimensionality reduction requires 'umap-learn'."
            ) from exc

        if self._umap_random_state is not None:
            warnings.filterwarnings(
                "ignore",
                message="n_jobs value .* overridden to 1 by setting random_state",
                category=UserWarning,
            )

        self._umap_model = umap.UMAP(
            n_components=self._umap_components,
            # n_neighbors=self._umap_neighbors,
            # min_dist=self._umap_min_dist,
            # metric=self._umap_metric,
            # random_state=self._umap_random_state,
        )


class TeamClassifier:
    """Backwards-compatible facade that selects the appropriate backend."""

    def __init__(
        self,
        device: Optional[str] = None,
        model_name: str = "resnet18",
        *,
        method: Optional[str] = None,
        siglip_config: Optional[Dict[str, Any]] = None,
        kmeans_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        resolved = self._resolve_method(method, model_name, siglip_config)
        backend_kwargs = {
            "device": device,
            "kmeans_kwargs": kmeans_kwargs,
        }
        if resolved == "siglip":
            self._backend: BaseTeamClassifier = SiglipTeamClassifier(
                model_name=model_name,
                siglip_config=siglip_config,
                **backend_kwargs,
            )
        else:
            self._backend = ResnetTeamClassifier(
                model_name=model_name,
                **backend_kwargs,
            )

    @property
    def is_fitted(self) -> bool:
        return self._backend.is_fitted

    def fit(self, crops: List[np.ndarray]) -> None:
        self._backend.fit(crops)

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        return self._backend.predict(crops)

    @property
    def backend(self) -> BaseTeamClassifier:
        return self._backend

    @staticmethod
    def _resolve_method(
        method: Optional[str],
        model_name: str,
        siglip_config: Optional[Dict[str, Any]],
    ) -> str:
        if method:
            return method.lower()
        if siglip_config is not None:
            return "siglip"
        if "siglip" in (model_name or "").lower():
            return "siglip"
        return "resnet"


class TeamAssignmentManager:
    """Stabilises per-tracker team assignments requiring consecutive agreement."""

    def __init__(self, consecutive_frames: int = 3):
        self._required = max(1, consecutive_frames)
        self._streak: Dict[int, int] = defaultdict(int)
        self._last: Dict[int, int] = {}
        self._assignment: Dict[int, int] = {}

    def update(self, tracker_ids: Iterable[int], predictions: Iterable[int]) -> List[Optional[int]]:
        results: List[Optional[int]] = []
        for tid, pred in zip(tracker_ids, predictions):
            if tid in self._assignment:
                results.append(self._assignment[tid])
                continue
            if pred is None:
                results.append(None)
                continue
            pred = int(pred)
            if self._last.get(tid) == pred:
                self._streak[tid] += 1
            else:
                self._streak[tid] = 1
                self._last[tid] = pred
            if self._streak[tid] >= self._required:
                self._assignment[tid] = pred
            results.append(self._assignment.get(tid))
        return results

    def prune(self, active_ids: Iterable[int]) -> None:
        active_set = {int(tid) for tid in active_ids}
        for store in (self._assignment, self._streak, self._last):
            for tid in list(store.keys()):
                if tid not in active_set:
                    store.pop(tid, None)

    def get_assignment(self, tracker_id: int) -> Optional[int]:
        return self._assignment.get(int(tracker_id))
