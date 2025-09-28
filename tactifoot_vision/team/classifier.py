from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from torchvision.models import ResNet18_Weights, resnet18


class TeamClassifier:
    """Clusters player crops into two visual groups using image embeddings."""

    def __init__(self, device: Optional[str] = None, model_name: str = "resnet18"):
        if model_name.lower() != "resnet18":
            raise ValueError("Currently only resnet18 embeddings are supported")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        weights = ResNet18_Weights.DEFAULT
        backbone = resnet18(weights=weights)
        backbone.fc = torch.nn.Identity()
        self.model = backbone.to(self.device)
        self.model.eval()
        self.preprocess = weights.transforms()
        self._kmeans: Optional[KMeans] = None
        self._cluster_remap: Dict[int, int] = {}
        with torch.inference_mode():
            dummy = torch.zeros(1, 3, 224, 224, device=self.device)
            self._feature_dim = int(self.model(dummy).shape[1])

    @property
    def is_fitted(self) -> bool:
        return self._kmeans is not None

    def fit(self, crops: List[np.ndarray]) -> None:
        features = self._extract_features(crops)
        if features.shape[0] < 2:
            raise ValueError("Need at least two crops to fit team classifier")
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)
        kmeans.fit(features)
        order = np.argsort(kmeans.cluster_centers_[:, 0])
        self._cluster_remap = {int(orig): int(rank) for rank, orig in enumerate(order)}
        self._kmeans = kmeans

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("TeamClassifier must be fitted before calling predict")
        if not crops:
            return np.empty((0,), dtype=int)
        features = self._extract_features(crops)
        if features.size == 0:
            return np.empty((0,), dtype=int)
        raw_preds = self._kmeans.predict(features)
        remapped = np.vectorize(self._cluster_remap.get)(raw_preds)
        return remapped.astype(int)

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
            return np.empty((0, self._feature_dim))
        batch = torch.stack(tensors).to(self.device)
        with torch.inference_mode():
            feats = self.model(batch)
        return feats.cpu().numpy()


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
