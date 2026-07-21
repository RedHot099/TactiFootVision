from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from tactifoot_vision.domain import AdapterUnavailable, BBox, ModelArtifactNotFound


class SAM2Cropper:
    def __init__(
        self,
        *,
        config_path: Path,
        checkpoint_path: Path,
        predictor_factory: Callable[[], Any] | None = None,
    ) -> None:
        if not config_path.is_file():
            raise ModelArtifactNotFound(f"SAM2 crop config not found: {config_path}")
        if not checkpoint_path.is_file():
            raise ModelArtifactNotFound(
                f"SAM2 crop checkpoint not found: {checkpoint_path}"
            )
        if predictor_factory is None:
            raise AdapterUnavailable(
                "SAM2 cropper requires a configured predictor factory in this phase."
            )
        self._predictor = predictor_factory()

    def prepare_frame(self, frame: NDArray[np.uint8]) -> None:
        self._predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def extract_crop(
        self, frame: NDArray[np.uint8], bbox: BBox
    ) -> NDArray[np.uint8] | None:
        masks, _, _ = self._predictor.predict(
            box=np.array([[bbox.x1, bbox.y1, bbox.x2, bbox.y2]], dtype=np.float32),
            multimask_output=False,
            normalize_coords=False,
        )
        if masks is None or np.asarray(masks).size == 0:
            return None
        mask = np.asarray(masks)[0]
        if mask.ndim == 3:
            mask = mask[0]
        mask_bool = mask.astype(bool)
        if not mask_bool.any():
            return None
        ys, xs = np.where(mask_bool)
        top, bottom = int(ys.min()), int(ys.max()) + 1
        left, right = int(xs.min()), int(xs.max()) + 1
        crop = frame[top:bottom, left:right].copy()
        crop[~mask_bool[top:bottom, left:right]] = 0
        return crop
