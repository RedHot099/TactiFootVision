from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tactifoot_vision.domain import BBox


@dataclass(frozen=True, slots=True)
class CropSample:
    frame_index: int
    track_id: int
    class_name: str
    bbox: BBox
    image: NDArray[np.uint8]
    team_label: int | None = None


@dataclass(frozen=True, slots=True)
class TeamAssignmentReport:
    samples: int
    clusters: int
    metrics: dict[str, float]
