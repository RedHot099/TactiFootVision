"""Multi-object tracking behind one interface (see :class:`Tracker`).

Importing this package registers the ``bytetrack`` and ``sam2`` backends.
"""

from tactifoot_vision.tracking.ball import clean_ball_path
from tactifoot_vision.tracking.base import (
    TRACKERS,
    Tracker,
    available_trackers,
    create_tracker,
)
from tactifoot_vision.tracking.bytetrack import ByteTrackTracker
from tactifoot_vision.tracking.sam2 import SAM2Tracker

__all__ = [
    "TRACKERS",
    "ByteTrackTracker",
    "SAM2Tracker",
    "Tracker",
    "available_trackers",
    "clean_ball_path",
    "create_tracker",
]
