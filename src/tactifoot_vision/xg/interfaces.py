from typing import Protocol

from tactifoot_vision.shots import ShotCandidate
from tactifoot_vision.xg.results import XgPrediction, XgShotFeatures


class XgEstimator(Protocol):
    def predict(
        self, features: XgShotFeatures, candidate: ShotCandidate
    ) -> XgPrediction:
        raise NotImplementedError
