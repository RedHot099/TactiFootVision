import numpy as np
from numpy.typing import NDArray

from tactifoot_vision.domain import Frame, PitchPoint, PitchProjection, TrackSet
from tactifoot_vision.keypoints import KeypointDetector, KeypointSet
from tactifoot_vision.projection.homography import HomographyEstimator, apply_homography


class PitchProjector:
    def __init__(
        self,
        *,
        keypoint_detector: KeypointDetector | None = None,
        estimator: HomographyEstimator | None = None,
        project_ball: bool = True,
    ) -> None:
        self.keypoint_detector = keypoint_detector
        self.estimator = estimator or HomographyEstimator()
        self.project_ball = project_ball

    def project(
        self, *, frame: Frame, keypoints: KeypointSet | None, tracks: TrackSet
    ) -> PitchProjection:
        if keypoints is None and self.keypoint_detector is not None:
            keypoints = self.keypoint_detector.predict(frame)
        if keypoints is None or len(keypoints) == 0:
            return PitchProjection.unavailable()
        homography = self.estimator.update(keypoints)
        if homography is None:
            return PitchProjection.unavailable()
        points_by_track_id: dict[int, PitchPoint] = {}
        ball: PitchPoint | None = None
        for track in tracks:
            x = (track.bbox.x1 + track.bbox.x2) / 2.0
            y = (
                (track.bbox.y1 + track.bbox.y2) / 2.0
                if track.class_name == "ball"
                else track.bbox.y2
            )
            projected = apply_homography_points([(x, y)], homography)[0]
            point = PitchPoint(x=float(projected[0]), y=float(projected[1]))
            if track.class_name == "ball":
                if self.project_ball:
                    ball = point
                continue
            else:
                points_by_track_id[track.track_id] = point
        return PitchProjection(
            status="available",
            points_by_track_id=points_by_track_id,
            ball=ball,
            homography=homography,
        )


def apply_homography_points(
    points: list[tuple[float, float]], homography: NDArray[np.float64]
) -> NDArray[np.float32]:
    return apply_homography(np.array(points, dtype=np.float32), homography)
