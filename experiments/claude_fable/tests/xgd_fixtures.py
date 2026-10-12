"""Planted shot frames for the detector tests: package `FrameResult`s with known pitch positions and teams."""

import numpy as np
import supervision as sv

from tactifoot_vision.pipeline import FrameResult


def frame_with(players, ball=None, *, homography=None, fallback=None, keypoint_conf=None) -> FrameResult:
    """`players`: (x, y, team, class) in pitch metres; `ball`: (x, y) or None; `fallback`: the ball's source index."""
    n = len(players)
    xy = np.array([(x, y) for x, y, _, _ in players], dtype=float).reshape(-1, 2)
    boxes = np.column_stack([xy[:, 0] - 0.5, xy[:, 1] - 2, xy[:, 0] + 0.5, xy[:, 1]]).astype(np.float32) if n else np.zeros((0, 4), np.float32)
    people = sv.Detections(xyxy=boxes, confidence=np.full(n, 0.9, np.float32),
                           data={"class_name": np.array([c for *_, c in players], dtype=str).reshape(n),
                                 "pitch_xy": xy, "team_id": np.array([t for _, _, t, _ in players], dtype=int).reshape(n)})
    if ball is None:
        balls = sv.Detections.empty()
        balls.data["pitch_xy"] = np.zeros((0, 2))
    else:
        balls = sv.Detections(xyxy=np.float32([[0, 0, 1, 1]]), confidence=np.float32([0.5]),
                              data={"class_name": np.array(["ball"]), "pitch_xy": np.array([ball], dtype=float)})
        if fallback is not None:
            balls.data["fallback"] = np.array([fallback])
    keypoints = None
    if keypoint_conf is not None:
        keypoints = sv.KeyPoints(xy=np.zeros((1, 32, 2), np.float32), confidence=np.asarray(keypoint_conf, np.float32)[None])
    return FrameResult(index=0, timestamp=0.0, detections=people, ball=balls, keypoints=keypoints,
                       homography=np.eye(3) if homography is None else homography)
