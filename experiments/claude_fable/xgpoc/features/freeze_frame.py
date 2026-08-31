"""Single-frame positional extraction: image -> players/ball in pitch metres.

Reuses the repo's trained YOLO detector (ball/goalkeeper/player/referee) and the
YOLO-pose pitch keypoint model (32 landmarks) via ultralytics directly, so this
experiment stays independent of the package config machinery.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from tactifoot_vision.geometry.pitch_definitions import SoccerPitchConfiguration

REPO = Path(os.environ.get("TACTIFOOT_REPO", "/home/kuba/projects/tactifoot_vision"))
DETECTOR_WEIGHTS = REPO / "results/project/raw/yolo11m/training/detect_yolo11m/weights/best.pt"
KEYPOINT_WEIGHTS = REPO / "results/project/raw/yolo_pose/training/kp_yolo_pose/weights/best.pt"
PITCH_LENGTH, PITCH_WIDTH = 105.0, 68.0
CLS_BALL, CLS_GK, CLS_PLAYER, CLS_REFEREE = 0, 1, 2, 3


@dataclass
class FreezeFrame:
    ball_xy: np.ndarray | None          # (2,) metres or None
    players_xy: np.ndarray              # (N, 2) metres
    players_cls: np.ndarray             # (N,) 1=goalkeeper 2=player
    players_team: np.ndarray            # (N,) 0/1 from colour clustering, -1 unknown
    attacking_team: int                 # team id of shooter (nearest outfield player to ball), -1 unknown
    attack_goal_x: float                # 0.0 or PITCH_LENGTH: goal being attacked
    n_keypoints: int
    homography_ok: bool
    boxes_xyxy: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    boxes_cls: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=int))

    def attacking_right(self) -> bool:
        return self.attack_goal_x > PITCH_LENGTH / 2

    def canonical(self) -> "FreezeFrame":
        """Mirror so the attacked goal is always at x = PITCH_LENGTH."""
        if self.attacking_right():
            return self
        def flip(a):
            return None if a is None else np.column_stack([PITCH_LENGTH - a[..., 0], a[..., 1]])

        return FreezeFrame(flip(self.ball_xy[None])[0] if self.ball_xy is not None else None, flip(self.players_xy),
                           self.players_cls, self.players_team, self.attacking_team, PITCH_LENGTH,
                           self.n_keypoints, self.homography_ok, self.boxes_xyxy, self.boxes_cls)


class FreezeFrameExtractor:
    def __init__(self, detector_weights=DETECTOR_WEIGHTS, keypoint_weights=KEYPOINT_WEIGHTS,
                 det_conf=0.3, kp_conf=0.5, device=None, det_imgsz=1920):
        self.detector = YOLO(str(detector_weights))
        self.keypoints = YOLO(str(keypoint_weights), task="pose")
        self.det_conf, self.kp_conf, self.device, self.det_imgsz = det_conf, kp_conf, device, det_imgsz
        cfg = SoccerPitchConfiguration(length=PITCH_LENGTH, width=PITCH_WIDTH)
        self.pitch_vertices = np.array(cfg.vertices, dtype=np.float32)

    def homography(self, img: np.ndarray) -> tuple[np.ndarray | None, int]:
        # The keypoint model was trained on frames stretched to 640x640 (Roboflow export),
        # so we stretch instead of letterboxing and scale the result back.
        h, w = img.shape[:2]
        small = cv2.resize(img, (640, 640))
        r = self.keypoints.predict(small, verbose=False, device=self.device, conf=0.1, imgsz=640)[0]
        if r.keypoints is None or r.keypoints.xy.shape[0] == 0:
            return None, 0
        xy = r.keypoints.xy[0].cpu().numpy() * np.array([w / 640, h / 640], np.float32)
        conf = r.keypoints.conf[0].cpu().numpy() if r.keypoints.conf is not None else np.ones(len(xy))
        m = conf >= self.kp_conf
        if m.sum() < 4:
            return None, int(m.sum())
        H, _ = cv2.findHomography(xy[m], self.pitch_vertices[m], cv2.RANSAC, 10.0)
        return H, int(m.sum())

    def detect(self, img: np.ndarray):
        # Large inference size: the ball is only a few pixels at the default 640.
        # Per-class thresholds: the tiny ball never reaches player-level confidence.
        r = self.detector.predict(img, verbose=False, device=self.device, conf=0.1, imgsz=self.det_imgsz)[0]
        boxes, cls, conf = r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()
        keep = np.where(cls == CLS_BALL, conf >= min(0.15, self.det_conf), conf >= self.det_conf)
        return boxes[keep], cls[keep], conf[keep]

    def __call__(self, img: np.ndarray) -> FreezeFrame:
        H, n_kp = self.homography(img)
        boxes, cls, conf = self.detect(img)
        empty = FreezeFrame(None, np.zeros((0, 2)), np.zeros(0, int), np.zeros(0, int), -1, PITCH_LENGTH, n_kp, False, boxes, cls)
        if H is None:
            return empty
        feet = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]]).astype(np.float32)
        pitch = cv2.perspectiveTransform(feet.reshape(-1, 1, 2), H).reshape(-1, 2) if len(feet) else np.zeros((0, 2))

        ball_idx = np.where(cls == CLS_BALL)[0]
        ball_xy = None
        if len(ball_idx):
            b = ball_idx[np.argmax(conf[ball_idx])]
            c = np.array([[(boxes[b, 0] + boxes[b, 2]) / 2, (boxes[b, 1] + boxes[b, 3]) / 2]], np.float32)
            ball_xy = cv2.perspectiveTransform(c.reshape(-1, 1, 2), H).reshape(2)

        pm = np.isin(cls, [CLS_GK, CLS_PLAYER])
        players_xy, players_cls = pitch[pm], cls[pm]
        teams = self._cluster_teams(img, boxes[pm], players_cls)

        # Attacked goal: the one nearer the ball (fallback: nearer to player centroid).
        ref = ball_xy if ball_xy is not None else (players_xy.mean(0) if len(players_xy) else np.array([PITCH_LENGTH, 0]))
        attack_goal_x = PITCH_LENGTH if ref[0] > PITCH_LENGTH / 2 else 0.0
        attacking_team = -1
        outfield = np.where((players_cls == CLS_PLAYER) & (teams >= 0))[0]
        if ball_xy is not None and len(outfield):
            attacking_team = int(teams[outfield[np.argmin(np.linalg.norm(players_xy[outfield] - ball_xy, axis=1))]])
        return FreezeFrame(ball_xy, players_xy, players_cls, teams, attacking_team, attack_goal_x, n_kp, True, boxes, cls)

    @staticmethod
    def _cluster_teams(img, boxes, cls) -> np.ndarray:
        """2-means on torso colour histograms of outfield players; goalkeepers get -1."""
        teams = np.full(len(boxes), -1, int)
        idx = np.where(cls == CLS_PLAYER)[0]
        if len(idx) < 2:
            return teams
        feats = []
        for i in idx:
            x1, y1, x2, y2 = boxes[i].astype(int)
            h = y2 - y1
            crop = img[y1 + int(0.15 * h): y1 + int(0.55 * h), x1:x2]
            if crop.size == 0:
                feats.append(np.zeros(48))
                continue
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [16, 3], [0, 180, 0, 256]).flatten()
            feats.append(hist / (hist.sum() + 1e-6))
        feats = np.array(feats, np.float32)
        _, labels, _ = cv2.kmeans(feats, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4), 5, cv2.KMEANS_PP_CENTERS)
        teams[idx] = labels.flatten()
        return teams
