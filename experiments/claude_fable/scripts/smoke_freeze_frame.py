import sys
from pathlib import Path
import cv2
from xgpoc.features.freeze_frame import FreezeFrameExtractor

img = cv2.imread(sys.argv[1])
ff = FreezeFrameExtractor()(img)
print("homography", ff.homography_ok, "kp", ff.n_keypoints, "ball", ff.ball_xy, "players", len(ff.players_xy))
print("attack goal x", ff.attack_goal_x, "attacking team", ff.attacking_team)
print(ff.canonical().players_xy.round(1)[:8])
