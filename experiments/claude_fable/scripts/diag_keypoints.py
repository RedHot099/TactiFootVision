import glob
import sys

import cv2
from ultralytics import YOLO

from xgpoc.features.freeze_frame import DETECTOR_WEIGHTS, KEYPOINT_WEIGHTS

kp = YOLO(str(KEYPOINT_WEIGHTS), task="pose")
det = YOLO(str(DETECTOR_WEIGHTS))
print(kp.names, det.names)
for f in sorted(glob.glob(sys.argv[1]))[::150][:4]:
    img = cv2.imread(f)
    r = kp.predict(img, verbose=False, conf=0.05)[0]
    c = r.keypoints.conf
    n = int((c[0] > 0.5).sum()) if c is not None and len(c) else None
    print(f[-10:], img.shape, "instances", len(r.boxes), "kp>0.5:", n, "det", len(det.predict(img, verbose=False)[0].boxes))
