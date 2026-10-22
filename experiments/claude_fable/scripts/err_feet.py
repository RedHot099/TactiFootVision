"""Shot-location variants from the cached retrained-detector frames (cache/xgd_full4/E3/frames.parquet), descriptive:
ball (E3 as is) vs the shooter's feet (the player whose image box is nearest the ball's image point; the ball of t0,
else -400 ms, else -800 ms). Writes results/err_decomp/feet.parquet for every cached shot."""

import numpy as np
import pandas as pd
from xgpoc.detector.conventions import align, attacks_right

FR = "cache/xgd_full4/E3/frames.parquet"


def rect_dist(boxes, pt):
    dx = np.maximum(np.maximum(boxes[:, 0] - pt[0], 0), pt[0] - boxes[:, 2])
    dy = np.maximum(np.maximum(boxes[:, 1] - pt[1], 0), pt[1] - boxes[:, 3])
    return np.hypot(dx, dy)


def main():
    fr = pd.read_parquet(
        FR,
        columns=[
            "shot_id",
            "offset_ms",
            "det_class",
            "det_xyxy",
            "det_pitch_xy",
            "ball_conf",
            "ball_xyxy",
        ],
    )
    by = {k: g.set_index("offset_ms") for k, g in fr.groupby("shot_id")}
    rows = []
    for sid, g in by.items():
        out = {"shot_id": sid}
        if 0 not in g.index:
            rows.append(out)
            continue
        t0 = g.loc[0]
        cls = np.asarray(t0.det_class)
        boxes = np.asarray(t0.det_xyxy, float).reshape(-1, 4)
        pxy = np.asarray(t0.det_pitch_xy, float).reshape(-1, 2)
        ball_pt, src = None, None
        for off in (0, -400, -800):
            if off in g.index and len(g.loc[off].ball_xyxy):
                b = np.asarray(g.loc[off].ball_xyxy, float).reshape(-1, 4)
                c = np.asarray(g.loc[off].ball_conf, float)
                b = b[np.argmax(c)]
                ball_pt, src = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]), off
                break
        players = np.flatnonzero((cls == "player") & np.isfinite(pxy).all(1))
        if ball_pt is not None and len(players):
            d = rect_dist(boxes[players], ball_pt)
            i = players[np.argmin(d)]
            feet = pxy[i]
            right = attacks_right(feet)
            a = align(feet, right)
            h = boxes[i, 3] - boxes[i, 1]
            out |= {
                "feet_x": a[0],
                "feet_y": a[1],
                "feet_src": src,
                "feet_box_dist_px": float(d.min()),
                "feet_box_h": float(h),
                "ball_above_feet_h": float((boxes[i, 3] - ball_pt[1]) / max(h, 1)),
            }
        rows.append(out)
    pd.DataFrame(rows).set_index("shot_id").to_parquet(
        "results/err_decomp/feet.parquet"
    )


if __name__ == "__main__":
    main()
