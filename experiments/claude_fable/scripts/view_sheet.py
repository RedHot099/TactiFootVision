"""Render a contact sheet of shooter/keeper crops for a visual check of the heuristic.

Diagnostic only. Each tile shows the t0 frame with the detected boxes (ball green,
players white, keeper cyan), the chosen shooter crop (yellow) and keeper crop (magenta),
next to the two crops at LOCAL_SIZE. Reports coverage of the views over the whole index.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from xgpoc.paths import INDEX
from xgpoc.v13.inputs import PRIMARY_POSITIONAL, normalized_crop_pixels, unique_map
from xgpoc.v14.views import LOCAL_SIZE, view_boxes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--out", type=Path, default=Path("cache/v14/view_sheet.jpg"))
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    index = pd.read_parquet(args.index)
    index = index[index.split != "test"]
    with np.load(PRIMARY_POSITIONAL / "tokens.npz") as data:
        boxes = unique_map(data["shot_id"].astype(str), data["boxes"], set(index.shot_id))
    stats = pd.DataFrame([{"shot_id": s, **dict(zip(("shooter", "keeper"), view_boxes(boxes.get(s)).valid)),
                           "distance": view_boxes(boxes.get(s)).shooter_distance} for s in index.shot_id])
    coverage = {"shots": int(len(stats)), "shooter_view": float(stats.shooter.mean()),
                "keeper_view": float(stats.keeper.mean()), "both": float((stats.shooter * stats.keeper).mean()),
                "shooter_distance_quantiles": stats.distance.dropna().quantile([.5, .9, .95, .99]).round(3).to_dict()}
    print(json.dumps(coverage, indent=2))
    sample = stats[stats.shooter > 0].sample(args.n, random_state=args.seed)
    rows = []
    for shot_id in sample.shot_id:
        row = index[index.shot_id == shot_id].iloc[0]
        raw = cv2.imread(str(row.frame_path))
        h, w = raw.shape[:2]
        vb = view_boxes(boxes[shot_id])
        canvas = raw.copy()
        for b in boxes[shot_id]:
            if b[8] <= 0:
                continue
            colour = (0, 255, 0) if b[4] else (255, 255, 0) if b[5] else (255, 255, 255)
            cv2.rectangle(canvas, (int(b[0] * w), int(b[1] * h)), (int(b[2] * w), int(b[3] * h)), colour, 2)
        tiles = []
        for box, colour in zip(vb.boxes, ((0, 255, 255), (255, 0, 255))):
            if box is None:
                tiles.append(np.zeros((LOCAL_SIZE[1] * 2, LOCAL_SIZE[0] * 2, 3), np.uint8))
                continue
            x0, y0, x1, y1 = normalized_crop_pixels(box, w, h)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, 3)
            tiles.append(cv2.resize(raw[y0:y1, x0:x1], (LOCAL_SIZE[0] * 2, LOCAL_SIZE[1] * 2)))
        full = cv2.resize(canvas, (448 * 2, 252 * 2))
        cv2.putText(full, f"{shot_id[-30:]} goal={row.goal} d={vb.shooter_distance:.3f}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
        rows.append(np.hstack([full, np.vstack(tiles)]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
    (args.out.with_suffix(".json")).write_text(json.dumps(coverage, indent=2) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
