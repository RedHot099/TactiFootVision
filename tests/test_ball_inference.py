"""ball-detection 03: tile geometry, the ball merge and the post-filter (scripts/ball_inference.py), on synthetic
frames."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ball_inference as bi  # noqa: E402

CLASSES = np.array(["ball", "goalkeeper", "player", "referee"])


def detections(xyxy, conf, classes) -> sv.Detections:
    ids = np.array([list(CLASSES).index(c) for c in classes], int)
    return sv.Detections(
        xyxy=np.asarray(xyxy, np.float32).reshape(-1, 4),
        confidence=np.asarray(conf, np.float32),
        class_id=ids,
        data={"class_name": CLASSES[ids]},
    )


class BallFinder:
    """A fake detector that sees every ball of a synthetic frame (white squares on black) in whatever image it gets,
    clipped to the image as YOLO clips its boxes, with a confidence that falls with the image size (tiles see the
    ball more surely than the full frame)."""

    def __init__(self, balls: np.ndarray, person: np.ndarray | None = None):
        self.balls, self.person = balls, person

    def predict(self, image: np.ndarray) -> sv.Detections:
        mask = image[..., 0] > 0
        boxes, conf, classes = [], [], []
        if mask.any():
            ys, xs = np.nonzero(mask)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
            conf.append(1000 / max(image.shape[:2]))
            classes.append("ball")
        if self.person is not None and image.shape[1] == 1920:
            boxes.append(self.person)
            conf.append(0.9)
            classes.append("player")
        return detections(np.array(boxes).reshape(-1, 4), conf, classes)


def frame_with_ball(box) -> np.ndarray:
    image = np.zeros((1080, 1920, 3), np.uint8)
    x1, y1, x2, y2 = box
    image[y1:y2, x1:x2] = 255
    return image


def tiled(finder: BallFinder, columns: int, rows: int) -> bi.Tiled:
    tiles = bi.Tiled.__new__(bi.Tiled)
    tiles.columns, tiles.rows = columns, rows
    tiles.full = finder
    tiles.augment, tiles.device = False, None
    tiles.models = {
        bi.tile_grid(1920, 1080, columns, rows)[1]: finder,
    }
    return tiles


def test_tile_grid_is_the_registered_geometry_and_covers_the_frame():
    tiles, imgsz = bi.tile_grid(1920, 1080, 2, 2)
    assert imgsz == 1024
    assert tiles == [
        (0, 0, 1024, 608),
        (896, 0, 1920, 608),
        (0, 472, 1024, 1080),
        (896, 472, 1920, 1080),
    ]
    tiles, imgsz = bi.tile_grid(1920, 1080, 3, 2)
    assert imgsz == 704
    assert [t[0] for t in tiles[:3]] == [0, 608, 1216]  # overlap 96 px
    assert bi.tile_grid(1280, 720, 2, 2) == (
        [
            (0, 0, 672, 416),
            (608, 0, 1280, 416),
            (0, 304, 672, 720),
            (608, 304, 1280, 720),
        ],
        672,
    )
    assert bi.tile_grid(1280, 720, 3, 2)[1] == 480
    for width, height in ((1920, 1080), (1280, 720)):
        for columns, rows in bi.TILES.values():
            tiles, imgsz = bi.tile_grid(width, height, columns, rows)
            covered = np.zeros((height, width), bool)
            for x0, y0, x1, y1 in tiles:
                assert x1 - x0 <= imgsz and y1 - y0 <= imgsz
                assert (x1 - x0) % bi.TILE_STRIDE == 0
                covered[y0:y1, x0:x1] = True
            assert covered.all()
            starts = sorted({t[0] for t in tiles})
            ends = sorted({t[2] for t in tiles})
            overlap = min(e - s for s, e in zip(starts[1:], ends[:-1], strict=True))
            assert overlap >= 2 * 43 * height / 1080


def test_ball_on_a_tile_boundary_is_detected_once_with_frame_coordinates():
    # a 20 px ball across the right edge of tile 0 (x 1024) and the bottom edge of the top row (y 608) of 2 x 2
    box = (1014, 600, 1034, 620)
    finder = BallFinder(np.array([box]), person=[100, 100, 140, 200])
    found = tiled(finder, 2, 2)(frame_with_ball(box))
    balls = found[found.data["class_name"] == "ball"]
    assert len(balls) == 1
    np.testing.assert_array_equal(balls.xyxy[0], box)
    # the tile's view (larger confidence) won over the full frame's
    assert balls.confidence[0] > 1000 / 1920
    people = found[found.data["class_name"] != "ball"]
    np.testing.assert_array_equal(people.xyxy, [[100, 100, 140, 200]])


def test_ball_in_the_overlap_inside_two_tiles_is_merged_and_shifted():
    box = (
        950,
        300,
        962,
        312,
    )  # inside tiles 0 and 1 of 2 x 2, away from every inner edge
    finder = BallFinder(np.array([box]))
    found = tiled(finder, 2, 2)(frame_with_ball(box))
    assert len(found) == 1
    np.testing.assert_array_equal(found.xyxy[0], box)
    found = tiled(finder, 3, 2)(frame_with_ball(box))  # tile columns 0-704, 608-1312
    assert len(found) == 1
    np.testing.assert_array_equal(found.xyxy[0], box)


def test_inner_edge_cut_ignores_the_image_border():
    tile = (896, 0, 1920, 608)
    boxes = np.array(
        [
            [0, 10, 10, 20],  # touches the inner left edge
            [1014, 10, 1024, 20],  # touches the image's right border: kept
            [500, 0, 510, 9],  # touches the image's top border: kept
            [500, 600, 510, 608],  # touches the inner bottom edge
        ],
        float,
    )
    np.testing.assert_array_equal(
        bi.inner_edge_cut(boxes, tile, 1920, 1080), [True, False, False, True]
    )


def test_greedy_ios_nms_keeps_the_most_confident_and_separate_balls():
    xyxy = np.array(
        [[0, 0, 10, 10], [1, 1, 9, 9], [2, 0, 12, 10], [50, 50, 60, 60]], float
    )
    conf = np.array([0.3, 0.9, 0.5, 0.1])
    kept = bi.greedy_ios_nms(xyxy, conf)
    # box 1 lies inside box 0 (IoS 1) and mostly inside box 2 (IoS 0.875): both dropped; box 3 is apart
    assert kept.tolist() == [1, 3]


def test_long_ball_drops_balls_over_45_px_at_1080p_only():
    candidates = pd.DataFrame(
        {
            "x1": [0, 0, 0, 0],
            "y1": [0, 0, 0, 0],
            "x2": [46, 44, 31, 100],
            "y2": [10, 10, 10, 100],
            "class_name": pd.Categorical(["ball", "ball", "ball", "player"]),
        }
    )
    heights = pd.Series([1080, 1080, 720, 1080])
    # 31 px at 720p is 46.5 px at 1080p
    np.testing.assert_array_equal(
        bi.long_ball(candidates, heights, 45), [True, False, True, False]
    )
