import numpy as np

from xgpoc.features.aux import AUX_NAMES, shot_aux_features
from xgpoc.features.cache import BOX_DIM, MAX_BOXES, MAX_TOKENS, TOKEN_DIM


def _tokens(entries):
    t = np.zeros((MAX_TOKENS, TOKEN_DIM), np.float32)
    for i, e in enumerate(entries):
        t[i] = e
    return t


def _norm(x, y):
    return [x / 105.0, y / 68.0]


def test_occlusion_high_when_defender_on_shot_line():
    # ball at 94m centre, defender at 99m centre (between ball and goal), GK far to the side
    entries = [
        [*_norm(94, 34), 1, 0, 0, 0, 1, 1],
        [*_norm(99, 34), 0, 0, 0, 1, 1, 1],
    ]
    f = shot_aux_features(_tokens(entries), np.zeros((MAX_BOXES, BOX_DIM), np.float32), False)
    assert f["occl_valid"] == 1 and f["occlusion"] > 0.2 and f["n_shadow_players"] == 1


def test_occlusion_zero_when_defender_behind_ball():
    entries = [
        [*_norm(94, 34), 1, 0, 0, 0, 1, 1],
        [*_norm(80, 34), 0, 0, 0, 1, 1, 1],
    ]
    f = shot_aux_features(_tokens(entries), np.zeros((MAX_BOXES, BOX_DIM), np.float32), False)
    assert f["occlusion"] < 0.01


def test_gk_proxy_deepest_defender():
    entries = [
        [*_norm(90, 30), 1, 0, 0, 0, 1, 1],
        [*_norm(96, 30), 0, 0, 0, 1, 1, 1],
        [*_norm(103, 34), 0, 0, 0, 1, 1, 1],  # deepest -> proxy GK
    ]
    f = shot_aux_features(_tokens(entries), np.zeros((MAX_BOXES, BOX_DIM), np.float32), True)
    assert f["gkp_valid"] == 1 and abs(f["gkp_depth"] - 2.0) < 1e-6 and f["is_penalty"] == 1


def test_ball_in_air_from_boxes():
    boxes = np.zeros((MAX_BOXES, BOX_DIM), np.float32)
    boxes[0] = [0.50, 0.20, 0.52, 0.23, 1, 0, 0, 0, 1]   # ball high in the frame
    boxes[1] = [0.48, 0.40, 0.53, 0.60, 0, 0, 1, 0, 1]   # player below it
    f = shot_aux_features(_tokens([]), boxes, False)
    assert f["ball_air_valid"] == 1 and f["ball_in_air"] == 1
    assert set(f) == set(AUX_NAMES)
