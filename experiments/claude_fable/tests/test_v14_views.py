import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from xgpoc.v14.models import MultiViewViTXG
from xgpoc.v14.views import LOCAL_SIZE, MAX_SHOOTER_DISTANCE, VIEWS, MultiViewDataset, view_boxes


def box(x0, y0, x1, y1, cls):
    row = np.zeros(9, np.float32)
    row[:4] = (x0, y0, x1, y1)
    row[4 + cls] = 1
    row[8] = 1
    return row


BALL, KEEPER, PLAYER = 0, 1, 2


def test_shooter_is_the_nearest_player_and_keeper_is_detected():
    boxes = np.stack([box(.50, .50, .51, .52, BALL), box(.52, .45, .55, .58, PLAYER), box(.80, .40, .83, .55, PLAYER),
                      box(.90, .45, .93, .58, KEEPER)])
    vb = view_boxes(boxes)
    assert vb.valid == (1.0, 1.0)
    assert vb.shooter_distance == pytest.approx(np.hypot(.535 - .505, .515 - .51))
    x0, y0, x1, y1 = vb.boxes[0]
    assert x0 <= .50 and x1 >= .55 and y0 <= .45 and y1 >= .58
    assert (x1 - x0) == pytest.approx(y1 - y0)
    for real, control in zip(vb.boxes, vb.controls):
        assert (control[2] - control[0]) == pytest.approx(real[2] - real[0])
        assert control != real


def test_far_player_is_not_a_shooter_and_missing_detections_mask_views():
    far = np.stack([box(.10, .10, .11, .12, BALL), box(.10 + MAX_SHOOTER_DISTANCE + .05, .10, .13 + .2, .25, PLAYER)])
    vb = view_boxes(far)
    assert vb.boxes[0] is None and vb.shooter_distance > MAX_SHOOTER_DISTANCE
    assert vb.valid == (0.0, 0.0)
    assert view_boxes(np.zeros((0, 9), np.float32)).valid == (0.0, 0.0)
    assert view_boxes(None).controls == (None, None)


def _frame(tmp_path, name="f.jpg"):
    img = np.random.default_rng(0).integers(0, 255, (144, 256, 3), np.uint8)
    path = tmp_path / name
    cv2.imwrite(str(path), img)
    return str(path)


def test_dataset_shapes_flags_and_frame_mismatch(tmp_path):
    frame = _frame(tmp_path)
    other = _frame(tmp_path, "g.jpg")
    df = pd.DataFrame({"shot_id": ["a", "b", "c"], "goal": [True, False, False], "on_target": [1., 0., 1.],
                       "frame_path": [frame, frame, other], "frame_path_t0": [frame, frame, frame]})
    boxes = {"a": np.stack([box(.5, .5, .51, .52, BALL), box(.52, .45, .55, .58, PLAYER)]),
             "c": np.stack([box(.5, .5, .51, .52, BALL), box(.52, .45, .55, .58, PLAYER), box(.9, .4, .93, .55, KEEPER)])}
    ds = MultiViewDataset(df, boxes, box_frame_col="frame_path_t0", aux_col="on_target")
    a, b, c = ds[0], ds[1], ds[2]
    assert a["image"].shape == (3, 252, 448)
    assert a["views"].shape == (len(VIEWS), 3, LOCAL_SIZE[1], LOCAL_SIZE[0])
    assert a["view_valid"].tolist() == [1., 0.] and a["y"] == 1 and a["y_aux"].tolist() == [1.]
    assert b["view_valid"].tolist() == [0., 0.]          # no boxes cached for this shot
    assert c["view_valid"].tolist() == [0., 0.]          # frame differs from the detection frame
    assert torch.equal(b["views"][0], b["views"][1])        # masked views carry a blank image
    assert float(b["views"][0].flatten(1).std(1).max()) == 0
    control = MultiViewDataset(df, boxes, control=True, box_frame_col="frame_path_t0")[0]
    assert control["view_valid"].tolist() == a["view_valid"].tolist()
    assert not torch.equal(control["views"][0], a["views"][0])
    with pytest.raises(ValueError, match="illegal"):
        MultiViewDataset(df, boxes, path_col="frame_path_+400")


def _model():
    torch.manual_seed(0)
    return MultiViewViTXG("dinov2_s", n_blocks=1, n_out=2, fusion_heads=4, fusion_ff=64,
                          grad_checkpointing=False, pretrained=False).eval()


def test_model_output_shape_and_missing_view_invariance():
    model = _model()
    image = torch.randn(2, 3, 252, 448)
    views = torch.randn(2, len(VIEWS), 3, LOCAL_SIZE[1], LOCAL_SIZE[0])
    valid = torch.tensor([[1., 0.], [0., 0.]])
    with torch.no_grad():
        out = model(image, views, valid)
        altered = views.clone()
        altered[:, 1] = 0                       # missing keeper view: pixels must not matter
        altered[1, 0] = 0                       # sample 1 has no views at all
        same = model(image, altered, valid)
        changed = model(image, views * 0 + 1, torch.ones(2, 2))
    assert out.shape == (2, 2)
    assert torch.allclose(out, same, atol=1e-5)
    assert not torch.allclose(out, changed, atol=1e-3)


def test_fusion_starts_as_identity_and_delegates_backbone_groups():
    model = _model()
    x = torch.randn(1, 7, model.base.backbone.embed_dim)
    with torch.no_grad():
        assert torch.allclose(model.fusion(x), x, atol=1e-6)
    assert len(model.backbone_param_groups()) == 1
    assert all(p.requires_grad for p in model.backbone_parameters())
    with pytest.raises(ValueError, match="views must be"):
        model(torch.randn(1, 3, 252, 448), torch.randn(1, 1, 3, LOCAL_SIZE[1], LOCAL_SIZE[0]), torch.ones(1, 1))


def test_all_views_missing_in_a_batch_still_runs():
    model = _model()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 252, 448), torch.zeros(1, 2, 3, LOCAL_SIZE[1], LOCAL_SIZE[0]), torch.zeros(1, 2))
    assert out.shape == (1, 2) and torch.isfinite(out).all()
