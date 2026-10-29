"""``BallCropScorer``: the second stage that rescores ball candidates on crops."""

from pathlib import Path

import numpy as np
import pytest
import supervision as sv

import tactifoot_vision as tv
from tactifoot_vision.models import BallCropScorer

FUSION = (0.5, 2.0, -1.0)  # a, b, c of sigmoid(a logit(p_crop) + b logit(conf) + c)


def _candidates(boxes: list[list[float]], confidence: list[float]) -> sv.Detections:
    return sv.Detections(
        xyxy=np.array(boxes, np.float32).reshape(-1, 4),
        confidence=np.array(confidence, np.float32),
        class_id=np.zeros(len(boxes), int),
        data={"class_name": np.array(["ball"] * len(boxes))},
    )


def _logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def test_the_scorer_fuses_the_crop_with_the_detector_and_centres_the_box(
    stub_ball_scorer,
):
    scorer = BallCropScorer(
        stub_ball_scorer.weights, min_confidence=0.0, fusion=FUSION, device="cpu"
    )
    image = np.zeros((540, 960, 3), np.uint8)  # half of 1080p: crops >= 16 px
    # a small ball (crop 16 px, the floor) and a 10 px one (crop 30 px)
    candidates = _candidates([[100, 100, 104, 104], [300, 200, 310, 208]], [0.3, 0.6])

    scored = scorer(image, candidates)

    a, b, c = FUSION
    expected = [
        1 / (1 + np.exp(-(a * stub_ball_scorer.logit + b * _logit(conf) + c)))
        for conf in (0.3, 0.6)
    ]
    np.testing.assert_allclose(scored.confidence, expected, rtol=1e-5)
    dx, dy = stub_ball_scorer.offset
    for box, moved, side in zip(candidates.xyxy, scored.xyxy, (16, 30), strict=True):
        np.testing.assert_allclose(moved, box + [dx * side, dy * side] * 2, atol=1e-3)
    assert scored.data["class_name"].tolist() == ["ball", "ball"]
    # the input is left alone
    assert candidates.confidence.tolist() == pytest.approx([0.3, 0.6])


def test_the_scorer_drops_weak_candidates_and_can_keep_the_boxes(
    stub_ball_scorer,
):
    image = np.zeros((1080, 1920, 3), np.uint8)
    candidates = _candidates([[100, 100, 110, 110], [500, 500, 510, 510]], [0.1, 0.6])
    scorer = BallCropScorer(
        stub_ball_scorer.weights,
        min_confidence=0.5,
        fusion=FUSION,
        move_boxes=False,
        device="cpu",
    )

    scored = scorer(image, candidates)

    # 0.1 fuses to ~0.03, 0.6 to ~0.67
    np.testing.assert_array_equal(scored.xyxy, candidates.xyxy[1:])
    assert len(scorer(image, candidates[:0])) == 0


def test_the_scorer_checks_its_options(stub_ball_scorer, tmp_path):
    with pytest.raises(ValueError, match="min_confidence"):
        BallCropScorer(stub_ball_scorer.weights, min_confidence=1.5, device="cpu")
    with pytest.raises(ValueError, match="fusion"):
        BallCropScorer(stub_ball_scorer.weights, fusion=(1.0, 2.0), device="cpu")
    with pytest.raises(FileNotFoundError, match="ball crop scorer"):
        BallCropScorer(tmp_path / "missing.pt", device="cpu")


@pytest.mark.model
def test_the_real_scorer_rescores_a_gsr_clip(detector_weights, ball_scorer_weights):
    frames = (
        Path(__file__).resolve().parents[1] / "data/SoccerNetGS/valid/SNGS-021/img1"
    )
    if not frames.is_dir():
        pytest.skip("SoccerNet GSR valid not available")
    detector = tv.load_model("yolo", detector_weights, class_conf={"ball": 0.01})
    scorer = BallCropScorer(ball_scorer_weights)
    result = tv.Pipeline(detector, tracker=None, ball_scorer=scorer).run(
        str(frames / "%06d.jpg"), start=170, end=195, stride=5, progress=False
    )
    for frame in result:
        confidence = frame.ball_candidates.confidence
        assert ((confidence >= 0.15) & (confidence <= 1)).all()
        assert (np.diff(confidence) <= 0).all()  # best first
    assert sum(len(frame.ball) for frame in result) >= 3
