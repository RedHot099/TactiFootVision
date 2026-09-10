import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_v14_initialization import check  # noqa: E402
from diagnostic_review import CHOICES, prepare, select_sample, validate_annotations, verify_packet  # noqa: E402
from xgpoc.v14.models import MultiViewViTXG  # noqa: E402


def test_selection_excludes_calibration_and_evaluation_and_separates_samples():
    pool = pd.DataFrame({"shot_id": [str(i) for i in range(40)], "role": ["fit"] * 20 + ["select"] * 10 +
                         ["calibrate"] * 5 + ["evaluate"] * 5})
    reviews = pd.DataFrame({"shot_id": [str(i) for i in range(40)], "interval_status": ["uncertain"] * 20 + ["known"] * 20})
    sample = select_sample(pool, reviews, 10, 8)
    assert not sample.shot_id.duplicated().any()
    assert set(sample.role) <= {"fit", "select"}
    assert sample.sampling_group.value_counts().to_dict() == {"random": 10, "targeted": 8}
    pd.testing.assert_frame_equal(sample, select_sample(pool.iloc[::-1], reviews, 10, 8))


def data():
    packet = {"packet_id": "abc", "items": [{"review_id": "r0000", "frames": [{"time_ms": t} for t in (-40, 0, 40)]}]}
    row = dict.fromkeys(CHOICES, "unknown") | {"review_id": "r0000", "contact_status": "bounded",
        "contact_lower_ms": -40, "contact_upper_ms": 0, "shooter_x": .5, "shooter_y": .6, "shooter_frame_ms": -40}
    payload = {"packet_id": "abc", "reviewer": "reviewer-a", "source": "human", "annotations": [row]}
    return packet, payload


def test_import_requires_real_packet_identity_and_explicit_human_source():
    packet, payload = data()
    assert len(validate_annotations(payload, packet)) == 1
    for change in ({"packet_id": "wrong"}, {"source": "model_review"}, {"reviewer": ""}, {"annotations": []}):
        with pytest.raises(ValueError):
            validate_annotations(payload | change, packet)


@pytest.mark.parametrize("change", [
    {"contact_lower_ms": 20}, {"contact_lower_ms": 40, "contact_upper_ms": -40},
    {"contact_lower_ms": False}, {"contact_status": "uncertain"}, {"shooter_frame_ms": 10},
    {"shooter_x": 2}, {"shooter_x": None}, {"outcome": "no_goal", "goal_attribution": "this_attempt"},
    {"review_id": "foreign"},
])
def test_invalid_annotations_are_rejected(change):
    packet, payload = data()
    payload["annotations"][0].update(change)
    with pytest.raises(ValueError):
        validate_annotations(payload, packet)


def test_uncertain_annotations_remain_unknown_and_duplicate_ids_fail():
    packet, payload = data()
    payload["annotations"][0].update(contact_status="uncertain", contact_lower_ms=None, contact_upper_ms=None)
    assert validate_annotations(payload, packet).contact_lower_ms.isna().all()
    payload["annotations"] *= 2
    with pytest.raises(ValueError, match="duplicate"):
        validate_annotations(payload, packet)


def test_identity_fusion_is_not_baseline_equivalence():
    torch.manual_seed(1300)
    model = MultiViewViTXG("dinov2_s", n_blocks=0, fusion_heads=4, fusion_ff=64,
                          pretrained=False, grad_checkpointing=False)
    rows = check(model, torch.randn(1, 3, 56, 56), torch.randn(1, 2, 3, 126, 224))
    assert all(r["fusion_identity_error"] < 1e-6 for r in rows)
    assert all(r["masked_local_logit_error"] < 1e-5 for r in rows)
    assert all(r["base_logit_error"] > 1e-5 for r in rows)


def test_real_packet_resume_and_corruption_detection(tmp_path):
    clip = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 36))
    for i in range(80):
        writer.write(np.full((36, 64, 3), i * 3, np.uint8))
    writer.release()
    index = tmp_path / "index.parquet"
    pd.DataFrame({"shot_id": [f"s{i}" for i in range(20)], "game": [f"game{i}" for i in range(20)],
                  "split": "train", "clip_path": str(clip), "goal": False}).to_parquet(index)
    review = tmp_path / "review.csv"
    pd.DataFrame({"shot_id": [], "interval_status": []}).to_csv(review, index=False)
    out = tmp_path / "packet"
    prepare(index, review, out, 2, 0)
    manifest, packet = verify_packet(out)
    assert len(packet["items"]) == 2
    assert set(packet["items"][0]) == {"review_id", "frames", "clip"}
    prepare(index, review, out, 2, 0)
    assert verify_packet(out)[0] == manifest
    frame = out / "public" / packet["items"][0]["frames"][0]["image"]
    frame.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Packet file changed"):
        prepare(index, review, out, 2, 0)
