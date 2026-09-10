import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import diagnostic_review  # noqa: E402
from compare_ai_contact import (  # noqa: E402
    VARIANTS, build_inputs, choose_coarse, comparisons, decode_inputs, review_sheet, verify_evidence,
)
from diagnostic_review import CHOICES, import_annotations, validate_annotations  # noqa: E402
from xgpoc.v13.protocol import digest  # noqa: E402


@pytest.fixture
def review(tmp_path):
    public = tmp_path / "public"
    public.mkdir()
    frames = []
    for t in (-40, 0, 40):
        path = public / f"{t}.png"
        assert cv2.imwrite(str(path), np.full((36, 64, 3), t + 80, np.uint8))
        frames.append({"image": path.name, "time_ms": t, "sha256": digest(path)})
    packet = {"packet_id": "abc", "items": [{"review_id": "r0000", "frames": frames, "clip": "clip.mp4"}]}
    (public / "packet.json").write_text(json.dumps(packet))
    row = dict.fromkeys(CHOICES, "unknown") | {
        "review_id": "r0000", "contact_status": "bounded", "contact_lower_ms": -40,
        "contact_upper_ms": 0, "prior_annotation_exposure": False, "inspected_frames": frames,
        "evidence": [f | {"supports": side, "observation": "visible contact boundary"}
                     for f, side in zip(frames, ("lower_bound", "upper_bound"))],
    }
    payload = {"packet_id": "abc", "reviewer": "ai-test", "source": "model_review", "annotations": [row],
               "run": {"prior_annotation_exposure": False, "public_packet_sha256": digest(public / "packet.json")}}
    return packet, payload


def test_ai_source_requires_opt_in_and_preserves_exposure(review):
    packet, payload = review
    with pytest.raises(ValueError, match="source"):
        validate_annotations(payload, packet)
    rows = validate_annotations(payload, packet, allow_model_review=True)
    assert rows.source.tolist() == ["model_review"]
    assert not rows.prior_annotation_exposure.any()
    payload["run"]["prior_annotation_exposure"] = True
    assert validate_annotations(payload, packet, allow_model_review=True).prior_annotation_exposure.all()
    payload["annotations"][0].pop("prior_annotation_exposure")
    with pytest.raises(ValueError, match="exposure"):
        validate_annotations(payload, packet, allow_model_review=True)


def test_ai_import_keeps_raw_evidence_and_refuses_conflicting_reimport(tmp_path, review, monkeypatch):
    packet, payload = review
    monkeypatch.setattr(diagnostic_review, "verify_packet", lambda _: ({}, packet))
    pd.DataFrame([{"review_id": "r0000", "shot_id": "s0", "role": "fit", "sampling_group": "random"}]).to_csv(
        tmp_path / "sample.csv", index=False)
    export, out = tmp_path / "export.json", tmp_path / "imported.csv"
    export.write_text(json.dumps(payload, indent=2))
    import_annotations(tmp_path, export, out)
    before = (digest(out), digest(out.with_suffix(".source.json")))
    assert out.with_suffix(".source.json").read_bytes() == export.read_bytes()
    assert pd.read_csv(out).source.tolist() == ["model_review"]
    import_annotations(tmp_path, export, out)
    assert before == (digest(out), digest(out.with_suffix(".source.json")))
    payload["annotations"][0]["evidence"][0]["observation"] = "revised observation"
    export.write_text(json.dumps(payload, indent=2))
    with pytest.raises(ValueError, match="source evidence"):
        import_annotations(tmp_path, export, out)
    assert before == (digest(out), digest(out.with_suffix(".source.json")))
    payload["run"]["public_packet_sha256"] = "wrong"
    export.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="checksum"):
        import_annotations(tmp_path, export, tmp_path / "new.csv")


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (64, 36))
    assert writer.isOpened()
    for i in range(120):
        writer.write(np.full((36, 64, 3), i * 2, np.uint8))
    writer.release()
    return path


@pytest.mark.parametrize(("lower", "expected"), [(0, -40), (-40, -80), (80, 0), (-4000, None), (None, None)])
def test_dense_frame_is_strictly_before_lower_bound_and_never_after_t0(clip, lower, expected):
    frames = decode_inputs(clip, lower)
    assert frames["clip_t0"][1:] == (0, 100)
    assert frames["clip_t0"][0].shape == (36, 64, 3)
    if expected is None:
        assert "dense_precontact" not in frames
    else:
        assert frames["dense_precontact"][1] == expected


def test_coarse_selection_uses_measured_time_and_excludes_unknowns():
    frames = pd.DataFrame({"nominal_ms": [-800, -400, 0, 0, 400],
                           "measured_ms": [-780, -380, np.nan, 40, -30]})
    assert choose_coarse(frames, -380).measured_ms == -780
    assert choose_coarse(frames, 80).measured_ms == -380
    assert choose_coarse(frames, -800) is None
    assert choose_coarse(frames, None) is None


@pytest.mark.parametrize("unusable", ["uncertain", "prior_exposure"])
def test_unknown_or_exposed_contact_remains_in_coverage(tmp_path, review, clip, unusable):
    packet, payload = review
    row = payload["annotations"][0]
    if unusable == "uncertain":
        row.update(contact_status="uncertain", contact_lower_ms=None, contact_upper_ms=None)
    else:
        row["prior_annotation_exposure"] = True
    annotations = validate_annotations(payload, packet, allow_model_review=True).set_index("review_id")
    sample = pd.DataFrame([{"review_id": "r0000", "shot_id": "s0", "role": "fit",
                            "sampling_group": "random", "clip_path": str(clip), "goal": False}])
    cached = pd.DataFrame([{"shot_id": "s0", "nominal_ms": 0, "measured_ms": 0,
                            "frame_path": str(tmp_path / "public/0.png"), "frame_sha256": "test"}])
    inputs = build_inputs(sample, annotations, cached, tmp_path)
    assert len(inputs) == 4
    assert set(inputs[inputs.availability.eq("available")].variant) == {"original_t0", "clip_t0"}
    assert inputs.contact_quality.eq("unresolved").all()
    inputs["p"] = [.2, np.nan, .3, np.nan]
    summary, details = comparisons(inputs, sample)
    assert len(details) == 4
    assert details.paired.sum() == 1
    assert summary.unpaired_shots.sum() == 3
    assert details[~details.paired].delta_p.isna().all()


def test_paired_differences_and_visual_sheet(tmp_path):
    rows = []
    for variant, p in zip(VARIANTS, (.1, .2, .3, .4)):
        path = tmp_path / f"{variant}.png"
        assert cv2.imwrite(str(path), np.zeros((36, 64, 3), np.uint8))
        rows.append({"shot_id": "s0", "variant": variant, "p": p, "image": path.name, "time_ms": 0,
                     "image_origin": "original_clip", "availability": "available"})
    inputs = pd.DataFrame(rows)
    sample = pd.DataFrame([{"shot_id": "s0", "review_id": "r0000", "role": "select",
                            "sampling_group": "targeted", "goal": False}])
    summary, details = comparisons(inputs, sample)
    assert details.delta_p.to_numpy() == pytest.approx([.3, .2, .1, .2])
    assert summary.source_goals_paired.sum() == 0
    assert details.delta_loss_original_label.to_numpy() == pytest.approx(
        [np.log(.9 / .6), np.log(.8 / .6), np.log(.7 / .6), np.log(.9 / .7)])
    review_sheet(details, inputs, tmp_path)
    assert cv2.imread(str(tmp_path / "largest_changes.jpg")).shape == (240, 1536, 3)


@pytest.mark.parametrize("fault", [None, "hash", "time", "boundary", "uninspected", "foreign_clip"])
def test_evidence_checks_pixels_timestamps_and_contact_boundaries(tmp_path, review, fault):
    packet, payload = review
    row = payload["annotations"][0]
    if fault == "hash":
        (tmp_path / "public/-40.png").write_bytes(b"corrupt")
    elif fault == "time":
        row["inspected_frames"] = [dict(f) for f in row["inspected_frames"]]
        row["inspected_frames"][0]["time_ms"] = -20
    elif fault == "boundary":
        row["evidence"][0]["supports"] = "upper_bound"
    elif fault == "uninspected":
        row["evidence"][0]["image"] = "unseen.png"
    elif fault == "foreign_clip":
        row["inspected_frames"][0].update(time_basis="clip_pts", source_clip="elsewhere.mp4")
    if fault is None:
        assert verify_evidence(payload, packet, tmp_path, tmp_path) == 3
    else:
        with pytest.raises(ValueError):
            verify_evidence(payload, packet, tmp_path, tmp_path)
