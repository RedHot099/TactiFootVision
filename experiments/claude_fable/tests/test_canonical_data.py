import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_canonical_data import paired_details, summarize  # noqa: E402
from diagnostic_review import CHOICES  # noqa: E402
from prepare_canonical_data import build_tables, complete, duplicate_frames, source_inventory, verify_result  # noqa: E402
from xgpoc.data.canonical_frames import (  # noqa: E402
    array_digest, decode_ffmpeg_frame, encode_jpeg_control, extract_inputs, image_quality,
    model_input_digest, probe_video, write_png,
)
from xgpoc.data.datasets import ImageDataset  # noqa: E402
from xgpoc.v13.protocol import digest  # noqa: E402


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (64, 36))
    assert writer.isOpened()
    rng = np.random.default_rng(6)
    for _ in range(120):
        writer.write(rng.integers(0, 256, (36, 64, 3), dtype=np.uint8))
    writer.release()
    return path


@pytest.mark.parametrize(("lower", "time"), [(0, -40), (-40, -80), (80, 0), (-4000, None), (None, None)])
def test_selection_uses_exact_timestamp_boundary(clip, lower, time):
    timeline, selected = extract_inputs(clip, lower)
    assert len(timeline.pts) == 120
    assert selected["t0"].index == 100
    if time is None:
        assert "precontact" not in selected
    else:
        assert selected["precontact"].relative_ms == time
        assert selected["precontact"].bgr.shape == (36, 64, 3)


def test_nonzero_pts_origin_does_not_shift_contact(clip, tmp_path):
    shifted = tmp_path / "shifted.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(clip), "-c", "copy", "-output_ts_offset", "3", str(shifted)], check=True)
    timeline, selected = extract_inputs(shifted, 0)
    assert timeline.milliseconds(0) == 3000
    assert selected["t0"].relative_ms == 0
    assert selected["precontact"].relative_ms == -40


def test_lossless_pixels_and_actual_model_tensor_match(clip, tmp_path):
    timeline, selected = extract_inputs(clip, 0)
    frame = selected["t0"]
    path = tmp_path / "frame.png"
    write_png(path, frame.bgr)
    assert array_digest(cv2.imread(str(path))) == array_digest(frame.bgr)
    dataset = ImageDataset(pd.DataFrame([{"frame_path": str(path), "goal": False}]), size_wh=(448, 252), mask="overlay")
    assert model_input_digest(path) == array_digest(dataset[0]["image"].numpy())
    decoded = decode_ffmpeg_frame(clip, frame.index, timeline.width, timeline.height)
    assert np.array_equal(frame.bgr, decoded)
    jpeg = tmp_path / "control.jpg"
    encode_jpeg_control(path, jpeg)
    assert cv2.imread(str(jpeg)).shape == frame.bgr.shape
    assert not np.array_equal(cv2.imread(str(jpeg)), frame.bgr)
    assert image_quality(frame)["laplacian_variance"] > 0


def test_corrupt_video_and_anchor_beyond_clip_are_rejected(clip, tmp_path):
    with pytest.raises(ValueError, match="ends before"):
        extract_inputs(clip, 0, anchor_ms=6000)
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"broken video")
    with pytest.raises(subprocess.CalledProcessError):
        probe_video(bad)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_boundaries_are_rejected(clip, value):
    with pytest.raises(ValueError, match="finite"):
        extract_inputs(clip, value)


@pytest.mark.parametrize("case", ["bounded", "uncertain", "exposed", "early", "corrupt"])
def test_manifest_preserves_each_shot_and_separate_label_uncertainty(clip, tmp_path, case):
    sample = pd.DataFrame([{"review_id": "r0000", "shot_id": "s0", "game": "game", "half": 1, "role": "fit",
                            "sampling_group": "random", "goal": False, "label": "Shots off target", "is_penalty": False,
                            "clip_path": str(clip), "position_ms": 10000 if case != "early" else 2000}])
    annotation = dict.fromkeys(CHOICES, "unknown") | {
        "review_id": "r0000", "reviewer": "ai", "source": "model_review", "prior_annotation_exposure": case == "exposed",
        "contact_status": "uncertain" if case == "uncertain" else "bounded", "contact_lower_ms": 0,
        "contact_upper_ms": 40,
    }
    if case == "uncertain":
        annotation.update(contact_lower_ms=None, contact_upper_ms=None)
    if case == "corrupt":
        clip.write_bytes(b"bad")
    annotations = pd.DataFrame([annotation]).set_index("review_id")
    recovery = pd.DataFrame([{"game": "game", "half": 1, "expected_sha256": "recorded", "source_available": False}])
    packet = {"items": [{"review_id": "r0000", "frames": [{"time_ms": t} for t in (-40, 0, 40)]}]}
    shots, frames = build_tables(sample, annotations, packet, recovery, tmp_path)
    assert len(shots) == 1 and len(frames) == 2
    assert shots.label_review_status.tolist() == ["unresolved"]
    assert shots.source_goal.tolist() == [False]
    assert shots.annotation_source.tolist() == ["model_review"]
    available = frames[frames.availability.eq("available")]
    if case == "bounded":
        assert len(available) == 2
        assert frames.loc[frames.variant.eq("precontact"), "gap_ms"].item() == 40
    elif case in {"uncertain", "exposed"}:
        assert available.variant.tolist() == ["t0"]
        assert shots.extraction_status.tolist() == ["ok"]
    else:
        assert len(available) == 0
        assert frames.availability.eq("decode_failed").all()


def test_duplicate_check_groups_distinct_shots_not_variants():
    frames = pd.DataFrame({"shot_id": ["s0", "s0", "s1"], "review_id": ["r0", "r0", "r1"],
                           "pixel_sha256": ["a", "a", "a"], "tensor_sha256": ["b", "b", "b"], "availability": "available"})
    shots = pd.DataFrame({"shot_id": ["s0", "s1"], "role": ["fit", "select"]})
    assert duplicate_frames(frames.iloc[:2], shots).empty
    found = duplicate_frames(frames, shots)
    assert len(found) == 2 and found.cross_role.all()


def test_inventory_retains_missing_raw_source_and_rejects_conflicts(tmp_path):
    sample = pd.DataFrame({"review_id": ["r0", "r1"], "game": "game", "half": 1})
    asset = {"game": "game", "half": 1, "component": "720p", "sha256": "a", "bytes": 100}
    path = tmp_path / "assets.jsonl"
    path.write_text(json.dumps(asset) + "\n")
    inventory = source_inventory(sample, path, tmp_path / "raw")
    assert len(inventory) == 1 and inventory.expected_bytes.item() == 100
    assert not inventory.source_available.any()
    path.write_text(path.read_text() + json.dumps(asset | {"sha256": "b"}) + "\n")
    with pytest.raises(ValueError, match="conflicting"):
        source_inventory(sample, path, tmp_path / "raw")


def test_completed_result_rejects_changed_files_and_sources(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("source")
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "file.txt").write_text("result")
    out = tmp_path / "out"
    identity = {"sources": {str(source): digest(source)}}
    complete(stage, out, identity)
    assert verify_result(out)["identity"] == identity
    assert not stage.exists()
    (out / "file.txt").write_text("changed")
    with pytest.raises(ValueError, match="Output changed"):
        verify_result(out)
    (out / "file.txt").write_text("result")
    source.write_text("changed source")
    with pytest.raises(ValueError, match="Source changed"):
        verify_result(out)


def test_control_comparison_keeps_missing_contact_and_measures_real_tensors(tmp_path):
    image = np.full((36, 64, 3), 80, np.uint8)
    write_png(tmp_path / "a.png", image)
    write_png(tmp_path / "b.png", image + 20)
    shots = pd.DataFrame([{"shot_id": "s0", "review_id": "r0", "role": "fit", "sampling_group": "random", "source_goal": False}])
    rows = []
    for variant, path, probability in [("canonical_t0", "a.png", .1), ("decoder_t0", "a.png", .1),
                                       ("jpeg_t0", "b.png", .2), ("legacy_t0", "b.png", .3),
                                       ("canonical_precontact", "", np.nan)]:
        rows.append({"shot_id": "s0", "variant": variant, "frame_path": path, "p": probability,
                     "availability": "available" if path else "uncertain"})
    details = paired_details(pd.DataFrame(rows), shots, tmp_path)
    assert len(details) == 4 and details.paired.sum() == 3
    same = details[details["first"].eq("decoder_t0")].iloc[0]
    assert same.same_pixels and same.same_tensor and same.delta_p == 0
    jpeg = details[details["first"].eq("jpeg_t0")].iloc[0]
    assert jpeg.native_pixel_mae == 20 and jpeg.tensor_mae > 0
    summary = summarize(details)
    unresolved = summary[summary["first"].eq("canonical_precontact")].iloc[0]
    assert unresolved.unpaired_shots == 1 and pd.isna(unresolved.mean_delta_p)
