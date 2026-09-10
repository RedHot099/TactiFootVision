import cv2
import numpy as np
import pytest

from xgpoc.v13.contact import (Admission, AdmissionStatus, ContactInterval, ContactSource, DecodedFrame, admit,
                               locate_in_clip)

REVIEW = ContactSource.MODEL_REVIEW


def frames(**measured):
    """Candidate frames at the nominal offsets; `measured` overrides by nominal offset."""
    return [DecodedFrame(f"{o}.jpg", o, measured.get(str(o))) for o in (-1600, -1200, -800, -400, 0)]


def test_contact_earlier_than_nominal_t0_rejects_the_t0_frame():
    result = admit(frames(), ContactInterval(-200, 0, REVIEW, "high"))
    assert result.status is AdmissionStatus.SAFE
    assert result.frame.nominal_ms == -400
    assert result.gap_ms == 200
    assert result.rejected == ("0.jpg",)


def test_frame_exactly_at_interval_start_is_rejected():
    result = admit(frames(), ContactInterval(-400, 200, REVIEW, "medium"))
    assert result.frame.nominal_ms == -800
    assert result.gap_ms == 400
    assert "-400.jpg" in result.rejected


def test_contact_after_t0_admits_the_t0_frame_with_its_gap():
    result = admit(frames(), ContactInterval(200, 400, REVIEW, "high"))
    assert result.frame.nominal_ms == 0
    assert result.gap_ms == 200
    assert result.rejected == ()


def test_unknown_contact_keeps_nominal_t0_and_says_so():
    result = admit(frames(), ContactInterval.unknown())
    assert result.status is AdmissionStatus.NOMINAL
    assert result.frame.nominal_ms == 0
    assert result.gap_ms is None
    assert result.measured is False


def test_unknown_contact_without_t0_frame_is_an_error_not_a_substitution():
    with pytest.raises(ValueError, match="t0"):
        admit(frames()[:-1], ContactInterval.unknown())


def test_late_decoded_frame_is_judged_by_its_measured_time():
    # The seek for the nominal -400 frame landed on the anchor itself.
    late = frames(**{"-400": 0, "0": 40, "-800": -800})
    result = admit(late, ContactInterval(0, 200, REVIEW, "high"))
    assert result.frame.nominal_ms == -800
    assert result.gap_ms == 800
    assert result.measured is True
    assert set(result.rejected) == {"-400.jpg", "0.jpg"}


def test_no_safe_frame_is_reported_and_no_frame_is_chosen():
    result = admit(frames(), ContactInterval(-2000, -1800, REVIEW, "low"))
    assert result.status is AdmissionStatus.UNSAFE
    assert result.frame is None
    assert result.gap_ms == -400
    assert len(result.rejected) == 5
    assert result.to_record()["frame_path"] == ""


def test_future_nominal_offsets_are_never_candidates():
    with pytest.raises(ValueError, match="never inputs"):
        DecodedFrame("+400.jpg", 400)


def test_duplicate_offsets_and_empty_candidates_are_errors():
    with pytest.raises(ValueError, match="no candidate"):
        admit([], ContactInterval.unknown())
    with pytest.raises(ValueError, match="duplicate"):
        admit([DecodedFrame("a.jpg", 0), DecodedFrame("b.jpg", 0)], ContactInterval.unknown())


def test_interval_validation():
    with pytest.raises(ValueError, match="both bounds"):
        ContactInterval(0, None, REVIEW)
    with pytest.raises(ValueError, match="lower bound above"):
        ContactInterval(200, 0, REVIEW)
    with pytest.raises(ValueError, match="not evidence"):
        ContactInterval(0, 200)


def test_review_rows_map_to_intervals():
    known = ContactInterval.from_review({"contact_lower_ms": -200.0, "contact_upper_ms": 200.0,
                                         "t0_status": "contact", "confidence": "medium", "timing_consistent": True})
    assert (known.lower_ms, known.upper_ms, known.known) == (-200, 200, True)
    assert known.source is REVIEW
    uncertain = ContactInterval.from_review({"contact_lower_ms": float("nan"), "contact_upper_ms": float("nan"),
                                             "t0_status": "uncertain", "confidence": "low"})
    assert not uncertain.known and uncertain.confidence == "low"
    inconsistent = ContactInterval.from_review({"contact_lower_ms": 200, "contact_upper_ms": 400,
                                                "t0_status": "post", "timing_consistent": False})
    assert not inconsistent.known


def _write_clip(path, n_frames=50, fps=25, duplicate=None):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (96, 64))
    assert writer.isOpened()
    images = []
    for i in range(n_frames):
        j = duplicate if duplicate is not None and i == duplicate + 1 else i
        img = np.full((64, 96, 3), (40, 120, 60), np.uint8)
        cv2.rectangle(img, (j, 10), (j + 20, 40), (255, 255, 255), -1)
        writer.write(img)
        images.append(img)
    writer.release()
    return images


def test_locate_in_clip_recovers_the_decoded_instant(tmp_path):
    clip = tmp_path / "clip.avi"
    images = _write_clip(clip)
    jpeg = tmp_path / "frame.jpg"
    cv2.imwrite(str(jpeg), images[30])
    match = locate_in_clip(str(jpeg), str(clip), anchor_ms=1000, window_ms=(-1000, 1000))
    assert match.offset_ms == 200          # frame 30 at 25 fps = 1200 ms, anchor 1000 ms
    assert match.margin > 0.25
    assert match.measured_offset() == 200


def test_locate_in_clip_declines_when_neighbouring_frames_are_identical(tmp_path):
    clip = tmp_path / "clip.avi"
    images = _write_clip(clip, duplicate=30)
    jpeg = tmp_path / "frame.jpg"
    cv2.imwrite(str(jpeg), images[30])
    match = locate_in_clip(str(jpeg), str(clip), anchor_ms=1000, window_ms=(-1000, 1000))
    assert match.offset_ms in (200, 240)
    assert match.measured_offset() is None


def test_admission_record_is_flat_and_complete():
    record = admit(frames(), ContactInterval(-200, 0, REVIEW, "high")).to_record()
    assert record == {"admission": "safe", "frame_path": "-400.jpg", "frame_nominal_ms": -400, "frame_time_ms": -400,
                      "gap_ms": 200, "timing_measured": False, "n_rejected": 1}
    assert isinstance(admit(frames(), ContactInterval.unknown()), Admission)
