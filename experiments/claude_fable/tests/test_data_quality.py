import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from report_data_quality import CATEGORIES, neighbouring_attempts, shot_flags, window_changes  # noqa: E402


@pytest.fixture
def cut_clip(tmp_path):
    """4.8 s at 25 fps; a hard cut between frame 59 (-1640 ms) and 60 (-1600 ms), then a second one at frame 90 (-400 ms)."""
    path = tmp_path / "cut.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (64, 36))
    assert writer.isOpened()
    rng = np.random.default_rng(3)
    for i in range(120):
        level = 40 if i < 60 else 120 if i < 90 else 220
        writer.write(np.clip(level + rng.integers(-3, 4, (36, 64, 3)), 0, 255).astype(np.uint8))
    writer.release()
    return path


def test_window_changes_measure_consecutive_frames_inside_window_only(cut_clip):
    changes = window_changes(cut_clip)
    assert changes.anchor_relative_ms.min() == -1600 and changes.anchor_relative_ms.max() == 0
    assert len(changes) == 41
    peak = changes.loc[changes.change.idxmax()]
    assert peak.anchor_relative_ms == -400 and peak.change > .3
    # the first window frame is compared with the frame before it, so a cut on the window edge still counts
    assert changes[changes.anchor_relative_ms.eq(-1600)].change.item() > .25
    assert changes[~changes.anchor_relative_ms.isin([-1600, -400])].change.max() < .05


def index_fixture():
    rows = [("s0", "g", "train", 1, 100000, False, "Shots on target"), ("s1", "g", "train", 1, 104000, True, "Shots on target"),
            ("s2", "g", "train", 2, 100000, False, "Shots off target"), ("s3", "g", "train", 1, 130000, False, "Shots off target")]
    rows += [(f"h{i}", f"other{i}/2015-01-01 - 20-00 Team{i} 1 - 0 X", "train" if i < 8 else "test", 1, 5000 * i, False, "Shots on target")
             for i in range(10)]
    return pd.DataFrame(rows, columns=["shot_id", "game", "split", "half", "position_ms", "goal", "label"])


def test_neighbours_are_same_half_within_window_and_report_partition():
    index = index_fixture()
    shots = pd.DataFrame([{"review_id": "r0", "shot_id": "s0", "game": "g", "half": 1, "role": "fit"}])
    found = neighbouring_attempts(shots, index, 15000)
    assert found.neighbour_shot_id.tolist() == ["s1"]
    assert found.neighbour_delta_ms.item() == 4000 and found.neighbour_goal.item()
    assert found.same_partition.item() == (found.neighbour_partition.item() == "fit")
    assert neighbouring_attempts(shots, index, 2000).empty


def flag_fixture(**shot):
    base = {"review_id": "r0", "shot_id": "s0", "role": "fit", "sampling_group": "random", "source_goal": False,
            "extraction_status": "ok", "prior_annotation_exposure": False, "contact_status": "bounded",
            "contact_lower_ms": -40.0, "contact_upper_ms": 0.0, "contact_width_ms": 40.0, "ai_outcome": "no_goal",
            "ai_goal_attribution": "no_goal", "label_review_status": "ai_agrees_with_source", "ball_visible": "yes",
            "keeper_visible": "yes", "shooter_x": .5, "shooter_y": .5}
    shots = pd.DataFrame([base | shot])
    frames = pd.DataFrame([
        {"review_id": "r0", "variant": "t0", "availability": "available", "gap_ms": -40.0, "pixel_sha256": "a", "near_black_review": False},
        {"review_id": "r0", "variant": "precontact", "availability": "available", "gap_ms": 40.0, "pixel_sha256": "b", "near_black_review": False},
    ])
    return shots, frames


def test_clean_shot_has_no_flags():
    shots, frames = flag_fixture()
    changes = pd.DataFrame([{"review_id": "r0", "anchor_relative_ms": -800.0, "change": .01}])
    flags = shot_flags(shots, frames, pd.DataFrame(columns=["review_ids"]), changes, pd.DataFrame(), .12, 80)
    assert flags.technical_flags.item() == "" and flags.review_flags.item() == ""
    assert flags.info_flags.item() == "t0_at_or_after_contact_start"
    assert flags.window_max_change.item() == .01 and flags.neighbour_count.item() == 0


def test_technical_and_review_flags_are_kept_apart():
    shots, frames = flag_fixture(contact_status="uncertain", contact_lower_ms=None, contact_upper_ms=None,
                                 contact_width_ms=None, ai_outcome="unknown", ai_goal_attribution="unknown",
                                 ball_visible="no", shooter_x=None, shooter_y=None)
    frames.loc[1, "availability"] = "uncertain"
    changes = pd.DataFrame([{"review_id": "r0", "anchor_relative_ms": -400.0, "change": .4}])
    neighbours = pd.DataFrame([{"review_id": "r0", "neighbour_delta_ms": -3000, "same_partition": False}])
    flags = shot_flags(shots, frames, pd.DataFrame(columns=["review_ids"]), changes, neighbours, .12, 80)
    assert set(flags.technical_flags.item().split("|")) == {"cut_suspect_in_window", "neighbour_in_other_partition"}
    assert set(flags.review_flags.item().split("|")) == {"contact_uncertain", "precontact_unavailable", "outcome_unknown",
                                                          "ball_not_visible", "shooter_unlocated", "neighbour_attempt_in_half"}
    assert flags.nearest_neighbour_ms.item() == 3000
    assert all(CATEGORIES[f] == "technical" for f in flags.technical_flags.item().split("|"))


def test_decode_failure_duplicate_and_timing_flags():
    shots, frames = flag_fixture(extraction_status="failed", contact_lower_ms=80.0, contact_upper_ms=160.0, contact_width_ms=80.0)
    frames.loc[1, "gap_ms"], frames.loc[1, "pixel_sha256"] = 80.0, "a"
    duplicates = pd.DataFrame([{"review_ids": "r0|r9"}])
    flags = shot_flags(shots, frames, duplicates, pd.DataFrame(columns=["review_id", "anchor_relative_ms", "change"]),
                       pd.DataFrame(), .12, 80)
    assert set(flags.technical_flags.item().split("|")) == {"decode_failed", "exact_duplicate_frame"}
    assert flags.review_flags.item() == "gap_above_frame_step"
    assert flags.info_flags.item() == "precontact_equals_t0"
    assert not flags.t0_at_or_after_contact_start.item() and not flags.wide_contact_interval.item()
    assert pd.isna(flags.window_max_change.item())
