import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_offset_frames as builder  # noqa: E402
from check_offset_zero import compare  # noqa: E402
from prepare_canonical_data import verify_result  # noqa: E402
from report_data_quality import window_changes  # noqa: E402
from xgpoc.data.canonical_frames import (  # noqa: E402
    array_digest,
    extract_inputs,
    write_png,
)
from xgpoc.data.offset_frames import (  # noqa: E402
    CUT_THRESHOLD,
    GRID_MS,
    TOLERANCE_MS,
    anchor_frame_index,
    anchor_selection,
    check_frame_rate,
    extract_offset_frames,
    max_change_between,
    select_offsets,
)

PROTOCOL = Path(__file__).resolve().parents[1] / "research/xgc_direction_probe_protocol.md"


def write_clip(path: Path, frames: int = 200, cut_at: int | None = None, seed: int = 6, fps: float = 25) -> Path:
    """Clip of a gradient scene with light noise; from `cut_at` on the scene is inverted (a hard cut)."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 36))
    assert writer.isOpened()
    rng = np.random.default_rng(seed)
    scene = np.repeat(np.linspace(40, 215, 64)[None, :, None], 36, axis=0).repeat(3, axis=2)
    for i in range(frames):
        base = 255 - scene if cut_at is not None and i >= cut_at else scene
        writer.write(np.clip(base + rng.integers(-4, 5, base.shape), 0, 255).astype(np.uint8))
    writer.release()
    return path


def drop_frames(source: Path, target: Path, numbers: tuple[int, ...]) -> Path:
    expression = "+".join(f"eq(n,{n})" for n in numbers)
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(source), "-vf", f"select='not({expression})'", "-fps_mode", "passthrough",
                    "-c:v", "mpeg4", str(target)], check=True)
    return target


@pytest.fixture
def clip(tmp_path):
    return write_clip(tmp_path / "clip.mp4")


# --- timing rule ------------------------------------------------------------------------------------------------------

def test_grid_is_the_protocol_grid():
    block = re.search(r"```\n(.*?)\n```", PROTOCOL.read_text().split("## Siatka offsetów")[1], re.S).group(1)
    grid = tuple(int(token.replace("−", "-").replace("+", "")) for token in re.findall(r"[−+]?\d+", block))
    assert grid == GRID_MS and len(GRID_MS) == 30
    assert TOLERANCE_MS == 20


def test_regular_timeline_gives_exact_frames_and_zero_error():
    times = np.arange(200) * 40.0 - 4000
    selections = select_offsets(times)
    assert [s.offset_ms for s in selections] == list(GRID_MS)
    assert all(s.status == "ok" and s.decoded_ms == s.offset_ms for s in selections)
    assert [s.frame_index for s in selections][GRID_MS.index(0)] == 100
    assert anchor_frame_index(times) == 100


def test_jitter_inside_tolerance_is_kept_with_its_error():
    times = np.arange(200) * 40.0 - 4000 + 15
    assert all(s.status == "ok" and s.decoded_ms - s.offset_ms == 15 for s in select_offsets(times))
    shifted = select_offsets(times + 4)                       # 19 ms late: still inside, 21 ms from the neighbour
    assert all(s.status == "ok" and s.decoded_ms - s.offset_ms == 19 for s in shifted)


def test_gaps_are_missing_and_duplicates_or_ties_are_ambiguous():
    times = list(np.arange(200) * 40.0 - 4000)
    del times[95:97]                                # frames at -200 and -160 ms never decoded
    by_offset = {s.offset_ms: s for s in select_offsets(np.array(times))}
    assert by_offset[-200].status == "missing" and by_offset[-160].status == "missing"
    assert by_offset[-200].nearest_ms == -240 and by_offset[-240].status == "ok" and by_offset[-120].status == "ok"
    assert by_offset[-160].candidates == ()
    times = np.arange(200) * 40.0 - 4000
    tied = select_offsets(np.concatenate([times[:100], [times[100] + 10], times[100:]]))   # two frames near 0 ms
    assert {s.offset_ms: s.status for s in tied}[0] == "ambiguous"
    assert select_offsets(times + 20)[0].status == "ambiguous"                          # exactly 20 ms from two frames
    beyond = select_offsets(times[:120])
    assert [s.status for s in beyond if s.offset_ms >= 800] == ["missing", "missing"]


def test_offset_zero_is_the_anchor_frame_even_where_the_nearest_frame_is_another():
    times = np.arange(200) * 40.0 - 4000 + 10          # frames at -30 and +10 ms around the anchor
    anchor = anchor_selection(times)
    assert anchor.frame_index == 99 and anchor.decoded_ms == -30 and anchor.status == "anchor_off_tolerance"
    assert [s for s in select_offsets(times) if s.offset_ms == 0][0].frame_index == 100   # the plain rule would differ
    regular = anchor_selection(np.arange(200) * 40.0 - 4000)
    assert (regular.frame_index, regular.status, regular.decoded_ms) == (100, "ok", 0)
    with pytest.raises(ValueError, match="no frame at or before"):
        anchor_selection(np.array([40.0, 80.0]))


def test_only_25_fps_clips_are_accepted():
    check_frame_rate(np.arange(200) * 40.0 - 4000)
    check_frame_rate(np.arange(200) * 40.0 - 4000 + np.where(np.arange(200) % 7 == 0, 40, 0).cumsum())   # gaps keep the median step
    with pytest.raises(ValueError, match="Median frame step"):
        check_frame_rate(np.arange(100) * 80.0 - 4000)
    with pytest.raises(ValueError, match="Median frame step"):
        check_frame_rate(np.arange(400) * 20.0 - 4000)


def test_cut_span_counts_only_transitions_after_the_earlier_frame():
    changes = {k: 0.0 for k in range(51, 101)} | {80: .5}
    assert max_change_between(changes, 100, 79) == .5          # transition 79->80 lies between
    assert max_change_between(changes, 100, 80) == 0           # frame 80 is already past the cut
    assert max_change_between(changes, 100, 100) == 0          # the anchor frame itself
    assert max_change_between(changes | {101: .7}, 100, 101) == .7
    assert max_change_between(changes, 100, 60) == .5


# --- extraction on tiny clips -----------------------------------------------------------------------------------------

def test_extracted_frames_match_the_canonical_anchor_frame_and_timeline(clip):
    result = extract_offset_frames(clip)
    _, canonical = extract_inputs(clip, None)
    anchor = result.frames[0]
    assert result.anchor_index == canonical["t0"].index == 100 and not result.offset0_rule_differs
    assert np.array_equal(anchor.bgr, canonical["t0"].bgr) and array_digest(anchor.bgr) == array_digest(canonical["t0"].bgr)
    assert anchor.cut_between is False and anchor.cut_max_change == 0
    for offset, frame in result.frames.items():
        assert frame.selection.status == "ok"
        assert frame.selection.frame_index == 100 + offset // 40
        assert result.timeline.relative_ms(100 + offset // 40, 4000) == offset
        assert frame.bgr.shape == (36, 64, 3)


def test_short_clip_reports_missing_offsets_without_substitution(tmp_path):
    result = extract_offset_frames(write_clip(tmp_path / "short.mp4", frames=120))
    missing = [o for o, f in result.frames.items() if f.selection.status == "missing"]
    assert missing == [800, 1000]
    assert all(result.frames[o].bgr is None and result.frames[o].cut_between is None for o in missing)
    assert result.frames[600].selection.decoded_ms == 600


def test_frames_the_encoder_dropped_leave_missing_offsets(clip, tmp_path):
    result = extract_offset_frames(drop_frames(clip, tmp_path / "dropped.mp4", (95, 96)))
    assert {o: f.selection.status for o, f in result.frames.items() if f.selection.status != "ok"} == {-200: "missing", -160: "missing"}
    assert result.frames[-240].selection.decoded_ms == -240 and result.anchor_index == 98


def test_dropped_anchor_frame_keeps_offset_zero_on_the_canonical_frame_and_flags_it(clip, tmp_path):
    result = extract_offset_frames(drop_frames(clip, tmp_path / "no_anchor.mp4", (100,)))
    _, canonical = extract_inputs(tmp_path / "no_anchor.mp4", None)
    zero = result.frames[0]
    assert result.anchor_index == canonical["t0"].index and np.array_equal(zero.bgr, canonical["t0"].bgr)
    assert zero.selection.decoded_ms == -40 and zero.selection.status == "anchor_off_tolerance"
    assert result.offset0_rule_differs                      # the nearest-within-20-ms rule would have called it missing
    assert result.frames[40].selection.status == "ok"


def test_clip_with_another_frame_rate_is_refused(tmp_path):
    with pytest.raises(ValueError, match="Median frame step"):
        extract_offset_frames(write_clip(tmp_path / "slow.mp4", frames=100, fps=12.5))


def test_cut_threshold_is_the_quality_report_default():
    source = (Path(__file__).resolve().parents[1] / "scripts/report_data_quality.py").read_text()
    default = re.search(r'"--cut-threshold", type=float, default=([\d.]+)', source).group(1)
    assert CUT_THRESHOLD == float(default)


def test_cut_entering_the_earliest_frame_is_not_flagged_but_the_next_transition_is(tmp_path):
    entering = extract_offset_frames(write_clip(tmp_path / "enter.mp4", cut_at=60))      # change at the -1600 ms frame itself
    after = extract_offset_frames(write_clip(tmp_path / "after.mp4", cut_at=61))         # change one frame later (-1560 ms)
    assert not any(f.cut_between for f in entering.frames.values())                      # the change exists in the clip but is not between -1600 and the anchor
    assert after.frames[-1600].cut_between is True and not after.frames[-1400].cut_between
    window = window_changes(tmp_path / "enter.mp4", 4000, (-1600, 0))                    # the quality report's window does count it
    assert window.change.iloc[0] > .08 and window.anchor_relative_ms.iloc[0] == -1600


def test_nonzero_first_pts_changes_neither_anchor_nor_frames(clip, tmp_path):
    shifted = tmp_path / "shifted.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(clip), "-c", "copy", "-output_ts_offset", "0.04", str(shifted)], check=True)
    plain, moved = extract_offset_frames(clip), extract_offset_frames(shifted)
    assert moved.timeline.pts[0] > 0 and plain.timeline.pts[0] == 0                       # the first PTS really differs
    assert moved.anchor_index == plain.anchor_index == 100
    for offset in GRID_MS:
        assert moved.frames[offset].selection.decoded_ms == plain.frames[offset].selection.decoded_ms == offset
        assert np.array_equal(moved.frames[offset].bgr, plain.frames[offset].bgr)


def test_cut_flag_marks_frames_before_the_cut_and_matches_the_quality_measure(tmp_path):
    cut = write_clip(tmp_path / "cut.mp4", cut_at=80)           # transition between 79 and 80, i.e. at -800 ms
    result = extract_offset_frames(cut)
    flags = {o: f.cut_between for o, f in result.frames.items()}
    assert all(flags[o] for o in GRID_MS if o < -800)
    assert not any(flags[o] for o in GRID_MS if o >= -800)
    window = window_changes(cut, 4000, (-1560, 0))              # transitions after the first selected frame (-1600 ms)
    assert result.frames[-1600].cut_max_change == pytest.approx(window.change.max())
    assert result.frames[-1600].cut_max_change > .08
    plain = extract_offset_frames(write_clip(tmp_path / "plain.mp4"))
    assert plain.frames[-1600].cut_max_change < .08 and not any(f.cut_between for f in plain.frames.values())


# --- builder: dummy index of tiny clips -----------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def lenient_failure_limit(monkeypatch):
    """The dummy index has failing clips on purpose; one test restores the real limit."""
    monkeypatch.setattr(builder, "MAX_FAILED_FRACTION", 1.0)


@pytest.fixture
def dummy(tmp_path):
    clips = {"good": write_clip(tmp_path / "good.mp4"), "cut": write_clip(tmp_path / "cut.mp4", cut_at=80),
             "short": write_clip(tmp_path / "short.mp4", frames=120), "early": write_clip(tmp_path / "early.mp4"),
             "broken": tmp_path / "broken.mp4"}
    clips["broken"].write_bytes(b"not a video")
    rows = [dict(shot_id=f"dummy_{name}", game="g", split="train", label="Shots off target", goal=name == "short",
                 is_penalty=name == "cut", half=1, position_ms=2000 if name == "early" else 90000, clip_path=str(path))
            for name, path in clips.items()]
    index = tmp_path / "index.parquet"
    pd.DataFrame(rows).to_parquet(index)
    protocol = tmp_path / "protocol.md"
    protocol.write_text("approved protocol")
    return index, protocol, tmp_path / "out" / "offset_frames_test"


def run_build(dummy, storage="native_png", **kwargs):
    index, protocol, out = dummy
    builder.build(index, out, storage, workers=2, protocol=protocol, **kwargs)
    return out


def test_build_records_timing_cut_status_and_completion(dummy):
    out = run_build(dummy)
    frames, shots = pd.read_csv(out / "frames.csv"), pd.read_csv(out / "shots.csv")
    assert len(frames) == 5 * 30 and set(frames.groupby("shot_id").size()) == {30}
    good = frames[frames.shot_id.eq("dummy_good")]
    assert good.status.eq("ok").all() and (good.decoded_relative_ms == good.offset_ms).all()
    assert (good.timing_error_ms == 0).all() and (good.requested_clip_ms == 4000 + good.offset_ms).all()
    assert good.source_frame_index.tolist() == [100 + o // 40 for o in good.offset_ms]       # integers, not 100.0
    image = cv2.imread(str(out / good[good.offset_ms.eq(0)].frame_path.item()))
    assert image.shape == (36, 64, 3) and array_digest(image) == good[good.offset_ms.eq(0)].pixel_sha256.item()
    cut = frames[frames.shot_id.eq("dummy_cut")].set_index("offset_ms")
    assert cut.cut_between_anchor[-1000] and not cut.cut_between_anchor[-800] and not cut.cut_between_anchor[0]
    short = frames[frames.shot_id.eq("dummy_short")]
    assert short[short.status.eq("missing")].offset_ms.tolist() == [800, 1000]
    assert short[short.status.eq("missing")].frame_path.isna().all()
    assert frames[frames.shot_id.eq("dummy_early")].status.eq("decode_failed").all()
    assert frames[frames.shot_id.eq("dummy_broken")].failure.str.len().gt(0).all()
    shots = shots.set_index("shot_id")
    assert shots.complete_case.to_dict() == {"dummy_good": True, "dummy_cut": True, "dummy_short": False,
                                             "dummy_early": False, "dummy_broken": False}
    assert shots.protocol_population.to_dict() == {"dummy_good": True, "dummy_cut": False, "dummy_short": False,
                                                   "dummy_early": True, "dummy_broken": True}
    coverage = pd.read_csv(out / "coverage_by_offset.csv")
    row = coverage[coverage.population.eq("all_shots") & coverage.offset_ms.eq(800)].iloc[0]
    assert (row.ok, row.missing, row.decode_failed) == (2, 1, 2)
    summary = json.loads((out / "summary.json").read_text())
    assert summary["protocol_population"]["shots"] == 3 and summary["protocol_population"]["complete_cases"] == 1
    assert summary["protocol_population"]["stop_rule_triggered"] is True       # 2 of 3 excluded, far above 5%
    assert summary["missing_frames"] == 2 and summary["timing_error_ms"]["max_abs"] == 0
    assert summary["penalties"] == 1 and summary["goals"] == 1 and summary["extraction_failed_shots"] == 2
    assert summary["offset0_nearest_rule_would_differ_shots"] == 0 and summary["offset0_not_anchor_frame"] == 0
    manifest = json.loads((out / "complete.json").read_text())
    assert manifest["identity"]["protocol"]["sha256"] == builder.file_digest(dummy[1])
    assert manifest["identity"]["protocol"]["approval"]["commit"] == "41dfd12"
    assert manifest["identity"]["grid_ms"] == list(GRID_MS) and manifest["identity"]["storage"] == "native_png"
    assert set(manifest["identity"]["sources"]) >= {str(Path(p).resolve()) for p in shots.clip_path}
    assert verify_result(out)["files"] == manifest["files"]     # also readable by the canonical verifier
    assert not (out.parent / f".{out.name}.partial").exists()
    assert not (out / "shots").exists() and not (out / "identity.json").exists()


def test_stop_rule_flag_stays_false_when_the_population_is_complete(tmp_path):
    clips = [write_clip(tmp_path / f"c{i}.mp4") for i in range(2)]
    index = tmp_path / "index.parquet"
    pd.DataFrame([dict(shot_id=f"s{i}", game="g", split="train", label="Shots on target", goal=False, is_penalty=False,
                       half=1, position_ms=90000, clip_path=str(c)) for i, c in enumerate(clips)]).to_parquet(index)
    protocol = tmp_path / "p.md"
    protocol.write_text("p")
    builder.build(index, tmp_path / "out", "native_png", 2, protocol=protocol)
    summary = json.loads((tmp_path / "out/summary.json").read_text())
    assert summary["protocol_population"]["stop_rule_triggered"] is False
    assert summary["protocol_population"]["excluded_fraction"] == 0


def test_rerun_verifies_instead_of_rebuilding_and_detects_changes(dummy, capsys):
    out = run_build(dummy)
    frame = next((out / "frames").rglob("off_p0000.png"))
    before = frame.stat().st_mtime_ns
    capsys.readouterr()
    run_build(dummy)
    assert "Verified existing" in capsys.readouterr().out and frame.stat().st_mtime_ns == before
    original = frame.read_bytes()
    frame.write_bytes(original + b"x")
    with pytest.raises(ValueError, match="Output changed"):
        run_build(dummy)
    frame.write_bytes(original)
    run_build(dummy)
    frame.unlink()
    with pytest.raises(ValueError, match="Missing file"):
        run_build(dummy)
    frame.write_bytes(original)
    extra = out / "frames" / "stray.png"
    extra.write_bytes(b"unlisted")
    with pytest.raises(ValueError, match="not in the completion record"):
        run_build(dummy)
    extra.unlink()
    run_build(dummy)
    protocol = dummy[1]
    protocol.write_text("a different protocol")
    with pytest.raises(ValueError, match="changed"):
        run_build(dummy)
    protocol.write_text("approved protocol")
    with pytest.raises(ValueError, match="changed"):
        run_build(dummy, storage="model_input_png")


def test_interrupted_build_resumes_and_redoes_damaged_or_unfinished_shots(dummy):
    index, protocol, out = dummy
    shots = builder.select_shots(pd.read_parquet(index), None)
    identity = builder.build_identity(shots, index, protocol, "native_png", 2, None)
    partial = out.with_name(f".{out.name}.partial")
    partial.mkdir(parents=True)
    (partial / "identity.json").write_text(json.dumps(identity | {"storage": "model_input_png"}))
    with pytest.raises(ValueError, match="different inputs"):
        run_build(dummy)
    (partial / "identity.json").write_text(json.dumps(identity))
    for name in ("good", "cut"):
        builder.process_shot((shots[shots.shot_id.eq(f"dummy_{name}")].iloc[0].to_dict(), partial, "native_png"))
    cut_png = partial / builder.shot_folder("dummy_cut") / "off_m1000.png"
    cut_png.write_bytes(b"damaged after the record was written")
    stray = partial / builder.shot_folder("dummy_short") / "off_p1000.png"   # the short clip never has this frame
    stray.parent.mkdir(parents=True)
    stray.write_bytes(b"half written")
    assert builder.finished_shots(partial, 2) == {"dummy_good"}
    run_build(dummy)
    assert not partial.exists()
    frames = pd.read_csv(out / "frames.csv").set_index(["shot_id", "offset_ms"])
    assert len(frames) == 150
    assert builder.file_digest(out / frames.loc[("dummy_cut", -1000)].frame_path) == frames.loc[("dummy_cut", -1000)].file_sha256
    assert not (out / builder.shot_folder("dummy_short") / "off_p1000.png").exists()
    assert builder.file_digest(out / frames.loc[("dummy_good", 0)].frame_path) == frames.loc[("dummy_good", 0)].file_sha256


def test_failed_record_is_retried_on_resume(dummy):
    index, protocol, out = dummy
    shots = builder.select_shots(pd.read_parquet(index), None)
    partial = out.with_name(f".{out.name}.partial")
    partial.mkdir(parents=True)
    (partial / "identity.json").write_text(json.dumps(builder.build_identity(shots, index, protocol, "native_png", 2, None)))
    builder.process_shot((shots[shots.shot_id.eq("dummy_good")].iloc[0].to_dict(), partial, "native_png"))
    path = builder.record_path(partial, "dummy_good")
    record = json.loads(path.read_text())
    record["shot"].update(extraction_status="failed", extraction_error="MemoryError: transient")
    record["frames"] = [builder.empty_row("dummy_good", o, "decode_failed", "MemoryError: transient") for o in GRID_MS]
    path.write_text(json.dumps(record))
    assert builder.finished_shots(partial, 2) == set()          # a recorded failure is not a finished shot
    run_build(dummy)
    frames = pd.read_csv(out / "frames.csv")
    assert frames[frames.shot_id.eq("dummy_good")].status.eq("ok").all()


def test_finished_shot_is_not_reprocessed_on_resume(dummy):
    index, protocol, out = dummy
    shots = builder.select_shots(pd.read_parquet(index), None)
    partial = out.with_name(f".{out.name}.partial")
    partial.mkdir(parents=True)
    (partial / "identity.json").write_text(json.dumps(builder.build_identity(shots, index, protocol, "native_png", 2, None)))
    builder.process_shot((shots[shots.shot_id.eq("dummy_good")].iloc[0].to_dict(), partial, "native_png"))
    frame = partial / builder.shot_folder("dummy_good") / "off_p0000.png"
    marker = frame.stat().st_mtime_ns
    run_build(dummy)
    assert (out / builder.shot_folder("dummy_good") / "off_p0000.png").stat().st_mtime_ns == marker


def test_build_refuses_to_publish_when_too_many_clips_fail(dummy, monkeypatch):
    monkeypatch.setattr(builder, "MAX_FAILED_FRACTION", 0.05)
    out = dummy[2]
    with pytest.raises(ValueError, match="not published"):
        run_build(dummy)
    assert not out.exists() and out.with_name(f".{out.name}.partial").exists()


def test_worker_error_aborts_the_pool_instead_of_continuing(tmp_path):
    with pytest.raises(ValueError):
        builder.run_pool([(1, 2)] * 3, tmp_path, 2, 0)     # malformed jobs fail inside the worker


def test_refuses_to_start_when_the_estimate_leaves_too_little_disk(dummy):
    index, protocol, out = dummy
    with pytest.raises(OSError, match="not started"):
        builder.build(index, out, "native_png", 1, protocol=protocol, min_free=2**60)
    assert not out.exists()


def test_index_problems_are_reported_before_any_work(tmp_path):
    base = pd.DataFrame([dict(shot_id="s", game="g", split="t", label="x", goal=False, is_penalty=False, half=1,
                              position_ms=9000, clip_path="c.mp4")])
    with pytest.raises(ValueError, match="lacks columns"):
        builder.select_shots(base.drop(columns="goal"), None)
    with pytest.raises(ValueError, match="empty values"):
        builder.select_shots(base.assign(goal=None), None)
    with pytest.raises(ValueError, match="Duplicate"):
        builder.select_shots(pd.concat([base, base]), None)
    with pytest.raises(ValueError, match="--limit"):
        builder.select_shots(base, 5)
    assert len(builder.select_shots(pd.concat([base.assign(shot_id=f"s{i}") for i in range(4)]), 3)) == 3


def test_model_input_storage_is_the_resized_frame_and_equals_canonical_offset_zero(dummy):
    out = run_build(dummy, storage="model_input_png")
    frames = pd.read_csv(out / "frames.csv")
    row = frames[frames.shot_id.eq("dummy_good") & frames.offset_ms.eq(0)].iloc[0]
    stored = cv2.imread(str(out / row.frame_path))
    clip = Path(pd.read_parquet(dummy[0]).set_index("shot_id").clip_path["dummy_good"])
    _, canonical = extract_inputs(clip, None)
    assert stored.shape == (252, 448, 3) and (row.stored_width, row.stored_height) == (448, 252)
    assert np.array_equal(stored, cv2.resize(canonical["t0"].bgr, (448, 252), interpolation=cv2.INTER_AREA))
    assert row.pixel_sha256 == array_digest(canonical["t0"].bgr)       # provenance stays the native frame's hash


@pytest.mark.parametrize("storage", ["native_png", "model_input_png"])
def test_offset_zero_sanity_check_compares_stored_pixels(dummy, tmp_path, storage):
    out = run_build(dummy, storage=storage)
    clip = Path(pd.read_parquet(dummy[0]).set_index("shot_id").clip_path["dummy_good"])
    _, canonical = extract_inputs(clip, None)
    root = tmp_path / "canonical"
    write_png(root / "frames/r0_t0.png", canonical["t0"].bgr)
    frame = canonical["t0"]

    def table(index_shift=0):
        pd.DataFrame([{"shot_id": "dummy_good", "variant": "t0", "availability": "available", "frame_path": "frames/r0_t0.png",
                       "source_frame_index": frame.index + index_shift, "source_pts": frame.pts}]).to_csv(root / "frames.csv", index=False)

    table()
    result = compare(out, root)
    assert result["compared_shots"] == 1 and result["all_identical"] and result["same_stored_pixels"] == 1
    table(index_shift=-1)
    assert not compare(out, root)["all_identical"]
    table()
    other = canonical["t0"].bgr.copy()
    other[0, 0] ^= 255
    write_png(root / "frames/r0_t0.png", other)                    # same recorded hashes, different pixels
    result = compare(out, root)
    assert not result["all_identical"] and result["same_stored_pixels"] == 0


def test_sanity_check_without_common_shots_does_not_pass(dummy, tmp_path):
    out = run_build(dummy)
    root = tmp_path / "canonical"
    root.mkdir()
    pd.DataFrame([{"shot_id": "other", "variant": "t0", "availability": "available", "frame_path": "x.png",
                   "source_frame_index": 1, "source_pts": 1}]).to_csv(root / "frames.csv", index=False)
    result = compare(out, root)
    assert result["compared_shots"] == 0 and not result["all_identical"]
