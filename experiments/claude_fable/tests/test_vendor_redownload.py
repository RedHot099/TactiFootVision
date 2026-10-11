"""SoccerNet re-download for the secondary population (xg-vendor ticket 12) on a synthetic 25 fps half, no network.

The half is 60 s of 192x108 video whose every frame carries its own number as a 16-stripe barcode, so a test can read
back which frame a JPEG or a clip anchor really is. The downloader is a fake that copies the synthetic half; the real
SoccerNet adapter is tested with its library's `downloadFile` replaced (the sockets are blocked for the whole file).
"""

import hashlib
import io
import json
import logging
import math
import os
import pickle
import socket
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import collect_soccernet  # noqa: E402
import vendor_redownload as script  # noqa: E402
from xgpoc.data.canonical_frames import probe_video  # noqa: E402
from xgpoc.data.frames import extract_shot_frames  # noqa: E402
from xgpoc.data.offset_frames import anchor_frame_index  # noqa: E402
from xgpoc.vendor import redownload as rd  # noqa: E402
from xgpoc.vendor import soccernet_access as access  # noqa: E402
from xgpoc.vendor import target_guard as guard  # noqa: E402

GAME = "england_epl/2015-2016/2015-08-16 - 18-00 Alpha 2 - 1 Beta"
SECRET = "Pa55/w0rd+with spaces&odd=chars"
FPS, WIDTH, HEIGHT, SECONDS = 25, 192, 108, 60
BITS = 16


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("network access in a test")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


def barcode(number: int) -> np.ndarray:
    image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    for bit in range(BITS):
        if number >> bit & 1:
            image[:, bit * (WIDTH // BITS):(bit + 1) * (WIDTH // BITS)] = 255
    return image


def frame_number(image: np.ndarray) -> int:
    width = WIDTH // BITS
    return sum(int(image[:, bit * width:(bit + 1) * width].mean() > 127) << bit for bit in range(BITS))


@pytest.fixture(scope="session")
def half_video(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("synthetic") / "half.mkv"
    process = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-",
         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "12", "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
    for number in range(SECONDS * FPS):
        process.stdin.write(barcode(number).tobytes())
    process.stdin.close()
    assert process.wait() == 0
    return path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def target(position_ms: int, number: int = 0, game: str = GAME, half: int = 1, kind: str = "secondary", usable=True) -> dict:
    return {"kind": kind, "game": game, "half": half, "sb_match_id": 7, "sb_event_id": f"{game[-4:]}-{half}-{number:03d}-{position_ms}",
            "sb_timestamp_ms": position_ms - 500, "offset_ms": 500.0, "target_position_ms": position_ms,
            "anchor_convention": "statsbomb_time_plus_half_offset", "usable": usable, "reason_unusable": ""}


def make_job(half_video: Path, positions: tuple[int, ...], game: str = GAME, half: int = 1, expected: str | None = None) -> rd.HalfJob:
    asset = rd.Asset(half_video.stat().st_size, expected or sha(half_video), float(SECONDS))
    return rd.HalfJob(game, "train", half, asset, tuple(target(p, i, game, half) for i, p in enumerate(positions)))


class FakeDownloader:
    def __init__(self, source: Path):
        self.source, self.calls, self.error = source, [], None

    def download(self, game, split, half, dest):
        self.calls.append((game, split, half))
        if self.error:
            raise self.error
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.source.read_bytes())


def no_space_check(path, need):
    return None


@pytest.fixture
def layout(tmp_path) -> rd.Layout:
    return rd.Layout(tmp_path / "out", tmp_path / "halves")


def run_plan(plan, layout, downloader, keep=True, **kwargs):
    return rd.run(plan, layout, downloader, keep_halves=keep, attempts=2, pause_s=0, space=no_space_check, **kwargs)


def read_record(layout, job):
    return json.loads(layout.record_file(job).read_text())


# ------------------------------------------------------------------ frames and clips
def test_frames_are_the_requested_frames_through_the_index_path(half_video, layout):
    job = make_job(half_video, (10_000, 17_330))
    assert rd.process_half(job, layout, FakeDownloader(half_video), keep_halves=True, space=no_space_check) == "cut"
    rows = {r["target_position_ms"]: r for r in read_record(layout, job)["rows"]}
    for position, expected in ((10_000, 250), (17_330, 434)):   # 17.330 s falls between frames 433 and 434: the next one
        row = rows[position]
        assert (row["frame_status_0"], row["frame_status_-400"]) == ("ok", "ok")
        assert frame_number(cv2.imread(str(layout.partial / row["frame_path"]))) == expected
        assert frame_number(cv2.imread(str(layout.partial / row["frame_path_-400"]))) == math.ceil((position - 400) * FPS / 1000)
        assert row["reason"] == ""
        assert 0 <= row["grass_ratio"] <= 1


def test_frame_files_are_byte_identical_to_the_collectors_function_and_layout(half_video, layout, tmp_path):
    job = make_job(half_video, (10_000,))
    rd.process_half(job, layout, FakeDownloader(half_video), keep_halves=True, space=no_space_check)
    row = read_record(layout, job)["rows"][0]
    direct = extract_shot_frames(half_video, 10_000, row["shot_id"], tmp_path / "direct", offsets_ms=(-400, 0))
    assert (layout.partial / row["frame_path"]).read_bytes() == direct[0].read_bytes()
    assert (layout.partial / row["frame_path_-400"]).read_bytes() == direct[-400].read_bytes()
    # the index columns of scripts/collect_soccernet.py: offset 0 is `frame_path`, the others `frame_path_<+d>`
    assert {"frame_path_-400" if o == -400 else "frame_path" for o in rd.OFFSETS_MS} == {"frame_path_-400", "frame_path"}
    assert f"frame_path_{-400:+d}" == "frame_path_-400"
    assert set(rd.OFFSETS_MS) <= set(collect_soccernet.OFFSETS)
    assert Path(row["frame_path"]).name == "+0.jpg" and Path(row["frame_path_-400"]).name == "-400.jpg"


def clip_anchor_number(clip: Path) -> int:
    timeline = probe_video(clip)
    relative = np.array([timeline.relative_ms(i, rd.ANCHOR_MS) for i in range(len(timeline.pts))])
    index = anchor_frame_index(relative)
    capture = cv2.VideoCapture(str(clip))
    for _ in range(index + 1):
        ok, image = capture.read()
        assert ok
    capture.release()
    return frame_number(image)


def test_clip_is_8_seconds_with_the_anchor_frame_at_4000_ms(half_video, layout):
    job = make_job(half_video, (10_000, 17_330))
    rd.process_half(job, layout, FakeDownloader(half_video), keep_halves=True, space=no_space_check)
    for row in read_record(layout, job)["rows"]:
        clip = layout.partial / row["clip_path"]
        assert row["clip_status"] == "ok" and abs(row["clip_anchor_error_ms"]) <= 20
        assert abs(row["clip_frames"] - 8 * FPS) <= 1   # the grid-aligned cut loses a frame at the end
        assert clip_anchor_number(clip) == math.ceil(row["target_position_ms"] * FPS / 1000)


def test_cut_clip_command_is_the_collectors(half_video, tmp_path, monkeypatch):
    commands = []

    def capture(cmd, **kwargs):
        commands.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 1)

    for module in (collect_soccernet, rd):
        monkeypatch.setattr(module.subprocess, "run", capture)
    collect_soccernet.cut_clip(half_video, 17_330, tmp_path / "a.mp4")
    rd.cut_clip(half_video, 17_330, tmp_path / "b.mp4")
    assert [c if c != str(tmp_path / "a.mp4") else "OUT" for c in commands[0]] == [c if c != str(tmp_path / "b.mp4") else "OUT" for c in commands[1]]
    assert rd.CLIP_SPAN_S == 4.0 and rd.ANCHOR_MS == 4000


def test_edge_targets_get_a_reason_and_are_never_substituted(half_video, layout):
    job = make_job(half_video, (200, 2_000, 60_100))
    rd.process_half(job, layout, FakeDownloader(half_video), keep_halves=True, space=no_space_check)
    early, start, past_end = (r for r in sorted(read_record(layout, job)["rows"], key=lambda r: r["target_position_ms"]))
    assert (early["frame_status_0"], early["frame_status_-400"]) == ("ok", rd.BEFORE_START) and "frame_path_-400" not in early
    assert "frame -400 ms: before_half_start" in early["reason"]
    assert early["clip_status"] == rd.CLIP_NOT_AT_ANCHOR       # the clip starts at the half's beginning: anchor not at 4 s
    assert frame_number(cv2.imread(str(layout.partial / start["frame_path_-400"]))) == 40 and start["clip_status"] == rd.CLIP_NOT_AT_ANCHOR
    assert (past_end["frame_status_0"], past_end["frame_status_-400"]) == (rd.BEYOND_END, rd.OK)   # -400 ms is still inside
    assert "frame_path" not in past_end and past_end["clip_status"].startswith("clip_failed")


# ------------------------------------------------------------------ sha256, resume, keeping halves
def test_wrong_sha256_refuses_the_half_lists_it_and_cuts_nothing(half_video, layout):
    job = make_job(half_video, (10_000, 20_000), expected="0" * 64)
    downloader = FakeDownloader(half_video)
    assert rd.process_half(job, layout, downloader, keep_halves=True, space=no_space_check) == "refused"
    record = read_record(layout, job)
    assert record["half"]["outcome"] == rd.REFUSED and record["half"]["actual_sha256"] == sha(half_video)
    assert record["files"] == {} and {r["frame_status_0"] for r in record["rows"]} == {rd.HALF_REFUSED}
    assert not layout.half_file(job).exists() and not list((layout.cut / job.half_id).glob("clips"))
    assert rd.process_half(job, layout, downloader, keep_halves=True, space=no_space_check) == "skipped"   # refusal is a record


def test_refused_half_appears_in_the_index_and_the_completion_record(half_video, layout):
    bad = make_job(half_video, (10_000,), half=1, expected="f" * 64)
    good = make_job(half_video, (12_000,), half=2)
    plan = rd.Plan((bad, good), ())
    run_plan(plan, layout, FakeDownloader(half_video))
    summary = rd.publish(layout, plan, {"sources": {}}, {}, {}, {})
    index = pd.read_parquet(layout.out / "frames_index.parquet")
    assert summary["halves_refused"] == 1 and summary["refused_halves"][0]["actual_sha256"] == sha(half_video)
    assert summary["refused_halves"][0]["expected_sha256"] == "f" * 64
    assert json.loads((layout.out / "refused_halves.json").read_text()) == summary["refused_halves"]
    refused = index[index.half == 1].iloc[0]
    assert refused.frame_status_0 == rd.HALF_REFUSED and refused.frame_path == "" and refused.reason == rd.HALF_REFUSED


def test_a_half_with_only_targets_after_the_end_of_its_video_is_not_downloaded(half_video, layout):
    job = make_job(half_video, (60_500, 75_000), half=2)       # the half video lasts 60 s
    assert not job.needs_video
    downloader = FakeDownloader(half_video)
    assert rd.process_half(job, layout, downloader, keep_halves=True, space=no_space_check) == "no_video"
    record = read_record(layout, job)
    assert downloader.calls == [] and not layout.half_file(job).exists() and record["half"]["outcome"] == rd.NO_VIDEO
    assert {r["frame_status_0"] for r in record["rows"]} == {rd.BEYOND_END} and "60.0 s" in record["rows"][0]["reason"]
    assert rd.process_half(job, layout, downloader, keep_halves=True, space=no_space_check) == "skipped"
    plan = rd.Plan((job,), ())
    summary = rd.publish(layout, plan, {"sources": {}}, {}, {}, {})
    assert summary["halves_not_downloaded_beyond_video_end"] == 1 and summary["targets_without_both_frames"] == 2


def test_resume_skips_finished_halves_and_recuts_a_damaged_one(half_video, layout):
    jobs = [make_job(half_video, (10_000, 20_000), half=1), make_job(half_video, (30_000,), half=2)]
    plan, downloader = rd.Plan(tuple(jobs), ()), FakeDownloader(half_video)
    first = run_plan(plan, layout, downloader)
    assert (first.cut, first.skipped, len(downloader.calls)) == (2, 0, 2)
    again = run_plan(plan, layout, downloader)
    assert (again.cut, again.skipped, len(downloader.calls)) == (0, 2, 2)
    damaged = layout.partial / sorted(read_record(layout, jobs[0])["files"])[0]
    damaged.write_bytes(damaged.read_bytes()[:-5] + b"xxxxx")
    third = run_plan(plan, layout, downloader)
    assert (third.cut, third.skipped) == (1, 1)
    assert len(downloader.calls) == 2           # the kept half was verified and reused: no second download
    assert rd.record_is_intact(layout, jobs[0])


def test_delete_halves_removes_the_half_and_resume_downloads_again(half_video, layout):
    job = make_job(half_video, (10_000,))
    plan, downloader = rd.Plan((job,), ()), FakeDownloader(half_video)
    run_plan(plan, layout, downloader, keep=False)
    assert not layout.half_file(job).exists() and not list(layout.download_dir.glob("*.mkv"))
    assert read_record(layout, job)["half"]["kept"] is False
    next(iter((layout.partial / "cut" / job.half_id / "clips").glob("*.mp4"))).unlink()
    run_plan(plan, layout, downloader, keep=False)
    assert len(downloader.calls) == 2


def test_keep_halves_keeps_the_verified_file(half_video, layout):
    job = make_job(half_video, (10_000,))
    run_plan(rd.Plan((job,), ()), layout, FakeDownloader(half_video))
    assert sha(layout.half_file(job)) == job.asset.sha256 and read_record(layout, job)["half"]["kept"] is True


def test_a_kept_half_with_a_wrong_hash_is_downloaded_again_once(half_video, layout):
    job = make_job(half_video, (10_000,))
    layout.half_file(job).parent.mkdir(parents=True)
    layout.half_file(job).write_bytes(b"truncated")
    downloader = FakeDownloader(half_video)
    assert rd.process_half(job, layout, downloader, keep_halves=True, space=no_space_check) == "cut"
    assert len(downloader.calls) == 1 and sha(layout.half_file(job)) == job.asset.sha256


def test_failed_downloads_retry_then_three_in_a_row_stop_the_run(half_video, layout):
    jobs = [make_job(half_video, (10_000,), game=f"england_epl/2015-2016/g{i}") for i in range(5)]
    downloader = FakeDownloader(half_video)
    downloader.error = access.DownloadError("HTTP Error 503")
    result = run_plan(rd.Plan(tuple(jobs), ()), layout, downloader)
    assert result.stopped_early and len(result.failed) == 3 and len(downloader.calls) == 3 * 2   # two attempts each
    assert not [j for j in jobs if layout.record_file(j).exists()]
    downloader.error = None
    resumed = run_plan(rd.Plan(tuple(jobs), ()), layout, downloader)
    assert resumed.cut == 5 and not resumed.stopped_early


def test_max_halves_makes_a_pilot_and_publish_refuses_an_unfinished_run(half_video, layout):
    jobs = [make_job(half_video, (10_000,), half=h) for h in (1, 2)]
    plan = rd.Plan(tuple(jobs), ())
    pilot = run_plan(plan, layout, FakeDownloader(half_video), max_halves=1)
    assert pilot.cut == 1 and pilot.stopped_early
    with pytest.raises(rd.Incomplete):
        rd.publish(layout, plan, {"sources": {}}, {}, {}, {})


def test_low_space_stops_before_the_download(half_video, layout):
    job = make_job(half_video, (10_000,))
    downloader = FakeDownloader(half_video)

    def full(path, need):
        raise guard.LowSpace("full")
    with pytest.raises(guard.LowSpace):
        rd.run(rd.Plan((job,), ()), layout, downloader, keep_halves=True, attempts=1, pause_s=0, space=full)
    assert downloader.calls == []


# ------------------------------------------------------------------ index and completion record
def test_publish_writes_one_row_per_target_and_a_verifiable_record(half_video, layout):
    unusable = target(5_000, 9, game=GAME + "x", half=2, usable=False) | {"reason_unusable": "offset unknown"}
    jobs = [make_job(half_video, (10_000, 20_000), half=1), make_job(half_video, (30_000,), half=2, game=GAME + "b")]
    plan = rd.Plan(tuple(jobs), (unusable,))
    run_plan(plan, layout, FakeDownloader(half_video))
    identity = {"sources": {}}
    summary = rd.publish(layout, plan, identity, {"sha": "abc"}, {}, {})
    assert not layout.partial.exists() and summary["targets"] == 4 and summary["targets_with_both_frames"] == 3
    index = pd.read_parquet(layout.out / "frames_index.parquet")
    assert list(index.columns) == rd.INDEX_COLUMNS and index.sb_event_id.is_unique and len(index) == 4
    ok = index[index.frame_status_0 == "ok"]
    assert all(Path(p).is_absolute() and Path(p).is_file() and str(layout.out) in p for p in ok.frame_path)
    assert all(Path(p).is_file() for p in ok["frame_path_-400"]) and all(Path(p).is_file() for p in ok.clip_path)
    skipped = index[index.sb_event_id == unusable["sb_event_id"]].iloc[0]
    assert skipped.frame_status_0 == rd.TARGET_UNUSABLE and "offset unknown" in skipped.reason and skipped.frame_path == ""
    assert (index.anchor_convention == "statsbomb_time_plus_half_offset").all()
    record = rd.verify_complete(layout.out, identity)
    hashed = set(record["files"])
    assert {str(p.relative_to(layout.out)) for p in layout.out.rglob("*") if p.is_file()} - {"complete.json"} == hashed
    assert not any(n.startswith(".") for n in hashed) and "frames_index.parquet" in hashed
    next(layout.out.rglob("*.jpg")).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Output changed"):
        rd.verify_complete(layout.out, identity)


def test_published_version_with_changed_identity_is_rejected(half_video, layout):
    job = make_job(half_video, (10_000,))
    plan = rd.Plan((job,), ())
    run_plan(plan, layout, FakeDownloader(half_video))
    rd.publish(layout, plan, {"sources": {}, "v": 1}, {}, {}, {})
    with pytest.raises(ValueError, match="changed since"):
        rd.verify_complete(layout.out, {"sources": {}, "v": 2})


def test_publish_stops_when_a_source_changes_during_the_run(half_video, layout, tmp_path):
    source = tmp_path / "code.py"
    source.write_text("a")
    identity = {"sources": {str(source): sha(source)}}
    job = make_job(half_video, (10_000,))
    plan = rd.Plan((job,), ())
    run_plan(plan, layout, FakeDownloader(half_video))
    source.write_text("b")
    with pytest.raises(ValueError, match="source changed"):
        rd.publish(layout, plan, identity, {}, {}, {str(source): source})


# ------------------------------------------------------------------ plan and pairing record
def pairing_fixture(root: Path, half_video: Path, with_attempts=True) -> dict:
    root.mkdir(parents=True)
    targets = pd.DataFrame([target(10_000, 0), target(20_000, 1, kind="recovered_goal"), target(30_000, 2, half=2)])
    targets["sb_match_id"] = targets.sb_match_id.astype("Int64")
    targets.to_parquet(root / "cut_targets.parquet", index=False)
    pd.DataFrame({"game": [GAME, GAME], "half": [1, 2]}).to_csv(root / "halves_to_cut.csv", index=False)
    attempts = root.parent / "attempts_all.parquet"
    pd.DataFrame({"game": [GAME, GAME], "split": ["test", "test"]}).to_parquet(attempts, index=False)
    files = {n: sha(root / n) for n in ("cut_targets.parquet", "halves_to_cut.csv")}
    record = {"identity": {"version": rd.PAIRING_VERSION, "inputs": {"attempts": {"path": str(attempts), "sha256": sha(attempts)}}},
              "files": files}
    (root / "complete.json").write_text(json.dumps(record))
    assets = root.parent / "raw_assets.jsonl"
    lines = [{"game": GAME, "half": h, "component": "720p", "bytes": half_video.stat().st_size, "sha256": sha(half_video),
              "error": "", "duration_s": float(SECONDS)} for h in (1, 2)]
    assets.write_text("\n".join(json.dumps(r) for r in lines) + "\n")
    return {"pairing": root, "assets": assets, "attempts": attempts}


def test_verify_pairing_refuses_a_changed_output(half_video, tmp_path):
    paths = pairing_fixture(tmp_path / "pairing", half_video)
    targets, halves, record = rd.verify_pairing(paths["pairing"])
    assert len(targets) == 3 and len(halves) == 2 and record["complete_json_sha256"]
    (paths["pairing"] / "halves_to_cut.csv").write_text("game,half\n")
    with pytest.raises(ValueError, match="Pairing output changed"):
        rd.verify_pairing(paths["pairing"])


def test_plan_orders_halves_and_carries_split_asset_and_sorted_targets(half_video, tmp_path):
    paths = pairing_fixture(tmp_path / "pairing", half_video)
    plan, identity = script.load_inputs(paths["pairing"], paths["assets"])
    assert [(j.half, j.split, len(j.targets)) for j in plan.halves] == [(1, "test", 2), (2, "test", 1)]
    assert plan.targets == 3 and plan.halves[0].asset.sha256 == sha(half_video)
    assert [t["target_position_ms"] for t in plan.halves[0].targets] == [10_000, 20_000]
    assert identity["version"] == rd.VERSION and identity["sources"]


def test_plan_rejects_a_halves_list_that_disagrees_with_the_targets(half_video, tmp_path):
    paths = pairing_fixture(tmp_path / "pairing", half_video)
    targets, halves, _ = rd.verify_pairing(paths["pairing"])
    with pytest.raises(ValueError, match="does not match"):
        rd.build_plan(targets, halves.iloc[:1], rd.read_assets(paths["assets"]), {GAME: "test"})


def test_read_assets_rejects_conflicting_hashes_and_ignores_errors(tmp_path):
    good = {"game": "g", "half": 1, "component": "720p", "bytes": 5, "sha256": "a" * 64, "error": "", "duration_s": 1.0}
    path = tmp_path / "assets.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in (good, good | {"half": 2, "error": "download_incomplete"})) + "\n")
    assert list(rd.read_assets(path)) == [("g", 1)]
    path.write_text("\n".join(json.dumps(r) for r in (good, good | {"sha256": "b" * 64})) + "\n")
    with pytest.raises(ValueError, match="Conflicting"):
        rd.read_assets(path)


REAL_PAIRING, REAL_ASSETS = script.PAIRING, script.RAW_ASSETS


@pytest.mark.skipif(not (REAL_PAIRING / "complete.json").exists() or not REAL_ASSETS.exists(), reason="frozen pairing not built")
def test_real_pairing_plan_covers_228_halves_and_847_targets_with_recorded_hashes():
    plan, _ = script.load_inputs(REAL_PAIRING, REAL_ASSETS)
    assert len(plan.halves) == 228 and plan.targets == 847 and not plan.unusable
    assert all(len(j.asset.sha256) == 64 and j.asset.bytes > 5e8 and j.split in {"train", "valid", "test"} for j in plan.halves)
    # 147 targets lie in stoppage time after the end of the recorded half video; 19 halves hold nothing else
    assert sum(not j.in_video(t) for j in plan.halves for t in j.targets) == 147
    assert sum(j.needs_video for j in plan.halves) == 209 and 200e9 < plan.total_bytes < 215e9


# ------------------------------------------------------------------ target and space guard
def info(target="/mnt/storage", options=("rw", "noatime")):
    return lambda path: guard.MountInfo(target, "ntfs3", tuple(options))


def test_target_on_the_root_filesystem_is_refused(tmp_path):
    with pytest.raises(guard.TargetRefused, match="root filesystem"):
        guard.check_target(tmp_path, lookup=info("/"))
    assert not list(tmp_path.iterdir())


def test_read_only_mount_and_unwritable_directory_are_refused(tmp_path):
    with pytest.raises(guard.TargetRefused, match="read-only"):
        guard.check_target(tmp_path, lookup=info(options=("ro", "relatime")))
    if os.geteuid() != 0:
        locked = tmp_path / "locked"
        locked.mkdir(mode=0o500)
        try:
            with pytest.raises(guard.TargetRefused, match="not writable"):
                guard.check_target(locked, lookup=info())
        finally:
            locked.chmod(0o700)


def test_mounted_writable_target_passes_and_leaves_no_probe(tmp_path):
    made = tmp_path / "new" / "dir"
    assert guard.check_target(made, lookup=info()).target == "/mnt/storage"
    assert made.is_dir() and not list(made.iterdir())


def test_findmnt_places_a_path_and_the_root():
    assert guard.mount_info(Path("/")).target == "/"
    assert guard.mount_info(Path("/nonexistent/deep/path")).target == "/"      # nearest existing parent
    assert guard.mount_info(Path("/proc")).fstype == "proc"


def test_space_rules():
    gib = 2**30
    home = Path("/home")
    free = {Path("/t"): 100 * gib, home: 80 * gib}.get
    separate = lambda a, b: False       # noqa: E731
    guard.check_space(Path("/t"), 50 * gib, free=free, shares_device=separate)
    with pytest.raises(guard.LowSpace, match="needed"):
        guard.check_space(Path("/t"), 150 * gib, free=free, shares_device=separate)
    with pytest.raises(guard.LowSpace, match="floor"):
        guard.check_space(Path("/t"), 1, free={Path("/t"): 100 * gib, home: 4 * gib}.get, shares_device=separate)
    same_free = {Path("/t"): 10 * gib, home: 10 * gib}.get       # the floor applies to the target's own free space
    guard.check_space(Path("/t"), 4 * gib, free=same_free, shares_device=lambda a, b: True)
    with pytest.raises(guard.LowSpace, match="floor"):
        guard.check_space(Path("/t"), 6 * gib, free=same_free, shares_device=lambda a, b: True)


def test_preflight_refuses_a_target_that_is_not_mounted(half_video, layout, monkeypatch):
    monkeypatch.setattr(guard, "mount_info", info("/"))
    with pytest.raises(guard.TargetRefused):
        script.preflight(rd.Plan((make_job(half_video, (10_000,)),), ()), layout, keep=True)
    assert not layout.halves.exists()


# ------------------------------------------------------------------ password
def test_secret_hides_itself():
    secret = access.Secret(SECRET)
    assert SECRET not in repr(secret) and SECRET not in str(secret) and SECRET not in f"{secret}"
    assert secret.reveal() == SECRET
    with pytest.raises(TypeError):
        pickle.dumps(secret)
    with pytest.raises(AttributeError):
        secret.other = 1
    assert secret.scrub(f"a {SECRET} b {quote(SECRET)} c {quote(SECRET, safe='')}") == "a *** b *** c ***"
    with pytest.raises(access.PasswordError):
        access.Secret("")


def test_password_comes_from_the_environment_or_a_private_file(tmp_path):
    path = tmp_path / "password"
    assert access.password_source({}, path) is None
    with pytest.raises(access.PasswordError, match="does not exist"):
        access.load_password({}, path)
    path.write_text(SECRET + "\n")
    path.chmod(0o644)
    with pytest.raises(access.PasswordError, match="owner only") as shown:
        access.load_password({}, path)
    assert SECRET not in str(shown.value)
    path.chmod(0o600)
    assert access.load_password({}, path).reveal() == SECRET
    assert access.load_password({access.PASSWORD_ENV: "from-env"}, path).reveal() == "from-env"
    assert access.password_source({access.PASSWORD_ENV: "x"}, path) == "env" and access.password_source({}, path) == "file"
    path.write_text("\n")
    with pytest.raises(access.PasswordError, match="empty"):
        access.load_password({}, path)


def test_password_source_does_not_open_the_file(tmp_path, monkeypatch):
    path = tmp_path / "password"
    path.write_text(SECRET)

    def refuse(*args, **kwargs):
        raise AssertionError("the file was read")
    monkeypatch.setattr(Path, "read_text", refuse)
    monkeypatch.setattr(Path, "read_bytes", refuse)
    monkeypatch.setattr(Path, "open", refuse)
    assert access.password_source({}, path) == "file"


def test_log_filter_scrubs_messages_and_tracebacks():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(access.ScrubFilter(access.Secret(SECRET)))
    logger = logging.getLogger("xgv.test_scrub")
    logger.addHandler(handler)
    logger.propagate = False
    logger.warning("failed with %s and %r", SECRET, SECRET)
    try:
        raise RuntimeError(f"auth {SECRET} rejected")
    except RuntimeError:
        logger.exception("download error")
    assert SECRET not in stream.getvalue() and "***" in stream.getvalue() and "rejected" in stream.getvalue()


def adapter_library(monkeypatch, behaviour):
    """The real SoccerNet downloader class with only `downloadFile` replaced (it makes the HTTP request)."""
    library = pytest.importorskip("SoccerNet.Downloader")
    calls = []

    def download_file(self, path_local, path_owncloud, user=None, password=None, verbose=True):
        calls.append({"path_local": path_local, "url": path_owncloud, "user": user, "password": password})
        return behaviour(path_local)

    monkeypatch.setattr(library.SoccerNetDownloader, "downloadFile", download_file)
    monkeypatch.setattr(library, "report", library.report)   # restored after the adapter replaces it
    return library, calls


def test_adapter_requests_the_720p_half_with_the_password_and_places_the_file(half_video, tmp_path, monkeypatch):
    def write(path_local):
        os.makedirs(os.path.dirname(path_local), exist_ok=True)
        Path(path_local).write_bytes(half_video.read_bytes())
        return 0
    library, calls = adapter_library(monkeypatch, write)
    dest = tmp_path / "halves" / ".partial" / "x.mkv"
    access.SoccerNetHalfDownloader(access.Secret(SECRET), tmp_path / "staging").download(GAME, "train", 2, dest)
    assert sha(dest) == sha(half_video) and len(calls) == 1
    assert calls[0]["password"] == SECRET and calls[0]["user"] == "xNGfp1W3wPeVOmQ"
    assert calls[0]["url"].endswith("video_720p/england_epl/2015-2016/2015-08-16%20-%2018-00%20Alpha%202%20-%201%20Beta/2_720p.mkv")
    assert library.report("anything") is None and not list((tmp_path / "staging").iterdir())     # analytics ping off, staging clean


def test_adapter_errors_never_carry_the_password(tmp_path, monkeypatch):
    def explode(path_local):
        raise OSError(f"cannot fetch with {SECRET}")
    adapter_library(monkeypatch, explode)
    with pytest.raises(access.DownloadError) as shown:
        access.SoccerNetHalfDownloader(access.Secret(SECRET), tmp_path / "s").download(GAME, "train", 1, tmp_path / "d.mkv")
    error = shown.value
    assert SECRET not in str(error) and "OSError" in str(error)
    assert error.__cause__ is None and error.__context__ is None
    adapter_library(monkeypatch, lambda path_local: 1)       # HTTP error: no file, return code 1
    with pytest.raises(access.DownloadError, match="No file after"):
        access.SoccerNetHalfDownloader(access.Secret(SECRET), tmp_path / "s").download(GAME, "train", 1, tmp_path / "d.mkv")


# ------------------------------------------------------------------ the script
def run_script(argv, monkeypatch, tmp_path, half_video):
    root = tmp_path / "fx" / "pairing"
    paths = pairing_fixture(root, half_video) if not root.exists() else {"pairing": root, "assets": root.parent / "raw_assets.jsonl"}
    monkeypatch.setattr(guard, "mount_info", info())
    base = ["--pairing", str(paths["pairing"]), "--raw-assets", str(paths["assets"]), "--target", str(tmp_path / "halves"),
            "--out", str(tmp_path / "out")]
    return script.main(base + argv)


def test_dry_run_creates_nothing_and_builds_no_downloader_and_reads_no_password(half_video, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("dry run built a downloader or read the password")
    monkeypatch.setattr(access, "SoccerNetHalfDownloader", forbidden)
    monkeypatch.setattr(access, "load_password", forbidden)
    assert run_script(["--dry-run"], monkeypatch, tmp_path, half_video) == 0
    assert not (tmp_path / "halves").exists() and not (tmp_path / "out").exists()


def test_dry_run_summary_content(half_video, tmp_path, monkeypatch, capsys):
    run_script(["--dry-run"], monkeypatch, tmp_path, half_video)
    text = capsys.readouterr().out
    summary = json.loads(text[text.index("{"):])
    assert (summary["halves_to_download"], summary["targets"], summary["matches"]) == (2, 3, 1)
    assert summary["by_kind_inside_video"] == {"secondary": 2, "recovered_goal": 1} and summary["keep_halves"] is True
    assert summary["resume"]["to_do"] == 2 and summary["network"].startswith("none")
    assert summary["halves_dir_mount"]["mount_point"] == "/mnt/storage"


def test_an_unrelated_existing_out_folder_is_refused_before_any_download(half_video, tmp_path, monkeypatch):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "something").write_text("x")
    with pytest.raises(SystemExit, match="refusing to publish over"):
        run_script(["--owner-approved"], monkeypatch, tmp_path, half_video)


def test_the_download_needs_the_owner_flag(half_video, tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="--owner-approved"):
        run_script([], monkeypatch, tmp_path, half_video)
    assert not (tmp_path / "halves").exists()


def test_full_run_through_the_script_publishes_and_never_logs_the_password(half_video, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(access.PASSWORD_ENV, SECRET)
    fake = FakeDownloader(half_video)
    monkeypatch.setattr(access, "SoccerNetHalfDownloader", lambda secret, staging: fake)
    monkeypatch.setattr(guard, "check_space", no_space_check)
    log_file = tmp_path / "run.log"
    assert run_script(["--owner-approved", "--log", str(log_file)], monkeypatch, tmp_path, half_video) == 0
    out = tmp_path / "out"
    assert (out / "complete.json").is_file() and not (tmp_path / "out.partial").exists()
    assert len(fake.calls) == 2 and sha(tmp_path / "halves" / GAME / "1_720p.mkv") == sha(half_video)   # halves kept by default
    everything = capsys.readouterr().out.encode() + log_file.read_bytes() + b"".join(p.read_bytes() for p in out.rglob("*") if p.is_file())
    assert SECRET.encode() not in everything and b"ETA" in log_file.read_bytes()
    assert run_script(["--owner-approved"], monkeypatch, tmp_path, half_video) == 0       # a finished version is verified, not rebuilt
    assert len(fake.calls) == 2


def test_script_exit_code_for_an_unfinished_run(half_video, tmp_path, monkeypatch):
    monkeypatch.setenv(access.PASSWORD_ENV, SECRET)
    fake = FakeDownloader(half_video)
    fake.error = access.DownloadError("HTTP Error 503")
    monkeypatch.setattr(access, "SoccerNetHalfDownloader", lambda secret, staging: fake)
    monkeypatch.setattr(guard, "check_space", no_space_check)
    assert run_script(["--owner-approved", "--attempts", "1"], monkeypatch, tmp_path, half_video) == 2
    assert not (tmp_path / "out").exists() and (tmp_path / "out.partial").exists()


def test_pilot_run_stops_after_the_requested_halves(half_video, tmp_path, monkeypatch):
    monkeypatch.setenv(access.PASSWORD_ENV, SECRET)
    fake = FakeDownloader(half_video)
    monkeypatch.setattr(access, "SoccerNetHalfDownloader", lambda secret, staging: fake)
    monkeypatch.setattr(guard, "check_space", no_space_check)
    assert run_script(["--owner-approved", "--max-halves", "1"], monkeypatch, tmp_path, half_video) == 2
    assert len(fake.calls) == 1 and not (tmp_path / "out").exists()
