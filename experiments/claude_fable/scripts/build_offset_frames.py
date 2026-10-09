"""Build the offset-frames data version: frames at the approved offset grid for every shot of the index.

Frames come from the canonical decode path (`xgpoc.data.offset_frames`, on top of `canonical_frames`). Each frame row
records the decoded time relative to the anchor, the timing error against the requested offset and a camera-cut flag.
Missing or ambiguous frames are counted per offset and never replaced. The build is resumable per shot (a shot is
done when its record file exists and its PNGs match the recorded hashes) and publishes atomically with a completion
record that holds source hashes, the approved protocol's hash and a hash of every output file. Re-running with the
same inputs verifies instead of rebuilding.
"""

import argparse
import fcntl
import hashlib
import json
import multiprocessing
import os
import shutil
import sys
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from xgpoc.data.canonical_frames import array_digest, runtime_identity
from xgpoc.data.offset_frames import (
    ANCHOR_MS,
    CUT_THRESHOLD,
    FRAME_STEP_MS,
    GRID_MS,
    TOLERANCE_MS,
    extract_offset_frames,
)
from xgpoc.paths import INDEX

ROOT = Path(__file__).resolve().parents[1]   # experiments/claude_fable, independent of the working directory
VERSION = "offset_frames_v1"
MIN_FREE_BYTES = 5 * 2**30
MAX_FAILED_FRACTION = 0.05                   # more failed clips than this and the build refuses to publish
MODEL_INPUT_SIZE = (448, 252)
FORMATS = {"native_png": None, "model_input_png": MODEL_INPUT_SIZE}
# Mean PNG size measured on 10 canonical frames and on a 200-shot pilot; only used for the refuse-to-start estimate.
BYTES_PER_FRAME_ESTIMATE = {"native_png": 860_000, "model_input_png": 170_000}
PROTOCOL = ROOT / "research/xgc_direction_probe_protocol.md"
# Owner approval of the protocol: commit 41dfd12, 2026-10-09, recorded in the Comments of ticket 06 (xg-credibility).
PROTOCOL_APPROVAL = {"commit": "41dfd12", "approved_on": "2026-10-09", "record": "ticket 06 (xg-credibility), Comments"}
EXCLUSION_STOP_FRACTION = 0.05               # protocol: more than 5% of the population excluded stops the analysis
RESUME_ENTRIES = ("shots", "identity.json", ".lock")
INDEX_COLUMNS = ["shot_id", "game", "half", "split", "label", "goal", "is_penalty", "position_ms", "clip_path"]
SHOT_COLUMNS = INDEX_COLUMNS + [
    "extraction_status", "extraction_error", "offset0_nearest_rule_differs", "anchor_frame_index", "anchor_relative_ms",
    "frame_count", "width", "height", "time_base", "codec", "pixel_format"]
FRAME_COLUMNS = [
    "shot_id", "offset_ms", "requested_clip_ms", "status", "decoded_relative_ms", "timing_error_ms", "frame_path",
    "source_frame_index", "source_pts", "time_base", "nearest_decoded_ms", "candidate_frame_indices",
    "stored_width", "stored_height", "file_bytes", "file_sha256", "pixel_sha256",
    "cut_between_anchor", "cut_max_change", "failure",
]
FLOAT_COLUMNS = ["decoded_relative_ms", "timing_error_ms", "nearest_decoded_ms", "cut_max_change"]
INTEGER_COLUMNS = ["source_frame_index", "source_pts", "stored_width", "stored_height", "file_bytes"]
FRAME_STATUSES = ("ok", "missing", "ambiguous", "anchor_off_tolerance", "decode_failed")


def json_text(value) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(2**20):
            h.update(block)
    return h.hexdigest()


def offset_name(offset_ms: int) -> str:
    return f"off_{'m' if offset_ms < 0 else 'p'}{abs(offset_ms):04d}"


def shard(shot_id: str) -> str:
    return hashlib.sha1(shot_id.encode()).hexdigest()[:2]


def shot_folder(shot_id: str) -> str:
    return f"frames/{shard(shot_id)}/{shot_id}"


def record_path(root: Path, shot_id: str) -> Path:
    return root / "shots" / shard(shot_id) / f"{shot_id}.json"


def encode_png(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok or not np.array_equal(cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED), image):
        raise ValueError("PNG roundtrip changed pixels")
    return buffer.tobytes()


def write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    temporary.rename(path)


def empty_row(shot_id: str, offset: int, status: str, failure: str = "") -> dict:
    return dict.fromkeys(FRAME_COLUMNS) | {
        "shot_id": shot_id, "offset_ms": offset, "requested_clip_ms": ANCHOR_MS + offset, "status": status, "failure": failure}


def plain(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialise {type(value)}")


def process_shot(job: tuple) -> dict:
    """Worker: decode one clip, write its frames, then write the shot record last (the resume marker).

    Anything wrong while reading the clip becomes a recorded `decode_failed` shot, which a resume retries. Failing to
    write (disk full, ...) is not a property of the data and propagates."""
    shot, root, storage = job
    shot_id = shot["shot_id"]
    folder = root / shot_folder(shot_id)
    if folder.exists():
        shutil.rmtree(folder)  # leftovers of an interrupted run
    record = {"shot": {k: shot[k] for k in INDEX_COLUMNS} | {"extraction_status": "ok", "extraction_error": ""}, "frames": []}
    try:
        if shot["position_ms"] < ANCHOR_MS:
            raise ValueError("Clip starts at the half's beginning: anchor is not at 4000 ms")
        result = extract_offset_frames(Path(shot["clip_path"]))
    except Exception as exc:  # noqa: BLE001 - any failure to read this clip is a data property, reported per shot
        failure = f"{type(exc).__name__}: {exc}"
        record["shot"].update(extraction_status="failed", extraction_error=failure)
        record["frames"] = [empty_row(shot_id, o, "decode_failed", failure) for o in GRID_MS]
    else:
        timeline = result.timeline
        record["shot"].update(
            anchor_frame_index=result.anchor_index, anchor_relative_ms=timeline.relative_ms(result.anchor_index, ANCHOR_MS),
            offset0_nearest_rule_differs=result.offset0_rule_differs, frame_count=len(timeline.pts), width=timeline.width,
            height=timeline.height, time_base=timeline.time_base, codec=timeline.codec, pixel_format=timeline.pixel_format)
        for offset, frame in result.frames.items():
            selection = frame.selection
            row = empty_row(shot_id, offset, selection.status) | {
                "nearest_decoded_ms": selection.nearest_ms,
                "candidate_frame_indices": "|".join(map(str, selection.candidates))}
            if frame.bgr is not None:
                index = selection.frame_index
                stored = frame.bgr if FORMATS[storage] is None else cv2.resize(frame.bgr, FORMATS[storage], interpolation=cv2.INTER_AREA)
                relative = f"{shot_folder(shot_id)}/{offset_name(offset)}.png"
                data = encode_png(stored)
                write_bytes_atomic(root / relative, data)
                row.update(decoded_relative_ms=selection.decoded_ms, timing_error_ms=round(selection.decoded_ms - offset, 6),
                           frame_path=relative, source_frame_index=index, source_pts=timeline.pts[index],
                           time_base=timeline.time_base, stored_width=stored.shape[1], stored_height=stored.shape[0],
                           file_bytes=len(data), file_sha256=file_digest(root / relative),   # from disk, not from memory
                           pixel_sha256=array_digest(frame.bgr), cut_between_anchor=bool(frame.cut_between),
                           cut_max_change=frame.cut_max_change)
            record["frames"].append(row)
    write_bytes_atomic(record_path(root, shot_id), json.dumps(record, default=plain).encode())
    return record


def init_worker() -> None:
    cv2.setNumThreads(1)
    os.nice(10)


def parallel_digests(paths: list[Path], workers: int) -> list[str]:
    """SHA-256 of each file; a file that is absent is reported as ValueError, not as an OS error."""
    def one(path: Path) -> str:
        if not path.is_file():
            raise ValueError(f"Missing file: {path}")
        return file_digest(path)
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, paths))


def record_is_intact(root: Path, record_file: Path) -> str | None:
    """The shot id of a finished shot whose recorded PNGs are all present and unchanged on disk, else None.

    A shot recorded as failed is not finished: the failure may have been transient (memory, file descriptors,
    a killed ffprobe), so a resume retries it. A data fault fails again the same way and is recorded again."""
    try:
        record = json.loads(record_file.read_text())
        if record["shot"]["extraction_status"] != "ok":
            return None
        for row in record["frames"]:
            if row["frame_path"] and (not (root / row["frame_path"]).is_file() or file_digest(root / row["frame_path"]) != row["file_sha256"]):
                return None
        return record["shot"]["shot_id"]
    except (OSError, ValueError, KeyError):
        return None


def finished_shots(partial: Path, workers: int) -> set[str]:
    files = sorted((partial / "shots").rglob("*.json")) if (partial / "shots").exists() else []
    with ThreadPoolExecutor(workers) as pool:
        return {s for s in pool.map(lambda f: record_is_intact(partial, f), files) if s}


def select_shots(index: pd.DataFrame, limit: int | None, only: list[str] | None = None) -> pd.DataFrame:
    missing = [c for c in INDEX_COLUMNS if c not in index.columns]
    if missing:
        raise ValueError(f"Index lacks columns {missing}")
    if index.shot_id.duplicated().any():
        raise ValueError("Duplicate shot ids in the index")
    if index[INDEX_COLUMNS].isna().any().any():
        raise ValueError("Index has empty values in the columns the build uses")
    shots = index.sort_values("shot_id").reset_index(drop=True)
    if only is not None:
        if unknown := set(only) - set(shots.shot_id):
            raise ValueError(f"{len(unknown)} requested shots are not in the index")
        shots = shots[shots.shot_id.isin(only)].reset_index(drop=True)
    if limit is not None:
        if not 0 < limit <= len(shots):
            raise ValueError(f"--limit must be between 1 and {len(shots)}")
        shots = shots.iloc[np.unique(np.linspace(0, len(shots) - 1, limit).round().astype(int))]
    if shots.empty:
        raise ValueError("No shots selected")
    return shots[INDEX_COLUMNS].reset_index(drop=True)


def build_identity(shots: pd.DataFrame, index_path: Path, protocol: Path, storage: str, workers: int,
                   limit: int | None) -> dict:
    code = [Path(__file__).resolve(), ROOT / "xgpoc/data/offset_frames.py", ROOT / "xgpoc/data/canonical_frames.py",
            index_path.resolve(), protocol.resolve()]
    clips = [Path(p).resolve() for p in shots.clip_path]
    hashes = parallel_digests(code + clips, workers)
    return {"version": VERSION, "canonical_extractor": "canonical_clip_v1", "anchor_ms": ANCHOR_MS,
            "protocol": {"path": str(protocol.resolve()), "sha256": file_digest(protocol), "approval": PROTOCOL_APPROVAL},
            "grid_ms": list(GRID_MS), "tolerance_ms": TOLERANCE_MS, "frame_step_ms": FRAME_STEP_MS,
            "frame_rule": "offset 0 is the anchor frame (latest frame <= 0, as canonical t0); other offsets take the decoded "
                          "frame within tolerance of the nominal time; none or several -> missing/ambiguous, never substituted",
            "cut_rule": {"measure": "mean abs change of consecutive 64x36 grey INTER_AREA thumbnails, as report_data_quality.window_changes",
                         "threshold": CUT_THRESHOLD,
                         "flag": "any transition between the anchor frame (latest frame <= 0) and the offset frame reaches the threshold"},
            "storage": storage, "storage_detail": "PNG compression 3, " + (
                "native BGR as decoded, no resize" if FORMATS[storage] is None else
                "BGR resized to 448x252 with INTER_AREA, the model input before the overlay mask"),
            "selection": {"shots": len(shots), "limit": limit, "shot_ids_sha256": hashlib.sha256("\n".join(shots.shot_id).encode()).hexdigest()},
            "runtime": runtime_identity(),
            "sources": {str(p): h for p, h in zip(code + clips, hashes, strict=True)}}


def stats(values: pd.Series) -> dict:
    return {} if values.empty else {"mean": float(values.mean()), "std": float(values.std(ddof=0)), "min": float(values.min()),
                                    "max": float(values.max()), "max_abs": float(values.abs().max())}


def coverage_tables(shots: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    population = frames.shot_id.map(shots.set_index("shot_id").protocol_population)
    rows = []
    for name, subset in (("all_shots", frames), ("protocol_population", frames[population.astype(bool)])):
        for offset, group in subset.groupby("offset_ms"):
            ok = group[group.status.eq("ok")]
            rows.append({"population": name, "offset_ms": offset, "shots": len(group),
                         **{status: int(group.status.eq(status).sum()) for status in FRAME_STATUSES},
                         "excluded": int(group.status.ne("ok").sum()),
                         "timing_error_mean_ms": ok.timing_error_ms.mean(), "timing_error_max_abs_ms": ok.timing_error_ms.abs().max(),
                         "cut_flagged": int(ok.cut_between_anchor.sum()),
                         "cut_rate": float(ok.cut_between_anchor.mean()) if len(ok) else np.nan})
    return pd.DataFrame(rows)


def assemble(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    records = [json.loads(p.read_text()) for p in sorted((root / "shots").rglob("*.json"))]
    shots = pd.DataFrame([r["shot"] for r in records]).reindex(columns=SHOT_COLUMNS).sort_values("shot_id").reset_index(drop=True)
    frames = pd.DataFrame([row for r in records for row in r["frames"]], columns=FRAME_COLUMNS)
    frames = frames.sort_values(["shot_id", "offset_ms"]).reset_index(drop=True)
    frames = frames.astype({c: "Int64" for c in INTEGER_COLUMNS} | {c: float for c in FLOAT_COLUMNS} | {"cut_between_anchor": "boolean"})
    # The probe population: shots without a goal and without penalties (research/xgc_direction_probe_protocol.md).
    shots["protocol_population"] = ~shots.goal.astype(bool) & ~shots.is_penalty.astype(bool)
    complete_case = frames.groupby("shot_id").status.apply(lambda s: bool(len(s) == len(GRID_MS) and s.eq("ok").all()))
    shots["complete_case"] = shots.shot_id.map(complete_case)
    offset0 = frames[frames.offset_ms.eq(0) & frames.source_frame_index.notna()].set_index("shot_id").source_frame_index
    shots["offset0_is_anchor_frame"] = (shots.shot_id.map(offset0).astype(float).eq(shots.anchor_frame_index.astype(float))
                                        .where(shots.shot_id.isin(offset0.index)))
    coverage = coverage_tables(shots, frames)
    ok = frames[frames.status.eq("ok")]
    population = shots[shots.protocol_population]
    excluded_fraction = float((~population.complete_case).mean()) if len(population) else None
    failed = shots.extraction_status.eq("failed")
    summary = {
        "version": VERSION, "shots": len(shots), "frames_requested": len(frames), "frames_ok": int(len(ok)),
        "extraction_failed_shots": int(failed.sum()),
        "frame_rate_refused_shots": int(shots.extraction_error.str.startswith("ValueError: Median frame step").sum()),
        **{f"{status}_frames": int(frames.status.eq(status).sum()) for status in FRAME_STATUSES if status != "ok"},
        "protocol_population": {
            "shots": len(population), "complete_cases": int(population.complete_case.sum()),
            "excluded_shots": int((~population.complete_case).sum()), "excluded_fraction": excluded_fraction,
            "stop_rule_threshold": EXCLUSION_STOP_FRACTION,
            "stop_rule_triggered": bool(excluded_fraction is not None and excluded_fraction > EXCLUSION_STOP_FRACTION),
            "stop_rule": "protocol: more than 5% excluded stops the analysis and returns to the owner"},
        "all_shots_complete_cases": int(shots.complete_case.sum()),
        "penalties": int(shots.is_penalty.sum()), "goals": int(shots.goal.sum()),
        "offset0_not_anchor_frame": int(shots.offset0_is_anchor_frame.eq(False).sum()),
        "offset0_nearest_rule_would_differ_shots": int(shots.offset0_nearest_rule_differs.eq(True).sum()),
        "timing_error_ms": stats(ok.timing_error_ms.astype(float)),
        "cut_flag_rate_ok_frames": float(ok.cut_between_anchor.mean()) if len(ok) else None,
        "shots_with_any_cut_flag": int(ok.groupby("shot_id").cut_between_anchor.any().sum()),
        "stored_bytes": int(frames.file_bytes.sum()), "stored_mean_bytes_per_frame": float(frames.file_bytes.mean()) if frames.file_bytes.notna().any() else None,
    }
    return shots, frames, coverage, summary


def complete(stage: Path, out: Path, identity: dict, frames: pd.DataFrame, workers: int) -> None:
    """Completion record. Frame hashes are those read back from disk when each frame was written; every other
    output file is hashed here. The resume entries stay in place until the rename has succeeded."""
    paths = sorted(p for p in stage.iterdir() if p.is_file() and p.name not in RESUME_ENTRIES)
    files = {str(p.relative_to(stage)): h for p, h in zip(paths, parallel_digests(paths, workers), strict=True)}
    files |= dict(zip(frames.frame_path.dropna(), frames.file_sha256.dropna(), strict=True))
    names = list(identity["sources"])
    if dict(zip(names, parallel_digests([Path(n) for n in names], workers), strict=True)) != identity["sources"]:
        raise ValueError("A source changed during the build")
    (stage / "complete.json").write_text(json_text({"identity": identity, "files": dict(sorted(files.items()))}))
    stage.rename(out)
    clear_resume_entries(out)


def clear_resume_entries(root: Path) -> None:
    shutil.rmtree(root / "shots", ignore_errors=True)
    for name in RESUME_ENTRIES[1:]:
        (root / name).unlink(missing_ok=True)


def verify_complete(root: Path, identity: dict, workers: int) -> dict:
    """Identity must equal the freshly hashed one (so sources are unchanged); every listed output is rehashed
    from disk, and files that the record does not list are an error."""
    manifest = json.loads((root / "complete.json").read_text())
    if manifest["identity"] != identity:
        raise ValueError("Data inputs, protocol or runtime changed; use a new result directory")
    names = list(manifest["files"])
    present = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()} - {"complete.json"}
    if unlisted := sorted(present - set(names)):
        raise ValueError(f"{len(unlisted)} files are not in the completion record, e.g. {unlisted[0]}")
    for name, found in zip(names, parallel_digests([root / n for n in names], workers), strict=True):
        if found != manifest["files"][name]:
            raise ValueError(f"Output changed: {name}")
    return manifest


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def run_pool(todo: list[tuple], partial: Path, workers: int, min_free: int) -> None:
    started, failed, finished = time.time(), 0, 0
    # spawn: the parent may hold threads (hash pool, OpenCV), and forking such a process can deadlock
    pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"), initializer=init_worker)
    try:
        pending = {pool.submit(process_shot, job) for job in todo}
        while pending:
            ready, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in ready:
                failed += future.result()["shot"]["extraction_status"] == "failed"
                finished += 1
            if finished % 100 < len(ready):
                rate = finished / (time.time() - started)
                print(f"{finished}/{len(todo)} shots, {failed} failed, {rate:.2f} shots/s, "
                      f"eta {(len(todo) - finished) / rate / 60:.0f} min, free {free_bytes(partial) / 2**30:.1f} GiB", flush=True)
            if free_bytes(partial) < min_free:
                raise OSError("Free disk fell under the floor; build stopped, rerun to resume")
            if finished >= 100 and failed / finished > MAX_FAILED_FRACTION:
                raise ValueError(f"{failed} of {finished} clips failed; build stopped (is ffmpeg/ffprobe working?)")
    except BaseException:
        pool.shutdown(wait=True, cancel_futures=True)   # no queued shot may keep writing after an abort
        raise
    pool.shutdown()


def build(index_path: Path, out: Path, storage: str, workers: int, limit: int | None = None,
          protocol: Path = PROTOCOL, min_free: int = MIN_FREE_BYTES, only: list[str] | None = None) -> None:
    if storage not in FORMATS:
        raise ValueError(f"storage must be one of {sorted(FORMATS)}")
    if not all(shutil.which(tool) for tool in ("ffmpeg", "ffprobe")):
        raise OSError("ffmpeg and ffprobe are required")
    shots = select_shots(pd.read_parquet(index_path), limit, only)
    out.parent.mkdir(parents=True, exist_ok=True)
    identity = build_identity(shots, index_path, protocol, storage, workers, limit)
    if out.exists():
        clear_resume_entries(out)   # leftovers of a crash right after publishing
        verify_complete(out, identity, workers)
        print("Verified existing offset frames; nothing rebuilt")
        return
    partial = out.with_name(f".{out.name}.partial")
    marker = partial / "identity.json"
    if partial.exists():
        if not marker.exists() or json.loads(marker.read_text()) != identity:
            raise ValueError(f"{partial} belongs to different inputs; remove it or use another --out")
    else:
        partial.mkdir()
        marker.write_text(json_text(identity))
    lock = os.open(partial / ".lock", os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise OSError(f"Another build is running in {partial}") from None
    done = finished_shots(partial, workers)
    todo = [(row, partial, storage) for row in shots.to_dict("records") if row["shot_id"] not in done]
    estimate = len(todo) * len(GRID_MS) * BYTES_PER_FRAME_ESTIMATE[storage]
    if free_bytes(partial) - estimate < min_free:
        raise OSError(f"Estimated {estimate / 2**30:.1f} GiB would leave under {min_free / 2**30:.0f} GiB free; not started "
                      f"(rerun resumes; {partial} keeps the finished shots)")
    print(f"{len(shots)} shots, {len(done)} already done, {len(todo)} to build with {workers} workers", flush=True)
    started = time.time()
    run_pool(todo, partial, workers, min_free)
    table_shots, frames, coverage, summary = assemble(partial)
    if len(table_shots) != len(shots):
        raise ValueError("Shot records missing after build")
    if summary["extraction_failed_shots"] > MAX_FAILED_FRACTION * len(shots):
        raise ValueError(f"{summary['extraction_failed_shots']} of {len(shots)} clips failed; not published")
    summary["build_seconds_this_run"] = round(time.time() - started, 1)
    table_shots.to_csv(partial / "shots.csv", index=False)
    frames.to_csv(partial / "frames.csv", index=False)
    coverage.to_csv(partial / "coverage_by_offset.csv", index=False)
    (partial / "summary.json").write_text(json_text(summary))
    complete(partial, out, identity, frames, workers)
    print(json_text(summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--out", type=Path, default=ROOT / "cache/xgc/offset_frames_v1")
    parser.add_argument("--storage", choices=sorted(FORMATS), required=True,
                        help="native_png is lossless at clip resolution; model_input_png is the lossless 448x252 model input")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 4))
    parser.add_argument("--limit", type=int, help="Evenly spaced sample of this many shots (pilot)")
    parser.add_argument("--shots-csv", type=Path, help="Build only the shots listed in the shot_id column (sanity checks)")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = parser.parse_args()
    only = pd.read_csv(args.shots_csv).shot_id.drop_duplicates().tolist() if args.shots_csv else None
    build(args.index, args.out, args.storage, args.workers, args.limit, args.protocol, only=only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
