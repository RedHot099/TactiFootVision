"""Re-download and cut of the SoccerNet halves that hold the secondary population (xg-vendor ticket 12).

Input: the frozen pairing's cut targets (`results/xgv_pairing/cut_targets.parquet`: recovered goals and StatsBomb-only
shots, each with `target_position_ms` = StatsBomb time + half offset in its SoccerNet half). Per half: download the 720p
mkv, check its sha256 against the dataset's raw-asset record (a mismatch refuses and lists the half), cut one 8 s clip
per target and extract the frames at -400 and 0 ms, then keep or delete the half. One half at a time, resumable.

Which function made which file (the retrained G-combo was trained and scored on the index frames, so the frames here
come through the very same path):
 * Frames. `xgpoc.data.frames.extract_shot_frames`, the function `scripts/collect_soccernet.py` called for the index:
   ffmpeg fast seek (`-ss <ms/1000:.3f> -i <half>`), one native 1280x720 JPEG (`-q:v 3`) per offset, called with
   `position_ms = target_position_ms`. File layout `<frames>/<shot_id>/<+0|-400>.jpg`, as the index's `frame_path`
   and `frame_path_-400` columns. Ticket 13 scores these files through `g_bundle.g_dataset`, as ticket 11 scored the
   index frames. The archived offset-frame PNGs decode differently (ticket 11 measured up to 1.3 logits), so they are
   not the reference here.
 * Clips. The 8 s clip around the target (anchor at 4000 ms) is cut exactly as the index clips were
   (`scripts/collect_soccernet.py:cut_clip`, libx264 veryfast crf 26, no audio; a test pins the command) and then
   checked with the canonical timeline code of `xgpoc.data.canonical_frames.probe_video` and
   `xgpoc.data.offset_frames` (strictly increasing PTS, 25 fps, anchor frame within the tolerance of 4000 ms). The
   clips keep the secondary population on the same footing as the primary one for any later decode-path check; the
   scoring reads the JPEG frames.

Every target gets a row with frame status per offset, or a reason: `before_half_start`, `beyond_half_end`,
`extraction_failed`, `unreadable`, `half_refused_sha256`, `target_unusable`. Nothing is substituted.

Resume and publish. A half is done when `cut/<half_id>/record.json` exists and every file it lists matches its recorded
sha256. A half's outputs are built in a scratch folder and renamed into `cut/` in one step. The whole version is built in
`<out>.partial` and renamed to `<out>` only after the completion record (source hashes, a hash of every output) is
written.
"""

import json
import logging
import os
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from xgpoc.data.canonical_frames import probe_video
from xgpoc.data.frames import extract_shot_frames, grass_ratio
from xgpoc.data.offset_frames import (
    ANCHOR_MS,
    anchor_selection,
    check_frame_rate,
)
from xgpoc.vendor import target_guard
from xgpoc.vendor.soccernet_access import DownloadError, HalfDownloader, half_file_name
from xgpoc.vendor.statsbomb_fetch import file_sha256

log = logging.getLogger("xgv.redownload")

VERSION = "xgv_secondary_frames_v1"
PAIRING_VERSION = "xgv_pairing_v1"
OFFSETS_MS = (-400, 0)          # the offsets ticket 13 scores; the index columns frame_path_-400 and frame_path
CLIP_SPAN_S = 4.0               # 8 s clip, anchor at 4000 ms
CLIP_BYTES_ESTIMATE = 3 * 2**20   # measured 1.75 MB per index clip; room for 8 s at crf 26 of busier play
SPACE_MARGIN_BYTES = 256 * 2**20
PRIOR_SECONDS_PER_HALF = 93.0   # measured by the first collection
PRIOR_BYTES_PER_SECOND = 11.7 * 2**20
RETRY_PAUSE_S = 30
MAX_CONSECUTIVE_FAILURES = 3

OK = "ok"
BEFORE_START, BEYOND_END, EXTRACT_FAILED, UNREADABLE = "before_half_start", "beyond_half_end", "extraction_failed", "unreadable"
HALF_REFUSED, TARGET_UNUSABLE = "half_refused_sha256", "target_unusable"
CLIP_NOT_AT_ANCHOR = "anchor_not_at_4000"
# outcome recorded for a half: cut, refused by the sha256 check, or not downloaded because no target lies inside its video
CUT, REFUSED, NO_VIDEO = "cut", "refused_sha256", "targets_beyond_video_end"

TARGET_COLUMNS = ["kind", "game", "half", "sb_match_id", "sb_event_id", "sb_timestamp_ms", "offset_ms",
                  "target_position_ms", "anchor_convention", "usable", "reason_unusable"]
PAIRING_FILES = ("cut_targets.parquet", "halves_to_cut.csv")


class Incomplete(RuntimeError):
    """The run has halves without a record; resume it."""


class HalfFailed(RuntimeError):
    """A half could not be downloaded after all attempts."""


def offset_name(offset_ms: int) -> str:
    return "0" if offset_ms == 0 else str(offset_ms)


def json_text(value) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False, default=_plain) + "\n"


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if value is pd.NA:
        return None
    raise TypeError(f"Cannot serialise {type(value)}")


# ------------------------------------------------------------------ inputs and plan
@dataclass(frozen=True)
class Asset:
    bytes: int
    sha256: str
    duration_s: float


@dataclass(frozen=True)
class HalfJob:
    game: str
    split: str
    half: int
    asset: Asset
    targets: tuple[dict, ...]

    @property
    def half_id(self) -> str:
        return f"{self.game.replace('/', '__').replace(' ', '-')}__h{self.half}"

    @property
    def file_name(self) -> str:
        return half_file_name(self.half)

    def in_video(self, target: dict) -> bool:
        """Some requested frame of the target lies inside the recorded video (the half ends before stoppage-time shots)."""
        return target["target_position_ms"] + min(OFFSETS_MS) < self.asset.duration_s * 1000

    @property
    def needs_video(self) -> bool:
        return any(self.in_video(t) for t in self.targets)


@dataclass(frozen=True)
class Plan:
    halves: tuple[HalfJob, ...]
    unusable: tuple[dict, ...]    # targets that are not cut, listed in the index with their reason

    @property
    def targets(self) -> int:
        return sum(len(h.targets) for h in self.halves) + len(self.unusable)

    @property
    def total_bytes(self) -> int:
        """Bytes of the halves that must be downloaded."""
        return sum(h.asset.bytes for h in self.halves if h.needs_video)


def shot_id(target: dict) -> str:
    return f"xgv__{target['kind']}__{target['sb_event_id']}"


def verify_pairing(directory: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """The cut targets and halves list of the frozen pairing, after its completion record has been checked.

    Every file in the record must match its sha256 and the version must be the frozen one. Returns the targets, the
    halves list and `{"complete_json_sha256", "files": {name: sha256}, "attempts": {...}}` for the identity."""
    record_file = directory / "complete.json"
    if not record_file.is_file():
        raise ValueError(f"{directory} has no completion record; run scripts/vendor_pair.py")
    record = json.loads(record_file.read_text())
    if record["identity"]["version"] != PAIRING_VERSION:
        raise ValueError(f"Pairing version {record['identity']['version']} is not {PAIRING_VERSION}")
    for name, expected in record["files"].items():
        if file_sha256(directory / name) != expected:
            raise ValueError(f"Pairing output changed: {name}")
    if missing := [n for n in PAIRING_FILES if n not in record["files"]]:
        raise ValueError(f"Pairing record lacks {missing}")
    targets = pd.read_parquet(directory / "cut_targets.parquet")
    halves = pd.read_csv(directory / "halves_to_cut.csv")
    if absent := [c for c in TARGET_COLUMNS if c not in targets.columns]:
        raise ValueError(f"cut_targets lacks {absent}")
    if targets.sb_event_id.duplicated().any():
        raise ValueError("cut_targets has duplicate sb_event_id values")
    return targets, halves, {"complete_json_sha256": file_sha256(record_file),
                             "files": {n: record["files"][n] for n in PAIRING_FILES},
                             "attempts": record["identity"]["inputs"]["attempts"]}


def read_assets(path: Path) -> dict[tuple[str, int], Asset]:
    """720p raw-asset records by (game, half). Records with an error are ignored; conflicting hashes are an error."""
    found: dict[tuple[str, int], dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("component") != "720p" or record.get("error") or not record.get("sha256"):
            continue
        key = (record["game"], int(record["half"]))
        if key in found and found[key]["sha256"] != record["sha256"]:
            raise ValueError(f"Conflicting raw-asset hashes for {key}")
        found[key] = record
    return {k: Asset(int(r["bytes"]), r["sha256"], float(r["duration_s"])) for k, r in found.items()}


def read_splits(attempts: pd.DataFrame) -> dict[str, str]:
    """Official split (train, valid, test) of each game; the downloader needs it to build the request."""
    pairs = attempts[["game", "split"]].drop_duplicates()
    if pairs.game.duplicated().any():
        raise ValueError("A game appears in more than one split")
    return dict(zip(pairs.game, pairs.split, strict=True))


def build_plan(targets: pd.DataFrame, halves_to_cut: pd.DataFrame, assets: dict[tuple[str, int], Asset],
               splits: dict[str, str]) -> Plan:
    """One job per (game, half) with a usable target, in the order of `halves_to_cut`."""
    usable = targets[targets.usable.astype(bool)]
    listed = [(g, int(h)) for g, h in zip(halves_to_cut.game, halves_to_cut.half, strict=True)]
    if sorted(listed) != sorted(usable.groupby(["game", "half"]).size().index.tolist()):
        raise ValueError("halves_to_cut.csv does not match the halves of the usable cut targets")
    jobs = []
    for key in listed:
        if key not in assets:
            raise ValueError(f"No raw-asset record for {key}")
        if key[0] not in splits:
            raise ValueError(f"No split for {key[0]}")
        rows = usable[(usable.game == key[0]) & (usable.half == key[1])].sort_values("target_position_ms")
        jobs.append(HalfJob(key[0], splits[key[0]], key[1], assets[key], tuple(rows.to_dict("records"))))
    unusable = tuple(targets[~targets.usable.astype(bool)].to_dict("records"))
    return Plan(tuple(jobs), unusable)


# ------------------------------------------------------------------ layout
@dataclass(frozen=True)
class Layout:
    out: Path       # published folder: clips, frames, per-half records, frames index, completion record
    halves: Path    # where the downloaded halves are kept

    @property
    def partial(self) -> Path:
        return self.out.with_name(self.out.name + ".partial")

    @property
    def cut(self) -> Path:
        return self.partial / "cut"

    @property
    def scratch(self) -> Path:
        return self.partial / ".work"

    @property
    def download_dir(self) -> Path:
        return self.halves / ".partial"

    def half_file(self, job: HalfJob) -> Path:
        return self.halves / job.game / job.file_name

    def record_file(self, job: HalfJob) -> Path:
        return self.cut / job.half_id / "record.json"


# ------------------------------------------------------------------ cutting
def cut_clip(video: Path, center_ms: int, out: Path, span_s: float = CLIP_SPAN_S) -> bool:
    """The 8 s clip around `center_ms`, cut as `scripts/collect_soccernet.py:cut_clip` cut the index clips."""
    out.parent.mkdir(parents=True, exist_ok=True)
    start = max(center_ms / 1000 - span_s, 0)
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{start:.3f}", "-i", str(video), "-t", f"{2 * span_s:.3f}",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-an", str(out)]
    return subprocess.run(cmd, capture_output=True).returncode == 0 and out.exists()


def check_clip(clip: Path, position_ms: int) -> dict:
    """Canonical timeline checks of a cut clip: status, decoded time of the anchor frame, frame count."""
    try:
        timeline = probe_video(clip)
        relative = np.array([timeline.relative_ms(i, ANCHOR_MS) for i in range(len(timeline.pts))])
        if relative[-1] < 0:
            raise ValueError("Clip ends before the annotation anchor")
        check_frame_rate(relative)
        selection = anchor_selection(relative)
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as exc:
        return {"clip_status": f"clip_failed: {type(exc).__name__}: {exc}"[:200], "clip_anchor_error_ms": None, "clip_frames": None}
    if position_ms < ANCHOR_MS:   # the clip starts at the half's beginning, so its anchor is not at 4000 ms
        return {"clip_status": CLIP_NOT_AT_ANCHOR, "clip_anchor_error_ms": None, "clip_frames": len(relative)}
    return {"clip_status": selection.status, "clip_anchor_error_ms": selection.decoded_ms, "clip_frames": len(relative)}


def cut_target(video: Path, target: dict, duration_ms: float, work: Path, relative: str) -> tuple[dict, dict[str, str]]:
    """Clip and frames of one target into `work`; returns its index row (paths relative to the version folder) and
    the sha256 of every file written. `relative` is where `work` will sit in the published tree."""
    sid, position = shot_id(target), int(target["target_position_ms"])
    row = {"shot_id": sid, "kind": target["kind"], "sb_event_id": target["sb_event_id"], "sb_match_id": target["sb_match_id"],
           "game": target["game"], "half": int(target["half"]), "target_position_ms": position,
           "half_offset_ms": target["offset_ms"], "anchor_convention": target["anchor_convention"], "half_outcome": CUT}
    files: dict[str, str] = {}
    status, allowed = {}, []
    for offset in OFFSETS_MS:
        if position + offset < 0:
            status[offset] = BEFORE_START
        elif position + offset >= duration_ms:
            status[offset] = BEYOND_END
        else:
            allowed.append(offset)
    got = extract_shot_frames(video, position, sid, work / "frames", offsets_ms=tuple(allowed))
    for offset in allowed:
        path = got.get(offset)
        image = cv2.imread(str(path)) if path else None
        if path is None:
            status[offset] = EXTRACT_FAILED
        elif image is None:
            status[offset] = UNREADABLE
            path.unlink()
        else:
            status[offset] = OK
            files[f"{relative}/frames/{sid}/{path.name}"] = file_sha256(path)
            row[f"frame_path{'' if offset == 0 else f'_{offset}'}"] = f"{relative}/frames/{sid}/{path.name}"
            row[f"frame_sha256_{offset_name(offset)}"] = files[f"{relative}/frames/{sid}/{path.name}"]
            if offset == 0:
                row["grass_ratio"] = grass_ratio(image)
    for offset in OFFSETS_MS:
        row[f"frame_status_{offset_name(offset)}"] = status[offset]
    clip = work / "clips" / f"{sid}.mp4"
    if cut_clip(video, position, clip):
        row |= check_clip(clip, position)
        row["clip_path"] = f"{relative}/clips/{sid}.mp4"
        row["clip_sha256"] = files[row["clip_path"]] = file_sha256(clip)
    else:
        row |= {"clip_status": "clip_failed: ffmpeg", "clip_anchor_error_ms": None, "clip_frames": None}
    row["reason"] = "; ".join(f"frame {offset_name(o)} ms: {status[o]}" for o in OFFSETS_MS if status[o] != OK)
    return row, files


def skipped_row(target: dict, outcome: str, status: str, reason: str) -> dict:
    """Index row of a target that is not cut: the status of both offsets and the listed reason."""
    position = target["target_position_ms"]
    return {"shot_id": shot_id(target), "kind": target["kind"], "sb_event_id": target["sb_event_id"],
            "sb_match_id": target["sb_match_id"], "game": target["game"], "half": int(target["half"]),
            "target_position_ms": None if pd.isna(position) else int(position),
            "half_offset_ms": target["offset_ms"], "anchor_convention": target["anchor_convention"],
            "half_outcome": outcome, **{f"frame_status_{offset_name(o)}": status for o in OFFSETS_MS},
            "clip_status": "none", "reason": reason}


def unusable_row(target: dict) -> dict:
    return skipped_row(target, TARGET_UNUSABLE, TARGET_UNUSABLE, f"{TARGET_UNUSABLE}: {target.get('reason_unusable', '')}")


def beyond_end_row(target: dict, job: HalfJob) -> dict:
    return skipped_row(target, NO_VIDEO, BEYOND_END, f"{BEYOND_END}: target at {target['target_position_ms'] / 1000:.1f} s, "
                                                       f"the half video lasts {job.asset.duration_s:.1f} s")


# ------------------------------------------------------------------ one half
def record_is_intact(layout: Layout, job: HalfJob) -> bool:
    """The half has a record for this asset and every file the record lists is present and unchanged."""
    try:
        record = json.loads(layout.record_file(job).read_text())
        half = record["half"]
        wanted = {t["sb_event_id"] for t in job.targets}
        if (record["version"], half["game"], half["half"], half["expected_sha256"]) != (VERSION, job.game, job.half, job.asset.sha256):
            return False
        if {r["sb_event_id"] for r in record["rows"]} != wanted or half["outcome"] not in (CUT, REFUSED, NO_VIDEO):
            return False
        return all((layout.partial / name).is_file() and file_sha256(layout.partial / name) == checksum
                   for name, checksum in record["files"].items())
    except (OSError, ValueError, KeyError, TypeError):
        return False


def fetch_half(job: HalfJob, layout: Layout, downloader: HalfDownloader, attempts: int, pause_s: float,
               space: Callable[[Path, int], None]) -> tuple[Path | None, str]:
    """The verified half file (None when refused) and its actual sha256.

    A file already in the keep folder is verified first; a mismatch there is deleted and downloaded again, once. A
    download that does not match the asset record is deleted and refused (the half is listed, not cut)."""
    final = layout.half_file(job)
    if final.is_file():
        if file_sha256(final) == job.asset.sha256:
            log.info("%s: kept half found and verified", job.half_id)
            return final, job.asset.sha256
        log.warning("%s: the kept half does not match the recorded sha256; downloading it again", job.half_id)
        final.unlink()
    layout.download_dir.mkdir(parents=True, exist_ok=True)
    part = layout.download_dir / f"{job.half_id}.mkv"
    for attempt in range(1, attempts + 1):
        part.unlink(missing_ok=True)
        space(layout.halves, job.asset.bytes + SPACE_MARGIN_BYTES)
        space(layout.out.parent, len(job.targets) * CLIP_BYTES_ESTIMATE + SPACE_MARGIN_BYTES)
        try:
            downloader.download(job.game, job.split, job.half, part)
            break
        except DownloadError as exc:
            log.warning("%s: download attempt %d of %d failed: %s", job.half_id, attempt, attempts, exc)
            if attempt == attempts:
                raise HalfFailed(f"{job.half_id}: {exc}") from None
            time.sleep(pause_s * attempt)
    actual = file_sha256(part)
    if actual != job.asset.sha256:
        log.error("%s: REFUSED, sha256 %s differs from the recorded %s (%d bytes, recorded %d)",
                  job.half_id, actual, job.asset.sha256, part.stat().st_size, job.asset.bytes)
        part.unlink()
        return None, actual
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(part, final)
    return final, actual


def process_half(job: HalfJob, layout: Layout, downloader: HalfDownloader, *, keep_halves: bool, attempts: int = 3,
                 pause_s: float = RETRY_PAUSE_S, space: Callable[[Path, int], None] | None = None) -> str:
    """Bring one half to a record: 'skipped' (already done), 'cut', 'refused' or 'no_video'. `space(path, need_bytes)`
    defaults to `target_guard.check_space`."""
    if record_is_intact(layout, job):
        return "skipped"
    work = layout.scratch / job.half_id
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    relative = f"cut/{job.half_id}"
    video, actual, files = None, None, {}
    if not job.needs_video:    # every target is in stoppage time after the end of the video: nothing to download
        outcome, rows = NO_VIDEO, [beyond_end_row(t, job) for t in job.targets]
    else:
        video, actual = fetch_half(job, layout, downloader, attempts, pause_s, space or target_guard.check_space)
        if video is None:
            outcome, rows = REFUSED, [skipped_row(t, REFUSED, HALF_REFUSED, HALF_REFUSED) for t in job.targets]
        else:
            outcome, rows = CUT, []
            for target in job.targets:
                row, written = cut_target(video, target, job.asset.duration_s * 1000, work, relative)
                rows.append(row)
                files |= written
    half = {"game": job.game, "split": job.split, "half": job.half, "half_id": job.half_id, "outcome": outcome,
            "expected_sha256": job.asset.sha256, "actual_sha256": actual, "expected_bytes": job.asset.bytes,
            "kept": bool(video is not None and keep_halves)}
    (work / "record.json").write_text(json_text({"version": VERSION, "half": half, "rows": rows, "files": files}))
    target = layout.cut / job.half_id
    shutil.rmtree(target, ignore_errors=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(work, target)
    if video is not None and not keep_halves:
        video.unlink()
    return {CUT: "cut", REFUSED: "refused", NO_VIDEO: "no_video"}[outcome]


# ------------------------------------------------------------------ the run
def fmt_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    return f"{hours} h {rest // 60:02d} min" if hours else f"{rest // 60} min {rest % 60:02d} s"


@dataclass
class RunResult:
    cut: int = 0
    refused: int = 0
    no_video: int = 0
    skipped: int = 0
    failed: tuple[str, ...] = ()
    stopped_early: bool = False


def run(plan: Plan, layout: Layout, downloader: HalfDownloader, *, keep_halves: bool, max_halves: int | None = None,
        attempts: int = 3, pause_s: float = RETRY_PAUSE_S, max_consecutive_failures: int = MAX_CONSECUTIVE_FAILURES,
        space: Callable[[Path, int], None] | None = None) -> RunResult:
    """Process the halves without an intact record, in plan order. A half that cannot be downloaded is logged and left
    for a resume; three in a row stop the run (a wrong password or a dead server would otherwise fail all 228 alike)."""
    space = space or target_guard.check_space
    pending = [job for job in plan.halves if not record_is_intact(layout, job)]
    result = RunResult(skipped=len(plan.halves) - len(pending))
    log.info("%d of %d halves done already; %d to do", result.skipped, len(plan.halves), len(pending))
    failed, streak, worked, spent, started = [], 0, 0, 0.0, time.monotonic()
    for number, job in enumerate(pending, 1):
        if max_halves is not None and worked >= max_halves:
            result.stopped_early = True
            break
        begin = time.monotonic()
        try:
            outcome = process_half(job, layout, downloader, keep_halves=keep_halves, attempts=attempts, pause_s=pause_s, space=space)
        except HalfFailed as exc:
            failed.append(job.half_id)
            streak += 1
            log.error("[%d/%d] %s", number, len(pending), exc)
            if streak >= max_consecutive_failures:
                log.error("%d halves in a row failed to download; stopping (resume with the same command)", streak)
                result.stopped_early = True
                break
            continue
        streak, worked = 0, worked + 1
        spent += time.monotonic() - begin
        setattr(result, outcome, getattr(result, outcome) + 1)
        per_half = spent / worked
        log.info("[%d/%d] %s: %s, %d targets, %.0f s; elapsed %s, ETA %s at %.0f s per half", number, len(pending), job.half_id,
                 outcome, len(job.targets), time.monotonic() - begin, fmt_duration(time.monotonic() - started),
                 fmt_duration((len(pending) - number) * per_half), per_half)
    result.failed = tuple(failed)
    return result


# ------------------------------------------------------------------ index, completion record
INDEX_COLUMNS = ["shot_id", "kind", "sb_event_id", "sb_match_id", "game", "half", "target_position_ms", "half_offset_ms",
                 "anchor_convention", "half_outcome", "frame_path", "frame_path_-400", "frame_status_0", "frame_status_-400",
                 "frame_sha256_0", "frame_sha256_-400", "grass_ratio", "clip_path", "clip_sha256", "clip_status",
                 "clip_anchor_error_ms", "clip_frames", "reason"]
PATH_COLUMNS = ("frame_path", "frame_path_-400", "clip_path")


def read_records(layout: Layout, plan: Plan) -> list[dict]:
    """Records of all halves; raises `Incomplete` listing the halves without an intact record."""
    missing = [j.half_id for j in plan.halves if not record_is_intact(layout, j)]
    if missing:
        raise Incomplete(f"{len(missing)} halves have no intact record, e.g. {missing[:3]}; resume the run")
    return [json.loads(layout.record_file(j).read_text()) for j in plan.halves]


def assemble_index(records: list[dict], plan: Plan, out: Path) -> pd.DataFrame:
    """One row per target (usable or not); path columns become absolute paths inside the published folder."""
    rows = [r for record in records for r in record["rows"]] + [unusable_row(t) for t in plan.unusable]
    frame = pd.DataFrame(rows).reindex(columns=INDEX_COLUMNS)
    if len(frame) != plan.targets or frame.sb_event_id.duplicated().any():
        raise ValueError("The index does not hold exactly one row per cut target")
    for column in PATH_COLUMNS:
        frame[column] = frame[column].map(lambda p: str(out.resolve() / p) if isinstance(p, str) and p else "")
    ok = frame.frame_status_0.eq(OK)
    if (frame.frame_path.ne("") != ok).any() or (frame["frame_path_-400"].ne("") != frame["frame_status_-400"].eq(OK)).any():
        raise ValueError("Frame path and frame status disagree")
    both = ok & frame["frame_status_-400"].eq(OK)
    if (frame.reason.fillna("").eq("") & ~both).any():
        raise ValueError("A shot without both frames has no reason")
    frame["sb_match_id"] = frame.sb_match_id.astype("Int64")
    return frame.sort_values(["kind", "sb_event_id"]).reset_index(drop=True)


def summarize(index: pd.DataFrame, records: list[dict], plan: Plan) -> dict:
    halves = [r["half"] for r in records]
    both = index.frame_status_0.eq(OK) & index["frame_status_-400"].eq(OK)
    return {"targets": len(index), "by_kind": index.kind.value_counts().to_dict(), "halves": len(plan.halves),
            "halves_cut": sum(h["outcome"] == CUT for h in halves), "halves_refused": sum(h["outcome"] == REFUSED for h in halves),
            "halves_not_downloaded_beyond_video_end": sum(h["outcome"] == NO_VIDEO for h in halves),
            "halves_kept_on_disk": sum(bool(h["kept"]) for h in halves),
            "frames_ok_0": int(index.frame_status_0.eq(OK).sum()), "frames_ok_m400": int(index["frame_status_-400"].eq(OK).sum()),
            "targets_with_both_frames": int(both.sum()), "targets_without_both_frames": int((~both).sum()),
            "frame_status_0": index.frame_status_0.value_counts().to_dict(),
            "frame_status_m400": index["frame_status_-400"].value_counts().to_dict(),
            "clip_status": index.clip_status.value_counts().to_dict(),
            "refused_halves": [{"game": h["game"], "half": h["half"], "expected_sha256": h["expected_sha256"],
                                "actual_sha256": h["actual_sha256"]} for h in halves if h["outcome"] == REFUSED]}


def output_hashes(layout: Layout, records: list[dict]) -> dict[str, str]:
    """sha256 of every file of the version: the clips and frames as recorded (and re-read on verification), the
    per-half records and the index."""
    files: dict[str, str] = {}
    for record in records:
        files |= record["files"]
        name = f"cut/{record['half']['half_id']}/record.json"
        files[name] = file_sha256(layout.partial / name)
    return files


def publish(layout: Layout, plan: Plan, identity: dict, code: dict, runtime: dict,
            source_files: dict[str, Path]) -> dict:
    """Write the index and the completion record in `<out>.partial`, then rename it to `<out>`."""
    records = read_records(layout, plan)
    index = assemble_index(records, plan, layout.out)
    index.to_parquet(layout.partial / "frames_index.parquet", index=False, compression="zstd")
    summary = summarize(index, records, plan)
    (layout.partial / "refused_halves.json").write_text(json_text(summary["refused_halves"]))
    files = output_hashes(layout, records)
    for name in ("frames_index.parquet", "refused_halves.json"):
        files[name] = file_sha256(layout.partial / name)
    if {name: file_sha256(path) for name, path in source_files.items()} != identity["sources"]:
        raise ValueError("A source changed during the run")
    (layout.partial / "complete.json").write_text(json_text(
        {"identity": identity, "code": code, "runtime": runtime, "summary": summary, "files": dict(sorted(files.items()))}))
    shutil.rmtree(layout.scratch, ignore_errors=True)
    (layout.partial / ".lock").unlink(missing_ok=True)
    os.replace(layout.partial, layout.out)
    return summary


def verify_complete(out: Path, identity: dict) -> dict:
    """The completion record of a published version, after every listed file has been re-hashed."""
    record = json.loads((out / "complete.json").read_text())
    if record["identity"] != identity:
        raise ValueError(f"Inputs or sources changed since {out / 'complete.json'}; use a new --out")
    present = {str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()} - {"complete.json"}
    if unlisted := sorted(present - set(record["files"])):
        raise ValueError(f"{len(unlisted)} files are not in the completion record, e.g. {unlisted[0]}")
    for name, expected in record["files"].items():
        if not (out / name).is_file() or file_sha256(out / name) != expected:
            raise ValueError(f"Output changed: {name}")
    return record


# ------------------------------------------------------------------ dry run
def plan_summary(plan: Plan, layout: Layout, *, keep_halves: bool, password: str | None) -> dict:
    """What a run would do, from local files only: halves, targets, bytes, time, space, resume state."""
    needed = [h for h in plan.halves if h.needs_video]
    done = {h.half_id for h in plan.halves if record_is_intact(layout, h)}
    todo = [h for h in needed if h.half_id not in done]
    fetch = [h for h in todo if not layout.half_file(h).is_file()]
    fetch_bytes = sum(h.asset.bytes for h in fetch)
    sizes = [h.asset.bytes for h in needed]
    in_video = [t for h in plan.halves for t in h.targets if h.in_video(t)]
    return {
        "halves_in_pairing": len(plan.halves), "halves_to_download": len(needed), "matches": len({h.game for h in needed}),
        "targets": plan.targets, "targets_unusable": len(plan.unusable),
        "targets_inside_the_half_video": len(in_video),
        "targets_beyond_video_end": sum(len(h.targets) for h in plan.halves) - len(in_video),
        "halves_with_only_targets_beyond_video_end": len(plan.halves) - len(needed),
        "by_kind_inside_video": pd.Series([t["kind"] for t in in_video]).value_counts().to_dict(),
        "by_split": pd.Series([h.split for h in needed]).value_counts().to_dict(),
        "halves_per_target_count": pd.Series([len(h.targets) for h in needed]).value_counts().sort_index().to_dict(),
        "estimated_download_gib_all": round(plan.total_bytes / 2**30, 1),
        "estimated_download_gib_remaining": round(fetch_bytes / 2**30, 1),
        "half_size_mib_min_max": [round(min(sizes) / 2**20), round(max(sizes) / 2**20)] if sizes else None,
        "estimated_time_remaining": {
            "at_prior_93_s_per_half": fmt_duration(len(fetch) * PRIOR_SECONDS_PER_HALF),
            "at_11_7_mib_per_s": fmt_duration(fetch_bytes / PRIOR_BYTES_PER_SECOND)},
        "peak_space_needed_gib": round((plan.total_bytes if keep_halves else max(sizes, default=0)) / 2**30, 1),
        "keep_halves": keep_halves, "halves_dir": str(layout.halves), "out": str(layout.out),
        "resume": {"halves_with_intact_record": len(done), "halves_file_already_present": sum(layout.half_file(h).is_file() for h in needed),
                   "to_do": len(todo)},
        "password_source": password,
    }
