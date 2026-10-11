"""Re-download the SoccerNet halves of the secondary population and cut their clips and frames (xg-vendor ticket 12).

Reads the frozen pairing's cut targets (completion record verified), downloads each needed 720p half one at a time,
checks its sha256 against the dataset's raw-asset record, cuts one 8 s clip per target (anchor at 4 s) and the frames
at offsets -400 and 0 ms through the path of the index frames (`xgpoc.vendor.redownload` documents which function
made what), and publishes `frames_index.parquet` plus a completion record. Halves stay on the target partition unless
`--delete-halves` is given. Resumable per half: run the same command again.

    ./run.sh python scripts/vendor_redownload.py --dry-run                        # plan only; no network, no password read
    nohup ./run.sh python scripts/vendor_redownload.py --owner-approved > results/xgv_redownload.log 2>&1 &

The download itself needs `--owner-approved` (the SoccerNet data are NDA-bound and the owner approves the run), the
password in `SOCCERNET_PASSWORD` or `~/.config/soccernet/password` (never printed, logged or recorded), a target on a
mounted, writable filesystem, and at least 5 GB free on /home throughout. `--max-halves N` makes a pilot of N halves.
A refused half (sha256 mismatch) is a recorded result; to retry it, delete its folder under `<out>.partial/cut/`.
Exit codes: 0 published, 3 published with refused halves (listed in refused_halves.json), 2 not finished (resume),
1 error.
"""

import argparse
import fcntl
import logging
import signal
import sys
from pathlib import Path

import pandas as pd
from xgpoc.bundle import code_version
from xgpoc.data.canonical_frames import runtime_identity
from xgpoc.paths import SOCCERNET
from xgpoc.vendor import redownload, soccernet_access, target_guard
from xgpoc.vendor.statsbomb_fetch import file_sha256

ROOT = Path(__file__).resolve().parents[1]   # experiments/claude_fable
STORAGE = Path("/mnt/storage/tactifoot_research")
PAIRING = ROOT / "results/xgv_pairing"
RAW_ASSETS = SOCCERNET / "raw_assets.jsonl"
SOURCE_FILES = (Path(__file__).resolve(), ROOT / "xgpoc/vendor/redownload.py", ROOT / "xgpoc/vendor/soccernet_access.py",
                ROOT / "xgpoc/vendor/target_guard.py", ROOT / "xgpoc/data/frames.py", ROOT / "xgpoc/data/canonical_frames.py",
                ROOT / "xgpoc/data/offset_frames.py")
log = logging.getLogger("xgv.redownload")


def identity_of(pairing: dict, raw_assets: Path) -> dict:
    return {"version": redownload.VERSION,
            "pairing": {"complete_json_sha256": pairing["complete_json_sha256"], "files": pairing["files"]},
            "raw_assets": {"path": str(raw_assets), "sha256": file_sha256(raw_assets)},
            "splits": pairing["attempts"],
            "offsets_ms": list(redownload.OFFSETS_MS), "clip_span_s": redownload.CLIP_SPAN_S,
            "frame_rule": "xgpoc.data.frames.extract_shot_frames at position_ms = target_position_ms (ffmpeg fast seek, JPEG q:v 3, "
                          "native 720p), as scripts/collect_soccernet.py made the index frames",
            "sources": {str(p): file_sha256(p) for p in SOURCE_FILES}}


def load_inputs(pairing_dir: Path, raw_assets: Path) -> tuple[redownload.Plan, dict]:
    targets, halves, pairing = redownload.verify_pairing(pairing_dir)
    attempts = pairing["attempts"]
    if file_sha256(Path(attempts["path"])) != attempts["sha256"]:
        raise ValueError(f"{attempts['path']} changed since the pairing")
    splits = redownload.read_splits(pd.read_parquet(attempts["path"], columns=["game", "split"]))
    plan = redownload.build_plan(targets, halves, redownload.read_assets(raw_assets), splits)
    return plan, identity_of(pairing, raw_assets)


def mount_report(directory: Path) -> dict:
    """Mount point, type and write probe of a directory, as data (the dry run reports problems instead of stopping)."""
    try:
        info = target_guard.mount_info(directory)
        report = {"mount_point": info.target, "fstype": info.fstype, "read_only": "ro" in info.options}
        if info.target == "/":
            report["problem"] = "on the root filesystem: the real run would refuse this target"
        elif directory.is_dir():
            target_guard.write_probe(directory)
            report["write_probe"] = "passed"
        else:
            report["write_probe"] = "directory does not exist yet; created by the real run after the mount check"
        return report
    except target_guard.TargetRefused as exc:
        return {"problem": str(exc)}


def dry_run(plan: redownload.Plan, layout: redownload.Layout, keep: bool) -> None:
    summary = redownload.plan_summary(plan, layout, keep_halves=keep, password=soccernet_access.password_source())
    summary["halves_dir_mount"] = mount_report(layout.halves)
    summary["out_mount"] = mount_report(layout.out.parent)
    summary["space"] = {"halves_dir_free_gib": round(target_guard.free_bytes(layout.halves) / 2**30, 1),
                        "home_free_gib": round(target_guard.free_bytes(Path("/home")) / 2**30, 1),
                        "home_floor_gib": target_guard.HOME_FLOOR_BYTES // 2**30}
    needed = summary["estimated_download_gib_remaining"] if keep else summary["peak_space_needed_gib"]
    summary["space"]["fits"] = bool(needed + 1 <= summary["space"]["halves_dir_free_gib"])
    summary["network"] = "none contacted: no login, no listing, no download"
    print(redownload.json_text(summary))


def preflight(plan: redownload.Plan, layout: redownload.Layout, keep: bool) -> None:
    """Refuse before any download unless the targets are mounted, writable and large enough."""
    for directory in (layout.halves, layout.out.parent):
        target_guard.check_target(directory)
    remaining = sum(j.asset.bytes for j in plan.halves if j.needs_video and not layout.half_file(j).is_file()
                    and not redownload.record_is_intact(layout, j))
    biggest = max((j.asset.bytes for j in plan.halves), default=0)
    target_guard.check_space(layout.halves, (remaining if keep else biggest) + redownload.SPACE_MARGIN_BYTES)


def terminate(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pairing", type=Path, default=PAIRING, help="frozen pairing folder (default %(default)s)")
    parser.add_argument("--raw-assets", type=Path, default=RAW_ASSETS)
    parser.add_argument("--target", type=Path, default=STORAGE / "soccernet_halves", help="where the halves are kept (default %(default)s)")
    parser.add_argument("--out", type=Path, default=STORAGE / "xgv_secondary_v1",
                        help="the published data version: clips, frames, records, frames_index.parquet (default %(default)s)")
    parser.add_argument("--delete-halves", action="store_true", help="delete each half after cutting (default: keep them)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit; no network, no password read")
    parser.add_argument("--owner-approved", action="store_true", help="required for the download: the owner has approved it")
    parser.add_argument("--max-halves", type=int, default=None, help="process at most this many halves (pilot)")
    parser.add_argument("--attempts", type=int, default=3, help="download attempts per half")
    parser.add_argument("--log", type=Path, default=None, help="also write the log to this file")
    args = parser.parse_args(argv)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log))
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True, format="%(asctime)s %(levelname)s %(message)s")

    plan, identity = load_inputs(args.pairing, args.raw_assets)
    layout = redownload.Layout(args.out, args.target)
    keep = not args.delete_halves
    if (args.out / "complete.json").exists():
        record = redownload.verify_complete(args.out, identity)
        print(f"Verified {args.out}; nothing to do\n{redownload.json_text(record['summary'])}")
        return 3 if record["summary"]["halves_refused"] else 0
    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} exists and holds no completion record; refusing to publish over it. Remove it or use another --out")
    if args.dry_run:
        dry_run(plan, layout, keep)
        return 0
    if not args.owner_approved:
        sys.exit("The download needs --owner-approved (and the owner's go-ahead). Use --dry-run to see the plan.")

    preflight(plan, layout, keep)
    try:
        secret = soccernet_access.load_password()
    except soccernet_access.PasswordError as exc:
        sys.exit(str(exc))
    for handler in handlers:
        handler.addFilter(soccernet_access.ScrubFilter(secret))
    signal.signal(signal.SIGTERM, terminate)
    layout.partial.mkdir(parents=True, exist_ok=True)
    with (layout.partial / ".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            sys.exit(f"Another run holds {layout.partial / '.lock'}")
        try:
            downloader = soccernet_access.SoccerNetHalfDownloader(secret, layout.halves / ".staging")
            log.info("start: %d halves, %d targets, keep_halves=%s, halves in %s, out %s", len(plan.halves), plan.targets, keep,
                     layout.halves, layout.out)
            result = redownload.run(plan, layout, downloader, keep_halves=keep, max_halves=args.max_halves, attempts=args.attempts)
        except (target_guard.LowSpace, target_guard.TargetRefused) as exc:
            log.error("stopped: %s (resume with the same command)", exc)
            return 2
        except KeyboardInterrupt:
            log.error("interrupted; resume with the same command")
            return 2
        except Exception:
            log.exception("failed")
            return 1
        log.info("run finished: %d cut, %d refused, %d already done, %d failed downloads", result.cut, result.refused,
                 result.skipped, len(result.failed))
        if result.stopped_early or result.failed:
            log.error("not finished (failed: %s); resume with the same command", list(result.failed))
            return 2
        try:
            summary = redownload.publish(layout, plan, identity, code_version(ROOT), runtime_identity(),
                                         {str(p): p for p in SOURCE_FILES})
        except redownload.Incomplete as exc:
            log.error("%s", exc)
            return 2
    log.info("published %s\n%s", args.out, redownload.json_text(summary))
    return 3 if summary["halves_refused"] else 0


if __name__ == "__main__":
    sys.exit(main())
