"""Download SoccerNet line-annotated frames for the pitch keypoint dataset (pitch-keypoints 01).

    ./run.sh --no-sync python scripts/pk_fetch_soccernet.py gamestate-2024 test
    ./run.sh --no-sync python scripts/pk_fetch_soccernet.py calibration-2023 train valid test

Calibration-2023 uses the `SoccerNet` package downloader with its public-data password (the default argument). The
package downloader fetched GSR test only to 6.26 of 8.85 GB before the server closed the connection
(`ContentTooShortError`) and then deleted the part, so GSR downloads the same URL with the same public share credentials
(`SoccerNet.Downloader`, task gamestate-2024) through `resumable_download`, which resumes with HTTP Range requests.
The personal SoccerNet password file is never read. One split at a time:

1. refuse unless /home keeps >= 15 GB free after the zip and its extraction (sizes below are upper bounds);
2. download `<root>/<task>/<split>.zip`;
3. extract into `<root>/<split>` (the Calibration zips hold a `<split>/` folder, so they go to `<root>`), checking
   that every member path stays inside;
   for GSR only the label files and every `FRAME_STEP`-th frame (frames 1, 6, 11, ...) are kept;
4. delete the GSR zip (Calibration zips are small and kept);
5. append a JSON line with the source, zip size, sha256, kept file counts and timing to `<root>/fetch_log.jsonl`.
"""

import argparse
import base64
import json
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from xgpoc.bundle import sha256_file
from xgpoc.pitchkp.gsr_eval import FRAME_STEP, keeps_frame

DATA = Path("/home/kuba/projects/tactifoot_vision/data")
ROOTS = {"gamestate-2024": DATA / "SoccerNetGS", "calibration-2023": DATA / "SoccerNetCalibration"}
# Upper bounds in GB of one split's zip + kept extraction, for the free-space check.
NEED_GB = {"gamestate-2024": 12.0 + 2.6, "calibration-2023": 3.0 + 3.0}
KEEP_FREE_GB = 15.0
OWNCLOUD = "https://exrcsdrive.kaust.edu.sa/public.php/webdav/"     # SoccerNetDownloader's default server
GSR_SHARE = "iOJmJH6rYnx7mOS"           # the public share SoccerNet.Downloader uses for task gamestate-2024
PUBLIC_PASSWORD = "SoccerNet"           # downloadDataTask's default password for public data
SOURCES = {"calibration-2023": "SoccerNet.Downloader.SoccerNetDownloader.downloadDataTask (public password)",
           "gamestate-2024": f"{OWNCLOUD}<split>.zip, share {GSR_SHARE} of SoccerNet.Downloader, resumable_download"}


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def resumable_download(url: str, user: str, password: str, dest: Path, attempts: int = 30, chunk: int = 1 << 22) -> int:
    """Download `url` (HTTP basic auth) to `dest`, resuming a `.part` file with Range requests; returns the size."""
    part = dest.with_name(dest.name + ".part")
    auth = "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()
    for attempt in range(attempts):
        have = part.stat().st_size if part.exists() else 0
        request = urllib.request.Request(url, headers={"Authorization": auth, "Range": f"bytes={have}-"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if have and response.status != 206:            # the server ignored the range: start over
                    have = 0
                total = have + int(response.headers["Content-Length"])
                with part.open("ab" if have else "wb") as out:
                    for block in iter(lambda: response.read(chunk), b""):
                        out.write(block)
        except urllib.error.HTTPError as error:
            if error.code == 416 and have and zipfile.is_zipfile(part):   # range past the end: the part is whole
                part.rename(dest)
                return dest.stat().st_size
            print(f"attempt {attempt + 1}: {error}", flush=True)
            time.sleep(15)
            continue
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            print(f"attempt {attempt + 1}: {error}", flush=True)
            time.sleep(15)
            continue
        size = part.stat().st_size
        print(f"attempt {attempt + 1}: {size} of {total} bytes", flush=True)
        if size == total:
            part.rename(dest)
            return size
    raise SystemExit(f"{url}: no complete download after {attempts} attempts")


def keep_member(task: str, name: str) -> bool:
    """Whether a zip member is extracted: everything for Calibration; GSR labels, info files and every 5th frame."""
    if task != "gamestate-2024" or not name.endswith(".jpg"):
        return True
    return keeps_frame(int(Path(name).stem))


def safe_extract(archive: Path, dest: Path, task: str) -> tuple[int, int]:
    """Extract the kept members of `archive` into `dest`; refuses member paths that leave `dest`. Returns (kept, skipped)."""
    dest = dest.resolve()
    kept = skipped = 0
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            target = (dest / info.filename).resolve()
            if not target.is_relative_to(dest):
                raise RuntimeError(f"zip member leaves the target folder: {info.filename!r}")
            if info.is_dir():
                continue
            if not keep_member(task, info.filename):
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out, 1 << 22)
            kept += 1
    return kept, skipped


def fetch(task: str, split: str) -> dict:
    root = ROOTS[task]
    root.mkdir(parents=True, exist_ok=True)
    archive = root / task / f"{split}.zip"
    free_before = free_gb(root)
    if not archive.exists() and free_before - NEED_GB[task] < KEEP_FREE_GB:
        raise SystemExit(f"{free_before:.1f} GB free; {task}/{split} needs {NEED_GB[task]} GB + {KEEP_FREE_GB} GB kept free")
    start = time.time()
    if not archive.exists() and task == "gamestate-2024":
        archive.parent.mkdir(parents=True, exist_ok=True)
        resumable_download(f"{OWNCLOUD}{split}.zip", GSR_SHARE, PUBLIC_PASSWORD, archive)
    elif not archive.exists():
        from SoccerNet.Downloader import SoccerNetDownloader

        SoccerNetDownloader(LocalDirectory=str(root)).downloadDataTask(task=task, split=[split])
    if not archive.exists():
        raise SystemExit(f"download produced no {archive}")
    zip_bytes = archive.stat().st_size
    digest = sha256_file(archive)
    downloaded = time.time()
    kept, skipped = safe_extract(archive, root if task == "calibration-2023" else root / split, task)
    if task == "gamestate-2024":
        archive.unlink()
    record = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "task": task,
        "split": split,
        "source": SOURCES[task],
        "zip": str(archive),
        "zip_bytes": zip_bytes,
        "zip_sha256": digest,
        "zip_deleted": task == "gamestate-2024",
        "frame_rule": f"GSR frames with (frame - 1) % {FRAME_STEP} == 0" if task == "gamestate-2024" else "all files",
        "files_kept": kept,
        "files_skipped": skipped,
        "kept_bytes": sum(p.stat().st_size for p in (root / split).rglob("*") if p.is_file()),
        "download_s": round(downloaded - start, 1),
        "extract_s": round(time.time() - downloaded, 1),
        "free_gb_before": round(free_before, 1),
        "free_gb_after": round(free_gb(root), 1),
    }
    with (root / "fetch_log.jsonl").open("a") as log:
        log.write(json.dumps(record) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", choices=sorted(ROOTS))
    parser.add_argument("splits", nargs="+", choices=("train", "valid", "test"))
    args = parser.parse_args()
    for split in args.splits:
        print(json.dumps(fetch(args.task, split)), flush=True)


if __name__ == "__main__":
    main()
