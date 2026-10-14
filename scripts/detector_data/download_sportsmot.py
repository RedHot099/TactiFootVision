"""Stream SportsMOT ``train.tar`` / ``val.tar`` from Hugging Face and keep only the football clips (stdlib only).

    python -I scripts/detector_data/download_sportsmot.py <raw_dir> [stride] [tar_dir]

With ``tar_dir`` (a resumable ``hf download MCG-NJU/SportsMOT --include dataset/train.tar dataset/val.tar``), the
local ``<tar_dir>/dataset/<split>.tar`` files are read instead of streaming (a stream cannot resume after a drop).

Writes ``<raw_dir>/<split>/<clip>/{gt/gt.txt, seqinfo.ini, img1/<every stride-th frame>.jpg}``. Nothing else from
the archives touches the disk, and output paths are built from validated parts (no member path is trusted).
The test split has no public ground truth and is not fetched.
"""

import re
import sys
import tarfile
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/MCG-NJU/SportsMOT/resolve/main"
CLIP = re.compile(r"^v_[A-Za-z0-9_-]+_c\d+$")
FRAME = re.compile(r"^(\d{6})\.jpg$")


def football_clips() -> set[str]:
    with urllib.request.urlopen(f"{BASE}/splits_txt/football.txt") as response:
        return {
            line.strip()
            for line in response.read().decode().splitlines()
            if CLIP.match(line.strip())
        }


def keep(name: str, clips: set[str], stride: int) -> Path | None:
    """Relative output path for a tar member we keep, else ``None``."""
    parts = Path(name).parts
    if len(parts) < 3 or parts[-3 if parts[-2] in ("gt", "img1") else -2] not in clips:
        return None
    if parts[-2] == "gt" and parts[-1] == "gt.txt":
        return Path(parts[-3], "gt", "gt.txt")
    if parts[-1] == "seqinfo.ini":
        return Path(parts[-2], "seqinfo.ini")
    match = FRAME.match(parts[-1])
    if parts[-2] == "img1" and match and (int(match.group(1)) - 1) % stride == 0:
        return Path(parts[-3], "img1", parts[-1])
    return None


def main(argv: list[str]) -> None:
    raw_dir, stride = Path(argv[0]), int(argv[1]) if len(argv) > 1 else 1
    tar_dir = Path(argv[2]) if len(argv) > 2 else None
    clips = football_clips()
    for split in ("train", "val"):
        if tar_dir is None:
            source = urllib.request.urlopen(f"{BASE}/dataset/{split}.tar")
        else:
            source = (tar_dir / "dataset" / f"{split}.tar").open("rb")
        with source as response:
            with tarfile.open(fileobj=response, mode="r|") as archive:
                for member in archive:
                    if not member.isfile() or Path(member.name).name.startswith("._"):
                        continue
                    relative = keep(member.name, clips, stride)
                    if relative is None:
                        continue
                    target = raw_dir / split / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(archive.extractfile(member).read())
        print(f"done {split}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
