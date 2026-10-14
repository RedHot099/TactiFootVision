"""Download SoccerNet-v3 labels for every game, then frames only for games outside the xG held-out list.

Runs in an environment that has the ``SoccerNet`` package (0.1.62), e.g. a tools venv:

    python -I scripts/detector_data/download_soccernet_v3.py labels <raw_dir>
    python -I scripts/detector_data/download_soccernet_v3.py frames <raw_dir> <plan.csv> [shard] [shards]

``Frames-v3.zip`` is a per-game archive, so held-out games (plan.csv role ``evaluation`` or ``gate``) are never fetched.
The downloader uses its built-in v3 share password; no NDA password is read.
"""

import csv
import sys
import time
import zipfile
from pathlib import Path

HELD_OUT_ROLES = {"evaluation", "gate"}
SPLITS = ("train", "valid", "test")


def held_out_games(plan_csv: Path) -> set[str]:
    with plan_csv.open(newline="") as handle:
        return {
            row["game"]
            for row in csv.DictReader(handle)
            if row["role"] in HELD_OUT_ROLES
        }


def complete(path: Path) -> bool:
    """A zip whose central directory reads (an interrupted download does not); other files: exists."""
    if not path.is_file():
        return False
    if path.suffix != ".zip":
        return True
    try:
        with zipfile.ZipFile(path):
            return True
    except zipfile.BadZipFile:
        return False


def fetch(downloader, game: str, file: str, split: str, attempts: int = 8) -> None:
    """``downloadGame`` with retries: the OwnCloud server refuses connections under load."""
    path = Path(downloader.LocalDirectory) / game / file
    for attempt in range(attempts):
        if path.is_file() and not complete(path):
            path.unlink()
        try:
            downloader.downloadGame(game, files=[file], spl=split, verbose=False)
            if complete(path):
                return
            raise OSError("incomplete file")
        except OSError as error:
            wait = min(30 * 2**attempt, 600)
            print(f"retry {attempt + 1} in {wait}s: {game} {file}: {error}", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"giving up on {game} {file}")


def main(argv: list[str]) -> None:
    from SoccerNet.Downloader import SoccerNetDownloader
    from SoccerNet.utils import getListGames

    what, raw_dir = argv[0], Path(argv[1])
    downloader = SoccerNetDownloader(LocalDirectory=str(raw_dir))
    held_out = held_out_games(Path(argv[2])) if what == "frames" else set()
    shard, shards = (int(argv[3]), int(argv[4])) if len(argv) > 4 else (0, 1)
    for split in SPLITS:
        for index, game in enumerate(getListGames(split, task="frames")):
            if index % shards != shard:
                continue
            if what == "labels":
                fetch(downloader, game, "Labels-v3.json", split)
            elif game in held_out:
                print(f"skip held-out {game}", flush=True)
            else:
                fetch(downloader, game, "Frames-v3.zip", split)
                print(f"frames {split} {game}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
