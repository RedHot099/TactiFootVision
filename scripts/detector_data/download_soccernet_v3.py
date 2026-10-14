"""Download SoccerNet-v3 labels, then frames, for every game (held-out xG games included: the pool is tagged, not cut).

Runs in an environment that has the ``SoccerNet`` package (0.1.62), e.g. a tools venv:

    python -I scripts/detector_data/download_soccernet_v3.py labels <raw_dir>
    python -I scripts/detector_data/download_soccernet_v3.py frames <raw_dir> [shard] [shards]

``Frames-v3.zip`` is a per-game archive; complete archives are kept, interrupted ones are fetched again.
The downloader uses its built-in v3 share password; no NDA password is read.
"""

import socket
import sys
import time
import zipfile
from pathlib import Path

SPLITS = ("train", "valid", "test")


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
    import SoccerNet.Downloader as sn_downloader
    from SoccerNet.Downloader import SoccerNetDownloader

    # The downloader reports every file to Google Analytics with no timeout, which can hang a worker for good;
    # skip it, and let a stalled transfer raise (and be retried) instead of blocking.
    sn_downloader.report = lambda *args, **kwargs: None
    socket.setdefaulttimeout(120)
    from SoccerNet.utils import getListGames

    what, raw_dir = argv[0], Path(argv[1])
    downloader = SoccerNetDownloader(LocalDirectory=str(raw_dir))
    shard, shards = (int(argv[2]), int(argv[3])) if len(argv) > 3 else (0, 1)
    for split in SPLITS:
        for index, game in enumerate(getListGames(split, task="frames")):
            if index % shards != shard:
                continue
            if what == "labels":
                fetch(downloader, game, "Labels-v3.json", split)
            else:
                fetch(downloader, game, "Frames-v3.zip", split)
                print(f"frames {split} {game}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
