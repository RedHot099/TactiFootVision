"""Record source bytes before execution and reject mid-run source changes."""

from pathlib import Path

from xgpoc.v13.protocol import digest


def snapshot(paths):
    return {str(Path(p).resolve()): digest(Path(p)) for p in paths}


def verify_sources(expected):
    for filename, checksum in expected.items():
        path = Path(filename)
        if not path.is_file() or digest(path) != checksum:
            raise ValueError(f"Source changed during execution: {filename}; restart with stable code")
