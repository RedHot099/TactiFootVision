"""Code identity and resource use of the xg-temporal runs: the files a completion record pins."""

import resource
import sys
from pathlib import Path

import numpy as np

from xgpoc.bundle import canonical_digest, sha256_file

EXPERIMENT = Path(__file__).resolve().parents[2]          # experiments/claude_fable


def loaded_files() -> list[Path]:
    """The source file of every module of the experiment's `xgpoc` package imported so far."""
    package = EXPERIMENT / "xgpoc"
    files = {Path(f).resolve() for m in list(sys.modules.values()) if (f := getattr(m, "__file__", None))}
    return sorted(p for p in files if p.is_relative_to(package) and p.suffix == ".py")


def code_hashes(script: Path) -> dict:
    """sha256 of every file of `xgpoc/temporal`, of every other `xgpoc` module the run imported (the closure of what
    ran, read from `sys.modules` when the record is written) and of the run script, and their digest."""
    files = {*(EXPERIMENT / "xgpoc/temporal").glob("*.py"), *loaded_files(), Path(script).resolve()}
    hashes = {str(p.relative_to(EXPERIMENT)): sha256_file(p) for p in sorted(files)}
    return {"digest": canonical_digest(hashes), "files": hashes}


def peak_rss_gib() -> float:
    """The peak resident memory of this process so far (Linux reports ru_maxrss in KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def json_default(value):
    """JSON for numpy scalars and arrays, keeping ints and bools as such."""
    if isinstance(value, np.generic | np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")
