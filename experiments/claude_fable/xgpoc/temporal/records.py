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


USED = ("xgpoc/__init__.py", "xgpoc/bundle.py", "xgpoc/compare/__init__.py", "xgpoc/compare/categories.py",
        "xgpoc/compare/power.py", "xgpoc/compare/protocol.py", "xgpoc/compare/stats.py", "xgpoc/data/__init__.py",
        "xgpoc/data/canonical_frames.py", "xgpoc/data/datasets.py", "xgpoc/data/offset_frames.py",
        "xgpoc/detector/__init__.py", "xgpoc/detector/approach1.py", "xgpoc/detector/conventions.py",
        "xgpoc/detector/extract.py", "xgpoc/detector/extract3.py", "xgpoc/detector/features.py",
        "xgpoc/detector/guard.py", "xgpoc/detector/records.py", "xgpoc/detector/sanitation.py",
        "xgpoc/detector/seal.py", "xgpoc/detector/testrule.py", "xgpoc/eval/__init__.py",
        "xgpoc/eval/calibration.py", "xgpoc/eval/metrics.py", "xgpoc/models/__init__.py", "xgpoc/models/common.py",
        "xgpoc/models/d_foundation.py", "xgpoc/models/g_bundle.py", "xgpoc/paths.py", "xgpoc/probe/__init__.py",
        "xgpoc/probe/readability.py", "xgpoc/probe/table.py", "xgpoc/stack/__init__.py", "xgpoc/stack/analysis.py",
        "xgpoc/stack/controls.py", "xgpoc/stack/fit.py", "xgpoc/stack/inputs.py", "xgpoc/stack/stats.py",
        "xgpoc/vendor/__init__.py", "xgpoc/vendor/draws.py", "xgpoc/vendor/evaltable.py", "xgpoc/vendor/ladder.py",
        "xgpoc/vendor/refit.py", "scripts/xgt_protocol.py")


def package_hashes() -> dict:
    """sha256 of every file of `xgpoc/temporal`, of the modules it uses (`USED`: the closure the run and protocol scripts
    import, a fixed list so the digest does not depend on what a process imported; a test keeps it complete) and of the protocol writer, and their digest: what the protocol pins. The run
    script is not in it (it pins the protocol's SHA-256); a run record adds it (`code_hashes`)."""
    files = [*sorted((EXPERIMENT / "xgpoc/temporal").glob("*.py")), *(EXPERIMENT / name for name in USED)]
    hashes = {str(p.relative_to(EXPERIMENT)): sha256_file(p) for p in files}
    return {"digest": canonical_digest(hashes), "files": dict(sorted(hashes.items()))}


def peak_rss_gib() -> float:
    """The peak resident memory of this process so far (Linux reports ru_maxrss in KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def json_default(value):
    """JSON for numpy scalars and arrays, keeping ints and bools as such."""
    if isinstance(value, np.generic | np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")
