"""Code identity for the completion records of the xg-compare scripts."""

from pathlib import Path

from xgpoc.bundle import canonical_digest, sha256_file

EXPERIMENT = Path(__file__).resolve().parents[2]          # experiments/claude_fable
USED_VENDOR = ("draws.py", "evaltable.py", "ladder.py", "refit.py", "statsbomb_fetch.py", "statsbomb_tables.py")


def code_hashes(script: Path) -> dict:
    """sha256 of every file of `xgpoc/compare`, the vendor modules it uses, the calibration helpers and the run script,
    and a digest of them."""
    files = sorted((EXPERIMENT / "xgpoc/compare").glob("*.py")) + [EXPERIMENT / "xgpoc/vendor" / name for name in USED_VENDOR]
    files += [EXPERIMENT / "xgpoc/eval/calibration.py", EXPERIMENT / "xgpoc/probe/readability.py", Path(script).resolve()]
    hashes = {str(p.relative_to(EXPERIMENT)): sha256_file(p) for p in files}
    return {"digest": canonical_digest(hashes), "files": dict(sorted(hashes.items()))}
