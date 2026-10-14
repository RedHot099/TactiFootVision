"""Code identity of xg-stack: the files the protocol pins and the run records."""

from pathlib import Path

from xgpoc.bundle import canonical_digest, sha256_file

EXPERIMENT = Path(__file__).resolve().parents[2]          # experiments/claude_fable
USED = ("xgpoc/__init__.py", "xgpoc/bundle.py", "xgpoc/paths.py", "xgpoc/compare/stats.py", "xgpoc/detector/approach1.py",
        "xgpoc/detector/conventions.py", "xgpoc/detector/extract.py", "xgpoc/detector/extract3.py",
        "xgpoc/detector/features.py", "xgpoc/detector/guard.py", "xgpoc/detector/records.py", "xgpoc/detector/sanitation.py",
        "xgpoc/detector/seal.py", "xgpoc/detector/testrule.py", "xgpoc/eval/calibration.py", "xgpoc/eval/metrics.py",
        "xgpoc/probe/readability.py", "xgpoc/vendor/refit.py", "scripts/xgs_protocol.py")


def code_hashes(script: Path | None = None) -> dict:
    """sha256 of every file of `xgpoc/stack`, the modules it uses and the protocol writer (`USED`), plus `script` when
    given, and a digest of them. The protocol pins the digest without a run script; a run record adds its script."""
    files = sorted((EXPERIMENT / "xgpoc/stack").glob("*.py")) + [EXPERIMENT / name for name in USED]
    if script is not None:
        files.append(Path(script).resolve())
    hashes = {str(p.relative_to(EXPERIMENT)): sha256_file(p) for p in files}
    return {"digest": canonical_digest(hashes), "files": dict(sorted(hashes.items()))}
