"""Completion records of the detector cache: what produced each configuration's files (spec, Extraction: Cache).

A record names the configuration and its settings, the sha256 of every model file (detector, keypoints, the SigLIP
snapshot), of every code file that ran (this package, the run script and the tactifoot_vision package of the same
checkout), of the inputs (the index file and the list of shot ids), of every output file, plus the git state,
library versions, timing and coverage. It is written last, so a folder without `complete.json` is incomplete.
"""

import json
from pathlib import Path

from xgpoc.bundle import canonical_digest, code_version, sha256_file

EXPERIMENT = Path(__file__).resolve().parents[2]          # experiments/claude_fable
CHECKOUT = EXPERIMENT.parents[1]


def code_hashes(script: Path) -> dict:
    """sha256 of every file of `xgpoc/detector`, the run script and the package (`src/tactifoot_vision`), and a digest."""
    files = sorted((EXPERIMENT / "xgpoc/detector").glob("*.py")) + [Path(script).resolve()]
    files += sorted((CHECKOUT / "src/tactifoot_vision").rglob("*.py"))
    hashes = {str(p.relative_to(CHECKOUT)): sha256_file(p) for p in files}
    for name in ("xgpoc/__init__.py", "xgpoc/bundle.py"):
        hashes[str((EXPERIMENT / name).relative_to(CHECKOUT))] = sha256_file(EXPERIMENT / name)
    return {"digest": canonical_digest(hashes), "files": dict(sorted(hashes.items()))}


def model_hashes(detector: Path, keypoints: Path, siglip_name: str) -> dict:
    """sha256 of the detector and keypoint checkpoints and of every file of the cached SigLIP snapshot (no download)."""
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(siglip_name, local_files_only=True))
    siglip = {p.name: sha256_file(p) for p in sorted(snapshot.iterdir()) if p.is_file()}
    return {"detector": {"path": str(detector), "sha256": sha256_file(detector)},
            "keypoints": {"path": str(keypoints), "sha256": sha256_file(keypoints)},
            "siglip": {"name": siglip_name, "revision": snapshot.name, "files": siglip}}


def library_versions() -> dict:
    from importlib.metadata import version

    return {name: version(name) for name in ("torch", "ultralytics", "transformers", "supervision", "opencv-python-headless",
                                             "scikit-learn", "numpy", "pandas")}


def verified_file(path: Path) -> str:
    """sha256 of a file after checking it against the completion record (`complete.json`) of its folder."""
    digest = sha256_file(path)
    if json.loads((Path(path).parent / "complete.json").read_text())["files"].get(Path(path).name) != digest:
        raise ValueError(f"{path} does not match its completion record")
    return digest


def output_hashes(folder: Path) -> dict:
    return {p.name: sha256_file(p) for p in sorted(Path(folder).iterdir()) if p.is_file() and p.name != "complete.json"}


def git_state() -> dict:
    return code_version(CHECKOUT)
