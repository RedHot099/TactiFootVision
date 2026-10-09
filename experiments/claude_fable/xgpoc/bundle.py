"""Trained-parameter bundles: save only what training changed, rebuild the rest.

A fine-tuned model that starts from a public checkpoint differs from that checkpoint in a
small set of tensors (for G-combo: the last four ViT-L blocks, the final norm and the head,
56M of 359M parameters). A bundle stores exactly those tensors (the EMA weights at the
selected epoch), plus everything needed to trust them later:

    spec         how to rebuild the architecture (consumed by the caller's factory)
    backbone     name, weight file name, hub repo/revision and sha256 of the pretrained weights
    calibration  per-model calibration constants (temperature, penalty rate, ...)
    recipe       training configuration and the selected epoch, for the record
    commit       git sha of the code that trained it (and whether the tree was dirty)
    provenance   digest and file hashes of the source set that trained it, and the sha256 of
                 the index, folds file and overlay mask it trained on; consumers assert these
                 with `check_data_files` before scoring so they cannot silently use other folds

Loading builds the model from the pretrained weights through a factory, refuses a weight
file whose hash differs from the one recorded, checks that the bundle holds exactly the
parameters the rebuilt model treats as trainable, and copies them in. Only tensors and
JSON-like values are stored, so a bundle loads with `torch.load(weights_only=True)`.
"""

import hashlib
import json
import os
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

FORMAT = "xgpoc-trained-params"
VERSION = 2
DATA_KEYS = ("index_sha256", "folds_sha256", "overlay_sha256")
FIELDS = ("spec", "backbone", "calibration", "recipe", "commit", "provenance", "params")


class BundleError(ValueError):
    """The bundle is malformed, or does not match the model/backbone it is loaded into."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(2**20):
            h.update(block)
    return h.hexdigest()


def canonical_digest(obj) -> str:
    """sha256 of the canonical JSON of `obj` (sorted keys, no whitespace)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def backbone_identity(name: str, weight_file: Path, **source) -> dict:
    """Which pretrained weights a model was built on: a name, the weight file's name and
    content hash, and `source` (e.g. hub repo and revision)."""
    return {"name": name, "file": Path(weight_file).name, "sha256": sha256_file(weight_file), **source}


def make_provenance(code_files: Mapping[str, str], *, index_sha256: str, folds_sha256: str, overlay_sha256: str) -> dict:
    """`code_files`: relative path -> sha256 of every source file loaded by the training run."""
    files = dict(code_files)
    return {"code": {"digest": canonical_digest(files), "files": files},
            "data": {"index_sha256": index_sha256, "folds_sha256": folds_sha256, "overlay_sha256": overlay_sha256}}


def check_data_files(provenance: Mapping, **files: Path) -> None:
    """Raise BundleError unless each file hashes to what the bundle was trained on, e.g.
    `check_data_files(b.provenance, folds_sha256=folds_path, overlay_sha256=overlay_path)`."""
    for key, path in files.items():
        if key not in DATA_KEYS:
            raise KeyError(f"{key!r} is not one of {DATA_KEYS}")
        actual = sha256_file(path)
        if actual != provenance["data"][key]:
            raise BundleError(f"{path} hashes to {actual}, the bundle was trained with {key} {provenance['data'][key]}")


def trained_keys(model: nn.Module) -> list[str]:
    """State-dict keys of the tensors a bundle stores: trainable parameters and buffers.
    Frozen parameters come back from the pretrained weights; buffers are kept because
    nothing guarantees the pretrained file reproduces them (the ViT has none). A submodule
    registered twice (FinetuneViTXG.trainable_blocks) is stored once, under its first name."""
    names = {n for n, p in model.named_parameters() if p.requires_grad} | {n for n, _ in model.named_buffers()}
    return [k for k in model.state_dict() if k in names]


def code_version(repo: Path | None = None) -> dict:
    """Commit the working tree is based on; `dirty` flags uncommitted changes to tracked files
    (the provenance source hashes pin the exact bytes in that case)."""
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()
    return {"sha": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def _validate(payload) -> dict:
    if not isinstance(payload, dict):
        raise BundleError(f"bundle payload is a {type(payload).__name__}, not a dict")
    if payload.get("format") != FORMAT or payload.get("version") != VERSION:
        raise BundleError(f"not a {FORMAT} v{VERSION} bundle")
    missing = [f for f in FIELDS if f not in payload]
    if missing:
        raise BundleError(f"bundle lacks {missing}")
    params = payload["params"]
    if not isinstance(params, dict) or not params or not all(
            isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in params.items()):
        raise BundleError("bundle params must be a non-empty dict of named tensors")
    prov = payload["provenance"]
    try:
        files = prov["code"]["files"]
        digest_ok = prov["code"]["digest"] == canonical_digest(files)
        data_ok = all(isinstance(prov["data"][k], str) for k in DATA_KEYS)
    except (KeyError, TypeError) as e:
        raise BundleError(f"malformed provenance: {e!r}") from e
    if not (digest_ok and data_ok):
        raise BundleError("provenance digest does not match its file hashes, or data hashes are missing")
    return payload


def write_bundle(path: Path, payload: dict) -> dict:
    """Validate and atomically write `payload`. Returns {"bytes", "sha256", "n_tensors", "n_params"}."""
    _validate(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    params = payload["params"]
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path), "n_tensors": len(params),
            "n_params": sum(v.numel() for v in params.values())}


def save_bundle(path: Path, model: nn.Module, *, spec: dict, backbone: dict, calibration: dict, recipe: dict,
                commit: dict, provenance: dict) -> dict:
    """Write `model`'s trained tensors and metadata to `path` (atomically)."""
    state = model.state_dict()
    params = {k: state[k].detach().cpu().clone().contiguous() for k in trained_keys(model)}
    return write_bundle(path, {"format": FORMAT, "version": VERSION, "spec": spec, "backbone": backbone,
                               "calibration": calibration, "recipe": recipe, "commit": commit,
                               "provenance": provenance, "params": params})


def read_bundle(path: Path) -> dict:
    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    except Exception as e:  # pickle errors of every kind: a file that is not a bundle
        raise BundleError(f"{path} cannot be read as a bundle: {e}") from e
    return _validate(payload)


@dataclass
class LoadedBundle:
    model: nn.Module
    spec: dict
    backbone: dict
    calibration: dict
    recipe: dict
    commit: dict
    provenance: dict


def load_bundle(path: Path, factory: Callable[[dict], tuple[nn.Module, Path]]) -> LoadedBundle:
    """`factory(spec)` must return the architecture built on the pretrained weights (same
    trainable/frozen split as at training time) and the path of the weight file it used.
    The returned model is in eval mode on the CPU."""
    payload = read_bundle(path)
    model, weight_file = factory(payload["spec"])
    expected = payload["backbone"]["sha256"]
    actual = sha256_file(weight_file)
    if actual != expected:
        raise BundleError(f"pretrained weights {weight_file} hash {actual} differs from the bundle's {expected} "
                          f"({payload['backbone']['name']})")
    params = payload["params"]
    wanted = set(trained_keys(model))
    if set(params) != wanted:
        raise BundleError(f"bundle holds {len(params)} tensors, the rebuilt model trains {len(wanted)}: "
                          f"missing {sorted(wanted - set(params))[:3]}, unexpected {sorted(set(params) - wanted)[:3]}")
    try:
        result = model.load_state_dict(params, strict=False)
    except RuntimeError as e:  # shape or dtype mismatch
        raise BundleError(f"bundle tensors do not fit the rebuilt model: {e}") from e
    frozen = set(model.state_dict()) - wanted
    if result.unexpected_keys or set(result.missing_keys) != frozen:
        raise BundleError(f"state dict mismatch: unexpected {result.unexpected_keys}, "
                          f"missing {sorted(set(result.missing_keys) - frozen)}")
    model.eval()
    return LoadedBundle(model, payload["spec"], payload["backbone"], payload["calibration"], payload["recipe"],
                        payload["commit"], payload["provenance"])
