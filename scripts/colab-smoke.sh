#!/usr/bin/env bash
# Install the built wheel the way pip would on the default Google Colab
# runtime (Python 3.13, the versions in tests/colab-constraints.txt), import
# the package, run the CLI and the fast suite on CPU. No uv.lock involved:
# this checks the ranges in pyproject.toml, not the lock. pip does the
# install, as in Colab; uv only creates the venv.
#
#   bash scripts/colab-smoke.sh                 # builds the wheel first
#   bash scripts/colab-smoke.sh path/to/x.whl   # a wheel built elsewhere
set -euo pipefail
cd "$(dirname "$0")/.."
if [ $# -eq 0 ]; then
    rm -rf dist && uv build --wheel -q -o dist
    set -- dist/*.whl
fi
[ $# -eq 1 ] || { echo "expected one wheel, got: $*" >&2; exit 2; }
WHEEL="$1"
VENV="${COLAB_SMOKE_VENV:-.venv-colab-smoke}"
rm -rf "$VENV"
uv venv -q -p 3.13 --seed "$VENV"   # --seed: a venv with pip
# The CPU wheels of torch keep the install small; the versions are the same.
# The dev tools let the whole fast suite run (notebook build, sdist test).
"$VENV/bin/python" -m pip install -q --disable-pip-version-check \
    --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
    -c tests/colab-constraints.txt "$WHEEL" pytest nbclient nbformat ipykernel hatchling hatch-vcs
"$VENV/bin/python" - <<'PY'
import importlib.metadata as metadata
import pathlib

pins = {}
for line in pathlib.Path("tests/colab-constraints.txt").read_text().splitlines():
    if "==" in line and not line.startswith("#"):
        name, version = line.split("==")
        pins[name.lower().replace("_", "-")] = version
installed = {d.metadata["Name"].lower().replace("_", "-"): d.version for d in metadata.distributions()}
differ = {n: (v, installed[n]) for n, v in pins.items() if n in installed and installed[n].split("+")[0] != v}
assert not differ, f"installed versions differ from the Colab constraints: {differ}"
import tactifoot_vision as tv
from transformers import SiglipImageProcessor, SiglipVisionModel  # the team embedder's classes

print("tactifoot-vision", tv.__version__, "imports; Pipeline:", tv.Pipeline.__name__)
PY
"$VENV/bin/tactifoot" info
"$VENV/bin/python" -m pytest -q -p no:cacheprovider tests
