#!/usr/bin/env bash
# Install the built wheel the way pip would on the default Google Colab
# runtime (Python 3.13, the versions in tests/colab-constraints.txt), import
# the package, run the CLI and the fast suite on CPU. No uv.lock involved:
# this checks the ranges in pyproject.toml, not the lock.
#
#   bash scripts/colab-smoke.sh            # builds dist/ first
#   bash scripts/colab-smoke.sh dist/*.whl # a wheel built elsewhere
set -euo pipefail
cd "$(dirname "$0")/.."
WHEEL="${1:-}"
if [ -z "$WHEEL" ]; then
    rm -rf dist && uv build --wheel -q -o dist
    WHEEL=$(ls dist/*.whl)
fi
VENV="${COLAB_SMOKE_VENV:-.venv-colab-smoke}"
rm -rf "$VENV"
uv venv -q -p 3.13 "$VENV"
# The CPU wheels of torch keep the install small; the versions are the same.
uv pip install -q -p "$VENV/bin/python" \
    --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match \
    -c tests/colab-constraints.txt "$WHEEL" pytest
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
"$VENV/bin/python" -m pytest -q -p no:cacheprovider tests \
    --ignore=tests/test_notebook_build.py --ignore=tests/test_packaging.py
