#!/usr/bin/env bash
# Run anything in this experiment with the repo uv env and xgpoc importable.
# The branch's own package (../../src) comes first on the path, ahead of the shared
# venv's editable install (which points at the main checkout), so research code runs
# on the package of the same commit.
cd "$(dirname "$0")"
PYTHONPATH="$PWD:$(cd ../.. && pwd)/src:$PYTHONPATH" exec uv run --project /home/kuba/projects/tactifoot_vision "$@"
