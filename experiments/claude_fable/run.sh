#!/usr/bin/env bash
# Run anything in this experiment with the repo uv env and xgpoc importable.
cd "$(dirname "$0")"
PYTHONPATH="$PWD:$PYTHONPATH" exec uv run --project /home/kuba/projects/tactifoot_vision "$@"
