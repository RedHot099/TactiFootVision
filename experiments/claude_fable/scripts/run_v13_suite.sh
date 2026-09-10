#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
result_root="${1:-results/v13_final}"
if [[ "${2:-}" != "--prepared" ]]; then
    ./run.sh python scripts/prepare_v13_inputs.py
fi
./run.sh python scripts/run_v13.py --out "$result_root" --variants g_base g_pre400 --seeds 0 1
./run.sh python scripts/run_v13.py --out "$result_root" --variants full roi control time_mean time_gru geometry time_mean_t0 time_gru_t0 geometry_zero --seeds 0 1 2
./run.sh python scripts/verify_v13_bundle.py "$result_root/g_base/seed0/bundle.pt"
./run.sh python scripts/verify_v13_bundle.py "$result_root/roi/seed0/bundle.pt"
./run.sh python scripts/evaluate_v13.py --root "$result_root" --mode screen
