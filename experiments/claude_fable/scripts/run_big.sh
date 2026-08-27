#!/usr/bin/env bash
# Big training run: every PoC to convergence, 3 seeds, plus a DINOv2-B ablation.
# Sequential on one GPU. Resumable at PoC granularity: a finished (poc, seed) leaves
# its metrics json, and we skip it on rerun.
set -u
cd "$(dirname "$0")/.."
INDEX=/home/kuba/projects/football_data/xg_shots_full/soccernet_720p/index.parquet
CACHE=cache/full_v1
OUT=results/big_v1

run() { # run <seed> <poc> <epochs> <minutes> <marker-file>
    local seed=$1 poc=$2 epochs=$3 minutes=$4 marker=$5
    if [ -f "$OUT/seed$seed/$marker" ]; then echo "skip $poc seed $seed"; return; fi
    echo "=== $poc seed $seed ($(date +%H:%M)) ==="
    ./run.sh python scripts/run_poc.py --index "$INDEX" --cache "$CACHE" \
        --out "$OUT/seed$seed" --poc "$poc" --epochs "$epochs" --max-minutes "$minutes" --seed "$seed"
}

for seed in 0 1 2; do
    if [ ! -f "$OUT/c8/seed$seed/c_cnn_e2e_metrics.json" ]; then
        echo "=== c8 seed $seed ($(date +%H:%M)) ==="
        ./run.sh python scripts/run_poc.py --index "$INDEX" --cache "$CACHE" \
            --out "$OUT/c8/seed$seed" --poc c --epochs 8 --max-minutes 20 --seed "$seed"
    fi
done

for seed in 0 1 2; do
    run "$seed" a 1 1 a_tabular_gbm_metrics.json
    run "$seed" b 120 12 b_set_transformer_metrics.json
    run "$seed" d 100 25 d_frozen_dinov2_s_metrics.json
    run "$seed" e 100 25 e_hybrid_dinov2_s_metrics.json
    run "$seed" c 40 90 c_cnn_e2e_metrics.json
done

# DINOv2-B ablation, one seed (feature extraction happens on first use)
if [ ! -f "$OUT/dinov2_b/d_frozen_dinov2_b_metrics.json" ]; then
    echo "=== dinov2_b ablation ($(date +%H:%M)) ==="
    ./run.sh python - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path("scripts").resolve()))
import run_poc as rp
from xgpoc.data.index import load_index
from xgpoc.models.common import TrainConfig
df = load_index(Path("/home/kuba/projects/football_data/xg_shots_full/soccernet_720p/index.parquet"))
tr, va, te = (df[df["split"] == s].reset_index(drop=True) for s in ("train", "valid", "test"))
cfg = TrainConfig(epochs=100, max_minutes=35, batch_size=32, seed=0)
rp.run_d(tr, va, te, Path("cache/full_v1"), Path("results/big_v1/dinov2_b"), cfg, backbone="dinov2_b")
rp.run_e(tr, va, te, Path("cache/full_v1"), Path("results/big_v1/dinov2_b"), cfg, backbone="dinov2_b")
PY
fi
echo "BIG RUN DONE ($(date +%H:%M))"
