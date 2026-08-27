#!/usr/bin/env bash
# Autonomous improvement queue (sequential GPU, marker-resumable).
set -u
cd "$(dirname "$0")/.."
IDX=/home/kuba/projects/football_data/xg_shots_full/soccernet_720p/index_v1_1.parquet
CACHE=cache/full_v1
OUT=results/night_v1

poc() { # poc <outdir> <marker> <args...>
    local dir=$1 marker=$2; shift 2
    if [ -f "$OUT/$dir/$marker" ]; then echo "skip $dir/$marker"; return; fi
    echo "=== $dir ($(date +%H:%M)) ==="
    ./run.sh python scripts/run_poc.py --index "$IDX" --cache "$CACHE" --out "$OUT/$dir" "$@"
}

# 1. C at higher resolution (matched 8-epoch cycle), 3 seeds; 896x512 single seed
for s in 0 1 2; do
    poc "c672/seed$s" c_cnn_e2e_672x384_metrics.json --poc c --img-size 672x384 --epochs 8 --max-minutes 40 --seed "$s"
done
poc "c896/seed0" c_cnn_e2e_896x512_metrics.json --poc c --img-size 896x512 --epochs 8 --max-minutes 60 --seed 0

# 2. C on the -400 ms frame (label-noise ablation), 3 seeds
for s in 0 1 2; do
    poc "coff400/seed$s" c_cnn_e2e_off-400_metrics.json --poc c --frame-col frame_path_-400 --epochs 8 --max-minutes 20 --seed "$s"
done

# 3. F temporal (frozen dinov2_s over offsets -1600..0), 3 seeds (extraction cached once)
for s in 0 1 2; do
    poc "f/seed$s" f_temporal_dinov2_s_metrics.json --poc f --epochs 100 --max-minutes 30 --seed "$s"
done

# 4. D with DINOv2-L, 3 seeds (extraction cached once)
if [ ! -f "$OUT/dl/seed2/d_frozen_dinov2_l_metrics.json" ]; then
    for s in 0 1 2; do
        if [ -f "$OUT/dl/seed$s/d_frozen_dinov2_l_metrics.json" ]; then continue; fi
        echo "=== dl/seed$s ($(date +%H:%M)) ==="
        ./run.sh python - <<PY
import sys
from pathlib import Path
sys.path.insert(0, str(Path("scripts").resolve()))
import run_poc as rp
from xgpoc.data.index import load_index
from xgpoc.models.common import TrainConfig
df = load_index(Path("$IDX"))
tr, va, te = (df[df["split"] == s].reset_index(drop=True) for s in ("train", "valid", "test"))
cfg = TrainConfig(epochs=100, max_minutes=45, batch_size=32, seed=$s)
rp.run_d(tr, va, te, Path("$CACHE"), Path("$OUT/dl/seed$s"), cfg, backbone="dinov2_l")
PY
    done
fi

# 5. Positional cache with ball fallback + A/B rerun
if [ ! -f cache/full_v2_fallback/positional/tokens.npz ]; then
    echo "=== fallback positional cache ($(date +%H:%M)) ==="
    ./run.sh python scripts/build_positional_cache.py "$IDX" cache/full_v2_fallback/positional --ball-fallback
fi
for s in 0 1 2; do
    if [ -f "$OUT/ab_fallback/seed$s/b_set_transformer_metrics.json" ]; then continue; fi
    echo "=== ab_fallback seed$s ($(date +%H:%M)) ==="
    ./run.sh python scripts/run_poc.py --index "$IDX" --cache cache/full_v2_fallback --out "$OUT/ab_fallback/seed$s" \
        --poc a b --epochs 120 --max-minutes 12 --seed "$s"
done

echo "NIGHT RUN DONE ($(date +%H:%M))"
