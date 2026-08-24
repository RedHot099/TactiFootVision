#!/usr/bin/env bash
# Integrity check for compressed GSR files: zstd test + decompressed size + JSON head.
for f in /home/kuba/projects/football_data/xg_shots_full/soccertrack_v2/gsr/*/*.zst; do
    if zstd -t -q "$f" 2>/dev/null; then
        sz=$(zstd -dc "$f" 2>/dev/null | wc -c)
        head=$(zstd -dc "$f" 2>/dev/null | head -c 60 | tr -d '\n')
        echo "OK $(basename "$f") raw=$((sz / 1000000))MB head=${head:0:40}"
    else
        echo "CORRUPT $(basename "$f")"
    fi
done
