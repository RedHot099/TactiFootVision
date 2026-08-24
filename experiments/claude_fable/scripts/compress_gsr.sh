#!/usr/bin/env bash
# Compress unused SoccerTrack GSR JSONs with zstd, keeping >=8 GB free at all times.
set -u
cd /home/kuba/projects/football_data/xg_shots_full/soccertrack_v2/gsr || exit 1
for f in */*.json; do
    [ -e "$f" ] || continue
    avail=$(df --output=avail -B1G . | tail -1)
    if [ "$avail" -lt 8 ]; then echo "low disk ($avail GB), stopping"; break; fi
    echo "compressing $f"
    zstd -T4 -8 --rm -q "$f"
done
du -sh .
