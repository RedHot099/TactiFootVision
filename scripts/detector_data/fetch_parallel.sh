#!/usr/bin/env bash
# Parallel ranged download: the rest of <output> (from its current size to <total>) in <n> segments, then joined.
# Usage: fetch_parallel.sh <url> <output> <total_bytes> [n_segments]
set -euo pipefail
url=$1 out=$2 total=$3 n=${4:-6}
here=$(dirname "$0")
touch "$out"
have=$(stat -c %s "$out")
step=$(( ( (total - have) / n / 268435456 + 1 ) * 268435456 ))
pids=()
for (( k = 0; k < n; k++ )); do
  s=$(( have + k * step )); e=$(( s + step )); (( e > total )) && e=$total
  (( s >= total )) && break
  "$here/fetch_ranged.sh" "$url" "$out.seg$k" "$e" 67108864 "$s" > "$out.seg$k.log" 2>&1 &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
for (( k = 0; k < ${#pids[@]}; k++ )); do cat "$out.seg$k" >> "$out"; rm -f "$out.seg$k"; done
[[ $(stat -c %s "$out") -eq $total ]] && echo DONE || { echo "size mismatch" >&2; exit 1; }
