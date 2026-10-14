#!/usr/bin/env bash
# Segmented ranged download: the rest of <output> (from its current size to <total>) in <n> segments, at most
# <par> at a time, then joined. Segment boundaries depend only on <output>'s size and <n>, so a rerun resumes.
# Drive throttles concurrent ranges of one file ("quota exceeded"); keep <par> small.
# Usage: fetch_parallel.sh <url> <output> <total_bytes> [n_segments] [par]
set -euo pipefail
url=$1 out=$2 total=$3 n=${4:-6} par=${5:-$n}
here=$(dirname "$0")
touch "$out"
have=$(stat -c %s "$out")
step=$(( ( (total - have) / n / 268435456 + 1 ) * 268435456 ))
segments=0
running=()
for (( k = 0; k < n; k++ )); do
  s=$(( have + k * step )); e=$(( s + step )); (( e > total )) && e=$total
  (( s >= total )) && break
  "$here/fetch_ranged.sh" "$url" "$out.seg$k" "$e" 67108864 "$s" > "$out.seg$k.log" 2>&1 &
  running+=($!)
  segments=$(( k + 1 ))
  if (( ${#running[@]} >= par )); then
    for pid in "${running[@]}"; do wait "$pid"; done
    running=()
  fi
done
for pid in "${running[@]}"; do wait "$pid"; done
for (( k = 0; k < segments; k++ )); do cat "$out.seg$k" >> "$out"; rm -f "$out.seg$k"; done
[[ $(stat -c %s "$out") -eq $total ]] && echo DONE || { echo "size mismatch" >&2; exit 1; }
