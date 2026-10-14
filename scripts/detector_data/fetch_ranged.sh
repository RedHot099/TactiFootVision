#!/usr/bin/env bash
# Resumable download in fixed-size HTTP range requests.
# Google Drive answers "quota exceeded" to a plain download of a popular large file but still serves byte ranges.
# Usage: fetch_ranged.sh <url> <output> <end_bytes> [chunk_bytes] [start_byte]
# <output> holds bytes [start_byte, end_bytes) of the remote file, so several segments can run in parallel and be
# concatenated afterwards (see fetch_parallel.sh).
set -euo pipefail
url=$1 out=$2 total=$3 chunk=${4:-67108864} start=${5:-0}
touch "$out"
while :; do
  have=$(( start + $(stat -c %s "$out") ))
  (( have >= total )) && break
  end=$(( have + chunk - 1 )); (( end >= total )) && end=$(( total - 1 ))
  part="$out.part"
  for attempt in $(seq 1 20); do
    if curl -sS -L --fail -r "$have-$end" -o "$part" "$url" && [[ $(stat -c %s "$part") -eq $(( end - have + 1 )) ]]; then
      cat "$part" >> "$out"; rm -f "$part"; break
    fi
    echo "retry $attempt at $have" >&2; sleep $(( attempt * 30 > 300 ? 300 : attempt * 30 ))
  done
  [[ -e "$part" ]] && { echo "failed at $have" >&2; exit 1; }
  echo "$(date +%T) $(( (end + 1) * 100 / total ))% $(( end + 1 ))/$total"
done
echo DONE
