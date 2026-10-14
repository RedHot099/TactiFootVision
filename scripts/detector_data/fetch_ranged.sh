#!/usr/bin/env bash
# Resumable download in fixed-size HTTP range requests.
# Google Drive answers "quota exceeded" to a plain download of a popular large file but still serves byte ranges.
# Usage: fetch_ranged.sh <url> <output> <total_bytes> [chunk_bytes]
set -euo pipefail
url=$1 out=$2 total=$3 chunk=${4:-268435456}
touch "$out"
while :; do
  have=$(stat -c %s "$out")
  (( have >= total )) && break
  end=$(( have + chunk - 1 )); (( end >= total )) && end=$(( total - 1 ))
  part="$out.part"
  for attempt in 1 2 3 4 5 6 7 8; do
    if curl -sS -L --fail -r "$have-$end" -o "$part" "$url" && [[ $(stat -c %s "$part") -eq $(( end - have + 1 )) ]]; then
      cat "$part" >> "$out"; rm -f "$part"; break
    fi
    echo "retry $attempt at $have" >&2; sleep $(( attempt * 30 ))
  done
  [[ -e "$part" ]] && { echo "failed at $have" >&2; exit 1; }
  echo "$(date +%T) $(( (end + 1) * 100 / total ))% $(( end + 1 ))/$total"
done
echo DONE
