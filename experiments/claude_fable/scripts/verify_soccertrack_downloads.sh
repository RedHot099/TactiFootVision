#!/usr/bin/env bash
# Verify manually-downloaded SoccerTrack v2 files, compress GSR, report what's missing.
set -u
BASE=/home/kuba/projects/football_data/xg_shots_full/soccertrack_v2
missing=0

for m in 117092 117093 118575 118576 118577 118578 128057 128058 132831 132877; do
    for h in 1st 2nd; do
        v="$BASE/videos/$m/${m}_${h}.mp4"
        if [ ! -f "$v" ]; then echo "MISSING video $m/${m}_${h}.mp4"; missing=$((missing+1)); continue; fi
        dur=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$v" 2>/dev/null | cut -d. -f1)
        if [ -z "$dur" ] || [ "$dur" -lt 2000 ]; then echo "BAD     video $m/${m}_${h}.mp4 (duration ${dur:-?} s - incomplete?)"; missing=$((missing+1)); else echo "ok      video $m/${m}_${h}.mp4 (${dur} s)"; fi
    done
done

for m in 117092 117093 118575 118576 118577 118578 128057 128058 132831 132877; do
    for h in 1st 2nd; do
        j="$BASE/gsr/$m/${m}_${h}.json"
        z="$j.zst"
        if [ -f "$j" ]; then
            sz=$(stat -c %s "$j")
            if [ "$sz" -lt 2000000000 ]; then echo "BAD     gsr $m/${m}_${h}.json (only $((sz/1000000)) MB - incomplete?)"; missing=$((missing+1)); continue; fi
            if ! head -c 20 "$j" | grep -q "{"; then echo "BAD     gsr $m/${m}_${h}.json (not JSON)"; missing=$((missing+1)); continue; fi
            avail=$(df --output=avail -B1G "$BASE" | tail -1)
            if [ "$avail" -lt 8 ]; then echo "low disk, skipping compression"; else
                zstd -T4 -8 --rm -q "$j" && echo "ok      gsr $m/${m}_${h}.json (compressed)"
            fi
        elif [ -f "$z" ]; then
            if zstd -t -q "$z" 2>/dev/null; then echo "ok      gsr $m/${m}_${h}.json.zst"; else echo "BAD     gsr $m/${m}_${h}.json.zst (corrupt archive)"; missing=$((missing+1)); fi
        else
            echo "MISSING gsr $m/${m}_${h}.json"; missing=$((missing+1))
        fi
    done
done

echo "---"
if [ "$missing" -eq 0 ]; then echo "COMPLETE: all 20 video halves + 20 GSR present and valid"; else echo "$missing item(s) missing or bad - see above and DOWNLOAD_CHECKLIST.md"; fi
