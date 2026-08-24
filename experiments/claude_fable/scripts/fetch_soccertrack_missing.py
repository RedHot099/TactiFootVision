"""Resume-fetch missing SoccerTrack v2 files from the Google Drive mirror.

Drive IDs come from the codex-branch downloader (fetch_soccertrack.py). Runs rounds
with backoff on quota errors (HTTP 429), keeps >= 8 GB free, skips existing files.

    ./run.sh python scripts/fetch_soccertrack_missing.py
"""

import shutil
import time
from pathlib import Path

import gdown

DEST = Path("/home/kuba/projects/football_data/xg_shots_full/soccertrack_v2")
GSR_IDS = {
    "117092": ("1wQtwcO6hsn7iMrd07r8-E3y2L19H8HCA", "1NOF9WQcVzgqV2cMUbCtw1X9pxvx3Ouoj"),
    "117093": ("18fRzxM5vEExz0WXOxVqGoOIPeEEFaur2", "1hMhlqrg5E28NSvYkGb4P_mlZmRO0IaZ6"),
    "118575": ("1yXFIt_LmqjD0UjJ9A_bpSvxYq91-ffpD", "1XnQcFw3dRjz-r6lDQ40kPdOGeyKtVxYC"),
    "118576": ("1j8yPViLAhFPjRPRa4fl0jjEK2JvwWe6U", "1vSfHhbKCDqGGZuVRb7WIo_dcBstiRmrk"),
    "118577": ("181u_SPmxBlMYE8J44bmUDkM6D3KFYTzE", "1ABDbhxaQJis6-3I3EBSnFM6Dww-ZQxG6"),
    "118578": ("1BRvL2Tded1vZ6I53xWzX9FyV5D8b4YLe", "1lle_oEVBZ1cwcmAeY9iuFAOG970MDPcz"),
    "128057": ("1WDo9Z_UIfyzFo6R0yAzbVsem70X5hiZ5", "1i_gpecmQbyqlpRe13NRxfzhbF9FK3VcB"),
    "128058": ("1TOMHgksG2yzVGauhpw6OaesgINFlLDE5", "1fZ72owSk5M9ykxKFNNu8jj6NbLSfRBie"),
    "132831": ("1w-MWIctfpzfDaa3ez2CXJM3h2irKdUOW", "1VfHXOBrjZrfNU-Z9YazlGI0BD3hPwa63"),
    "132877": ("19DcUJRhcZMMGoSkB-kvyiWBywLONpVci", "1QityYzZh0ZqZgITEL4535i0N_H4y-IR-"),
}
FULL_VIDEO_IDS = {
    "117092": ("1QMqmairWx_U8TCCPZ4ke56p_vkmLa1Ea", "1UThd91AkBiQAogezNnJoOB4kKAz0p6IS"),
    "117093": ("1gPoN2h2SxEjWORK6FwKTEsCuKPylar53", "1dJNSKYtiuynakFDO7Hj7dCp7yjb2iMXy"),
    "118575": ("1-FZNCqVSx8YleM2E7UK90-SAdNAlTeSD", "1p6N77kKsZZffpzX2IfipLHCt7mZMoPiK"),
    "118576": ("1W4orUiMHgr6P6ZbmT1-AafnGT6EB9Suy", "1LfGYXNZi6tgWgwd-MU65K5AJbv4qc3N2"),
    "118577": ("1JXm4ejrKWSrfTNgvkn-pCaWn2pXWxnlt", "1azEcdXrS48_JvFe_WougzhUQZHLMJuM0"),
    "118578": ("1M-XK17CWykXVPrycZBE6-MSMjfvnwRIu", "1lZtT_qkUpgfOI1LMbEaxNmXDeKzqrIAv"),
    "128057": ("1A2y5s0xgxU7yedVgqTW3bgLTZWfJO05C", "1Vah7favjAsm5yYCjU4ic8Vlcyi0whnQ4"),
    "128058": ("18uzsM20v5B9Nn7alW65UG4A2_tsBM_SC", "1ysEDbhKDzEDuE6cGl0THS_2LUEl-pnHX"),
    "132831": ("1xiIQijpN9lqNCGzk440rrKb126DvVGLt", "1I_lWwqJQdSOHzaTwH-ILixzhQmAYhTLP"),
    "132877": ("1t0krXIfOD-xLAWv0EqVB7oTom1Xqn4ss", "17yFORz6J-PaQTOPDBt0hFjI7BLrxyc85"),
}
HALF = {1: "1st", 2: "2nd"}


def jobs():
    out = []
    for match, ids in FULL_VIDEO_IDS.items():
        for half, fid in enumerate(ids, start=1):
            out.append((fid, DEST / "videos" / match / f"{match}_{HALF[half]}.mp4"))
    for match, ids in GSR_IDS.items():
        for half, fid in enumerate(ids, start=1):
            plain = DEST / "gsr" / match / f"{match}_{HALF[half]}.json"
            if plain.with_suffix(".json.zst").exists():
                continue  # already fetched and compressed
            out.append((fid, plain))
    return [(fid, p) for fid, p in out if not p.exists()]


def main():
    for round_no in range(1, 13):
        todo = jobs()
        if not todo:
            print("all files present")
            return
        print(f"round {round_no}: {len(todo)} files missing")
        for fid, path in todo:
            if shutil.disk_usage(DEST).free / 1e9 < 8:
                print("low disk (<8 GB), stopping")
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                got = gdown.download(id=fid, output=str(path), quiet=True, resume=True)
                print(("ok      " if got else "failed  ") + str(path.name), flush=True)
            except Exception as e:  # noqa: BLE001 - quota errors etc., retry next round
                print(f"error   {path.name}: {type(e).__name__}: {str(e)[:120]}", flush=True)
            time.sleep(20)
        time.sleep(600)
    print("gave up after 12 rounds; rerun later")


if __name__ == "__main__":
    main()
