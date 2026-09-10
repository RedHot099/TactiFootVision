"""Check video coverage and goal-anchor completeness before external xG evaluation."""

import argparse
import json
from pathlib import Path

from xgpoc.data.soccertrack_bas import parse_bas
from xgpoc.v13.protocol import digest


def audit(root):
    rows = []
    for path in sorted((root / "bas").glob("*_12_class_events.json")):
        attempts = parse_bas(path)
        match = str(json.loads(path.read_text())["match_id"])
        videos = [root / "videos" / match / f"{match}_{half}.mp4" for half in ("1st", "2nd")]
        total_goals = sum(a.goal for a in attempts)
        anchored_goals = sum(a.goal and a.trusted for a in attempts)
        complete = all(p.is_file() for p in videos)
        rows.append({"match": match, "complete_video": complete,
                     "trusted_attempts": sum(a.trusted for a in attempts),
                     "total_goals": total_goals, "trusted_anchored_goals": anchored_goals,
                     "eligible_for_xg_evaluation": complete and total_goals == anchored_goals,
                     "bas_sha256": digest(path)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/kuba/projects/football_data/xg_shots_full/soccertrack_v2"))
    parser.add_argument("--out", type=Path, default=Path("research/v13_data_inventory/external_eligibility.json"))
    args = parser.parse_args()
    rows = audit(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"matches": rows,
        "limitation": "Eligibility checks coverage only; timestamp accuracy and shot intent still need visual validation."}, indent=2) + "\n")
    print(f"Complete videos: {sum(r['complete_video'] for r in rows)}/{len(rows)} matches")
    print(f"Complete goal anchors: {sum(r['eligible_for_xg_evaluation'] for r in rows)}/{len(rows)} matches")


if __name__ == "__main__":
    main()
