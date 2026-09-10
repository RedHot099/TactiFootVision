"""Inventory full local matches and prepare human completeness checks, without declaring a test set."""

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from diagnostic_review import json_text
from xgpoc.data.soccertrack_bas import parse_bas
from xgpoc.v13.protocol import digest


def readiness(halves, attempts):
    required = ["match_id", "half", "duration_ms", "reviewed_from_ms", "reviewed_until_ms", "reviewer",
                "all_attempts_checked", "clock_alignment_checked", "overlap_status", "goal_count_verified"]
    if not set(required).issubset(halves.columns):
        raise ValueError("Missing half-review columns")
    if not {"review_id", "match_id", "half", "is_attempt", "outcome", "contact_lower_ms", "contact_upper_ms",
            "reviewer"}.issubset(attempts.columns):
        raise ValueError("Missing attempt-review columns")
    if halves.duplicated(["match_id", "half"]).any() or attempts.review_id.duplicated().any():
        raise ValueError("Duplicate halves or attempt IDs")
    pairs = set(zip(halves.match_id, halves.half))
    if not set(zip(attempts.match_id, attempts.half)).issubset(pairs):
        raise ValueError("Attempt refers to an unknown match or half")
    rows = []
    for match, group in halves.groupby("match_id"):
        problems = []
        if set(group.half) != {1, 2}:
            problems.append("both_halves_required")
        for row in group.itertuples():
            prefix = f"half_{row.half}:"
            if pd.isna(row.reviewer) or not str(row.reviewer).strip():
                problems.append(prefix + "human_reviewer_missing")
            if not (pd.notna(row.reviewed_from_ms) and pd.notna(row.reviewed_until_ms) and
                    row.reviewed_from_ms == 0 and row.reviewed_until_ms >= row.duration_ms):
                problems.append(prefix + "full_video_review_missing")
            for key in ("all_attempts_checked", "clock_alignment_checked"):
                if getattr(row, key) != "yes":
                    problems.append(prefix + key)
            evidence = getattr(row, "overlap_evidence", "")
            if row.overlap_status != "distinct" or pd.isna(evidence) or not str(evidence).strip():
                problems.append(prefix + "dataset_overlap_unresolved")
            candidates = attempts[(attempts.match_id == match) & (attempts.half == row.half)]
            if candidates.empty or not candidates.is_attempt.isin(["yes", "no"]).all():
                problems.append(prefix + "shot_intent_unresolved")
            actual = candidates[candidates.is_attempt.eq("yes")]
            if not actual.outcome.isin(["goal", "no_goal"]).all():
                problems.append(prefix + "outcomes_unresolved")
            if actual.contact_lower_ms.isna().any() or actual.contact_upper_ms.isna().any() or \
                    (actual.contact_lower_ms > actual.contact_upper_ms).any() or (actual.contact_lower_ms < 0).any() or \
                    (actual.contact_upper_ms > row.duration_ms).any():
                problems.append(prefix + "contacts_unresolved")
            if pd.isna(row.goal_count_verified) or int(actual.outcome.eq("goal").sum()) != row.goal_count_verified:
                problems.append(prefix + "goal_accounting_unresolved")
            if actual.reviewer.fillna("").astype(str).str.strip().eq("").any():
                problems.append(prefix + "attempt_reviewer_missing")
        rows.append({"match_id": str(match), "ready_for_protocol_review": not problems, "open_checks": problems})
    return rows


def prepare(root, out):
    if out.exists():
        raise ValueError("Use a new inventory directory; do not overwrite human work")
    halves, attempts, sources = [], [], {}
    for bas in sorted((root / "bas").glob("*_12_class_events.json")):
        match = str(json.loads(bas.read_text())["match_id"])
        videos = [root / "videos" / match / f"{match}_{half}.mp4" for half in ("1st", "2nd")]
        if not all(path.is_file() for path in videos):
            continue
        sources[str(bas)] = digest(bas)
        for half, video in enumerate(videos, 1):
            info = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                                      "-of", "json", str(video)], text=True))
            duration = round(float(info["format"]["duration"]) * 1000)
            sources[str(video)] = digest(video)
            halves.append({"match_id": match, "half": half, "video_path": str(video), "duration_ms": duration,
                           "reviewed_from_ms": None, "reviewed_until_ms": None, "reviewer": "",
                           "all_attempts_checked": "unknown", "clock_alignment_checked": "unknown",
                           "overlap_status": "unknown", "overlap_evidence": "", "goal_count_verified": None})
        for i, attempt in enumerate(parse_bas(bas)):
            # Source events are candidates only. A missing goal anchor is not a shot timestamp.
            source = asdict(attempt)
            attempts.append({"review_id": f"{match}-{i:04d}", "match_id": match, "half": attempt.half,
                             "candidate_video_ms": attempt.video_ms, "source_label": attempt.label,
                             "source_goal": attempt.goal, "source_trusted": attempt.trusted, "source_reason": attempt.reason,
                             "source_team": source["team"], "is_attempt": "unknown", "outcome": "unknown",
                             "contact_lower_ms": None, "contact_upper_ms": None, "reviewer": "", "note": ""})
    if not halves:
        raise ValueError("No complete matches found")
    out.mkdir(parents=True)
    h, a = pd.DataFrame(halves), pd.DataFrame(attempts)
    h.to_csv(out / "half_review.csv", index=False)
    a.to_csv(out / "attempt_review.csv", index=False)
    (out / "readiness.json").write_text(json_text(readiness(h, a)))
    summary = {"complete_matches": h.match_id.nunique(), "halves": len(h),
               "video_hours": float(h.duration_ms.sum() / 3600000), "candidate_rows": len(a),
               "source_goal_rows": int(a.source_goal.sum()),
               "source_untrusted_goal_rows": int((a.source_goal & ~a.source_trusted).sum()),
               "eligible_confirmatory_test_matches": 0,
               "note": "Human full-video review, shot intent, goal accounting, timing and cross-dataset identity "
                       "are required. Add missed attempts to a copy of attempt_review.csv. These are pilot candidates."}
    (out / "summary.json").write_text(json_text(summary))
    (out / "manifest.json").write_text(json_text({"sources": sources, "script_sha256": digest(Path(__file__)),
        "parser_sha256": digest(Path("xgpoc/data/soccertrack_bas.py")),
        "templates": {name: digest(out / name) for name in ("half_review.csv", "attempt_review.csv")}}))
    print(json_text(summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--root", type=Path, default=Path("/home/kuba/projects/football_data/xg_shots_full/soccertrack_v2"))
    prep.add_argument("--out", type=Path, default=Path("research/v15_diagnostics/external_pilot"))
    check = commands.add_parser("check")
    check.add_argument("--halves", type=Path, required=True)
    check.add_argument("--attempts", type=Path, required=True)
    check.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.root, args.out)
    else:
        if args.out.exists():
            raise ValueError("Readiness output already exists")
        result = readiness(pd.read_csv(args.halves), pd.read_csv(args.attempts))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json_text({"readiness": result, "halves_sha256": digest(args.halves),
                                      "attempts_sha256": digest(args.attempts),
                                      "warning": "Readiness checks human declarations, not their truth; no test is automatically opened."}))
        print(json_text(result))


if __name__ == "__main__":
    main()
