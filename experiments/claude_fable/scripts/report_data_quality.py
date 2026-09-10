"""Data quality report for a canonical data version: technical defects separated from hard but valid shots.

Flags never remove a shot. Technical flags describe inputs that do not show the live shot as intended
(decode failure, near-black image, exact duplicate, suspected camera cut or replay inside the input window,
partition conflict with a neighbouring attempt). Review flags describe genuinely hard or unresolved cases
(occlusion, uncertain contact, unknown outcome, wide contact interval, neighbouring attempt); removing them
would make the data easier than real broadcast footage.
"""

import argparse
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from diagnostic_review import json_text
from prepare_canonical_data import complete, verify_result
from xgpoc.data.canonical_frames import probe_video
from xgpoc.paths import INDEX
from xgpoc.v13.protocol import assign_roles, digest

WINDOW_MS = (-1600, 0)
POST_WINDOW_MS = 4000
FRAME_STEP_MS = 40
CATEGORIES = {
    "decode_failed": "technical", "near_black_frame": "technical", "exact_duplicate_frame": "technical",
    "cut_suspect_in_window": "technical", "neighbour_in_other_partition": "technical",
    "contact_uncertain": "review", "precontact_unavailable": "review", "outcome_unknown": "review",
    "ai_source_label_disagreement": "review", "ball_not_visible": "review", "keeper_not_visible": "review",
    "shooter_unlocated": "review", "wide_contact_interval": "review", "gap_above_frame_step": "review",
    "neighbour_attempt_in_half": "review",
    # Descriptive only: with the rule "t <= 0 and strictly before the contact start" exactly one of these holds
    # for every bounded shot. The first says the historical t0 input sat at or after the contact start.
    "t0_at_or_after_contact_start": "info", "precontact_equals_t0": "info",
}


def window_changes(clip: Path, anchor_ms: float = 4000, window_ms: tuple[float, float] = WINDOW_MS) -> pd.DataFrame:
    """Mean absolute change between consecutive decoded frames inside the input window, on 64x36 grey thumbnails.
    Same measurement as canonical_frames.image_quality; a large value marks a candidate cut, not a verified one."""
    timeline = probe_video(clip)
    capture = cv2.VideoCapture(str(clip), cv2.CAP_FFMPEG)
    rows, previous, index = [], None, 0
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot decode {clip}")
        while True:
            ok, bgr = capture.read()
            if not ok or index >= len(timeline.pts):
                break
            relative_ms = timeline.relative_ms(index, anchor_ms)
            if relative_ms > window_ms[1]:
                break
            thumb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (64, 36), interpolation=cv2.INTER_AREA).astype(float)
            if previous is not None and relative_ms >= window_ms[0]:
                rows.append({"frame_index": index, "anchor_relative_ms": relative_ms,
                             "change": float(np.abs(thumb - previous).mean() / 255)})
            previous, index = thumb, index + 1
    finally:
        capture.release()
    if not rows:
        raise ValueError(f"No consecutive frames inside {window_ms} for {clip}")
    return pd.DataFrame(rows)


def neighbouring_attempts(shots: pd.DataFrame, index: pd.DataFrame, window_ms: int) -> pd.DataFrame:
    """Other index shots of the same half within `window_ms`: candidates for the same action, a rebound or a replay."""
    pool = assign_roles(index)
    partition = pool.set_index("shot_id").role.to_dict() | {s: "test" for s in index.loc[index.split.eq("test"), "shot_id"]}
    position = index.set_index("shot_id").position_ms
    rows = []
    for shot in shots.itertuples():
        same_half = index[index.game.eq(shot.game) & index.half.eq(shot.half) & ~index.shot_id.eq(shot.shot_id)]
        candidates = same_half.assign(delta_ms=same_half.position_ms - position[shot.shot_id])
        for other in candidates[candidates.delta_ms.abs().le(window_ms)].itertuples():
            rows.append({"review_id": shot.review_id, "shot_id": shot.shot_id, "role": shot.role,
                         "neighbour_shot_id": other.shot_id, "neighbour_delta_ms": int(other.delta_ms),
                         "neighbour_goal": bool(other.goal), "neighbour_label": other.label,
                         "neighbour_partition": partition[other.shot_id],
                         "same_partition": partition[other.shot_id] == shot.role})
    return pd.DataFrame(rows, columns=["review_id", "shot_id", "role", "neighbour_shot_id", "neighbour_delta_ms",
                                       "neighbour_goal", "neighbour_label", "neighbour_partition", "same_partition"])


def shot_flags(shots: pd.DataFrame, frames: pd.DataFrame, duplicates: pd.DataFrame, changes: pd.DataFrame,
               neighbours: pd.DataFrame, cut_threshold: float, wide_interval_ms: float) -> pd.DataFrame:
    by_variant = {v: g.set_index("review_id") for v, g in frames.groupby("variant")}
    t0, pre = by_variant.get("t0", pd.DataFrame()), by_variant.get("precontact", pd.DataFrame())
    duplicated = set("|".join(duplicates.review_ids).split("|")) - {""} if len(duplicates) else set()
    peak = changes.sort_values("change").groupby("review_id").tail(1).set_index("review_id")
    near = neighbours.groupby("review_id") if len(neighbours) else None
    rows = []
    for shot in shots.itertuples():
        rid = shot.review_id
        t0_row = t0.loc[rid] if rid in t0.index and t0.loc[rid].availability == "available" else None
        pre_row = pre.loc[rid] if rid in pre.index and pre.loc[rid].availability == "available" else None
        bounded = shot.contact_status == "bounded" and not shot.prior_annotation_exposure
        mine = near.get_group(rid) if near is not None and rid in near.groups else pd.DataFrame(columns=neighbours.columns)
        flags = {
            "decode_failed": shot.extraction_status != "ok",
            "near_black_frame": any(bool(r.near_black_review) for r in (t0_row, pre_row) if r is not None),
            "exact_duplicate_frame": rid in duplicated,
            "cut_suspect_in_window": rid in peak.index and float(peak.loc[rid, "change"]) >= cut_threshold,
            "neighbour_in_other_partition": bool(len(mine)) and not mine.same_partition.all(),
            "contact_uncertain": shot.contact_status != "bounded",
            "precontact_unavailable": pre_row is None and shot.extraction_status == "ok",
            "outcome_unknown": shot.ai_outcome == "unknown" or shot.ai_goal_attribution == "unknown",
            "ai_source_label_disagreement": shot.label_review_status == "ai_source_disagreement",
            "ball_not_visible": shot.ball_visible == "no",
            "keeper_not_visible": shot.keeper_visible == "no",
            "shooter_unlocated": pd.isna(shot.shooter_x) or pd.isna(shot.shooter_y),
            "wide_contact_interval": bounded and float(shot.contact_width_ms) > wide_interval_ms,
            "gap_above_frame_step": pre_row is not None and float(pre_row.gap_ms) > FRAME_STEP_MS,
            "t0_at_or_after_contact_start": bounded and float(shot.contact_lower_ms) <= 0,
            "precontact_equals_t0": pre_row is not None and t0_row is not None and pre_row.pixel_sha256 == t0_row.pixel_sha256,
            "neighbour_attempt_in_half": bool(len(mine)),
        }
        rows.append({"review_id": rid, "shot_id": shot.shot_id, "role": shot.role, "sampling_group": shot.sampling_group,
                     "source_goal": bool(shot.source_goal), **{k: bool(v) for k, v in flags.items()},
                     "technical_flags": "|".join(k for k, v in flags.items() if v and CATEGORIES[k] == "technical"),
                     "review_flags": "|".join(k for k, v in flags.items() if v and CATEGORIES[k] == "review"),
                     "info_flags": "|".join(k for k, v in flags.items() if v and CATEGORIES[k] == "info"),
                     "window_max_change": float(peak.loc[rid, "change"]) if rid in peak.index else np.nan,
                     "window_max_change_ms": float(peak.loc[rid, "anchor_relative_ms"]) if rid in peak.index else np.nan,
                     "neighbour_count": int(len(mine)),
                     "nearest_neighbour_ms": int(mine.neighbour_delta_ms.abs().min()) if len(mine) else None})
    return pd.DataFrame(rows)


def markdown(summary: dict, flags: pd.DataFrame) -> str:
    lines = [f"# Data quality: {summary['data_version']}", "",
             f"Shots {summary['shots']}; technical flags on {summary['shots_with_technical_flags']} shots; "
             f"review flags on {summary['shots_with_review_flags']} shots. Flags describe, they do not exclude.", "",
             "| Flag | Category | Shots |", "| --- | --- | ---: |"]
    lines += [f"| {k} | {CATEGORIES[k]} | {summary['flag_counts'][k]} |" for k in CATEGORIES]
    lines += ["", "| Review | Role | Group | Technical | Review flags | Info | Max change before t0 | Max change after t0 |",
              "| --- | --- | --- | --- | --- | --- | ---: | ---: |"]
    lines += [f"| {r.review_id} | {r.role} | {r.sampling_group} | {r.technical_flags or '-'} | {r.review_flags or '-'} | "
              f"{r.info_flags or '-'} | {r.window_max_change:.3f} | {r.post_shot_max_change:.3f} |" for r in flags.itertuples()]
    return "\n".join(lines) + "\n"


def run(data: Path, index: Path, out: Path, neighbour_window_ms: int, cut_threshold: float, wide_interval_ms: float) -> None:
    manifest = verify_result(data)
    shots = pd.read_csv(data / "shots.csv")
    frames = pd.read_csv(data / "frames.csv")
    duplicates = pd.read_csv(data / "duplicates.csv")
    if "shooter_x" not in shots.columns:
        source = json.loads((data / "annotations.source.json").read_text())
        shooter = {a["review_id"]: (a.get("shooter_x"), a.get("shooter_y")) for a in source["annotations"]}
        shots["shooter_x"] = shots.review_id.map(lambda r: shooter[r][0])
        shots["shooter_y"] = shots.review_id.map(lambda r: shooter[r][1])
    sources = [data / "complete.json", index, Path(__file__), Path("scripts/prepare_canonical_data.py"),
               Path("xgpoc/data/canonical_frames.py"), Path("xgpoc/v13/protocol.py")]
    sources += [Path(p) for p in shots.clip_path]
    identity = {"data_version": manifest["identity"]["version"], "window_ms": list(WINDOW_MS),
                "neighbour_window_ms": neighbour_window_ms, "cut_threshold": cut_threshold,
                "wide_interval_ms": wide_interval_ms, "sources": {str(p.resolve()): digest(p) for p in sources}}
    if out.exists():
        if verify_result(out)["identity"] != identity:
            raise ValueError("Report inputs or thresholds changed; use a new result directory")
        print("Verified existing quality report; nothing recomputed")
        return
    anchor = manifest["identity"]["anchor_ms"]
    decoded = [s for s in shots.itertuples() if s.extraction_status == "ok"]
    changes = pd.concat([window_changes(Path(s.clip_path), anchor).assign(review_id=s.review_id) for s in decoded],
                        ignore_index=True)
    # Calibration only: the clip after the shot usually holds a real cut (replay, close-up), so its peak change
    # shows what a cut looks like under this measurement, next to the input-window peaks.
    after = pd.concat([window_changes(Path(s.clip_path), anchor, (0, POST_WINDOW_MS)).assign(review_id=s.review_id)
                       for s in decoded], ignore_index=True)
    neighbours = neighbouring_attempts(shots, pd.read_parquet(index), neighbour_window_ms)
    flags = shot_flags(shots, frames, duplicates, changes, neighbours, cut_threshold, wide_interval_ms)
    flags["post_shot_max_change"] = flags.review_id.map(after.groupby("review_id").change.max())
    summary = {"data_version": identity["data_version"], "status": "descriptive_only", "shots": len(flags),
               "shots_with_technical_flags": int(flags.technical_flags.ne("").sum()),
               "shots_with_review_flags": int(flags.review_flags.ne("").sum()),
               "flag_counts": {k: int(flags[k].sum()) for k in CATEGORIES},
               "window_max_change": {"median": float(flags.window_max_change.median()), "max": float(flags.window_max_change.max()),
                                     "threshold": cut_threshold, "validated_cut_detector": False},
               "post_shot_max_change": {"median": float(flags.post_shot_max_change.median()),
                                        "min": float(flags.post_shot_max_change.min()), "max": float(flags.post_shot_max_change.max()),
                                        "clips_above_threshold": int(flags.post_shot_max_change.ge(cut_threshold).sum()),
                                        "window_ms": [0, POST_WINDOW_MS]},
               "neighbour_pairs": int(len(neighbours)), "neighbour_pairs_across_partitions": int((~neighbours.same_partition).sum()),
               "note": "Flags are descriptive. Technical flags mark inputs that may not show the live shot; review flags mark hard "
                       "or unresolved cases that stay in the data. Cut detection and neighbour windows are unvalidated heuristics."}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{out.name}.", dir=out.parent) as temporary:
        stage = Path(temporary)
        flags.to_csv(stage / "quality_flags.csv", index=False)
        changes.to_csv(stage / "window_changes.csv", index=False)
        after.to_csv(stage / "post_shot_changes.csv", index=False)
        neighbours.to_csv(stage / "neighbours.csv", index=False)
        (stage / "summary.json").write_text(json_text(summary))
        (stage / "report.md").write_text(markdown(summary, flags))
        complete(stage, out, identity)
    print(json_text(summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("cache/v15/canonical_data_v1"))
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--out", type=Path, default=Path("cache/v15/canonical_quality_v1"))
    parser.add_argument("--neighbour-window-ms", type=int, default=15000)
    parser.add_argument("--cut-threshold", type=float, default=.08)
    parser.add_argument("--wide-interval-ms", type=float, default=2 * FRAME_STEP_MS)
    args = parser.parse_args()
    run(args.data, args.index, args.out, args.neighbour_window_ms, args.cut_threshold, args.wide_interval_ms)


if __name__ == "__main__":
    main()
