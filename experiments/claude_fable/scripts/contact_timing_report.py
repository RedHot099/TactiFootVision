"""Measure real decoded times of the pre-shot frames and apply the admission rule.

Diagnostic only: nothing here changes labels, frames or caches. For every reviewed shot
(or a random sample) each nominal pre-shot frame is located inside its clip by image
matching, the contact interval from the review is attached, and the admission rule from
xgpoc.v13.contact decides which frame a pre-shot xG input may use. The output table
keeps every record, including the ones without a safe frame.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from xgpoc.paths import INDEX
from xgpoc.v13.contact import ContactInterval, DecodedFrame, admit, locate_frames_in_clip
from xgpoc.v13.inputs import OFFSETS
from xgpoc.v13.protocol import digest


def frame_column(offset: int) -> str:
    return "frame_path" if offset == 0 else f"frame_path_{offset}"


def measure(row, min_margin: float) -> tuple[list[DecodedFrame], list[dict]]:
    offsets = [o for o in OFFSETS if str(row.get(frame_column(o), "") or "") not in ("", "nan")]
    paths = [str(row[frame_column(o)]) for o in offsets]
    frames, timing = [], []
    if not paths:
        raise ValueError(f"no pre-shot frames for {row.shot_id}")
    for offset, path, match in zip(offsets, paths, locate_frames_in_clip(paths, str(row.clip_path))):
        measured = match.measured_offset(min_margin)
        frames.append(DecodedFrame(path, offset, measured))
        timing.append({"shot_id": row.shot_id, "nominal_ms": offset, "matched_ms": match.offset_ms,
                       "measured_ms": measured, "distance": match.distance, "margin": match.margin,
                       "fps": match.fps})
    return frames, timing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--review", type=Path, default=Path("cache/v13/contact_review/combined_review.csv"))
    parser.add_argument("--out", type=Path, default=Path("cache/v13/contact_timing"))
    parser.add_argument("--sample", type=int, default=0, help="add this many random unreviewed shots (seed 1300)")
    parser.add_argument("--min-margin", type=float, default=0.25)
    args = parser.parse_args()
    index = pd.read_parquet(args.index)
    index = index[index.clip_path.astype(str).str.len() > 0]
    review = pd.read_csv(args.review)
    if review.shot_id.duplicated().any():
        raise ValueError("review has duplicate shots")
    pool = index[index.shot_id.isin(review.shot_id)]
    if len(pool) != len(review):
        raise ValueError("review shots missing from the index")
    if args.sample:
        rest = index[~index.shot_id.isin(review.shot_id)]
        pool = pd.concat([pool, rest.sample(args.sample, random_state=1300)])
    review = review.set_index("shot_id")
    admissions, timing = [], []
    # iterrows keeps column names such as frame_path_-400 intact; itertuples renames them.
    for i, (_, row) in enumerate(pool.iterrows()):
        annotated = row.shot_id in review.index
        contact = ContactInterval.from_review(review.loc[row.shot_id]) if annotated else ContactInterval.unknown()
        frames, rows = measure(row, args.min_margin)
        timing.extend(rows)
        record = admit(frames, contact).to_record()
        record.update({"shot_id": row.shot_id, "reviewed": annotated, "contact_known": contact.known,
                       "contact_lower_ms": contact.lower_ms, "contact_upper_ms": contact.upper_ms,
                       "confidence": contact.confidence,
                       "t0_measured_ms": next((f.measured_ms for f in frames if f.nominal_ms == 0), None)})
        admissions.append(record)
        if i % 50 == 0:
            print(f"{i}/{len(pool)}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    admissions = pd.DataFrame(admissions)
    timing = pd.DataFrame(timing)
    admissions.to_csv(args.out / "admissions.csv", index=False)
    timing.to_csv(args.out / "frame_timing.csv", index=False)
    measured = timing.dropna(subset=["measured_ms"])
    deviation = (measured.measured_ms - measured.nominal_ms).astype(int)
    summary = {
        "review_sha256": digest(args.review), "index": str(args.index), "min_margin": args.min_margin,
        "shots": int(len(admissions)), "reviewed": int(admissions.reviewed.sum()),
        "contact_known": int(admissions.contact_known.sum()),
        "admission": admissions.admission.value_counts().to_dict(),
        "admission_by_known": admissions.groupby("contact_known").admission.value_counts().unstack(fill_value=0)
                                        .to_dict("index"),
        "admitted_nominal_ms": admissions.frame_nominal_ms.value_counts(dropna=False).to_dict(),
        "gap_ms_known": admissions[admissions.contact_known].gap_ms.describe().to_dict(),
        "frames_measured": int(len(measured)), "frames_total": int(len(timing)),
        "measured_minus_nominal_ms": {str(k): int(v) for k, v in deviation.value_counts().sort_index().items()},
        "abs_deviation_over_40ms": int((deviation.abs() > 40).sum()),
        "warning": "Diagnostic only. Contact intervals are a model review on a 200 ms grid, not a human reference.",
    }
    summary = json.loads(json.dumps(summary, default=lambda v: None if pd.isna(v) else float(v)))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("median match margin", float(np.median(timing.margin)))


if __name__ == "__main__":
    main()
