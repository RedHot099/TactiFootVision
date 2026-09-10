"""Validate visual-review coverage and summarize timing uncertainty without changing labels."""

import argparse
import json
from pathlib import Path

import pandas as pd

from xgpoc.v13.protocol import digest


def combine(sample, reviews):
    review = pd.concat(reviews, ignore_index=True)
    review["body_part"] = review.body_part.replace({"uncertain": "unknown"}) if "body_part" in review else None
    if review.review_id.duplicated().any():
        raise ValueError("Duplicate contact review IDs")
    if set(review.review_id) != set(sample.review_id):
        raise ValueError(f"Contact review incomplete: {len(review)}/{len(sample)}")
    allowed = {"t0_status": {"pre", "contact", "post", "uncertain"},
               "body_part": {"foot", "head", "unknown"},
               "confidence": {"high", "medium", "low"}}
    for column, values in allowed.items():
        if not review[column].isin(values).all():
            raise ValueError(f"Invalid {column} values")
    lo, hi = review.contact_lower_ms, review.contact_upper_ms
    if (lo.isna() != hi.isna()).any() or (lo > hi).any():
        raise ValueError("Invalid contact interval")
    if review.note.fillna("").str.strip().eq("").any():
        raise ValueError("Every review needs visual evidence or a visibility limitation")
    merged = sample.merge(review, on="review_id", validate="one_to_one").sort_values("review_id")
    merged["timing_consistent"] = (
        merged.t0_status.eq("uncertain")
        | (merged.t0_status.eq("pre") & merged.contact_lower_ms.ge(0))
        | (merged.t0_status.eq("post") & merged.contact_upper_ms.le(0))
        | (merged.t0_status.eq("contact") & merged.contact_lower_ms.le(0) & merged.contact_upper_ms.ge(0)))
    merged["interval_status"] = "uncertain"
    merged.loc[merged.contact_lower_ms.gt(0), "interval_status"] = "pre"
    merged.loc[merged.contact_upper_ms.lt(0), "interval_status"] = "post"
    touches_zero = merged.contact_lower_ms.le(0) & merged.contact_upper_ms.ge(0)
    merged.loc[touches_zero, "interval_status"] = "contains_t0"
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("cache/v13/contact_review"))
    parser.add_argument("--reviews", nargs="+", default=["review_b_rechecked.csv", "review_root.csv", "review_c.csv", "review_d_rechecked.csv"])
    args = parser.parse_args()
    sample_path = args.root / "sample.csv"
    paths = [args.root / name for name in args.reviews]
    table = combine(pd.read_csv(sample_path), [pd.read_csv(p).assign(review_source=p.name) for p in paths])
    table.to_csv(args.root / "combined_review.csv", index=False)
    summary = {"n": len(table), "games": table.game.nunique(),
               "status": table.t0_status.value_counts().to_dict(),
               "interval_status": table.interval_status.value_counts().to_dict(),
               "confidence": table.confidence.value_counts().to_dict(),
               "status_by_review_source": pd.crosstab(table.review_source, table.t0_status).to_dict("index"),
               "body_part": table.body_part.value_counts().to_dict(),
               "inconsistent_review_ids": table.loc[~table.timing_consistent, "review_id"].tolist(),
               "post_high_confidence": int((table.t0_status.eq("post") & table.confidence.eq("high")).sum()),
               "source_sha256": {p.name: digest(p) for p in [sample_path, *paths]},
               "scope": "Visual model review; coarse 200 ms sampling, not human-validated contact ground truth. No outcome labels or model frames changed."}
    (args.root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
