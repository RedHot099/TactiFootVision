"""Write the frozen xg-compare protocol, research/xgc_compare_protocol.json (xg-compare ticket 01). Label-free.

    ./run.sh python scripts/xgc_protocol.py [--table results/xgc_compare/table] [--power results/xgc_compare/power]

Assembles the C4 design (`xgpoc.compare.protocol.design`), the category dictionary with its label-free cuts
(`xgpoc.compare.categories`), the shot table's identity (hashes of the table and of the label file, counts, the scan
split), the label-free coverage of every family item, the readings this ticket made where C4 leaves a detail open, and
the expected power (`results/xgc_compare/power/power.json`). Both input folders are checked against their completion
records. Refuses to overwrite an existing protocol. Prints the file's SHA-256, which ticket 01 records.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from xgpoc.bundle import canonical_digest, sha256_file
from xgpoc.compare import categories as cat
from xgpoc.compare import protocol as pr
from xgpoc.compare import table as tb
from xgpoc.detector import records

ROOT = Path(__file__).resolve().parents[1]

READINGS = [
    "Category bins (distance, angle, goalkeeper depth and lateral offset, nearest opponent, keypoints, minute) were fixed in "
    "ticket 01 from football conventions and the label-free distributions, since the spec names these categories but not their "
    "bins.",
    "Item 6a's '4 m' is the comparator's gk_depth >= 4 in StatsBomb units (yards), the units of the label-free median 2.1 and "
    "p75 3.4 quoted in C4. 'Goalkeeper distance from goal' is the gk_depth category; gk_dist_goal is not binned separately.",
    "Key-pass type is four binary categories (cross, through ball, cut-back, none) because StatsBomb flags overlap (a cross can "
    "also be a cut-back); key-pass height is defined only for shots with a key pass.",
    "A shot whose category value is missing (no goalkeeper in the freeze frame, no detector descriptor) is left out of both "
    "sides of that category's comparison.",
    "The previous shot of a rebound is any StatsBomb Shot event (penalties included) of the same match, period and possession, "
    "by event index; the window is 0 to 5 s (0 to 3 s) inclusive.",
    "Item 16's adjusted effect uses z_D as its control, like item 10 (C4 names the control for item 10 only).",
    "Spearman correlations (items 8, 9 and the agreement matrix) are of the raw probabilities, as the spec's Analysis 1 "
    "says; they are label-free. Pearson correlations are of the main-scale logits; Pearson of the raw logits is the "
    "sensitivity check.",
    "Item 9 is tested only when headers and foot shots each meet the category floor (50 shots, headers in 20 matches).",
    "The rebound subtypes (after a save, after a block) are descriptive, as C4 says, so they are not scanned.",
    "Item 1: StatsBomb 'Saved Off Target' is reported apart, with Blocked, because C4 defines on target as a save and off "
    "target as off target, post or wayward, and this outcome is a save of a shot going wide.",
    "The open scan covers the levels of the categories with role 'category'. The post-shot categories, the 3 s rebound "
    "(a sensitivity check of item 2) and the half (the negative control) are not scanned. A two-level category scans one "
    "level, because the other is its mirror. 'Floors met in both halves' means the category floor on the whole primary "
    "population plus the scan floor in each half.",
    "The combined population is the paper's: it also leaves out the one StatsBomb-only row of the pre-registered frame "
    "exclusion (decode-corrupted frame), so it has 3,401 shots, not 3,402.",
    "Bootstrap p-values are shares of draws (no +1 correction), as the paper's superiority p.",
    "Power simulation (C4: 'permute category labels across shots, preserving sizes; inject a 0.15-logit effect'): it holds "
    "the paper's frozen recalibration fixed instead of refitting it in every draw and half, which makes the intervals "
    "narrower. The shot-level permutation also breaks the match clustering of match-level categories (league-season, "
    "camera view) and their correlation with z_D. Both make the power optimistic, most of all for match-level levels. "
    "Labels for the log-loss items are drawn from the recalibrated vendor. Log-loss and correlation items plant their "
    "own thresholds (0.005 LL, 0.10), since a 0.15-logit gap has no log-loss or correlation unit; scenario '<id>:logit' "
    "also shifts F by 0.15 logit for the log-loss items. Item 8 has no category to permute and is simulated as 0 or 0.10 "
    "plus the real statistic's bootstrap deviations, so the simulation computes that statistic's spread, never its value.",
]


def verified(path: Path) -> str:
    try:
        return records.verified_file(path)
    except ValueError as e:
        raise SystemExit(str(e)) from None


def family_coverage(coverage: pd.DataFrame, table: pd.DataFrame) -> dict:
    """Per family item and population: shots inside and outside and matches inside (label-free)."""
    def level(population: str, category: str, value: str) -> dict:
        row = coverage[coverage.population.eq(population) & coverage.category.eq(category) & coverage.level.eq(value)].iloc[0]
        return {"n_in": int(row.n_in), "n_out": int(row.n_out), "matches_in": int(row.matches_in), "n_undefined": int(row.n_undefined),
                **{f"n_in_{h}": int(row[f"n_in_{h}"]) for h in pr.HALVES}}

    out = {}
    primary = table[table.in_primary]
    for item in pr.FAMILY:
        if item.kind == "correlation_difference":
            out[item.id] = {p: {"n": int(len(t)), "matches": int(t.match.nunique())} for p, t in (("primary", primary), ("combined", table))}
        elif item.kind == "correlation_contrast":
            out[item.id] = {p: {"n_headers": int(t.cat_body.eq("Head").sum()), "n_foot": int(t.cat_body.isin(cat.FOOT).sum())}
                            for p, t in (("primary", primary), ("combined", table))}
        else:
            populations = cat.BY_NAME[item.category].populations
            out[item.id] = {p: level(p, item.category, item.level) for p in populations}
    return out


def split_record(table: pd.DataFrame) -> dict:
    pairs = table[["match", "league_season", "scan_half"]].drop_duplicates().sort_values("match")
    by_stratum = pairs.groupby(["league_season", "scan_half"]).size().unstack(fill_value=0)
    return {"matches": {h: int(pairs.scan_half.eq(h).sum()) for h in pr.HALVES},
            "primary_shots": {h: int((table.in_primary & table.scan_half.eq(h)).sum()) for h in pr.HALVES},
            "by_league_season": {s: {h: int(by_stratum.loc[s].get(h, 0)) for h in pr.HALVES} for s in by_stratum.index},
            "assignment_digest": canonical_digest(pairs[["match", "scan_half"]].values.tolist())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--table", type=Path, default=ROOT / "results/xgc_compare/table")
    parser.add_argument("--power", type=Path, default=ROOT / "results/xgc_compare/power")
    parser.add_argument("--out", type=Path, default=ROOT / pr.PROTOCOL)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; the protocol is written once")

    table_sha, power_sha = verified(args.table / tb.TABLE_FILE), verified(args.power / "power.json")
    table, meta = tb.read(args.table)
    coverage = pd.read_csv(args.table / tb.COVERAGE_FILE)
    power = json.loads((args.power / "power.json").read_text())
    protocol = {
        "purpose": "xg-compare: the frozen design (council C4), the category dictionary, the shot table it runs on and the "
                   "expected power, written before any category-level outcome or gap was computed (xg-compare ticket 01)",
        "provenance": "Transcribed from .scratch/xg-compare/spec.md, Decisions -> C4, and council record C4 "
                      "(.scratch/xg-detector/council.md). Ticket 02 refuses to run unless this file's SHA-256 is the one "
                      "recorded in ticket 01. This file is not edited afterwards.",
        "design": pr.design(),
        "readings": READINGS,
        "dictionary": cat.dictionary_record(cat.Cuts(**meta["cuts"])),
        "shot_table": {"path": "results/xgc_compare/table", "table_sha256": table_sha, "labels_sha256": meta["labels_sha256"],
                       "schema_version": meta["schema_version"], "primary": meta["primary"], "combined": meta["combined"],
                       "by_population": meta["by_population"], "a1_coverage": meta["a1_coverage"],
                       "sources": {k: v.get("sha256") for k, v in meta["sources"].items()},
                       "statsbomb_pin": meta["sources"]["statsbomb_events"]["pin"], "split": split_record(table)},
        "family_coverage": family_coverage(coverage, table),
        "power": {"path": "results/xgc_compare/power/power.json", "sha256": power_sha, **power},
    }
    args.out.write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
    print(sha256_file(args.out))


if __name__ == "__main__":
    main()
