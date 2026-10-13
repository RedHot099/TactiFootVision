"""Two descriptive readings of the xg-compare family, added after the run for the report (xg-compare ticket 03; CPU, no
frame read). Neither is in the frozen protocol; the report labels both "opisowe, dodane po przebiegu".

    ./run.sh --no-sync python scripts/xgc_recovery.py      # -> results/xgc_compare/recovery

For every gap item of the family (`protocol.FAMILY`, kind "gap"), on the primary population and the main scale:

- recovery share, 1 - (F - V) / (D - V), with F - V and D - V the adjusted effects of the item (control z_D) that
  ticket 02 reports. D knows only distance and angle, so D - V is the vendor's category-specific information beyond
  geometry, and the share is the part of it the frame recovers: 1 when the frame matches the vendor beyond geometry,
  0 when it knows no more than geometry, below 0 when it is further from the vendor than geometry is, above 1 when it
  overshoots past the vendor. Unstable where D - V is near 0.
- mean xG inside the category: the mean main-scale probability of F and of V over the category's shots and their
  difference in xG points, i.e. whether the frame gives the category more or less xG than StatsBomb, unadjusted.

Intervals are 95% percentile intervals over the protocol's 2,000 match draws (seed 0), with the main scale refitted
in every draw, i.e. the same draws and population as ticket 02 (`xgpoc.compare.population.build_population`). The
inputs are checked as `scripts/xgc_analysis.py` checks them, and the observed F - V and D - V effects must equal the
family table of the run (results/xgc_compare/analysis/family.csv). complete.json, written last, holds the input,
code and output hashes.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from xgc_analysis import PROTOCOL_SHA256, RUN_OUT, TABLE, verified
from xgpoc.compare import pipeline as pl
from xgpoc.compare import protocol as pr
from xgpoc.compare import records as xr
from xgpoc.compare import stats as st
from xgpoc.compare import table as tb
from xgpoc.compare.population import Population, build_population, flat, gap, membership
from xgpoc.detector import records
from xgpoc.vendor import ladder as ld

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/xgc_compare/recovery"
FAMILY_FILE = RUN_OUT / "family.csv"
TOLERANCE = 1e-9      # observed effects against the run's family table


def weighted_mean_inside(values: np.ndarray, weights: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """(W,) mean of `values` (W, n) over the shots in `inside`, per row of shot weights (W, n)."""
    w = weights[:, inside]
    return (w * values[:, inside]).sum(1) / w.sum(1)


def recovery(pop: Population, items: tuple[pr.Item, ...]) -> pd.DataFrame:
    """One row per gap item: the adjusted F - V and D - V effects, the recovery share and the mean F and V xG inside
    the category, each with its point and interval (module docstring)."""
    rows = []
    p_f, p_v = pop.p("F"), pop.p("V")
    for item in items:
        inside, defined = membership(pop, item.category, item.level)
        effects = {}
        for a in ("F", "D"):
            adjusted, _ = st.level_effects(gap(pop, a, "V"), pop.z("D"), inside[:, None], defined[:, None], pop.weights)
            effects[a] = adjusted[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            share = 1.0 - effects["F"] / effects["D"]
        mean_f = st.XG_POINTS * weighted_mean_inside(p_f, pop.weights, inside)
        mean_v = st.XG_POINTS * weighted_mean_inside(p_v, pop.weights, inside)
        stats = {"FV": effects["F"], "DV": effects["D"], "share": share, "mean_xg_F": mean_f, "mean_xg_V": mean_v,
                 "mean_xg_F_minus_V": mean_f - mean_v}
        row = {"item": item.id, "category": item.category, "level": item.level, "n_in": int(inside.sum())}
        for prefix, stacked in stats.items():
            row.update(flat(prefix, st.summarise(stacked)))
        rows.append(row)
    return pd.DataFrame(rows)


def check_against_family(frame: pd.DataFrame, family: pd.DataFrame) -> None:
    """Raise unless the observed F - V and D - V effects equal the run's family table (a missing value fails too)."""
    reference = family.set_index("item").loc[frame.item]
    for ours, theirs in (("FV_point", "effect_point"), ("DV_point", "ref_DV_point")):
        worst = float(np.abs(frame[ours].to_numpy(float) - reference[theirs].to_numpy(float)).max())   # NaN if any is
        if not np.isfinite(worst) or worst > TOLERANCE:
            raise SystemExit(f"{ours} departs from the run's family table by {worst:.2e}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; every run goes to a new folder")
    started = datetime.now(UTC)
    try:
        protocol = pr.load(ROOT / pr.PROTOCOL, PROTOCOL_SHA256)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    if protocol["design"] != json.loads(json.dumps(pr.design())):
        raise SystemExit("the protocol's design differs from xgpoc.compare.protocol.design()")
    inputs = {"protocol": {"path": str(pr.PROTOCOL), "sha256": PROTOCOL_SHA256}}
    for name, path in (("table", TABLE / tb.TABLE_FILE), ("labels", TABLE / tb.LABEL_FILE), ("family", FAMILY_FILE)):
        inputs[name] = {"path": str(path.relative_to(ROOT)), "sha256": verified(path)}
    pinned = protocol["shot_table"]
    if (inputs["table"]["sha256"], inputs["labels"]["sha256"]) != (pinned["table_sha256"], pinned["labels_sha256"]):
        raise SystemExit("the shot table or its label file is not the one the protocol pins")
    table, meta = tb.read(TABLE)
    if meta["cuts"] != protocol["dictionary"]["cuts"]:
        raise SystemExit("the table's label-free cuts are not the protocol's")
    labels = tb.read_labels(TABLE, meta)
    draws = ld.bootstrap(table.match.to_numpy(), pr.N_BOOT, pr.BOOT_SEED)
    primary = build_population(table, labels, table.in_primary.to_numpy(), draws, "primary", "primary")

    frame = recovery(primary, tuple(item for item in pr.FAMILY if item.kind == "gap"))
    check_against_family(frame, pd.read_csv(FAMILY_FILE, dtype={"item": str}))
    args.out.mkdir(parents=True)
    frame.to_csv(args.out / "recovery.csv", index=False)
    complete = {"notice": pl.NOTICE, "status": "descriptive, added after the run (not in the frozen protocol)",
                "protocol_sha256": PROTOCOL_SHA256, "inputs": inputs, "n_boot": pr.N_BOOT, "boot_seed": pr.BOOT_SEED,
                "code": xr.code_hashes(Path(__file__)), "git": records.git_state(),
                "run": {"started": started.isoformat(timespec="seconds"), "finished": datetime.now(UTC).isoformat(timespec="seconds")},
                "files": records.output_hashes(args.out)}
    (args.out / "complete.json").write_text(json.dumps(complete, indent=2, sort_keys=True) + "\n")
    print(frame[["item", "FV_point", "DV_point", "share_point", "share_lo", "share_hi", "mean_xg_F_minus_V_point"]].to_string(index=False))


if __name__ == "__main__":
    main()
