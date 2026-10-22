"""The 2026-10-18 method sweep (`sweep_test.py`, rule research/sweep_2026_10_rule.md) repeated with the feet_t0 E3 of
the retrained detector in place of xgd_full4 (2026-10-22, owner-ordered error decomposition; descriptive: the 172
matches are spent). The `*_new` methods use the frozen shot-location rule `feet_t0` (research/feet_location_rule.md,
`xgpoc.detector.location.FEET_T0`: shot location = shooter's feet, the t0 ball picks the shooter, E3's t0 ball-source
columns kept) on cache/xgd_full4_feet_t0/E3.

    ./run.sh --no-sync python scripts/feet_sweep.py --out results/<new>

Writes <out>/ (as sweep_test) and prints the feet-minus-dd2c paired deltas per arm. results/sweep_feet_2026_10 was
written on 2026-10-22 by the monkey-patched version of this script (scripts/feet_patch.py, deleted); the library
tables equal the patched ones exactly (scripts/location_check.py), so a re-run reproduces it.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import sweep_test as st
from xgpoc.detector import location as loc
from xgpoc.stack import inputs as si
from xgpoc.stack_heldout import rule as tr
from xgpoc.stack_heldout import testset as ts
from xgpoc.stack_location import inputs as sl
from xgs_test import read_test_labels

ROOT = Path(__file__).resolve().parents[1]
FEET = ROOT / "cache/xgd_full4_feet_t0/E3"
RULE = loc.FEET_T0
_population, _heldout = si.build_population, ts.heldout_table


def population(*args, e3=si.E3, **kw):
    """sweep_test's population: the feet cache under `RULE`, any other cache with the frozen code."""
    return sl.build_population(RULE, *args, e3=e3, **kw) if Path(e3) == FEET else _population(*args, e3=e3, **kw)


def heldout(shots, z, e3, annotations):
    """sweep_test's held-out table: the feet cache under `RULE`, any other cache with the frozen code."""
    if Path(e3) == FEET:
        return sl.heldout_table(shots, z, e3, annotations, RULE)
    return _heldout(shots, z, e3, annotations)


st.si.build_population, st.ts.heldout_table, st.NEW_E3 = population, heldout, FEET


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    out = parser.parse_args().out
    st.run_sweep(out)
    feet = pd.read_csv(out / "predictions.csv").set_index("shot_id")
    base = (
        pd.read_csv(ROOT / "results/sweep_2026_10/predictions.csv")
        .set_index("shot_id")
        .loc[feet.index]
    )
    y = read_test_labels(pd.Series(feet.index))
    weights = tr.bootstrap_weights(feet.game.to_numpy(), tr.load_rule())
    seen = (
        pd.read_csv(ROOT / "results/sweep_2026_10/seen.csv")
        .set_index("shot_id")
        .loc[feet.index]
    )
    rows = []
    for subset, mask in {
        "all": np.ones(len(feet), bool),
        "unseen": ~seen.seen.to_numpy(),
    }.items():
        for m in (
            "stack_G_new",
            "stack_Q_new",
            "stack_ALL_new",
            "pos_DA_new",
            "pos_ALL_new",
        ):
            rows.append(
                {"subset": subset, "method": m, "base": "dd2c same method"}
                | st.paired_delta(
                    y,
                    feet[f"p_{m}"].to_numpy(),
                    base[f"p_{m}"].to_numpy(),
                    weights,
                    mask,
                )
            )
    table = pd.DataFrame(rows)
    table.to_csv(out / "feet_minus_dd2c.csv", index=False, float_format="%.6g")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
