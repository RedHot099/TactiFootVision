"""The 2026-10-18 method sweep (`sweep_test.py`, rule research/sweep_2026_10_rule.md) repeated with the feet variant
of the retrained detector's E3 in place of xgd_full4 (2026-10-22, owner-ordered error decomposition; descriptive: the
172 matches are spent). Variant: shot location = shooter's feet, the t0 ball picks the shooter (`feet_patch`, t0_ball),
the t0 ball-source flags kept from xgd_full4 (the development ablation). The `*_new` methods are the feet variant.

    ./run.sh --no-sync python scripts/feet_sweep.py

Writes results/sweep_feet_2026_10/ (as sweep_test) and prints the feet-minus-dd2c paired deltas per arm.
"""

from pathlib import Path

import feet_patch
import numpy as np
import pandas as pd
import sweep_test as st
from xgpoc.detector import features as ft
from xgpoc.stack import inputs as si
from xgpoc.stack_heldout import rule as tr
from xgpoc.stack_heldout import testset as ts
from xgs_test import read_test_labels

ROOT = Path(__file__).resolve().parents[1]
FEET = ROOT / "cache/xgd_full5_feet_t0/E3"
DD2C = ROOT / "cache/xgd_full4/E3"
OUT = ROOT / "results/sweep_feet_2026_10"
FLAGS = ["ball_none@0", "ball_own@0", "ball_m400@0", "ball_m800@0", "shot_from_ball@0"]
FROZEN_CONFIG = si.E3_CONFIG


def unpatched() -> None:
    ft.situation, si.situation, si.E3_CONFIG = (
        feet_patch._original,
        feet_patch._original,
        FROZEN_CONFIG,
    )


def with_dd2c_flags(build, e3, *args, **kw):
    """`build` on the feet cache (patched) with xgd_full4's t0 ball-source flags; any other cache unpatched."""
    unpatched()
    if Path(e3) != FEET:
        return (
            build(*args, e3=e3, **kw)
            if build is _population
            else build(*args[:2], e3, *args[2:], **kw)
        )
    ref = (
        build(*args, e3=DD2C, **kw)
        if build is _population
        else build(*args[:2], DD2C, *args[2:], **kw)
    )
    feet_patch.apply(t0_ball=True)
    out = (
        build(*args, e3=e3, **kw)
        if build is _population
        else build(*args[:2], e3, *args[2:], **kw)
    )
    unpatched()
    if build is _population:
        out.table[FLAGS] = ref.table.loc[out.table.index, FLAGS]
        return out
    table, wrapped = out
    table[FLAGS] = ref[0].loc[table.index, FLAGS]
    return table, wrapped


_population, _heldout = si.build_population, ts.heldout_table
st.si.build_population = lambda *a, e3=si.E3, **kw: with_dd2c_flags(
    _population, e3, *a, **kw
)
st.ts.heldout_table = lambda shots, z, e3, ann: with_dd2c_flags(
    _heldout, e3, shots, z, ann
)
st.NEW_E3 = FEET


def main() -> None:
    st.run_sweep(OUT)
    feet = pd.read_csv(OUT / "predictions.csv").set_index("shot_id")
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
    table.to_csv(OUT / "feet_minus_dd2c.csv", index=False, float_format="%.6g")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
