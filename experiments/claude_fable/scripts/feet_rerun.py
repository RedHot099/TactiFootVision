"""xgs_detector_rerun.py (development read-out on the 328 training matches) with the feet_patch features.

./run.sh --no-sync python scripts/feet_rerun.py [--t0] --e3 cache/xgd_full5_feet[_t0]/E3 --out results/xgs_stack_feet[_t0]
"""

import sys

import feet_patch

t0 = "--t0" in sys.argv
if t0:
    sys.argv.remove("--t0")
keep_flags = (
    "--keep-ball-flags" in sys.argv
)  # ablation: t0 variant's locations, dd2c's ball-source flags at t0
if keep_flags:
    sys.argv.remove("--keep-ball-flags")
FLAG_COLUMNS = [
    "ball_none@0",
    "ball_own@0",
    "ball_m400@0",
    "ball_m800@0",
    "shot_from_ball@0",
]
import xgs_detector_rerun as rerun  # noqa: E402
from xgpoc.stack import inputs  # noqa: E402

FROZEN_CONFIG = inputs.E3_CONFIG
_build = inputs.build_population


def build_population(*args, e3=inputs.E3, **kw):
    """The reference population replays the frozen E3 unpatched; any other cache replays with the patch."""
    from xgpoc.detector import features as ft

    if e3 == inputs.E3:
        ft.situation, inputs.situation, inputs.E3_CONFIG = (
            feet_patch._original,
            feet_patch._original,
            FROZEN_CONFIG,
        )
        return _build(*args, e3=e3, **kw)
    if keep_flags:
        ft.situation, inputs.situation, inputs.E3_CONFIG = (
            feet_patch._original,
            feet_patch._original,
            FROZEN_CONFIG,
        )
        dd2c = _build(*args, e3=rerun.ROOT / "cache/xgd_full4/E3", **kw)
    feet_patch.apply(t0_ball=t0)
    population = _build(*args, e3=e3, **kw)
    if keep_flags:
        population.table[FLAG_COLUMNS] = dd2c.table.loc[
            population.table.index, FLAG_COLUMNS
        ]
    return population


def read_out(population, y, frozen):
    """As `rerun.read_out`, but the only allowed design difference is `shot_masked@0` dropped for zero variance (the
    feet location is never masked); anything else still stops the run."""
    import json

    spec = rerun.sf.spec(frozen["design"]["chosen"], population.table)
    new = json.loads(json.dumps(rerun.pr.design(spec)))
    strip = lambda d: json.loads(
        json.dumps(d).replace('"shot_masked@0", ', "").replace(', "shot_masked@0"', "")
    )  # noqa: E731
    a, b = strip(new), strip(frozen["design"])
    if t0:  # the t0 variant changes the E3 ball order on purpose
        a.pop("e3", None)
        b.pop("e3", None)
    if a != b:
        raise SystemExit(
            "the population's design differs from the protocol's beyond shot_masked@0 (and the t0 ball order)"
        )
    return rerun.sa.analyse(
        population.table,
        population.folds,
        y,
        rerun.ss.bootstrap_weights(population.games),
        spec,
    )


_frozen_read_out = rerun.read_out
_calls = []


def read_out_switch(population, y, frozen):
    _calls.append(1)
    return (
        _frozen_read_out(population, y, frozen)
        if len(_calls) == 1
        else read_out(population, y, frozen)
    )


rerun.si.build_population = build_population
rerun.read_out = read_out_switch
rerun.main()
