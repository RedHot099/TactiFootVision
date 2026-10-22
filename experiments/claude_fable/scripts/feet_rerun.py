"""`xgs_detector_rerun.py` (development read-out on the 328 training matches) with the E3 table under a shot-location
rule of `xgpoc.detector.location` (frozen rule: research/feet_location_rule.md). Development only, CPU.

    ./run.sh --no-sync python scripts/feet_rerun.py --rule feet_t0 --e3 cache/xgd_full4_feet_t0/E3 --out results/<new>

Rules: `feet_t0` (the frozen rule, E3's t0 ball-source columns kept), and the two development ablations of 2026-10-22
that chose it, `feet_t0_own_flags` (the same cache, the rule's own t0 ball-source columns; results/xgs_stack_feet_t0)
and `feet_e3_order` (E3's ball order -400, -800, 0 with the feet location; cache/xgd_full5_feet, results/xgs_stack_feet).
The reference population still replays the frozen E3 (cache/xgd_full3/E3) with the frozen code and must reproduce
results/xgs_stack/real/summary.json (xgs_detector_rerun). The rule's population is `stack_location.inputs.
build_population`; the only design difference allowed is `shot_masked@0` dropped for zero variance (the feet location
is never masked).

Until 2026-10-22 this script monkey-patched `features.situation` (scripts/feet_patch.py, deleted); the library tables
equal the patched ones exactly (scripts/location_check.py, research/feet_location_rule.json evidence).
"""

import argparse
import json
import sys

import xgs_detector_rerun as rerun
from xgpoc.detector import location as loc
from xgpoc.stack import inputs as si
from xgpoc.stack_location import inputs as sl

RULES = {
    "feet_t0": loc.FEET_T0,
    "feet_t0_own_flags": loc.LocationRule("feet_t0_own_flags", loc.FEET_T0.source, loc.feet_situation),
    "feet_e3_order": loc.LocationRule("feet_e3_order", loc.E3.source, loc.feet_situation),
}

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--rule", choices=sorted(RULES), required=True)
known, rest = parser.parse_known_args()
RULE = RULES[known.rule]
sys.argv = [sys.argv[0], *rest]
_frozen_build = si.build_population


def build_population(*args, e3=si.E3, **kw):
    """The reference population replays the frozen E3 with the frozen code; any other cache under `RULE`."""
    if e3 == si.E3:
        return _frozen_build(*args, e3=e3, **kw)
    return sl.build_population(RULE, *args, e3=e3, **kw)


def read_out(population, y, frozen):
    """As `rerun.read_out`, but `shot_masked@0` may be dropped for zero variance; anything else still stops the run."""
    spec = rerun.sf.spec(frozen["design"]["chosen"], population.table)

    def strip(d):
        return json.loads(json.dumps(d).replace('"shot_masked@0", ', "").replace(', "shot_masked@0"', ""))
    if strip(json.loads(json.dumps(rerun.pr.design(spec)))) != strip(frozen["design"]):
        raise SystemExit("the population's design differs from the protocol's beyond shot_masked@0")
    return rerun.sa.analyse(population.table, population.folds, y, rerun.ss.bootstrap_weights(population.games), spec)


_frozen_read_out = rerun.read_out
_calls = []


def read_out_switch(population, y, frozen):
    """The first read-out is the frozen reference (exact design check), the second the rule's."""
    _calls.append(1)
    return _frozen_read_out(population, y, frozen) if len(_calls) == 1 else read_out(population, y, frozen)


rerun.si.build_population = build_population
rerun.read_out = read_out_switch

if __name__ == "__main__":
    rerun.main()
