"""Paired development comparison: each feet variant minus the retrained-detector read-out (xgs_stack_dd2c), per arm,
shot log-loss difference with a 2,000-draw match bootstrap (seed 0). Development only (328 training matches)."""

import json

import pandas as pd
from xgpoc.stack import inputs as si
from xgs_detector_rerun import paired

base = pd.read_csv("results/xgs_stack_dd2c/probabilities.csv", index_col=0)
pop = si.build_population()
y = si.load_outcomes(pop).aligned(pop)
out = {}
for k in ("xgs_stack_feet", "xgs_stack_feet_t0", "xgs_stack_feet_t0_flags"):
    new = pd.read_csv(f"results/{k}/probabilities.csv", index_col=0).loc[base.index]
    arms = [
        c for c in base.columns if c in new.columns and c not in ("game", "fold", "y")
    ]
    out[k] = paired(
        {a: new[a].to_numpy() for a in arms},
        {a: base[a].to_numpy() for a in arms},
        y,
        pop.games,
    )
print(json.dumps(out, indent=1, default=float))
json.dump(
    out,
    open("results/err_decomp/feet_minus_dd2c_dev.json", "w"),
    indent=1,
    default=float,
)
