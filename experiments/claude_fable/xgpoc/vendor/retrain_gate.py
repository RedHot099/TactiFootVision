"""The gate check of the retrained G-combo against the archived one (xg-vendor ticket 10).

`gate_report` compares two probability vectors on the same gate shots: the point estimates decide, through the tolerance
of the frozen plan (`xgpoc.vendor.split_check.gate_passes`); the paired match bootstrap of both differences, on the shared
draws of `xgpoc.vendor.ladder.bootstrap` (count-weighted, 5,000 draws, the sequence of every other analysis), is reported
and does not decide. Every difference is retrained minus archived.
"""

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

from xgpoc.vendor import ladder
from xgpoc.vendor.draws import N_BOOT, SEED
from xgpoc.vendor.split_check import gate_passes

METRICS = ("log_loss", "auc")


def gate_report(y: np.ndarray, p_retrained: np.ndarray, p_archived: np.ndarray, matches: np.ndarray, n_boot: int = N_BOOT,
                seed: int = SEED) -> dict:
    """Metrics of both models on the gate shots, the differences (retrained minus archived), the tolerance decision and
    the 95% percentile interval of each difference over paired match-bootstrap draws."""
    y = np.asarray(y, int)
    point = {name: {"retrained": float(f(y, np.asarray(p_retrained))), "archived": float(f(y, np.asarray(p_archived)))}
             for name, f in (("log_loss", log_loss), ("auc", roc_auc_score))}
    for m in point.values():
        m["delta"] = m["retrained"] - m["archived"]
    boot = ladder.bootstrap(np.asarray(matches), n_boot, seed)
    new = ladder.metric_draws(y, p_retrained, boot, METRICS, calibration=False)
    old = ladder.metric_draws(y, p_archived, boot, METRICS, calibration=False)
    for name, m in point.items():
        diff = new[name] - old[name]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        m["bootstrap"] = {"n_boot": n_boot, "seed": seed, "lo95": float(lo), "hi95": float(hi), "sd": float(np.std(diff, ddof=1))}
    decision = gate_passes(point["log_loss"]["delta"], point["auc"]["delta"])
    return {"n_shots": len(y), "n_goals": int(y.sum()), "n_matches": len(np.unique(matches)), "metrics": point, "decision": decision,
            "passed": decision["passed"]}
