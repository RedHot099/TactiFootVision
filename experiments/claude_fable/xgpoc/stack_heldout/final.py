"""The final, training-only fit of the xg-stack test (ticket 04, council C7 Q2).

R, G and GS of the C6 design ((b)+(c), `fit.CANDIDATES["bc"]`) are fitted once on all 7,062 training shots:
    1. the arms come from `fit.spec("bc", training table)`, so the zero-variance list is the training rows' (and must
       equal the C6 protocol's frozen columns, checked by the caller);
    2. C is chosen by match-grouped CV over the 5 host folds on the frozen grid (`fit.C_GRID`): for each host fold the
       design is preprocessed on the other four folds' rows only, every C is fitted there and scored on the fold, and
       the summed held-out log-loss picks C (ties: the smaller C, as `fit.chosen_c`). An edge value of the grid is
       accepted and recorded. R (z alone) has no penalised column and needs no CV; z is unpenalised in every arm;
    3. the preprocessing (imputation means, natural-spline knots, standardisation) is refitted on all training rows
       (`fit.Preprocessor.fit`) and the logistic model is fitted there at the chosen C (`fit.logistic_fit`).
`FinalFit.predict` applies the training transforms and coefficients to any table; `FinalFit.record` writes them.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from xgpoc.stack import fit as sf


@dataclass(frozen=True)
class FinalFit:
    """One arm fitted on every training shot: its transforms, coefficients (intercept first), chosen C and the summed
    inner held-out log-loss per grid value (None for R)."""

    arm: sf.Arm
    preprocessor: sf.Preprocessor
    beta: np.ndarray
    c: float | None
    inner_losses: dict[float, float] | None

    def predict(self, table: pd.DataFrame) -> np.ndarray:
        return sf.predict(self.beta, self.preprocessor(table))

    def record(self) -> dict:
        pre = self.preprocessor
        grid = sf.INNER_CV.grid
        return {"arm": self.arm.name, "status": self.arm.status, "inputs": self.arm.inputs, "columns": self.arm.columns,
                "chosen_c": self.c, "c_grid": list(grid),
                "c_at_grid_edge": None if self.c is None else bool(self.c in (grid[0], grid[-1])),
                "inner_summed_log_loss": None if self.inner_losses is None else {f"{c:g}": v for c, v in self.inner_losses.items()},
                "intercept": float(self.beta[0]),
                "coefficients": dict(zip(self.arm.columns, map(float, self.beta[1:]), strict=True)),
                "imputation_means": pre.means, "knots": {c: list(map(float, k)) for c, k in pre.knots.items()},
                "centre": dict(zip(self.arm.columns, map(float, pre.centre), strict=True)),
                "scale": dict(zip(self.arm.columns, map(float, pre.scale), strict=True))}


def inner_losses(table: pd.DataFrame, folds: np.ndarray, y: np.ndarray, arm: sf.Arm, model: sf.Model) -> dict[float, float]:
    """C -> summed held-out log-loss over the host folds, each fold scored by the arm fitted on the other folds with
    its own preprocessing (module docstring, step 2)."""
    designs = []
    for k in np.unique(folds):
        train, test = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        pre = sf.Preprocessor.fit(arm, table.iloc[train])
        designs.append((pre(table.iloc[train]), y[train], pre(table.iloc[test]), y[test]))
    out = {}
    for c in model.grid:
        total = 0.0
        for x_train, y_train, x_test, y_test in designs:
            beta = sf.logistic_fit(x_train, y_train, model.penalties(x_train.shape[1], c))
            total += sf.mean_log_loss(y_test, sf.predict(beta, x_test)) * len(y_test)
        out[float(c)] = total
    return out


def final_fit(table: pd.DataFrame, folds: np.ndarray, y: np.ndarray, arm: sf.Arm, model: sf.Model = sf.INNER_CV) -> FinalFit:
    """The arm fitted once on every row of `table` (module docstring)."""
    y = np.asarray(y, float)
    losses = inner_losses(table, folds, y, arm, model) if model.inner_cv and len(arm.columns) > 1 else None
    c = None if losses is None else min(losses, key=lambda value: (losses[value], value))
    pre = sf.Preprocessor.fit(arm, table)
    x = pre(table)
    beta = sf.logistic_fit(x, y, model.penalties(x.shape[1], model.grid[-1] if c is None else c))
    return FinalFit(arm, pre, beta, c, losses)
