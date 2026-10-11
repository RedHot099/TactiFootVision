"""Per-draw refit of the cross-fitted logistic models of the ladder (xg-vendor ticket 14 fix round).

A match draw of the bootstrap is a resampled evaluation population: match m enters c_m times. Holding the cross-fitted
predictions fixed ignores that the recalibration and the stacks would have been fitted on that resample too, and
understates the spread (ticket 08 measured about 8% on H2). Here every draw refits them.

Folds     the population's match folds (`ladder.match_folds`) stay. Every copy of a match sits in its match's fold, so no
          shot is scored by a model fitted on any copy of its own match.
Draw b    fold k's model is the weighted logistic regression of the label on the features over the other folds' shots,
          weight = count of the shot's match in draw b; it scores the shots of fold k. This is cross-fitting the
          resampled data set with the drawn matches duplicated (tested against sklearn on the duplicated rows).
Fit       the objective of sklearn's LogisticRegression(C = 1e6): weighted log-likelihood plus `PENALTY` / 2 times
          the squared slopes, intercept unpenalised, solved to convergence by Newton's method, vectorised over draws
          and started from the observed sample's fit of the same fold (`ladder.crossfit` uses the same solver). The
          penalty also gives a constant feature (the constant rung) a zero slope.
"""

import numpy as np

from xgpoc.eval.calibration import sigmoid

PENALTY = 1e-6          # 1 / C
NEWTON_ITERATIONS = 100
TOLERANCE = 1e-10       # largest change of a linear predictor at convergence (a coefficient step along a collinear
                        # direction, e.g. the constant rung's slope against the intercept, changes no prediction)


def logistic_newton(x: np.ndarray, y: np.ndarray, weights: np.ndarray, start: np.ndarray, penalty: float = PENALTY) -> np.ndarray:
    """(b, p) coefficients of the penalised weighted logistic regressions of `y` (n,) on `x` (n, p; column 0 is the
    intercept), one per row of `weights` (b, n), from `start` (p,). Raises when Newton does not converge."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    p = x.shape[1]
    ridge = np.diag(np.r_[0.0, np.full(p - 1, penalty)])
    outer = (x[:, :, None] * x[:, None, :]).reshape(len(x), p * p)
    beta = np.tile(np.asarray(start, float), (len(weights), 1))
    for _ in range(NEWTON_ITERATIONS):
        q = sigmoid(beta @ x.T)
        gradient = (weights * (y - q)) @ x - beta @ ridge
        hessian = ((weights * q * (1 - q)) @ outer).reshape(-1, p, p) + ridge
        try:
            step = np.linalg.solve(hessian, gradient[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            break
        beta += step
        if not np.isfinite(beta).all():
            break
        if np.abs(step @ x.T).max() < TOLERANCE:
            return beta
    raise ValueError("the per-draw logistic refit did not converge")


def crossfit_draws(features: np.ndarray, y: np.ndarray, folds: np.ndarray, weights: np.ndarray, start: np.ndarray) -> np.ndarray:
    """(b, n) out-of-fold probabilities of the cross-fitted logistic regression of `y` on `features` (n, k), refitted in
    every draw (module docstring). `weights` (b, n) are the draws' shot weights; `start` (n_folds, 1 + k) the observed
    fit per fold in the order of `np.unique(folds)` (as `ladder.crossfit` returns it)."""
    x = np.column_stack([np.ones(len(y)), np.asarray(features, float).reshape(len(y), -1)])
    out = np.empty(np.shape(weights))
    for i, fold in enumerate(np.unique(folds)):
        held_out = folds == fold
        beta = logistic_newton(x, y, np.where(held_out, 0.0, weights), start[i])
        out[:, held_out] = sigmoid(beta @ x[held_out].T)
    return out
