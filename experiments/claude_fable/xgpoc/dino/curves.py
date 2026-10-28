"""Per-layer readability curves of xg-dino (ticket 06; spec §Probes, councils C1 and C2, which win where they differ).

The probes of `xgpoc.dino.probes` (ticket 04) on every block, made cheap enough for 28 layers x 2 offsets:

    Ridge for every target. One eigendecomposition per layer x outer fold x inner split x variant of the
    Frisch-Waugh Gram matrix A = X' M_Z X (Z the unpenalised columns: the intercept, or the intercept and the distance
    spline) solves all targets and all penalties at once: beta(alpha) = V diag(1 / (lambda + alpha)) V' X' M_Z y.
    `ridge_paths` has the signature and output of `probes.ridge_paths` (one Cholesky per penalty) and equals it.
    Binary targets get the same ridge on their 0/1 values (penalty by inner-CV squared error) and are scored by the AUC
    of the ridge output; the logistic probe of ticket 04 is refitted at the best layer only (`probes.run`).
    Targets sharing their rows are fitted together (binary ones included), so one decomposition serves a row group.

Everything else is ticket 04's: the host outer folds, the inner 3-fold GroupKFold by match, standardisation and
spline knots on the rows a model is trained on, the penalty grid `probes.ALPHAS` (ties to the smaller penalty), the
references (constant; distance-only spline by OLS, or by logistic for binary targets; distance + embedding), the
shuffled-target control of the continuous targets (same permutations: one generator per row group, in target order),
the C1 routing (encoder k for outer fold k), and the read-out (`probes.target_metrics`, `decide`, `nested`).
The result is a `probes.Run`, so the ticket 04 read-out applies unchanged.
"""

import time
from collections.abc import Callable, Sequence

import numpy as np
import torch
from scipy import sparse
from scipy.stats import spearmanr

from xgpoc.dino import probes as pr
from xgpoc.dino import targets as tg


def ridge_paths(x: np.ndarray, y: np.ndarray, zs: dict[str, np.ndarray], evals: Sequence[tuple[np.ndarray, dict]],
                alphas: Sequence[float] = pr.ALPHAS, device: str = "cpu") -> dict[str, list[np.ndarray]]:
    """`probes.ridge_paths` by one eigendecomposition per variant (float64 on `device`)."""
    f64 = {"dtype": torch.float64, "device": device}
    xt = torch.as_tensor(np.ascontiguousarray(x), **f64)
    yt = torch.as_tensor(np.asarray(y, float), **f64)
    gram, xty = xt.T @ xt, xt.T @ yt
    al = torch.as_tensor(np.asarray(alphas, float), **f64)
    out = {}
    for variant, z in zs.items():
        zt = torch.as_tensor(np.asarray(z, float), **f64)
        zz_inv = torch.linalg.inv(zt.T @ zt)
        xz = xt.T @ zt
        a = gram - xz @ zz_inv @ xz.T
        lam, v = torch.linalg.eigh(0.5 * (a + a.T))
        zty = zt.T @ yt
        c = v.T @ (xty - xz @ (zz_inv @ zty))
        beta = v @ (c[None] / (lam[None, :, None] + al[:, None, None]))      # (P, d, T)
        gamma = zz_inv @ (zty[None] - xz.T @ beta)                           # (P, q, T)
        preds = []
        for xe, ze in evals:
            xet = torch.as_tensor(np.ascontiguousarray(xe), **f64)
            zet = torch.as_tensor(np.asarray(ze[variant], float), **f64)
            preds.append((zet @ gamma + xet @ beta).cpu().numpy())
        out[variant] = preds
    return out


def mask_groups(data: pr.ProbeData, targets: Sequence[str]) -> list[tuple[np.ndarray, list[str]]]:
    """Targets with the same present rows, in `targets` order (binary ones included)."""
    groups: dict[bytes, list[str]] = {}
    for t in targets:
        groups.setdefault(data.targets[t].notna().to_numpy().tobytes(), []).append(t)
    return [(np.flatnonzero(data.targets[ts[0]].notna().to_numpy()), ts) for ts in groups.values()]


def references(ztr: np.ndarray, ytr: np.ndarray, z_eval: np.ndarray, binary: np.ndarray) -> np.ndarray:
    """Distance-only predictions per column: OLS, or the logistic logit for binary columns (ticket 04's)."""
    out = np.empty((len(z_eval), ytr.shape[1]))
    if (~binary).any():
        out[:, ~binary] = pr.distance_reference(ztr, ytr[:, ~binary], [z_eval], False)[0]
    for j in np.flatnonzero(binary):
        out[:, j] = pr.distance_reference(ztr, ytr[:, [j]], [z_eval], True)[0][:, 0]
    return out


def geometry_basis(geo_train: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    """[1, standardised geometry projected on its non-null principal directions] for the training rows and each other
    set: NaN is the training mean, constant columns drop out, collinear columns collapse, so the unpenalised design of
    the `geometry_embedding` probe has full rank."""
    mean = np.nanmean(geo_train, 0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    filled = [np.where(np.isnan(a), mean, a) for a in (geo_train, *others)]
    sd = filled[0].std(0)
    std = [(a - mean) / np.where(sd > pr.STD_EPS, sd, 1.0) for a in filled]
    _, s, vt = np.linalg.svd(std[0], full_matrices=False)
    basis = vt[s > 1e-6 * s[0]].T
    return [np.column_stack([np.ones(len(a)), a @ basis]) for a in std]


def fold_fit(x_all: np.ndarray, data: pr.ProbeData, rows: np.ndarray, y: np.ndarray, binary: np.ndarray, k: int,
             x_paired: np.ndarray | None, device: str = "cpu", geometry: np.ndarray | None = None) -> dict:
    """`probes._fold_fit` with the eigendecomposition ridge for every column of one row group. `geometry` (one row per
    population row, NaN allowed): a further probe `geometry_embedding` (unpenalised geometry design plus the embedding)
    and the reference `geometry` (that design alone); continuous columns only, no paired rows."""
    if geometry is not None and (binary.any() or x_paired is not None):
        raise ValueError("the geometry probe takes continuous columns and no paired rows")
    variants = (*pr.VARIANTS, *(("geometry_embedding",) if geometry is not None else ()))
    dist = data.targets.distance.to_numpy()
    in_train = data.folds[rows] != k
    train, test, ytr = rows[in_train], rows[~in_train], y[in_train]
    grid = pr.ALPHAS

    def split(tr, evals, ytrain):
        knots = np.quantile(dist[tr], pr.KNOTS)
        xs = pr.standardise(x_all[tr], *[e[0] for e in evals])
        ztr = pr.spline(dist[tr], knots)
        zs = {"embedding": ztr[:, :1], "distance_embedding": ztr}
        zgs = geometry_basis(geometry[tr], *[geometry[e[2]] for e in evals]) if geometry is not None else None
        if zgs is not None:
            zs["geometry_embedding"] = zgs[0]
        ev = []
        for i, ((_, de, _), xse) in enumerate(zip(evals, xs[1:], strict=True)):
            ze = np.nan_to_num(pr.spline(de, knots))
            ev.append((xse, {"embedding": ze[:, :1], "distance_embedding": ze}
                       | ({"geometry_embedding": zgs[1 + i]} if zgs is not None else {})))
        return ridge_paths(xs[0], ytrain, zs, ev, grid, device), ztr, knots, zgs

    loss = {v: 0.0 for v in variants}
    inner_pred = {v: np.empty((len(grid), len(train), y.shape[1])) for v in variants}
    for itr, iva in pr.inner_splits(data.games[train]):
        out, _, _, _ = split(train[itr], [(x_all[train[iva]], dist[train[iva]], train[iva])], ytr[itr])
        for v in variants:
            loss[v] = loss[v] + pr._loss(out[v][0], ytr[iva], False)
            inner_pred[v][:, iva] = out[v][0]
    chosen = {v: np.argmin(loss[v], axis=0) for v in variants}
    inner = {v: [pr.point_metric(ytr[:, j], inner_pred[v][chosen[v][j], :, j], bool(binary[j])) for j in range(y.shape[1])]
             for v in pr.VARIANTS}

    evals = [(x_all[test], dist[test], test)]
    if x_paired is not None:
        evals.append((x_paired, data.paired.distance.to_numpy(), None))
    out, ztr, knots, zgs = split(train, evals, ytr)
    res = {"test": test, "chosen": {v: [grid[c] for c in chosen[v]] for v in variants}, "inner": inner, "pred": {}}
    for i, name in enumerate(("test", "paired")[:len(evals)]):
        res["pred"][name] = {v: out[v][i][chosen[v], :, np.arange(y.shape[1])].T for v in variants}
        if zgs is not None:
            res["pred"][name]["geometry"] = zgs[1 + i] @ np.linalg.lstsq(zgs[0], ytr, rcond=None)[0]
        z_eval = pr.spline(evals[i][1], knots)
        ref = references(ztr, ytr, np.nan_to_num(z_eval), binary)
        ref[~np.isfinite(z_eval).all(1)] = np.nan
        res["pred"][name]["distance"] = ref
        res["pred"][name]["constant"] = np.broadcast_to(ytr.mean(0), (len(evals[i][0]), y.shape[1])).copy()
    if "paired" in res["pred"]:
        res["pred"]["paired"]["distance_embedding"][~np.isfinite(data.paired.distance.to_numpy())] = np.nan
    return res


def run(data: pr.ProbeData, encoder: pr.Encoder, targets: Sequence[str] = tg.TARGETS, folds: Sequence[int] | None = None,
        paired: bool = False, device: str = "cpu", log: Callable[[str], None] | None = None,
        geometry: np.ndarray | None = None) -> pr.Run:
    """Out-of-fold ridge probes of every target for one encoder (module docstring); paired reads only when `paired`;
    `geometry` adds the `geometry_embedding` probe and the `geometry` reference (`fold_fit`)."""
    folds = sorted(np.unique(data.folds)) if folds is None else list(folds)
    results: dict[str, pr.TargetRun] = {}
    plan = []
    for rows, names in mask_groups(data, targets):
        y = data.targets.iloc[rows][names].to_numpy(float)
        cont = [n for n in names if tg.kind(n) != "binary"]
        rng = np.random.default_rng(pr.SHUFFLE_SEED)
        shuffled = [rng.permutation(y[:, names.index(n)]) for n in cont]
        cols = [*names, *(n + pr.SHUFFLED for n in cont)]
        y = np.column_stack([y, *shuffled]) if shuffled else y
        binary = np.array([tg.kind(c.removesuffix(pr.SHUFFLED)) == "binary" and not c.endswith(pr.SHUFFLED) for c in cols])
        plan.append((rows, cols, y, binary))
        for j, n in enumerate(names):
            results[n] = pr.TargetRun(n, rows, y[:, j], {}, {v: [] for v in pr.VARIANTS}, {v: [] for v in pr.VARIANTS})
    oof = [{} for _ in plan]
    paired_sum = [{} for _ in plan]
    timing = []
    for k in folds:
        x_all = np.asarray(encoder.train(k), np.float32)
        x_paired = np.asarray(encoder.paired(k), np.float32) if paired else None
        for g, (rows, cols, y, binary) in enumerate(plan):
            t0, c0 = time.perf_counter(), time.process_time()
            res = fold_fit(x_all, data, rows, y, binary, k, x_paired, device, geometry)
            timing.append({"encoder": encoder.name, "fold": k, "targets": "+".join(c for c in cols if not c.endswith(pr.SHUFFLED)),
                           "model": "ridge_eigh", "rows": len(rows), "dim": x_all.shape[1],
                           "wall_s": time.perf_counter() - t0, "cpu_s": time.process_time() - c0})
            if log:
                log(f"{encoder.name} fold {k} {timing[-1]['targets']}: {timing[-1]['wall_s']:.1f} s")
            mask = data.folds[rows] == k
            for name, pred in res["pred"]["test"].items():
                oof[g].setdefault(name, np.full((len(rows), y.shape[1]), np.nan))[mask] = pred
            for name, pred in res["pred"].get("paired", {}).items():
                paired_sum[g].setdefault(name, []).append(pred)
            for j, c in enumerate(cols):
                if c in results:
                    for v in pr.VARIANTS:
                        results[c].penalty[v].append(float(res["chosen"][v][j]))
                        results[c].inner[v].append(float(res["inner"][v][j]))
    for g, (_, cols, y, _) in enumerate(plan):
        for j, c in enumerate(cols):
            name = c.removesuffix(pr.SHUFFLED)
            if c.endswith(pr.SHUFFLED):
                results[name].oof["embedding_shuffled"] = oof[g]["embedding"][:, j]
                results[name].oof["shuffled_target"] = y[:, j]
                continue
            results[name].oof |= {v: p[:, j] for v, p in oof[g].items()}
            results[name].paired_folds = {v: [p[:, j] for p in ps] for v, ps in paired_sum[g].items()}
    return pr.Run(encoder.name, folds, results, timing)


# ------------------------------------------------------------------ read-out helpers
def first_readable(scores: dict[str, float], threshold: float) -> str | None:
    """The first layer (in `scores` order) whose score reaches `threshold`, None if none does."""
    return next((name for name, s in scores.items() if s >= threshold), None)


def weighted_spearman(y: np.ndarray, pred: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Spearman correlation with row weights (B, n) read as replication counts (midranks for ties); a scalar array
    without weights."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    if weights is None:
        return np.asarray(spearmanr(y, pred).statistic)
    w = np.asarray(weights, float)

    def ranks(v):
        uniq, inv = np.unique(v, return_inverse=True)
        onehot = sparse.csr_matrix((np.ones(len(v)), (np.arange(len(v)), inv)), shape=(len(v), len(uniq)))
        per = np.asarray((onehot.T @ w.T).T)                    # (B, U) weight per unique value
        mid = np.cumsum(per, 1) - per + (per + 1) / 2
        return mid[:, inv]
    ry, rp = ranks(y), ranks(pred)
    total = w.sum(1, keepdims=True)
    my, mp = (w * ry).sum(1, keepdims=True) / total, (w * rp).sum(1, keepdims=True) / total
    cov = (w * (ry - my) * (rp - mp)).sum(1)
    return cov / np.sqrt((w * (ry - my) ** 2).sum(1) * (w * (rp - mp) ** 2).sum(1))
