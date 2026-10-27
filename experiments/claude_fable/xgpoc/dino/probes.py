"""Linear probes of xg-dino: what the embeddings encode about the automatic geometry (ticket 04; spec §Probes, council
C1 and C2, which win where they differ).

Fitting (`run`) only ever sees automatic targets (`xgpoc.dino.targets`, source "automatic"); a StatsBomb table is
refused. Population: the training shots with a stored frame; per target the rows where it is present. Every encoder
is probed on the same rows per target (the rows do not depend on the encoder).

Routing (C1). Outer folds = the host folds. For outer fold k the probe is fitted on encoder k's embeddings of the
rows in folds != k and read on fold k: for G-combo, encoder k is fold model k (it never saw fold k); frozen DINOv2 is
the same array for every k, on the same splits. Paired shots are read with each of the five fold probes, each on its
own fold model's embedding, and the five predictions are averaged (`Run.paired`).

Probes, per target and outer fold
    embedding            ridge (continuous and counts) or L2 logistic (binary; torch L-BFGS) on the standardised embedding
    distance_embedding   the same plus an unpenalised natural cubic spline of the automatic distance (knots at the
                         `KNOTS` quantiles of the training rows); the incremental gain over `distance` is the
                         embedding's information beyond distance
    distance             the unpenalised spline alone (OLS, or logistic by Newton): the flexible distance-only baseline
    constant             the training mean
    The penalty (`ALPHAS` for ridge, `LAMBDAS` for logistic, on the summed loss) is chosen per outer fold by an inner
    3-fold GroupKFold by match on the training rows (summed squared error / log-loss; ties to the smaller penalty).
    Standardisation and knots are fitted on the rows a model is trained on. Ridge with unpenalised columns Z is
    solved exactly by Frisch-Waugh (ridge of M_Z y on M_Z X), one Gram matrix per split serving every penalty and
    both variants (Cholesky per penalty, float64).
    Shuffled-target control: every continuous target permuted across its rows (`SHUFFLE_SEED`), embedding probe; it
    must not decode (R² <= 0).

Read-out (`metrics`): R² and MAE (continuous / counts), AUC (binary) of the out-of-fold predictions; 95% percentile
intervals from a match bootstrap of the population (`N_BOOT` draws, `BOOT_SEED`; probes are not refitted in draws, the
metric is recomputed with the draw's match weights; differences use the same draws). Rules (`Rules`, C2 defaults,
overridable by the protocol):
    encodes beyond distance   distance_embedding - distance: 95% lower bound > 0 and the gain >= `gain_min`
                              (ΔR² 0.05, ΔAUC 0.03)
    readable                  embedding-only R² >= 0.30 / AUC >= 0.70
    distance itself           "encodes" when embedding-only R² >= 0.50 (the reference is the constant)
Best layer (`best_layer`): per encoder and target, the layer with the largest mean over outer folds of the
embedding-only metric; any layer within `tie` (0.01 R² / 0.005 AUC) of the maximum counts as tied, and the latest
tied layer wins. For the "encodes" interval the layer is chosen inside the training folds (`nested`): per outer fold
by the inner-CV metric of the chosen penalty, with the same rule.
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from xgpoc.dino import targets as tg
from xgpoc.probe.readability import weighted_auc
from xgpoc.stack import fit as sf
from xgpoc.stack import stats as ss
from xgpoc.vendor.probes import weighted_r2

ALPHAS = tuple(float(a) for a in np.logspace(0, 6, 13))
LAMBDAS = tuple(float(a) for a in np.logspace(1, 6, 6))
N_INNER = 3
KNOTS = (0.05, 0.275, 0.5, 0.725, 0.95)
STD_EPS = 1e-6
LBFGS_MAX_ITER, LBFGS_TOLERANCE = 300, 1e-6     # AUC equal to 1e-4 against 1e-9 / 1000 on real embeddings
SHUFFLE_SEED = 0
N_BOOT, BOOT_SEED = ss.N_BOOT, ss.BOOT_SEED
VARIANTS = ("embedding", "distance_embedding")
REFERENCES = ("constant", "distance")
SHUFFLED = "_shuffled"


@dataclass(frozen=True)
class Rules:
    """The probe rules of C2; `from_protocol` takes the protocol's values where it sets them."""

    gain_min: dict = field(default_factory=lambda: {"continuous": 0.05, "binary": 0.03})
    readable_min: dict = field(default_factory=lambda: {"continuous": 0.30, "binary": 0.70})
    distance_encodes_r2: float = 0.50
    tie: dict = field(default_factory=lambda: {"continuous": 0.01, "binary": 0.005})

    @classmethod
    def from_protocol(cls, values: dict | None) -> "Rules":
        values = {k: v for k, v in (values or {}).items() if v is not None and k in cls.__dataclass_fields__}
        return cls(**values)


DEFAULT_RULES = Rules()


def metric_name(target: str) -> str:
    return "AUC" if tg.kind(target) == "binary" else "R2"


# ------------------------------------------------------------------ fits
def standardise(train: np.ndarray, *others: np.ndarray) -> list[np.ndarray]:
    mean, sd = train.mean(0), train.std(0) + STD_EPS
    return [((a - mean) / sd).astype(np.float32) for a in (train, *others)]


def spline(distance: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """[1, natural spline basis of distance]; NaN distance gives NaN rows."""
    d = np.asarray(distance, float)
    basis = sf.natural_spline_basis(np.nan_to_num(d), knots)
    out = np.column_stack([np.ones(len(d)), basis])
    out[~np.isfinite(d)] = np.nan
    return out


def ridge_paths(x: np.ndarray, y: np.ndarray, zs: dict[str, np.ndarray], evals: Sequence[tuple[np.ndarray, dict]],
                alphas: Sequence[float] = ALPHAS) -> dict[str, list[np.ndarray]]:
    """Per variant (unpenalised design zs[v], intercept included) the predictions (len(alphas), n_eval, targets) on
    every eval set (x_eval, {variant: z_eval}). `x` and the eval x standardised already."""
    xt = torch.from_numpy(np.ascontiguousarray(x, np.float32))
    gram = (xt.T @ xt).double()
    xty = torch.from_numpy(x.T.astype(np.float64) @ y)
    eye = torch.eye(x.shape[1], dtype=torch.float64)
    out = {}
    for variant, z in zs.items():
        zz_inv = np.linalg.inv(z.T @ z)
        xz = torch.from_numpy(x.T.astype(np.float64) @ z)
        a = gram - xz @ torch.from_numpy(zz_inv) @ xz.T
        b = xty - xz @ torch.from_numpy(zz_inv @ (z.T @ y))
        preds = [[] for _ in evals]
        for alpha in alphas:
            beta = torch.cholesky_solve(b, torch.linalg.cholesky(a + alpha * eye)).numpy()
            gamma = zz_inv @ (z.T @ (y - x @ beta))
            for i, (xe, ze) in enumerate(evals):
                preds[i].append(ze[variant] @ gamma + xe @ beta)
        out[variant] = [np.stack(p) for p in preds]
    return out


def _logistic(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, lam: float, start: torch.Tensor) -> torch.Tensor:
    """[gamma (unpenalised z), beta] minimising mean log-loss + 0.5 lam / n |beta|² (torch L-BFGS, strong Wolfe, float64)."""
    n, q = len(y), z.shape[1]
    w = start.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([w], lr=1, max_iter=LBFGS_MAX_ITER, tolerance_grad=LBFGS_TOLERANCE, tolerance_change=1e-9,
                            history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        eta = z @ w[:q] + x @ w[q:]
        loss = (torch.nn.functional.softplus(eta) - y * eta).mean() + 0.5 * lam / n * (w[q:] @ w[q:])
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach()


def logistic_paths(x: np.ndarray, y: np.ndarray, zs: dict[str, np.ndarray], evals: Sequence[tuple[np.ndarray, dict]],
                   lambdas: Sequence[float] = LAMBDAS, valid_y: np.ndarray | None = None,
                   down_to: dict[str, float] | None = None) -> dict[str, list[np.ndarray]]:
    """`ridge_paths` for one binary target with L2 logistic; predictions are logits (len(lambdas), n_eval, 1),
    warm-started from the largest penalty down. The path stops early (later penalties NaN): below `down_to[variant]`
    (the outer fit needs only the chosen penalty), and, given `valid_y`, once the log-loss on the first eval set has
    risen at two consecutive penalties (inner CV; the chosen penalty lies before)."""
    xt = torch.from_numpy(x.astype(np.float64))
    yt = torch.from_numpy(np.asarray(y, np.float64).reshape(-1))
    out = {}
    order = np.argsort(lambdas)[::-1]
    for variant, z in zs.items():
        zt = torch.from_numpy(z.astype(np.float64))
        w = torch.zeros(z.shape[1] + x.shape[1], dtype=torch.float64)
        preds = [[np.full((len(xe), 1), np.nan)] * len(lambdas) for xe, _ in evals]
        losses = []
        for j in order:
            if down_to is not None and lambdas[j] < down_to[variant]:
                break
            w = _logistic(xt, yt, zt, lambdas[j], w)
            coef = w.numpy()
            for i, (xe, ze) in enumerate(evals):
                preds[i] = [*preds[i][:j], (ze[variant] @ coef[:z.shape[1]] + xe @ coef[z.shape[1]:])[:, None], *preds[i][j + 1:]]
            if valid_y is not None:
                losses.append(float(_loss(preds[0][j][None], valid_y.reshape(-1, 1), True)[0, 0]))
                if len(losses) >= 3 and losses[-1] > losses[-2] > losses[-3]:
                    break
        out[variant] = [np.stack(p) for p in preds]
    return out


def distance_reference(z: np.ndarray, y: np.ndarray, z_evals: Sequence[np.ndarray], binary: bool) -> list[np.ndarray]:
    """Distance-only predictions (OLS / unpenalised logistic logit) on each eval design, (n_eval, targets)."""
    if binary:
        beta = sf.logistic_fit(z[:, 1:], y[:, 0], np.zeros(z.shape[1] - 1))
        return [(ze @ beta)[:, None] for ze in z_evals]
    coef = np.linalg.lstsq(z, y, rcond=None)[0]
    return [ze @ coef for ze in z_evals]


def _loss(pred: np.ndarray, y: np.ndarray, binary: bool) -> np.ndarray:
    """Summed loss per penalty and target, pred (P, n, T)."""
    with np.errstate(invalid="ignore"):
        out = (np.logaddexp(0.0, pred) - y * pred).sum(1) if binary else ((pred - y) ** 2).sum(1)
    return np.where(np.isnan(out), np.inf, out)     # penalties a stopped path did not fit


def point_metric(y: np.ndarray, pred: np.ndarray, binary: bool) -> float:
    if binary:
        return float(weighted_auc(pred, y, np.zeros(len(y), int)))
    return float(weighted_r2(y, pred))


# ------------------------------------------------------------------ data and runs
@dataclass
class ProbeData:
    """The probe population: automatic targets (index shot_id), match and host fold per row, and, given, the paired
    shots' automatic targets (for their distance; never fitted)."""

    targets: pd.DataFrame
    games: np.ndarray
    folds: np.ndarray
    paired: pd.DataFrame | None = None

    def __post_init__(self):
        for t in (self.targets, self.paired):
            if t is not None and t.attrs.get("source") != "automatic":
                raise ValueError("probes are fitted and read on automatic targets only; StatsBomb is evaluation only")
        if not (len(self.targets) == len(self.games) == len(self.folds)):
            raise ValueError("one game and fold per target row")
        if pd.DataFrame({"g": self.games, "f": self.folds}).groupby("g").f.nunique().max() > 1:
            raise ValueError("a match lies in two outer folds")


@dataclass
class Encoder:
    """`train(k)`: (n, d) embedding of the population rows by the encoder of outer fold k; `paired(k)` the same for
    the paired shots (None: no paired read-out)."""

    name: str
    train: Callable[[int], np.ndarray]
    paired: Callable[[int], np.ndarray] | None = None


@dataclass
class TargetRun:
    target: str
    rows: np.ndarray                       # population row indices
    y: np.ndarray
    oof: dict[str, np.ndarray]             # variant / reference / "embedding_shuffled" -> OOF prediction per row
    penalty: dict[str, list[float]]        # variant -> chosen penalty per outer fold
    inner: dict[str, list[float]]          # variant -> inner-CV metric at the chosen penalty, per outer fold
    paired_folds: dict[str, list[np.ndarray]] = field(default_factory=dict)   # variant / reference -> per fold probe

    @property
    def paired(self) -> dict[str, np.ndarray]:
        """Paired predictions: the mean over the fold probes (each read on its own fold model's embedding)."""
        return {k: np.mean(v, axis=0) for k, v in self.paired_folds.items()}


@dataclass
class Run:
    encoder: str
    folds: list[int]
    targets: dict[str, TargetRun]
    timing: list[dict]


def row_groups(data: ProbeData, targets: Sequence[str]) -> list[tuple[np.ndarray, list[str]]]:
    """Continuous targets sharing the same rows are fitted together; each binary target alone."""
    groups: dict[tuple, list[str]] = {}
    for t in targets:
        mask = data.targets[t].notna().to_numpy()
        groups.setdefault((tg.kind(t) == "binary", t if tg.kind(t) == "binary" else None, mask.tobytes()), []).append(t)
    return [(np.flatnonzero(data.targets[ts[0]].notna().to_numpy()), ts) for ts in groups.values()]


def inner_splits(games: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """The inner 3-fold GroupKFold by match (positions into `games`)."""
    return list(GroupKFold(N_INNER).split(games, groups=games))


def _fold_fit(x_all: np.ndarray, data: ProbeData, rows: np.ndarray, y: np.ndarray, binary: bool, k: int,
              x_paired: np.ndarray | None) -> dict:
    """Inner CV, outer fit and reads of one row group in outer fold k."""
    dist = data.targets.distance.to_numpy()
    train, test = rows[data.folds[rows] != k], rows[data.folds[rows] == k]
    ytr = y[data.folds[rows] != k]
    paths, grid = (logistic_paths, LAMBDAS) if binary else (ridge_paths, ALPHAS)

    def split(tr: np.ndarray, evals: list[tuple[np.ndarray, np.ndarray | None]], ytrain: np.ndarray, **stop):
        knots = np.quantile(dist[tr], KNOTS)
        xs = standardise(x_all[tr], *[e[0] for e in evals])
        ztr = spline(dist[tr], knots)
        zs = {"embedding": ztr[:, :1], "distance_embedding": ztr}
        ev = []
        for (_, de), xse in zip(evals, xs[1:], strict=True):
            ze = np.nan_to_num(spline(de, knots))
            ev.append((xse, {"embedding": ze[:, :1], "distance_embedding": ze}))
        return paths(xs[0], ytrain, zs, ev, grid, **(stop if binary else {})), ztr, knots

    loss = {v: 0.0 for v in VARIANTS}
    inner_pred = {v: np.empty((len(grid), len(train), y.shape[1])) for v in VARIANTS}
    for itr, iva in inner_splits(data.games[train]):
        out, _, _ = split(train[itr], [(x_all[train[iva]], dist[train[iva]])], ytr[itr], valid_y=ytr[iva])
        for v in VARIANTS:
            loss[v] = loss[v] + _loss(out[v][0], ytr[iva], binary)
            inner_pred[v][:, iva] = out[v][0]
    chosen = {v: np.argmin(loss[v], axis=0) for v in VARIANTS}
    inner = {v: [point_metric(ytr[:, j], inner_pred[v][chosen[v][j], :, j], binary) for j in range(y.shape[1])]
             for v in VARIANTS}

    evals = [(x_all[test], dist[test])]
    if x_paired is not None:
        evals.append((x_paired, data.paired.distance.to_numpy()))
    out, ztr, knots = split(train, evals, ytr, down_to={v: grid[chosen[v][0]] for v in VARIANTS})
    res = {"test": test, "chosen": {v: [grid[c] for c in chosen[v]] for v in VARIANTS}, "inner": inner, "pred": {}}
    for i, name in enumerate(("test", "paired")[:len(evals)]):
        res["pred"][name] = {v: out[v][i][chosen[v], :, np.arange(y.shape[1])].T for v in VARIANTS}
        z_eval = spline(evals[i][1], knots)
        ref = distance_reference(ztr, ytr, [np.nan_to_num(z_eval)], binary)[0]
        ref[~np.isfinite(z_eval).all(1)] = np.nan
        res["pred"][name]["distance"] = ref
        res["pred"][name]["constant"] = np.broadcast_to(ytr.mean(0), (len(evals[i][0]), y.shape[1])).copy()
    for name in res["pred"].get("paired", {}):
        if name == "distance_embedding":
            bad = ~np.isfinite(data.paired.distance.to_numpy())
            res["pred"]["paired"][name][bad] = np.nan
    return res


def run(data: ProbeData, encoder: Encoder, targets: Sequence[str] = tg.TARGETS, folds: Sequence[int] | None = None,
        log: Callable[[str], None] | None = None) -> Run:
    """Out-of-fold probes of every target for one encoder (module docstring)."""
    folds = sorted(np.unique(data.folds)) if folds is None else list(folds)
    groups = row_groups(data, targets)
    results: dict[str, TargetRun] = {}
    shuffled: dict[str, np.ndarray] = {}
    plan = []
    for rows, names in groups:
        binary = tg.kind(names[0]) == "binary"
        y = data.targets.iloc[rows][names].to_numpy(float)
        cols = list(names)
        if not binary:
            rng = np.random.default_rng(SHUFFLE_SEED)
            y = np.column_stack([y, *[rng.permutation(y[:, j]) for j in range(len(names))]])
            cols += [n + SHUFFLED for n in names]
        plan.append((rows, cols, y, binary))
        for j, n in enumerate(names):
            results[n] = TargetRun(n, rows, y[:, j], {}, {v: [] for v in VARIANTS}, {v: [] for v in VARIANTS})
    oof = {id(p[0]): {} for p in plan}
    paired_sum: dict[int, dict] = {id(p[0]): {} for p in plan}
    timing = []
    for k in folds:
        x_all = np.asarray(encoder.train(k), np.float32)
        x_paired = np.asarray(encoder.paired(k), np.float32) if encoder.paired is not None and data.paired is not None else None
        for rows, cols, y, binary in plan:
            t0, c0 = time.perf_counter(), time.process_time()
            res = _fold_fit(x_all, data, rows, y, binary, k, x_paired)
            timing.append({"encoder": encoder.name, "fold": k, "targets": cols[0] if binary else "+".join(c for c in cols if not c.endswith(SHUFFLED)),
                           "model": "logistic" if binary else "ridge", "rows": len(rows), "dim": x_all.shape[1],
                           "wall_s": time.perf_counter() - t0, "cpu_s": time.process_time() - c0})
            if log:
                log(f"{encoder.name} fold {k} {timing[-1]['model']} {timing[-1]['targets']}: {timing[-1]['wall_s']:.1f} s")
            mask = data.folds[rows] == k
            store = oof[id(rows)]
            for name, pred in res["pred"]["test"].items():
                store.setdefault(name, np.full((len(rows), y.shape[1]), np.nan))[mask] = pred
            if "paired" in res["pred"]:
                for name, pred in res["pred"]["paired"].items():
                    paired_sum[id(rows)].setdefault(name, []).append(pred)
            for j, c in enumerate(cols):
                if c in results:
                    for v in VARIANTS:
                        results[c].penalty[v].append(float(res["chosen"][v][j]))
                        results[c].inner[v].append(float(res["inner"][v][j]))
    for rows, cols, y, _ in plan:
        for j, c in enumerate(cols):
            name = c.removesuffix(SHUFFLED)
            if c.endswith(SHUFFLED):
                results[name].oof["embedding_shuffled"] = oof[id(rows)]["embedding"][:, j]
                shuffled[name] = y[:, j]
                continue
            results[name].oof |= {v: p[:, j] for v, p in oof[id(rows)].items()}
            results[name].paired_folds = {v: [p[:, j] for p in ps] for v, ps in paired_sum[id(rows)].items()}
    for name, y in shuffled.items():
        results[name].oof["shuffled_target"] = y
    return Run(encoder.name, folds, results, timing)


def merge(parts: Sequence[Run]) -> Run:
    """One run from runs of the same encoder on disjoint outer folds (in fold order)."""
    parts = sorted(parts, key=lambda r: r.folds[0])
    folds = [k for r in parts for k in r.folds]
    if len(set(folds)) != len(folds) or len({r.encoder for r in parts}) != 1:
        raise ValueError("merge needs one encoder and disjoint folds")
    targets = {}
    for name, first in parts[0].targets.items():
        oof = {key: np.where(np.isnan(v), 0.0, v) for key, v in first.oof.items()}
        seen = {key: ~np.isnan(v) for key, v in first.oof.items()}
        out = TargetRun(name, first.rows, first.y, oof, {v: [] for v in VARIANTS}, {v: [] for v in VARIANTS},
                        {k: [] for k in first.paired_folds})
        for r in parts:
            t = r.targets[name]
            if not np.array_equal(t.rows, first.rows):
                raise ValueError("parts differ in rows")
            for key, v in t.oof.items():
                if key == "shuffled_target":
                    continue
                fill = ~np.isnan(v)
                out.oof[key][fill] = v[fill]
                seen[key] |= fill
            for v in VARIANTS:
                out.penalty[v] += t.penalty[v]
                out.inner[v] += t.inner[v]
            for k, ps in t.paired_folds.items():
                out.paired_folds[k] += ps
        for key in out.oof:
            if key != "shuffled_target":
                out.oof[key][~seen[key]] = np.nan
        targets[name] = out
    return Run(parts[0].encoder, folds, targets, [t for r in parts for t in r.timing])


# ------------------------------------------------------------------ layer choice
def best_layer(scores: dict[str, float], binary: bool, rules: Rules = DEFAULT_RULES) -> str:
    """The layer with the best score; layers within the tie margin of the best are tied and the latest (in the
    order of `scores`) wins."""
    names = list(scores)
    values = np.array([scores[n] for n in names], float)
    tied = np.flatnonzero(values >= np.nanmax(values) - rules.tie["binary" if binary else "continuous"])
    return names[int(tied.max())]


def fold_scores(run: Run, data: ProbeData, target: str, key: str = "embedding") -> list[float]:
    """Embedding-only metric of each outer fold's held-out rows."""
    t, binary = run.targets[target], tg.kind(target) == "binary"
    folds = data.folds[t.rows]
    return [point_metric(t.y[folds == k], t.oof[key][folds == k], binary) for k in run.folds]


def chosen_layer(runs: dict[str, Run], data: ProbeData, target: str, rules: Rules = DEFAULT_RULES) -> str:
    """Best layer on all development OOF (`runs` in block order): mean over outer folds of the embedding-only metric."""
    return best_layer({layer: float(np.mean(fold_scores(r, data, target))) for layer, r in runs.items()},
                      tg.kind(target) == "binary", rules)


def nested(runs: dict[str, Run], data: ProbeData, target: str, rules: Rules = DEFAULT_RULES) -> tuple[TargetRun, list[str]]:
    """OOF predictions with the layer chosen per outer fold inside its training folds (inner-CV embedding-only metric
    at the chosen penalty), and the layer per fold."""
    first = next(iter(runs.values()))
    t0, binary = first.targets[target], tg.kind(target) == "binary"
    for r in runs.values():
        if not np.array_equal(r.targets[target].rows, t0.rows) or r.folds != first.folds:
            raise ValueError("layers must be probed on the same rows and folds")
    folds = data.folds[t0.rows]
    oof = {key: np.full(len(t0.rows), np.nan) for key in t0.oof}
    chosen = []
    for i, k in enumerate(first.folds):
        layer = best_layer({name: r.targets[target].inner["embedding"][i] for name, r in runs.items()}, binary, rules)
        chosen.append(layer)
        for key in oof:
            oof[key][folds == k] = runs[layer].targets[target].oof[key][folds == k]
    return TargetRun(target, t0.rows, t0.y, oof, {}, {}), chosen


# ------------------------------------------------------------------ read-out
@dataclass(frozen=True)
class Draws:
    """Match-bootstrap draws over a population: per-row weights (B, n) and per-match counts (B, matches)."""

    codes: np.ndarray
    weights: np.ndarray
    counts: np.ndarray


def draws(games: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> Draws:
    games = np.asarray(games)
    codes, uniques = pd.factorize(games)
    weights = ss.bootstrap_weights(games, n_boot, seed)
    first = np.array([np.flatnonzero(codes == c)[0] for c in range(len(uniques))])
    return Draws(codes, weights, weights[:, first])


def metric_draws(target: str, y: np.ndarray, pred: np.ndarray, rows: np.ndarray, d: Draws) -> dict[str, tuple[float, np.ndarray]]:
    """{metric: (point, draws)}: AUC for binary targets, R² and MAE otherwise, on population rows `rows`."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    if tg.kind(target) == "binary":
        codes = d.codes[rows]
        return {"AUC": (float(weighted_auc(pred, y, codes)), weighted_auc(pred, y, codes, d.counts))}
    w = d.weights[:, rows]
    err = np.abs(y - pred)
    with np.errstate(invalid="ignore", divide="ignore"):
        mae = (w @ err) / w.sum(1)
    return {"R2": (float(weighted_r2(y, pred)), weighted_r2(y, pred, w)), "MAE": (float(err.mean()), mae)}


def summary(point: float, boot: np.ndarray) -> dict:
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return {"point": point, "lo": float(lo), "hi": float(hi)}


def target_metrics(t: TargetRun, d: Draws, keys: Sequence[str] | None = None, mask: np.ndarray | None = None) -> dict:
    """{key: {metric: (point, draws)}} for the OOF predictions of `t` (optionally on a row mask of t.rows)."""
    keys = [k for k in (keys or t.oof) if k in t.oof and k != "shuffled_target"]
    m = np.ones(len(t.rows), bool) if mask is None else mask
    out = {}
    for key in keys:
        y = t.oof["shuffled_target"] if key == "embedding_shuffled" else t.y
        out[key] = metric_draws(t.target, y[m], t.oof[key][m], t.rows[m], d)
    return out


def decide(target: str, md: dict, rules: Rules = DEFAULT_RULES) -> dict:
    """The C2 rules on one encoder's metrics of one target (`target_metrics` output)."""
    kind = tg.kind(target)
    name = metric_name(target)
    emb = md["embedding"][name][0]
    out = {"metric": name, "readable": bool(emb >= rules.readable_min[kind])}
    if target == "distance":
        out["encodes"] = bool(emb >= rules.distance_encodes_r2)
        return out
    gain = md["distance_embedding"][name][0] - md["distance"][name][0]
    boot = md["distance_embedding"][name][1] - md["distance"][name][1]
    s = summary(gain, boot)
    out |= {"gain": s, "encodes_beyond_distance": bool(s["lo"] > 0 and gain >= rules.gain_min[kind])}
    return out


def rows_table(encoder: str, target: str, md: dict, decision: dict, layer: str, stratum: str = "all") -> list[dict]:
    rows = []
    for key, metrics in md.items():
        for metric, (point, boot) in metrics.items():
            rows.append({"encoder": encoder, "layer": layer, "target": target, "stratum": stratum, "probe": key,
                         "metric": metric} | summary(point, boot))
    if "gain" in decision:
        rows.append({"encoder": encoder, "layer": layer, "target": target, "stratum": stratum, "probe": "gain_over_distance",
                     "metric": decision["metric"]} | decision["gain"])
    return rows


def difference(a: dict, b: dict, key: str, metric: str) -> dict:
    """a - b of one probe's metric on the same rows and draws."""
    return summary(a[key][metric][0] - b[key][metric][0], a[key][metric][1] - b[key][metric][1])


def paired_metrics(t: TargetRun, statsbomb: pd.Series, detector: pd.Series, d: Draws) -> dict:
    """Frozen predictions of the paired shots against StatsBomb targets (descriptive): {key: {metric: (point,
    draws)}, "n": rows}. Probes and references on rows with a StatsBomb value and a finite prediction; the detector's
    own value on rows where it and StatsBomb exist. Nothing here is fitted."""
    y = statsbomb.to_numpy(float)
    out = {}
    for key, pred in [*t.paired.items(), ("detector", detector.to_numpy(float))]:
        m = np.isfinite(y) & np.isfinite(pred)
        out[key] = metric_draws(t.target, y[m], pred[m], np.flatnonzero(m), d) | {"n": int(m.sum())}
    return out
