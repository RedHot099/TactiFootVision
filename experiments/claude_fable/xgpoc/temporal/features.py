"""Frame PCA, the window features and the fold-wise design matrices of arms R, S and T (ticket 01; spec, "Dane i
arms").

Arms (`arm_columns`; every column but z_g_combo is named `<feature>@<offset>`, the offset being the latest frame the
column uses, `provenance.check_columns`)
    R   z_g_combo                                                                   reference
    S   R + pc01..pcK@o_max (the PCA-K scores of the o_max frame)
          + missing@<offset> for every window offset                                static control
    T   S + pcJ_vs_start@o_max  (score at o_max minus score at the window start)
          + pcJ_vs_mid@o_max    (score at o_max minus score at the middle frame; omitted when the middle frame is the
                                 start, a two-frame window)
          + pcJ_slope@o_max     (least-squares slope per second of the score over every window frame; omitted when the
                                 window holds at most three frames, where it is the start difference divided by the
                                 window length)                                     confirmatory
    Every missing indicator is in S, so T - S holds only the motion scores: which frames a cut removed (the cut
    pattern) cannot by itself make T beat S.
    K = 32 (`K`), fixed without labels; 16 and 64 are a descriptive sensitivity.
Fold-wise preprocessing (`fold_designs`), fitted on the training rows only, never on y
    1. imputation (`inputs.impute`): an unusable frame takes the nearest earlier usable window frame, else the nearest
       later one, with a missing indicator per offset (row-wise, no fitted state);
    2. PCA-K (`FramePCA`): fitted on every usable window frame of the training shots pooled (randomized SVD with a
       fixed seed, `PCA_SEED`), applied to every frame of the scored shots, imputed frames included;
    3. features (`feature_table`) and standardisation with the training mean and standard deviation (scale 1 if
       constant).
    Each outer xgv fold is scored by the design fitted on the other four; its inner designs (one per other fold, for
    the match-grouped penalty CV of `xgpoc.stack.fit.INNER_CV`) refit steps 2 and 3 on the inner training rows.
    The designs depend on the inputs and the folds only, so every outcome vector reuses them.
Explained variance (`explained_variance`): the share of the training frames' total variance explained by the first
    16, 32 and 64 components of a PCA fitted on each outer training fold.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from xgpoc.stack import fit as sf
from xgpoc.temporal import inputs as ti
from xgpoc.temporal.provenance import REFERENCE

K = 32
VARIANCE_KS = (16, 32, 64)
PCA_SEED = 0
CHUNK = 8192
ARMS = ("R", "S", "T")


def col(name: str, offset: int) -> str:
    return f"{name}@{offset}"


def pc(j: int) -> str:
    return f"pc{j:02d}"


def motion_kinds(window: ti.Window) -> list[str]:
    """The motion features of `window` (module docstring), in column order."""
    return ["vs_start", *(["vs_mid"] if window.middle != window.start else []),
            *(["slope"] if len(window.offsets) > 3 else [])]


def score_columns(name: str, window: ti.Window, k: int) -> list[str]:
    """pc01<name>@o_max ... pcK<name>@o_max."""
    return [col(f"{pc(j)}{name}", window.o_max) for j in range(1, k + 1)]


def missing_columns(window: ti.Window) -> list[str]:
    return [col("missing", o) for o in window.offsets]


def static_columns(window: ti.Window, k: int = K) -> list[str]:
    return [*score_columns("", window, k), *missing_columns(window)]


def motion_columns(window: ti.Window, k: int = K) -> list[str]:
    return [c for kind in motion_kinds(window) for c in score_columns(f"_{kind}", window, k)]


def arm_columns(window: ti.Window, k: int = K) -> dict[str, list[str]]:
    static = [REFERENCE, *static_columns(window, k)]
    return {"R": [REFERENCE], "S": static, "T": [*static, *motion_columns(window, k)]}


# ------------------------------------------------------------------ PCA
@dataclass(frozen=True)
class FramePCA:
    """A PCA of frame embeddings: the training mean, the components (k, dim) and their explained variance ratios."""

    mean: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray

    @classmethod
    def fit(cls, rows: np.ndarray, k: int, seed: int = PCA_SEED) -> "FramePCA":
        x = np.array(rows, dtype=np.float32, copy=True)        # a private copy: the PCA centres it in place
        pca = PCA(n_components=k, svd_solver="randomized", random_state=seed, copy=False).fit(x)
        return cls(pca.mean_.astype(np.float32), pca.components_.astype(np.float32),
                   pca.explained_variance_ratio_.astype(float))

    def transform(self, rows: np.ndarray) -> np.ndarray:
        """(n, k) scores, computed in chunks of `CHUNK` rows."""
        out = np.empty((len(rows), len(self.components)))
        for i in range(0, len(rows), CHUNK):
            out[i:i + CHUNK] = (np.asarray(rows[i:i + CHUNK], np.float32) - self.mean) @ self.components.T
        return out


def training_frames(position: np.ndarray, shots: np.ndarray) -> np.ndarray:
    """The embedding rows of the usable window frames of `shots` (each frame once)."""
    rows = position[shots]
    return np.unique(rows[rows >= 0])


def window_scores(pca: FramePCA, embeddings: np.ndarray, position: np.ndarray, take: np.ndarray, shots: np.ndarray) -> np.ndarray:
    """(len(shots), window offsets, k) PCA scores of every window frame of `shots`, unusable frames imputed by `take`."""
    filled = np.take_along_axis(position[shots], take[shots], axis=1)
    if (filled < 0).any():
        raise ValueError("an imputed frame has no embedding")
    rows, inverse = np.unique(filled, return_inverse=True)
    return pca.transform(embeddings[rows])[inverse.reshape(filled.shape)]


# ------------------------------------------------------------------ features
def motion_scores(scores: np.ndarray, window: ti.Window, kind: str) -> np.ndarray:
    """(shots, k) values of one motion feature from the window scores (shots, offsets, k)."""
    offsets = list(window.offsets)
    last = scores[:, -1, :]
    if kind == "vs_start":
        return last - scores[:, 0, :]
    if kind == "vs_mid":
        return last - scores[:, offsets.index(window.middle), :]
    if kind == "slope":
        t = np.asarray(offsets, float) / 1000.0
        t -= t.mean()
        return np.einsum("nmk,m->nk", scores, t) / float(t @ t)
    raise ValueError(f"unknown motion feature {kind!r}")


def feature_table(scores: np.ndarray, missing: np.ndarray, z: np.ndarray, window: ti.Window) -> pd.DataFrame:
    """Every column of arm T (z_g_combo first) from the window scores (shots, offsets, k) and missing indicators;
    each block is named by the helpers `arm_columns` uses, and the order is checked against it."""
    k = scores.shape[2]
    blocks = [([REFERENCE], np.asarray(z, float)[:, None]), (score_columns("", window, k), scores[:, -1, :]),
              (missing_columns(window), np.asarray(missing, float))]
    blocks += [(score_columns(f"_{kind}", window, k), motion_scores(scores, window, kind)) for kind in motion_kinds(window)]
    columns = [c for names, _ in blocks for c in names]
    if columns != arm_columns(window, k)["T"]:
        raise ValueError("the feature blocks are not in arm T's column order")
    return pd.DataFrame(np.hstack([values for _, values in blocks]), columns=columns)


def standardise(x: np.ndarray, n_train: int) -> tuple[np.ndarray, np.ndarray]:
    """(x[:n_train], x[n_train:]) standardised with the first `n_train` rows' mean and SD (scale 1 if constant)."""
    centre, scale = x[:n_train].mean(axis=0), x[:n_train].std(axis=0)
    x = (x - centre) / np.where(scale > 1e-12, scale, 1.0)
    return x[:n_train], x[n_train:]


@dataclass(frozen=True)
class DesignData:
    """What the designs are built from: the population's window, z, folds, usable-frame embedding positions and the
    imputation (`inputs.impute`), and the embedding rows."""

    window: ti.Window
    z: np.ndarray
    folds: np.ndarray
    position: np.ndarray
    embeddings: np.ndarray
    k: int = K

    @property
    def imputation(self) -> tuple[np.ndarray, np.ndarray]:
        return ti.impute(self.position >= 0)


def split_designs(data: DesignData, train: np.ndarray, test: np.ndarray,
                  imputation: tuple[np.ndarray, np.ndarray]) -> tuple[dict[str, tuple], FramePCA]:
    """arm -> (x_train, x_test) of one split, preprocessing fitted on `train` (module docstring), and the PCA."""
    take, missing = imputation
    pca = FramePCA.fit(data.embeddings[training_frames(data.position, train)], data.k)
    rows = np.r_[train, test]
    table = feature_table(window_scores(pca, data.embeddings, data.position, take, rows), missing[rows], data.z[rows],
                          data.window)
    columns = arm_columns(data.window, data.k)
    return {arm: standardise(table[columns[arm]].to_numpy(float), len(train)) for arm in columns}, pca


def fold_designs(data: DesignData) -> tuple[dict[str, list[sf.FoldDesign]], dict]:
    """(arm -> per outer fold in sorted order the `xgpoc.stack.fit.FoldDesign` with, for S and T, the inner designs of
    each other fold; per outer fold the explained variance ratio of the K components)."""
    imputation = data.imputation
    folds = data.folds
    designs: dict[str, list[sf.FoldDesign]] = {arm: [] for arm in ARMS}
    variance = {}
    for f in np.unique(folds):
        train, test = np.flatnonzero(folds != f), np.flatnonzero(folds == f)
        outer, pca = split_designs(data, train, test, imputation)
        variance[str(int(f))] = float(pca.explained_variance_ratio.sum())
        inner = {arm: [] for arm in ARMS}
        for j in np.unique(folds[train]):          # R's inner designs are built with the others but never used: R has no C
            itrain, itest = train[folds[train] != j], train[folds[train] == j]
            nested, _ = split_designs(data, itrain, itest, imputation)
            for arm in ARMS:
                inner[arm].append(sf.FoldDesign(int(j), itrain, itest, *nested[arm]))
        for arm in ARMS:
            nested = () if arm == "R" else tuple(inner[arm])
            designs[arm].append(sf.FoldDesign(int(f), train, test, *outer[arm], nested))
    return designs, {"explained_variance_k": variance, "k": data.k}


def explained_variance(data: DesignData, ks: tuple[int, ...] = VARIANCE_KS) -> dict:
    """Per outer training fold, the share of the total variance of its usable window frames explained by the first
    `ks` components, and the mean over folds."""
    out = {}
    for f in np.unique(data.folds):
        ratio = FramePCA.fit(data.embeddings[training_frames(data.position, np.flatnonzero(data.folds != f))],
                             max(ks)).explained_variance_ratio
        out[str(int(f))] = {str(k): float(ratio[:k].sum()) for k in ks}
    out["mean"] = {str(k): float(np.mean([out[str(int(f))][str(k)] for f in np.unique(data.folds)])) for k in ks}
    return out
