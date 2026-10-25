"""Synthetic inputs of the xg-temporal tests: frame embeddings with a static part and a motion part, matches and folds
like the xgv split, and the frame rows of an embedding source."""

import numpy as np
import pandas as pd
from xgpoc.temporal import features as tf
from xgpoc.temporal import inputs as ti

MOTION_SCALE = 6.0


def synthetic_inputs(window: ti.Window | None = None, games: int = 40, shots_per_game: int = 50, dim: int = 12, k: int = 4,
                     leading_gaps: int = 100, seed: int = 0) -> tuple[tf.DesignData, np.ndarray]:
    """(design inputs, the per-shot motion speed v). Frame j of shot i is static_i + v_i (t_j - t_o_max) MOTION_SCALE
    along the first axis plus noise, so the o_max frame carries nothing of v; the first `leading_gaps` shots lose
    their first two window frames (a cut)."""
    window = ti.Window() if window is None else window
    rng = np.random.default_rng(seed)
    n, m = games * shots_per_game, len(window.offsets)
    game = np.repeat(np.arange(games), shots_per_game)
    t = (np.asarray(window.offsets, float) - window.o_max) / 1000.0
    v = rng.normal(size=n)
    frames = rng.normal(size=(n, 1, dim)) + 0.3 * rng.normal(size=(n, m, dim))
    frames[:, :, 0] += MOTION_SCALE * v[:, None] * t[None, :]
    position = np.arange(n * m).reshape(n, m)
    position[:leading_gaps, :2] = -1
    data = tf.DesignData(window, rng.normal(-2.0, 1.0, n), game % 5, position, frames.reshape(n * m, dim).astype(np.float16), k)
    return data, v


def source_frames(rows: list[tuple]) -> pd.DataFrame:
    """Frame rows (shot_id, game, offset_ms, frame_status, cut_between_anchor, embedding_ref, fold_source) of one
    encoder."""
    frames = pd.DataFrame(rows, columns=["shot_id", "game", "offset_ms", "frame_status", "cut_between_anchor",
                                         "embedding_ref", "fold_source"])
    frames["encoder"] = "g_combo"
    frames["cut_between_anchor"] = frames.cut_between_anchor.astype("boolean")
    return frames


def base_table(rows: list[tuple]) -> pd.DataFrame:
    """Base shots (shot_id, game, fold, z_g_combo), index shot_id."""
    return pd.DataFrame(rows, columns=["shot_id", "game", "fold", "z_g_combo"]).set_index("shot_id").sort_index()


def synthetic_population(window: ti.Window, games: int = 20, shots_per_game: int = 8, dim: int = 80,
                         seed: int = 0) -> tuple[ti.Population, np.ndarray, pd.DataFrame]:
    """(population of `window` over synthetic training matches with every grid frame usable, the embedding array its
    refs point into, the plan)."""
    rng = np.random.default_rng(seed)
    names = [f"G{g:02d}" for g in range(games)]
    base = base_table([(f"{names[g]}_{i}", names[g], g % 5, float(rng.normal(-2, 1)))
                       for g in range(games) for i in range(shots_per_game)])
    rows = [(shot, game, o, "ok", False, n * len(ti.GRID) + j, fold)
            for n, (shot, game, fold) in enumerate(zip(base.index, base.game, base.fold, strict=True))
            for j, o in enumerate(ti.GRID)]
    population = ti.population_from(base, source_frames(rows), window, ti.SOURCES["g_xgv"])
    embeddings = rng.normal(size=(len(rows), dim)).astype(np.float16)
    plan = pd.DataFrame({"game": names, "role": "training", "fold": [g % 5 for g in range(games)]})
    return population, embeddings, plan
