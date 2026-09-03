"""The ensemble builder must treat seed directories and fold directories differently by
construction: test files are averaged per shot, out-of-fold validation files are
concatenated - both fall out of averaging per shot_id."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def _write(d: Path, name: str, shots, ps):
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"shot_id": shots, "goal": [0] * len(shots), "p": ps}).to_csv(d / name, index=False)


def test_collect_averages_test_and_concatenates_oof(tmp_path):
    from build_ensemble import collect

    root = tmp_path / "res"
    # two folds x one seed: disjoint OOF shots, identical test shots
    _write(root / "m" / "fold0" / "seed0", "m_valid_predictions.csv", ["a", "b"], [0.2, 0.4])
    _write(root / "m" / "fold1" / "seed0", "m_valid_predictions.csv", ["c"], [0.6])
    _write(root / "m" / "fold0" / "seed0", "m_test_predictions.csv", ["t1", "t2"], [0.1, 0.5])
    _write(root / "m" / "fold1" / "seed0", "m_test_predictions.csv", ["t1", "t2"], [0.3, 0.5])
    # plain seed layout for a second variant
    _write(root / "n" / "seed0", "n_valid_predictions.csv", ["a", "b", "c"], [0.1, 0.1, 0.1])
    _write(root / "n" / "seed1", "n_valid_predictions.csv", ["a", "b", "c"], [0.3, 0.3, 0.3])
    _write(root / "n" / "seed0", "n_test_predictions.csv", ["t1", "t2"], [0.2, 0.2])
    out = collect(root)
    assert set(out) == {"m", "n"}
    oof = out["m"]["valid"].set_index("shot_id")
    assert sorted(oof.index) == ["a", "b", "c"] and np.allclose(oof.loc[["a", "b", "c"], "p"], [0.2, 0.4, 0.6])
    te = out["m"]["test"].set_index("shot_id")
    z = np.mean([np.log(0.1 / 0.9), np.log(0.3 / 0.7)])
    assert abs(te.loc["t1", "p"] - 1 / (1 + np.exp(-z))) < 1e-9 and abs(te.loc["t2", "p"] - 0.5) < 1e-9
    assert int(te.loc["t1", "seeds"]) == 2
    n_va = out["n"]["valid"].set_index("shot_id")
    zn = np.mean([np.log(0.1 / 0.9), np.log(0.3 / 0.7)])
    assert np.allclose(n_va.p, 1 / (1 + np.exp(-zn))) and int(n_va.seeds.iloc[0]) == 2
