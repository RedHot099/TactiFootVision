import numpy as np
import pandas as pd
import pytest

from xgpoc.models.d_foundation import extract_patch_features
from xgpoc.models.f_temporal import TemporalFeatureDataset


def test_cached_features_cannot_bypass_post_shot_guard(tmp_path):
    path = tmp_path / "cached.npz"
    np.savez(path, shot_id=np.array(["s"]), feats=np.zeros((1, 2, 4)))
    df = pd.DataFrame({"shot_id": ["s"], "goal": [False]})
    with pytest.raises(ValueError, match="illegal"):
        extract_patch_features(df, path, path_col="frame_path_+400", device="cpu")


def test_temporal_dataset_rejects_future_even_when_features_exist():
    df = pd.DataFrame({"shot_id": ["s"], "goal": [False]})
    features = {0: {"s": np.zeros((2, 4))}, 400: {"s": np.zeros((2, 4))}}
    with pytest.raises(ValueError, match="offset"):
        TemporalFeatureDataset(df, features)


def test_cached_features_must_cover_requested_shots(tmp_path):
    path = tmp_path / "cached.npz"
    np.savez(path, shot_id=np.array(["other"]), feats=np.zeros((1, 2, 4)))
    with pytest.raises(ValueError, match="missing"):
        extract_patch_features(pd.DataFrame({"shot_id": ["s"]}), path, device="cpu")
