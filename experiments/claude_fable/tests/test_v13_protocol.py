import pandas as pd
import pytest

from xgpoc.v13.protocol import ROLES, assign_roles


def cohort():
    return pd.DataFrame({"shot_id": [f"s{i}" for i in range(60)],
                         "game": [f"game{i // 2}" for i in range(60)],
                         "split": ["train"] * 40 + ["valid"] * 10 + ["test"] * 10,
                         "goal": [True, False] * 30})


def test_roles_are_disjoint_deterministic_and_label_independent():
    index = cohort()
    p = assign_roles(index)
    assert set(p.role) == set(ROLES)
    assert p.groupby("home").role.nunique().max() == 1
    assert not set(p.shot_id) & set(index[index.split == "test"].shot_id)
    changed = assign_roles(index.assign(goal=~index.goal).sample(frac=1, random_state=2))
    assert p[["shot_id", "role"]].equals(changed[["shot_id", "role"]])


def test_cross_split_game_is_rejected_before_test_exclusion():
    index = cohort()
    index.loc[55, "game"] = "game1"
    with pytest.raises(ValueError, match="overlapping"):
        assign_roles(index)
