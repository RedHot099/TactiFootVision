"""One immutable development cohort; the historical test is excluded."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROLES = ("fit", "select", "calibrate", "evaluate")
HOME_RE = r"/\d{4}-\d{2}-\d{2} - \d{2}-\d{2} (.+?) \d+ - \d+ "


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(2**20):
            h.update(block)
    return h.hexdigest()


def assign_roles(index: pd.DataFrame, seed: int = 1300) -> pd.DataFrame:
    if index.shot_id.duplicated().any() or index.groupby("game").split.nunique().max() != 1:
        raise ValueError("Duplicate shots or overlapping games")
    pool = index[index.split != "test"].copy()
    if pool.shot_id.duplicated().any() or pool.groupby("game").split.nunique().max() != 1:
        raise ValueError("Duplicate shots or overlapping games")
    pool["home"] = pool.game.str.extract(HOME_RE)[0].fillna(pool.game)
    homes = np.random.default_rng(seed).permutation(sorted(pool.home.unique()))
    cuts = np.rint(np.array([0, .6, .7, .8, 1]) * len(homes)).astype(int)
    if (np.diff(cuts) == 0).any():
        raise ValueError("Too few home groups for four independent roles")
    mapping = {home: role for role, lo, hi in zip(ROLES, cuts[:-1], cuts[1:]) for home in homes[lo:hi]}
    pool["role"] = pool.home.map(mapping)
    return pool.sort_values("shot_id").reset_index(drop=True)


def prepare(index: Path, out: Path) -> tuple[pd.DataFrame, dict]:
    pool = assign_roles(pd.read_parquet(index))
    assignment = pool[["shot_id", "game", "home", "role"]].to_csv(index=False)
    manifest = {"version": 1, "index": str(index.resolve()), "index_sha256": digest(index), "seed": 1300,
                "assignment_sha256": hashlib.sha256(assignment.encode()).hexdigest(),
                "roles": list(ROLES), "historical_test_excluded": True,
                "counts": pool.groupby("role").agg(shots=("shot_id", "size"), games=("game", "nunique")).to_dict("index")}
    target = out / "protocol.json"
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise ValueError("Protocol changed; use a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_text(json.dumps(manifest, indent=2) + "\n")
        (out / "assignments.csv").write_text(assignment)
    if (out / "assignments.csv").read_text() != assignment:
        raise ValueError("Stored assignments differ from protocol")
    return pool, manifest
