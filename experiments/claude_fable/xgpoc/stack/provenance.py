"""Control 1 of xg-stack: the provenance assertions (ticket 01, council C5). Each check raises `ProvenanceError`;
`control` runs them all and records which passed.

    columns     every arm's design columns equal the frozen list, a fitted design matrix has exactly that width, and no
                input of an arm is constant (the zero-variance rule of council C6)
    names       no design column name contains a banned fragment (`BANNED`, case-insensitive)
    held_out    no shot and no match of the population is one of the 172 held-out matches of the xg-vendor split plan
                (the 135 evaluation and the 37 gate matches), by match and by shot id
    flag_code   the SoccerNet flags do not change when "Goal" annotations of both teams are added around every shot
                (from -20 s to +1 s) and every shot annotation's type is swapped (on target <-> off target), the own
                shot's included
"""

import numpy as np
import pandas as pd

from xgpoc.stack import fit as sf
from xgpoc.stack import inputs as si

BANNED = ("goal", "label", "outcome", "visibility", "statsbomb", "sb_")
HELD_OUT_ROLES = {"evaluation": 135, "gate": 37}
SWAPPED = {"Shots on target": "Shots off target", "Shots off target": "Shots on target"}
GOAL_OFFSETS_MS = (-20000, -3000, -300, 0, 300, 1000)


class ProvenanceError(AssertionError):
    """A provenance assertion of control 1 failed."""


def check_names(columns) -> None:
    for c in columns:
        if found := [f for f in BANNED if f in c.lower()]:
            raise ProvenanceError(f"design column {c!r} contains the banned fragment(s) {found}")


def check_columns(arm: sf.Arm, frozen: list[str], x: np.ndarray | None = None) -> None:
    """`arm`'s design columns are `frozen`, and the design matrix `x` (if given) has one column per name."""
    if arm.columns != list(frozen):
        raise ProvenanceError(f"arm {arm.name}: design columns differ from the frozen list")
    if x is not None and np.shape(x)[1] != len(frozen):
        raise ProvenanceError(f"arm {arm.name}: the design matrix has {np.shape(x)[1]} columns, the frozen list {len(frozen)}")
    check_names(frozen)


def held_out(plan: pd.DataFrame) -> frozenset[str]:
    """The 172 held-out matches of the plan (evaluation and gate), refused unless the role counts are the plan's."""
    counts = plan.role.value_counts()
    if any(int(counts.get(role, 0)) != n for role, n in HELD_OUT_ROLES.items()):
        raise ProvenanceError(f"the plan's held-out roles are {counts.to_dict()}, not {HELD_OUT_ROLES}")
    return frozenset(plan.game[plan.role.isin(list(HELD_OUT_ROLES))])


def check_held_out(games, shot_ids, plan: pd.DataFrame, index_shots: pd.DataFrame) -> None:
    """No population match and no population shot belongs to a held-out match; `index_shots` has the index's shot_id
    and game (no label column)."""
    held = held_out(plan)
    if overlap := sorted(set(games) & held):
        raise ProvenanceError(f"{len(overlap)} population matches are held out, e.g. {overlap[0]}")
    held_shots = set(index_shots.shot_id[index_shots.game.isin(held)])
    if overlap := sorted(set(shot_ids) & held_shots):
        raise ProvenanceError(f"{len(overlap)} population shots are from held-out matches, e.g. {overlap[0]}")


def perturbed(raw: list[dict], shots: pd.DataFrame) -> list[dict]:
    """`raw` with every shot type swapped and "Goal" annotations of both teams added around each of `shots`."""
    out = [dict(a, label=SWAPPED.get(a["label"], a["label"])) for a in raw]
    for r in shots.itertuples():
        out += [{"gameTime": f"{int(r.half)} - 00:00", "label": "Goal", "position": str(int(r.position_ms) + dt), "team": team,
                 "visibility": "visible"} for dt in GOAL_OFFSETS_MS for team in sorted(si.TEAMS)]
    return out


def check_flag_code(shots: pd.DataFrame, annotations: dict[str, list[dict]]) -> None:
    """The flags of `shots` (game, half, position_ms) are the same on the raw and the `perturbed` annotations."""
    changed = {g: perturbed(raw, shots[shots.game == g]) for g, raw in annotations.items()}
    if not si.flags_table(shots, annotations).equals(si.flags_table(shots, changed)):
        raise ProvenanceError("the SoccerNet flags depend on a Goal annotation or on a shot's label type")


def check_spec(spec: sf.Spec, table: pd.DataFrame, frozen_columns: dict[str, list[str]]) -> None:
    """Raise unless every arm of `spec` has the frozen columns, a design matrix of that width and no constant input,
    and the GBM inputs carry no banned name."""
    if set(frozen_columns) != set(spec.arms):
        raise ProvenanceError(f"frozen arms {sorted(frozen_columns)} are not the spec's {sorted(spec.arms)}")
    for name, arm in spec.arms.items():
        check_columns(arm, frozen_columns[name], sf.Preprocessor.fit(arm, table)(table.iloc[:5]))
        check_constant(table, arm)
    for columns in spec.gbm_columns.values():
        check_names(columns)


def check_constant(table: pd.DataFrame, arm: sf.Arm) -> None:
    try:
        sf.check_no_constant(table, arm)
    except ValueError as e:
        raise ProvenanceError(str(e)) from None


def control(population: si.Population, spec: sf.Spec, frozen_columns: dict[str, list[str]], plan: pd.DataFrame,
            index_shots: pd.DataFrame, annotations: dict[str, list[dict]]) -> dict:
    """Run every check (module docstring) and return {"passed", "checks": [{"check", "passed", "detail"}]}."""
    table = population.table
    checks = []

    def run(name: str, fn, *args) -> None:
        try:
            fn(*args)
            checks.append({"check": name, "passed": True, "detail": ""})
        except ProvenanceError as e:
            checks.append({"check": name, "passed": False, "detail": str(e)})

    if set(frozen_columns) != set(spec.arms):
        checks.append({"check": "columns", "passed": False, "detail": f"frozen arms {sorted(frozen_columns)}"})
    for name, arm in spec.arms.items():
        if name in frozen_columns:
            x = sf.Preprocessor.fit(arm, table)(table.iloc[:5])
            run(f"columns:{name}", check_columns, arm, frozen_columns[name], x)
            run(f"constant:{name}", check_constant, table, arm)
    for name in sf.GBM_ARMS:
        run(f"names:gbm_{name}", check_names, spec.gbm_columns[name])
    run("held_out", check_held_out, population.games, table.index, plan, index_shots)
    run("flag_code", check_flag_code, table[["game", "half", "position_ms"]], annotations)
    return {"passed": all(c["passed"] for c in checks), "checks": checks}
