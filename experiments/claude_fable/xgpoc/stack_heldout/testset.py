"""The label-free test inputs of the xg-stack test and their checks (ticket 04, council C7 Q5).

Population (`heldout_shots`)
    every index v1.2 shot of the plan's evaluation and gate matches whose `is_penalty` is false (the A1 test's
    population, `research/xg_detector_test_rule.md` §3), read from label-free index columns, sorted by shot id.
Frame logit (`invert_platt`, `member_z`)
    the test z is the pre-Platt averaged G-combo logit, recovered from the deployed probability of
    `results/xgd_gcombo_test/predictions.csv` by inverting the frozen Platt of `results/xgv_g_combo/ensemble.json` at
    full precision: z = (logit(p) - b) / a. The prior correction is the identity there (oof_rate = pooled_rate) and is
    refused otherwise. Where member logits are stored (the gate shots' stored predictions and the unpaired evaluation
    shots' `unpaired_members.csv`), the same z is the mean over the ten members of logit(sigmoid(logit / T)).
E3 and SoccerNet inputs (`heldout_table`)
    the training population's builders, unchanged: `inputs.e3_table` (the exact replay check against the E3 cache, the
    frozen sanitation, the angle wrap with its count) and `inputs.flags_table` ("Goal" dropped at parse).
Checks (each returns a record with `passed`; a failed check stops the test)
    `z_inversion`      the round trip Platt(z) = p and the member-logit agreement, within their tolerances, and no p at
                       the probability clip (where the inversion is not exact)
    `z_distribution`   the test z's mean and SD over the training out-of-fold z's: each ratio inside [0.8, 1.25]
    `coverage`         3,819 shots, 172 matches (135 evaluation, 37 gate; 3,005 and 814 shots), unique shot ids,
                       every shot with its G-combo probability and flags, finite z, predictions inside (0, 1); the E3 rows'
                       exact coverage is enforced by `heldout_table`, which raises
    `ball_levels`      the t0 ball-source levels of the test against the training rows; a level never seen in training
                       enters no model (the zero-variance list is the training rows'); an unknown level stops earlier,
                       in `inputs.offset_row`
    `provenance`       no shot or match shared with the training population, every test shot of an evaluation or gate
                       match, the arms' columns equal the frozen C6 protocol's and carry no banned name, the flags
                       invariant to added "Goal" annotations and swapped shot types
"""

import contextlib
import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from xgpoc.detector import guard
from xgpoc.eval.calibration import EPS, logit, sigmoid
from xgpoc.stack import fit as sf
from xgpoc.stack import inputs as si
from xgpoc.stack import provenance as sp

ROLES = {"evaluation": 135, "gate": 37}
SHOTS_BY_ROLE = {"evaluation": 3005, "gate": 814}
INDEX_COLUMNS = ["shot_id", "game", "half", "position_ms", "is_penalty"]       # never goal, label or visibility
Z_RATIO = (0.8, 1.25)
ROUND_TRIP_TOLERANCE = 1e-12               # |Platt(z) - p|
MEMBER_TOLERANCE = 1e-6                    # |z - member z|: unpaired_members.csv stores logits with 9 significant digits
VENDOR_PATTERN = r"xgc_compare"            # xg-compare's table holds vendor xG and StatsBomb locations


@contextlib.contextmanager
def no_vendor_reads() -> Iterator[None]:
    """`guard.no_statsbomb_reads()` with xg-compare's outputs also refused: its table (results/xgc_compare) holds the
    vendor's xG and StatsBomb locations under a path the guard's pattern does not name."""
    original = guard.FORBIDDEN
    guard.FORBIDDEN = re.compile(f"{original.pattern}|{VENDOR_PATTERN}", original.flags)
    try:
        with guard.no_statsbomb_reads():
            yield
    finally:
        guard.FORBIDDEN = original


def heldout_shots(index: Path, plan: pd.DataFrame) -> pd.DataFrame:
    """game, half, position_ms, role of the population (module docstring), index shot_id sorted."""
    shots = pd.read_parquet(index, columns=INDEX_COLUMNS)
    shots["role"] = shots.game.map(plan.set_index("game").role)
    shots = shots[shots.role.isin(list(ROLES)) & ~shots.is_penalty.astype(bool)]
    return shots.drop(columns="is_penalty").set_index("shot_id").sort_index()


def invert_platt(p: np.ndarray, calibration: dict) -> np.ndarray:
    """z = (logit(p) - b) / a, the pre-Platt logit of a deployed probability (module docstring)."""
    if calibration["oof_rate"] != calibration["pooled_rate"]:
        raise ValueError("the prior correction is not the identity: the Platt inversion would not be exact")
    platt = calibration["platt"]
    return (np.log(p / (1 - p)) - platt["b"]) / platt["a"]


def platt(z: np.ndarray, calibration: dict) -> np.ndarray:
    return sigmoid(calibration["platt"]["a"] * np.asarray(z, float) + calibration["platt"]["b"])


def member_z(logits: pd.DataFrame, temperatures: dict[str, float]) -> pd.Series:
    """The mean over the members (columns, named as `temperatures`) of logit(sigmoid(logit / T)), per row: the z of
    `g_bundle.ensemble_probability`."""
    z = [logit(sigmoid(logits[name].to_numpy(float) / t)) for name, t in temperatures.items()]
    return pd.Series(np.mean(z, axis=0), index=logits.index)


def heldout_table(shots: pd.DataFrame, z: pd.Series, e3: Path, annotations: dict[str, list[dict]]) -> tuple[pd.DataFrame, dict]:
    """(the test table: index shot_id sorted; game, half, position_ms, role, z_g_combo, `inputs.E3_COLUMNS`,
    `inputs.S_FLAGS`; the angle-wrap counts per offset). Raises unless E3 covers the shots exactly, after
    `inputs.e3_table`'s exact replay check."""
    ids = sorted(shots.index)
    frames = pd.read_parquet(e3 / "frames.parquet", filters=[("shot_id", "in", ids)])
    features = pd.read_parquet(e3 / "features.parquet", filters=[("shot_id", "in", ids)])
    detector, wrapped = si.e3_table(frames, features)
    if not detector.index.equals(shots.index):
        raise ValueError("E3 does not cover the test population exactly")
    if not z.index.equals(shots.index):
        raise ValueError("the test z is not in the population's shot order")
    table = pd.concat([shots, z.rename(sf.Z), detector, si.flags_table(shots, annotations)], axis=1)
    return table, wrapped


def z_inversion(p: pd.Series, z: pd.Series, members: pd.Series, calibration: dict) -> dict:
    """The inversion check (module docstring) on the population's p and z (index shot_id) and the member z of the
    shots that have member logits."""
    round_trip = float(np.max(np.abs(platt(z.to_numpy(), calibration) - p.to_numpy())))
    common = members.index.intersection(z.index)
    gap = np.abs(z.loc[common].to_numpy() - members.loc[common].to_numpy())
    at_clip = int(((p <= EPS) | (p >= 1 - EPS)).sum())
    passed = round_trip <= ROUND_TRIP_TOLERANCE and float(gap.max()) <= MEMBER_TOLERANCE and at_clip == 0 and len(common) > 0
    return {"shots": int(len(z)), "round_trip_max_abs_dp": round_trip, "round_trip_tolerance": ROUND_TRIP_TOLERANCE,
            "member_shots": int(len(common)), "member_max_abs_dz": float(gap.max()), "member_mean_abs_dz": float(gap.mean()),
            "member_tolerance": MEMBER_TOLERANCE, "p_at_clip": at_clip, "passed": bool(passed)}


def z_distribution(z_test: np.ndarray, z_train: np.ndarray) -> dict:
    """The test z's mean and SD (ddof 1) over the training out-of-fold z's; stop when either ratio leaves `Z_RATIO`."""
    stats = {name: {"mean": float(np.mean(z)), "sd": float(np.std(z, ddof=1)), "n": int(len(z))}
             for name, z in (("test", z_test), ("training", z_train))}
    ratios = {m: stats["test"][m] / stats["training"][m] for m in ("mean", "sd")}
    lo, hi = Z_RATIO
    return stats | {"ratio": ratios, "bounds": list(Z_RATIO), "passed": all(lo <= r <= hi for r in ratios.values())}


def coverage(table: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    """The coverage check (module docstring) of the test table (index shot_id; game, role, the inputs) and its
    predictions (shot_id and probability columns)."""
    games = table.groupby("role").game.nunique().to_dict()
    shots = table.role.value_counts().to_dict()
    probabilities = predictions.filter(like="p_")
    checks = {
        "shots": len(table) == sum(SHOTS_BY_ROLE.values()),
        "matches": table.game.nunique() == sum(ROLES.values()),
        "matches_by_role": games == ROLES,
        "shots_by_role": shots == SHOTS_BY_ROLE,
        "unique_shot_ids": bool(table.index.is_unique and predictions.shot_id.is_unique),
        "predictions_align": predictions.shot_id.tolist() == table.index.tolist(),
        "z_finite": bool(np.isfinite(table[sf.Z].to_numpy(float)).all()),
        "flags": bool(table[si.S_FLAGS].notna().all().all()),
        "probabilities_open_unit": bool(((probabilities > 0) & (probabilities < 1)).all().all()),
    }
    return {"shots": len(table), "matches": int(table.game.nunique()), "matches_by_role": games, "shots_by_role": shots,
            "checks": checks, "passed": all(checks.values())}


def ball_levels(test: pd.DataFrame, training: pd.DataFrame, offset: int = 0) -> dict:
    """Shots per ball-source level at `offset` in the test and training rows, and the levels seen in the test but never
    in training (module docstring)."""
    columns = [si.col(f"ball_{level}", offset) for level in si.ball_levels(offset)]
    counts = {name: {c: int(frame[c].sum()) for c in columns} for name, frame in (("test", test), ("training", training))}
    unseen = [c for c in columns if counts["test"][c] > 0 and counts["training"][c] == 0]
    return {"offset": offset, "counts": counts, "unseen_in_training": unseen,
            "rule": "a level unseen in training enters no model (training zero-variance list); reported, not a stop",
            "passed": True}


def provenance(test: pd.DataFrame, training: pd.DataFrame, spec: sf.Spec, frozen: dict[str, list[str]],
               annotations: dict[str, list[dict]]) -> dict:
    """The provenance check (module docstring); `frozen` maps R, G and GS to the C6 protocol's design columns."""
    checks = {}

    def run(name: str, fn, *args) -> None:
        try:
            fn(*args)
            checks[name] = {"passed": True, "detail": ""}
        except (sp.ProvenanceError, ValueError) as e:
            checks[name] = {"passed": False, "detail": str(e)}

    def disjoint() -> None:
        if shared := set(test.index) & set(training.index):
            raise sp.ProvenanceError(f"{len(shared)} test shots are training shots")
        if shared := set(test.game) & set(training.game):
            raise sp.ProvenanceError(f"{len(shared)} test matches are training matches")
        if not test.role.isin(list(ROLES)).all():
            raise sp.ProvenanceError("a test shot is outside the evaluation and gate matches")

    run("disjoint_from_training", disjoint)
    for name, columns in frozen.items():
        run(f"columns:{name}", sp.check_columns, spec.arms[name], columns)
    run("flag_code", sp.check_flag_code, test[["game", "half", "position_ms"]], annotations)
    return {"checks": checks, "passed": all(c["passed"] for c in checks.values())}
