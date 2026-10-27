"""Replay of the xg-dino A/B/C read-out (ticket 02): same inputs, same outcomes, same seeds, then compare with the stored
results/xgdino_ablation/real files to 1e-9. Writes a new folder; the run script and the digest closure are untouched."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from xgpoc.detector.guard import no_statsbomb_reads
from xgpoc.dino import ablation as ab
from xgpoc.stack import inputs as si

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "results/xgdino_ablation/real"
TOL = 1e-9


def compare(a, b, path: str, worst: list) -> int:
    """Number of leaves compared; records the largest numeric difference and any mismatch in `worst`."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            worst.append(f"{path}: keys differ")
            return 0
        return sum(compare(a[k], b[k], f"{path}/{k}", worst) for k in a)
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return sum(compare(x, y, f"{path}[{i}]", worst) for i, (x, y) in enumerate(zip(a, b)))
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or a is None or b is None:
        if a != b:
            worst.append(f"{path}: {a!r} != {b!r}")
        return 1
    if (isinstance(a, float) and math.isnan(a)) and (isinstance(b, float) and math.isnan(b)):
        return 1
    d = abs(float(a) - float(b))
    worst.append(d) if d > TOL else None
    compare.max_diff = max(compare.max_diff, d if d == d else math.inf)
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--real", type=Path, default=REAL)
    args = parser.parse_args()
    compare.max_diff = 0.0
    inputs = ab.load_inputs()
    with no_statsbomb_reads():
        y = si.load_outcomes(inputs.primary).aligned(inputs.primary)
    result = ab.read_out(ab.prepare(inputs), y, log=lambda *_: None)
    problems: list = []
    stored = json.loads((args.real / "summary.json").read_text())
    n = compare(stored, json.loads(json.dumps(result["summary"], default=float)), "summary", problems)
    for name, frame in result["probabilities"].items():
        old = pd.read_csv(args.real / f"probabilities_{name.replace(':', '_')}.csv")
        assert list(old.columns) == list(frame.columns), name
        for col in frame.columns:
            if frame[col].dtype.kind in "fi":
                d = float(np.nanmax(np.abs(old[col].to_numpy(float) - frame[col].to_numpy(float))))
                compare.max_diff = max(compare.max_diff, d)
                n += len(frame)
                if d > TOL:
                    problems.append(f"probabilities_{name}.{col}: {d}")
    for name, frame, file in (("curves", result["curves"], "equivalence_curves.csv"),
                              ("reliability", result["reliability"], "reliability.csv")):
        old = pd.read_csv(args.real / file)
        for col in frame.columns:
            if frame[col].dtype.kind in "fi":
                d = float(np.nanmax(np.abs(old[col].to_numpy(float) - frame[col].to_numpy(float))))
                compare.max_diff = max(compare.max_diff, d)
                n += len(frame)
                if d > TOL:
                    problems.append(f"{file}.{col}: {d}")
    print(f"compared {n} values; max |difference| {compare.max_diff:.3g}; problems {problems[:10]}")
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
