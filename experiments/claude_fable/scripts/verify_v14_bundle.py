"""Load a v1.4 bundle in a fresh process and reproduce its blind evaluate predictions."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_v14 import construct, inputs, load_pool, predict
from xgpoc.eval.calibration import PlattScaler, route_penalties
from xgpoc.models.common import TrainConfig
from xgpoc.v13.protocol import digest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bundle", type=Path)
    ap.add_argument("--limit", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    bundle = torch.load(a.bundle, map_location="cpu", weights_only=False)
    identity = bundle["identity"]
    index = Path(identity["protocol"]["index"])
    if digest(index) != identity["protocol"]["index_sha256"]:
        raise ValueError("Source index has changed")
    root = a.bundle.parents[2]
    pool, _, boxes, sources = load_pool(index, root, Path(identity["sources"]["admissions"]["path"]))
    if sources != identity["sources"]:
        raise ValueError("Admissions or detection boxes differ from the trained model")
    df = pool[pool.role == "evaluate"].head(a.limit).reset_index(drop=True).assign(goal=False, on_target=0.)
    model = construct(bundle["spec"], pretrained=False)
    model.load_state_dict(bundle["state_dict"], strict=True)
    model.to(a.device)
    cfg = TrainConfig(**(identity["cfg"] | {"device": a.device}))
    z = predict(model, inputs(df, bundle["spec"], boxes), cfg)
    p = route_penalties(PlattScaler(**bundle["platt"])(z), df.is_penalty.to_numpy(), bundle["penalty_rate"])
    reference = pd.read_csv(a.bundle.parent / "evaluate_predictions.csv").set_index("shot_id").loc[df.shot_id]
    error = float(np.max(np.abs(p - reference.p.to_numpy())))
    if not np.isfinite(p).all() or error > 1e-5:
        raise ValueError(f"Replay differs by {error}")
    print(f"VERIFIED {len(df)} shots, maximum absolute probability error {error:.3g}")


if __name__ == "__main__":
    main()
