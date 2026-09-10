"""Screen on selection only, freeze calibrated models, then open development evaluation."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from significance import paired_bootstrap
from xgpoc.eval.calibration import PlattScaler, penalty_rate, route_penalties
from xgpoc.eval.metrics import evaluate
from xgpoc.v13.protocol import digest, prepare
from xgpoc.v13.inputs import VARIANTS

PAIRS = [(v, "full") for v in ("roi", "control", "time_mean", "time_gru", "geometry")]
PAIRS += [("roi", "control"), ("time_gru", "time_mean"), ("g_pre400", "g_base"),
          ("time_mean", "time_mean_t0"), ("time_gru", "time_gru_t0"), ("geometry", "geometry_zero")]


def analysis_spec(n_boot):
    return {"n_boot": n_boot, "seed": 0, "sampling_unit": "home group", "pairs": [list(p) for p in PAIRS],
            "source_sha256": {str(p): digest(p) for p in
                [Path(__file__), Path("scripts/significance.py"), Path("xgpoc/eval/metrics.py")]}}


def save_csv(table, path):
    temporary = path.with_suffix(".tmp")
    table.to_csv(temporary, index=False)
    os.replace(temporary, path)


def runs(root, variants):
    if (root / "invalidated.json").exists():
        raise ValueError("These results were invalidated; inspect invalidated.json")
    protocol = json.loads((root / "protocol.json").read_text())
    suite_identity = json.loads((root / "suite_identity.json").read_text())
    found = {}
    for variant in variants:
        files = sorted((root / variant).glob("seed*/complete.json"))
        if not files:
            raise ValueError(f"No completed runs for {variant}")
        for f in files:
            m = json.loads(f.read_text())
            identity = m["identity"]
            if identity["protocol"] != protocol or identity["spec"]["variant"] != variant:
                raise ValueError(f"Run belongs to another protocol or variant: {f}")
            for key in ("code_sha256", "pixels"):
                if identity[key] != suite_identity[key]:
                    raise ValueError(f"Run differs from suite {key}: {f}")
            if f.parent.name != f"seed{identity['cfg']['seed']}":
                raise ValueError(f"Run seed differs from directory: {f}")
            if m["identity"]["smoke"]:
                raise ValueError("Smoke results cannot enter the quality report")
            for name, checksum in m["files"].items():
                if digest(f.parent / name) != checksum:
                    raise ValueError(f"Modified {f.parent / name}")
        found[variant] = files
        identities = [json.loads(f.read_text())["identity"] for f in files]
        reference = identities[0]
        for identity in identities[1:]:
            for key in ("spec", "code_sha256", "input_fingerprint"):
                if identity.get(key) != reference.get(key):
                    raise ValueError(f"Inconsistent {key} across seeds for {variant}")
            cfg = {k: v for k, v in identity["cfg"].items() if k != "seed"}
            expected = {k: v for k, v in reference["cfg"].items() if k != "seed"}
            if cfg != expected:
                raise ValueError(f"Inconsistent training settings for {variant}")
    return found


def screen(root, found):
    rows = []
    for variant, files in found.items():
        ll = [json.loads((f.parent / "training.json").read_text())["selection_log_loss"] for f in files]
        rows.append({"variant": variant, "seeds": len(ll), "selection_ll": float(np.mean(ll)),
                     "selection_ll_std": float(np.std(ll)), "scope": "selection only; optimistic model selection score"})
    table = pd.DataFrame(rows).sort_values("selection_ll")
    table.to_csv(root / "screen.csv", index=False)
    print(table.to_string(index=False))


def mean_logits(files, role, ids):
    values = []
    for f in files:
        p = pd.read_csv(f.parent / f"{role}_predictions.csv")
        if p.shot_id.duplicated().any() or set(p.shot_id) != set(ids):
            raise ValueError(f"{f} changed {role} cohort")
        if role == "evaluate" and "goal" in p:
            raise ValueError("Evaluation predictions should be blind")
        values.append(p.set_index("shot_id").loc[ids, "logit"].to_numpy())
    return np.mean(values, axis=0)


def freeze(root, found, pool, n_boot=2000):
    if (root / "frozen.json").exists() or (root / "evaluated.json").exists():
        raise ValueError("Run is already frozen or evaluated")
    suite = json.loads((root / "suite.json").read_text())
    if set(found) != set(suite):
        raise ValueError("Freeze must include the complete predeclared suite")
    for variant, files in found.items():
        actual = sorted(json.loads(f.read_text())["identity"]["cfg"]["seed"] for f in files)
        if actual != suite[variant]:
            raise ValueError(f"Missing or duplicate seeds for {variant}: {actual}")
    cal = pool[pool.role == "calibrate"]
    blind = pool[pool.role == "evaluate"].drop(columns="goal")
    pen = pool[(pool.role == "fit") & pool.is_penalty]
    pp = penalty_rate(int(pen.goal.sum()), len(pen))
    record = {"variants": {}, "protocol_sha256": digest(root / "protocol.json"), "analysis": analysis_spec(n_boot)}
    for variant, files in found.items():
        zc = mean_logits(files, "calibrate", cal.shot_id)
        ze = mean_logits(files, "evaluate", blind.shot_id)
        nonpen = ~cal.is_penalty.to_numpy()
        scaler = PlattScaler().fit(zc[nonpen], cal.goal.to_numpy()[nonpen])
        p = route_penalties(scaler(ze), blind.is_penalty.to_numpy(), pp)
        target = root / f"blind_{variant}.csv"
        pd.DataFrame({"shot_id": blind.shot_id, "p": p}).to_csv(target, index=False)
        record["variants"][variant] = {"members": {str(f.parent / "bundle.pt"): digest(f.parent / "bundle.pt") for f in files},
                                       "platt": {"a": scaler.a, "b": scaler.b}, "penalty_rate": pp,
                                       "prediction_file": target.name, "prediction_sha256": digest(target)}
    (root / "frozen.json").write_text(json.dumps(record, indent=2) + "\n")
    print("Frozen calibrated seed ensembles without scoring evaluation labels")


def report(root, pool, n_boot):
    frozen_path = root / "frozen.json"
    frozen = json.loads(frozen_path.read_text())
    if frozen["analysis"] != analysis_spec(n_boot):
        raise ValueError("Analysis differs from the specification frozen before evaluation")
    if frozen["protocol_sha256"] != digest(root / "protocol.json"):
        raise ValueError("Protocol changed after freeze")
    for v in frozen["variants"].values():
        if digest(root / v["prediction_file"]) != v["prediction_sha256"]:
            raise ValueError("Predictions changed after freeze")
        for filename, expected in v["members"].items():
            if digest(Path(filename)) != expected:
                raise ValueError("Model changed after freeze")
    marker = {"frozen_sha256": digest(frozen_path), "analysis": frozen["analysis"],
              "note": "Development evaluation has been opened; further training prohibited"}
    marked = root / "evaluated.json"
    if marked.exists() and json.loads(marked.read_text()) != marker:
        raise ValueError("Evaluation snapshot changed")
    marked.write_text(json.dumps(marker, indent=2) + "\n")
    completed = root / "analysis_complete.json"
    if completed.exists():
        saved = json.loads(completed.read_text())
        if saved["evaluated_sha256"] != digest(marked):
            raise ValueError("Completed analysis identity changed")
        for name, checksum in saved["files"].items():
            if digest(root / name) != checksum:
                raise ValueError("Completed analysis outputs changed")
        print("Verified existing immutable evaluation")
        return
    labels = pool[pool.role == "evaluate"][["shot_id", "game", "home", "goal", "is_penalty"]]
    predictions, rows = {}, []
    for variant, v in frozen["variants"].items():
        p = pd.read_csv(root / v["prediction_file"]).merge(labels, on="shot_id", validate="one_to_one")
        if len(p) != len(labels):
            raise ValueError("Evaluation coverage differs")
        predictions[variant] = p
        for group, data in (("all", p), ("non_penalty", p[~p.is_penalty])):
            rows.append({"variant": variant, "group": group, "games": data.game.nunique(),
                         **evaluate(data.goal, data.p), "mean_xg": float(data.p.mean())})
    save_csv(pd.DataFrame(rows), root / "evaluation.csv")
    comparisons = []
    for first, second in PAIRS:
        if first not in predictions or second not in predictions:
            continue
        result = paired_bootstrap(predictions[first][["shot_id", "p"]], predictions[second][["shot_id", "p"]],
                                  labels[["shot_id", "home", "goal"]].rename(columns={"home": "game"}), n_boot=n_boot)
        result["sampling_unit"] = "home group"
        result["home_groups"] = result.pop("games")
        comparisons.append({"first": first, "second": second, **result})
        print(first, second, "delta LL", result["d_log_loss"], "CI", result["d_log_loss_lo"], result["d_log_loss_hi"], flush=True)
    save_csv(pd.DataFrame(comparisons), root / "paired_evaluation.csv")
    if frozen["analysis"] != analysis_spec(n_boot):
        raise ValueError("Analysis source changed during evaluation")
    completed.write_text(json.dumps({"evaluated_sha256": digest(marked), "files": {
        name: digest(root / name) for name in ("evaluation.csv", "paired_evaluation.csv")}}, indent=2) + "\n")
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("results/v13"))
    ap.add_argument("--variants", nargs="+", default=["g_base", "g_pre400", *VARIANTS])
    ap.add_argument("--mode", choices=["screen", "freeze", "evaluate"], default="screen")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    manifest = json.loads((a.root / "protocol.json").read_text())
    pool, _ = prepare(Path(manifest["index"]), a.root)
    if a.mode == "evaluate":
        report(a.root, pool, a.n_boot)
    else:
        found = runs(a.root, a.variants)
        if a.mode == "screen":
            screen(a.root, found)
        else:
            freeze(a.root, found, pool, a.n_boot)


if __name__ == "__main__":
    main()
