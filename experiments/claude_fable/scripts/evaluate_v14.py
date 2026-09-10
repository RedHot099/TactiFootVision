"""v1.4 screening, freeze and development evaluation: evaluate_v13 machinery, v1.4 pairs.

The v1.3 analysis script is left untouched because its hash is part of the frozen
v13_final analysis; this wrapper only swaps the predeclared comparisons and records its
own hash alongside."""

import argparse
import json
from pathlib import Path

import evaluate_v13
from xgpoc.v13.protocol import digest, prepare

PAIRS = [("g_views", "g_base"), ("g_views_control", "g_base"), ("g_views", "g_views_control")]
VARIANTS = ["g_base", "g_views", "g_views_control"]


def analysis_spec(n_boot):
    return {"n_boot": n_boot, "seed": 0, "sampling_unit": "home group", "pairs": [list(p) for p in PAIRS],
            "scope": "development evaluation; the cohort was already opened in v1.3",
            "source_sha256": {str(p): digest(p) for p in
                [Path(__file__), Path("scripts/evaluate_v13.py"), Path("scripts/significance.py"),
                 Path("xgpoc/eval/metrics.py")]}}


evaluate_v13.PAIRS = PAIRS
evaluate_v13.analysis_spec = analysis_spec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("results/v14"))
    ap.add_argument("--mode", choices=["screen", "freeze", "evaluate"], default="screen")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    manifest = json.loads((a.root / "protocol.json").read_text())
    pool, _ = prepare(Path(manifest["index"]), a.root)
    if a.mode == "evaluate":
        evaluate_v13.report(a.root, pool, a.n_boot)
    else:
        found = evaluate_v13.runs(a.root, VARIANTS)
        if a.mode == "screen":
            evaluate_v13.screen(a.root, found)
        else:
            evaluate_v13.freeze(a.root, found, pool, a.n_boot)


if __name__ == "__main__":
    main()
