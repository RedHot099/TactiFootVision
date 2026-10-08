"""Write the split plan of the leave-one-league-out transfer test.

    ./run.sh python scripts/make_transfer_plan.py

Outputs in --out (default research/xgc_transfer_plan, committed with the protocol):
  plan.csv       league, half_set, arm, role, source, game: the whole plan (xgpoc.transfer.plan)
  summary.csv    matches, held-out-league matches, shots and goals per league/half-set/arm/role
  pools.csv      per held-out league: size, clubs, Champions League matches removed, pool size
  manifest.json  index hash, seed, sizes, source hashes and the plan's sha256

A different plan is never written over an existing one without --replace: once the
protocol is approved, the plan it names is fixed.
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from xgpoc.paths import INDEX
from xgpoc.transfer import plan as tp
from xgpoc.v13.protocol import digest

OUT = Path("research/xgc_transfer_plan")


def write_plan_files(out: Path, files: dict[str, str], manifest: dict, replace: bool = False) -> None:
    """Write `files` (name -> text) and manifest.json into `out`. A manifest naming a different
    plan_sha256 is kept unless `replace`; the same plan with new source hashes is rewritten."""
    target = out / "manifest.json"
    if target.exists() and not replace:
        old = json.loads(target.read_text()).get("plan_sha256")
        if old != manifest["plan_sha256"]:
            raise SystemExit(f"{target} holds plan {old}, this run gives {manifest['plan_sha256']}; "
                             "pass --replace to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (out / name).write_text(text)
    target.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=INDEX)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--replace", action="store_true", help="overwrite an existing, different plan")
    a = ap.parse_args()

    index = pd.read_parquet(a.index)
    plan = tp.build_plan(index, seed=a.seed)
    plan_csv = plan.to_csv(index=False)
    summary, pools = tp.summarise(plan, index), tp.pool_summary(index, aliases=tp.CLUB_ALIASES)
    manifest = {
        "index": str(a.index.resolve()), "index_sha256": digest(a.index), "seed": a.seed,
        "n_matches": tp.N_MATCHES, "role_weights": tp.ROLE_WEIGHTS, "leagues": list(tp.DOMESTIC_LEAGUES),
        "club_aliases": tp.CLUB_ALIASES, "reviewed_distinct": sorted(sorted(p) for p in tp.REVIEWED_DISTINCT),
        "sources": {p: digest(Path(p)) for p in ("xgpoc/transfer/plan.py", "scripts/make_transfer_plan.py")},
        "plan_rows": len(plan), "plan_sha256": hashlib.sha256(plan_csv.encode()).hexdigest(),
    }
    write_plan_files(a.out, {"plan.csv": plan_csv, "summary.csv": summary.to_csv(index=False),
                             "pools.csv": pools.to_csv(index=False)}, manifest, a.replace)

    pd.set_option("display.width", 200)
    print(pools.to_string(index=False))
    wide = summary.pivot_table(index=["league", "half_set", "arm"], columns="role", values="matches", sort=False).astype(int)
    print(wide[list(tp.ROLES)].to_string())
    print(f"plan {manifest['plan_sha256']} ({len(plan)} rows) -> {a.out}")


if __name__ == "__main__":
    main()
