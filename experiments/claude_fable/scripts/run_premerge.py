"""Pre-merge GPU queue: overlay masking, flip TTA, and unfrozen-backbone fine-tuning.

    ./run.sh python scripts/run_premerge.py            # runs everything still missing
    ./run.sh python scripts/run_premerge.py --stages probe c_mask

Sequential on one GPU and resumable at (stage, seed) granularity: a finished run
leaves its metrics json and is skipped on the next invocation.

The fine-tuning depth sweep picks its winner on *validation* log-loss from one seed
per configuration and only then spends seeds on it, so the test set never influences
which architecture is reported.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

import run_poc as rp
from xgpoc.data.index import load_index
from xgpoc.models.common import TrainConfig

INDEX = "/home/kuba/projects/football_data/xg_shots_full/soccernet_720p/index_v1_1.parquet"
CACHE = Path("cache/full_v1")
OUT = Path("results/premerge")
# (backbone, unfrozen blocks) candidates for the depth sweep, cheapest first
G_SWEEP = [("dinov2_b", 12), ("dinov2_l", 4), ("dinov2_l", 12), ("dinov2_l", 24)]


def splits():
    df = load_index(Path(INDEX))
    return [df[df.split == s].reset_index(drop=True) for s in ("train", "valid", "test")]


def done(out: Path, name: str) -> bool:
    return (out / f"{name}_metrics.json").exists()


def stage(runner, out: Path, name: str, cfg: TrainConfig, data, **kw):
    if done(out, name):
        print(f"skip {out}/{name}")
        return json.loads((out / f"{name}_metrics.json").read_text())
    print(f"=== {out}/{name} ({time.strftime('%H:%M')}) ===", flush=True)
    t0 = time.time()
    m = runner(*data, CACHE, out, cfg, **kw)
    print(f"--- {name} done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return m


def best_valid_log_loss(out: Path, name: str) -> float:
    m = json.loads((out / f"{name}_metrics.json").read_text())
    return min(h["valid_log_loss"] for h in m["history"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    data = splits()
    print(f"train/valid/test = {[len(d) for d in data]}", flush=True)
    def want(stage_name: str) -> bool:
        return a.stages is None or stage_name in a.stages


    # 1. control probe: how much can a model score seeing only the broadcast graphics?
    if want("probe"):
        cfg = TrainConfig(epochs=8, max_minutes=25, batch_size=32, seed=0)
        stage(rp.run_c, OUT / "probe", "c_cnn_e2e_overlay_only", cfg, data,
              mask="overlay_only", tag="_overlay_only")

    # 2. C: baseline re-run under the current code (so every variant, including the
    #    baseline, has validation predictions available for honest ensembling), then
    #    masking and TTA isolated one at a time and combined
    for tag, kw in (("c_base", dict(tag="")),
                    ("c_mask", dict(mask="overlay", tag="_overlay")),
                    ("c_tta", dict(tta=True, tag="_tta")),
                    ("c_mask_tta", dict(mask="overlay", tta=True, tag="_overlay_tta"))):
        if not want(tag):
            continue
        for s in range(a.seeds):
            cfg = TrainConfig(epochs=8, max_minutes=25, batch_size=32, seed=s)
            stage(rp.run_c, OUT / tag / f"seed{s}", "c_cnn_e2e" + kw["tag"], cfg, data, **kw)

    # 3. same two questions for the frozen DINOv2-L head
    for tag, kw in (("dl_base", dict(tag="")),
                    ("dl_mask", dict(mask="overlay", tag="_overlay")),
                    ("dl_tta", dict(tta=True, tag="_tta")),
                    ("dl_mask_tta", dict(mask="overlay", tta=True, tag="_overlay_tta"))):
        if not want(tag):
            continue
        for s in range(a.seeds):
            cfg = TrainConfig(epochs=100, max_minutes=25, batch_size=32, seed=s)
            stage(rp.run_d, OUT / tag / f"seed{s}", f"d_frozen_dinov2_l{kw['tag']}", cfg, data,
                  backbone="dinov2_l", **kw)

    # 4. fine-tuning depth sweep, one seed per configuration
    if want("g_sweep"):
        for bb, nb in G_SWEEP:
            cfg = TrainConfig(epochs=10, max_minutes=70, batch_size=16, seed=0)
            stage(rp.run_g, OUT / "g_sweep" / f"{bb}_b{nb}", f"g_finetune_{bb}_b{nb}", cfg, data,
                  backbone=bb, n_blocks=nb)

    # 5. seeds + masking + TTA for whichever depth won on validation
    if want("g_best"):
        scored = {(bb, nb): best_valid_log_loss(OUT / "g_sweep" / f"{bb}_b{nb}", f"g_finetune_{bb}_b{nb}")
                  for bb, nb in G_SWEEP if done(OUT / "g_sweep" / f"{bb}_b{nb}", f"g_finetune_{bb}_b{nb}")}
        if not scored:
            print("g_sweep not run yet, skipping g_best")
        else:
            (bb, nb), ll = min(scored.items(), key=lambda kv: kv[1])
            print(f"g_sweep winner on valid log-loss: {bb}/{nb} blocks ({ll:.4f}); "
                  f"all = {[(k, round(v, 4)) for k, v in scored.items()]}", flush=True)
            (OUT / "g_sweep").mkdir(parents=True, exist_ok=True)
            (OUT / "g_sweep" / "selection.json").write_text(json.dumps(
                {"winner": {"backbone": bb, "n_blocks": nb}, "valid_log_loss": ll,
                 "all": {f"{k[0]}_b{k[1]}": v for k, v in scored.items()}}, indent=2))
            for s in range(a.seeds):
                cfg = TrainConfig(epochs=10, max_minutes=70, batch_size=16, seed=s)
                stage(rp.run_g, OUT / "g_best" / f"seed{s}", f"g_finetune_{bb}_b{nb}", cfg, data,
                      backbone=bb, n_blocks=nb)
                stage(rp.run_g, OUT / "g_best_mask_tta" / f"seed{s}", f"g_finetune_{bb}_b{nb}_overlay_tta",
                      cfg, data, backbone=bb, n_blocks=nb, mask="overlay", tta=True, tag="_overlay_tta")

    print(f"PREMERGE QUEUE DONE ({time.strftime('%H:%M')})", flush=True)
    rows = []
    for f in sorted(OUT.glob("**/*_metrics.json")):
        m = json.loads(f.read_text())
        rows.append({"run": str(f.parent.relative_to(OUT)), "model": m["model"], "auc": m["auc"],
                     "log_loss": m["log_loss"], "bss": m["brier_skill_score"], "ece": m["ece"]})
    if rows:
        print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
