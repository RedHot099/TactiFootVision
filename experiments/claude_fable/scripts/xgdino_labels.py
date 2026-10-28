"""Header labels from captions and the GPT sample, part 1 (ticket 10, steps 1-3; `xgpoc.dino.labels`, council C3.4).

    ./run.sh python scripts/xgdino_labels.py

Downloads the caption files (once), matches the 7,062 training shots to comments (the outcome is read afterwards, for
the coverage tables and the sample strata only), selects the stratified GPT sample and writes the anonymised frames,
the private key and the instructions to /mnt/storage/tactifoot_research/xg_dino/info/gpt_frames/ (never in git).
Small outputs: results/xgdino_info_labels/ (`complete.json` last). Step 4 (GPT answers, precision / recall) is a later part.
"""

import json
import time
from pathlib import Path

import pandas as pd
from xgpoc.bundle import sha256_file
from xgpoc.dino import ablation as ab
from xgpoc.dino import labels as lb
from xgpoc.paths import INDEX, LABELS_ROOT
from xgpoc.stack import inputs as si

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/xgdino_info_labels"
FRAMES = ROOT / "cache/xgd_full4_feet_t0/E3/frames.parquet"


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "complete.json").unlink(missing_ok=True)
    inputs = ab.load_inputs()
    pop = inputs.primary
    games = sorted(pop.table.game.unique())
    status = lb.download_captions(games)
    shots = pop.table[["game", "half", "position_ms", "fold"]].copy()
    shots["dist"] = pop.table["dist@0"]
    captions = lb.caption_labels(shots, LABELS_ROOT)
    shots["goal"] = si.load_outcomes(pop).aligned(pop)          # after the matching; coverage and strata only
    table = shots.join(captions.rename(columns={"label": "caption_label"}))

    coverage = (table.assign(outcome=table.goal.map({0: "no_goal", 1: "goal"}))
                .pivot_table(index="outcome", columns="caption_label", values="fold", aggfunc="size", fill_value=0)
                .reindex(columns=list(lb.LABELS), fill_value=0))
    coverage["shots"] = coverage.sum(axis=1)
    coverage.loc["all"] = coverage.sum()
    reasons = pd.crosstab(table.reason, table.goal.map({0: "no_goal", 1: "goal"}), margins=True)
    print(coverage, reasons, sep="\n\n")

    sample = lb.select_sample(table)
    ids = lb.anon_ids(sample.index)
    frames = pd.read_parquet(FRAMES, columns=["shot_id", "offset_ms", "frame_path"])
    n_files = lb.write_frames(sample.index, frames, ids)
    lb.KEY_DIR.mkdir(parents=True, exist_ok=True)
    key = pd.DataFrame({"anon_id": ids.loc[sample.index].to_numpy(), "shot_id": sample.index}).sort_values("anon_id")
    key.to_csv(lb.KEY_DIR / "key.csv", index=False)
    (lb.GPT_DIR / "instructions.md").write_text(lb.INSTRUCTIONS)
    (OUT / "instructions.md").write_text(lb.INSTRUCTIONS)

    table[["game", "half", "position_ms", "caption_label", "reason", "lag_ms", "n_candidates", "source"]].rename(
        columns={"caption_label": "header_caption"}).to_csv(OUT / "header_caption.csv")
    sample[["outcome", "caption_label", "band", "stratum", "weight"]].to_csv(OUT / "gpt_sample.csv")
    coverage.to_csv(OUT / "caption_coverage_by_outcome.csv")
    reasons.to_csv(OUT / "caption_match_reasons.csv")
    strata = sample.groupby(["outcome", "caption_label", "band"]).size().rename("n_sample").reset_index()
    strata.to_csv(OUT / "gpt_sample_strata.csv", index=False)
    record = {"ticket": "xg-dino 10 part 1", "shots": len(table), "matches": len(games),
              "caption_files": {"present": sum(v == "ok" for v in status.values()), "absent": sum(v == "absent" for v in status.values())},
              "flashscore_terms": "unchecked (SN-Labels card has no licence; comments scraped from flashscore)",
              "window_ms": list(lb.WINDOW_MS), "sample": {"size": len(sample), "seed": lb.SAMPLE_SEED, "files": n_files},
              "coverage_by_outcome": coverage.to_dict("index"), "sources": list(lb.SOURCES),
              "frames_dir": str(lb.FRAMES_DIR), "key": str(lb.KEY_DIR / "key.csv"), "instructions": str(lb.GPT_DIR / "instructions.md"),
              "inputs": {"index": sha256_file(INDEX), "oof": sha256_file(ab.FROZEN.oof)},
              "outputs": {p.name: sha256_file(p) for p in sorted(OUT.glob("*.csv"))}, "seconds": round(time.time() - started)}
    (OUT / "complete.json").write_text(json.dumps(record, indent=1, default=int) + "\n")
    print(strata.to_string())


if __name__ == "__main__":
    main()
