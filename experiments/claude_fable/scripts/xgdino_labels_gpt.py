"""Header labels, part 2 (ticket 10, step 4): GPT answers joined to shots, caption precision / recall against GPT.

    ./run.sh python scripts/xgdino_labels_gpt.py

Reads the answers (`gpt_frames/answers/part*.csv`), joins them through the private key, writes `header_gpt.csv`
(source="gpt"; `unclear` is missing) and `caption_vs_gpt.json` / `.csv` (weighted by the stratum weights of the sample,
overall and by outcome) to results/xgdino_info_labels/, then rewrites `complete.json` last. GPT labels are a noisy
measurement, not ground truth. No StatsBomb read; the outcome is the training label of the sample file.
"""

import json
from pathlib import Path

import pandas as pd
from xgpoc.bundle import sha256_file
from xgpoc.dino import labels as lb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/xgdino_info_labels"


def flat(prefix: str, node: dict, rows: list) -> None:
    for k, v in node.items():
        if isinstance(v, dict):
            flat(f"{prefix}.{k}" if prefix else k, v, rows)
        else:
            rows.append({"metric": f"{prefix}.{k}" if prefix else k, "value": v})


def main() -> None:
    record = json.loads((OUT / "complete.json").read_text())
    (OUT / "complete.json").unlink()
    sample = pd.read_csv(OUT / "gpt_sample.csv", index_col="shot_id")
    key = pd.read_csv(lb.KEY_DIR / "key.csv")
    paths = sorted(lb.ANSWER_DIR.glob("part*.csv"))
    answers = lb.read_answers(paths)
    gpt = lb.gpt_labels(answers, key, sample)
    gpt.to_csv(OUT / "header_gpt.csv")
    d = sample.join(gpt)
    result = lb.agreement(d)
    (OUT / "caption_vs_gpt.json").write_text(json.dumps(result, indent=1) + "\n")
    rows: list = []
    flat("", {k: v for k, v in result.items() if k != "note"}, rows)
    pd.DataFrame(rows).to_csv(OUT / "caption_vs_gpt.csv", index=False)
    print(json.dumps({k: result[k] for k in ("answers_by_outcome",)}, indent=1))
    for name in ("overall", "goal", "no_goal"):
        b = result[name]
        print(name, "| caption header precision", round(b["caption_header"]["precision"], 3), "recall", round(b["gpt_head"]["recall"], 3),
              "| agreement", round(b["agreement_on_caption_labelled"]["weighted"], 3),
              "| unknown->head", round(b["caption_unknown"]["gpt_head_share"], 3))
    record["part2"] = {"answers": [str(p) for p in paths], "answers_sha256": {p.name: sha256_file(p) for p in paths},
                       "answer_counts": {k: int(v) for k, v in answers.value_counts().items()}, "note": lb.GPT_NOTE,
                       "outputs": {p.name: sha256_file(p) for p in (OUT / "header_gpt.csv", OUT / "caption_vs_gpt.json", OUT / "caption_vs_gpt.csv")}}
    (OUT / "complete.json").write_text(json.dumps(record, indent=1, default=int) + "\n")


if __name__ == "__main__":
    main()
