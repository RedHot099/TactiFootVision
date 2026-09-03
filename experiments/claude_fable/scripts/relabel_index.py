"""Rebuild the trusted index with the v1.2 label rules without touching frames or clips.

    ./run.sh python scripts/relabel_index.py [--src index_v1_1.parquet] [--out index_v1_2.parquet]

Re-parses Labels-v2 with the nearest-shot goal attribution and the tight penalty window,
joins on (game, half, position_ms) - every row must match - and writes the new index with
`goal_v1_1` / `is_penalty_v1_1` kept for traceability. Prints a diff summary, stores it as
JSON next to the index and appends a changelog entry to the dataset card.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from xgpoc.data.soccernet_labels import parse_labels
from xgpoc.paths import DATASET_CARD, INDEX, INDEX_V1_1, LABELS_ROOT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(INDEX_V1_1))
    ap.add_argument("--out", default=str(INDEX))
    ap.add_argument("--card", default=str(DATASET_CARD))
    ap.add_argument("--no-card", action="store_true")
    a = ap.parse_args()

    df = pd.read_parquet(a.src)
    rows = []
    for game in sorted(df.game.unique()):
        for e in parse_labels(LABELS_ROOT / game / "Labels-v2.json"):
            if e.source == "shot":
                rows.append({"game": game, "half": e.half, "position_ms": e.position_ms,
                             "goal_new": e.goal, "is_penalty_new": e.is_penalty})
    # two shot annotations at the same millisecond exist (5 cases): merge them
    new = (pd.DataFrame(rows).groupby(["game", "half", "position_ms"], as_index=False)
             .agg(goal_new=("goal_new", "max"), is_penalty_new=("is_penalty_new", "max")))
    m = df.merge(new, on=["game", "half", "position_ms"], how="left", validate="one_to_one")
    if m.goal_new.isna().any():
        raise SystemExit(f"{int(m.goal_new.isna().sum())} index rows have no matching shot annotation")

    out = m.copy()
    out["goal_v1_1"], out["is_penalty_v1_1"] = out["goal"].astype(bool), out["is_penalty"].astype(bool)
    out["goal"], out["is_penalty"] = out.goal_new.astype(bool), out.is_penalty_new.astype(bool)
    out = out.drop(columns=["goal_new", "is_penalty_new"])

    flipped = out.goal != out.goal_v1_1
    pen_changed = out.is_penalty != out.is_penalty_v1_1
    rep = {
        "source": a.src, "rows": int(len(out)),
        "goals_v1_1": int(out.goal_v1_1.sum()), "goals_v1_2": int(out.goal.sum()),
        "labels_flipped": int(flipped.sum()),
        "flipped_to_goal": int((flipped & out.goal).sum()), "flipped_to_miss": int((flipped & ~out.goal).sum()),
        "flipped_per_split": out[flipped].groupby("split").size().to_dict(),
        "off_target_goals_v1_1": int(((out.label == "Shots off target") & out.goal_v1_1).sum()),
        "off_target_goals_v1_2": int(((out.label == "Shots off target") & out.goal).sum()),
        "penalties_v1_1": int(out.is_penalty_v1_1.sum()), "penalties_v1_2": int(out.is_penalty.sum()),
        "penalty_flags_changed": int(pen_changed.sum()),
        "penalty_conversion_v1_2": out[out.is_penalty].groupby("split").goal.agg(["size", "mean"]).round(3).to_dict("index"),
        "goal_rate_per_split": out.groupby("split").goal.mean().round(4).to_dict(),
    }
    out.to_parquet(a.out, index=False)
    Path(a.out).with_suffix(".relabel.json").write_text(json.dumps(rep, indent=2, default=float))
    print(json.dumps(rep, indent=2, default=float))
    print("->", a.out)

    if not a.no_card and Path(a.card).exists():
        entry = (f"\n## Changelog v1.2 ({pd.Timestamp.today():%Y-%m-%d})\n\n"
                 f"- `soccernet_720p/index_v1_2.parquet`: te same {len(out)} strzały co v1.1, nowe reguły etykiet.\n"
                 f"- Gol przypisany do **najbliższego** poprzedzającego strzału tej samej drużyny (v1.1: do "
                 f"najwcześniejszego w oknie 6 s, co odwracało etykiety w łańcuchach dobitek). Zmienionych etykiet: "
                 f"{rep['labels_flipped']} ({rep['flipped_to_goal']} → gol, {rep['flipped_to_miss']} → nie-gol); "
                 f"per split {rep['flipped_per_split']}. „Shots off target” z golem: {rep['off_target_goals_v1_1']} → "
                 f"{rep['off_target_goals_v1_2']}.\n"
                 f"- `is_penalty`: flaga tylko dla strzału najbliższego etykiecie `Penalty` w oknie [−1 s, +2 s] "
                 f"(v1.1: każdy strzał w 60 s). Karnych: {rep['penalties_v1_1']} → {rep['penalties_v1_2']}.\n"
                 f"- Kolumny `goal_v1_1`, `is_penalty_v1_1` zachowane do śledzenia. Wyniki liczone na v1.1 i v1.2 "
                 f"nie są porównywalne (test: {rep['flipped_per_split'].get('test', 0)} zmienionych etykiet).\n")
        with Path(a.card).open("a") as f:
            f.write(entry)
        print("dataset card updated:", a.card)


if __name__ == "__main__":
    main()
