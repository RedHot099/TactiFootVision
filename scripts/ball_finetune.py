"""ball-detection 05: the per-epoch table of the ball fine-tune and its control, from the scorecards.

    uv run python scripts/ball_finetune.py table     # ball/05/{A,B}_e{0,1} and B_e{0,1}_vs_A -> ball/05/table.*

Every card is `scripts/ball_scorecard.py card` on the selection split (GSR valid games 2+3), rows 1280 (+ e3 where
dumped). `<arm>_e<epoch>` is paired with dd2c (the card's default reference); `B_e<epoch>_vs_A` is B's dump paired
with A's dump of the same epoch (`--reference A_e<epoch>/dump`). The primary comparison (ticket 02 rule,
2026-10-24 21:02) is B - A in well placed at 1280, both after the last epoch (epoch 1 = last.pt); the rest is
descriptive. Guards for the winner: people mAP50-95 >= -0.01 and Ekstraklasa dev mAP50-95 >= -0.02 (points), against
A (the primary row's reference) and against dd2c (they protect the pipeline).
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ball_scorecard as bs  # noqa: E402

OUT = bs.ROOT / "ball/05"
ARMS = ("A", "B")
EPOCHS = (0, 1)
LAST = 1
# (column, card section, number) of the table; selection split unless the section is Ekstraklasa's.
NUMBERS = (
    ("well_placed", "placement", "well_placed"),
    ("well_placed_with_h", "placement", "well_placed_with_homography"),
    ("far_of_placed", "placement", "far_of_placed"),
    ("median_m", "placement", "median_m"),
    ("ball_ap50", "detection", "ball_ap50"),
    ("ball_ap50_95", "detection", "ball_ap50_95"),
    ("ball_recall", "detection", "ball_recall"),
    ("recall_centre", "hits", "recall_centre@0.15"),
    ("false_per_frame", "hits", "false_balls_per_frame"),
    ("people_map50_95", "detection", "people_map50_95"),
)
EKSTRAKLASA = (
    ("eks_ball_ap50", "detection", "ball_ap50"),
    ("eks_map50_95", "detection", "map50_95"),
)


def read(folder: Path) -> dict:
    return json.loads((folder / "scorecard.json").read_text())


def values(card: dict, row: str, paired: bool) -> dict:
    """The table's numbers of one card row: values with CIs, or differences against the card's reference."""
    out = {}
    for prefix, split, numbers in (
        ("", "selection", NUMBERS),
        ("", "ekstraklasa", EKSTRAKLASA),
    ):
        entry = card["cards"].get(f"{split}/{row}")
        if entry is None:
            continue
        for column, section, name in numbers:
            if paired:
                found = entry[f"{section}_vs_reference"][name]
                out[prefix + column] = found["difference"]
            else:
                found = entry[section][name]
                out[prefix + column] = found["value"]
            out[prefix + column + "_ci"] = found["ci"]
    return out


def table(out: Path) -> tuple[pd.DataFrame, dict]:
    rows = []
    cards = {}
    for arm in ARMS:
        for epoch in EPOCHS:
            folder = out / f"{arm}_e{epoch}"
            if not (folder / "scorecard.json").is_file():
                continue
            cards[(arm, epoch)] = card = read(folder)
            for row in card["rows"]:
                rows.append(
                    {"arm": arm, "epoch": epoch, "row": row, "kind": "value"}
                    | values(card, row, False)
                    | {"ms_per_frame": card["cost"]["ms_per_frame"].get(f"gsr/{row}")}
                )
                rows.append(
                    {"arm": arm, "epoch": epoch, "row": row, "kind": "minus dd2c"}
                    | values(card, row, True)
                )
            vs_a = out / f"B_e{epoch}_vs_A"
            if arm == "B" and (vs_a / "scorecard.json").is_file():
                cards[("B-A", epoch)] = paired = read(vs_a)
                for row in paired["rows"]:
                    rows.append(
                        {"arm": "B", "epoch": epoch, "row": row, "kind": "minus A"}
                        | values(paired, row, True)
                    )
    frame = pd.DataFrame(rows)
    return frame, verdict(cards)


def verdict(cards: dict) -> dict:
    """The rule's read: primary B - A at the last epoch, guards against A and dd2c, the winner."""
    paired = cards.get(("B-A", LAST))
    b = cards.get(("B", LAST))
    if paired is None or b is None:
        return {"complete": False}
    primary = paired["rule"]["primary"]
    guards = {
        "people_vs_A": paired["rule"]["guard_people"],
        "ekstraklasa_vs_A": paired["rule"]["guard_ekstraklasa"],
        "people_vs_dd2c": b["rule"]["guard_people"],
        "ekstraklasa_vs_dd2c": b["rule"]["guard_ekstraklasa"],
    }
    return {
        "complete": True,
        "primary": "B - A, well placed, row 1280, selection split (games 2+3), epoch 1 (last.pt)",
        "difference": primary["difference"],
        "ci": primary["ci"],
        "success": primary["success"],
        "guards": guards,
        "winner": "B" if primary["success"] and all(guards.values()) else None,
        "ekstraklasa_ball_ap50_direction_B": b["rule"].get(
            "ekstraklasa_ball_ap50_direction"
        ),
    }


def _fmt(value: float, ci, pp: bool) -> str:
    scale, digits = (100, 2) if pp else (1, 3)
    text = f"{scale * value:.{digits}f}"
    if ci is not None and isinstance(ci, list):
        text += f" [{scale * ci[0]:.{digits}f}; {scale * ci[1]:.{digits}f}]"
    return text


PERCENT = {
    "well_placed",
    "well_placed_with_h",
    "far_of_placed",
    "ball_recall",
    "recall_centre",
}


def markdown(frame: pd.DataFrame, result: dict) -> str:
    columns = [
        ("well_placed", "well placed % (2+3)"),
        ("well_placed_with_h", "with H %"),
        ("far_of_placed", "> 5 m of placed %"),
        ("ball_ap50", "ball AP50"),
        ("ball_ap50_95", "ball AP50-95"),
        ("people_map50_95", "people mAP50-95"),
        ("eks_ball_ap50", "Eks. ball AP50"),
        ("eks_map50_95", "Eks. mAP50-95"),
    ]
    lines = [
        "# ball-detection 05: per-epoch scorecards (selection split, games 2+3)",
        "",
        "Values with 95% clip-bootstrap CIs (2,000 draws, seed 0); differences are paired (same frames and draws).",
        "Percent columns in %, differences in pp.",
        "",
    ]
    for row in sorted(frame.row.unique(), key=lambda r: r != "1280"):
        lines += [f"## Row {row}", ""]
        lines.append("| arm | epoch | | " + " | ".join(t for _, t in columns) + " |")
        lines.append("|---|---|---|" + "---|" * len(columns))
        for _, r in frame[frame.row == row].iterrows():
            cells = []
            for c, _ in columns:
                if c not in r or pd.isna(r.get(c)):
                    cells.append("")
                    continue
                cells.append(_fmt(r[c], r.get(c + "_ci"), c in PERCENT))
            lines.append(
                f"| {r.arm} | {r.epoch} | {r.kind} | " + " | ".join(cells) + " |"
            )
        lines.append("")
    if result.get("complete"):
        lines += [
            "## Rule",
            "",
            f"Primary ({result['primary']}): {100 * result['difference']:+.2f} pp "
            f"[{100 * result['ci'][0]:+.2f}; {100 * result['ci'][1]:+.2f}], success: {result['success']}.",
            f"Guards: {result['guards']}. Winner: {result['winner']}.",
            "",
        ]
    return "\n".join(lines)


def table_command(args: argparse.Namespace) -> None:
    frame, result = table(args.out)
    frame.to_csv(args.out / "table.csv", index=False)
    (args.out / "verdict.json").write_text(
        json.dumps(result, indent=2, default=float) + "\n"
    )
    (args.out / "table.md").write_text(markdown(frame, result))
    print((args.out / "table.md").read_text())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(required=True)
    t = commands.add_parser("table")
    t.add_argument("--out", type=Path, default=OUT)
    t.set_defaults(handler=table_command)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
