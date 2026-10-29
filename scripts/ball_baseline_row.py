"""ball-detection 25: the old baseline detector's placement row on GSR test and the whole GSR valid (descriptive).

    uv run python scripts/ball_baseline_row.py dump-test --weights W --out OUT   # GPU: row 1280 on GSR test
    uv run python scripts/ball_baseline_row.py score --valid-dump D --test-dump T --out OUT

The measure is ticket 02's: top-1 ball >= 0.15 of the 1280 row, no 45 px filter, centre through the frozen
homographies, well placed <= 5 m, frames without a homography are misses. The baseline's frames are paired with the
pipeline's per-frame outputs of ticket 09 (``read_test``, ``read_valid``: dd2c top-1 and dd2c + 45 px + track_ball).
Reads game 5 and GSR test again for the baseline; descriptive after the closed research, selects nothing.
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ball_errors as be  # noqa: E402
import ball_scorecard as bs  # noqa: E402
import detector_bench as bench  # noqa: E402

ROW = "1280"
READ = be.ROOT / "ball/09"
SUMMARY_KEYS = ("well_placed", "far_of_placed")
PIPELINE = {
    "valid": (
        "read_valid/placement_full_top1.parquet",
        "read_valid/placement_full_track.parquet",
    ),
    "test": (
        "read_test/placement_test_top1.parquet",
        "read_test/placement_test_track.parquet",
    ),
}
# controls that must reproduce: (split, name) -> well placed
CONTROLS = {
    ("valid", "dd2c_top1"): 0.6018,
    ("valid", "pipeline"): 0.7630,
    ("test", "dd2c_top1"): 0.6896,
    ("test", "pipeline"): 0.7707,
    ("selection", "base"): 0.2475,
}


def dump_test(args: argparse.Namespace) -> None:
    """The baseline's candidates (conf 0.001, row 1280) on the stride-5 GSR test frames."""
    bench.dump(
        SimpleNamespace(
            weights=args.weights,
            out=args.out,
            rows=[ROW],
            device=args.device,
            split="test",
        )
    )


def split_clips(split: str) -> list[str]:
    return [p.parent.name for p in bench.split_label_files(split)]


def base_rows(folder: Path, split: str, games=None) -> pd.DataFrame:
    """The baseline's placement rows on ``split`` (``games``: all by default) from its dump."""
    _, images, found = be.load_found(folder)
    gsr = images[images.set == "gsr"]
    games = set(gsr.game.astype(str)) if games is None else set(games)
    gsr = gsr.assign(game=gsr.game.astype(str))
    return bs.placement_table(
        found[("gsr", ROW)],
        gsr.assign(set="gsr"),
        bench.frozen_homographies(split),
        games,
        split,
    )


def pct(value: float) -> float:
    return round(100 * value, 2)


def read_split(split: str, base: pd.DataFrame, reads: Path = READ) -> dict:
    """Baseline, dd2c top-1 and pipeline on the same frames: shares with clip-bootstrap intervals and pairs."""
    top1, track = (pd.read_parquet(reads / name) for name in PIPELINE[split])
    keys = ["clip", "frame"]
    for other in (top1, track):
        if not base[keys].equals(other[keys]):
            raise SystemExit(f"{split}: baseline frames differ from the pipeline's")
    draws = bs.Draws(split_clips(split))
    found = {
        "base": bs.placement_draws(base, draws),
        "dd2c_top1": bs.placement_draws(top1, draws),
        "pipeline": bs.placement_draws(track, draws),
    }
    out = {
        "frames": len(base),
        "no_homography": int((~base.homography).sum()),
        "rows": {
            k: bs.summary({n: v[n] for n in SUMMARY_KEYS}) for k, v in found.items()
        },
        "base_to_dd2c": bs.paired(found["dd2c_top1"], found["base"]),
        "base_to_pipeline": bs.paired(found["pipeline"], found["base"]),
    }
    out["base_to_dd2c"] = {k: out["base_to_dd2c"][k] for k in SUMMARY_KEYS}
    out["base_to_pipeline"] = {k: out["base_to_pipeline"][k] for k in SUMMARY_KEYS}
    return out


def selection_control(base: pd.DataFrame) -> dict:
    """The selection games' row from the same dump: ticket 02's 24.75% [16.3; 34.0] and 26.18% > 5 m."""
    part = base[base.game.isin(bs.SPLITS["selection"])].reset_index(drop=True)
    clips = sorted(set(part["clip"]))
    # ticket 02 draws over the games' clips, in the dump's clip order
    found = bs.placement_draws(part, bs.Draws(clips))
    return bs.summary({n: found[n] for n in SUMMARY_KEYS}) | {"frames": len(part)}


def check_controls(result: dict) -> dict:
    values = {
        (s, k): result[s]["rows"][k]["well_placed"]["value"]
        for s in ("valid", "test")
        for k in ("dd2c_top1", "pipeline")
    }
    values[("selection", "base")] = result["selection"]["well_placed"]["value"]
    report = {}
    for key, expected in CONTROLS.items():
        report["/".join(key)] = {
            "expected": expected,
            "got": round(values[key], 4),
            "ok": bool(round(values[key], 4) == expected),
        }
    return report


def score(args: argparse.Namespace) -> None:
    valid = base_rows(args.valid_dump, "valid")
    test = base_rows(args.test_dump, "test")
    result = {
        "valid": read_split("valid", valid),
        "test": read_split("test", test),
        "selection": selection_control(valid),
        "bootstrap": bs.BOOTSTRAP,
    }
    result["controls"] = check_controls(result)
    args.out.mkdir(parents=True, exist_ok=True)
    valid.to_parquet(args.out / "placement_valid_base.parquet", index=False)
    test.to_parquet(args.out / "placement_test_base.parquet", index=False)
    (args.out / "read.json").write_text(
        json.dumps(result, indent=2, default=float) + "\n"
    )
    print(json.dumps(result, indent=2, default=float))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    d = commands.add_parser("dump-test", help="baseline row 1280 on GSR test (GPU)")
    d.add_argument("--weights", type=Path, required=True)
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--device", default=None)
    s = commands.add_parser("score", help="placement rows, pairs, controls (CPU)")
    s.add_argument("--valid-dump", type=Path, required=True)
    s.add_argument("--test-dump", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    {"dump-test": dump_test, "score": score}[args.command](args)


if __name__ == "__main__":
    main()
