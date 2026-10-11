"""Compare the first and independent implementations of the frozen pairing rule."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIRST = ROOT / "results/xgv_pairing"
CHECK = ROOT / "results/xgv_pairing_check"
CHECK_IMPLEMENTATION_SOURCES = (
    ROOT / "xgpoc/vendor_check/__init__.py",
    ROOT / "xgpoc/vendor_check/pairing_check.py",
    ROOT / "scripts/vendor_pair_check.py",
)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_table(root: Path, stem: str) -> Path:
    for suffix in (".parquet", ".csv"):
        candidate = root / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No {stem} parquet or CSV under {root}")


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def unique_mapping(frame: pd.DataFrame, keys: list[str], value: str) -> dict:
    if frame.duplicated(keys).any():
        raise ValueError(f"Duplicate keys {keys} in comparison table")
    return {
        tuple(row[key] for key in keys): row[value]
        for row in frame[keys + [value]].to_dict("records")
    }


def json_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def values_equal(first, check, *, tolerance: float | None = None) -> bool:
    if pd.isna(first) and pd.isna(check):
        return True
    if pd.isna(first) or pd.isna(check):
        return False
    if tolerance is not None:
        return math.isclose(float(first), float(check), rel_tol=0.0, abs_tol=tolerance)
    return first == check


def compare_mappings(
    first: dict,
    check: dict,
    *,
    tolerance: float | None = None,
) -> list[dict]:
    differences = []
    for key in sorted(set(first) | set(check)):
        first_value = first.get(key)
        check_value = check.get(key)
        if (
            key not in first
            or key not in check
            or not values_equal(
                first_value,
                check_value,
                tolerance=tolerance,
            )
        ):
            differences.append(
                {
                    "key": [json_value(part) for part in key],
                    "first": json_value(first_value) if key in first else "<missing>",
                    "check": json_value(check_value) if key in check else "<missing>",
                }
            )
    return differences


def main() -> None:
    first_paths = {
        "match_map": find_table(FIRST, "match_map"),
        "pairs": find_table(FIRST, "pairs"),
        "half_offsets": find_table(FIRST, "half_offsets"),
        "recovered_goals": find_table(FIRST, "recovered_goals"),
        "populations": find_table(FIRST, "sb_unpaired"),
    }
    check_paths = {
        name: find_table(CHECK, name)
        for name in (
            "match_map",
            "pairs",
            "half_offsets",
            "recovered_goals",
            "populations",
        )
    }
    first_tables = {name: read_table(path) for name, path in first_paths.items()}
    check_tables = {name: read_table(path) for name, path in check_paths.items()}
    first_completion = json.loads((FIRST / "complete.json").read_text())

    first_pairs = first_tables["pairs"].loc[
        first_tables["pairs"]["paired"], ["shot_id", "sb_event_id"]
    ]
    check_pairs = check_tables["pairs"][["shot_id", "sb_event_id"]]
    first_recovered = first_tables["recovered_goals"].loc[
        first_tables["recovered_goals"]["recovered"],
        ["game", "half", "anchor_ms", "sb_event_id"],
    ]
    check_recovered = check_tables["recovered_goals"][
        ["game", "half", "anchor_ms", "sb_event_id"]
    ]

    mappings = {
        "match_map": (
            unique_mapping(first_tables["match_map"], ["game"], "sb_match_id"),
            unique_mapping(check_tables["match_map"], ["game"], "sb_match_id"),
            None,
        ),
        "pairs": (
            unique_mapping(first_pairs, ["shot_id"], "sb_event_id"),
            unique_mapping(check_pairs, ["shot_id"], "sb_event_id"),
            None,
        ),
        "half_offsets": (
            unique_mapping(first_tables["half_offsets"], ["game", "half"], "offset_ms"),
            unique_mapping(check_tables["half_offsets"], ["game", "half"], "offset_ms"),
            1e-6,
        ),
        "recovered_goals": (
            unique_mapping(
                first_recovered, ["game", "half", "anchor_ms"], "sb_event_id"
            ),
            unique_mapping(
                check_recovered, ["game", "half", "anchor_ms"], "sb_event_id"
            ),
            None,
        ),
        "populations": (
            unique_mapping(first_tables["populations"], ["event_id"], "population"),
            unique_mapping(check_tables["populations"], ["event_id"], "population"),
            None,
        ),
    }
    comparisons = {}
    all_equal = True
    for name, (first_mapping, check_mapping, tolerance) in mappings.items():
        differences = compare_mappings(
            first_mapping, check_mapping, tolerance=tolerance
        )
        equal = not differences
        all_equal &= equal
        comparisons[name] = {
            "equal": equal,
            "first_count": len(first_mapping),
            "check_count": len(check_mapping),
            "difference_count": len(differences),
            "differences": differences,
        }

    report = {
        "equal": all_equal,
        "files": {
            "first": {
                name: {"path": str(path.relative_to(ROOT)), "sha256": file_hash(path)}
                for name, path in first_paths.items()
            },
            "check": {
                name: {"path": str(path.relative_to(ROOT)), "sha256": file_hash(path)}
                for name, path in check_paths.items()
            },
        },
        "implementation_sources_sha256": {
            "first": first_completion["identity"]["sources"],
            "check": {
                str(path.relative_to(ROOT)): file_hash(path)
                for path in CHECK_IMPLEMENTATION_SOURCES
            },
        },
        "comparison_script_sha256": file_hash(Path(__file__)),
        "comparisons": comparisons,
    }
    destination = CHECK / "compare.json"
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(f"pairing implementations equal: {all_equal}")
    for name, comparison in comparisons.items():
        print(
            f"{name}: first={comparison['first_count']} check={comparison['check_count']} "
            f"differences={comparison['difference_count']}"
        )
    if not all_equal:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
