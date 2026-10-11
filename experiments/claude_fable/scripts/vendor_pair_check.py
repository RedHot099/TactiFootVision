"""Run the independent implementation of the frozen SoccerNet/StatsBomb pairing rule."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from xgpoc.vendor_check.pairing_check import build_pairing_tables

ROOT = Path(__file__).resolve().parents[1]
SOCCERNET_ROOT = Path("/home/kuba/projects/football_data/xg_shots_full")
STATSBOMB_ROOT = Path("/home/kuba/projects/football_data/statsbomb_open_v1/tables")
INDEX = SOCCERNET_ROOT / "soccernet_720p/index_v1_2.parquet"
ATTEMPTS = SOCCERNET_ROOT / "soccernet_720p/attempts_all.parquet"
LABELS = SOCCERNET_ROOT / "soccernet_labels"
MATCHES = STATSBOMB_ROOT / "matches.parquet"
SHOTS = STATSBOMB_ROOT / "shots.parquet"
CONTEXT = STATSBOMB_ROOT / "context.parquet"
OUTPUT = ROOT / "results/xgv_pairing_check"
IMPLEMENTATION_SOURCES = (
    ROOT / "xgpoc/vendor_check/__init__.py",
    ROOT / "xgpoc/vendor_check/pairing_check.py",
    ROOT / "scripts/vendor_pair_check.py",
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(2**20):
            digest.update(block)
    return digest.hexdigest()


def parse_date(value: str) -> datetime:
    for pattern in ("%d/%m/%Y - %H:%M:%S", "%d/%m/%Y - %H:%M"):
        try:
            return datetime.strptime(value, pattern)
        except ValueError:
            pass
    raise ValueError(f"Unsupported Labels-v2 gameDate: {value!r}")


def load_game_metadata(games: list[str]) -> tuple[pd.DataFrame, str]:
    rows = []
    labels_digest = hashlib.sha256()
    for game in sorted(games):
        path = LABELS / game / "Labels-v2.json"
        raw = path.read_bytes()
        labels_digest.update(game.encode())
        labels_digest.update(b"\0")
        labels_digest.update(hashlib.sha256(raw).digest())
        document = json.loads(raw)
        if document["UrlLocal"].rstrip("/") != game:
            raise ValueError(f"Labels-v2 UrlLocal does not match index game: {game}")
        score_match = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", document["gameScore"])
        if score_match is None:
            raise ValueError(
                f"Unsupported Labels-v2 gameScore: {document['gameScore']!r}"
            )
        rows.append(
            {
                "game": game,
                "league": game.split("/", 1)[0],
                "game_date": parse_date(document["gameDate"]),
                "home_team": document["gameHomeTeam"],
                "away_team": document["gameAwayTeam"],
                "home_score": int(score_match.group(1)),
                "away_score": int(score_match.group(2)),
            }
        )
    return pd.DataFrame(rows), labels_digest.hexdigest()


def main() -> None:
    index = pd.read_parquet(
        INDEX,
        columns=[
            "shot_id",
            "game",
            "half",
            "position_ms",
            "label",
            "goal",
            "is_penalty",
            "split",
        ],
    )
    if index["shot_id"].duplicated().any():
        raise ValueError("SoccerNet index contains duplicate shot_id values")
    metadata, labels_hash = load_game_metadata(index["game"].drop_duplicates().tolist())
    attempts = pd.read_parquet(
        ATTEMPTS,
        columns=["game", "half", "position_ms", "status"],
    )
    attempts = (
        attempts[attempts["status"].eq("goal_without_shot_annotation")]
        .drop_duplicates(["game", "half", "position_ms"])
        .reset_index(drop=True)
    )
    matches = pd.read_parquet(
        MATCHES,
        columns=[
            "match_id",
            "match_date",
            "competition",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
        ],
    )
    shots = pd.read_parquet(
        SHOTS,
        columns=[
            "match_id",
            "event_id",
            "index",
            "period",
            "timestamp_ms",
            "outcome",
            "shot_type",
        ],
    )
    context = pd.read_parquet(CONTEXT, columns=["event_id", "type"])
    own_goal_count = int(context["type"].eq("Own Goal For").sum())

    tables = build_pairing_tables(index, attempts, metadata, matches, shots)
    sort_keys = {
        "match_map": ["game"],
        "pairs": ["shot_id"],
        "half_offsets": ["game", "half"],
        "recovered_goals": ["game", "half", "anchor_ms"],
        "populations": ["event_id"],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_paths = {}
    for name, frame in tables.items():
        output_path = OUTPUT / f"{name}.parquet"
        frame.sort_values(sort_keys[name]).reset_index(drop=True).to_parquet(
            output_path, index=False
        )
        output_paths[name] = output_path

    sources = {
        path.name: file_hash(path)
        for path in (INDEX, ATTEMPTS, MATCHES, SHOTS, CONTEXT)
    }
    sources["Labels-v2"] = labels_hash
    counts = {name: len(frame) for name, frame in tables.items()}
    record = {
        "implementation": "xgpoc.vendor_check.pairing_check",
        "implementation_sources_sha256": {
            str(path.relative_to(ROOT)): file_hash(path)
            for path in IMPLEMENTATION_SOURCES
        },
        "sources_sha256": sources,
        "outputs_sha256": {
            path.name: file_hash(path) for path in output_paths.values()
        },
        "own_goal_context_rows_read": own_goal_count,
        "counts": counts,
    }
    (OUTPUT / "complete.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
