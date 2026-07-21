from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from statsbombpy import api_client


MATCHES_ROOT = Path("/home/kuba/projects/ball-vision/data/20232024")
OUT_ROOT = Path("results/project/raw/statsbomb_full")

CREDS_PATH = Path("/home/kuba/projects/ball-vision/secrets/statsbomb_api.json")


def _load_creds() -> Dict[str, str]:
    payload = json.loads(CREDS_PATH.read_text(encoding="utf-8"))
    user = payload.get("user")
    passwd = payload.get("passwd")
    if not user or not passwd:
        raise RuntimeError(f"Invalid creds file (missing user/passwd): {CREDS_PATH}")
    return {"user": user, "passwd": passwd}


def _events_to_dataframe(events_dict: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for _, ev in events_dict.items():
        if not isinstance(ev, dict):
            continue
        ev_type = ev.get("type") or {}
        team = ev.get("team") or {}
        possession_team = ev.get("possession_team") or {}
        play_pattern = ev.get("play_pattern") or {}
        rows.append(
            {
                "event_uuid": ev.get("id"),
                "event_index": ev.get("index"),
                "period": ev.get("period"),
                "timestamp": ev.get("timestamp"),
                "minute": ev.get("minute"),
                "second": ev.get("second"),
                "event_type": ev_type.get("name") if isinstance(ev_type, dict) else None,
                "team": team.get("name") if isinstance(team, dict) else None,
                "possession": ev.get("possession"),
                "possession_team": possession_team.get("name")
                if isinstance(possession_team, dict)
                else None,
                "play_pattern": play_pattern.get("name")
                if isinstance(play_pattern, dict)
                else None,
                "duration": ev.get("duration"),
                "event_location": ev.get("location"),
                "match_id": ev.get("match_id"),
                "raw_event": json.dumps(ev, ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows)


def build_for_match(match_dir: Path, creds: Dict[str, str]) -> Path:
    frames_path = match_dir / "frames.parquet"
    if not frames_path.is_file():
        raise FileNotFoundError(f"Missing frames.parquet: {frames_path}")

    df_frames = pd.read_parquet(frames_path)
    match_id: Optional[int] = None
    if "match_id" in df_frames.columns and df_frames["match_id"].notna().any():
        match_id = int(df_frames["match_id"].dropna().iloc[0])
    else:
        # Fallback: parse from directory name prefix.
        try:
            match_id = int(match_dir.name.split("_", 1)[0])
        except Exception:
            match_id = None
    if match_id is None:
        raise RuntimeError(f"Cannot determine match_id for: {match_dir}")

    events_dict = api_client.events(match_id, creds)
    df_events = _events_to_dataframe(events_dict)
    if df_events.empty:
        raise RuntimeError(f"No events fetched for match_id={match_id}")

    df_full = df_frames.merge(
        df_events,
        how="left",
        on=["event_uuid", "match_id"],
        validate="many_to_one",
    )

    out_path = OUT_ROOT / f"{match_dir.name}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_full.to_parquet(out_path, index=False, compression="zstd")
    return out_path


def main() -> None:
    if not MATCHES_ROOT.is_dir():
        raise FileNotFoundError(f"Missing matches root: {MATCHES_ROOT}")
    if not CREDS_PATH.is_file():
        raise FileNotFoundError(f"Missing StatsBomb creds file: {CREDS_PATH}")

    creds = _load_creds()

    match_dirs = sorted([p for p in MATCHES_ROOT.iterdir() if p.is_dir()])
    if not match_dirs:
        raise RuntimeError(f"No match dirs found under: {MATCHES_ROOT}")

    outputs = []
    for match_dir in match_dirs:
        outputs.append(build_for_match(match_dir, creds))

    df_index = pd.DataFrame(
        [{"match": p.stem, "path": str(p)} for p in outputs]
    )
    df_index.to_csv(OUT_ROOT / "index.csv", index=False)


if __name__ == "__main__":
    main()
