import json
from pathlib import Path
from typing import Any

import pandas as pd


def stage_path(output_dir: Path, stage_name: str, suffix: str) -> Path:
    return output_dir / f"{stage_name}.{suffix}"


def write_json_artifact(value: object, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    return path


def read_json_artifact(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_dataframe_artifact(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".csv":
        frame.to_csv(path, index=False)
    elif path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported dataframe artifact format: {path}")
    return path


def read_dataframe_artifact(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported dataframe artifact format: {path}")
