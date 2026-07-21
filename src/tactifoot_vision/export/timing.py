import json
import math
from datetime import UTC, datetime
from pathlib import Path

from tactifoot_vision.domain import ExportArtifact


def format_seconds_to_hmsms(timestamp_seconds: float | None) -> str | None:
    if timestamp_seconds is None or not math.isfinite(timestamp_seconds):
        return None
    hours = math.floor(timestamp_seconds / 3600)
    minutes = math.floor((timestamp_seconds % 3600) / 60)
    seconds = math.floor(timestamp_seconds % 60)
    milliseconds = math.floor(
        (timestamp_seconds - math.floor(timestamp_seconds)) * 1000
    )
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


class ExportManifestWriter:
    def write(
        self,
        *,
        artifacts: tuple[ExportArtifact, ...],
        metrics: dict[str, float],
        path: Path,
    ) -> ExportArtifact:
        payload = {
            "created_at": datetime.now(UTC).isoformat(),
            "artifacts": [
                {
                    "path": str(artifact.path),
                    "format": artifact.format,
                    "rows": artifact.rows,
                }
                for artifact in artifacts
            ],
            "metrics_keys": sorted(metrics),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return ExportArtifact(path, "manifest_json", len(artifacts))
