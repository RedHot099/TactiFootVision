import json
from pathlib import Path

import pandas as pd

from tactifoot_vision.domain import ExportArtifact, PipelineResult


class PipelineCsvExporter:
    def write(self, result: PipelineResult, path: Path) -> ExportArtifact:
        rows: list[dict[str, object]] = []
        for frame in result.frames:
            projection = frame.projection
            for track in frame.tracks:
                pitch_point = (
                    projection.points_by_track_id.get(track.track_id)
                    if projection is not None
                    else None
                )
                rows.append(
                    {
                        "frame": frame.frame_index,
                        "timestamp_seconds": frame.timestamp_seconds,
                        "track_id": track.track_id,
                        "class_id": track.class_id,
                        "class_name": track.class_name,
                        "confidence": track.confidence,
                        "team_id": track.team_id,
                        "x": track.bbox.x1,
                        "y": track.bbox.y1,
                        "width": track.bbox.width,
                        "height": track.bbox.height,
                        "pitch_x": pitch_point.x if pitch_point else None,
                        "pitch_y": pitch_point.y if pitch_point else None,
                        "projection_status": projection.status if projection else None,
                        "homography": _json_or_none(
                            projection.homography.tolist()
                            if projection is not None
                            and projection.homography is not None
                            else None
                        ),
                    }
                )
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)
        return ExportArtifact(path=path, format="pipeline_csv", rows=len(rows))


def _json_or_none(value: object | None) -> str | None:
    return json.dumps(value) if value is not None else None
