from pathlib import Path

from tactifoot_vision.domain import ExportArtifact, PipelineResult


class MotExporter:
    def write(self, result: PipelineResult, path: Path) -> ExportArtifact:
        rows: list[str] = []
        for frame in result.frames:
            for track in frame.tracks:
                x, y, width, height = track.bbox.xywh
                confidence = track.confidence if track.confidence is not None else 1.0
                rows.append(
                    f"{frame.frame_index + 1},{track.track_id},{x:.3f},{y:.3f},"
                    f"{width:.3f},{height:.3f},{confidence:.6f},-1,-1,-1"
                )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        return ExportArtifact(path=path, format="mot", rows=len(rows))
