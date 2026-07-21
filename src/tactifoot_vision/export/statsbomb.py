from pathlib import Path

from tactifoot_vision.domain import PipelineResult


class StatsBombExporter:
    def write(self, result: PipelineResult, path: Path) -> None:
        _ = result, path
        raise NotImplementedError(
            "Native StatsBomb360 export is not implemented. Use "
            "PipelineCsvExporter for generic projection CSV output."
        )
