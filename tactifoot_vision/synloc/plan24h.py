from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


EXPERIMENT_LEDGER_COLUMNS = [
    "timestamp",
    "phase",
    "run_name",
    "detector_checkpoint",
    "point_strategy",
    "confidence_threshold",
    "image_nms_iou",
    "world_nms_radius_m",
    "behind_camera_policy",
    "clip_margin_m",
    "tile_size",
    "tile_overlap",
    "topk_per_image",
    "aux_data_used",
    "val_map_locsim",
    "val_recall",
    "val_precision",
    "val_f1",
    "notes",
]


@dataclass(frozen=True)
class ExperimentRecord:
    timestamp: str
    phase: str
    run_name: str
    detector_checkpoint: str
    point_strategy: str
    confidence_threshold: float
    image_nms_iou: float
    world_nms_radius_m: float
    behind_camera_policy: str
    clip_margin_m: float
    tile_size: int
    tile_overlap: int
    topk_per_image: int
    aux_data_used: str
    val_map_locsim: float
    val_recall: float
    val_precision: float
    val_f1: float
    notes: str

    def to_csv_row(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FinalistSummary:
    run_name: str
    phase: str
    map_locsim: float
    precision: float
    recall: float
    f1: float
    score_threshold: float
    final_predictions: int
    archive_path: Path


@dataclass(frozen=True)
class PhaseRunSelection:
    run_name: str
    run_dir: Path
    archive_path: Path
    results_path: Path
    metadata_path: Path
    raw_results_path: Path
    metrics: dict[str, float]
    config_snapshot: dict[str, Any]
    notes: str = ""


@dataclass(frozen=True)
class FinalizedRunArtifacts:
    output_dir: Path
    best_submission_zip: Path
    best_results_json: Path
    best_metadata_json: Path
    val_metrics_summary_json: Path
    experiments_csv: Path
    run_report_md: Path


def finalize_24h_run(
    *,
    output_dir: Path,
    best_run: PhaseRunSelection,
    finalists: Iterable[FinalistSummary],
    experiments: Iterable[ExperimentRecord],
    run_started_at: str,
    run_finished_at: str | None = None,
) -> FinalizedRunArtifacts:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    best_submission_zip = output_dir / "best_submission.zip"
    best_results_json = output_dir / "best_results.json"
    best_metadata_json = output_dir / "best_metadata.json"
    val_metrics_summary_json = output_dir / "val_metrics_summary.json"
    experiments_csv = output_dir / "experiments.csv"
    run_report_md = output_dir / "run_report.md"

    shutil.copy2(best_run.archive_path, best_submission_zip)
    shutil.copy2(best_run.results_path, best_results_json)
    shutil.copy2(best_run.metadata_path, best_metadata_json)

    finalists_payload = [asdict(item) for item in finalists]
    for item in finalists_payload:
        item["archive_path"] = str(item["archive_path"])
    val_metrics_summary_json.write_text(
        json.dumps(
            {
                "generated_at": _utc_timestamp(),
                "best_run": {
                    "run_name": best_run.run_name,
                    "metrics": best_run.metrics,
                    "archive_path": str(best_submission_zip),
                    "raw_results_path": str(best_run.raw_results_path),
                    "notes": best_run.notes,
                },
                "finalists": finalists_payload,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    rows = list(experiments)
    with experiments_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPERIMENT_LEDGER_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_row())

    run_report_md.write_text(
        _build_run_report(
            best_run=best_run,
            finalists=list(finalists),
            experiments=rows,
            run_started_at=run_started_at,
            run_finished_at=run_finished_at or _utc_timestamp(),
            submission_zip=best_submission_zip,
        ),
        encoding="utf-8",
    )

    return FinalizedRunArtifacts(
        output_dir=output_dir,
        best_submission_zip=best_submission_zip,
        best_results_json=best_results_json,
        best_metadata_json=best_metadata_json,
        val_metrics_summary_json=val_metrics_summary_json,
        experiments_csv=experiments_csv,
        run_report_md=run_report_md,
    )


def _build_run_report(
    *,
    best_run: PhaseRunSelection,
    finalists: list[FinalistSummary],
    experiments: list[ExperimentRecord],
    run_started_at: str,
    run_finished_at: str,
    submission_zip: Path,
) -> str:
    config_blocks = []
    for item in experiments:
        config_blocks.append(
            (
                f"- `{item.run_name}` [{item.phase}] "
                f"conf={item.confidence_threshold:.2f}, "
                f"img_nms={item.image_nms_iou:.2f}, "
                f"world_nms={item.world_nms_radius_m:.2f}, "
                f"point={item.point_strategy}, "
                f"policy={item.behind_camera_policy}, "
                f"clip_margin={item.clip_margin_m:.2f}, "
                f"tile={item.tile_size}/{item.tile_overlap}, "
                f"topk={item.topk_per_image}, "
                f"map_locsim={item.val_map_locsim:.6f}"
            )
        )
    finalist_lines = [
        (
            f"- `{item.run_name}` [{item.phase}] "
            f"map_locsim={item.map_locsim:.6f}, precision={item.precision:.6f}, "
            f"recall={item.recall:.6f}, f1={item.f1:.6f}, "
            f"score_threshold={item.score_threshold:.6f}, final_predictions={item.final_predictions}"
        )
        for item in finalists
    ]

    return "\n".join(
        [
            "# SynLoc 24h Run Report",
            "",
            f"- Run started: `{run_started_at}`",
            f"- Run finished: `{run_finished_at}`",
            f"- Best variant: `{best_run.run_name}`",
            f"- Final val map_locsim: `{best_run.metrics.get('map_locsim', 0.0):.6f}`",
            f"- Selection rationale: chose the highest full-val `map_locsim` under the ranking-first rule. {best_run.notes}".strip(),
            f"- Submission ZIP: `{submission_zip}`",
            "",
            "## Finalists",
            *finalist_lines,
            "",
            "## Configurations",
            *config_blocks,
            "",
            "## Best Config Snapshot",
            "```json",
            json.dumps(best_run.config_snapshot, indent=2, sort_keys=True),
            "```",
        ]
    )


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
