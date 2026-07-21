import json
from pathlib import Path

from tactifoot_vision.cli import main
from tactifoot_vision.domain import (
    BBox,
    DetectionSet,
    FrameResult,
    PipelineResult,
    Track,
    TrackSet,
)


class FakePipeline:
    def run_video(
        self, path: object, *, max_frames: int | None = None
    ) -> PipelineResult:
        _ = path
        frames = max_frames or 1
        return PipelineResult(
            tuple(
                FrameResult(
                    frame_index=index,
                    timestamp_seconds=None,
                    detections=DetectionSet.empty(),
                    tracks=TrackSet(
                        (
                            Track(
                                track_id=1,
                                bbox=BBox(0.0, 0.0, 1.0, 1.0),
                                class_id=2,
                                class_name="player",
                            ),
                            Track(
                                track_id=2,
                                bbox=BBox(2.0, 2.0, 3.0, 3.0),
                                class_id=0,
                                class_name="ball",
                            ),
                        )
                    ),
                )
                for index in range(frames)
            )
        )


def test_cli_track_images_fake_pipeline(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "tactifoot_vision.cli.build_pipeline", lambda config: FakePipeline()
    )

    assert (
        main(
            [
                "track",
                "images",
                "--config",
                "configs/pipeline/yolo_bytetrack_smoke.yaml",
                "--max-frames",
                "1",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["backend"] == "bytetrack"
    assert payload["frames"] == 1
    assert payload["tracks"] == 2
    assert payload["track_ids"] == [1, 2]
    assert payload["classes"] == ["ball", "player"]


def test_cli_evaluate_tracking_auto_offsets_pipeline_csv(
    tmp_path: Path, capsys
) -> None:
    pred = tmp_path / "pipeline.csv"
    gt = tmp_path / "gt.txt"
    pred.write_text(
        "\n".join(
            [
                "frame,timestamp_seconds,track_id,class_id,class_name,x,y,width,height",
                "0,0.0,5,2,player,0,0,10,10",
            ]
        ),
        encoding="utf-8",
    )
    gt.write_text("1,1,0,0,10,10,1,-1,-1,-1\n", encoding="utf-8")

    assert main(["evaluate", "tracking", "--pred", str(pred), "--gt", str(gt)]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["prediction_frame_offset"] == 1.0
    assert payload["tp"] == 1.0
