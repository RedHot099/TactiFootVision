from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from tactifoot_vision.visualization import (
    OverlayStyle,
    PipelineFrameAnnotator,
    PipelineVideoRenderer,
    PitchMinimapRenderer,
    SoccerPitchMinimap,
    TrackOverlayRenderer,
    estimate_team_colors_from_crops,
)


def test_pipeline_frame_annotator_draws_overlay_without_mutating_input() -> None:
    frame = np.zeros((120, 180, 3), dtype=np.uint8)
    rows = pd.DataFrame(
        [
            {
                "frame": 0,
                "track_id": 1,
                "class_name": "player",
                "x": 20,
                "y": 30,
                "width": 40,
                "height": 50,
                "pitch_x": 52.5,
                "pitch_y": 34.0,
            }
        ]
    )

    annotated = PipelineFrameAnnotator().annotate(frame, rows)

    assert frame.sum() == 0
    assert annotated.shape == frame.shape
    assert annotated.sum() > 0


def test_minimap_can_fallback_to_image_position_when_projection_is_missing() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    rows = pd.DataFrame(
        [
            {
                "frame": 0,
                "track_id": 1,
                "class_name": "player",
                "x": 20,
                "y": 30,
                "width": 40,
                "height": 50,
                "pitch_x": np.nan,
                "pitch_y": np.nan,
            }
        ]
    )
    minimap = PitchMinimapRenderer(fallback_to_image_position=True)

    minimap.draw(frame, rows)

    style = minimap.style
    radar_width = int(frame.shape[1] * style.radar_width_ratio)
    radar_height = int(frame.shape[0] * style.radar_height_ratio)
    radar_region = frame[
        frame.shape[0] - radar_height : frame.shape[0],
        frame.shape[1] // 2 - radar_width // 2 : frame.shape[1] // 2 + radar_width // 2,
    ]
    assert radar_region.sum() > 0
    assert frame[:40, :40].sum() == 0

    pitch = SoccerPitchMinimap(
        style=style,
        fallback_to_image_position=True,
    ).draw(rows, frame_shape=frame.shape)
    assert pitch.max() > max(style.minimap_background)


def test_annotator_uses_team_color_for_assigned_players() -> None:
    frame = np.zeros((120, 180, 3), dtype=np.uint8)
    rows = pd.DataFrame(
        [
            {
                "frame": 0,
                "track_id": 1,
                "class_name": "player",
                "team_id": 1,
                "x": 20,
                "y": 30,
                "width": 40,
                "height": 50,
                "pitch_x": 52.5,
                "pitch_y": 34.0,
            }
        ]
    )

    annotated = PipelineFrameAnnotator().annotate(frame, rows)

    team_color = PipelineFrameAnnotator().track_renderer.style.team_colors[1]
    assert np.any(np.all(annotated == team_color, axis=2))


def test_annotator_draws_bottom_center_track_id_label() -> None:
    frame = np.zeros((120, 180, 3), dtype=np.uint8)
    style = OverlayStyle(team_colors={1: (10, 20, 230)})
    rows = pd.DataFrame(
        [
            {
                "frame": 0,
                "track_id": 17,
                "class_name": "player",
                "team_id": 1,
                "x": 20,
                "y": 30,
                "width": 40,
                "height": 50,
            }
        ]
    )

    renderer = TrackOverlayRenderer(style=style)
    annotated = frame.copy()
    renderer.draw(annotated, rows)

    label_region = annotated[80:105, 25:55]
    assert np.any(np.all(label_region == style.team_colors[1], axis=2))
    assert np.any(np.all(label_region > 200, axis=2))


def test_estimate_team_colors_from_crops_uses_visible_kit_pixels() -> None:
    red_crop = np.full((20, 10, 3), (30, 40, 220), dtype=np.uint8)
    blue_crop = np.full((20, 10, 3), (220, 40, 30), dtype=np.uint8)

    colors = estimate_team_colors_from_crops(
        [red_crop, blue_crop], np.asarray([0, 1], dtype=np.int_)
    )

    assert colors[0] == (30, 40, 220)
    assert colors[1] == (220, 40, 30)


def test_pipeline_video_renderer_writes_annotated_video(tmp_path: Path) -> None:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    for index in range(2):
        image = np.zeros((80, 120, 3), dtype=np.uint8)
        cv2.imwrite(str(frame_dir / f"{index:06d}.jpg"), image)
    annotations = pd.DataFrame(
        [
            {
                "frame": 0,
                "track_id": 1,
                "class_name": "player",
                "x": 10,
                "y": 10,
                "width": 20,
                "height": 30,
                "pitch_x": 10.0,
                "pitch_y": 20.0,
            },
            {
                "frame": 1,
                "track_id": 2,
                "class_name": "ball",
                "x": 30,
                "y": 20,
                "width": 8,
                "height": 8,
                "pitch_x": 50.0,
                "pitch_y": 40.0,
            },
        ]
    )
    output = tmp_path / "annotated.mp4"

    artifact = PipelineVideoRenderer(default_fps=10.0).render(
        source=frame_dir,
        annotations=annotations,
        output_path=output,
        max_frames=2,
    )

    assert artifact.path == output
    assert artifact.format == "annotated_video"
    assert artifact.rows == 2
    assert output.is_file()
    assert output.stat().st_size > 0


def test_pipeline_video_renderer_accepts_empty_annotations(tmp_path: Path) -> None:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    cv2.imwrite(str(frame_dir / "000000.jpg"), np.zeros((80, 120, 3), dtype=np.uint8))

    artifact = PipelineVideoRenderer().render(
        source=frame_dir,
        annotations=pd.DataFrame(),
        output_path=tmp_path / "empty.mp4",
    )

    assert artifact.rows == 1
