from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from numpy.typing import NDArray

from tactifoot_vision.datasets.soccernet_gsr import read_gsr_labels
from tactifoot_vision.projection import PitchModel
from tactifoot_vision.visualization.video import (
    OverlayStyle,
    SoccerPitchMinimap,
    TrackOverlayRenderer,
)

ROOT = Path(__file__).resolve().parents[2]
SEQUENCE = "SNGS-021"
SOURCE_DIR = ROOT / "data/SoccerNetGS/valid" / SEQUENCE / "img1"
PROJECTIONS_PATH = (
    ROOT
    / "results/experiments/homography_comparison_valid_current_oracle/projections.parquet"
)
OUTPUT_PATH = ROOT / "presentation_assets/video/homography_radar_pipeline_minimap.mp4"
MAX_FRAMES = 750
FPS = 25.0


class LowerLeftMinimapRenderer:
    def __init__(
        self,
        *,
        style: OverlayStyle,
        pitch: PitchModel | None = None,
        margin: int = 28,
    ) -> None:
        self.style = style
        self.pitch = pitch or PitchModel()
        self.margin = margin

    def draw(self, frame: NDArray[np.uint8], rows: pd.DataFrame) -> None:
        minimap = SoccerPitchMinimap(style=self.style, pitch=self.pitch).draw(
            rows,
            frame_shape=frame.shape,
        )
        target_size = (
            max(1, int(frame.shape[1] * self.style.radar_width_ratio)),
            max(1, int(frame.shape[0] * self.style.radar_height_ratio)),
        )
        radar = sv.resize_image(minimap, target_size, keep_aspect_ratio=True)
        radar_height, radar_width = radar.shape[:2]
        x0 = self.margin
        y0 = frame.shape[0] - radar_height - self.margin
        if y0 < 0:
            return
        sv.draw_image(
            scene=frame,
            image=radar,
            opacity=self.style.minimap_alpha,
            rect=sv.Rect(x=x0, y=y0, width=radar_width, height=radar_height),
        )


def main() -> None:
    labels = read_gsr_labels(SOURCE_DIR.parent)
    annotations = _build_annotations(labels_path=SOURCE_DIR.parent)
    style = OverlayStyle(
        minimap_size=(1050, 680),
        minimap_padding=42,
        minimap_alpha=0.78,
        minimap_player_radius=10,
        minimap_ball_radius=7,
        radar_width_ratio=0.32,
        radar_height_ratio=0.32,
        team_colors={0: (70, 120, 255), 1: (255, 85, 85)},
    )
    track_renderer = TrackOverlayRenderer(style=style)
    minimap_renderer = LowerLeftMinimapRenderer(style=style, pitch=PitchModel())

    frame_paths = sorted(SOURCE_DIR.glob("*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"No source frames found in {SOURCE_DIR}")
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise ValueError(f"Could not read first frame: {frame_paths[0]}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (first.shape[1], first.shape[0]),
    )
    rows_by_frame = {
        int(frame): rows for frame, rows in annotations.groupby("frame", sort=False)
    }
    try:
        for offset, frame_path in enumerate(frame_paths[:MAX_FRAMES]):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            rows = rows_by_frame.get(offset, _empty_annotations())
            output = np.asarray(frame, dtype=np.uint8).copy()
            track_renderer.draw(output, rows)
            minimap_renderer.draw(output, rows)
            writer.write(output)
    finally:
        writer.release()

    print(f"Rendered {min(MAX_FRAMES, len(frame_paths))} frames to {OUTPUT_PATH}")
    print(f"Loaded {len(labels.athletes)} athlete labels for {SEQUENCE}")


def _build_annotations(*, labels_path: Path) -> pd.DataFrame:
    labels = read_gsr_labels(labels_path)
    projections = pd.read_parquet(PROJECTIONS_PATH)
    oracle = projections[
        (projections["sequence"] == SEQUENCE)
        & (projections["method"] == "oracle_gsr_lines_ransac")
    ][["frame", "track_id", "pitch_x_pred", "pitch_y_pred"]]
    projection_lookup = {
        (int(row.frame), int(row.track_id)): (
            float(row.pitch_x_pred),
            float(row.pitch_y_pred),
        )
        for row in oracle.itertuples(index=False)
    }

    rows: list[dict[str, object]] = []
    for athlete in labels.athletes:
        if athlete.bbox_image is None:
            continue
        projection = projection_lookup.get((athlete.frame, athlete.track_id))
        if projection is None:
            continue
        pitch_x, pitch_y = projection
        rows.append(
            {
                "frame": athlete.frame - 1,
                "track_id": athlete.track_id,
                "class_name": _class_name(athlete.role),
                "team_id": _team_id(athlete.team),
                "x": athlete.bbox_image.x,
                "y": athlete.bbox_image.y,
                "width": athlete.bbox_image.w,
                "height": athlete.bbox_image.h,
                "pitch_x": pitch_x + 52.5,
                "pitch_y": 34.0 - pitch_y,
            }
        )
    return pd.DataFrame(rows)


def _class_name(role: str | None) -> str:
    if role == "goalkeeper":
        return "goalkeeper"
    if role == "referee":
        return "referee"
    if role == "ball":
        return "ball"
    return "player"


def _team_id(team: str | None) -> int | float:
    if team == "left":
        return 0
    if team == "right":
        return 1
    return np.nan


def _empty_annotations() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "frame",
            "track_id",
            "class_name",
            "team_id",
            "x",
            "y",
            "width",
            "height",
            "pitch_x",
            "pitch_y",
        ]
    )


if __name__ == "__main__":
    main()
