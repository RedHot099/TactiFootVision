# tactifoot_vision/export/pipeline_exporter.py
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import numpy as np
import pandas as pd
import supervision as sv
import math

logger = logging.getLogger(__name__)


def format_seconds_to_hmsms(timestamp_seconds: Optional[float]) -> Optional[str]:
    if timestamp_seconds is None or not math.isfinite(timestamp_seconds):
        return None
    try:
        total_seconds = timestamp_seconds
        hours = math.floor(total_seconds / 3600)
        minutes = math.floor((total_seconds % 3600) / 60)
        seconds = math.floor(total_seconds % 60)
        milliseconds = math.floor((total_seconds - math.floor(total_seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    except Exception:
        return None


class PipelineExporter:
    def __init__(self, class_id_to_name: Dict[int, str]):
        self.data_per_frame: List[Dict[str, Any]] = []
        self.class_id_to_name = class_id_to_name
        self.keeper_class_ids = {
            k for k, v in class_id_to_name.items() if v.lower() == "goalkeeper"
        }
        logger.info("PipelineExporter initialized.")

    def add_frame_data(
        self,
        frame_id: int,
        period: int,  # Added period
        tracked_detections: sv.Detections,
        pitch_coords: Optional[np.ndarray],
        ball_pitch_coords: Optional[np.ndarray],
        homography: Optional[np.ndarray] = None,
        visible_area: Optional[List[List[float]]] = None,
        timestamp_seconds: Optional[float] = None,
    ):
        formatted_timestamp = format_seconds_to_hmsms(timestamp_seconds)
        minute = (
            math.floor(timestamp_seconds / 60)
            if timestamp_seconds is not None
            else None
        )
        second = (
            math.floor(timestamp_seconds % 60)
            if timestamp_seconds is not None
            else None
        )

        frame_level_data: Dict[str, Any] = {
            "frame_id": frame_id,
            "period": period,  # Store period
            "timestamp": formatted_timestamp,
            "timestamp_seconds": timestamp_seconds,
            "minute": minute,
            "second": second,
            "freeze_frame": [],
            "homography_matrix": homography.tolist()
            if homography is not None
            else None,
            "visible_area": visible_area,
        }

        current_freeze_frame = []
        if pitch_coords is not None and len(tracked_detections) == len(pitch_coords):
            for i in range(len(tracked_detections)):
                track_id = (
                    int(tracked_detections.tracker_id[i])
                    if tracked_detections.tracker_id is not None
                    else -1
                )
                class_id = int(tracked_detections.class_id[i])
                class_name = self.class_id_to_name.get(class_id, f"unknown_{class_id}")
                team_id = None
                if "team_id" in tracked_detections.data:
                    raw_team = tracked_detections.data["team_id"][i]
                    team_id = int(raw_team) if raw_team is not None else None
                location = (
                    pitch_coords[i].tolist() if pitch_coords[i].size == 2 else None
                )
                bbox = tracked_detections.xyxy[i].astype(int).tolist()
                conf = (
                    float(tracked_detections.confidence[i])
                    if tracked_detections.confidence is not None
                    else None
                )

                if class_name == "goalkeeper":
                    type_name = "goalkeeper"
                elif class_name == "player":
                    type_name = "player"
                elif class_name == "referee":
                    type_name = "referee"
                else:
                    type_name = "other"

                current_freeze_frame.append(
                    {
                        "player_id": track_id,
                        "type": type_name,
                        "class_name": class_name,
                        "location": location,
                        "frame_bbox": bbox,
                        "confidence": conf,
                        "keeper": class_id in self.keeper_class_ids,
                        "teammate": None,
                        "team_id": team_id,
                        "actor": False,
                    }
                )
        elif len(tracked_detections) > 0:
            logger.warning(
                f"Frame {frame_id}: Mismatch detections/coords. Skipping player/ref export."
            )

        if ball_pitch_coords is not None and ball_pitch_coords.shape == (1, 2):
            current_freeze_frame.append(
                {
                    "player_id": -99,
                    "type": "ball",
                    "class_name": "ball",
                    "location": ball_pitch_coords[0].tolist(),
                    "frame_bbox": None,
                    "confidence": None,
                    "keeper": False,
                    "teammate": None,
                    "actor": False,
                }
            )

        frame_level_data["freeze_frame"] = current_freeze_frame
        self.data_per_frame.append(frame_level_data)

    def update_team_assignments(self, team_assignments: Dict[int, int]) -> None:
        if not team_assignments or not self.data_per_frame:
            return
        for frame_entry in self.data_per_frame:
            freeze = frame_entry.get("freeze_frame", [])
            if not freeze:
                continue
            for obj in freeze:
                if not isinstance(obj, dict):
                    continue
                player_id = obj.get("player_id")
                if player_id is None or player_id < 0:
                    continue
                if player_id in team_assignments:
                    obj["team_id"] = int(team_assignments[player_id])

    def save(self, output_csv_path: Union[str, Path]):
        output_path = Path(output_csv_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.data_per_frame:
            logger.warning("No frame data collected, skipping save.")
            return

        try:
            df = pd.DataFrame(self.data_per_frame)

            logger.info("Exploding freeze_frame data for row-per-object output...")
            df_exploded = df.explode("freeze_frame", ignore_index=True)
            frame_cols = [col for col in df.columns if col != "freeze_frame"]
            df_frame_data = df_exploded[frame_cols]
            df_object_data = (
                df_exploded["freeze_frame"]
                .apply(lambda x: pd.Series(x) if isinstance(x, dict) else pd.Series({}))
                .fillna("")
            )
            df_final = pd.concat([df_frame_data, df_object_data], axis=1)

            # Serialize complex columns that might remain after explode (less likely now)
            cols_to_serialize = [
                "homography_matrix",
                "visible_area",
                "frame_bbox",
                "location",
            ]
            for col in cols_to_serialize:
                if col in df_final.columns:
                    # Check if column actually contains lists/dicts before serializing
                    if df_final[col].apply(lambda x: isinstance(x, (list, dict))).any():
                        logger.debug(
                            f"Serializing column '{col}' to JSON string for CSV output."
                        )
                        df_final[col] = df_final[col].apply(
                            lambda x: json.dumps(x)
                            if isinstance(x, (list, dict))
                            else x
                        )

            logger.info(
                f"Saving pipeline output data ({len(df_final)} rows) to CSV: {output_path}"
            )
            df_final.to_csv(output_path, index=False, encoding="utf-8")
            logger.info("Save complete.")

        except Exception as e:
            logger.error(f"Failed to save pipeline output data: {e}", exc_info=True)
