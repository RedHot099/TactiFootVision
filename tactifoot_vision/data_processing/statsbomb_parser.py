# tactifoot_vision/data_processing/statsbomb_parser.py
import json
import logging
from pathlib import Path
from typing import Union, Optional
import pandas as pd
import numpy as np

# import argparse # Removed argparse
import math

logger = logging.getLogger(__name__)  # Use logger defined at module level

FINAL_COLUMNS_TO_KEEP = [
    "event_uuid",
    "index",
    "period",
    "timestamp",
    "timestamp_seconds",
    "minute",
    "second",
    "type_id",
    "type_name",
    "possession",
    "possession_team_id",
    "possession_team_name",
    "play_pattern_id",
    "play_pattern_name",
    "team_id",
    "team_name",
    "player_id",
    "player_name",
    "pitch_location",
    "teammate",
    "actor",
    "keeper",
    "type",
    "visible_area",
    "under_pressure",  # Added under_pressure back if needed
]


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


def load_statsbomb_match_data(  # Renamed function
    json_path_or_dir: Union[str, Path], output_csv_path: Union[str, Path]
) -> Optional[pd.DataFrame]:
    input_path = Path(json_path_or_dir)
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_event_data = []
    df_intermediate = None

    try:
        if input_path.is_file() and input_path.suffix.lower() == ".json":
            logger.info(f"Loading combined data from single file: {input_path}")
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                all_event_data = data
            else:
                logger.error(f"Expected list in {input_path}")
                return None
            logger.info(f"Loaded {len(all_event_data)} events.")
            if all_event_data:
                df_intermediate = pd.DataFrame(all_event_data)
                if (
                    "id" in df_intermediate.columns
                    and "event_uuid" not in df_intermediate.columns
                ):
                    logger.warning("Renaming 'id' to 'event_uuid'.")
                    df_intermediate.rename(columns={"id": "event_uuid"}, inplace=True)

        elif input_path.is_dir():
            event_files = list(input_path.glob("*_events.json"))
            three_sixty_files = list(input_path.glob("*_360.json"))
            if not event_files:
                logger.error(f"No event JSON files found: {input_path}")
                return None
            if not three_sixty_files:
                logger.error(f"No 360 JSON files found: {input_path}")
                return None  # Require both for this workflow

            all_events_list = []
            for event_file in event_files:
                logger.info(f"Loading event time data from: {event_file}")
                with open(event_file, "r", encoding="utf-8") as f:
                    events = json.load(f)
                if isinstance(events, list):
                    all_events_list.extend(events)
            if not all_events_list:
                logger.error("No event data loaded.")
                return None
            df_events_raw = pd.DataFrame(all_events_list)
            logger.info(f"Loaded {len(df_events_raw)} total event entries.")

            if "id" in df_events_raw.columns:
                df_events_raw.rename(columns={"id": "event_uuid"}, inplace=True)
            time_cols = [
                "event_uuid",
                "timestamp",
                "minute",
                "second",
                "period",
                "type_name",
                "team_id",
                "player_id",
                "player_name",
                "under_pressure",
                "possession",
                "possession_team_id",
                "possession_team_name",
                "play_pattern_id",
                "play_pattern_name",
                "index",
                "type_id",
            ]  # Added more context cols
            cols_to_extract = [col for col in time_cols if col in df_events_raw.columns]
            if "event_uuid" not in cols_to_extract:
                logger.error("Event data missing 'event_uuid'.")
                return None
            df_event_times = df_events_raw[cols_to_extract].copy()
            if (
                "timestamp" in df_event_times.columns
                and df_event_times["timestamp"].dtype == "object"
            ):
                try:
                    df_event_times["timestamp_seconds"] = pd.to_datetime(
                        df_event_times["timestamp"], format="%H:%M:%S.%f"
                    ).dt.time.apply(
                        lambda t: t.hour * 3600
                        + t.minute * 60
                        + t.second
                        + t.microsecond / 1e6
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not parse timestamp string: {e}. Keeping original."
                    )
                    df_event_times["timestamp_seconds"] = None
            elif "timestamp" in df_event_times.columns:
                df_event_times["timestamp_seconds"] = pd.to_numeric(
                    df_event_times["timestamp"], errors="coerce"
                )
            else:
                df_event_times["timestamp_seconds"] = None

            all_360_list = []
            for t60_file in three_sixty_files:
                logger.info(f"Loading 360 data from: {t60_file}")
                with open(t60_file, "r", encoding="utf-8") as f:
                    t60_data = json.load(f)
                if isinstance(t60_data, list):
                    all_360_list.extend(t60_data)
            if not all_360_list:
                logger.error("No 360 data loaded.")
                return None
            df_360 = pd.DataFrame(all_360_list)
            logger.info(f"Loaded {len(df_360)} 360 entries.")
            if "event_uuid" not in df_360.columns:
                logger.error("360 data missing 'event_uuid'.")
                return None
            if "freeze_frame" not in df_360.columns:
                logger.error("360 data missing 'freeze_frame'.")
                return None

            logger.info("Exploding freeze_frame data...")
            df_360_exploded = df_360.dropna(subset=["freeze_frame"]).explode(
                "freeze_frame", ignore_index=True
            )
            if df_360_exploded.empty:
                logger.error("No valid freeze_frame data after dropna/explode.")
                return None

            logger.info("Expanding freeze_frame dictionary...")
            base_cols = [col for col in df_360.columns if col != "freeze_frame"]
            df_base_data = df_360_exploded[base_cols]
            df_object_data = (
                df_360_exploded["freeze_frame"]
                .apply(lambda x: pd.Series(x) if isinstance(x, dict) else pd.Series({}))
                .fillna("")
            )
            df_expanded = pd.concat(
                [
                    df_base_data.reset_index(drop=True),
                    df_object_data.reset_index(drop=True),
                ],
                axis=1,
            )

            if "player" in df_expanded.columns:
                player_info = df_expanded["player"].apply(
                    lambda x: pd.Series(x)
                    if isinstance(x, dict)
                    else pd.Series({"id": None, "name": None})
                )
                df_expanded["ff_player_id"] = player_info["id"]
                df_expanded["ff_player_name"] = player_info["name"]
                df_expanded = df_expanded.drop(columns=["player"])

            if "location" in df_expanded.columns:
                df_expanded.rename(columns={"location": "pitch_location"}, inplace=True)

            if "keeper" in df_expanded.columns:
                df_expanded["type"] = np.where(
                    df_expanded["keeper"] == True, "goalkeeper", "player"
                )
            else:
                df_expanded["type"] = "player"
            logger.info(f"Freeze frame expanded. Result has {len(df_expanded)} rows.")

            logger.info("Merging event time data...")
            merge_cols = [
                "event_uuid",
                "timestamp",
                "timestamp_seconds",
                "minute",
                "second",
                "period",
                "type_name",
                "team_id",
                "under_pressure",
                "possession",
                "possession_team_id",
                "possession_team_name",
                "play_pattern_id",
                "play_pattern_name",
                "index",
                "type_id",
            ]
            # Add event player details only if not present from freeze frame
            if (
                "ff_player_id" not in df_expanded.columns
                and "player_id" in df_event_times.columns
            ):
                merge_cols.append("player_id")
            if (
                "ff_player_name" not in df_expanded.columns
                and "player_name" in df_event_times.columns
            ):
                merge_cols.append("player_name")

            cols_to_merge_existing = [
                col for col in merge_cols if col in df_event_times.columns
            ]
            df_intermediate = pd.merge(
                df_expanded,
                df_event_times[cols_to_merge_existing],
                on="event_uuid",
                how="left",
            )

            if "ff_player_id" in df_intermediate.columns:
                df_intermediate.rename(
                    columns={"ff_player_id": "player_id"}, inplace=True
                )
            if "ff_player_name" in df_intermediate.columns:
                df_intermediate.rename(
                    columns={"ff_player_name": "player_name"}, inplace=True
                )
            logger.info(f"Time data merged. DataFrame shape: {df_intermediate.shape}")

        else:
            logger.error(f"Invalid input path: {input_path}.")
            return None

        if df_intermediate is None or df_intermediate.empty:
            logger.warning("No data loaded or merged.")
            return None

        if "timestamp_seconds" in df_intermediate.columns:
            df_intermediate["timestamp"] = df_intermediate["timestamp_seconds"].apply(
                format_seconds_to_hmsms
            )
        elif "timestamp" in df_intermediate.columns:
            df_intermediate["timestamp_seconds"] = None

        final_cols_exist = [
            col for col in FINAL_COLUMNS_TO_KEEP if col in df_intermediate.columns
        ]
        df_final = df_intermediate[final_cols_exist].copy()
        logger.info(f"Filtered final columns: {final_cols_exist}")

        # TODO: Implement coordinate conversion if needed

        cols_to_serialize = ["visible_area", "pitch_location"]
        for col in cols_to_serialize:
            if col in df_final.columns:
                if df_final[col].apply(lambda x: isinstance(x, list)).any():
                    logger.debug(f"Serializing column '{col}' to JSON string.")
                    df_final[col] = df_final[col].apply(
                        lambda x: json.dumps(x) if isinstance(x, list) else x
                    )

        logger.info(f"Saving final processed data to CSV: {output_path}")
        df_final.to_csv(output_path, index=False, encoding="utf-8")
        logger.info("Save complete.")
        return df_final

    except FileNotFoundError:
        logger.error(f"Input not found: {input_path}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    # --- Hardcoded paths and logging setup ---
    input_path = Path("./data/statsbomb")  # Or directory path
    output_path = Path("./output/statsbomb_merged.csv")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    # -----------------------------------------

    if input_path.exists():
        # Use the renamed function load_and_process_statsbomb
        merged_df = load_statsbomb_match_data(input_path, output_path)
        if merged_df is not None:
            print(f"\nSuccessfully created {output_path}")
            # print(merged_df.info()) # Optional: print info
    else:
        print(f"Input path not found: {input_path}")
