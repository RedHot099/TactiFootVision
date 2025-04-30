# scripts/merge_pipeline_statsbomb.py
import argparse
import sys
import json
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from loguru import logger
from config.loaders import load_config
from tactifoot_vision.utils.logging_config import setup_logging

# Assuming config and utils are accessible from the project root
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))


def parse_location(location_str):
    """Safely parses a location string (JSON list) into a NumPy array."""
    if pd.isna(location_str) or not isinstance(location_str, str):
        return None
    try:
        loc = json.loads(location_str)
        if isinstance(loc, list) and len(loc) == 2:
            return np.array(loc, dtype=float)
        else:
            logger.trace(f"Parsed location is not a list of 2 elements: {loc}")
            return None
    except (json.JSONDecodeError, TypeError):
        logger.trace(f"Failed to parse location string: {location_str}")
        return None


def calculate_distance(loc1, loc2):
    """Calculates Euclidean distance between two NumPy arrays."""
    if loc1 is None or loc2 is None:
        return np.inf
    try:
        # Ensure they are numpy arrays for vectorized operation
        arr1 = np.asarray(loc1)
        arr2 = np.asarray(loc2)
        if arr1.shape == (2,) and arr2.shape == (2,):
            dist = np.linalg.norm(arr1 - arr2)
            return dist if np.isfinite(dist) else np.inf
        else:
            logger.trace(
                f"Invalid shapes for distance calc: {arr1.shape}, {arr2.shape}"
            )
            return np.inf
    except Exception as e:
        logger.trace(f"Error calculating distance between {loc1} and {loc2}: {e}")
        return np.inf

def plot_distance_histogram(
    distances: list[float],
    output_path: Path,
    title: str = "Distribution of Euclidean Distances",
    xlabel: str = "Euclidean Distance",
    ylabel: str = "Frequency",
    bins: int = 50,
):
    """
    Generates and saves a histogram of the provided distances.

    Args:
        distances: A list of float values representing the distances.
        output_path: The Path object where the histogram image will be saved.
        title: The title for the histogram plot.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        bins: The number of bins to use in the histogram.
    """
    if not distances:
        logger.warning("No distances provided to plot_distance_histogram. Skipping.")
        return

    logger.info(f"Generating distance histogram with {len(distances)} values...")
    try:
        plt.figure(figsize=(10, 6))  # Create a new figure
        plt.hist(distances, bins=bins, color="skyblue", edgecolor="black")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(axis="y", alpha=0.75)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()  # Close the figure to free memory
        logger.success(f"Histogram saved successfully to: {output_path}")

    except Exception as e:
        logger.error(f"Failed to generate or save histogram: {e}", exc_info=True)


def main(config_path: Path):
    try:
        # --- 1. Setup & Configuration ---
        if not config_path.is_absolute():
            config_path = (Path.cwd() / config_path).resolve()
        # The loader now resolves paths relative to the config file
        config = load_config(config_path)

        setup_logging(level=config.logging_level)
        logger.info(
            f"Starting StatsBomb and Pipeline Data Merge: {config.project_name}"
        )
        logger.debug(f"Config loaded from: {config_path}")

        target_period = config.processing.period
        logger.info(f"Processing data for period: {target_period}")

        # --- Use configured simple paths ---
        # Paths are already resolved Path objects by the loader
        statsbomb_csv_path = config.paths.statsbomb_input_csv
        detection_csv_path = config.paths.pipeline_input_csv
        output_csv_path = config.paths.merged_output_csv
        # --- End path configuration usage ---

        logger.info(f"StatsBomb input CSV: {statsbomb_csv_path}")
        logger.info(f"Detection input CSV: {detection_csv_path}")
        logger.info(f"Merged output CSV: {output_csv_path}")

        # --- Important Note ---
        logger.warning(
            f"Ensure the file at '{detection_csv_path}' contains the data "
            f"corresponding to the configured period ({target_period})."
        )
        # --------------------

        if not statsbomb_csv_path.is_file():
            raise FileNotFoundError(f"StatsBomb CSV not found: {statsbomb_csv_path}")
        if not detection_csv_path.is_file():
            raise FileNotFoundError(f"Detection CSV not found: {detection_csv_path}")

        # --- 2. Data Loading ---
        logger.info("Loading CSV files into DataFrames...")
        df_statsbomb = pd.read_csv(statsbomb_csv_path)
        df_detection = pd.read_csv(detection_csv_path)
        logger.info(f"Loaded {len(df_statsbomb)} rows from StatsBomb data.")
        logger.info(f"Loaded {len(df_detection)} rows from Detection pipeline data.")

        # --- 3. Preprocessing ---
        logger.info("Preprocessing DataFrames...")

        # Ensure required columns exist
        required_sb_cols = ["period", "minute", "second", "pitch_location", "type"]
        required_det_cols = [
            "period",
            "minute",
            "second",
            "location",
            "type",
            "frame_bbox",
            "confidence",
            "visible_area",
            "player_id",  # Keep track of the matched player/object ID
        ]
        print(df_statsbomb.columns)
        if not all(col in df_statsbomb.columns for col in required_sb_cols):
            raise ValueError(
                f"StatsBomb CSV missing required columns: {required_sb_cols}"
            )
        if not all(col in df_detection.columns for col in required_det_cols):
            raise ValueError(
                f"Detection CSV missing required columns: {required_det_cols}"
            )

        # Convert timestamp columns to numeric, coercing errors
        for df in [df_statsbomb, df_detection]:
            for col in ["period", "minute", "second"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Filter by period *after* converting to numeric
        df_statsbomb = df_statsbomb[df_statsbomb["period"] == target_period].copy()
        df_detection = df_detection[df_detection["period"] == target_period].copy()

        # Drop rows where timestamp conversion failed
        df_statsbomb.dropna(subset=["period", "minute", "second"], inplace=True)
        df_detection.dropna(subset=["period", "minute", "second"], inplace=True)

        # Convert timestamp columns to integers
        for df in [df_statsbomb, df_detection]:
            for col in ["period", "minute", "second"]:
                df[col] = df[col].astype(int)

        logger.info(
            f"Filtered StatsBomb data for period {target_period}: {len(df_statsbomb)} rows"
        )
        logger.info(
            f"Filtered Detection data for period {target_period}: {len(df_detection)} rows"
        )

        # Parse location columns
        df_statsbomb["parsed_location"] = df_statsbomb["pitch_location"].apply(
            parse_location
        )
        df_detection["parsed_location"] = df_detection["location"].apply(parse_location)

        # Filter out rows where location parsing failed or is missing
        df_statsbomb_filtered = df_statsbomb.dropna(subset=["parsed_location"]).copy()
        df_detection_filtered = df_detection.dropna(subset=["parsed_location"]).copy()

        logger.info(
            f"StatsBomb rows with valid locations: {len(df_statsbomb_filtered)}"
        )
        logger.info(
            f"Detection rows with valid locations: {len(df_detection_filtered)}"
        )

        if df_statsbomb_filtered.empty or df_detection_filtered.empty:
            logger.warning("One or both DataFrames are empty after filtering. Exiting.")
            sys.exit(0)

        # --- 4. Matching Logic ---
        logger.info("Finding closest matches...")
        output_columns = [
            "detected_location",
            "detected_type",
            "detected_player_id",
            "detected_frame_bbox",
            "detected_confidence",
            "detected_visible_area",
            "euclidean_distance",
        ]
        for col in output_columns:
            # Initialize with appropriate dtype to avoid object conversion later if possible
            if "distance" in col or "confidence" in col:
                df_statsbomb_filtered[col] = (
                    pd.NA
                )  # Use pandas NA for float compatibility
            elif "player_id" in col:
                df_statsbomb_filtered[col] = pd.NA  # Int64 allows NA
            else:
                df_statsbomb_filtered[col] = pd.NA  # Object type for strings/lists

        match_count = 0
        distances = []

        # Group detection data by timestamp for faster lookup
        detection_groups = df_detection_filtered.groupby(["period", "minute", "second"])

        for index, sb_row in tqdm(
            df_statsbomb_filtered.iterrows(),
            total=len(df_statsbomb_filtered),
            desc="Matching rows",
        ):
            timestamp = (sb_row["period"], sb_row["minute"], sb_row["second"])
            sb_loc = sb_row["parsed_location"]

            if timestamp in detection_groups.groups:
                potential_matches = detection_groups.get_group(timestamp)

                min_dist = np.inf
                best_match_idx = None

                # Calculate distances for all potential matches at this timestamp
                dist_series = potential_matches["parsed_location"].apply(
                    lambda det_loc: calculate_distance(sb_loc, det_loc)
                )

                if not dist_series.empty:
                    min_dist = dist_series.min()
                    if np.isfinite(min_dist):
                        best_match_idx = (
                            dist_series.idxmin()
                        )  # Index in df_detection_filtered

                if best_match_idx is not None:
                    match_row = df_detection_filtered.loc[best_match_idx]

                    # Assign matched data using .loc
                    df_statsbomb_filtered.loc[index, "detected_location"] = match_row[
                        "location"
                    ]  # Store original string
                    df_statsbomb_filtered.loc[index, "detected_type"] = match_row[
                        "type"
                    ]
                    df_statsbomb_filtered.loc[index, "detected_player_id"] = match_row[
                        "player_id"
                    ]
                    df_statsbomb_filtered.loc[index, "detected_frame_bbox"] = match_row[
                        "frame_bbox"
                    ]
                    df_statsbomb_filtered.loc[index, "detected_confidence"] = match_row[
                        "confidence"
                    ]
                    df_statsbomb_filtered.loc[index, "detected_visible_area"] = (
                        match_row["visible_area"]
                    )
                    df_statsbomb_filtered.loc[index, "euclidean_distance"] = min_dist

                    match_count += 1
                    distances.append(min_dist)

        # --- 5. Analysis & Output ---
        logger.info("Matching complete.")
        if match_count > 0:
            mean_distance = np.mean(distances)
            logger.success(f"Successfully matched {match_count} StatsBomb rows.")
            logger.success(f"Mean Euclidean distance for matches: {mean_distance:.4f}")

            # --- Generate and save histogram ---
            try:
                # Define path for the histogram image (same dir as CSV, different extension)
                histogram_output_path = output_csv_path.with_suffix(".png")
                plot_distance_histogram(
                    distances=distances,  # Pass the collected distances
                    output_path=histogram_output_path,
                    title=f"Distribution of Positional Errors (Period {target_period})",
                    xlabel="Euclidean Distance (Pitch Units)",  # Adjust label if units are known (e.g., meters, cm)
                    ylabel="Frequency (Number of Matches)",
                    bins=50,  # Adjust number of bins if needed
                )
            except Exception as hist_err:
                # Log error but don't stop the script from saving the CSV
                logger.error(f"Could not generate histogram: {hist_err}", exc_info=True)
            # ------------------------------------

        else:
            logger.warning("No matches found between the two datasets for the period.")
            logger.warning(
                "Skipping histogram generation as there are no distances."
            )  # Added warning

        # Drop the temporary parsed location column before saving
        df_statsbomb_filtered.drop(columns=["parsed_location"], inplace=True)

        logger.info(f"Saving merged data to: {output_csv_path}")
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_statsbomb_filtered.to_csv(output_csv_path, index=False, encoding="utf-8")
        logger.success("Merged data saved successfully.")

    except FileNotFoundError as e:
        logger.error(f"File not found error: {e}", exc_info=True)
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Data validation or value error: {e}", exc_info=True)
        sys.exit(1)
    except KeyError as e:
        logger.error(f"Missing expected column in CSV: {e}", exc_info=True)
        sys.exit(1)
    except Exception:
        logger.exception("An unexpected error occurred during the merge process.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge StatsBomb data with Detection Pipeline output."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "config" / "default_config.yaml",
        help="Path to the main configuration YAML file.",
    )
    args = parser.parse_args()
    main(args.config)
