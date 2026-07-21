# scripts/merge_pipeline_statsbomb.py
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional

# Add pyarrow imports
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    print("Warning: pyarrow not installed. Cannot save distances to Parquet format.")
    print("Install using: pip install pyarrow")


from loguru import logger
from tactifoot_vision.utils.logging_config import setup_logging

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


def parse_location(location_str):
    """Safely parses a location string (JSON list) into a NumPy array."""
    if pd.isna(location_str) or not isinstance(location_str, str):
        return None
    try:
        if location_str.startswith('"') and location_str.endswith('"'):
             location_str = location_str[1:-1].replace('\\"', '"')
        loc = json.loads(location_str)
        if isinstance(loc, list) and len(loc) == 2:
            return np.array(loc, dtype=float)
        else:
            # logger.trace(f"Parsed location is not a list of 2 elements: {loc}")
            return None
    except (json.JSONDecodeError, TypeError, ValueError):
        # logger.trace(f"Failed to parse location string: '{location_str}'. Error: {e}")
        return None

def calculate_distance(loc1, loc2):
    """Calculates Euclidean distance between two NumPy arrays."""
    if loc1 is None or loc2 is None:
        return np.inf
    try:
        arr1 = np.asarray(loc1)
        arr2 = np.asarray(loc2)
        if arr1.shape == (2,) and arr2.shape == (2,):
            dist = np.linalg.norm(arr1 - arr2)
            return dist if np.isfinite(dist) else np.inf
        else:
            # logger.trace(f"Invalid shapes for distance calc: {arr1.shape}, {arr2.shape}")
            return np.inf
    except Exception:
        # logger.trace(f"Error calculating distance between {loc1} and {loc2}: {e}")
        return np.inf

# Renamed function and changed plotting method
def plot_comparison_histogram(
    all_distances_dict: Dict[str, List[float]],
    output_path: Path,
    title: str = "Comparison of Positional Error Distributions",
    xlabel: str = "Euclidean Distance (Pitch Units)",
    ylabel: str = "Frequency (Number of Matches)", # Default Y label changed
    bins: int = 50, # Bins are now directly used by histplot
    xlim_max: Optional[float] = None,
):
    """
    Generates and saves a comparison histogram (frequency plot) using Seaborn.
    """
    if not all_distances_dict:
        logger.warning("No distance data provided for comparison plot. Skipping.")
        return

    logger.info(f"Generating comparison histogram for models: {list(all_distances_dict.keys())}")

    plot_data = []
    models_with_data = []
    for model_name, distances in all_distances_dict.items():
        finite_distances = [d for d in distances if np.isfinite(d)]
        if finite_distances:
            models_with_data.append(model_name)
            for distance in finite_distances:
                if distance >= 0:
                    plot_data.append({"model_name": model_name, "distance": distance})
        else:
            logger.warning(f"No valid finite distances found for model '{model_name}'. It will be excluded from the plot.")

    if not plot_data:
        logger.error("No valid finite distance data found across all models. Cannot generate plot.")
        return

    df_plot = pd.DataFrame(plot_data)

    try:
        plt.figure(figsize=(12, 7))

        # Use Seaborn's histplot with stat="count"
        plot_object = sns.histplot(
            data=df_plot,
            x="distance",
            hue="model_name",
            hue_order=models_with_data,
            bins=bins,           # Use specified bins
            stat="count",        # <<< Ensure Y-axis is count
            common_norm=False,   # Not relevant for count
            element="step",      # Use step for clearer overlay
            fill=False,          # No fill
            alpha=0.8,           # Alpha for lines
            linewidth=1.5,
        )

        plt.title(title, fontsize=16)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12) # Use the ylabel parameter
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        current_xlim = plt.xlim()
        new_xlim_min = min(0, current_xlim[0])
        new_xlim_max = xlim_max if xlim_max is not None else current_xlim[1]
        plt.xlim(new_xlim_min, new_xlim_max)

        handles, labels = plot_object.get_legend_handles_labels()
        if handles and labels and len(labels) > 1:
             if labels[0] == 'model_name':
                 handles = handles[1:]
                 labels = labels[1:]
             plt.legend(handles=handles, labels=labels, title='Model', title_fontsize='13', fontsize='11')
        else:
             logger.warning("Could not automatically generate legend items. Legend might be missing or incomplete.")

        plt.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        logger.success(f"Comparison histogram saved successfully to: {output_path}")

    except Exception as e:
        logger.error(f"Failed to generate or save comparison histogram: {e}", exc_info=True)

def main(
    statsbomb_csv_path: Path,
    model_data_list: List[Tuple[Path, Path, str]],
    output_comparison_plot_path: Path,
    output_distances_parquet_path: Path,
    # bw_adjust_plot: float = 1.5, # Removed bw_adjust
    xlim_max_plot: Optional[float] = None,
):
    try:
        logger.info("Starting Multi-Model StatsBomb/Pipeline Data Comparison")
        logger.info(f"StatsBomb input CSV: {statsbomb_csv_path}")
        logger.info(f"Output Comparison Plot: {output_comparison_plot_path}")
        logger.info(f"Output Distances Parquet: {output_distances_parquet_path}")
        logger.info(f"Processing {len(model_data_list)} models...")

        if not statsbomb_csv_path.is_file():
            raise FileNotFoundError(f"StatsBomb CSV not found: {statsbomb_csv_path}")

        logger.info("Loading StatsBomb data...")
        df_statsbomb_raw = pd.read_csv(statsbomb_csv_path)
        logger.info(f"Loaded {len(df_statsbomb_raw)} rows from StatsBomb data.")

        logger.info("Preprocessing StatsBomb data...")
        required_sb_cols = ["period", "minute", "second", "pitch_location", "type"]
        if not all(col in df_statsbomb_raw.columns for col in required_sb_cols):
            raise ValueError(f"StatsBomb CSV missing required columns: {required_sb_cols}")

        for col in ["period", "minute", "second"]:
            df_statsbomb_raw[col] = pd.to_numeric(df_statsbomb_raw[col], errors="coerce")
        df_statsbomb_raw.dropna(subset=["period", "minute", "second"], inplace=True)
        for col in ["period", "minute", "second"]:
            df_statsbomb_raw[col] = df_statsbomb_raw[col].astype(int)

        df_statsbomb_raw["parsed_location"] = df_statsbomb_raw["pitch_location"].apply(parse_location)
        df_statsbomb_filtered = df_statsbomb_raw.dropna(subset=["parsed_location"]).copy()
        logger.info(f"StatsBomb rows with valid locations after preprocessing: {len(df_statsbomb_filtered)}")

        if df_statsbomb_filtered.empty:
            logger.error("No valid StatsBomb data after preprocessing. Exiting.")
            sys.exit(1)

        all_distances_dict: Dict[str, List[float]] = {}
        all_distances_for_parquet: List[Dict] = []

        for p1_path, p2_path, model_name in model_data_list:
            logger.info(f"--- Processing Model: {model_name} ---")
            logger.info(f"  P1 File: {p1_path}")
            logger.info(f"  P2 File: {p2_path}")

            if not p1_path.is_file():
                logger.error(f"Pipeline P1 CSV not found for model {model_name}: {p1_path}")
                continue
            if not p2_path.is_file():
                logger.error(f"Pipeline P2 CSV not found for model {model_name}: {p2_path}")
                continue

            try:
                df_pipeline_p1 = pd.read_csv(p1_path)
                df_pipeline_p2 = pd.read_csv(p2_path)
                df_detection_raw = pd.concat([df_pipeline_p1, df_pipeline_p2], ignore_index=True)
                logger.info(f"  Loaded and combined {len(df_detection_raw)} rows for {model_name}.")
            except Exception as e:
                logger.error(f"  Failed to load or concat pipeline data for {model_name}: {e}")
                continue

            required_det_cols = ["period", "minute", "second", "location", "type", "player_id"]
            if not all(col in df_detection_raw.columns for col in required_det_cols):
                logger.error(f"  Detection CSV for {model_name} missing required columns: {required_det_cols}")
                continue

            for col in ["period", "minute", "second"]:
                df_detection_raw[col] = pd.to_numeric(df_detection_raw[col], errors="coerce")
            df_detection_raw.dropna(subset=["period", "minute", "second"], inplace=True)
            for col in ["period", "minute", "second"]:
                df_detection_raw[col] = df_detection_raw[col].astype(int)

            df_detection_raw["parsed_location"] = df_detection_raw["location"].apply(parse_location)
            df_detection_filtered = df_detection_raw.dropna(subset=["parsed_location"]).copy()
            logger.info(f"  Detection rows with valid locations for {model_name}: {len(df_detection_filtered)}")

            if df_detection_filtered.empty:
                logger.warning(f"  No valid detection data for model {model_name} after preprocessing.")
                all_distances_dict[model_name] = []
                continue

            logger.info(f"  Finding matches for {model_name}...")
            model_distances = []
            match_count = 0
            detection_groups = df_detection_filtered.groupby(["period", "minute", "second"])
            df_sb_temp = df_statsbomb_filtered.copy()

            for index, sb_row in tqdm(
                df_sb_temp.iterrows(),
                total=len(df_sb_temp),
                desc=f"Matching {model_name}",
                leave=False
            ):
                timestamp = (sb_row["period"], sb_row["minute"], sb_row["second"])
                sb_loc = sb_row["parsed_location"]
                sb_type = str(sb_row["type"]).lower()

                if sb_type in ['ball', 'other', 'bad behaviour', '50/50']:
                    continue

                if timestamp in detection_groups.groups:
                    potential_matches = detection_groups.get_group(timestamp)
                    dist_series = potential_matches["parsed_location"].apply(
                        lambda det_loc: calculate_distance(sb_loc, det_loc)
                    )

                    if not dist_series.empty:
                        min_dist = dist_series.min()
                        if np.isfinite(min_dist):
                            model_distances.append(min_dist)
                            all_distances_for_parquet.append({
                                "model_name": model_name,
                                "distance": min_dist,
                                "period": sb_row["period"],
                                "minute": sb_row["minute"],
                                "second": sb_row["second"],
                            })
                            match_count += 1

            logger.info(f"  Found {match_count} matches for {model_name}.")
            all_distances_dict[model_name] = model_distances

        logger.info("--- Generating Final Comparison Plot ---")
        if not all_distances_dict:
             logger.error("No distance data collected for any model. Cannot generate plot.")
        else:
            plot_comparison_histogram( # Use renamed function
                all_distances_dict=all_distances_dict,
                output_path=output_comparison_plot_path,
                title="Comparison of Model Positional Error Distributions",
                xlabel="Positional Error (Euclidean Distance, Pitch Units)",
                ylabel="Frequency (Number of Matches)", # Pass correct label
                # bins=50, # Optional: Adjust number of bins
                xlim_max=xlim_max_plot
            )

        logger.info("--- Saving Distances to Parquet File ---")
        if not PYARROW_AVAILABLE:
            logger.warning("pyarrow not found. Skipping saving distances to Parquet.")
        elif not all_distances_for_parquet:
            logger.warning("No distance data collected for any model. Skipping saving to Parquet.")
        else:
            try:
                df_distances = pd.DataFrame(all_distances_for_parquet)
                logger.info(f"Saving {len(df_distances)} distance records to Parquet: {output_distances_parquet_path}")
                output_distances_parquet_path.parent.mkdir(parents=True, exist_ok=True)
                df_distances.to_parquet(output_distances_parquet_path, index=False)
                logger.success("Distances saved successfully to Parquet.")
            except Exception as e:
                logger.error(f"Failed to save distances to Parquet: {e}", exc_info=True)

        logger.success("Multi-model comparison script finished.")

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
        logger.exception("An unexpected error occurred during the comparison process.")
        sys.exit(1)


if __name__ == "__main__":
    # --- Define Model Data and Paths ---
    # !! MODIFY THIS LIST WITH YOUR ACTUAL DATA !!
    MODELS_TO_COMPARE: List[Tuple[Path, Path, str]] = [
        (
            Path("./results/output/csv/csv/tactifoot_vision../output/rfdetr/inference_vid1/video_output_p1_pipelinedata_p1.csv"),
            Path("./results/output/csv/csv/tactifoot_vision../output/rfdetr/inference_vid2/video_output_p2_pipelinedata_p2.csv"),
            "RFDETR"
        ),
        (
            Path("./results/output/csv/csv/tactifoot_vision../output/yolo11x/inference_vid1/video_output_p1_pipelinedata_p1.csv"),
            Path("./results/output/csv/csv/tactifoot_vision../output/yolo11x/inference_vid2/video_output_p2_pipelinedata_p2.csv"),
            "YOLOv11x"
        ),
        (
            Path("./results/output/csv/csv/tactifoot_vision../output/yolo12x/inference_vid1/video_output_p1_pipelinedata_p1.csv"),
            Path("./results/output/csv/csv/tactifoot_vision../output/yolo12x/inference_vid2/video_output_p2_pipelinedata_p2.csv"),
            "YOLOv12x"
        ),
        (
            Path("./results/output/csv/csv/tactifoot_vision../output/yolov8x/inference_vid1/video_output_p1_pipelinedata_p1.csv"),
            Path("./results/output/csv/csv/tactifoot_vision../output/yolov8x/inference_vid2/video_output_p2_pipelinedata_p2.csv"),
            "YOLOv8x"
        ),
    ]

    STATS_BOMB_FILE = Path("./data/statsbomb/statsbomb.csv")
    OUTPUT_PLOT_FILE = Path("./data/output/model_comparison_histogram.png") # Updated name
    OUTPUT_PARQUET_FILE = Path("./data/output/model_distances.parquet")
    X_LIMIT_MAX = 30.0
    # BW_ADJUST = 1.5 # Removed - not used by histplot
    LOGGING_LEVEL = "INFO"
    # ------------------------------------

    setup_logging(level=LOGGING_LEVEL)

    if not PYARROW_AVAILABLE:
         logger.error("Exiting script because pyarrow is required for saving Parquet files.")
         sys.exit(1)

    main(
        statsbomb_csv_path=STATS_BOMB_FILE,
        model_data_list=MODELS_TO_COMPARE,
        output_comparison_plot_path=OUTPUT_PLOT_FILE,
        output_distances_parquet_path=OUTPUT_PARQUET_FILE,
        # bw_adjust_plot=BW_ADJUST, # Removed
        xlim_max_plot=None
    )
