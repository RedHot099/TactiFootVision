import re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# --- Data Loading and Processing ---
# This section is adapted from the existing `scripts/plot_team_classification_publication.py`
# to ensure data is loaded and processed consistently.

RESULTS_DIR = Path("results/team_classification")
NUMERIC_DIR = RESULTS_DIR / "numeric"

def _infer_clustering_method(path: Path) -> str | None:
    name = path.name.lower()
    if "kmeans" in name:
        return "kmeans"
    if "dbscan" in name:
        return "dbscan"
    if "cmeans" in name:
        return "cmeans"
    return None

def _load_metrics() -> pd.DataFrame:
    # Using the file list from the successful `find` command
    file_paths = [
        NUMERIC_DIR / "team_classification_metrics.csv",
        NUMERIC_DIR / "team_classification_metrics_kmeans.csv",
        NUMERIC_DIR / "team_classification_metrics_dbscan.csv",
        NUMERIC_DIR / "team_classification_metrics_aug.csv",
        NUMERIC_DIR / "team_classification_metrics_full_sam2.csv",
        NUMERIC_DIR / "team_classification_metrics_full_sam2_resnet.csv",
    ]
    paths = [Path(p) for p in file_paths if Path(p).exists()]
    if not paths:
        raise FileNotFoundError("Could not find any of the expected metrics files.")

    frames: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_csv(path)
        df["source_file"] = path.name

        clustering_method = _infer_clustering_method(path)
        if clustering_method is not None:
            df["clustering_method"] = clustering_method

        if "embedding_backend" not in df.columns:
            df["embedding_backend"] = "clip"
        else:
            df["embedding_backend"] = df["embedding_backend"].astype(str).str.lower()

        df["crop_method"] = df["crop_method"].astype(str).str.lower()
        df["color_space"] = df["color_space"].astype(str).str.lower()
        df["umap_applied"] = df["umap_applied"].astype(bool)

        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    return data

# --- Dumbbell Plot Generation ---

def plot_dumbbell_with_delta(df: pd.DataFrame):
    # 1. Prepare data for dumbbells
    df_filtered = df[df['color_space'].isin(['rgb', 'h'])].copy()

    identifier_cols = [
        'source_file', 'embedding_backend', 'crop_method', 'crop_center_ratio',
        'umap_applied', 'clustering_method', 'experiment_name'
    ]
    identifier_cols = [col for col in identifier_cols if col in df_filtered.columns]

    df_pivot = df_filtered.pivot_table(
        index=identifier_cols,
        columns='color_space',
        values='accuracy'
    ).reset_index()

    df_pivot.rename(columns={'rgb': 'RGB', 'h': 'Hue'}, inplace=True)
    df_pivot.dropna(subset=['RGB', 'Hue'], inplace=True)

    if df_pivot.empty:
        print("No paired RGB and Hue data found to plot.")
        return

    # Sort by RGB - Hue difference, descending
    df_pivot['diff'] = df_pivot['RGB'] - df_pivot['Hue']
    df_pivot = df_pivot.sort_values('diff', ascending=False, ignore_index=True)

    # 2. Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(
        figsize=(14, 10),
        ncols=2,
        sharey=True,
        gridspec_kw={'width_ratios': [2.5, 1]}
    )
    fig.suptitle('Effect of Color Space on Accuracy (RGB vs. Hue)', fontsize=20)

    # -- LEFT PLOT: Dumbbell --
    ax1.scatter(df_pivot['RGB'], df_pivot.index, color='#1f77b4', s=60, label='RGB', zorder=3)
    ax1.scatter(df_pivot['Hue'], df_pivot.index, color='#ff7f0e', s=60, label='Hue', zorder=3)
    ax1.plot([df_pivot['RGB'], df_pivot['Hue']], [df_pivot.index, df_pivot.index],
             color='grey', linewidth=1.5, zorder=1, solid_capstyle='round')

    ax1.set_xlabel('Accuracy', fontsize=14)
    ax1.set_ylabel('Matches', fontsize=14)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.grid(axis='y', linestyle='-', linewidth=0.3)
    ax1.grid(axis='x', linestyle='--', linewidth=0.5)
    ax1.legend(loc='lower right', fontsize=12, frameon=True, shadow=True)
    ax1.set_title('Accuracy Comparison', fontsize=16, pad=15)
    
    # -- RIGHT PLOT: Delta --
    colors = np.where(df_pivot['diff'] > 0, '#1f77b4', '#ff7f0e')
    ax2.scatter(df_pivot['diff'], df_pivot.index, color=colors, s=60)
    ax2.axvline(0, color='grey', linestyle='--', linewidth=1.5)
    
    ax2.set_xlabel('Delta Accuracy (RGB - Hue)', fontsize=14)
    ax2.tick_params(axis='x', labelsize=12)
    ax2.grid(axis='y', linestyle='-', linewidth=0.3)
    ax2.grid(axis='x', linestyle='--', linewidth=0.5)
    ax2.set_title('Effect Size', fontsize=16, pad=15)

    # General aesthetics
    ax1.set_yticks([])
    ax2.set_yticks([])
    sns.despine(left=True, bottom=False)

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make room for suptitle
    
    # Save the plot
    out_dir = RESULTS_DIR / "plots" / "plots_classification_improved"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "dumbbell_color_space_comparison_v2.png"
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    import seaborn as sns
    try:
        data = _load_metrics()
        plot_dumbbell_with_delta(data)
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
