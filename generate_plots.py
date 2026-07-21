
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# General Style Guidelines
sns.set_theme(style="whitegrid")
palette = "viridis"
title_fontsize = 16
label_fontsize = 12
dpi = 300

RESULTS_DIR = Path("results/detection_tracking")
DATA_DIR = RESULTS_DIR / "raw" / "soccernet_tracking_2023_tiny_seg"
OUT_DIR = RESULTS_DIR / "plots" / "soccernet_tracking_2023_tiny_seg"

# Data Loading and Preprocessing
try:
    summary_df = pd.read_csv(DATA_DIR / "summary.csv")
    metrics_per_class_df = pd.read_csv(DATA_DIR / "metrics_per_class.csv")
    per_sequence_stats_df = pd.read_csv(DATA_DIR / "per_sequence_stats.csv")
except FileNotFoundError as e:
    print(f"Error loading data: {e}. Make sure the script is run from the project root.")
    exit()

def clean_tracker_name(name):
    return name.replace('rfdetr_base__', '')

summary_df['tracker'] = summary_df['tracker'].apply(clean_tracker_name)
metrics_per_class_df['tracker'] = metrics_per_class_df['tracker'].apply(clean_tracker_name)
per_sequence_stats_df['tracker'] = per_sequence_stats_df['tracker'].apply(clean_tracker_name)

# --- 1. Tracker Performance Overview (HOTA/IDF1) ---
plt.figure(figsize=(10, 6))
plot_data = summary_df[summary_df['tracker'].isin(['botsort', 'bytetrack', 'sam2'])]
plot_data_melted = plot_data.melt(id_vars='tracker', value_vars=['weighted_HOTA', 'weighted_IDF1', 'weighted_MOTA'],
                                  var_name='Metric', value_name='Score')

ax = sns.barplot(data=plot_data_melted, x='tracker', y='Score', hue='Metric', palette=palette)
ax.set_title("Overall Tracking Performance (Weighted Metrics)", fontsize=title_fontsize)
ax.set_ylabel("Score (0-1)", fontsize=label_fontsize)
ax.set_xlabel("Tracker", fontsize=label_fontsize)
ax.tick_params(axis='x', rotation=45)
plt.legend(title='Metric')
plt.tight_layout()
OUT_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_DIR / "plot_tracker_overview.png", dpi=dpi)
plt.close()

print("Generated plot_tracker_overview.png")

# --- 2. Speed vs Accuracy Trade-off ---
plt.figure(figsize=(10, 8))
plot_data = summary_df.dropna(subset=['fps', 'weighted_HOTA'])

ax = sns.scatterplot(data=plot_data, x='fps', y='weighted_HOTA', hue='tracker', palette=palette, s=200, legend='full')

for i, row in plot_data.iterrows():
    ax.text(row['fps'] + 0.5, row['weighted_HOTA'], f"{row['tracker']} ({row['fps']:.0f} FPS)", fontsize=10)

ax.set_title("Speed vs. Accuracy Trade-off", fontsize=title_fontsize)
ax.set_xlabel("Inference Speed (FPS)", fontsize=label_fontsize)
ax.set_ylabel("HOTA Score", fontsize=label_fontsize)
plt.legend(title='Tracker')
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_speed_accuracy.png", dpi=dpi)
plt.close()

print("Generated plot_speed_accuracy.png")


# --- 3. Class-Specific Performance ---
plt.figure(figsize=(12, 7))
class_order = ['player', 'goalkeeper', 'referee', 'ball']
plot_data = metrics_per_class_df[metrics_per_class_df['tracker'].isin(['botsort', 'bytetrack'])]
plot_data = plot_data[plot_data['class'].isin(class_order)]


ax = sns.barplot(data=plot_data, x='class', y='HOTA', hue='tracker', palette=palette, order=class_order)
ax.set_title("HOTA Score by Object Class", fontsize=title_fontsize)
ax.set_ylabel("HOTA Score", fontsize=label_fontsize)
ax.set_xlabel("Object Class", fontsize=label_fontsize)
plt.legend(title='Tracker')
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_class_performance.png", dpi=dpi)
plt.close()

print("Generated plot_class_performance.png")

# --- 4. Tracking Stability (ID Switches) ---
# We need to sum ID switches per tracker from the class-specific file, as summary might be weighted.
id_switches = metrics_per_class_df.groupby('tracker')['ID-switch'].sum().reset_index()
id_switches = id_switches[id_switches['tracker'].isin(['botsort', 'bytetrack', 'sam2'])]


ax = sns.barplot(data=id_switches, x='tracker', y='ID-switch', palette="rocket_r") # Inverted rocket for worse
ax.set_title("Identity Stability: Total ID Switches", fontsize=title_fontsize)
ax.set_ylabel("Count of ID Switches (Lower is Better)", fontsize=label_fontsize)
ax.set_xlabel("Tracker", fontsize=label_fontsize)
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(OUT_DIR / "plot_id_switches.png", dpi=dpi)
plt.close()

print("Generated plot_id_switches.png")

print("\nAll plots generated successfully.")
