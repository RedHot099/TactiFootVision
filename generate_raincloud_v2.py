
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- Global Style Settings ---
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial']
DPI = 300
OUTPUT_DIR = "results/detection_tracking/plots/detection_tracking_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Data Loading and Preprocessing ---
try:
    per_sequence_stats_df = pd.read_csv(
        "results/detection_tracking/raw/soccernet_tracking_train2seq_100ep_infer5seq/per_sequence_stats.csv"
    )
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure the script is run from the project root and the data files exist.")
    exit()

def clean_tracker_name(name):
    return name.replace('rfdetr_base__', '')

per_sequence_stats_df['tracker'] = per_sequence_stats_df['tracker'].apply(clean_tracker_name)

# --- Improved Horizontal Raincloud Plot ---
raincloud_data = per_sequence_stats_df[per_sequence_stats_df['class'] == 'player']
raincloud_data = raincloud_data[raincloud_data['tracker'].isin(['botsort_reid', 'bytetrack', 'sam2'])]

plt.figure(figsize=(14, 8))

# Define a light palette
palette = sns.color_palette("viridis", 3)

# Step A (Cloud): Violin Plot
sns.violinplot(
    data=raincloud_data, y='tracker', x='id_ratio',
    orient='h', inner=None, palette=palette, cut=0
)

# Step C (Rain): Stripplot (drawn before boxplot to be in the background)
sns.stripplot(
    data=raincloud_data, y='tracker', x='id_ratio',
    orient='h', jitter=True, alpha=0.6, size=4, zorder=1, color=".25"
)

# Step B (Umbrella): Boxplot
sns.boxplot(
    data=raincloud_data, y='tracker', x='id_ratio',
    orient='h', width=0.15, boxprops={'zorder': 2, 'facecolor': 'white'},
    showfliers=False, whiskerprops={'linewidth': 2, 'zorder': 2},
    capprops={"zorder": 2}, medianprops={'color': 'red', 'zorder': 3}
)

# Reference Line
plt.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Ideal Ratio (1.0)')
plt.text(1.0, -0.5, 'Ideal', color='red', ha='center', va='center', fontsize=12)

# Scaling and Aesthetics
X_LIMIT_MAX = 5.0
plt.xlim(0.5, X_LIMIT_MAX)
sns.despine(left=True)

# Titles and Labels
title_text = f"Tracking Stability: Identity Ratio Distribution (Outliers > {X_LIMIT_MAX} clipped)"
plt.title(title_text, fontsize=18)
plt.xlabel("Predicted Tracks / GT Tracks (ID Ratio)", fontsize=14)
plt.ylabel("Tracker", fontsize=14)

plt.legend()
plt.tight_layout()

# Save the plot
output_path = os.path.join(OUTPUT_DIR, 'raincloud_plot_v2.png')
plt.savefig(output_path, dpi=DPI)
plt.close()

print(f"Generated improved raincloud plot: {output_path}")
