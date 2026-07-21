
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
    summary_df = pd.read_csv(
        "results/detection_tracking/raw/soccernet_tracking_train2seq_100ep_infer5seq/summary.csv"
    )
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure the script is run from the project root and the data files exist.")
    exit()

def clean_tracker_name(name):
    return name.replace('rfdetr_base__', '')

summary_df['tracker'] = summary_df['tracker'].apply(clean_tracker_name)
plot_data = summary_df[summary_df['tracker'].isin(['botsort_reid', 'bytetrack', 'sam2'])].sort_values('weighted_HOTA', ascending=False)

# --- Dual-Axis Combo Chart ---
fig, ax1 = plt.subplots(figsize=(12, 8))

# Style & Colors
bar_color = 'teal'
line_color = 'darkorange'

# Primary Y-axis (Left): HOTA Bar Chart
sns.barplot(data=plot_data, x='tracker', y='weighted_HOTA', color=bar_color, alpha=0.7, ax=ax1, label='HOTA')
ax1.set_ylabel('Weighted HOTA Score (Higher is Better)', fontsize=14)
ax1.tick_params(axis='y')
ax1.set_xlabel('Tracker', fontsize=14)

# Secondary Y-axis (Right): FPS Line Plot
ax2 = ax1.twinx()
sns.lineplot(data=plot_data, x='tracker', y='fps', color=line_color, marker='o', 
             linewidth=3, markersize=10, ax=ax2, label='FPS', sort=False)
ax2.set_ylabel('Inference Speed (FPS)', fontsize=14)
ax2.tick_params(axis='y')

# Annotations
for index, row in plot_data.iterrows():
    # HOTA on bars
    ax1.text(row['tracker'], row['weighted_HOTA'] / 2, f"{row['weighted_HOTA']:.2f}", 
             color='white', ha="center", va="center", fontsize=14, weight='bold')
    # FPS on line markers
    ax2.text(row['tracker'], row['fps'] + 2, f"{row['fps']:.1f}", 
             color=line_color, ha="center", va="bottom", fontsize=12)

# Layout
ax1.set_title("The Cost of Quality: Tracking Accuracy (HOTA) vs. Inference Speed (FPS)", fontsize=18, pad=20)
ax1.grid(axis='x') # Hide vertical grid lines for a cleaner look
ax2.grid(False) # Turn off the grid for the secondary axis

# Combined Legend
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right')
ax1.get_legend().remove() # Remove the default legend from ax1

plt.tight_layout()

# Save the plot
output_path = os.path.join(OUTPUT_DIR, 'combo_hota_fps.png')
plt.savefig(output_path, dpi=DPI)
plt.close()

print(f"Generated combo chart: {output_path}")
