
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import minmax_scale
import os

# --- Global Style Settings ---
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial']
DPI = 300
OUTPUT_DIR = "results/detection_tracking/plots/soccernet_tracking_2023_tiny_seg/plots_advanced"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Data Loading and Preprocessing ---
try:
    summary_df = pd.read_csv('results/detection_tracking/raw/soccernet_tracking_2023_tiny_seg/summary.csv')
    metrics_per_class_df = pd.read_csv('results/detection_tracking/raw/soccernet_tracking_2023_tiny_seg/metrics_per_class.csv')
    per_sequence_stats_df = pd.read_csv('results/detection_tracking/raw/soccernet_tracking_2023_tiny_seg/per_sequence_stats.csv')
except FileNotFoundError as e:
    print(f"Error: {e}. Ensure the script is run from the project root and the data files exist.")
    exit()

def clean_tracker_name(name):
    return name.replace('rfdetr_base__', '')

summary_df['tracker'] = summary_df['tracker'].apply(clean_tracker_name)
metrics_per_class_df['tracker'] = metrics_per_class_df['tracker'].apply(clean_tracker_name)
per_sequence_stats_df['tracker'] = per_sequence_stats_df['tracker'].apply(clean_tracker_name)


# --- 1. Radar Chart: Holistic Tracker Evaluation ---
def radar_chart(df, title, filename):
    labels = df.columns
    num_vars = len(labels)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    palette = sns.color_palette("viridis", len(df))
    
    for i, (index, row) in enumerate(df.iterrows()):
        values = row.values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, color=palette[i], linewidth=2, linestyle='solid', label=index)
        ax.fill(angles, values, color=palette[i], alpha=0.2)
        
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=12)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_title(title, size=18, y=1.15)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=DPI)
    plt.close()
    print(f"Generated {filename}")

radar_data = summary_df[summary_df['tracker'].isin(['botsort', 'bytetrack', 'sam2'])]
radar_data = radar_data.set_index('tracker')
metrics_to_normalize = ['weighted_HOTA', 'weighted_IDF1', 'weighted_MOTA', 'weighted_FP', 'weighted_ID-switch']
normalized_df = pd.DataFrame(index=radar_data.index)

for metric in metrics_to_normalize:
    normalized_values = minmax_scale(radar_data[metric])
    if metric in ['weighted_FP', 'weighted_ID-switch']: # Lower is better, so invert
        normalized_df[metric] = 1 - normalized_values
    else:
        normalized_df[metric] = normalized_values

radar_chart(normalized_df, "Holistic Performance Profile (Normalized Metrics)", os.path.join(OUTPUT_DIR, 'radar_chart.png'))


# --- 2. Pareto Efficiency Frontier: Speed vs. Accuracy ---
pareto_data = summary_df.dropna(subset=['fps', 'weighted_HOTA'])
pareto_data = pareto_data.sort_values(by='fps')

# Find Pareto frontier
pareto_frontier = []
max_hota = -1
for i, row in pareto_data.iterrows():
    if row['weighted_HOTA'] > max_hota:
        pareto_frontier.append(row)
        max_hota = row['weighted_HOTA']
pareto_frontier_df = pd.DataFrame(pareto_frontier)

plt.figure(figsize=(12, 8))
ax = sns.scatterplot(
    data=pareto_data, x='fps', y='weighted_HOTA', hue='tracker',
    palette='plasma', s=250, style='tracker', markers=True, legend='full'
)
ax.plot(pareto_frontier_df['fps'], pareto_frontier_df['weighted_HOTA'], 'r--', alpha=0.7, label='Pareto Frontier')

for i, row in pareto_data.iterrows():
    plt.text(row['fps'] + 1, row['weighted_HOTA'], row['tracker'], fontsize=12)

ax.axvspan(25, ax.get_xlim()[1], color='green', alpha=0.1, label='Real-Time Zone (>25 FPS)')
ax.set_title("Efficiency Frontier: Inference Speed vs. Tracking Quality", fontsize=18)
ax.set_xlabel("Inference Speed (FPS)", fontsize=14)
ax.set_ylabel("Tracking Quality (weighted HOTA)", fontsize=14)
ax.legend(loc='lower right')

# Annotation arrows
plt.annotate('', xy=(ax.get_xlim()[1] - 5, 0.45), xytext=(ax.get_xlim()[1] - 20, 0.45),
             arrowprops=dict(facecolor='black', shrink=0.05),
             )
plt.text(ax.get_xlim()[1] - 25, 0.455, 'Better Speed', fontsize=12)
plt.annotate('', xy=(10, ax.get_ylim()[1] - 0.02), xytext=(10, ax.get_ylim()[1] - 0.05),
             arrowprops=dict(facecolor='black', shrink=0.05),
             )
plt.text(12, ax.get_ylim()[1] - 0.05, 'Better Accuracy', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'pareto_plot.png'), dpi=DPI)
plt.close()
print(f"Generated {os.path.join(OUTPUT_DIR, 'pareto_plot.png')}")

# --- 3. Raincloud Plot: Stability Analysis (ID Ratio) ---
raincloud_data = per_sequence_stats_df[per_sequence_stats_df['class'] == 'player']
raincloud_data = raincloud_data[raincloud_data['tracker'].isin(['botsort', 'bytetrack', 'sam2'])]


plt.figure(figsize=(14, 8))
ax = sns.violinplot(data=raincloud_data, x='tracker', y='id_ratio', palette='viridis', inner=None, cut=0)
sns.boxplot(data=raincloud_data, x='tracker', y='id_ratio', width=0.2, boxprops={'zorder': 2}, ax=ax, color='white')
sns.stripplot(data=raincloud_data, x='tracker', y='id_ratio', jitter=True, alpha=0.5, color='grey', ax=ax)

ax.axhline(1.0, ls='--', color='red', label='Ideal Ratio (1.0)')
ax.set_title("Tracking Stability: Distribution of Identity Ratio per Sequence", fontsize=18)
ax.set_xlabel("Tracker", fontsize=14)
ax.set_ylabel("ID Ratio (Pred Tracks / GT Tracks)", fontsize=14)
if raincloud_data['id_ratio'].max() > 5: # Use log scale if there are significant outliers
    ax.set_yscale('log')
    ax.set_ylabel("ID Ratio (Log Scale)", fontsize=14)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'raincloud_plot.png'), dpi=DPI)
plt.close()
print(f"Generated {os.path.join(OUTPUT_DIR, 'raincloud_plot.png')}")


# --- 4. Diverging Lollipop Chart: Relative Class Performance ---
class_perf = metrics_per_class_df.pivot_table(index='class', columns='tracker', values='HOTA')
baseline = 'bytetrack'
competitors = ['botsort', 'sam2']

for comp in competitors:
    class_perf[f'{comp}_rel_change'] = (class_perf[comp] - class_perf[baseline]) / class_perf[baseline] * 100

plot_data = class_perf[[f'{comp}_rel_change' for comp in competitors]].reset_index()
plot_data = plot_data.melt(id_vars='class', var_name='tracker', value_name='rel_change')
plot_data['tracker'] = plot_data['tracker'].str.replace('_rel_change', '')

plot_data['color'] = ['green' if x > 0 else 'red' for x in plot_data['rel_change']]
plot_data = plot_data.sort_values('rel_change', ascending=False)
classes_order = plot_data.groupby('class')['rel_change'].mean().sort_values().index

fig, ax = plt.subplots(figsize=(12, 8))
ax.hlines(y=plot_data['class'], xmin=0, xmax=plot_data['rel_change'], color=plot_data['color'], alpha=0.6, linewidth=3)
sns.scatterplot(data=plot_data, y='class', x='rel_change', hue='tracker', style='tracker', s=200, palette='plasma')

ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax.set_title(f"Relative HOTA Performance vs. {baseline.title()} Baseline", fontsize=18)
ax.set_xlabel("Percentage Improvement / Decline (%)", fontsize=14)
ax.set_ylabel("Object Class", fontsize=14)
ax.grid(axis='y', linestyle='')
plt.legend(title='Tracker Competitor')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'lollipop_chart.png'), dpi=DPI)
plt.close()
print(f"Generated {os.path.join(OUTPUT_DIR, 'lollipop_chart.png')}")

print("\nAll advanced plots generated successfully.")
