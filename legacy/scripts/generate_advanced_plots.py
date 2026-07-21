#!/usr/bin/env python3
"""
Generates advanced plots for tracking comparison:
1. HOTA Breakdown Bar Chart (HOTA, DetA, AssA)
2. Per-Class Radar Chart (Player vs Ball)
3. HOTA Heatmap (Variant vs Class)
4. ID Switches Bar Chart
5. Sanity Plots (from sanity_summary.csv): Total Rows, Total Tracks
6. Detailed Sanity Plots (from sanity.json): Track Lengths, ID Ratios, Density Heatmap
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path
from math import pi

# Set style
sns.set_theme(style="whitegrid")

def collect_sanity_details(results_dir: Path):
    """
    Walks through results_dir/inference/SEQ/VARIANT/sanity.json to collect detailed stats.
    Assumes structure: results_dir/inference/SEQ_NAME/VARIANT_NAME/sanity.json
    """
    records = []
    
    # We look for inference folder
    infer_dir = results_dir / "inference"
    if not infer_dir.exists():
        # Maybe results_dir is already the inference dir or sequence dir
        # Let's try recursive search for sanity.json
        search_dir = results_dir
    else:
        search_dir = infer_dir

    for sanity_file in search_dir.rglob("sanity.json"):
        variant_name = sanity_file.parent.name
        try:
            with open(sanity_file, 'r') as f:
                data = json.load(f)
                
            frames = data.get('frames_processed', 0)
            if frames == 0: continue
            
            per_class = data.get('per_class', {})
            for cls, stats in per_class.items():
                records.append({
                    'variant': variant_name,
                    'class': cls,
                    'frames': frames,
                    'rows': stats.get('rows', 0),
                    'tracks': stats.get('tracks', 0),
                    'mean_track_len': stats.get('mean_track_len', 0),
                    'median_track_len': stats.get('median_track_len', 0),
                    'rows_per_frame': stats.get('rows_per_frame', 0)
                })
        except Exception as e:
            print(f"Error reading {sanity_file}: {e}")
            
    return pd.DataFrame(records)

def shorten_names(df):
    """Shortens variant names for cleaner plots."""
    df = df.copy()
    replacements = {
        'rfdetr_base__': '',
        'rfdetr_seg__': '',
        'botsort_reid': 'BoT-SORT',
        'bytetrack': 'ByteTrack',
        'sam2': 'SAM2'
    }
    for k, v in replacements.items():
        df['variant'] = df['variant'].str.replace(k, v)
    return df

def plot_with_baseline(df, x_col, y_col, hue_col, output_path, title, baseline_variant_substr="ByteTrack"):
    """Generic plotter for grouped bars with baseline lines."""
    plt.figure(figsize=(12, 6))
    
    # Draw bars
    ax = sns.barplot(data=df, x=x_col, y=y_col, hue=hue_col, palette='viridis', edgecolor="black")
    
    # Identify unique metrics on X axis to draw baseline segments
    x_categories = df[x_col].unique()
    
    # Find baseline values
    baseline_rows = df[df[hue_col].str.contains(baseline_variant_substr, case=False)]
    
    if not baseline_rows.empty:
        # Create a dict mapping Metric -> Score for baseline
        baseline_map = {}
        for _, row in baseline_rows.iterrows():
            baseline_map[row[x_col]] = row[y_col]
            
        # Draw lines
        # Get x-tick locations
        locs = ax.get_xticks()
        
        # We need to map x_categories to locations based on order
        # Seaborn usually preserves order of unique found or explicit order.
        # We assume locs correspond to x_categories in order.
        
        added_legend = False
        for i, cat in enumerate(x_categories):
            if cat in baseline_map:
                val = baseline_map[cat]
                # Draw a line slightly wider than the group of bars
                # Default width is usually 0.8 total for the group
                ax.hlines(y=val, xmin=locs[i]-0.4, xmax=locs[i]+0.4, 
                          colors='red', linestyles='--', linewidth=2, 
                          label='Baseline (ByteTrack)' if not added_legend else "")
                added_legend = True

    plt.title(title)
    plt.ylim(0, 100)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_hota_breakdown(df, output_dir):
    """Grouped bar chart for HOTA, DetA, AssA."""
    df = shorten_names(df)
    metrics = ['HOTA', 'DetA', 'AssA']
    df_melt = df.melt(id_vars=['variant', 'class'], value_vars=metrics, var_name='Metric', value_name='Score')
    
    for cls in df['class'].unique():
        cls_df = df_melt[df_melt['class'] == cls]
        if cls_df.empty:
            continue
        
        plot_with_baseline(
            cls_df, 
            x_col='Metric', 
            y_col='Score', 
            hue_col='variant',
            output_path=output_dir / f'hota_breakdown_{cls}.png',
            title=f'HOTA Metrics Breakdown - {cls}'
        )

def plot_weighted_summary(df, output_dir):
    """Average HOTA across classes."""
    df = shorten_names(df)
    # Simple macro average for now
    avg_df = df.groupby('variant')[['HOTA', 'IDF1']].mean().reset_index()
    avg_df_melt = avg_df.melt(id_vars='variant', var_name='Metric', value_name='Score')
    
    plot_with_baseline(
        avg_df_melt,
        x_col='Metric',
        y_col='Score',
        hue_col='variant',
        output_path=output_dir / 'weighted_metrics.png',
        title='Macro-Average Performance (All Classes)'
    )

def plot_radar_charts(df, output_dir):
    """Radar chart comparing variants across multiple metrics."""
    df = shorten_names(df)
    metrics = ['HOTA', 'DetA', 'AssA', 'IDF1', 'MOTA']
    
    for cls in df['class'].unique():
        cls_df = df[df['class'] == cls].copy()
        if cls_df.empty:
            continue
            
        cls_df['MOTA'] = cls_df['MOTA'].clip(lower=0)
        
        N = len(metrics)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        colors = sns.color_palette("deep", len(cls_df))
        
        for i, (_, row) in enumerate(cls_df.iterrows()):
            values = row[metrics].tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=row['variant'], color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])
            
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_rlabel_position(0)
        plt.yticks([20, 40, 60, 80], ["20", "40", "60", "80"], color="grey", size=7)
        plt.ylim(0, 100)
        
        plt.title(f'Performance Radar - {cls}', y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        plt.savefig(output_dir / f'radar_chart_{cls}.png', dpi=300)
        plt.close()

def plot_hota_heatmap(df, output_dir):
    """Heatmap of HOTA scores."""
    df = shorten_names(df)
    pivot = df.pivot(index='variant', columns='class', values='HOTA')
    plt.figure(figsize=(8, len(pivot)*0.8 + 2))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", vmin=0, vmax=100)
    plt.title('HOTA Score Heatmap')
    plt.tight_layout()
    plt.savefig(output_dir / 'hota_per_class_heatmap.png', dpi=300)
    plt.close()

def plot_id_switches(df, output_dir):
    """Bar chart of ID Switches."""
    df = shorten_names(df)
    if 'IDSW' not in df.columns:
        return
        
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='variant', y='IDSW', hue='class', palette='rocket', edgecolor="black")
    plt.title('Identity Switches (Lower is Better)')
    plt.xticks(rotation=30, ha='right')
    plt.ylabel('Count')
    plt.yscale('log')
    plt.grid(axis='y', which='minor', linestyle=':', alpha=0.5)
    plt.legend(title='Class')
    plt.tight_layout()
    plt.savefig(output_dir / 'id_switches_by_tracker.png', dpi=300)
    plt.close()

def plot_track_length_stats(df, output_dir):
    """Plots track length statistics per class and variant."""
    df = shorten_names(df)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='variant', y='mean_track_len', hue='class', palette='Blues', edgecolor="black")
    plt.title('Mean Track Length (Frames)')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'track_length_players_boxplot.png', dpi=300)
    plt.close()

def plot_id_ratio(df, output_dir):
    """Plots ID Ratio."""
    df = shorten_names(df)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='variant', y='tracks', hue='class', palette='Reds', edgecolor="black")
    plt.title('Total Predicted Tracks')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'id_ratio_players_boxplot.png', dpi=300)
    plt.close()

def plot_sanity_metrics(sanity_path, output_dir):
    """Re-creates sanity plots if sanity_summary.csv is provided."""
    if not sanity_path or not sanity_path.exists():
        return
        
    df = pd.read_csv(sanity_path)
    df = shorten_names(df)
    
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x='variant', y='total_rows', color='skyblue', edgecolor="black")
    plt.title('Total Detection Rows (Sanity Check)')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'sanity_total_rows.png', dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x='variant', y='total_tracks', color='salmon', edgecolor="black")
    plt.title('Total Unique Tracks Created')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'sanity_total_tracks.png', dpi=300)
    plt.close()

def plot_sanity_heatmap(df, output_dir):
    """Heatmap of rows per frame."""
    df = shorten_names(df)
    pivot = df.pivot(index='class', columns='variant', values='rows_per_frame')
    plt.figure(figsize=(10, len(pivot)*0.8 + 2))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="magma")
    plt.title('Detections per Frame (Density)')
    plt.tight_layout()
    plt.savefig(output_dir / 'sanity_rows_per_frame_heatmap.png', dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True, help="CSV file from run_trackeval.py")
    parser.add_argument("--sanity", type=Path, help="sanity_summary.csv file")
    parser.add_argument("--results-dir", type=Path, help="Root directory to search for sanity.json files")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for plots")
    args = parser.parse_args()
    
    args.output.mkdir(parents=True, exist_ok=True)
    
    df_metrics = pd.read_csv(args.metrics)
    
    print("Generating HOTA Breakdown...")
    plot_hota_breakdown(df_metrics, args.output)
    
    print("Generating Radar Charts...")
    plot_radar_charts(df_metrics, args.output)
    
    print("Generating HOTA Heatmap...")
    plot_hota_heatmap(df_metrics, args.output)
    
    print("Generating ID Switches Plot...")
    plot_id_switches(df_metrics, args.output)
    
    print("Generating Summary Plot...")
    plot_weighted_summary(df_metrics, args.output)
    
    if args.sanity:
        print("Generating Sanity Plots (Summary)...")
        plot_sanity_metrics(args.sanity, args.output)
        
    if args.results_dir:
        print("Collecting Detailed Sanity Data...")
        df_detailed = collect_sanity_details(args.results_dir)
        if not df_detailed.empty:
            print("Generating Detailed Plots (Track Len, ID Ratio, Density)...")
            plot_track_length_stats(df_detailed, args.output)
            plot_id_ratio(df_detailed, args.output)
            plot_sanity_heatmap(df_detailed, args.output)
        else:
            print("No detailed sanity.json files found.")
    
    print(f"All plots saved to {args.output}")

if __name__ == "__main__":
    main()