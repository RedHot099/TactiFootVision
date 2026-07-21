#!/usr/bin/env python3
"""
Simple offline track stitching (GSI-like) for MOT files.
Links tracklets based on spatial proximity and motion.
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cdist
from loguru import logger

def load_mot(path):
    # frame, id, x, y, w, h, conf, ...
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, header=None, names=['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'x3', 'y3', 'z3'])

def save_mot(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Sort
    df = df.sort_values(['frame', 'id'])
    df.to_csv(path, header=False, index=False, float_format='%.3f')

def predict_position(track_tail, delta_frames):
    # Simple linear motion using last 5 frames
    hist = track_tail.tail(5)
    if len(hist) < 2:
        return hist.iloc[-1][['x', 'y', 'w', 'h']].values
        
    # Calculate avg velocity
    vx = np.mean(np.diff(hist['x']))
    vy = np.mean(np.diff(hist['y']))
    vw = np.mean(np.diff(hist['w']))
    vh = np.mean(np.diff(hist['h']))
    
    last = hist.iloc[-1]
    pred = last[['x', 'y', 'w', 'h']].values + delta_frames * np.array([vx, vy, vw, vh])
    return pred

def stitch_tracks(df, max_time_gap=30, max_dist_thr=50.0):
    if df.empty: return df
    
    # Get start and end of each track
    track_ids = df['id'].unique()
    tracks = {tid: df[df['id'] == tid].sort_values('frame') for tid in track_ids}
    
    track_summaries = []
    for tid, tdf in tracks.items():
        track_summaries.append({
            'id': tid,
            'start_frame': tdf['frame'].min(),
            'end_frame': tdf['frame'].max(),
            'start_box': tdf.iloc[0][['x', 'y', 'w', 'h']].values,
            'end_box': tdf.iloc[-1][['x', 'y', 'w', 'h']].values,
            'data': tdf
        })
        
    # Sort by start time
    track_summaries.sort(key=lambda x: x['start_frame'])
    
    # Greedy matching
    # Map old_id -> new_id
    id_map = {x['id']: x['id'] for x in track_summaries}
    
    # We want to link END of Track A to START of Track B
    # Track B must start AFTER Track A ends
    
    # Iterate through tracks, looking for a predecessor
    matched_ids = set()
    
    for i, curr in enumerate(track_summaries):
        if curr['id'] in matched_ids: continue
        
        # Look for possible predecessors (tracks that ended recently)
        best_match = None
        min_dist = float('inf')
        
        # Look backwards
        for j in range(i-1, -1, -1):
            prev = track_summaries[j]
            
            # Check if prev is already merged into something else (it is not a tail anymore)
            if id_map[prev['id']] != prev['id']:
                continue
            
            # Simple check: prev must end before curr starts
            gap = curr['start_frame'] - prev['end_frame']
            if gap <= 0 or gap > max_time_gap:
                continue
                
            # If prev is already merged into something else (it's not a tail anymore), skip
            # We need to know if prev is a "sink".
            # Let's simplify: only match un-merged tracks.
            if prev['id'] in matched_ids: continue 
            
            # Predict pos of prev at curr start time
            pred_box = predict_position(prev['data'], gap)
            curr_box = curr['start_box']
            
            # Center distance
            pc = pred_box[:2] + pred_box[2:]/2
            cc = curr_box[:2] + curr_box[2:]/2
            dist = np.linalg.norm(pc - cc)
            
            if dist < max_dist_thr and dist < min_dist:
                min_dist = dist
                best_match = prev
                
        if best_match:
            # Merge curr into best_match
            # Update ID map
            id_map[curr['id']] = best_match['id']
            matched_ids.add(curr['id']) # Curr is consumed
            # Note: best_match remains available to consume MORE tracks later? 
            # No, best_match is now extended. It ends at curr's end.
            # We should update best_match's end info if we want to chain A->B->C
            best_match['end_frame'] = curr['end_frame']
            best_match['end_box'] = curr['end_box']
            best_match['data'] = pd.concat([best_match['data'], curr['data']])
            
    # Apply ID map to dataframe
    
    new_rows = []
    for chunk in track_summaries:
        # Only process roots (tracks that were not merged into another)
        # Their 'data' field has accumulated all merged segments
        if id_map[chunk['id']] == chunk['id']:
            chunk_df = chunk['data'].copy()
            new_rows.append(chunk_df)
    
    if not new_rows:
        return df
        
    final_df = pd.concat(new_rows)
    # Safety: Remove duplicates (frame, id) keeping the first one
    # This handles any edge cases where stitching might have overlapped
    final_df = final_df.drop_duplicates(subset=['frame', 'id'], keep='first')
    
    return final_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mot-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-gap", type=int, default=30)
    parser.add_argument("--dist-thr", type=float, default=100.0)
    args = parser.parse_args()
    
    df = load_mot(args.mot_file)
    logger.info(f"Loaded {len(df)} rows, {df['id'].nunique()} tracks.")
    
    stitched_df = stitch_tracks(df, args.max_gap, args.dist_thr)
    
    logger.info(f"Stitched to {stitched_df['id'].nunique()} tracks.")
    save_mot(stitched_df, args.output)

if __name__ == "__main__":
    main()
