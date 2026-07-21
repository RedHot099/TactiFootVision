#!/usr/bin/env python3
"""
Wrapper script to run TrackEval on the inference outputs.
Calculates HOTA, DetA, AssA, IDF1, etc.
"""

import sys
import argparse
import json
import csv
import shutil
from pathlib import Path
import numpy as np
# Hotfix for TrackEval using deprecated np.float and np.int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int

from loguru import logger

# Add TrackEval to path
project_root = Path(__file__).resolve().parents[1]
trackeval_path = project_root / "external" / "TrackEval"
if str(trackeval_path) not in sys.path:
    sys.path.append(str(trackeval_path))

try:
    import trackeval  # type: ignore
except ImportError:
    logger.error("Could not import trackeval. Please ensure external/TrackEval exists.")
    sys.exit(1)

def run_evaluation(
    gt_dir: Path,
    tracker_dir: Path,
    output_dir: Path,
    seq_map: dict[str, str] | None = None
) -> dict:
    """
    Runs TrackEval on a specific tracker output directory against a ground truth directory.
    
    Args:
        gt_dir: Directory containing ground truth (SNMOT-XXX/gt/gt.txt)
        tracker_dir: Directory containing tracker results (SNMOT-XXX.txt or similar)
        output_dir: Directory to save results
        seq_map: Optional map of sequence names to load.
    """
    
    # We will use the generic 'MotChallenge2DBox' dataset class or similar configuration
    # However, TrackEval is a bit picky about folder structure. 
    # For flexibility, we can use the helper generic functions or construct a config.

    # Configuration for TrackEval
    eval_config = trackeval.Evaluator.get_default_eval_config()
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}

    # Adjust config to our paths
    # TrackEval expects: GT_LOC_FORMAT = '{gt_folder}/{seq}/gt/gt.txt'
    # We might need to symlink or adapt if our structure differs.
    # Our GT: data/soccernet/tracking/extracted/train/SNMOT-XXX/gt/gt.txt
    # Our Pred: results/.../inference/SNMOT-XXX/variant_name/mot/player.txt
    
    # Since we are evaluating ONE sequence or a set of sequences that might be just files,
    # and TrackEval expects a dataset structure, it's often easier to define a Custom Dataset class 
    # or trick the paths.
    
    # Let's try to set the paths explicitly.
    dataset_config['GT_FOLDER'] = str(gt_dir.parent) # Parent of SNMOT-XXX
    dataset_config['TRACKERS_FOLDER'] = str(tracker_dir) # Folder containing tracker output files
    
    # We need to ensure the tracker file naming matches the sequence name in GT.
    # If GT is SNMOT-060, tracker file should probably be SNMOT-060.txt
    
    # But wait, our output is per-class: 'player.txt', 'ball.txt'.
    # TrackEval usually evaluates per-class if classes are mixed, or single class.
    # SoccerNet tracking is multi-class.
    
    # If we want to evaluate 'player', we should pass the 'player.txt' as if it was the tracker file for that sequence.
    # This is tricky because standard MOT is single class (usually).
    
    # Let's create a temporary structure for TrackEval to consume.
    # Temp structure:
    # temp_gt/SNMOT-XXX/gt/gt.txt
    # temp_tracker/tracker_name/data/SNMOT-XXX.txt
    
    # We will run this separately for each class we want to evaluate (e.g. player, ball).
    
    return {}

def process_single_variant_sequence(
    gt_seq_dir: Path,
    pred_mot_dir: Path,
    output_dir: Path,
    classes: list[str] = ["player", "ball", "goalkeeper", "referee"]
):
    """
    Runs TrackEval for a single sequence and a single variant, per class.
    """
    seq_name = gt_seq_dir.name
    results = {}
    
    # TrackEval expects a dataset object. We will instantiate MotChallenge2DBox
    # But we override _get_sublayers to just return our specific class file.
    
    # To avoid complex monkeypatching, we'll use a simplified approach:
    # 1. Create a temporary GT file containing ONLY the class of interest.
    # 2. Use the prediction file for that class.
    # 3. Run evaluation.
    
    tmp_root = output_dir / "temp_trackeval"
    tmp_root.mkdir(parents=True, exist_ok=True)
    
    # Read GT once
    gt_file = gt_seq_dir / "gt" / "gt.txt"
    if not gt_file.exists():
        logger.warning(f"GT file not found: {gt_file}")
        return {}
        
    gt_data = np.genfromtxt(gt_file, delimiter=',')
    # GT format: frame, id, x, y, w, h, conf, class_id, vis, ... (SoccerNet specific: class is not in standard MOT columns?)
    # Wait, SoccerNet data format description in docs:
    # "The ground truth and detections are stored in comma-separate csv files with 10 columns... remaining values are set to -1"
    # Actually, SoccerNet tracking uses `gameinfo.ini` to map track_id -> label.
    # We need to read `gameinfo.ini` to filter GT by class.
    
    from tactifoot_vision.data.soccernet_tracking import read_seqinfo, SOCCERNET_CLASS_TO_ID
    # We assume we can read gameinfo.ini
    gameinfo_path = gt_seq_dir / "gameinfo.ini"
    track_to_class = {}
    if gameinfo_path.exists():
        # Parsing INI manually or using configparser
        import configparser
        config = configparser.ConfigParser()
        config.read(gameinfo_path)
        if 'traklets' in config: # Typo in some versions? usually 'tracklets' or just keys
             # SoccerNet gameinfo usually has [trackletID] -> label
             pass
        # Actually, let's check how the codebase parses this. 
        # `tactifoot_vision/data/soccernet_tracking.py` might have helpers.
        
    # Helper to parse gameinfo manually
    with open(gameinfo_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('[') or line.startswith(';'):
                continue
            if '=' in line:
                k, v = line.split('=', 1)
                k = k.strip()
                v = v.strip()
                
                if k.startswith('trackletID_'):
                    try:
                        # trackletID_1 -> 1
                        tid_str = k.split('_')[1]
                        tid = int(tid_str)
                        
                        # value format: "player team left;10" or "ball;1"
                        raw_label = v.split(';')[0].strip().lower()
                        
                        # Map to generic classes
                        if "goalkeeper" in raw_label:
                            label = "goalkeeper"
                        elif "player" in raw_label:
                            label = "player"
                        elif "referee" in raw_label:
                            label = "referee"
                        elif "ball" in raw_label:
                            label = "ball"
                        else:
                            label = raw_label # Fallback
                            
                        track_to_class[tid] = label
                    except (ValueError, IndexError):
                        continue

    # Map string class to ID for filtering
    # We can rely on string labels from gameinfo
    
    for cls in classes:
        pred_file = pred_mot_dir / f"{cls}.txt"
        if not pred_file.exists():
            continue
            
        # Filter GT for this class
        # Class filtering: track_id in GT must match a class in gameinfo
        
        # Filter GT rows
        relevant_gt_rows = []
        for row in gt_data:
            tid = int(row[1])
            if tid in track_to_class:
                label = track_to_class[tid]
                # Normalize label
                if cls == "goalkeeper":
                    if "goalkeeper" in label:
                        relevant_gt_rows.append(row)
                elif label == cls:
                    relevant_gt_rows.append(row)
        
        if not relevant_gt_rows:
            logger.warning(f"No GT found for class {cls} in {seq_name}")
            continue
            
        # Write Temp GT
        tmp_gt_seq_dir = tmp_root / "gt" / "MOT17-01" # Fake seq name
        (tmp_gt_seq_dir / "gt").mkdir(parents=True, exist_ok=True)
        
        # Need seqinfo.ini for TrackEval to know seq length
        shutil.copy(gt_seq_dir / "seqinfo.ini", tmp_gt_seq_dir / "seqinfo.ini")
        
        # Prepare data for saving with forced class_id=1 and visibility=1.0
        # MOT format: frame, id, x, y, w, h, conf, class_id, vis
        # We take original data but override class_id (col 7) to 1 and vis (col 8) to 1.0
        clean_gt_rows = []
        for row in relevant_gt_rows:
            new_row = list(row)
            # Ensure enough columns
            while len(new_row) < 9:
                new_row.append(-1)
            new_row[7] = 1 # class_id = 1 (pedestrian/target)
            new_row[8] = 1.0 # visibility = 1.0
            clean_gt_rows.append(new_row)

        np.savetxt(tmp_gt_seq_dir / "gt" / "gt.txt", np.array(clean_gt_rows), fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%.2f,%d', delimiter=',')
        
        # Prepare Tracker Input
        # TrackEval Generic expects: tracker_folder / seq_name.txt
        tmp_tracker_dir = tmp_root / "trackers" / "variant" / "data"
        tmp_tracker_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(pred_file, tmp_tracker_dir / "MOT17-01.txt")
        
        # Run TrackEval
        # We perform a localized run
        
        # Redirect stdout to suppress TrackEval spam
        # sys.stdout = open(os.devnull, 'w')
        
        eval_config = trackeval.Evaluator.get_default_eval_config()
        eval_config['DISPLAY_LESS_PROGRESS'] = True
        eval_config['PRINT_RESULTS'] = False
        eval_config['PRINT_ONLY_COMBINED'] = False
        eval_config['TIME_PROGRESS'] = False
        
        dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        dataset_config['GT_FOLDER'] = str(tmp_root / "gt")
        dataset_config['TRACKERS_FOLDER'] = str(tmp_root / "trackers")
        dataset_config['TRACKERS_TO_EVAL'] = ['variant']
        dataset_config['BENCHMARK'] = 'MOT17' # Just to satisfy config
        dataset_config['SEQ_INFO'] = {'MOT17-01': 1000} # Length doesn't matter much if seqinfo is present, but dict is needed
        dataset_config['SKIP_SPLIT_FOL'] = True
        # Overwrite SEQ_INFO with actual length from seqinfo
        
        evaluator = trackeval.Evaluator(eval_config)
        dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
        metrics_list = []
        for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity]:
            metrics_list.append(metric())
            
        raw_results, msg = evaluator.evaluate([dataset], metrics_list)
        
        # Extract results
        # Structure: raw_results['MotChallenge2DBox']['variant']['MOT17-01']['pedestrian']['HOTA']['HOTA']
        try:
            # We always check 'pedestrian' because we forced class_id=1 in GT and config
            seq_res = raw_results['MotChallenge2DBox']['variant']['MOT17-01']['pedestrian']
            results[cls] = {
                'HOTA': float(np.mean(seq_res['HOTA']['HOTA']) * 100), # Take mean over thresholds
                'DetA': float(np.mean(seq_res['HOTA']['DetA']) * 100),
                'AssA': float(np.mean(seq_res['HOTA']['AssA']) * 100),
                'IDF1': float(np.mean(seq_res['Identity']['IDF1']) * 100),
                'MOTA': float(np.mean(seq_res['CLEAR']['MOTA']) * 100),
                'IDSW': int(np.sum(seq_res['CLEAR']['IDSW'])), # Sum over thresholds? No, IDSW is usually a single count for the sequence. 
                # Wait, seq_res['CLEAR']['IDSW'] might be an array if computed per threshold?
                # For CLEAR metrics in TrackEval, they are usually scalar per sequence if threshold is fixed, or array if multiple.
                # In my config I see THRESHOLD: 0.5. But normally HOTA is over range.
                # Let's check the previous raw dump.
                # 'CLEAR': {'IDSW': np.int64(0), ...} -> It's a scalar (or 0-d array).
            }
            # For scalars, np.sum or direct access works. 
            results[cls]['IDSW'] = int(seq_res['CLEAR']['IDSW'])
            
        except KeyError as e:
            logger.error(f"Failed to parse TrackEval results for {cls}: {e}")
            logger.error(f"Raw results dump: {raw_results}")
    
    # Cleanup
    shutil.rmtree(tmp_root)
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer-root", type=Path, required=True, help="Root directory containing inference results (inference/SNMOT-XXX/...)")
    parser.add_argument("--gt-root", type=Path, required=True, help="Root directory containing GT sequences")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV file")
    args = parser.parse_args()
    
    results = []
    
    # Iterate over variants in infer-root
    # infer-root might be .../inference/SNMOT-060/
    # Variants are subdirectories
    
    seq_name = args.infer_root.name
    gt_seq_dir = args.gt_root / seq_name
    
    if not gt_seq_dir.exists():
        logger.error(f"GT directory for {seq_name} not found at {gt_seq_dir}")
        sys.exit(1)
        
    for variant_dir in args.infer_root.iterdir():
        if not variant_dir.is_dir() or variant_dir.name == "preview":
            continue
            
        mot_dir = variant_dir / "mot"
        if not mot_dir.exists():
            continue
            
        logger.info(f"Evaluating variant: {variant_dir.name}")
        
        # Calculate metrics per class
        cls_metrics = process_single_variant_sequence(
            gt_seq_dir,
            mot_dir,
            args.output.parent,
            classes=["player", "ball"]
        )
        
        for cls, metrics in cls_metrics.items():
            row = {
                "variant": variant_dir.name,
                "class": cls,
                **metrics
            }
            results.append(row)
            
    # Save to CSV
    if results:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        keys = list(results[0].keys())
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        logger.success(f"Saved evaluation results to {args.output}")
    else:
        logger.warning("No results computed.")

if __name__ == "__main__":
    main()
