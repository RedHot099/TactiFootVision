from __future__ import annotations

import argparse
import json
from pathlib import Path

from config.synloc_models import SynLocDatasetConfig
from tactifoot_vision.synloc.data import load_synloc_split
from tactifoot_vision.synloc.prediction_io import load_predictions_from_results_json
from tactifoot_vision.synloc.submission import serialize_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize official author-baseline results into local SynLoc format.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "valid", "test", "challenge"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--position-from-keypoint-index", type=int, default=1)
    args = parser.parse_args()

    split_data = load_synloc_split(SynLocDatasetConfig(root=args.dataset_root.resolve(), split=args.split))
    predictions = load_predictions_from_results_json(
        args.results,
        split_data=split_data,
        position_from_keypoint_index=args.position_from_keypoint_index,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(serialize_predictions(predictions), indent=2), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
