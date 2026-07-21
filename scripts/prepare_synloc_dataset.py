from __future__ import annotations

import argparse
import json
from pathlib import Path

from tactifoot_vision.synloc.data import (
    download_gamestate_dataset,
    download_synloc_dataset,
    export_synloc_detection_dataset,
    smoke_check_gamestate_root,
    smoke_check_synloc_root,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and smoke-check the SynLoc dataset.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/SoccerNet/SpiideoSynLoc"),
        help="Dataset root or the target SpiideoSynLoc directory.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test", "challenge"],
        help="Splits to download when --download is used.",
    )
    parser.add_argument(
        "--image-version",
        choices=["fullres", "fullhd"],
        default="fullres",
        help="Which image variant to request from SoccerNet.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the requested dataset splits before running smoke checks.",
    )
    parser.add_argument(
        "--auxiliary-tasks",
        nargs="*",
        choices=["gamestate-2024", "gamestate-2025"],
        default=[],
        help="Optional public SoccerNet auxiliary datasets to use alongside SynLoc.",
    )
    parser.add_argument(
        "--auxiliary-root",
        type=Path,
        default=Path("data/SoccerNetGS"),
        help="Root directory containing gamestate-2024/gamestate-2025.",
    )
    parser.add_argument(
        "--download-auxiliary",
        action="store_true",
        help="Download the auxiliary Game State datasets listed in --auxiliary-tasks.",
    )
    parser.add_argument(
        "--export-detection-dataset",
        action="store_true",
        help="Export a merged person-detection dataset for detector training.",
    )
    parser.add_argument(
        "--prepared-dataset-dir",
        type=Path,
        default=Path("data/SoccerNet/SpiideoSynLoc_detection"),
        help="Output directory for merged detection exports.",
    )
    parser.add_argument(
        "--max-aux-images-per-split",
        type=int,
        default=None,
        help="Optional cap for auxiliary Game State images per split.",
    )
    parser.add_argument(
        "--smoke-check",
        action="store_true",
        help="Print split counts and annotation paths.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if args.download:
        root = download_synloc_dataset(root, splits=args.splits, image_version=args.image_version)
    auxiliary_roots = []
    if args.auxiliary_tasks:
        auxiliary_base = args.auxiliary_root.resolve()
        for task in args.auxiliary_tasks:
            task_root = auxiliary_base / task
            if args.download_auxiliary:
                task_root = download_gamestate_dataset(auxiliary_base, task=task, splits=args.splits)
            auxiliary_roots.append(task_root.resolve())

    if args.smoke_check or not args.download:
        payload: dict[str, object] = {"synloc": smoke_check_synloc_root(root)}
        if auxiliary_roots:
            payload["auxiliary"] = {
                task: smoke_check_gamestate_root(task_root)
                for task, task_root in zip(args.auxiliary_tasks, auxiliary_roots)
            }
        print(json.dumps(payload, indent=2))

    if args.export_detection_dataset:
        prepared = export_synloc_detection_dataset(
            root,
            args.prepared_dataset_dir,
            auxiliary_roots=tuple(auxiliary_roots),
            auxiliary_tasks=tuple(args.auxiliary_tasks),
            max_aux_images_per_split=args.max_aux_images_per_split,
        )
        print(json.dumps({key: str(value) for key, value in prepared.items()}, indent=2))


if __name__ == "__main__":
    main()
