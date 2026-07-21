from __future__ import annotations

import argparse
import json
from pathlib import Path

from tactifoot_vision.synloc.author_baseline import (
    AUTHOR_BASELINE_CONFIG,
    write_author_baseline_sources_doc,
    prepare_author_baseline_workspace,
)
from tactifoot_vision.synloc.data import download_synloc_dataset, smoke_check_synloc_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the official SynLoc author-baseline workspace.")
    parser.add_argument("--root", type=Path, default=Path("data/SoccerNet/SpiideoSynLoc"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test", "challenge"])
    parser.add_argument("--image-version", choices=["fullres", "fullhd"], default="fullres")
    parser.add_argument("--official-repo-root", type=Path, default=Path("external/mmpose-synloc"))
    parser.add_argument("--config", type=str, default=AUTHOR_BASELINE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=Path("results/synloc/author_pose_workspace"))
    parser.add_argument("--split", choices=["val", "test", "challenge"], default="val")
    parser.add_argument("--smoke-check", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    if args.download:
        root = download_synloc_dataset(root, splits=args.splits, image_version=args.image_version)

    status = smoke_check_synloc_root(root)
    prepared = prepare_author_baseline_workspace(
        dataset_root=root,
        output_dir=args.output_dir.resolve(),
        official_repo_root=args.official_repo_root.resolve(),
        split=args.split,
        official_config=args.config,
    )
    project_root = Path(__file__).resolve().parents[1]
    sources_doc = write_author_baseline_sources_doc(project_root / "docs/synloc_author_baseline_sources.md")

    payload: dict[str, object] = {
        "dataset_root": str(root),
        "sources_doc": str(sources_doc),
        **{key: str(value) for key, value in prepared.items()},
    }
    if args.smoke_check or not args.download:
        payload["smoke_check"] = status
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
