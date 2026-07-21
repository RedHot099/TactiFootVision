from __future__ import annotations

import json
from pathlib import Path

from tactifoot_vision.synloc.data import smoke_check_synloc_root


AUTHOR_BASELINE_REPO_URL = "https://github.com/Spiideo/mmpose.git"
AUTHOR_BASELINE_BRANCH = "spiideo_scenes"
AUTHOR_BASELINE_COMMIT = "bf1b4401a0f12b1f7d2a2e007d287e0f26ca789e"
AUTHOR_BASELINE_SSKIT_REPO_URL = "https://github.com/Spiideo/sskit.git"
AUTHOR_BASELINE_SSKIT_COMMIT = "9e28ad1bdc9b5a79deb82c337eeaa19f481b415e"
AUTHOR_BASELINE_CONFIG = "configs/body_bev_position/spiideo_soccernet/yoloxpose_m_4xb64-300e_960.py"
AUTHOR_BASELINE_INIT_CHECKPOINT = (
    "https://download.openmmlab.com/mmpose/v1/pretrained_models/"
    "yolox_m_8x8_300e_coco_20230829.pth"
)
AUTHOR_BASELINE_EXPECTED_MAP = 79.3
AUTHOR_BASELINE_EXPECTED_LEADERBOARD_MAP = 76.17


def build_author_baseline_sources_markdown() -> str:
    return f"""# SynLoc Author Baseline Sources

- Official baseline repo: `{AUTHOR_BASELINE_REPO_URL}`
- Official baseline branch: `{AUTHOR_BASELINE_BRANCH}`
- Pinned baseline commit: `{AUTHOR_BASELINE_COMMIT}`
- Official devkit repo: `{AUTHOR_BASELINE_SSKIT_REPO_URL}`
- Pinned devkit commit: `{AUTHOR_BASELINE_SSKIT_COMMIT}`
- Target baseline config: `{AUTHOR_BASELINE_CONFIG}`
- Public init checkpoint: `{AUTHOR_BASELINE_INIT_CHECKPOINT}`
- Final SynLoc checkpoint: available behind `research.spiideo.com` login, not as a public direct URL
- Training fallback: train from `{AUTHOR_BASELINE_CONFIG}` and use `work_dirs/yoloxpose_m_4xb64-300e_960/epoch_300.pth`
- Paper reference: `mAP-LocSim {AUTHOR_BASELINE_EXPECTED_MAP}`
- Leaderboard-adjusted reference after the sskit bug fix: `mAP-LocSim {AUTHOR_BASELINE_EXPECTED_LEADERBOARD_MAP}`
"""


def write_author_baseline_sources_doc(output_path: Path) -> Path:
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_author_baseline_sources_markdown(), encoding="utf-8")
    return output_path


def prepare_author_baseline_workspace(
    *,
    dataset_root: Path,
    output_dir: Path,
    official_repo_root: Path,
    split: str,
    official_config: str = AUTHOR_BASELINE_CONFIG,
) -> dict[str, Path]:
    dataset_root = Path(dataset_root).resolve()
    output_dir = Path(output_dir).resolve()
    official_repo_root = Path(official_repo_root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    smoke_status = smoke_check_synloc_root(dataset_root)
    manifest = {
        "dataset_root": str(dataset_root),
        "split": split,
        "official_repo_root": str(official_repo_root),
        "official_config": official_config,
        "position_from_keypoint_index": 1,
        "smoke_status": smoke_status,
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    override_config_path = output_dir / f"{split}_override.py"
    override_config_path.write_text(
        _build_override_config_text(
            dataset_root=dataset_root,
            output_dir=output_dir,
            official_repo_root=official_repo_root,
            official_config=official_config,
            split=split,
        ),
        encoding="utf-8",
    )

    return {
        "manifest_path": manifest_path,
        "override_config_path": override_config_path,
    }


def _build_override_config_text(
    *,
    dataset_root: Path,
    output_dir: Path,
    official_repo_root: Path,
    official_config: str,
    split: str,
) -> str:
    config_path = (official_repo_root / official_config).resolve()
    output_prefix = (output_dir / "official").resolve()
    return f"""from copy import deepcopy

_base_ = {str(config_path)!r}

dataset_root = {str(dataset_root)!r}
outfile_prefix = {str(output_prefix)!r}

def _patch_dataset(dataset, ann_file, image_dir):
    dataset = deepcopy(dataset)
    dataset['data_root'] = dataset_root
    dataset['ann_file'] = ann_file
    dataset['data_prefix'] = dict(img=image_dir)
    return dataset

def _patch_evaluator(evaluators, ann_file):
    evaluators = deepcopy(evaluators)
    for evaluator in evaluators:
        evaluator['ann_file'] = ann_file
        if evaluator.get('iou_type') == 'locsim_bbox':
            evaluator['outfile_prefix'] = outfile_prefix + '_locsim_bbox'
        elif evaluator.get('iou_type') == 'locsim':
            evaluator['outfile_prefix'] = outfile_prefix + '_locsim'
    return evaluators

train_dataset = _patch_dataset(_base_.train_dataset, 'annotations/train.json', 'train')
val_dataset = _patch_dataset(_base_.val_dataset, 'annotations/val.json', 'val')
test_dataset = _patch_dataset(_base_.test_dataset, 'annotations/test.json', 'test')
challenge_dataset = _patch_dataset(_base_.challenge_dataset, 'annotations/challenge_public.json', 'challenge')

val_evaluator = _patch_evaluator(_base_.bev_val_evaluator, dataset_root + '/annotations/val.json')

if {split!r} == 'val':
    test_dataloader = dict(dataset=_patch_dataset(_base_.val_dataset, 'annotations/val.json', 'val'))
    test_evaluator = _patch_evaluator(_base_.bev_val_evaluator, dataset_root + '/annotations/val.json')
elif {split!r} == 'test':
    test_dataloader = dict(dataset=test_dataset)
    test_evaluator = _patch_evaluator(_base_.bev_test_evaluator, dataset_root + '/annotations/test.json')
else:
    test_dataloader = dict(dataset=challenge_dataset)
    test_evaluator = _patch_evaluator(_base_.bev_challenge_evaluator, dataset_root + '/annotations/challenge_public.json')

train_dataloader = dict(dataset=train_dataset)
val_dataloader = dict(dataset=val_dataset)
"""
