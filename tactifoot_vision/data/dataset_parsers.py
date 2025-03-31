# tactifoot_vision/data/dataset_parsers.py
import json
import logging
from pathlib import Path
import yaml
import shutil
import supervision as sv

logger = logging.getLogger(__name__)

CONVERTED_COCO_SUBDIR_NAME = "_coco_converted"
CONVERTED_YOLO_SUBDIR_NAME = "_yolo_pose_converted"


def convert_yolo_to_coco(yolo_yaml_path: Path, force_reconvert: bool = False) -> Path:
    if not yolo_yaml_path.is_file():
        raise FileNotFoundError(f"YOLO data.yaml file not found: {yolo_yaml_path}")

    yolo_root_dir = yolo_yaml_path.parent
    coco_root_dir = yolo_root_dir / CONVERTED_COCO_SUBDIR_NAME
    logger.info("Attempting YOLO to COCO conversion.")
    logger.info(f"Source (YOLO): {yolo_root_dir}")
    logger.info(f"Target (COCO): {coco_root_dir}")

    required_files = [
        coco_root_dir / "train" / "_annotations.coco.json",
        coco_root_dir / "valid" / "_annotations.coco.json",
    ]
    if not force_reconvert and all(f.exists() for f in required_files):
        logger.info(
            f"Found existing valid COCO conversion at {coco_root_dir}. Skipping conversion."
        )
        return coco_root_dir
    elif force_reconvert and coco_root_dir.exists():
        logger.warning(
            f"Force reconvert enabled. Removing existing directory: {coco_root_dir}"
        )
        shutil.rmtree(coco_root_dir)

    try:
        with open(yolo_yaml_path, "r") as f:
            yolo_config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to read or parse {yolo_yaml_path}: {e}", exc_info=True)
        raise RuntimeError(f"Failed to process data.yaml: {e}") from e

    train_key = "train"
    valid_key = (
        "valid" if "valid" in yolo_config else ("val" if "val" in yolo_config else None)
    )
    test_key = "test" if "test" in yolo_config else None

    if not yolo_config.get(train_key) or not valid_key:
        error_msg = (
            f"YOLO data.yaml ('{yolo_yaml_path}') must define paths for "
            f"'train' and either 'valid' or 'val' splits."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    coco_root_dir.mkdir(exist_ok=True)

    splits_to_process = {"train": train_key, "valid": valid_key}
    if test_key:
        splits_to_process["test"] = test_key

    for coco_split_name, yolo_split_key in splits_to_process.items():
        yolo_img_dir_rel = yolo_config.get(yolo_split_key)
        if not yolo_img_dir_rel:
            logger.warning(
                f"Path for split key '{yolo_split_key}' not found unexpectedly. Skipping."
            )
            continue

        yolo_img_dir = (yolo_root_dir / yolo_img_dir_rel).resolve()
        yolo_label_dir_rel = yolo_img_dir_rel.replace("images", "labels")
        yolo_label_dir = (yolo_root_dir / yolo_label_dir_rel).resolve()
        if not yolo_label_dir.exists():
            yolo_label_dir = yolo_img_dir.parent / "labels"

        if not yolo_img_dir.exists():
            msg = f"YOLO image directory not found for split '{coco_split_name}' (key: {yolo_split_key}): {yolo_img_dir}"
            logger.error(msg)
            if coco_split_name in ["train", "valid"]:
                raise FileNotFoundError(msg)
            else:
                continue
        if not yolo_label_dir.exists():
            msg = f"YOLO label directory not found for split '{coco_split_name}' (key: {yolo_split_key}): {yolo_label_dir}"
            logger.error(msg)
            if coco_split_name in ["train", "valid"]:
                raise FileNotFoundError(msg)
            else:
                continue

        coco_split_dir = coco_root_dir / coco_split_name
        coco_split_img_dir = coco_split_dir
        coco_split_dir.mkdir(exist_ok=True)

        logger.info(f"Processing split: {coco_split_name} (from key: {yolo_split_key})")
        try:
            ds = sv.DetectionDataset.from_yolo(
                images_directory_path=str(yolo_img_dir),
                annotations_directory_path=str(yolo_label_dir),
                data_yaml_path=str(yolo_yaml_path),
                force_masks=False,
            )

            original_image_paths = list(ds.image_paths)

            annotation_path = str(coco_split_dir / "_annotations.coco.json")
            ds.as_coco(annotations_path=annotation_path)
            logger.info(
                f"Saved COCO annotations for {coco_split_name} to {annotation_path}"
            )

            logger.info(f"Copying images for {coco_split_name} split...")
            copied_count = 0
            for img_path_str in original_image_paths:
                img_path = Path(img_path_str)
                target_img_path = coco_split_img_dir / img_path.name
                if not target_img_path.exists():
                    try:
                        shutil.copy2(img_path, target_img_path)
                        copied_count += 1
                    except Exception as copy_err:
                        logger.warning(
                            f"Failed to copy image {img_path} to {target_img_path}: {copy_err}"
                        )
            logger.info(
                f"Copied {copied_count} images for {coco_split_name} to {coco_split_img_dir}"
            )

        except Exception as e:
            logger.error(
                f"Failed to convert split '{coco_split_name}': {e}",
                exc_info=True,
            )
            raise RuntimeError(
                f"Failed during YOLO to COCO conversion for split '{coco_split_name}': {e}"
            ) from e

    logger.info("YOLO to COCO conversion completed.")
    return coco_root_dir


def convert_coco_to_yolo_pose(
    coco_root_path: Path, force_reconvert: bool = False
) -> Path:
    if not coco_root_path.is_dir():
        raise FileNotFoundError(f"COCO root directory not found: {coco_root_path}")

    yolo_root_dir = coco_root_path / CONVERTED_YOLO_SUBDIR_NAME
    logger.info("Attempting COCO to YOLO-Pose conversion.")
    logger.info(f"Source (COCO): {coco_root_path}")
    logger.info(f"Target (YOLO): {yolo_root_dir}")

    yolo_yaml_path = yolo_root_dir / "data.yaml"
    required_dirs = [
        yolo_root_dir / "train" / "images",
        yolo_root_dir / "train" / "labels",
        yolo_root_dir / "valid" / "images",
        yolo_root_dir / "valid" / "labels",
    ]
    if (
        not force_reconvert
        and yolo_yaml_path.exists()
        and all(d.exists() for d in required_dirs)
    ):
        logger.info(
            f"Found existing valid YOLO-Pose conversion at {yolo_root_dir}. Skipping conversion."
        )
        return yolo_yaml_path
    elif force_reconvert and yolo_root_dir.exists():
        logger.warning(
            f"Force reconvert enabled. Removing existing directory: {yolo_root_dir}"
        )
        shutil.rmtree(yolo_root_dir)

    yolo_root_dir.mkdir(exist_ok=True)
    all_classes = None
    num_keypoints = None  # Variable to store number of keypoints
    split_paths = {}

    for split in ["train", "valid", "test"]:
        coco_split_dir = coco_root_path / split
        coco_img_dir = coco_split_dir
        coco_annot_path = coco_split_dir / "_annotations.coco.json"

        if not coco_split_dir.exists() or not coco_annot_path.exists():
            if split in ["train", "valid"]:
                msg = f"Required COCO split directory or annotation file not found for '{split}' in {coco_root_path}"
                logger.error(msg)
                raise FileNotFoundError(msg)
            else:
                logger.info(f"Optional COCO split '{split}' not found. Skipping.")
                continue

        yolo_split_dir = yolo_root_dir / split
        yolo_img_dir = yolo_split_dir / "images"
        yolo_label_dir = yolo_split_dir / "labels"
        yolo_split_dir.mkdir(exist_ok=True)
        yolo_img_dir.mkdir(exist_ok=True)
        yolo_label_dir.mkdir(exist_ok=True)

        logger.info(f"Processing split: {split}")
        try:
            ds = sv.DetectionDataset.from_coco(
                images_directory_path=str(coco_img_dir),
                annotations_path=str(coco_annot_path),
                force_masks=False,
            )

            if all_classes is None:
                all_classes = ds.classes
            elif all_classes != ds.classes:
                logger.warning(f"Class names mismatch! Using classes from '{split}'.")
                all_classes = ds.classes

            # --- Determine number of keypoints from COCO categories ---
            if num_keypoints is None:
                try:
                    with open(coco_annot_path, "r") as f:
                        coco_data = json.load(f)
                    # Find the category for the 'pitch' (assuming ID 1 based on example)
                    pitch_cat = next(
                        (
                            cat
                            for cat in coco_data.get("categories", [])
                            if cat["id"] == 1
                        ),
                        None,
                    )
                    if pitch_cat and "keypoints" in pitch_cat:
                        num_keypoints = len(pitch_cat["keypoints"])
                        logger.info(
                            f"Determined number of keypoints from COCO categories: {num_keypoints}"
                        )
                    else:
                        logger.warning(
                            "Could not determine number of keypoints from COCO categories. Will attempt detection later."
                        )
                except Exception as json_e:
                    logger.warning(
                        f"Could not read COCO json to determine keypoint count: {json_e}"
                    )

            # --- Fallback keypoint shape detection (if needed) ---
            if num_keypoints is None:
                logger.warning(
                    "Attempting to detect keypoint count from first annotation..."
                )
                for _, _, detections_obj in ds:
                    if (
                        "keypoints" in detections_obj.data
                        and detections_obj.data["keypoints"] is not None
                    ):
                        keypoints_data = detections_obj.data["keypoints"]
                        if keypoints_data.size > 0:
                            if keypoints_data.ndim == 3:
                                num_keypoints = keypoints_data.shape[1]
                                logger.info(
                                    f"Detected number of keypoints from data: {num_keypoints}"
                                )
                                break
                if num_keypoints is None:
                    logger.error(
                        "Failed to determine number of keypoints for kpt_shape."
                    )
                    # Decide: raise error or proceed without kpt_shape in yaml?
                    # Let's proceed without it for now, Ultralytics might handle it.
                    # raise ValueError("Could not determine number of keypoints.")

            ds.as_yolo(
                annotations_directory_path=str(yolo_label_dir),
                # label_directory_path=str(yolo_img_dir), # Incorrect arg
            )
            logger.info(f"Saved YOLO-Pose annotations for {split} to {yolo_label_dir}")

            logger.info(f"Copying images for {split} split...")
            copied_count = 0
            for (
                img_key
            ) in ds.image_paths:  # Iterate over original paths stored by from_coco
                img_path = Path(img_key)
                target_img_path = yolo_img_dir / img_path.name
                if not target_img_path.exists():
                    try:
                        # Source path is now directly in coco_split_dir
                        source_img_path = coco_img_dir / img_path.name
                        if source_img_path.exists():
                            shutil.copy2(source_img_path, target_img_path)
                            copied_count += 1
                        else:
                            logger.warning(
                                f"Source image not found for copying: {source_img_path}"
                            )
                    except Exception as copy_err:
                        logger.warning(
                            f"Failed to copy image {source_img_path} to {target_img_path}: {copy_err}"
                        )
            logger.info(f"Copied {copied_count} images for {split} to {yolo_img_dir}")

            split_paths[split] = str(
                (yolo_split_dir / "images").relative_to(yolo_root_dir)
            )

        except Exception as e:
            logger.error(f"Failed to convert split '{split}': {e}", exc_info=True)
            raise RuntimeError(
                f"Failed during COCO to YOLO-Pose conversion for split '{split}': {e}"
            ) from e

    if all_classes is None:
        raise ValueError("Could not determine class names from COCO annotations.")

    yaml_content = {
        "path": str(yolo_root_dir.resolve()),
        "train": split_paths.get("train"),
        "val": split_paths.get("valid"),
        "test": split_paths.get("test"),
        "nc": len(all_classes),
        "names": all_classes,
    }
    # --- Explicitly add kpt_shape ---
    if num_keypoints is not None:
        # YOLO expects [NumKeypoints, 3] for (x, y, visibility)
        yaml_content["kpt_shape"] = [num_keypoints, 3]
        logger.info(f"Setting kpt_shape in data.yaml: {yaml_content['kpt_shape']}")
    else:
        logger.warning(
            "kpt_shape could not be determined and will be omitted from data.yaml. Training might fail."
        )
    # --- End kpt_shape ---

    if yaml_content.get("test") is None:
        yaml_content.pop("test", None)

    try:
        with open(yolo_yaml_path, "w") as f:
            yaml.dump(yaml_content, f, sort_keys=False, default_flow_style=None)
        logger.info(f"Generated YOLO-Pose data.yaml at {yolo_yaml_path}")
    except Exception as e:
        logger.error(f"Failed to write data.yaml: {e}", exc_info=True)
        raise RuntimeError("Failed to generate data.yaml") from e

    logger.info("COCO to YOLO-Pose conversion completed.")
    return yolo_yaml_path
