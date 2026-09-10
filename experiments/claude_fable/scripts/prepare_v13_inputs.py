"""Prepare compact frozen visual contexts and primary-frame geometry for v1.3."""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import timm
import torch
from torch.utils.data import DataLoader, Dataset

from xgpoc.data.datasets import load_image, to_tensor
from xgpoc.models.d_foundation import BACKBONES
from xgpoc.paths import INDEX
from xgpoc.v13.inputs import (FULL_FEATURES, OFFSETS, PRIMARY_POSITIONAL, V13_CACHE,
                              input_files_fingerprint, metadata_fingerprint,
                              normalize_geometry, normalized_crop_pixels,
                              pool_fingerprint, roi_and_control_boxes, unique_map)
from xgpoc.v13.protocol import assign_roles, digest


def freeze_encoder(model):
    model.eval()
    model.requires_grad_(False)
    return model


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "size": path.stat().st_size, "sha256": digest(path)}


class Views(Dataset):
    def __init__(self, pool, boxes, view):
        self.pool, self.boxes, self.view = pool, boxes, view

    def __len__(self):
        return len(self.pool)

    def __getitem__(self, index):
        row = self.pool.iloc[index]
        crop = None
        flags = (1.,)
        if self.view in ("roi", "control"):
            path = str(row.frame_path)
            raw = cv2.imread(path)
            if raw is None:
                raise FileNotFoundError(path)
            boxes = self.boxes.get(str(row.shot_id), np.zeros((0, 9), np.float32))
            roi, control, flags = roi_and_control_boxes(boxes)
            box = roi if self.view == "roi" else control
            if box != (0., 0., 1., 1.):
                crop = normalized_crop_pixels(box, raw.shape[1], raw.shape[0])
        else:
            column = "frame_path" if self.view == 0 else f"frame_path_{self.view}"
            path = row.get(column, "")
            if pd.isna(path) or not str(path):
                return torch.zeros(3, 252, 448), torch.tensor([0.])
        image = load_image(str(path), (448, 252), mask="overlay", crop=crop)
        return to_tensor(image), torch.tensor(flags)


@torch.inference_mode()
def encode(model, dataset, batch_size, device):
    result = []
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    for i, (image, flags) in enumerate(loader):
        with torch.autocast(device_type=device, enabled=device == "cuda"):
            tokens = model.forward_features(image.to(device))
        compact = torch.cat((tokens[:, 0], tokens[:, 1:].mean(1)), dim=1).float().cpu()
        if flags.shape[1] == 1:
            compact *= flags
        result.append(torch.cat((compact, flags), dim=1).half().numpy())
        if i % 100 == 0:
            print(f"{dataset.view}: {i * batch_size}/{len(dataset)}", flush=True)
    return np.concatenate(result)


def save_context(path, ids, values):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, shot_id=np.asarray(ids, dtype=str), context=values)
    os.replace(temporary, path)


def geometry(pool):
    source = pd.read_parquet(PRIMARY_POSITIONAL / "tabular.parquet").drop_duplicates()
    table = pool[["shot_id"]].merge(source, on="shot_id", how="left", validate="one_to_one")
    flags = {"has_ball", "homography_ok", "gk_present", "attack_team_known"}
    valid = table.has_ball.eq(1) & table.homography_ok.eq(1)
    coordinates = [c for c in table if c not in flags | {"shot_id", "n_keypoints", "n_players"}]
    table.loc[~valid, coordinates] = np.nan
    table.loc[~table.gk_present.eq(1), ["gk_dist_to_ball", "gk_dist_to_goal_line"]] = np.nan
    team_columns = [c for c in table if "defender" in c or "attacker" in c or "opponent" in c]
    table.loc[~table.attack_team_known.eq(1), team_columns] = np.nan
    table["geometry_valid"] = valid.astype(float)
    flags.add("geometry_valid")
    normalized, stats = normalize_geometry(table, set(pool.loc[pool.role == "fit", "shot_id"]), flags)
    return normalized.drop(columns="shot_id").to_numpy(np.float32), stats, normalized.columns[1:].tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--out", type=Path, default=V13_CACHE)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, help="Smoke preparation only; use a separate output directory")
    args = parser.parse_args()
    pool = assign_roles(pd.read_parquet(args.index))
    if args.limit:
        if args.out.resolve() == V13_CACHE.resolve():
            raise ValueError("Smoke preparation requires a separate --out")
        pool = pool.groupby("role", sort=False).head(args.limit).reset_index(drop=True)
    args.out.mkdir(parents=True, exist_ok=True)
    sources = {"full_features": record(FULL_FEATURES),
               "positional_tokens": record(PRIMARY_POSITIONAL / "tokens.npz"),
               "positional_table": record(PRIMARY_POSITIONAL / "tabular.parquet")}
    identity = {"version": 1, "mask": "overlay", "offsets": list(OFFSETS),
                "pool_sha256": pool_fingerprint(pool), "input_files": input_files_fingerprint(pool),
                "sources": sources, "preparation_sha256": digest(Path(__file__)),
                "inputs_sha256": digest(Path("xgpoc/v13/inputs.py"))}
    declaration = args.out / "preparation.json"
    if declaration.exists() and json.loads(declaration.read_text()) != identity:
        raise ValueError("Preparation identity changed; use a fresh cache directory")
    declaration.write_text(json.dumps(identity, indent=2) + "\n")
    with np.load(PRIMARY_POSITIONAL / "tokens.npz") as data:
        boxes = unique_map(data["shot_id"].astype(str), data["boxes"], set(pool.shot_id))
    name, size = BACKBONES["dinov2_l"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = freeze_encoder(timm.create_model(name, pretrained=True, num_classes=0,
                            img_size=(size[1], size[0]), dynamic_img_size=True)).to(device)
    outputs = {}
    for view in ("roi", "control", *OFFSETS):
        path = args.out / f"view_{view}.npz"
        receipt = path.with_suffix(".json")
        if path.exists() and receipt.exists() and json.loads(receipt.read_text()) == record(path):
            print("verified resume", view, flush=True)
        else:
            values = encode(model, Views(pool, boxes, view), args.batch_size, device)
            save_context(path, pool.shot_id, values)
            receipt.write_text(json.dumps(record(path), indent=2) + "\n")
    for view in ("roi", "control"):
        with np.load(args.out / f"view_{view}.npz") as data:
            target = args.out / f"{view}_context.npz"
            save_context(target, pool.shot_id, data["context"])
            outputs[target.name] = record(target)
    temporal = []
    for offset in OFFSETS:
        with np.load(args.out / f"view_{offset}.npz") as data:
            temporal.append(data["context"])
    target = args.out / "temporal_context.npz"
    save_context(target, pool.shot_id, np.stack(temporal, axis=1))
    outputs[target.name] = record(target)
    values, stats, columns = geometry(pool)
    target = args.out / "geometry_context.npz"
    save_context(target, pool.shot_id, values)
    outputs[target.name] = record(target)
    metadata = {**identity, "outputs": outputs, "geometry_normalization": stats,
                "geometry_columns": columns, "geometry_source": "primary t0; detected GK only; no fallback proxy"}
    metadata["fingerprint"] = metadata_fingerprint(metadata)
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Prepared", len(pool), "shots", args.out, flush=True)


if __name__ == "__main__":
    main()
