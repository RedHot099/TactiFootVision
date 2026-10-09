"""Score offset frames with many fine-tuned G models and the frozen DINOv2 from one pass over the shared frozen trunk.

Every G fold model (ticket 04 G-combo, ticket 05 head-free control) is the same pretrained DINOv2-L with only its last
`n_blocks` blocks, the final norm and the head trained. The frozen blocks are identical in all of them, so a frame goes
through the trunk once and then through each model's trained tail. The split is exactly timm's `forward_features`
(patch embedding, position embedding, pre-norm, blocks, final norm) cut after block `n_blocks_total - n_blocks`;
`tests/test_probe_scoring.py` pins it to `FinetuneViTXG.forward` (equal to 1e-6). Before a model's tail is used its frozen
part is compared tensor by tensor with the trunk, so a bundle built on other frozen weights cannot slip in.

The frozen encoder (`dinov2_frozen`) is the same pretrained network with no training: its tail is the untouched last
blocks and norm, and its embedding is the concatenation of the CLS token and the mean of the patch tokens of the last
layer (after the final norm), 2 x 1024 = 2048 features for ViT-L.

Frames are read from the offset-frames version as stored (lossless 448x252 PNG, the model input before the overlay
mask), the sha256 of each file is checked against the recorded one, and the model input is built like
`xgpoc.data.datasets.load_image` + `to_tensor` (overlay mask, no augmentation). A test pins this to `ImageDataset`.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from xgpoc.data.datasets import FILL, overlay_mask, to_tensor


def trunk_features(vit: nn.Module, x: torch.Tensor, n_trained: int) -> torch.Tensor:
    """timm `VisionTransformer.forward_features` up to (not including) the first trained block."""
    x = vit.norm_pre(vit.patch_drop(vit._pos_embed(vit.patch_embed(x))))
    for block in vit.blocks[: len(vit.blocks) - n_trained]:
        x = block(x)
    return x


def trunk_state(vit: nn.Module, n_trained: int) -> dict[str, torch.Tensor]:
    """The frozen part of a ViT's state: everything except the last `n_trained` blocks and the final norm."""
    first_trained = len(vit.blocks) - n_trained
    trained = tuple(f"blocks.{i}." for i in range(first_trained, len(vit.blocks))) + ("norm.",)
    return {k: v for k, v in vit.state_dict().items() if not k.startswith(trained)}


def check_same_trunk(reference: dict[str, torch.Tensor], vit: nn.Module, n_trained: int) -> None:
    """Raise unless the frozen part of `vit` equals `reference` (from `trunk_state`) tensor by tensor."""
    state = trunk_state(vit, n_trained)
    if state.keys() != reference.keys():
        raise ValueError("frozen trunk has different tensors than the shared trunk")
    differs = [k for k, v in state.items() if not torch.equal(v.cpu(), reference[k])]
    if differs:
        raise ValueError(f"frozen trunk differs from the shared trunk in {len(differs)} tensors, e.g. {differs[0]}")


class GTail(nn.Module):
    """The trained part of one G fold model: last blocks, final norm and head. `forward` returns the two logits
    [B, 2] (xG, on-target); the pooled embedding that entered the head's MLP is kept in `pooled` (as in
    `xgpoc.models.g_bundle.g_outputs`, a hook on `head.head`)."""

    def __init__(self, model: nn.Module, n_trained: int):
        super().__init__()
        vit = model.backbone
        self.blocks = nn.Sequential(*vit.blocks[len(vit.blocks) - n_trained:])
        self.norm, self.head = vit.norm, model.head
        self.pooled: torch.Tensor | None = None
        self.head.head.register_forward_hook(lambda _m, inputs, _out: setattr(self, "pooled", inputs[0].detach()))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(self.blocks(features)))


class FrozenTail(nn.Module):
    """The pretrained last blocks and norm, untouched; the embedding is CLS token + mean patch token."""

    def __init__(self, model: nn.Module, n_trained: int):
        super().__init__()
        vit = model.backbone
        self.blocks = nn.Sequential(*vit.blocks[len(vit.blocks) - n_trained:])
        self.norm = vit.norm
        self.n_prefix = int(vit.num_prefix_tokens)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        tokens = self.norm(self.blocks(features))
        return torch.cat([tokens[:, 0], tokens[:, self.n_prefix:].mean(1)], 1).float()


@dataclass
class GMember:
    encoder: str
    fold: int
    seed: int
    temperature: float
    tail: GTail


@dataclass
class MemberScores:
    xg: np.ndarray                    # [B] raw xG logit
    on_target: np.ndarray             # [B] raw on-target logit
    embedding: np.ndarray | None      # [B, 5 x width] float32, only when asked for


class SharedScorer:
    """A shared trunk, the frozen tail and any number of G tails on one device."""

    def __init__(self, model: nn.Module, n_trained: int, device: str):
        """`model`: the pretrained (never fine-tuned) FinetuneViTXG; it provides the trunk and the frozen tail."""
        self.n_trained, self.device = n_trained, device
        self.reference = {k: v.detach().cpu().clone() for k, v in trunk_state(model.backbone, n_trained).items()}
        self.vit = model.backbone.to(device).eval()
        self.frozen = FrozenTail(model, n_trained).to(device).eval()
        self.members: dict[tuple[str, int, int], GMember] = {}

    @torch.no_grad()
    def add(self, encoder: str, fold: int, seed: int, temperature: float, model: nn.Module) -> GMember:
        """Register a fold model after checking that its frozen part is the shared trunk."""
        check_same_trunk(self.reference, model.backbone, self.n_trained)
        member = GMember(encoder, fold, seed, temperature, GTail(model, self.n_trained).to(self.device).eval())
        self.members[(encoder, fold, seed)] = member
        return member

    @torch.no_grad()
    def score(self, images: torch.Tensor, members: Sequence[GMember], embed: Sequence[GMember] = (),
              frozen: bool = True) -> tuple[list[MemberScores], np.ndarray | None]:
        """Scores of every member in `members` (embeddings only for those also in `embed`) and the frozen embedding."""
        features = trunk_features(self.vit, images.to(self.device, non_blocking=True), self.n_trained)
        out = []
        for member in members:
            logits = member.tail(features).float().cpu().numpy()
            wanted = any(member is e for e in embed)
            out.append(MemberScores(logits[:, 0], logits[:, 1], member.tail.pooled.float().cpu().numpy() if wanted else None))
        return out, (self.frozen(features).cpu().numpy() if frozen else None)


def ensemble_logits(scores: Sequence[MemberScores], members: Sequence[GMember]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mean temperature-scaled xG logit, mean raw xG logit, mean raw on-target logit) over the members, in float64.
    The first is the input of the ensemble Platt step (`g_bundle.ensemble_probability` averages `logit / T`)."""
    raw = np.stack([s.xg.astype(np.float64) for s in scores])
    temps = np.array([m.temperature for m in members])[:, None]
    on_target = np.stack([s.on_target.astype(np.float64) for s in scores])
    return (raw / temps).mean(0), raw.mean(0), on_target.mean(0)


class FrameDataset(Dataset):
    """Stored offset frames as model input tensors. Every file is checked against its recorded sha256 before use."""

    def __init__(self, paths: Sequence[Path], sha256: Sequence[str], size_wh: tuple[int, int], mask: str):
        if len(paths) != len(sha256):
            raise ValueError("one sha256 per frame is required")
        self.paths, self.sha256, self.size, self.mask = list(paths), list(sha256), tuple(size_wh), mask
        self.blank = overlay_mask(self.size, mask) if mask != "none" else None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> torch.Tensor:
        path = self.paths[i]
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != self.sha256[i]:
            raise ValueError(f"{path} does not match the recorded sha256; the offset frames changed")
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"cannot decode {path}")
        if (img.shape[1], img.shape[0]) != self.size:
            img = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
        if self.blank is not None:
            img[self.blank] = FILL
        return to_tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
