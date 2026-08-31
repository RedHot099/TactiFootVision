"""PoC D: frozen vision foundation model (DINOv2 ViT-S/14) -> patch tokens -> small
attention-pooling head. The backbone runs once per frame (cached), so training the
head costs seconds and the approach is data-efficient; this is the SoccerNet-challenge
recipe (frozen features + light head)."""

from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from xgpoc.data.datasets import ImageDataset

BACKBONES = {
    # name: (timm id, input (w, h) multiple of patch size)
    "dinov2_s": ("vit_small_patch14_dinov2.lvd142m", (448, 252)),
    "dinov2_b": ("vit_base_patch14_dinov2.lvd142m", (448, 252)),
    "siglip_b": ("vit_base_patch16_siglip_256.webli", (256, 256)),
    "dinov2_l": ("vit_large_patch14_dinov2.lvd142m", (448, 252)),
}


@torch.no_grad()
def extract_patch_features(df: pd.DataFrame, out: Path, backbone: str = "dinov2_s", batch_size: int = 16,
                           device: str = "cuda", flip: bool = False, path_col: str = "frame_path",
                           mask: str = "none") -> dict[str, np.ndarray]:
    """Cache [N_patches+1, C] float16 tokens (CLS first) per shot. Returns the dict.

    `flip` and `mask` change the pixels the backbone sees, so they belong to the cache
    identity - callers must pass a distinct `out` path for each combination."""
    if out.exists():
        z = np.load(out)
        return dict(zip(z["shot_id"].tolist(), z["feats"]))
    timm_id, size = BACKBONES[backbone]
    model = timm.create_model(timm_id, pretrained=True, num_classes=0, img_size=(size[1], size[0]), dynamic_img_size=True).to(device).eval()
    ds = ImageDataset(df, size_wh=size, path_col=path_col, mask=mask, flip=flip)
    loader = DataLoader(ds, batch_size, num_workers=4)
    feats, ids = [], ds.df["shot_id"].tolist()
    for batch in tqdm(loader, desc=f"features:{backbone}"):
        x = batch["image"].to(device)
        with torch.autocast("cuda", enabled=device == "cuda"):
            t = model.forward_features(x)  # [B, 1+P, C] for ViTs
        feats.append(t.half().cpu().numpy())
    feats = np.concatenate(feats)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, shot_id=np.array(ids), feats=feats)
    return dict(zip(ids, feats))


class AttentionPoolHead(nn.Module):
    """Learned queries cross-attend the frozen patch grid; concat CLS (+ optional
    engineered aux vector, layer-normed) ; MLP -> logit."""

    def __init__(self, dim: int = 384, n_queries: int = 4, n_heads: int = 4, dropout: float = 0.2, aux_dim: int = 0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.queries = nn.Parameter(torch.randn(1, n_queries, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.aux_norm = nn.LayerNorm(aux_dim) if aux_dim else None
        in_dim = dim * (n_queries + 1) + aux_dim
        self.head = nn.Sequential(nn.LayerNorm(in_dim), nn.Dropout(dropout),
                                  nn.Linear(in_dim, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 1))

    def forward(self, patches: torch.Tensor, aux: torch.Tensor | None = None) -> torch.Tensor:
        p = self.norm(patches.float())
        cls, grid = p[:, :1], p[:, 1:]
        q = self.queries.expand(p.size(0), -1, -1)
        pooled, _ = self.attn(q, grid, grid)
        z = torch.cat([cls, pooled], 1).flatten(1)
        if self.aux_norm is not None:
            z = torch.cat([z, self.aux_norm(aux.float())], 1)
        return self.head(z).squeeze(-1)


class FinetuneViTXG(nn.Module):
    """PoC D unfrozen: the same attention-pooling head, but the last `n_blocks`
    transformer blocks of DINOv2 train with it.

    Frozen-backbone D improved monotonically with backbone size (S 0.713 -> B 0.722 ->
    L 0.741 AUC) while only ~0.6M head parameters were learning, which is the reason to
    test whether the representation itself still has slack. Only the top blocks are
    released: with 6.7k training shots, unfreezing everything overfits long before it
    adapts, and the lower blocks encode generic geometry that needs no adaptation.
    Gradient checkpointing keeps ViT-L at 448x252 inside 16 GB.
    """

    def __init__(self, backbone: str = "dinov2_b", n_blocks: int = 4, dropout: float = 0.2,
                 aux_dim: int = 0, grad_checkpointing: bool = True, pretrained: bool = True):
        super().__init__()
        timm_id, size = BACKBONES[backbone]
        self.size_wh = size
        self.backbone = timm.create_model(timm_id, pretrained=pretrained, num_classes=0,
                                          img_size=(size[1], size[0]), dynamic_img_size=True)
        if grad_checkpointing:
            self.backbone.set_grad_checkpointing(True)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.trainable_blocks = self.backbone.blocks[-n_blocks:] if n_blocks else []
        for m in list(self.trainable_blocks) + ([self.backbone.norm] if n_blocks else []):
            for p in m.parameters():
                p.requires_grad = True
        self.head = AttentionPoolHead(self.backbone.embed_dim, dropout=dropout, aux_dim=aux_dim)

    def backbone_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def forward(self, image: torch.Tensor, aux: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.backbone.forward_features(image), aux)
