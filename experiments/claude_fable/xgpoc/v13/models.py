"""Small context heads over the unchanged frozen DINOv2-L patch input."""

import torch
from torch import nn

from xgpoc.models.d_foundation import AttentionPoolHead
from xgpoc.v13.inputs import CONTROLS, InputCache, VARIANTS


def variant_kwargs(variant: str, cache: InputCache) -> dict:
    if variant not in VARIANTS:
        raise ValueError(f"unknown v1.3 variant {variant!r}")
    return {"variant": variant,
            "context_shape": None if variant == "full" else list(cache.context_shapes[variant])}


class ContextHead(nn.Module):
    """Existing attention head plus one 128-wide context encoder."""

    def __init__(self, variant: str, context_shape: list[int] | tuple[int, ...] | None = None):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown v1.3 variant {variant!r}")
        variant = CONTROLS.get(variant, variant)
        self.variant = variant
        if variant == "full":
            self.head = AttentionPoolHead(1024)
            return
        shape = tuple(context_shape or ())
        if not shape:
            raise ValueError(f"{variant} needs a context shape")
        if variant in ("time_mean", "time_gru"):
            if len(shape) != 2 or shape[-1] < 2:
                raise ValueError("temporal context must be [frames, embedding + validity]")
            embedding_dim = shape[-1] - 1
            self.project = nn.Sequential(nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, 128), nn.GELU())
            self.temporal = nn.GRU(128, 128, batch_first=True) if variant == "time_gru" else None
        else:
            if len(shape) != 1:
                raise ValueError(f"{variant} context must be a vector")
            self.project = nn.Sequential(nn.LayerNorm(shape[0]), nn.Linear(shape[0], 128), nn.GELU())
            self.temporal = None
        self.head = AttentionPoolHead(1024, aux_dim=128)

    def forward(self, patches: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        if self.variant == "full":
            return self.head(patches)
        if context is None:
            raise ValueError(f"{self.variant} requires context")
        context = context.float()
        if self.variant in ("time_mean", "time_gru"):
            valid = context[..., -1:].clamp(0, 1)
            encoded = self.project(context[..., :-1]) * valid
            if self.temporal is not None:
                encoded, _ = self.temporal(encoded)
                encoded = encoded * valid
            aux = encoded.sum(1) / valid.sum(1).clamp_min(1.0)
        else:
            aux = self.project(context)
        return self.head(patches, aux)
