"""Multi-view G: one shared DINOv2 for the full frame and the local views, local tokens
kept until a single attention layer joins the views, then the existing pooling head.

The last `n_blocks` backbone blocks, the fusion layer and the head train together, so
the representation of the crops can adapt; the v1.3 heads on frozen compact features
could not test that. A missing view contributes one learned token and nothing else.
The fusion layer starts as the identity (zeroed output projections), so at
initialisation the model is G with extra, ignorable tokens."""

import torch
from torch import nn

from xgpoc.models.d_foundation import FinetuneViTXG
from xgpoc.v14.views import LOCAL_SIZE, VIEWS


class MultiViewViTXG(nn.Module):
    def __init__(self, backbone: str = "dinov2_l", n_blocks: int = 4, dropout: float = 0.2, n_out: int = 2,
                 n_views: int = len(VIEWS), local_size: tuple[int, int] = LOCAL_SIZE, fusion_heads: int = 8,
                 fusion_ff: int = 2048, grad_checkpointing: bool = True, pretrained: bool = True):
        super().__init__()
        self.base = FinetuneViTXG(backbone, n_blocks, dropout, grad_checkpointing=grad_checkpointing,
                                  pretrained=pretrained, n_out=n_out)
        dim = self.base.backbone.embed_dim
        patch = self.base.backbone.patch_embed.patch_size[0]
        self.n_views = n_views
        self.local_tokens = self.base.head.n_prefix + (local_size[1] // patch) * (local_size[0] // patch)
        self.view_embed = nn.Parameter(torch.zeros(1 + n_views, 1, dim))
        self.missing = nn.Parameter(torch.randn(n_views, dim) * 0.02)
        self.fusion = nn.TransformerEncoderLayer(dim, fusion_heads, fusion_ff, dropout=0.1, batch_first=True,
                                                 norm_first=True)
        for module in (self.fusion.self_attn.out_proj, self.fusion.linear2):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    def backbone_parameters(self) -> list[nn.Parameter]:
        return self.base.backbone_parameters()

    def backbone_param_groups(self) -> list[list[nn.Parameter]]:
        return self.base.backbone_param_groups()

    def encode_views(self, views: torch.Tensor, view_valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokens [B, V, T, D] and a padding mask [B, V, T]. Only present views go through
        the backbone; a missing view keeps its first token (the learned missing embedding)
        and pads the rest out."""
        b, v = views.shape[:2]
        flat = views.flatten(0, 1)
        present = view_valid.flatten() > 0
        dim = self.base.backbone.embed_dim
        tokens = flat.new_zeros(b * v, self.local_tokens, dim)
        if bool(present.any()):
            encoded = self.base.backbone.forward_features(flat[present])
            if encoded.shape[1:] != (self.local_tokens, dim):
                raise ValueError(f"local view tokens {tuple(encoded.shape[1:])} != {(self.local_tokens, dim)}")
            tokens[present] = encoded.to(tokens.dtype)
        tokens = tokens.view(b, v, self.local_tokens, dim)
        missing = view_valid <= 0
        filler = self.missing[None, :, None, :].expand(b, v, self.local_tokens, dim).to(tokens.dtype)
        tokens = torch.where(missing[..., None, None], filler, tokens)
        pad = missing[..., None].expand(b, v, self.local_tokens).clone()
        pad[..., 0] = False
        return tokens, pad

    def forward(self, image: torch.Tensor, views: torch.Tensor, view_valid: torch.Tensor,
                aux: torch.Tensor | None = None) -> torch.Tensor:
        if views.shape[1] != self.n_views or view_valid.shape != views.shape[:2]:
            raise ValueError("views must be [B, n_views, 3, H, W] with view_valid [B, n_views]")
        full = self.base.backbone.forward_features(image)
        local, local_pad = self.encode_views(views, view_valid)
        local = local.to(full.dtype)
        sequence = [full + self.view_embed[0]] + [local[:, k] + self.view_embed[k + 1] for k in range(self.n_views)]
        x = torch.cat(sequence, 1)
        pad = torch.cat([torch.zeros(full.shape[:2], dtype=torch.bool, device=x.device), local_pad.flatten(1)], 1)
        x = self.fusion(x, src_key_padding_mask=pad)
        return self.base.head(x, aux, key_padding_mask=pad)
