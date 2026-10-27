"""Per-layer embeddings of frozen DINOv2-L and of the G-combo fold models (xg-dino ticket 05, council C1).

One frame goes through the pretrained DINOv2 once; after every block the CLS token and the mean of the patch tokens
are kept, before the final norm (`dino_blocks`), and the final-normed CLS / mean patch separately (`dino_final`, the
2048-d frozen embedding of `xgpoc.probe.scoring.FrozenTail`). The G-combo fold models fine-tuned only their last
`n_trained` blocks, the final norm and the head; their first blocks are the pretrained ones (checked tensor by tensor
by `SharedScorer.add`), so their per-block output there is `dino_blocks` and is stored once. Each fold model's tail
runs on the shared output of the last frozen block: per tail block CLS / mean patch before the final norm
(`g_blocks`), after the model's own final norm (`g_final`), and the 5 x 1024 attention-pool vector that enters the
head's MLP (`g_pool`, the embedding of `xgpoc.models.g_bundle.g_outputs`).

Shapes for a batch of B frames, width D, L blocks, F fold models, T trained blocks; token axis (CLS, mean patch):

    dino_blocks  [B, L, 2, D]     dino_final  [B, 2, D]
    g_blocks     [B, F, T, 2, D]  g_final     [B, F, 2, D]   g_pool  [B, F, 5 D]

Fold routing. Every shot is embedded by every fold model; out of fold for a training shot is the model of its own
held-out fold (`own_fold`). Paired shots have no fold (-1) and are read with every model.
"""

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from xgpoc.probe.scoring import SharedScorer, trunk_features

KINDS = ("dino_blocks", "dino_final", "g_blocks", "g_final", "g_pool")
NO_FOLD = -1


def cls_and_mean(tokens: torch.Tensor, n_prefix: int) -> torch.Tensor:
    """[B, 2, D]: the CLS token and the mean of the patch tokens (prefix tokens excluded), float32."""
    return torch.stack([tokens[:, 0], tokens[:, n_prefix:].mean(1)], 1).float()


class LayerEncoder:
    """The pretrained DINOv2 and the tails of the registered fold models, on one device."""

    def __init__(self, pretrained: nn.Module, n_trained: int, device: str):
        """`pretrained`: the never fine-tuned FinetuneViTXG the fold models start from."""
        self.scorer = SharedScorer(pretrained.eval(), n_trained, device)
        self.n_trained = n_trained
        self.n_prefix = int(self.scorer.vit.num_prefix_tokens)
        self.folds: list[int] = []

    def add(self, fold: int, seed: int, model: nn.Module) -> None:
        """Register a fold model (its frozen part must be the shared trunk). Fold order = the order of `add` calls."""
        if fold in self.folds:
            raise ValueError(f"fold {fold} is registered twice")
        self.scorer.add("g_combo", fold, seed, 1.0, model)
        self.folds.append(fold)

    @property
    def dims(self) -> dict[str, tuple[int, ...]]:
        """Per-frame shape of each kind."""
        d, n_blocks, f = int(self.scorer.vit.embed_dim), len(self.scorer.vit.blocks), len(self.folds)
        return {"dino_blocks": (n_blocks, 2, d), "dino_final": (2, d), "g_blocks": (f, self.n_trained, 2, d),
                "g_final": (f, 2, d), "g_pool": (f, 5 * d)}

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> dict[str, np.ndarray]:
        """Every kind for a batch of model-input images (module docstring), float32 numpy."""
        vit = self.scorer.vit
        x = trunk_features(vit, images.to(self.scorer.device, non_blocking=True), len(vit.blocks))  # embedding only
        dino, shared = [], None
        for i, block in enumerate(vit.blocks):
            x = block(x)
            dino.append(cls_and_mean(x, self.n_prefix))
            if i == len(vit.blocks) - self.n_trained - 1:
                shared = x
        out = {"dino_blocks": torch.stack(dino, 1), "dino_final": cls_and_mean(vit.norm(x), self.n_prefix)}
        g_blocks, g_final, g_pool = [], [], []
        for fold in self.folds:
            tail = next(m for k, m in self.scorer.members.items() if k[1] == fold).tail
            y, per_block = shared, []
            for block in tail.blocks:
                y = block(y)
                per_block.append(cls_and_mean(y, self.n_prefix))
            normed = tail.norm(y)
            tail.head(normed)                         # fills tail.pooled (hook on the head's MLP)
            g_blocks.append(torch.stack(per_block, 1))
            g_final.append(cls_and_mean(normed, self.n_prefix))
            g_pool.append(tail.pooled.float())
        if self.folds:
            out |= {"g_blocks": torch.stack(g_blocks, 1), "g_final": torch.stack(g_final, 1), "g_pool": torch.stack(g_pool, 1)}
        return {k: v.cpu().numpy() for k, v in out.items()}


@torch.no_grad()
def full_forward_gap(encoder: LayerEncoder, fold: int, model: nn.Module, images: torch.Tensor) -> dict[str, float]:
    """Max |Δ| between `encoder` and the fold model's own full forward (its own copy of every block, hooks on each
    block and on the head's MLP) on `images`: `shared_blocks` over the first L - T blocks against `dino_blocks`, and
    the tail kinds against the encoder's tail path. `model` must be on the encoder's device."""
    encoded = encoder.encode(images)
    k = encoder.folds.index(fold)
    vit, n_prefix = model.backbone, encoder.n_prefix
    captured: list[torch.Tensor] = []
    hooks = [b.register_forward_hook(lambda _m, _i, out: captured.append(cls_and_mean(out, n_prefix))) for b in vit.blocks]
    pooled: list[torch.Tensor] = []
    hooks.append(model.head.head.register_forward_hook(lambda _m, inputs, _o: pooled.append(inputs[0].float())))
    final: list[torch.Tensor] = []
    hooks.append(vit.norm.register_forward_hook(lambda _m, _i, out: final.append(cls_and_mean(out, n_prefix))))
    try:
        model.eval()(image=images.to(encoder.scorer.device))
    finally:
        for h in hooks:
            h.remove()
    blocks = torch.stack(captured, 1).cpu().numpy()
    n_shared = len(vit.blocks) - encoder.n_trained

    def gap(a, b) -> float:
        return float(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).max())

    return {"shared_blocks": gap(blocks[:, :n_shared], encoded["dino_blocks"][:, :n_shared]),
            "g_blocks": gap(blocks[:, n_shared:], encoded["g_blocks"][:, k]),
            "g_final": gap(final[0].cpu().numpy(), encoded["g_final"][:, k]),
            "g_pool": gap(pooled[0].cpu().numpy(), encoded["g_pool"][:, k])}


def own_fold(array: np.ndarray, rows: np.ndarray, folds: np.ndarray, model_folds: Sequence[int]) -> np.ndarray:
    """Out-of-fold selection of a G kind: for array row `rows[i]` the slice of the model whose fold is `folds[i]`
    (the shot's own held-out fold). `model_folds`: fold of each model along axis 1. Refuses shots without a fold."""
    rows, folds = np.asarray(rows), np.asarray(folds)
    if (folds == NO_FOLD).any() or (rows < 0).any():
        raise ValueError("own_fold needs training rows with a frame and a fold")
    position = {f: i for i, f in enumerate(model_folds)}
    if unknown := set(folds.tolist()) - set(position):
        raise ValueError(f"no model for folds {sorted(unknown)}")
    return array[rows, [position[f] for f in folds.tolist()]]
