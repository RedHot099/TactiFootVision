"""PoC C: end-to-end image -> xG. An ImageNet-pretrained convnet fine-tuned on the
whole frame. No intermediate detection, no geometry; the network must learn pitch
layout, players and ball from pixels."""

import timm
import torch
from torch import nn


class CnnXG(nn.Module):
    def __init__(self, backbone: str = "convnext_tiny.fb_in22k_ft_in1k", drop_rate: float = 0.2, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, drop_path_rate=0.1)
        self.head = nn.Sequential(nn.Dropout(drop_rate), nn.Linear(self.backbone.num_features, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(image)).squeeze(-1)
