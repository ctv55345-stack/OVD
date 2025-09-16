from __future__ import annotations

import torch
from torch import nn
import timm


class TimmBackbone(nn.Module):
    def __init__(self, model_name: str = "resnet18", pretrained: bool = True, out_dim: int = 256) -> None:
        super().__init__()
        self.body = timm.create_model(model_name, pretrained=pretrained, features_only=True, out_indices=(4,))
        feat_dim = self.body.feature_info.channels()[-1]
        self.out_proj = nn.Conv2d(feat_dim, out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.body(x)[0]  # last feature map
        return self.out_proj(feats)


