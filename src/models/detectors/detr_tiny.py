from __future__ import annotations

from typing import Dict

import torch
from torch import nn, Tensor


class TinyDETR(nn.Module):
    def __init__(self, hidden_dim: int = 256, num_queries: int = 100, num_decoder_layers: int = 3, num_heads: int = 4) -> None:
        super().__init__()
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.class_embed = nn.Linear(hidden_dim, 2)
        self.bbox_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid(),
        )

        self.pos_embed = PositionEmbeddingSine(hidden_dim // 2)

    def forward(self, feat: Tensor) -> Dict[str, Tensor]:
        # feat: [B, C, H, W]
        B, C, H, W = feat.shape
        src = feat.flatten(2).transpose(1, 2)  # [B, HW, C]
        pos = self.pos_embed(feat).flatten(2).transpose(1, 2)  # [B, HW, C]
        query = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)

        tgt = torch.zeros_like(query)
        hs = self.decoder(tgt, src + pos)  # [B, Q, C]
        logits = self.class_embed(hs)
        boxes = self.bbox_embed(hs)
        return {"pred_logits": logits, "pred_boxes": boxes}


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats: int = 128, temperature: int = 10000, normalize: bool = True, scale: float = 2 * 3.141592653589793) -> None:
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        device = x.device
        y_embed = torch.arange(H, device=device).float().unsqueeze(1).repeat(1, W)
        x_embed = torch.arange(W, device=device).float().unsqueeze(0).repeat(H, 1)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[-1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, device=device).float()
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1)  # [C, H, W]
        return pos.unsqueeze(0).repeat(B, 1, 1, 1)


