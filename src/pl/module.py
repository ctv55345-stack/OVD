from __future__ import annotations

from typing import Any, Dict, List

import lightning as L
import torch
from torch import nn

from src.models.backbones.timm_backbone import TimmBackbone
from src.models.detectors.detr_tiny import TinyDETR
from src.models.text.bert_encoder import TextEncoder
from src.models.fusion.cross_attention import CrossAttentionBlock
from src.losses.matching import HungarianMatcher
from src.losses.losses import loss_labels, loss_boxes


class VODLightningModule(L.LightningModule):
    def __init__(self, backbone_name: str = "resnet18", pretrained: bool = True, hidden_dim: int = 256, num_queries: int = 100, num_heads: int = 4, num_decoder_layers: int = 3, text_model: str = "distilbert-base-uncased", text_max_len: int = 32, lr: float = 2e-4, weight_decay: float = 1e-4, cls_w: float = 2.0, l1_w: float = 5.0, giou_w: float = 2.0) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.backbone = TimmBackbone(backbone_name, pretrained, out_dim=hidden_dim)
        self.detector = TinyDETR(hidden_dim=hidden_dim, num_queries=num_queries, num_decoder_layers=num_decoder_layers, num_heads=num_heads)
        self.text = TextEncoder(text_model, text_max_len, hidden_dim)
        self.fusion = CrossAttentionBlock(hidden_dim=hidden_dim, num_heads=num_heads)

        self.matcher = HungarianMatcher(cls_cost=cls_w, l1_cost=l1_w, giou_cost=giou_w)
        self.cls_w = cls_w
        self.l1_w = l1_w
        self.giou_w = giou_w

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.backbone(images)
        out = self.detector(feats)
        return out

    def training_step(self, batch, batch_idx):
        images, targets = batch
        phrases = [t["text"] for t in targets]
        # text features computed but not deeply fused in this tiny baseline
        _ = self.text(phrases)
        out = self(images)
        indices = self.matcher(out, targets)
        l_cls = loss_labels(out, targets, indices, self.cls_w)
        l_box = loss_boxes(out, targets, indices, self.l1_w, self.giou_w)
        loss = l_cls + l_box
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        out = self(images)
        prob = out["pred_logits"].softmax(-1)[..., 1]
        pred_any = (prob.max(dim=1).values > 0.5)
        has_target = torch.tensor([1 if len(t["labels"]) > 0 else 0 for t in targets], device=images.device).bool()
        acc = (pred_any == has_target).float().mean()
        self.log("val/proxy_acc", acc, prog_bar=True)
        return acc

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)


