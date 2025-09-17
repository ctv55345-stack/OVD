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
    def __init__(self, backbone_name: str = "resnet18", pretrained: bool = True, hidden_dim: int = 128, num_queries: int = 50, num_heads: int = 2, num_decoder_layers: int = 2, text_model: str = "distilbert-base-uncased", text_max_len: int = 16, lr: float = 1e-3, weight_decay: float = 1e-4, cls_w: float = 1.0, l1_w: float = 5.0, giou_w: float = 2.0) -> None:
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
        out = self(images)
        indices = self.matcher(out, targets)
        l_cls = loss_labels(out, targets, indices, self.cls_w)
        l_box = loss_boxes(out, targets, indices, self.l1_w, self.giou_w)
        loss = l_cls + l_box
        
        # Check for NaN/Inf
        # No sanitization: allow Lightning detect_anomaly to catch invalid grads
            
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/loss_cls", l_cls, prog_bar=False)
        self.log("train/loss_box", l_box, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        out = self(images)
        
        # Compute VOD metrics: mAP@0.5 and Recall@K
        prob = out["pred_logits"].softmax(-1)[..., 1]  # [B, Q]
        boxes = out["pred_boxes"]  # [B, Q, 4] normalized
        
        batch_ap = []
        batch_recall_at_1 = []
        batch_recall_at_5 = []
        
        for b in range(len(targets)):
            target = targets[b]
            if len(target["labels"]) == 0:
                # No target: all predictions should be low confidence
                max_conf = prob[b].max().item()
                batch_ap.append(1.0 if max_conf < 0.5 else 0.0)
                batch_recall_at_1.append(1.0 if max_conf < 0.5 else 0.0)
                batch_recall_at_5.append(1.0 if max_conf < 0.5 else 0.0)
                continue
                
            # Convert predictions to absolute coordinates
            W, H = target["size"]
            pred_boxes = boxes[b]  # [Q, 4] normalized
            pred_conf = prob[b]  # [Q]
            
            # Convert targets to absolute coordinates
            tgt_boxes_abs = self._normalized_to_abs(target["boxes"], W, H)
            
            # Sort predictions by confidence (descending)
            sorted_indices = pred_conf.argsort(descending=True)
            pred_boxes_sorted = pred_boxes[sorted_indices]
            pred_conf_sorted = pred_conf[sorted_indices]
            
            # Compute IoU matrix for all predictions
            pred_boxes_abs = self._normalized_to_abs(pred_boxes, W, H)
            ious = self._compute_iou_matrix(pred_boxes_abs, tgt_boxes_abs)
            
            # Compute mAP@0.5
            ap = self._compute_ap(ious, pred_conf, iou_threshold=0.5)
            batch_ap.append(ap)
            
            # Compute Recall@1 and Recall@5
            recall_at_1 = self._compute_recall_at_k(ious, k=1)
            recall_at_5 = self._compute_recall_at_k(ious, k=5)
            batch_recall_at_1.append(recall_at_1)
            batch_recall_at_5.append(recall_at_5)
        
        # Average metrics across batch
        avg_map = sum(batch_ap) / len(batch_ap)
        avg_recall_1 = sum(batch_recall_at_1) / len(batch_recall_at_1)
        avg_recall_5 = sum(batch_recall_at_5) / len(batch_recall_at_5)
        
        self.log("val/mAP@0.5", avg_map, prog_bar=True)
        self.log("val/Recall@1", avg_recall_1, prog_bar=True)
        self.log("val/Recall@5", avg_recall_5, prog_bar=True)
        
        return {"mAP@0.5": avg_map, "Recall@1": avg_recall_1, "Recall@5": avg_recall_5}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def _normalized_to_abs(self, boxes, W, H):
        """Convert normalized [cx,cy,w,h] to absolute [x1,y1,x2,y2]"""
        cx, cy, w, h = boxes.unbind(-1)
        x1 = (cx - 0.5 * w) * W
        y1 = (cy - 0.5 * h) * H
        x2 = (cx + 0.5 * w) * W
        y2 = (cy + 0.5 * h) * H
        return torch.stack([x1, y1, x2, y2], dim=-1)
    
    def _compute_iou_matrix(self, boxes1, boxes2):
        """Compute IoU matrix between two sets of boxes"""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]
        
        union = area1[:, None] + area2 - inter
        iou = inter / (union + 1e-7)
        return iou
    
    def _compute_ap(self, ious, confidences, iou_threshold=0.5):
        """Compute Average Precision (AP) for a single image"""
        if ious.numel() == 0:
            return 0.0
            
        # Sort by confidence (descending)
        sorted_indices = confidences.argsort(descending=True)
        ious_sorted = ious[sorted_indices]
        
        # For each target, find best matching prediction
        num_targets = ious_sorted.shape[1]
        if num_targets == 0:
            return 0.0
            
        # Track which targets have been matched
        matched_targets = torch.zeros(num_targets, dtype=torch.bool)
        tp = []  # true positives
        fp = []  # false positives
        
        for i in range(len(ious_sorted)):
            # Find best IoU for this prediction
            best_iou, best_target = ious_sorted[i].max(dim=0)
            
            if best_iou >= iou_threshold and not matched_targets[best_target]:
                # True positive: IoU >= threshold and target not yet matched
                tp.append(1)
                fp.append(0)
                matched_targets[best_target] = True
            else:
                # False positive: IoU < threshold or target already matched
                tp.append(0)
                fp.append(1)
        
        # Convert to cumulative sums
        tp = torch.cumsum(torch.tensor(tp, dtype=torch.float32), dim=0)
        fp = torch.cumsum(torch.tensor(fp, dtype=torch.float32), dim=0)
        
        # Compute precision and recall
        precision = tp / (tp + fp + 1e-8)
        recall = tp / num_targets
        
        # Compute AP using 11-point interpolation (simplified)
        ap = 0.0
        for t in torch.linspace(0, 1, 11):
            if len(recall) > 0 and recall.max() >= t:
                p = precision[recall >= t].max()
                ap += p / 11
                
        return ap.item()
    
    def _compute_recall_at_k(self, ious, k=1):
        """Compute Recall@K for a single image"""
        if ious.numel() == 0:
            return 0.0
            
        num_targets = ious.shape[1]
        if num_targets == 0:
            return 0.0
            
        # For each target, check if any of top-k predictions have IoU >= 0.5
        matched_targets = 0
        for t in range(num_targets):
            # Get IoU scores for this target across all predictions
            target_ious = ious[:, t]
            # Check if any of top-k predictions match this target
            top_k_ious = target_ious.topk(min(k, len(target_ious)))[0]
            if (top_k_ious >= 0.5).any():
                matched_targets += 1
                
        return matched_targets / num_targets


