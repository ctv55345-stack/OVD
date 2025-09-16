from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor

from src.utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


def loss_labels(outputs: Dict[str, Tensor], targets: List[Dict[str, Tensor]], indices, cls_weight: float) -> Tensor:
    src_logits = outputs["pred_logits"]  # [B, Q, 2]
    B, Q, C = src_logits.shape

    target_classes = torch.zeros((B, Q), dtype=torch.long, device=src_logits.device)
    for b, (src_idx, tgt_idx) in enumerate(indices):
        target_classes[b, src_idx] = 1  # positive class index 1

    loss = F.cross_entropy(src_logits.flatten(0, 1), target_classes.flatten(0, 1))
    return cls_weight * loss


def loss_boxes(outputs: Dict[str, Tensor], targets: List[Dict[str, Tensor]], indices, l1_weight: float, giou_weight: float) -> Tensor:
    src_boxes = outputs["pred_boxes"]  # [B, Q, 4]
    B, Q, _ = src_boxes.shape

    l1_loss = torch.tensor(0.0, device=src_boxes.device)
    giou_loss = torch.tensor(0.0, device=src_boxes.device)
    num_pos = 0

    for b, (src_idx, tgt_idx) in enumerate(indices):
        if len(src_idx) == 0:
            continue
        src_b = src_boxes[b, src_idx]
        tgt_b = targets[b]["boxes"][tgt_idx]
        l1_loss = l1_loss + F.l1_loss(src_b, tgt_b, reduction="sum")

        # compute GIoU on xyxy absolute scale for stability
        size = targets[b]["size"]  # [W, H]
        src_xyxy = _cxcywh_to_xyxy_abs(src_b, size)
        tgt_xyxy = _cxcywh_to_xyxy_abs(tgt_b, size)
        giou = generalized_box_iou(src_xyxy, tgt_xyxy)
        giou_loss = giou_loss + (1.0 - giou.diag()).sum()
        num_pos += len(src_idx)

    num_pos = max(num_pos, 1)
    l1_loss = l1_loss / num_pos
    giou_loss = giou_loss / num_pos
    return l1_weight * l1_loss + giou_weight * giou_loss


def _cxcywh_to_xyxy_abs(boxes: Tensor, size: Tensor) -> Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    W, H = size.unbind(-1)
    x1 = (cx - 0.5 * w) * W
    y1 = (cy - 0.5 * h) * H
    x2 = (cx + 0.5 * w) * W
    y2 = (cy + 0.5 * h) * H
    return torch.stack([x1, y1, x2, y2], dim=-1)


