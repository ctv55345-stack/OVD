from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import Tensor

from src.utils.box_ops import generalized_box_iou


class HungarianMatcher:
    def __init__(self, cls_cost: float = 1.0, l1_cost: float = 5.0, giou_cost: float = 2.0) -> None:
        self.cls_cost = cls_cost
        self.l1_cost = l1_cost
        self.giou_cost = giou_cost

    @torch.no_grad()
    def __call__(self, out: Dict[str, Tensor], targets: List[Dict[str, Tensor]]) -> List[Tuple[Tensor, Tensor]]:
        """Match predictions to targets per batch item.

        out: {"pred_logits": [B, Q, 2], "pred_boxes": [B, Q, 4] (cxcywh normalized)}
        targets: list of dicts with keys: "labels" [Ni], "boxes" [Ni, 4] (cxcywh normalized)
        """
        bs, num_queries = out["pred_logits"].shape[:2]
        indices: List[Tuple[Tensor, Tensor]] = []

        for b in range(bs):
            out_prob = out["pred_logits"][b].softmax(-1)  # [Q, 2]
            out_boxes = out["pred_boxes"][b]
            tgt_ids = targets[b]["labels"]  # [N]
            tgt_boxes = targets[b]["boxes"]

            if tgt_boxes.numel() == 0:
                indices.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
                continue

            # Classification cost: negative prob of the target class = 1 (match)
            cost_cls = -out_prob[:, 1][:, None]  # [Q, 1] -> [Q, N]
            # L1 cost on boxes (cxcywh)
            cost_l1 = torch.cdist(out_boxes, tgt_boxes, p=1)
            # GIoU cost on xyxy
            out_xyxy = box_cxcywh_to_xyxy_abs(out_boxes, targets[b]["size"])  # [Q, 4]
            tgt_xyxy = box_cxcywh_to_xyxy_abs(tgt_boxes, targets[b]["size"])  # [N, 4]
            cost_giou = -generalized_box_iou(out_xyxy, tgt_xyxy)

            # Sanitize costs to avoid NaN/Inf
            cost_cls = torch.nan_to_num(cost_cls, nan=1e6, posinf=1e6, neginf=1e6)
            cost_l1 = torch.nan_to_num(cost_l1, nan=1e6, posinf=1e6, neginf=1e6)
            cost_giou = torch.nan_to_num(cost_giou, nan=1e6, posinf=1e6, neginf=1e6)

            C = self.cls_cost * cost_cls + self.l1_cost * cost_l1 + self.giou_cost * cost_giou
            C = torch.nan_to_num(C, nan=1e6, posinf=1e6, neginf=1e6).cpu()

            # Hungarian assignment via scipy (optimal matching)
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(C.numpy())
            i = torch.as_tensor(row_ind, dtype=torch.long)
            j = torch.as_tensor(col_ind, dtype=torch.long)


            indices.append((i, j))

        return indices


def box_cxcywh_to_xyxy_abs(boxes: Tensor, size: Tensor) -> Tensor:
    # boxes normalized [0,1]; size = [W, H]
    cx, cy, w, h = boxes.unbind(-1)
    W, H = size.unbind(-1)
    x1 = (cx - 0.5 * w) * W
    y1 = (cy - 0.5 * h) * H
    x2 = (cx + 0.5 * w) * W
    y2 = (cy + 0.5 * h) * H
    return torch.stack([x1, y1, x2, y2], dim=-1)


