from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image

from src.utils.xml_parser import parse_flickr30k_entities
import numpy as np


class Flickr30kEntitiesPhraseDataset(Dataset):
    def __init__(self, root: str | Path, id_list_file: str | Path, split: str = "train", image_min_size: int = 640, image_max_size: int = 640) -> None:
        self.root = Path(root)
        self.split = split
        self.image_min_size = image_min_size
        self.image_max_size = image_max_size

        with open(id_list_file, "r", encoding="utf-8") as f:
            self.ids = [line.strip() for line in f if line.strip()]

        self.img_dir = self.root / "flickr30k_images"
        self.ann_dir = self.root / "Annotations"

        assert self.img_dir.exists(), f"Missing images at {self.img_dir}"
        assert self.ann_dir.exists(), f"Missing annotations at {self.ann_dir}"

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        img_id = self.ids[index]
        img_path = self.img_dir / f"{img_id}.jpg"
        ann_path = self.ann_dir / f"{img_id}.xml"

        image = Image.open(img_path).convert("RGB")
        W, H = image.size

        phrase_to_boxes = parse_flickr30k_entities(ann_path)
        # Chọn ngẫu nhiên một phrase có box để tạo mẫu phrase-level
        phrases = [p for p, boxes in phrase_to_boxes.items() if len(boxes) > 0]
        if len(phrases) == 0:
            # fallback: trả về ảnh với no-object
            phrase = ""
            boxes = []
        else:
            import random
            phrase = random.choice(phrases)
            boxes = phrase_to_boxes[phrase]

        # Resize ngắn nhất về image_min_size, giữ tỉ lệ và clamp max_size
        image, scale = self._resize(image, self.image_min_size, self.image_max_size)
        # Letterbox pad to square (max_size x max_size) at top-left (pad right/bottom)
        target_size = self.image_max_size
        Wrsz, Hrsz = image.size
        if (Wrsz, Hrsz) != (target_size, target_size):
            new_im = Image.new("RGB", (target_size, target_size), (0, 0, 0))
            new_im.paste(image, (0, 0))
            image = new_im
        Wn, Hn = image.size  # should be (target_size, target_size)

        target_boxes = []
        for (xmin, ymin, xmax, ymax) in boxes:
            xmin = xmin * scale
            ymin = ymin * scale
            xmax = xmax * scale
            ymax = ymax * scale
            target_boxes.append([xmin, ymin, xmax, ymax])

        image_tensor = self._to_tensor(image)
        target = self._build_target(target_boxes, Wn, Hn)
        target["text"] = phrase
        return image_tensor, target

    @staticmethod
    def _to_tensor(image: Image.Image) -> torch.Tensor:
        img = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        # normalize using ImageNet
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (img - mean) / std
        return img

    @staticmethod
    def _resize(image: Image.Image, min_size: int, max_size: int) -> Tuple[Image.Image, float]:
        W, H = image.size
        scale = min_size / min(W, H)
        if max(W, H) * scale > max_size:
            scale = max_size / max(W, H)
        new_w, new_h = int(round(W * scale)), int(round(H * scale))
        return image.resize((new_w, new_h), Image.BILINEAR), scale

    @staticmethod
    def _build_target(boxes_xyxy: List[List[float]], W: int, H: int) -> Dict:
        import numpy as np
        import torch
        from src.utils.box_ops import box_xyxy_to_cxcywh

        if len(boxes_xyxy) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
        else:
            boxes = torch.tensor(boxes_xyxy, dtype=torch.float32)
            labels = torch.ones((boxes.size(0),), dtype=torch.long)
        # normalize cxcywh
        boxes_c = box_xyxy_to_cxcywh(boxes)
        boxes_c[:, 0] = boxes_c[:, 0] / W
        boxes_c[:, 1] = boxes_c[:, 1] / H
        boxes_c[:, 2] = boxes_c[:, 2] / W
        boxes_c[:, 3] = boxes_c[:, 3] / H
        # ensure non-degenerate boxes in normalized coords
        eps = 1e-6
        if boxes_c.numel() > 0:
            boxes_c[:, 2] = boxes_c[:, 2].clamp(min=eps, max=1.0)
            boxes_c[:, 3] = boxes_c[:, 3].clamp(min=eps, max=1.0)
            boxes_c[:, 0] = boxes_c[:, 0].clamp(min=0.0, max=1.0)
            boxes_c[:, 1] = boxes_c[:, 1].clamp(min=0.0, max=1.0)

        target = {
            "boxes": boxes_c,
            "labels": labels,
            "size": torch.tensor([W, H], dtype=torch.float32),
        }
        return target


def collate_fn(batch):
    images, targets = list(zip(*batch))
    images = torch.stack(images, dim=0)
    return images, list(targets)


