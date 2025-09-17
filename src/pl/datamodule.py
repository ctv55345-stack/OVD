from __future__ import annotations

from typing import Optional

import lightning as L
from torch.utils.data import DataLoader

from src.datasets.flickr30k_entities import Flickr30kEntitiesPhraseDataset, collate_fn


class Flickr30kDataModule(L.LightningDataModule):
    def __init__(self, data_root: str = "data", train_list: str = "data/train.txt", val_list: str = "data/val.txt", test_list: str = "data/test.txt", image_size: int = 384, batch_size: int = 8, num_workers: int = 2) -> None:
        super().__init__()
        self.data_root = data_root
        self.train_list = train_list
        self.val_list = val_list
        self.test_list = test_list
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        self.train_ds = Flickr30kEntitiesPhraseDataset(self.data_root, self.train_list, split="train", image_min_size=self.image_size, image_max_size=self.image_size)
        self.val_ds = Flickr30kEntitiesPhraseDataset(self.data_root, self.val_list, split="val", image_min_size=self.image_size, image_max_size=self.image_size)
        self.test_ds = Flickr30kEntitiesPhraseDataset(self.data_root, self.test_list, split="test", image_min_size=self.image_size, image_max_size=self.image_size)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=1, shuffle=False, num_workers=self.num_workers, collate_fn=collate_fn)


