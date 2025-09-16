from __future__ import annotations

import lightning as L
from lightning.pytorch.loggers import CSVLogger
import hydra
from omegaconf import DictConfig

from src.pl.datamodule import Flickr30kDataModule
from src.pl.module import VODLightningModule


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    data = Flickr30kDataModule(
        data_root=cfg.data.root,
        train_list=cfg.data.train_list,
        val_list=cfg.data.val_list,
        test_list=cfg.data.test_list,
        image_size=cfg.train.image_size,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
    )

    model = VODLightningModule(
        backbone_name=cfg.model.backbone,
        pretrained=cfg.model.backbone_pretrained,
        hidden_dim=cfg.model.hidden_dim,
        num_queries=cfg.model.num_queries,
        num_heads=cfg.model.num_heads,
        num_decoder_layers=cfg.model.num_decoder_layers,
        text_model=cfg.model.text_encoder,
        text_max_len=cfg.model.text_max_len,
        lr=cfg.train.base_lr,
        weight_decay=cfg.train.weight_decay,
        cls_w=cfg.loss.cls_weight,
        l1_w=cfg.loss.l1_weight,
        giou_w=cfg.loss.giou_weight,
    )

    logger = CSVLogger("outputs", name="pl")
    trainer = L.Trainer(
        max_epochs=cfg.train.epochs,
        precision=16 if cfg.train.amp else 32,
        logger=logger,
        gradient_clip_val=0.0,
        deterministic=False,
        devices=1,
        accelerator="gpu" if cfg.train.use_gpu else "cpu",
    )

    trainer.fit(model, datamodule=data)
    trainer.validate(model, datamodule=data)
    trainer.test(model, datamodule=data)


if __name__ == "__main__":
    main()


