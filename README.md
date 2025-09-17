# VOD: Vision-language Object Detection trên Flickr30k Entities

## Mô tả
VOD (Vision-language Object Detection) - phát hiện và định vị hộp giới hạn cho các đối tượng được mô tả bằng cụm từ ngôn ngữ tự nhiên trong ảnh.

## Kiến trúc
- **Backbone**: ResNet-18 (timm) - pretrained ImageNet
- **Detector**: Tiny DETR (3 decoder layers, 4 heads, 100 queries)
- **Text Encoder**: DistilBERT-base-uncased
- **Fusion**: Cross-attention giữa visual features và text tokens
- **Loss**: Hungarian matching + CE + L1 + GIoU

## Cài đặt
```bash
conda activate aka_env
pip install -r requirements.txt
```

## Sử dụng

### Huấn luyện
```bash
# Config chuẩn (640px, AMP, workers)
python -m src.engine.train_lightning

# Config nhanh (512px, tối ưu tốc độ)
python -m src.engine.train_lightning --config-name=config_fast

# Tùy chỉnh nhanh
python -m src.engine.train_lightning model.backbone=resnet34 train.batch_size=32
```

### Cấu hình
- **conf/config.yaml**: Config chuẩn
- **conf/config_fast.yaml**: Config tối ưu tốc độ
- **Hydra**: Override bất kỳ tham số nào qua command line

## Metrics
- **mAP@0.5**: Average Precision với IoU ≥ 0.5
- **Recall@1**: Recall với top-1 prediction
- **Recall@5**: Recall với top-5 predictions

## Cấu trúc dữ liệu
```
data/
├── flickr30k_images/     # Ảnh JPG
├── Annotations/          # XML labels
├── train.txt            # Danh sách train
├── val.txt              # Danh sách validation
└── test.txt             # Danh sách test
```

## Cấu trúc code
```
src/
├── pl/                  # PyTorch Lightning
│   ├── module.py        # LightningModule
│   └── datamodule.py    # LightningDataModule
├── models/              # Model components
│   ├── backbones/       # timm backbones
│   ├── detectors/       # DETR variants
│   ├── text/           # Text encoders
│   └── fusion/         # Vision-text fusion
├── datasets/           # Data loading
├── losses/             # Loss functions
├── utils/              # Utilities
└── engine/             # Training scripts
```

## Tối ưu hóa
- **AMP**: Mixed precision training (FP16)
- **Workers**: Parallel data loading
- **Hungarian matching**: Optimal assignment với scipy
- **GIoU**: Stable computation cho FP16

## Kết quả mong đợi
- **Tốc độ**: ~1.3 it/s trên GPU tầm trung
- **mAP@0.5**: 0.1-0.3 (model tốt)
- **Recall@1**: 0.2-0.4
- **Recall@5**: 0.4-0.6