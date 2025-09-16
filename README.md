## Thiết kế mô hình VOD cho dữ liệu Flickr30k Entities (thư mục `VOD/data`)

### 1) Mục tiêu
- **VOD (Vision-language Object Detection)**: Phát hiện và định vị hộp giới hạn (bounding boxes) cho các đối tượng được mô tả bởi cụm từ ngôn ngữ tự nhiên trong chú thích ảnh.
- Áp dụng trên tập Flickr30k Entities: mỗi ảnh có 5 câu mô tả, kèm chú thích thực thể (phrases) và ánh xạ tới hộp đối tượng trong `Annotations/*.xml`.

### 2) Cấu trúc dữ liệu hiện có
- Thư mục dữ liệu: `VOD/data`
  - Ảnh: `flickr30k_images/*.jpg`
  - Nhãn hộp: `Annotations/*.xml` (mỗi file chứa nhiều thực thể, có thể có nhiều box cho 1 thực thể)
  - Danh sách chia tập: `train.txt`, `val.txt`, `test.txt` (mỗi dòng là id ảnh không đuôi mở rộng)

Ví dụ một dòng `train.txt`:
```
1000092795
```
Suy ra ảnh: `flickr30k_images/1000092795.jpg`, nhãn: `Annotations/1000092795.xml`.

### 3) Định nghĩa bài toán
- Input: ảnh I và cụm từ/đoạn văn P (ví dụ: "a man", "the red shirt").
- Output: tập hộp giới hạn B = {b_i} cùng điểm tin cậy cho các đối tượng khớp P.
- Các chế độ:
  - Phrase-level detection: đầu vào là 1 cụm từ P; tìm các box khớp.
  - Caption-level grounding: đầu vào là toàn bộ câu; tìm box cho từng cụm từ đánh nhãn.

### 4) Kiến trúc đề xuất
- **Backbone ảnh**: ResNet-50/101 hoặc ViT-B (ImageNet pretrain) để trích đặc trưng không gian.
- **Detector**: DETR/Deformable-DETR (truyền thống: Faster R-CNN cũng khả thi). Đề xuất dùng Deformable-DETR để hội tụ nhanh và xử lý đa tỉ lệ.
- **Mã hóa ngôn ngữ**: BERT-base hoặc RoBERTa-base (tách từ và trích đặc trưng câu/cụm từ). Có thể thay bằng CLIP text encoder để tận dụng tiền huấn luyện thị giác-ngôn ngữ.
- **Tương tác thị giác-ngôn ngữ**:
  - Cross-attention giữa token ngôn ngữ và đặc trưng ảnh (multi-head). 
  - Hoặc dùng projection vào không gian chung (CLIP-style) + attention điều kiện.
- **Đầu ra**:
  - Nhánh phân loại: dự đoán nhãn "match" cho cụm từ đích.
  - Nhánh hồi quy box: dự đoán [cx, cy, w, h] được chuẩn hóa [0,1].
  - Tùy chọn: mask (nếu mở rộng sang segmentation).

Sơ đồ dòng dữ liệu ngắn gọn:
```
Image -> Backbone -> Feature maps ----\
                                     Cross-Attn -> Decoder -> {boxes, scores}
Text  -> Text Encoder -> Tokens  -----/
```

### 5) Tiền xử lý và loader
- Ảnh: resize ngắn nhất về 800 px (giữ tỉ lệ), chuẩn hóa theo ImageNet mean/std; data augmentation: random horizontal flip, color jitter (train).
- Văn bản: tách từ, cắt/pad về độ dài cố định (ví dụ 32 token cho cụm từ, 64–128 cho câu).
- Dataset/Loader:
  - Đọc id từ `train.txt`/`val.txt`/`test.txt`.
  - Parse `Annotations/{id}.xml` để lấy danh sách (phrase, boxes). Một phrase có thể ánh xạ nhiều box: gom thành tập box mục tiêu.
  - Chế độ phrase-level: sinh các mẫu (image_id, phrase, target_boxes).
  - Sampling cân bằng: ưu tiên phân phối độ dài phrase, tần suất đối tượng.

### 6) Hàm mất mát và gán nhãn
- Sử dụng Hungarian matching (như DETR) giữa dự đoán và box mục tiêu của phrase.
- Loss:
  - L_cls: focal loss hoặc cross-entropy cho "match" vs "no-object".
  - L_box: L1 loss cho tọa độ box.
  - L_giou: Generalized IoU loss tăng độ ổn định.
- Tổng: L = λ1·L_cls + λ2·L_box + λ3·L_giou (ví dụ: 2.0/5.0/2.0).

### 7) Chỉ số đánh giá
- mAP@IoU=0.5 cho phrase-level detection.
- Recall@K (R@1, R@5) với ngưỡng IoU ≥ 0.5 cho mỗi phrase.
- Nếu caption-level: trung bình theo số phrase/ảnh.

### 8) Tổ chức mã nguồn (đề xuất)
```
VOD/
  data/
    flickr30k_images/
    Annotations/
    train.txt
    val.txt
    test.txt
  src/
    datasets/
      flickr30k_entities.py
    models/
      backbones/{resnet.py, vit.py}
      detectors/{detr.py, deformable_detr.py}
      text/{bert_encoder.py, clip_text.py}
      fusion/{cross_attention.py}
      heads/{box_head.py, cls_head.py}
    losses/
      matching.py
      losses.py
    engine/
      train.py
      eval.py
    configs/
      vod_deformable_detr_bert.yaml
    utils/
      xml_parser.py
      box_ops.py
  README.md
```

### 9) Luồng huấn luyện
1. Khởi tạo backbone ảnh (pretrained) và encoder ngôn ngữ (BERT/CLIP).
2. Xây decoder với cross-attention điều kiện theo token ngôn ngữ.
3. Nạp dữ liệu từ `train.txt`, parse XML -> (phrase, boxes).
4. Hungarian matching giữa dự đoán và box mục tiêu của phrase.
5. Tối ưu bằng AdamW, cosine schedule, warmup 1–5 epochs.
6. Đánh giá trên `val.txt`, lưu checkpoint tốt nhất theo mAP.

Siêu tham số tham khảo:
- Batch size: 16 (tùy GPU)
- Base LR: 2e-4 (backbone LR nhỏ hơn 10x)
- Epochs: 50–100
- Max tokens (phrase): 32

### 10) Lệnh chạy mẫu (Lightning + Hydra)
```bash
# Cài đặt
pip install -r requirements.txt

# Huấn luyện/đánh giá/test với Lightning + Hydra
python -m src.engine.train_lightning

# Ghi đè nhanh qua Hydra
python -m src.engine.train_lightning model.backbone=resnet34 train.batch_size=32 train.use_gpu=false
```

### 11) Gợi ý triển khai `xml_parser.py`
```python
from pathlib import Path
import xml.etree.ElementTree as ET

def parse_flickr30k_entities(xml_path: str):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    phrase_to_boxes = {}
    for obj in root.iter('object'):
        phrase = obj.find('name').text.strip()
        bbox = obj.find('bndbox')
        box = [
            float(bbox.find('xmin').text),
            float(bbox.find('ymin').text),
            float(bbox.find('xmax').text),
            float(bbox.find('ymax').text),
        ]
        phrase_to_boxes.setdefault(phrase, []).append(box)
    return phrase_to_boxes
```

### 12) Ghi chú
- Một phrase có thể xuất hiện nhiều lần trong một ảnh; xử lý nhiều box mục tiêu là bắt buộc.
- Nếu thiếu GPU lớn, cân nhắc freeze backbone và dùng CLIP text encoder để tận dụng không gian chung.
- Có thể tăng cường bằng hard negative mining từ các phrase không khớp trong cùng ảnh.

### 13) Tài liệu tham khảo
- DETR: Carion et al., 2020.
- Deformable DETR: Zhu et al., 2021.
- Flickr30k Entities: Plummer et al., 2015.
- CLIP: Radford et al., 2021.

### 14) Cấu hình mô hình nhẹ (dễ huấn luyện, suy luận nhanh)
- **Mục tiêu**: Tối ưu tốc độ và tài nguyên, vẫn đạt chất lượng hợp lý trên Flickr30k Entities.
- **Gợi ý thành phần**:
  - Backbone ảnh: ResNet-18 (pretrained ImageNet), output dim 256.
  - Detector: Deformable-DETR "tiny" (3 encoder/3 decoder layers, 4 heads, 256 dim, 100 queries).
  - Mã hóa ngôn ngữ: DistilBERT-base-uncased (max 32 token cho phrase), hoặc MiniLM.
  - Fusion: 1–2 lớp cross-attention 256-dim, 4 heads.
  - Kích thước ảnh train/infer: cạnh ngắn 640 px, giữ tỉ lệ.
  - AMP (mixed precision): bật để tăng tốc trên GPU hỗ trợ Tensor Cores.
- **Huấn luyện đề xuất**:
  - Optim: AdamW, base LR 2e-4, backbone LR 2e-5, weight_decay 1e-4.
  - Epochs: 40 (warmup 1 epoch), cosine decay.
  - Batch: 16–32 (tùy GPU), num_workers 8.
  - Augment: horizontal flip + color jitter nhẹ; tránh augment nặng để giữ tốc độ hội tụ.
- **Suy luận nhanh**:
  - Score threshold: 0.30–0.40, top-K = 50.
  - DETR thường không cần NMS; nếu nhiều box nhiễu, bật NMS nhẹ.
  - Export ONNX (opset ≥ 17), TensorRT FP16 để đạt độ trễ thấp hơn.
- **Kỳ vọng hiệu năng** (tham khảo, tùy phần cứng):
  - GPU tầm trung (e.g., T4/3060): vài chục ms/ảnh ở 640 px, batch=1.
  - CPU: ~0.1–0.3 s/ảnh; cân nhắc giảm kích thước ảnh còn 512 px.

Ví dụ cấu hình YAML tối giản (đặt tại `src/configs/vod_deformable_detr_tiny_distilbert.yaml`):
```yaml
model:
  backbone: resnet18
  backbone_pretrained: true
  hidden_dim: 256
  detector: deformable_detr
  num_encoder_layers: 3
  num_decoder_layers: 3
  num_heads: 4
  num_queries: 100
  text_encoder: distilbert-base-uncased
  text_max_len: 32
  fusion:
    type: cross_attention
    num_layers: 1

train:
  image_min_size: 640
  image_max_size: 640
  batch_size: 16
  epochs: 40
  amp: true
  optimizer: adamw
  base_lr: 2.0e-4
  backbone_lr: 2.0e-5
  weight_decay: 1.0e-4
  lr_schedule: cosine
  warmup_epochs: 1
  augment:
    hflip: true
    color_jitter: true

loss:
  cls_weight: 2.0
  l1_weight: 5.0
  giou_weight: 2.0

data:
  root: VOD/data
  train_list: VOD/data/train.txt
  val_list: VOD/data/val.txt

infer:
  score_thresh: 0.35
  topk: 50
  apply_nms: false
```

Mẹo tối ưu thêm:
- Giảm `num_queries` xuống 50 nếu ít đối tượng/phrase, sẽ nhanh hơn.
- Dùng `resnet18_tvts` hoặc `mobilenetv3-large` nếu đã có head/neck tương thích để giảm FLOPs.
- Với CPU, đặt `image_min_size=512`, `num_heads=2`, `hidden_dim=192` để giảm độ trễ.


