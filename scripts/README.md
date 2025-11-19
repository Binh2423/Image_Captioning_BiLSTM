# Hướng dẫn sử dụng MSG_C Caption Generator

## Tổng quan

Script `caption_ar.py` triển khai mô hình sinh caption với kiến trúc:
- **DeiT** (Data-efficient Image Transformer) làm feature extractor
- **MSG_C** (Multi-Scale Semantic Guidance Component) để trích xuất đặc trưng đa tỷ lệ
- **Encoder** (BiLSTM hoặc Transformer) để xử lý visual features
- **Decoder** (Autoregressive LSTM với attention) với gated fusion của MSG_C vector

## Cài đặt

Cài đặt các dependencies cần thiết:

```bash
pip install torch torchvision timm nltk joblib tqdm numpy
```

## Chuẩn bị dữ liệu

### 1. Load vocabulary từ tokenizer.pkl

Nếu bạn có file `vocab.pkl` từ tokenizer, có thể convert sang JSON:

```bash
python scripts/load_vocab_from_tokenizer.py \
    --vocab-pkl vocab.pkl \
    --output data/vocab.json \
    --verbose
```

Script này sẽ:
- Load vocab.pkl bằng joblib hoặc pickle (với safe unpickler)
- Export ra file vocab.json dễ sử dụng hơn
- In ra thống kê về vocabulary

### 2. Sử dụng vocab.pkl trực tiếp

Script `caption_ar.py` có thể load trực tiếp file vocab.pkl:

```bash
python scripts/caption_ar.py --vocab-path data/vocab.pkl ...
```

## Các chế độ sử dụng

### 1. Smoke Test - Kiểm tra nhanh

Chạy training với dữ liệu ngẫu nhiên để kiểm tra:

```bash
python scripts/caption_ar.py \
    --pretrained \
    --vocab-path data/vocab.pkl \
    --batch-size 4 \
    --tgt-seq-len 12 \
    --epochs 1 \
    --compute-metrics
```

Lệnh này sẽ:
- Load DeiT model với pretrained weights
- Tạo dữ liệu ngẫu nhiên để test
- Train 1 epoch
- In ra shapes của tất cả tensors
- In parameter counts của từng module
- Tính BLEU và METEOR (nếu NLTK có sẵn)

### 2. Pre-extraction - Trích xuất features trước

Để tăng tốc training, có thể trích xuất DeiT features trước:

```bash
python scripts/caption_ar.py \
    --pretrained \
    --vocab-path data/vocab.pkl \
    --preextract-features ./features_dir \
    --batch-size 8
```

Features sẽ được lưu vào `./features_dir/` dưới dạng file `.pt`

### 3. Training với pre-extracted features

Sử dụng features đã trích xuất để training nhanh hơn:

```bash
python scripts/caption_ar.py \
    --use-preextracted ./features_dir \
    --vocab-path data/vocab.pkl \
    --batch-size 8 \
    --epochs 10 \
    --lr 1e-4 \
    --compute-metrics
```

### 4. Inference Demo

Chạy inference trên một batch mẫu:

```bash
python scripts/caption_ar.py \
    --pretrained \
    --vocab-path data/vocab.pkl \
    --inference \
    --batch-size 4
```

## Tùy chọn cấu hình

### Kiến trúc mô hình

```bash
# Dimension của các layer
--proj-dim 512              # Projection dimension sau DeiT
--msgc-dim 256              # MSG_C output dimension
--emb-dim 512               # Embedding dimension

# MSG_C configuration
--msgc-scales 1 2 4         # Multi-scale pooling scales
--use-msgc                  # Bật MSG_C module
--msgc-fusion gated         # Fusion type: gated, concat, hoặc cross

# Encoder configuration
--enc-type bilstm           # Encoder type: bilstm, transformer, hoặc none
--enc-layers 2              # Số layer của encoder
--enc-dropout 0.1           # Dropout của encoder

# Decoder configuration
--dec-layers 2              # Số layer của LSTM decoder
--dec-dropout 0.1           # Dropout của decoder
--tie-embeddings            # Tie input/output embeddings (weight sharing)
```

### DeiT options

```bash
--pretrained                # Sử dụng pretrained DeiT weights
--freeze-deit               # Freeze DeiT parameters (không train)
```

### Training options

```bash
--batch-size 8              # Batch size
--epochs 10                 # Số epochs
--lr 1e-4                   # Learning rate
--teacher-forcing-ratio 1.0 # Teacher forcing ratio (1.0 = 100%)
--tgt-seq-len 20            # Target sequence length
```

### Metrics và logging

```bash
--compute-metrics           # Tính BLEU và METEOR khi validate
--checkpoint-dir ./ckpt     # Thư mục lưu checkpoints
```

## Ví dụ đầy đủ

### Training hoàn chỉnh với MSG_C

```bash
python scripts/caption_ar.py \
    --pretrained \
    --freeze-deit \
    --vocab-path data/vocab.pkl \
    --proj-dim 512 \
    --msgc-dim 256 \
    --msgc-scales 1 2 4 \
    --use-msgc \
    --msgc-fusion gated \
    --enc-type bilstm \
    --enc-layers 2 \
    --dec-layers 2 \
    --emb-dim 512 \
    --batch-size 16 \
    --tgt-seq-len 20 \
    --epochs 20 \
    --lr 1e-4 \
    --compute-metrics \
    --checkpoint-dir ./checkpoints
```

### So sánh các fusion types

Thử nghiệm các phương pháp fusion khác nhau:

**Gated fusion (recommended):**
```bash
python scripts/caption_ar.py --msgc-fusion gated ...
```

**Concat fusion:**
```bash
python scripts/caption_ar.py --msgc-fusion concat ...
```

**Cross-attention fusion:**
```bash
python scripts/caption_ar.py --msgc-fusion cross ...
```

## Output

Khi chạy script, bạn sẽ thấy:

1. **Device information**: GPU hoặc CPU
2. **Vocabulary statistics**: Vocab size, special token indices
3. **Model parameter counts**: 
   - DeiT parameters (total và trainable)
   - Encoder parameters
   - MSG_C parameters
   - Decoder parameters
   - Total parameters
4. **Training progress**: Loss per batch và epoch
5. **Tensor shapes** tại mỗi bước:
   - Visual features: `(batch, num_patches, deit_dim)`
   - Projected features: `(batch, num_patches, proj_dim)`
   - Encoder outputs: `(batch, seq_len, proj_dim)`
   - MSG_C tokens: `(batch, num_scales*seq_len, msgc_dim)`
   - MSG_C vec: `(batch, msgc_dim)`
   - Decoder logits: `(batch, tgt_seq_len, vocab_size)`
6. **Validation metrics**: Loss, BLEU, METEOR (nếu enabled)

## Checkpoints

Checkpoints được lưu tại `--checkpoint-dir`:
- `best_model.pt`: Model tốt nhất dựa trên validation loss

Checkpoint chứa:
- `model_state_dict`: Weights của model
- `optimizer_state_dict`: Optimizer state
- `epoch`: Epoch number
- `val_loss`: Validation loss
- `val_metrics`: Validation metrics (BLEU, METEOR)

## Ghi chú

- Script tự động phát hiện GPU và sử dụng nếu có
- Nếu không có dataset thật, script sẽ tạo dummy data để test
- Gradient clipping được áp dụng (max_norm=5.0) để stability
- Learning rate scheduler (ReduceLROnPlateau) được sử dụng
- Hỗ trợ teacher forcing với configurable ratio

## Khắc phục sự cố

**Lỗi: "timm not installed"**
```bash
pip install timm
```

**Lỗi: "NLTK not available"**
```bash
pip install nltk
```

**Lỗi loading vocab.pkl:**
- Thử convert sang JSON trước: `python scripts/load_vocab_from_tokenizer.py`
- Hoặc đảm bảo package `app.src.utils` có trong PYTHONPATH

**Out of memory:**
- Giảm `--batch-size`
- Giảm `--proj-dim` hoặc `--msgc-dim`
- Sử dụng `--freeze-deit` để giảm memory

**Training quá chậm:**
- Sử dụng pre-extraction: `--preextract-features` rồi `--use-preextracted`
- Freeze DeiT: `--freeze-deit`
- Tăng batch size nếu có GPU mạnh

## Liên hệ

Nếu có vấn đề, mở issue trên GitHub repository.
