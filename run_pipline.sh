#!/bin/bash

# --- CẤU HÌNH PIPELINE (Sửa tham số tại đây) ---
PROJECT_NAME="vi_en_transformer"
DATA_DIR="models"
CHECKPOINT_DIR="checkpoints"
SPM_MODEL_PATH="$DATA_DIR/spm.model"

# Tham số Model & Train
VOCAB_SIZE=8000
EMBED_DIM=64
HEADS=4
LAYERS=3
EPOCHS=1         
BATCH_SIZE=32     
LEARNING_RATE=3e-4

PAD_IDX=0

# Dừng script ngay lập tức nếu có lệnh bị lỗi
set -e

echo "========================================================"
echo "🚀 STARTING AI PIPELINE: $PROJECT_NAME"
echo "========================================================"

# 1. SETUP MÔI TRƯỜNG
echo ""
echo "[1/3] Setting up directories..."
mkdir -p $DATA_DIR
mkdir -p $CHECKPOINT_DIR
# Nếu chưa cài thư viện thì cài luôn (Optional)
pip install -r setup.txt

# 2. PREPROCESSING
# Bước này tải data và train SentencePiece
# Nếu file model đã tồn tại, ta có thể bỏ qua để tiết kiệm thời gian (Optional)
if [ -f "$SPM_MODEL_PATH" ]; then
    echo "[2/3] SPM Model found at $SPM_MODEL_PATH. Skipping preprocessing..."
    echo "      (Delete the file if you want to retrain tokenizer)"
else
    echo "[2/3] Running Preprocessing (Download Data & Train SPM)..."
    python src/preprocess.py
fi



# 3. TRAINING
echo ""
echo "[3/3] Starting Training..."
# Lưu ý: Anh dùng tee để vừa in ra màn hình vừa lưu log file
python src/train.py \
    --data_path "$SPM_MODEL_PATH" \
    --save_dir "$CHECKPOINT_DIR" \
    --vocab_size $VOCAB_SIZE \
    --embed_dim $EMBED_DIM \
    --heads $HEADS \
    --layers $LAYERS \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --pad_idx $PAD_IDX \
    | tee training_log.txt

# 4. EVALUATION
echo ""
echo "========================================================"
echo "📊 Training Finished. Running Evaluation..."
echo "========================================================"

# Tìm file checkpoint tốt nhất
BEST_MODEL="$CHECKPOINT_DIR/best_transformer.pth"

if [ -f "$BEST_MODEL" ]; then
    python src/evaluate.py \
        --checkpoint "$BEST_MODEL" \
        --spm_model "$SPM_MODEL_PATH" \
        --vocab_size $VOCAB_SIZE \
        --embed_dim $EMBED_DIM \
        --heads $HEADS \
        --layers $LAYERS \
        --pad_idx $PAD_IDX \
        --beam_size 4 \
        --test_samples 200
else
    echo "❌ Error: Best model not found at $BEST_MODEL"
    exit 1
fi

echo ""
echo "✅ PIPELINE COMPLETED SUCCESSFULLY!"
echo "   - Model saved in: $CHECKPOINT_DIR"
echo "   - Logs saved in: training_log.txt"