#!/bin/bash

# --- 스크립트 설정 ---
set -e
set -x

# --- 사용자 설정 변수 ---
export CUDA_VISIBLE_DEVICES=0,1,2,3
N_GPUS=4
DATA_PATH="$HOME/.cache/huggingface/datasets"
OUTPUT_DIR="./training_output_soap"
# [수정] 스크립트를 모듈 형태로 지정
SCRIPT_MODULE="distributed_shampoo.vit_soap"

# 학습 기본 하이퍼파라미터
EPOCHS=90
BATCH_SIZE_PER_GPU=128
WORKERS=4

# 옵티마이저 및 스케줄러 하이퍼파라미터
BASE_LR=0.001
WARMUP_STEPS=11268
WEIGHT_DECAY=0.0001
BETA1=0.95

# 데이터 증강 하이퍼파라미터
MIXUP=0.2
LABEL_SMOOTHING=0.1
RESUME_FROM=""

# --- 실행 설정 ---
RUN_NAME="vit_soap_LR${BASE_LR}_WD${WEIGHT_DECAY}_B1${BETA1}_$(date +%Y%m%d_%H%M%S)"
LOG_PATH="$OUTPUT_DIR/$RUN_NAME/logs"
SAVE_DIR="$OUTPUT_DIR/$RUN_NAME/checkpoints"
mkdir -p "$LOG_PATH"
mkdir -p "$SAVE_DIR"

RESUME_OPTION=""
if [ ! -z "$RESUME_FROM" ]; then
    RESUME_OPTION="--resume $RESUME_FROM"
    echo "Resuming training from: $RESUME_FROM"
fi

# --- 분산 학습 실행 ---
echo "========================================================"
echo "Vision Transformer on ImageNet-1k Training (Algoperf Spec)"
echo "Optimizer: SOAP"
echo "GPUs: $N_GPUS"
echo "Total Batch Size: $(($N_GPUS * $BATCH_SIZE_PER_GPU))"
echo "LR: $BASE_LR, WD: $WEIGHT_DECAY, Beta1: $BETA1"
echo "Augmentations: RandAugment(m15-n2), Mixup($MIXUP), LS($LABEL_SMOOTHING)"
echo "Log Path: $LOG_PATH"
echo "========================================================"

# [수정] torchrun 실행 방식을 모듈(-m)로 변경
torchrun --standalone --nnodes=1 --nproc_per_node=$N_GPUS -m $SCRIPT_MODULE \
    --data-path "$DATA_PATH" \
    --log-dir "$LOG_PATH" \
    --save-dir "$SAVE_DIR" \
    $RESUME_OPTION \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE_PER_GPU \
    --workers $WORKERS \
    --base-lr $BASE_LR \
    --warmup-steps $WARMUP_STEPS \
    --weight-decay $WEIGHT_DECAY \
    --beta1 $BETA1 \
    --mixup $MIXUP \
    --label-smoothing $LABEL_SMOOTHING \
    --log-interval 200 \
    --save-interval 10

echo "Training finished successfully."
