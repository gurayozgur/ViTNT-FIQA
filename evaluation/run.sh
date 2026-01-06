#!/bin/bash

MODEL_PATH="../pretrained/minchul_cvlface_adaface_vit_base_webface4m.pt"
MODEL_NAME="minchul_cvlface_adaface_vit_base_webface4m.pt"
BACKBONE="vitb"
DATASETS="IJBC"
GPU_ID=0
SCALING=8.0
BATCH_SIZE=32
COLOR_CHANNEL="BGR"
OUTPUT_DIR="../results/vitb_wf4m/scores"
BLOCKS=$(seq -s, 0 19)
python getQualityScore.py \
    --data-dir "../data/" \
    --output-dir "$OUTPUT_DIR" \
    --datasets "$DATASETS" \
    --gpu-id $GPU_ID \
    --model-path "../pretrained/" \
    --model-name "$MODEL_NAME" \
    --backbone "$BACKBONE" \
    --scaling $SCALING \
    --ntfiq-use-attention-weights true \
    --last-block-attention-only true \
    --blocks-to-use "$BLOCKS" \
    --batch-size $BATCH_SIZE \
    --color-channel $COLOR_CHANNEL
echo "Finished"
