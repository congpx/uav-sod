#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

PROJECT="${PROJECT:-$ROOT_DIR/runs/phase3_p2}"
MODEL_CFG="${MODEL_CFG:-$ROOT_DIR/configs/yolov8-p2-visdrone.yaml}"
EPOCHS="${EPOCHS:-20}"
PATIENCE="${PATIENCE:-20}"
RUN_800="${RUN_800:-1}"
BATCH_640="${BATCH_640:-16}"
BATCH_800="${BATCH_800:-8}"

apply_patch_original
train_and_val "P3_p2_base_640" "$PROJECT" "$MODEL_CFG" 640 "$BATCH_640" "$EPOCHS" "$PATIENCE"

apply_patch_iouterm 0.05
train_and_val "P3_p2_tal_a005_640" "$PROJECT" "$MODEL_CFG" 640 "$BATCH_640" "$EPOCHS" "$PATIENCE"

if [ "$RUN_800" = "1" ]; then
  apply_patch_original
  train_and_val "P3_p2_base_800" "$PROJECT" "$MODEL_CFG" 800 "$BATCH_800" "$EPOCHS" "$PATIENCE"

  apply_patch_iouterm 0.05
  train_and_val "P3_p2_tal_a005_800" "$PROJECT" "$MODEL_CFG" 800 "$BATCH_800" "$EPOCHS" "$PATIENCE"
fi

apply_patch_original
