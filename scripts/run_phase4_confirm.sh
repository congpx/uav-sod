#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

PROJECT="${PROJECT:-$ROOT_DIR/runs/phase4_confirm}"
MODEL_CFG="${MODEL_CFG:-$ROOT_DIR/configs/yolov8-p2-visdrone.yaml}"
EPOCHS="${EPOCHS:-50}"
PATIENCE="${PATIENCE:-30}"
IMGSZ="${IMGSZ:-800}"
BATCH="${BATCH:-8}"

apply_patch_original
train_and_val "P4_p2_base_800_confirm" "$PROJECT" "$MODEL_CFG" "$IMGSZ" "$BATCH" "$EPOCHS" "$PATIENCE"

apply_patch_iouterm 0.05
train_and_val "P4_p2_tal_a005_800_confirm" "$PROJECT" "$MODEL_CFG" "$IMGSZ" "$BATCH" "$EPOCHS" "$PATIENCE"

apply_patch_original
