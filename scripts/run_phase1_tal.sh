#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

PROJECT="${PROJECT:-$ROOT_DIR/runs/phase1_tal}"
MODEL="${MODEL:-yolov8n.pt}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-16}"
EPOCHS="${EPOCHS:-20}"
PATIENCE="${PATIENCE:-20}"

apply_patch_original
train_and_val "P1_baseline" "$PROJECT" "$MODEL" "$IMGSZ" "$BATCH" "$EPOCHS" "$PATIENCE"

apply_patch_safe 0.5
train_and_val "P1_tal_safe_g05" "$PROJECT" "$MODEL" "$IMGSZ" "$BATCH" "$EPOCHS" "$PATIENCE"

apply_patch_safe 1.0
train_and_val "P1_tal_safe_g10" "$PROJECT" "$MODEL" "$IMGSZ" "$BATCH" "$EPOCHS" "$PATIENCE"

apply_patch_iouterm 0.05
train_and_val "P1_tal_iouterm_a005_tau512_m13" "$PROJECT" "$MODEL" "$IMGSZ" "$BATCH" "$EPOCHS" "$PATIENCE"

apply_patch_iouterm 0.10
train_and_val "P1_tal_iouterm_a010_tau512_m13" "$PROJECT" "$MODEL" "$IMGSZ" "$BATCH" "$EPOCHS" "$PATIENCE"

apply_patch_original
