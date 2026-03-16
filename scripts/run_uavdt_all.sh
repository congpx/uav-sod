#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/congpx/uav_yolo_visdrone}"
source "$ROOT_DIR/scripts/common_uavdt.sh"

PROJECT="${PROJECT:-$ROOT_DIR/runs/uavdt_study}"
EPOCHS="${EPOCHS:-50}"
PATIENCE="${PATIENCE:-30}"

# Safe defaults for RTX 5060 Ti 16 GB
BATCH_Y8_640="${BATCH_Y8_640:-16}"
BATCH_P2_640="${BATCH_P2_640:-12}"
BATCH_P2_800="${BATCH_P2_800:-8}"

log() {
  echo "[$(date '+%F %T')] $*"
}

log "======================================================"
log "UAVDT study started"
log "Project root: $PROJECT"
log "======================================================"

mkdir -p "$PROJECT"

# --------------------------
# Stage A: TAL on YOLOv8n baseline
# --------------------------
log "Stage A: TAL effect on YOLOv8n baseline @640"

apply_patch_original
train_and_val \
  "UAVDT_A_y8n_base_640" \
  "$PROJECT/stageA" \
  "$Y8_BASE_CFG" \
  640 \
  "$BATCH_Y8_640" \
  "$EPOCHS" \
  "$PATIENCE"

apply_patch_iouterm 0.05
train_and_val \
  "UAVDT_A_y8n_tal_a005_640" \
  "$PROJECT/stageA" \
  "$Y8_BASE_CFG" \
  640 \
  "$BATCH_Y8_640" \
  "$EPOCHS" \
  "$PATIENCE"

# --------------------------
# Stage B: P2 representation screening
# --------------------------
log "Stage B: P2 representation screening @640 and @800"

apply_patch_original
train_and_val \
  "UAVDT_B_p2_base_640" \
  "$PROJECT/stageB" \
  "$P2_CFG" \
  640 \
  "$BATCH_P2_640" \
  "$EPOCHS" \
  "$PATIENCE"

apply_patch_original
train_and_val \
  "UAVDT_B_p2_base_800" \
  "$PROJECT/stageB" \
  "$P2_CFG" \
  800 \
  "$BATCH_P2_800" \
  "$EPOCHS" \
  "$PATIENCE"

# --------------------------
# Stage C: TAL on top of P2
# --------------------------
log "Stage C: TAL effect on P2 detector @640 and @800"

apply_patch_iouterm 0.05
train_and_val \
  "UAVDT_C_p2_tal_a005_640" \
  "$PROJECT/stageC" \
  "$P2_CFG" \
  640 \
  "$BATCH_P2_640" \
  "$EPOCHS" \
  "$PATIENCE"

apply_patch_iouterm 0.05
train_and_val \
  "UAVDT_C_p2_tal_a005_800" \
  "$PROJECT/stageC" \
  "$P2_CFG" \
  800 \
  "$BATCH_P2_800" \
  "$EPOCHS" \
  "$PATIENCE"

apply_patch_original

log "======================================================"
log "UAVDT study finished"
log "Now summarizing results..."
log "======================================================"

python "$ROOT_DIR/scripts/summarize_uavdt.py"
