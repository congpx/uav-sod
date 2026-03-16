#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/congpx/uav_yolo_visdrone}"
export DATA_YAML="${DATA_YAML:-/home/congpx/datasets/UAVDT-YOLO/uavdt.yaml}"

# Reuse common helpers from the VisDrone project
source "$ROOT_DIR/scripts/common.sh"

# Model configs
Y8_BASE_CFG="${Y8_BASE_CFG:-yolov8n.yaml}"
P2_CFG="${P2_CFG:-$ROOT_DIR/configs/yolov8-p2-uavdt.yaml}"

if [ ! -f "$DATA_YAML" ]; then
  echo "[ERROR] UAVDT yaml not found: $DATA_YAML" >&2
  exit 1
fi

if [ ! -f "$P2_CFG" ]; then
  echo "[ERROR] P2 config not found: $P2_CFG" >&2
  exit 1
fi

echo "[INFO] DATA_YAML = $DATA_YAML"
echo "[INFO] Y8_BASE_CFG = $Y8_BASE_CFG"
echo "[INFO] P2_CFG = $P2_CFG"
