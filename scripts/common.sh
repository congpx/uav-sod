#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/congpx/uav_yolo_visdrone}"
REPO_DIR="${REPO_DIR:-$ROOT_DIR/ultralytics}"
cd "$REPO_DIR"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON="$VENV_DIR/bin/python"
DATA_YAML="${DATA_YAML:-/home/congpx/datasets/VisDrone-YOLO/visdrone.yaml}"
PRETRAINED="${PRETRAINED:-yolov8n.pt}"
CACHE_MODE="${CACHE_MODE:-disk}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-0}"
SEED="${SEED:-42}"

if [ ! -x "$PYTHON" ]; then
  echo "[ERROR] Python not found: $PYTHON" >&2
  exit 1
fi

if [ ! -f "$DATA_YAML" ]; then
  echo "[ERROR] DATA_YAML not found: $DATA_YAML" >&2
  exit 1
fi

YOLO_BIN="$VENV_DIR/bin/yolo"

run_yolo() {
  "$YOLO_BIN" "$@"
}

apply_patch_original() {
  "$PYTHON" "$ROOT_DIR/scripts/apply_tal_patch.py" --repo-root "$REPO_DIR" --mode original
}

apply_patch_safe() {
  local gamma="$1"
  "$PYTHON" "$ROOT_DIR/scripts/apply_tal_patch.py" --repo-root "$REPO_DIR" --mode safe --gamma "$gamma" --aa-alpha 0.05 --aa-tau 512 --aa-max 1.3
}

apply_patch_iouterm() {
  local aa_alpha="$1"
  "$PYTHON" "$ROOT_DIR/scripts/apply_tal_patch.py" --repo-root "$REPO_DIR" --mode iouterm --aa-alpha "$aa_alpha" --aa-tau 512 --aa-max 1.3
}

train_and_val() {
  local runname="$1"
  local project="$2"
  local model_cfg="$3"
  local imgsz="$4"
  local batch="$5"
  local epochs="$6"
  local patience="$7"

  rm -rf "$project/$runname" "$project/${runname}_val_final"
  mkdir -p "$project"

  run_yolo detect train \
    model="$model_cfg" \
    data="$DATA_YAML" \
    pretrained="$PRETRAINED" \
    epochs="$epochs" \
    imgsz="$imgsz" \
    batch="$batch" \
    workers="$WORKERS" \
    device="$DEVICE" \
    project="$project" \
    name="$runname" \
    cache="$CACHE_MODE" \
    seed="$SEED" \
    patience="$patience" \
    exist_ok=False

  if [ ! -f "$project/$runname/weights/best.pt" ]; then
    echo "[ERROR] best.pt not found for $runname" >&2
    exit 1
  fi

  run_yolo detect val \
    model="$project/$runname/weights/best.pt" \
    data="$DATA_YAML" \
    split=val \
    imgsz="$imgsz" \
    batch="$batch" \
    workers="$WORKERS" \
    device="$DEVICE" \
    project="$project" \
    name="${runname}_val_final" \
    exist_ok=False
}
