#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/congpx/uav_yolo_visdrone"
VENV="$ROOT/.venv"
REPO="$ROOT/ultralytics"
YOLO_BIN="$VENV/bin/yolo"
PYTHON_BIN="$VENV/bin/python"

VAL_IMG_DIR="/home/congpx/datasets/VisDrone-YOLO/images/val"
PRED_ROOT="$ROOT/runs/paper_pred_dump"
OUT_DIR="$ROOT/paper_figures/qualitative"

MODEL_640="$ROOT/runs/phase4_confirm_640/P4_p2_base_640_confirm/weights/best.pt"
MODEL_800="$ROOT/runs/phase4_confirm/P4_p2_base_800_confirm/weights/best.pt"
MODEL_TAL_800="$ROOT/runs/phase4_confirm/P4_p2_tal_a005_800_confirm/weights/best.pt"

mkdir -p "$PRED_ROOT" "$OUT_DIR"

source "$VENV/bin/activate"
cd "$REPO"

need_pred_dir() {
  local d="$1"
  if [ ! -d "$d/labels" ]; then
    return 0
  fi
  find "$d/labels" -maxdepth 1 -type f -name '*.txt' | grep -q . || return 0
  return 1
}

run_predict_if_needed() {
  local name="$1"
  local model="$2"
  local imgsz="$3"
  local outdir="$PRED_ROOT/$name"

  if need_pred_dir "$outdir"; then
    echo "[RUN] predict $name"
    "$YOLO_BIN" detect predict \
      model="$model" \
      source="$VAL_IMG_DIR" \
      imgsz="$imgsz" \
      conf=0.001 \
      save_txt=True \
      save_conf=True \
      project="$PRED_ROOT" \
      name="$name" \
      exist_ok=True
  else
    echo "[SKIP] prediction exists: $name"
  fi
}

run_predict_if_needed "p2_base_640" "$MODEL_640" 640
run_predict_if_needed "p2_base_800" "$MODEL_800" 800
run_predict_if_needed "p2_tal_800" "$MODEL_TAL_800" 800

"$PYTHON_BIN" "$ROOT/scripts/make_paper_qualitative.py"