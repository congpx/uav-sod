#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/congpx/uav_yolo_visdrone}"
REPO_DIR="$ROOT_DIR/ultralytics"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ULTRA_TAG="${ULTRA_TAG:-v8.4.11}"

mkdir -p "$ROOT_DIR"
cd "$ROOT_DIR"

sudo apt-get update
sudo apt-get install -y git python3-venv python3-pip build-essential libgl1 libglib2.0-0

if [ ! -d "$REPO_DIR/.git" ]; then
  git clone https://github.com/ultralytics/ultralytics.git "$REPO_DIR"
fi

cd "$REPO_DIR"
git fetch --tags
# Detached HEAD at the exact tag used in your Windows runs
git checkout "$ULTRA_TAG"

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel

# Let PyTorch choose the CUDA build suited to the installed driver/toolkit.
# Replace the next command with the exact command from https://pytorch.org/get-started/locally if needed.
pip install torch torchvision torchaudio
pip install -e "$REPO_DIR"

mkdir -p "$ROOT_DIR/configs" "$ROOT_DIR/runs" "$ROOT_DIR/legacy_results"
cp -f "$ROOT_DIR/ultralytics/cfg/models/v8/yolov8-p2.yaml" "$ROOT_DIR/configs/yolov8-p2-upstream.yaml" 2>/dev/null || true

echo "Setup done. Run:"
echo "  source $VENV_DIR/bin/activate"
echo "  python $ROOT_DIR/scripts/check_env.py"
