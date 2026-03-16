#!/usr/bin/env bash
set -euo pipefail

# =========================
# Config
# =========================
ROOT_DIR="${ROOT_DIR:-/home/congpx/datasets/UAVDT}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-$ROOT_DIR/downloads}"
EXTRACT_DIR="${EXTRACT_DIR:-$ROOT_DIR/raw}"
INSTALL_TOOLKIT="${INSTALL_TOOLKIT:-1}"
INSTALL_ATTRS="${INSTALL_ATTRS:-1}"

# Official UAVDT Google Drive file IDs
# Source: official UAVDT Google Sites "Downloads" page
DATASET_ID="1m8KA6oPIRK_Iwt9TYFquC87vBc_8wRVc"      # UAV-benchmark-M.zip
TOOLKIT_ID="19498uJd7T9w4quwnQEy62nibt3uyT9pq"      # UAV-benchmark-MOTD_v1.0.zip
ATTRS_ID="1qjipvuk3XE3qU3udluQRRcYuiKzhMXB1"        # M_attr.zip

DATASET_ZIP="$DOWNLOAD_DIR/UAV-benchmark-M.zip"
TOOLKIT_ZIP="$DOWNLOAD_DIR/UAV-benchmark-MOTD_v1.0.zip"
ATTRS_ZIP="$DOWNLOAD_DIR/M_attr.zip"

# =========================
# Helpers
# =========================
log() {
  echo "[$(date '+%F %T')] $*"
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "[ERROR] Missing command: $1" >&2
    exit 1
  }
}

download_gdrive() {
  local file_id="$1"
  local out_path="$2"

  if [ -f "$out_path" ]; then
    log "Already exists, skip: $out_path"
    return 0
  fi

  log "Downloading to: $out_path"
  gdown --fuzzy "https://drive.google.com/file/d/${file_id}/view?usp=sharing" -O "$out_path"
}

extract_zip() {
  local zip_path="$1"
  local out_dir="$2"

  mkdir -p "$out_dir"
  log "Extracting: $zip_path -> $out_dir"
  unzip -q -o "$zip_path" -d "$out_dir"
}

# =========================
# Main
# =========================
mkdir -p "$DOWNLOAD_DIR" "$EXTRACT_DIR"

need_cmd python3
need_cmd unzip

if ! command -v gdown >/dev/null 2>&1; then
  log "gdown not found. Installing with pip..."
  python3 -m pip install --user --upgrade gdown
  export PATH="$HOME/.local/bin:$PATH"
fi

need_cmd gdown

log "=========================================="
log "Downloading official UAVDT files"
log "ROOT_DIR      = $ROOT_DIR"
log "DOWNLOAD_DIR  = $DOWNLOAD_DIR"
log "EXTRACT_DIR   = $EXTRACT_DIR"
log "=========================================="

# Dataset
download_gdrive "$DATASET_ID" "$DATASET_ZIP"
extract_zip "$DATASET_ZIP" "$EXTRACT_DIR"

# Optional toolkit
if [ "$INSTALL_TOOLKIT" = "1" ]; then
  download_gdrive "$TOOLKIT_ID" "$TOOLKIT_ZIP"
  extract_zip "$TOOLKIT_ZIP" "$EXTRACT_DIR/toolkit"
fi

# Optional attributes
if [ "$INSTALL_ATTRS" = "1" ]; then
  download_gdrive "$ATTRS_ID" "$ATTRS_ZIP"
  extract_zip "$ATTRS_ZIP" "$EXTRACT_DIR/attributes"
fi

log "=========================================="
log "Done."
log "Dataset extracted under: $EXTRACT_DIR"
log "=========================================="

echo
echo "Next step:"
echo "  Inspect the extracted folders, then convert UAVDT to your YOLO layout if needed."