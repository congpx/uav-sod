#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/congpx/uav_yolo_visdrone}"
DATA_YAML="${DATA_YAML:-/data/visdrone/visdrone.yaml}"
export ROOT_DIR DATA_YAML

"$ROOT_DIR/scripts/run_phase1_tal.sh"
"$ROOT_DIR/scripts/run_phase3_p2.sh"
"$ROOT_DIR/scripts/run_phase4_confirm.sh"
