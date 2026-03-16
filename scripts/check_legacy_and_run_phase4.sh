#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/congpx/uav_yolo_visdrone}"
PHASE1_DIR="${PHASE1_DIR:-$ROOT_DIR/legacy_results/phase1}"
PHASE3_DIR="${PHASE3_DIR:-$ROOT_DIR/legacy_results/phase3}"
CHECK_ONLY="${CHECK_ONLY:-0}"

if [ ! -d "$ROOT_DIR" ]; then
  echo "[ERROR] ROOT_DIR not found: $ROOT_DIR"
  exit 1
fi

if [ ! -f "$ROOT_DIR/scripts/common.sh" ]; then
  echo "[ERROR] Missing common.sh: $ROOT_DIR/scripts/common.sh"
  exit 1
fi

# source env/config, but suppress tal-path print from common.sh
source "$ROOT_DIR/scripts/common.sh" >/dev/null
cd "$REPO_DIR"

if [ ! -f "$DATA_YAML" ]; then
  echo "[ERROR] DATA_YAML not found: $DATA_YAML"
  exit 1
fi

if [ ! -f "$ROOT_DIR/scripts/run_phase4_confirm.sh" ]; then
  echo "[ERROR] Missing phase4 runner: $ROOT_DIR/scripts/run_phase4_confirm.sh"
  exit 1
fi

python3 - "$PHASE1_DIR" "$PHASE3_DIR" <<'PY'
import sys
from pathlib import Path
import pandas as pd

phase1_dir = Path(sys.argv[1])
phase3_dir = Path(sys.argv[2])

required_phase1 = {
    "baseline": phase1_dir / "P1_baseline_results.csv",
    "safe_g05": phase1_dir / "P1_tal_safe_g05_results.csv",
    "safe_g10": phase1_dir / "P1_tal_safe_g10_results.csv",
    "iouterm_a005": phase1_dir / "P1_tal_iouterm_a005_tau512_m13_results.csv",
    "iouterm_a010": phase1_dir / "P1_tal_iouterm_a010_tau512_m13_results.csv",
}
required_phase3 = {
    "p3_640": phase3_dir / "P3_p2_tal_a005_640.csv",
    "p3_800": phase3_dir / "P3_p2_tal_a005_800.csv",
}

optional_files = [
    phase1_dir / "P1_baseline_confusion_matrix.png",
    phase1_dir / "P1_tal_iouterm_a005_tau512_m13_confusion_matrix.png",
]

missing = [str(p) for p in list(required_phase1.values()) + list(required_phase3.values()) if not p.exists()]
if missing:
    print("[ERROR] Missing required files:")
    for m in missing:
        print("  -", m)
    sys.exit(2)

def load_last_metrics(csv_path: Path):
    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError(f"Empty CSV: {csv_path}")

    last = df.iloc[-1]
    cols = {
        "P": "metrics/precision(B)",
        "R": "metrics/recall(B)",
        "mAP50": "metrics/mAP50(B)",
        "mAP5095": "metrics/mAP50-95(B)",
    }
    out = {}
    for k, c in cols.items():
        if c not in df.columns:
            raise RuntimeError(f"Missing column '{c}' in {csv_path}")
        out[k] = float(last[c])
    return out

m = {k: load_last_metrics(v) for k, v in {**required_phase1, **required_phase3}.items()}

baseline = m["baseline"]
safe_g05 = m["safe_g05"]
safe_g10 = m["safe_g10"]
a005 = m["iouterm_a005"]
a010 = m["iouterm_a010"]
p3_640 = m["p3_640"]
p3_800 = m["p3_800"]

best_phase1_name = max(
    ["baseline", "safe_g05", "safe_g10", "iouterm_a005", "iouterm_a010"],
    key=lambda k: m[k]["mAP5095"]
)
best_phase1 = m[best_phase1_name]

issues = []

# Phase 1 sanity
if best_phase1["mAP5095"] + 1e-12 < baseline["mAP5095"]:
    issues.append("Phase 1 không có run nào bằng hoặc vượt baseline theo mAP50-95.")
if max(a005["mAP5095"], a010["mAP5095"]) + 1e-12 < max(safe_g05["mAP5095"], safe_g10["mAP5095"]):
    issues.append("Phase 1 không còn cho thấy iouterm tốt ngang hoặc tốt hơn safe.")

# Phase 3 sanity
if p3_800["mAP5095"] <= p3_640["mAP5095"]:
    issues.append("Phase 3: P3_p2_tal_a005_800 không tốt hơn P3_p2_tal_a005_640.")
if p3_800["mAP5095"] <= best_phase1["mAP5095"]:
    issues.append("Phase 3: P3_p2_tal_a005_800 không tốt hơn best Phase 1.")
if (p3_800["mAP5095"] - p3_640["mAP5095"]) < 0.01:
    issues.append("Phase 3: lợi thế của imgsz=800 so với 640 quá nhỏ (<0.01 mAP50-95).")

print("\n===== SUMMARY =====")
for name in ["baseline", "safe_g05", "safe_g10", "iouterm_a005", "iouterm_a010", "p3_640", "p3_800"]:
    mm = m[name]
    print(f"{name:12s}  P={mm['P']:.5f}  R={mm['R']:.5f}  mAP50={mm['mAP50']:.5f}  mAP50-95={mm['mAP5095']:.5f}")

print("\nBest Phase 1:", best_phase1_name, f"(mAP50-95={best_phase1['mAP5095']:.5f})")
print("Phase 3 best :", "p3_800", f"(mAP50-95={p3_800['mAP5095']:.5f})")

optional_missing = [str(p) for p in optional_files if not p.exists()]
if optional_missing:
    print("\n[WARN] Missing optional files:")
    for p in optional_missing:
        print("  -", p)

if issues:
    print("\n[FAIL] Validation failed:")
    for s in issues:
        print("  -", s)
    sys.exit(3)

print("\n[PASS] Legacy Phase 1-3 results look consistent. Safe to start Phase 4.")
PY

if [ "$CHECK_ONLY" = "1" ]; then
  echo "[INFO] CHECK_ONLY=1, not running Phase 4."
  exit 0
fi

echo "[INFO] Starting Phase 4..."
bash "$ROOT_DIR/scripts/run_phase4_confirm.sh"
