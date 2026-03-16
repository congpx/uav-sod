#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

ROOT = Path("/home/congpx/uav_yolo_visdrone/runs/uavdt_study")

RUNS = {
    "UAVDT_A_y8n_base_640": ROOT / "stageA/UAVDT_A_y8n_base_640/results.csv",
    "UAVDT_A_y8n_tal_a005_640": ROOT / "stageA/UAVDT_A_y8n_tal_a005_640/results.csv",
    "UAVDT_B_p2_base_640": ROOT / "stageB/UAVDT_B_p2_base_640/results.csv",
    "UAVDT_B_p2_base_800": ROOT / "stageB/UAVDT_B_p2_base_800/results.csv",
    "UAVDT_C_p2_tal_a005_640": ROOT / "stageC/UAVDT_C_p2_tal_a005_640/results.csv",
    "UAVDT_C_p2_tal_a005_800": ROOT / "stageC/UAVDT_C_p2_tal_a005_800/results.csv",
}

def load_last(csv_path: Path):
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    last = df.iloc[-1]
    return {
        "P": float(last["metrics/precision(B)"]),
        "R": float(last["metrics/recall(B)"]),
        "mAP50": float(last["metrics/mAP50(B)"]),
        "mAP50-95": float(last["metrics/mAP50-95(B)"]),
    }

rows = []
for name, path in RUNS.items():
    m = load_last(path)
    if m is None:
        rows.append({"run": name, "P": None, "R": None, "mAP50": None, "mAP50-95": None})
    else:
        rows.append({"run": name, **m})

out = pd.DataFrame(rows)

print("\n===== UAVDT SUMMARY =====")
print(out.to_string(index=False))

def metric(run, col):
    row = out[out["run"] == run]
    if row.empty or pd.isna(row.iloc[0][col]):
        return None
    return float(row.iloc[0][col])

def show_delta(name1, name2, col, label):
    a = metric(name1, col)
    b = metric(name2, col)
    if a is None or b is None:
        return
    print(f"{label:<35s} {col:<10s}: {a - b:+.4f}")

print("\n===== KEY DELTAS =====")
for col in ["P", "R", "mAP50", "mAP50-95"]:
    show_delta("UAVDT_A_y8n_tal_a005_640", "UAVDT_A_y8n_base_640", col, "TAL effect on Y8n @640")
    show_delta("UAVDT_B_p2_base_640", "UAVDT_A_y8n_base_640", col, "P2 benefit @640 vs Y8n")
    show_delta("UAVDT_B_p2_base_800", "UAVDT_B_p2_base_640", col, "800 vs 640 on P2 base")
    show_delta("UAVDT_C_p2_tal_a005_640", "UAVDT_B_p2_base_640", col, "TAL effect on P2 @640")
    show_delta("UAVDT_C_p2_tal_a005_800", "UAVDT_B_p2_base_800", col, "TAL effect on P2 @800")
    print("-" * 62)
