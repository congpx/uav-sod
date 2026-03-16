from pathlib import Path
from PIL import Image
import math
import csv

# =========================
# Config
# =========================
VAL_IMG_DIR = Path("/home/congpx/datasets/VisDrone-YOLO/images/val")
VAL_GT_DIR = Path("/home/congpx/datasets/VisDrone-YOLO/labels/val")

MODEL_PRED_DIRS = {
    "P2-base-640": Path("/home/congpx/uav_yolo_visdrone/runs/visdrone_pred_dump/p2_base_640/labels"),
    "P2+TAL-640": Path("/home/congpx/uav_yolo_visdrone/runs/visdrone_pred_dump/p2_tal_640/labels"),
    "P2-base-800": Path("/home/congpx/uav_yolo_visdrone/runs/visdrone_pred_dump/p2_base_800/labels"),
    "P2+TAL-800": Path("/home/congpx/uav_yolo_visdrone/runs/visdrone_pred_dump/p2_tal_800/labels"),
}

OUT_CSV = Path("/home/congpx/uav_yolo_visdrone/runs/visdrone_pred_dump/size_recall_summary.csv")
IOU_THR = 0.5

# size bins by box area ratio
# tiny: < 0.1%
# small: 0.1% - 0.5%
# medium+: >= 0.5%
def size_bin(area_ratio: float) -> str:
    if area_ratio < 0.001:
        return "tiny"
    elif area_ratio < 0.005:
        return "small"
    return "medium+"

def xywhn_to_xyxy(xc, yc, w, h, W, H):
    bw = w * W
    bh = h * H
    x1 = (xc * W) - bw / 2
    y1 = (yc * H) - bh / 2
    x2 = (xc * W) + bw / 2
    y2 = (yc * H) + bh / 2
    return [x1, y1, x2, y2]

def box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    a1 = box_area(box1)
    a2 = box_area(box2)
    return inter / max(a1 + a2 - inter, 1e-9)

def load_gt(txt_path, W, H):
    gts = []
    if not txt_path.exists():
        return gts
    for line in txt_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:5])
        box = xywhn_to_xyxy(xc, yc, w, h, W, H)
        gts.append({"cls": cls_id, "box": box})
    return gts

def load_preds(txt_path, W, H):
    preds = []
    if not txt_path.exists():
        return preds
    for line in txt_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:5])
        conf = float(parts[5]) if len(parts) >= 6 else 1.0
        box = xywhn_to_xyxy(xc, yc, w, h, W, H)
        preds.append({"cls": cls_id, "box": box, "conf": conf})
    preds.sort(key=lambda x: x["conf"], reverse=True)
    return preds

def match_predictions(gts, preds, W, H, iou_thr=0.5):
    matched_gt = [False] * len(gts)

    for pred in preds:
        best_iou = 0.0
        best_idx = -1
        for i, gt in enumerate(gts):
            if matched_gt[i]:
                continue
            if pred["cls"] != gt["cls"]:
                continue
            cur_iou = iou(pred["box"], gt["box"])
            if cur_iou >= iou_thr and cur_iou > best_iou:
                best_iou = cur_iou
                best_idx = i
        if best_idx >= 0:
            matched_gt[best_idx] = True

    stats = {"tiny": {"gt": 0, "tp": 0},
             "small": {"gt": 0, "tp": 0},
             "medium+": {"gt": 0, "tp": 0}}

    img_area = float(W * H)
    for i, gt in enumerate(gts):
        ratio = box_area(gt["box"]) / img_area
        b = size_bin(ratio)
        stats[b]["gt"] += 1
        if matched_gt[i]:
            stats[b]["tp"] += 1

    return stats

def merge_stats(dst, src):
    for b in dst:
        dst[b]["gt"] += src[b]["gt"]
        dst[b]["tp"] += src[b]["tp"]

def main():
    image_files = sorted([p for p in VAL_IMG_DIR.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]])

    total = {
        name: {
            "tiny": {"gt": 0, "tp": 0},
            "small": {"gt": 0, "tp": 0},
            "medium+": {"gt": 0, "tp": 0},
        } for name in MODEL_PRED_DIRS
    }

    for img_path in image_files:
        with Image.open(img_path) as im:
            W, H = im.size

        stem = img_path.stem
        gt_path = VAL_GT_DIR / f"{stem}.txt"
        gts = load_gt(gt_path, W, H)

        for name, pred_dir in MODEL_PRED_DIRS.items():
            pred_path = pred_dir / f"{stem}.txt"
            preds = load_preds(pred_path, W, H)
            cur_stats = match_predictions(gts, preds, W, H, iou_thr=IOU_THR)
            merge_stats(total[name], cur_stats)

    rows = []
    for name, stats in total.items():
        row = {"model": name}
        for b in ["tiny", "small", "medium+"]:
            gt = stats[b]["gt"]
            tp = stats[b]["tp"]
            rec = tp / gt if gt > 0 else 0.0
            row[f"{b}_gt"] = gt
            row[f"{b}_tp"] = tp
            row[f"{b}_recall"] = rec
        rows.append(row)

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\n===== SIZE-AWARE RECALL SUMMARY =====")
    for row in rows:
        print(row)

    print(f"\nSaved to: {OUT_CSV}")

if __name__ == "__main__":
    main()