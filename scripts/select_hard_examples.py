from pathlib import Path
from PIL import Image
import csv

# =========================================================
# CONFIG
# =========================================================

VAL_IMG_DIR = Path("/home/congpx/datasets/VisDrone-YOLO/images/val")
VAL_GT_DIR = Path("/home/congpx/datasets/VisDrone-YOLO/labels/val")

PRED_DIRS = {
    "P2-base-640": Path("/home/congpx/uav_yolo_visdrone/runs/visdrone_pred_dump/p2_base_640/labels"),
    "P2-base-800": Path("/home/congpx/uav_yolo_visdrone/runs/visdrone_pred_dump/p2_base_800/labels"),
    "P2+TAL-800": Path("/home/congpx/uav_yolo_visdrone/runs/visdrone_pred_dump/p2_tal_800/labels"),
}

# -------------------------
# Mode 1: xếp theo số GT bị bỏ sót nhiều nhất của 1 model
# -------------------------
RANK_MODE = "most_recovered"
WEAK_MODEL = "P2-base-640"
STRONG_MODEL = "P2-base-800"
USE_TINY_PRIORITY = True
TOP_K = 20
# ------------------------------------------------
# Tham số matching
# ------------------------------------------------
IOU_THR = 0.5

# Chỉ ưu tiên các ảnh có nhiều tiny object?
USE_TINY_PRIORITY = True
TINY_AREA_RATIO = 0.001  # <0.1% diện tích ảnh

TOP_K = 20

OUT_DIR = Path("/home/congpx/uav_yolo_visdrone/runs/visdrone_pred_dump/hard_examples")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "hard_examples_summary.csv"
OUT_TXT = OUT_DIR / "selected_image_stems.txt"


# =========================================================
# HELPERS
# =========================================================

def find_image_by_stem(stem: str) -> Path:
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        p = VAL_IMG_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Image not found for stem: {stem}")


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


def load_yolo_txt(txt_path: Path, W: int, H: int):
    objs = []
    if txt_path is None or not txt_path.exists():
        return objs

    text = txt_path.read_text().strip()
    if not text:
        return objs

    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:5])
        conf = float(parts[5]) if len(parts) >= 6 else None
        box = xywhn_to_xyxy(xc, yc, w, h, W, H)
        objs.append({
            "cls": cls_id,
            "box": box,
            "conf": conf
        })

    if any(obj["conf"] is not None for obj in objs):
        objs.sort(key=lambda x: x["conf"] if x["conf"] is not None else 1.0, reverse=True)

    return objs


def match_gt_with_preds(gts, preds, iou_thr=0.5):
    """
    Match 1-1 by class and IoU.
    Return:
      matched_gt: list[bool]
    """
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

    return matched_gt


def compute_image_stats_for_model(stem, model_name):
    img_path = find_image_by_stem(stem)
    with Image.open(img_path) as im:
        W, H = im.size

    gt_path = VAL_GT_DIR / f"{stem}.txt"
    pred_path = PRED_DIRS[model_name] / f"{stem}.txt"

    gts = load_yolo_txt(gt_path, W, H)
    preds = load_yolo_txt(pred_path, W, H)

    matched_gt = match_gt_with_preds(gts, preds, iou_thr=IOU_THR)

    img_area = float(W * H)

    total_gt = len(gts)
    missed_total = 0
    tiny_gt = 0
    tiny_missed = 0

    for gt, matched in zip(gts, matched_gt):
        area_ratio = box_area(gt["box"]) / img_area
        is_tiny = area_ratio < TINY_AREA_RATIO

        if is_tiny:
            tiny_gt += 1

        if not matched:
            missed_total += 1
            if is_tiny:
                tiny_missed += 1

    return {
        "stem": stem,
        "total_gt": total_gt,
        "missed_total": missed_total,
        "tiny_gt": tiny_gt,
        "tiny_missed": tiny_missed,
    }


def compute_recovery_stats(stem, weak_model, strong_model):
    weak_stats = compute_image_stats_for_model(stem, weak_model)
    strong_stats = compute_image_stats_for_model(stem, strong_model)

    recovered_total = weak_stats["missed_total"] - strong_stats["missed_total"]
    recovered_tiny = weak_stats["tiny_missed"] - strong_stats["tiny_missed"]

    return {
        "stem": stem,
        "weak_model": weak_model,
        "strong_model": strong_model,
        "total_gt": weak_stats["total_gt"],
        "weak_missed_total": weak_stats["missed_total"],
        "strong_missed_total": strong_stats["missed_total"],
        "recovered_total": recovered_total,
        "tiny_gt": weak_stats["tiny_gt"],
        "weak_tiny_missed": weak_stats["tiny_missed"],
        "strong_tiny_missed": strong_stats["tiny_missed"],
        "recovered_tiny": recovered_tiny,
    }


# =========================================================
# MAIN
# =========================================================

def main():
    stems = sorted([p.stem for p in VAL_GT_DIR.glob("*.txt")])

    rows = []

    if RANK_MODE == "most_missed":
        for stem in stems:
            stats = compute_image_stats_for_model(stem, TARGET_MODEL)
            rows.append({
                "stem": stem,
                "model": TARGET_MODEL,
                **stats
            })

        if USE_TINY_PRIORITY:
            rows.sort(key=lambda x: (x["tiny_missed"], x["missed_total"], x["tiny_gt"], x["total_gt"]), reverse=True)
        else:
            rows.sort(key=lambda x: (x["missed_total"], x["total_gt"]), reverse=True)

    elif RANK_MODE == "most_recovered":
        for stem in stems:
            stats = compute_recovery_stats(stem, WEAK_MODEL, STRONG_MODEL)
            rows.append(stats)

        if USE_TINY_PRIORITY:
            rows.sort(key=lambda x: (x["recovered_tiny"], x["recovered_total"], x["tiny_gt"], x["total_gt"]), reverse=True)
        else:
            rows.sort(key=lambda x: (x["recovered_total"], x["total_gt"]), reverse=True)

    else:
        raise ValueError("RANK_MODE must be 'most_missed' or 'most_recovered'.")

    top_rows = rows[:TOP_K]

    # Save CSV
    if top_rows:
        fieldnames = list(top_rows[0].keys())
        with open(OUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(top_rows)

    # Save TXT of stems
    with open(OUT_TXT, "w") as f:
        for row in top_rows:
            f.write(row["stem"] + "\n")

    print("\n===== TOP HARD EXAMPLES =====")
    for i, row in enumerate(top_rows, 1):
        print(f"{i:02d}. {row}")

    print(f"\nSaved CSV : {OUT_CSV}")
    print(f"Saved TXT : {OUT_TXT}")


if __name__ == "__main__":
    main()