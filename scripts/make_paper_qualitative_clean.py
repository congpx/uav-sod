#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import csv

# =========================================================
# CONFIG
# =========================================================
VAL_IMG_DIR = Path("/home/congpx/datasets/VisDrone-YOLO/images/val")
VAL_GT_DIR = Path("/home/congpx/datasets/VisDrone-YOLO/labels/val")

PRED_DIRS = {
    "Ground Truth": None,
    "P2-base-640": Path("/home/congpx/uav_yolo_visdrone/runs/paper_pred_dump/p2_base_640/labels"),
    "P2-base-800": Path("/home/congpx/uav_yolo_visdrone/runs/paper_pred_dump/p2_base_800/labels"),
    "P2+TAL-800": Path("/home/congpx/uav_yolo_visdrone/runs/paper_pred_dump/p2_tal_800/labels"),
}

WEAK_MODEL = "P2-base-640"
STRONG_MODEL = "P2-base-800"
TAL_MODEL = "P2+TAL-800"

OUT_DIR = Path("/home/congpx/uav_yolo_visdrone/paper_figures/qualitative_clean")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "qualitative_panel_3x4_clean.png"
OUT_PDF = OUT_DIR / "qualitative_panel_3x4_clean.pdf"
OUT_TXT = OUT_DIR / "selected_stems.txt"
OUT_CSV = OUT_DIR / "selected_stems_scores.csv"

TOP_K_CANDIDATES = 30
FINAL_N = 3
IOU_THR = 0.5
TINY_AREA_RATIO = 0.001
PANEL_CELL_WIDTH = 460
BOX_WIDTH = 3
MAX_HIGHLIGHT_REGIONS = 3

COLORS = {
    "Ground Truth": (0, 200, 0),    # green
    "P2-base-640": (220, 20, 60),   # red
    "P2-base-800": (255, 140, 0),   # orange
    "P2+TAL-800": (0, 102, 255),    # blue
}

HIGHLIGHT_COLOR = (255, 215, 0)     # yellow


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
    x1 = (xc * W) - bw / 2.0
    y1 = (yc * H) - bh / 2.0
    x2 = (xc * W) + bw / 2.0
    y2 = (yc * H) + bh / 2.0
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


def clamp(v, lo, hi):
    return max(lo, min(v, hi))


def load_yolo_txt(txt_path: Path | None, W: int, H: int):
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
        objs.append({"cls": cls_id, "box": box, "conf": conf})

    if any(obj["conf"] is not None for obj in objs):
        objs.sort(key=lambda x: x["conf"] if x["conf"] is not None else 1.0, reverse=True)
    return objs


def get_default_font(size=16):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            try:
                return ImageFont.truetype(c, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_single_box(draw, box, color, width=3):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    for k in range(width):
        draw.rectangle([x1 - k, y1 - k, x2 + k, y2 + k], outline=color)


def draw_dashed_box(draw, box, color, width=3, dash=16, gap=10):
    x1, y1, x2, y2 = [int(round(v)) for v in box]

    x = x1
    while x < x2:
        draw.line([(x, y1), (min(x + dash, x2), y1)], fill=color, width=width)
        x += dash + gap

    x = x1
    while x < x2:
        draw.line([(x, y2), (min(x + dash, x2), y2)], fill=color, width=width)
        x += dash + gap

    y = y1
    while y < y2:
        draw.line([(x1, y), (x1, min(y + dash, y2))], fill=color, width=width)
        y += dash + gap

    y = y1
    while y < y2:
        draw.line([(x2, y), (x2, min(y + dash, y2))], fill=color, width=width)
        y += dash + gap


def resize_keep_ratio(img: Image.Image, target_w: int) -> Image.Image:
    W, H = img.size
    scale = target_w / W
    new_h = int(round(H * scale))
    return img.resize((target_w, new_h), Image.Resampling.LANCZOS)


def match_gt_with_preds(gts, preds, iou_thr=0.5):
    matched_gt = [False] * len(gts)
    matched_pred = [False] * len(preds)

    for pi, pred in enumerate(preds):
        best_iou = 0.0
        best_idx = -1
        for gi, gt in enumerate(gts):
            if matched_gt[gi]:
                continue
            if pred["cls"] != gt["cls"]:
                continue
            cur = iou(pred["box"], gt["box"])
            if cur >= iou_thr and cur > best_iou:
                best_iou = cur
                best_idx = gi
        if best_idx >= 0:
            matched_gt[best_idx] = True
            matched_pred[pi] = True

    return matched_gt, matched_pred


def compute_recovery_row(stem: str):
    img_path = find_image_by_stem(stem)
    with Image.open(img_path) as im:
        W, H = im.size

    gts = load_yolo_txt(VAL_GT_DIR / f"{stem}.txt", W, H)
    weak_preds = load_yolo_txt(PRED_DIRS[WEAK_MODEL] / f"{stem}.txt", W, H)
    strong_preds = load_yolo_txt(PRED_DIRS[STRONG_MODEL] / f"{stem}.txt", W, H)

    weak_match, _ = match_gt_with_preds(gts, weak_preds, IOU_THR)
    strong_match, _ = match_gt_with_preds(gts, strong_preds, IOU_THR)

    img_area = float(W * H)

    recovered_indices = []
    recovered_tiny = 0
    tiny_gt = 0
    weak_missed_total = 0
    strong_missed_total = 0
    weak_tiny_missed = 0
    strong_tiny_missed = 0

    for i, gt in enumerate(gts):
        ratio = box_area(gt["box"]) / img_area
        is_tiny = ratio < TINY_AREA_RATIO
        if is_tiny:
            tiny_gt += 1

        if not weak_match[i]:
            weak_missed_total += 1
            if is_tiny:
                weak_tiny_missed += 1

        if not strong_match[i]:
            strong_missed_total += 1
            if is_tiny:
                strong_tiny_missed += 1

        if (not weak_match[i]) and strong_match[i]:
            recovered_indices.append(i)
            if is_tiny:
                recovered_tiny += 1

    return {
        "stem": stem,
        "total_gt": len(gts),
        "tiny_gt": tiny_gt,
        "weak_missed_total": weak_missed_total,
        "strong_missed_total": strong_missed_total,
        "recovered_total": len(recovered_indices),
        "weak_tiny_missed": weak_tiny_missed,
        "strong_tiny_missed": strong_tiny_missed,
        "recovered_tiny": recovered_tiny,
        "recovered_indices": recovered_indices,
    }


def choose_diverse_top_examples():
    stems = sorted([p.stem for p in VAL_GT_DIR.glob("*.txt")])
    rows = [compute_recovery_row(stem) for stem in stems]
    rows.sort(key=lambda x: (x["recovered_tiny"], x["recovered_total"], x["tiny_gt"], x["total_gt"]), reverse=True)

    top_rows = rows[:TOP_K_CANDIDATES]
    selected = []
    used_prefix = set()

    for row in top_rows:
        prefix = row["stem"].split("_")[0]
        if prefix in used_prefix:
            continue
        selected.append(row)
        used_prefix.add(prefix)
        if len(selected) == FINAL_N:
            break

    if len(selected) < FINAL_N:
        for row in top_rows:
            if row not in selected:
                selected.append(row)
                if len(selected) == FINAL_N:
                    break

    return selected


def make_candidate_region(center_x, center_y, W, H):
    win_w = max(180, int(0.18 * W))
    win_h = max(140, int(0.18 * H))

    x1 = clamp(center_x - win_w / 2, 0, W - win_w)
    y1 = clamp(center_y - win_h / 2, 0, H - win_h)
    x2 = x1 + win_w
    y2 = y1 + win_h
    return [x1, y1, x2, y2]


def choose_recovered_regions(gts, recovered_indices, W, H, max_regions=3):
    if not recovered_indices:
        return []

    recovered_boxes = [gts[i]["box"] for i in recovered_indices]

    # candidate windows centered at recovered GTs
    candidates = []
    for box in recovered_boxes:
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        region = make_candidate_region(cx, cy, W, H)

        score = 0
        for rb in recovered_boxes:
            rcx = (rb[0] + rb[2]) / 2.0
            rcy = (rb[1] + rb[3]) / 2.0
            if region[0] <= rcx <= region[2] and region[1] <= rcy <= region[3]:
                score += 1

        candidates.append((score, region))

    candidates.sort(key=lambda x: (x[0], -box_area(x[1])), reverse=True)

    selected = []
    for score, region in candidates:
        ok = True
        for prev in selected:
            if iou(region, prev) > 0.35:
                ok = False
                break
        if ok:
            selected.append(region)
        if len(selected) == max_regions:
            break

    return selected


def draw_panel_for_column(stem: str, column_name: str, highlight_regions):
    img_path = find_image_by_stem(stem)
    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    if column_name == "Ground Truth":
        txt_path = VAL_GT_DIR / f"{stem}.txt"
    else:
        txt_path = PRED_DIRS[column_name] / f"{stem}.txt"

    objs = load_yolo_txt(txt_path, W, H)

    title_h = 38
    canvas = Image.new("RGB", (W, H + title_h), (255, 255, 255))
    canvas.paste(img, (0, title_h))
    draw = ImageDraw.Draw(canvas)

    title_font = get_default_font(size=18)
    draw.text((10, 7), column_name, fill=(0, 0, 0), font=title_font)

    color = COLORS[column_name]

    # draw only boxes, no class labels/confidence
    for obj in objs:
        x1, y1, x2, y2 = obj["box"]
        box = [x1, y1 + title_h, x2, y2 + title_h]
        draw_single_box(draw, box, color, width=BOX_WIDTH)

    # draw highlight regions on all columns
    for region in highlight_regions:
        x1, y1, x2, y2 = region
        draw_dashed_box(draw, [x1, y1 + title_h, x2, y2 + title_h], HIGHLIGHT_COLOR, width=4)

    return resize_keep_ratio(canvas, PANEL_CELL_WIDTH)


def save_selected_metadata(selected_rows):
    with open(OUT_TXT, "w") as f:
        for row in selected_rows:
            f.write(row["stem"] + "\n")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stem", "total_gt", "tiny_gt",
                "weak_missed_total", "strong_missed_total", "recovered_total",
                "weak_tiny_missed", "strong_tiny_missed", "recovered_tiny"
            ]
        )
        writer.writeheader()
        for row in selected_rows:
            writer.writerow({
                "stem": row["stem"],
                "total_gt": row["total_gt"],
                "tiny_gt": row["tiny_gt"],
                "weak_missed_total": row["weak_missed_total"],
                "strong_missed_total": row["strong_missed_total"],
                "recovered_total": row["recovered_total"],
                "weak_tiny_missed": row["weak_tiny_missed"],
                "strong_tiny_missed": row["strong_tiny_missed"],
                "recovered_tiny": row["recovered_tiny"],
            })


def make_figure():
    selected_rows = choose_diverse_top_examples()
    save_selected_metadata(selected_rows)

    columns = ["Ground Truth", "P2-base-640", "P2-base-800", "P2+TAL-800"]
    row_panels = []

    for row in selected_rows:
        stem = row["stem"]
        img_path = find_image_by_stem(stem)

        with Image.open(img_path) as im:
            W, H = im.size

        gts = load_yolo_txt(VAL_GT_DIR / f"{stem}.txt", W, H)
        highlight_regions = choose_recovered_regions(
            gts=gts,
            recovered_indices=row["recovered_indices"],
            W=W,
            H=H,
            max_regions=MAX_HIGHLIGHT_REGIONS
        )

        panels = [draw_panel_for_column(stem, c, highlight_regions) for c in columns]
        row_panels.append((row, panels))

    row_heights = []
    for _, panels in row_panels:
        row_heights.append(max(im.size[1] for im in panels))

    total_w = len(columns) * PANEL_CELL_WIDTH
    total_h = sum(row_heights)
    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))

    y_cursor = 0
    for (_, panels), row_h in zip(row_panels, row_heights):
        x_cursor = 0
        for panel in panels:
            pw, ph = panel.size
            y_offset = y_cursor + (row_h - ph) // 2
            canvas.paste(panel, (x_cursor, y_offset))
            x_cursor += PANEL_CELL_WIDTH
        y_cursor += row_h

    canvas.save(OUT_PNG)

    plt.figure(figsize=(18, 13))
    plt.imshow(canvas)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
    plt.close()

    print("[DONE] selected stems:")
    for row in selected_rows:
        print(
            row["stem"],
            "| recovered_total =", row["recovered_total"],
            "| recovered_tiny =", row["recovered_tiny"]
        )
    print(f"[DONE] PNG: {OUT_PNG}")
    print(f"[DONE] PDF: {OUT_PDF}")
    print(f"[DONE] TXT: {OUT_TXT}")
    print(f"[DONE] CSV: {OUT_CSV}")


if __name__ == "__main__":
    make_figure()