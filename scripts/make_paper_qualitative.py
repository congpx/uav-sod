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

OUT_DIR = Path("/home/congpx/uav_yolo_visdrone/paper_figures/qualitative")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PNG = OUT_DIR / "qualitative_panel_3x4.png"
OUT_PDF = OUT_DIR / "qualitative_panel_3x4.pdf"
OUT_STEMS_TXT = OUT_DIR / "selected_stems.txt"
OUT_CSV = OUT_DIR / "selected_stems_scores.csv"

TOP_K_CANDIDATES = 30
FINAL_N = 3
IOU_THR = 0.5
TINY_AREA_RATIO = 0.001  # 0.1%
PANEL_CELL_WIDTH = 440
SHOW_CONF = False

CLASS_NAMES = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor",
}

COLUMN_COLORS = {
    "Ground Truth": (0, 180, 0),
    "P2-base-640": (220, 20, 60),
    "P2-base-800": (255, 140, 0),
    "P2+TAL-800": (0, 102, 255),
}

RECOVERED_HIGHLIGHT = (255, 215, 0)   # vàng
TITLE_FONT_SIZE = 18
LABEL_FONT_SIZE = 13
SMALL_FONT_SIZE = 9


# =========================================================
# HELPERS
# =========================================================
def find_image_by_stem(stem: str) -> Path:
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        p = VAL_IMG_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Không tìm thấy ảnh cho stem: {stem}")


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


def get_default_font(size=14):
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


def draw_single_box(draw, box, color, width=2):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    for k in range(width):
        draw.rectangle([x1 - k, y1 - k, x2 + k, y2 + k], outline=color)


def draw_dashed_box(draw, box, color, width=2, dash=8, gap=6):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    # top
    x = x1
    while x < x2:
        draw.line([(x, y1), (min(x + dash, x2), y1)], fill=color, width=width)
        x += dash + gap
    # bottom
    x = x1
    while x < x2:
        draw.line([(x, y2), (min(x + dash, x2), y2)], fill=color, width=width)
        x += dash + gap
    # left
    y = y1
    while y < y2:
        draw.line([(x1, y), (x1, min(y + dash, y2))], fill=color, width=width)
        y += dash + gap
    # right
    y = y1
    while y < y2:
        draw.line([(x2, y), (x2, min(y + dash, y2))], fill=color, width=width)
        y += dash + gap


def draw_label(draw, x, y, label, color, font):
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx1 = x
    ty1 = max(0, y - th - 6)
    tx2 = tx1 + tw + 8
    ty2 = ty1 + th + 4
    draw.rectangle([tx1, ty1, tx2, ty2], fill=color)
    draw.text((tx1 + 4, ty1 + 1), label, fill=(255, 255, 255), font=font)


def resize_keep_ratio(img: Image.Image, target_w: int) -> Image.Image:
    W, H = img.size
    scale = target_w / W
    new_h = int(round(H * scale))
    return img.resize((target_w, new_h), Image.Resampling.LANCZOS)


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
            if row in selected:
                continue
            selected.append(row)
            if len(selected) == FINAL_N:
                break

    return selected


def draw_panel_for_column(stem: str, column_name: str, recovered_indices):
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

    title_font = get_default_font(TITLE_FONT_SIZE)
    label_font = get_default_font(LABEL_FONT_SIZE)
    small_font = get_default_font(SMALL_FONT_SIZE)

    draw.text((10, 7), column_name, fill=(0, 0, 0), font=title_font)

    color = COLUMN_COLORS[column_name]

    # draw boxes
    for obj in objs:
        x1, y1, x2, y2 = obj["box"]
        box = [x1, y1 + title_h, x2, y2 + title_h]
        draw_single_box(draw, box, color, width=2)

        label = CLASS_NAMES.get(obj["cls"], str(obj["cls"]))
        if SHOW_CONF and obj["conf"] is not None:
            label = f"{label} {obj['conf']:.2f}"
        draw_label(draw, int(round(x1)), int(round(y1 + title_h)), label, color, label_font)

    # highlight recovered GT boxes on GT / strong / TAL columns
    if column_name in ["Ground Truth", STRONG_MODEL, TAL_MODEL]:
        gts = load_yolo_txt(VAL_GT_DIR / f"{stem}.txt", W, H)
        for idx in recovered_indices:
            if idx >= len(gts):
                continue
            x1, y1, x2, y2 = gts[idx]["box"]
            box = [x1, y1 + title_h, x2, y2 + title_h]
            draw_dashed_box(draw, box, RECOVERED_HIGHLIGHT, width=3, dash=10, gap=6)

        if recovered_indices:
            tag = f"Recovered GTs: {len(recovered_indices)}"
            bbox = draw.textbbox((0, 0), tag, font=small_font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            x0 = 10
            y0 = title_h + 8
            draw.rectangle([x0, y0, x0 + tw + 10, y0 + th + 6], fill=(255, 255, 255))
            draw.text((x0 + 5, y0 + 3), tag, fill=RECOVERED_HIGHLIGHT, font=small_font)

    panel = resize_keep_ratio(canvas, PANEL_CELL_WIDTH)
    return panel


def save_selected_metadata(selected_rows):
    with open(OUT_STEMS_TXT, "w") as f:
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
        rec_idx = row["recovered_indices"]
        panels = [draw_panel_for_column(stem, c, rec_idx) for c in columns]
        row_panels.append((row, panels))

    row_heights = []
    for _, panels in row_panels:
        row_heights.append(max(im.size[1] for im in panels))

    total_w = len(columns) * PANEL_CELL_WIDTH
    total_h = sum(row_heights)
    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))

    y_cursor = 0
    for (row, panels), row_h in zip(row_panels, row_heights):
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
        print(row["stem"], "| recovered_total =", row["recovered_total"], "| recovered_tiny =", row["recovered_tiny"])
    print(f"[DONE] PNG: {OUT_PNG}")
    print(f"[DONE] PDF: {OUT_PDF}")
    print(f"[DONE] TXT: {OUT_STEMS_TXT}")
    print(f"[DONE] CSV: {OUT_CSV}")


if __name__ == "__main__":
    make_figure()