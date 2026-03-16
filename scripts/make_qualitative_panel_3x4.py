from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import math

# =========================================================
# CONFIG
# =========================================================

# Ảnh validation của VisDrone-YOLO
VAL_IMG_DIR = Path("/home/congpx/datasets/VisDrone-YOLO/images/val")
VAL_GT_DIR = Path("/home/congpx/datasets/VisDrone-YOLO/labels/val")

# Prediction txt đã dump bằng yolo predict --save_txt --save_conf
PRED_DIRS = {
    "Ground Truth": None,
    "P2-base-640": Path("/home/congpx/uav_yolo_visdrone/runs/visdrone_pred_dump/p2_base_640/labels"),
    "P2-base-800": Path("/home/congpx/uav_yolo_visdrone/runs/visdrone_pred_dump/p2_base_800/labels"),
    "P2+TAL-800": Path("/home/congpx/uav_yolo_visdrone/runs/visdrone_pred_dump/p2_tal_800/labels"),
}

# Chọn đúng 3 ảnh để làm panel 3x4
# Điền stem của file ảnh, không có đuôi .jpg/.png
SELECTED_IMAGES = [
    "0000010_04096_d_0000007",
    "0000026_00001_d_0000005",
    "0000081_04096_d_0000186",
]

# Tên class của VisDrone
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

# Màu box cho từng cột
COLUMN_COLORS = {
    "Ground Truth": (0, 255, 0),   # xanh lá
    "P2-base-640": (255, 0, 0),    # đỏ
    "P2-base-800": (255, 140, 0),  # cam
    "P2+TAL-800": (0, 102, 255),   # xanh dương
}

# Có hiển thị confidence không
SHOW_CONF = False

# Resize hiển thị mỗi ảnh con về cùng width
PANEL_CELL_WIDTH = 420

# Output
OUT_DIR = Path("/home/congpx/uav_yolo_visdrone/runs/visdrone_pred_dump/qualitative")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "qualitative_panel_3x4.png"


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
    x1 = (xc * W) - bw / 2
    y1 = (yc * H) - bh / 2
    x2 = (xc * W) + bw / 2
    y2 = (yc * H) + bh / 2
    return [x1, y1, x2, y2]


def load_yolo_txt(txt_path: Path, W: int, H: int):
    """
    Return list of dict:
    {
        "cls": int,
        "box": [x1, y1, x2, y2],
        "conf": float or None
    }
    """
    results = []

    if txt_path is None or not txt_path.exists():
        return results

    text = txt_path.read_text().strip()
    if not text:
        return results

    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        cls_id = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:5])
        conf = float(parts[5]) if len(parts) >= 6 else None

        box = xywhn_to_xyxy(xc, yc, w, h, W, H)
        results.append({
            "cls": cls_id,
            "box": box,
            "conf": conf
        })

    return results


def get_default_font(size=14):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def clamp(v, lo, hi):
    return max(lo, min(v, hi))


def draw_boxes_on_image(img: Image.Image, objects, color, title=None):
    img = img.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    W, H = img.size

    box_width = max(2, int(round(min(W, H) / 250)))
    font = get_default_font(size=max(12, int(round(min(W, H) / 45))))
    title_font = get_default_font(size=max(16, int(round(min(W, H) / 30))))

    # Nếu có title thì thêm dải trắng ở trên
    title_h = 34
    canvas = Image.new("RGB", (W, H + title_h), (255, 255, 255))
    canvas.paste(img, (0, title_h))
    draw = ImageDraw.Draw(canvas)

    if title is not None:
        draw.text((10, 6), title, fill=(0, 0, 0), font=title_font)

    for obj in objects:
        cls_id = obj["cls"]
        x1, y1, x2, y2 = obj["box"]
        conf = obj["conf"]

        y1 += title_h
        y2 += title_h

        x1 = clamp(int(round(x1)), 0, W - 1)
        y1 = clamp(int(round(y1)), 0, H + title_h - 1)
        x2 = clamp(int(round(x2)), 0, W - 1)
        y2 = clamp(int(round(y2)), 0, H + title_h - 1)

        if x2 <= x1 or y2 <= y1:
            continue

        # draw rectangle
        for k in range(box_width):
            draw.rectangle([x1 - k, y1 - k, x2 + k, y2 + k], outline=color)

        class_name = CLASS_NAMES.get(cls_id, str(cls_id))
        if SHOW_CONF and conf is not None:
            label = f"{class_name} {conf:.2f}"
        else:
            label = class_name

        # text box
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        tx1 = x1
        ty1 = max(title_h, y1 - th - 6)
        tx2 = min(W - 1, tx1 + tw + 8)
        ty2 = ty1 + th + 4

        draw.rectangle([tx1, ty1, tx2, ty2], fill=color)
        draw.text((tx1 + 4, ty1 + 1), label, fill=(255, 255, 255), font=font)

    return canvas


def resize_keep_ratio(img: Image.Image, target_w: int) -> Image.Image:
    W, H = img.size
    scale = target_w / W
    new_h = int(round(H * scale))
    return img.resize((target_w, new_h), Image.Resampling.LANCZOS)


# =========================================================
# MAIN
# =========================================================

def main():
    if len(SELECTED_IMAGES) != 3:
        raise ValueError("SELECTED_IMAGES phải chứa đúng 3 ảnh để tạo panel 3x4.")

    row_images = []

    for stem in SELECTED_IMAGES:
        img_path = find_image_by_stem(stem)
        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        row_panels = []

        for col_name, pred_dir in PRED_DIRS.items():
            if col_name == "Ground Truth":
                txt_path = VAL_GT_DIR / f"{stem}.txt"
            else:
                txt_path = pred_dir / f"{stem}.txt"

            objs = load_yolo_txt(txt_path, W, H)
            panel = draw_boxes_on_image(
                img,
                objs,
                color=COLUMN_COLORS[col_name],
                title=col_name
            )
            panel = resize_keep_ratio(panel, PANEL_CELL_WIDTH)
            row_panels.append(panel)

        row_images.append(row_panels)

    # Tính kích thước canvas cuối
    row_heights = []
    for row in row_images:
        max_h = max(im.size[1] for im in row)
        row_heights.append(max_h)

    total_w = 4 * PANEL_CELL_WIDTH
    total_h = sum(row_heights)

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))

    y_cursor = 0
    for r, row in enumerate(row_images):
        row_h = row_heights[r]
        x_cursor = 0
        for panel in row:
            pw, ph = panel.size
            y_offset = y_cursor + (row_h - ph) // 2
            canvas.paste(panel, (x_cursor, y_offset))
            x_cursor += PANEL_CELL_WIDTH
        y_cursor += row_h

    canvas.save(OUT_PATH)
    print(f"[DONE] Saved qualitative panel to: {OUT_PATH}")

    # show preview with matplotlib
    plt.figure(figsize=(16, 12))
    plt.imshow(canvas)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()