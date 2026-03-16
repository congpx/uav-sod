#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image

RAW_ROOT = Path("/home/congpx/datasets/UAVDT/raw")
OUT_ROOT = Path("/home/congpx/datasets/UAVDT-YOLO")

COPY_FILES = False
VAL_RATIO = 0.2

CAT_COL = 7

CLASS_MAP = {
    1: 0,  # car
    2: 1,  # truck
    3: 2,  # bus
}

CLASS_NAMES = {
    0: "car",
    1: "truck",
    2: "bus",
}

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def find_dataset_root(raw_root: Path) -> Path:
    for p in raw_root.rglob("UAV-benchmark-M"):
        if p.is_dir():
            return p
    return raw_root


def find_gt_dir(dataset_root: Path) -> Path | None:
    # 1) try inside dataset root
    p = dataset_root / "GT"
    if p.exists() and p.is_dir():
        return p

    for cand in dataset_root.rglob("GT"):
        if cand.is_dir():
            return cand

    # 2) fallback: search in RAW_ROOT as toolkit may contain annotations
    for cand in RAW_ROOT.rglob("GT"):
        if cand.is_dir():
            txts = list(cand.glob("*gt_whole*.txt"))
            if txts:
                return cand

    # 3) fallback: use parent folder of any gt_whole file
    txts = list(RAW_ROOT.rglob("*gt_whole*.txt"))
    if txts:
        return txts[0].parent

    return None
    

def find_sequences(dataset_root: Path) -> List[Path]:
    seqs = []
    for p in dataset_root.rglob("M*"):
        if not p.is_dir():
            continue
        img_files = []
        for ext in IMG_EXTS:
            img_files.extend(list(p.rglob(f"*{ext}")))
        if img_files:
            seqs.append(p)
    # unique by path
    seqs = sorted(set(seqs), key=lambda x: x.name)
    return seqs


def find_gt_file(gt_dir: Path | None, seq_name: str) -> Path | None:
    if gt_dir is None:
        return None
    direct = gt_dir / f"{seq_name}_gt_whole.txt"
    if direct.exists():
        return direct
    for p in gt_dir.rglob(f"{seq_name}_gt_whole.txt"):
        return p
    return None


def read_image_map(seq_dir: Path) -> Dict[int, Path]:
    imgs = []
    for ext in IMG_EXTS:
        imgs.extend(seq_dir.rglob(f"*{ext}"))
    imgs = sorted(imgs)

    frame_map = {}
    for img in imgs:
        stem = img.stem
        digits = "".join(ch for ch in stem if ch.isdigit())
        if digits:
            frame_id = int(digits)
            frame_map[frame_id] = img
    return frame_map


def parse_gt_file(gt_file: Path) -> Dict[int, List[Tuple[int, float, float, float, float]]]:
    per_frame: Dict[int, List[Tuple[int, float, float, float, float]]] = {}
    with gt_file.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 8:
                continue
            try:
                frame_id = int(float(row[0]))
                x = float(row[2])
                y = float(row[3])
                w = float(row[4])
                h = float(row[5])
                raw_cat = int(float(row[CAT_COL]))
            except Exception:
                continue

            if w <= 0 or h <= 0:
                continue
            if raw_cat not in CLASS_MAP:
                continue

            yolo_cls = CLASS_MAP[raw_cat]
            per_frame.setdefault(frame_id, []).append((yolo_cls, x, y, w, h))
    return per_frame


def yolo_line_from_xywh(cls_id: int, x: float, y: float, w: float, h: float, W: int, H: int) -> str:
    xc = (x + w / 2.0) / W
    yc = (y + h / 2.0) / H
    wn = w / W
    hn = h / H

    xc = min(max(xc, 0.0), 1.0)
    yc = min(max(yc, 0.0), 1.0)
    wn = min(max(wn, 0.0), 1.0)
    hn = min(max(hn, 0.0), 1.0)

    return f"{cls_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"


def sequence_split(seq_names: List[str], val_ratio: float = 0.2):
    n = len(seq_names)
    n_val = max(1, int(round(n * val_ratio)))
    val = seq_names[-n_val:]
    train = seq_names[:-n_val]
    return train, val


def link_or_copy(src: Path, dst: Path, copy_files: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_files:
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def main():
    dataset_root = find_dataset_root(RAW_ROOT)
    gt_dir = find_gt_dir(dataset_root)
    seq_dirs = find_sequences(dataset_root)

    if not seq_dirs:
        raise RuntimeError(f"Khong tim thay sequence UAVDT trong: {dataset_root}")

    seq_names = [p.name for p in seq_dirs]
    train_seqs, val_seqs = sequence_split(seq_names, VAL_RATIO)

    log(f"[INFO] dataset_root = {dataset_root}")
    log(f"[INFO] gt_dir       = {gt_dir}")
    log(f"[INFO] #sequences   = {len(seq_dirs)}")
    log(f"[INFO] train seqs   = {len(train_seqs)}")
    log(f"[INFO] val seqs     = {len(val_seqs)}")

    if OUT_ROOT.exists():
        log(f"[INFO] Removing old output root: {OUT_ROOT}")
        shutil.rmtree(OUT_ROOT)

    for split in ["train", "val"]:
        ensure_dir(OUT_ROOT / "images" / split)
        ensure_dir(OUT_ROOT / "labels" / split)

    num_images = 0
    num_boxes = 0

    seq_dir_map = {p.name: p for p in seq_dirs}

    for split, split_seqs in [("train", train_seqs), ("val", val_seqs)]:
        log(f"\n[INFO] Processing split: {split}")
        for seq_name in split_seqs:
            seq_dir = seq_dir_map[seq_name]
            gt_file = find_gt_file(gt_dir, seq_name)

            if gt_file is None:
                log(f"[WARN] No gt_whole for {seq_name}, skip")
                continue

            frame_to_boxes = parse_gt_file(gt_file)
            frame_to_img = read_image_map(seq_dir)

            log(f"[INFO] {seq_name}: {len(frame_to_img)} images, {sum(len(v) for v in frame_to_boxes.values())} boxes")

            for frame_id, img_path in frame_to_img.items():
                out_stem = f"{seq_name}__{img_path.stem}"
                out_img = OUT_ROOT / "images" / split / f"{out_stem}{img_path.suffix.lower()}"
                out_lbl = OUT_ROOT / "labels" / split / f"{out_stem}.txt"

                link_or_copy(img_path, out_img, COPY_FILES)

                with Image.open(img_path) as im:
                    W, H = im.size

                lines = []
                for cls_id, x, y, w, h in frame_to_boxes.get(frame_id, []):
                    lines.append(yolo_line_from_xywh(cls_id, x, y, w, h, W, H))

                out_lbl.write_text("\n".join(lines))
                num_images += 1
                num_boxes += len(lines)

    yaml_path = OUT_ROOT / "uavdt.yaml"
    names_lines = "\n".join([f"  {k}: {v}" for k, v in sorted(CLASS_NAMES.items())])
    yaml_text = f"""path: {OUT_ROOT}
train: images/train
val: images/val

names:
{names_lines}
"""
    yaml_path.write_text(yaml_text)

    log("\n====================================================")
    log("[DONE] UAVDT -> YOLO conversion finished")
    log(f"[DONE] Output root : {OUT_ROOT}")
    log(f"[DONE] YAML        : {yaml_path}")
    log(f"[DONE] Images      : {num_images}")
    log(f"[DONE] Boxes       : {num_boxes}")
    log("====================================================")


if __name__ == "__main__":
    main()
