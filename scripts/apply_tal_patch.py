#!/usr/bin/env python3
from __future__ import annotations
import argparse
import shutil
from pathlib import Path


def build_block(mode: str, gamma: float, aa_alpha: float, aa_tau: float, aa_max: float) -> str:
    if mode == "original":
        return "        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)\n"
    area = (
        "        # area-aware weighting for small objects\n"
        "        w = (gt_bboxes[..., 2] - gt_bboxes[..., 0]).clamp(min=0)\n"
        "        h = (gt_bboxes[..., 3] - gt_bboxes[..., 1]).clamp(min=0)\n"
        "        area = (w * h).clamp(min=1e-6)\n\n"
        f"        aa_alpha = {aa_alpha}\n"
        f"        aa_tau = {aa_tau}\n"
        f"        aa_max = {aa_max}\n\n"
        "        area_weight = (1.0 + aa_alpha * torch.exp(-area / aa_tau)).clamp(max=aa_max).unsqueeze(-1)\n\n"
    )
    if mode == "safe":
        return area + f"        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta) * area_weight.pow({gamma})\n"
    if mode == "iouterm":
        return area + "        align_metric = bbox_scores.pow(self.alpha) * (overlaps * area_weight).pow(self.beta)\n"
    raise ValueError(f"unsupported mode: {mode}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--mode", choices=["original", "safe", "iouterm"], required=True)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--aa-alpha", type=float, default=0.05)
    ap.add_argument("--aa-tau", type=float, default=512.0)
    ap.add_argument("--aa-max", type=float, default=1.3)
    args = ap.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()
    tal_path = repo_root / "ultralytics" / "utils" / "tal.py"
    backup = repo_root / "ultralytics" / "utils" / "tal_original_backup.py"

    if not tal_path.exists():
        raise SystemExit(f"tal.py not found: {tal_path}")

    if not backup.exists():
        shutil.copy2(tal_path, backup)

    shutil.copy2(backup, tal_path)  # restore original first
    text = tal_path.read_text(encoding="utf-8")

    target = "        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)\n"
    if target not in text:
        raise SystemExit("Original align_metric line not found; restore from clean v8.4.11 repo first.")

    replacement = build_block(args.mode, args.gamma, args.aa_alpha, args.aa_tau, args.aa_max)
    text = text.replace(target, replacement, 1)
    tal_path.write_text(text, encoding="utf-8")
    print(f"Patched {tal_path} with mode={args.mode}")


if __name__ == "__main__":
    main()
