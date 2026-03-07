"""Manual bounding box labels for battle character heads in run_060337.

Generates YOLO format labels for training character head detection.
Class 0 = player_head (our 4 strikers)

Labels are approximate normalized (cx, cy, w, h) from visual inspection.
"""
import shutil, os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC_DIR = REPO / "data" / "raw_images" / "run_20260307_060337"
OUT_DIR = REPO / "data" / "yolo_battle_dataset"
IMG_DIR = OUT_DIR / "images"
LBL_DIR = OUT_DIR / "labels"

# Manual labels: frame -> list of (cx, cy, w, h) normalized
# 4 player characters: 伊織(silver), 寧瑠(orange), 旬(black), 潔莉諾(white hat)
# Positions estimated from screenshots at 3840x2160
LABELS = {
    "frame_000019.jpg": [
        # Far left - 伊織 (silver hair, large weapon)
        (0.075, 0.47, 0.055, 0.07),
        # Center-left - 寧瑠 (orange hair, in fire)
        (0.265, 0.43, 0.050, 0.06),
        # Center - 旬 (dark hair/black)
        (0.400, 0.44, 0.045, 0.06),
        # Center-right - 潔莉諾 (white hat/cape)
        (0.520, 0.42, 0.050, 0.07),
    ],
    "frame_000020.jpg": [
        # 伊織 - far left, moving
        (0.095, 0.50, 0.055, 0.07),
        # 寧瑠 - center with fire
        (0.310, 0.44, 0.050, 0.06),
        # 旬 - center
        (0.245, 0.47, 0.045, 0.06),
        # 潔莉諾 - right side
        (0.555, 0.43, 0.050, 0.07),
    ],
    "frame_000022.jpg": [
        (0.090, 0.50, 0.055, 0.07),
        (0.290, 0.44, 0.050, 0.06),
        (0.230, 0.47, 0.045, 0.06),
        (0.540, 0.43, 0.050, 0.07),
    ],
    "frame_000023.jpg": [
        (0.085, 0.50, 0.055, 0.07),
        (0.280, 0.44, 0.050, 0.06),
        (0.220, 0.47, 0.045, 0.06),
        (0.530, 0.43, 0.050, 0.07),
    ],
    "frame_000024.jpg": [
        # Chars clustered left - skill being used
        (0.070, 0.62, 0.055, 0.07),
        (0.250, 0.44, 0.050, 0.06),
        (0.200, 0.47, 0.045, 0.06),
        (0.430, 0.44, 0.050, 0.07),
    ],
    "frame_000025.jpg": [
        # Chars clustered left
        (0.075, 0.65, 0.055, 0.07),
        (0.210, 0.42, 0.050, 0.06),
        (0.170, 0.48, 0.055, 0.07),
        (0.380, 0.42, 0.050, 0.07),
    ],
    "frame_000026.jpg": [
        (0.080, 0.63, 0.055, 0.07),
        (0.220, 0.43, 0.050, 0.06),
        (0.180, 0.47, 0.050, 0.06),
        (0.390, 0.42, 0.050, 0.07),
    ],
    "frame_000027.jpg": [
        (0.080, 0.62, 0.055, 0.07),
        (0.215, 0.43, 0.050, 0.06),
        (0.175, 0.48, 0.050, 0.06),
        (0.385, 0.43, 0.050, 0.07),
    ],
    "frame_000029.jpg": [
        (0.080, 0.60, 0.055, 0.07),
        (0.210, 0.43, 0.050, 0.06),
        (0.170, 0.47, 0.050, 0.06),
        (0.370, 0.42, 0.050, 0.07),
    ],
    "frame_000031.jpg": [
        (0.080, 0.58, 0.055, 0.07),
        (0.200, 0.42, 0.050, 0.06),
        (0.160, 0.46, 0.050, 0.06),
        (0.350, 0.42, 0.050, 0.07),
    ],
    "frame_000033.jpg": [
        # Clustered on left, enemies right
        (0.075, 0.58, 0.055, 0.07),
        (0.200, 0.42, 0.050, 0.06),
        (0.155, 0.45, 0.050, 0.06),
        (0.300, 0.40, 0.050, 0.07),
    ],
    "frame_000034.jpg": [
        (0.080, 0.57, 0.055, 0.07),
        (0.195, 0.42, 0.050, 0.06),
        (0.155, 0.45, 0.050, 0.06),
        (0.300, 0.41, 0.050, 0.07),
    ],
    "frame_000035.jpg": [
        (0.080, 0.56, 0.055, 0.07),
        (0.190, 0.42, 0.050, 0.06),
        (0.150, 0.45, 0.050, 0.06),
        (0.290, 0.41, 0.050, 0.07),
    ],
    "frame_000036.jpg": [
        # Tight cluster, moving forward
        (0.090, 0.55, 0.055, 0.07),
        (0.155, 0.41, 0.050, 0.06),
        (0.125, 0.44, 0.050, 0.06),
        (0.175, 0.39, 0.050, 0.07),
    ],
}

# Also add negative frames (non-battle) from trajectories for robustness
NEGATIVE_DIRS = []


def main():
    # Clean and create output
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    IMG_DIR.mkdir(parents=True)
    LBL_DIR.mkdir(parents=True)

    count = 0
    for fname, boxes in LABELS.items():
        src = SRC_DIR / fname
        if not src.exists():
            print(f"SKIP {fname} (not found)")
            continue
        dst_name = f"battle_{count:04d}.jpg"
        shutil.copy2(str(src), str(IMG_DIR / dst_name))
        lbl_path = LBL_DIR / f"battle_{count:04d}.txt"
        with open(lbl_path, "w") as f:
            for cx, cy, bw, bh in boxes:
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        count += 1
        print(f"Labeled {fname} -> {dst_name} ({len(boxes)} heads)")

    # Add augmented copies (flipped, brightness shifted) for more training data
    import cv2
    import numpy as np
    aug_count = count
    for fname, boxes in LABELS.items():
        src = SRC_DIR / fname
        if not src.exists():
            continue
        img = cv2.imread(str(src))
        if img is None:
            continue

        # Horizontal flip
        flipped = cv2.flip(img, 1)
        dst_name = f"battle_{aug_count:04d}.jpg"
        cv2.imwrite(str(IMG_DIR / dst_name), flipped, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lbl_path = LBL_DIR / f"battle_{aug_count:04d}.txt"
        with open(lbl_path, "w") as f:
            for cx, cy, bw, bh in boxes:
                f.write(f"0 {1.0 - cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        aug_count += 1

        # Brightness +20
        bright = cv2.convertScaleAbs(img, alpha=1.0, beta=20)
        dst_name = f"battle_{aug_count:04d}.jpg"
        cv2.imwrite(str(IMG_DIR / dst_name), bright, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lbl_path = LBL_DIR / f"battle_{aug_count:04d}.txt"
        with open(lbl_path, "w") as f:
            for cx, cy, bw, bh in boxes:
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        aug_count += 1

        # Brightness -20
        dark = cv2.convertScaleAbs(img, alpha=1.0, beta=-20)
        dst_name = f"battle_{aug_count:04d}.jpg"
        cv2.imwrite(str(IMG_DIR / dst_name), dark, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lbl_path = LBL_DIR / f"battle_{aug_count:04d}.txt"
        with open(lbl_path, "w") as f:
            for cx, cy, bw, bh in boxes:
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        aug_count += 1

    # Write dataset.yaml
    yaml_path = OUT_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {OUT_DIR}\n")
        f.write("train: images\n")
        f.write("val: images\n")
        f.write("nc: 1\n")
        f.write("names: ['player_head']\n")

    print(f"\nDone: {count} originals + {aug_count - count} augmented = {aug_count} total")
    print(f"Dataset: {yaml_path}")


if __name__ == "__main__":
    main()
