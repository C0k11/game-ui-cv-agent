"""Build YOLO dataset from user's manual labels in run_060337.

Copies originals + creates augmented versions (flip, bright, dark).
"""
import shutil
import cv2
import numpy as np
from pathlib import Path

SRC = Path(r"D:\Project\ai game secretary\data\raw_images\run_20260307_060337")
OUT = Path(r"D:\Project\ai game secretary\data\yolo_battle_dataset")

def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    img_dir = OUT / "images"
    lbl_dir = OUT / "labels"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    count = 0
    for txt in sorted(SRC.glob("frame_*.txt")):
        jpg = txt.with_suffix(".jpg")
        if not jpg.exists():
            continue
        # Copy original
        shutil.copy2(str(jpg), str(img_dir / jpg.name))
        shutil.copy2(str(txt), str(lbl_dir / txt.name))
        count += 1

        img = cv2.imread(str(jpg))
        lines = txt.read_text().strip().split("\n")
        base = txt.stem

        # Horizontal flip
        flipped = cv2.flip(img, 1)
        cv2.imwrite(str(img_dir / f"{base}_flip.jpg"), flipped, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(lbl_dir / f"{base}_flip.txt", "w") as f:
            for line in lines:
                p = line.strip().split()
                if len(p) < 5:
                    continue
                cx_flipped = 1.0 - float(p[1])
                f.write(f"{p[0]} {cx_flipped:.6f} {p[2]} {p[3]} {p[4]}\n")
        count += 1

        # Brightness up
        bright = cv2.convertScaleAbs(img, alpha=1.15, beta=15)
        cv2.imwrite(str(img_dir / f"{base}_bright.jpg"), bright, [cv2.IMWRITE_JPEG_QUALITY, 95])
        shutil.copy2(str(txt), str(lbl_dir / f"{base}_bright.txt"))
        count += 1

        # Brightness down
        dark = cv2.convertScaleAbs(img, alpha=0.85, beta=-15)
        cv2.imwrite(str(img_dir / f"{base}_dark.jpg"), dark, [cv2.IMWRITE_JPEG_QUALITY, 95])
        shutil.copy2(str(txt), str(lbl_dir / f"{base}_dark.txt"))
        count += 1

    # Write dataset.yaml
    yaml_path = OUT / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {OUT}\n")
        f.write("train: images\n")
        f.write("val: images\n")
        f.write("nc: 4\n")
        f.write("names:\n")
        f.write("  0: c0\n")
        f.write("  1: c1\n")
        f.write("  2: c2\n")
        f.write("  3: c3\n")

    print(f"Dataset: {count} images (originals + augmented)")
    print(f"YAML: {yaml_path}")


if __name__ == "__main__":
    main()
