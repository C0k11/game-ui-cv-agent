"""Augment 10 Emoticon_Action images → 50 and train YOLO for headpat bubble detection.

Reads raw labeled frames from data/raw_images/run_20260306_150801,
strips polygon info to standard YOLO bbox format, applies augmentations,
and trains a lightweight YOLO model.

Usage:
    python scripts/train_emoticon_yolo.py              # augment + train
    python scripts/train_emoticon_yolo.py --augment     # augment only (no train)
    python scripts/train_emoticon_yolo.py --train       # train only (dataset must exist)
"""
import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw_images" / "run_20260306_150801"
ML_CACHE = Path(r"D:\Project\ml_cache\models\yolo")
DATASET_DIR = ML_CACHE / "dataset" / "emoticon"

CLASS_NAME = "Emoticon_Action"
TARGET_TOTAL = 50
VAL_RATIO = 0.2
SEED = 42


def load_raw_pairs() -> List[Tuple[Path, Path]]:
    """Load image+label pairs from the raw directory."""
    pairs = []
    for lbl in sorted(RAW_DIR.glob("*.txt")):
        if lbl.name == "classes.txt" or lbl.stat().st_size == 0:
            continue
        img = RAW_DIR / (lbl.stem + ".jpg")
        if img.exists():
            pairs.append((img, lbl))
    return pairs


def strip_polygon_labels(label_path: Path) -> str:
    """Convert polygon labels to standard YOLO bbox format (cls xc yc w h)."""
    lines = label_path.read_text("utf-8").strip().splitlines()
    out = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        # Format: cls xc yc w h [angle] [polygon ...]
        cls, xc, yc, w, h = parts[0], parts[1], parts[2], parts[3], parts[4]
        out.append(f"0 {xc} {yc} {w} {h}")
    return "\n".join(out)


def augment_image(img: np.ndarray, seed: int) -> np.ndarray:
    """Apply random augmentations to an image."""
    rng = np.random.RandomState(seed)
    h, w = img.shape[:2]
    result = img.copy()

    # Brightness jitter
    if rng.random() < 0.7:
        delta = rng.randint(-40, 40)
        result = np.clip(result.astype(np.int16) + delta, 0, 255).astype(np.uint8)

    # Contrast jitter
    if rng.random() < 0.6:
        factor = rng.uniform(0.7, 1.3)
        mean = result.mean()
        result = np.clip((result.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)

    # Hue/Saturation jitter (in HSV)
    if rng.random() < 0.5:
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + rng.randint(-8, 8)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + rng.randint(-20, 20), 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Gaussian noise
    if rng.random() < 0.4:
        noise = rng.normal(0, rng.uniform(3, 12), result.shape).astype(np.int16)
        result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Gaussian blur
    if rng.random() < 0.3:
        k = rng.choice([3, 5])
        result = cv2.GaussianBlur(result, (k, k), 0)

    # JPEG compression artifact simulation
    if rng.random() < 0.3:
        quality = rng.randint(50, 85)
        _, enc = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, quality])
        result = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return result


def generate_dataset(pairs: List[Tuple[Path, Path]]):
    """Augment raw pairs to TARGET_TOTAL images and set up YOLO dataset."""
    random.seed(SEED)
    np.random.seed(SEED)

    # Clean old dataset
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    all_images: List[Tuple[np.ndarray, str, str]] = []  # (img, label_text, name)

    # Original images
    for img_path, lbl_path in pairs:
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        label_text = strip_polygon_labels(lbl_path)
        all_images.append((img, label_text, img_path.stem))

    n_orig = len(all_images)
    print(f"  Original images: {n_orig}")

    if n_orig == 0:
        print("ERROR: No valid images found")
        return

    # Augmented copies
    aug_needed = TARGET_TOTAL - n_orig
    aug_idx = 0
    while aug_idx < aug_needed:
        # Cycle through originals
        src_idx = aug_idx % n_orig
        orig_img, orig_label, orig_name = all_images[src_idx]

        aug_img = augment_image(orig_img, seed=SEED + 1000 + aug_idx)
        aug_name = f"{orig_name}_aug{aug_idx:03d}"
        all_images.append((aug_img, orig_label, aug_name))
        aug_idx += 1

    print(f"  Total images: {len(all_images)} ({n_orig} orig + {aug_needed} augmented)")

    # Shuffle and split
    indices = list(range(len(all_images)))
    random.shuffle(indices)
    n_val = max(1, int(len(indices) * VAL_RATIO))
    val_indices = set(indices[:n_val])

    # Write dataset
    for split in ["train", "val"]:
        img_dir = DATASET_DIR / "images" / split
        lbl_dir = DATASET_DIR / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

    train_n = val_n = 0
    for i, (img, label_text, name) in enumerate(all_images):
        split = "val" if i in val_indices else "train"
        img_path = DATASET_DIR / "images" / split / f"{name}.jpg"
        lbl_path = DATASET_DIR / "labels" / split / f"{name}.txt"

        cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(str(img_path))
        lbl_path.write_text(label_text, encoding="utf-8")

        if split == "val":
            val_n += 1
        else:
            train_n += 1

    print(f"  Split: train={train_n}, val={val_n}")

    # Write data.yaml
    import yaml
    data = {
        "path": str(DATASET_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": {0: CLASS_NAME},
    }
    yaml_path = DATASET_DIR / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
    print(f"  data.yaml → {yaml_path}")


def train_model(epochs: int = 80, imgsz: int = 640, batch: int = 16, device: str = "0"):
    """Train YOLO on the Emoticon_Action dataset."""
    yaml_path = DATASET_DIR / "data.yaml"
    if not yaml_path.exists():
        print("ERROR: data.yaml not found. Run augmentation first.")
        return

    # Use yolov8n as base (lightweight, fast)
    pretrained = "yolov8n.pt"
    print(f"\n=== Training Emoticon_Action YOLO ===")
    print(f"  Pretrained: {pretrained}")
    print(f"  Epochs: {epochs}, ImgSz: {imgsz}, Batch: {batch}, Device: {device}")

    output_dir = ML_CACHE / "runs" / "emoticon"
    production_pt = ML_CACHE / "emoticon.pt"

    from ultralytics import YOLO
    model = YOLO(pretrained)
    model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        batch=batch,
        workers=4,
        project=str(output_dir),
        name="train",
        exist_ok=True,
        patience=25,
        save=True,
        plots=True,
        cache=True,
        amp=True,
        # Augmentation tuned for small UI markers
        degrees=5.0,
        translate=0.08,
        scale=0.25,
        mosaic=0.3,
        mixup=0.0,
        flipud=0.0,
        fliplr=0.0,         # bubble orientation matters
        hsv_h=0.01,
        hsv_s=0.15,
        hsv_v=0.2,
    )

    # Copy best.pt to production
    trained_pt = output_dir / "train" / "weights" / "best.pt"
    if trained_pt.exists():
        ML_CACHE.mkdir(parents=True, exist_ok=True)
        if production_pt.exists():
            backup = ML_CACHE / "emoticon_backup.pt"
            shutil.copy2(production_pt, backup)
            print(f"  Backed up old model → {backup}")
        shutil.copy2(trained_pt, production_pt)
        print(f"\nDone! Model saved to: {production_pt}")
    else:
        print(f"\nTraining done. Check {output_dir / 'train'} for results.")


def main():
    parser = argparse.ArgumentParser(description="Augment + Train YOLO for Emoticon_Action")
    parser.add_argument("--augment", action="store_true", help="Augment only (no train)")
    parser.add_argument("--train", action="store_true", help="Train only (dataset must exist)")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    if args.train:
        train_model(args.epochs, args.imgsz, args.batch, args.device)
        return

    print("\n=== Emoticon_Action YOLO Pipeline ===")
    pairs = load_raw_pairs()
    if not pairs:
        print("ERROR: No labeled images found in", RAW_DIR)
        return
    print(f"\nFound {len(pairs)} labeled images in {RAW_DIR.name}")

    print("\nStep 1: Augmenting dataset...")
    generate_dataset(pairs)

    if args.augment:
        print("\n--augment mode: skipping training.")
        return

    print("\nStep 2: Training...")
    train_model(args.epochs, args.imgsz, args.batch, args.device)


if __name__ == "__main__":
    main()
