"""Train improved Emoticon_Action YOLO model by merging old + new labeled data.

Merges:
  - Old: data/raw_images/run_20260306_150801 (11 imgs, polygon labels)
  - New: data/raw_images/run_20260317_093958 (50 imgs, ellipse labels)

Converts all labels to standard YOLO bbox format (cls cx cy w h),
applies augmentations to boost dataset size, and trains YOLOv8n.

Usage:
    python scripts/train_emoticon_v2.py              # full pipeline
    python scripts/train_emoticon_v2.py --augment     # augment only
    python scripts/train_emoticon_v2.py --train       # train only (dataset must exist)
    python scripts/train_emoticon_v2.py --epochs 120  # custom epochs
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
RAW_DIRS = [
    PROJECT_ROOT / "data" / "raw_images" / "run_20260306_150801",  # old: 11 imgs
    PROJECT_ROOT / "data" / "raw_images" / "run_20260317_093958",  # new: 50 imgs
]
ML_CACHE = Path(r"D:\Project\ml_cache\models\yolo")
DATASET_DIR = ML_CACHE / "dataset" / "emoticon_v2"

CLASS_NAME = "Emoticon_Action"
# The annotations mark individual dots of the yellow bubble (3 dots per bubble).
# For YOLO detection we want the BOUNDING BOX of the entire bubble cluster,
# so we group nearby dots and produce a merged bbox per bubble.
# Minimum size for merged bubble bbox (normalized):
MIN_BUBBLE_W = 0.025
MIN_BUBBLE_H = 0.025

TARGET_TOTAL = 200  # augment to this many
VAL_RATIO = 0.15
SEED = 42


def strip_to_yolo_bbox(label_path: Path) -> str:
    """Convert polygon/ellipse labels to standard YOLO bbox format.

    The raw annotations mark individual dots of the yellow emotion bubble
    (3 dots per bubble, each ~0.02x0.007). We group nearby dots into
    bubble clusters and output one bbox per cluster.

    Format in:  cls cx cy w h [angle] [polygon|ellipse]
    Format out: cls cx cy w h  (one per bubble cluster)
    """
    lines = label_path.read_text("utf-8").strip().splitlines()
    dots = []  # (cls, cx, cy, w, h)
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        dots.append((cls, cx, cy, w, h))

    if not dots:
        return ""

    # Group dots into bubble clusters by proximity
    # Each bubble has ~3 dots stacked vertically, spanning ~0.04 in Y
    clusters = _cluster_dots(dots, dist_threshold=0.06)

    out_lines = []
    for cluster in clusters:
        # Compute merged bbox for each cluster
        min_x = min(cx - w / 2 for _, cx, _, w, _ in cluster)
        max_x = max(cx + w / 2 for _, cx, _, w, _ in cluster)
        min_y = min(cy - h / 2 for _, _, cy, _, h in cluster)
        max_y = max(cy + h / 2 for _, _, cy, _, h in cluster)
        # Add margin around bubble
        margin_x = 0.008
        margin_y = 0.008
        min_x = max(0, min_x - margin_x)
        max_x = min(1, max_x + margin_x)
        min_y = max(0, min_y - margin_y)
        max_y = min(1, max_y + margin_y)
        bw = max(max_x - min_x, MIN_BUBBLE_W)
        bh = max(max_y - min_y, MIN_BUBBLE_H)
        bcx = (min_x + max_x) / 2
        bcy = (min_y + max_y) / 2
        out_lines.append(f"0 {bcx:.6f} {bcy:.6f} {bw:.6f} {bh:.6f}")

    return "\n".join(out_lines)


def _cluster_dots(dots, dist_threshold=0.06):
    """Group dots into clusters where each dot is within dist_threshold of at least one other."""
    assigned = [False] * len(dots)
    clusters = []

    for i in range(len(dots)):
        if assigned[i]:
            continue
        cluster = [dots[i]]
        assigned[i] = True
        queue = [i]
        while queue:
            cur = queue.pop(0)
            _, cx0, cy0, _, _ = dots[cur]
            for j in range(len(dots)):
                if assigned[j]:
                    continue
                _, cx1, cy1, _, _ = dots[j]
                dist = ((cx0 - cx1) ** 2 + (cy0 - cy1) ** 2) ** 0.5
                if dist < dist_threshold:
                    assigned[j] = True
                    cluster.append(dots[j])
                    queue.append(j)
        clusters.append(cluster)

    return clusters


def load_all_pairs() -> List[Tuple[Path, Path]]:
    """Load image+label pairs from all raw directories."""
    pairs = []
    for raw_dir in RAW_DIRS:
        if not raw_dir.exists():
            print(f"  [WARN] Directory not found: {raw_dir}")
            continue
        dir_pairs = []
        for lbl in sorted(raw_dir.glob("*.txt")):
            if lbl.name == "classes.txt" or lbl.stat().st_size == 0:
                continue
            img = raw_dir / (lbl.stem + ".jpg")
            if img.exists():
                dir_pairs.append((img, lbl))
        print(f"  {raw_dir.name}: {len(dir_pairs)} labeled images")
        pairs.extend(dir_pairs)
    return pairs


def augment_image(img: np.ndarray, seed: int) -> np.ndarray:
    """Apply random augmentations for data diversity."""
    rng = np.random.RandomState(seed)
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

    # Hue/Saturation jitter
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
    """Build YOLO dataset from all labeled pairs with augmentation."""
    random.seed(SEED)
    np.random.seed(SEED)

    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    all_images = []  # (img, label_text, name)

    for img_path, lbl_path in pairs:
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        label_text = strip_to_yolo_bbox(lbl_path)
        if not label_text.strip():
            continue
        # Use dir+stem to avoid name collisions
        dir_name = lbl_path.parent.name
        name = f"{dir_name}_{img_path.stem}"
        all_images.append((img, label_text, name))

    n_orig = len(all_images)
    print(f"  Original images with labels: {n_orig}")
    total_boxes = sum(len(l.strip().splitlines()) for _, l, _ in all_images)
    print(f"  Total bubble annotations: {total_boxes}")

    if n_orig == 0:
        print("ERROR: No valid images found")
        return

    # Augment to TARGET_TOTAL
    aug_needed = max(0, TARGET_TOTAL - n_orig)
    for aug_idx in range(aug_needed):
        src_idx = aug_idx % n_orig
        orig_img, orig_label, orig_name = all_images[src_idx]
        aug_img = augment_image(orig_img, seed=SEED + 1000 + aug_idx)
        all_images.append((aug_img, orig_label, f"{orig_name}_aug{aug_idx:03d}"))

    print(f"  Total after augmentation: {len(all_images)} ({n_orig} orig + {aug_needed} aug)")

    # Shuffle and split
    indices = list(range(len(all_images)))
    random.shuffle(indices)
    n_val = max(1, int(len(indices) * VAL_RATIO))
    val_indices = set(indices[:n_val])

    for split in ["train", "val"]:
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

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


def train_model(epochs: int = 100, imgsz: int = 640, batch: int = 16, device: str = "0"):
    """Train YOLOv8n on the merged Emoticon_Action dataset."""
    yaml_path = DATASET_DIR / "data.yaml"
    if not yaml_path.exists():
        print("ERROR: data.yaml not found. Run augmentation first.")
        return

    pretrained = "yolov8n.pt"
    output_dir = ML_CACHE / "runs" / "emoticon_v2"
    production_pt = ML_CACHE / "emoticon.pt"

    print(f"\n=== Training Emoticon_Action YOLO v2 ===")
    print(f"  Pretrained: {pretrained}")
    print(f"  Epochs: {epochs}, ImgSz: {imgsz}, Batch: {batch}, Device: {device}")

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
        patience=30,
        save=True,
        plots=True,
        cache=True,
        amp=True,
        # Augmentation tuned for small UI markers
        degrees=5.0,
        translate=0.08,
        scale=0.25,
        mosaic=0.5,
        mixup=0.0,
        flipud=0.0,
        fliplr=0.0,         # bubble orientation matters
        hsv_h=0.01,
        hsv_s=0.15,
        hsv_v=0.2,
    )

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
    parser = argparse.ArgumentParser(description="Train Emoticon YOLO v2 (merged dataset)")
    parser.add_argument("--augment", action="store_true", help="Augment only")
    parser.add_argument("--train", action="store_true", help="Train only")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    if args.train:
        train_model(args.epochs, args.imgsz, args.batch, args.device)
        return

    print("\n=== Emoticon_Action YOLO v2 Pipeline ===")
    pairs = load_all_pairs()
    if not pairs:
        print("ERROR: No labeled images found")
        return
    print(f"\nTotal: {len(pairs)} labeled images from {len(RAW_DIRS)} directories")

    print("\nStep 1: Building dataset...")
    generate_dataset(pairs)

    if args.augment:
        print("\n--augment mode: skipping training.")
        return

    print("\nStep 2: Training...")
    train_model(args.epochs, args.imgsz, args.batch, args.device)


if __name__ == "__main__":
    main()
