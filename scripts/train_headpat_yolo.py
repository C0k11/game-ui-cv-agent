"""Split dataset into train/val and train YOLOv8n on headpat bubble data.

Usage:
    python scripts/train_headpat_yolo.py
"""
import os
import sys
import random
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "data" / "yolo_headpat_dataset"
TRAIN_DIR = DATASET / "train"
VAL_DIR = DATASET / "val"
TRAIN_IMAGES = TRAIN_DIR / "images"
TRAIN_LABELS = TRAIN_DIR / "labels"
VAL_IMAGES = VAL_DIR / "images"
VAL_LABELS = VAL_DIR / "labels"

VAL_RATIO = 0.15
SEED = 42


def split_dataset():
    """Split images/labels into train/val."""
    src_images = DATASET / "images"
    src_labels = DATASET / "labels"
    
    all_images = sorted(src_images.glob("*.jpg"))
    print(f"Total labeled images: {len(all_images)}")
    
    random.seed(SEED)
    random.shuffle(all_images)
    
    val_count = int(len(all_images) * VAL_RATIO)
    val_set = set(all_images[:val_count])
    
    for d in [TRAIN_IMAGES, TRAIN_LABELS, VAL_IMAGES, VAL_LABELS]:
        d.mkdir(parents=True, exist_ok=True)
    
    train_n = val_n = 0
    for img_path in all_images:
        label_path = src_labels / img_path.with_suffix(".txt").name
        if not label_path.exists():
            continue
        
        if img_path in val_set:
            shutil.copy2(img_path, VAL_IMAGES / img_path.name)
            shutil.copy2(label_path, VAL_LABELS / label_path.name)
            val_n += 1
        else:
            shutil.copy2(img_path, TRAIN_IMAGES / img_path.name)
            shutil.copy2(label_path, TRAIN_LABELS / label_path.name)
            train_n += 1
    
    print(f"Split: train={train_n}, val={val_n}")
    return train_n, val_n


def write_yaml():
    """Write dataset.yaml for YOLO training."""
    yaml_path = DATASET / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {DATASET}\n")
        f.write(f"train: train/images\n")
        f.write(f"val: val/images\n")
        f.write(f"nc: 1\n")
        f.write(f"names: ['headpat_bubble']\n")
    print(f"Dataset YAML: {yaml_path}")
    return yaml_path


def train(yaml_path):
    """Train YOLOv8n."""
    from ultralytics import YOLO
    
    model = YOLO("yolov8n.pt")  # start from pretrained nano
    print(f"\n=== Starting YOLO Training ===")
    print(f"Model: yolov8n")
    print(f"Dataset: {yaml_path}")
    print(f"Epochs: 50")
    print(f"Image size: 640")
    print()
    
    results = model.train(
        data=str(yaml_path),
        epochs=50,
        imgsz=640,
        batch=16,
        name="headpat_bubble",
        project=str(REPO / "data" / "yolo_training"),
        exist_ok=True,
        verbose=True,
        device="0",  # GPU
    )
    
    # Copy best model to ml_cache
    best_path = REPO / "data" / "yolo_training" / "headpat_bubble" / "weights" / "best.pt"
    if best_path.exists():
        dst = REPO.parent / "ml_cache" / "models" / "yolo" / "headpat.pt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_path, dst)
        print(f"\n=== Training Complete ===")
        print(f"Best model: {best_path}")
        print(f"Deployed to: {dst}")
    else:
        print(f"WARNING: best.pt not found at {best_path}")


if __name__ == "__main__":
    print("=== YOLO Headpat Training Pipeline ===\n")
    
    # Step 1: Split
    train_n, val_n = split_dataset()
    
    # Step 2: Write YAML
    yaml_path = write_yaml()
    
    # Step 3: Train
    train(yaml_path)
