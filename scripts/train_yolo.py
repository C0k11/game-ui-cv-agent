"""Train YOLOv8-Nano for Blue Archive UI detection.

Prerequisites:
    pip install ultralytics
    Label images with AnyLabeling (YOLO format) first.

Usage:
    python scripts/train_yolo.py
    python scripts/train_yolo.py --epochs 150 --batch 8
"""

import sys
from pathlib import Path

DATASET_YAML = Path(r"D:\Project\ml_cache\models\yolo\dataset\data.yaml")
OUTPUT_DIR = Path(r"D:\Project\ml_cache\models\yolo")
PRETRAINED = "yolov8n.pt"  # Nano — 3.2M params, ~5-10ms/frame


def main():
    epochs = 100
    batch = 16
    imgsz = 640

    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--epochs" and i + 1 < len(args):
            epochs = int(args[i + 1])
        elif a == "--batch" and i + 1 < len(args):
            batch = int(args[i + 1])
        elif a == "--imgsz" and i + 1 < len(args):
            imgsz = int(args[i + 1])

    # Verify dataset has images
    train_dir = DATASET_YAML.parent / "images" / "train"
    label_dir = DATASET_YAML.parent / "labels" / "train"
    n_images = len(list(train_dir.glob("*.png"))) + len(list(train_dir.glob("*.jpg")))
    n_labels = len(list(label_dir.glob("*.txt")))
    print(f"Dataset: {n_images} images, {n_labels} labels")
    if n_labels == 0:
        print("ERROR: No labels found. Label your images first!")
        print(f"  Images: {train_dir}")
        print(f"  Labels: {label_dir}")
        sys.exit(1)

    from ultralytics import YOLO

    model = YOLO(PRETRAINED)
    results = model.train(
        data=str(DATASET_YAML),
        epochs=epochs,
        imgsz=imgsz,
        device=0,
        batch=batch,
        project=str(OUTPUT_DIR),
        name="train",
        exist_ok=True,
        patience=20,
        save=True,
        plots=True,
    )

    best_pt = OUTPUT_DIR / "train" / "weights" / "best.pt"
    print(f"\nTraining complete!")
    print(f"Best weights: {best_pt}")
    print(f"\nTo use in pipeline, set YOLO_MODEL_PATH={best_pt}")
    print(f"Or copy to: D:\\Project\\ml_cache\\models\\yolo\\best.pt")


if __name__ == "__main__":
    main()
