"""Train YOLO26-Nano for Blue Archive UI detection.

YOLO26n: 2.4M params, NMS-free end-to-end, ~39ms CPU ONNX.

Prerequisites:
    pip install ultralytics
    Label images with AnyLabeling (YOLO format) first.

Usage:
    python scripts/train_yolo.py
    python scripts/train_yolo.py --epochs 150 --batch 8 --imgsz 960
"""

import argparse
import shutil
from pathlib import Path

DATASET_DIR = Path(r"D:\Project\ml_cache\models\yolo\dataset")
OUTPUT_DIR = Path(r"D:\Project\ml_cache\models\yolo")
PRODUCTION_PT = OUTPUT_DIR / "best.pt"
PRETRAINED = "yolo26n.pt"  # YOLO26 Nano — 2.4M params, NMS-free, ~39ms CPU ONNX


def ensure_dataset_yaml() -> Path:
    """Auto-generate data.yaml. Uses train as val if no val split exists."""
    yaml_path = DATASET_DIR / "data.yaml"
    val_dir = DATASET_DIR / "images" / "val"
    has_val = val_dir.exists() and any(val_dir.iterdir())

    content = (
        f"path: {DATASET_DIR.resolve()}\n"
        f"train: images/train\n"
        f"val: {'images/val' if has_val else 'images/train'}\n"
        f"\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: headpat_bubble\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    if not has_val:
        print("NOTE: no val split found, using train as val (split later for better metrics)")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Train YOLO26n for Blue Archive UI detection")
    parser.add_argument("--epochs", type=int, default=80, help="training epochs")
    parser.add_argument("--batch", type=int, default=16, help="batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="image size (640 or 960 for small targets)")
    parser.add_argument("--device", type=str, default="0", help="device: 0=GPU, cpu=CPU")
    args = parser.parse_args()

    # Verify dataset
    train_dir = DATASET_DIR / "images" / "train"
    label_dir = DATASET_DIR / "labels" / "train"
    n_images = len(list(train_dir.glob("*.png"))) + len(list(train_dir.glob("*.jpg")))
    n_labels = len(list(label_dir.glob("*.txt")))
    print(f"Dataset: {n_images} images, {n_labels} labels")
    if n_labels == 0:
        print("ERROR: No labels found. Label your images first!")
        print(f"  Images: {train_dir}")
        print(f"  Labels: {label_dir}")
        return

    yaml_path = ensure_dataset_yaml()

    from ultralytics import YOLO

    model = YOLO(PRETRAINED)
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        workers=8,
        project=str(OUTPUT_DIR),
        name="train",
        exist_ok=True,
        patience=20,
        save=True,
        plots=True,
        cache=True,
    )

    # Auto-copy best.pt to production path
    trained_pt = OUTPUT_DIR / "train" / "weights" / "best.pt"
    if trained_pt.exists():
        shutil.copy2(trained_pt, PRODUCTION_PT)
        print(f"\nTraining complete! best.pt copied to production path:")
        print(f"  {PRODUCTION_PT}")
        print(f"Pipeline will auto-detect it on next startup.")
    else:
        print(f"\nTraining complete but best.pt not found at {trained_pt}")
        print(f"Check {OUTPUT_DIR / 'train'} for results.")


if __name__ == "__main__":
    main()
