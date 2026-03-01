"""Train YOLO model for Blue Archive UI detection.

Supports two modes:
  1. Full model: Train on ALL labeled classes from raw_images datasets
  2. Skill model: Train on specific classes for one agent skill

Usage:
    # Train full model on all labeled data:
    python scripts/train_yolo.py

    # Custom settings:
    python scripts/train_yolo.py --epochs 120 --imgsz 960 --batch 8

    # Train skill-specific model (future):
    python scripts/train_yolo.py --skill cafe
"""

import argparse
import random
import shutil
import yaml
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw_images"
ML_CACHE = Path(r"D:\Project\ml_cache\models\yolo")
DATASET_DIR = ML_CACHE / "dataset" / "full"
PRETRAINED = str(Path(__file__).resolve().parent.parent / "yolo26n.pt")


def collect_labeled_pairs() -> list[tuple[Path, Path]]:
    """Scan all raw_images datasets and collect (image, label) pairs."""
    pairs = []
    for ds_dir in sorted(RAW_DIR.iterdir()):
        if not ds_dir.is_dir():
            continue
        classes_file = ds_dir / "classes.txt"
        if not classes_file.exists():
            continue
        for lbl in sorted(ds_dir.glob("*.txt")):
            if lbl.name == "classes.txt":
                continue
            if lbl.stat().st_size == 0:
                continue
            # Find matching image
            img = None
            for ext in (".jpg", ".png", ".jpeg"):
                candidate = ds_dir / (lbl.stem + ext)
                if candidate.exists():
                    img = candidate
                    break
            if img:
                pairs.append((img, lbl))
    return pairs


def read_classes() -> list[str]:
    """Read classes.txt from the first dataset that has one."""
    for ds_dir in sorted(RAW_DIR.iterdir()):
        cf = ds_dir / "classes.txt"
        if cf.exists():
            lines = cf.read_text("utf-8").strip().splitlines()
            return [l.strip() for l in lines if l.strip()]
    return []


def prepare_dataset(pairs: list[tuple[Path, Path]], classes: list[str],
                    val_ratio: float = 0.2, seed: int = 42):
    """Create YOLO directory structure with train/val split."""
    random.seed(seed)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_ratio))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    for split, split_pairs in [("train", train_pairs), ("val", val_pairs)]:
        img_dir = DATASET_DIR / "images" / split
        lbl_dir = DATASET_DIR / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        # Clean old files
        for f in img_dir.glob("*"):
            f.unlink()
        for f in lbl_dir.glob("*"):
            f.unlink()
        for img, lbl in split_pairs:
            shutil.copy2(img, img_dir / img.name)
            shutil.copy2(lbl, lbl_dir / lbl.name)
        print(f"  {split}: {len(split_pairs)} images")

    # Write data.yaml
    names = {i: n for i, n in enumerate(classes)}
    data = {
        "path": str(DATASET_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(classes),
        "names": names,
    }
    yaml_path = DATASET_DIR / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=False,
                  allow_unicode=True)
    print(f"  data.yaml: {yaml_path}")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Train YOLO for Blue Archive UI")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--skill", type=str, default=None,
                        help="Train skill-specific model (future)")
    args = parser.parse_args()

    print("\n=== Blue Archive YOLO Training ===")

    # Collect data
    classes = read_classes()
    if not classes:
        print("ERROR: No classes.txt found in any dataset")
        return
    print(f"\nClasses ({len(classes)}):")
    for i, c in enumerate(classes):
        print(f"  {i}: {c}")

    pairs = collect_labeled_pairs()
    if not pairs:
        print("\nERROR: No labeled images found")
        return
    print(f"\nLabeled images: {len(pairs)}")

    # Prepare dataset
    print("\nPreparing dataset...")
    yaml_path = prepare_dataset(pairs, classes)

    # Train
    print(f"\nStarting training...")
    print(f"  Pretrained: {PRETRAINED}")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch}, ImgSz: {args.imgsz}")
    print(f"  Device: {args.device}")

    output_dir = ML_CACHE / "runs" / "full"
    production_pt = ML_CACHE / "full.pt"

    from ultralytics import YOLO
    model = YOLO(PRETRAINED)
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        workers=4,
        project=str(output_dir),
        name="train",
        exist_ok=True,
        patience=25,
        save=True,
        plots=True,
        cache=True,
        amp=True,
    )

    # Copy best.pt
    trained_pt = output_dir / "train" / "weights" / "best.pt"
    if trained_pt.exists():
        ML_CACHE.mkdir(parents=True, exist_ok=True)
        shutil.copy2(trained_pt, production_pt)
        print(f"\nDone! Model saved to: {production_pt}")
    else:
        print(f"\nTraining done. Check {output_dir / 'train'} for results.")


if __name__ == "__main__":
    main()
