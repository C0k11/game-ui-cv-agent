"""Train YOLO26-Nano for Blue Archive UI detection with agent skill separation.

YOLO26n: 2.4M params, NMS-free end-to-end, ~39ms CPU ONNX.

Prerequisites:
    pip install ultralytics
    Label images with AnyLabeling (YOLO format) first.

Usage:
    python scripts/train_yolo.py --skill cafe
    python scripts/train_yolo.py --skill schedule --epochs 150 --batch 8 --imgsz 960
"""

import argparse
import shutil
import yaml
from pathlib import Path

# Base paths
ML_CACHE_DIR = Path(r"D:\Project\ml_cache\models\yolo")
PRETRAINED = "yolo26n.pt"  # YOLO26 Nano — 2.4M params, NMS-free, ~39ms CPU ONNX

# Define skills and their target classes
AGENT_SKILLS = {
    "cafe": {
        "classes": {0: "headpat_bubble"},
        "desc": "Detects headpat bubbles for cafe interactions."
    },
    "schedule": {
        "classes": {0: "student_avatar"},
        "desc": "Detects student avatars in the schedule rooms."
    },
    "combat": {
        "classes": {0: "enemy", 1: "ex_skill"},
        "desc": "Future placeholder for combat detection."
    }
}

def ensure_dataset_yaml(skill_name: str, dataset_dir: Path) -> Path:
    """Auto-generate data.yaml for the specific skill."""
    yaml_path = dataset_dir / "data.yaml"
    val_dir = dataset_dir / "images" / "val"
    has_val = val_dir.exists() and any(val_dir.iterdir())
    
    classes = AGENT_SKILLS[skill_name]["classes"]
    
    data = {
        "path": str(dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val" if has_val else "images/train",
        "nc": len(classes),
        "names": classes
    }
    
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=False)
        
    if not has_val:
        print("NOTE: no val split found, using train as val (split later for better metrics)")
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Train YOLO26n for Blue Archive Agent Skills")
    parser.add_argument("--skill", type=str, required=True, choices=AGENT_SKILLS.keys(),
                        help="Which agent skill model to train (e.g., cafe, schedule)")
    parser.add_argument("--epochs", type=int, default=80, help="training epochs")
    parser.add_argument("--batch", type=int, default=16, help="batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="image size (640 or 960 for small targets)")
    parser.add_argument("--device", type=str, default="0", help="device: 0=GPU, cpu=CPU")
    args = parser.parse_args()

    skill = args.skill
    print(f"\n--- Training Agent Skill: [{skill.upper()}] ---")
    print(f"Description: {AGENT_SKILLS[skill]['desc']}")
    
    # Dataset and Output paths specific to the skill
    dataset_dir = ML_CACHE_DIR / "datasets" / skill
    output_dir = ML_CACHE_DIR / "runs" / skill
    production_pt = ML_CACHE_DIR / f"{skill}.pt"

    # Verify dataset
    train_dir = dataset_dir / "images" / "train"
    label_dir = dataset_dir / "labels" / "train"
    
    if not train_dir.exists() or not label_dir.exists():
        print(f"\nERROR: Dataset directories not found for skill '{skill}'!")
        print(f"Expected to find:")
        print(f"  Images: {train_dir}")
        print(f"  Labels: {label_dir}")
        print("\nPlease create these folders and add your labeled data.")
        return

    n_images = len(list(train_dir.glob("*.png"))) + len(list(train_dir.glob("*.jpg")))
    n_labels = len(list(label_dir.glob("*.txt")))
    print(f"\nDataset: {n_images} images, {n_labels} labels")
    
    if n_labels == 0:
        print("\nERROR: No labels found. Label your images first!")
        return

    yaml_path = ensure_dataset_yaml(skill, dataset_dir)

    from ultralytics import YOLO

    model = YOLO(PRETRAINED)
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        workers=8,
        project=str(output_dir),
        name="train",
        exist_ok=True,
        patience=20,
        save=True,
        plots=True,
        cache=True,
    )

    # Auto-copy best.pt to production path named after the skill
    trained_pt = output_dir / "train" / "weights" / "best.pt"
    if trained_pt.exists():
        shutil.copy2(trained_pt, production_pt)
        print(f"\nTraining complete! Model copied to production path:")
        print(f"  {production_pt}")
        print(f"Pipeline will auto-detect '{skill}.pt' on next startup.")
    else:
        print(f"\nTraining complete but best.pt not found at {trained_pt}")
        print(f"Check {output_dir / 'train'} for results.")

if __name__ == "__main__":
    main()
