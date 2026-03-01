"""Resume training for headpat and schedule skills (UI already done)."""
import shutil
from pathlib import Path

MODEL_OUTPUT = Path(r"D:\Project\ml_cache\models\yolo")
DATASET_ROOT = Path("data/yolo_datasets")

SKILLS_TO_TRAIN = ["headpat", "schedule"]

def main():
    from ultralytics import YOLO
    for skill_name in SKILLS_TO_TRAIN:
        yaml_path = DATASET_ROOT / skill_name / "data.yaml"
        if not yaml_path.exists():
            print(f"[SKIP] {skill_name}: data.yaml not found")
            continue

        print(f"\n{'='*60}")
        print(f"  Training skill: {skill_name}")
        print(f"{'='*60}")

        model = YOLO("yolo26n.pt")
        results = model.train(
            data=str(yaml_path),
            epochs=100,
            imgsz=1280,
            batch=4,
            device=0,
            workers=2,
            name=skill_name,
            project=str(MODEL_OUTPUT / "runs"),
            patience=20,
            save=True,
            exist_ok=True,
        )
        best_src = MODEL_OUTPUT / "runs" / skill_name / "weights" / "best.pt"
        best_dst = MODEL_OUTPUT / f"{skill_name}.pt"
        if best_src.exists():
            shutil.copy2(best_src, best_dst)
            print(f"  → Saved model: {best_dst}")

    print("\nDONE!")

if __name__ == "__main__":
    main()
