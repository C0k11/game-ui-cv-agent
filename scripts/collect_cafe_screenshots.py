"""Extract cafe interior screenshots from trajectory data for YOLO training.

Scans all trajectory runs, identifies cafe screenshots (CAFE_HEADPAT phase),
and copies them to the YOLO dataset images directory for labeling.

Usage:
    python scripts/collect_cafe_screenshots.py
    python scripts/collect_cafe_screenshots.py --limit 200
"""

import json
import shutil
import sys
from pathlib import Path

TRAJECTORIES_DIR = Path("data/trajectories")
DATASET_DIR = Path(r"D:\Project\ml_cache\models\yolo\dataset\images\train")

CAFE_PHASES = {"CAFE_HEADPAT", "CAFE_2_HEADPAT", "CAFE_EARNINGS", "CAFE_INVITE",
               "CAFE_2_INVITE", "CAFE_SWITCH"}


def main():
    limit = 200
    if "--limit" in sys.argv:
        idx = sys.argv.index("--limit")
        if idx + 1 < len(sys.argv):
            limit = int(sys.argv[idx + 1])

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    collected = 0

    for run_dir in sorted(TRAJECTORIES_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        traj_file = run_dir / "trajectory.jsonl"
        if not traj_file.exists():
            continue

        with open(traj_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                action = d.get("action")
                if action is None:
                    continue
                phase = action.get("_pipeline_phase", "")
                if phase not in CAFE_PHASES:
                    continue
                step = d.get("step", 0)
                screenshot = run_dir / f"step_{step:06d}.png"
                if not screenshot.exists():
                    continue

                dst_name = f"{run_dir.name}_s{step:04d}.png"
                dst = DATASET_DIR / dst_name
                if dst.exists():
                    continue

                shutil.copy2(screenshot, dst)
                collected += 1
                if collected % 20 == 0:
                    print(f"  collected {collected} screenshots...")
                if collected >= limit:
                    break
        if collected >= limit:
            break

    print(f"\nDone. Collected {collected} cafe screenshots to:")
    print(f"  {DATASET_DIR}")
    print(f"\nNext step: label them with AnyLabeling or LabelImg.")
    print(f"  - Class: headpat_bubble")
    print(f"  - Format: YOLO (normalized xywh)")
    print(f"  - Save labels to: {DATASET_DIR.parent.parent / 'labels' / 'train'}")


if __name__ == "__main__":
    main()
