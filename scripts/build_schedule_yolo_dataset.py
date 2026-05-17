"""Build a YOLO dataset for Schedule roster overlay detection.

User goal (2026-05-17): replace the current "crop strip + per-cell
template-match" approach for the 全體課程表 popup with a single YOLO
pass that outputs (bbox, character_class) per visible avatar.  Empty
slots = no detection (white/grey background, model learns ignore).

This script:
  1. Walks all trajectory dirs under data/trajectories/
  2. Picks ticks whose OCR contains "全體課程表" (= the overlay is open)
  3. Copies those JPGs to data/yolo_datasets/schedule_roster/images_raw/
  4. Writes a starter data.yaml that lists every avatar file under
     data/captures/角色头像/ as a class (one class per character).

User then labels (or uses our auto-labeler) and runs train_yolo26.py.

Usage:
  python scripts/build_schedule_yolo_dataset.py
  # then move/split images into train/ + val/ subdirs (we provide a
  # helper at the end) and label.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRAJ_ROOT = REPO / "data" / "trajectories"
AVATAR_ROOT = REPO / "data" / "captures" / "角色头像"
OUT_DIR = REPO / "data" / "yolo_datasets" / "schedule_roster"


def find_overlay_ticks() -> list[Path]:
    """Return paths to JPGs where the schedule overlay is open."""
    hits: list[Path] = []
    if not TRAJ_ROOT.is_dir():
        print(f"trajectory root missing: {TRAJ_ROOT}")
        return hits
    for run_dir in sorted(TRAJ_ROOT.iterdir()):
        if not run_dir.is_dir():
            continue
        for json_path in run_dir.glob("tick_*.json"):
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            ocr = payload.get("ocr_boxes") or []
            # Signal: header "全體課程表" present AND multiple room
            # labels visible (any of 視聽室 / 教室 / 圖書館 / 射擊場 /
            # 體育館 / 實驗室 / 載具庫).
            has_header = any(
                "全體課程表" in (b.get("text") or "")
                or "全体课程表" in (b.get("text") or "")
                for b in ocr
            )
            if not has_header:
                continue
            rooms = sum(
                1
                for b in ocr
                if any(
                    rm in (b.get("text") or "")
                    for rm in (
                        "視聽室", "视听室", "教室", "圖書館", "图书馆",
                        "射擊場", "射击场", "體育館", "体育馆",
                        "實驗室", "实验室", "載具庫", "载具库",
                    )
                )
            )
            if rooms < 2:
                continue
            jpg = json_path.with_suffix(".jpg")
            if jpg.exists():
                hits.append(jpg)
    return hits


def build_classes() -> list[str]:
    """List class names = avatar filenames under 角色头像/, stripped of .png."""
    if not AVATAR_ROOT.is_dir():
        print(f"avatar dir missing: {AVATAR_ROOT}")
        return []
    classes = sorted(
        p.stem for p in AVATAR_ROOT.glob("*.png") if not p.stem.startswith(".")
    )
    return classes


def write_dataset(images: list[Path], classes: list[str]) -> None:
    out_imgs = OUT_DIR / "images_raw"
    out_labels = OUT_DIR / "labels_raw"
    out_imgs.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    print(f"copying {len(images)} overlay screenshots → {out_imgs}")
    for src in images:
        # Prefix with run dir name so same-tick numbers don't collide
        run_name = src.parent.name
        dst = out_imgs / f"{run_name}_{src.name}"
        if not dst.exists():
            shutil.copy2(src, dst)
            # Create empty label file (user will fill via labeller)
            (out_labels / f"{dst.stem}.txt").touch()
    print(f"  -> {out_imgs}")

    # data.yaml with classes from 角色头像/
    yaml_path = OUT_DIR / "data.yaml"
    lines = [
        f"path: {OUT_DIR.as_posix()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(classes)}",
        "names:",
    ]
    for i, name in enumerate(classes):
        lines.append(f"  {i}: {name}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote data.yaml with {len(classes)} classes → {yaml_path}")
    print("\nNext steps:")
    print(f"  1. Label the JPGs in {out_imgs} (one bbox per visible avatar,")
    print(f"     class = character key).  Use the dashboard Annotate tool")
    print(f"     or LabelImg / labelme / CVAT.")
    print(f"  2. Split labeled pairs into images/train, images/val,")
    print(f"     labels/train, labels/val (~80/20).")
    print(f"  3. Add a config to scripts/train_yolo26.py and train.")


def main() -> int:
    classes = build_classes()
    print(f"classes from 角色头像/: {len(classes)} characters")
    if not classes:
        print("no avatar reference images found — aborting")
        return 1
    images = find_overlay_ticks()
    print(f"found {len(images)} overlay screenshots across trajectories")
    if not images:
        print("no overlay screenshots found — run pipeline through Schedule")
        return 1
    write_dataset(images, classes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
