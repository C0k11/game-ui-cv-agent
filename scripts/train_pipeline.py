"""
Expanded auto-annotation + per-skill dataset preparation + YOLO training pipeline.

Agent Skills:
  - ui:       General navigation buttons (lobby, menus, popups)
  - headpat:  Cafe headpat bubbles, earnings, invites
  - schedule: Schedule page elements (locks, switches, all-schedule btn)
  - battle:   Combat UI (placeholder - no data yet)

Flow:
  1. Re-annotate all raw images with expanded template set
  2. Split annotations into per-skill datasets (train/val 85/15)
  3. Generate data.yaml for each skill
  4. Train YOLO26n per skill
"""
import cv2
import numpy as np
import os
import sys
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# ─── Skill Definitions ───────────────────────────────────────────────────────

SKILLS = {
    "ui": {
        "classes": {
            0: {"name": "start_btn",    "templates": ["点击开始.png"],                                              "max": 1},
            1: {"name": "close_btn",    "templates": ["游戏内很多页面窗口的叉.png", "内嵌公告的叉.png", "公告叉叉.png"], "max": 3},
            2: {"name": "confirm_btn",  "templates": ["确认(可以点space）.png"],                                     "max": 1},
            3: {"name": "cancel_btn",   "templates": ["取消（可点Esc）.png"],                                        "max": 1},
            4: {"name": "back_btn",     "templates": ["返回按钮.png"],                                              "max": 1},
            5: {"name": "home_btn",     "templates": ["Home按钮.png"],                                              "max": 1},
            6: {"name": "cafe_btn",     "templates": ["咖啡厅.png"],                                                "max": 1},
            7: {"name": "schedule_btn", "templates": ["课程表.png"],                                                "max": 1},
            8: {"name": "club_btn",     "templates": ["社交.png"],                                                  "max": 1},
            9: {"name": "mail_btn",     "templates": ["邮箱.png"],                                                  "max": 1},
            10: {"name": "shop_btn",    "templates": ["商店.png"],                                                  "max": 1},
            11: {"name": "recruit_btn", "templates": ["招募.png"],                                                  "max": 1},
            12: {"name": "student_btn", "templates": ["学生.png"],                                                  "max": 1},
            13: {"name": "craft_btn",   "templates": ["制造.png"],                                                  "max": 1},
            14: {"name": "settings_btn","templates": ["设置齿轮.png"],                                              "max": 1},
            15: {"name": "red_dot",     "templates": ["红点.png"],                                                  "max": 10},
        },
    },
    "headpat": {
        "classes": {
            0: {"name": "headpat_bubble",    "templates": ["可摸头的标志.png"],       "max": 5},
            1: {"name": "cafe_earnings_btn", "templates": ["咖啡厅收益按钮.png"],    "max": 1},
            2: {"name": "move_to_shop2",     "templates": ["移动至2号店.png"],        "max": 1},
            3: {"name": "invite_ticket",     "templates": ["邀请卷（带黄点）.png"],   "max": 1},
            4: {"name": "invite_btn",        "templates": ["邀请.png"],              "max": 1},
            5: {"name": "emoticon_action",   "templates": ["Emoticon_Action.png"],   "max": 5},
        },
    },
    "schedule": {
        "classes": {
            0: {"name": "left_switch",       "templates": ["左切换.png"],             "max": 1},
            1: {"name": "right_switch",      "templates": ["右切换.png"],             "max": 1},
            2: {"name": "all_schedule_btn",  "templates": ["全体课程表.png"],          "max": 1},
            3: {"name": "schedule_lock",     "templates": ["课程表锁.png"],           "max": 6},
            4: {"name": "ticket_count",      "templates": ["课程表票持有数量.png"],    "max": 1},
        },
    },
    # battle: placeholder for future - no templates yet
}

RAW_DIR = Path("data/raw_images/run_20260226_193214")
TEMPLATES_DIR = Path("data/captures")
DATASET_ROOT = Path("data/yolo_datasets")
MODEL_OUTPUT = Path(r"D:\Project\ml_cache\models\yolo")

THRESHOLD = 0.80
TRAIN_RATIO = 0.85

# ─── Template Matching ────────────────────────────────────────────────────────

def load_template(tmpl_path: Path):
    raw = cv2.imdecode(np.fromfile(str(tmpl_path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None, None
    if raw.ndim == 3 and raw.shape[2] == 4:
        return raw[:, :, :3], raw[:, :, 3]
    return raw, None

def match_fast(image, bgr_tmpl, mask, max_instances, threshold):
    h, w = bgr_tmpl.shape[:2]
    if h > image.shape[0] or w > image.shape[1]:
        return []
    if mask is not None:
        res = cv2.matchTemplate(image, bgr_tmpl, cv2.TM_CCOEFF_NORMED, mask=mask)
    else:
        res = cv2.matchTemplate(image, bgr_tmpl, cv2.TM_CCOEFF_NORMED)
    matches = []
    for _ in range(max_instances):
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val < threshold:
            break
        x, y = max_loc
        matches.append((x, y, w, h, float(max_val)))
        # suppress
        x1, y1 = max(0, x - w // 2), max(0, y - h // 2)
        x2, y2 = min(res.shape[1], x + w // 2 + 1), min(res.shape[0], y + h // 2 + 1)
        res[y1:y2, x1:x2] = 0.0
    return matches

# ─── Step 1: Annotate per skill ──────────────────────────────────────────────

def annotate_skill(skill_name, skill_def, image_files, threshold):
    print(f"\n{'='*60}")
    print(f"  Annotating skill: {skill_name}  ({len(skill_def['classes'])} classes)")
    print(f"{'='*60}")

    # Load templates
    loaded = {}
    for cls_id, info in skill_def["classes"].items():
        entries = []
        for tmpl_name in info["templates"]:
            p = TEMPLATES_DIR / tmpl_name
            if not p.exists():
                print(f"  [WARN] Missing template: {tmpl_name}")
                continue
            bgr, mask = load_template(p)
            if bgr is None:
                continue
            entries.append((tmpl_name, bgr, mask, info["max"]))
        loaded[cls_id] = entries

    # Annotate all images
    results = {}  # img_path -> list of (cls_id, x_center, y_center, w, h)
    for img_path in tqdm(image_files, desc=f"  {skill_name}"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        labels = []
        for cls_id, entries in loaded.items():
            for tmpl_name, bgr_tmpl, mask, max_inst in entries:
                for (x, y, w, h, score) in match_fast(img, bgr_tmpl, mask, max_inst, threshold):
                    xc = (x + w / 2) / img_w
                    yc = (y + h / 2) / img_h
                    nw = w / img_w
                    nh = h / img_h
                    labels.append(f"{cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
        results[img_path] = labels
    
    total = sum(len(v) for v in results.values())
    has_labels = sum(1 for v in results.values() if len(v) > 0)
    print(f"  → {total} boxes across {has_labels}/{len(image_files)} images with detections")
    return results

# ─── Step 2: Split into train/val and write YOLO dataset ─────────────────────

def write_yolo_dataset(skill_name, skill_def, annotations: dict):
    ds_dir = DATASET_ROOT / skill_name
    train_img = ds_dir / "images" / "train"
    val_img = ds_dir / "images" / "val"
    train_lbl = ds_dir / "labels" / "train"
    val_lbl = ds_dir / "labels" / "val"
    for d in [train_img, val_img, train_lbl, val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    # Only include images that have at least one annotation for this skill
    annotated = [(p, lbls) for p, lbls in annotations.items() if len(lbls) > 0]
    # Also include some negative samples (no labels) for robustness
    negatives = [(p, lbls) for p, lbls in annotations.items() if len(lbls) == 0]
    # Take up to 20% negatives
    max_neg = max(1, len(annotated) // 5)
    random.shuffle(negatives)
    all_samples = annotated + negatives[:max_neg]
    random.shuffle(all_samples)

    split_idx = int(len(all_samples) * TRAIN_RATIO)
    train_set = all_samples[:split_idx]
    val_set = all_samples[split_idx:]

    for subset, img_dir, lbl_dir in [(train_set, train_img, train_lbl), (val_set, val_img, val_lbl)]:
        for img_path, labels in subset:
            dst_img = img_dir / img_path.name
            dst_lbl = lbl_dir / (img_path.stem + ".txt")
            shutil.copy2(img_path, dst_img)
            with open(dst_lbl, "w") as f:
                f.write("\n".join(labels))

    # Write data.yaml
    class_names = [skill_def["classes"][i]["name"] for i in sorted(skill_def["classes"].keys())]
    yaml_path = ds_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {ds_dir.resolve()}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write(f"nc: {len(class_names)}\n")
        f.write(f"names: {class_names}\n")

    print(f"  Dataset: {len(train_set)} train + {len(val_set)} val")
    print(f"  Config:  {yaml_path}")
    return yaml_path

# ─── Step 3: Train ───────────────────────────────────────────────────────────

def train_skill(skill_name, yaml_path):
    from ultralytics import YOLO
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
        name=f"{skill_name}",
        project=str(MODEL_OUTPUT / "runs"),
        patience=20,
        save=True,
        exist_ok=True,
    )
    # Copy best.pt to the standard location
    best_src = MODEL_OUTPUT / "runs" / skill_name / "weights" / "best.pt"
    best_dst = MODEL_OUTPUT / f"{skill_name}.pt"
    if best_src.exists():
        shutil.copy2(best_src, best_dst)
        print(f"  → Saved model: {best_dst}")
    return best_dst

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    random.seed(42)
    image_files = sorted(RAW_DIR.glob("*.jpg"))
    print(f"Found {len(image_files)} raw images in {RAW_DIR}")

    if not image_files:
        print("No images found. Exiting.")
        return

    DATASET_ROOT.mkdir(parents=True, exist_ok=True)

    yaml_paths = {}
    for skill_name, skill_def in SKILLS.items():
        annotations = annotate_skill(skill_name, skill_def, image_files, THRESHOLD)
        yaml_path = write_yolo_dataset(skill_name, skill_def, annotations)
        yaml_paths[skill_name] = yaml_path

    print(f"\n{'='*60}")
    print(f"  All datasets prepared. Starting training...")
    print(f"{'='*60}")

    for skill_name, yaml_path in yaml_paths.items():
        train_skill(skill_name, yaml_path)

    print(f"\n{'='*60}")
    print(f"  ALL DONE! Models saved to: {MODEL_OUTPUT}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
