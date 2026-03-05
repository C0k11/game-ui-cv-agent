"""Expanded YOLO Training — add per-character avatars + 收起咖啡厅UI.

Builds a unified dataset that merges:
  1. Existing raw labeled game screenshots (31 classes, indices 0-30)
  2. Synthetic 收起咖啡厅UI samples (class 31) — prevents false 叉叉 detections
  3. Per-character avatar classes (classes 32+) from wiki portrait augmentation

Usage:
  python scripts/train_expanded_yolo.py                  # Generate dataset only
  python scripts/train_expanded_yolo.py --train           # Generate + train
  python scripts/train_expanded_yolo.py --preview 10      # Preview samples
"""
import argparse
import math
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
AVATARS_DIR = PROJECT_ROOT / "data" / "captures" / "角色头像"
CAFE_UI_ICON = PROJECT_ROOT / "data" / "captures" / "收起咖啡厅UI.png"
RAW_DIR = PROJECT_ROOT / "data" / "raw_images"
ML_CACHE = Path(r"D:\Project\ml_cache\models\yolo")
DATASET_DIR = ML_CACHE / "dataset" / "expanded"
# Use current production model as pretrained base
PRETRAINED = str(ML_CACHE / "full.pt")

# ── Existing class list (indices 0-30) ──
BASE_CLASSES = [
    "叉叉", "叉叉1", "清辉石", "信用点", "AP体力",        # 0-4
    "邮箱", "叉叉2", "红点", "黄点", "角色头像",            # 5-9
    "角色可摸头黄色感叹号", "返回键", "主界面按钮",          # 10-12
    "momotalk的叉叉", "上下滑动的游标",                     # 13-14
    "提升好感度的后的爱心", "左切换", "右切换", "锁",        # 15-18
    "打勾", "快进", "快速编辑", "黄勾勾",                   # 19-22
    "战斗暂停键", "敌人数量", "时间",                       # 23-25
    "战斗速度调节第三档", "自动战斗已开启", "⭐",           # 26-28
    "跳过战斗的勾勾", "可领取奖",                           # 29-30
]

# ── Avatar augmentation params (reused from augment_avatar_dataset.py) ──
AVATAR_SIZES = [64, 80, 96, 112, 128]


# ═══════════════════════════════════════════════════════════════════════
#  Class List Builder
# ═══════════════════════════════════════════════════════════════════════

def build_class_list() -> Tuple[List[str], int, Dict[str, int]]:
    """Build full class list: base 31 + 收起咖啡厅UI + per-character.

    Returns:
        (full_class_list, cafe_ui_class_id, char_name_to_id)
    """
    classes = list(BASE_CLASSES)  # 0-30
    cafe_ui_id = len(classes)
    classes.append("收起咖啡厅UI")  # 31

    # Per-character classes sorted alphabetically
    char_map: Dict[str, int] = {}
    if AVATARS_DIR.exists():
        for png in sorted(AVATARS_DIR.glob("*.png")):
            name = png.stem  # e.g. "Hina_(Dress)"
            cid = len(classes)
            classes.append(f"avatar_{name}")
            char_map[name] = cid

    return classes, cafe_ui_id, char_map


# ═══════════════════════════════════════════════════════════════════════
#  收起咖啡厅UI Synthetic Data
# ═══════════════════════════════════════════════════════════════════════

def generate_cafe_ui_samples(class_id: int, count: int = 200) -> List[Tuple[np.ndarray, str]]:
    """Generate synthetic training samples for the 收起咖啡厅UI icon.

    Takes the captured icon and applies random transforms:
    - Scale variations
    - Rotation (slight)
    - Brightness/contrast jitter
    - Random backgrounds (game-UI-like grays/blues)

    Returns list of (image_bgr, yolo_label_line).
    """
    if not CAFE_UI_ICON.exists():
        print(f"  WARNING: {CAFE_UI_ICON} not found, skipping")
        return []

    icon_bgra = cv2.imdecode(
        np.fromfile(str(CAFE_UI_ICON), dtype=np.uint8), cv2.IMREAD_UNCHANGED
    )
    if icon_bgra is None:
        print(f"  WARNING: Failed to load {CAFE_UI_ICON}")
        return []

    samples = []
    for i in range(count):
        canvas_size = random.choice([128, 160, 192, 224, 256])

        # Random background
        bg_type = random.choice(["gray", "blue", "white", "gradient"])
        if bg_type == "gray":
            v = random.randint(180, 240)
            canvas = np.full((canvas_size, canvas_size, 3), v, dtype=np.uint8)
        elif bg_type == "blue":
            canvas = np.full((canvas_size, canvas_size, 3),
                             (random.randint(200, 240), random.randint(200, 230),
                              random.randint(180, 210)), dtype=np.uint8)
        elif bg_type == "white":
            canvas = np.full((canvas_size, canvas_size, 3),
                             random.randint(230, 255), dtype=np.uint8)
        else:  # gradient
            canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
            for y in range(canvas_size):
                v = int(200 + 40 * y / canvas_size)
                canvas[y, :] = (v, v, v)

        # Add slight noise
        if random.random() < 0.4:
            noise = np.random.randint(-8, 8, canvas.shape, dtype=np.int16)
            canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Scale icon
        icon_scale = random.uniform(0.3, 0.7)
        icon_size = max(16, int(canvas_size * icon_scale))
        resized = cv2.resize(icon_bgra, (icon_size, icon_size), interpolation=cv2.INTER_AREA)

        # Slight rotation
        if random.random() < 0.3:
            angle = random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((icon_size / 2, icon_size / 2), angle, 1.0)
            resized = cv2.warpAffine(resized, M, (icon_size, icon_size),
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

        # Brightness/contrast jitter
        if random.random() < 0.5:
            pil_icon = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGRA2RGBA))
            pil_icon = ImageEnhance.Brightness(pil_icon).enhance(random.uniform(0.7, 1.3))
            pil_icon = ImageEnhance.Contrast(pil_icon).enhance(random.uniform(0.8, 1.2))
            resized = cv2.cvtColor(np.array(pil_icon), cv2.COLOR_RGBA2BGRA)

        # Place on canvas
        max_x = canvas_size - icon_size
        max_y = canvas_size - icon_size
        if max_x <= 0 or max_y <= 0:
            continue
        ox = random.randint(0, max_x)
        oy = random.randint(0, max_y)

        # Alpha composite
        if resized.shape[2] == 4:
            alpha = resized[:, :, 3:4].astype(np.float32) / 255.0
            bgr = resized[:, :, :3].astype(np.float32)
            roi = canvas[oy:oy + icon_size, ox:ox + icon_size].astype(np.float32)
            blended = (bgr * alpha + roi * (1 - alpha)).astype(np.uint8)
            canvas[oy:oy + icon_size, ox:ox + icon_size] = blended
        else:
            canvas[oy:oy + icon_size, ox:ox + icon_size] = resized[:, :, :3]

        # YOLO label: class cx cy w h (normalized)
        cx = (ox + icon_size / 2) / canvas_size
        cy = (oy + icon_size / 2) / canvas_size
        bw = icon_size / canvas_size
        bh = icon_size / canvas_size
        label = f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
        samples.append((canvas, label))

    return samples


# ═══════════════════════════════════════════════════════════════════════
#  Per-Character Avatar Synthetic Data (reusing augmentation logic)
# ═══════════════════════════════════════════════════════════════════════

def crop_face_from_wiki(img_path: Path, target_size: int = 96) -> Optional[np.ndarray]:
    """Crop face area from wiki portrait (456×404 RGBA) and resize to square."""
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    h, w = img.shape[:2]
    crop_top = 0
    crop_bot = int(h * 0.65)
    crop_left = int(w * 0.125)
    crop_right = int(w * 0.875)
    img = img[crop_top:crop_bot, crop_left:crop_right]
    ch, cw = img.shape[:2]
    if cw > ch:
        off = (cw - ch) // 2
        img = img[:, off:off + ch]
    elif ch > cw:
        off = (ch - cw) // 2
        img = img[off:off + cw, :]
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return img


def apply_circular_mask(img: np.ndarray) -> np.ndarray:
    """Apply circular mask to make avatar look like roster thumbnail."""
    h, w = img.shape[:2]
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 2
    yy, xx = np.ogrid[:h, :w]
    circle = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius ** 2
    if img.shape[2] == 4:
        img[~circle, 3] = 0
    else:
        img[~circle] = [40, 40, 40]
    return img


def draw_heart(size: int = 24, color: Tuple[int, ...] = (255, 105, 180)) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    r = size // 4
    draw.ellipse([cx - r * 2, cy - r, cx, cy + r], fill=(*color, 220))
    draw.ellipse([cx, cy - r, cx + r * 2, cy + r], fill=(*color, 220))
    draw.polygon([(cx - r * 2, cy), (cx + r * 2, cy), (cx, cy + r * 2)], fill=(*color, 220))
    return img


def draw_checkmark(size: int = 20, color: Tuple[int, ...] = (50, 205, 50)) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    lw = max(2, size // 8)
    draw.line([(size * 0.15, size * 0.55), (size * 0.4, size * 0.8)], fill=(*color, 230), width=lw)
    draw.line([(size * 0.4, size * 0.8), (size * 0.85, size * 0.2)], fill=(*color, 230), width=lw)
    return img


def draw_number(text: str, size: int = 16, color: Tuple[int, ...] = (255, 255, 255)) -> Image.Image:
    img = Image.new("RGBA", (size * len(text), size + 4), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
    except (OSError, IOError):
        font = ImageFont.load_default()
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx or dy:
                draw.text((2 + dx, 2 + dy), text, font=font, fill=(0, 0, 0, 200))
    draw.text((2, 2), text, font=font, fill=(*color, 255))
    return img


def draw_star(size: int = 18, color: Tuple[int, ...] = (255, 215, 0)) -> Image.Image:
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx, cy = size / 2, size / 2
    r_outer = size * 0.45
    r_inner = size * 0.18
    points = []
    for i in range(10):
        angle = math.pi / 2 + i * math.pi / 5
        r = r_outer if i % 2 == 0 else r_inner
        points.append((cx + r * math.cos(angle), cy - r * math.sin(angle)))
    draw.polygon(points, fill=(*color, 230))
    return img


def pollute_avatar(avatar_bgra: np.ndarray) -> np.ndarray:
    """Apply random UI pollution to a clean avatar image."""
    h, w = avatar_bgra.shape[:2]
    if avatar_bgra.shape[2] == 4:
        pil_img = Image.fromarray(cv2.cvtColor(avatar_bgra, cv2.COLOR_BGRA2RGBA))
    else:
        pil_img = Image.fromarray(cv2.cvtColor(avatar_bgra, cv2.COLOR_BGR2RGB)).convert("RGBA")

    if random.random() < 0.80:
        heart_size = random.randint(max(8, w // 6), max(12, w // 4))
        heart_colors = [(255, 105, 180), (255, 80, 80), (255, 150, 200), (255, 50, 100)]
        heart = draw_heart(heart_size, random.choice(heart_colors))
        hx = w - heart_size - random.randint(0, max(1, w // 10))
        hy = h - heart_size - random.randint(0, max(1, h // 10))
        pil_img.paste(heart, (hx, hy), heart)
        if random.random() < 0.7:
            num_text = str(random.randint(1, 99))
            num_size = random.randint(max(8, w // 8), max(10, w // 5))
            num_img = draw_number(num_text, num_size)
            nx = w - num_img.width - random.randint(0, max(1, w // 12))
            ny = h - num_img.height - random.randint(0, max(1, h // 15))
            pil_img.paste(num_img, (nx, ny), num_img)

    if random.random() < 0.50:
        check_size = random.randint(max(8, w // 6), max(12, w // 3))
        check_colors = [(50, 205, 50), (0, 180, 0), (100, 255, 100)]
        check = draw_checkmark(check_size, random.choice(check_colors))
        cx_pos = w - check_size - random.randint(0, max(1, w // 10))
        cy_pos = random.randint(0, max(1, h // 10))
        pil_img.paste(check, (cx_pos, cy_pos), check)

    if random.random() < 0.30:
        star_size = random.randint(max(6, w // 8), max(10, w // 5))
        n_stars = random.randint(1, 3)
        for i in range(n_stars):
            star = draw_star(star_size)
            sx = random.randint(0, max(1, w // 10)) + i * (star_size - 2)
            sy = h - star_size - random.randint(0, max(1, h // 10))
            if sx + star_size < w:
                pil_img.paste(star, (sx, sy), star)

    if random.random() < 0.40:
        arr = np.array(pil_img).astype(np.float32)
        arr[:, :, :3] *= random.uniform(0.85, 1.15)
        arr[:, :, random.randint(0, 2)] += random.uniform(-10, 10)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(arr)

    result = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


def clean_avatar_to_bgr(avatar_bgra: np.ndarray) -> np.ndarray:
    if avatar_bgra.shape[2] == 4:
        alpha = avatar_bgra[:, :, 3:4].astype(np.float32) / 255.0
        bgr = avatar_bgra[:, :, :3].astype(np.float32)
        bg = np.full_like(bgr, 40.0)
        return (bgr * alpha + bg * (1 - alpha)).astype(np.uint8)
    return avatar_bgra[:, :, :3]


def generate_character_samples(
    char_name: str, char_class_id: int, generic_class_id: int,
    png_path: Path, samples_per: int = 6, pollute_ratio: float = 0.8,
) -> List[Tuple[np.ndarray, str]]:
    """Generate synthetic samples for one character.

    Each sample is labeled with BOTH the character-specific class AND the
    generic 角色头像 class so the model learns both simultaneously.
    """
    results = []
    for i in range(samples_per):
        size = random.choice(AVATAR_SIZES)
        img = crop_face_from_wiki(png_path, target_size=size)
        if img is None:
            continue
        img = apply_circular_mask(img)

        if random.random() < pollute_ratio:
            avatar_bgr = pollute_avatar(img)
        else:
            avatar_bgr = clean_avatar_to_bgr(img)

        # Place on background canvas
        bg_size = random.choice([128, 160, 192])
        if size >= bg_size:
            bg_size = size + random.randint(16, 32)

        bg_v = random.randint(20, 80)
        canvas = np.full((bg_size, bg_size, 3), bg_v, dtype=np.uint8)
        if random.random() < 0.3:
            noise = np.random.randint(-15, 15, canvas.shape, dtype=np.int16)
            canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        ah, aw = avatar_bgr.shape[:2]
        max_off = max(1, (bg_size - aw) // 2)
        ox = random.randint(
            max(0, bg_size // 2 - aw // 2 - max_off),
            min(bg_size - aw, bg_size // 2 - aw // 2 + max_off),
        )
        oy = random.randint(
            max(0, bg_size // 2 - ah // 2 - max_off),
            min(bg_size - ah, bg_size // 2 - ah // 2 + max_off),
        )
        canvas[oy:oy + ah, ox:ox + aw] = avatar_bgr

        cx = (ox + aw / 2) / bg_size
        cy = (oy + ah / 2) / bg_size
        bw = aw / bg_size
        bh = ah / bg_size

        # Two labels: generic avatar + character-specific
        label = (
            f"{generic_class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
            f"{char_class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
        )
        results.append((canvas, label))

    return results


# ═══════════════════════════════════════════════════════════════════════
#  Raw Data Collection
# ═══════════════════════════════════════════════════════════════════════

def collect_raw_pairs() -> List[Tuple[Path, Path]]:
    """Collect (image, label) pairs from raw_images datasets."""
    pairs = []
    if not RAW_DIR.exists():
        return pairs
    for ds_dir in sorted(RAW_DIR.iterdir()):
        if not ds_dir.is_dir():
            continue
        for lbl in sorted(ds_dir.glob("*.txt")):
            if lbl.name == "classes.txt" or lbl.stat().st_size == 0:
                continue
            img = None
            for ext in (".jpg", ".png", ".jpeg"):
                candidate = ds_dir / (lbl.stem + ext)
                if candidate.exists():
                    img = candidate
                    break
            if img:
                pairs.append((img, lbl))
    return pairs


# ═══════════════════════════════════════════════════════════════════════
#  Dataset Builder
# ═══════════════════════════════════════════════════════════════════════

def generate_dataset(samples_per_char: int = 6, cafe_ui_count: int = 200,
                     val_ratio: float = 0.15, seed: int = 42):
    """Build the full expanded YOLO dataset."""
    random.seed(seed)
    np.random.seed(seed)

    print("\n=== Expanded YOLO Dataset Builder ===\n")

    # 1. Build class list
    classes, cafe_ui_id, char_map = build_class_list()
    n_chars = len(char_map)
    print(f"  Base classes: {len(BASE_CLASSES)}")
    print(f"  + 收起咖啡厅UI: class {cafe_ui_id}")
    print(f"  + Character avatars: {n_chars} characters (classes {cafe_ui_id + 1}-{cafe_ui_id + n_chars})")
    print(f"  Total classes: {len(classes)}")

    all_images: List[Tuple[np.ndarray, str, str]] = []  # (img, label, name)

    # 2. Generate 收起咖啡厅UI samples
    print(f"\nStep 1: Generating {cafe_ui_count} 收起咖啡厅UI samples...")
    cafe_samples = generate_cafe_ui_samples(cafe_ui_id, count=cafe_ui_count)
    for idx, (img, label) in enumerate(cafe_samples):
        all_images.append((img, label, f"cafe_ui_{idx:04d}"))
    print(f"  Generated {len(cafe_samples)} samples")

    # 3. Generate per-character avatar samples
    print(f"\nStep 2: Generating {samples_per_char} samples per character...")
    generic_id = 9  # 角色头像
    total_chars = 0
    for char_name, char_id in char_map.items():
        png_path = AVATARS_DIR / f"{char_name}.png"
        if not png_path.exists():
            continue
        char_samples = generate_character_samples(
            char_name, char_id, generic_id, png_path,
            samples_per=samples_per_char,
        )
        for idx, (img, label) in enumerate(char_samples):
            all_images.append((img, label, f"char_{char_name}_{idx}"))
        total_chars += 1
    print(f"  Generated {total_chars * samples_per_char} samples for {total_chars} characters")

    # 4. Shuffle and split
    random.shuffle(all_images)
    n_val = max(1, int(len(all_images) * val_ratio))
    val_set = all_images[:n_val]
    train_set = all_images[n_val:]
    print(f"\n  Synthetic: Train={len(train_set)}, Val={n_val}")

    # 5. Write to disk
    print("\nStep 3: Writing dataset...")
    for split, samples in [("train", train_set), ("val", val_set)]:
        img_dir = DATASET_DIR / "images" / split
        lbl_dir = DATASET_DIR / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        # Clean old synthetic files
        for f in img_dir.glob("cafe_ui_*"):
            f.unlink()
        for f in img_dir.glob("char_*"):
            f.unlink()
        for f in lbl_dir.glob("cafe_ui_*"):
            f.unlink()
        for f in lbl_dir.glob("char_*"):
            f.unlink()

        for img, label, name in samples:
            cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(
                str(img_dir / f"{name}.jpg")
            )
            with open(lbl_dir / f"{name}.txt", "w") as f:
                f.write(label + "\n")

    # 6. Merge raw labeled data into train split
    print("\nStep 4: Merging raw labeled data...")
    raw_pairs = collect_raw_pairs()
    if raw_pairs:
        img_dir = DATASET_DIR / "images" / "train"
        lbl_dir = DATASET_DIR / "labels" / "train"
        # Clean old raw copies
        for f in img_dir.glob("raw_*"):
            f.unlink()
        for f in lbl_dir.glob("raw_*"):
            f.unlink()
        for img_src, lbl_src in raw_pairs:
            dst_img = img_dir / f"raw_{img_src.name}"
            dst_lbl = lbl_dir / f"raw_{lbl_src.name}"
            if not dst_img.exists():
                shutil.copy2(img_src, dst_img)
                shutil.copy2(lbl_src, dst_lbl)
        print(f"  Added {len(raw_pairs)} raw labeled images to train")

    # 7. Write data.yaml
    _write_data_yaml(classes)

    total_train = len(train_set) + len(raw_pairs)
    print(f"\n  Dataset ready at: {DATASET_DIR}")
    print(f"  Total: {total_train} train + {n_val} val images, {len(classes)} classes")
    return DATASET_DIR


def _write_data_yaml(classes: List[str]):
    """Write YOLO data.yaml."""
    import yaml
    names = {i: n for i, n in enumerate(classes)}
    data = {
        "path": str(DATASET_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(classes),
        "names": names,
    }
    yaml_path = DATASET_DIR / "data.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
    print(f"  data.yaml → {yaml_path} ({len(classes)} classes)")


# ═══════════════════════════════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════════════════════════════

def train_model(epochs: int = 120, imgsz: int = 256, batch: int = 64, device: str = "0"):
    """Train YOLO on the expanded dataset."""
    yaml_path = DATASET_DIR / "data.yaml"
    if not yaml_path.exists():
        print("ERROR: data.yaml not found. Run generate_dataset() first.")
        return

    # Use existing full.pt as pretrained if available, else base YOLO
    pretrained = PRETRAINED
    if not Path(pretrained).exists():
        pretrained = "yolo11n.pt"
        print(f"  full.pt not found, using {pretrained} as base")

    print(f"\n=== Training Expanded YOLO ===")
    print(f"  Pretrained: {pretrained}")
    print(f"  Epochs: {epochs}, ImgSz: {imgsz}, Batch: {batch}, Device: {device}")

    output_dir = ML_CACHE / "runs" / "expanded"
    production_pt = ML_CACHE / "full.pt"

    from ultralytics import YOLO
    model = YOLO(pretrained)
    model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        batch=batch,
        workers=8,
        project=str(output_dir),
        name="train",
        exist_ok=True,
        patience=30,
        save=True,
        plots=True,
        cache=True,
        amp=True,
        # Augmentation tuned for UI elements + avatars
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        mosaic=0.5,
        mixup=0.0,
        flipud=0.0,
        fliplr=0.0,         # Face orientation matters
        hsv_h=0.01,
        hsv_s=0.2,
        hsv_v=0.2,
    )

    # Copy best.pt to production
    trained_pt = output_dir / "train" / "weights" / "best.pt"
    if trained_pt.exists():
        ML_CACHE.mkdir(parents=True, exist_ok=True)
        # Backup old model
        if production_pt.exists():
            backup = ML_CACHE / "full_backup.pt"
            shutil.copy2(production_pt, backup)
            print(f"  Backed up old model → {backup}")
        shutil.copy2(trained_pt, production_pt)
        print(f"\nDone! New model saved to: {production_pt}")
        # Delete old TensorRT engine (needs re-export)
        engine_path = ML_CACHE / "full.engine"
        if engine_path.exists():
            engine_path.unlink()
            print(f"  Deleted old TensorRT engine (re-export needed)")
    else:
        print(f"\nTraining done. Check {output_dir / 'train'} for results.")


def preview_samples(n: int = 10):
    """Generate and display N preview samples."""
    random.seed(None)
    np.random.seed(None)

    classes, cafe_ui_id, char_map = build_class_list()
    preview_dir = DATASET_DIR / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    # Preview 收起咖啡厅UI
    print(f"Generating 收起咖啡厅UI previews...")
    cafe_samples = generate_cafe_ui_samples(cafe_ui_id, count=min(n, 5))
    for i, (img, label) in enumerate(cafe_samples):
        cv2.imencode(".jpg", img)[1].tofile(str(preview_dir / f"preview_cafe_ui_{i}.jpg"))

    # Preview character avatars
    print(f"Generating character avatar previews...")
    png_files = sorted(AVATARS_DIR.glob("*.png"))
    for i in range(min(n, len(png_files))):
        png_path = random.choice(png_files)
        name = png_path.stem
        char_id = char_map.get(name, 9)
        samples = generate_character_samples(name, char_id, 9, png_path, samples_per=1)
        if samples:
            img, label = samples[0]
            cv2.imencode(".jpg", img)[1].tofile(str(preview_dir / f"preview_char_{name}.jpg"))

    print(f"  Preview images → {preview_dir}")


def main():
    parser = argparse.ArgumentParser(description="Expanded YOLO Training")
    parser.add_argument("--train", action="store_true", help="Generate + train")
    parser.add_argument("--preview", type=int, default=0, help="Preview N samples")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--samples-per-char", type=int, default=6)
    parser.add_argument("--cafe-ui-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.preview > 0:
        preview_samples(args.preview)
        return

    generate_dataset(
        samples_per_char=args.samples_per_char,
        cafe_ui_count=args.cafe_ui_count,
        seed=args.seed,
    )

    if args.train:
        train_model(
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
        )


if __name__ == "__main__":
    main()
