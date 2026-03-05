"""Poison Data Augmentation for YOLO Avatar Detection.

Goal: Make YOLO robust to UI overlays on student avatars — hearts, checkmarks,
intimacy numbers — so it detects 角色头像 regardless of these visual pollutants.

Pipeline:
  Step 1: Crop clean base avatars from wiki portraits (data/captures/角色头像/)
  Step 2: Generate poison overlays programmatically (hearts, checkmarks, numbers)
  Step 3: Apply random poison combinations to each avatar
  Step 4: Build YOLO dataset (80% polluted, 20% clean) and train

Usage:
  python scripts/augment_avatar_dataset.py              # Generate dataset only
  python scripts/augment_avatar_dataset.py --train       # Generate + train
  python scripts/augment_avatar_dataset.py --preview 10  # Preview N samples
"""

import argparse
import math
import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
AVATARS_DIR = PROJECT_ROOT / "data" / "captures" / "角色头像"
RAW_DIR = PROJECT_ROOT / "data" / "raw_images"
ML_CACHE = Path(r"D:\Project\ml_cache\models\yolo")
DATASET_DIR = ML_CACHE / "dataset" / "avatar_augmented"
PRETRAINED = str(PROJECT_ROOT / "yolo26n.pt")

# ── Avatar class ID in the YOLO model (角色头像 = class 9) ──
AVATAR_CLASS_ID = 9

# ── Target output sizes (matching real roster thumbnail sizes) ──
AVATAR_SIZES = [64, 80, 96, 112, 128]


# ═══════════════════════════════════════════════════════════════════════
#  Step 1: Crop clean base avatars from wiki portraits
# ═══════════════════════════════════════════════════════════════════════

def crop_face_from_wiki(img_path: Path, target_size: int = 96) -> np.ndarray:
    """Crop face area from wiki portrait (456×404 RGBA) and resize to square.

    Wiki portraits are bust shots; the face is in the upper-center region.
    Crop to top 65%, center 75%, make square, then resize.
    """
    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    h, w = img.shape[:2]

    # Crop to face area
    crop_top = 0
    crop_bot = int(h * 0.65)
    crop_left = int(w * 0.125)
    crop_right = int(w * 0.875)
    img = img[crop_top:crop_bot, crop_left:crop_right]

    # Make square
    ch, cw = img.shape[:2]
    if cw > ch:
        off = (cw - ch) // 2
        img = img[:, off:off + ch]
    elif ch > cw:
        off = (ch - cw) // 2
        img = img[off:off + cw, :]

    # Resize
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
        img[~circle, 3] = 0  # Set alpha to 0 outside circle
    else:
        img[~circle] = [40, 40, 40]  # Dark background outside circle

    return img


def load_clean_avatars(max_per_size: int = None) -> List[Tuple[str, np.ndarray]]:
    """Load and crop all wiki portraits into clean avatar thumbnails."""
    avatars = []
    png_files = sorted(AVATARS_DIR.glob("*.png"))
    for png_path in png_files:
        name = png_path.stem
        size = random.choice(AVATAR_SIZES)
        img = crop_face_from_wiki(png_path, target_size=size)
        if img is not None:
            img = apply_circular_mask(img)
            avatars.append((name, img))
            if max_per_size and len(avatars) >= max_per_size:
                break
    return avatars


# ═══════════════════════════════════════════════════════════════════════
#  Step 2: Generate poison overlays programmatically
# ═══════════════════════════════════════════════════════════════════════

def draw_heart(size: int = 24, color: Tuple[int, int, int] = (255, 105, 180)) -> Image.Image:
    """Draw a heart shape on a transparent background using PIL."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Heart shape using polygon + circles
    cx, cy = size // 2, size // 2
    r = size // 4
    # Two circles for the top bumps
    draw.ellipse([cx - r * 2, cy - r, cx, cy + r], fill=(*color, 220))
    draw.ellipse([cx, cy - r, cx + r * 2, cy + r], fill=(*color, 220))
    # Triangle for the bottom point
    draw.polygon([
        (cx - r * 2, cy),
        (cx + r * 2, cy),
        (cx, cy + r * 2)
    ], fill=(*color, 220))
    return img


def draw_checkmark(size: int = 20, color: Tuple[int, int, int] = (50, 205, 50)) -> Image.Image:
    """Draw a green checkmark on a transparent background."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Checkmark path (thick lines)
    lw = max(2, size // 8)
    # Short leg: from bottom-left to center-bottom
    draw.line([(size * 0.15, size * 0.55), (size * 0.4, size * 0.8)], fill=(*color, 230), width=lw)
    # Long leg: from center-bottom to top-right
    draw.line([(size * 0.4, size * 0.8), (size * 0.85, size * 0.2)], fill=(*color, 230), width=lw)
    return img


def draw_number(text: str, size: int = 16,
                color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """Draw a number on a transparent background with a dark outline."""
    img = Image.new("RGBA", (size * len(text), size + 4), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Try to use a built-in font; fall back to default
    try:
        font = ImageFont.truetype("arial.ttf", size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
        except (OSError, IOError):
            font = ImageFont.load_default()
    # Dark outline
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx or dy:
                draw.text((2 + dx, 2 + dy), text, font=font, fill=(0, 0, 0, 200))
    # White text
    draw.text((2, 2), text, font=font, fill=(*color, 255))
    return img


def draw_star(size: int = 18, color: Tuple[int, int, int] = (255, 215, 0)) -> Image.Image:
    """Draw a gold star on transparent background."""
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


# ═══════════════════════════════════════════════════════════════════════
#  Step 3: Pollute avatars with random overlays
# ═══════════════════════════════════════════════════════════════════════

def pollute_avatar(avatar_bgra: np.ndarray) -> np.ndarray:
    """Apply random UI pollution to a clean avatar image.

    Possible overlays:
    - Bottom-right: heart icon + intimacy number (80% chance)
    - Top-right: green checkmark (50% chance)
    - Bottom-left: star rating (30% chance)
    - Color jitter: slight brightness/contrast variation (40% chance)
    """
    h, w = avatar_bgra.shape[:2]

    # Convert to PIL for overlay compositing
    if avatar_bgra.shape[2] == 4:
        pil_img = Image.fromarray(cv2.cvtColor(avatar_bgra, cv2.COLOR_BGRA2RGBA))
    else:
        pil_img = Image.fromarray(cv2.cvtColor(avatar_bgra, cv2.COLOR_BGR2RGB)).convert("RGBA")

    # ── Heart + intimacy number (bottom-right, 80% chance) ──
    if random.random() < 0.80:
        heart_size = random.randint(max(8, w // 6), max(12, w // 4))
        heart_colors = [
            (255, 105, 180),  # Pink
            (255, 80, 80),    # Red
            (255, 150, 200),  # Light pink
            (255, 50, 100),   # Deep pink
        ]
        heart = draw_heart(heart_size, random.choice(heart_colors))
        # Random slight position jitter
        hx = w - heart_size - random.randint(0, max(1, w // 10))
        hy = h - heart_size - random.randint(0, max(1, h // 10))
        pil_img.paste(heart, (hx, hy), heart)

        # Intimacy number (next to or overlapping heart)
        if random.random() < 0.7:
            num_text = str(random.randint(1, 99))
            num_size = random.randint(max(8, w // 8), max(10, w // 5))
            num_img = draw_number(num_text, num_size)
            nx = w - num_img.width - random.randint(0, max(1, w // 12))
            ny = h - num_img.height - random.randint(0, max(1, h // 15))
            pil_img.paste(num_img, (nx, ny), num_img)

    # ── Checkmark (top-right, 50% chance) ──
    if random.random() < 0.50:
        check_size = random.randint(max(8, w // 6), max(12, w // 3))
        check_colors = [
            (50, 205, 50),   # Lime green
            (0, 180, 0),     # Green
            (100, 255, 100), # Light green
        ]
        check = draw_checkmark(check_size, random.choice(check_colors))
        cx_pos = w - check_size - random.randint(0, max(1, w // 10))
        cy_pos = random.randint(0, max(1, h // 10))
        pil_img.paste(check, (cx_pos, cy_pos), check)

    # ── Star rating (bottom-left, 30% chance) ──
    if random.random() < 0.30:
        star_size = random.randint(max(6, w // 8), max(10, w // 5))
        n_stars = random.randint(1, 3)
        for i in range(n_stars):
            star = draw_star(star_size)
            sx = random.randint(0, max(1, w // 10)) + i * (star_size - 2)
            sy = h - star_size - random.randint(0, max(1, h // 10))
            if sx + star_size < w:
                pil_img.paste(star, (sx, sy), star)

    # ── Color jitter (40% chance) ──
    if random.random() < 0.40:
        arr = np.array(pil_img).astype(np.float32)
        # Brightness
        arr[:, :, :3] *= random.uniform(0.85, 1.15)
        # Slight color shift
        arr[:, :, random.randint(0, 2)] += random.uniform(-10, 10)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(arr)

    # Convert back to BGR (no alpha for YOLO training)
    result = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)


def clean_avatar_to_bgr(avatar_bgra: np.ndarray) -> np.ndarray:
    """Convert clean avatar (BGRA) to BGR for YOLO training."""
    if avatar_bgra.shape[2] == 4:
        # Composite onto gray background
        alpha = avatar_bgra[:, :, 3:4].astype(np.float32) / 255.0
        bgr = avatar_bgra[:, :, :3].astype(np.float32)
        bg = np.full_like(bgr, 40.0)  # Dark gray background
        result = (bgr * alpha + bg * (1 - alpha)).astype(np.uint8)
        return result
    return avatar_bgra[:, :, :3]


# ═══════════════════════════════════════════════════════════════════════
#  Step 4: Build YOLO dataset and train
# ═══════════════════════════════════════════════════════════════════════

def place_avatar_on_background(avatar_bgr: np.ndarray, bg_size: int = 128) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """Place a single avatar on a random background, return image + normalized bbox.

    Returns:
        (image, (cx, cy, w, h)) all in normalized [0,1] coordinates.
    """
    ah, aw = avatar_bgr.shape[:2]

    # Random background color (dark grays to simulate game UI)
    bg_r = random.randint(20, 80)
    bg_g = random.randint(20, 80)
    bg_b = random.randint(20, 80)
    canvas = np.full((bg_size, bg_size, 3), (bg_b, bg_g, bg_r), dtype=np.uint8)

    # Random noise on background
    if random.random() < 0.3:
        noise = np.random.randint(-15, 15, canvas.shape, dtype=np.int16)
        canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Place avatar at random position (can be slightly off-center)
    max_offset = max(1, (bg_size - aw) // 2)
    ox = random.randint(max(0, bg_size // 2 - aw // 2 - max_offset),
                        min(bg_size - aw, bg_size // 2 - aw // 2 + max_offset))
    oy = random.randint(max(0, bg_size // 2 - ah // 2 - max_offset),
                        min(bg_size - ah, bg_size // 2 - ah // 2 + max_offset))

    canvas[oy:oy + ah, ox:ox + aw] = avatar_bgr

    # Normalized bounding box
    cx = (ox + aw / 2) / bg_size
    cy = (oy + ah / 2) / bg_size
    bw = aw / bg_size
    bh = ah / bg_size

    return canvas, (cx, cy, bw, bh)


def generate_dataset(pollute_ratio: float = 0.8, samples_per_avatar: int = 5,
                     val_ratio: float = 0.2, seed: int = 42):
    """Generate the full augmented YOLO dataset.

    Args:
        pollute_ratio: fraction of samples that get poison overlays (0.8 = 80%)
        samples_per_avatar: number of augmented samples per source avatar
        val_ratio: fraction of samples for validation
        seed: random seed
    """
    random.seed(seed)
    np.random.seed(seed)

    print("\n=== Avatar Poison Augmentation Pipeline ===\n")

    # Step 1: Load clean avatars
    print("Step 1: Loading clean avatars...")
    png_files = sorted(AVATARS_DIR.glob("*.png"))
    print(f"  Found {len(png_files)} avatar PNGs")

    # Step 2: Generate samples
    print(f"Step 2: Generating {samples_per_avatar} samples per avatar "
          f"({pollute_ratio:.0%} polluted, {1 - pollute_ratio:.0%} clean)...")

    all_samples = []  # (image_bgr, bbox, name)

    for png_path in png_files:
        name = png_path.stem
        for i in range(samples_per_avatar):
            # Random size for this sample
            size = random.choice(AVATAR_SIZES)
            img = crop_face_from_wiki(png_path, target_size=size)
            if img is None:
                continue
            img = apply_circular_mask(img)

            # Decide clean vs polluted
            if random.random() < pollute_ratio:
                avatar_bgr = pollute_avatar(img)
            else:
                avatar_bgr = clean_avatar_to_bgr(img)

            # Place on background
            bg_size = random.choice([96, 128])
            if size > bg_size:
                bg_size = size + 16
            canvas, bbox = place_avatar_on_background(avatar_bgr, bg_size)
            all_samples.append((canvas, bbox, f"{name}_{i}"))

    print(f"  Generated {len(all_samples)} total samples")

    # Shuffle and split
    random.shuffle(all_samples)
    n_val = max(1, int(len(all_samples) * val_ratio))
    val_samples = all_samples[:n_val]
    train_samples = all_samples[n_val:]
    print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}")

    # Step 3: Write to disk
    print("Step 3: Writing YOLO dataset...")

    for split, samples in [("train", train_samples), ("val", val_samples)]:
        img_dir = DATASET_DIR / "images" / split
        lbl_dir = DATASET_DIR / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        # Clean old files
        for f in img_dir.glob("*"):
            f.unlink()
        for f in lbl_dir.glob("*"):
            f.unlink()

        for canvas, bbox, name in samples:
            img_path = img_dir / f"{name}.jpg"
            lbl_path = lbl_dir / f"{name}.txt"

            cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(str(img_path))

            cx, cy, bw, bh = bbox
            with open(lbl_path, "w") as f:
                f.write(f"{AVATAR_CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        print(f"  {split}: {len(samples)} images → {img_dir}")

    # Step 4: Also include existing labeled data from raw_images
    print("Step 4: Merging with existing raw_images labels...")
    existing_pairs = _collect_existing_pairs()
    if existing_pairs:
        for split_name in ["train", "val"]:
            img_dir = DATASET_DIR / "images" / split_name
            lbl_dir = DATASET_DIR / "labels" / split_name
            # Add existing data to train split only (val stays purely synthetic)
            if split_name == "train":
                for img_src, lbl_src in existing_pairs:
                    dst_img = img_dir / f"raw_{img_src.name}"
                    dst_lbl = lbl_dir / f"raw_{img_src.stem}.txt"
                    if not dst_img.exists():
                        shutil.copy2(img_src, dst_img)
                        shutil.copy2(lbl_src, dst_lbl)
        print(f"  Added {len(existing_pairs)} existing labeled images to train set")

    # Write data.yaml
    _write_data_yaml()
    print(f"\n  Dataset ready at: {DATASET_DIR}")

    return DATASET_DIR


def _collect_existing_pairs() -> List[Tuple[Path, Path]]:
    """Collect (image, label) pairs from raw_images datasets."""
    pairs = []
    for ds_dir in sorted(RAW_DIR.iterdir()):
        if not ds_dir.is_dir():
            continue
        classes_file = ds_dir / "classes.txt"
        if not classes_file.exists():
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


def _read_all_classes() -> List[str]:
    """Read classes from first available dataset."""
    for ds_dir in sorted(RAW_DIR.iterdir()):
        cf = ds_dir / "classes.txt"
        if cf.exists():
            lines = cf.read_text("utf-8").strip().splitlines()
            return [l.strip() for l in lines if l.strip()]
    return []


def _write_data_yaml():
    """Write YOLO data.yaml for the augmented dataset."""
    import yaml

    classes = _read_all_classes()
    if not classes:
        # Fallback: minimal class list
        classes = ["角色头像"]

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


def train_model(epochs: int = 100, imgsz: int = 128, batch: int = 32, device: str = "0"):
    """Train YOLO on the augmented avatar dataset."""
    yaml_path = DATASET_DIR / "data.yaml"
    if not yaml_path.exists():
        print("ERROR: data.yaml not found. Run generate_dataset() first.")
        return

    print(f"\n=== Training YOLO Avatar Detector ===")
    print(f"  Pretrained: {PRETRAINED}")
    print(f"  Epochs: {epochs}, ImgSz: {imgsz}, Batch: {batch}, Device: {device}")
    print(f"  Dataset: {yaml_path}")

    output_dir = ML_CACHE / "runs" / "avatar_augmented"
    production_pt = ML_CACHE / "avatar_augmented.pt"

    from ultralytics import YOLO
    model = YOLO(PRETRAINED)
    model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        batch=batch,
        workers=4,
        project=str(output_dir),
        name="train",
        exist_ok=True,
        patience=25,
        save=True,
        plots=True,
        cache=True,
        amp=True,
        # Reduced augmentation for UI elements (position is usually fixed)
        degrees=5.0,        # Slight rotation only
        translate=0.1,
        scale=0.3,
        mosaic=0.0,         # Disable mosaic (avatars are small, mosaic breaks context)
        mixup=0.0,          # Disable mixup
        flipud=0.0,         # No vertical flip
        fliplr=0.0,         # No horizontal flip (face orientation matters)
        hsv_h=0.01,         # Minimal hue shift
        hsv_s=0.2,
        hsv_v=0.2,
    )

    # Copy best.pt to production
    trained_pt = output_dir / "train" / "weights" / "best.pt"
    if trained_pt.exists():
        ML_CACHE.mkdir(parents=True, exist_ok=True)
        shutil.copy2(trained_pt, production_pt)
        print(f"\nDone! Model saved to: {production_pt}")
    else:
        print(f"\nTraining done. Check {output_dir / 'train'} for results.")


def preview_samples(n: int = 10):
    """Generate and display N sample images for visual inspection."""
    print(f"\nGenerating {n} preview samples...")
    preview_dir = DATASET_DIR / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    png_files = sorted(AVATARS_DIR.glob("*.png"))
    if not png_files:
        print("ERROR: No avatar PNGs found")
        return

    for i in range(n):
        png_path = random.choice(png_files)
        name = png_path.stem
        size = random.choice(AVATAR_SIZES)
        img = crop_face_from_wiki(png_path, target_size=size)
        if img is None:
            continue
        img = apply_circular_mask(img)

        # Save clean version
        clean = clean_avatar_to_bgr(img)
        cv2.imencode(".jpg", clean)[1].tofile(str(preview_dir / f"{i:03d}_{name}_clean.jpg"))

        # Save polluted version
        polluted = pollute_avatar(img)
        cv2.imencode(".jpg", polluted)[1].tofile(str(preview_dir / f"{i:03d}_{name}_polluted.jpg"))

        # Save on-background version
        bg_size = random.choice([96, 128])
        if size > bg_size:
            bg_size = size + 16
        canvas, bbox = place_avatar_on_background(polluted, bg_size)
        cv2.imencode(".jpg", canvas)[1].tofile(str(preview_dir / f"{i:03d}_{name}_canvas.jpg"))

    print(f"  Preview images saved to: {preview_dir}")


def main():
    parser = argparse.ArgumentParser(description="Avatar Poison Data Augmentation")
    parser.add_argument("--train", action="store_true", help="Generate dataset AND train")
    parser.add_argument("--preview", type=int, default=0, help="Preview N samples")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=128)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--samples", type=int, default=5,
                        help="Augmented samples per source avatar")
    parser.add_argument("--pollute-ratio", type=float, default=0.8,
                        help="Fraction of polluted vs clean samples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.preview > 0:
        preview_samples(args.preview)
        return

    generate_dataset(
        pollute_ratio=args.pollute_ratio,
        samples_per_avatar=args.samples,
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
