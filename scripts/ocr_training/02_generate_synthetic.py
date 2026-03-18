"""Step 2: Generate synthetic training data with CJK fonts + Blue Archive styling.

Creates text-line images with:
- Various CJK fonts (system fonts + any custom fonts in data/fonts/)
- Blue Archive UI styling: white-on-dark, dark-on-light, button text, header text
- Random augmentations: blur, noise, color jitter, slight rotation
- Perfectly labeled (ground truth is known)

Output goes to data/ocr_training/synthetic/ with labels appended to labels.txt

Usage:
    py -3 scripts/ocr_training/02_generate_synthetic.py [--count 50000]
"""
import argparse
import hashlib
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "ocr_training"))
from ba_vocab import get_all_vocab

OUT_DIR = REPO / "data" / "ocr_training"
SYNTH_DIR = OUT_DIR / "synthetic"
FONTS_DIR = REPO / "data" / "fonts"

# ── Font discovery ──
def find_system_fonts() -> list[Path]:
    """Find CJK-capable fonts on the system."""
    candidates = []
    # Windows font directories
    win_fonts = Path("C:/Windows/Fonts")
    if win_fonts.exists():
        for ext in ("*.ttf", "*.ttc", "*.otf"):
            candidates.extend(win_fonts.glob(ext))

    # Filter to fonts likely to have CJK glyphs
    cjk_keywords = [
        "msyh", "msjh", "simsun", "simhei", "mingliu", "kaiu",
        "noto", "cjk", "yahei", "jhenghei", "heiti", "songti",
        "fangsong", "source", "sarasa", "wenquanyi", "arial",
        "meiryo", "malgun", "batang", "gulim",
    ]
    cjk_fonts = []
    for f in candidates:
        name_lower = f.stem.lower()
        if any(kw in name_lower for kw in cjk_keywords):
            cjk_fonts.append(f)

    # Also add custom fonts from data/fonts/
    if FONTS_DIR.exists():
        for ext in ("*.ttf", "*.ttc", "*.otf"):
            cjk_fonts.extend(FONTS_DIR.glob(ext))

    return cjk_fonts


# ── Blue Archive color palettes ──
# (text_color, bg_color) tuples
BA_PALETTES = [
    # White text on dark blue (battle HUD, headers)
    ((255, 255, 255), (30, 50, 90)),
    ((255, 255, 255), (40, 40, 60)),
    # Dark text on white/light (menus, lists)
    ((40, 40, 50), (245, 245, 250)),
    ((30, 30, 40), (230, 235, 245)),
    # Yellow/gold text on dark (rewards, notifications)
    ((255, 215, 80), (30, 30, 50)),
    ((255, 200, 50), (50, 40, 60)),
    # Blue text on white (buttons, links)
    ((50, 100, 220), (245, 245, 255)),
    ((60, 120, 255), (240, 240, 250)),
    # White text on blue button
    ((255, 255, 255), (60, 130, 255)),
    ((255, 255, 255), (80, 150, 240)),
    # Red/pink text (warnings, limited time)
    ((220, 50, 50), (245, 245, 250)),
    ((255, 80, 80), (40, 40, 50)),
    # Green text (confirmed, success)
    ((50, 180, 80), (245, 245, 250)),
    # Gray text on white (disabled, secondary)
    ((140, 140, 150), (245, 245, 250)),
]


def render_text_image(
    text: str,
    font: ImageFont.FreeTypeFont,
    text_color: tuple,
    bg_color: tuple,
    target_h: int = 48,
    max_w: int = 640,
) -> np.ndarray:
    """Render a single text line as an image with Blue Archive styling."""
    # Measure text
    dummy = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    if tw <= 0 or th <= 0:
        return None

    # Scale to target height
    scale = target_h / max(th, 1)
    font_size = max(12, int(font.size * scale))
    try:
        font = font.font_variant(size=font_size)
    except Exception:
        return None

    # Re-measure
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    # Create image with padding
    pad_x = random.randint(4, 16)
    pad_y = random.randint(2, 8)
    img_w = min(tw + pad_x * 2, max_w)
    img_h = th + pad_y * 2

    img = Image.new("RGB", (img_w, img_h), bg_color)
    draw = ImageDraw.Draw(img)

    # Center text
    x = (img_w - tw) // 2
    y = (img_h - th) // 2 - bbox[1]

    # Optional: slight shadow for game-like text
    if random.random() < 0.3:
        shadow_color = tuple(max(0, c - 60) for c in bg_color)
        draw.text((x + 1, y + 1), text, fill=shadow_color, font=font)

    draw.text((x, y), text, fill=text_color, font=font)

    return np.array(img)


def augment_image(img: np.ndarray) -> np.ndarray:
    """Apply random augmentations to simulate real game captures."""
    h, w = img.shape[:2]

    # Random slight rotation (-2 to +2 degrees)
    if random.random() < 0.3:
        angle = random.uniform(-2, 2)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # Random blur (simulates capture quality)
    if random.random() < 0.2:
        ksize = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    # Random noise
    if random.random() < 0.2:
        noise = np.random.normal(0, random.uniform(3, 10), img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Random brightness/contrast
    if random.random() < 0.3:
        alpha = random.uniform(0.85, 1.15)  # contrast
        beta = random.uniform(-15, 15)       # brightness
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # Random JPEG compression artifact
    if random.random() < 0.3:
        quality = random.randint(50, 90)
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    # Random resize (simulate different capture resolutions)
    if random.random() < 0.2:
        scale = random.uniform(0.7, 1.3)
        new_w = max(8, int(w * scale))
        new_h = max(8, int(h * scale))
        img = cv2.resize(img, (new_w, new_h))
        img = cv2.resize(img, (w, h))

    return img


def generate_text_variants(text: str) -> list[str]:
    """Generate additional text variants for a base string."""
    variants = [text]

    # Add with common suffixes/prefixes
    if len(text) >= 2 and not text.isascii():
        # Number suffix (e.g., "區域29", "第3章")
        variants.append(text + str(random.randint(1, 99)))

    return variants


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic OCR training data")
    parser.add_argument("--count", type=int, default=50000,
                        help="Number of synthetic images to generate (default: 50000)")
    parser.add_argument("--target-h", type=int, default=48,
                        help="Target crop height in pixels (default: 48)")
    args = parser.parse_args()

    SYNTH_DIR.mkdir(parents=True, exist_ok=True)

    # Find fonts
    fonts = find_system_fonts()
    if not fonts:
        print("[ERROR] No CJK fonts found! Install Noto Sans CJK or place .ttf files in data/fonts/")
        sys.exit(1)
    print(f"Found {len(fonts)} CJK fonts:")
    for f in fonts[:10]:
        print(f"  {f.name}")
    if len(fonts) > 10:
        print(f"  ... and {len(fonts) - 10} more")

    # Load fonts at base sizes
    font_objects = []
    base_sizes = [24, 28, 32, 36, 40, 48]
    for font_path in fonts:
        for size in base_sizes:
            try:
                f = ImageFont.truetype(str(font_path), size)
                # Test that it can render CJK
                test_img = Image.new("RGB", (100, 100))
                test_draw = ImageDraw.Draw(test_img)
                test_draw.text((0, 0), "確認", font=f)
                font_objects.append(f)
            except Exception:
                continue
    print(f"Loaded {len(font_objects)} font variants")

    if not font_objects:
        print("[ERROR] No usable CJK font objects. Check font files.")
        sys.exit(1)

    # Get vocabulary
    vocab = get_all_vocab()
    print(f"Vocabulary: {len(vocab)} strings")

    # Expand with variants
    all_texts = []
    for text in vocab:
        all_texts.extend(generate_text_variants(text))
    print(f"Text variants: {len(all_texts)} strings")

    # Generate
    label_lines = []
    generated = 0

    print(f"\nGenerating {args.count} synthetic images...")
    while generated < args.count:
        text = random.choice(all_texts)
        font = random.choice(font_objects)
        text_color, bg_color = random.choice(BA_PALETTES)
        target_h = args.target_h + random.randint(-8, 8)

        img = render_text_image(text, font, text_color, bg_color, target_h)
        if img is None:
            continue

        img = augment_image(img)

        # Save
        text_hash = hashlib.md5(f"{text}_{generated}".encode()).hexdigest()[:10]
        fname = f"syn_{generated:06d}_{text_hash}.png"
        crop_path = SYNTH_DIR / fname

        success, buf = cv2.imencode(".png", img)
        if not success:
            continue
        buf.tofile(str(crop_path))

        rel_path = f"synthetic/{fname}"
        label_lines.append(f"{rel_path}\t{text}")
        generated += 1

        if generated % 5000 == 0:
            print(f"  ... {generated}/{args.count}")

    # Append to main labels file
    labels_path = OUT_DIR / "labels.txt"
    synth_labels_path = OUT_DIR / "labels_synthetic.txt"

    # Write synthetic-only labels
    synth_labels_path.write_text("\n".join(label_lines), encoding="utf-8")
    print(f"\nSynthetic labels: {synth_labels_path}")

    # Merge with trajectory labels if they exist
    existing = ""
    if labels_path.exists():
        existing = labels_path.read_text("utf-8").rstrip("\n")

    merged = existing + "\n" + "\n".join(label_lines) if existing else "\n".join(label_lines)
    labels_path.write_text(merged, encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"Synthetic generation complete:")
    print(f"  Generated: {generated} images")
    print(f"  Output:    {SYNTH_DIR}")
    print(f"  Labels:    {synth_labels_path}")
    print(f"  Merged into: {labels_path}")


if __name__ == "__main__":
    main()
