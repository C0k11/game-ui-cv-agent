"""Step 1: Extract text region crops from ALL trajectory screenshots.

Reads every tick_XXXX.json + tick_XXXX.jpg pair, crops text regions using
the OCR bounding boxes, applies auto-corrections from ba_vocab, and writes:
  data/ocr_training/crops/       — cropped text images (PNG)
  data/ocr_training/labels.txt   — PaddleOCR format: image_path\tlabel
  data/ocr_training/labels_raw.txt — before corrections, for auditing

Usage:
    py -3 scripts/ocr_training/01_extract_crops.py [--min-conf 0.5] [--max-crops 200000]
"""
import argparse
import json
import sys
import hashlib
from pathlib import Path
from collections import Counter

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "ocr_training"))
from ba_vocab import CORRECTIONS, get_all_vocab

TRAJ_DIR = REPO / "data" / "trajectories"
OUT_DIR = REPO / "data" / "ocr_training"
CROP_DIR = OUT_DIR / "crops"

# Minimum crop dimensions (pixels) — skip tiny/degenerate boxes
MIN_W, MIN_H = 8, 8
# Padding around crop (fraction of box size) to capture context
PAD_FRAC = 0.08


def apply_corrections(text: str) -> str:
    """Apply known OCR misread corrections."""
    if text in CORRECTIONS:
        return CORRECTIONS[text]
    # Try partial corrections (longest match first)
    for wrong, right in sorted(CORRECTIONS.items(), key=lambda x: -len(x[0])):
        if wrong in text:
            text = text.replace(wrong, right)
    return text


def crop_text_region(img: np.ndarray, box: dict, pad_frac: float = PAD_FRAC) -> np.ndarray:
    """Crop a text region from the image using normalized coordinates."""
    h, w = img.shape[:2]
    x1 = int(box["x1"] * w)
    y1 = int(box["y1"] * h)
    x2 = int(box["x2"] * w)
    y2 = int(box["y2"] * h)

    # Add padding
    bw, bh = x2 - x1, y2 - y1
    px = int(bw * pad_frac)
    py = int(bh * pad_frac)
    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(w, x2 + px)
    y2 = min(h, y2 + py)

    crop = img[y1:y2, x1:x2]
    return crop


def make_crop_filename(run_name: str, tick: int, box_idx: int, text: str) -> str:
    """Generate a unique filename for a crop."""
    # Use hash to avoid filesystem issues with CJK filenames
    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    return f"{run_name}_t{tick:04d}_b{box_idx}_{text_hash}.png"


def main():
    parser = argparse.ArgumentParser(description="Extract OCR training crops from trajectories")
    parser.add_argument("--min-conf", type=float, default=0.3,
                        help="Minimum OCR confidence to include (default: 0.3)")
    parser.add_argument("--max-crops", type=int, default=300000,
                        help="Maximum number of crops to extract (default: 300000)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip if output directory already has crops")
    args = parser.parse_args()

    CROP_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.skip_existing and any(CROP_DIR.glob("*.png")):
        existing = len(list(CROP_DIR.glob("*.png")))
        print(f"[Skip] {existing} crops already exist. Use --no-skip-existing to regenerate.")
        return

    runs = sorted(TRAJ_DIR.glob("run_*"))
    print(f"Found {len(runs)} trajectory runs")

    vocab_set = set(get_all_vocab())
    label_lines = []
    label_raw_lines = []
    stats = Counter()
    total_crops = 0

    for run_dir in runs:
        run_name = run_dir.name
        tick_files = sorted(run_dir.glob("tick_*.json"))

        for tick_file in tick_files:
            if total_crops >= args.max_crops:
                break

            # Find corresponding image
            img_file = tick_file.with_suffix(".jpg")
            if not img_file.exists():
                img_file = tick_file.with_suffix(".png")
            if not img_file.exists():
                stats["missing_img"] += 1
                continue

            try:
                tick_data = json.loads(tick_file.read_text("utf-8"))
            except Exception:
                stats["bad_json"] += 1
                continue

            ocr_boxes = tick_data.get("ocr_boxes", [])
            if not ocr_boxes:
                stats["no_boxes"] += 1
                continue

            # Lazy-load image only when we have boxes to crop
            img = cv2.imdecode(np.fromfile(str(img_file), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                stats["bad_img"] += 1
                continue

            tick_num = tick_data.get("tick", 0)

            for box_idx, box in enumerate(ocr_boxes):
                if total_crops >= args.max_crops:
                    break

                text = box.get("text", "").strip()
                conf = box.get("conf", 0.0)

                if not text or conf < args.min_conf:
                    stats["low_conf"] += 1
                    continue

                # Crop the text region
                crop = crop_text_region(img, box)
                ch, cw = crop.shape[:2]
                if cw < MIN_W or ch < MIN_H:
                    stats["too_small"] += 1
                    continue

                # Apply corrections
                corrected = apply_corrections(text)

                # Generate filename and save
                fname = make_crop_filename(run_name, tick_num, box_idx, text)
                crop_path = CROP_DIR / fname

                # Encode to PNG in memory, then write (handles CJK paths)
                success, buf = cv2.imencode(".png", crop)
                if not success:
                    stats["encode_fail"] += 1
                    continue
                buf.tofile(str(crop_path))

                # PaddleOCR label format: relative_path\tlabel
                rel_path = f"crops/{fname}"
                label_lines.append(f"{rel_path}\t{corrected}")
                label_raw_lines.append(f"{rel_path}\t{text}")

                if corrected != text:
                    stats["corrected"] += 1
                if text in vocab_set or corrected in vocab_set:
                    stats["vocab_match"] += 1

                total_crops += 1
                stats["extracted"] += 1

        if total_crops >= args.max_crops:
            print(f"[Limit] Reached max crops ({args.max_crops})")
            break

        if total_crops % 5000 == 0 and total_crops > 0:
            print(f"  ... {total_crops} crops from {run_name}")

    # Write label files
    labels_path = OUT_DIR / "labels.txt"
    labels_raw_path = OUT_DIR / "labels_raw.txt"
    labels_path.write_text("\n".join(label_lines), encoding="utf-8")
    labels_raw_path.write_text("\n".join(label_raw_lines), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"Extraction complete:")
    print(f"  Total crops:   {total_crops}")
    print(f"  Corrected:     {stats['corrected']}")
    print(f"  Vocab matches: {stats['vocab_match']}")
    print(f"  Low confidence: {stats['low_conf']}")
    print(f"  Too small:     {stats['too_small']}")
    print(f"  Missing image: {stats['missing_img']}")
    print(f"  Bad JSON:      {stats['bad_json']}")
    print(f"  Bad image:     {stats['bad_img']}")
    print(f"\nOutput:")
    print(f"  Crops:      {CROP_DIR}")
    print(f"  Labels:     {labels_path}")
    print(f"  Raw labels: {labels_raw_path}")


if __name__ == "__main__":
    main()
