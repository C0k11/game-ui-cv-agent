"""Auto-label headpat bubbles — V3: HSV color detection on cafe-only frames.

Uses HSV filtering to find the distinctive yellow-orange headpat bubble
(Emoticon_Action.png) color signature. Only processes cafe frames verified
by OCR JSON containing "咖啡" in header area.

Usage:
    python scripts/auto_label_headpat_v3.py
"""
import cv2
import numpy as np
import os
import sys
import glob
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO / "data" / "yolo_headpat_dataset"
IMAGES_DIR = OUTPUT_DIR / "images"
LABELS_DIR = OUTPUT_DIR / "labels"

CLASS_ID = 0
CLASS_NAME = "headpat_bubble"

# HSV range for headpat bubble yellow-orange color
# Core: H=15-30 (yellow-orange), S>150 (saturated), V>180 (bright)
HSV_LO = np.array([15, 150, 180])
HSV_HI = np.array([32, 255, 255])

# Minimum blob area (relative to image area) to count as a bubble
MIN_BLOB_AREA_RATIO = 0.0002  # 0.02% of image
MAX_BLOB_AREA_RATIO = 0.015   # 1.5% of image

# Region filter: only detect in cafe play area (exclude UI bars)
REGION_Y_MIN = 0.10  # skip top header
REGION_Y_MAX = 0.88  # skip bottom toolbar
REGION_X_MIN = 0.05
REGION_X_MAX = 0.95

# Aspect ratio filter: bubble is wider than tall (capsule shape)
MIN_ASPECT = 1.2  # w/h minimum
MAX_ASPECT = 5.0  # w/h maximum


def is_cafe_frame_by_json(json_path: str) -> bool:
    """Check OCR JSON for cafe screen indicators."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        for b in d.get("ocr_boxes", []):
            if b.get("conf", 0) < 0.5:
                continue
            text = b.get("text", "")
            if "咖啡" in text and b.get("y2", 1) < 0.15:
                return True
        return False
    except Exception:
        return False


def detect_bubbles_hsv(frame_bgr):
    """Detect headpat bubbles via HSV color filtering.
    
    Returns list of (cx, cy, w, h) in normalized 0-1 coords.
    """
    fh, fw = frame_bgr.shape[:2]
    img_area = fh * fw
    
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LO, HSV_HI)
    
    # Morphological ops to clean noise and merge nearby pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        area_ratio = area / img_area
        
        if area_ratio < MIN_BLOB_AREA_RATIO or area_ratio > MAX_BLOB_AREA_RATIO:
            continue
        
        x, y, bw, bh = cv2.boundingRect(cnt)
        
        # Aspect ratio filter
        if bh == 0:
            continue
        aspect = bw / bh
        if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
            continue
        
        # Normalize
        nx = x / fw
        ny = y / fh
        nw = bw / fw
        nh = bh / fh
        cx = nx + nw / 2
        cy = ny + nh / 2
        
        # Region filter
        if cx < REGION_X_MIN or cx > REGION_X_MAX:
            continue
        if cy < REGION_Y_MIN or cy > REGION_Y_MAX:
            continue
        
        # Add padding (10% each side) for better bounding box
        pad_x = nw * 0.10
        pad_y = nh * 0.10
        nw_padded = nw + pad_x * 2
        nh_padded = nh + pad_y * 2
        
        detections.append((cx, cy, nw_padded, nh_padded))
    
    return detections[:5]  # max 5 per frame


def main():
    print("=== YOLO Headpat Bubble Auto-Labeler V3 (HSV, cafe-only) ===")
    print(f"HSV range: H={HSV_LO[0]}-{HSV_HI[0]} S={HSV_LO[1]}-{HSV_HI[1]} V={HSV_LO[2]}-{HSV_HI[2]}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    # Collect trajectory tick files with JSON
    tick_pairs = []
    for run_dir in sorted(glob.glob(str(REPO / "data" / "trajectories" / "run_*"))):
        for jpg in sorted(glob.glob(os.path.join(run_dir, "tick_*.jpg"))):
            json_path = jpg.replace(".jpg", ".json")
            if os.path.exists(json_path):
                tick_pairs.append((jpg, json_path))

    print(f"Total trajectory ticks with JSON: {len(tick_pairs)}")

    # Phase 1: filter cafe frames
    cafe_frames = []
    for i, (jpg, js) in enumerate(tick_pairs):
        if is_cafe_frame_by_json(js):
            cafe_frames.append(jpg)
        if (i + 1) % 1000 == 0:
            print(f"  [filter {i+1}/{len(tick_pairs)}] cafe frames: {len(cafe_frames)}")
    print(f"Cafe frames found: {len(cafe_frames)}")
    print()

    # Phase 2: HSV detection on cafe frames
    labeled = 0
    positive = 0
    negative = 0
    total_det = 0
    max_neg = len(cafe_frames) // 5  # 20% negative cap

    for i, frame_path in enumerate(cafe_frames):
        img = cv2.imread(frame_path)
        if img is None:
            continue

        detections = detect_bubbles_hsv(img)

        fname = f"cafe_{labeled:05d}.jpg"
        if detections:
            shutil.copy2(frame_path, str(IMAGES_DIR / fname))
            label_path = LABELS_DIR / f"cafe_{labeled:05d}.txt"
            with open(label_path, "w") as f:
                for cx, cy, w, h in detections:
                    f.write(f"{CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            total_det += len(detections)
            positive += 1
            labeled += 1
        elif negative < max_neg:
            shutil.copy2(frame_path, str(IMAGES_DIR / fname))
            label_path = LABELS_DIR / f"cafe_{labeled:05d}.txt"
            label_path.touch()
            negative += 1
            labeled += 1

        if (i + 1) % 50 == 0 or i == len(cafe_frames) - 1:
            print(f"[{i+1}/{len(cafe_frames)}] labeled={labeled} pos={positive} neg={negative} det={total_det}")

    # Write dataset.yaml
    yaml_path = OUTPUT_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {OUTPUT_DIR}\n")
        f.write(f"train: images\n")
        f.write(f"val: images\n")
        f.write(f"nc: 1\n")
        f.write(f"names: ['{CLASS_NAME}']\n")

    print()
    print("=== DONE ===")
    print(f"Labeled: {labeled} (pos={positive}, neg={negative})")
    print(f"Total detections: {total_det}")
    print(f"Dataset: {yaml_path}")


if __name__ == "__main__":
    main()
