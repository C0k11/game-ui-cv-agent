"""Auto-label headpat bubbles — V2: only cafe frames verified by OCR JSON.

Scans trajectory runs, checks each tick's JSON for "咖啡廳" in OCR text
(confirming it's a cafe screen), THEN runs template matching for bubbles.

Usage:
    python scripts/auto_label_headpat_v2.py
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
TEMPLATE_PATH = REPO / "data" / "captures" / "Emoticon_Action.png"
OUTPUT_DIR = REPO / "data" / "yolo_headpat_dataset"
IMAGES_DIR = OUTPUT_DIR / "images"
LABELS_DIR = OUTPUT_DIR / "labels"

CLASS_ID = 0
CLASS_NAME = "headpat_bubble"

WORK_W = 960
MATCH_THRESHOLD = 0.78
NMS_IOU_THRESHOLD = 0.3
SCALES = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
REGION = (0.05, 0.15, 0.95, 0.85)


def load_template():
    tmpl = cv2.imread(str(TEMPLATE_PATH), cv2.IMREAD_UNCHANGED)
    if tmpl is None:
        print(f"ERROR: template not found at {TEMPLATE_PATH}")
        sys.exit(1)
    if tmpl.shape[2] == 4:
        tmpl = cv2.cvtColor(tmpl, cv2.COLOR_BGRA2BGR)
    return tmpl


def is_cafe_frame_by_json(json_path: str) -> bool:
    """Check OCR JSON for cafe screen indicators."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        for b in d.get("ocr_boxes", []):
            if b.get("conf", 0) < 0.5:
                continue
            text = b.get("text", "")
            # "咖啡廳" in header area (top-left)
            if "咖啡" in text and b.get("y2", 1) < 0.15:
                return True
        return False
    except Exception:
        return False


def nms(boxes, iou_thresh=0.3):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    keep = []
    for b in boxes:
        overlap = False
        for k in keep:
            ix1 = max(b[0], k[0])
            iy1 = max(b[1], k[1])
            ix2 = min(b[2], k[2])
            iy2 = min(b[3], k[3])
            inter = max(0, ix2-ix1) * max(0, iy2-iy1)
            area_b = (b[2]-b[0]) * (b[3]-b[1])
            area_k = (k[2]-k[0]) * (k[3]-k[1])
            union = area_b + area_k - inter
            if union > 0 and inter / union > iou_thresh:
                overlap = True
                break
        if not overlap:
            keep.append(b)
    return keep


def detect_bubbles(frame_bgr, tmpl):
    fh, fw = frame_bgr.shape[:2]
    ratio = WORK_W / fw
    work_h = max(1, int(fh * ratio))
    work = cv2.resize(frame_bgr, (WORK_W, work_h), interpolation=cv2.INTER_AREA)
    tw = max(4, int(tmpl.shape[1] * ratio))
    th = max(4, int(tmpl.shape[0] * ratio))

    all_hits = []
    for scale in SCALES:
        sw = max(4, int(tw * scale))
        sh = max(4, int(th * scale))
        if sw >= WORK_W or sh >= work_h:
            continue
        scaled = cv2.resize(tmpl, (sw, sh), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(work, scaled, cv2.TM_CCOEFF_NORMED)
        locs = np.where(result >= MATCH_THRESHOLD)
        for py, px in zip(*locs):
            conf = float(result[py, px])
            all_hits.append((px, py, px + sw, py + sh, conf))

    kept = nms(all_hits, NMS_IOU_THRESHOLD)

    rx1, ry1, rx2, ry2 = REGION
    detections = []
    for x1, y1, x2, y2, conf in kept:
        nx1 = x1 / WORK_W
        ny1 = y1 / work_h
        nx2 = x2 / WORK_W
        ny2 = y2 / work_h
        cx = (nx1 + nx2) / 2
        cy = (ny1 + ny2) / 2
        w = nx2 - nx1
        h = ny2 - ny1
        if cx < rx1 or cx > rx2 or cy < ry1 or cy > ry2:
            continue
        detections.append((cx, cy, w, h, conf))

    return detections[:5]


def main():
    print("=== YOLO Headpat Bubble Auto-Labeler V2 (cafe-only) ===")
    print(f"Template: {TEMPLATE_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    tmpl = load_template()
    print(f"Template loaded: {tmpl.shape[1]}x{tmpl.shape[0]}")

    # Collect trajectory tick files that have matching JSON
    tick_pairs = []  # (jpg_path, json_path)
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
        if (i + 1) % 500 == 0:
            print(f"  [filter {i+1}/{len(tick_pairs)}] cafe frames so far: {len(cafe_frames)}")
    print(f"Cafe frames found: {len(cafe_frames)}")
    print()

    # Phase 2: label cafe frames
    labeled = 0
    total_det = 0
    negative = 0  # cafe frames with no bubble (negative examples)

    for i, frame_path in enumerate(cafe_frames):
        img = cv2.imread(frame_path)
        if img is None:
            continue

        detections = detect_bubbles(img, tmpl)

        # Save image + label (even empty label = negative example, but limit negatives)
        fname = f"cafe_{labeled:05d}.jpg"
        if detections:
            shutil.copy2(frame_path, str(IMAGES_DIR / fname))
            label_path = LABELS_DIR / f"cafe_{labeled:05d}.txt"
            with open(label_path, "w") as f:
                for cx, cy, w, h, conf in detections:
                    f.write(f"{CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            total_det += len(detections)
            labeled += 1
        elif negative < len(cafe_frames) // 5:
            # Keep ~20% negative examples (cafe frames without bubbles)
            shutil.copy2(frame_path, str(IMAGES_DIR / fname))
            label_path = LABELS_DIR / f"cafe_{labeled:05d}.txt"
            label_path.touch()  # empty label = no objects
            negative += 1
            labeled += 1

        if (i + 1) % 50 == 0 or i == len(cafe_frames) - 1:
            print(f"[{i+1}/{len(cafe_frames)}] labeled={labeled} positive={labeled-negative} negative={negative} detections={total_det}")

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
    print(f"Labeled: {labeled} (positive={labeled-negative}, negative={negative})")
    print(f"Total detections: {total_det}")
    print(f"Dataset: {yaml_path}")


if __name__ == "__main__":
    main()
