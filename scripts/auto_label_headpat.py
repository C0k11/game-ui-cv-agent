"""Auto-label headpat bubbles in trajectory/raw frames using template matching.

Scans cafe frames for Emoticon_Action.png bubble icon, outputs YOLO-format
.txt label files alongside each image.

Usage:
    python scripts/auto_label_headpat.py

Output: data/yolo_headpat_dataset/ with images/ and labels/ in YOLO format.
"""
import cv2
import numpy as np
import os
import sys
import glob
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = REPO / "data" / "captures" / "Emoticon_Action.png"
OUTPUT_DIR = REPO / "data" / "yolo_headpat_dataset"
IMAGES_DIR = OUTPUT_DIR / "images"
LABELS_DIR = OUTPUT_DIR / "labels"

# Class ID for headpat bubble
CLASS_ID = 0
CLASS_NAME = "headpat_bubble"

# Template matching settings
WORK_W = 960  # downscale frames to this width for matching
MATCH_THRESHOLD = 0.80
NMS_IOU_THRESHOLD = 0.3
SCALES = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Region filter: only keep detections in cafe play area
REGION = (0.05, 0.10, 0.95, 0.85)  # x1, y1, x2, y2 normalized


def load_template():
    tmpl = cv2.imread(str(TEMPLATE_PATH), cv2.IMREAD_UNCHANGED)
    if tmpl is None:
        print(f"ERROR: template not found at {TEMPLATE_PATH}")
        sys.exit(1)
    if tmpl.shape[2] == 4:
        tmpl = cv2.cvtColor(tmpl, cv2.COLOR_BGRA2BGR)
    return tmpl


def is_cafe_frame(img_path: str) -> bool:
    """Quick check if frame looks like cafe (has cafe UI elements)."""
    # Read small portion for speed
    img = cv2.imread(img_path)
    if img is None:
        return False
    h, w = img.shape[:2]
    # Check top-left for "咖啡廳" text region (dark header bar)
    header = img[0:int(h*0.08), 0:int(w*0.20)]
    # Cafe header has dark blue/black background
    mean_val = header.mean()
    return mean_val < 120  # dark header = likely in-game screen


def nms(boxes, iou_thresh=0.3):
    """Simple NMS on list of (x1,y1,x2,y2,conf) tuples."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    keep = []
    for b in boxes:
        overlap = False
        for k in keep:
            # IoU
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


def detect_bubbles(frame_bgr, tmpl, threshold=MATCH_THRESHOLD):
    """Detect headpat bubbles via multi-scale template matching.
    
    Returns list of (cx, cy, w, h) in normalized 0-1 coords.
    """
    fh, fw = frame_bgr.shape[:2]
    ratio = WORK_W / fw
    work_h = max(1, int(fh * ratio))
    work = cv2.resize(frame_bgr, (WORK_W, work_h), interpolation=cv2.INTER_AREA)
    
    # Scale template to working resolution
    tw = max(4, int(tmpl.shape[1] * ratio))
    th = max(4, int(tmpl.shape[0] * ratio))
    
    all_hits = []  # (x1, y1, x2, y2, conf) in work-pixel coords
    
    for scale in SCALES:
        sw = max(4, int(tw * scale))
        sh = max(4, int(th * scale))
        if sw >= WORK_W or sh >= work_h:
            continue
        scaled = cv2.resize(tmpl, (sw, sh), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(work, scaled, cv2.TM_CCOEFF_NORMED)
        
        locs = np.where(result >= threshold)
        for py, px in zip(*locs):
            conf = float(result[py, px])
            all_hits.append((px, py, px + sw, py + sh, conf))
    
    # NMS
    kept = nms(all_hits, NMS_IOU_THRESHOLD)
    
    # Convert to normalized coords + filter by region
    detections = []
    rx1, ry1, rx2, ry2 = REGION
    for x1, y1, x2, y2, conf in kept:
        nx1 = x1 / WORK_W
        ny1 = y1 / work_h
        nx2 = x2 / WORK_W
        ny2 = y2 / work_h
        cx = (nx1 + nx2) / 2
        cy = (ny1 + ny2) / 2
        w = nx2 - nx1
        h = ny2 - ny1
        # Region filter
        if cx < rx1 or cx > rx2 or cy < ry1 or cy > ry2:
            continue
        detections.append((cx, cy, w, h, conf))
    
    return detections[:3]  # max 3 per frame to avoid false positive floods


def collect_frames():
    """Collect all trajectory and raw_image cafe frames."""
    frames = []
    
    # Trajectory frames
    for run_dir in sorted(glob.glob(str(REPO / "data" / "trajectories" / "run_20260306_*"))):
        for jpg in sorted(glob.glob(os.path.join(run_dir, "tick_*.jpg"))):
            frames.append(jpg)
    
    return frames


def main():
    print(f"=== YOLO Headpat Bubble Auto-Labeler ===")
    print(f"Template: {TEMPLATE_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print()
    
    # Setup output dirs
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load template
    tmpl = load_template()
    print(f"Template loaded: {tmpl.shape[1]}x{tmpl.shape[0]}")
    
    # Collect frames
    frames = collect_frames()
    print(f"Total frames to scan: {len(frames)}")
    print()
    
    # Process frames
    labeled = 0
    skipped = 0
    total_detections = 0
    
    for i, frame_path in enumerate(frames):
        img = cv2.imread(frame_path)
        if img is None:
            skipped += 1
            continue
        
        # Detect bubbles
        detections = detect_bubbles(img, tmpl)
        
        if detections:
            # Copy image to dataset
            fname = f"frame_{labeled:05d}.jpg"
            dst_img = IMAGES_DIR / fname
            shutil.copy2(frame_path, str(dst_img))
            
            # Write YOLO label file
            label_path = LABELS_DIR / f"frame_{labeled:05d}.txt"
            with open(label_path, "w") as f:
                for cx, cy, w, h, conf in detections:
                    f.write(f"{CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            
            total_detections += len(detections)
            labeled += 1
        
        # Progress every 100 frames
        if (i + 1) % 100 == 0 or i == len(frames) - 1:
            print(f"[{i+1}/{len(frames)}] scanned={i+1} labeled={labeled} detections={total_detections} skipped={skipped}")
    
    # Write dataset.yaml
    yaml_path = OUTPUT_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {OUTPUT_DIR}\n")
        f.write(f"train: images\n")
        f.write(f"val: images\n")
        f.write(f"nc: 1\n")
        f.write(f"names: ['{CLASS_NAME}']\n")
    
    print()
    print(f"=== DONE ===")
    print(f"Labeled frames: {labeled}")
    print(f"Total detections: {total_detections}")
    print(f"Skipped: {skipped}")
    print(f"Dataset YAML: {yaml_path}")
    print(f"Images: {IMAGES_DIR}")
    print(f"Labels: {LABELS_DIR}")


if __name__ == "__main__":
    main()
