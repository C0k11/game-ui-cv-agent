"""Auto-label headpat bubbles using HSV color detection.

Scans all images in the YOLO dataset and generates YOLO-format label files
by detecting bright yellow pill-shaped interaction bubbles via HSV thresholding
and contour analysis. Much faster than manual labeling for this specific target.

Usage:
    python scripts/auto_label_existing.py
    python scripts/auto_label_existing.py --preview 5   # show 5 images with drawn boxes
"""

import argparse
import os

import cv2
import numpy as np

DATASET_ROOT = r"D:\Project\ml_cache\models\yolo\dataset"
CLASS_ID = 0

# Measured from actual in-game pills: orange-yellow H~13, not pure yellow H~25
LOWER_YELLOW = np.array([5, 80, 150])
UPPER_YELLOW = np.array([35, 255, 255])
MIN_CONTOUR_AREA = 500    # pills are 700-2000px; UI panels >3000; dots <200
MAX_CONTOUR_AREA = 3000
MERGE_DIST = 50           # merge contour segments of same pill
# Final box size constraints (pixels at 1602x894)
MIN_BOX_W, MAX_BOX_W = 40, 130
MIN_BOX_H, MAX_BOX_H = 15, 45
MIN_ASPECT = 1.8          # pills are strongly elongated (measured 2.0-3.7)
MAX_ASPECT = 5.0
# ROI: exclude top UI bar, bottom toolbar, and left sidebar
ROI_TOP_FRAC = 0.12
ROI_BOT_FRAC = 0.88
ROI_LEFT_FRAC = 0.10      # exclude left sidebar (指定訪問/隨機訪問)


def _merge_boxes(boxes, dist):
    """Merge overlapping/nearby boxes."""
    if not boxes:
        return []
    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        new_merged = []
        used = [False] * len(merged)
        for i in range(len(merged)):
            if used[i]:
                continue
            x1, y1, x2, y2 = merged[i]
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                bx1, by1, bx2, by2 = merged[j]
                # Check if boxes overlap or are within merge distance
                if (x1 - dist <= bx2 and x2 + dist >= bx1 and
                        y1 - dist <= by2 and y2 + dist >= by1):
                    x1, y1 = min(x1, bx1), min(y1, by1)
                    x2, y2 = max(x2, bx2), max(y2, by2)
                    used[j] = True
                    changed = True
            new_merged.append((x1, y1, x2, y2))
        merged = new_merged
    return merged


def detect_headpat_bubbles(frame):
    h, w = frame.shape[:2]
    # Crop to game area (exclude UI bars and left sidebar)
    y_top, y_bot = int(h * ROI_TOP_FRAC), int(h * ROI_BOT_FRAC)
    x_left = int(w * ROI_LEFT_FRAC)
    roi = frame[y_top:y_bot, x_left:]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)
    # Aggressive close to merge pill segments, then open to remove noise
    kernel_close = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=3)
    kernel_open = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw_boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA or area > MAX_CONTOUR_AREA:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        # Translate back to full-image coordinates
        raw_boxes.append((x + x_left, y + y_top, x + x_left + bw, y + y_top + bh))

    # Filter by final box size and aspect ratio (no merge — morphological
    # close already connects segments; separate pills should stay separate)
    boxes = []
    for x1, y1, x2, y2 in raw_boxes:
        bw, bh = x2 - x1, y2 - y1
        aspect = bw / max(bh, 1)
        if (MIN_BOX_W <= bw <= MAX_BOX_W and MIN_BOX_H <= bh <= MAX_BOX_H
                and MIN_ASPECT <= aspect <= MAX_ASPECT):
            boxes.append((x1, y1, x2, y2))
    return boxes


def save_yolo_label(boxes, w, h, label_path):
    with open(label_path, "w", encoding="utf-8") as f:
        for x1, y1, x2, y2 in boxes:
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            f.write(f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def main():
    parser = argparse.ArgumentParser(description="Auto-label headpat bubbles via HSV detection")
    parser.add_argument("--preview", type=int, default=0,
                        help="show N images with drawn boxes for visual verification")
    args = parser.parse_args()

    images_dir = os.path.join(DATASET_ROOT, "images", "train")
    labels_dir = os.path.join(DATASET_ROOT, "labels", "train")
    os.makedirs(labels_dir, exist_ok=True)

    image_files = sorted(f for f in os.listdir(images_dir) if f.endswith((".jpg", ".png")))
    print(f"Auto-labeling {len(image_files)} images...")

    total_bubbles = 0
    labeled_count = 0
    previewed = 0

    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        boxes = detect_headpat_bubbles(frame)
        total_bubbles += len(boxes)

        base_name = os.path.splitext(img_file)[0]
        label_path = os.path.join(labels_dir, f"{base_name}.txt")
        save_yolo_label(boxes, frame.shape[1], frame.shape[0], label_path)

        if len(boxes) > 0:
            labeled_count += 1

        # Optional preview
        if args.preview > 0 and previewed < args.preview and len(boxes) > 0:
            for x1, y1, x2, y2 in boxes:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            preview_path = os.path.join(DATASET_ROOT, f"preview_{previewed}.jpg")
            cv2.imwrite(preview_path, frame)
            print(f"  Preview saved: {preview_path} ({len(boxes)} boxes)")
            previewed += 1

    print(f"\nAuto-labeling complete!")
    print(f"  Total images: {len(image_files)}")
    print(f"  Images with bubbles: {labeled_count}")
    print(f"  Total bubbles: {total_bubbles} (avg {total_bubbles / max(len(image_files), 1):.1f}/image)")
    print(f"  Labels saved to: {labels_dir}")
    print(f"\nNext: python scripts/train_yolo.py")


if __name__ == "__main__":
    main()
