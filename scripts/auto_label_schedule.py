"""Auto-label student avatars in schedule rooms using OpenCV template matching.

Scans schedule screenshots and uses `AvatarMatcher` to detect where student avatars
are located in the 6 schedule rooms, then outputs YOLO format bounding boxes.

Usage:
    python scripts/auto_label_schedule.py
    python scripts/auto_label_schedule.py --preview 5   # show 5 images with drawn boxes
"""

import argparse
import os
import cv2
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path to import vision modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vision.avatar_matcher import AvatarMatcher

DATASET_ROOT = r"D:\Project\ml_cache\models\yolo\datasets\schedule"
CLASS_ID = 0  # student_avatar

def detect_students_in_rooms(frame: np.ndarray, matcher: AvatarMatcher, candidate_names: list[str]) -> list[tuple[int, int, int, int]]:
    """Detect students in the 6 hardcoded room regions."""
    h, w = frame.shape[:2]
    
    # 6 Room regions (same as in opencv_pipeline.py)
    rooms = [
        (int(0.05 * w), int(0.15 * h), int(0.38 * w), int(0.48 * h)),
        (int(0.38 * w), int(0.15 * h), int(0.66 * w), int(0.48 * h)),
        (int(0.66 * w), int(0.15 * h), int(0.95 * w), int(0.48 * h)),
        (int(0.05 * w), int(0.50 * h), int(0.38 * w), int(0.85 * h)),
        (int(0.38 * w), int(0.50 * h), int(0.66 * w), int(0.85 * h)),
        (int(0.66 * w), int(0.50 * h), int(0.95 * w), int(0.85 * h)),
    ]
    
    boxes = []
    
    for rx1, ry1, rx2, ry2 in rooms:
        room_roi = frame[ry1:ry2, rx1:rx2]
        if room_roi.size == 0:
            continue
            
        # We need to find the avatar inside the room ROI.
        # Since we don't have YOLO yet, we can do a reverse sliding window or simply
        # assume the avatar is somewhat centered/fixed size in the room if present.
        # But a better way for auto-labeling without YOLO is to actually use cv2.matchTemplate
        # against all known avatar templates in the room ROI.
        
        best_score = -1.0
        best_box = None
        
        for name in candidate_names:
            tmpl_img = matcher._get_avatar_img(name)
            if tmpl_img is None:
                continue
                
            # Resize template to expected avatar size in schedule (approx 120x120 at 1080p, scale by w/1920)
            expected_size = int(120 * (w / 1920))
            if expected_size <= 0: continue
            
            tmpl_resized = cv2.resize(tmpl_img, (expected_size, expected_size))
            
            # Apply anti-occlusion mask to both template and the room search area
            tmpl_masked = matcher._crop_bottom_right(tmpl_resized)
            
            # We must use TM_CCORR_NORMED or TM_CCOEFF_NORMED with the masked template
            # Convert to grayscale for simple template matching to find the bounding box
            tmpl_gray = cv2.cvtColor(tmpl_masked, cv2.COLOR_BGR2GRAY)
            room_gray = cv2.cvtColor(room_roi, cv2.COLOR_BGR2GRAY)
            
            # Mask out the zero pixels from cropping
            mask = np.where(tmpl_gray > 0, 255, 0).astype(np.uint8)
            
            if room_gray.shape[0] < tmpl_gray.shape[0] or room_gray.shape[1] < tmpl_gray.shape[1]:
                continue
                
            res = cv2.matchTemplate(room_gray, tmpl_gray, cv2.TM_CCORR_NORMED, mask=mask)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            
            if max_val > best_score and max_val > 0.85: # High threshold for auto-labeling
                best_score = max_val
                tx, ty = max_loc
                best_box = (rx1 + tx, ry1 + ty, rx1 + tx + expected_size, ry1 + ty + expected_size)
                
        if best_box is not None:
            boxes.append(best_box)
            
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
    parser = argparse.ArgumentParser(description="Auto-label schedule student avatars via OpenCV matching")
    parser.add_argument("--preview", type=int, default=0, help="show N images with drawn boxes")
    args = parser.parse_args()

    images_dir = os.path.join(DATASET_ROOT, "images", "train")
    labels_dir = os.path.join(DATASET_ROOT, "labels", "train")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    image_files = sorted(f for f in os.listdir(images_dir) if f.endswith((".jpg", ".png")))
    if not image_files:
        print(f"No images found in {images_dir}")
        print("Please place schedule screenshots there first.")
        return

    print(f"Auto-labeling {len(image_files)} schedule images...")
    
    # Initialize AvatarMatcher
    avatars_dir = r"D:\Project\ai game secretary\data\captures\角色头像"
    matcher = AvatarMatcher(avatars_dir)
    candidate_names = [f.stem for f in Path(avatars_dir).glob("*.png")]
    print(f"Loaded {len(candidate_names)} avatar templates for matching.")

    total_avatars = 0
    labeled_count = 0
    previewed = 0

    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        frame = cv2.imread(img_path)
        if frame is None:
            continue

        boxes = detect_students_in_rooms(frame, matcher, candidate_names)
        total_avatars += len(boxes)

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
    print(f"  Images with avatars: {labeled_count}")
    print(f"  Total avatars: {total_avatars} (avg {total_avatars / max(len(image_files), 1):.1f}/image)")
    print(f"  Labels saved to: {labels_dir}")
    print(f"\nNext: python scripts/train_yolo.py --skill schedule")

if __name__ == "__main__":
    main()
