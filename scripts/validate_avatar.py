"""Validate avatar detection + matching pipeline.

1. Run avatar_augmented YOLO on raw game screenshots
2. Compare detections vs ground truth labels (class 9 = 角色头像)
3. Use AvatarMatcher to identify which character each detection is
4. Report precision, recall, and identification accuracy

Usage:
    py scripts/validate_avatar.py
    py scripts/validate_avatar.py --show   # visualize detections
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

ML_CACHE = Path(r"D:\Project\ml_cache\models\yolo")
RAW_DIR = REPO_ROOT / "data" / "raw_images" / "run_20260228_235254"
AVATAR_DIR = REPO_ROOT / "data" / "captures" / "角色头像"
CLASSES_FILE = RAW_DIR / "classes.txt"

# Class index for 角色头像 in the raw annotations
AVATAR_CLASS_ID = 9
# IoU threshold for matching detections to ground truth
IOU_THRESH = 0.40


def load_classes(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    return {i: name.strip() for i, name in enumerate(lines)}


def load_gt_boxes(txt_path: Path, target_cls: int):
    """Load YOLO-format ground truth boxes for a specific class."""
    if not txt_path.is_file():
        return []
    boxes = []
    for line in txt_path.read_text(encoding="utf-8").strip().split("\n"):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        if cls_id != target_cls:
            continue
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        boxes.append((cx, cy, w, h))
    return boxes


def iou_xywh(a, b):
    """Compute IoU between two (cx, cy, w, h) normalized boxes."""
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / max(union, 1e-9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true", help="Visualize detections")
    parser.add_argument("--conf", type=float, default=0.15, help="Detection confidence")
    args = parser.parse_args()

    # Load YOLO model — use full.engine (avatar_augmented is broken, 0 detections)
    from ultralytics import YOLO
    engine_path = ML_CACHE / "full.engine"
    pt_path = ML_CACHE / "full.pt"
    model_path = engine_path if engine_path.is_file() else pt_path
    if not model_path.is_file():
        print(f"Model not found: {engine_path} or {pt_path}")
        return
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    # Find the class index for 角色头像 in this model
    avatar_cls_ids = [k for k, v in model.names.items() if v == "角色头像"]
    if not avatar_cls_ids:
        print("ERROR: Model has no '角色头像' class!")
        return
    det_avatar_cls = avatar_cls_ids[0]
    print(f"Avatar class in model: id={det_avatar_cls} name='{model.names[det_avatar_cls]}'")

    # Load AvatarMatcher
    from vision.avatar_matcher import AvatarMatcher
    matcher = AvatarMatcher(str(AVATAR_DIR))
    all_avatar_names = [p.stem for p in sorted(AVATAR_DIR.glob("*.png"))]
    print(f"Avatar references: {len(all_avatar_names)} characters")
    # Use a small subset for matching speed (like real pipeline uses target_favorites)
    # Pick first 20 for validation timing
    avatar_names = all_avatar_names[:20]
    print(f"Using {len(avatar_names)} candidates for matching benchmark")

    # Load classes
    classes = load_classes(CLASSES_FILE) if CLASSES_FILE.is_file() else {}
    print(f"Raw label classes: {classes.get(AVATAR_CLASS_ID, '?')} (id={AVATAR_CLASS_ID})")

    # Collect images with GT labels
    images = sorted(RAW_DIR.glob("*.jpg"))
    print(f"Raw images: {len(images)}")

    total_gt = 0
    total_det = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_matched = 0  # avatar identity matched
    match_scores = []

    for img_path in images:
        gt_path = img_path.with_suffix(".txt")
        gt_boxes = load_gt_boxes(gt_path, AVATAR_CLASS_ID)

        # Run YOLO detection
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]

        results = model(img, conf=args.conf, verbose=False)
        det_boxes = []
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) != det_avatar_cls:
                    continue  # only count avatar detections
                bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                cx = ((bx1 + bx2) / 2) / w
                cy = ((by1 + by2) / 2) / h
                bw = (bx2 - bx1) / w
                bh = (by2 - by1) / h
                det_boxes.append((cx, cy, bw, bh, float(box.conf[0]),
                                  int(bx1), int(by1), int(bx2), int(by2)))

        if not gt_boxes and not det_boxes:
            continue

        total_gt += len(gt_boxes)
        total_det += len(det_boxes)

        # Match detections to GT
        matched_gt = set()
        tp = 0
        for det in det_boxes:
            best_iou = 0
            best_gi = -1
            for gi, gt in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                val = iou_xywh(det[:4], gt)
                if val > best_iou:
                    best_iou = val
                    best_gi = gi
            if best_iou >= IOU_THRESH and best_gi >= 0:
                matched_gt.add(best_gi)
                tp += 1
            else:
                total_fp += 1

        total_tp += tp
        total_fn += len(gt_boxes) - tp

        # Try avatar matching on detected ROIs
        for det in det_boxes:
            _, _, _, _, conf, bx1, by1, bx2, by2 = det
            roi = img[max(0, by1):by2, max(0, bx1):bx2]
            if roi.size == 0:
                continue
            name, score = matcher.match_avatar(roi, avatar_names)
            if name and score > 0.3:
                total_matched += 1
                match_scores.append(score)

        # Visualization
        if args.show and (gt_boxes or det_boxes):
            vis = img.copy()
            for gt in gt_boxes:
                x1 = int((gt[0] - gt[2] / 2) * w)
                y1 = int((gt[1] - gt[3] / 2) * h)
                x2 = int((gt[0] + gt[2] / 2) * w)
                y2 = int((gt[1] + gt[3] / 2) * h)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, "GT", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            for det in det_boxes:
                _, _, _, _, conf, bx1, by1, bx2, by2 = det
                roi = img[max(0, by1):by2, max(0, bx1):bx2]
                name, score = matcher.match_avatar(roi, avatar_names) if roi.size > 0 else (None, -1)
                color = (255, 0, 0) if score > 0.3 else (0, 0, 255)
                cv2.rectangle(vis, (bx1, by1), (bx2, by2), color, 2)
                label = f"{name}:{score:.2f}" if name else f"?:{conf:.2f}"
                cv2.putText(vis, label, (bx1, by1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.imshow(f"Avatar Validation - {img_path.name}", cv2.resize(vis, (960, 540)))
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()
            if key == 27:  # ESC to quit
                break

    # Report
    print(f"\n{'='*60}")
    print(f"Avatar Detection Results (IoU>={IOU_THRESH:.2f}, conf>={args.conf:.2f})")
    print(f"{'='*60}")
    print(f"  Ground truth avatars: {total_gt}")
    print(f"  Detections:           {total_det}")
    print(f"  True positives:       {total_tp}")
    print(f"  False positives:      {total_fp}")
    print(f"  False negatives:      {total_fn}")
    prec = total_tp / max(total_tp + total_fp, 1)
    rec = total_tp / max(total_gt, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    print(f"  Precision:            {prec:.3f}")
    print(f"  Recall:               {rec:.3f}")
    print(f"  F1:                   {f1:.3f}")
    print(f"\nAvatar Identification (AvatarMatcher, score>0.3):")
    print(f"  Identified:           {total_matched}/{total_det} "
          f"({100 * total_matched / max(total_det, 1):.1f}%)")
    if match_scores:
        print(f"  Avg match score:      {sum(match_scores) / len(match_scores):.3f}")
        print(f"  Min/Max score:        {min(match_scores):.3f} / {max(match_scores):.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
