"""Build a single-class `角色头像` detector training dataset.

Two data sources combined:
  1. SEED — 14 frames in data/raw_images/_backups/<ts>/ with hand-labeled
     `角色头像` bboxes (from before they got trimmed out of master).
  2. AUTO — schedule trajectory frames where avatar_cls v2 is very
     confident (top1 >= 0.85) on a sliding-window crop → that window's
     bbox is a pseudo-label.  Treats the classifier as a teacher.

Why this works:
  avatar_cls v2 has 95.91% top1 on trajectory val.  At conf >= 0.85 the
  precision is much higher (probably >98%).  Using those high-conf
  windows as bbox labels gives us hundreds of clean head-position
  examples without manual labeling.

Output:
  D:/Project/ml_cache/models/yolo/dataset/head_detector_v1/
    images/train/*.jpg
    images/val/*.jpg
    labels/train/*.txt  (single class 0 = 角色头像)
    labels/val/*.txt
    data.yaml
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT_ROOT = Path(r"D:\Project\ml_cache\models\yolo\dataset\head_detector_v1")

# Where avatar_cls v2 + static_ui v3 + bilingual map live
STATIC_UI_MODEL = Path(r"D:\Project\ml_cache\models\yolo\runs\static_ui_v3_yolo26n\weights\best.pt")
AVATAR_CLS_MODEL = Path(r"D:\Project\ml_cache\models\yolo\runs\avatar_cls_v2_yolo26n\weights\best.pt")

VAL_RATIO = 0.20
SEED = 42


def find_seed_labels() -> List[Tuple[Path, Path]]:
    """Find (image, label) pairs where the backup label has 角色头像 entries."""
    backup_root = REPO / "data" / "raw_images" / "_backups"
    pairs: List[Tuple[Path, Path]] = []
    if not backup_root.is_dir():
        return pairs
    # Use most recent backup
    backups = sorted([d for d in backup_root.iterdir() if d.is_dir()], reverse=True)
    if not backups:
        return pairs
    backup = backups[0]
    master_pre = [c.strip() for c in (backup / "_classes.txt").read_text(encoding="utf-8").splitlines() if c.strip()]
    if "角色头像" not in master_pre:
        print(f"[seed] no 角色头像 in backup master")
        return pairs
    head_idx = master_pre.index("角色头像")

    # Walk backup .txt files looking for any with cls == head_idx
    for d in (backup / "raw").iterdir():
        if not d.is_dir():
            continue
        for lf in d.glob("*.txt"):
            if lf.name == "classes.txt":
                continue
            try:
                txt = lf.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            has_head = False
            head_lines = []
            for line in txt.splitlines():
                parts = line.strip().split()
                if not parts or not parts[0].lstrip("-").isdigit():
                    continue
                if int(parts[0]) == head_idx:
                    has_head = True
                    # Single-class label: cls=0, keep cx cy w h
                    head_lines.append("0 " + " ".join(parts[1:5]))
            if has_head:
                # Find the corresponding image in raw_images current state
                # backup keeps only .txt; image is in original raw_images/<dataset>/<frame>.jpg
                ds_name = d.name
                orig_dir = REPO / "data" / "raw_images" / ds_name
                if not orig_dir.is_dir():
                    continue
                sub = orig_dir / "frames" if (orig_dir / "frames").is_dir() else orig_dir
                img_path = sub / (lf.stem + ".jpg")
                if not img_path.exists():
                    continue
                pairs.append((img_path, head_lines))
    print(f"[seed] backup={backup.name}  found {len(pairs)} frames with 角色头像 labels")
    return pairs


def find_trajectory_schedule_frames(limit: int = 50) -> List[Path]:
    """Walk trajectories for schedule-open frames."""
    ROOM_NAMES = ("視聽室","體育館","圖書館","教室","實驗室","射擊場","載具庫")
    import json
    out: List[Path] = []
    for run in sorted((REPO / "data" / "trajectories").iterdir(), reverse=True)[:50]:
        if not run.is_dir() or not run.name.startswith("run_"):
            continue
        for tj in sorted(run.glob("tick_*.json")):
            try:
                d = json.loads(tj.read_text(encoding="utf-8"))
            except Exception:
                continue
            if d.get("skill") != "Schedule":
                continue
            has_header = False
            has_room = False
            for b in d.get("ocr_boxes", []) or []:
                t = b.get("text") or ""
                if "全體課程表" in t:
                    has_header = True
                if any(rn in t for rn in ROOM_NAMES):
                    has_room = True
                if has_header and has_room:
                    break
            if has_header and has_room:
                jpg = tj.with_suffix(".jpg")
                if jpg.exists() and jpg.stat().st_size > 1000:
                    out.append(jpg)
                    if len(out) >= limit:
                        return out
    return out


def auto_label_frame(img: np.ndarray, detect_model, room_idx: int,
                     clf_model, cls_conf_threshold: float = 0.85) -> List[str]:
    """Run static_ui→房间区域, then slide window classifying each.
    Returns YOLO-format lines for windows where classifier conf >= threshold.
    """
    h, w = img.shape[:2]
    out_lines: List[str] = []

    # Detect 房间区域
    det = detect_model(img, conf=0.15, verbose=False)[0]
    raw_rooms = []
    for b in det.boxes:
        if int(b.cls[0]) == room_idx:
            raw_rooms.append((b.xyxy[0].tolist(), float(b.conf[0])))
    # NMS dedupe
    raw_rooms.sort(key=lambda r: -r[1])
    rooms = []
    for (x1, y1, x2, y2), c in raw_rooms:
        keep = True
        for (kx1, ky1, kx2, ky2), _ in rooms:
            ix1 = max(x1, kx1); iy1 = max(y1, ky1)
            ix2 = min(x2, kx2); iy2 = min(y2, ky2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2-ix1)*(iy2-iy1)
                a = (x2-x1)*(y2-y1)
                b_area = (kx2-kx1)*(ky2-ky1)
                if inter / max(1, a + b_area - inter) > 0.5:
                    keep = False; break
        if keep:
            rooms.append(((x1,y1,x2,y2), c))

    # For each room, slide window in bottom strip
    for (rx1, ry1, rx2, ry2), _ in rooms:
        rx1, ry1, rx2, ry2 = int(rx1), int(ry1), int(rx2), int(ry2)
        rh = ry2 - ry1
        sy1 = ry1 + int(0.55 * rh)
        sy2 = ry2
        strip_h = sy2 - sy1
        if strip_h < 20:
            continue
        side = strip_h
        stride = max(8, side // 3)
        candidates = []
        x_cursor = rx1
        while x_cursor + side <= rx2:
            crop = img[sy1:sy2, x_cursor:x_cursor+side]
            if crop.size > 0:
                r2 = clf_model.predict(crop, verbose=False, imgsz=224)[0]
                probs = r2.probs.data.cpu().numpy()
                top1_conf = float(np.max(probs))
                if top1_conf >= cls_conf_threshold:
                    candidates.append({
                        "conf": top1_conf,
                        "x1": x_cursor, "y1": sy1,
                        "x2": x_cursor + side, "y2": sy2,
                    })
            x_cursor += stride

        # IoU-NMS dedupe (drop overlapping windows, keep highest conf, regardless of class)
        candidates.sort(key=lambda c: -c["conf"])
        kept = []
        for c in candidates:
            drop = False
            for k in kept:
                ix1 = max(c["x1"], k["x1"]); iy1 = max(c["y1"], k["y1"])
                ix2 = min(c["x2"], k["x2"]); iy2 = min(c["y2"], k["y2"])
                if ix2 > ix1 and iy2 > iy1:
                    inter = (ix2-ix1)*(iy2-iy1)
                    a = (c["x2"]-c["x1"])*(c["y2"]-c["y1"])
                    b_area = (k["x2"]-k["x1"])*(k["y2"]-k["y1"])
                    if inter / max(1, a+b_area-inter) > 0.4:
                        drop = True; break
            if not drop:
                kept.append(c)

        for c in kept:
            cx = (c["x1"]+c["x2"])/2/w
            cy = (c["y1"]+c["y2"])/2/h
            ww = (c["x2"]-c["x1"])/w
            hh = (c["y2"]-c["y1"])/h
            out_lines.append(f"0 {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}")
    return out_lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-traj-frames", type=int, default=50,
                    help="How many trajectory schedule frames to auto-label")
    ap.add_argument("--cls-conf", type=float, default=0.85,
                    help="Min avatar_cls conf to accept window as head bbox")
    args = ap.parse_args()

    # ── 1. Seed labels from backup (manual) ──
    seed_pairs = find_seed_labels()

    # ── 2. Auto-label trajectory frames ──
    print(f"[auto] loading detection + classifier models...")
    from ultralytics import YOLO
    detect_model = YOLO(str(STATIC_UI_MODEL))
    detect_names = detect_model.names
    room_idx = next((k for k, v in detect_names.items() if v == "房间区域"), None)
    if room_idx is None:
        print(f"[err] static_ui has no '房间区域' class")
        return 1
    clf_model = YOLO(str(AVATAR_CLS_MODEL))

    print(f"[auto] scanning trajectory schedule frames...")
    traj_frames = find_trajectory_schedule_frames(args.max_traj_frames)
    print(f"[auto] {len(traj_frames)} schedule frames found")
    auto_pairs: List[Tuple[Path, List[str]]] = []
    for fi, jpg in enumerate(traj_frames):
        img = cv2.imdecode(np.fromfile(str(jpg), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        lines = auto_label_frame(img, detect_model, room_idx, clf_model, args.cls_conf)
        if lines:
            auto_pairs.append((jpg, lines))
        if (fi + 1) % 10 == 0:
            print(f"  scanned {fi+1}/{len(traj_frames)} | {len(auto_pairs)} labeled so far | "
                  f"total head boxes: {sum(len(l) for _, l in auto_pairs)}")

    # ── 3. Merge + split ──
    all_pairs = []
    for img_path, lines in seed_pairs:
        all_pairs.append((img_path, lines, "seed"))
    for img_path, lines in auto_pairs:
        all_pairs.append((img_path, lines, "auto"))
    print(f"\n[merge] total {len(all_pairs)} (image, labels) pairs")
    print(f"  seed: {len(seed_pairs)}, auto: {len(auto_pairs)}")
    total_boxes = sum(len(l) for _, l, _ in all_pairs)
    print(f"  total head boxes: {total_boxes}")

    if not all_pairs:
        print("[err] no training data — abort")
        return 1

    # Shuffle + split
    rng = random.Random(SEED)
    rng.shuffle(all_pairs)
    n_val = max(1, int(len(all_pairs) * VAL_RATIO))
    val_set = all_pairs[:n_val]
    train_set = all_pairs[n_val:]

    # Clean output
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    for sub in ("images/train","images/val","labels/train","labels/val"):
        (OUT_ROOT / sub).mkdir(parents=True, exist_ok=True)

    def emit(pair_list, split):
        n_boxes = 0
        for img_path, lines, tag in pair_list:
            stem = f"{tag}__{img_path.parent.name}__{img_path.stem}"
            shutil.copy2(img_path, OUT_ROOT / "images" / split / (stem + ".jpg"))
            (OUT_ROOT / "labels" / split / (stem + ".txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")
            n_boxes += len(lines)
        print(f"[emit] {split}: {len(pair_list)} files, {n_boxes} head boxes")
    emit(train_set, "train")
    emit(val_set, "val")

    # data.yaml — single class
    (OUT_ROOT / "data.yaml").write_text(
        f"path: {OUT_ROOT.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names:\n  0: '角色头像'\n",
        encoding="utf-8"
    )
    print(f"\n[done] dataset → {OUT_ROOT}")
    print("Next: py scripts/train_yolo26.py head_detector")
    return 0


if __name__ == "__main__":
    sys.exit(main())
