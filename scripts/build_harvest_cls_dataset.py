"""Build a YOLO-cls training dataset from MomoTalk + Cafe invite-list
trajectories — using OCR boxes already stored in tick_*.json.

Why this approach (decided 2026-05-17 after preset-coord mismatch
debug):
  - Trajectory JSON's ocr_boxes were captured AT THE FRAME'S OWN
    resolution by the live pipeline.  Their coords are pixel-correct
    for that frame.
  - The harvest preset's name_box coord was calibrated against one
    specific window size; on other sizes it drifts (catches affinity
    numbers instead of the name text).
  - Solution: scan the existing OCR boxes for character names
    (matched against data/captures/角色头像/ + student_name_map.json).
    Each name box's spatial position (cy) identifies the row; crop
    the avatar from the LEFT of that box.

Pipeline:
  1. Walk trajectory frames where Cafe.sub_state=="invite" AND there
     are ≥3 邀請 OCR boxes (real invite list, not lobby sidebar).
  2. For each frame, scan its ocr_boxes for character names.
  3. For each name match, crop the avatar region to the LEFT of the
     box at the same y-band.
  4. Save crop to data/yolo_datasets/schedule_cells/train/<class>/...
  5. pHash dedup per class with cap N=PER_CLASS_CAP.

Output is yolo26n-cls ready (train/<class>/*.jpg + val/<class>/*.jpg).
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import imagehash

REPO = Path(__file__).resolve().parents[1]
TRAJ_ROOT = REPO / "data" / "trajectories"
NAME_MAP_JSON = REPO / "data" / "student_name_map.json"
# Two avatar reference dirs:
#   AVATAR_REF_DIR_EN: 250 CG-quality English-named PNGs (Wakamo.png)
#   AVATAR_REF_DIR_CN: 191 in-game-OCR'd Chinese-named PNGs (麻白.png)
# Use BOTH as the known class set — trajectory OCR returns Chinese
# names directly, so CN refs match without going through name_map.
# EN refs cover legacy paths + characters not yet harvested in CN.
AVATAR_REF_DIR_EN = REPO / "data" / "captures" / "角色头像"
AVATAR_REF_DIR_CN = REPO / "data" / "captures" / "角色头像_crop_harvested_named"
OUT_ROOT = REPO / "data" / "yolo_datasets" / "schedule_cells"

PER_CLASS_CAP = 80        # bumped 40 → 80; YOLO cls needs more diversity per class
PHASH_THRESHOLD = 0       # 0 = only drop exact (hamming=0) duplicates
                          # was 2 — that collapsed affinity-number variants
                          # the model NEEDS to learn to ignore
VAL_RATIO = 0.20
SEED = 42

# Avatar crop geometry: BA's invite-list row has a circular ~80px
# avatar IMMEDIATELY left of the name text.  On a 2255px-wide frame
# that's ~3.5-4% of frame width.  Use 5% for a small margin.
# Vertical: tightly square (avatar is round).
AVATAR_SIZE_REL = 0.05    # avatar bbox side = 5% of frame width
AVATAR_RIGHT_PAD = 0.003  # tiny gap between avatar edge and name text


def _sanitize(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"[\\/:*?\"<>|]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def load_name_map() -> dict[str, str]:
    if not NAME_MAP_JSON.exists():
        return {}
    return json.loads(NAME_MAP_JSON.read_text(encoding="utf-8"))


def load_known_classes() -> tuple[set[str], set[str]]:
    """Return (cn_classes, en_classes) — Chinese names + English names.

    Trajectory OCR gives Chinese; matching against CN avoids the
    student_name_map indirection (and the OCR variants that don't
    appear in the map).  EN remains as fallback via the map.
    """
    cn = {p.stem for p in AVATAR_REF_DIR_CN.glob("*.png")} if AVATAR_REF_DIR_CN.exists() else set()
    en = {p.stem for p in AVATAR_REF_DIR_EN.glob("*.png")} if AVATAR_REF_DIR_EN.exists() else set()
    return cn, en


def phash_of(roi_bgr: np.ndarray) -> imagehash.ImageHash:
    rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    return imagehash.phash(Image.fromarray(rgb))


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=10000)
    ap.add_argument(
        "--sub-state",
        default="invite",
        help="Cafe sub_state to filter on (invite, switch, etc.)",
    )
    args = ap.parse_args()

    name_map = load_name_map()
    cn_known, en_known = load_known_classes()
    print(f"[refs] {len(cn_known)} CN-named refs (harvested) + {len(en_known)} EN-named refs (CG), "
          f"{len(name_map)} CN→EN map entries")

    rng = random.Random(SEED)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Match resolver — given OCR text, return canonical class or None.
    # Preference order:
    #   1. Direct CN match (most coverage, no indirection)
    #   2. Direct EN match (legacy)
    #   3. CN→EN via student_name_map.json (fallback for unknown CN)
    def resolve_class(text: str) -> str | None:
        if not text:
            return None
        t = _sanitize(text)
        if t in cn_known:
            return t
        if t in en_known:
            return t
        if t in name_map:
            mapped = name_map[t]
            if mapped in cn_known or mapped in en_known:
                return mapped
        return None

    class_phashes: dict[str, list[imagehash.ImageHash]] = {}
    class_counts: dict[str, int] = {}
    dropped_dup = 0
    dropped_cap = 0
    dropped_load = 0
    dropped_crop = 0
    n_frames = 0
    n_matched_rows = 0

    if not TRAJ_ROOT.exists():
        print(f"trajectory root missing: {TRAJ_ROOT}")
        return 1

    for run in sorted(TRAJ_ROOT.iterdir(), reverse=True):
        if not run.is_dir() or not run.name.startswith("run_"):
            continue
        for js in sorted(run.glob("tick_*.json")):
            if n_frames >= args.max_frames:
                break
            try:
                d = json.loads(js.read_text(encoding="utf-8"))
            except Exception:
                continue
            if d.get("skill") != "Cafe":
                continue
            if d.get("sub_state") != args.sub_state:
                continue
            ocr_boxes = d.get("ocr_boxes") or []
            invite_count = sum(
                1 for b in ocr_boxes
                if any(k in (b.get("text") or "") for k in ("邀請", "邀请", "邀睛"))
            )
            if invite_count < 3:
                continue

            # Find character-name OCR boxes
            name_boxes: list[tuple[str, dict]] = []
            for b in ocr_boxes:
                txt = b.get("text") or ""
                conf = float(b.get("conf") or 0.0)
                if conf < 0.55:
                    continue
                cls = resolve_class(txt)
                if cls is not None:
                    name_boxes.append((cls, b))

            if not name_boxes:
                continue

            jpg = js.with_suffix(".jpg")
            if not jpg.exists():
                continue
            img = cv2.imread(str(jpg))
            if img is None:
                dropped_load += 1
                continue
            n_frames += 1
            h, w = img.shape[:2]

            for cls, box in name_boxes:
                # Crop avatar: small square immediately left of name box.
                # bcy = vertical center of name text; avatar is centered
                # at same y but slightly above (circle taller than text).
                bx1 = float(box["x1"])
                by1 = float(box["y1"])
                by2 = float(box["y2"])
                bcy = (by1 + by2) / 2.0
                # Aspect ratio: image_w/image_h matters because side is
                # taken as 5% of WIDTH; in normalized coords vertical
                # extent must scale by aspect to stay square in pixels.
                aspect = w / max(1, h)
                side_x = AVATAR_SIZE_REL                       # x extent (norm)
                side_y = AVATAR_SIZE_REL * aspect              # y extent (norm)
                av_x2 = max(0.0, bx1 - AVATAR_RIGHT_PAD)
                av_x1 = max(0.0, av_x2 - side_x)
                av_y1 = max(0.0, bcy - side_y / 2.0)
                av_y2 = min(1.0, bcy + side_y / 2.0)

                px1 = max(0, int(av_x1 * w))
                py1 = max(0, int(av_y1 * h))
                px2 = min(w, int(av_x2 * w))
                py2 = min(h, int(av_y2 * h))
                if px2 - px1 < 20 or py2 - py1 < 20:
                    dropped_crop += 1
                    continue
                roi = img[py1:py2, px1:px2]
                if roi.size == 0:
                    dropped_crop += 1
                    continue

                # pHash dedup + cap
                if class_counts.get(cls, 0) >= PER_CLASS_CAP:
                    dropped_cap += 1
                    continue
                kept = class_phashes.setdefault(cls, [])
                ph = phash_of(roi)
                if any((ph - prev) <= PHASH_THRESHOLD for prev in kept):
                    dropped_dup += 1
                    continue
                kept.append(ph)

                split = "val" if rng.random() < VAL_RATIO else "train"
                out_dir = OUT_ROOT / split / cls
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{run.name}_{js.stem}_{box.get('x1',0):.3f}_{box.get('y1',0):.3f}.jpg"
                cv2.imwrite(str(out_path), roi)
                class_counts[cls] = class_counts.get(cls, 0) + 1
                n_matched_rows += 1

            if n_frames % 200 == 0:
                print(
                    f"  {n_frames} frames | matches {n_matched_rows} | "
                    f"classes {len(class_counts)} | "
                    f"dup {dropped_dup} cap {dropped_cap} "
                    f"load {dropped_load} crop {dropped_crop}"
                )
        if n_frames >= args.max_frames:
            break

    print(f"\n[done] {OUT_ROOT}")
    print(f"  frames scanned       : {n_frames}")
    print(f"  rows matched         : {n_matched_rows}")
    print(f"  classes covered      : {len(class_counts)}")
    print(f"  total kept           : {sum(class_counts.values())}")
    print(f"  dropped (dup)        : {dropped_dup}")
    print(f"  dropped (cap)        : {dropped_cap}")
    print(f"  dropped (crop)       : {dropped_crop}")
    print(f"  dropped (load)       : {dropped_load}")
    print(f"\n  top 20 classes:")
    for cls in sorted(class_counts, key=lambda c: -class_counts[c])[:20]:
        print(f"    {class_counts[cls]:3d}  {cls}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
