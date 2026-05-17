"""Build YOLO-cls dataset for the 全體課程表 cells using strip pre-sets.

Pipeline:
  1. Read data/schedule_avatar_regions.json (9 strips × cells_per_room = 27 cells)
  2. Walk data/yolo_datasets/schedule_roster/images_raw/ (1765 overlay shots)
  3. For each overlay image, crop the 27 cell regions
  4. For each cell:
       a. Color-stats check → if mostly grey/white → bucket `__empty__/`
       b. Otherwise run AvatarMatcher against all 250 reference avatars
          and pick the best score
       c. score >= STRONG_THRESHOLD (0.55) → bucket `<character>/`
       d. otherwise → bucket `__uncertain__/` for manual review
  5. Output structure ready for yolo26n-cls training:
       data/yolo_datasets/schedule_cells/
         train/<class>/<run>_<tick>_<cell_idx>.jpg
         val/<class>/<run>_<tick>_<cell_idx>.jpg
       (80/20 random split)

User workflow:
  python scripts/build_schedule_cells_dataset.py
  # → opens the dataset dir
  # User reviews __uncertain__/ — drags into correct class folder
  # User spot-checks a few mainstream classes for mis-labels
  # Then: add config to scripts/train_yolo26.py and train yolo26n-cls
"""
from __future__ import annotations

import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import imagehash

REPO = Path(__file__).resolve().parents[1]
REGIONS_JSON = REPO / "data" / "schedule_avatar_regions.json"
IMAGES_RAW = REPO / "data" / "yolo_datasets" / "schedule_roster" / "images_raw"
AVATAR_REF_DIR = REPO / "data" / "captures" / "角色头像"
OUT_ROOT = REPO / "data" / "yolo_datasets" / "schedule_cells"

# Match thresholds (tuned against AvatarMatcher's combined template+hist score)
STRONG_THRESHOLD = 0.55  # score >= → confident character match
UNCERTAIN_BUCKET = "__uncertain__"
EMPTY_BUCKET = "__empty__"
VAL_RATIO = 0.20
SEED = 42
# Empty-cell color heuristic: mean BGR stays close to grey (low chroma)
# AND brightness is high (white-ish) OR mid-tone grey.  BA's empty slot
# is a flat grey-pink with very low saturation.
EMPTY_SAT_MAX = 20      # HSV S <= 20 means basically neutral grey
EMPTY_VAL_MIN = 140     # V >= 140 means not a dark gap

# Per-character class dedup cap.  Static UI: 30 cells per character is
# plenty diversity for a 128px classifier (BA avatars are sprites that
# render identically beyond the affinity number badge).
CLASS_MAX_PER_BUCKET = 30
# Perceptual-hash hamming distance threshold for treating two cells as
# "the same avatar variant".  <= 2 catches affinity-number-only diffs
# AND tiny JPEG edge bleed but lets through real visual changes (cafe
# event uniform variants etc.).
PHASH_HAMMING_THRESHOLD = 2
# Strict MD5 dedup is applied to __empty__ and __uncertain__ buckets —
# user is going to manually review those, no point keeping duplicates.


def md5_of(roi_bgr: np.ndarray) -> str:
    """Hash raw pixels (exact match required)."""
    return hashlib.md5(roi_bgr.tobytes()).hexdigest()


def phash_of(roi_bgr: np.ndarray) -> imagehash.ImageHash:
    """Perceptual hash — robust to JPEG / small visual diffs."""
    rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    return imagehash.phash(Image.fromarray(rgb))


def load_strips() -> tuple[list[dict], int]:
    if not REGIONS_JSON.exists():
        raise FileNotFoundError(f"strip config missing: {REGIONS_JSON}")
    data = json.loads(REGIONS_JSON.read_text(encoding="utf-8"))
    strips = data.get("strips", [])
    cpr = int(data.get("cells_per_room", 3))
    if not strips:
        raise ValueError(f"no strips in {REGIONS_JSON}")
    return strips, cpr


def strip_cells(strip: dict, cpr: int) -> list[tuple[float, float, float, float]]:
    """Slice a strip horizontally into N equal-width cells in normalised coords."""
    x1, y1, x2, y2 = strip["x1"], strip["y1"], strip["x2"], strip["y2"]
    width = x2 - x1
    out = []
    for i in range(cpr):
        cx1 = x1 + width * i / cpr
        cx2 = x1 + width * (i + 1) / cpr
        out.append((cx1, y1, cx2, y2))
    return out


def is_empty_cell(roi_bgr: np.ndarray) -> bool:
    """Heuristic empty-cell detection via HSV color stats."""
    if roi_bgr is None or roi_bgr.size == 0:
        return True
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    s_mean = float(np.mean(hsv[:, :, 1]))
    v_mean = float(np.mean(hsv[:, :, 2]))
    # Low saturation + medium-to-high brightness = empty grey slot
    return s_mean <= EMPTY_SAT_MAX and v_mean >= EMPTY_VAL_MIN


def main() -> int:
    strips, cpr = load_strips()
    cells_per_image = len(strips) * cpr
    print(f"[regions] {len(strips)} strips × {cpr} cells = {cells_per_image} cells/image")

    # Build cell coordinates once (normalised)
    cell_regions = []
    for si, strip in enumerate(strips):
        for ci, cell in enumerate(strip_cells(strip, cpr)):
            cell_regions.append((si, ci, cell))

    images = sorted(IMAGES_RAW.glob("*.jpg"))
    print(f"[input]  {len(images)} overlay screenshots from {IMAGES_RAW}")
    if not images:
        print(f"  no input images — run build_schedule_yolo_dataset.py first")
        return 1

    # Avatar matcher — load all 250 reference avatars
    sys.path.insert(0, str(REPO))
    from vision.avatar_matcher import AvatarMatcher
    matcher = AvatarMatcher(str(AVATAR_REF_DIR))
    candidates = sorted(p.stem for p in AVATAR_REF_DIR.glob("*.png"))
    print(f"[refs]   {len(candidates)} character avatars loaded")

    # Output dirs prepared lazily on first write
    rng = random.Random(SEED)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    bucket_counts: dict[str, int] = {}
    bucket_dropped: dict[str, int] = {}

    # Dedup state per bucket:
    #   character buckets → pHash list (hamming <= threshold = drop), cap N=30
    #   __empty__, __uncertain__ → MD5 set (exact match = drop, no cap)
    char_phashes: dict[str, list[imagehash.ImageHash]] = {}
    strict_md5: dict[str, set[str]] = {EMPTY_BUCKET: set(), UNCERTAIN_BUCKET: set()}

    total_cells = 0
    for img_idx, img_path in enumerate(images):
        if (img_idx + 1) % 100 == 0:
            print(f"  processing {img_idx + 1}/{len(images)}... "
                  f"kept={sum(bucket_counts.values())} "
                  f"dropped={sum(bucket_dropped.values())}")
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        run_tick = img_path.stem  # e.g. "run_20260516_234945_tick_0087"
        for si, ci, (nx1, ny1, nx2, ny2) in cell_regions:
            px1 = max(0, int(nx1 * w))
            py1 = max(0, int(ny1 * h))
            px2 = min(w, int(nx2 * w))
            py2 = min(h, int(ny2 * h))
            if px2 <= px1 or py2 <= py1:
                continue
            roi = bgr[py1:py2, px1:px2]
            if roi.size == 0:
                continue
            total_cells += 1

            # Bucket selection
            if is_empty_cell(roi):
                bucket = EMPTY_BUCKET
            else:
                name, score = matcher.match_avatar(roi, candidates)
                if name is not None and score >= STRONG_THRESHOLD:
                    bucket = name
                else:
                    bucket = UNCERTAIN_BUCKET

            # Dedup
            if bucket in strict_md5:
                # __empty__ / __uncertain__ — strict MD5 (user reviews)
                key = md5_of(roi)
                if key in strict_md5[bucket]:
                    bucket_dropped[bucket] = bucket_dropped.get(bucket, 0) + 1
                    continue
                strict_md5[bucket].add(key)
            else:
                # Character bucket — pHash hamming dedup + N=30 cap
                kept_phashes = char_phashes.setdefault(bucket, [])
                if len(kept_phashes) >= CLASS_MAX_PER_BUCKET:
                    bucket_dropped[bucket] = bucket_dropped.get(bucket, 0) + 1
                    continue
                ph = phash_of(roi)
                is_near_dup = any(
                    (ph - kept_ph) <= PHASH_HAMMING_THRESHOLD
                    for kept_ph in kept_phashes
                )
                if is_near_dup:
                    bucket_dropped[bucket] = bucket_dropped.get(bucket, 0) + 1
                    continue
                kept_phashes.append(ph)

            # Save
            split = "val" if rng.random() < VAL_RATIO else "train"
            out_dir = OUT_ROOT / split / bucket
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{run_tick}_s{si}_c{ci}.jpg"
            cv2.imwrite(str(out_path), roi)
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    print("\n[output] bucket counts (after dedup):")
    for name in sorted(bucket_counts, key=lambda b: -bucket_counts[b])[:30]:
        dropped = bucket_dropped.get(name, 0)
        print(f"  {bucket_counts[name]:5d} kept  {dropped:6d} dropped   {name}")
    if len(bucket_counts) > 30:
        print(f"  ... and {len(bucket_counts) - 30} more classes")
    print(f"\n[done] {OUT_ROOT}")
    print(f"  total cells scanned   : {total_cells}")
    print(f"  total cells kept      : {sum(bucket_counts.values())}")
    print(f"  total cells dropped   : {sum(bucket_dropped.values())}")
    print(f"  __empty__ kept        : {bucket_counts.get(EMPTY_BUCKET, 0)} "
          f"(dropped {bucket_dropped.get(EMPTY_BUCKET, 0)})")
    print(f"  __uncertain__ kept    : {bucket_counts.get(UNCERTAIN_BUCKET, 0)} "
          f"(dropped {bucket_dropped.get(UNCERTAIN_BUCKET, 0)})")
    print(f"  character classes     : "
          f"{len([k for k in bucket_counts if k not in (EMPTY_BUCKET, UNCERTAIN_BUCKET)])}")
    print(f"\nNext: review {OUT_ROOT}/train/{UNCERTAIN_BUCKET}/ (move into correct")
    print(f"      class folder).  Then add a yolo-cls config to train_yolo26.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
