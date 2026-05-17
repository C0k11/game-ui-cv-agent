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

import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

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

    for img_idx, img_path in enumerate(images):
        if (img_idx + 1) % 50 == 0:
            print(f"  processing {img_idx + 1}/{len(images)}...")
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
            # First: is it empty?
            if is_empty_cell(roi):
                bucket = EMPTY_BUCKET
            else:
                name, score = matcher.match_avatar(roi, candidates)
                if name is not None and score >= STRONG_THRESHOLD:
                    bucket = name
                else:
                    bucket = UNCERTAIN_BUCKET
            split = "val" if rng.random() < VAL_RATIO else "train"
            out_dir = OUT_ROOT / split / bucket
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{run_tick}_s{si}_c{ci}.jpg"
            cv2.imwrite(str(out_path), roi)
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    print("\n[output] bucket counts:")
    for name in sorted(bucket_counts, key=lambda b: -bucket_counts[b])[:25]:
        print(f"  {bucket_counts[name]:6d}  {name}")
    if len(bucket_counts) > 25:
        print(f"  ... and {len(bucket_counts) - 25} more classes")
    print(f"\n[done] {OUT_ROOT}")
    print(f"  total cells classified: {sum(bucket_counts.values())}")
    print(f"  __empty__ : {bucket_counts.get(EMPTY_BUCKET, 0)}")
    print(f"  __uncertain__ : {bucket_counts.get(UNCERTAIN_BUCKET, 0)}")
    print(f"  character classes : "
          f"{len([k for k in bucket_counts if k not in (EMPTY_BUCKET, UNCERTAIN_BUCKET)])}")
    print(f"\nNext: review {OUT_ROOT}/train/{UNCERTAIN_BUCKET}/ (move into correct")
    print(f"      class folder).  Then add a yolo-cls config to train_yolo26.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
