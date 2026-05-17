"""Build a CLEAN-AS-GROUND-TRUTH cell dataset for the 全體課程表 grid.

Strategy decided 2026-05-17 with user + Gemini + Opus alignment:

  "5000 张干净的人工标注图，碾压 50000 张掺了 2% 水分的自动化图."

  Don't use AvatarMatcher (template+hist score) for pre-labeling — it
  generates false positives (similar-palette characters cross-match
  at ~0.55) and silently poisons the training set.  Instead:

    1. Walk all overlay screenshots.
    2. For each of the 27 cell crops:
       a. HSV color stats check → if mostly grey/white (S<=20 AND
          V>=140) → __empty__ bucket with strict MD5 dedup.
       b. Otherwise → __uncertain__ bucket with pHash dedup
          (hamming distance <= 2 means already kept).
    3. User manually creates character folders (Airi/, Wakamo/, etc.)
       and drags from __uncertain__ → correct class.
    4. Train YOLO26n-cls on the human-verified folder structure.

  Expected compression: 47K cells → 5-10K unique cells in __uncertain__
  plus a handful of __empty__ variants.  User reviews ~5-10K items in
  Windows Explorer (large-icon view, multi-select drag), maybe 1-2 hours
  of mindless work for a 100%-clean dataset.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import imagehash

REPO = Path(__file__).resolve().parents[1]
REGIONS_JSON = REPO / "data" / "schedule_avatar_regions.json"
IMAGES_RAW = REPO / "data" / "yolo_datasets" / "schedule_roster" / "images_raw"
OUT_ROOT = REPO / "data" / "yolo_datasets" / "schedule_cells"

UNCERTAIN_BUCKET = "__uncertain__"
EMPTY_BUCKET = "__empty__"
VAL_RATIO = 0.20
SEED = 42

# Empty-cell heuristic — flat grey-white with very low chroma.
EMPTY_SAT_MAX = 20
EMPTY_VAL_MIN = 140

# Perceptual hash threshold for "essentially the same crop".
# <= 2 catches affinity-number-only diffs + JPEG noise.
PHASH_HAMMING_THRESHOLD = 2


def md5_of(roi_bgr: np.ndarray) -> str:
    return hashlib.md5(roi_bgr.tobytes()).hexdigest()


def phash_of(roi_bgr: np.ndarray) -> imagehash.ImageHash:
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
    x1, y1, x2, y2 = strip["x1"], strip["y1"], strip["x2"], strip["y2"]
    width = x2 - x1
    out = []
    for i in range(cpr):
        out.append((x1 + width * i / cpr, y1, x1 + width * (i + 1) / cpr, y2))
    return out


def is_empty_cell(roi_bgr: np.ndarray) -> bool:
    if roi_bgr is None or roi_bgr.size == 0:
        return True
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    s_mean = float(np.mean(hsv[:, :, 1]))
    v_mean = float(np.mean(hsv[:, :, 2]))
    return s_mean <= EMPTY_SAT_MAX and v_mean >= EMPTY_VAL_MIN


def main() -> int:
    strips, cpr = load_strips()
    cell_regions = [
        (si, ci, cell)
        for si, strip in enumerate(strips)
        for ci, cell in enumerate(strip_cells(strip, cpr))
    ]
    print(f"[regions] {len(strips)} strips × {cpr} cells = {len(cell_regions)} cells/image")

    images = sorted(IMAGES_RAW.glob("*.jpg"))
    print(f"[input]  {len(images)} overlay screenshots from {IMAGES_RAW}")
    if not images:
        print(f"  no input images — run build_schedule_yolo_dataset.py first")
        return 1

    rng = random.Random(SEED)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    # Pre-create the two output buckets (val + train) so structure exists
    # even on early exit / kill.
    for split in ("train", "val"):
        for bucket in (EMPTY_BUCKET, UNCERTAIN_BUCKET):
            (OUT_ROOT / split / bucket).mkdir(parents=True, exist_ok=True)

    # Dedup state (global, not per-split — same image shouldn't appear
    # twice across train + val either).
    empty_md5_seen: set[str] = set()
    uncertain_phash_seen: list[imagehash.ImageHash] = []

    kept_empty = 0
    kept_uncertain = 0
    dropped_empty = 0
    dropped_uncertain = 0
    total_cells = 0

    for img_idx, img_path in enumerate(images):
        if (img_idx + 1) % 100 == 0:
            print(
                f"  {img_idx + 1}/{len(images)} imgs "
                f"| empty kept {kept_empty} (drop {dropped_empty}) "
                f"| uncertain kept {kept_uncertain} (drop {dropped_uncertain})"
            )
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        run_tick = img_path.stem
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

            if is_empty_cell(roi):
                key = md5_of(roi)
                if key in empty_md5_seen:
                    dropped_empty += 1
                    continue
                empty_md5_seen.add(key)
                bucket = EMPTY_BUCKET
                kept_empty += 1
            else:
                ph = phash_of(roi)
                is_near_dup = any(
                    (ph - seen_ph) <= PHASH_HAMMING_THRESHOLD
                    for seen_ph in uncertain_phash_seen
                )
                if is_near_dup:
                    dropped_uncertain += 1
                    continue
                uncertain_phash_seen.append(ph)
                bucket = UNCERTAIN_BUCKET
                kept_uncertain += 1

            split = "val" if rng.random() < VAL_RATIO else "train"
            out_path = OUT_ROOT / split / bucket / f"{run_tick}_s{si}_c{ci}.jpg"
            cv2.imwrite(str(out_path), roi)

    print()
    print(f"[done] {OUT_ROOT}")
    print(f"  total cells scanned   : {total_cells}")
    print(f"  __empty__     kept    : {kept_empty}    (dropped {dropped_empty})")
    print(f"  __uncertain__ kept    : {kept_uncertain}    (dropped {dropped_uncertain})")
    print()
    print("Next:")
    print(f"  1. Open Windows Explorer at:")
    print(f"       {OUT_ROOT / 'train' / UNCERTAIN_BUCKET}")
    print(f"  2. Set view to Large Icons.  Create character subfolders")
    print(f"     (Airi/, Wakamo/, ...) IN train/ and val/ siblings.")
    print(f"  3. Multi-select + drag from {UNCERTAIN_BUCKET}/ into the right class.")
    print(f"  4. When __uncertain__ is empty (or only weird edge-cases), say go.")
    print(f"     We then train yolo26n-cls on the verified folder structure.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
