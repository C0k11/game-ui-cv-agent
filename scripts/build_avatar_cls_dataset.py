"""Build a YOLO-cls dataset for avatar classification.

PIVOT 2026-05-17 (after 3 failed training rounds at 9-18% top1):

Previous strategy (train=CG refs + val=trajectory) failed because:
  - 250 EN classes are in `角色头像/` but only 35 of them have val data
    → 215 ghost classes with zero val signal poison the model
  - For the 35 meaningful classes, train was 5 CG face-crops vs 20-40
    in-game circular avatars in val. Domain gap (oil-painting style vs
    UI render) is too wide for a 47-class classifier to bridge with
    5 train samples per class.
  - 191 CN refs in `角色头像_crop_harvested_named/` overlap with val
    by only 3 classes because the two harvest passes captured
    different characters.

New strategy: TRAIN ON THE SAME DISTRIBUTION AS VAL.
  - Source: data/yolo_datasets/schedule_cells/{train,val}/<cls>/*.jpg
    (already-split harvest crops from build_harvest_cls_dataset.py)
  - For each class with >=3 samples total, re-do 80/20 split into
    avatar_cls/{train,val}/<cls>/
  - Skip `__empty__` / `__uncertain__` (they're the unlabelled
    overflow from build_schedule_cells_dataset.py)
  - Skip empty class folders (CN-only labels the user never populated)
  - BONUS: CN ref images from `角色头像_crop_harvested_named/` get
    added to TRAIN only, for classes that map to an existing class
    name via name_map.json.  Same domain (in-game crop), free signal.
  - CG portraits in `角色头像/` are DROPPED entirely — different domain
    is more harm than help.

All images resized to TARGET_SIZE × TARGET_SIZE (224×224 matches
yolo cls standard + train_yolo26.py imgsz).
"""
from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


def imread_unicode(path) -> np.ndarray | None:
    """cv2.imread that handles Unicode paths on Windows."""
    try:
        with open(path, "rb") as f:
            data = f.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path, img: np.ndarray) -> bool:
    """cv2.imwrite that handles Unicode paths on Windows."""
    try:
        path = str(path)
        ext = "." + path.rsplit(".", 1)[-1] if "." in path else ".jpg"
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        with open(path, "wb") as f:
            f.write(buf.tobytes())
        return True
    except Exception:
        return False


REPO = Path(__file__).resolve().parents[1]
NAME_MAP_JSON = REPO / "data" / "student_name_map.json"
AVATAR_REF_CN = REPO / "data" / "captures" / "角色头像_crop_harvested_named"
TRAJ_CROPS = REPO / "data" / "yolo_datasets" / "schedule_cells"
OUT_ROOT = REPO / "data" / "yolo_datasets" / "avatar_cls"

TARGET_SIZE = 224         # matches train_yolo26.py imgsz
VAL_RATIO = 0.20          # 80/20 re-split (override the existing one)
SEED = 42
MIN_SAMPLES_PER_CLASS = 3 # skip classes with <3 samples (can't split)

# Buckets / folder names that are NOT character classes
SKIP_FOLDERS = {"__empty__", "__uncertain__"}


_PAREN_TRANSLATE = str.maketrans({
    "（": "(",
    "）": ")",
    "【": "(",
    "】": ")",
})


def _normalize_cn(name: str) -> str:
    return (name or "").translate(_PAREN_TRANSLATE).strip()


def load_name_map() -> dict[str, str]:
    """Chinese name → English name (normalized keys)."""
    if not NAME_MAP_JSON.exists():
        return {}
    raw = json.loads(NAME_MAP_JSON.read_text(encoding="utf-8"))
    return {_normalize_cn(k): v for k, v in raw.items()}


def resize_square(img_bgr: np.ndarray, side: int = TARGET_SIZE) -> np.ndarray:
    """Resize to square `side × side`, center-crop if non-square.

    For BA's in-game crops:
      - harvest tool produces ~square crops (112×112, 192×192) — direct
        resize keeps full content
      - schedule overlay crops are ~2:1 wide — center crop loses the
        avatar's right side; but we filter those out in the source
        anyway (they live in __uncertain__).
    """
    h, w = img_bgr.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    sq = img_bgr[y0:y0 + s, x0:x0 + s]
    return cv2.resize(sq, (side, side), interpolation=cv2.INTER_AREA)


def collect_class_samples(src_root: Path) -> dict[str, list[Path]]:
    """Walk schedule_cells/{train,val}/<cls>/*.jpg → {cls: [paths]}."""
    out: dict[str, list[Path]] = {}
    for split in ("train", "val"):
        split_dir = src_root / split
        if not split_dir.is_dir():
            continue
        for cls_dir in split_dir.iterdir():
            if not cls_dir.is_dir():
                continue
            cls = cls_dir.name
            if cls in SKIP_FOLDERS:
                continue
            for jpg in cls_dir.glob("*.jpg"):
                out.setdefault(cls, []).append(jpg)
    return out


def main() -> int:
    name_map = load_name_map()
    print(f"[refs] CN→EN map: {len(name_map)} entries")

    # ── Step 1: collect all trajectory crops by class ──
    if not TRAJ_CROPS.exists():
        print(f"[err] {TRAJ_CROPS} does not exist — run build_harvest_cls_dataset.py first")
        return 1

    class_samples = collect_class_samples(TRAJ_CROPS)
    # Drop classes below MIN_SAMPLES_PER_CLASS
    too_small = [c for c, files in class_samples.items() if len(files) < MIN_SAMPLES_PER_CLASS]
    for c in too_small:
        class_samples.pop(c)

    print(f"[scan] {len(class_samples)} classes with ≥{MIN_SAMPLES_PER_CLASS} trajectory samples")
    print(f"[scan] dropped {len(too_small)} small classes")
    print(f"[scan] total trajectory samples: {sum(len(v) for v in class_samples.values())}")

    if not class_samples:
        print("[err] no usable classes — nothing to train on")
        return 1

    # ── Step 2: 80/20 split per class, fresh ──
    if OUT_ROOT.exists():
        print(f"[clean] removing previous {OUT_ROOT}")
        shutil.rmtree(OUT_ROOT)
    train_root = OUT_ROOT / "train"
    val_root = OUT_ROOT / "val"
    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    n_train = 0
    n_val = 0
    class_counts: dict[str, tuple[int, int]] = {}  # cls -> (train, val)

    for cls, files in sorted(class_samples.items()):
        files = list(files)
        rng.shuffle(files)
        n_val_for_cls = max(1, int(len(files) * VAL_RATIO))
        val_files = files[:n_val_for_cls]
        train_files = files[n_val_for_cls:]

        train_cls_dir = train_root / cls
        val_cls_dir = val_root / cls
        train_cls_dir.mkdir(parents=True, exist_ok=True)
        val_cls_dir.mkdir(parents=True, exist_ok=True)

        for src in train_files:
            img = imread_unicode(src)
            if img is None:
                continue
            sq = resize_square(img)
            imwrite_unicode(train_cls_dir / src.name, sq)
            n_train += 1
        for src in val_files:
            img = imread_unicode(src)
            if img is None:
                continue
            sq = resize_square(img)
            imwrite_unicode(val_cls_dir / src.name, sq)
            n_val += 1

        class_counts[cls] = (len(train_files), len(val_files))

    print(f"\n[traj] train {n_train} / val {n_val}")

    # ── Step 3: BONUS — add CN refs (same in-game distribution) to TRAIN ──
    # Each CN ref is one PNG named with a CN character name.  Resolve to
    # canonical class via:  CN -> EN via name_map  OR  direct CN match.
    # Only add to classes that already exist in train (have val signal).
    cn_added = 0
    cn_skipped_unmapped = 0
    cn_skipped_no_class = 0
    if AVATAR_REF_CN.exists():
        existing_classes = set(class_counts.keys())
        for cn_path in sorted(AVATAR_REF_CN.glob("*.png")):
            cn_stem = _normalize_cn(cn_path.stem)
            # Try direct match first (in case the trajectory uses CN folder name)
            if cn_stem in existing_classes:
                target_cls = cn_stem
            else:
                mapped = name_map.get(cn_stem)
                if mapped and mapped in existing_classes:
                    target_cls = mapped
                elif mapped:
                    cn_skipped_no_class += 1
                    continue
                else:
                    cn_skipped_unmapped += 1
                    continue
            img = imread_unicode(cn_path)
            if img is None:
                continue
            sq = resize_square(img)
            # Tag with prefix so it doesn't collide with trajectory crops
            out_name = f"cnref_{cn_path.stem}.jpg"
            imwrite_unicode(train_root / target_cls / out_name, sq)
            cn_added += 1

    print(f"[cnref] +{cn_added} CN refs added to train")
    print(f"[cnref] skipped {cn_skipped_unmapped} unmapped (no name_map entry)")
    print(f"[cnref] skipped {cn_skipped_no_class} mapped but class has no val data")

    # ── Summary ──
    final_train_classes = sorted(p.name for p in train_root.iterdir() if p.is_dir())
    final_val_classes = sorted(p.name for p in val_root.iterdir() if p.is_dir())
    final_train_count = sum(len(list((train_root / c).glob("*.jpg"))) for c in final_train_classes)
    final_val_count = sum(len(list((val_root / c).glob("*.jpg"))) for c in final_val_classes)

    print(f"\n[done] {OUT_ROOT}")
    print(f"  train: {len(final_train_classes)} classes / {final_train_count} samples")
    print(f"  val:   {len(final_val_classes)} classes / {final_val_count} samples")
    # Sanity: train and val class sets should match exactly
    missing_val = set(final_train_classes) - set(final_val_classes)
    missing_train = set(final_val_classes) - set(final_train_classes)
    if missing_val:
        print(f"  WARN: {len(missing_val)} train classes have no val: {sorted(missing_val)[:5]}")
    if missing_train:
        print(f"  WARN: {len(missing_train)} val classes have no train: {sorted(missing_train)[:5]}")
    print()
    print("Per-class breakdown:")
    print(f"  {'class':<32s}{'train':>7s}{'val':>5s}")
    for cls in sorted(class_counts, key=lambda c: -class_counts[c][0]):
        tr, vl = class_counts[cls]
        try:
            print(f"  {cls:<32s}{tr:>7d}{vl:>5d}")
        except UnicodeEncodeError:
            print(f"  {'<cjk>':<32s}{tr:>7d}{vl:>5d}")
    print()
    print("Next: py scripts/train_yolo26.py avatar_cls")
    return 0


if __name__ == "__main__":
    sys.exit(main())
