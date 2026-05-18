"""Build avatar_cls v2: expanded character coverage from BA's local capture refs.

Three source dirs (all under data/captures/):
  角色头像/                      250 EN-named CG portraits (404×456, oil-painting style)
  角色头像_crop/                 267 EN-named face crops  (~54×59, in-game style) ✨
  角色头像_crop_harvested_named/ 191 CN-named harvested   (~54×59, in-game style) → VAL

Architecture:
  - Train: 角色头像 (face-cropped) + 角色头像_crop, with augmentation
  - Val: 角色头像_crop_harvested_named (mapped CN→EN via student_name_map.json)
  - Auto-drop train images byte-identical to any val image (clean leak prevention)

This expands the trained class set from 29 (current trajectory-only) to ~250
BA characters, enabling recognition across cafe invite / MomoTalk / character
select UIs — wherever a BA student face appears at ~50-100px crop.

Naming convention: master uses EN names (Wakamo, Akari_(New_Year)).  CN-named
val files get mapped via student_name_map.json.  CN refs that can't be mapped
to an existing EN class are dumped to __unmapped__ for inspection (not used).
"""
from __future__ import annotations

import hashlib
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
CG_DIR = REPO / "data" / "captures" / "角色头像"
CROP_DIR = REPO / "data" / "captures" / "角色头像_crop"
HARVEST_DIR = REPO / "data" / "captures" / "角色头像_crop_harvested_named"
NAME_MAP = REPO / "data" / "student_name_map.json"
TRAJ_DIR = REPO / "data" / "yolo_datasets" / "avatar_cls"  # v1 dataset (trajectory crops)
OUT_ROOT = REPO / "data" / "yolo_datasets" / "avatar_cls_v2"

TARGET_SIZE = 224           # YOLO-cls standard; classifier upsizes 54px crops
CG_FACE_X = (0.20, 0.80)    # CG portrait face-crop bounds (center 60%)
CG_FACE_Y = (0.10, 0.70)    # biased up (BA art puts head higher)


# ── Unicode-safe IO (Windows + Chinese filenames) ──────────────────────────

def imread_u(p: Path) -> Optional[np.ndarray]:
    try:
        return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_u(p: Path, img: np.ndarray) -> bool:
    try:
        ext = p.suffix or ".jpg"
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            return False
        with open(p, "wb") as f:
            f.write(buf.tobytes())
        return True
    except Exception:
        return False


def file_hash(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


# ── CN paren normalization (CN refs use full-width parens) ─────────────────

_PAREN = str.maketrans({"（": "(", "）": ")", "【": "(", "】": ")"})


def norm_cn(s: str) -> str:
    return (s or "").translate(_PAREN).strip()


def load_cn_to_en() -> Dict[str, str]:
    if not NAME_MAP.exists():
        return {}
    raw = json.loads(NAME_MAP.read_text(encoding="utf-8"))
    return {norm_cn(k): v for k, v in raw.items()}


# ── Image transforms ───────────────────────────────────────────────────────

def face_crop_cg(img: np.ndarray) -> np.ndarray:
    """Center-face crop from a 404×456 CG portrait."""
    h, w = img.shape[:2]
    x1, x2 = int(CG_FACE_X[0] * w), int(CG_FACE_X[1] * w)
    y1, y2 = int(CG_FACE_Y[0] * h), int(CG_FACE_Y[1] * h)
    return img[y1:y2, x1:x2]


def resize_square(img: np.ndarray, side: int = TARGET_SIZE) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    sq = img[y0:y0 + s, x0:x0 + s]
    return cv2.resize(sq, (side, side), interpolation=cv2.INTER_AREA)


def hsv_jitter(img: np.ndarray, dv: float) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + int(dv * 255), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def augment(canonical: np.ndarray, prefix: str) -> Dict[str, np.ndarray]:
    """Generate 5 augmented variants of one canonical reference crop."""
    base = resize_square(canonical)
    out = {f"{prefix}_orig": base}
    h, w = base.shape[:2]
    pad = max(1, int(0.06 * h))
    # zoom in
    zin = base[pad:h - pad, pad:w - pad]
    out[f"{prefix}_zin"] = cv2.resize(zin, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
    # zoom out
    zout = cv2.copyMakeBorder(base, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    out[f"{prefix}_zout"] = cv2.resize(zout, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
    # brightness +/- 5%
    out[f"{prefix}_vup"] = hsv_jitter(base, +0.05)
    out[f"{prefix}_vdn"] = hsv_jitter(base, -0.05)
    return out


# ── Main pipeline ──────────────────────────────────────────────────────────

def main() -> int:
    print(f"[refs] CG dir:      {CG_DIR.exists()} ({len(list(CG_DIR.glob('*.png'))) if CG_DIR.exists() else 0})")
    print(f"[refs] crop dir:    {CROP_DIR.exists()} ({len(list(CROP_DIR.glob('*.png'))) if CROP_DIR.exists() else 0})")
    print(f"[refs] harvest dir: {HARVEST_DIR.exists()} ({len(list(HARVEST_DIR.glob('*.png'))) if HARVEST_DIR.exists() else 0})")
    cn_to_en = load_cn_to_en()
    print(f"[refs] CN→EN map:   {len(cn_to_en)} entries")

    # ── 1. Build val set from harvest (CN-named) ──
    # Group val by EN class.  Note byte-hashes for leak prevention.
    val_files: Dict[str, List[Path]] = {}
    val_hashes: set = set()
    unmapped_cn: List[str] = []
    for p in sorted(HARVEST_DIR.glob("*.png")):
        cn = norm_cn(p.stem)
        en = cn_to_en.get(cn)
        if not en:
            unmapped_cn.append(cn)
            continue
        val_files.setdefault(en, []).append(p)
        val_hashes.add(file_hash(p))
    print(f"[val] {sum(len(v) for v in val_files.values())} val images across "
          f"{len(val_files)} EN classes (mapped from CN)")
    print(f"[val] unmapped CN refs (skipped): {len(unmapped_cn)}")

    # ── 2. Build train: CG + crop, but DROP any image whose bytes are in val ──
    en_classes: set = {p.stem for p in CG_DIR.glob("*.png")} | {p.stem for p in CROP_DIR.glob("*.png")}
    print(f"[train] EN classes available: {len(en_classes)}")

    # Clean output dir
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    (OUT_ROOT / "train").mkdir(parents=True)
    (OUT_ROOT / "val").mkdir(parents=True)

    train_count = 0
    train_dropped_leak = 0
    classes_with_train = 0
    for en in sorted(en_classes):
        cls_dir = OUT_ROOT / "train" / en
        cls_dir.mkdir(parents=True, exist_ok=True)
        n_for_class = 0
        # CG face-crop
        cg_path = CG_DIR / f"{en}.png"
        if cg_path.exists():
            cg_img = imread_u(cg_path)
            if cg_img is not None:
                # Leak check (unlikely for CG since they're 404×456, but be safe)
                if file_hash(cg_path) not in val_hashes:
                    face = face_crop_cg(cg_img)
                    for tag, im in augment(face, "cg").items():
                        imwrite_u(cls_dir / f"{tag}.jpg", im)
                        n_for_class += 1
                else:
                    train_dropped_leak += 1
        # In-game crop
        crop_path = CROP_DIR / f"{en}.png"
        if crop_path.exists():
            crop_img = imread_u(crop_path)
            if crop_img is not None:
                # Critical leak check: this image MIGHT be byte-identical
                # to a CN-named val image
                if file_hash(crop_path) not in val_hashes:
                    for tag, im in augment(crop_img, "crop").items():
                        imwrite_u(cls_dir / f"{tag}.jpg", im)
                        n_for_class += 1
                else:
                    train_dropped_leak += 1
        if n_for_class > 0:
            classes_with_train += 1
            train_count += n_for_class
        else:
            cls_dir.rmdir()  # remove empty class

    print(f"[train] from refs: {train_count} samples across {classes_with_train} classes "
          f"(dropped {train_dropped_leak} CG/crop refs due to val leak)")

    # ── 2.5. Merge in trajectory crops (high-quality real BA frames) ──
    # These are the gold-standard training data — actual cafe-invite crops
    # captured during BA play.  Adding them keeps v1's 86%+ trajectory val
    # accuracy while v2 expands coverage to 250+ chars.
    traj_added = 0
    traj_classes_new = set()
    if TRAJ_DIR.is_dir():
        for split in ("train", "val"):
            src = TRAJ_DIR / split
            if not src.is_dir():
                continue
            for cls_dir in src.iterdir():
                if not cls_dir.is_dir():
                    continue
                cls = cls_dir.name
                # Add to train (we'll use harvest as val instead, so traj val also folds into train)
                dst = OUT_ROOT / "train" / cls
                dst.mkdir(parents=True, exist_ok=True)
                if not (OUT_ROOT / "train" / cls).is_dir() or len(list((OUT_ROOT / "train" / cls).iterdir())) == 0:
                    traj_classes_new.add(cls)
                for jpg in cls_dir.glob("*.jpg"):
                    img = imread_u(jpg)
                    if img is None:
                        continue
                    sq = resize_square(img)
                    imwrite_u(dst / f"traj_{split}_{jpg.name}", sq)
                    traj_added += 1
    print(f"[train] +trajectory: {traj_added} more samples ({len(traj_classes_new)} new classes)")

    # ── 3. Emit val ──
    val_count = 0
    val_classes_with_train: set = set()
    for en, paths in val_files.items():
        if not (OUT_ROOT / "train" / en).is_dir():
            # Class has val but no train — skip (model wouldn't have learned it)
            continue
        val_classes_with_train.add(en)
        cls_dir = OUT_ROOT / "val" / en
        cls_dir.mkdir(parents=True, exist_ok=True)
        for p in paths:
            img = imread_u(p)
            if img is None:
                continue
            sq = resize_square(img)
            imwrite_u(cls_dir / f"{p.stem}.jpg", sq)
            val_count += 1
    print(f"[val] emitted {val_count} val samples across {len(val_classes_with_train)} classes")

    # Final tally
    train_classes = sorted(p.name for p in (OUT_ROOT / "train").iterdir() if p.is_dir())
    val_classes = sorted(p.name for p in (OUT_ROOT / "val").iterdir() if p.is_dir())
    train_only = set(train_classes) - set(val_classes)
    print()
    print(f"[done] {OUT_ROOT}")
    print(f"  train classes: {len(train_classes)}  (of which {len(train_only)} have NO val)")
    print(f"  val   classes: {len(val_classes)}")
    print()
    print("Class-name registry preview (sorted train):")
    for c in train_classes[:5]:
        print(f"  ✓ {c}")
    print(f"  ... ({len(train_classes)} total)")
    print()
    print("Next: py scripts/train_yolo26.py avatar_cls   (after updating its data path)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
