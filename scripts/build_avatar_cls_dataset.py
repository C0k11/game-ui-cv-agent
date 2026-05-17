"""Build a YOLO-cls dataset for avatar classification.

User's insight (2026-05-17):
  data/captures/角色头像/                       (250 EN-named CG portraits)  → train (face-crop)
  data/captures/角色头像_crop_harvested_named/  (191 CN-named in-game crops) → train (drop-in)
  data/yolo_datasets/schedule_cells/<cls>/      (710 trajectory crops)       → val

Class names: canonicalize to ENGLISH (matching 角色头像/Wakamo.png style)
  - CN refs use student_name_map.json (CN→EN) to resolve
  - EN refs are already in canonical form
  - Trajectory crops in schedule_cells/ are already EN-named (built by
    build_harvest_cls_dataset.py which resolves via name_map)

Face crop for EN refs:
  Center 60% horizontally × center 60% vertically (configurable).
  BA's character portraits centre the face well enough that this
  catches the head consistently, dropping the decorative outer ring
  + weapons + background that aren't in deployment crops.

Augmentation per ref (mild, to multiply 1 → 3-4 train samples):
  - Original
  - +/- 5% scale (zoom in / zoom out via resize)
  - HSV jitter (±5% V) for slight brightness variance

Final layout:
  data/yolo_datasets/avatar_cls/
  ├── train/<Class>/{ref_en, ref_en_zoom_in, ref_en_zoom_out, ref_cn, ref_cn_*}.jpg
  └── val/<Class>/<trajectory_crop>.jpg
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
AVATAR_REF_EN = REPO / "data" / "captures" / "角色头像"
AVATAR_REF_CN = REPO / "data" / "captures" / "角色头像_crop_harvested_named"
TRAJ_CROPS = REPO / "data" / "yolo_datasets" / "schedule_cells"
OUT_ROOT = REPO / "data" / "yolo_datasets" / "avatar_cls"

# Face-crop fraction of EN reference image
FACE_CROP_X = (0.20, 0.80)   # keep middle 60% horizontally
FACE_CROP_Y = (0.10, 0.70)   # keep middle 60% vertically, biased up (BA art puts head upper)
TARGET_SIZE = 128            # final square side in pixels (matches train_yolo26.py imgsz)


_PAREN_TRANSLATE = str.maketrans({
    "（": "(",   # full-width open paren → half
    "）": ")",
    "【": "(",   # square bracket variants sometimes used by OCR
    "】": ")",
})


def _normalize_cn(name: str) -> str:
    return (name or "").translate(_PAREN_TRANSLATE).strip()


def load_name_map_reverse() -> dict[str, str]:
    """Chinese name → English name (canonical class).

    Both the map's keys and lookups go through _normalize_cn so the
    full-width vs half-width paren variants (e.g. 佳世子（禮服） vs
    佳世子(禮服)) resolve to the same English class.
    """
    if not NAME_MAP_JSON.exists():
        return {}
    raw = json.loads(NAME_MAP_JSON.read_text(encoding="utf-8"))
    return {_normalize_cn(k): v for k, v in raw.items()}


def crop_face_en(img_bgr: np.ndarray) -> np.ndarray:
    """Center-crop an EN reference portrait to the face region."""
    h, w = img_bgr.shape[:2]
    x1 = int(FACE_CROP_X[0] * w)
    x2 = int(FACE_CROP_X[1] * w)
    y1 = int(FACE_CROP_Y[0] * h)
    y2 = int(FACE_CROP_Y[1] * h)
    crop = img_bgr[y1:y2, x1:x2]
    return crop


def resize_square(img_bgr: np.ndarray, side: int = TARGET_SIZE) -> np.ndarray:
    """Resize to a square `side × side`, cropping center if non-square."""
    h, w = img_bgr.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    sq = img_bgr[y0:y0 + s, x0:x0 + s]
    return cv2.resize(sq, (side, side), interpolation=cv2.INTER_AREA)


def hsv_jitter(img_bgr: np.ndarray, dv: float) -> np.ndarray:
    """Shift V channel by dv (±0..1 scale)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.int32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + int(dv * 255), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def make_train_samples(canonical: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
    """Generate aug variants of one canonical reference crop."""
    base = resize_square(canonical)
    out = {f"{prefix}_orig": base}
    # Slight zoom in (crop center 90% then re-resize)
    h, w = base.shape[:2]
    pad = int(0.05 * h)
    if pad > 0:
        zin = base[pad:h - pad, pad:w - pad]
        out[f"{prefix}_zoom_in"] = cv2.resize(zin, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
    # Slight zoom out (pad with edge replicate, re-resize)
    zout = cv2.copyMakeBorder(base, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    out[f"{prefix}_zoom_out"] = cv2.resize(zout, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
    # HSV brightness +5% / -5%
    out[f"{prefix}_v_up"] = hsv_jitter(base, +0.05)
    out[f"{prefix}_v_dn"] = hsv_jitter(base, -0.05)
    return out


def main() -> int:
    name_map = load_name_map_reverse()
    print(f"[refs] CN→EN map: {len(name_map)} entries")

    # Build EN class list from the EN ref dir (canonical source of truth)
    en_classes = sorted(p.stem for p in AVATAR_REF_EN.glob("*.png"))
    print(f"[refs] EN reference set: {len(en_classes)} classes")

    if OUT_ROOT.exists():
        print(f"[clean] removing previous {OUT_ROOT}")
        shutil.rmtree(OUT_ROOT)

    train_root = OUT_ROOT / "train"
    val_root = OUT_ROOT / "val"
    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)

    # ── TRAIN: EN refs (face-cropped + aug) ──
    en_added = 0
    for en_name in en_classes:
        ref = imread_unicode((AVATAR_REF_EN / f"{en_name}.png"))
        if ref is None:
            continue
        face = crop_face_en(ref)
        samples = make_train_samples(face, prefix="en")
        cls_dir = train_root / en_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        for tag, img in samples.items():
            imwrite_unicode((cls_dir / f"{tag}.jpg"), img)
        en_added += 1
    print(f"[train] +{en_added} classes from EN refs (5 aug variants each)")

    # ── TRAIN: CN refs (drop-in, already correct format) ──
    cn_added = 0
    cn_unmapped: list[str] = []
    if AVATAR_REF_CN.exists():
        for cn_path in sorted(AVATAR_REF_CN.glob("*.png")):
            cn_name = cn_path.stem
            en_name = name_map.get(_normalize_cn(cn_name))
            if en_name is None or en_name not in set(en_classes):
                cn_unmapped.append(cn_name)
                continue
            ref = imread_unicode((cn_path))
            if ref is None:
                continue
            samples = make_train_samples(ref, prefix="cn")
            cls_dir = train_root / en_name
            cls_dir.mkdir(parents=True, exist_ok=True)
            for tag, img in samples.items():
                # Don't overwrite EN samples — CN samples get their own prefix
                imwrite_unicode((cls_dir / f"{tag}.jpg"), img)
            cn_added += 1
    print(f"[train] +{cn_added} CN refs mapped & added (5 aug variants each)")
    if cn_unmapped:
        print(f"[train] {len(cn_unmapped)} CN refs unmapped (no entry in name_map): "
              f"{cn_unmapped[:10]}{'...' if len(cn_unmapped) > 10 else ''}")

    # ── VAL: trajectory crops from schedule_cells (already EN-named) ──
    val_added_classes = 0
    val_added_samples = 0
    if TRAJ_CROPS.exists():
        # Source layout: schedule_cells/{train,val}/<class>/*.jpg
        for split_dir in ("train", "val"):
            src_root = TRAJ_CROPS / split_dir
            if not src_root.is_dir():
                continue
            for cls_dir in src_root.iterdir():
                if not cls_dir.is_dir():
                    continue
                cls = cls_dir.name
                if cls.startswith("__"):
                    continue  # __empty__ / __uncertain__ are not training targets
                if cls not in set(en_classes):
                    continue
                dst_dir = val_root / cls
                dst_dir.mkdir(parents=True, exist_ok=True)
                started_this_class = (dst_dir.iterdir().__next__() if any(dst_dir.iterdir()) else None) is None
                for jpg in cls_dir.glob("*.jpg"):
                    img = imread_unicode((jpg))
                    if img is None:
                        continue
                    resized = resize_square(img)
                    imwrite_unicode((dst_dir / jpg.name), resized)
                    val_added_samples += 1
                if started_this_class:
                    val_added_classes += 1
    print(f"[val]   {val_added_classes} classes / {val_added_samples} samples from trajectory crops")

    # Final stats
    final_train_classes = sorted(p.name for p in train_root.iterdir() if p.is_dir())
    final_val_classes = sorted(p.name for p in val_root.iterdir() if p.is_dir())
    print(f"\n[done] {OUT_ROOT}")
    print(f"  train classes : {len(final_train_classes)}")
    print(f"  val classes   : {len(final_val_classes)} (subset of train, only those with trajectory data)")
    print(f"  classes in val but missing train (should be empty): "
          f"{sorted(set(final_val_classes) - set(final_train_classes))[:5]}")
    print()
    print("Next: py scripts/train_yolo26.py schedule_cells")
    print(f"  (update data path to {OUT_ROOT} first)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
