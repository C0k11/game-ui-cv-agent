"""Train YOLO26n on the existing BA UI datasets.

Replaces:
  - full.pt / expanded.pt (31 UI + 282 incl. avatars)  →  ba_ui_yolo26n.pt
  - emoticon.pt (cafe headpat bubbles)                 →  emoticon_yolo26n.pt
  - battle_heads.pt (currently OCR-only, no dataset)   →  TODO when dataset

YOLO26n claims: NMS-free dual-head, no DFL, ~43% faster CPU inference,
MuSGD optimizer for small-model convergence.  For static-UI detection
on BA (NEVER rotates, NEVER mirrors) we keep augmentation minimal:
no rotation / flip, mild HSV jitter, half-mosaic, no mixup.

Hyperparameters chosen for 24G RTX 4090 + static UI:
  - imgsz=960   (BA's lobby OCR-friendly; previous 256 was too small
                 to see 5-15px lobby badge dots)
  - epochs=200  (small dataset; static UI converges fast, early-stop
                 via patience=30)
  - batch=16    (room to spare on 4090; can bump to 32 if VRAM allows)
  - hsv_h=0.01  (BA palette is fixed)
  - degrees=0   (UI never rotates)
  - fliplr=0, flipud=0
  - mosaic=0.5  (helps with cropped contexts)
  - mixup=0     (UI doesn't need synthetic blends)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
ML_CACHE = Path("D:/Project/ml_cache")
YOLO_ROOT = ML_CACHE / "models" / "yolo"

# Base weight — already in repo root + data/models
BASE_WEIGHT_CANDIDATES = [
    REPO_ROOT / "yolo26n.pt",
    REPO_ROOT / "data" / "models" / "yolo26n.pt",
]
# Classifier weight — kind=="classify" configs prefer this.  If not
# found locally, ultralytics will fetch from its model registry on
# first use (`YOLO("yolo26n-cls.pt")`).
CLS_BASE_WEIGHT_CANDIDATES = [
    REPO_ROOT / "yolo26n-cls.pt",
    REPO_ROOT / "data" / "models" / "yolo26n-cls.pt",
    ML_CACHE / "models" / "yolo" / "yolo26n-cls.pt",
]

# Training configs.  Each entry produces one trained .pt.
#
# kind="detect" (default) uses yolo26n.pt and a data.yaml.
# kind="classify" uses yolo26n-cls.pt and a folder path (each subfolder
# is a class, images directly inside).
TRAIN_CONFIGS = {
    "expanded": {
        "kind": "detect",
        "data": YOLO_ROOT / "dataset" / "expanded" / "data.yaml",
        "epochs": 200,
        "imgsz": 960,
        "batch": 16,
        "out_name": "expanded_yolo26n",
    },
    "emoticon_v2": {
        "kind": "detect",
        "data": YOLO_ROOT / "dataset" / "emoticon_v2" / "data.yaml",
        "epochs": 150,
        "imgsz": 640,
        "batch": 32,
        "out_name": "emoticon_yolo26n",
    },
    "full": {
        # 31-class UI only (no avatars) — smaller, faster, fine for
        # pure UI element detection.  Falls back here if expanded is
        # too slow / hits a class imbalance issue.
        "kind": "detect",
        "data": YOLO_ROOT / "dataset" / "full" / "data.yaml",
        "epochs": 200,
        "imgsz": 960,
        "batch": 32,
        "out_name": "ba_ui_yolo26n_31",
    },
    "schedule_cells": {
        # YOLO classifier on cafe-invite trajectory crops.
        # 80/20 split within data/yolo_datasets/schedule_cells/ (built
        # by build_harvest_cls_dataset.py with PHASH_THRESHOLD=0 so
        # affinity-number variants stay as separate training samples).
        # imgsz=224 (224×224 is yolo cls standard, gives the model
        # enough pixels to learn avatar features; 128 was too small).
        "kind": "classify",
        "data": REPO_ROOT / "data" / "yolo_datasets" / "schedule_cells",
        "epochs": 100,
        "imgsz": 224,
        "batch": 64,
        "out_name": "avatar_cls_yolo26n",
    },
    "avatar_cls": {
        # Curated subset of schedule_cells.  Built by build_avatar_cls_dataset.py
        # which drops __empty__ / __uncertain__ and any class with <3 trajectory
        # samples (the unsplittable ones).  Adds CN refs (if name_map matches)
        # as bonus train samples in the same in-game distribution.
        "kind": "classify",
        "data": REPO_ROOT / "data" / "yolo_datasets" / "avatar_cls",
        "epochs": 100,
        "imgsz": 224,
        "batch": 64,
        "out_name": "avatar_cls_yolo26n",
    },
}


def find_base_weight(kind: str = "detect") -> str:
    """Return base weight name/path for the given task kind.

    For detect: looks for local yolo26n.pt copies, else falls back to
    the bare model name so ultralytics fetches it.
    For classify: looks for local yolo26n-cls.pt copies, else falls back
    to the bare model name "yolo26n-cls.pt".
    """
    candidates = (
        CLS_BASE_WEIGHT_CANDIDATES if kind == "classify"
        else BASE_WEIGHT_CANDIDATES
    )
    default_name = "yolo26n-cls.pt" if kind == "classify" else "yolo26n.pt"
    for p in candidates:
        if p.exists():
            return str(p)
    return default_name  # ultralytics auto-downloads


def train_one(config_name: str, dry_run: bool = False) -> Optional[Path]:
    """Train one model.  Returns path to best.pt or None on dry_run."""
    cfg = TRAIN_CONFIGS[config_name]
    kind = cfg.get("kind", "detect")
    data_arg = cfg["data"]
    # For detect, data is a yaml file.  For classify, it's a folder.
    if not Path(data_arg).exists():
        print(f"  data missing: {data_arg}")
        return None

    base = find_base_weight(kind)
    print(f"\n==== TRAIN {config_name} ({kind}) ====")
    print(f"  base:    {base}")
    print(f"  data:    {data_arg}")
    print(f"  epochs:  {cfg['epochs']}")
    print(f"  imgsz:   {cfg['imgsz']}")
    print(f"  batch:   {cfg['batch']}")
    print(f"  out:     {cfg['out_name']}")
    if dry_run:
        print("  (dry run — skipping actual training)")
        return None

    from ultralytics import YOLO
    model = YOLO(base)
    train_kwargs = dict(
        data=str(data_arg),
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        device=0,
        workers=4,
        patience=30,
        project=str(YOLO_ROOT / "runs"),
        name=cfg["out_name"],
        exist_ok=True,
        save=True,
        save_period=-1,
        verbose=True,
    )
    if kind == "detect":
        # Static-UI augmentation: minimal
        train_kwargs.update(
            degrees=0.0,
            fliplr=0.0,
            flipud=0.0,
            hsv_h=0.01,
            hsv_s=0.3,
            hsv_v=0.3,
            mosaic=0.5,
            mixup=0.0,
        )
    elif kind == "classify":
        # Classifier on cropped avatars: MODERATE augmentation.
        # Train and val are DIFFERENT frames of the same character
        # (different lighting, sub-pixel jitter, JPEG variance) — we
        # need generalization, not memorization.  Previous "zero aug"
        # setup produced top1=7% (train loss → 0.08 while val loss
        # climbed to 10 — textbook overfit).
        #
        # Keep DISABLED:
        #   - fliplr (face flip is wrong)
        #   - flipud (avatars never flip vertically)
        #   - degrees (avatars never rotate)
        #   - mosaic / mixup (not useful for portrait classification)
        # Keep ENABLED:
        #   - hsv jitter (lobby tint shifts slightly between sessions)
        #   - slight scale / translate (crop position varies ±2-3px)
        #   - erasing (occlusion robustness, helps with affinity-number
        #     badges that partially overlay some avatars)
        train_kwargs.update(
            degrees=0.0,
            fliplr=0.0,
            flipud=0.0,
            hsv_h=0.01,
            hsv_s=0.2,
            hsv_v=0.2,
            erasing=0.2,
            scale=0.1,
            translate=0.05,
            mosaic=0.0,
            mixup=0.0,
        )
    results = model.train(**train_kwargs)
    best = YOLO_ROOT / "runs" / cfg["out_name"] / "weights" / "best.pt"
    if best.exists():
        print(f"  done: {best}")
        return best
    print("  warning: best.pt not found after training")
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "config",
        nargs="?",
        default="all",
        choices=list(TRAIN_CONFIGS.keys()) + ["all"],
        help="Which dataset to train on (default: all)",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print plan without training")
    args = ap.parse_args()

    targets = list(TRAIN_CONFIGS.keys()) if args.config == "all" else [args.config]
    results = []
    for cfg_name in targets:
        try:
            out = train_one(cfg_name, dry_run=args.dry_run)
            results.append((cfg_name, out))
        except Exception as exc:
            print(f"  ERROR training {cfg_name}: {exc}")
            results.append((cfg_name, None))

    print("\n==== SUMMARY ====")
    for name, out in results:
        status = "OK" if out else "FAIL/SKIP"
        print(f"  {status:10s} {name:15s} {out or ''}")
    return 0 if all(out for _, out in results) else 1


if __name__ == "__main__":
    sys.exit(main())
