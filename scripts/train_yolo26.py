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

# Training configs.  Each entry produces one trained .pt.
TRAIN_CONFIGS = {
    "expanded": {
        "data": YOLO_ROOT / "dataset" / "expanded" / "data.yaml",
        "epochs": 200,
        "imgsz": 960,
        "batch": 16,
        "out_name": "expanded_yolo26n",
    },
    "emoticon_v2": {
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
        "data": YOLO_ROOT / "dataset" / "full" / "data.yaml",
        "epochs": 200,
        "imgsz": 960,
        "batch": 32,
        "out_name": "ba_ui_yolo26n_31",
    },
}


def find_base_weight() -> Path:
    for p in BASE_WEIGHT_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"yolo26n.pt base weight not found in any of: {BASE_WEIGHT_CANDIDATES}"
    )


def train_one(config_name: str, dry_run: bool = False) -> Optional[Path]:
    """Train one model.  Returns path to best.pt or None on dry_run."""
    cfg = TRAIN_CONFIGS[config_name]
    data_yaml = cfg["data"]
    if not data_yaml.exists():
        print(f"  data yaml missing: {data_yaml}")
        return None

    base = find_base_weight()
    print(f"\n==== TRAIN {config_name} ====")
    print(f"  base:    {base}")
    print(f"  data:    {data_yaml}")
    print(f"  epochs:  {cfg['epochs']}")
    print(f"  imgsz:   {cfg['imgsz']}")
    print(f"  batch:   {cfg['batch']}")
    print(f"  out:     {cfg['out_name']}")
    if dry_run:
        print("  (dry run — skipping actual training)")
        return None

    from ultralytics import YOLO
    model = YOLO(str(base))
    results = model.train(
        data=str(data_yaml),
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        device=0,
        workers=4,
        patience=30,
        # Static-UI augmentation: minimal
        degrees=0.0,
        fliplr=0.0,
        flipud=0.0,
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.3,
        mosaic=0.5,
        mixup=0.0,
        # Output location
        project=str(YOLO_ROOT / "runs"),
        name=cfg["out_name"],
        exist_ok=True,
        # Save best + last
        save=True,
        save_period=-1,
        # Verbose enough to track progress in log
        verbose=True,
    )
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
