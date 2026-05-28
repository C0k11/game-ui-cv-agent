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

# Global resume flag — set by main() from --resume CLI arg
RESUME_FLAG: bool = False

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
    "static_ui": {
        # 141 UI-state classes harvested from labeling sessions 2026-05-18+.
        # Built by scripts/build_static_ui_dataset.py from every labeled
        # capture under data/raw_images/ (plus trajectory dirs with labels).
        # Static UI: BA sprites are pixel-identical at deploy = training,
        # so overfit by design — high epoch count, near-zero augmentation.
        "kind": "detect",
        "data": YOLO_ROOT / "dataset" / "static_ui_v1" / "data.yaml",
        "epochs": 250,
        "imgsz": 960,
        "batch": 16,
        "out_name": "static_ui_yolo26n",
    },
    "static_ui_v3": {
        # imgsz=1920 retrain to fix small-icon regression (red dot / yellow
        # dot / 青辉石 lost 30-50% mAP from v1→v2 because strict 5/18 data
        # reduced per-class samples for small targets).
        #
        # At imgsz=1920, an 8px source icon becomes 6.8px in network input —
        # right at P3 stride=8 detection floor.  At v2's 960 it was 3.4px,
        # below detection minimum.  batch=8 because VRAM scales (imgsz/960)².
        # Same data as v2 (only run_20260518_002646).
        "kind": "detect",
        "data": YOLO_ROOT / "dataset" / "static_ui_v1" / "data.yaml",
        "epochs": 250,
        "imgsz": 1920,
        "batch": 8,
        "out_name": "static_ui_v3_yolo26n",
        "patience": 50,
    },
    "head_detector": {
        # Single-class 角色头像 detector.  Replaces sliding-window-with-
        # classifier-confidence approach in schedule popup eval with a true
        # YOLO bbox.  Training data:
        #   - seed: 14 manual frames from pre-trim backup
        #   - auto: trajectory schedule frames auto-labeled by avatar_cls v2
        #     where top1 conf >= 0.85 (treats classifier as teacher)
        # Single-class detection is much easier than multi-class; emoticon
        # yolo26n got mAP 0.99 on similar task with 170 train.
        "kind": "detect",
        "data": YOLO_ROOT / "dataset" / "head_detector_v1" / "data.yaml",
        "epochs": 200,
        "imgsz": 960,
        "batch": 16,
        "out_name": "head_detector_yolo26n",
    },
    "static_ui_v4": {
        # v3 (imgsz=1920 batch=8) regressed top-bar classes (信用点/体力/青辉石)
        # because halved batch size = halved per-epoch iterations for those
        # already-sparse classes (5-7 train instances each).
        # v4 attempt 1 (batch=12): cuDNN engine error — not OOM, FP16 algo
        # heuristic failed at that specific tensor shape.
        # v4 attempt 2 (batch=10): conservative middle ground vs v3's 8.
        "kind": "detect",
        "data": YOLO_ROOT / "dataset" / "static_ui_v1" / "data.yaml",
        "epochs": 250,
        "imgsz": 1920,
        "batch": 10,
        "out_name": "static_ui_v4_yolo26n",
        "patience": 50,
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
    "fused_avatar_26m": {
        # Fused multi-class avatar DETECTOR: simultaneous bbox + character ID
        # in one model, replaces the current 2-stage (head_detector → avatar_cls).
        # 250 classes is fine-grained — yolo26n's 2.4M params can't discriminate
        # all characters reliably (~10k params/class).  yolo26m's ~20M params
        # = ~80k params/class, much more discriminative capacity.
        #
        # Data: manual user labels across 5 UI contexts (MomoTalk/cafe/schedule/
        # 学生/battle) + synthetic composites (角色头像_crop refs pasted onto
        # real schedule popup backgrounds at static_ui-detected room slots).
        # Target: ~13k samples across 250 classes ≈ 50/class average.
        "kind": "detect",
        "data": YOLO_ROOT / "dataset" / "fused_avatar_v1" / "data.yaml",
        "base": "yolo26m.pt",
        "epochs": 200,
        "imgsz": 960,
        "batch": 16,
        "out_name": "fused_avatar_yolo26m",
        "patience": 60,
    },
    "fused_avatar_26x": {
        # v3 配置 (2026-05-20 训完, best 0.68 但中期过拟合) — 留作历史 baseline
        "kind": "detect",
        "data": YOLO_ROOT / "dataset" / "fused_avatar_v1" / "data.yaml",
        "base": "yolo26x.pt",
        "epochs": 200,
        "imgsz": 960,
        "batch": 8,
        "out_name": "fused_avatar_yolo26x",
        "patience": 0,
        "weight_decay": 0.001,
        "dropout": 0.1,
        "mosaic": 0.7,
        "close_mosaic": 10,
        "mixup": 0.10,
        "copy_paste": 0.10,
    },
    "fused_avatar_26x_v4": {
        # v4: warm-start from v3 best.pt + lighter aug (v3 教训).
        # 目标: cafe/momotalk/schedule 维持 88-95% recall, battle/tactical
        # 从 30% → 65-75%, 整体 mAP 0.78-0.85.
        "kind": "detect",
        "data": YOLO_ROOT / "dataset" / "fused_avatar_v1" / "data.yaml",
        # v3 best.pt 当 warm-start (复用 252 类特征)
        "base": str(YOLO_ROOT / "runs" / "fused_avatar_yolo26x" / "weights" / "best.pt"),
        "epochs": 100,           # warm-start 不需要 200
        "imgsz": 960,
        "batch": 8,
        "out_name": "fused_avatar_yolo26x_v4",
        "patience": 30,          # 加回 early stop, v3 过拟合是教训
        # Regularization 回保守 (v3 重正则反而过拟合, 因为 aug 太狠)
        "weight_decay": 0.0005,
        "dropout": 0.0,
        # Aug 大幅降低 — v3 学到的 lesson
        "mosaic": 0.3,           # 0.7 → 0.3
        "close_mosaic": 5,       # 100 epoch 里最后 5 关 mosaic
        "mixup": 0.0,            # 直接移除 (细粒度致命毒药)
        "copy_paste": 0.0,       # 移除
        # 低 LR 保护 warm-start 特征
        "lr0": 0.003,            # 默认 0.01 的 1/3
        # 翻转 (BA 角色无方向区别, 加倍数据)
        "fliplr": 0.5,
    },
    "ui_yolo26m_v1": {
        # Static UI detector — first proper train.
        # Schema: 447 classes (145 in actual use after audit), 4 themes:
        #   - 顶栏 info (清辉石/体力/信用点/红点/黄点/...)
        #   - 通用按钮 (确认/取消/X/返回/领取/...)
        #   - 弹窗触发 (X 关闭 / 弹窗内 buttons)
        #   - context-specific (cafe invite / craft buttons / momotalk markers / ...)
        #
        # Data: 3 train dirs (run_20260521_103956_distinct + 2 补录) + _ui_val_pool.
        # Minority classes oversampled via symlink (scripts/oversample_minority_classes.py).
        #
        # Aug rationale for static UI:
        #   - UI is essentially overfit-tolerant (train ≈ test distribution)
        #   - mosaic 0.5 = middle ground, gives context diversity
        #   - mixup / copy_paste / fliplr / rotate ALL OFF (UI has direction +
        #     no "translucent button fade-in" reality)
        #   - hsv jitter + scale/translate KEPT for cross-resolution robustness
        "kind": "detect",
        "data": YOLO_ROOT / "dataset" / "ui_v1" / "data.yaml",
        "base": "yolo26m.pt",     # COCO from-scratch, NOT warm-start
        "epochs": 100,
        "patience": 30,
        "imgsz": 960,
        "batch": 16,              # 26m lighter than 26x, can double batch
        "out_name": "ui_yolo26m_v1",
        "lr0": 0.01,              # default
        "weight_decay": 0.0005,
        "dropout": 0.0,
        # AUG — UI-specific
        "mosaic": 0.5,
        "close_mosaic": 10,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "fliplr": 0.0,            # ✗ UI has direction (X always top-right)
        "flipud": 0.0,            # ✗
        "degrees": 0.0,           # ✗ UI orthogonal only
        "perspective": 0.0,       # ✗
        "hsv_h": 0.015,
        "hsv_s": 0.3,
        "hsv_v": 0.3,
        "scale": 0.5,             # ★ key for cross-aspect-ratio robustness
        "translate": 0.1,
    },
    "avatar_cls_v2": {
        # Combined-source classifier: ~250-class BA student recognition.
        # Train sources (per class, when available):
        #   1. Trajectory cafe-invite crops (29 classes, ~25 high-quality samples)
        #   2. 角色头像 CG portrait face-crop + 4 augmentations
        #   3. 角色头像_crop in-game style ref + 4 augmentations
        # Val: 角色头像_crop_harvested_named (CN-named, mapped to EN, ~35 classes)
        #
        # v2 first attempt dropped trajectory data — regressed to 16% on traj val.
        # This version keeps it (~25 trajectory + ~10 ref per class for 29 chars,
        # ~10 ref-only per class for the other 220).  patience=80 because
        # 250-class convergence is slower than 29-class.
        "kind": "classify",
        "data": REPO_ROOT / "data" / "yolo_datasets" / "avatar_cls_v2",
        "epochs": 200,
        "imgsz": 224,
        "batch": 64,
        "out_name": "avatar_cls_v2_yolo26n",
        "patience": 80,
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

    # Per-config base weight override (e.g. "yolo26m.pt" / "yolo26x.pt").
    # Bare name lets ultralytics auto-fetch if not in repo root.
    base_override = cfg.get("base")
    if base_override:
        base = base_override
    else:
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
    # If --resume, load last.pt and resume (epoch + lr scheduler state preserved)
    # Ultralytics resume mode: ONLY pass resume=True to .train(), all other args
    # come from the saved args.yaml beside last.pt. Passing extra kwargs makes
    # ultralytics silently fall back to "fresh training with last.pt as base".
    last_pt = YOLO_ROOT / "runs" / cfg["out_name"] / "weights" / "last.pt"
    if RESUME_FLAG and last_pt.exists():
        print(f"  RESUME from: {last_pt}")
        model = YOLO(str(last_pt))
        results = model.train(resume=True)
        best = YOLO_ROOT / "runs" / cfg["out_name"] / "weights" / "best.pt"
        if best.exists():
            print(f"  done: {best}")
            return best
        print("  warning: best.pt not found after training")
        return None

    if RESUME_FLAG:
        print(f"  --resume requested but {last_pt} missing → starting fresh")
    model = YOLO(base)
    train_kwargs = dict(
        data=str(data_arg),
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        device=0,
        workers=4,
        patience=cfg.get("patience", 30),
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
        # Per-config aug / regularization overrides (e.g. fused_avatar_26x)
        for k in ("mosaic", "mixup", "copy_paste", "close_mosaic",
                  "weight_decay", "dropout", "hsv_h", "hsv_s", "hsv_v",
                  "degrees", "fliplr", "flipud", "scale", "translate", "lr0",
                  "perspective"):
            if k in cfg:
                train_kwargs[k] = cfg[k]
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
    ap.add_argument("--resume", action="store_true",
                    help="Resume from last.pt (preserves epoch + LR scheduler state)")
    args = ap.parse_args()

    global RESUME_FLAG
    RESUME_FLAG = args.resume
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
