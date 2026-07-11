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
    "fused_avatar_26x_v5": {
        # v5: warm-start from v4 best_manual + battle_cards 技能牌 synth(多底图+灰白aug).
        # 目标: 维持 252 角色(cafe/编成 ~0.99) + 新增战斗技能牌灰彩识别.
        # ⛔ 训练时并行 `py scripts/manual_fitness_watcher.py --run fused_avatar_yolo26x_v5`,
        #    最终取 best_manual.pt — synth 主导(synth:real~12:1), best.pt 会被 synth val 带偏.
        "kind": "detect",
        "data": YOLO_ROOT / "dataset" / "fused_avatar_v2" / "data.yaml",
        "base": str(YOLO_ROOT / "runs" / "fused_avatar_yolo26x_v4" / "weights" / "best_manual.pt"),
        "epochs": 100,
        "imgsz": 960,
        "batch": 8,
        "out_name": "fused_avatar_yolo26x_v5",
        "patience": 30,
        "weight_decay": 0.0005,
        "dropout": 0.0,
        "mosaic": 0.3,
        "close_mosaic": 5,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "lr0": 0.002,            # v4 的 2/3, 保护已学 252 角色不被技能牌数据冲垮
        "fliplr": 0.5,
    },
    "fused_avatar_26x_v6": {
        # v6: v5 同配方, 数据集 build 三处根治(2026-06-07): ①load_character_names 排 UI-B
        #     → 纯 252 头像(不再混 60 UI cls, nc 312→252, 与 v4 base 完全对齐) ②灰牌明度修正
        #     (实测真实灰牌是暗灰 dim0.45-0.80, 旧 v5 用 g*0.5+110 提亮成亮灰白→真实暗灰牌漏35%)
        #     ③val 泄漏根治(VAL_SOURCE_RUNS 排除 173604, 防同帧既 train 又 val).
        # 目标: 252 角色不退(cafe/编成 ~0.96) + 彩牌稳 + 灰牌 recall 从 v5 的 0.557 大升.
        # ⛔ 用 best.pt 即可: v2 val 85% 真实主导, watcher 非必须(memory 踩坑6); 独占 GPU 防 OOM.
        "kind": "detect",
        "data": YOLO_ROOT / "dataset" / "fused_avatar_v2" / "data.yaml",
        "base": str(YOLO_ROOT / "runs" / "fused_avatar_yolo26x_v4" / "weights" / "best_manual.pt"),
        "epochs": 100,
        "imgsz": 960,
        "batch": 8,
        "out_name": "fused_avatar_yolo26x_v6",
        "patience": 30,
        "save_period": 5,        # 存中间 epoch: v6 val synth占63%(manual去虚高后180)→best.pt偏synth拟合, 训完用 manual val(cafe/编成+battle)选真实峰
        "weight_decay": 0.0005,
        "dropout": 0.0,
        "mosaic": 0.3,
        "close_mosaic": 5,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "lr0": 0.002,
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
    "ui_yolo26m_v2": {
        # v2 retrain: user feedback (2026-05-28) that咖啡厅入口/邮件箱/确认键/
        # 一次領取 等关键 UI 在 v1 上 conf 太低 (some 12-19 frames only).
        # Re-oversampled to target=60 (script bumped 770 → 7400+ copies),
        # AND set all per-frame augmentation to ZERO to let the model learn
        # UI text glyphs + spatial context cleanly.  Mosaic destroys spatial
        # context (4 frames cropped together) — user explicitly said
        # "不要马赛克以及其他干扰, ui字体以及特征就是要学的干净也要有空间上下文理解".
        "kind": "detect",
        "data": YOLO_ROOT / "dataset" / "ui_v1" / "data.yaml",
        "base": "yolo26m.pt",
        "epochs": 150,
        "patience": 40,
        "imgsz": 960,
        "batch": 16,
        "out_name": "ui_yolo26m_v2",
        "lr0": 0.01,
        "weight_decay": 0.0005,
        "dropout": 0.0,
        # AUG — ALL OFF (clean spatial context for UI text/icon learning)
        "mosaic": 0.0,           # ✗ destroys spatial context
        "close_mosaic": 0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "fliplr": 0.0,
        "flipud": 0.0,
        "degrees": 0.0,
        "perspective": 0.0,
        "hsv_h": 0.0,            # ✗ UI palette is fixed
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "scale": 0.0,            # ✗ UI sizes are fixed
        "translate": 0.0,        # ✗ UI positions are fixed
    },
    "ui_yolo26m_v3": {
        # v3: lobby 入口 cls 改标文字(底栏8+MomoTalk+每日领奖+任务大厅) + 邮件箱
        # 保图标, 全部 oversample target=200 (入口16→200=12x曝光)。5 训练目录
        # (加 run_20260529_000756 新capture)。
        # from COCO 重训 (NOT warm-start — 续训会让大按钮退化, 已验证)。
        # patience=30 合理早停 (val 噪声但够 stop 信号)。cache=ram(~22GB) +
        # workers=0 (Windows cache+workers>0 崩)。batch=12 (24GB 极限, 别OOM)。
        "kind": "detect",
        "data": YOLO_ROOT / "dataset" / "ui_v1" / "data.yaml",
        "base": "yolo26m.pt",
        "epochs": 150,
        "patience": 30,
        "imgsz": 960,
        "batch": 12,
        "out_name": "ui_yolo26m_v3",
        # cache 决策史 (14562帧):
        #  - cache=ram 需21GB, 空闲RAM仅13.7GB → 被跳过 → on-the-fly 1.1s/it
        #  - cache=False+workers8 → 仍每ep decode 14562次, GPU 86%等数据, 16min/ep
        #  - cache=disk: 21GB存D盘.npy(210GB free够), 避免重复decode, 读.npy快。
        #    首次建~15min, 之后 ~5min/ep。workers=0 (disk读单进程够, 不引入崩溃)。
        "cache": "disk",
        # disk 读 .npy 有 IO 延迟, workers=0 单进程扛不住 → GPU 饿(18%util)。
        # workers=8 多进程并行读盘喂满 GPU。(cache=disk+workers 不崩, 崩的是
        # cache=ram+workers 的 pickle)。
        "workers": 8,
        "lr0": 0.01,
        "weight_decay": 0.0005,
        "dropout": 0.0,
        "mosaic": 0.0, "close_mosaic": 0, "mixup": 0.0, "copy_paste": 0.0,
        "fliplr": 0.0, "flipud": 0.0, "degrees": 0.0, "perspective": 0.0,
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0, "scale": 0.0, "translate": 0.0,
    },
    "ui_yolo26m_v4": {
        # v4 修 v3 弱检测的全部根因:
        #  ① 修 487 corrupt 标签(build sanitize 截5字段) ② 删 cls92(不可学大区域)
        #  ③ 加 run_20260529_123209(581 YOLO预标, 含 momotalk) + synth 合成帧
        #  ④ 开空间 aug 治"过度依赖位置/邻居"过拟合(侧栏图标随布局偏移就崩):
        #     translate/scale=位置+尺度不变性, mosaic=换邻居/上下文,
        #     copy_paste=治稀有 cls, close_mosaic 最后10ep关mosaic干净微调(保清晰)
        #  ⑤ flip/rotate 保持 0 —— UI 有左右/朝向语义(左切换↔右切换), 翻转会搞反标签
        # from COCO (非 warm-start)。
        "kind": "detect",
        "data": YOLO_ROOT / "dataset" / "ui_v1" / "data.yaml",
        "base": "yolo26m.pt",
        "epochs": 150,
        "patience": 30,
        "imgsz": 960,
        "batch": 12,
        "out_name": "ui_yolo26m_v4",
        "cache": "disk",
        "workers": 8,
        "lr0": 0.01,
        "weight_decay": 0.0005,
        "dropout": 0.0,
        # ── 空间增强 ON (治过拟合 / 位置依赖, v3 的核心病) ──
        "mosaic": 0.5, "close_mosaic": 10, "copy_paste": 0.3, "mixup": 0.0,
        "scale": 0.3, "translate": 0.1,
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.3,   # hsv_v: 立绘背景明暗多样
        # ── 几何翻转/旋转保持 0 (UI 左右/朝向语义不可翻) ──
        "fliplr": 0.0, "flipud": 0.0, "degrees": 0.0, "perspective": 0.0,
    },
    "ui_yolo26m_v5": {
        # v5 = v4 成功后的正式版 (warm-start 微调):
        #  ① 数据集 ui_v2: 旧 train 去重 (16157→2377 唯一, 砍 200x oversample 过拟合源)
        #     + run_20260531_110516(1052 真实) + run_20260531_143201(37 商店) + bond synth
        #  ② cls92 战术大赛对战选择区域 恢复 (v4 误删; v5 给 arena 用区域选对手)
        #  ③ nc=451 (新增 450 选择购买); 薄类适度 oversample 到30 (450→50), 不再 200x
        #  ④ warm-start from v4 best_real.pt: backbone/neck 特征继承, cls头因nc450→451重学;
        #     lr0=0.005 (half) 保护 v4 特征不被大lr冲垮
        #  ⑤ aug 照搬 v4 (修过拟合关键); val=_ui_val_pool 仅 early-stop 机制 (盲弱类, 真实
        #     验收看 dashboard 目检). patience=60 = 不 early-stop (盲 val 不可信), 跑满60.
        "kind": "detect",
        "data": YOLO_ROOT / "dataset" / "ui_v2" / "data.yaml",
        "base": str(YOLO_ROOT / "runs" / "ui_yolo26m_v4" / "weights" / "best_real.pt"),
        "epochs": 60,
        "patience": 60,
        "imgsz": 960,
        "batch": 12,
        "out_name": "ui_yolo26m_v5",
        "cache": "disk",
        "workers": 8,
        "lr0": 0.005,
        "weight_decay": 0.0005,
        "dropout": 0.0,
        "mosaic": 0.5, "close_mosaic": 10, "copy_paste": 0.3, "mixup": 0.0,
        "scale": 0.3, "translate": 0.1,
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.3,
        "fliplr": 0.0, "flipud": 0.0, "degrees": 0.0, "perspective": 0.0,
    },
    "ui_yolo26m_v6": {
        # v6 = v5 + emoticon 折叠 (+ 飞轮补弱 cls, 当其 source 入库后一并重训):
        #  ① 数据集 ui_v2 重建 (nc 451→452): 新增 cls451 Emoticon_Action — 把独立的
        #     emoticon_yolo26n 并进 ui 模型, pipeline 每个 cafe tick 少跑一次 YOLO。
        #     emoticon 帧经 build_emoticon_ui_source.py teacher 重标 (现役 ui_v5 当老师
        #     补回 cafe UI chrome 框, 避免 200 帧把 收益/邀请卷 等弱 cls 训成负样本)。
        #  ② warm-start from v5 best_real.pt: backbone/neck 继承, cls 头因 nc451→452
        #     重学新增行 (同 v4→v5 的 450→451, registry 已验证可行)。lr0=0.005 护特征。
        #  ③ aug/epochs/val 照搬 v5 (修过拟合关键)。patience=60=跑满不 early-stop
        #     (盲 _ui_val_pool 不含 emoticon, mAP 不可信); emoticon 真实验收 =
        #     dashboard / live cafe 帧目检 (见 money_safety: 上线前飞轮帧实测)。
        #  ④ 飞轮: 补弱 cls 时把新 source 加进 build_ui_v2.py REAL_SOURCES → 重建 → 训。
        "kind": "detect",
        "data": YOLO_ROOT / "dataset" / "ui_v2" / "data.yaml",
        "base": str(YOLO_ROOT / "runs" / "ui_yolo26m_v5" / "weights" / "best_real.pt"),
        "epochs": 60,
        "patience": 60,
        "imgsz": 960,
        "batch": 12,
        "out_name": "ui_yolo26m_v6",
        "cache": "disk",
        "workers": 8,
        "lr0": 0.005,
        "weight_decay": 0.0005,
        "dropout": 0.0,
        "mosaic": 0.5, "close_mosaic": 10, "copy_paste": 0.3, "mixup": 0.0,
        "scale": 0.3, "translate": 0.1,
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.3,
        "fliplr": 0.0, "flipud": 0.0, "degrees": 0.0, "perspective": 0.0,
    },
    "ui_yolo26m_v7": {
        # v7 = 纯 UI+emoticon (头像归 fused v6, build_ui_v2 _keep_ui_lines drop 143-394).
        #  补真实弱类: 制造入口(v5 flywheel 0!)/活动类452-454/灰色领取 + 折叠 emoticon451.
        #  warm v5 (m 同arch 全继承; cls头 451→455 重学新增4类). 数据: 新2run(193003+140123
        #  两背景真实)+旧真实+_emoticon_v2, 砍全部synth(伪背景毒, v6c实锤污染lobby入口).
        #  ⭐val=flywheel 477 纯真实(头像已过滤) → best.pt 可信(不像 fused synth-bias);
        #  save_period 保险, 训完仍 eval flywheel by-cls 核弱类不退. patience30 早停(真实val).
        "kind": "detect",
        "data": YOLO_ROOT / "dataset" / "ui_v2" / "data.yaml",
        "base": str(YOLO_ROOT / "runs" / "ui_yolo26m_v5" / "weights" / "best_real.pt"),
        "epochs": 70,
        "patience": 30,
        "save_period": 5,
        "imgsz": 960,
        "batch": 12,
        "out_name": "ui_yolo26m_v7",
        "cache": "disk",
        "workers": 8,
        "lr0": 0.005,
        "weight_decay": 0.0005,
        "dropout": 0.0,
        "mosaic": 0.5, "close_mosaic": 10, "copy_paste": 0.3, "mixup": 0.0,
        "scale": 0.3, "translate": 0.1,
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.3,
        "fliplr": 0.0, "flipud": 0.0, "degrees": 0.0, "perspective": 0.0,
    },
    "ui_yolo26m_v8": {
        # v8 (2026-06-10) = v7 + 全技能飞轮素材(v8queue 2075 + 用户手标批量扫荡127):
        #  新类455-468(批量扫荡dialog/战斗完成/立即前往, cls头 455→469 重学)。
        #  补强: 制造入口544/任务大厅入口542/双倍三倍617+76(hub badge模板锚定,
        #  破位置先验)/绿勾1569/短篇网格/剧情战斗结算。红黄点已HSV仲裁全清洗
        #  (源头+queue 635处) — 位置先验毒源已断。emoticon 仍弱样本(5框), 折叠
        #  目标0.99继续等。val = flywheel477 + _val_v8flywheel38(整run抽防泄漏)。
        #  配方完全复刻 v7 成功版(hsv_h/s=0 保点色相信号)。
        "kind": "detect",
        "data": YOLO_ROOT / "dataset" / "ui_v2" / "data.yaml",
        "base": str(YOLO_ROOT / "runs" / "ui_yolo26m_v7" / "weights" / "best_real.pt"),
        "epochs": 70,
        "patience": 30,
        "save_period": 5,
        "imgsz": 960,
        "batch": 12,
        "out_name": "ui_yolo26m_v8",
        "cache": "disk",
        "workers": 8,
        "lr0": 0.005,
        "weight_decay": 0.0005,
        "dropout": 0.0,
        "mosaic": 0.5, "close_mosaic": 10, "copy_paste": 0.3, "mixup": 0.0,
        "scale": 0.3, "translate": 0.1,
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.3,
        "fliplr": 0.0, "flipud": 0.0, "degrees": 0.0, "perspective": 0.0,
    },
    "ui_yolo26m_v8b": {
        # v8b = v8 best(ep26) 短程微调 + _arrow_boost(旧款箭头帧×1287重编码副本):
        #  v8 在 ep15→20 间用旧款左右切换换了双倍三倍(风格灾难遗忘, momo新款47/47
        #  vs 旧款15/238)。增压旧款把锚点拉回, 其余能力(双倍三倍0.51/红黄点0修复/
        #  新类455-468)从 best.pt 继承。验收: 箭头≥0.9 且 双倍三倍≥0.45 且 红黄混淆=0。
        "kind": "detect",
        "data": YOLO_ROOT / "dataset" / "ui_v2" / "data.yaml",
        "base": str(YOLO_ROOT / "runs" / "ui_yolo26m_v8" / "weights" / "best.pt"),
        "epochs": 14,
        "patience": 14,
        "save_period": 2,
        "imgsz": 960,
        "batch": 12,
        "out_name": "ui_yolo26m_v8b",
        "cache": "disk",
        "workers": 8,
        "lr0": 0.002,
        "weight_decay": 0.0005,
        "dropout": 0.0,
        "mosaic": 0.3, "close_mosaic": 4, "copy_paste": 0.3, "mixup": 0.0,
        "scale": 0.3, "translate": 0.1,
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.3,
        "fliplr": 0.0, "flipud": 0.0, "degrees": 0.0, "perspective": 0.0,
    },
    "ui_yolo26m_v9": {
        # v9 (2026-06-11) = v8b + 全天live干净帧18池(~1,493帧用户手标, review零问题)
        #  + 老训练集审计修复(238框auto-add回源池, HSV仲裁拦32可疑点色)。
        #  新类469-473(战术大赛商店/货币/能量饮料, cls头 469→474 重学)。
        #  靶子(val=冻结回归考卷, v8b PHANTOM 1,356 → 看降幅): 旧皮箭头651 /
        #  hub活動進行中ribbon 350(452新增576实例) / dialog调暗大厅 / 451折叠
        #  (新增747实例, emoticon已退役出live管线 — v9的451就是摸头唯一来源) /
        #  格黑娜vs阿拜多斯混淆(87帧解药)。配方复刻 v8 成功版(hsv_h/s=0 保点色)。
        "kind": "detect",
        "data": YOLO_ROOT / "dataset" / "ui_v2" / "data.yaml",
        "base": str(YOLO_ROOT / "runs" / "ui_yolo26m_v8b" / "weights" / "best_real.pt"),
        "epochs": 70,
        "patience": 30,
        "save_period": 5,
        "imgsz": 960,
        "batch": 12,
        "out_name": "ui_yolo26m_v9",
        "cache": "disk",
        "workers": 8,
        "lr0": 0.005,
        "weight_decay": 0.0005,
        "dropout": 0.0,
        "mosaic": 0.5, "close_mosaic": 10, "copy_paste": 0.3, "mixup": 0.0,
        "scale": 0.3, "translate": 0.1,
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.3,
        "fliplr": 0.0, "flipud": 0.0, "degrees": 0.0, "perspective": 0.0,
    },
    "ui_yolo26m_v10": {
        # v10 (2026-06-12) = v9 warm + 今日整链live素材(run_20260612_191319 313帧 +
        #  run_20260612_chainlive 452帧, ⚠ 必须 DASHBOARD 人审后才解注释入 REAL_SOURCES)。
        #  补强靶子: shop确认弹窗/战术大赛商店/能量饮料/批量扫荡 normal页/课程表popout。
        #  ⚠ 前置: build_ui_v2 REAL_SOURCES 末两行解注释; 配方复刻 v9 成功版。
        "kind": "detect",
        "data": YOLO_ROOT / "dataset" / "ui_v2" / "data.yaml",
        "base": str(YOLO_ROOT / "runs" / "ui_yolo26m_v9" / "weights" / "best.pt"),
        "epochs": 70,
        "patience": 30,
        "save_period": 5,
        "imgsz": 960,
        "batch": 12,
        "out_name": "ui_yolo26m_v10",
        "cache": "disk",
        "workers": 8,
        "lr0": 0.005,
        "weight_decay": 0.0005,
        "dropout": 0.0,
        "mosaic": 0.5, "close_mosaic": 10, "copy_paste": 0.3, "mixup": 0.0,
        "scale": 0.3, "translate": 0.1,
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.3,
        "fliplr": 0.0, "flipud": 0.0, "degrees": 0.0, "perspective": 0.0,
    },
    "ui_yolo26m_v11": {
        # v11 (2026-06-13) = v10 warm + 今日全天 live 素材(11 个 _clean 池/~1028 帧:
        #  整链/arena_shop/schedule/bounty/jfd step_mode walk + autonomous 尾链)。
        #  v10 预标 + 用户 dashboard 人审(无可见假阳, 仅 cafe 漏摸1)。配方复刻 v10 成功版。
        #  ⚠ cafe 451 摸头弱本版不修(今日飞轮无 cafe 摸头帧, 老 _emoticon_v2 200 帧已在源)
        #  — 451 强化待明天 cafe walk 多录摸头帧; 眼下 inference conf 0.40 stopgap 兜着。
        "kind": "detect",
        "data": YOLO_ROOT / "dataset" / "ui_v2" / "data.yaml",
        "base": str(YOLO_ROOT / "runs" / "ui_yolo26m_v10" / "weights" / "best.pt"),
        "epochs": 70,
        "patience": 30,
        "save_period": 5,
        "imgsz": 960,
        "batch": 12,
        "out_name": "ui_yolo26m_v11",
        "cache": "disk",
        "workers": 8,
        "lr0": 0.005,
        "weight_decay": 0.0005,
        "dropout": 0.0,
        "mosaic": 0.5, "close_mosaic": 10, "copy_paste": 0.3, "mixup": 0.0,
        "scale": 0.3, "translate": 0.1,
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.3,
        "fliplr": 0.0, "flipud": 0.0, "degrees": 0.0, "perspective": 0.0,
    },
    "ui_yolo26m_v12": {
        # v12 (2026-06-14) = v11 warm + 今日9个审过的池(任务大厅live + arena_shop walk).
        #  ⭐核心补强: run_20260614_205540 用户手录战术大赛商店素材(245帧) → 把 cls469
        #  (战术大赛商店未选中tab)从饿着的27实例大幅补足, 根治 arena_shop locate 弱检测
        #  (虽已用固定位/swipe代码绕开, 数据补上更稳)。配方复刻 v11/v10 成功版。
        #  ⚠ cafe 451 摸头本版仍不修(今日无cafe摸头帧); 待明天cafe walk录摸头帧再强化。
        "kind": "detect",
        "data": YOLO_ROOT / "dataset" / "ui_v2" / "data.yaml",
        "base": str(YOLO_ROOT / "runs" / "ui_yolo26m_v11" / "weights" / "best.pt"),
        "epochs": 70,
        "patience": 30,
        "save_period": 5,
        "imgsz": 960,
        "batch": 12,
        "out_name": "ui_yolo26m_v12",
        "cache": "disk",
        "workers": 8,
        "lr0": 0.005,
        "weight_decay": 0.0005,
        "dropout": 0.0,
        "mosaic": 0.5, "close_mosaic": 10, "copy_paste": 0.3, "mixup": 0.0,
        "scale": 0.3, "translate": 0.1,
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.3,
        "fliplr": 0.0, "flipud": 0.0, "degrees": 0.0, "perspective": 0.0,
    },
    "battle_yolo26n_v2": {
        # 战斗模型 v2 (2026-07-09) = battle_heads(老v8n 4cls) 的 yolo26 重生。
        #  数据: run_battle_material_20260708 用户手标 133 帧 → battle_v2 (114/19),
        #  7 类 = 我方/敌方(新身份类, combat AI 2.0 检测层) + 战斗HUD 5 类
        #  (暂停/三倍速/自动开/自动关/胜利 — 与 ui 模型有意重复: 战斗高频循环
        #  单模型拿 AUTO gate, 不等 5s 主 tick)。
        #  ⭐26n@640 = battle_lock_upgrade_plan 定案: 2 类身份不缺容量, 高频循环
        #  实测 98 FPS@640(4090 half, 2026-07-09 bench); 26x 反而掉到 60。
        #  ⚠fliplr=0 铁律: 我方永远左侧/敌方右侧, 翻转破坏方向语义。
        #  hsv 开得比 UI 大(h0.015/s0.3/v0.4): 3D 战场光效/VFX 色变打底,
        #  真 VFX 合成 augmentation 留 Phase 2。copy_paste=0(小人重叠假样本)。
        "kind": "detect",
        "data": YOLO_ROOT / "dataset" / "battle_v2" / "data.yaml",
        "base": str(YOLO_ROOT / "yolo26n.pt"),
        "epochs": 150,
        "patience": 40,
        "save_period": 10,
        "imgsz": 640,
        "batch": 32,
        "out_name": "battle_yolo26n_v2",
        "cache": True,          # 133帧 RAM cache, 绝不 disk(v13 爆盘教训不适用但统一心智)
        "workers": 8,
        "mosaic": 0.5, "close_mosaic": 15, "copy_paste": 0.0, "mixup": 0.0,
        "scale": 0.4, "translate": 0.1,
        "hsv_h": 0.015, "hsv_s": 0.3, "hsv_v": 0.4,
        "fliplr": 0.0, "flipud": 0.0, "degrees": 0.0, "perspective": 0.0,
    },
    "battle_yolo26n_v3": {
        # 战斗 v3 (2026-07-10) = v2 warm + 用户审定3池(498帧: 活动关133 +
        #  总力战賽特233[固定物传播] + 综合战术考试132[Boss 89框])。
        #  nc 7→14: +塞特的愤怒/Boss(BOSS图标)/一倍速/二倍速/暂停菜单三键
        #  (重开/继续/放弃 — 用户: 战斗模型要会操作暂停菜单)。v2 idx 0-6 不动。
        #  ⚠nc变化: head cls 分支重初始化, 早期 ep 低是正常(v13 同款)。
        #  弱类实况: 胜利7框/二倍速11/菜单三键各23 — 小样本UI固定元素, 26n
        #  历史证明能学(52帧0.995); 敌方172仍是形态类最弱项。
        #  ⭐aug 收敛(用户 2026-07-10: 战斗本身VFX噪声大, 别叠合成噪声学歪):
        #  hsv 几乎关(我方/敌方区分线索一半在色彩, 大抖毁特征), scale 减半;
        #  dim/选中微高亮由离线合成承担(synth_battle_dim.py, train 12%,
        #  模拟卡牌指目标瞄准态)。mosaic 0.3 保留不关: 总力战固定框大量重复,
        #  不打散位置会学成纯位置先验(x=0.15必有柱), 换boss战就崩。
        "kind": "detect",
        "data": YOLO_ROOT / "dataset" / "battle_v3" / "data.yaml",
        "base": str(YOLO_ROOT / "runs" / "battle_yolo26n_v2" / "weights" / "best.pt"),
        "epochs": 150,
        "patience": 40,
        "save_period": 10,
        "imgsz": 640,
        "batch": 32,
        "out_name": "battle_yolo26n_v3",
        "cache": True,
        "workers": 8,
        "mosaic": 0.3, "close_mosaic": 15, "copy_paste": 0.0, "mixup": 0.0,
        "scale": 0.2, "translate": 0.1,
        "hsv_h": 0.01, "hsv_s": 0.1, "hsv_v": 0.25,
        "fliplr": 0.0, "flipud": 0.0, "degrees": 0.0, "perspective": 0.0,
    },
    "battle_yolo26n_v4": {
        # 战斗 v4 (2026-07-11) = v3 warm + 两新池(110759总力战157帧 +
        #  104427综合考试86帧: v3预标→用户人审身份类 + ui v13重建HUD —
        #  v3 的 HUD 双框/1倍速误标三倍速/暂停漏标已根治, 见
        #  fix_battle_ui_labels.py)。713对: train607+86dim合成 / val106。
        #  nc=14 不变(纯增量 warm, 无 head 重初始化)。配方=v3 aug收敛版原样。
        "kind": "detect",
        "data": YOLO_ROOT / "dataset" / "battle_v4" / "data.yaml",
        "base": str(YOLO_ROOT / "runs" / "battle_yolo26n_v3" / "weights" / "best.pt"),
        "epochs": 150,
        "patience": 40,
        "save_period": 10,
        "imgsz": 640,
        "batch": 32,
        "out_name": "battle_yolo26n_v4",
        "cache": True,
        "workers": 8,
        "mosaic": 0.3, "close_mosaic": 15, "copy_paste": 0.0, "mixup": 0.0,
        "scale": 0.2, "translate": 0.1,
        "hsv_h": 0.01, "hsv_s": 0.1, "hsv_v": 0.25,
        "fliplr": 0.0, "flipud": 0.0, "degrees": 0.0, "perspective": 0.0,
    },
    "unified_yolo26x_v6": {
        # 通用 26x = ui + 头像(251) + 摸头, nc=455. warm-start from fused_avatar_26x_v4:
        #  26x backbone 已学满 251 角色脸特征 → 头像部分继承 v4 的 0.966 起点(不从零学、
        #  压住"头像退化"风险); cls 头 251→455 重学(角色行继承, UI/emoticon 行新增).
        #  UI 比角色脸简单, 26x 容量足 + avatar_md 把每角色填到 min31/avg70, 都学得动.
        #  ⚠ fliplr=0: 通用模型混了 UI(左/右切换/返回键 有方向) — 翻转会破坏 UI 方向语义.
        #  copy_paste/mixup=0: 角色细粒度毒药(fused v4 教训). 数据 synth:real=1.5:1.
        "kind": "detect",
        "data": YOLO_ROOT / "dataset" / "ui_v2" / "data.yaml",
        "base": str(YOLO_ROOT / "runs" / "fused_avatar_yolo26x_v4" / "weights" / "best_manual.pt"),
        "epochs": 70,            # warm-start: backbone 已强, cls头(252→455)重学 ~60-70 足
        "patience": 30,
        "imgsz": 960,
        "batch": 8,
        "out_name": "unified_yolo26x_v6",
        "cache": "disk", "workers": 8,
        "optimizer": "SGD",      # 锁死! 否则 auto 覆盖 lr0、用 0.01 冲掉 fused 角色特征
        "lr0": 0.003,            # 低 LR 护 fused v4 的 251 角色特征(防头像退化)
        "momentum": 0.937,
        "weight_decay": 0.0005, "dropout": 0.0,
        "mosaic": 0.3, "close_mosaic": 5,
        "copy_paste": 0.0, "mixup": 0.0,
        "scale": 0.3, "translate": 0.1,
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.3,
        "fliplr": 0.0, "flipud": 0.0, "degrees": 0.0, "perspective": 0.0,
    },
    "unified_yolo26x_v6b": {
        # v6 重训 (2026-06-04): v6 ep12 见顶 mAP50-95 0.748 后 ep13-18 退化到 0.63-0.66
        # (lr0=0.003 对 warm-start 偏高, 早期冲顶后持续扰动). 验收: UI 域 0.87 强,
        # 头像域仅 0.77 (fused v4 专门干是 0.966, 整合退化 ~0.2). 改 lr0→0.0015 更护
        # fused 头像特征 + patience→40 够到 close_mosaic. 其余同 v6. 目标 UI 保 0.87 + 头像拉回 0.9+.
        "kind": "detect",
        "data": YOLO_ROOT / "dataset" / "ui_v2" / "data.yaml",
        "base": str(YOLO_ROOT / "runs" / "fused_avatar_yolo26x_v4" / "weights" / "best_manual.pt"),
        "epochs": 70,
        "patience": 40,
        "imgsz": 960,
        "batch": 8,
        "out_name": "unified_yolo26x_v6b",
        "cache": "disk", "workers": 8,
        "optimizer": "SGD",
        "lr0": 0.0015,
        "momentum": 0.937,
        "weight_decay": 0.0005, "dropout": 0.0,
        "mosaic": 0.3, "close_mosaic": 5,
        "copy_paste": 0.0, "mixup": 0.0,
        "scale": 0.3, "translate": 0.1,
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.3,
        "fliplr": 0.0, "flipud": 0.0, "degrees": 0.0, "perspective": 0.0,
    },
    "unified_yolo26x_v6c": {
        # v6c (2026-06-06 用户决策): 砍 UI synth (_synth_ui_swap), UI 只真实帧根治 synth
        # 过拟合 (v6b 实锤: UI val 0.892 高 / live 崩, 咖啡厅入口 val>0.9 live 仅 0.25)。头像
        # synth (_fused_synth_remap) 保留 (影响小)。synth:real 2.03→~0.89。同 v6b 参数
        # (lr0 0.0015 护头像 / mosaic 0.3 / close_mosaic 5 / patience 40 / from fused v4 重来)。
        # ⚠️ 训前必须重建 ui_v2 (build_ui_v2 已砍 _synth_ui_swap)。UI 弱类暂靠 skill 兜底, v7 飞轮补。
        "kind": "detect",
        "data": YOLO_ROOT / "dataset" / "ui_v2" / "data.yaml",
        "base": str(YOLO_ROOT / "runs" / "fused_avatar_yolo26x_v4" / "weights" / "best_manual.pt"),
        "epochs": 70,
        "patience": 40,
        "imgsz": 960,
        "batch": 8,
        "out_name": "unified_yolo26x_v6c",
        "cache": "disk", "workers": 8,
        "optimizer": "SGD",
        "lr0": 0.0015,
        "momentum": 0.937,
        "weight_decay": 0.0005, "dropout": 0.0,
        "mosaic": 0.3, "close_mosaic": 5,
        "copy_paste": 0.0, "mixup": 0.0,
        "scale": 0.3, "translate": 0.1,
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.3,
        "fliplr": 0.0, "flipud": 0.0, "degrees": 0.0, "perspective": 0.0,
    },
    "unified_yolo26x_v6b_ft": {
        # v6b ep10 见顶 0.864 后退化(ep23 跌 0.747, R 0.773→0.656 = 过拟合非震荡).
        # 关 mosaic 微调救一波(v4 同款思路): from ep10 best warm-start + mosaic 全关(去拼图伪分布) +
        # 极低 lr(防扰动已收敛特征) + 短训 + 小 patience(防再退化). scale/translate 留(空间aug无害).
        "kind": "detect",
        "data": YOLO_ROOT / "dataset" / "ui_v2" / "data.yaml",
        "base": str(YOLO_ROOT / "runs" / "unified_yolo26x_v6b" / "weights" / "best.pt"),
        "epochs": 15,
        "patience": 8,
        "imgsz": 960,
        "batch": 8,
        "out_name": "unified_yolo26x_v6b_ft",
        "cache": "disk", "workers": 8,
        "optimizer": "SGD",
        "lr0": 0.0003,
        "momentum": 0.937,
        "weight_decay": 0.0005, "dropout": 0.0,
        "mosaic": 0.0, "close_mosaic": 0,
        "copy_paste": 0.0, "mixup": 0.0,
        "scale": 0.3, "translate": 0.1,
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.3,
        "fliplr": 0.0, "flipud": 0.0, "degrees": 0.0, "perspective": 0.0,
    },
    "ui_yolo26m_v2_cont": {
        # Continue v2 from ep16 (best_real.pt) — warm-start, NOT --resume
        # (resume deadlocked on 4472-img re-scan). User confirmed static UI
        # doesn't overfit (train≈test), so aug stays 0 and we train hard
        # with a large patience (don't trust the tiny 51-frame val mAP for
        # early-stop — judge best by manual eval on real screenshots).
        # Goal: push small-icon recall (lobby entries 咖啡厅入口 etc, 15
        # original backgrounds) which ep16 was too early to learn.
        "kind": "detect",
        "data": YOLO_ROOT / "dataset" / "ui_v1" / "data.yaml",
        "base": str(YOLO_ROOT / "runs" / "ui_yolo26m_v2" / "weights" / "best_real.pt"),
        "epochs": 150,
        "patience": 120,        # ~no early-stop; UI doesn't overfit
        "imgsz": 960,
        "batch": 12,            # 16 偶发 CUDA OOM (TaskAlignedAssigner 峰值); 12 留余量
        "out_name": "ui_yolo26m_v2_cont",
        # cache/workers 回最初稳定配置 (ep1-16 用这个没崩). cache=ram 没加速
        # (瓶颈是 GPU 算力, ~20min/ep 固有, 不是数据IO), 还引入 Windows
        # DataLoader spawn 崩溃 — 故删除.
        "lr0": 0.005,           # warm-start: half default to protect ep16 features
        # AUG ALL 0 (static UI, user spec — overfit is fine, train≈test)
        "mosaic": 0.0, "close_mosaic": 0, "mixup": 0.0, "copy_paste": 0.0,
        "fliplr": 0.0, "flipud": 0.0, "degrees": 0.0, "perspective": 0.0,
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0, "scale": 0.0, "translate": 0.0,
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
        workers=cfg.get("workers", 4),
        cache=cfg.get("cache", False),   # True=ram (resize后~7GB), "disk"=.npy
        patience=cfg.get("patience", 30),
        project=str(YOLO_ROOT / "runs"),
        name=cfg["out_name"],
        exist_ok=True,
        save=True,
        save_period=cfg.get("save_period", -1),
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
                  "perspective", "optimizer", "momentum", "lrf", "cos_lr"):
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
