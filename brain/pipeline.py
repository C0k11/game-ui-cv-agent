"""DailyPipeline: Orchestrates skill-based daily routine automation.

Sequences skills in order, handles timeouts, retries, and recovery.
Uses OCR as the primary screen-reading method (portable across resolutions).

Usage:
    from brain.pipeline import DailyPipeline
    pipe = DailyPipeline()
    pipe.start()
    # Each tick: pipe.tick_from_frame(frame_bgr) -> action dict
"""
from __future__ import annotations

import json
import queue
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from brain.skills.base import (
    BaseSkill, OcrBox, YoloBox, ScreenState,
    action_click, action_click_box, action_back, action_wait, action_done,
)
from brain.skills.cafe import CafeSkill
from brain.skills.schedule import ScheduleSkill
from brain.skills.club import ClubSkill
from brain.skills.bounty import BountySkill
from brain.skills.jfd import JointFiringDrillSkill
from brain.skills.batch_sweep import BatchSweepSkill
from brain.skills.special_sweep import SpecialSweepSkill
from brain.skills.event_quest import EventQuestSkill
from brain.skills.arena_shop import ArenaShopSkill
from brain.skills.mail import MailSkill
from brain.skills.arena import ArenaSkill
from brain.skills.shop import ShopSkill
from brain.skills.craft import CraftSkill
from brain.skills.buy_pyroxene import BuyPyroxeneSkill
from brain.skills.daily_mission import DailyMissionSkill
from brain.skills.momo_talk import MomoTalkSkill
from brain.skills.story_mining import StoryMiningSkill
from brain.skills.daily_routine import DailyRoutineSkill


# ── OCR Engine (singleton) ──────────────────────────────────────────────

_ocr_engine = None
_ocr_lock = None

def _try_enable_cuda_dlls() -> bool:
    """Load CUDA/cuDNN + TensorRT DLLs so onnxruntime-gpu can register both
    the CUDA and TensorRT execution providers without a separate CUDA/TRT
    Toolkit install. Pip packages supply all needed DLLs:

      - torch (cu124) bundles cublasLt / cudart / cudnn under torch/lib/
      - tensorrt-cu12-libs (pip) bundles nvinfer_10.dll under tensorrt_libs/

    Returns True if at least one DLL dir was added.
    """
    added = False
    try:
        import os
        if not hasattr(os, "add_dll_directory"):
            return False
        try:
            import torch  # type: ignore
            lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
            if os.path.isdir(lib_dir):
                os.add_dll_directory(lib_dir)
                added = True
        except Exception:
            pass
        try:
            import tensorrt_libs  # type: ignore
            trt_dir = os.path.dirname(tensorrt_libs.__file__)
            if os.path.isdir(trt_dir):
                os.add_dll_directory(trt_dir)
                added = True
        except Exception:
            pass
    except Exception:
        pass
    return added


def _get_ocr():
    """Get or create RapidOCR engine (thread-safe singleton).

    Automatically:
    - Loads fine-tuned Blue Archive rec model from data/ocr_model/ba_rec.onnx
      if present, otherwise uses default PP-OCRv3.
    - Tries to enable CUDA provider (det + rec). Falls back to CPU if CUDA
      runtime DLLs are unavailable. Measured on RTX 4090: full-frame 1262x2243
      CPU 2.2 FPS → CUDA 3.3 FPS; ROI-sized (~45% screen) CUDA 9 FPS.
    """
    global _ocr_engine, _ocr_lock
    import threading
    if _ocr_lock is None:
        _ocr_lock = threading.Lock()
    with _ocr_lock:
        if _ocr_engine is None:
            _try_enable_cuda_dlls()
            from rapidocr_onnxruntime import RapidOCR
            custom_rec = Path(__file__).resolve().parent.parent / "data" / "ocr_model" / "ba_rec.onnx"
            kw = dict(
                det_use_cuda=True, det_model_path=None,
                rec_use_cuda=True,
                cls_use_cuda=True, cls_model_path=None,
            )
            kw["rec_model_path"] = str(custom_rec) if custom_rec.exists() else None
            try:
                _ocr_engine = RapidOCR(**kw)
                # Inspect what provider det actually got — if CPU, we know
                # CUDA load failed silently.
                try:
                    det_providers = _ocr_engine.text_detector.infer.session.get_providers()
                    if "CUDAExecutionProvider" in det_providers:
                        print(f"[OCR] CUDA provider active (det={det_providers})")
                    else:
                        print(f"[OCR] CUDA requested but fell back to CPU "
                              f"(det={det_providers}). Install nvidia-cuda-runtime-cu12 "
                              f"or add CUDA Toolkit to PATH.")
                except Exception:
                    pass
                if custom_rec.exists():
                    print(f"[OCR] Using fine-tuned BA rec model: {custom_rec.name}")
            except Exception as e:
                # Fall back to pure CPU with default kwargs
                print(f"[OCR] CUDA init failed ({e!r}); using CPU")
                _ocr_engine = (
                    RapidOCR(rec_model_path=str(custom_rec))
                    if custom_rec.exists()
                    else RapidOCR()
                )
        return _ocr_engine


# ── YOLO Detector (singleton) ───────────────────────────────────────

_yolo_models = []   # list of (model, conf_threshold, model_tag) tuples
_yolo_lock = None
# Only two purpose-built models: battle character heads + cafe emoticon bubbles.
# emoticon migrated to YOLO26n (2026-05-17) — same architecture family as the
# previous v8n but NMS-free, 122 layers / 2.4M params / 5.2 GFLOPs.  Validation
# on emoticon_v2 dataset: P=0.994 R=1.000 mAP50=0.995 mAP50-95=0.994, inference
# 0.4ms/frame.  Drop-in replacement — same single class "Emoticon_Action".
# Per-process session id, used for grouping hard-example dumps from one run
import datetime as _dt
_PIPELINE_SESSION_ID = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

# ── Model registry — single source of truth for active model paths ──
# data/model_registry.json drives which version is "live".  Hardcoded paths
# below are fallbacks for back-compat / when registry is unreachable.

def _load_model_registry() -> dict:
    """Read data/model_registry.json. Returns {} on any error."""
    try:
        reg_path = Path(__file__).resolve().parent.parent / "data" / "model_registry.json"
        if reg_path.is_file():
            import json
            return json.loads(reg_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[Pipeline] model_registry load failed: {e}")
    return {}

def _resolve_path(model_key: str, fallback: Path) -> Path:
    """Resolve an active model's weights path via registry.

    ⛔fail-closed(2026-07-16 审计): registry 存在但该 key 解析不出 →
    raise, 绝不静默回落硬编码老路径 — 旧行为是"registry JSON 手编出错
    /active 指错"时带着 5 月 v1(145类, 缺全部商店/confirm 钱防线类)照常
    开跑, 正是"改了 registry 没效果、旧模型还在跑"的复现路径。
    fallback 仅在 registry 文件整体缺失(全新环境)时使用。"""
    reg = _load_model_registry()
    if not reg:                     # registry 文件缺失/坏 → 见下
        if fallback.is_file():
            print(f"[Pipeline] ⚠registry 缺失, {model_key} 回落 {fallback}")
            return fallback
        raise RuntimeError(f"model registry 缺失且 {model_key} fallback 不存在")
    section = reg.get(model_key)
    if not section:
        raise RuntimeError(f"registry 无 '{model_key}' 节 — 修 registry, 不回落老模型")
    active = section.get("active")
    versions = section.get("versions", {})
    info = versions.get(active, {})
    p = info.get("path")
    if p and Path(p).is_file():
        return Path(p)
    raise RuntimeError(
        f"registry {model_key}.active='{active}' 路径无效({p}) — "
        f"修 registry, 绝不静默回落老模型")


_YOLO_BATTLE_HEADS = Path(r"D:\Project\ml_cache\models\yolo\battle_heads.pt")
_YOLO_EMOTICON_V26 = Path(r"D:\Project\ml_cache\models\yolo\runs\emoticon_yolo26n\weights\best.pt")
_YOLO_EMOTICON = _YOLO_EMOTICON_V26

# ── Fused avatar v4 (251-class student head detector) ──
# Manual mAP50 = 0.9657 on hand-curated 29-frame val (vs v3 baseline 0.683).
# best_manual.pt = ep11 weights (true best, vs best.pt = ep15 nominal-best
# that was synth-fitness-biased). See yolo_migration memory for details.
_YOLO_FUSED_AVATAR_V4 = _resolve_path("fused_avatar", Path(
    r"D:\Project\ml_cache\models\yolo\runs\fused_avatar_yolo26x_v4\weights\best_manual.pt"
))

# ── UI v1 (~145-class static UI detector) ──
# Replaces OCR-driven button finding in most skills. Trained 2026-05-27 from
# COCO yolo26m + 1220 frames (oversampled minority classes target=12) + 51
# hand-curated val frames. Target mAP50 ≥ 0.85.
_YOLO_UI_V1 = _resolve_path("ui", Path(
    r"D:\Project\ml_cache\models\yolo\runs\ui_yolo26m_v1\weights\best.pt"
))

# Context-aware YOLO: controls which models run each tick.
# Value semantics:
#   'none'                   skip YOLO entirely (default)
#   'all'                    run every loaded model
#   single tag e.g. 'cafe'   run only that detector
#   '+'-joined e.g. 'ui+avatar' or 'cafe+ui'   run multiple detectors
# Legacy callers passing 'cafe' / 'battle' still work unchanged.
_yolo_context = "none"
_yolo_context_lock = None

def set_yolo_context(ctx: str) -> None:
    """Set which YOLO models should run. Called by pipeline on skill change."""
    global _yolo_context, _yolo_context_lock
    import threading
    if _yolo_context_lock is None:
        _yolo_context_lock = threading.Lock()
    with _yolo_context_lock:
        if _yolo_context != ctx:
            print(f"[Pipeline] YOLO context: {_yolo_context} → {ctx}")
            _yolo_context = ctx

def get_yolo_context() -> str:
    """Get current YOLO context (thread-safe)."""
    global _yolo_context, _yolo_context_lock
    import threading
    if _yolo_context_lock is None:
        _yolo_context_lock = threading.Lock()
    with _yolo_context_lock:
        return _yolo_context


# ── Lobby resource snapshot (user 2026-06-09: 大厅顶栏三资源稳读 → 全局复用) ──
# Refreshed on lobby frames every ~30s. Consumers:
#   • shop budget fallback (in-shop top-bar credit cls is flaky → use snapshot)
#   • ⛔ PYROXENE KILL-SWITCH: a DROP between two consecutive confirmed reads =
#     money breach somewhere → abort the whole pipeline (global audit on top of
#     every per-skill guard).
_RESOURCES: Dict[str, Any] = {"ap": None, "credits": None, "pyroxene": None, "ts": 0.0}


def get_resource_snapshot() -> Dict[str, Any]:
    """Latest lobby top-bar reads: {ap, credits, pyroxene, ts}. Values None
    until the first successful lobby read; ts is time.time() of last refresh."""
    return dict(_RESOURCES)


# Per-currency digit-field span (fraction of frame width) right of the icon.
# Calibrated 2026-06-11 from the live top-bar layout (体力→加号 gap 0.084,
# 信用点→青辉石 0.141, 青辉石→加号 0.082). AP/pyrox narrow, credit wide.
def _topbar_span_map():
    # Calibrated 12-sample live 2026-06-11:
    #   AP 0.06   → "999" only (excludes "/240"; 0.078 caught the slash → 9999)
    #   credit 0.118 → reads ~1.8B (a stable +1-digit OCR over-read of the
    #                  9-digit 1.8亿; good enough for shop's "rich enough?" gate.
    #                  wider spans → None). Exact credit is an OCR-model limit.
    #   pyrox 0.078 → "6587" reliably (6587 ×12).
    from brain.skills.ui_classes import TOPBAR_AP, TOPBAR_CREDIT, TOPBAR_PYROXENE
    return {TOPBAR_AP: 0.06, TOPBAR_CREDIT: 0.118, TOPBAR_PYROXENE: 0.078}


try:
    _TOPBAR_SPAN = _topbar_span_map()
except Exception:
    _TOPBAR_SPAN = {}


def _read_topbar_count(screen, cls_name: str):
    """DIGIT-only read of the number right of a top-bar icon (cy<0.10)."""
    best = None
    for b in (screen.yolo_boxes or []):
        if b.cls_name != cls_name or b.confidence < 0.25:
            continue
        if b.cy >= 0.10:
            continue
        if best is None or b.confidence > best.confidence:
            best = b
    if best is None or screen.frame is None:
        return None
    bh = best.y2 - best.y1
    # Right edge = a per-currency FIXED span from the icon. The old
    # neighbour-clip (clip at the next 加号/icon) was the bug: the neighbour
    # flickers frame-to-frame, and when AP's 加号 dropped the span over-reached
    # into credit and read 999→9999 (systematic, sampled 12× live 2026-06-11).
    # The top bar is a fixed layout, so a per-field span is deterministic and
    # frame-independent: AP/pyrox fields are narrow (~0.078), credit is a wide
    # 9-digit field (~0.135). (parse_count takes the numerator of AP's
    # "999/240", so a touch of slack on AP is harmless.)
    _span = _TOPBAR_SPAN.get(cls_name, 0.078)
    x_right = min(1.0, best.x2 + _span)
    raw = run_digit_ocr(screen.frame, (
        min(1.0, best.x2 + 0.003), max(0.0, best.y1 - bh * 0.25),
        x_right, min(1.0, best.y2 + bh * 0.25)))
    res = parse_count(raw)
    return res[0] if (res is not None and res[0] is not None) else None


class _FrameShim:
    """Minimal screen-like holder for _read_topbar_count on a raw frame."""
    __slots__ = ("yolo_boxes", "frame")

    def __init__(self, boxes, frame):
        self.yolo_boxes = boxes
        self.frame = frame


def _read_topbar_clean(cls_name, samples: int = 5):
    """Top-bar count from fresh overlay-free ADB frames (2026-06-10 money rule),
    made robust to the digit OCR's BOTH-WAY instability (live 2026-06-11:
    leading-digit DROP 6587→587 AND right-edge OVER-read 999→9999 / credit
    reaching into the neighbour). Neither "fewest" nor "most" digits is right,
    so VOTE: read up to `samples` clean frames, return the MODE (the value the
    OCR agrees on most often — correct more often than any single error mode).
    Returns int or None.

    ⚠️ KNOWN GAP (task#5): AP/credit still mis-crop on many frames; pyroxene is
    reliable. Until per-currency right-edge crop is calibrated, callers that
    spend on a balance (shop) must treat a low-confidence read as unverifiable
    and skip — never over-trust an inflated read."""
    from collections import Counter
    reads = []
    for _ in range(max(1, samples)):
        frame = get_clean_frame()
        if frame is None:
            continue
        try:
            h, w = frame.shape[:2]
            boxes = _run_yolo_on_image(frame, w, h)
            v = _read_topbar_count(_FrameShim(boxes, frame), cls_name)
        except Exception:
            v = None
        if v is not None:
            reads.append(v)
            if reads.count(v) >= 3:   # strong agreement → done early
                return v
    if not reads:
        return None
    cnt = Counter(reads)
    top, n = cnt.most_common(1)[0]
    # Require a real majority (≥2 agreeing) before trusting; a single noisy
    # read is not authoritative for money decisions.
    return top if n >= 2 else None


def _read_topbar_clean_multi(cls_names, samples: int = 5):
    """快照专用: 一批 clean 帧共享给多个货币读数 (2026-07-11 链路审计).
    语义与 _read_topbar_clean 完全一致(≤samples 帧 / 每 cls mode 投票 /
    3 票强共识早退 / ≥2 票才信), 但 3 货币共享同批帧 → captures 15→5、
    YOLO 15→5(旧版 lobby 快照单 tick 16 连拍阻塞主循环 7-20s 的元凶)。
    _read_topbar_clean 本体保持原样 — shop/ticket_sweep 等金钱敏感调用方
    的单币种投票语义不动。Returns {cls_name: int|None}."""
    from collections import Counter
    reads = {c: [] for c in cls_names}
    done = set()
    for _ in range(max(1, samples)):
        if len(done) == len(cls_names):
            break
        frame = get_clean_frame()
        if frame is None:
            continue
        try:
            h, w = frame.shape[:2]
            boxes = _run_yolo_on_image(frame, w, h)
            shim = _FrameShim(boxes, frame)
        except Exception:
            continue
        for c in cls_names:
            if c in done:
                continue
            try:
                v = _read_topbar_count(shim, c)
            except Exception:
                v = None
            if v is not None:
                reads[c].append(v)
                if reads[c].count(v) >= 3:
                    done.add(c)
    out = {}
    for c in cls_names:
        r = reads[c]
        if not r:
            out[c] = None
            continue
        top, n = Counter(r).most_common(1)[0]
        out[c] = top if (n >= 2 or r.count(top) >= 3) else None
    return out


def _read_pyroxene_clean():
    """Kill-switch helper: authoritative pyroxene from a clean frame."""
    from brain.skills.ui_classes import TOPBAR_PYROXENE
    return _read_topbar_clean(TOPBAR_PYROXENE)

# Per-skill YOLO detector loadout (module-level = single source of truth).
# base = "ui" ONLY (FPS: avatar=fused yolo26X / battle are heavy nets, added
# only where a skill needs them). _start_skill sets context from this at
# skill start.
BASE_DETECTORS = "ui"
SKILL_YOLO_MAP = {
    # Cafe needs avatar too: the invite list identifies each row's student via
    # the fused_avatar head model (model_tag=="avatar", 中文角色名) so it can
    # invite the configured cafe_invite_targets. +cafe = emoticon headpat marks.
    # +emoticon bubbles +student heads. The "cafe" tag loads the standalone
    # emoticon model; from ui v6 (which carries Emoticon_Action) _get_yolo drops
    # that model, so "cafe" becomes a harmless no-op and emoticon comes from ui.
    "Cafe": f"{BASE_DETECTORS}+cafe+avatar",
    "Bounty": f"{BASE_DETECTORS}+battle",      # +battle_heads
    # Arena selects opponents via cls92 (ARENA_OPPONENT_ROW) in the UI model —
    # no avatar model needed (dropped 2026-05-31, v5 added cls92). +battle for
    # the in-fight skip/heads.
    "Arena": f"{BASE_DETECTORS}+battle",
    # Schedule needs avatar to identify which student sits in each room / 全体
    # 课程表 list (fused_avatar 中文角色名) so it can place the dashboard-chosen
    # targets. NO emoticon — headpat is cafe-only. (probe-derived 2026-06-01)
    "Schedule": f"{BASE_DETECTORS}+avatar",
    "JointFiringDrill": f"{BASE_DETECTORS}+battle",
    # DailyRoutine wraps cafe (emoticon headpat + fused_avatar invite targets)
    # and schedule (fused_avatar room placement) → needs cafe + avatar. No
    # battle: bounty/jfd/arena are separate top-level skills now.
    "DailyRoutine": f"{BASE_DETECTORS}+cafe+avatar",
}

_yolo_load_attempts = 0
_MAX_YOLO_LOAD_ATTEMPTS = 3
_yolo_status = "not_attempted"
# ui 模型加载失败旗标 (2026-07-07): True = 导航之眼缺失, tick 循环立刻 abort,
# 绝不 blind/wake-tap (假 no-UI 下 wake-tap 曾反复戳开購買AP框)。
_UI_LOAD_FAILED = False
# No class-name filter — purpose-built models (battle_heads, emoticon) only
# contain relevant classes.  The old _YOLO_ALLOWED_SUBSTRINGS gate silently
# dropped every detection from battle_heads whose classes are c0-c3.

def _get_yolo():
    """Get or create YOLO model(s) (lazy singleton). Only loads on first call.

    Loads available model files (battle_heads + emoticon).
    Each entry is a (model, conf_threshold) tuple.
    Returns the first model for backward compat; _yolo_models holds all.
    """
    global _yolo_models, _yolo_lock, _yolo_load_attempts, _yolo_status
    import threading
    if _yolo_lock is None:
        _yolo_lock = threading.Lock()
    with _yolo_lock:
        if _yolo_models:
            return _yolo_models[0]
        if _yolo_load_attempts >= _MAX_YOLO_LOAD_ATTEMPTS:
            return None
        _yolo_load_attempts += 1
        # Per-model confidence thresholds:
        # battle_heads: 0.45 (well-defined targets; 0.15 causes false positives
        #   on cafe sprites at conf 0.25-0.47)
        # emoticon: 0.15 (headpat bubbles on 2F score as low as 0.18)
        candidates = []  # (path, conf_threshold, tag)
        # battle 走 registry 最新 vN(2026-07-16 历史遗留修复: 旧代码硬编码
        # legacy battle_heads.pt 无视 registry — server 侧 Bounty/Arena/JFD
        # 一直用老模型, v9(0.989 nc18) 只有战斗脚本在用)
        # active 优先(2026-07-16 审计: max vN 会在"先登记 v10 条目未验收"
        # 时静默上未验收模型; active 是验收后才 bump 的正式指针), 无 active
        # 才回退 max vN。
        _battle_path = _YOLO_BATTLE_HEADS
        try:
            import re as _re
            _bh = _load_model_registry().get("battle_heads", {})
            _vers = _bh.get("versions", {})
            _pick = _bh.get("active")
            if _pick not in _vers or not _re.fullmatch(r"v\d+", str(_pick)):
                _pick = max((v for v in _vers if _re.fullmatch(r"v\d+", v)),
                            key=lambda x: int(x[1:]), default=None)
            if _pick:
                _p = Path(_vers[_pick]["path"])
                if not _p.is_absolute():
                    _p = Path("D:/Project") / str(_p).lstrip("/\\")
                if _p.is_file():
                    _battle_path = _p
                    print(f"[Pipeline] battle_heads → registry {_pick}")
        except Exception as _e:
            print(f"[Pipeline] battle registry resolve failed({_e}), legacy")
        if _battle_path.is_file():
            candidates.append((_battle_path, 0.45, "battle"))
        # standalone emoticon (tag "cafe") 不在这里 append — fold-in 判定提前
        # (2026-07-17): ui v6+ 自带 Emoticon_Action, 仅当 ui 类表缺该类时才在
        # 加载环节末尾补载 v26n(见 load loop 之后), 省一次白加载即丢的模型。
        # Fused avatar (251 BA student heads).  conf 0.35 = balanced
        # precision/recall on manual val.  Tagged "avatar" — opt-in per skill.
        if _YOLO_FUSED_AVATAR_V4.is_file():
            candidates.append((_YOLO_FUSED_AVATAR_V4, 0.35, "avatar"))
        # UI (registry active — buttons, dots, banners, etc).  Tagged "ui" —
        # most skills need this. (unified v6b 接线已拆除 2026-07-17: registry
        # unified.active 恒 PENDING 从未通电, v6b nc=455 缺后续金钱防线类,
        # 见 registry unified._deprecated 注。)
        _ui_path = _YOLO_UI_V1
        if _ui_path is not None and _ui_path.is_file():
            # 0.20 — within the dashboard's own prefill range (server/app.py:
            # single-frame suggest 0.15, batch prefill 0.25), the settings the
            # user verifies cls against. Live at 0.30 dropped weak cls the
            # dashboard catches (免费 14f live ~0.18-0.30). Strong cls (0.9+)
            # unaffected. Skills still gate money paths structurally (2-button
            # confirm + 免费/币种 checks), so a lower floor doesn't risk spend.
            candidates.append((_ui_path, 0.20, "ui"))
        if not candidates:
            _yolo_status = "model_not_found"
            print(f"[Pipeline] YOLO model NOT found")
            return None
        from ultralytics import YOLO
        import numpy as np
        loaded_names = []
        global _UI_LOAD_FAILED
        _UI_LOAD_FAILED = False

        def _load_model(model_path, model_conf, model_tag) -> bool:
            # 加载失败(fresh-server CUDA dtype 瞬态, 2026-07-07 实锤
            # "float != c10::Half")重试一次 — dtype 病随机咬任意模型。
            for _t in range(2):
                try:
                    m = YOLO(str(model_path))
                    m(np.zeros((64, 64, 3), dtype=np.uint8), verbose=False)
                    _yolo_models.append((m, model_conf, model_tag))
                    loaded_names.append(f"{model_path.stem}({len(m.names)}cls)")
                    print(f"[Pipeline] YOLO loaded from {model_path} (conf={model_conf}, tag={model_tag})")
                    return True
                except Exception as e:
                    print(f"[Pipeline] YOLO load failed for {model_path} (try {_t+1}/2): {e}")
            return False

        for model_path, model_conf, model_tag in candidates:
            # ui 模型是导航的眼睛 — 加载失败(重试后仍败)打 _UI_LOAD_FAILED 旗标,
            # tick 循环见旗标立刻 abort(绝不带着假 no-UI 去 wake-tap/blind-tap —
            # 那次假 no-UI 让 wake-tap 反复戳开「購買AP」框, 差点碰钱)。
            if not _load_model(model_path, model_conf, model_tag) and model_tag == "ui":
                _UI_LOAD_FAILED = True
                print("[Pipeline] ⛔ ui model FAILED to load after retry — pipeline will "
                      "abort immediately (fail-closed: no taps without UI eyes)")
        # ── emoticon fold-in (ui v6+, 判定提前 2026-07-17) ────────────────
        # ui 类表含 Emoticon_Action → 摸头泡泡由 ui forward pass 提供,
        # standalone v26n 不再加载(旧代码先完整加载再 fold-in 丢弃 = 每次
        # 启动白加载一个模型)。ui 缺该类(pre-v6)才补载 v26n 保 cafe headpat。
        # cafe.py + 摸头过滤按 cls_name 匹配, box 来自哪个模型无感。
        # (2026-06-11 用户决策) ui 接管摸头, v26n 退役出 live 管线, 只留
        # dashboard 预标注 teacher (server prefill 走 registry, 不受影响)。
        _FOLD_IN_EMOTICON = True
        ui_has_emoticon = _FOLD_IN_EMOTICON and any(
            "emoticon" in str(n).lower()
            for m, _c, t in _yolo_models if t == "ui"
            for n in m.names.values()
        )
        if ui_has_emoticon:
            print("[Pipeline] ui model carries Emoticon_Action → standalone "
                  "emoticon model not loaded (one fewer inference per cafe tick)")
        elif _YOLO_EMOTICON.is_file():
            # 0.50 conf (was 0.15→0.30 same day): live 2026-06-09 credit-card
            # icons kept firing as Emoticon_Action (0.36-0.75). Business gate is
            # 0.55 (cafe.py _EMOTICON_CONF), real v26n bubbles score 0.9+, so
            # 0.50 costs nothing and kills the remaining mid-conf FPs.
            _load_model(_YOLO_EMOTICON, 0.50, "cafe")
        if _yolo_models:
            _yolo_status = f"loaded_ok: {', '.join(loaded_names)}"
            return _yolo_models[0][0]
        _yolo_status = "all_candidates_failed"
        return None


_OCR_WORK_W = 1280  # Downscale wide frames for faster OCR

# ── PURE-YOLO MODE (user spec 2026-05-29) ────────────────────────────────
# OCR is fully disabled to force every skill's navigation + click logic
# through YOLO cls — NO OCR fallback. This surfaces every place still
# secretly relying on OCR (they go blind → log+wait → we migrate them).
# Once the YOLO pipeline is verified end-to-end, flip this back on and
# scope OCR to DIGIT-ONLY scanning (AP / ticket / mail counts).
_OCR_ENABLED = False

# ── BRING-UP EXPOSE MODE (user spec 2026-05-29) ──────────────────────────
# While bringing up pure-YOLO navigation we want every stuck point to be a
# VISIBLE hole, not papered over by blind recovery. With this True the
# stuck-recovery FALLBACKS are disabled:
#   - blind-tap escalation (clicks dismiss positions when stuck)
#   - ESC-burst (presses back when stuck)
# A stuck skill instead FREEZES in place + logs loudly (skill, sub_state,
# wait reason, tick, screenshot path) so we can open that exact tick's
# trajectory frame and see why YOLO couldn't find its target. KEPT (these
# are real game-popup handling / safety, NOT navigation fallbacks): the
# interceptor's reward/level-up/exit-cancel dismissals, and the popup
# 取消/X dismiss (never ESC the exit dialog). Flip False to restore the
# recovery nets for unattended production runs.
# (2026-06-12) flipped False — production nightly chains now. The exposed-hole
# mode cost a run: BA's idle showcase (放置立绘屏, zero UI cls) sat through the
# 30-tick no-UI abort because the dismiss-tap net was disabled. The net's
# rotation taps (0.5,0.88 / X / corners) dismiss it; harmless elsewhere.
# (2026-06-14) flipped True — user iron rule: "不要瞎点返回大厅或者返回, skill 没完成
# 不要瞎退". The blind-tap escalation + ESC-burst nets are exactly that blind nav —
# they wandered arena into the task hall after battle 2. During supervised step_mode
# bring-up a stuck skill must FREEZE+log (we see the hole, fix it event-driven), NOT
# wander. Idle-showcase 放置立绘屏 only triggers on lobby idle (not mid-skill); the
# human supervisor taps it. Flip back False only for unattended nightly.
_BRINGUP_EXPOSE = True

# Gated-click hold cap (user 2026-06-13): after a transition click whose screen
# hasn't changed, hold ~this many ticks before assuming the tap was lost and
# allowing one retry. 16→5 (user 2026-06-13 "无端等待": 16 was ~22s of dead
# waiting; a real nav transition renders in 2-3 ticks so the fingerprint flips
# and releases well before the cap — the cap only fires on a genuinely stuck
# screen, where re-tapping after ~5 ticks is right). Reward/dismiss popups are
# exempted from holding entirely in _dedup_click (see "看到目标就点").
_CLICK_HOLD_CAP = 5

# 启动期窗口(秒, 2026-07-16 强更通知重构): 强更下载确认框只在 pipeline 刚起
# 的这段时间内自动确认(patch-day 冷启动, TOUCH TO START 前后)。窗口外出现的
# "只有确认"弹窗一律不在 interceptor 层碰 — 交给当前 skill 的语境处理。
_STARTUP_UPDATE_WINDOW_S = 180.0

# Debug: force EVERY skill to run, bypassing the red/yellow-dot should_run gate.
# Set via mumu_runner --force-skills. For testing a skill's internals when the
# account has no dot on it today (e.g. cafe with nothing to collect).
_FORCE_ALL_SKILLS = False


def set_force_all_skills(v: bool) -> None:
    global _FORCE_ALL_SKILLS
    _FORCE_ALL_SKILLS = bool(v)


def _run_ocr_on_image(img, w: int, h: int) -> List[OcrBox]:
    """Run OCR on a BGR numpy array and return normalized OcrBox list.

    Downscales frames wider than _OCR_WORK_W for speed (4K→1280px ≈ 9x faster).
    Coordinates are normalized 0-1 so the caller is resolution-independent.
    """
    if not _OCR_ENABLED:
        return []  # pure-YOLO mode — see _OCR_ENABLED note above
    import cv2
    ocr = _get_ocr()
    # Downscale for speed if frame is very wide (e.g. 3840px 4K)
    ocr_img = img
    ocr_w, ocr_h = w, h
    if w > _OCR_WORK_W:
        ratio = _OCR_WORK_W / w
        ocr_h = max(1, int(h * ratio))
        ocr_w = _OCR_WORK_W
        ocr_img = cv2.resize(img, (ocr_w, ocr_h), interpolation=cv2.INTER_AREA)
    result, _ = ocr(ocr_img)
    boxes: List[OcrBox] = []
    if result:
        for line in result:
            pts, text, conf = line
            conf = float(conf)
            if conf < 0.4:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            boxes.append(OcrBox(
                text=text,
                confidence=conf,
                x1=min(xs) / ocr_w,
                y1=min(ys) / ocr_h,
                x2=max(xs) / ocr_w,
                y2=max(ys) / ocr_h,
            ))
    return boxes


# ── Clean-frame source (money-read defense) ────────────────────────────
# The Win32 YoloOverlay burns boxes/labels into every DXcam frame, which can
# KILL detection of small icons (live 2026-06-09: arena 战术大赛票 icon got a
# tight green box + label burned over it → ui_v7 detected NOTHING → ticket
# read None every tick → fail-closed exit with tickets unspent). ADB screencap
# runs inside Android where the overlay physically doesn't exist. Money-
# critical reads should prefer this source. The server registers the ADB
# capture function here once the pipeline's ADB connection is up.
_CLEAN_FRAME_SOURCE = None


def set_clean_frame_source(fn) -> None:
    """Register a zero-arg callable returning a clean BGR frame (or None)."""
    global _CLEAN_FRAME_SOURCE
    _CLEAN_FRAME_SOURCE = fn


def get_clean_frame():
    """A fresh overlay-free frame via the registered source, or None."""
    if _CLEAN_FRAME_SOURCE is None:
        return None
    try:
        return _CLEAN_FRAME_SOURCE()
    except Exception:
        return None


def run_digit_ocr(frame, region_norm) -> Optional[str]:
    """DIGIT-ONLY OCR on a normalized sub-region of a BGR frame.

    The pure-YOLO design (user spec): YOLO locates an icon/region, OCR reads
    ONLY the digits inside a crop next to it. This is INDEPENDENT of the global
    `_OCR_ENABLED` flag (that gates full-screen text OCR for navigation, which
    stays off) — digit reads are always allowed because that's OCR's one job.

    Args:
        frame: BGR numpy array (ScreenState.frame).
        region_norm: (x1, y1, x2, y2) normalized 0-1 crop to read.
    Returns:
        The raw recognized string filtered to digits/separators ("240/240",
        "25117", "7/8"), or None if nothing digit-like was read. Caller parses.
    """
    if frame is None:
        return None
    # 数字读数需要 4K 细节(1080p 实测伤金钱读数): 主 tick 帧换 scrcpy
    # 1440p 后(2026-07-16 Phase2), 凡传入低于 4K 的帧自动升级 ADB 干净帧
    # 重抓 — 一处兜底, 9 个 skill 调用点零改动。抓帧失败用原帧(降级可用)。
    try:
        if frame.shape[1] < 3200:
            _cf = get_clean_frame()
            if _cf is not None:
                frame = _cf
    except Exception:
        pass
    import cv2
    import re as _re
    try:
        h, w = frame.shape[:2]
        x1 = max(0, int(region_norm[0] * w)); y1 = max(0, int(region_norm[1] * h))
        x2 = min(w, int(region_norm[2] * w)); y2 = min(h, int(region_norm[3] * h))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        crop = frame[y1:y2, x1:x2]
        # upscale small crops — OCR is far more accurate on larger glyphs
        ch, cw = crop.shape[:2]
        if ch < 40:
            sc = 40.0 / ch
            crop = cv2.resize(crop, (int(cw * sc), 40), interpolation=cv2.INTER_CUBIC)
        ocr = _get_ocr()
        result, _ = ocr(crop)
        if not result:
            return None
        # Sort fragments LEFT→RIGHT before joining — the detector returns text
        # boxes in arbitrary order, which scrambles comma-grouped numbers
        # (live 2026-06-09: "179,958,141" came back as '9581179414').
        try:
            result = sorted(result, key=lambda ln: min(p[0] for p in ln[0]))
        except Exception:
            pass
        # concat all recognized text on the strip, keep only digits + / and ,
        raw = "".join(line[1] for line in result)
        # Comma-grouped big numbers (credit 25,583,379 etc): blind strip-and-
        # join DUPLICATES digits when OCR fragments overlap (live 2026-06-12:
        # '25,583,379' → '255833379' = 10x over-read → shop budget chaos).
        # The comma grouping VALIDATES digit structure — when present, trust
        # only a clean single group; several disjoint groups = fragment mess →
        # fail-closed None (multi-sample voting retries).
        raw_n = raw.replace("，", ",")
        groups = _re.findall(r"\d{1,3}(?:,\d{3})+", raw_n)
        if groups:
            longest = max(groups, key=len)
            others = [g for g in groups if g != longest and g not in longest]
            if others:
                return None   # ambiguous overlapping fragments
            return longest.replace(",", "")
        # Keep the decimal point too (deep-dive r2 C1, 2026-06-09): stripping it
        # turned "0.0%" into "00" and "58.3" into "583" — consumers that parse
        # floats (cafe earnings % gate) need the dot. parse_count() is dot-free
        # by domain (counts/AP/tickets never render decimals) so this is safe.
        kept = _re.sub(r"[^0-9/.]", "", raw_n.replace(",", ""))
        return kept or None
    except Exception as e:
        print(f"[digit-OCR] error: {e}")
        return None


def parse_count(s: Optional[str]):
    """Parse a digit-OCR string into useful numbers.

    "7/8" → (7, 8); "25117" → (25117, None); "240/240" → (240, 240).
    Returns (current, total) where total may be None. None on unparseable.
    """
    if not s:
        return None
    try:
        if "/" in s:
            a, _, b = s.partition("/")
            cur = int(a) if a.isdigit() else None
            tot = int(b) if b.isdigit() else None
            if cur is None:
                return None
            return (cur, tot)
        if s.isdigit():
            return (int(s), None)
    except Exception:
        pass
    return None


def _run_yolo_on_image(img, w: int, h: int, context: str = "") -> List[YoloBox]:
    """Run YOLO on a BGR numpy array and return normalized YoloBox list.

    Only runs models matching the current context:
      'cafe'   → emoticon model only
      'battle' → battle_heads model only
      'all'    → all models
      'none'/'' → skip entirely (returns empty)

    Non-blocking: if YOLO is still loading in the pre-warm thread, returns
    empty immediately instead of blocking the pipeline worker.
    """
    if not context:
        context = get_yolo_context()
    yolo_boxes: List[YoloBox] = []
    if context == "none":
        return yolo_boxes
    # Non-blocking check: if lock is held (pre-warm loading), skip this tick
    if _yolo_lock is not None and not _yolo_models:
        acquired = _yolo_lock.acquire(blocking=False)
        if not acquired:
            return yolo_boxes
        _yolo_lock.release()
    _get_yolo()  # ensure models are loaded
    if not _yolo_models:
        if not getattr(_run_yolo_on_image, '_warned', False):
            print(f"[Pipeline] YOLO unavailable: {_yolo_status}")
            _run_yolo_on_image._warned = True
        return yolo_boxes
    # Parse context: 'all' = run everything, 'a+b' = run tags a or b,
    # single tag = run only that tag.
    if context == "all":
        wanted_tags = None  # None = no filter
    else:
        wanted_tags = set(t.strip() for t in context.split("+") if t.strip())
    # Per-detector inference imgsz. ui_v1 was trained on 2255×1268 frames
    # at imgsz=960, but pipeline captures at 3840×2160 (4K MuMu). Default
    # imgsz=640 loses small UI elements completely (verified: 0 detections
    # on lobby tick 1). 1920 brings detection back to expected mAP. Other
    # detectors were trained at smaller native frame sizes — 960 is fine.
    _IMGSZ_BY_TAG = {
        # ui_v2 trained at imgsz=960 on 2475×1392 frames. MUST infer at 960:
        # verified 2026-05-28 that v2 @ imgsz=1920 → 0 detections, but @ 960 →
        # 一次领取黄色 conf 0.936 etc. (The earlier 1920 "4K fix" was wrong —
        # production frames are 2475×1392, not 4K; the occasional 4K frame was
        # a capture-path glitch, not the norm.)
        "ui": 960,
        "avatar": 960,     # fused_avatar trained at 960
        "battle": 960,
        "cafe": 640,       # emoticon — 1 class, simple, default ok
    }
    # Standalone emoticon (tag "cafe") actually running in THIS call? The
    # ui-emoticon yield rule below must only fire when v26n is really there —
    # after fold-in the model is unloaded and ui's 451 IS the headpat source.
    _standalone_emo_active = any(
        t == "cafe" and (wanted_tags is None or t in wanted_tags)
        for _y, _c, t in _yolo_models)
    for yolo, model_conf, model_tag in _yolo_models:
        if wanted_tags is not None and model_tag not in wanted_tags:
            continue
        try:
            ifsz = _IMGSZ_BY_TAG.get(model_tag, 960)
            yolo_results = yolo(img, conf=model_conf, imgsz=ifsz, verbose=False)
            for r in yolo_results:
                for box in r.boxes:
                    bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0])
                    cls_name = yolo.names.get(cls_id, str(cls_id))
                    cls_low = str(cls_name).lower()
                    # ui carries a folded Emoticon_Action (cls451). When the
                    # standalone v26n (tag "cafe", 0.995) is ALSO running it is
                    # the emoticon AUTHORITY — drop the ui copy, else the two
                    # models double-box every bubble (offset boxes, IoU<0.6 →
                    # dedup can't catch) = ghosting + "emoticon 和 ui 抢信用点"
                    # (live 2026-06-09). After fold-in (2026-06-11) v26n is
                    # unloaded → _standalone_emo_active False → ui 451 passes.
                    if (model_tag == "ui" and "emoticon" in cls_low
                            and _standalone_emo_active):
                        continue
                    nx1, ny1, nx2, ny2 = bx1/w, by1/h, bx2/w, by2/h
                    # Filter headpat/emoticon: only accept in cafe play area
                    if "headpat" in cls_low or "emoticon" in cls_low:
                        bcx = (nx1 + nx2) / 2
                        bcy = (ny1 + ny2) / 2
                        if bcy < 0.15 or bcy > 0.85:
                            continue
                        bw = nx2 - nx1
                        bh = ny2 - ny1
                        if bw < 0.02 or bh < 0.02:
                            continue
                    yolo_boxes.append(YoloBox(
                        cls_id=cls_id,
                        cls_name=cls_name,
                        confidence=float(box.conf[0]),
                        x1=nx1,
                        y1=ny1,
                        x2=nx2,
                        y2=ny2,
                        model_tag=model_tag,
                    ))
        except Exception as e:
            print(f"[Pipeline] YOLO detect error: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
    # ── Cross-class / cross-model region dedup (user rule 2026-06-09:
    # "一个框的区域不能重复然后检测出另外的东西"). One screen region = ONE
    # detection: when boxes from different classes/models overlap heavily
    # (IoU>0.6), keep only the highest-confidence one. Kills e.g. the
    # earnings-popup 信用点 icon (ui 0.9+) ALSO firing as Emoticon_Action
    # (cafe model 0.38-0.58) = ghosted double box. Small-on-big overlaps
    # (红点 on an entry icon) have tiny IoU and are never deduped.
    if len(yolo_boxes) > 1:
        # Pre-pass — DOMAIN AUTHORITY, not confidence: the emoticon model's
        # (tag "cafe") only legit target is a headpat bubble, which never
        # overlaps a UI element. Any emoticon box overlapping (IoU>0.3) a box
        # from another model is an FP on that element → drop it EVEN IF its
        # conf is higher (live 2026-06-09: emoticon 0.75 on the 2號店 credit
        # icon outranked ui and "won" the conf-desc dedup — wrong winner).
        _others = [b for b in yolo_boxes if b.model_tag != "cafe"]
        if _others:
            def _emo_on_ui(e: YoloBox) -> bool:
                ea = max((e.x2 - e.x1) * (e.y2 - e.y1), 1e-9)
                for o in _others:
                    ix1, iy1 = max(e.x1, o.x1), max(e.y1, o.y1)
                    ix2, iy2 = min(e.x2, o.x2), min(e.y2, o.y2)
                    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                    oa = max((o.x2 - o.x1) * (o.y2 - o.y1), 1e-9)
                    if inter / (ea + oa - inter) > 0.3:
                        return True
                return False
            yolo_boxes = [b for b in yolo_boxes
                          if b.model_tag != "cafe" or not _emo_on_ui(b)]
        kept: List[YoloBox] = []
        for b in sorted(yolo_boxes, key=lambda x: -x.confidence):
            bw, bh = b.x2 - b.x1, b.y2 - b.y1
            area_b = max(bw * bh, 1e-9)
            dup = False
            for k in kept:
                ix1, iy1 = max(b.x1, k.x1), max(b.y1, k.y1)
                ix2, iy2 = min(b.x2, k.x2), min(b.y2, k.y2)
                iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
                inter = iw * ih
                area_k = max((k.x2 - k.x1) * (k.y2 - k.y1), 1e-9)
                iou = inter / (area_b + area_k - inter)
                # emoticon (tag "cafe") vs a higher-conf box from another
                # model: suppress at the LOOSER 0.3 — its FPs sit ON ui icons
                # (信用点 card 0.9 ui vs 0.38-0.58 emoticon) but with offset
                # boxes that rarely clear 0.6. Real bubbles overlap nothing.
                thr = 0.3 if ("cafe" in (b.model_tag, k.model_tag)
                              and b.model_tag != k.model_tag) else 0.6
                if iou > thr:
                    dup = True
                    break
            if not dup:
                kept.append(b)
        yolo_boxes = kept
    return yolo_boxes


# ── Top-level detector helpers for skills ──────────────────────────────
# Skills should NOT call _run_yolo_on_image directly — use these.  They
# operate on the current ScreenState's yolo_boxes (already populated each
# tick by the pipeline observation step) so there's no extra inference cost.

def find_yolo_box(screen: ScreenState, class_names: List[str],
                   min_conf: float = 0.3) -> Optional[YoloBox]:
    """Return the highest-confidence YoloBox matching any of class_names,
    or None.  Use this in skills to replace OCR-driven button finding:

        # OLD:
        btn = screen.find_any_text(["一次領取", "一次领取"], min_conf=0.6)
        # NEW:
        btn = find_yolo_box(screen, ["一次领取_黄", "一次领取"], min_conf=0.5) \\
              or screen.find_any_text(["一次領取", "一次领取"], min_conf=0.6)

    Matches by class_name exact-equal first, then case-insensitive substring
    fallback.  Returns the box with highest confidence among matches.
    """
    if not screen.yolo_boxes:
        return None
    name_set = set(class_names)
    name_lower = [n.lower() for n in class_names]
    hits: List[YoloBox] = []
    for b in screen.yolo_boxes:
        if b.confidence < min_conf:
            continue
        # exact match
        if b.cls_name in name_set:
            hits.append(b)
            continue
        # substring fallback
        bn = (b.cls_name or "").lower()
        if any(q in bn or bn in q for q in name_lower):
            hits.append(b)
    if not hits:
        return None
    return max(hits, key=lambda x: x.confidence)


def find_all_yolo_boxes(screen: ScreenState, class_names: List[str],
                         min_conf: float = 0.3) -> List[YoloBox]:
    """Like find_yolo_box but returns ALL matches sorted by confidence."""
    if not screen.yolo_boxes:
        return []
    name_set = set(class_names)
    name_lower = [n.lower() for n in class_names]
    hits: List[YoloBox] = []
    for b in screen.yolo_boxes:
        if b.confidence < min_conf:
            continue
        if b.cls_name in name_set:
            hits.append(b); continue
        bn = (b.cls_name or "").lower()
        if any(q in bn or bn in q for q in name_lower):
            hits.append(b)
    return sorted(hits, key=lambda x: -x.confidence)


def read_screen_from_frame(frame_bgr, *, screenshot_path: str = "",
                           skip_ocr: bool = False,
                           prev_ocr_boxes=None,
                           injected_yolo_boxes=None) -> ScreenState:
    """Build ScreenState from an in-memory BGR numpy array (no file I/O).

    Used by the MuMu runner for zero-copy capture → detect pipeline.

    Args:
        skip_ocr: if True, skip OCR (expensive ~50ms) and reuse prev_ocr_boxes.
        prev_ocr_boxes: OCR boxes from a previous tick to reuse when skip_ocr=True.
        injected_yolo_boxes: pre-computed YOLO boxes from high-FPS thread.
            If provided, skip running YOLO here (already done at high FPS).
    """
    if frame_bgr is None:
        return ScreenState(screenshot_path=screenshot_path)
    h, w = frame_bgr.shape[:2]
    if injected_yolo_boxes is not None:
        yolo_boxes = injected_yolo_boxes
    else:
        yolo_boxes = _run_yolo_on_image(frame_bgr, w, h)
    # ⭐OCR on-demand (2026-07-11 用户铁律: OCR 只在读数字/盲区兜底时跑):
    # YOLO ≥3 框 = 已知屏, cls 主导, 整帧 OCR 纯浪费(实测每次 1-1.5s, 旧策略
    # 每 3 tick 一跑拖慢全链)。YOLO <3 框 = 未知屏(羁绊升级/通知弹窗/TOUCH
    # START 等 OCR 拦截器兜底场景) → 才跑整帧 OCR。数字读取(票数/AP/总价)
    # 各 skill 本来就走 screen.frame 裁剪 digit-OCR, 不依赖这里。
    # 2026-07-11 用户二次收紧("OCR只有花钱/用票时启动"): 门从 <3框 收到
    # **完全零检出的亮屏**才跑 — 加载中转场(加载中 cls ≥1框)/普通页一律
    # 零 OCR(旧 <3框 门在每个转场帧白跑 1-1.5s); 钱/票数字=skill 内裁剪
    # digit-OCR 天然按需, 与整帧 OCR 无关。
    if skip_ocr and len(yolo_boxes) >= 1:
        ocr_boxes = prev_ocr_boxes if prev_ocr_boxes is not None else []
    else:
        ocr_boxes = _run_ocr_on_image(frame_bgr, w, h)
    return ScreenState(
        ocr_boxes=ocr_boxes,
        yolo_boxes=yolo_boxes,
        image_w=w,
        image_h=h,
        screenshot_path=screenshot_path,
        frame=frame_bgr,   # kept for on-demand digit-OCR cropping
    )


# ── Pipeline ────────────────────────────────────────────────────────────

@dataclass
class SkillResult:
    skill_name: str
    status: str  # "done", "timeout", "error"
    ticks: int
    duration_s: float
    reason: str = ""


class DailyPipeline:
    """Main automation pipeline. Sequences skills for daily routine.

    The pipeline:
    1. Reads the screen (OCR)
    2. Passes ScreenState to the current skill
    3. Skill returns an action
    4. Pipeline executes the action via WindowsInput
    5. If skill returns "done", advance to next skill
    6. If skill times out, retry or skip
    """

    # Fallback skill sequence for direct DailyPipeline() use (no server).
    # ⭐canonical 在 server/app.py _DEFAULT_SKILL_ORDER(2026-07-11 用户定死,
    # 那边是唯一权威, 改序改那边) — 此处仅同步拷贝: 收菜攒AP → 纯票扫荡 →
    # 学园交流会(吃AP) → 活动(剩余AP全灌) → 战术大赛 → 邮件 → 每日领奖 →
    # 活动再跑一轮(消化 mail/任务回灌的新AP, AP<20 自动秒过)。
    DEFAULT_SKILLS = [
        "daily_routine",
        "bounty",
        "jfd",
        "event_quest",
        "arena",
        "mail",
        "daily_mission",
        "event_quest",
    ]

    TRAJECTORIES_DIR = Path(__file__).resolve().parents[1] / "data" / "trajectories"

    def __init__(self, skill_names: Optional[List[str]] = None,
                 profile_options: Optional[Dict[str, Any]] = None):
        opts = dict(profile_options or {})  # reserved for future per-skill config

        self._skill_registry: Dict[str, BaseSkill] = {
            "buy_pyroxene": BuyPyroxeneSkill(),
            "cafe": CafeSkill(),
            "schedule": ScheduleSkill(),
            "club": ClubSkill(),
            "shop": ShopSkill(),
            "craft": CraftSkill(),
            "momo_talk": MomoTalkSkill(),
            "story_mining": StoryMiningSkill(),
            "bounty": BountySkill(),
            "jfd": JointFiringDrillSkill(),
            "batch_sweep": BatchSweepSkill(),
            # 智能 AP 分配: 扫 2x/3x bonus 板块(今天特殊任务). 排在 batch_sweep 前 —
            # 有 bonus 先吃, 没有就退、batch_sweep 兜底扫正常关.
            "special_sweep": SpecialSweepSkill(),
            # 活动 AP 规划器 (2026-07-08): 排 special_sweep 前, 活动吃 AP 优先。
            "event_quest": EventQuestSkill(),
            "arena": ArenaSkill(),
            # 战术大赛商店买体力 (花战术大赛货币, 非青辉石). NOT in DEFAULT_SKILLS —
            # run via skill_order/sub_only for the confirm-step live calibration,
            # integrate into harvest after verify.
            "arena_shop": ArenaShopSkill(),
            "mail": MailSkill(),
            "daily_mission": DailyMissionSkill(),
            # Single dispatcher for all dot-gated daily-harvest sub-flows
            # (buy_pyroxene / club / craft / shop / cafe / schedule / momo_talk
            #  / story_mining / mail / daily_mission). Battle skills
            # (bounty / jfd / arena) stay as separate skill_order entries.
            "daily_routine": DailyRoutineSkill(sub_only=opts.get("sub_only")),
        }
        # 审计 #3 (2026-06-17): registry 的 shop 是 TOP-LEVEL 单跑实例, 后面没有
        # arena_shop 承接留店 → 必须正常退大厅(否则单跑完游戏卡在商店网格)。
        # daily_routine 内部自己 new 的 ShopSkill 保持 chain_in_shop=True(紧跟
        # arena_shop, 同访问切战术大赛 tab)。
        self._skill_registry["shop"].chain_in_shop = False

        names = skill_names or self.DEFAULT_SKILLS
        self._skill_order: List[str] = [n for n in names if n in self._skill_registry]
        self._current_idx: int = 0
        self._running: bool = False
        self._results: List[SkillResult] = []
        self._skill_start_time: float = 0.0
        self._run_start_ts: float = 0.0  # start() 时刻 — 启动期(强更确认)窗口锚点
        self._total_ticks: int = 0
        self._max_retries: int = 1
        self._retry_count: int = 0
        self._traj_dir: Optional[Path] = None
        self._traj_writer_queue: "queue.Queue" = queue.Queue(maxsize=64)
        self._traj_writer_thread: Optional[threading.Thread] = None
        self._interceptor_streak: int = 0  # consecutive interceptor fires
        self._last_sub_state: str = ""
        self._last_wait_reason: str = ""
        self._last_action_reason: str = ""
        self._stuck_counter: int = 0  # ticks in same sub_state
        self._consecutive_timeouts: int = 0  # skills that timed out in a row
        self._last_click_target: Optional[list] = None
        self._last_click_reason: str = ""
        self._click_repeat_count: int = 0

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def current_skill(self) -> Optional[BaseSkill]:
        if not self._running or self._current_idx >= len(self._skill_order):
            return None
        name = self._skill_order[self._current_idx]
        return self._skill_registry.get(name)

    @property
    def progress(self) -> Dict[str, Any]:
        """Get current pipeline progress."""
        skill = self.current_skill
        return {
            "running": self._running,
            "current_skill": skill.name if skill else None,
            "current_sub_state": skill.sub_state if skill else None,
            "skill_index": self._current_idx,
            "total_skills": len(self._skill_order),
            "skill_ticks": skill.ticks if skill else 0,
            "total_ticks": self._total_ticks,
            "current_reason": self._last_action_reason,
            "results": [
                {
                    "skill": r.skill_name,
                    "status": r.status,
                    "ticks": r.ticks,
                    "reason": r.reason,
                }
                for r in self._results
            ],
        }

    def start(self) -> None:
        """Start the pipeline from the first skill."""
        self._running = True
        self._current_idx = 0
        self._results = []
        self._total_ticks = 0
        self._run_start_ts = time.time()
        self._retry_count = 0
        # Create trajectory directory for this run
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._traj_dir = self.TRAJECTORIES_DIR / f"run_{ts}"
        self._traj_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Pipeline] Trajectory dir: {self._traj_dir}")
        # Start async trajectory writer (drops oldest when queue full)
        self._traj_writer_queue = queue.Queue(maxsize=64)
        self._traj_writer_thread = threading.Thread(
            target=self._traj_writer_loop, name="TrajWriter", daemon=True,
        )
        self._traj_writer_thread.start()
        self._start_current_skill()
        print(f"[Pipeline] Started with {len(self._skill_order)} skills: {self._skill_order}")

    def stop(self) -> None:
        """Stop the pipeline."""
        self._running = False
        set_yolo_context("none")
        # Signal writer to drain and exit (don't block agent UI)
        try:
            self._traj_writer_queue.put_nowait(None)
        except (queue.Full, AttributeError):
            pass
        print("[Pipeline] Stopped")

    # Skill name → lobby-badge key.  User rule (2026-05-13):
    # "黄/红点才进，没点 pass to next" applies GLOBALLY to every
    # collection skill, not just bottom-nav ones.  The lobby badge
    # scanner returns badges from multiple regions:
    #   - bottom nav: cafe / schedule / student / edit / social /
    #     craft / shop / recruit
    #   - top-right cluster: mail (envelope icon)
    #   - left sidebar: daily_tasks_nav (任務 8/8 indicator)
    #   - right sidebar: campaign_nav (活動進行中 / 任務 tile)
    #
    # Bounty / Arena both enter via the right-sidebar 任務 tile.  The
    # tile dot semantics (per user 2026-05-13):
    #   yellow → bounty/arena tickets available (悬赏通缉用黄点识别)
    #   red    → event tasks have unclaimed rewards (活动任务红点)
    # Either way the campaign tile is the entry, so any non-none state
    # means "go look in there".  When the tile shows "none" both
    # tickets are drained AND tasks claimed — safe to skip.
    #
    # EventActivity stays unmapped — its primary entry is the lobby
    # carousel banner (which has its own indicator), and the inside-
    # event _scan_event_nav_red_badges handles the within-event nav
    # task-claim routing.
    _SKILL_BADGE_MAP: Dict[str, str] = {
        "Cafe":       "cafe",
        "Schedule":   "schedule",
        "Club":       "social",
        "Craft":      "craft",
        "Shop":       "shop",
        "Mail":       "mail",
        "Bounty":     "campaign_nav",
        "JointFiringDrill": "campaign_nav",
        "Arena":      "campaign_nav",
    }

    def _should_skip_skill_by_badge(self, skill: BaseSkill) -> Optional[str]:
        """Return a skip reason if this skill's nav-icon has no badge.

        User rule: "黄点就说明没打完还有的东西打，红点就说明有东西可以
        领取".  No dot means there's nothing to claim and nothing pending
        at that location — safely skip the skill instead of burning
        ticks navigating in and back out.

        Important: consult the MOST RECENT lobby snapshot, not the first
        one — mid-run actions (e.g. DailyTasks claiming) can refresh
        notifications that light up social/craft/etc. badges that
        weren't there at pipeline start.  Using first_seen would skip
        Club even though social just lit up red, losing a claim
        (run_20260513_205321: initial all-clear, social=red appeared
        mid-run, Club still skipped → user noticed missing AP claim).

        Returns:
            str reason if we should skip (caller logs + advances).
            None if the skill should run normally.
        """
        badge_key = self._SKILL_BADGE_MAP.get(skill.name)
        if not badge_key:
            return None  # No badge mapping → always run
        # Prefer last_seen (refreshed every advance_skill); fall back to
        # first_seen if last_seen hasn't populated yet.
        snapshot = (
            getattr(self, "_lobby_badges_last_seen", None)
            or getattr(self, "_lobby_badges_first_seen", None)
        )
        if not snapshot:
            return None  # We haven't seen the lobby yet → can't decide, run
        state = snapshot.get(badge_key, "unknown")
        if state in ("red", "yellow"):
            return None  # Has a dot → something to do, run
        if state == "none":
            return f"no {badge_key} badge on lobby nav (nothing to claim/play)"
        return None  # unknown / not detected → run

    def _start_current_skill(self) -> None:
        """Reset and start the currently selected skill.

        If lobby-badge skip routing decides the skill has no work to do,
        record a skip result and recurse into the next skill instead.
        """
        if not self._running:
            return
        skill = self.current_skill
        if skill is None:
            self._running = False
            return

        # Badge-based skip BEFORE skill.reset() (avoid resetting state we
        # won't use).  Recurse into the next skill if we skip.
        skip_reason = self._should_skip_skill_by_badge(skill)
        if skip_reason:
            print(f"[Pipeline] Skipping skill '{skill.name}': {skip_reason}")
            self._results.append(
                SkillResult(
                    skill_name=skill.name,
                    status="skipped",
                    ticks=0,
                    duration_s=0.0,
                    reason=skip_reason,
                )
            )
            self._current_idx += 1
            if self._current_idx >= len(self._skill_order):
                self._running = False
                self._log_lobby_badge_summary()
                print("[Pipeline] All skills complete")
                return
            self._start_current_skill()
            return

        try:
            skill.reset()
        except Exception as e:
            print(f"[Pipeline] Failed to reset skill '{skill.name}': {e}")
            raise
        self._skill_start_time = time.time()
        self._retry_count = 0
        self._last_sub_state = ""
        self._last_wait_reason = ""
        self._stuck_counter = 0
        # Multi-frame dot gate state (reset per skill). The red/yellow dot cls
        # flickers frame-to-frame, so should_run is voted over the first few
        # frames instead of decided on a single (possibly flickered) frame.
        self._dot_gate_done = False
        self._dot_gate_ticks = 0
        # Set YOLO context based on skill — only run relevant model(s).
        # FPS FIX (2026-05-29): base is now "ui" ONLY. The avatar model is
        # fused_avatar yolo26X (the heaviest net) — running it every tick just
        # to click buttons was the main inference cost. Nothing on the daily
        # nav/sweep path needs per-student identification, so avatar is dropped
        # from the base and only added back where a skill genuinely needs to
        # know WHICH student (none in the current run path). Add 'avatar' to a
        # skill's context here if/when student-id is required.
        # Loadout map lives at module level (SKILL_YOLO_MAP).
        yolo_ctx = SKILL_YOLO_MAP.get(skill.name, BASE_DETECTORS)
        set_yolo_context(yolo_ctx)
        print(f"[Pipeline] Starting skill '{skill.name}'")

    def _advance_skill(self, status: str, reason: str = "") -> None:
        """Record current skill result and advance to the next skill."""
        skill = self.current_skill
        if skill is None:
            self._running = False
            return
        duration_s = max(0.0, time.time() - self._skill_start_time)
        self._results.append(
            SkillResult(
                skill_name=skill.name,
                status=status,
                ticks=skill.ticks,
                duration_s=duration_s,
                reason=str(reason or ""),
            )
        )
        if status == "timeout":
            self._consecutive_timeouts += 1
        else:
            self._consecutive_timeouts = 0
        self._current_idx += 1
        self._retry_count = 0
        self._last_sub_state = ""
        self._last_wait_reason = ""
        self._stuck_counter = 0
        # Opportunistic lobby badge scan between skills — we're usually
        # transiting through lobby anyway, so it's free.  See
        # _maybe_scan_lobby_badges for the deduplication logic.
        self._maybe_scan_lobby_badges(reason="advancing skill")
        if self._current_idx >= len(self._skill_order):
            self._running = False
            self._log_lobby_badge_summary()
            print("[Pipeline] All skills complete")
            return
        self._start_current_skill()

    # ── Lobby badge tracking ─────────────────────────────────────────
    #
    # The 8 bottom-nav icons each render a small dot when something is
    # actionable:
    #   • RED dot   → unclaimed reward at that location
    #   • YELLOW dot → unfinished task / new content at that location
    # We scan the lobby a few times during a run (at start, between
    # skills, at end) and diff to see what got cleared vs. what was
    # missed.  Future: use the initial scan to skip skills whose icon
    # has no dot (no work to do there).
    def _maybe_scan_lobby_badges(self, *, reason: str = "") -> None:
        screen = getattr(self, "_last_screen", None)
        if screen is None or not screen.is_lobby():
            return
        try:
            badges = screen.scan_lobby_nav_badges()
        except Exception as exc:  # noqa: BLE001 — never break the pipeline
            print(f"[LobbyBadge] scan failed: {exc}")
            return
        if not badges:
            return
        # First-seen snapshot (locked once we get our first lobby-on read)
        if not getattr(self, "_lobby_badges_first_seen", None):
            self._lobby_badges_first_seen = dict(badges)
            non_none = {k: v for k, v in badges.items() if v != "none"}
            if non_none:
                print(f"[LobbyBadge] initial dots ({reason}): {non_none}")
            else:
                print(f"[LobbyBadge] initial scan ({reason}): all clear")
        # Most-recent snapshot always updated
        self._lobby_badges_last_seen = dict(badges)

    def _log_lobby_badge_summary(self) -> None:
        first = getattr(self, "_lobby_badges_first_seen", None) or {}
        last = getattr(self, "_lobby_badges_last_seen", None) or {}
        if not first and not last:
            return
        cleared = []   # had dot at start, gone at end
        remaining = [] # still has dot at end (missed work / partial run)
        new = []       # didn't have at start, appeared by end
        keys = set(first) | set(last)
        for k in sorted(keys):
            f = first.get(k, "none")
            l = last.get(k, "none")
            if f != "none" and l == "none":
                cleared.append(f"{k}:{f}→clear")
            elif f != "none" and l != "none":
                remaining.append(f"{k}:{l}")
            elif f == "none" and l != "none":
                new.append(f"{k}:{l}")
        if cleared:
            print(f"[LobbyBadge] cleared this run: {cleared}")
        if remaining:
            print(f"[LobbyBadge] STILL pending: {remaining}")
        if new:
            print(f"[LobbyBadge] newly appeared: {new}")

    def _global_interceptor(self, screen: ScreenState, skill: BaseSkill) -> Optional[Dict[str, Any]]:
        """Global interceptor — runs BEFORE every skill tick.

        Handles "rude" popups that can appear at any time regardless of
        skill — pure cls: 获得奖励 / 羁绊·地区升级 / 启动期强更下载确认 /
        加载中等待。其余弹窗归 skill 语境 + base._handle_common_popups。
        """
        # ════════════════════════════════════════════════════════════════
        # PURE-YOLO interceptor (2026-05-29; 2026-07-17 死 OCR 段整删).
        # Only cls-backed popups are auto-handled here; ambiguous
        # confirm/cancel dialogs are left to the owning skill /
        # base._handle_common_popups (they know the context) so we don't
        # blind-confirm a 'visit friend cafe' / 'exit game' prompt.
        # 旧 OCR 分支段(P0 断线/退出框, P0.5 签到/选项/公告/指南任务,
        # P1 通知/promo/课程表弹窗X, P2 level-up/TAP TO CONTINUE/獲得獎勵
        # /羈絆)已整段删除(2026-07-17) — _OCR_ENABLED=False 恒假 →
        # ocr_boxes 恒空 → find_any_text 恒 None, 全部 dead code;
        # 活的同功能保护在 base._handle_common_popups cls 段。
        # ════════════════════════════════════════════════════════════════
        # Reward-result popup (获得奖励) — dismiss via 确认键, else tap.
        reward_y = find_yolo_box(screen, ["获得奖励"], min_conf=0.35)
        if reward_y:
            confirm_y = find_yolo_box(screen, ["确认键"], min_conf=0.30)
            if confirm_y:
                print("[Interceptor] YOLO reward popup → 确认键")
                return action_click_box(confirm_y, "interceptor: confirm reward (YOLO)")
            print("[Interceptor] YOLO reward popup → tap dismiss")
            return action_click(0.5, 0.92, "interceptor: dismiss reward (YOLO)")
        # Full-screen bond / region level-up — tap anywhere to advance.
        levelup_y = find_yolo_box(screen, ["羁绊升级", "地区升级"], min_conf=0.35)
        if levelup_y:
            print(f"[Interceptor] YOLO level-up ({levelup_y.cls_name}) → tap dismiss")
            return action_click(0.5, 0.5, "interceptor: dismiss level-up (YOLO)")

        # ── P-1: 强更下载确认框 (pure YOLO, 启动期专用, 2026-07-16 重构) ──
        # patch-day 冷启动在标题屏前后(TOUCH TO START 前后)弹"需要下載遊戲所
        # 需的檔案 X.XX GB"强更框。旧版 OCR 文字匹配(通知标题+下載 body 词表)
        # 已删 — 纯 相位+结构 判定:
        #   ① 启动期: 距 pipeline start() < _STARTUP_UPDATE_WINDOW_S
        #   ② 标题屏语境: 标题屏/强更框是未训练面, 全帧几乎零 ui cls
        #      (大厅误判免疫: lobby 常态 10+ 框)
        #   ③ 结构: 确认键在场, 且全帧无 取消键/弹窗叉叉/灰色确认
        #      (用户 spec: 强更框只有確認; 可取消的框不是强更 → 不碰)
        # 窗口外 / 结构不符: 这里一律不动(其余不动) — 只有确认的弹窗归当前
        # skill 的语境 handler。
        if (time.time() - getattr(self, "_run_start_ts", 0.0)
                < _STARTUP_UPDATE_WINDOW_S):
            _upd_confirm = find_yolo_box(screen, ["确认键"], min_conf=0.30)
            _upd_blocker = find_yolo_box(
                screen, ["取消键", "弹窗叉叉", "灰色确认"], min_conf=0.30)
            if (_upd_confirm is not None and _upd_blocker is None
                    and len(screen.yolo_boxes or []) <= 4):
                print("[Interceptor] 启动期 confirm-only 弹窗(强更下载框结构), "
                      "clicking 确认键")
                return action_click_box(
                    _upd_confirm,
                    "interceptor: confirm force-update download (startup YOLO)")

        # ── P-1: Global loading / update / download ──
        # During game startup the screen shows "正在更新", "Now Loading",
        # "驗證下載檔案中", "重置遊戲資料中", etc.  No skill should act
        # during these — just wait.  Also reset the current skill's enter
        # ticks so a long download doesn't trigger a premature timeout.
        if screen.is_loading():
            # Reset skill enter-tick counter so downloads don't count as
            # "stuck" time.  Many skills have _enter_ticks with hard limits.
            if hasattr(skill, '_enter_ticks'):
                skill._enter_ticks = max(0, getattr(skill, '_enter_ticks', 0) - 1)
            # no-UI escape 护栏喂点(2026-07-11): 加载序列的后期是零框帧,
            # no-UI back 曾在活动页加载到一半时把它拆掉(实锤×7)。记下
            # 最近一次「加载中」时刻, 12s 内禁止 no-UI escape/wake。
            self._last_loading_ts = time.time()
            return action_wait(1500, "interceptor: game loading / updating")

        # No interceptor fired — reset streak + do-not-show-again flag
        self._interceptor_streak = 0
        self._dnsa_toggled = False
        return None

    def tick_from_frame(self, frame_bgr, *, screenshot_path: str = "",
                        skip_ocr: bool = False,
                        prev_ocr_boxes=None,
                        injected_yolo_boxes=None,
                        fresh_boxes=None, fresh_frame=None,
                        fresh_ts: float = 0.0) -> Dict[str, Any]:
        """Process one in-memory BGR frame. Returns an action dict.

        Args:
            skip_ocr: skip expensive OCR, reuse prev_ocr_boxes.
            prev_ocr_boxes: cached OCR boxes from a previous tick.
            injected_yolo_boxes: pre-computed YOLO boxes from high-FPS thread.
                If provided, skip running YOLO in read_screen_from_frame.
            fresh_boxes/fresh_ts: 高频 DXcam 线程的最新检出+时间戳(2026-07-11
                工业级链路: 主 tick 帧龄 ~2.2s 对轮播类时敏目标必错位, skill
                可读 screen.fresh_boxes(帧龄≤0.5s@2FPS)做"有目标就点"判定)。
        """
        screen = read_screen_from_frame(frame_bgr, screenshot_path=screenshot_path,
                                        skip_ocr=skip_ocr,
                                        prev_ocr_boxes=prev_ocr_boxes,
                                        injected_yolo_boxes=injected_yolo_boxes)
        screen.fresh_boxes = fresh_boxes
        screen.fresh_frame = fresh_frame
        screen.fresh_ts = fresh_ts
        return self._tick_with_screen(screen, screenshot_path=screenshot_path)

    @property
    def last_screen(self) -> Optional[ScreenState]:
        """Last ScreenState processed by tick (for overlay access)."""
        return getattr(self, '_last_screen', None)

    # cls that flicker frame-to-frame (badges / currency digits) — excluded
    # from the stage fingerprint so a single dot blinking doesn't read as a
    # screen transition.
    _SIG_SKIP = frozenset({
        "红点", "黄点", "绿勾", "加号", "减号", "加载中",
        "信用点", "体力", "青辉石", "战术大赛商店货币",
    })

    def _screen_sig(self, screen) -> frozenset:
        """Coarse 'which stage are we on' fingerprint = the set of structural
        cls on screen (buttons/entries/tabs, conf≥0.5; badges/currency dropped).
        Transitioning to the next stage changes this set; flicker does not."""
        return frozenset(
            b.cls_name for b in (getattr(screen, "yolo_boxes", None) or [])
            if b.confidence >= 0.50 and b.cls_name not in self._SIG_SKIP)

    def _dedup_click(self, action: Dict[str, Any], screen=None) -> Dict[str, Any]:
        """Gated click (user 2026-06-13: 点了一个cls, 等下一阶段cls出现再点).

        A transition click is allowed ONCE, then HELD (converted to wait) until
        the screen fingerprint actually changes — the next stage rendered — or a
        generous cap (lost tap → one retry). This kills the frantic ADB re-click
        that re-fired the SAME button before the screen transitioned (live
        2026-06-13: arena_shop 商店入口 ×2 / schedule popout thrash → Location
        Select 乱走). 加载中 always waits (暂停 process 给游戏加载时间).

        Same-target steppers (e.g. MAX clicked defensively ×3) are unaffected:
        the FIRST tap is allowed (sets the value); the rest hold harmlessly.
        Different-target clicks (multi-select cards, room heads) always pass.
        """
        action_type = action.get("action", "")
        if action_type != "click":
            self._last_click_target = None
            self._last_click_sig = None
            self._click_hold = 0
            return action

        # ── universal loading gate: never tap mid-transition ──
        try:
            if screen is not None and screen.is_loading():
                return action_wait(500, "加载中 → 暂停, 等加载完成")
        except Exception:
            pass

        target = action.get("target")
        reason = str(action.get("reason", "") or "")
        sig = self._screen_sig(screen) if screen is not None else frozenset()
        last_target = getattr(self, "_last_click_target", None)
        last_sig = getattr(self, "_last_click_sig", None)

        # ── "看到目标就点" exemption (user 2026-06-13: 无端等待) ──────────────
        # Stacked popups (sweep/battle/event rewards, 領取/確認/continue, X-close)
        # sit at the SAME position and EACH layer must be clicked through. The
        # same-target hold below would HOLD the dismiss → the popup never gets
        # clicked → deadlock until the cap (the ~22s "无端等待"). These act on a
        # popup that IS on screen right now → click it immediately, never hold.
        _r = reason
        if any(k in _r for k in (
                "interceptor", "dismiss", "reward", "獎勵", "奖励", "result",
                "結果", "结果", "确认键", "確認", "continue", "繼續", "領取",
                "领取", "claim", "close", "关闭", "關閉", "叉叉")):
            self._last_click_target = target
            self._last_click_reason = reason
            self._last_click_sig = sig
            self._click_hold = 0
            return action

        # ── frame-settle gate(2026-07-17 用户"不要强拍"): 导航/进入类点击
        # 只在稳定帧放行 — 连续两帧结构指纹+质心一致(_tick_with_screen 每
        # tick 记录)。转场动画/列表滚动中指纹或质心持续变化 → 自然等待;
        # 稳定后首帧立即放行 = 零固定延迟。弹窗 dismiss/领取类在上方豁免
        # 通道已 return(点弹窗强拍无害)。>4s 未稳定放行一次(背景动画让
        # 某 cls 持续抖动时的死锁保险)。
        if not getattr(self, "_frame_stable", True):
            if not getattr(self, "_settle_block_t0", 0.0):
                self._settle_block_t0 = time.time()
            if time.time() - self._settle_block_t0 < 4.0:
                return action_wait(150, f"帧未稳定(转场/滚动) — 等稳定帧: {reason}")
            print(f"[Pipeline] settle-gate 4s 超时放行: {reason}", flush=True)
        self._settle_block_t0 = 0.0

        same_target = False
        if target and last_target:
            try:
                same_target = (abs(target[0] - last_target[0]) < 0.03
                               and abs(target[1] - last_target[1]) < 0.03)
            except (TypeError, IndexError):
                same_target = target == last_target

        if same_target and last_sig is not None:
            if not sig or not last_sig:
                # 空指纹 = 转场黑屏/模型盲区帧: 空∩空/空∪空 Jaccard=1.0 的
                # 数学假象把"转场中"误判成"页面没变"(2026-07-11 实锤 hold 拖
                # 25s)。转场中不重点击、也不吃 hold 计数 — 等渲染出 cls 再判。
                return action_wait(450, "转场渲染中(空指纹) — 等 cls 出现")
            union = sig | last_sig
            jacc = (len(sig & last_sig) / len(union)) if union else 1.0
            if jacc >= 0.5:
                # Screen still the same stage → the click hasn't landed/rendered
                # → HOLD instead of re-tapping. 墙钟上限(2026-07-11 链路审计:
                # 旧 5-tick 上限 ×~2s/tick ≈9s, 丢 tap 恢复太慢, 轮播类目标必
                # miss) — 2.5s 覆盖正常转场渲染, 丢 tap 快速重试。
                self._click_hold = getattr(self, "_click_hold", 0) + 1
                if self._click_hold == 1:
                    self._click_hold_t0 = time.time()
                if time.time() - getattr(self, "_click_hold_t0", 0.0) < 2.5:
                    return action_wait(
                        450, f"等下一阶段cls出现 "
                             f"(hold {self._click_hold}, <2.5s): {reason}")
                # 墙钟到 → tap likely lost → allow ONE retry, reset.
                print(f"[Pipeline] click hold 2.5s → 重试: {reason}", flush=True)
                self._click_hold = 0
                self._last_click_sig = sig
                return action
            # else: screen advanced → genuine new action, fall through to record.

        self._last_click_target = target
        self._last_click_reason = reason
        self._last_click_sig = sig
        self._click_hold = 0
        return action

    def _tick_with_screen(self, screen: ScreenState, *, screenshot_path: str = "") -> Dict[str, Any]:
        """Internal tick logic for tick_from_frame()."""
        self._last_screen = screen
        if not self._running:
            return action_done("pipeline not running")

        self._total_ticks += 1

        # ── frame-settle 历史(_dedup_click 稳定门用, 2026-07-17 用户"不要
        # 强拍"): 每 tick 记录 结构指纹+各cls质心, 连续两帧一致=页面稳定。
        # 转场动画/列表滚动期间指纹或质心持续变化 → 导航类点击自然等待;
        # 稳定后首帧立即放行 = 零人为延迟, 纯事件驱动。
        _sig_now = self._screen_sig(screen)
        _cent_now = {}
        for _b in (getattr(screen, "yolo_boxes", None) or []):
            if _b.confidence >= 0.50 and _b.cls_name in _sig_now:
                _c = _cent_now.setdefault(_b.cls_name, [0.0, 0.0, 0])
                _c[0] += _b.cx
                _c[1] += _b.cy
                _c[2] += 1
        _cent_now = {k: (v[0] / v[2], v[1] / v[2])
                     for k, v in _cent_now.items() if v[2]}
        _sig_prev = getattr(self, "_settle_sig", None)
        _cent_prev = getattr(self, "_settle_cent", None)
        _stable = False
        if _sig_prev is not None and _sig_now and _sig_now == _sig_prev:
            _stable = all(
                abs(_cent_now[k][0] - _cent_prev.get(k, _cent_now[k])[0]) < 0.02
                and abs(_cent_now[k][1] - _cent_prev.get(k, _cent_now[k])[1]) < 0.02
                for k in _cent_now)
        self._settle_sig, self._settle_cent = _sig_now, _cent_now
        self._frame_stable = _stable

        # Hard-example mining 已停用(2026-07-16 审计 A 级): 每 tick 写盘
        # data/hard_examples 累积 7GB/2万文件, dashboard 从未接消费入口 —
        # 同需求由干净帧飞轮 + scripts/mine_hard_examples.py(读 trajectories
        # 写 raw_images/_hard_examples_*) 覆盖。目录确认无用后可整体回收。

        # 专注模型 loadout per sub-state (user 2026-06-09: "emoticon 只在要开始
        # 摸头的时候才识别, 摸完/换楼就关"): refine the skill-level
        # SKILL_YOLO_MAP context by the ACTIVE sub-skill + sub_state each tick.
        # Only Cafe has state-dependent needs today; everything else keeps the
        # skill-level loadout set at skill start.
        try:
            _cur = self.current_skill
            _sname = getattr(_cur, "name", "")
            _sstate = getattr(_cur, "sub_state", "")
            if _cur is not None and hasattr(_cur, "_plan") and hasattr(_cur, "_cur_idx"):
                try:
                    _sub = _cur._plan[_cur._cur_idx][0]
                    _sname = getattr(_sub, "name", _sname)
                    _sstate = getattr(_sub, "sub_state", "")
                except Exception:
                    pass
            if _sname == "Cafe":
                if _sstate in ("headpat", "headpat2"):
                    set_yolo_context("ui+cafe")    # emoticon ON only while patting
                elif _sstate == "invite":
                    set_yolo_context("ui+avatar")  # row avatars; no emoticon
                else:
                    set_yolo_context("ui")         # enter/earnings/switch/exit
        except Exception:
            pass

        # ── Lobby resource snapshot + 青辉石 kill-switch ──────────────────
        try:
            _now = time.time()
            if _now - _RESOURCES["ts"] > 30 and screen.yolo_boxes:
                from brain.skills.ui_classes import (
                    LOBBY_NAV_ICONS, TOPBAR_AP, TOPBAR_CREDIT, TOPBAR_PYROXENE)
                _navs = sum(1 for b in screen.yolo_boxes
                            if b.cls_name in set(LOBBY_NAV_ICONS) and b.confidence >= 0.30)
                if _navs >= 2:  # lobby (nav bar fully visible) — top bar reliable
                    # Read from a CLEAN ADB frame (2026-06-11 rule): the live
                    # frame left-truncates these on transition (6497→497,
                    # AP→9999, credit→None). One clean YOLO pass per 30s snapshot
                    # is cheap; fall back to the live read only if no clean
                    # source is registered.
                    if get_clean_frame() is not None:
                        # 共享帧批读(2026-07-11): 旧版 3×5=16 连拍单 tick 阻塞
                        # 7-20s(链路审计元凶之一), 现 5 帧共享, 投票语义不变。
                        _vals = _read_topbar_clean_multi(
                            [TOPBAR_AP, TOPBAR_CREDIT, TOPBAR_PYROXENE])
                        _ap = _vals.get(TOPBAR_AP)
                        _cr = _vals.get(TOPBAR_CREDIT)
                        _py = _vals.get(TOPBAR_PYROXENE)
                    else:
                        _ap = _read_topbar_count(screen, TOPBAR_AP)
                        _cr = _read_topbar_count(screen, TOPBAR_CREDIT)
                        _py = _read_topbar_count(screen, TOPBAR_PYROXENE)
                    _prev_py = _RESOURCES.get("pyroxene")
                    if _py is not None:
                        if _prev_py is not None and _py < _prev_py:
                            # Suspected drop. The #1 false positive is OCR
                            # left-truncation (6497→497, stable across reads so
                            # the old "2 consecutive" guard never helped, and
                            # it false-tripped a breach 2026-06-11).
                            # ① A digit-count SHRINK is a truncation misread —
                            #    our skills never spend ~90% of the balance in
                            #    one 30s window. Reject outright.
                            if len(str(_py)) < len(str(_prev_py)):
                                print(f"[Pipeline] pyroxene {_prev_py}→{_py}: digit "
                                      f"shrink = OCR truncation, ignored (no spend)",
                                      flush=True)
                            else:
                                # ② Same/more digits → confirm on TWO independent
                                #    CLEAN ADB frames that AGREE (2026-07-17: 单次
                                #    干净帧复读被 7/4 换位误读骗过 12471→12447
                                #    误急停; 换位型误读每次错法不同, 两次独立读
                                #    到同一错值概率极低; 真掉钱两读同真值必抓)。
                                _py_c1 = _read_pyroxene_clean()
                                time.sleep(0.5)
                                _py_c2 = _read_pyroxene_clean()
                                if (_py_c1 is not None and _py_c1 == _py_c2
                                        and _py_c1 < _prev_py
                                        and len(str(_py_c1)) == len(str(_prev_py))):
                                    print(f"[Pipeline] ⛔⛔ PYROXENE DROPPED {_prev_py} → "
                                          f"{_py_c1} (2x clean-frame confirmed) — MONEY "
                                          f"BREACH, ABORTING PIPELINE", flush=True)
                                    self._running = False
                                    return action_done("⛔ pyroxene drop detected — aborted")
                                elif (_py_c1 is not None and _py_c2 is not None
                                        and max(_py_c1, _py_c2) >= _prev_py):
                                    # clean frame shows balance intact → glitch
                                    _RESOURCES["pyroxene"] = max(_py_c1, _py_c2)
                                else:
                                    print(f"[Pipeline] pyroxene {_prev_py}→{_py}: clean "
                                          f"re-reads={_py_c1}/{_py_c2} disagree or "
                                          f"unconfirmed, ignoring", flush=True)
                        else:
                            self._py_drop_pending = None
                            _RESOURCES["pyroxene"] = _py
                    if _ap is not None:
                        _RESOURCES["ap"] = _ap
                    if _cr is not None:
                        _RESOURCES["credits"] = _cr
                    # ts 无条件更新(2026-07-11): 旧版只在读到值时更新 → 全
                    # None 时下个 lobby tick 立即重跑整个连拍 = 重试风暴。
                    # 失败也等 30s 再试(kill-switch 周期不变)。
                    _RESOURCES["ts"] = _now
                    if _ap is not None or _cr is not None or _py is not None:
                        print(f"[Pipeline] resources: AP={_RESOURCES['ap']} "
                              f"credits={_RESOURCES['credits']} pyroxene={_RESOURCES['pyroxene']}",
                              flush=True)
        except Exception:
            pass

        # Early bail-out: BA / MuMu not actually visible.  Wide set of
        # markers so legitimate BA states (title screen, loading, login,
        # battle, any in-game menu) all count as "BA detected".  Only
        # genuinely-foreign captures (Claude Code chat, browser, desktop)
        # produce 0 matches.  Threshold raised to 30 because long boot /
        # asset-download sequences can show "Now Loading" for 15+ ticks
        # with no Chinese text in view.
        ba_markers = [
            # Lobby bottom-nav (always visible on lobby)
            "咖啡", "課程", "课程", "學生", "学生", "編輯", "编辑",
            "社交", "製造", "制造", "商店", "招募",
            # Lobby sidebar / top-right
            "MomoTalk", "公告", "任務", "任务", "信箱", "郵件", "邮件",
            # In-game / battle / event / mission markers
            "活動", "活动", "AP", "Auto", "AUTO", "戰鬥", "战斗",
            "入場", "入场", "出擊", "出击", "掃蕩", "扫荡",
            # Boot / title / loading (so we don't abort during start-up)
            "TOUCH", "TAP", "START", "Loading", "loading", "正在更新",
            "下載", "下载", "Now",
            # Cafe / Schedule / Club internals
            "咖啡廳", "咖啡厅", "課程表", "课程表", "社團", "社团",
            "Lv.", "Cok11",
            # Animated full-screen transitions (no nav-bar visible).
            # Bond level-up screen (run_20260516_234050 t232 stuck here
            # 29/30 because OCR only saw 絲升級 + 治愈力 + heart-25):
            "羈絆", "羁绊", "升級", "升级", "治癒", "治愈", "體力", "体力",
            # Mission-result / event-result / reward popup transitions
            "獲得", "获得", "獎勵", "奖励", "結果", "结果",
            "勝利", "胜利", "Victory", "VICTORY",
            # Battle skip / cutscene
            "Skip", "SKIP", "MENU", "Menu",
        ]
        # PURE-YOLO: BA is visible if the UI/avatar YOLO model detected
        # anything this tick. The ui model fires on virtually every BA
        # screen (nav bar, buttons, dots, currency widgets); foreign
        # captures (desktop/browser) produce ~none. The OCR-marker list
        # above is dead while _OCR_ENABLED is False (kept for the digit-OCR
        # re-enable phase) — without this YOLO primary, the pipeline would
        # mis-detect "no BA" every tick and self-abort at 30 ticks.
        has_ba_ui = bool(screen.yolo_boxes) or any(
            screen.find_any_text([m], min_conf=0.50) is not None
            for m in ba_markers
        )
        if has_ba_ui:
            self._no_ba_ticks = 0
            self._no_ba_since = 0.0
        else:
            self._no_ba_ticks = getattr(self, "_no_ba_ticks", 0) + 1
            if getattr(self, "_no_ba_since", 0.0) <= 0.0:
                self._no_ba_since = time.time()
            # Save trajectory of these wait ticks too so we can debug
            # WHY OCR didn't find any markers (was the frame black?  Did
            # OCR run at all?).  Old code skipped trajectory save which
            # left an empty run dir with no evidence.
            wait_action = action_wait(
                800,
                f"no YOLO boxes detected — waiting ({self._no_ba_ticks}/30) "
                f"[black screen / loading, or a full-screen overlay with no "
                f"trained cls — 有框才操作: we do NOT blind-tap]"
            )
            self._save_trajectory(screenshot_path, screen, None, wait_action)
            # ⛔ ui 模型没加载成功 = 假 no-UI(眼睛缺失非画面空), 立刻 abort,
            # 绝不走下面的 blind/wake-tap(2026-07-07: 假 no-UI + wake-tap 固定位
            # 撞上 topbar AP「+」→ 反复戳开購買AP框)。fail-closed: 无眼不动手。
            if _UI_LOAD_FAILED:
                print("[Pipeline] ⛔ ui model failed to load — aborting immediately "
                      "(fail-closed, no blind/wake taps without UI eyes). "
                      "Restart the server to reload the model.")
                self._running = False
                return action_done("pipeline aborted: ui model load failed")
            # abort 上限改时间制(2026-07-16): tick 提速到 ~0.3-0.7s 后,
            # 30 tick 只有 ~10-20s — 放置立绘屏(UI 自动隐藏)wake 还没生效
            # 就 abort 了(实锤)。45s 与旧版 30x1s+ 的等效窗口一致。
            if (self._no_ba_ticks >= 30
                    and time.time() - self._no_ba_since > 45.0):
                print(
                    f"[Pipeline] No Blue Archive UI detected for "
                    f"{self._no_ba_ticks} ticks / "
                    f"{time.time() - self._no_ba_since:.0f}s — aborting "
                    f"pipeline.  Check that MuMu is running, BA is "
                    f"launched, the emulator window isn't minimised, "
                    f"and the Window Title setting matches.  "
                    f"Saved {self._no_ba_ticks} no-UI frames to the "
                    f"trajectory dir for inspection."
                )
                self._running = False
                return action_done("pipeline aborted: no BA window detected")
            # PURE-YOLO: a full-screen "TOUCH TO CONTINUE" overlay (account
            # level-up etc.) carries NO YOLO cls, so it lands here. Blind-tap
            # to dismiss it (harmless on real loading / foreign frames, which
            # ignore taps) — alternate center / bottom-corner so we hit the
            # prompt without landing on a reward card that absorbs the tap.
            # Without this the overlay would sit until the 30-tick abort.
            # BRING-UP (有框才操作): with _BRINGUP_EXPOSE we do NOT blind-tap on
            # a 0-box screen — a black screen/loading should just wait, and an
            # undetected overlay is a HOLE to fix (train a cls), not paper over.
            # It then waits → 30-tick abort saves frames for inspection.
            if self._no_ba_ticks >= 2 and not _BRINGUP_EXPOSE:
                # Reward popups (獲得獎勵 + TOUCH TO CONTINUE) that the ui model
                # doesn't detect at all land here. Target the actual dismiss
                # zones FIRST: TOUCH-TO-CONTINUE text (~0.5,0.88) and the X
                # close (~0.88,0.15); a center tap hits the reward CARD which
                # absorbs the tap without dismissing. Then fall back to corners.
                pts = [(0.5, 0.88), (0.88, 0.15), (0.5, 0.95), (0.08, 0.92)]
                px, py = pts[self._no_ba_ticks % len(pts)]
                blind = action_click(px, py,
                    f"interceptor: blind-tap dismiss overlay ({self._no_ba_ticks}/30)")
                self._save_trajectory(screenshot_path, screen, None, blind)
                return blind
            # 放置立绘屏 (lobby idle showcase) wake-tap — EXPOSE-compatible.
            # The lobby auto-hides ALL UI after ~15s idle, leaving only the
            # Live2D character → ZERO YOLO cls → counts as no-UI → 30-tick abort
            # (live 2026-06-15: momo_talk 等大厅门控 skill 全被它卡死/误skip).
            # A BRIGHT (non-black) 0-box frame is the showcase, NOT a black
            # loading screen — a SINGLE benign wake-tap on the empty TOP-CENTER
            # sky (0.5,0.05: never the character → no touch-dialogue; never a nav
            # button → no wandering) reveals the UI so the next tick sees it.
            # This is NOT the prohibited blind-nav (no ESC/返回/return-to-lobby):
            # it only wakes an idle screen. Abort safety net preserved — if the
            # tap can't reveal UI, _no_ba_ticks keeps climbing → 30-tick abort.
            try:
                _bright = screen.frame is not None and float(screen.frame.mean()) > 25.0
            except Exception:
                _bright = False
            # skill 声明的 no-UI 逃生 (2026-07-09): event_quest 点 405 可能落进
            # 模型盲区页(特殊作戰運輸船主页 v13 全零检出), wake-tap 救不了 —
            # skill 设 no_ui_escape="back" 时按返回键回已知页面, skill 恢复 tick
            # 后自行重试。立绘屏唤醒场景不受影响(daily 系 skill 不声明)。
            _cur_skill = self.current_skill
            # 转场护栏(2026-07-11 实锤×2: 轮播 tap 点对活动页, 落地加载的零框
            # 帧在第 3 个 tick 就被 back 拆掉): ①暗帧=加载转场, 只等不 back
            # (盲区页如運輸船主页是亮的) ②距上次 click/back/swipe <8s = 转场
            # 窗口, 不 escape。
            _recent_act = (time.time()
                           - getattr(self, "_last_act_ts", 0.0)) < 8.0
            # 加载序列护栏(2026-07-11 实锤×7: 点对活动页→加载中→后期零框帧
            # →第3个零框tick被back拆掉): 距最近「加载中」检出<30s = 大概率
            # 仍在加载/进场动画序列, 零框帧只等不动。(活动进场动画实测零框
            # 暗帧持续 18s+, 12s 窗口不够又被 6/30 拆一次; 代价=真盲区页
            # 的逃生推迟到 30s, 可接受 — no_ba 30-tick abort 兜底仍在。)
            _recent_loading = (time.time()
                               - getattr(self, "_last_loading_ts", 0.0)) < 30.0
            if (getattr(_cur_skill, "no_ui_escape", None) == "back"
                    and _bright and not _recent_act and not _recent_loading
                    and self._no_ba_ticks >= 3 and self._no_ba_ticks % 3 == 0):
                esc = {"action": "back",
                       "reason": f"no-UI escape: back to known page "
                                 f"({self._no_ba_ticks}/30)"}
                self._save_trajectory(screenshot_path, screen, None, esc)
                self._last_act_ts = time.time()
                return esc
            if (_bright and not _recent_act and not _recent_loading
                    and self._no_ba_ticks >= 3 and self._no_ba_ticks % 3 == 0):
                # ⚠️位置从 (0.5,0.05) 挪到 (0.35,0.12) — 旧点自以为是"空天区", 实际
                # 压在 topbar 的 AP「+」按钮带上(2026-07-07 假 no-UI 时反复戳开
                # 購買AP框)。(0.35,0.12) = topbar 下方 / 左侧图标列右侧 / 角色左侧,
                # 两代 lobby 皮肤实测都是空背景; 真立绘屏(UI 全隐)点哪都安全。
                # 兼职标题屏点透(2026-07-16, 零 OCR): 开屏 TOUCH TO START 页
                # 同样是「亮帧+双模型零检出」且全屏任意点即进 — 本 tap 即点透,
                # 接替拦截器里已改造的 OCR "(?:TOUCH|TAP).*START" 检测。
                wake = action_click(
                    0.35, 0.12,
                    f"wake 放置立绘屏/点透标题屏 (zero-det bright, "
                    f"{self._no_ba_ticks}/30)")
                self._save_trajectory(screenshot_path, screen, None, wake)
                self._last_act_ts = time.time()
                return wake
            return wait_action

        # First lobby-visit of the run → snapshot all 8 nav-icon badges.
        # Inexpensive (~5ms on a 4K screenshot) and only runs until the
        # first snapshot lands, so it can't impact steady-state perf.
        was_empty = not getattr(self, "_lobby_badges_first_seen", None)
        if was_empty:
            self._maybe_scan_lobby_badges(reason="pipeline start")
        skill = self.current_skill
        if skill is None:
            self._running = False
            return action_done("no more skills")

        # If the badge snapshot JUST populated AND the current skill (we
        # already started, since _start_current_skill ran before any
        # tick) has no dot, bail out NOW — including the very first
        # skill of the run, which couldn't be skipped pre-tick because
        # the lobby hadn't been scanned yet.  User-reported case: cafe
        # icon shows no dot but bot went in anyway (run_20260513_205321).
        if was_empty and getattr(self, "_lobby_badges_first_seen", None):
            skip_reason = self._should_skip_skill_by_badge(skill)
            if skip_reason:
                print(f"[Pipeline] Skipping skill '{skill.name}': {skip_reason} (post-scan)")
                # Mimic the skip path in _start_current_skill but on the
                # already-started skill: drop its bookkeeping and advance.
                self._results.append(
                    SkillResult(
                        skill_name=skill.name,
                        status="skipped",
                        ticks=0,
                        duration_s=max(0.0, time.time() - self._skill_start_time),
                        reason=skip_reason,
                    )
                )
                self._current_idx += 1
                self._retry_count = 0
                if self._current_idx >= len(self._skill_order):
                    self._running = False
                    self._log_lobby_badge_summary()
                    print("[Pipeline] All skills complete")
                    return action_done("all skills complete after skip")
                self._start_current_skill()
                # The next skill is ready; emit a 0.2s wait so the next
                # tick can run its first action cleanly.
                return action_wait(200, f"advanced past {skill.name} (no dot)")

        # ── Global Interceptor (runs before any skill) ──
        intercept = self._global_interceptor(screen, skill)
        if intercept:
            intercept = self._dedup_click(intercept)
            self._save_trajectory(screenshot_path, screen, skill, intercept)
            return intercept

        # Dot-driven skip — MULTI-FRAME VOTE (2026-06-15). Daily-harvest skills
        # (cafe / mail / schedule / club / craft / momo_talk / ...) override
        # should_run to look for their red/yellow dot — no dot = no work = skip.
        # The dot cls FLICKERS frame-to-frame (live: social 红点 detected on some
        # lobby frames, missed on others → single-frame should_run false-skipped
        # a real 未读 day, repeatedly). So we VOTE: should_run True on ANY of the
        # first _DOT_GATE_FRAMES frames = has work (run); only skip after that
        # many CONSECUTIVE no-dot frames. Keeps the dot-gate (user: 通过红黄点进入,
        # 别 always-enter) while being robust to weak-cls flicker. Battle/sweep
        # skills don't override should_run (returns True) → pass on frame 1.
        _DOT_GATE_FRAMES = 4
        if not getattr(self, "_dot_gate_done", False):
            try:
                if _FORCE_ALL_SKILLS:
                    self._dot_gate_done = True
                    print(f"[Pipeline] '{skill.name}' FORCE-RUN (--force-skills, dot gate bypassed)")
                elif skill.should_run(screen):
                    self._dot_gate_done = True  # dot seen this frame → run the skill
                else:
                    g = int(getattr(self, "_dot_gate_ticks", 0)) + 1
                    self._dot_gate_ticks = g
                    if g >= _DOT_GATE_FRAMES:
                        print(f"[Pipeline] '{skill.name}' should_run=False "
                              f"({g}-frame vote, no dot) → skip")
                        self._results.append(
                            SkillResult(
                                skill_name=skill.name,
                                status="skipped",
                                ticks=0,
                                duration_s=0.0,
                                reason="no relevant dot on lobby",
                            )
                        )
                        self._current_idx += 1
                        if self._current_idx >= len(self._skill_order):
                            self._running = False
                            return action_done("all skills complete (last was dot-skipped)")
                        self._start_current_skill()
                        return action_wait(200, f"{skill.name} skipped — moving on")
                    # still voting — wait a frame for a clearer dot read
                    return action_wait(300, f"{skill.name} dot-gate vote ({g}/{_DOT_GATE_FRAMES})")
            except Exception as e:
                print(f"[Pipeline] should_run check failed for {skill.name}: {e}")
                self._dot_gate_done = True

        # ── universal loading gate (user 2026-06-13: 有加载中就暂停process给
        # 游戏加载时间). Pause the skill entirely while the 加载中 spinner is up —
        # don't run skill logic on a transient loading frame.
        try:
            if screen.is_loading():
                self._last_action_reason = "加载中 → 暂停等待"
                return {"action": "wait", "duration_ms": 600,
                        "reason": "加载中 → 暂停等待", "_pipeline": True,
                        "_skill": skill.name, "_tick": skill.ticks}
        except Exception:
            pass

        # Let skill decide
        action = skill.tick(screen)
        action = self._dedup_click(action, screen)
        action_type = action.get("action", "")
        action_reason = str(action.get("reason", "") or "")
        self._last_action_reason = action_reason

        # ── State lockout: detect truly stuck repeated waits ──
        same_wait = (
            action_type == "wait"
            and skill.sub_state == self._last_sub_state
            and action_reason == self._last_wait_reason
        )
        if same_wait:
            self._stuck_counter += 1
        else:
            self._stuck_counter = 0
        self._last_sub_state = skill.sub_state
        self._last_wait_reason = action_reason if action_type == "wait" else ""

        # If the exact same wait reason repeats for 20+ ticks, burst ESC to break out.
        # CRITICAL: Do NOT send ESC when on lobby — ESC on lobby opens
        # the "是否結束？" exit dialog which can cause cascading failures.
        # Also do NOT ESC during active battles — they legitimately repeat
        # the same wait reason for many ticks while combat is in progress.
        _battle_wait_keywords = (
            "battle in progress", "battle speed", "loading/battle",
            # event phase scanning: legitimate waits between popup/reward
            # cycles, ESC here drops us out of the event page to lobby
            "event quest scanning", "event story scanning", "event challenge scanning",
            "no 入場 anywhere, waiting", "no 入場 visible",
            "ocr flicker",
            # bonus-setup battle: formation/quick-edit FSM + battle wait
            # — ESC mid-battle backs us out of the active fight
            "bonus-setup", "bonus-team", "quick-edit",
        )
        _is_battle_wait = any(kw in action_reason.lower() for kw in _battle_wait_keywords)
        # NEVER ESC-burst when a popup is currently on screen.  ESC on
        # the exit-prompt popup ("是否結束?") confirms exit → lobby,
        # which is exactly the "经常点进一个地方然后就喜欢退回主界面"
        # the user complained about.  Popups have their own handler
        # (see _handle_common_popups).
        # Pure-YOLO popup detection (OCR is disabled). A "backout-able" modal
        # = a 取消/X button is detected. We must NEVER ESC such a popup: ESC on
        # the exit prompt ("是否結束?") CONFIRMS exit → drops to lobby (the
        # "点进一个地方就退回主界面" bug). 确认键 alone is NOT treated as a
        # popup here — a stuck confirm-only screen should still ESC-recover.
        _cancel_btn = find_yolo_box(screen, ["取消键"], min_conf=0.40)
        _x_btn = find_yolo_box(screen, ["弹窗叉叉"], min_conf=0.40)
        _popup_on_screen = bool(_cancel_btn or _x_btn)
        # ── EARLIER escalation: blind-TAP to dismiss full-screen overlays ──
        # A repeated wait often means an undismissed "TOUCH TO CONTINUE" /
        # 獲得獎勵 reward overlay (e.g. arena ranking reward) or an account
        # level-up — full-screen prompts that carry NO reliable YOLO cls, so
        # neither the interceptor's GOT_REWARD handler nor the no-BA blind-tap
        # (a few background boxes keep has_ba_ui True) fires. A blind TAP
        # dismisses these WITHOUT backing out of the screen (unlike the ESC
        # burst below). Fire at 9/12/15/18 stuck ticks, alternating
        # center / corners so we hit the prompt, not a reward card. Skip
        # during battles (legit long waits with changing/known reasons).
        if (action_type == "wait" and not _is_battle_wait and not _popup_on_screen
                and not _BRINGUP_EXPOSE
                and self._stuck_counter >= 9 and self._stuck_counter % 3 == 0):
            self._blind_tap_count = getattr(self, "_blind_tap_count", 0) + 1
            # Target real dismiss zones first (TOUCH-TO-CONTINUE / X), then
            # corners — center hits reward cards that absorb the tap.
            _tap_pts = [(0.5, 0.88), (0.88, 0.15), (0.08, 0.92), (0.92, 0.92)]
            px, py = _tap_pts[self._blind_tap_count % len(_tap_pts)]
            print(f"[Pipeline] Skill '{skill.name}' stuck {self._stuck_counter} "
                  f"ticks on '{action_reason}', blind-tap dismiss overlay "
                  f"#{self._blind_tap_count} at ({px},{py})")
            action = action_click(px, py,
                f"blind-tap dismiss overlay (stuck {self._stuck_counter})")
            action_type = "click"
            action_reason = str(action.get("reason", "") or "")

        if action_type == "wait" and self._stuck_counter > 0 and self._stuck_counter % 20 == 0:
            if screen.is_lobby():
                print(f"[Pipeline] Skill '{skill.name}' repeating wait on lobby for {self._stuck_counter} ticks, skipping ESC (unsafe on lobby)")
            elif _is_battle_wait:
                print(f"[Pipeline] Skill '{skill.name}' battle in progress for {self._stuck_counter} ticks, skipping ESC (active battle)")
            elif _popup_on_screen:
                # Backout-able modal stuck (exit prompt / friend-cafe-visit /
                # unhandled dialog). Dismiss via 取消/X — SAFE. Never ESC: ESC
                # can CONFIRM the exit dialog → quits to lobby.
                _btn = _cancel_btn or _x_btn
                _which = "取消键" if _cancel_btn else "弹窗叉叉"
                action = action_click_box(
                    _btn, f"stuck {self._stuck_counter}: dismiss popup ({_which}, not ESC)")
                action_type = action.get("action", "")
                action_reason = str(action.get("reason", "") or "")
                print(f"[Pipeline] Skill '{skill.name}' stuck {self._stuck_counter} "
                      f"ticks on popup, clicking {_which} (safe dismiss, not ESC)")
            elif _BRINGUP_EXPOSE:
                # BRING-UP: no ESC-burst fallback. Freeze in place + log loudly
                # so the exact stuck tick + its trajectory screenshot can be
                # inspected — being stuck here MEANS this stage has a YOLO
                # navigation/detection hole to fix (not paper over).
                print(
                    f"[Pipeline] *** BRINGUP FREEZE *** Skill '{skill.name}' STUCK "
                    f"{self._stuck_counter} ticks  sub_state='{skill.sub_state}'  "
                    f"waiting='{action_reason}'  tick={skill.ticks}  frame={screenshot_path}"
                )
                # action stays the original wait → bot freezes here for inspection.
            else:
                print(
                    f"[Pipeline] Skill '{skill.name}' repeating wait '{action_reason}' "
                    f"in '{skill.sub_state}' for {self._stuck_counter} ticks, ESC burst"
                )
                action = action_back(f"ESC burst: stuck in {skill.sub_state}")
                action_type = action.get("action", "")
                action_reason = str(action.get("reason", "") or "")

        # Save trajectory
        self._save_trajectory(screenshot_path, screen, skill, action)

        # Skill finished
        if action_type == "done":
            if "timeout" in action_reason.lower():
                if self._retry_count < self._max_retries:
                    self._retry_count += 1
                    print(f"[Pipeline] Skill '{skill.name}' reported timeout, retry {self._retry_count}")
                    skill.reset()
                    return action_wait(500, f"skill '{skill.name}' retry")
                print(f"[Pipeline] Skill '{skill.name}' reported timeout, skipping")
                self._advance_skill("timeout", action_reason)
                if action_reason:
                    return action_wait(300, f"skill '{skill.name}' timed out, skipping ({action_reason})")
                return action_wait(300, f"skill '{skill.name}' timed out, skipping")
            self._advance_skill("done", action_reason)
            if action_reason:
                return action_wait(300, f"skill '{skill.name}' done ({action_reason}), advancing")
            return action_wait(300, f"skill '{skill.name}' done, advancing")

        # Timeout check
        if skill.ticks >= skill.max_ticks:
            if self._retry_count < self._max_retries:
                self._retry_count += 1
                print(f"[Pipeline] Skill '{skill.name}' timeout, retry {self._retry_count}")
                skill.reset()
                return action_wait(500, f"skill '{skill.name}' retry")
            print(f"[Pipeline] Skill '{skill.name}' timeout, skipping")
            self._advance_skill("timeout", "max ticks exceeded")
            return action_wait(300, f"skill '{skill.name}' timed out, skipping")

        # Tag action with pipeline metadata
        action["_pipeline"] = True
        action["_skill"] = skill.name
        action["_tick"] = self._total_ticks
        if action_type in ("click", "back", "swipe", "swipe_tap"):
            # no-UI escape 的转场护栏用: 最近有动作 = 可能在转场窗口内
            self._last_act_ts = time.time()
        return action

    def _save_trajectory(self, screenshot_path: str, screen: ScreenState,
                         skill: Optional[BaseSkill], action: Dict[str, Any]) -> None:
        """Enqueue frame + OCR + action for async write to trajectory dir.

        Returns immediately — actual disk I/O happens on a background thread.
        Prevents ~10-50ms/tick blocking when writing to slow disks.

        ``skill`` may be None for pre-skill no-BA-UI wait frames so the
        debug trace shows WHY the pipeline couldn't reach the skill loop.
        """
        if self._traj_dir is None:
            return
        tick_id = f"tick_{self._total_ticks:04d}"
        ocr_data = [
            {
                "text": b.text,
                "conf": round(b.confidence, 3),
                "x1": round(b.x1, 4), "y1": round(b.y1, 4),
                "x2": round(b.x2, 4), "y2": round(b.y2, 4),
            }
            for b in screen.ocr_boxes
        ]
        yolo_data = [
            {
                "cls": b.cls_name,
                "conf": round(b.confidence, 3),
                "x1": round(b.x1, 4), "y1": round(b.y1, 4),
                "x2": round(b.x2, 4), "y2": round(b.y2, 4),
            }
            for b in screen.yolo_boxes
        ]
        record = {
            "tick": self._total_ticks,
            "skill": skill.name if skill else "(no-ba-ui-wait)",
            "sub_state": skill.sub_state if skill else "",
            "skill_ticks": skill.ticks if skill else 0,
            "action": action,
            "ocr_boxes": ocr_data,
            "ocr_count": len(ocr_data),
            "yolo_boxes": yolo_data,
            "yolo_count": len(yolo_data),
            "yolo_status": _yolo_status,
            "image_w": screen.image_w,
            "image_h": screen.image_h,
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        job = (screenshot_path, str(self._traj_dir), tick_id, record)
        try:
            self._traj_writer_queue.put_nowait(job)
        except queue.Full:
            # Writer is overwhelmed — drop the oldest job to keep agent fluid.
            try:
                self._traj_writer_queue.get_nowait()
                self._traj_writer_queue.put_nowait(job)
            except (queue.Empty, queue.Full):
                pass

    def _traj_writer_loop(self) -> None:
        """Background worker that consumes trajectory save jobs."""
        while True:
            try:
                job = self._traj_writer_queue.get()
            except Exception:
                continue
            if job is None:
                break
            try:
                src_path, traj_dir_str, tick_id, record = job
                traj_dir = Path(traj_dir_str)
                src = Path(src_path)
                if src.exists():
                    dst = traj_dir / f"{tick_id}.jpg"
                    # Plain byte copy is faster than copy2 and good enough.
                    try:
                        shutil.copyfile(str(src), str(dst))
                    except Exception:
                        pass
                json_path = traj_dir / f"{tick_id}.json"
                # Compact JSON: separators without spaces reduces size ~35%
                # and speeds up serialize.
                json_path.write_text(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":")),
                    encoding="utf-8",
                )
            except Exception as e:
                print(f"[Pipeline] trajectory save error: {e}")

    def get_summary(self) -> str:
        """Get human-readable summary of pipeline execution."""
        lines = [f"Pipeline: {'Running' if self._running else 'Stopped'}"]
        lines.append(f"Total ticks: {self._total_ticks}")
        for r in self._results:
            if r.reason:
                lines.append(f"  {r.skill_name}: {r.status} ({r.ticks} ticks, {r.duration_s:.1f}s) - {r.reason}")
            else:
                lines.append(f"  {r.skill_name}: {r.status} ({r.ticks} ticks, {r.duration_s:.1f}s)")
        skill = self.current_skill
        if skill:
            lines.append(f"  {skill.name}: in progress ({skill.ticks} ticks, sub={skill.sub_state})")
        return "\n".join(lines)
