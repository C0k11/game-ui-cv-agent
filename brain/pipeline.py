"""DailyPipeline: Orchestrates skill-based daily routine automation.

Sequences skills in order, handles timeouts, retries, and recovery.
Uses OCR as the primary screen-reading method (portable across resolutions).

Usage:
    from brain.pipeline import DailyPipeline
    pipe = DailyPipeline()
    pipe.start()
    # Each tick: pipe.tick(screenshot_path) -> action dict
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
from typing import Any, Dict, List, Optional, Tuple

from brain.skills.base import (
    BaseSkill, OcrBox, YoloBox, ScreenState,
    action_click, action_click_box, action_back, action_wait, action_done,
)
from brain.skills.cafe import CafeSkill
from brain.skills.schedule import ScheduleSkill
from brain.skills.club import ClubSkill
from brain.skills.bounty import BountySkill
from brain.skills.jfd import JointFiringDrillSkill
from brain.skills.mail import MailSkill
from brain.skills.arena import ArenaSkill
# event_farming + event_shop merged into event_activity (刷活动).
# The standalone skill classes still exist but aren't registered.
from brain.skills.daily_tasks import DailyTasksSkill
from brain.skills.shop import ShopSkill
from brain.skills.craft import CraftSkill
from brain.skills.buy_pyroxene import BuyPyroxeneSkill
from brain.skills.daily_mission import DailyMissionSkill
from brain.skills.momo_talk import MomoTalkSkill
from brain.skills.event_activity import EventActivitySkill
from brain.skills.pass_reward import PassRewardSkill
from brain.skills.story_mining import StoryMiningSkill
from brain.skills.ap_planning import ApPlanningSkill
from brain.skills.campaign_sweep import CampaignSweepSkill
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
    """Resolve an active model's weights path via registry, fall back to literal."""
    reg = _load_model_registry()
    section = reg.get(model_key)
    if not section:
        return fallback
    active = section.get("active")
    versions = section.get("versions", {})
    info = versions.get(active, {})
    p = info.get("path")
    if p and Path(p).is_file():
        return Path(p)
    return fallback


_YOLO_BATTLE_HEADS = Path(r"D:\Project\ml_cache\models\yolo\battle_heads.pt")
_YOLO_EMOTICON_V26 = Path(r"D:\Project\ml_cache\models\yolo\runs\emoticon_yolo26n\weights\best.pt")
_YOLO_EMOTICON_V8_LEGACY = Path(r"D:\Project\ml_cache\models\yolo\emoticon.pt")
_YOLO_EMOTICON = _YOLO_EMOTICON_V26 if _YOLO_EMOTICON_V26.exists() else _YOLO_EMOTICON_V8_LEGACY

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

# Per-skill YOLO detector loadout (module-level = single source of truth).
# base = "ui" ONLY (FPS: avatar=fused yolo26X / battle are heavy nets, added
# only where a skill needs them). _start_skill sets context from this at skill
# start; CampaignSweep imports it (lazily, to avoid an import cycle) to set the
# right context for each DELEGATED sub-skill — without this, arena run via the
# sweep gets no avatar boxes and selects 0 opponents (H2).
BASE_DETECTORS = "ui"
SKILL_YOLO_MAP = {
    # Cafe needs avatar too: the invite list identifies each row's student via
    # the fused_avatar head model (model_tag=="avatar", 中文角色名) so it can
    # invite the configured cafe_invite_targets. +cafe = emoticon headpat marks.
    "Cafe": f"{BASE_DETECTORS}+cafe+avatar",    # +emoticon bubbles +student heads
    "Bounty": f"{BASE_DETECTORS}+battle",      # +battle_heads
    # Arena selects opponents via cls92 (ARENA_OPPONENT_ROW) in the UI model —
    # no avatar model needed (dropped 2026-05-31, v5 added cls92). +battle for
    # the in-fight skip/heads.
    "Arena": f"{BASE_DETECTORS}+battle",
    # Schedule needs avatar to identify which student sits in each room / 全体
    # 课程表 list (fused_avatar 中文角色名) so it can place the dashboard-chosen
    # targets. NO emoticon — headpat is cafe-only. (probe-derived 2026-06-01)
    "Schedule": f"{BASE_DETECTORS}+avatar",
    "EventFarming": f"{BASE_DETECTORS}+battle",
    "TotalAssault": f"{BASE_DETECTORS}+battle",
    "JointFiringDrill": f"{BASE_DETECTORS}+battle",
    "CampaignPush": f"{BASE_DETECTORS}+battle",
    "EventActivity": f"{BASE_DETECTORS}+battle",
    # DailyRoutine wraps cafe (emoticon headpat) + would need battle for event
    # quests (event currently disabled). No avatar — student-id isn't needed.
    "DailyRoutine": f"{BASE_DETECTORS}+cafe+battle",
}

_yolo_load_attempts = 0
_MAX_YOLO_LOAD_ATTEMPTS = 3
_yolo_status = "not_attempted"
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
        if _YOLO_BATTLE_HEADS.is_file():
            candidates.append((_YOLO_BATTLE_HEADS, 0.45, "battle"))
        if _YOLO_EMOTICON.is_file():
            candidates.append((_YOLO_EMOTICON, 0.15, "cafe"))
        # Fused avatar (251 BA student heads).  conf 0.35 = balanced
        # precision/recall on manual val.  Tagged "avatar" — opt-in per skill.
        if _YOLO_FUSED_AVATAR_V4.is_file():
            candidates.append((_YOLO_FUSED_AVATAR_V4, 0.35, "avatar"))
        # UI v1 (~145 static UI classes — buttons, dots, banners, etc).
        # conf 0.30 = lower threshold since UI bbox quality is high but some
        # minority classes need slack.  Tagged "ui" — most skills will need this.
        if _YOLO_UI_V1.is_file():
            candidates.append((_YOLO_UI_V1, 0.30, "ui"))
        if not candidates:
            _yolo_status = "model_not_found"
            print(f"[Pipeline] YOLO model NOT found")
            return None
        from ultralytics import YOLO
        import numpy as np
        loaded_names = []
        for model_path, model_conf, model_tag in candidates:
            try:
                m = YOLO(str(model_path))
                m(np.zeros((64, 64, 3), dtype=np.uint8), verbose=False)
                _yolo_models.append((m, model_conf, model_tag))
                loaded_names.append(f"{model_path.stem}({len(m.names)}cls)")
                print(f"[Pipeline] YOLO loaded from {model_path} (conf={model_conf}, tag={model_tag})")
            except Exception as e:
                print(f"[Pipeline] YOLO load failed for {model_path}: {e}")
                continue
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
_BRINGUP_EXPOSE = True

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
        # concat all recognized text on the strip, keep only digits + / and ,
        raw = "".join(line[1] for line in result)
        kept = _re.sub(r"[^0-9/]", "", raw.replace(",", "").replace("，", ""))
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


def _find_florence_hit(screen: ScreenState, queries: List[str], *, region: Optional[Tuple[float, float, float, float]] = None) -> Optional[OcrBox]:
    if not screen.screenshot_path:
        return None
    try:
        import cv2
        import numpy as np
        from vision.florence_vision import get_florence_vision

        img = cv2.imdecode(np.fromfile(screen.screenshot_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        rx1, ry1, rx2, ry2 = region or (0.0, 0.0, 1.0, 1.0)
        x1 = max(0, int(rx1 * w))
        y1 = max(0, int(ry1 * h))
        x2 = min(w, int(rx2 * w))
        y2 = min(h, int(ry2 * h))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = img[y1:y2, x1:x2]
        hits = get_florence_vision().detect_open_vocabulary(crop, queries)
        boxes: List[OcrBox] = []
        for hit in hits:
            bbox = hit.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            bx1, by1, bx2, by2 = [float(v) for v in bbox]
            nx1 = min(max((x1 + bx1) / w, 0.0), 1.0)
            ny1 = min(max((y1 + by1) / h, 0.0), 1.0)
            nx2 = min(max((x1 + bx2) / w, 0.0), 1.0)
            ny2 = min(max((y1 + by2) / h, 0.0), 1.0)
            if nx2 <= nx1 or ny2 <= ny1:
                continue
            boxes.append(OcrBox(text=str(hit.get("query") or hit.get("label") or queries[0]), confidence=float(hit.get("score") or 0.5), x1=nx1, y1=ny1, x2=nx2, y2=ny2))
        if not boxes:
            return None
        screen.add_florence_boxes(boxes)
        return max(boxes, key=lambda b: (b.confidence, b.w * b.h))
    except Exception:
        return None


def _run_template_matching(frame_bgr) -> List:
    """Run headpat bubble detection via happy_face templates. Returns list of TemplateHitBox."""
    from brain.skills.base import TemplateHitBox
    hits = []
    try:
        from vision.template_matcher import find_headpat_bubbles
        # Restrict to cafe play area (exclude UI bars + left sidebar icons)
        raw = find_headpat_bubbles(frame_bgr, threshold=0.75,
                                   region=(0.12, 0.25, 0.98, 0.80))
        for h in raw:
            hits.append(TemplateHitBox(
                label=h.label,
                confidence=h.confidence,
                x1=h.x1, y1=h.y1,
                x2=h.x2, y2=h.y2,
            ))
    except Exception as e:
        if not getattr(_run_template_matching, '_warned', False):
            print(f"[Pipeline] Template matching error: {e}")
            _run_template_matching._warned = True
    return hits


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
    if skip_ocr and prev_ocr_boxes is not None:
        ocr_boxes = prev_ocr_boxes
    else:
        ocr_boxes = _run_ocr_on_image(frame_bgr, w, h)
    if injected_yolo_boxes is not None:
        yolo_boxes = injected_yolo_boxes
    else:
        yolo_boxes = _run_yolo_on_image(frame_bgr, w, h)
    template_hits = _run_template_matching(frame_bgr)
    return ScreenState(
        ocr_boxes=ocr_boxes,
        yolo_boxes=yolo_boxes,
        template_hits=template_hits,
        image_w=w,
        image_h=h,
        screenshot_path=screenshot_path,
        frame=frame_bgr,   # kept for on-demand digit-OCR cropping
    )


def read_screen(screenshot_path: str) -> ScreenState:
    """Run OCR on a screenshot and return a ScreenState.

    This is the core perception function. It reads the screen using
    RapidOCR and returns structured data about what text is visible.
    All coordinates are normalized 0-1 for portability.
    """
    import cv2
    import numpy as np

    img = cv2.imdecode(np.fromfile(screenshot_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return ScreenState(screenshot_path=screenshot_path)

    h, w = img.shape[:2]
    return ScreenState(
        ocr_boxes=_run_ocr_on_image(img, w, h),
        yolo_boxes=_run_yolo_on_image(img, w, h),
        image_w=w,
        image_h=h,
        screenshot_path=screenshot_path,
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

    # Default skill sequence for daily routine.
    #
    # Phase 0: AP overflow protection — if AP≥900, farm first or cafe earnings block.
    # Phase 1: Daily skills (cafe, schedule, club, bounty, arena).
    # Phase 2: Event-aware farming — detect 活動進行中 double-drop events,
    #          prioritize credit stages, scroll to bottom for highest stage.
    # Phase 3: Mail + tasks → collect AP rewards.
    # Phase 4: 回马枪 (boomerang) — second farming pass with collected AP.
    @staticmethod
    def _make_event_activity(opts):
        skill = EventActivitySkill()
        # Profile-driven toggles for EventActivity.
        skill._enable_bonus_team = bool(opts.get("enable_bonus_team", False))
        # Farming auto-setup battle (one quick-edit battle to populate the
        # saved sweep team with rate-up students).  Defaults False per
        # user — they don't want a surprise second battle after quest
        # finishes.  Enable via profile option `enable_bonus_setup_battle`.
        # Default True (per skill's own __init__ default).  Was False
        # here for a while, which silently disabled the FSM the user
        # explicitly wants ("活动还是没有把加成打满").  Profile may
        # still explicitly disable by setting the option to False.
        skill._enable_bonus_setup_battle = bool(opts.get("enable_bonus_setup_battle", True))
        # Farming + shop settings (formerly separate skills, now merged in)
        try:
            skill._preferred_stage = max(0, int(opts.get("event_farming_stage") or 0))
        except Exception:
            skill._preferred_stage = 0
        try:
            skill._farming_ap_budget = max(0, int(opts.get("event_farming_ap_budget") or 0))
        except Exception:
            skill._farming_ap_budget = 0
        try:
            skill._min_ap_for_sweep = max(1, int(opts.get("event_min_ap_for_sweep") or 20))
        except Exception:
            skill._min_ap_for_sweep = 20
        try:
            skill._farm_max_rounds = max(0, int(opts.get("event_max_rounds") or 0))
        except Exception:
            skill._farm_max_rounds = 0
        skill._shop_auto_buy = bool(opts.get("event_shop_auto_buy", False))
        skill._shop_spend_currencies = tuple(opts.get("event_shop_currencies") or ())
        skill._shop_furniture_first = bool(opts.get("event_shop_furniture_first", False))
        return skill

    DEFAULT_SKILLS = [
        # ── Battle / sweep skills (explicit user control) ──
        "campaign_sweep",   # 1.  One-tap sweep of tickets (bounty/arena/event)
        "bounty",           # 2.  Sweep bounty tickets (if not covered by sweep)
        "jfd",              # 2b. 学院交流会 ticket sweep (tickets + AP)
        "arena",            # 3.  PvP fights + claim rewards
        # ── Daily harvest (all bundled) ──
        # Cycles through mail / event_activity / cafe / schedule / club /
        # daily_tasks / craft / pass_reward / momo_talk / story_mining /
        # shop / ap_planning. Each sub uses its own dot check (craft skips
        # the dot check per user spec — always enters).
        "daily_routine",    # 4.  All daily harvest in one go (dot-driven)
    ]
    # 回马枪 (post-loop second event sweep): handled inline at pipeline level
    # after all skills finish — see _maybe_huimaqiang().

    TRAJECTORIES_DIR = Path(__file__).resolve().parents[1] / "data" / "trajectories"

    def __init__(self, skill_names: Optional[List[str]] = None,
                 profile_options: Optional[Dict[str, Any]] = None):
        opts = dict(profile_options or {})
        try:
            ap_purchase_limit = max(0, int(opts.get("ap_purchase_limit") or 0))
        except Exception:
            ap_purchase_limit = 0
        forbid_premium_currency = bool(opts.get("forbid_premium_currency", True))
        # Event-farming budget (adapted from reference activity_sweep_times).
        # max_rounds: how many sweep cycles per run (1 = legacy, >1 = loop).
        # ap_reserve: don't loop when current AP ≤ this floor.
        try:
            event_max_rounds = max(1, int(opts.get("event_max_rounds") or 1))
        except Exception:
            event_max_rounds = 1
        try:
            event_ap_reserve = max(0, int(opts.get("event_ap_reserve") or 0))
        except Exception:
            event_ap_reserve = 0

        self._skill_registry: Dict[str, BaseSkill] = {
            "buy_pyroxene": BuyPyroxeneSkill(),
            "cafe": CafeSkill(),
            "schedule": ScheduleSkill(),
            "club": ClubSkill(),
            "shop": ShopSkill(),
            "craft": CraftSkill(),
            "momo_talk": MomoTalkSkill(),
            "story_mining": StoryMiningSkill(),
            "event_activity": self._make_event_activity(opts),
            "bounty": BountySkill(),
            "jfd": JointFiringDrillSkill(),
            "arena": ArenaSkill(),
            "ap_planning": ApPlanningSkill(
                forbid_premium_currency=forbid_premium_currency,
                paid_purchase_limit=ap_purchase_limit,
            ),
            "mail": MailSkill(),
            "daily_tasks": DailyTasksSkill(),
            "daily_mission": DailyMissionSkill(),   # rewrite of daily_tasks (probe-driven)
            "pass_reward": PassRewardSkill(),
            "campaign_sweep": CampaignSweepSkill(),
            # Single dispatcher for all daily-harvest sub-flows
            # (mail / cafe / schedule / club / daily_tasks / event_activity
            #  / craft / pass_reward / momo_talk / story_mining / shop /
            #  ap_planning). Battle skills (bounty / arena / campaign_sweep)
            # stay as separate skill_order entries.
            "daily_routine": DailyRoutineSkill(),
        }
        # CampaignSweep needs a reference to the registry so it can
        # delegate to Bounty / Arena / EventActivity without creating
        # an import cycle.  Wire after the dict is built.
        self._skill_registry["campaign_sweep"].set_registry(self._skill_registry)

        names = skill_names or self.DEFAULT_SKILLS
        self._skill_order: List[str] = [n for n in names if n in self._skill_registry]
        self._current_idx: int = 0
        self._running: bool = False
        self._results: List[SkillResult] = []
        self._skill_start_time: float = 0.0
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
        "PassReward": "recruit",
        "Mail":       "mail",
        "DailyTasks": "daily_tasks_nav",
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
        # Set YOLO context based on skill — only run relevant model(s).
        # FPS FIX (2026-05-29): base is now "ui" ONLY. The avatar model is
        # fused_avatar yolo26X (the heaviest net) — running it every tick just
        # to click buttons was the main inference cost. Nothing on the daily
        # nav/sweep path needs per-student identification, so avatar is dropped
        # from the base and only added back where a skill genuinely needs to
        # know WHICH student (none in the current run path). Add 'avatar' to a
        # skill's context here if/when student-id is required.
        # Loadout map lives at module level (SKILL_YOLO_MAP) so CampaignSweep
        # can reuse it for its delegated sub-skills. See note there.
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

        Handles "rude" popups that can appear at any time regardless of skill:
        P0: Disconnect / reconnect / download data
        P1: Stale sign-in / activity popups that leaked through
        P2: Account / student Level Up full-screen effects
        """
        # ════════════════════════════════════════════════════════════════
        # PURE-YOLO prelude (2026-05-29). Handle the safe, unambiguous
        # global popups via YOLO cls. EVERYTHING below this block is OCR and
        # therefore DEAD while pipeline._OCR_ENABLED is False — kept for the
        # digit-OCR re-enable phase. Only cls-backed popups are auto-handled
        # here; ambiguous confirm/cancel dialogs are left to the owning skill
        # (it knows the context) so we don't blind-confirm a 'visit friend
        # cafe' / 'exit game' prompt.
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

        # ── P-1: Title screen ("TOUCH TO START") ──
        # Game may restart / disconnect and land on title. All skills need this.
        tap_start = screen.find_text_one("(?:TOUCH|TAP).*START", min_conf=0.8)
        if tap_start:
            print(f"[Interceptor] Title screen detected, tapping to start")
            return action_click(0.5, 0.85, "interceptor: tap to start")

        # ── P-1: Download / update confirmation modal ──
        # BA first-run and patch-day show a "通知" modal:
        #   "下載必要內容 XXX MB"  [取消] [確認]
        # Residual "下載檔案驗證完" text from the background loading screen
        # still renders at the bottom and would trick is_loading() into
        # returning True, stalling the pipeline forever. Handle the modal
        # explicitly first so the confirm click always goes through.
        update_notice = screen.find_text_one(
            "通知", region=(0.30, 0.12, 0.70, 0.32), min_conf=0.55
        )
        if update_notice:
            update_body = screen.find_any_text(
                ["下載必要", "下载必要", "下載遊戲", "下载游戏",
                 "需要下載", "需要下载", "遊戲所需", "游戏所需",
                 "更新資源", "更新资源", "下載資源", "下载资源",
                 "下載內容", "下载内容"],
                region=(0.25, 0.35, 0.75, 0.65), min_conf=0.45,
            )
            if update_body:
                confirm_btn = screen.find_any_text(
                    ["確認", "确认", "確定", "确定", "確", "确", "OK"],
                    region=(0.45, 0.60, 0.75, 0.80),
                    min_conf=0.40,
                )
                if confirm_btn:
                    print(f"[Interceptor] update/download notice '{update_body.text}', clicking confirm")
                    return action_click_box(confirm_btn, "interceptor: confirm download notice")
                # OCR missed the button — use the notification handler's
                # hardcoded confirm position (right half of the modal).
                print(f"[Interceptor] update/download notice '{update_body.text}', fallback click")
                return action_click(0.598, 0.701, "interceptor: confirm download notice (fallback)")

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
            return action_wait(1500, "interceptor: game loading / updating")

        # ── P0: Disconnect / reconnect ──
        disconnect = screen.find_any_text(
            ["网络连接失败", "網絡連接失敗", "返回标题画面", "返回標題畫面",
             "下载数据", "下載數據", "网络错误", "網絡錯誤",
             "連線中斷", "连线中断", "重新連接", "重新连接"],
            min_conf=0.6
        )
        if disconnect:
            confirm = screen.find_any_text(
                ["確認", "确认", "確", "确", "OK", "重試", "重试", "Retry"],
                min_conf=0.6
            )
            if confirm:
                print(f"[Interceptor] P0 disconnect: '{disconnect.text}', clicking confirm")
                self._interceptor_streak += 1
                # If we've been hitting disconnect for 10+ ticks, something is very wrong
                if self._interceptor_streak > 10:
                    print("[Interceptor] P0 disconnect persists >10 ticks, resetting to lobby")
                    self._current_idx = 0
                    self._start_current_skill()
                    self._interceptor_streak = 0
                return action_click_box(confirm, f"interceptor: confirm disconnect ({disconnect.text})")
            return action_click(0.5, 0.7, f"interceptor: dismiss disconnect ({disconnect.text})")

        # ── P0: Exit dialog ("是否結束？") — triggered by accidental ESC on lobby ──
        exit_dialog = screen.find_any_text(
            ["是否結束", "是否结束"],
            region=screen.CENTER, min_conf=0.6
        )
        if exit_dialog:
            cancel = screen.find_any_text(
                ["取消"],
                region=screen.CENTER, min_conf=0.6
            )
            if cancel:
                print(f"[Interceptor] P0 exit dialog detected, clicking 取消")
                return action_click_box(cancel, "interceptor: cancel exit dialog")
            # Fallback: press ESC to dismiss (ESC = cancel in this dialog)
            return action_back("interceptor: dismiss exit dialog")

        # ── P0.5: Settings popup (選項) ──
        # Accidentally opened settings. Has X at top-right. Detect "選項"/"选项" header.
        # Must check BEFORE check-in calendar (彩奈 appears in Mail "From彩奈" too).
        settings_popup = screen.find_any_text(
            ["選項", "选项"],
            region=(0.30, 0.05, 0.70, 0.18), min_conf=0.6
        )
        if settings_popup:
            print(f"[Interceptor] P0.5 Settings popup: '{settings_popup.text}', clicking X")
            self._interceptor_streak += 1
            return action_click(0.83, 0.09, f"interceptor: close settings popup")

        # ── P0.5: Updates / Patch Notes WebView ──
        updates = screen.find_any_text(
            ["Updates", "Patch Notes"],
            region=(0.30, 0.04, 0.70, 0.14), min_conf=0.5
        )
        if updates:
            print(f"[Interceptor] P0.5 Updates WebView: '{updates.text}', BACK to close")
            self._interceptor_streak += 1
            return action_back(f"interceptor: close Updates WebView ({updates.text})")

        # ── P0.5: Daily check-in calendar (彩奈签到簿) ──
        # Full-screen popup with NO close button. Just click anywhere to dismiss.
        # IMPORTANT: "彩奈" restricted to TITLE area (top-center) to avoid matching
        # Mail "From彩奈" text. Use "簽到/到薄/到簿" or "第N天" grid as primary.
        #
        # CAVEAT: "新上任指南任務" (guide mission panel) also has "第N天" grids
        # but is NOT a check-in calendar — center-click doesn't dismiss it.
        # OCR often garbles the header ("加班指南任務", "咖班指南任務") so we
        # detect structurally: "第N天" grid + "立即前往" or "全部领取" buttons.
        #
        # Guard: skip guide mission / check-in if a notification popup (通知) is
        # in front — the skill's _handle_common_popups will handle it instead.
        _has_notification_popup = screen.find_text_one(
            "通知", region=(0.30, 0.10, 0.70, 0.30), min_conf=0.55
        )
        if _has_notification_popup:
            # Let skill popup handler deal with it; don't touch guide panel behind.
            return None

        # IMPORTANT: Lobby event widgets (e.g. Serenade Promenade's top-right
        # cycler) stamp "指南任務" on the character portrait at x≈0.90, y≈0.31.
        # A region-less match fired the interceptor on a pristine lobby,
        # causing it to click (0.98, 0.03) — which is the event widget's
        # fullscreen-expand button, NOT the tutorial panel's home icon. That
        # opened a fullscreen event splash and trapped the event-entry flow.
        # Constrain the header text search to the top-center band where the
        # actual tutorial panel's title lives.
        guide_mission = screen.find_any_text(
            ["指南任務", "指南任务", "新上任指南"],
            region=(0.20, 0.00, 0.80, 0.22),
            min_conf=0.5
        )
        if not guide_mission:
            # Structural fallback: "立即前往" + "第1天" together = guide mission
            has_goto = screen.find_any_text(
                ["立即前往"],
                region=(0.30, 0.60, 0.95, 0.80), min_conf=0.7
            )
            has_day_grid = screen.find_any_text(
                ["第1天", "第2天", "第3天"],
                region=(0.25, 0.10, 0.80, 0.35), min_conf=0.6
            )
            if has_goto and has_day_grid:
                guide_mission = has_goto  # use as trigger
        if guide_mission:
            # Panel has ← back arrow (top-left) and 🏠 home icon (top-right).
            # ← goes to previous screen (may be Account Info, not lobby).
            # 🏠 goes directly to lobby — safer for all callers.
            print(f"[Interceptor] P0.5 guide mission panel: '{guide_mission.text}', clicking home icon")
            self._interceptor_streak += 1
            return action_click(0.98, 0.03, f"interceptor: close guide mission panel (home)")

        checkin = screen.find_any_text(
            ["簽到", "签到", "到薄", "到簿"],
            min_conf=0.5
        )
        if not checkin:
            # "彩奈" only in title area (top-center), NOT in mail list
            checkin = screen.find_any_text(
                ["彩奈"],
                region=(0.35, 0.05, 0.85, 0.25), min_conf=0.6
            )
        if not checkin:
            checkin = screen.find_any_text(
                ["第1天", "第2天", "第3天"],
                region=(0.25, 0.10, 0.80, 0.35), min_conf=0.6
            )
        if checkin:
            print(f"[Interceptor] P0.5 daily check-in calendar: '{checkin.text}', clicking to dismiss")
            self._interceptor_streak += 1
            return action_click(0.5, 0.5, f"interceptor: dismiss check-in calendar ({checkin.text})")

        # ── P1: In-game announcement popup (内嵌公告) ──
        # Detected by "主要消息" text in lower-left area. X button at top-right (0.98, 0.04).
        announcement = screen.find_any_text(
            ["主要消息", "主要消息", "Maintenance Notice", "Ban Notice"],
            region=(0.02, 0.55, 0.35, 0.70), min_conf=0.6
        )
        if announcement:
            self._interceptor_streak += 1
            # Announcement is a WebView overlay that absorbs ALL touch events.
            # Only Android BACK key can close it. This triggers exit dialog ("是否結束"),
            # which P0 handler catches on the NEXT tick and clicks 取消 → lobby restored.
            print(f"[Interceptor] P1 内嵌公告 '{announcement.text}', BACK to close WebView")
            return action_back(f"interceptor: close announcement WebView")

        # ── P1: Generic popup with X close button ──
        # Popups like 全體課程表, 通知, 課程表資訊 etc. have an X button.
        # Exempt Schedule (legacy direct skill) AND DailyRoutine's Schedule
        # sub-state (current default — Schedule runs as DailyRoutine's sub).
        # Without this, Schedule opens 全體課程表 and immediately interceptor
        # closes it → infinite reopen/close loop. (bug 2026-05-28)
        popup_titles = screen.find_any_text(
            ["全體課程表", "全体课程表", "課程表資訊", "课程表资讯", "課程表報告", "课程表报告"],
            region=(0.20, 0.05, 0.80, 0.25), min_conf=0.6
        )
        _in_schedule = (
            skill.name == "Schedule"
            or (skill.name == "DailyRoutine" and getattr(skill, "sub_state", "") == "Schedule")
        )
        if popup_titles and not _in_schedule:
            x_btn = screen.find_text_one("X", region=(0.85, 0.05, 0.95, 0.20), min_conf=0.4)
            if x_btn:
                print(f"[Interceptor] P1 stale popup '{popup_titles.text}', clicking X")
                return action_click_box(x_btn, f"interceptor: close popup X ({popup_titles.text})")
            print(f"[Interceptor] P1 stale popup '{popup_titles.text}', hardcoded X")
            return action_click(0.890, 0.142, f"interceptor: close popup ({popup_titles.text})")

        # ── P2: Account / Student Level Up ──
        # Full-screen "Level Up" / "Touch to Continue" effect
        levelup = screen.find_any_text(
            ["Level Up", "LEVEL UP", "Touch to Continue", "Touch To Continue",
             "タッチして続ける", "觸摸繼續", "触摸继续", "点击继续", "點擊繼續"],
            min_conf=0.6
        )
        if levelup:
            print(f"[Interceptor] P2 level-up: '{levelup.text}', blind-clicking edge")
            self._interceptor_streak += 1
            return action_click(0.05, 0.95, f"interceptor: dismiss level-up ({levelup.text})")

        # ── P2: "TAP TO CONTINUE" full-screen overlay ──
        # Reward popups, bond level-ups, rank-ups etc. show this prompt.
        # OCR may read as "TAP TO CONTINUE" or "TAPTO CONTINUE" (merged).
        tap_continue = screen.find_any_text(
            ["TAP TO CONTINUE", "TAPTO CONTINUE", "TAP TO", "CONTINUE"],
            region=(0.2, 0.75, 0.8, 0.98), min_conf=0.7
        )
        if tap_continue:
            self._interceptor_streak += 1
            # Click on the TAP TO CONTINUE text itself — NOT center (0.5,0.5)
            # which lands on reward cards that absorb clicks without dismissing.
            if self._interceptor_streak > 5:
                close_btn = _find_florence_hit(
                    screen,
                    ["close button icon", "close dialog x button", "x close icon"],
                    region=(0.62, 0.06, 0.94, 0.28),
                )
                if close_btn:
                    print(f"[Interceptor] P2 TAP TO CONTINUE stuck, clicking X")
                    return action_click_box(close_btn, "interceptor: close reward X")
                # Fallback: click bottom-left corner (outside cards)
                print(f"[Interceptor] P2 TAP TO CONTINUE stuck, clicking corner")
                return action_click(0.15, 0.85, f"interceptor: tap corner ({tap_continue.text})")
            print(f"[Interceptor] P2 TAP TO CONTINUE: '{tap_continue.text}', clicking text")
            return action_click_box(tap_continue, f"interceptor: tap to continue ({tap_continue.text})")

        # ── P2: Reward popup (獲得獎勵!) ──
        # Shows after cafe earnings, sweep, etc. Has 領取 button or TAP TO CONTINUE.
        reward = screen.find_any_text(
            ["獲得獎勵", "获得奖励", "獲得奖", "獲得獎"],
            min_conf=0.6
        )
        if reward:
            # Try clicking 領取/确认/確認 button first
            dismiss_btn = screen.find_any_text(
                ["確認", "确认", "確定", "确定", "領取", "领取", "確", "确", "OK"],
                region=(0.25, 0.80, 0.80, 0.98), min_conf=0.6
            )
            if dismiss_btn:
                print(f"[Interceptor] P2 reward popup, clicking {dismiss_btn.text}")
                self._interceptor_streak += 1
                return action_click_box(dismiss_btn, f"interceptor: claim reward ({reward.text})")
            # Fallback: tap 確認 button area (right-side button on Battle Complete reward)
            print(f"[Interceptor] P2 reward popup '{reward.text}', tapping confirm area")
            self._interceptor_streak += 1
            return action_click(0.60, 0.92, f"interceptor: dismiss reward ({reward.text})")

        # ── P2: Bond / Rank-up popups (好感度升級, 羈絆升級, Rank Up) ──
        bond_popup = screen.find_any_text(
            ["好感度", "羈絆升級", "羁绊升级", "Rank Up"],
            min_conf=0.6
        )
        if bond_popup:
            # These are full-screen dismiss-by-tapping popups
            if not screen.is_lobby():
                print(f"[Interceptor] P2 bond/rank popup: '{bond_popup.text}', tapping to dismiss")
                self._interceptor_streak += 1
                return action_click(0.5, 0.5, f"interceptor: dismiss bond/rank ({bond_popup.text})")

        # ── P1: Stale popups (only fire if current skill is NOT lobby) ──
        # Lobby already handles its own popups thoroughly.
        if skill.name != "Lobby":
            # "Strong" indicators: only appear in popup overlays, never on
            # the normal lobby.  One strong match is enough to fire.
            _STRONG_POPUP = [
                "今日不再", "Main News", "Patch Notes", "Pick-Up",
                "到簿", "签到", "簽到", "Maintenance", "Webpage",
            ]
            # "Weak" indicators: can appear on normal screens too
            # (公告 = lobby sidebar, 通知 = dialog header, Events = banner,
            #  Discord/Forum = Club UI shows "社团DISCORD群" as normal text).
            # A weak match alone MUST NOT fire the hardcoded X fallback.
            _WEAK_POPUP = ["公告", "通知", "Official", "Events", "Update",
                           "My Office", "Sensei", "Discord", "Forum"]

            strong_hit = screen.find_any_text(_STRONG_POPUP, min_conf=0.7)
            weak_hit = screen.find_any_text(_WEAK_POPUP, min_conf=0.7)
            popup_text = strong_hit or weak_hit
            is_strong = strong_hit is not None

            if popup_text:
                # ── Universal 通知 confirm dialog handler ──
                if popup_text.text in ("通知",):
                    cancel_btn = screen.find_any_text(
                        ["取消"],
                        region=screen.CENTER, min_conf=0.6
                    )
                    confirm_btn = screen.find_any_text(
                        ["確認", "确认", "確定", "确定", "確", "确"],
                        region=screen.CENTER, min_conf=0.6
                    )
                    if cancel_btn and confirm_btn:
                        print(f"[Interceptor] P1 通知 confirm dialog (取消+確認), clicking confirm")
                        return action_click_box(confirm_btn, "interceptor: confirm 通知 dialog")
                    if cancel_btn or confirm_btn:
                        return None

                # 今日不再顯示 checkbox: the OCR text label is the only
                # OCR-detectable anchor, but clicking the label itself
                # doesn't toggle the checkbox (it sits to the LEFT of the
                # label, not on it).  Old code clicked the label and
                # looped forever (run_20260504_215753 burned 160 ticks).
                # Click the checkbox spot once (text.x1 - small offset),
                # then PROCEED to the X close button.  One-shot via
                # _dnsa_toggled flag — don't re-toggle every tick.
                do_not_show = screen.find_any_text(
                    ["今日不再", "今日不再提示", "今日不再顯示", "今日不再显示"],
                    min_conf=0.7
                )
                if do_not_show and not getattr(self, "_dnsa_toggled", False):
                    self._dnsa_toggled = True
                    # Checkbox sits to the LEFT of the text label.  Click
                    # at ~text.x1 - 0.025 to hit the box itself (BA's
                    # checkbox is ~0.02-0.03 wide).
                    cx_left = max(0.005, do_not_show.x1 - 0.025)
                    print(f"[Interceptor] P1 popup: clicking 今日不再 checkbox (left of label)")
                    return action_click(
                        cx_left, do_not_show.cy,
                        "interceptor: toggle do-not-show-again checkbox"
                    )
                # If do_not_show isn't on screen anymore, the promo
                # popup has closed — reset toggle so a future popup
                # gets a fresh attempt.
                if not do_not_show:
                    self._dnsa_toggled = False

                # After toggling the do-not-show checkbox, close the popup
                # via top-right X.  ONLY fire when we previously detected
                # a real promo (do_not_show was found on a prior tick →
                # _dnsa_toggled set).  Without this gate, ANY weak popup
                # hit (e.g. "公告" on lobby) would fall here and click
                # empty space at (0.955, 0.065) forever
                # (run_20260504_221135 burned 86 ticks doing exactly that).
                if getattr(self, "_dnsa_toggled", False) and is_strong:
                    self._interceptor_streak += 1
                    if self._interceptor_streak > 8:
                        self._interceptor_streak = 0
                        self._dnsa_toggled = False
                        return action_back("interceptor: ESC burst (promo popup stuck)")
                    _PROMO_X_POSITIONS = [(0.955, 0.065), (0.95, 0.07), (0.97, 0.05)]
                    px, py = _PROMO_X_POSITIONS[
                        (self._interceptor_streak - 1) % len(_PROMO_X_POSITIONS)
                    ]
                    print(f"[Interceptor] P1 promo popup: hardcoded X at ({px},{py})")
                    return action_click(
                        px, py,
                        f"interceptor: close promo popup X (streak={self._interceptor_streak})"
                    )

                close_btn = _find_florence_hit(
                    screen,
                    ["close button icon", "close dialog x button", "x close icon"],
                    region=(0.0, 0.0, 0.93, 0.35),
                )
                if close_btn:
                    self._interceptor_streak += 1
                    print(f"[Interceptor] P1 stale popup: '{popup_text.text}' + Florence X, clicking X")
                    if self._interceptor_streak > 8:
                        print("[Interceptor] P1 popup won't close after 8 attempts, ESC burst")
                        self._interceptor_streak = 0
                        return action_back("interceptor: ESC burst (popup stuck)")
                    return action_click_box(close_btn, f"interceptor: close stale popup ({popup_text.text})")

                # No YOLO X detected — hardcoded X fallback.
                # ONLY use hardcoded fallback for STRONG indicators.
                # Weak indicators (e.g. "公告" on lobby sidebar) must NOT
                # trigger hardcoded clicks — that hits the "1/2" page counter.
                if is_strong:
                    self._interceptor_streak += 1
                    _ANNOUNCE_X_POSITIONS = [
                        (0.8735, 0.1575),
                        (0.87, 0.16),
                        (0.88, 0.15),
                    ]
                    idx = (self._interceptor_streak - 1) % len(_ANNOUNCE_X_POSITIONS)
                    px, py = _ANNOUNCE_X_POSITIONS[idx]
                    print(f"[Interceptor] P1 popup: '{popup_text.text}' (strong, no YOLO X), hardcoded X at ({px},{py})")
                    if self._interceptor_streak > 10:
                        self._interceptor_streak = 0
                        return action_back("interceptor: ESC burst (popup stuck, no X)")
                    return action_click(px, py, f"interceptor: close popup X hardcoded ({popup_text.text})")

        # No interceptor fired — reset streak + do-not-show-again flag
        self._interceptor_streak = 0
        self._dnsa_toggled = False
        return None

    def tick(self, screenshot_path: str) -> Dict[str, Any]:
        """Process one frame. Returns an action dict for the executor.

        Call this every frame with a fresh screenshot path.
        The returned action should be executed by the input system.
        """
        screen = read_screen(screenshot_path)
        return self._tick_with_screen(screen, screenshot_path=screenshot_path)

    def tick_from_frame(self, frame_bgr, *, screenshot_path: str = "",
                        skip_ocr: bool = False,
                        prev_ocr_boxes=None,
                        injected_yolo_boxes=None) -> Dict[str, Any]:
        """Process one in-memory BGR frame. Returns an action dict.

        Args:
            skip_ocr: skip expensive OCR, reuse prev_ocr_boxes.
            prev_ocr_boxes: cached OCR boxes from a previous tick.
            injected_yolo_boxes: pre-computed YOLO boxes from high-FPS thread.
                If provided, skip running YOLO in read_screen_from_frame.
        """
        screen = read_screen_from_frame(frame_bgr, screenshot_path=screenshot_path,
                                        skip_ocr=skip_ocr,
                                        prev_ocr_boxes=prev_ocr_boxes,
                                        injected_yolo_boxes=injected_yolo_boxes)
        return self._tick_with_screen(screen, screenshot_path=screenshot_path)

    @property
    def last_screen(self) -> Optional[ScreenState]:
        """Last ScreenState processed by tick (for overlay access)."""
        return getattr(self, '_last_screen', None)

    def _dedup_click(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Suppress stale repeated clicks.

        If the same click target+reason fires twice in a row, the second
        time is almost certainly acting on stale OCR data from before the
        screen transitioned.  Convert it to a short wait instead.
        """
        action_type = action.get("action", "")
        if action_type != "click":
            # Reset tracking on non-click actions
            self._last_click_target = None
            self._last_click_reason = ""
            self._click_repeat_count = 0
            return action

        target = action.get("target")
        reason = str(action.get("reason", "") or "")

        # Compare with positional tolerance — OCR returns slightly different
        # bounding boxes each frame, so the "same" button gets different coords.
        same_target = False
        if target and self._last_click_target:
            try:
                dx = abs(target[0] - self._last_click_target[0])
                dy = abs(target[1] - self._last_click_target[1])
                same_target = dx < 0.03 and dy < 0.03
            except (TypeError, IndexError):
                same_target = target == self._last_click_target

        if same_target and reason == self._last_click_reason:
            self._click_repeat_count += 1
            if self._click_repeat_count <= 2:
                # Allow max 2 repeats (3 total), then throttle
                return action
            # Throttle — convert to wait AND reset counter so the next
            # attempt gets a fresh start.  This prevents permanent deadlock
            # when a click genuinely needs retrying (game lag, slow transition).
            print(f"[Pipeline] Click throttled (repeat #{self._click_repeat_count}): {reason}")
            self._last_click_target = None
            self._last_click_reason = ""
            self._click_repeat_count = 0
            return action_wait(400, f"click throttled: {reason}")
        else:
            self._last_click_target = target
            self._last_click_reason = reason
            self._click_repeat_count = 0
            return action

    def _tick_with_screen(self, screen: ScreenState, *, screenshot_path: str = "") -> Dict[str, Any]:
        """Internal tick logic shared by tick() and tick_from_frame()."""
        self._last_screen = screen
        if not self._running:
            return action_done("pipeline not running")

        self._total_ticks += 1

        # Hard-example mining: auto-save borderline-conf detections for human
        # review later. No-op if no detections in the noisy conf band, so cost
        # is tiny (a list comp).  Module-level cap prevents disk spam.
        try:
            from brain.hard_example_mining import maybe_save_hard_example
            cur_skill = self.current_skill
            maybe_save_hard_example(
                screenshot_path=screenshot_path or None,
                yolo_boxes=screen.yolo_boxes or [],
                run_id=_PIPELINE_SESSION_ID,
                tick=self._total_ticks,
                skill_name=cur_skill.name if cur_skill else "",
                sub_state=getattr(cur_skill, "sub_state", "") if cur_skill else "",
            )
        except Exception:
            pass  # mining is best-effort, never break the tick loop

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
        else:
            self._no_ba_ticks = getattr(self, "_no_ba_ticks", 0) + 1
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
            if self._no_ba_ticks >= 30:
                print(
                    f"[Pipeline] No Blue Archive UI detected for "
                    f"{self._no_ba_ticks} consecutive ticks — aborting "
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

        # Dot-driven skip: on the FIRST tick after skill start, call
        # should_run(screen). Daily-harvest skills (cafe / mail / schedule /
        # club / daily_tasks / event_activity) override should_run to look
        # for their red/yellow dot — no dot = no work = advance to next skill.
        # Battle / sweep skills don't override should_run so always pass.
        if skill.ticks == 0:
            try:
                if _FORCE_ALL_SKILLS:
                    print(f"[Pipeline] '{skill.name}' FORCE-RUN (--force-skills, dot gate bypassed)")
                elif not skill.should_run(screen):
                    print(f"[Pipeline] '{skill.name}' should_run=False (no dot) → skip")
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
            except Exception as e:
                print(f"[Pipeline] should_run check failed for {skill.name}: {e}")

        # Let skill decide
        action = skill.tick(screen)
        action = self._dedup_click(action)
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
