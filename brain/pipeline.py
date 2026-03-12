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
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from brain.skills.base import (
    BaseSkill, OcrBox, YoloBox, ScreenState,
    action_click, action_click_box, action_back, action_wait, action_done,
)
from brain.skills.lobby import LobbySkill
from brain.skills.cafe import CafeSkill
from brain.skills.schedule import ScheduleSkill
from brain.skills.club import ClubSkill
from brain.skills.bounty import BountySkill
from brain.skills.mail import MailSkill
from brain.skills.arena import ArenaSkill
from brain.skills.farming import FarmingSkill
from brain.skills.event_farming import EventFarmingSkill
from brain.skills.daily_tasks import DailyTasksSkill
from brain.skills.shop import ShopSkill
from brain.skills.craft import CraftSkill


# ── OCR Engine (singleton) ──────────────────────────────────────────────

_ocr_engine = None
_ocr_lock = None

def _get_ocr():
    """Get or create RapidOCR engine (thread-safe singleton)."""
    global _ocr_engine, _ocr_lock
    import threading
    if _ocr_lock is None:
        _ocr_lock = threading.Lock()
    with _ocr_lock:
        if _ocr_engine is None:
            from rapidocr_onnxruntime import RapidOCR
            _ocr_engine = RapidOCR()
        return _ocr_engine


# ── YOLO Detector (singleton) ───────────────────────────────────────

_yolo_model = None
_yolo_lock = None
_YOLO_MODEL_PATH = Path(__file__).resolve().parents[1] / "data" / "_yolo_full.pt"
# Prefer custom headpat model > TensorRT > full.pt
_YOLO_HEADPAT = Path(r"D:\Project\ml_cache\models\yolo\headpat.pt")
_YOLO_TRT_ENGINE = Path(r"D:\Project\ml_cache\models\yolo\full.engine")
_YOLO_ML_CACHE = Path(r"D:\Project\ml_cache\models\yolo\full.pt")

_yolo_load_attempts = 0
_MAX_YOLO_LOAD_ATTEMPTS = 3
_yolo_status = "not_attempted"
_YOLO_ALLOWED_SUBSTRINGS = (
    "角色头像",
    "角色可摸头黄色感叹号",
    "感叹号",
    "Emoticon_Action",
    "headpat_bubble",
)

def _get_yolo():
    """Get or create YOLO model (lazy singleton). Only loads on first call.

    Deferred loading so startup is fast; model loads when Cafe headpat
    first needs it (~30s one-time cost).
    """
    global _yolo_model, _yolo_lock, _yolo_load_attempts, _yolo_status
    import threading
    if _yolo_lock is None:
        _yolo_lock = threading.Lock()
    with _yolo_lock:
        if _yolo_model is not None:
            return _yolo_model
        if _yolo_load_attempts >= _MAX_YOLO_LOAD_ATTEMPTS:
            return None
        _yolo_load_attempts += 1
        candidates = []
        if _YOLO_HEADPAT.is_file():
            candidates.append(_YOLO_HEADPAT)
        if _YOLO_TRT_ENGINE.is_file():
            candidates.append(_YOLO_TRT_ENGINE)
        if _YOLO_ML_CACHE.is_file():
            candidates.append(_YOLO_ML_CACHE)
        if _YOLO_MODEL_PATH.is_file():
            candidates.append(_YOLO_MODEL_PATH)
        if not candidates:
            _yolo_status = "model_not_found"
            print(f"[Pipeline] YOLO model NOT found")
            return None
        from ultralytics import YOLO
        import numpy as np
        for model_path in candidates:
            try:
                m = YOLO(str(model_path))
                m(np.zeros((64, 64, 3), dtype=np.uint8), verbose=False)
                _yolo_model = m
                _yolo_status = f"loaded_ok ({len(m.names)} classes)"
                print(f"[Pipeline] YOLO loaded from {model_path}")
                return _yolo_model
            except Exception as e:
                print(f"[Pipeline] YOLO load failed for {model_path}: {e}")
                continue
        _yolo_status = "all_candidates_failed"
        return None


_OCR_WORK_W = 1280  # Downscale wide frames for faster OCR


def _run_ocr_on_image(img, w: int, h: int) -> List[OcrBox]:
    """Run OCR on a BGR numpy array and return normalized OcrBox list.

    Downscales frames wider than _OCR_WORK_W for speed (4K→1280px ≈ 9x faster).
    Coordinates are normalized 0-1 so the caller is resolution-independent.
    """
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


def _run_yolo_on_image(img, w: int, h: int) -> List[YoloBox]:
    """Run YOLO on a BGR numpy array and return normalized YoloBox list.

    Non-blocking: if YOLO is still loading in the pre-warm thread, returns
    empty immediately instead of blocking the pipeline worker.
    """
    yolo_boxes: List[YoloBox] = []
    # Non-blocking check: if lock is held (pre-warm loading), skip this tick
    if _yolo_lock is not None and _yolo_model is None:
        acquired = _yolo_lock.acquire(blocking=False)
        if not acquired:
            # Pre-warm thread is loading YOLO, skip silently
            return yolo_boxes
        _yolo_lock.release()
    yolo = _get_yolo()
    if yolo is None:
        if not getattr(_run_yolo_on_image, '_warned', False):
            print(f"[Pipeline] YOLO unavailable: {_yolo_status}")
            _run_yolo_on_image._warned = True
        return yolo_boxes
    try:
        yolo_results = yolo(img, conf=0.15, verbose=False)
        for r in yolo_results:
            for box in r.boxes:
                bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                cls_name = yolo.names.get(cls_id, str(cls_id))
                cls_low = str(cls_name).lower()
                if not any(token.lower() in cls_low for token in _YOLO_ALLOWED_SUBSTRINGS):
                    continue
                nx1, ny1, nx2, ny2 = bx1/w, by1/h, bx2/w, by2/h
                # Filter headpat_bubble: only accept in cafe play area
                # (reject detections on popups, earnings icons, UI overlays)
                if "headpat" in cls_low:
                    # Cafe play area: x 0.05-0.95, y 0.15-0.85
                    # Reject if center is in popup overlay zone (center screen)
                    bcx = (nx1 + nx2) / 2
                    bcy = (ny1 + ny2) / 2
                    if bcy < 0.15 or bcy > 0.85:
                        continue  # In top/bottom UI bars
                    # Reject very small boxes (UI icons, not real bubbles)
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
                ))
    except Exception as e:
        print(f"[Pipeline] YOLO detect error: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
    return yolo_boxes


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
    """Run headpat bubble detection via HSV color filtering. Returns list of TemplateHitBox."""
    from brain.skills.base import TemplateHitBox
    hits = []
    try:
        from vision.template_matcher import find_headpat_bubbles
        # Restrict to cafe play area (exclude UI bars + left sidebar icons)
        raw = find_headpat_bubbles(frame_bgr, threshold=0.78,
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
                           prev_ocr_boxes=None) -> ScreenState:
    """Build ScreenState from an in-memory BGR numpy array (no file I/O).

    Used by the MuMu runner for zero-copy capture → detect pipeline.

    Args:
        skip_ocr: if True, skip OCR (expensive ~50ms) and reuse prev_ocr_boxes.
                  YOLO + template still run every frame (~3ms).
        prev_ocr_boxes: OCR boxes from a previous tick to reuse when skip_ocr=True.
    """
    if frame_bgr is None:
        return ScreenState(screenshot_path=screenshot_path)
    h, w = frame_bgr.shape[:2]
    if skip_ocr and prev_ocr_boxes is not None:
        ocr_boxes = prev_ocr_boxes
    else:
        ocr_boxes = _run_ocr_on_image(frame_bgr, w, h)
    yolo_boxes = _run_yolo_on_image(frame_bgr, w, h)
    template_hits = _run_template_matching(frame_bgr)
    return ScreenState(
        ocr_boxes=ocr_boxes,
        yolo_boxes=yolo_boxes,
        template_hits=template_hits,
        image_w=w,
        image_h=h,
        screenshot_path=screenshot_path,
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
    DEFAULT_SKILLS = [
        "lobby",            # 1.  Login, close popups, confirm lobby
        "ap_overflow",      # 2.  Emergency farm if AP≥900 (prevents cafe block)
        "cafe",             # 3.  Collect earnings, headpat students
        "schedule",         # 4.  Run schedules with tickets
        "club",             # 5.  Claim club AP (社交→社團)
        "shop",             # 6.  Buy daily items (一般 tab, select all, purchase)
        "craft",            # 7.  Quick-craft items + claim finished crafts
        "event_farming",    # 8.  Burn AP first when entering campaign
        "bounty",           # 9.  Sweep bounty tickets (3 branches)
        "arena",            # 10. PvP fights + claim rewards
        "mail",             # 11. Collect mail rewards (AP, items)
        "daily_tasks",      # 12. Claim daily task rewards + activity chests
        "hard_farming",     # 13. Hard mode shard farming (remaining AP)
        "event_farming_2",  # 14. 回马枪: second event sweep with collected AP
    ]

    TRAJECTORIES_DIR = Path(__file__).resolve().parents[1] / "data" / "trajectories"

    def __init__(self, skill_names: Optional[List[str]] = None):
        self._skill_registry: Dict[str, BaseSkill] = {
            "lobby": LobbySkill(),
            "ap_overflow": EventFarmingSkill(ap_threshold=900),  # only farms if AP≥900
            "cafe": CafeSkill(),
            "schedule": ScheduleSkill(),
            "club": ClubSkill(),
            "shop": ShopSkill(),
            "craft": CraftSkill(),
            "bounty": BountySkill(),
            "arena": ArenaSkill(),
            "event_farming": EventFarmingSkill(),       # event-aware (always farms)
            "hard_farming": FarmingSkill(),              # Hard mode shard farming
            "mail": MailSkill(),
            "daily_tasks": DailyTasksSkill(),
            "event_farming_2": EventFarmingSkill(),      # 回马枪 second pass
        }

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
        self._interceptor_streak: int = 0  # consecutive interceptor fires
        self._last_sub_state: str = ""
        self._last_wait_reason: str = ""
        self._stuck_counter: int = 0  # ticks in same sub_state
        self._consecutive_timeouts: int = 0  # skills that timed out in a row

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
            "results": [
                {"skill": r.skill_name, "status": r.status, "ticks": r.ticks}
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
        self._start_current_skill()
        print(f"[Pipeline] Started with {len(self._skill_order)} skills: {self._skill_order}")

    def stop(self) -> None:
        """Stop the pipeline."""
        self._running = False
        print("[Pipeline] Stopped")

    def _start_current_skill(self) -> None:
        """Reset and start the currently selected skill."""
        if not self._running:
            return
        skill = self.current_skill
        if skill is None:
            self._running = False
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
        print(f"[Pipeline] Starting skill '{skill.name}'")

    def _advance_skill(self, status: str) -> None:
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
        if self._current_idx >= len(self._skill_order):
            self._running = False
            print("[Pipeline] All skills complete")
            return
        self._start_current_skill()

    def _global_interceptor(self, screen: ScreenState, skill: BaseSkill) -> Optional[Dict[str, Any]]:
        """Global interceptor — runs BEFORE every skill tick.

        Handles "rude" popups that can appear at any time regardless of skill:
        P0: Disconnect / reconnect / download data
        P1: Stale sign-in / activity popups that leaked through
        P2: Account / student Level Up full-screen effects
        """
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

        # ── P0.5: Daily check-in calendar (彩奈签到簿) ──
        # Full-screen popup with NO close button. Just click anywhere to dismiss.
        # Detected by "簽到" / "签到" / "彩奈" text, or "第1天" / "第2天" grid.
        checkin = screen.find_any_text(
            ["簽到", "签到", "彩奈", "到薄", "到簿"],
            min_conf=0.5
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

        # ── P0.5: Updates / Patch Notes WebView ──
        # In-game WebView (same system as 主要消息). X button absorbs clicks unreliably.
        # BACK key closes it reliably → triggers exit dialog → P0 handler clicks 取消.
        updates = screen.find_any_text(
            ["Updates", "Patch Notes"],
            region=(0.30, 0.04, 0.70, 0.14), min_conf=0.5
        )
        if updates:
            print(f"[Interceptor] P0.5 Updates WebView: '{updates.text}', BACK to close")
            self._interceptor_streak += 1
            return action_back(f"interceptor: close Updates WebView ({updates.text})")

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
        popup_titles = screen.find_any_text(
            ["全體課程表", "全体课程表", "課程表資訊", "课程表资讯", "課程表報告", "课程表报告"],
            region=(0.20, 0.05, 0.80, 0.25), min_conf=0.6
        )
        if popup_titles and skill.name != "Schedule":
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
            # Try clicking 領取/领取 button first
            claim_btn = screen.find_any_text(
                ["領取", "领取"],
                region=screen.CENTER, min_conf=0.7
            )
            if claim_btn:
                print(f"[Interceptor] P2 reward popup, clicking 領取")
                self._interceptor_streak += 1
                return action_click_box(claim_btn, f"interceptor: claim reward ({reward.text})")
            # Fallback: tap bottom area to dismiss (NOT center — cards absorb clicks)
            print(f"[Interceptor] P2 reward popup '{reward.text}', tapping bottom to dismiss")
            self._interceptor_streak += 1
            return action_click(0.5, 0.87, f"interceptor: dismiss reward ({reward.text})")

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

                # Strong user preference: click 今日不再 checkbox first
                do_not_show = screen.find_any_text(
                    ["今日不再", "今日不再提示", "今日不再顯示", "今日不再显示"],
                    min_conf=0.7
                )
                if do_not_show:
                    print(f"[Interceptor] P1 popup: clicking 今日不再提示")
                    return action_click_box(do_not_show, "interceptor: do not show again today")

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

        # No interceptor fired — reset streak
        self._interceptor_streak = 0
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
                        prev_ocr_boxes=None) -> Dict[str, Any]:
        """Process one in-memory BGR frame. Returns an action dict.

        Args:
            skip_ocr: skip expensive OCR, reuse prev_ocr_boxes. YOLO still runs.
            prev_ocr_boxes: cached OCR boxes from a previous tick.
        """
        screen = read_screen_from_frame(frame_bgr, screenshot_path=screenshot_path,
                                        skip_ocr=skip_ocr,
                                        prev_ocr_boxes=prev_ocr_boxes)
        return self._tick_with_screen(screen, screenshot_path=screenshot_path)

    @property
    def last_screen(self) -> Optional[ScreenState]:
        """Last ScreenState processed by tick (for overlay access)."""
        return getattr(self, '_last_screen', None)

    def _tick_with_screen(self, screen: ScreenState, *, screenshot_path: str = "") -> Dict[str, Any]:
        """Internal tick logic shared by tick() and tick_from_frame()."""
        self._last_screen = screen
        if not self._running:
            return action_done("pipeline not running")

        self._total_ticks += 1
        skill = self.current_skill
        if skill is None:
            self._running = False
            return action_done("no more skills")

        # ── Global Interceptor (runs before any skill) ──
        intercept = self._global_interceptor(screen, skill)
        if intercept:
            self._save_trajectory(screenshot_path, screen, skill, intercept)
            return intercept

        # Let skill decide
        action = skill.tick(screen)
        action_type = action.get("action", "")
        action_reason = str(action.get("reason", "") or "")

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
        if action_type == "wait" and self._stuck_counter > 0 and self._stuck_counter % 20 == 0:
            if screen.is_lobby():
                print(f"[Pipeline] Skill '{skill.name}' repeating wait on lobby for {self._stuck_counter} ticks, skipping ESC (unsafe on lobby)")
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
                self._advance_skill("timeout")
                return action_wait(300, f"skill '{skill.name}' timed out, skipping")
            self._advance_skill("done")
            return action_wait(300, f"skill '{skill.name}' done, advancing")

        # Timeout check
        if skill.ticks >= skill.max_ticks:
            if self._retry_count < self._max_retries:
                self._retry_count += 1
                print(f"[Pipeline] Skill '{skill.name}' timeout, retry {self._retry_count}")
                skill.reset()
                return action_wait(500, f"skill '{skill.name}' retry")
            print(f"[Pipeline] Skill '{skill.name}' timeout, skipping")
            self._advance_skill("timeout")
            return action_wait(300, f"skill '{skill.name}' timed out, skipping")

        # Tag action with pipeline metadata
        action["_pipeline"] = True
        action["_skill"] = skill.name
        action["_tick"] = self._total_ticks
        return action

    def _save_trajectory(self, screenshot_path: str, screen: ScreenState,
                         skill: BaseSkill, action: Dict[str, Any]) -> None:
        """Save frame + OCR + action for debugging."""
        if self._traj_dir is None:
            return
        try:
            tick_id = f"tick_{self._total_ticks:04d}"
            # Copy frame image
            src = Path(screenshot_path)
            if src.exists():
                shutil.copy2(str(src), str(self._traj_dir / f"{tick_id}.jpg"))
            # Save OCR + action JSON
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
                "skill": skill.name,
                "sub_state": skill.sub_state,
                "skill_ticks": skill.ticks,
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
            json_path = self._traj_dir / f"{tick_id}.json"
            json_path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            print(f"[Pipeline] trajectory save error: {e}")

    def get_summary(self) -> str:
        """Get human-readable summary of pipeline execution."""
        lines = [f"Pipeline: {'Running' if self._running else 'Stopped'}"]
        lines.append(f"Total ticks: {self._total_ticks}")
        for r in self._results:
            lines.append(f"  {r.skill_name}: {r.status} ({r.ticks} ticks, {r.duration_s:.1f}s)")
        skill = self.current_skill
        if skill:
            lines.append(f"  {skill.name}: in progress ({skill.ticks} ticks, sub={skill.sub_state})")
        return "\n".join(lines)
