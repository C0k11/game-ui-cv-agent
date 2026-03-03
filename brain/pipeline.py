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
    action_wait, action_done,
)
from brain.skills.lobby import LobbySkill
from brain.skills.cafe import CafeSkill
from brain.skills.schedule import ScheduleSkill
from brain.skills.club import ClubSkill
from brain.skills.bounty import BountySkill
from brain.skills.mail import MailSkill


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
# Prefer the ml_cache path if it exists
_YOLO_ML_CACHE = Path(r"D:\Project\ml_cache\models\yolo\full.pt")

_yolo_load_attempted = False

def _get_yolo():
    """Get or create YOLO model (thread-safe singleton). Returns None if unavailable."""
    global _yolo_model, _yolo_lock, _yolo_load_attempted
    import threading
    if _yolo_lock is None:
        _yolo_lock = threading.Lock()
    with _yolo_lock:
        if _yolo_model is not None:
            return _yolo_model
        if _yolo_load_attempted:
            return None
        _yolo_load_attempted = True
        model_path = None
        if _YOLO_ML_CACHE.is_file():
            model_path = _YOLO_ML_CACHE
        elif _YOLO_MODEL_PATH.is_file():
            model_path = _YOLO_MODEL_PATH
        if model_path is None:
            print(f"[Pipeline] YOLO model NOT found at {_YOLO_ML_CACHE} or {_YOLO_MODEL_PATH}")
            return None
        try:
            from ultralytics import YOLO
            _yolo_model = YOLO(str(model_path))
            # Warm-up inference to catch early errors
            import numpy as np
            _yolo_model(np.zeros((64, 64, 3), dtype=np.uint8), verbose=False)
            print(f"[Pipeline] YOLO model loaded OK from {model_path} (classes: {_yolo_model.names})")
            return _yolo_model
        except Exception as e:
            print(f"[Pipeline] YOLO load FAILED: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            return None


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
    ocr = _get_ocr()
    result, _ = ocr(img)

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
                x1=min(xs) / w,
                y1=min(ys) / h,
                x2=max(xs) / w,
                y2=max(ys) / h,
            ))

    # YOLO detection — pass the in-memory image (numpy array) instead of
    # the file path to avoid potential path-encoding issues on Windows.
    yolo_boxes: List[YoloBox] = []
    yolo = _get_yolo()
    if yolo is not None:
        try:
            yolo_results = yolo(img, conf=0.15, verbose=False)
            for r in yolo_results:
                for box in r.boxes:
                    bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0])
                    cls_name = yolo.names.get(cls_id, str(cls_id))
                    yolo_boxes.append(YoloBox(
                        cls_id=cls_id,
                        cls_name=cls_name,
                        confidence=float(box.conf[0]),
                        x1=bx1 / w,
                        y1=by1 / h,
                        x2=bx2 / w,
                        y2=by2 / h,
                    ))
        except Exception as e:
            print(f"[Pipeline] YOLO detect error: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()

    return ScreenState(
        ocr_boxes=boxes,
        yolo_boxes=yolo_boxes,
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

    # Default skill sequence for daily routine
    DEFAULT_SKILLS = [
        "lobby",
        "cafe",
        "schedule",
        "club",
        "bounty",
        "mail",
    ]

    TRAJECTORIES_DIR = Path(__file__).resolve().parents[1] / "data" / "trajectories"

    def __init__(self, skill_names: Optional[List[str]] = None):
        self._skill_registry: Dict[str, BaseSkill] = {
            "lobby": LobbySkill(),
            "cafe": CafeSkill(),
            "schedule": ScheduleSkill(),
            "club": ClubSkill(),
            "bounty": BountySkill(),
            "mail": MailSkill(),
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
        """Reset and start the current skill."""
        skill = self.current_skill
        if skill:
            skill.reset()
            self._skill_start_time = time.time()
            print(f"[Pipeline] Starting skill: {skill.name} ({self._current_idx + 1}/{len(self._skill_order)})")

    def _advance_skill(self, status: str) -> None:
        """Record result and move to next skill."""
        skill = self.current_skill
        if skill:
            result = SkillResult(
                skill_name=skill.name,
                status=status,
                ticks=skill.ticks,
                duration_s=time.time() - self._skill_start_time,
            )
            self._results.append(result)
            print(f"[Pipeline] Skill '{skill.name}' {status} ({skill.ticks} ticks, {result.duration_s:.1f}s)")

        self._current_idx += 1
        self._retry_count = 0

        if self._current_idx >= len(self._skill_order):
            self._running = False
            print(f"[Pipeline] All skills complete! Total ticks: {self._total_ticks}")
            for r in self._results:
                print(f"  {r.skill_name}: {r.status} ({r.ticks} ticks, {r.duration_s:.1f}s)")
        else:
            self._start_current_skill()

    def tick(self, screenshot_path: str) -> Dict[str, Any]:
        """Process one frame. Returns an action dict for the executor.

        Call this every frame with a fresh screenshot path.
        The returned action should be executed by the input system.
        """
        if not self._running:
            return action_done("pipeline not running")

        self._total_ticks += 1
        skill = self.current_skill
        if skill is None:
            self._running = False
            return action_done("no more skills")

        # Read screen via OCR
        screen = read_screen(screenshot_path)

        # Let skill decide
        action = skill.tick(screen)
        action_type = action.get("action", "")

        # Save trajectory
        self._save_trajectory(screenshot_path, screen, skill, action)

        # Skill finished
        if action_type == "done":
            self._advance_skill("done")
            return action_wait(300, f"skill '{skill.name}' done, advancing")

        # Timeout check
        if skill.ticks >= skill.max_ticks:
            if self._retry_count < self._max_retries:
                self._retry_count += 1
                print(f"[Pipeline] Skill '{skill.name}' timeout, retry {self._retry_count}")
                skill.reset()
                return action_wait(500, f"skill '{skill.name}' retry")
            else:
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
