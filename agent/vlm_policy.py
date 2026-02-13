import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, List

import traceback

from PIL import Image, ImageStat

from action.windows_input import WindowsInput
from agent.daily_routine import DailyRoutineManager
from config import HF_CACHE_DIR, LOCAL_VLM_DEVICE, LOCAL_VLM_MAX_NEW_TOKENS, LOCAL_VLM_MODEL, LOCAL_VLM_MODELS_DIR
from vision.local_vlm_runtime import get_local_vlm

try:
    from cerebellum import Cerebellum
except Exception:
    Cerebellum = None  # type: ignore


def _center(bbox: Sequence[int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


def _parse_json_content(content: str) -> Dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    s = (content or "").strip()
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 3:
            s = parts[1]
            if "\n" in s:
                s = s.split("\n", 1)[1]
            s = s.strip()

    l = s.find("{")
    r = s.rfind("}")
    if l != -1 and r != -1 and r > l:
        return json.loads(s[l : r + 1])

    raise json.JSONDecodeError("Unable to parse JSON", content, 0)


def _parse_action_content(content: str) -> Dict[str, Any]:
    try:
        return _parse_json_content(content)
    except Exception:
        pass

    s = (content or "").strip()
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 3:
            s = parts[1]
            if "\n" in s:
                s = s.split("\n", 1)[1]
            s = s.strip()

    l = s.find("{")
    r = s.rfind("}")
    if l != -1 and r != -1 and r > l:
        s = s[l : r + 1]

    s2 = s
    try:
        s2 = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', s2)
        s2 = s2.replace("'", '"')
        s2 = re.sub(r"\bNone\b", "null", s2)
        s2 = re.sub(r"\bTrue\b", "true", s2)
        s2 = re.sub(r"\bFalse\b", "false", s2)
        s2 = re.sub(r",\s*([}\]])", r"\1", s2)
        return json.loads(s2)
    except Exception:
        pass

    out: Dict[str, Any] = {}
    try:
        m = re.search(r"\baction\b\s*[:=]\s*['\"]?(?P<a>[A-Za-z_]+)", s, flags=re.IGNORECASE)
        if m:
            out["action"] = str(m.group("a") or "").lower().strip()
    except Exception:
        pass
    try:
        m = re.search(r"\btarget\b\s*[:=]\s*\[\s*(?P<x>-?\d+)\s*,\s*(?P<y>-?\d+)\s*\]", s, flags=re.IGNORECASE)
        if m:
            out["target"] = [int(m.group("x")), int(m.group("y"))]
    except Exception:
        pass
    try:
        m = re.search(r"\bduration_ms\b\s*[:=]\s*(?P<d>\d+)", s, flags=re.IGNORECASE)
        if m:
            out["duration_ms"] = int(m.group("d"))
    except Exception:
        pass
    return out


def _parse_perception_items_fallback(content: str) -> List[Dict[str, Any]]:
    s = content or ""
    if not s:
        return []
    items: List[Dict[str, Any]] = []
    try:
        pattern = re.compile(
            r'\{"label"\s*:\s*"(?P<label>[^"]+)"[^{}]*?"bbox"\s*:\s*\[(?P<x1>-?\d+)\s*,\s*(?P<y1>-?\d+)\s*,\s*(?P<x2>-?\d+)\s*,\s*(?P<y2>-?\d+)\]'
        )
        for match in pattern.finditer(s):
            label = (match.group("label") or "").strip()
            if not label:
                continue
            try:
                bbox = [
                    int(match.group("x1")),
                    int(match.group("y1")),
                    int(match.group("x2")),
                    int(match.group("y2")),
                ]
            except Exception:
                continue
            items.append({"label": label, "bbox": bbox})
    except Exception:
        return items
    return items


def _looks_like_notice_board(items: Any) -> bool:
    if not isinstance(items, list) or not items:
        return False

    keys = (
        "公告",
        "通知",
        "活動",
        "活动",
        "更新",
        "維護",
        "维护",
        "news",
        "notice",
        "announcement",
        "event",
        "update",
        "maintenance",
    )
    hit = 0
    for it in items:
        if not isinstance(it, dict):
            continue
        lbl = str(it.get("label") or "").strip()
        if not lbl:
            continue
        ll = lbl.lower()
        if any((k in lbl) for k in keys if any(ord(c) > 127 for c in k)):
            hit += 1
        elif any((k in ll) for k in keys if all(ord(c) <= 127 for c in k)):
            hit += 1
        if hit >= 2:
            return True
    return False


def _looks_like_title_screen(items: Any) -> bool:
    if not isinstance(items, list) or not items:
        return False
    score = 0
    for it in items:
        if not isinstance(it, dict):
            continue
        lbl0 = str(it.get("label") or "").strip()
        if not lbl0:
            continue
        lbl = lbl0.lower()
        if ("menu" in lbl) or ("公告" in lbl0) or ("notice" in lbl):
            score += 2
        if ("uid" in lbl) or ("ver" in lbl) or ("version" in lbl) or ("region" in lbl) or ("blue archive" in lbl):
            score += 1
        if score >= 3:
            return True
    return False


def _looks_like_embedded_webview(items: Any) -> bool:
    if not isinstance(items, list) or not items:
        return False
    keys = (
        "webpage",
        "official",
        "discord",
        "forum",
        "twitter",
        "events",
        "updates",
        "announcements",
        "login",
        "ban notice",
        "blue archive",
        "my office",
    )
    hit = 0
    for it in items:
        if not isinstance(it, dict):
            continue
        lbl = str(it.get("label") or "").strip()
        if not lbl:
            continue
        ll = lbl.lower()
        if any(k in ll for k in keys):
            hit += 1
        if hit >= 2:
            return True
    return False


@dataclass
class VlmPolicyConfig:
    window_title: str = "Blue Archive"
    goal: str = "Keep the game running safely."
    steps: int = 0
    dry_run: bool = True
    step_sleep_s: float = 0.6

    vlm_image_max_side: int = field(
        default_factory=lambda: int(os.environ.get("VLM_IMAGE_MAX_SIDE", "1280") or "1280")
        if str(os.environ.get("VLM_IMAGE_MAX_SIDE", "1280") or "").strip().lstrip("-").isdigit()
        else 1280
    )

    model: str = LOCAL_VLM_MODEL
    models_dir: str = LOCAL_VLM_MODELS_DIR
    hf_home: str = HF_CACHE_DIR
    device: str = LOCAL_VLM_DEVICE
    max_new_tokens: int = LOCAL_VLM_MAX_NEW_TOKENS

    perception_max_new_tokens: int = 256
    perception_max_items: int = 40

    perception_every_n_steps: int = 4

    exploration_click: bool = False
    exploration_click_cooldown_steps: int = 3

    forbid_premium_currency: bool = True

    autoclick_safe_buttons: bool = True
    autoclick_safe_cooldown_steps: int = 2

    cafe_headpat: bool = True
    cafe_headpat_cooldown_steps: int = 1

    click_debounce_steps: int = 2
    click_debounce_dist_px: int = 24

    prompt_history_steps: int = 2

    supervision_enabled: bool = True
    supervision_image_max_side: int = 1280
    supervision_max_new_tokens: int = 128
    supervision_min_step_interval: int = 6
    supervision_fail_escalate_n: int = 3

    supervision_always: bool = True
    supervision_always_min_step_interval: int = 2

    cerebellum_enabled: bool = True
    cerebellum_assets_dir: str = "data/captures"
    cerebellum_confidence: float = 0.80
    cerebellum_template_start_anchor: str = "点击开始.png"
    cerebellum_template_notice_close: str = "内嵌公告的叉.png"
    cerebellum_template_lobby_check: str = "学生.png"

    out_dir: str = "data/trajectories"


class VlmPolicyAgent:
    def __init__(self, cfg: Optional[VlmPolicyConfig] = None):
        self.cfg = cfg or VlmPolicyConfig()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._last_action: Optional[Dict[str, Any]] = None
        self._last_error: str = ""

        self._routine = DailyRoutineManager()
        self._sync_routine_from_goal(self.cfg.goal)

        self._device = WindowsInput(title_substring=self.cfg.window_title)

        self._last_exploration_step: int = -10_000
        self._last_autoclick_step: int = -10_000
        self._last_headpat_step: int = -10_000
        self._last_headpat_xy: Optional[Tuple[int, int]] = None

        self._last_tap_to_start_step: int = -10_000
        self._last_tap_to_start_xy: Optional[Tuple[int, int]] = None

        self._startup_finished: bool = False

        self._startup_tap_attempts: int = 0
        self._startup_tap_attempts_step0: int = -10_000

        self._last_notice_x_step: int = -10_000
        self._last_notice_x_xy: Optional[Tuple[int, int]] = None
        self._notice_x_attempts: int = 0

        self._last_cerebellum_notice_step: int = -10_000
        self._cerebellum_notice_streak: int = 0

        self._cafe_last_claim_step: int = -10_000
        self._cafe_last_invite_step: int = -10_000
        self._cafe_last_store2_step: int = -10_000
        self._cafe_idle_steps: int = 0

        self._last_click_step: int = -10_000
        self._last_click_xy: Optional[Tuple[int, int]] = None
        self._last_click_reason: str = ""

        self._nav_force_miss_step_name: str = ""
        self._nav_force_miss_count: int = 0
        self._last_delegate_step: int = -10_000
        self._last_delegate_key: str = ""

        self._screenshot_fail_count: int = 0

        self._expected_state: Optional[str] = None
        self._expected_set_ts: float = 0.0
        self._expected_delay_s: float = 0.0
        self._last_supervision_step: int = -10_000
        self._last_supervision_any_step: int = -10_000
        self._supervision_fail_count: int = 0

        self._vlm_force_small_until_step: int = -10_000

        self._cerebellum_failed_once: bool = False

        self._cerebellum = None
        try:
            if bool(getattr(self.cfg, "cerebellum_enabled", True)) and Cerebellum is not None:
                assets_dir = str(getattr(self.cfg, "cerebellum_assets_dir", "assets") or "assets")
                try:
                    p0 = Path(assets_dir)
                    if assets_dir.strip() == "assets":
                        need_fallback = (not p0.exists()) or (not p0.is_dir())
                        if not need_fallback:
                            cand = [
                                str(getattr(self.cfg, "cerebellum_template_notice_close", "notice_close.png") or ""),
                                str(getattr(self.cfg, "cerebellum_template_start_anchor", "start_anchor.png") or ""),
                                str(getattr(self.cfg, "cerebellum_template_lobby_check", "lobby_check.png") or ""),
                                "内嵌公告的叉.png",
                                "游戏内很多页面窗口的叉.png",
                                "点击开始.png",
                            ]
                            found_any = False
                            for n in cand:
                                nn = str(n or "").strip()
                                if not nn:
                                    continue
                                try:
                                    if (p0 / nn).is_file():
                                        found_any = True
                                        break
                                except Exception:
                                    continue
                            need_fallback = not bool(found_any)

                        if need_fallback:
                            p1 = Path("data") / "captures"
                            if p1.exists() and p1.is_dir():
                                assets_dir = str(p1)
                except Exception:
                    pass
                conf = float(getattr(self.cfg, "cerebellum_confidence", 0.80) or 0.80)
                if assets_dir:
                    self._cerebellum = Cerebellum(assets_dir=assets_dir, confidence=float(conf))
        except Exception:
            self._cerebellum = None

        self._recent: list[dict[str, Any]] = []

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._run_id = f"run_{ts}"
        self._run_dir = Path(self.cfg.out_dir) / self._run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._traj_path = self._run_dir / "trajectory.jsonl"
        self._usage_path = self._run_dir / "model_usage.jsonl"

        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        self._out_log = logs_dir / "agent.out.log"
        self._err_log = logs_dir / "agent.err.log"

        try:
            if getattr(self, "_cerebellum", None) is not None:
                try:
                    import cv2  # type: ignore

                    _ = cv2
                except Exception as e:
                    ts2 = datetime.now().isoformat(timespec="seconds")
                    self._log_out(
                        f"[{ts2}] cerebellum disabled: OpenCV import failed: {type(e).__name__}: {e} "
                        f"(try: python -m pip install opencv-python-headless)"
                    )
                    self._cerebellum = None
                else:
                    c = getattr(self, "_cerebellum")
                    ts2 = datetime.now().isoformat(timespec="seconds")
                    ad = ""
                    try:
                        ad = str(getattr(c, "assets_dir", "") or "")
                    except Exception:
                        ad = ""
                    cf = None
                    try:
                        cf = float(getattr(c, "confidence", 0.0) or 0.0)
                    except Exception:
                        cf = None
                    self._log_out(f"[{ts2}] cerebellum enabled: assets_dir={ad} confidence={cf}")
        except Exception:
            pass

    @property
    def last_action(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._last_action

    @property
    def last_error(self) -> str:
        with self._lock:
            return self._last_error

    def is_running(self) -> bool:
        try:
            t = self._thread
            return bool(t is not None and t.is_alive())
        except Exception:
            return False

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._last_error = ""
            try:
                self._startup_finished = False
            except Exception:
                pass
            try:
                self._startup_tap_attempts = 0
                self._startup_tap_attempts_step0 = -10_000
            except Exception:
                pass
            try:
                self._last_tap_to_start_step = -10_000
                self._last_tap_to_start_xy = None
            except Exception:
                pass
            try:
                self._last_notice_x_step = -10_000
                self._last_notice_x_xy = None
                self._notice_x_attempts = 0
            except Exception:
                pass
            try:
                self._last_cerebellum_notice_step = -10_000
                self._cerebellum_notice_streak = 0
            except Exception:
                pass
            try:
                self._last_supervision_any_step = -10_000
            except Exception:
                pass
        try:
            self._sync_routine_from_goal(self.cfg.goal)
        except Exception:
            pass
        t = threading.Thread(target=self._run_loop, name="vlm_policy", daemon=True)
        self._thread = t
        t.start()

    def stop(self) -> None:
        try:
            self._stop.set()
        except Exception:
            pass
        t = None
        try:
            t = self._thread
        except Exception:
            t = None
        if t is not None:
            try:
                t.join(timeout=3.0)
            except Exception:
                pass

    def _sync_routine_from_goal(self, goal: str) -> None:
        g = str(goal or "")
        gl = g.lower()
        enable = False
        if g.strip().startswith("[Routine]"):
            enable = True
        elif "daily routine" in gl:
            enable = True
        elif "日常" in g or "收菜" in g:
            enable = True

        if enable:
            try:
                if not bool(getattr(self._routine, "is_active", False)):
                    self._routine.start_routine()
            except Exception:
                pass
        else:
            try:
                if bool(getattr(self._routine, "is_active", False)):
                    self._routine.stop_routine()
            except Exception:
                pass

    def _set_stage(self, *, step_id: int, stage: str, detail: str = "") -> None:
        try:
            a: Dict[str, Any] = {"action": "_stage", "step": int(step_id), "stage": str(stage)}
            if detail:
                a["detail"] = str(detail)
            self._write_traj({"ts": datetime.now().isoformat(timespec="seconds"), "step": int(step_id), "action": a})
        except Exception:
            pass

    def _maybe_cerebellum_action(self, *, screenshot_path: str, step_id: int) -> Optional[Dict[str, Any]]:
        try:
            if not bool(getattr(self.cfg, "cerebellum_enabled", True)):
                return None
        except Exception:
            return None
        c = getattr(self, "_cerebellum", None)
        if c is None:
            return None

        try:
            if bool(getattr(self, "_startup_finished", False)):
                return None
        except Exception:
            pass

        sw = 0
        sh = 0
        try:
            with Image.open(screenshot_path) as im:
                sw, sh = im.size
        except Exception:
            sw, sh = 0, 0

        roi_notice = None
        roi_start = None
        try:
            if sw > 0 and sh > 0:
                roi_notice = (int(round(float(sw) * 0.72)), int(round(float(sh) * 0.00)), int(sw), int(round(float(sh) * 0.26)))
                roi_start = (int(round(float(sw) * 0.40)), int(round(float(sh) * 0.74)), int(round(float(sw) * 0.60)), int(round(float(sh) * 0.92)))
        except Exception:
            roi_notice = None
            roi_start = None

        im_l = None
        try:
            if sw > 0 and sh > 0:
                with Image.open(screenshot_path) as im:
                    im_l = im.convert("L")
        except Exception:
            im_l = None

        try:
            import cv2  # type: ignore

            _ = cv2
        except Exception as e:
            try:
                if not bool(getattr(self, "_cerebellum_failed_once", False)):
                    self._cerebellum_failed_once = True
                    ts = datetime.now().isoformat(timespec="seconds")
                    self._log_out(f"[{ts}] cerebellum disabled: OpenCV import failed: {type(e).__name__}: {e}")
            except Exception:
                pass
            try:
                self._cerebellum = None
            except Exception:
                pass
            return None

        def _uniq(names: list[str]) -> list[str]:
            out: list[str] = []
            seen: set[str] = set()
            for n in names:
                nn = str(n or "").strip()
                if not nn or nn in seen:
                    continue
                out.append(nn)
                seen.add(nn)
            return out

        try:
            allow_nav_check = True
            try:
                if int(getattr(self, "_startup_tap_attempts", 0) or 0) <= 0 and int(step_id) <= 60:
                    allow_nav_check = False
            except Exception:
                allow_nav_check = True

            if allow_nav_check and sw > 0 and sh > 0:
                roi_nav = (int(0), int(round(float(sh) * 0.80)), int(sw), int(sh))
                tmpl_lobby0 = str(getattr(self.cfg, "cerebellum_template_lobby_check", "lobby_check.png") or "")
                nav_tmps = _uniq(
                    [
                        tmpl_lobby0,
                        "咖啡厅.png",
                        "课程表.png",
                        "社交.png",
                        "制造.png",
                        "学生.png",
                        "商店.png",
                        "招募.png",
                    ]
                )
                hits = 0
                try:
                    m0 = None
                    if tmpl_lobby0:
                        m0 = c.best_match(screenshot_path=screenshot_path, template_name=tmpl_lobby0, roi=roi_nav)
                    if m0 is not None and float(m0.score) >= 0.960:
                        self._startup_finished = True
                        return None
                except Exception:
                    pass

                for tmpl in nav_tmps:
                    if str(tmpl or "").strip() == str(tmpl_lobby0 or "").strip():
                        continue
                    try:
                        m = c.best_match(screenshot_path=screenshot_path, template_name=tmpl, roi=roi_nav)
                    except Exception:
                        m = None
                    try:
                        if m is None or float(m.score) < 0.935:
                            continue
                    except Exception:
                        continue
                    hits += 1
                    if hits >= 2:
                        self._startup_finished = True
                        return None
        except Exception:
            pass

        try:
            try:
                cd_steps = 4
                if int(step_id) - int(getattr(self, "_last_tap_to_start_step", -10_000)) < int(cd_steps):
                    return {
                        "action": "wait",
                        "duration_ms": 1200,
                        "reason": "Startup: tap-to-start cooldown; waiting for UI transition.",
                        "_startup_tap": True,
                    }
            except Exception:
                pass

            try:
                w0 = int(getattr(self, "_startup_tap_attempts_step0", -10_000) or -10_000)
                if int(step_id) - int(w0) > 18:
                    self._startup_tap_attempts_step0 = int(step_id)
                    self._startup_tap_attempts = 0
                if int(getattr(self, "_startup_tap_attempts", 0) or 0) >= 3:
                    return None
            except Exception:
                pass

            tmpl0 = str(getattr(self.cfg, "cerebellum_template_start_anchor", "start_anchor.png") or "")
            for tmpl in _uniq([tmpl0, "点击开始.png"]):
                act = c.click_action(
                    screenshot_path=screenshot_path,
                    template_name=tmpl,
                    reason_prefix="Cerebellum: tap to start.",
                    roi=roi_start,
                )
                if isinstance(act, dict):
                    try:
                        cb = act.get("_cerebellum", {})
                        ctr = cb.get("center")
                        if isinstance(ctr, (list, tuple)) and len(ctr) == 2 and sw > 0 and sh > 0:
                            cx, cy = int(ctr[0]), int(ctr[1])
                            ex = int(round(float(sw) * 0.50))
                            ey = int(round(float(sh) * 0.82))
                            if abs(int(cy) - int(ey)) > int(round(float(sh) * 0.06)):
                                continue
                            if abs(int(cx) - int(ex)) > int(round(float(sw) * 0.10)):
                                continue
                    except Exception:
                        pass

                    try:
                        cb = act.get("_cerebellum", {})
                        tname = str(cb.get("template") or "")
                        score = float(cb.get("score") or 0.0)
                        if tname == "点击开始.png" and float(score) < 0.87:
                            continue
                    except Exception:
                        pass

                    try:
                        cb = act.get("_cerebellum", {})
                        tname = str(cb.get("template") or "")
                        bb = cb.get("bbox")
                        if tname == "点击开始.png" and im_l is not None and isinstance(bb, (list, tuple)) and len(bb) == 4:
                            x1, y1, x2, y2 = [int(v) for v in bb]
                            x1 = max(0, min(int(sw) - 1, int(x1)))
                            y1 = max(0, min(int(sh) - 1, int(y1)))
                            x2 = max(int(x1) + 1, min(int(sw), int(x2)))
                            y2 = max(int(y1) + 1, min(int(sh), int(y2)))
                            crop = im_l.crop((int(x1), int(y1), int(x2), int(y2)))
                            std = 0.0
                            try:
                                std = float((ImageStat.Stat(crop).stddev or [0.0])[0])
                            except Exception:
                                std = 0.0
                            # Reject false positives on near-uniform/blank areas during loading.
                            if float(std) < 10.0:
                                continue
                    except Exception:
                        pass

                    act["_startup_tap"] = True
                    try:
                        self._last_tap_to_start_step = int(step_id)
                        self._last_tap_to_start_xy = (int(sw * 0.50), int(sh * 0.82)) if (sw > 0 and sh > 0) else None
                    except Exception:
                        pass
                    try:
                        if int(getattr(self, "_startup_tap_attempts_step0", -10_000) or -10_000) <= -9_000:
                            self._startup_tap_attempts_step0 = int(step_id)
                        self._startup_tap_attempts = int(getattr(self, "_startup_tap_attempts", 0) or 0) + 1
                    except Exception:
                        pass
                    return act
        except Exception:
            pass

        return None

    def _maybe_delegate_notice_close_to_cerebellum(self, action: Dict[str, Any], *, screenshot_path: str, step_id: int) -> Dict[str, Any]:
        if not isinstance(action, dict):
            return action
        if action.get("_close_heuristic") is not None:
            return action
        c = getattr(self, "_cerebellum", None)
        if c is None:
            return action

        try:
            boot_preinteractive = (not bool(getattr(self, "_startup_finished", False))) and int(getattr(self, "_startup_tap_attempts", 0) or 0) <= 0
            if boot_preinteractive and int(step_id) <= 60:
                return action
        except Exception:
            pass

        try:
            if not bool(getattr(self.cfg, "cerebellum_enabled", True)):
                return action
        except Exception:
            return action

        reason = str(action.get("reason") or "")
        raw = str(action.get("raw") or "")
        blob = reason + "\n" + raw
        blob_low = blob.lower()

        wants_delegate = "delegate: close_notice" in blob_low or "delegate_close_notice" in blob_low
        close_hint = wants_delegate or (
            ("公告" in blob)
            or ("通知" in blob)
            or ("webview" in blob_low)
            or ("notice" in blob_low)
            or ("announcement" in blob_low)
            or ("close" in blob_low)
            or ("关闭" in blob)
            or ("关掉" in blob)
            or ("popup" in blob_low)
        )
        if not close_hint:
            return action

        try:
            cd = 1 if wants_delegate else 2
            if int(step_id) - int(getattr(self, "_last_cerebellum_notice_step", -10_000)) <= int(cd):
                return action
        except Exception:
            pass

        try:
            streak = int(getattr(self, "_cerebellum_notice_streak", 0) or 0)
            last_step = int(getattr(self, "_last_cerebellum_notice_step", -10_000) or -10_000)
            total_att = int(getattr(self, "_notice_close_total_attempts", 0) or 0)
            if (streak >= 3 or total_att >= 3) and (int(step_id) - int(last_step)) <= 10:
                self._cerebellum_notice_streak = 0
                self._notice_close_total_attempts = 0
                self._last_cerebellum_notice_step = int(step_id)
                out = {
                    "action": "back",
                    "reason": "Repeated notice-close attempts; pressing ESC instead.",
                    "raw": action.get("raw"),
                    "_prompt": action.get("_prompt"),
                    "_perception": action.get("_perception"),
                    "_model": action.get("_model"),
                    "_routine": action.get("_routine"),
                    "_recovery": "cerebellum_notice_streak",
                }
                return out
        except Exception:
            pass

        sw = 0
        sh = 0
        try:
            with Image.open(screenshot_path) as im:
                sw, sh = im.size
        except Exception:
            sw, sh = 0, 0

        roi_notice = None
        try:
            if sw > 0 and sh > 0:
                roi_notice = (int(round(float(sw) * 0.72)), int(round(float(sh) * 0.00)), int(sw), int(round(float(sh) * 0.26)))
        except Exception:
            roi_notice = None

        def _uniq(names: list[str]) -> list[str]:
            out: list[str] = []
            seen: set[str] = set()
            for n in names:
                nn = str(n or "").strip()
                if not nn or nn in seen:
                    continue
                out.append(nn)
                seen.add(nn)
            return out

        # --- try closing X button via template matching (priority: close popup first) ---
        try:
            tmpl0 = str(getattr(self.cfg, "cerebellum_template_notice_close", "notice_close.png") or "")
            close_templates = [
                ("公告叉叉.png", 0.98),
                (tmpl0, 0.975),
                ("内嵌公告的叉.png", 0.975),
                ("游戏内很多页面窗口的叉.png", 0.975),
            ]
            seen_tmpls: set[str] = set()
            for tmpl, min_conf in close_templates:
                tmpl = str(tmpl or "").strip()
                if not tmpl or tmpl in seen_tmpls:
                    continue
                seen_tmpls.add(tmpl)
                act = c.click_action(
                    screenshot_path=screenshot_path,
                    template_name=tmpl,
                    reason_prefix="Cerebellum(delegate): close notice/webview.",
                    roi=roi_notice,
                )
                if isinstance(act, dict):
                    try:
                        cb = act.get("_cerebellum", {})
                        if float(cb.get("score") or 0.0) < float(min_conf):
                            continue
                    except Exception:
                        pass

                    act["raw"] = action.get("raw")
                    act["_prompt"] = action.get("_prompt")
                    act["_perception"] = action.get("_perception")
                    act["_model"] = action.get("_model")
                    act["_routine"] = action.get("_routine")
                    act["_close_heuristic"] = "cerebellum_notice_close"
                    act.setdefault("_delegate", {})
                    act["_delegate"]["from_action"] = str(action.get("action") or "")
                    act["_delegate"]["from_reason"] = str(reason)
                    act["_delegate"]["wants_delegate"] = bool(wants_delegate)

                    try:
                        if int(getattr(self, "_startup_tap_attempts", 0) or 0) > 0:
                            self._startup_finished = True
                    except Exception:
                        pass

                    try:
                        prev = int(getattr(self, "_last_cerebellum_notice_step", -10_000) or -10_000)
                        if int(step_id) <= int(prev) + 4:
                            self._cerebellum_notice_streak = int(getattr(self, "_cerebellum_notice_streak", 0) or 0) + 1
                        else:
                            self._cerebellum_notice_streak = 1
                        self._last_cerebellum_notice_step = int(step_id)
                        self._notice_close_total_attempts = int(getattr(self, "_notice_close_total_attempts", 0) or 0) + 1
                    except Exception:
                        pass
                    return act
        except Exception:
            pass

        try:
            prev = int(getattr(self, "_last_cerebellum_notice_step", -10_000) or -10_000)
            if int(step_id) <= int(prev) + 4:
                self._cerebellum_notice_streak = int(getattr(self, "_cerebellum_notice_streak", 0) or 0) + 1
            else:
                self._cerebellum_notice_streak = 1
            self._last_cerebellum_notice_step = int(step_id)
            self._notice_close_total_attempts = int(getattr(self, "_notice_close_total_attempts", 0) or 0) + 1
        except Exception:
            pass

        # --- try "今日不再提示" checkbox once per streak (only if X not found) ---
        try:
            streak = int(getattr(self, "_cerebellum_notice_streak", 0) or 0)
            if streak <= 1 and sw > 0 and sh > 0:
                roi_dismiss = (int(round(float(sw) * 0.10)), int(round(float(sh) * 0.55)), int(round(float(sw) * 0.50)), int(round(float(sh) * 0.80)))
                dismiss_act = c.click_action(
                    screenshot_path=screenshot_path,
                    template_name="今日不再提示（点完今日不再有这个弹窗）.png",
                    reason_prefix="Cerebellum(delegate): dismiss today notice.",
                    roi=roi_dismiss,
                )
                if isinstance(dismiss_act, dict):
                    cb = dismiss_act.get("_cerebellum", {})
                    if float(cb.get("score") or 0.0) >= 0.90:
                        dismiss_act["raw"] = action.get("raw")
                        dismiss_act["_prompt"] = action.get("_prompt")
                        dismiss_act["_perception"] = action.get("_perception")
                        dismiss_act["_model"] = action.get("_model")
                        dismiss_act["_routine"] = action.get("_routine")
                        dismiss_act["_close_heuristic"] = "cerebellum_dismiss_today"
                        return dismiss_act
        except Exception:
            pass

        if wants_delegate:
            return {
                "action": "back",
                "reason": "delegate: close_notice — Cerebellum template not matched; pressing ESC to close popup.",
                "raw": action.get("raw"),
                "_prompt": action.get("_prompt"),
                "_perception": action.get("_perception"),
                "_model": action.get("_model"),
                "_routine": action.get("_routine"),
                "_close_heuristic": "cerebellum_notice_esc_fallback",
            }
        return action

    def _block_startup_vlm_clicks(self, action: Dict[str, Any], *, step_id: int, screenshot_path: str) -> Dict[str, Any]:
        if not isinstance(action, dict):
            return action

        try:
            if str(action.get("action") or "").lower().strip() != "click":
                return action
        except Exception:
            return action

        try:
            if bool(action.get("_startup_tap")):
                return action
        except Exception:
            pass
        try:
            if action.get("_close_heuristic") is not None:
                return action
        except Exception:
            pass
        try:
            if isinstance(action.get("_cerebellum"), dict):
                return action
        except Exception:
            pass

        boot_preinteractive = False
        try:
            boot_preinteractive = (not bool(getattr(self, "_startup_finished", False))) and int(getattr(self, "_startup_tap_attempts", 0) or 0) <= 0
        except Exception:
            boot_preinteractive = False

        w = h = 0
        try:
            with Image.open(screenshot_path) as im:
                w, h = im.size
        except Exception:
            w, h = 0, 0

        try:
            tgt = action.get("target")
        except Exception:
            tgt = None

        if boot_preinteractive and int(step_id) <= 60:
            if isinstance(tgt, (list, tuple)) and len(tgt) == 2 and w > 0 and h > 0:
                try:
                    x, y = int(tgt[0]), int(tgt[1])
                    ex = int(round(float(w) * 0.50))
                    ey = int(round(float(h) * 0.82))
                    near_bottom = int(y) >= int(round(float(h) * 0.68))
                    ok_x = abs(int(x) - int(ex)) <= int(round(float(w) * 0.20))
                    ok_y = abs(int(y) - int(ey)) <= int(round(float(h) * 0.12))
                    if near_bottom and ok_x and ok_y:
                        out = dict(action)
                        out["target"] = [int(ex), int(ey)]
                        out["_startup_tap"] = True
                        out["reason"] = "Startup: forcing tap-to-start click at bottom-center."
                        return out
                except Exception:
                    pass

            return {
                "action": "wait",
                "duration_ms": 900,
                "reason": "Startup: blocked VLM coordinate click before tap-to-start is confirmed.",
                "raw": action.get("raw"),
                "_prompt": action.get("_prompt"),
                "_perception": action.get("_perception"),
                "_model": action.get("_model"),
                "_routine": action.get("_routine"),
                "_blocked": True,
                "_blocked_reason": "startup_preinteractive_click",
            }

        try:
            if bool(getattr(self, "_startup_finished", False)):
                return action
        except Exception:
            pass

        if not (isinstance(tgt, (list, tuple)) and len(tgt) == 2 and w > 0 and h > 0):
            return {
                "action": "wait",
                "duration_ms": 900,
                "reason": "Startup: blocked ambiguous click (missing target).",
                "raw": action.get("raw"),
                "_prompt": action.get("_prompt"),
                "_perception": action.get("_perception"),
                "_model": action.get("_model"),
                "_routine": action.get("_routine"),
                "_blocked": True,
                "_blocked_reason": "startup_missing_target",
            }

        try:
            x, y = int(tgt[0]), int(tgt[1])
            ex = int(round(float(w) * 0.50))
            ey = int(round(float(h) * 0.82))
            near_bottom = int(y) >= int(round(float(h) * 0.68))
            ok_x = abs(int(x) - int(ex)) <= int(round(float(w) * 0.20))
            ok_y = abs(int(y) - int(ey)) <= int(round(float(h) * 0.12))
            if near_bottom and ok_x and ok_y:
                out = dict(action)
                out["target"] = [int(ex), int(ey)]
                out["_startup_tap"] = True
                out["reason"] = "Startup: forcing tap-to-start click at bottom-center."
                return out
        except Exception:
            pass

        return {
            "action": "wait",
            "duration_ms": 900,
            "reason": "Startup: blocked VLM coordinate click outside bottom-center.",
            "raw": action.get("raw"),
            "_prompt": action.get("_prompt"),
            "_perception": action.get("_perception"),
            "_model": action.get("_model"),
            "_routine": action.get("_routine"),
            "_blocked": True,
            "_blocked_reason": "startup_outside_bottom_center",
        }

    def _rescale_action(self, action: Dict[str, Any], *, sx: float, sy: float) -> Dict[str, Any]:
        if not isinstance(action, dict):
            return action

        def _sc_xy(x: int, y: int) -> list[int]:
            return [int(round(float(x) * sx)), int(round(float(y) * sy))]

        def _sc_bbox(bb: list[int]) -> list[int]:
            if len(bb) != 4:
                return bb
            x1, y1, x2, y2 = [int(v) for v in bb]
            p1 = _sc_xy(x1, y1)
            p2 = _sc_xy(x2, y2)
            return [int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])]

        try:
            tgt = action.get("target")
            if isinstance(tgt, (list, tuple)) and len(tgt) == 2:
                action["target"] = _sc_xy(int(tgt[0]), int(tgt[1]))
        except Exception:
            pass

        try:
            bb = action.get("bbox")
            if isinstance(bb, (list, tuple)) and len(bb) == 4:
                action["bbox"] = _sc_bbox([int(v) for v in bb])
        except Exception:
            pass

        try:
            p1 = action.get("from")
            p2 = action.get("to")
            if isinstance(p1, (list, tuple)) and len(p1) == 2:
                action["from"] = _sc_xy(int(p1[0]), int(p1[1]))
            if isinstance(p2, (list, tuple)) and len(p2) == 2:
                action["to"] = _sc_xy(int(p2[0]), int(p2[1]))
        except Exception:
            pass

        try:
            p = action.get("_perception")
            if isinstance(p, dict) and isinstance(p.get("items"), list):
                for it in p.get("items"):
                    if not isinstance(it, dict):
                        continue
                    bb = it.get("bbox")
                    if isinstance(bb, (list, tuple)) and len(bb) == 4:
                        it["bbox"] = _sc_bbox([int(v) for v in bb])
        except Exception:
            pass

        return action

    def _sanitize_action(self, act: Any) -> Dict[str, Any]:
        if not isinstance(act, dict):
            return {
                "action": "wait",
                "duration_ms": 900,
                "reason": "Invalid action type; waiting.",
            }

        out: Dict[str, Any] = dict(act)
        a = str(out.get("action") or "").lower().strip()
        if a in ("", "none", "null"):
            a = "wait"
        if a == "tap":
            a = "click"
        if a in ("esc", "escape"):
            a = "back"
        if a in ("quit", "exit"):
            a = "stop"
        out["action"] = a

        if a == "wait":
            try:
                out["duration_ms"] = int(out.get("duration_ms") or 800)
            except Exception:
                out["duration_ms"] = 800

        if a == "click":
            tgt = out.get("target")
            bb = out.get("bbox")
            if not (
                isinstance(tgt, (list, tuple))
                and len(tgt) == 2
                or isinstance(bb, (list, tuple))
                and len(bb) == 4
            ):
                return {
                    "action": "wait",
                    "duration_ms": 900,
                    "reason": "click action missing target/bbox; waiting.",
                    "raw": out.get("raw"),
                    "_prompt": out.get("_prompt"),
                    "_perception": out.get("_perception"),
                    "_model": out.get("_model"),
                    "_routine": out.get("_routine"),
                }
            if isinstance(tgt, (list, tuple)) and len(tgt) == 2:
                try:
                    out["target"] = [int(tgt[0]), int(tgt[1])]
                except Exception:
                    pass
            if isinstance(bb, (list, tuple)) and len(bb) == 4:
                try:
                    out["bbox"] = [int(v) for v in bb]
                except Exception:
                    pass

        if a == "swipe":
            p1 = out.get("from")
            p2 = out.get("to")
            if not (
                isinstance(p1, (list, tuple))
                and isinstance(p2, (list, tuple))
                and len(p1) == 2
                and len(p2) == 2
            ):
                out["action"] = "wait"
                out["duration_ms"] = 900
                out["reason"] = "swipe action missing from/to; waiting."
            else:
                out["from"] = [int(p1[0]), int(p1[1])]
                out["to"] = [int(p2[0]), int(p2[1])]
                try:
                    out["duration_ms"] = int(out.get("duration_ms") or 500)
                except Exception:
                    out["duration_ms"] = 500

        if "reason" not in out or not str(out.get("reason") or "").strip():
            out["reason"] = f"Action: {a}"
        return out

    def _format_items(self, items: Any) -> str:
        if not isinstance(items, list) or not items:
            return ""
        lines: List[str] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            lbl = str(it.get("label") or "").strip()
            bb = it.get("bbox")
            if not lbl or not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                continue
            try:
                x1, y1, x2, y2 = [int(v) for v in bb]
            except Exception:
                continue
            lines.append(f"[{x1},{y1},{x2},{y2}] {lbl}")
            if len(lines) >= 60:
                break
        return "\n".join(lines)

    def _routine_meta(self) -> Dict[str, Any]:
        r = self._routine
        try:
            active = bool(getattr(r, "is_active", False))
        except Exception:
            active = False

        step = None
        if active:
            try:
                step = r.get_current_step()
            except Exception:
                step = None

        step_name = ""
        if step is not None:
            try:
                step_name = str(getattr(step, "name", "") or "")
            except Exception:
                step_name = ""

        total_steps = 0
        try:
            total_steps = len(getattr(r, "steps", []) or [])
        except Exception:
            total_steps = 0

        try:
            step_index = int(getattr(r, "current_step_index", 0) or 0)
        except Exception:
            step_index = 0

        try:
            max_turns = int(getattr(r, "max_turns_per_attempt", 0) or 0)
        except Exception:
            max_turns = 0
        try:
            turns = int(getattr(r, "turns_in_current_attempt", 0) or 0)
        except Exception:
            turns = 0
        try:
            rc = int(getattr(r, "current_step_retry_count", 0) or 0)
        except Exception:
            rc = 0
        try:
            mr = int(getattr(r, "max_retries", 0) or 0)
        except Exception:
            mr = 0
        try:
            recovery = bool(getattr(r, "in_recovery_mode", False))
        except Exception:
            recovery = False

        return {
            "active": bool(active),
            "step_name": step_name,
            "step_index": int(step_index),
            "total_steps": int(total_steps),
            "max_turns_per_attempt": int(max_turns),
            "turns_in_current_attempt": int(turns),
            "current_step_retry_count": int(rc),
            "max_retries": int(mr),
            "in_recovery_mode": bool(recovery),
        }

    def _perception_prompt(self, *, width: int, height: int) -> str:
        try:
            max_items = int(getattr(self.cfg, "perception_max_items", 40) or 40)
        except Exception:
            max_items = 40
        return (
            "You are a UI OCR system. Extract clickable UI texts with their bounding boxes. "
            "Return ONLY valid JSON with the following schema: "
            "{\"items\":[{\"label\":string,\"bbox\":[x1,y1,x2,y2]}]}. "
            f"The image size is {int(width)}x{int(height)} pixels. "
            f"Limit to at most {int(max_items)} items. "
            "Include close buttons like X/×/Close/关闭, page indicators like 1/2, and primary buttons like OK/Confirm/Skip/Next." 
        )

    def _prompt(self, *, width: int, height: int) -> str:
        routine_block = ""
        try:
            routine_block = str(self._routine.get_prompt_block() or "")
        except Exception:
            routine_block = ""

        forbid = bool(getattr(self.cfg, "forbid_premium_currency", True))
        safety = "Do NOT spend premium currency." if forbid else ""
        return (
            f"You are controlling a game via mouse/keyboard. Image size: {int(width)}x{int(height)}.\n"
            "Output ONLY a single JSON object (no markdown).\n"
            "IMPORTANT: You are a SUPERVISOR. Do NOT output click/back/swipe coordinates.\n"
            "Always output action='wait' unless you need to stop.\n"
            "To request a click via OpenCV (Cerebellum), put a delegate command in the reason field.\n"
            "Examples:\n"
            "- Close notices/webview: {action:'wait', duration_ms:200, reason:'delegate: close_notice'}\n"
            "- From Lobby, open Cafe: {action:'wait', duration_ms:200, reason:'delegate: open_cafe'}\n"
            "- Confirm/Cancel dialogs: {action:'wait', duration_ms:200, reason:'delegate: confirm'}\n"
            "Supported delegates: close_notice, open_cafe, open_schedule, open_mail, open_social, open_craft, confirm, cancel, touch_head.\n"
            "Valid actions: wait, stop, done.\n"
            f"Safety: {safety}\n"
            + routine_block
        )

    def _maybe_delegate_intent_to_cerebellum(self, action: Dict[str, Any], *, screenshot_path: str, step_id: int) -> Dict[str, Any]:
        if not isinstance(action, dict):
            return action
        try:
            if action.get("_close_heuristic") is not None:
                return action
        except Exception:
            pass

        c = getattr(self, "_cerebellum", None)
        if c is None:
            return action

        try:
            if not bool(getattr(self.cfg, "cerebellum_enabled", True)):
                return action
        except Exception:
            return action

        reason = str(action.get("reason") or "")
        raw = str(action.get("raw") or "")
        blob = (reason + "\n" + raw)
        blob_low = (blob or "").lower()

        key = ""
        try:
            m = re.search(r"delegate\s*:\s*([a-zA-Z0-9_]+)", blob_low)
            if m is not None:
                key = str(m.group(1) or "").strip().lower()
        except Exception:
            key = ""
        if not key:
            return action

        alias = {
            "open_club": "open_social",
            "open_guild": "open_social",
            "open_manufacture": "open_craft",
            "open_factory": "open_craft",
            "open_crafting": "open_craft",
            "open_tasks": "open_mail",
            "open_mailbox": "open_mail",
            "touch_head": "touch_head",
        }
        key = str(alias.get(key, key) or key)

        supported = {
            "open_cafe",
            "open_schedule",
            "open_mail",
            "open_social",
            "open_craft",
            "confirm",
            "cancel",
            "touch_head",
        }
        if key not in supported:
            return action

        try:
            cd = 2
            if int(step_id) - int(getattr(self, "_last_delegate_step", -10_000)) <= int(cd) and str(getattr(self, "_last_delegate_key", "") or "") == str(key):
                return action
        except Exception:
            pass

        sw = sh = 0
        try:
            with Image.open(screenshot_path) as im:
                sw, sh = im.size
        except Exception:
            sw, sh = 0, 0
        if sw <= 0 or sh <= 0:
            return action

        roi = None
        try:
            if key.startswith("open_"):
                try:
                    items = action.get("_perception", {}).get("items")
                except Exception:
                    items = None
                if not self._is_lobby_view(items, width=int(sw), height=int(sh)):
                    return action
                roi = (int(0), int(round(float(sh) * 0.76)), int(sw), int(sh))
            elif key in ("confirm", "cancel"):
                roi = (int(round(float(sw) * 0.18)), int(round(float(sh) * 0.34)), int(round(float(sw) * 0.82)), int(round(float(sh) * 0.96)))
            elif key == "touch_head":
                try:
                    items = action.get("_perception", {}).get("items")
                except Exception:
                    items = None
                if self._is_lobby_view(items, width=int(sw), height=int(sh)):
                    return action
                if not self._is_cafe_interior(items, width=int(sw), height=int(sh)):
                    return action
                roi = (int(0), int(0), int(sw), int(round(float(sh) * 0.85)))
        except Exception:
            roi = None

        def _uniq(names: list[str]) -> list[str]:
            out: list[str] = []
            seen: set[str] = set()
            for n in names:
                nn = str(n or "").strip()
                if not nn or nn in seen:
                    continue
                out.append(nn)
                seen.add(nn)
            return out

        templates: list[str] = []
        if key == "open_cafe":
            templates = ["咖啡厅.png"]
        elif key == "open_schedule":
            templates = ["课程表.png"]
        elif key == "open_mail":
            templates = ["邮箱.png"]
        elif key == "open_social":
            templates = ["社交.png"]
        elif key == "open_craft":
            templates = ["制造.png"]
        elif key == "confirm":
            templates = ["确认(可以点space）.png"]
        elif key == "cancel":
            templates = ["取消（可点Esc）.png"]
        elif key == "touch_head":
            templates = ["可摸头的标志.png"]

        try:
            for tmpl in _uniq(list(templates)):
                act = c.click_action(
                    screenshot_path=screenshot_path,
                    template_name=tmpl,
                    reason_prefix=f"Cerebellum(delegate): {key}.",
                    roi=roi,
                )
                if isinstance(act, dict):
                    act["raw"] = action.get("raw")
                    act["_prompt"] = action.get("_prompt")
                    act["_perception"] = action.get("_perception")
                    act["_model"] = action.get("_model")
                    act["_routine"] = action.get("_routine")
                    act.setdefault("_delegate", {})
                    act["_delegate"]["type"] = str(key)
                    act["_delegate"]["from_action"] = str(action.get("action") or "")
                    act["_delegate"]["from_reason"] = str(reason)
                    try:
                        self._last_delegate_step = int(step_id)
                        self._last_delegate_key = str(key)
                    except Exception:
                        pass

                    if key == "touch_head":
                        try:
                            cb = act.get("_cerebellum", {})
                            ctr = cb.get("center")
                            if isinstance(ctr, (list, tuple)) and len(ctr) == 2:
                                cx, cy = int(ctr[0]), int(ctr[1])
                                act["target"] = [int(cx + 24), int(cy + 34)]
                        except Exception:
                            pass
                    return act
        except Exception:
            pass

        if key == "touch_head":
            try:
                marks = self._detect_yellow_markers(screenshot_path=screenshot_path)
            except Exception:
                marks = []
            if marks:
                chosen = marks[0]
                x1, y1, x2, y2 = [int(v) for v in chosen]
                cx, cy = _center([x1, y1, x2, y2])
                bw = max(1, x2 - x1)
                bh = max(1, y2 - y1)
                tx = int(cx + max(18, int(round(bw * 0.6))))
                ty = int(cy + max(18, int(round(bh * 0.9))))
                try:
                    self._last_delegate_step = int(step_id)
                    self._last_delegate_key = str(key)
                except Exception:
                    pass
                return {
                    "action": "click",
                    "target": [int(tx), int(ty)],
                    "reason": "Cerebellum(delegate): touch_head. yellow interaction marker detected; clicking slightly right/below to interact.",
                    "raw": action.get("raw"),
                    "_prompt": action.get("_prompt"),
                    "_perception": action.get("_perception"),
                    "_model": action.get("_model"),
                    "_routine": action.get("_routine"),
                    "_delegate": {"type": "touch_head", "from_action": str(action.get("action") or ""), "from_reason": str(reason)},
                }

        return action

    def _resolve_image_size(self, items: Any, *, meta: Any = None) -> Tuple[int, int]:
        try:
            if isinstance(meta, dict):
                sz = meta.get("image_size")
                if isinstance(sz, (list, tuple)) and len(sz) == 2:
                    w, h = int(sz[0]), int(sz[1])
                    if w > 0 and h > 0:
                        return int(w), int(h)
        except Exception:
            pass

        mx, my = 0, 0
        if isinstance(items, list):
            for it in items:
                if not isinstance(it, dict):
                    continue
                bb = it.get("bbox")
                if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                    continue
                try:
                    x1, y1, x2, y2 = [int(v) for v in bb]
                except Exception:
                    continue
                mx = max(mx, x2, x1)
                my = max(my, y2, y1)
        return int(mx), int(my)

    def _is_lobby_view(self, items: Any, *, width: int, height: int) -> bool:
        if not isinstance(items, list) or not items:
            return False
        hit = 0
        for it in items:
            if not isinstance(it, dict):
                continue
            lbl = str(it.get("label") or "").strip()
            if not lbl:
                continue
            ll = lbl.lower()
            if "cafe" in ll or "咖啡" in lbl or "咖啡廳" in lbl:
                hit += 1
            if "schale" in ll or "夏莱" in lbl or "夏萊" in lbl:
                hit += 1
            if "campaign" in ll or "mission" in ll or "任务" in lbl or "任務" in lbl:
                hit += 1
            if hit >= 2:
                return True
        return False

    def _is_cafe_interior(self, items: Any, *, width: int, height: int) -> bool:
        if not isinstance(items, list) or not items:
            return False
        hit = 0
        for it in items:
            if not isinstance(it, dict):
                continue
            lbl = str(it.get("label") or "").strip()
            if not lbl:
                continue
            ll = lbl.lower()
            if "cafe" in ll or "咖啡" in lbl or "咖啡廳" in lbl:
                hit += 1
            if "earn" in ll or "earning" in ll or "claim" in ll or "收益" in lbl or "领取" in lbl or "領取" in lbl:
                hit += 1
            if "invite" in ll or "邀請" in lbl or "邀请" in lbl:
                hit += 1
            if hit >= 2:
                return True
        return False

    def _has_cafe_claim_ui(self, items: Any, *, height: int) -> bool:
        if not isinstance(items, list) or not items:
            return False
        for it in items:
            if not isinstance(it, dict):
                continue
            lbl = str(it.get("label") or "").strip()
            if not lbl:
                continue
            ll = lbl.lower()
            if "earn" in ll or "earning" in ll or "claim" in ll or "收益" in lbl or "收取" in lbl or "领取" in lbl or "領取" in lbl:
                return True
        return False

    def _has_cafe_control_ui(self, items: Any, *, height: int) -> bool:
        if not isinstance(items, list) or not items:
            return False
        for it in items:
            if not isinstance(it, dict):
                continue
            lbl = str(it.get("label") or "").strip()
            if not lbl:
                continue
            ll = lbl.lower()
            if "invite" in ll or "邀請" in lbl or "邀请" in lbl or "store" in ll or "shop" in ll:
                return True
        return False

    def _detect_close_x_roi(self, *, screenshot_path: str, step_id: int) -> Optional[List[int]]:
        try:
            with Image.open(screenshot_path) as im:
                w, h = im.size
                if w <= 0 or h <= 0:
                    return None
                x0 = int(round(float(w) * 0.75))
                y0 = 0
                x1 = int(w)
                y1 = int(round(float(h) * 0.18))
                if x1 <= x0 or y1 <= y0:
                    return None
                crop = im.crop((x0, y0, x1, y1)).convert("RGB")
        except Exception:
            return None

        scale = 2
        try:
            cw, ch = crop.size
            crop2 = crop.resize((max(1, int(cw) * int(scale)), max(1, int(ch) * int(scale))), resample=Image.BILINEAR)
        except Exception:
            crop2 = crop
            scale = 1

        tmp = None
        try:
            tmp = (self._run_dir / f"roi_close_{int(step_id):06d}.png")
            crop2.save(tmp)
        except Exception:
            tmp = None

        try:
            if tmp is None:
                return None
            engine = get_local_vlm(
                model=self.cfg.model,
                models_dir=self.cfg.models_dir,
                hf_home=self.cfg.hf_home,
                device=self.cfg.device,
            )
            prompt = (
                "This is a zoomed crop of the TOP-RIGHT corner of a game window. "
                "Find the close button (X or ×). "
                "Return ONLY JSON: {\"items\":[{\"label\":string,\"bbox\":[x1,y1,x2,y2]}]}. "
                "Only include close/X candidates."
            )
            res = engine.ocr(image_path=str(tmp), prompt=prompt, max_new_tokens=128)
            raw = str(res.get("raw") or "")
            parsed: Dict[str, Any] = {}
            try:
                parsed = _parse_json_content(raw)
            except Exception:
                parsed = {}
            items: List[Dict[str, Any]] = []
            if isinstance(parsed, dict) and isinstance(parsed.get("items"), list):
                items = [it for it in parsed.get("items") if isinstance(it, dict)]
            if not items and raw:
                try:
                    items = _parse_perception_items_fallback(raw)
                except Exception:
                    items = []

            best = None
            for it in items:
                lbl = str(it.get("label") or "").strip()
                bb = it.get("bbox")
                if not lbl or not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                    continue
                ll = lbl.lower()
                if not (ll in ("x", "×") or "close" in ll or "×" in lbl or "关闭" in lbl):
                    continue
                try:
                    bb2 = [int(v) for v in bb]
                except Exception:
                    continue
                if best is None:
                    best = bb2
                    continue
                if bb2[2] > best[2] or (bb2[2] == best[2] and bb2[1] < best[1]):
                    best = bb2

            if best is None:
                return None

            bx1, by1, bx2, by2 = [int(v) for v in best]
            if scale > 1:
                bx1 = int(round(float(bx1) / float(scale)))
                by1 = int(round(float(by1) / float(scale)))
                bx2 = int(round(float(bx2) / float(scale)))
                by2 = int(round(float(by2) / float(scale)))
            bx1 += int(x0)
            bx2 += int(x0)
            by1 += int(y0)
            by2 += int(y0)
            bx1 = max(0, min(int(w) - 1, int(bx1)))
            by1 = max(0, min(int(h) - 1, int(by1)))
            bx2 = max(0, min(int(w), int(bx2)))
            by2 = max(0, min(int(h), int(by2)))
            if bx2 <= bx1 or by2 <= by1:
                return None
            return [int(bx1), int(by1), int(bx2), int(by2)]
        except Exception:
            return None
        finally:
            try:
                if tmp is not None:
                    tmp.unlink(missing_ok=True)
            except Exception:
                pass

    def _snap_click_to_perception_label(self, action: Dict[str, Any]) -> Dict[str, Any]:
        # Currently a no-op: keep for backwards-compatibility with older pipelines.
        return action

    def _maybe_force_cafe_nav(self, action: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(action, dict):
            return action
        try:
            if not bool(getattr(self._routine, "is_active", False)):
                return action
            step = self._routine.get_current_step()
            step_name = str(getattr(step, "name", "") or "") if step is not None else ""
        except Exception:
            return action

        nav_map = {
            "Cafe": ("cafe", "咖啡", "咖啡廳"),
            "Schedule": ("schedule", "日程"),
            "Club": ("club", "社团", "社團"),
            "Bounties": ("bounty", "wanted", "悬赏", "懸賞", "通缉", "通緝"),
            "Mail & Tasks": ("mail", "mailbox", "task", "tasks", "邮箱", "郵箱", "邮件", "郵件", "任务", "任務"),
        }
        if step_name not in nav_map:
            return action

        try:
            items = action.get("_perception", {}).get("items")
        except Exception:
            items = None
        if not isinstance(items, list) or not items:
            return action

        iw, ih = self._resolve_image_size(items, meta=action.get("_perception"))
        if iw <= 0 or ih <= 0:
            return action

        # Only force nav when we are clearly in lobby view.
        if not self._is_lobby_view(items, width=iw, height=ih):
            try:
                self._nav_force_miss_step_name = ""
                self._nav_force_miss_count = 0
            except Exception:
                pass
            return action

        try:
            if action.get("_delegate") is not None:
                self._nav_force_miss_step_name = ""
                self._nav_force_miss_count = 0
                return action
        except Exception:
            pass

        try:
            a0 = str(action.get("action") or "").lower().strip()
            if a0 == "wait":
                if str(self._nav_force_miss_step_name or "") != str(step_name):
                    self._nav_force_miss_step_name = str(step_name)
                    self._nav_force_miss_count = 0
                self._nav_force_miss_count = int(self._nav_force_miss_count) + 1
                if int(self._nav_force_miss_count) < 3:
                    return action
            else:
                self._nav_force_miss_step_name = ""
                self._nav_force_miss_count = 0
                return action
        except Exception:
            pass

        # Don't interfere with popup closing.
        if action.get("_close_heuristic") is not None:
            return action

        kws = nav_map.get(step_name, ())
        best_bb = None
        best_score = -1
        for it in items:
            if not isinstance(it, dict):
                continue
            lbl = str(it.get("label") or "").strip()
            bb = it.get("bbox")
            if not lbl or not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                continue
            ll = lbl.lower()
            score = 0
            for k in kws:
                if any(ord(c) > 127 for c in k):
                    if k in lbl:
                        score += 3
                else:
                    if k in ll:
                        score += 2
            if score <= 0:
                continue
            try:
                bb2 = [int(v) for v in bb]
            except Exception:
                continue
            if score > best_score:
                best_score = score
                best_bb = bb2

        if best_bb is None:
            return action
        x, y = _center(best_bb)
        try:
            self._nav_force_miss_step_name = ""
            self._nav_force_miss_count = 0
        except Exception:
            pass

        key = ""
        try:
            if step_name == "Cafe":
                key = "open_cafe"
            elif step_name == "Schedule":
                key = "open_schedule"
            elif step_name == "Club":
                key = "open_social"
            elif step_name == "Mail & Tasks":
                key = "open_mail"
        except Exception:
            key = ""
        try:
            c = getattr(self, "_cerebellum", None)
            if key and c is not None and bool(getattr(self.cfg, "cerebellum_enabled", True)):
                return {
                    "action": "wait",
                    "duration_ms": 200,
                    "reason": f"delegate: {key}",
                    "raw": action.get("raw"),
                    "_prompt": action.get("_prompt"),
                    "_perception": action.get("_perception"),
                    "_model": action.get("_model"),
                    "_routine": action.get("_routine"),
                    "_nav_force_delegate": True,
                    "_nav_force_delegate_key": str(key),
                }
        except Exception:
            pass
        return {
            "action": "click",
            "target": [int(x), int(y)],
            "reason": f"Routine navigation: clicking '{step_name}' entry in lobby.",
            "raw": action.get("raw"),
            "_prompt": action.get("_prompt"),
            "_perception": action.get("_perception"),
            "_model": action.get("_model"),
            "_routine": action.get("_routine"),
            "_cafe_nav_override": True,
        }

    def _maybe_cafe_actions(self, action: Dict[str, Any], *, screenshot_path: str, step_id: int) -> Dict[str, Any]:
        if not isinstance(action, dict):
            return action
        try:
            if not bool(getattr(self._routine, "is_active", False)):
                return action
            step = self._routine.get_current_step()
            if step is None or str(getattr(step, "name", "") or "") != "Cafe":
                return action
        except Exception:
            return action

        try:
            items = action.get("_perception", {}).get("items")
        except Exception:
            items = None
        if not isinstance(items, list) or not items:
            return action

        iw, ih = self._resolve_image_size(items, meta=action.get("_perception"))
        if iw <= 0 or ih <= 0:
            return action
        if not self._is_cafe_interior(items, width=iw, height=ih):
            return action

        try:
            if step_id - int(self._cafe_last_claim_step) <= 6:
                return action
        except Exception:
            pass

        claim_kws = ("claim", "earn", "earning", "收益", "收取", "领取", "領取")
        best_bb = None
        for it in items:
            if not isinstance(it, dict):
                continue
            lbl = str(it.get("label") or "").strip()
            bb = it.get("bbox")
            if not lbl or not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                continue
            ll = lbl.lower()
            if any(k in ll for k in ("claim", "earn", "earning")) or any(k in lbl for k in ("收益", "收取", "领取", "領取")):
                try:
                    best_bb = [int(v) for v in bb]
                    break
                except Exception:
                    best_bb = None
        if best_bb is None:
            return action

        x, y = _center(best_bb)
        try:
            self._cafe_last_claim_step = int(step_id)
        except Exception:
            pass
        return {
            "action": "click",
            "target": [int(x), int(y)],
            "reason": "Cafe: claim earnings button detected; clicking.",
            "raw": action.get("raw"),
            "_prompt": action.get("_prompt"),
            "_perception": action.get("_perception"),
            "_model": action.get("_model"),
            "_routine": action.get("_routine"),
            "_cafe_action": True,
        }

    def _maybe_autoclick_safe_button(
        self, *, action: Dict[str, Any], action_before: Optional[Dict[str, Any]], step_id: int, screenshot_path: str
    ) -> Dict[str, Any]:
        if not isinstance(action, dict):
            return action
        if not bool(getattr(self.cfg, "autoclick_safe_buttons", True)):
            return action
        if action.get("_close_heuristic") is not None:
            return action

        a = str(action.get("action") or "").lower().strip()
        if a == "click":
            return action

        try:
            cd = int(getattr(self.cfg, "autoclick_safe_cooldown_steps", 2) or 2)
        except Exception:
            cd = 2
        try:
            if step_id - int(self._last_autoclick_step) <= int(cd):
                return action
        except Exception:
            pass

        try:
            c = getattr(self, "_cerebellum", None)
        except Exception:
            c = None

        sw = sh = 0
        try:
            with Image.open(screenshot_path) as im:
                sw, sh = im.size
        except Exception:
            sw, sh = 0, 0

        try:
            if c is not None and bool(getattr(self.cfg, "cerebellum_enabled", True)) and sw > 0 and sh > 0:
                roi = (
                    int(round(float(sw) * 0.18)),
                    int(round(float(sh) * 0.34)),
                    int(round(float(sw) * 0.82)),
                    int(round(float(sh) * 0.96)),
                )
                for tmpl, key in (("确认(可以点space）.png", "confirm"), ("取消（可点Esc）.png", "cancel")):
                    act2 = c.click_action(
                        screenshot_path=screenshot_path,
                        template_name=tmpl,
                        reason_prefix=f"Cerebellum(autoclick): {key}.",
                        roi=roi,
                    )
                    if isinstance(act2, dict):
                        try:
                            cb = act2.get("_cerebellum", {})
                            if float(cb.get("score") or 0.0) < 0.98:
                                continue
                        except Exception:
                            pass
                        self._last_autoclick_step = int(step_id)
                        act2["raw"] = action.get("raw")
                        act2["_prompt"] = action.get("_prompt")
                        act2["_perception"] = action.get("_perception")
                        act2["_model"] = action.get("_model")
                        act2["_routine"] = action.get("_routine")
                        act2["_autoclick"] = True
                        act2.setdefault("_delegate", {})
                        act2["_delegate"]["type"] = str(key)
                        return act2
        except Exception:
            pass

        try:
            items = action.get("_perception", {}).get("items")
        except Exception:
            items = None
        if not isinstance(items, list) or not items:
            return action

        forbid = bool(getattr(self.cfg, "forbid_premium_currency", True))

        def _looks_premium(lbl: str) -> bool:
            s = (lbl or "").lower()
            if any(k in s for k in ("pyrox", "gem", "diamond", "purchase", "pay", "top up", "充值")):
                return True
            if any(k in (lbl or "") for k in ("青辉石", "青輝石", "钻石", "鑽石", "购买", "購買")):
                return True
            return False

        safe_kws_ascii = ("ok", "confirm", "yes", "next", "skip", "close", "cancel")
        safe_kws_zh = ("确认", "確認", "确定", "確定", "是", "下一步", "下一个", "下個", "跳过", "跳過", "关闭", "關閉", "取消")

        best_bb = None
        for it in items:
            if not isinstance(it, dict):
                continue
            lbl = str(it.get("label") or "").strip()
            bb = it.get("bbox")
            if not lbl or not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                continue
            ll = lbl.lower().strip()
            if forbid and _looks_premium(lbl):
                continue
            if (ll in ("x", "×")):
                continue
            if any(k in ll for k in safe_kws_ascii) or any(k in lbl for k in safe_kws_zh):
                try:
                    best_bb = [int(v) for v in bb]
                    break
                except Exception:
                    best_bb = None

        if best_bb is None:
            return action

        x, y = _center(best_bb)
        self._last_autoclick_step = int(step_id)
        return {
            "action": "click",
            "target": [int(x), int(y)],
            "reason": "Auto-click: detected a safe confirmation button.",
            "raw": action.get("raw"),
            "_prompt": action.get("_prompt"),
            "_perception": action.get("_perception"),
            "_model": action.get("_model"),
            "_routine": action.get("_routine"),
            "_autoclick": True,
        }

    def _maybe_exploration_click(self, *, action: Dict[str, Any], action_before: Optional[Dict[str, Any]], screenshot_path: str, step_id: int) -> Dict[str, Any]:
        if not isinstance(action, dict):
            return action
        if not bool(getattr(self.cfg, "exploration_click", False)):
            return action
        if action.get("_close_heuristic") is not None:
            return action
        if bool(action.get("_autoclick")):
            return action

        a = str(action.get("action") or "").lower().strip()
        if a != "wait":
            return action

        try:
            cd = int(getattr(self.cfg, "exploration_click_cooldown_steps", 3) or 3)
        except Exception:
            cd = 3
        try:
            if step_id - int(self._last_exploration_step) <= int(cd):
                return action
        except Exception:
            pass

        try:
            items = action.get("_perception", {}).get("items")
        except Exception:
            items = None
        if not isinstance(items, list) or not items:
            return action

        # Only click very safe progress buttons.
        progress_kws_ascii = ("skip", "next")
        progress_kws_zh = ("跳过", "跳過", "下一步", "下一个", "下個")
        best_bb = None
        for it in items:
            if not isinstance(it, dict):
                continue
            lbl = str(it.get("label") or "").strip()
            bb = it.get("bbox")
            if not lbl or not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                continue
            ll = lbl.lower()
            if any(k in ll for k in progress_kws_ascii) or any(k in lbl for k in progress_kws_zh):
                try:
                    best_bb = [int(v) for v in bb]
                    break
                except Exception:
                    best_bb = None

        if best_bb is None:
            return action
        x, y = _center(best_bb)
        self._last_exploration_step = int(step_id)
        return {
            "action": "click",
            "target": [int(x), int(y)],
            "reason": "Exploration: clicking a safe progress button (Skip/Next).",
            "raw": action.get("raw"),
            "_prompt": action.get("_prompt"),
            "_perception": action.get("_perception"),
            "_model": action.get("_model"),
            "_routine": action.get("_routine"),
            "_exploration": True,
        }

    def _maybe_advance_routine(self, action: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(action, dict):
            return action

        a = str(action.get("action") or "").lower().strip()
        if a != "done":
            return action

        advanced = False
        try:
            if bool(getattr(self._routine, "is_active", False)):
                advanced = bool(self._routine.handle_done_signal())
        except Exception:
            advanced = False

        # Convert 'done' to 'wait' because _execute() does not implement 'done'.
        return {
            "action": "wait",
            "duration_ms": 700,
            "reason": "Routine: received 'done' signal; advancing routine step." if advanced else "Routine: received 'done' but routine not active; waiting.",
            "raw": action.get("raw"),
            "_prompt": action.get("_prompt"),
            "_perception": action.get("_perception"),
            "_model": action.get("_model"),
            "_routine": action.get("_routine"),
            "_routine_advanced": bool(advanced),
        }

    def _maybe_tap_to_start(self, action: Dict[str, Any], *, step_id: int) -> Dict[str, Any]:
        if not isinstance(action, dict):
            return action

        try:
            if bool(action.get("_startup_tap")):
                return action
        except Exception:
            pass

        try:
            if action.get("raw") is None and action.get("_perception") is None and action.get("_model") is None:
                return action
        except Exception:
            pass

        blob = ""
        try:
            blob = str(action.get("reason") or "") + "\n" + str(action.get("raw") or "")
        except Exception:
            blob = ""
        blob_low = (blob or "").lower()
        kws_zh = ("点击开始", "點擊開始", "点按开始", "點按開始")
        mentions_tap = ("tap to start" in blob_low) or ("tap" in blob_low and "start" in blob_low) or any(k in blob for k in kws_zh)

        a0 = str(action.get("action") or "").lower().strip()
        if a0 not in ("click", "wait"):
            return action
        try:
            if action.get("_close_heuristic") is not None:
                return action
        except Exception:
            pass
        if ("close" in blob_low) or ("popup" in blob_low) or ("关闭" in blob) or ("弹窗" in blob):
            return action
        if mentions_tap and str(action.get("action") or "").lower().strip() == "click":
            try:
                if not bool(action.get("_startup_tap")):
                    action = dict(action)
                    action["_startup_tap"] = True
            except Exception:
                action = dict(action)
                action["_startup_tap"] = True

        iw = ih = 0
        try:
            sz = action.get("_perception", {}).get("image_size")
            if isinstance(sz, (list, tuple)) and len(sz) == 2:
                iw, ih = int(sz[0]), int(sz[1])
        except Exception:
            iw, ih = 0, 0

        def _emit_click(x: int, y: int, *, bbox: Optional[List[int]] = None, why: str) -> Dict[str, Any]:
            self._last_tap_to_start_step = int(step_id)
            self._last_tap_to_start_xy = (int(x), int(y))
            out = {
                "action": "click",
                "target": [int(x), int(y)],
                "reason": why,
                "raw": action.get("raw"),
                "_prompt": action.get("_prompt"),
                "_perception": action.get("_perception"),
                "_model": action.get("_model"),
                "_routine": action.get("_routine"),
                "_startup_tap": True,
            }
            if bbox is not None:
                out["bbox"] = [int(v) for v in bbox]
            return out

        try:
            cooldown = 1
            if step_id - int(self._last_tap_to_start_step) < max(0, int(cooldown)):
                return action
        except Exception:
            pass

        items = None
        try:
            items = action.get("_perception", {}).get("items")
        except Exception:
            items = None

        if _looks_like_title_screen(items) and iw > 0 and ih > 0:
            x = int(round(float(iw) * 0.50))
            y = int(round(float(ih) * 0.82))
            x = max(0, min(int(iw) - 1, int(x)))
            y = max(0, min(int(ih) - 1, int(y)))
            return _emit_click(int(x), int(y), why="Startup: title screen detected; clicking bottom-center to start.")

        if not isinstance(items, list) or not items:
            if mentions_tap and iw > 0 and ih > 0:
                x = int(round(float(iw) * 0.50))
                y = int(round(float(ih) * 0.82))
                x = max(0, min(int(iw) - 1, int(x)))
                y = max(0, min(int(ih) - 1, int(y)))
                return _emit_click(int(x), int(y), why="Startup: Tap-to-start mentioned; clicking bottom-center to start.")
            last_xy = self._last_tap_to_start_xy
            loading_hint = ("loading" in blob_low) or ("now loading" in blob_low) or ("加载" in blob) or ("載入" in blob)
            if last_xy is not None and (step_id - int(self._last_tap_to_start_step)) <= 3 and (mentions_tap or loading_hint):
                x, y = int(last_xy[0]), int(last_xy[1])
                self._last_tap_to_start_step = int(step_id)
                out = {
                    "action": "click",
                    "target": [int(x), int(y)],
                    "reason": "Startup: repeating Tap-to-Start click (no OCR items this step).",
                    "raw": action.get("raw"),
                    "_prompt": action.get("_prompt"),
                    "_perception": action.get("_perception"),
                    "_model": action.get("_model"),
                    "_routine": action.get("_routine"),
                    "_startup_tap": True,
                }
                return out
            return action

        saw_tap = False
        saw_start = False
        saw_candidate = False
        saw_exact = False
        for it in items:
            if not isinstance(it, dict):
                continue
            label = str(it.get("label") or "").strip()
            bb = it.get("bbox")
            if not label:
                continue
            ll = label.lower()

            if "tap" in ll:
                saw_tap = True
            if "start" in ll or any(k in label for k in kws_zh):
                saw_start = True

            if "tap to start" in ll or any(k in label for k in kws_zh):
                saw_exact = True
                saw_candidate = True
                continue

            if ("tap" in ll and "start" in ll) or ("tap" in ll and any(k in label for k in kws_zh)):
                saw_candidate = True
                continue

            if iw > 0 and ih > 0:
                try:
                    cx, cy = _center([int(v) for v in bb])
                    if cy >= float(ih) * 0.70 and ("tap" in ll or "start" in ll):
                        saw_candidate = True
                except Exception:
                    pass

        if iw > 0 and ih > 0 and ((saw_tap and saw_start) or saw_candidate):
            x = int(round(float(iw) * 0.50))
            y = int(round(float(ih) * 0.82))
            x = max(0, min(int(iw) - 1, int(x)))
            y = max(0, min(int(ih) - 1, int(y)))
            return _emit_click(int(x), int(y), why="Startup: Tap-to-start text detected; clicking bottom-center to start.")

        try:
            if self._last_tap_to_start_xy is not None and not mentions_tap and iw > 0 and ih > 0:
                self._last_tap_to_start_xy = None
                self._last_tap_to_start_step = -10_000
        except Exception:
            pass

        if mentions_tap and iw > 0 and ih > 0:
            x = int(round(float(iw) * 0.50))
            y = int(round(float(ih) * 0.82))
            x = max(0, min(int(iw) - 1, int(x)))
            y = max(0, min(int(ih) - 1, int(y)))
            return _emit_click(int(x), int(y), why="Startup: Tap-to-start mentioned; clicking bottom-center to start.")

        return action

    def _maybe_close_popup_heuristic(self, action: Dict[str, Any], *, step_id: int, screenshot_path: str) -> Dict[str, Any]:
        if not isinstance(action, dict):
            return action
        try:
            boot_preinteractive = (not bool(getattr(self, "_startup_finished", False))) and int(getattr(self, "_startup_tap_attempts", 0) or 0) <= 0
            if boot_preinteractive and int(step_id) <= 60:
                return action
        except Exception:
            pass
        try:
            if action.get("_close_heuristic"):
                return action
        except Exception:
            pass
        a = str(action.get("action") or "").lower().strip()

        reason = str(action.get("reason") or "")
        rlow = reason.lower()
        sw, sh = 0, 0
        im0 = None
        try:
            im0 = Image.open(screenshot_path).convert("RGB")
            sw, sh = im0.size
        except Exception:
            sw, sh = 0, 0
            im0 = None
        items = None
        try:
            items = action.get("_perception", {}).get("items")
        except Exception:
            items = None

        looks_like_webview_pixels = False
        try:
            if im0 is not None and sw > 0 and sh > 0 and not (isinstance(items, list) and items):
                x0 = int(round(float(sw) * 0.10))
                x1 = int(round(float(sw) * 0.95))
                y0 = int(round(float(sh) * 0.45))
                y1 = int(round(float(sh) * 0.95))
                if x1 > x0 and y1 > y0:
                    crop = im0.crop((x0, y0, x1, y1)).convert("L")
                    crop = crop.resize((160, 90), resample=Image.BILINEAR)
                    data = list(crop.getdata())
                    if data:
                        bright = 0
                        for px in data:
                            if int(px) >= 240:
                                bright += 1
                        looks_like_webview_pixels = (float(bright) / float(len(data))) >= 0.40
        except Exception:
            looks_like_webview_pixels = False

        page_bb = None
        try:
            if isinstance(items, list) and items and sw > 0 and sh > 0:
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    lbl = str(it.get("label") or "").strip()
                    bb = it.get("bbox")
                    if not lbl or not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                        continue
                    ll = lbl.lower()
                    if "/" not in ll:
                        continue
                    if not re.search(r"^\s*\d+\s*/\s*\d+\s*$", ll):
                        continue
                    bb2 = [int(v) for v in bb]
                    cx, cy = _center(bb2)
                    if int(cx) < int(float(sw) * 0.75) or int(cy) > int(float(sh) * 0.20):
                        continue
                    if page_bb is None or (bb2[0] > page_bb[0]):
                        page_bb = bb2
        except Exception:
            page_bb = None

        try:
            if isinstance(items, list) and items:
                if (not _looks_like_embedded_webview(items)) and (not _looks_like_notice_board(items)):
                    self._last_notice_x_xy = None
                    self._last_notice_x_step = -10_000
                    self._notice_x_attempts = 0
        except Exception:
            pass

        try:
            last_xy = self._last_notice_x_xy
            last_step = int(self._last_notice_x_step)
            if last_xy is not None and (0 <= (int(step_id) - last_step) <= 3):
                keep = True
                try:
                    if isinstance(items, list) and items:
                        keep = bool(_looks_like_embedded_webview(items) or _looks_like_notice_board(items))
                except Exception:
                    keep = True
                if keep and int(self._notice_x_attempts) <= 12 and not (isinstance(items, list) and items):
                    x0, y0 = int(last_xy[0]), int(last_xy[1])
                    try:
                        self._notice_x_attempts = int(self._notice_x_attempts) + 1
                    except Exception:
                        pass
                    return {
                        "action": "click",
                        "target": [int(x0), int(y0)],
                        "reason": "Close embedded announcement/webview via persistent X click.",
                        "raw": action.get("raw"),
                        "_prompt": action.get("_prompt"),
                        "_perception": action.get("_perception"),
                        "_model": action.get("_model"),
                        "_routine": action.get("_routine"),
                        "_close_heuristic": "notice_x_fallback",
                    }
        except Exception:
            pass

        if sw > 0 and sh > 0 and (_looks_like_embedded_webview(items) or page_bb is not None):
            try:
                c = getattr(self, "_cerebellum", None)
            except Exception:
                c = None

            try:
                if c is not None and bool(getattr(self.cfg, "cerebellum_enabled", True)):
                    roi_notice = (
                        int(round(float(sw) * 0.72)),
                        int(round(float(sh) * 0.00)),
                        int(sw),
                        int(round(float(sh) * 0.26)),
                    )

                    def _uniq(names: list[str]) -> list[str]:
                        out: list[str] = []
                        seen: set[str] = set()
                        for n in names:
                            nn = str(n or "").strip()
                            if not nn or nn in seen:
                                continue
                            out.append(nn)
                            seen.add(nn)
                        return out

                    tmpl0 = str(getattr(self.cfg, "cerebellum_template_notice_close", "notice_close.png") or "")
                    for tmpl in _uniq([tmpl0, "内嵌公告的叉.png", "游戏内很多页面窗口的叉.png"]):
                        act = c.click_action(
                            screenshot_path=screenshot_path,
                            template_name=tmpl,
                            reason_prefix="Cerebellum: close notice/webview.",
                            roi=roi_notice,
                        )
                        if isinstance(act, dict):
                            try:
                                cb = act.get("_cerebellum", {})
                                if float(cb.get("score") or 0.0) < 0.975:
                                    continue
                            except Exception:
                                pass
                            act["raw"] = action.get("raw")
                            act["_prompt"] = action.get("_prompt")
                            act["_perception"] = action.get("_perception")
                            act["_model"] = action.get("_model")
                            act["_routine"] = action.get("_routine")
                            act["_close_heuristic"] = "cerebellum_notice_close"

                            try:
                                if int(getattr(self, "_startup_tap_attempts", 0) or 0) > 0:
                                    self._startup_finished = True
                            except Exception:
                                pass
                            return act
            except Exception:
                pass

            best_bb = None
            try:
                if isinstance(items, list) and items:
                    for it in items:
                        if not isinstance(it, dict):
                            continue
                        lbl = str(it.get("label") or "").strip()
                        bb = it.get("bbox")
                        if not lbl or not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                            continue
                        ll = lbl.lower()
                        if not (ll in ("x", "×") or ("close" in ll) or ("关闭" in lbl) or ("×" in lbl)):
                            continue
                        bb2 = [int(v) for v in bb]
                        cx, cy = _center(bb2)
                        if int(cx) < int(float(sw) * 0.70) or int(cy) > int(float(sh) * 0.20):
                            continue
                        if best_bb is None or (bb2[0] > best_bb[0]) or (bb2[0] == best_bb[0] and bb2[1] < best_bb[1]):
                            best_bb = bb2
            except Exception:
                best_bb = None

            if best_bb is None:
                try:
                    bbx = self._detect_close_x_roi(screenshot_path=screenshot_path, step_id=int(step_id))
                except Exception:
                    bbx = None
                if isinstance(bbx, (list, tuple)) and len(bbx) == 4:
                    try:
                        best_bb = [int(v) for v in bbx]
                    except Exception:
                        best_bb = None

            if best_bb is not None:
                x, y = _center([int(v) for v in best_bb])
            else:
                if page_bb is not None:
                    pw = int(page_bb[2]) - int(page_bb[0])
                    ph = int(page_bb[3]) - int(page_bb[1])
                    dx = int(max(28, min(72, round(float(pw) * 0.60))))
                    dy = int(max(6, min(18, round(float(ph) * 0.25))))
                    x = int(page_bb[0]) - int(dx)
                    y = int(page_bb[3]) - int(dy)
                else:
                    x = int(round(float(sw) * 0.885))
                    y = int(round(float(sh) * 0.115))
            x = max(0, min(int(sw) - 1, int(x)))
            y = max(0, min(int(sh) - 1, int(y)))
            try:
                self._last_notice_x_xy = (int(x), int(y))
                self._last_notice_x_step = int(step_id)
                self._notice_x_attempts = 0
            except Exception:
                pass
            return {
                "action": "click",
                "target": [int(x), int(y)],
                "reason": "Close embedded announcement/webview via top-right X fallback.",
                "raw": action.get("raw"),
                "_prompt": action.get("_prompt"),
                "_perception": action.get("_perception"),
                "_model": action.get("_model"),
                "_routine": action.get("_routine"),
                "_close_heuristic": "notice_x_fallback",
            }

        if a != "click":
            return action

        if not ("关闭" in reason or "close" in rlow or "弹窗" in reason or "popup" in rlow):
            return action

        # Prefer closing by clicking an actual close/X button detected in perception.
        # Strict Trigger: Must imply closing/cancelling.
        if not (
            "关闭" in reason
            or "close" in rlow
            or "cancel" in rlow
            or "dismiss" in rlow
            or "取消" in reason
            or "关掉" in reason
        ):
            return action

        sw, sh = 0, 0
        try:
            with Image.open(screenshot_path) as im:
                sw, sh = im.size
        except Exception:
            sw, sh = 0, 0

        items = None
        try:
            items = action.get("_perception", {}).get("items")
        except Exception:
            items = None

        best_bb = None
        if isinstance(items, list) and items:
            for it in items:
                if not isinstance(it, dict):
                    continue
                label = str(it.get("label") or "").strip()
                bb = it.get("bbox")
                if not label or not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                    continue
                ll = label.lower()
                if (
                    ll in ("x", "×")
                    or "close" in ll
                    or "关闭" in label
                    or "关" == label
                    or "×" in label
                ):
                    bb2 = [int(v) for v in bb]
                    if sw > 0 and sh > 0:
                        cx, cy = _center(bb2)
                        if int(cx) < int(sw * 0.55) or int(cy) > int(sh * 0.45):
                            continue
                    best_bb = bb2
                    break

        if best_bb is not None:
            x, y = _center([int(v) for v in best_bb])
            out = dict(action)
            out["target"] = [int(x), int(y)]
            out["bbox"] = [int(v) for v in best_bb]
            out.setdefault("_close_heuristic", "perception_close")
            return out

        try:
            if isinstance(items, list) and items and (("通知" in reason) or ("notice" in rlow)):
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    lbl = str(it.get("label") or "").strip()
                    bb = it.get("bbox")
                    if not lbl or not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                        continue
                    ll = lbl.lower()
                    if ll in ("ok",) or ("确认" in lbl) or ("確認" in lbl) or ("确定" in lbl) or ("確定" in lbl):
                        x, y = _center([int(v) for v in bb])
                        out = dict(action)
                        out["target"] = [int(x), int(y)]
                        out["bbox"] = [int(v) for v in bb]
                        out.setdefault("_close_heuristic", "confirm")
                        return out
        except Exception:
            pass

        raw_txt = ""
        try:
            raw_txt = str(action.get("raw") or "")
        except Exception:
            raw_txt = ""
        raw_low = raw_txt.lower()
        is_notice = (
            ("公告" in reason)
            or ("announcement" in rlow)
            or ("公告" in raw_txt)
            or ("announcement" in raw_low)
        )
        tgt_tr = False
        try:
            tgt0 = action.get("target")
            if isinstance(tgt0, (list, tuple)) and len(tgt0) == 2 and sw > 0 and sh > 0:
                tx0, ty0 = int(tgt0[0]), int(tgt0[1])
                if tx0 >= int(float(sw) * 0.70) and ty0 <= int(float(sh) * 0.28):
                    tgt_tr = True
        except Exception:
            tgt_tr = False
        if sw > 0 and sh > 0 and is_notice:
            x = int(round(float(sw) * 0.915))
            y = int(round(float(sh) * 0.115))
            x = max(0, min(int(sw) - 1, int(x)))
            y = max(0, min(int(sh) - 1, int(y)))
            try:
                self._last_notice_x_xy = (int(x), int(y))
                self._last_notice_x_step = int(step_id)
                self._notice_x_attempts = 0
            except Exception:
                pass
            return {
                "action": "click",
                "target": [int(x), int(y)],
                "reason": "Close announcement/notice board via top-right X fallback.",
                "raw": action.get("raw"),
                "_prompt": action.get("_prompt"),
                "_perception": action.get("_perception"),
                "_model": action.get("_model"),
                "_routine": action.get("_routine"),
                "_close_heuristic": "notice_x_fallback",
            }

        if sw > 0 and sh > 0 and _looks_like_notice_board(items):
            x = int(round(float(sw) * 0.915))
            y = int(round(float(sh) * 0.115))
            x = max(0, min(int(sw) - 1, int(x)))
            y = max(0, min(int(sh) - 1, int(y)))
            try:
                self._last_notice_x_xy = (int(x), int(y))
                self._last_notice_x_step = int(step_id)
                self._notice_x_attempts = 0
            except Exception:
                pass
            return {
                "action": "click",
                "target": [int(x), int(y)],
                "reason": "Close announcement/notice board via top-right X fallback.",
                "raw": action.get("raw"),
                "_prompt": action.get("_prompt"),
                "_perception": action.get("_perception"),
                "_model": action.get("_model"),
                "_routine": action.get("_routine"),
                "_close_heuristic": "notice_x_fallback",
            }

        # Fallback: ESC usually closes menus/popups and is safer than clicking window top-right.
        return {
            "action": "back",
            "reason": "Close popup via ESC heuristic (no close button detected).",
            "raw": action.get("raw"),
            "_prompt": action.get("_prompt"),
            "_perception": action.get("_perception"),
            "_model": action.get("_model"),
            "_routine": action.get("_routine"),
            "_close_heuristic": "esc",
        }

    def _handle_stuck_in_recruit(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Detect if we are accidentally in the Recruit/Gacha screen and back out if needed."""
        if not isinstance(action, dict):
            return action
        
        # If we intend to recruit, don't interfere
        reason = str(action.get("reason") or "").lower()
        if "recruit" in reason or "gacha" in reason or "招募" in reason or "抽卡" in reason:
            return action

        items = None
        try:
            items = action.get("_perception", {}).get("items")
        except Exception:
            items = None
        
        if not isinstance(items, list) or not items:
            return action

        # Keywords that strongly suggest we are in the Recruit screen
        recruit_keywords = ["recruit", "pick up", "gacha", "招募", "募集", "概率", "rates", "points", "exchange"]
        
        hit_count = 0
        for it in items:
            if not isinstance(it, dict): continue
            lbl = str(it.get("label") or "").lower()
            if any(k in lbl for k in recruit_keywords):
                hit_count += 1
        
        # If we see multiple recruit keywords, we are likely stuck
        if hit_count >= 2:
            return {
                "action": "back",
                "reason": "Detected 'Recruit' screen while not intending to recruit. Backing out.",
                "raw": action.get("raw"),
                "_prompt": action.get("_prompt"),
                "_perception": action.get("_perception"),
                "_model": action.get("_model"),
                "_routine": action.get("_routine"),
                "_recovery": "stuck_in_recruit",
            }
        return action

    def _maybe_recover_cafe_wrong_screen(self, action: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(action, dict):
            return action
        try:
            if not bool(getattr(self._routine, "is_active", False)):
                return action
            step = self._routine.get_current_step()
            if step is None or str(getattr(step, "name", "") or "") != "Cafe":
                return action
        except Exception:
            return action

        a = str(action.get("action") or "").lower().strip()
        if a == "back":
            return action

        reason = str(action.get("reason") or "")
        rlow = reason.lower()
        if any(k in rlow or k in reason for k in ("close", "cancel", "popup", "关闭", "取消", "弹窗")):
            return action

        items = None
        try:
            items = action.get("_perception", {}).get("items")
        except Exception:
            items = None
        if not isinstance(items, list) or not items:
            return action

        iw, ih = self._resolve_image_size(items, meta=action.get("_perception"))
        if iw <= 0 or ih <= 0:
            return action

        if self._is_lobby_view(items, width=iw, height=ih):
            return action
        if self._is_cafe_interior(items, width=iw, height=ih):
            return action

        return {
            "action": "back",
            "reason": "Cafe step recovery: detected neither Lobby nor Cafe interior; backing out.",
            "raw": action.get("raw"),
            "_prompt": action.get("_prompt"),
            "_perception": action.get("_perception"),
            "_model": action.get("_model"),
            "_routine": action.get("_routine"),
            "_recovery": "cafe_wrong_screen",
        }

    def _debounce_click(self, action: Dict[str, Any], *, step_id: int) -> Dict[str, Any]:
        if not isinstance(action, dict):
            return action
        a = str(action.get("action") or "").lower().strip()
        if a != "click":
            return action
        if bool(action.get("_startup_tap")):
            return action
        tgt = action.get("target")
        if not (isinstance(tgt, (list, tuple)) and len(tgt) == 2):
            return action

        try:
            cd = int(getattr(self.cfg, "click_debounce_steps", 2) or 2)
        except Exception:
            cd = 2
        try:
            dist = int(getattr(self.cfg, "click_debounce_dist_px", 24) or 24)
        except Exception:
            dist = 24

        x, y = int(tgt[0]), int(tgt[1])
        reason = str(action.get("reason") or "")
        rnorm = reason.strip().lower()

        ch = str(action.get("_close_heuristic") or "").strip().lower()
        if ch in ("notice_x_fallback", "perception_close"):
            return action
        if bool(action.get("_cafe_nav_override")):
            return action

        try:
            if bool(getattr(self._routine, "is_active", False)):
                step = self._routine.get_current_step()
                if step is not None and str(getattr(step, "name", "") or "") == "Cafe":
                    if any(k in rnorm or k in reason for k in ("cafe", "咖啡", "咖啡厅", "咖啡廳")):
                        return action
        except Exception:
            pass

        last_xy = self._last_click_xy
        if last_xy is not None and (int(step_id) - int(self._last_click_step)) <= int(cd):
            if abs(int(x) - int(last_xy[0])) + abs(int(y) - int(last_xy[1])) <= int(dist) and (not rnorm or rnorm == self._last_click_reason):
                return {
                    "action": "wait",
                    "duration_ms": 600,
                    "reason": "Debounced repeated click at nearly the same location.",
                    "raw": action.get("raw"),
                    "_prompt": action.get("_prompt"),
                    "_perception": action.get("_perception"),
                    "_model": action.get("_model"),
                    "_routine": action.get("_routine"),
                    "_debounced": True,
                }

        self._last_click_step = int(step_id)
        self._last_click_xy = (int(x), int(y))
        self._last_click_reason = rnorm
        return action

    def _block_check_lobby_noise(self, action: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(action, dict):
            return action
        try:
            if not bool(getattr(self._routine, "is_active", False)):
                return action
            step = self._routine.get_current_step()
            if step is None or str(getattr(step, "name", "") or "") != "Check Lobby":
                return action
        except Exception:
            return action

        a = str(action.get("action") or "").lower().strip()
        if a != "click":
            return action

        reason = str(action.get("reason") or "")
        rlow = reason.lower()

        # Allow closing popups (even if the text contains words like "task").
        try:
            raw = str(action.get("raw") or "")
        except Exception:
            raw = ""
        blob = (reason + "\n" + raw)
        blob_low = (blob or "").lower()
        if (
            "关闭" in blob
            or "关掉" in blob
            or "close" in blob_low
            or "popup" in blob_low
            or "弹窗" in blob
            or "公告" in blob
        ):
            return action

        # During lobby check, don't open secondary menus like Tasks/Schedule/etc.
        if any(k in rlow for k in ("task", "schedule", "club", "bounty", "mail", "recruit", "recruitment", "gacha", "notice", "announcement", "notification", "event")) or any(
            k in reason for k in ("任务", "日程", "社团", "悬赏", "邮箱", "邮件", "招募", "抽卡", "公告", "通知", "红点", "red dot", "活动")
        ):
            return {
                "action": "wait",
                "duration_ms": 800,
                "reason": "Blocked click to secondary menu during Check Lobby; waiting for a clearer state.",
                "raw": action.get("raw"),
                "_prompt": action.get("_prompt"),
                "_perception": action.get("_perception"),
                "_model": action.get("_model"),
                "_routine": action.get("_routine"),
                "_blocked": True,
            }
        return action

    def _detect_yellow_markers(self, *, screenshot_path: str) -> List[List[int]]:
        try:
            with Image.open(screenshot_path) as im:
                im = im.convert("RGB")
                ow, oh = im.size
                if ow <= 0 or oh <= 0:
                    return []

                scale = 0.25
                nw = max(1, int(round(float(ow) * scale)))
                nh = max(1, int(round(float(oh) * scale)))
                sim = im.resize((nw, nh), resample=Image.BILINEAR)
                px = sim.load()

                visited = [[False] * nw for _ in range(nh)]
                out: List[List[int]] = []

                def is_yellow(r: int, g: int, b: int) -> bool:
                    return r >= 200 and g >= 150 and b <= 140 and (r - b) >= 60

                for y in range(nh):
                    for x in range(nw):
                        if visited[y][x]:
                            continue
                        r, g, b = px[x, y]
                        if not is_yellow(int(r), int(g), int(b)):
                            visited[y][x] = True
                            continue

                        # BFS component
                        stack = [(x, y)]
                        visited[y][x] = True
                        minx = maxx = x
                        miny = maxy = y
                        cnt = 0
                        while stack:
                            cx, cy = stack.pop()
                            cnt += 1
                            if cx < minx:
                                minx = cx
                            if cx > maxx:
                                maxx = cx
                            if cy < miny:
                                miny = cy
                            if cy > maxy:
                                maxy = cy

                            for nx2, ny2 in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                                if nx2 < 0 or nx2 >= nw or ny2 < 0 or ny2 >= nh:
                                    continue
                                if visited[ny2][nx2]:
                                    continue
                                rr, gg, bb = px[nx2, ny2]
                                if is_yellow(int(rr), int(gg), int(bb)):
                                    visited[ny2][nx2] = True
                                    stack.append((nx2, ny2))
                                else:
                                    visited[ny2][nx2] = True

                        bw = maxx - minx + 1
                        bh = maxy - miny + 1
                        area = bw * bh

                        # Filter out too small/large blobs.
                        if cnt < 10 or cnt > 2000:
                            continue
                        if area < 20 or area > 8000:
                            continue
                        if bw < 2 or bh < 2:
                            continue

                        # Convert bbox back to original coords.
                        x1 = int(round(float(minx) / scale))
                        y1 = int(round(float(miny) / scale))
                        x2 = int(round(float(maxx + 1) / scale))
                        y2 = int(round(float(maxy + 1) / scale))
                        x1 = max(0, min(int(ow) - 1, x1))
                        y1 = max(0, min(int(oh) - 1, y1))
                        x2 = max(0, min(int(ow), x2))
                        y2 = max(0, min(int(oh), y2))
                        if x2 <= x1 or y2 <= y1:
                            continue

                        # Headpat markers are usually not near the very bottom UI.
                        if y1 > int(oh * 0.85):
                            continue

                        out.append([int(x1), int(y1), int(x2), int(y2)])

                out.sort(key=lambda bb: (bb[1], bb[0]))
                return out[:12]
        except Exception:
            return []

    def _maybe_cafe_headpat(self, *, action: Dict[str, Any], screenshot_path: str, step_id: int) -> Dict[str, Any]:
        if not bool(getattr(self.cfg, "cafe_headpat", False)):
            return action
        try:
            if not bool(getattr(self._routine, "is_active", False)):
                return action
            step = self._routine.get_current_step()
            if step is None or str(getattr(step, "name", "") or "") != "Cafe":
                return action
        except Exception:
            return action

        if isinstance(action, dict) and action.get("_cafe_action"):
            return action

        # Don't override the navigation click that enters Cafe.
        try:
            reason = str(action.get("reason") or "")
            rlow = reason.lower()
            if ("cafe" in rlow or "咖啡" in reason) and any(k in rlow or k in reason for k in ("navigate", "go to", "enter", "前往", "进入")):
                return action
        except Exception:
            pass

        sw, sh = 0, 0
        try:
            with Image.open(screenshot_path) as im:
                sw, sh = im.size
        except Exception:
            sw, sh = 0, 0

        try:
            items = action.get("_perception", {}).get("items")
        except Exception:
            items = None
        if not (sw > 0 and sh > 0 and isinstance(items, list) and items):
            return action
        if self._is_lobby_view(items, width=sw, height=sh):
            return action
        if not self._is_cafe_interior(items, width=sw, height=sh):
            return action

        try:
            cd = int(getattr(self.cfg, "cafe_headpat_cooldown_steps", 1) or 1)
        except Exception:
            cd = 1
        if step_id - int(self._last_headpat_step) <= cd:
            return action

        # Don't override explicit "claim earnings" actions.
        try:
            rlow = str(action.get("reason") or "").lower()
            if "claim" in rlow and ("earn" in rlow or "credit" in rlow or "ap" in rlow):
                return action
        except Exception:
            pass

        marks = self._detect_yellow_markers(screenshot_path=screenshot_path)
        if marks and sh > 0:
            marks = [bb for bb in marks if ((bb[1] + bb[3]) / 2) >= float(sh) * 0.2]
        if not marks:
            return action

        chosen = None
        for bb in marks:
            cx, cy = _center(bb)
            last = self._last_headpat_xy
            if last is None:
                chosen = bb
                break
            if abs(int(cx) - int(last[0])) + abs(int(cy) - int(last[1])) > 60:
                chosen = bb
                break
        if chosen is None:
            chosen = marks[0]

        x1, y1, x2, y2 = [int(v) for v in chosen]
        cx, cy = _center(chosen)
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        tx = int(cx + max(18, int(round(bw * 0.6))))
        ty = int(cy + max(18, int(round(bh * 0.9))))

        self._last_headpat_step = int(step_id)
        self._last_headpat_xy = (int(cx), int(cy))

        out = {
            "action": "click",
            "target": [int(tx), int(ty)],
            "reason": "Cafe headpat: yellow interaction marker detected; clicking slightly right/below to interact with the student.",
            "raw": action.get("raw"),
            "_prompt": action.get("_prompt"),
            "_perception": action.get("_perception"),
            "_model": action.get("_model"),
            "_routine": action.get("_routine"),
            "_headpat": True,
        }
        return out

    def _maybe_cafe_idle_exit(self, *, action: Dict[str, Any], screenshot_path: str, step_id: int) -> Dict[str, Any]:
        if not isinstance(action, dict):
            return action
        try:
            if not bool(getattr(self._routine, "is_active", False)):
                self._cafe_idle_steps = 0
                return action
            step = self._routine.get_current_step()
            if step is None or str(getattr(step, "name", "") or "") != "Cafe":
                self._cafe_idle_steps = 0
                return action
        except Exception:
            return action

        if action.get("_cafe_action") or action.get("_headpat"):
            self._cafe_idle_steps = 0
            return action

        a = str(action.get("action") or "").lower().strip()
        if a != "wait":
            self._cafe_idle_steps = 0
            return action

        try:
            items = action.get("_perception", {}).get("items")
        except Exception:
            items = None
        if not isinstance(items, list) or not items:
            return action

        iw, ih = self._resolve_image_size(items, meta=action.get("_perception"))
        if iw <= 0 or ih <= 0:
            return action
        if not self._is_cafe_interior(items, width=iw, height=ih):
            self._cafe_idle_steps = 0
            return action
        if self._has_cafe_claim_ui(items, height=ih) or self._has_cafe_control_ui(items, height=ih):
            self._cafe_idle_steps = 0
            return action
        if step_id - int(self._cafe_last_claim_step) <= 2:
            self._cafe_idle_steps = 0
            return action
        if step_id - int(self._cafe_last_invite_step) <= 2:
            self._cafe_idle_steps = 0
            return action
        if step_id - int(self._cafe_last_store2_step) <= 2:
            self._cafe_idle_steps = 0
            return action

        try:
            marks = self._detect_yellow_markers(screenshot_path=screenshot_path)
        except Exception:
            marks = []
        if marks:
            self._cafe_idle_steps = 0
            return action

        self._cafe_idle_steps = int(self._cafe_idle_steps) + 1
        if self._cafe_idle_steps < 4:
            return action
        self._cafe_idle_steps = 0
        return {
            "action": "back",
            "reason": "Cafe idle: no claim/invite/headpat targets; backing out to lobby.",
            "raw": action.get("raw"),
            "_prompt": action.get("_prompt"),
            "_perception": action.get("_perception"),
            "_model": action.get("_model"),
            "_routine": action.get("_routine"),
            "_cafe_idle_exit": True,
        }

    def _decide(self, *, screenshot_path: str, step_id: int = -1, orig_size: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        with Image.open(screenshot_path) as im:
            w, h = im.size

        force_small = False
        try:
            force_small = int(step_id) <= int(getattr(self, "_vlm_force_small_until_step", -10_000))
        except Exception:
            force_small = False

        self._set_stage(step_id=int(step_id), stage="load_model")
        engine = get_local_vlm(
            model=self.cfg.model,
            models_dir=self.cfg.models_dir,
            hf_home=self.cfg.hf_home,
            device=self.cfg.device,
        )
        try:
            t_load0 = time.time()
            if hasattr(engine, "ensure_loaded"):
                engine.ensure_loaded()
            t_load = time.time() - t_load0
            if t_load > 0.25:
                ts = datetime.now().isoformat(timespec="seconds")
                self._log_out(f"[{ts}] step={int(step_id)} stage=load_model detail=loaded_in_s:{t_load:.2f}")
        except Exception as e:
            return {
                "action": "wait",
                "duration_ms": 1500,
                "reason": "VLM model load failed; waiting.",
                "raw": "",
                "_vlm_error": str(e),
                "_routine": self._routine_meta(),
            }

        p_prompt = self._perception_prompt(width=int(w), height=int(h))
        items = []
        p_raw = ""
        try:
            n = int(getattr(self.cfg, "perception_every_n_steps", 1) or 1)
        except Exception:
            n = 1
        run_perception = n <= 1 or step_id < 0 or (step_id % n == 0)
        try:
            if self._last_notice_x_xy is not None and (int(step_id) - int(self._last_notice_x_step)) <= 6:
                run_perception = True
        except Exception:
            pass
        if run_perception:
            self._set_stage(step_id=int(step_id), stage="perception")
            t_p0 = time.time()
            p_dt = 0.0
            try:
                ts = datetime.now().isoformat(timespec="seconds")
                self._log_out(f"[{ts}] step={int(step_id)} stage=perception_begin")
            except Exception:
                pass
            p_mnt = int(self.cfg.perception_max_new_tokens)
            if force_small:
                p_mnt = int(min(int(p_mnt), 128))
            p_res = engine.ocr(image_path=screenshot_path, prompt=p_prompt, max_new_tokens=int(p_mnt))
            try:
                dt = time.time() - t_p0
                p_dt = float(dt)
                ts = datetime.now().isoformat(timespec="seconds")
                self._log_out(f"[{ts}] step={int(step_id)} stage=perception_end elapsed_s={dt:.2f}")
            except Exception:
                pass
            p_err = ""
            try:
                p_err = str(p_res.get("error") or "")
            except Exception:
                p_err = ""
            p_raw = str(p_res.get("raw") or "")
            if p_err == "hard_timeout" or p_err.startswith("startup_error"):
                try:
                    self._vlm_force_small_until_step = max(int(getattr(self, "_vlm_force_small_until_step", -10_000)), int(step_id) + 12)
                except Exception:
                    pass
                act = {
                    "action": "wait",
                    "duration_ms": 900,
                    "reason": "VLM perception timed out; waiting and retrying next step.",
                    "raw": p_raw,
                    "_vlm_error": p_err,
                }
                act.setdefault("_prompt", p_prompt)
                act.setdefault("_perception", {"prompt": p_prompt, "raw": p_raw, "items": [], "image_size": [int(w), int(h)]})
                try:
                    if orig_size and isinstance(orig_size, (tuple, list)) and len(orig_size) == 2:
                        ow2, oh2 = int(orig_size[0]), int(orig_size[1])
                        if ow2 > 0 and oh2 > 0:
                            act["_perception"]["image_size"] = [int(ow2), int(oh2)]
                except Exception:
                    pass
                act.setdefault("_routine", self._routine_meta())
                return act
            p_parsed = {}
            try:
                p_parsed = _parse_json_content(p_raw)
            except Exception:
                p_parsed = {}
            if isinstance(p_parsed, dict) and isinstance(p_parsed.get("items"), list):
                items = p_parsed.get("items")
            if not items and p_raw:
                try:
                    fb_items = _parse_perception_items_fallback(p_raw)
                except Exception:
                    fb_items = []
                if fb_items:
                    items = fb_items

            try:
                if (not force_small) and float(p_dt) >= 45.0:
                    self._vlm_force_small_until_step = max(int(getattr(self, "_vlm_force_small_until_step", -10_000)), int(step_id) + 12)
                    items2 = items
                    img_size = [int(w), int(h)]
                    try:
                        if orig_size and isinstance(orig_size, (tuple, list)) and len(orig_size) == 2:
                            ow2, oh2 = int(orig_size[0]), int(orig_size[1])
                            if ow2 > 0 and oh2 > 0 and int(w) > 0 and int(h) > 0 and (ow2 != int(w) or oh2 != int(h)) and isinstance(items, list):
                                sx = float(ow2) / float(w)
                                sy = float(oh2) / float(h)
                                scaled = []
                                for it in items:
                                    if not isinstance(it, dict):
                                        continue
                                    bb = it.get("bbox")
                                    if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                                        scaled.append(it)
                                        continue
                                    x1, y1, x2, y2 = [int(v) for v in bb]
                                    it2 = dict(it)
                                    it2["bbox"] = [
                                        int(round(float(x1) * sx)),
                                        int(round(float(y1) * sy)),
                                        int(round(float(x2) * sx)),
                                        int(round(float(y2) * sy)),
                                    ]
                                    scaled.append(it2)
                                items2 = scaled
                                img_size = [int(ow2), int(oh2)]
                    except Exception:
                        items2 = items
                        img_size = [int(w), int(h)]
                    act = {
                        "action": "wait",
                        "duration_ms": 450,
                        "reason": "VLM perception was very slow; skipping policy this step and forcing smaller VLM input for next steps.",
                        "raw": p_raw,
                        "_vlm_error": "slow_perception",
                    }
                    act.setdefault("_prompt", p_prompt)
                    act.setdefault("_perception", {"prompt": p_prompt, "raw": p_raw, "items": items2, "image_size": img_size})
                    act.setdefault("_routine", self._routine_meta())
                    return act
            except Exception:
                pass

        prompt = self._prompt(width=int(w), height=int(h))
        items_txt = self._format_items(items)
        if items_txt:
            prompt = prompt + "\nDetected UI text elements (bbox label):\n" + items_txt + "\n"

        self._set_stage(step_id=int(step_id), stage="policy")
        t_a0 = time.time()
        a_dt = 0.0
        try:
            ts = datetime.now().isoformat(timespec="seconds")
            self._log_out(f"[{ts}] step={int(step_id)} stage=policy_begin")
        except Exception:
            pass
        a_mnt = int(self.cfg.max_new_tokens)
        if force_small:
            a_mnt = int(min(int(a_mnt), 128))
        res = engine.ocr(image_path=screenshot_path, prompt=prompt, max_new_tokens=int(a_mnt))
        try:
            dt = time.time() - t_a0
            a_dt = float(dt)
            ts = datetime.now().isoformat(timespec="seconds")
            self._log_out(f"[{ts}] step={int(step_id)} stage=policy_end elapsed_s={dt:.2f}")
        except Exception:
            pass
        raw = str(res.get("raw") or "")
        vlm_err = ""
        try:
            vlm_err = str(res.get("error") or "")
        except Exception:
            vlm_err = ""
        if vlm_err:
            try:
                ts = datetime.now().isoformat(timespec="seconds")
                self._log_out(f"[{ts}] step={int(step_id)} stage=policy_vlm_error error={vlm_err}")
            except Exception:
                pass

        try:
            if (not force_small) and float(a_dt) >= 45.0:
                self._vlm_force_small_until_step = max(int(getattr(self, "_vlm_force_small_until_step", -10_000)), int(step_id) + 12)
        except Exception:
            pass

        if vlm_err == "hard_timeout":
            try:
                self._vlm_force_small_until_step = max(int(getattr(self, "_vlm_force_small_until_step", -10_000)), int(step_id) + 12)
            except Exception:
                pass
            act = {
                "action": "wait",
                "duration_ms": 800,
                "reason": "VLM policy timed out; model worker restarted. Waiting and retrying next step.",
                "raw": raw,
                "_vlm_error": vlm_err,
            }
            act.setdefault("_prompt", prompt)
            act.setdefault("_perception", {"prompt": p_prompt, "raw": p_raw, "items": items, "image_size": [int(w), int(h)]})
            try:
                p = act.get("_perception")
                if isinstance(p, dict) and orig_size and isinstance(orig_size, (tuple, list)) and len(orig_size) == 2:
                    ow2, oh2 = int(orig_size[0]), int(orig_size[1])
                    if ow2 > 0 and oh2 > 0 and int(w) > 0 and int(h) > 0:
                        if (ow2 != int(w) or oh2 != int(h)) and isinstance(p.get("items"), list):
                            sx = float(ow2) / float(w)
                            sy = float(oh2) / float(h)
                            scaled = []
                            for it in p.get("items"):
                                if not isinstance(it, dict):
                                    continue
                                bb = it.get("bbox")
                                if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                                    scaled.append(it)
                                    continue
                                x1, y1, x2, y2 = [int(v) for v in bb]
                                it2 = dict(it)
                                it2["bbox"] = [
                                    int(round(float(x1) * sx)),
                                    int(round(float(y1) * sy)),
                                    int(round(float(x2) * sx)),
                                    int(round(float(y2) * sy)),
                                ]
                                scaled.append(it2)
                            p["items"] = scaled
                        p["image_size"] = [int(ow2), int(oh2)]
            except Exception:
                pass
            act.setdefault("_routine", self._routine_meta())
            return act
        try:
            act = _parse_action_content(raw)
        except Exception:
            act = {
                "action": "wait",
                "duration_ms": 1200,
                "reason": "Policy output could not be parsed as JSON; falling back to wait.",
            }
        try:
            if not isinstance(act, dict):
                act = {}
            a0 = str(act.get("action") or "").strip().lower()
            s0 = str(raw or "").strip()
            if (not a0) and (not s0 or s0 == "{" or (s0.startswith("{") and ("}" not in s0))):
                act = {
                    "action": "wait",
                    "duration_ms": 250,
                    "reason": "delegate: close_notice",
                    "_vlm_error": "truncated_policy_output",
                }
        except Exception:
            pass
        if vlm_err:
            act.setdefault("_vlm_error", vlm_err)
        act.setdefault("raw", raw)
        act.setdefault("_prompt", prompt)
        act.setdefault("_perception", {"prompt": p_prompt, "raw": p_raw, "items": items, "image_size": [int(w), int(h)]})
        act.setdefault(
            "_model",
            {
                "model": self.cfg.model,
                "models_dir": self.cfg.models_dir,
                "device": self.cfg.device,
                "max_new_tokens": int(self.cfg.max_new_tokens),
                "perception_max_new_tokens": int(self.cfg.perception_max_new_tokens),
                "exploration_click": bool(self.cfg.exploration_click),
                "forbid_premium_currency": bool(self.cfg.forbid_premium_currency),
                "autoclick_safe_buttons": bool(self.cfg.autoclick_safe_buttons),
            },
        )
        act.setdefault("_routine", self._routine_meta())

        try:
            a0 = str(act.get("action") or "").lower().strip()
            r0 = str(act.get("reason") or "")
            if not a0:
                a0 = "wait"
                act["action"] = "wait"
            if a0 not in ("wait", "stop", "done"):
                act["_vlm_action_original"] = a0
                act["_vlm_action_blocked"] = True
                act.pop("target", None)
                act.pop("bbox", None)
                act.pop("from", None)
                act.pop("to", None)
                act["action"] = "wait"
                act.setdefault("duration_ms", 350)
                if "delegate:" in str(r0).lower():
                    act["reason"] = str(r0)
                else:
                    act["reason"] = f"Blocked VLM action '{a0}'; VLM is restricted to wait/delegate only."
        except Exception:
            pass

        ow, oh = 0, 0
        try:
            if orig_size and isinstance(orig_size, (tuple, list)) and len(orig_size) == 2:
                ow, oh = int(orig_size[0]), int(orig_size[1])
        except Exception:
            ow, oh = 0, 0

        def _looks_like_960_space_xy(x: int, y: int) -> bool:
            try:
                if int(w) <= 0 or int(h) <= 0:
                    return False
                if int(w) <= 1280 and int(h) <= 720:
                    return False
                if not (0 <= int(x) <= 960 and 0 <= int(y) <= 540):
                    return False
                rx = float(int(w)) / 960.0
                ry = float(int(h)) / 540.0
                if rx < 1.25 or ry < 1.25:
                    return False
                edge = (int(x) >= int(round(960 * 0.65))) or (int(y) >= int(round(540 * 0.65)))
                return bool(edge)
            except Exception:
                return False

        def _960_to_infer_xy(x: int, y: int) -> tuple[int, int]:
            if not _looks_like_960_space_xy(int(x), int(y)):
                return int(x), int(y)
            try:
                sx2 = float(int(w)) / 960.0
                sy2 = float(int(h)) / 540.0
                x2 = int(round(float(int(x)) * sx2))
                y2 = int(round(float(int(y)) * sy2))
                act.setdefault("_coord_remap", {})
                act["_coord_remap"]["baseline_960_to_infer"] = True
                return int(x2), int(y2)
            except Exception:
                return int(x), int(y)

        def _clamp_to_xy(x: int, y: int, ww: int, hh: int) -> tuple[int, int]:
            if int(ww) > 0:
                x = int(max(0, min(int(ww) - 1, int(x))))
            if int(hh) > 0:
                y = int(max(0, min(int(hh) - 1, int(y))))
            return int(x), int(y)

        def _clamp_to_image_xy(x: int, y: int) -> tuple[int, int]:
            return _clamp_to_xy(int(x), int(y), int(w), int(h))

        def _clamp_to_orig_xy(x: int, y: int) -> tuple[int, int]:
            return _clamp_to_xy(int(x), int(y), int(ow), int(oh))

        def _looks_like_orig_space_xy(x: int, y: int) -> bool:
            if not (ow > 0 and oh > 0 and int(w) > 0 and int(h) > 0):
                return False
            if ow == int(w) and oh == int(h):
                return False
            mx = max(24, int(round(0.05 * float(int(w)))))
            my = max(24, int(round(0.05 * float(int(h)))))
            within_orig = 0 <= int(x) <= (ow - 1) and 0 <= int(y) <= (oh - 1)
            if not within_orig:
                return False
            too_big_x = int(x) > (int(w) - 1 + mx)
            too_big_y = int(y) > (int(h) - 1 + my)
            return bool(too_big_x or too_big_y)

        def _orig_to_infer_xy(x: int, y: int) -> tuple[int, int]:
            if not _looks_like_orig_space_xy(int(x), int(y)):
                return int(x), int(y)
            try:
                x2 = int(round(float(int(x)) * float(int(w)) / float(ow)))
                y2 = int(round(float(int(y)) * float(int(h)) / float(oh)))
                act.setdefault("_coord_remap", {})
                act["_coord_remap"]["orig_to_infer"] = True
                return int(x2), int(y2)
            except Exception:
                return int(x), int(y)

        def _clamp_bbox(bb: list[int]) -> list[int]:
            if len(bb) != 4:
                return bb
            x1, y1, x2, y2 = [int(v) for v in bb]
            x1, y1 = _clamp_to_image_xy(int(x1), int(y1))
            x2, y2 = _clamp_to_image_xy(int(x2), int(y2))
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            return [int(x1), int(y1), int(x2), int(y2)]

        def _clamp_bbox_orig(bb: list[int]) -> list[int]:
            if len(bb) != 4:
                return bb
            x1, y1, x2, y2 = [int(v) for v in bb]
            x1, y1 = _clamp_to_orig_xy(int(x1), int(y1))
            x2, y2 = _clamp_to_orig_xy(int(x2), int(y2))
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            return [int(x1), int(y1), int(x2), int(y2)]

        # If the model outputs coords that don't fit the current inference image (w,h)
        # but DO fit the original screenshot (orig_size), treat them as original-space coords
        # and skip the later rescale step.
        # Clamp coordinates to inference image bounds BEFORE rescaling to orig_size.
        try:
            tgt = act.get("target")
            if isinstance(tgt, (list, tuple)) and len(tgt) == 2:
                x0, y0 = int(tgt[0]), int(tgt[1])
                x1, y1 = _960_to_infer_xy(int(x0), int(y0))
                x2, y2 = _orig_to_infer_xy(int(x1), int(y1))
                x3, y3 = _clamp_to_image_xy(int(x2), int(y2))
                act["target"] = [int(x3), int(y3)]
        except Exception:
            pass

        try:
            bb = act.get("bbox")
            if isinstance(bb, (list, tuple)) and len(bb) == 4:
                ax1, ay1, ax2, ay2 = [int(v) for v in bb]
                bx1, by1 = _960_to_infer_xy(int(ax1), int(ay1))
                bx2, by2 = _960_to_infer_xy(int(ax2), int(ay2))
                cx1, cy1 = _orig_to_infer_xy(int(bx1), int(by1))
                cx2, cy2 = _orig_to_infer_xy(int(bx2), int(by2))
                act["bbox"] = _clamp_bbox([int(cx1), int(cy1), int(cx2), int(cy2)])
        except Exception:
            pass

        try:
            p1 = act.get("from")
            p2 = act.get("to")
            if isinstance(p1, (list, tuple)) and len(p1) == 2:
                x0, y0 = int(p1[0]), int(p1[1])
                x1, y1 = _960_to_infer_xy(int(x0), int(y0))
                x2, y2 = _orig_to_infer_xy(int(x1), int(y1))
                x3, y3 = _clamp_to_image_xy(int(x2), int(y2))
                act["from"] = [int(x3), int(y3)]
            if isinstance(p2, (list, tuple)) and len(p2) == 2:
                x0, y0 = int(p2[0]), int(p2[1])
                x1, y1 = _960_to_infer_xy(int(x0), int(y0))
                x2, y2 = _orig_to_infer_xy(int(x1), int(y1))
                x3, y3 = _clamp_to_image_xy(int(x2), int(y2))
                act["to"] = [int(x3), int(y3)]
        except Exception:
            pass

        try:
            p = act.get("_perception")
            if isinstance(p, dict) and isinstance(p.get("items"), list):
                for it in p.get("items"):
                    if not isinstance(it, dict):
                        continue
                    bb = it.get("bbox")
                    if isinstance(bb, (list, tuple)) and len(bb) == 4:
                        it["bbox"] = _clamp_bbox([int(v) for v in bb])
        except Exception:
            pass

        if orig_size and isinstance(orig_size, (tuple, list)) and len(orig_size) == 2:
            try:
                ow2, oh2 = int(orig_size[0]), int(orig_size[1])
                if ow2 > 0 and oh2 > 0 and int(w) > 0 and int(h) > 0 and (ow2 != int(w) or oh2 != int(h)):
                    sx = float(ow) / float(w)
                    sy = float(oh) / float(h)
                    act = self._rescale_action(act, sx=sx, sy=sy)
            except Exception:
                pass

        try:
            p = act.get("_perception")
            if isinstance(p, dict) and isinstance(p.get("items"), list) and orig_size and isinstance(orig_size, (tuple, list)) and len(orig_size) == 2:
                ow2, oh2 = int(orig_size[0]), int(orig_size[1])
                if ow2 > 0 and oh2 > 0 and int(w) > 0 and int(h) > 0 and (ow2 != int(w) or oh2 != int(h)):
                    sx = float(ow2) / float(w)
                    sy = float(oh2) / float(h)
                    items2 = []
                    for it in p.get("items"):
                        if not isinstance(it, dict):
                            continue
                        bb = it.get("bbox")
                        if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                            items2.append(it)
                            continue
                        x1, y1, x2, y2 = [int(v) for v in bb]
                        x1 = int(round(float(x1) * sx))
                        y1 = int(round(float(y1) * sy))
                        x2 = int(round(float(x2) * sx))
                        y2 = int(round(float(y2) * sy))
                        it2 = dict(it)
                        it2["bbox"] = [int(x1), int(y1), int(x2), int(y2)]
                        items2.append(it2)
                    p["items"] = items2
        except Exception:
            pass

        try:
            p = act.get("_perception")
            if isinstance(p, dict) and orig_size and isinstance(orig_size, (tuple, list)) and len(orig_size) == 2:
                ow2, oh2 = int(orig_size[0]), int(orig_size[1])
                if ow2 > 0 and oh2 > 0:
                    p["image_size"] = [int(ow2), int(oh2)]
        except Exception:
            pass

        return act

    def _execute(self, action: Dict[str, Any], *, screenshot_path: str, step_id: int = -1) -> None:
        a = str(action.get("action", "")).lower().strip()
        if a == "wait":
            ms = int(action.get("duration_ms", 800))
            time.sleep(max(0, ms) / 1000.0)
            return

        if a == "stop":
            self._stop.set()
            return

        if self.cfg.dry_run:
            return

        try:
            with Image.open(screenshot_path) as im:
                w, h = im.size
        except Exception:
            w, h = 0, 0

        cw, ch = 0, 0
        try:
            if hasattr(self._device, "client_size"):
                cw, ch = self._device.client_size()
        except Exception:
            cw, ch = 0, 0

        sx, sy = 1.0, 1.0
        if w > 0 and h > 0 and cw > 0 and ch > 0 and (cw != w or ch != h):
            sx = float(cw) / float(w)
            sy = float(ch) / float(h)

        try:
            action.setdefault(
                "_telemetry",
                {
                    "shot_size": [int(w), int(h)],
                    "client_size": [int(cw), int(ch)],
                    "scale": [float(round(sx, 6)), float(round(sy, 6))],
                },
            )
        except Exception:
            pass

        def _scale_xy(x: int, y: int) -> tuple[int, int]:
            if sx == 1.0 and sy == 1.0:
                return int(x), int(y)
            return int(round(float(x) * sx)), int(round(float(y) * sy))

        def _clamp_xy(x: int, y: int) -> tuple[int, int]:
            ox, oy = int(x), int(y)
            if cw > 0:
                x = int(max(0, min(int(cw) - 1, int(x))))
            if ch > 0:
                y = int(max(0, min(int(ch) - 1, int(y))))
            nx, ny = int(x), int(y)
            if (ox, oy) != (nx, ny):
                try:
                    ts = datetime.now().isoformat(timespec="seconds")
                    self._log_out(
                        f"[{ts}] exec_clamp step={int(step_id)} target=[{ox},{oy}] -> [{nx},{ny}] "
                        f"shot=[{w},{h}] client=[{cw},{ch}] scale=[{sx:.3f},{sy:.3f}]"
                    )
                except Exception:
                    pass
            return int(x), int(y)

        def _scale_bbox(bb: list[int]) -> list[int]:
            if len(bb) != 4:
                return bb
            x1, y1, x2, y2 = [int(v) for v in bb]
            p1x, p1y = _scale_xy(int(x1), int(y1))
            p2x, p2y = _scale_xy(int(x2), int(y2))
            x1c, y1c = _clamp_xy(int(p1x), int(p1y))
            x2c, y2c = _clamp_xy(int(p2x), int(p2y))
            if x2c < x1c:
                x1c, x2c = x2c, x1c
            if y2c < y1c:
                y1c, y2c = y2c, y1c
            return [int(x1c), int(y1c), int(x2c), int(y2c)]

        try:
            p = action.get("_perception")
            if isinstance(p, dict) and isinstance(p.get("items"), list):
                items2 = []
                for it in p.get("items"):
                    if not isinstance(it, dict):
                        continue
                    bb = it.get("bbox")
                    if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                        continue
                    it2 = dict(it)
                    it2["bbox"] = _scale_bbox([int(v) for v in bb])
                    items2.append(it2)
                action["_perception_client"] = {"items": items2}
        except Exception:
            pass

        if a == "click":
            target = action.get("target")
            if isinstance(target, (list, tuple)) and len(target) == 2:
                try:
                    action["target_model"] = [int(target[0]), int(target[1])]
                except Exception:
                    pass
                tx, ty = _scale_xy(int(target[0]), int(target[1]))
                x, y = _clamp_xy(int(tx), int(ty))
                try:
                    action["target_client"] = [int(x), int(y)]
                    if hasattr(self._device, "client_to_screen"):
                        sx2, sy2 = self._device.client_to_screen(int(x), int(y))
                        action["target_screen"] = [int(sx2), int(sy2)]
                except Exception:
                    pass

                if str(action.get("_close_heuristic") or "") == "notice_x_fallback":
                    try:
                        ts2 = datetime.now().isoformat(timespec="seconds")
                        self._log_out(f"[{ts2}] exec_notice_x_fallback step={int(step_id)} client=[{int(x)},{int(y)}]")
                    except Exception:
                        pass
                    boot = False
                    try:
                        boot = (not bool(getattr(self, "_startup_finished", False))) and int(getattr(self, "_startup_tap_attempts", 0) or 0) <= 0
                    except Exception:
                        boot = False
                    try:
                        k = int(getattr(self, "_notice_x_attempts", 0) or 0)
                    except Exception:
                        k = 0
                    if boot:
                        offsets = ((0, 0),)
                    elif k <= 4:
                        offsets = (
                            (0, 0),
                            (-12, 0),
                            (12, 0),
                            (0, 12),
                            (-12, 12),
                            (12, 12),
                            (0, 24),
                            (-12, 24),
                            (12, 24),
                        )
                    else:
                        offsets = (
                            (0, 0),
                            (-12, 0),
                            (12, 0),
                            (-32, 0),
                            (-48, 0),
                            (0, 12),
                            (-12, 12),
                            (12, 12),
                            (-32, 12),
                            (-48, 12),
                            (0, 24),
                            (-12, 24),
                            (12, 24),
                            (-32, 24),
                            (-48, 24),
                        )
                    for dx, dy in offsets:
                        try:
                            xo = int(x) + int(dx)
                            yo = int(y) + int(dy)
                            if cw > 0:
                                xo = max(0, min(int(cw) - 1, int(xo)))
                            if ch > 0:
                                yo = max(0, min(int(ch) - 1, int(yo)))
                            self._device.click_client(int(xo), int(yo))
                            if hasattr(self._device, "click_client_message"):
                                self._device.click_client_message(int(xo), int(yo))
                        except Exception:
                            pass
                        time.sleep(0.05)
                    try:
                        self._device.press_escape()
                    except Exception:
                        pass
                    return

                if bool(action.get("_startup_tap")):
                    try:
                        ts2 = datetime.now().isoformat(timespec="seconds")
                        self._log_out(f"[{ts2}] exec_startup_tap step={int(step_id)} client=[{int(x)},{int(y)}]")
                    except Exception:
                        pass
                    try:
                        if cw > 0 and ch > 0:
                            x2 = int(round(float(cw) * 0.50))
                            y2 = int(round(float(ch) * 0.82))
                            x2 = max(0, min(int(cw) - 1, int(x2)))
                            y2 = max(0, min(int(ch) - 1, int(y2)))
                            if (int(x2), int(y2)) != (int(x), int(y)):
                                ts3 = datetime.now().isoformat(timespec="seconds")
                                self._log_out(
                                    f"[{ts3}] exec_startup_tap_adjust step={int(step_id)} client=[{int(x)},{int(y)}] -> [{int(x2)},{int(y2)}]"
                                )
                            x, y = int(x2), int(y2)
                    except Exception:
                        pass
                    reached1 = False
                    msg_ok = False
                    try:
                        reached1 = bool(self._device.click_client(int(x), int(y)))
                    except Exception:
                        reached1 = False
                    try:
                        if hasattr(self._device, "click_client_message"):
                            msg_ok = bool(self._device.click_client_message(int(x), int(y)))
                    except Exception:
                        msg_ok = False
                    time.sleep(0.06)
                    try:
                        self._device.click_client(int(x), int(y))
                    except Exception:
                        pass
                    try:
                        if hasattr(self._device, "press_space"):
                            self._device.press_space()
                        if hasattr(self._device, "press_enter"):
                            self._device.press_enter()
                    except Exception:
                        pass
                    try:
                        ts3 = datetime.now().isoformat(timespec="seconds")
                        self._log_out(
                            f"[{ts3}] exec_startup_tap_result step={int(step_id)} reached1={int(bool(reached1))} msg_ok={int(bool(msg_ok))}"
                        )
                    except Exception:
                        pass
                    time.sleep(0.9)
                else:
                    self._device.click_client(int(x), int(y))
                    try:
                        if hasattr(self._device, "click_client_message"):
                            self._device.click_client_message(int(x), int(y))
                    except Exception:
                        pass
                return

            bbox = action.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                try:
                    action["bbox_model"] = [int(v) for v in bbox]
                except Exception:
                    pass
                cx0, cy0 = _center([int(x) for x in bbox])
                tx, ty = _scale_xy(int(cx0), int(cy0))
                x, y = _clamp_xy(int(tx), int(ty))
                try:
                    action["target_client"] = [int(x), int(y)]
                    action["bbox_client"] = _scale_bbox([int(v) for v in bbox])
                    if hasattr(self._device, "client_to_screen"):
                        sx2, sy2 = self._device.client_to_screen(int(x), int(y))
                        action["target_screen"] = [int(sx2), int(sy2)]
                except Exception:
                    pass
                self._device.click_client(int(x), int(y))
                try:
                    if hasattr(self._device, "click_client_message"):
                        self._device.click_client_message(int(x), int(y))
                except Exception:
                    pass
                return

            raise ValueError("click action requires target [x,y] or bbox [x1,y1,x2,y2]")

        if a == "swipe":
            p1 = action.get("from")
            p2 = action.get("to")
            d = int(action.get("duration_ms", 500))
            if not (
                isinstance(p1, (list, tuple))
                and isinstance(p2, (list, tuple))
                and len(p1) == 2
                and len(p2) == 2
            ):
                raise ValueError("swipe action requires from/to")
            p1x, p1y = _scale_xy(int(p1[0]), int(p1[1]))
            p2x, p2y = _scale_xy(int(p2[0]), int(p2[1]))
            x1, y1 = _clamp_xy(int(p1x), int(p1y))
            x2, y2 = _clamp_xy(int(p2x), int(p2y))
            try:
                action["from_model"] = [int(p1[0]), int(p1[1])]
                action["to_model"] = [int(p2[0]), int(p2[1])]
                action["from_client"] = [int(x1), int(y1)]
                action["to_client"] = [int(x2), int(y2)]
                if hasattr(self._device, "client_to_screen"):
                    sx1, sy1 = self._device.client_to_screen(int(x1), int(y1))
                    sx2, sy2 = self._device.client_to_screen(int(x2), int(y2))
                    action["from_screen"] = [int(sx1), int(sy1)]
                    action["to_screen"] = [int(sx2), int(sy2)]
            except Exception:
                pass
            self._device.swipe_client(int(x1), int(y1), int(x2), int(y2), duration_ms=d)
            return

        if a == "back":
            self._device.press_escape()
            return

        raise ValueError(f"unknown action: {a}")

    def _write_traj(self, rec: Dict[str, Any]) -> None:
        try:
            with self._traj_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _write_usage(self, rec: Dict[str, Any]) -> None:
        try:
            with self._usage_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _log_out(self, msg: str) -> None:
        try:
            with self._out_log.open("a", encoding="utf-8") as f:
                f.write(msg.rstrip() + "\n")
        except Exception:
            pass

    def _log_err(self, msg: str) -> None:
        try:
            with self._err_log.open("a", encoding="utf-8") as f:
                f.write(msg.rstrip() + "\n")
        except Exception:
            pass

    def _set_expected_state(self, state: Optional[str], *, delay_s: float = 0.0) -> None:
        try:
            self._expected_state = str(state) if state else None
        except Exception:
            self._expected_state = None
        try:
            self._expected_set_ts = float(time.time())
        except Exception:
            self._expected_set_ts = 0.0
        try:
            self._expected_delay_s = float(max(0.0, float(delay_s)))
        except Exception:
            self._expected_delay_s = 0.0

    def _maybe_set_expected_state_from_action(self, action: Dict[str, Any]) -> None:
        if not isinstance(action, dict):
            return
        try:
            if bool(action.get("_close_heuristic")):
                return
        except Exception:
            pass

        try:
            a = str(action.get("action") or "").lower().strip()
        except Exception:
            a = ""
        try:
            reason = str(action.get("reason") or "")
        except Exception:
            reason = ""
        try:
            rlow = str(reason).lower()
        except Exception:
            rlow = str(reason)

        lbl_hint = ""
        try:
            bb = action.get("bbox")
            items = action.get("_perception", {}).get("items")
            if isinstance(bb, (list, tuple)) and len(bb) == 4 and isinstance(items, list) and items:
                bb2 = [int(v) for v in bb]
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    bb_it = it.get("bbox")
                    if not (isinstance(bb_it, (list, tuple)) and len(bb_it) == 4):
                        continue
                    if [int(v) for v in bb_it] == bb2:
                        lbl_hint = str(it.get("label") or "")
                        break
        except Exception:
            lbl_hint = ""
        try:
            llh = str(lbl_hint).lower()
        except Exception:
            llh = str(lbl_hint)

        try:
            if bool(action.get("_startup_tap")):
                self._set_expected_state("Lobby", delay_s=10.0)
                return
        except Exception:
            pass

        try:
            if bool(action.get("_cafe_nav_override")):
                self._set_expected_state("Cafe_Inside", delay_s=5.0)
                return
        except Exception:
            pass

        if a == "click" and ("cafe" in rlow or ("咖啡" in reason) or ("咖啡廳" in reason) or ("cafe" in llh) or ("咖啡" in lbl_hint) or ("咖啡廳" in lbl_hint)):
            self._set_expected_state("Cafe_Inside", delay_s=4.0)
            return

        if a == "click":
            if ("sweep" in rlow) or ("扫荡" in reason) or ("掃蕩" in reason) or ("sweep" in llh) or ("扫荡" in lbl_hint) or ("掃蕩" in lbl_hint):
                self._set_expected_state("Reward_Popup", delay_s=2.5)
                return
            if ("result" in rlow) or ("结算" in reason) or ("結算" in reason) or ("reward" in rlow) or ("獎勵" in reason) or ("奖励" in reason) or ("result" in llh) or ("结算" in lbl_hint) or ("結算" in lbl_hint) or ("reward" in llh) or ("獎勵" in lbl_hint) or ("奖励" in lbl_hint):
                self._set_expected_state("Result_Screen", delay_s=2.0)
                return
            if ("battle" in rlow) or ("raid" in rlow) or ("出击" in reason) or ("出擊" in reason) or ("开始任务" in reason) or ("開始任務" in reason) or ("battle" in llh) or ("raid" in llh) or ("出击" in lbl_hint) or ("出擊" in lbl_hint):
                self._set_expected_state("Battle_In_Progress", delay_s=4.0)
                return

    def _should_run_supervision(self, *, step_id: int) -> bool:
        try:
            if not bool(getattr(self.cfg, "supervision_enabled", True)):
                return False
        except Exception:
            return False
        try:
            if not self._expected_state:
                return False
        except Exception:
            return False
        try:
            if (int(step_id) - int(self._last_supervision_step)) < int(getattr(self.cfg, "supervision_min_step_interval", 6) or 6):
                return False
        except Exception:
            pass
        try:
            now = float(time.time())
        except Exception:
            now = 0.0
        try:
            if now and (now - float(self._expected_set_ts)) < float(self._expected_delay_s):
                return False
        except Exception:
            pass
        return True

    def _should_run_supervision_any(self, *, step_id: int) -> bool:
        try:
            if not bool(getattr(self.cfg, "supervision_enabled", True)):
                return False
        except Exception:
            return False
        try:
            if not bool(getattr(self.cfg, "supervision_always", True)):
                return False
        except Exception:
            return False

        try:
            boot_preinteractive = (not bool(getattr(self, "_startup_finished", False))) and int(getattr(self, "_startup_tap_attempts", 0) or 0) <= 0
            if boot_preinteractive and int(step_id) <= 60:
                return False
        except Exception:
            pass

        try:
            interval = int(getattr(self.cfg, "supervision_always_min_step_interval", 2) or 2)
        except Exception:
            interval = 2
        try:
            if (int(step_id) - int(getattr(self, "_last_supervision_any_step", -10_000) or -10_000)) < int(interval):
                return False
        except Exception:
            pass
        return True

    def _supervise(self, *, screenshot_path: str, expected_state: str, step_id: int) -> Dict[str, Any]:
        state = str(expected_state or "Lobby").strip() or "Lobby"

        tmp_sup_path = ""
        sup_path = screenshot_path
        try:
            with Image.open(screenshot_path) as im:
                w0, h0 = im.size
                max_side = max(int(w0), int(h0))
                try:
                    target_max_side = int(getattr(self.cfg, "supervision_image_max_side", 960) or 960)
                except Exception:
                    target_max_side = 960
                if int(target_max_side) > 0 and max_side > int(target_max_side):
                    scale = float(target_max_side) / float(max_side)
                    nw = max(1, int(round(float(w0) * scale)))
                    nh = max(1, int(round(float(h0) * scale)))
                    im2 = im.resize((nw, nh), resample=Image.BILINEAR)
                    tmp_sup_path = str((self._run_dir / f"step_{int(step_id):06d}_sup.jpg").resolve())
                    im2.save(tmp_sup_path, quality=82)
                    sup_path = tmp_sup_path
        except Exception:
            tmp_sup_path = ""
            sup_path = screenshot_path

        if state == "Unknown":
            prompt = (
                "You are a screen supervisor for Blue Archive (PC).\n"
                "Task: Classify the current screen state.\n\n"
                "Possible states: Lobby, Cafe_Inside, Battle_In_Progress, Result_Screen, Reward_Popup, Popup, Maintenance, Loading, Unknown.\n"
                "If you see a popup blocking the view, state is 'Popup' and suggested_recovery is 'close_popup'.\n"
                "If you see maintenance/servers down, state is 'Maintenance' and suggested_recovery is 'restart_game'.\n"
                "Return JSON ONLY:\n"
                "{\n"
                "  \"ok\": true,\n"
                "  \"state\": \"Lobby|Cafe_Inside|Battle_In_Progress|Result_Screen|Reward_Popup|Popup|Maintenance|Loading|Unknown\",\n"
                "  \"confidence\": 0.0,\n"
                "  \"reason\": \"short\",\n"
                "  \"suggested_recovery\": \"none|close_popup|nav_home|restart_game\"\n"
                "}\n"
            )
        else:
            prompt = (
                "You are a screen supervisor for Blue Archive (PC).\n"
                f"Task: Verify if current screen matches: [{state}].\n\n"
                "Possible states: Lobby, Cafe_Inside, Battle_In_Progress, Result_Screen, Reward_Popup, Popup, Maintenance, Loading, Unknown.\n"
                "If you see a popup blocking the view, state is 'Popup' and suggested_recovery is 'close_popup'.\n"
                "If you see maintenance/servers down, state is 'Maintenance' and suggested_recovery is 'restart_game'.\n"
                "Return JSON ONLY:\n"
                "{\n"
                "  \"ok\": true/false,\n"
                "  \"state\": \"Lobby|Cafe_Inside|Battle_In_Progress|Result_Screen|Reward_Popup|Popup|Maintenance|Loading|Unknown\",\n"
                "  \"confidence\": 0.0,\n"
                "  \"reason\": \"short\",\n"
                "  \"suggested_recovery\": \"none|close_popup|nav_home|restart_game\"\n"
                "}\n"
            )

        self._set_stage(step_id=int(step_id), stage="supervise")
        engine = get_local_vlm(
            model=self.cfg.model,
            models_dir=self.cfg.models_dir,
            hf_home=self.cfg.hf_home,
            device=self.cfg.device,
        )
        try:
            if hasattr(engine, "ensure_loaded"):
                engine.ensure_loaded()
        except Exception:
            pass

        t0 = time.time()
        raw = ""
        err = ""
        try:
            res = engine.ocr(
                image_path=sup_path,
                prompt=prompt,
                max_new_tokens=int(getattr(self.cfg, "supervision_max_new_tokens", 128) or 128),
            )
            raw = str(res.get("raw") or "")
            err = str(res.get("error") or "")
        except Exception as e:
            raw = ""
            err = str(e)
        try:
            dt = time.time() - t0
            ts = datetime.now().isoformat(timespec="seconds")
            self._log_out(f"[{ts}] step={int(step_id)} stage=supervise_end elapsed_s={dt:.2f}")
        except Exception:
            pass
        try:
            if tmp_sup_path:
                Path(tmp_sup_path).unlink(missing_ok=True)
        except Exception:
            pass

        out: Dict[str, Any] = {"raw": raw}
        if err:
            out["error"] = err
        try:
            parsed = _parse_json_content(raw)
            if isinstance(parsed, dict):
                out.update(parsed)
        except Exception:
            pass
        out.setdefault("ok", True)
        out.setdefault("state", "Unknown")
        out.setdefault("suggested_recovery", "none")
        out["expected_state"] = state
        return out

    def _supervision_to_recovery_action(self, *, sup: Dict[str, Any], screenshot_path: str) -> Dict[str, Any]:
        try:
            sug = str(sup.get("suggested_recovery") or "none").strip().lower()
        except Exception:
            sug = "none"
        try:
            seen = str(sup.get("state") or "Unknown").strip()
        except Exception:
            seen = "Unknown"

        if sug == "nav_home":
            return {
                "action": "back",
                "reason": f"Supervisor recovery: nav_home (seen={seen}).",
                "_supervision": sup,
            }

        if sug == "restart_game":
            return {
                "action": "stop",
                "reason": f"Supervisor recovery: restart_game requested (seen={seen}).",
                "_supervision": sup,
            }

        if sug == "close_popup":
            return {
                "action": "wait",
                "duration_ms": 200,
                "reason": "delegate: close_notice",
                "_supervision": sup,
                "_supervision_recovery": f"close_popup (seen={seen})",
            }

        return {
            "action": "wait",
            "duration_ms": 500,
            "reason": f"Supervisor recovery: none (seen={seen}).",
            "_supervision": sup,
        }

    def _run_loop(self) -> None:
        os.environ.setdefault("HF_HOME", self.cfg.hf_home)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", self.cfg.hf_home)
        try:
            os.environ.pop("TRANSFORMERS_CACHE", None)
        except Exception:
            pass

        i = 0
        while not self._stop.is_set():
            t0 = time.time()
            ts = datetime.now().isoformat(timespec="seconds")
            step_id = i
            shot_path = str((self._run_dir / f"step_{step_id:06d}.png").resolve())

            extra_sleep_s = 0.0

            sup_any = None

            vlm_called = False

            self._sync_routine_from_goal(self.cfg.goal)
            try:
                self._routine.on_turn_start()
            except Exception:
                pass

            err = ""
            act: Optional[Dict[str, Any]] = None
            act_before: Optional[Dict[str, Any]] = None
            try:
                self._set_stage(step_id=step_id, stage="screenshot")
                self._device.screenshot_client(shot_path)
                self._screenshot_fail_count = 0

                if act is None:
                    cb = None
                    try:
                        if bool(getattr(self.cfg, "cerebellum_enabled", True)) and getattr(self, "_cerebellum", None) is not None:
                            self._set_stage(step_id=step_id, stage="cerebellum")
                        cb = self._maybe_cerebellum_action(screenshot_path=shot_path, step_id=int(step_id))
                    except Exception:
                        cb = None
                    if isinstance(cb, dict) and str(cb.get("action") or ""):
                        act = cb

                try:
                    if act is None and not bool(getattr(self, "_startup_finished", False)):
                        if int(getattr(self, "_startup_tap_attempts", 0) or 0) <= 0 and int(step_id) <= 60:
                            act = {
                                "action": "wait",
                                "duration_ms": 900,
                                "reason": "Startup: waiting for game to become interactive (tap-to-start not seen yet).",
                                "_startup_tap": True,
                            }
                except Exception:
                    pass

                try:
                    if act is None and self._expected_state:
                        now = float(time.time())
                        if now and (now - float(self._expected_set_ts)) < float(self._expected_delay_s):
                            act = {
                                "action": "wait",
                                "duration_ms": 650,
                                "reason": f"Waiting for transition before supervisor check (expected={self._expected_state}).",
                            }
                except Exception:
                    pass

                if act is None and self._should_run_supervision(step_id=int(step_id)):
                    try:
                        self._last_supervision_step = int(step_id)
                    except Exception:
                        pass
                    vlm_called = True
                    sup = self._supervise(screenshot_path=shot_path, expected_state=str(self._expected_state or "Lobby"), step_id=int(step_id))
                    try:
                        self._last_supervision_any_step = int(step_id)
                    except Exception:
                        pass
                    seen = "Unknown"
                    try:
                        seen = str(sup.get("state") or "Unknown").strip()
                    except Exception:
                        seen = "Unknown"
                    try:
                        seen_low = str(seen).lower().strip()
                    except Exception:
                        seen_low = str(seen)

                    expected0 = ""
                    try:
                        expected0 = str(sup.get("expected_state") or self._expected_state or "").strip()
                    except Exception:
                        expected0 = ""
                    try:
                        expected_low = str(expected0).lower().strip()
                    except Exception:
                        expected_low = str(expected0)

                    conf = 0.0
                    try:
                        conf = float(sup.get("confidence") or 0.0)
                    except Exception:
                        conf = 0.0

                    passed = False
                    try:
                        if expected_low == "lobby":
                            passed = seen_low in ("lobby", "title")
                        elif expected_low and expected_low != "unknown":
                            passed = seen_low == expected_low
                    except Exception:
                        passed = False

                    if (not passed) and expected_low and expected_low != "unknown" and seen_low == "loading":
                        self._supervision_fail_count = 0
                        try:
                            if self._expected_state:
                                self._set_expected_state(self._expected_state, delay_s=1.2)
                        except Exception:
                            pass
                        act = {
                            "action": "wait",
                            "duration_ms": 650,
                            "reason": f"Supervisor: still Loading while waiting for {expected0 or self._expected_state}.",
                            "_supervision": sup,
                        }
                    elif passed and float(conf) >= 0.60:
                        self._supervision_fail_count = 0
                        self._set_expected_state(None)

                        try:
                            if not bool(getattr(self, "_startup_finished", False)) and int(getattr(self, "_startup_tap_attempts", 0) or 0) > 0:
                                st = str(sup.get("state") or "").strip().lower()
                                if st in ("lobby", "title") and float(conf) >= 0.85:
                                    sw0 = sh0 = 0
                                    try:
                                        with Image.open(shot_path) as im:
                                            sw0, sh0 = im.size
                                    except Exception:
                                        sw0, sh0 = 0, 0

                                    act2 = None
                                    try:
                                        c2 = getattr(self, "_cerebellum", None)
                                        if c2 is not None and bool(getattr(self.cfg, "cerebellum_enabled", True)) and sw0 > 0 and sh0 > 0:
                                            roi_start = (
                                                int(round(float(sw0) * 0.40)),
                                                int(round(float(sh0) * 0.74)),
                                                int(round(float(sw0) * 0.60)),
                                                int(round(float(sh0) * 0.92)),
                                            )
                                            act2 = c2.click_action(
                                                screenshot_path=shot_path,
                                                template_name="点击开始.png",
                                                reason_prefix="Cerebellum: tap to start (supervisor-confirmed).",
                                                roi=roi_start,
                                            )
                                            if isinstance(act2, dict):
                                                try:
                                                    cb = act2.get("_cerebellum", {})
                                                    if float(cb.get("score") or 0.0) < 0.90:
                                                        act2 = None
                                                except Exception:
                                                    pass
                                    except Exception:
                                        act2 = None

                                    if isinstance(act2, dict):
                                        act2["_startup_tap"] = True
                                        act2["_supervision"] = sup
                                        try:
                                            self._last_tap_to_start_step = int(step_id)
                                        except Exception:
                                            pass
                                        act = act2
                                    elif sw0 > 0 and sh0 > 0:
                                        x = int(round(float(sw0) * 0.50))
                                        y = int(round(float(sh0) * 0.82))
                                        x = max(0, min(int(sw0) - 1, int(x)))
                                        y = max(0, min(int(sh0) - 1, int(y)))
                                        try:
                                            self._last_tap_to_start_step = int(step_id)
                                            self._last_tap_to_start_xy = (int(x), int(y))
                                        except Exception:
                                            pass
                                        act = {
                                            "action": "click",
                                            "target": [int(x), int(y)],
                                            "reason": "Startup: supervisor indicates tap-to-start screen; clicking bottom-center to start.",
                                            "_startup_tap": True,
                                            "_supervision": sup,
                                        }
                        except Exception:
                            pass

                        if act is None:
                            act = {
                                "action": "wait",
                                "duration_ms": 220,
                                "reason": f"Supervisor check passed (state={sup.get('state')}).",
                                "_supervision": sup,
                            }
                    else:
                        self._supervision_fail_count = int(self._supervision_fail_count) + 1
                        act = self._supervision_to_recovery_action(sup=sup, screenshot_path=shot_path)
                        try:
                            if self._expected_state:
                                self._set_expected_state(self._expected_state, delay_s=1.2)
                        except Exception:
                            pass
                        try:
                            if int(self._supervision_fail_count) >= int(getattr(self.cfg, "supervision_fail_escalate_n", 3) or 3):
                                act = {
                                    "action": "stop",
                                    "reason": "Supervisor recovery failed repeatedly; stopping agent.",
                                    "_supervision": sup,
                                }
                        except Exception:
                            pass

                self._set_stage(step_id=step_id, stage="decide")
                if act is None:

                    try:
                        if not bool(getattr(self, "_startup_finished", False)):
                            if int(getattr(self, "_startup_tap_attempts", 0) or 0) <= 0 and int(step_id) <= 60:
                                act = {
                                    "action": "wait",
                                    "duration_ms": 900,
                                    "reason": "Startup: waiting for game to become interactive (tap-to-start not seen yet).",
                                    "_startup_tap": True,
                                }
                    except Exception:
                        pass

                if act is None:

                    fast_close = None
                    boot_preinteractive = False
                    try:
                        boot_preinteractive = (not bool(getattr(self, "_startup_finished", False))) and int(getattr(self, "_startup_tap_attempts", 0) or 0) <= 0
                    except Exception:
                        boot_preinteractive = False

                    try:
                        c = getattr(self, "_cerebellum", None)
                    except Exception:
                        c = None

                    try:
                        if (not boot_preinteractive) and c is not None and bool(getattr(self.cfg, "cerebellum_enabled", True)):
                            with Image.open(shot_path) as im:
                                sw0, sh0 = im.size
                            if sw0 > 0 and sh0 > 0:
                                try:
                                    if int(step_id) - int(getattr(self, "_last_cerebellum_notice_step", -10_000) or -10_000) <= 2:
                                        raise RuntimeError("fast_close cooldown")
                                    if int(getattr(self, "_cerebellum_notice_streak", 0) or 0) >= 2:
                                        raise RuntimeError("fast_close disabled by streak")
                                except Exception:
                                    raise

                                roi_notice = (
                                    int(round(float(sw0) * 0.84)),
                                    int(round(float(sh0) * 0.00)),
                                    int(sw0),
                                    int(round(float(sh0) * 0.18)),
                                )

                                def _uniq(names: list[str]) -> list[str]:
                                    out: list[str] = []
                                    seen: set[str] = set()
                                    for n in names:
                                        nn = str(n or "").strip()
                                        if not nn or nn in seen:
                                            continue
                                        out.append(nn)
                                        seen.add(nn)
                                    return out

                                tmpl0 = str(getattr(self.cfg, "cerebellum_template_notice_close", "notice_close.png") or "")
                                for tmpl in _uniq([tmpl0, "内嵌公告的叉.png", "游戏内很多页面窗口的叉.png"]):
                                    act2 = c.click_action(
                                        screenshot_path=shot_path,
                                        template_name=tmpl,
                                        reason_prefix="Cerebellum(fast): close notice/webview.",
                                        roi=roi_notice,
                                    )
                                    if isinstance(act2, dict):
                                        try:
                                            cb = act2.get("_cerebellum", {})
                                            if float(cb.get("score") or 0.0) < 0.985:
                                                continue
                                        except Exception:
                                            pass
                                        try:
                                            ctr = act2.get("_cerebellum", {}).get("center")
                                            if isinstance(ctr, (list, tuple)) and len(ctr) == 2:
                                                cx, cy = int(ctr[0]), int(ctr[1])
                                                if int(cx) < int(round(float(sw0) * 0.88)):
                                                    continue
                                                if int(cy) > int(round(float(sh0) * 0.14)):
                                                    continue
                                        except Exception:
                                            pass
                                        act2["raw"] = ""
                                        act2["_close_heuristic"] = "cerebellum_notice_close"
                                        fast_close = act2
                                        try:
                                            if int(getattr(self, "_startup_tap_attempts", 0) or 0) > 0:
                                                self._startup_finished = True
                                        except Exception:
                                            pass
                                        try:
                                            prev = int(getattr(self, "_last_cerebellum_notice_step", -10_000) or -10_000)
                                            if int(step_id) == int(prev) + 1:
                                                self._cerebellum_notice_streak = int(getattr(self, "_cerebellum_notice_streak", 0) or 0) + 1
                                            else:
                                                self._cerebellum_notice_streak = 1
                                            self._last_cerebellum_notice_step = int(step_id)
                                        except Exception:
                                            pass
                                        break
                    except Exception:
                        fast_close = None

                    vlm_path = shot_path
                    tmp_vlm_path = ""
                    orig_size: Optional[Tuple[int, int]] = None
                    try:
                        with Image.open(shot_path) as im:
                            ow, oh = im.size
                            orig_size = (int(ow), int(oh))
                            max_side = max(int(ow), int(oh))
                            try:
                                target_max_side = int(getattr(self.cfg, "vlm_image_max_side", 960))
                            except Exception:
                                target_max_side = 960
                            try:
                                if int(target_max_side) <= 0 and int(step_id) <= int(getattr(self, "_vlm_force_small_until_step", -10_000)):
                                    target_max_side = int(getattr(self.cfg, "supervision_image_max_side", 960) or 960)
                            except Exception:
                                pass
                            if int(target_max_side) > 0 and max_side > int(target_max_side):
                                scale = float(target_max_side) / float(max_side)
                                nw = max(1, int(round(float(ow) * scale)))
                                nh = max(1, int(round(float(oh) * scale)))
                                im2 = im.resize((nw, nh), resample=Image.BILINEAR)
                                tmp_vlm_path = str((self._run_dir / f"step_{step_id:06d}_vlm.jpg").resolve())
                                im2.save(tmp_vlm_path, quality=92)
                                vlm_path = tmp_vlm_path
                                try:
                                    ts2 = datetime.now().isoformat(timespec="seconds")
                                    self._log_out(
                                        f"[{ts2}] step={int(step_id)} vlm_input_resize orig=[{int(ow)},{int(oh)}] resized=[{int(nw)},{int(nh)}] max_side={int(target_max_side)}"
                                    )
                                except Exception:
                                    pass
                            else:
                                try:
                                    ts2 = datetime.now().isoformat(timespec="seconds")
                                    self._log_out(
                                        f"[{ts2}] step={int(step_id)} vlm_input_no_resize size=[{int(ow)},{int(oh)}] max_side={int(target_max_side)}"
                                    )
                                except Exception:
                                    pass
                    except Exception:
                        tmp_vlm_path = ""
                        vlm_path = shot_path

                    try:
                        if fast_close is not None:
                            act = fast_close
                        else:
                            vlm_called = True
                            act = self._decide(screenshot_path=vlm_path, step_id=step_id, orig_size=orig_size)
                    finally:
                        try:
                            if tmp_vlm_path:
                                Path(tmp_vlm_path).unlink(missing_ok=True)
                        except Exception:
                            pass

                act_before = act
                act = self._sanitize_action(act)
                act = self._maybe_delegate_notice_close_to_cerebellum(act, screenshot_path=shot_path, step_id=step_id)
                act = self._maybe_close_popup_heuristic(act, step_id=step_id, screenshot_path=shot_path)
                act = self._maybe_delegate_intent_to_cerebellum(act, screenshot_path=shot_path, step_id=step_id)
                act = self._maybe_tap_to_start(act, step_id=step_id)
                act = self._block_startup_vlm_clicks(act, step_id=step_id, screenshot_path=shot_path)
                act = self._block_check_lobby_noise(act)
                act = self._handle_stuck_in_recruit(act)
                act = self._maybe_recover_cafe_wrong_screen(act)
                act = self._snap_click_to_perception_label(act)
                act = self._maybe_force_cafe_nav(act)
                act = self._maybe_cafe_actions(act, screenshot_path=shot_path, step_id=step_id)
                act = self._maybe_cafe_headpat(action=act, screenshot_path=shot_path, step_id=step_id)
                act = self._maybe_cafe_idle_exit(action=act, screenshot_path=shot_path, step_id=step_id)
                act = self._maybe_autoclick_safe_button(action=act, action_before=act_before, step_id=step_id, screenshot_path=shot_path)
                act = self._sanitize_action(act)
                act = self._debounce_click(act, step_id=step_id)
                act = self._maybe_exploration_click(action=act, action_before=act_before, screenshot_path=shot_path, step_id=step_id)
                act = self._sanitize_action(act)
                act = self._maybe_advance_routine(act)

                try:
                    if (not vlm_called) and sup_any is None and self._should_run_supervision_any(step_id=int(step_id)):
                        sup_any = self._supervise(screenshot_path=shot_path, expected_state="Unknown", step_id=int(step_id))
                        try:
                            self._last_supervision_any_step = int(step_id)
                        except Exception:
                            pass
                except Exception:
                    sup_any = None

                try:
                    if sup_any is not None and isinstance(act, dict):
                        act["_supervision_any"] = sup_any
                except Exception:
                    pass
                try:
                    self._maybe_set_expected_state_from_action(act)
                except Exception:
                    pass
                self._set_stage(step_id=step_id, stage="execute")
                self._execute(act, screenshot_path=shot_path, step_id=step_id)
                with self._lock:
                    self._last_action = act
                    self._last_error = ""
            except Exception as e:
                err = str(e)
                with self._lock:
                    self._last_error = err
                try:
                    low = err.lower()
                except Exception:
                    low = err
                if str(err).strip() == "1400" or "1400" in str(err) or "window not found" in low or "invalid" in low:
                    self._screenshot_fail_count += 1
                    n = min(6, max(0, int(self._screenshot_fail_count) - 1))
                    extra_sleep_s = float(min(10.0, 0.5 * (2 ** n)))
                    if self._screenshot_fail_count >= 30:
                        self._stop.set()

            try:
                rel_shot = str(Path(shot_path).relative_to(Path.cwd())).replace("\\", "/")
            except Exception:
                rel_shot = shot_path

            rec = {
                "ts": ts,
                "step": step_id,
                "screenshot": rel_shot,
                "dry_run": bool(self.cfg.dry_run),
                "goal": self.cfg.goal,
                "action": act,
                "error": err,
                "elapsed_s": round(time.time() - t0, 3),
            }
            self._write_traj(rec)

            usage = {
                "ts": rec["ts"],
                "run_id": self._run_id,
                "step": step_id,
                "window_title": self.cfg.window_title,
                "goal": self.cfg.goal,
                "dry_run": bool(self.cfg.dry_run),
                "step_sleep_s": float(self.cfg.step_sleep_s),
                "screenshot": rel_shot,
                "elapsed_s": rec["elapsed_s"],
                "error": err,
                "supervision_any": (act or {}).get("_supervision_any") if isinstance(act, dict) else None,
                "model": (act or {}).get("_model") if isinstance(act, dict) else None,
                "perception": (act_before or act or {}).get("_perception") if isinstance((act_before or act), dict) else None,
                "policy_prompt": (act_before or act or {}).get("_prompt") if isinstance((act_before or act), dict) else None,
                "policy_raw": (act_before or act or {}).get("raw") if isinstance((act_before or act), dict) else None,
                "action_model": act_before,
                "action_final": act,
                "blocked": bool(isinstance(act, dict) and act.get("_blocked")),
            }
            self._write_usage(usage)

            try:
                if isinstance(act, dict):
                    self._recent.append(
                        {
                            "step": step_id,
                            "action": str(act.get("action") or ""),
                            "reason": str(act.get("reason") or ""),
                            "blocked": bool(act.get("_blocked")),
                            "autoclick": bool(act.get("_autoclick")),
                            "exploration": bool(act.get("_exploration")),
                        }
                    )
                    if len(self._recent) > 50:
                        self._recent = self._recent[-50:]
            except Exception:
                pass

            if act is not None:
                self._log_out(f"[{rec['ts']}] step={step_id} dry_run={int(self.cfg.dry_run)} action={json.dumps(act, ensure_ascii=False)[:600]}")
            if err:
                self._log_err(f"[{rec['ts']}] step={step_id} error={err}")

            i += 1
            if self.cfg.steps and i >= self.cfg.steps:
                break

            time.sleep(max(0.0, float(self.cfg.step_sleep_s)) + max(0.0, float(extra_sleep_s)))

        self._stop.set()
