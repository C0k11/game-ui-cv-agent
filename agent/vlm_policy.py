import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, List

from PIL import Image

from action.windows_input import WindowsInput
from agent.daily_routine import DailyRoutineManager
from config import HF_CACHE_DIR, LOCAL_VLM_DEVICE, LOCAL_VLM_MAX_NEW_TOKENS, LOCAL_VLM_MODEL, LOCAL_VLM_MODELS_DIR
from vision.local_vlm_runtime import get_local_vlm


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


@dataclass
class VlmPolicyConfig:
    window_title: str = "Blue Archive"
    goal: str = "Keep the game running safely."
    steps: int = 0
    dry_run: bool = True
    step_sleep_s: float = 0.6

    model: str = LOCAL_VLM_MODEL
    models_dir: str = LOCAL_VLM_MODELS_DIR
    hf_home: str = HF_CACHE_DIR
    device: str = LOCAL_VLM_DEVICE
    max_new_tokens: int = LOCAL_VLM_MAX_NEW_TOKENS

    perception_max_new_tokens: int = 128
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

        self._last_click_step: int = -10_000
        self._last_click_xy: Optional[Tuple[int, int]] = None
        self._last_click_reason: str = ""

        self._screenshot_fail_count: int = 0

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

    def _set_stage(self, *, step_id: int, stage: str, detail: str = "") -> None:
        try:
            a: Dict[str, Any] = {"action": "_stage", "step": int(step_id), "stage": str(stage)}
            if detail:
                a["detail"] = str(detail)
            with self._lock:
                self._last_action = a
            try:
                ts = datetime.now().isoformat(timespec="seconds")
                self._log_out(f"[{ts}] step={int(step_id)} stage={str(stage)}")
            except Exception:
                pass
        except Exception:
            pass

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

    def _routine_should_enable(self, goal: str) -> bool:
        g = (goal or "").strip()
        if not g:
            return False
        gl = g.lower()
        return (
            "[routine]" in gl
            or "daily routine" in gl
            or "自动收菜" in g
            or "收菜" in g
            or "日常" in g
        )

    def _sync_routine_from_goal(self, goal: str) -> None:
        try:
            enable = self._routine_should_enable(goal)
        except Exception:
            enable = False

        if enable and not bool(self._routine.is_active):
            self._routine.start_routine()
        if (not enable) and bool(self._routine.is_active):
            self._routine.stop_routine()

    def _routine_meta(self) -> Dict[str, Any]:
        step = None
        try:
            step = self._routine.get_current_step()
        except Exception:
            step = None
        total_steps = 0
        try:
            total_steps = len(getattr(self._routine, "steps", []) or [])
        except Exception:
            total_steps = 0
        return {
            "active": bool(self._routine.is_active),
            "progress": self._routine.get_progress_str(),
            "step_index": int(self._routine.current_step_index) if bool(self._routine.is_active) else -1,
            "step_name": (step.name if step is not None else ""),
            "total_steps": int(total_steps),
            "in_recovery_mode": bool(getattr(self._routine, "in_recovery_mode", False)),
            "turns_in_current_attempt": int(getattr(self._routine, "turns_in_current_attempt", 0)),
            "max_turns_per_attempt": int(getattr(self._routine, "max_turns_per_attempt", 0)),
            "current_step_retry_count": int(getattr(self._routine, "current_step_retry_count", 0)),
            "max_retries": int(getattr(self._routine, "max_retries", 0)),
        }

    def _routine_context(self) -> str:
        try:
            return str(self._routine.get_prompt_block() or "")
        except Exception:
            return ""

    def _maybe_advance_routine(self, action: Dict[str, Any]) -> Dict[str, Any]:
        if not bool(self._routine.is_active):
            return action
        a = str(action.get("action") or "").lower().strip()
        if a not in ("done", "finish", "complete", "next_step"):
            return action

        prev = None
        try:
            prev = self._routine.get_current_step()
        except Exception:
            prev = None

        cont = False
        try:
            cont = bool(self._routine.handle_done_signal())
        except Exception:
            cont = False

        nm = prev.name if prev is not None else "(unknown)"
        if not bool(self._routine.is_active):
            return {
                "action": "wait",
                "duration_ms": 800,
                "reason": f"Routine all complete (last phase: {nm}). You can stop now.",
                "raw": action.get("raw"),
                "_prompt": action.get("_prompt"),
                "_perception": action.get("_perception"),
                "_model": action.get("_model"),
                "_routine": self._routine_meta(),
                "_routine_advanced": True,
            }

        step_name = "Next Step"
        try:
            if bool(getattr(self._routine, "in_recovery_mode", False)):
                step_name = "Recovery (Back to Lobby)"
            else:
                cur = self._routine.get_current_step()
                if cur is not None:
                    step_name = cur.name
        except Exception:
            step_name = "Next Step"

        return {
            "action": "wait",
            "duration_ms": 500,
            "reason": f"Routine transition after '{nm}': {step_name}",
            "raw": action.get("raw"),
            "_prompt": action.get("_prompt"),
            "_perception": action.get("_perception"),
            "_model": action.get("_model"),
            "_routine": self._routine_meta(),
            "_routine_advanced": True,
        }

    def _negative_reason(self, reason: str) -> bool:
        r = (reason or "").lower()
        if not r:
            return False
        bad = [
            "avoid",
            "do not",
            "don't",
            "should not",
            "not aligned",
            "not safe",
            "unsafe",
            "不会",
            "不要",
            "避免",
            "不建议",
            "不应该",
            "不符合",
        ]
        return any(x in r for x in bad)

    def _safe_label(self, label: str) -> bool:
        t = (label or "").strip().lower()
        if not t:
            return False
        good = [
            "start",
            "ok",
            "yes",
            "confirm",
            "continue",
            "next",
            "skip",
            "claim",
            "collect",
            "receive",
            "free",
            "开始",
            "进入",
            "继续",
            "确认",
            "确定",
            "好的",
            "领取",
            "全部领取",
            "一键领取",
            "扫荡",
            "免费",
            "使用",
        ]
        if any(x in t for x in good):
            return True
        return False

    def _raid_label(self, label: str) -> bool:
        t = (label or "").strip().lower()
        if not t:
            return False
        bad = [
            "total assault",
            "grand assault",
            "assault",
            "raid",
            "setlik",
            "总力战",
            "大决战",
            "制约解除",
            "制约解除决战",
        ]
        return any(x in t for x in bad)

    def _raid_present(self, items: Any) -> bool:
        if not isinstance(items, list):
            return False
        for it in items:
            if not isinstance(it, dict):
                continue
            label = str(it.get("label") or "")
            if self._raid_label(label):
                return True
        return False

    def _maybe_autoclick_safe_button(self, *, action: Dict[str, Any], action_before: Optional[Dict[str, Any]], step_id: int) -> Dict[str, Any]:
        if not self.cfg.autoclick_safe_buttons:
            return action

        try:
            cooldown = int(self.cfg.autoclick_safe_cooldown_steps)
        except Exception:
            cooldown = 2

        if step_id - int(self._last_autoclick_step) < max(0, cooldown):
            return action

        if str(action.get("action") or "").lower().strip() != "wait":
            return action

        items = None
        try:
            items = (action_before or action).get("_perception", {}).get("items")
        except Exception:
            items = None

        if not isinstance(items, list) or not items:
            return action

        if self._raid_present(items):
            return action

        for it in items:
            if not isinstance(it, dict):
                continue
            label = str(it.get("label") or "").strip()
            bbox = it.get("bbox")
            if not label:
                continue
            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                continue
            if self._dangerous_label(label):
                continue
            if self.cfg.forbid_premium_currency and self._premium_label(label):
                continue
            if not self._safe_label(label):
                continue

            try:
                x, y = _center([int(v) for v in bbox])
            except Exception:
                continue

            self._last_autoclick_step = int(step_id)
            return {
                "action": "click",
                "target": [int(x), int(y)],
                "reason": f"Auto-click safe button detected by OCR: {label}",
                "raw": (action_before or action).get("raw"),
                "_prompt": (action_before or action).get("_prompt"),
                "_perception": (action_before or action).get("_perception"),
                "_model": (action_before or action).get("_model"),
                "_autoclick": True,
            }

        return action

    def _dangerous_label(self, label: str) -> bool:
        t = (label or "").strip().lower()
        if not t:
            return False
        if self._raid_label(t):
            return True
        bad = [
            "stop",
            "exit",
            "close",
            "quit",
            "cancel",
            "log out",
            "logout",
            "terminate",
            "停止",
            "退出",
            "关闭",
            "取消",
            "离开",
            "结束",
            "登出",
        ]
        return any(x in t for x in bad)

    def _premium_label(self, label: str) -> bool:
        t = (label or "").strip().lower()
        if not t:
            return False
        bad = [
            "pyroxene",
            "青辉石",
            "清輝石",
            "充值",
            "儲值",
            "recharge",
            "top up",
            "topup",
            "purchase",
            "buy",
            "購買",
            "购买",
            "購入",
            "課金",
            "招募",
            "recruit",
            "gacha",
            "扭蛋",
        ]
        return any(x in t for x in bad)

    def _premium_reason(self, reason: str) -> bool:
        r = (reason or "").lower()
        if not r:
            return False
        bad = [
            "pyroxene",
            "青辉石",
            "清輝石",
            "充值",
            "儲值",
            "top up",
            "topup",
            "recharge",
            "purchase",
            "buy",
            "購買",
            "购买",
            "課金",
            "招募",
            "recruit",
            "gacha",
            "抽",
            "抽卡",
        ]
        return any(x in r for x in bad)

    def _bbox_contains(self, bbox: Any, x: int, y: int) -> bool:
        try:
            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                return False
            x1, y1, x2, y2 = [int(v) for v in bbox]
            return x1 <= x <= x2 and y1 <= y <= y2
        except Exception:
            return False

    def _expanded_bbox_contains(self, bbox: Any, x: int, y: int, pad: int = 6) -> bool:
        try:
            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                return False
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1 -= int(pad)
            y1 -= int(pad)
            x2 += int(pad)
            y2 += int(pad)
            return x1 <= x <= x2 and y1 <= y <= y2
        except Exception:
            return False

    def _dangerous_bboxes(self, items: Any) -> List[List[int]]:
        out: List[List[int]] = []
        if not isinstance(items, list):
            return out
        for it in items:
            if not isinstance(it, dict):
                continue
            label = str(it.get("label") or "")
            bbox = it.get("bbox")
            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                continue
            if self._dangerous_label(label):
                try:
                    out.append([int(v) for v in bbox])
                except Exception:
                    continue
        return out

    def _maybe_exploration_click(self, *, action: Dict[str, Any], action_before: Optional[Dict[str, Any]], screenshot_path: str, step_id: int) -> Dict[str, Any]:
        if not self.cfg.exploration_click:
            return action

        try:
            cooldown = int(self.cfg.exploration_click_cooldown_steps)
        except Exception:
            cooldown = 3

        if step_id - int(self._last_exploration_step) < max(0, cooldown):
            return action

        if str(action.get("action") or "").lower().strip() != "wait":
            return action

        items = None
        try:
            items = (action_before or action).get("_perception", {}).get("items")
        except Exception:
            items = None

        if not isinstance(items, list) or not items:
            return action

        has_danger = False
        has_non_danger = False
        for it in items:
            if not isinstance(it, dict):
                continue
            label = str(it.get("label") or "")
            if not label:
                continue
            if self._dangerous_label(label):
                has_danger = True
            else:
                has_non_danger = True

        if not has_danger or has_non_danger:
            return action

        with Image.open(screenshot_path) as im:
            w, h = im.size

        danger_bboxes = self._dangerous_bboxes(items)
        candidates = [
            (0.5, 0.72),
            (0.5, 0.55),
            (0.5, 0.85),
            (0.25, 0.72),
            (0.75, 0.72),
        ]

        chosen: Optional[Tuple[int, int]] = None
        for rx, ry in candidates:
            x = int(max(0, min(w - 1, round(float(rx) * w))))
            y = int(max(0, min(h - 1, round(float(ry) * h))))
            bad = False
            for bb in danger_bboxes:
                if self._expanded_bbox_contains(bb, x, y, pad=10):
                    bad = True
                    break
            if not bad:
                chosen = (x, y)
                break

        if chosen is None:
            return action

        self._last_exploration_step = int(step_id)
        x, y = chosen
        return {
            "action": "click",
            "target": [int(x), int(y)],
            "reason": "Exploration click: only dangerous UI elements detected (e.g. stop/exit), so click a safe empty area to try to advance.",
            "raw": (action_before or action).get("raw"),
            "_prompt": (action_before or action).get("_prompt"),
            "_perception": (action_before or action).get("_perception"),
            "_model": (action_before or action).get("_model"),
            "_exploration": True,
        }

    def _sanitize_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        a = str(action.get("action", "")).lower().strip()
        reason = str(action.get("reason") or "")

        items = None
        try:
            items = action.get("_perception", {}).get("items")
        except Exception:
            items = None

        if self._raid_present(items) and a in ("click", "swipe"):
            return {
                "action": "back",
                "reason": "Raid content detected (Total Assault/Grand Assault/制约解除决战). Leaving via back.",
                "raw": action.get("raw"),
                "_prompt": action.get("_prompt"),
                "_perception": action.get("_perception"),
                "_model": action.get("_model"),
                "_blocked": True,
            }

        if a == "click":
            x, y = -1, -1
            tgt = action.get("target")
            if isinstance(tgt, (list, tuple)) and len(tgt) == 2:
                try:
                    x, y = int(tgt[0]), int(tgt[1])
                except Exception:
                    x, y = -1, -1
            else:
                bb = action.get("bbox")
                if isinstance(bb, (list, tuple)) and len(bb) == 4:
                    try:
                        x, y = _center([int(v) for v in bb])
                    except Exception:
                        x, y = -1, -1

            if isinstance(items, list) and x >= 0 and y >= 0:
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    label = str(it.get("label") or "")
                    bbox = it.get("bbox")
                    if self._dangerous_label(label) and self._bbox_contains(bbox, x, y):
                        return {
                            "action": "wait",
                            "duration_ms": 1200,
                            "reason": f"Blocked click on potentially destructive UI element: {label}",
                            "raw": action.get("raw"),
                            "_prompt": action.get("_prompt"),
                            "_perception": action.get("_perception"),
                            "_blocked": True,
                        }

                    if self.cfg.forbid_premium_currency and self._premium_label(label) and self._bbox_contains(bbox, x, y):
                        return {
                            "action": "wait",
                            "duration_ms": 1200,
                            "reason": f"Blocked click related to premium currency spending: {label}",
                            "raw": action.get("raw"),
                            "_prompt": action.get("_prompt"),
                            "_perception": action.get("_perception"),
                            "_blocked": True,
                        }

            if self.cfg.forbid_premium_currency and self._premium_reason(reason):
                return {
                    "action": "wait",
                    "duration_ms": 1200,
                    "reason": "Blocked click because it appears to involve premium currency spending.",
                    "raw": action.get("raw"),
                    "_prompt": action.get("_prompt"),
                    "_perception": action.get("_perception"),
                    "_blocked": True,
                }

            if self._negative_reason(reason):
                return {
                    "action": "wait",
                    "duration_ms": 1200,
                    "reason": "Blocked click because the model reason indicates avoidance/unsafety.",
                    "raw": action.get("raw"),
                    "_prompt": action.get("_prompt"),
                    "_perception": action.get("_perception"),
                    "_blocked": True,
                }

        if a in ("stop",):
            if self._negative_reason(reason):
                return {
                    "action": "wait",
                    "duration_ms": 1200,
                    "reason": "Blocked stop because the model reason indicates avoidance/unsafety.",
                    "raw": action.get("raw"),
                    "_prompt": action.get("_prompt"),
                    "_perception": action.get("_perception"),
                    "_blocked": True,
                }

        return action

    @property
    def last_action(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._last_action

    @property
    def last_error(self) -> str:
        with self._lock:
            return self._last_error

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        try:
            self._out_log.write_text("", encoding="utf-8")
            self._err_log.write_text("", encoding="utf-8")
        except Exception:
            pass
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_loop, name="vlm_policy_agent", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=3)

    def is_running(self) -> bool:
        t = self._thread
        return t is not None and t.is_alive() and not self._stop.is_set()

    def _format_recent(self) -> str:
        try:
            n = int(self.cfg.prompt_history_steps)
        except Exception:
            n = 0
        if n <= 0:
            return ""

        tail = self._recent[-n:]
        lines = []
        for r in tail:
            try:
                step = int(r.get("step"))
            except Exception:
                step = -1
            a = str(r.get("action") or "")
            reason = str(r.get("reason") or "")
            if len(reason) > 120:
                reason = reason[:120]
            if step >= 0:
                lines.append(f"- step={step} action={a} reason={reason}")
            else:
                lines.append(f"- action={a} reason={reason}")
        if not lines:
            return ""
        return "Recent actions (do NOT repeat obvious loops; continue the routine):\n" + "\n".join(lines)

    def _prompt(self, *, width: int, height: int) -> str:
        goal = (self.cfg.goal or "").strip() or "Keep the game running safely."
        schema = (
            "Return JSON only. Allowed actions:\n"
            "- wait: {\\\"action\\\":\\\"wait\\\", \\\"duration_ms\\\":<int>, \\\"reason\\\":<string>}\n"
            "- click: {\\\"action\\\":\\\"click\\\", \\\"target\\\":[x,y], \\\"reason\\\":<string>}\n"
            "- click: {\\\"action\\\":\\\"click\\\", \\\"bbox\\\":[x1,y1,x2,y2], \\\"reason\\\":<string>}\n"
            "- done: {\\\"action\\\":\\\"done\\\", \\\"reason\\\":<string>}\n"
            "- {\"action\":\"swipe\", \"from\":[x1,y1], \"to\":[x2,y2], \"duration_ms\":int, \"reason\":string}\n"
            "- {\"action\":\"wait\", \"duration_ms\":int, \"reason\":string}\n"
            "- {\"action\":\"back\", \"reason\":string}\n"
            "- {\"action\":\"stop\", \"reason\":string}\n"
        )
        constraints = (
            f"Coordinates must be integer pixels in screenshot coordinates. width={width}, height={height}. "
            "Only click inside the window (0<=x<width, 0<=y<height). "
            "If uncertain, use wait. If you repeatedly choose wait, consider clicking a safe UI button such as Start/OK/Confirm/Skip/Next when detected."
        )
        rules = (
            "Safety rules:\n"
            "- NEVER click buttons that stop/exit/close/cancel the game or app.\n"
            "- If the screen is unclear, prefer wait.\n"
            "- If you decide to click, click a specific UI button (not random).\n"
            "- NEVER spend premium currency (Pyroxene / 青辉石 / 清輝石). Do not click recharge/top-up/buy/purchase/recruit/gacha actions.\n"
        )

        recent = self._format_recent()
        if recent:
            recent = "\n\n" + recent + "\n"
        routine_ctx = self._routine_context()
        return (
            f"{routine_ctx}"
            "You are an autonomous agent controlling a Windows game window via mouse and keyboard. "
            f"Goal: {goal}\n\n"
            f"{schema}\n\n"
            f"{rules}"
            f"\n\n{constraints}"
            f"{recent}"
            f"Image size: width={width}, height={height}."
        )

    def _perception_prompt(self, *, width: int, height: int) -> str:
        return (
            "Extract actionable UI text elements from the image. Return JSON only. "
            "Format: {\"items\": [{\"label\": <text>, \"bbox\": [x1,y1,x2,y2]}]}. "
            f"Image size: width={width}, height={height}. "
            "Do not include any extra keys. Do not wrap in markdown."
        )

    def _format_items(self, items: Sequence[Dict[str, Any]]) -> str:
        rows = []
        for it in items:
            try:
                label = str(it.get("label") or "").strip()
                bbox = it.get("bbox")
                if not label:
                    continue
                if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                    continue
                x1, y1, x2, y2 = [int(v) for v in bbox]
                rows.append((y1, x1, f"[{x1},{y1},{x2},{y2}] {label}"))
            except Exception:
                continue
        rows.sort(key=lambda t: (t[0], t[1]))
        lines = [r[2] for r in rows[: int(self.cfg.perception_max_items)]]
        return "\n".join(lines)

    def _snap_click_to_perception_label(self, action: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(action, dict):
            return action
        a = str(action.get("action") or "").lower().strip()
        if a != "click":
            return action

        items = None
        try:
            items = action.get("_perception", {}).get("items")
        except Exception:
            items = None
        if not isinstance(items, list) or not items:
            return action

        reason = str(action.get("reason") or "")
        phase = ""
        try:
            rm = action.get("_routine")
            if isinstance(rm, dict):
                phase = str(rm.get("phase") or "")
        except Exception:
            phase = ""

        # Heuristic label mapping for routine navigation.
        want: list[str] = []
        rlow = reason.lower()
        plow = phase.lower()
        if "cafe" in rlow or "咖啡" in reason or "cafe" in plow or "咖啡" in phase:
            want = ["cafe", "咖啡", "咖啡厅"]
        elif "task" in rlow or "任务" in reason:
            want = ["task", "任务"]
        elif "mail" in rlow or "邮箱" in reason:
            want = ["mail", "邮箱", "邮件"]
        elif "schedule" in rlow or "日程" in reason or "schedule" in plow or "日程" in phase:
            want = ["schedule", "日程"]
        elif "club" in rlow or "社团" in reason or "club" in plow or "社团" in phase:
            want = ["club", "社团"]
        elif "bounty" in rlow or "悬赏" in reason or "bounty" in plow or "悬赏" in phase:
            want = ["bounty", "wanted", "悬赏", "通缉"]

        if not want:
            return action

        best = None
        for it in items:
            if not isinstance(it, dict):
                continue
            label = str(it.get("label") or "").strip()
            bb = it.get("bbox")
            if not label or not (isinstance(bb, (list, tuple)) and len(bb) == 4):
                continue
            llow = label.lower()
            if any(k in llow or k in label for k in want):
                best = [int(v) for v in bb]
                break

        if best is None:
            return action

        x, y = _center([int(v) for v in best])
        out = dict(action)
        out["target"] = [int(x), int(y)]
        out["bbox"] = [int(v) for v in best]
        out.setdefault("_snapped", True)
        return out

    def _maybe_close_popup_heuristic(self, action: Dict[str, Any], *, screenshot_path: str) -> Dict[str, Any]:
        if not isinstance(action, dict):
            return action
        a = str(action.get("action") or "").lower().strip()
        if a != "click":
            return action

        reason = str(action.get("reason") or "")
        rlow = reason.lower()
        if not ("关闭" in reason or "close" in rlow or "弹窗" in reason or "popup" in rlow):
            return action

        # Prefer closing by clicking an actual close/X button detected in perception.
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

    def _debounce_click(self, action: Dict[str, Any], *, step_id: int) -> Dict[str, Any]:
        if not isinstance(action, dict):
            return action
        a = str(action.get("action") or "").lower().strip()
        if a != "click":
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
        if any(k in rlow for k in ("task", "schedule", "club", "bounty", "mail", "recruit", "recruitment", "gacha")) or any(
            k in reason for k in ("任务", "日程", "社团", "悬赏", "邮箱", "邮件", "招募", "抽卡")
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

    def _decide(self, *, screenshot_path: str, step_id: int = -1, orig_size: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        with Image.open(screenshot_path) as im:
            w, h = im.size

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
            if bool(getattr(self._routine, "is_active", False)):
                step = None
                try:
                    step = self._routine.get_current_step()
                except Exception:
                    step = None
                name = ""
                if step is not None:
                    name = str(getattr(step, "name", "") or "")
                if name in ("Cafe", "Schedule", "Club", "Bounties", "Mail & Tasks"):
                    run_perception = True
        except Exception:
            pass
        if run_perception:
            self._set_stage(step_id=int(step_id), stage="perception")
            t_p0 = time.time()
            try:
                ts = datetime.now().isoformat(timespec="seconds")
                self._log_out(f"[{ts}] step={int(step_id)} stage=perception_begin")
            except Exception:
                pass
            done_evt = threading.Event()
            def _watch_perception() -> None:
                try:
                    time.sleep(45.0)
                    if not done_evt.is_set():
                        ts2 = datetime.now().isoformat(timespec="seconds")
                        self._log_out(f"[{ts2}] step={int(step_id)} stage=perception_still_running")
                except Exception:
                    pass
            try:
                threading.Thread(target=_watch_perception, daemon=True).start()
            except Exception:
                pass
            p_res = engine.ocr(image_path=screenshot_path, prompt=p_prompt, max_new_tokens=int(self.cfg.perception_max_new_tokens))
            try:
                done_evt.set()
            except Exception:
                pass
            try:
                dt = time.time() - t_p0
                ts = datetime.now().isoformat(timespec="seconds")
                self._log_out(f"[{ts}] step={int(step_id)} stage=perception_end elapsed_s={dt:.2f}")
            except Exception:
                pass
            p_raw = str(p_res.get("raw") or "")
            p_parsed = {}
            try:
                p_parsed = _parse_json_content(p_raw)
            except Exception:
                p_parsed = {}
            if isinstance(p_parsed, dict) and isinstance(p_parsed.get("items"), list):
                items = p_parsed.get("items")

        prompt = self._prompt(width=int(w), height=int(h))
        items_txt = self._format_items(items)
        if items_txt:
            prompt = prompt + "\nDetected UI text elements (bbox label):\n" + items_txt + "\n"

        self._set_stage(step_id=int(step_id), stage="policy")
        t_a0 = time.time()
        try:
            ts = datetime.now().isoformat(timespec="seconds")
            self._log_out(f"[{ts}] step={int(step_id)} stage=policy_begin")
        except Exception:
            pass
        done_evt2 = threading.Event()
        def _watch_policy() -> None:
            try:
                time.sleep(45.0)
                if not done_evt2.is_set():
                    ts2 = datetime.now().isoformat(timespec="seconds")
                    self._log_out(f"[{ts2}] step={int(step_id)} stage=policy_still_running")
            except Exception:
                pass
        try:
            threading.Thread(target=_watch_policy, daemon=True).start()
        except Exception:
            pass
        res = engine.ocr(image_path=screenshot_path, prompt=prompt, max_new_tokens=int(self.cfg.max_new_tokens))
        try:
            done_evt2.set()
        except Exception:
            pass
        try:
            dt = time.time() - t_a0
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

        if vlm_err == "hard_timeout":
            act = {
                "action": "wait",
                "duration_ms": 800,
                "reason": "VLM policy timed out; model worker restarted. Waiting and retrying next step.",
                "raw": raw,
                "_vlm_error": vlm_err,
            }
            act.setdefault("_prompt", prompt)
            act.setdefault("_perception", {"prompt": p_prompt, "raw": p_raw, "items": items})
            act.setdefault("_routine", self._routine_meta())
            return act
        try:
            act = _parse_json_content(raw)
        except Exception:
            act = {
                "action": "wait",
                "duration_ms": 1200,
                "reason": "Policy output could not be parsed as JSON; falling back to wait.",
            }
        if vlm_err:
            act.setdefault("_vlm_error", vlm_err)
        act.setdefault("raw", raw)
        act.setdefault("_prompt", prompt)
        act.setdefault("_perception", {"prompt": p_prompt, "raw": p_raw, "items": items})
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

        ow, oh = 0, 0
        try:
            if orig_size and isinstance(orig_size, (tuple, list)) and len(orig_size) == 2:
                ow, oh = int(orig_size[0]), int(orig_size[1])
        except Exception:
            ow, oh = 0, 0

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
        use_orig_coords = False
        try:
            if ow > 0 and oh > 0:
                tgt = act.get("target")
                if isinstance(tgt, (list, tuple)) and len(tgt) == 2:
                    x, y = int(tgt[0]), int(tgt[1])
                    if (x >= int(w) or y >= int(h)) and (0 <= x < ow and 0 <= y < oh):
                        use_orig_coords = True
        except Exception:
            pass

        # Clamp coordinates to inference image bounds BEFORE rescaling to orig_size.
        try:
            tgt = act.get("target")
            if isinstance(tgt, (list, tuple)) and len(tgt) == 2:
                if use_orig_coords:
                    x, y = _clamp_to_orig_xy(int(tgt[0]), int(tgt[1]))
                else:
                    x, y = _clamp_to_image_xy(int(tgt[0]), int(tgt[1]))
                act["target"] = [int(x), int(y)]
        except Exception:
            pass

        try:
            bb = act.get("bbox")
            if isinstance(bb, (list, tuple)) and len(bb) == 4:
                if use_orig_coords:
                    act["bbox"] = _clamp_bbox_orig([int(v) for v in bb])
                else:
                    act["bbox"] = _clamp_bbox([int(v) for v in bb])
        except Exception:
            pass

        try:
            p1 = act.get("from")
            p2 = act.get("to")
            if isinstance(p1, (list, tuple)) and len(p1) == 2:
                if use_orig_coords:
                    x, y = _clamp_to_orig_xy(int(p1[0]), int(p1[1]))
                else:
                    x, y = _clamp_to_image_xy(int(p1[0]), int(p1[1]))
                act["from"] = [int(x), int(y)]
            if isinstance(p2, (list, tuple)) and len(p2) == 2:
                if use_orig_coords:
                    x, y = _clamp_to_orig_xy(int(p2[0]), int(p2[1]))
                else:
                    x, y = _clamp_to_image_xy(int(p2[0]), int(p2[1]))
                act["to"] = [int(x), int(y)]
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
                        if use_orig_coords:
                            it["bbox"] = _clamp_bbox_orig([int(v) for v in bb])
                        else:
                            it["bbox"] = _clamp_bbox([int(v) for v in bb])
        except Exception:
            pass

        if (not use_orig_coords) and orig_size and isinstance(orig_size, (tuple, list)) and len(orig_size) == 2:
            try:
                ow2, oh2 = int(orig_size[0]), int(orig_size[1])
                if ow2 > 0 and oh2 > 0 and int(w) > 0 and int(h) > 0 and (ow2 != int(w) or oh2 != int(h)):
                    sx = float(ow) / float(w)
                    sy = float(oh) / float(h)
                    act = self._rescale_action(act, sx=sx, sy=sy)
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
                try:
                    pad = int(max(10, min(80, round(float(ch) * 0.03))))
                except Exception:
                    pad = 20
                if pad > 0 and int(ch) - 1 - pad >= 0:
                    if int(y) > int(ch) - 1 - pad:
                        y = int(ch) - 1 - pad
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
                self._device.click_client(int(x), int(y))
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
                self._set_stage(step_id=step_id, stage="decide")

                vlm_path = shot_path
                tmp_vlm_path = ""
                orig_size: Optional[Tuple[int, int]] = None
                try:
                    with Image.open(shot_path) as im:
                        ow, oh = im.size
                        orig_size = (int(ow), int(oh))
                        max_side = max(int(ow), int(oh))
                        try:
                            target_max_side = int(os.environ.get("VLM_IMAGE_MAX_SIDE", "960"))
                        except Exception:
                            target_max_side = 960
                        if max_side > target_max_side:
                            scale = float(target_max_side) / float(max_side)
                            nw = max(1, int(round(float(ow) * scale)))
                            nh = max(1, int(round(float(oh) * scale)))
                            im2 = im.resize((nw, nh), resample=Image.BILINEAR)
                            tmp_vlm_path = str((self._run_dir / f"step_{step_id:06d}_vlm.jpg").resolve())
                            im2.save(tmp_vlm_path, quality=85)
                            vlm_path = tmp_vlm_path
                except Exception:
                    tmp_vlm_path = ""
                    vlm_path = shot_path

                try:
                    act = self._decide(screenshot_path=vlm_path, step_id=step_id, orig_size=orig_size)
                finally:
                    try:
                        if tmp_vlm_path:
                            Path(tmp_vlm_path).unlink(missing_ok=True)
                    except Exception:
                        pass

                act_before = act
                act = self._sanitize_action(act)
                act = self._maybe_close_popup_heuristic(act, screenshot_path=shot_path)
                act = self._block_check_lobby_noise(act)
                act = self._snap_click_to_perception_label(act)
                act = self._maybe_cafe_headpat(action=act, screenshot_path=shot_path, step_id=step_id)
                act = self._maybe_autoclick_safe_button(action=act, action_before=act_before, step_id=step_id)
                act = self._sanitize_action(act)
                act = self._debounce_click(act, step_id=step_id)
                act = self._maybe_exploration_click(action=act, action_before=act_before, screenshot_path=shot_path, step_id=step_id)
                act = self._sanitize_action(act)
                act = self._maybe_advance_routine(act)
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
