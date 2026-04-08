"""Base skill framework for Blue Archive automation.

Core concepts:
- ScreenState: OCR + metadata snapshot of current game screen
- Action: dict returned by skills telling the pipeline what to do
- BaseSkill: abstract class all skills inherit from

OCR text matching is the PRIMARY navigation method (portable across resolutions).
All coordinates are normalized 0-1 ratios.
"""
from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── Detection boxes ────────────────────────────────────────────────────

@dataclass
class YoloBox:
    """YOLO detection result (pixel coords + normalized)."""
    cls_id: int
    cls_name: str
    confidence: float
    # Normalized 0-1
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2


@dataclass
class OcrBox:
    text: str
    confidence: float
    x1: float  # normalized 0-1
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1


# ── Screen state ────────────────────────────────────────────────────────

@dataclass
class TemplateHitBox:
    """Template matching result (normalized 0-1 coords)."""
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2


@dataclass
class ScreenState:
    """Snapshot of what's on screen right now."""
    ocr_boxes: List[OcrBox] = field(default_factory=list)
    yolo_boxes: List[YoloBox] = field(default_factory=list)
    template_hits: List[TemplateHitBox] = field(default_factory=list)
    florence_boxes: List[OcrBox] = field(default_factory=list)
    image_w: int = 0
    image_h: int = 0
    screenshot_path: str = ""
    timestamp: float = field(default_factory=time.time)

    def add_florence_boxes(self, boxes: List[OcrBox]) -> None:
        """Append Florence detection boxes."""
        self.florence_boxes.extend(boxes)

    # ── OCR text search helpers ──

    def find_text(self, pattern: str, *, min_conf: float = 0.5,
                  region: Optional[Tuple[float, float, float, float]] = None) -> List[OcrBox]:
        """Find OCR boxes matching a text pattern (substring or regex).

        Args:
            pattern: text to search for (case-insensitive substring match,
                     or regex if it contains special chars)
            min_conf: minimum confidence threshold
            region: optional (x1, y1, x2, y2) normalized region filter
        """
        results = []
        for box in self.ocr_boxes:
            if box.confidence < min_conf:
                continue
            if region:
                rx1, ry1, rx2, ry2 = region
                if box.cx < rx1 or box.cx > rx2 or box.cy < ry1 or box.cy > ry2:
                    continue
            # Try substring match first, then regex
            if pattern.lower() in box.text.lower():
                results.append(box)
            else:
                try:
                    if re.search(pattern, box.text, re.IGNORECASE):
                        results.append(box)
                except re.error:
                    pass
        return results

    def find_text_one(self, pattern: str, **kwargs) -> Optional[OcrBox]:
        """Find first matching OCR box."""
        hits = self.find_text(pattern, **kwargs)
        return hits[0] if hits else None

    def has_text(self, pattern: str, **kwargs) -> bool:
        """Check if text exists on screen."""
        return len(self.find_text(pattern, **kwargs)) > 0

    def find_any_text(self, patterns: List[str], **kwargs) -> Optional[OcrBox]:
        """Find first match from multiple patterns."""
        for p in patterns:
            hit = self.find_text_one(p, **kwargs)
            if hit:
                return hit
        return None

    # ── YOLO detection helpers ──

    def find_yolo(self, cls_name: str, *, min_conf: float = 0.3,
                  region: Optional[Tuple[float, float, float, float]] = None) -> List[YoloBox]:
        """Find YOLO detections by class name (substring match)."""
        results = []
        for box in self.yolo_boxes:
            if box.confidence < min_conf:
                continue
            if cls_name.lower() not in box.cls_name.lower():
                continue
            if region:
                rx1, ry1, rx2, ry2 = region
                if box.cx < rx1 or box.cx > rx2 or box.cy < ry1 or box.cy > ry2:
                    continue
            results.append(box)
        return results

    def find_yolo_one(self, cls_name: str, **kwargs) -> Optional[YoloBox]:
        """Find best (highest conf) YOLO detection by class name."""
        hits = self.find_yolo(cls_name, **kwargs)
        if not hits:
            return None
        return max(hits, key=lambda b: b.confidence)

    def has_yolo(self, cls_name: str, **kwargs) -> bool:
        return len(self.find_yolo(cls_name, **kwargs)) > 0

    # ── Template match helpers ──

    def find_template(self, label: str, *, min_conf: float = 0.5,
                      region: Optional[Tuple[float, float, float, float]] = None) -> List[TemplateHitBox]:
        """Find template hits by label."""
        results = []
        for h in self.template_hits:
            if h.confidence < min_conf:
                continue
            if label.lower() not in h.label.lower():
                continue
            if region:
                rx1, ry1, rx2, ry2 = region
                if h.cx < rx1 or h.cx > rx2 or h.cy < ry1 or h.cy > ry2:
                    continue
            results.append(h)
        return sorted(results, key=lambda h: h.confidence, reverse=True)

    def find_template_one(self, label: str, **kwargs) -> Optional[TemplateHitBox]:
        """Find best template hit by label."""
        hits = self.find_template(label, **kwargs)
        return hits[0] if hits else None

    # ── Region constants for Blue Archive ──

    # Bottom navigation bar (咖啡廳, 課程表, 學生, 编辑, 社交, 製造, 商店, 招募)
    NAV_BAR = (0.0, 0.90, 1.0, 1.0)
    # Top status bar (AP, credits, gems)
    TOP_BAR = (0.0, 0.0, 1.0, 0.10)
    # Left sidebar (公告, MomoTalk, 任務)
    LEFT_SIDE = (0.0, 0.20, 0.20, 0.50)
    # Center dialog area
    CENTER = (0.25, 0.15, 0.75, 0.85)

    def is_lobby(self) -> bool:
        """Detect if we're on the main lobby screen.

        Lobby has bottom nav with multiple buttons: 咖啡廳/咖啡厅, 課程表, 商店, etc.
        """
        nav_hits = 0
        nav_texts = ["咖啡", "課程", "课程", "學生", "学生", "社交", "製造", "制造", "商店", "招募"]
        for t in nav_texts:
            if self.find_text_one(t, region=self.NAV_BAR, min_conf=0.5):
                nav_hits += 1
        return nav_hits >= 3

    def is_loading(self) -> bool:
        """Detect loading screen, download progress, or game update.

        Covers the FULL startup sequence:
        1. "Now Loading..." — post-title-screen loading
        2. "正在更新。[N/M] X.XX%" — game asset download progress
        3. "驗證下載檔案中" — file verification after download
        4. "重置遊戲資料中" — game data reset / cache rebuild
        5. "下載中" / "Downloading" — generic download state
        """
        if self.has_text("Loading", min_conf=0.7) or self.has_text("loading", min_conf=0.7):
            return True
        # Bottom-of-screen progress text during startup / update.
        # OCR garbles these regularly (e.g. "驗證下载檔案中" → mixed Trad/Simp).
        if self.find_any_text(
            [
                # Download in progress
                "下載中", "下载中", "Downloading", "ダウンロード",
                "資料下載", "资料下载",
                # Game update ("正在更新。 [N/M] X%")
                "正在更新", "正在更新。", "正在更新°",
                # File verification
                "檔案驗證", "档案验证", "驗證下載", "验证下载",
                "驗證下载", "驗證檔案", "验证档案",
                # Data reset
                "重置遊戲", "重置游戏", "重置游資", "重置游资",
                "資料中", "资料中",
                # Generic download completion
                "下載檔案", "下载档案",
            ],
            region=(0.0, 0.80, 0.60, 1.0), min_conf=0.40,
        ):
            return True
        return False

    def is_dialog(self) -> bool:
        """Detect a popup/dialog (has 確認/取消 or 確定 buttons)."""
        return (self.has_text("確認", region=self.CENTER, min_conf=0.8) or
                self.has_text("确认", region=self.CENTER, min_conf=0.8) or
                self.has_text("確定", region=self.CENTER, min_conf=0.8))


# ── Action helpers ──────────────────────────────────────────────────────

def action_click(nx: float, ny: float, reason: str = "") -> Dict[str, Any]:
    """Click at normalized coordinates (0-1)."""
    return {"action": "click", "target": [nx, ny], "reason": reason}


def action_click_box(box: OcrBox, reason: str = "") -> Dict[str, Any]:
    """Click center of an OCR box."""
    return action_click(box.cx, box.cy, reason or f"click '{box.text}'")


def action_click_yolo(box: YoloBox, reason: str = "") -> Dict[str, Any]:
    """Click center of a YOLO detection box."""
    return action_click(box.cx, box.cy, reason or f"click yolo '{box.cls_name}'")


def action_wait(ms: int = 500, reason: str = "") -> Dict[str, Any]:
    """Wait for specified duration."""
    return {"action": "wait", "duration_ms": ms, "reason": reason}


def action_back(reason: str = "") -> Dict[str, Any]:
    """Press Escape / Back."""
    return {"action": "back", "reason": reason}


def action_swipe(fx: float, fy: float, tx: float, ty: float,
                 duration_ms: int = 400, reason: str = "") -> Dict[str, Any]:
    """Swipe between two normalized points."""
    return {"action": "swipe", "from": [fx, fy], "to": [tx, ty],
            "duration_ms": duration_ms, "reason": reason}


def action_scroll(nx: float, ny: float, clicks: int = -3, reason: str = "") -> Dict[str, Any]:
    """Mouse wheel scroll at normalized position."""
    return {"action": "scroll", "target": [nx, ny], "clicks": clicks, "reason": reason}


def action_done(reason: str = "") -> Dict[str, Any]:
    """Signal that this skill is finished."""
    return {"action": "done", "reason": reason}


# ── Base Skill ──────────────────────────────────────────────────────────

class BaseSkill(ABC):
    """Abstract base for all agent skills.

    Each skill handles one game section (e.g. cafe, schedule, bounties).
    Skills are stateful - they track their sub-state across ticks.

    Lifecycle:
        1. Pipeline calls reset() before starting the skill
        2. Pipeline calls tick(screen) each frame
        3. Skill returns an action dict
        4. If action is "done", pipeline moves to next skill
    """

    def __init__(self, name: str):
        self.name = name
        self.sub_state: str = ""
        self.ticks: int = 0
        self.max_ticks: int = 60  # timeout per skill
        self._log_lines: List[str] = []

    def reset(self) -> None:
        """Reset skill state for a fresh run."""
        self.sub_state = ""
        self.ticks = 0
        self._log_lines = []

    def log(self, msg: str) -> None:
        line = f"[{self.name}] {msg}"
        self._log_lines.append(line)
        print(line)

    def _find_florence_hits(self, screen: ScreenState, queries: List[str], *, region: Optional[Tuple[float, float, float, float]] = None) -> List[OcrBox]:
        """Deprecated — Florence removed. Always returns empty list."""
        return []

    def _find_florence_hit(self, screen: ScreenState, queries: List[str], *, region: Optional[Tuple[float, float, float, float]] = None) -> Optional[OcrBox]:
        """Deprecated — Florence removed. Always returns None."""
        return None

    @abstractmethod
    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        """Process one frame and return an action.

        Returns action dict. Return action_done() when skill is complete.
        """
        ...

    def detect_current_screen(self, screen: ScreenState) -> Optional[str]:
        """Detect current screen based on OCR header and nav bar cues."""
        if screen.is_lobby():
            return "Lobby"

        header_region = (0.0, 0.0, 0.3, 0.15)

        # Check "任務" header first — but disambiguate Daily Tasks vs Campaign.
        # Daily Tasks has tabs: 全體/每天/每週/成就/挑戰任務 in the tab bar area.
        # Campaign (mission hub) does NOT have these tabs.
        mission_header = screen.find_any_text(
            ["任務", "任务"], region=header_region, min_conf=0.5
        )
        if mission_header:
            daily_tabs = screen.find_any_text(
                ["全體", "全体", "每天", "每週", "每周", "挑戰任務", "挑战任务"],
                region=(0.35, 0.10, 1.0, 0.20), min_conf=0.6
            )
            if daily_tabs:
                return "DailyTasks"
            return "Mission"

        headers = {
            "Cafe": ["咖啡廳", "咖啡厅"],
            "Schedule": ["課程表", "课程表", "全体課程", "全体课程"],
            "Shop": ["商店"],
            "Club": ["社團", "社团", "Club"],
            "Bounty": ["懸賞通緝", "悬赏通缉", "懸賞", "悬赏", "通緝", "通缉", "悬通", "Bounty"],
            "PVP": ["戰術對抗", "战术对抗", "戰術大賽", "战术大赛", "術大赛", "術大賽", "大賽", "大赛"],
            "TotalAssault": ["總力戰", "总力战", "大決戰", "大决战"],
            "Pass": ["通行證", "通行证", "Pass", "PASS"],
            "Mail": ["郵件", "邮件", "郵箱", "邮箱", "信箱", "Mail"],
            "Event": ["活動", "活动"],
            "Craft": ["製造", "制造", "Craft"],
            "Student": ["學生", "学生", "Student"],
            "Formation": ["部隊", "编队", "部隊編成"],
        }
        for screen_name, texts in headers.items():
            if screen.find_any_text(texts, region=header_region, min_conf=0.6):
                return screen_name

        # Fallback: campaign hub detection via grid markers (OCR often misses 任務 header)
        # Keep this AFTER specific header checks to avoid classifying total assault/pass as Mission.
        hub_markers = screen.find_any_text(
            ["懸賞通緝", "悬赏通缉", "學園交流會", "学园交流会",
             "戰術大賽", "战术大赛", "特殊任務", "特殊任务",
             "制約解除", "劇情", "剧情"],
            min_conf=0.5
        )
        if hub_markers:
            return "Mission"
        area_marker = screen.find_text_one(r"Area\s*\d+", min_conf=0.5)
        if area_marker:
            return "Mission"
        stage_tabs = screen.find_any_text(
            ["Normal", "Hard"],
            region=(0.50, 0.16, 0.98, 0.28),
            min_conf=0.6,
        )
        stage_id = screen.find_text_one(r"\d+\-\d+", min_conf=0.5)
        entry_btn = screen.find_any_text(
            ["入場", "入场"],
            region=(0.78, 0.22, 0.98, 0.78),
            min_conf=0.5,
        )
        if stage_tabs and (stage_id or entry_btn or area_marker):
            return "Mission"
        return None

    def _handle_common_popups(self, screen: ScreenState) -> Optional[Dict[str, Any]]:
        """Handle common popups that can appear in any skill.

        Returns an action if a popup was handled, None otherwise.
        """
        # Notification modal (通知): OCR confidence can be low (~0.55-0.70),
        # and this dialog may have both 取消/確認 buttons. Treat it as safe
        # to dismiss so skills don't stall in enter states.
        notification = screen.find_text_one(
            "通知", region=(0.30, 0.12, 0.70, 0.32), min_conf=0.55
        )
        if notification:
            # Check if this notification MUST be confirmed (not canceled):
            # 1. Cafe invite: "邀.*咖啡"
            # 2. Game update download: title "更新通知" or body keywords
            #    Actual body example: "需要下載遊戲所需的檔案6.27GB"
            invite_hint = screen.find_text_one(
                r"邀.*咖啡", region=screen.CENTER, min_conf=0.5
            )
            # Detect "更新" in the title text itself (OCR: "更新通知")
            update_title = "更新" in (notification.text or "")
            update_hint = screen.find_any_text(
                ["下載必要", "下载必要", "更新資源", "更新资源",
                 "下載資源", "下载资源", "下載內容", "下载内容",
                 "需要下載", "需要下载", "遊戲所需", "游戏所需",
                 "下載遊戲", "下载游戏", "檔案"],
                region=screen.CENTER, min_conf=0.45,
            )
            must_confirm = invite_hint or update_title or update_hint

            confirm_btn = screen.find_any_text(
                ["確認", "确认", "確定", "确定", "確", "确", "OK"],
                region=(0.30, 0.55, 0.74, 0.82),
                min_conf=0.40,
            )
            cancel_btn = screen.find_any_text(
                ["取消"],
                region=(0.28, 0.60, 0.56, 0.82),
                min_conf=0.55,
            )

            # Must-confirm notifications: always click 確認
            if must_confirm and confirm_btn:
                tag = "invite" if invite_hint else "update"
                self.log(f"notification popup ({tag}): clicking confirm")
                return action_click_box(confirm_btn, f"confirm {tag} notification")
            if must_confirm and cancel_btn:
                # OCR missed confirm but found cancel → use hardcoded confirm position
                self.log("notification popup (must-confirm): confirm fallback click")
                return action_click(0.598, 0.701, "confirm notification fallback")

            # Regular notification: prefer cancel to dismiss
            if cancel_btn:
                self.log("notification popup: clicking cancel to close")
                return action_click_box(cancel_btn, "dismiss notification popup (cancel)")

            if confirm_btn:
                self.log("notification popup: clicking confirm")
                return action_click_box(confirm_btn, "dismiss notification popup (confirm)")

            if must_confirm:
                self.log("notification popup (must-confirm): no buttons, fallback click confirm area")
                return action_click(0.598, 0.701, "confirm notification (no btn fallback)")

            # No buttons detected at all — OCR missed them.
            # Click 確認 button area (center-bottom of dialog, large target).
            self.log("notification popup: no buttons detected, clicking confirm area")
            return action_click(0.50, 0.64, "dismiss notification popup (confirm area fallback)")

        # Confirm dialogs: full two-char buttons (確認/確定)
        confirm = screen.find_any_text(
            ["確認", "确认", "確定", "确定", "確", "确"],
            region=screen.CENTER, min_conf=0.8
        )
        if confirm:
            cancel = screen.find_any_text(
                ["取消"],
                region=screen.CENTER, min_conf=0.8
            )
            if cancel:
                return None
            self.log(f"confirm popup: '{confirm.text}'")
            return action_click_box(confirm, f"confirm popup '{confirm.text}'")

        # Single-char confirm button (確/确) — game sometimes renders
        # only one character for the confirm button (e.g. inventory-full popup).
        # Only click if there's supporting popup body text nearby.
        single_confirm = screen.find_any_text(
            ["確", "确"],
            region=(0.35, 0.65, 0.65, 0.80), min_conf=0.9
        )
        if single_confirm:
            popup_body = screen.find_any_text(
                ["背包已满", "背包已滿", "整理背包", "道具背包",
                 "已達上限", "已达上限", "空間不足", "空间不足",
                 "超過上限", "超过上限"],
                region=screen.CENTER, min_conf=0.6
            )
            if popup_body:
                self.log(f"single-char confirm popup: '{single_confirm.text}' (body: '{popup_body.text}')")
                return action_click_box(single_confirm, f"confirm popup '{single_confirm.text}'")

        # "是否跳過" (skip) dialog - always confirm
        skip = screen.find_text_one("跳過", region=screen.CENTER, min_conf=0.8)
        if skip:
            confirm2 = screen.find_any_text(["確", "确"], region=screen.CENTER, min_conf=0.8)
            if confirm2:
                self.log("skip dialog: confirming")
                return action_click_box(confirm2, "confirm skip dialog")

        return None

    def _try_go_lobby(self, screen: ScreenState) -> Optional[Dict[str, Any]]:
        """Try to navigate back to lobby."""
        if screen.is_lobby():
            return None
        return action_back(f"{self.name}: press back toward lobby")

    def _nav_to(self, screen: ScreenState, nav_texts: List[str]) -> Optional[Dict[str, Any]]:
        """Click a bottom nav bar button by text."""
        for t in nav_texts:
            hit = screen.find_text_one(t, region=screen.NAV_BAR, min_conf=0.5)
            if hit:
                self.log(f"nav click '{hit.text}'")
                return action_click_box(hit, f"nav to '{hit.text}'")
        return None
