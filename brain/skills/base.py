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
    # Which detector produced this box: "ui" / "avatar" / "battle" / "cafe".
    # Lets skills filter by source model (e.g. arena opponent heads come from
    # the avatar model, not the ui model) without guessing from cls_name.
    model_tag: str = ""

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
class ScreenState:
    """Snapshot of what's on screen right now."""
    ocr_boxes: List[OcrBox] = field(default_factory=list)
    yolo_boxes: List[YoloBox] = field(default_factory=list)
    image_w: int = 0
    image_h: int = 0
    screenshot_path: str = ""
    timestamp: float = field(default_factory=time.time)
    # Raw BGR frame (numpy) kept for on-demand digit-OCR cropping. Not always
    # populated (e.g. read_screen from a path skips it); digit-OCR helpers must
    # null-check. Excluded from any serialization (it's a big array).
    frame: Any = field(default=None, repr=False, compare=False)
    # 高频 DXcam 线程的最新检出(2026-07-11 工业级链路): 帧龄≤0.5s@2FPS,
    # 主 tick 帧龄 ~2.2s 对轮播类时敏目标必错位 — skill 做"有目标就点"
    # 判定时优先读这里。None = 线程未跑/未接线, 调用方必须 null-check。
    fresh_boxes: Any = field(default=None, repr=False, compare=False)
    fresh_frame: Any = field(default=None, repr=False, compare=False)
    fresh_ts: float = 0.0

    # ── OCR text search helpers ──

    def find_text(self, pattern: str, *, min_conf: float = 0.5,
                  region: Optional[Tuple[float, float, float, float]] = None) -> List[OcrBox]:
        """Find OCR boxes matching a text pattern (substring or regex).

        Matching runs on *normalized* text: OCR output and the pattern are
        both passed through `vision.ocr_normalize.normalize` (known misreads
        fixed, Traditional chars folded to Simplified). Skills no longer
        need to enumerate Trad/Simp/mixed permutations of a keyword.

        Args:
            pattern: text to search for (case-insensitive substring match,
                     or regex if it contains special chars)
            min_conf: minimum confidence threshold
            region: optional (x1, y1, x2, y2) normalized region filter
        """
        # Lazy import so vision module tests don't require brain.
        try:
            from vision.ocr_normalize import normalize as _norm
        except Exception:
            _norm = lambda s: s  # fall through to raw match
        norm_pattern = _norm(pattern).lower()
        # Only fall through to regex if the pattern contains *clearly intentional*
        # regex syntax. A bare `?` / `+` after a literal char (e.g. "次?") is
        # syntactically valid regex but almost always a keyword author's literal-
        # punctuation typo, and `次?` matches the empty string at every position
        # → false-positive on every OCR box. Keep regex opt-in via clear markers.
        _regex_markers = (".*", ".+", "[", "\\", "|", "^", "$", "(?")
        treat_as_regex = any(m in norm_pattern for m in _regex_markers)
        results = []
        for box in self.ocr_boxes:
            if box.confidence < min_conf:
                continue
            if region:
                rx1, ry1, rx2, ry2 = region
                if box.cx < rx1 or box.cx > rx2 or box.cy < ry1 or box.cy > ry2:
                    continue
            norm_text = _norm(box.text).lower()
            # Try substring match first, then regex (only if pattern looks regex-y).
            if norm_pattern in norm_text:
                results.append(box)
            elif treat_as_regex:
                try:
                    if re.search(norm_pattern, norm_text, re.IGNORECASE):
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

    # ── Region constants for Blue Archive ──

    # Center dialog area
    CENTER = (0.25, 0.15, 0.75, 0.85)

    def is_lobby(self) -> bool:
        """Detect if we're on the main lobby screen — pure YOLO (no OCR).

        Lobby shows the 8 bottom-nav entry icons (咖啡厅入口/课程表入口/...).
        Seeing >=2 of those cls = lobby. v3 detects all 8 at 0.91-0.99 on
        real frames, so 2 is a safe floor that still rejects sub-pages
        (which only carry a lingering nav bar partially, if at all).
        """
        from brain.skills.ui_classes import LOBBY_NAV_ICONS
        want = set(LOBBY_NAV_ICONS)
        hits = sum(
            1 for b in (self.yolo_boxes or [])
            if b.cls_name in want and b.confidence >= 0.30
        )
        return hits >= 2

    def scan_lobby_nav_badges(self) -> Dict[str, str]:
        """Scan the 8 bottom-nav icons for red/yellow badges.

        Returns a dict {nav_name: state} where state ∈ {"red", "yellow",
        "none"} and nav_name is the stable english key (cafe / schedule /
        student / edit / social / craft / shop / recruit).  Names not
        currently visible on screen are omitted from the dict — callers
        should treat absence as "unknown" (we're probably not on lobby).

        Where the dot lives: BA renders the badge as a small saturated
        dot just BELOW the icon body and to the RIGHT of the label.  In
        the standard 4K-window the dot centroid is at roughly
        (label_cx + 0.012, ~0.93).  We scan a tight ROI just above the
        label top and right of label centre (`label_cx + [0.005, 0.045]`,
        y `[0.905, 0.945]`).  This ROI deliberately excludes the icon
        body (above) — that part is full of decorative colors like the
        cafe heart, recruit gold stars, etc. — to avoid false positives.

        Caller should ensure they're on lobby (is_lobby() == True) before
        calling — running on non-lobby screens may catch false positives
        from whatever happens to live in the bottom strip.

        PURE-YOLO (2026-05-29): maps DOT_RED / DOT_YELLOW detections to the
        nearest nav-entry cls by horizontal proximity (a badge sits just
        above-right of its icon). Only entries WITH a nearby dot are
        returned ('red'/'yellow'); entries with no dot are OMITTED rather
        than marked 'none', so the badge-skip optimiser treats them as
        'unknown' → runs the skill. That's the safe choice during bring-up
        (we never wrongly skip real work). The 'none'-marking optimisation
        comes back once the YOLO flow is verified end-to-end.
        """
        boxes = [b for b in (self.yolo_boxes or []) if b.confidence >= 0.30]
        if not boxes:
            return {}
        from brain.skills import ui_classes as UC
        # nav-entry cls -> stable english key (matches _SKILL_BADGE_MAP)
        entry_key = {
            UC.NAV_CAFE: "cafe", UC.NAV_SCHEDULE: "schedule",
            UC.NAV_STUDENT: "student", UC.NAV_EDIT: "edit",
            UC.NAV_SOCIAL: "social", UC.NAV_CRAFT: "craft",
            UC.NAV_SHOP: "shop", UC.NAV_RECRUIT: "recruit",
            UC.NAV_MAIL: "mail", UC.NAV_TASKS: "campaign_nav",
        }
        entries = [(entry_key[b.cls_name], b)
                   for b in boxes if b.cls_name in entry_key]
        reds = [b for b in boxes if b.cls_name == UC.DOT_RED]
        yellows = [b for b in boxes if b.cls_name == UC.DOT_YELLOW]
        results: Dict[str, str] = {}
        for key, eb in entries:
            # campaign tile (任务大厅入口) is large — its dots sit well above
            # the tile centre; bottom-nav dots sit just above their icon.
            dx = 0.06 if key == "campaign_nav" else 0.05
            dy_above = 0.30 if key == "campaign_nav" else 0.03
            def near(d):
                return abs(d.cx - eb.cx) <= dx and (eb.cy - dy_above) <= d.cy <= eb.cy + 0.02
            if any(near(d) for d in reds):
                results[key] = "red"
            elif any(near(d) for d in yellows):
                results[key] = "yellow"
        return results

    def is_loading(self) -> bool:
        """Detect in-game loading spinner — pure YOLO (加载中 cls, no OCR).

        KNOWN GAP (pure-YOLO bring-up): the startup download/verify/reset
        screens ("Now Loading", "正在更新", "驗證下載檔案中") have NO YOLO
        cls yet, so with OCR disabled they are not detected here. In-game
        transient scene loads carry the 加载中 spinner (cls 22, 45f trained),
        which this catches. If a startup/download screen ever stalls the
        pipeline, the fix is to train a cls for it (NOT re-enable OCR here).
        """
        for b in (self.yolo_boxes or []):
            if b.cls_name == "加载中" and b.confidence >= 0.35:
                return True
        return False


# ── Dot color posterior (2026-07-08) ────────────────────────────────────
# v12 红点/黄点位置先验强: 社交入口的蓝「+」badge 被标红点 conf0.72, 已爬进
# 真点 conf 区间(0.85-0.92, 闪烁真点低至 0.69) → conf 阈值不可分。但点类是
# 纯色小圆, 颜色占比完美分离(同帧实测: 真红点 red%=0.67-0.75 / 真黄点
# yellow%=0.72-0.74 / 假点(蓝+) 两者=0.00)。所有 dot 判定过此闸。

def classify_dot_color(frame, x1: float, y1: float, x2: float, y2: float
                       ) -> Optional[str]:
    """HSV posterior for a dot bbox (normalized coords). Returns '红点' /
    '黄点' / None (neither red nor yellow = position-prior false fire).
    frame None → caller should treat as "cannot verify" (pass-through)."""
    if frame is None:
        return None
    try:
        import cv2
        import numpy as np
        h, w = frame.shape[:2]
        px1, py1 = max(0, int(x1 * w)), max(0, int(y1 * h))
        px2, py2 = min(w, int(x2 * w)), min(h, int(y2 * h))
        crop = frame[py1:py2, px1:px2]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        vivid = (sat > 120) & (val > 120)
        red = float(np.mean(((hue < 10) | (hue > 170)) & vivid))
        yel = float(np.mean((hue >= 15) & (hue <= 35) & vivid))
        if max(red, yel) < 0.25:
            return None
        return "红点" if red >= yel else "黄点"
    except Exception:
        return None


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


def action_swipe_tap(fx: float, fy: float, tx: float, ty: float,
                     cx: float, cy: float, duration_ms: int = 150,
                     reason: str = "") -> Dict[str, Any]:
    """原子 swipe→tap(一条 adb shell 连发, 间隔<0.5s)。

    对自动轮播 UI: swipe 把轮播拉停数秒并翻到确定项, tap 在静止期内落点
    无时序竞争(hub banner 帧龄 2.2s vs 项周期 2.6s 的错位问题唯一硬解)。
    """
    return {"action": "swipe_tap", "from": [fx, fy], "to": [tx, ty],
            "target": [cx, cy], "duration_ms": duration_ms, "reason": reason}


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

    # ── Dot-driven skip check (overridden by daily-harvest skills) ──
    # When pipeline is about to start this skill, it first calls should_run()
    # on the current ScreenState. Default = True (always run). Daily-harvest
    # skills (cafe / mail / schedule / club / daily_tasks / event_activity)
    # override this to look for their associated red/yellow dot on the lobby
    # screen and return False if there's no work to do.
    #
    # Battle / sweep / arena / bounty skills DO NOT override — they always
    # run when the user enables them in skill_order.
    def should_run(self, screen: ScreenState) -> bool:
        """Return False to make pipeline skip this skill entirely.
        Called once at skill entry before tick(). Default = always run."""
        return True

    def hall_tile_dot(self, screen: ScreenState, tile_cls: str,
                      *, dot_classes: Tuple[str, ...] = ("红点", "黄点")
                      ) -> Optional[bool]:
        """Task-hall per-activity work check (user iron rule 2026-06-11: the
        LOBBY entry dot must never gate these skills — enter the hall and scan
        each activity's own dot).

        Returns None when the tile isn't visible (not in the hall — can't
        decide), True when a red/yellow dot sits at the tile's TOP-RIGHT
        (live-measured 2026-06-11: 悬赏 tile (0.561,0.550) → dot (0.634,0.512)),
        False when the tile is visible with no dot (= no work today)."""
        tile = self.find_cls(screen, tile_cls, conf=0.40)
        if tile is None:
            return None
        region = (tile.x1, tile.y1 - 0.08, tile.x2 + 0.11, tile.y2)
        return self.dot_in_region(screen, region, dot_classes=dot_classes)

    def dot_in_region(self, screen: ScreenState,
                       region: Tuple[float, float, float, float],
                       *, dot_classes: Tuple[str, ...] = ("红点", "黄点"),
                       min_conf: float = 0.35) -> bool:
        """Helper for should_run: is there a red/yellow dot inside this
        normalized rect (x1, y1, x2, y2)?  Uses ui_yolo26m_v1 detections
        already on screen.yolo_boxes (no extra inference)."""
        if not screen.yolo_boxes:
            return False
        x1, y1, x2, y2 = region
        for b in screen.yolo_boxes:
            if b.confidence < min_conf:
                continue
            if b.cls_name not in dot_classes:
                continue
            cx, cy = (b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                # 颜色后验 (2026-07-08): 位置先验假点(社交蓝+ conf0.72)过滤。
                # frame 缺失时放行(验色只做增量过滤, 不引入新失败模式)。
                if screen.frame is not None:
                    seen = classify_dot_color(
                        screen.frame, b.x1, b.y1, b.x2, b.y2)
                    if seen is None or seen not in dot_classes:
                        continue
                return True
        return False

    def dot_on_entry(self, screen: ScreenState,
                      entry_class_names,
                      *, dot_classes: Tuple[str, ...] = ("红点", "黄点"),
                      min_conf_entry: float = 0.4,
                      min_conf_dot: float = 0.35) -> bool:
        """Stronger should_run helper: are any of the listed entry icons
        currently on screen AND covered by a red/yellow dot?

        Returns True when:
          (a) Entry icon NOT visible (we're probably not on lobby, can't
              decide here — defer to skill's own logic), OR
          (b) Entry icon visible AND a red/yellow dot center sits inside it.
        Returns False ONLY when entry IS visible but NO dot covers it
        (clean "no work to do" signal → skill can be skipped).
        """
        if not screen.yolo_boxes:
            return True  # detector disabled, can't decide → pass
        targets = list(entry_class_names) if isinstance(entry_class_names, (list, tuple)) else [entry_class_names]
        target_set = set(targets)
        target_lower = [t.lower() for t in targets]
        entries = []
        dots = []
        for b in screen.yolo_boxes:
            cn = b.cls_name
            cn_low = (cn or "").lower()
            if b.confidence >= min_conf_entry and (
                cn in target_set or any(t in cn_low or cn_low in t for t in target_lower)
            ):
                entries.append(b)
            elif b.confidence >= min_conf_dot and cn in dot_classes:
                dots.append(b)
        if not entries:
            return True  # not on lobby (entry not visible) → defer
        # Margin: red/yellow badges sit at the entry's TOP-RIGHT corner, often
        # a hair OUTSIDE the icon bbox. A strict inside-bbox test false-skips
        # (live 2026-06-02: cafe/schedule yellow dots present but skipped). Allow
        # a small expansion so a badge near the entry counts.
        mx, my = 0.03, 0.06
        for d in dots:
            # 颜色后验 (2026-07-08): 蓝+/灰位假点过滤 (club 假重进同根)。
            if screen.frame is not None:
                seen = classify_dot_color(screen.frame, d.x1, d.y1, d.x2, d.y2)
                if seen is None or seen not in dot_classes:
                    continue
            dcx = (d.x1 + d.x2) / 2
            dcy = (d.y1 + d.y2) / 2
            for e in entries:
                if (e.x1 - mx) <= dcx <= (e.x2 + mx) and (e.y1 - my) <= dcy <= (e.y2 + my):
                    return True
        return False  # entry visible, no dot → no work

    def log(self, msg: str) -> None:
        line = f"[{self.name}] {msg}"
        self._log_lines.append(line)
        print(line)

    # ── Shared helpers: reusable mini-flows used by multiple skills ──

    # ════════════════════════════════════════════════════════════════
    # YOLO-only UI resolution (canonical — use these, NOT OCR text match).
    # cls names come from brain/skills/ui_classes.py. Exact-match only
    # (substring matching would mis-match e.g.
    # "领取_黄" vs "全部领取_黄"). No OCR fallback by design: if YOLO
    # can't see a cls, surface the gap (log+wait) instead of hiding it.
    # ════════════════════════════════════════════════════════════════
    def find_cls(
        self,
        screen: ScreenState,
        cls_names,
        *,
        conf: float = 0.30,
        region: Optional[Tuple[float, float, float, float]] = None,
    ) -> Optional[YoloBox]:
        """Return highest-conf YOLO box whose cls_name EXACTLY equals one of
        cls_names (str or list), optionally constrained to a region. None if
        no match. This is the primary click-target resolver for all skills."""
        if isinstance(cls_names, str):
            cls_names = [cls_names]
        want = set(cls_names)
        best = None
        for b in (screen.yolo_boxes or []):
            if b.confidence < conf:
                continue
            if b.cls_name not in want:
                continue
            if region is not None:
                if not (region[0] <= b.cx <= region[2] and region[1] <= b.cy <= region[3]):
                    continue
            if best is None or b.confidence > best.confidence:
                best = b
        return best

    def read_count(self, screen: ScreenState, icon_cls, *, conf: float = 0.30,
                   side: str = "right", span: float = 0.10, pad: float = 0.005):
        """DIGIT-ONLY read of the number next to a currency/count icon.

        YOLO locates the icon (e.g. TICKET_BOUNTY, TOPBAR_AP); we crop the
        digit strip beside it and OCR only the digits. This is the ONLY place
        OCR is used (per spec: YOLO for everything, OCR for digits) and works
        regardless of the global _OCR_ENABLED nav-OCR switch.

        Args:
            icon_cls: cls name(s) of the icon box to anchor on.
            side: which side the number sits — "right" (top-bar currencies,
                  tickets) or "left".
            span: width of the digit strip as a fraction of screen width.
            pad: gap between icon edge and strip start (fraction of width).
        Returns:
            (current, total) from pipeline.parse_count — total may be None;
            or None if the icon isn't found / nothing read. Caller decides.
        """
        box = self.find_cls(screen, icon_cls, conf=conf)
        if box is None or screen.frame is None:
            return None
        # vertical band aligned to the icon (a touch taller for glyph margin)
        bh = box.y2 - box.y1
        y1 = max(0.0, box.y1 - bh * 0.25)
        y2 = min(1.0, box.y2 + bh * 0.25)
        if side == "right":
            x1 = min(1.0, box.x2 + pad)
            x2 = min(1.0, x1 + span)
        else:  # left
            x2 = max(0.0, box.x1 - pad)
            x1 = max(0.0, x2 - span)
        try:
            from brain.pipeline import run_digit_ocr, parse_count
        except Exception:
            return None
        raw = run_digit_ocr(screen.frame, (x1, y1, x2, y2))
        result = parse_count(raw)
        if result is not None:
            self.log(f"read_count({icon_cls})={result} (raw {raw!r})")
        return result

    def find_all_cls(
        self,
        screen: ScreenState,
        cls_names,
        *,
        conf: float = 0.30,
        region: Optional[Tuple[float, float, float, float]] = None,
    ) -> List[YoloBox]:
        """Like find_cls but returns ALL exact matches sorted by conf desc."""
        if isinstance(cls_names, str):
            cls_names = [cls_names]
        want = set(cls_names)
        hits = []
        for b in (screen.yolo_boxes or []):
            if b.confidence < conf:
                continue
            if b.cls_name not in want:
                continue
            if region is not None:
                if not (region[0] <= b.cx <= region[2] and region[1] <= b.cy <= region[3]):
                    continue
            hits.append(b)
        return sorted(hits, key=lambda b: -b.confidence)

    def click_cls(
        self,
        screen: ScreenState,
        cls_names,
        reason: str,
        *,
        conf: float = 0.30,
        region: Optional[Tuple[float, float, float, float]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Find a cls by exact name + click it. Returns action dict or None
        (caller decides what to do on miss — usually log+wait)."""
        box = self.find_cls(screen, cls_names, conf=conf, region=region)
        if box is None:
            return None
        return action_click_box(box, f"{reason} (YOLO {box.cls_name} {box.confidence:.2f})")

    def nav_home(self, screen: ScreenState, reason: str = "回大厅") -> Dict[str, Any]:
        """Navigate toward the lobby using ONLY in-game buttons — NEVER a blind
        ESC / back keyevent (user 2026-06-15 iron rule: 反复 ESC-spam recovery
        多次触发 Unity ANR「Blue Archive没有响应」, freezing the game; "只点基于游戏
        内的返回大厅还是叉叉"). Preference: 回大厅按钮(home → lobby directly) →
        弹窗叉叉(close popup) → 返回键(back one screen). If NONE is detected this
        frame, WAIT — do not blind-tap a guessed position and do not ESC; the
        caller's own _phase_ticks timeout ends the skill cleanly if truly stuck.
        """
        from brain.skills import ui_classes as UC
        home = self.find_cls(screen, UC.BTN_HOME, conf=0.30)
        if home is not None:
            return action_click_box(home, f"{reason}: 回大厅按钮")
        x = self.find_cls(screen, UC.BTN_CLOSE_X, conf=0.30)
        if x is not None:
            return action_click_box(x, f"{reason}: 弹窗叉叉")
        back = self.find_cls(screen, UC.BTN_BACK, conf=0.30)
        if back is not None:
            return action_click_box(back, f"{reason}: 返回键")
        return action_wait(450, f"{reason}: 无 home/X/返回键 → 等待 (绝不瞎按 ESC)")

    def detect_screen_yolo(self, screen: ScreenState) -> Optional[str]:
        """Detect current page from YOLO cls signatures (no OCR).

        Returns page name (Lobby/Mail/Schedule/Cafe/Craft/MomoTalk/Story/
        Battle) or None if no signature matches. See ui_classes.PAGE_SIGNATURES.
        Lobby is checked last so a sub-page's own cls wins over a lingering
        nav-bar (nav icons are visible inside many pages)."""
        from brain.skills import ui_classes as UC
        # Non-lobby pages first (more specific)
        for page, (cls_list, min_n) in UC.PAGE_SIGNATURES.items():
            if page == "Lobby":
                continue
            n = sum(1 for c in cls_list if self.find_cls(screen, c, conf=0.30) is not None)
            if n >= min_n:
                return page
        # Lobby last
        lobby_cls, lobby_min = UC.PAGE_SIGNATURES["Lobby"]
        n = sum(1 for c in lobby_cls if self.find_cls(screen, c, conf=0.30) is not None)
        if n >= lobby_min:
            return "Lobby"
        return None

    @abstractmethod
    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        """Process one frame and return an action.

        Returns action dict. Return action_done() when skill is complete.
        """
        ...

    def _handle_common_popups(self, screen: ScreenState) -> Optional[Dict[str, Any]]:
        """Handle common popups that can appear in any skill.

        Returns an action if a popup was handled, None otherwise.
        """
        # Bond level-up screen (羈絆升級！) — full-screen transition that
        # any affinity-earning skill can trigger (cafe headpat, schedule
        # lesson, club AP claim, event battles, etc.).  Tap anywhere
        # advances.  Run_20260516_234050 t232: bot got stuck here for
        # 29 ticks waiting because no skill-specific handler caught it.
        # 2026-07-16 纯 cls 化: 词表有专类 398羁绊升级/399地区升级,
        # OCR 文字匹配(羈絆升級/治癒力 fallback)全删。
        bond_screen = self.find_cls(
            screen, ["羁绊升级", "地区升级"], conf=0.35)
        if bond_screen is not None:
            self.log(f"bond level-up screen detected "
                     f"'{bond_screen.cls_name}', tap to dismiss")
            return action_click(0.5, 0.5, "dismiss bond level-up")

        # ── 通知弹窗 (pure YOLO, 2026-07-16 重构) ────────────────────────
        # 旧版在这里用 OCR 文字给弹窗分类(通知/提示标题 + 邀.*咖啡=确认 /
        # 更新通知+下載=确认 / 掃蕩=确认 / 訪問好友=取消 / 是否結束=取消 …),
        # _OCR_ENABLED=False 后全是死代码, 且"读文字定动作"违反感知铁律
        # (判断一律 cls)。新策略(用户 spec 2026-07-16):
        #   • 语境内的"该确认"由拥有语境的 skill 自己处理, 并且必须排在调用
        #     本 helper 之前:
        #       - 咖啡厅邀请确认 → cafe._invite stage2 / _recover_invite_overlay
        #       - 课程表报告确认 → schedule.tick PRIORITY 1 (调本 helper 前)
        #       - 扫荡确认      → 各 sweep skill 的结构闸
        #       - 强更下载确认   → pipeline._global_interceptor 启动期结构闸
        #   • 因此能落到这个通用 helper 的「确认+取消/叉」结构弹窗, 定义上就是
        #     当前 skill 没预期的通知弹窗 → 默认安全路径: 一律点取消/叉掉。
        #     绝不盲点确认(2026-06-02 买票事故根因 = 盲确认, 见 money_safety)。
        #   • 只有确认键、无取消/叉的弹窗: 这里不动 (fail-closed — 交给 skill
        #     自己的 handler / tick 预算; 启动期强更框由 interceptor 接)。
        from brain.skills import ui_classes as UC
        _POPUP_BTN_BAND = (0.20, 0.45, 0.80, 0.95)  # 居中对话框按钮带
        popup_confirm = self.find_cls(
            screen, [UC.BTN_CONFIRM, UC.BTN_CONFIRM_GREY],
            conf=0.30, region=_POPUP_BTN_BAND,
        )
        if popup_confirm is not None:
            popup_cancel = self.find_cls(
                screen, UC.BTN_CANCEL, conf=0.30, region=_POPUP_BTN_BAND
            )
            if popup_cancel is not None:
                self.log("通知弹窗(确认+取消结构) → 默认安全路径: 取消")
                return action_click_box(
                    popup_cancel, "dismiss notification popup (取消键)")
            popup_x = self.find_cls(screen, UC.BTN_CLOSE_X, conf=0.30)
            if popup_x is not None:
                self.log("通知弹窗(确认+叉结构) → 默认安全路径: 叉掉")
                return action_click_box(
                    popup_x, "dismiss notification popup (弹窗叉叉)")
            # 只有确认、无取消/叉 → fail-closed, 这里不碰。

        return None

    def _try_go_lobby(self, screen: ScreenState) -> Optional[Dict[str, Any]]:
        """Try to navigate back to lobby."""
        if screen.is_lobby():
            return None
        return self.nav_home(screen, f"{self.name} go-lobby")
