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


# ── harness-aware 墙钟 (2026-07-25) ──────────────────────────────────────
# ⛔为什么不能直接用 time.time(): 昨天把 survey 超时从 tick 计数改成墙钟
# (_SURVEY_MAX_SEC=90) 是对的, 但**逐帧门控时我每步审核要 ~60s**, 那 60s
# 也算进了 skill 的预算 → survey 必然假超时收工 = 我自己把"AP 一点没灌"
# 的故障重新造出来一遍。step 门的停顿是 harness 偷走的时间, 不是游戏时间。
# ⇒ 一切 skill 超时预算走 game_clock(); 只有"真实世界过了多久"(录制时间戳/
#   日志时间)才用 time.time()。
_HARNESS_PAUSED: float = 0.0


def add_harness_pause(secs: float) -> None:
    """执行层告知: 刚刚有 `secs` 秒是 harness 停顿(step 门等人放行)。"""
    global _HARNESS_PAUSED
    if secs and secs > 0:
        _HARNESS_PAUSED += secs


def harness_paused_total() -> float:
    return _HARNESS_PAUSED


def game_clock() -> float:
    """墙钟, 扣掉 harness 停顿。所有 skill 超时预算的唯一时间源。"""
    return time.time() - _HARNESS_PAUSED


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
        # pipeline 置位(2026-07-21): 上一 tick 本 skill 的 click/back/swipe 被
        # _dedup_click 稳定门/hold 转成 wait(未落地) → True。skill 可据此避免
        # 假前进(mutate-before-ack 防线的 root 信号)。
        self.action_suppressed: bool = False
        # ⭐pipeline 置位(2026-07-25): 上一 tick 全局 interceptor 抢先接管了本帧
        # (skill.tick 那一帧**根本没被调用**), 这里记它处理掉了什么。
        # 为什么需要: 很多 skill 的"落地证据"就是奖励弹窗本身(buy_pyroxene
        # `看到 获得奖励 → _bought=True`), 而 interceptor 的职责恰恰是关掉奖励
        # 弹窗 —— **证据被截胡**, skill 永远等不到, 于是回退去重按购买键。
        # live 实锤 2026-07-25: 每日免費包已买成(信用点+10K/AP+10 已到账), 但
        # _bought 仍 False → 下一 tick "re-press FREE buy"。这次无害(免費键已
        # 消失), 但同形状落在 shop/ticket_sweep 上就是对着仍可点的付费键重复购买。
        self.interceptor_handled: str = ""
        self._timers: Dict[str, float] = {}

    # ── 墙钟计时器 (2026-07-25) ────────────────────────────────────────────
    # ⛔为什么必须有: 全仓大量超时写成 `self._phase_ticks > N` / `_hold = N`
    # 这类 **tick 计数**, 但 tick 的墙钟长度不是常量 —— 2026-06-14 zero-wait
    # 改造把非 loading wait 的 sleep 压到 0.12s(server/app.py:1425), 那批按
    # ~1.6 s/tick 年代写的常量集体缩水。
    #
    # ⭐实测口径(scratchpad/tickclock3.py, 28 个 run / 2646 个连续 wait tick,
    #   取每段 (t_end-t_start)/(n-1) 摊薄整秒量化误差):
    #     per-run s/tick  min 0.154 / median 0.379 / max 2.294
    #   **高位那几个是 step_mode 人工停顿的指纹**(最慢段 32s = 我在按 go),
    #   不是链路真实速度 —— 所以**绝不能用均值**(0.563, 已被污染; 我第一版
    #   就是这么算错的, 还据此错误驳回了 workflow 的正确结论)。自主跑真实
    #   区间 **0.15-0.25 s/tick**。
    #   ⇒ 估算 tick 超时的墙钟, 必须用**最快**那一端 0.15 s/tick:
    #     超时是"最早什么时候会触发", 快 tick 让它提前到期, 那才是事故面。
    #     N=4 → 0.6s;  N=8 → 1.2s;  N=30 → 4.5s;  N=120 → 18s。
    # 而且这个速率还会随帧源(scrcpy/ADB)、模型、分辨率、是否录轨迹继续漂,
    # 换句话说 **tick 数根本不是时间单位**。
    # ⇒ 凡是"等真实世界发生某事"(动画/弹窗/战斗/网络/OCR 稳定)的超时一律
    #   走这里。仍适合 tick 计数的只有"扫了几帧都没看到东西"这种纯感知计数。
    # ⚠时间源一律 game_clock() 而非 time.time() —— step 门的人工停顿必须扣掉,
    # 否则逐帧门控本身会把每个 skill 的超时预算烧光(见模块顶部注释)。
    def clock(self) -> float:
        """skill 可见的墙钟(已扣 harness 停顿)。"""
        return game_clock()

    def mark(self, key: str) -> None:
        """给 key 打时间点(重复调用 = 重新计时)。"""
        self._timers[key] = game_clock()

    def since(self, key: str) -> float:
        """距上次 mark(key) 的秒数。没打过点 → 就地打点并返回 0.0
        (让 `if self.since(k) > T` 在首次调用时天然不触发)。"""
        t = self._timers.get(key)
        if t is None:
            self._timers[key] = game_clock()
            return 0.0
        return game_clock() - t

    def expired(self, key: str, secs: float) -> bool:
        """墙钟超时判据。首次调用必为 False(见 since)。"""
        return self.since(key) >= secs

    def clear_timer(self, key: str) -> None:
        self._timers.pop(key, None)

    # ── 购买框结构判据(全仓共用) ─────────────────────────────────────────
    # ⛔2026-07-25 全仓金钱审计的共同结论: 各 skill 各写一份 `_buy_dialog`,
    # 而它们**全都单点依赖"body 里有青辉石 cls"**。schedule 那起 30 青辉石
    # 事故的帧(run_20260724_201229 t0101)实锤: 屏上写着「購買課程表票券 單價
    # 💎30」, YOLO **body 一个青辉石都没检出**(小图标压在深色价格条上) →
    # 整条防线哑火。**单点防线 = 没有防线**。
    #
    # 这里给全仓一个正交的**结构**通道: 数量步进器(MIN/MAX/加减号)。
    # 它是"要你选购买数量"这件事的物理必然, 与图标识别完全独立:
    #   購買AP 框    run_20260711_144712/t30: 加号0.963 MAX0.962 MIN_灰0.957
    #   購買课程表票框 run_20260724_201229/t0101: MAX0.97 减号灰0.97 加号0.96
    # 顶栏那排 加号(TOPBAR_PLUS, 货币旁的"+")必须排除 → 只看 cy > 0.12。
    # 阈值 0.20 = 模型下限: 危险检测器要尽可能灵敏, 误报的代价只是取消+退出。
    _QTY_STEPPER_CLS = ("MAX_可点击", "MAX_灰色", "MIN_可点击", "MIN_灰色",
                        "加号", "加号灰色", "减号", "减号灰色")

    def has_qty_stepper(self, screen: "ScreenState",
                        body_top: float = 0.12, conf: float = 0.20) -> bool:
        """body(cy>body_top)里出现数量步进器 = 这是个"要花钱选数量"的框。"""
        for b in (getattr(screen, "yolo_boxes", None) or []):
            if b.confidence < conf:
                continue
            if (b.y1 + b.y2) / 2 <= body_top:
                continue                      # 顶栏货币旁的 "+" 不算
            if b.cls_name in self._QTY_STEPPER_CLS:
                return True
        return False

    # ── 竣工判据 exit assertion (2026-07-25 v1: 只观测不干预) ──────────────
    # ⛔用户 2026-07-25 点破全项目最贵的盲区: "为什么不把课程表票用干净, 咖啡厅
    # 为什么干活也不干干净, 你这测试没有意义啊" —— 我们一直在验**代码路径跑通**,
    # 从来没有验过**活干完了**。每个 skill 按自己内部循环条件退出(Schedule 问
    # "转完一圈没", Bounty 问"还有红点没", EventQuest 问"阶段走完没"), **没有
    # 一个在出口处问"我该消耗的资源归零了没"** —— 所以 7 票 / 5 票 / 253 AP /
    # swept=0 两连, 全是靠用户肉眼发现的。
    #
    # v1 故意**只观测不干预**: 出口处报三态, 写进日志和 SkillResult, 先攒真实
    # 分布再谈自动补跑。UNKNOWN 与 LEFTOVER 必须分开 —— "读不出" 和 "确实没干完"
    # 是两种病(前者是感知, 后者是策略), 混成一个 bool 就永远查不出是哪个。
    #
    # 子类覆写它, 返回 (verdict, detail):
    #   "CLEAN"    该消耗的都消耗干净了
    #   "LEFTOVER" 确认还有剩(detail 写清剩多少)  ← 要人看的
    #   "UNKNOWN"  读不出/没测量, 不知道           ← 也要人看的
    # 默认 UNKNOWN: 没声明判据 = 没人审计, 如实说不知道, 绝不默认 CLEAN。
    def exit_report(self) -> Tuple[str, str]:
        return ("UNKNOWN", "未声明竣工判据")

    def reset(self) -> None:
        """Reset skill state for a fresh run."""
        self.sub_state = ""
        self.ticks = 0
        self._log_lines = []
        self._timers = {}
        # 被吞信号是"上一个动作没落地", 全新一轮没有上一个动作 —— 不清会让刚
        # 进场的 skill 继承别人的信号: daily_routine 委托链上 sub.reset() 后紧接
        # 着 sub.action_suppressed = self.action_suppressed, 于是新 sub 第 1 tick
        # 就拿前一个 sub 被吞的点击去做入口对账, 回滚一个它从没推进过的状态。
        self.action_suppressed = False
        self.interceptor_handled = ""

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
                   side: str = "right", span_iw: float = 5.0,
                   pad_iw: float = 0.10, y_pad_bh: float = 0.80):
        """DIGIT-ONLY read of the number next to a currency/count icon.

        YOLO locates the icon (e.g. TICKET_BOUNTY, TOPBAR_AP); we crop the
        digit strip beside it and OCR only the digits. This is the ONLY place
        OCR is used (per spec: YOLO for everything, OCR for digits) and works
        regardless of the global _OCR_ENABLED nav-OCR switch.

        ⛔2026-07-27 单位换成**图标自身尺寸**(原来是屏幕宽度比例)。屏幕比例
        只在标定它的那个分辨率/宽高比上成立, 而本系统的帧有 scrcpy/ADB/DXcam
        三条来路、语料实测 19 种分辨率、宽高比 1.48~1.79。理由与实测见
        `brain.pipeline.icon_strip` 的注释。
        y 留白默认 0.80bh 而不是原来的 0.25bh —— DB 文本检测器要行外留白才
        肯出框, 0.25 会整条返回空(arena 2026-07-17 已实锤过一次, 当时没传导)。

        Args:
            icon_cls: cls name(s) of the icon box to anchor on.
            side: which side the number sits — "right" (top-bar currencies,
                  tickets) or "left".
            span_iw: strip width in ICON WIDTHS.
            pad_iw: gap between icon edge and strip start, in icon widths.
            y_pad_bh: vertical margin above/below the icon box, in icon heights.
        Returns:
            (current, total) from pipeline.parse_count — total may be None;
            or None if the icon isn't found / nothing read. Caller decides.
        """
        box = self.find_cls(screen, icon_cls, conf=conf)
        if box is None or screen.frame is None:
            return None
        iw = max(1e-6, box.x2 - box.x1)
        bh = max(1e-6, box.y2 - box.y1)
        y1 = max(0.0, box.y1 - bh * y_pad_bh)
        y2 = min(1.0, box.y2 + bh * y_pad_bh)
        if side == "right":
            x1 = min(1.0, box.x2 + pad_iw * iw)
            x2 = min(1.0, x1 + span_iw * iw)
        else:  # left
            x2 = max(0.0, box.x1 - pad_iw * iw)
            x1 = max(0.0, x2 - span_iw * iw)
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
            # ⛔`_force_settle`(2026-07-28 live 实锤): 这条兜底**绝不许在动画帧上
            # 动手**。它的全部合法性建立在上面那句契约上 ——「语境内的该确认由拥有
            # 语境的 skill 排在本 helper 之前处理掉」。而 skill 的判定带是**窄**的
            # (schedule `_DIALOG_BAND` y 0.66~0.90), 本 helper 的 `_POPUP_BTN_BAND`
            # y 0.45~0.95 **宽得多** —— 弹窗**弹入动画**中確認还没落到最终位置时,
            # 窄带漏掉、宽带接住 ⇒ 契约在动画那一两帧上必然破裂, 于是
            # **skill 该点的「確認」被这里当成非预期弹窗叉掉了**。
            # 实测 2026-07-28 tick15: 課程表報告 弹出瞬间, PRIORITY 1 没命中,
            # 这里打出 "通知弹窗(确认+叉结构) → 叉掉", 落点是全體課程表 popout 的
            # X (0.889,0.141) —— 报告被叉掉, PRIORITY 1 的派遣落账(_ticket_read_
            # pending / _day_dispatched)整段跳过 = 单日上限台账少记一次。
            # 同族第三例(前两: 掃蕩确认框被当完成弹窗 / cafe 領取锚在动画帧)。
            # ⇒ 等稳定帧再判。settle 门自带 >4s 逃生放行, 不会死锁。
            def _settled(act):
                act["_force_settle"] = True
                return act

            popup_cancel = self.find_cls(
                screen, UC.BTN_CANCEL, conf=0.30, region=_POPUP_BTN_BAND
            )
            if popup_cancel is not None:
                self.log("通知弹窗(确认+取消结构) → 默认安全路径: 取消(等稳定帧)")
                return _settled(action_click_box(
                    popup_cancel, "dismiss notification popup (取消键)"))
            popup_x = self.find_cls(screen, UC.BTN_CLOSE_X, conf=0.30)
            if popup_x is not None:
                self.log("通知弹窗(确认+叉结构) → 默认安全路径: 叉掉(等稳定帧)")
                return _settled(action_click_box(
                    popup_x, "dismiss notification popup (弹窗叉叉)"))
            # 只有确认、无取消/叉 → fail-closed, 这里不碰。

        return None

    def _try_go_lobby(self, screen: ScreenState) -> Optional[Dict[str, Any]]:
        """Try to navigate back to lobby."""
        if screen.is_lobby():
            return None
        return self.nav_home(screen, f"{self.name} go-lobby")
