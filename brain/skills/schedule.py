"""ScheduleSkill — Blue Archive 課程表 daily routine (clean pure-YOLO rewrite).

Probe-verified flow (interactive probe 2026-06-01, full spec in
data/_schedule_probe_log.md). Every click resolves through a YOLO ui cls
(ui_classes) or a fused_avatar head box — NO OCR for navigation, NO hardcoded
room coordinates. OCR is used for the ONE thing it's good at: reading the
held-ticket digits 「持有票券 X/7」.

High-level flow
---------------
1. lobby → click NAV_SCHEDULE → region-select screen.
2. Click 夏莱办公室 (list row 0 — region tiles are under-trained, GAP) → enter
   its region-internal 選擇課程表 screen → ARROW_LEFT once = jump to the LAST
   (newest) academy region (list wraps). Traverse backwards (ARROW_LEFT) from
   there.
3. Per region: click SCHED_ALL (全體課程表) → popout. fused_avatar reads the
   heads, clustered into rooms (≤3 heads/room, x-gap ~0.057, room-gap ~0.15;
   rows at y≈0.39 / 0.60 / 0.81). Detect ROOM_LOCKED:
     • Case A (locks present, region not max level): goal = level this region.
       Dispatch EVERY un-clicked room to spend tickets here.
     • Case B (no locks, max level): goal = only the dashboard target students.
       Dispatch a room iff a target head sits in it; else skip.
   When no room is left to click → close popout (BTN_CLOSE_X) → ARROW_LEFT next.
4. Per-room dispatch (one room): click a head → 課程表資訊 (SCHED_START) → click
   SCHED_START → 課程表報告 (BTN_CONFIRM) → click confirm → heads go green, ticket
   −1, back to popout.
5. End: digit-OCR 「持有票券 X/7」 == 0 → done.
6. Fallback: traversed every region (wrapped to start) and targets not found but
   tickets remain → dispatch any un-clicked room to spend the rest (don't waste).

Single-step state machine (strict — one state, one action per tick; see probe
log "自動循環寫糙的教訓"). The two dispatch popups are checked FIRST in tick(),
above the sub-state dispatch, by priority:
   1. report popup   : BTN_CONFIRM in center-bottom band  → click confirm
   2. info popup      : SCHED_START                        → click start
so they're handled identically no matter which sub_state we're in. We never
blind-click head positions to "advance" — that lands on popup backgrounds
(the bug that drained tickets). Ticket dispatch is confirmed by the digit-OCR
count dropping.

Detectors (pipeline.SKILL_YOLO_MAP["Schedule"] = "ui+avatar"):
  ui      — UC.* button classes (find_cls / find_all_cls).
  avatar  — fused_avatar (251 student heads), model_tag=="avatar", 中文角色名.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from brain.skills.base import (
    BaseSkill, ScreenState, YoloBox,
    action_click, action_click_box,
    action_wait, action_back, action_done,
)
from brain.skills import ui_classes as UC

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_APP_CONFIG_FILE = _DATA_DIR / "app_config.json"
# ⛔按游戏日累计的派遣台账(2026-07-25 事故后新增)。旧的 _dispatch_count 是
# **per-run** 的: 事故当天早些 session 已派 4 次, 本跑只数到 3, 于是 >=7 的硬
# 上限形同虚设, 票早已耗尽却照点「課程表開始」→ 弹出青辉石购买框。
_SCHED_STATE_FILE = _DATA_DIR / "schedule_state.json"


def _game_day() -> str:
    """BA 游戏日(04:00 刷新) ISO 日期 —— 日累计台账的 key。

    ⛔2026-07-27 实测事故: 旧版用**裸 `datetime.now()`(host 本地时钟)**, 而游戏
    日界锚在**设备/服务器时区**。实测 host=UTC-4 / device=Asia/Shanghai(UTC+8)
    → **device = host + 12h**:
        host 03:15 → _game_day() = 2026-07-26
        host 04:32 → _game_day() = 2026-07-27     (同一个游戏日内翻了天)
    后果: 台账 key 一翻, `_load_sched_state()` 判 game_day 不匹配 → 返回
    `dispatched=0` → **单日上限闸(_day_dispatched >= 7)从零重新计数**。
    今晚实况: 台账 = {"game_day":"2026-07-26","dispatched":4}, 而当天真实派遣
    **7** 次(票 7→0) —— 闸已经被打断, 而票=0 还去点開始正是弹青辉石购买框的路。

    改成锚在**服务器时区**(繁中服 UTC+8), 与 host 在哪个时区无关。
    (不用 adb 读设备时钟: 那是每次调用一次 IPC, 挡在热路径上; 服务器时区是
     常量, 不会因为换机器而变。)
    """
    from datetime import datetime, timedelta, timezone
    _SERVER_TZ = timezone(timedelta(hours=8))       # BA 繁中服
    return (datetime.now(_SERVER_TZ) - timedelta(hours=4)).date().isoformat()


def _load_sched_state() -> dict:
    try:
        if _SCHED_STATE_FILE.exists():
            s = json.loads(_SCHED_STATE_FILE.read_text(encoding="utf-8"))
            if s.get("game_day") == _game_day():
                return s
    except Exception:
        pass
    return {"game_day": _game_day(), "dispatched": 0}


def _save_sched_state(state: dict) -> None:
    try:
        _SCHED_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _SCHED_STATE_FILE.write_text(json.dumps(state, ensure_ascii=False),
                                     encoding="utf-8")
    except Exception:
        pass

# ── tuning knobs ──────────────────────────────────────────────────────────
_CLS_CONF = 0.30            # default UI cls confidence floor
_AVATAR_CONF = 0.30         # fused_avatar head confidence (probe used 0.30)

_MAX_TICKETS = 7            # 持有票券 X/7 — total ticket capacity
_ROOM_X_GAP = 0.085         # heads within this x-gap belong to the same room
_ROOM_MAX_HEADS = 3         # a room holds at most 3 students
# Dispatched rooms render DIMMED (live calib 2026-06-09: dispatched rooms mean
# head brightness 101-110 vs fresh 174-210 — clean bimodal split). Below this
# ⇒ already dispatched, never re-click. Model-free, works across sessions.
_DIM_DISPATCHED_MAX = 140.0

# Dispatch popups (info / report) live in a center-bottom band. Both
# SCHED_START and the report's BTN_CONFIRM sit at cx≈0.499, cy≈0.76-0.77.
_DIALOG_BAND = (0.30, 0.66, 0.70, 0.90)
# ★ MONEY DEFENSE: a 青辉石 icon in the dialog BODY (NOT the top-bar balance at
# cy<0.10) = a spend-pyroxene 购买课程表券 dialog. VERIFIED on run_20260602_200900:
# the buy-dialog pyroxene sits at cx≈0.59/0.33, cy≈0.577 (conf 0.43-0.88). This
# band covers it with margin and stays clear of the report's _DIALOG_BAND (cy
# 0.66+), so it never false-fires on a normal 课程表报告.
_PYROXENE_BODY_REGION = (0.20, 0.12, 0.82, 0.64)
# ⛔购买框结构判据(2026-07-25 事故后新增, 坐标取自 run_20260724_201229
# tick_0101 实测): 数量步进器一排在 cy≈0.480, 覆盖带留足余量。
# 課程表報告 弹窗**没有**这排控件, 所以不会误伤正常派遣流程。
_QTY_STEPPER_CLS = (UC.QTY_MIN, UC.QTY_MIN_GREY, UC.QTY_MAX, UC.QTY_MAX_GREY,
                    UC.QTY_MINUS, UC.QTY_MINUS_GREY, UC.TOPBAR_PLUS)
_QTY_STEPPER_REGION = (0.22, 0.30, 0.80, 0.62)

# ── GAP fallbacks (under-trained cls — documented in probe log) ───────────
# Region tiles (SCHOOL_*) are ~3-12 frames each and routinely missed (v6 gap
# #29). We click 夏莱办公室 by its list-row position when the cls isn't seen.
_OFFICE_ROW_POS = (0.64, 0.22)        # region-select list row 1 (夏莱办公室)
# ARROW_LEFT (117f) is usually solid, but fall back to its symmetric left-edge
# coord if the cls isn't detected this frame.
_ARROW_LEFT_POS = (0.023, 0.500)

# Per-sub-state tick budgets — every phase is bounded, never dead-waits.
_ENTER_MAX = 25
_NAVIGATE_MAX = 14
_ROSTER_MAX = 18
_OPEN_ROOM_MAX = 12
# Reaction time: after clicking a room head, the 課程表資訊 info popup takes
# ~1-2s to open. Wait this many ticks for it (tick() PRIORITY 2 catches
# SCHED_START) before concluding the click failed (user: 给点击反应时间).
_OPEN_ROOM_SETTLE = 4          # (保留: 其他处引用) — 开房等待已改墙钟, 见下
_OPEN_ROOM_SETTLE_SEC = 3.0    # 課程表資訊 弹窗渲染实测 1-2s, 3s 宽松兜底
_SWITCH_MAX = 12
_EXIT_MAX = 14
# ⭐区域身份指纹(2026-07-25 帧证据修): 5 个区域的**区域内屏 cls 集合完全一样**
# (全体课程表/左右切换/顶栏), YOLO 分不出谁是谁 —— 唯一区分是右上「RANK N 区域名」
# 横幅。用感知哈希当到达态判据(纯图像变化, 不是 OCR 文字匹配)。
# 实测(run_20260724_185527, 228 tick): 同区域哈希恒定成一簇, 换区汉明距远大于 6。
# 只裁标题那一行, 不含下方会动的奖励进度条。
_REGION_TITLE_BAND = (0.705, 0.125, 0.995, 0.178)
# 标定(2026-07-25, 实测 6 屏目检过标题 + trajectory 231 帧):
#   同区最大 MAD = 4.41(含背景不同/右箭头退回同一区)
#   异区最小 MAD = 12.05
# 取 8.0 居中。⚠popout 淡出转场帧 MAD 可达 77 → 到达判定必须**连续两帧**确认。
_REGION_DIFF_MAD = 8.0
_SWITCH_GAP_SEC = 5.0          # 这么久标题还没变 → 改用坐标 GAP 落点
_SWITCH_VERIFY_SEC = 14.0      # 再等不到 → 大声报警 fail-closed, 绝不假装切过
_POPOUT_CLOSE_SEC = 1.6        # 关 popout 后等它真关掉的墙钟(到期才允许重发)
# Reaction time: before concluding a region has "no room/target", re-scan this
# many frames so fused_avatar (flickery per-frame on the small popout heads) and
# the layout have time to settle (user: 给点击和模型识别一些反应时间).
# 10 (was 6) + settle 6 (was 3): live 2026-06-09 the post-dispatch celebration
# ANIMATION (floating chibi art) occludes the middle room cards for seconds —
# rescans inside that window saw "no rooms" and the bot LEFT the region with
# 3 tickets unspent. Wider budget lets the animation clear.
_BARREN_SCAN_MAX = 10
_HEAD_RENDER_SETTLE = 6     # ticks to let popout heads render before first pick

# Traverse at most this many regions before giving up the full circle. BA has
# ~10 schedule regions; 14 leaves margin for re-detect retries.
_MAX_REGIONS = 14


def _load_schedule_targets() -> List[str]:
    """Read the schedule target-student list from the active app_config profile.

    Same pattern as CafeSkill._load_invite_targets: app_config.json →
    active_profile → profiles[active].schedule_target_students. Returns a list
    of 中文角色名 matching fused_avatar cls_name (e.g. "莉央(战斗)"). Empty list
    ⇒ Case B has no targets, so the fallback (spend remaining tickets on any
    room) takes over.
    """
    try:
        if not _APP_CONFIG_FILE.exists():
            return []
        data = json.loads(_APP_CONFIG_FILE.read_text("utf-8"))
        active = data.get("active_profile", "default")
        profile = (data.get("profiles") or {}).get(active, {})
        raw = profile.get("schedule_target_students")
        if not isinstance(raw, list):
            return []
        out: List[str] = []
        for item in raw:
            name = str(item or "").strip()
            if name and name not in out:
                out.append(name)
        return out
    except Exception:
        return []


class ScheduleSkill(BaseSkill):
    # Lobby schedule entries that carry a dot when there's work to do.
    _LOBBY_DOT_ENTRIES = [UC.NAV_SCHEDULE, UC.SCHED_TICKET]

    def should_run(self, screen: ScreenState) -> bool:
        # Badge-gating is a LOBBY decision only: mid-flow screens (e.g. the
        # 全體課程表 popout we may be resumed on) show 课程表票 WITHOUT a dot
        # and would wrongly skip the whole skill (live 2026-06-09: resumed on
        # the popout → "skip Schedule (no dot)" with 3 tickets left). Off-lobby
        # ⇒ run; enter/recover handles navigation from wherever we are.
        if not screen.is_lobby():
            return True
        return self.dot_on_entry(screen, self._LOBBY_DOT_ENTRIES)

    def __init__(self):
        super().__init__("Schedule")
        # 一圈 ≈ 5 区 × ~25 tick ≈ 125。放开捡漏圈次数后(2026-07-25 修), 花光
        # 7 张票最坏要 3-4 圈 = 375-500 tick → 旧值 320 会在第 3 圈被 pipeline
        # 判超时 reset(与 event_quest 400 同款: 上限卡在工作量之下, 活没干完
        # 就被砍)。给到 900 留足余量; 真跑飞由"某圈零派出即退"+tickets==0 收敛。
        self.max_ticks = 900
        self._init_state()

    # ── state init / reset ────────────────────────────────────────────────

    def _init_state(self) -> None:
        self._phase_ticks: int = 0          # ticks spent in current sub_state
        self._open_room_t0: float = 0.0     # 開房等待墙钟起点(0=未开始计时)
        self._enter_attempts: int = 0
        self._targets: List[str] = []
        # ⛔2026-07-25 三态化: 旧值是 `int = -1` 的**假三态 sentinel**, 一个 bug
        # 两头出血 —— ①金钱侧几道闸写的是 `_tickets == 0`, -1 不等于 0 于是
        # 源头闸哑火; ②资源侧的兜底闸写的是 `_tickets is not None and > 0`,
        # -1 既不是 None 也不 >0, 于是**捡漏永远够不着**, 剩票全废。同一个变量
        # 被两批代码按两套语义读, 因为 -1 到底算"未知"还是算"一个数"从没定义过。
        # 改成真三态: None=未知 / 0=用光 / >0=还有。语义写死在这里:
        #   花钱类判断 → 必须 `_tickets == 0` 或 `is not None and > 0`(未知不放行)
        #   找活类判断 → 可以 `is None or > 0`(未知时继续找, 反正花不出去)
        self._tickets: Optional[int] = None   # None=未知 / 0=用光 / >0=还有
        self._dispatch_count: int = 0       # confirmed reports = tickets spent (hard cap)
        # 今日累计派遣(跨 run 持久, 硬上限真正的依据)
        self._day_dispatched: int = int(_load_sched_state().get("dispatched", 0))
        # traversal bookkeeping
        self._office_clicked: bool = False  # clicked 夏莱办公室 yet?
        self._jumped_to_last: bool = False  # done the initial ARROW_LEFT jump?
        self._regions_seen: int = 0         # regions whose popout we've opened
        self._full_circle: bool = False     # traversed all regions → fallback
        self._circle_start_dispatch: int = 0  # _dispatch_count at circle start(空转退出用)
        self._circle_start_switches: int = 0  # _verified_switches at circle start(假圈检测)
        self._region_count: int = 0         # 区域列表帧实测区数(0=没测到, 回落 _MAX_REGIONS)
        self._circle_first_sig = None       # 本圈起点区域的标题指纹(None=还没锚)
        self._circle_closed: bool = False   # 标题指纹已转回起点 = 一圈走完
        self._popout_close_issued: bool = False  # 关 popout 已发出, 等帧证据(after-ack)
        self._ls_recoveries: int = 0        # Location-Select bounce count (row-walk + cap)
        # ⭐区域切换到达态验证(2026-07-25): 点 ARROW_LEFT 前的标题指纹 / 起点墙钟 /
        # 已发 tap 数。标题真变了才算切成功 —— 绝不用"我发过点击"当证据。
        self._switch_from_sig = None          # 起点标题缩略图(np.ndarray|None)
        self._sig_pending = None              # 基线候选(需连续两帧一致)
        self._arrive_pending = None           # 到达候选(需连续两帧一致)
        self._switch_t0: float = 0.0
        self._switch_taps: int = 0
        self._verified_switches: int = 0    # 帧证据确认的真实换区次数
        self._jump_issued: bool = False     # jump-to-last 的点击已发出(等 ack)
        # per-region (reset on each region switch)
        self._region_locked: Optional[bool] = None   # case A (True) / B (False)
        self._clicked_heads: List[Tuple[float, float]] = []  # heads dispatched here
        self._ticket_read_pending: bool = False  # re-read ticket after a dispatch
        self._barren_scans: int = 0          # consecutive no-room scans this region
        self._head_settle: int = 0           # ticks left for popout heads to render

    def reset(self) -> None:
        super().reset()
        self._init_state()
        self._targets = _load_schedule_targets()
        if self._targets:
            self.log(f"schedule targets: {self._targets}")
        else:
            self.log("no schedule_target_students configured — Case B will "
                     "spend leftover tickets on any room (fallback)")

    def _goto(self, sub_state: str) -> None:
        """Switch sub_state and reset the per-phase tick counter."""
        self.sub_state = sub_state
        self._phase_ticks = 0
        self._open_room_t0 = 0.0      # 墙钟计时随状态切换归零
        if sub_state == "switch":
            # 每次重新进入 switch 都要重新取起点指纹(上一次的已作废)
            self._switch_from_sig = None
            self._sig_pending = None
            self._arrive_pending = None
            self._switch_t0 = 0.0
            self._switch_taps = 0

    def _reset_region(self) -> None:
        """Clear per-region state on entering a fresh region's popout."""
        self._region_locked = None
        self._clicked_heads = []
        self._green_marks = []   # accumulated 绿勾 sightings (positions repeat across regions)
        self._lock_seen = False
        self._lock_scan_frames = 0
        self._ticket_read_pending = False
        self._barren_scans = 0          # consecutive no-room scans this region
        self._head_settle = _HEAD_RENDER_SETTLE  # let heads render before 1st pick
        self._open_room_t0 = 0.0     # 開房等待墙钟起点(0=未开始计时)

    # ── ticket digit-OCR ──────────────────────────────────────────────────

    # 「持有票券 X/7」 sits BELOW the popout title at cy≈0.215 — the probe's
    # initial guess (cy≈0.10-0.17) was ~0.1 too HIGH, so digit-OCR read EMPTY
    # the entire run → tickets=-1 → the ==0 money-gates never fired → 青辉石 buy
    # bug. Calibrated on run_20260602_200900 tick_0064: this band reads "1/7"
    # cleanly. screen.frame is the raw full-res BGR array (run_digit_ocr crops +
    # upscales).
    _TICKET_REGION = (0.48, 0.185, 0.67, 0.25)

    # ── 竣工判据 ─────────────────────────────────────────────────────────
    def exit_report(self):
        """课程表的竣工判据 = 票用干净了没。

        ⛔用户原话: "为什么不把课程表票用干净" —— 昨天剩 7 张、今天剩 5 张,
        两次都是**用户肉眼**发现的, 因为 skill 的出口只问"转完一圈没"。"""
        _day = self._day_dispatched
        if self._tickets == 0:
            return ("CLEAN", f"票已用光(今日累计派 {_day} 次)")
        if self._tickets is None:
            return ("UNKNOWN", f"票数从未读出(今日累计派 {_day} 次) — "
                               f"**不知道**还剩几张")
        if self._tickets > 0:
            return ("LEFTOVER",
                    f"还剩 {self._tickets}/{_MAX_TICKETS} 张票没花掉"
                    f"(今日累计派 {_day} 次, 走过 {self._regions_seen} 个区, "
                    f"帧证实换区 {self._verified_switches} 次)")
        return ("UNKNOWN", f"票数异常 {self._tickets}")

    def _read_tickets(self, screen: ScreenState) -> Optional[int]:
        """Read 持有票券 X/7 via digit-OCR. Returns current count, or None.

        Independent of the global nav-OCR switch (run_digit_ocr always works).
        Never raises / never blocks — None just means "couldn't read this
        frame", and the caller relies on max_ticks + "no room to click" as the
        real safety net rather than dead-waiting on the count.
        """
        if screen.frame is None:
            return None
        try:
            from brain.pipeline import run_digit_ocr, parse_count
        except Exception:
            return None
        raw = run_digit_ocr(screen.frame, self._TICKET_REGION)
        parsed = parse_count(raw)
        if parsed is None:
            if raw:
                # ⭐拒绝必留痕(2026-07-24 workflow[30]): 静默 None 使 tickets=-1
                # 的根因不可见, 今天 5 区全 -1 只能帧考古
                self.log(f"tickets 读拒: parse_count fail (raw {raw!r})")
            return None
        cur, _tot = parsed
        # ⭐首位复读修复(2026-07-25 live 实锤): run_digit_ocr 碎片拼接会复读首位
        # 数字 —— 实测 raw '55/7'(真值 5/7), memory 早记过 1/7→11/7 同款。
        # 旧码只会"读拒"→ _tickets 停在 -1 → _pick_room 的兜底闸
        # (self._tickets is not None and > 0) 永远挡住 → **空房间+剩票全废**。
        # 只在**已经越界**(cur>上限, 本来就要丢弃)时尝试去掉重复的首位, 且结果
        # 必须 <= 上限才接受。方向保守: 修出来的值只会更小 = 派得更少, 不会多花。
        if cur is not None and cur > _MAX_TICKETS:
            s = str(cur)
            if len(s) >= 2 and s[0] == s[1]:
                fixed = int(s[1:])
                if 0 <= fixed <= _MAX_TICKETS:
                    self.log(f"tickets 首位复读修正: {cur}→{fixed} (raw {raw!r})")
                    cur = fixed
        if cur is None or cur < 0 or cur > _MAX_TICKETS:
            self.log(f"tickets 读拒: cur={cur} 越界/复读伪值 (raw {raw!r})")
            return None
        if cur != self._tickets:
            self.log(f"tickets: {cur}/{_MAX_TICKETS} (raw {raw!r})")
        self._tickets = cur
        return cur

    # ── screen-state helpers (pure YOLO) ──────────────────────────────────

    def _buy_dialog(self, screen: ScreenState) -> bool:
        """购买/数量选择对话框判据。★ HARD money stop。

        ⛔2026-07-25 实锤事故(30 青辉石被花掉): 旧版**单点**依赖"对话框体内检出
        青辉石 cls"。那一帧(run_20260724_201229 tick_0101)屏上明明是
        「購買課程表票券 單價💎30 總購買價格💎30」, 但 YOLO **一个 body 青辉石
        都没检出**(小图标压在深色价格条上), 只检出顶栏余额那个 —— 判据返回
        False, 防线整条哑火, PRIORITY 1 把购买框的「確認」当成報告框的「確認」
        点了下去 = 亲手买票。**单点防线 = 没有防线。**

        改成正交多信号, 任一命中即判购买框(全部取自那一帧的实测检出, conf
        0.94-0.98, 远比那个检不出的 💎 可靠):
          A 数量步进器: MIN/MAX/减号/加号 出现在对话框体内(cy 0.30-0.62)
            —— 只有"选数量再买"的框才有, 課程表報告 绝无。
          B 取消键与确认键同屏 —— 報告框只有孤零零一个 確認, 没有取消。
          C 旧的 body 青辉石(保留, 多一路零成本)。
        """
        # C: 旧路(body 青辉石)
# ⛔2026-07-25 全量 cls 审计删除: 原来这里还并了一路 "清辉石"(master idx2),
        # 注释写着"危险检测器多收一路零成本" —— 实测**训练 0 框 / 92k tick 实战
        # 0 检出**, 那一路从来没收到过任何东西, 只是制造"有两路信号"的假象。
        # idx2 是 idx30「青辉石」的错别字重复类(BA 官方写作 青輝石), 本就不该
        # 被标注。真正有效的正交第二路是**结构信号** has_qty_stepper。
        if self.find_cls(screen, UC.TOPBAR_PYROXENE,
                         conf=_CLS_CONF,
                         region=_PYROXENE_BODY_REGION) is not None:
            return True
        # A: 数量步进器在对话框体内
        if self.find_cls(screen, _QTY_STEPPER_CLS, conf=_CLS_CONF,
                         region=_QTY_STEPPER_REGION) is not None:
            return True
        # B: 取消键 + 确认键 同屏 = 有代价的确认框
        if (self.find_cls(screen, UC.BTN_CANCEL, conf=_CLS_CONF) is not None
                and self.find_cls(screen, UC.BTN_CONFIRM,
                                  conf=_CLS_CONF) is not None):
            return True
        return False

    def _is_schedule(self, screen: ScreenState) -> bool:
        """Any schedule surface: detect_screen_yolo()=='Schedule' OR a region
        tile is on screen (the region-select list carries SCHOOL_* tiles but
        may have none of the SCHED_* cls)."""
        if self.detect_screen_yolo(screen) == "Schedule":
            return True
        return self.find_cls(screen, _SCHOOL_TILES, conf=_CLS_CONF) is not None

    def _sched_all_btn(self, screen: ScreenState) -> Optional[YoloBox]:
        """The 全體課程表 button on a region-internal screen (bottom-right,
        @~0.910,0.921)."""
        return self.find_cls(
            screen, UC.SCHED_ALL, conf=_CLS_CONF, region=(0.74, 0.80, 1.0, 1.0)
        )

    def _roster_open(self, screen: ScreenState) -> bool:
        """The 全體課程表 popout is open when its close-X sits top-right
        (@~0.888,0.138) AND fused_avatar heads / SCHED_ALL title are visible.

        We key off the popout close-X in the top-right band — the region-
        internal screen's SCHED_ALL is a bottom-right BUTTON, not a popout, so
        the close-X is the disambiguator that never appears on the plain
        region screen."""
        if self._popout_close(screen) is not None and (
            len(self._roster_heads(screen)) > 0
            or self.find_cls(screen, UC.SCHED_ALL, conf=_CLS_CONF,
                             region=(0.10, 0.0, 0.90, 0.25)) is not None
        ):
            return True
        # The close-X flickers (live 2026-06-09: popout clearly open, X cls
        # missed one frame → "popout closed" re-click loop). Backup signature
        # that exists ONLY on the popout:
        #   • 课程表票 in the TOP-CENTER header band — the region screen's ticket
        #     counter sits top-LEFT (cx≈0.06), the popout's at cx≈0.45,cy≈0.21.
        # ⛔ REMOVED (2026-06-13, user-caught bug): ROOM_LOCKED → open. The map's
        #    locked BUILDING (需要RANKx, e.g. 需要RANK9) ALSO fires 房间区域未解锁
        #    (live run_20260613_051748 t13: map screen, 房间区域未解锁@0.305,0.764
        #    conf0.95, NO popout) → _roster_open falsely True → skill scanned the
        #    MAP heads as a roster, judged "all dispatched", and SKIPPED the whole
        #    locked region without ever opening the popout. The lock cls is NOT
        #    popout-exclusive, so it can't gate popout-open.
        # y-band widened 0.16→0.26: the popout's centred ticket sits at cy≈0.21,
        # which the old 0.04-0.16 band missed entirely.
        if self.find_cls(screen, UC.SCHED_TICKET, conf=_CLS_CONF,
                         region=(0.30, 0.04, 0.70, 0.26)) is not None:
            return True
        return False

    def _popout_close(self, screen: ScreenState) -> Optional[YoloBox]:
        """The 弹窗叉叉 at the popout's top-right (@~0.888,0.138)."""
        return self.find_cls(
            screen, UC.BTN_CLOSE_X, conf=_CLS_CONF, region=(0.78, 0.05, 1.0, 0.25)
        )

    def _arrow_left(self, screen: ScreenState) -> Optional[YoloBox]:
        """ARROW_LEFT (左切换) on the left edge (@~0.023,0.500)."""
        return self.find_cls(
            screen, UC.ARROW_LEFT, conf=_CLS_CONF, region=(0.0, 0.30, 0.12, 0.70)
        )

    # ── 区域身份(到达态判据) ────────────────────────────────────────────────

    def _region_sig(self, screen: ScreenState):
        """区域内屏「RANK N 区域名」横幅的 96x24 灰度缩略图; 拿不到返回 None。

        为什么需要: 区域内屏各区 cls 集合一模一样(全体课程表/左右切换/顶栏),
        YOLO 无从判断"我到底在哪个区" —— 于是 2026-07-25 那次 5 条 ARROW_LEFT
        日志全是空头支票也没人发现。popout 开着时标题被弹窗盖住, 所以只在
        SCHED_ALL 锚在场(popout 已关)的帧上比。
        ⚠试过 DCT 感知哈希, **判据不成立**: 1050x115 的窄横幅压成 32x32 文字糊
        掉, 实测同区 d=24-34 / 异区 d=4-16 完全混叠。改用缩略图 MAD 后干净可分
        (标定见下方 _REGION_DIFF_MAD)。
        """
        frame = getattr(screen, "frame", None)
        if frame is None:
            return None
        try:
            import cv2
            import numpy as np
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = _REGION_TITLE_BAND
            crop = frame[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]
            if crop.size == 0:
                return None
            g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            return cv2.resize(g, (96, 24),
                              interpolation=cv2.INTER_AREA).astype(np.float32)
        except Exception:
            return None

    @staticmethod
    def _sig_same(a, b) -> bool:
        """两张标题缩略图是否同一个区域(任一为 None 时保守判 True = 没换)。"""
        if a is None or b is None:
            return True
        try:
            import numpy as np
            return float(np.mean(np.abs(a - b))) < _REGION_DIFF_MAD
        except Exception:
            return True

    # ── roster head clustering ────────────────────────────────────────────

    def _roster_heads(self, screen: ScreenState) -> List[YoloBox]:
        """All fused_avatar head boxes in the popout (model_tag=='avatar')."""
        return [
            b for b in (screen.yolo_boxes or [])
            if b.model_tag == "avatar" and b.confidence >= _AVATAR_CONF
        ]

    def _cluster_rooms(self, heads: List[YoloBox]) -> List[List[YoloBox]]:
        """Group heads into rooms. Heads in the same row (similar cy) within
        _ROOM_X_GAP of each other = one room; ≤3 heads/room.

        Probe layout: rows at y≈0.39 / 0.60 / 0.81, each row up to 6 rooms,
        head x-gap within a room ~0.057, room-to-room gap ~0.15.
        """
        if not heads:
            return []
        # bucket by row (cy) — 0.10 tolerance separates the ~0.21-apart rows
        rows: List[List[YoloBox]] = []
        for h in sorted(heads, key=lambda b: b.cy):
            placed = False
            for row in rows:
                if abs(row[0].cy - h.cy) < 0.10:
                    row.append(h)
                    placed = True
                    break
            if not placed:
                rows.append([h])
        rooms: List[List[YoloBox]] = []
        for row in rows:
            row.sort(key=lambda b: b.cx)
            cur: List[YoloBox] = []
            for h in row:
                if not cur:
                    cur = [h]
                    continue
                if (h.cx - cur[-1].cx) <= _ROOM_X_GAP and len(cur) < _ROOM_MAX_HEADS:
                    cur.append(h)
                else:
                    rooms.append(cur)
                    cur = [h]
            if cur:
                rooms.append(cur)
        return rooms

    @staticmethod
    def _room_center(room: List[YoloBox]) -> Tuple[float, float]:
        cx = sum(h.cx for h in room) / len(room)
        cy = sum(h.cy for h in room) / len(room)
        return (cx, cy)

    def _room_already_clicked(self, room: List[YoloBox]) -> bool:
        """True if we've already dispatched this room this region (matched by
        proximity to a stored click point — heads don't move within a region)."""
        cx, cy = self._room_center(room)
        for px, py in self._clicked_heads:
            if abs(px - cx) < _ROOM_X_GAP and abs(py - cy) < 0.10:
                return True
        return False

    def _accumulate_green_marks(self, screen: ScreenState) -> None:
        """绿勾 (12f, FLICKERY) detections are ACCUMULATED per region — one
        sighting anywhere this region permanently marks that spot dispatched,
        so the flicker works FOR us instead of against us. Cleared on region
        switch (_reset_region_state) since popout positions repeat across
        regions."""
        marks = getattr(self, "_green_marks", None)
        if marks is None:
            marks = self._green_marks = []
        for b in (screen.yolo_boxes or []):
            if b.cls_name != UC.GREEN_CHECK or b.confidence < 0.25:
                continue
            if not any(abs(b.cx - mx) < 0.03 and abs(b.cy - my) < 0.03 for mx, my in marks):
                marks.append((b.cx, b.cy))

    def _room_brightness(self, screen: ScreenState, room: List[YoloBox]) -> Optional[float]:
        """Mean pixel brightness over this room's head crops (dim ⇒ dispatched)."""
        try:
            if screen.frame is None:
                return None
            h, w = screen.frame.shape[:2]
            vs = []
            for hd in room:
                x1, y1 = max(0, int(hd.x1 * w)), max(0, int(hd.y1 * h))
                x2, y2 = int(hd.x2 * w), int(hd.y2 * h)
                crop = screen.frame[y1:y2, x1:x2]
                if crop.size:
                    vs.append(float(crop.mean()))
            return (sum(vs) / len(vs)) if vs else None
        except Exception:
            return None

    def _room_dispatched_visual(self, screen: ScreenState, room: List[YoloBox]) -> bool:
        """User mechanic (2026-06-09): after a room is dispatched, OWNED heads
        get a 绿勾 overlay and un-owned heads dim out. Any ACCUMULATED green
        mark inside the room cluster ⇒ already dispatched — never re-click
        (works ACROSS sessions too: fresh entry re-sees the in-game checks)."""
        cx, cy = self._room_center(room)
        for mx, my in getattr(self, "_green_marks", []):
            if abs(mx - cx) < _ROOM_X_GAP * 1.6 and abs(my - cy) < 0.12:
                return True
        return False

    def _log_room_brightness(self, screen: ScreenState, rooms: List[List[YoloBox]]) -> None:
        """Calibration probe for DIM-detection (un-owned heads dim after a room
        is dispatched): log mean head brightness per room so live runs give us
        the threshold data before we enforce anything."""
        try:
            if screen.frame is None or not rooms:
                return
            h, w = screen.frame.shape[:2]
            vals = []
            for r in rooms:
                vs = []
                for hd in r:
                    x1, y1 = max(0, int(hd.x1 * w)), max(0, int(hd.y1 * h))
                    x2, y2 = int(hd.x2 * w), int(hd.y2 * h)
                    crop = screen.frame[y1:y2, x1:x2]
                    if crop.size:
                        vs.append(float(crop.mean()))
                if vs:
                    vals.append(round(sum(vs) / len(vs), 1))
            if vals:
                self.log(f"room-brightness (dim-calib): {vals}")
        except Exception:
            pass

    def _room_has_target(self, room: List[YoloBox]) -> Optional[str]:
        """Return the matched target name if any head in the room is a
        configured target student, else None."""
        if not self._targets:
            return None
        target_set = set(self._targets)
        for h in room:
            if h.cls_name in target_set:
                return h.cls_name
        return None

    def _pick_room(self, screen: ScreenState, rooms: List[List[YoloBox]]) -> Tuple[
            Optional[List[YoloBox]], str]:
        """Choose the next room to dispatch. A room is OFF the table when:
          • clicked this session (proximity memory), OR
          • it visually shows a 绿勾 (user mechanic 2026-06-09: dispatched
            rooms mark owned heads with a green check — cross-session signal).
        Case A (locks) = fill every remaining room to upgrade the region;
        Case B (no locks) = only rooms with a configured target student."""
        self._accumulate_green_marks(screen)
        self._log_room_brightness(screen, rooms)
        candidates = []
        for r in rooms:
            if self._room_already_clicked(r):
                continue
            if self._room_dispatched_visual(screen, r):
                cx, cy = self._room_center(r)
                self.log(f"room@({cx:.2f},{cy:.2f}) 绿勾 → already dispatched, skip")
                continue
            br = self._room_brightness(screen, r)
            if br is not None and br < _DIM_DISPATCHED_MAX:
                cx, cy = self._room_center(r)
                self.log(f"room@({cx:.2f},{cy:.2f}) dim {br:.0f}<{_DIM_DISPATCHED_MAX:.0f} → dispatched, skip")
                continue
            candidates.append(r)
        if not candidates:
            return None, "all rooms dispatched (clicked/绿勾)"

        # Case A (region has locks → level it up): fill EVERY remaining room.
        if self._region_locked:
            # top-to-bottom, left-to-right for determinism
            candidates.sort(key=lambda r: (round(self._room_center(r)[1], 2),
                                           self._room_center(r)[0]))
            return candidates[0], "case-A fill-room (upgrade locked region)"

        # Case B (no locks): only rooms containing a configured target.
        for r in candidates:
            name = self._room_has_target(r)
            if name:
                return r, f"case-B target '{name}'"

        # Fallback: full circle done + tickets remain → spend on any room.
        # Deep-dive C3 (2026-06-09): require a POSITIVE read — the old
        # `is None or > 0` treated "unknown" as "has budget" (and -1 slipped
        # past both '==0' gates). Unknown tickets ⇒ no fallback spending.
        if self._full_circle and (self._tickets is not None and self._tickets > 0):
            candidates.sort(key=lambda r: (round(self._room_center(r)[1], 2),
                                           self._room_center(r)[0]))
            return candidates[0], "fallback spend-leftover"

        return None, "case-B no target in region"

    # ── tick: global popup guards + state dispatch ────────────────────────

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1
        self._phase_ticks += 1

        if self.ticks >= self.max_ticks:
            self.log("timeout, exiting")
            return action_done("schedule timeout")

        # ⛔⛔ HARD MONEY STOP (highest priority — above every other handler):
        # 青辉石 in the dialog body = a 购买课程表券 buy-ticket dialog. Ticket
        # digit-OCR was unreliable (tickets=-1 the whole run → the ==0 gates
        # never fired → it排课 past 0 tickets → the buy popup, whose 确认 got
        # mis-clicked by PRIORITY 1). NEVER let any confirm-handler reach it.
        if self._buy_dialog(screen):
            self.log("⛔ 青辉石买票对话框 — 取消并退出,绝不买")
            self._goto("exit")
            cancel = self.find_cls(screen, [UC.BTN_CANCEL, UC.BTN_CLOSE_X], conf=_CLS_CONF)
            if cancel is not None:
                return action_click_box(cancel, "cancel 买票 dialog (青辉石)")
            return action_back("ESC 买票 dialog (青辉石)")

        # ── popups that can appear in any sub_state (pure YOLO) ──

        # Reward-result popup ("獲得獎勵") — tap to dismiss.
        got_reward = self.find_cls(screen, UC.GOT_REWARD, conf=_CLS_CONF)
        if got_reward is not None:
            self.log("reward-result popup, dismissing (YOLO 获得奖励)")
            return action_click_box(got_reward, "dismiss reward result")

        # Bond / region level-up full-screen splash — tap anywhere advances.
        splash = self.find_cls(
            screen, [UC.BOND_LEVELUP, UC.REGION_LEVELUP], conf=_CLS_CONF
        )
        if splash is not None:
            self.log(f"level-up splash ({splash.cls_name}), tap to dismiss")
            return action_click(0.5, 0.5, f"dismiss splash ({splash.cls_name})")

        # ── DISPATCH POPUP PRIORITY (strict single-step, runs above states) ──
        # These two are the per-room dispatch sequence and must be handled
        # FIRST so they win regardless of sub_state (probe教训: state-machine
        # mis-judgement here = mis-clicks that drained tickets).

        # PRIORITY 1: 課程表報告 result popup — its only actionable is 確認
        # (center-bottom band). Present iff BTN_CONFIRM is in the band AND
        # SCHED_START is NOT (a SCHED_START in-band means the info popup, not
        # the report).
        report_confirm = self.find_cls(
            screen, UC.BTN_CONFIRM, conf=_CLS_CONF, region=_DIALOG_BAND
        )
        if report_confirm is not None and self.find_cls(
                screen, UC.SCHED_START, conf=_CLS_CONF, region=_DIALOG_BAND) is None:
            # ★ Defense ③ (hard cap): each confirmed report = one ticket spent. A
            # day caps at _MAX_TICKETS; exceeding it means tickets ran out and the
            # game is charging 青辉石 to continue — STOP (backstop if defense ②
            # somehow misses the buy dialog).
            # Cap check BEFORE increment and WITHOUT clicking (deep-dive C1,
            # 2026-06-09): the old `+=1 then > cap` let the 8th dispatch through
            # (7>7 False), and the cap-hit path还 action_click_box(confirm) —
            # 第8次那个"确认"可能正是买票框的确认键 = 亲手买票. Cap hit ⇒ wait
            # out, click NOTHING.
            # 上限用**今日累计**(持久台账), 不是本跑计数 —— 2026-07-25 事故当天
            # 早些 session 已派 4 次, 本跑只数到 3, 旧的 per-run 上限完全没兜住。
            if max(self._dispatch_count, self._day_dispatched) >= _MAX_TICKETS:
                self.log(f"⛔ 今日已排课{max(self._dispatch_count, self._day_dispatched)}次 "
                         f">= 单日上限{_MAX_TICKETS} — 票必耗尽,停止防买票(不点任何确认)")
                self._goto("exit")
                return action_wait(300, "at ticket cap → EXIT, do NOT confirm")
            # ⭐dispatch 计数不再在此提前 +1(2026-07-22 实锤): 旧码
            # `if not action_suppressed: +=1` 信号语义错位 — suppressed 反映
            # **上一 tick** 的吞, 预知不了本次点击; 上一 tick 吞标志挂着 →
            # 派了1票计数=0 → 圈末误判"本圈零派出"退出剩6票。计数移到
            # _roster 的票数重读处(票实际减少=报告确认真落地, after-ack)。
            self.log(f"schedule report → confirm (count={self._dispatch_count}, YOLO 确认键)")
            self._ticket_read_pending = True  # re-read count back on the popout
            self._goto("roster")
            return action_click_box(report_confirm, "confirm schedule report (確認键)")

        # PRIORITY 2: 課程表資訊 info popup — click 課程表開始 (SCHED_START).
        start = self.find_cls(
            screen, UC.SCHED_START, conf=_CLS_CONF, region=_DIALOG_BAND
        )
        if start is None:  # SCHED_START sometimes mid-frame outside the band
            start = self.find_cls(screen, UC.SCHED_START, conf=_CLS_CONF)
        if start is not None:
            # ⛔⛔ 源头闸(2026-07-25 事故后新增): 票 == 0 时**绝不点課程表開始**。
            # 事故链证据(run_20260724_201229): 票已 0 → 仍点開始 → 游戏弹
            # 「購買課程表票券 單價💎30」→ 那个框的「確認」被 PRIORITY 1 当成
            # 報告框的確認点掉 = 花了 30 青辉石。**不点開始, 购买框根本不会出现**
            # —— 这是比"识别购买框"更靠前、更便宜的一道闸。
            # 读不出(-1/None)时不拦(fail-open)会重蹈覆辙, 但读不出就全停也会让
            # 正常日子干不了活 —— 折中: 读不出时交给 _buy_dialog + 硬上限兜底,
            # 这里只拦**确知为 0** 的情形(确知 0 还点 = 必然弹购买框)。
            if self._tickets == 0:
                self.log("⛔ 票券已 0 — 绝不点課程表開始(点了必弹青辉石购买框)")
                self._goto("exit")
                return action_wait(300, "tickets 0 → 不点開始, 退出")
            # ⛔2026-07-27 改: 旧版靠**在 reason 里塞"確認键"**去命中 _dedup_click
            # 的关键词豁免(为了让 課程表開始 渲染好就能点、不被稳定门吞)。代价是
            # 那条豁免 `return action` 把 **dedup hold 也一起跳过** → 本动作每次
            # 连发两发, 第二发落在**下一屏**上。实测第二发撞过「購買課程表票券」
            # (青辉石 30/张)的确認键旁 Δx=0.097 —— 只差一点就是真掏钱。
            # 现在改成显式 `_settle_exempt`: 稳定门照旧豁免(不被吞), same-target
            # hold 恢复生效(第二发被 hold 成 wait, 等指纹变化后 skill 重新看帧 →
            # 那时若是购买框, _buy_dialog 三路防线接管取消)。
            # reason 也改回**如实描述点的是什么** —— 措辞不该再兼任控制信号。
            self.log("schedule info → start (YOLO 课程表开始)")
            self._goto("open_room")
            act = action_click_box(start, "start schedule (課程表開始)")
            act["_settle_exempt"] = True
            return act

        # Generic / ticket-shortage popups — base helper is pure cls now
        # (确认+取消/叉 结构 → 默认点取消/叉掉, 绝不盲确认). NOTE: SCHED_ALL is the
        # WORK surface, never a popup — the helper keys off dialog buttons (确认
        # 键/取消键 cls), not SCHED_ALL, so it won't touch the popout (task #3
        # dead-loop guard).
        popup = self._handle_common_popups(screen)
        if popup:
            return popup

        if screen.is_loading():
            return action_wait(700, "schedule loading")

        # ── state machine ──
        if self.sub_state == "":
            self._goto("enter")

        handler = {
            "enter": self._enter,
            "navigate": self._navigate,
            "roster": self._roster,
            "open_room": self._open_room,
            "switch": self._switch,
            "exit": self._exit,
        }.get(self.sub_state)
        if handler is None:
            return action_wait(300, "schedule unknown state")
        return handler(screen)

    # ── enter ─────────────────────────────────────────────────────────────

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        self._enter_attempts += 1
        page = self.detect_screen_yolo(screen)

        if page == "Schedule" or self._is_schedule(screen):
            self.log("inside schedule")
            self._goto("navigate")
            return action_wait(400, "entered schedule")

        if page == "Lobby":
            nav = self.click_cls(screen, UC.NAV_SCHEDULE, "click schedule nav",
                                 conf=_CLS_CONF)
            if nav:
                return nav
            self.log("on lobby but no 课程表入口 cls — YOLO gap; waiting")
            return action_wait(400, "waiting for 课程表入口 cls")

        if page is not None:
            self.log(f"wrong screen '{page}', backing out")
            return action_back(f"back from {page}")

        if self._phase_ticks > _ENTER_MAX:
            self.log("enter budget exhausted, giving up on schedule")
            return action_done("could not reach schedule")
        if screen.is_loading() or len(screen.yolo_boxes or []) < 2:
            return action_wait(700, "no UI detected, likely loading")
        return action_wait(400, "entering schedule")

    # ── navigate (region-select / region-internal) ────────────────────────

    def _navigate(self, screen: ScreenState) -> Dict[str, Any]:
        """Drive from the region-select list to a region's popout.

        Sequence (once):  click 夏莱办公室 → ARROW_LEFT (jump to last region).
        Then per region:  click SCHED_ALL → open popout → roster.
        """
        if not self._is_schedule(screen):
            if self.detect_screen_yolo(screen) == "Lobby":
                self.log("on lobby, schedule exited — done")
                return action_done("schedule done (returned to lobby)")
            if self._phase_ticks > _NAVIGATE_MAX:
                self.log("lost schedule UI, backing out")
                return action_back("back (schedule UI lost)")
            return action_wait(500, "waiting for schedule UI")

        # If a popout is already open, go scan it.
        if self._roster_open(screen):
            self._reset_region()
            self._goto("roster")
            return action_wait(300, "popout already open → roster")

        if self._tickets == 0:
            self.log("no tickets remaining, exiting")
            self._goto("exit")
            return action_wait(300, "no tickets")

        # Step 1: first click 夏莱办公室 (region-select list row 0).
        if not self._office_clicked:
            # ⭐动态区域数(2026-07-21 逐帧审修): 区域列表帧数区域名 cls (36-40) →
            # 一圈判定用真实区数, 不再用 _MAX_REGIONS=14 兜底 — 5 区账号旧码要
            # 绕近 3 圈才触发 full-circle/exit, live 抓到 200+ tick 开关 popout
            # 空转。检出 <2 (列表帧没抓全) → 保守回落 _MAX_REGIONS。
            # ⚠新校区上线要加词表重训, 否则计数偏低 → fallback 提前(策略损失,
            # 非金钱风险)。
            _region_cls = (UC.SCHOOL_OFFICE, UC.SCHOOL_DORM, UC.SCHOOL_GEHENNA,
                           UC.SCHOOL_ABYDOS, UC.SCHOOL_MILLENNIUM)
            _n = sum(1 for c in _region_cls
                     if self.find_cls(screen, c, conf=_CLS_CONF) is not None)
            if _n >= 2:
                self._region_count = _n
                self.log(f"region list: {_n} regions detected (dynamic circle size)")
            office = self.find_cls(screen, UC.SCHOOL_OFFICE, conf=_CLS_CONF)
            if office is not None:
                self.log("entering 夏莱办公室 (YOLO 夏莱办公室)")
                self._office_clicked = True
                return action_click_box(office, "enter 夏莱办公室")
            # GAP: region tile cls under-trained → click its list-row position.
            self.log("夏莱办公室 cls not seen — clicking list row 0 (GAP)")
            self._office_clicked = True
            return action_click(*_OFFICE_ROW_POS, "enter 夏莱办公室 (list row, GAP)")

        # Step 2: inside a region now → ARROW_LEFT once = jump to last region.
        if not self._jumped_to_last:
            # Only jump once we're actually on a region-internal screen (the
            # SCHED_ALL bottom-right button is present there).
            if self._sched_all_btn(screen) is None and self._phase_ticks < 4:
                return action_wait(400, "waiting for region-internal screen")
            # ⭐after-ack(2026-07-25, 与 _switch 同族): 旧码点之前就置
            # _jumped_to_last, 稳定门一吞就永远跳不成。后果比 _switch 轻(遍历
            # 本身是环, 只是没从最新区起步), 但同一个 mutate-before-ack 形状,
            # 一并按"发出→下一 tick 看 action_suppressed 再落账"修。
            if self._jump_issued:
                if self.action_suppressed:
                    self.log("jump-to-last 被稳定门吞(未落屏) — 重发")
                    self._jump_issued = False
                else:
                    self._jumped_to_last = True
                    return action_wait(250, "jump-to-last landed")
            arrow = self._arrow_left(screen)
            self._jump_issued = True
            if arrow is not None:
                self.log("ARROW_LEFT → jump to last (newest) region (YOLO 左切换)")
                return action_click_box(arrow, "jump to last region")
            self.log("ARROW_LEFT cls not seen — symmetric left-edge coord (GAP)")
            return action_click(*_ARROW_LEFT_POS, "jump to last region (GAP)")

        # Step 3: on a region-internal screen → open its 全體課程表 popout.
        sched_all = self._sched_all_btn(screen)
        if sched_all is not None:
            self.log("opening 全體課程表 popout (YOLO 全体课程表)")
            return action_click_box(sched_all, "open 全體課程表 popout")

        # SCHED_ALL missing + region LIST visible = bounced to Location Select
        # (no SCHED_ALL there, ever) → route to _switch, whose row-walk
        # recovery enters the next region directly (v2, 2026-06-11).
        if self.find_cls(screen, UC.SCHOOL_OFFICE, conf=_CLS_CONF) is not None:
            self.log("Location Select detected (navigate) → switch row-walk")
            self._goto("switch")
            return action_wait(250, "location select → switch recovery")

        # SCHED_ALL button not seen — give it a few ticks (region transition),
        # then surface the gap and try the next region rather than stalling.
        if self._phase_ticks > _NAVIGATE_MAX:
            self.log("SCHED_ALL button cls missing — YOLO gap; switching region")
            self._goto("switch")
            return action_wait(300, "no SCHED_ALL, switching region")
        return action_wait(400, "waiting for 全体课程表 button cls")

    # ── roster (popout open) ──────────────────────────────────────────────

    def _roster(self, screen: ScreenState) -> Dict[str, Any]:
        """Popout open: read tickets, detect case A/B, pick + click a room."""
        if not self._roster_open(screen):
            # Popout closed unexpectedly (or game re-rendered). If we're on a
            # region-internal screen, re-open; if lost schedule, bail.
            if not self._is_schedule(screen):
                if self.detect_screen_yolo(screen) == "Lobby":
                    return action_done("schedule done (lobby)")
                if self._phase_ticks > _ROSTER_MAX:
                    return action_back("back (lost schedule in roster)")
                return action_wait(450, "waiting for popout")
            if self._sched_all_btn(screen) is not None:
                self.log("popout closed, re-opening 全體課程表")
                return action_click_box(self._sched_all_btn(screen),
                                        "re-open 全體課程表 popout")
            if self._phase_ticks > _ROSTER_MAX:
                self._goto("switch")
                return action_wait(300, "popout gone, switching region")
            return action_wait(400, "waiting for popout to render")

        # First ticks on a fresh popout: count this region + read tickets.
        # Lock judgment is MULTI-FRAME (live 2026-06-09: ROOM_LOCKED flickered
        # out on the single judgment frame → a LOCKED region was judged case B
        # and only target rooms got dispatched). One sighting ⇒ locked; only
        # 3 consecutive lock-free frames ⇒ case B.
        if self._region_locked is None:
            locked_now = self.find_cls(screen, UC.ROOM_LOCKED, conf=_CLS_CONF) is not None
            self._lock_seen = getattr(self, "_lock_seen", False) or locked_now
            self._lock_scan_frames = getattr(self, "_lock_scan_frames", 0) + 1
            if not self._lock_seen and self._lock_scan_frames < 3:
                return action_wait(300, f"lock scan ({self._lock_scan_frames}/3)")
            self._regions_seen += 1
            self._tickets = None  # 强制本 popout 重读一次(None = 未知)
            self._read_tickets(screen)
            self._region_locked = self._lock_seen
            self._lock_seen = False
            self._lock_scan_frames = 0
            self.log(f"region #{self._regions_seen}: "
                     f"{'LOCKED (case A — level it)' if self._region_locked else 'no locks (case B — targets only)'}"
                     f", tickets={self._tickets}")
            return action_wait(300, "scanned popout")

        # Re-read ticket after a dispatch (confirms it actually decremented).
        if self._ticket_read_pending:
            self._ticket_read_pending = False
            _prev = self._tickets
            self._read_tickets(screen)
            # ⭐dispatch 落账(after-ack, 2026-07-22): 票数实际减少=报告确认
            # 真落地, 按扣减量计数。旧码在点确认前看 action_suppressed 预判
            # — 信号反映的是上一 tick 的吞, 语义错位: 上一 tick 吞标志挂着
            # → 真派了 1 票计数=0 → 圈末误判"本圈零派出"退出剩 6 票(实锤)。
            # 读不出(_prev/-1/None)不计 — 计数偏低最多多绕一圈, cap 防线仍在。
            if (_prev is not None and _prev > 0
                    and self._tickets is not None
                    and 0 <= self._tickets < _prev):
                _spent = _prev - self._tickets
                self._dispatch_count += _spent
                # ⛔同步写按游戏日的持久台账 —— per-run 计数跨 session 归零,
                # 那正是 2026-07-25 硬上限没兜住 30 青辉石那次的原因之一。
                self._day_dispatched += _spent
                _save_sched_state({"game_day": _game_day(),
                                   "dispatched": self._day_dispatched})
                self.log(f"dispatch 落账: 票 {_prev}→{self._tickets}, "
                         f"count={self._dispatch_count} "
                         f"(今日累计 {self._day_dispatched})")
            if self._tickets == 0:
                self.log("tickets exhausted → exit")
                self._goto("exit")
                return action_wait(300, "tickets exhausted")

        # Reaction time #1: let the popout heads finish rendering before the
        # first pick — fused_avatar is flickery on frame 1 of a fresh popout.
        if self._head_settle > 0:
            self._head_settle -= 1
            return action_wait(400, f"letting popout heads render ({self._head_settle})")

        heads = self._roster_heads(screen)
        rooms = self._cluster_rooms(heads)
        room, reason = self._pick_room(screen, rooms)

        if room is not None:
            self._barren_scans = 0
            cx, cy = self._room_center(room)
            names = [h.cls_name for h in room]
            self.log(f"dispatch room @({cx:.3f},{cy:.3f}) {names} — {reason}")
            self._clicked_heads.append((cx, cy))
            # Click the first (left-most) head — opens the whole room's info
            # popup (not a single-student select). The dispatch popups in
            # tick() take over from here (info → start → report → confirm).
            head = sorted(room, key=lambda b: b.cx)[0]
            self._goto("open_room")
            return action_click_box(head, "click room head → 課程表資訊")

        # Reaction time #2: no room THIS scan. Re-scan a few frames before
        # leaving — gives fused_avatar time to surface heads/targets that
        # flickered out this frame (user: 给模型识别反应时间). Only switch region
        # after consistent empties.
        self._barren_scans += 1
        if self._barren_scans < _BARREN_SCAN_MAX:
            return action_wait(500, f"re-scan roster {self._barren_scans}/{_BARREN_SCAN_MAX} ({reason})")

        self.log(f"no room after {self._barren_scans} scans ({reason}) → close popout, next region")
        close = self._popout_close(screen)
        if close is not None:
            self._goto("switch")
            return action_click_box(close, "close popout (YOLO 弹窗叉叉)")
        if self._phase_ticks > _ROSTER_MAX:
            self.log("popout close-X cls missing — YOLO gap; ESC to switch")
            self._goto("switch")
            return action_back("close popout (no X cls)")
        return action_wait(400, "waiting for popout close-X cls")

    # ── open_room (info → start → report → confirm handled in tick) ────────

    def _open_room(self, screen: ScreenState) -> Dict[str, Any]:
        """After clicking a head, wait for the dispatch popups. The info/report
        popups are handled by the PRIORITY guards in tick(); here we only handle
        the gaps: popup didn't appear, drifted off-screen, or we're back on the
        popout already."""
        # Back on the popout? The 課程表資訊 info popup takes ~1-2s to open after
        # the head click — give it time (tick() PRIORITY 2 grabs SCHED_START the
        # moment it renders). Only conclude "click failed / back to roster" if
        # the popout is STILL the top surface after the settle. Without this, the
        # bot deduped 6 rooms but only 1 actually opened its info popup (live
        # 2026-06-02: clicked 6 heads, 1 start). (user: 给点击反应时间)
        if self._roster_open(screen):
            # ⭐墙钟而非 tick 计数(2026-07-25 live 实锤): 上面注释写的是"弹窗要
            # ~1-2s", 但实现在数 tick —— 而非 loading 的 action_wait 被 server
            # 压到 0.12s, 4 tick 可能只有 0.5-1s, 窗口在弹窗渲染完之前就过期,
            # skill 判"没开"退回 roster 另挑一间 → 正是本常量当初要修的
            # "clicked 6 heads, 1 start"(2026-06-02)重现。tick 速率不可依赖
            # (7月录像全是 step 门控, 0.65~8.7 s/tick), 计时就得用墙钟。
            if not self._open_room_t0:
                self._open_room_t0 = self.clock()
            _el = self.clock() - self._open_room_t0
            if _el < _OPEN_ROOM_SETTLE_SEC:
                return action_wait(450, f"waiting for 課程表資訊 popup "
                                        f"({_el:.1f}s/{_OPEN_ROOM_SETTLE_SEC}s)")
            self.log(f"info popup never opened after head click ({_el:.1f}s) → back to roster")
            self._goto("roster")
            return action_wait(300, "back to roster")

        if not self._is_schedule(screen):
            if self.detect_screen_yolo(screen) == "Lobby":
                self.log("drifted to lobby during dispatch, re-entering")
                self._office_clicked = True   # don't restart the whole nav
                self._jumped_to_last = True
                self._goto("enter")
                return action_wait(300, "re-enter from lobby")
            return action_wait(450, "waiting for dispatch popup")

        # On a schedule surface but neither popup nor popout detected. The
        # tick() guards catch SCHED_START / BTN_CONFIRM the moment they render;
        # if neither appears within budget, treat the room as done and go back
        # to the roster (the popout re-opens after a report).
        if self._phase_ticks > _OPEN_ROOM_MAX:
            self.log("dispatch popup never appeared — back to roster")
            self._goto("roster")
            return action_wait(300, "dispatch timeout → roster")
        return action_wait(400, "waiting for SCHED_START / report cls")

    # ── switch (ARROW_LEFT to next region) ────────────────────────────────

    def _switch(self, screen: ScreenState) -> Dict[str, Any]:
        """Close any lingering popout, then ARROW_LEFT to the next region."""
        if self._tickets == 0:
            self._goto("exit")
            return action_wait(300, "tickets exhausted → exit")

        # Full circle: traversed every region. 2026-07-21 逐帧审修: 一整圈**没派出
        # 任何学生**(_dispatch_count 没增 = 无可派对象)就直接退出, 不再因 tickets>0
        # 硬撑 fallback 空转(旧码: 学生排完但剩票时绕 ~28 区到 max_ticks=320 才退,
        # 中断的 autonomous 重跑复现 tick 201 空转)。只有本圈真派出过学生+还剩票, 才
        # 值得再扫一圈捡漏。
        # 动态区数(列表帧实测)。没探测到(sub_only/中途恢复没经过区域列表帧)
        # 时用保守下限 5(BA 满解锁 5 区; 少区账号多绕几次无害)——绝不回落
        # _MAX_REGIONS=14: 2026-07-22 实锤恢复路径 fallback 永远够不到, 零派出
        # 绕到 max_ticks 死, 6 票滞留。
        # ⛔2026-07-25 实测: 区域数**远大于 5**。Location Select 列表可滚动, 可见
        # 5 行只是第一页(帐号 `地區等級總和 RANK 142`), ADB 逐个切实测至少 9 个:
        # 夏萊辦公室/夏萊居住區/格黑娜學園中央區/阿拜多斯高中/千年研究區域/
        # 狂獵綜合藝術區/春葉原/山海經中央特區/D.U.白鳥區。而上面数 SCHOOL_* cls
        # 只认得头 5 个 → 走满 5 个就宣布 full circle, 后 4 个区里的目标学生
        # **从来没被排过课**(策略损失, 非金钱风险)。
        # 正解: 别数 cls, 用**回到起点**判一圈 —— _region_sig 已经能把区认出来
        # (标定见 [[region-switch-truth]])。_circle_closed 在 _switch 的到达确认
        # 处置位: 那里 SCHED_ALL 正锚在场(popout 已关、RANK 标题没被遮), 是唯一
        # 测得准的时刻。cls 计数退居"指纹不可用时的兜底", _MAX_REGIONS 是硬上限。
        _circle_size = self._region_count or 5
        if self._circle_closed:
            self.log(f"一圈完成: 标题指纹已转回起点(走过 {self._regions_seen} 个区)")
        elif self._regions_seen >= _MAX_REGIONS:
            self.log(f"⚠走满硬上限 {_MAX_REGIONS} 个区仍没转回起点 — 当一圈处理")
            self._circle_closed = True
        elif self._regions_seen >= _circle_size and self._circle_first_sig is None:
            # 指纹通道整趟没工作过(帧拿不到/cv2 缺席) → 退回旧的 cls 计数判据,
            # 保持老行为, 不至于一路绕到硬上限。
            self.log(f"指纹通道不可用 → 回退 cls 计数判一圈 ({_circle_size} 区)")
            self._circle_closed = True
        if self._circle_closed:
            # ⛔防空转真闸(2026-07-25, 取代原来那个数圈数的 `not _full_circle`):
            # 一圈 N 个区必须伴随 N-1 次**帧证据确认的换区**。对不上 = 我们其实
            # 在原地反复开关同一个区的 popout(2026-07-25 那 200 tick 的形状) →
            # 立刻大声收工, 绝不靠"日志说我切过"续命。
            # 需要的换区次数按**本圈实际扫过的 popout 数**算(_regions_seen-1),
            # 不再按 cls 猜的 _circle_size —— 一圈现在可能是 9 个区而不是 5 个。
            _switches_this_circle = (self._verified_switches
                                     - self._circle_start_switches)
            _need_switches = max(0, self._regions_seen - 1)
            if _switches_this_circle < _need_switches:
                self.log(f"⚠假圈: 扫了 {self._regions_seen} 个 popout 但只有 "
                         f"{_switches_this_circle} 次帧证实换区(需 "
                         f"{_need_switches}) — 原地空转, 收工"
                         f"(剩 {self._tickets} 票)")
                self._goto("exit")
                return action_wait(300, "假圈(换区未落地) → exit")
            _dispatched_this_circle = self._dispatch_count > self._circle_start_dispatch
            # ⛔2026-07-25 实锤: 旧闸带 `not self._full_circle`, 捡漏圈**只准跑
            # 一次** —— 今天第1圈派1人→开捡漏→第2圈又派1人→但 _full_circle 已
            # True → 直接退出, **5 张票原地作废**。而同一设施能派多人(教室一格
            # 上 2 个绿勾实证), 7 设施十几个位置, 7 张票本该花得完。
            # 正确规则: **只要这圈还派出过人且还剩票就继续下一圈**; 收敛性由
            # "某圈零派出即退" 保证(每圈必须至少派 1 人才有资格续圈), 外加
            # tickets==0 与 max_ticks 两道硬闸, 不会空转。
            # ⛔2026-07-25 第二轮实锤(第一版没修够): 兜底 fallback spend-leftover
            # 要求 _full_circle=True, 而 _full_circle 又要求"某圈派出过人" ——
            # 学生已在上一轮派完时**第1圈必然零派出 → 兜底永远够不着**, 亮着的
            # 空房间(实测 7 间里 4 间无目标学生但可派)+剩票全废。
            # 正解: **第1圈结束无条件开兜底模式**; 之后只要这圈还在派就续圈。
            # 收敛: 兜底圈零派出即退 → 最坏只多绕 1 圈, 外加 tickets==0 与
            # max_ticks 两道硬闸。
            if ((self._tickets is None or self._tickets > 0)
                    and (not self._full_circle or _dispatched_this_circle)):
                self.log(f"full circle: 剩 {self._tickets} 票 → "
                         f"{'开' if not self._full_circle else '续'}捡漏圈")
                self._full_circle = True
                self._regions_seen = 0
                self._circle_start_dispatch = self._dispatch_count
                self._circle_start_switches = self._verified_switches
                # 新一圈重新锚起点(否则第 2 圈开局就"已经在起点"= 立刻假完成)
                self._circle_closed = False
                self._circle_first_sig = None
            else:
                _why = ("本圈零派出(无可派对象)" if not _dispatched_this_circle
                        else f"full circle done ({self._regions_seen} regions)")
                self.log(f"{_why} → exit")
                self._goto("exit")
                return action_wait(300, "schedule circle done → exit")

        # Close a lingering popout first. 双发的 X/ESC 会落到后面的区域屏上,
        # 弹回 Location Select —— 正是那个把票扣在半路的 bounce。
        # ⛔2026-07-25 实锤: 旧的 `_phase_ticks % 2` 节流**拦不住** —— 帧滞后使
        # `_roster_open` 在点击后仍为 True, 而 `_dedup_click` 看到结构指纹变了
        # 就放行重复点击, 于是 `close popout before switch` 连发两次, 第二发
        # 落在区域屏上误开了一个设施。tick 奇偶不是时间, 也不是证据。
        # 改 after-ack: 发过一次就等**帧证据**(popout 真关掉)或墙钟到期才重发;
        # 被稳定门吞了(action_suppressed)则立刻重发 —— 那次根本没落屏。
        if self._roster_open(screen):
            if self._popout_close_issued and not self.action_suppressed:
                if self.since("popout_close") < _POPOUT_CLOSE_SEC:
                    return action_wait(300, "popout closing — 等它真关掉(after-ack)")
                self.log(f"⚠popout {_POPOUT_CLOSE_SEC:.1f}s 还没关掉 — 重发关闭")
            elif self._popout_close_issued:
                self.log("关 popout 被稳定门吞(未落屏) — 立刻重发")
            elif ("popout_close" in getattr(self, "_timers", {})
                    and self.since("popout_close") < _POPOUT_CLOSE_SEC
                    and not self.action_suppressed):
                # ⚠单帧 roster-negative 闪断(popout 淡出中)会把 _popout_close_issued
                # 清掉 — 但墙钟 mark 不受闪断影响: 冷却期内绝不二次发射(2026-07-25
                # 审计: 原来闪断一次即重置 refractory, 双发洞没堵死)。
                return action_wait(300, "popout close 冷却中(闪断防双发)")
            self._popout_close_issued = True
            self.mark("popout_close")
            close = self._popout_close(screen)
            if close is not None:
                return action_click_box(close, "close popout before switch")
            return action_back("close popout before switch (no X cls)")
        self._popout_close_issued = False

        # Location Select recovery v2 (live 2026-06-11): closing a popout can
        # bounce us to the region LIST (no ARROW_LEFT/SCHED_ALL there). v1
        # re-entered row 0 every time → infinite loop on a fully-dispatched
        # first region. v2 turns the bounce INTO the traversal: each bounce
        # enters the NEXT list row directly (rows sit at fixed y), capped.
        # ⚠必须排在到达态验证**之前**: LS 屏没有 RANK 标题, 指纹一定跟区域内屏
        # 不同 → 会被误判成"切换成功"。
        if self.find_cls(screen, UC.SCHOOL_OFFICE, conf=_CLS_CONF) is not None:
            self._switch_from_sig = None
            self._ls_recoveries += 1
            if self._ls_recoveries > 8:
                self.log("Location Select bounce cap (8) → exit with leftovers")
                self._goto("exit")
                return action_wait(300, "ls bounce cap → exit")
            rows_y = (0.262, 0.407, 0.552, 0.697, 0.842)
            row = self._ls_recoveries % len(rows_y)
            self.log(f"Location Select → entering list row {row} directly "
                     f"(bounce #{self._ls_recoveries})")
            self._office_clicked = True   # row click IS the region entry
            self._jumped_to_last = True   # no arrow jump; scan this region as-is
            self._reset_region()
            self._goto("navigate")
            return action_click(0.70, rows_y[row], f"enter region list row {row}")

        # ⭐到达态验证(2026-07-25 帧证据实锤 — 见 memory log_is_not_truth)
        # 旧码在 click **之前**就 _reset_region()+_goto("navigate") = mutate-
        # before-ack。而 _switch 必然先关 popout, **紧接的那一 tick 结构指纹必然
        # 剧变**(-16 个 popout/头像 cls, +4 个区域屏 cls) → _dedup_click 的
        # frame-settle 稳定门 100% 把 ARROW_LEFT 转成 wait。点击从来没发出去,
        # skill 却已经"前进"到 navigate → 重开同一个区域的 popout → 200 tick
        # 空转, 日志刷 5 条 ARROW_LEFT 一次也没生效, _regions_seen 假计数凑出
        # 假 full-circle。(实证: run_20260724_185527 标题哈希 228 tick 只在
        # 「夏萊辦公室」与「popout 遮挡」两簇间摆动。)
        # 现在: 点之前记标题指纹, **点完留在 switch**, 标题真变了才算切成功。
        # 正锚: SCHED_ALL(全體課程表) 只在 popout 关掉的区域内屏渲染(帧实测
        # t0078 popout 开时无此 cls, t0079 关掉才出现) —— 有它才说明标题带没被
        # 弹窗盖住, 指纹才可信。
        if self._sched_all_btn(screen) is None:
            if self._phase_ticks > _SWITCH_MAX:
                # 锚一直不出现 = 既不在区域内屏也不在 LS(转场卡住/UI 变了)。
                # 不许在这儿盲点箭头, 退回 navigate 让它重新认屏。
                self.log("SCHED_ALL 锚迟迟不出现 — 回 navigate 重认屏")
                self._goto("navigate")
                return action_wait(300, "no SCHED_ALL anchor → navigate")
            return action_wait(250, "等区域内屏渲染(SCHED_ALL 锚)")
        _sig = self._region_sig(screen)
        if self._switch_from_sig is None:
            # 基线要连续两帧一致才锁 —— popout 淡出中途的混合帧会让下一帧"看起来
            # 变了", 那是假到达。
            if self._sig_pending is not None and self._sig_same(_sig, self._sig_pending):
                self._switch_from_sig = _sig
                self._switch_t0 = self.clock()
                self._switch_taps = 0
                self._sig_pending = None
                # ⭐一圈的"起点"就在这里记 —— 这是**唯一**能测准的时刻:
                # SCHED_ALL 正锚在场(popout 已关, RANK 标题没被盖住) + 指纹已
                # 连续两帧一致。popout 开着时标题被遮, 在别处取指纹必然错。
                if self._circle_first_sig is None:
                    self._circle_first_sig = _sig
                    self.log("记下本圈起点区域指纹")
            else:
                self._sig_pending = _sig
                return action_wait(200, "锁定区域指纹(需连续两帧一致)")
        elif not self._sig_same(_sig, self._switch_from_sig):
            # 到达也要连续两帧确认: popout 淡出的转场残影帧(SCHED_ALL 已渲染但
            # 标题带还糊着)实测 MAD 高达 77, 单帧判据会假宣布"切成功"——
            # 那就退化成另一种"拿自己的动作当证据"。
            if (self._arrive_pending is None
                    or not self._sig_same(_sig, self._arrive_pending)):
                self._arrive_pending = _sig
                return action_wait(200, "疑似换区 — 等第二帧确认")
            self._arrive_pending = None
            self._verified_switches += 1
            self.log(f"✔区域切换到达确认(标题变了, {self._switch_taps} 次 tap, "
                     f"累计 {self._verified_switches} 次真切换)")
            # ⭐转回起点 = 一圈走完。这比"数了几个 SCHOOL_* cls"可靠得多:
            # cls 只认得头 5 个区, 实测账号至少 9 个(Location Select 可滚动),
            # 按 cls 计数会在第 5 个区就宣布 full circle, 后面几个区里的目标
            # 学生**一次都没被排过课**。见 [[region-switch-truth]]。
            if (self._circle_first_sig is not None
                    and self._sig_same(_sig, self._circle_first_sig)):
                self._circle_closed = True
                self.log(f"↩标题指纹转回起点 = 一圈走完(共 {self._regions_seen} 个区)")
            self._switch_from_sig = None
            self._reset_region()
            self._goto("navigate")
            return action_wait(250, "region switched (verified) → navigate")
        else:
            self._arrive_pending = None

        _waited = self.clock() - (self._switch_t0 or self.clock())
        if _waited > _SWITCH_VERIFY_SEC:
            # 绝不假装切过: 读不出/切不动就大声收工, 把剩票如实报出来。
            self.log(f"⚠区域切换失效: {self._switch_taps} 次 ARROW_LEFT / "
                     f"{_waited:.1f}s 标题始终未变 — 收工(剩 {self._tickets} 票)")
            self._goto("exit")
            return action_wait(300, "region switch 验证失败 → exit")

        if self.action_suppressed:
            # 稳定门吞掉了上一发 —— 留痕, 下一稳定帧自然重发(不计 tap)。
            self.log("ARROW_LEFT 被稳定门吞(未落屏) — 等稳定帧重发")
            return action_wait(200, "ARROW_LEFT suppressed → 重发")

        # ARROW_LEFT to the next region.
        arrow = self._arrow_left(screen)
        if arrow is not None and _waited <= _SWITCH_GAP_SEC:
            self._switch_taps += 1
            self.log(f"ARROW_LEFT → next region (YOLO 左切换, tap "
                     f"#{self._switch_taps}, 等到达)")
            return action_click_box(arrow, "ARROW_LEFT next region")
        if arrow is not None or _waited > _SWITCH_GAP_SEC:
            # cls 在但迟迟不生效 → 换坐标 GAP 落点(实测 ADB tap 0.023/0.502 有效)
            self._switch_taps += 1
            self.log(f"ARROW_LEFT 坐标 GAP 落点 (tap #{self._switch_taps}, "
                     f"{_waited:.1f}s 未到达)")
            return action_click(*_ARROW_LEFT_POS, "ARROW_LEFT next region (GAP)")

        return action_wait(400, "waiting for ARROW_LEFT cls")

    # ── exit ──────────────────────────────────────────────────────────────

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if self.detect_screen_yolo(screen) == "Lobby":
            self.log("back in lobby, schedule done")
            return action_done("schedule complete")
        if self._phase_ticks > _EXIT_MAX:
            self.log("exit budget exhausted, reporting done")
            return action_done("schedule exit timeout")
        # Close a lingering popout before leaving.
        close = self._popout_close(screen)
        if close is not None and self._roster_open(screen):
            return action_click_box(close, "close popout on exit")
        home = self.find_cls(screen, UC.BTN_HOME, conf=_CLS_CONF)
        if home is not None:
            return action_click_box(home, "schedule exit: home (YOLO 回大厅按钮)")
        back = self.find_cls(screen, UC.BTN_BACK, conf=_CLS_CONF)
        if back is not None:
            return action_click_box(back, "schedule exit: back (YOLO 返回键)")
        return self.nav_home(screen, "schedule exit")


# Region tiles that HAVE a YOLO ui cls (region-select list). Ordered the way
# the list lays them out. Used only for is-schedule detection + the 夏莱办公室
# anchor; traversal is ARROW_LEFT-driven, NOT tile-clicking (tiles are
# under-trained — see _OFFICE_ROW_POS GAP note).
_SCHOOL_TILES = [
    UC.SCHOOL_OFFICE,      # 夏莱办公室   (list row 0 — full-circle anchor)
    UC.SCHOOL_DORM,        # 夏莱居住区
    UC.SCHOOL_GEHENNA,     # 格黑娜学院中央区
    UC.SCHOOL_ABYDOS,      # 阿拜多斯高中
    UC.SCHOOL_MILLENNIUM,  # 千年研究所
]
