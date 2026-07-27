"""TicketSweepSkill — shared base for 悬赏通缉(bounty) + 学院交流会(JFD).

Verified flow (interactive probe 2026-06-01, data/_missions_probe_log.md).
bounty & JFD are isomorphic ("票券扫荡型"); this base captures the common flow
and subclasses fill in the ticket cls / hub tile / branch picker / AP cost.

★★ THREE-LAYER pyroxene protection (the buy-ticket money bug) ★★
  ① digit-OCR the ticket count at entry → 0 (or unreadable-and-confirm-greyed)
     ⇒ NEVER sortie/sweep (a 0-ticket sweep pops a 購買票券 青辉石 dialog).
  ② never click 立即完成 / buy buttons.
  ③ at the sweep-confirm dialog: if a 青辉石 icon sits in the dialog BODY
     (a buy dialog) OR the confirm is greyed (灰色确认 = insufficient) ⇒ CANCEL.
Tickets are SHARED across branches (probe: 1/6 total), so one MAX sweep on a
single branch drains them all — we pick ONE configured branch, no iteration.

State machine
-------------
enter        lobby → NAV_TASKS → hub → _HUB_TILE → on-page (ticket cls).
ticket_check digit-OCR ticket X/Y. 0 → exit. >0 → branch.
branch       subclass _click_branch() navigates to the configured branch's
             stage list (bounty: cls tiles; JFD: position — no cls, v6 gap).
stage        find 入场键 in the right panel, swipe to the bottom (positions
             stabilize), click the lowest (= highest difficulty) 入场键.
sortie       任務資訊 popup. ⛔ pyroxene-buy guard. If _COSTS_AP, gate on AP.
             MAX_可点击 → MAX (when affordable); else single sweep. → 扫荡开始.
confirm      sweep-confirm dialog. ⛔ pyroxene/grey guard → 确认键.
result       掃蕩完成 popup (WGC transition → poll/re-detect) → 确认键 dismiss.
             Re-read tickets: 0 → exit; >0 → sortie again.
exit         返回键 / 回大厅 → lobby (or hub) → done.

Detectors: base "ui" + "battle" (set by SKILL_YOLO_MAP).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from brain.skills.base import (
    BaseSkill, ScreenState, YoloBox,
    action_click, action_click_box, action_wait, action_back, action_done,
    action_swipe,
)
from brain.skills import ui_classes as UC

_APP_CONFIG_FILE = Path(__file__).resolve().parents[2] / "data" / "app_config.json"
_CLS_CONF = 0.30
# Sweep-confirm 确认键/取消键 band (probe: y≈0.70).
_CONFIRM_BAND = (0.28, 0.60, 0.72, 0.82)
# 掃蕩完成 reward popup 确认键 sits LOWER (probe: ~0.5, 0.81).
_DONE_CONFIRM_BAND = (0.30, 0.74, 0.70, 0.90)
# 扫完重读票数的墙钟窗(旧值是 `_phase_ticks > 8` = 真实仅 1.2-2.0s, 一次
# OCR 抖动就够把整轮票判死; 单位口径见 BaseSkill.mark)
_POST_SWEEP_READ_SEC = 6.0
# A 青辉石 icon inside THIS band = a buy dialog (NOT the top-bar balance at cy<0.10).
# Deep-dive C5 (2026-06-09): aligned to schedule's LIVE-VERIFIED region (icon
# at cy≈0.577 > old 0.48 bound — same miss risk as arena C4).
_PYROXENE_BODY_REGION = (0.20, 0.12, 0.82, 0.64)
# Stage list lives in the right panel.
_STAGE_PANEL = (0.58, 0.12, 1.0, 0.98)
# 任務資訊 popup MAX button fixed pos (right of 加号; proven on special_sweep
# 2026-06-15). Fallback when cls111 MAX_可点击 is missed → 防只扫1票.
_POS_TICKET_MAX = (0.84, 0.42)
# Re-click the 入場键 this many times if 任務資訊 never opens (a dropped tap —
# root-fixed by AdbInput._IO_LOCK, but kept as self-healing so a single lost
# enter never costs the whole sweep). Live 2026-06-15: swept 0, manual same-pos
# tap opened it → tap was lost, not mis-aimed.
_SORTIE_MAX_RETRIES = 2


def _load_profile_list(key: str) -> List[str]:
    """Read an ordered string list from the active app_config profile."""
    try:
        if not _APP_CONFIG_FILE.exists():
            return []
        data = json.loads(_APP_CONFIG_FILE.read_text("utf-8"))
        active = data.get("active_profile", "default")
        profile = (data.get("profiles") or {}).get(active, {})
        raw = profile.get(key)
        if not isinstance(raw, list):
            return []
        out: List[str] = []
        for item in raw:
            s = str(item or "").strip()
            if s and s not in out:
                out.append(s)
        return out
    except Exception:
        return []


class TicketSweepSkill(BaseSkill):
    # ── subclass config (override) ──
    _TICKET_CLS: str = UC.TICKET_BOUNTY        # ticket icon (digit-OCR anchor)
    _HUB_TILE: str = UC.HUB_BOUNTY             # hub entry tile cls
    _PAGE_NAME: str = "Bounty"                 # detect_screen_yolo page (or "")
    _CONFIG_KEY: str = "bounty_branches"       # app_config profile key
    _COSTS_AP: bool = False                    # JFD sweeps cost AP
    _AP_PER_SWEEP: int = 15                    # estimated AP per sweep (JFD)
    _MAX_RENDER_WAIT: int = 5                  # frames to wait for solid MAX before giving up
    _MAX_SWEEP_CYCLES: int = 8                 # safety cap on sweep cycles

    def __init__(self, name: str):
        super().__init__(name)
        self.max_ticks = 120
        self._branches: List[str] = []
        self._init_state()

    def _init_state(self) -> None:
        self._phase_ticks: int = 0
        self._enter_ticks: int = 0
        self._swipe_count: int = 0
        self._last_stage_y: float = -1.0
        self._sweep_cycles: int = 0
        self._tickets: Optional[int] = None
        self._maxed: bool = False
        self._safe_to_max: bool = False
        self._max_wait: int = 0
        self._max_fires: int = 0          # MAX 双发 latch(稳定门吞首发防线)
        self._branch_clicks: int = 0
        self._branch_settle: int = 0
        self._sortie_retries: int = 0
        self._zero_sweep_retried: bool = False   # 0-sweep 对账回炉只做一次
        self._post_sweep_unread: bool = False    # 收尾时票数读不出(竣工判据用)

    def reset(self) -> None:
        super().reset()
        self._init_state()
        self._branches = _load_profile_list(self._CONFIG_KEY)
        self.log(f"{self.name} branches (config {self._CONFIG_KEY}): {self._branches or 'default'}")

    def _goto(self, sub_state: str) -> None:
        self.sub_state = sub_state
        self._phase_ticks = 0
        self.mark("phase")                 # _phase_ticks 的墙钟版
        self.clear_timer("post_sweep")     # 重读票数窗随阶段重置

    # ── 竣工判据 ─────────────────────────────────────────────────────────
    def exit_report(self):
        """票券型 skill 的竣工判据 = 票扫光了没。

        ⛔[[completion-gap]] 里"悬赏票 剩多少=未知"那一格就是这里缺判据造成的:
        旧码票数读不出时直接 log "MAX likely drained" 走人, 对外看起来干干净净,
        实际没人知道剩几张。UNKNOWN 必须与 CLEAN 严格分开。"""
        if self._sweep_cycles == 0:
            if (self._tickets or 0) > 0:
                return ("LEFTOVER",
                        f"入场读到 {self._tickets} 张票但一次都没扫出去")
            if self._tickets == 0:
                return ("CLEAN", "入场票数就是 0, 无事可做")
            return ("UNKNOWN", "票数从未读出, 且 0 次扫荡")
        if getattr(self, "_post_sweep_unread", False):
            return ("UNKNOWN",
                    f"扫了 {self._sweep_cycles} 轮, 但收尾时票数读不出 —— "
                    f"**不知道**是否扫光")
        if self._tickets is None:
            return ("UNKNOWN", f"扫了 {self._sweep_cycles} 轮, 票数未知")
        if self._tickets > 0:
            return ("LEFTOVER",
                    f"扫了 {self._sweep_cycles} 轮, 仍剩 {self._tickets} 张票")
        return ("CLEAN", f"{self._sweep_cycles} 轮扫光, 票 0")

    # ── subclass hooks ───────────────────────────────────────────────────
    def _click_branch(self, screen: ScreenState) -> Optional[Dict[str, Any]]:
        """Navigate to the configured branch's stage list. Return an action to
        click the branch, or None to wait. Default: no branch select needed."""
        return None

    # ── shared helpers ────────────────────────────────────────────────────
    def _on_page(self, screen: ScreenState) -> bool:
        if self.find_cls(screen, self._TICKET_CLS, conf=_CLS_CONF) is not None:
            return True
        if self._PAGE_NAME and self.detect_screen_yolo(screen) == self._PAGE_NAME:
            return True
        return False

    # Counter strips, anchored on the YOLO ticket badge and measured in ICON
    # WIDTHS so they track the badge instead of hard-coded screen fractions.
    # ⛔2026-07-27 全语料标定(579 帧, `scratchpad/tk_*.py`): the counter has TWO
    # layouts and the old single strip only ever covered one of them —
    #   cy~0.14 分支页  「持有票券   X/Y」 4 字标签 → 数字在 4.19–5.39 iw
    #   cy~0.28 关卡列表「懸賞通緝票券 X/Y」6 字标签 → 数字在 6.12–7.31 iw
    # 多两个汉字就把数字推出了旧 strip 的右界(icon.x2+0.112) ⇒ **关卡列表页的
    # 票数 0/211 帧读得出, 0.0%** —— 不是"这一次读不出", 是从来没读出来过。
    _TICKET_WINDOWS = ((4.0, 6.4), (6.0, 8.4), (4.0, 8.0))
    # ⛔y 留白是真正的开关: 旧值 0.4*bh 让 DB 检测器在关卡列表页**整条返回空**
    # (同一张帧把上下留白放到 1.2*bh 立刻检出 '6/6' score 0.75)。加宽 x、放大、
    # 拉对比度全部无效 —— 试过, 全线 None, 拉对比度还把本来能读的那条弄坏了。
    _TICKET_YPAD = 1.2

    def _read_tickets(self, screen: ScreenState) -> Optional[int]:
        """digit-OCR the 持有票券 X/Y next to the ticket icon. ★ money defense #1
        (0 tickets ⇒ never sortie → never the buy-pyroxene trap), so the read
        must be robust — and "robust" cuts both ways: an over-read is worse than
        no read, because `tickets == 0 → exit` is the SOURCE gate that keeps the
        buy-ticket dialog from ever appearing.

        ⛔2026-07-27 全语料实测(4K 帧 351 张 = live 口径, 低分辨率帧 live 不会
        走到 —— run_digit_ocr 对 <3200 宽的帧自动换 ADB 4K 干净帧重抓):
        | 方案  | cy0.14 分支页 | cy0.28 关卡列表 | 零票屏读成 >0 |
        | 旧    | 139/140 99.3% | **0/211 0.0%**  | **18/114 = 15.8%** |
        | 现    | 139/140 99.3% | **211/211 100%**| **0/114** |
        那 18 次是旧 strip 把屏幕上明明白白的「持有票券 0/6」读成 `'9/0'`(8 张
        逐张目检过真值全是 0/6) ⇒ `_ticket_check` 拿到 9 → 出击 → 0 票出击弹出的
        正是青辉石買票框。**fail-closed 只挡 None, 挡不住读大** —— 这是旧代码里
        真实存在的掉钱路径, 不是本次改动引入的。
        """
        if screen.frame is None:
            return None
        try:
            from brain.pipeline import run_digit_ocr, parse_count
        except Exception:
            return None
        icon = self.find_cls(screen, self._TICKET_CLS, conf=0.20,
                             region=(0.0, 0.04, 0.26, 0.36))
        if icon is None:
            # Ticket cls anchor FLICKERS (live 2026-06-09: bounty 票 cls
            # zero-detected on the branch page while the counter rendered
            # plainly top-left). The counter is a stable page fixture →
            # fixed-region OCR fallback on the DIGITS zone only (0708 新皮肤
            # 「持有票券 6/6」布局, 两页帧离线验证 '6/6' ✓)。
            # ⚠这条没有 YOLO 锚, 所以要求 raw 里必须有 '/' —— 不带斜杠的裸数字
            # 可能是页面标题/别的读数蹭进来的, 宁可 None。
            raw = run_digit_ocr(screen.frame, (0.115, 0.121, 0.185, 0.163))
            res = parse_count(raw)
            if res is not None and res[0] is not None and raw and "/" in raw \
                    and res[1] != 0:
                self.log(f"tickets via fixed-region fallback: {res[0]} (raw {raw!r})")
                return res[0]
            self.log(f"[tkdbg] no icon anchor; fallback raw={raw!r}")
            return None
        iw = icon.x2 - icon.x1
        bh = icon.y2 - icon.y1
        y1s = max(0.0, icon.y1 - bh * self._TICKET_YPAD)
        y2s = min(1.0, icon.y2 + bh * self._TICKET_YPAD)
        def _read(xl, xr):
            raw = run_digit_ocr(screen.frame, (min(1.0, icon.x2 + xl * iw), y1s,
                                               min(1.0, icon.x2 + xr * iw), y2s))
            r = parse_count(raw)
            if r is None or r[0] is None:
                return raw, None
            # ⛔ 分母为 0 的读数一律丢弃: 计数器的分母是**上限**, 永远不会是 0。
            # `9/0` 就是零票屏被读大的那个形状(实测 15 次, 真值全是 0/6)。
            # ⚠反过来 `cur > tot` **不能**当无效 —— 帧上确凿存在 `14/6`/`15/6`,
            # 票是可以超出每日回满上限的。拿它当闸会误杀合法读数。
            if r[1] == 0:
                return raw, None
            return raw, r[0]

        last_raw = None
        for xl, xr in self._TICKET_WINDOWS:
            last_raw, v = _read(xl, xr)
            if v is None:
                continue
            # ⭐交叉复核(2026-07-27): 同一串数字换一个**位移过的**窗口再读一次,
            # 两次必须给出同一个 cur 才采信。单窗口 OCR 会把斜体 6 认成 9
            # (实测: 语料 4K 帧 1/139, 低分辨率与形变帧上更高) —— 而票数**读大**
            # 意味着 0 票时仍去出击, 撞的正是青辉石買票框。位移窗口的 crop 内容
            # 不同, 识别错误不完全相关, 所以"两窗独立同意"能滤掉相当一部分。
            # ⚠不一致时**跳过这个窗口继续试**而不是直接 None: 直接 None 会把
            # 读出率打回去, 那正是本次修复要解决的浪费票问题。
            _r2, v2 = _read(xl - 0.5, xr + 0.5)
            if v2 != v:
                self.log(f"[tkdbg] 交叉复核不一致 win({xl},{xr})={v} "
                         f"vs 位移窗={v2} (raw {last_raw!r} / {_r2!r}) — 弃这个窗口")
                continue
            return v
        self.log(f"[tkdbg] icon@({icon.x1:.3f},{icon.y1:.3f},{icon.x2:.3f},"
                 f"{icon.y2:.3f}) conf={icon.confidence:.2f} all "
                 f"{len(self._TICKET_WINDOWS)} strips unread (last raw={last_raw!r})")
        return None

    def _read_ap(self, screen: ScreenState) -> Optional[int]:
        # Calibrated clean-frame read (2026-06-11): the generic read_count span
        # left-truncated 199/240 → '1/240' live → JFD exited with 13 sweeps of
        # AP unspent. _read_topbar_clean votes over clean ADB frames with the
        # per-currency span (AP 0.06) — fail to the live read only if no clean
        # source is registered.
        try:
            from brain.pipeline import _read_topbar_clean
            ap = _read_topbar_clean(UC.TOPBAR_AP)
            if ap is not None:
                return ap
        except Exception:
            pass
        # span 单位 = 图标宽(2026-07-27): 旧 `span=0.10` 是屏幕宽度比例, 换分辨率/
        # 窗口大小就失准。AP 数字串实测右界 5.44 iw(含 "/240"), 取 4.1 只吃分子。
        res = self.read_count(screen, UC.TOPBAR_AP, side="right", span_iw=4.1)
        return res[0] if res is not None else None

    def _pyroxene_buy_dialog(self, screen: ScreenState) -> bool:
        """A 青辉石 icon in the dialog body = a buy-ticket/buy-AP dialog."""
        return self.find_cls(
            screen, UC.TOPBAR_PYROXENE, conf=_CLS_CONF,
            region=_PYROXENE_BODY_REGION
        ) is not None

    def _purchase_structure(self, screen: ScreenState) -> bool:
        """⛔结构白名单闸: 确认框语境下 body(y>0.12) 出现数量 stepper = 购买框。
        (青辉石黑名单在購買AP框上被 v13 漏检打穿过 = 30 青辉石事故, 需要这条
         正交的结构判据兜底; 真購買AP框 stepper@0.96/0.97, 纯AP/票确认框被 dim
         盖住底层 popup → stepper 零检出, 不误伤。)

        ⛔**体力通道已删除(2026-07-26)** —— 与 event_quest._dialog_is_purchase
        同形, 一起改(铁律: 任何一处金钱判据被证伪, 当天 grep 全仓同形一起改;
        2026-07-11 就是因为只改了 event_quest 没迁 schedule, 同一个洞留了两周
        才铸成 30 青辉石课程表票事故)。
        全语料实测(798 帧"确认+取消同屏"): 只靠体力通道判出来的 6 帧
        **6/6 全是纯AP扫荡确认框**, 零真购买框依赖它; 纯AP框体力 conf 可达 0.92,
        与真框 0.77 完全重叠 ⇒ 阈值分层不成立。详见 event_quest 那边的长注释。
        """
        _markers = ("MAX_可点击", "MIN_灰色", "MIN_可点击", "加号",
                    "减号", "加号灰色", "减号灰色")
        for b in (screen.yolo_boxes or []):
            cy = (b.y1 + b.y2) / 2
            if cy <= 0.12 or b.confidence < 0.20:
                continue
            if b.cls_name in _markers:
                return True
        return False

    def _stage_enters(self, screen: ScreenState) -> List[YoloBox]:
        return self.find_all_cls(screen, UC.STAGE_ENTER, conf=_CLS_CONF, region=_STAGE_PANEL)

    # ── tick ────────────────────────────────────────────────────────────────
    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1
        self._phase_ticks += 1

        if self.ticks >= self.max_ticks:
            self.log("timeout")
            return action_done(f"{self.name} timeout")

        if screen.is_loading():
            return action_wait(700, f"{self.name} loading")

        if self.sub_state == "":
            self._goto("enter")

        handler = {
            "enter": self._enter,
            "ticket_check": self._ticket_check,
            "branch": self._branch,
            "stage": self._stage,
            "sortie": self._sortie,
            "confirm": self._confirm,
            "result": self._result,
            "exit": self._exit,
        }.get(self.sub_state)
        if handler is None:
            return action_wait(300, f"{self.name} unknown state")
        return handler(screen)

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        self._enter_ticks += 1
        if self._on_page(screen):
            self.log(f"inside {self.name} → ticket_check")
            self._goto("ticket_check")
            return action_wait(400, "entered page")

        page = self.detect_screen_yolo(screen)
        if page == "Lobby":
            act = self.click_cls(screen, UC.NAV_TASKS, "open campaign hub", conf=0.20)
            if act is not None:
                return act
            # 任务大厅入口 (19f) systematically misses on event-skinned lobbies
            # (live 2026-06-09: the 任務 tile wears a 正在進行考試 banner → cls
            # never fired once all day → enter timed out, 0 sweeps). The tile
            # is a fixed right-side fixture → fixed-slot fallback after a few
            # patient ticks. 根治 = 补标 (clean-flywheel frames have it).
            if self._enter_ticks > 4:
                self.log("任务大厅入口 cls missed → fixed-pos fallback (0.935,0.80)")
                return action_click(0.935, 0.80, "open campaign hub (fixed pos)")
            return action_wait(400, "lobby: NAV_TASKS not seen")
        if page == "Mission":
            # ★ Hall scan (user iron rule 2026-06-11): the per-activity dot is
            # only visible HERE — tile with no red/yellow dot = no work today,
            # exit gracefully instead of entering blind.
            has_work = self.hall_tile_dot(screen, self._HUB_TILE)
            if has_work is False:
                self.log(f"hall scan: {self._HUB_TILE} 无红黄点 → no work today, done")
                return action_done(f"{self.name} no work (hall scan)")
            act = self.click_cls(screen, self._HUB_TILE, f"click {self.name} tile", conf=_CLS_CONF)
            if act is not None:
                return act
            return action_wait(450, "hub: tile not seen")

        if self._enter_ticks > 22:
            self.log("can't reach page, exiting")
            self._goto("exit")
            return action_wait(300, "enter timeout")
        if page is not None:
            return action_back(f"back from {page}")
        return action_wait(450, "entering page")

    def _ticket_check(self, screen: ScreenState) -> Dict[str, Any]:
        # ⛔ Defense ①: read the ticket count; 0 ⇒ never sortie (buy-ticket trap).
        tickets = self._read_tickets(screen)
        if tickets is not None:
            self._tickets = tickets
            if tickets <= 0:
                self.log("tickets = 0 → exit (never buy tickets)")
                self._goto("exit")
                return action_wait(300, "0 tickets → exit")
            self.log(f"tickets = {tickets} → branch")
            self._goto("branch")
            return action_wait(250, "tickets ok → branch")

        # Deep-dive C7 (2026-06-09): unreadable ticket count must FAIL CLOSED.
        # The old "proceed, confirm-dialog guard backstops" relied on a guard
        # whose region was mis-sized (C5) — 票数读不出 ⇒ 不出击, period
        # (money rule #3: 0/unknown tickets → never sortie).
        if self._phase_ticks > 8:
            self.log("ticket count unreadable after retries → exit (money fail-closed)")
            self._goto("exit")
            return action_wait(300, "ticket unread → exit")
        return action_wait(350, "reading ticket count")

    def _branch(self, screen: ScreenState) -> Dict[str, Any]:
        # Already on a stage list (入场键 visible) → stage select.
        if self._stage_enters(screen):
            self._goto("stage")
            return action_wait(250, "stage list visible → stage")

        if self._branch_settle > 0:
            self._branch_settle -= 1
            return action_wait(350, f"branch settle ({self._branch_settle})")

        act = self._click_branch(screen)
        if act is not None:
            self._branch_clicks += 1
            self._branch_settle = 2
            return act

        if self._phase_ticks > 12:
            self.log("branch select timeout — trying stage anyway")
            self._goto("stage")
            return action_wait(300, "branch timeout → stage")
        return action_wait(400, "selecting branch")

    def _stage(self, screen: ScreenState) -> Dict[str, Any]:
        enters = self._stage_enters(screen)
        if not enters:
            # Only locked stages? bail.
            if self.find_cls(screen, UC.STAGE_ENTER_LOCKED, conf=_CLS_CONF, region=_STAGE_PANEL):
                self.log("only locked stages → exit")
                self._goto("exit")
                return action_wait(300, "locked stages → exit")
            if self._phase_ticks > 10:
                self.log("no 入场键 found → exit")
                self._goto("exit")
                return action_wait(300, "no stage cls → exit")
            return action_wait(400, "waiting for 入场键")

        # Swipe to the bottom: when the lowest 入场键 y stops moving, we're there.
        max_y = max(b.cy for b in enters)
        if self._swipe_count < 6 and abs(max_y - self._last_stage_y) > 0.03:
            self._last_stage_y = max_y
            self._swipe_count += 1
            return action_swipe(0.75, 0.72, 0.75, 0.32, 500, "swipe stage list to bottom")

        # Bottom reached → click the lowest (= highest difficulty) 入场键.
        last = max(enters, key=lambda b: b.cy)
        self.log(f"enter highest stage 入场键 ({last.cx:.2f},{last.cy:.2f})")
        self._goto("sortie")
        return action_click_box(last, "enter highest stage")

    def _sortie(self, screen: ScreenState) -> Dict[str, Any]:
        # ⛔ Defense ③ (early): a buy dialog can pop here too.
        if self._pyroxene_buy_dialog(screen):
            self.log("⛔ pyroxene buy dialog at sortie — cancel + exit")
            return self._cancel_and_exit(screen)

        # Confirm dialog already up → confirm state.
        if self.find_cls(screen, [UC.BTN_CONFIRM, UC.BTN_CONFIRM_GREY], conf=_CLS_CONF, region=_CONFIRM_BAND):
            self._goto("confirm")
            return action_wait(200, "confirm dialog → confirm")

        if not self.find_cls(screen, [UC.SWEEP_START, UC.QTY_MAX, UC.QTY_MAX_GREY], conf=_CLS_CONF):
            # 任務資訊 not open yet. The 入場 tap intermittently DROPS under adbd
            # contention (live 2026-06-15: skill tap lost → popup never showed →
            # 0 tickets swept; manual same-pos tap opened it). Root-fixed by the
            # AdbInput I/O lock; self-healing backstop here — re-click the 入場键
            # (bounded) instead of giving up with tickets unspent.
            # ⛔2026-07-27 live 实锤: 旧判据是 `_phase_ticks > 7`(**tick 当计时器**)。
            # tick 速率实测跨度极大(memory frame_age/completion_gap: 自主跑
            # 0.15-0.25 s/tick, 慢 run 到 2.294 s/tick) ⇒ 同一个 "7 tick" 实际
            # 是 **1.05s ~ 16s，差 15 倍**。太短就把"游戏还在加载"误判成"tap 丢了"，
            # 白重试甚至耗尽退出(本轮实测: 连点 3 次入場 → retries 用尽 → exit，
            # **6 张悬赏票一张没花**)。
            # 改墙钟, 且走 `since()`(= game_clock, 会扣掉 step 门的人工停顿) ——
            # 否则逐帧门控时人审一慢就必然触发这条误判。
            # ⚠️3.0s 这个值**没有"入場→任務資訊"的实测支撑**, 是按"比自主跑 7tick
            #   的 1.75s 宽、比慢 run 的 16s 严"取的折中。待用飞轮帧量准后再定。
            #   (仍留 `_phase_ticks >= 2`: 至少观察两帧再判, 防第一帧就误触。)
            # ⚠️同文件还有 4 处 tick-as-timer 未改(L363 >8 / L385 >12 / L399 >10 /
            #   L571 <=18) —— 它们今天没有实锤, 按"不修没坏的判据"暂不动。
            if self._phase_ticks >= 2 and self.since("phase") > 3.0:
                if self._sortie_retries < _SORTIE_MAX_RETRIES:
                    self._sortie_retries += 1
                    self.log(f"任務資訊 未开 (入場 tap 可能丢失) → 回 stage 重点入場键 "
                             f"retry {self._sortie_retries}/{_SORTIE_MAX_RETRIES}")
                    self._goto("stage")
                    return action_wait(300, "re-enter stage (tap-loss retry)")
                self.log("任務資訊 never opened after retries → exit")
                self._goto("exit")
                return action_wait(300, "no sortie popup → exit")
            return action_wait(400, "waiting for 任務資訊 popup")

        # AP gate (JFD): the MAX button sweeps as many times as AP+tickets allow
        # — the GAME caps it, never overspends (you can't go negative on AP).
        # So MAX is safe whenever AP ≥ one sweep; the old `ap ≥ tix×AP_PER_SWEEP`
        # gate wrongly fell back to a SINGLE sweep whenever full tickets couldn't
        # all be afforded (live 2026-06-09: 24 tickets needed 360 AP > 240 cap →
        # safe_to_max False → swept only 1, leaving AP+tickets unspent).
        if self._COSTS_AP and not self._maxed:
            ap = self._read_ap(screen)
            if ap is not None:
                if ap < self._AP_PER_SWEEP:
                    self.log(f"AP {ap} < {self._AP_PER_SWEEP}/sweep → exit (no buy-AP)")
                    self._goto("exit")
                    return action_wait(300, "insufficient AP → exit")
                self._safe_to_max = True  # game caps MAX at affordable count
                self.log(f"AP {ap} ≥ {self._AP_PER_SWEEP} → MAX (game caps to affordable)")
            else:
                # AP unreadable → don't MAX (sweep 1 at a time; grey-confirm guards).
                self._safe_to_max = False
        else:
            self._safe_to_max = True  # bounty: no AP cost → always MAX

        # MAX (one shot) only when affordable; else leave qty=1 (single sweep).
        if not self._maxed and self._safe_to_max:
            # QTY_MAX(MAX_可点击)是弱类: 在明显可点的蓝 MAX 上只 fire 到 conf≈0.26
            # (<0.30, special_sweep.py:228 实测), 用 _CLS_CONF=0.30 检不到 → 退化固定位
            # MAX(2026-06-26 实测 bounty/jfd 都走了固定位)。降到 0.20 地板对齐 special_sweep
            # 让 cls 路径优先(action_click_box 比硬编码 0.84,0.42 跨分辨率更鲁棒)。bounty/jfd
            # 纯票券: 即便 0.20 在灰 MAX 上误 fire, 游戏把 count 钳到持有票数→confirm "用N票券"
            # 不弹买票框, 且 _confirm 青辉石防线兜底 → 安全。固定位 fallback 仍保留兜底。
            max_btn = self.find_cls(screen, UC.QTY_MAX, conf=0.20)
            if max_btn is not None:
                # 幂等钮: 首发可能被稳定门吞(2026-07-21 tick638/667 实锤,
                # _maxed 先置位 → MAX 从没点到只扫1票), 连发两 tick 再落 latch。
                # 第二发同目标被 same-target hold 缓冲, 无连射; 已生效再点无害。
                # ⭐被吞不计发(root 信号, 2026-07-22 tick18/19 实锤): fire1/fire2
                # 连续被稳定门吞 → 两发都没出手 double 额度却耗尽 → _maxed=True
                # 只扫1票。上一发被吞就回滚计数, 直到真点出去两发才落 latch。
                if self.action_suppressed and self._max_fires > 0:
                    self._max_fires -= 1
                    self.log("MAX fire 被稳定门吞 — 回滚计数")
                self._max_fires = getattr(self, "_max_fires", 0) + 1
                if self._max_fires >= 2:
                    self._maxed = True
                self.log(f"set sweep MAX (fire {self._max_fires})")
                return action_click_box(max_btn, "set sweep MAX")
            # No SOLID MAX yet. During the 任務資訊 popup open-animation the MAX
            # button renders grey-then-solid; abandoning on the first miss (live
            # 2026-06-09 JFD) set _maxed=True too early → swept ONCE with 24
            # tickets unspent. Wait a few frames for the solid MAX to settle;
            # only if it never appears (truly greyed = 1 ticket) fall back.
            if self._max_wait < self._MAX_RENDER_WAIT:
                self._max_wait += 1
                return action_wait(350, f"waiting MAX render ({self._max_wait})")
            # cls111 MAX_可点击 flaky (live 2026-06-15: bounty/jfd 只扫了1票, MAX 没
            # 检到就退化 single sweep, 浪费票). 等待后仍没检到 → 点 任務資訊 popup 的
            # 固定位 MAX(special_sweep 已验证 0.84,0.42)再扫, 不退 single。游戏把 MAX
            # 钳到可负担数, 安全; 若 MAX 是灰的(真只剩1票/到上限), 点固定位是无害空操作。
            if not getattr(self, "_max_fixed_tried", False):
                self._max_fixed_tried = True
                self._maxed = True
                self.log("MAX cls 没检到 → 固定位 MAX (防只扫1票)")
                return action_click(*_POS_TICKET_MAX, "set sweep MAX (fixed pos)")
            self._maxed = True
            self.log("MAX greyed after wait → single sweep")

        sweep = self.find_cls(screen, UC.SWEEP_START, conf=_CLS_CONF)
        if sweep is not None:
            self._sweep_btn_wait = 0
            self.log("click 扫荡开始")
            self._goto("confirm")
            return action_click_box(sweep, "click sweep start")
        # ⛔死等根治(2026-07-11 live 实锤: jfd AP=12<15 时 _maxed 已置位跳过
        # AP 闸, 扫荡开始灰色在 0.30 检不出 → 该分支无超时干等 70+ tick 直到
        # skill 上限)。popup 已开而 扫荡开始 连续 6 tick 不出现 = 灰(AP/票
        # 不可负担) → fail-closed 收工, 绝不点任何东西(不给補AP框出现机会)。
        self._sweep_btn_wait = getattr(self, "_sweep_btn_wait", 0) + 1
        if self._sweep_btn_wait > 6:
            self._sweep_btn_wait = 0
            self.log("扫荡开始 6-tick 未出现(灰=不可负担) → exit (fail-closed)")
            self._goto("exit")
            return action_wait(300, "扫荡开始 unavailable → exit")
        return action_wait(400, "waiting for 扫荡开始")

    def _confirm(self, screen: ScreenState) -> Dict[str, Any]:
        # ⭐被吞对账(root 信号, 2026-07-21 tick338 实锤): 上 tick 的 扫荡开始
        # click 被稳定门吞 → 实际没点出去, 屏仍是任務資訊, 但 _goto("confirm")
        # 已提前发生(mutate-before-ack) → pre-transition 闸 8 tick 过期后
        # _purchase_structure 把 popup 自带 stepper 误判成购买框 → cancel+exit
        # 白弃扫(tickets=6 全丢)。回 sortie 重点; confirm/cancel 自身被吞走这
        # 条也只是安全绕行(sortie 会重找 扫荡开始 → 回 confirm)。
        if self.action_suppressed:
            # ⭐吞的是哪个 click 由屏上证据分流(2026-07-22 tick25 实锤: 确认键
            # click 被吞时无条件回 sortie 会在 dim 帧上找不到 扫荡开始 → 假
            # exit): 任務資訊仍在(扫荡开始可见+无确认键) → 回 sortie 重点
            # 扫荡开始; 确认框已开 → 留在 confirm, 本 tick 判定链会再检出
            # 确认键重点。
            if (self.find_cls(screen, UC.SWEEP_START, conf=_CLS_CONF) is not None
                    and self.find_cls(screen, UC.BTN_CONFIRM, conf=_CLS_CONF,
                                      region=_CONFIRM_BAND) is None):
                self.log("扫荡开始 click 被稳定门吞 → 回 sortie 重点")
                self._goto("sortie")
                return action_wait(250, "suppressed click → re-run sortie")
            self.log("confirm click 被稳定门吞 → 留在 confirm 重点")

        # Sweep done already (掃蕩完成 popped) → result.
        # ⛔2026-07-27 live 实锤(bounty 6 张票 + 2 倍奖励差点又白丢): 这条分支在
        # **掃蕩前的确认框**上误触发 —— 「要使用6懸賞通緝票券掃蕩6次嗎?」弹出时,
        # 確認键从下往上**弹入动画**, 途中扫过 _DONE_CONFIRM_BAND(y 0.74-0.90),
        # 而它的终值在 cy 0.699(带外)。于是: 假计一次 cycle → goto result →
        # **真正的確認从没点过** → result 里票数被弹窗挡住读不出 → exit → 点叉叉
        # = 取消 = 6 张票一张没花。与今早 cafe「領取」在弹入动画帧上锚点是**同一
        # 个机制**(0.825→0.733), 只是这里错的是判定不是落点。
        #
        # 负门禁: **结果弹窗永远没有取消键**。全语料 44,669 有检出帧实测:
        #   获得奖励在屏 494 帧 → 其中带取消键 **0 帧**(规则 0 反例)
        #   確認键落在 DONE 带内 588 帧 → **其中 252 帧同屏有取消键**(全是确认框)
        #   且这 588 帧**没有一帧**带 获得奖励
        # ⇒ 只加负门禁, 不删裸 確認 信号(剩下 336 帧无取消键的可能是 获得奖励
        #   漏检的真完成弹窗, 那正是这条裸信号存在的理由)。
        # 这条规则 money_safety 2026-06-10 就写过("確認+取消同帧 = 不是结果弹窗"),
        # 一直没传导到这里 —— 与 y 留白同款"修一处没 grep 全仓"。
        _cancel_up = self.find_cls(screen, UC.BTN_CANCEL, conf=_CLS_CONF) is not None
        if (not _cancel_up) and (
                self.find_cls(screen, UC.GOT_REWARD, conf=_CLS_CONF, region=_DONE_CONFIRM_BAND) is not None
                or self.find_cls(screen, UC.BTN_CONFIRM, conf=_CLS_CONF, region=_DONE_CONFIRM_BAND) is not None):
            # Could be the 掃蕩完成 reward popup (确认键 lower at ~0.81).
            # ⭐sweep_cycles 落账移到这里(到达证据=掃蕩完成弹窗) — 旧码在点
            # 确认键前 +1+goto result, 确认被吞时假报 cycle 完成(JFD"没去"
            # 真相, 2026-07-22 用户抓)。goto 一次性, 不会重复计。
            self._sweep_cycles += 1
            self.log(f"sweep done popup (cycle {self._sweep_cycles} 落地)")
            self._goto("result")
            return action_wait(150, "sweep done popup → result")

        # ⛔⭐帧龄防误杀(2026-07-21 live tick640/669 实锤): 点完 扫荡开始 的下一
        # tick 帧仍是未变暗的 任務資訊(扫荡开始+stepper 可见, 无确认键) —
        # _purchase_structure 会把 popup 自带 stepper 当购买框 X 掉整个弹窗
        # (0 票扫出+假报完成)。确认框 cls 证据出现前不跑 abort 判定; 真购买框
        # 下 扫荡开始 被 dim 压掉检不出(t46 同源实证), 不会误放行。
        # 时限: 丢 tap 时不无限遮蔽 result 兜底。⚠2026-07-22 实锤: 7 tick/s 化
        # 后 8 tick≈1.1s < 确认框渲染 1-2s → 闸必过期误判自家 stepper 为购买框
        # (bounty 两连 cancel+exit 弃 6 票) → 提到 18(≈2.5s), result 兜底同步 24。
        if (self._phase_ticks <= 18
                and self.find_cls(screen, UC.SWEEP_START, conf=_CLS_CONF) is not None
                and self.find_cls(screen, UC.BTN_CONFIRM, conf=_CLS_CONF,
                                  region=_CONFIRM_BAND) is None
                and self.find_cls(screen, UC.BTN_CONFIRM_GREY, conf=_CLS_CONF,
                                  region=_CONFIRM_BAND) is None
                and not self._pyroxene_buy_dialog(screen)):
            return action_wait(350, "confirm: 帧仍停任務資訊(pre-transition) — 等确认框")

        # ⛔ pyroxene buy dialog → cancel + exit.
        if self._pyroxene_buy_dialog(screen):
            self.log("⛔ pyroxene buy dialog at confirm — cancel + exit")
            return self._cancel_and_exit(screen)

        # ⛔ 购买框结构闸(青辉石漏检兜底, 2026-07-11): stepper/体力在 body
        # = 購買AP/購買票券框 → 取消收工, 绝不点确认。
        # ⚠语境前置(2026-07-22 三连误判实锤): 确认框弹出的过渡 dim 帧上
        # SWEEP_START 被压检不出(闸不 hold)而 stepper 在 conf 0.20 地板下仍
        # 检出 → 自家任務資訊被判购买框 cancel+exit 弃扫。购买框必有确认/取消
        # 大按钮(conf 0.97+ 稳定) → 无按钮=过渡态, 不判定只 wait(不点=安全);
        # 有按钮才跑结构闸。真购买框语境不变, fail-closed 方向不变。
        _dialog_btn_up = (
            self.find_cls(screen, UC.BTN_CONFIRM, conf=_CLS_CONF, region=_CONFIRM_BAND) is not None
            or self.find_cls(screen, UC.BTN_CONFIRM_GREY, conf=_CLS_CONF, region=_CONFIRM_BAND) is not None
            or self.find_cls(screen, UC.BTN_CANCEL, conf=_CLS_CONF) is not None)
        if _dialog_btn_up and self._purchase_structure(screen):
            self.log("⛔ purchase-dialog structure at confirm — cancel + exit")
            return self._cancel_and_exit(screen)

        # ⛔ greyed confirm = insufficient (AP/ticket) → cancel + exit.
        if self.find_cls(screen, UC.BTN_CONFIRM_GREY, conf=_CLS_CONF, region=_CONFIRM_BAND) is not None:
            self.log("⛔ confirm greyed (insufficient) — cancel + exit")
            return self._cancel_and_exit(screen)

        confirm = self.find_cls(screen, UC.BTN_CONFIRM, conf=_CLS_CONF, region=_CONFIRM_BAND)
        if confirm is not None:
            # ⭐after-ack(2026-07-22): 不提前 goto result / 不提前计 cycle —
            # 点击被吞时留在 confirm(入口对账 fall-through 会重点); 真点上后
            # 掃蕩完成弹窗出现 → 顶部 sweep-done 分支落账+进 result。
            self.log("confirm sweep — currency verified (票券), 等掃蕩完成落地")
            return action_click_box(confirm, "confirm sweep (tickets, not pyroxene)")

        if self._phase_ticks > 24:
            self._goto("result")
            return action_wait(300, "no confirm dialog → result")
        return action_wait(350, "waiting for sweep-confirm dialog")

    def _result(self, screen: ScreenState) -> Dict[str, Any]:
        # 掃蕩完成 reward popup — re-detect (WGC transition frames). Dismiss via
        # the lower 确认键, or GOT_REWARD header / 点击继续字样.
        # ⛔2026-07-27: 取消键在屏 ⇒ 这**不是**结果弹窗, 是个确认框(全语料 494 帧
        # 结果弹窗里带取消键的 **0 帧**)。走到这里说明上游误判了 —— 不能拿"关奖励"
        # 的手去点确认框的確認(那是不受控消费), 也不能干等到票数读不出就收工
        # (今天 bounty 就是这么把 6 张票 + 2 倍奖励丢掉的)。退回 confirm, 交给带
        # 全部金钱闸的正规 handler 判定。
        if self.find_cls(screen, UC.BTN_CANCEL, conf=_CLS_CONF) is not None:
            self.log("result 见取消键 ⇒ 不是结果弹窗(是确认框) — 退回 confirm")
            self._goto("confirm")
            return action_wait(200, "result 误入 → 回 confirm")
        cont = self.find_cls(screen, UC.STORY_TAP_CONTINUE, conf=_CLS_CONF)
        if cont is not None:
            return action_click_box(cont, "dismiss sweep reward (continue)")
        done_confirm = self.find_cls(screen, UC.BTN_CONFIRM, conf=_CLS_CONF, region=_DONE_CONFIRM_BAND)
        if done_confirm is not None:
            self.log("dismiss 掃蕩完成 (确认键)")
            return action_click_box(done_confirm, "dismiss 掃蕩完成")
        got = self.find_cls(screen, UC.GOT_REWARD, conf=_CLS_CONF)
        if got is not None:
            return action_click_box(got, "dismiss sweep reward (header)")

        # Reward dismissed → re-read tickets. MAX drains all → usually 0 now.
        if self._sweep_cycles >= self._MAX_SWEEP_CYCLES:
            self.log("sweep cycle cap → exit")
            # 帽退时票数没重读过 — 报 UNKNOWN, 不带入场陈旧值假报 LEFTOVER
            self._post_sweep_unread = True
            self._goto("exit")
            return action_wait(300, "sweep cap → exit")

        tickets = self._read_tickets(screen)
        if tickets is not None and tickets <= 0:
            # ⚠必须回写 _tickets(2026-07-25 workflow 实锤): 主成功路径原先不回写,
            # exit_report 拿着入场旧值(如 6)把每一次"扫光收工"误报成 LEFTOVER —
            # 且 exit_report 的 CLEAN-after-sweep 分支因此成了不可达死码。
            self._tickets = tickets
            self.log("tickets drained (0) → exit")
            self._goto("exit")
            return action_wait(300, "tickets 0 → exit")
        if tickets is not None and tickets > 0:
            # Single-sweep path (JFD low-AP) leaves tickets → sweep again.
            self._tickets = tickets
            self._maxed = self._safe_to_max  # keep MAX state if we maxed
            self._max_fires = 0              # 双发 latch 随轮次复位
            self.log(f"{tickets} tickets remain → sortie again")
            self._goto("sortie")
            return action_wait(300, "more tickets → sortie")

        # 票数读不出 → 不再"假设扫光了"就走人。
        # ⛔2026-07-25 双重问题, 一起修:
        #  ① 单位错: `_phase_ticks > 8` 是 **tick**, zero-wait 后自主跑
        #     0.15-0.25 s/tick(口径见 BaseSkill.mark) ⇒ 真实只有 **1.2-2.0s**,
        #     一次 OCR 抖动就够把整轮票判死。改墙钟 6s 重读窗。
        #  ② 结论错: "读不出" 被写成 "MAX likely drained" —— 这就是
        #     [[completion-gap]] 里"悬赏票剩多少 未知"那一格的来源。**没读到
        #     就不许下资源结论**, 改成 UNKNOWN 并落进竣工判据供出口审计。
        if self.since("post_sweep") > _POST_SWEEP_READ_SEC:
            self._post_sweep_unread = True
            self.log(f"⚠post-sweep 票数连续 {_POST_SWEEP_READ_SEC:.0f}s 读不出 "
                     f"→ 收工(⚠**未读到票数, 不做'已扫光'结论**; "
                     f"入场时 {self._tickets} 张)")
            self._goto("exit")
            return action_wait(300, "post-sweep 票数未知 → exit")
        return action_wait(350, "settling after sweep (重读票数中)")

    def _cancel_and_exit(self, screen: ScreenState) -> Dict[str, Any]:
        self._goto("exit")
        cancel = self.find_cls(screen, UC.BTN_CANCEL, conf=_CLS_CONF)
        if cancel is not None:
            return action_click_box(cancel, "cancel (never buy pyroxene)")
        close = self.find_cls(screen, UC.BTN_CLOSE_X, conf=_CLS_CONF)
        if close is not None:
            return action_click_box(close, "close buy dialog (X)")
        return action_back("dismiss buy dialog (ESC)")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        page = self.detect_screen_yolo(screen)
        if page == "Lobby":
            self.log(f"done ({self._sweep_cycles} sweeps)")
            return action_done(f"{self.name} complete")
        if page == "Mission":
            # ⭐0-sweep 对账(2026-07-21 实锤: 误 abort 后 0 票扫出仍报 complete):
            # 入场读到票>0 却一次没扫 → 一次性回炉重走(全部金钱防线重跑, 最坏
            # =再 abort 收工, 绝不多点确认); 仍 0 则 done reason 亮明供对账。
            if (self._sweep_cycles == 0 and (self._tickets or 0) > 0
                    and not self._zero_sweep_retried):
                self._zero_sweep_retried = True
                self.log(f"⚠ tickets={self._tickets} but 0 sweeps → re-enter once")
                self._goto("enter")
                self._enter_ticks = 0
                return action_wait(300, "0-sweep 对账不平 → 回炉重试")
            # Stay on the hub — the next campaign skill re-uses it.
            self.log(f"done on hub ({self._sweep_cycles} sweeps)")
            _tag = (" [⚠0 sweeps, tickets unspent]"
                    if (self._sweep_cycles == 0 and (self._tickets or 0) > 0) else "")
            return action_done(f"{self.name} complete (on hub){_tag}")
        if self._phase_ticks > 16:
            return action_done(f"{self.name} exit timeout")
        # A leftover 掃蕩完成 / dialog blocks ESC — dismiss its 确认键/X first.
        done_confirm = self.find_cls(screen, UC.BTN_CONFIRM, conf=_CLS_CONF, region=_DONE_CONFIRM_BAND)
        if done_confirm is not None:
            return action_click_box(done_confirm, "exit: dismiss leftover result")
        close = self.find_cls(screen, UC.BTN_CLOSE_X, conf=_CLS_CONF)
        if close is not None:
            return action_click_box(close, "exit: close dialog")
        back = self.find_cls(screen, UC.BTN_BACK, conf=_CLS_CONF)
        if back is not None:
            return action_click_box(back, "exit: back key")
        return self.nav_home(screen, "ticket exit")
