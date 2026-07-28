"""CraftSkill — start quick-crafts + collect finished ones (pure-YOLO rewrite).

Verified flow (interactive probe 2026-06-01, data/_craft_probe_log.md). The old
skill had a "进制造就退" bug (it never started crafts) AND a SAFETY hole: it
looked for the wrong claim cls and could trigger the 立即完成 rush dialog.

★★ HARD RULES (probe-derived) ★★
- The craft-main 一次領取 (CLAIM_ONCE_YELLOW) is AMBIGUOUS:
    • crafts FINISHED → tapping it collects them FREE (direct GOT_REWARD).
    • crafts RUNNING  → tapping it pops 「是否立即完成?」 which spends 製造券
      (a consumable). We NEVER spend券 → if a confirm dialog appears after the
      tap, CANCEL it. (製造券 isn't pyroxene, but user spec: 让制造自然跑完.)
- per-slot 立即完成 (blue, no cls) is NEVER clicked (we have no cls → never
  blind-click there).
- START is safe: 快速制造 costs 信用点 only (probe: 19,000/unit). 快速制造 →
  MAX → 开始制造 → 确认键(確定製造N次) spends credits (abundant) — fine.

State machine
-------------
enter    lobby → NAV_CRAFT → Craft page (CRAFT_QUICK/CRAFT_START signature).
collect  一次领取黄色 → tap. If a confirm dialog (确认键+取消键) appears = the
         券-rush 立即完成 → CANCEL (never spend券). GOT_REWARD = free collect
         (dismissed globally). 一次领取灰色 / none = nothing to collect.
start    快速制造 → MAX_可点击 → 开始制造 → 确认键(确定製造N次, credits, safe).
         Nothing startable (slots busy) → exit.
exit     BTN_HOME / BTN_BACK → lobby → done.

Detectors: base "ui" only.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from brain.skills.base import (
    BaseSkill, ScreenState, YoloBox,
    action_click, action_click_box, action_wait, action_back, action_done,
)
from brain.skills import ui_classes as UC

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
# ⛔按游戏日的"今天进过没有"台账 —— 红点门控第三态的解药, 见 should_run。
_CRAFT_STATE_FILE = _DATA_DIR / "craft_state.json"


def _game_day() -> str:
    """BA 游戏日(04:00 刷新) ISO 日期 —— 与 schedule/cafe 的 `_game_day` 同一份
    实现(锚服务器时区 UTC+8, 不用裸 datetime.now(); 2026-07-27 时区事故结论)。"""
    from datetime import datetime, timedelta, timezone
    _SERVER_TZ = timezone(timedelta(hours=8))       # BA 繁中服
    return (datetime.now(_SERVER_TZ) - timedelta(hours=4)).date().isoformat()


def _load_craft_state() -> dict:
    try:
        if _CRAFT_STATE_FILE.exists():
            s = json.loads(_CRAFT_STATE_FILE.read_text(encoding="utf-8"))
            if s.get("game_day") == _game_day():
                return s
    except Exception:
        pass
    return {"game_day": _game_day(), "visited": False, "started": 0}


def _save_craft_state(state: dict) -> None:
    try:
        _CRAFT_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CRAFT_STATE_FILE.write_text(json.dumps(state, ensure_ascii=False),
                                     encoding="utf-8")
    except Exception:
        pass


_CLS_CONF = 0.30
# Center-bottom band where craft confirm / 立即完成 dialogs put 确认键/取消键
# (probe: y≈0.70 for both the start-confirm and the券-rush dialog).
_DIALOG_BAND = (0.28, 0.60, 0.72, 0.82)

_ENTER_MAX = 20
_COLLECT_MAX = 16
_START_MAX = 18
_EXIT_MAX = 14
# 等**真实动画**的超时一律墙钟(tick 数不是时间单位, 见 BaseSkill.mark 注释)
_COLLECT_SETTLE_SEC = 3.2   # 一次领取 → GOT_REWARD 弹窗实测 2.5-3s
_QUICK_SETTLE_SEC = 2.0     # 立即完成(券)对话框渲染
_CONFIRM_WAIT_SEC = 2.5     # 开始制造 → 「確定製造N次嗎?」确认框渲染(after-ack 窗口)
_CONFIRM_ACK_SEC = 1.5      # 点完確認 → 确认框关掉(after-ack, 防尾发第二次確認)


class CraftSkill(BaseSkill):
    def should_run(self, screen: ScreenState) -> bool:
        """红点(可领) **OR** 今日未进过 —— 红点门控只有两态是个纯产出损失洞。

        ⛔2026-07-28 帧实锤(用户点名要考的三件事之一): 旧门控只认两态 ——
          ① 有红点 = 造好了可领 → 进
          ② 无红点 = 还在造 → skip
        实际存在**第三态: 三个槽位全空、根本没在造**, 它同样**没有红点**
        (当天帧: 材料清單三行全是「＋ 開始製造」, 一次領取灰) ⇒ 永远走 ②
        ⇒ **一次都不会开新的制造**。制造是纯免费产素材的, 槽位空着 = 纯损失,
        而且一旦空了就再也不会有红点 —— 这个洞会**自锁**, 越久越亏。

        修法与 buy_pyroxene/schedule 的日台账同构: 无红点时, 只要**今天还没进
        过制造页**就进去看一眼(进页后 _collect/_start 自己按帧决定领还是造)。
        进过且没红点才 skip —— 这样每个游戏日最多多花一次进页往返(~20 tick),
        代价远小于它防的损失(判据的代价不能大于它要防的损失, 2026-07-28 教训)。
        """
        if self.dot_on_entry(screen, [UC.NAV_CRAFT], dot_classes=(UC.DOT_RED,)):
            return True
        st = _load_craft_state()
        if not st.get("visited"):
            self.log("制造无红点, 但今日未进过 → 仍进(第三态: 槽位可能全空)")
            return True
        self.log(f"制造无红点 且今日已进过(started={st.get('started', 0)}) → skip")
        return False

    def __init__(self):
        super().__init__("Craft")
        self.max_ticks = 90
        self._init_state()

    def _init_state(self) -> None:
        self._phase_ticks: int = 0
        self._entered: bool = False
        # ⭐"真的站在制造页上过"的帧证据(不是"点过入口")。日台账只在它为真时才写,
        # 否则一次入口点空就把今天的制造锁死(fail-closed 方向: 宁可明天再进)。
        self._reached: bool = False
        self._collect_done: bool = False
        self._collect_settle: bool = False   # 墙钟 timer "collect_settle" 计时
        self._maxed_clicks: int = 0
        self._quick_settle: bool = False     # 墙钟 timer "quick_settle" 计时
        self._started: bool = False
        self._start_clicked: bool = False  # we pressed 开始制造 → confirm is ours
        self._craft_confirm_clicked: bool = False  # 点過確認; latch _started 仅凭到达证据
        self._start_taps: int = 0          # 开始制造 clicks w/o resulting confirm
        self._claims: int = 0

    def reset(self) -> None:
        super().reset()
        self._init_state()

    def _goto(self, sub_state: str) -> None:
        self.sub_state = sub_state
        self._phase_ticks = 0

    def _mark_visited(self) -> None:
        """写今日台账 —— 只在**帧证实到过制造页**之后写(见 _reached)。"""
        if not self._reached:
            return
        st = _load_craft_state()
        st["game_day"] = _game_day()
        st["visited"] = True
        st["started"] = int(st.get("started", 0)) + (1 if self._started else 0)
        _save_craft_state(st)

    def exit_report(self):
        """竣工判据 —— 这个 skill 的价值就两件事: 领成品 + 把空槽填上。

        ⛔它 2026-07-28 之前**从来没开过一次新制造**(should_run 第三态洞), 而
        出口只报 UNKNOWN, 所以没人审计得出来。判据必须能把"进去了但既没领也
        没造"这种空跑单独标出来。"""
        if not self._reached:
            return ("UNKNOWN", f"没到过制造页(sub={self.sub_state}) — 入口没点开")
        if self._started and self._claims:
            return ("CLEAN", f"领了 {self._claims} 次成品 + 开了新制造")
        if self._started:
            return ("CLEAN", "开了新制造(无成品可领)")
        if self._claims:
            return ("LEFTOVER",
                    f"领了 {self._claims} 次成品但**没开新制造** — "
                    f"槽位可能空着(材料不足? 開始製造 灰?)")
        return ("LEFTOVER",
                "进了制造页但**既没领也没造** — 查 一次領取黄/快速制造 是否漏检")

    # ── helpers ──────────────────────────────────────────────────────────
    def _is_craft(self, screen: ScreenState) -> bool:
        page = self.detect_screen_yolo(screen)
        if page == "Lobby":
            self._entered = False
            return False
        if page == "Craft":
            self._reached = True
            return True
        # Direct craft markers (快速制造 / 开始制造 are craft-only).
        if self.find_cls(screen, [UC.CRAFT_QUICK, UC.CRAFT_START], conf=_CLS_CONF) is not None:
            # ⭐强证据(craft-only cls 在帧上) → 才算"到过", 日台账认这个。
            # 下面 `return self._entered` 那条是弱证据(只是"点过入口且不在大厅"),
            # 不写台账 —— 否则一次误判就把今天的制造锁死。
            self._reached = True
            return True
        # ★ Mis-ID guard: the craft main row shows 一次领取黄/灰, which is ALSO
        # Mail's PAGE_SIGNATURE → detect_screen_yolo wrongly returns "Mail" on the
        # craft page (live 2026-06-02: craft entered fine but bounced back to
        # lobby forever). Once we've clicked our own craft entry (_entered),
        # trust ANY non-lobby screen as craft rather than the false Mail ID.
        return self._entered

    def _confirm_dialog(self, screen: ScreenState) -> Optional[YoloBox]:
        """A 2-button confirm dialog (确认键 AND 取消键 in the button band)."""
        confirm = self.find_cls(screen, UC.BTN_CONFIRM, conf=_CLS_CONF, region=_DIALOG_BAND)
        cancel = self.find_cls(screen, UC.BTN_CANCEL, conf=_CLS_CONF, region=_DIALOG_BAND)
        if confirm is not None and cancel is not None:
            return confirm
        return None

    # ── tick ────────────────────────────────────────────────────────────────
    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1
        self._phase_ticks += 1

        if self.ticks >= self.max_ticks:
            self.log(f"timeout (claims={self._claims}, started={self._started})")
            self._mark_visited()
            return action_done("craft timeout")

        # Global: free-collect reward popup → dismiss via continue / header.
        cont = self.find_cls(screen, UC.STORY_TAP_CONTINUE, conf=_CLS_CONF)
        if cont is not None:
            return action_click_box(cont, "dismiss reward via continue")
        got = self.find_cls(screen, UC.GOT_REWARD, conf=_CLS_CONF)
        if got is not None:
            self._claims += 1
            return action_click_box(got, "dismiss reward (got crafts)")

        if screen.is_loading():
            return action_wait(700, "craft loading")

        if self.sub_state == "":
            self._goto("enter")

        handler = {
            "enter": self._enter,
            "collect": self._collect,
            "start": self._start,
            "exit": self._exit,
        }.get(self.sub_state)
        if handler is None:
            return action_wait(300, "craft unknown state")
        return handler(screen)

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        if self._is_craft(screen):
            self.log("inside craft → collect")
            self._goto("collect")
            return action_wait(400, "entered craft")

        if screen.is_lobby():
            # 0.20 (not _CLS_CONF=0.30): 制造入口 is a weak cls that often fires
            # in the 0.2-0.3 band — the model floor is already 0.20, so a 0.30
            # skill filter throws away real hits (user 2026-06-09: dashboard
            # 低conf标记也抓得准). Total misses still fall to adjacency below.
            act = self.click_cls(screen, UC.NAV_CRAFT, "open craft", conf=0.20)
            if act is not None:
                self._entered = True
                return act
            # NAV_CRAFT (制造入口) is weak (16f) and flickers below threshold on
            # live frames (verified 2026-06-02: missed all 90 ticks while shop/
            # social detected fine). Fall back to its FIXED bottom-nav slot:
            # order is …社交·制造·商店·招募, evenly spaced ~0.092. Anchor on the
            # reliably-detected 商店入口 (craft = one slot LEFT) or 社交入口.
            shop = self.find_cls(screen, UC.NAV_SHOP, conf=_CLS_CONF)
            if shop is not None:
                self._entered = True
                self.log("制造入口 missed → adjacent fallback (shop − 1 slot)")
                return action_click(max(0.05, shop.cx - 0.092), shop.cy, "open craft (adjacent to shop)")
            social = self.find_cls(screen, UC.NAV_SOCIAL, conf=_CLS_CONF)
            if social is not None:
                self._entered = True
                self.log("制造入口 missed → adjacent fallback (social + 1 slot)")
                return action_click(min(0.95, social.cx + 0.094), social.cy, "open craft (adjacent to social)")
            if self._phase_ticks > _ENTER_MAX:
                self.log("制造入口 + adjacency unavailable → give up")
                return action_done("craft entry unreachable")
            self.log("on lobby but no 制造入口/邻位 — waiting")
            return action_wait(400, "waiting for 制造入口 cls")

        if self._phase_ticks > _ENTER_MAX:
            return action_done("could not reach craft")
        if len(screen.yolo_boxes or []) < 2:
            return action_wait(700, "no UI detected, likely loading")
        return self.nav_home(screen, "craft recover")

    def _collect(self, screen: ScreenState) -> Dict[str, Any]:
        if not self._is_craft(screen) and not self._confirm_dialog(screen):
            if screen.is_lobby():
                self._goto("enter")
                return action_wait(300, "collect: back on lobby, re-enter")
            if self._phase_ticks > _COLLECT_MAX:
                self._goto("exit")
                return action_wait(300, "collect lost craft → exit")
            return action_wait(400, "waiting for craft UI (collect)")

        # ★ A confirm dialog here = the 立即完成(券) rush from tapping 一次領取
        # while crafts are RUNNING → CANCEL, never spend券.
        dlg = self._confirm_dialog(screen)
        if dlg is not None:
            self.log("⛔ 一次领取 → 立即完成(券) dialog → cancel, never spend券")
            self._collect_done = True
            cancel = self.find_cls(screen, UC.BTN_CANCEL, conf=_CLS_CONF, region=_DIALOG_BAND)
            if cancel is not None:
                return action_click_box(cancel, "cancel 立即完成 (keep券)")
            return action_back("cancel 立即完成 (ESC)")

        if self._collect_settle:
            _w = self.since("collect_settle")
            if _w < _COLLECT_SETTLE_SEC:
                return action_wait(400, f"collect settle ({_w:.1f}s)")
            self._collect_settle = False

        if self._collect_done:
            self._goto("start")
            return action_wait(250, "collect done → start")

        # 一次领取黄色: finished crafts collect FREE; running ones pop the券
        # dialog (caught above). Tap it, then settle to see which happens.
        yellow = self.find_cls(screen, UC.CLAIM_ONCE_YELLOW, conf=_CLS_CONF)
        if yellow is not None:
            self.log("tapping 一次领取黄色 (collect finished crafts)")
            # GOT_REWARD popup renders ~2.5-3s after the tap — 2026-06-12
            # (t0134-0139) 用 2 tick 时, skill 在弹窗出现**之前**就回到 start、
            # 看到被遮住的屏、判 "nothing startable" 退出 = 收了昨天的制造却
            # 一个新的都没开。当时改成 4 tick。
            # ⛔2026-07-25: 那个 4 是**tick**。zero-wait 后自主跑 0.15-0.25
            # s/tick(口径见 BaseSkill.mark) ⇒ 真实只剩 **0.6-1.0s**, 远低于
            # 2.5-3s 需求 —— **2026-06-12 那次修复在 zero-wait 上线后已被悄悄
            # 作废**, 同一个 bug 修过又复活。改墙钟, 直接对着现象的真实时长写。
            self._collect_settle = True
            self.mark("collect_settle")
            return action_click_box(yellow, "collect finished crafts")

        # Grey claim-all or nothing → nothing (free) to collect.
        self.log("no 一次领取黄色 → nothing to collect, start phase")
        self._goto("start")
        return action_wait(250, "no free collect → start")

    def _start(self, screen: ScreenState) -> Dict[str, Any]:
        if not self._is_craft(screen) and not self._confirm_dialog(screen):
            if screen.is_lobby():
                self._goto("exit")
                return action_wait(300, "start: on lobby → exit")
            if self._phase_ticks > _START_MAX:
                self._goto("exit")
                return action_wait(300, "start lost craft → exit")
            return action_wait(400, "waiting for craft UI (start)")

        # craft-start confirm (確定製造N次) — ONLY trust it after WE clicked
        # 开始制造 (deep-dive r2 C5): an unconditional confirm here would also
        # confirm a leaked 立即完成 dialog = spends 製造券. Unexpected dialog
        # in start state ⇒ cancel, never confirm.
        dlg = self._confirm_dialog(screen)
        if dlg is not None:
            # ⛔尾发闸(2026-07-28 补, double_fire_family 第五例): 点完 確認 之后
            # 下一 tick 关闭动画还没走完 → 框仍在屏 → _start_clicked 依旧 True
            # ⇒ **再点一发確認**。第二发落在 (0.598,0.699), 框关掉后那个位置是
            # 快速製造 面板的 材料清單/節點 区 —— 这次碰巧无害, 但"这次碰巧压到
            # 什么"从来不是评估尾发的口径(见 memory double_fire_family)。
            # 状态挂在**物理动作**(確認已发)而不是代码路径上, 等帧证据(框消失)。
            if self._craft_confirm_clicked:
                _w = self.since("craft_confirm")
                if _w < _CONFIRM_ACK_SEC:
                    return action_wait(300, f"確認已发 — 等确认框关掉"
                                            f"(after-ack {_w:.1f}s)")
                # 窗口过了框还在 ⇒ 那一发被吞了 → 允许重发一次(不是连发)
                self.log("確認发出后确认框仍在 — 判为被吞, 重发一次")
                self._craft_confirm_clicked = False
            if self._start_clicked:
                # 不提前 latch _started(2026-07-21 mutate-before-ack 根治: reason
                # 无"確認"被稳定门吞时旧码已 _started=True+goto exit → craft 没开始
                # 却报 done)。latch 延迟到确认框消失(到达证据, 下方); reason 加
                # "確認键"稳定门豁免立即点。
                self.log("confirming craft start (確定製造N次, 耗信用点 — safe)")
                self._craft_confirm_clicked = True
                self.mark("craft_confirm")
                return action_click_box(dlg, "confirm craft start (credits, 確認键)")
            self.log("⛔ unexpected confirm dialog in start state (开始制造 not clicked) → cancel (券-safe)")
            cancel = self.find_cls(screen, UC.BTN_CANCEL, conf=_CLS_CONF)
            if cancel is not None:
                return action_click_box(cancel, "cancel unexpected dialog (券-safe)")
            return action_back("dismiss unexpected dialog (券-safe)")

        # 到达证据: 已点過確認 且确认框现已消失 → craft 已开始 → latch _started
        # (post-ack, 绝不在点击前)。
        if self._craft_confirm_clicked:
            self.log("确认框消失 → craft 已开始 → exit")
            self._started = True
            self._goto("exit")
            return action_wait(300, "craft started → exit")

        # In the 快速制造 dialog? MAX_可点击 / 开始制造 present.
        # max_btn 找 可点击 或 灰色(已到顶) — 后者也是"在 dialog 里"的锚点。
        max_btn = self.find_cls(screen, [UC.QTY_MAX, UC.QTY_MAX_GREY], conf=_CLS_CONF)
        if (max_btn is not None and max_btn.cls_name == UC.QTY_MAX
                and self._maxed_clicks < 3):
            self._maxed_clicks += 1
            self.log("set MAX quantity (YOLO MAX_可点击)")
            return action_click_box(max_btn, "set craft quantity MAX")
        start_btn = self.find_cls(screen, UC.CRAFT_START, conf=_CLS_CONF)
        if start_btn is not None:
            # ⚠2026-06-16 实测: 材料不足时 開始製造 变灰但 CRAFT_START cls 仍 fire,
            # 点了没反应/不弹確定製造框 → 旧码每 tick 重点到 max_ticks 超时空转 60+ 拍。
            # 成功时点 1 次就弹確定製造 → _confirm_dialog 接走(_start_taps 到不了 3)。
            # 点≥3 次仍无确认框 = 灰按钮材料不足 = 今天造不了 → pass(用户 2026-06-16:
            # "材料不够造不了就 pass 就行")。钱安全(灰按钮本就不扣信用点)。
            if self._start_taps >= 8:
                self.log("开始制造 ×8 无確定製造框 → 灰(材料不足) → skip craft(8>dedup hold-cap5, 让丢点重试先落)")
                self._goto("exit")
                return action_wait(300, "craft 开始制造 greyed (材料不足) → skip")
            self._start_taps += 1
            self.log("clicking 开始制造")
            self._start_clicked = True
            self.mark("start_click")     # after-ack 窗口起点(见下方 fall-through 闸)
            return action_click_box(start_btn, "start craft (开始制造)")
        # 兜底: 开始制造(idx444) 是 v6b 漏检的 missing cls (probe 旧模型 0.93,
        # v6b 退步漏检 → craft 卡死, live 2026-06-06)。已点过 MAX (= 在 dialog 里)
        # 且 MAX/MAX灰 检出 → 用它外推开始制造位置 (probe: MAX(0.926,0.713) →
        # 开始制造(0.870,0.812), 偏移 cx-0.056/cy+0.10; dialog 布局固定, 归一化
        # 跨分辨率)。点中 → 弹「確定製造N次」确认框 → _confirm_dialog 收口(信用点,
        # 安全)。根治靠飞轮补 开始制造 样本 → v6c。
        if self._maxed_clicks > 0 and max_btn is not None:
            if self._start_taps >= 8:
                self.log("开始制造(外推) ×8 无確定製造框 → 灰(材料不足) → skip craft(8>dedup hold-cap5, 让丢点重试先落)")
                self._goto("exit")
                return action_wait(300, "craft 开始制造 greyed (材料不足) → skip")
            sx = max(0.0, max_btn.cx - 0.056)
            sy = min(1.0, max_btn.cy + 0.10)
            self.log(f"开始制造漏检 → MAX 外推点击 ({sx:.3f},{sy:.3f})")
            self._start_taps += 1
            self._start_clicked = True
            self.mark("start_click")     # after-ack 窗口起点(见下方 fall-through 闸)
            return action_click(sx, sy, "start craft (MAX 外推开始制造)")

        # Not in dialog → open it via 快速制造.
        quick = self.find_cls(screen, UC.CRAFT_QUICK, conf=_CLS_CONF)
        if quick is not None and self._phase_ticks <= _START_MAX:
            self.log("opening 快速制造 dialog")
            self._quick_settle = True
            self.mark("quick_settle")
            return action_click_box(quick, "open quick-craft")
        # 刚点过快速制造 → 等 dialog 渲染再判 (防点后下一 tick dialog 没好就误判
        # nothing startable 立即 exit, live 2026-06-06 t0011→t0012 就是这样挂的)。
        if self._quick_settle:
            _w = self.since("quick_settle")
            if _w < _QUICK_SETTLE_SEC:
                return action_wait(350, f"quick-craft dialog settle ({_w:.1f}s)")
            self._quick_settle = False

        # ⛔after-ack(2026-07-28 帧实测补): 点过 开始制造 之后、確定製造N次 确认框
        # 还没渲染出来的那几帧, 屏上**一个 craft 锚点都不剩** —— 实测确认框帧
        # (data/craft_probe_20260728) 全帧只有 8 框: 取消/确认/返回/回大厅/弹窗叉叉
        # + 顶栏三币, MAX/开始制造/快速制造 **三条全落空** ⇒ 直接掉到下面
        # "nothing startable → exit", 而 _exit 的第一件事就是点 弹窗叉叉 关掉
        # 那个还没確認的框 ⇒ **制造点了等于没点**。
        # 这是 2026-07-28「判定带宽窄不一致 / 动画帧契约必破」那一族的第四例:
        # 差别只在这里破的不是判定带而是"整套锚点同时消失"。修法同族: 动作发出后
        # 必须先等**帧证据**, 不许在证据窗口内下"没活干"的结论。
        # ⚠只挡这条 fall-through: 若 开始制造 仍在屏上(灰=材料不足), 上面的
        #   _start_taps 重试路径先接走, 不会被这段拖慢。
        if self._start_clicked and not self._craft_confirm_clicked:
            _w = self.since("start_click")
            if _w < _CONFIRM_WAIT_SEC:
                return action_wait(300,
                                   f"点过 开始制造 — 等 確定製造N次 确认框 "
                                   f"({_w:.1f}/{_CONFIRM_WAIT_SEC:.1f}s)")
            # 窗口过了还没框 ⇒ 那一下确实没落地(被吞/灰按钮) → 允许重试
            self._start_clicked = False

        # Patience window (live 2026-06-12): right after a collect the screen
        # is mid-transition / covered by the incoming reward popup — 快速制造
        # is invisible for a few ticks. Verdict only after a real window.
        if self._phase_ticks < 6:
            return action_wait(450, "start: no 快速制造 yet — settling before verdict")
        # Nothing startable (slots busy / no free slot) or budget out → exit.
        self.log("nothing startable (busy slots / YOLO gap) → exit")
        self._goto("exit")
        return action_wait(300, "no startable craft → exit")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if self.detect_screen_yolo(screen) == "Lobby":
            self.log(f"done (claims={self._claims}, started={self._started})")
            self._mark_visited()
            return action_done(f"craft complete (claims={self._claims}, started={self._started})")
        if self._phase_ticks > _EXIT_MAX:
            self._mark_visited()
            return action_done("craft exit timeout")
        # Close any leftover dialog first, then home/back.
        close = self.find_cls(screen, UC.BTN_CLOSE_X, conf=_CLS_CONF, region=(0.55, 0.08, 0.97, 0.30))
        if close is not None:
            return action_click_box(close, "close leftover craft dialog")
        home = self.find_cls(screen, UC.BTN_HOME, conf=_CLS_CONF)
        if home is not None:
            return action_click_box(home, "craft exit: home button")
        back = self.find_cls(screen, UC.BTN_BACK, conf=_CLS_CONF)
        if back is not None:
            return action_click_box(back, "craft exit: back button")
        return self.nav_home(screen, "craft exit")
