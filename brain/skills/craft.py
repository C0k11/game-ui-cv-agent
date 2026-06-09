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

from typing import Any, Dict, Optional

from brain.skills.base import (
    BaseSkill, ScreenState, YoloBox,
    action_click, action_click_box, action_wait, action_back, action_done,
)
from brain.skills import ui_classes as UC

_CLS_CONF = 0.30
# Center-bottom band where craft confirm / 立即完成 dialogs put 确认键/取消键
# (probe: y≈0.70 for both the start-confirm and the券-rush dialog).
_DIALOG_BAND = (0.28, 0.60, 0.72, 0.82)

_ENTER_MAX = 20
_COLLECT_MAX = 16
_START_MAX = 18
_EXIT_MAX = 14


class CraftSkill(BaseSkill):
    def should_run(self, screen: ScreenState) -> bool:
        # Craft is NOT dot-gated — always enter to collect finished + start new
        # (user spec: 制造不需要靠黄点识别). Self-limiting: busy slots can't
        # restart, so re-runs the same day spend nothing.
        return True

    def __init__(self):
        super().__init__("Craft")
        self.max_ticks = 90
        self._init_state()

    def _init_state(self) -> None:
        self._phase_ticks: int = 0
        self._entered: bool = False
        self._collect_done: bool = False
        self._collect_settle: int = 0
        self._maxed_clicks: int = 0
        self._quick_settle: int = 0
        self._started: bool = False
        self._start_clicked: bool = False  # we pressed 开始制造 → confirm is ours
        self._claims: int = 0

    def reset(self) -> None:
        super().reset()
        self._init_state()

    def _goto(self, sub_state: str) -> None:
        self.sub_state = sub_state
        self._phase_ticks = 0

    # ── helpers ──────────────────────────────────────────────────────────
    def _is_craft(self, screen: ScreenState) -> bool:
        page = self.detect_screen_yolo(screen)
        if page == "Lobby":
            self._entered = False
            return False
        if page == "Craft":
            return True
        # Direct craft markers (快速制造 / 开始制造 are craft-only).
        if self.find_cls(screen, [UC.CRAFT_QUICK, UC.CRAFT_START], conf=_CLS_CONF) is not None:
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
        return action_back("craft: recover toward lobby")

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

        if self._collect_settle > 0:
            self._collect_settle -= 1
            return action_wait(400, f"collect settle ({self._collect_settle})")

        if self._collect_done:
            self._goto("start")
            return action_wait(250, "collect done → start")

        # 一次领取黄色: finished crafts collect FREE; running ones pop the券
        # dialog (caught above). Tap it, then settle to see which happens.
        yellow = self.find_cls(screen, UC.CLAIM_ONCE_YELLOW, conf=_CLS_CONF)
        if yellow is not None:
            self.log("tapping 一次领取黄色 (collect finished crafts)")
            self._collect_settle = 2
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
            if self._start_clicked:
                self.log("confirming craft start (確定製造N次, 耗信用点 — safe)")
                self._started = True
                self._goto("exit")
                return action_click_box(dlg, "confirm craft start (credits)")
            self.log("⛔ unexpected confirm dialog in start state (开始制造 not clicked) → cancel (券-safe)")
            cancel = self.find_cls(screen, UC.BTN_CANCEL, conf=_CLS_CONF)
            if cancel is not None:
                return action_click_box(cancel, "cancel unexpected dialog (券-safe)")
            return action_back("dismiss unexpected dialog (券-safe)")

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
            self.log("clicking 开始制造")
            self._start_clicked = True
            return action_click_box(start_btn, "start craft (开始制造)")
        # 兜底: 开始制造(idx444) 是 v6b 漏检的 missing cls (probe 旧模型 0.93,
        # v6b 退步漏检 → craft 卡死, live 2026-06-06)。已点过 MAX (= 在 dialog 里)
        # 且 MAX/MAX灰 检出 → 用它外推开始制造位置 (probe: MAX(0.926,0.713) →
        # 开始制造(0.870,0.812), 偏移 cx-0.056/cy+0.10; dialog 布局固定, 归一化
        # 跨分辨率)。点中 → 弹「確定製造N次」确认框 → _confirm_dialog 收口(信用点,
        # 安全)。根治靠飞轮补 开始制造 样本 → v6c。
        if self._maxed_clicks > 0 and max_btn is not None:
            sx = max(0.0, max_btn.cx - 0.056)
            sy = min(1.0, max_btn.cy + 0.10)
            self.log(f"开始制造漏检 → MAX 外推点击 ({sx:.3f},{sy:.3f})")
            self._start_clicked = True
            return action_click(sx, sy, "start craft (MAX 外推开始制造)")

        # Not in dialog → open it via 快速制造.
        quick = self.find_cls(screen, UC.CRAFT_QUICK, conf=_CLS_CONF)
        if quick is not None and self._phase_ticks <= _START_MAX:
            self.log("opening 快速制造 dialog")
            self._quick_settle = 3
            return action_click_box(quick, "open quick-craft")
        # 刚点过快速制造 → 等 dialog 渲染再判 (防点后下一 tick dialog 没好就误判
        # nothing startable 立即 exit, live 2026-06-06 t0011→t0012 就是这样挂的)。
        if self._quick_settle > 0:
            self._quick_settle -= 1
            return action_wait(350, f"quick-craft dialog settle ({self._quick_settle})")

        # Nothing startable (slots busy / no free slot) or budget out → exit.
        self.log("nothing startable (busy slots / YOLO gap) → exit")
        self._goto("exit")
        return action_wait(300, "no startable craft → exit")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if self.detect_screen_yolo(screen) == "Lobby":
            self.log(f"done (claims={self._claims}, started={self._started})")
            return action_done(f"craft complete (claims={self._claims}, started={self._started})")
        if self._phase_ticks > _EXIT_MAX:
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
        return action_back("craft exit: ESC toward lobby")
