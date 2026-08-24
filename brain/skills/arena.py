"""ArenaSkill — 战术大赛 (PVP) daily routine (pure-YOLO rewrite).

Verified flow (interactive probe 2026-06-01, data/_missions_probe_log.md
Step 23-34). Arena is PVP — NOT a sweep. Key probe findings vs the old skill:
- Battles are FULLY AUTO-resolved (a few seconds). NO 跳过战斗 toggle needed —
  just 出击 then poll-dismiss the result dialog(s).
- Winning + setting a new season-best pops an EXTRA 達成賽季最高紀錄 popup that
  AWARDS pyroxene (a GAIN — dismiss it, never confuse with a cost).
- ~25s 等待時間 cooldown after each fight (blind tick wait here; OCR mm:ss is
  unreliable — v6 could refine).

 pyroxene protection
  digit-OCR 战术大赛票 X/5. Ticket 0  STOP challenging (a 0-ticket 出击 pops a
  購買戰術大賽券 青辉石 dialog). The buy-dialog guard requires a 取消键 present
  (a cost dialog has cancel) so it never misfires on the cancel-less
  達成賽季最高紀錄 REWARD popup.

State machine
----
enter   lobby  NAV_TASKS  hub  HUB_ARENA  arena main.
claim   click every 领取奖励_黄 (获得奖励  点击继续字样 dismiss). 领取奖励_灰
        ignored. Done when no 领取奖励_黄 remains.
fight_check  digit-OCR ticket X/5. 0 / cap  exit. Cooldown wait between fights.
select  click TOP 战术大赛对战选择区域 (cls92) row.
fight   對戰對象  攻击编制  编队屏 出击  auto-battle  poll-dismiss all 确认键.
exit    返回键 / 回大厅  lobby (or hub)  done.

Detectors: base "ui" + "battle" (SKILL_YOLO_MAP).
"""
from __future__ import annotations

import time

from typing import Any, Dict, Optional

from brain.skills.base import (
    BaseSkill, ScreenState,
    action_click, action_click_box, action_wait, action_back, action_done,
)
from brain.skills import ui_classes as UC

_CLS_CONF = 0.30
# A 青辉石 icon in this body band = buy dialog (NOT top-bar balance at cy<0.10).
# Deep-dive C4 (2026-06-09): aligned to schedule's LIVE-VERIFIED region — the
# buy-dialog pyroxene icon sits at cy≈0.577 (> the old 0.48 upper bound, which
# would have MISSED it = bought a ticket).
_PYROXENE_BODY_REGION = (0.20, 0.12, 0.82, 0.64)
# Centered result 确认键 band (戰鬥結果 / 達成賽季最高紀錄).
_RESULT_BAND = (0.32, 0.55, 0.68, 0.85)

_MAX_FIGHTS = 5            # daily arena ticket cap (must complete all 5)
# ~25s 等待時間 between fights. NOW countdown-aware (user 2026-06-13: 倒计时早没了
# bot还傻等22tick≈30s — the game countdown STARTS when the battle ends, well before
# the bot finishes dismissing results and begins counting, so a full blind 22-tick
# wait massively over-shoots). _read_cooldown OCRs the mm:ss 等待時間 value; we go
# the moment it reads --:-- (ready). _COOLDOWN_TICKS stays as the hard fallback cap;
# _COOLDOWN_MIN is a short floor so a one-frame unreadable '--:--' can't false-ready
# into a greyed opponent row (which _select recovers from anyway via extra cooldown).
_COOLDOWN_TICKS = 22
_COOLDOWN_MIN = 4
# 等待時間 value region (fixed left panel; calibrated 2026-06-13 on a live arena
# frame: '00:13'  digit-OCR '013'). --:-- (ready)  no digits  None.
_COOLDOWN_REGION = (0.12, 0.71, 0.27, 0.77)

_ENTER_MAX = 24
_CLAIM_MAX = 12
# tick-vs-墙钟家族(2026-07-28): _ENTER_MAX 24 tick 被 zero-wait 压到
# 3.6-6.0s 盖不住两次页面加载( 'arena never reached' 假成功, 5 票作废);
# _CLAIM_MAX 12 tick=1.8-3.0s 比 _CLAIM_SETTLE_SEC=2.5 还短 —— 那个墙钟
# 修复被它顶掉, 每日獎勵 漏领复发。改「帧数 AND 墙钟」合取(×1.6 等效):
_ENTER_MAX_SEC = 38.4
_CLAIM_MAX_SEC = 19.2
# 领完一个黄钮后 toast 动画期间下一个检不出 —— 用墙钟等, 别数 tick
# (2026-07-25: 旧的 "2 tick" ≈0.24s 盖不住动画, 每日獎勵 直接漏领)。
_CLAIM_SETTLE_SEC = 2.5
# 点了黄钮之后到帧证据(toast/按钮消失)前不重发 — reason 含 claim 命中关键词
# 豁免会跳过稳定门+hold, 旧码 ~0.2s 一发把 6 次上限虚耗光(2026-07-28)。
_CLAIM_RETRY_SEC = 6.0
_EXIT_MAX = 16


class ArenaSkill(BaseSkill):
    def should_run(self, screen: ScreenState) -> bool:
        # Always enter (user iron rule 2026-06-11): real signal = the 战术大赛
        # tile's own dot inside the hall (hall scan in _enter), never the lobby
        # entry dot.
        return True

    def __init__(self):
        super().__init__("Arena")
        self.max_ticks = 320
        self._init_state()

    def _init_state(self) -> None:
        self._phase_ticks: int = 0
        self._enter_ticks: int = 0
        self._claim_clicks: int = 0
        self._claim_pending: bool = False   # 上一 tick 发了 claim 点击, 等落账
        self._claim_fired: bool = False     # 发过 claim(重发用墙钟 gate)
        self._claim_t0: float = 0.0         # 无新黄钮的墙钟起点(0=未计时)
        self._fights_done: int = 0
        self._cooldown: int = 0
        self._fight_stage: int = 0          # 0=對戰對象 1=编队 2=battle
        self._fight_ticks: int = 0
        self._stage_settle: int = 0         # blind wait after each click (page transition)
        self._enter_settle: int = 0         # blind wait after a nav click in enter
        self._select_attempts: int = 0
        self._select_rounds: int = 0        # extra-cooldown retries when select spins
        self._result_pending: bool = False
        self._tickets: Optional[int] = None
        # 2026-07-26: "到底进没进过战术大赛页"。_exit 的 Lobby 分支原来完全
        # 不看这个, 把"从别的页逃回大厅"也报成 arena complete(见 _exit 注释)。
        self._reached: bool = False
        self._ticket_misses: int = 0        # consecutive failed ticket reads

    def reset(self) -> None:
        super().reset()
        self._init_state()

    def exit_report(self):
        """战术大赛的竣工判据 = 票打光了没。

        为什么必须有: 2026-07-25 晚 Arena 报了 `done (0 fights, 0 rewards)`,
        而当天票 5/5 满 + 2 个未领奖励(重跑后 5 场全打完, 排名 3732)。
        它是 7 次收工里 5 个 "UNKNOWN — 未声明竣工判据" 之一 —— **没有判据
        就没人审计出口**, 假成功只能靠用户肉眼发现([[completion-gap]])。
        `self._tickets`(arena.py:100/424) 是离开 fight_check 前最后一次成功
        读数: 正常收工路径读到 0 才 exit  CLEAN 成立; select 失败那条路留下
        陈旧非零值  正确地报 LEFTOVER。
        """
        if not self._reached:
            return ("UNKNOWN",
                    f"从未进到战术大赛页(enter 走了 {self._enter_ticks} tick) "
                    f"— 票/奖励状态**完全未知**")
        if self._tickets is None:
            return ("UNKNOWN",
                    f"打了 {self._fights_done} 场、领了 {self._claim_clicks} 个"
                    f"奖励, 但票数从未读出 — 不知道打光没")
        if self._tickets > 0:
            return ("LEFTOVER",
                    f"还剩 {self._tickets} 张票没打(已打 {self._fights_done} 场, "
                    f"领 {self._claim_clicks} 个奖励)")
        return ("CLEAN",
                f"票 0, 打了 {self._fights_done} 场, "
                f"领了 {self._claim_clicks} 个奖励")

    def _goto(self, sub_state: str) -> None:
        self.sub_state = sub_state
        self._phase_ticks = 0
        self.mark("phase")                 # _phase_ticks 的墙钟版

    #  helpers
    def _on_arena(self, screen: ScreenState) -> bool:
        return self.find_cls(
            screen, [UC.TICKET_ARENA, UC.ATTACK_FORMATION, UC.SORTIE,
                     UC.ARENA_OPPONENT_ROW], conf=_CLS_CONF
        ) is not None

    def _read_tickets(self, screen: ScreenState) -> Optional[int]:
        """digit-OCR 持有票券 X/5 next to the arena ticket icon (LEFT panel,
        ~0.055,0.678 — NOT the top bar).  money defense #1 (0  stop, never the
        buy-ticket-pyroxene trap).

         Runs on a CLEAN ADB frame, not screen.frame: the overlay burns a
        tight box+label onto the small ticket icon in every DXcam frame, which
        killed ui_v7's detection of it outright (live 2026-06-09: 0 detections
        on burned frames vs conf 0.95 on the clean frame  every read None
        fail-closed exit with tickets unspent). Falls back to screen.frame
        only if no clean source is registered."""
        try:
            from brain.pipeline import (get_clean_frame, _run_yolo_on_image,
                                        run_digit_ocr, parse_count)
        except Exception:
            return None
        frame = get_clean_frame()
        icon = None
        if frame is not None:
            h, w = frame.shape[:2]
            cands = [b for b in _run_yolo_on_image(frame, w, h, context="ui+battle")
                     if b.cls_name == UC.TICKET_ARENA and b.confidence >= _CLS_CONF
                     and b.cx <= 0.22 and 0.58 <= b.cy <= 0.78]
            icon = max(cands, key=lambda b: b.confidence) if cands else None
        else:
            frame = screen.frame
            if frame is None:
                return None
            icon = self.find_cls(screen, UC.TICKET_ARENA, conf=_CLS_CONF,
                                 region=(0.0, 0.58, 0.22, 0.78))
        if icon is None:
            return None
        #  Strip must SKIP the fixed "持有票券" label (x≈0.068-0.118): feeding
        # the digit-OCR Chinese text made it return None on EVERY frame (live
        # 2026-06-09: fight 1 ran with zero successful reads). Digits "X/5" sit
        # at x≈0.131-0.145; icon.x2≈0.066  offset +0.05, span 0.08 verified
        # offline on the live frame ('4/5'  (4,5)) with margin both sides.
        # 2026-07-27 单位换算成图标宽度(战术大赛票 iw 实测 0.0191, n=3085,
        # 四分位 0.0190/0.0193): +0.0502.62iw, span 0.084.19iw  右界 6.81iw。
        # 16:9 下几何完全等价, 换分辨率/窗口大小不再失准(理由见 icon_strip)。
        # strip 高 ±0.8bh: ±0.4bh 是临界高度, icon 检测框轻微抖动就把
        # 分母/整串裁没(2026-07-17 live 12 连 None 实锤; ±0.8bh 变体
        # 同帧完整读出 '3/5', 上邻活动框/下邻重新开始键都够不着)
        #  这条教训 2026-07-27 才传导到 ticket_sweep / 顶栏, 中间躺了 10 天。
        from brain.pipeline import icon_strip
        raw = run_digit_ocr(frame, icon_strip(icon, 2.62, 6.81, 0.80))
        res = parse_count(raw)
        if res is None or res[0] is None:
            return None
        # Strict numerator proof: a bare number with no '/' could be the
        # DENOMINATOR of a left-clipped "X/5" — the 2026-06-02 incident class
        # ("0/5" clipped to '5' would read 0 tickets as 5  出击 at 0  buy
        # dialog). Right-clipped '3/' (denominator lost) is fine — numerator
        # is provably the leading digit.
        s = (raw or "").strip()
        if "/" not in s or not s[:1].isdigit():
            return None
        return res[0]

    def _read_cooldown(self, screen: ScreenState) -> Optional[int]:
        """等待時間 mm:ss countdown remaining seconds, or None if --:-- (ready)
        / unreadable. Clean ADB frame preferred. User 2026-06-13: align with the
        real countdown instead of a blind 22-tick wait."""
        try:
            from brain.pipeline import get_clean_frame, run_digit_ocr
        except Exception:
            return None
        frame = get_clean_frame()
        if frame is None:
            frame = getattr(screen, "frame", None)
        if frame is None:
            return None
        raw = run_digit_ocr(frame, _COOLDOWN_REGION)
        digits = "".join(c for c in (raw or "") if c.isdigit())
        if not digits:
            return None   # --:-- (ready) or unreadable
        try:
            # '013'0:13=13s, '0024'0:24=24s, '13'13s (last 2 = seconds)
            return int(digits[:-2] or 0) * 60 + int(digits[-2:]) if len(digits) > 2 else int(digits)
        except ValueError:
            return None

    def _buy_dialog(self, screen: ScreenState) -> bool:
        """买票花费框 = 结构上"要你选数量并付费"的框。

        2026-07-25 全仓金钱审计: 旧版是 `body青辉石 AND 取消键` —— 一条
        **合取的单点链**, 任一环漏检整条防线哑火。schedule 那起 30 青辉石事故
        的帧就是反例: 屏上明写「單價30」而 YOLO body **零青辉石检出**。
        arena 这里是逐字同型, 只是还没轮到它出事。
        改成**析取的正交多信号**(任一命中即判买票框):
          A 数量步进器在 body(has_qty_stepper, 与图标识别完全独立)
          B body 里有青辉石(原判据, 保留)
        取消键 不再当必要条件 —— 它漏检时正是最危险的时刻(下面 result-dismiss
        分支唯一的拦阻就是"取消键还在", 那个单点信号一旦丢, 购买框的確認
        就落在 _RESULT_BAND 里被当成战斗结算点掉)。
        conf 0.20 = 模型下限: 危险检测器要尽可能灵敏, 误报代价只是取消+退出。
        """
        if self.has_qty_stepper(screen):
            return True
# 2026-07-25 全量 cls 审计删除: 原来这里还并了一路 "清辉石"(master idx2),
        # 注释写着"危险检测器多收一路零成本" —— 实测**训练 0 框 / 92k tick 实战
        # 0 检出**, 那一路从来没收到过任何东西, 只是制造"有两路信号"的假象。
        # idx2 是 idx30「青辉石」的错别字重复类(BA 官方写作 青輝石), 本就不该
        # 被标注。真正有效的正交第二路是**结构信号** has_qty_stepper。
        return self.find_cls(screen, UC.TOPBAR_PYROXENE, conf=0.20,
                             region=_PYROXENE_BODY_REGION) is not None

    #  tick
    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1
        self._phase_ticks += 1

        if self.ticks >= self.max_ticks:
            self.log(f"timeout ({self._fights_done} fights)")
            return action_done("arena timeout")

        #  buy-ticket dialog (青辉石 cost + 取消键)  cancel + exit. Primary
        # safety is the ticket gate; this is the backstop.
        if self.sub_state in ("fight", "select", "fight_check") and self._buy_dialog(screen):
            self.log(" ticket-purchase dialog (青辉石) — cancel, never buy")
            self._goto("exit")
            cancel = self.find_cls(screen, UC.BTN_CANCEL, conf=_CLS_CONF)
            if cancel is not None:
                return action_click_box(cancel, "cancel ticket purchase")
            return action_back("dismiss buy dialog")

        # Battle-result dialogs (戰鬥結果 WIN/LOSE + 達成賽季最高紀錄)  dismiss
        # ALL centered 确认键. Can land over the arena main, so handled here for
        # the in-fight states. _result_pending dedups one fight per battle.
        if self.sub_state in ("fight", "select", "fight_check"):
            res_confirm = self.find_cls(screen, UC.BTN_CONFIRM, conf=_CLS_CONF, region=_RESULT_BAND)
            res_marker = self.find_cls(screen, [UC.BATTLE_WIN, UC.GOT_REWARD], conf=0.35)
            if res_confirm is not None or res_marker is not None:
                #  Never click a centered 确认键 while a 取消键 is also visible.
                # Result popups are cancel-less; confirm+cancel together = a
                # cost dialog mid-render racing past the _buy_dialog guard
                # (deep-dive C2: one missing component on an animation frame
                # defeats the conjunctive guard, and this block would then
                # click the BUY button). Wait a frame — the fully-rendered
                # dialog is caught by the guard above next tick.
                if self.find_cls(screen, UC.BTN_CANCEL, conf=0.20) is not None:
                    return action_wait(400, "confirm+cancel both visible — not a result dialog, re-read")
                # Count a fight ONLY if 出击 actually launched (stage>=2). A
                # centered 确认键 also appears on NON-result notices — live
                # 2026-06-09: 通知「已超過清單更新時間」(opponent-list refresh
                # expired) at fight stage 1 was counted as fight 1  cap would
                # end arena one real fight early with a ticket unspent.
                if not self._result_pending and self.sub_state == "fight" \
                        and self._fight_stage >= 2:
                    self._fights_done += 1
                    self._result_pending = True
                    self.log(f"fight {self._fights_done} result  dismiss")
                self._fight_stage = 0
                self._fight_ticks = 0
                self._cooldown = 0
                self._goto("fight_check")
                if res_confirm is not None:
                    return action_click_box(res_confirm, "dismiss battle result (确认键)")
                cont = self.find_cls(screen, UC.STORY_TAP_CONTINUE, conf=_CLS_CONF)
                if cont is not None:
                    return action_click_box(cont, "dismiss result (continue)")
                return action_wait(300, "result settling")
            else:
                self._result_pending = False

        if screen.is_loading():
            return action_wait(700, "arena loading")

        if self.sub_state == "":
            self._goto("enter")

        handler = {
            "enter": self._enter,
            "claim": self._claim,
            "fight_check": self._fight_check,
            "select": self._select,
            "fight": self._fight,
            "exit": self._exit,
        }.get(self.sub_state)
        if handler is None:
            return action_wait(300, "arena unknown state")
        return handler(screen)

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        self._enter_ticks += 1
        if self._on_arena(screen):
            self.log("inside arena  claim")
            self._reached = True          # 唯一置位点: 正锚确认真的到了
            self._goto("claim")
            return action_wait(400, "entered arena")

        #  Settle after any nav click — during the page transition the OLD page
        # (and its cls boxes) linger at low conf for a frame or two; re-clicking
        # the same spot then lands on the NEW page's UI. Live 2026-06-09: the
        # tick-2 "arena tile" re-click hit the freshly-loaded arena main and
        # opened the 對戰對象 popup  select spun on covered rows  false exit.
        if self._enter_settle > 0:
            self._enter_settle -= 1
            return action_wait(600, f"enter transition ({self._enter_settle} left)")

        page = self.detect_screen_yolo(screen)
        if page == "Lobby":
            act = self.click_cls(screen, UC.NAV_TASKS, "open campaign hub", conf=_CLS_CONF)
            if act is not None:
                self._enter_settle = 3
                return act
            return action_wait(400, "lobby: NAV_TASKS not seen")
        if page == "Mission":
            #  Hall scan (user iron rule 2026-06-11): the 战术大赛 tile's own
            # red/yellow dot is the work signal — visible only here. No dot
            # nothing to claim/fight today  graceful exit.
            has_work = self.hall_tile_dot(screen, UC.HUB_ARENA)
            if has_work is False:
                self.log("hall scan: 战术大赛 无红黄点  no work today, done")
                return action_done("arena no work (hall scan)")
            act = self.click_cls(screen, UC.HUB_ARENA, "click arena tile", conf=_CLS_CONF)
            if act is not None:
                self._enter_settle = 4
                return act
            return action_wait(450, "hub: arena tile not seen (transition)")

        if self._enter_ticks > _ENTER_MAX and self.since("enter_wall") > _ENTER_MAX_SEC:
            self.log("can't reach arena, exiting")
            self._goto("exit")
            return action_wait(300, "enter timeout")
        if page is not None:
            return action_back(f"back from {page}")
        return action_wait(450, "entering arena")

    def _claim(self, screen: ScreenState) -> Dict[str, Any]:
        # Reward reveal popup  dismiss via continue / header (NEVER center).
        cont = self.find_cls(screen, UC.STORY_TAP_CONTINUE, conf=_CLS_CONF)
        if cont is not None:
            return action_click_box(cont, "dismiss reward (continue)")
        got = self.find_cls(screen, UC.GOT_REWARD, conf=_CLS_CONF)
        if got is not None:
            return action_click_box(got, "dismiss reward (header)")

        # after-ack 落账(2026-07-25): 旧码在 return action_click_box 之前就
        # _claim_clicks += 1 —— 点击被稳定门吞时计数照加, 上限被虚耗, 真黄钮
        # 还没领就判 "claim done"。改成上一 tick 的点击**确认没被吞**才计数。
        if getattr(self, "_claim_pending", False):
            if self.action_suppressed:
                self.log("claim 点击被吞 — 计数不落账, 重试")
            else:
                self._claim_clicks += 1
                self.log(f"claim arena reward #{self._claim_clicks} 已落账")
            self._claim_pending = False

        if self._claim_clicks >= 6 or (
                self._phase_ticks > _CLAIM_MAX
                and self.since("phase") > _CLAIM_MAX_SEC):
            self.log(f"claim done ({self._claim_clicks})")
            self._goto("fight_check")
            return action_wait(250, "claim done  fight_check")

        # Click any active 领取奖励_黄 (灰 = already claimed, ignore).
        claim = self.find_cls(screen, [UC.CLAIM_REWARD_YELLOW, UC.CLAIM_YELLOW], conf=_CLS_CONF)
        if claim is not None:
            # after-ack(2026-07-28): reason 含 claim  关键词豁免跳过稳定门+
            # hold, 旧码在 toast 渲染出来前 ~0.2s 一发连点同一个黄钮, 每发都
            # 落账把 6 次上限吃光  真没领的黄钮被 "claim done" 抛下。
            # 点完等帧证据(toast 在顶部分支处理/黄钮消失), 期间不重发;
            # _force_settle 让它老实走稳定门(别再吃豁免)。
            if getattr(self, "_claim_fired", False) and self.since("claim_fire") < _CLAIM_RETRY_SEC:
                return action_wait(300, "claim 已发 — 等 toast/黄钮消失的帧证据")
            self._claim_pending = True
            self._claim_t0 = 0.0
            self._claim_fired = True
            self.mark("claim_fire")
            self.log(f"claim arena reward ({claim.cls_name})")
            act = action_click_box(claim, "claim arena reward")
            act["_force_settle"] = True
            return act

        # 墙钟而非 tick(2026-07-25 live 实锤: 每日獎勵 漏领):
        # 领完一个后 toast 动画期间下一个黄钮检不出, 旧码等 "2 tick" —— 而非
        # loading 的 action_wait 被 server 压到 0.12s, 2 tick ≈ 0.24s, 盖不住
        # 动画  直接判 "no 领取奖励_黄" 去打架, 每日獎勵 那个黄钮就丢了
        # (2026-07-09 注释记的同款事故, 当时的修法 2 tick 标定失准)。
        if not getattr(self, "_claim_t0", 0.0):
            self._claim_t0 = self.clock()
        _el = self.clock() - self._claim_t0
        if _el < _CLAIM_SETTLE_SEC:
            return action_wait(600, f"claim settle re-check ({_el:.1f}s/{_CLAIM_SETTLE_SEC}s)")
        self.log(f"no 领取奖励_黄 ({_el:.1f}s 无新黄钮)  fight_check")
        self._goto("fight_check")
        return action_wait(250, "no active rewards  fight_check")

    def _fight_check(self, screen: ScreenState) -> Dict[str, Any]:
        #  Hard ticket gate (money-safety): NO path may leave fight_check
        # toward select/fight without a SUCCESSFUL ticket read this phase.
        # Deep-dive C1 + live 2026-06-09: the old ">8 ticks  select anyway"
        # fallback let the entire first fight run with _tickets=None (the OCR
        # strip was mis-geometried and EVERY read failed silently).
        tickets = self._read_tickets(screen)
        if tickets is None:
            self._ticket_misses += 1
            page = self.detect_screen_yolo(screen)
            if page in ("Lobby", "Mission"):
                self.log(f"drifted to {page}  arena over")
                self._goto("exit")
                return action_wait(300, "drifted out  exit")
            if self._ticket_misses > 12:
                self.log("tickets unreadable after retries  exit (money fail-closed)")
                self._goto("exit")
                return action_wait(300, "ticket unreadable  exit")
            # 2026-07-09 live: 第1场结算后停在过渡页(Rank变动/奖励toast/TOUCH),
            # 票icon不在屏上  干等12次必败(4/5票没打+奖励没领全)。retry 时
            # 主动清过渡元素。负门禁: 確認+取消同屏=可能是购买框, 绝不点確認。
            got = self.find_cls(screen, UC.GOT_REWARD, conf=0.5)
            if got is not None:
                return action_click(0.5, 0.90, "dismiss reward (ticket retry)")
            cont = self.find_cls(screen, UC.STORY_TAP_CONTINUE, conf=0.5)
            if cont is not None:
                return action_click_box(cont, "tap continue (ticket retry)")
            conf_btn = self.find_cls(screen, UC.BTN_CONFIRM, conf=0.6)
            cancel_btn = self.find_cls(screen, UC.BTN_CANCEL, conf=0.5)
            if conf_btn is not None and cancel_btn is None:
                return action_click_box(conf_btn, "dismiss result dialog (ticket retry)")
            return action_wait(400, f"ticket read retry {self._ticket_misses}/12 (fail-closed gate)")
        self._ticket_misses = 0
        self._tickets = tickets
        if tickets <= 0:
            self.log("tickets 0/5  arena done")
            self._goto("exit")
            return action_wait(300, "0 tickets  exit")

        # Safety cap.
        if self._fights_done >= _MAX_FIGHTS:
            self.log(f"fight cap reached ({self._fights_done})")
            self._goto("exit")
            return action_wait(300, "fight cap  exit")

        # Cooldown between fights — countdown-aware (user 2026-06-13: 倒计时早没了
        # 别傻等). Read the 等待時間 mm:ss: go the moment it reads --:-- (ready),
        # only wait while it actually shows time left. Hard cap + short floor guard.
        if self._fights_done > 0:
            self._cooldown += 1
            secs = self._read_cooldown(screen)
            if secs is not None and secs > 1 and self._cooldown < _COOLDOWN_TICKS:
                return action_wait(1000, f"arena 等待時間 {secs}s 倒计时 "
                                          f"(cd {self._cooldown}/{_COOLDOWN_TICKS})")
            if secs is None and self._cooldown < _COOLDOWN_MIN:
                # --:-- unreadable this early  don't trust "ready" yet (avoid a
                # greyed-row select); short floor then proceed.
                return action_wait(1000, f"arena cooldown floor "
                                          f"{self._cooldown}/{_COOLDOWN_MIN}")
            self._cooldown = 0   # countdown cleared / floor passed / hard cap  go

        # A successful ticket read  the arena left panel is on screen — safe
        # to select. (The old _on_arena/blind-select fallbacks are gone; they
        # were the leak.)
        self._goto("select")
        return action_wait(250, "arena main (tickets read)  select")

    def _select(self, screen: ScreenState) -> Dict[str, Any]:
        # 對戰對象 popup may ALREADY be open (stray click / re-entry) — its body
        # covers the opponent rows, so spinning for cls92 here would falsely
        # exit (live 2026-06-09). Hand over to fight stage 0, which clicks the
        # visible 攻击编制.
        if self.find_cls(screen, UC.ATTACK_FORMATION, conf=_CLS_CONF) is not None:
            self.log("對戰對象 already open  fight stage0")
            self._fight_stage = 0
            self._fight_ticks = 0
            self._stage_settle = 0
            self._goto("fight")
            return action_wait(250, "popup already open  fight")

        self._select_attempts += 1
        if self._select_attempts > 8:
            # Between fights a spin usually means the cooldown bar hadn't fully
            # cleared (greyed opponent row). Re-enter cooldown for one more round
            # rather than falsely ending arena early (we must finish all 5).
            if self._fights_done < _MAX_FIGHTS and self._select_rounds < 2:
                self._select_rounds += 1
                self._select_attempts = 0
                self._cooldown = 0
                self.log(f"select spun  extra cooldown (round {self._select_rounds})")
                self._goto("fight_check")
                return action_wait(400, "select spun  re-cooldown")
            self.log("opponent select failed  exit")
            self._goto("exit")
            return action_wait(300, "select failed  exit")

        # cls92 opponent rows (right panel). Click the TOP (lowest cy).
        rows = [b for b in self.find_all_cls(screen, UC.ARENA_OPPONENT_ROW, conf=0.25)
                if b.cx > 0.5]
        if not rows:
            return action_wait(400, "waiting for opponent rows (cls92)")
        top = min(rows, key=lambda b: b.cy)
        self.log(f"select top opponent ({top.cx:.2f},{top.cy:.2f}) of {len(rows)}")
        self._fight_stage = 0
        self._fight_ticks = 0
        self._stage_settle = 3          # let 對戰對象 popup finish opening
        self._goto("fight")
        return action_click_box(top, "select top opponent")

    def _fight(self, screen: ScreenState) -> Dict[str, Any]:
        # 被吞回退(root 信号, 2026-07-22 tick26 实锤): 攻击编制/出击 click 被
        # 稳定门吞时 stage 已提前推进(mutate-before-ack)  编队页从没出击却
        # "auto-battle in progress" 干等到 fight timeout(两连实锤, 5 票 0 打)。
        # 必须在 settle 检查之前对账 — settle 的 wait 会把下一 tick 的
        # action_suppressed 刷成 False, 信号只活一个 tick。
        if self.action_suppressed and self._fight_stage in (1, 2):
            self._fight_stage -= 1
            self._stage_settle = 0
            self.log(f"fight click 被稳定门吞  stage 回退到 {self._fight_stage}")

        #  Blind settle after every click — give the page time to transition
        # before reading it. Without this we act on a stale/animating frame and
        # the click looks like it "did nothing" (点了没反应  误判重选/空点).
        if self._stage_settle > 0:
            self._stage_settle -= 1
            return action_wait(700, f"stage {self._fight_stage} 转场 ({self._stage_settle} left)")

        self._fight_ticks += 1
        # Battle (stage2) can run a while; nav stages are short.
        max_t = 60 if self._fight_stage >= 2 else 30
        if self._fight_ticks > max_t:
            self.log(f"fight stage {self._fight_stage} timeout")
            self._goto("fight_check")
            return action_back("fight timeout")

        # Stage 0: 對戰對象 popup  攻击编制. The popup needs real transition
        # time; be patient (don't re-select early and interrupt it opening).
        if self._fight_stage == 0:
            af = self.find_cls(screen, UC.ATTACK_FORMATION, conf=_CLS_CONF)
            if af is not None:
                self.log("對戰對象 open  click 攻击编制")
                self._fight_stage = 1
                self._select_attempts = 0
                self._select_rounds = 0     # select succeeded  reset retry budget
                self._stage_settle = 4      # 编队屏 loads characters  slowest screen
                return action_click_box(af, "click 攻击编制")
            if self._fight_ticks > 10 and self._on_arena(screen):
                self._goto("select")
                return action_wait(400, "對戰對象 didn't open after 10t  re-select")
            return action_wait(650, "waiting for 對戰對象 popup (transition)")

        # Stage 1: 编队屏  出击 (no skip toggle; arena auto-resolves).
        if self._fight_stage == 1:
            sortie = self.find_cls(screen, UC.SORTIE, conf=_CLS_CONF)
            if sortie is not None:
                self.log("click 出击 (auto-battle)")
                self._fight_stage = 2
                self._fight_ticks = 0       # time the battle itself, not the nav
                self._stage_settle = 2      # battle intro transition
                return action_click_box(sortie, "sortie (出击)")
            return action_wait(650, "waiting for 出击 (编队屏 loading)")

        # Stage 2: auto-battle. Result dialogs handled in tick(). If we're back
        # on a clean arena main (result auto-dismissed), count + continue.
        if self._fight_ticks > 4 and self.find_cls(screen, UC.TICKET_ARENA, conf=_CLS_CONF) is not None \
                and self.find_cls(screen, UC.BTN_CONFIRM, conf=_CLS_CONF, region=_RESULT_BAND) is None:
            self.log("back on arena main  fight complete")
            if not self._result_pending:
                self._fights_done += 1
            self._result_pending = False
            self._fight_stage = 0
            self._fight_ticks = 0
            self._cooldown = 0
            self._goto("fight_check")
            return action_wait(300, "fight done  fight_check")
        return action_wait(1000, "auto-battle in progress")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        page = self.detect_screen_yolo(screen)
        # 2026-07-25 live 实锤的假成功(run_20260725_231337):
        #   t0015 DailyRoutine done(收在**信用点商店**里, 因为 shop.chain_in_shop
        #         默认 True 要给 arena_shop 接力, 而 sub_only 路径下 arena_shop
        #         根本不在 plan 里 —— 见 daily_routine.__init__ 的修复)
        #   t0016..0039 Arena/enter 干等 24 tick(屏上 cls = 信用点商店_已选中/
        #         全部选择/回大厅按钮0.97, PAGE_SIGNATURES 没收录商店  page=None,
        #         上面那条 `if page is not None: action_back` 根本够不到)
        #   t0040 enter timeout  t0041 back key  回到大厅
        #   t0044 **`done (0 fights, 0 rewards)` + action_done("arena complete")**
        # 而当天真有活: 重跑后票 5/5 满 + 2 个未领奖励, 5 场全打完、排名 3732。
        #  这条分支**与打完 5 场的成功路径共用**, 从不问"到底进没进去过"。
        #   "从别的页逃回大厅" ≠ "干完了"。
        if page in ("Lobby", "Mission"):
            if not self._reached:
                self.log(f" 从未进到战术大赛页(enter {self._enter_ticks} tick) "
                         f" 报 timeout, **不是完成**")
                # reason 带 timeout  pipeline 走重试分支(_max_retries=1),
                # 从大厅重进一次; 再失败记 status=timeout 而不是 done。
                return action_done("arena never reached (enter timeout)")
            _where = "" if page == "Lobby" else " (on hub)"
            self.log(f"done ({self._fights_done} fights, "
                     f"{self._claim_clicks} rewards){_where}")
            return action_done(f"arena complete{_where}")
        if self._phase_ticks > _EXIT_MAX:
            return action_done("arena exit timeout")
        #  A 取消键 on screen while exiting = some cost/choice dialog is up
        # (result dialogs are cancel-less). Cancel is ALWAYS the safe button
        # on the way out — never the confirm (deep-dive: exit had no buy-dialog
        # guard and clicked 确认键 first = BUY on a surviving purchase dialog).
        cancel = self.find_cls(screen, UC.BTN_CANCEL, conf=0.20)
        if cancel is not None:
            return action_click_box(cancel, "exit: cancel pending dialog (never confirm)")
        # Dismiss a leftover result dialog before ESC.
        confirm = self.find_cls(screen, UC.BTN_CONFIRM, conf=_CLS_CONF, region=_RESULT_BAND)
        if confirm is not None:
            return action_click_box(confirm, "exit: dismiss result dialog")
        close = self.find_cls(screen, UC.BTN_CLOSE_X, conf=_CLS_CONF)
        if close is not None:
            return action_click_box(close, "exit: close dialog")
        back = self.find_cls(screen, UC.BTN_BACK, conf=_CLS_CONF)
        if back is not None:
            return action_click_box(back, "exit: back key")
        # Pace blind ESC — every-tick spam outruns transitions (and on the
        # lobby pops the 是否結束 quit prompt repeatedly).
        if self._phase_ticks % 3 != 0:
            return action_wait(600, "exit: settle before next ESC")
        return self.nav_home(screen, "arena exit")
