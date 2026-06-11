"""ArenaSkill — 战术大赛 (PVP) daily routine (pure-YOLO rewrite).

Verified flow (interactive probe 2026-06-01, data/_missions_probe_log.md
Step 23-34). Arena is PVP — NOT a sweep. Key probe findings vs the old skill:
- Battles are FULLY AUTO-resolved (a few seconds). NO 跳过战斗 toggle needed —
  just 出击 then poll-dismiss the result dialog(s).
- Winning + setting a new season-best pops an EXTRA 達成賽季最高紀錄 popup that
  AWARDS pyroxene (a GAIN — dismiss it, never confuse with a cost).
- ~25s 等待時間 cooldown after each fight (blind tick wait here; OCR mm:ss is
  unreliable — v6 could refine).

★★ pyroxene protection ★★
  digit-OCR 战术大赛票 X/5. Ticket 0 ⇒ STOP challenging (a 0-ticket 出击 pops a
  購買戰術大賽券 青辉石 dialog). The buy-dialog guard requires a 取消键 present
  (a cost dialog has cancel) so it never misfires on the cancel-less
  達成賽季最高紀錄 REWARD popup.

State machine
-------------
enter   lobby → NAV_TASKS → hub → HUB_ARENA → arena main.
claim   click every 领取奖励_黄 (获得奖励 → 点击继续字样 dismiss). 领取奖励_灰
        ignored. Done when no 领取奖励_黄 remains.
fight_check  digit-OCR ticket X/5. 0 / cap → exit. Cooldown wait between fights.
select  click TOP 战术大赛对战选择区域 (cls92) row.
fight   對戰對象 → 攻击编制 → 编队屏 出击 → auto-battle → poll-dismiss all 确认键.
exit    返回键 / 回大厅 → lobby (or hub) → done.

Detectors: base "ui" + "battle" (SKILL_YOLO_MAP).
"""
from __future__ import annotations

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
# ~25s 等待時間 between fights. Blind wait @1s/tick (+~0.15s capture/infer) so the
# cooldown bar clears before re-selecting (else the opponent row is greyed /
# unclickable → select spins → false exit). If it still spins, _select re-enters
# cooldown for one more round. v6 could OCR the mm:ss countdown.
_COOLDOWN_TICKS = 22

_ENTER_MAX = 24
_CLAIM_MAX = 12
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
        self._ticket_misses: int = 0        # consecutive failed ticket reads

    def reset(self) -> None:
        super().reset()
        self._init_state()

    def _goto(self, sub_state: str) -> None:
        self.sub_state = sub_state
        self._phase_ticks = 0

    # ── helpers ──────────────────────────────────────────────────────────
    def _on_arena(self, screen: ScreenState) -> bool:
        return self.find_cls(
            screen, [UC.TICKET_ARENA, UC.ATTACK_FORMATION, UC.SORTIE,
                     UC.ARENA_OPPONENT_ROW], conf=_CLS_CONF
        ) is not None

    def _read_tickets(self, screen: ScreenState) -> Optional[int]:
        """digit-OCR 持有票券 X/5 next to the arena ticket icon (LEFT panel,
        ~0.055,0.678 — NOT the top bar). ★ money defense #1 (0 → stop, never the
        buy-ticket-pyroxene trap).

        ★ Runs on a CLEAN ADB frame, not screen.frame: the overlay burns a
        tight box+label onto the small ticket icon in every DXcam frame, which
        killed ui_v7's detection of it outright (live 2026-06-09: 0 detections
        on burned frames vs conf 0.95 on the clean frame → every read None →
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
        bh = icon.y2 - icon.y1
        # ★ Strip must SKIP the fixed "持有票券" label (x≈0.068-0.118): feeding
        # the digit-OCR Chinese text made it return None on EVERY frame (live
        # 2026-06-09: fight 1 ran with zero successful reads). Digits "X/5" sit
        # at x≈0.131-0.145; icon.x2≈0.066 → offset +0.05, span 0.08 verified
        # offline on the live frame ('4/5' → (4,5)) with margin both sides.
        x1 = max(0.0, icon.x2 + 0.050)
        x2 = min(1.0, x1 + 0.08)
        raw = run_digit_ocr(frame, (x1, icon.y1 - bh * 0.4, x2, icon.y2 + bh * 0.4))
        res = parse_count(raw)
        if res is None or res[0] is None:
            return None
        # Strict numerator proof: a bare number with no '/' could be the
        # DENOMINATOR of a left-clipped "X/5" — the 2026-06-02 incident class
        # ("0/5" clipped to '5' would read 0 tickets as 5 → 出击 at 0 → buy
        # dialog). Right-clipped '3/' (denominator lost) is fine — numerator
        # is provably the leading digit.
        s = (raw or "").strip()
        if "/" not in s or not s[:1].isdigit():
            return None
        return res[0]

    def _buy_dialog(self, screen: ScreenState) -> bool:
        """A 青辉石 icon in the body AND a 取消键 = a buy-ticket cost dialog
        (distinct from the cancel-less 達成賽季最高紀錄 reward popup).
        conf 0.20 (model floor, deep-dive C4): a DANGER detector must be as
        sensitive as the model allows — a false positive merely cancels+exits."""
        if self.find_cls(screen, UC.TOPBAR_PYROXENE, conf=0.20, region=_PYROXENE_BODY_REGION) is None:
            return False
        return self.find_cls(screen, UC.BTN_CANCEL, conf=0.20) is not None

    # ── tick ────────────────────────────────────────────────────────────────
    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1
        self._phase_ticks += 1

        if self.ticks >= self.max_ticks:
            self.log(f"timeout ({self._fights_done} fights)")
            return action_done("arena timeout")

        # ⛔ buy-ticket dialog (青辉石 cost + 取消键) → cancel + exit. Primary
        # safety is the ticket gate; this is the backstop.
        if self.sub_state in ("fight", "select", "fight_check") and self._buy_dialog(screen):
            self.log("⛔ ticket-purchase dialog (青辉石) — cancel, never buy")
            self._goto("exit")
            cancel = self.find_cls(screen, UC.BTN_CANCEL, conf=_CLS_CONF)
            if cancel is not None:
                return action_click_box(cancel, "cancel ticket purchase")
            return action_back("dismiss buy dialog")

        # Battle-result dialogs (戰鬥結果 WIN/LOSE + 達成賽季最高紀錄) → dismiss
        # ALL centered 确认键. Can land over the arena main, so handled here for
        # the in-fight states. _result_pending dedups one fight per battle.
        if self.sub_state in ("fight", "select", "fight_check"):
            res_confirm = self.find_cls(screen, UC.BTN_CONFIRM, conf=_CLS_CONF, region=_RESULT_BAND)
            res_marker = self.find_cls(screen, [UC.BATTLE_WIN, UC.GOT_REWARD], conf=0.35)
            if res_confirm is not None or res_marker is not None:
                # ⛔ Never click a centered 确认键 while a 取消键 is also visible.
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
                # expired) at fight stage 1 was counted as fight 1 → cap would
                # end arena one real fight early with a ticket unspent.
                if not self._result_pending and self.sub_state == "fight" \
                        and self._fight_stage >= 2:
                    self._fights_done += 1
                    self._result_pending = True
                    self.log(f"fight {self._fights_done} result → dismiss")
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
            self.log("inside arena → claim")
            self._goto("claim")
            return action_wait(400, "entered arena")

        # ★ Settle after any nav click — during the page transition the OLD page
        # (and its cls boxes) linger at low conf for a frame or two; re-clicking
        # the same spot then lands on the NEW page's UI. Live 2026-06-09: the
        # tick-2 "arena tile" re-click hit the freshly-loaded arena main and
        # opened the 對戰對象 popup → select spun on covered rows → false exit.
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
            # ★ Hall scan (user iron rule 2026-06-11): the 战术大赛 tile's own
            # red/yellow dot is the work signal — visible only here. No dot →
            # nothing to claim/fight today → graceful exit.
            has_work = self.hall_tile_dot(screen, UC.HUB_ARENA)
            if has_work is False:
                self.log("hall scan: 战术大赛 无红黄点 → no work today, done")
                return action_done("arena no work (hall scan)")
            act = self.click_cls(screen, UC.HUB_ARENA, "click arena tile", conf=_CLS_CONF)
            if act is not None:
                self._enter_settle = 4
                return act
            return action_wait(450, "hub: arena tile not seen (transition)")

        if self._enter_ticks > _ENTER_MAX:
            self.log("can't reach arena, exiting")
            self._goto("exit")
            return action_wait(300, "enter timeout")
        if page is not None:
            return action_back(f"back from {page}")
        return action_wait(450, "entering arena")

    def _claim(self, screen: ScreenState) -> Dict[str, Any]:
        # Reward reveal popup → dismiss via continue / header (NEVER center).
        cont = self.find_cls(screen, UC.STORY_TAP_CONTINUE, conf=_CLS_CONF)
        if cont is not None:
            return action_click_box(cont, "dismiss reward (continue)")
        got = self.find_cls(screen, UC.GOT_REWARD, conf=_CLS_CONF)
        if got is not None:
            return action_click_box(got, "dismiss reward (header)")

        if self._claim_clicks >= 4 or self._phase_ticks > _CLAIM_MAX:
            self.log(f"claim done ({self._claim_clicks})")
            self._goto("fight_check")
            return action_wait(250, "claim done → fight_check")

        # Click any active 领取奖励_黄 (灰 = already claimed, ignore).
        claim = self.find_cls(screen, [UC.CLAIM_REWARD_YELLOW, UC.CLAIM_YELLOW], conf=_CLS_CONF)
        if claim is not None:
            self._claim_clicks += 1
            self.log(f"claim arena reward #{self._claim_clicks} ({claim.cls_name})")
            return action_click_box(claim, "claim arena reward")

        self.log("no 领取奖励_黄 → fight_check")
        self._goto("fight_check")
        return action_wait(250, "no active rewards → fight_check")

    def _fight_check(self, screen: ScreenState) -> Dict[str, Any]:
        # ★ Hard ticket gate (money-safety): NO path may leave fight_check
        # toward select/fight without a SUCCESSFUL ticket read this phase.
        # Deep-dive C1 + live 2026-06-09: the old ">8 ticks → select anyway"
        # fallback let the entire first fight run with _tickets=None (the OCR
        # strip was mis-geometried and EVERY read failed silently).
        tickets = self._read_tickets(screen)
        if tickets is None:
            self._ticket_misses += 1
            page = self.detect_screen_yolo(screen)
            if page in ("Lobby", "Mission"):
                self.log(f"drifted to {page} → arena over")
                self._goto("exit")
                return action_wait(300, "drifted out → exit")
            if self._ticket_misses > 12:
                self.log("tickets unreadable after retries → exit (money fail-closed)")
                self._goto("exit")
                return action_wait(300, "ticket unreadable → exit")
            return action_wait(400, f"ticket read retry {self._ticket_misses}/12 (fail-closed gate)")
        self._ticket_misses = 0
        self._tickets = tickets
        if tickets <= 0:
            self.log("tickets 0/5 → arena done")
            self._goto("exit")
            return action_wait(300, "0 tickets → exit")

        # Safety cap.
        if self._fights_done >= _MAX_FIGHTS:
            self.log(f"fight cap reached ({self._fights_done})")
            self._goto("exit")
            return action_wait(300, "fight cap → exit")

        # Cooldown between fights (~25s 等待時間). Wait it out fully — selecting an
        # opponent mid-cooldown does nothing (greyed row) and burns select attempts.
        if self._fights_done > 0 and self._cooldown < _COOLDOWN_TICKS:
            self._cooldown += 1
            return action_wait(1000, f"arena cooldown {self._cooldown}/{_COOLDOWN_TICKS} (~25s)")

        # A successful ticket read ⇒ the arena left panel is on screen — safe
        # to select. (The old _on_arena/blind-select fallbacks are gone; they
        # were the leak.)
        self._goto("select")
        return action_wait(250, "arena main (tickets read) → select")

    def _select(self, screen: ScreenState) -> Dict[str, Any]:
        # 對戰對象 popup may ALREADY be open (stray click / re-entry) — its body
        # covers the opponent rows, so spinning for cls92 here would falsely
        # exit (live 2026-06-09). Hand over to fight stage 0, which clicks the
        # visible 攻击编制.
        if self.find_cls(screen, UC.ATTACK_FORMATION, conf=_CLS_CONF) is not None:
            self.log("對戰對象 already open → fight stage0")
            self._fight_stage = 0
            self._fight_ticks = 0
            self._stage_settle = 0
            self._goto("fight")
            return action_wait(250, "popup already open → fight")

        self._select_attempts += 1
        if self._select_attempts > 8:
            # Between fights a spin usually means the cooldown bar hadn't fully
            # cleared (greyed opponent row). Re-enter cooldown for one more round
            # rather than falsely ending arena early (we must finish all 5).
            if self._fights_done < _MAX_FIGHTS and self._select_rounds < 2:
                self._select_rounds += 1
                self._select_attempts = 0
                self._cooldown = 0
                self.log(f"select spun → extra cooldown (round {self._select_rounds})")
                self._goto("fight_check")
                return action_wait(400, "select spun → re-cooldown")
            self.log("opponent select failed → exit")
            self._goto("exit")
            return action_wait(300, "select failed → exit")

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
        # ★ Blind settle after every click — give the page time to transition
        # before reading it. Without this we act on a stale/animating frame and
        # the click looks like it "did nothing" (点了没反应 → 误判重选/空点).
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

        # Stage 0: 對戰對象 popup → 攻击编制. The popup needs real transition
        # time; be patient (don't re-select early and interrupt it opening).
        if self._fight_stage == 0:
            af = self.find_cls(screen, UC.ATTACK_FORMATION, conf=_CLS_CONF)
            if af is not None:
                self.log("對戰對象 open → click 攻击编制")
                self._fight_stage = 1
                self._select_attempts = 0
                self._select_rounds = 0     # select succeeded → reset retry budget
                self._stage_settle = 4      # 编队屏 loads characters → slowest screen
                return action_click_box(af, "click 攻击编制")
            if self._fight_ticks > 10 and self._on_arena(screen):
                self._goto("select")
                return action_wait(400, "對戰對象 didn't open after 10t → re-select")
            return action_wait(650, "waiting for 對戰對象 popup (transition)")

        # Stage 1: 编队屏 → 出击 (no skip toggle; arena auto-resolves).
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
            self.log("back on arena main → fight complete")
            if not self._result_pending:
                self._fights_done += 1
            self._result_pending = False
            self._fight_stage = 0
            self._fight_ticks = 0
            self._cooldown = 0
            self._goto("fight_check")
            return action_wait(300, "fight done → fight_check")
        return action_wait(1000, "auto-battle in progress")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        page = self.detect_screen_yolo(screen)
        if page == "Lobby":
            self.log(f"done ({self._fights_done} fights, {self._claim_clicks} rewards)")
            return action_done("arena complete")
        if page == "Mission":
            self.log(f"done on hub ({self._fights_done} fights)")
            return action_done("arena complete (on hub)")
        if self._phase_ticks > _EXIT_MAX:
            return action_done("arena exit timeout")
        # ⛔ A 取消键 on screen while exiting = some cost/choice dialog is up
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
        return action_back("arena exit: ESC toward hub/lobby")
