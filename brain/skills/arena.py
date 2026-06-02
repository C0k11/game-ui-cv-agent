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
_PYROXENE_BODY_REGION = (0.30, 0.16, 0.75, 0.48)
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
        return self.dot_on_entry(screen, [UC.NAV_TASKS])

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
        self._select_attempts: int = 0
        self._select_rounds: int = 0        # extra-cooldown retries when select spins
        self._result_pending: bool = False
        self._tickets: Optional[int] = None

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
        buy-ticket-pyroxene trap). read_count's generic strip clipped the first
        digit → None (live 2026-06-02: ticket unread → kept retrying fights at
        0 tickets). Anchor on the icon + OCR a tighter-left strip."""
        if screen.frame is None:
            return None
        icon = self.find_cls(screen, UC.TICKET_ARENA, conf=_CLS_CONF,
                             region=(0.0, 0.58, 0.22, 0.78))
        if icon is None:
            return None
        try:
            from brain.pipeline import run_digit_ocr, parse_count
        except Exception:
            return None
        bh = icon.y2 - icon.y1
        x1 = max(0.0, icon.x2 + 0.002)
        x2 = min(1.0, x1 + 0.11)
        raw = run_digit_ocr(screen.frame, (x1, icon.y1 - bh * 0.4, x2, icon.y2 + bh * 0.4))
        res = parse_count(raw)
        if res is None or res[0] is None:
            return None
        return res[0]

    def _buy_dialog(self, screen: ScreenState) -> bool:
        """A 青辉石 icon in the body AND a 取消键 = a buy-ticket cost dialog
        (distinct from the cancel-less 達成賽季最高紀錄 reward popup)."""
        if self.find_cls(screen, UC.TOPBAR_PYROXENE, conf=_CLS_CONF, region=_PYROXENE_BODY_REGION) is None:
            return False
        return self.find_cls(screen, UC.BTN_CANCEL, conf=_CLS_CONF) is not None

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
                if not self._result_pending and self.sub_state == "fight":
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

        page = self.detect_screen_yolo(screen)
        if page == "Lobby":
            act = self.click_cls(screen, UC.NAV_TASKS, "open campaign hub", conf=_CLS_CONF)
            if act is not None:
                return act
            return action_wait(400, "lobby: NAV_TASKS not seen")
        if page == "Mission":
            act = self.click_cls(screen, UC.HUB_ARENA, "click arena tile", conf=_CLS_CONF)
            if act is not None:
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
        # Ticket gate (money-safety): 0 → done.
        tickets = self._read_tickets(screen)
        if tickets is not None:
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

        if self._on_arena(screen):
            self._goto("select")
            return action_wait(250, "arena main → select")
        page = self.detect_screen_yolo(screen)
        if page in ("Lobby", "Mission"):
            self.log(f"drifted to {page} → arena over")
            self._goto("exit")
            return action_wait(300, "drifted out → exit")
        if self._phase_ticks > 8:
            self._goto("select")
            return action_wait(300, "arena unconfirmed → select")
        return action_wait(400, "waiting for arena main")

    def _select(self, screen: ScreenState) -> Dict[str, Any]:
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
        return action_back("arena exit: ESC toward hub/lobby")
