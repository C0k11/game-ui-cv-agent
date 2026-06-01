"""ArenaSkill: Handle 戰術對抗賽 (Tactical PvP Arena) daily routine.

PURE-YOLO (2026-05-29): all navigation + clicking goes through the UI YOLO
model's cls names (NO OCR — `screen.ocr_boxes` is empty). cls constants live
in brain/skills/ui_classes.py. Counts (tickets X/5, cooldown timer) are read
from DIGITS via OCR — that path is DEFERRED (OCR re-enabled later, digit-only),
so for bring-up we drive the flow by BUTTON STATE + safety caps instead of
parsed numbers. Every removed count check is marked `# DIGIT-DEFERRED:`.

Flow:
1. ENTER: lobby → 任务大厅入口(NAV_TASKS) → hub → 战术大赛(HUB_ARENA) tile
2. CLAIM_REWARDS: claim daily ranking rewards (领取奖励_黄)
3. CHECK_TICKETS: (DIGIT-DEFERRED) gate by fight cap + button state
4. SELECT_OPPONENT: click top opponent row via cls92 (ARENA_OPPONENT_ROW)
5. FIGHT: 攻击编制(ATTACK_FORMATION) → 跳过战斗 + 出击(SORTIE) → result
6. Loop until cap / no tickets, then EXIT to lobby
"""
from __future__ import annotations

from typing import Any, Dict

from brain.skills.base import (
    BaseSkill, ScreenState,
    action_click, action_click_box, action_click_yolo,
    action_wait, action_back, action_done,
)
from brain.skills import ui_classes as UC


class ArenaSkill(BaseSkill):
    # DIGIT-DEFERRED: can't read the X/5 ticket count without OCR digits.
    # Daily max is 5 fights; cap there as a safety bound. The flow also
    # exits early when it can't actually start a fight (出击 greyed/absent).
    # Daily max is 5 PVP fights — cap there as a safety bound.
    _MAX_FIGHTS_BRINGUP = 5
    # Blind cooldown between fights (~30s real cooldown; can't read the
    # timer without OCR). 0.8s/tick ADB → ~38 ticks. DIGIT-DEFERRED.
    _COOLDOWN_TICKS = 38

    def __init__(self):
        super().__init__("Arena")
        self.max_ticks = 300
        self._fights_done: int = 0
        self._claim_ticks: int = 0
        self._claim_clicks: int = 0
        self._fight_stage: int = 0   # 0=opponent popup, 1=formation, 2=battle
        self._fight_ticks: int = 0
        self._skip_clicked: bool = False  # per-fight 跳过战斗 toggled?
        self._cooldown_ticks: int = 0
        self._check_ticks: int = 0
        self._result_pending: bool = False  # battle-result dialog dedup

    def reset(self) -> None:
        super().reset()
        self._fights_done = 0
        self._claim_ticks = 0
        self._claim_clicks = 0
        self._fight_stage = 0
        self._fight_ticks = 0
        self._skip_clicked = False
        self._cooldown_ticks = 0
        self._check_ticks = 0
        self._result_pending = False
        self._select_attempts = 0
        self._enter_ticks = 0   # FIX: was never reset → stale count caused
        self._hub_clicks = 0    # premature ESC right after clicking the tile

    # ── screen detection (pure YOLO) ─────────────────────────────────
    def _is_arena_screen(self, screen: ScreenState) -> bool:
        """In arena if the ticket icon (main screen) OR the formation
        buttons (攻击编制 / 出击) are visible."""
        return bool(
            self.find_cls(screen, UC.TICKET_ARENA, conf=0.30)
            or self.find_cls(screen, UC.ATTACK_FORMATION, conf=0.30)
            or self.find_cls(screen, UC.SORTIE, conf=0.30)
        )

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1
        if self.ticks >= self.max_ticks:
            self.log("timeout")
            return action_done("arena timeout")

        # ── Battle-result dialog (戰鬥結果 WIN/LOSE) → confirm, fight done ──
        # The dialog shows WIN *or* LOSE with a centered 确认键 (a LOSE has
        # neither 战斗胜利 nor 获得奖励). It can land while we're in fight /
        # check_tickets / select_opponent, and it sits OVER the arena screen
        # (持有票券/TICKET_ARENA visible behind it), so it MUST be handled from
        # ALL those states — otherwise the dialog blocks opponent clicks and we
        # loop forever "waiting for 對戰對象 popup" (the t0103-0153 bug, where a
        # stage-2 TICKET_ARENA check matched the dialog's background and falsely
        # declared the fight done). _result_pending dedups so one dialog counts
        # as one fight even if it persists a few ticks.
        if self.sub_state in ("fight", "check_tickets", "select_opponent"):
            res_confirm = self.find_cls(screen, UC.BTN_CONFIRM, conf=0.30,
                                        region=(0.34, 0.55, 0.66, 0.80))
            res_marker = (self.find_cls(screen, UC.BATTLE_WIN, conf=0.35)
                          or self.find_cls(screen, UC.GOT_REWARD, conf=0.35))
            if res_confirm or res_marker:
                if not self._result_pending:
                    self._fights_done += 1
                    self._result_pending = True
                    self.log(f"fight {self._fights_done} result → dismiss")
                self._fight_stage = 0
                self._fight_ticks = 0
                self._skip_clicked = False
                self._cooldown_ticks = 0
                self.sub_state = "check_tickets"
                if res_confirm:
                    return action_click_box(res_confirm, "dismiss battle result (确认键)")
                return action_click(0.5, 0.9, "tap to dismiss battle result")
            else:
                self._result_pending = False

        popup = self._handle_common_popups(screen)
        if popup:
            return popup

        if screen.is_loading():
            return action_wait(800, "arena loading")

        if self.sub_state == "":
            self.sub_state = "enter"

        if self.sub_state == "enter":
            return self._enter(screen)
        if self.sub_state == "claim_rewards":
            return self._claim_rewards(screen)
        if self.sub_state == "check_tickets":
            return self._check_tickets(screen)
        if self.sub_state == "select_opponent":
            return self._select_opponent(screen)
        if self.sub_state == "fight":
            return self._fight(screen)
        if self.sub_state == "exit":
            return self._exit(screen)
        return action_wait(300, "arena unknown state")

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        self._enter_ticks = getattr(self, "_enter_ticks", 0) + 1
        current = self.detect_current_screen(screen)

        if current == "PVP" or self._is_arena_screen(screen):
            self.log("inside arena")
            self.sub_state = "claim_rewards"
            return action_wait(500, "entered arena")

        if current == "Lobby":
            # Arena is NOT in the bottom nav — enter via the right-side
            # 任务大厅入口 tile → campaign hub.
            act = self.click_cls(screen, UC.NAV_TASKS,
                                 "open campaign hub for arena", conf=0.30)
            if act:
                return act
            # cls miss: surface it (no blind hardcoded click).
            return action_wait(400, "lobby: 任务大厅入口 cls not seen yet")

        if current == "Mission":
            # On campaign hub — click the 战术大赛 tile.
            act = self.click_cls(screen, UC.HUB_ARENA,
                                 "click arena tile on hub", conf=0.30)
            if act:
                self._hub_clicks += 1
                return act
            # Tile not visible THIS tick. If we already clicked it, we're
            # mid-transition into arena — WAIT, do NOT ESC (ESC right after
            # the click backs out of the loading arena → drifts to lobby,
            # the t0032 bug). Only ESC if we NEVER found the tile (likely a
            # detail sub-page) after many ticks.
            if self._hub_clicks == 0 and self._enter_ticks > 8:
                return action_back("hub: 战术大赛 tile never found, ESC out")
            return action_wait(500, "hub: waiting for arena (post-click transition)")

        if self._enter_ticks > 20:
            self.log("can't reach arena, giving up")
            self.sub_state = "exit"
            return action_wait(300, "arena enter timeout")
        if current:
            return action_back(f"back from {current}")
        return action_wait(500, "entering arena")

    def _claim_rewards(self, screen: ScreenState) -> Dict[str, Any]:
        """Claim 领取奖励 buttons in the left panel. YOLO: 领取奖励_黄 /
        领取_黄 active; 领取奖励_灰 means already claimed."""
        self._claim_ticks += 1
        if self._claim_clicks >= 3 or self._claim_ticks > 10:
            self.log(f"reward claim done ({self._claim_clicks} clicks)")
            self.sub_state = "check_tickets"
            return action_wait(300, "claim rewards done")

        # Active claim buttons = the YELLOW cls (grey/claimed buttons are a
        # SEPARATE trained cls 领取奖励_灰, so they simply don't match here —
        # no HSV grey-check needed, per pure-YOLO spec).
        claims = self.find_all_cls(
            screen,
            [UC.CLAIM_REWARD_YELLOW, UC.CLAIM_YELLOW, UC.CLAIM_ALL_YELLOW],
            conf=0.30,
        )
        if claims:
            # bottom-first (daily reward regen-free; time reward regenerates)
            claims.sort(key=lambda b: b.cy, reverse=True)
            self._claim_clicks += 1
            self.log(f"claim reward #{self._claim_clicks}: {claims[0].cls_name}")
            return action_click_box(claims[0], "claim arena reward")

        # Only grey claims (or none) → nothing left to claim.
        self.log("no active arena rewards (grey/none)")
        self.sub_state = "check_tickets"
        return action_wait(200, "claim rewards done (grey)")

    def _check_tickets(self, screen: ScreenState) -> Dict[str, Any]:
        """DIGIT-DEFERRED: original parsed 持有票券 X/5 + 等待時間 timer.
        Without OCR digits we gate by fight cap + blind cooldown, and let
        _fight detect 'can't start' (出击 greyed/absent) to stop early."""
        self._check_ticks += 1

        # Safety cap (daily max).
        if self._fights_done >= self._MAX_FIGHTS_BRINGUP:
            self.log(f"fight cap reached ({self._fights_done}), exiting")
            self.sub_state = "exit"
            return action_wait(300, "arena fight cap reached")

        # Blind cooldown between fights (~30s real). DIGIT-DEFERRED.
        if self._fights_done > 0 and self._cooldown_ticks < self._COOLDOWN_TICKS:
            self._cooldown_ticks += 1
            return action_wait(800, f"arena cooldown {self._cooldown_ticks}/{self._COOLDOWN_TICKS}")

        # Confirm we're on the arena main screen before selecting an opponent.
        if self._is_arena_screen(screen):
            self._check_ticks = 0
            self.sub_state = "select_opponent"
            return action_wait(300, "selecting opponent")
        # Not confirmed. If we've drifted back to lobby/hub, the arena run is
        # over (or entry failed) — exit cleanly instead of waiting forever
        # (the t0035-48 'waiting for arena main screen' wedge on lobby).
        cur = self.detect_current_screen(screen)
        if cur in ("Lobby", "Mission"):
            self.log(f"drifted to {cur}, arena run over")
            self.sub_state = "exit"
            return action_wait(300, "arena: drifted out, exit")
        # Mid-transition / TICKET_ARENA weak — proceed optimistically after a
        # few ticks (DIGIT-DEFERRED: select_opponent uses a fixed position).
        if self._check_ticks > 6:
            self._check_ticks = 0
            self.sub_state = "select_opponent"
            return action_wait(300, "arena screen unconfirmed, proceeding")
        return action_wait(400, "waiting for arena main screen")

    def _select_opponent(self, screen: ScreenState) -> Dict[str, Any]:
        """Pick an opponent by clicking the TOP-most opponent-row region
        (cls92 ARENA_OPPONENT_ROW, detected by the UI model). v5 bounds each
        of the 3 opponent rows cleanly (cy≈0.34/0.57/0.79 on the right panel),
        so this needs NO avatar model and NO hardcoded position — works at any
        window size / scaling. Click the top row (lowest cy).
        DIGIT-DEFERRED: rank-based 'easiest opponent' pick returns with OCR."""
        self._select_attempts = getattr(self, "_select_attempts", 0) + 1
        if self._select_attempts > 8:
            self.log("opponent select kept failing (no opponent row / popup), exiting")
            self.sub_state = "exit"
            return action_wait(300, "arena: opponent select failed")
        # cls92 opponent-row regions (UI model). Right panel, cx>0.5.
        rows = self.find_all_cls(screen, UC.ARENA_OPPONENT_ROW, conf=0.25)
        rows = [b for b in rows if b.cx > 0.5]
        if not rows:
            # No opponent row detected yet — surface the gap, don't blind-click.
            self.log(f"no opponent row (cls92) #{self._select_attempts}, waiting")
            return action_wait(400, "waiting for opponent rows (cls92)")
        top = min(rows, key=lambda b: b.cy)   # top-most opponent row
        self.log(f"selecting opponent row ({top.cx:.2f},{top.cy:.2f}) of {len(rows)}")
        self.sub_state = "fight"
        self._fight_stage = 0
        self._fight_ticks = 0
        self._skip_clicked = False
        return action_click_box(top, "select opponent (cls92 row)")

    def _fight(self, screen: ScreenState) -> Dict[str, Any]:
        """Stage 0: 對戰對象 popup → 攻击编制. Stage 1: formation →
        (跳过战斗) + 出击. Stage 2: battle (result handled in tick)."""
        self._fight_ticks += 1
        # Stage2 = the actual PVP battle: up to ~4 min real-time when 跳过战斗
        # didn't toggle. @ ~0.8-1s/tick that's ~300 ticks — the old 80 cap timed
        # out mid-fight (live t93-134: battle still had 0:53 left). Stages 0/1
        # (popup/formation nav) stay short. (bug 2026-06-01)
        max_ticks = 320 if self._fight_stage >= 2 else 40
        if self._fight_ticks > max_ticks:
            self.log(f"fight timeout (stage={self._fight_stage})")
            self._fight_ticks = 0
            self.sub_state = "check_tickets"
            if self._fight_stage >= 2:
                return action_wait(500, "fight timeout (battle)")
            return action_back("fight timeout, back")

        # Stage 0: opponent popup → 攻击编制
        if self._fight_stage == 0:
            act = self.click_cls(screen, UC.ATTACK_FORMATION,
                                 "click 攻击编制", conf=0.30)
            if act:
                self._fight_stage = 1
                self._select_attempts = 0   # popup opened — real progress
                return act
            # popup didn't open / closed → re-select after a few ticks
            if self._fight_ticks > 3:
                if self._is_arena_screen(screen):
                    self.sub_state = "select_opponent"
                    return action_wait(300, "popup didn't open, re-selecting")
                if self._fight_ticks > 10:
                    self.sub_state = "check_tickets"
                    self._fight_ticks = 0
                    return action_wait(500, "fight stuck, re-checking")
            return action_wait(500, "waiting for 對戰對象 popup")

        # Stage 1: formation → optionally skip-battle, then 出击
        if self._fight_stage == 1:
            sortie = self.find_cls(screen, UC.SORTIE, conf=0.30)
            if sortie:
                # NOTE: there's no 出击_灰 cls trained, so a ticket-less greyed
                # 出击 can't be cls-detected (count is DIGIT-DEFERRED). Per the
                # no-HSV spec we don't pixel-check; the fight-cap + opponent-
                # select cap bound the loop. (TRAIN: 出击_灰 cls.)
                # Toggle 跳过战斗 once (instant resolve, faster) before sortie.
                if not self._skip_clicked:
                    skip = self.find_cls(screen, UC.BATTLE_SKIP_TOGGLE, conf=0.30)
                    if skip:
                        self._skip_clicked = True
                        return action_click_box(skip, "toggle 跳过战斗")
                self.log("clicking 出击")
                self._fight_stage = 2
                return action_click_box(sortie, "click 出击")
            return action_wait(500, "waiting for formation 出击")

        # Stage 2: battle in progress. The 戰鬥結果 WIN/LOSE dialog is caught
        # by the top-level result handler (runs every tick from the "fight"
        # state). Here we only cover the case where a WIN/获得奖励 popup was
        # already dismissed by the global interceptor — leaving us back on the
        # CLEAN arena main screen with no dialog. (_result_pending dedups vs
        # the top handler so the fight isn't counted twice.)
        if self._fight_ticks > 4 and self.find_cls(screen, UC.TICKET_ARENA, conf=0.30):
            self.log("back on arena main (result auto-dismissed) — fight complete")
            self._fight_stage = 0
            self._fight_ticks = 0
            self._skip_clicked = False
            if not self._result_pending:
                self._fights_done += 1
            self._result_pending = False
            self._cooldown_ticks = 0
            self.sub_state = "check_tickets"
            return action_wait(300, "fight done, back on arena main")
        skip = self.find_cls(screen, UC.BATTLE_SKIP, conf=0.30)
        if skip:
            return action_click_box(skip, "skip battle")
        return action_wait(1000, "battle in progress")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if screen.is_lobby():
            self.log(f"done ({self._fights_done} fights, {self._claim_clicks} rewards)")
            return action_done("arena complete")
        # If back on the campaign hub, stop here so the next campaign skill
        # reuses the hub (no lobby round-trip).
        if self.detect_current_screen(screen) == "Mission":
            self.log(f"done on hub ({self._fights_done} fights)")
            return action_done("arena complete (on hub)")
        # A leftover 戰鬥結果 / reward dialog blocks the way out — ESC can't
        # dismiss it (live t151-257: ~100 ticks looping back on 确认键). Click
        # its confirm / X first. (bug 2026-06-01, same as bounty._exit)
        confirm = self.find_cls(screen, UC.BTN_CONFIRM, conf=0.30,
                                region=(0.30, 0.55, 0.70, 0.85))
        if confirm:
            return action_click_box(confirm, "arena exit: dismiss result dialog (确认键)")
        x_btn = self.find_cls(screen, UC.BTN_CLOSE_X, conf=0.30)
        if x_btn:
            return action_click_box(x_btn, "arena exit: close result dialog (X)")
        return action_back("arena exit: back to lobby")
