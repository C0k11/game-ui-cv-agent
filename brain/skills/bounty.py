"""BountySkill: Handle 悬赏通缉 (Bounties) sweep — PURE YOLO (OCR-free).

Flow:
1. ENTER: From lobby click 任务大厅入口 (NAV_TASKS) → campaign hub → click
   悬赏通缉 (HUB_BOUNTY) → bounty screen.
2. CHECK_TICKETS: confirm we're on bounty screen via 悬赏通缉票 (TICKET_BOUNTY).
   DIGIT-DEFERRED — we no longer parse "剩余 X/Y"; the sweep loop self-
   terminates when 入场键/扫荡开始 grey out or disappear.
3. SELECT_STAGE: click the bottom-most 入场键 (STAGE_ENTER) = hardest stage.
4. SWEEP: MAX_可点击 (QTY_MAX) → 扫荡开始 (SWEEP_START) → 确认键 (BTN_CONFIRM)
   → dismiss 获得奖励 (GOT_REWARD). MAX sweeps ALL tickets in one shot.
5. EXIT: back to lobby (or stay on hub for the next campaign skill).

Migration note (2026-05-29): every OCR find_text*/find_any_text/has_text
call was replaced by find_cls/click_cls against ui_classes constants. OCR
is globally disabled (pipeline._OCR_ENABLED=False) so screen.ocr_boxes is
always empty. Digit reads (ticket / sortie counts) are DEFERRED — see the
`# DIGIT-DEFERRED:` notes — and the flow is driven by button STATE instead.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from brain.skills.base import (
    BaseSkill, ScreenState,
    action_click, action_click_box, action_click_yolo,
    action_wait, action_back, action_done, action_scroll, action_swipe,
)
from brain.skills import ui_classes as UC


class BountySkill(BaseSkill):
    # Each branch maps to its YOLO cls (weak — 12f each in v1, but exists)
    # plus a hardcoded fallback card position in the right-side panel,
    # gated behind a tick counter (NO cls fires after a few ticks).
    _ALL_BRANCHES = [
        ("高架公路", UC.STAGE_HIGHWAY, (0.90, 0.25)),
        ("沙漠铁道", UC.STAGE_DESERT_RAIL, (0.90, 0.41)),
        ("教室", UC.STAGE_CLASSROOM, (0.90, 0.57)),
    ]

    @staticmethod
    def _load_enabled_branches() -> List:
        """Load enabled branches from app_config.json (active profile).

        Returns a filtered subset of _ALL_BRANCHES preserving the ORDER
        the user specified in the config. If no config / empty list, all
        branches are enabled (default behavior).

        Config values may use traditional-Chinese names (沙漠鐵道) — match
        those to our simplified cls-keyed branch names too.
        """
        try:
            import json
            cfg_path = Path(__file__).resolve().parents[2] / "data" / "app_config.json"
            if not cfg_path.exists():
                return list(BountySkill._ALL_BRANCHES)
            data = json.loads(cfg_path.read_text("utf-8"))
            active = data.get("active_profile", "default")
            profile = (data.get("profiles") or {}).get(active, {})
            enabled = profile.get("bounty_branches")
            if not isinstance(enabled, list) or not enabled:
                return list(BountySkill._ALL_BRANCHES)
            # Accept both simplified (cls) names and the legacy traditional
            # aliases the config might still hold.
            alias_map = {
                "高架公路": "高架公路", "高架": "高架公路", "Overpass": "高架公路",
                "沙漠铁道": "沙漠铁道", "沙漠鐵道": "沙漠铁道", "沙漠": "沙漠铁道",
                "Railway": "沙漠铁道",
                "教室": "教室", "Classroom": "教室",
            }
            by_name = {b[0]: b for b in BountySkill._ALL_BRANCHES}
            out = []
            for n in enabled:
                key = alias_map.get(n, n)
                if key in by_name and by_name[key] not in out:
                    out.append(by_name[key])
            return out or list(BountySkill._ALL_BRANCHES)
        except Exception:
            return list(BountySkill._ALL_BRANCHES)

    def __init__(self):
        super().__init__("Bounty")
        self.max_ticks = 60
        self._sweep_stage: int = 0  # 0=select MAX, 1=sweep start, 2=confirm, 3=dismiss result
        self._sweep_attempts: int = 0
        self._sweep_clicks: int = 0  # SAFETY CAP — caps sweep cycles (DIGIT-DEFERRED)
        self._enter_attempts: int = 0
        self._branch_idx: int = 0
        self._branches = list(self._ALL_BRANCHES)
        self._done_branches: set = set()   # branch indices already swept
        self._all_done: bool = False       # every enabled branch attempted

    def reset(self) -> None:
        super().reset()
        self._sweep_stage = 0
        self._sweep_attempts = 0
        self._sweep_clicks = 0
        self._stage_ticks = 0
        self._loc_ticks = 0
        self._enter_attempts = 0
        self._branch_idx = 0
        self._done_branches = set()
        self._all_done = False
        self._branches = self._load_enabled_branches()
        if self._branches:
            self.log(f"bounty branches enabled: {[b[0] for b in self._branches]}")
        else:
            self.log("no bounty branches enabled — will exit")

    def _advance_branch(self) -> None:
        """Mark the current branch as swept (one MAX sweep clears all its
        tickets) and rotate to the next UNDONE branch. Sets self._all_done
        when every enabled branch has been attempted once.

        FIX 2026-05-29: the old modulo-rotate looped forever on a single
        enabled branch (教室) — sweep → 'advance' back to 教室 → sweep …
        until max_ticks, and it also reset _sweep_clicks each time so the
        safety cap never accumulated. Now each branch is done exactly once."""
        if not self._branches:
            self._all_done = True
            return
        self._done_branches.add(self._branch_idx)
        remaining = [i for i in range(len(self._branches))
                     if i not in self._done_branches]
        if not remaining:
            self._all_done = True
            return
        self._branch_idx = remaining[0]
        self._loc_ticks = 0
        self._stage_ticks = 0
        self._sweep_stage = 0
        self._sweep_attempts = 0
        self._sweep_clicks = 0

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1

        if self.ticks >= self.max_ticks:
            self.log("timeout")
            return action_done("bounty timeout")

        if not self._branches:
            return action_done("no bounty branches enabled in config")

        # AP-exhausted popup — can appear at any time during sweep.
        # GAP: no YOLO cls for the "AP不足/體力不足" popup. _handle_common_popups
        # (still OCR, migrated separately) catches the exit/buy prompt and
        # cancels it. Until that lands we surface it via the generic
        # cancel/close path below rather than blind-clicking.

        # Sweep-result popup — dismiss and continue. (OCR "獲得獎勵" → GOT_REWARD)
        result = self.find_cls(screen, UC.GOT_REWARD, conf=0.30)
        if result:
            self.log("sweep result popup, dismissing")
            return action_click(0.5, 0.9, "dismiss sweep result")

        popup = self._handle_common_popups(screen)
        if popup:
            return popup

        if screen.is_loading():
            return action_wait(800, "bounty loading")

        if self.sub_state == "":
            self.sub_state = "enter"

        if self.sub_state == "enter":
            return self._enter(screen)
        if self.sub_state == "check_tickets":
            return self._check_tickets(screen)
        if self.sub_state == "select_location":
            return self._select_location(screen)
        if self.sub_state == "select_stage":
            return self._select_stage(screen)
        if self.sub_state == "sweep":
            return self._sweep(screen)
        if self.sub_state == "exit":
            return self._exit(screen)

        return action_wait(300, "bounty unknown state")

    def _is_bounty_screen(self, screen: ScreenState) -> bool:
        """Detect the bounty stage screen via its YOLO signature.

        The bounty screen's unique marker is the 悬赏通缉票 (TICKET_BOUNTY)
        sweep-ticket icon — exactly what PAGE_SIGNATURES["Bounty"] keys on.
        (OCR header/票券/關卡目錄/LocationSelect probes → TICKET_BOUNTY cls.)
        """
        return self.find_cls(screen, UC.TICKET_BOUNTY, conf=0.30) is not None

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        self._enter_attempts += 1
        current = self.detect_current_screen(screen)

        # Already on the bounty stage screen?
        if current == "Bounty" or self._is_bounty_screen(screen):
            self.log("inside bounty")
            self.sub_state = "check_tickets"
            return action_wait(500, "entered bounty")

        if current == "Lobby":
            # 任务 is NOT in the bottom nav bar — it's the right-side campaign
            # entry tile (NAV_TASKS = 任务大厅入口). (OCR "任務"/"任务" → NAV_TASKS)
            tap = self.click_cls(screen, UC.NAV_TASKS, "open campaign hub (NAV_TASKS)", conf=0.30)
            if tap:
                self.log("clicking campaign entry (NAV_TASKS)")
                return tap
            # NO hardcoded fallback (no-hardcode rule — fixed fractions break on
            # window resize / scaling / layout shift). NAV_TASKS detects ~0.97
            # on real lobby; if it momentarily misses, wait for a clean frame.
            return action_wait(400, "lobby: NAV_TASKS not seen, waiting")

        # On the campaign / Mission hub — click 悬赏通缉 (HUB_BOUNTY) tile.
        if current == "Mission":
            tap = self.click_cls(screen, UC.HUB_BOUNTY, "click bounty tile (HUB_BOUNTY)", conf=0.30)
            if tap:
                self.log("clicking bounty tile (HUB_BOUNTY)")
                return tap
            # NO hardcoded fallback. HUB_BOUNTY conf fluctuates (under-trained,
            # to be fixed in v4); wait for a frame where it resolves rather than
            # blind-clicking a fixed grid position. enter_attempts>20 exits.
            return action_wait(500, "hub: HUB_BOUNTY not seen, waiting")

        # On some other known screen — back out toward the hub/lobby.
        if current and current not in ("Bounty", "Mission"):
            self.log(f"wrong screen '{current}', backing out")
            return action_back(f"back from {current}")

        if self._enter_attempts > 20:
            self.log("can't reach bounty, giving up")
            self.sub_state = "exit"
            return action_wait(300, "bounty enter timeout")

        return action_wait(500, "entering bounty")

    def _check_tickets(self, screen: ScreenState) -> Dict[str, Any]:
        """Confirm we're on the bounty screen, then proceed to select a stage.

        DIGIT-DEFERRED: the original parsed "持有票券 X/Y" via OCR digits and
        exited when X==0. We no longer read digits. Instead we confirm the
        bounty screen via the 悬赏通缉票 (TICKET_BOUNTY) cls and let the sweep
        loop self-terminate when 入场键/扫荡开始 grey out or vanish (= no
        tickets / no AP). The 0-ticket early-exit is therefore handled
        downstream by button state, not a count read.
        """
        # All enabled branches have been swept once → done.
        if self._all_done:
            self.sub_state = "exit"
            return action_wait(300, "all bounty branches swept → exit")
        if self._is_bounty_screen(screen):
            self.sub_state = "select_location"
            branch_name = self._branches[self._branch_idx][0]
            # DIGIT-DEFERRED: no ticket count to log/gate on.
            return action_wait(300, f"on bounty screen, selecting '{branch_name}'")

        # Not confirmed on bounty screen yet — wait a few ticks, then proceed
        # optimistically (TICKET_BOUNTY is weak @20f; absence != not-here).
        if self.ticks > 10:
            self.log("TICKET_BOUNTY not confirmed, proceeding to select location")
            self.sub_state = "select_location"
            return action_wait(300, "bounty screen unconfirmed, proceeding")

        return action_wait(500, "confirming bounty screen (TICKET_BOUNTY)")

    def _select_location(self, screen: ScreenState) -> Dict[str, Any]:
        """Select the target bounty branch from the right-side location panel.

        The bounty LocationSelect shows 3 branch cards (高架公路 / 沙漠铁道 /
        教室). Click the configured branch to open its stage list.
        (OCR alias match + hardcoded card pos → per-branch cls + gated pos.)
        """
        self._loc_ticks = getattr(self, '_loc_ticks', 0) + 1
        branch_name, branch_cls, fallback_pos = self._branches[self._branch_idx]

        # Already in the stage list? (入场键 visible on the right side) → skip ahead.
        if self.find_cls(screen, [UC.STAGE_ENTER, UC.STAGE_ENTER_LOCKED],
                         conf=0.30, region=(0.60, 0.15, 1.0, 0.95)):
            self.log(f"stage list visible for branch '{branch_name}'")
            self.sub_state = "select_stage"
            self._stage_ticks = 0
            return action_wait(300, "stage list visible")

        # Select the target branch card by its cls (right panel).
        tap = self.click_cls(screen, branch_cls,
                             f"select bounty branch '{branch_name}'",
                             conf=0.30, region=(0.60, 0.12, 1.0, 0.80))
        if tap:
            self.log(f"selecting branch '{branch_name}' (cls {branch_cls})")
            self.sub_state = "select_stage"
            self._stage_ticks = 0
            return tap

        # NO hardcoded fallback (no-hardcode rule). branch cls is under-trained
        # (to be fixed in v4); wait for a frame where it resolves. After a
        # timeout, proceed to select_stage anyway (the stage list may already
        # be showing this branch) rather than blind-clicking a fixed card pos.
        if self._loc_ticks > 8:
            self.log("location select timeout (branch cls not seen), trying select_stage anyway")
            self.sub_state = "select_stage"
            self._stage_ticks = 0
            return action_wait(300, "location select timeout")

        return action_wait(500, f"waiting for branch '{branch_name}' (cls {branch_cls})")

    def _select_stage(self, screen: ScreenState) -> Dict[str, Any]:
        """Enter the last (highest / hardest) stage from the stage list.

        Right panel shows numbered stages each with an 入场键 (STAGE_ENTER)
        button. We swipe the list to the bottom, then click the bottom-most
        unlocked 入场键 (= hardest stage). (OCR 入場 boxes → STAGE_ENTER cls.)
        """
        self._stage_ticks = getattr(self, '_stage_ticks', 0) + 1

        # 入场键 buttons on the RIGHT side (stage list). Prefer unlocked over
        # locked; if only locked present this branch isn't accessible.
        enter_btns = self.find_all_cls(screen, UC.STAGE_ENTER,
                                       conf=0.30, region=(0.60, 0.15, 1.0, 0.98))

        if enter_btns:
            # First 2 ticks: swipe stage list down to reveal the last stage.
            # Swipe (never wheel-scroll) — MuMu interprets wheel as zoom and
            # de-centers the page (user 2026-05-13).
            if self._stage_ticks <= 2:
                self.log("swiping stage list down")
                return action_swipe(0.75, 0.70, 0.75, 0.30, 500,
                                    reason="swipe stage list to bottom")

            # Click the bottom-most 入场键 (= last / hardest stage).
            last = max(enter_btns, key=lambda b: b.cy)
            self.log(f"clicking last stage 入场键 at ({last.cx:.2f},{last.cy:.2f})")
            self.sub_state = "sweep"
            self._sweep_stage = 0
            self._sweep_attempts = 0
            return action_click_box(last, "enter last stage (STAGE_ENTER)")

        # Only locked entries? This branch has no playable stage — rotate.
        locked = self.find_all_cls(screen, UC.STAGE_ENTER_LOCKED,
                                   conf=0.30, region=(0.60, 0.15, 1.0, 0.98))
        if locked and self._stage_ticks > 2:
            self.log("only locked stages on this branch, rotating")
            self._advance_branch()
            self.sub_state = "check_tickets"
            return action_wait(300, "branch locked, next branch")

        # No 入场键 visible yet — swipe to find stages.
        if self._stage_ticks <= 4:
            return action_swipe(0.75, 0.70, 0.75, 0.30, 500,
                                reason="swipe to find stages")

        # GAP: no STAGE_ENTER cls resolved after swiping. Surface it instead
        # of blind-clicking a hardcoded coord (per OCR-free spec). Rotate
        # branch so we don't get wedged on a screen we can't read.
        self.log("select_stage: no STAGE_ENTER cls found (training gap) — rotating branch")
        self._advance_branch()
        self.sub_state = "check_tickets"
        return action_wait(300, "no STAGE_ENTER cls, next branch")

    def _sweep(self, screen: ScreenState) -> Dict[str, Any]:
        """Multi-step sweep inside the 任务信息 popup.

        Flow after clicking 入场键:
        1. Popup opens → click MAX_可点击 (QTY_MAX) to set sweep count to max.
        2. Click 扫荡开始 (SWEEP_START) to start.
        3. Confirm dialog → 确认键 (BTN_CONFIRM).
        4. Dismiss 获得奖励 (GOT_REWARD) result popup.

        DIGIT-DEFERRED: MAX sweeps ALL remaining tickets in a single confirm,
        so this is normally ONE cycle per stage entry — we do NOT loop on a
        ticket count. _sweep_clicks is a hard SAFETY CAP on confirm cycles.
        """
        self._sweep_attempts += 1

        # Lost navigation: drifted off the bounty stage/popup screens to the
        # campaign hub or lobby mid-sweep → this branch is over, finish it.
        # (Prevents the t0041+ wedge where the bot sat on the hub forever
        # "looking for 扫荡开始" because the sweep flow had already exited.)
        if self._sweep_attempts > 3 and not (
            self.find_cls(screen, [UC.QTY_MAX, UC.QTY_MAX_GREY, UC.SWEEP_START], conf=0.30)
            or self.find_cls(screen, [UC.STAGE_ENTER, UC.STAGE_ENTER_LOCKED], conf=0.30)
        ):
            cur = self.detect_current_screen(screen)
            if cur in ("Mission", "Lobby"):
                self.log(f"drifted to {cur} mid-sweep — finishing branch")
                self._advance_branch()
                self.sub_state = "exit" if self._all_done else "check_tickets"
                return action_wait(300, f"drifted to {cur}, finishing branch")

        if self._sweep_attempts > 25:
            self.log("sweep stuck, rotating branch")
            self._advance_branch()
            self.sub_state = "check_tickets"
            return action_wait(300, "sweep timeout, try next branch")

        # SAFETY CAP: bail after ~10 sweep confirm cycles regardless (DIGIT-
        # DEFERRED — we can't read remaining count to stop precisely).
        if self._sweep_clicks >= 10:
            self.log("sweep safety cap (10 cycles) hit, rotating branch")
            self._advance_branch()
            self.sub_state = "check_tickets"
            return action_wait(300, "sweep cap reached, next branch")

        # Stage 0: popup open → click MAX (set sweep count to max).
        # (OCR "MAX"/"MIN"/"任務資訊" → QTY_MAX cls)
        if self._sweep_stage == 0:
            max_btn = self.find_cls(screen, UC.QTY_MAX, conf=0.30)
            if max_btn:
                self.log("popup open, clicking MAX_可点击")
                self._sweep_stage = 1
                return action_click_box(max_btn, "set sweep count MAX (QTY_MAX)")

            # MAX greyed out → already at max (count==1 or capped). Proceed.
            if self.find_cls(screen, UC.QTY_MAX_GREY, conf=0.30):
                self.log("MAX greyed (already max), proceeding to 扫荡开始")
                self._sweep_stage = 1
                return action_wait(200, "MAX grey, go to sweep start")

            # Popup not open yet — re-click the last 入场键 to (re)open it.
            re_enter = self.click_cls(
                screen, UC.STAGE_ENTER, "re-open stage popup (STAGE_ENTER)",
                conf=0.30, region=(0.60, 0.15, 1.0, 0.98))
            if re_enter:
                return re_enter
            return action_wait(400, "waiting for stage popup (QTY_MAX)")

        # Stage 1: click 扫荡开始 (SWEEP_START).
        if self._sweep_stage == 1:
            sweep_start = self.find_cls(screen, UC.SWEEP_START, conf=0.30)
            if sweep_start:
                # NOTE: no 扫荡开始_灰 cls trained, so a ticket-less greyed
                # button can't be cls-detected (count is DIGIT-DEFERRED). Per
                # the no-HSV spec we don't pixel-check; one MAX sweep + one-pass
                # branch exit bounds it instead. (TRAIN: 扫荡开始_灰 cls.)
                self.log("clicking 扫荡开始 (SWEEP_START)")
                self._sweep_stage = 2
                return action_click_box(sweep_start, "click 扫荡开始 (SWEEP_START)")

            # SWEEP_START gone entirely after we already swept at least once
            # → finished. (DIGIT-DEFERRED termination signal.)
            if self._sweep_clicks > 0:
                self.log("扫荡开始 gone after sweep — branch done")
                self._advance_branch()
                self.sub_state = "check_tickets"
                return action_wait(300, "sweep start gone, next branch")
            return action_wait(400, "looking for 扫荡开始 (SWEEP_START)")

        # Stage 2: confirm sweep dialog → 确认键 (BTN_CONFIRM).
        # (OCR "確認/确认/確定..." → BTN_CONFIRM cls)
        if self._sweep_stage == 2:
            confirm = self.find_cls(screen, UC.BTN_CONFIRM, conf=0.30,
                                    region=(0.30, 0.30, 0.95, 0.95))
            if confirm:
                self._sweep_clicks += 1
                self.log(f"confirming sweep (cycle {self._sweep_clicks})")
                self._sweep_stage = 3
                return action_click_box(confirm, "confirm sweep (BTN_CONFIRM)")

            # Confirm dialog handled out-of-band (by _handle_common_popups
            # once migrated, or auto-dismissed) → result popup appears.
            if self.find_cls(screen, UC.GOT_REWARD, conf=0.30):
                self._sweep_stage = 3
                return action_wait(200, "result popup appeared")

            # If the 任务信息 popup is back (QTY_MAX visible again), the sweep
            # already completed and the dialog/result were dismissed for us.
            if self._sweep_attempts > 8 and self.find_cls(screen, UC.QTY_MAX, conf=0.30):
                self.log("sweep completed (popup returned)")
                self._advance_branch()
                self.sub_state = "check_tickets"
                return action_back("close popup after sweep")

            return action_wait(400, "waiting for confirm dialog (BTN_CONFIRM)")

        # Stage 3+: dismiss 获得奖励 result, then loop / rotate branch.
        if self._sweep_stage >= 3:
            got = self.find_cls(screen, UC.GOT_REWARD, conf=0.30)
            if got:
                done_branch = self._branches[self._branch_idx][0]
                self.log(f"sweep done on '{done_branch}', dismissing result")
                self._advance_branch()
                self.sub_state = "check_tickets"
                return action_click(0.5, 0.9, "dismiss sweep result (GOT_REWARD)")

            # A confirm dialog may also appear here (BTN_CONFIRM) on some
            # sweep result layouts.
            confirm = self.find_cls(screen, UC.BTN_CONFIRM, conf=0.30,
                                    region=(0.30, 0.30, 0.95, 0.95))
            if confirm:
                self._advance_branch()
                self.sub_state = "check_tickets"
                return action_click_box(confirm, "dismiss sweep result (BTN_CONFIRM)")

            # Nothing recognized — tap center-bottom to dismiss, then rotate.
            self._advance_branch()
            self.sub_state = "check_tickets"
            return action_click(0.5, 0.9, "tap to dismiss sweep result")

        return action_wait(400, "sweep processing")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if screen.is_lobby():
            self.log("done")
            return action_done("bounty complete")
        # If we're back on the campaign hub (Mission screen), STOP exiting —
        # Arena/other campaign skills enter via the same hub. Exiting to
        # lobby just to re-enter wastes 10-20 ticks per round-trip
        # (user 2026-05-13). Detect the hub via YOLO signature.
        current = self.detect_current_screen(screen)
        if current == "Mission":
            self.log("done (left on campaign hub for next skill)")
            return action_done("bounty complete (on hub)")
        # A leftover "掃蕩完成" result dialog blocks the way out — ESC does NOT
        # dismiss it (verified t22-27: 6× back stuck on 确认键). Click its
        # confirm / X button FIRST, then continue exiting. (bug 2026-06-01)
        confirm = self.find_cls(screen, UC.BTN_CONFIRM, conf=0.30,
                                region=(0.30, 0.55, 0.70, 0.85))
        if confirm:
            return action_click_box(confirm, "bounty exit: dismiss result dialog (确认键)")
        x_btn = self.find_cls(screen, UC.BTN_CLOSE_X, conf=0.30)
        if x_btn:
            return action_click_box(x_btn, "bounty exit: close result dialog (X)")
        return action_back("bounty exit: back to lobby")
