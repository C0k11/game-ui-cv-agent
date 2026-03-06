"""ArenaSkill: Handle 戰術對抗賽 (Tactical PvP Arena) daily routine.

Flow:
1. ENTER: From lobby, click 戰術對抗賽
2. CLAIM_REWARDS: Claim daily ranking rewards
3. CHECK_TICKETS: Parse remaining fight count (X/5)
4. SELECT_OPPONENT: Pick bottommost opponent (easiest win or fast loss)
5. FIGHT: Click 出擊 → skip battle → handle result
6. Loop back to CHECK_TICKETS until 0
7. EXIT: Back to lobby

Key patterns:
- Header: "戰術對抗賽" / "战术对抗赛"
- Rewards: "領取獎勵" / "领取奖励"
- Ticket: "剩余 X/5" or "X/5"
- Battle: "出擊" / "出击" / "跳過" / "Skip"
"""
from __future__ import annotations

import re
from typing import Any, Dict

from brain.skills.base import (
    BaseSkill, ScreenState,
    action_click, action_click_box, action_click_yolo,
    action_wait, action_back, action_done,
)


class ArenaSkill(BaseSkill):
    def __init__(self):
        super().__init__("Arena")
        self.max_ticks = 200  # increased for cooldown waits
        self._tickets_remaining: int = -1
        self._fights_done: int = 0
        self._claim_ticks: int = 0
        self._claim_clicks: int = 0
        self._fight_stage: int = 0  # 0=opponent popup, 1=formation, 2=battle
        self._fight_ticks: int = 0
        self._cooldown_ticks: int = 0  # wait ticks after a fight (30s cooldown)

    def reset(self) -> None:
        super().reset()
        self._tickets_remaining = -1
        self._fights_done = 0
        self._claim_ticks = 0
        self._claim_clicks = 0
        self._fight_stage = 0
        self._fight_ticks = 0
        self._cooldown_ticks = 0

    def _is_arena(self, screen: ScreenState) -> bool:
        return self._is_arena_screen(screen)

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1

        if self.ticks >= self.max_ticks:
            self.log("timeout")
            return action_done("arena timeout")

        # Battle result — dismiss and loop back to check_tickets
        battle_result = screen.find_any_text(
            ["戰鬥結果", "战斗结果"],
            min_conf=0.6
        )
        if not battle_result:
            battle_result = screen.find_any_text(
                ["勝利", "胜利", "敗北", "败北", "Victory", "Defeat",
                 "WIN", "LOSE"],
                min_conf=0.6
            )
        if battle_result:
            self.log(f"battle result: {battle_result.text}")
            confirm = screen.find_any_text(
                ["確認", "确认", "確定", "确定", "確", "确", "OK"],
                min_conf=0.6
            )
            if confirm:
                self._fight_stage = 0
                self._fight_ticks = 0
                self._fights_done += 1
                self.sub_state = "check_tickets"
                return action_click_box(confirm, "dismiss battle result")
            return action_click(0.5, 0.9, "tap to dismiss battle result")

        # EXP bar / rank-up screen — tap to dismiss
        exp_screen = screen.find_any_text(
            ["經驗值", "经验值", "EXP", "Touch to Continue"],
            min_conf=0.6
        )
        if exp_screen:
            return action_click(0.5, 0.5, "dismiss exp/rank screen")

        popup = self._handle_common_popups(screen)
        if popup:
            return popup

        if screen.is_loading():
            return action_wait(800, "arena loading")

        # If 對戰對象 popup is open but we're NOT in fight mode, close it.
        # This happens when a double-click on campaign entry accidentally opens
        # the popup with the wrong (bottom) opponent.
        if self.sub_state not in ("fight", ""):
            vs_marker = screen.find_any_text(
                ["VS"], region=(0.35, 0.35, 0.65, 0.55), min_conf=0.8
            )
            if vs_marker:
                self.log("opponent popup unexpectedly open, closing")
                close_btn = self._find_florence_hit(
                    screen,
                    ["close button icon", "close dialog x button", "x close icon"],
                    region=(0.62, 0.06, 0.94, 0.28),
                )
                if close_btn:
                    return action_click_box(close_btn, "close opponent popup")
                return action_back("close opponent popup")

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

    def _is_arena_screen(self, screen: ScreenState) -> bool:
        """Detect arena screen via unique markers (header OCR is unreliable).

        Arena screen has: "對戰對象" / "對手情報", "攻擊編制",
        "持有票券" with X/5 pattern.
        """
        # Check header region for any arena-like text (including partial OCR)
        header = screen.find_any_text(
            ["戰術大賽", "战术大赛", "戰術對抗", "战术对抗",
             "術大赛", "術大賽", "大賽", "大赛"],
            region=(0.0, 0.0, 0.3, 0.08), min_conf=0.5
        )
        if header:
            return True
        # Fallback: arena-specific UI elements
        opponent = screen.find_any_text(
            ["對戰對象", "对战对象", "對對象", "對手情報", "对手情报"],
            region=(0.2, 0.08, 0.7, 0.50), min_conf=0.7
        )
        if opponent:
            return True
        attack = screen.find_any_text(
            ["攻擊編制", "攻击编制", "攻撃编制", "攻擎编制", "攻擎編制"],
            region=(0.3, 0.70, 0.7, 0.90), min_conf=0.5
        )
        if attack:
            return True
        return False

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        self._enter_ticks = getattr(self, '_enter_ticks', 0) + 1
        current = self.detect_current_screen(screen)

        # Direct arena screen detection (header OCR often fails)
        if current == "PVP" or self._is_arena_screen(screen):
            self.log("inside arena")
            self.sub_state = "claim_rewards"
            return action_wait(500, "entered arena")

        # Accidentally entered Daily Tasks? Back out.
        if current == "DailyTasks":
            return action_back("back from Daily Tasks")

        if current == "Lobby":
            # Arena is NOT in the bottom nav bar.
            # Must enter via campaign menu: click right-side 任務 button → then 對抗賽
            campaign_btn = screen.find_any_text(
                ["任務", "任务"],
                region=(0.80, 0.70, 1.0, 0.90), min_conf=0.6
            )
            if campaign_btn:
                self.log(f"clicking campaign entry '{campaign_btn.text}'")
                return action_click_box(campaign_btn, "click campaign entry for arena")
            return action_click(0.95, 0.83, "click campaign area (hardcoded)")

        if current == "Mission":
            # Inside campaign menu — find 戰術大賽 (PvP Arena)
            # OCR frequently misreads: "戰術大賽" → "术大赛" / "戰術" / "大賽" / partial
            pvp = screen.find_any_text(
                ["戰術大賽", "战术大赛", "對抗", "对抗", "大賽", "大赛",
                 "术大赛", "戰術", "PVP", "Arena"],
                min_conf=0.5
            )
            if pvp:
                self.log(f"clicking arena '{pvp.text}'")
                return action_click_box(pvp, "click arena in campaign")

            # Hardcoded fallback: 戰術大賽 position in campaign grid (~0.73, 0.81)
            if self._enter_ticks > 5:
                self.log("clicking arena at hardcoded position")
                return action_click(0.73, 0.81, "click arena (hardcoded)")

            return action_wait(500, "looking for arena in campaign menu")

        if self._enter_ticks > 20:
            self.log("can't reach arena, giving up")
            self.sub_state = "exit"
            return action_wait(300, "arena enter timeout")

        if current and current != "PVP":
            return action_back(f"back from {current}")

        return action_wait(500, "entering arena")

    def _claim_rewards(self, screen: ScreenState) -> Dict[str, Any]:
        """Click all 領取獎勵 buttons on the left panel (time + daily rewards).

        Arena layout (left panel, x < 0.35):
          - 時間獎勵 +80/分  →  領取獎勵  (累積量 reward, y≈0.52)
          - 每日獎勵 --:--    →  領取獎勵  (持有票券 reward, y≈0.64)
        OCR often reads the yellow buttons as just "领取" (2 chars), not "領取獎勵".

        Strategy: click bottom (daily) first, then click top (time) once.
        The time reward regenerates instantly, so limit to max 3 total clicks
        to avoid wasting ticks in an infinite claim loop.
        """
        self._claim_ticks += 1

        # Hard limit: 3 actual clicks is enough (1 daily + 1 time + 1 buffer)
        if self._claim_clicks >= 3 or self._claim_ticks > 10:
            self.log(f"reward claim done ({self._claim_clicks} clicks)")
            self.sub_state = "check_tickets"
            return action_wait(300, "claim rewards done")

        # Find ALL 領取/领取 buttons in the left panel (x < 0.35).
        # OCR reads "领取" not "領取獎勵", so search for the short form.
        claims = screen.find_text(
            "領取|领取",
            region=(0.15, 0.40, 0.40, 0.75), min_conf=0.5
        )
        if claims:
            # Click BOTTOM first (sort by y descending).
            # Reason: the top button (累積量 time reward) regenerates instantly
            # after claiming, so clicking topmost first causes an infinite loop.
            # The bottom button (持有票券 daily reward) only appears once per day.
            claims.sort(key=lambda b: b.cy, reverse=True)
            claim = claims[0]
            self._claim_clicks += 1
            self.log(f"claiming reward #{self._claim_clicks}: '{claim.text}' at y={claim.cy:.2f} ({len(claims)} buttons)")
            return action_click_box(claim, "claim arena reward")

        # No claim buttons visible — either all claimed or popup blocking
        if self._claim_clicks > 0:
            # Already clicked at least once, wait for popup to clear
            return action_wait(400, "waiting for reward popup to clear")

        # No rewards found at all — move on
        self.sub_state = "check_tickets"
        return action_wait(300, "no rewards to claim")

    def _check_tickets(self, screen: ScreenState) -> Dict[str, Any]:
        """Check remaining arena fight tickets.

        OCR reads "持有票券 5/5". Must filter by "票券" to avoid matching
        "累量0/1,000K" (accumulated points) which regex reads as 0/1.

        Also handles 30-second cooldown between fights: if "等待時間" shows
        an active timer (not --:--), wait before selecting next opponent.
        """
        # ── Cooldown check (30s between fights) ──
        if self._fights_done > 0:
            wait_time = screen.find_any_text(
                ["等待時間", "等待时间"],
                region=(0.0, 0.55, 0.55, 0.80), min_conf=0.5
            )
            if wait_time:
                # Check if timer is active (not --:--)
                # Look for digits NEAR the "等待時間" label (within ±0.04 y).
                # Must not match "每日02:02" (daily reward timer at y≈0.61).
                wt_y = wait_time.cy  # y of "等待時間" label (~0.73)
                for box in screen.ocr_boxes:
                    if box.confidence < 0.4:
                        continue
                    if abs(box.cy - wt_y) > 0.04:
                        continue  # Must be same row as 等待時間
                    if box.cx > 0.55:
                        continue
                    # Active timer: contains digits like "0:28" or "00:28"
                    timer_match = re.search(r'(\d+):(\d+)', box.text)
                    if timer_match and "--" not in box.text:
                        mins = int(timer_match.group(1))
                        secs = int(timer_match.group(2))
                        total_secs = mins * 60 + secs
                        # Cap at 60s — cooldown is 30s, anything longer is a misread
                        if 0 < total_secs <= 60:
                            self._cooldown_ticks += 1
                            if self._cooldown_ticks > 15:
                                self.log("cooldown wait too long, forcing proceed")
                                self._cooldown_ticks = 0
                                break
                            self.log(f"cooldown active: {mins}:{secs:02d} (tick {self._cooldown_ticks})")
                            return action_wait(3000, f"arena cooldown {mins}:{secs:02d}")
            # Reset cooldown counter once timer is gone
            self._cooldown_ticks = 0

        for box in screen.ocr_boxes:
            if box.confidence < 0.5:
                continue
            m = re.search(r'(\d+)\s*/\s*(\d+)', box.text)
            if not m:
                continue
            # Must contain 票券 to be the ticket count
            if "票券" not in box.text:
                continue
            remaining = int(m.group(1))
            total = int(m.group(2))
            self._tickets_remaining = remaining
            self.log(f"arena tickets: {remaining}/{total}")
            if remaining == 0:
                self.log("no tickets left, exiting")
                self.sub_state = "exit"
                return action_wait(300, "no arena tickets")
            self.sub_state = "select_opponent"
            return action_wait(300, f"{remaining} fights remaining")

        # Couldn't parse — try to proceed anyway
        if self.ticks > 15:
            self.sub_state = "select_opponent"
            return action_wait(300, "ticket parse timeout, trying to fight")

        return action_wait(500, "looking for ticket count")

    def _select_opponent(self, screen: ScreenState) -> Dict[str, Any]:
        """Pick the top-ranked opponent (lowest 第XXX名 number).

        Arena layout: left panel (player info) + right panel (opponent list).
        Opponents show 第XXX名 (rank) at x > 0.35.
        Player's own rank is at x < 0.30.
        """
        # Find all 第XXX名 patterns in the right half
        candidates = []
        for box in screen.ocr_boxes:
            if box.confidence < 0.7 or box.cx < 0.35:
                continue  # Skip left panel (player's own rank)
            m = re.search(r'第(\d+)名', box.text)
            if m:
                rank = int(m.group(1))
                candidates.append((rank, box))

        if candidates:
            # Pick lowest rank number = highest position
            candidates.sort(key=lambda x: x[0])
            best_rank, best_box = candidates[0]
            self.log(f"selecting opponent #{best_rank}")
            self.sub_state = "fight"
            self._fight_stage = 0
            self._fight_ticks = 0
            return action_click_box(best_box, f"select opponent #{best_rank}")

        # Fallback: click top-right area where first opponent row is
        self.log("selecting top opponent (fallback)")
        self.sub_state = "fight"
        self._fight_stage = 0
        self._fight_ticks = 0
        return action_click(0.65, 0.30, "select top opponent (fallback)")

    def _fight(self, screen: ScreenState) -> Dict[str, Any]:
        """Handle the fight sequence:

        Stage 0: 對戰對象 popup → click 攻擊編制
        Stage 1: Formation screen → click 出擊
        Stage 2: Battle in progress (auto/skip, result handled in tick())
        """
        self._fight_ticks += 1

        # Stage 2 (battle) can take 40+ ticks — use higher timeout.
        # NEVER action_back during stage 2: it triggers exit dialog → lobby.
        max_ticks = 70 if self._fight_stage >= 2 else 40
        if self._fight_ticks > max_ticks:
            self.log(f"fight timeout (stage={self._fight_stage})")
            self.sub_state = "check_tickets"
            self._fight_ticks = 0
            if self._fight_stage >= 2:
                # Battle likely ended but result was missed — just wait
                return action_wait(500, "fight timeout (battle)")
            return action_back("fight timeout")

        # Stage 0: 對戰對象 popup — click 攻擊編制 button
        if self._fight_stage == 0:
            # OCR frequently misreads 擊 as 擎 ("攻擎编制" instead of "攻擊編制")
            attack_form = screen.find_any_text(
                ["攻擊編制", "攻击编制", "攻撃编制", "攻擊編", "攻击编",
                 "攻擎编制", "攻擎編制"],  # OCR misreads 擊 as 擎
                min_conf=0.5
            )
            if not attack_form:
                # Region-filtered partial match for bottom-center button area
                attack_form = screen.find_any_text(
                    ["編制", "编制"],
                    region=(0.3, 0.70, 0.7, 0.90), min_conf=0.6
                )
            if attack_form:
                self.log("clicking 攻擊編制")
                self._fight_stage = 1
                return action_click_box(attack_form, "click 攻擊編制")

            # If still on arena main screen (no popup), re-select opponent
            if self._fight_ticks > 3:
                if self._is_arena_screen(screen):
                    self.sub_state = "select_opponent"
                    return action_wait(300, "popup didn't open, re-selecting")
                # After 10 ticks without progress, go back to check_tickets
                # to re-establish arena state (screen may have refreshed)
                if self._fight_ticks > 10:
                    self.sub_state = "check_tickets"
                    self._fight_ticks = 0
                    return action_wait(500, "fight stuck, re-checking tickets")

            return action_wait(500, "waiting for 對戰對象 popup")

        # Stage 1: Formation screen — click 出擊
        if self._fight_stage == 1:
            # OCR frequently misreads 擊 as 擎 ("出擎" instead of "出擊")
            sortie = screen.find_any_text(
                ["出擊", "出击", "出擎"],
                min_conf=0.6
            )
            if sortie:
                self.log("clicking 出擊")
                self._fight_stage = 2
                return action_click_box(sortie, "click 出擊")

            # Check for 跳過戰鬥 checkbox — it's on the formation screen
            # but we just need to click 出擊, not the checkbox
            return action_wait(500, "waiting for formation screen")

        # Stage 2: Battle in progress — result handled globally in tick()
        if self._fight_stage >= 2:
            # Try SKIP during battle (Skip button, not 跳過戰鬥 checkbox)
            skip = screen.find_any_text(
                ["SKIP", "Skip"],
                min_conf=0.7
            )
            if skip:
                return action_click_box(skip, "skip battle")
            return action_wait(1000, "battle in progress")

        return action_wait(500, "fight processing")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if screen.is_lobby():
            self.log(f"done ({self._fights_done} fights, {self._claim_clicks} rewards)")
            return action_done("arena complete")
        return action_back("arena exit: back to lobby")
