from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DailyRoutineStep:
    name: str
    instruction: str
    exit_condition: str


class DailyRoutineManager:
    def __init__(self) -> None:
        self.steps: List[DailyRoutineStep] = self._build_steps()
        self.current_step_index: int = 0
        self.is_active: bool = False

        self.max_turns_per_attempt: int = 15
        self.max_retries: int = 1

        self.turns_in_current_attempt: int = 0
        self.current_step_retry_count: int = 0
        self.in_recovery_mode: bool = False

    def _build_steps(self) -> List[DailyRoutineStep]:
        return [
            DailyRoutineStep(
                name="Check Lobby",
                instruction=(
                    "Goal: Ensure you are in the Main Lobby. If there are any popups (Check-in, Announcements), close them. If you see 'Cafe' or 'Schale', you are ready.\n"
                    "IMPORTANT: Do NOT click on 'Tasks', 'Mail', 'Club', 'Schedule', 'Recruit' or 'Bounties' during this phase, even if they have Red Dots/Notifications. IGNORE them. Only handle popups.",
                ),
                exit_condition="In Lobby and no popups blocking view.",
            ),
            DailyRoutineStep(
                name="Cafe",
                instruction=(
                    "Navigate to 'Cafe'. Inside Cafe: 1. Claim Earnings (Credits/AP button). "
                    "2. Tap all students with yellow interactive marks (head pats). "
                    "3. Return to Lobby when done."
                ),
                exit_condition="Returned to Lobby after claiming rewards.",
            ),
            DailyRoutineStep(
                name="Schedule",
                instruction=(
                    "Navigate to 'Schedule'. Use all available tickets (Tickets: X/X). "
                    "Priority: Locations with highest rank or red notifications. After using tickets, Return to Lobby."
                ),
                exit_condition="Returned to Lobby after using tickets.",
            ),
            DailyRoutineStep(
                name="Club",
                instruction=(
                    "Navigate to 'Club'. Enter the Club Lobby, Claim the daily AP (10 AP). Return to Lobby."
                ),
                exit_condition="Returned to Lobby after claiming AP.",
            ),
            DailyRoutineStep(
                name="Bounties",
                instruction=(
                    "Navigate to 'Campaign' -> 'Bounty' (or Wanted). Sweep (Clear) the highest level available for Highway/Desert/Classroom if you have tickets. "
                    "Do NOT use Pyroxenes. Return to Lobby."
                ),
                exit_condition="Returned to Lobby after sweeping.",
            ),
            DailyRoutineStep(
                name="AP Dump",
                instruction=(
                    "Spend AP (stamina) safely. Prefer Event if a clear Event banner exists: enter Event -> Quest/Stage -> choose the recommended farming stage -> Sweep using available AP. "
                    "If no Event, go to Campaign -> Hard and find a bookmarked/pinned/favorited stage (star/pin icon) then Sweep/Auto-clear until AP is insufficient. "
                    "Do NOT use Pyroxenes. When you cannot sweep more, return to Lobby."
                ),
                exit_condition="AP spent (cannot sweep more) and back in Lobby.",
            ),
            DailyRoutineStep(
                name="Mail & Tasks",
                instruction=(
                    "1. Go to Mailbox -> Claim All -> Confirm. 2. Go to Tasks -> Claim All -> Confirm. Return to Lobby."
                ),
                exit_condition="All rewards claimed and back in Lobby.",
            ),
        ]

    def start_routine(self) -> None:
        print(f"[RoutineManager] Starting Daily Routine with {len(self.steps)} steps.")
        self.is_active = True
        self.current_step_index = 0

        self._reset_step_state()

    def stop_routine(self) -> None:
        print("[RoutineManager] Routine Stopped.")
        self.is_active = False
        self.current_step_index = 0

        self._reset_step_state()

    def _reset_step_state(self) -> None:
        self.turns_in_current_attempt = 0
        self.current_step_retry_count = 0
        self.in_recovery_mode = False

    def on_turn_start(self) -> None:
        if not self.is_active:
            return

        self.turns_in_current_attempt += 1

        if self.turns_in_current_attempt <= int(self.max_turns_per_attempt):
            return

        step_name = "Unknown"
        try:
            step = self.get_current_step()
            if step is not None:
                step_name = step.name
        except Exception:
            step_name = "Unknown"

        mode = "Recovery" if self.in_recovery_mode else "Normal"
        print(f"[RoutineManager] Timeout detected in '{step_name}' (Mode: {mode}).")

        if self.in_recovery_mode:
            print(f"[RoutineManager] Recovery failed. FORCING SKIP of step '{step_name}'.")
            self._force_skip()
            return

        if self.current_step_retry_count < int(self.max_retries):
            print(f"[RoutineManager] Triggering RECOVERY to Lobby. Will retry '{step_name}' afterwards.")
            self.in_recovery_mode = True
            self.turns_in_current_attempt = 0
            return

        print(f"[RoutineManager] Max retries reached for '{step_name}'. SKIPPING.")
        self._force_skip()

    def _force_skip(self) -> None:
        self.in_recovery_mode = False
        self.advance_step(force=True)

    def handle_done_signal(self) -> bool:
        if not self.is_active:
            return False

        if self.in_recovery_mode:
            print("[RoutineManager] Recovery successful (Back in Lobby). Retrying original step.")
            self.in_recovery_mode = False
            self.turns_in_current_attempt = 0
            self.current_step_retry_count += 1
            return True

        step_name = "Unknown"
        try:
            step = self.get_current_step()
            if step is not None:
                step_name = step.name
        except Exception:
            step_name = "Unknown"

        print(f"[RoutineManager] Step '{step_name}' completed success.")
        return self.advance_step(force=False)

    def get_current_step(self) -> Optional[DailyRoutineStep]:
        if not self.is_active:
            return None
        if self.current_step_index >= len(self.steps):
            return None
        return self.steps[self.current_step_index]

    def advance_step(self, force: bool = False) -> bool:
        if self.is_active:
            self.current_step_index += 1

            self.turns_in_current_attempt = 0
            self.current_step_retry_count = 0
            self.in_recovery_mode = False

            if self.current_step_index >= len(self.steps):
                print("[RoutineManager] All steps completed!")
                self.is_active = False
                return False

            verb = "Skipping to" if force else "Advancing to"
            print(f"[RoutineManager] {verb} step {self.current_step_index + 1}: {self.steps[self.current_step_index].name}")
            return True
        return False

    def get_progress_str(self) -> str:
        if not self.is_active:
            return "Idle"
        return f"Step {self.current_step_index + 1}/{len(self.steps)}: {self.steps[self.current_step_index].name}"

    def get_prompt_block(self) -> str:
        if not self.is_active:
            return ""

        if self.in_recovery_mode:
            return (
                "\n\n*** RECOVERY MODE ACTIVATED ***\n"
                "The previous step TIMED OUT. We need to reset.\n"
                "CURRENT GOAL: Navigate BACK to the Main Lobby immediately.\n"
                "INSTRUCTION: Click Back/Home or close windows until you see the Lobby (Cafe/Schale visible).\n"
                "EXIT CONDITION: When you clearly see the Lobby, output action: 'done'.\n"
                f"Turns Spent: {self.turns_in_current_attempt}/{self.max_turns_per_attempt}\n"
            )

        step = self.get_current_step()
        if step is None:
            return ""

        retry_info = ""
        if self.current_step_retry_count > 0:
            retry_info = f"(Retry {self.current_step_retry_count}/{self.max_retries})"
        return (
            f"\n\n*** DAILY ROUTINE EXECUTION {retry_info} ***\n"
            f"Phase: {step.name} ({self.current_step_index + 1}/{len(self.steps)})\n"
            f"Goal: {step.instruction}\n"
            f"Exit Condition: {step.exit_condition}\n"
            f"Turns Spent: {self.turns_in_current_attempt}/{self.max_turns_per_attempt}\n"
            "WHEN FINISHED: If met Exit Condition, output action: 'done'.\n"
        )
