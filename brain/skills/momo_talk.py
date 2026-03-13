"""MomoTalkSkill: Auto-complete all unread MomoTalk conversations.

reference-based flow:
1. Navigate to MomoTalk from lobby sidebar icon
2. Set sort to "unread" and direction to descending
3. Detect unread conversation indicators (red notification dots)
4. Click each unread → auto-complete dialogue via state machine
5. Re-scan after completing (new messages may appear)
6. Exit when no more unread messages

Key OCR patterns:
- Header: "MomoTalk"
- Sort: "未讀" / "未读" (unread), "最新" (newest)
- Dialogue: "回覆"/"回复" (reply), story buttons, SKIP
- Unread dots: red circles at x≈0.49 in conversation list
"""
from __future__ import annotations

from typing import Any, Dict

from brain.skills.base import (
    BaseSkill, ScreenState,
    action_click, action_click_box,
    action_wait, action_back, action_done, action_swipe,
)


class MomoTalkSkill(BaseSkill):
    # Conversation list slot Y positions (normalized, for 1280x720 base)
    # reference scans y from 210 to 620 at x=637 (≈0.497) for red unread dots.
    # We approximate with discrete slot centers.
    _SLOT_YS = [0.32, 0.40, 0.48, 0.56, 0.64, 0.72, 0.80, 0.88]

    def __init__(self):
        super().__init__("MomoTalk")
        self.max_ticks = 100
        self._conversations_completed: int = 0
        self._scan_ticks: int = 0
        self._dialogue_ticks: int = 0
        self._sort_set: bool = False
        self._scan_rounds: int = 0
        self._click_slot_idx: int = 0
        self._enter_ticks: int = 0
        self._story_mode: bool = False

    def reset(self) -> None:
        super().reset()
        self._conversations_completed = 0
        self._scan_ticks = 0
        self._dialogue_ticks = 0
        self._sort_set = False
        self._scan_rounds = 0
        self._click_slot_idx = 0
        self._enter_ticks = 0
        self._story_mode = False

    def _is_momotalk(self, screen: ScreenState) -> bool:
        return (
            screen.has_text("MomoTalk", region=(0.0, 0.0, 0.50, 0.15), min_conf=0.5)
            or screen.has_text("Momo", region=(0.0, 0.0, 0.50, 0.15), min_conf=0.5)
        )

    def _is_conversation_view(self, screen: ScreenState) -> bool:
        """Detect if we're inside a conversation (right panel has chat bubbles)."""
        reply = screen.find_any_text(
            ["回覆", "回复", "Reply"],
            region=(0.50, 0.20, 1.0, 0.95), min_conf=0.45
        )
        if reply:
            return True
        # Check for relationship story button (pink "前往羈絆劇情")
        story_btn = screen.find_any_text(
            ["羈絆", "羁绊", "前往", "To Relationship"],
            region=(0.50, 0.60, 1.0, 0.95), min_conf=0.45
        )
        if story_btn:
            return True
        return False

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1

        if self.ticks >= self.max_ticks:
            self.log(f"timeout ({self._conversations_completed} conversations completed)")
            return action_done("momotalk timeout")

        # Reward popup — can appear after story completion
        reward = screen.find_any_text(
            ["獲得道具", "获得道具", "獲得獎勵", "获得奖励"],
            region=screen.CENTER, min_conf=0.6
        )
        if reward:
            return action_click(0.5, 0.9, "dismiss reward popup")

        popup = self._handle_common_popups(screen)
        if popup:
            return popup

        if screen.is_loading():
            return action_wait(800, "momotalk loading")

        if self.sub_state == "":
            self.sub_state = "enter"

        if self.sub_state == "enter":
            return self._enter(screen)
        if self.sub_state == "set_sort":
            return self._set_sort(screen)
        if self.sub_state == "scan":
            return self._scan(screen)
        if self.sub_state == "dialogue":
            return self._dialogue(screen)
        if self.sub_state == "story":
            return self._story(screen)
        if self.sub_state == "exit":
            return self._exit(screen)

        return action_wait(300, "momotalk unknown state")

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        self._enter_ticks += 1

        if self._is_momotalk(screen):
            self.log("inside MomoTalk")
            if not self._sort_set:
                self.sub_state = "set_sort"
            else:
                self.sub_state = "scan"
                self._scan_ticks = 0
            return action_wait(400, "entered MomoTalk")

        current = self.detect_current_screen(screen)
        if current == "Lobby":
            # MomoTalk is accessed via the left sidebar icon on lobby.
            # reference uses pixel-based navigation; we use OCR + hardcoded fallback.
            momo = screen.find_any_text(
                ["MomoTalk", "Momo"],
                region=(0.0, 0.08, 0.22, 0.28), min_conf=0.4
            )
            if momo:
                return action_click_box(momo, "click MomoTalk sidebar icon")
            # Hardcoded: MomoTalk icon at ~(0.11, 0.18) on lobby sidebar
            return action_click(0.11, 0.18, "click MomoTalk icon (hardcoded)")

        if current and current != "MomoTalk":
            return action_back(f"back from {current}")

        if self._enter_ticks > 15:
            self.log("can't reach MomoTalk, skipping")
            return action_done("momotalk unavailable")

        return action_wait(500, "entering MomoTalk")

    def _set_sort(self, screen: ScreenState) -> Dict[str, Any]:
        """Set sort mode to 'unread' descending.

        reference clicks the sort dropdown (peach icon at ~511,177) → selects 'unread'
        → then clicks the direction toggle. We use OCR to find sort options.
        """
        if not self._is_momotalk(screen):
            return action_wait(400, "waiting for MomoTalk to set sort")

        # Check if sort menu is open (has sort option text visible)
        sort_menu = screen.find_any_text(
            ["未讀", "未读", "最新", "名稱", "名称", "好感度", "好感"],
            region=(0.20, 0.30, 0.55, 0.65), min_conf=0.5
        )
        if sort_menu:
            # Click "未讀" (unread) sort option
            unread_opt = screen.find_any_text(
                ["未讀", "未读"],
                region=(0.30, 0.30, 0.55, 0.55), min_conf=0.5
            )
            if unread_opt:
                self.log("selecting 'unread' sort")
                self._sort_set = True
                self.sub_state = "scan"
                self._scan_ticks = 0
                return action_click_box(unread_opt, "select unread sort")
            # Fallback: click unread position (~555/1280, 296/720 ≈ 0.434, 0.411)
            self._sort_set = True
            self.sub_state = "scan"
            self._scan_ticks = 0
            return action_click(0.434, 0.411, "select unread sort (hardcoded)")

        # Open sort menu by clicking the sort dropdown
        # reference: sort dropdown "peach icon" at (511,177) ≈ (0.399, 0.246)
        sort_btn = screen.find_any_text(
            ["排序", "Sort"],
            region=(0.30, 0.18, 0.50, 0.30), min_conf=0.4
        )
        if sort_btn:
            return action_click_box(sort_btn, "open sort menu")
        # Hardcoded sort dropdown position
        return action_click(0.399, 0.246, "open sort menu (hardcoded)")

    def _scan(self, screen: ScreenState) -> Dict[str, Any]:
        """Scan for unread conversations and click the first one found."""
        self._scan_ticks += 1

        # If we ended up in a dialogue, switch state
        if self._is_conversation_view(screen) and self._is_momotalk(screen):
            self.sub_state = "dialogue"
            self._dialogue_ticks = 0
            return action_wait(300, "conversation detected during scan")

        if not self._is_momotalk(screen):
            # Might have entered a story or left MomoTalk
            if self._scan_ticks > 3:
                self.sub_state = "enter"
                self._enter_ticks = 0
            return action_wait(500, "waiting for MomoTalk UI")

        # Look for unread notification indicators.
        # reference checks pixel color at x=637 (~0.497) for red dots (R:241-255, G:61-81, B:15-35).
        # In our OCR pipeline, unread numbers appear as small digits near x≈0.49.
        # We look for small red number indicators or "NEW" text in the conversation list.
        unread = screen.find_any_text(
            ["NEW"],
            region=(0.42, 0.25, 0.55, 0.92), min_conf=0.5
        )
        if unread:
            self.log(f"unread indicator found at y={unread.cy:.2f}")
            self.sub_state = "dialogue"
            self._dialogue_ticks = 0
            return action_click(0.35, unread.cy, "click unread conversation")

        # Look for notification badge numbers (1, 2, 3, etc.) on the right side
        for box in screen.ocr_boxes:
            if box.confidence < 0.5:
                continue
            if 0.44 < box.cx < 0.54 and 0.25 < box.cy < 0.92:
                if box.text.strip().isdigit() and 1 <= int(box.text.strip()) <= 99:
                    self.log(f"unread badge '{box.text}' at y={box.cy:.2f}")
                    self.sub_state = "dialogue"
                    self._dialogue_ticks = 0
                    return action_click(0.35, box.cy, "click conversation with badge")

        # Try clicking conversation slots sequentially
        if self._click_slot_idx < len(self._SLOT_YS):
            slot_y = self._SLOT_YS[self._click_slot_idx]
            self._click_slot_idx += 1
            return action_click(0.35, slot_y, f"try conversation slot {self._click_slot_idx}")

        # Scroll down and retry
        if self._scan_rounds < 2:
            self._scan_rounds += 1
            self._click_slot_idx = 0
            self._scan_ticks = 0
            return action_swipe(0.35, 0.80, 0.35, 0.30, 400, "scroll conversation list")

        self.log(f"no more unread conversations ({self._conversations_completed} completed)")
        self.sub_state = "exit"
        return action_wait(300, "scan complete")

    def _dialogue(self, screen: ScreenState) -> Dict[str, Any]:
        """Auto-complete a MomoTalk dialogue.

        reference conversation states:
        - 'reply': blue reply button visible → click it
        - 'affection': pink "前往羈絆劇情" → click it
        - 'enter': blue "開始羈絆劇情" → click it
        - 'plot_menu': story cutscene → skip via MENU→SKIP→confirm
        - 'end': no actionable element → conversation done
        """
        self._dialogue_ticks += 1

        if self._dialogue_ticks > 40:
            self.log("dialogue timeout, returning to scan")
            self._dialogue_ticks = 0
            self._story_mode = False
            self.sub_state = "scan"
            self._scan_ticks = 0
            self._click_slot_idx = 0
            return action_back("dialogue timeout")

        # Reply button — the main interaction in MomoTalk
        # reference checks for blue pixels at specific Y positions; we use OCR.
        reply = screen.find_any_text(
            ["回覆", "回复", "Reply"],
            region=(0.50, 0.20, 1.0, 0.95), min_conf=0.45
        )
        if reply:
            self._dialogue_ticks = 0  # reset timeout on each interaction
            # reference: if reply y >= 625/720 ≈ 0.868, swipe up first
            if reply.cy >= 0.85:
                self.log("reply button too low, swiping up")
                return action_swipe(0.72, 0.50, 0.72, 0.35, 200, "swipe up for reply")
            return action_click_box(reply, "click reply")

        # Relationship story button (pink "前往羈絆劇情")
        story_enter = screen.find_any_text(
            ["羈絆", "羁绊", "前往", "To Relationship", "羈絆劇情", "羁绊剧情"],
            region=(0.50, 0.55, 1.0, 0.95), min_conf=0.45
        )
        if story_enter:
            self.log("relationship story available")
            self._dialogue_ticks = 0
            return action_click_box(story_enter, "enter relationship story")

        # "開始羈絆劇情" / Begin Relationship Story button (blue)
        begin_story = screen.find_any_text(
            ["開始", "开始", "Begin"],
            region=(0.60, 0.70, 1.0, 0.95), min_conf=0.5
        )
        if begin_story:
            self.log("begin relationship story")
            self._story_mode = True
            self.sub_state = "story"
            return action_click_box(begin_story, "begin relationship story")

        # Story/cutscene mode indicators
        skip = screen.find_any_text(
            ["SKIP", "Skip", "跳過", "跳过"],
            min_conf=0.65
        )
        if skip:
            self._story_mode = True
            self.sub_state = "story"
            return action_click_box(skip, "skip story")

        menu_btn = screen.find_any_text(
            ["MENU"],
            region=(0.88, 0.0, 1.0, 0.12), min_conf=0.6
        )
        if menu_btn:
            self._story_mode = True
            self.sub_state = "story"
            return action_click_box(menu_btn, "open story menu for skip")

        # Confirm/next button
        confirm = screen.find_any_text(
            ["確認", "确认", "確定", "确定", "下一步", "Next"],
            region=(0.30, 0.60, 0.80, 0.95), min_conf=0.6
        )
        if confirm:
            self._dialogue_ticks = 0
            return action_click_box(confirm, "confirm/next in dialogue")

        # If back to MomoTalk list without conversation view, dialogue complete
        if self._is_momotalk(screen) and not self._is_conversation_view(screen):
            self._conversations_completed += 1
            self._dialogue_ticks = 0
            self._story_mode = False
            self.log(f"conversation #{self._conversations_completed} completed")
            self.sub_state = "scan"
            self._scan_ticks = 0
            self._click_slot_idx = 0
            self._scan_rounds = 0
            return action_wait(300, "dialogue complete, scanning for more")

        # If MomoTalk with conversation open but no buttons: scroll/tap to reveal
        if self._is_momotalk(screen):
            # Try swiping the chat area up to reveal off-screen reply buttons
            if self._dialogue_ticks % 5 == 0:
                return action_swipe(0.72, 0.50, 0.72, 0.35, 200, "swipe chat up")
            return action_wait(400, "waiting for reply button")

        # Tap center to advance dialogue text (story narration)
        return action_click(0.50, 0.50, "tap to advance dialogue")

    def _story(self, screen: ScreenState) -> Dict[str, Any]:
        """Handle relationship story cutscene: MENU → skip → confirm."""
        self._dialogue_ticks += 1

        if self._dialogue_ticks > 50:
            self.log("story timeout")
            self._story_mode = False
            self.sub_state = "scan"
            self._scan_ticks = 0
            self._click_slot_idx = 0
            return action_back("story timeout")

        # Reward popup after story completion
        reward = screen.find_any_text(
            ["獲得道具", "获得道具", "獲得獎勵", "获得奖励"],
            region=screen.CENTER, min_conf=0.6
        )
        if reward:
            self._conversations_completed += 1
            self._story_mode = False
            self.log(f"story #{self._conversations_completed} completed (reward)")
            self.sub_state = "enter"
            self._enter_ticks = 0
            return action_click(0.5, 0.9, "dismiss story reward")

        # Skip confirmation dialog ("是否略過此劇情？")
        skip_confirm = screen.find_any_text(
            ["是否略過", "是否略过", "略過此", "略过此"],
            region=screen.CENTER, min_conf=0.55
        )
        if skip_confirm:
            confirm = screen.find_any_text(
                ["確認", "确认", "確定", "确定", "確", "确"],
                region=(0.50, 0.55, 0.80, 0.85), min_conf=0.55
            )
            if confirm:
                return action_click_box(confirm, "confirm story skip")
            return action_click(0.61, 0.73, "confirm story skip (fallback)")

        # Skip button
        skip = screen.find_any_text(
            ["SKIP", "Skip", "跳過", "跳过"],
            min_conf=0.65
        )
        if skip:
            return action_click_box(skip, "skip story")

        # MENU button → leads to skip option
        menu_btn = screen.find_any_text(
            ["MENU"],
            region=(0.88, 0.0, 1.0, 0.12), min_conf=0.6
        )
        if menu_btn:
            return action_click_box(menu_btn, "open story menu")

        # Story menu skip button (after clicking MENU)
        skip_plot = screen.find_any_text(
            ["跳過劇情", "跳过剧情", "Skip Story"],
            region=(0.80, 0.10, 1.0, 0.25), min_conf=0.5
        )
        if skip_plot:
            return action_click_box(skip_plot, "skip plot from menu")

        # If back to MomoTalk, story ended
        if self._is_momotalk(screen):
            self._conversations_completed += 1
            self._story_mode = False
            self.log(f"story #{self._conversations_completed} completed (back to list)")
            self.sub_state = "scan"
            self._scan_ticks = 0
            self._click_slot_idx = 0
            self._scan_rounds = 0
            return action_wait(300, "story complete, scanning for more")

        # Tap to advance story narration
        return action_click(0.50, 0.50, "tap to advance story")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if screen.is_lobby():
            self.log(f"done ({self._conversations_completed} conversations)")
            return action_done("momotalk complete")
        return action_back("momotalk exit: back to lobby")
