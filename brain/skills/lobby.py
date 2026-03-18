"""LobbySkill: Handle startup, popups, and ensure we're in the main lobby.

Responsibilities:
- Close announcement popups (公告) — uses YOLO 叉叉1 for X button
- Close check-in popups (签到/到簿)
- Handle "是否跳過" skip dialogs
- Tap past title screen ("TAP TO START")
- Wait through loading screens
- Confirm we're in lobby with bottom nav visible AND no popups
"""
from __future__ import annotations

from typing import Any, Dict

from brain.skills.base import (
    BaseSkill, ScreenState,
    action_click, action_click_box, action_back, action_wait, action_done,
)


class LobbySkill(BaseSkill):
    def __init__(self):
        super().__init__("Lobby")
        self.max_ticks = 40
        self._popup_close_attempts = 0
        self._lobby_confirm_frames = 0  # consecutive frames with lobby nav visible
        self._start_flow_active = False
        self._blank_intro_taps = 0
        self._story_skip_armed = 0

    def reset(self) -> None:
        super().reset()
        self._popup_close_attempts = 0
        self._lobby_confirm_frames = 0
        self._start_flow_active = False
        self._blank_intro_taps = 0
        self._story_skip_armed = 0

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1

        # Loading screen → wait
        if screen.is_loading():
            return action_wait(1000, "loading screen")

        # Popup hints used to avoid misfiring TAP TO START while a popup is open
        popup_hint = screen.find_any_text(
            ["今日不再", "今日不再提示", "Main News", "Patch Notes", "Maintenance",
             "Pick-Up", "Official", "Webpage", "通知"],
            min_conf=0.7
        )
        story_auto = screen.find_any_text(["AUTO"], region=(0.72, 0.0, 0.92, 0.14), min_conf=0.65)
        story_menu = screen.find_any_text(["MENU"], region=(0.82, 0.0, 1.0, 0.14), min_conf=0.65)
        story_skip = screen.find_any_text(["SKIP", "Skip", "跳過", "跳过"], min_conf=0.65)
        story_skip_prompt = screen.find_any_text(
            ["是否略過", "是否略过", "略過此", "略过此", "略此情"],
            region=screen.CENTER, min_conf=0.6
        )
        story_summary = screen.find_any_text(
            ["概要", "序幕"],
            region=screen.CENTER, min_conf=0.6
        )

        # Strong user preference: if "今日不再提示" exists, click it first.
        # This suppresses repeat popups for the day.
        do_not_show = screen.find_any_text(
            ["今日不再", "今日不再提示", "今日不再顯示", "今日不再显示",
             "今日不再示", "今日不再題示", "今日不再题示"],
            region=(0.02, 0.60, 0.40, 0.99), min_conf=0.65
        )
        if do_not_show:
            self.log("popup: click 今日不再提示")
            return action_click_box(do_not_show, "popup: do not show again today")

        # Title screen: "TOUCH TO START" / "TAP TO START"
        # OCR often merges to "TOUCHTOSTART" — match all variants
        tap = screen.find_text_one("(?:TOUCH|TAP).*START", min_conf=0.8)
        if tap and not popup_hint:
            self._start_flow_active = True
            self._blank_intro_taps = 0
            self.log("title screen: tap to start")
            return action_click(0.5, 0.85, "tap to start")

        if self._start_flow_active and not screen.ocr_boxes and not screen.yolo_boxes:
            self._blank_intro_taps += 1
            if self._blank_intro_taps <= 6:
                self.log(f"startup intro: tap to advance ({self._blank_intro_taps}/6)")
                return action_click(0.5, 0.85, "advance startup intro")
            return action_wait(500, "waiting for startup intro")
        if screen.ocr_boxes or screen.yolo_boxes:
            self._blank_intro_taps = 0

        browser_hint = screen.find_any_text(
            ["discord.com/", "Blue Archive Official", "Official(GL)", "Official (GL)"],
            min_conf=0.7
        )
        if browser_hint:
            browser_close = screen.find_text_one(
                r"^[Xx×]$",
                region=(0.0, 0.0, 0.08, 0.12),
                min_conf=0.55,
            )
            if browser_close:
                self.log("browser view detected → close")
                return action_click_box(browser_close, "close in-game browser")
            self.log("browser view detected → back")
            return action_back("close in-game browser")

        if story_skip_prompt or (story_summary and screen.find_any_text(["取消"], region=screen.CENTER, min_conf=0.6)):
            self._story_skip_armed = 0
            confirm = screen.find_any_text(
                ["確認", "确认", "確定", "确定", "確", "确"],
                region=(0.5, 0.6, 0.75, 0.82), min_conf=0.6
            )
            if confirm:
                self.log("story skip popup → confirm")
                return action_click_box(confirm, "confirm story skip")
            return action_click(0.61, 0.73, "confirm story skip (fallback)")

        if self._story_skip_armed > 0 and story_auto and story_menu:
            self._story_skip_armed -= 1
            self.log("story skip prompt assumed → confirm fallback")
            return action_click(0.61, 0.73, "confirm story skip (armed fallback)")

        if story_skip:
            self._story_skip_armed = 0
            self.log("story screen → click skip")
            return action_click_box(story_skip, "skip story during lobby recovery")

        if story_auto and story_menu:
            self._story_skip_armed = 2
            self.log("story/cutscene detected during lobby recovery → open skip prompt")
            return action_back("open story skip prompt")

        self._story_skip_armed = 0

        mission_info = screen.find_any_text(
            ["任務資訊", "任务资讯"],
            region=(0.25, 0.08, 0.75, 0.22), min_conf=0.5
        )
        mission_max = screen.find_any_text(["MAX"], region=(0.75, 0.38, 0.92, 0.52), min_conf=0.65)
        mission_min = screen.find_any_text(["MIN"], region=(0.52, 0.38, 0.70, 0.52), min_conf=0.65)
        if mission_info or (mission_max and mission_min):
            popup_x = screen.find_text_one(
                r"^[Xx×]$",
                region=(0.82, 0.08, 0.95, 0.22),
                min_conf=0.5,
            )
            if popup_x:
                self.log("mission info popup detected during lobby recovery → close X")
                return action_click_box(popup_x, "close mission info popup")
            self.log("mission info popup detected during lobby recovery → close fallback")
            return action_click(0.883, 0.159, "close mission info popup (fallback)")

        # ── Popup give-up check ──
        # If we've tried many times to close a popup and failed,
        # and the lobby nav bar is visible, just proceed
        if self._popup_close_attempts >= 6 and screen.is_lobby():
            self.log(f"popup won't close after {self._popup_close_attempts} attempts, but lobby nav visible — proceeding")
            return action_done("in lobby (popup overlay skipped)")

        # ── Popup detection (BEFORE lobby check) ──
        # Popups overlay on top of lobby, so bottom nav may be visible
        # even when a popup is blocking interaction.
        # OCR-based close button detection for popups
        has_popup_text = screen.find_any_text(
            ["Main News", "Update", "Events", "Patch Notes",
             "Maintenance", "Pick-Up", "Official", "Discord",
             "Webpage", "My Office", "到簿", "签到", "簽到",
             "通知", "今日不再", "公告"],
            min_conf=0.7
        )
        if has_popup_text:
            x_ocr_btn = screen.find_text_one(
                r"^[Xx×]$",
                region=(0.72, 0.0, 0.98, 0.30),
                min_conf=0.55,
            )
            if x_ocr_btn:
                self._popup_close_attempts += 1
                self.log(f"popup detected (OCR X + '{has_popup_text.text}'), clicking X")
                return action_click_box(x_ocr_btn, "close popup via OCR X")

        # 3. Announcement popup (公告) — detect via OCR text, hardcoded X positions
        # NOTE: "公告" is ALSO a permanent lobby sidebar button (x~0.05, y~0.25).
        # Only treat it as a popup indicator when it appears in the center area,
        # not on the left sidebar.  English texts are popup-only so no filter needed.
        news = screen.find_any_text(
            ["Main News", "Patch Notes", "Maintenance", "Pick-Up",
             "Webpage Open", "My Office", "Official Twitter",
             "Official Forum", "更新消息", "更新資訊", "更新资讯"],
            min_conf=0.7
        )
        if not news:
            news = screen.find_any_text(
                ["公告"],
                region=(0.15, 0.05, 0.85, 0.30),
                min_conf=0.7
            )
        if news:
            self._popup_close_attempts += 1
            news_x = screen.find_text_one(
                r"^[Xx×]$",
                region=(0.94, 0.0, 1.0, 0.10),
                min_conf=0.55,
            )
            if news_x:
                self.log("announcement popup: click OCR X")
                return action_click_box(news_x, "close announcement popup via OCR")
            positions = [
                (0.969, 0.068),
                (0.966, 0.072),
                (0.972, 0.064),
            ]
            idx = (self._popup_close_attempts - 1) % len(positions)
            px, py = positions[idx]
            self.log(f"announcement popup: click X at ({px},{py}) attempt {self._popup_close_attempts}")
            return action_click(px, py, f"close announcement X (attempt {self._popup_close_attempts})")

        # 3. Skip dialog: "是否跳過？" → click 確認
        skip = screen.find_text_one("跳過", region=screen.CENTER, min_conf=0.8)
        if skip:
            confirm = screen.find_any_text(
                ["確", "确", "Space"],
                region=(0.5, 0.6, 0.7, 0.8), min_conf=0.7
            )
            if confirm:
                self.log("skip dialog → confirm")
                return action_click_box(confirm, "confirm skip")
            return action_click(0.6, 0.7, "confirm skip (fallback)")

        # 4. Check-in popup (到簿/签到/每日签到)
        # NOTE: Do NOT include standalone "ARONA" — it appears on the title
        # screen logo and causes false positives.  Only match ARONA when
        # combined with 签到 context (e.g. "ARONA每日签到").
        checkin = screen.find_any_text(
            ["到簿", "签到", "簽到", "彩奈", "每日签到", "每日簽到",
             "ARONA每日", "ARONA签到", "ARONA簽到"],
            min_conf=0.7
        )
        if checkin:
            close_area = screen.find_any_text(
                ["今日不再", "今日不再提示", "今日不再顯示", "今日不再显示"],
                region=(0.15, 0.65, 0.35, 0.80), min_conf=0.8
            )
            if close_area:
                self.log("check-in popup: click '今日不再' to dismiss")
                return action_click_box(close_area, "dismiss check-in today")
            continue_hint = screen.find_any_text(
                ["点击屏幕", "點擊螢幕", "轻触屏幕", "輕觸螢幕", "继续", "繼續"],
                min_conf=0.7,
            )
            if continue_hint:
                self.log("check-in popup: tap to continue")
                return action_click(0.5, 0.8, "check-in: tap to continue")
            self.log("check-in popup: dismiss")
            return action_click(0.95, 0.05, "close check-in popup")

        # 5. Generic "通知" (notification) popup
        notice = screen.find_any_text(
            ["通知", "無法再恢復", "无法再恢复", "挑戰次數", "挑战次数"],
            region=screen.CENTER, min_conf=0.55
        )
        if notice:
            confirm = screen.find_any_text(
                ["確認", "确认", "確", "确"],
                region=screen.CENTER, min_conf=0.55
            )
            if confirm:
                self.log("notification: confirm")
                return action_click_box(confirm, "confirm notification")
            self.log("notification: confirm fallback")
            return action_click(0.50, 0.73, "confirm notification (fallback)")

        # 6. Common popup handler
        popup_action = self._handle_common_popups(screen)
        if popup_action:
            return popup_action

        # ── Lobby check (require 3 consecutive clean frames) ──
        current_screen = self.detect_current_screen(screen)
        if current_screen == "Lobby":
            self._start_flow_active = False
            self._lobby_confirm_frames += 1
            if self._lobby_confirm_frames >= 3:
                self.log(f"lobby confirmed ({self._lobby_confirm_frames} clean frames), done")
                return action_done("in lobby")
            return action_wait(300, f"lobby detected, confirming ({self._lobby_confirm_frames}/3)")
        else:
            self._lobby_confirm_frames = 0

        # ── Already inside a sub-screen? ──
        # If we detected a specific screen other than Lobby, we are deep in a menu.
        # We should navigate back to lobby to ensure a clean state.
        # EXCEPTION: If already on the Event page, skip lobby and let
        # EventActivity handle it directly — backing out wastes time and
        # can trigger unwanted popups (e.g. 新上任指南任務).
        if current_screen and current_screen != "Lobby":
            if current_screen == "Event":
                self.log("already on Event page, skipping lobby recovery")
                return action_done("in Event (skip lobby)")
            self.log(f"detected sub-screen '{current_screen}', returning to lobby")
            return action_back(f"back from {current_screen} to lobby")


        # Timeout fallback
        if self.ticks >= self.max_ticks:
            self.log("timeout, forcing done")
            return action_done("lobby timeout")

        return action_wait(500, "waiting for lobby")
