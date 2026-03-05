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
    action_click, action_click_box, action_click_yolo, action_back, action_wait, action_done,
)


class LobbySkill(BaseSkill):
    def __init__(self):
        super().__init__("Lobby")
        self.max_ticks = 40
        self._popup_close_attempts = 0
        self._lobby_confirm_frames = 0  # consecutive frames with lobby nav visible

    def reset(self) -> None:
        super().reset()
        self._popup_close_attempts = 0
        self._lobby_confirm_frames = 0

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

        # Strong user preference: if "今日不再提示" exists, click it first.
        # This suppresses repeat popups for the day.
        do_not_show = screen.find_any_text(
            ["今日不再", "今日不再提示", "今日不再顯示", "今日不再显示", "今日不再示"],
            region=(0.10, 0.62, 0.40, 0.82), min_conf=0.75
        )
        if do_not_show:
            self.log("popup: click 今日不再提示")
            return action_click_box(do_not_show, "popup: do not show again today")

        # Title screen: "TOUCH TO START" / "TAP TO START"
        # OCR often merges to "TOUCHTOSTART" — match all variants
        tap = screen.find_text_one("(?:TOUCH|TAP).*START", min_conf=0.8)
        if tap and not popup_hint:
            self.log("title screen: tap to start")
            return action_click(0.5, 0.85, "tap to start")

        # ── Popup give-up check ──
        # If we've tried many times to close a popup and failed,
        # and the lobby nav bar is visible, just proceed
        if self._popup_close_attempts >= 6 and screen.is_lobby():
            self.log(f"popup won't close after {self._popup_close_attempts} attempts, but lobby nav visible — proceeding")
            return action_done("in lobby (popup overlay skipped)")

        # ── Popup detection (BEFORE lobby check) ──
        # Popups overlay on top of lobby, so bottom nav may be visible
        # even when a popup is blocking interaction.
        # NOTE: All X/close button detection via YOLO only (OCR misdetects icons as "X")

        # 1. YOLO: detect close buttons (叉叉, 叉叉1, 叉叉2, momotalk的叉叉)
        # Filter out fullscreen toggle button at top-right (x>0.93, y<0.20)
        # which YOLO misclassifies as 叉叉1.
        x_btn = screen.find_yolo_one("叉叉", min_conf=0.15,
                                      region=(0.0, 0.0, 0.93, 1.0))
        if x_btn:
            has_popup_text = screen.find_any_text(
                ["Main News", "Update", "Events", "Patch Notes",
                 "Maintenance", "Pick-Up", "Official", "Discord",
                 "Webpage", "My Office", "到簿", "签到", "簽到",
                 "通知", "今日不再", "公告"],
                min_conf=0.7
            )
            if has_popup_text:
                self._popup_close_attempts += 1
                self.log(f"popup detected (YOLO {x_btn.cls_name} + '{has_popup_text.text}'), clicking X")
                return action_click_yolo(x_btn, f"close popup via YOLO {x_btn.cls_name}")

        # 2.5 OCR fallback for close X (only when popup hints exist)
        # Keep this constrained to avoid old false positives.
        if popup_hint:
            x_ocr = screen.find_text_one(
                r"^[Xx×]$",
                region=(0.72, 0.08, 0.90, 0.28),
                min_conf=0.75,
            )
            if x_ocr:
                self._popup_close_attempts += 1
                self.log("popup detected (OCR X fallback), clicking X")
                return action_click_box(x_ocr, "close popup via OCR X fallback")

        # 3. Announcement popup (公告) — detect via OCR text, hardcoded X positions
        # NOTE: "公告" is ALSO a permanent lobby sidebar button (x~0.05, y~0.25).
        # Only treat it as a popup indicator when it appears in the center area,
        # not on the left sidebar.  English texts are popup-only so no filter needed.
        news = screen.find_any_text(
            ["Main News", "Patch Notes", "Maintenance", "Pick-Up",
             "Webpage Open", "My Office", "Official Twitter",
             "Official Forum"],
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
            # X button position from YOLO training label (叉叉2 on frame_000056)
            positions = [
                (0.8735, 0.1575),  # Exact YOLO label center
                (0.87, 0.16),      # Slight variant
                (0.88, 0.15),      # Slight variant
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
        notice = screen.find_text_one("通知", region=screen.CENTER, min_conf=0.8)
        if notice:
            confirm = screen.find_any_text(
                ["確認", "确认", "確", "确"],
                region=screen.CENTER, min_conf=0.7
            )
            if confirm:
                self.log("notification: confirm")
                return action_click_box(confirm, "confirm notification")

        # 6. Common popup handler
        popup_action = self._handle_common_popups(screen)
        if popup_action:
            return popup_action

        # ── Lobby check (require 3 consecutive clean frames) ──
        current_screen = self.detect_current_screen(screen)
        if current_screen == "Lobby":
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
        if current_screen and current_screen != "Lobby":
            self.log(f"detected sub-screen '{current_screen}', returning to lobby")
            # Try YOLO Home button first
            home = screen.find_yolo_one("主界面按钮", min_conf=0.3)
            if home:
                return action_click_yolo(home, "click home button")
            # Fallback to Back
            return action_back(f"back from {current_screen} to lobby")

        # Timeout fallback
        if self.ticks >= self.max_ticks:
            self.log("timeout, forcing done")
            return action_done("lobby timeout")

        return action_wait(500, "waiting for lobby")
