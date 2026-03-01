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

    def reset(self) -> None:
        super().reset()
        self._popup_close_attempts = 0

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1

        # Loading screen → wait
        if screen.is_loading():
            return action_wait(1000, "loading screen")

        # Title screen: "TAP TO START"
        tap = screen.find_text_one("TAP.*START", min_conf=0.8)
        if tap:
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

        # 1. OCR: look for "X" text in the top-right area (the close button)
        x_ocr = screen.find_text_one(
            r"^[Xx×]$", min_conf=0.4,
            region=(0.70, 0.05, 1.0, 0.25)
        )
        if x_ocr:
            self._popup_close_attempts += 1
            self.log(f"close button detected via OCR 'X' at ({x_ocr.cx:.3f},{x_ocr.cy:.3f})")
            return action_click_box(x_ocr, "close popup via OCR X")

        # 2. YOLO: detect close buttons (叉叉, 叉叉1, 叉叉2, momotalk的叉叉)
        x_btn = screen.find_yolo_one("叉叉", min_conf=0.15)
        if x_btn:
            has_popup_text = screen.find_any_text(
                ["Main News", "Update", "Events", "Patch Notes",
                 "Maintenance", "Pick-Up", "Official", "Discord",
                 "Webpage", "My Office", "到簿", "签到", "簽到",
                 "通知"],
                min_conf=0.7
            )
            if has_popup_text:
                self._popup_close_attempts += 1
                self.log(f"popup detected (YOLO {x_btn.cls_name} + '{has_popup_text.text}'), clicking X")
                return action_click_yolo(x_btn, f"close popup via YOLO {x_btn.cls_name}")

        # 3. Announcement popup (公告) — detect via OCR text, hardcoded X positions
        news = screen.find_any_text(
            ["Main News", "Patch Notes", "Maintenance", "Pick-Up",
             "Webpage Open", "My Office", "Official Twitter",
             "Official Forum"],
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

        # 4. Check-in popup (到簿/签到)
        checkin = screen.find_any_text(["到簿", "签到", "簽到"], min_conf=0.7)
        if checkin:
            close_area = screen.find_any_text(
                ["今日不再"],
                region=(0.15, 0.65, 0.35, 0.80), min_conf=0.8
            )
            if close_area:
                self.log("check-in popup: click '今日不再' to dismiss")
                return action_click_box(close_area, "dismiss check-in today")
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

        # ── Lobby check ──
        if screen.is_lobby():
            self.log("lobby detected, no popups, done")
            return action_done("in lobby")

        # Timeout fallback
        if self.ticks >= self.max_ticks:
            self.log("timeout, forcing done")
            return action_done("lobby timeout")

        return action_wait(500, "waiting for lobby")
