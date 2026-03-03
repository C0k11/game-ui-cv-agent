"""CafeSkill: Handle cafe daily routine.

Flow:
1. ENTER: From lobby, click 咖啡廳 in nav bar
2. EARNINGS: Click 收益 area to claim accumulated credits/AP
3. HEADPAT: YOLO-detect 角色可摸头黄色感叹号 (yellow !) and click each one
4. SWITCH: Click 移動至2號店 to go to cafe 2F
5. HEADPAT2: Same headpat logic on 2F
6. EXIT: Press back until lobby

Key YOLO classes:
- 角色可摸头黄色感叹号 (cls 10): yellow ! above student = headpat target
- 提升好感度的后的爱心 (cls 15): heart after headpat = success
- 叉叉1 (cls 1): close button on earnings/popups
"""
from __future__ import annotations

from typing import Any, Dict

from brain.skills.base import (
    BaseSkill, ScreenState,
    action_click, action_click_box, action_click_yolo,
    action_wait, action_back, action_done,
)

# Min confidence for headpat markers (real marks score 0.50-0.90, false positives 0.15-0.40)
_HEADPAT_CONF = 0.40
# Max consecutive empty scans before giving up on headpats
_MAX_EMPTY_SCANS = 3
# Max headpats per floor (cafe typically has 3-5 students per floor)
_MAX_HEADPATS_PER_FLOOR = 5


class CafeSkill(BaseSkill):
    def __init__(self):
        super().__init__("Cafe")
        self.max_ticks = 80
        self._headpat_count: int = 0
        self._empty_scans: int = 0
        self._earnings_claimed: bool = False
        self._invite_attempted: bool = False
        self._invite_ticks: int = 0

    def reset(self) -> None:
        super().reset()
        self._headpat_count = 0
        self._empty_scans = 0
        self._earnings_claimed = False
        self._invite_attempted = False
        self._invite_ticks = 0

    def _is_cafe(self, screen: ScreenState) -> bool:
        """Detect cafe interior: header '咖啡廳' or '移動至' button visible."""
        if screen.has_text("咖啡", region=(0.0, 0.0, 0.3, 0.08), min_conf=0.5):
            return True
        # Fallback: '移動至' switch button is unique to cafe
        if screen.find_any_text(["移動至", "移动至"], min_conf=0.5):
            return True
        # Fallback: cafe bottom bar has unique text
        if screen.find_any_text(["編輯模式", "编辑模式", "禮物", "家具資訊"], min_conf=0.5):
            return True
        return False

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1

        if self.ticks >= self.max_ticks:
            self.log("timeout, exiting")
            return action_done("cafe timeout")

        # ── Handle popups that can appear at any point ──

        # Earnings popup: only triggers on popup-specific text
        # (NOT '咖啡廳收益' which is a permanent label on cafe main screen)
        if screen.find_any_text(["每小時收益", "收益現况", "收益現況"], min_conf=0.6):
            claim_btn = screen.find_any_text(["領取", "领取"], min_conf=0.7)
            if claim_btn:
                self.log("earnings popup detected, clicking claim")
                self._earnings_claimed = True
                return action_click_box(claim_btn, "claim earnings from popup")
            # OCR can miss the claim text on some frames; use stable popup button coordinate.
            self.log("earnings popup detected, claim text missing -> click claim fallback")
            self._earnings_claimed = True
            return action_click(0.5, 0.734, "claim earnings fallback")
            x_btn = screen.find_yolo_one("叉叉1", min_conf=0.3)
            if x_btn:
                self.log("closing earnings popup via X")
                self._earnings_claimed = True
                return action_click_yolo(x_btn, "close earnings popup")
            close = screen.find_any_text(["確認", "确认", "關閉"], min_conf=0.7)
            if close:
                self._earnings_claimed = True
                return action_click_box(close, "confirm earnings popup")
            self._earnings_claimed = True
            return action_click(0.5, 0.9, "dismiss earnings popup")

        # Tutorial/説明 popup (cafe 2F first visit)
        if screen.find_text_one("說明", region=(0.3, 0.1, 0.7, 0.3), min_conf=0.7):
            confirm = screen.find_text_one("確認", min_conf=0.7)
            if confirm:
                self.log("dismissing tutorial popup")
                return action_click_box(confirm, "dismiss tutorial")
            x_btn = screen.find_yolo_one("叉叉", min_conf=0.3)
            if x_btn:
                return action_click_yolo(x_btn, "close tutorial X")

        # Rank-up popup (好感度升級)
        if screen.find_any_text(["好感度", "Rank"], min_conf=0.6):
            self.log("rank-up popup, tapping to dismiss")
            return action_click(0.5, 0.5, "dismiss rankup popup")

        # Generic popups (confirm/cancel dialogs)
        popup = self._handle_common_popups(screen)
        if popup:
            return popup

        # Loading
        if screen.is_loading():
            return action_wait(800, "cafe loading")

        # ── State machine ──

        if self.sub_state == "":
            self.sub_state = "enter"

        if self.sub_state == "enter":
            return self._enter(screen)
        if self.sub_state == "earnings":
            return self._earnings(screen)
        if self.sub_state == "invite":
            return self._invite(screen)
        if self.sub_state == "headpat":
            return self._headpat(screen)
        if self.sub_state == "switch":
            return self._switch_floor(screen)
        if self.sub_state == "headpat2":
            return self._headpat(screen)
        if self.sub_state == "exit":
            return self._exit(screen)

        return action_wait(300, "cafe unknown state")

    # ── Sub-state handlers ──

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        """Navigate from lobby to cafe."""
        current = self.detect_current_screen(screen)
        
        if current == "Cafe":
            self.log("inside cafe")
            self.sub_state = "earnings"
            return action_wait(500, "entered cafe")
            
        if current == "Lobby":
            nav = self._nav_to(screen, ["咖啡廳", "咖啡厅", "咖啡"])
            if nav:
                return nav
            return action_wait(300, "waiting for cafe button")
            
        if current and current != "Cafe":
            self.log(f"wrong screen '{current}', backing out")
            return action_back(f"back from {current}")

        return action_wait(500, "entering cafe")

    def _earnings(self, screen: ScreenState) -> Dict[str, Any]:
        """Claim cafe earnings.

        Flow from raw data (frame_000083 → frame_000136):
        1. On cafe main screen, click '咖啡廳收益' label at bottom-right (0.913, 0.893)
        2. Earnings popup opens showing '每小時收益', '收益現況'
        3. Click '領取' button at center-bottom (0.5, 0.734) to claim
        4. Popup closes automatically or via X
        """
        if self._earnings_claimed:
            self.sub_state = "invite"
            self._invite_ticks = 0
            self._empty_scans = 0
            return action_wait(300, "earnings done, moving to invite")

        if not self._is_cafe(screen):
            return action_wait(500, "waiting for cafe UI")

        # If earnings popup is already open (領取 button visible)
        claim_btn = screen.find_any_text(["領取", "领取"], min_conf=0.8)
        if claim_btn:
            self.log("earnings popup open, clicking '領取' to claim")
            self._earnings_claimed = True
            return action_click_box(claim_btn, "claim earnings")

        # If earnings popup is open but no claim button (already claimed?)
        if screen.find_any_text(["每小時收益", "收益現況", "收益現況"], min_conf=0.6):
            # OCR might miss claim text; click known claim position before closing.
            self.log("earnings popup open but no claim OCR, clicking fallback claim")
            self._earnings_claimed = True
            return action_click(0.5, 0.734, "claim earnings fallback")

            # Close the popup
            x_btn = screen.find_yolo_one("叉叉1", min_conf=0.3)
            if x_btn:
                self.log("earnings popup open but no claim btn, closing")
                self._earnings_claimed = True
                return action_click_yolo(x_btn, "close earnings popup")
            self._earnings_claimed = True
            return action_click(0.5, 0.9, "dismiss earnings popup")

        # Cafe main screen: click '咖啡廳收益' label to open earnings popup
        # OCR position from frame_000083: cx=0.913, cy=0.893
        earn_label = screen.find_text_one("咖啡廳收益", min_conf=0.5)
        if earn_label:
            self.log("clicking '咖啡廳收益' to open earnings popup")
            return action_click_box(earn_label, "open earnings popup")

        # FULL! means earnings are maxed — click the earnings area
        full = screen.find_text_one("FULL", min_conf=0.6)
        if full:
            self.log("FULL detected, clicking earnings area")
            return action_click(0.913, 0.893, "open earnings via FULL")

        # No earnings indicators — skip
        self._earnings_claimed = True
        self.sub_state = "invite"
        self._invite_ticks = 0
        self._empty_scans = 0
        return action_wait(300, "no earnings visible, moving to invite")

    def _invite(self, screen: ScreenState) -> Dict[str, Any]:
        """Try inviting a student once before headpat loop."""
        if self._invite_attempted:
            self.sub_state = "headpat"
            return action_wait(300, "invite done/skip, starting headpat")

        if not self._is_cafe(screen):
            return action_wait(500, "waiting for cafe UI (invite)")

        self._invite_ticks += 1

        # If invite list is open, click the first 邀請 button.
        invite_btn = screen.find_any_text(["邀請", "邀请"], region=(0.35, 0.10, 0.90, 0.90), min_conf=0.65)
        if invite_btn:
            self.log("invite list detected, clicking 邀請")
            self._invite_attempted = True
            return action_click_box(invite_btn, "invite student")

        # On cafe main screen, click invite ticket area in bottom-right panel.
        ticket = screen.find_any_text(
            ["邀請券", "邀请券", "客外邀請劵", "客外邀请券", "可使用"],
            region=(0.55, 0.78, 0.78, 0.98),
            min_conf=0.65,
        )
        if ticket:
            self.log("clicking invite ticket")
            return action_click_box(ticket, "open invite ticket")

        if self._invite_ticks >= 6:
            self.log("invite UI not found, skipping invite")
            self._invite_attempted = True
            self.sub_state = "headpat"
            return action_wait(300, "invite skipped")

        return action_wait(400, "waiting for invite UI")

    def _headpat(self, screen: ScreenState) -> Dict[str, Any]:
        """Tap students with yellow exclamation marks (角色可摸头黄色感叹号).

        Uses YOLO detection at low confidence (marks often score 0.15-0.47).
        Clicks the highest-confidence mark each tick.
        After _MAX_EMPTY_SCANS consecutive ticks with no marks, moves on.
        """
        if not self._is_cafe(screen):
            if screen.is_lobby():
                self.log("back in lobby during headpat, done")
                return action_done("back in lobby")
            return action_wait(500, "waiting for cafe")

        # Check if we've hit the per-floor headpat limit
        if self._headpat_count >= _MAX_HEADPATS_PER_FLOOR:
            if self.sub_state == "headpat":
                self.log(f"reached max {_MAX_HEADPATS_PER_FLOOR} headpats on 1F, switching")
                self.sub_state = "switch"
                self._headpat_count = 0
                self._empty_scans = 0
                return action_wait(300, "headpat max reached, switching")
            else:
                self.log(f"reached max {_MAX_HEADPATS_PER_FLOOR} headpats on 2F, exiting")
                self.sub_state = "exit"
                return action_wait(300, "headpat2 max reached, exiting")

        # Find headpat markers via YOLO
        mark = screen.find_yolo_one("角色可摸头黄色感叹号", min_conf=_HEADPAT_CONF)
        if not mark:
            mark = screen.find_yolo_one("感叹号", min_conf=_HEADPAT_CONF)
        if mark:
            self._empty_scans = 0
            self._headpat_count += 1
            # Click slightly RIGHT of the marker to hit the student's head
            click_x = min(mark.cx + 0.03, 0.98)
            click_y = mark.cy
            self.log(f"headpat #{self._headpat_count}: conf={mark.confidence:.2f} marker=({mark.cx:.2f},{mark.cy:.2f}) click=({click_x:.2f},{click_y:.2f})")
            return action_click(click_x, click_y, f"headpat student #{self._headpat_count}")

        # OCR fallback when YOLO has no detections at all in this frame.
        if len(screen.yolo_boxes) == 0:
            lv_hits = screen.find_text(r"Lv\\.?\\d+", region=(0.22, 0.30, 0.78, 0.78), min_conf=0.75)
            if lv_hits:
                self._empty_scans = 0
                lv_hits = sorted(lv_hits, key=lambda b: (b.cy, b.cx))
                idx = min(self._headpat_count, len(lv_hits) - 1)
                target = lv_hits[idx]
                self._headpat_count += 1
                click_x = target.cx
                click_y = max(target.cy - 0.08, 0.20)
                self.log(f"headpat fallback #{self._headpat_count}: click above LV at ({click_x:.2f},{click_y:.2f})")
                return action_click(click_x, click_y, f"headpat fallback #{self._headpat_count}")

        # No marks found this tick
        self._empty_scans += 1
        if self._empty_scans >= _MAX_EMPTY_SCANS:
            if self.sub_state == "headpat":
                self.log(f"no more headpat marks after {self._headpat_count} pats, switching floors")
                self.sub_state = "switch"
                self._empty_scans = 0
                return action_wait(300, "headpat done, switching")
            else:  # headpat2
                self.log(f"no more headpat marks on 2F after {self._headpat_count} pats, exiting")
                self.sub_state = "exit"
                return action_wait(300, "headpat2 done, exiting")

        return action_wait(600, f"scanning for headpat marks (empty={self._empty_scans})")

    def _switch_floor(self, screen: ScreenState) -> Dict[str, Any]:
        """Switch from cafe 1F to 2F."""
        switch = screen.find_any_text(
            ["移動至2號店", "移动至2号店", "2號店", "2号店"],
            min_conf=0.5
        )
        if switch:
            self.log("switching to cafe 2F")
            self.sub_state = "headpat2"
            self._empty_scans = 0
            return action_click_box(switch, "switch to cafe 2F")

        # TAP TO START during transition
        tap = screen.find_text_one("TAP.*START", min_conf=0.8)
        if tap:
            return action_click(0.5, 0.85, "tap to start during cafe switch")

        if self.ticks % 10 == 0:
            self.log("switch timeout, skipping 2F")
            self.sub_state = "exit"
            return action_wait(200, "switch timeout")

        return action_wait(500, "waiting for switch button")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        """Return to lobby from cafe."""
        if screen.is_lobby():
            self.log("back in lobby, cafe done")
            return action_done("cafe complete")

        # Click back button or home button if YOLO detects them
        back = screen.find_yolo_one("返回键", min_conf=0.3)
        if back:
            return action_click_yolo(back, "cafe exit: click back button")

        home = screen.find_yolo_one("主界面按钮", min_conf=0.3)
        if home:
            return action_click_yolo(home, "cafe exit: click home button")

        return action_back("cafe exit: press ESC")
