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

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from brain.skills.base import (
    BaseSkill, ScreenState,
    action_click, action_click_box, action_click_yolo,
    action_wait, action_back, action_done, action_swipe,
)


def _load_target_favorites() -> List[str]:
    """Load target character names from app_config.json."""
    try:
        cfg_path = Path(__file__).resolve().parents[2] / "data" / "app_config.json"
        if cfg_path.exists():
            data = json.loads(cfg_path.read_text("utf-8"))
            return data.get("target_favorites", [])
    except Exception:
        pass
    return []


_AVATAR_MATCH_THRESHOLD = 0.50

# Min confidence for headpat markers.
# 1F marks score ~0.40+, but 2F marks only score 0.18-0.26. Use 0.15 to catch both.
_HEADPAT_CONF = 0.15
# Max consecutive empty scans before giving up on headpats
_MAX_EMPTY_SCANS = 5
# Max headpats per floor (cafe typically has 3-5 students per floor)
_MAX_HEADPATS_PER_FLOOR = 7


class CafeSkill(BaseSkill):
    def __init__(self):
        super().__init__("Cafe")
        self.max_ticks = 100
        self._headpat_count: int = 0
        self._empty_scans: int = 0
        self._earnings_claimed: bool = False
        self._earnings_attempts: int = 0
        self._invite_attempted: bool = False
        self._invite_ticks: int = 0
        self._invite_stage: int = 0  # 0=open ticket, 1=click invite, 2=confirm, 3=done
        self._invite_next_state: str = "headpat"  # where to go after invite
        self._pan_phase: int = 0  # 0=not started, 1=panned right, 2=panned left, 3=done
        self._target_favorites: List[str] = []
        self._avatar_matcher = None
        self._invite_scroll_count: int = 0
        self._headpat_cooldown: int = 0
        self._1f_headpat_started: bool = False  # True once 1F headpat phase begins
        self._1f_done: bool = False  # True once 1F headpat is complete (switch to 2F)

    def reset(self) -> None:
        super().reset()
        self._headpat_count = 0
        self._empty_scans = 0
        self._earnings_claimed = False
        self._earnings_attempts = 0
        self._invite_attempted = False
        self._invite_ticks = 0
        self._invite_stage = 0
        self._invite_next_state = "headpat"
        self._pan_phase = 0
        self._invite_scroll_count = 0
        self._headpat_cooldown = 0
        self._1f_headpat_started = False
        self._1f_done = False
        self._target_favorites = _load_target_favorites()
        if self._target_favorites:
            self.log(f"loaded {len(self._target_favorites)} favorite characters")
        if self._avatar_matcher is None and self._target_favorites:
            try:
                from vision.avatar_matcher import AvatarMatcher
                avatar_dir = Path(__file__).resolve().parents[2] / "data" / "captures" / "角色头像"
                self._avatar_matcher = AvatarMatcher(str(avatar_dir))
                self.log(f"avatar matcher loaded from {avatar_dir}")
            except Exception as e:
                self.log(f"avatar matcher init failed: {e}")
                self._avatar_matcher = None

    def _find_favorite_in_invite(self, screen: ScreenState, invite_btns) -> Optional[Any]:
        """Scan YOLO-detected avatars in the invite list and match against favorites.

        Returns the invite button (OcrBox) nearest to the matched favorite avatar,
        or None if no favorite is found.
        """
        if not self._target_favorites or self._avatar_matcher is None:
            return None

        avatars = screen.find_yolo("角色头像", min_conf=0.3)
        if not avatars:
            return None

        try:
            import cv2
            import numpy as np
            img = cv2.imdecode(
                np.fromfile(screen.screenshot_path, dtype=np.uint8),
                cv2.IMREAD_COLOR
            )
            if img is None:
                return None
            h, w = img.shape[:2]
        except Exception:
            return None

        # Only consider avatars in the invite list area (left half, y 0.20-0.90)
        list_avatars = [a for a in avatars if a.cx < 0.50 and 0.15 < a.cy < 0.90]
        if not list_avatars:
            return None

        for av in list_avatars:
            bx1 = max(0, int(av.x1 * w))
            by1 = max(0, int(av.y1 * h))
            bx2 = min(w, int(av.x2 * w))
            by2 = min(h, int(av.y2 * h))
            roi = img[by1:by2, bx1:bx2]
            if roi.size == 0:
                continue
            matched_name, score = self._avatar_matcher.match_avatar(
                roi, self._target_favorites
            )
            if matched_name and score > _AVATAR_MATCH_THRESHOLD:
                self.log(f"FAVORITE MATCH: '{matched_name}' score={score:.2f} at ({av.cx:.2f},{av.cy:.2f})")
                # Find the invite button closest in Y to this avatar
                best_btn = None
                best_dist = 999.0
                for btn in invite_btns:
                    dist = abs(btn.cy - av.cy)
                    if dist < best_dist:
                        best_dist = dist
                        best_btn = btn
                if best_btn and best_dist < 0.10:
                    return best_btn
                self.log(f"matched avatar but no nearby invite button (dist={best_dist:.2f})")

        return None

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
        # Skip if we already claimed — prevents infinite loop when inventory is full.
        if not self._earnings_claimed and screen.find_any_text(["每小時收益", "收益現况", "收益現況"], min_conf=0.6):
            claim_btn = screen.find_any_text(["領取", "领取"], min_conf=0.7)
            if claim_btn:
                self.log("earnings popup detected, clicking claim")
                self._earnings_claimed = True
                return action_click_box(claim_btn, "claim earnings from popup")
            # OCR can miss the claim text on some frames; use stable popup button coordinate.
            self.log("earnings popup detected, claim text missing -> click claim fallback")
            self._earnings_claimed = True
            return action_click(0.5, 0.734, "claim earnings fallback")

        # Tutorial/説明 popup (cafe 2F first visit)
        # OCR sometimes only detects single char '明' instead of '說明'.
        tutorial = screen.find_text_one("說明", region=(0.3, 0.1, 0.7, 0.3), min_conf=0.7)
        if not tutorial:
            tutorial = screen.find_any_text(
                ["訪問學生目錄", "訪問學生", "訪间學生"],
                region=screen.CENTER, min_conf=0.6
            )
        if tutorial:
            confirm = screen.find_any_text(
                ["確認", "确认", "確", "确"],
                region=screen.CENTER, min_conf=0.7
            )
            if confirm:
                self.log("dismissing tutorial popup")
                return action_click_box(confirm, "dismiss tutorial")
            x_btn = screen.find_yolo_one("叉叉", min_conf=0.3)
            if x_btn:
                return action_click_yolo(x_btn, "close tutorial X")

        # Notification popup (通知) — e.g. invite cooldown, generic alerts
        # Has "通知" title + "確" button, no cancel; always safe to dismiss.
        notif = screen.find_text_one("通知", region=(0.35, 0.15, 0.65, 0.30), min_conf=0.8)
        if notif:
            confirm = screen.find_any_text(
                ["確認", "确认", "確", "确"],
                region=screen.CENTER, min_conf=0.7
            )
            if confirm:
                self.log(f"notification popup, clicking confirm (sub={self.sub_state})")
                if self.sub_state == "invite":
                    self._invite_attempted = True
                return action_click_box(confirm, "dismiss notification")
            x_btn = screen.find_yolo_one("叉叉", min_conf=0.3)
            if x_btn:
                if self.sub_state == "invite":
                    self._invite_attempted = True
                return action_click_yolo(x_btn, "close notification X")

        # Rank-up / bond level up popup (好感度升級 / 羈絆升級)
        # OCR often misreads 羈絆升級 as 鲜升級 due to stylized font.
        if screen.find_any_text(["好感度", "Rank Up"], min_conf=0.6):
            self.log("rank-up popup, tapping to dismiss")
            return action_click(0.5, 0.5, "dismiss rankup popup")

        # Bond level up screen (羈絆升級！) — full-screen animation
        # Detected via stat text at bottom (治愈力/最大體力) which OCR reads reliably.
        # GUARD: exclude student profile screen which also shows 最大體力 but has
        # unique markers like 基本情報, EX技能, Tip!, 神秘解放.
        # Student profile screen: has unique markers like 基本情報, EX技能, 神秘解放.
        # NOTE: Do NOT include "Tip!" — it also appears on loading tip screens.
        student_profile = screen.find_any_text(
            ["基本情報", "EX技能", "神秘解放"],
            min_conf=0.6
        )
        if student_profile:
            self.log("on student profile, pressing back to return to cafe")
            return action_back("back from student profile")
        bond_stat = screen.find_any_text(
            ["治愈力", "治癒力", "最大體力", "最大体力"],
            min_conf=0.6
        )
        if bond_stat:
            self.log("bond level up screen (stat text), tapping to dismiss")
            return action_click(0.5, 0.5, "dismiss bond level up")
        # Pre-level-up blank screen: "在咖啡廳獲得學生的羈絆點數" at top center
        if not self._is_cafe(screen):
            bond_notif = screen.find_any_text(
                ["羈絆升級", "鲜升級", "羈絆點數", "羈絆"],
                min_conf=0.6
            )
            if not bond_notif:
                bond_notif = screen.find_any_text(
                    ["在咖啡"],
                    region=(0.25, 0.0, 0.75, 0.12),
                    min_conf=0.6
                )
            if bond_notif:
                self.log("bond notification screen, tapping to dismiss")
                return action_click(0.5, 0.5, "dismiss bond notification")

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
            # If 1F headpat was already done (kicked to lobby by bond level-up),
            # skip earnings/invite and go straight to switch to 2F.
            if self._1f_done:
                self.log("re-entered cafe after bond level-up, switching to 2F")
                self.sub_state = "switch"
                self._headpat_count = 0
                self._empty_scans = 0
                self._pan_phase = 0
                return action_wait(300, "skip to cafe 2F")
            if self._1f_headpat_started:
                self.log("re-entered cafe, resuming headpat")
                self.sub_state = "headpat"
                self._empty_scans = 0
                self._pan_phase = 0
                return action_wait(300, "resume headpat after re-entry")
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
            self._invite_next_state = "headpat"
            self._invite_ticks = 0
            self._empty_scans = 0
            return action_wait(300, "earnings done, moving to invite")

        if not self._is_cafe(screen):
            return action_wait(500, "waiting for cafe UI")

        # Check for 0% earnings — skip claim entirely
        # OCR reads "0.0%" or "0.0 %" at bottom-right; regex ^0\.0 avoids matching "100.0%"
        zero_pct = screen.find_text_one(
            r"^0\.0", region=(0.85, 0.88, 0.98, 0.98), min_conf=0.8
        )
        if zero_pct:
            self.log(f"earnings 0% ({zero_pct.text}), skipping")
            self._earnings_claimed = True
            # Only close popup if earnings popup is actually open (收益 text visible)
            if screen.find_any_text(["每小時收益", "收益現况", "收益現況"], min_conf=0.6):
                x_btn = screen.find_yolo_one("叉叉", min_conf=0.4)
                if x_btn and 0.15 < x_btn.cy < 0.25:
                    return action_click_yolo(x_btn, "close earnings popup (0%)")
            self.sub_state = "invite"
            self._invite_next_state = "headpat"
            self._invite_ticks = 0
            return action_wait(300, "earnings 0%, skipping to invite")

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

        # Cafe main screen: click '咖啡廳收益' label to open earnings popup
        # Only open if FULL! is visible (earnings at max capacity)
        full = screen.find_text_one("FULL", min_conf=0.6)
        if full:
            self.log("FULL detected, clicking earnings area")
            return action_click(0.913, 0.893, "open earnings via FULL")

        # Also try earnings label if percentage is not 0
        earn_label = screen.find_text_one("咖啡廳收益", min_conf=0.5)
        if earn_label:
            self.log("clicking '咖啡廳收益' to open earnings popup")
            return action_click_box(earn_label, "open earnings popup")

        # No earnings indicators — skip
        self._earnings_claimed = True
        self.sub_state = "invite"
        self._invite_next_state = "headpat"
        self._invite_ticks = 0
        self._empty_scans = 0
        return action_wait(300, "no earnings visible, moving to invite")

    def _invite(self, screen: ScreenState) -> Dict[str, Any]:
        """Try inviting a student before headpat loop.

        Stages: 0=open ticket panel, 1=click 邀請, 2=confirm invite, 3=done
        """
        if self._invite_attempted:
            self.sub_state = self._invite_next_state
            self._headpat_count = 0
            self._empty_scans = 0
            if self._invite_next_state == "headpat":
                self._1f_headpat_started = True
            return action_wait(300, f"invite done/skip, starting {self._invite_next_state}")

        if not self._is_cafe(screen):
            return action_wait(500, "waiting for cafe UI (invite)")

        self._invite_ticks += 1

        # Stage 2: Confirm the invite in the confirmation popup
        if self._invite_stage == 2:
            confirm = screen.find_any_text(
                ["確認", "确认", "確定", "确定", "確", "确"],
                region=screen.CENTER, min_conf=0.7
            )
            if confirm:
                self.log("confirming student invite")
                self._invite_stage = 3
                self._invite_attempted = True
                return action_click_box(confirm, "confirm invite")

            # Invite popup often uses single-char confirm ('確'). If OCR misses it,
            # use popup-body + cancel as a safe signal and click known confirm spot.
            invite_popup = screen.find_text_one(r"邀請.*咖啡廳", region=screen.CENTER, min_conf=0.7)
            cancel_btn = screen.find_any_text(["取消"], region=(0.30, 0.62, 0.50, 0.78), min_conf=0.8)
            if invite_popup and cancel_btn:
                self.log("invite popup detected without confirm OCR, clicking confirm fallback")
                self._invite_stage = 3
                self._invite_attempted = True
                return action_click(0.598, 0.701, "confirm invite fallback")

            # If popup is gone (global interceptor already confirmed it), detect
            # by seeing normal cafe view with no popup after a few ticks.
            if self._invite_ticks >= 3 and self._is_cafe(screen):
                notif = screen.find_text_one("通知", region=(0.35, 0.15, 0.65, 0.30), min_conf=0.7)
                if not notif:
                    self.log("invite confirm popup gone (interceptor handled it)")
                    self._invite_attempted = True
                    return action_wait(300, "invite done (popup dismissed)")

            # If no confirm button yet, wait for popup
            if self._invite_ticks >= 15:
                self.log("invite confirm timeout, skipping")
                self._invite_attempted = True
            return action_wait(400, "waiting for invite confirm popup")

        # Stage 1: Invite list is open, find favorite student or click first
        if self._invite_stage == 1:
            invite_btns = screen.find_text(
                "邀請", region=(0.50, 0.20, 0.70, 0.90), min_conf=0.65
            )
            if not invite_btns:
                invite_btns = screen.find_text(
                    "邀请", region=(0.50, 0.20, 0.70, 0.90), min_conf=0.65
                )
            if invite_btns:
                # Try to find a favorite student via avatar matching
                fav_btn = self._find_favorite_in_invite(screen, invite_btns)
                if fav_btn:
                    self.log(f"inviting favorite at ({fav_btn.cx:.2f},{fav_btn.cy:.2f})")
                    self._invite_stage = 2
                    self._invite_ticks = 0
                    return action_click_box(fav_btn, "invite favorite student")
                # No favorite found — scroll if we haven't much, else click first
                if self._invite_scroll_count < 3:
                    self._invite_scroll_count += 1
                    self.log(f"no favorite found, scrolling invite list ({self._invite_scroll_count}/3)")
                    return action_swipe(0.35, 0.70, 0.35, 0.35, 400, "scroll invite list")
                btn = invite_btns[0]
                self.log(f"no favorite after scrolling, clicking first 邀請 at ({btn.cx:.2f},{btn.cy:.2f})")
                self._invite_stage = 2
                self._invite_ticks = 0
                return action_click_box(btn, "invite student (no fav match)")
            if self._invite_ticks >= 10:
                self.log("invite list not found, skipping")
                self._invite_attempted = True
            return action_wait(400, "waiting for invite list")

        # Close any leftover earnings popup before invite (first tick only)
        if self._invite_ticks == 1:
            if screen.find_any_text(["每小時收益", "收益現况", "收益現況"], min_conf=0.6):
                x_btn = screen.find_yolo_one("叉叉", min_conf=0.4)
                if x_btn and 0.15 < x_btn.cy < 0.25:
                    return action_click_yolo(x_btn, "close earnings popup before invite")

        # Stage 0: Open the invite ticket panel from cafe main screen
        # Skip if cooldown timer (HH:MM:SS) visible near the REGULAR ticket area.
        # NOTE: "可購買" at cy≈0.83 is the EXTRA ticket purchase label — ignore it.
        # The regular ticket cooldown timer appears above the regular ticket (cy≈0.88-0.96).
        cooldown = screen.find_text_one(
            r"\d+:\d+:\d+", region=(0.55, 0.88, 0.78, 0.98), min_conf=0.7
        )
        if cooldown:
            self.log(f"invite cooldown ({cooldown.text}), skipping")
            self._invite_attempted = True
            return action_wait(200, f"invite unavailable: {cooldown.text}")

        # Find the REGULAR invite ticket (not the paid 額外邀請券)
        # OCR reads 額外 as "额外", "客外", or "額外" — exclude all variants.
        _EXTRA_PREFIXES = ("客外", "额外", "額外", "客")
        ticket_hits = screen.find_text(
            "邀請券", region=(0.55, 0.88, 0.78, 0.98), min_conf=0.60
        )
        ticket = None
        for hit in ticket_hits:
            if not any(p in hit.text for p in _EXTRA_PREFIXES):
                ticket = hit
                break
        if not ticket:
            ticket = screen.find_text_one(
                "可使用", region=(0.55, 0.88, 0.78, 0.98), min_conf=0.60
            )
        if ticket:
            self.log(f"clicking invite ticket '{ticket.text}' at ({ticket.cx:.2f},{ticket.cy:.2f})")
            self._invite_stage = 1
            self._invite_ticks = 0
            return action_click_box(ticket, "open invite ticket")

        # Also check if invite list is already open (e.g. from previous attempt)
        invite_btn = screen.find_any_text(
            ["邀請", "邀请"],
            region=(0.50, 0.20, 0.70, 0.90), min_conf=0.65
        )
        if invite_btn:
            self._invite_stage = 1
            self._invite_ticks = 0
            return action_wait(200, "invite list already open")

        if self._invite_ticks >= 6:
            self.log("invite UI not found, skipping invite")
            self._invite_attempted = True
            self.sub_state = self._invite_next_state
            return action_wait(300, "invite skipped")

        return action_wait(400, "waiting for invite UI")

    def _headpat(self, screen: ScreenState) -> Dict[str, Any]:
        """Tap students with yellow exclamation marks (角色可摸头黄色感叹号).

        Camera panning: students can be in corners that are off-screen.
        Phase 0: pan right (drag from right→left) to reveal left corner, then scan.
        Phase 1: pan left (drag from left→right) to reveal right corner, then scan.
        Phase 2: pan back to center, then scan.
        Phase 3: done panning, scan only.

        After _MAX_EMPTY_SCANS consecutive ticks with no marks in current view,
        advance pan phase. When all phases exhausted, move on.
        """
        if not self._is_cafe(screen):
            if screen.is_lobby():
                if self.sub_state == "headpat2":
                    self.log("back in lobby during headpat2, done")
                    return action_done("back in lobby")
                # On 1F headpat: lobby means we got kicked out — try to re-enter cafe
                self.log("lobby during headpat1, re-entering cafe")
                self.sub_state = "enter"
                return action_wait(300, "re-enter cafe from lobby")
            return action_wait(300, "waiting for cafe")

        # Check if we've hit the per-floor headpat limit
        if self._headpat_count >= _MAX_HEADPATS_PER_FLOOR:
            if self.sub_state == "headpat":
                self.log(f"reached max {_MAX_HEADPATS_PER_FLOOR} headpats on 1F, switching")
                self.sub_state = "switch"
                self._1f_done = True
                self._headpat_count = 0
                self._empty_scans = 0
                self._pan_phase = 0
                return action_wait(300, "headpat max reached, switching")
            else:
                self.log(f"reached max {_MAX_HEADPATS_PER_FLOOR} headpats on 2F, exiting")
                self.sub_state = "exit"
                return action_wait(300, "headpat2 max reached, exiting")

        # Camera panning: swipe to reveal corners before scanning
        if self._pan_phase == 0:
            self._pan_phase = 1
            self._empty_scans = 0
            self.log("pan camera: drag right→left to reveal left corner")
            return action_swipe(0.75, 0.45, 0.25, 0.45, 500, "pan camera right")
        if self._pan_phase == 2:
            self._pan_phase = 3
            self._empty_scans = 0
            self.log("pan camera: drag left→right to reveal right corner")
            return action_swipe(0.25, 0.45, 0.75, 0.45, 500, "pan camera left")

        # After a successful headpat, wait for the heart animation to finish
        # before scanning again (animation takes ~1 second).
        if hasattr(self, '_headpat_cooldown') and self._headpat_cooldown > 0:
            self._headpat_cooldown -= 1
            return action_wait(500, f"waiting for headpat animation ({self._headpat_cooldown} left)")

        # Find headpat markers via YOLO
        mark = screen.find_yolo_one("角色可摸头黄色感叹号", min_conf=_HEADPAT_CONF)
        if not mark:
            mark = screen.find_yolo_one("感叹号", min_conf=_HEADPAT_CONF)
        if mark:
            self._empty_scans = 0
            self._headpat_count += 1
            self._headpat_cooldown = 2  # Wait 2 ticks (~1s) for animation
            # The yellow ! mark floats ABOVE the student, offset slightly LEFT.
            # The student's head is to the RIGHT of the mark center.
            # Click right-of-center and at the bottom of the mark to hit the head.
            click_x = mark.cx + 0.03
            click_y = mark.y2
            self.log(f"headpat #{self._headpat_count}: conf={mark.confidence:.2f} marker=({mark.cx:.2f},{mark.cy:.2f}) click=({click_x:.2f},{click_y:.2f})")
            return action_click(click_x, click_y, f"headpat student #{self._headpat_count}")

        # OCR fallback when YOLO has no detections at all in this frame.
        if len(screen.yolo_boxes) == 0:
            lv_hits = screen.find_text(r"Lv\.?\d+", region=(0.22, 0.30, 0.78, 0.78), min_conf=0.75)
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

        # Bond progress bar overlay ("在咖啡獲得學生的羈絆點數 X/Y") reduces
        # YOLO confidence for headpat marks. Don't count empty scans while visible.
        bond_bar = screen.find_any_text(
            ["在咖啡"],
            region=(0.25, 0.0, 0.75, 0.12),
            min_conf=0.6
        )
        if bond_bar:
            return action_wait(800, "bond progress bar visible, waiting")

        # No marks found this tick
        self._empty_scans += 1

        # After a few empty scans, advance to next pan phase
        if self._empty_scans >= _MAX_EMPTY_SCANS:
            if self._pan_phase < 4:
                # Advance pan phase: 1→2 (will pan left next), 3→4 (done panning)
                self._pan_phase += 1
                self._empty_scans = 0
                self.log(f"empty scans exhausted, advancing pan phase to {self._pan_phase}")
                return action_wait(300, f"advance pan phase {self._pan_phase}")

            # All pan phases done, move to next state
            if self.sub_state == "headpat":
                self.log(f"no more headpat marks after {self._headpat_count} pats, switching floors")
                self.sub_state = "switch"
                self._1f_done = True
                self._headpat_count = 0
                self._empty_scans = 0
                self._pan_phase = 0
                return action_wait(300, "headpat done, switching")
            else:  # headpat2
                self.log(f"no more headpat marks on 2F after {self._headpat_count} pats, exiting")
                self.sub_state = "exit"
                return action_wait(300, "headpat2 done, exiting")

        return action_wait(300, f"scanning for headpat marks (empty={self._empty_scans}, pan={self._pan_phase})")

    def _switch_floor(self, screen: ScreenState) -> Dict[str, Any]:
        """Switch from cafe 1F to 2F."""
        switch = screen.find_any_text(
            ["移動至2號店", "移动至2号店", "2號店", "2号店"],
            min_conf=0.5
        )
        if switch:
            self.log("switching to cafe 2F")
            # Reset invite state for cafe 2F invite
            self._invite_attempted = False
            self._invite_ticks = 0
            self._invite_stage = 0
            self._invite_next_state = "headpat2"
            self.sub_state = "invite"
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
