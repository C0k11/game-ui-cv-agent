"""CafeSkill: Handle cafe daily routine.

Flow:
1. ENTER: From lobby, click 咖啡廳 in nav bar
2. EARNINGS: Click 收益 area to claim accumulated credits/AP
3. INVITE: Use invitation ticket (favorite student priority)
4. HEADPAT: Template-match happy_face markers and click each student
5. SWITCH: Click 移動至2號店 to go to cafe 2F
6. INVITE2 + HEADPAT2: Same invite + headpat logic on 2F
7. EXIT: Press back until lobby

Detection priority:
- Primary: happy_face template matching (4 templates, threshold 0.75)
- Fallback: Emoticon_Action template, then YOLO headpat_bubble
- Panning: 1F left→right, 2F right→left (template-based)
"""
from __future__ import annotations
import importlib.util
import json
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import unquote

_CAFE_STATE_FILE = Path(__file__).resolve().parents[2] / "data" / "cafe_state.json"


def _game_day() -> str:
    """Return the BA game-day (ISO date) for 'today'.

    BA resets daily at 04:00 local.  Anything before 04:00 still counts as
    the previous game day, so we shift the clock back 4 hours before
    taking the date component.  This keeps invite-state persistent across
    pipeline retries within the same game day and auto-clears after the
    next 04:00 reset.
    """
    return (datetime.now() - timedelta(hours=4)).date().isoformat()


def _load_cafe_state() -> dict:
    """Load persisted cafe state (invited names etc.).  Empty on error."""
    try:
        if _CAFE_STATE_FILE.exists():
            return json.loads(_CAFE_STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_cafe_state(state: dict) -> None:
    """Persist cafe state to disk (best-effort, swallow errors)."""
    try:
        _CAFE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CAFE_STATE_FILE.write_text(
            json.dumps(state, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass

from brain.skills.base import (
    BaseSkill, ScreenState,
    action_click, action_click_box,
    action_wait, action_back, action_done, action_swipe, action_scroll,
)
from brain.skills import ui_classes as UC


def _load_target_favorites() -> List[str]:
    """Load target character names from app_config.json."""
    try:
        cfg_path = Path(__file__).resolve().parents[2] / "data" / "app_config.json"
        if cfg_path.exists():
            data = json.loads(cfg_path.read_text("utf-8"))
            raw = data.get("target_favorites", [])
            normalized: List[str] = []
            seen = set()
            for item in raw:
                name = str(item or "").strip()
                if not name:
                    continue
                candidates = [name]
                decoded = unquote(name)
                if decoded and decoded != name:
                    candidates.append(decoded)
                for candidate in candidates:
                    key = candidate.lower()
                    if key in seen:
                        continue
                    normalized.append(candidate)
                    seen.add(key)
            return normalized
    except Exception:
        pass
    return []


_AVATAR_MATCH_THRESHOLD = 0.50

# Chinese→English student name map for OCR-based invite matching.
# Expanded at load time with SC↔TC character variants so OCR mixing
# simplified/traditional characters still matches.
_STUDENT_NAME_MAP: Dict[str, str] = {}
_SC_TC_PAIRS = "装裝 团團 战戰 导導 营營 队隊 仆僕 诞誕 骑騎 乐樂 礼禮 温溫 运運 应應 烧燒 历歷 声聲 绘繪 爱愛 丽麗 实實 织織 优優 饰飾 宝寶 护護 风風 语語 梦夢 备備 关關 觉覺 银銀 龙龍 结結 满滿 纪紀 闪閃 创創 灵靈 弹彈"
try:
    _name_map_path = Path(__file__).resolve().parents[2] / "data" / "student_name_map.json"
    if _name_map_path.exists():
        _raw_map = json.loads(_name_map_path.read_text("utf-8"))
        _STUDENT_NAME_MAP.update(_raw_map)
        # Generate SC↔TC variants for each entry
        _sc2tc = {}
        _tc2sc = {}
        for pair in _SC_TC_PAIRS.split():
            if len(pair) == 2:
                _sc2tc[pair[0]] = pair[1]
                _tc2sc[pair[1]] = pair[0]
        for cn_name, en_name in list(_raw_map.items()):
            # SC→TC variant
            tc = cn_name
            for sc, t in _sc2tc.items():
                tc = tc.replace(sc, t)
            if tc != cn_name and tc not in _STUDENT_NAME_MAP:
                _STUDENT_NAME_MAP[tc] = en_name
            # TC→SC variant
            sc = cn_name
            for t, s in _tc2sc.items():
                sc = sc.replace(t, s)
            if sc != cn_name and sc not in _STUDENT_NAME_MAP:
                _STUDENT_NAME_MAP[sc] = en_name
except Exception:
    pass

# Min confidence for headpat markers.
# 1F marks score ~0.40+, but 2F marks only score 0.18-0.26. Use 0.15 to catch both.
_HEADPAT_CONF = 0.15
# Max consecutive empty scans before giving up on headpats
_MAX_EMPTY_SCANS = 4
# Max headpats per floor.  Each floor can seat up to 8 students at a
# time (Schale Cafe 1F default chair count) — plus the lounging spots
# add a couple more.  Bumped 7 → 10 because user reported a missed
# headpat on 1F (run_20260513_185751 patted 7, missed at least one).
# Higher cap costs nothing — extra empty scans short-circuit via
# _MAX_EMPTY_SCANS regardless.
_MAX_HEADPATS_PER_FLOOR = 10
_INVITE_MATCH_BUTTON_LIMIT = 4
_INVITE_MATCH_FAVORITE_LIMIT = 12
_INVITE_MATCH_TIME_BUDGET_S = 0.75


def _has_florence_runtime() -> bool:
    return (
        importlib.util.find_spec("einops") is not None
        and importlib.util.find_spec("timm") is not None
    )


class CafeSkill(BaseSkill):
    # Cafe-related lobby entries that may carry a red/yellow dot when
    # there's something to do (earnings / invite slot open / pet ready).
    _LOBBY_DOT_ENTRIES = ["咖啡厅入口", "咖啡厅邀请卷", "咖啡厅收益"]

    def should_run(self, screen):
        return self.dot_on_entry(screen, self._LOBBY_DOT_ENTRIES)

    def __init__(self):
        super().__init__("Cafe")
        # Bumped 100 → 160 (2026-05-13).  Full cafe flow on 1F+2F when
        # both invites need long scrolls to find priority students
        # easily exceeds 100 ticks: 1F invite (~20) + 1F headpat (~30)
        # + switch (~5) + 2F invite (~40 with 9-12 scrolls to find
        # priority) + 2F headpat (~30) ≈ 125 ticks.  Run 2026-05-13
        # ~22:30 hit timeout right after starting 2F headpat (Wakamo
        # invite succeeded, then "centering cafe view" → timeout).
        self.max_ticks = 160
        self._enter_attempts: int = 0
        self._headpat_count: int = 0
        self._empty_scans: int = 0
        self._earnings_claimed: bool = False
        self._earnings_attempts: int = 0
        self._invite_attempted: bool = False
        self._invite_ticks: int = 0
        self._invite_stage: int = 0  # 0=open ticket, 1=sort, 2=find+invite, 3=confirm, 4=done
        self._invite_next_state: str = "headpat"  # where to go after invite
        self._pan_phase: int = 0  # 0=not started, 1=panned right, 2=panned left, 3=done
        self._target_favorites: List[str] = []
        self._avatar_matcher = None
        self._invite_scroll_count: int = 0
        # After firing a swipe, wait this many ticks before re-OCR to let
        # the animation settle.  Without this gate, every tick was firing
        # a fresh swipe before the previous one's animation completed →
        # OCR captured mid-blur frames → "no fav found" → swipe again →
        # 15 swipes burned in 15 ticks (run_20260504_221729 t033-047).
        self._invite_swipe_cooldown: int = 0
        # Last visible student signature (joined names string).  When two
        # consecutive scans return the same signature, the list is stuck
        # at the bottom — stop scrolling.
        self._invite_last_signature: str = ""
        self._invite_signature_repeat: int = 0
        self._invite_sorted: bool = False  # True once sorted by 精選
        self._sort_option_clicked: bool = False  # True after clicking 精選 option, waiting for 確認
        # Names invited this cafe run (1F pick so 2F can skip duplicate).
        # Stored as English filenames (e.g. "aru", "saori_(Dress)").
        self._invited_names: Set[str] = set()
        self._headpat_cooldown: int = 0
        self._1f_headpat_started: bool = False  # True once 1F headpat phase begins
        self._1f_done: bool = False  # True once 1F headpat is complete (switch to 2F)
        self._florence_matcher = None
        self._florence_vision = None

    def reset(self) -> None:
        super().reset()
        self._enter_attempts = 0
        self._headpat_count = 0
        self._empty_scans = 0
        self._friend_dodge_count = 0
        self._earnings_claimed = False
        self._earnings_attempts = 0
        self._invite_attempted = False
        self._invite_ticks = 0
        self._invite_stage = 0
        self._invite_next_state = "headpat"
        self._pan_phase = 0
        self._invite_scroll_count = 0
        self._invite_swipe_cooldown = 0
        self._invite_last_signature = ""
        self._invite_signature_repeat = 0
        self._invite_sorted = False
        self._sort_option_clicked = False
        # Restore cross-instance invited tracking for today's game day so
        # a cafe retry (after a previous timeout) doesn't re-invite the
        # same student and waste a ticket.  State auto-expires at the
        # next 04:00 reset — see _game_day().
        self._invited_names = set()
        try:
            saved = _load_cafe_state()
            if saved.get("game_day") == _game_day():
                restored = [str(n) for n in saved.get("invited_names", []) if n]
                if restored:
                    self._invited_names = set(restored)
                    self.log(f"restored invited_names from disk: {sorted(self._invited_names)}")
        except Exception:
            pass
        self._headpat_cooldown = 0
        self._1f_headpat_started = False
        self._1f_done = False
        self._switch_wait_ticks = 0
        self._florence_matcher = None
        self._florence_vision = None
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

    def _load_screen_image(self, screen: ScreenState):
        try:
            import cv2
            import numpy as np
            img = cv2.imdecode(
                np.fromfile(screen.screenshot_path, dtype=np.uint8),
                cv2.IMREAD_COLOR
            )
            if img is None:
                return None, 0, 0
            h, w = img.shape[:2]
            return img, w, h
        except Exception:
            return None, 0, 0

    def _find_nearest_invite_button(self, invite_btns, avatar_cy: float) -> Optional[Any]:
        best_btn = None
        best_dist = 999.0
        for btn in invite_btns:
            dist = abs(btn.cy - avatar_cy)
            if dist < best_dist:
                best_dist = dist
                best_btn = btn
        if best_btn and best_dist < 0.10:
            return best_btn
        return None

    def _find_close_button(self, screen: ScreenState, region=(0.62, 0.06, 0.94, 0.30)) -> Optional[Any]:
        """Find the popup close-X via YOLO cls (UC.BTN_CLOSE_X). No OCR."""
        return self.find_cls(screen, UC.BTN_CLOSE_X, conf=0.30, region=region)

    def _invite_avatar_roi(self, img, w: int, h: int, invite_btn) -> Optional[Any]:
        cy = int(invite_btn.cy * h)
        x1 = max(0, int(0.04 * w))
        x2 = min(w, int(0.22 * w))
        y1 = max(0, cy - int(0.085 * h))
        y2 = min(h, cy + int(0.085 * h))
        roi = img[y1:y2, x1:x2]
        if roi is None or getattr(roi, "size", 0) == 0:
            return None
        return roi

    def _florence_button_enabled(self, screen: ScreenState, region, *, hint: str, default: bool = True) -> bool:
        img, w, h = self._load_screen_image(screen)
        if img is None or w <= 0 or h <= 0:
            return default
        rx1, ry1, rx2, ry2 = region
        x1 = max(0, int(rx1 * w))
        y1 = max(0, int(ry1 * h))
        x2 = min(w, int(rx2 * w))
        y2 = min(h, int(ry2 * h))
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return default
        try:
            if self._florence_vision is None:
                from vision.florence_vision import get_florence_vision
                self._florence_vision = get_florence_vision()
                self.log("Florence vision loaded for cafe button-state checks")
            return self._florence_vision.classify_button_enabled(crop, hint=hint, default=default)
        except Exception as e:
            self.log(f"Florence button-state unavailable: {e}")
            return default

    def _invite_visible_signature(self, invite_btns) -> str:
        """Return a stable signature of the invite list's current scroll
        position — used to detect when scrolling has stopped advancing
        (list bottom reached).

        Pure YOLO (OCR off): the per-row CAFE_INVITE_BTN y-positions move as
        the list scrolls and stop moving at the bottom, so the rounded set of
        button cy values is a reliable, name-free position fingerprint. (The
        old version read OCR student names; that signal is gone with OCR off.)
        """
        if not invite_btns:
            return ""
        ys = sorted(round(b.cy, 2) for b in invite_btns)
        return "|".join(f"{y:.2f}" for y in ys)

    def _find_favorite_in_invite(self, screen: ScreenState, invite_btns, floor: int = 1) -> Optional[Tuple[Any, str, bool]]:
        """Find a favorite student in the MomoTalk invite list.

        Priority-based: 1F invites the #1 priority favorite, 2F invites #2.
        Skips students whose English name is already in self._invited_names
        (so 2F doesn't pick the same person 1F already invited).

        Returns (invite_button, english_name, is_priority) tuple, or None.
        `is_priority=True` only when the floor's priority target was hit
        (the caller can choose to scroll further when it's just a fallback).
        If the priority target is not visible, falls back to any favorite.

        Strategy (fastest to slowest):
        1. OCR name matching: read Chinese student names from the list,
           map them to English filenames via student_name_map.json,
           and check if any are in the favorites config.
        2. Avatar template matching (fallback): crop each avatar ROI
           and compare against reference images.

        Returns the invite button (OcrBox) nearest to the matched row,
        or None if no favorite is found.
        """
        if not self._target_favorites:
            return None

        if not invite_btns:
            return None

        # Build set of favorite filenames (without .png) for fast lookup
        fav_set = set()
        for name in self._target_favorites:
            base = name[:-4] if name.lower().endswith(".png") else name
            fav_set.add(base)

        # Priority target: 1F uses favorites[0], 2F uses favorites[1]
        priority_idx = 0 if floor == 1 else 1
        priority_target = None
        if priority_idx < len(self._target_favorites):
            raw = self._target_favorites[priority_idx]
            priority_target = raw[:-4] if raw.lower().endswith(".png") else raw

        excluded = self._invited_names
        if excluded:
            self.log(f"excluding already-invited: {sorted(excluded)}")

        # --- Strategy 1: OCR name matching \u2014 REMOVED (OCR off) ---
        # This read student names from screen.ocr_boxes and mapped them via
        # student_name_map.json. With OCR globally disabled the name read is
        # impossible, so name matching is dropped and avatar template matching
        # (Strategy 2, below) becomes the sole favorite-finder. Avatar matching
        # is NOT OCR \u2014 it crops each row's avatar and compares against reference
        # images \u2014 so it keeps working end-to-end. If a fast name-based match is
        # wanted back, it returns when digit/text-OCR is re-enabled.

        # --- Strategy 2: Avatar template matching (primary, OCR-free) ---
        candidate_buttons = sorted(invite_btns, key=lambda b: b.cy)[:_INVITE_MATCH_BUTTON_LIMIT]
        target_names = [n for n in self._target_favorites[:_INVITE_MATCH_FAVORITE_LIMIT]
                        if (n[:-4] if n.lower().endswith(".png") else n) not in excluded]

        img, w, h = self._load_screen_image(screen)
        if img is None:
            return None

        deadline = time.perf_counter() + _INVITE_MATCH_TIME_BUDGET_S
        for btn in candidate_buttons:
            if time.perf_counter() >= deadline:
                break
            roi = self._invite_avatar_roi(img, w, h, btn)
            if roi is None or roi.size == 0:
                continue
            if self._avatar_matcher is not None:
                matched_name, score = self._avatar_matcher.match_avatar(
                    roi, target_names
                )
                if matched_name and score > _AVATAR_MATCH_THRESHOLD:
                    base = matched_name[:-4] if matched_name.lower().endswith(".png") else matched_name
                    is_pri = priority_target is not None and base == priority_target
                    self.log(f"AVATAR MATCH: '{matched_name}' score={score:.2f} at ({btn.cx:.2f},{btn.cy:.2f}) floor={floor} priority={is_pri}")
                    return btn, base, is_pri

        return None

    def _is_cafe(self, screen: ScreenState) -> bool:
        """Detect cafe interior via YOLO cls signature.

        Cafe page = any of CAFE_EARNINGS / CAFE_INVITE_TICKET /
        CAFE_MOVE_1F / CAFE_MOVE_2F (see ui_classes.PAGE_SIGNATURES["Cafe"]).
        Resolved through detect_screen_yolo so there is one source of truth.
        """
        return self.detect_screen_yolo(screen) == "Cafe"

    def _looks_like_lobby(self, screen: ScreenState) -> bool:
        """Lobby detector via YOLO nav-icon signature (>=2 nav icons)."""
        return self.detect_screen_yolo(screen) == "Lobby"

    def _invite_confirm_visible(self, screen: ScreenState) -> bool:
        """Detect the invite confirmation popup via YOLO cls.

        The "邀請XXX到咖啡廳" dialog shows a blue BTN_CONFIRM (+ BTN_CANCEL)
        in the center. Seeing the confirm button in the center region is the
        signal. No OCR.
        """
        return self.find_cls(
            screen, UC.BTN_CONFIRM, conf=0.30, region=(0.42, 0.55, 0.78, 0.85)
        ) is not None

    def _invite_list_visible(self, screen: ScreenState) -> bool:
        """Detect the MomoTalk invite list overlay via YOLO cls.

        The list shows per-row CAFE_INVITE_BTN (邀請键). Seeing one in the
        right-hand button column = list is open. No OCR.
        """
        return self.find_cls(
            screen, UC.CAFE_INVITE_BTN, conf=0.30, region=(0.50, 0.20, 0.70, 0.90)
        ) is not None

    def _recover_invite_overlay(self, screen: ScreenState, phase_name: str) -> Optional[Dict[str, Any]]:
        """If invite UI is still visible, dismiss it before proceeding."""
        if self._invite_confirm_visible(screen):
            # YOLO '确认键' (BTN_CONFIRM) — direct cls hit; the blue confirm
            # button has small text that OCR struggles with.
            confirm = self.find_cls(
                screen, UC.BTN_CONFIRM, conf=0.30, region=(0.42, 0.55, 0.78, 0.85),
            )
            if confirm:
                self.log(f"invite confirm still visible before {phase_name}, clicking confirm")
                return action_click_box(confirm, f"confirm invite before {phase_name} (YOLO 确认键)")
            # cls gap: confirm popup signalled but BTN_CONFIRM not resolved.
            self.log(f"invite confirm visible but BTN_CONFIRM not found before {phase_name} — YOLO gap; waiting")
            return action_wait(400, f"waiting for BTN_CONFIRM before {phase_name}")
        if self._invite_list_visible(screen):
            close_btn = self._find_close_button(screen, region=(0.56, 0.04, 0.90, 0.24))
            if close_btn:
                self.log(f"invite list still visible before {phase_name}, closing")
                return action_click_box(close_btn, f"close invite list before {phase_name}")
            self.log(f"invite list still visible before {phase_name}, pressing back")
            return action_back(f"close invite list before {phase_name}")
        return None

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1

        if self.ticks >= self.max_ticks:
            self.log("timeout, exiting")
            return action_done("cafe timeout")

        # ── Handle popups that can appear at any point ──

        # Earnings popup guard — pure YOLO. There is NO cls for the popup
        # TITLE ("每小時收益/收益現況"), so detect the popup by a CLAIM button
        # (active OR grey) sitting in the centered claim region, and/or the
        # CAFE_EARNINGS label leaking through. Both the active-claim and
        # grey-claim handling below still run as before.
        # Skip if we already claimed — prevents infinite loop when inventory is full.
        _earnings_popup_claim = self.find_cls(
            screen,
            [UC.CLAIM_REWARD_YELLOW, UC.CLAIM_YELLOW, UC.CLAIM_BLUE,
             UC.CLAIM_REWARD_GREY, UC.CLAIM_GREY],
            conf=0.30, region=(0.30, 0.58, 0.70, 0.86),
        )
        _earnings_popup_open = (
            _earnings_popup_claim is not None
            or self.find_cls(
                screen, UC.CAFE_EARNINGS, conf=0.30, region=(0.30, 0.04, 0.70, 0.40)
            ) is not None
        )
        if not self._earnings_claimed and _earnings_popup_open:
            claim_btn = self.find_cls(
                screen,
                [UC.CLAIM_REWARD_YELLOW, UC.CLAIM_YELLOW, UC.CLAIM_BLUE],
                conf=0.30, region=(0.30, 0.58, 0.70, 0.86),
            )
            if claim_btn:
                self.log(f"earnings popup detected, clicking claim (YOLO {claim_btn.cls_name})")
                self._earnings_claimed = True
                return action_click_box(claim_btn, f"claim earnings from popup (YOLO {claim_btn.cls_name})")
            # No active claim cls — either already-claimed/greyed or YOLO gap.
            # If a grey claim cls is visible, treat as nothing-to-claim and close.
            done_btn = self.find_cls(
                screen, [UC.CLAIM_REWARD_GREY, UC.CLAIM_GREY], conf=0.30,
                region=(0.30, 0.58, 0.70, 0.86),
            )
            if done_btn:
                self.log("earnings popup claim greyed (already claimed/disabled), closing")
                self._earnings_claimed = True
                close_btn = self._find_close_button(screen)
                if close_btn:
                    return action_click_box(close_btn, "close earnings popup (greyed)")
                return action_wait(300, "earnings popup greyed")
            # Popup signalled only by CAFE_EARNINGS label leak but no claim cls
            # resolved in the claim band — genuine YOLO gap; close it so we don't
            # stall, then let the cafe-main flow re-open earnings cleanly.
            self.log("earnings popup (label) but no claim cls in band — YOLO gap; closing")
            close_btn = self._find_close_button(screen)
            if close_btn:
                return action_click_box(close_btn, "close earnings popup (no claim cls)")
            return action_wait(400, "earnings popup: waiting for claim cls")

        # Reward-result popup ("獲得獎勵") after claiming earnings/rewards —
        # tap the GOT_REWARD cls to dismiss. (Replaces relying on OCR/blind
        # taps for the settlement popup.)
        got_reward = self.find_cls(screen, UC.GOT_REWARD, conf=0.30)
        if got_reward is not None:
            self.log("reward-result popup (獲得獎勵), tapping to dismiss (YOLO 获得奖励)")
            return action_click_box(got_reward, "dismiss reward result (YOLO 获得奖励)")

        # Tutorial/説明 popup (cafe 2F first visit, "訪問學生目錄" header).
        # PRIMARY signal stays the reference `cafe_students_arrived` TEMPLATE
        # (template/emoticon model, not OCR — left untouched per migration
        # scope). The OCR keyword fallbacks ("說明"/"訪問學生目錄") are DEAD
        # with OCR off, so they're removed; the template is the sole detector.
        # Close via BTN_CONFIRM / BTN_CLOSE_X cls.
        tutorial = None
        if self.sub_state != "invite":
            tutorial = screen.find_template_one(
                "cafe_students_arrived", region=(0.20, 0.05, 0.80, 0.40),
            )
        if tutorial:
            confirm = self.find_cls(
                screen, UC.BTN_CONFIRM, conf=0.30, region=screen.CENTER,
            )
            if confirm:
                self.log("dismissing tutorial popup (YOLO 确认键)")
                return action_click_box(confirm, "dismiss tutorial (YOLO 确认键)")
            close_btn = self._find_close_button(screen)
            if close_btn:
                return action_click_box(close_btn, "close tutorial X")

        # Notification / invite-confirm popup — pure YOLO. There is NO cls for
        # the generic 通知 TITLE, so we detect the only popup we MUST act on —
        # the invite-confirm dialog ("邀請XXX到咖啡廳") — by its blue BTN_CONFIRM
        # in the center-bottom band. The invite confirm REQUIRES that button
        # (clicking X/取消 cancels the invite and the student never spawns).
        #
        # DIGIT-DEFERRED: the old invite-cooldown skip read the OCR cooldown
        # text ("冷時間過後即可邀請。") to bail out of invite early. With OCR off
        # that read is impossible (and the HH:MM:SS ticket-timer skip in stage 0
        # is likewise deferred). A stuck/empty invite list self-recovers via its
        # own tick budget, so we no longer block on a cooldown string we can't read.
        if self.sub_state == "invite":
            confirm = self.find_cls(
                screen, UC.BTN_CONFIRM, conf=0.30, region=(0.42, 0.55, 0.78, 0.85),
            )
            if confirm is not None:
                self.log("invite-confirm popup, clicking confirm (YOLO 确认键)")
                return action_click_box(confirm, "confirm invite (YOLO 确认键)")
        # Generic 通知 popups (cooldown / system alerts) have no title cls.
        # YOLO-GAP: leave them to the (OCR-disabled) common-popup interceptor /
        # let the state machine continue — do NOT blind-tap. If one ever blocks
        # the flow, the fix is a 通知-title cls, not an OCR guard.

        # Rank-up / bond level up full-screen overlay (羈絆升級 / 地區升級).
        # Use YOLO cls — NOT OCR "好感度": that label is a BA-permanent tag
        # (student cards / invite list / bond bars all show it), so matching
        # it blind-tapped the screen center on normal cafe screens.
        levelup = self.find_cls(
            screen, [UC.BOND_LEVELUP, UC.REGION_LEVELUP], conf=0.30,
        )
        if levelup is not None:
            self.log(f"level-up overlay ({levelup.cls_name}), tapping to dismiss")
            return action_click(0.5, 0.5, "dismiss level-up overlay")

        # Bond level-up screen + stat-text variant: the PRIMARY detector is the
        # BOND_LEVELUP / REGION_LEVELUP cls handled just above. The old OCR
        # stat-text reads (治愈力/最大體力) and the "在咖啡廳…羈絆點數" pre-level-up
        # banner were SECONDARY fallbacks for the same overlay — dead with OCR
        # off and now covered by the cls. No replacement guard needed; if the
        # cls misses a level-up frame, the not-_is_cafe waits below let it settle
        # and the cls catches the next frame.
        # YOLO-GAP: student-profile screen (基本情報/EX技能/神秘解放) had no cls,
        # so the explicit "back out of accidentally-opened profile" recovery is
        # dropped. It does NOT block the main flow — detect_screen_yolo returns
        # non-Cafe for the profile, so the sub-state handlers' not-_is_cafe
        # branches wait/recover (and _enter/_headpat back out after their
        # attempt budgets). Needs a profile-screen cls to restore fast recovery.
        # YOLO-GAP: furniture edit-mode recovery (結束編輯模式) had no cls. Dropped
        # to a no-op; cafe never enters edit mode on its own, so this is only a
        # rare external-state recovery. Needs an edit-mode-exit cls to restore.

        # Generic popups (confirm/cancel dialogs)
        # SKIP when the MomoTalk sort dropdown is open (invite stage 1) —
        # the sort menu has a 確認 button that _handle_common_popups would
        # click prematurely before we select 羈絆等級.
        _in_sort_dropdown = (self.sub_state == "invite"
                             and self._invite_stage == 1
                             and not self._invite_sorted)
        if not _in_sort_dropdown:
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
        self._enter_attempts += 1
        current = self.detect_screen_yolo(screen)

        if current == "Cafe":
            self._enter_attempts = 0
            # If 1F headpat was already done (kicked to lobby by bond level-up),
            # skip earnings/invite and go straight to switch to 2F.
            if self._1f_done:
                self.log("re-entered cafe after bond level-up, switching to 2F")
                self.sub_state = "switch"
                self._headpat_count = 0
                self._empty_scans = 0
                self._pan_phase = 0
                return action_wait(300, "skip to cafe 2F")
            if self._1f_headpat_started or self._headpat_count > 0:
                self.log(f"re-entered cafe, resuming headpat ({self._headpat_count} pats kept)")
                self.sub_state = "headpat"
                self._empty_scans = 0
                self._pan_phase = 0
                return action_wait(300, "resume headpat after re-entry")
            self.log("inside cafe")
            self.sub_state = "earnings"
            return action_wait(500, "entered cafe")

        if current == "Lobby":
            # YOLO 咖啡厅入口 cls (lobby nav cafe icon).
            cafe_nav = self.click_cls(
                screen, UC.NAV_CAFE, "click cafe nav", conf=0.30,
            )
            if cafe_nav:
                return cafe_nav
            self.log("on lobby but no 咖啡厅入口 cls — YOLO gap; waiting")
            return action_wait(400, "waiting for 咖啡厅入口 cls")

        if current is not None and current != "Cafe":
            self.log(f"wrong screen '{current}', backing out")
            return action_back(f"back from {current}")

        # current is None — unknown/transition screen.
        if self._enter_attempts > 8:
            # Don't send back on loading/transition screens — wait instead.
            # YOLO signal (OCR off): an explicit loading spinner, or a frame
            # with almost no detected UI (full-screen art / fade) means we're
            # mid-transition; otherwise YOLO sees UI but no known page → back out.
            if screen.is_loading() or len(screen.yolo_boxes or []) < 2:
                return action_wait(800, "no UI detected, likely loading — waiting")
            return action_back("recover from unknown screen before entering cafe")

        return action_wait(500, "entering cafe")

    def _earnings(self, screen: ScreenState) -> Dict[str, Any]:
        """Claim cafe earnings.

        Flow:
        1. On cafe main screen, click CAFE_EARNINGS (咖啡厅收益) to open popup.
        2. Earnings popup opens showing '每小時收益', '收益現況'.
        3. Click the active CLAIM cls (領取) to claim.
        4. Popup closes via X.

        Pure YOLO: the popup is detected by its CLAIM button (active OR grey)
        in the centered claim band; claiming and the disabled/empty case are
        driven by the button STATE (yellow/blue = claim, grey = nothing to
        claim → close). All clicks are cls.
        """
        if self._earnings_claimed:
            self.sub_state = "invite"
            self._invite_next_state = "headpat"
            self._invite_ticks = 0
            self._empty_scans = 0
            return action_wait(300, "earnings done, moving to invite")

        if not self._is_cafe(screen):
            # If we're clearly on the lobby, go back to enter state
            if self._looks_like_lobby(screen):
                self.log("earnings: on lobby, resetting to enter")
                self.sub_state = "enter"
                self._enter_attempts = 0
                return action_wait(300, "earnings: back on lobby, re-entering cafe")
            return action_wait(500, "waiting for cafe UI")

        # Earnings popup detection — pure YOLO. NO cls for the popup TITLE
        # ("每小時收益/收益現況"), so detect it by a CLAIM button (active OR
        # grey) sitting in the centered claim band, and/or the CAFE_EARNINGS
        # label leaking through near the top of the popup.
        _CLAIM_BAND = (0.30, 0.58, 0.70, 0.86)
        _popup_claim = self.find_cls(
            screen,
            [UC.CLAIM_REWARD_YELLOW, UC.CLAIM_YELLOW, UC.CLAIM_BLUE,
             UC.CLAIM_REWARD_GREY, UC.CLAIM_GREY],
            conf=0.30, region=_CLAIM_BAND,
        )
        _earnings_popup_open = (
            _popup_claim is not None
            or self.find_cls(
                screen, UC.CAFE_EARNINGS, conf=0.30, region=(0.30, 0.04, 0.70, 0.40)
            ) is not None
        )

        # DIGIT-DEFERRED: the old flow read the earnings % (bottom-right, and
        # the in-popup 0% / 0/N balance rows) to decide whether claiming was
        # worthwhile and to skip a 0% claim. With OCR off those digits are
        # unreadable, so we no longer gate on % — instead we drive purely by
        # the claim-button STATE: an ACTIVE (yellow/blue) claim → claim it; a
        # GREY claim → nothing to collect → just close. Re-enable the %-aware
        # "skip 0% to save the cap" optimization once digit-OCR returns.

        # Earnings popup open → resolve the CLAIM button via YOLO cls.
        if _earnings_popup_open:
            # Active (yellow/blue) claim cls → claim.
            claim_btn = self.find_cls(
                screen,
                [UC.CLAIM_REWARD_YELLOW, UC.CLAIM_YELLOW, UC.CLAIM_BLUE],
                conf=0.30, region=_CLAIM_BAND,
            )
            if claim_btn:
                self.log(f"earnings popup open, claiming (YOLO {claim_btn.cls_name})")
                self._earnings_claimed = True
                return action_click_box(claim_btn, f"claim earnings (YOLO {claim_btn.cls_name})")

            # Grey claim cls → button disabled (state read from cls colour).
            # DIGIT-DEFERRED: this grey state now stands in for the old
            # "all-zero balance / 0%" close path — nothing to claim → close.
            done_btn = self.find_cls(
                screen, [UC.CLAIM_REWARD_GREY, UC.CLAIM_GREY], conf=0.30,
                region=_CLAIM_BAND,
            )
            if done_btn:
                self.log("earnings claim greyed (disabled / nothing to claim), skipping")
                self._earnings_claimed = True
                close_btn = self._find_close_button(screen)
                if close_btn:
                    return action_click_box(close_btn, "close earnings popup (greyed)")
                self.sub_state = "invite"
                self._invite_next_state = "headpat"
                self._invite_ticks = 0
                return action_wait(300, "earnings button greyed, skipping")

            self.log("earnings popup open but no claim cls — YOLO gap; waiting")
            return action_wait(400, "earnings popup: waiting for claim cls")

        # Cafe main screen: open the earnings popup via CAFE_EARNINGS cls.
        earn_label = self.find_cls(screen, UC.CAFE_EARNINGS, conf=0.30)
        if earn_label:
            self._earnings_attempts += 1
            self.log("clicking 咖啡厅收益 to open earnings popup (YOLO)")
            return action_click_box(earn_label, "open earnings popup (YOLO 咖啡厅收益)")

        # CAFE_EARNINGS cls not seen yet. DIGIT-DEFERRED: we used to read the
        # bottom-right % to know earnings existed even when the label cls was
        # missed; without that we give the cls a few ticks to appear, then skip
        # so we never stall the daily loop on a missing label.
        if self._earnings_attempts < 3:
            self._earnings_attempts += 1
            self.log("CAFE_EARNINGS cls not found yet — YOLO gap; waiting")
            return action_wait(400, "earnings: waiting for CAFE_EARNINGS cls")

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
            # GUARD: don't transition to headpat while invite UI is still showing
            recover = self._recover_invite_overlay(screen, self._invite_next_state)
            if recover:
                return recover
            self.sub_state = self._invite_next_state
            self._empty_scans = 0
            if self._invite_next_state != "headpat" or not self._1f_headpat_started:
                self._headpat_count = 0
            if self._invite_next_state == "headpat":
                self._1f_headpat_started = True
            return action_wait(300, f"invite done/skip, starting {self._invite_next_state}")

        if not self._is_cafe(screen):
            if self._looks_like_lobby(screen):
                self.log("invite: on lobby, resetting to enter")
                self.sub_state = "enter"
                self._enter_attempts = 0
                return action_wait(300, "invite: back on lobby, re-entering cafe")
            return action_wait(500, "waiting for cafe UI (invite)")

        self._invite_ticks += 1

        # Stage 3: Confirm the invite in the confirmation popup.
        # The "邀請XXX到咖啡廳" dialog REQUIRES the blue 確認 button — clicking
        # 取消/X cancels the invite. Resolve it via YOLO BTN_CONFIRM cls.
        if self._invite_stage == 3:
            confirm = self.find_cls(
                screen, UC.BTN_CONFIRM, conf=0.30, region=(0.42, 0.55, 0.78, 0.85),
            )
            if confirm:
                self.log("confirming student invite (YOLO 确认键)")
                self._invite_stage = 4
                self._invite_attempted = True
                return action_click_box(confirm, "confirm invite (YOLO 确认键)")

            # No BTN_CONFIRM visible. Either the popup is already gone (global
            # 通知 handler confirmed it) or it hasn't rendered yet. We infer
            # "popup gone" from being back on a clean cafe view with no
            # confirm button after a few ticks — no OCR needed.
            if self._invite_ticks >= 3 and self._is_cafe(screen):
                self.log("invite confirm popup gone (BTN_CONFIRM absent, back on cafe)")
                self._invite_attempted = True
                return action_wait(300, "invite done (popup dismissed)")

            # Still waiting for the confirm popup to render.
            if self._invite_ticks >= 15:
                self.log("invite confirm timeout (no BTN_CONFIRM cls), skipping")
                self._invite_attempted = True
            return action_wait(400, "waiting for invite confirm popup (BTN_CONFIRM cls)")

        # Stage 1: (was) sort invite list by 精選 so favorites surface first.
        #
        # YOLO-GAP: the MomoTalk sort UI has NO trained cls — not the sort
        # LABEL (名字/學園/羈絆等級/精選), the 排列 dropdown header, the option
        # grid, nor the dropdown's 確認 button. Every detector here was OCR and
        # is now dead. Per the no-blind-click rule we DON'T fire the old
        # hardcoded dropdown taps blind; sorting is only an optimization
        # (stage 2 scrolls the whole list via CAFE_INVITE_BTN + avatar matching
        # and falls back to the first invite button), so we SKIP sorting and go
        # straight to stage 2. To restore featured-first ordering, train cls for
        # the sort label + 精選 option + sort-confirm button.
        if self._invite_stage == 1:
            if not self._invite_sorted and self._target_favorites:
                # Wait for the list to actually open (per-row CAFE_INVITE_BTN).
                if not self._invite_list_visible(screen):
                    if self._invite_ticks >= 8:
                        self.log("invite list didn't open (no CAFE_INVITE_BTN cls), proceeding")
                        self._invite_sorted = True
                        self._invite_stage = 2
                        self._invite_ticks = 0
                        return action_wait(200, "skip sort (list not open)")
                    return action_wait(300, "waiting for invite list (CAFE_INVITE_BTN cls)")
                # List is open. No sort cls → skip sorting (logged gap once).
                self.log("invite sort skipped — no cls for sort UI (YOLO-GAP); "
                         "proceeding unsorted, stage 2 scans full list")
            self._invite_sorted = True
            self._invite_stage = 2
            self._invite_ticks = 0
            return action_wait(200, "sort skipped/not needed, proceed to invite")

        # Stage 2: Invite list is open, find favorite student or click first.
        #
        # CRITICAL FLOW (rewritten 2026-05-04 per user's spec): the avatar scan
        # finishes BEFORE any swipe.  After every swipe we cooldown 2-3 ticks for
        # the animation to settle, then scan fresh, then decide.  Old code fired
        # one swipe per tick → 15 burned swipes in 15 ticks
        # (run_20260504_221729) → list overshooting target rows.
        # NOTE: favorite matching is now avatar-template-based only (OCR name
        # matching removed with OCR off — see _find_favorite_in_invite).
        if self._invite_stage == 2:
            # Post-swipe settle gate — block scans/decisions while animation runs.
            if self._invite_swipe_cooldown > 0:
                self._invite_swipe_cooldown -= 1
                return action_wait(
                    250,
                    f"post-swipe settle ({self._invite_swipe_cooldown} ticks left)"
                )

            # Per-row 邀請 buttons via YOLO CAFE_INVITE_BTN cls (right column).
            # The favorite-matcher below pairs each button row with the
            # student avatar/name on its left (avatar matching preserved).
            invite_btns = self.find_all_cls(
                screen, UC.CAFE_INVITE_BTN, conf=0.30,
                region=(0.50, 0.20, 0.70, 0.90),
            )
            if invite_btns:
                # Try to find a favorite student via OCR name matching + avatar fallback.
                # Floor detection (robust to retry resets):
                #   1. _invite_next_state == "headpat2" — set when we explicitly
                #      enter 2F invite via _switch_floor.
                #   2. Screen shows "1號店"/"1号店" button — means we're CURRENTLY
                #      on 2F (the switch button takes us TO 1F).
                #   3. invited_names already has a student — implies 1F invite
                #      already happened.
                # Bug fixed (2026-05-13 / run_20260513_185751 t100+): cafe got
                # reset during cafe2 setup, which clobbered _invite_next_state
                # back to default "headpat".  _floor incorrectly computed as 1,
                # priority_target became Rio (already invited / excluded), so
                # Wakamo (the floor-2 priority) only matched as fallback — bot
                # scrolled through the entire list never inviting her.
                _on_2f = (
                    self._invite_next_state == "headpat2"
                    or self.find_cls(
                        screen, UC.CAFE_MOVE_1F, conf=0.30,
                        region=(0.0, 0.03, 0.30, 0.20),
                    ) is not None
                    or len(self._invited_names) >= 1
                )
                if _on_2f and self._invite_next_state != "headpat2":
                    self.log("invite: detected 2F state, repairing _invite_next_state")
                    self._invite_next_state = "headpat2"
                _floor = 2 if _on_2f else 1
                _MAX_SCROLLS = 12
                _SWIPE_COOLDOWN = 3  # ticks of wait after each swipe (~750ms)
                fav_result = self._find_favorite_in_invite(screen, invite_btns, floor=_floor)

                # Build a signature of the current scroll position from the
                # invite-button row y-coords (YOLO) — if two consecutive
                # post-swipe scans return the same set, we've hit the list
                # bottom (or the list isn't scrolling).
                visible_sig = self._invite_visible_signature(invite_btns)

                if fav_result:
                    fav_btn, fav_name, is_priority = fav_result
                    # Priority hit → invite immediately, no more scrolling.
                    if is_priority:
                        self._invited_names.add(fav_name)
                        _save_cafe_state({
                            "game_day": _game_day(),
                            "invited_names": sorted(self._invited_names),
                        })
                        self.log(f"inviting PRIORITY '{fav_name}' at "
                                 f"({fav_btn.cx:.2f},{fav_btn.cy:.2f}) floor={_floor}")
                        self._invite_stage = 3
                        self._invite_ticks = 0
                        return action_click_box(fav_btn, f"invite priority {fav_name}")

                    # Fallback fav found.  Decide: keep hunting priority, or
                    # accept fallback?  Stop hunting if (a) scroll budget
                    # exhausted, OR (b) list bottom detected.
                    list_stuck = (
                        visible_sig
                        and visible_sig == self._invite_last_signature
                    )
                    if list_stuck:
                        self._invite_signature_repeat += 1
                    else:
                        self._invite_signature_repeat = 0
                    self._invite_last_signature = visible_sig

                    if (self._invite_scroll_count >= _MAX_SCROLLS
                            or self._invite_signature_repeat >= 2):
                        reason = ("scroll budget" if self._invite_scroll_count >= _MAX_SCROLLS
                                  else "list bottom (signature repeat)")
                        self._invited_names.add(fav_name)
                        _save_cafe_state({
                            "game_day": _game_day(),
                            "invited_names": sorted(self._invited_names),
                        })
                        self.log(f"accepting fallback '{fav_name}' ({reason}) floor={_floor}")
                        self._invite_stage = 3
                        self._invite_ticks = 0
                        return action_click_box(fav_btn, f"invite fallback {fav_name}")

                    # Keep hunting priority.  Swipe + cooldown.
                    self._invite_scroll_count += 1
                    self._invite_swipe_cooldown = _SWIPE_COOLDOWN
                    self.log(f"fallback '{fav_name}' found but hunting priority — "
                             f"scroll ({self._invite_scroll_count}/{_MAX_SCROLLS}) floor={_floor}")
                    return action_swipe(0.35, 0.68, 0.35, 0.46, 800,
                                        "scroll to hunt priority favorite")

                # No favorite at all on screen — scroll more if budget left.
                list_stuck = (
                    visible_sig
                    and visible_sig == self._invite_last_signature
                )
                if list_stuck:
                    self._invite_signature_repeat += 1
                else:
                    self._invite_signature_repeat = 0
                self._invite_last_signature = visible_sig

                if (self._invite_scroll_count >= _MAX_SCROLLS
                        or self._invite_signature_repeat >= 2):
                    btn = invite_btns[0]
                    reason = ("scroll budget" if self._invite_scroll_count >= _MAX_SCROLLS
                              else "list bottom")
                    self.log(f"no favorite after scrolling ({reason}), clicking first 邀請 "
                             f"at ({btn.cx:.2f},{btn.cy:.2f})")
                    self._invite_stage = 3
                    self._invite_ticks = 0
                    return action_click_box(btn, "invite student (no fav match)")

                self._invite_scroll_count += 1
                self._invite_swipe_cooldown = _SWIPE_COOLDOWN
                self.log(f"no favorite found, scrolling invite list "
                         f"({self._invite_scroll_count}/{_MAX_SCROLLS})")
                return action_swipe(0.35, 0.68, 0.35, 0.46, 800, "scroll invite list")
            if self._invite_ticks in (8, 16):
                self.log("invite list missing after wait, back to stage 0 to re-find ticket cls")
                self._invite_stage = 0
                return action_wait(300, "re-open invite ticket via cls (list missing)")
            if self._invite_ticks >= 20:
                self.log("invite list not found, skipping")
                self._invite_attempted = True
            return action_wait(400, "waiting for invite list")

        # Close any leftover earnings popup before invite (first tick only).
        # Pure YOLO: detect the popup by a CLAIM button (active/grey) in the
        # centered claim band, or the CAFE_EARNINGS label near the popup title.
        if self._invite_ticks == 1:
            _leftover_earnings = (
                self.find_cls(
                    screen,
                    [UC.CLAIM_REWARD_YELLOW, UC.CLAIM_YELLOW, UC.CLAIM_BLUE,
                     UC.CLAIM_REWARD_GREY, UC.CLAIM_GREY],
                    conf=0.30, region=(0.30, 0.58, 0.70, 0.86),
                ) is not None
                or self.find_cls(
                    screen, UC.CAFE_EARNINGS, conf=0.30, region=(0.30, 0.04, 0.70, 0.40)
                ) is not None
            )
            if _leftover_earnings:
                close_btn = self._find_close_button(screen)
                if close_btn:
                    return action_click_box(close_btn, "close earnings popup before invite")

        # Stage 0: Open the invite ticket panel from cafe main screen.

        # FIRST: check if the invite list is already open (e.g. from a
        # previous attempt or toggle race). Must run BEFORE the ticket
        # check — the 邀請券 ticket is still visible behind the MomoTalk
        # overlay, so clicking it would CLOSE the already-open list.
        # List-open = a per-row CAFE_INVITE_BTN is visible.
        if self.find_cls(
            screen, UC.CAFE_INVITE_BTN, conf=0.30, region=(0.50, 0.20, 0.70, 0.90)
        ) is not None:
            self._invite_stage = 1
            self._invite_ticks = 0
            return action_wait(200, "invite list already open")

        # DIGIT-DEFERRED: the regular-ticket cooldown skip read the HH:MM:SS
        # timer above the ticket to bail out of invite when no ticket was ready.
        # That timer is digits-only and unreadable with OCR off, so we no longer
        # pre-skip on it. Instead we attempt the ticket via cls below; if it's on
        # cooldown the opened list yields no CAFE_INVITE_BTN rows and stage 2
        # self-skips on its tick budget (_invite_ticks >= 20). Re-add the
        # cooldown short-circuit when digit-OCR returns to save the wasted ticks.

        # Open the invite ticket via YOLO CAFE_INVITE_TICKET cls. Region is
        # the bottom-right regular-ticket zone — the paid 額外邀請券 lives
        # elsewhere, so a region-constrained hit picks the regular ticket.
        ticket = self.find_cls(
            screen, UC.CAFE_INVITE_TICKET, conf=0.30, region=(0.55, 0.78, 0.82, 0.99),
        )
        if ticket:
            self.log(f"opening invite ticket at ({ticket.cx:.2f},{ticket.cy:.2f}) (YOLO 咖啡厅邀请卷)")
            self._invite_stage = 1
            self._invite_ticks = 0
            return action_click_box(ticket, "open invite ticket (YOLO 咖啡厅邀请卷)")

        if self._invite_ticks >= 10:
            self.log("CAFE_INVITE_TICKET cls not found after 10 ticks (YOLO gap), skipping invite")
            self._invite_attempted = True
            self.sub_state = self._invite_next_state
            return action_wait(300, "invite skipped (no CAFE_INVITE_TICKET cls)")

        return action_wait(400, "waiting for CAFE_INVITE_TICKET cls (YOLO gap)")

    # Top-left cafe overlay contains 指定訪問/隨機訪問 buttons which stack on
    # students standing in the bottom-left corner. Clicking a headpat marker
    # here accidentally opens the friend-cafe flow. Instead, pan the camera so
    # the student slides out from under the buttons.
    # Top-left column holds 指定訪問 (y~0.05-0.22) and 隨機訪問 (y~0.22-0.42)
    # buttons. Any click inside this column risks opening the friend-cafe flow.
    _FRIEND_BTN_ZONE = (0.00, 0.00, 0.14, 0.42)  # x1, y1, x2, y2 (normalized)
    _MAX_FRIEND_DODGES = 2  # give up and click anyway after N pans (rare edge)

    def _maybe_dodge_friend_buttons(self, mx: float, my: float,
                                    cx: float, cy: float):
        """Return a pan action if the marker/click overlaps the friend-visit
        buttons at top-left, else None. Uses `_friend_dodge_count` to avoid
        infinite loops when the student truly can't be moved out of the zone.
        """
        zx1, zy1, zx2, zy2 = self._FRIEND_BTN_ZONE
        in_zone = (
            (zx1 <= mx <= zx2 and zy1 <= my <= zy2)
            or (zx1 <= cx <= zx2 and zy1 <= cy <= zy2)
        )
        if not in_zone:
            return None
        dodges = getattr(self, "_friend_dodge_count", 0)
        if dodges >= self._MAX_FRIEND_DODGES:
            self.log(f"friend-btn dodge budget exhausted ({dodges}), clicking anyway")
            return None
        self._friend_dodge_count = dodges + 1
        self._headpat_cooldown = 1
        self.log(
            f"headpat marker at ({mx:.2f},{my:.2f}) overlaps 指定/隨機訪問 buttons — "
            f"panning cafe right to dodge ({self._friend_dodge_count}/{self._MAX_FRIEND_DODGES})"
        )
        # Drag the cafe content right+down so the student slides away from the
        # top-left buttons. Start from center, end toward bottom-right.
        return action_swipe(0.30, 0.40, 0.70, 0.60, 400,
                            "pan cafe to dodge friend-visit buttons")

    def _headpat(self, screen: ScreenState) -> Dict[str, Any]:
        """Tap students with happy_face template markers (primary) or YOLO (fallback).

        Camera panning order (template-based):
        - 1F: left→right (pan left first to reveal left corner, then right)
        - 2F: right→left (pan right first to reveal right corner, then left)
        Phase 0: zoom out.
        Phase 1: pan first direction, then scan.
        Phase 2: scan current view.
        Phase 3: pan opposite direction, then scan.
        Phase 4: scan current view.
        Phase 5: done panning, scan only.

        After _MAX_EMPTY_SCANS consecutive ticks with no marks in current view,
        advance pan phase. When all phases exhausted, move on.
        """
        if not self._is_cafe(screen):
            if screen.is_lobby():
                if self.sub_state == "headpat2":
                    self.log("back in lobby during headpat2, done")
                    return action_done("back in lobby")
                if self._headpat_count >= 5:
                    self._1f_done = True
                    self.log(f"lobby during headpat1 after {self._headpat_count} pats, resuming from cafe 2F")
                # On 1F headpat: lobby means we got kicked out — try to re-enter cafe
                self.log("lobby during headpat1, re-entering cafe")
                self.sub_state = "enter"
                return action_wait(300, "re-enter cafe from lobby")
            if self.sub_state == "headpat2":
                self.log("lost cafe during headpat2, exiting cafe flow")
                self.sub_state = "exit"
                return action_wait(300, "exit cafe after headpat2 recovery")
            return action_wait(300, "waiting for cafe")

        # GUARD: if invite overlay leaked into headpat state, dismiss it first
        recover = self._recover_invite_overlay(screen, self.sub_state)
        if recover:
            return recover

        # NOTE: friend-cafe confirm popup ("要訪問好友的咖啡廳嗎？") is handled
        # by base._handle_common_popups which is gated by the "通知" dialog
        # title. Do NOT re-check for 指定訪問/隨機訪問 text here — those strings
        # are always-visible BUTTON LABELS on the normal cafe screen, and
        # matching them would trigger false action_back() → kick to lobby.

        # Check if we've hit the per-floor headpat limit
        if self._headpat_count >= _MAX_HEADPATS_PER_FLOOR:
            if self.sub_state == "headpat":
                self.log(f"reached max {_MAX_HEADPATS_PER_FLOOR} headpats on 1F, switching")
                self.sub_state = "switch"
                self._1f_done = True
                self._headpat_count = 0
                self._empty_scans = 0
                self._pan_phase = 0
                self._friend_dodge_count = 0
                return action_wait(300, "headpat max reached, switching")
            else:
                self.log(f"reached max {_MAX_HEADPATS_PER_FLOOR} headpats on 2F, exiting")
                self.sub_state = "exit"
                return action_wait(300, "headpat2 max reached, exiting")

        is_2f = (self.sub_state == "headpat2")

        # PRIORITY: Check for headpat emote BEFORE pan/zoom. YOLO often detects
        # Emoticon_Action during transitions (zoom, center, pan) and the emote
        # may fade before the scan phase. User-trained model is very accurate
        # (mAP50 99.5%), so any detection ≥0.5 is reliable — click it immediately.
        # Skip during animation cooldown.
        if not (hasattr(self, '_headpat_cooldown') and self._headpat_cooldown > 0):
            early_mark = screen.find_yolo_one("Emoticon_Action", min_conf=0.40)
            if early_mark is None:
                early_mark = screen.find_yolo_one("headpat_bubble", min_conf=0.40)
            if early_mark is not None:
                click_x = early_mark.cx + 0.03
                click_y = early_mark.cy + 0.02
                dodge = self._maybe_dodge_friend_buttons(early_mark.cx, early_mark.cy, click_x, click_y)
                if dodge is not None:
                    return dodge
                self._empty_scans = 0
                self._headpat_count += 1
                self._headpat_cooldown = 1
                self.log(f"early headpat #{self._headpat_count}: cls={getattr(early_mark,'cls','?')} "
                         f"conf={early_mark.confidence:.2f} at ({early_mark.cx:.2f},{early_mark.cy:.2f}) "
                         f"pan_phase={self._pan_phase}")
                return action_click(click_x, click_y, f"early headpat student #{self._headpat_count}")

        # Zoom-out disabled per user request (2026-04-19) — it wasn't helping
        # visibility and introduced camera noise. Phase 0 and 1 now skip
        # directly to centering + pan.
        if self._pan_phase == 0:
            self._pan_phase = 2
            self._empty_scans = 0
            # Drag down slightly to center the cafe view (reference: 709,558→709,309)
            self.log("centering cafe view (drag down, no zoom)")
            # Post-pan cooldown: pan animation is ~600ms; we need at
            # least 2 ticks (~500ms) of settle before scanning for
            # headpat marks, otherwise OCR/template detection runs on
            # blurry mid-pan frames and misses students (cafe1
            # run_20260504_221729 missed at least one student).
            self._headpat_cooldown = 2
            return action_swipe(0.50, 0.60, 0.50, 0.40, 400, "center cafe view down")
        if self._pan_phase == 1:
            # Legacy state — fall through to phase 2 without action
            self._pan_phase = 2
        if self._pan_phase == 2:
            self._pan_phase = 3
            self._empty_scans = 0
            # Both floors pan LEFT. 1F starts showing right side, so pan left
            # reveals the left corner. 2F inherits 1F's final position (right
            # side after second pan) so also starts with pan-left.
            self.log(f"{'2F' if is_2f else '1F'} pan camera: sweep LEFT")
            self._headpat_cooldown = 2  # post-pan settle
            return action_swipe(0.90, 0.50, 0.10, 0.50, 600, f"pan camera left ({'2F' if is_2f else '1F'})")
        if self._pan_phase == 4:
            self._pan_phase = 5
            self._empty_scans = 0
            # Both 1F and 2F need a pan-RIGHT after the pan-LEFT, otherwise
            # students that were visible in the initial view (and panned
            # off the left edge) are lost. Previously 2F skipped this which
            # caused missed headpats (run_20260420_191257 2F had 2 markers
            # in initial view but only 1 got patted).
            floor_tag = "2F" if is_2f else "1F"
            self.log(f"{floor_tag} pan camera: sweep RIGHT")
            self._headpat_cooldown = 2  # post-pan settle
            return action_swipe(0.10, 0.50, 0.90, 0.50, 600, f"pan camera right ({floor_tag} second)")

        # After a successful headpat, wait for the heart animation to finish
        # before scanning again (animation takes ~1 second).
        if hasattr(self, '_headpat_cooldown') and self._headpat_cooldown > 0:
            self._headpat_cooldown -= 1
            return action_wait(500, f"waiting for headpat animation ({self._headpat_cooldown} left)")

        # Find headpat markers — template matching primary (happy_face),
        # YOLO as fallback only.
        mark = screen.find_template_one("happy_face", min_conf=0.75,
                                        region=(0.05, 0.15, 0.98, 0.85))
        if not mark:
            mark = screen.find_template_one("headpat", min_conf=0.78,
                                            region=(0.05, 0.15, 0.98, 0.85))
        if not mark:
            # YOLO fallback — try dedicated emoticon model first, then full model classes
            mark = screen.find_yolo_one("Emoticon_Action", min_conf=_HEADPAT_CONF)
        if not mark:
            mark = screen.find_yolo_one("headpat_bubble", min_conf=_HEADPAT_CONF)
        if not mark:
            mark = screen.find_yolo_one("角色可摸头黄色感叹号", min_conf=_HEADPAT_CONF)
        if not mark:
            mark = screen.find_yolo_one("感叹号", min_conf=_HEADPAT_CONF)

        if mark:
            click_x = mark.cx + 0.03
            click_y = mark.cy + 0.02
            dodge = self._maybe_dodge_friend_buttons(mark.cx, mark.cy, click_x, click_y)
            if dodge is not None:
                return dodge
            self._empty_scans = 0
            self._headpat_count += 1
            self._headpat_cooldown = 1  # Wait 1 tick (~0.5s) for animation
            # Click slightly right of the bubble (student body is just right of bubble)
            self.log(f"headpat #{self._headpat_count}: conf={mark.confidence:.2f} marker=({mark.cx:.2f},{mark.cy:.2f}) click=({click_x:.2f},{click_y:.2f})")
            return action_click(click_x, click_y, f"headpat student #{self._headpat_count}")

        # YOLO-GAP: the bond-progress banner ("在咖啡廳獲得學生的羈絆點數 X/Y") at
        # the top reduces YOLO confidence for headpat marks, so we used to OCR it
        # and pause empty-scan counting while it showed. It has no cls and the
        # read is dead with OCR off, so the guard is dropped. Low risk: the
        # post-headpat _headpat_cooldown already skips scans right after a pat
        # (when this banner appears), and _MAX_EMPTY_SCANS tolerates a stray
        # empty tick. Train a cls for the banner to restore the pause.

        # No marks found this tick
        self._empty_scans += 1

        # After a few empty scans, advance to next pan phase
        if self._empty_scans >= _MAX_EMPTY_SCANS:
            if self._pan_phase < 6:
                # Advance pan phase: 3→4 (triggers second pan), 5→6 (done panning)
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
                self._friend_dodge_count = 0
                return action_wait(300, "headpat done, switching")
            else:  # headpat2
                self.log(f"no more headpat marks on 2F after {self._headpat_count} pats, exiting")
                self.sub_state = "exit"
                return action_wait(300, "headpat2 done, exiting")

        return action_wait(300, f"scanning for headpat marks (empty={self._empty_scans}, pan={self._pan_phase})")

    def _switch_floor(self, screen: ScreenState) -> Dict[str, Any]:
        """Switch from cafe 1F to 2F."""
        # Already on 2F? The switch button reads "移動至一號店" (CAFE_MOVE_1F)
        # when we're standing on 2F (it takes us back to 1F).
        already_2f = self.find_cls(
            screen, UC.CAFE_MOVE_1F, conf=0.30, region=(0.0, 0.03, 0.30, 0.20)
        )
        if already_2f:
            # If this skill instance already ran a headpat cycle AND we're
            # on 2F, it means the cafe started with the player already on
            # 2F (e.g. after a previous-run timeout) and we just finished
            # an inv+pat cycle on 2F.  Don't loop into another invite —
            # just exit.  Without this guard, fixing the OCR pattern match
            # causes a wasted 3rd inv+pat cycle burning invite tickets.
            if self._1f_headpat_started:
                self.log("already on 2F and headpat cycle done, exiting")
                self.sub_state = "exit"
                return action_wait(300, "2F cycle complete, exiting")
            self.log("already on 2F, skipping switch")
            self._invite_attempted = False
            self._invite_ticks = 0
            self._invite_stage = 0
            self._invite_sorted = False
            self._sort_option_clicked = False
            self._invite_scroll_count = 0
            self._invite_next_state = "headpat2"
            self.sub_state = "invite"
            self._empty_scans = 0
            return action_wait(300, "already on 2F, starting invite")

        switch = self.find_cls(screen, UC.CAFE_MOVE_2F, conf=0.30)
        if switch:
            self.log("switching to cafe 2F (YOLO 移动至2号点)")
            # Reset invite state for cafe 2F invite
            self._invite_attempted = False
            self._invite_ticks = 0
            self._invite_stage = 0
            self._invite_sorted = False
            self._sort_option_clicked = False
            self._invite_scroll_count = 0
            self._invite_next_state = "headpat2"
            self.sub_state = "invite"
            self._empty_scans = 0
            return action_click_box(switch, "switch to cafe 2F (YOLO 移动至2号点)")

        # Use a dedicated counter instead of self.ticks (which counts total skill ticks)
        if not hasattr(self, '_switch_wait_ticks'):
            self._switch_wait_ticks = 0
        self._switch_wait_ticks += 1
        if self._switch_wait_ticks > 8:
            self.log("switch timeout after 8 ticks, skipping 2F")
            self._switch_wait_ticks = 0
            self.sub_state = "exit"
            return action_wait(200, "switch timeout")

        # YOLO-GAP: the floor-transition "TAP TO START" prompt is a full-screen
        # tap-anywhere overlay with NO cls (its text read is dead with OCR off).
        # If we see an explicit loading spinner, just wait it out. Otherwise,
        # when neither cls page nor switch button is resolving, give a single
        # gated center-bottom nudge (not every frame) to advance a possible
        # TAP-TO-START prompt; the wait-tick timeout above bounds this. Train a
        # cls for the prompt to replace the blind nudge.
        if screen.is_loading():
            return action_wait(800, "cafe switch loading")
        if (self._switch_wait_ticks in (3, 6)
                and self.detect_screen_yolo(screen) is None):
            self.log("switch: possible TAP-TO-START (no cls) — gated nudge (YOLO-GAP)")
            return action_click(0.5, 0.85, "nudge tap-to-start during cafe switch (YOLO-GAP)")

        return action_wait(500, "waiting for switch button")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        """Return to lobby from cafe."""
        if self.detect_screen_yolo(screen) == "Lobby":
            self.log("back in lobby, cafe done")
            return action_done("cafe complete")

        # Prefer YOLO home/back button over a blind ESC.
        home = self.find_cls(screen, UC.BTN_HOME, conf=0.30)
        if home is not None:
            return action_click_box(home, "cafe exit: home button (YOLO 回大厅按钮)")
        back = self.find_cls(screen, UC.BTN_BACK, conf=0.30)
        if back is not None:
            return action_click_box(back, "cafe exit: back button (YOLO 返回键)")
        return action_back("cafe exit: ESC toward lobby")
