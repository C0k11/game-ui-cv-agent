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
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import unquote

from brain.skills.base import (
    BaseSkill, ScreenState,
    action_click, action_click_box,
    action_wait, action_back, action_done, action_swipe, action_scroll,
)


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
# Max headpats per floor (cafe typically has 3-5 students per floor)
_MAX_HEADPATS_PER_FLOOR = 7
_INVITE_MATCH_BUTTON_LIMIT = 4
_INVITE_MATCH_FAVORITE_LIMIT = 12
_INVITE_MATCH_TIME_BUDGET_S = 0.75


def _has_florence_runtime() -> bool:
    return (
        importlib.util.find_spec("einops") is not None
        and importlib.util.find_spec("timm") is not None
    )


class CafeSkill(BaseSkill):
    def __init__(self):
        super().__init__("Cafe")
        self.max_ticks = 100
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
        self._invite_sorted = False
        self._sort_option_clicked = False
        self._invited_names = set()
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
        return screen.find_text_one(r"^[Xx×]$", region=region, min_conf=0.55)

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

        # --- Strategy 1: OCR name matching (fast, reliable) ---
        if _STUDENT_NAME_MAP:
            priority_btn = None
            any_fav_btn = None
            any_fav_name = None
            any_fav_info = None
            for box in screen.ocr_boxes:
                if box.confidence < 0.55:
                    continue
                if not (0.30 <= box.x1 <= 0.52 and 0.15 <= box.y1 <= 0.90):
                    continue
                text = box.text.replace("\uff08", "(").replace("\uff09", ")").strip()
                en_name = _STUDENT_NAME_MAP.get(text)
                if en_name and en_name in fav_set and en_name not in excluded:
                    btn = self._find_nearest_invite_button(invite_btns, box.cy)
                    if not btn:
                        continue
                    if priority_target and en_name == priority_target:
                        self.log(f"OCR PRIORITY #{priority_idx+1} MATCH: '{text}'\u2192'{en_name}' floor={floor}")
                        return btn, en_name, True
                    if any_fav_btn is None:
                        any_fav_btn = btn
                        any_fav_name = en_name
                        any_fav_info = f"'{text}'\u2192'{en_name}'"
            if any_fav_btn:
                self.log(f"OCR FALLBACK MATCH: {any_fav_info} (priority #{priority_idx+1} not found) floor={floor}")
                return any_fav_btn, any_fav_name, False

        # --- Strategy 2: Avatar template matching (fallback) ---
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

    def _looks_like_lobby(self, screen: ScreenState) -> bool:
        """Fallback lobby detector when strict screen classification misses."""
        nav_tokens = ["課程", "课程", "社交", "商店", "製造", "制造", "招募", "學生", "学生"]
        hits = 0
        for token in nav_tokens:
            if screen.find_text_one(token, region=screen.NAV_BAR, min_conf=0.5):
                hits += 1
        return hits >= 2

    def _invite_confirm_visible(self, screen: ScreenState) -> bool:
        """Detect if the invite confirmation popup is still on screen."""
        if screen.find_text_one(r"邀.*咖啡", region=screen.CENTER, min_conf=0.5):
            return True
        if screen.find_any_text(["要把正在拜訪", "要把正在拜访"], region=screen.CENTER, min_conf=0.5):
            return True
        cancel_btn = screen.find_any_text(["取消"], region=(0.28, 0.60, 0.52, 0.80), min_conf=0.6)
        target_store = screen.find_any_text(["1號店", "1号店", "2號店", "2号店"], region=screen.CENTER, min_conf=0.5)
        return bool(cancel_btn and target_store)

    def _invite_list_visible(self, screen: ScreenState) -> bool:
        """Detect if the MomoTalk invite list overlay is still open."""
        momo = screen.find_any_text(["MomoTalk"], region=(0.25, 0.08, 0.58, 0.18), min_conf=0.6)
        if momo:
            return True
        invite_btn = screen.find_any_text(["邀請", "邀请", "邀睛"], region=(0.50, 0.20, 0.70, 0.90), min_conf=0.5)
        student_label = screen.find_text_one(r"學生.{0,3}\d", region=(0.25, 0.16, 0.52, 0.28), min_conf=0.55)
        return bool(invite_btn and student_label)

    def _recover_invite_overlay(self, screen: ScreenState, phase_name: str) -> Optional[Dict[str, Any]]:
        """If invite UI is still visible, dismiss it before proceeding."""
        if self._invite_confirm_visible(screen):
            confirm = screen.find_any_text(
                ["確認", "确认", "確定", "确定", "確", "确"],
                region=(0.42, 0.60, 0.74, 0.82), min_conf=0.55
            )
            if confirm:
                self.log(f"invite confirm still visible before {phase_name}, clicking confirm")
                return action_click_box(confirm, f"confirm invite before {phase_name}")
            self.log(f"invite confirm still visible before {phase_name}, fallback confirm")
            return action_click(0.598, 0.701, f"confirm invite before {phase_name} (fallback)")
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

        # Earnings popup: only triggers on popup-specific text
        # (NOT '咖啡廳收益' which is a permanent label on cafe main screen)
        # Skip if we already claimed — prevents infinite loop when inventory is full.
        if not self._earnings_claimed and screen.find_any_text(["每小時收益", "收益現况", "收益現況"], min_conf=0.6):
            claim_btn = screen.find_any_text(["領取", "领取"], min_conf=0.7)
            if claim_btn:
                self.log("earnings popup detected, clicking claim")
                self._earnings_claimed = True
                return action_click_box(claim_btn, "claim earnings from popup")
            enabled = self._florence_button_enabled(
                screen,
                (0.35, 0.66, 0.66, 0.80),
                hint="earnings claim button",
                default=True,
            )
            if enabled:
                self.log("earnings popup detected, claim text missing -> click claim fallback")
                self._earnings_claimed = True
                return action_click(0.5, 0.734, "claim earnings fallback")
            self.log("earnings popup detected but Florence says button is disabled")
            self._earnings_claimed = True
            close_btn = self._find_close_button(screen)
            if close_btn:
                return action_click_box(close_btn, "close earnings popup (disabled)")
            return action_wait(300, "earnings popup disabled")

        # Tutorial/説明 popup (cafe 2F first visit)
        # OCR sometimes only detects single char '明' instead of '說明'.
        tutorial = screen.find_any_text(
            ["說明", "说明", "説明", "明"],
            region=(0.3, 0.1, 0.7, 0.3), min_conf=0.5
        )
        if not tutorial:
            tutorial = screen.find_any_text(
                ["訪問學生目錄", "訪問學生", "訪问学生目录", "訪间學生", "訪周學生目緣", "訪周學生", "學生目緣"],
                region=screen.CENTER, min_conf=0.5
            )
        if tutorial:
            confirm = screen.find_any_text(
                ["確認", "确认", "確", "确"],
                region=screen.CENTER, min_conf=0.7
            )
            if confirm:
                self.log("dismissing tutorial popup")
                return action_click_box(confirm, "dismiss tutorial")
            close_btn = self._find_close_button(screen)
            if close_btn:
                return action_click_box(close_btn, "close tutorial X")

        # Notification popup (通知) — e.g. invite cooldown, generic alerts
        # Has "通知" title + "確" button, no cancel; always safe to dismiss.
        notif = screen.find_text_one("通知", region=(0.35, 0.15, 0.65, 0.30), min_conf=0.8)
        if notif:
            # Detect invite cooldown notification ("冷時間過後即可邀請。")
            # If we're in invite phase and see cooldown text, skip invite entirely.
            if self.sub_state == "invite" and not self._invite_attempted:
                cd_hint = screen.find_any_text(
                    ["冷時間", "冷却", "冷印", "即可邀請", "即可邀请"],
                    region=(0.25, 0.35, 0.75, 0.60), min_conf=0.5
                )
                if cd_hint:
                    self.log(f"invite cooldown notification detected: '{cd_hint.text}', skipping invite")
                    self._invite_attempted = True
            confirm = screen.find_any_text(
                ["確認", "确认", "確定", "确定"],
                region=(0.45, 0.68, 0.75, 0.82), min_conf=0.5
            )
            if not confirm:
                confirm = screen.find_text_one(
                    r"^[確确]$",
                    region=(0.45, 0.68, 0.75, 0.82), min_conf=0.6
                )
            if confirm:
                self.log(f"notification popup, clicking confirm (sub={self.sub_state})")
                return action_click_box(confirm, "dismiss notification")
            # Invite confirm popup ("邀請XXX到咖啡廳") REQUIRES clicking 確認
            # (right button at ~0.60, 0.75) — clicking X/取消 cancels the invite
            # and the student never spawns in the cafe.
            if self.sub_state == "invite":
                self.log("invite 通知 popup, clicking 確認 (hardcoded)")
                return action_click(0.60, 0.75, "confirm invite (hardcoded)")
            close_btn = self._find_close_button(screen)
            if close_btn:
                return action_click_box(close_btn, "close notification X")
            return action_click(0.5, 0.70, "dismiss notification fallback")

        # Rank-up / bond level up popup (好感度升級 / 羈絆升級)
        # OCR often misreads 羈絆升級 as 鲜升級 due to stylized font.
        if screen.find_any_text(["好感度", "Rank Up"], min_conf=0.6):
            self.log("rank-up popup, tapping to dismiss")
            return action_click(0.5, 0.5, "dismiss rankup popup")

        # Bond level up screen (羈絆升級！) — full-screen animation
        # Detected via stat text at bottom (治愈力/最大體力) which OCR reads reliably.
        # GUARD: exclude student profile screen which also shows 最大體力 but has
        # unique markers like 基本情報, EX技能, Tip!, 神秘解放.
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

        # Furniture edit mode recovery: click "結束編輯模式" to escape
        edit_btn = screen.find_any_text(
            ["結束編輯模式", "结束编辑模式", "結束編輯"],
            region=(0.80, 0.05, 1.0, 0.18), min_conf=0.5
        )
        if edit_btn:
            self.log("EDIT MODE detected, clicking exit button")
            return action_click_box(edit_btn, "exit furniture edit mode")

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
        current = self.detect_current_screen(screen)

        if current == "Cafe" or self._is_cafe(screen):
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

        if current == "Lobby" or self._looks_like_lobby(screen):
            nav = self._nav_to(screen, ["咖啡廳", "咖啡厅", "咖啡"])
            if nav:
                return nav
            if self._enter_attempts >= 3:
                # Bottom nav first slot is cafe in BA lobby.
                return action_click(0.08, 0.95, "click cafe nav (hardcoded fallback)")
            return action_wait(300, "waiting for cafe button")

        if current and current != "Cafe":
            self.log(f"wrong screen '{current}', backing out")
            return action_back(f"back from {current}")

        if self._enter_attempts > 8:
            if self._looks_like_lobby(screen):
                return action_click(0.08, 0.95, "force click cafe nav from lobby-like screen")
            # Don't send back on loading/transition screens (low OCR) — wait instead
            if len(screen.ocr_boxes) < 5:
                return action_wait(800, "low OCR, likely loading — waiting")
            return action_back("recover from unknown screen before entering cafe")

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
            # If we're clearly on the lobby, go back to enter state
            if self._looks_like_lobby(screen):
                self.log("earnings: on lobby, resetting to enter")
                self.sub_state = "enter"
                self._enter_attempts = 0
                return action_wait(300, "earnings: back on lobby, re-entering cafe")
            return action_wait(500, "waiting for cafe UI")

        def _read_earnings_pct() -> float:
            pct_hits = screen.find_text(
                r"(\d{1,3}(?:\.\d+)?)\s*%",
                region=(0.83, 0.86, 0.99, 0.99),
                min_conf=0.5,
            )
            best = -1.0
            for hit in pct_hits:
                m = re.search(r"(\d{1,3}(?:\.\d+)?)", hit.text)
                if not m:
                    continue
                try:
                    best = max(best, float(m.group(1)))
                except Exception:
                    continue
            return best

        # Check for 0% earnings — skip claim entirely
        # OCR reads "0.0%" or "0.0 %" at bottom-right; regex ^0\.0 avoids matching "100.0%"
        zero_pct = screen.find_text_one(
            r"^0\.0", region=(0.85, 0.88, 0.98, 0.98), min_conf=0.7
        )
        if zero_pct:
            self.log(f"earnings 0% ({zero_pct.text}), skipping")
            self._earnings_claimed = True
            # Only close popup if earnings popup is actually open (收益 text visible)
            if screen.find_any_text(["每小時收益", "收益現況", "收益現況"], min_conf=0.6):
                close_btn = self._find_close_button(screen)
                if close_btn:
                    return action_click_box(close_btn, "close earnings popup (0%)")
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
            # Check for zero-balance rows "0 / NNN" anywhere in popup
            zero_rows = screen.find_text(
                r"^0\s*/\s*\d",
                region=(0.20, 0.40, 0.80, 0.75), min_conf=0.60
            )
            # Also check for "0.0 %" indicating no earnings
            zero_pct_in_popup = screen.find_text(
                r"^0\.0",
                region=(0.20, 0.40, 0.80, 0.75), min_conf=0.60
            )
            if len(zero_rows) >= 2 or len(zero_pct_in_popup) >= 2:
                self.log(f"earnings popup all zero ({len(zero_rows)} zero rows, {len(zero_pct_in_popup)} zero pcts), closing")
                self._earnings_claimed = True
                close_btn = self._find_close_button(screen)
                if close_btn:
                    return action_click_box(close_btn, "close empty earnings popup")
                self.sub_state = "invite"
                self._invite_next_state = "headpat"
                self._invite_ticks = 0
                return action_wait(300, "empty earnings popup, skipping")
            # Florence check whether the claim button is enabled.
            # DEFAULT=FALSE: if Florence can't tell, assume disabled (safer than clicking blindly)
            enabled = self._florence_button_enabled(
                screen,
                (0.35, 0.66, 0.66, 0.80),
                hint="earnings claim button",
                default=False,
            )
            if not enabled:
                self.log("earnings popup button appears disabled, skipping claim")
                self._earnings_claimed = True
                close_btn = self._find_close_button(screen)
                if close_btn:
                    return action_click_box(close_btn, "close earnings popup (disabled)")
                self.sub_state = "invite"
                self._invite_next_state = "headpat"
                self._invite_ticks = 0
                return action_wait(300, "earnings button disabled, skipping")
            self.log("earnings popup open, claim enabled, clicking")
            self._earnings_claimed = True
            return action_click(0.5, 0.734, "claim earnings fallback")

        # Cafe main screen: click '咖啡廳收益' label to open earnings popup
        # Only open if FULL! is visible (earnings at max capacity)
        full = screen.find_text_one("FULL", min_conf=0.6)
        if full:
            self.log("FULL detected, clicking earnings area")
            return action_click(0.913, 0.893, "open earnings via FULL")

        # OCR on this label is noisy (e.g. 咖啡魔收益). Match broader regex.
        earn_label_regex = screen.find_text_one(
            r"咖啡.*收益",
            region=(0.82, 0.84, 0.99, 0.98),
            min_conf=0.45,
        )
        if earn_label_regex:
            self._earnings_attempts += 1
            self.log(f"clicking earnings area via regex label '{earn_label_regex.text}'")
            return action_click_box(earn_label_regex, "open earnings popup (regex label)")

        # Also try earnings label if percentage is not 0
        earn_label = screen.find_any_text(
            ["咖啡廳收益", "咖啡收益", "咖啡厅收益"],
            min_conf=0.5
        )
        if earn_label:
            self._earnings_attempts += 1
            self.log("clicking '咖啡廳收益' to open earnings popup")
            return action_click_box(earn_label, "open earnings popup")

        # Last fallback: if visible percentage is >0, try opening earnings by fixed spot.
        # template-based fixed click is much more stable than OCR-only in this corner.
        pct_val = _read_earnings_pct()
        if pct_val > 0.0 and self._earnings_attempts < 3:
            self._earnings_attempts += 1
            self.log(f"earnings percent {pct_val:.1f}% detected, opening earnings by fixed spot")
            return action_click(0.913, 0.893, f"open earnings via percent {pct_val:.1f}%")

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

        # Stage 3: Confirm the invite in the confirmation popup
        if self._invite_stage == 3:
            confirm = screen.find_any_text(
                ["確認", "确认", "確定", "确定", "確", "确"],
                region=screen.CENTER, min_conf=0.7
            )
            if confirm:
                self.log("confirming student invite")
                self._invite_stage = 4
                self._invite_attempted = True
                return action_click_box(confirm, "confirm invite")

            # Invite popup often uses single-char confirm ('確'). If OCR misses it,
            # use popup-body + cancel as a safe signal and click known confirm spot.
            invite_popup = screen.find_text_one(r"邀請.*咖啡廳", region=screen.CENTER, min_conf=0.7)
            cancel_btn = screen.find_any_text(["取消"], region=(0.30, 0.62, 0.50, 0.78), min_conf=0.8)
            if invite_popup and cancel_btn:
                self.log("invite popup detected without confirm OCR, clicking confirm fallback")
                self._invite_stage = 4
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

        # Stage 1: Sort invite list by affinity (so high-affinity = favorites first)
        # MomoTalk header sort controls are at approximately:
        #   sort label ("名字"/"羈絆"): cx ~0.55, cy ~0.21
        #   sort arrows ("≡↑"): cx ~0.64, cy ~0.21
        _SORT_LABEL_REGION = (0.49, 0.17, 0.60, 0.25)
        # When opened, dropdown menu spans ~x=0.29-0.68, y=0.27-0.62
        # containing 排列 header, 名字/學園/羈絆等級/精選 options, 確認 button
        _SORT_MENU_REGION = (0.29, 0.27, 0.68, 0.62)
        # Hardcoded positions inside the sort dropdown (verified from screenshots):
        #   排列 header: y ~0.29
        #   2x2 option grid:
        #     名字 (0.42, 0.37)   學園 (0.58, 0.37)
        #     羈絆等級 (0.42, 0.44)   精選 (0.58, 0.44)
        #   確認 button: (0.50, 0.55)
        _FEATURED_POS = (0.58, 0.44)   # 精選 — starred students first
        _CONFIRM_POS = (0.50, 0.55)    # 確認 button inside sort popup
        if self._invite_stage == 1:
            if not self._invite_sorted and self._target_favorites:
                # First, wait for the MomoTalk list to actually be open.
                list_open = screen.find_any_text(
                    ["MomoTalk", "學生"],
                    region=(0.28, 0.10, 0.50, 0.25), min_conf=0.50
                )
                if not list_open:
                    if self._invite_ticks >= 8:
                        self.log("invite list didn't open for sort, proceeding")
                        self._invite_sorted = True
                        self._invite_stage = 2
                        self._invite_ticks = 0
                        return action_wait(200, "skip sort (list not open)")
                    return action_wait(300, "waiting for invite list to open for sort")

                # Is the sort dropdown popup currently open? (排列 header visible)
                sort_menu_open = screen.find_any_text(
                    ["排列"],
                    region=(0.30, 0.25, 0.60, 0.35), min_conf=0.50
                )

                # If we've already clicked 精選 and popup still open, click 確認 to apply
                if sort_menu_open and self._sort_option_clicked:
                    self.log("clicking 確認 to apply sort (hardcoded)")
                    self._invite_sorted = True
                    self._invite_stage = 2
                    self._invite_ticks = 0
                    self._sort_option_clicked = False
                    return action_click(*_CONFIRM_POS, "confirm sort selection")

                # Popup open but option not yet clicked → click 精選 (hardcoded)
                if sort_menu_open:
                    self.log("sort popup open, clicking 精選 (hardcoded)")
                    self._sort_option_clicked = True
                    self._invite_ticks = 0
                    return action_click(*_FEATURED_POS, "select sort by 精選")

                # Popup not open — check if already sorted (label says 精選)
                featured_label = screen.find_any_text(
                    ["精選", "精选"],
                    region=_SORT_LABEL_REGION, min_conf=0.50
                )
                if featured_label:
                    self.log("invite list already sorted by 精選")
                    self._invite_sorted = True
                    self._invite_stage = 2
                    self._invite_ticks = 0
                    self._sort_option_clicked = False
                    return action_wait(200, "sort confirmed, proceeding")

                # Popup not open and not sorted → click sort label to open dropdown
                sort_label = screen.find_any_text(
                    ["名字", "名宇", "學園", "学园", "羈絆", "羁绊"],
                    region=_SORT_LABEL_REGION, min_conf=0.50
                )
                if sort_label and self._invite_ticks <= 6:
                    self.log(f"current sort '{sort_label.text}', opening sort dropdown")
                    self._sort_option_clicked = False
                    return action_click_box(sort_label, "open sort dropdown")

                # After enough ticks, give up sorting and proceed
                if self._invite_ticks >= 8:
                    self.log("sort switch timeout, proceeding with current sort")
                    self._invite_sorted = True
                    self._invite_stage = 2
                    self._invite_ticks = 0
                    return action_wait(200, "skip sort, find invite btn")

                return action_wait(300, "waiting for sort switch")
            else:
                # Already sorted or no favorites configured
                self._invite_sorted = True
                self._invite_stage = 2
                self._invite_ticks = 0
                return action_wait(200, "sort not needed, proceed to invite")

        # Stage 2: Invite list is open + sorted, find favorite student or click first
        if self._invite_stage == 2:
            # OCR frequently misreads 邀請 as 邀睛
            invite_btns = screen.find_text(
                "邀請", region=(0.50, 0.20, 0.70, 0.90), min_conf=0.50
            )
            if not invite_btns:
                invite_btns = screen.find_text(
                    "邀请", region=(0.50, 0.20, 0.70, 0.90), min_conf=0.50
                )
            if not invite_btns:
                invite_btns = screen.find_text(
                    "邀睛", region=(0.50, 0.20, 0.70, 0.90), min_conf=0.50
                )
            if invite_btns:
                # Try to find a favorite student via OCR name matching + avatar fallback
                _floor = 2 if self._invite_next_state == "headpat2" else 1
                _MAX_SCROLLS = 15
                fav_result = self._find_favorite_in_invite(screen, invite_btns, floor=_floor)
                if fav_result:
                    fav_btn, fav_name, is_priority = fav_result
                    # If we only found a non-priority fallback AND we still have
                    # scrolls left, keep scrolling to hunt the floor's priority
                    # target (e.g. 1F=Rio). Only accept fallback once exhausted.
                    if not is_priority and self._invite_scroll_count < _MAX_SCROLLS:
                        self._invite_scroll_count += 1
                        self.log(f"fallback '{fav_name}' found but hunting priority — "
                                 f"scroll ({self._invite_scroll_count}/{_MAX_SCROLLS}) floor={_floor}")
                        # Slower, shorter swipe → less inertia, less row-skipping.
                        # y delta reduced 0.35→0.22 (≈2 rows), duration 400→800ms.
                        return action_swipe(0.35, 0.68, 0.35, 0.46, 800,
                                            "scroll to hunt priority favorite")
                    self._invited_names.add(fav_name)
                    tag = "priority" if is_priority else "fallback"
                    self.log(f"inviting {tag} '{fav_name}' at ({fav_btn.cx:.2f},{fav_btn.cy:.2f}) floor={_floor}")
                    self._invite_stage = 3
                    self._invite_ticks = 0
                    return action_click_box(fav_btn, f"invite {tag} {fav_name}")
                # No favorite found at all — scroll more, else click first
                if self._invite_scroll_count < _MAX_SCROLLS:
                    self._invite_scroll_count += 1
                    self.log(f"no favorite found, scrolling invite list ({self._invite_scroll_count}/{_MAX_SCROLLS})")
                    return action_swipe(0.35, 0.68, 0.35, 0.46, 800, "scroll invite list")
                btn = invite_btns[0]
                self.log(f"no favorite after scrolling, clicking first 邀請 at ({btn.cx:.2f},{btn.cy:.2f})")
                self._invite_stage = 3
                self._invite_ticks = 0
                return action_click_box(btn, "invite student (no fav match)")
            if self._invite_ticks in (8, 16):
                self.log("invite list missing after wait, retry opening ticket")
                self._invite_stage = 0
                return action_click(0.69, 0.93, "re-open invite ticket (list missing)")
            if self._invite_ticks >= 20:
                self.log("invite list not found, skipping")
                self._invite_attempted = True
            return action_wait(400, "waiting for invite list")

        # Close any leftover earnings popup before invite (first tick only)
        if self._invite_ticks == 1:
            if screen.find_any_text(["每小時收益", "收益現況", "收益現況"], min_conf=0.6):
                close_btn = self._find_close_button(screen)
                if close_btn:
                    return action_click_box(close_btn, "close earnings popup before invite")

        # Stage 0: Open the invite ticket panel from cafe main screen

        # FIRST: check if invite list is already open (e.g. from previous
        # attempt or toggle race).  Must run BEFORE ticket check — the
        # 邀請券 ticket is still visible behind the MomoTalk overlay, so
        # clicking it would CLOSE the already-open list.
        invite_btn = screen.find_any_text(
            ["邀請", "邀请", "邀睛"],
            region=(0.50, 0.20, 0.70, 0.90), min_conf=0.55
        )
        if invite_btn:
            self._invite_stage = 1
            self._invite_ticks = 0
            return action_wait(200, "invite list already open")

        # Skip if cooldown timer (HH:MM:SS) visible near the REGULAR ticket area.
        # NOTE: "可購買" at cy≈0.83 is the EXTRA ticket purchase label — ignore it.
        # The regular ticket cooldown timer appears above the regular ticket (cy≈0.88-0.96).
        cooldown = screen.find_text_one(
            r"\d+[\uff1a:]\d+[\uff1a:]\d+", region=(0.55, 0.78, 0.78, 0.98), min_conf=0.7
        )
        if cooldown:
            self.log(f"invite cooldown ({cooldown.text}), skipping")
            self._invite_attempted = True
            return action_wait(200, f"invite unavailable: {cooldown.text}")

        # Find the REGULAR invite ticket (not the paid 額外邀請券)
        # OCR reads 額外 as "额外", "客外", or "額外" — exclude all variants.
        _EXTRA_PREFIXES = ("客外", "额外", "額外", "客", "外")
        ticket_hits = screen.find_text(
            "邀請券", region=(0.55, 0.78, 0.78, 0.98), min_conf=0.50
        )
        # OCR frequently misreads 邀請券 as 邀睛券
        if not ticket_hits:
            ticket_hits = screen.find_text(
                "邀睛券", region=(0.55, 0.78, 0.78, 0.98), min_conf=0.50
            )
        ticket = None
        for hit in ticket_hits:
            if not any(p in hit.text for p in _EXTRA_PREFIXES):
                ticket = hit
                break
        if not ticket:
            ticket = screen.find_text_one(
                "可使用", region=(0.55, 0.78, 0.78, 0.98), min_conf=0.50
            )
        if ticket:
            self.log(f"clicking invite ticket '{ticket.text}' at ({ticket.cx:.2f},{ticket.cy:.2f})")
            self._invite_stage = 1
            self._invite_ticks = 0
            return action_click_box(ticket, "open invite ticket")

        if self._invite_ticks in (3, 6):
            self.log("invite UI unresolved, retry fixed click on regular ticket")
            return action_click(0.69, 0.93, "open invite ticket (hardcoded retry)")

        if self._invite_ticks >= 10:
            self.log("invite UI not found, skipping invite")
            self._invite_attempted = True
            self.sub_state = self._invite_next_state
            return action_wait(300, "invite skipped")

        return action_wait(400, "waiting for invite UI")

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
            return action_swipe(0.90, 0.50, 0.10, 0.50, 600, f"pan camera left ({'2F' if is_2f else '1F'})")
        if self._pan_phase == 4:
            self._pan_phase = 5
            self._empty_scans = 0
            if is_2f:
                # 2F: single LEFT sweep is enough (small floor, fewer students).
                self.log("2F: skip second pan (one sweep covers 2F)")
                return action_wait(200, "2F single pan sufficient")
            # 1F: pan RIGHT to return and cover the right side
            self.log("1F pan camera: sweep RIGHT")
            return action_swipe(0.10, 0.50, 0.90, 0.50, 600, "pan camera right (1F second)")

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
        # Already on 2F? (button says "移動至1號店" = we're on 2F)
        already_2f = screen.find_any_text(
            ["移動至1號店", "移动至1号店", "1號店"],
            region=(0.0, 0.03, 0.25, 0.12), min_conf=0.5
        )
        if already_2f:
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
            self._invite_sorted = False
            self._sort_option_clicked = False
            self._invite_scroll_count = 0
            self._invite_next_state = "headpat2"
            self.sub_state = "invite"
            self._empty_scans = 0
            return action_click_box(switch, "switch to cafe 2F")

        # TAP TO START during transition
        tap = screen.find_text_one("TAP.*START", min_conf=0.8)
        if tap:
            return action_click(0.5, 0.85, "tap to start during cafe switch")

        # Use a dedicated counter instead of self.ticks (which counts total skill ticks)
        if not hasattr(self, '_switch_wait_ticks'):
            self._switch_wait_ticks = 0
        self._switch_wait_ticks += 1
        if self._switch_wait_ticks > 8:
            self.log("switch timeout after 8 ticks, skipping 2F")
            self._switch_wait_ticks = 0
            self.sub_state = "exit"
            return action_wait(200, "switch timeout")

        return action_wait(500, "waiting for switch button")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        """Return to lobby from cafe."""
        if screen.is_lobby():
            self.log("back in lobby, cafe done")
            return action_done("cafe complete")

        return action_back("cafe exit: press ESC")
