"""StoryMiningSkill — auto-play unplayed chapters in 短篇 / 支線 stories.

Corrected UI model (v2, based on user playtest 2026-04-24)
-----------------------------------------------------------

    1.  **Hub** — the 劇情 page hosts four sub-cards.
            · 主線劇情 (main)  — deferred to a later commit (grid-based push)
            · 短篇劇情 (short) — this skill
            · 支線劇情 (side)  — this skill
            · 重播 (replay)    — ignored (no rewards)

    2.  **Category page** — a *horizontal carousel* of volume cards
        (篇).  Only ONE volume is centred at a time; siblings peek in
        at the edges.  Navigation is via a left/right arrow button at
        the screen edges (~x=0.03 and x=0.97); if the arrow template
        doesn't land we fall back to a centred horizontal swipe.

        A volume card displays a "New" badge (top-left) when it
        contains unplayed chapters.

    3.  **Volume detail** — clicking the centred volume card opens a
        detail view with:
            · big volume art + description on the left
            · chapter list on the right, each row labelled
              「第N章 標題」
        Each chapter row shows a tiny yellow dot when it's playable,
        a lock icon 🔒 when it's gated behind progression, and nothing
        when already cleared.  We don't rely on colour detection for
        this MVP — we use the combination of OCR-ordered rows plus
        our progress ledger to pick the first unplayed chapter in
        sequence.  Locked chapters surface as a modal popup we detect
        and dismiss.

    4.  **Cutscene** — standard MENU → Skip → 確認 dance, same machine
        as event_activity's story branch.

State machine
-------------

    enter        — figure out where we are
    hub          — on 劇情 hub; click the next category card
    category     — on a carousel page; inspect current volume, click
                   it if it has a 'New' badge, else advance carousel
    volume       — on a volume detail page; click the first unplayed
                   chapter row, or back out when none remain
    playing      — inside a chapter cutscene; run the skip dance
    finished     — done; advance pipeline

Main-story push is still deferred; when we encounter 主線劇情 we log
and skip so the skill doesn't hang on an unsupported page.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from brain.skills.base import (
    BaseSkill,
    OcrBox,
    ScreenState,
    action_back,
    action_click,
    action_click_box,
    action_done,
    action_swipe,
    action_wait,
)
from brain.skills.story_progress import StoryProgressStore, get_store


# ── UI regions (normalised 0..1) ────────────────────────────────────────

# Sub-card click centres on the 劇情 hub (from OCR evidence at
# data/trajectories/run_20260423_024430/tick_0004.json).
_HUB_SHORT_CLICK = (0.61, 0.45)
_HUB_SIDE_CLICK = (0.83, 0.45)
_HUB_MAIN_CLICK = (0.27, 0.60)    # deferred
_HUB_REPLAY_CLICK = (0.83, 0.85)  # ignored

# Category-page carousel arrows (sprite-only, no OCR text; derived
# from the user's uploaded arrow screenshots).
_ARROW_RIGHT_CLICK = (0.97, 0.50)
_ARROW_LEFT_CLICK = (0.03, 0.50)

# Centre region of the carousel where the current volume card sits —
# "New" badge detection is restricted here to ignore cards peeking in
# from the sides.
_CAROUSEL_CENTRE_REGION = (0.28, 0.15, 0.70, 0.75)

# Volume-card click target when a New badge is detected inside the
# centre region (slightly below the badge lands on the art centre).
_CAROUSEL_CENTRE_CLICK = (0.47, 0.45)

# Breadcrumb / category title region.
_BREADCRUMB_REGION = (0.0, 0.0, 0.25, 0.09)

# Volume-detail chapter list lives on the right side of the screen.
_CHAPTER_LIST_REGION = (0.55, 0.18, 1.0, 0.85)

# Chapter rows match "第N章 ..." (both Traditional and Simplified).
_CHAPTER_RE = re.compile(r"^第\s*([0-9０-９一二三四五六七八九十]+)\s*章")


# ── Skill ──────────────────────────────────────────────────────────────

class StoryMiningSkill(BaseSkill):
    """Auto-play unplayed chapters in every 短篇 / 支線 volume."""

    def __init__(self) -> None:
        super().__init__("StoryMining")
        self.max_ticks = 600  # many volumes × many chapters × skip ticks

        self._store: StoryProgressStore = get_store()

        self.sub_state = "enter"
        self._enter_ticks: int = 0

        # Which category we're processing (short / side / main).
        self._category: Optional[str] = None
        self._exhausted: List[str] = []

        # Carousel navigation state.  We track the title OCR'd on the
        # centred card; if it doesn't change after an arrow click we
        # assume the arrow didn't register and fall back to a swipe.
        self._current_volume: str = ""
        self._prev_volume: str = ""
        self._carousel_steps: int = 0          # arrow/swipe clicks so far
        self._carousel_stall: bool = False     # last step didn't advance
        # Max volumes to scan before giving up on a category (typical
        # game build ships ~15-20 volumes per category; 30 is a safe
        # upper bound).
        self._CAROUSEL_LIMIT: int = 30

        # Currently-open volume (entered from category carousel).
        self._open_volume: str = ""
        # Chapters inside the open volume we've tried in this visit —
        # guards against an infinite retry loop on a misread row.
        self._tried_chapters: List[str] = []

        # Cutscene we're playing; keyed by (volume, chapter).
        self._pending_volume: str = ""
        self._pending_chapter: str = ""

        # Skip-machine state (same shape as event_activity).
        self._skip_stage: int = 0
        self._cutscene_taps: int = 0

        # Generic stall counter for failure recovery.
        self._stall_ticks: int = 0

    def reset(self) -> None:
        super().reset()
        self.sub_state = "enter"
        self._enter_ticks = 0
        self._category = None
        self._exhausted = []
        self._current_volume = ""
        self._prev_volume = ""
        self._carousel_steps = 0
        self._carousel_stall = False
        self._open_volume = ""
        self._tried_chapters = []
        self._pending_volume = ""
        self._pending_chapter = ""
        self._skip_stage = 0
        self._cutscene_taps = 0
        self._stall_ticks = 0

    # ── Tick dispatch ───────────────────────────────────────────────

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1
        self._enter_ticks += 1

        # Dismiss generic popups (locked-chapter notice, confirmation,
        # etc.) before any sub-state branches so we don't get stuck
        # behind a modal.
        popup_action = self._dismiss_popup(screen)
        if popup_action is not None:
            return popup_action

        handler = {
            "enter": self._on_enter,
            "hub": self._on_hub,
            "category": self._on_category,
            "volume": self._on_volume,
            "playing": self._on_playing,
            "finished": self._on_finished,
        }.get(self.sub_state, self._on_enter)
        return handler(screen)

    # ── Popup handling ─────────────────────────────────────────────

    def _dismiss_popup(self, screen: ScreenState) -> Optional[Dict[str, Any]]:
        """Close a locked-chapter / cannot-enter modal if visible.

        Only fires on screens that *don't* look like our workflow —
        confirmation dialogs on the cutscene skip path are handled in
        _on_playing.
        """
        if self.sub_state == "playing":
            return None
        # Common locked / unavailable messages.
        blocked = screen.find_any_text(
            [
                "尚未開放", "尚未开放", "尚未達成", "尚未达成",
                "無法進入", "无法进入", "請先完成", "请先完成",
                "未解鎖", "未解锁",
            ],
            region=screen.CENTER,
            min_conf=0.55,
        )
        if not blocked:
            return None
        # Prefer an explicit confirm, else tap centre-bottom.
        confirm = screen.find_any_text(
            ["確認", "确认", "確定", "确定", "OK", "Yes", "確", "确"],
            region=(0.30, 0.55, 0.75, 0.82),
            min_conf=0.55,
        )
        self.log(f"locked/blocked popup: '{blocked.text}', dismissing")
        if confirm:
            return action_click_box(confirm, "dismiss locked popup")
        return action_click(0.50, 0.72, "dismiss locked popup (fallback)")

    # ── Sub-state: enter ───────────────────────────────────────────

    def _on_enter(self, screen: ScreenState) -> Dict[str, Any]:
        if self._is_story_hub(screen):
            self.log("on story hub — routing to category picker")
            self.sub_state = "hub"
            return self._on_hub(screen)

        cat = self._detect_category_page(screen)
        if cat is not None:
            if cat == StoryProgressStore.MAIN:
                # Main-story push not implemented yet — back out and
                # mark as exhausted so we don't loop on it.
                if StoryProgressStore.MAIN not in self._exhausted:
                    self._exhausted.append(StoryProgressStore.MAIN)
                self.log("主線劇情 detected but pusher not implemented; backing out")
                return action_back("main story not supported yet")
            if cat in (StoryProgressStore.SHORT, StoryProgressStore.SIDE):
                self.log(f"already on {cat!r} category page")
                self._category = cat
                self._reset_category_state()
                self.sub_state = "category"
                return self._on_category(screen)

        # Not on any recognised story page within 6 ticks — let the
        # pipeline move on.
        if self._enter_ticks > 6:
            self.log("story hub not detected within 6 ticks, skipping")
            return action_done("story hub not reached")
        return action_wait(400, "searching for story hub")

    # ── Sub-state: hub ─────────────────────────────────────────────

    def _on_hub(self, screen: ScreenState) -> Dict[str, Any]:
        for cat, target in (
            (StoryProgressStore.SHORT, _HUB_SHORT_CLICK),
            (StoryProgressStore.SIDE, _HUB_SIDE_CLICK),
        ):
            if cat in self._exhausted:
                continue
            self._category = cat
            self._reset_category_state()
            self.sub_state = "category"
            self.log(f"entering {cat} category")
            return action_click(target[0], target[1], f"open {cat} category")

        if StoryProgressStore.MAIN not in self._exhausted:
            self._exhausted.append(StoryProgressStore.MAIN)
            self.log("主線劇情 push not implemented — skipped")

        self.log("all categories exhausted")
        self.sub_state = "finished"
        return action_done("story mining finished")

    def _reset_category_state(self) -> None:
        self._current_volume = ""
        self._prev_volume = ""
        self._carousel_steps = 0
        self._carousel_stall = False
        self._open_volume = ""
        self._tried_chapters = []
        self._stall_ticks = 0

    # ── Sub-state: category (carousel) ─────────────────────────────

    def _on_category(self, screen: ScreenState) -> Dict[str, Any]:
        # Bounced back to hub? Mark category done & return.
        if self._is_story_hub(screen):
            self.log(f"category '{self._category}' closed unexpectedly — back to hub")
            if self._category:
                self._exhausted.append(self._category)
                self._category = None
            self.sub_state = "hub"
            return self._on_hub(screen)

        # Step 1: who's on centre stage?
        centre_title = self._detect_centre_volume(screen)

        # Detect carousel stall — arrow click didn't change title.
        if centre_title and centre_title == self._prev_volume:
            self._carousel_stall = True
        else:
            self._carousel_stall = False

        if centre_title:
            self._current_volume = centre_title

        # Step 2: if the centre volume has a 'New' badge, enter it.
        new_visible = self._has_new_badge_centre(screen)
        if new_visible and centre_title:
            self.log(f"volume '{centre_title}' has New badge — entering")
            self._open_volume = centre_title
            self._tried_chapters = []
            self.sub_state = "volume"
            return action_click(
                _CAROUSEL_CENTRE_CLICK[0], _CAROUSEL_CENTRE_CLICK[1],
                f"open volume '{centre_title}'",
            )

        # Step 3: no New → advance carousel.
        if self._carousel_steps >= self._CAROUSEL_LIMIT:
            self.log(
                f"scanned {self._carousel_steps} volumes in '{self._category}' "
                f"without finding unplayed content — exhausted"
            )
            if self._category:
                self._exhausted.append(self._category)
                self._category = None
            self._reset_category_state()
            self.sub_state = "enter"
            self._enter_ticks = 0
            return action_back("category exhausted, back to hub")

        self._prev_volume = centre_title or self._prev_volume
        self._carousel_steps += 1

        # Prefer arrow click; swipe fallback when previous arrow
        # didn't advance the centre volume.
        if self._carousel_stall:
            self._carousel_stall = False
            self.log(
                f"arrow click stalled on '{centre_title or '?'}' — "
                f"swiping as fallback"
            )
            return action_swipe(
                0.75, 0.50, 0.25, 0.50, duration_ms=400,
                reason="carousel swipe fallback (arrow stalled)",
            )
        return action_click(
            _ARROW_RIGHT_CLICK[0], _ARROW_RIGHT_CLICK[1],
            f"carousel arrow next (step {self._carousel_steps})",
        )

    # ── Sub-state: volume ──────────────────────────────────────────

    def _on_volume(self, screen: ScreenState) -> Dict[str, Any]:
        # Bounced back to carousel (user closed panel, or ESC)
        if self._detect_category_page(screen) == self._category \
                and not self._find_chapter_rows(screen):
            self.log(f"left volume '{self._open_volume}' without finishing — back to carousel")
            self._open_volume = ""
            self._tried_chapters = []
            self.sub_state = "category"
            return self._on_category(screen)

        rows = self._find_chapter_rows(screen)
        if not rows:
            # Volume detail not rendered yet; wait.
            self._stall_ticks += 1
            if self._stall_ticks > 10:
                self.log("volume detail not detected, backing out")
                self._stall_ticks = 0
                return action_back("volume detail not rendered")
            return action_wait(300, "waiting for chapter list")

        self._stall_ticks = 0

        # Pick the first chapter row we haven't already tried this
        # visit AND that isn't already in the progress ledger.  Rows
        # are sorted top-to-bottom by the helper.
        for row in rows:
            chapter = row.text.strip()
            if chapter in self._tried_chapters:
                continue
            composite_key = f"{self._open_volume}|{chapter}"
            if self._category and self._store.is_done(
                self._category, composite_key,
            ):
                continue
            # This is our candidate.
            self._tried_chapters.append(chapter)
            self._pending_volume = self._open_volume
            self._pending_chapter = chapter
            self._skip_stage = 0
            self._cutscene_taps = 0
            self.sub_state = "playing"
            self.log(
                f"starting {self._category}|{self._open_volume}|{chapter}"
            )
            return action_click_box(row, f"open chapter '{chapter}'")

        # All chapters either tried or already done — leave volume.
        self.log(
            f"volume '{self._open_volume}' has no more unplayed chapters "
            f"(tried={len(self._tried_chapters)})"
        )
        self._open_volume = ""
        self._tried_chapters = []
        self.sub_state = "category"
        return action_back("volume complete, back to carousel")

    # ── Sub-state: playing ─────────────────────────────────────────

    def _on_playing(self, screen: ScreenState) -> Dict[str, Any]:
        # Detect cutscene dismissal: either the volume detail re-appears
        # (chapter list visible) or we bounce all the way to the hub.
        back_in_volume = bool(self._find_chapter_rows(screen))
        back_in_hub = self._is_story_hub(screen)
        on_category = (
            self._detect_category_page(screen) == self._category
            and not back_in_volume
        )
        if back_in_volume or back_in_hub or on_category:
            if self._category and self._pending_volume and self._pending_chapter:
                composite_key = f"{self._pending_volume}|{self._pending_chapter}"
                newly = self._store.mark_done(self._category, composite_key)
                self.log(
                    f"cutscene dismissed, marked '{composite_key}' "
                    f"done in {self._category} (newly={newly})"
                )
            self._pending_volume = ""
            self._pending_chapter = ""
            self._skip_stage = 0
            self._cutscene_taps = 0
            # Return to the most-specific context visible.
            if back_in_volume:
                self.sub_state = "volume"
                return action_wait(250, "back in volume detail")
            if on_category:
                self._open_volume = ""
                self._tried_chapters = []
                self.sub_state = "category"
                return action_wait(250, "back in category carousel")
            # Hub — mark this category exhausted to avoid re-entering
            # (defensive; normally the chapter exit lands on volume).
            if self._category:
                self._exhausted.append(self._category)
                self._category = None
            self.sub_state = "hub"
            return action_wait(250, "bounced to hub after cutscene")

        # Skip-machine: same stages as event_activity's story branch.
        if self._skip_stage >= 3:
            self._cutscene_taps += 1
            if self._cutscene_taps > 25:
                self.log("stall after skip confirm, pressing back")
                self._skip_stage = 0
                self._cutscene_taps = 0
                return action_back("post-skip stall, back out")
            return action_wait(300, "waiting for cutscene to dismiss")

        if self._skip_stage == 2:
            self._cutscene_taps += 1
            confirm = screen.find_any_text(
                ["確認", "确认", "確定", "确定", "OK", "Yes", "確", "确"],
                region=(0.30, 0.55, 0.75, 0.82),
                min_conf=0.55,
            )
            if confirm:
                self._skip_stage = 3
                self._cutscene_taps = 0
                return action_click_box(confirm, "confirm story skip")
            prompt = screen.find_any_text(
                ["略過", "略过", "是否略", "跳過此"],
                region=(0.20, 0.30, 0.80, 0.70),
                min_conf=0.55,
            )
            if prompt:
                self._skip_stage = 3
                self._cutscene_taps = 0
                return action_click(0.60, 0.72, "confirm skip (hardcoded)")
            if self._cutscene_taps >= 10:
                self._cutscene_taps = 0
                self._skip_stage = 0
                return action_wait(200, "skip confirm timeout, retrying")
            if self._cutscene_taps >= 5:
                return action_click(0.60, 0.72, "confirm skip (timeout fallback)")
            return action_wait(300, "waiting for skip confirm dialog")

        if self._skip_stage == 0:
            menu = screen.find_any_text(
                ["MENU"], region=(0.82, 0.0, 1.0, 0.14), min_conf=0.65,
            )
            if menu:
                self._skip_stage = 1
                return action_click_box(menu, "click MENU to reveal skip")
            self._skip_stage = 1
            return action_click(0.94, 0.05, "click MENU (hardcoded)")

        # _skip_stage == 1: Skip button
        skip = screen.find_any_text(
            ["SKIP", "Skip", "跳過", "跳过", "スキップ"], min_conf=0.55,
        )
        if skip:
            self._skip_stage = 2
            return action_click_box(skip, "click Skip in menu")
        self._skip_stage = 2
        return action_click(0.95, 0.16, "click Skip (hardcoded)")

    # ── Sub-state: finished ────────────────────────────────────────

    def _on_finished(self, screen: ScreenState) -> Dict[str, Any]:
        return action_done("story mining finished")

    # ── Screen classifiers ─────────────────────────────────────────

    def _is_story_hub(self, screen: ScreenState) -> bool:
        """Hub: triad of 主線/短篇/支線 sub-card titles visible."""
        have_main = screen.has_text("主線劇情", min_conf=0.6)
        have_short = screen.has_text("短篇劇情", min_conf=0.6)
        have_side = screen.has_text("支線劇情", min_conf=0.6)
        return have_main and have_short and have_side

    def _detect_category_page(self, screen: ScreenState) -> Optional[str]:
        """Return 'short' / 'side' / 'main' if we're on that category
        page (hub has no breadcrumb so this is disambiguated).
        """
        if self._is_story_hub(screen):
            return None
        if screen.find_text("短篇劇情", region=_BREADCRUMB_REGION, min_conf=0.55):
            return StoryProgressStore.SHORT
        if screen.find_text("支線劇情", region=_BREADCRUMB_REGION, min_conf=0.55):
            return StoryProgressStore.SIDE
        if screen.find_text("主線劇情", region=_BREADCRUMB_REGION, min_conf=0.55):
            return StoryProgressStore.MAIN
        return None

    # ── Carousel helpers ───────────────────────────────────────────

    def _detect_centre_volume(self, screen: ScreenState) -> str:
        """Read the title of the volume currently centred on the carousel.

        Volume titles look like "X.名稱篇" or "Ex.名稱篇".  We pick the
        longest '*篇' OCR box inside the centre region — the centred
        card renders at full size so its OCR confidence and bounding
        box are largest.
        """
        best: Optional[OcrBox] = None
        best_area = 0.0
        cx_lo, cy_lo, cx_hi, cy_hi = _CAROUSEL_CENTRE_REGION
        for b in screen.ocr_boxes:
            if b.confidence < 0.6:
                continue
            t = (b.text or "").strip()
            if "篇" not in t:
                continue
            if not (cx_lo <= b.cx <= cx_hi and cy_lo <= b.cy <= cy_hi):
                continue
            # Exclude the category breadcrumb itself.
            if t in ("短篇劇情", "支線劇情", "主線劇情"):
                continue
            area = b.w * b.h
            if area > best_area:
                best_area = area
                best = b
        return best.text.strip() if best else ""

    def _has_new_badge_centre(self, screen: ScreenState) -> bool:
        """True if a 'New' OCR box is visible on the centred card."""
        cx_lo, cy_lo, cx_hi, cy_hi = _CAROUSEL_CENTRE_REGION
        for b in screen.ocr_boxes:
            if b.confidence < 0.45:
                continue
            if (b.text or "").strip().lower() != "new":
                continue
            if cx_lo <= b.cx <= cx_hi and cy_lo <= b.cy <= cy_hi:
                return True
        return False

    # ── Chapter-row helpers ────────────────────────────────────────

    def _find_chapter_rows(self, screen: ScreenState) -> List[OcrBox]:
        """Return "第N章 ..." OCR boxes in the chapter-list region,
        sorted top-to-bottom.
        """
        rows: List[OcrBox] = []
        cx_lo, cy_lo, cx_hi, cy_hi = _CHAPTER_LIST_REGION
        for b in screen.ocr_boxes:
            if b.confidence < 0.55:
                continue
            t = (b.text or "").strip()
            if not _CHAPTER_RE.match(t):
                continue
            if not (cx_lo <= b.cx <= cx_hi and cy_lo <= b.cy <= cy_hi):
                continue
            rows.append(b)
        rows.sort(key=lambda r: (round(r.cy, 3), round(r.cx, 3)))
        return rows
