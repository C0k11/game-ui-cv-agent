"""StoryMiningSkill — auto-play unplayed chapters in 短篇 / 支線 stories.

v3 design — grid-based grid model
---------------------------------------

After playtests (run_20260424_010918/011049) and a deep read of reference
(``study/ref/module/mini_story.py`` + ``group_story.py``), the correct
UI model is:

    * **Category page = 2×3 grid of volume cards** (NOT a single-card
      carousel as v2 assumed).  reference's pixel coords at 1280×720 map
      to these normalised card centres::

          col_x = [0.275, 0.727]        # 352/1280,  931/1280
          row_y = [0.333, 0.550, 0.746] # 240/720,   396/720,   537/720

      Each card has a small 'New' / '新' badge at its top-left corner::

          badge_col_x = [0.067, 0.516]     # 86/1280,   660/1280
          badge_row_y = [0.215, 0.424, 0.633] # 155/720, 305/720, 456/720
          badge_size   = (0.042, 0.035)      # 54×25 px

      A card shows 'New' when it contains unplayed chapters.

    * **Page navigation**: a right-arrow at (0.98, 0.50) advances to the
      next grid page; reference only probes the right arrow because short /
      side stories scroll one-way.  We add a left-swipe fallback for
      robustness (user-requested v2 policy).

    * **Volume detail page**: up to 2 visible chapter entry rows.  reference
      probes the 入場 button colour at (1073, 251) and (1073, 351) —
      normalised (0.838, 0.349) and (0.838, 0.488).  We mirror this via
      OCR of the 入場 / Enter text + fall back to the fixed click
      anchors when OCR misses the button.

    * **Cutscene**: same MENU → Skip → 確認 dance as v2.

The category pages are entered from the 劇情 hub (the 主線/短篇/支線/
重播 triad page).  主線劇情 has a *different* UI that reference handles
separately in ``main_story.py`` — we still back out of it here and
leave the port to a dedicated follow-up commit.
"""
from __future__ import annotations

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


# ── Hub sub-card click targets (劇情 hub) ────────────────────────────
_HUB_SHORT_CLICK = (0.61, 0.45)
_HUB_SIDE_CLICK = (0.83, 0.45)


# ── Grid-page constants (reference mini_story.py mapped to 0..1) ──────────
# Card click centres (2 cols × 3 rows).
_GRID_CARD_COL_X = (0.275, 0.727)
_GRID_CARD_ROW_Y = (0.333, 0.550, 0.746)

# 'New' badge regions, top-left of each card.
_GRID_BADGE_COL_X = (0.067, 0.516)
_GRID_BADGE_ROW_Y = (0.215, 0.424, 0.633)
_GRID_BADGE_W = 0.12   # bit wider than reference's strict 0.042 to catch
_GRID_BADGE_H = 0.06   # OCR boxes whose padding varies by font size

# Page-next arrow at right edge; click advances to next grid page.
_GRID_PAGE_NEXT_CLICK = (0.98, 0.497)
# Swipe fallback when the arrow click doesn't advance.
_GRID_SWIPE_FROM = (0.80, 0.50)
_GRID_SWIPE_TO = (0.20, 0.50)

# Max grid pages we scan before declaring a category exhausted.
_GRID_PAGE_LIMIT = 4  # short/side categories ship 2-3 pages; 4 is safe.


# ── Volume-detail constants ──────────────────────────────────────────
# reference probes (1073, 251) / (1073, 351) for 入場 button colour.
# We click slightly earlier on the row (x=0.78) so we hit the chapter
# body rather than the button sprite alone — same visual effect.
_VOLUME_CHAPTER_CLICKS = (
    (0.78, 0.349),  # top chapter row
    (0.78, 0.488),  # bottom chapter row
)
# 入場 / Enter OCR search region (both rows combined).
_VOLUME_ENTER_REGION = (0.70, 0.30, 0.99, 0.55)


# ── Hub / breadcrumb detection ───────────────────────────────────────
_BREADCRUMB_REGION = (0.0, 0.0, 0.25, 0.09)


# ──────────────────────────────────────────────────────────────────────

class StoryMiningSkill(BaseSkill):
    """Auto-play unplayed chapters in 短篇劇情 and 支線劇情."""

    def __init__(self) -> None:
        super().__init__("StoryMining")
        self.max_ticks = 600

        self._store: StoryProgressStore = get_store()

        # Sub-states:
        #   enter   — figure out where we are
        #   hub     — on 劇情 hub; click next category card
        #   grid    — on a 2×3 grid category page; find a 'New' cell
        #   volume  — inside a volume; click an active chapter row
        #   playing — cutscene skip dance
        #   finished — done; advance pipeline
        self.sub_state = "enter"
        self._enter_ticks: int = 0

        self._category: Optional[str] = None
        self._exhausted: List[str] = []

        # Grid state.
        self._grid_page: int = 0              # 0-indexed page number
        self._prev_grid_fingerprint: str = ""  # stall detection
        self._arrow_stalled: bool = False

        # Volume state.
        self._open_volume_hint: str = ""      # card index (0-5) + page for the ledger
        self._volume_chapter_attempts: int = 0

        # Cutscene state.
        self._pending_volume: str = ""
        self._pending_chapter: str = ""
        self._skip_stage: int = 0
        self._cutscene_taps: int = 0

        self._stall_ticks: int = 0

    def reset(self) -> None:
        super().reset()
        self.sub_state = "enter"
        self._enter_ticks = 0
        self._category = None
        self._exhausted = []
        self._grid_page = 0
        self._prev_grid_fingerprint = ""
        self._arrow_stalled = False
        self._open_volume_hint = ""
        self._volume_chapter_attempts = 0
        self._pending_volume = ""
        self._pending_chapter = ""
        self._skip_stage = 0
        self._cutscene_taps = 0
        self._stall_ticks = 0

    # ── Tick dispatch ───────────────────────────────────────────────

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1
        self._enter_ticks += 1

        popup_action = self._dismiss_popup(screen)
        if popup_action is not None:
            return popup_action

        handler = {
            "enter": self._on_enter,
            "hub": self._on_hub,
            "grid": self._on_grid,
            "volume": self._on_volume,
            "playing": self._on_playing,
            "finished": self._on_finished,
        }.get(self.sub_state, self._on_enter)
        return handler(screen)

    # ── Popup handling ─────────────────────────────────────────────

    def _dismiss_popup(self, screen: ScreenState) -> Optional[Dict[str, Any]]:
        if self.sub_state == "playing":
            return None
        blocked = screen.find_any_text(
            [
                "尚未開放", "尚未开放", "尚未達成", "尚未达成",
                "無法進入", "无法进入", "請先完成", "请先完成",
                "未解鎖", "未解锁",
            ],
            region=screen.CENTER, min_conf=0.55,
        )
        if not blocked:
            return None
        confirm = screen.find_any_text(
            ["確認", "确认", "確定", "确定", "OK", "Yes", "確", "确"],
            region=(0.30, 0.55, 0.75, 0.82), min_conf=0.55,
        )
        self.log(f"locked popup: '{blocked.text}', dismissing")
        if confirm:
            return action_click_box(confirm, "dismiss locked popup")
        return action_click(0.50, 0.72, "dismiss locked popup (fallback)")

    # ── enter ──────────────────────────────────────────────────────

    def _on_enter(self, screen: ScreenState) -> Dict[str, Any]:
        if self._is_story_hub(screen):
            self.sub_state = "hub"
            return self._on_hub(screen)

        cat = self._detect_category_page(screen)
        if cat == StoryProgressStore.MAIN:
            if StoryProgressStore.MAIN not in self._exhausted:
                self._exhausted.append(StoryProgressStore.MAIN)
            self.log("主線劇情 detected; pusher not implemented — backing out")
            return action_back("main story not supported yet")
        if cat in (StoryProgressStore.SHORT, StoryProgressStore.SIDE):
            self.log(f"already on {cat} category grid")
            self._category = cat
            self._reset_category_state()
            self.sub_state = "grid"
            return self._on_grid(screen)

        # Not on hub / category page → navigate there from wherever we are.
        # Mirrors StoryCleanup._enter's traversal: Lobby -> 任務 button ->
        # Mission page -> 劇情 entry -> story hub.
        current = self.detect_current_screen(screen)

        if current == "Lobby":
            campaign_btn = screen.find_any_text(
                ["任務", "任务"],
                region=(0.80, 0.70, 1.0, 0.90),
                min_conf=0.6,
            )
            if campaign_btn:
                return action_click_box(
                    campaign_btn, "open 任務 for story mining",
                )
            return action_click(0.95, 0.83, "open 任務 (hardcoded)")

        if current == "Mission":
            story_entry = screen.find_any_text(
                ["劇情", "剧情", "Story"],
                min_conf=0.5,
            )
            if story_entry:
                return action_click_box(
                    story_entry, "enter 劇情 hub",
                )
            if self._enter_ticks > 6:
                return action_click(0.26, 0.71, "enter 劇情 hub (hardcoded)")
            return action_wait(350, "looking for 劇情 entry")

        if current and current not in ("Lobby", "Mission"):
            return action_back(f"back from {current} toward story hub")

        # Unknown screen — wait briefly and retry, cap at 28 ticks so
        # we don't block the pipeline forever.
        if self._enter_ticks > 28:
            self.log("story hub unreachable, skipping")
            return action_done("story hub not reached")
        return action_wait(450, "searching for story hub")

    # ── hub ────────────────────────────────────────────────────────

    def _on_hub(self, screen: ScreenState) -> Dict[str, Any]:
        for cat, target in (
            (StoryProgressStore.SHORT, _HUB_SHORT_CLICK),
            (StoryProgressStore.SIDE, _HUB_SIDE_CLICK),
        ):
            if cat in self._exhausted:
                continue
            self._category = cat
            self._reset_category_state()
            self.sub_state = "grid"
            self.log(f"entering {cat} category")
            return action_click(target[0], target[1], f"open {cat} category")

        if StoryProgressStore.MAIN not in self._exhausted:
            self._exhausted.append(StoryProgressStore.MAIN)
            self.log("主線劇情 push not implemented — skipped")

        self.sub_state = "finished"
        return action_done("story mining finished")

    def _reset_category_state(self) -> None:
        self._grid_page = 0
        self._prev_grid_fingerprint = ""
        self._arrow_stalled = False
        self._open_volume_hint = ""
        self._volume_chapter_attempts = 0
        self._stall_ticks = 0

    # ── grid (2×3 volume carousel) ─────────────────────────────────

    def _on_grid(self, screen: ScreenState) -> Dict[str, Any]:
        # Bounced back to hub? Finish this category.
        if self._is_story_hub(screen):
            self.log(f"category '{self._category}' auto-closed, back to hub")
            if self._category:
                self._exhausted.append(self._category)
                self._category = None
            self.sub_state = "hub"
            return self._on_hub(screen)

        # Stall detection: grid fingerprint (titles visible) unchanged
        # across two ticks means the arrow click didn't register.
        fingerprint = self._grid_fingerprint(screen)
        if fingerprint and fingerprint == self._prev_grid_fingerprint:
            self._arrow_stalled = True
        else:
            self._arrow_stalled = False
        if fingerprint:
            self._prev_grid_fingerprint = fingerprint

        # Probe 6 cells for a 'New' badge.  Return first hit
        # (top-to-bottom, left-to-right) so selection is deterministic.
        cell_idx = self._find_new_cell(screen)
        if cell_idx is not None:
            col = cell_idx % 2
            row = cell_idx // 2
            cx = _GRID_CARD_COL_X[col]
            cy = _GRID_CARD_ROW_Y[row]
            self._open_volume_hint = f"{self._category}:p{self._grid_page}:c{cell_idx}"
            self._volume_chapter_attempts = 0
            self.sub_state = "volume"
            self.log(
                f"cell {cell_idx} (row={row}, col={col}) has 'New' — entering"
            )
            return action_click(cx, cy, f"open volume cell {cell_idx}")

        # No 'New' on this page → try next page.
        if self._grid_page >= _GRID_PAGE_LIMIT - 1:
            self.log(
                f"scanned {self._grid_page + 1} grid pages in '{self._category}' "
                f"with no 'New' — category exhausted"
            )
            if self._category:
                self._exhausted.append(self._category)
                self._category = None
            self._reset_category_state()
            self.sub_state = "enter"
            self._enter_ticks = 0
            return action_back("category exhausted, back to hub")

        self._grid_page += 1
        if self._arrow_stalled:
            self._arrow_stalled = False
            self.log(
                f"grid arrow stalled on page {self._grid_page - 1} — "
                f"swiping as fallback"
            )
            fx, fy = _GRID_SWIPE_FROM
            tx, ty = _GRID_SWIPE_TO
            return action_swipe(
                fx, fy, tx, ty, duration_ms=400,
                reason=f"swipe to grid page {self._grid_page}",
            )
        return action_click(
            _GRID_PAGE_NEXT_CLICK[0], _GRID_PAGE_NEXT_CLICK[1],
            f"arrow → grid page {self._grid_page}",
        )

    def _grid_fingerprint(self, screen: ScreenState) -> str:
        """Return a cheap hashable summary of the grid page contents.

        We use the concatenated OCR text of every '篇' box on the page
        — this changes the moment the grid page advances.
        """
        parts: List[str] = []
        for b in screen.ocr_boxes:
            if b.confidence < 0.6:
                continue
            t = (b.text or "").strip()
            if "篇" in t and t not in ("短篇劇情", "支線劇情", "主線劇情"):
                parts.append(t)
        return "|".join(sorted(parts))

    def _find_new_cell(self, screen: ScreenState) -> Optional[int]:
        """Return the grid cell index (0..5) with a 'New' / '新' badge.

        Cell index layout::

            0  1
            2  3
            4  5
        """
        for b in screen.ocr_boxes:
            if b.confidence < 0.5:
                continue
            t = (b.text or "").strip()
            if t.lower() != "new" and t != "新":
                continue
            # Which cell does this badge belong to?  Pick the cell
            # whose badge region contains the box centre.
            for row_idx, ry in enumerate(_GRID_BADGE_ROW_Y):
                for col_idx, rx in enumerate(_GRID_BADGE_COL_X):
                    if rx <= b.cx <= rx + _GRID_BADGE_W \
                            and ry <= b.cy <= ry + _GRID_BADGE_H:
                        return row_idx * 2 + col_idx
        return None

    # ── volume detail ──────────────────────────────────────────────

    def _on_volume(self, screen: ScreenState) -> Dict[str, Any]:
        # Bounced back to grid (user closed via ESC or game auto-back).
        cat_here = self._detect_category_page(screen)
        if cat_here == self._category and not self._volume_has_chapters(screen):
            self.log(f"left volume {self._open_volume_hint} without entering")
            self.sub_state = "grid"
            return self._on_grid(screen)

        # Find an 入場 / Enter button or click a fixed chapter row anchor.
        enter_box = screen.find_any_text(
            ["入場", "入场", "Enter", "開始", "开始", "進入", "进入"],
            region=_VOLUME_ENTER_REGION, min_conf=0.55,
        )

        if enter_box is not None:
            chapter_id = self._chapter_key_from_y(enter_box.cy)
            composite = f"{self._open_volume_hint}|{chapter_id}"
            if self._category and self._store.is_done(self._category, composite):
                # Already done per ledger; skip to the next row anchor.
                return self._try_next_chapter_anchor(composite_skip=composite)
            return self._start_chapter(composite, click_target=enter_box)

        # No Enter button visible — try the fixed anchors reference uses.
        return self._try_next_chapter_anchor()

    def _chapter_key_from_y(self, y: float) -> str:
        """Bin a vertical coord to 'r0' / 'r1' — two chapter rows."""
        mid = (_VOLUME_CHAPTER_CLICKS[0][1] + _VOLUME_CHAPTER_CLICKS[1][1]) / 2
        return "r0" if y < mid else "r1"

    def _try_next_chapter_anchor(
        self, *, composite_skip: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Click the next un-attempted chapter anchor (top row first).

        When we exhaust both anchors we back out to the grid.  Each
        attempt is tracked so a second tick on the same screen without
        state change doesn't loop forever on the same row.
        """
        if self._volume_chapter_attempts >= len(_VOLUME_CHAPTER_CLICKS):
            self.log(
                f"no more chapter anchors in volume {self._open_volume_hint}"
            )
            self._volume_chapter_attempts = 0
            self.sub_state = "grid"
            return action_back("volume exhausted, back to grid")

        idx = self._volume_chapter_attempts
        self._volume_chapter_attempts += 1
        cx, cy = _VOLUME_CHAPTER_CLICKS[idx]
        chapter_id = f"r{idx}"
        composite = f"{self._open_volume_hint}|{chapter_id}"

        # Skip if ledger already has this chapter.
        if self._category and self._store.is_done(self._category, composite):
            self.log(f"{composite} already done per ledger, skipping")
            return self._try_next_chapter_anchor()

        return self._start_chapter(
            composite,
            click_target=None,
            click_xy=(cx, cy),
            chapter_id=chapter_id,
        )

    def _start_chapter(
        self,
        composite: str,
        *,
        click_target: Optional[OcrBox] = None,
        click_xy: Optional[Tuple[float, float]] = None,
        chapter_id: str = "",
    ) -> Dict[str, Any]:
        self._pending_volume = self._open_volume_hint
        self._pending_chapter = chapter_id or composite.rsplit("|", 1)[-1]
        self._skip_stage = 0
        self._cutscene_taps = 0
        self.sub_state = "playing"
        self.log(f"starting {self._category}|{composite}")
        if click_target is not None:
            return action_click_box(click_target, f"open chapter {composite}")
        assert click_xy is not None
        return action_click(
            click_xy[0], click_xy[1], f"open chapter anchor {composite}",
        )

    def _volume_has_chapters(self, screen: ScreenState) -> bool:
        """Heuristic: the volume detail page always shows '入場' or
        a '第N章' row.  If neither is visible we're on the grid.
        """
        if screen.find_any_text(
            ["入場", "入场", "Enter"],
            region=_VOLUME_ENTER_REGION, min_conf=0.55,
        ):
            return True
        for b in screen.ocr_boxes:
            if b.confidence < 0.55:
                continue
            t = (b.text or "").strip()
            if t.startswith("第") and "章" in t and b.cx > 0.55:
                return True
        return False

    # ── playing (cutscene skip) ────────────────────────────────────

    def _on_playing(self, screen: ScreenState) -> Dict[str, Any]:
        # Detect dismissal.
        back_in_volume = self._volume_has_chapters(screen)
        back_on_grid = (
            self._detect_category_page(screen) == self._category
            and not back_in_volume
        )
        back_in_hub = self._is_story_hub(screen)
        if back_in_volume or back_on_grid or back_in_hub:
            if self._category and self._pending_volume and self._pending_chapter:
                key = f"{self._pending_volume}|{self._pending_chapter}"
                newly = self._store.mark_done(self._category, key)
                self.log(
                    f"cutscene dismissed, marked '{key}' done "
                    f"(newly={newly})"
                )
            self._pending_volume = ""
            self._pending_chapter = ""
            self._skip_stage = 0
            self._cutscene_taps = 0
            if back_in_volume:
                self.sub_state = "volume"
                return action_wait(250, "back in volume detail")
            if back_on_grid:
                self._open_volume_hint = ""
                self._volume_chapter_attempts = 0
                self.sub_state = "grid"
                return action_wait(250, "back on grid")
            # hub (rare defensive path)
            if self._category:
                self._exhausted.append(self._category)
                self._category = None
            self.sub_state = "hub"
            return action_wait(250, "bounced to hub after cutscene")

        if self._skip_stage >= 3:
            self._cutscene_taps += 1
            if self._cutscene_taps > 25:
                self.log("post-skip stall, pressing back")
                self._skip_stage = 0
                self._cutscene_taps = 0
                return action_back("post-skip stall")
            return action_wait(300, "waiting for cutscene to dismiss")

        if self._skip_stage == 2:
            self._cutscene_taps += 1
            confirm = screen.find_any_text(
                ["確認", "确认", "確定", "确定", "OK", "Yes", "確", "确"],
                region=(0.30, 0.55, 0.75, 0.82), min_conf=0.55,
            )
            if confirm:
                self._skip_stage = 3
                self._cutscene_taps = 0
                return action_click_box(confirm, "confirm story skip")
            prompt = screen.find_any_text(
                ["略過", "略过", "是否略", "跳過此"],
                region=(0.20, 0.30, 0.80, 0.70), min_conf=0.55,
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
                return action_click(
                    0.60, 0.72, "confirm skip (timeout fallback)",
                )
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

        # _skip_stage == 1
        skip = screen.find_any_text(
            ["SKIP", "Skip", "跳過", "跳过", "スキップ"], min_conf=0.55,
        )
        if skip:
            self._skip_stage = 2
            return action_click_box(skip, "click Skip in menu")
        self._skip_stage = 2
        return action_click(0.95, 0.16, "click Skip (hardcoded)")

    # ── finished ──────────────────────────────────────────────────

    def _on_finished(self, screen: ScreenState) -> Dict[str, Any]:
        return action_done("story mining finished")

    # ── classifiers ──────────────────────────────────────────────

    def _is_story_hub(self, screen: ScreenState) -> bool:
        return (
            screen.has_text("主線劇情", min_conf=0.6)
            and screen.has_text("短篇劇情", min_conf=0.6)
            and screen.has_text("支線劇情", min_conf=0.6)
        )

    def _detect_category_page(self, screen: ScreenState) -> Optional[str]:
        if self._is_story_hub(screen):
            return None
        if screen.find_text("短篇劇情", region=_BREADCRUMB_REGION, min_conf=0.55):
            return StoryProgressStore.SHORT
        if screen.find_text("支線劇情", region=_BREADCRUMB_REGION, min_conf=0.55):
            return StoryProgressStore.SIDE
        if screen.find_text("主線劇情", region=_BREADCRUMB_REGION, min_conf=0.55):
            return StoryProgressStore.MAIN
        return None
