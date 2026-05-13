"""StoryMiningSkill v5 — full short / side story mining.

Lessons baked in (from re-reading study/ref/module/{mini_story,
group_story,main_story,picture}.py + actual capture data under
data/captures/{mini_story,group_story,plot,main_story} and the
April playtest run_20260424_014538):

UI hierarchy (5 layers, not 3):

    Story Hub        : 主線/短篇/支線/重播 cards
    Category grid    : 短篇劇情 or 支線劇情, 2x3 cards per page,
                       'New' yellow badge at top-left of each card
    Chapter list     : 章節目錄, 2 visible chapter rows on the right
                       panel, 'New' badge on left of row, 入場 button
                       on right of row
    Chapter info     : 章節資訊 modal (centre overlay), shows reward
                       preview + 進入章節 button at bottom-centre
    Cutscene         : MENU (top right) -> Skip icon -> 概要 confirm

Key fixes vs v4:

  * v4 collapsed chapter list + chapter info as "volume detail" and
    happily clicked 入場 (right of list row) on the chapter info
    modal where the live button is 進入章節 (centre-bottom of modal).
    Result: dead clicks, never started a chapter on chapter info.
  * v4 fallback chapter anchors were at x=0.78 (mid-row body); reference
    + capture data put the 入場 button at x=0.838-0.87. Fixed.
  * v4 had no signal for 已完成所有章節 (volume cleared) — we'd
    just retry until the volume timeout, then back out.  Now we
    detect that text and back out cleanly.
  * Transitional/loading frames (only breadcrumb visible, no card
    OCR) tricked v4 into thinking "no New, advance page".  Now we
    require a populated grid before deciding to advance.
  * Reward popup after each cutscene was unhandled.  Now handled.
  * Cutscene confirm-dialog signal is 概要 (reference plot_skip-plot-
    notice header), not "是否略過".

Architecture remains template-based flat reactive rule list (single
tick() function + per-screen handlers).  No nested sub_state
control flow.
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


# ── Story hub (主線/短篇/支線/重播 cards) ────────────────────────────

HUB_SHORT_CLICK = (0.61, 0.45)
HUB_SIDE_CLICK = (0.83, 0.45)


# ── Category grid (短篇劇情 / 支線劇情) ─────────────────────────────

# Card click centres (book layout, 2 cols x 3 rows).  reference pixel
# coords (352,240) (931,240) (352,396) (931,396) (352,537) (931,537)
# at 1280x720 -> normalised below.  Capture data confirms these are
# inside-the-card targets even on the user's wider aspect ratio.
GRID_CARD_COL_X: Tuple[float, float] = (0.275, 0.727)
GRID_CARD_ROW_Y: Tuple[float, float, float] = (0.333, 0.550, 0.746)

# 'New' yellow badge top-left anchors per card.  reference check_6_region_
# status uses (86,155) (660,155) ... 54x25.  We bin OCR boxes whose
# centre falls inside an enlarged badge region (OCR boxes for 'New'
# are bigger than the underlying sprite).
GRID_BADGE_COL_X: Tuple[float, float] = (0.067, 0.516)
GRID_BADGE_ROW_Y: Tuple[float, float, float] = (0.215, 0.424, 0.633)
GRID_BADGE_W = 0.16
GRID_BADGE_H = 0.07

# Page-next arrow.  reference clicks (1255, 357).  We can't probe its
# active-colour RGB; instead we detect grid-fingerprint stalls + cap
# at 5 pages per category (reference short / side ship 2-3 pages).
GRID_PAGE_NEXT_CLICK = (0.98, 0.497)
GRID_SWIPE_FROM = (0.80, 0.50)
GRID_SWIPE_TO = (0.20, 0.50)
GRID_PAGE_LIMIT = 5


# ── Chapter list (章節目錄) ──────────────────────────────────────────

# 入場 button click anchors.  reference one_detect probes (1073, 251) and
# (1073, 351) -> normalised (0.838, 0.349) / (0.838, 0.488).  The
# actual 入場 button on the captured screenshot has its OCR centre
# at (0.87, 0.485) and (0.87, 0.32) when both rows are populated;
# both fall inside a click radius of these anchors.
CHAPTER_LIST_ANCHORS: Tuple[Tuple[float, float], Tuple[float, float]] = (
    (0.84, 0.349),  # top chapter row
    (0.84, 0.488),  # bottom chapter row
)

# OCR search region for 入場 (right half of screen, chapter rows).
CHAPTER_LIST_ENTER_REGION = (0.65, 0.20, 0.99, 0.62)

# Volume-cleared signal — reference image: mini_story_episode-cleared-
# feature.png == "已完成所有章節".
VOLUME_CLEARED_REGION = (0.55, 0.30, 0.99, 0.70)


# ── Chapter info (章節資訊 modal) ───────────────────────────────────

# 進入章節 button at bottom-centre of the modal.  reference clicks
# (650, 511) -> (0.508, 0.710).
CHAPTER_INFO_PLAY_REGION = (0.30, 0.55, 0.75, 0.85)
CHAPTER_INFO_PLAY_FALLBACK = (0.508, 0.710)


# ── Cutscene skip ───────────────────────────────────────────────────

# reference plot_menu (1202, 37) / plot_skip-plot-button (1208, 116) /
# plot_skip-plot-notice (770, 519).  The skip-button itself is an
# icon (chevron >>), not text — OCR may or may not pick it up.
CUTSCENE_MENU_REGION = (0.85, 0.0, 1.0, 0.12)
CUTSCENE_MENU_FALLBACK = (0.94, 0.05)
CUTSCENE_SKIP_REGION = (0.85, 0.10, 1.0, 0.25)
CUTSCENE_SKIP_FALLBACK = (0.943, 0.161)
# 概要 is the title of the skip-confirm modal.
CUTSCENE_NOTICE_REGION = (0.30, 0.10, 0.70, 0.40)
CUTSCENE_NOTICE_OK_FALLBACK = (0.602, 0.721)


# ── Categories ──────────────────────────────────────────────────────

CATEGORY_SHORT = "short"
CATEGORY_SIDE = "side"
CATEGORY_MAIN = "main"  # not implemented; we mark it exhausted up-front
CATEGORIES_TO_TRY: Tuple[str, ...] = (CATEGORY_SHORT, CATEGORY_SIDE)


# ── Helpers ─────────────────────────────────────────────────────────

def _grid_cells():
    cells = []
    for row in range(3):
        for col in range(2):
            idx = row * 2 + col
            bx = GRID_BADGE_COL_X[col]
            by = GRID_BADGE_ROW_Y[row]
            cells.append((idx, row, col, bx, by,
                          bx + GRID_BADGE_W, by + GRID_BADGE_H))
    return cells


# ──────────────────────────────────────────────────────────────────────

class StoryMiningSkill(BaseSkill):
    """Auto-play unplayed chapters in 短篇劇情 / 支線劇情 — reference-
    style reactive rule chain.
    """

    def __init__(self) -> None:
        super().__init__("StoryMining")
        self.max_ticks = 1200

        # Per-category exhaustion flags.
        self._exhausted: List[str] = []
        self._category: Optional[str] = None

        # Grid pagination state for the active category.
        self._grid_pages_seen: int = 0
        self._grid_prev_fingerprint: str = ""
        self._grid_arrow_attempts: int = 0

        # Per-grid-page settle: don't scan immediately after a page
        # transition (avoids OCR-on-loading false negatives).
        self._grid_settle_ticks: int = 0
        self._grid_settle_required: int = 1

        # Volume / chapter-list traversal state.
        self._chapter_anchor_idx: int = 0
        self._chapter_idle_ticks: int = 0
        self._chapter_max_idle: int = 8

        # Cutscene + post-cutscene state.
        self._cutscene_taps: int = 0

        # Lobby / Mission navigation cap.
        self._hub_search_ticks: int = 0
        self._hub_search_limit: int = 40

    # ── Lifecycle ───────────────────────────────────────────────────

    def reset(self) -> None:
        super().reset()
        self._exhausted = []
        self._category = None
        self._grid_pages_seen = 0
        self._grid_prev_fingerprint = ""
        self._grid_arrow_attempts = 0
        self._grid_settle_ticks = 0
        self._chapter_anchor_idx = 0
        self._chapter_idle_ticks = 0
        self._cutscene_taps = 0
        self._hub_search_ticks = 0

    # ── Tick: ordered rule chain ────────────────────────────────────

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1

        # 1. Always-on transient popups + cutscene skip.
        for handler in (
            self._handle_locked_popup,
            self._handle_reward_popup,
            self._handle_cutscene_confirm,
            self._handle_cutscene_skip,
            self._handle_cutscene_menu,
        ):
            a = handler(screen)
            if a is not None:
                return a
        self._cutscene_taps = 0

        # 2. Done?
        if self._all_categories_exhausted():
            self.sub_state = "finished"
            return action_done("story mining finished")

        # 3. Screen-kind dispatch (most specific first).
        kind = self._classify(screen)
        self.sub_state = kind

        dispatch = {
            "chapter_info":     self._on_chapter_info,
            "chapter_list":     self._on_chapter_list,
            "volume_cleared":   self._on_volume_cleared,
            "category_short":   lambda s: self._on_category(s, CATEGORY_SHORT),
            "category_side":    lambda s: self._on_category(s, CATEGORY_SIDE),
            "category_main":    self._on_category_main,
            "story_hub":        self._on_hub,
            "mission":          self._on_mission,
            "lobby":            self._on_lobby,
        }
        if kind in dispatch:
            return dispatch[kind](screen)

        # Unknown screen — wait, give up if it persists.
        self._hub_search_ticks += 1
        if self._hub_search_ticks > self._hub_search_limit:
            self.log("unknown screen for too long, giving up")
            return action_done("story mining unreachable")
        return action_wait(400, "unknown screen, waiting")

    # ── Screen classifier ──────────────────────────────────────────

    def _classify(self, screen: ScreenState) -> str:
        # Most specific first: the chapter info modal.
        if self._is_chapter_info(screen):
            return "chapter_info"
        if self._is_volume_cleared(screen):
            return "volume_cleared"
        if self._is_chapter_list(screen):
            return "chapter_list"

        breadcrumb = (0.0, 0.0, 0.30, 0.10)
        on_hub = self._is_story_hub(screen)
        if not on_hub:
            if screen.find_any_text(
                ["短篇劇情", "短篇剧情"],
                region=breadcrumb, min_conf=0.55,
            ):
                return "category_short"
            if screen.find_any_text(
                ["支線劇情", "支线剧情"],
                region=breadcrumb, min_conf=0.55,
            ):
                return "category_side"
            if screen.find_any_text(
                ["主線劇情", "主线剧情"],
                region=breadcrumb, min_conf=0.55,
            ):
                return "category_main"
        if on_hub:
            return "story_hub"

        current = self.detect_current_screen(screen)
        if current == "Lobby":
            return "lobby"
        if current == "Mission":
            return "mission"
        return "unknown"

    def _is_story_hub(self, screen: ScreenState) -> bool:
        return (
            screen.has_text("主線劇情", min_conf=0.55)
            and screen.has_text("短篇劇情", min_conf=0.55)
            and screen.has_text("支線劇情", min_conf=0.55)
        )

    def _is_chapter_info(self, screen: ScreenState) -> bool:
        # Modal title 章節資訊 in the centre-top region.
        return bool(screen.find_any_text(
            ["章節資訊", "章节资讯", "章节信息"],
            region=(0.30, 0.10, 0.70, 0.30), min_conf=0.55,
        ))

    def _is_chapter_list(self, screen: ScreenState) -> bool:
        # Right-panel header 章節目錄.  Or fallback: 入場 button visible
        # in the right-half region while NOT on chapter_info.
        if screen.find_any_text(
            ["章節目錄", "章节目录"],
            region=(0.55, 0.10, 0.95, 0.30), min_conf=0.55,
        ):
            return True
        if screen.find_any_text(
            ["入場", "入场", "Enter"],
            region=CHAPTER_LIST_ENTER_REGION, min_conf=0.55,
        ):
            return True
        return False

    def _is_volume_cleared(self, screen: ScreenState) -> bool:
        return bool(screen.find_any_text(
            ["已完成所有章節", "已完成所有章节"],
            region=VOLUME_CLEARED_REGION, min_conf=0.55,
        ))

    # ── Always-on handlers ─────────────────────────────────────────

    def _handle_locked_popup(
        self, screen: ScreenState,
    ) -> Optional[Dict[str, Any]]:
        blocked = screen.find_any_text(
            [
                "尚未開放", "尚未开放", "尚未達成", "尚未达成",
                "無法進入", "无法进入", "請先完成", "请先完成",
                "未解鎖", "未解锁", "未開放", "未开放",
            ],
            region=screen.CENTER, min_conf=0.55,
        )
        if not blocked:
            return None
        confirm = screen.find_any_text(
            ["確認", "确认", "確定", "确定", "OK", "Yes"],
            region=(0.30, 0.55, 0.80, 0.82), min_conf=0.55,
        )
        self.log(f"locked popup: '{blocked.text}', dismissing")
        if confirm is not None:
            return action_click_box(confirm, "dismiss locked popup")
        return action_click(0.50, 0.72, "dismiss locked popup (fallback)")

    def _handle_reward_popup(
        self, screen: ScreenState,
    ) -> Optional[Dict[str, Any]]:
        # Post-cutscene reward splash.  reference uses RGB at (640, 100);
        # we OCR for the banner text + tap-through.
        reward = screen.find_any_text(
            ["獲得獎勵", "获得奖励", "領取獎勵", "领取奖励",
             "恭喜獲得", "恭喜获得", "獲得期待獎勵"],
            region=(0.20, 0.05, 0.80, 0.45), min_conf=0.55,
        )
        if reward is None:
            return None
        # Heuristic: only tap-through reward popups when we're NOT on
        # the chapter info modal (which also contains 獲得期待獎勵 text
        # but is interactive).  The chapter info modal has 進入章節
        # button visible in its play region.
        if screen.find_any_text(
            ["進入章節", "进入章节"],
            region=CHAPTER_INFO_PLAY_REGION, min_conf=0.55,
        ):
            return None
        self.log(f"reward popup: '{reward.text}', tapping through")
        return action_click(0.50, 0.90, "dismiss reward popup")

    def _handle_cutscene_menu(
        self, screen: ScreenState,
    ) -> Optional[Dict[str, Any]]:
        # MENU button in the top-right corner during a cutscene.
        # Only fires if we're inside a cutscene (no breadcrumb at
        # top-left, no chapter list / chapter info / hub).
        if self._classify_quick_breadcrumb(screen):
            return None
        menu = screen.find_text_one(
            "MENU", region=CUTSCENE_MENU_REGION, min_conf=0.65,
        )
        if menu is not None:
            self._cutscene_taps += 1
            return action_click_box(menu, "click MENU to reveal Skip")
        # AUTO toggle present implies cutscene is rolling.
        auto = screen.find_text_one(
            "AUTO", region=CUTSCENE_MENU_REGION, min_conf=0.6,
        )
        if auto is not None:
            self._cutscene_taps += 1
            return action_click(
                CUTSCENE_MENU_FALLBACK[0], CUTSCENE_MENU_FALLBACK[1],
                "click MENU (AUTO seen, hardcoded)",
            )
        return None

    def _handle_cutscene_skip(
        self, screen: ScreenState,
    ) -> Optional[Dict[str, Any]]:
        # The Skip button is a chevron icon — OCR may catch SKIP /
        # 跳過 / 略過 text but not always.
        if self._classify_quick_breadcrumb(screen):
            return None
        skip = screen.find_any_text(
            ["跳過劇情", "跳过剧情", "Skip Story",
             "略過劇情", "SKIP", "Skip", "跳過", "跳过"],
            region=CUTSCENE_SKIP_REGION, min_conf=0.55,
        )
        if skip is not None:
            self._cutscene_taps += 1
            return action_click_box(skip, "click Skip")
        # If we just clicked MENU and don't see Skip yet, click the
        # known Skip coord.  Tracked via _cutscene_taps so we don't
        # spam this every tick.
        if self._cutscene_taps == 1:
            self._cutscene_taps += 1
            return action_click(
                CUTSCENE_SKIP_FALLBACK[0], CUTSCENE_SKIP_FALLBACK[1],
                "click Skip (hardcoded after MENU)",
            )
        return None

    def _handle_cutscene_confirm(
        self, screen: ScreenState,
    ) -> Optional[Dict[str, Any]]:
        # 概要 dialog (skip-confirm).  Active anywhere — even if we
        # missed the previous MENU/Skip clicks, this dialog appearing
        # means we should confirm and move on.
        notice = screen.find_any_text(
            ["概要", "是否略過", "是否略过", "略過此", "略过此"],
            region=CUTSCENE_NOTICE_REGION, min_conf=0.55,
        )
        if notice is None:
            return None
        confirm = screen.find_any_text(
            ["確認", "确认", "確定", "确定", "OK", "Yes"],
            region=(0.30, 0.60, 0.80, 0.85), min_conf=0.55,
        )
        if confirm is not None:
            self._cutscene_taps += 1
            return action_click_box(confirm, "confirm story skip")
        self._cutscene_taps += 1
        return action_click(
            CUTSCENE_NOTICE_OK_FALLBACK[0], CUTSCENE_NOTICE_OK_FALLBACK[1],
            "confirm story skip (fallback)",
        )

    def _classify_quick_breadcrumb(self, screen: ScreenState) -> bool:
        """Cheap test: are we on a normal navigable screen (story hub
        / category page / chapter list / chapter info)?  If yes, the
        cutscene-skip rules should NOT fire.
        """
        if self._is_chapter_info(screen) or self._is_chapter_list(screen):
            return True
        breadcrumb = (0.0, 0.0, 0.30, 0.10)
        if screen.find_any_text(
            ["短篇劇情", "支線劇情", "主線劇情",
             "短篇剧情", "支线剧情", "主线剧情",
             "劇情", "剧情"],
            region=breadcrumb, min_conf=0.55,
        ):
            return True
        return False

    # ── Lobby / Mission / Hub ─────────────────────────────────────

    def _on_lobby(self, screen: ScreenState) -> Dict[str, Any]:
        self._hub_search_ticks += 1
        campaign = screen.find_any_text(
            ["任務", "任务"], region=(0.80, 0.70, 1.0, 0.92),
            min_conf=0.6,
        )
        if campaign is not None:
            return action_click_box(campaign, "open 任務 (campaign)")
        return action_click(0.95, 0.83, "open 任務 (hardcoded)")

    def _on_mission(self, screen: ScreenState) -> Dict[str, Any]:
        self._hub_search_ticks += 1
        # Look for 劇情 specifically in the campaign-tile area, not
        # anywhere on screen — the breadcrumb at top-left is "劇情"
        # too and clicking it doesn't navigate anywhere.
        story_entry = screen.find_any_text(
            ["劇情", "剧情", "Story"],
            region=(0.55, 0.18, 0.95, 0.45), min_conf=0.5,
        )
        if story_entry is not None:
            return action_click_box(story_entry, "enter 劇情 hub")
        return action_click(0.817, 0.245, "enter 劇情 hub (hardcoded)")

    def _on_hub(self, screen: ScreenState) -> Dict[str, Any]:
        self._hub_search_ticks = 0
        for cat in CATEGORIES_TO_TRY:
            if cat in self._exhausted:
                continue
            target = HUB_SHORT_CLICK if cat == CATEGORY_SHORT \
                else HUB_SIDE_CLICK
            self._category = cat
            self._reset_grid_state()
            self.log(f"hub -> entering {cat} category")
            return action_click(
                target[0], target[1], f"open {cat} category",
            )
        if CATEGORY_MAIN not in self._exhausted:
            self._exhausted.append(CATEGORY_MAIN)
            self.log("主線劇情 push not implemented — skipping")
        return action_done("story mining finished (hub: nothing to do)")

    def _on_category_main(
        self, screen: ScreenState,
    ) -> Dict[str, Any]:
        if CATEGORY_MAIN not in self._exhausted:
            self._exhausted.append(CATEGORY_MAIN)
        self.log("on 主線劇情 page (not implemented) — backing out")
        return action_back("main story not supported")

    def _reset_grid_state(self) -> None:
        self._grid_pages_seen = 0
        self._grid_prev_fingerprint = ""
        self._grid_arrow_attempts = 0
        self._grid_settle_ticks = 0
        self._chapter_anchor_idx = 0
        self._chapter_idle_ticks = 0

    # ── Category grid ─────────────────────────────────────────────

    def _on_category(
        self, screen: ScreenState, expected_cat: str,
    ) -> Dict[str, Any]:
        if self._category != expected_cat:
            self._category = expected_cat
            self._reset_grid_state()

        # Settle: don't scan a freshly-loaded page until OCR has had
        # at least one tick to populate.  Prevents the "blank
        # transitional frame -> page-arrow click" v4 bug.
        populated = self._grid_is_populated(screen)
        if not populated:
            self._grid_settle_ticks += 1
            if self._grid_settle_ticks > 6:
                # Stuck on a blank frame for too long — try a swipe to
                # nudge the UI.
                self._grid_settle_ticks = 0
                self.log(f"{expected_cat}: blank grid stuck, swiping")
                return action_swipe(
                    GRID_SWIPE_FROM[0], GRID_SWIPE_FROM[1],
                    GRID_SWIPE_TO[0], GRID_SWIPE_TO[1],
                    duration_ms=400,
                    reason=f"swipe to nudge {expected_cat} grid",
                )
            return action_wait(350, "waiting for grid to populate")
        self._grid_settle_ticks = 0

        fingerprint = self._grid_fingerprint(screen)
        stalled = bool(fingerprint) \
            and fingerprint == self._grid_prev_fingerprint
        if fingerprint:
            self._grid_prev_fingerprint = fingerprint

        cell_idx = self._find_new_cell(screen)
        if cell_idx is not None:
            col = cell_idx % 2
            row = cell_idx // 2
            cx = GRID_CARD_COL_X[col]
            cy = GRID_CARD_ROW_Y[row]
            self._chapter_anchor_idx = 0
            self._chapter_idle_ticks = 0
            self.log(
                f"{expected_cat}: cell {cell_idx} (row={row} col={col}) "
                f"has 'New' -> entering"
            )
            return action_click(cx, cy, f"open {expected_cat} cell {cell_idx}")

        # No 'New' on this page.  Have we hit the page limit?
        if self._grid_pages_seen >= GRID_PAGE_LIMIT - 1:
            self.log(
                f"{expected_cat}: scanned {self._grid_pages_seen + 1} "
                f"pages with no 'New' -> category exhausted"
            )
            self._exhausted.append(expected_cat)
            self._category = None
            self._reset_grid_state()
            return action_back(f"{expected_cat} exhausted, back to hub")

        # Advance to next page.
        self._grid_pages_seen += 1
        self._grid_settle_ticks = 0
        if stalled:
            self._grid_arrow_attempts = 0
            self.log(
                f"{expected_cat}: page-arrow stalled, swiping to "
                f"page {self._grid_pages_seen}"
            )
            return action_swipe(
                GRID_SWIPE_FROM[0], GRID_SWIPE_FROM[1],
                GRID_SWIPE_TO[0], GRID_SWIPE_TO[1],
                duration_ms=400,
                reason=f"swipe to {expected_cat} page {self._grid_pages_seen}",
            )

        self._grid_arrow_attempts += 1
        return action_click(
            GRID_PAGE_NEXT_CLICK[0], GRID_PAGE_NEXT_CLICK[1],
            f"arrow -> {expected_cat} page {self._grid_pages_seen}",
        )

    def _grid_is_populated(self, screen: ScreenState) -> bool:
        """A grid page is 'populated' once we can OCR at least one
        volume title or 'New' badge inside the card body region.
        Loading frames typically only show the breadcrumb at top-left.
        """
        for b in screen.ocr_boxes:
            if b.confidence < 0.55:
                continue
            if not (0.05 < b.cx < 0.95 and 0.18 < b.cy < 0.85):
                continue
            t = (b.text or "").strip()
            if not t or len(t) < 2:
                continue
            return True
        return False

    def _grid_fingerprint(self, screen: ScreenState) -> str:
        parts: List[str] = []
        for b in screen.ocr_boxes:
            if b.confidence < 0.6:
                continue
            t = (b.text or "").strip()
            # Volume titles often contain '篇', '部', or are 2+ char
            # Chinese phrases.  Use anything inside the body region.
            if not (0.05 < b.cx < 0.95 and 0.18 < b.cy < 0.85):
                continue
            if len(t) >= 3 and t not in (
                "短篇劇情", "支線劇情", "主線劇情",
                "短篇剧情", "支线剧情", "主线剧情",
            ):
                parts.append(t)
        return "|".join(sorted(parts)[:8])  # cap to prevent OCR noise

    def _find_new_cell(self, screen: ScreenState) -> Optional[int]:
        for b in screen.ocr_boxes:
            if b.confidence < 0.5:
                continue
            t = (b.text or "").strip()
            if t.lower() != "new" and t != "新":
                continue
            for idx, _, _, x1, y1, x2, y2 in _grid_cells():
                if x1 <= b.cx <= x2 and y1 <= b.cy <= y2:
                    return idx
        return None

    # ── Chapter list (章節目錄) ───────────────────────────────────

    def _on_chapter_list(self, screen: ScreenState) -> Dict[str, Any]:
        # Active 入場 button -> click it.  Pick the topmost so we
        # play chapters in order.
        enter_buttons = self._find_enter_buttons(screen)
        if enter_buttons:
            top = min(enter_buttons, key=lambda b: b.cy)
            self._chapter_anchor_idx = 0
            self._chapter_idle_ticks = 0
            self.log(f"chapter_list: clicking 入場 at y={top.cy:.3f}")
            return action_click_box(top, "click 入場")

        # No 入場 OCR.  Either (a) the chapter list is loading,
        # (b) all chapters in this volume are cleared, or (c) OCR
        # missed the button.  We stayed in chapter_list classifier
        # so 章節目錄 breadcrumb was visible — meaning we ARE on
        # the list page.  Try fixed anchors before backing out.
        self._chapter_idle_ticks += 1
        if self._chapter_idle_ticks > self._chapter_max_idle:
            self.log("chapter_list: idle too long, backing to grid")
            self._chapter_anchor_idx = 0
            self._chapter_idle_ticks = 0
            return action_back("chapter list idle, back to grid")

        if self._chapter_anchor_idx < len(CHAPTER_LIST_ANCHORS):
            cx, cy = CHAPTER_LIST_ANCHORS[self._chapter_anchor_idx]
            self._chapter_anchor_idx += 1
            return action_click(
                cx, cy,
                f"chapter_list anchor {self._chapter_anchor_idx - 1}",
            )

        # Cycled anchors and still nothing — wait briefly.
        self._chapter_anchor_idx = 0
        return action_wait(350, "waiting for chapter list to populate")

    def _find_enter_buttons(self, screen: ScreenState) -> List[OcrBox]:
        boxes: List[OcrBox] = []
        for kw in ("入場", "入场", "Enter"):
            boxes.extend(screen.find_text(
                kw, region=CHAPTER_LIST_ENTER_REGION, min_conf=0.55,
            ))
        # Dedup by approximate position.
        unique: List[OcrBox] = []
        for b in boxes:
            if any(abs(u.cy - b.cy) < 0.03 for u in unique):
                continue
            unique.append(b)
        return unique

    # ── Chapter info (章節資訊) ─────────────────────────────────────

    def _on_chapter_info(self, screen: ScreenState) -> Dict[str, Any]:
        play = screen.find_any_text(
            ["進入章節", "进入章节", "Enter Chapter", "進入劇情"],
            region=CHAPTER_INFO_PLAY_REGION, min_conf=0.55,
        )
        if play is not None:
            self.log("chapter_info: clicking 進入章節")
            return action_click_box(play, "click 進入章節")
        # Fallback to reference's fixed coord (650, 511).
        self.log("chapter_info: 進入章節 not found, hardcoded")
        return action_click(
            CHAPTER_INFO_PLAY_FALLBACK[0], CHAPTER_INFO_PLAY_FALLBACK[1],
            "click 進入章節 (hardcoded)",
        )

    # ── Volume cleared (已完成所有章節) ──────────────────────────

    def _on_volume_cleared(self, screen: ScreenState) -> Dict[str, Any]:
        self.log("volume cleared (已完成所有章節), backing out to grid")
        self._chapter_anchor_idx = 0
        self._chapter_idle_ticks = 0
        return action_back("volume cleared, back to grid")

    # ── Helpers ──────────────────────────────────────────────────

    def _all_categories_exhausted(self) -> bool:
        for cat in CATEGORIES_TO_TRY:
            if cat not in self._exhausted:
                return False
        return True
