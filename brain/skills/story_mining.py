"""StoryMiningSkill v4 — template-based reactive rule list.

Lessons from re-reading ``study/ref/module/mini_story.py``,
``group_story.py``, and ``main_story.py``:

1.  **Flat feature -> action rules**, not a nested sub_state FSM.
    reference's ``picture.co_detect`` is a loop that on each iteration
    scans the screen for known features (image/colour/text), and for
    the *first* match it triggers the associated click.  This is much
    more robust to surprise popups and async transitions than a
    hierarchical FSM that only handles the screens it expected.

2.  **No persisted "what we played" ledger** for short / side
    stories.  The game itself shows the 'New' badge until a volume's
    chapters are all played, and the 入場 / Enter button is only
    blue/active for unplayed chapters.  Those *in-game* indicators
    are the source of truth.  Once a category page has no 'New'
    badges and the page-next arrow is grey, we're done.

3.  **Cutscene skip rules are always-on**.  reference folds
    ``plot_menu`` (top-right), ``plot_skip-plot-button``,
    ``plot_skip-plot-notice`` into *every* navigation call so any
    cutscene we end up in mid-task gets dismissed without us
    having to know "we're in a cutscene now".

4.  **Transient popups always-on too**: locked-chapter dialog,
    reward-acquired splash, news/notice modals.

So this skill is just a single ``tick()`` that runs an ordered list
of feature detectors.  The first one whose precondition matches
fires the corresponding action.  No sub_state branching.

Sub-state names are still recorded for trajectory diagnostics, but
they're labels, not control flow.

Coordinates remain reference-derived (mapped to normalised 0..1 against
its 1280x720 canvas).
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


# ── reference-derived coordinates (1280x720 -> 0..1) ──────────────────────

# Story hub sub-card click targets (the 4-card 主線/短篇/支線/重播 page).
HUB_SHORT_CLICK = (0.61, 0.45)
HUB_SIDE_CLICK = (0.83, 0.45)

# 2x3 grid card click centres (mini_story.to_region:
#   px (352,240) (931,240) (352,396) (931,396) (352,537) (931,537)
# normalised: col 0.275 / 0.727, rows 0.333 / 0.550 / 0.746).
GRID_CARD_COL_X: Tuple[float, float] = (0.275, 0.727)
GRID_CARD_ROW_Y: Tuple[float, float, float] = (0.333, 0.550, 0.746)

# 'New' badge top-left corners (mini_story.check_6_region_status: anchor
# px (86,155) (660,155) ... 54x25). Widened a bit because OCR boxes are
# bigger than the underlying badge sprite.
GRID_BADGE_COL_X: Tuple[float, float] = (0.067, 0.516)
GRID_BADGE_ROW_Y: Tuple[float, float, float] = (0.215, 0.424, 0.633)
GRID_BADGE_W = 0.16
GRID_BADGE_H = 0.07

# Page-next arrow (mini_story: pixel (1255, 357)).
GRID_PAGE_NEXT_CLICK = (0.98, 0.497)
GRID_SWIPE_FROM = (0.80, 0.50)
GRID_SWIPE_TO = (0.20, 0.50)

# Volume detail: reference one_detect probes (1073, 251) and (1073, 351)
# for the 入場 button colour. Click slightly inside the row so we
# hit either the row body or the button.
VOLUME_CHAPTER_CLICKS: Tuple[Tuple[float, float], Tuple[float, float]] = (
    (0.78, 0.349),  # top chapter row
    (0.78, 0.488),  # bottom chapter row
)
VOLUME_ENTER_REGION = (0.65, 0.28, 0.99, 0.58)

# Cutscene skip: reference folds plot_menu (1202, 37), plot_skip-plot-button
# (1208, 116), plot_skip-plot-notice (770, 519) into every nav call.
CUTSCENE_MENU_REGION = (0.82, 0.0, 1.0, 0.14)
CUTSCENE_MENU_FALLBACK = (0.94, 0.05)
CUTSCENE_SKIP_FALLBACK = (0.95, 0.16)
CUTSCENE_CONFIRM_REGION = (0.30, 0.55, 0.80, 0.82)
CUTSCENE_CONFIRM_FALLBACK = (0.60, 0.72)

# Categories we attempt, in priority order.
CATEGORY_SHORT = "short"
CATEGORY_SIDE = "side"
CATEGORY_MAIN = "main"  # not implemented — we skip it
CATEGORIES_TO_TRY: Tuple[str, ...] = (CATEGORY_SHORT, CATEGORY_SIDE)


def _grid_cells() -> List[Tuple[int, int, float, float, float, float]]:
    """Yield ``(idx, row, col, badge_x1, badge_y1, badge_x2, badge_y2)``
    for the six grid badge regions.  Used by the 'New' badge scanner.
    """
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
    """Auto-play unplayed chapters in 短篇劇情 and 支線劇情.

    Implementation is a single ordered rule chain (template-based).
    """

    def __init__(self) -> None:
        super().__init__("StoryMining")
        self.max_ticks = 800

        # Per-category exhaustion flags.  When a category page has no
        # 'New' badges across all pages we mark it done.
        self._exhausted: List[str] = []
        # Current category we're working on (set when we click a hub card).
        self._category: Optional[str] = None

        # Grid pagination state for the active category.
        self._grid_pages_seen: int = 0
        self._grid_prev_fingerprint: str = ""
        self._grid_arrow_attempts: int = 0
        # Hard limit on grid pages per category.  Short / side ship 2-3
        # pages; 4 is a conservative cap.
        self._grid_page_limit: int = 4

        # Volume detail click round-robin.  Two anchor rows; we cycle
        # through them on consecutive ticks if OCR doesn't find an 入場
        # button.  Reset whenever we leave the volume.
        self._volume_anchor_idx: int = 0
        self._volume_consecutive_ticks: int = 0

        # Cutscene skip stage tracker (label only, used for trajectory
        # readability — flow is decided by what's on screen).
        self._cutscene_taps: int = 0

        # Lobby/Mission navigation: if we've been searching for the hub
        # for too long give up.
        self._hub_search_ticks: int = 0
        self._hub_search_limit: int = 30

    # ── Lifecycle ───────────────────────────────────────────────────

    def reset(self) -> None:
        super().reset()
        self._exhausted = []
        self._category = None
        self._grid_pages_seen = 0
        self._grid_prev_fingerprint = ""
        self._grid_arrow_attempts = 0
        self._volume_anchor_idx = 0
        self._volume_consecutive_ticks = 0
        self._cutscene_taps = 0
        self._hub_search_ticks = 0

    # ── Tick: ordered rule chain ────────────────────────────────────

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1

        # 1. Always-on transient popups.
        a = self._handle_locked_popup(screen)
        if a is not None:
            self.sub_state = "popup"
            return a
        a = self._handle_reward_popup(screen)
        if a is not None:
            self.sub_state = "popup"
            return a

        # 2. Always-on cutscene skip (reference plot_menu / plot_skip rules).
        a = self._handle_cutscene(screen)
        if a is not None:
            self.sub_state = "playing"
            return a
        # Reset cutscene counter when not in a cutscene.
        self._cutscene_taps = 0

        # 3. Done?
        if self._all_categories_exhausted():
            self.sub_state = "finished"
            return action_done("story mining finished")

        # 4. Screen-kind dispatch.  We pick the most-specific
        # classifier first (volume detail) and fall through to the
        # broadest (lobby).
        kind = self._classify(screen)
        self.sub_state = kind

        if kind == "volume":
            return self._on_volume(screen)
        if kind == "category_short":
            return self._on_category(screen, CATEGORY_SHORT)
        if kind == "category_side":
            return self._on_category(screen, CATEGORY_SIDE)
        if kind == "category_main":
            return self._on_category_main(screen)
        if kind == "story_hub":
            return self._on_hub(screen)
        if kind == "mission":
            return self._on_mission(screen)
        if kind == "lobby":
            return self._on_lobby(screen)

        # Unknown screen — wait briefly, give up if it persists.
        self._hub_search_ticks += 1
        if self._hub_search_ticks > self._hub_search_limit:
            self.log("unknown screen for too long, giving up")
            return action_done("story mining unreachable")
        return action_wait(400, "unknown screen, waiting")

    # ── Screen classifiers ──────────────────────────────────────────

    def _classify(self, screen: ScreenState) -> str:
        # Volume detail: presence of 入場/Enter button OR a 第N章 row
        # with the right-hand list region populated.
        if self._is_volume_detail(screen):
            return "volume"

        # Category pages: the breadcrumb in the top-left tells us.
        breadcrumb = (0.0, 0.0, 0.30, 0.10)
        if screen.find_any_text(["短篇劇情", "短篇剧情"],
                                region=breadcrumb, min_conf=0.55):
            return "category_short"
        if screen.find_any_text(["支線劇情", "支线剧情"],
                                region=breadcrumb, min_conf=0.55):
            return "category_side"
        if screen.find_any_text(["主線劇情", "主线剧情"],
                                region=breadcrumb, min_conf=0.55) \
                and not self._is_story_hub(screen):
            return "category_main"

        # Story hub: all four sub-cards visible.
        if self._is_story_hub(screen):
            return "story_hub"

        current = self.detect_current_screen(screen)
        if current == "Lobby":
            return "lobby"
        if current == "Mission":
            return "mission"
        return "unknown"

    def _is_story_hub(self, screen: ScreenState) -> bool:
        # The hub shows all of 主線劇情, 短篇劇情, 支線劇情 + 重播.
        return (
            screen.has_text("主線劇情", min_conf=0.55)
            and screen.has_text("短篇劇情", min_conf=0.55)
            and screen.has_text("支線劇情", min_conf=0.55)
        )

    def _is_volume_detail(self, screen: ScreenState) -> bool:
        if screen.find_any_text(
            ["入場", "入场", "Enter", "進入", "进入"],
            region=VOLUME_ENTER_REGION, min_conf=0.55,
        ):
            return True
        # Fallback: a 第N章 row in the right half of screen.  This is
        # what visually marks the volume detail page even when 入場
        # OCR misses.
        for b in screen.ocr_boxes:
            if b.confidence < 0.55:
                continue
            t = (b.text or "").strip()
            if t.startswith("第") and "章" in t and b.cx > 0.55:
                return True
        return False

    # ── Always-on rules ─────────────────────────────────────────────

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
            region=CUTSCENE_CONFIRM_REGION, min_conf=0.55,
        )
        self.log(f"locked popup: '{blocked.text}', dismissing")
        if confirm is not None:
            return action_click_box(confirm, "dismiss locked popup")
        return action_click(0.50, 0.72, "dismiss locked popup (fallback)")

    def _handle_reward_popup(
        self, screen: ScreenState,
    ) -> Optional[Dict[str, Any]]:
        # reference marks reward_acquired by RGB at (640, 100).  We can't do
        # colour probes; instead OCR for "獲得"/"恭喜" reward banner.
        reward = screen.find_any_text(
            ["獲得獎勵", "获得奖励", "領取獎勵", "领取奖励",
             "恭喜獲得", "恭喜获得"],
            region=(0.20, 0.05, 0.80, 0.40), min_conf=0.55,
        )
        if reward is None:
            return None
        self.log(f"reward popup: '{reward.text}', tapping through")
        return action_click(0.50, 0.90, "dismiss reward popup")

    def _handle_cutscene(
        self, screen: ScreenState,
    ) -> Optional[Dict[str, Any]]:
        """template-based always-on cutscene skip.

        Trigger sequence on encountering a story dialogue:
            * MENU button visible top-right -> click it
            * Skip button revealed below MENU -> click it
            * 確認 / 是否略過 dialog -> confirm

        We don't track which stage we're in via state — every tick we
        just look for the deepest visible signal first (confirm
        dialog), then the skip button, then MENU.  This way if the
        game opens MENU itself or a popup interrupts us, we always
        do the right thing on the next tick.
        """
        confirm = screen.find_any_text(
            ["是否略過", "是否略过", "略過此", "略过此",
             "略過劇情", "略过剧情"],
            region=screen.CENTER, min_conf=0.55,
        )
        if confirm is not None:
            self._cutscene_taps += 1
            confirm_btn = screen.find_any_text(
                ["確認", "确认", "確定", "确定", "OK", "Yes"],
                region=CUTSCENE_CONFIRM_REGION, min_conf=0.55,
            )
            if confirm_btn is not None:
                return action_click_box(confirm_btn, "confirm story skip")
            return action_click(
                CUTSCENE_CONFIRM_FALLBACK[0], CUTSCENE_CONFIRM_FALLBACK[1],
                "confirm story skip (fallback)",
            )

        skip = screen.find_any_text(
            ["跳過劇情", "跳过剧情", "Skip Story", "略過劇情"],
            region=(0.78, 0.05, 1.0, 0.30), min_conf=0.55,
        )
        if skip is not None:
            self._cutscene_taps += 1
            return action_click_box(skip, "click Skip Story")

        skip_short = screen.find_any_text(
            ["SKIP", "Skip", "跳過", "跳过"],
            region=(0.78, 0.05, 1.0, 0.30), min_conf=0.55,
        )
        if skip_short is not None:
            self._cutscene_taps += 1
            return action_click_box(skip_short, "click Skip")

        menu = screen.find_any_text(
            ["MENU"], region=CUTSCENE_MENU_REGION, min_conf=0.65,
        )
        if menu is not None:
            self._cutscene_taps += 1
            return action_click_box(menu, "click MENU to reveal Skip")

        # Auto-mode toggle / dialog text — heuristic: if we see "AUTO"
        # in top-right and no breadcrumb, we're inside a cutscene that
        # didn't expose MENU yet.  Fallback to MENU coord.
        auto = screen.find_text(
            "AUTO", region=CUTSCENE_MENU_REGION, min_conf=0.6,
        )
        if auto:
            self._cutscene_taps += 1
            return action_click(
                CUTSCENE_MENU_FALLBACK[0], CUTSCENE_MENU_FALLBACK[1],
                "click MENU (AUTO seen, hardcoded)",
            )

        return None

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
        story_entry = screen.find_any_text(
            ["劇情", "剧情", "Story"], min_conf=0.5,
        )
        if story_entry is not None:
            return action_click_box(story_entry, "enter 劇情 hub")
        return action_click(0.26, 0.71, "enter 劇情 hub (hardcoded)")

    def _on_hub(self, screen: ScreenState) -> Dict[str, Any]:
        # Reset hub-search counter — we got there.
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

        # Both done; main not supported.
        if CATEGORY_MAIN not in self._exhausted:
            self._exhausted.append(CATEGORY_MAIN)
            self.log("主線劇情 push not implemented — skipping")
        return action_done("story mining finished (hub: nothing to do)")

    def _on_category_main(
        self, screen: ScreenState,
    ) -> Dict[str, Any]:
        if CATEGORY_MAIN not in self._exhausted:
            self._exhausted.append(CATEGORY_MAIN)
        self.log("on 主線劇情 page (not yet implemented) — backing out")
        return action_back("main story not supported")

    def _reset_grid_state(self) -> None:
        self._grid_pages_seen = 0
        self._grid_prev_fingerprint = ""
        self._grid_arrow_attempts = 0
        self._volume_anchor_idx = 0
        self._volume_consecutive_ticks = 0

    # ── Category grid ─────────────────────────────────────────────

    def _on_category(
        self, screen: ScreenState, expected_cat: str,
    ) -> Dict[str, Any]:
        # Sanity check: did we land on the wrong category page?  Switch
        # focus to whatever category we're actually on so the per-page
        # traversal makes sense.
        if self._category != expected_cat:
            self._category = expected_cat
            self._reset_grid_state()

        # Update grid fingerprint (for stall detection).
        fingerprint = self._grid_fingerprint(screen)
        stalled = bool(fingerprint) and fingerprint == self._grid_prev_fingerprint
        if fingerprint:
            self._grid_prev_fingerprint = fingerprint

        # Scan 6 cells for a 'New' badge.
        cell_idx = self._find_new_cell(screen)
        if cell_idx is not None:
            col = cell_idx % 2
            row = cell_idx // 2
            cx = GRID_CARD_COL_X[col]
            cy = GRID_CARD_ROW_Y[row]
            # Reset volume traversal state for the new volume.
            self._volume_anchor_idx = 0
            self._volume_consecutive_ticks = 0
            self.log(
                f"{expected_cat}: cell {cell_idx} (row={row} col={col}) "
                f"has 'New' — entering"
            )
            return action_click(cx, cy, f"open {expected_cat} cell {cell_idx}")

        # No 'New' on this page.  Have we hit our page limit?
        if self._grid_pages_seen >= self._grid_page_limit - 1:
            self.log(
                f"{expected_cat}: scanned {self._grid_pages_seen + 1} "
                f"pages with no 'New' — category exhausted"
            )
            self._exhausted.append(expected_cat)
            self._category = None
            self._reset_grid_state()
            return action_back(
                f"{expected_cat} exhausted, back to hub",
            )

        # Advance to next page.  If the previous click didn't change
        # the page contents (stalled), use a swipe instead.
        self._grid_pages_seen += 1
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

    def _grid_fingerprint(self, screen: ScreenState) -> str:
        parts: List[str] = []
        for b in screen.ocr_boxes:
            if b.confidence < 0.6:
                continue
            t = (b.text or "").strip()
            if "篇" in t and t not in (
                "短篇劇情", "支線劇情", "主線劇情",
                "短篇剧情", "支线剧情", "主线剧情",
            ):
                parts.append(t)
        return "|".join(sorted(parts))

    def _find_new_cell(self, screen: ScreenState) -> Optional[int]:
        for b in screen.ocr_boxes:
            if b.confidence < 0.5:
                continue
            t = (b.text or "").strip()
            if t.lower() != "new" and t != "新":
                continue
            for idx, row, col, x1, y1, x2, y2 in _grid_cells():
                if x1 <= b.cx <= x2 and y1 <= b.cy <= y2:
                    return idx
        return None

    # ── Volume detail ─────────────────────────────────────────────

    def _on_volume(self, screen: ScreenState) -> Dict[str, Any]:
        # Always prefer an 入場 OCR hit.  Click the topmost one — that's
        # the next unplayed chapter.
        enter_buttons = screen.find_text(
            "入場", region=VOLUME_ENTER_REGION, min_conf=0.55,
        ) + screen.find_text(
            "入场", region=VOLUME_ENTER_REGION, min_conf=0.55,
        ) + screen.find_text(
            "Enter", region=VOLUME_ENTER_REGION, min_conf=0.55,
        )
        if enter_buttons:
            top = min(enter_buttons, key=lambda b: b.cy)
            self.log(f"volume: clicking 入場 at y={top.cy:.3f}")
            self._volume_consecutive_ticks = 0
            return action_click_box(top, "click 入場")

        # No 入場 OCR but we're on the volume page (chapter rows
        # visible).  Fall back to reference's two fixed anchors, top first.
        # If both anchors fail (still on volume after multiple ticks),
        # back out.
        self._volume_consecutive_ticks += 1
        if self._volume_consecutive_ticks > 6:
            self.log("volume: stuck without 入場, backing out to grid")
            self._volume_anchor_idx = 0
            self._volume_consecutive_ticks = 0
            return action_back("volume exhausted, back to grid")

        idx = self._volume_anchor_idx
        if idx >= len(VOLUME_CHAPTER_CLICKS):
            self.log("volume: anchors exhausted, backing out to grid")
            self._volume_anchor_idx = 0
            return action_back("volume anchors exhausted")

        self._volume_anchor_idx += 1
        cx, cy = VOLUME_CHAPTER_CLICKS[idx]
        return action_click(cx, cy, f"volume row {idx} (anchor)")

    # ── Helpers ──────────────────────────────────────────────────

    def _all_categories_exhausted(self) -> bool:
        for cat in CATEGORIES_TO_TRY:
            if cat not in self._exhausted:
                return False
        return True
