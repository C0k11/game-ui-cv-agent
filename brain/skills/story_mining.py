"""StoryMiningSkill — auto-play unfinished 短篇/支線 stories.

This is the "劇情挖礦" feature.  The 劇情 hub hosts four sub-sections
(主線 / 短篇 / 支線 / 重播); of those, short and side are trivial to
automate: each card is a one-shot cutscene, there's no battle, and the
game visually marks unplayed cards with a blue "New" badge and cleared
cards with a grey "完成" label.

This MVP handles 短篇 and 支線 only:

    1.  The skill expects to start somewhere in the 劇情 navigation
        tree.  If it lands on the main story-hub page (MAIN STORY +
        三 sub-card triad visible) it picks an outstanding category;
        if it's already on a category list (short/side) it processes
        that one; if it's nowhere recognisable it signals done and
        defers to the next pipeline skill.  No lobby-entry auto-click
        yet — a thin follow-up can add that once we've proven the
        core flow on a real run.

    2.  Inside a category list we enumerate visible "New" badges, map
        each to the story card whose title sits below/near it, and
        click the first one.  (The 完成 label is consulted only as a
        sanity check — if the ledger says done but the screen shows
        New, the ledger wins.)

    3.  Inside a story we run the same MENU → Skip → confirm machine
        the event-activity skill uses.  Once the cutscene dismisses
        we land back on the category list and mark the title in the
        persistent store.

    4.  When a page shows no "New" and no "EMPTY" placeholder we
        swipe left/tap page-dots to iterate.  After cycling all
        visible pages without finding unplayed content we back out
        to the hub and try the other category, then exit the skill.

Main story (主線劇情) is intentionally deferred — reference's pusher is
379 lines and hinges on pixel-exact coordinates; a proper port needs
its own commit with its own tests.  When the skill encounters 主線劇情
it logs and skips it rather than crashing.
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


# ── UI regions (normalised 0..1) ────────────────────────────────────────

# Story hub sub-cards.  Click centres derived from the OCR evidence at
# data/trajectories/run_20260423_024430/tick_0004.json.
_HUB_CARD_SHORT_CLICK = (0.61, 0.45)   # 短篇劇情
_HUB_CARD_SIDE_CLICK = (0.83, 0.45)    # 支線劇情
_HUB_CARD_MAIN_CLICK = (0.27, 0.60)    # 主線劇情 (deferred)
_HUB_CARD_REPLAY_CLICK = (0.83, 0.85)  # 重播 (ignored)

# Top-left breadcrumb.  "劇情" here means we're on the hub page OR a
# category.  We disambiguate by looking for the triad of sub-cards.
_BREADCRUMB_REGION = (0.0, 0.0, 0.25, 0.09)

# Story-list card grid occupies roughly y in [0.10, 0.85].
_LIST_CARD_REGION = (0.03, 0.10, 1.0, 0.88)

# Page-indicator dots at the bottom of the category list.
_PAGE_DOT_REGION = (0.40, 0.90, 0.70, 1.0)


# ── Skill ──────────────────────────────────────────────────────────────

class StoryMiningSkill(BaseSkill):
    """Auto-play every short/side story that's still marked 'New'."""

    def __init__(self) -> None:
        super().__init__("StoryMining")
        self.max_ticks = 400  # many stories × a handful of ticks each

        self._store: StoryProgressStore = get_store()

        # Sub-state machine; see class docstring.
        #   enter       — just started, figuring out where we are
        #   hub         — on story-hub page, pick a category
        #   list        — on a category list, look for New cards
        #   playing     — inside a story, running the skip dance
        #   exit_story  — cutscene dismissed, return to list
        #   finished    — everything done, signal pipeline to advance
        self.sub_state = "enter"
        self._enter_ticks: int = 0

        # Which category is currently being processed.  None while on
        # the hub; set before switching to "list"; cleared on return.
        self._category: Optional[str] = None

        # Categories we've already exhausted this run — don't re-enter.
        self._exhausted: List[str] = []

        # Title of the story we're currently playing; mark_done gets
        # called once the skip machine completes.
        self._pending_title: str = ""

        # Skip-machine state, same shape as event_activity._skip_stage.
        #   0 = not skipping
        #   1 = clicked MENU
        #   2 = clicked Skip
        #   3 = confirmed — waiting for cutscene to dismiss
        self._skip_stage: int = 0
        self._cutscene_taps: int = 0

        # Count pages we've scrolled past on the current list so we
        # stop cycling if every page is empty.
        self._list_swipes: int = 0
        self._LIST_SWIPE_LIMIT: int = 3  # ≤3 pages per game build

        # Stall-detection.  If the screen never changes shape across
        # many ticks we abort the skill and let the pipeline timeout
        # machinery kick in.
        self._stall_ticks: int = 0

    def reset(self) -> None:
        super().reset()
        self.sub_state = "enter"
        self._enter_ticks = 0
        self._category = None
        self._exhausted = []
        self._pending_title = ""
        self._skip_stage = 0
        self._cutscene_taps = 0
        self._list_swipes = 0
        self._stall_ticks = 0

    # ── Tick dispatch ───────────────────────────────────────────────

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1
        self._enter_ticks += 1

        handler = {
            "enter": self._on_enter,
            "hub": self._on_hub,
            "list": self._on_list,
            "playing": self._on_playing,
            "exit_story": self._on_exit_story,
            "finished": self._on_finished,
        }.get(self.sub_state, self._on_enter)
        return handler(screen)

    # ── Sub-state: enter ───────────────────────────────────────────

    def _on_enter(self, screen: ScreenState) -> Dict[str, Any]:
        """Figure out where we are and route to the right sub-state."""
        if self._is_story_hub(screen):
            self.log("on story hub — routing to category picker")
            self.sub_state = "hub"
            return self._on_hub(screen)

        cat = self._detect_category_page(screen)
        if cat is not None:
            self.log(f"already on {cat!r} category list — starting scan")
            self._category = cat
            self._list_swipes = 0
            self.sub_state = "list"
            return self._on_list(screen)

        # Not in the story tree.  Without a lobby entry step we can't
        # navigate here ourselves yet (TODO), so signal done.
        if self._enter_ticks > 6:
            self.log("story hub not detected within 6 ticks, skipping")
            return action_done("story hub not reached")
        return action_wait(500, "searching for story hub")

    # ── Sub-state: hub ─────────────────────────────────────────────

    def _on_hub(self, screen: ScreenState) -> Dict[str, Any]:
        # Choose the first category we haven't exhausted yet.  Short
        # before side simply because 短篇 sits above 支線 on the page.
        for cat, target in (
            (StoryProgressStore.SHORT, _HUB_CARD_SHORT_CLICK),
            (StoryProgressStore.SIDE, _HUB_CARD_SIDE_CLICK),
        ):
            if cat in self._exhausted:
                continue
            self._category = cat
            self._list_swipes = 0
            self.sub_state = "list"
            self.log(f"entering {cat} category")
            return action_click(
                target[0], target[1], f"open {cat} story list",
            )

        # Main story deferred; explicitly log.
        if StoryProgressStore.MAIN not in self._exhausted:
            self._exhausted.append(StoryProgressStore.MAIN)
            self.log("主線劇情 push not yet implemented — skipping")

        self.log("all available categories exhausted")
        self.sub_state = "finished"
        return action_done("all story categories done")

    # ── Sub-state: list ────────────────────────────────────────────

    def _on_list(self, screen: ScreenState) -> Dict[str, Any]:
        # Sanity: if we got bounced back to the hub (back-arrow too
        # eager, or the game auto-closed us) reroute.
        if self._is_story_hub(screen):
            self.log(f"category list closed unexpectedly, back to hub")
            if self._category:
                self._exhausted.append(self._category)
                self._category = None
            self.sub_state = "hub"
            return self._on_hub(screen)

        # Find a "New" badge → find its nearest story title → click it.
        new_card = self._find_new_card(screen)
        if new_card is not None:
            cx, cy, title = new_card
            # Ledger sanity: if we've already marked this title done
            # but the screen still shows New, trust the screen (game
            # state is ground truth for rewards) but log it.
            if self._category and self._store.is_done(self._category, title):
                self.log(
                    f"ledger says '{title}' done but screen shows New "
                    f"— re-playing per screen state"
                )
            self._pending_title = title
            self._skip_stage = 0
            self._cutscene_taps = 0
            self.sub_state = "playing"
            self.log(f"starting story '{title}' (category={self._category})")
            return action_click(cx, cy, f"open story '{title}'")

        # No New visible — try next page.  Max 3 pages per build, so
        # exceeding that marks the category exhausted.
        if self._list_swipes < self._LIST_SWIPE_LIMIT:
            self._list_swipes += 1
            self.log(
                f"no 'New' on page {self._list_swipes}; swiping to next",
            )
            return action_swipe(
                0.85, 0.50, 0.15, 0.50, duration_ms=400,
                reason="swipe category list to next page",
            )

        # Exhausted this category.
        self.log(f"category '{self._category}' exhausted")
        if self._category:
            self._exhausted.append(self._category)
            self._category = None
        self.sub_state = "enter"
        self._enter_ticks = 0
        return action_back("back to story hub")

    # ── Sub-state: playing ─────────────────────────────────────────

    def _on_playing(self, screen: ScreenState) -> Dict[str, Any]:
        """Skip the cutscene via the MENU → Skip → confirm dance.

        Same control flow as event_activity._story's skip branch, but
        lifted out here to avoid cross-skill state coupling.  Once the
        skip confirmation is clicked and the screen leaves cutscene
        shape we transition to exit_story.
        """
        # Leaving cutscene → we're done.  Detect by: category list
        # reappears OR the story-hub page reappears.
        if self._detect_category_page(screen) is not None or self._is_story_hub(screen):
            self.log("cutscene dismissed — marking story complete")
            if self._category and self._pending_title:
                newly = self._store.mark_done(self._category, self._pending_title)
                self.log(
                    f"marked '{self._pending_title}' done in {self._category} "
                    f"(newly={newly})"
                )
            self._pending_title = ""
            self._skip_stage = 0
            self._cutscene_taps = 0
            self.sub_state = "exit_story"
            return action_wait(300, "cutscene over, back to list")

        # Already confirmed? Wait for game to transition out.
        if self._skip_stage >= 3:
            self._cutscene_taps += 1
            if self._cutscene_taps > 20:
                self.log("stuck after skip confirm, pressing back")
                self._skip_stage = 0
                self._cutscene_taps = 0
                return action_back("post-skip stall, back out")
            return action_wait(300, "waiting for cutscene to dismiss")

        if self._skip_stage == 2:
            # Confirm-skip dialog.
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

        # _skip_stage == 1: MENU was clicked, look for Skip button.
        skip = screen.find_any_text(
            ["SKIP", "Skip", "跳過", "跳过", "スキップ"], min_conf=0.55,
        )
        if skip:
            self._skip_stage = 2
            return action_click_box(skip, "click Skip in menu")
        self._skip_stage = 2
        return action_click(0.95, 0.16, "click Skip (hardcoded)")

    # ── Sub-state: exit_story ──────────────────────────────────────

    def _on_exit_story(self, screen: ScreenState) -> Dict[str, Any]:
        # Detect landing on list or hub; continue accordingly.
        if self._category and self._detect_category_page(screen) == self._category:
            self._list_swipes = 0  # restart scan from page 1
            self.sub_state = "list"
            return self._on_list(screen)
        if self._is_story_hub(screen):
            self._category = None
            self.sub_state = "hub"
            return self._on_hub(screen)
        # Still transitioning — small wait; but after too long press back.
        self._stall_ticks += 1
        if self._stall_ticks > 15:
            self._stall_ticks = 0
            return action_back("exit_story stalled, back")
        return action_wait(300, "returning to story list")

    # ── Sub-state: finished ────────────────────────────────────────

    def _on_finished(self, screen: ScreenState) -> Dict[str, Any]:
        return action_done("story mining finished")

    # ── Screen classifiers ─────────────────────────────────────────

    def _is_story_hub(self, screen: ScreenState) -> bool:
        """Hub page detected when all three sub-card titles visible."""
        have_main = screen.has_text("主線劇情", min_conf=0.6)
        have_short = screen.has_text("短篇劇情", min_conf=0.6)
        have_side = screen.has_text("支線劇情", min_conf=0.6)
        return have_main and have_short and have_side

    def _detect_category_page(self, screen: ScreenState) -> Optional[str]:
        """Return 'short' / 'side' / 'main' if we're on a category list.

        Category pages show the category name in the top-left
        breadcrumb and do NOT show the triad of sub-cards (which only
        appears on the hub).
        """
        if self._is_story_hub(screen):
            return None
        bc_short = screen.find_text(
            "短篇劇情", region=_BREADCRUMB_REGION, min_conf=0.55,
        )
        if bc_short:
            return StoryProgressStore.SHORT
        bc_side = screen.find_text(
            "支線劇情", region=_BREADCRUMB_REGION, min_conf=0.55,
        )
        if bc_side:
            return StoryProgressStore.SIDE
        bc_main = screen.find_text(
            "主線劇情", region=_BREADCRUMB_REGION, min_conf=0.55,
        )
        if bc_main:
            return StoryProgressStore.MAIN
        return None

    # ── Card detection ─────────────────────────────────────────────

    def _find_new_card(
        self, screen: ScreenState,
    ) -> Optional[Tuple[float, float, str]]:
        """Locate a story card bearing a 'New' badge.

        Returns (click_x, click_y, title) or None.

        Strategy:
          * Collect every OCR box whose text is 'New' / 'NEW' in the
            card grid region.
          * For each, find the closest Chinese-title OCR box below it
            (same column).  That's the card's label.
          * Compute a click anchor near the card centre — a bit below
            the New badge works because the badge sits on the card's
            top-left.
        """
        new_boxes: List[OcrBox] = []
        for b in screen.ocr_boxes:
            t = (b.text or "").strip()
            if t.lower() != "new":
                continue
            if b.confidence < 0.45:
                continue
            if not (_LIST_CARD_REGION[0] <= b.cx <= _LIST_CARD_REGION[2]):
                continue
            if not (_LIST_CARD_REGION[1] <= b.cy <= _LIST_CARD_REGION[3]):
                continue
            new_boxes.append(b)

        if not new_boxes:
            return None

        # Sort top-to-bottom, left-to-right so we play the earliest
        # visible card first (deterministic ordering is easier to
        # reason about and de-duplicate against the store).
        new_boxes.sort(key=lambda b: (round(b.cy, 2), round(b.cx, 2)))

        first = new_boxes[0]
        title = self._title_near(screen, first)
        if not title:
            # Fallback: treat as an anonymous story.  Hash of the
            # click-anchor coordinates is NOT stable across runs, so
            # we store a sentinel like "__unnamed@0.55x0.18" — still
            # unique-ish per page/slot, avoids re-clicking this tick.
            title = f"__unnamed@{round(first.cx, 2)}x{round(first.cy, 2)}"

        # Card body sits to the RIGHT and BELOW the New badge (badge
        # is at the top-left corner).  Click around card centre: New
        # box + (0.15, 0.03) empirically lands in the middle of the
        # card art.  Clamp to the card region.
        cx = max(0.05, min(0.97, first.cx + 0.15))
        cy = max(0.12, min(0.85, first.cy + 0.03))
        return cx, cy, title

    def _title_near(self, screen: ScreenState, badge: OcrBox) -> str:
        """Find the Chinese title OCR box closest to ``badge`` and
        below-right of it (story titles are on the card's right half).
        """
        best: Optional[OcrBox] = None
        best_dist = 0.30  # max normalised distance cap
        for b in screen.ocr_boxes:
            if b is badge:
                continue
            if b.confidence < 0.45:
                continue
            t = (b.text or "").strip()
            if not t or t.lower() in ("new", "完成", "empty"):
                continue
            # Title must be to the right (or below) of the badge; same
            # card.  Badges are top-LEFT of the card so titles live
            # within ~0.25 to the right and ~0.15 below.
            dx = b.cx - badge.cx
            dy = b.cy - badge.cy
            if dx < -0.02 or dx > 0.28:
                continue
            if dy < -0.03 or dy > 0.18:
                continue
            # Skip obviously non-title strings.
            if t.isdigit() or t.startswith("O+"):
                continue
            d = (dx ** 2 + dy ** 2) ** 0.5
            if d < best_dist:
                best_dist = d
                best = b
        return best.text.strip() if best else ""
