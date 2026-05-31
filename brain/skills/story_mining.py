"""StoryMiningSkill v6 — pure-YOLO. Find & clear unplayed story chapters.

Full rewrite (2026-05-28): the v5 design used hardcoded 2x3 grid cell
coords + OCR "new"/"入場"/"進入章節" text + fixed cutscene-skip anchors.
That capped the skill to a fixed layout and OCR reliability. The ui model
now has direct cls for everything story-related:

  NEW_MARK (429)          NEW! badge on a card/chapter
  STORY_ICON_UNDONE (430) uncleared chapter node      ← user's "没打过"
  STORY_ICON_DONE  (427)  cleared chapter node
  DOT_YELLOW (6)          unfinished/new badge        ← user's "有黄点"
  STORY_SHORT/SIDE/MAIN   hub category cards
  STAGE_ENTER (79)        入場 button (chapter list)
  STORY_ENTER_CHAPTER(139) 進入章節 button (chapter info modal)
  STORY_MENU (431)        cutscene MENU
  STORY_SKIP (141)        跳過 (STORY_SKIP_DISABLED 432 = greyed)
  STORY_TAP_CONTINUE(142) 點擊繼續
  GOT_REWARD (397)        post-chapter reward splash
  HUB_STORY (68)          劇情 tile in mission hub

Strategy: reactive priority chain (template-based, no nested sub_state). Each
tick, resolve the highest-priority actionable cls. To "mine" we click the
first NEW_MARK / STORY_ICON_UNDONE / yellow-dot node we see — YOLO gives us
its exact coords, so NO grid math. When a category shows no unplayed node
for N ticks, back out and try the next category; when all categories are
exhausted, done.

NO OCR anywhere. If a needed cls isn't visible, log the gap + wait.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from brain.skills.base import (
    BaseSkill, ScreenState, YoloBox,
    action_click, action_click_box, action_back, action_done, action_wait,
)
from brain.skills import ui_classes as UC


class StoryMiningSkill(BaseSkill):
    def __init__(self) -> None:
        super().__init__("StoryMining")
        self.max_ticks = 1200
        # Category rotation: short → side. (main = not auto-mined.)
        self._categories = [UC.STORY_SHORT, UC.STORY_SIDE]
        self._cat_idx = 0
        self._exhausted: List[str] = []
        # Per-category "no unplayed node" streak → advance/exhaust.
        self._barren_ticks = 0
        self._barren_limit = 6
        # nav caps
        self._nav_ticks = 0
        self._nav_limit = 40
        # cutscene anti-spam
        self._cut_ticks = 0
        # entered-node cooldown (after clicking a node, wait for transition)
        self._click_cooldown = 0

    def reset(self) -> None:
        super().reset()
        self._cat_idx = 0
        self._exhausted = []
        self._barren_ticks = 0
        self._nav_ticks = 0
        self._cut_ticks = 0
        self._click_cooldown = 0

    # ── tick: reactive priority chain ─────────────────────────────────
    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1
        if self.ticks >= self.max_ticks:
            self.log("timeout")
            return action_done("story mining timeout")

        if self._click_cooldown > 0:
            self._click_cooldown -= 1
            return action_wait(400, f"story: settle ({self._click_cooldown})")

        # P0: full-screen splashes (level-up / reward) — tap to dismiss.
        splash = self.find_cls(screen, [UC.BOND_LEVELUP, UC.REGION_LEVELUP], conf=0.30)
        if splash is not None:
            return action_click(0.5, 0.5, f"dismiss splash ({splash.cls_name})")
        reward = self.find_cls(screen, UC.GOT_REWARD, conf=0.30)
        if reward is not None:
            self._cut_ticks = 0
            return action_click_box(reward, "dismiss reward (YOLO 获得奖励)")

        # P1: cutscene skip chain. STORY_MENU → STORY_SKIP → confirm.
        cut = self._handle_cutscene(screen)
        if cut is not None:
            return cut

        # P2: chapter-info modal — 進入章節.
        play = self.find_cls(screen, UC.STORY_ENTER_CHAPTER, conf=0.30,
                             region=(0.25, 0.50, 0.80, 0.90))
        if play is not None:
            self._barren_ticks = 0
            self._click_cooldown = 3
            return action_click_box(play, "enter chapter (YOLO 进入章节)")

        # P3: MINE — click first unplayed node (NEW / undone-icon / yellow
        # dot) or an active 入場 button. THIS is the core "find 没打过/
        # 黄点/new" logic the user asked for — YOLO gives coords directly.
        node = self._find_unplayed_node(screen)
        if node is not None:
            self._barren_ticks = 0
            self._click_cooldown = 3
            return action_click_box(node, f"mine unplayed (YOLO {node.cls_name})")
        enter_btn = self.find_cls(screen, UC.STAGE_ENTER, conf=0.30,
                                  region=(0.55, 0.15, 0.99, 0.70))
        if enter_btn is not None:
            self._barren_ticks = 0
            self._click_cooldown = 3
            return action_click_box(enter_btn, "enter chapter (YOLO 入场键)")

        # P4: on a story page but no unplayed node visible → barren.
        page = self.detect_screen_yolo(screen)
        on_story = (page == "Story") or self._on_any_story_page(screen)
        if on_story:
            self._barren_ticks += 1
            if self._barren_ticks <= self._barren_limit:
                return action_wait(350, f"story: scanning for unplayed ({self._barren_ticks})")
            # No unplayed node for this whole category — exhaust + advance.
            cur_cat = self._categories[self._cat_idx] if self._cat_idx < len(self._categories) else None
            if cur_cat and cur_cat not in self._exhausted:
                self._exhausted.append(cur_cat)
                self.log(f"category {cur_cat} exhausted (no unplayed node)")
            self._barren_ticks = 0
            self._cat_idx += 1
            if self._cat_idx >= len(self._categories):
                return action_done("story mining finished (all categories cleared)")
            # back out to hub to pick next category
            back = self.find_cls(screen, [UC.BTN_BACK, UC.BTN_HOME], conf=0.30)
            return (action_click_box(back, "back to hub for next category")
                    if back else action_back("back to hub for next category"))

        # P5: story hub — pick the current (non-exhausted) category card.
        hub_card = self._pick_hub_card(screen)
        if hub_card is not None:
            self._barren_ticks = 0
            self._click_cooldown = 2
            return action_click_box(hub_card, f"open category (YOLO {hub_card.cls_name})")

        # P6: mission hub — 劇情 tile.
        story_tile = self.find_cls(screen, UC.HUB_STORY, conf=0.30)
        if story_tile is not None:
            self._nav_ticks = 0
            self._click_cooldown = 2
            return action_click_box(story_tile, "enter 剧情 hub (YOLO 剧情)")

        # P7: navigation gaps — lobby → mission hub.
        return self._navigate(screen)

    # ── cutscene ──────────────────────────────────────────────────────
    def _handle_cutscene(self, screen: ScreenState) -> Optional[Dict[str, Any]]:
        # FIX (review 高危2): once we've clicked Skip/MENU (_cut_ticks>0),
        # the 跳過 icon often stays visible IN the skip-confirm dialog
        # frame. The old order checked Skip first → re-clicked Skip every
        # tick → never reached confirm → stuck to max_ticks=1200. So when
        # _cut_ticks>0, look for the confirm dialog FIRST.
        if self._cut_ticks > 0:
            confirm = self.find_cls(screen, UC.BTN_CONFIRM, conf=0.30,
                                    region=(0.30, 0.55, 0.80, 0.85))
            if confirm is not None:
                self._cut_ticks = 0
                self._click_cooldown = 2
                return action_click_box(confirm, "confirm story skip (YOLO 确认键)")
            # narration scrolling again (skip took effect) → reset + continue
            cont = self.find_cls(screen, UC.STORY_TAP_CONTINUE, conf=0.30)
            if cont is not None:
                self._cut_ticks = 0
                return action_click_box(cont, "tap continue (YOLO 点击继续)")

        # Click Skip — but cap retries so a persistent skip icon (with no
        # confirm appearing) can't loop forever.
        if self._cut_ticks < 3:
            skip = self.find_cls(screen, UC.STORY_SKIP, conf=0.40,
                                 region=(0.82, 0.05, 1.0, 0.30))
            if skip is not None:
                self._cut_ticks += 1
                return action_click_box(skip, "click Skip (YOLO 跳过故事键)")
            menu = self.find_cls(screen, UC.STORY_MENU, conf=0.40,
                                 region=(0.82, 0.0, 1.0, 0.15))
            if menu is not None:
                self._cut_ticks += 1
                return action_click_box(menu, "open cutscene MENU (YOLO 剧情menu)")
        # stuck narration (not yet in skip flow) → tap continue
        cont = self.find_cls(screen, UC.STORY_TAP_CONTINUE, conf=0.30)
        if cont is not None:
            return action_click_box(cont, "tap continue (YOLO 点击继续)")
        # Skip retries exhausted with no confirm/continue → decay so we
        # don't stay wedged in cutscene-handling forever; let P2-P7 run.
        if self._cut_ticks > 0:
            self._cut_ticks = max(0, self._cut_ticks - 1)
        return None

    # ── node finding (the "mine 没打过/黄点/new" core) ─────────────────
    def _find_unplayed_node(self, screen: ScreenState) -> Optional[YoloBox]:
        """Return the top-most unplayed chapter node, or None.

        Priority: NEW badge > undone icon > yellow dot (in the node area,
        not the top-bar). All give exact coords — click directly, no grid.
        """
        # NEW! badge — strongest "未玩过" signal.
        news = self.find_all_cls(screen, UC.NEW_MARK, conf=0.30)
        if news:
            return min(news, key=lambda b: b.cy)  # top-most first
        # uncleared chapter-node icon.
        undone = self.find_all_cls(screen, UC.STORY_ICON_UNDONE, conf=0.30)
        if undone:
            return min(undone, key=lambda b: b.cy)
        # yellow dot inside the content area (exclude top HUD y<0.12 and
        # bottom nav y>0.92 where dots are nav badges, not story nodes).
        dots = self.find_all_cls(screen, UC.DOT_YELLOW, conf=0.40,
                                 region=(0.05, 0.12, 0.99, 0.92))
        if dots:
            return min(dots, key=lambda b: b.cy)
        return None

    def _on_any_story_page(self, screen: ScreenState) -> bool:
        """True if any story-context cls visible (hub card, node, enter btn,
        chapter info, cutscene)."""
        return self.find_cls(
            screen,
            [UC.STORY_SHORT, UC.STORY_SIDE, UC.STORY_MAIN,
             UC.STORY_ICON_DONE, UC.STORY_ICON_UNDONE, UC.STORY_ENTER_CHAPTER,
             UC.STORY_MENU, UC.STAGE_ENTER],
            conf=0.30,
        ) is not None

    def _pick_hub_card(self, screen: ScreenState) -> Optional[YoloBox]:
        """On the story hub, return the card cls for the current category
        (skipping exhausted ones)."""
        if not self.find_cls(screen, [UC.STORY_SHORT, UC.STORY_SIDE, UC.STORY_MAIN], conf=0.30):
            return None  # not on hub
        while self._cat_idx < len(self._categories):
            cat = self._categories[self._cat_idx]
            if cat in self._exhausted:
                self._cat_idx += 1
                continue
            card = self.find_cls(screen, cat, conf=0.30)
            if card is not None:
                return card
            # card cls not seen — try next category rather than stall
            self._cat_idx += 1
        return None

    # ── navigation (lobby → mission hub) ──────────────────────────────
    def _navigate(self, screen: ScreenState) -> Dict[str, Any]:
        self._nav_ticks += 1
        if self._nav_ticks > self._nav_limit:
            self.log("nav: can't reach story hub — giving up")
            return action_done("story mining unreachable")
        page = self.detect_screen_yolo(screen)
        if page == "Lobby":
            # Open the mission hub via the right-side 任务大厅入口 tile (YOLO
            # cls) — same entry arena/bounty/campaign_sweep use. NO hardcoded
            # position (universal across window size / desktop scaling).
            act = self.click_cls(
                screen, UC.NAV_TASKS,
                "story mining: open hub from lobby", conf=0.30,
            )
            if act:
                self._click_cooldown = 2
                return act
            return action_wait(400, "nav: 任务大厅入口 cls not seen yet")
        # Unknown / transitional — wait.
        return action_wait(400, "nav: waiting (no known page cls)")
