"""StoryMiningSkill — mine unplayed story chapters for pyroxene (pure-YOLO).

Verified flow (interactive probe 2026-06-01, data/_mining_probe_log.md). Each
cleared story node ≈ 80 pyroxene — alongside MomoTalk, the top free-pyroxene
source. Mines 主线 / 短篇 / 支线 (重播 = replay, no reward, skipped).

Three-level drill (main; short/side are flatter), all via reactive YOLO
clicking — YOLO gives exact coords, no grid math:
  篇 (main 卷 list): swipe LEFT to reveal newer 卷 → click a `new`/`剧情new`
       badge (vs `完成`) to SELECT it (chapters appear on the right).
  章 (chapter list): click the row with a 黄点 (DOT_YELLOW) = unplayed chapter.
  节点 (node list):  click the 入场键 paired (same row) with a 剧情图标未完成 /
       剧情new node (skip 入场键没解锁 = locked). → 进入章节 → scene.
  短篇/支线 grids: find a `new` card; none on this page → 右切换 to the next.

Skip flow (story auto-PLAYS — menu→skip ASAP): 剧情menu → 跳过故事键 → 确认键 →
获得奖励 (≈80💎) → 点击继续字样 → 剧情中断退出 (中断) to leave the node.

NO OCR. ⚠️ Main MAY contain battle nodes (no 剧情menu) — the cutscene timeout
backs out of those (user to verify; v6 may add battle-node handling).

Detectors: base "ui" only (SKILL_YOLO_MAP Story = base).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from brain.skills.base import (
    BaseSkill, ScreenState, YoloBox,
    action_click, action_click_box, action_back, action_done, action_wait,
    action_swipe,
)
from brain.skills import ui_classes as UC

_CLS_CONF = 0.30
_CONTENT_REGION = (0.05, 0.12, 0.99, 0.92)   # exclude top HUD + bottom nav
_NODE_PANEL = (0.30, 0.12, 1.0, 0.95)        # node/chapter list (right side)
_ROW_DY = 0.06                               # node-icon ↔ 入场键 same-row gap

_MAX_PAGE_TURNS = 6      # 右切换 paging cap per category (short/side grids)
_MAX_MAIN_SWIPES = 5     # 卷-list swipe-left cap (main)
_BARREN_LIMIT = 5        # empty scans before exhausting a category


class StoryMiningSkill(BaseSkill):
    def __init__(self) -> None:
        super().__init__("StoryMining")
        self.max_ticks = 1500
        # 主线 first (user: 先去主线), then 短篇 = 支线.
        self._categories = [UC.STORY_MAIN, UC.STORY_SHORT, UC.STORY_SIDE]
        self._init_state()

    def _init_state(self) -> None:
        self._cat_idx = 0
        self._current_cat: Optional[str] = None
        self._exhausted: List[str] = []
        self._barren = 0
        self._page_turns = 0
        self._main_swipes = 0
        self._nav_ticks = 0
        self._cut_ticks = 0
        self._cooldown = 0

    def reset(self) -> None:
        super().reset()
        self._init_state()

    # ── tick: reactive priority chain ─────────────────────────────────────
    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1
        if self.ticks >= self.max_ticks:
            self.log("timeout")
            return action_done("story mining timeout")

        if self._cooldown > 0:
            self._cooldown -= 1
            return action_wait(400, f"story settle ({self._cooldown})")

        # P0: splashes / reward.
        splash = self.find_cls(screen, [UC.BOND_LEVELUP, UC.REGION_LEVELUP], conf=_CLS_CONF)
        if splash is not None:
            return action_click(0.5, 0.5, f"dismiss splash ({splash.cls_name})")
        reward = self.find_cls(screen, UC.GOT_REWARD, conf=_CLS_CONF)
        if reward is not None:
            self._cut_ticks = 0
            return action_click_box(reward, "dismiss reward (≈80💎)")

        # P0.5: after reward, 中断 (剧情中断退出) leaves the node cleanly
        # (观看 = keep watching — ignore). Mining wants out.
        quit_node = self.find_cls(screen, UC.STORY_QUIT, conf=_CLS_CONF)
        if quit_node is not None:
            self._cut_ticks = 0
            self._cooldown = 2
            self._barren = 0
            return action_click_box(quit_node, "中断 — leave node after mining")

        # P1: cutscene skip chain (story auto-plays → skip ASAP).
        cut = self._handle_cutscene(screen)
        if cut is not None:
            return cut

        # P2: chapter-info modal → 进入章节.
        play = self.find_cls(screen, UC.STORY_ENTER_CHAPTER, conf=_CLS_CONF, region=(0.25, 0.50, 0.80, 0.92))
        if play is not None:
            self._barren = 0
            self._cooldown = 3
            return action_click_box(play, "enter chapter (进入章节)")

        # P3: MINE — drill deepest-first (node 入场键 > 黄点章 > new 篇/卡).
        mine = self._mine_action(screen)
        if mine is not None:
            self._barren = 0
            self._cooldown = 3
            return mine

        # P4: on a story page but nothing unplayed visible → reveal more, then
        # (if truly barren) exhaust the category and advance.
        page = self.detect_screen_yolo(screen)
        on_story = (page == "Story") or self._on_any_story_page(screen)
        if on_story:
            reveal = self._reveal_more(screen)
            if reveal is not None:
                return reveal
            self._barren += 1
            if self._barren <= _BARREN_LIMIT:
                return action_wait(350, f"scanning for unplayed ({self._barren})")
            return self._exhaust_and_advance(screen)

        # P5: story hub — open the current (non-exhausted) category card.
        hub_card = self._pick_hub_card(screen)
        if hub_card is not None:
            self._barren = 0
            self._page_turns = 0
            self._main_swipes = 0
            self._cooldown = 2
            return action_click_box(hub_card, f"open category ({hub_card.cls_name})")

        # P6: mission hub — 劇情 tile.
        story_tile = self.find_cls(screen, UC.HUB_STORY, conf=_CLS_CONF)
        if story_tile is not None:
            self._nav_ticks = 0
            self._cooldown = 2
            return action_click_box(story_tile, "enter 剧情 hub")

        # P7: navigation (lobby → mission hub).
        return self._navigate(screen)

    # ── cutscene skip ──────────────────────────────────────────────────────
    def _handle_cutscene(self, screen: ScreenState) -> Optional[Dict[str, Any]]:
        # After clicking skip/menu, the skip-confirm dialog appears → confirm.
        if self._cut_ticks > 0:
            confirm = self.find_cls(screen, UC.BTN_CONFIRM, conf=_CLS_CONF, region=(0.30, 0.55, 0.85, 0.85))
            if confirm is not None:
                self._cut_ticks = 0
                self._cooldown = 2
                return action_click_box(confirm, "confirm story skip (确认键)")
            cont = self.find_cls(screen, UC.STORY_TAP_CONTINUE, conf=_CLS_CONF)
            if cont is not None:
                self._cut_ticks = 0
                return action_click_box(cont, "tap continue")

        if self._cut_ticks < 3:
            skip = self.find_cls(screen, UC.STORY_SKIP, conf=0.40, region=(0.82, 0.05, 1.0, 0.30))
            if skip is not None:
                self._cut_ticks += 1
                return action_click_box(skip, "跳过故事键")
            menu = self.find_cls(screen, UC.STORY_MENU, conf=0.40, region=(0.82, 0.0, 1.0, 0.16))
            if menu is not None:
                self._cut_ticks += 1
                return action_click_box(menu, "open 剧情menu")
        cont = self.find_cls(screen, UC.STORY_TAP_CONTINUE, conf=_CLS_CONF)
        if cont is not None:
            return action_click_box(cont, "tap continue")
        if self._cut_ticks > 0:
            self._cut_ticks = max(0, self._cut_ticks - 1)
        return None

    # ── mining (drill deepest-first) ───────────────────────────────────────
    def _mine_action(self, screen: ScreenState) -> Optional[Dict[str, Any]]:
        # ★ Mine ONLY on a real story page. A 黄点 (DOT_YELLOW) on the LOBBY / 任务
        # 大厅 is a nav badge (student / campaign_nav / cafe), NOT an unplayed
        # chapter — mining off the lobby clicked the campaign-nav badge and landed
        # on 任务关卡 instead of 剧情. No story cls on screen ⇒ defer to navigation
        # (P5/P6/P7) instead of clicking a stray dot.
        if not self._on_any_story_page(screen):
            return None

        # 1) NODE level: enter the 入场键 paired (same row) with an unplayed
        #    node (剧情图标未完成 / 剧情new). Skip 入场键没解锁 (locked).
        undone = self.find_all_cls(screen, [UC.STORY_ICON_UNDONE, UC.STORY_NEW],
                                   conf=_CLS_CONF, region=_NODE_PANEL)
        enters = self.find_all_cls(screen, UC.STAGE_ENTER, conf=_CLS_CONF, region=_NODE_PANEL)
        if undone and enters:
            for nd in sorted(undone, key=lambda b: b.cy):
                row_enter = min(enters, key=lambda e: abs(e.cy - nd.cy))
                if abs(row_enter.cy - nd.cy) < _ROW_DY:
                    return action_click_box(row_enter, "enter unplayed node (入场键)")

        # 2) CHAPTER level: a 黄点 in the content area = unplayed chapter → click
        #    its row (use the dot's y; click toward the row center-left).
        dot = self._content_yellow_dot(screen)
        if dot is not None:
            row_x = min(0.88, max(0.55, dot.cx - 0.05))
            return action_click(row_x, dot.cy, "open unplayed chapter (黄点 row)")

        # 3) 篇/CARD level: a `new` / `剧情new` badge → click to select/enter.
        new = self.find_cls(screen, [UC.NEW_MARK, UC.STORY_NEW], conf=_CLS_CONF, region=_CONTENT_REGION)
        if new is not None:
            return action_click_box(new, "select new 篇 / enter new card")
        return None

    def _content_yellow_dot(self, screen: ScreenState) -> Optional[YoloBox]:
        dots = self.find_all_cls(screen, UC.DOT_YELLOW, conf=0.40, region=_CONTENT_REGION)
        # exclude dots that sit on a category/nav tile (very top) — keep node area.
        dots = [d for d in dots if d.cy > 0.18]
        return min(dots, key=lambda b: b.cy) if dots else None

    # ── reveal more (page-turn grids / swipe-left main 卷 list) ─────────────
    def _reveal_more(self, screen: ScreenState) -> Optional[Dict[str, Any]]:
        # Short/side grids paginate with 右切换.
        arrow = self.find_cls(screen, UC.ARROW_RIGHT, conf=_CLS_CONF, region=(0.85, 0.30, 1.0, 0.70))
        if arrow is not None and self._page_turns < _MAX_PAGE_TURNS:
            self._page_turns += 1
            self._barren = 0
            self._cooldown = 2
            self.log(f"右切换 next page ({self._page_turns}/{_MAX_PAGE_TURNS})")
            return action_click_box(arrow, "next page (find new card)")
        # Main 卷 list scrolls horizontally — swipe LEFT to reveal newer 卷.
        if self._current_cat == UC.STORY_MAIN and self._main_swipes < _MAX_MAIN_SWIPES:
            self._main_swipes += 1
            self._barren = 0
            self._cooldown = 2
            self.log(f"swipe-left 卷 list ({self._main_swipes}/{_MAX_MAIN_SWIPES})")
            return action_swipe(0.65, 0.42, 0.30, 0.42, 500, "reveal newer 卷")
        return None

    def _exhaust_and_advance(self, screen: ScreenState) -> Dict[str, Any]:
        cur = self._categories[self._cat_idx] if self._cat_idx < len(self._categories) else None
        if cur and cur not in self._exhausted:
            self._exhausted.append(cur)
            self.log(f"category {cur} exhausted")
        self._barren = 0
        self._page_turns = 0
        self._main_swipes = 0
        self._cat_idx += 1
        if self._cat_idx >= len(self._categories):
            return action_done("story mining finished (all categories)")
        back = self.find_cls(screen, [UC.BTN_BACK, UC.BTN_HOME], conf=_CLS_CONF)
        self._cooldown = 2
        return (action_click_box(back, "back to hub for next category")
                if back else action_back("back to hub for next category"))

    # ── helpers ──────────────────────────────────────────────────────────
    def _on_any_story_page(self, screen: ScreenState) -> bool:
        return self.find_cls(
            screen,
            [UC.STORY_SHORT, UC.STORY_SIDE, UC.STORY_MAIN,
             UC.STORY_ICON_DONE, UC.STORY_ICON_UNDONE, UC.STORY_ENTER_CHAPTER,
             UC.STORY_MENU, UC.STAGE_ENTER, UC.NEW_MARK, UC.STORY_NEW, UC.NODE_DONE],
            conf=_CLS_CONF,
        ) is not None

    def _pick_hub_card(self, screen: ScreenState) -> Optional[YoloBox]:
        if not self.find_cls(screen, [UC.STORY_SHORT, UC.STORY_SIDE, UC.STORY_MAIN], conf=_CLS_CONF):
            return None
        while self._cat_idx < len(self._categories):
            cat = self._categories[self._cat_idx]
            if cat in self._exhausted:
                self._cat_idx += 1
                continue
            card = self.find_cls(screen, cat, conf=_CLS_CONF)
            if card is not None:
                self._current_cat = cat
                return card
            self._cat_idx += 1
        return None

    def _navigate(self, screen: ScreenState) -> Dict[str, Any]:
        self._nav_ticks += 1
        if self._nav_ticks > 40:
            self.log("nav: can't reach story hub")
            return action_done("story mining unreachable")
        if self.detect_screen_yolo(screen) == "Lobby":
            act = self.click_cls(screen, UC.NAV_TASKS, "open hub from lobby", conf=_CLS_CONF)
            if act is not None:
                self._cooldown = 2
                return act
            return action_wait(400, "nav: 任务大厅入口 not seen")
        return action_wait(400, "nav: waiting (no known page)")
