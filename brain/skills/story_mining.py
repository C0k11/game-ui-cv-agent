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
_RESULT_BAND = (0.32, 0.55, 0.68, 0.85)  # centered battle-result 确认键 band
_FIGHT_HOLD = 120        # ticks to hold for a story auto-battle (~2min)


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
        self._tried_enters: List[tuple] = []   # node 入场键 positions already entered
                                               # (battle nodes we back out of → skip next)
        self._tried_chapters: List[float] = [] # chapter 黄点 cy already opened (a
                                               # battle-only chapter keeps its dot →
                                               # don't reopen it forever)
        self._tried_cards: List[tuple] = []    # new-篇/卡 positions already selected
                                               # (a battle-gated 篇 keeps its New badge
                                               # forever AND selecting it resets barren
                                               # → infinite category loop without this)
        self._back_streak: int = 0             # consecutive back-outs with nothing
                                               # mined (→ next category when high)
        self._card_misses: int = 0             # consecutive hub frames missing the
                                               # next category's card cls (3 → skip;
                                               # 1-frame flicker must NOT skip a cat)
        self._fighting: int = 0                # >0 = story battle in progress (hold
                                               # ticks; battle frames carry no known
                                               # ui cls → would otherwise nav-lose)
        self._cat_opened_tick: int = -99       # tick when current category was opened
                                               # (hub-re-reach exhaust needs >6 ticks
                                               # gap; transition lag false-exhausted
                                               # 主線 right after opening it)

    def reset(self) -> None:
        super().reset()
        self._init_state()

    # ── tick: reactive priority chain ─────────────────────────────────────
    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1
        if self.ticks >= self.max_ticks:
            self.log("timeout")
            return action_done("story mining timeout")

        # ★ Decision inputs from a CLEAN ADB frame (overlay burn defense).
        # Live 2026-06-10: the tick's box feed lost BOTH 剧情图标未完成 icons
        # and the battle-node 入场键 (which detect at 0.98 on the same screen's
        # clean frame) → _mine_action saw nothing → bogus reveal-swipes. This
        # skill is fully detection-driven, so re-detect every tick on the
        # overlay-free frame (~250ms, fine for turn-based mining).
        try:
            from brain.pipeline import get_clean_frame, _run_yolo_on_image
            fr = get_clean_frame()
            if fr is not None:
                h, w = fr.shape[:2]
                screen.frame = fr
                screen.yolo_boxes = _run_yolo_on_image(fr, w, h, context="ui")
                screen.image_w, screen.image_h = w, h
        except Exception:
            pass

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

        # P0.5: 下一章節 prompt after an episode. ★ 觀看 chains STRAIGHT into
        # the next episode (user 2026-06-10: battle win/lose are both scripted
        # story; 連看連挖 beats 中断+re-navigation). 中断 only as fallback when
        # the 观看 cls (12f weak) misses — the node-list scan re-enters then.
        watch = self.find_cls(screen, UC.STORY_WATCH, conf=_CLS_CONF)
        if watch is not None:
            self._cut_ticks = 0
            self._cooldown = 2
            self._barren = 0
            return action_click_box(watch, "觀看 — chain next episode")
        quit_node = self.find_cls(screen, UC.STORY_QUIT, conf=_CLS_CONF)
        if quit_node is not None:
            self._cut_ticks = 0
            self._cooldown = 2
            self._barren = 0
            return action_click_box(quit_node, "中断 — leave node (觀看 cls missed)")

        # P0.6: battle result (戰鬥結果) — a centered cancel-less 确认键. The
        # confirm+cancel-both-visible case = a COST dialog → never click confirm
        # (arena C2 lesson); story flows have no legit cost dialogs, but keep
        # the negative gate anyway.
        res_confirm = self.find_cls(screen, UC.BTN_CONFIRM, conf=_CLS_CONF, region=_RESULT_BAND)
        if res_confirm is None:
            # Story "Battle Complete" puts its 確認 at the BOTTOM-RIGHT
            # (live 2026-06-10: (0.89,0.91) — outside the centered arena-style
            # band; the hold ran out staring at a finished battle).
            res_confirm = self.find_cls(screen, UC.BTN_CONFIRM, conf=_CLS_CONF,
                                        region=(0.78, 0.78, 1.0, 0.98))
        if res_confirm is not None:
            if self.find_cls(screen, UC.BTN_CANCEL, conf=0.20) is None:
                self._fighting = 0
                self._cooldown = 2
                return action_click_box(res_confirm, "dismiss battle/result dialog (确认键)")
            # confirm+cancel together = the story SKIP-CONFIRM dialog (是否略過
            # 此劇情? — the only confirm+cancel dialog in the story flow). The
            # old "wait and re-read" here deadlocked against the decayed
            # _cut_ticks gate in the cutscene handler (live 2026-06-10: stuck
            # 20 ticks staring at it). Fall through — P1 handles it.

        # P0.7: BATTLE node — a story node's 部队/出击 squad screen. ★ Story
        # battles cost NO AP (user 2026-06-10) and the account trivially
        # out-levels them — FIGHT (arena-style): 出击 → auto-battle → result
        # dialog handled by P0.6. Completing the battle is what clears the
        # node/chapter dot (backing out left the mine permanently blocked).
        if self.find_cls(screen, [UC.SORTIE, UC.SQUAD_1, UC.SQUAD_1_HI],
                         conf=_CLS_CONF) is not None:
            sortie = self.find_cls(screen, UC.SORTIE, conf=_CLS_CONF)
            if sortie is not None:
                self._cut_ticks = 0
                self._barren = 0
                self._fighting = _FIGHT_HOLD
                self._cooldown = 3
                self.log("story battle node → 出击 (free, no AP)")
                return action_click_box(sortie, "story battle 出击 (no AP)")
            return action_wait(500, "squad screen — waiting for 出击")

        # P0.8: battle in progress — battle frames carry no known ui cls; hold
        # instead of nav-wandering. Result/reward popups are caught by P0/P0.6
        # above; a post-battle story resume (menu/skip/continue cls) releases
        # the hold so the P1 skip chain takes over.
        if self._fighting > 0:
            if (self._on_any_story_page(screen)
                    or self.find_cls(screen, [UC.STORY_SKIP, UC.STORY_TAP_CONTINUE],
                                     conf=0.30) is not None):
                self._fighting = 0
            else:
                self._fighting -= 1
                if self._fighting == 0:
                    self.log("battle hold expired → back out")
                    return action_back("battle hold expired")
                return action_wait(1000, f"story battle in progress ({self._fighting})")

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
            self._back_streak = 0
            self._cooldown = 3
            return mine

        # P3.5: on the 剧情 hub CATEGORY page → open the current category card to
        # drill IN. MUST run before the barren scan: the category page carries
        # category cls but NO chapters, so scanning it just exhausts the whole
        # category without ever entering it (the bug that "exhausted" all three).
        hub_card = self._pick_hub_card(screen)
        if hub_card is not None:
            self._barren = 0
            self._page_turns = 0
            self._main_swipes = 0
            self._tried_cards = []
            self._cooldown = 2
            return action_click_box(hub_card, f"open category ({hub_card.cls_name})")
        # All categories exhausted/skipped → finish cleanly (the old finish in
        # _exhaust_and_advance is unreachable from the hub fast-exhaust path —
        # live 2026-06-10 the skill instead wandered into "nav: can't reach").
        if self._cat_idx >= len(self._categories):
            self.log(f"all categories done ({len(self._exhausted)} exhausted)")
            return action_done("story mining finished (all categories)")

        # P4: INSIDE a category but nothing unplayed visible → reveal more, then
        # (if truly barren) exhaust the category and advance.
        page = self.detect_screen_yolo(screen)
        on_story = (page == "Story") or self._on_any_story_page(screen)
        # ★ Short/side card GRID pages carry NO trained page cls when no New
        # card is visible (ui v8 backlog) — but their 右切换 pagination arrow
        # detects fine (0.91). If we just opened a grid category and see the
        # arrow on an otherwise-unknown page, treat it as in-category so
        # _reveal_more pages toward the New card instead of nav-losing.
        if (not on_story and page is None
                and self._current_cat in (UC.STORY_SHORT, UC.STORY_SIDE)
                and self.find_cls(screen, UC.ARROW_RIGHT, conf=_CLS_CONF,
                                  region=(0.85, 0.30, 1.0, 0.70)) is not None):
            on_story = True
        if on_story:
            reveal = self._reveal_more(screen)
            if reveal is not None:
                return reveal
            self._barren += 1
            if self._barren <= _BARREN_LIMIT:
                return action_wait(350, f"scanning for unplayed ({self._barren})")
            # Nothing minable here (e.g. a chapter with only battle/locked/done
            # nodes). Back OUT one level to find the next unplayed chapter/篇 —
            # NOT exhaust the whole category. Only give up to the next category
            # after backing out many times with nothing mined.
            self._barren = 0
            self._back_streak += 1
            if self._back_streak > 8:
                self._back_streak = 0
                return self._exhaust_and_advance(screen)
            self._cooldown = 2
            back = self.find_cls(screen, [UC.BTN_BACK], conf=_CLS_CONF)
            return (action_click_box(back, "back out → next unplayed chapter/篇")
                    if back else action_back("back out → next unplayed chapter/篇"))

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
        # Skip-confirm dialog (是否略過此劇情? 取消/確認) → 確認. Recognized by
        # confirm-in-band + story chrome (MENU/skip top-right stays rendered
        # behind the dialog) — NOT only by _cut_ticks: that counter decays one
        # per empty frame, so a 2-frame cls flicker disabled the branch and
        # deadlocked on the dialog (live 2026-06-10).
        story_chrome = self.find_cls(screen, [UC.STORY_MENU, UC.STORY_SKIP],
                                     conf=0.30, region=(0.80, 0.0, 1.0, 0.30)) is not None
        if self._cut_ticks > 0 or story_chrome:
            confirm = self.find_cls(screen, UC.BTN_CONFIRM, conf=_CLS_CONF, region=(0.30, 0.55, 0.85, 0.85))
            if confirm is not None:
                self._cut_ticks = 0
                self._cooldown = 2
                return action_click_box(confirm, "confirm story skip (确认键)")
        if self._cut_ticks > 0:
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
                    epos = (round(row_enter.cx, 2), round(row_enter.cy, 2))
                    # Skip nodes we already entered: a battle node we backed out
                    # of stays "unplayed", so without this we'd re-enter it
                    # forever (battle → back → battle …).
                    if any(abs(px - epos[0]) < 0.04 and abs(py - epos[1]) < 0.04
                           for px, py in self._tried_enters):
                        continue
                    self._tried_enters.append(epos)
                    return action_click_box(row_enter, "enter unplayed node (入场键)")

        # 2) CHAPTER level: a 黄点 in the content area = unplayed chapter → click
        #    its row (use the dot's y; click toward the row center-left).
        dot = self._content_yellow_dot(screen)
        if dot is not None:
            self._tried_chapters.append(dot.cy)   # opened → don't reopen (battle-only)
            self._tried_enters = []   # new chapter → reset node-dedup
            # The 黄点 sits at the LEFT edge of the chapter row; the clickable
            # chapter TITLE is to its RIGHT (e.g. 第2章 與往日訣別). Clicking
            # dot.cx-0.05 landed LEFT of the chapter panel (on the 篇 area) and
            # never opened it (live: clicked 0.685 forever). Click into the title
            # to the right of the dot.
            row_x = min(0.93, dot.cx + 0.10)
            return action_click(row_x, dot.cy, "open unplayed chapter (right of 黄点)")

        # 3) 篇/CARD level: a `new` badge → select/enter. ONLY when NO node-level
        #    入场键 is present. A New badge on a NODE (新节点, e.g. 巢穴 New) is
        #    NOT a 篇/卡 entry — clicking it does nothing and loops forever. New
        #    means "open this 篇/card" only on the 篇/grid screens (no 入场键 there).
        #    ★ Dedup by position: a battle-gated 篇 keeps its New badge forever
        #    AND selecting it resets barren/back_streak → infinite category loop
        #    (live 2026-06-10: 卷6 re-selected endlessly). Tried positions reset
        #    on swipe/page-turn (cards shift), bounded by the swipe caps.
        if self.find_cls(screen, UC.STAGE_ENTER, conf=_CLS_CONF, region=_NODE_PANEL) is None:
            news = self.find_all_cls(screen, [UC.NEW_MARK, UC.STORY_NEW],
                                     conf=_CLS_CONF, region=_CONTENT_REGION)
            for new in sorted(news, key=lambda b: (b.cy, b.cx)):
                npos = (round(new.cx, 2), round(new.cy, 2))
                if any(abs(px - npos[0]) < 0.05 and abs(py - npos[1]) < 0.05
                       for px, py in self._tried_cards):
                    continue
                self._tried_cards.append(npos)
                return action_click_box(new, "select new 篇 / enter new card")
        return None

    def _content_yellow_dot(self, screen: ScreenState) -> Optional[YoloBox]:
        dots = self.find_all_cls(screen, UC.DOT_YELLOW, conf=0.40, region=_CONTENT_REGION)
        # exclude top tiles (cy<0.18) AND chapters already opened (a battle-only
        # chapter keeps showing its 黄点 → skip so we don't reopen it forever).
        dots = [d for d in dots if d.cy > 0.18
                and not any(abs(c - d.cy) < 0.05 for c in self._tried_chapters)]
        return min(dots, key=lambda b: b.cy) if dots else None

    # ── reveal more (page-turn grids / swipe-left main 卷 list) ─────────────
    def _reveal_more(self, screen: ScreenState) -> Optional[Dict[str, Any]]:
        # Short/side grids paginate with 右切换.
        arrow = self.find_cls(screen, UC.ARROW_RIGHT, conf=_CLS_CONF, region=(0.85, 0.30, 1.0, 0.70))
        if arrow is not None and self._page_turns < _MAX_PAGE_TURNS:
            self._page_turns += 1
            self._barren = 0
            self._cooldown = 2
            self._tried_cards = []   # cards shift on page turn
            self.log(f"右切换 next page ({self._page_turns}/{_MAX_PAGE_TURNS})")
            return action_click_box(arrow, "next page (find new card)")
        # Main 卷 list scrolls horizontally — swipe LEFT to reveal newer 卷.
        if self._current_cat == UC.STORY_MAIN and self._main_swipes < _MAX_MAIN_SWIPES:
            self._main_swipes += 1
            self._barren = 0
            self._cooldown = 2
            self._tried_cards = []   # cards shift on swipe
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
        self._tried_cards = []
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

    def _card_has_mine_dot(self, screen: ScreenState, card: YoloBox) -> bool:
        """黄点 near the card's TOP-RIGHT = this category has mines (user
        2026-06-10: 通过黄点识别有没有矿, 没点的类别根本不进).

        Geometry (live-measured 2026-06-10, story_hub.png): the category cls
        box is the TITLE TEXT near the card BOTTOM, while the dot sits at the
        card's TOP edge (y≈0.165) — 0.35-0.55 ABOVE the title. Search the
        vertical strip above the title: x within (title.x1, title.x2+0.06),
        y from 0.10 down to the title top. Measured: 主線 title(0.31-0.43,
        y0.72) dot(0.445,0.165) ✓; 短篇 title(0.57-0.65,y0.52) dot(0.667,
        0.167) ✓; strips don't cross-talk (0.486<0.662, 0.574>0.450)."""
        region = (card.x1, 0.10, min(1.0, card.x2 + 0.06), max(0.12, card.y1))
        return self.find_cls(screen, UC.DOT_YELLOW, conf=0.35, region=region) is not None

    def _pick_hub_card(self, screen: ScreenState) -> Optional[YoloBox]:
        # ★ The hub CATEGORY page shows ALL category cards; inside a category
        # only its own header cls shows. Require ≥2 distinct category cls to
        # call this the hub page — otherwise we're inside one (let P4 scan).
        present = [c for c in (UC.STORY_MAIN, UC.STORY_SHORT, UC.STORY_SIDE)
                   if self.find_cls(screen, c, conf=_CLS_CONF) is not None]
        if len(present) < 2:
            return None
        while self._cat_idx < len(self._categories):
            cat = self._categories[self._cat_idx]
            if cat in self._exhausted:
                self._cat_idx += 1
                continue
            if cat == self._current_cat:
                # Back ON the hub page after drilling this category ⇒ its
                # mineable content is done (battle-gated nodes keep the dot/New
                # but can't be mined) — exhaust and advance NOW. The old
                # "return None → P4 barren scan" churned: hub scan → back out
                # to 任务大厅 → re-enter → scan … ×8 before exhausting (live
                # 2026-06-10, ~20s per lap).
                # ★ But NOT right after opening it: the open-click's transition
                # lags 2-3 ticks with the hub still on screen — that false-
                # exhausted 主線 the moment it was opened (live 2026-06-10).
                if self.ticks - self._cat_opened_tick <= 6:
                    return None   # transition settling — re-read next tick
                self.log(f"category {cat}: hub re-reached after drill → exhausted")
                self._exhausted.append(cat)
                self._current_cat = None
                self._cat_idx += 1
                continue
            card = self.find_cls(screen, cat, conf=_CLS_CONF)
            if card is not None:
                self._card_misses = 0
                # ★ Signal-driven category gate: no 黄点 on the card = nothing
                # to mine inside — skip without entering (e.g. 支線 today).
                if not self._card_has_mine_dot(screen, card):
                    self.log(f"category {cat}: 无黄点 → no mine, skip")
                    self._exhausted.append(cat)
                    self._cat_idx += 1
                    continue
                self._current_cat = cat
                self._cat_opened_tick = self.ticks
                return card
            # Card cls not seen THIS frame. A 1-frame flicker must not skip the
            # whole category (live 2026-06-10: 短篇 flickered out right after
            # 主線 exhausted → its mine was skipped entirely). Retry a few hub
            # frames before giving up on it.
            self._card_misses += 1
            if self._card_misses < 3:
                return None
            self._card_misses = 0
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
