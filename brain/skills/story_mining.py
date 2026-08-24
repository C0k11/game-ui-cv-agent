"""StoryMiningSkill — mine unplayed story chapters for pyroxene (pure-YOLO).

Verified flow (interactive probe 2026-06-01, data/_mining_probe_log.md). Each
cleared story node ≈ 80 pyroxene — alongside MomoTalk, the top free-pyroxene
source. Mines 主线 / 短篇 / 支线 (重播 = replay, no reward, skipped).

Three-level drill (main; short/side are flatter), all via reactive YOLO
clicking — YOLO gives exact coords, no grid math:
  篇 (main 卷 list): swipe LEFT to reveal newer 卷  click a `new`/`剧情new`
       badge (vs `完成`) to SELECT it (chapters appear on the right).
  章 (chapter list): click the row with a 黄点 (DOT_YELLOW) = unplayed chapter.
  节点 (node list):  click the 入场键 paired (same row) with a 剧情图标未完成 /
       剧情new node (skip 入场键没解锁 = locked).  进入章节  scene.
  短篇/支线 grids: find a `new` card; none on this page  右切换 to the next.

Skip flow (story auto-PLAYS — menuskip ASAP): 剧情menu  跳过故事键  确认键
获得奖励 (≈80)  点击继续字样  剧情中断退出 (中断) to leave the node.

NO OCR. ️ Main MAY contain battle nodes (no 剧情menu) — the cutscene timeout
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
_ROW_DY = 0.06                               # node-icon <-> 入场键 same-row gap

_MAX_PAGE_TURNS = 6      # 右切换 paging cap per category (short/side grids)
_MAX_MAIN_SWIPES = 5     # 卷-list swipe-left cap (main)
# 2026-07-28 全文件墙钟化(tick-vs-wallclock 家族, 见 memory)。
# 旧值全是 **tick**; zero-wait 后 0.15-0.25 s/tick, 相对 1.6 s/tick 年代缩水 6.4-10 倍。
# 本文件是**受灾最重的一个**(静态审计):
#   · `_cooldown = 2/3` **出现 15 处** —— 每一次页面跳转/翻页/滑动后的稳定等待,
#     3.2-4.8s  **0.5-0.75s**。剧情各级列表(篇/章/节点)的转场都在 1-3s 量级,
#     0.5s 的"稳定"等于没等  下一 tick 读到的是**转场中的半张屏**:
#     该有的 cls 还没渲染  走进 barren 分支  误判"这页没矿" 往回退一级。
#   · `max_ticks = 1500`  **~6min** 跑完三个类别(旧 ~40min)  挖到一半静默超时。
#   · `_nav_ticks > 40`  **10s** 就报 "can't reach story hub"(旧 64s)。
#   · `_BARREN_LIMIT = 5`  **~1.3s** 就判一页没矿(旧 ~8s)。
# 换算口径: 旧注释都写于 ~1.6s/tick 年代, 一律 ×1.6 还原成秒, **绝不趁机收窄**
# (那些数当初就是为修事故调大的)。
_SETTLE_SHORT = 3.2      # 旧 _cooldown=2
_SETTLE_LONG = 4.8       # 旧 _cooldown=3
_BARREN_SEC = 8.0        # 旧 _BARREN_LIMIT=5 empty scans
_NAV_MAX_SEC = 64.0      # 旧 _nav_ticks>40
_SKILL_BUDGET_SEC = 2400.0   # 旧 max_ticks=1500 (三个类别全挖完)
# after-ack 窗口必须**同时覆盖自主跑(0.25s/tick)与 step 门控(步间隔 4-6s)**两种
# 节奏 —— momo_talk 那边第一版写 2.0s, 在 step_walk 下必然过期, 照样连发 5 次。
# 放宽近乎零代价(框 1-2s 就渲染, 上面的 confirm 分支立刻接走)。
_SKIP_ACK_SEC = 6.0      # 跳过故事键/menu  「是否略過」确认框渲染(after-ack)
_BARREN_LIMIT = 5        # 仅留作日志计数(判据已改墙钟 _BARREN_SEC)
_RESULT_BAND = (0.32, 0.55, 0.68, 0.85)  # centered battle-result 确认键 band
# 2026-07-25 墙钟化: 旧值 `_FIGHT_HOLD = 120` **ticks**, 注释自称 "~2min" ——
# 那是 ~1.6s/tick 年代的账。zero-wait 后自主跑实测 0.15-0.25 s/tick(口径见
# BaseSkill.mark 注释)  真实只有 **18-30s**, 而剧情自动战斗常跑 1.5-3 分钟
#  超时那一下是 `action_back` = **在战斗里盲按返回**(开暂停菜单, 状态机随后
# 在暂停菜单上乱走)。改墙钟 240s: 战斗真结束时靠结算框/剧情 cls 立刻释放
# (下面 P0.6/P1), 这个数只是"什么 cls 都认不出"的兜底上限 —— 放宽近乎零代价,
# 放窄却直接毁掉一个节点的进度。
_FIGHT_HOLD_SEC = 240.0


class StoryMiningSkill(BaseSkill):
    def __init__(self) -> None:
        super().__init__("StoryMining")
        # max_ticks 只当**跑飞兜底**(不再是有效上限); 真上限走 _SKILL_BUDGET_SEC。
        self.max_ticks = 20000
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
        self._nav_t0 = None            # 导航阶段墙钟起点(见 _NAV_MAX_SEC)
        self._skill_t0 = None          # 整个 skill 墙钟起点(见 _SKILL_BUDGET_SEC)
        self._cut_ticks = 0
        self._skipped_stories = 0      # 跳完的剧情段数(竣工判据用)
        self._no_dot_skips = 0         # 因"卡上无黄点=没矿"被跳过的类别数
        self._settle_sec = 0.0         # 当前这次转场还要等多少秒(墙钟)
        self._tried_enters: List[tuple] = []   # node 入场键 positions already entered
                                               # (battle nodes we back out of  skip next)
        self._tried_chapters: List[float] = [] # chapter 黄点 cy already opened (a
                                               # battle-only chapter keeps its dot
                                               # don't reopen it forever)
        self._tried_cards: List[tuple] = []    # new-篇/卡 positions already selected
                                               # (a battle-gated 篇 keeps its New badge
                                               # forever AND selecting it resets barren
                                               #  infinite category loop without this)
        self._back_streak: int = 0             # consecutive back-outs with nothing
                                               # mined ( next category when high)
        self._card_misses: int = 0             # consecutive hub frames missing the
                                               # next category's card cls (3  skip;
                                               # 1-frame flicker must NOT skip a cat)
        self._fighting: bool = False           # True = story battle in progress
                                               # (battle frames carry no known ui cls
                                               #  would otherwise nav-lose). 到期靠
                                               # 墙钟 timer "fight" 判, 不再数 tick。
        self._cat_opened_tick: int = -99       # tick when current category was opened
                                               # (hub-re-reach exhaust needs >6 ticks
                                               # gap; transition lag false-exhausted
                                               # 主線 right after opening it)

    def reset(self) -> None:
        super().reset()
        self._init_state()

    def exit_report(self):
        """竣工判据 —— 此前报 UNKNOWN, 于是"跑了 6 分钟一个节点都没挖到"这种
        空跑没人审计得出来(而它正是 max_ticks 缩水后的典型结局)。

        2026-07-28 live 当场修的**我自己的误报**: 首跑时三个类别全部
        「无黄点  no mine, skip」, 帧证实剧情 hub 上主線/短篇/支線/重播四张卡
        **一个黄点都没有**(该账号剧情已挖完) —— 这是 **没活可干(CLEAN)**,
        而第一版把它报成 LEFTOVER「一个节点都没进」。
         必须区分两种 exhausted:
           · 「无黄点」= 信号说没矿  CLEAN
           · 「进去过又退回 hub」= 有矿挖不动(战斗门/漏检)  值得关注
        """
        if self._skipped_stories:
            return ("CLEAN", f"跳完 {self._skipped_stories} 段剧情"
                             f"(类别 exhausted={self._exhausted or '-'}, "
                             f"进过 {len(self._tried_enters)} 个节点)")
        if self._tried_enters or self._tried_chapters:
            return ("LEFTOVER",
                    f"进了 {len(self._tried_enters)} 节点/"
                    f"{len(self._tried_chapters)} 章但**一段剧情都没跳完** — "
                    f"多半卡在战斗节点或跳过链")
        if self._no_dot_skips and self._no_dot_skips >= len(self._categories):
            return ("CLEAN",
                    f"{self._no_dot_skips} 个类别全部**无黄点 = 没矿可挖**"
                    f"(不是没干活) — 剧情已挖完, 等新剧情/新学生")
        return ("LEFTOVER",
                f"一个节点都没进(exhausted={self._exhausted or '-'}, "
                f"无黄点跳过 {self._no_dot_skips}/{len(self._categories)}) — "
                f"查 黄点/new 徽章 与 入场键 是否检出")

    def _settle(self, sec: float) -> None:
        """转场后等 sec 秒再读屏 —— 墙钟, 不是 tick(见文件头换算口径)。"""
        self._settle_sec = sec
        self.mark("settle")

    #  tick: reactive priority chain
    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1
        if self._skill_t0 is None:
            self._skill_t0 = self.clock()
        _run = self.clock() - self._skill_t0
        if _run >= _SKILL_BUDGET_SEC or self.ticks >= self.max_ticks:
            self.log(f"timeout ({_run:.0f}s / {self.ticks} tick, "
                     f"exhausted={self._exhausted})")
            return action_done("story mining timeout")

        #  Decision inputs from a CLEAN ADB frame (overlay burn defense).
        # Live 2026-06-10: the tick's box feed lost BOTH 剧情图标未完成 icons
        # and the battle-node 入场键 (which detect at 0.98 on the same screen's
        # clean frame)  _mine_action saw nothing  bogus reveal-swipes. This
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

        if self._settle_sec > 0.0:
            _sw = self.since("settle")
            if _sw < self._settle_sec:
                return action_wait(
                    400, f"story settle ({_sw:.1f}/{self._settle_sec:.1f}s)")
            self._settle_sec = 0.0

        # P0: splashes / reward.
        splash = self.find_cls(screen, [UC.BOND_LEVELUP, UC.REGION_LEVELUP], conf=_CLS_CONF)
        if splash is not None:
            return action_click(0.5, 0.5, f"dismiss splash ({splash.cls_name})")
        reward = self.find_cls(screen, UC.GOT_REWARD, conf=_CLS_CONF)
        if reward is not None:
            self._cut_ticks = 0
            return action_click_box(reward, "dismiss reward (≈80)")

        # P0.5: 下一章節 prompt after an episode.  觀看 chains STRAIGHT into
        # the next episode (user 2026-06-10: battle win/lose are both scripted
        # story; 連看連挖 beats 中断+re-navigation). 中断 only as fallback when
        # the 观看 cls (12f weak) misses — the node-list scan re-enters then.
        watch = self.find_cls(screen, UC.STORY_WATCH, conf=_CLS_CONF)
        if watch is not None:
            self._cut_ticks = 0
            self._settle(_SETTLE_SHORT)
            self._barren = 0
            return action_click_box(watch, "觀看 — chain next episode")
        quit_node = self.find_cls(screen, UC.STORY_QUIT, conf=_CLS_CONF)
        if quit_node is not None:
            self._cut_ticks = 0
            self._settle(_SETTLE_SHORT)
            self._barren = 0
            return action_click_box(quit_node, "中断 — leave node (觀看 cls missed)")

        # P0.6: battle result (戰鬥結果) — a centered cancel-less 确认键. The
        # confirm+cancel-both-visible case = a COST dialog  never click confirm
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
                self._fighting = False
                self._settle(_SETTLE_SHORT)
                return action_click_box(res_confirm, "dismiss battle/result dialog (确认键)")
            # confirm+cancel together = the story SKIP-CONFIRM dialog (是否略過
            # 此劇情? — the only confirm+cancel dialog in the story flow). The
            # old "wait and re-read" here deadlocked against the decayed
            # _cut_ticks gate in the cutscene handler (live 2026-06-10: stuck
            # 20 ticks staring at it). Fall through — P1 handles it.

        # P0.7: BATTLE node — a story node's 部队/出击 squad screen.  Story
        # battles cost NO AP (user 2026-06-10) and the account trivially
        # out-levels them — FIGHT (arena-style): 出击  auto-battle  result
        # dialog handled by P0.6. Completing the battle is what clears the
        # node/chapter dot (backing out left the mine permanently blocked).
        if self.find_cls(screen, [UC.SORTIE, UC.SQUAD_1, UC.SQUAD_1_HI],
                         conf=_CLS_CONF) is not None:
            sortie = self.find_cls(screen, UC.SORTIE, conf=_CLS_CONF)
            if sortie is not None:
                self._cut_ticks = 0
                self._barren = 0
                self._fighting = True
                self.mark("fight")
                self._settle(_SETTLE_LONG)
                self.log("story battle node  出击 (free, no AP)")
                return action_click_box(sortie, "story battle 出击 (no AP)")
            return action_wait(500, "squad screen — waiting for 出击")

        # P0.8: battle in progress — battle frames carry no known ui cls; hold
        # instead of nav-wandering. Result/reward popups are caught by P0/P0.6
        # above; a post-battle story resume (menu/skip/continue cls) releases
        # the hold so the P1 skip chain takes over.
        if self._fighting:
            if (self._on_any_story_page(screen)
                    or self.find_cls(screen, [UC.STORY_SKIP, UC.STORY_TAP_CONTINUE],
                                     conf=0.30) is not None):
                self._fighting = False
                self.clear_timer("fight")
            else:
                _held = self.since("fight")
                if _held >= _FIGHT_HOLD_SEC:
                    self._fighting = False
                    self.clear_timer("fight")
                    self.log(f"battle hold expired ({_held:.0f}s ≥ "
                             f"{_FIGHT_HOLD_SEC:.0f}s)  back out")
                    return action_back("battle hold expired")
                return action_wait(1000,
                                   f"story battle in progress ({_held:.0f}s)")

        # P0.9: a dialog offering NAVIGATION-AWAY (取消键 present, 确认键
        # absent — e.g. 獲得新收藏!是否立即移動? 取消/立即前往, live
        # 2026-06-10; 立即前往 has no trained cls)  always 取消, stay mining.
        # The skip-confirm dialog has BOTH buttons  unaffected (P1 handles).
        cancel_only = self.find_cls(screen, UC.BTN_CANCEL, conf=_CLS_CONF)
        if cancel_only is not None and self.find_cls(screen, UC.BTN_CONFIRM, conf=0.20) is None:
            self._settle(_SETTLE_SHORT)
            return action_click_box(cancel_only, "取消 — decline navigation offer, keep mining")

        # P1: cutscene skip chain (story auto-plays  skip ASAP).
        cut = self._handle_cutscene(screen)
        if cut is not None:
            return cut

        # P2: chapter-info modal  进入章节.
        play = self.find_cls(screen, UC.STORY_ENTER_CHAPTER, conf=_CLS_CONF, region=(0.25, 0.50, 0.80, 0.92))
        if play is not None:
            self._barren = 0
            self._settle(_SETTLE_LONG)
            return action_click_box(play, "enter chapter (进入章节)")

        # P3: MINE — drill deepest-first (node 入场键 > 黄点章 > new 篇/卡).
        mine = self._mine_action(screen)
        if mine is not None:
            self._barren = 0
            self._back_streak = 0
            self._settle(_SETTLE_LONG)
            return mine

        # P3.5: on the 剧情 hub CATEGORY page  open the current category card to
        # drill IN. MUST run before the barren scan: the category page carries
        # category cls but NO chapters, so scanning it just exhausts the whole
        # category without ever entering it (the bug that "exhausted" all three).
        hub_card = self._pick_hub_card(screen)
        if hub_card is not None:
            self._barren = 0
            self._page_turns = 0
            self._main_swipes = 0
            self._tried_cards = []
            self._settle(_SETTLE_SHORT)
            return action_click_box(hub_card, f"open category ({hub_card.cls_name})")
        # All categories exhausted/skipped  finish cleanly (the old finish in
        # _exhaust_and_advance is unreachable from the hub fast-exhaust path —
        # live 2026-06-10 the skill instead wandered into "nav: can't reach").
        if self._cat_idx >= len(self._categories):
            self.log(f"all categories done ({len(self._exhausted)} exhausted)")
            return action_done("story mining finished (all categories)")

        # P4: INSIDE a category but nothing unplayed visible  reveal more, then
        # (if truly barren) exhaust the category and advance.
        page = self.detect_screen_yolo(screen)
        on_story = (page == "Story") or self._on_any_story_page(screen)
        #  Short/side card GRID pages carry NO trained page cls when no New
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
            if self._barren == 1:
                self.mark("barren")
            _bw = self.since("barren")
            if _bw < _BARREN_SEC:
                return action_wait(
                    350, f"scanning for unplayed "
                         f"({_bw:.1f}/{_BARREN_SEC:.0f}s, n={self._barren})")
            # Nothing minable here (e.g. a chapter with only battle/locked/done
            # nodes). Back OUT one level to find the next unplayed chapter/篇 —
            # NOT exhaust the whole category. Only give up to the next category
            # after backing out many times with nothing mined.
            self._barren = 0
            self._back_streak += 1
            if self._back_streak > 8:
                self._back_streak = 0
                return self._exhaust_and_advance(screen)
            self._settle(_SETTLE_SHORT)
            back = self.find_cls(screen, [UC.BTN_BACK], conf=_CLS_CONF)
            return (action_click_box(back, "back out  next unplayed chapter/篇")
                    if back else action_back("back out  next unplayed chapter/篇"))

        # P6: mission hub — 劇情 tile.
        story_tile = self.find_cls(screen, UC.HUB_STORY, conf=_CLS_CONF)
        if story_tile is not None:
            self._nav_ticks = 0
            self._nav_t0 = None
            self._settle(_SETTLE_SHORT)
            return action_click_box(story_tile, "enter 剧情 hub")

        # P7: navigation (lobby  mission hub).
        return self._navigate(screen)

    #  cutscene skip
    def _handle_cutscene(self, screen: ScreenState) -> Optional[Dict[str, Any]]:
        # Skip-confirm dialog (是否略過此劇情? 取消/確認)  確認. Recognized by
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
                self._skipped_stories += 1
                self._settle(_SETTLE_SHORT)
                return action_click_box(confirm, "confirm story skip (确认键)")
        if self._cut_ticks > 0:
            cont = self.find_cls(screen, UC.STORY_TAP_CONTINUE, conf=_CLS_CONF)
            if cont is not None:
                self._cut_ticks = 0
                return action_click_box(cont, "tap continue")

        if self._cut_ticks < 3:
            # after-ack(2026-07-28, 与 momo_talk 同病 —— 那边 step_walk 实拦到
            # **连点 5 次**): 「是否略過此劇情?」确认框要时间渲染, 而这里只要
            # 跳过键/menu 还在屏上就每 tick 再点一次, **第二发正好把刚弹出的框
            # 又关掉**  自锁。这边有 `_cut_ticks < 3` 兜着所以最多三连发, 但
            # 三发同样能把框关掉两次。发过就等帧证据(上面那段接走確認)。
            if self._cut_ticks > 0 and self.since("story_cut") < _SKIP_ACK_SEC:
                return action_wait(300, f"跳过/menu 已发 — 等略過确认框 "
                                        f"(after-ack {self.since('story_cut'):.1f}"
                                        f"/{_SKIP_ACK_SEC:.1f}s)")
            skip = self.find_cls(screen, UC.STORY_SKIP, conf=0.40, region=(0.82, 0.05, 1.0, 0.30))
            if skip is not None:
                self._cut_ticks += 1
                self.mark("story_cut")
                return action_click_box(skip, "跳过故事键")
            menu = self.find_cls(screen, UC.STORY_MENU, conf=0.40, region=(0.82, 0.0, 1.0, 0.16))
            if menu is not None:
                self._cut_ticks += 1
                self.mark("story_cut")
                return action_click_box(menu, "open 剧情menu")
        cont = self.find_cls(screen, UC.STORY_TAP_CONTINUE, conf=_CLS_CONF)
        if cont is not None:
            return action_click_box(cont, "tap continue")
        if self._cut_ticks > 0:
            self._cut_ticks = max(0, self._cut_ticks - 1)
        return None

    #  mining (drill deepest-first)
    def _mine_action(self, screen: ScreenState) -> Optional[Dict[str, Any]]:
        #  Mine ONLY on a real story page. A 黄点 (DOT_YELLOW) on the LOBBY / 任务
        # 大厅 is a nav badge (student / campaign_nav / cafe), NOT an unplayed
        # chapter — mining off the lobby clicked the campaign-nav badge and landed
        # on 任务关卡 instead of 剧情. No story cls on screen  defer to navigation
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
                    # forever (battle  back  battle …).
                    if any(abs(px - epos[0]) < 0.04 and abs(py - epos[1]) < 0.04
                           for px, py in self._tried_enters):
                        continue
                    self._tried_enters.append(epos)
                    return action_click_box(row_enter, "enter unplayed node (入场键)")

        # 2) CHAPTER level: a 黄点 in the content area = unplayed chapter  click
        #    its row (use the dot's y; click toward the row center-left).
        dot = self._content_yellow_dot(screen)
        if dot is not None:
            self._tried_chapters.append(dot.cy)   # opened  don't reopen (battle-only)
            self._tried_enters = []   # new chapter  reset node-dedup
            # The 黄点 sits at the LEFT edge of the chapter row; the clickable
            # chapter TITLE is to its RIGHT (e.g. 第2章 與往日訣別). Clicking
            # dot.cx-0.05 landed LEFT of the chapter panel (on the 篇 area) and
            # never opened it (live: clicked 0.685 forever). Click into the title
            # to the right of the dot.
            row_x = min(0.93, dot.cx + 0.10)
            return action_click(row_x, dot.cy, "open unplayed chapter (right of 黄点)")

        # 3) 篇/CARD level: a `new` badge  select/enter. ONLY when NO node-level
        #    入场键 is present. A New badge on a NODE (新节点, e.g. 巢穴 New) is
        #    NOT a 篇/卡 entry — clicking it does nothing and loops forever. New
        #    means "open this 篇/card" only on the 篇/grid screens (no 入场键 there).
        #     Dedup by position: a battle-gated 篇 keeps its New badge forever
        #    AND selecting it resets barren/back_streak  infinite category loop
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
        # chapter keeps showing its 黄点  skip so we don't reopen it forever).
        dots = [d for d in dots if d.cy > 0.18
                and not any(abs(c - d.cy) < 0.05 for c in self._tried_chapters)]
        return min(dots, key=lambda b: b.cy) if dots else None

    #  reveal more (page-turn grids / swipe-left main 卷 list)
    def _reveal_more(self, screen: ScreenState) -> Optional[Dict[str, Any]]:
        # Short/side grids paginate with 右切换.
        arrow = self.find_cls(screen, UC.ARROW_RIGHT, conf=_CLS_CONF, region=(0.85, 0.30, 1.0, 0.70))
        if arrow is not None and self._page_turns < _MAX_PAGE_TURNS:
            self._page_turns += 1
            self._barren = 0
            self._settle(_SETTLE_SHORT)
            self._tried_cards = []   # cards shift on page turn
            self.log(f"右切换 next page ({self._page_turns}/{_MAX_PAGE_TURNS})")
            return action_click_box(arrow, "next page (find new card)")
        # Main 卷 list scrolls horizontally — swipe LEFT to reveal newer 卷.
        if self._current_cat == UC.STORY_MAIN and self._main_swipes < _MAX_MAIN_SWIPES:
            self._main_swipes += 1
            self._barren = 0
            self._settle(_SETTLE_SHORT)
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
        self._settle(_SETTLE_SHORT)
        return (action_click_box(back, "back to hub for next category")
                if back else action_back("back to hub for next category"))

    #  helpers
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
        y0.72) dot(0.445,0.165) ; 短篇 title(0.57-0.65,y0.52) dot(0.667,
        0.167) ; strips don't cross-talk (0.486<0.662, 0.574>0.450)."""
        region = (card.x1, 0.10, min(1.0, card.x2 + 0.06), max(0.12, card.y1))
        return self.find_cls(screen, UC.DOT_YELLOW, conf=0.35, region=region) is not None

    def _pick_hub_card(self, screen: ScreenState) -> Optional[YoloBox]:
        #  The hub CATEGORY page shows ALL category cards; inside a category
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
                # Back ON the hub page after drilling this category  its
                # mineable content is done (battle-gated nodes keep the dot/New
                # but can't be mined) — exhaust and advance NOW. The old
                # "return None  P4 barren scan" churned: hub scan  back out
                # to 任务大厅  re-enter  scan … ×8 before exhausting (live
                # 2026-06-10, ~20s per lap).
                #  But NOT right after opening it: the open-click's transition
                # lags 2-3 ticks with the hub still on screen — that false-
                # exhausted 主線 the moment it was opened (live 2026-06-10).
                if self.ticks - self._cat_opened_tick <= 6:
                    return None   # transition settling — re-read next tick
                self.log(f"category {cat}: hub re-reached after drill  exhausted")
                self._exhausted.append(cat)
                self._current_cat = None
                self._cat_idx += 1
                continue
            card = self.find_cls(screen, cat, conf=_CLS_CONF)
            if card is not None:
                self._card_misses = 0
                #  Signal-driven category gate: no 黄点 on the card = nothing
                # to mine inside — skip without entering (e.g. 支線 today).
                if not self._card_has_mine_dot(screen, card):
                    self.log(f"category {cat}: 无黄点  no mine, skip")
                    self._exhausted.append(cat)
                    self._no_dot_skips += 1   # 竣工判据靠它区分"没矿"与"没干活"
                    self._cat_idx += 1
                    continue
                self._current_cat = cat
                self._cat_opened_tick = self.ticks
                return card
            # Card cls not seen THIS frame. A 1-frame flicker must not skip the
            # whole category (live 2026-06-10: 短篇 flickered out right after
            # 主線 exhausted  its mine was skipped entirely). Retry a few hub
            # frames before giving up on it.
            self._card_misses += 1
            if self._card_misses < 3:
                return None
            self._card_misses = 0
            self._cat_idx += 1
        return None

    def _navigate(self, screen: ScreenState) -> Dict[str, Any]:
        self._nav_ticks += 1
        if self._nav_t0 is None:
            self._nav_t0 = self.clock()
        _nw = self.clock() - self._nav_t0
        if _nw > _NAV_MAX_SEC:
            self.log(f"nav: can't reach story hub ({_nw:.0f}s / "
                     f"{self._nav_ticks} tick) — 抓这一刻的帧看缺什么 cls")
            return action_done(f"story mining unreachable ({_nw:.0f}s)")
        if self.detect_screen_yolo(screen) == "Lobby":
            act = self.click_cls(screen, UC.NAV_TASKS, "open hub from lobby", conf=_CLS_CONF)
            if act is not None:
                self._settle(_SETTLE_SHORT)
                return act
            return action_wait(400, "nav: 任务大厅入口 not seen")
        return action_wait(400, "nav: waiting (no known page)")
