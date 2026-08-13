# -*- coding: utf-8 -*-
"""全局打断处理 —— **一处定义，所有 flow 共享**（§A2）。

老代码的病：「剧情过场不吃 ESC」这件事，13 个 skill **一处都不知道**。
于是 2026-08-07 整个活动被开场剧情挡死 —— bot 连按 5 次返回键全无响应，
而每个 skill 都以为"按返回就能退出去"。

这里登记的打断优先于任何 flow 的决策:
  money_popup     停整条链，交人审
  story_cutscene  剧情逃生（menu  跳过  确认）
  levelup         点掉
  loading         等

**打断 ≠ 覆盖层**（08-11 审计时写清楚，免得又有人来"修"错层）:
   · 打断(interrupt) = 登记在 `pages.INTERRUPTS`，与"在哪一页"无关，由 runner
     在 flow.decide **之前**调用本文件；
   · 覆盖层(overlay) = 登记在 `pages.PAGES` 里 `overlay=True` 的那几条
     （confirm_dialog / claim_panel / sweep_results / ack_dialog / reward…），
     由 `Flow.decide()` 按 **overlay  pre_page  底页** 的顺序派发，
     默认实现在 `flow/base.py`。
    单键通知框（ack_dialog）属于**后者**，任何页面（含 arena）都已经被优先
     处理，不需要也不该往这里塞。08-11 全天 408 帧离线复验:
     **带 `确认键` 却拿不到 overlay 身份的帧 = 0**。

真正的缺口在另一头（同一次审计的实测，交给 pages.py 那边处理）:
   408 帧里有 **108 帧屏上有 `弹窗叉叉`（= 有个弹窗盖着）但 overlay 判成 None**
   —— 这种弹窗对覆盖层是**隐形**的，flow 会继续对着它底下的按钮点。
   战术大赛「對戰對象」详情面板就是其中一例（见 flow/arena.py 的死面板逃生）。
"""
from __future__ import annotations

import time
from typing import Optional

from routing_v2.act.action import Action, halt, tap_box, wait
from routing_v2.percept.observe import Observation
from routing_v2.state import vocab as V

# 剧情逃生的三段链在 UI 上的位置约束（都实测过）
_TOP_RIGHT = (0.78, 0.0, 1.0, 0.30)
_MID = (0.28, 0.52, 0.88, 0.88)


class Interrupts:
    def __init__(self, log=None):
        self._log = log or (lambda m: print(m, flush=True))
        self._story_ts = 0.0
        # 「下一章節」框点哪个键 —— 由当前 flow 说了算（用户 2026-08-11:
        #   「要连着推的话可以不用中断出来」）。
        #   默认 False = 中斷（逃生语义：别的 flow 撞上剧情就是要离开）；
        #   StoryMiningFlow 会把它置 True，因为它**就是来看剧情的**。
        self.watch_next_chapter = False
        self._load_t0 = 0.0
        self.load_total = 0.0
        self.load_count = 0
        self.counts = {}
        self._warned = set()

    def handle(self, kind: str, obs: Observation) -> Optional[Action]:
        self.counts[kind] = self.counts.get(kind, 0) + 1
        fn = getattr(self, f"_on_{kind}", None)
        if fn is None:
            # **没实现的打断不许静默**（同「死开关不再静默」那条纪律）:
            #    `pages.INTERRUPTS` 里新登记一条却忘了在这里写 handler 时，
            #    runner 拿到 None 会当成"这一页不归打断管"直接交给 flow ——
            #    而打断的语义恰恰是"flow 不许在这个局面上动手"。出声一次。
            if kind not in self._warned:
                self._warned.add(kind)
                self._log(f"    打断 `{kind}` 在 interrupt.py 里没有 handler"
                          f" — 这一帧交回 flow 决策了，去补 `_on_{kind}`")
            return None
        return fn(obs)

    # ── 金钱 ─────────────────────────────────────────────────────────
    def _on_money_popup(self, obs: Observation) -> Optional[Action]:
        import routing_v2.act.money as money_rules
        # **組合包页不停机**（08-09 审查实锤）：bot 是**主动**进来拿免費包的，
        #    不是意外撞上购买框。旧代码无条件 halt  默认配置下每次自主跑
        #    都在 shop 这一步整轮停机，`on_combo_pack` 永远执行不到。
        #    放行的只是"别停机"：真正的钱仍被 Gate 拦死 —— 购买页上只有
        #      `_NAV_SAFE` 里的导航键放行，其余一律人审；`money=True` 再过一道。
        if money_rules.is_combo_pack_page(obs):
            self._log("    組合包页（免費包入口）— 不停机，交给 flow 走，"
                      "花钱的每一下仍要过闸人审")
            return None
        # 站在**購買青輝石页**上、但当前页签不是「組合包」——2026-08-13 小号实帧:
        #    这一页默认开在「特別販售」，整页 CAD 25.99 / 16.99 的真钱货架，
        #    而 `组合包未选择`(414) 只是旁边那个**没选中的页签标签**。
        #    停整轮是**过度反应**: 我们知道自己是怎么进来的（shop 主动点了大厅
        #    广告位），屏上就摆着叉叉，而**离开严格比站在真钱货架上更安全**。
        #    halt 的本意是"意外撞上购买框, 停下来交人"，不是"关掉整天的日常"。
        #     只放行**关闭**这一个动作；关不掉才 halt。
        #    判据要排掉双键确认框（那种才是真·成交前一刻，必须停）。
        if (obs.has(V.COMBO_PACK, money_rules.CONF)
                and not (obs.has(V.CONFIRM, money_rules.CONF)
                         and obs.has(V.CANCEL, money_rules.CONF))):
            x = obs.find(V.CLOSE_X, 0.55)
            if x is not None:
                self._log("    購買青輝石页但页签不是組合包（真钱货架）— 关掉走人，"
                          "不停整轮")
                return tap_box(x, "真钱货架页 — 关掉离开（免費組合包这段跳过）")
        why = money_rules.purchase_context(obs) or "（判据已不成立）"
        return halt(f"{why} —— 立即停，交人逐帧审（青辉石只进不出）")

    # ── 系统退出确认框 ─────────────────────────────────────────────
    def _on_quit_dialog(self, obs: Observation) -> Optional[Action]:
        """「是否結束？」—— **確認就是退出游戏**，只许点取消。

        这个框不是游戏内容，是我们自己按返回键按出来的（大厅按返回会弹它）。
        按返回那件事已经在 nav/base 里禁掉了，这里是**第二道**：万一还是弹出来，
        必须点取消，绝不能落到 `on_confirm_dialog` 的"默认点確認"上。
        """
        c = obs.find(V.CANCEL, 0.45)
        if c is not None:
            return tap_box(c, "系统「是否結束？」框  点取消（確認=退出游戏）")
        x = obs.find(V.CLOSE_X, 0.50)
        if x is not None:
            return tap_box(x, "系统退出框  关掉")
        return wait("系统退出框，但取消键没检出 — 绝不点確認")

    # ── 剧情过场 ───────────────────────────────────────────────────────
    def _on_story_cutscene(self, obs: Observation) -> Optional[Action]:
        """BA 的过场**不吃 KEYCODE_BACK**（08-07 连按 5 次实测全无响应）。
        唯一有效链: 剧情menu(右上)  跳过故事键  确认键。

        中间要留 after-ack: 点完「跳过」到确认框弹出有动画时间，这段时间里
        跳过键还在屏上 —— 不留冷却就会连点两下，第二下落到确认框外面。
        冷却必须显式记时间戳（裸 since() 首次返回 0.0 会被当成"冷却已过"）。
        """
        # 「下一章節」框（2026-08-11 小号实测）: 跳完一段剧情，游戏会问
        #    「要觀看下一章節嗎？」，两个键 中斷(剧情中断退出 0.99) / 觀看(剧情观看 0.99)。
        #    **老逃生链三段全匹配不上** —— 这一页没有 剧情menu / 跳过故事键 / 確認，
        #    于是落到最后那句 `wait(...不瞎点)` 卡死。
        #     逃生的语义是"离开剧情"，所以点 **中斷**；`观看` 会把我们继续拖进下一段。
        quit_b = obs.find(V.STORY_QUIT, 0.40)
        if quit_b is not None and obs.has(V.STORY_WATCH, 0.40):
            self._story_ts = 0.0
            if self.watch_next_chapter:
                # 剧情挖矿在跑：连着推下去比退出来再进快得多，也少一次导航
                w = obs.find(V.STORY_WATCH, 0.40)
                if w is not None:
                    return tap_box(w, "「下一章節」框  觀看（剧情挖矿连推）")
            return tap_box(quit_b, "剧情逃生: 「下一章節」框  中斷")

        cf = obs.find(V.CONFIRM, 0.35, region=_MID)
        if cf is not None:
            self._story_ts = 0.0
            return tap_box(cf, "剧情逃生: 确认略過")
        if time.time() - self._story_ts < 3.0:
            return wait("剧情逃生: 等略過确认框弹出")
        skip = obs.find(V.STORY_SKIP, 0.40, region=_TOP_RIGHT)
        if skip is not None:
            self._story_ts = time.time()
            return tap_box(skip, "剧情逃生: 跳过故事键")
        menu = obs.find(V.STORY_MENU, 0.40, region=_TOP_RIGHT)
        if menu is not None:
            self._story_ts = time.time()
            return tap_box(menu, "剧情逃生: 打开剧情menu")
        tap_cont = obs.find(V.STORY_TAP_CONTINUE, 0.40)
        if tap_cont is not None:
            return tap_box(tap_cont, "剧情逃生: 点击继续")
        return wait("剧情过场，但逃生链的 cls 一个都没检出 — 不瞎点")

    # ── 升级过场 ───────────────────────────────────────────────────────
    def _on_levelup(self, obs: Observation) -> Optional[Action]:
        """全屏升级过场：点掉。

        找不到落点时返回 **wait 而不是 None**（08-11 对齐三个 handler 的口径）。
           打断是"连续 N 帧确认"才锁上的，解除同样要 N 帧 —— 中间那几帧里
           cls 可能一时掉到 0.45 以下，而**过场还实实在在盖在屏上**。
           返回 None 等于把这几帧交回 flow，flow 就会去点过场底下的按钮
           （点击被过场吞掉 = 连发族那个"判定带宽不一致  弹入/弹出动画帧
             契约必破"的老病）。`_on_story_cutscene` / `_on_quit_dialog`
           早就是 wait 收尾，这里是唯一的例外，补齐。
           代价上限 = 多等 confirm_frames 帧（过场真没了打断自然解除）。
        """
        b = obs.find([V.BOND_LEVELUP, V.REGION_LEVELUP], 0.45)
        if b is not None:
            return tap_box(b, "升级过场: 点掉")
        return wait("升级过场还锁着，但这一帧没检出可点的横幅 — 等，不交回 flow 乱点")

    # ── 加载 ───────────────────────────────────────────────────────────
    def _on_loading(self, obs: Observation) -> Action:
        """用户 2026-08-08 定死的等待语义:

           「只有没 cls 的时候，或者出现『加载中』这个 cls 的时候才等，
             而且**等待时间也是『加载中』这个 cls 的持续时间**」

        所以这里**没有任何时长参数** —— 只要 `加载中`(cls 22, train 855/val 44)
        还在屏上就返回 wait，它一消失下一帧立刻继续。等多久由游戏决定，不由
        我们猜。这是全 bot 唯一合法的"等"，另一种是屏上没有对应 cls（no-op）。
        """
        if self._load_t0 == 0.0:
            self._load_t0 = time.time()
        return wait("加载中 — 等它自己消失（时长由 cls 决定，不设上限）")

    def note_no_loading(self) -> None:
        """`加载中` 不在场时由 runner 调一次，用来结算并打印这次加载的真实时长。"""
        if self._load_t0:
            dur = time.time() - self._load_t0
            self._load_t0 = 0.0
            self.load_total += dur
            self.load_count += 1
            return dur
        return None
