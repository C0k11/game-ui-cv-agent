# -*- coding: utf-8 -*-
"""活动 —— 用户规则最多的一条链，逐条编码进来。

用户规则（memory event_ops_playbook + 2026-08-07/08 现场口述）:
  1. **先 Q1  Qn 整体打通，再打加成**（`order: clear_then_bonus`）
  2. **首通用部队1**（速推主力）—— 活动关卡的 **Best Record 会把首通时的
     加成倍率永久锁定在那一关上**，用错队伍不可逆
  3. 加成队 = 部队2，且**先看商店推算缺哪种币**再编队（`shop_plan_before_bonus`）
  4. **别自己瞎滑关卡栏**：从 1 开始打没打过的关，游戏会自动把下一关归位过来

今天（08-07/08）在这条链上踩的坑，逐条对应到代码:

  【轮播】活动入口是 2 项轮播，周期实测 **3.00s**（页间 0.08s 空白）。
     405「距離結束還剩」= 当期可打；474「距離獎勵獲得結束」= 上期余韵期。
     在 405 窗口的**尾巴**上点，落地时已翻成 474  进上期活动。
      只在观测到 **(非405  405) 这个跃迁**的第一帧动手，有满满 2.9s 余量。
      落点还要 **+0.075**：框标的是**倒计时气泡**，可点的是下面的卡片本体。
      派发前还有 Gate 的 JIT 复验兜底（§A7：加闸，不赛跑）。

  【新活动首日】老代码四处判据把"内容多"当页面前提，CODE:BOX 首日
     1 关开 + 4 关锁，全线崩：
       · 要 ≥2 个已解锁入场键才认列表  不认
       · 无条件滑到底找尾关  **把唯一能打的关滑走了**
       · 要齐 3 行才算列表加载完  永远等不到
       · 弹窗只认扫荡面板  首通弹的是「章節資訊 + 進入章節」，不认
      页面身份交给 state/pages.py（结构判据），这里只管"打哪一关"。

  【关卡完成度】`活动剧情关卡_已看`/`活动站斗关卡_已打` **train 都是 0**，
     判不了"这关打过没"  改用 `关卡得星_0`(1263框) / `关卡得星_3`(3765框)
     按行匹配间接判。
"""
from __future__ import annotations

from typing import List, Optional

from routing_v2.act.action import Action, swipe, tap_box, wait
from routing_v2.flow import nav
from routing_v2.flow.base import ExitMixin, Flow, Outcome
from routing_v2.flow.battle import BattleMixin, FormationMixin
from routing_v2.percept import read as R
from routing_v2.percept.observe import Box, Observation
from routing_v2.state import vocab as V
from routing_v2.state.machine import StateView

# HUB 活动入口的落点下移量。405/474 的框标的是倒计时气泡，点气泡完全不响应。
#   由 08-07 手动实测反推（点 cy=0.222 生效，框心 cy=0.149）。
HUB_TILE_DY = 0.075
STAGE_PANEL = (0.42, 0.10, 1.0, 0.95)
ROW_TOL = 0.055          # 入场键 与 得星 认为"同一行"的 cy 容差

# 轮播位两个活动共用 405（见 `on_event_guide_hub`） 进错了要退出去重进，
#   但**必须有上限**：轮播是随机的，无限重试 = 把整个 flow 预算烧在进出上，
#   而且日志看起来一切正常（08-11 shop 死循环就是这么烧掉 max_minutes 的）。
#   3 次 ≈ 至少两次完整轮播周期都没抽中另一个活动  该交人了。
GUIDE_HUB_MAX_TRIES = 3


class EventEntryMixin:
    """大厅 -> 任务大厅(轮播闸) -> 活动页 的进场链，event / event_shop 共用。

    2026-08-13 实锤: event_shop 单独跑（不经 event 交棒）时在 lobby 上
       401 帧没有任何可执行动作 -- 它压根没有进场的手。轮播闸这套判据
       （405 跃迁窗口 / 引导型活动认错退出）只能有一份，抽 mixin 共用。
    状态键全部用 .get / setdefault 惰性取 -- mixin 不依赖宿主 setup 初始化。
    """

    def on_lobby(self, obs, st):
        return nav.enter(obs, V.NAV_TASKS, "任务大厅")

    def on_task_hall(self, obs, st):
        """轮播闸：只在「刚翻成 距离结束还剩」那一帧动手。"""
        cur = obs.find(V.EVENT_LIVE, 0.45, region=(0.0, 0.0, 1.0, 0.42))
        if cur is None:
            # 现在显示的是上期入口 / 页间空白  记下"看见过别的了"
            self.state["saw_other"] = True
            if self.stalled(st, 300):
                return self.finish(
                    Outcome.SKIPPED,
                    "任务大厅盯了很久都没等到「距离结束还剩」— 当前没有进行中的活动")
            return wait("等轮播翻到「距离结束还剩」")
        if not self.state.get("saw_other"):
            # 405 已经显示一会儿了，可能正处在窗口尾巴上 -- 等下一轮跃迁
            return wait("405 在场但不是刚翻过来的 — 等下一次跃迁（避开轮播尾巴）")
        # `saw_other=False` **必须挂 post**（08-10 live 实锤，同族第 7 处）：
        #    写在这里 = 「决策了要点」就把跃迁标记清掉，而这一发未必真发得出去
        #    -- gate 可能吞掉它，`--ticks N` 循环里前 N-1 个 tick 的 decide
        #    结果更是**直接丢弃**。实测表现：连试 4 次都停在
        #    「405 在场但不是刚翻过来的」，**窗口全被那些丢弃的 decide 吃掉了**。
        #    注意上面那句 `saw_other=True` 留在 decide 期是**对的**：那是
        #      「我这一帧看见了别的入口」这个**观测事实**，不是动作副作用。
        #      观测可以在 decide 期记，动作后果只能挂 post -- 这条界线要分清。
        return tap_box(cur, "进当期活动（捕到 405 跃迁）",
                       dy=HUB_TILE_DY, counter="entered",
                       post=lambda: self.state.update(saw_other=False))

    # -- 进错活动: 上期余韵 ------------------------------------------------
    def on_event_ended(self, obs, st):
        """`後日談` 在场 = 上期活动余韵期。退出去重进，别在这里刷。"""
        self.log("进到上期活动了（检出 後日談） 退出去重进")
        self.note_lines.append("误入上期活动一次，已退出重进")
        return self.exit_step(obs, prefer_close=False) or wait("等退出控件")

    # -- 进错活动: 引导型（没有活动关卡的那种）------------------------------
    def on_event_guide_hub(self, obs, st):
        """进到「引导型活动」了（夏萊總結算这类）-- 这不是我要打的活动。

        2026-08-11 live：任务大厅左上**一个轮播位轮流显示两个活动**，
        「夏萊總結算」和「CODE BOX(复刻)」**都只检出 405 距离结束还剩**
        点哪个纯看运气，实测连进两次都进了夏萊。时序赛跑赢不了（轮播 3.0s，
        转场帧上的点击还会被稳定门吞）， 只能进去之后按内容认出来再退。

        **绝不点这一页的「入場」**：夏萊那两个入场键通向的是**普通
           任務 / 特殊任務**（游戏指南原文「不存在活動關卡」，道具靠去打普通
           关卡掉落）。**刷什么是用户的策略**（08-10 我擅自改 arena 配对已经
           被当场叫停一次），bot 不许自己决定去刷任務。这一版只做三件事：
           **认出来 / 不进错 / 说清楚**。

        退出走 `exit_step`（取消 > 確認 > 返回 > 回大厅），全程 `tap_box` 打在
        真检出的框上，零写死坐标。没走 `nav.to_lobby` 是因为它的 `_EXITABLE`
        白名单在 nav.py 里，新页面名还没登记进去（那个文件不在本次改动范围）；
        语义上两者都是「只点屏上确实存在的退出控件」。
        """
        # 计"进错了几次"要**数事实不数意图**（N5 全仓复发过 3 次）：
        #    这一页会连续几十 tick，每 tick 加一次的话第一次进来就直接超限。
        #    `st.changed` = 状态机刚把身份确认成这一页 = **真的又进来了一次**。
        #    `guide_seen` 兜住"flow 是在这一页上被创建/接手的"（step 模式常见，
        #    那种情况下 st.changed 永远不会为真  计数恒 0  上限形同虚设）。
        if st.changed or not self.state.get("guide_seen"):
            self.state["guide_seen"] = True
            n = self.bump("guide_hits")
            self.log(f"进到**引导型活动**了（第 {n} 次）：这个活动没有活动关卡，"
                     f"页面上那两个「入場」通向普通 任務/特殊任務 -- 退出去重进")
            # 竣工报告里要看得见（completion_gap：跑完了 != 活干完了）
            if self.once("guide_note"):
                self.note_lines.append(
                    "轮播抽中了没有活动关卡的引导型活动（夏萊總結算这类），已退出重进")
        n = int(self.state.get("guide_hits", 0))

        if n >= int(self.cfg.get("guide_hub_max_tries", GUIDE_HUB_MAX_TRIES)
                    or GUIDE_HUB_MAX_TRIES):
            # 先**走出去**再收工，别把 bot 丢在这一页上：新页面名不在
            #    `nav._EXITABLE` 里  runner 的兜底导航对它返回 None
            #    下一条 flow 会在这儿空转到自己超时（08-11 shop 死循环同形）。
            #    真正的 finish 在 `pre_page` 里、等页面换走之后才发。
            if self.once("guide_giveup_log"):
                self.log("连续进错到达上限  先退出这一页，然后收工交人审")
            self.state["guide_giveup"] = True
            if self.stalled(st, 200):
                # 退不出去（退出控件全检不出）-- 那就当场收工，别无限耗着
                return self.finish(
                    Outcome.BLOCKED,
                    "连续进到引导型活动（夏萊總結算这类，没有活动关卡），"
                    "而且退不出这一页 — 停手交人审")

        return self.exit_step(obs, prefer_close=False) or wait("等退出控件")


class EventFlow(EventEntryMixin, FormationMixin, BattleMixin, ExitMixin, Flow):
    name = "event"
    module = "event"
    entry_page = "task_hall"

    def setup(self) -> None:
        self.state.update(
            phase="clear",          # clear  shop_plan  bonus  done
            saw_other=False,        # 轮播: 上一帧不是 405
            cleared=0, farmed=0, swept=0, entered=0,
            # 轮播抽错活动（进到"没有活动关卡"的引导型活动）的台账
            guide_hits=0, guide_seen=False, guide_giveup=False,
        )
        self.want_team = int(self.cfg.get("clear_first_with_team", 1))
        order = self.cfg.get("order", "clear_then_bonus")
        if order == "bonus_only":
            # 阶段名必须和 `_bonus_step` 认的那两个对上（"bonus_clear" /
            #    "bonus_sweep"）。第一版这里写的是 "bonus"，派发处认不出 
            #    flow 在关卡列表前干等到超时，一个 tap 都没有（08-08 实测）。
            self.state["phase"] = "shop_plan" if self.cfg.get(
                "shop_plan_before_bonus", True) else "bonus_clear"
            self.want_team = int(self.cfg.get("bonus_team", 2))

    # ══ 活动页：切到 Quest 页签 ════════════════════════════════════════
    def on_event_page(self, obs, st):
        """别用写死坐标点页签（`_POS_QUEST_TAB=(0.635,0.151)` 是**另一个
        活动的版面**上标的，CODE:BOX 两页签版面下正好落在 Story 上）。
        用 cls：`活动quest`(未选中) 在场就点它；`活动quest_已选择` 在场说明
        已经在 Quest 页了。"""
        if obs.has(V.EVENT_QUEST_SEL, 0.40):
            return wait("已在 Quest 页签，等关卡行渲染")
        tab = obs.find(V.EVENT_QUEST, 0.35, region=(0.0, 0.0, 1.0, 0.34))
        if tab is not None:
            self.state["tab_tries"] += 1
            return tap_box(tab, "切到 Quest 页签")
        if self.stalled(st, 120):
            return self.finish(Outcome.UNKNOWN,
                               "活动页里既没有 `活动quest` 也没有 `活动quest_已选择`"
                               " — 这个活动的版面没见过，交人看")
        return wait("等 Quest 页签 cls")

    # ══ 关卡列表 ══════════════════════════════════════════════════════
    def _rows(self, obs: Observation):
        """把 (入场键, 得星) 配成行。**一行就够**，绝不要求"至少 N 行"。"""
        enters = obs.rows(V.STAGE_ENTER, 0.45, region=STAGE_PANEL)
        locked = obs.rows(V.STAGE_ENTER_LOCKED, 0.45, region=STAGE_PANEL)
        stars = obs.all([V.STAR_0, V.STAR_3], 0.40)
        out = []
        for e in enters:
            s = None
            best = ROW_TOL
            for k in stars:
                d = abs(k.cy - e.cy)
                if d < best:
                    best, s = d, k
            out.append((e, s.cls if s is not None else None))
        return out, locked

    def on_event_quest_list(self, obs, st):
        self.state["in_event"] = True      # 到过活动页（stage_popup 上下文守卫用）
        # **从别的页面回到关卡列表** = 上一轮编队子链已经翻篇  解锁 af_* once。
        #    必须用 `st.changed`（页面**刚切**过来那一 tick）而不是"人在
        #    列表页"（08-09 审查抓到）：列表页会连续几十 tick，每 tick 都清
        #    等于**把 once 保护打成死码**。
        # ⛔ 进关 once（enter*/sweep_enter*）**不在这里清**（08-15 日常 live
        #    实锤）: 活动页身份会抖（event_quest_list/unknown/facility 来回翻,
        #    收尾日志整屏「3s 内页面身份变了 5 次」）, 每次翻回来 changed 边沿
        #    都触发一次全清 —— 入场键点完 16 tick 内就被重武装再点一发,
        #    第二发落在**滚动后的列表**上, 离「掃蕩開始」只差 2% 屏高。
        #    进关 once 的解锁改在 `_bonus_step`/扫荡分支里**按事实**做:
        #    「点了入场键却 40 个列表帧等不到弹窗」才重武装（有界, 抖动免疫）。
        if st.changed:
            for k in [k for k in list(self.state)
                      if k.startswith("once:af_")]:
                self.state.pop(k, None)
        ph = self.state["phase"]
        # "这个活动有没有活动点数通道"只有在关卡列表页看得到（`奖励资讯`
        #   就挂在活動點數 进度条旁边）。记给商店推算用：
        #   有点数  点数=最后一关、商店最下面的币=倒数第二关；
        #   没点数  **商店最下面的币就是最后一关**，再往上推。
        # 不许单帧定案（08-09 实锤：这活动明明有 1610/15000 点数条，第一帧
        #    恰好没检出 `奖励资讯` 就被记成「无」 推算映射整体偏一档）。
        #    正向一见即记；「无」要连续 60 帧都没见过才定。
        if not self.state.get("points_decided"):
            # 阈值必须低（08-09 实锤）：`奖励资讯` train=95/val=0 是**弱类**，
            #    真机 4 帧只检出 1 帧、conf 仅 **0.27**。旧阈值 0.40  永远命中
            #    不了「有」 60 帧后误判「无」 **推算整体偏一档，少打一个加成关**。
            #    用模型下限 0.20 + 底部区域限定（它固定在进度条那行 cy≈0.90）压假阳。
            #    这个 cls 该补训练样本，属于感知缺口。
            info = obs.find(V.EVENT_REWARD_INFO, 0.20, region=(0.0, 0.80, 1.0, 1.0))
            if info is not None:
                self.ctx.bag["event_has_points"] = True
                self.state["points_decided"] = True
                self.log(f"活动点数通道: 有（`奖励资讯` conf {info.conf:.2f} "
                         f"@cy={info.cy:.2f}）")
            elif self.hold("no_reward_info", 60):
                self.ctx.bag["event_has_points"] = False
                self.state["points_decided"] = True
                self.log("活动点数通道: 无（连续 60 帧没见过 `奖励资讯`）")
        # 活动任务：**红点驱动**去领（纯收入不花 AP，用户 08-09 点名"有红点
        #    不知道领取"）。红点画在按钮右上角，用 `_dot_on` 判归属 ——
        #    别全屏找红点：那一页到处都是红点，会把不相干的按钮点开。
        if (self.cfg.get("claim_tasks", True) and not self.state.get("task_done")
                and self.pending("open_task")):
            t = obs.find(V.EVENT_TASK, 0.40)
            if t is not None and self._dot_on(obs, t):
                return tap_box(t, "活动任务有红点  去领", once="open_task")

        # 活动点数奖励：**同样红点驱动**（用户 08-11 点名「奖励咨询有红点也没领取」）。
        #    `奖励资讯` 之前全仓**只被用来判断"有没有活动点数通道"**，从没被点开过 ——
        #    于是活动点数过了档位、奖励一直躺在里面。实测 5340/15000 时
        #    「活動點數 5000」那档早就达成、`領取獎勵` 亮着。
        #    阈值必须低 + 带 region：它是弱类（train 95/val 0），实测 0.807@底部，
        #      全屏 0.45 检不出（和上面那处判"有没有点数通道"用的是同一套参数）。
        if (self.cfg.get("claim_point_rewards", True)
                and not self.state.get("reward_done")
                and self.pending("open_reward")):
            r = obs.find(V.EVENT_REWARD_INFO, 0.20, region=(0.0, 0.80, 1.0, 1.0))
            if r is not None and self._dot_on(obs, r):
                return tap_box(r, "活动点数奖励有红点  去领", once="open_reward",
                               post=lambda: self.state.update(in_reward=True))

        rows, locked = self._rows(obs)
        if not rows:
            # 同样要 hold：入场后的过渡帧上整列关卡都会短暂消失
            if not self.hold("no_rows", 60):
                return wait(f"这一帧没看到可打的关（锁着 {len(locked)} 关）"
                            f"— 连续确认中")
            if locked:
                return self._end_clear(f"屏上只有 {len(locked)} 个锁着的关，没有能打的")
            return self._end_clear("关卡列表里没有入场键")

        if ph == "clear":
            return self._clear_step(obs, st, rows, locked)
        if ph in ("bonus_clear", "bonus_sweep"):
            return self._bonus_step(obs, st, rows)
        # 阶段名写错会静默变成"永远干等"。宁可当场喊出来。
        return self.finish(Outcome.UNKNOWN,
                           f"内部阶段名 '{ph}' 没有对应处理器 — 这是代码 bug")

    # ── 通关阶段 ────────────────────────────────────────────────────────
    def _clear_step(self, obs, st, rows, locked):
        # 得星_0 = 没打过。`活动站斗关卡_已打` train=0，用不了，只能这么判。
        undone = [e for e, star in rows if star == V.STAR_0]
        unknown = [e for e, star in rows if star is None]
        if undone:
            target = undone[0]                  # 从上往下 = Q1  Qn
            return tap_box(target, f"入场：本屏第一个未通关的关"
                                   f"（还有 {len(undone)} 关没打）")
        if unknown and self.state["cleared"] == 0:
            # 一颗星都没配上（可能是剧情关，没有星） 打最上面那个
            return tap_box(unknown[0], "入场：本屏第一关（这一行没有得星 cls）")
        # "没关可打"必须**连续 60 tick** 都成立才认（§A3 内容层）。
        #    点完入场键的过渡帧上，刚点的那一行会瞬时消失 —— 单帧就收工的话，
        #    每次入场都会被自己判成"打完了"（08-08 新架构第一次跑活动实锤）。
        if not self.hold("no_playable", 60):
            return wait(f"本屏暂时没有未通关的关（锁着 {len(locked)} 关）"
                        f"— 连续确认中，别被过渡帧骗了")
        if locked:
            return self._end_clear(
                f"本屏能打的关都通了（还有 {len(locked)} 关锁着，"
                f"需要推进度/場地探索解锁）")
        # 这里**不滑动**：从 1 开始打的话游戏会自动把下一关归位过来。
        return self._end_clear("全部关卡已通关")

    def _end_clear(self, why: str) -> Action:
        self.log(f"通关阶段结束: {why}")
        self.note_lines.append(f"通关阶段: {why}（{self.battle_stats()}）")
        order = self.cfg.get("order", "clear_then_bonus")
        if order == "clear_only":
            return self.finish(Outcome.CLEAN, why)
        # "推算做过没"要认**文件副本**，不能只看 ctx.bag（进程内存）：
        #    step 模式每次新进程 bag 都是空的  永远转回商店推算，进不了加成
        #    （08-09 实测死循环）。
        if self.cfg.get("shop_plan_before_bonus", True) \
                and not (self.ctx.bag.get("event_shop_plan")
                         or self._plan_from_file()):
            self.state["phase"] = "shop_plan"
            self.log(" 进活动商店推算缺哪种币（用户规则：先推算再编加成队）")
            return wait("转入商店推算阶段")
        return self._start_bonus()

    # ── 加成阶段 ────────────────────────────────────────────────────────
    #
    # **Best Record 机制**（memory event_ops_playbook:17，这条决定整个打法）:
    #    「完成關卡時套用的獎勵獲得量的最高數值會在該關卡中持續套用」
    #     用加成队**首通一次**该关，那一次的倍率就被**永久锁定**在这一关上；
    #      之后**扫荡**同一关会一直套用这个最高纪录。
    #     所以加成阶段是**两段**的：
    #         bonus_clear —— 用加成队打一次，把纪录顶上去
    #         bonus_sweep —— 扫荡同一关，把 AP 都花掉（自动套用的纪录）
    #    只做等于白打（AP 没花掉）；只做等于用旧纪录刷（倍率低）。
    #      我第一版只写了，是漏的。
    #    纪录**不可逆**：用错队伍首通，那一关这一期就永远是低倍率。
    def _plan_from_file(self):
        """推算计划的权威文件副本（event_shop 落盘, 账号桶内）。8h 内的才认。"""
        try:
            import json as _json, time as _t
            from routing_v2.config import data_dir
            f = data_dir(self.ctx.cfg) / "event_farm_plan.json"
            d = _json.loads(f.read_text(encoding="utf-8"))
            if _t.time() - float(d.get("ts", 0)) < 8 * 3600:
                return d.get("plan") or None
        except Exception:
            pass
        return None

    # ── 「哪些关的 Best Record 本期已顶过」台账（落盘，跨进程/重启）──────
    # 08-15 分桶: 键只有 from_bottom, 大小号会把对方顶过的关当成自己顶过的
    #    （顶纪录跳过 = 少打一场加成, 扫荡按旧纪录低倍率刷 —— 真金白银的 AP）。
    def _topped_path(self):
        from routing_v2.config import data_dir
        return data_dir(self.ctx.cfg) / "event_topped.json"

    def _topped_load(self) -> dict:
        # bag 优先（测试注入 fixture 用；线上没人设就读文件）——
        # 离线测试绝不能依赖 data/ 下的真实台账（08-09 已经红过一次）
        b = self.ctx.bag.get("event_topped")
        if b is not None:
            return b
        try:
            import json as _json
            return _json.loads(self._topped_path().read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _topped_mark(self, from_bottom: int, detail: str) -> None:
        """**读不出来就绝不写**（2026-08-12 数据丢失实录）。

        用户:「进活动**又去打加成了**，程序没台账逻辑记忆吗？」——
        台账 03:40 时是 4 条 `{"0","1","2","3"}`，04:51 打完一场后变回 **1 条**，
        通道 1/2/3 的纪录被整份抹掉，于是下一轮又从头打一遍加成（每关 20AP）。
        根因：原来写的是 `d = self._topped_load()` 再整份写回 —— 而
        `_topped_load()` 里 `except Exception: return {}` **把读失败吞成空字典**，
        空字典加一条再写回去，就把文件里已有的全覆盖了。
         读**直接读文件**（不走 `_topped_load`，那条会优先取测试注入的 bag），
          读失败/解析失败一律**放弃这次落账**，宁可少记一条，
          也绝不能拿一个空 dict 去覆盖真实台账。
        """
        import json as _json
        import time as _t
        # bag 注入的 fixture 台账: **只写内存, 绝不碰真实文件**。
        #    08-12 与 08-15 两次实锤同一事故: 离线套件驱动"赢一场"路径时,
        #    读走的是 bag fixture, 写却落进真 data/routing_v2/event_topped.json
        #    （08-15 02:58 的 "0" 就是测试盖的; 08-12 那次整份被覆盖后人工补回）。
        #    读写必须同源, 从写入口根治, 测试永远污染不到生产台账。
        b = self.ctx.bag.get("event_topped")
        if b is not None:
            b[str(from_bottom)] = f"{_t.strftime('%m-%d %H:%M')} {detail}"
            return
        p = self._topped_path()
        d = None
        if p.exists():
            try:
                d = _json.loads(p.read_text(encoding="utf-8"))
            except Exception as e:
                self.log(f"顶纪录台账读不出来（{e}）— **放弃本次落账**，"
                         f"绝不用空表覆盖已有记录")
                return
            if not isinstance(d, dict):
                self.log("顶纪录台账内容不是 dict — 放弃本次落账，不覆盖")
                return
        d = dict(d or {})
        d[str(from_bottom)] = f"{_t.strftime('%m-%d %H:%M')} {detail}"
        try:
            p.write_text(_json.dumps(d, ensure_ascii=False, indent=1),
                         encoding="utf-8")
            self.log(f"📒顶纪录台账已记：倒数第 {from_bottom+1} 关"
                     f"（本期共 {len(d)} 条）")
        except Exception as e:
            self.log(f"顶纪录台账落盘失败: {e}")

    def _start_bonus(self) -> Action:
        self.state["phase"] = "bonus_clear"
        self.want_team = int(self.cfg.get("bonus_team", 2))
        short = self.ctx.bag.get("event_short_currencies")
        if short:
            self.log(f"商店推算：还缺 {short}  加成队按这些币种的加成学生编")
        self.log(f"加成阶段：用部队{self.want_team} 首通，把 Best Record 顶上去")
        return wait("转入加成阶段（顶纪录）")

    def _bonus_step(self, obs, st, rows):
        # want_team 必须**每帧从配置推导**，不能只在 `_start_bonus()` 里设
        #    一次（08-09 实测）：step 模式每次都是新实例，从 state 恢复
        #    phase="bonus_clear" 时**根本不会走 _start_bonus**  want_team
        #    还是类默认的 **1（推图队）**  加成队没上场，白烧 AP。
        #    「加成阶段  部队2」是配置事实，不是一次性副作用。
        self.want_team = int(self.cfg.get("bonus_team", 2))
        ph = self.state["phase"]
        ap = R.read_topbar(obs, R.AP)
        reserve = int(self.cfg.get("ap_reserve", 0) or 0)
        need = int(self.cfg.get("min_ap_for_sweep", 20) or 20)
        if ap is None:
            # **fail-CLOSED**（08-09 血泪）：旧写法是 `ap is not None and ...`，
            #    顶栏被弹窗压暗时 AP 读不出（实测 conf 0.22） 闸整条跳过 
            #    AP 已经 0 了还去点「掃蕩開始」 游戏弹「購買AP 單價💎30」，
            #    差一步就花青辉石。**刷关是纯消耗动作，读不出就不许推进。**
            if self.hold("ap_unknown", 20):
                return self.finish(
                    Outcome.UNKNOWN,
                    f"顶栏 AP 连续 20 帧读不出  fail-closed 停手（绝不买 AP）；"
                    f"{self.battle_stats()} / 扫荡 {self.state['swept']} 次")
            return wait("顶栏 AP 这一帧读不出 — 连续确认中，先不推进")
        self.state.pop("hold:ap_unknown", None)
        if ap - reserve < need:
            return self.finish(
                Outcome.LEFTOVER,
                f"AP {ap} 不够再刷（留 {reserve}，一轮要 {need}）— 绝不买 AP；"
                f"{self.battle_stats()} / 扫荡 {self.state['swept']} 次")

        # 目标关由**商店推算**给（`event_farm_plan`），口径是"倒数第几关"：
        #   通道列表 = [活动点数(若有)] + [商店币种自下而上]，依次对上倒数第 1、2、3 关。
        #   没有推算结果时兜底打最后一关（产出最高）。
        #   bag 空（runner 崩过/step 跨进程） 读**权威文件副本**。
        plan = self.ctx.bag.get("event_farm_plan") or self._plan_from_file() or []
        if not plan:
            # 兜底目标也必须过台账（08-15 日常 live 实锤）: event_shop 被金钱闸
            #    拦在推算落盘之前 -> plan 空 -> 下面"已顶过就跳过"的 while 挂在
            #    `plan and ...` 上被整个短路 -> 对着台账里几分钟前刚顶过的
            #    倒数第 1 关**又打了一场**（20AP 白花, 用户当场看见
            #    "莫名其妙又打加成"）。合成单目标计划, 让跳过闸和扫荡闸
            #    走同一条路, 不再有裸 fb=0 旁路。
            plan = [{"from_bottom": 0, "why": "无推算结果，兜底打最后一关"}]
        # 用户口径（08-09）：**推算出的每一关都先各打一场顶好纪录，最后再统一
        #    扫荡**。`ti` = 当前在打第几个目标；每赢一场推进一个。
        ti = self.state.setdefault("target_i", 0)
        # **Best Record 是本期永久的**  顶过的关不用再顶（每次 20AP）。
        #    台账落盘（跨进程/跨重启），key = 倒数第几关。
        topped = self._topped_load()
        while (ph == "bonus_clear" and plan and ti < len(plan)
               and str(plan[ti]["from_bottom"]) in topped):
            self.log(f"↷ 倒数第 {plan[ti]['from_bottom']+1} 关的纪录本期已顶过"
                     f"（{topped[str(plan[ti]['from_bottom'])]}）— 跳过")
            ti += 1
            self.state["target_i"] = ti
        if ph == "bonus_clear" and plan and ti >= len(plan):
            self.state["phase"] = "bonus_sweep"
            self.log("所有推算目标的纪录都顶好了  转入扫荡")
            return wait("转入加成阶段（扫荡）")
        if ph == "bonus_clear" and plan and ti < len(plan):
            fb = int(plan[ti]["from_bottom"])
        else:
            fb = int(plan[0]["from_bottom"]) if plan else 0
        idx = len(rows) - 1 - fb
        if idx < 0:
            self.log(f"推算要倒数第 {fb+1} 关，但屏上只有 {len(rows)} 关  打最后一关")
            idx = len(rows) - 1
        target = rows[idx][0]
        if self.once(f"target{ti}"):
            why = plan[ti]["why"] if plan and ti < len(plan) else "无推算结果，兜底打最后一关"
            self.log(f"加成目标[{ti+1}/{max(1,len(plan))}] = 倒数第 {len(rows)-idx} 关"
                     f"（屏上第 {idx+1}/{len(rows)} 行）  {why}")

        if ph == "bonus_clear":
            # "纪录顶好了"的判据 = **真的赢了一场**，不是"我点了入场键"。
            #    2026-08-08 live 实锤：第一版把 counter 加在 tap 入场键上，
            #    然后拿 `farmed>=1` 当"打完了"  点开关卡弹窗的下一帧就跳去扫荡，
            #    **加成队一次都没上场**，780 AP 全按旧纪录（部队1 的低倍率）刷掉。
            #    这是 N5「数意图不数事实」的第 2 次复发 —— 那次数的是"决策"，
            #    这次数的是"点击"，本质一样：**都不是"这件事完成了"**。
            # 赢一场 = 这个目标的纪录顶好了  **换下一个推算目标**，
            #    全部顶完才转扫荡（用户 08-09：提前把要打的加成都打好）。
            need = max(1, len(plan))
            # 判据不能是 `win >= ti + 1`（2026-08-10 用户发现，昨天 Q11 的
            #    顶纪录台账就是这么丢的）：那是拿**全局累加的胜场数**去比目标序号，
            #    默认「win 从 0 开始单调涨到 need」。可 `win` 存在 flow 实例状态里，
            #    **中途任何一次 --fresh / flow 重建都会把它清零**  打赢第 2 个
            #    目标时 win 只有 1，`1 >= 2` 不成立  **台账永远不写**，
            #    第二天再跑就以为没顶过，白打一次（AP 浪费，纪录本身无损）。
            #     改成**每个目标记自己的胜场基线**：只要比进入这个目标时多赢一场
            #      就算顶好了，跟全局计数从几开始无关。
            base = int(self.state.setdefault("win_base", self._bt()["win"]))
            if self._bt()["win"] > base:
                self._topped_mark(fb, f"部队{self.want_team} 打赢")   # 落台账
                if ti + 1 < need:
                    self.state["target_i"] = ti + 1
                    self.state["win_base"] = self._bt()["win"]   # 下一目标的基线
                    self.log(f"加成目标[{ti+1}/{need}] 纪录已顶  打下一个目标")
                    return wait("换下一个加成目标")
                self.state["phase"] = "bonus_sweep"
                self.log(f"加成全部完成：部队{self.want_team} 打赢 "
                         f"{self._bt()['win']} 场（{need} 个目标关纪录都顶好了）"
                         f"  转入扫荡（自动套用这些最高纪录）")
                return wait("转入加成阶段（扫荡）")
            if self._bt()["unknown"] >= 3:
                return self.finish(
                    Outcome.BLOCKED,
                    f"加成打了 {self._bt()['unknown']} 场都没看到胜利横幅 —"
                    f" 纪录顶没顶上去无法确认，**拒绝扫荡**（低倍率扫荡等于浪费 AP）")
            # once 保护（08-09 实锤）：入场键点完弹窗在开、状态"没变"  重发，
            #    而列表这时会滚动  实测点成 0.732**0.573**0.732，
            #    **第二发点到了隔壁关**。
            # 08-15 复发（同一根病的另一条腿）: once 键原来带**屏上行号** idx,
            #    且 st.changed 会全清 —— 页面身份一抖, 键一换/一清, 保护就穿,
            #    16 tick 内连发两处不同的入场键。改法:
            #    ①键只带目标序号 ti（行号会漂, 不进键）;
            #    ②解锁**按事实**: 点了入场键、又在列表页攒了 40 帧还没等到
            #      关卡弹窗 = 那一发被吞了, 才重武装（计数只在列表帧涨,
            #      弹窗一开自然停, 页面抖动打不断它）。
            if not self.pending(f"enter{ti}"):
                if self.bump(f"enter_miss{ti}") >= 40:
                    self.state[f"enter_miss{ti}"] = 0
                    self.state.pop(f"once:enter{ti}", None)
                    self.log("入场键点了 40 个列表帧还没见到关卡弹窗 — "
                             "判定被吞, 重武装再点一次")
                return wait("入场键已点，等关卡弹窗打开")
            self.state[f"enter_miss{ti}"] = 0
            return tap_box(target,
                           f"加成进关顶纪录（要用部队{self.want_team}）",
                           counter="bonus_enters", once=f"enter{ti}")

        #  扫荡 —— 进这一步的前提是「**这些关的纪录确实是加成队顶的**」
        #    2026-08-12 用户实测「**体力感知没有触发**」的真根因就在这:
        #    原判据是 `本轮 win >= 1`，可**纪录已经顶过的关会被正确跳过**
        #    （台账 4 个通道全记着） 这一轮一场都不用打  `win` 恒为 0
        #     永远 BLOCKED，AP 一点花不掉、堆在 240 以上白白亏回复。
        #    **「本轮赢过」和「纪录是加成队顶的」是两件事** —— 后者才是扫荡
        #      能吃到高倍率的真前提，而它**恰恰就写在台账里**（跨进程持久化）。
        #       本轮赢过 **或** 计划里的关在台账里都顶过  放行扫荡。
        #    台账为空且本轮没赢  仍然 BLOCKED（原来的保护一分不减）。
        _topped = self._topped_load()
        # 和上面同一条兜底合成: plan 空时扫荡闸也按"倒数第 1 关"这一个通道
        #    对台账（不然 `bool(_plan)` 恒 False, 台账里明明顶过也拒绝扫荡）。
        _plan = (self.ctx.bag.get("event_farm_plan") or self._plan_from_file()
                 or [{"from_bottom": 0, "why": "无推算结果，兜底打最后一关"}])
        _all_topped = bool(_plan) and all(
            str(x["from_bottom"]) in _topped for x in _plan)
        if self._bt()["win"] < 1 and not _all_topped:
            return self.finish(
                Outcome.BLOCKED,
                "没有确认过加成队赢下一场、台账里也没有本期顶纪录的记录，"
                "拒绝扫荡（会按旧纪录的低倍率刷）")
        if self._bt()["win"] < 1:
            self.log(f"↷ 本轮没打，但台账显示这 {len(_plan)} 个通道的纪录"
                     f"本期都由加成队顶过  允许扫荡（高倍率纪录会被自动套用）")
        budget = int(self.cfg.get("max_rounds", 0) or 0)
        if budget and self.state["swept"] >= budget:
            return self.finish(
                Outcome.CLEAN,
                f"扫荡跑满 {budget} 轮；{self.battle_stats()}")
        # 扫荡也按推算目标**逐个**来（每关都顶过纪录了）：`sweep_i` 指向当前
        #    在扫哪个目标；某关扫完（AP 不够/次数满）由 on_stage_popup 推进。
        si = self.state.setdefault("sweep_i", 0)
        sidx = idx
        # 扫荡顺序 = **哪种货币少就刷哪个**（用户 08-09 拍板）：
        #    两关产的活动点数一样多，差别只在第二产出  先刷「商店还缺的那档币」
        #    对应的关，点数照拿还能补上缺口。plan 里 why 带「商店」的就是它。
        order = sorted(range(len(plan)),
                       key=lambda k: 0 if "商店" in str(plan[k].get("why", "")) else 1)
        if plan and si < len(order):
            sfb = int(plan[order[si]]["from_bottom"])
            _i = len(rows) - 1 - sfb
            if 0 <= _i < len(rows):
                sidx, target = _i, rows[_i][0]
        # once 键只带 sweep 序号（同 bonus 侧 08-15 的修法: 行号会漂不进键,
        #    解锁按「40 个列表帧等不到弹窗」的事实做, 不靠 st.changed）。
        if not self.pending(f"sweep_enter{si}"):
            if self.bump(f"sweep_miss{si}") >= 40:
                self.state[f"sweep_miss{si}"] = 0
                self.state.pop(f"once:sweep_enter{si}", None)
                self.log("扫荡入场键点了 40 个列表帧还没见到弹窗 — "
                         "判定被吞, 重武装再点一次")
            return wait("扫荡入场键已点，等关卡弹窗打开")
        self.state[f"sweep_miss{si}"] = 0
        why = (plan[order[si]].get("why", "") if plan and si < len(order) else "兜底")
        return tap_box(target,
                       f"加成进关准备扫荡（{si+1}/{max(1,len(plan))}  {why}）",
                       once=f"sweep_enter{si}")

    # ══ 弹窗 / 编队 / 战斗 ════════════════════════════════════════════
    def on_stage_popup(self, obs, st):
        """首通弹的是「章節資訊 + 進入章節」，不是扫荡面板。两种都认。

        加成阶段要走**扫荡**分支：先把数量拉 MAX（游戏会自己钳到 AP 付得起
        的次数），再点扫荡开始。

        上下文守卫（08-09 实锤）：event 冷启动时屏上可能挂着**上一条 flow
           的关卡弹窗**（jfd 千年D 的 stage_popup 长得一模一样），没进过活动
           就敢点『任務開始』= 在别人的关上开真战斗。没到过活动页先关掉它。
        """
        if not self.state.get("in_event"):
            x = obs.find(V.CLOSE_X, 0.45)
            if x is not None:
                return tap_box(x, "还没进过活动 — 这是别的 flow 残留的弹窗，关掉")
            return self.exit_step(obs) or wait("残留弹窗：等退出控件")
        if self.state["phase"] == "bonus_sweep":
            # AP 闸也要在**这一页**再判一次（08-09 血泪）：闸原来只写在
            #    `_bonus_step`（关卡列表页的处理器），而扫荡面板是另一个
            #    处理器  AP 已经 0 了照样点「掃蕩開始」 游戏弹
            #    「購買AP 單價💎30」。**花钱动作的闸必须贴着那一下点击**。
            ap_now = R.read_topbar(obs, R.AP)
            need = int(self.cfg.get("min_ap_for_sweep", 20) or 20)
            # 读不出必须 fail-CLOSED。原来写的是 `ap_now is not None and
            #    ap_now < need` —— 读不出时整条闸**直接跳过**，照样往下
            #    拉 MAX、点掃蕩開始。而扫荡面板恰恰会把顶栏压暗，读不出
            #    是常态；AP 真为 0 时那一下点出去，游戏弹的就是
            #    「購買AP 單價 30 青辉石」。同文件的 `_bonus_step` 对同一个
            #    读数是 fail-CLOSED 的 —— 一条链两套口径，就差这两行。
            if ap_now is None:
                if not self.hold("ap_unknown_pop", 20):
                    return wait("扫荡面板: AP 读不出，连续确认中")
                return self.finish(
                    Outcome.UNKNOWN,
                    "扫荡面板上 AP 读不出 — 不点掃蕩開始（AP 为 0 时点下去"
                    "弹的是購買AP 框）；"
                    f"{self.battle_stats()} / 扫荡 {self.state['swept']} 次")
            if ap_now < need:
                return self.finish(
                    Outcome.LEFTOVER,
                    f"AP {ap_now} < 一轮 {need} — 收工，**绝不买 AP**；"
                    f"{self.battle_stats()} / 扫荡 {self.state['swept']} 次")
            mx = obs.find(V.QTY_MAX, 0.20)          # 弱类，蓝 MAX 上 conf≈0.26
            if mx is not None and self.pending("sweepmax"):
                return tap_box(mx, "扫荡数量拉 MAX（游戏钳到 AP 付得起的次数）",
                               once="sweepmax")
            sw = obs.find(V.SWEEP_START, 0.35)
            if sw is not None:
                # 数量 0 时「掃蕩開始」**照样是亮的**（实测 conf 0.986，
                #    不是灰态），点下去弹的就是購買AP 框。sweep.py 早就
                #    有这道闸，活动这条链一直没有 —— 补齐。
                n = R.read_qty(obs)
                if n == 0:
                    return self.finish(
                        Outcome.LEFTOVER,
                        "步进器数量读出来是 0 — 一次也扫不了，掃蕩鍵亮着"
                        "也不点（点下去就是買 AP 的框）；"
                        f"扫荡 {self.state['swept']} 次")
                self.once_reset("sweepmax")
                return tap_box(sw, f"扫荡开始（数量 {n if n is not None else chr(63)}）",
                               counter="swept")
            if obs.has(V.CONFIRM_GREY, 0.45):
                return self.finish(
                    Outcome.LEFTOVER,
                    f"扫荡键是灰的（AP 不够/次数用尽）；"
                    f"顶纪录 {self.state['farmed']} 次，扫荡 {self.state['swept']} 次")
            if self.stalled(st, 120):
                return wait("扫荡面板里没有扫荡开始键 — 不瞎点")
            return wait("等扫荡面板")

        # 通关阶段要点的是「任务开始/进入章节」，**不是**「扫荡开始」。
        #    两个按钮同时在场时，第一版按 conf argmax 挑  两个按钮来回抢，
        #    日志里 扫荡开始 / 任务开始 交替点（08-08 实测）—— 这正是用户
        #    说的「按钮打架」。按阶段固定优先级，不靠 conf 比大小。
        b = obs.find([V.TASK_START, V.STORY_ENTER_CHAPTER], 0.35)
        if b is not None:
            return tap_box(b, f"关卡弹窗（通关阶段）: {b.cls}")
        if self.stalled(st, 90):
            return wait("关卡弹窗里没有「任务开始/进入章节」— 不瞎点扫荡")
        return wait("等关卡弹窗按钮")

    # ══ 活动任务（红点驱动，纯收入不花 AP）════════════════════════════
    def on_daily_mission(self, obs, st):
        """活动任务页 —— 页面身份复用 `daily_mission`（都是"整页领取列表"，
        08-09 实测点「活動任務」进去就判成它）。

        领法和每日领奖同口径（用户 08-09）：**全部领取** 优先，其次单项。
           领完返回活动页，不 finish（活动 flow 还没跑完）。
        """
        y = obs.find(V.CLAIM_ALL_YELLOW, 0.40) or obs.find(V.CLAIM_ACTIVE, 0.40)
        if y is not None:
            return tap_box(y, f"活动任务: 领取（{y.cls}）", counter="task_claims")
        # "领完了"要过 hold（点完的过渡帧上黄键会瞬时消失 —— §A3）
        if not self.hold("task_done", 30):
            return wait("活动任务：确认真的领完了")
        return tap_box(obs.find(V.BACK, 0.45), "活动任务领完  返回活动页",
                       post=lambda: self.state.update(task_done=True)) \
            if obs.find(V.BACK, 0.45) is not None else wait("等返回键")

    # ══ 活动点数奖励（红点驱动，纯收入不花 AP）════════════════════════
    def on_facility(self, obs, st):
        """「獎勵資訊」弹窗 —— 页面身份落在通用 `facility` 上（屏上只剩
        返回键/回大厅 + 一个弹窗），所以靠 `in_reward` 标记认领这一页，
        和 ClubFlow 进社团后那套做法同源。

        别给它单独造页面身份：判据只能靠 `领取奖励_黄`，而那个 cls
           在**战术大赛领奖**上也是 0.9+（day3 用过）—— 会串页。
        """
        if not self.state.get("in_reward"):
            return None                    # 不是我打开的，这个 facility 不归我管
        y = obs.find(V.CLAIM_REWARD_YELLOW, 0.40)
        if y is not None:
            return tap_box(y, "活动点数奖励: 領取獎勵", counter="point_claims")
        # "领完了"要过 hold（点完的过渡帧上黄键会瞬时消失 —— §A3）
        if not self.hold("reward_done", 30):
            return wait("活动点数奖励：确认真的领完了")
        x = obs.find(V.CLOSE_X, 0.45) or obs.find(V.BACK, 0.45)
        if x is not None:
            return tap_box(x, "活动点数奖励领完  关弹窗回活动页",
                           post=lambda: self.state.update(reward_done=True,
                                                          in_reward=False))
        return wait("等关闭控件")

    @staticmethod
    def _dot_on(obs, box, r: float = 0.05) -> bool:
        """`box` 身上挂着红点吗（红点画在按钮**右上角**，不是正中）。

        半径**不能是常数**（08-11 用户点名「奖励咨询有红点也没领取」的根因）：
           红点挂在框的右上角，**框越宽、红点离框心越远**。实测同一页上：
             活动任务   w=0.029  红点 dx≈+0.02   固定 0.05 够用
             奖励资讯   w=0.076  红点 **dx=+0.056**、dy=-0.038  **固定 0.05 判不到**
           于是"有红点"这件事对宽按钮永远为 False，红点驱动的领取整条失效。
         半径随框自身尺寸放大，同时保留 `r` 作为下限（小图标不受影响，不误伤）。
        """
        rx = max(r, box.w * 0.6 + 0.02)
        ry = max(r, box.h * 0.6 + 0.02)
        return any(abs(d.cx - box.cx) < rx and abs(d.cy - box.cy) < ry
                   for d in obs.all(V.DOT_RED, 0.40))

    def on_formation(self, obs, st):
        """加成阶段**必须**确认选中的是加成队才出击。
        `formation_step` 在看不到目标部队 cls 时会拒绝出击（宁可 BLOCKED）；
        「部队2  自动编队」的规则只活在 formation_step 一处（§A2）。

        `want_team` **每次都从 phase 推导**，绝不靠 `_start_bonus()` 那次
           赋值（08-09 连撞两次）：编队页是**页面派发**进来的，跨进程/跨页面时
           `_bonus_step` 根本没执行过  want_team 还是类默认 1  屏上明明
           `2部队高亮`，flow 却要"切到部队1"，加成队被换掉。
           **一次性副作用不是状态**；能从 state 推的就别存副本。"""
        ph = str(self.state.get("phase", ""))
        self.want_team = (int(self.cfg.get("bonus_team", 2))
                          if ph.startswith("bonus")
                          else int(self.cfg.get("clear_first_with_team", 1)))
        return self.formation_step(obs, st)

    def on_battle_result(self, obs, st):
        act = BattleMixin.on_battle_result(self, obs, st)
        if self.state["phase"] == "clear" and self._bt()["seen_win"]:
            self.state["cleared"] += 1
        return act

    def on_shop(self, obs, st):
        return None          # 商店推算由 event_shop flow 负责

    # ══ 阶段机（商店推算阶段把控制权交出去）════════════════════════════
    # 挂 pre_page 不覆写 decide —— overlay（对话框/奖励框）必须先处理，
    #    第一版覆写 decide 把这个顺序跳过了（架构不变量测试现在拦）。
    def pre_page(self, obs: Observation, st: StateView) -> Optional[Action]:
        # 轮播抽不中要打的活动  收工。**等真的离开那一页之后**才 finish
        #   （见 on_event_guide_hub：留在那一页收工会把下一条 flow 一起坑死）。
        if self.state.get("guide_giveup") and st.page != "event_guide_hub":
            tries = int(self.state.get("guide_hits", 0))
            return self.finish(
                Outcome.BLOCKED,
                f"轮播位上「夏萊總結算」和当期活动**共用同一个 405 入口**，"
                f"连续 {tries} 次都进到了没有活动关卡的引导型活动。"
                f"已退回 {st.page}，本轮不刷活动 —— "
                f"**绝不自作主张去打 任務/特殊任務**（刷什么是用户的策略）。"
                f"要么等轮播换一轮再跑 event，要么在前端明确指定活动")
        if self.state["phase"] == "shop_plan":
            # 交给 runner：把 event_shop flow 插到队列最前面，回来后继续。
            # 必须**结束本轮**而不是 wait（08-09 实锤）：runner 只在 flow
            #    收尾后才处理插队请求，挂着请求继续跑  插队永不发生 
            #    推算被跳过、加成目标落到"兜底打最后一关"。
            #    同一实例会被塞回队列（runner queue.insert(1, f)），state 原样。
            self.ctx.bag["request_flow"] = "event_shop"
            self.state["phase"] = "bonus_pending"
            return self.finish("HANDOFF", "交棒 event_shop 推算，回来接着打加成")
        if self.state["phase"] == "bonus_pending":
            return self._start_bonus()
        return None
