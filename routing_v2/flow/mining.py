# -*- coding: utf-8 -*-
"""剧情挖矿 —— 用户点名要的模块，**默认关，前端按钮开**。

挖什么由 `story_mining.sources` 配：羁绊剧情 / 主线剧情 / 支线剧情 / 短篇剧情。
判"这个节点看过没"靠 `剧情图标未完成`(430, train 580) vs `剧情图标已完成`(427,
train 576) —— 这两个类很结实，是这条链能成立的基础。

过场处理交给全局 interceptor（`flow/interrupt.py`）:
   BA 的过场**不吃 KEYCODE_BACK**，只能走 剧情menu  跳过故事键  确认键。
   这件事全 bot 只写一处，别在这里重复。

AP: 剧情节点要花 AP。`max_ap` 是**花掉多少**的上限（0=不限）；
   AP 不够时收工，**绝不买 AP**。
"""
from __future__ import annotations

from typing import Optional

from routing_v2.act.action import Action, swipe, tap_box, wait
from routing_v2.flow import nav
from routing_v2.flow.base import ExitMixin, Flow, Outcome
from routing_v2.flow.battle import BattleMixin, FormationMixin
from routing_v2.percept import read as R
from routing_v2.state import vocab as V

_SOURCE_CLS = {
    "主线剧情": V.STORY_MAIN,
    "短篇剧情": V.STORY_SHORT,
    "支线剧情": V.STORY_SIDE,
}


class StoryMiningFlow(FormationMixin, BattleMixin, ExitMixin, Flow):
    name = "story_mining"
    module = "story_mining"
    # 撞到「下一章節」框时点 觀看 而不是 中斷（只对这条 flow 成立）
    watch_next_chapter = True
    entry_page = "task_hall"

    def setup(self) -> None:
        self.state.update(nodes=0, src_i=0, ap0=None, scrolls=0)
        self.want_team = 1
        # 「下一章節」框点 觀看 还是 中斷 —— 这条 flow 是来看剧情的，
        #   要连着推。**这个意图属于本 flow，不能写进全局 ctx.bag**:
        #   所有 flow 在 `build()` 时就一次性构造完，`setup()` 里往 bag 一写，
        #   整轮每条 flow 的剧情逃生语义都被反转成「觀看」，而且没有清除路径。
        #   改成类属性，由 runner 按**当前**flow 读（见 runner 主循环）。

    # ── 进场 ────────────────────────────────────────────────────────────
    def on_lobby(self, obs, st):
        return nav.enter(obs, V.NAV_TASKS, "任务大厅")

    def on_task_hall(self, obs, st):
        t = obs.find(V.HUB_STORY, 0.40)
        if t is not None:
            return tap_box(t, "进入剧情")
        if self.stalled(st, 120):
            return self.finish(Outcome.UNKNOWN, "任务大厅没检出 剧情 tile")
        return wait("等 剧情 tile")

    # ── 剧情大厅：选来源 ────────────────────────────────────────────────
    def on_story_hub(self, obs, st):
        srcs = [s for s in (self.cfg.get("sources") or []) if s in _SOURCE_CLS]
        if not srcs:
            srcs = list(_SOURCE_CLS)
        i = self.state["src_i"]
        if i >= len(srcs):
            return self._wrap(f"{len(srcs)} 类剧情都扫过了")
        cls = _SOURCE_CLS[srcs[i]]
        b = obs.find(cls, 0.40)
        if b is not None:
            self.log(f"进 [{i+1}/{len(srcs)}] {srcs[i]}")
            return tap_box(b, f"进 {srcs[i]}")
        if self.stalled(st, 90):
            self.state["src_i"] += 1
            return wait(f"{srcs[i]} 的 cls 没检出  换下一类")
        return wait(f"等 {srcs[i]} 的 cls")

    # ── 大章节图  小章节（用户 2026-08-11 口述的路由规则）──────────────
    def on_story_chapter_map(self, obs, st):
        """「主线剧情」点进来是**大章节图**，不是节点图。

        这一层以前**整个不存在** —— 章节图判成 `facility`，而本 flow 没有
           `on_facility`  进去就卡死，剧情挖矿链从来没走通过。
        用户口述的规则（原话）:
           「从最左边的大章节 banner 开始（以后检查路由也是这样 **new 以及完成的
             x 坐标判断**）的小章节根据**黄点**一步步推」
          大章节 = `new` ∪ `完成` 里 **cx 最小**那个（最左 = 进度最早）
            点完右侧面板刷新，小章节 = 面板里的 `黄点` 所在那一行
        黄点只当**定位锚**，落点是"同一行往右一个行宽"——行本身没有 cls。
           这是本域唯一允许的"从锚点推落点"，因为锚和目标在同一个控件上。
        """
        # 右侧面板已经有黄点  大章节已选好，直接进小章节
        dots = [b for b in obs.all(V.DOT_YELLOW, 0.40)
                if b.cx > 0.55 and 0.15 < b.cy < 0.90]
        if dots:
            d = min(dots, key=lambda b: b.cy)       # 最上面那个 = 进度最靠前
            if self.pending("subchap"):
                # 用 `dx` 而不是 tap_at：落点仍然**挂在检出框上**（JIT 复验、
                #   require 锚点都还在）。黄点贴在小章节行的行首，行往右延伸，
                #   +0.11 落在行中部。同 tap_box 注释里 HUB 活动入口 +0.075 的先例。
                return tap_box(d, f"进小章节（黄点@{d.cx:.3f},{d.cy:.3f} 同行往右）",
                               dx=0.11, once="subchap")
            return wait("小章节已点，等页面切到节点图")

        # **`完成` 只定顺序，不当落点**（2026-08-11 第一版实现就错在这儿，
        #    live 单步一跑就露馅）。用户原话：
        #      「是**完成都不点的**，完成那个位置有 new 的话就点 new」
        #      「以后检查路由也是这样 **new 以及完成的 x 坐标判断**」
        #     两个角标的 cx **一起**用来判进度到哪了（完成的在左、没做的在右），
        #      但**能点的只有 `new`**。取 `new` 里 cx 最小的 = 最早那个没做的大章节。
        marks = [b for b in obs.all([V.NEW_MARK, V.NODE_DONE], 0.40) if b.cy < 0.85]
        news = [b for b in marks if b.cls == V.NEW_MARK]
        if not news:
            if self.stalled(st, 90):
                self.state["src_i"] += 1
                self.once_reset("subchap", "bigchap")
                done = sum(1 for b in marks if b.cls == V.NODE_DONE)
                return wait(f"章节图上没有 new（{done} 个已完成） 这一类挖完了，换下一类")
            return wait("等章节图上的 new 角标")
        lead = min(news, key=lambda b: b.cx)          # 最左的 new = 最早没做的
        if self.pending("bigchap"):
            self.log("大章节: 角标 %s  取最左的 **new** cx=%.3f"
                     % ([f"{b.cls}@{b.cx:.3f}" for b in sorted(marks, key=lambda x: x.cx)],
                        lead.cx))
            return tap_box(lead, f"选最左的**未完成**大章节（new@{lead.cx:.3f}）",
                           once="bigchap")
        return wait("大章节已点，等右侧小章节面板出黄点")

    # ── 节点图：挖未完成的 ──────────────────────────────────────────────
    def on_story_nodes(self, obs, st):
        self.once_reset("bigchap", "subchap")   # 进到节点图就重置上一层的一次性闸
        # AP 记账（**不是闸**）
        # 2026-08-11 用户纠正：「**剧情是不耗体力的**」。
        #    原来这里有一道 `ap < 10 就收工`，会把一个**免费**活动直接停掉 ——
        #    典型的"照抄别的 flow 的闸"：bounty/event 那些是真花 AP 的，剧情不是。
        #     只留记账（`max_ap` 万一将来有花 AP 的剧情节点仍能兜底），
        #      **删掉低 AP 收工那条**。
        ap = R.read_topbar(obs, R.AP)
        cap = int(self.cfg.get("max_ap", 0) or 0)
        if ap is not None:
            if self.state["ap0"] is None:
                self.state["ap0"] = ap
            if cap and (self.state["ap0"] - ap) >= cap:
                return self._wrap(f"AP 预算用满（花了 {self.state['ap0']-ap}/{cap}）")

        cap_nodes = int(self.cfg.get("max_nodes", 0) or 0)
        if cap_nodes and self.state["nodes"] >= cap_nodes:
            return self._wrap(f"挖满 {cap_nodes} 个节点")

        # 2026-08-11 live 单步抓到两个 bug，都在这四行里：
        #     **点错目标**：`剧情图标未完成` 是**状态标记不是按钮** ——
        #       点它页面纹丝不动（实测连点两次 story_nodes  story_nodes）。
        #       能点的是**同一行右侧的 `入场键`**。
        #     **没判锁**：那一行的入场键可能是 `入场键没解锁`（实测第一行
        #       cy=0.353 就是锁着的）。不判就会一直去点一个进不去的行。
        #     按 cafe「邀请键按同行 cy 锚定」那套：图标定**哪一行**，
        #      入场键定**点哪里**，锁着的整行跳过。
        undone = obs.rows(V.STORY_NODE_UNDONE, 0.40)
        enters = obs.all(V.STAGE_ENTER, 0.45)
        locked = obs.all(V.STAGE_ENTER_LOCKED, 0.45)
        for ic in undone:
            same = [e for e in enters if abs(e.cy - ic.cy) < 0.04]
            if same:
                self.state["scrolls"] = 0
                return tap_box(min(same, key=lambda e: abs(e.cy - ic.cy)),
                               f"挖节点（未完成图标@{ic.cy:.3f} 同行的入场键）",
                               counter="nodes")
            if any(abs(l.cy - ic.cy) < 0.04 for l in locked):
                self.log(f"节点 @cy={ic.cy:.3f} 未完成但**入场键没解锁**  跳过这一行")
        if undone and not enters:
            return self._wrap(f"{len(undone)} 个未完成节点全是锁着的 — 推不动了")

        # 本屏没有未完成的  往下找一屏。滑到没变化就是到底了。
        if self.state["scrolls"] < 6:
            n = self.state["scrolls"] + 1             # 计数挂 post，见 sweep.py
            return swipe(0.5, 0.72, 0.5, 0.38,
                         f"节点图下滑找未完成（第 {n} 次）",
                         post=lambda: self.state.update(scrolls=n))
        self.state["src_i"] += 1
        self.state["scrolls"] = 0
        self.log("这一类挖完了  换下一类")
        return self.exit_step(obs, prefer_close=False) or wait("等返回控件")

    # ── 弹窗 / 战斗 ─────────────────────────────────────────────────────
    def on_stage_popup(self, obs, st):
        # 同 momotalk：只认「進入章節」，**绝不碰 `任务开始`**。
        #    argmax 会让 0.99 的 `任务开始` 压过 `进入章节`，而剧情节点上出现
        #    那个键就说明我们站错了页面 —— 点下去打战斗吃 AP，AP 不够就弹
        #    「購買AP 單價💎30」（2026-08-10 arena 上实锤过一次）。
        b = obs.find(V.STORY_ENTER_CHAPTER, 0.35)
        return tap_box(b, "进入章节") if b is not None else wait("等进入章节键")

    def on_formation(self, obs, st):
        if self.cfg.get("stop_on_battle", True):
            self.log("这个节点要打战斗，配置是「遇战斗跳过」 退出该节点")
            self.note_lines.append("跳过了一个带战斗的节点")
            return self.exit_step(obs, prefer_close=False) or wait("等返回控件")
        return self.formation_step(obs, st)

    def on_confirm_dialog(self, obs, st):
        cf = obs.find(V.CONFIRM, 0.45)
        return tap_box(cf, "确认") if cf is not None else wait("等確認键")

    def _wrap(self, why):
        ap_used = ""
        if self.state["ap0"] is not None:
            ap_used = f"，AP 起点 {self.state['ap0']}"
        return self.finish(
            Outcome.CLEAN if self.state["nodes"] else Outcome.SKIPPED,
            f"{why}；挖了 {self.state['nodes']} 个节点{ap_used}")
