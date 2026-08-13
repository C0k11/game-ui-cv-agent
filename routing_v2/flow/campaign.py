# -*- coding: utf-8 -*-
"""任務推图（走格子, 集中指挥模式）-- 按 BAAH 答案走一关。

三层分工:
   感知层  v16 的走格子族 cls（小号实测 0.85-0.98）
   几何层  flow/grid.py（方向语义 -> 本帧检出的格心, 真实帧 6/6 验证）
   答案层  data/baah_grid_solution/{stage}.json（MIT, normal 150 + Hard 90）

**打哪一关是用户的策略**（cfg campaign.stage）, bot 只负责走。
战斗期靠游戏内 AUTO: 检出 `自动战斗关闭` 才点开（状态钮绝不盲 toggle）。
BAAH 是盲走, 我们每步都验: 落点取检出的真格心, 到达靠"队伍绑格 == 目标格"。
"""
from __future__ import annotations

from routing_v2.act.action import Action, tap_box, wait
from routing_v2.flow import grid, nav
from routing_v2.flow.base import ExitMixin, Flow, Outcome
from routing_v2.percept.observe import Observation
from routing_v2.state import vocab as V
from routing_v2.state.machine import StateView

# 队伍绑格和目标格心的判等距离（相对列步长的比例, 不是绝对值）
REACH = 0.45


class CampaignFlow(ExitMixin, Flow):
    name = "campaign"
    module = "campaign"
    entry_page = "task_hall"
    phases = ("enter", "stage_list", "popup", "grid", "walk", "result")
    # 战斗一场 2-4 分钟, 默认 600 tick 会把打着一半的仗判成卡死
    phase_cap = 4000

    def setup(self) -> None:
        self.state.update(round_i=0, area_i=0, target=None, need_end=False,
                          battles=0)
        stage = str(self.cfg.get("stage", "") or "")
        self.state["stage"] = stage
        self.state["answer"] = grid.load_answer(stage) if stage else None

    def _plan(self):
        a = self.state.get("answer")
        if not a or not a["areas"]:
            return []
        i = min(self.state["area_i"], len(a["areas"]) - 1)
        return a["areas"][i]["fight_plan"]

    # enter
    def do_enter(self, obs, st):
        if not self.state.get("stage"):
            return self.finish(Outcome.BLOCKED,
                               "没配 campaign.stage -- 打哪关是用户的策略, 不猜")
        if self.state.get("answer") is None:
            return self.finish(Outcome.BLOCKED,
                               f"没有 {self.state['stage']} 的 BAAH 答案文件")
        if st.page == "lobby":
            return nav.enter(obs, V.NAV_TASKS, "任务大厅")
        if st.page == "task_hall":
            t = obs.find(V.HUB_CAMPAIGN, 0.45)
            if t is not None:
                return tap_box(t, "进 任務 推图",
                               expect=(V.STAGE_NORMAL_SEL, V.STAGE_NORMAL))
            return wait("找 任務 磁贴")
        if st.page == "campaign_stage":
            self.goto("stage_list", "到关卡列表了")
            return wait("进相位 stage_list")
        if st.page == "grid_quest":
            # 上一轮没退干净, 直接从地图接手
            self.goto("grid", "已经在走格子地图上")
            return wait("进相位 grid")
        return wait("等任务大厅")

    # stage_list: 点**得星_0 那一行的入場键**（下一关就是没有星的那关 --
    #    游戏会自动把它归位到可视区, 老规矩）。
    #    2026-08-13 实帧纠错: `普通关卡选中` 是顶部 Normal **页签**, 不是
    #    选中的关卡行, 点它开不了弹窗。行内按钮锚定和咖啡厅邀请键同款。
    def do_stage_list(self, obs, st):
        if st.page == "stage_popup":
            self.goto("popup", "关卡弹窗开了")
            return wait("进相位 popup")
        star0 = obs.find(V.STAR_0, 0.45)
        if star0 is not None:
            rows = obs.all(V.STAGE_ENTER, 0.45)
            same = min(rows, key=lambda b: abs(b.cy - star0.cy), default=None)
            if same is not None and abs(same.cy - star0.cy) < 0.05:
                act = tap_box(same, f"没打过的关（得星_0 同行入場; "
                                    f"配置目标 {self.state['stage']}）",
                              expect=(V.TASK_START, V.TASK_START_GREY))
                act.anchor_tol = 0.030      # 行内按钮容差要小于半行距
                return act
            return wait("看到得星_0 但同行入場键没检出")
        if self.phase_ticks > 90:
            return self.finish(
                Outcome.UNKNOWN,
                "列表上找不到 得星_0 的关（都打过了? 还是要翻页?）-- 不瞎点")
        return wait("等关卡列表/找得星_0")

    # popup: 保证在「集中指挥」页签, 然后 任務開始
    def do_popup(self, obs, st):
        if st.page == "grid_quest":
            self.goto("grid", "进地图了")
            return wait("进相位 grid")
        # 页签两态: 未选中的 `集中指挥` 是绿底(495), 选中是 421。
        #    在「简易攻略」页签上点開始会走简化模式 -- 必须先切过来。
        tab = obs.find(V.TAB_COMMAND, 0.45)
        if tab is not None and not obs.has(V.TAB_COMMAND_SEL, 0.45):
            return tap_box(tab, "切到「集中指揮」页签",
                           expect=(V.TAB_COMMAND_SEL,))
        start = obs.find(V.TASK_START, 0.45)
        if start is not None:
            # 进关花 AP（非 premium）; 走格子地图的格子就是到达证据
            return tap_box(start, "任務開始（进关花 AP）",
                           expect=(V.GRID_CELL, V.GRID_CELL_OPEN))
        if obs.has(V.TASK_START_GREY, 0.45) and self.hold("start_grey", 20):
            return self.finish(Outcome.BLOCKED,
                               "弹窗里 任務開始 是灰的（AP 不够？）-- 不硬点")
        return wait("等弹窗控件")

    # grid: 部署 -- 点起点上队, 然后 任務開始
    def do_grid(self, obs, st):
        # 点了起点会弹编队页(出击键) -- 相位机下页面 handler 不跑, 在这处理
        if st.page == "formation" or obs.has(V.SORTIE, 0.45):
            s = obs.find(V.SORTIE, 0.45)
            if s is not None:
                return tap_box(s, "编队确认: 出击（队伍上到起点格）",
                               expect=(V.TASK_START, V.TASK_START_GREY))
            return wait("编队页, 等出击键")
        start_btn = obs.find(V.TASK_START, 0.45)
        if start_btn is not None:
            def _go():
                self.goto("walk", "任務開始, 进回合了")
            return tap_box(start_btn, "任務開始（部署完成）", post=_go)
        if obs.has(V.TASK_START_GREY, 0.45):
            # 还没上队: 点起点格（黄 = 可部署）
            sp = obs.find(V.GRID_START, 0.45)
            if sp is not None:
                return tap_box(sp, "点起点格上队", expect=(V.SORTIE,))
            return wait("找起点格")
        return wait("等部署界面")

    # walk: 按 fight_plan 逐回合走
    def do_walk(self, obs, st):
        # 战斗期: 时钟重置(一场 2-4 分钟), 只管把 AUTO 打开
        if st.page == "battle":
            self._phase_t0 = self.ticks - 1
            off = obs.find(V.BATTLE_AUTO_OFF, 0.50)
            if off is not None and self.pending(f"auto{self.state['battles']}"):
                return tap_box(off, "AUTO 是关的 - 打开（状态钮只在检出关态时才点）",
                               once=f"auto{self.state['battles']}",
                               expect=(V.BATTLE_AUTO_ON,))
            return wait("战斗中（AUTO）")
        if st.page == "battle_result":
            cf = obs.find(V.CONFIRM, 0.45)
            if cf is not None:
                def _won():
                    self.state["battles"] += 1
                return tap_box(cf, "战斗结算: 確認", post=_won)
            return wait("等结算確認")
        if st.page == "campaign_stage":
            # 走完之前就回到列表 = 关卡打完被送出来了
            self.goto("result", "回到关卡列表")
            return wait("进相位 result")
        if st.page != "grid_quest":
            return wait(f"过场/动画（page={st.page}）")

        plan = self._plan()
        if self.state["round_i"] >= len(plan):
            self.goto("result", "fight_plan 走完了")
            return wait("等结算")

        cs = grid.cells(obs)
        stp = grid.steps(cs)
        if len(cs) < 2 or stp is None:
            return wait("格子检出不足, 等一帧")
        dx, dy = stp
        unit = (obs.find(V.GRID_ARROW, 0.45) or obs.find(V.GRID_ALLY, 0.45)
                or obs.find(V.GRID_START, 0.45))
        if unit is None:
            return wait("找不到我方队伍标记")
        cur = grid.below(unit, cs, dx)
        if cur is None:
            return wait("队伍绑不到格子")

        tgt = self.state.get("target")
        if tgt is not None:
            if (cur[0] - tgt[0]) ** 2 + (cur[1] - tgt[1]) ** 2 < (REACH * dx) ** 2:
                self.state.update(target=None, need_end=True)
                self.state["round_i"] += 1
                self.log(f"回合 {self.state['round_i']} 移动到位 {tgt}")
                return wait("移动到位")
            return wait(f"移动中 -> {tgt}")

        if self.state.get("need_end"):
            if obs.has(V.PHASE_AUTO_ON, 0.45):
                self.state["need_end"] = False
                return wait("PHASE 自动结束已勾, 不用点")
            pe = obs.find(V.PHASE_END, 0.45)
            if pe is not None:
                return tap_box(pe, "PHASE結束（本回合走完了）",
                               post=lambda: self.state.update(need_end=False))
            return wait("等 PHASE 控件/敌方回合")

        # 我方回合: 发本回合的移动。多队/exchange/portal 先不做, fail-closed。
        moves = [m for m in plan[self.state["round_i"]]
                 if m.get("action") == "move"]
        if not moves:
            return self.finish(Outcome.UNKNOWN,
                               f"回合 {self.state['round_i']} 里有不认识的动作"
                               f"（exchange/portal 还没实现）-- 不瞎走")
        d = moves[0]["target"]
        goal = grid.resolve(cur, d, cs, dx, dy)
        if goal is None:
            if self.hold("no_goal", 20):
                return self.finish(Outcome.UNKNOWN,
                                   f"方向 {d} 落不到任何检出的格子 -- 不瞎点"
                                   f"（cur={cur} dx={dx:.3f}）")
            return wait(f"方向 {d} 暂时解析不到格子, 再看几帧")
        cell_box = min(obs.all(grid.CELL_CLS, 0.45),
                       key=lambda b: (b.cx - goal[0]) ** 2 + (b.cy - goal[1]) ** 2)
        return tap_box(cell_box,
                       f"回合 {self.state['round_i'] + 1}: 走 {d} -> "
                       f"({goal[0]:.3f},{goal[1]:.3f})",
                       post=lambda g=goal: self.state.update(target=g))

    # result: 等结算链把我们送回关卡列表
    def do_result(self, obs, st):
        if st.page == "battle":
            self._phase_t0 = self.ticks - 1
            return wait("收尾战斗中")
        if st.page == "battle_result":
            cf = obs.find(V.CONFIRM, 0.45)
            return tap_box(cf, "结算確認") if cf is not None else wait("等結算")
        if st.page == "campaign_stage":
            n = self.state["round_i"]
            return self.finish(
                Outcome.CLEAN,
                f"{self.state['stage']} 按答案走完 {n} 回合, "
                f"战斗 {self.state['battles']} 场, 已回到关卡列表")
        cf = obs.find(V.CONFIRM, 0.40)
        if cf is not None:
            return tap_box(cf, "结算页: 確認")
        if self.phase_ticks > 900:
            return self.finish(Outcome.UNKNOWN,
                               "结算后 900 tick 没回到关卡列表 -- 交人看")
        return wait("等结算/回列表")

    def on_confirm_dialog(self, obs, st):
        cf = obs.find(V.CONFIRM, 0.45)
        return tap_box(cf, "确认") if cf is not None else wait("等確認键")
