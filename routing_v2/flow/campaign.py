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
from routing_v2.percept import read as R
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
        # stage 配置是**可选的**（用户 2026-08-13 拍板:「找关卡还是需要
        #    digitOCR 读数字, 只是点击是 cls 主导」）。空 = 自动: 得星_0 那行
        #    的关号从屏上读, 答案按读到的关号查。配置了 = 用户点名, 读出来的
        #    必须和配置一致才进（防止在错的关上用错的答案走）。
        stage = str(self.cfg.get("stage", "") or "")
        self.state["stage"] = stage
        self.state["answer"] = grid.load_answer(stage) if stage else None
        # 断点续走: 设备上的任务已经走完前 N 步时从第 N+1 步接着走
        #    （中断任务在任务内没有可点的返回键, 拆不掉时用这个接上）。
        #    一次性参数, 用完就该清回 0。
        skip = int(self.cfg.get("skip_rounds", 0) or 0)
        if skip:
            self.state["round_i"] = skip
            self.log(f"断点续走: 跳过前 {skip} 步, 从第 {skip + 1} 步开始")

    @staticmethod
    def _parse_stage(raw, hard: bool):
        """OCR 原始串 -> 关号。"2-5" / 带噪 "12-5." 都收；解析不出返回 None。"""
        import re as _re
        if not raw:
            return None
        m = _re.search(r"(\d{1,2})-(\d)", raw)
        if m is None:
            return None
        return ("H" if hard else "") + f"{m.group(1)}-{m.group(2)}"

    # 观测: 格心跨帧累积（绿勾累积同款）。同一张地图上格子是**静态**的,
    #    单帧 conf 抖动(同图实测 0.31-0.97 帧间波动)不该抖掉我们对地图的认知。
    #    锁定**不用**战斗那套 ReID -- 格子不动, 位置累积就够; 单位离散跳格,
    #    每步重新按「正下方」绑格, 没有连续跟踪问题。
    def observe(self, obs, st) -> None:
        if self.phase not in ("grid", "walk"):
            return
        acc = self.state.setdefault("cell_acc", [])
        for x, y in grid.cells(obs, 0.35):
            if all((x - a) ** 2 + (y - b) ** 2 > 0.04 ** 2 for a, b in acc):
                acc.append((x, y))
        sp = obs.find([V.GRID_START, V.GRID_START_GREY], 0.35)
        if sp is not None and self.state.get("start_xy") is None:
            self.state["start_xy"] = (sp.cx, sp.cy)

    def _acc_cells(self, obs):
        """本帧检出 + 历史累积的格心合集。"""
        cs = grid.cells(obs, 0.35)
        for a in self.state.get("cell_acc", []):
            if all((a[0] - x) ** 2 + (a[1] - y) ** 2 > 0.04 ** 2 for x, y in cs):
                cs.append(a)
        return cs

    def _plan(self):
        a = self.state.get("answer")
        if not a or not a["areas"]:
            return []
        i = min(self.state["area_i"], len(a["areas"]) - 1)
        return a["areas"][i]["fight_plan"]

    # enter
    def do_enter(self, obs, st):
        if self.state.get("stage") and self.state.get("answer") is None:
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
        # 任務資訊框压在任务上（back 键会弹它, 三键: 中斷任務/重新挑戰/確認;
        #    中断/重挑 两个 cls 欠拟合, 框常被判成 battle_result）——
        #    点確認关掉继续, 关掉后地图露出来走上面的接管分支。
        if (st.page in ("battle_result", "unknown")
                and obs.has(V.CONFIRM, 0.45) and obs.has(V.CLOSE_X, 0.45)):
            cf = obs.find(V.CONFIRM, 0.45)
            return tap_box(cf, "关掉任務資訊框（点確認, 任务继续）")
        return wait("等任务大厅")

    # stage_list: 点**得星_0 那一行的入場键**（下一关就是没有星的那关 --
    #    游戏会自动把它归位到可视区, 老规矩）。
    #    2026-08-13 实帧纠错: `普通关卡选中` 是顶部 Normal **页签**, 不是
    #    选中的关卡行, 点它开不了弹窗。行内按钮锚定和咖啡厅邀请键同款。
    def do_stage_list(self, obs, st):
        # 弹窗开没开也看**内容证据**(页签 0.99 稳), 不只等页面签名 --
        #    复打版式上 任务开始 只有 0.22, 光靠签名会卡死(2026-08-13 实录 4001 tick)
        if (st.page == "stage_popup"
                or obs.has([V.TAB_COMMAND_SEL, V.TAB_GUIDE_SEL], 0.45)):
            self.goto("popup", "关卡弹窗开了")
            return wait("进相位 popup")
        # 页签对齐目标: 配置以 H 开头才去 Hard, 否则必须在 Normal
        #    （2026-08-13 实帧: Normal 没打完 Hard 整列 `入场键没解锁`,
        #    上一轮的 Hard 页签还留在屏上, 不切回去会干等）。
        want_hard = self.state["stage"].startswith("H")
        on_hard = obs.has(V.STAGE_HARD_SEL, 0.45)
        if want_hard and not on_hard:
            hd = obs.find(V.STAGE_HARD, 0.45)
            if hd is not None:
                return tap_box(hd, "切到 Hard 页签", expect=(V.STAGE_HARD_SEL,))
            return wait("找 Hard 页签")
        if not want_hard and on_hard:
            nm = obs.find(V.STAGE_NORMAL, 0.45)
            if nm is not None:
                return tap_box(nm, "切回 Normal 页签",
                               expect=(V.STAGE_NORMAL_SEL,))
            return wait("找 Normal 页签")
        # **Hard 永远 3 关; Hard 全锁 = Normal 没打完**（用户 2026-08-13 口述
        #    规则）。全锁列直说, 别在这一页耗。
        if (not obs.has(V.STAGE_ENTER, 0.45)
                and obs.has(V.STAGE_ENTER_LOCKED, 0.45)
                and self.hold("all_locked", 20)):
            return self.finish(Outcome.BLOCKED,
                               "这一列的入場全是锁的 -- Hard 全锁说明 Normal "
                               "没打完, 先推 Normal")
        # 行模型（用户拍板:「每一关都有入场以及对应的关号」）:
        #    行 = 星标(得星_0/得星_3) + 左上关号文本 + 右侧入場键。
        #    没配置 -> 打 `得星_0` 那行(下一关); 配置了 -> 逐行读号找那一行
        #    （复打已通关卡用: 测试走格子闭环拿已 3 星的关当靶, 零进度风险）。
        anchor = self.state.get("row_anchor")
        if anchor is None:
            stars = obs.all([V.STAR_0, V.STAR_3], 0.45)
            cfgd = self.state.get("stage")
            hard = on_hard
            cands = [b for b in stars if b.cls == V.STAR_0] if not cfgd else stars
            reads = []
            hit = None
            cache = self.state.setdefault("row_reads", {})
            for sb in cands:
                key = round(sb.cy, 2)
                if key in cache:
                    got = cache[key]
                else:
                    h = max(sb.y2 - sb.y1, 0.015)
                    rect = (sb.x1 - 1.25 * h, sb.y1 - 4.2 * h,
                            sb.x1 + 5.0 * h, sb.y1 + 0.4 * h)
                    got = self._parse_stage(R.digits(obs.frame, rect), hard)
                    cache[key] = got
                if got is None:
                    continue
                reads.append(got)
                if not cfgd or got == cfgd:
                    hit = (sb, got)
                    break
            if hit is None:
                # 目标不在可视区 -> **先扫后滑**: 按读到的号判方向翻列表
                #    （2026-08-13 live: 列表自动归位在最新进度, 配置 2-1 时
                #    可视区只有 2-2..2-5 -- 目标在上面, 要往回翻）。
                #    方向从**读到的数字**推, 几何从星标行距推, 零写死。
                if cfgd and reads and self.state.get("scrolls", 0) < 6:
                    def _k(t):
                        a, b = t.lstrip("H").split("-")
                        return (int(a), int(b))
                    up = _k(cfgd) < min(_k(r) for r in reads)
                    ys = sorted(b.cy for b in stars)
                    rowh = 0.16
                    if len(ys) >= 2:
                        gaps = [b - a for a, b in zip(ys, ys[1:]) if b - a > 0.05]
                        if gaps:
                            rowh = sorted(gaps)[len(gaps) // 2]
                    x = sorted(b.cx for b in stars)[len(stars) // 2]
                    y0 = ys[len(ys) // 2]
                    y1 = min(0.92, y0 + rowh * 2.5) if up else max(0.10, y0 - rowh * 2.5)
                    n = self.state.get("scrolls", 0) + 1

                    def _sc(k=n):
                        self.state["scrolls"] = k
                        self.state["row_reads"] = {}
                        self.state.pop("stage_vote", None)
                    from routing_v2.act.action import swipe as _swipe
                    return _swipe(x, y0, x, y1,
                                  f"目标 {cfgd} 不在可视区({'/'.join(reads)}) -- "
                                  f"往{'前' if up else '后'}翻列表（第 {n} 次）",
                                  post=_sc)
                if self.hold("stage_ocr", 30):
                    return self.finish(
                        Outcome.UNKNOWN,
                        f"逐行读号没找到目标关（配置 {cfgd or '自动'}, "
                        f"读到 {reads or '无'}）-- 不进错关")
                return wait("逐行读关号中")
            sb, got = hit
            # **两帧共识才算读到**（用户:「进的太快又没读到关卡号」）:
            #    列表刚弹出/惯性滚时单帧 OCR 是孤证, 连续两帧同号才放行。
            if self.state.get("stage_vote") != got:
                self.state["stage_vote"] = got
                return wait(f"读到 {got}, 等下一帧复读确认")
            ans = grid.load_answer(got)
            if ans is None:
                return self.finish(Outcome.BLOCKED,
                                   f"没有 {got} 的 BAAH 答案文件")
            self.state.update(stage=got, answer=ans,
                              row_anchor=(sb.cx, sb.cy))
            self.log(f"锁定目标关 = {got}"
                     f"（fight_plan {sum(len(a['fight_plan']) for a in ans['areas'])} 回合）")
            anchor = (sb.cx, sb.cy)
        rows = obs.all(V.STAGE_ENTER, 0.45)
        same = min(rows, key=lambda b: abs(b.cy - anchor[1]), default=None)
        if same is not None and abs(same.cy - anchor[1]) < 0.05:
            act = tap_box(same, f"目标关 {self.state['stage']}（同行入場）",
                          expect=(V.TASK_START, V.TASK_START_GREY))
            act.anchor_tol = 0.030      # 行内按钮容差要小于半行距
            return act
        return wait("目标行的入場键没检出（锁行等它）")

    # popup: 保证在「集中指挥」页签, 然后 任務開始
    def do_popup(self, obs, st):
        # 进没进地图不能只等页面签名（2026-08-13 live: 2 章地图格子检出
        #    断崖式掉到 0.77x1, `grid_quest` pred 不成立, page 落在 facility）。
        #    **屏上有任何格子 = 已经在地图上**, 低阈直判。
        if (st.page == "grid_quest"
                or obs.has([V.GRID_CELL, V.GRID_CELL_OPEN, V.GRID_START],
                           0.35)):
            self.goto("grid", "进地图了（格子在屏上）")
            return wait("进相位 grid")
        # 页签两态: 未选中的 `集中指挥` 是绿底(495), 选中是 421。
        #    在「简易攻略」页签上点開始会走简化模式 -- 必须先切过来。
        tab = obs.find(V.TAB_COMMAND, 0.45)
        if tab is not None and not obs.has(V.TAB_COMMAND_SEL, 0.45):
            return tap_box(tab, "切到「集中指揮」页签",
                           expect=(V.TAB_COMMAND_SEL,))
        start = obs.find(V.TASK_START, 0.45)
        if start is None:
            # 复打版式的黄色大按钮欠拟合(实测 0.22) -- 带 region 降阈值:
            #    弹窗右下那一片只有它一个大黄键, 别处不这么降
            start = obs.find(V.TASK_START, 0.18,
                             region=(0.55, 0.62, 1.0, 0.94))
        if start is not None:
            # 进关花 AP（非 premium）; 走格子地图的格子就是到达证据
            return tap_box(start, "任務開始（进关花 AP）",
                           expect=(V.GRID_CELL, V.GRID_CELL_OPEN))
        # 灰的 任務開始 有两个语义: 弹窗上灰 = AP 不够; **部署屏上灰 = 还没
        #    上队**（2026-08-13 live 误报实录: 人已经在地图上, 报了"AP不够"）。
        #    只有弹窗页签还在场时才许下 AP 结论。
        if (obs.has(V.TASK_START_GREY, 0.45)
                and obs.has([V.TAB_COMMAND, V.TAB_COMMAND_SEL, V.TAB_GUIDE,
                             V.TAB_GUIDE_SEL], 0.40)
                and self.hold("start_grey", 20)):
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
        # **真开局的事实 = PHASE結束 出现**（任務開始被 HUD 替掉）。
        #    2026-08-13 live 实锤: 任務開始那一发 tap 发出去了但游戏没收
        #    (面板收起动画期), 而 post 挂在"发出"上 -> flow 已经在 walk 相位
        #    "走"了, 屏上任務開始还亮着 -- 数意图不数事实在新代码里复发。
        if obs.has(V.PHASE_END, 0.40) or obs.has(V.PHASE_AUTO_ON, 0.40):
            self.goto("walk", "PHASE 控件出现 = 真开局了")
            return wait("进回合")
        start_btn = obs.find(V.TASK_START, 0.45)
        if start_btn is not None:
            # 严格契约: 等 PHASE 控件出现才算开了; 没收下会超时自动重发
            return tap_box(start_btn, "任務開始（部署完成）",
                           expect=(V.PHASE_END, V.PHASE_AUTO_ON,
                                   V.PHASE_AUTO_OFF))
        if obs.has(V.TASK_START_GREY, 0.35):
            # 还没上队: 点起点格（黄 = 可部署）。起点降到 0.35 -- 2 章地图
            #    整族 conf 断崖（见下面那条 UNKNOWN 的理由）。
            sp = obs.find(V.GRID_START, 0.35)
            if sp is not None:
                # 起点框位置逐帧漂(实测 tap 打在格子边缘没开出选队面板),
                #    有累积位置就用累积的(第一次见到时的稳定值)
                sxy = self.state.get("start_xy")
                if sxy is not None:
                    act = tap_box(sp, "点起点格上队(按累积位置)",
                                  expect=(V.SORTIE,))
                    act.x, act.y = sxy
                    return act
                return tap_box(sp, "点起点格上队", expect=(V.SORTIE,))
            if self.hold("no_start_cell", 40):
                # fail-closed 的**诚实版本**: 不是 AP、不是 bug, 是感知在这章
                #    地图上不够用（2-5 实测: 起点 0 检出/格子 0.77x1/敌方 3 出 1
                #    -- 训练素材全来自 1 章同一张图, [[grid_quest_baah]] 的债）。
                #    这一轮采到的帧就是补这个洞的料。
                return self.finish(
                    Outcome.UNKNOWN,
                    "部署屏上起点格检不出 -- 这章地图的走格子族泛化不足, "
                    "已采帧待标注, 不瞎点")
            return wait("找起点格（这章地图检出弱, 多看几帧）")
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
            a = self.state.get("answer") or {"areas": []}
            if self.state["area_i"] + 1 < len(a["areas"]):
                if obs.has([V.TASK_START, V.TASK_START_GREY, V.SORTIE], 0.35):
                    self.state.update(area_i=self.state["area_i"] + 1,
                                      round_i=0, issued=False)
                    self.goto("grid", f"进区域 {self.state['area_i'] + 1} 重新部署")
                    return wait("下一区域部署")
                return wait("区域间过场, 等下一张图的部署界面")
            self.goto("result", "fight_plan 走完了")
            return wait("等结算")

        # **回合时钟 = 相位循环**（镜头无关的事实）: PHASE結束 消失(敌方回合/
        #    战斗把 HUD 藏起来) 再出现 = 新回合开始。坐标相等那套到位判据
        #    已废 -- 镜头一动(开局居中/回合间/战斗前后)它就假到位（05 轮实锤:
        #    两回合"到位"在同一坐标, 队伍没真动）。
        #    前提: PHASE 自动结束已勾（walk 入口保证, 下面）。
        pe = obs.has(V.PHASE_END, 0.40)
        if pe and obs.has(V.PHASE_AUTO_OFF, 0.45):
            # 状态钮只在检出"关"态时才点（和战斗 AUTO 同款纪律）
            ao = obs.find(V.PHASE_AUTO_OFF, 0.45)
            return tap_box(ao, "勾上 PHASE 自动结束（回合时钟靠相位循环）",
                           expect=(V.PHASE_AUTO_ON,))
        if self.state.get("issued"):
            if not pe:
                self.state["cycling"] = True
                return wait("敌方回合/战斗中（PHASE 控件不在）")
            if self.state.get("cycling"):
                self.state.update(issued=False, cycling=False)
                self.state["round_i"] += 1
                self.state.pop("hold:move_wait", None)
                self.log(f"相位循环完成 - 进回合 {self.state['round_i'] + 1}")
                return wait("新回合")
            # 落子发了但相位一直没走 = 那一发没被游戏收下 -> 有界重发
            if self.hold("move_wait", 150):
                n = self.bump(f"reissue:{self.state['round_i']}")
                if n > 3:
                    return self.finish(Outcome.UNKNOWN,
                                       f"回合 {self.state['round_i'] + 1} 落子 3 次"
                                       f"都没走动 -- 交人看")
                self.state["issued"] = False
                self.log(f"落子后相位 150 tick 没循环 - 重发（第 {n} 次）")
            return wait("等相位循环")
        if not pe:
            return wait("等我方回合（PHASE 控件出现）")

        # 我方回合, 发本回合的落子。绑格**只用 我方 身体框** -- 队伍箭头浮在
        #    头顶约 1.6 行高, 拿它绑格会把身后的格子认成所在格（探针实锤:
        #    箭头绑到灰起点, 而队伍明明站在下一格）。
        # 绑格锚定链: **箭头 -> 正下方最近的我方框 -> 该框正下方的格子**。
        #    只用 我方 会中敌我单向混淆的枪（memory battle_side_confusion:
        #    敌->我 22.5%; 本关实锤: 绑回灰起点因为某个敌方立绘被检成我方）;
        #    只用箭头会把身后的格子认成所在格（箭头悬高不定, 0.5-1.6 行波动）。
        #    箭头全场唯一属于我队 -> 拿它筛掉不在其正下方的假我方框。
        arrow = obs.find(V.GRID_ARROW, 0.25)
        allies = [b for b in obs.all(V.GRID_ALLY, 0.30)]
        unit = None
        if arrow is not None and allies:
            under = [b for b in allies
                     if b.cy > arrow.cy and abs(b.cx - arrow.cx) < 0.05]
            if under:
                unit = min(under, key=lambda b: b.cy - arrow.cy)
        if unit is None and len(allies) == 1:
            unit = allies[0]          # 只有一个我方框, 没有歧义
        if unit is None:
            return wait("等 箭头+我方 同框（防敌我混淆的假我方框）")
        cs = grid.cells(obs, 0.35)
        stp = grid.steps(cs)
        if len(cs) < 2 or stp is None:
            return wait("格子检出不足, 等一帧")
        dx, dy = stp
        cur = grid.below(unit, cs, dx)
        if cur is None:
            return wait("队伍绑不到格子")
        moves = [m for m in plan[self.state["round_i"]]
                 if m.get("action") == "move"]
        if not moves:
            return self.finish(Outcome.UNKNOWN,
                               f"回合 {self.state['round_i'] + 1} 里有不认识的动作"
                               f"（exchange/portal 还没实现）-- 不瞎走")
        d = moves[0]["target"]
        # **目的格候选排除自己站的格**（05 轮实锤: 绑格偏一格时 resolve 会解析
        #    回自己 -> 点自己=no-op -> 假到位）
        goal = grid.resolve(cur, d,
                            [c for c in cs
                             if (c[0] - cur[0]) ** 2 + (c[1] - cur[1]) ** 2
                             > (0.4 * dx) ** 2],
                            dx, dy)
        if goal is None:
            if self.hold("no_goal", 20):
                return self.finish(Outcome.UNKNOWN,
                                   f"方向 {d} 落不到任何检出的格子 -- 不瞎点"
                                   f"（cur={cur} dx={dx:.3f}）")
            return wait(f"方向 {d} 暂时解析不到格子, 再看几帧")
        cell_box = min(obs.all(grid.CELL_CLS, 0.35),
                       key=lambda b: (b.cx - goal[0]) ** 2 + (b.cy - goal[1]) ** 2)
        return tap_box(cell_box,
                       f"回合 {self.state['round_i'] + 1}: 走 {d} -> "
                       f"({goal[0]:.3f},{goal[1]:.3f})",
                       post=lambda: self.state.update(issued=True, cycling=False))

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
