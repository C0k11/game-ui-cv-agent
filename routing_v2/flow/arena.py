# -*- coding: utf-8 -*-
"""战术大赛 —— 把大赛票打完。

金钱: **绝不用青辉石买票**（`purchase_caps.arena` LOCKED=0）。
   大赛商店买体力花的是**大赛币**不是青辉石，但仍走 money 步（人审）。

对手选择靠 `战术大赛对战选择区域`(92, train 2283) —— 每个对手行一个框，
点**最上面那一行**（BA 的排法是上面的对手更弱）。不用头像模型，不写死坐标。
"""
from __future__ import annotations

from routing_v2.act.action import tap_box, wait
from routing_v2.flow import nav
from routing_v2.flow.base import ExitMixin, Flow, Outcome
from routing_v2.flow.battle import BattleMixin, FormationMixin
from routing_v2.percept import read as R
from routing_v2.state import vocab as V

# 「對戰對象」详情面板会**变成死面板**（2026-08-11 live）:
#    大赛对手清单有更新倒计时（`距離更新清單 mm:ss`）。倒计时归零后清单就作废了，
#    此时面板还好端端地开着、`攻击编制` 照样 0.97 检出、落点也没错，
#    但**点下去游戏一声不响**（帧证: 01:53:1001:55:11 十帧完全同构，
#    倒计时从 01:23 一路跑到 00:00 —— 游戏在渲染、没死机，就是不接这一下）。
#    真正的出路只有：把面板关掉  回清单  游戏弹「已超過清單更新時間」通知
#     点掉通知，清单刷新（这正是当晚人工救场的路径）。
#   这里给"点了 N 次面板还在原地"一条**自己的逃生路**。判据是**数事实**:
#    `counter=` 只在 tap 真派发之后才 +1，被闸吞掉的决策不算数。
_AF_DEAD_TAPS = 3        # 攻击编制真派发这么多次面板还在  判它是死的
_AF_DEAD_ROUNDS = 3      # 关掉重挑最多来这么多轮，再多就是清单整体过期，收工

# 大赛票读数的位置约束：**底页左栏**那个「持有票券 n/5」。
# 详情面板打开时屏上有**两个** `战术大赛票`：底页左栏 (cx≈0.056, conf 0.965)
#    和面板里「2  1」的消耗角标 (cx≈0.531, conf 0.957) —— 只差 0.008，
#    `find` 是全屏 conf argmax，哪天角标赢了就拿角标当锚。
#    实测拿角标当锚 `read_ticket` 返回 None（fail-closed，不会读大），
#    但代价是 arena 判成"票数读不出"干等 60 帧收工。
#    这就是 840AP 事故那条通用铁律的又一处落点：**读数锚点必须带 region**。
_TICKET_RAIL = (0.0, 0.10, 0.30, 1.0)


class ArenaFlow(FormationMixin, BattleMixin, ExitMixin, Flow):
    name = "arena"
    module = "arena"

    def setup(self) -> None:
        self.state.update(fights=0, tickets0=None, tickets=None,
                          af_taps=0, af_dead=0)

    # **别在这里覆写 `on_ack_dialog`**（08-11 审计结论，写下来免得再来一次）:
    #    单键通知框在 arena 页**本来就能被优先处理** —— `pages.classify()` 把
    #    overlay 和底页分成两个桶各自打分（overlay 不参与"我在哪一页"的竞争），
    #    `Flow.decide()` 是 **overlay  pre_page  底页** 的顺序，而 arena 没有
    #    自己的 `on_ack_dialog`  落到 `Flow.on_ack_dialog` 点掉確認。
    #    离线复验（08-11 那帧「通知: 已超過清單更新時間」）: classify 给
    #    `<arena +ack_dialog>`，`ArenaFlow.decide` 给 `tap(确认键@0.499,0.699)`；
    #    当晚 live 日志里 arena 页上"确认（单键通知框）"也确实点成功过两次。
    #   卡住的真因**不是**通知框挡住按钮（那 10 帧里 conf 0.02 都检不出确认键），
    #    是上面那个**死面板**。覆写只会把好用的一层弄坏。

    def observe(self, obs, st) -> None:
        # **第一行必须是 super()**：`BattleMixin.observe` 也挂在这个名字上，
        #    胜利横幅是一闪而过的、就靠它逐帧粘住。这里不转发的话
        #    `win` 永远是 0 —— 正是注释里那个「Q12 又打一遍」的病根。
        super().observe(obs, st)
        # 面板不在屏上 = 上一发点进去了（或面板已被关掉） 死面板计数归零。
        # 放 `observe` 不放 `on_arena`：打完一场是从 formation/battle 页回来的，
        #    只在 arena 页清的话会带着上一轮的计数进新面板，第一下就误触逃生。
        if obs.find(V.ARENA_ATTACK_FORM, 0.45) is None:
            self.state["af_taps"] = 0

    def on_lobby(self, obs, st):
        return nav.enter(obs, V.NAV_TASKS, "任务大厅")

    def on_task_hall(self, obs, st):
        t = obs.find(V.HUB_ARENA, 0.40)
        if t is not None:
            return tap_box(t, "进入战术大赛")
        if self.stalled(st, 120):
            return self.finish(Outcome.UNKNOWN, "任务大厅没检出 战术大赛 tile")
        return wait("等 战术大赛 tile")

    def on_arena(self, obs, st):
        t = obs.find(V.TICKET_ARENA, 0.30, region=_TICKET_RAIL)
        # **领奖励不能排在打大赛前面**（2026-08-10 live 实锤，票 5/5 一场没打）:
        #    左栏「時間獎勵 **+90/分**」是**持续累积、永远领不完**的 —— 每打完一场
        #    回到这一页，又攒出一点可领  第一版把领奖励放在最前面，于是每一轮
        #    都停在领奖励上，**挑战对手那一支永远执行不到**。
        #    （这是 memory 那条「arena 干等着不打、票 55」的变体：上次是不认识
        #      攻击编制面板，这次是被一个"永动"的收入项挤掉。）
        #   有票就先打；票用完了再把剩下的奖励扫干净。
        #  奖励是纯收入，晚一点领零损失；票每天刷新，不打就是**永久浪费**。
        tix = R.read_ticket(obs, t) if t is not None else None
        # 2026-08-12 用户点名:「我们是打完了票就出去，每日领完了才想起来回来把
        #   战术大赛的两个奖励领了，**应该是第一次进入战术大赛就领取**」。
        #   和下面那条「有票先打」**不矛盾**：day3 治的是"领奖励把打大赛挤没了"
        #     （時間獎勵持续累积  每轮都停在领奖励，票 5/5 一场没打）。
        #     正确形态是 **进门先领一轮（限次、once 保护），再打票**——
        #     进门那会儿积压的就那么两三个，领完就没了，挤不掉打票。
        #   仍然限次：`entry_claims` 只在**进门这一轮**用，且上限 3；
        #     之后的持续累积走下面「没票才领」那一支，逻辑不变。
        if int(self.state.get("entry_claims", 0)) < 3:
            cl = obs.find(V.CLAIM_REWARD_YELLOW, 0.45)
            if cl is not None:
                return tap_box(cl, "进门先领大赛奖励（黄）", counter="entry_claims")
            # 进门这一轮没有可领的  直接标记做完，别每帧再找
            self.state["entry_claims"] = 3
        if not (tix is not None and tix > 0):
            # 没票（或读不出票）才去领奖励 —— 读不出时也允许领，因为领奖励
            # 是纯收入、零风险；真正 fail-closed 的是**出战**那一支（见下）。
            # **必须限次**（08-10 我自己引入又当场发现）：「時間獎勵 +90/分」
            #    是**持续累积、永远领不完**的，票打光后每隔几秒就又攒出可领的一点
            #     这一支排在 `tix<=0  收工` 前面，等于 **arena 永远收不了工**，
            #    自主跑会一直卡在大赛页刷奖励。
            #     领够 8 次就停手（今天实测一轮下来 5~7 次就把积压的领干净了），
            #      剩下的下一轮再领 —— 反正它自己会继续累积，一分钱不会丢。
            if int(self.state.get("claims", 0)) < 8:
                cl = obs.find(V.CLAIM_REWARD_YELLOW, 0.45)
                if cl is not None:
                    return tap_box(cl, "领大赛奖励（黄）", counter="claims")

        if tix is not None:
            if self.state["tickets0"] is None:
                self.state["tickets0"] = tix
                self.log(f"大赛票 {tix}")
            self.state["tickets"] = tix
            if tix <= 0:
                return self._wrap("大赛票用完了（绝不买票）")

        # 点对手行后是**对手详情面板**（页面身份同为 arena，08-09 帧证）：
        #    `攻击编制` 在场 = 详情面板开着  点它进编队。第一版不认识这个态，
        #    点了两次对手行就干等到 LEFTOVER（票 55 一场没打）。
        af = obs.find(V.ARENA_ATTACK_FORM, 0.45)
        if af is not None:
            n = int(self.state.get("af_taps", 0))
            if n >= _AF_DEAD_TAPS:
                # 死面板逃生（见文件头 _AF_DEAD_TAPS 那段）。
                rounds = int(self.state.get("af_dead", 0))
                if rounds >= _AF_DEAD_ROUNDS:
                    return self._wrap(
                        f"對戰對象面板连续 {rounds} 轮点不动"
                        f"（多半是清单已过更新时间，游戏静默拒绝）")
                x = obs.find(V.CLOSE_X, 0.55)
                if x is not None:
                    # 关面板**不花票也不花钱**，是纯撤退动作；关掉后清单会自己
                    # 弹「已超過清單更新時間」通知，那个框由基类的
                    # `on_ack_dialog` 点掉，清单随即刷新。
                    return tap_box(
                        x, f"攻击编制 已真派发 {n} 次面板纹丝不动  "
                           f"关掉面板重挑（第 {rounds + 1} 轮）",
                        counter="af_dead",
                        post=lambda: self.state.__setitem__("af_taps", 0))
                # 没找到叉叉就什么都不做 —— §A1，绝不为了"动起来"乱点。
                return wait("对手详情面板点不动，但屏上没有叉叉 — 不瞎点")
            return tap_box(af, "对手详情  攻击编制（进编队）", counter="af_taps")

        rows = obs.rows(V.ARENA_ROW, 0.40, region=(0.45, 0.10, 1.0, 0.95))
        if rows:
            # **配对策略由用户定，别自作聪明**（2026-08-10 用户当场叫停）：
            #    我看到连输三场就把落点从 `rows[0]` 改成 `rows[-1]`（挑名次最近
            #    的对手），理由听着合理，但**这是用户的策略选择不是 bug**，
            #    用户原话：「arena 不要改配对，不要自动配队」。已改回。
            #    同理不要给大赛加自动编队（那次改动也当场回滚了，见 on_formation）。
            return tap_box(rows[0], "挑战最上面那个对手", counter="fights")
        if tix is None and self.stalled(st, 60):
            return self._wrap("票数读不出  fail-closed，不出战")
        if self.stalled(st, 120):
            return self._wrap("大赛页没有对手行 cls")
        return wait("等对手列表")

    def on_formation(self, obs, st):
        # 大赛用当前编制直接出击。
        # **别复用 `_auto_form_chain`**（2026-08-10 我改了又当场回滚）：
        #    动机是对的——默认队伍在这个号上**三战全 LOSE**，票每天只有 5 张。
        #    但大赛这一页是**「攻擊編制」页**，`快速編輯` 只是右侧竖排四个入口
        #    之一（快速編輯/起始技能/部隊資訊/預設），点开后的面板结构和活动
        #    加成那条链**不一样**；`_auto_form_chain` 一上来就按活动的版面找
        #    「自動」键，找不到就 wait，而它的 `af_edit` once 已经落下 
        #    再也不会去开面板  **死等**。实测当场卡住，只能回滚。
        #     想给大赛做自动编队，得**单独适配这一页**并用票之外的方式验证，
        #      不能拿每日限量的票去试错。（待办，别再顺手复用。）
        # 2026-08-12 用户口述规则：「**在选人阶段应该点跳过战斗**，然后出
        #   `跳过战斗未选``跳过战斗`(已选中) 的 cls」。
        #   起因：当天最后一张卷进了战斗画面，agent 不知道点跳过故事键，白等。
        #    正解不是进战斗后再想办法逃，而是**出击前就把「跳過戰鬥」勾上**，
        #     大赛本来就是瞬时结算（出击后直接弹 WIN/LOSE 单键框，见 day3）。
        #   勾选是**幂等的开关**：只在检出**未选态**时点，绝不盲 toggle
        #     （[[execution_doctrine]]「状态钮绝不盲 toggle」——已勾上时再点
        #      会把它关掉，正好反过来）。
        off = obs.find(V.SKIP_BATTLE_OFF, 0.40)
        if off is not None and self.pending("skipbattle"):
            return tap_box(off, "勾上「跳過戰鬥」（出击前，免得进战斗画面干等）",
                           once="skipbattle")
        # 2026-08-12 用户实测：**出击后有 20 秒倒计时冷却**，冷却没走完再点
        #    出击只会弹提示框  bot 反复「点出击点掉提示」空转。
        #    这跟上面勾「跳過戰鬥」**直接冲突**：勾了战斗瞬间结束，倒计时反而
        #      没走完（这次没勾，看过场动画正好耗掉冷却，误打误撞对上了）。
        #    **屏上没有倒计时的 cls**（master 表里一条都没有） 感知层暂时
        #      看不见这个冷却，只能用**显式墙钟**兜住。
        #      [[tick_vs_wallclock]]：墙钟必须显式 mark，绝不能拿 tick 计数当秒。
        #     待补 cls：大赛出击倒计时（补上之后就能改成"看得见才点"，
        #      把这个墙钟兜底撤掉）。
        import time as _t
        last = self.state.get("sortie_ts")
        if last is not None and _t.time() - last < 20.0:
            return wait(f"大赛出击冷却中（还剩 {20.0 - (_t.time() - last):.0f}s）"
                        f" — 现在点只会弹提示框")
        go = obs.find(V.SORTIE, 0.45)
        if go is None:
            return wait("等出击键")
        return tap_box(go, "出击（大赛）",
                       post=lambda: self.state.__setitem__("sortie_ts", _t.time()))

    # **删掉了 `on_stage_popup`**（2026-08-10 live 差一步花 30 青辉石）。
    #    原实现是:
    #        b = obs.find([V.TASK_START, V.SWEEP_START], 0.35)
    #        return tap_box(b, "开始") if b else wait("等开始键")
    #    三重错，每一重单独都够出事:
    #      **战术大赛流程里根本没有关卡弹窗**（它是 选对手  攻击编制  战斗）。
    #        这个 handler 等于让 arena 在**别的 flow 的关卡弹窗**上乱点 ——
    #        实测：event 扫荡完收工停在活动关卡 11 的 stage_popup 上，arena 接手，
    #        一上来就点了「任务开始」。
    #      `find([TASK_START, SWEEP_START])` 是 **conf argmax**，而
    #        `任务开始`(0.99) 常压过 `扫荡开始`(0.98)  专挑真打战斗那个键。
    #      **没有任何 AP 闸**：当时 AP=9，点下去游戏直接弹「購買AP 單價💎30」，
    #        全靠 money 闸 halt 才没成交（青辉石 20,916 一分没动）。
    #     不认识的页面就**什么都不做**，交给 nav 逐层退出。这才是 §A1。
    def on_confirm_dialog(self, obs, st):
        # 框里出现青辉石价签一律取消（arena 这条链只花大赛票，绝不花石头）。
        pyx = obs.find(V.PYROXENE, 0.40, region=(0.12, 0.12, 1.0, 1.0))
        if pyx is not None:
            c = obs.find(V.CANCEL, 0.45)
            if c is not None:
                return tap_box(c, f"框内有青辉石价签（conf {pyx.conf:.2f}） 取消")
            return wait("框内有青辉石价签但没找到取消键 — 绝不点確認")
        cf = obs.find(V.CONFIRM, 0.45)
        return tap_box(cf, "确认") if cf is not None else wait("等確認键")

    def _wrap(self, why):
        t0, t1 = self.state["tickets0"], self.state["tickets"]
        det = f"打 {self.state['fights']} 场，{self.battle_stats()}"
        if t0 is not None and t1 is not None:
            det += f"，票 {t0}{t1}"
        if t1 is not None and t1 > 0:
            return self.finish(Outcome.LEFTOVER, f"{why}；{det}")
        if t1 is None:
            return self.finish(Outcome.UNKNOWN, f"{why}；{det}（票数读不出）")
        return self.finish(Outcome.CLEAN, f"{why}；{det}")
