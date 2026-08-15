# -*- coding: utf-8 -*-
"""战斗链 + 编队 —— 活动/悬赏/JFD/大赛/推图/挖矿全都复用这一份。

链路 2026-08-08 真机逐帧验过（10 次出击 / 3.6 分钟，全 cls 驱动零 wait）:
    编队(出击)  战斗内(三倍速/暂停/自动战斗开启)  **战斗胜利** 
    确认键  跳过故事键  获得奖励  回到入场键

 `战斗失败` **train=0，UI 模型一框都没有  输了看不见**。
   所以胜负判定是这样的：看见 `战斗胜利`/`战斗完成` 才算 WIN，其余一律
   **UNKNOWN，不许记成"打过了"**。谎报 CLEAN 比报 UNKNOWN 危险得多 ——
   下游会据此跳过这一关。

编队铁律（用户规则，memory event_ops_playbook）:
   · **首通用部队1（速推主力）** —— 活动关卡的 Best Record 会把首通时的
     加成倍率**永久锁定**在那一关上，用错队伍是不可逆的损失。
   · 加成队 = 部队2，且要**先看商店推算缺哪种币**再编。
    这里的 `formation_step` 在**确认不了当前选的是哪支部队时拒绝出击**，
     宁可报 BLOCKED 也不赌。
"""
from __future__ import annotations

from typing import Optional

from routing_v2.act.action import Action, tap_box, wait
from routing_v2.percept import read as R
from routing_v2.percept.observe import Observation
from routing_v2.state import vocab as V
from routing_v2.state.machine import StateView

# 编队屏左下「活動編輯效果」格。老码 `_R_SQUAD_PCT` 原值，08-09 实测复核：
# 真值 105%/110% 两格  读出 `'105110'`（两格连读，不拆）。判据"只认干净的 0"
# 在这种连读下仍然安全：有任何加成就不可能是纯 "0"。
_R_SQUAD_PCT = (0.005, 0.92, 0.085, 0.99)


class FormationMixin:
    """编队页处理。子类给 `want_team` 属性（1-4）。

    用户定死的全局约定（2026-08-09，学费 840AP）：
       **部队1 = 推图队；部队2 = 加成专用队，选了部队2就必然自动编队**。
       自動按「当前关」优化且各关共享同一个部队2  每一关都要重排
       （老 event_quest.py:14 实测：Q10 优化后的同队对 Q11 只有 55%）。
       光切到部队2 ≠ 加成队就绪 —— 队里是上次的旧阵容；漏了自动编队，
       Best Record 按弱加成阵容锁定，之后每一发扫荡都吃亏。
    """
    want_team: int = 1
    _FORM_GIVEUP_FRAMES = 120

    def formation_step(self, obs: Observation, st: StateView,
                       team: Optional[int] = None) -> Optional[Action]:
        team = team or self.want_team
        auto_form = (team == 2)          # 唯一定义点：部队2  自动编队
        tab, hi = V.SQUAD_TABS.get(team, (None, None))
        if tab is None:
            return wait(f"部队{team} 没登记")

        # 已经选中目标部队  （部队2 先走自动编队子链） 出击
        if hi and obs.has(hi, 0.45):
            if auto_form:
                act = self._auto_form_chain(obs)
                if act is not None:
                    return act
            go = obs.find(V.SORTIE, 0.45)
            if go is not None:
                if auto_form:
                    # 加成% 闸（老 event_quest.py:1437 原样搬）：编队屏左下
                    #   「活動編輯效果」第一格。这个读数很脏（"90%"读成'06'）
                    #    只认**干干净净的 0** 才拒绝出击，脏读一律保守出击。
                    pct = R.digits(obs.frame, _R_SQUAD_PCT)
                    clean = (pct or "").strip().rstrip("%")
                    if clean == "0":
                        return self.finish(
                            "BLOCKED",
                            f"自动编队后加成仍是干净的 0%（raw={pct!r}）— 不出击")
                def _new_episode():
                    self._bt()["seen_win"] = False      # 新一场，横幅重新算
                return tap_box(go, f"⚔ 出击（部队{team}）", counter="sorties",
                               post=_new_episode)
            return wait("已选中目标部队，等出击键")

        # 没选中  切过去
        t = obs.find(tab, 0.45)
        if t is not None:
            return tap_box(t, f"切到部队{team}")

        # 3 部队 / 4 部队没有高亮态 cls：只能靠"未选中态消失"反推，不可靠。
        #  只要选中态确认不了，就**不出击**。
        if st.frames_in_page >= self._FORM_GIVEUP_FRAMES:
            return self.finish(
                "BLOCKED",
                f"编队页确认不了当前是不是部队{team}（`{tab}`/`{hi}` 都没检出）"
                f" — 拒绝盲目出击。用错队伍会把 Best Record 永久锁死。")
        return wait(f"编队页: 等 部队{team} 的 cls 出现")

    def on_squad_quick_edit(self, obs: Observation, st: StateView) -> Optional[Action]:
        """快速編輯面板（专属页面身份，见 pages.py）：走自动编队子链。
        这里**绝不能**落到通用的"点確認"处理器 —— 那会把旧阵容原样提交。"""
        act = self._auto_form_chain(obs)
        if act is not None:
            self.state["qe_close_wait"] = 0
            return act
        # 链已走完但面板还开着  只点確認提交（此时自動已经点过了）
        cf = obs.find(V.CONFIRM, 0.45)
        if cf is not None:
            self.state["qe_close_wait"] = 0
            return tap_box(cf, "加成编队: 確認（提交）")
        # 確認已提交、面板自收动画/页面身份滞后的窗口。**这里绝不能返回 None**:
        #    None 会落到 nav 归位, 它在"弹窗页"上找不到叉叉就点返回键 —— 和
        #    面板自收赛跑, 输了就把编队页整层退掉（08-15 日常 live 实锤:
        #    確認 -> 被踢回关卡列表 -> 重进 -> 再編隊, 连环三圈全是白工,
        #    每圈 ~40s; 赢了才轮到出击）。面板正常 <1s 自收; 有界等待,
        #    真卡死（60 个本页 tick）才放回老路兜底。
        if self.bump("qe_close_wait") >= 60:
            self.state["qe_close_wait"] = 0
            return None
        return wait("確認已提交，等快速编辑面板自己收起")

    def _auto_form_chain(self, obs: Observation) -> Optional[Action]:
        """加成自动编队子链（老 event_quest.py:1363 子步机原样搬）：

            快速編輯  自動  自動保险补点（幂等，防"自動被稳定门吞"） 確認

        走完返回 None（落回出击分支）。once 键由 runner 在 tap 落地后才消耗，
        被闸吞掉会自动重试（N5 数事实契约）。"""
        if not self.pending("af_confirm"):
            return None                              # 链已走完
        if self.pending("af_edit"):
            qe = obs.find(V.SQUAD_QUICK_EDIT, 0.40)
            if qe is not None:
                return tap_box(qe, "加成编队: 快速編輯", once="af_edit")
            return None      # 快速編輯键还没渲染出来 —— 落回等出击帧也无害
        if self.pending("af_auto"):
            au = obs.find(V.SQUAD_AUTO_EDIT, 0.40)
            if au is not None:
                return tap_box(au, "加成编队: 自動（按本关最优填充）",
                               once="af_auto")
            # 2026-08-12 live 死锁（day3 在 arena 上发生过，这次 event 复发）:
            #    `快速編輯` 点出去了、`once:af_edit` 已经落下，但**面板没打开**
            #    （点击被吞 / 游戏卡） 找不到「自動」键  原来这里是裸 `wait`，
            #    于是永远等下去 —— 实测卡满 3 分钟、体力 999 一点没消耗，
            #    而屏上 `出击` conf 0.978 一直亮着。
            #    修的方向是**重试打开面板**，不是跳过编队 —— 活动加成队的自动
            #      编队是设计内功能（老 event_quest.py:1363 搬来的）。
            #      用户 2026-08-10 叫停的是**战术大赛**那边的自动编队
            #        （arena 也有队伍和快速编队，那边不许点进去自动编成），
            #        **别把那条禁令泛化到活动加成队上**。
            #     等不到就清掉 `af_edit` 重点一次「快速編輯」；重试 3 轮仍不行
            #      才放弃整条链落回出击（终极逃生，防死等）。
            if self.hold("af_auto_missing", 40):
                tries = self.state.get("af_retry", 0) + 1
                self.state["af_retry"] = tries
                self.state.pop("hold:af_auto_missing", None)
                self.state.pop("hold:af_auto_missing:t", None)
                if tries <= 3:
                    self.state.pop("once:af_edit", None)   # 允许重开面板
                    self.log(f"等不到「自動」键 — 面板多半没打开，"
                             f"重点一次「快速編輯」（第 {tries}/3 次）")
                    return None
                self.state["once:af_auto"] = True
                self.state["once:af_ins"] = True
                self.state["once:af_confirm"] = True
                self.log("重开 3 次仍等不到「自動」键 — 放弃自动编队链，"
                         "用当前编队出击（不再死等）")
                return None
            return wait("编辑面板: 等 自動 键")
        # 自動保险（老码 2026-07-24 方案C）：自動+確認同屏共存，自動的点击
        #   被稳定门吞过两次 live 实锤  確認前无条件补点一次自動（幂等）。
        if self.pending("af_ins"):
            au = obs.find(V.SQUAD_AUTO_EDIT, 0.40)
            if au is not None:
                return tap_box(au, "加成编队': 自動保险补点（幂等）",
                               once="af_ins")
        cf = obs.find(V.CONFIRM, 0.45)
        if cf is not None:
            return tap_box(cf, "加成编队: 確認（提交自动编成）",
                           once="af_confirm")
        return wait("编辑面板: 等 確認 键")


class BattleMixin:
    """战斗内 + 结算 + 奖励。"""

    def _bt(self) -> dict:
        return self.state.setdefault("battle", {"win": 0, "unknown": 0,
                                                "seen_win": False})

    def observe(self, obs: Observation, st: StateView) -> None:
        """胜利横幅是**一闪而过**的，而且常出现在 page=unknown 的帧上
        （08-08 实测 `战斗胜利` conf 0.99 就在一个 unknown 帧里）。
        所以在这里**逐帧粘住**，别等結算頁才看 —— 那时横幅早没了。

        win 计数也必须在**这里**转正（08-09 live 实锤）：原来写成
        「on_battle_result 里 seen_win  win+=1」，但结算屏大多数帧被判成
        unknown/blank，確認键全走了 ack_dialog/reward 的处理器 
        on_battle_result **一次都没跑到**  win 永远 0  加成阶段以为
        没赢过，把 Q12 又打了一遍（会无限循环）。
        横幅锁存的那一刻就是胜利事实；`seen_win` 在**下一次出击**时清。"""
        win_sig = obs.has([V.BATTLE_WIN, V.BATTLE_COMPLETE], 0.40)
        # **补一条不依赖横幅的胜利判据**（2026-08-10 live 实锤）：
        #    `战斗胜利`(train 33) / `战斗完成`(train 90) 都是弱类，横幅只闪几帧，
        #    逐帧 step 或 tick 稍慢就整场错过 —— 实测这一场**活動點數 27152740、
        #    AP 909889 明明打赢了，`win` 却还是 0**，flow 于是要把同一关再打一遍
        #    （和注释里那次 Q12 无限循环同形，只是换了个漏检的入口）。
        #     结算页上出现 `获得奖励`（train 量大、天天在用）= 这一场有产出 = 赢了。
        #    必须**限定在结算页**：`获得奖励` 在领取/购买链里到处都是，
        #      不限页面会把"领了个邮件"也算成打赢一场。
        if not win_sig and st.page == "battle_result" \
                and obs.has(V.GOT_REWARD, 0.40):
            win_sig = True
            self.log(" 横幅没抓到，但结算页有「獲得獎勵」 判定这一场赢了")
        if win_sig:
            bt = self._bt()
            if not bt["seen_win"]:
                bt["seen_win"] = True
                bt["win"] += 1
                self.log(f" 捕到胜利  第 {bt['win']} 场胜利"
                         f"（page={st.page}）")

    # ── 战斗内 ──────────────────────────────────────────────────────────
    def on_battle(self, obs: Observation, st: StateView) -> Optional[Action]:
        bt = self._bt()
        # 三倍速：只在明确看到 1x/2x 时才切（看到 3x 说明已经是了）
        if not obs.has(V.BATTLE_3X, 0.45):
            spd = obs.find([V.BATTLE_1X, V.BATTLE_2X], 0.45)
            if spd is not None and self.pending("speed3x"):
                return tap_box(spd, "战斗: 切三倍速", once="speed3x")
        # 自动战斗：只在明确看到"关闭"态时才开
        off = obs.find(V.BATTLE_AUTO_OFF, 0.45)
        if off is not None and self.pending("autoon"):
            return tap_box(off, "战斗: 开自动战斗", once="autoon")
        return wait("战斗中 — 等结算")

    # ── 结算 ────────────────────────────────────────────────────────────
    def on_battle_result(self, obs: Observation, st: StateView) -> Optional[Action]:
        # win 计数在 observe() 里转正（结算页身份靠不住，这里常跑不到）。
        #    这里只负责：没见过横幅的场次记"胜负未知" + 点確認。
        bt = self._bt()
        cf = obs.find(V.CONFIRM, 0.55, region=(0.0, 0.80, 1.0, 1.0))
        if cf is None:
            return wait("结算页: 等確認键")
        self.once_reset("speed3x", "autoon")

        def _mark():
            # `unknown += 1` 必须挂 post（08-09 实锤）：结算页上
            #    `on_battle_result` **每帧**都会被调用，写在 return 之前
            #     每帧 +1  「打了 20 场都没看到胜利横幅」这种假数字，
            #    还会误触发 BLOCKED 把扫荡拒了。数的是 decide 调用次数，
            #    不是场次 —— 「数意图不数事实」又一次。
            if not self._bt()["seen_win"]:
                self._bt()["unknown"] += 1
                self.log(f"这一场**胜负未知**（`战斗失败` train=0，输了看不见）"
                         f"— 不记为已通关")
        return tap_box(cf, "结算: 確認", post=_mark)

    # ── 奖励页 ──────────────────────────────────────────────────────────
    def on_reward(self, obs: Observation, st: StateView) -> Optional[Action]:
        """奖励层出口收口到 `Flow._reward_exit`（08-15 抢拍复盘）:

        原来的链是 確認 > **前往大厅** > 点击继续 > **获得奖励（横幅兜底）**，
        两个都出过事（trace t13901t13953）:
          · 横幅兜底真的点下去了（08-08 写"点它常常没反应"却留着当兜底），
            默认契约 expect_gone 等横幅消失，永不兑现  70 帧超时换目标；
          · 「前往大厅」在活动结算上是**回大厅导航**不是点掉奖励，t13939
            点完人直接被送回 lobby，活动链当场拆断。
        现在：只认 確認/点击继续 两个真按钮，同层锁一个出口，没按钮就 wait。
        """
        return self._reward_exit(obs, allow=(V.CONFIRM, V.STORY_TAP_CONTINUE))

    # ── 工具 ────────────────────────────────────────────────────────────
    def battle_stats(self) -> str:
        bt = self._bt()
        return f"胜 {bt['win']} / 胜负未知 {bt['unknown']}"
