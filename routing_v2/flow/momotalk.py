# -*- coding: utf-8 -*-
"""MomoTalk 学生好感度 —— 用户点名要的模块，**默认关，前端按钮开**。

信号质量很好: `学生momotalk信息未读`(439) **train 6462 / val 241**，
`学生信息回复选项`(440) train 418。这条链的感知基础是全 bot 里最扎实的之一。

`学生发送信息中`(438) 是**瞬时态**（学生正在打字）—— 见到就等，别在这时候
   去点回复选项，点了会落空。这也是为什么这里**不能用固定 sleep**：打字时长
   随消息长度变，只能靠 cls 消失来判。

回复策略（前端可选）:
  first_option  选第一个（最稳，默认）
  last_option   选最后一个
  random        随机（用 tick 数取模，不用 random 模块以便复现）

好感度：BA 的 MomoTalk 选项对好感度**没有对错**（选哪个都加），所以默认
   first_option 就够了。真正加好感的是**跟进羁绊剧情**（`follow_bond_story`）。
"""
from __future__ import annotations

from typing import Optional

from routing_v2.act.action import swipe, tap_box, wait
from routing_v2.flow import nav
from routing_v2.flow.base import ExitMixin, Flow, Outcome
from routing_v2.state import vocab as V


class MomoTalkFlow(ExitMixin, Flow):
    name = "momotalk"
    module = "momotalk"

    def setup(self) -> None:
        self.state.update(students=0, replies=0, bonds=0, scrolls=0)

    def on_lobby(self, obs, st):
        e = obs.find(V.NAV_MOMOTALK, 0.35)
        if e is not None:
            return tap_box(e, "打开 MomoTalk")
        if self.stalled(st, 120):
            return self.finish(Outcome.UNKNOWN, "大厅没检出 MomoTalk 入口 cls")
        return wait("等 MomoTalk 入口")

    # ── 列表 ────────────────────────────────────────────────────────────
    def on_momo_list(self, obs, st):
        cap = int(self.cfg.get("max_students_per_run", 0) or 0)
        if cap and self.state["students"] >= cap:
            return self._wrap(f"跑满 {cap} 个学生")

        if not obs.has(V.MOMO_TAB_SEL, 0.40):
            t = obs.find(V.MOMO_TAB, 0.40)
            if t is not None and self.pending("tab"):
                return tap_box(t, "切到对话区域", once="tab")

        # 只回配置里的学生（留空 = 全部）
        targets = self.cfg.get("target_students") or []
        unread = obs.rows(V.MOMO_UNREAD, 0.40)
        if targets:
            picked = []
            for u in unread:
                who = obs.nearest(targets, to=(u.cx, u.cy), conf=0.40)
                if who is not None and ((who.cx - u.cx) ** 2
                                        + (who.cy - u.cy) ** 2) ** 0.5 < 0.12:
                    picked.append(u)
            unread = picked

        # 已经聊干净过的行要跳过（用户 08-09：三千留没有"发送信息中"也没有
        #    回复选项了，就该去找**列表里下一个**）。红色徽章有时不会立刻消，
        #    只认徽章会在同一个学生上死循环。按**行的 cy** 记账。
        done_cys = self.state.setdefault("done_rows", [])
        fresh = [u for u in unread
                 if all(abs(u.cy - c) > 0.04 for c in done_cys)]

        if fresh:
            self.state["scrolls"] = 0
            self.once_reset("bond")
            u = fresh[0]
            # 落点必须在**行上**，不是徽章上（08-09 帧证）：
            #    `学生momotalk信息未读` 标的是右侧那个**红色数字徽章**
            #    （三千留的"1"在 cx=0.506），徽章本身不吃点击  连点不进去。
            #    行的可点区在左边（头像/名字，cx≈0.25） dx≈-0.25。
            #    和 cafe 摸头 dx=+0.025 同类：**cls 框的位置 ≠ 可点位置**。
            return tap_box(u, "进未读对话（落点移到行上，不是徽章）",
                           dx=-0.25, counter="students",
                           post=lambda cy=u.cy: self.state["done_rows"].append(cy))

        # 本屏的都清完了  **再滑动**找下面的（用户口述顺序）
        if self.state["scrolls"] < 8:
            # 全仓最毒的一处 mutate-before-ack：`done_rows` 是**已处理行的
            #    台账**，在 decide 期就清空 —— 滑动一旦没发出去（被闸吞/被 step
            #    只看一眼），台账没了而屏幕没动  本屏已聊过的行会被**重新点一遍**。
            n = self.state["scrolls"] + 1
            return swipe(0.28, 0.72, 0.28, 0.40,
                         f"本屏未读都清完了  下滑找（第 {n} 次）",
                         post=lambda: self.state.update(scrolls=n, done_rows=[]))
        return self._wrap("没有未读消息了")

    # ── 对话中 ──────────────────────────────────────────────────────────
    def on_momo_chat(self, obs, st):
        # 学生正在打字  见即等。**不能用 sleep 猜时长。**
        if obs.has(V.MOMO_SENDING, 0.35):
            return wait("学生打字中 — 等它说完")

        # 羁绊剧情入口
        bond = obs.find([V.MOMO_GOTO_BOND, V.MOMO_ENTER_BOND], 0.35)
        if bond is not None:
            if self.cfg.get("follow_bond_story", True):
                if self.pending("bond"):
                    return tap_box(bond, "跟进羁绊剧情", counter="bonds", once="bond")
            else:
                return self.exit_step(obs) or wait("等退出控件")

        opts = obs.rows(V.MOMO_REPLY_OPT, 0.35)
        if opts:
            policy = self.cfg.get("reply_policy", "first_option")
            if policy == "last_option":
                pick = opts[-1]
            elif policy == "random":
                pick = opts[self.ticks % len(opts)]
            else:
                pick = opts[0]
            # 回复后**解锁 bond once**：羁绊剧情看完回来还会有后续对话，
            #   聊着聊着可能再次解锁羁绊入口（用户 08-09 指出）。
            return tap_box(pick, f"回复（策略 {policy}，共 {len(opts)} 个选项）",
                           counter="replies",
                           post=lambda: self.state.pop("once:bond", None))

        # 「这段聊干净了」= **连续 N 帧既没有三个小点、也没有回复选项、
        #    也没有羁绊入口**（用户 08-09 口述判据）。
        #    旧写法用 `frames_in_page > 45` 这种**时间判据**当收工 ——
        #      学生打字慢一点就被判成聊完，后续对话（羁绊剧情看完回来还有）
        #      直接丢掉。时间不是事实，**cls 才是**。
        #    这里能走到说明本帧三样都没有；hold 负责确认它不是过渡帧。
        if not self.hold("chat_clean", 40):
            return wait("对话看着聊完了 — 连续确认中（防学生正要开口的过渡帧）")
        self.log("这段对话已聊干净（无 学生发送信息中 / 回复选项 / 羁绊入口）")
        return self.exit_step(obs, prefer_close=False) or wait("等返回列表")

    def on_bond_story_panel(self, obs, st):
        """羁绊剧情面板（overlay，盖在 MomoTalk 列表/对话上）。

        08-09 实锤：这个面板没有独立身份时，页面被左边列表的未读 cls 判成
           `momo_list`，flow 转头又去点未读对话  **面板永远点不进去**，
           羁绊剧情这条链等于没实现。
        面板里的「羈絆劇情獎勵 青辉石 x40」是**收入预览**，已在 money.py 的
           income 语境里排除，不会误报购买。
        """
        if not self.cfg.get("follow_bond_story", True):
            x = obs.find(V.CLOSE_X, 0.45)
            return tap_box(x, "配置不跟进羁绊剧情  关面板") if x is not None else None
        b = obs.find(V.MOMO_ENTER_BOND, 0.40)
        if b is not None and self.pending("enter_bond"):
            return tap_box(b, "進入羈絆劇情", counter="bonds", once="enter_bond")
        return None

    def on_story_nodes(self, obs, st):
        """羁绊剧情的节点图（跟进后会到这里）。挖一个就回去。"""
        undone = obs.rows(V.STORY_NODE_UNDONE, 0.40)
        if undone and self.pending("bondnode"):
            return tap_box(undone[0], "看羁绊剧情节点", once="bondnode")
        return self.exit_step(obs, prefer_close=False) or wait("等返回控件")

    def on_stage_popup(self, obs, st):
        # **别把 `任务开始` 混进来**（2026-08-10 arena 同形 bug 的连坐排查）：
        #    `find([A, B])` 是 conf argmax，而 `任务开始`(常 0.99) 会压过
        #    `进入章节`；更要命的是**羁绊剧情根本不该出现「任務開始」**——
        #    真出现了说明我们站在**别的 flow 的关卡弹窗**上，点下去就是打战斗、
        #    吃 AP，AP 不够时游戏直接弹「購買AP 單價💎30」。
        #     只认「進入章節」，认不出就不动，交给 nav 退出去。
        b = obs.find(V.STORY_ENTER_CHAPTER, 0.35)
        return tap_box(b, "进入羁绊章节") if b is not None else wait("等进入章节键")

    def on_confirm_dialog(self, obs, st):
        cf = obs.find(V.CONFIRM, 0.45)
        return tap_box(cf, "确认") if cf is not None else wait("等確認键")

    def _wrap(self, why):
        return self.finish(
            Outcome.CLEAN if self.state["students"] else Outcome.SKIPPED,
            f"{why}；对话 {self.state['students']} 个学生，"
            f"回复 {self.state['replies']} 次，跟进羁绊 {self.state['bonds']} 次")
