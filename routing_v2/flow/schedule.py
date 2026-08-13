# -*- coding: utf-8 -*-
"""课程表 -- 把课程表票花完。

2026-08-13 重写成**相位机**（`Flow.phases`）。用户报「课程表这边也是抢拍，
   乱点然后卡死」，轨迹实证 t1138 晴露营 -> t1144 羽留奈 -> t1147 优香睡衣，
   中间**一次 `課程表開始` 都没有**。
   根因不是节流，是原来按 `st.page` 分派: 点了学生、房间面板开出来了，可
   roster 列表还在屏上、页面身份也还没切，于是下一帧 `_roster_panel` 又去点
   下一个学生。配上「点过就拉黑」，一发没兑现的点击就永久拉黑一个学生。
   相位机下这件事**结构上不可能发生**: 点完学生就进 `open_room` 相位，
   那个相位里**只有**「按開始」一条路，选学生的代码根本不在调用链上。
   老代码 `brain/skills/schedule.py` 的 `sub_state` 表就是这个模型
   （enter / navigate / roster / open_room / switch / exit）。

区域选择**动态扫**（夏莱办公室 / 夏莱居住区 / 格黑娜 / 阿拜多斯 / 千年 都有 cls），
配置里没写就按屏上从左到右依次进。

`房间区域未解锁`(50) 在场的房间跳过。`房间区域`(42) train=0，别用。
"""
from __future__ import annotations

from routing_v2.act.action import tap_box, wait
from routing_v2.flow import nav
from routing_v2.flow.base import ExitMixin, Flow, Outcome
from routing_v2.percept import read as R
from routing_v2.state import vocab as V

# 同一个绿勾在不同帧的位置抖动上限：小于这个距离算同一个勾（去重用）。
GREEN_SAME = 0.02
# 一个房间面板最多等多少 tick 等「課程表開始」出现。超了 = 这个房间今天已经
#    开过课（点进去没反应），把人拉黑换下一个。
ROOM_OPEN_CAP = 30
MAX_REGIONS = 14


class ScheduleFlow(ExitMixin, Flow):
    name = "schedule"
    module = "schedule"
    # avatar 也要：全體課程表面板的房间卡**本身无 cls**，但卡上的学生头像
    #    是 avatar 模型的 cls -- 点头像即跳到该房间（08-08 实测面板页）。
    yolo = ("ui", "avatar")
    phases = ("enter", "navigate", "roster", "open_room", "switch", "exit")

    def setup(self) -> None:
        self.state.update(lessons=0, tickets0=None, tickets=None, area_i=0)

    # 观测: 每帧都跑, 跟当前相位无关
    def observe(self, obs, st) -> None:
        t = obs.find(V.SCHED_TICKET, 0.30)
        tix = R.read_ticket(obs, t, hard_max=20) if t is not None else None
        if tix is not None:
            if self.state["tickets0"] is None:
                self.state["tickets0"] = tix
                self.log(f"课程表票 {tix}")
            self.state["tickets"] = tix
        # 绿勾**会闪**（老代码 `_accumulate_green_marks`）-- 逐帧判必漏。
        #    按位置累积，一旦见过就一直算数。
        acc = self.state.setdefault("green_acc", [])
        for c in obs.all(V.GREEN_CHECK, 0.40):
            if all((c.cx - x) ** 2 + (c.cy - y) ** 2 > GREEN_SAME ** 2
                   for x, y in acc):
                acc.append((c.cx, c.cy))

    def _panel_open(self, obs) -> bool:
        """全體課程表面板（无专属页面 cls，身份在 facility/schedule_region
        之间摆）。身份判据：**体内**出现 `课程表票`（「持有票券 7/7」那一行，
        cy>0.15；顶栏那个被 region 排掉）。"""
        return obs.find(V.SCHED_TICKET, 0.30,
                        region=(0.0, 0.15, 1.0, 1.0)) is not None

    def _owned(self, obs) -> set:
        """哪些学生头像已经带绿勾了 = 今天已经上过课。

        归属必须是 **1:1 最近邻**，不能是"附近有勾就算"（2026-08-13 实测）:
           绿勾长在自己学生的右上角、偏移约 (+0.028, -0.030)，而**学生横向
           间距只有 0.057** -- 原判据的容差 `|dx| < 0.06` 比间距还大，
           于是每个学生都能蹭到**邻居**的勾。真帧实证: 屏上只有 2 个绿勾，
           却判出 3 个学生"已选"，`美咲泳装` 蹭了 `花子` 的勾被当成上过课。
        这条**不含任何写死的比例**，换分辨率/宽高比都成立。
        """
        studs = [b for b in obs.boxes if b.model == "avatar" and b.conf >= 0.45]
        owned = set()
        for cx, cy in self.state.get("green_acc", []):
            if not studs:
                break
            near = min(studs, key=lambda s: (s.cx - cx) ** 2 + (s.cy - cy) ** 2)
            owned.add(id(near))
        return owned

    def _ticket_gate(self):
        t = self.state.get("tickets")
        if t is not None and t <= 0:
            self.goto("exit", "课程表票用完了")
            return wait("票用完，收工")
        return None

    # enter
    def do_enter(self, obs, st):
        if st.page == "lobby":
            return nav.enter(obs, V.NAV_SCHEDULE, "课程表")
        if self._panel_open(obs) or obs.has(V.SCHED_ALL, 0.40):
            return self._go("roster", "已经在课程表里")
        if obs.cols(V.SCHOOL_AREAS, 0.40):
            return self._go("navigate", "落在区域选择页")
        if self.phase_ticks > 60:
            return self.goto_and_wait("exit", "进不去课程表")
        return wait("等课程表加载")

    def _go(self, phase: str, why: str):
        self.goto(phase, why)
        return wait(f"进入相位 {phase}")

    def goto_and_wait(self, phase: str, why: str):
        return self._go(phase, why)

    # navigate: 区域选择页
    def do_navigate(self, obs, st):
        gate = self._ticket_gate()
        if gate is not None:
            return gate
        if self._panel_open(obs) or obs.has(V.SCHED_ALL, 0.40):
            return self._go("roster", "已经进到区域里了")
        seen = obs.cols(V.SCHOOL_AREAS, 0.40)          # 从左到右
        if not seen:
            if self.phase_ticks > 90:
                return self._go("exit", "区域选择页一个区域 cls 都没检出")
            return wait("等区域 cls")
        want = self.cfg.get("areas") or []
        ordered = [b for n in want for b in seen if b.cls == n] or seen
        i = self.state["area_i"]
        if i >= len(ordered):
            return self._go("exit", f"{len(ordered)} 个区域都跑过了")
        return tap_box(ordered[i],
                       f"进区域 [{i+1}/{len(ordered)}] {ordered[i].cls}",
                       expect=(V.SCHED_ALL,))

    # roster: 挑一个房间进去
    def do_roster(self, obs, st):
        gate = self._ticket_gate()
        if gate is not None:
            return gate
        # 房间面板已经开了（比如上一发点击刚生效）-> 交给 open_room，别在这
        #    继续挑人。**这是抢拍的结构性断点。**
        if obs.has(V.SCHED_START, 0.40):
            return self._go("open_room", "房间面板已经开着")
        if not self._panel_open(obs):
            roster = obs.find(V.SCHED_ALL, 0.40)
            if roster is not None:
                return tap_box(roster, "打开全体课程表",
                               expect=(V.SCHED_TICKET,))
            if self.phase_ticks > 60:
                return self._go("switch", "这一区打不开全体课程表")
            return wait("找全体课程表入口")

        owned = self._owned(obs)
        # **防空转黑名单**: 一个没绿勾的学生，如果他所在的房间今天已经开过课，
        #    点他就是没反应（绿勾漏检也会走到同一个死胡同）。不去猜哪种，
        #    **点进去等不到「開始」就拉黑**（`open_room` 相位超时时落账）。
        tried = self.state.setdefault("tried", [])
        for name in (self.cfg.get("target_students") or []):
            hit = obs.find(name, 0.45, model="avatar")
            if hit is not None and id(hit) not in owned and name not in tried:
                return self._enter_room(hit, f"进目标 {name} 的房间")
        studs = [b for b in obs.boxes
                 if b.model == "avatar" and b.conf >= 0.45
                 and 0.20 < b.cy < 0.92 and id(b) not in owned
                 and b.cls not in tried]
        if studs:
            b = max(studs, key=lambda x: x.conf)
            return self._enter_room(b, f"进 {b.cls} 的房间")
        # 面板上没绿勾的都点过一轮了 -> 这一区的课上完了
        x = obs.find(V.CLOSE_X, 0.55)
        if x is not None:
            why = (f"没绿勾的 {len(tried)} 位都试过了 - 关面板换区域"
                   if tried else "全体课程表已空 - 关面板")
            return tap_box(x, why, post=lambda: self.goto("switch", why))
        return self._go("switch", "这一区没人可选了")

    def _enter_room(self, box, why: str):
        """点学生头像进房间。契约 = 真进了房间**必然**出现「課程表開始」。

        它同时给了节流（兑现前一发新 tap 都出不去）和**证据**。
        `post` 里换相位: tap **真发出去了**才推进（数事实不数意图）。
        """
        def _went(n=box.cls):
            self.state["room_of"] = n
            self.goto("open_room", f"点了 {n}")
        return tap_box(box, f"全体课程表: {why}", expect=(V.SCHED_START,),
                       post=_went)

    # open_room: 房间面板开着, **只有一条路** -- 按開始
    def do_open_room(self, obs, st):
        gate = self._ticket_gate()
        if gate is not None:
            return gate
        start = obs.find(V.SCHED_START, 0.40)
        if start is not None:
            def _did():
                # 上完这一节, 回去挑下一个房间。绿勾累积表不清 -- 刚上课的
                #    那位下一帧自己就带勾了。
                self.state.pop("room_of", None)
                self.goto("roster", "课上完了, 挑下一个")
            return tap_box(start, "課程表開始（上课）", counter="lessons",
                           post=_did)
        if self.phase_ticks > ROOM_OPEN_CAP:
            # 等不到「開始」= 这个房间今天已经开过课，点进去没反应。
            #    **这才是拉黑的正确时机** -- 原来是"点了就拉黑"，一发被闸吞掉
            #    的点击就永久拉黑一个学生（和商店那次 `once` 被无效点击消耗
            #    掉同形）。
            n = self.state.pop("room_of", None)
            if n:
                self.state.setdefault("tried", []).append(n)
                self.log(f"{n} 的房间今天已经开过课（等不到「開始」）- 拉黑")
            return self._go("roster", "换一个房间")
        return wait(f"等「課程表開始」（{self.phase_ticks}/{ROOM_OPEN_CAP}）")

    # switch: 翻到下一个区域
    def do_switch(self, obs, st):
        """老 `brain/skills/schedule.py` 的做法（文件头注释写得很清楚）：
           **不回列表** -- 在区域内屏直接按 `ARROW_LEFT` 翻到下一个区域，
           列表是环绕的，一路 ARROW_LEFT 就能走遍全部。
        原因: 退回区域选择页只认得 `SCHOOL_AREAS` 那 5 个 cls，而实测屏上
           区域数 **>=9**（[[region_switch_truth]]）-> 大半区域根本进不去，
           08-12 live 只跑到 `[2/2]`，票用不完。
        """
        gate = self._ticket_gate()
        if gate is not None:
            return gate
        n = int(self.state.get("regions_seen", 0)) + 1
        if n > MAX_REGIONS:
            return self._go("exit", f"已经翻过 {n-1} 个区域（一圈都走完了）")
        # 面板还开着就先关掉，不然箭头点不到
        if self._panel_open(obs):
            x = obs.find(V.CLOSE_X, 0.55)
            if x is not None:
                return tap_box(x, "关掉全体课程表面板，准备翻区域",
                               expect_gone=(V.SCHED_TICKET,))
        arrow = obs.find(V.ARROW_LEFT, 0.40)
        if arrow is not None:
            def _next(k=n):
                # 换区域 = 换了一批房间，绿勾累积表和黑名单都要清。
                self.state["regions_seen"] = k
                self.state["green_acc"] = []
                self.state["tried"] = []
                self.goto("roster", f"翻到第 {k} 个区域")
            return tap_box(arrow, f"这区没课可上 - 翻到下一个区域"
                                  f"（第 {n}/{MAX_REGIONS} 个）", post=_next)
        if self.phase_ticks > 60:
            # 没有左箭头 = 不在区域内屏（或版面变了）-> 退回列表那条老路兜底
            self.state["area_i"] += 1
            self.log("没检出左切换 - 退回区域列表")
            self.goto("navigate", "退回区域列表")
            return self.exit_step(obs, prefer_close=False) or wait("等返回控件")
        return wait("找左切换箭头")

    # exit
    def do_exit(self, obs, st):
        t0, t1 = self.state["tickets0"], self.state["tickets"]
        det = f"上课 {self.state['lessons']} 次"
        if t0 is not None and t1 is not None:
            det += f"，票 {t0}->{t1}"
        why = self.state.get("exit_why", "课程表走完")
        if t1 is not None and t1 > 0:
            return self.finish(Outcome.LEFTOVER, f"{why}；{det}（还剩 {t1} 张票）")
        if t1 is None:
            return self.finish(Outcome.UNKNOWN, f"{why}；{det}（票数读不出）")
        return self.finish(Outcome.CLEAN, f"{why}；{det}")

    # overlay
    def on_confirm_dialog(self, obs, st):
        cf = obs.find(V.CONFIRM, 0.45)
        return tap_box(cf, "确认上课") if cf is not None else wait("等確認键")

    def on_reward(self, obs, st):
        # `find([A, B])` 是**全屏 conf argmax**，不是"先 A 后 B"。
        #    「獲得獎勵！」全屏 overlay 上 `获得奖励` 0.98 是**横幅**、
        #    `点击继续字样` 0.93 才是能点的，argmax 永远选横幅，
        #    连点 11 次画面纹丝不动。
        b = (obs.find(V.CONFIRM, 0.40)
             or obs.find(V.STORY_TAP_CONTINUE, 0.40)
             or obs.find(V.GOT_REWARD, 0.40))
        if b is None:
            return wait("等结果页")
        return tap_box(b, "关掉课程结果")
