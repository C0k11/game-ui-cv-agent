# -*- coding: utf-8 -*-
"""课程表 —— 把课程表票花完。

区域选择**动态扫**（夏莱办公室 / 夏莱居住区 / 格黑娜 / 阿拜多斯 / 千年 都有 cls），
配置里没写就按屏上从左到右依次进。

`房间区域未解锁`(50) 在场的房间跳过。`房间区域`(42) train=0，别用。
"""
from __future__ import annotations

from routing_v2.act.action import swipe, tap_box, wait
from routing_v2.flow import nav
from routing_v2.flow.base import ExitMixin, Flow, Outcome
from routing_v2.percept import read as R
from routing_v2.state import vocab as V


class ScheduleFlow(ExitMixin, Flow):
    name = "schedule"
    module = "schedule"
    # avatar 也要：全體課程表面板的房间卡**本身无 cls**，但卡上的学生头像
    #    是 avatar 模型的 cls —— 点头像即跳到该房间（08-08 实测面板页）。
    yolo = ("ui", "avatar")

    def setup(self) -> None:
        self.state.update(lessons=0, tickets0=None, tickets=None, area_i=0)

    def on_facility(self, obs, st):
        return self._roster_panel(obs)

    def _roster_panel(self, obs):
        """全體課程表面板（无专属页面 cls，身份在 facility/schedule_region
        之间摆 —— 所以两个 handler 都先试这里）。
        身份判据：**体内**出现 `课程表票`（「持有票券 7/7」那一行，cy>0.15；
        顶栏那个被 region 排掉）。它列出**全部区域**的在住房间，比逐区域扫
        高效得多 —— 点学生头像 = 直接跳进那个房间。"""
        if obs.find(V.SCHED_TICKET, 0.30, region=(0.0, 0.15, 1.0, 1.0)) is None:
            return None                      # 不是这个面板，交回去
        # 顺序：**先看有没有「課程表開始」**（房间资讯弹窗盖在面板上时，
        #    面板的头像还检得到 —— 再点头像会把弹窗关掉，开/关乒乓）。
        start = obs.find(V.SCHED_START, 0.40)
        if start is not None:
            # 别在这里清空黑名单（第一版这么写，live 立刻打脸）：
            #   「点了进不去」的原因是**那个房间今天已经开过课**——这是持久事实，
            #   不会因为我在别的房间上了一节课就变回来。清空  每上一节课就
            #   把「冬」重试一遍，每轮白扔一次点击。
            #   成功上课的那位下一帧自己就带绿勾了，本来也不会再被选中。
            return tap_box(start, "課程表開始（上课）", counter="lessons")
        # **带绿勾的头像 = 今天已经上过课**（08-08 实测：conf argmax 总选中
        #    已完成的 0.99 那位  她的资讯弹窗里没有開始键  死循环）。
        checks = obs.all(V.GREEN_CHECK, 0.40)
        def _done(b):
            return any(abs(c.cx - b.cx) < 0.06 and abs(c.cy - b.cy) < 0.08
                       for c in checks)
        # **防空转黑名单**（08-10 live 实锤：连点「冬」三次，每次都
        #    「页内生效」但页面纹丝不动，票券卡在 5/7 不再消耗）。
        #    根因是绿勾判据的**粒度错了**：绿勾长在**学生**头像上，而"还能不能
        #    上课"是**房间**级的 —— 一个没绿勾的学生，如果他所在的房间今天
        #    已经开过课，点他就是没反应。（绿勾漏检也会走到同一个死胡同。）
        #     不去猜哪种，直接**点过就拉黑**：真进了房间会走到 SCHED_START
        #      分支并清空黑名单；没进去的下一帧自动换人。
        #    这正是 memory 那条「一圈 N 区必须伴随 N-1 次帧证」的同族 ——
        #      **每一发点击都必须有"确实发生了什么"的证据，否则不许重复。**
        tried = self.state.setdefault("tried", [])
        for name in (self.cfg.get("target_students") or []):
            hit = obs.find(name, 0.45, model="avatar")
            if hit is not None and not _done(hit) and name not in tried:
                return tap_box(hit, f"全体课程表: 进目标 {name} 的房间",
                               expect=(V.SCHED_START,),
                               post=lambda n=name: tried.append(n))
        studs = [b for b in obs.boxes
                 if b.model == "avatar" and b.conf >= 0.45
                 and 0.20 < b.cy < 0.92 and not _done(b) and b.cls not in tried]
        if studs:
            b = max(studs, key=lambda x: x.conf)
            # 契约 = 真进了房间**必然**出现「課程表開始」。它同时给了节流
            #   （兑现前一发新 tap 都出不去）和**证据**（没兑现 = 这个学生
            #   所在的房间今天已经开过课，拉黑是对的）。
            return tap_box(b, f"全体课程表: 进 {b.cls} 的房间",
                           expect=(V.SCHED_START,),
                           post=lambda n=b.cls: tried.append(n))
        if tried:
            # 面板上没绿勾的都点过一轮了 —— 这一区的课上完了（剩下的学生
            # 所在房间今天都开过课），别在这儿耗着。
            x = obs.find(V.CLOSE_X, 0.55)
            if x is not None:
                return tap_box(x, f"没绿勾的 {len(tried)} 位都试过了  关面板换区域")
        # 面板上一个学生头像都没有 = 所有房间都上完课了
        x = obs.find(V.CLOSE_X, 0.55)
        if x is not None:
            return tap_box(x, "全体课程表已空  关面板")
        return None

    def on_lobby(self, obs, st):
        return nav.enter(obs, V.NAV_SCHEDULE, "课程表")

    def _tickets(self, obs):
        t = obs.find(V.SCHED_TICKET, 0.30)
        if t is None:
            return None
        # 课程表票上限很小（自带 cur<=7 的量级），hard_max 收紧一点
        return R.read_ticket(obs, t, hard_max=20)

    # ── 区域选择 ────────────────────────────────────────────────────────
    def on_schedule_area(self, obs, st):
        tix = self._tickets(obs)
        if tix is not None:
            if self.state["tickets0"] is None:
                self.state["tickets0"] = tix
                self.log(f"课程表票 {tix}")
            self.state["tickets"] = tix
            if tix <= 0:
                return self._wrap("课程表票用完了")

        want = self.cfg.get("areas") or []
        seen = obs.cols(V.SCHOOL_AREAS, 0.40)      # 从左到右
        if not seen:
            if self.stalled(st, 90):
                return self._wrap("区域选择页一个区域 cls 都没检出")
            return wait("等区域 cls")
        ordered = [b for n in want for b in seen if b.cls == n] or seen
        i = self.state["area_i"]
        if i >= len(ordered):
            return self._wrap(f"{len(ordered)} 个区域都跑过了")
        return tap_box(ordered[i], f"进区域 [{i+1}/{len(ordered)}] {ordered[i].cls}")

    # ── 区域内 ──────────────────────────────────────────────────────────
    def on_schedule_region(self, obs, st):
        tix = self._tickets(obs)
        if tix is not None:
            self.state["tickets"] = tix
            if tix <= 0:
                return self._wrap("课程表票用完了")

        # 全体课程表面板可能被判成本页（身份摆动）——先试面板逻辑
        # **「課程表開始」必须排在选学生之前**（2026-08-13 用户:「课程表这边
        #    也是抢拍，乱点然后卡死」）。轨迹实证: t1138 晴露营 -> t1144 羽留奈
        #    -> t1147 优香睡衣，间隔 6/3 tick，中间**一次 課程表開始 都没有**。
        #    根因不是节流，是**优先级倒置**: 点了学生、房间面板开出来了，可
        #    roster 列表还在屏上，于是下一帧 `_roster_panel` 又去点下一个学生。
        #    配上「点过就拉黑」，一发没兑现的点击就永久拉黑一个学生 ——
        #    和商店那次 `once` 被一发无效点击消耗掉是同一个形状。
        #     屏上有 `課程表開始` = 房间面板已经开了 = 这一步该上课，不是换人。
        start = obs.find(V.SCHED_START, 0.40)
        if start is not None:
            return tap_box(start, "开始课程", counter="lessons")

        act = self._roster_panel(obs)
        if act is not None:
            return act

        # 房间：优先配置里的学生（学生头像有 cls），其次任意可用房间
        for name in (self.cfg.get("target_students") or []):
            hit = obs.find(name, 0.45)
            if hit is not None:
                return tap_box(hit, f"选目标学生 {name}")

        roster = obs.find(V.SCHED_ALL, 0.40)
        if roster is not None and self.pending("roster"):
            return tap_box(roster, "打开全体课程表", once="roster")

        if self.stalled(st, 120):
            # 2026-08-12 用户点名:「课程表游历方式有问题，**去看老的代码，
            #    那个游历逻辑是打磨过的**」。
            #    原来这里是 `exit_step()` **退回区域选择页再挑下一个** ——
            #    而那一页能认出的区域只有 `SCHOOL_AREAS` 那 5 个 cls，
            #    实测屏上区域数 **≥9**（[[region_switch_truth]]） 大半区域
            #    根本进不去，08-12 live 只跑到 `[2/2]`，票用不完。
            #    老 `brain/skills/schedule.py` 的做法（文件头注释写得很清楚）：
            #      **不回列表** —— 在区域内屏直接按 `ARROW_LEFT` 翻到下一个区域，
            #      列表是环绕的，一路 ARROW_LEFT 就能走遍全部，
            #      最多走 14 个（BA 约 10 个）就收工。
            #     照抄这套：优先 ARROW_LEFT 翻页，翻不动（没检出箭头）才退回列表。
            n = int(self.state.get("regions_seen", 0)) + 1
            if n > 14:
                return self._wrap(f"已经翻过 {n-1} 个区域（一圈都走完了）")
            arrow = obs.find(V.ARROW_LEFT, 0.40)
            if arrow is not None:
                def _next(k=n):
                    self.state["regions_seen"] = k
                    self.once_reset()
                return tap_box(arrow, f"这区没课可上   翻到下一个区域"
                                      f"（第 {n}/14 个）", post=_next)
            # 没有左箭头 = 不在区域内屏（或版面变了） 退回列表那条老路兜底
            self.state["area_i"] += 1
            self.once_reset()
            self.log("这个区域没有可上的课，且没检出左切换  退回区域列表")
            return self.exit_step(obs, prefer_close=False) or wait("等返回控件")
        return wait("等课程表控件")

    def on_confirm_dialog(self, obs, st):
        cf = obs.find(V.CONFIRM, 0.45)
        return tap_box(cf, "确认上课") if cf is not None else wait("等確認键")

    def on_reward(self, obs, st):
        # `find([A, B])` 是**全屏 conf argmax**，不是"先 A 后 B"。
        #    base.py 的 on_reward 里有实锤:「獲得獎勵！」全屏 overlay 上
        #    `获得奖励` 0.98 是**横幅**、`点击继续字样` 0.93 才是能点的，
        #    argmax 永远选横幅，连点 11 次画面纹丝不动。
        #    这里只保留本 flow 特有的副作用，优先级链和基类一致。
        b = (obs.find(V.CONFIRM, 0.40)
             or obs.find(V.STORY_TAP_CONTINUE, 0.40)
             or obs.find(V.GOT_REWARD, 0.40))
        if b is None:
            return wait("等结果页")
        # 上完一节课  允许重开全体课程表挑下一个房间（once 标记归还）
        return tap_box(b, "关掉课程结果",
                       post=lambda: self.once_reset("roster"))

    def _wrap(self, why):
        t0, t1 = self.state["tickets0"], self.state["tickets"]
        det = f"上课 {self.state['lessons']} 次"
        if t0 is not None and t1 is not None:
            det += f"，票 {t0}{t1}"
        if t1 is not None and t1 > 0:
            return self.finish(Outcome.LEFTOVER, f"{why}；{det}（还剩 {t1} 张票）")
        if t1 is None:
            return self.finish(Outcome.UNKNOWN, f"{why}；{det}（票数读不出）")
        return self.finish(Outcome.CLEAN, f"{why}；{det}")
