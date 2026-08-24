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

上锁区域**结构上碰不到**(2026-08-21 用户口述 + live 复核): 路由是
   夏莱办公室 -> 全体课程表面板选学生 -> 開始 -> 关面板 -> **ARROW_LEFT 翻下一区**,
   全程不回区域选择列表; 而选人是从面板列表里选的, 锁房间根本不在面板上。
   所以这里**没有也不需要**显式查锁 —— 早先写的「`房间区域未解锁`(50) 在场的
   房间跳过」是过时说法, 代码里从来没有那一步, 别照着它去加。
cls 50 本身是好的: 区域内页那种绿锁 live 0.95-0.97(小号 Lv30 实测 5 个锁房间全中);
   区域选择列表上「招募XX學生」那种灰锁 0 检出, 但路由不走那儿, 不影响。
`房间区域`(42) train=0，别用。
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
        # **上过课的房间落本地台账，按游戏日对齐**（用户 2026-08-13 拍板:
        #    「辨别不出这个角色的房间选没选过，因为没拥有的角色摸了头像只会
        #    dim，拥有的角色摸了才会有勾，我认为的解决办法是本地记录，
        #    然后根据游戏每日刷新对齐，这样就很简单了」）。
        #    绿勾只是**已拥有角色**的副产品，唯一可靠的"用过没"是自己记。
        #    游戏日 = UTC+8 减 3 小时（繁中服 JST04:00 刷新，[[game_day_timezone]]
        #    那次 12h 错窗就是拿裸 now() 惹的）。跨 run 有效，同一天不重复点。
        self.state["tried"] = self._load_rooms()

    @staticmethod
    def _game_day() -> str:
        from datetime import datetime, timedelta, timezone
        return (datetime.now(timezone(timedelta(hours=8)))
                - timedelta(hours=3)).strftime("%Y%m%d")

    def _rooms_file(self):
        """账号桶里的房间台账（08-15 分桶: 键只有游戏日+学生名, 大小号同一天
        会把对方上过的课当成自己上过的）。"""
        from routing_v2.config import data_dir
        return data_dir(self.ctx.cfg) / "schedule_rooms.json"

    def _load_rooms(self) -> list:
        import json
        try:
            d = json.loads(self._rooms_file().read_text(encoding="utf-8"))
            if d.get("day") == self._game_day():
                rooms = list(d.get("rooms", []))
                if rooms:
                    self.log(f"本游戏日已上过课的房间 {len(rooms)} 间（台账续用）")
                return rooms
        except Exception:
            pass
        return []

    def _save_rooms(self) -> None:
        import json
        p = self._rooms_file()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"day": self._game_day(),
                                 "rooms": self.state.get("tried", [])},
                                ensure_ascii=False), encoding="utf-8")

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
            act = nav.enter(obs, V.NAV_SCHEDULE, "课程表")
            if act is not None and act.kind == "tap":
                # 计数挂 post = **tap 真发出去了才算点过**（被闸吞掉的那一发
                #    不该消耗重试预算）。
                act.post = lambda: self.bump("enter_taps")
            return act
        if self._panel_open(obs) or obs.has(V.SCHED_ALL, 0.40):
            return self._go("roster", "已经在课程表里")
        if obs.cols(V.SCHOOL_AREAS, 0.40):
            return self._go("navigate", "落在区域选择页")
        # 别按 tick 数收工 —— 2026-08-13 live 实测: 进课程表要走两次加载，
        #    转场帧全是 blank/unknown，`phase_ticks > 60` 在**还在加载**的时候
        #    就把整条 flow 判死（当轮 schedule 一节课都没上，报 UNKNOWN）。
        #    真正的失败判据是「**在大厅上点了入口却始终进不去**」——
        #    数的是"点了几次没生效"这个事实，不是干等了几帧。
        if self.state.get("enter_taps", 0) >= 4:
            return self._go("exit", "点了 4 次课程表入口都没进去")
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
        # **起点是硬规则: 夏莱办公室**（用户 2026-08-13 纠正:「我们应该最先去的
        #    就是夏莱办公室然后开课程表找人，然后往左箭头路由，我们起点都进错了」）。
        #    老代码 brain/skills/schedule.py:1058 白纸黑字:
        #      `Sequence (once): click 夏莱办公室 -> ARROW_LEFT (jump to last region)`
        #    —— 办公室是列表第 0 行的**全环锚点**，从它起跳 ARROW_LEFT 一圈才闭合。
        #    我上一版写成"按屏上从左到右取第一个"，于是进了夏莱居住区。
        want = self.cfg.get("areas") or []
        ordered = [b for n in want for b in seen if b.cls == n] if want else []
        if not ordered:
            office = [b for b in seen if b.cls == V.SCHOOL_OFFICE]
            ordered = office + [b for b in seen if b.cls != V.SCHOOL_OFFICE]
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
            if (hit is not None and id(hit) not in owned and name not in tried
                    and not self._on_locked_card(obs, hit)):
                return self._enter_room(hit, f"进目标 {name} 的房间")
        studs = [b for b in obs.boxes
                 if b.model == "avatar" and b.conf >= 0.45
                 and 0.20 < b.cy < 0.92 and id(b) not in owned
                 and b.cls not in tried
                 and not self._on_locked_card(obs, b)]
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

    # 全體課程表面板是 3 列 x N 行的房间卡。锁图标在卡的**标题行**, 学生头像在
    #   它下面同一张卡里。实测卡宽约 0.26 / 高约 0.20(锁 cx 落在 .2339/.5028/.7718
    #   三列, cy 落在 .3310/.5411/.7512 三行), 所以「同一张卡」= 横向 <0.13
    #   且在锁**下方** 0.15 以内。不含写死坐标, 只用锁框本身的相对位置。
    _CARD_DX = 0.13
    _CARD_DY = 0.15

    def _on_locked_card(self, obs, box) -> bool:
        """这个头像是不是长在一张**锁着的**房间卡上。

        锁着的房间点进去没反应, 早先只能靠 `ROOM_OPEN_CAP` 超时拉黑,
        每个白烧 30 tick。锁本身 live 0.98, 直接拿它当负判据便宜得多。
        """
        for lk in obs.all(V.ROOM_LOCKED, 0.45):
            if (abs(box.cx - lk.cx) < self._CARD_DX
                    and 0.0 < box.cy - lk.cy < self._CARD_DY):
                return True
        return False

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
                # **上过课的房间必须自己记，不能只等绿勾**（用户 2026-08-13:
                #    「这些角色我没获得，所以你给那个房间用了票没有绿勾」）。
                #    绿勾只长在**已获得**的学生头像上；房间里摆的是没获得的角色时,
                #    票照样消耗、课照样上，屏上却什么都不变 -> 光靠绿勾的话
                #    roster 下一帧又挑中同一个房间，票一张张烧光在同一间屋子里。
                #     记在 `tried` 里 —— 它本来就是"这一区别再点这个"的台账。
                n = self.state.pop("room_of", None)
                if n:
                    self.state.setdefault("tried", []).append(n)
                    self._save_rooms()      # 写穿: 掉线/重启也不丢今天的账
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
                self._save_rooms()
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
                # 换区域清绿勾累积表（位置相关）。**`tried` 不清** —— 它现在是
                #    按游戏日落盘的房间台账，学生名全服唯一，跨区域仍然有效；
                #    清了就等于把今天的账撕了（本地记录方案的要点）。
                self.state["regions_seen"] = k
                self.state["green_acc"] = []
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
