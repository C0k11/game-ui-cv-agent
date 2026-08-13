# -*- coding: utf-8 -*-
"""咖啡厅 -- 用户口述的权威走法（memory cafe_flow_spec）:

    弹窗叉 -> 收益!=0 就领 -> 邀请卷找角色（下滑）-> 摸头**给足时间**
    -> 2号厅 -> 无黄点才返回

2026-08-13 重写成**相位机**（`Flow.phases`）。原来是按 `st.page` 分派的，
   于是「点开邀请卷」和「关掉挡路弹窗」两条分支在同一个 `on_cafe` 里轮流赢:
   面板弹出来了但页面身份要 3 帧才切，那几帧 `on_cafe` 从头重跑，第 2 条分支
   把自己刚开的面板叉掉 -- live 实测开-叉循环 **11 次**（t81 到 t223）。
   相位机下这件事**结构上不可能发生**: 进了 `invite` 相位，`do_invite` 是
   唯一在跑的处理器，那些分支根本不在调用链上。
   老代码 `brain/skills/cafe.py:434` 的 `sub_state` 表就是这个模型。

阶段序列对齐老代码:
    enter -> earnings -> invite -> headpat -> switch -> invite2 -> headpat2 -> exit
"""
from __future__ import annotations

import time
from typing import Optional

from routing_v2.act.action import Action, swipe, tap_box, wait
from routing_v2.flow import nav
from routing_v2.flow.base import ExitMixin, Flow, Outcome
from routing_v2.percept.observe import Observation
from routing_v2.state import vocab as V
from routing_v2.state.machine import StateView

# **点气泡本身摸不到头** -- 落点要在气泡**右边一点点**，那里才是学生身体。
#    用户 2026-08-08 当场纠正；老代码 `brain/skills/cafe.py:69` 早就写着
#    `_HEADPAT_DX = 0.025  # click this much right of the bubble = student body`
HEADPAT_DX = 0.025
# 摸过的位置记下来，避免对同一个学生反复点（气泡消失有延迟）
HEADPAT_DEDUP = 0.035
HEADPAT_KEEP = 4
# 连续多少帧没气泡才算这一屏摸干净。老代码 `_HEADPAT_DRY_FRAMES = 7`:
#    气泡最晚 600ms 才渲染，且 `Emoticon_Action` 折进 ui 后 conf 偏弱，
#    提前判 dry 就 pan 走 = 漏摸（live 2026-06-09 实锤）。
HEADPAT_DRY_FRAMES = 7
# pan 之后要等的沉降时间（老代码 `_PAT_SETTLE_SEC`）-- pan 完有迟到气泡。
PAT_SETTLE_SEC = 1.4
MAX_PANS_PER_FLOOR = 2
MAX_PATS_PER_FLOOR = 40


class CafeFlow(ExitMixin, Flow):
    name = "cafe"
    module = "cafe"
    # Emoticon_Action 已折进 ui 模型（451/452）；
    # avatar 也要：邀请列表按**学生名 cls** 找目标（凯伊/爱丽丝...是
    #    fused_avatar 的类，只跑 ui 的话 `obs.find("凯伊")` 永远 None，
    #    表现成"滑 8 次没找到"然后放弃邀请）。
    yolo = ("ui", "avatar")
    phases = ("enter", "earnings", "invite", "headpat",
              "switch", "invite2", "headpat2", "exit")

    def setup(self) -> None:
        self.state.update(claimed=False, pats=0, invited=0, floor=1, backs=0,
                          pat_ts=0.0, invite_scrolls=0)

    # 观测: 每帧都跑，跟当前相位无关
    def observe(self, obs: Observation, st: StateView) -> None:
        """从**屏上 cls** 认当前在几号厅，别信自己维护的计数器。

        2026-08-12 用户抓到:「怎么咖啡厅 2 邀请了凯伊，那**咖啡厅 1 邀请的
           是谁**？」根因：`floor` 是 flow 自己维护的 -- 典型的**数意图不数事实**。
        屏上本来就有分辨方式（互斥的一对）:
           `移动至2號店` 在场 = 我在 **1 号厅**（按钮指向别处）
           `移动至一號店` 在场 = 我在 **2 号厅**
        """
        to2 = obs.find(V.CAFE_MOVE_2F, 0.45)
        to1 = obs.find(V.CAFE_MOVE_1F, 0.45)
        real = 1 if (to2 is not None and to1 is None) else (
            2 if (to1 is not None and to2 is None) else None)
        if real is not None and real != self.state.get("floor"):
            self.log(f"按屏上按钮校正：当前其实在 {real} 号厅"
                     f"（原以为 {self.state.get('floor')} 号）")
            self.state["floor"] = real

    def _in_cafe(self, obs: Observation) -> bool:
        """人在咖啡厅主视图里 -- 判据是那对互斥的换厅按钮（见 observe）。"""
        return (obs.has(V.CAFE_MOVE_2F, 0.45) or obs.has(V.CAFE_MOVE_1F, 0.45))

    # enter
    def do_enter(self, obs, st):
        if self._in_cafe(obs):
            return self.goto_and_wait("earnings", "已在咖啡厅")
        if st.page == "lobby":
            return nav.enter(obs, V.NAV_CAFE, "咖啡厅")
        return wait("等咖啡厅加载出来")

    def goto_and_wait(self, phase: str, why: str) -> Action:
        self.goto(phase, why)
        return wait(f"进入相位 {phase}")

    # earnings
    def do_earnings(self, obs, st):
        """领收益。**能领就先领**，顺序不能反：收益弹窗**自己也有叉叉**，
        "叉叉优先"会把它关掉而不是领（2026-08-08 逐帧当场抓到）。
        用户口述的"进来先点叉叉"指的是**首进的角色说明弹窗** -- 那种弹窗上
        没有领取键。所以判据是「**有领取键就领，没有才关**」。

        `claimed` 和 `earn_done` 是**两件事**（2026-08-13 用户报「体力 999 了
           领不了咖啡厅收益了，也没法辨别」后拆开的）:
             · `claimed`   = 真的点到了**可点的**领取键 = 钱到手了（事实）
             · `earn_done` = 这一步不用再试了（收敛用）
           原来两者共用一个 flag，于是"面板里没有可点的领取键"被写成
           `claimed=True`，收尾报告堂而皇之地写「收益已领」-- **谎报**。
           体力满 999 时收益就是领不动，这条路径当场就会踩到。
        """
        claim = obs.find(V.CLAIM_ACTIVE, 0.45)
        if claim is not None:
            return tap_box(claim, f"领取（{claim.cls}）", counter="claims",
                           post=lambda: self.state.update(claimed=True,
                                                          earn_done=True))
        if self.state.get("earn_done"):
            return self.goto_and_wait("invite", "收益这一步处理完了")
        # 面板开着但领取键是灰的 = 领不了。**不许当成领过了。**
        if obs.has(V.CLAIM_DONE, 0.45):
            self.state.update(earn_done=True,
                              earn_why="收益领取键是灰的（体力满了领不动？）")
            return self.goto_and_wait("invite", "收益领不了")
        # 挡路的弹窗（首进的角色说明）-- 相位机下这里不会再误伤邀请面板:
        #    邀请面板是 `invite` 相位的事，那个相位里根本不跑这条分支。
        x = obs.find(V.CLOSE_X, 0.55)
        if x is not None:
            return tap_box(x, "关掉挡路的弹窗（屏上没有可领的东西）")
        earn = obs.find(V.CAFE_EARNINGS, 0.45)
        if earn is not None:
            # 契约用默认的 `expect_gone=(收益图标,)`：面板一盖上来图标就没了。
            #    这里**不能**写 `expect=(领取键,)` -- 收益为 0 时面板里压根没有
            #    可点的领取键，严格契约会一直不兑现、把整条链按死。
            return tap_box(earn, "打开咖啡厅收益面板")
        if self.phase_ticks > 30:
            self.state.update(earn_done=True, earn_why="屏上找不到收益入口")
            return self.goto_and_wait("invite", "屏上没有收益入口，跳过")
        return wait("找收益入口")

    # invite (每厅一张券)
    def do_invite(self, obs, st):
        return self._invite(obs, st, nxt="headpat")

    def do_invite2(self, obs, st):
        return self._invite(obs, st, nxt="headpat2")

    def _invite(self, obs, st, *, nxt: str):
        """开券 -> 列表里找目标 -> 点同行邀请键 -> 等确认框。

        整条链在**同一个相位**里，所以「开券」和「退出/叉掉」不再是两条
           互相抢的分支 -- 这正是老代码 `_invite_stage: 0=开卷 1=列表 2=确认`
           的等价物（`brain/skills/cafe.py:236`）。
        """
        fl = self.state.get("floor", 1)
        if (not self.cfg.get("invite_targets")
                or self.cfg.get("skip_invite", False)):
            return self.goto_and_wait(nxt, "没配邀请目标")
        if self.state.get(f"invited_f{fl}"):
            # `post` 只保证「**tap 指令发出去了**」，而邀请**要再点一次確認
            #    才算数**。退出前先给确认框 25 帧时间弹出来（overlay 优先级
            #    高于相位处理器，确认框一出现就会被 `on_confirm_dialog` 接走）。
            if not self.hold("after_invite", 25):
                return wait("邀请已发出 - 等確認框弹出来（别急着走，走了这张券就白点了）")
            return self.goto_and_wait(nxt, f"{fl} 号厅邀请完成")

        panel = obs.has(V.CAFE_INVITE, 0.40)
        if not panel:
            tk = obs.find(V.CAFE_TICKET, 0.45)
            if tk is not None:
                # 严格契约: 等列表里的邀请键**真的出现**，在那之前一发 tap 都
                #    不许出去。默认契约 `expect_gone=(邀请卷,)` 在这里不够准 --
                #    面板一弹出来就把券图标盖住了，契约立刻"兑现"。
                return tap_box(tk, f"开邀请卷（{fl} 号厅）",
                               expect=(V.CAFE_INVITE,))
            if self.phase_ticks > 40:
                return self.goto_and_wait(nxt, f"{fl} 号厅没有可用的邀请卷")
            return wait("找邀请卷")

        # 面板开着 -- 在列表里找配置的角色
        targets = self.cfg.get("invite_targets") or []
        done_names = self.state.get("invited_names", [])
        for name in targets:
            if name in done_names:
                continue
            hit = obs.find(name, 0.45)      # 学生名 = avatar 模型的 cls 名
            if hit is None:
                continue
            # 列表**每行自带一个邀请键**，所以不是"选中再邀请"两步，而是
            #    **点目标同行（cy 最近）的邀请键**一步到位。全屏 argmax 找
            #    邀请键会点到**别人那一行**（840AP 同族病）。
            self.state["saw_invite_target"] = True
            btns = obs.all(V.CAFE_INVITE, 0.45)
            same = min(btns, key=lambda b: abs(b.cy - hit.cy), default=None)
            if same is None or abs(same.cy - hit.cy) >= 0.05:
                return wait(f"看到 {name} 但同行邀请键没检出")

            def _did(f=fl, n=name, cy=hit.cy):
                self.log(f"邀请了 {n}（行 cy={cy:.3f}）")
                self.state[f"invited_f{f}"] = True
                self.state.setdefault("invited_names", []).append(n)
            act = tap_box(same, f"邀请 {name}（同行邀请键）",
                          counter="invited", post=_did)
            # **行内按钮的容差必须显著小于半行距**，否则"同行"这个前提在 JIT
            #    复验阶段就被容差自己破坏掉（默认 ~0.048 vs 半行距 0.053）。
            act.anchor_tol = 0.030
            return act

        # 只要这一轮**见过**目标，就绝不再滑：原地等它重新检出来再点。
        #    2026-08-12 实测感知层是好的（12/12 同行匹配），**不是找不到，
        #    是找到了又被自己滑掉**。
        if self.state.get("saw_invite_target"):
            return wait("这一轮见过邀请目标 - 原地等它重新检出，**绝不滑动**")
        # **进去先扫，扫完没有才滑**（用户口述的正确节奏）。
        if self.phase_ticks < 12:
            return wait("刚开邀请列表 - 先让头像模型把这一屏扫一遍，别急着滑")
        if self.state["invite_scrolls"] < 8:
            n = self.state["invite_scrolls"] + 1
            # 几何从检出推：邀请键就是这一列的行（见 nav.list_swipe）。
            sw = nav.list_swipe(obs, [V.CAFE_INVITE],
                                f"下滑找邀请目标（第 {n} 次）",
                                post=lambda: self.state.update(invite_scrolls=n))
            if sw is not None:
                return sw
            return wait("这一屏没检出邀请键行 - 推不出滑动几何，不瞎滑")
        if self.pending("no_invite_note"):
            self.state["once:no_invite_note"] = True
            self.log("配置的邀请目标都没找到 - 不乱邀请，退出列表")
            self.note_lines.append("邀请：配置的角色没找到，未消耗邀请卷")
        x = obs.find(V.CLOSE_X, 0.55)
        if x is not None:
            return tap_box(x, "关掉邀请列表", expect_gone=(V.CAFE_INVITE,))
        return self.goto_and_wait(nxt, "邀请列表处理完")

    # headpat (含 pan 扫视)
    def do_headpat(self, obs, st):
        return self._headpat(obs, st, nxt="switch")

    def do_headpat2(self, obs, st):
        return self._headpat(obs, st, nxt="exit")

    def _next_bubble(self, obs: Observation):
        """还没摸过的那个气泡。

        `Emoticon_Action` 是**弱检出**（mean conf 0.65，35% 低于 0.55）--
          门槛必须放到 0.30 这一档，否则真气泡会被门槛滤掉（漏摸）。
        """
        done = self.state.get("patted", [])
        for b in obs.all(V.EMOTICON, 0.30):
            if all((b.cx - x) ** 2 + (b.cy - y) ** 2 > HEADPAT_DEDUP ** 2
                   for x, y in done):
                return b
        return None

    def _headpat(self, obs, st, *, nxt: str):
        """当前视角 loop-until-dry -> 左滑露出右侧 -> 右滑（2 倍距离）露出左侧 -> 收工。

        这是老代码 `brain/skills/cafe.py:1100 _headpat` 的算法，探针确认过。
        **新代码原来完全没有 pan 扫视** -- 只摸当前视角看得见的，屏幕左右两边
           的学生永远摸不到。用户 2026-08-13 点名要移植。
        SWIPE_HARDCODED_OK:
        pan 的几何是**写死的全幅扫**（0.80->0.10 / 0.10->0.90），这是"先扫再滑"
           规则的具名豁免: 咖啡厅是 3D 相机视角，屏上没有任何可检出的轨道能
           推出几何来（列表滑动才有，见 `nav.list_swipe`）。老代码注释记着
           为什么是全幅: 原来那对 0.75->0.25 / 0.25->0.80 不对称，中间留了一条
           盲带，站在那儿的学生两次停留都不在屏上（live 2026-06-09 1F 漏摸）。
        """
        if not self.cfg.get("headpat", True):
            return self.goto_and_wait(nxt, "配置关了摸头")
        if not self._in_cafe(obs):
            if self.phase_ticks > 12:
                return self.goto_and_wait(nxt, "不在咖啡厅主视图了")
            return wait("等回到咖啡厅主视图")
        if self.state.get("pats", 0) >= MAX_PATS_PER_FLOOR:
            return self.goto_and_wait(nxt, "摸头到上限")

        # 摸完一个要等动画走完再摸下一个，不然第二下会落在动画中途的空位上。
        dwell = float(self.cfg.get("headpat_dwell_s", 2.0))
        if time.time() - self.state["pat_ts"] < dwell:
            return wait(f"摸头动画中（等 {dwell}s）")

        emo = self._next_bubble(obs)
        if emo is not None:
            self.state["dry"] = 0

            def _mark(e=emo):
                # 真点出去了才记「摸过」。写在 decide() 里的话，被闸吞掉的
                #    那个气泡会被记成已摸且永不重试。
                self.state["pat_ts"] = time.time()
                self.state.setdefault("patted", []).append((e.cx, e.cy))
                self.state["patted"] = self.state["patted"][-HEADPAT_KEEP:]
            return tap_box(emo, "摸头（落点在气泡右侧 = 学生身体）",
                           dx=HEADPAT_DX, counter="pats", post=_mark)

        # 这一帧没气泡
        dry = self.state.get("dry", 0) + 1
        self.state["dry"] = dry
        if dry < HEADPAT_DRY_FRAMES:
            return wait(f"扫气泡（连续 {dry}/{HEADPAT_DRY_FRAMES} 帧空）")
        # 判 dry 前最后一扫: 清 dedup 残留。被闸吞掉的那一发会毒化 `patted`，
        #    真拍过的气泡已经消失，清账无害；假拍的（点击被吞）气泡还在 -> 捡回。
        if self.state.get("patted"):
            self.state["patted"] = []
            self.state["dry"] = 0
            return wait("清一次摸头去重账，再扫一遍（防被吞的那一发毒化）")

        pans = self.state.get("pans", 0)
        if pans >= MAX_PANS_PER_FLOOR:
            return self.goto_and_wait(nxt, "两个方向都扫干净了")
        fl = self.state.get("floor", 1)
        done = lambda n=pans + 1, d=dwell: self._after_pan(n, d)
        if pans == 0:
            return swipe(0.80, 0.45, 0.10, 0.45,
                         f"左滑露出右侧的学生（{fl} 号厅）", post=done)
        return swipe(0.10, 0.45, 0.90, 0.45,
                     f"右滑露出左侧的学生（{fl} 号厅）", post=done)

    def _after_pan(self, n: int, dwell: float) -> None:
        """pan **真的发出去了**才记账（数事实不数意图）。"""
        self.state["pans"] = n
        self.state["dry"] = 0
        self.state["patted"] = []
        # pan 完有迟到气泡，给足沉降时间再扫（老代码 `_PAT_SETTLE_SEC = 1.4`）
        self.state["pat_ts"] = time.time() + PAT_SETTLE_SEC - dwell

    # switch
    def do_switch(self, obs, st):
        """去还没去过的厅。**每个厅只主动去一次** -- 否则「floor==1 就去 2F」
        和「有黄点就回 1F」会互相打乒乓（08-08 实测空转 3 个来回）。"""
        floors = self.cfg.get("floors", [1, 2])
        cur = self.state.get("floor", 1)
        vis = set(self.state.setdefault("visited_floors", []))
        vis.add(cur)
        self.state["visited_floors"] = sorted(vis)
        todo = [f for f in floors if f not in vis]
        if not todo:
            return self.goto_and_wait("exit", "配置的厅都去过了")
        dst = todo[0]
        btn = obs.find(V.CAFE_MOVE_2F if dst == 2 else V.CAFE_MOVE_1F, 0.45)
        if btn is None:
            if self.phase_ticks > 30:
                return self.goto_and_wait("exit", f"找不到去 {dst} 号厅的按钮")
            return wait(f"找去 {dst} 号厅的按钮")

        def _moved(d=dst):
            self.state.update(floor=d, pat_ts=0.0, dry=0, pans=0, patted=[],
                              invite_scrolls=0)
            self.state.pop("saw_invite_target", None)
            # **换完厅要去干活，不是回来再判一次"该去哪"**（2026-08-13 live）:
            #    原来点完就留在 switch 相位，下一帧 `observe` 把 floor 校正成 2、
            #    2 也进了 visited -> todo 空 -> 直接 exit。人到了 2 号厅，
            #    邀请和摸头**一次都没做**。
            self.goto("invite2", f"到了 {d} 号厅，接着干活")
        # 换厅必须**显式严格契约**: `observe` 每帧从屏上校正厅号，而换厅要加载
        #    好几百毫秒 -- 点击刚发出、页面还没切，下一帧就把 `floor` 校正回去，
        #    于是无限重点。到了目的厅，按钮会变成指向另一边的那个。
        proof = V.CAFE_MOVE_1F if dst == 2 else V.CAFE_MOVE_2F
        return tap_box(btn, f"去 {dst} 号厅", post=_moved, expect=(proof,))

    # exit
    def do_exit(self, obs, st):
        """收工判据：**没有黄点才算干净**（用户口述）。"""
        dots = obs.all(V.DOT_YELLOW, 0.40)
        if dots and self.state["backs"] < 4:
            # 黄点**挂在移动按钮上** = 对面厅还有活（气泡会再生），跟着走；
            #    挂在别处（礼物/家具等）= 本 flow 消不掉，别为它来回跑。
            mv = obs.find([V.CAFE_MOVE_2F, V.CAFE_MOVE_1F], 0.45)
            on_move = mv is not None and any(
                abs(d.cx - mv.cx) < 0.10 and abs(d.cy - mv.cy) < 0.08
                for d in dots)
            if on_move:
                dst = 2 if mv.cls == V.CAFE_MOVE_2F else 1

                def _mv(d=dst):
                    self.state["backs"] += 1
                    self.state.update(floor=d, pat_ts=0.0, dry=0, pans=0,
                                      patted=[])
                if not self.state.get("reswept"):
                    self.state["reswept"] = True
                    self.goto("headpat2", "黄点挂在移动按钮上，过去再扫一遍")
                return tap_box(mv, f"黄点挂在移动按钮上 - 去 {dst} 号厅",
                               post=_mv)
        pats, inv = self.state.get("pats", 0), self.state.get("invited", 0)
        if not self.hold("cafe_done", 20):
            return wait("确认一下真的没活了")
        # **黄点是竣工的权威判据**（用户 2026-08-13:「判断有没有活没干的
        #    可以靠黄点来判断」）。有黄点 = 还有活，不管我自己以为干完了没有。
        det = f"摸头 {pats} 次，邀请 {inv} 次"
        det += "，收益已领" if self.state.get("claimed") else \
               f"，**收益没领到**（{self.state.get('earn_why', '原因不明')}）"
        if dots:
            return self.finish(Outcome.LEFTOVER, f"{det}；还有 {len(dots)} 个黄点没消掉")
        if not self.state.get("claimed"):
            return self.finish(Outcome.LEFTOVER, f"{det}；黄点已无")
        return self.finish(Outcome.CLEAN, f"{det}，无黄点")

    # overlay 处理器: 相位之上，任何相位都可能弹（老代码的 global popup guards）
    def on_claim_panel(self, obs, st):
        """收敛性：面板**全灰** = 这一步做不下去了（领过了，或体力满领不动）。
        不标 `earn_done` 的话会 开面板->全灰->叉掉->又开面板 无限循环。
        **但绝不标 `claimed`** -- 全灰不等于钱到手了（见 do_earnings 的说明）。"""
        if self.phase == "earnings":
            if obs.find(V.CLAIM_ACTIVE, 0.45) is not None:
                # 收益的领取链整条走的是**覆盖层**处理器（领取 -> 結果框確認
                #    -> 再领），`do_earnings` 那一支根本没机会跑。所以
                #    `claimed` 必须在**这里**落账 —— 2026-08-13 live 实测
                #    `claims=3` 明明领到了，收尾却报「收益没领到」。
                self.state.update(claimed=True, earn_done=True)
            elif not self.state.get("earn_done"):
                self.state.update(earn_done=True,
                                  earn_why="收益面板里没有可点的领取键")
        # 邀请相位里这个面板就是邀请列表，别让基类的"领完了叉掉"去关它。
        if self.phase in ("invite", "invite2") and obs.has(V.CAFE_INVITE, 0.40):
            return None
        return super().on_claim_panel(obs, st)

    def on_confirm_dialog(self, obs, st):
        cf = obs.find(V.CONFIRM, 0.45)
        return tap_box(cf, "确认") if cf is not None else wait("等確認键")
