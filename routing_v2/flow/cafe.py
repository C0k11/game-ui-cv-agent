# -*- coding: utf-8 -*-
"""咖啡厅 —— 用户口述的权威走法（memory cafe_flow_spec）:

    弹窗叉  收益≠0 就领  邀请卷找角色（下滑） 摸头**给足时间**
     2号厅  无黄点才返回

2026-08-02 事故: 进厅的**转场帧**上 emoticon 检出了、导航 cls 还没渲染出来，
   被判成「UI 被隐藏」 盲点写死坐标 (0.12, 0.88) 唤回 UI  那一下正压在
   「編輯模式」按钮上。两层修法都保留:
      页面身份本身已经要连续 N 帧确认（state/machine.py）
      「UI 隐藏」这个**页内子状态**再单独要求连续 3 帧
   而且现在优先关弹窗：叉叉在场就先关它，不去猜 UI 是不是被藏了。
"""
from __future__ import annotations

import time
from typing import Optional

from routing_v2.act.action import Action, swipe, tap_at, tap_box, wait
from routing_v2.flow import nav
from routing_v2.flow.base import ExitMixin, Flow, Outcome
from routing_v2.percept.observe import Observation
from routing_v2.state import vocab as V
from routing_v2.state.machine import StateView

# **点气泡本身摸不到头** —— 落点要在气泡**右边一点点**，那里才是学生身体。
#    用户 2026-08-08 当场纠正；老代码 `brain/skills/cafe.py:69` 早就写着
#    `_HEADPAT_DX = 0.025  # click this much right of the bubble = student body`
#    （探针确认过的值），我搬的时候把这个偏移漏了。
HEADPAT_DX = 0.025
# 摸过的位置记下来，避免对同一个学生反复点（气泡消失有延迟）
HEADPAT_DEDUP = 0.035
HEADPAT_KEEP = 4


class CafeFlow(ExitMixin, Flow):
    name = "cafe"
    module = "cafe"
    # Emoticon_Action 已折进 ui 模型（451/452）；
    # avatar 也要：邀请列表按**学生名 cls** 找目标（凯伊/爱丽丝…是
    #    fused_avatar 的类，只跑 ui 的话 `obs.find("凯伊")` 永远 None，
    #    表现成"滑 8 次没找到"然后放弃邀请）。
    yolo = ("ui", "avatar")

    def setup(self) -> None:
        self.state.update(claimed=False, pats=0, invited=0, floor=1, backs=0,
                          pat_ts=0.0, invite_scrolls=0)

    def on_lobby(self, obs, st):
        return nav.enter(obs, V.NAV_CAFE, "咖啡厅")

    def _next_bubble(self, obs: Observation):
        """还没摸过的那个气泡。

        `Emoticon_Action` 是**弱检出**（v10 折进 ui 后 mean conf 0.65，
          35% 低于 0.55）—— 门槛必须放到 0.30/0.40 这一档，否则真气泡会被
          门槛滤掉，表现成"漏摸"（memory cafe_flow_spec 实测 650 帧）。
        气泡消失有延迟，所以要按位置去重，别对同一个学生反复点。
        """
        done = self.state.get("patted", [])
        for b in obs.all(V.EMOTICON, 0.30):
            if all((b.cx - x) ** 2 + (b.cy - y) ** 2 > HEADPAT_DEDUP ** 2
                   for x, y in done):
                return b
        return None

    # ══ 主厅 ══════════════════════════════════════════════════════════
    def _sync_floor(self, obs) -> None:
        """从**屏上 cls** 认当前在几号厅，别信自己维护的计数器。

        2026-08-12 用户抓到:「怎么咖啡厅 2 邀请了凯伊，那**咖啡厅 1 邀请的
           是谁**？说明咖啡厅 1 那会就有问题，只不过随便邀了个人出来」。
           根因：`floor` 是 flow 自己 `state.update(floor=…)` 维护的 —— 典型的
           **数意图不数事实**。它初始化成 1，可游戏完全可能停在 2 号厅
           （上一轮收尾就停在那儿），于是日志说"1 号厅"、实际在 2 号厅，
           `invited_f1` / `invited_f2` 这些按厅记的台账**全部记错格子**，
           连"这个厅的券用过没"都判错。
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

    def on_cafe(self, obs, st):
        self._sync_floor(obs)
        #  **能领就先领**。顺序不能反：收益弹窗**自己也有叉叉**，
        #    "叉叉优先"会把它关掉而不是领（2026-08-08 逐帧当场抓到）。
        #    用户口述的"进来先点叉叉"指的是**首进的角色说明弹窗**——那种弹窗
        #    上没有领取键。所以判据是「**有领取键就领，没有才关**」。
        claim = obs.find(V.CLAIM_ACTIVE, 0.45)
        if claim is not None:
            # 状态突变走 post：真点出去了才算领过（数事实不数意图）
            return tap_box(claim, f"领取（{claim.cls}）", counter="claims",
                           post=lambda: self.state.update(claimed=True))

        #  没得领  关掉挡路的弹窗（首进的角色说明弹窗）
        x = obs.find(V.CLOSE_X, 0.55)
        if x is not None:
            return tap_box(x, "关掉挡路的弹窗（屏上没有可领的东西）")

        #  打开收益面板
        if not self.state["claimed"]:
            earn = obs.find(V.CAFE_EARNINGS, 0.45)
            if earn is not None:
                return tap_box(earn, "打开咖啡厅收益面板")

        #  摸头 —— **给足时间**（用户强调）。摸完一个要等动画走完再摸下一个，
        #    不然第二下会落在动画中途的空位上。
        if self.cfg.get("headpat", True):
            dwell = float(self.cfg.get("headpat_dwell_s", 2.0))
            if time.time() - self.state["pat_ts"] < dwell:
                return wait(f"摸头动画中（等 {dwell}s）")
            emo = self._next_bubble(obs)
            if emo is not None:
                def _mark(e=emo):
                    # 真点出去了才记「摸过」。写在 decide() 里的话，被闸吞掉的
                    #    那个气泡会被记成已摸且永不重试（N5 同病）。
                    self.state["pat_ts"] = time.time()
                    self.state.setdefault("patted", []).append((e.cx, e.cy))
                    self.state["patted"] = self.state["patted"][-HEADPAT_KEEP:]
                return tap_box(emo, "摸头（落点在气泡右侧 = 学生身体）",
                               dx=HEADPAT_DX, counter="pats", post=_mark)

        #  邀请 —— **没配目标就不开列表**（邀请卷是消耗品，开了也只是空滑 8 次）
        #    邀请卷是**每厅一张**（08-08 实测：1F 用掉后 2F 的还标着
        #      「可使用」+黄点） 按厅记 flag，不是全局一次。
        if (self.cfg.get("invite_targets")
                and not self.cfg.get("skip_invite", False)
                and not self.state.get(f"invited_f{self.state['floor']}")):
            tk = obs.find(V.CAFE_TICKET, 0.45)
            if tk is not None:
                # 2026-08-12 用户 live 抓到:「进去摸完头，**刚点邀请卷就点返回
                #    出来了**」—— 这一发点完，邀请面板要几帧才弹出来，那几帧里
                #    `on_cafe` 照常从头跑一遍、 不再命中（图标被面板盖住）
                #    一路落到 「没黄点  finish」 `exit_step` 点返回，
                #    **把自己刚打开的面板关掉**。
                #    **这一处必须写显式契约，默认契约在这里不够准**
                #      （2026-08-12 用户实测:「一打开马上就关闭，然后滑动是直接
                #        滑在咖啡厅的，不是 momotalk popout」）:
                #      默认契约是 `expect_gone=(咖啡厅邀请卷,)` —— 而面板一弹出来
                #      就把券图标盖住了  契约**立刻"兑现"**  闸放行。
                #      可这时 `st.page` 还是 `cafe`（页面身份要连续
                #      `confirm_frames=3` 帧才切到 `cafe_invite_list`），
                #      于是那 1~2 帧里 `on_cafe` 照常从头跑： 看不到券
                #      （被面板盖住） 一路落到 「没活了」 **`exit_step`
                #      把自己刚打开的面板关掉**。之后的滑动就滑在咖啡厅背景上了。
                #       契约改成**等列表里的邀请键真的出现**，在那之前
                #        一发 tap 都不许出去（包括退出/滑动）。
                return tap_box(tk, f"开邀请卷（{self.state['floor']} 号厅）",
                               expect=(V.CAFE_INVITE,))

        #  换厅 —— **只主动去一次**。之后的来回全部交给  的黄点驱动，
        #    否则「floor==1 就去 2F」和「有黄点就回 1F」会互相打乒乓
        #    （08-08 实测 1F↔2F 空转了 3 个来回）。
        # 配置的厅**每个都要去一次**（2026-08-12）：原来只写了「floor==1 就去 2F」，
        #   而 `_sync_floor` 校正出真实起点可能就是 2 号厅  那样 1 号厅永远不去。
        #   仍然**每个厅只主动去一次**（`visited`），之后的来回全交给  的黄点
        #     驱动，否则「去2F」和「回1F」会互相打乒乓（08-08 实测空转 3 个来回）。
        floors = self.cfg.get("floors", [1, 2])
        cur = self.state.get("floor", 1)
        vis = set(self.state.setdefault("visited_floors", []))
        vis.add(cur)
        self.state["visited_floors"] = sorted(vis)
        todo = [f for f in floors if f not in vis]
        if todo and cur == 2 and 1 in todo:
            b1 = obs.find(V.CAFE_MOVE_1F, 0.45)
            if b1 is not None:
                def _to1():
                    self.state.update(floor=1, pat_ts=0.0)
                    self.once_reset()
                # 换厅必须**显式严格契约**（2026-08-12 我自己引入又当场抓到）:
                #    `_sync_floor` 每帧从屏上校正厅号，而换厅要加载好几百毫秒 ——
                #    点击刚发出、页面还没切，下一帧就把 `floor` 校正回 2，
                #    于是「还没去过 1 号厅  过去」**无限重点**，人始终在 2 号店。
                #    默认宽松契约（超时才 11 帧）挡不住，必须等**到了 1 号厅的
                #    标志**出现：到了 1 号厅，按钮就变成「移动至2號店」。
                return tap_box(b1, "还没去过 1 号厅  过去", post=_to1,
                               expect=(V.CAFE_MOVE_2F,))
        if (2 in floors and self.state["floor"] == 1
                and not self.state.get("seen_2f")):
            f2 = obs.find(V.CAFE_MOVE_2F, 0.45)
            if f2 is not None:
                def _to2f():
                    # claimed 不重置：收益面板是**全咖啡厅共享**的，换厅再开
                    #    一次只会全灰空转（08-08 实测白开了两回）
                    self.state.update(floor=2, seen_2f=True, pat_ts=0.0)
                    self.once_reset()
                return tap_box(f2, "去 2 号厅", post=_to2f,
                               expect=(V.CAFE_MOVE_1F,))   # 同理: 到了2号厅按钮变回"移动至一號店"

        #  收工判据：**没有黄点才算干净**（用户口述）
        dots = obs.all(V.DOT_YELLOW, 0.40)
        if dots:
            # 黄点**挂在移动按钮上** = 对面厅还有活（气泡会再生），跟着走；
            #    挂在别处（礼物/家具等）= 本 flow 消不掉，别为它来回跑。
            mv = obs.find([V.CAFE_MOVE_2F, V.CAFE_MOVE_1F], 0.45)
            on_move = mv is not None and any(
                abs(d.cx - mv.cx) < 0.10 and abs(d.cy - mv.cy) < 0.08
                for d in dots)
            if on_move and self.state["backs"] < 4:
                dst = 2 if mv.cls == V.CAFE_MOVE_2F else 1
                def _mv(d=dst):
                    self.state["backs"] += 1
                    self.state.update(floor=d, pat_ts=0.0)
                    self.once_reset()
                return tap_box(mv, f"黄点挂在移动按钮上  去 {dst} 号厅"
                                   f"（第 {self.state['backs']+1} 次跟点移动）",
                               post=_mv)
            if st.frames_in_page > 240:
                return self.finish(
                    Outcome.LEFTOVER,
                    f"还有黄点没消掉（摸头 {self.state['pats']} 次，"
                    f"邀请 {self.state['invited']} 次）")
            return wait("还有黄点 — 再扫一遍")
        if st.frames_in_page < 30:
            return wait("确认一下真的没活了")
        return self.finish(Outcome.CLEAN,
                           f"摸头 {self.state['pats']} 次，邀请 {self.state['invited']} 次，"
                           f"{self.state['floor']} 号厅结束，无黄点")

    # ══ 邀请列表 ══════════════════════════════════════════════════════
    def on_cafe_invite_list(self, obs, st):
        """在学生列表里找配置的角色。找不到就**下滑**（用户口述：下滑找）。

        滑动同样不写死次数：滑完屏上内容没变 = 到底了。
        """
        # 本厅的卷用过就走 —— 不然会顺着 targets 把第二个也邀了
        if self.state.get(f"invited_f{self.state.get('floor', 1)}"):
            # 2026-08-12 用户 live 抓到:「**根本就没邀请啊，还是有票**，准确说
            #    都没动过」「还是在打架啊，**直接点了票马上返回键跑了**」。
            #    根因：`invited_f` 挂在邀请键的 `post` 上 —— post 只保证
            #    「**tap 指令发出去了**」，而邀请**要再点一次確認才算数**
            #    （memory 链路：开券  邀请  確認）。于是 tap 一出去标记就落，
            #    下一帧这里立刻 `exit_step` **点返回**，把还没弹出来的确认框
            #    连同整个邀请一起丢掉  券没消耗、黄点还在，而日志却报"邀请了"。
            #    **post 数的是"我发了 tap"，不是"这件事完成了"** —— 这是
            #      「数事实」的边界，之前没写清楚过。
            #     退出前先给确认框 25 帧时间弹出来（overlay 优先级高于本 handler，
            #      确认框一出现就会被 `on_confirm_dialog` 接走并点掉）。
            if not self.hold("after_invite", 25):
                return wait("邀请已发出 — 等確認框弹出来（别急着退，退了这张券就白点了）")
            return self.exit_step(obs) or wait("邀请完成，等退出控件")
        targets = self.cfg.get("invite_targets") or []
        # 2026-08-08 live 实测：列表**每行自带一个邀请键**（凯伊行 cy=0.308、
        #    季行 cy=0.525 各配一个 0.96+ 的邀请键）。所以不是"选中再邀请"
        #    两步，而是**点目标同行（cy 最近）的邀请键**一步到位。
        #    全屏 argmax 找邀请键会点到**别人那一行**（840AP 同族病：
        #      任何行内按钮锚定必须带行归属）。
        # 已邀过的人跳过：两个厅的列表里**同一个学生都会出现**，2F 再邀
        #    1F 已请来的人 = 把她搬过去，白烧一张卷（08-08 差点重邀凯伊）。
        done_names = self.state.get("invited_names", [])
        for name in targets:
            if name in done_names:
                continue
            hit = obs.find(name, 0.45)      # 学生名 = avatar 模型的 cls 名
            if hit is not None:
                btns = obs.all(V.CAFE_INVITE, 0.45)
                same = min(btns, key=lambda b: abs(b.cy - hit.cy), default=None)
                # "这一帧看见目标了" = **观测事实**，留在 decide 期记是对的
                #   （同 event 的 `saw_other`）。它专门用来否掉下面那句
                #   "配置的角色都没找到" —— 08-12 live 打脸实录:
                #     `[cafe] 找到邀请目标 凯伊（行 cy=0.301）`
                #     `[cafe] 配置的邀请目标都没找到  退出列表`    自己打自己
                #   两句其实隔着好几个 tick：目标看见了，但**邀请键那一发被 JIT
                #   复验反复丢掉**（当轮 `jit_drop: 10`），而 `invite_scrolls`
                #   照样滑到 8  报"没找到"掉头就走。
                self.state["saw_invite_target"] = True
                if same is not None and abs(same.cy - hit.cy) < 0.05:
                    fl = self.state.get("floor", 1)
                    # log 挂 post：写在这里等于"我打算点"，而它可能根本没发出去
                    #   （[[log_is_not_truth]] 最贵的那条教训）。
                    def _did(f=fl, n=name, cy=hit.cy):
                        self.log(f"邀请了 {n}（行 cy={cy:.3f}）")
                        self.state[f"invited_f{f}"] = True
                        self.state.setdefault("invited_names", []).append(n)
                    act = tap_box(same, f"邀请 {name}（同行邀请键）",
                                  counter="invited", post=_did)
                    # 2026-08-12 用户实测:「咖啡厅 1 在发现问题之前就邀请出来了，
                    #    **但是不是凯伊**」—— 邀到了别人。
                    #    风险在 JIT 复验的容差上：`tap_box` 默认
                    #    `anchor_tol = max(0.025, w*0.6, h*0.6)` ≈ **0.048**，
                    #    而邀请键的**行距只有 0.107**（cy 0.308/0.415/0.525/…）
                    #     半行距 0.053，两者只差 0.005。列表哪怕只滚半行，
                    #    JIT 里 `near = min(still, 距锚点最近)` 就会取到**隔壁行**
                    #    的邀请键、drift 仍在容差内  照样放行  **点到别人那一行**。
                    #    **行内按钮的容差必须显著小于半行距**，否则"同行"这个
                    #      前提在复验阶段就被容差自己破坏掉了
                    #      （和 840AP 那次「行内按钮锚定必须带行归属」同族）。
                    act.anchor_tol = 0.030
                    return act
                return wait(f"看到 {name} 但同行邀请键没检出")
        # 2026-08-12 用户 live 抓到的核心错误:「**爱丽丝战斗就在前面，
        #    为什么会选择滑动**」+「打开了然后关闭，又在滑动，打开又关闭，
        #    整个过程是混乱的，按键逻辑是打架的」。
        #    根因链（一帧漏检引爆的死循环）:
        #      某一帧学生没检出（列表在动/动画帧） 落到下面的滑动分支
        #       **把本来就在屏上的目标滑走了**  滑到底判"没找到"  退出关面板
        #       回 `on_cafe` 又开一次券  开券/关面板反复循环。
        #      连带 `jit_drop` 飙到 10~16 —— 那是**结果不是原因**（面板被自己
        #      关掉了，邀请键在新帧上当然"已消失"）。
        #    实测感知层是好的：6 帧 × 2 目标 **12/12 全部同行匹配成功**
        #      （爱丽丝(战斗) 0.99@cy0.748 / 凯伊 0.97@cy0.302，邀请键 5 个齐全）。
        #       **不是找不到，是找到了又被自己滑掉。**
        #     只要这一轮**见过**目标，就绝不再滑：原地等它重新检出来再点。
        if self.state.get("saw_invite_target"):
            return wait("这一轮见过邀请目标 — 原地等它重新检出，**绝不滑动**"
                        "（滑一下就把眼前的目标滑走了）")
        # **进去先扫，扫完没有才滑**（2026-08-12 用户口述的正确节奏:
        #   「第一次进去就扫，**扫完没有**发送信号去滑」）。
        #   刚切进列表页的头几帧，面板还在弹入动画里、头像模型也才刚拿到
        #     第一帧 —— 这时候直接落到滑动分支，等于**没扫就滑**，
        #     把本来第一屏就有的目标滑走了（爱丽丝(战斗) 实测就在第一屏
        #     cy=0.748）。 先给模型 12 帧把这一屏扫干净。
        if st.frames_in_page < 12:
            return wait("刚进邀请列表 — 先让头像模型把这一屏扫一遍，别急着滑")
        if self.state["invite_scrolls"] < 8:
            n = self.state["invite_scrolls"] + 1      # 计数挂 post，见 sweep.py
            # 几何从检出推：邀请键就是这一列的行（见 nav.list_swipe）。
            #   找不到锚点 = 这一屏根本没有可邀请的行  别瞎滑。
            sw = nav.list_swipe(obs, [V.CAFE_INVITE],
                                f"下滑找邀请目标（第 {n} 次）",
                                post=lambda: self.state.update(invite_scrolls=n))
            if sw is not None:
                return sw
            return wait("这一屏没检出邀请键行 — 推不出滑动几何，不瞎滑")
        # 配置的角色一个都没找到  别乱邀请，直接退出（邀请卷是消耗品）
        # **见过目标就不许说"没找到"**（2026-08-12 live 自相矛盾实录）：
        #    滑到底只说明"这一帧的可视区里没有"，而目标可能早就见过、
        #    只是那一发邀请被闸吞了。此时退出 = 白白浪费本厅的邀请券。
        #     见过就退回去重滑一轮（invite_scrolls 清零），让它再试。
        #     只给一次重试机会（`retried` 标记），否则找不到同行邀请键时会死循环。
        if self.state.get("saw_invite_target") and self.pending("invite_retry"):
            self.state["once:invite_retry"] = True
            self.state["invite_scrolls"] = 0
            self.state.pop("saw_invite_target", None)
            self.log("这一轮见过邀请目标但没点成（多半被闸吞了） 重滑一轮再试")
            return wait("重滑找邀请目标")
        # note/log 挂 once：这两行原来**每帧都跑**，而 `exit_step` 要好几帧才
        #    退得出去  收尾报告里同一句刷了 18 遍（2026-08-12 live）。
        #    竣工说明是"这一轮的结论"，不是"每一帧的观感"。
        if self.pending("no_invite_note"):
            self.state["once:no_invite_note"] = True
            self.log("配置的邀请目标都没找到  不乱邀请，退出列表")
            self.note_lines.append("邀请：配置的角色没找到，未消耗邀请卷")
        return self.exit_step(obs) or wait("等退出控件")

    def on_claim_panel(self, obs, st):
        """收敛性：面板**全灰** = 收益已领过（比如 bot 中途重启过）。
        不标 `claimed` 的话会 开面板全灰叉掉又开面板 无限循环。"""
        if obs.find(V.CLAIM_ACTIVE, 0.45) is None:
            self.state["claimed"] = True
        return super().on_claim_panel(obs, st)

    def on_confirm_dialog(self, obs, st):
        cf = obs.find(V.CONFIRM, 0.45)
        return tap_box(cf, "确认") if cf is not None else wait("等確認键")
