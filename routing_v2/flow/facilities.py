# -*- coding: utf-8 -*-
"""设施类小流程：社团 / 制造 / 商店 / 邮件 / 每日任务。

每个都短，但每个都埋过雷 —— 雷在注释里。
"""
from __future__ import annotations

from typing import Optional

import routing_v2.percept.read as R
from routing_v2.act.action import Action, tap_box, wait
from routing_v2.flow import nav
from routing_v2.flow.base import ExitMixin, Flow, Outcome, qty_max_ok
from routing_v2.percept.observe import Box, Observation
from routing_v2.state import vocab as V
from routing_v2.state.machine import StateView


def _shelf_checked(obs: Observation) -> bool:
    """勾选证据. 全部选择键在场不算 -- 空闲店也有."""
    return obs.has(V.GREEN_CHECK, 0.35) or obs.has(V.SHOP_ALL_SELECTED, 0.35)


def _real_buy_selected(obs: Observation, conf: float = 0.40, *,
                       grey: bool = False, region=None):
    """底栏 450/527 必须有勾选证据.

    真「選擇購買」也在 y>0.90. v17 把大厅店空闲底栏「立即更新」训成 450,
    v18 已剔毒标但仍可能误火. 无勾选就当没看见.
    """
    name = V.SHOP_BUY_SELECTED_GREY if grey else V.SHOP_BUY_SELECTED
    kw = {} if region is None else {"region": region}
    box = obs.find(name, conf, **kw)
    if box is None:
        return None
    if box.cy <= 0.90 or _shelf_checked(obs):
        return box
    return None


def _drink_side_ok(obs: Observation, cls_: str, box) -> bool:
    """双瓶相对位: 下级在左、一般在右. 绝对 0.21/0.43 是训标货架, live 夹具在 0.76/0.90."""
    other = (V.ENERGY_DRINK_MID if cls_ == V.ENERGY_DRINK_LOW
             else V.ENERGY_DRINK_LOW)
    sib = obs.find(other, 0.40)
    if sib is None:
        return True
    if cls_ == V.ENERGY_DRINK_LOW and box.cx >= sib.cx:
        return False
    if cls_ == V.ENERGY_DRINK_MID and box.cx <= sib.cx:
        return False
    return True


#
class ClubFlow(ExitMixin, Flow):
    """社团。

    2026-08-08 实测：点大厅「社交」出来的是一层**浮层**（社團/好友/幫手 三张卡），
       底栏被压暗所以 YOLO 检不到，屏上**一个退出控件都没有**。而 `社团`(51) 这个
       cls 标的是**那张卡**，不是社团内部 —— 第一版直接在浮层上找领取按钮，
       找不到就报 "领取 0 次 CLEAN"，**根本没进过社团**。
        必须先点卡进去，进去之后才谈领取；退出靠系统返回键（浮层专用）。
    """
    name = "club"
    module = "daily_routine"

    def on_lobby(self, obs, st):
        return nav.lobby_enter(self, obs, V.NAV_SOCIAL, "社交/社团",
                               expect=(V.CLUB,))

    def _overlay(self, obs) -> bool:
        # `社团` cls 标的是社交浮层那张卡, 不是内部. 卡还在 = 没进门.
        return obs.find(V.CLUB, 0.45) is not None

    def on_club(self, obs, st):
        # 08-16 remain: 点卡 counter=entered 后 0.39s(_inside hold 12) 当内部
        #    CLEAN, 真内部加载出来已被 handoff 点返回. 浮层绝不能当内部.
        if self._overlay(obs):
            if not self.pending("enter_card"):
                return wait("已点社团卡，等浮层收起（浮层不算内部）")
            card = obs.find(V.CLUB, 0.45)
            return tap_box(card, "点「社團」卡进入社团",
                           once="enter_card", expect_gone=(V.CLUB,))
        if self.hold("no_card", 45) and self.pending("enter_card"):
            return self.finish(Outcome.UNKNOWN, "社交浮层上没检出「社團」卡")
        self.state["inside"] = True
        return self._inside(obs)

    # 08-10 live：点进社團之后页面身份**从 `club` 变成 `facility`**（社团内部
    #    是聊天页 + 成員目錄，屏上只剩返回键/回大厅 —— 正好是 facility 的签名）
    #     `on_club` 再也不会被调，第  段是**死码**，flow 停在
    #      「（无 —— 没有对应 cls，不点）」直到超时。
    #     社团内部的处理必须挂在 `on_facility` 上。
    #    实测社团的日常活儿只有一件：**进门自动弹「社團簽到獎勵 AP×10」，
    #      点確認就完了**（奖励进信箱，mail flow 会领）。聊天页/成員目錄里
    #      没有任何可领取控件 —— 所以这里不是"找不到就等"，是**本来就没有**。
    def on_facility(self, obs, st):
        if self._overlay(obs):
            return self.on_club(obs, st)
        if self.pending("enter_card") and not self.state.get("inside"):
            return None                    # 还没点过社团卡，这个 facility 不是我的
        self.state["inside"] = True
        return self._inside(obs)

    def on_offsite(self, obs, st):
        # 点卡后加载会判 blank/unknown. 默认 None 会被 recover 点返回,
        #    remain 就是内部还没出来人已经被带走.
        if not self.pending("enter_card"):
            if self._overlay(obs):
                return self.on_club(obs, st)
            return wait("社团加载中，等内部（聊天/成员），不返回")
        return None

    def _inside(self, obs) -> Optional[Action]:
        if self._overlay(obs):
            return wait("还在社交浮层，不收工")
        if not self.state.get("inside"):
            if self.hold("wait_inside", 45):
                return self.finish(Outcome.UNKNOWN,
                                   "点了社团卡但没进内部（无聊天/成员帧）")
            return wait("等社团内部加载（聊天/成员），浮层不算")
        c = obs.find(V.CLAIM_ACTIVE, 0.45)
        if c is not None:
            return tap_box(c, f"社团领取（{c.cls}）", counter="claims")
        # remain: 加载中 12 帧无领取键就 CLEAN. 内部要看够一会.
        if not self.hold("nothing", 40):
            return wait("确认社团内部没东西可领了")
        return self.finish(
            Outcome.CLEAN,
            f"社团已进入（簽到獎勵进信箱），页内额外领取 "
            f"{self.state.get('claims', 0)} 次")


#
class CraftFlow(ExitMixin, Flow):
    """制造。

    【§A6 的教科书案例】老代码的灰键闸写成
        `if find_cls(开始制造) and _btn_is_grey(那个框)`
       这个前提成立**只因为 v14 把灰键误检成亮态 444**。v15 学会了独立的
       `开始制造灰色`(485) 之后，`find_cls(开始制造)` 返回 None  **整道闸
       被跳过**  落到"MAX 外推"去盲点一个禁用按钮。
       **把模型修好，反而把为模型缺陷写的兜底打死了。**
        新写法：**先查灰态 cls，再查亮态**。顺序不能反。

    【第三态】老 skill 的红点门控只有两态（有红点=有活/无红点=没活），
       漏掉了「三个槽全空、根本没在造」这一态  永远 skip  **一次都不会
       开新制造**。这里不看红点，直接看按钮态。
    """
    name = "craft"
    module = "daily_routine"

    def on_lobby(self, obs, st):
        """进制造。**进不去就收工，别无限重敲**。

        2026-08-13 小号实测: 制造在低级号上是**锁着的** —— 点入口游戏弹一个
           单键框（「尚未開放」之类），bot 点掉確認回到大厅、再点入口、再弹…
           一轮里循环 8 次、烧掉 167 tick，最后还把大厅广告位 `購買青輝石`
           和那个框的 `确认键` 凑成"购买流程"误报 HALT，把整条日常停掉。
        用户 2026-08-13:「craft 上有个锁是进不去，但是识别不出来」——
           锁本身没有 cls（`房间区域未解锁` 只覆盖课程表房间），泛化那个 cls
           是后面的事。**现在先把"敲不开就别敲"这条纪律补上**。
        数的是**事实**（tap 真发出去了几次），不是干等了几帧。
        """
        if self.state.get("enter_taps", 0) >= 3:
            return self.finish(
                Outcome.SKIPPED,
                "点了 3 次制造入口都没进去 — 这个号的制造多半还没解锁")
        act = nav.enter(obs, V.NAV_CRAFT, "制造")
        if act is not None and act.kind == "tap":
            act.post = lambda: self.bump("enter_taps")
        return act

    def on_craft(self, obs, st):
        #  先收成品：`完成`(亮) 可领；`完成_灰色` 是还没好。
        #    `一次领取黄色` 也算 —— 快速製造是**即时完成**的，开完回到本页
        #      它就亮着（08-08 实测），只认 `完成` 会站在成品前面干等。
        if self.cfg.get("claim_finished", True):
            fin = obs.find([V.NODE_DONE, V.CLAIM_ONCE_YELLOW], 0.45)
            if fin is not None:
                return tap_box(fin, f"领制造成品（{fin.cls}）", counter="claimed")

        #  灰态 cls 优先（顺序不能反）
        grey = obs.find(V.CRAFT_START_GREY, 0.45)
        if grey is not None:
            # 灰键 ≠ 直接收工：**数量还降得下去就减 1 再试**。
            #   08-08 实测：MAX=3 时首格材料 5/15 红字「材料不足」 灰；
            #   材料够 1 发，减到 1 就能开。MIN 亮着 = 还没降到底。
            minus = obs.find(V.QTY_MINUS, 0.45)
            if (minus is not None and obs.has(V.QTY_MIN, 0.30)
                    and self.state.get("lowered", 0) < 6):
                return tap_box(minus, "開始製造灰 + 数量没到底  减 1 再试",
                               counter="lowered")
            return self.finish(
                Outcome.CLEAN,
                f"「開始製造」是灰的（cls {V.CRAFT_START_GREY} "
                f"conf {grey.conf:.2f}）且数量已到底 — 真材料不够，零无效点击")
        if obs.has(V.CRAFT_NO_MATERIAL, 0.45):
            return self.finish(Outcome.CLEAN, "屏上写着「材料不足」— 不开新制造")

        #  开新制造
        start = obs.find(V.CRAFT_START, 0.45)
        if start is not None:
            mx = qty_max_ok(obs, 0.20)
            if mx is not None and self.pending("max"):
                return tap_box(mx, "数量拉 MAX", once="max")
            return tap_box(start, "开始制造", counter="started")

        quick = obs.find(V.CRAFT_QUICK, 0.45)
        if quick is not None and self.pending("quick"):
            # 2026-08-12 用户实测:「制造那边有问题，**没有点快速制造进去**，
            #    真的就是进去看了一眼马上就回大厅了，也是按键打架」。
            #    和咖啡厅邀请券**一模一样的病**：点了"开面板"的键，面板要几帧
            #    才弹出来，而那几帧里 `on_craft` 照常从头跑一遍 —— `quick` 的
            #    `pending` 已经消耗、 也没有 `开始制造` 可点  一路落到
            #    `frames_in_page >= 40`  `finish(CLEAN, "开新制造 0 次")`，
            #    于是**报着"干净"退出，其实一次制造都没开**。
            #     显式契约：等**快速製造面板里的控件**真的出现再放行下一发。
            #      面板内必有其一：`开始制造` / `开始制造灰色` / `MAX_可点击`。
            return tap_box(quick, "快速制造", once="quick",
                           expect=(V.CRAFT_START, V.CRAFT_START_GREY, V.QTY_MAX))

        # 2026-08-12 用户复测:「制造这次进去**还是没有看到 popout 新的 cls
        #    就跑出去**，估计是点快速制造但制造界面还没加载好就没感应到」。
        #    —— 判断完全正确。契约兑现只保证"新东西出现过"，可这里的收尾判据
        #    仍然只看 `frames_in_page >= 40` **就直接下结论**：面板慢一点加载，
        #    这 40 帧就在"还没渲染出来"的空窗里烧完，然后报 `开新制造 0 次`。
        #     **点过快速製造 = 承诺过要进面板**，那就必须真看到面板控件才准收工。
        #      看不到就一直等（`no_panel` 连续 200 帧≈10s 才认输），
        #      认输时也不许报 CLEAN —— 报 UNKNOWN，让它在报告里红着。
        if self.state.get("once:quick"):
            if obs.has([V.CRAFT_START, V.CRAFT_START_GREY, V.QTY_MAX,
                        V.CRAFT_NO_MATERIAL], 0.40):
                pass                       # 面板真出来了  落到下面正常收尾
            elif self.state.get("started", 0) or self.state.get("claimed", 0):
                # 08-16 live after_craft: 开完领完面板已关, 禁止 200 帧谎报 UNKNOWN。
                #    标记留到收尾再清, 免得 pending(quick) 复活再点一次快速制造。
                pass
            elif not self.hold("no_panel", 200):
                return wait("已点快速製造，但面板控件还没渲染出来 — 等它，"
                            "别在没看到 popout 的情况下下结论")
            else:
                return self.finish(
                    Outcome.UNKNOWN,
                    "点过快速製造，但等了 200 帧都没看到面板里的任何控件 "
                    "— 制造到底开没开**不确定**（别报干净）")
        if st.frames_in_page < 40:
            return wait("等制造页控件")
        # 08-16 live: 开完/领完必须清 once:quick, 成功不得报 UNKNOWN。
        self.state.pop("once:quick", None)
        return self.finish(
            Outcome.CLEAN,
            f"领成品 {self.state.get('claimed',0)} 次，"
            f"开新制造 {self.state.get('started',0)} 次")

    def on_confirm_dialog(self, obs, st):
        # 灰确认 = 条件不满足，取消掉别硬点
        if obs.has(V.CONFIRM_GREY, 0.45):
            c = obs.find(V.CANCEL, 0.45)
            return tap_box(c, "确认键是灰的  取消") if c is not None else wait("等取消键")
        cf = obs.find(V.CONFIRM, 0.45)
        return tap_box(cf, "确认") if cf is not None else wait("等確認键")


#
class FreePackFlow(ExitMixin, Flow):
    """日常第一枪: 大厅 `购买青辉石` -> 组合包页 -> 只领免费每日。

    入口/页签/领取全用已有 cls, OCR 不主导。领完退回大厅, 再交班社团。
    只点免费/领取, spend 空; 购买键和花石确认 fail-closed 不点。
    """
    name = "free_pack"
    module = "daily_routine"
    entry_page = "lobby"

    def setup(self) -> None:
        if not self._want_free_pack():
            self.state["pack_done"] = True
            self.state["skipped"] = True
            return
        from routing_v2.flow import daybook
        if daybook.done("free_pack", self.ctx.cfg):
            self.state["pack_done"] = True
            self.log("免費包本游戏日已领过（台账）-- 这一枪跳过")

    def _want_free_pack(self) -> bool:
        if "buy_free_pack" in (self.cfg or {}):
            return bool(self.cfg.get("buy_free_pack"))
        shop = (self.ctx.cfg.get("shop") or {})
        return bool(shop.get("buy_free_pack", True))

    def _mark_free_pack_done(self) -> None:
        from routing_v2.flow import daybook
        self.state["pack_done"] = True
        daybook.mark("free_pack", self.ctx.cfg)

    def on_lobby(self, obs, st):
        if self.state.get("pack_done") or self.state.get("skipped"):
            why = ("免费包已关" if self.state.get("skipped")
                   else "免费包本游戏日已领过(台账)")
            return self.finish(
                Outcome.SKIPPED if self.state.get("skipped") else Outcome.CLEAN,
                why)
        p = obs.find(V.SHOP_BUY_PYROXENE, 0.40)
        if p is not None and self.pending("open_pack"):
            return tap_box(p, "日常第一枪: 购买青辉石入口  进页领免费包",
                           once="open_pack",
                           expect=(V.COMBO_PACK, V.COMBO_PACK_SEL, V.FREE))
        if self.hold("no_entry", 30):
            return self.finish(Outcome.UNKNOWN, "大厅没检出购买青辉石入口")
        return wait("等购买青辉石入口")

    def on_combo_pack(self, obs, st):
        """组合包页。只点免费/领取, 不点旁边的购买键。

        帧 main_combo.jpg: 每日免费组合包那颗蓝键 YOLO 同时出 免费 0.97
           和 购买 0.96。旧代码点购买 + money=True, 没 --money-ok 就被拒。
           现在只点免费, spend 空。分不清免费还是购买: 不点购买。
        """
        if not self._want_free_pack() or self.state.get("pack_done"):
            return self.exit_step(obs) or wait("等退出控件")

        free = obs.find(V.FREE, 0.30)
        if free is None:
            tab = obs.find(V.COMBO_PACK, 0.40)
            if tab is not None and self.pending("combotab"):
                return tap_box(tab, "切到 組合包 tab", once="combotab")
            if not self.hold("no_free", 30):
                return wait("找免费标, 连续确认中")
            self.log("組合包 tab 里没有免费标(今天领过了?)  一个都不碰, 退出")
            return self.exit_step(obs) or wait("等退出控件")

        greys = [b for b in obs.all(V.SHOP_BUY_GREY, 0.30)
                 if abs(b.cx - free.cx) < 0.06]
        buys = [b for b in obs.all(V.SHOP_BUY, 0.35)
                if abs(b.cx - free.cx) < 0.06]
        if greys and not buys:
            if self.hold("free_greyed", 8):
                self._mark_free_pack_done()
                self.log("免費組合包今天已经领过了(购买键是灰的)  跳过")
                return self.exit_step(obs) or wait("等退出控件")
            return wait("免费列购买键是灰的, 确认已领")

        claim = None
        for cls in (V.CLAIM_YELLOW, V.CLAIM_ONCE_YELLOW,
                    V.CLAIM_ALL_YELLOW, V.CLAIM_BLUE):
            c = obs.find(cls, 0.40)
            if c is not None and abs(c.cx - free.cx) < 0.10:
                claim = c
                break
        target = claim or free
        return tap_box(target, "领免费包(只点免费/领取, 不点购买)",
                       money=False, spend="",
                       expect=(V.CONFIRM, V.GOT_REWARD, V.SHOP_BUY_GREY))

    def on_confirm_dialog(self, obs, st):
        if obs.has(V.CONFIRM_GREY, 0.45):
            c = obs.find(V.CANCEL, 0.45)
            return tap_box(c, "灰确认  取消") if c is not None else wait("等取消")
        pyx = obs.find(V.PYROXENE, 0.40, region=(0.12, 0.12, 1.0, 1.0))
        if pyx is not None and not obs.has(V.FREE, 0.30):
            c = obs.find(V.CANCEL, 0.45)
            if c is not None:
                return tap_box(c, f"框内出现青辉石价签（conf {pyx.conf:.2f}） 取消")
            return wait("框内有青辉石价签但没找到取消键, 绝不点确认")
        cf = obs.find(V.CONFIRM, 0.45)
        if cf is None:
            return wait("等确认键")
        if obs.has(V.FREE, 0.30):
            return tap_box(cf, "确认领取免费包", money=False, spend="",
                           post=self._mark_free_pack_done)
        c = obs.find(V.CANCEL, 0.45)
        if c is not None:
            return tap_box(c, "确认框没有免费标  fail-closed 取消")
        return wait("确认框没有免费标, 绝不点确认")

    def on_shop(self, obs, st):
        return self.exit_step(obs) or wait("免费包链不买信用点货架")


#
class ShopFlow(ExitMixin, Flow):
    """信用点商店 + 战术大赛商店。免费包已拆到 FreePackFlow(日常第一枪)。

    落到青辉石 tab 只切回信用点/退出, 不点货架。
    刷新商店要花青辉石, `shop.refresh_times` 在 config 里 LOCKED 成 0。
    `shop.credit_buy` 默认关: 进店不点选择购买。`shop.arena_shop` 默认开: 去买饮料。
    """
    name = "shop"
    module = "daily_routine"

    def on_lobby(self, obs, st):
        # 两段都干完了就收工 —— 08-11 live 实锤：`bought`/`arena_done`
        #    全 True，退回大厅后这里仍然无条件 `nav.enter(商店)`，
        #    于是**再进一次商店**。
        if self._all_done():
            return self.finish(Outcome.CLEAN,
                               "信用点商店 / 战术大赛商店 两段都处理完了")
        # 只点底栏商店入口, 不要点到左侧购买青辉石广告位.
        box = obs.find(V.NAV_SHOP, 0.45, region=(0.35, 0.88, 0.85, 1.0))
        if box is None:
            box = obs.find(V.NAV_SHOP, 0.45, region=(0.0, 0.88, 1.0, 1.0))
        if box is not None:
            return tap_box(box, "底栏进商店")
        return wait("等底栏商店入口")

    def _shop_opt(self, key: str, default):
        """商店开关在 cfg.shop。测试可以写在 flow.cfg 上盖掉。"""
        if key in (self.cfg or {}):
            return self.cfg[key]
        return (self.ctx.cfg.get("shop") or {}).get(key, default)

    def _credit_buy_on(self) -> bool:
        return bool(self._shop_opt("credit_buy", False))

    def _arena_shop_on(self) -> bool:
        return bool(self._shop_opt("arena_shop", True))

    def _all_done(self) -> bool:
        """两段各自「做完或本就不做」。关掉的那段不能算没做完，否则永不收工。"""
        # `arena_skip`（左栏翻遍也没找到 tab）也要算"这段做完了"，否则
        #   08-12 拆开 skip/done 语义之后会**永不收工** —— 正是 day4 那个
        #   「三段做完不收工  死循环，日志看起来一切正常」的复发路径。
        #   `arena_skip` 是可被推翻的：真站到 arena_shop 上会把它 pop 掉，
        #     那时 `arena_done` 才是唯一出口。
        arena_ok = (self.state.get("arena_done")
                    or self.state.get("arena_skip")
                    or not self._arena_shop_on())
        credit_ok = (self.state.get("bought")
                     or not self._credit_buy_on())
        return bool(credit_ok and arena_ok)

    def on_shop_pyroxene_tab(self, obs, st):
        self.log("在青辉石 tab 上 — 立刻切回信用点 tab")
        act = self._to_credit_tab(obs)
        if act is not None:
            return act
        return wait("青辉石 tab 上, 等切回信用点(不退出、不买)")

    def _to_credit_tab(self, obs: Observation) -> Optional[Action]:
        """信用点商店未选中态 train=0, 检不出就点青辉石选中态上方那一格(一般)."""
        t = obs.find(V.SHOP_TAB_CREDIT, 0.35)
        if t is not None and self.pending("credittab"):
            return tap_box(t, "切到信用点 tab（安全 tab）", once="credittab")
        pyx = obs.find(V.SHOP_TAB_PYROXENE_SEL, 0.40,
                       region=(0.0, 0.08, 0.22, 0.98))
        if pyx is not None and self.pending("credittab"):
            h = max(pyx.y2 - pyx.y1, 0.04)
            cy = max(0.12, pyx.cy - h * 1.15)
            fake = Box(cls=V.SHOP_TAB_CREDIT, conf=0.40,
                       x1=pyx.x1, y1=cy - h / 2, x2=pyx.x2, y2=cy + h / 2)
            return tap_box(fake, "青辉石上方点一般/信用点 tab", once="credittab")
        return None

    def _on_credit_shelf(self, obs: Observation) -> bool:
        if obs.has(V.SHOP_TAB_PYROXENE_SEL, 0.40):
            return False
        arena_sel = obs.find(V.ARENA_SHOP_TAB_SEL, 0.40)
        if arena_sel is not None and arena_sel.cx >= 0.22:
            return False
        if obs.has(V.SHOP_TAB_CREDIT_SEL, 0.70):
            return True
        # 未选中态 train=0: 全部选择/选择购买系列 + 货架信用点价签.
        # 大赛货架也有全部选择 0.98, 但价签是大赛币不是信用点.
        sel = (obs.has([V.SHOP_SELECT_ALL, V.SHOP_SELECT_ALL_GREY], 0.35)
               or _real_buy_selected(obs, 0.35) is not None
               or _real_buy_selected(obs, 0.35, grey=True) is not None)
        shelf_credit = obs.all(V.CREDIT, 0.30, region=(0.20, 0.15, 1.0, 0.92))
        return bool(sel) and len(shelf_credit) >= 1

    def _segments(self) -> str:
        """收尾说清**三段各自**做没做（2026-08-13）。

        原来两处都写死「信用点商店已处理」—— 而这一轮实际成交的是**大赛商店**
           （屏上大赛币 3,133->3,088、体力 +90），报告却只字未提。
        竣工判据要说清干了什么，不是报个 CLEAN 就算（[[completion_gap]]）。
        """
        seg = []
        # 「买不起」和「买过了」是两件事 —— 都靠 `bought` 收敛，但报告要分开说，
        #    否则小号上那句「信用点商店 已处理」就是谎报（同 cafe 领收益那条）。
        seg.append("信用点商店 " + ("**买不起**（信用点不够）"
                                  if self.state.get("credit_short")
                                  else "已处理" if self.state.get("bought")
                                  else "未处理"))
        if self.state.get("credit_off"):
            seg[-1] = "信用点商店 已关（不买）"
        if not self._arena_shop_on():
            seg.append("战术大赛商店 已关")
        elif self.state.get("arena_short"):
            seg.append("战术大赛商店 **买不起**（大赛币不够）")
        elif self.state.get("arena_done"):
            seg.append("战术大赛商店 已处理")
        elif self.state.get("arena_skip"):
            seg.append("战术大赛商店 **没找到入口**")
        else:
            seg.append("战术大赛商店 未处理")
        return " / ".join(seg)

    def on_shop(self, obs, st):
        if obs.has(V.SHOP_TAB_PYROXENE_SEL, 0.40):
            return self._to_credit_tab(obs) or wait("青辉石 tab, 先切回信用点")
        on_credit = self._on_credit_shelf(obs)
        # credit_buy 关: 不点全部选择/选择购买, 这段记跳过, 去大赛店买饮料.
        if not self._credit_buy_on():
            if not self.state.get("bought"):
                self.state["bought"] = True
                self.state["credit_off"] = True
                self.log("信用点商店买已关 — 不点选择购买, 转战术大赛商店")
        else:
            if not on_credit:
                act = self._to_credit_tab(obs)
                if act is not None:
                    return act

        # 大厅信用点店: 全部选择 + 选择购买/选择购买灰色. 不滑货架单卡买.
        # 认不出信用点货架就不买(大赛货架上全部选择 0.98 的 fail-open).
        if (self._credit_buy_on()
                and not self.state.get("bought") and on_credit):
            sel = obs.find(V.SHOP_SELECT_ALL, 0.40)
            if (sel is not None and self.pending("selectall")
                    and not obs.has(V.GREEN_CHECK, 0.40)):
                return tap_box(sel, "全部选择", once="selectall",
                               expect=(V.GREEN_CHECK,
                                       V.SHOP_BUY_SELECTED,
                                       V.SHOP_BUY_SELECTED_GREY))
            if (_real_buy_selected(obs, 0.40, grey=True) is not None
                    and _real_buy_selected(obs, 0.40) is None
                    and self.hold("cant_afford", 8)):
                self.state["bought"] = True
                self.state["credit_short"] = True
                self.log("「選擇購買」是灰的 - 信用点不够，这一段没得买")
            buy = _real_buy_selected(obs, 0.40)
            if buy is not None:
                # 花钱只认买AP/买票/CAD 确认。信用点批量买只记账, 不走人审,
                #    也不 HALT。用户抽卡造成的石/信用下降同口径: EXTERNAL 更新基线。
                return tap_box(buy, "批量购买（花信用点）",
                               spend="信用点",
                               post=lambda: self.state.update(bought=True))
        if (self._credit_buy_on()
                and obs.has(V.SHOP_SELECT_ALL_GREY, 0.40)
                and not obs.has(V.SHOP_SELECT_ALL, 0.40)
                and self.hold("selall_grey", 20)):
            self.state["bought"] = True
            self.log("「全部選擇」持续是灰的且没有亮态 — 今天这栏买过了")

        #  信用点这一栏处理完  **转战术大赛商店**（同一个商店页的另一个 tab）
        act = self._goto_arena_tab(obs)
        if act is not None:
            return act

        if st.frames_in_page < 40:
            return wait("确认商店没东西可买了")
        # **信用点这一栏完了不等于整条 flow 完了**（用户 2026-08-13:「也没去
        #    战术大赛商店那边接着选择饮料然后检测」）。
        #    原来只要 `bought` 一 True 就 `finish` —— 而 `_goto_arena_tab` 上面
        #    那一步返回 None（滑不出 tab / 推不出滑动几何）时就直接掉到这里，
        #    **大赛商店整段被跳过，报告还写 CLEAN**。
        #    收工判据必须是「三段都有交代」，不是「第一段做完了」。
        if (self._arena_shop_on()
                and not self.state.get("arena_done")
                and not self.state.get("arena_skip")):
            if not self.hold("no_arena_tab", 60):
                return wait("信用点这栏完了 — 找战术大赛 tab（还没放弃）")
            self.state["arena_skip"] = True
            self.log("左栏里滑不出战术大赛 tab — 这一段记成没找到入口，不谎报")
        if self.state.get("bought"):
            return self.finish(Outcome.CLEAN, self._segments())
        # 2026-08-12 live 实锤：**「没买成」不等于「没得买」**。
        #    `bought` 走 post，被金钱闸拦下时**正确地**保持 False（数事实）——
        #    但这里直接把 False 读成"没有可买项"报 CLEAN，于是 24 件商品
        #    明明躺在货架上、一件没买，收尾却说"干净"。
        #    这是 [[completion_gap]]「验代码跑通≠验活干完」在新架构的复发。
        #     屏上还有购买控件 = 有货没买成  **LEFTOVER**，让它在报告里红着。
        #    credit_buy 关时货架上仍有选择购买是预期, 不许报没买成.
        left = (_real_buy_selected(obs, 0.40)
                or obs.find(V.SHOP_SELECT_ALL, 0.40)
                or obs.find(V.SHOP_ALL_SELECTED, 0.40))
        if left is not None and self._credit_buy_on():
            return self.finish(Outcome.LEFTOVER,
                               f"货架上还有可买项（`{left.cls}` conf {left.conf:.2f}）"
                               f"但一件都没买成 — 多半是金钱步没放行")
        return self.finish(Outcome.CLEAN, "信用点商店没有可买项；" + self._segments())

    #  战术大赛商店
    # 用户 2026-08-10 点名的缺失板块：「战术大赛商店应该也是商店部分的」。
    #   它不是独立页面，就是**商店页左栏的一个 tab**，开局在「一般」栏，
    #   战术大赛沉在折叠线以下 —— 左栏往下滑一次就露出来（用户："栏目滑动
    #   一下就看到了啊"）。老 skill(brain/skills/arena_shop.py) 有完整 spec，
    #   这里按新架构重写：**落点全部 based on cls，零硬坐标**。
    #   老代码用了写死的 (0.068,0.395)，理由是"cls469 只有 27 个训练实例、
    #     YOLO 检不出" —— 08-10 实测 v15 **检出 conf 0.95**（样本已涨到 95 框），
    #     所以那条硬坐标的前提**已经不成立**，不搬过来。
    def _goto_arena_tab(self, obs: Observation) -> Optional[Action]:
        if not self._arena_shop_on() or self.state.get("arena_done"):
            return None
        if not self.state.get("bought"):
            return None                    # 先把信用点那栏做完
        # **屏上还压着东西的时候不许碰左栏**（2026-08-13 用户点名:「这次是左边
        #    栏目该滑动了，没检测到就滑动一次**再检测**」—— 规矩没错，是执行
        #    环境不对）。轨迹实证:
        #      t8  点「選擇購買」(会弹确认框)
        #      t17 滑左栏 第1次   <- 确认框还压在屏上
        #      t25 点確認         <- 确认框到这儿才处理
        #      t28/t38 第2/3次    <- 还在弹窗+奖励期, 3 次预算烧光
        #    三次全滑在被盖住的屏上：滑了没动、扫也扫不到，于是
        #    「滑一次扫一次」退化成「猛滑三次」，然后判定"没有这个 tab"。
        #     对话框/奖励框在场就**等**，别动左栏也别消耗滑动预算。
        #      返回 wait 而不是 None：None 会让 on_shop 一路走到 finish(CLEAN)，
        #      把整个战术大赛商店跳过（这一轮就是这么丢的）。
        if obs.has([V.CONFIRM, V.CANCEL, V.GOT_REWARD, V.STORY_TAP_CONTINUE,
                    V.CLOSE_X], 0.40):
            return wait("屏上还压着对话框/奖励框 — 左栏被盖住，先不滑")
        tab = obs.find(V.ARENA_SHOP_TAB, 0.40)
        # cls469 会整帧漏检, 但 tab 行上的**币图标**检得出（2026-08-13 实帧:
        #    滑动其实成功了、「戰術大賽」行已露出, cls469 零检出而
        #    `战术大赛商店货币 0.72` 就打在那一行上 —— 只认 469 就误判
        #    "没入口"收工）。左栏区域(cx<0.22)里的币图标 = tab 行本身,
        #    点它 = 点 tab。region 排掉顶栏余额(cy<0.10)和货架价签(cx>0.4)。
        if tab is None:
            tab = obs.find(V.ARENA_SHOP_CURRENCY, 0.40,
                           region=(0.0, 0.40, 0.22, 0.98))
        # 2026-08-12 用户抓到「战术大赛商店点选项打架了，我才发现原来我们
        #    今天没买能量饮料」。根因之一：这里只写了 `once="arenatab"` 却
        #    **没配 `self.pending()` 检查** —— `once=` 的作用只是"tap 真发出去
        #    之后落个标记"，它**本身不阻止重复决策**。于是 live 连点两次：
        #    `@(0.067,0.503)` 和 `@(0.067,0.442)`（左栏滚了、落点变了，
        #    连 dedup 都当成新目标放行）。 补上 pending 闸。
        if tab is not None and self.pending("arenatab"):
            # **滑完要等左栏停稳再点**（2026-08-13 live 实锤: t30 滑、t40 点，
            #    那一刻列表还在惯性滚，落点落到了上一行 —— 点到了**大決戰**）。
            #    JIT 会把落点校到最新帧的框上，但"最新帧"本身还在动，校了也白校。
            #     要求这个 tab 的 cy 连续几帧几乎不动才允许点。
            prev = self.state.get("tab_cy")
            self.state["tab_cy"] = tab.cy
            if prev is None or abs(prev - tab.cy) > 0.006:
                self.once_reset("tabsteady")
                self.state["tab_steady"] = 0
                return wait(f"「戰術大賽」tab 还在动"
                            f"（cy {prev if prev is None else round(prev,3)}"
                            f" -> {tab.cy:.3f}）— 等停稳再点")
            n = int(self.state.get("tab_steady", 0)) + 1
            self.state["tab_steady"] = n
            if n < 4:
                return wait(f"「戰術大賽」tab 稳定 {n}/4 帧")
            return tap_box(tab, "切到「戰術大賽」商店 tab", once="arenatab")
        if tab is not None:
            return wait("已经点过「戰術大賽」tab 了 — 等页面切过去，别重复点")
        # **点过之后就绝不再滑**（2026-08-13 live 轨迹实锤）:
        #    t588 滑第 1 次  t596 看到 tab 并点了  t600 **又滑了第 2 次**。
        #    根因：上面那道 `pending` 只守住了「tab 检出到」的分支，而切页面的
        #    过渡帧上 tab 会瞬时检不出  `tab is None`  直接掉进下面"还没露出来
        #    就滑"的分支。用户原话:「看到了就不滑了，还是有逻辑打架」。
        #     滑动的前提不是"这一帧没看见"，而是"**从来没看见过**"。
        #    这和 §A3「单帧当真相」是同一族：一帧没检出不等于它不在。
        if not self.pending("arenatab"):
            return wait("已经点过「戰術大賽」tab（这一帧没检出而已）— 不再滑")
        # 还没露出来  左栏往下滑。**滑动的 x 也要 based on cls**：
        #   拿当前选中的那个 tab 的 cx 当左栏轴线，别写死 0.068。
        col = obs.find([V.SHOP_TAB_CREDIT_SEL, V.SHOP_TAB_CREDIT,
                        V.ARENA_SHOP_TAB_SEL], 0.35)
        if col is None or col.cx > 0.20:
            return None                    # 找不到左栏轴线就别瞎滑
        n = int(self.state.get("tabscroll", 0))
        if n >= 3:                         # capped（用户："滑一下就行别猛滑"）
            # 2026-08-12：这里原来设的是 `arena_done=True` —— 和"**买完了**"
            #    共用同一个标记。后果在 live 上非常刺眼：
            #      `左栏滑到底也没见到 tab  跳过`  下一帧 `shop  arena_shop`
            #      （**其实已经进去了**，只是点击生效慢） 而 `on_arena_shop`
            #      第一行看到 `arena_done` 就 `_back_to_credit_tab` **退了出去**
            #     站在战术大赛商店里、货架就在眼前，却因为上一帧的结论掉头就走，
            #      今天的能量饮料就是这么没买成的。
            #     拆成两个语义：`arena_skip`=没找到入口（可被"真进去了"推翻），
            #      `arena_done`=真买完了（终态）。
            self.state["arena_skip"] = True
            self.log("左栏滑到底也没见到「戰術大賽」tab  暂时跳过"
                     "（若之后页面真变成 arena_shop，这个结论会被推翻）")
            return None
        # 计数挂 post（我自己 08-10 写的时候又犯了同一个 decide 期突变，
        #    半小时后在 sweep.py 抓到同族 bug 才回头把这处一起改了）
        # 几何全部从检出推（x 本来就是 col.cx，y 之前还写死 0.75->0.35）。
        #   左栏 tab 的框高就是"一行多高"，滑 3 行。
        sw = nav.list_swipe(
            obs, [V.SHOP_TAB_CREDIT_SEL, V.SHOP_TAB_CREDIT,
                  V.ARENA_SHOP_TAB, V.ARENA_SHOP_TAB_SEL],
            f"左栏往下滑露出「戰術大賽」tab（第 {n+1} 次）",
            post=lambda: self.state.update(tabscroll=n + 1))
        if sw is not None:
            return sw
        return wait("左栏没检出 tab 锚点 — 推不出滑动几何，不瞎滑")

    def on_arena_shop(self, obs, st):
        """**绝不用「全部選擇」**（08-10 帧证）：这一栏货架上除了两瓶
        能量饮料，还有 6 个「XX的神名文字」x5 各 50 币 —— 全选会把它们
        一起买走。 只逐件点白名单（下級/一般能量飲料），别的一律不碰。

        买的是**战术大赛货币**（非 premium，打大赛白拿的），不是青辉石。
        但闸照旧：余额读不出  fail-CLOSED 不买。
        """
        # **站到这一页本身就推翻"没找到入口"的结论**（2026-08-12 live 实锤：
        #   报了"跳过"之后下一帧页面就变成 arena_shop 了）。到都到了，就买。
        if self.state.pop("arena_skip", None):
            self.state["tabscroll"] = 0
            self.log("其实已经站在战术大赛商店里了 — 撤销刚才那个「跳过」结论")
        # 2026-08-12 用户实测:「战术商店，体力饮料**选了购买的时候点的取消**
        #    然后就跑了」—— 截图铁证：两瓶饮料購買键还亮着、勾选框全空、
        #    大赛币 3,178 一分没花。
        #    根因和 cafe 的 `invited_f` **一模一样**：`arena_done` 挂在
        #    「選擇購買」的 `post` 上  tap 一发出去就标记"这段做完了"，
        #    可购买**还要再点一次確認**才算数。下一帧 `arena_done` 已 True
        #     `_back_to_credit_tab`  `exit_step`  「取消（模态框唯一有效出口）」
        #     **把自己刚弹出来的购买确认框取消掉**。
        #    （`exit_step` 本身没错：模态框上取消确实是唯一安全出口；
        #      错在**确认框还开着的时候就去执行退出**。）
        #     屏上还有双键确认框时，什么都不做，交给 `on_confirm_dialog`。
        if obs.has(V.CONFIRM, 0.45) and obs.has(V.CANCEL, 0.45):
            return wait("购买确认框还开着 — 交给 confirm_dialog 处理，绝不在这儿退出")
        if self.state.get("arena_done"):
            return self._back_to_credit_tab(obs) or self.finish(
                Outcome.CLEAN, "战术大赛商店已处理")

        #  tab lock：确认真的站在大赛商店里才允许点货架（金钱防线第一层）。
        #    别只认 `战术大赛商店已选择` —— 08-11 live：tab 已高亮，该 cls
        #    conf 仅 **0.116**（选中态几乎没训过） 这道门永远打不开，
        #    整条支线静默失效。改成「tab 选中态 **或** 货架价签是大赛币」，
        #    后者是**多实例** 0.97，且信用点/活动商店的价签是别的 cls，不会串。
        shelf = [b for b in obs.all(V.ARENA_SHOP_CURRENCY, 0.35) if b.cy >= 0.20]
        if not (obs.has(V.ARENA_SHOP_TAB_SEL, 0.40) or len(shelf) >= 2):
            return wait("等「戰術大賽」tab 选中确认（tab cls 或货架大赛币价签）")

        #  余额闸 fail-CLOSED。余额锚是**最上面那个**货币图标（cy≈0.12）；
        #    货架价签上也是同一个 cls，全屏 argmax 会抓到价签  必须按 cy 取最小。
        #    （「任何顶栏读数必须带 region」那条教训的同形。）
        #    顶部余额锚**稳定欠拟合**（08-11 实测连续 3 帧 conf 0.281/0.282/0.311，
        #      始终够不到 0.35） 恒 None  fail-closed 一件不买，和 08-10
        #      「活动商店余额锚点错整天一件没买」是同一个病。
        #      降到 0.20 是**带 region 的**降阈值：顶栏 cy<0.20 只有这一个大赛币图标，
        #      货架价签（0.97）在 cy≥0.20，不会被抓错。别在没有 region 时这么降。
        coins = [b for b in obs.all(V.ARENA_SHOP_CURRENCY, 0.20) if b.cy < 0.20]
        bal = None
        if coins:
            r = R.read_beside(obs, min(coins, key=lambda b: b.cy),
                              R.STRIP[R.ARENA_COIN])
            bal = r[0] if r and r[0] is not None else None
        if bal is None:
            # 读不出不再一票否决（勾选探针上线后, 成交判据是「選擇購買」的
            #    亮灰, 那是游戏自己的授权; 花的又是非 premium 的大赛币,
            #    真钱防线在 money=True 的人审闸上）。读数降级成参考。
            self.log("战术大赛货币余额读不出 — 不拦, 以勾选后的按钮状态为准")

        #  白名单逐件点选（未勾选的才点；勾选后 `选择购买` 会出现）
        #    **余额读数不许一票否决**（用户 2026-08-13:「也没选饮料然后辨别
        #    是否买得起啊？」）。读数是 OCR，错一次就永远不买；真判据是
        #    你定的那条链: **勾选 -> 看「選擇購買」亮还是灰**。勾选本身
        #    不花一分钱（钱只在 選擇購買+確認 那两步动），拿它当探针是安全的。
        #    余额只降级成日志。
        for cls_, price in ((V.ENERGY_DRINK_MID, 30), (V.ENERGY_DRINK_LOW, 15)):
            if self.state.get(f"picked:{cls_}"):
                continue
            it = obs.find(cls_, 0.40)
            if it is None:
                self.state[f"picked:{cls_}"] = True     # 售罄/不在货架上
                continue
            if not _drink_side_ok(obs, cls_, it):
                self.log(f"{cls_} 与对侧瓶位冲突 — 拒勾, 记 dirty")
                self.state[f"picked:{cls_}"] = True
                self.state["drink_side_dirty"] = True
                continue
            if bal is not None and bal < price:
                self.log(f"余额读数 {bal} < {cls_} 单价 {price} — 大概率买不起，"
                         f"但以勾选后的按钮状态为准（读数只是参考）")
            # 2026-08-12 用户实测:「两瓶饮料是 **dim 状态说明买过了**，
            #    但 agent 不知道，**还是点了然后傻等了一会儿就跑**」。
            #    根因：已买过的商品只是**变灰**，cls 照样检出  照样点  点不动。
            #    不能靠 `购买灰色` 判——那个 cls 在买完态实测 0 检出（欠拟合，
            #      [[routing_v2_live_day3]] 三处实证），拿它当判据等于没判。
            #    用**结构判据**：勾选成功**必然**让「選擇購買」出现。
            #       契约 `expect=(选择购买,)`；契约超时 = 根本没勾上 = 已买过/买不了。
            #      这就是「看下一步 cls 再判断」在这里的正确落地。
            # 契约加两个备份证据: 卡上的**绿勾**(勾选成功的直接产物, 0.97)
            #    和**灰的選擇購買**(勾上了但买不起, 本页实测只有 0.33,
            #    低于全局 0.40 -- 所以下面判灰要带 region 降阈值)。
            return tap_box(it, f"勾选 {cls_}（{price} 大赛币，余额读数 {bal}）",
                           expect=(V.GREEN_CHECK,),
                           post=lambda c=cls_: self.state.update({f"picked:{c}": True}))

        #  成交 / 买不起 -- 判据就是「選擇購買」的亮灰（用户口述的链条）
        #    450 本身不进勾选契约: 空闲底栏误火会瞬间兑现 (v18 P0).
        buy = _real_buy_selected(obs, 0.40)
        if buy is not None:
            return tap_box(buy, f"購買所選能量飲料（余额读数 {bal} 大赛币）",
                           spend="战术大赛货币",
                           post=lambda: self.state.update(arena_done=True))
        # 灰态判定带 region 降阈值（按钮固定在右下角; 本页灰态实测仅 0.33,
        #    全局 0.40 门槛看不见它 -- [[routing_v2_live_day4]] 大赛币余额锚
        #    「带region降阈值」同一个方法）。亮态在场不算（半渲染两态同检）。
        grey = _real_buy_selected(obs, 0.25, grey=True,
                                  region=(0.70, 0.80, 1.0, 1.0))
        if grey is not None and self.hold("arena_short", 8):
            self.state["arena_done"] = True
            self.state["arena_short"] = True
            self.log("「選擇購買」是灰的 - 大赛币不够，这一段没得买（勾选探针给的结论）")
            return wait("大赛币不够，收工")
        # 等待缩短到 12 帧（用户:「点了然后**傻等了一会儿**就跑」）:
        #   勾选的契约已经在等「選擇購買」了，走到这儿说明契约都超时了，
        #   再干等只是把时间烧掉。**dim 态（已买过）就该快速判定并收工。**
        if self.hold("no_buy_btn", 12):
            self.state["arena_done"] = True
            # 「买不起」和「已买过」是两个结论, 靠**绿勾**区分（2026-08-13 实帧:
            #    余额 0 勾上饮料后 選擇購買 灰得肉眼可见, 但 cls527 在这页
            #    连 0.25 都不到 -- 上面那条灰态判据是瞎的。而绿勾 0.96 稳定:
            #    勾选成功了 + 亮的選擇購買始终不出现 = 买不起;
            #    连勾都勾不上（无绿勾）= dim 态, 今天已买过）。
            if obs.has(V.GREEN_CHECK, 0.40, region=(0.45, 0.15, 1.0, 0.75)):
                self.state["arena_short"] = True
                return wait("勾上了但「選擇購買」一直不亮 — 大赛币不够，收工")
            return wait("勾不上、也没出现「選擇購買」— 饮料是 dim 态（今天已买过），收工")
        return wait("等「選擇購買」出现")

    def _back_to_credit_tab(self, obs: Observation) -> Optional[Action]:
        """买完别停在货架上。

        08-10 live：这里原本只写"切回信用点 tab"，但为了露出战术大赛 tab
           左栏已经往下滑过 —— `信用点商店` 早滑出视野、检不出  这段是死码，
           bot 直接停在战术大赛货架上收工。**切不回去就关页面**，别假装做了。
        """
        t = obs.find(V.SHOP_TAB_CREDIT, 0.40)
        if t is not None and self.pending("backcredit"):
            return tap_box(t, "买完切回信用点 tab（安全 tab）", once="backcredit")
        return self.exit_step(obs, prefer_close=False)

    def on_combo_pack(self, obs, st):
        # 免费包已拆到 FreePackFlow。商店链落到组合包页只许退出。
        return self.exit_step(obs) or wait("等退出控件")

    def on_confirm_dialog(self, obs, st):
        if obs.has(V.CONFIRM_GREY, 0.45):
            c = obs.find(V.CANCEL, 0.45)
            return tap_box(c, "灰确认  取消") if c is not None else wait("等取消")
        pyx = obs.find(V.PYROXENE, 0.40, region=(0.12, 0.12, 1.0, 1.0))
        if pyx is not None and not obs.has(V.FREE, 0.30):
            c = obs.find(V.CANCEL, 0.45)
            if c is not None:
                return tap_box(c, f"框内出现青辉石价签（conf {pyx.conf:.2f}） 取消")
            return wait("框内有青辉石价签但没找到取消键, 绝不点确认")
        spend = ("战术大赛货币" if obs.has(V.ARENA_SHOP_CURRENCY, 0.35)
                 else "信用点")
        cf = obs.find(V.CONFIRM, 0.45)
        # 框内已排除青辉石价签。信用点/大赛币确认不是买AP/买票/CAD, 不挡。
        return tap_box(cf, f"确认购买（花 {spend}）",
                       spend=spend) if cf is not None else wait("等確認键")


#
class MailFlow(ExitMixin, Flow):
    name = "mail"
    module = "mail"

    def on_lobby(self, obs, st):
        a = nav.lobby_enter(self, obs, V.NAV_MAIL, "邮件箱",
                            expect=(V.CLAIM_ONCE_YELLOW, V.CLAIM_ONCE_GREY,
                                    V.CLAIM_BLUE),
                            conf=0.35)
        return a if a is not None else wait("等邮件箱入口 cls")

    def on_mail(self, obs, st):
        y = obs.find(V.CLAIM_ONCE_YELLOW, 0.40)
        if y is not None:
            return tap_box(y, "一次領取", counter="claims")
        if obs.has(V.CLAIM_ONCE_GREY, 0.40):
            return self.finish(Outcome.CLEAN,
                               f"邮件领取 {self.state.get('claims',0)} 次（领取键已变灰）")
        # 领完的奖励层会把黄/灰键全盖住 -- 08-28 实锤: 旧代码 40 帧兜底
        #    CLEAN 就走人, 层后面还有没有邮件根本没看(用户手动又收了一轮)。
        #    先把层点掉; 只有**亲眼看到灰键**才算领完, 看不清就诚实 LEFTOVER。
        cont = obs.find(V.STORY_TAP_CONTINUE, 0.45)
        if cont is not None:
            return tap_box(cont, "关掉领取奖励层（后面才看得到黄/灰键）")
        if st.frames_in_page < 120:
            return wait("等邮件页控件（黄键/灰键都没露出来）")
        return self.finish(
            Outcome.LEFTOVER,
            f"邮件领取 {self.state.get('claims',0)} 次, 但 120 帧没等到灰键"
            f" — 不定论领完（页面可能被层盖着）")

    # on_reward 的覆写已删（08-11）：原来是 `find([GOT_REWARD, CONFIRM], 0.40)`，
    #    而 `find(列表)` 是**全屏 conf argmax**不是优先级  永远选中
    #    `获得奖励` 横幅（0.98）而不是底部 `点击继续字样`（0.93）。
    #    基类 `Flow.on_reward` 的 `or` 链才是真优先级，直接继承。


#
class DailyMissionFlow(ExitMixin, Flow):
    """每日任务。

    `一键领取灰色`(415) **train=0** —— 不能拿它判"已领完"（那是死判据）。
       用 `全部领取_灰色`(413) + "亮态领取键消失" 双条件判。
    """
    name = "daily_mission"
    module = "daily_mission"

    def on_lobby(self, obs, st):
        # 08-16 remain: 大厅 8/8 每日领奖红点还在, 本轮没再进任务页.
        #    once+expect 挡 38 帧连点; 入口还在就进, 不在大厅空 CLEAN.
        a = nav.lobby_enter(self, obs, V.NAV_DAILY_REWARD, "每日领奖",
                            expect=(V.CLAIM_ALL_YELLOW, V.CLAIM_ALL_GREY),
                            conf=0.35)
        return a if a is not None else wait("等每日领奖入口 cls")

    def on_daily_mission(self, obs, st):
        # 用户口述权威（08-09）：就点两下 —— 底部「全部领取」
        #    它**旁边**那个「领取」（底栏 cx≈0.76 处）。中间列表的单项
        #    不用一个个点（全部领取全覆盖，argmax 逐个点是磨洋工）。
        y = obs.find(V.CLAIM_ALL_YELLOW, 0.40)
        if y is not None:
            return tap_box(y, "全部领取", counter="claims")
        y2 = obs.find(V.CLAIM_YELLOW, 0.40, region=(0.0, 0.85, 1.0, 1.0))
        if y2 is not None:
            return tap_box(y2, "全部领取旁边的领取（底栏）", counter="claims")
        # 「领完了」判据加一个"或"成员：`完成_灰色`(490, train 81/val 3)。
        #    08-11 live 帧证（v2step_20260811）：底栏  那个键随状态换 cls ——
        #      · 还能领      `领取_黄`   0.975 @(0.761,0.928)
        #      · 领过但未完  `领取_灰`   0.965 @(0.760,0.928)
        #      · 全做完了    `完成_灰色` **0.985** @(0.760,0.926)
        #    该 cls 之前**全仓零引用**。这里是**加法**：`全部领取_灰色`
        #    （同帧 0.963）那条判据原样保留，只是多一条独立证据。
        #    带 region 限死底栏（cy≥0.85）：中间列表行万一也吐 `完成_灰色`，
        #      不能让它把"还有活"的帧误判成收工（多领少领事小，谎报 CLEAN 事大）。
        if obs.has(V.CLAIM_ALL_GREY, 0.40) \
                or obs.has(V.NODE_DONE_GREY, 0.40, region=(0.0, 0.85, 1.0, 1.0)):
            return self.finish(Outcome.CLEAN,
                               f"每日任务领取 {self.state.get('claims',0)} 次（已全灰）")
        if st.frames_in_page < 40:
            return wait("等每日任务页")
        return self.finish(Outcome.CLEAN,
                           f"每日任务领取 {self.state.get('claims',0)} 次")

    # on_reward 的覆写已删（08-11 live 真因）：`find([GOT_REWARD, CONFIRM], 0.40)`
    #    是 conf argmax，把 `获得奖励` 横幅（0.98，**不是按钮**）当落点
    #    「獲得獎勵！」全屏 overlay 上连点 11 次没反应（02:00:4102:02:59），
    #    而真正该点的 `点击继续字样`（TOUCH TO CONTINUE，0.93 @cy0.877）
    #    压根不在候选里。基类的 `or` 链已经把它排在第二档，继承即可。
