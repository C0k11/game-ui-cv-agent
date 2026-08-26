# -*- coding: utf-8 -*-
"""活动商店 —— **先推算，再决定加成队打哪种币**（用户规则）。

产出放进 `ctx.bag["event_shop_plan"]`，格式:
    {币种cls: {"balance": int|None, "buyable": int, "soldout": int}}
`event` flow 的加成阶段读它来定编队方向。

两条金钱纪律:
  · **最后一个 tab 是盒抽币，绝不自动买**（`skip_last_tab`，默认 True）
  · 活动币不是青辉石，但仍走 money 步（人审）—— 买错了这一期就补不回来了

滑动纪律: 有亮购买(103)先把当前可见该买的买完, 没 103 才滑.
   往下找该买的; 上滑只因本屏全是购买灰色(489)或没有该买的亮购买.
   禁止「画面不变/刚滑到底就立刻向上」. YOLO 103/489 主导, 灰不买.
"""
from __future__ import annotations

import time
from typing import Optional

from routing_v2.act.action import tap_box, wait
from routing_v2.flow.base import ExitMixin, Flow, Outcome
from routing_v2.flow.event import EventEntryMixin
from routing_v2.flow.shelf_walk import make_swipe
from routing_v2.percept import read as R
from routing_v2.percept.observe import Observation
from routing_v2.state import vocab as V

SHELF = (0.18, 0.14, 1.0, 0.92)
LEFT_COL = (0.0, 0.10, 0.24, 0.98)

# **推算映射（用户 2026-08-08 口述，权威）—— 这是个相对规则，不是写死关号**
#
#   把"产出通道"从最珍贵到次之排成一列，**依次对上关卡列表从下往上的关**：
#
#       通道列表 = [活动点数（如果这个活动有）] + [商店币种，从**最下面**往上]
#       通道[k]    关卡列表**倒数第 k+1 关**
#
#   · 有活动点数时：点数最后一关(Q12)，商店最下面的币Q11，上面一个Q10
#   · **没有活动点数时：商店最下面的币就是 Q12，然后依次往上推**
#
# 两边都用"从底部数第几个"，所以换活动、换关卡数、换商店档数全都成立，
#   不需要认关卡编号，也不需要给币种训 cls。
# tab 序位只能按 **cy** 定，不能按 conf 排序（老代码在商店 tab 上串过位）。
def farm_targets(tabs_bottom_up, has_event_points: bool):
    """(通道  倒数第几关) 的推导。返回 [{"from_bottom": k, "why": str}]。"""
    out = []
    k = 0
    if has_event_points:
        out.append({"from_bottom": 0, "why": "活动点数（獎勵資訊 进度条）"})
        k = 1
    for t in tabs_bottom_up:                 # 最下面的币排在最前
        out.append({"from_bottom": k,
                    "why": f"商店自下第 {t['from_bottom']+1} 档货币"
                           f"（还有 {t['buyable']} 件没买）"})
        k += 1
    return out


class EventShopFlow(EventEntryMixin, ExitMixin, Flow):
    """进场链（大厅/任务大厅/认错活动）继承 EventEntryMixin —— 2026-08-13
    实锤: 单独跑（不经 event 交棒）时在 lobby 401 帧没有任何可执行动作。"""
    name = "event_shop"
    module = "event"
    entry_page = "event_page"
    # 购买框由本 flow 自己处理（interrupt 三重合取之一; 全仓只有这里敢开）。
    #   反向保险在 on_confirm_dialog: 框内有青辉石或体力图标 -> 一律取消。
    handles_purchase_dialog = True

    def setup(self) -> None:
        self.state.update(tab_i=0, scrolls=0, sig="", plan={}, bought=0)
        self.shop_cfg = (self.cfg.get("shop") or {})

    #  进场
    def on_event_page(self, obs, st):
        s = obs.find(V.EVENT_SHOP, 0.40)
        if s is not None and self.pending("enter_shop"):
            # once 而不是靠重发（08-09 实锤）：点完商店入口后页面在加载，
            #    状态"没变"  70 帧后重发，而那时页面已经换了  **点到别的
            #    东西**（实测坐标 0.2650.1360.265 乱跳，把商店点关了）。
            #    导航类点击是**一次性动作**，重发的代价远大于收益。
            return tap_box(s, "进活动商店", once="enter_shop")
        if self.stalled(st, 120):
            return self._wrap("活动页没检出 活动商店 入口")
        return wait("等 活动商店 入口")

    on_event_quest_list = on_event_page

    #  货架
    def _tabs(self, obs: Observation):
        """左栏币种 tab，**按 cy 升序**（从上到下）。

        身份只能按 **cy 序位** 定，不能按 conf 排序 —— 老代码在商店 tab 上
           就是这么串位的。推算映射也是按序位来的（见 FARM_MAP）。
        """
        return obs.rows([V.CURRENCY, V.CURRENCY_SEL], 0.30, region=LEFT_COL)

    def on_event_shop(self, obs, st):
        tabs = self._tabs(obs)               # 按 cy 升序 = 左栏从上到下
        if not tabs:
            # `货币` cls 会整页检不出（08-09 CODE:BOX 实锤：0.20 模型下限
            #    以上一个都没有，而屏上左栏明明有一档「貨幣」）。这是**感知缺口**，
            #    不是"不在商店页"——页面身份已由货架 `购买` 独立确认过了。
            #     回退：按**单档**推算（左栏只有一档是最常见形态），
            #      日志明说是回退，免得以后把漏档当成真相。
            if self.hold("no_tabs", 45):
                if not self.state.get("single_fallback"):
                    self.state["single_fallback"] = True
                    self.state["ntabs"] = 1
                    self.log("左栏币种 tab 一个都检不出（cls `货币` 感知缺口）"
                             "  **回退按单档推算**，直接扫当前货架")
                return self._scan_shelf(obs, st, cur_name="tab1/1(自下第1)",
                                        from_bottom=0, i=0, n=1)
            return wait("等左栏币种 tab")

        # 用**见过的最大 tab 数**，不用当帧的。切 tab 的那一瞬未选中态会漏检，
        #   拿当帧数量算上限的话，2 个 tab 会被算成 1 个  limit=0  一个都不扫
        #   （08-08 实测「扫完 0/1 个 tab」就是这么来的）。
        self.state["ntabs"] = n = max(self.state.get("ntabs", 0), len(tabs))
        i = self.state["tab_i"]
        # `skip_last_tab` 的语义是「跳过盒抽币那一档」。而用户口述的映射里
        #   **最下面那个币是 Quest11 的产出，是要买的** —— 所以这里只在
        #   tab 数 ≥3 时才跳最后一个；两个 tab 的活动全都要扫。
        skip_last = bool(self.shop_cfg.get("skip_last_tab", True)) and n >= 3
        limit = max(0, n - (1 if skip_last else 0))
        if i >= limit:
            return self._wrap(
                f"扫完 {limit}/{n} 个 tab"
                + ("（最后一个是盒抽币，按配置跳过）" if skip_last else ""))
        # tab 身份按 **cy 序位** 定，但**别索引当帧列表**：
        #    切 tab 瞬间未选中态漏检  tabs[i] IndexError 崩 runner（08-09 崩过两轮）
        #    选中后**未选中态可以长期检不出**（tab1 未选中在这个活动的皮上
        #      持续漏检） "等检出"会永久卡死（08-09 tab2 上停了 10 分钟）。
        #    正解 = tab 的**纵坐标不变**：把见过的 cy 列表记满就一直用它，
        #    选中态/目标全按 cy 对号，当帧检出几个都无所谓。
        cys = self.state.setdefault("tab_cys", [])
        if len(tabs) > len(cys):
            cys = [round(b.cy, 3) for b in tabs]
            self.state["tab_cys"] = cys
        if i >= len(cys):
            return wait(f"第 {i+1} 个 tab 的位置还没见过（已知 {len(cys)} 个）")
        target_cy = cys[i]
        sel = obs.find(V.CURRENCY_SEL, 0.30, region=LEFT_COL)
        on_target = sel is not None and abs(sel.cy - target_cy) < 0.03
        if not on_target:
            near = min(tabs, key=lambda b: abs(b.cy - target_cy))
            if abs(near.cy - target_cy) < 0.03:
                return tap_box(near, f"切到第 {i+1}/{n} 个币种 tab"
                                     f"（cy={target_cy:.3f}）")
            return wait(f"目标 tab（cy={target_cy:.3f}）这一帧没检出框，等一帧")

        # 模型没有"具体是哪种币"的 cls  身份只能靠**从下往上数第几个**，
        #   这正好就是用户给的映射口径（最下面Q11，上面一个Q10）。
        from_bottom = n - 1 - i              # 0 = 最下面那个
        cur_name = f"tab{i+1}/{n}(自下第{from_bottom+1})"
        return self._scan_shelf(obs, st, cur_name, from_bottom, i, n, cur=sel)

    def _shelf_sig(self, obs: Observation) -> str:
        """带 cls: 同坐标 103<->489 也算变了. 只写 cx,cy 会在网格上假到底."""
        pts = obs.all([V.SHOP_BUY, V.SHOP_BUY_GREY], 0.30, region=SHELF)
        return "|".join(sorted(f"{b.cx:.2f},{b.cy:.2f},{b.cls}" for b in pts))

    def _shelf_reset(self) -> None:
        for k in ("shelf_await_settle", "shelf_pre_swipe", "shelf_last_up",
                  "shelf_settle_watch", "shelf_went_down", "shelf_moved",
                  "hold:shelf_settle", "hold:shelf_settle:t"):
            self.state.pop(k, None)

    def _shelf_swipe(self, obs, up: bool, why: str, post=None):
        return make_swipe(obs, up, why, post=post)

    def _clean_buys(self, buyable, soldout):
        """同槽 103+489 = 半渲染当灰, 不点."""
        out = []
        for b in buyable:
            if any(abs(b.cx - g.cx) < 0.04 and abs(b.cy - g.cy) < 0.04
                   for g in soldout):
                continue
            out.append(b)
        return out

    def _scan_shelf(self, obs, st, cur_name, from_bottom, i, n, cur=None):
        """有该买的 103 先买完当前可见, 没 103 才滑.

        往下找该买的. 上滑只因本屏没有亮购买(全是 489 或没有该买的),
        不是「画面不变=探底完成」. 上一刀连续 3 tick 同画面就回滑是坑.
        """
        bal = R.read_event_coin(obs)

        buyable = obs.all(V.SHOP_BUY, 0.35, region=SHELF)
        soldout = obs.all(V.SHOP_BUY_GREY, 0.35, region=SHELF)
        rec = self.state["plan"].setdefault(
            cur_name, {"balance": bal, "buyable": 0, "soldout": 0,
                       "from_bottom": from_bottom})
        rec["balance"] = bal if bal is not None else rec["balance"]
        rec["buyable"] = max(rec["buyable"], len(buyable))
        rec["soldout"] = max(rec["soldout"], len(soldout))

        denied = any(k.startswith("moneyno:") for k in self.state)
        if denied and self.pending("deny_note"):
            self.state["once:deny_note"] = True
            self.log(f"{cur_name}: 购买未授权(--money-ok 没带) — "
                     f"本轮只扫描出推算，不再试买")
            self.note_lines.append("购买未授权 — 本轮只扫描出推算")

        # 08-16 remain: 选择购买/确认在场 = 已经勾上, 先成交, 不滑走.
        sel = obs.find(V.SHOP_BUY_SELECTED, 0.40)
        if sel is not None:
            return tap_box(sel, f"{cur_name}: 选择购买（已勾上, 不滑走）",
                           money=True, spend="活动币", expect=(V.CONFIRM,))
        if obs.find(V.CONFIRM, 0.45) is not None:
            return self.on_confirm_dialog(obs, st)

        if not buyable and not soldout:
            return wait(f"{cur_name}: 货架不可见（面板盖住/加载中）— 不下结论")

        if self.state.get("shelf_await_settle"):
            sig_now = self._shelf_sig(obs)
            prev = self.state.get("shelf_settle_watch")
            if prev != sig_now:
                self.state["shelf_settle_watch"] = sig_now
                return wait(f"{cur_name}: 滑动后等货架停稳")
            if not self.hold("shelf_settle", 2):
                return wait(f"{cur_name}: 滑动后等货架停稳")
            self.state["shelf_await_settle"] = False
            pre = self.state.get("shelf_pre_swipe", "")
            self.state["shelf_moved"] = bool(sig_now) and sig_now != pre

        clean = self._clean_buys(buyable, soldout)
        want = self.shop_cfg.get("currencies") or []
        auto = bool(self.shop_cfg.get("auto_buy", True)) and (
            not want or cur_name in want)

        # 有该买的 103: 买, 绝不滑(尤其绝不向上).
        if clean and (not denied) and auto:
            if bal is None:
                if self.pending("no_bal"):
                    self.state["once:no_bal"] = True
                    self.log(f"{cur_name}: 余额读不出 — 不自动买(fail-closed)")
                return wait(f"{cur_name}: 余额读不出, 不动")
            if obs.frame is None:
                return wait(f"{cur_name}: 有亮购买未读价, 不滑走")
            sig = f"{cur_name}|{len(buyable)}|{len(soldout)}"
            act = self._buy_visible(obs, cur_name, clean, bal, sig)
            if act is not None:
                return act
            # 全是家具/买不起/拉黑: 本屏没有该买的, 可以滑

        n_down = int(self.state.get("scrolls", 0))
        n_up = int(self.state.get("upscrolls", 0))
        moved = self.state.get("shelf_moved")
        went_down = bool(self.state.get("shelf_went_down"))
        last_up = bool(self.state.get("shelf_last_up"))

        # 上滑: 买路只允许「本屏没有该买的 103」. 拒买扫描才在下行探完后上翻.
        no_wanted = (not clean) or denied or (not auto)
        want_up = False
        if no_wanted:
            if last_up:
                want_up = True
            elif went_down and moved is False:
                want_up = True
            elif n_down >= 12:
                want_up = True

        def _arm(up, why, kk):
            s = self._shelf_sig(obs)

            def _post(ss=s, k=kk, u=up):
                self.state.update(
                    shelf_pre_swipe=ss, shelf_await_settle=True,
                    shelf_last_up=u, shelf_settle_watch="",
                    shelf_went_down=True if not u else self.state.get(
                        "shelf_went_down"),
                    scrolls=k if not u else self.state.get("scrolls", 0),
                    upscrolls=k if u else self.state.get("upscrolls", 0))

            return self._shelf_swipe(obs, up, why, post=_post)

        # 拒买/不自动买 = 纯扫描口径: 只下行一趟数存货, 触底即收 tab,
        #    不上翻(切 tab 自己回顶)。08-26 用户实看: 扫描模式跑着买路的
        #    "找该买的"寻路, 每 tab 下探到底再上翻回顶, 纯瞎滑。
        if denied or (not auto):
            if (went_down and moved is False) or n_down >= 12:
                return self._tab_done(cur_name, rec, bal)
            sw = _arm(False, f"{cur_name}: 扫描存货 下行(第 {n_down + 1} 次)",
                      n_down + 1)
            if sw is not None:
                return sw
            return wait(f"{cur_name}: 货架上没检出行锚点 — 不瞎滑")

        if want_up:
            if n_up >= 12 or (last_up and moved is False):
                return self._tab_done(cur_name, rec, bal)
            sw = _arm(True, f"{cur_name}: 本屏无该买的亮购买 — 上滑(第 {n_up + 1} 次)",
                      n_up + 1)
            if sw is not None:
                return sw
            return wait(f"{cur_name}: 货架上没检出行锚点 — 不瞎滑")

        sw = _arm(False, f"{cur_name}: 往下找该买的(第 {n_down + 1} 次)",
                  n_down + 1)
        if sw is not None:
            return sw
        return wait(f"{cur_name}: 货架上没检出行锚点 — 不瞎滑")

    def _tab_done(self, cur_name, rec, bal):
        self.log(f"{cur_name}: 余额 {bal}，可买 {rec['buyable']}，"
                 f"售罄 {rec['soldout']} — 这个 tab 扫完")
        self.state["tab_i"] += 1
        self.state.update(scrolls=0, upscrolls=0, sig="")
        self._shelf_reset()
        self.state.pop("dead_spots", None)
        self.state.pop("buy_tries", None)
        self.once_reset()
        if self.state.get("single_fallback"):
            return self._wrap("单档回退：扫完唯一货架")
        return wait(f"{cur_name} 扫完  下一个 tab")

    on_shop = on_event_shop      # 万一被判成通用商店也照样处理

    def _price_of(self, frame, b, pitch):
        """单价 = 按钮上方深蓝带里的**右对齐**数字。

        本期版面 2026-08-13 帧标定: det 框只框住購買二字, 数字一半在框右外;
        07-28 那套「按钮上沿一条」窗口在这个皮上读到的是「可購買N次」黑条,
        次数被当成单价 -- **价签窗口是随活动皮肤变的, 每期都要用实帧校准**。
        窗口参数拿两张实帧对 [5,15,50,200]x2 真值网格扫出来(14/16, 失败向
        =None 跳过)。单字符价("5")必须走 small_number 平铺 -- 识别引擎的
        检测阶段拒绝孤立单字符(digit_ocr_small_number), 自带 5 票一致性。"""
        bw = b.x2 - b.x1
        bh = b.y2 - b.y1
        r_edge = b.x2 + 1.3 * bw
        if pitch is not None:                 # 别伸进右边邻卡(防幻影后缀)
            r_edge = min(r_edge, b.cx + 0.45 * pitch)
        return R.small_number(
            frame, (b.x1 - 0.3 * bw, b.y1 - 1.2 * bh, r_edge, b.y1), inset=0.0)


    def _buy_visible(self, obs, cur_name, buyable, bal, sig):
        """当前取景内: 读价 -> 家具行剔除 -> 单价降序买一件. None = 没有该买的."""
        tries = self.state.setdefault("buy_tries", {})
        if self.state.get("bought", 0) >= 60:
            return None
        bh_all = sorted(max(b.y2 - b.y1, 0.01) for b in buyable)
        bh = bh_all[len(bh_all) // 2]
        xs = sorted(b.cx for b in buyable)
        gaps = [b - a for a, b in zip(xs, xs[1:]) if b - a > 0.03]
        pitch = min(gaps) if gaps else None
        rows = []
        for b in sorted(buyable, key=lambda x: x.cy):
            if rows and abs(b.cy - rows[-1][-1].cy) < bh * 1.2:
                rows[-1].append(b)
            else:
                rows.append([b])
        cache = self.state.setdefault("price_cache", {})
        priced_all = cache.get(sig)
        if priced_all is None:
            priced_all = {}
            for row in rows:
                for b in row:
                    priced_all[f"{b.cx:.2f},{b.cy:.2f}"] = self._price_of(
                        obs.frame, b, pitch)
            cache[sig] = priced_all
            while len(cache) > 6:
                cache.pop(next(iter(cache)))
        cands = []
        for row in rows:
            priced = [(priced_all.get(f"{b.cx:.2f},{b.cy:.2f}"), b) for b in row]
            vals = [p for p, _ in priced if p is not None]
            if vals and (len(vals) != len(set(vals)) or max(vals) > 1000):
                if self.pending(f"furn{round(row[0].cy, 2)}"):
                    self.state[f"once:furn{round(row[0].cy, 2)}"] = True
                    self.log(f"{cur_name}: 行价签 {sorted(vals)} = 家具行, 整行跳过")
                continue
            for p, b in priced:
                if p is None or not (0 < p <= 1000):
                    continue
                k = f"{b.cx:.2f},{b.cy:.2f}"
                if tries.get(k, 0) >= 3:
                    continue
                cands.append((p, b, k))
        if not cands:
            return None
        afford = [c for c in cands if c[0] <= bal]
        if not afford:
            return None
        p, b, k = max(afford, key=lambda c: c[0])

        def _tried(kk=k):
            t = self.state.setdefault("buy_tries", {})
            t[kk] = t.get(kk, 0) + 1
        return tap_box(b, f"{cur_name}: 买单价 {p}（非家具最高价优先，"
                          f"余额 {bal}）",
                       money=True, spend=f"{cur_name}单价{p}",
                       post=_tried, expect=(V.CONFIRM,))

    def on_confirm_dialog(self, obs, st):
        # **反向保险**: 活动商店的商品框价签是活动币。框体内出现 青辉石
        #    （真钱风险）或 体力图标（購買AP 框 —— 青辉石图标在那个框上
        #    检不出, 08-09 实锤, 但体力图标在）都不是我点出来的商品框,
        #    一律取消, 绝不确认。
        # 只看对话框中部. 顶栏体力在 y<0.10, 扫到顶栏会把活动币确认误取消.
        bad = obs.find([V.PYROXENE, V.AP], 0.40, region=(0.20, 0.25, 0.80, 0.75))
        if bad is not None:
            c = obs.find(V.CANCEL, 0.45)
            if c is not None:
                return tap_box(c, f"框内检出 `{bad.cls}` — 不是活动币商品框，取消")
            return wait(f"框内检出 `{bad.cls}` 但没找到取消键 — 绝不点確認")
        if obs.has(V.CONFIRM_GREY, 0.45):
            c = obs.find(V.CANCEL, 0.45)
            return tap_box(c, "灰确认  取消") if c is not None else wait("等取消")
        # 购买数量框: 先拉 MAX 再确认（老码 buy_one 同序）。MAX 变灰 = 拉满。
        #   MAX 也标 money —— 它决定成交数量, 就该走金钱授权;
        #   不标的话 gate.money 的「购买语境非白名单 tap」规则会 halt（实测）。
        mx = obs.find(V.QTY_MAX, 0.40, region=(0.0, 0.12, 1.0, 1.0))
        if mx is not None:
            return tap_box(mx, "数量拉 MAX", money=True, spend="活动币",
                           expect_gone=(V.QTY_MAX,))
        cf = obs.find(V.CONFIRM, 0.45)
        return tap_box(cf, "确认购买", money=True, spend="活动币",
                       counter="bought") if cf is not None else wait("等確認键")

    # on_reward 一律继承基类：`find([A, B])` 是全屏 conf argmax，会永远
    #   选中 `获得奖励` 那条**横幅**而不是能点的 `点击继续字样`（实锤见
    #   base.py 的 on_reward），覆写没有任何收益。

    #  收尾：推算「该刷哪一关」并交给 event flow
    def _wrap(self, why):
        """**推算**（用户 2026-08-08 给的映射，见 FARM_MAP）:

            商店最下面那个币还缺  去刷**倒数第 2 关**(Q11)
            上面一个币还缺        去刷**倒数第 3 关**(Q10)
            活动点数没满          去刷**最后一关**  (Q12)

        产出 `ctx.bag["event_farm_plan"]` = 按优先级排好的
        `[{"from_bottom": k, "why": "..."}]`，`from_bottom` 是**关卡列表从下往上
        数第几个**（0=最后一关）。event flow 直接拿它选关，不需要认关卡编号。
        """
        plan = self.state["plan"]
        self.ctx.bag["event_shop_plan"] = plan
        lines = [f"{k}: 余额={v['balance']} 可买={v['buyable']} 售罄={v['soldout']}"
                 for k, v in plan.items()]
        # `_wrap` 会被多次走到（收工被推进闸按回 -> 下一 tick 又收工）——
        #    note 只记第一次, 别把报告刷成 12 遍复读（2026-08-13 实锤）。
        if self.state.get("wrap_noted"):
            return self.finish(
                Outcome.CLEAN if plan else Outcome.UNKNOWN,
                f"{why}；购买 {self.state['bought']} 次；" + "；".join(lines))
        self.state["wrap_noted"] = True
        for l in lines:
            self.note_lines.append(l)

        # 商店币种按**从下往上**排（最下面那档排第一）
        tabs_bottom_up = sorted(
            [dict(v, key=k) for k, v in plan.items()],
            key=lambda v: v.get("from_bottom", 0))
        # 有没有"活动点数"这条通道，由 event flow 在关卡列表上看 `奖励资讯`
        # 得出（那个页面才看得到进度条）。默认 True —— 多数活动都有。
        has_pts = bool(self.ctx.bag.get("event_has_points", True))
        targets = farm_targets(tabs_bottom_up, has_pts)
        # 只留"还缺"的通道；全不缺就保留最后一关当兜底
        short = [t for t, v in zip(targets[1:] if has_pts else targets,
                                   tabs_bottom_up) if v["buyable"] > 0]
        ordered = ([targets[0]] if has_pts else []) + short or targets[:1]

        self.ctx.bag["event_farm_plan"] = ordered
        self.ctx.bag["event_shop_tabs"] = tabs_bottom_up
        # 计划落盘（08-09）：ctx.bag 是进程内存 —— runner 崩溃/step 模式跨
        #    进程都会把推算丢掉，event 只能"兜底打最后一关"。文件是权威副本。
        #    08-15 起写进账号桶（读侧 event._plan_from_file 同桶）。
        try:
            import json as _json
            from routing_v2.config import data_dir
            _out = data_dir(self.ctx.cfg)
            (_out / "event_farm_plan.json").write_text(
                _json.dumps({"ts": time.time(), "has_points": has_pts,
                             "plan": ordered}, ensure_ascii=False, indent=1),
                encoding="utf-8")
        except Exception as e:
            self.log(f"推算计划落盘失败: {e}")
        self.log(f"推算（通道  倒数第几关，{'有' if has_pts else '无'}活动点数）:")
        for t in ordered:
            self.log(f"     倒数第 {t['from_bottom']+1} 关   {t['why']}")
            self.note_lines.append(
                f"推算目标: 倒数第 {t['from_bottom']+1} 关  {t['why']}")
        return self.finish(
            Outcome.CLEAN if plan else Outcome.UNKNOWN,
            f"{why}；购买 {self.state['bought']} 次；" + "；".join(lines))
