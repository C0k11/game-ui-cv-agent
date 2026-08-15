# -*- coding: utf-8 -*-
"""活动商店 —— **先推算，再决定加成队打哪种币**（用户规则）。

产出放进 `ctx.bag["event_shop_plan"]`，格式:
    {币种cls: {"balance": int|None, "buyable": int, "soldout": int}}
`event` flow 的加成阶段读它来定编队方向。

两条金钱纪律:
  · **最后一个 tab 是盒抽币，绝不自动买**（`skip_last_tab`，默认 True）
  · 活动币不是青辉石，但仍走 money 步（人审）—— 买错了这一期就补不回来了

滑动纪律（用户 2026-08-07 纠正）: 老代码 `for si in range(5)` 固定滑 5 次。
   这里改成**指纹比对**：滑之前记货架签名，滑之后比，没变化 = 到底了。
   签名用"屏上全部 购买/购买灰色 框的 (cx,cy) 四舍五入"—— 不用像素比对，
   因为 UI 动画会让像素一直在变，而框位是稳的。
"""
from __future__ import annotations

import time
from typing import Optional

from routing_v2.act.action import swipe, tap_box, wait
from routing_v2.flow.base import ExitMixin, Flow, Outcome
from routing_v2.flow.event import EventEntryMixin
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

    # ── 进场 ────────────────────────────────────────────────────────────
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

    # ── 货架 ────────────────────────────────────────────────────────────
    def _shelf_sig(self, obs: Observation) -> str:
        """货架指纹 = 全部商品按钮框位（不用像素，动画不影响）。"""
        pts = obs.all([V.SHOP_BUY, V.SHOP_BUY_GREY], 0.30, region=SHELF)
        return "|".join(sorted(f"{b.cx:.2f},{b.cy:.2f}" for b in pts))

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

    def _shelf_swipe(self, obs, up: bool, why: str, post=None):
        """货架滑动 —— 几何全从检出推，**双向共用同一把尺**。

        锚只用货架里的 购买/购买灰色（老写法把 [SHOP_BUY, CURRENCY] 混着喂
           nav.list_swipe：左栏 tab 的行距和货架行距差一倍，哪帧多检出谁、
           滑幅就跟谁走 —— 用户 08-15 看到的「滑动幅度一下大一下小」就是它）。
        行距进 rowh_hist 取中位（单帧检出抖动不再改变滑幅），距离钳在
           [0.24, 0.60] 屏内。"""
        bs = obs.all([V.SHOP_BUY, V.SHOP_BUY_GREY], 0.30, region=SHELF)
        if not bs:
            return None
        xs = sorted(b.cx for b in bs)
        cx = xs[len(xs) // 2]
        hs = sorted(max(b.y2 - b.y1, 0.008) for b in bs)
        rowh = hs[len(hs) // 2]
        row_cys = []
        for cy in sorted(b.cy for b in bs):
            if not row_cys or cy - row_cys[-1] > rowh * 0.5:
                row_cys.append(cy)
        gaps = [b - a for a, b in zip(row_cys, row_cys[1:])]
        if gaps:
            g = sorted(gaps)[len(gaps) // 2]
            if rowh < g < 0.35:
                rowh = g
        hist = self.state.setdefault("rowh_hist", [])
        hist.append(round(rowh, 4))
        del hist[:-9]
        rowh = sorted(hist)[len(hist) // 2]
        dist = min(0.60, max(0.24, rowh * 3.0))
        if up:
            y0 = max(0.16, min(b.cy for b in bs) - rowh * 0.2)
            y1 = min(0.90, y0 + dist)
        else:
            y0 = min(0.90, max(b.cy for b in bs) + rowh * 0.6)
            y1 = max(0.12, y0 - dist)
        return swipe(cx, y0, cx, y1, why, post=post)

    def _scan_shelf(self, obs, st, cur_name, from_bottom, i, n, cur=None):
        """扫当前货架: **先滑到底，再从最底部边买边上滑**。

        用户 2026-08-15:「从最底部开始买起然后开始对比大小」—— 替换掉旧的
           高价档/低价档两遍回顶策略（07-28 移植版）。旧结构当天实锤两条病:
           ①下滑发出去 8 tick 内动画还没走、指纹和滑前一样 -> 被当「到底了」
             一次下滑就回顶，第二屏永远没人看过（"一会儿下一会儿上"）;
           ②回顶+两遍 = 三趟往返，方向来回换。
           新结构: 下行只扫不买（顺路记 buyable/soldout 全量）; 到真底后
           每个取景内仍按**单价降序**买，买完一屏上滑一格接着比 —— 家具行、
           余额 fail-closed、黑名单、金钱闸全部原样保留。"""
        bal = R.read_event_coin(obs)

        buyable = obs.all(V.SHOP_BUY, 0.35, region=SHELF)
        soldout = obs.all(V.SHOP_BUY_GREY, 0.35, region=SHELF)
        rec = self.state["plan"].setdefault(
            cur_name, {"balance": bal, "buyable": 0, "soldout": 0,
                       "from_bottom": from_bottom})
        rec["balance"] = bal if bal is not None else rec["balance"]
        rec["buyable"] = max(rec["buyable"], len(buyable))
        rec["soldout"] = max(rec["soldout"], len(soldout))

        # **滑动后先等货架停稳，再下任何结论**（08-15 日常 live 实锤:
        #    判「到底」的指纹比对发生在滑动动画启动之前，指纹自然没变 ->
        #    假到底）。10 tick 约 0.7s，盖住滚动动画。
        if self.ticks - self.state.get("swipe_tick", -99) < 10:
            return wait(f"{cur_name}: 滑动后等货架停稳")

        # **货架不可见时不许下「到底/扫完」的结论**（2026-08-13 round6 实锤:
        #    买完一件的确认框把货架整个盖住 -> 购买键全检不出 -> 指纹为空 ->
        #    被当「到底了」提前收 tab, tab1 剩 8 件 tab2 剩 6 件没吃完）。
        if not buyable and not soldout:
            return wait(f"{cur_name}: 货架不可见（面板盖住/加载中）— 不下结论")

        # 金钱步被人审拒绝过 = 本轮没带 --money-ok，买不成。**不再试买，
        #    但扫描推算照做** —— 08-15 实锤: 原来第二发被拒就把整条 flow 收成
        #    BLOCKED，_wrap 没跑 -> event 拿不到计划 -> 兜底关绕过顶纪录台账
        #    又打了一场加成。推算是免费的，授权只该拦「花钱」这一步。
        denied = any(k.startswith("moneyno:") for k in self.state)
        if denied and self.pending("deny_note"):
            self.state["once:deny_note"] = True
            self.log(f"{cur_name}: 购买未授权(--money-ok 没带) — "
                     f"本轮只扫描出推算，不再试买")
            self.note_lines.append("购买未授权 — 本轮只扫描出推算")

        sig_now = self._shelf_sig(obs)
        if self.state.get("shelf_phase", "down") == "down":
            # 下行: 指纹还在变 = 还没到底，接着滑（这一段不买）
            if sig_now and sig_now != self.state["sig"] \
                    and self.state["scrolls"] < 12:
                # mutate-before-ack 在这里最毒：`sig` 是防空转基准，滑动没
                #    发出去就更新掉 -> 下一帧「指纹没变」-> 假到底。只在 post 写。
                k = self.state["scrolls"] + 1
                sw = self._shelf_swipe(
                    obs, up=False,
                    why=f"{cur_name}: 货架下滑（第 {k} 次，还没到底）",
                    post=lambda s=sig_now, kk=k: self.state.update(
                        sig=s, scrolls=kk, swipe_tick=self.ticks))
                if sw is not None:
                    return sw
                return wait(f"{cur_name}: 货架上没检出行锚点 — 不瞎滑")
            # 指纹停变（且已过停稳闸）= 真到底 -> 转入自底向上购买
            self.state.update(shelf_phase="up", sig="", upscrolls=0)
            self.log(f"{cur_name}: 到底了 — 从最底部开始买起，边上滑边比价")
            return wait(f"{cur_name}: 转入自底向上购买")

        # 上行: 先把当前取景买干净，再上滑一格
        # **货架静止才许买**（2026-08-13 首跑实锤: tap 打在滚动惯性没停的
        #    货架上，按钮已错位，弹框根本没弹 -- 幽灵点击家族）。
        shelf_still = (sig_now == self.state.get("sig_prev")
                       and self.hold("shelf_still", 2))
        self.state["sig_prev"] = sig_now     # 观测事实, decide 期可写
        if (not denied) and self.shop_cfg.get("auto_buy", True) \
                and buyable and shelf_still:
            want = self.shop_cfg.get("currencies") or []
            if not want or cur_name in want:
                act = self._buy_visible(obs, cur_name, buyable, bal, sig_now)
                if act is not None:
                    return act
        if sig_now and sig_now != self.state["sig"] \
                and self.state.get("upscrolls", 0) < 12:
            k = self.state.get("upscrolls", 0) + 1
            sw = self._shelf_swipe(
                obs, up=True,
                why=f"{cur_name}: 这屏买完 — 回看上一屏（第 {k} 次）",
                post=lambda s=sig_now, kk=k: self.state.update(
                    sig=s, upscrolls=kk, swipe_tick=self.ticks))
            if sw is not None:
                return sw
            return wait(f"{cur_name}: 货架上没检出行锚点 — 不瞎滑")
        # 指纹停变 = 回到顶 = 这个 tab 完了
        self.log(f"{cur_name}: 余额 {bal}，可买 {rec['buyable']}，"
                 f"售罄 {rec['soldout']} — 这个 tab 扫完")
        self.state["tab_i"] += 1
        self.state.update(scrolls=0, upscrolls=0, sig="", shelf_phase="down")
        self.state.pop("dead_spots", None)
        self.state.pop("buy_tries", None)
        self.once_reset()
        # 单档回退模式下没有"下一个 tab"，直接收工出推算
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
        """当前取景内: 读价 -> 家具行剔除 -> 单价降序买一件。None = 没有可买的。

        判据来自 scripts/buy_event_shop.py(07-28 清仓实测标定):
          家具行判据: 行内出现重复价 或 行内 max>1000。素材行是严格递增
            四档绝无重复价; 价集/单价过滤挡不住 300 档家具, 只有整行判。
          数学闸: 余额读不出就不买(fail-closed -- 活动币买错了这一期补不
            回来); 单价 > 余额的位置直接不点, 不许"点了才知道买不起"。
        价格按**货架指纹**缓存 -- 首跑 OCR 1917 次/173s 就是每 tick 全货架
          重读(慢 IO 挡热路径老病)。指纹没变 = 货架没变 = 价不用重读。
        08-15 起没有 buy_floor 两档了(自底向上单遍替代两遍, 见 _scan_shelf),
          取景内仍是单价降序。
        """
        tries = self.state.setdefault("buy_tries", {})
        if self.state.get("bought", 0) >= 60:
            return None                       # 安全帽
        bh_all = sorted(max(b.y2 - b.y1, 0.01) for b in buyable)
        bh = bh_all[len(bh_all) // 2]
        xs = sorted(b.cx for b in buyable)
        gaps = [b - a for a, b in zip(xs, xs[1:]) if b - a > 0.03]
        pitch = min(gaps) if gaps else None   # 列距(取最小=相邻列)
        # 行分组: cy 相差 < 1.2 倍按钮高 = 同一行(货架是网格布局)
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
                    continue                  # 点了 3 次都没弹框: 拉黑防死磕
                cands.append((p, b, k))
        if not cands:
            return None
        if bal is None:
            if self.pending("no_bal"):
                self.state["once:no_bal"] = True
                self.log(f"{cur_name}: 余额读不出 — **不自动买**（fail-closed）")
            return None
        afford = [c for c in cands if c[0] <= bal]
        if not afford:
            return None
        p, b, k = max(afford, key=lambda c: c[0])

        def _tried(kk=k):
            t = self.state.setdefault("buy_tries", {})
            t[kk] = t.get(kk, 0) + 1
        # 严格契约: 购买确认框弹出来才算这一发生效（没弹 = 买不起/售罄，
        #    heartbeat 超时后重试，3 次拉黑该位置）。
        # `bought` 只在确认框真的按下確認时才 +1（on_confirm_dialog 那一发）
        #    —— 这里点货架只算"尝试"，没弹框的尝试不算成交（数事实）。
        return tap_box(b, f"{cur_name}: 买单价 {p}（非家具最高价优先，"
                          f"余额 {bal}）",
                       money=True, spend=f"{cur_name}单价{p}",
                       post=_tried, expect=(V.CONFIRM,))

    def on_confirm_dialog(self, obs, st):
        # **反向保险**: 活动商店的商品框价签是活动币。框体内出现 青辉石
        #    （真钱风险）或 体力图标（購買AP 框 —— 青辉石图标在那个框上
        #    检不出, 08-09 实锤, 但体力图标在）都不是我点出来的商品框,
        #    一律取消, 绝不确认。
        bad = obs.find([V.PYROXENE, V.AP], 0.40, region=(0.12, 0.12, 1.0, 1.0))
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

    # ── 收尾：推算「该刷哪一关」并交给 event flow ───────────────────────
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
        try:
            import json as _json
            from pathlib import Path as _P
            _out = _P(__file__).resolve().parents[2] / "data" / "routing_v2"
            _out.mkdir(parents=True, exist_ok=True)
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
