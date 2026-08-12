# -*- coding: utf-8 -*-
"""活动商店 —— **先推算，再决定加成队打哪种币**（用户规则）。

产出放进 `ctx.bag["event_shop_plan"]`，格式:
    {币种cls: {"balance": int|None, "buyable": int, "soldout": int}}
`event` flow 的加成阶段读它来定编队方向。

⛔两条金钱纪律:
  · **最后一个 tab 是盒抽币，绝不自动买**（`skip_last_tab`，默认 True）
  · 活动币不是青辉石，但仍走 money 步（人审）—— 买错了这一期就补不回来了

⛔滑动纪律（用户 2026-08-07 纠正）: 老代码 `for si in range(5)` 固定滑 5 次。
   这里改成**指纹比对**：滑之前记货架签名，滑之后比，没变化 = 到底了。
   签名用"屏上全部 购买/购买灰色 框的 (cx,cy) 四舍五入"—— 不用像素比对，
   因为 UI 动画会让像素一直在变，而框位是稳的。
"""
from __future__ import annotations

import time
from typing import Optional

from routing_v2.act.action import swipe, tap_box, wait
from routing_v2.flow.base import ExitMixin, Flow, Outcome
from routing_v2.percept import read as R
from routing_v2.percept.observe import Observation
from routing_v2.state import vocab as V

SHELF = (0.18, 0.14, 1.0, 0.92)
LEFT_COL = (0.0, 0.10, 0.24, 0.98)

# ⭐⭐**推算映射（用户 2026-08-08 口述，权威）—— 这是个相对规则，不是写死关号**
#
#   把"产出通道"从最珍贵到次之排成一列，**依次对上关卡列表从下往上的关**：
#
#       通道列表 = [活动点数（如果这个活动有）] + [商店币种，从**最下面**往上]
#       通道[k]  →  关卡列表**倒数第 k+1 关**
#
#   · 有活动点数时：点数→最后一关(Q12)，商店最下面的币→Q11，上面一个→Q10
#   · **没有活动点数时：商店最下面的币就是 Q12，然后依次往上推**
#
# ⛔两边都用"从底部数第几个"，所以换活动、换关卡数、换商店档数全都成立，
#   不需要认关卡编号，也不需要给币种训 cls。
# ⛔tab 序位只能按 **cy** 定，不能按 conf 排序（老代码在商店 tab 上串过位）。
def farm_targets(tabs_bottom_up, has_event_points: bool):
    """(通道 → 倒数第几关) 的推导。返回 [{"from_bottom": k, "why": str}]。"""
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


class EventShopFlow(ExitMixin, Flow):
    name = "event_shop"
    module = "event"

    def setup(self) -> None:
        self.state.update(tab_i=0, scrolls=0, sig="", plan={}, bought=0)
        self.shop_cfg = (self.cfg.get("shop") or {})

    # ── 进场 ────────────────────────────────────────────────────────────
    def on_event_page(self, obs, st):
        s = obs.find(V.EVENT_SHOP, 0.40)
        if s is not None and self.pending("enter_shop"):
            # ⛔once 而不是靠重发（08-09 实锤）：点完商店入口后页面在加载，
            #    状态"没变" ⇒ 70 帧后重发，而那时页面已经换了 ⇒ **点到别的
            #    东西**（实测坐标 0.265→0.136→0.265 乱跳，把商店点关了）。
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

        ⛔身份只能按 **cy 序位** 定，不能按 conf 排序 —— 老代码在商店 tab 上
           就是这么串位的。推算映射也是按序位来的（见 FARM_MAP）。
        """
        return obs.rows([V.CURRENCY, V.CURRENCY_SEL], 0.30, region=LEFT_COL)

    def on_event_shop(self, obs, st):
        tabs = self._tabs(obs)               # 按 cy 升序 = 左栏从上到下
        if not tabs:
            # ⛔⛔`货币` cls 会整页检不出（08-09 CODE:BOX 实锤：0.20 模型下限
            #    以上一个都没有，而屏上左栏明明有一档「貨幣」）。这是**感知缺口**，
            #    不是"不在商店页"——页面身份已由货架 `购买` 独立确认过了。
            #    ⇒ 回退：按**单档**推算（左栏只有一档是最常见形态），
            #      日志明说是回退，免得以后把漏档当成真相。
            if self.hold("no_tabs", 45):
                if not self.state.get("single_fallback"):
                    self.state["single_fallback"] = True
                    self.state["ntabs"] = 1
                    self.log("⚠左栏币种 tab 一个都检不出（cls `货币` 感知缺口）"
                             " → **回退按单档推算**，直接扫当前货架")
                return self._scan_shelf(obs, st, cur_name="tab1/1(自下第1)",
                                        from_bottom=0, i=0, n=1)
            return wait("等左栏币种 tab")

        # ⛔用**见过的最大 tab 数**，不用当帧的。切 tab 的那一瞬未选中态会漏检，
        #   拿当帧数量算上限的话，2 个 tab 会被算成 1 个 → limit=0 → 一个都不扫
        #   （08-08 实测「扫完 0/1 个 tab」就是这么来的）。
        self.state["ntabs"] = n = max(self.state.get("ntabs", 0), len(tabs))
        i = self.state["tab_i"]
        # ⚠`skip_last_tab` 的语义是「跳过盒抽币那一档」。而用户口述的映射里
        #   **最下面那个币是 Quest11 的产出，是要买的** —— 所以这里只在
        #   tab 数 ≥3 时才跳最后一个；两个 tab 的活动全都要扫。
        skip_last = bool(self.shop_cfg.get("skip_last_tab", True)) and n >= 3
        limit = max(0, n - (1 if skip_last else 0))
        if i >= limit:
            return self._wrap(
                f"扫完 {limit}/{n} 个 tab"
                + ("（最后一个是盒抽币，按配置跳过）" if skip_last else ""))
        # ⛔tab 身份按 **cy 序位** 定，但**别索引当帧列表**：
        #    ①切 tab 瞬间未选中态漏检 → tabs[i] IndexError 崩 runner（08-09 崩过两轮）
        #    ②选中后**未选中态可以长期检不出**（tab1 未选中在这个活动的皮上
        #      持续漏检）→ "等检出"会永久卡死（08-09 tab2 上停了 10 分钟）。
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

        # ⚠模型没有"具体是哪种币"的 cls ⇒ 身份只能靠**从下往上数第几个**，
        #   这正好就是用户给的映射口径（最下面→Q11，上面一个→Q10）。
        from_bottom = n - 1 - i              # 0 = 最下面那个
        cur_name = f"tab{i+1}/{n}(自下第{from_bottom+1})"
        return self._scan_shelf(obs, st, cur_name, from_bottom, i, n, cur=sel)

    def _scan_shelf(self, obs, st, cur_name, from_bottom, i, n, cur=None):
        """扫当前货架（滑到底 + 记账 + 可选购买）。tab 检出与否都走这里。"""
        # ⛔旧写法 `R.read_beside(obs, cur)` 拿**左栏币种 tab** 当锚 —— 那玩意
        #    右边是币种名字、没有数字 ⇒ 余额恒 None ⇒ 余额闸 fail-closed ⇒
        #    **活动商店一件都买不了**（08-10 实测两个 tab 都是 `余额=None`）。
        #    真锚是右上角的 `货币数量显示区域`，判据封装在 read.read_event_coin。
        bal = R.read_event_coin(obs)

        buyable = obs.all(V.SHOP_BUY, 0.35, region=SHELF)
        soldout = obs.all(V.SHOP_BUY_GREY, 0.35, region=SHELF)
        rec = self.state["plan"].setdefault(
            cur_name, {"balance": bal, "buyable": 0, "soldout": 0,
                       "from_bottom": from_bottom})
        rec["balance"] = bal if bal is not None else rec["balance"]
        rec["buyable"] = max(rec["buyable"], len(buyable))
        rec["soldout"] = max(rec["soldout"], len(soldout))

        # 买（可选）
        if self.shop_cfg.get("auto_buy", True) and buyable:
            want = self.shop_cfg.get("currencies") or []
            if not want or cur_name in want:
                sel = obs.find(V.SHOP_SELECT_ALL, 0.35)
                if sel is not None and self.pending(f"sel{i}"):
                    return tap_box(sel, f"{cur_name}: 全部选择", once=f"sel{i}")
                go = obs.find(V.SHOP_BUY_SELECTED, 0.35)
                if go is not None:
                    return tap_box(go, f"{cur_name}: 批量购买",
                                   money=True, spend=cur_name, counter="bought")
                return tap_box(buyable[0], f"{cur_name}: 购买单项",
                               money=True, spend=cur_name)

        # 货架滑动：指纹没变 = 到底了
        sig = self._shelf_sig(obs)
        if sig and sig != self.state["sig"] and self.state["scrolls"] < 10:
            # ⛔⛔这里的 mutate-before-ack 比别处更毒：`sig` 是**防空转的基准**，
            #    滑动没发出去就把它更新掉 ⇒ 下一帧比对「指纹没变」⇒ 判定到底了
            #    ⇒ 货架后半截永远扫不到。（正是换关假象那次的同一形态。）
            n = self.state["scrolls"] + 1
            return swipe(0.6, 0.72, 0.6, 0.40,
                         f"{cur_name}: 货架下滑（第 {n} 次，指纹变化 → 还没到底）",
                         post=lambda: self.state.update(sig=sig, scrolls=n))
        self.log(f"{cur_name}: 余额 {bal}，可买 {rec['buyable']}，"
                 f"售罄 {rec['soldout']} — 这个 tab 扫完")
        self.state["tab_i"] += 1
        self.state.update(scrolls=0, sig="")
        self.once_reset()
        # 单档回退模式下没有"下一个 tab"，直接收工出推算
        if self.state.get("single_fallback"):
            return self._wrap("单档回退：扫完唯一货架")
        return wait(f"{cur_name} 扫完 → 下一个 tab")

    on_shop = on_event_shop      # 万一被判成通用商店也照样处理

    def on_confirm_dialog(self, obs, st):
        if obs.has(V.CONFIRM_GREY, 0.45):
            c = obs.find(V.CANCEL, 0.45)
            return tap_box(c, "灰确认 → 取消") if c is not None else wait("等取消")
        cf = obs.find(V.CONFIRM, 0.45)
        return tap_box(cf, "确认购买", money=True,
                       spend="活动币") if cf is not None else wait("等確認键")

    def on_reward(self, obs, st):
        b = obs.find([V.CONFIRM, V.GOT_REWARD], 0.40)
        return tap_box(b, "关掉购买结果") if b is not None else wait("等结果页")

    # ── 收尾：推算「该刷哪一关」并交给 event flow ───────────────────────
    def _wrap(self, why):
        """⭐**推算**（用户 2026-08-08 给的映射，见 FARM_MAP）:

            商店最下面那个币还缺 → 去刷**倒数第 2 关**(Q11)
            上面一个币还缺       → 去刷**倒数第 3 关**(Q10)
            活动点数没满         → 去刷**最后一关**  (Q12)

        产出 `ctx.bag["event_farm_plan"]` = 按优先级排好的
        `[{"from_bottom": k, "why": "..."}]`，`from_bottom` 是**关卡列表从下往上
        数第几个**（0=最后一关）。event flow 直接拿它选关，不需要认关卡编号。
        """
        plan = self.state["plan"]
        self.ctx.bag["event_shop_plan"] = plan
        lines = [f"{k}: 余额={v['balance']} 可买={v['buyable']} 售罄={v['soldout']}"
                 for k, v in plan.items()]
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
        # ⭐计划落盘（08-09）：ctx.bag 是进程内存 —— runner 崩溃/step 模式跨
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
            self.log(f"⚠推算计划落盘失败: {e}")
        self.log(f"⭐推算（通道 → 倒数第几关，{'有' if has_pts else '无'}活动点数）:")
        for t in ordered:
            self.log(f"     倒数第 {t['from_bottom']+1} 关  ← {t['why']}")
            self.note_lines.append(
                f"推算目标: 倒数第 {t['from_bottom']+1} 关 ← {t['why']}")
        return self.finish(
            Outcome.CLEAN if plan else Outcome.UNKNOWN,
            f"{why}；购买 {self.state['bought']} 次；" + "；".join(lines))
