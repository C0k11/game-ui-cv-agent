# -*- coding: utf-8 -*-
"""活动店货架滑动几何/行聚类. 大厅信用点店禁止 import 本模块.

08-16 用户否决: 大厅店不走探底单卡, 只走全部选择/选择购买.
活动店上滑只因本屏没有 103, 禁止「同画面 3 tick=探底完成就回滑」.
make_swipe / cluster_rows 仍给活动店用. ShopFlow 不许再接 ShelfWalkMixin.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

from routing_v2.act.action import Action, swipe, tap_box, wait
from routing_v2.percept.observe import Box, Observation
from routing_v2.state import vocab as V

SHELF = (0.18, 0.14, 1.0, 0.92)
SETTLE_TICKS = 3
SCROLL_CAP = 12
CLIP_Y2 = 0.88


def shelf_btns(obs: Observation, conf: float = 0.30) -> List[Box]:
    return obs.all([V.SHOP_BUY, V.SHOP_BUY_GREY], conf, region=SHELF)


def box_sig(btns: Sequence[Box]) -> str:
    # 带 cls: 同坐标 103<->489 也算画面变了. 只写 cx,cy 会在网格货架上假到底.
    return "|".join(sorted(f"{b.cx:.2f},{b.cy:.2f},{b.cls}" for b in btns))


def coarse_hash(obs: Observation) -> str:
    """货架区粗哈希. 网格滑一屏后按钮坐标不变, 要靠画面内容认新货."""
    frame = getattr(obs, "frame", None)
    if frame is None:
        return ""
    try:
        import numpy as np
        img = np.asarray(frame)
        if img.ndim < 2 or img.size == 0:
            return ""
        h, w = img.shape[:2]
        x1, y1, x2, y2 = SHELF
        xa, xb = max(0, int(x1 * w)), min(w, int(x2 * w))
        ya, yb = max(0, int(y1 * h)), min(h, int(y2 * h))
        crop = img[ya:yb, xa:xb]
        if crop.size == 0:
            return ""
        g = crop.mean(axis=2) if crop.ndim == 3 else crop.astype("float64")
        gh, gw = 6, 8
        ys = np.linspace(0, g.shape[0], gh + 1, dtype=int)
        xs = np.linspace(0, g.shape[1], gw + 1, dtype=int)
        parts = []
        for i in range(gh):
            for j in range(gw):
                block = g[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
                parts.append(str(int(block.mean()) // 8) if block.size else "0")
        return ",".join(parts)
    except Exception:
        return ""


def content_sig(obs: Observation, btns: Optional[Sequence[Box]] = None) -> str:
    if btns is None:
        btns = shelf_btns(obs)
    return box_sig(btns) + "#" + coarse_hash(obs)


def cluster_rows(btns: Sequence[Box]) -> List[List[Box]]:
    if not btns:
        return []
    hs = sorted(max(b.y2 - b.y1, 0.01) for b in btns)
    bh = hs[len(hs) // 2]
    rows: List[List[Box]] = []
    for b in sorted(btns, key=lambda x: x.cy):
        if rows and abs(b.cy - rows[-1][-1].cy) < bh * 1.2:
            rows[-1].append(b)
        else:
            rows.append([b])
    return rows


def bottom_clipped(btns: Sequence[Box]) -> bool:
    return bool(btns) and max(b.y2 for b in btns) > CLIP_Y2


def pick_row_buy(
    rows: Sequence[Sequence[Box]],
    price_of: Callable[[Box], Optional[int]],
    bal: Optional[int] = None,
    row_skip: Optional[Callable[[Sequence[Box]], bool]] = None,
    tried: Optional[dict] = None,
) -> Optional[Tuple[Box, Optional[int]]]:
    """当前最底行里还有 103 就只处理这一行; 全是 489 才看上一行.

    价从高到低; 读不出价按 x 从右到左(货架通常右贵). 489 不点.
    同一槽 103+489 同框 = 半渲染, 当灰, 不点(灰键当亮键旧坑).
    """
    tried = tried or {}
    for row in reversed(list(rows)):
        if row_skip is not None and row_skip(row):
            continue
        bright = [b for b in row if b.cls == V.SHOP_BUY]
        grey = [b for b in row if b.cls == V.SHOP_BUY_GREY]
        clean = []
        for b in bright:
            if any(abs(b.cx - g.cx) < 0.04 and abs(b.cy - g.cy) < 0.04
                   for g in grey):
                continue
            k = f"{b.cx:.2f},{b.cy:.2f}"
            if tried.get(k, 0) >= 3:
                continue
            p = price_of(b)
            if bal is not None and p is not None and p > bal:
                continue
            clean.append((b, p))
        if not clean:
            continue
        clean.sort(key=lambda t: (0, -t[1], -t[0].cx) if t[1] is not None
                   else (1, 0, -t[0].cx))
        return clean[0]
    return None


def make_swipe(obs: Observation, up: bool, why: str, post=None) -> Optional[Action]:
    """货架滑: 锚只用购买/购买灰色, 双向同一把尺. 左栏 tab 不许混进来."""
    bs = shelf_btns(obs)
    if not bs:
        return None
    xs = sorted(b.cx for b in bs)
    cx = xs[len(xs) // 2]
    hs = sorted(max(b.y2 - b.y1, 0.008) for b in bs)
    rowh = hs[len(hs) // 2]
    row_cys: List[float] = []
    for cy in sorted(b.cy for b in bs):
        if not row_cys or cy - row_cys[-1] > rowh * 0.5:
            row_cys.append(cy)
    gaps = [b - a for a, b in zip(row_cys, row_cys[1:])]
    if gaps:
        g = sorted(gaps)[len(gaps) // 2]
        if rowh < g < 0.35:
            rowh = g
    dist = min(0.60, max(0.24, rowh * 3.0))
    if up:
        y0 = max(0.16, min(b.cy for b in bs) - rowh * 0.2)
        y1 = min(0.90, y0 + dist)
    else:
        y0 = min(0.90, max(b.cy for b in bs) + rowh * 0.6)
        y1 = max(0.12, y0 - dist)
    return swipe(cx, y0, cx, y1, why, post=post)


class ShelfWalkMixin:
    """探底 -> 底行 103 贵到便宜 -> 该行清完才上一行. 向上滑只在底行已清之后."""

    def shelf_reset(self) -> None:
        for k in ("shelf_phase", "shelf_scrolls", "shelf_upscrolls",
                  "shelf_await_settle", "shelf_pre_swipe", "shelf_last_up",
                  "shelf_settle_watch", "hold:shelf_settle",
                  "hold:shelf_settle:t"):
            self.state.pop(k, None)

    def shelf_walk(
        self,
        obs: Observation,
        *,
        name: str,
        bal: Optional[int],
        price_of: Callable[[Box], Optional[int]],
        auto_buy: bool,
        denied: bool,
        spend: str,
        expect,
        row_skip: Optional[Callable[[Sequence[Box]], bool]] = None,
        stop_when_denied: bool = False,
    ):
        """返回 Action, 或 'done'(本货架走完). 探底期间绝不点购买、绝不向上滑."""
        btns = shelf_btns(obs)
        buyable = [b for b in btns if b.cls == V.SHOP_BUY]
        soldout = [b for b in btns if b.cls == V.SHOP_BUY_GREY]
        if not buyable and not soldout:
            return wait(f"{name}: 货架不可见(面板盖住/加载中) — 不下结论")

        sig = content_sig(obs, btns)
        if self.state.get("shelf_await_settle"):
            prev = self.state.get("shelf_settle_watch")
            if prev != sig:
                self.state["shelf_settle_watch"] = sig
                return wait(f"{name}: 滑动后等货架停稳")
            if not self.hold("shelf_settle", SETTLE_TICKS):
                return wait(f"{name}: 滑动后等货架停稳")
            self.state["shelf_await_settle"] = False
            pre = self.state.get("shelf_pre_swipe", "")
            changed = bool(sig) and sig != pre
            went_up = bool(self.state.get("shelf_last_up"))
            if not went_up:
                if changed or bottom_clipped(btns):
                    pass
                else:
                    self.state["shelf_phase"] = "buy"
                    self.log(f"{name}: 探底完成, 从最底行开始买")
                    return wait(f"{name}: 探底完成, 从最底行开始买")
            elif not changed:
                return "done"

        # 有亮购买(103)就先买, 探底中途也不滑走. 刚滑到底就回滑是 bug.
        if buyable and (not denied) and auto_buy:
            self.state["shelf_phase"] = "buy"

        phase = self.state.get("shelf_phase", "probe")
        if phase != "buy":
            n = int(self.state.get("shelf_scrolls", 0))
            if n >= SCROLL_CAP:
                self.state["shelf_phase"] = "buy"
                return wait(f"{name}: 下滑次数到顶, 按已到底处理")
            k = n + 1
            sw = make_swipe(
                obs, False, f"{name}: 货架探底下滑(第 {k} 次, 还没到底)",
                post=lambda s=sig, kk=k: self.state.update(
                    shelf_scrolls=kk, shelf_pre_swipe=s,
                    shelf_await_settle=True, shelf_last_up=False,
                    shelf_settle_watch=""))
            if sw is not None:
                return sw
            return wait(f"{name}: 货架上没检出行锚点 — 不瞎滑")

        if denied and stop_when_denied:
            return "done"
        if (not denied) and auto_buy and buyable:
            if bal is None:
                if self.pending("no_bal"):
                    self.state["once:no_bal"] = True
                    self.log(f"{name}: 余额读不出 — 不自动买(fail-closed)")
                return wait(f"{name}: 余额读不出, 不动")
            rows = cluster_rows(btns)
            picked = pick_row_buy(
                rows, price_of, bal=bal, row_skip=row_skip,
                tried=self.state.get("buy_tries"))
            if picked is not None:
                b, p = picked
                k = f"{b.cx:.2f},{b.cy:.2f}"

                def _tried(kk=k):
                    t = self.state.setdefault("buy_tries", {})
                    t[kk] = t.get(kk, 0) + 1

                why = (f"{name}: 底行买" +
                       (f"单价 {p}" if p is not None else "价未读, 右到左") +
                       f"（余额 {bal}）")
                return tap_box(b, why, money=True, spend=spend,
                               post=_tried, expect=expect)

        n = int(self.state.get("shelf_upscrolls", 0))
        if n >= SCROLL_CAP:
            return "done"
        k = n + 1
        sw = make_swipe(
            obs, True, f"{name}: 底行已清, 上滑露上一行(第 {k} 次)",
            post=lambda s=sig, kk=k: self.state.update(
                shelf_upscrolls=kk, shelf_pre_swipe=s,
                shelf_await_settle=True, shelf_last_up=True,
                shelf_settle_watch=""))
        if sw is not None:
            return sw
        return wait(f"{name}: 货架上没检出行锚点 — 不瞎滑")
