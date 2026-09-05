# -*- coding: utf-8 -*-
"""預設面板子链 -- v20 新族 531-541 的唯一消费者。

走法(2026-08-30 实测, flywheel_v20_preset _MANIFEST):
    預設入口(编成页 / 部署侧编队面板右栏) -> 面板开(预设标题) -> 页签 k(1..4,
    身份 = cx 顺位, 最近 在最右) -> 第 r 行的 組成 -> 弹「變更編輯」确认框 ->
    確認 才写入当前部队 -> 叉掉面板。
    讀取 的语义还没单独验证, 这里只走 組成; 空预设行的 組成 是灰的 -> BLOCKED 不点。

安全边界:
  · state['preset_want'] 有值子链才动; 套到哪支部队由调用方先切好部队页签
    (部队1 是用户推图队, 不许覆盖 -- 见 campaign._preset_before_sortie)。
  · 變更編輯 框上的 確認 只在 state['preset_confirm'] 置位时由
    base.Flow.on_preset_change_dialog 点; 别的 flow 撞上这个框一律取消。
  · 面板上 讀取/編輯/複製 永不点。
"""
from __future__ import annotations

import re
import time
from typing import Optional

from routing_v2.act.action import Action, tap_at, tap_box, wait
from routing_v2.percept import read as R
from routing_v2.percept.observe import Box, Observation
from routing_v2.state import vocab as V

_SCROLL_CAP = 4
# 切页签后行区重绘约 1s: 立刻点 組成 会被吞或点到旧行(09-05 部署侧 7 连发无一见到 變更編輯; 手驾等 5s 后一发就弹)
TAB_SETTLE_S = 1.2
# 行头「N部隊」标签与该行 組成 钮的 cy 容差(同一行)
_ROW_TOL = 0.06
# 页签几何兜底(2026-09-03): v20 live 未选中页签 538 一个都检不出(选中态 539 也只 0.4-0.7), 子链会卡在
#    "页签只检出 1 个"。面板是全屏固定 overlay, 5 个页签槽在 16:9 归一化坐标下位置固定
#    (flywheel_v21_preset_20260903 26 帧金标量得: cx 等距, cy 0.226, 0.116 x 0.055)。
#    只在检出凑不齐时用: 有检出的槽用检出(cls 定态), 没检出的槽按几何合成, 选中态用帧像素
#    (选中 = 深蓝底 V~90, 未选中 = 白底 V~245)。锚 预设标题 必须在标准位(cy 0.135), 偏了 = 版式不同, 不用几何。
#    几何落点走 tap_at(锚 预设标题 JIT 复验), 点完仍只认"页签 k 已选中"才往下走, 版式变了最多点到隔壁页签, 不会误套。
_TAB_SLOT_CX = (0.0955, 0.2197, 0.345, 0.470, 0.595)
_TAB_CY, _TAB_W, _TAB_H = 0.226, 0.116, 0.055
_TAB_MATCH_TOL = 0.05
_TAB_TITLE_CY = 0.135
_TAB_SEL_V_MAX = 120


def _tab_slots(obs: Observation, title):
    """5 个页签槽 -> [(Box, selected, src)]。selected None = 没检出也没帧像素可判;
    整体 None = 版式不认识(标题不在标准位 / 检出的页签一个都不在槽上), 调用方回退老口径。"""
    if title is None or abs(title.cy - _TAB_TITLE_CY) > 0.05:
        return None
    det = obs.all([V.PRESET_TAB, V.PRESET_TAB_SEL], 0.40)
    arr = None
    fr = getattr(obs, "frame", None)
    if fr is not None:
        try:
            import numpy as np
            a = np.asarray(fr)
            if a.ndim == 3 and a.shape[0] >= 8 and a.shape[1] >= 8:
                arr = a
        except Exception:
            arr = None
    out, used = [], set()
    for cx in _TAB_SLOT_CX:
        near = [b for b in det if id(b) not in used
                and abs(b.cx - cx) <= _TAB_MATCH_TOL and abs(b.cy - _TAB_CY) <= _TAB_MATCH_TOL]
        if near:
            b = max(near, key=lambda b: b.conf)
            used.add(id(b))
            out.append((b, b.cls == V.PRESET_TAB_SEL, "cls"))
            continue
        box = Box(cls=V.PRESET_TAB, conf=0.0, x1=cx - _TAB_W / 2, y1=_TAB_CY - _TAB_H / 2,
                  x2=cx + _TAB_W / 2, y2=_TAB_CY + _TAB_H / 2)
        sel = None
        if arr is not None:
            h, w = arr.shape[:2]
            xa, xb = int((cx - _TAB_W * 0.3) * w), int((cx + _TAB_W * 0.3) * w)
            ya, yb = int((_TAB_CY - _TAB_H * 0.3) * h), int((_TAB_CY + _TAB_H * 0.3) * h)
            roi = arr[ya:yb, xa:xb]
            if roi.size:
                sel = bool(float(roi.max(axis=2).mean()) < _TAB_SEL_V_MAX)
        out.append((box, sel, "geom"))
    if det and not used:
        return None
    return out


class PresetMixin:
    """混进要套预设的 flow。子类必须是 Flow(用到 state / pending / finish / once_reset)。"""

    def preset_start(self, tab: int, row: int) -> None:
        """登记要套的预设: 页签 tab(1..4), 行 row(1..4)。清掉上一轮的进度标记。"""
        self.state["preset_want"] = {"tab": int(tab), "row": int(row)}
        for k in ("preset_applied", "preset_confirm", "pr_scroll", "pr_tab_t", "preset_dialog_seen"):
            self.state.pop(k, None)
        self.once_reset("pr_open", "pr_tab", "pr_apply", "pr_confirm", "pr_close")

    def preset_done(self) -> bool:
        return bool(self.state.get("preset_applied"))

    def preset_step(self, obs: Observation) -> Optional[Action]:
        """推进一步。返回 None = 没有待办(未登记, 或已套用且面板已关)。"""
        want = self.state.get("preset_want")
        if not want:
            return None
        panel = obs.has(V.PRESET_TITLE, 0.40)
        if obs.has(V.PRESET_CHANGE_TITLE, 0.40) and self.state.get("preset_confirm"):
            self.state["preset_dialog_seen"] = True
        elif self.state.get("preset_dialog_seen") and self.state.get("preset_confirm") and not self.state.get("preset_applied"):
            # 09-05 10-3 第 6 跑: 變更編輯 框的 確認 被通用 ack 处理器抢先点掉, 预设处理器的 post 没跑 -> preset_applied 没置 ->
            #    子链以为面板没开又去重开(第二次 組成). 組成 点过 + 框见过又没了 = 套上了, 不依赖是谁点的 確認.
            self.state["preset_applied"] = True
            self.state.pop("preset_dialog_seen", None)
        if self.state.get("preset_applied"):
            if panel:
                x = obs.find(V.CLOSE_X, 0.55)
                if x is not None:
                    return tap_box(x, "預設已套用 -- 叉掉面板", once="pr_close",
                                   expect_gone=(V.PRESET_TITLE,))
                return wait("預設已套用, 等面板叉叉")
            self.state.pop("preset_want", None)
            return None
        if obs.has(V.PRESET_CHANGE_TITLE, 0.40):
            # 确认框是 overlay, 由 base.on_preset_change_dialog 点確認
            return wait("變更編輯框在场, 交 overlay 处理器確認")
        if not panel:
            ent = obs.find(V.PRESET_ENTRY, 0.40)
            if ent is not None and self.pending("pr_open"):
                return tap_box(ent, "預設: 打开面板", once="pr_open",
                               expect=(V.PRESET_TITLE,))
            return wait("預設: 等入口键" if self.pending("pr_open") else "預設: 等面板打开")
        # 页签: 身份 = cx 顺位; 检出凑不齐时按面板几何补槽(见 _tab_slots)
        k = want["tab"]
        slots = _tab_slots(obs, obs.find(V.PRESET_TITLE, 0.40))
        if slots is None or k > len(slots):
            tabs = sorted(obs.all([V.PRESET_TAB, V.PRESET_TAB_SEL], 0.40), key=lambda b: b.cx)
            if len(tabs) < k:
                return wait(f"預設: 页签只检出 {len(tabs)} 个, 要第 {k} 个")
            slot, selected, src = tabs[k - 1], tabs[k - 1].cls == V.PRESET_TAB_SEL, "cls"
        else:
            slot, selected, src = slots[k - 1]
        if selected is None:
            return wait(f"預設: 页签 {k} 没检出, 帧像素也判不了选中态 -- 等")
        if not selected:
            if self.pending("pr_tab"):
                if src == "cls":
                    return tap_box(slot, f"預設: 切到页签 {k}", once="pr_tab",
                                   post=lambda: self.state.update(pr_tab_t=time.time()))
                a = tap_at(slot.cx, slot.cy, f"預設: 切到页签 {k} (几何兜底: 页签 cls 没检出)",
                              justify="預設面板是全屏固定 overlay, 5 个页签槽位按 flywheel_v21_preset 26 帧金标量得; "
                                      "只在 预设标题 处于标准位时用; 版式变了最多点到隔壁页签, "
                                      "点完仍只认 页签 k 选中态(cls 或像素) 才会去点 組成",
                              require=V.PRESET_TITLE, once="pr_tab")
                a.post = lambda: self.state.update(pr_tab_t=time.time())
                return a
            return wait(f"預設: 等页签 {k} 变选中态")
        tt = float(self.state.get("pr_tab_t", 0) or 0)
        if tt and time.time() - tt < TAB_SETTLE_S:
            return wait(f"預設: 页签刚切换, 等 {TAB_SETTLE_S:.1f}s 行区停稳再点 組成")
        # 行: 面板可滚动且不止 4 行(09-05 live: 4部隊/5部隊 也在), 顺位/滚动计数都不可靠。
        #    定行 = 每个 組成 钮左上方行头「N部隊」的 N 用数字 OCR 读(只读数字, 铁律内; 09-05 实帧 4/5 读准);
        #    其次行头 cls, 再次未滚动时的 cy 顺位。找不到就按读到的行号决定往上还是往下滑。
        r = want["row"]
        applies = obs.rows([V.PRESET_APPLY, V.PRESET_APPLY_GREY], 0.40)
        target = None
        seen_nums = []
        fr = getattr(obs, "frame", None)
        if fr is not None:
            for b in applies:
                txt = R.digits(fr, (0.05, b.cy - 0.155, 0.14, b.cy - 0.095))
                m = re.search(r"\d", txt or "")
                if m:
                    num = int(m.group())
                    seen_nums.append(num)
                    if num == r and target is None:
                        target = b
        tab_cls = V.SQUAD_TABS.get(r, (None, None))[0]
        label = obs.find(tab_cls, 0.40) if tab_cls else None
        if target is None and label is not None and applies:
            near = min(applies, key=lambda b: abs(b.cy - label.cy))
            if abs(near.cy - label.cy) <= _ROW_TOL:
                target = near
        if (target is None and not seen_nums and len(applies) >= r
                and not self.state.get("pr_scroll")):
            target = applies[r - 1]
        if target is None:
            n = int(self.state.get("pr_scroll", 0))
            if n >= _SCROLL_CAP:
                return self.finish("BLOCKED",
                                   f"預設面板滑了 {n} 次仍找不到第 {r} 行(读到行号 {seen_nums}) -- 不瞎点")
            # 滑动几何全从行内钮推: 轴 = 钮的 cx 中位, 行距 = 相邻行 cy 差, 一次滑 1.2 行(面板一屏只见 2 行, 滑 3 行会一步跳到底,
            #    09-05 live: 1/2 行直接跳到 4/5 行); 往上滑要从**最上一行钮**起拖, 起点落在页签栏上不会滚(live 三次白拖)。
            btns = obs.all([V.PRESET_APPLY, V.PRESET_APPLY_GREY, V.PRESET_LOAD], 0.35)
            if not btns:
                return wait("預設面板: 行内钮一个都没检出, 推不出滑动几何")
            from routing_v2.act.action import swipe as _swipe
            xs = sorted(b.cx for b in btns)
            cx = xs[len(xs) // 2]
            cys = sorted(set(round(b.cy, 2) for b in btns))
            gaps = [b - a for a, b in zip(cys, cys[1:]) if b - a > 0.08]
            rowh = sorted(gaps)[len(gaps) // 2] if gaps else max(b.y2 - b.y1 for b in btns) * 6.5
            up = bool(seen_nums) and min(seen_nums) > r
            if up:
                y0 = min(b.cy for b in btns)
                y1 = min(0.92, y0 + 1.2 * rowh)
            else:
                y0 = max(b.cy for b in btns)
                y1 = max(0.08, y0 - 1.2 * rowh)
            if abs(y1 - y0) < 0.05:
                return wait("預設面板: 滑动距离推不出来")
            return _swipe(cx, y0, cx, y1,
                          f"預設面板往{'上' if up else '下'}滑露出第 {r} 行(读到 {seen_nums}, 第 {n + 1} 次)",
                          post=lambda: self.state.update(pr_scroll=n + 1))
        if target.cls == V.PRESET_APPLY_GREY:
            return self.finish("BLOCKED", f"預設 页签{k} 第{r}行 是空预设(組成灰) -- 不套")
        if self.pending("pr_apply"):
            def _armed():
                self.state["preset_confirm"] = True
            return tap_box(target, f"預設: 組成(页签{k} 第{r}行)", once="pr_apply",
                           expect=(V.PRESET_CHANGE_TITLE,), post=_armed)
        return wait("預設: 等變更編輯确认框")

    def on_preset_panel(self, obs, st):
        act = self.preset_step(obs)
        if act is not None:
            return act
        return super().on_preset_panel(obs, st)
