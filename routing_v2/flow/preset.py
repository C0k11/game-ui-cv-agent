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

from typing import Optional

from routing_v2.act.action import Action, tap_box, wait
from routing_v2.percept.observe import Observation
from routing_v2.state import vocab as V

_SCROLL_CAP = 3
# 行头「N部隊」标签与该行 組成 钮的 cy 容差(同一行)
_ROW_TOL = 0.06


class PresetMixin:
    """混进要套预设的 flow。子类必须是 Flow(用到 state / pending / finish / once_reset)。"""

    def preset_start(self, tab: int, row: int) -> None:
        """登记要套的预设: 页签 tab(1..4), 行 row(1..4)。清掉上一轮的进度标记。"""
        self.state["preset_want"] = {"tab": int(tab), "row": int(row)}
        for k in ("preset_applied", "preset_confirm", "pr_scroll"):
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
        # 页签: 身份 = cx 顺位
        tabs = sorted(obs.all([V.PRESET_TAB, V.PRESET_TAB_SEL], 0.40), key=lambda b: b.cx)
        k = want["tab"]
        if len(tabs) < k:
            return wait(f"預設: 页签只检出 {len(tabs)} 个, 要第 {k} 个")
        if tabs[k - 1].cls != V.PRESET_TAB_SEL:
            if self.pending("pr_tab"):
                return tap_box(tabs[k - 1], f"預設: 切到页签 {k}", once="pr_tab")
            return wait(f"預設: 等页签 {k} 变选中态")
        # 行: 优先用行头「N部隊」白字标签定行; 检不出时按 cy 顺位(只在未滚动时可信)
        r = want["row"]
        applies = obs.rows([V.PRESET_APPLY, V.PRESET_APPLY_GREY], 0.40)
        target = None
        tab_cls = V.SQUAD_TABS.get(r, (None, None))[0]
        label = obs.find(tab_cls, 0.40) if tab_cls else None
        if label is not None and applies:
            near = min(applies, key=lambda b: abs(b.cy - label.cy))
            if abs(near.cy - label.cy) <= _ROW_TOL:
                target = near
        if target is None and len(applies) >= r and not self.state.get("pr_scroll"):
            target = applies[r - 1]
        if target is None:
            n = int(self.state.get("pr_scroll", 0))
            if n >= _SCROLL_CAP:
                return self.finish("BLOCKED",
                                   f"預設面板滑了 {n} 次仍找不到第 {r} 行 -- 不瞎点")
            from routing_v2.flow.nav import list_swipe
            sw = list_swipe(obs, [V.PRESET_APPLY, V.PRESET_APPLY_GREY, V.PRESET_LOAD],
                            f"預設面板往下滑露出第 {r} 行(第 {n + 1} 次)",
                            post=lambda: self.state.update(pr_scroll=n + 1))
            if sw is not None:
                return sw
            return wait("預設面板: 行内钮一个都没检出, 推不出滑动几何")
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
