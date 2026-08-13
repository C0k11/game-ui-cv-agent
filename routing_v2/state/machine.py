# -*- coding: utf-8 -*-
"""状态机 —— 把"单帧打分"变成"可信状态"。

 §A3 单帧当真相，一天出现 4 次:
   · cafe 进厅转场帧  判「UI 被隐藏」 盲点写死坐标压在「編輯模式」上
   · 点完入场键的过渡帧  判「无可打关，收工」（其实是弹窗盖住列表）
   · 我的"到达验证"  判「已到达」（其实只是轮播翻了一页）
   · 活动 verify  判「上期领奖页」（其实是当期，只是刚开只解锁 1 关）
   共同点：**转场帧是一瞬，真状态是持续的**。所以这里所有状态都要连续 N 帧。

例外只有一个：**打断型的 `money_popup` 一帧即认**。方向是安全的 —— 停下来
永远不会花钱，而多等 2 帧可能就点下去了。其余打断型照样要连续帧。
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

import routing_v2.act.money as _money
from routing_v2.percept.observe import Observation
from routing_v2.state.pages import Match, classify

# 一帧即认的打断（只允许"绝对安全方向"的那些）
# `quit_dialog` **必须**在这里：2026-08-08 实测，它要连续 3 帧确认时，
#   点完取消的**关闭动画帧**上打断掉了确认，而 `confirm_dialog` 覆盖层还在，
#   于是落到"默认点確認"—— **在退出游戏框上点了確認**。
#   这就是「判定带宽不一致  弹入/弹出动画帧契约必破」那一族。
# `loading` 也必须一帧即认（2026-08-09 task_hall live 实锤）：
#   「Now Loading...」toast 是**呼吸闪烁**的，检出隔帧断续  3 连续帧确认
#   永远凑不齐  interrupt 从没锁上，而 flow 在加载动画上连点了 6 次
#   學園交流會全部无效。单帧 0.97 的 `加载中` 就足以证明页面在加载，
#   误报的代价只是多等一帧 —— 不对称，instant 是对的。
_INSTANT_INTERRUPTS = {"money_popup", "quit_dialog", "loading"}

# 退出框只从这些"大厅系"底页弹得出来（大厅 + 大厅浮层）。
# **绝不能包含 "unknown"**（08-09 实锤）：Machine 初始 last_solid 就是
#    "unknown"，step 模式每次新建  白名单形同虚设，「要使用340AP掃蕩17次嗎」
#    被判成退出框去点取消，扫荡被取消。
#    不对称权衡：**误判**（游戏内框当退出框）没有任何兜底，直接毁流程；
#      **漏判**（真退出框没认出）还有第二道防线 —— base.on_confirm_dialog 里
#      的 `money.system_dialog()`（顶栏货币全无 + 不在设施页  只点取消）。
#     所以这里宁可收紧。
# 这张表**只能有一份**（§A2）。原来这里是 {"lobby", "facility"}，而
#   `money.LOBBY_LIKE` 是 ("lobby",) 且注释明写「不含 facility」并附了事故:
#   facility 是常态页（schedule 的全體課程表面板 / mining / club 掉 cls
#   都落到它），把它算进"可能弹退出框的底页"，会让设施页里的
#   「确认上课 / 确认制造」被当成退出框点取消，flow 再点再取消，
#   乒乓到超时。而这里是 **instant 打断**、优先于 flow —— 两份取值相反时，
#   危险的正是这一份。
_LOBBY_LIKE = set(_money.LOBBY_LIKE)


@dataclass
class StateView:
    """当前可信状态。flow 只看这个，不自己去 classify。"""
    page: str = "unknown"                 # 已确认的**底页**
    raw: str = "unknown"                  # 本帧打分结果（可能是转场帧）
    interrupt: Optional[str] = None       # 已确认的打断
    overlay: Optional[str] = None         # 已确认的**覆盖层**（对话框/奖励框）
    score: float = 0.0
    frames_in_page: int = 0               # 确认后停留了几帧
    secs_in_page: float = 0.0
    changed: bool = False                 # 本 tick 页面身份变了
    flapping: bool = False                # 最近在两个页面之间反复横跳
    last_solid: str = "unknown"           # 上一个**认得出来**的页面（非 blank/unknown）
    history: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        i = f" !{self.interrupt}" if self.interrupt else ""
        o = f" +{self.overlay}" if self.overlay else ""
        return (f"<{self.page}{o}{i} f={self.frames_in_page} "
                f"t={self.secs_in_page:.1f}s raw={self.raw}>")


class Machine:
    def __init__(self, confirm_frames: int = 3, log=None):
        self.confirm = max(1, confirm_frames)
        self._log = log or (lambda m: None)
        self._recent: Deque[str] = deque(maxlen=max(8, self.confirm * 3))
        self._recent_int: Deque[Optional[str]] = deque(maxlen=max(8, self.confirm * 3))
        self._recent_ov: Deque[Optional[str]] = deque(maxlen=max(8, self.confirm * 3))
        self._page = "unknown"
        self._last_solid = "unknown"
        self._interrupt: Optional[str] = None
        self._overlay: Optional[str] = None
        self._since_frames = 0
        self._since_ts = time.time()
        self._changes: Deque[float] = deque(maxlen=12)

    def update(self, obs: Observation) -> StateView:
        m: Match = classify(obs)
        self._recent.append(m.page)
        self._recent_int.append(m.interrupt)
        self._recent_ov.append(m.overlay)
        # 覆盖层同样要连续 N 帧确认（弹入/弹出都有动画帧）
        _ov_tail = list(self._recent_ov)[-self.confirm:]
        self._overlay = (_ov_tail[0] if len(_ov_tail) == self.confirm
                         and len(set(_ov_tail)) == 1 else None)

        # ── 打断 ────────────────────────────────────────────────────────
        if m.interrupt in _INSTANT_INTERRUPTS:
            new_int = m.interrupt
        else:
            tail = list(self._recent_int)[-self.confirm:]
            new_int = (tail[0] if len(tail) == self.confirm
                       and len(set(tail)) == 1 else None)
        # quit_dialog 只可能**从大厅按返回**弹出（这是它的物理来源）。
        #    2026-08-09 血泪：加成扫荡的「要使用 840AP 掃蕩 42 次嗎?」确认框，
        #    在弹入动画帧上顶栏货币+回大厅都还没露出来，只剩 CONFIRM+CANCEL
        #     命中 quit_dialog 签名  instant 打断去点取消  扫荡被取消、
        #    回 stage_popup 又点扫荡开始，死循环 38 次(AP 没掉但流程卡死)。
        #    结构门：底页(last_solid)不是大厅系  这个双键框绝不是退出框，
        #    降级成普通 confirm_dialog 交给 flow 处理。
        if new_int == "quit_dialog" and self._last_solid not in _LOBBY_LIKE:
            new_int = None
        self._interrupt = new_int

        # ── 页面：连续 confirm 帧一致才换 ────────────────────────────────
        changed = False
        tail = list(self._recent)[-self.confirm:]
        if len(tail) == self.confirm and len(set(tail)) == 1 and tail[0] != self._page:
            old = self._page
            self._page = tail[0]
            self._since_frames = 0
            self._since_ts = time.time()
            self._changes.append(time.time())
            changed = True
            if old not in ("blank", "unknown"):
                self._last_solid = old
            self._log(f"[state] {old}  {self._page}  (连续{self.confirm}帧确认)")
        else:
            self._since_frames += 1

        # ── 抖动检测：3s 内换了 ≥4 次页 = 判据打架，别让 flow 继续动手 ──
        now = time.time()
        recent_changes = [t for t in self._changes if now - t < 3.0]
        flapping = len(recent_changes) >= 4
        if flapping:
            # 只用来**抑制兜底恢复**（退出/回大厅那类），flow 自己的动作照常 ——
            #   flow 知道自己在干什么，兜底不知道。抖动往往说明某个页面签名漏了
            #   一个态（08-08 实测: mail 领完后只剩 `领取蓝色`，签名里没有它 
            #   mail↔unknown 来回抖）。看到这条日志就去补签名，别去调阈值。
            self._log(f"[state] 抖动: 3s 内页面身份变了 {len(recent_changes)} 次"
                      f" — 兜底恢复暂停（多半是某个页面签名漏了一个状态）")

        return StateView(
            page=self._page, raw=m.page, interrupt=self._interrupt,
            overlay=self._overlay,
            score=m.score, frames_in_page=self._since_frames,
            secs_in_page=now - self._since_ts, changed=changed,
            flapping=flapping, last_solid=self._last_solid,
            history=list(self._recent),
        )

    def force(self, page: str) -> None:
        """外部强制置位（只给测试/单步工具用，主循环不许调）。"""
        self._page = page
        self._recent.extend([page] * self.confirm)
        self._since_frames = 0
        self._since_ts = time.time()
