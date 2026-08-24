# -*- coding: utf-8 -*-
"""Action —— flow 想干什么。**构造 Action 时就得把复验契约填好。**

设计要点:

【落点只能来自当帧 cls】`tap_box()` 是常规入口；`tap_at()` 存在但强制要求
   `justify=` 说明为什么这里没有 cls 可用。这是给 §A8 加的摩擦力 ——
   `_POS_QUEST_TAB = (0.635, 0.151)` 那种写死坐标之所以能活一年多，就是因为
   写它的时候没有任何阻力，读它的时候也看不出它是哪个版面标的。

【复验契约随 Action 走】`require` 指定"落地前这个 cls 必须还在，且落点还在它
   框里"。这一处集中的 JIT 复验，抵得上老代码里散在各 skill 的十几个 after-ack
   （§A7：知道失效模式就加闸，不许赛跑）。

【金钱步显式声明】`money=True` 的 Action 会被 Gate 拦下交人审，不是靠猜。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from routing_v2.percept.observe import Box


@dataclass
class Action:
    kind: str                       # tap | swipe | key | wait | halt | done | note
    reason: str = ""
    x: float = 0.0
    y: float = 0.0
    x2: float = 0.0
    y2: float = 0.0
    ms: int = 350
    keycode: str = "KEYCODE_BACK"

    #  落地前复验契约（Gate 执行）
    # 复验的是**锚点框有没有移动**，不是"落点在不在框内"。
    #    因为有的落点是**故意**打到框外的（HUB 活动入口 +0.075 打卡片本体，
    #    框标的是倒计时气泡）。第一版写成"落点必须在框内"，离线测试当场把
    #    每一发带偏移的点击都丢掉了 —— 两条规则自己打自己。
    require: Optional[str] = None       # 这个 cls 必须仍在屏上
    require_box: Optional[Box] = None   # 决策时的锚点框（日志用）
    anchor: Optional[tuple] = None      # 决策时锚点框的框心 (cx, cy)
    anchor_tol: float = 0.0             # 允许它漂多少（默认取框的半宽/半高）
    require_conf: float = 0.45

    #  落地**后**的推进契约（Gate 第道闸执行）
    # 2026-08-12 用户点名：「点了一次就去识别下一个要点的 cls，而不是看到
    #    什么点什么，思考逻辑得 **1 step ahead**」。
    #
    #    `require` 管的是**点之前**（目标还在不在），这两个管的是**点之后**
    #    （我期待屏幕变成什么样）。二者合起来才是完整的一发。
    #
    #    为什么必须有它 —— 老的「生效」判据全是**猜**的:
    #      · `page_changed`       弹出的是 overlay 时页面身份根本不变
    #      · 「目标框消失」         同一个 cls 在原位刷新成新内容时不成立
    #      · 「屏上 cls 组成变了」  **弹入动画帧必然让它成立**  291ms 就误判
    #        "已生效"（[[double_fire_family]] 那条「判定带宽不一致  弹入动画
    #        帧契约必破」，2026-08-12 在免費組合包上又复现一次）
    #     三条都是"看起来变了吗"，而不是"**变成我要的样子了吗**"。
    #
    #    **零 wait**：满足即推进，纯帧驱动，不许出现任何 sleep。
    #    **顺带治「一次把接下来的全点一遍」**：pending 期间闸拦掉**所有** tap，
    #      不只是同落点的那一发 —— 屏上同时有好几个可点 cls 时不会连串点下去。
    expect: tuple = ()          # 点完后**应出现**的 cls，任一出现 = 生效
    expect_gone: tuple = ()     # 点完后**应消失**的 cls，全部消失 = 生效
    #    两个都填 = 「或」（谁先满足算谁），不是「与」——弹窗切换时两边
    #      往往不同帧到位，要求同时满足会把正常的一发判成丢了。

    #  金钱
    money: bool = False                 # 声明是金钱步  需人审
    spend: str = ""                     # 花什么（"信用点" / "战术大赛币" / …）

    #  元
    target_cls: str = ""                # 打的是哪个 cls（日志/连发闸用）
    outcome: str = ""                   # kind == done 时的竣工判据
    justify: str = ""                   # tap_at 必填
    # 计数器名。**runner 只在真的 tap 出去之后才 +1**，flow 不许在 decide()
    #    里自己加。2026-08-08 live 实锤：mail 的"領取 #24/#25"全是被闸吞掉的
    #    决策，屏上其实只领了几次 —— 这就是 memory 里那条最贵的教训
    #    「日志是意图不是事实」在新代码里的复发。数事实，别数意图。
    counter: str = ""
    # 同理：`once` 标记也只在 tap **真的发出去之后**才由 runner 落下。
    # 配 `Flow.pending(key)` 用。
    once_key: str = ""
    # 任意状态突变的「数事实」通道：闭包在 tap **真发出去之后**由 runner 调。
    #    counter/once 覆盖不了的场景用它（比如 cafe 摸头要记「摸过哪些位置」——
    #    2026-08-08 发现这段写在 decide() 里，被闸吞掉的那个气泡会被记成
    #    "摸过了"且永不重试，和 N5 是同一个病）。
    post: Optional[object] = None       # Callable[[], None]

    @property
    def is_tap(self) -> bool:
        return self.kind == "tap"

    def __repr__(self) -> str:
        if self.kind == "tap":
            return f"tap({self.target_cls or '?'}@{self.x:.3f},{self.y:.3f}) {self.reason}"
        return f"{self.kind}({self.reason})"


#  构造器

def tap_box(box: Box, reason: str, *, dy: float = 0.0, dx: float = 0.0,
            money: bool = False, spend: str = "", counter: str = "",
            once: str = "", require: Optional[str] = None,
            post: Optional[object] = None,
            expect=(), expect_gone=()) -> Action:
    """点一个检出框。**这是唯一的常规点击入口。**

    `dy` / `dx`: 相对框心的偏移，单位归一化。已知需要偏移的只有一处 ——
      HUB 活动入口 +0.075（405/474 的框标的是**倒计时气泡**，可点的是下面的
      卡片本体；点气泡完全不响应，08-07 手动实测反推）。加偏移必须写清理由。

    `require`: 默认就用这个框自己的 cls 做复验锚点。

    `expect` / `expect_gone`: **1 step ahead 契约** —— 点完之后我期待屏幕
      变成什么样。填了就走推进闸（见 `Gate.advance`），在它满足之前不许发
      任何新 tap。不填 = 行为完全不变（向后兼容，13 条 flow 可以逐步接）。
      单个字符串会被自动包成一元组，省得每处都写 `("x",)` 那个尾逗号。
    """
    x, y = box.cx + dx, box.cy + dy
    if isinstance(expect, str):
        expect = (expect,)
    if isinstance(expect_gone, str):
        expect_gone = (expect_gone,)
    return Action(kind="tap", reason=reason, x=x, y=y,
                  require=require or box.cls, require_box=box,
                  expect=tuple(expect), expect_gone=tuple(expect_gone),
                  anchor=(box.cx, box.cy),
                  # 容差 = 框自身尺寸的 0.6 倍（**不是 0.5**）。
                  # 2026-08-08 实测：`快速制造` 的框在相邻帧间漂了 0.051，
                  # 而半宽是 0.050  合法的一发被自己的复验闸丢掉，craft 那一轮
                  # 一次制造都没开成。框自身的抖动可以超过它的半尺寸。
                  anchor_tol=max(0.025, box.w * 0.6, box.h * 0.6),
                  target_cls=box.cls, money=money, spend=spend, counter=counter,
                  once_key=once, post=post)


def tap_at(x: float, y: float, reason: str, *, justify: str,
           require: Optional[str] = None, money: bool = False,
           once: str = "") -> Action:
    """点一个**没有 cls 支撑**的坐标。

    `justify` 是必填的，而且要具体：写清楚为什么这里没有 cls 可用、这个坐标
       是在什么版面上标的、以及版面变了会怎样。
    如果你写的是"版面固定"，先回去看 §A8 —— `_POS_QUEST_TAB` 当年也是这么写的，
       结果 CODE:BOX 换成两页签版面后，那个坐标正好落在隔壁页签上。
    """
    if not justify:
        raise ValueError("tap_at 必须给 justify —— 没有 cls 支撑的坐标要说明理由")
    return Action(kind="tap", reason=reason, x=x, y=y, justify=justify,
                  require=require, money=money, target_cls="(no-cls)",
                  once_key=once)


def swipe(x1: float, y1: float, x2: float, y2: float, reason: str,
          ms: int = 350, post=None) -> Action:
    """滑动。

    **别写死"滑 N 次"**（用户 2026-08-07 纠正）：
       · 关卡列表：从 1 开始打没打过的关，游戏会自动把下一关归位过来，根本不用滑
       · 活动商店：固定滑 5 次的写法在货架短的时候会滑过头
        滑动必须带**指纹比对**：滑之前记屏，滑之后比，没动就是到底了（见 flow/shop）
    """
    return Action(kind="swipe", reason=reason, x=x1, y=y1, x2=x2, y2=y2, ms=ms,
                  post=post)


def wait(reason: str) -> Action:
    """这一帧什么都不做。**这是 flow 认不出局面时的正确答案**（§A1）。"""
    return Action(kind="wait", reason=reason)


def halt(reason: str) -> Action:
    """立即停整条链，交人。金钱异常/游戏崩溃/判据打架用它。"""
    return Action(kind="halt", reason=reason)


def done(outcome: str, reason: str = "") -> Action:
    """本 flow 收工。outcome ∈ CLEAN / LEFTOVER / UNKNOWN。

    **不许"跑完了"就算完**（§竣工判据）：
      CLEAN    = 该干的都干了，且有帧证据
      LEFTOVER = 还有活没干（票没用完/关没打完），说清剩什么
      UNKNOWN  = 干没干完判断不了（比如战斗胜负看不见）—— 这也是合法答案，
                 比谎报 CLEAN 强得多
    """
    return Action(kind="done", reason=reason or outcome, outcome=outcome)


def note(reason: str) -> Action:
    """只记一笔日志，不动作。"""
    return Action(kind="note", reason=reason)
