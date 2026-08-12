# -*- coding: utf-8 -*-
"""Observation —— 一帧的全部结构化事实。上层**只**通过它查询屏幕。

为什么把查询 API 收在这里（README §A2）:
  老代码里 `find_cls` / `find_yolo_box` / `F(bs, ...)` 各写各的，region 参数
  有的支持有的不支持 —— 于是「840AP 事故」那种 bug（顶栏读数没带 region，
  被弹窗体内的同名图标抢走锚点）会在每个 skill 各复发一次。
  这里只有一套查询语义，region 是**一等公民**。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple, Union

Rect = Tuple[float, float, float, float]          # (x1, y1, x2, y2) 归一化
Names = Union[str, Sequence[str]]


@dataclass(frozen=True)
class Box:
    """一个检出框。坐标一律归一化 0-1（与分辨率无关）。"""
    cls: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float
    model: str = "ui"

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[float, float]:
        return (self.cx, self.cy)

    def contains(self, nx: float, ny: float, pad: float = 0.0) -> bool:
        """落点在不在框里（JIT 复验用；`pad` 单位是归一化坐标）。"""
        return (self.x1 - pad <= nx <= self.x2 + pad
                and self.y1 - pad <= ny <= self.y2 + pad)

    def offset(self, dx: float = 0.0, dy: float = 0.0) -> Tuple[float, float]:
        """相对框心的落点。⭐HUB 活动入口要 +0.075 —— 405/474 的框标的是
        **倒计时气泡**，可点的是下面的卡片本体（08-07 手动实测反推）。"""
        return (self.cx + dx, self.cy + dy)

    def __repr__(self) -> str:
        return f"<{self.cls} {self.conf:.2f} @({self.cx:.3f},{self.cy:.3f})>"


def _as_list(names: Names) -> List[str]:
    return [names] if isinstance(names, str) else list(names)


@dataclass
class Observation:
    """一帧 + 它的全部检出 + 元信息。不可变语义（不要就地改 boxes）。"""
    boxes: List[Box] = field(default_factory=list)
    frame: object = field(default=None, repr=False)
    seq: int = 0
    age: float = 0.0
    ts: float = 0.0
    w: int = 0
    h: int = 0
    # 每 tick 的余额台账（read.py 填；可能为 None = 这帧没读/读不出）
    balances: dict = field(default_factory=dict)

    # ── 查询 ────────────────────────────────────────────────────────────
    def all(self, names: Names, conf: float = 0.45, *,
            region: Optional[Rect] = None,
            ymin: float = 0.0, ymax: float = 1.0,
            xmin: float = 0.0, xmax: float = 1.0,
            model: Optional[str] = None) -> List[Box]:
        """按 conf 降序返回全部匹配框。region 按**框心**判定。"""
        want = set(_as_list(names))
        if region is not None:
            xmin, ymin, xmax, ymax = region
        out = [b for b in self.boxes
               if b.cls in want and b.conf >= conf
               and xmin <= b.cx <= xmax and ymin <= b.cy <= ymax
               and (model is None or b.model == model)]
        out.sort(key=lambda b: -b.conf)
        return out

    def find(self, names: Names, conf: float = 0.45, **kw) -> Optional[Box]:
        """最高 conf 的那个，没有就 None。**None 意味着不点** —— 别兜底。"""
        got = self.all(names, conf, **kw)
        return got[0] if got else None

    def has(self, names: Names, conf: float = 0.45, **kw) -> bool:
        return bool(self.all(names, conf, **kw))

    def count(self, names: Names, conf: float = 0.45, **kw) -> int:
        return len(self.all(names, conf, **kw))

    def rows(self, names: Names, conf: float = 0.45, **kw) -> List[Box]:
        """按 cy 升序 —— 列表行、tab 顺位这类"从上到下"的语义用它。
        ⛔别用 conf 排序去猜顺位（商店 tab 身份就是这么错过位的）。"""
        got = self.all(names, conf, **kw)
        got.sort(key=lambda b: b.cy)
        return got

    def cols(self, names: Names, conf: float = 0.45, **kw) -> List[Box]:
        """按 cx 升序 —— 横排 tab / 页签用它。"""
        got = self.all(names, conf, **kw)
        got.sort(key=lambda b: b.cx)
        return got

    def nearest(self, names: Names, to: Tuple[float, float],
                conf: float = 0.45, **kw) -> Optional[Box]:
        """离某点最近的框 —— 红点归属这类"按距离判"的语义用它
        （大厅红点属于旁边九宫格还是邮件，只能按距离判）。"""
        got = self.all(names, conf, **kw)
        if not got:
            return None
        return min(got, key=lambda b: (b.cx - to[0]) ** 2 + (b.cy - to[1]) ** 2)

    @property
    def names(self) -> set:
        """本帧出现过的全部 cls 名（conf≥0.45）—— 页面签名匹配用。"""
        return {b.cls for b in self.boxes if b.conf >= 0.45}

    def names_at(self, conf: float) -> set:
        return {b.cls for b in self.boxes if b.conf >= conf}

    def summary(self, n: int = 10) -> str:
        top = sorted(self.boxes, key=lambda b: -b.conf)[:n]
        return " | ".join(f"{b.cls}:{b.conf:.2f}" for b in top)


def annotate(obs: Observation, target: Optional[Box] = None,
             note: str = "", width: int = 1280):
    """画锁定框的调试图。逐帧门控时**必须看这个**，不是自己截图
    （用户 2026-08-08: 「而且你也不看 yolo 锁定画面」）。"""
    import cv2
    if obs.frame is None:
        return None
    ann = obs.frame.copy()
    h, w = ann.shape[:2]
    for b in obs.boxes:
        hit = target is not None and b is target
        cv2.rectangle(ann, (int(b.x1 * w), int(b.y1 * h)),
                      (int(b.x2 * w), int(b.y2 * h)),
                      (0, 0, 255) if hit else (0, 255, 0), 8 if hit else 3)
        cv2.putText(ann, f"{b.cls} {b.conf:.2f}",
                    (int(b.x1 * w), max(28, int(b.y1 * h) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3)
    if target is not None:
        cv2.drawMarker(ann, (int(target.cx * w), int(target.cy * h)),
                       (255, 0, 255), cv2.MARKER_CROSS, 110, 8)
    if note:
        cv2.putText(ann, note, (24, 56), cv2.FONT_HERSHEY_SIMPLEX,
                    1.6, (0, 255, 255), 5)
    return cv2.resize(ann, (width, int(width * h / w)))
