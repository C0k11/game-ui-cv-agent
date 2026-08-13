# -*- coding: utf-8 -*-
"""走格子几何层 -- 把 BAAH 的方向语义翻译成**本帧检出的格心坐标**。

BAAH 的答案(`data/baah_grid_solution/`, MIT)是方向序列(`fight_plan`), 走法本身
   是分辨率无关的语义; 脆弱的只是"方向对应哪个像素"。这里所有几何量都从
   **本帧检出的格子**现量, 零写死坐标（用户的全局规矩）。

2026-08-13 小号实帧定标（walk_20260813_083604 帧 119, 全部人工核对过）:
   - 六边形栅格: 同行格距 dx ~0.093, 斜向 (+-dx/2, +-0.120)
   - **单位不站自己格子的中心**: 立绘框心比格心**高 ~0.09**。朴素最近邻
     会把敌方绑到错误格子（实测第一个敌方离起点 0.048 < 离真格子 0.091,
     直接绑反）-> 归属判据是「**正下方**最近的格子」。
   - `走格子_格子_可走`/`起点` 会叠在同一格上, 数格子必须按中心去重。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from routing_v2.percept.observe import Box, Observation
from routing_v2.state import vocab as V

ANSWERS = Path("data/baah_grid_solution")

# 格子族: 所有"这里有一个六边形格子"的证据（可走/迷雾/起点都叠在格子位上）
CELL_CLS = [V.GRID_CELL, V.GRID_CELL_OPEN, V.GRID_CELL_FOG,
            V.GRID_START, V.GRID_START_GREY]

# BAAH 的 6 方向 -> (列步数的倍率, 行步数的倍率)。六边形没有 up/down。
DIRS: Dict[str, Tuple[float, float]] = {
    "right":      (+1.0, 0.0),
    "left":       (-1.0, 0.0),
    "right-up":   (+0.5, -1.0),
    "right-down": (+0.5, +1.0),
    "left-up":    (-0.5, -1.0),
    "left-down":  (-0.5, +1.0),
}


def cells(obs: Observation, conf: float = 0.45) -> List[Tuple[float, float]]:
    """本帧全部格心（去重后）。"""
    out: List[Tuple[float, float]] = []
    boxes = sorted(obs.all(CELL_CLS, conf), key=lambda b: -b.conf)
    for b in boxes:
        # 起点/可走的框往往比格子本体框高一点, conf 排序让格子本体先占位;
        # 去重半径取格距的一小半
        if all((b.cx - x) ** 2 + (b.cy - y) ** 2 > 0.04 ** 2 for x, y in out):
            out.append((b.cx, b.cy))
    return out


def steps(cs: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    """从检出的格心现量 (列步长 dx, 行步长 dy)。量不出返回 None(fail-closed)。

    同一行 = |dy| < 0.03; 相邻行 = dy 在 (0.06, 0.20)。
    只检出一行时 dy 现量不了 -> 用实测比率 dy = dx * 1.28 兜底
    （16:9 下量出来的, 归一化坐标里这个比率跟宽高比走; 检出够两行就不用它）。
    """
    if len(cs) < 2:
        return None
    dxs, dys = [], []
    for i, (x1, y1) in enumerate(cs):
        for x2, y2 in cs[i + 1:]:
            ddx, ddy = abs(x2 - x1), abs(y2 - y1)
            if ddy < 0.03 and 0.05 < ddx < 0.15:
                dxs.append(ddx)
            if 0.06 < ddy < 0.20 and ddx < 0.08:
                dys.append(ddy)
    if not dxs and not dys:
        return None
    dx = sorted(dxs)[len(dxs) // 2] if dxs else None
    dy = sorted(dys)[len(dys) // 2] if dys else None
    if dx is None:
        dx = dy / 1.28
    if dy is None:
        dy = dx * 1.28
    return dx, dy


def below(unit: Box, cs: List[Tuple[float, float]], dx: float
          ) -> Optional[Tuple[float, float]]:
    """这个单位站在哪个格子上 = **正下方**最近的格心（见模块 docstring）。"""
    under = [(x, y) for x, y in cs
             if y > unit.cy - 0.01 and abs(x - unit.cx) < dx * 0.6]
    if under:
        return min(under,
                   key=lambda c: (c[0] - unit.cx) ** 2 + (c[1] - unit.cy) ** 2)
    # 全都不在正下方(格子本体漏检) -> 退回普通最近邻, 但调用方该当心
    return min(cs, key=lambda c: (c[0] - unit.cx) ** 2 + (c[1] - unit.cy) ** 2,
               default=None)


def resolve(at: Tuple[float, float], direction: str,
            cs: List[Tuple[float, float]], dx: float, dy: float
            ) -> Optional[Tuple[float, float]]:
    """从 `at` 沿 `direction` 走一步落在哪个**真格子**上。

    落点取检出的格心而不是推算点（BAAH 是盲走, 我们每步验真格子）。
    推算点附近没有检出的格子 -> None（fail-closed, 宁可停别乱点）。
    """
    mul = DIRS.get(direction)
    if mul is None:
        return None
    ex, ey = at[0] + mul[0] * dx, at[1] + mul[1] * dy
    near = min(cs, key=lambda c: (c[0] - ex) ** 2 + (c[1] - ey) ** 2,
               default=None)
    if near is None:
        return None
    if (near[0] - ex) ** 2 + (near[1] - ey) ** 2 > (0.5 * dx) ** 2:
        return None
    return near


def load_answer(stage: str) -> Optional[dict]:
    """读 BAAH 答案。`stage` 形如 "1-2" / "H3-1"。

    返回 {"areas": [ {"initial_teams": [...], "fight_plan": [round, ...]} ]}
    （多区域关卡按数字键顺序排; `1-1` 这类没有 fight_plan 的返回 areas=[]）。
    """
    p = ANSWERS / f"{stage}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    areas = []
    for k in sorted(k for k in d if k.isdigit()):
        a = d[k]
        areas.append({"initial_teams": a.get("initial_teams", []),
                      "fight_plan": a.get("fight_plan", [])})
    return {"stage": stage, "type": d.get("task_type", "normal"),
            "areas": areas}
