# -*- coding: utf-8 -*-
"""多队走格子(2 队 / 3 队, exchange, portal) -- campaign 的子链, 2026-09-05 大号 10-1 / 10-2 / 11-1 手驾实测后写。

实测机制(全部 live 验证, 见 memory grid_multi_20260905):
  部署   每个 START 上方挂一个黄色倒三角(543 起点悬停, 模型常把它认成 501 队伍箭头 -- 同一张贴图,
         部署阶段两类一律并成"标记"); 点 START 开编队面板, 游戏自动把下一支没上的部队高亮,
         出击 后相机会平移; 上一队后 任務開始 就已经变黄 -- **不能像单队那样见黄就点**,
         必须数到 needs.teams 支都上了。被底部学生卡条挡住的 START 要拖地图(11-1 实锤)。
  回合   开局聚焦部队 1(箭头在它头上); 一队行动完游戏自动把焦点切到下一支没行动的队;
         行动完自动结束 PHASE(勾着 自动结束)。答案里某回合不含某队时不会自动结束, 要手点 PHASE結束
         + 确认「尚未行動」框。
  切队   点友军格**不换焦点**, 弹「選擇 / 變更位置」双键菜单; 左下角循环图标「N部隊」药丸键循环切焦点
         (BAAH 写死 (82,554) 的那个键), 相机跟着平移到新焦点队。
  换位   exchange = 聚焦 A 后点 B 所在格 -> 菜单 -> 變更位置 -> A/B 互换, **不消耗行动**, 焦点仍在 A。
  传送   portal = 点传送格 -> 「通知 / 是否移動該部隊？」双键框 -> 確認 -> 队伍消失在别处出现,
         **消耗行动**, 焦点切到下一队。传送后该队位置未知, 等它再被聚焦时用箭头绑回。
  地图   相机每次行动后都平移, 屏幕坐标一帧一变; 唯一不变的是格子点阵。所以这里把地图存成
         **点阵坐标**(起点 A 为原点, 列用半格计 c2, 行 r), 每帧用可见起点/格子把点阵对齐到屏幕,
         队伍位置按答案方向在点阵里航位推算(exchange 互换, portal 置未知)。
         踩开关格会**长出新格**(10-1/10-2 的桥), 对齐可信时把新格并进地图。

菜单键 / 药丸键 / 传送格 都还没有 cls(v22 采了料), 本版用几何落点 + 事后证据(箭头位置)兜底;
   每一发几何点击都带 justify, 版式变了最坏是点空, 不会点到危险控件(部署菜单_解除 由 gate 拦)。
"""
from __future__ import annotations

import itertools
import math
import time
from typing import Dict, List, Optional, Tuple

from routing_v2.act.action import Action, swipe, tap_at, tap_box, wait
from routing_v2.flow import grid
from routing_v2.flow.base import Outcome
from routing_v2.percept.observe import Box, Observation
from routing_v2.state import vocab as V

# 部署阶段"这个起点还没上队"的标记: 543 悬停倒三角; 模型常把同一张贴图认成 501 箭头(部署屏上不可能有真箭头)
DEPLOY_MARKS = [V.GRID_START_HOVER, V.GRID_ARROW]
STARTS = [V.GRID_START, V.GRID_START_GREY]

# 左下角「N部隊」切队药丸(16:9 归一化, 10-1/10-2/11-1 三关 40+ 帧位置不变: 药丸 x 0.02-0.11, y 0.75-0.80)
SWITCH_PILL = (0.062, 0.775)
# 点友军格弹出的双键菜单, 相对被点格心的偏移(10-2 r1 实测: 格 (0.515,0.544), 變更位置 图标 (0.432,0.528),
#    選擇 图标 (0.437,0.421)); 菜单挂在单位左侧, 相机不缩放, 偏移是常量
MENU_EXCHANGE_DXY = (-0.083, -0.016)
MENU_SELECT_DXY = (-0.078, -0.123)
# 拖地图找被挡住的起点: 一次拖 0.28 屏高/宽
DRAG = 0.28
# 编队面板右栏「預設」图标(16:9 归一化; 快速編輯 0.247 / 起始技能 0.39 / 部隊資訊 0.53 / 預設 0.672)
PRESET_ENTRY_XY = (0.937, 0.672)
# 墙钟门槛(离线用例置 0): 空闲要持续多久才许落子; 点完友军格等菜单弹稳多久
IDLE_HOLD_S = 1.5
CAM_MOVE_MIN = 0.015     # 真走过一步相机必平移(实测 0.05-0.19); move 的"焦点切走"证据要伴随原点漂移 >= 这个值(测试里置 0)
CELL_TAP_DOWN = 0.33     # 走格落点压到格心下方这么多行距: 单位立绘从格心向上长, 相机放大时会盖住相邻格心(10-3 第 4 跑实锤)
MENU_WAIT_S = 1.2
FORM_SETTLE_S = 1.5      # 编队面板滑入动画期间点右栏图标会被吞(09-05 live), 面板出现后先等这么久
PR_OPEN_RETRY_S = 4.0    # 点了預設入口这么久还没见 预设标题 = 那一发被吞, 重点(最多 3 次)

# 6 方向在点阵里的步(列按半格 c2 计, 行 r; 右下 = 半格右 + 一行下)
DIR_LAT: Dict[str, Tuple[int, int]] = {
    "right": (2, 0), "left": (-2, 0),
    "right-up": (1, -1), "right-down": (1, 1),
    "left-up": (-1, -1), "left-down": (-1, 1),
}
# 8 向部署方位 -> 屏幕方向(y 向下); center = 零向量(离质心最近的那个起点)
_S2 = 0.7071
POS_VEC = {"up": (0.0, -1.0), "down": (0.0, 1.0), "left": (-1.0, 0.0), "right": (1.0, 0.0),
           "left-up": (-_S2, -_S2), "right-up": (_S2, -_S2),
           "left-down": (-_S2, _S2), "right-down": (_S2, _S2), "center": (0.0, 0.0)}


#  纯几何(无状态, 离线可测)

def assign_starts(starts: List[Tuple[float, float]], teams: List[dict],
                  dx: Optional[float] = None, dy: Optional[float] = None) -> Optional[Dict[str, int]]:
    """起点框心 x 答案 teams(pos) -> {队名: 起点下标}。起点数 != 队数 -> None。

    方位 = 相对全部起点质心的方向(BAAH position 语义: 2 队 left/right, 3 队 up/center/down)。
    穷举排列取总分最高: 有向方位得 cos(方位, 偏移), center 得 1-|偏移|/max。偏移按格距归一(dx,dy)
    再算角度, 不然 16:9 下斜向会被纵向拉偏。排列打分比逐个最近方位稳: 11-1 的 up 起点单独看更像
    right-up, 但 (up,down) 的排列总分 1.78 远高于反过来的 -1.78。"""
    n = len(starts)
    if n == 0 or n != len(teams):
        return None
    if n == 1:
        return {teams[0]["name"]: 0}
    sx = dx or 1.0
    sy = dy or 1.0
    cx = sum(s[0] for s in starts) / n
    cy = sum(s[1] for s in starts) / n
    offs = [((s[0] - cx) / sx, (s[1] - cy) / sy) for s in starts]
    mx = max(math.hypot(*o) for o in offs) or 1.0
    best = None
    for perm in itertools.permutations(range(n)):
        score = 0.0
        for t, si in zip(teams, perm):
            v = POS_VEC.get(t.get("pos"), (0.0, 0.0))
            o = offs[si]
            d = math.hypot(*o)
            if v == (0.0, 0.0):
                score += 1.0 - d / mx
            elif d > 1e-6:
                score += (v[0] * o[0] + v[1] * o[1]) / d
        if best is None or score > best[0]:
            best = (score, perm)
    return {t["name"]: si for t, si in zip(teams, best[1])}


def lat_of(px: Tuple[float, float], origin: Tuple[float, float], dx: float, dy: float) -> Tuple[int, int]:
    """屏幕点 -> 点阵坐标 (c2, r)。原点是点阵 (0,0) 的屏幕位置。六边形错行: 同一连通点阵里 c2 与 r 同奇偶,
    远处格子 dx 累计误差可能把 c2 舍到错的奇偶, 往精确值那侧拨一格纠正。"""
    r = int(round((px[1] - origin[1]) / dy))
    fc = 2.0 * (px[0] - origin[0]) / dx
    c2 = int(round(fc))
    if (c2 + r) % 2 != 0:
        c2 = c2 + 1 if fc > c2 else c2 - 1
    return (c2, r)


def px_of(lat: Tuple[int, int], origin: Tuple[float, float], dx: float, dy: float) -> Tuple[float, float]:
    return (origin[0] + lat[0] * dx / 2.0, origin[1] + lat[1] * dy)


def lat_add(a: Tuple[int, int], d: str) -> Optional[Tuple[int, int]]:
    v = DIR_LAT.get(d)
    if v is None:
        return None
    return (a[0] + v[0], a[1] + v[1])


def build_map(cells_px: List[Tuple[float, float]], starts_px: List[Tuple[float, float]],
              teams: List[dict], dx: float, dy: float) -> Optional[dict]:
    """部署屏首帧(全部起点可见)建点阵地图。原点 = 答案第一队的起点。"""
    asg = assign_starts(starts_px, teams, dx, dy)
    if asg is None:
        return None
    origin = starts_px[asg[teams[0]["name"]]]
    starts = {name: list(lat_of(starts_px[i], origin, dx, dy)) for name, i in asg.items()}
    cells = set(tuple(v) for v in starts.values())
    for c in cells_px:
        cells.add(lat_of(c, origin, dx, dy))
    return {"cells": sorted(cells), "starts": starts}


def align(mapd: dict, cells_px: List[Tuple[float, float]], starts_px: List[Tuple[float, float]],
          dx: float, dy: float) -> Optional[Tuple[Tuple[float, float], int]]:
    """把点阵地图对齐到本帧 -> (原点屏幕坐标, 得分)。对不齐 -> None(fail-closed, 调用方等下一帧)。

    假设 = 每个可见起点 x 每个已知起点; 一个起点都没检出时退回 格子 x 地图格(最多 12x40 组)。
    得分 = 落在地图格上的检出格数 + 2 x 落在起点位的检出起点数; 起点假设优先。
    门槛 max(3, 半数检出格), 且要比次优高 >= 1, 不然对称地图会二义。"""
    mcells = set(tuple(c) for c in mapd["cells"])
    mstarts = {k: tuple(v) for k, v in mapd["starts"].items()}
    sset = set(mstarts.values())
    hyps = []
    for s in starts_px:
        for L, sl in mstarts.items():
            hyps.append((s[0] - sl[0] * dx / 2.0, s[1] - sl[1] * dy, 1))
    if not hyps:
        for c in cells_px[:12]:
            for m in list(mcells)[:40]:
                hyps.append((c[0] - m[0] * dx / 2.0, c[1] - m[1] * dy, 0))
    scored = []
    for ox, oy, pri in hyps:
        sc = 0
        for x, y in cells_px:
            c2, r = lat_of((x, y), (ox, oy), dx, dy)
            if (c2, r) in mcells:
                ex, ey = px_of((c2, r), (ox, oy), dx, dy)
                if abs(ex - x) < 0.35 * dx and abs(ey - y) < 0.35 * dy:
                    sc += 1
        for x, y in starts_px:
            if lat_of((x, y), (ox, oy), dx, dy) in sset:
                sc += 2
        scored.append((sc, pri, ox, oy))
    if not scored:
        return None
    scored.sort(key=lambda t: (-t[0], -t[1]))
    sc, _, ox, oy = scored[0]
    if sc < max(3, len(cells_px) // 2):
        return None
    # 次优若是**不同原点**且分数只差 <1 -> 二义
    for sc2, _, ox2, oy2 in scored[1:]:
        if abs(ox2 - ox) > 0.3 * dx or abs(oy2 - oy) > 0.3 * dy:
            if sc - sc2 < 1:
                return None
            break
    return (ox, oy), sc


def _menu_icon_at(frame, x: float, y: float):
    """(x,y) 归一化处是不是「選擇/變更位置」那种白圆底蓝 glyph 的菜单图标。True/False; 没帧 -> None(不核对)。"""
    if frame is None:
        return None
    try:
        import cv2
        import numpy as np
        h, w = frame.shape[:2]
        cx, cy = int(x * w), int(y * h)
        r = max(3, int(0.006 * w))
        patch = frame[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1]
        if patch.size == 0:
            return None
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        blue = (hsv[..., 1] >= 90) & (hsv[..., 0] >= 95) & (hsv[..., 0] <= 130) & (hsv[..., 2] >= 120)
        return bool(blue.mean() >= 0.15)
    except Exception:
        return None


class GridMultiMixin:
    """挂在 CampaignFlow 上。状态全在 self.state['mt_*'], 单队关不碰这里。"""

    #  开关 / 状态

    def _multi(self) -> bool:
        a = self.state.get("answer") or {}
        try:
            return int((a.get("needs") or {}).get("teams", 1)) > 1
        except (TypeError, ValueError):
            return False

    def _mt_teams(self) -> List[dict]:
        return list((self.state.get("answer") or {}).get("teams") or [])

    def _mt_names(self) -> List[str]:
        return [t["name"] for t in self._mt_teams()]

    def mt_reset(self) -> None:
        for k in [k for k in self.state if k.startswith("mt_")]:
            self.state.pop(k, None)

    def mt_new_round(self) -> None:
        """相位循环完成(do_walk 里 round_i += 1 之后)调。还挂着的动作 = 被这次循环消费了, 先记账。"""
        pend = self.state.get("mt_pending")
        if pend:
            self._mt_apply(pend)
        # 新回合开场有 MY PHASE 横幅 + 相机回摆 1-2s(09-05 第 14 次 live: 每回合首发都在这窗口里打空, 10s 后重发才中)
        self.state["mt_settle_until"] = time.time() + 5.0
        self.state["mt_focus_prev"] = None
        self.state["mt_ai"] = 0
        self.state["mt_pending"] = None
        self.state["mt_acted"] = []
        self.state["mt_need_end"] = False
        self.state.pop("mt_end_taps", None)
        for k in [k for k in self.state if k.startswith("mt_focus:") or k.startswith("mt_reissue:")]:
            self.state.pop(k, None)

    def _mt_dbg(self, obs: Observation, note: str) -> None:
        """决策帧落盘(复盘用; runner 存的是换页帧不是决策帧, 09-05 复盘被这个坑了两轮)。"""
        try:
            path = self._dump_grid_miss(obs)
            if path:
                self.log(f"决策帧 {path.rsplit('/', 1)[-1]}: {note}")
        except Exception:
            pass

    def _mt_idle_ok(self, obs: Observation) -> bool:
        """游戏空闲闸(09-05 10-3 live 实锤): 踩敌人格的 SKIP 战斗有 3-5s 爆炸/VICTORY 动画, 期间右下角轮流盖着
        「Now Loading」和横幅, 箭头会闪一下 -- 这一闪被当成"上一发已消费", 紧接着的两发全打在动画里被吞。
        空闲 = PHASE結束 高分在场 且 没有 加载中, **连续 3 帧**(VICTORY 那一帧 PHASE 会露出来一下)。
        不空闲时既不落子也不采信证据, 挂起动作的超时钟也暂停。"""
        idle = obs.has(V.PHASE_END, 0.60) and not obs.has(V.LOADING, 0.40)
        now = time.time()
        if not idle:
            self.state["mt_idle_since"] = None
            self.state["mt_idle_n"] = 0
            return False
        if not self.state.get("mt_idle_since"):
            self.state["mt_idle_since"] = now
        self.state["mt_idle_n"] = int(self.state.get("mt_idle_n", 0)) + 1
        # 连续空闲 >= 1.5s: SKIP 战斗的两段加载之间会露出 VICTORY 横幅帧(PHASE 在、无加载中), 按帧数会漏过去
        return (now - float(self.state["mt_idle_since"])) >= IDLE_HOLD_S and self.state["mt_idle_n"] >= 3

    def _mt_frame(self, obs: Observation, conf: float = 0.30):
        """本帧几何: (格心列表, 起点框列表, dx, dy, 原点) 或 None。"""
        cs = grid.cells(obs, conf)
        stp = grid.steps(cs)
        if stp is None:
            return None
        dx, dy = stp
        sb = obs.all(STARTS, 0.35)
        mapd = self.state.get("mt_map")
        origin = None
        if mapd:
            got = align(mapd, cs, [(b.cx, b.cy) for b in sb], dx, dy)
            if got is not None:
                origin = got[0]
                self._mt_grow_map(mapd, cs, origin, dx, dy, got[1], obs)
        return cs, sb, dx, dy, origin

    def _mt_grow_map(self, mapd, cs, origin, dx, dy, score, obs) -> None:
        """对齐可信(>=5 分)时把连续两帧都看到的新格并进地图(踩开关长出来的桥)。"""
        if score < 5:
            return
        known = set(tuple(c) for c in mapd["cells"])
        seen = set()
        for x, y in cs:
            l = lat_of((x, y), origin, dx, dy)
            ex, ey = px_of(l, origin, dx, dy)
            if l not in known and abs(ex - x) < 0.35 * dx and abs(ey - y) < 0.35 * dy:
                seen.add(l)
        prev = set(tuple(v) for v in self.state.get("mt_newc") or [])
        add = seen & prev
        if add:
            mapd["cells"] = sorted(known | add)
            self.log(f"地图长出新格 {sorted(add)}(连续两帧检出)")
        self.state["mt_newc"] = sorted(seen - add)

    #  部署

    def mt_deploy_step(self, obs: Observation, st) -> Optional[Action]:
        """多队部署: 数着上队, 上齐了才 任務開始。返回 None 表示交回 do_grid 的通用分支(弹窗/菜单已在上面处理)。"""
        ans = self.state.get("answer") or {}
        need = int((ans.get("needs") or {}).get("teams", 1))
        teams = self._mt_teams()
        dep: List[str] = self.state.setdefault("mt_deployed", [])
        # 真开局的事实 = PHASE 控件出现
        if obs.has(V.PHASE_END, 0.40) or obs.has(V.PHASE_AUTO_ON, 0.40):
            if len(dep) < need:
                self.log(f"PHASE 控件出现但只记到 {len(dep)}/{need} 队上场 -- 按屏上事实进回合, 剩下的队按未部署处理")
            self._mt_start_walk()
            self.goto("walk", "PHASE 控件出现 = 真开局了(多队)")
            return wait("进回合")
        # 编队面板: 先按配置挑部队, 再出击
        if (st.page in ("formation", "preset_panel") or obs.has(V.SORTIE, 0.45)
                or obs.has(V.PRESET_TITLE, 0.40)):
            return self._mt_formation(obs)
        fr = self._mt_frame(obs, 0.35)
        if fr is None:
            if self._overdue("mt_nogrid", 60):
                return self.finish(Outcome.UNKNOWN, "部署屏 60s 量不出格距 -- 感知不足, 不瞎点")
            return wait("部署屏: 等格子检出")
        self._wt_clear("mt_nogrid")
        cs, sb, dx, dy, origin = fr
        # 首帧建地图: 起点数必须等于答案队数
        if self.state.get("mt_map") is None:
            if len(sb) == need and len(cs) >= 3:
                m = build_map(cs, [(b.cx, b.cy) for b in sb], teams, dx, dy)
                if m is not None:
                    self.state["mt_map"] = m
                    self.log(f"多队地图建好: 起点 {m['starts']} 格 {len(m['cells'])}")
                    return wait("地图建好, 下一帧开始部署")
            n = self.bump("mt_map_wait")
            if n > 40:
                return self.finish(Outcome.UNKNOWN,
                                   f"部署屏 40 帧里可见起点 {len(sb)} 个 != 答案 {need} 队 -- 起点检出不全, 不瞎点")
            return wait(f"等全部 {need} 个起点同时可见({len(sb)} 个)")
        mapd = self.state["mt_map"]
        # 上一队刚出击: 它的标记消失了才算真上了(数事实)
        pend = self.state.get("mt_dep_pending")
        marks = self._mt_marks(obs)
        under = []
        for m in marks:
            sp = grid.start_under_hover(obs, m)
            if sp is not None:
                under.append(sp)
        if origin is not None:
            names_under = set()
            for sp in under:
                L = self._mt_start_letter(mapd, sp, origin, dx, dy)
                if L:
                    names_under.add(L)
            if pend and pend not in names_under:
                if pend not in dep:
                    dep.append(pend)
                    self.log(f"队 {pend} 已上场({len(dep)}/{need})")
                self.state["mt_dep_pending"] = None
            elif pend and pend in names_under:
                # 出击了标记还在 = 那一发没成, 让它重新点
                self.state["mt_dep_pending"] = None
        if len(dep) >= need:
            start_btn = obs.find(V.TASK_START, 0.45)
            if start_btn is not None:
                return tap_box(start_btn, f"任務開始({need}/{need} 队都上了)",
                               expect=(V.PHASE_END, V.PHASE_AUTO_ON, V.PHASE_AUTO_OFF))
            return wait("上齐了, 等 任務開始 变黄")
        # 还差队: 找一个没上的起点(优先答案顺序里下一个)
        if origin is None:
            n = self.bump("mt_align_fail")
            if n > 30:
                return self.finish(Outcome.UNKNOWN, "部署屏 30 帧点阵对不齐 -- 不瞎点")
            return wait("点阵对不齐, 再看一帧")
        self.state["mt_align_fail"] = 0
        sortied = set((self.state.get("mt_team_squad") or {}).keys())
        remaining = [n for n in self._mt_names() if n not in dep and n not in sortied]
        if not remaining and len(dep) < need:
            # 出击都发过了只是标记消失还没看到: 等标记帧, 别再去点起点(点已上队的起点会弹 解除 菜单)
            return wait(f"{len(dep)}/{need} 上场已记, 其余出击已发, 等标记消失确认")
        want = None
        for L in remaining:
            for sp in under:
                if self._mt_start_letter(mapd, sp, origin, dx, dy) == L:
                    want = (L, sp)
                    break
            if want:
                break
        if want is None and under:
            sp = under[0]
            L = self._mt_start_letter(mapd, sp, origin, dx, dy)
            if L and L not in dep:
                want = (L, sp)
        if want is None:
            # 标记(543/501)检不出但那个起点本身就在屏内(09-05 replay: v21 对倒三角时有时无) -> 等 8 帧, 还没标记就
            #    直接点它: 它不在 mt_deployed 里, 上面没队(有队的起点标记才会消失, 而我们数的就是这个)。
            #    真被卡条/屏边挡住(推算位置在屏外或底部卡条带)才拖地图。
            for L in remaining:
                v = mapd["starts"].get(L)
                if not v:
                    continue
                ex, ey = px_of(tuple(v), origin, dx, dy)
                if 0.05 < ex < 0.95 and 0.10 < ey < 0.70:
                    sb_near = min(sb, key=lambda b: (b.cx - ex) ** 2 + (b.cy - ey) ** 2, default=None)
                    if sb_near is not None and (sb_near.cx - ex) ** 2 + (sb_near.cy - ey) ** 2 < (0.5 * dx) ** 2:
                        if self.bump("mt_nomark") < 8:
                            return wait(f"起点 {L} 在屏内但标记没检出, 等几帧再点(第 {self.state['mt_nomark']} 帧)")
                        self.state["mt_nomark"] = 0
                        act = tap_box(sb_near, f"点起点 {L} 上队(标记 8 帧没检出, 起点在屏内且未记上场, 第 {len(dep) + 1}/{need} 队)",
                                      expect=(V.SORTIE,),
                                      post=lambda L=L: self.state.update(mt_dep_target=L, mt_probe=L))
                        # 这是探针: 弹编队 = 真没上; 弹 解除 菜单 = 其实已上(进程重开续局), mt_note_menu 会记上场
                        act.x, act.y = sb_near.cx, sb_near.y1 + 0.30 * (sb_near.y2 - sb_near.y1)
                        return act
            # 剩下的起点没有标记可见且不在屏内: 多半被底部卡条/屏边挡住 -> 朝它的推算位置拖地图
            return self._mt_drag_to_start(mapd, remaining, origin, dx, dy, cs)
        L, sp = want
        act = tap_box(sp, f"点起点 {L} 上队(框上 1/3 处, 第 {len(dep) + 1}/{need} 队)", expect=(V.SORTIE,),
                      post=lambda L=L: self.state.update(mt_dep_target=L, mt_probe=None))
        # 归属只在这一发真发出去后才记(post): 被连发闸吞掉的决策帧也改状态 = 数意图, 09-05 live 把 A 起点上的队记成了 B
        act.x, act.y = sp.cx, sp.y1 + 0.30 * (sp.y2 - sp.y1)
        return act

    def mt_note_menu(self, obs: Observation) -> None:
        """do_grid 在处理 部署菜单_解除 之前调: 刚探针点过的起点弹出了这个菜单 = 那格已经有队(续局), 记上场。"""
        L = self.state.get("mt_probe")
        if L and obs.has(V.GRID_UNIT_UNDEPLOY, 0.40):
            dep = self.state.setdefault("mt_deployed", [])
            if L not in dep:
                dep.append(L)
                self.log(f"起点 {L} 点开了 解除 菜单 = 已有队在上面(续局), 记为已上场({len(dep)})")
            self.state["mt_probe"] = None
            self.state["mt_dep_target"] = None

    def _mt_marks(self, obs: Observation) -> List[Box]:
        return obs.all(DEPLOY_MARKS, 0.30)

    def _mt_start_letter(self, mapd, sp: Box, origin, dx, dy) -> Optional[str]:
        l = lat_of((sp.cx, sp.cy), origin, dx, dy)
        for L, v in mapd["starts"].items():
            if tuple(v) == l:
                return L
        # 起点文字框心比格心略偏, 容 1 格内最近
        best = None
        for L, v in mapd["starts"].items():
            d = abs(v[0] - l[0]) + abs(v[1] - l[1])
            if d <= 2 and (best is None or d < best[0]):
                best = (d, L)
        return best[1] if best else None

    def _mt_drag_to_start(self, mapd, remaining, origin, dx, dy, cs) -> Action:
        """未上队起点的标记看不见(被底部卡条/屏边挡, 11-1 实锤): 把地图朝屏中拖, 拖动几何全部从检出推:
        手柄 = 离屏中最近的检出格心(拖真格子, 不拖 HUD), 位移 = 把起点推算位置拉到屏中的向量(封顶 DRAG)。"""
        n = self.bump("mt_drags")
        if n > 4:
            return self.finish(Outcome.UNKNOWN, f"拖了 4 次地图仍找不到未上队起点 {remaining} 的标记 -- 交人看")
        tx, ty = None, None
        for L in remaining:
            v = mapd["starts"].get(L)
            if v:
                tx, ty = px_of(tuple(v), origin, dx, dy)
                break
        if not cs:
            return wait("没有检出格子可当拖动手柄, 等一帧")
        mid_x = sum(c[0] for c in cs) / len(cs)
        mid_y = sum(c[1] for c in cs) / len(cs)
        handle = min(cs, key=lambda c: (c[0] - mid_x) ** 2 + (c[1] - mid_y) ** 2)
        if tx is None:
            ddx, ddy = 0.0, -DRAG            # 位置未知: 先往上拖(卡条在底部是最常见的遮挡)
        else:
            ddx = max(-DRAG, min(DRAG, mid_x - tx))
            ddy = max(-DRAG, min(DRAG, mid_y - ty))
            if abs(ddx) < 0.05 and abs(ddy) < 0.05:
                ddx, ddy = 0.0, -DRAG
        return swipe(handle[0], handle[1], handle[0] + ddx, handle[1] + ddy,
                     f"起点 {remaining[0]} 的标记检不出(推算在 {tx if tx is None else round(tx, 2)},"
                     f"{ty if ty is None else round(ty, 2)}) -- 拖地图 ({ddx:+.2f},{ddy:+.2f})(第 {n} 次)")

    def _mt_formation(self, obs: Observation) -> Optional[Action]:
        """编队面板: grid_presets 配置了属性->預設(页签,行)就先给当前部队套預設; grid_squads 配置了属性->部队号
        就先切到那支部队; 然后出击。"""
        pre = self._preset_before_sortie(obs)
        if pre is not None:
            return pre
        L = self.state.get("mt_dep_target")
        hi = None
        for n, (tab, hi_cls) in V.SQUAD_TABS.items():
            if obs.has(hi_cls, 0.45):
                hi = n
                break
        pr = self._mt_preset_step(obs, L, hi)
        if pr is not None:
            return pr
        want = self._mt_want_squad(L)
        if want and hi is not None and hi != want:
            k = self.bump(f"mt_sq:{L}")
            if k <= 3:
                tab = obs.find(V.SQUAD_TABS[want][0], 0.45)
                if tab is not None:
                    return tap_box(tab, f"队 {L} 按配置用部队{want}(当前高亮部队{hi})", expect=(V.SQUAD_TABS[want][1],))
            elif k == 4:
                self.log(f"部队{want} 切不过去(可能已上场/页签检不出), 队 {L} 就用当前高亮的部队{hi}")
        s = obs.find(V.SORTIE, 0.45)
        if s is None:
            return wait("编队页, 等出击键")
        if L is None:
            return tap_box(s, "编队确认: 出击(多队, 起点归属未知)", expect=(V.TASK_START, V.TASK_START_GREY))

        def _post(L=L, hi=hi):
            self.state["mt_dep_pending"] = L
            self.state["mt_form_seen_t"] = 0
            self.state["mt_pr_entry_wait"] = 0
            if hi is not None:
                self.state.setdefault("mt_team_squad", {})[L] = hi
        return tap_box(s, f"编队确认: 出击(队 {L} = 部队{hi or '?'})",
                       expect=(V.TASK_START, V.TASK_START_GREY), post=_post)

    def _mt_preset_step(self, obs: Observation, L: Optional[str], hi: Optional[int]) -> Optional[Action]:
        """按答案属性给**当前高亮部队**套預設(用户 09-05: 預設栏目 2 的三行 = 红/黄/紫蓝三队)。
        cfg campaign.grid_presets = {"red": [2, 1], "yellow": [2, 2], "blue": [2, 3], "purple": [2, 3]}; attr any / 没配 -> 不套。
        走 PresetMixin 子链(預設入口 -> 页签 -> 行 組成 -> 變更編輯 確認 -> 叉掉面板); 每支队只套一次(mt_preset_done)。
        多队关按属性配队是用户明确要的, 这里不受"部队1 不许覆盖"限制(那条是日常单队跑的保护)。"""
        cfgp = self.cfg.get("grid_presets") or None
        if not isinstance(cfgp, dict) or L is None:
            return None
        attr = "any"
        for t in self._mt_teams():
            if t["name"] == L:
                attr = t.get("attr") or "any"
        spec = cfgp.get(attr)
        if not spec:
            return None
        try:
            tab, row = int(spec[0]), int(spec[1])
        except (TypeError, ValueError, IndexError):
            self.log(f"grid_presets[{attr}] 配置不合法: {spec!r}, 不套")
            return None
        if not (1 <= tab <= 4 and 1 <= row <= 4):
            self.log(f"grid_presets[{attr}] 越界: {spec!r}, 不套")
            return None
        done = self.state.setdefault("mt_preset_done", {})
        if done.get(L):
            return None
        if done.get(f"{L}:giveup"):
            return None
        # 面板刚滑入时点右栏图标会被吞(09-05 live 第 8 次: 几何点了一下, once 卡死 4001 tick): 面板出现后先等 FORM_SETTLE_S
        if not self.state.get("mt_form_seen_t"):
            self.state["mt_form_seen_t"] = time.time()
            return wait(f"队 {L}: 编队面板刚出现, 等 {FORM_SETTLE_S:.1f}s 再套預設")
        if time.time() - float(self.state["mt_form_seen_t"]) < FORM_SETTLE_S:
            return wait(f"队 {L}: 等编队面板停稳再套預設")
        if not self.state.get("preset_want") and not self.state.get("preset_applied"):
            if hi is None:
                return wait(f"队 {L}: 等编队面板部队高亮出来再套預設")
            self.preset_start(tab, row)
            self.log(f"队 {L}({attr}) 用部队{hi}, 套預設 页签{tab} 第{row}行")
        panel = obs.has(V.PRESET_TITLE, 0.40)
        # 開面板那一发有界重试: 点过 pr_open 但 PR_OPEN_RETRY_S 内没见 预设标题 -> 清 once 再点(最多 3 次), 3 次都吞就放弃套預設
        t_open = float(self.state.get("mt_pr_open_t", 0) or 0)
        # 變更編輯 确认框盖住面板标题时 panel=False, 不是面板没开(09-05 10-3 live: 每支队都多打一行"面板没开重点"并清了 once)
        if (not panel and t_open and time.time() - t_open > PR_OPEN_RETRY_S and not self.pending("pr_open")
                and not obs.has(V.PRESET_CHANGE_TITLE, 0.40) and not self.state.get("preset_applied")):
            k = self.bump(f"mt_pr_open_n:{L}")
            if k >= 3:
                self.log(f"队 {L}: 預設入口点了 {k} 次面板都没开 -- 放弃套預設, 用当前部队出击(要人工检查 預設入口 检出/版式)")
                done[f"{L}:giveup"] = True
                self.state.pop("preset_want", None)
                return None
            self.log(f"队 {L}: 預設入口点了 {PR_OPEN_RETRY_S:.0f}s 面板没开 -- 重点(第 {k + 1} 次)")
            self.once_reset("pr_open")
            self.state["mt_pr_open_t"] = 0
        act = self.preset_step(obs)
        if (act is not None and act.kind == "tap" and act.once_key == "pr_open"):
            act.post = lambda: self.state.update(mt_pr_open_t=time.time())
        if (act is not None and act.kind == "wait" and "等入口键" in act.reason
                and not panel and obs.has(V.SORTIE, 0.45)):
            # 部署侧编队面板右栏第 4 个图标就是 預設, v21 在这一版式上常检不出(09-05 live 3 帧 0 检出, 单队 09-03 时 0.88);
            #    面板是全屏固定版式(快速編輯/起始技能/部隊資訊/預設 竖排), 等 6 帧没检出就按几何点, 点完只认 预设标题 出现。
            n = self.bump("mt_pr_entry_wait")
            if n >= 6 and self.pending("pr_open"):
                a = tap_at(PRESET_ENTRY_XY[0], PRESET_ENTRY_XY[1],
                           f"預設入口 6 帧没检出, 按编队面板右栏几何点開面板(队 {L})",
                           justify="编队面板右栏四个图标竖排固定(16:9 归一化 預設 在 (0.937,0.672), 08-31/09-03 金标与 09-05 实帧一致); "
                                   "锚 出击 在场才点; 点空只是面板没开, 下一帧再来; 开没开只认 预设标题",
                           require=V.SORTIE, once="pr_open")
                a.post = lambda: self.state.update(mt_pr_open_t=time.time())
                return a
        if act is not None:
            return act
        if self.preset_done():
            done[L] = {"tab": tab, "row": row}
            self.state.pop("preset_applied", None)
            self.log(f"队 {L} 預設已套用, 出击")
            return None
        return wait(f"队 {L} 套預設中")

    def _mt_want_squad(self, L: Optional[str]) -> Optional[int]:
        cfgm = self.cfg.get("grid_squads") or None
        if not isinstance(cfgm, dict) or L is None:
            return None
        attr = "any"
        for t in self._mt_teams():
            if t["name"] == L:
                attr = t.get("attr") or "any"
        v = cfgm.get(attr, cfgm.get("any"))
        try:
            v = int(v)
        except (TypeError, ValueError):
            return None
        return v if 1 <= v <= 4 else None

    def _mt_start_walk(self) -> None:
        mapd = self.state.get("mt_map") or {"starts": {}}
        self.state["mt_pos"] = {L: list(v) for L, v in mapd["starts"].items()}
        self.mt_new_round()

    #  回合

    def _mt_focus(self, obs: Observation, cs, dx, dy, origin) -> Tuple[Optional[str], Optional[Tuple[int, int]]]:
        """箭头 -> 正下方格 -> 点阵 -> 是哪支队。(队名或 None, 箭头格点阵坐标或 None)"""
        # 箭头阈值 0.25: 10-3 实帧箭头只有 0.28-0.53(浅底 + 立绘遮), 0.30 会时有时无 -> 焦点 None 25s 交人。
        #    两帧共识兜误检。
        arrow = obs.find(V.GRID_ARROW, 0.25)
        if arrow is None or origin is None:
            return None, None
        # 队伍脚下的格常被立绘挡住检不出(10-3 起点 B 上站着人, 格/起点都没检出), below() 会就近绑到别的格。
        #    四关实测箭头心到脚下格心 = 1.50-1.57 行距, 用这个几何直接投到点阵; 检出格与投影一致才用检出格。
        pos = self.state.get("mt_pos") or {}
        # 先按**已知队伍位置**匹配: 箭头挂在单位头顶, 头顶高度随角色变(10-3 实测 1.0-1.6 行), 自由投影会把
        #    邻格的队认错(把站在 (2,0) 的 B 投到 (1,1), 触发假"已到目标")。同列(|dx|<0.5 格距)且箭头在格心上方
        #    0.6-2.0 行的队里取 x 最近的。
        cands = []
        for L, v in pos.items():
            if v is None:
                continue
            ex, ey = px_of(tuple(v), origin, dx, dy)
            if abs(ex - arrow.cx) <= 0.5 * dx and 0.6 * dy <= (ey - arrow.cy) <= 2.0 * dy:
                cands.append((abs(ex - arrow.cx), L, tuple(v)))
        if cands:
            cands.sort()
            return cands[0][1], cands[0][2]
        # 没有已知队在箭头下面: 自由投影(传送后位置未知的队靠这个绑回)
        est = lat_of((arrow.cx, arrow.cy + 1.3 * dy), origin, dx, dy)
        cell = grid.below(arrow, cs, dx)
        l = est
        if cell is not None:
            dl = lat_of(cell, origin, dx, dy)
            if abs(dl[0] - est[0]) + abs(dl[1] - est[1]) <= 1:
                l = dl
        unknown = [L for L, v in pos.items() if v is None]
        if len(unknown) == 1:
            pos[unknown[0]] = list(l)
            self.log(f"队 {unknown[0]}(传送后位置未知) 按箭头绑回 {l}")
            return unknown[0], l
        return None, l

    def mt_walk_step(self, obs: Observation, st, plan, cs, dx, dy) -> Action:
        """do_walk 在我方回合(PHASE 在场, 未 issued)时进来: 逐个执行本回合的动作。"""
        acts = list(plan[self.state["round_i"]])
        ai = int(self.state.get("mt_ai", 0))
        fr = self._mt_frame(obs)
        origin = fr[4] if fr else None
        if origin is None:
            if self._overdue("mt_walk_align", 60):
                return self.finish(Outcome.UNKNOWN, "回合中 60s 点阵对不齐(起点/格子检出不足) -- 不瞎点")
            return wait("点阵对齐中")
        self._wt_clear("mt_walk_align")
        # 相机两帧共识: 开局/每次行动后地图会平移 1-2s, 平移中按上一帧算的落点会打到格界(09-05 第 6 次 live:
        #    首发落在两格缝上没走, 10s 后才重发)。原点比上一帧漂 > 0.01 就只观察不落子。
        lo = self.state.get("mt_last_origin")
        self.state["mt_last_origin"] = [origin[0], origin[1]]
        still = (lo is not None and abs(lo[0] - origin[0]) < 0.006 and abs(lo[1] - origin[1]) < 0.006)
        sn = (int(self.state.get("mt_still_n", 0)) + 1) if still else 0
        self.state["mt_still_n"] = sn
        camera_still = sn >= 4        # 缓动收尾极慢, 两帧 0.01 内会误判停稳(10-3/10-4 实锤首发被吞), 要连续 4 帧
        focus, focus_lat = self._mt_focus(obs, cs, dx, dy, origin)
        pos: Dict[str, Optional[list]] = self.state.setdefault("mt_pos", {})
        idle_ok = self._mt_idle_ok(obs)
        pend = self.state.get("mt_pending")
        # 箭头压在白发/浅色立绘上时 v21 整段检不出(10-3 第 4 跑 25s 全 None). 游戏空闲、没有挂起动作、本回合只剩一队
        #    没行动 -> 按游戏"自动切给下一支没行动的队"的规则, 焦点就是它。
        if (focus is None and idle_ok and not pend and obs.find(V.GRID_ARROW, 0.25) is None):
            acted = {acts[i].get("team") for i in range(min(ai, len(acts)))}
            rest = [L for L, v in pos.items() if v is not None and L not in acted]
            if len(rest) == 1:
                focus, focus_lat = rest[0], tuple(pos[rest[0]])
                if self.bump("mt_focus_infer") % 40 == 1:
                    self.log(f"箭头没检出(立绘遮挡?), 本回合只剩队 {focus} 没行动, 按自动切队规则视焦点为它")
        if focus is not None:
            self._wt_clear("mt_no_arrow")      # 看见箭头就清, 不管这一帧走哪个分支(10-3 实锤: 挂着换位证据时计时器没清, 一转身就交人)
        if not idle_ok:
            if pend:
                pend["t"] = max(float(pend.get("t", 0)), time.time() - 5.0)   # 动画期不计超时(最多回拨 5s)
            return wait("游戏在放动画/加载(PHASE 弱或加载中), 不落子不判证据")
        # 上一发的事后证据
        if pend:
            o0 = pend.get("origin0")
            moved = (o0 is None) or (abs(origin[0] - o0[0]) + abs(origin[1] - o0[1]) >= CAM_MOVE_MIN)
            done = self._mt_pending_done(pend, focus, focus_lat, moved)
            if done:
                self._mt_dbg(obs, f"evidence {pend['team']} {pend['do']} focus={focus}@{focus_lat}")
                self._mt_apply(pend)
                self.state["mt_pending"] = None
                ai = int(pend["ai"]) + 1
                self.state["mt_ai"] = ai
                # 动作被消费后游戏还要放完移动动画、自己把焦点切给下一队、相机再平移(实测 1-2s);
                #    这段时间里读到的焦点是过渡态, 拿它去点药丸会把游戏刚切好的焦点又翻回去(09-05 第 7 次 live)。
                self.state["mt_settle_until"] = time.time() + 4.0
                if ai >= len(acts):
                    return self._mt_round_issued(acts)
                return wait(f"动作 {ai}/{len(acts)} 已确认, 下一动作(先等 4s 让游戏切焦点/相机停稳)")
            if time.time() - float(pend.get("t", 0)) > (10.0 if pend["do"] != "exchange" else 6.0):
                n = self.bump(f"mt_reissue:{self.state['round_i']}:{pend['ai']}")
                if n > 3:
                    return self.finish(Outcome.UNKNOWN,
                                       f"回合 {self.state['round_i'] + 1} 动作 {pend['ai'] + 1}({pend['team']} {pend['do']} {pend.get('dir')}) 重发 3 次都没证据 -- 交人看")
                self.log(f"动作 {pend['ai'] + 1} 超时无证据, 重发(第 {n} 次)")
                self.state["mt_pending"] = None
                if pend["do"] == "exchange":
                    self.state["mt_ex_stage"] = 0
            else:
                if pend["do"] == "exchange" and pend.get("stage") == 1:
                    return self._mt_exchange_menu(obs, pend)
                return wait(f"等动作 {pend['ai'] + 1} 的事后证据(焦点/相位)")
        if ai >= len(acts):
            return self._mt_round_issued(acts)
        if not camera_still:
            return wait("相机在平移(原点两帧不一致), 等它停稳再落子")
        if time.time() < float(self.state.get("mt_settle_until", 0)):
            return wait("等游戏放完动画/自己切焦点(2.5s 静默期)")
        act = acts[ai]
        team = act.get("team")
        do = act.get("do", "move")
        d = act.get("dir")
        if team not in pos:
            return self.finish(Outcome.UNKNOWN, f"答案里的队 {team} 不在部署记录 {list(pos)} 里 -- 不瞎点")
        # 无箭头时钟: 只要这一帧焦点认出来了(不管是不是要动的那支队)就清, 不能只在"焦点在别队"分支里清
        #    (09-05 10-3 实锤: 开局 STAGE START 横幅 20 多秒焦点 None 起了钟, 第 1 发焦点==队 走了"相等"分支没清,
        #     第 2 发焦点一时没认出就直接判 25s 无箭头收工).
        if focus is not None:
            self._wt_clear("mt_no_arrow")
        # 焦点读数两帧共识(箭头在切换/平移中会闪)
        fprev = self.state.get("mt_focus_prev")
        self.state["mt_focus_prev"] = focus
        if focus != fprev:
            return wait(f"焦点读数 {fprev} -> {focus}, 等下一帧共识")
        # 先把焦点切到这支队。**只在明确看见箭头在别的队头上时才点药丸**: 箭头没检出(回合开始横幅/动画)
        #    时点药丸会把默认焦点翻走(09-05 第 7 次 live 开局就翻了两下)。次数按真发出去的药丸计(post), 不按帧计。
        if focus != team:
            key = f"mt_pill:{self.state['round_i']}:{ai}"
            n = int(self.state.get(key, 0))
            cap = 2 * max(2, len(pos)) + 2
            if n >= cap:
                return self.finish(Outcome.UNKNOWN,
                                   f"药丸点了 {n} 次焦点仍聚不到队 {team}(现在 {focus}) -- 交人看")
            if focus is None:
                # 09-05 10-3 第 3 跑同处第二次收工: 等待期没有任何决策帧, 复盘无据. 每 40 帧和收工前各落一帧带状态.
                n = self.bump("mt_noarrow_frames")
                late = self._overdue("mt_no_arrow", 25)
                if n % 150 == 1 or late:
                    ar = obs.find(V.GRID_ARROW, 0.25)
                    self._mt_dbg(obs, f"noarrow#{n} arrow={None if ar is None else (round(ar.cx, 3), round(ar.cy, 3), round(ar.conf, 2))} "
                                      f"focus_lat={focus_lat} origin={None if origin is None else (round(origin[0], 3), round(origin[1], 3))} "
                                      f"pos={pos} pend={self.state.get('mt_pending')}")
                if late:
                    return self.finish(Outcome.UNKNOWN, "25s 没看到队伍箭头(焦点未知), 不瞎点药丸 -- 交人看")
                return wait("箭头没检出(焦点未知), 等它出现再决定要不要切队")
            self._wt_clear("mt_no_arrow")

            def _pill(key=key):
                self.state[key] = int(self.state.get(key, 0)) + 1
                self.state["mt_settle_until"] = time.time() + 5.0    # 切队后相机缓动 3-4s, 早到的点击被吞(10-3 实锤)
                self.state["mt_focus_prev"] = None
            a = tap_at(SWITCH_PILL[0], SWITCH_PILL[1],
                       f"焦点在 {focus}, 要 {team} -- 点左下切队药丸(第 {n + 1} 次)",
                       justify="左下角「N部隊」切队药丸没有 cls(v22 采了料); 16:9 部署/回合 HUD 固定, 三关 40+ 帧位置不变; "
                               "点空只是不换焦点, 下一帧箭头位置就是证据, 有界重试后 UNKNOWN",
                       require=V.PHASE_END)
            a.post = _pill
            # progress 只放**观测到的焦点**, 不放尝试次数: 09-05 live 第 6 次 run 把次数放进去, 连发闸把每一发
            #    都当"有进展", 6 tick 连点 6 下药丸, 焦点在 A/B 间来回翻, 读数永远追不上。焦点没变时交给连发闸
            #    按 retry_frames 节流(约 2-4s 一发), 正好给相机平移和箭头刷新留时间。
            a.progress = f"focus:{focus or '?'}"
            return a
        self._wt_clear("mt_no_arrow")
        cur = pos.get(team)
        if cur is None:
            return wait(f"队 {team} 位置未知(传送后), 等箭头绑回")
        cur = tuple(cur)
        if do == "move" or do == "portal":
            tgt = cur if (do == "portal" and d == "center") else lat_add(cur, d or "")
            if tgt is None:
                return self.finish(Outcome.UNKNOWN, f"答案方向 {d!r} 不认识 -- 不瞎点")
            px = self._mt_cell_px(tgt, cs, origin, dx, dy)
            if px is None:
                if self.hold("mt_no_goal", 20):
                    path = self._dump_grid_miss(obs)
                    return self.finish(Outcome.UNKNOWN,
                                       f"队 {team} {do} {d}: 目标点阵 {tgt} 处没有检出的格子也不在地图里 -- 不瞎点"
                                       + (f" 干净帧 {path}" if path else ""))
                return wait(f"队 {team} {do} {d}: 目标格暂未检出, 再看几帧")

            def _issued(ai=ai, team=team, do=do, d=d, tgt=tgt):
                self.state["mt_pending"] = {"ai": ai, "team": team, "do": do, "dir": d,
                                            "from": list(cur), "target": list(tgt), "t": time.time(),
                                            "origin0": [origin[0], origin[1]]}
                self.state["mt_ex_stage"] = 0
                if ai == len(acts) - 1:
                    self._mt_mark_issued(acts)
            # 落点压到格子下部: 单位立绘从格心向上长, 相机放大时盖住相邻格心, 点在友军身上 = 切焦点不是走(10-3 第 4 跑实锤)
            ty = px[1] + (CELL_TAP_DOWN * dy if do != "portal" else 0.0)
            a = tap_at(px[0], ty,
                       f"回合 {self.state['round_i'] + 1} 动作 {ai + 1}/{len(acts)}: 队 {team} "
                       f"{'踩传送门' if do == 'portal' else '走'} {d} -> {tgt}",
                       justify="落点 = 本帧检出的格心(或对齐可信的地图格心)向下压 0.33 行距避开立绘, 由点阵对齐得来, 不是版面常量",
                       require=V.PHASE_END)
            a.post = _issued
            self._mt_dbg(obs, f"tap move {team} {d} focus={focus}@{focus_lat} px={px[0]:.3f},{px[1]:.3f} origin={origin[0]:.3f},{origin[1]:.3f}")
            return a
        if do == "exchange":
            tgt = lat_add(cur, d or "")
            other = None
            for L, v in pos.items():
                if L != team and v is not None and tuple(v) == tgt:
                    other = L
            if tgt is None or other is None:
                return self.finish(Outcome.UNKNOWN,
                                   f"队 {team} exchange {d}: 目标 {tgt} 上没有友军(位置 {pos}) -- 答案与推算不符, 不瞎点")
            px = self._mt_cell_px(tgt, cs, origin, dx, dy)
            if px is None:
                return wait(f"队 {team} exchange: 友军 {other} 所在格未检出, 再看一帧")

            def _issued2(ai=ai, team=team, d=d, tgt=tgt, other=other, px=px):
                self.state["mt_pending"] = {"ai": ai, "team": team, "do": "exchange", "dir": d, "other": other,
                                            "from": list(cur), "target": list(tgt), "px": list(px),
                                            "stage": 1, "t": time.time()}
                if ai == len(acts) - 1:
                    self._mt_mark_issued(acts)
            a = tap_at(px[0], px[1],
                       f"回合 {self.state['round_i'] + 1} 动作 {ai + 1}/{len(acts)}: 队 {team} 与 {other} 换位, 先点友军格 {tgt}",
                       justify="点友军格弹「選擇/變更位置」菜单(10-2 实测), 落点是本帧检出格心",
                       require=V.PHASE_END)
            a.post = _issued2
            self._mt_dbg(obs, f"tap exchange1 {team}->{other} focus={focus}@{focus_lat} px={px[0]:.3f},{px[1]:.3f}")
            return a
        return self.finish(Outcome.UNKNOWN, f"答案动作 {do!r} 不认识 -- 不瞎点")

    def _mt_exchange_menu(self, obs: Observation, pend: dict) -> Action:
        """菜单已弹(上一发点了友军格): 点 變更位置。**先用像素核对菜单图标真在那**: 10-3 live 实锤, 友军格那一发没把
        菜单点出来, 第二发几何落点就成了 A 往左下走一格, 整回合白给。图标 = 白圆底 + 蓝色箭头 glyph, 圆心 7x7 里饱和蓝
        像素占比 >= 0.15 才算菜单在; 地图底色是低饱和浅蓝, 分得开。没帧(离线)时不核对。"""
        px = pend.get("px") or [0.5, 0.5]
        x, y = px[0] + MENU_EXCHANGE_DXY[0], px[1] + MENU_EXCHANGE_DXY[1]
        # 菜单弹出有 ~1s 动画, 早点会穿到底下的格子(10-3 live: 0.3s 后就点, 变成乱走一步); 手驾 2s 后点必中
        if time.time() - float(pend.get("t", 0)) < MENU_WAIT_S:
            return wait(f"友军格点了, 等 {MENU_WAIT_S:.1f}s 让菜单弹稳")
        fr = getattr(obs, "frame", None)
        seen = _menu_icon_at(fr, x, y)
        seen2 = _menu_icon_at(fr, px[0] + MENU_SELECT_DXY[0], px[1] + MENU_SELECT_DXY[1])
        if seen is True and seen2 is False:
            seen = False          # 两个图标同在才算菜单(单个蓝斑可能是爆炸特效/图标误判)
        if seen is False:
            n = self.bump(f"mt_menu_wait:{pend['ai']}")
            if n <= 6:
                return wait(f"友军格点了但菜单图标还没出现(第 {n} 帧), 不盲点 變更位置")
            # 菜单没弹出来: 回到第一步重点友军格(有界, 由外层 6s 超时重发兜底)
            self.state[f"mt_menu_wait:{pend['ai']}"] = 0
            pend["stage"] = 0
            self.state["mt_pending"] = None
            return wait("菜单 6 帧没出现, 收回换位第一步重点友军格")

        def _st2():
            pend["stage"] = 2
            pend["t"] = time.time()
            self.state["mt_pending"] = pend
        self._mt_dbg(obs, f"tap exchange2 menu at {x:.3f},{y:.3f} icon={seen}")
        a = tap_at(x, y, f"队 {pend['team']} 与 {pend['other']} 换位: 点菜单「變更位置」",
                   justify="菜单挂在被点单位左侧, 图标相对格心偏移 (-0.083,-0.016) 为 10-2 实测常量(相机不缩放); "
                           "菜单键无 cls(v22 采了料); 点空的后果只是没换位, 事后证据 = 箭头落到目标格, 超时重发有界",
                   require=V.PHASE_END)
        a.post = _st2
        return a

    def _mt_cell_px(self, tgt, cs, origin, dx, dy) -> Optional[Tuple[float, float]]:
        """目标点阵 -> 屏幕落点: 优先本帧检出格心(0.5 格内), 其次地图里有这格(对齐可信)就用推算点。"""
        ex, ey = px_of(tgt, origin, dx, dy)
        near = min(cs, key=lambda c: (c[0] - ex) ** 2 + (c[1] - ey) ** 2, default=None)
        if near is not None and (near[0] - ex) ** 2 + (near[1] - ey) ** 2 <= (0.5 * dx) ** 2:
            return near
        mapd = self.state.get("mt_map") or {"cells": []}
        if list(tgt) in [list(c) for c in mapd["cells"]] and 0.03 < ex < 0.97 and 0.08 < ey < 0.88:
            return (ex, ey)
        return None

    def _mt_pending_done(self, pend: dict, focus: Optional[str], focus_lat, moved: bool = True) -> bool:
        do = pend["do"]
        tgt = tuple(pend["target"])
        # 位置表要等这里返回 True 才更新, 所以看"箭头落在哪个格"(focus_lat), 别看按旧位置表猜出来的队名:
        #    换位后箭头在友军原来的格上, 旧表会把它认成友军; 走完后箭头在目标格上, 旧表谁也对不上。
        if do == "exchange":
            # 换位不消耗行动: 焦点仍在本队, 箭头落到友军原来的格 = 换成了
            return pend.get("stage") == 2 and focus_lat == tgt
        if do == "move" and focus_lat is not None and focus_lat == tgt:
            return True                     # 箭头已在目标格(只剩它一队时焦点不切, 它已站过去)
        if focus is not None and focus != pend["team"]:
            # 10-3 第 4 跑: 落点打在友军立绘上 -> 游戏只是切了焦点, 队没走, 却被当"行动被消费". 真走过相机必平移,
            #    move 的这条证据要伴随原点漂移(moved); 换位/传送不走这里.
            return bool(moved) if do == "move" else True
        if self.state.get("cycling"):
            return True                     # 相位循环了
        return False

    def _mt_apply(self, pend: dict) -> None:
        pos = self.state.setdefault("mt_pos", {})
        team = pend["team"]
        do = pend["do"]
        if do == "move":
            pos[team] = list(pend["target"])
        elif do == "portal":
            pos[team] = None
        elif do == "exchange":
            other = pend["other"]
            pos[team], pos[other] = list(pend["target"]), list(pend["from"])
        if do != "exchange":
            acted = self.state.setdefault("mt_acted", [])
            if team not in acted:
                acted.append(team)
        self.log(f"动作确认: 队 {team} {do} {pend.get('dir')} -> 位置 {pos}")

    def _mt_mark_issued(self, acts) -> None:
        """本回合最后一个动作发出去了: 置 issued 交给 do_walk 的相位循环时钟(最后一发之后 PHASE 立刻消失,
        事后证据来不及看, 循环本身就是证据; mt_new_round 会把还挂着的动作记账)。
        need_end = 答案本回合有队不行动(exchange 不算行动) -> 游戏不会自动结束, 之后手点 PHASE結束。"""
        actors = {m.get("team") for m in acts if m.get("do", "move") != "exchange"}
        all_teams = set((self.state.get("mt_pos") or {}).keys())
        self.state["mt_need_end"] = bool(all_teams - actors)
        self.state.update(issued=True, cycling=False, pe_absent=0, pre_vec=None,
                          moved_t=None, moved_frames=0, issued_do="multi")
        self.state["mt_issue_t"] = time.time()
        self._wt_clear()

    def _mt_round_issued(self, acts) -> Action:
        """本回合动作都有证据了(pe 仍在场): 交给 do_walk 的相位循环等待。"""
        if not self.state.get("issued"):
            self._mt_mark_issued(acts)
        return wait(f"回合 {self.state['round_i'] + 1} 的 {len(acts)} 个动作都发了, 等相位循环"
                    + ("(有队没行动, 要手点 PHASE結束)" if self.state.get("mt_need_end") else ""))

    def mt_issued_step(self, obs: Observation, pe: bool) -> Optional[Action]:
        """do_walk 的 issued 分支里先问这里(每帧): 还挂着的动作先看事后证据; 有队没行动 -> 点 PHASE結束
        (「尚未行動」确认框由 base 通用处理器点確認)。返回 None = 交回 do_walk 的通用相位循环等待。"""
        if not pe or self.state.get("cycling"):
            return None
        pend = self.state.get("mt_pending")
        if not self._mt_idle_ok(obs):
            if pend:
                pend["t"] = max(float(pend.get("t", 0)), time.time() - 5.0)
            return wait("游戏在放动画/加载, 等空闲再判最后一发的证据")
        if pend:
            fr = self._mt_frame(obs)
            if fr is not None and fr[4] is not None:
                cs, sb, dx, dy, origin = fr
                focus, focus_lat = self._mt_focus(obs, cs, dx, dy, origin)
                if pend.get("do") == "exchange" and pend.get("stage") == 1:
                    return self._mt_exchange_menu(obs, pend)
                if self._mt_pending_done(pend, focus, focus_lat):
                    self._mt_apply(pend)
                    self.state["mt_pending"] = None
                    self.state["mt_ai"] = int(pend["ai"]) + 1
                    self.state["mt_settle_until"] = time.time() + 2.5
            pend = self.state.get("mt_pending")
            if pend and time.time() - float(pend.get("t", 0)) > 10.0:
                # 最后一发没被游戏收下(09-05 第 8 次 live: B 的落子发了, 回合資訊仍是 1, flow 干等到相位上限):
                #    和非最后一发同样有界重发 -- 收回 issued, 让 mt_walk_step 按原 ai 再发一次。
                n = self.bump(f"mt_reissue:{self.state['round_i']}:{pend['ai']}")
                if n > 3:
                    return self.finish(Outcome.UNKNOWN,
                                       f"回合 {self.state['round_i'] + 1} 最后动作({pend['team']} {pend['do']} {pend.get('dir')}) 重发 3 次都没证据 -- 交人看")
                self.log(f"最后动作 {pend['ai'] + 1} 超时无证据, 收回 issued 重发(第 {n} 次)")
                self.state.update(mt_pending=None, issued=False, mt_need_end=False, cycling=False, pe_absent=0)
                self.state["mt_settle_until"] = 0
                self.state["mt_focus_prev"] = None
                self.state.pop("hold:move_wait", None)
                return wait("重发最后一个动作")
        if not self.state.get("mt_need_end"):
            # 第二路回合时钟(09-05 第 10 次 live): 相位切换的横幅帧被当「加载中」打断, do_walk 看不到 PHASE 消失,
            #    observe 的 pe_absent 永远数不到 3, 新回合开始了 flow 还在"等相位循环"。全员都行动完之后, 箭头再次
            #    出现在**部队 1(第一支上场的队)**头上 = 游戏开了新回合(同一回合里行动完的队不会再被聚焦)。
            #    最后一个行动的队就是部队 1 时分不清(它没走成也会这样), 那种情况仍走原时钟 + 超时重发。
            if (not self.state.get("mt_pending") and not self.state.get("cycling")
                    and time.time() - float(self.state.get("mt_issue_t", 0)) > 3.0):
                fr = self._mt_frame(obs)
                if fr is not None and fr[4] is not None:
                    cs, sb, dx, dy, origin = fr
                    focus, _fl = self._mt_focus(obs, cs, dx, dy, origin)
                    names = self._mt_names()
                    # 新回合游戏聚焦的是**部队 1**, 而部队 1 未必是答案里的第一支队(10-3 第 5 跑: 编队面板记着上次选的
                    #    部队 2, A 用了部队 2、B 用了部队 1; 按 names[0]=A 等箭头, 回合 2 箭头在 B 头上, 4001 tick 没等到)
                    sq = self.state.get("mt_team_squad") or {}
                    ones = [L for L, h in sq.items() if int(h or 0) == 1]
                    first = ones[0] if ones else (names[0] if names else None)
                    acted = list(self.state.get("mt_acted") or [])
                    last_actor = acted[-1] if acted else None
                    if self.bump("mt_issued_dbg") % 150 == 1:
                        self._mt_dbg(obs, f"issued-wait focus={focus} first={first} last={last_actor} acted={acted} pos={self.state.get('mt_pos')}")
                    if focus is not None and focus == first and last_actor != first:
                        k = self.bump("mt_newround_frames")
                        if k >= 2:
                            self.state["mt_newround_frames"] = 0
                            self.state["cycling"] = True
                            self.log("箭头回到部队 1 头上(全员已行动) = 新回合(相位横幅帧被打断时的第二路时钟)")
                            return wait("新回合证据: 箭头回到部队 1")
                        return wait("箭头在部队 1 头上, 再看一帧确认新回合")
                    if (focus is not None and focus == first and last_actor == first
                            and time.time() - float(self.state.get("mt_issue_t", 0)) > 8.0):
                        # 10-3 第 8 跑回合 2: 最后行动的就是部队 1, 箭头留在它头上, "箭头回到部队 1"分不出新回合.
                        #    最后一发发出 8s 以上且空闲(前面已过空闲闸) = 自动结束+敌方回合早已过去, 判新回合(第三路时钟).
                        if self.bump("mt_newround_same") >= 3:
                            self.state["mt_newround_same"] = 0
                            self.state["cycling"] = True
                            self.log("部队 1 收尾且箭头留在它头上, 最后一发后已空闲 8s 以上 = 新回合(第三路时钟)")
                            return wait("新回合证据: 部队 1 收尾 + 空闲 8s")
                        return wait("部队 1 收尾, 空闲计数中确认新回合")
                    self.state["mt_newround_frames"] = 0
            return None
        if self.state.get("mt_pending"):
            return wait("最后一个动作还没看到事后证据, 先不手点 PHASE結束")
        if time.time() - float(self.state.get("mt_issue_t", 0)) < 4.0:
            return wait("答案本回合有队不动, 4s 内相位没自动结束就手点 PHASE結束")
        n = self.bump("mt_end_taps")
        if n > 4:
            return self.finish(Outcome.UNKNOWN, "PHASE結束 点了 4 次相位仍不循环 -- 交人看")
        b = obs.find(V.PHASE_END, 0.40)
        if b is None:
            return None
        return tap_box(b, f"本回合答案不含全部队伍, 手点 PHASE結束(第 {n} 次; 「尚未行動」框由通用确认处理)",
                       expect_gone=(V.PHASE_END,))
