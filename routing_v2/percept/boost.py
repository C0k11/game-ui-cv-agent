# -*- coding: utf-8 -*-
"""双三倍加成归属 —— 把 cls452「双倍或三倍活动进行中」翻译成**哪个玩法**在加成。

## 为什么以前说它"只说有、不说哪类"

memory 里这条结论挂了很久，全仓零使用（老代码只有 `brain/skills/special_sweep.py`
拿它做过一次"落在特殊任务上吗"的判断，半径 0.13 是拍脑袋的）。
2026-08-11 看图才发现结论下早了：**452 检出的不是一个全局横幅，而是任务大厅
每个磁贴左上角那枚红色「活動進行中」小标** —— 人眼看
`data/raw_images/v2step_20260811/013124_000162_task_hall.png` 就一目了然。
 **它的位置就指明了是哪个磁贴**，配合同屏的磁贴 cls 就能说清是哪一类。

    ┌ 活動進行中 ┐           452 的框（红标）
    │  懸賞通緝  │           V.HUB_BOUNTY 的框（只框住标题文字，不是整张卡）
    └────────────┘

## 本模块只做感知，不碰策略

刷什么、往哪投 AP 是**用户的决定**（memory `routing_v2_live_day3` §「策略是用户的」
—— 我 08-10 擅自改 arena 配对被当场叫停）。这里只回答"屏上写着哪几类有加成"，
**不给任何 flow 改刷取顺序**。上层要不要用、怎么用，交配置和用户。

## 归属半径怎么标出来的（实测，不是"我觉得 0.1 差不多"）

memory `template_label_assets` 记着五个自造几何判据全废的教训，共同点是
**我发明像素统计判据然后当因果**。所以这里的每个数都挂了人眼真值：

真值来源（看图确认，两天天然不同源  满足 `val_set_crisis` 的「单位是天」）:
  * `data/raw_images/v2step_20260811/*task_hall*.png` (25 帧) —— 任務 / 懸賞通緝 /
    特殊任務 三个磁贴挂标；學園交流會 / 戰術大賽 / 劇情 没挂
  * `data/raw_images/v2step_20260810/*task_hall*.png` (27 帧) —— 只有學園交流會挂
  （标定快照 2026-08-11；这两个目录是 step 模式的飞轮录帧，**还在长**，
    重跑 `_selftest()` 时帧数会比这里大，结论不受影响。）

这 52 帧上 452 共 **89** 个检出（conf 压到 0.02，落在 47 帧上），
**"最近磁贴 == 人眼真值" 89/89**：

  | 量                       | 真归属 (n=89)        | 同帧其它磁贴 (n=445) |
  |--------------------------|----------------------|----------------------|
  | 中心距 dc                | 0.0373 ~ 0.0543      | **0.0898** ~ 0.6192  |
  | dc ÷ 磁贴框高            | 0.93 ~ 1.30          | **2.50** ~ 19.85     |
  | dy = 452.cy − 磁贴.cy    | −0.0526 ~ −0.0366    | （全负 = 恒在上方）  |
  | dx = 452.cx − 磁贴.cx    | −0.0179 ~ −0.0074    | （全负 = 恒偏左）    |
  | margin = 次近距 ÷ 最近距 | 2.12 ~ 4.37          | —                    |

   真/假之间有 0.0543 / 0.0898 的**空档**，取几何中点 sqrt(·)≈**0.070** 当绝对半径；
    比例判据同理 sqrt(1.30×2.50)≈**1.80**。两条**都要过**（见下面为什么要两条）。

为什么不只用绝对半径：归一化坐标下的欧氏距离是**依赖宽高比**的
（memory `read_layer_icon_units`：屏幕比例只在标定它的那个分辨率上成立，
语料里实测过 19 种分辨率）。上表全部标定自 2560×1440。
`dc ÷ 磁贴框高` 是**布局常数**（磁贴自身尺寸当单位，同 read.py 的图标单位铁律），
换分辨率/宽高比时它比绝对值稳。两条并联 = 一条失准另一条兜着。

## 边界：这个几何**只对任务大厅磁贴成立**，别往别处套

同样在 08-10/08-11 语料里扫（**全部 1003 帧**，conf≥0.20），452 还出现在：
  lobby 83 次 (0.931,0.792)+(0.180,0.865) / 悬赏分支页 8 / 悬赏关卡列表 9 /
  学院选择页 6 / 课程表区域页 5 / 组合包 1 / 关卡弹窗 1 …
其中悬赏分支页那三个 452 在 cx≈0.555，而分支 cls（高架公路等）在 cx≈0.904 ——
**中心距 0.35，布局完全是另一套**（红标在宽行卡左端，名字标签在右端）。
 本模块只认 `V.HUB_TILES`；换一套锚点必须重新标定。屏上没有磁贴时**不表态**
（返回空 + `in_task_hall=False`），绝不"就近凑一个"—— §A1 认不出就什么都不做。
这 1003 帧跑 `read_boosts()`：归属命中 74 次，**全部落在 task_hall 且磁贴全对，
  非 task_hall 页面零归属**；其余 113 个 452 一律进 `orphans`（如实说"说不清"）。

## 已知感知缺口（诚实记账，别当成 bug 去修代码）

452 本身数据很足（`ui_v2` 数据集实测 **train 6219 框 / 4900 帧，val 1319 框 / 900 帧**），
但 **conf 波动极大**：08-11 任務/懸賞 稳在 0.95，特殊任務 0.53~0.89，
而 08-10 學園交流會 那枚只有 **0.10~0.51**（中位 0.19）。
detect 层 ui 的地板就是 0.20  弱的那枚**多数帧根本进不了 Observation**。
实测单帧命中率（分母 = `in_task_hall` 的帧）:
  08-11 任務 20/22 · 懸賞 20/22 · 特殊 21/22 = 0.91~0.96
  08-10 學園交流會 **13/27 = 0.481**    就是它拖低了整体
 `hit.weak` 标出 conf<0.45 的，调用方自己决定信不信；
  多帧要结论请用 `consensus()`，别拿单帧当真相（README §A3 / 病根「单帧当真相」）；
  真正的修法是**补样本**（08-10 那 27 帧已在飞轮里），不是在这里加 OCR 兜底
    —— 「YOLO 看不见就补数据，不加 OCR 把问题藏起来」是 read.py 开篇那条铁律。

## 用法

    from routing_v2.percept import boost
    rep = boost.read_boosts(obs)
    log(rep.describe())                 # 「加成: 悬赏通缉(0.97) / 特殊任务(0.86)」
    if rep.has("bounty"): ...           # flow 名或磁贴 cls 都能问
    boost.consensus(reps_of_many_frames)  # 多帧投票，返回 {磁贴: 命中帧数}

自检: `python -m routing_v2.percept.boost`（对上面两天的实测帧复算，见 `_selftest`）
"""
from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from routing_v2.percept.observe import Box, Observation
from routing_v2.state import vocab as V

# ── 标定常量（上面表格的落地值，改之前先重跑 `python -m routing_v2.percept.boost`）──
MAX_DIST = 0.070          # 中心距绝对上限：真 ≤0.0543 / 假 ≥0.0898，取几何中点
MAX_DIST_TILE_H = 1.80    # 中心距 ÷ 磁贴框高：真 ≤1.30 / 假 ≥2.50，取几何中点
BADGE_CONF = 0.20         # 452 的下限 = detect 层 ui 地板（再低也进不了 Observation）
TILE_CONF = 0.45          # 磁贴 cls 的下限；实测 6 个磁贴恒在 0.977~0.994，很稳
WEAK_CONF = 0.45          # 低于此值只当"疑似"，`hit.weak=True`

# 磁贴  routing_v2 flow 名（`flow/registry.py` 的 key）。
# `batch_sweep` / `special_sweep` 在 registry 里是 **PLANNED（有开关没实现）**，
#   所以"任務/特殊任務 在加成"这个事实目前**没有 flow 能消费** —— 这正是
#   memory `mainline_ap_routing` 那条「全仓没有『无活动自动切回推图/扫荡』分支」。
#   这里如实映射，不假装它能跑。
TILE_FLOW: Dict[str, str] = {
    V.HUB_CAMPAIGN: "batch_sweep",      # PLANNED，未实现
    V.HUB_STORY:    "story_mining",
    V.HUB_BOUNTY:   "bounty",
    V.HUB_SPECIAL:  "special_sweep",    # PLANNED，未实现
    V.HUB_JFD:      "jfd",
    V.HUB_ARENA:    "arena",
}
# 声明依赖（死判据在 import 期就该被打回来，见 vocab.require 的注释）。
# 452 不在 HEALTH 表里，require 对它是 no-op；写在这里是为了让"我靠谁做判断"可 grep。
_BADGE = V.require(V.EVENT_MULTIPLIER, sole_signal=True)


@dataclass(frozen=True)
class BoostHit:
    """一枚 452 归属到了一个磁贴。带上全部中间量，方便逐帧人审时对账。"""
    tile: str                 # 磁贴 cls 名（V.HUB_* 之一）
    flow: str                 # 对应的 flow 名；PLANNED 的也照给，见 TILE_FLOW
    conf: float               # 452 的 conf
    tile_conf: float          # 磁贴 cls 的 conf
    dist: float               # 中心距（归一化）
    dist_ratio: float         # 中心距 ÷ 磁贴框高
    dx: float                 # 452.cx − 磁贴.cx（实测恒负）
    dy: float                 # 452.cy − 磁贴.cy（实测恒负 = 红标在标题上方）
    margin: float             # 次近磁贴距离 ÷ 本磁贴距离；越大越没歧义（实测 ≥2.2）
    badge: Box
    tile_box: Box

    @property
    def weak(self) -> bool:
        """conf 低于 WEAK_CONF —— 位置是对的，但这一帧的证据弱，别单帧下结论。"""
        return self.conf < WEAK_CONF

    def __repr__(self) -> str:
        return (f"<boost {self.tile}({self.flow}) conf={self.conf:.2f}"
                f"{'弱' if self.weak else ''} d={self.dist:.4f}"
                f"/{self.dist_ratio:.2f}h margin={self.margin:.1f}x>")


@dataclass(frozen=True)
class BoostOrphan:
    """检出了 452 但**说不清是谁的** —— 超半径 / 方向不对 / 屏上没有磁贴。

    它不是噪声，是"有加成但我认不出"，必须让调用方看见：
       典型成因是那个磁贴自己漏检了（比如加载中），这时最近磁贴会是隔壁那个，
       距离 ~0.09 被半径挡掉 —— 挡对了，但事实是"这里确实有加成"。
    """
    badge: Box
    nearest: Optional[str]    # 最近的磁贴 cls（可能 None = 屏上一个磁贴都没有）
    dist: Optional[float]
    why: str                  # 中文原因，直接进日志


@dataclass(frozen=True)
class BoostReport:
    """一帧的加成读数。`hits` 已按屏上从上到下排序。"""
    hits: Tuple[BoostHit, ...] = ()
    orphans: Tuple[BoostOrphan, ...] = ()
    tiles_seen: Tuple[str, ...] = ()      # 这一帧认出来的磁贴（按 cy 从上到下）
    in_task_hall: bool = False            # 屏上有磁贴 = 这帧能说话

    @property
    def tiles(self) -> List[str]:
        return [h.tile for h in self.hits]

    @property
    def flows(self) -> List[str]:
        return [h.flow for h in self.hits]

    def has(self, flow_or_tile: str) -> bool:
        return any(h.flow == flow_or_tile or h.tile == flow_or_tile
                   for h in self.hits)

    def get(self, flow_or_tile: str) -> Optional[BoostHit]:
        for h in self.hits:
            if h.flow == flow_or_tile or h.tile == flow_or_tile:
                return h
        return None

    def describe(self) -> str:
        """一行人话，直接丢日志。"""
        if not self.in_task_hall:
            return "加成: 不在任务大厅（屏上没有磁贴）— 不表态"
        if not self.hits and not self.orphans:
            return f"加成: 无（认出 {len(self.tiles_seen)} 个磁贴，均无「活動進行中」）"
        parts = [f"{h.tile}{'?' if h.weak else ''}({h.conf:.2f})" for h in self.hits]
        tail = f" +{len(self.orphans)} 个说不清" if self.orphans else ""
        return f"加成: {' / '.join(parts) or '无'}{tail}"

    def __bool__(self) -> bool:
        return bool(self.hits)


def read_boosts(obs: Observation, *,
                badge_conf: float = BADGE_CONF,
                tile_conf: float = TILE_CONF,
                tiles: Optional[Sequence[str]] = None) -> BoostReport:
    """任务大厅这一帧：哪几个磁贴挂着「活動進行中」。**纯函数，不点任何东西。**

    `tiles` 默认 `V.HUB_TILES`。传别的锚点集合前先重标半径 —— 上面文档里
    悬赏分支页那个反例说明**换一套页面，几何整个变**。

    读不出就返回空报告（`in_task_hall=False`），绝不猜。
    """
    names = list(tiles) if tiles else list(V.HUB_TILES)

    #  磁贴：每个 cls 只留 conf 最高那个。
    #    实测踩到过：012241_000017 那帧 YOLO 给了**两个**「悬赏通缉」框，其中一个
    #      正好压在红标上（中心距 0.0004）—— 不去重的话 margin 会被这个 DUP 毁掉。
    #      YOLO26 是 NMS-free，DUP 是它的常态（v15 验收时按 DUP/WRONG 拆过 FP）。
    tile_boxes: Dict[str, Box] = {}
    for name in names:
        b = obs.find(name, tile_conf)      # find = 同名取 argmax，天然去重
        if b is not None:
            tile_boxes[name] = b
    seen = tuple(sorted(tile_boxes, key=lambda n: tile_boxes[n].cy))

    badges = obs.all(_BADGE, badge_conf)
    if not tile_boxes:
        # 屏上没有磁贴 = 这里根本不是任务大厅（lobby / 分支页 / 关卡列表也会出 452）。
        return BoostReport(
            orphans=tuple(BoostOrphan(b, None, None, "屏上没有任务大厅磁贴")
                          for b in badges),
            in_task_hall=False)

    #  每枚 452 归到最近的磁贴，再过三道闸。
    cand: List[BoostHit] = []
    orphans: List[BoostOrphan] = []
    for m in badges:
        ranked = sorted(((hypot(m.cx - t.cx, m.cy - t.cy), t)
                         for t in tile_boxes.values()), key=lambda p: p[0])
        d1, t1 = ranked[0]
        d2 = ranked[1][0] if len(ranked) > 1 else float("inf")
        ratio = d1 / t1.h if t1.h > 0 else float("inf")
        if d1 > MAX_DIST:
            orphans.append(BoostOrphan(m, t1.cls, d1,
                                       f"离最近的「{t1.cls}」{d1:.4f} > 半径 {MAX_DIST}"
                                       f"（多半是那个磁贴自己漏检了）"))
            continue
        if ratio > MAX_DIST_TILE_H:
            orphans.append(BoostOrphan(m, t1.cls, d1,
                                       f"离「{t1.cls}」{ratio:.2f} 倍磁贴高 > "
                                       f"{MAX_DIST_TILE_H}"))
            continue
        if m.cy >= t1.cy:
            # 红标是卡片左上角的子元件，**恒在标题文字上方**（实测 dy 全负，
            # 89/89 全负）。这是布局因果不是统计巧合  出现在下方的一律不认。
            orphans.append(BoostOrphan(m, t1.cls, d1,
                                       f"在「{t1.cls}」下方（dy={m.cy - t1.cy:+.4f}），"
                                       f"红标应恒在标题上方"))
            continue
        cand.append(BoostHit(
            tile=t1.cls, flow=TILE_FLOW.get(t1.cls, ""), conf=m.conf,
            tile_conf=t1.conf, dist=d1, dist_ratio=ratio,
            dx=m.cx - t1.cx, dy=m.cy - t1.cy,
            margin=(d2 / d1) if d1 > 0 else float("inf"),
            badge=m, tile_box=t1))

    #  一个磁贴只能有一枚红标 —— 走到这里还同属一个磁贴的，**必然是同一枚的 DUP**：
    #    相邻磁贴的红标间距 ≈ 磁贴行距 0.128，而半径只有 0.070，两枚真红标不可能
    #    同时进同一个磁贴的圈。 取 conf 最高那个当代表（同一物件，用模型最好的
    #    那次估计；按距离取会把 0.95 的丢掉留下 0.22 的，`weak` 就报假了），
    #    并列再比距离。被丢掉的照样进 orphans，逐帧人审时看得见。
    best: Dict[str, BoostHit] = {}
    for h in cand:
        old = best.get(h.tile)
        if old is None or (-h.conf, h.dist) < (-old.conf, old.dist):
            if old is not None:
                orphans.append(BoostOrphan(old.badge, old.tile, old.dist,
                                           f"「{old.tile}」的重复红标（DUP），已取更强的那个"))
            best[h.tile] = h
        else:
            orphans.append(BoostOrphan(h.badge, h.tile, h.dist,
                                       f"「{h.tile}」的重复红标（DUP），已取更强的那个"))

    hits = tuple(sorted(best.values(), key=lambda h: h.tile_box.cy))
    return BoostReport(hits=hits, orphans=tuple(orphans),
                       tiles_seen=seen, in_task_hall=True)


def consensus(reports: Iterable[BoostReport], *,
              min_ratio: float = 0.25) -> Dict[str, int]:
    """多帧共识 —— 返回 {磁贴 cls: 命中帧数}，只保留占比 ≥ `min_ratio` 的。

    为什么要有这个：单帧命中率实测最低只有 **48.1%**（弱 conf 的那枚红标一半帧
      根本进不了 Observation），而"这个玩法今天有没有加成"是**持久事实**，用一帧
      下结论就是 README 病根里的「单帧当真相」。
      分母只算 `in_task_hall=True` 的帧（不在任务大厅的帧没有发言权）。

    这是**或**的语义不是**与**：某帧漏检不代表没加成，所以按帧数占比投票。
      `min_ratio=0.25` 也是标出来的，不是拍的：
        真: 08-11 三枚 20/22、20/22、21/22 = 0.91~0.96；08-10 那枚弱的 13/27 = **0.481**
        假: 全语料 1003 帧扫过，位置归属的误报 **0 次**  假阳占比 0.000
      真值下限 0.481 和 0.000 之间取一半  0.25。
      别抬到 0.5：08-10 學園交流會 那枚正好卡在 0.481，抬了就整天报"没有加成"。
    """
    votes: Dict[str, int] = {}
    total = 0
    for r in reports:
        if not r.in_task_hall:
            continue
        total += 1
        for h in r.hits:
            votes[h.tile] = votes.get(h.tile, 0) + 1
    if total == 0:
        return {}
    return {k: v for k, v in sorted(votes.items(), key=lambda kv: -kv[1])
            if v / total >= min_ratio}


# ── 自检：把上面文档里的标定重跑一遍。改常量后必须跑这个 ──────────────────
def _selftest() -> int:
    """`python -m routing_v2.percept.boost` —— 对实测帧复算归属准确率。

    它验的是"归属对不对"，真值是**人眼看图**定的（见模块文档），
      不是拿模型自己的输出当真值（memory `template_label_assets` 那五个废掉的
      自动检测器，共同点就是自己发明判据又自己当真值）。
    """
    import glob
    import os
    import cv2                                    # noqa: F401  (脚本路径才需要)
    from routing_v2.percept import detect

    truth = {"v2step_20260811": {V.HUB_CAMPAIGN, V.HUB_BOUNTY, V.HUB_SPECIAL},
             "v2step_20260810": {V.HUB_JFD}}
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "data", "raw_images")
    detect.warm(("ui",))
    ok = bad = miss = frames = 0
    for day, want in truth.items():
        reps: List[BoostReport] = []
        for f in sorted(glob.glob(os.path.join(root, day, "*task_hall*.png"))):
            im = cv2.imread(f)
            obs = Observation(boxes=detect.infer(im, ("ui",)), frame=im)
            r = read_boosts(obs)
            reps.append(r)
            frames += 1
            got = set(r.tiles)
            ok += len(got & want)
            bad += len(got - want)
            miss += len(want - got)
        con = consensus(reps)
        flag = "" if set(con) == want else ""
        print(f"{flag} {day}: 共识 {sorted(con)}  真值 {sorted(want)}  ({len(reps)} 帧)")
    print(f"逐帧: 命中 {ok} / 错归 {bad} / 漏 {miss}  （{frames} 帧；"
          f"漏的是 conf<{BADGE_CONF} 进不了 Observation 的弱红标）")
    return 0 if bad == 0 else 1


if __name__ == "__main__":                        # pragma: no cover
    raise SystemExit(_selftest())
