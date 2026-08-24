# -*- coding: utf-8 -*-
"""L2 决策层 · 页面图 (2026-07-25 建, v1 **只读不接管**)

## 这是什么
把"我在哪个画面 / 从这里怎么走到那里"从**每个 skill 各写一遍的硬编码导航**
抽出来, 变成一张图: 节点=画面(带 cls 指纹), 边=点某个 cls 会去哪。
skill 将来只需要声明目标("我要到 EventQuestList"), 路径交给 BFS。

## 数据来源(不是拍脑袋)
`data/screen_flow_draft.md` —— 2026-07-17 用 ui v13 对 **1029 张干净帧**批量
推理, 按结构 cls 集合 Jaccard≥0.65 union-find 聚成 30+ 画面簇, 组内 ≥80% 帧
出现的 cls = 核心指纹; 转移边取自 trajectory 里 click 动作落点 + 后续帧组变化。
本文件是把那份稿子**形式化成机器可读**, 并把稿子里列出的坑编码成负条件。

## v1 纪律: 只观测, 不接管
和 exit_report 一样 —— 先挂在 pipeline 上每 tick 输出"我认为在哪一页", 跟
skill 自己的 sub_state 对账, 攒够真实分布再谈让它接管导航。
**绝不在没有实测准确率之前把导航交给它。**

## 编码进来的已证实陷阱(全部来自 screen_flow_draft 实测)
1. `课程表票` 有**域外假阳**: 悬赏扫荡确认框里的票券图标被检成它  任何
   "见课程表票Schedule" 的单 cls 判据都会误判。 Schedule 族一律要求
   ≥2 个核心 cls, 且加 `取消键` 负条件。
2. **确认框族高度同构**: {确认+取消+叉叉(+MAX/MIN)} 在 扫荡确认/活动商店购买/
   AP购买/退出确认 之间 cls 集合几乎不可分  本图**拒绝**给它们单独定名,
   统一归 `ConfirmDialog_ambiguous`, 由调用方的上下文闸(base.has_qty_stepper /
   event_quest._dialog_is_purchase)处理。**图不做它做不到的判断。**
3. `弹窗叉叉`-only 帧(弹窗动画中, 实测 ~31 帧): 输出 UNKNOWN 等下一帧,
   **不硬归类**。
4. 摸头特写只有 `Emoticon_Action` 且 chrome 全隐是**合法画面**, 不是 stuck。
5. EventShop 在本期活动皮上 v13 只剩 `购买`+chrome, 无法正锚  只能反向排除,
   标记 `weak=True`, 识别结果不可用于导航决策。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

#  页面定义
# core     : 核心指纹 cls(组内 ≥80% 帧出现)
# min_core : 至少命中几个 core 才算这一页(单 cls 判据的坑见文件头 )
# require  : **必须全部出现**, 少一个直接否决。
#            为什么需要它(2026-07-25 全语料实测才发现): 光有 min_core 时,
#            "扫荡开始+任务开始" 两个通用 cls 就能凑够 Bounty_SweepPanel 的
#            门槛, **票种压根没参与判定** —— 活动关的扫荡面板被判成悬赏的,
#            实测 219 帧。凡是"靠某个专属物件区分的同构页面"(悬赏/交流会/
#            活动 三家扫荡面板长得一模一样, 只有票据不同), 那个物件必须
#            进 require 而不是 core。
# neg      : 出现任一个就**否决**这一页
# weak     : 指纹不可靠, 只可观测不可用于决策
# parent   : 返回键的默认去向(全局边, 不在 EDGES 里重复列)
P = Dict[str, Any]

PAGES: Dict[str, P] = {
    "Lobby": {
        "core": ["咖啡厅入口", "课程表入口", "学生入口", "编辑入口",
                 "任务大厅入口", "邮件箱", "每日领奖"],
        "min_core": 3, "parent": None,
    },
    "MissionHub": {
        "core": ["任务关卡推图", "剧情", "悬赏通缉", "学院交流会",
                 "战术大赛", "特殊任务"],
        "min_core": 3, "parent": "Lobby",
    },
    #  Schedule 族 (坑: 一律 ≥2 core + 取消键负条件)
    "Schedule_RegionSelect": {
        "core": ["课程表票", "夏莱办公室", "夏莱居住区", "格黑娜学院中央区",
                 "阿拜多斯高中", "千年研究所"],
        "min_core": 2, "neg": ["扫荡开始", "任务开始", "悬赏通缉票", "学院交流会票", "取消键"], "parent": "Lobby",
    },
    "Schedule_RoomCarousel": {
        "core": ["全体课程表", "左切换", "右切换", "课程表票"],
        # 2026-07-27 live 实锤: 只有 min_core=2 时, **社交页**(社團/好友/幫手
        # 三卡)靠通用的 左切换+右切换 就凑够 2  被判成课程表轮播, 而它同帧的
        # 「社团」0.99 让 Club(core 1/1) 只拿到 1 分, 打分按**绝对命中数**排序
        # 通用 chrome 打赢专属物件。这正是 memory 经验 "靠专属物件区分的同构
        # 页面, 那个物件必须进 require" 的第二次复发。
        # 两帧验证: 真轮播帧 全体课程表=0.98 在; 社交页帧无此 cls。
        # 用 require_any(OR) 不是 require(AND): 全语料实测 AND 会误伤 65 帧
        # 真课程表页(那些帧只检出 课程表票、没检出 全体课程表)。两个都是课程表
        # **域专属**, 社交页两个都不带  OR 既挡住误判又不误伤。
        "require_any": ["全体课程表", "课程表票"],
        "min_core": 2, "neg": ["扫荡开始", "任务开始", "悬赏通缉票", "学院交流会票", "取消键", "弹窗叉叉"],
        "parent": "Schedule_RegionSelect",
    },
    "Schedule_RosterPopout": {
        "core": ["弹窗叉叉", "课程表票"],
        "min_core": 2, "neg": ["扫荡开始", "任务开始", "悬赏通缉票", "学院交流会票", "取消键"], "parent": "Schedule_RoomCarousel",
    },
    "Schedule_Report": {
        "core": ["确认键", "弹窗叉叉", "课程表票"],
        "min_core": 3, "neg": ["扫荡开始", "任务开始", "悬赏通缉票", "学院交流会票", "取消键"], "parent": "Schedule_RosterPopout",
    },
    #  编队
    "Formation_Attack": {
        "core": ["1部队高亮", "快速编辑", "跳过战斗", "出击"],
        "min_core": 2, "parent": None,
    },
    #  Arena
    "Arena_Opponents": {
        "core": ["战术大赛对战选择区域", "战术大赛票"],
        "min_core": 2, "parent": "MissionHub",
    },
    "Arena_Challenge": {
        "core": ["攻击编制", "战术大赛票", "弹窗叉叉"],
        "min_core": 2, "parent": "Arena_Opponents",
    },
    "ArenaShop": {
        "core": ["战术大赛商店已选择", "下级能量饮料", "一般能量饮料",
                 "全部选择", "购买"],
        "min_core": 2, "parent": "Arena_Opponents",
    },
    #  活动 Quest 列表(2026-07-25 补: 草稿 §6 当时无帧, 现有 479 帧)
    "EventQuestList": {
        "core": ["活动quest_已选择", "入场键", "关卡得星_3",
                 "活动商店", "活动任务"],
        "min_core": 3, "neg": ["取消键"], "parent": "MissionHub",
    },
    #  Cafe (1号厅  2号厅 靠"移动至X"互斥)
    "Cafe_Hall1": {
        "core": ["咖啡厅收益", "咖啡厅邀请卷", "移动至2号点"],
        "min_core": 2, "parent": "Lobby",
    },
    "Cafe_Hall2": {
        "core": ["咖啡厅收益", "移动至一号店"],
        "min_core": 2, "neg": ["咖啡厅邀请卷"], "parent": "Lobby",
    },
    "Cafe_Invite": {
        "core": ["邀请键", "收藏图标", "弹窗叉叉"],
        "min_core": 2, "parent": "Cafe_Hall1",
    },
    #  Bounty
    "Bounty_SceneSelect": {
        "core": ["悬赏通缉票", "高架公路", "沙漠铁道", "教室"],
        "min_core": 2, "parent": "MissionHub",
    },
    "Bounty_StageList": {
        "require": ["悬赏通缉票"], "core": ["入场键", "关卡得星_3"],
        "min_core": 2, "neg": ["取消键", "扫荡开始"],
        "parent": "Bounty_SceneSelect",
    },
    "Bounty_SweepPanel": {
        "require": ["悬赏通缉票", "扫荡开始"], "core": ["任务开始", "弹窗叉叉"],
        "min_core": 1, "neg": ["取消键"], "parent": "Bounty_StageList",
    },
    #  Exchange(学院交流会)
    "Exchange_SchoolSelect": {
        "core": ["三一", "千年", "格黑娜"],
        "min_core": 2, "parent": "MissionHub",
    },
    "Exchange_StageList": {
        "require": ["学院交流会票"], "core": ["入场键", "关卡得星_3"],
        "min_core": 2, "neg": ["取消键", "扫荡开始"],
        "parent": "Exchange_SchoolSelect",
    },
    "Exchange_SweepPanel": {
        "require": ["学院交流会票", "扫荡开始"], "core": ["任务开始", "弹窗叉叉"],
        "min_core": 1, "neg": ["取消键"], "parent": "Exchange_StageList",
    },
    # 活动关的扫荡面板 —— 与上面两家**长得一模一样**, 只靠"没有那两种票"区分。
    # 草稿 §6 当时无帧所以缺了这一节, 结果它的帧全被判成悬赏的(实测 219 帧)。
    "EventSweepPanel": {
        "require": ["扫荡开始"], "core": ["任务开始", "弹窗叉叉"],
        "min_core": 1, "neg": ["取消键", "悬赏通缉票", "学院交流会票"],
        "parent": "EventQuestList",
    },
    #  商店族
    "CreditShop": {
        "core": ["信用点商店_已选中", "全部选择", "购买"],
        "min_core": 2, "parent": "Lobby",
    },
    "PyroxenePackPage": {
        "core": ["免费", "组合包已选择", "购买", "弹窗叉叉"],
        "min_core": 2, "parent": "Lobby",
    },
    "EventShop": {
        # 坑: 本期活动皮上 v13 只剩 购买+chrome, 无正锚  weak, 仅观测
        "core": ["购买"],
        "min_core": 1, "weak": True,
        "neg": ["全部选择", "战术大赛商店已选择", "信用点商店_已选中", "免费"],
        "parent": None,
    },
    #  全域通用
    "RewardObtained": {
        "core": ["获得奖励", "点击继续字样"], "min_core": 1, "parent": None,
    },
    "BondLevelUp": {"core": ["羁绊升级"], "min_core": 1, "parent": None},
    "Club": {"core": ["社团"], "min_core": 1, "parent": "Lobby"},
    "Loading": {"core": ["加载中"], "min_core": 1, "parent": None},
    "Cafe_Headpat": {  # 坑: chrome 全隐 + 只有 Emoticon 是合法态
        "core": ["Emoticon_Action"], "min_core": 1, "parent": "Cafe_Hall1",
    },
}

# 坑: 确认框族**故意不定名** —— cls 集合不可分, 图不做它做不到的判断。
# 命中这个形状就报 ambiguous, 让调用方用上下文闸(结构/位置)去分。
CONFIRM_SHAPE = ("确认键", "取消键")

#  转移边: (from_page, 点这个 cls, to_page)
# 全部来自 screen_flow_draft 的"点击去向(观测)"。返回键parent /
# 回大厅按钮Lobby 是全局默认边, 不在这里重复。
EDGES: List[Tuple[str, str, str]] = [
    ("Lobby", "任务大厅入口", "MissionHub"),
    ("Lobby", "课程表入口", "Schedule_RegionSelect"),
    ("Lobby", "咖啡厅入口", "Cafe_Hall1"),
    ("Lobby", "商店入口", "CreditShop"),
    ("Lobby", "社交入口", "Club"),
    ("Lobby", "购买青辉石", "PyroxenePackPage"),
    ("MissionHub", "战术大赛", "Arena_Opponents"),
    ("MissionHub", "悬赏通缉", "Bounty_SceneSelect"),
    ("MissionHub", "学院交流会", "Exchange_SchoolSelect"),
    # 2026-07-25 审计补边: EventQuestList/EventSweepPanel 原先**零入边** —
    # 文件头旗舰用例 route(...,'EventQuestList') 永远 None, 而 event_quest 是
    # 当前 skill_order 唯一主链。banner cls 取 live 实际点击的 405(event_quest
    # _enter 点「距离结束还剩」进活动, 不是 idx77 那个入口框)。
    ("MissionHub", "距离结束还剩", "EventQuestList"),
    ("EventQuestList", "入场键", "EventSweepPanel"),
    ("Schedule_RegionSelect", "夏莱办公室", "Schedule_RoomCarousel"),
    ("Schedule_RoomCarousel", "全体课程表", "Schedule_RosterPopout"),
    ("Schedule_RosterPopout", "课程表开始", "Schedule_Report"),
    ("Arena_Opponents", "战术大赛对战选择区域", "Arena_Challenge"),
    ("Arena_Challenge", "攻击编制", "Formation_Attack"),
    ("Cafe_Hall1", "移动至2号点", "Cafe_Hall2"),
    ("Cafe_Hall2", "移动至一号店", "Cafe_Hall1"),
    ("Cafe_Hall1", "咖啡厅邀请卷", "Cafe_Invite"),
    ("Bounty_SceneSelect", "教室", "Bounty_StageList"),
    ("Bounty_StageList", "入场键", "Bounty_SweepPanel"),
    ("Exchange_SchoolSelect", "三一", "Exchange_StageList"),
    ("Exchange_StageList", "入场键", "Exchange_SweepPanel"),
    ("Arena_Opponents", "战术大赛商店", "ArenaShop"),
]

_GLOBAL_HOME = "回大厅按钮"
_GLOBAL_BACK = "返回键"


def _names(screen, conf: float = 0.30) -> set:
    return {b.cls_name for b in (getattr(screen, "yolo_boxes", None) or [])
            if b.confidence >= conf}


def identify(screen, conf: float = 0.30) -> Tuple[Optional[str], str]:
    """返回 (page|None, 说明)。识别不出一律 None —— **绝不硬归类**。

    多个页面同时命中时取 (命中 core 数, min_core) 最大的那个: 子画面的指纹
    总比父画面更具体(例如 Schedule_Report 比 Schedule_RosterPopout 多一个
    确认键), 更具体的应该赢。
    """
    ns = _names(screen, conf)
    if not ns:
        return None, "零检出(黑屏/过场/CG) — 等下一帧"
    # 坑: 只剩一个叉叉 = 弹窗动画中, 不判
    if ns <= {"弹窗叉叉", _GLOBAL_BACK, _GLOBAL_HOME}:
        return None, "只剩弹窗/chrome cls — 等下一帧"
    hits = []
    for name, p in PAGES.items():
        if any(n in ns for n in p.get("neg", ())):
            continue
        req = p.get("require", ())
        if req and not all(n in ns for n in req):
            continue
        # require_any(2026-07-27): "这一页必须至少带一个**本域专属**物件"。
        # 与 require(AND) 的分工: AND 适合"只有一个专属锚且必现"的页; 域里有
        # **多个专属锚、但每帧只保证出现其中之一**时只能用 OR。
        # 实测逼出来的(全语料 44,177 有检出帧): Schedule_RoomCarousel 用
        # AND require=[全体课程表] 翻 182 帧 —— 修正 117(社交页误判 57/咖啡厅 4/
        # 纯"左切换+右切换"凑数垃圾 56), 但**误伤 65 帧真课程表页**(那些帧
        # 全体课程表 没检出、只检出 课程表票)。改 OR 后误伤归零。
        req_any = p.get("require_any", ())
        if req_any and not any(n in ns for n in req_any):
            continue
        k = sum(1 for c in p["core"] if c in ns)
        if k >= p["min_core"]:
            # require 命中也算"具体度", 让专属判据赢过通用判据
            hits.append((k + len(req), p["min_core"], name))
    if not hits:
        # 坑: 确认框形状但认不出是哪个框
        if all(c in ns for c in CONFIRM_SHAPE):
            return None, ("ConfirmDialog_ambiguous — 确认框族 cls 不可分, "
                          "交上下文闸(结构/位置)判, 图不猜")
        return None, f"无匹配页面 (在屏 {len(ns)} 类)"
    hits.sort(key=lambda t: (t[0], t[1]), reverse=True)
    best = hits[0]
    tag = " weak" if PAGES[best[2]].get("weak") else ""
    extra = ""
    if len(hits) > 1 and hits[1][0] == best[0]:
        extra = f" (并列: {[h[2] for h in hits[:3]]})"
    return best[2], f"core {best[0]}/{len(PAGES[best[2]]['core'])}{tag}{extra}"


def neighbors(page: str) -> List[Tuple[str, str]]:
    """[(点这个 cls, 到这一页), ...] 含全局默认边。"""
    out = [(c, to) for (f, c, to) in EDGES if f == page]
    par = PAGES.get(page, {}).get("parent")
    if par:
        out.append((_GLOBAL_BACK, par))
    if page != "Lobby":
        out.append((_GLOBAL_HOME, "Lobby"))
    return out


def route(src: str, dst: str, max_hops: int = 8) -> Optional[List[str]]:
    """BFS: 从 src 走到 dst 要依次点哪些 cls。走不到  None。"""
    if src == dst:
        return []
    if src not in PAGES or dst not in PAGES:
        return None
    from collections import deque
    q = deque([(src, [])])
    seen = {src}
    while q:
        cur, path = q.popleft()
        if len(path) >= max_hops:
            continue
        for click, nxt in neighbors(cur):
            if nxt in seen:
                continue
            if nxt == dst:
                return path + [click]
            seen.add(nxt)
            q.append((nxt, path + [click]))
    return None


def validate_vocab(master: List[str]) -> List[str]:
    """所有引用的 cls 必须在模型词表里 —— 拼错一个字就是一条永不命中的死判据。
    返回不存在的名字列表(空 = 全对)。"""
    used = set()
    for p in PAGES.values():
        # require 必须进校验(2026-07-25 审计): 它语义是"少一个直接否决", 拼错
        # 不是死判据而是**永久否决该页** — 比 core 拼错更静默。之前只是碰巧
        # 3 个 require 名都出现在别页 neg 里才被盖到。
        used |= (set(p["core"]) | set(p.get("neg", ()))
                 | set(p.get("require", ())))
    for _f, c, _t in EDGES:
        used.add(c)
    used |= set(CONFIRM_SHAPE) | {_GLOBAL_BACK, _GLOBAL_HOME}
    return sorted(n for n in used if n not in set(master))
