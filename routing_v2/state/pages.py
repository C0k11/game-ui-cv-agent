# -*- coding: utf-8 -*-
"""页面身份 —— **全 bot 唯一的定义处**（README §A2）。

 三条不可违反的建模规则（每条都对应一次实锤事故）:

【§A9】页面身份只取决于**结构**，不取决于"有几个 / 打过没"。
   反例（新活动首日全线崩）: `_on_quest_list` 要求 **≥2 个已解锁入场键**，
   而 CODE:BOX 首日是 1 开 + 4 锁  关卡列表页整个不被承认。
    这里的签名只问"有没有一列关卡行"，不问有几行、能不能打。

【§A1】没有匹配 = `UNKNOWN`，**UNKNOWN 是一等公民**，对应动作是 **no-op**。
   绝不设"兜底页面"，更不能给兜底页面配破坏性动作。老代码那句
   `x = F(bs, ["弹窗叉叉","回大厅按钮","返回键"]); return "STRAY", x`
   就是把"我不认识这一帧"翻译成了"那就点回大厅吧" —— `回大厅按钮` 几乎每页
   都在，于是每个转场帧都能把 bot 弹回大厅（用户现场目击）。

【§A3】单帧不算数。这里只做**单帧打分**，"到底在哪一页"由 `machine.py` 用
   连续 N 帧确认。转场帧是一瞬，真状态是持续的。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import routing_v2.act.money as _money
from routing_v2.percept.observe import Observation, Rect
from routing_v2.state import vocab as V

CONF = 0.45          # 页面签名的统一门槛。金钱相关判据自己再收紧。

# 常用 region（顶栏读数/角落按钮**必须**带 region，见 840AP 事故）
TOPBAR: Rect = (0.0, 0.0, 1.0, 0.12)
BODY: Rect = (0.0, 0.12, 1.0, 1.0)          # 弹窗体（顶栏以下）
TOP_RIGHT: Rect = (0.78, 0.0, 1.0, 0.30)
BOTTOM: Rect = (0.0, 0.80, 1.0, 1.0)
RIGHT_PANEL: Rect = (0.55, 0.10, 1.0, 0.92)


@dataclass(frozen=True)
class Sig:
    """一个页面的结构签名。"""
    id: str
    need: Sequence[str] = ()             # 全部必须在
    any_of: Sequence[str] = ()           # 至少 min_any 个
    min_any: int = 1
    forbid: Sequence[str] = ()           # 任一出现即否决
    where: Dict[str, Rect] = field(default_factory=dict)   # 某个 cls 的位置约束
    priority: int = 0                    # 同分时更大的赢（越具体越大）
    interrupt: bool = False              # 打断型：先于任何 flow 处理
    overlay: bool = False                # 覆盖层：盖在某一页之上，不是"另一页"
    pred: Optional[Callable] = None      # 代码判据（复杂到写不成 cls 组合时）
    note: str = ""

    def match(self, obs: Observation) -> Optional[float]:
        """匹配则返回分数（满足的锚点数），否则 None。"""
        if self.pred is not None:
            return 1.0 if self.pred(obs) else None
        for c in self.need:
            if obs.find(c, CONF, region=self.where.get(c)) is None:
                return None
        for c in self.forbid:
            if obs.find(c, CONF, region=self.where.get(c)) is not None:
                return None
        hits = sum(1 for c in self.any_of
                   if obs.find(c, CONF, region=self.where.get(c)) is not None)
        if self.any_of and hits < self.min_any:
            return None
        return float(len(self.need) + hits)


# ══════════════════════════════════════════════════════════════════════
#  打断型（interrupt）—— 与"在哪一页"无关，见到就得先处理
# ══════════════════════════════════════════════════════════════════════
INTERRUPTS: List[Sig] = [
    Sig("money_popup",
        pred=lambda o: _money.purchase_context(o) is not None,
        priority=100, interrupt=True,
        note="真的站在花钱的位置上（判据见 act/money.py，三条结构判据）。"
             "单独一个 `购买青辉石` **不算** —— 大厅左侧常驻买石广告位，"
             "conf 0.95 稳定在场，按词判会每轮第一帧就 HALT（08-08 live 实锤）"),

    Sig("story_cutscene",
        # 2026-08-11 小号实测补 `剧情中断退出`：跳完一段剧情弹「下一章節」框
        #    （中斷 / 觀看），那一屏**没有** 剧情menu / 跳过故事键 / 点击继续 ——
        #    三个老成员一个都匹配不上  页面不判成 story_cutscene  逃生链根本
        #    不会被调用  卡死在那个框上。它属于过场链的一环，必须认进来。
        any_of=[V.STORY_MENU, V.STORY_SKIP, V.STORY_TAP_CONTINUE, V.STORY_QUIT],
        where={V.STORY_MENU: TOP_RIGHT, V.STORY_SKIP: TOP_RIGHT},
        # `点击继续字样` 在**奖励页**上也有（领完东西那一屏）——
        #    2026-08-08 实测咖啡厅收益结算被误判成剧情过场。动作碰巧同样是
        #    "点掉"所以无害，但判据该收紧：屏上在**给**你东西就不是过场。
        forbid=[V.GOT_REWARD, V.BATTLE_WIN, V.BATTLE_COMPLETE],
        priority=90, interrupt=True,
        note="剧情过场**不吃 KEYCODE_BACK**（08-07 连按5次实测无响应）。"
             "整个活动被开场剧情挡死过一次。唯一出路：menu跳过确认。"),

    Sig("levelup",
        any_of=[V.BOND_LEVELUP, V.REGION_LEVELUP],
        priority=85, interrupt=True,
        note="全屏升级过场，点任意处消掉"),

    # 系统级「是否結束？」退出游戏确认框 —— **確認 = 退出游戏**
    Sig("quit_dialog",
        need=[V.CONFIRM, V.CANCEL],
        forbid=[V.AP, V.CREDIT, V.PYROXENE, V.BACK, V.HOME,
                # 剧情「跳過」确认框也是 確認+取消、没顶栏没返回键  完全
                #    命中本签名，而它 priority 95 > story_cutscene 90 且 instant
                #     稳赢  逃生链最后一步被点成"取消"，回到 08-07
                #    「整个活动被开场剧情挡死」。117 帧真实剧情跳過框实测
                #    117/117 全被误判成 quit_dialog（08-09 审查）。
                V.STORY_MENU, V.STORY_SKIP, V.STORY_TAP_CONTINUE],
        where={V.AP: TOPBAR, V.CREDIT: TOPBAR, V.PYROXENE: TOPBAR},
        priority=95, interrupt=True,
        note="2026-08-08 实锤：大厅按返回键会弹「通知/是否結束？/取消·確認」，"
             "而我给 confirm_dialog 的默认动作是**点確認** = **退出游戏**。"
             "判据：游戏内的确认框永远挂在某个页面上（**顶栏货币在**）；"
             "系统退出框把整个 UI 都盖住了，顶栏一个货币都没有。"
             "顶栏货币是**弱证据**（扫荡确认框把屏压暗时顶栏常检不出，"
             "day2 悬赏扫荡确认被误判成退出框差点点取消把扫荡取消掉）。"
             "「有叉叉=游戏框」也是错的 —— 真实退出框帧上就检出过 0.96 的"
             "叉叉（fixture 为证）。真正的结构证据：**退出框只会从大厅弹出**，"
             "屏上有 `返回键`/`回大厅按钮` = 在设施页里 = 游戏内对话框。"
             "处置：**只允许点取消**，永不点確認"),

    # 阈值必须和 `money.purchase_context` 里那句 `obs.has(V.LOADING, 0.40)`
    #   **对齐**：那里一旦认定"这是加载帧"就整段不判金钱风险；如果这边的
    #   打断用 0.45，[0.40,0.45) 这一段就是**金钱闸全瞎、而 flow 照常动作**。
    #   对齐之后：money 认为不可判的帧，flow 也一定被打断按住。
    Sig("loading", pred=lambda o: o.has(V.LOADING, 0.40),
        priority=80, interrupt=True,
        note="起跑必须过预热闸：冷启动的加载帧会把收菜预算烧光（08-01 事故）"),
]


# ══════════════════════════════════════════════════════════════════════
#  轮播位**两个活动共用同一个 405** —— 只能用「进去之后的内容」区分
# ══════════════════════════════════════════════════════════════════════
# 2026-08-11 live 实锤：任务大厅左上**一个轮播位轮流显示两个活动**，二者都只
# 检出 `距离结束还剩`(405)，点进去纯看运气 —— 实测**连进两次都进了「夏萊總結算」**。
# 时序赛跑赢不了（轮播 3.0s，转场帧上的点击还会被稳定门吞，当天试 7 次没进对），
#  **判据只能放在"进去之后"**，用页面内容认。
#
#   夏萊總結算（**引导型**活动；遊戲指南原文「不存在活動關卡」，道具靠去打
#   普通 任務/特殊任務 掉落，屏上写着 道具持有量 90 / 進行狀況 90/15,000）
#   —— `v2step_20260811/013410`、`014248` 两帧实测:
#       入场键 ×2 (0.97/0.96, 任務 / 特殊任務) + 活动商店 0.55~0.58
#       **一个 关卡得星 都没有**；**没有** 活动quest / 活动剧情 页签；
#         没有 活动任务；没有 奖励资讯（conf 压到 0.02 也是 0 检出）
#   CODE:BOX（正常活动，`v2step_20260810` 13 帧同模型同分辨率对照）:
#       入场键 ×4 + 关卡得星_3 ×4 (0.96~0.97) + 活动quest_已选择 0.77~0.96
#       + **活动剧情 0.99（13/13 帧全在）** + 活动任务 0.87~0.97 + 奖励资讯 0.34~0.61
#
# 判据只能问"有没有这一族"，**绝不能问"有几个入场键"**（§A9 —— 正是新活动
#    首日全线崩的那条：CODE:BOX 首日 1 开 4 锁；反过来引导型活动也完全可能
#    只有 1 行或 3 行。今天恰好是 2 vs 4，明天就未必）。
# 正向锚 = `活动商店`（这一页仍然是活动页的一部分）；
#   否定锚 = 关卡得星族 + quest/剧情页签 —— 后者在正常活动页上稳定 0.9+，
#   比数数可靠一个数量级，而且**两族要同时消失才会误判**（实测那 13 帧里
#   `活动商店` 掉到 0.10 过一次，得星和页签照样在，否定锚兜住了）。
_GUIDE_HUB_NOT_TABS = [V.EVENT_QUEST, V.EVENT_QUEST_SEL,
                       V.EVENT_STORY, V.EVENT_STORY_SEL,
                       V.EVENT_TASK, V.EVENT_REWARD_INFO, V.EVENT_AFTERSTORY]
_GUIDE_HUB_NOT_STARS = [V.STAR_0, V.STAR_3]
# 悬赏 / JFD / 大赛的关卡列表也有一排入场键 —— 票券 cls 在场就不是活动页
_GUIDE_HUB_NOT_TICKETS = [V.TICKET_BOUNTY, V.TICKET_JFD, V.TICKET_ARENA]


def _is_event_guide_hub(o: Observation) -> bool:
    """「引导型活动主页」= 活动页 + 有入场行 + **既没关卡得星也没 quest 页签**。"""
    #  正向：还在活动页里（左下角那个商店入口）。
    #    夏萊 cy≈0.884、CODE:BOX cy≈0.934 —— region 只取左下角，用来挡掉
    #    别处同名框，不用来区分这两个活动。
    if o.find(V.EVENT_SHOP, 0.35, region=(0.0, 0.72, 0.50, 1.0)) is None:
        return False
    #  正向：有「入場」行（引导型活动把 任務/特殊任務 做成入场行）
    if not o.has(V.STAGE_ENTER, CONF):
        return False
    #  否定：有关卡得星 = 这是正常活动的关卡列表
    if o.has(_GUIDE_HUB_NOT_STARS, 0.35):
        return False
    #  否定：有 quest/剧情 页签或活动任务 = 正常活动页
    if o.has(_GUIDE_HUB_NOT_TABS, 0.40):
        return False
    #  排他：票券型关卡列表
    if o.has(_GUIDE_HUB_NOT_TICKETS, 0.40):
        return False
    return True


# ══════════════════════════════════════════════════════════════════════
#  普通页面
# ══════════════════════════════════════════════════════════════════════
PAGES: List[Sig] = [
    # ── 大厅 / 大厅级 ────────────────────────────────────────────────
    Sig("lobby", any_of=V.LOBBY_NAV, min_any=3,
        forbid=[V.COMBO_PACK, V.COMBO_PACK_SEL, V.SHOP_SELECT_ALL, V.SORTIE],
        priority=10, note="底栏 3 个以上设施入口 = 大厅"),

    Sig("task_hall", any_of=V.HUB_TILES + [V.SPECIAL_DEFENSE, V.SPECIAL_CREDIT],
        min_any=3, forbid=[V.NAV_CAFE],
        priority=20, note="各玩法的红点只在这里可见；大厅入口那个红点不是工作信号"),

    # ── 战斗链（优先级最高的普通页，别被别的页遮住）──────────────────
    Sig("battle", any_of=V.BATTLE_CHROME, min_any=2,
        priority=70, note="战斗内。单个 chrome cls 会在别处漏出，要 2 个"),

    Sig("battle_result",
        need=[V.CONFIRM], where={V.CONFIRM: BOTTOM},
        forbid=[V.STAGE_ENTER, V.BATTLE_PAUSE, V.CANCEL, V.SORTIE,
                # 别把设施页的底部確認当成战斗结算（邮件/每日任务的領取键
                #    也在底部）—— 它优先级 60，一旦误判会盖掉真正的页面
                V.CLAIM_ONCE_YELLOW, V.CLAIM_ONCE_GREY, V.CLAIM_ALL_YELLOW,
                V.NAV_CAFE],
        priority=60,
        note="战斗结算：確認在底部 + 没有入场键。"
             "`战斗失败` 只有 train=6（v16 重建后从 0 涨上来的，仍然是 WEAK）"
             " —— **输了基本还是看不见**，胜负未知时不许记 cleared"),

    # 羁绊剧情面板盖在 MomoTalk 上（右半屏），左边列表的未读 cls 还在
    #     页面身份被 momo_list 抢走，flow 又去点未读对话，**面板永远点不进**
    #    （08-09 实锤）。它有独一无二的 `进入羁绊剧情`，给它 overlay 身份。
    Sig("bond_story_panel", need=[V.MOMO_ENTER_BOND], priority=56, overlay=True,
        note="羁绊剧情面板（MomoTalk 聊完解锁）：進入羈絆劇情。"
             "面板里的青辉石是**奖励预览**（EP 奖励 x40），不是价签"),

    Sig("reward", any_of=[V.GOT_REWARD, V.STORY_TAP_CONTINUE],
        priority=58, overlay=True,
        note="奖励明细，点掉即可（盖在底页上）。"
             "`领取奖励_黄` 已从信号里拿掉（08-09）：它是 arena 列表/每日"
             "任务页的**常驻按钮**，当 overlay 信号会把整页派发劫持成干等"),

    # 编队的「快速編輯」面板必须有**专属身份**（08-09 实锤）：它底部也有
    #    確認键，于是被 battle_result / ack_dialog 轮流认领  flow 去点確認
    #    = **把旧阵容原样提交**（自動还没点），加成白丢。
    #    判据 = `自動編輯按鈕` 只在这个面板上出现，独一无二。
    #    （到处给别的签名加 forbid 是治标；给它专属身份才是 §A2 一处定义。）
    Sig("squad_quick_edit", need=[V.SQUAD_AUTO_EDIT], priority=70,
        note="编队-快速編輯面板：自動填充 + 確認。绝不能被当成结算/通知框"),

    Sig("formation", need=[V.SORTIE],
        priority=65,
        note="编队页。老状态机不认它  落到 STRAY  差点去点返回键。"
             "1部队高亮/2部队 都有且好用，部队选择走 cls 不写死坐标"),

    # 走格子(集中指挥)地图。结构判据 = 格子数 + 起点/箭头/任务开始灰
    #    （2026-08-13 小号实测: 这页零专属锚点, 原来在 facility(score2)/unknown
    #    之间摆 -- 而格子族 cls 0.93-0.97 稳定, 数量就是最硬的证据）。
    #    priority 要压过 stage_popup(55): 部署完、`任务开始` 变亮的那几帧
    #    两个签名同时满足, 这时人在地图上不在弹窗上。
    Sig("grid_quest", pred=lambda obs: (
            len(obs.all([V.GRID_CELL, V.GRID_CELL_OPEN, V.GRID_CELL_FOG],
                        0.45)) >= 3
            and obs.has([V.GRID_START, V.GRID_START_GREY, V.GRID_ARROW,
                         V.GRID_ALLY, V.TASK_START_GREY], 0.45)),
        priority=58,
        note="走格子地图（3D 俯视, 同资产不同相机域）。格子>=3 + 任一结构锚"),

    Sig("stage_popup",
        any_of=[V.TASK_START, V.SWEEP_START, V.STORY_ENTER_CHAPTER],
        priority=55,
        note="关卡信息弹窗。首通弹的是「章節資訊 + 進入章節」而不是扫荡面板 —— "
             "老代码只认扫荡面板，新活动首日直接卡死"),

    Sig("sweep_dialog",
        any_of=[V.SWEEP_BATCH_START, V.SWEEP_BATCH_START_GREY],
        priority=57, note="批量扫荡对话框"),

    # ── 活动 ─────────────────────────────────────────────────────────
    Sig("event_quest_list",
        any_of=[V.EVENT_SHOP, V.EVENT_TASK, V.EVENT_QUEST_SEL],
        need=[],
        priority=45,
        where={},
        note="活动关卡列表。结构判据 = 活动页 chrome + 至少一行关卡（下面补）。"
             "**「有入场行」不等于「有活动关卡」** —— 引导型活动（夏萊總結算）"
             "也有两行入场键，08-11 就是在这里被认走的  见 `event_guide_hub`"),

    Sig("event_page",
        any_of=[V.EVENT_SHOP, V.EVENT_TASK, V.EVENT_QUEST, V.EVENT_QUEST_SEL,
                V.EVENT_STORY, V.EVENT_STORY_SEL, V.EVENT_REWARD_INFO],
        priority=40, note="活动主页（还没确定在哪个页签）"),

    Sig("event_ended",
        any_of=[V.EVENT_AFTERSTORY],
        priority=48, note="上期活动余韵期（後日談是它的特征物）—— 只能领尾奖，退出去"),

    # 它必须**盖过 event_quest_list(45)**：夏萊那两帧上 `活动商店` 0.55 +
    #    入场键 2 个，正好满足 event_quest_list 的 any_of + "至少一行关卡"，
    #    08-11 live 就是这么被认成关卡列表、然后去点「入場」的 —— 而那一下
    #    直接把 bot 送去普通 任務 关卡列表（**刷什么是用户的策略**，bot 不许自己选）。
    #    priority 也要压过 event_shop(47) / event_ended(48)。
    Sig("event_guide_hub", pred=_is_event_guide_hub, priority=49,
        note="引导型活动主页（夏萊總結算这类）：遊戲指南写「不存在活動關卡」，"
             "「入場」按钮直接把你送去**普通** 任務/特殊任務。"
             "轮播位两个活动共用 405  进错了只能在这一页认出来  退出去重进。"
             "bot 绝不自作主张去打 任務/特殊任務 —— 刷什么是用户的策略"),

    # ── 票券扫荡型 ───────────────────────────────────────────────────
    Sig("bounty_branch",
        need=[V.TICKET_BOUNTY],
        any_of=V.BOUNTY_BRANCHES, min_any=1,
        priority=50, note="悬赏分支选择。分支**动态从屏上 cls 列**，不写死清单"),

    Sig("jfd_academy",
        need=[V.TICKET_JFD],
        any_of=V.JFD_ACADEMIES, min_any=1,
        priority=50,
        note="学院交流会学院选择。三一/千年/格黑娜 各 84 train —— "
             "老代码那句「没有 cls（v6 gap）」早就过期了，它却一直在用写死坐标"),

    Sig("bounty_stage", need=[V.TICKET_BOUNTY], priority=42, note="悬赏关卡列表"),
    Sig("jfd_stage", need=[V.TICKET_JFD], priority=42, note="JFD 关卡列表"),
    Sig("arena", need=[V.TICKET_ARENA], priority=42, note="战术大赛"),

    Sig("campaign_stage",
        any_of=[V.STAGE_NORMAL_SEL, V.STAGE_HARD_SEL, V.STAGE_NORMAL,
                V.STAGE_HARD, V.SWEEP_BATCH],
        forbid=[V.SWEEP_BATCH_START, V.SWEEP_BATCH_START_GREY],
        priority=35, note="主线关卡列表"),

    # ── 设施 ─────────────────────────────────────────────────────────
    Sig("cafe",
        any_of=[V.CAFE_EARNINGS, V.CAFE_TICKET, V.CAFE_MOVE_1F, V.CAFE_MOVE_2F],
        priority=30, note="咖啡厅"),

    Sig("cafe_invite_list", need=[V.CAFE_INVITE], priority=44,
        note="邀请学生列表（邀请键在场）"),

    Sig("schedule_region", need=[V.SCHED_ALL], priority=34, note="课程表某区域内"),
    Sig("schedule_area", any_of=V.SCHOOL_AREAS, min_any=1,
        forbid=[V.SCHED_ALL], priority=32, note="课程表区域选择"),

    Sig("craft", any_of=[V.CRAFT_QUICK, V.CRAFT_START, V.CRAFT_START_GREY],
        priority=30, note="制造"),

    # 活动商店：结构 = **左栏币种 tab**（货币/货币_已选择）。通用商店那套
    #   （全部选择/信用点商店_已选中）在这里一个都没有 —— 08-08 实测因此落到
    #   unknown，bot 在货架前干瞪眼 401 帧。
    # 身份不能只靠左栏 `货币` cls（08-09 实锤：CODE:BOX 活动的商店左栏
    #    **一个 tab 都检不出**，页面掉成 facility  flow 无出路干等）。
    #    货架那排 `购买` 才是商店页的稳定结构特征（实测 0.86/0.83/0.83），
    #    再用 forbid 把信用点商店/青辉石tab/大赛商店排掉。
    Sig("event_shop",
        any_of=[V.CURRENCY_SEL, V.CURRENCY, V.SHOP_BUY], min_any=1,
        where={V.CURRENCY_SEL: (0.0, 0.10, 0.24, 0.98),
               V.CURRENCY: (0.0, 0.10, 0.24, 0.98),
               V.SHOP_BUY: (0.40, 0.15, 1.0, 0.98)},
        forbid=[V.SHOP_TAB_CREDIT_SEL, V.SHOP_TAB_PYROXENE_SEL,
                V.ARENA_SHOP_TAB_SEL, V.SHOP_SELECT_ALL, V.SHOP_SELECT_ALL_GREY],
        priority=47, note="活动商店（左栏币种 tab 或货架购买键；见 forbid 排他）"),

    Sig("shop_pyroxene_tab", need=[V.SHOP_TAB_PYROXENE_SEL], priority=52,
        note="进了青辉石 tab = 正在花青辉石的位置。立刻退出去"),
    # 别只靠 `战术大赛商店已选择` 认这一页（08-11 live 实测：tab **明明已高亮**，
    #    该 cls conf 只有 **0.116**，而同一个 tab 的未选中态 `战术大赛商店` 反而 0.81
    #    —— 这个族的**选中态几乎没训过**）。结果是页面判不成 arena_shop、
    #    `on_arena_shop` 根本不被调用，整条大赛商店支线静默失效。
    #     加一条**强得多的内容判据**：货架价签上的大赛币是**多实例** 0.97。
    #      信用点商店的价签是 `信用点`、活动商店是 `货币`，都不会误命中。
    # 2026-08-13 live 实锤，判据从「或」收紧成「必须有货架证据」:
    #    `战术大赛商店已选择` 认的是**左栏里被选中的那一行**，不是「战术大赛」
    #    这一行 —— 帧证：大決戰 被选中时它照样给 0.90~0.95，而 `戰術大賽`
    #    在它下面、未选中。于是 bot 点歪到大決戰之后，页面照样判成 arena_shop、
    #    `on_arena_shop` 照常执行，站在**大決戰商店**里找能量饮料，找不到就退出，
    #    全程自洽而全程是错的。
    #     tab 的选中态降级成**加分项**，货架上的大赛币价签才是准入证据
    #      （多实例 0.95+，且信用点/活动商店的价签是别的 cls，不会误命中）。
    #    这条 cls 的标注问题另行补（见 cls 语义表）。
    Sig("arena_shop",
        need=[V.ARENA_SHOP_CURRENCY],
        # region 只排掉**左栏那个 tab 图标**（cx≈0.03），不排顶部余额栏:
        #    这一页有条**只有大赛商店才有**的大赛币余额栏（实测 0.97 @cx≈0.57,
        #    cy≈0.12），它是页面级、常驻、强检出的证据。
        #    ⛔ 我第一版把 region 收成"货架区 cy 0.20-0.98"，当场把这一页判不出来了
        #      —— 货架上只有那两瓶饮料的价签会被检出（神名文字那 6 张的价签和
        #      購買一样是系统性漏标），饮料一买完货架区就一个都没有，
        #      页面掉成 unknown、flow 0 tap 空转 36 秒。
        #      **别拿一个自己也欠拟合的 cls 当准入证据。**
        where={V.ARENA_SHOP_CURRENCY: (0.40, 0.02, 1.0, 0.98)},
        forbid=[V.SHOP_TAB_PYROXENE_SEL],
        priority=46, note="战术大赛商店：**货架区**出现大赛币价签才算数。"
             "tab 选中态 `战术大赛商店已选择` 不可信 —— 它认的是"
             "「左栏选中的那一行」，大決戰被选中时也给 0.95"),
    Sig("shop",
        any_of=[V.SHOP_SELECT_ALL, V.SHOP_SELECT_ALL_GREY, V.SHOP_ALL_SELECTED,
                V.SHOP_TAB_CREDIT_SEL, V.SHOP_BUY_SELECTED],
        forbid=[V.COMBO_PACK, V.COMBO_PACK_SEL],
        priority=30, note="商店货架"),
    Sig("combo_pack", any_of=[V.COMBO_PACK, V.COMBO_PACK_SEL], priority=54,
        note="组合包页（免費組合包旁边就是 CAD$16.99 真钱包）—— 逐帧人审"),

    Sig("mail", any_of=[V.CLAIM_ONCE_YELLOW, V.CLAIM_ONCE_GREY, V.CLAIM_BLUE],
        forbid=[V.NAV_CAFE], priority=30,
        note="邮件箱。`领取蓝色` 也算：领完一次后一次領取键消失、只剩逐条的"
             "蓝色領取键，不带它页面身份会在 mail↔unknown 之间抖（08-08 实测"
             "3s 内抖了 4 次）"),

    Sig("daily_mission",
        any_of=[V.CLAIM_ALL_YELLOW, V.CLAIM_ALL_GREY],
        forbid=[V.CRAFT_QUICK, V.CLAIM_ONCE_YELLOW], priority=28,
        note="每日任务。一键领取灰色 train=0，不能拿它判'已领完'"),

    Sig("club", need=[V.CLUB], priority=30, note="社团"),

    # ── MomoTalk ─────────────────────────────────────────────────────
    Sig("momo_chat",
        any_of=[V.MOMO_REPLY_OPT, V.MOMO_SENDING, V.MOMO_GOTO_BOND],
        priority=44, note="MomoTalk 对话中"),
    Sig("momo_list", any_of=[V.MOMO_TAB_SEL, V.MOMO_TAB, V.MOMO_UNREAD],
        priority=30, note="MomoTalk 列表"),

    # ── 剧情 / 挖矿 ──────────────────────────────────────────────────
    Sig("story_hub", any_of=[V.STORY_MAIN, V.STORY_SHORT, V.STORY_SIDE],
        min_any=2, priority=33, note="剧情大厅"),
    Sig("story_nodes", any_of=[V.STORY_NODE_DONE, V.STORY_NODE_UNDONE],
        priority=36, note="剧情节点图"),

    # 2026-08-11 补的一层：`主线剧情` 点进去**不是**节点图，是**大章节图**
    #    （對策委員會篇 / 序幕 / 最終篇 那一屏），右侧还有小章节面板。
    #    以前它判成 `facility`(priority 4 兜底)，而 `StoryMiningFlow` 没有
    #    `on_facility`  **进去就卡死**，整条剧情挖矿链从来没走通过。
    # 身份判据（用户口述的路由规则同源）: banner 上的 `new`/`完成` 角标，
    #    且**没有**节点图那两个 cls、也没有入场键（那是节点图/关卡列表的特征）。
    Sig("story_chapter_map", pred=lambda o: (
        o.has([V.NEW_MARK, V.NODE_DONE], 0.40)
        and not o.has([V.STORY_NODE_DONE, V.STORY_NODE_UNDONE], 0.40)
        and not o.has(V.STAGE_ENTER, 0.45)
        and o.has(V.BACK, 0.40)),
        priority=32,
        note="剧情大章节图。推进顺序 = `new`/`完成` 按 **cx 从左到右**；"
             "右侧小章节靠 `黄点` 定位（用户 2026-08-11 口述的路由规则）"),

    # ── 覆盖层：盖在某一页之上，**不参与"我在哪一页"的竞争** ──────────
    # 2026-08-08 live 实锤（为什么必须有 overlay 这个概念）:
    #    邮件页上「一次領取黄色」和结算「確認键」会交替被检出，两者当成两个
    #    **页面**去竞争，于是页面身份在 mail ↔ ack_dialog 之间来回抖了几十次，
    #    flow 每次都拿到"刚换页"的状态、闸每次都放行  同一个領取键点了 65 下。
    #    真相是：底页一直是 mail，確認框只是盖在上面。分开跟踪，抖动立刻消失。
    Sig("confirm_dialog", need=[V.CONFIRM, V.CANCEL], priority=15, overlay=True,
        note="双键决策框 —— 有取消键 = 这是个**决策**，按各 flow 的意图处理"),

    # 2026-08-08 live：cafe 收益面板领完变 `领取_灰`，底页身份掉成 unknown，
    #    flow 没有任何出路  卡死。这是所有「领取面板」的共性：领完之后
    #    屏上只剩灰键+叉叉。带叉叉是关键 —— mail 那种**整页**的领取列表
    #    没有弹窗叉叉，不会被它抢走。
    Sig("claim_panel", need=[V.CLOSE_X],
        any_of=V.CLAIM_ACTIVE + V.CLAIM_DONE,
        # 2026-08-12 live 抓到的「咖啡厅一打开面板马上就关」的**真凶**:
        #    claim_panel 是 **overlay（优先于页面身份）**，只要"有叉叉 + 任一
        #    黄色领取键"就抢身份 —— 而**邀请学生列表里五个黄色「邀請」键**
        #    正好命中  邀请面板被当成"可领取面板"，一看"没有可领的东西"
        #    就 `tap 弹窗叉叉` 叉掉，然后 on_cafe 又去点邀请券，
        #    **开叉开叉** 无限循环（用户原话：「打开了然后关闭，
        #    又在滑动，打开又关闭，整个过程是混乱的」）。
        #    连带把滑动也甩到咖啡厅背景上（面板已经没了），
        #    以及 `jit_drop` 飙高（邀请键在新帧上真的消失了）。
        #     forbid 邀请键：**邀请面板不是领取面板**。
        forbid=[V.CANCEL, V.CAFE_INVITE],
        priority=14, overlay=True,
        note="可关闭的领取面板：有黄键就领，全灰就叉掉"),

    # 掃蕩完成结算面板的奖励动画期：屏上只有 SKIP（0.99）+叉叉，確認还没出来。
    # 08-09 实锤：42 发扫荡的结算停在这一帧，全 bot 没人认识 skip键  no-op 卡住。
    Sig("sweep_results", need=[V.BATTLE_SKIP], priority=13, overlay=True,
        note="结算奖励动画：点 SKIP 直达总账（永远安全）"),

    Sig("ack_dialog", need=[V.CONFIRM],
        forbid=[V.CANCEL, V.CONFIRM_GREY, V.SQUAD_AUTO_EDIT],
        priority=12, overlay=True,
        note="单键确认框（只有確認、没有取消）= **通知/结果**，点掉即可。"
             "安全性靠金钱闸兜底：真是购买框的话 act/money.py 会先拦成 halt。"
             "forbid `自動編輯按鈕`：编队的快速編輯面板也是『確認无取消』，"
             "被当通知框抢点確認会把**旧阵容原样提交**（自動还没点）——"
             "有 自動 键在场 = 编辑面板，交回 formation 的子链处理"),

    # ── 「在某个设施里，但不知道是哪个」（优先级极低，任何具体页都盖过它）──
    # 这是从老 `brain/screens.py` 搬过来的 FACILITY_MARKERS，我第一版漏了。
    # 全语料实证：`回大厅按钮`+`返回键` 同时在场，在**每个设施页 ≈1.0**，
    # 在大厅 / MomoTalk / 剧情播放上 **0.0**。
    # 它的价值不是"多认一个页面"，而是把「没登记的页面」从 UNKNOWN(no-op，
    #    会永久卡死) 变成一个**有明确退出动作**的具名页面 —— 2026-08-08 实测：
    #    社团内部没登记  unknown  兜底 no-op  后面 9 条 flow 全部空转到超时。
    Sig("facility", need=[V.HOME, V.BACK], priority=4,
        note="在某个设施内部（具体哪个不知道）。有退出控件，能安全走人"),

    # ── 空屏（优先级最低，任何一个检出都能盖过它）────────────────────
    Sig("blank", pred=lambda o: len(o.boxes) == 0, priority=1,
        note="**一个框都没有**。三种可能：大厅 UI 被收起（点背景会收，"
             "08-08 我一发飞出屏幕的点击就把它收了）全屏过场 加载。"
             "与老代码那个「UI 被隐藏」误判的区别：那次是**转场帧**上"
             "emoticon 已检出、导航 cls 还没渲染 —— 屏上是有框的，UI 其实在，"
             "于是盲点坐标正压在「編輯模式」上。这里要求 **len(boxes)==0**，"
             "屏上什么都没有  也就没有任何按钮可以被误点。"),
]


# `event_quest_list` 需要"至少一行关卡"这个额外结构条件，用代码补
# （Sig 只表达 cls 的存在性，行的存在性得看 入场键/入场键没解锁 任一）。
_EVENT_ROW_CLS = [V.STAGE_ENTER, V.STAGE_ENTER_LOCKED]


@dataclass
class Match:
    page: str
    score: float
    interrupt: Optional[str] = None
    overlay: Optional[str] = None
    sig: Optional[Sig] = None

    def __repr__(self) -> str:
        i = f" !{self.interrupt}" if self.interrupt else ""
        o = f" +{self.overlay}" if self.overlay else ""
        return f"<{self.page}{o} {self.score:.0f}{i}>"


def classify(obs: Observation) -> Match:
    """单帧  (底页, 覆盖层, 打断)。**认不出就是 UNKNOWN，UNKNOWN 对应 no-op。**"""
    interrupt = None
    for sig in sorted(INTERRUPTS, key=lambda s: -s.priority):
        if sig.match(obs) is not None:
            interrupt = sig.id
            break

    pages: List[Tuple[int, float, Sig]] = []
    overlays: List[Tuple[int, float, Sig]] = []
    for sig in PAGES:
        s = sig.match(obs)
        if s is None:
            continue
        # `event_quest_list` 的额外结构条件：屏上得有关卡行（**一行就够**）。
        # 别加"≥2 行"或"至少一关能打" —— 那正是新活动首日崩掉的写法。
        if sig.id == "event_quest_list" and not obs.has(_EVENT_ROW_CLS, CONF):
            continue
        (overlays if sig.overlay else pages).append((sig.priority, s, sig))

    # 先 priority 再 score：越具体的页面越该赢。battle_result 只有 1 个锚点，
    #    但它比 lobby 的 3 个锚点更能说明"现在在哪"。拿分数当第一排序键的话，
    #    锚点多的宽泛页面会把具体页面盖掉。
    overlays.sort(key=lambda c: (-c[0], -c[1]))
    ov = overlays[0][2].id if overlays else None
    if not pages:
        return Match("unknown", 0.0, interrupt, ov)
    pages.sort(key=lambda c: (-c[0], -c[1]))
    _, sc, sig = pages[0]
    return Match(sig.id, sc, interrupt, ov, sig)


def describe(page: str) -> str:
    for sig in PAGES + INTERRUPTS:
        if sig.id == page:
            return sig.note or sig.id
    return "（未登记）"
