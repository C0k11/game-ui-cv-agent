# -*- coding: utf-8 -*-
"""EventQuestSkill — 活动 AP 规划器 (2026-07-08, 机制全 live 实锤后落码).

用户 AP 分配铁律 (2026-07-08):
  ①先把「活動點數」关(尾关)刷满里程碑(每200点一档全大件, 10000pt 档有青輝石)
  ②商店货币关(尾部前 3 关)先各用加成队真打一次解锁 Bonus
  ③点数满 15000 后 AP 才灌货币关
  ④活动 > 双倍三倍 > 正常关 (编排上本 skill 排在 special_sweep 之前)

核心机制 (live 实验 + kamigame 印证, 2026-07-08):
  - 扫荡 Bonus = 该关「最高加成通关记录」。编队屏挂加成队≠生效; 必须真出击
    通关一次, 之后扫荡永久按该记录结算 (ドロップ数の最大値は引き継がれる)。
  - 判据 cls = 「活动关卡产出额外加成」(popup 内, 解锁前 0 检出 / 解锁后 0.95+)。
  - 自动编队按「当前关」优化且覆盖共享 2 部队 (Q10 优化后同队对 Q11 只 55%)
    → 每个关解锁前都要重跑 快速編輯→自動→確認。
  - 405 banner 轮播歧义: hub 左上 banner 槽多活动轮播, 点进去必须验证落点;
    错活动 → (顺手领免费成就) → back → 重试, cap 3 次。
  - 落在 Challenge tab → 固定位切 Quest tab (0.635,0.151, 2026-07-07 实测)。

Money (⛔铁律): 全程只走 入场/编队/出击/MAX/扫荡开始/确认/叉/back — 无购买路径。
  - 扫荡确认框 gate: body(y>0.12) 检到 青辉石 → 取消 + abort (fail-closed 黑名单
    backstop; 正向源头闸 = MAX_可点击 才扫, MAX_灰=AP不足 天然不触发買AP框)。
  - 商店购买是独立 skill (event_shop), 本 skill 绝不进商店。

Flow:
  enter    lobby → 任务大厅入口; hub → 405(距离结束还剩); 无405 → done(no event)
  verify   落点校验: quest列表特征(入场键+关卡得星) / Challenge tab→切Quest /
           错活动(如情人節約定成就页: 全部领取_黄)→顺手领→back 重试
  survey   列表滑到底, 自底向上开尾部 N 关 popup, 记录 Bonus 解锁状态
  unlock   未解锁关: 任務開始→编队(2部队→快速編輯→自動→確認→%读数)→出击
           → battle(AUTO gate)→结算→回列表
  points   尾关(点数关) MAX 扫循环, 直到 活動點數≥目标 或 MAX灰(AP不足)
  currency 点数满后: 货币关(倒数第2起轮流) MAX 扫到 AP 尽
  tasks    底栏 活动任务 红点门控 → 全部领取/领取 → back
  close    back → hub, done on hub
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from brain.skills.base import (
    BaseSkill, ScreenState, YoloBox,
    action_click, action_click_box, action_wait, action_back, action_done,
    action_swipe, action_swipe_tap,
)
from brain.skills import ui_classes as UC

_CLS_CONF = 0.30
_WEAK_CONF = 0.20          # 活动皮肤弱类 (活动商店/活动任务/活动quest_已选择) 地板
_ENTER_MAX = 24
_VERIFY_RETRY_MAX = 10     # 405 轮播重试上限 (双相位修正后期望 ≤3 轮命中)
_PHASE_MAX = 18
# ⚠`_PHASE_MAX` 是 **tick** 不是秒 —— 实测 tick 速率 0.15~2.3 s/tick, 同一个 "18"
# 落在 2.7s ~ 41s 之间。凡"等某个东西出现"的地方一律另用墙钟, 别复用它。
# ⛔tick-vs-墙钟家族(2026-07-28): 上面那行注释早就写明了病, 但 enter/verify/
# unlock/tasks 四处判据还在裸用 tick。全部改「帧数 AND 墙钟」合取(×1.6 等效):
_ENTER_MAX_SEC = 38.4      # 24×1.6 — enter 等的是 3.2s/张的 banner 轮播
_PHASE_MAX_SEC = 28.8      # 18×1.6 — verify/milestone/tasks 等整场景加载+领奖链
_UNLOCK_QUEST_SEC = 420.0  # unlock **每关**墙钟(战斗 300 + 编队/结算 120),
                           # 旧 518 tick 总和=78-130s, 第一场解锁战打一半就开枪
_MS_CLAIM_RETRY_SEC = 6.0  # milestone 領取 after-ack(盖 step 门控节奏)
_MILESTONE_ANCHOR_SEC = 4.0   # 等「獎勵資訊」入口渲染的墙钟窗口
_BATTLE_MAX = 2400         # ⚠纯轮询次数上限(跑飞兜底), 计时权威是
                           # _BATTLE_MAX_SEC 墙钟。旧值 500 在 0.15-0.25s/tick
                           # 下=75-125s < 战斗 70-140s, 比墙钟先开枪 → 战斗中
                           # 途 done(与墙钟修复极性相反, 2026-07-28 agent 扫描核实)。
_SWEEP_ROUNDS_MAX = 30     # 点数期一次 MAX 就把 AP 扫光, 这是保险帽
_BATTLE_MAX_SEC = 300      # 单场战斗墙钟上限(实测活动关 70-90s, 5min 极宽松)
_AP_READ_RETRY_SEC = 6.0   # AP 读不出时的墙钟重试窗(读失败≠没AP, 别把整轮判死)
# survey 墙钟预算(2026-07-25 live 定): 开/关 4 关 popup = 8 次点击, 而活动页背景
# 常驻动画让帧几乎永不"稳定" → 每次点击最坏要等帧稳定门 4s 超时逃生 = 32s 下限。
# 90s 给足余量。⚠**绝不能用 tick 计**: 等待期间 _phase_ticks 照涨, 旧的 36 tick
# 预算全被"等稳定"吃光 → survey 没做完就 timeout → 跳过 sweep → AP 一点没灌。
_SURVEY_MAX_SEC = 90.0
# ⭐时敏目标(hub 活动 banner)的观测新鲜度约定 —— 2026-07-25 用户点破后定死。
# 405(当期)与 474(上期)交替占**同一槽位**, 从不共存。
# "看到目标就点"要成立, **"看到"必须是现在**: 观测年龄必须远小于窗口, 否则
# 你点的是"过去那一帧里的东西", 落到现在的屏幕上就是另一张卡。
# ⭐窗口实测(2026-07-25, ADB 8 连拍 1.55s 间隔, scratchpad/slotstrip.py):
#   紫(上期)/橙(当期)交替, 切换点落在 +3.4~4.9 / +6.4~8.0 / +9.6~11.2 →
#   **每张停留 ~3.2s**(旧注释写的 2-2.5s 偏小, 按小的定更安全但会白等)。
# 全链实测(server [lat] 埋点): 帧龄 14ms + 推理 35ms + 决策 3ms = 看到→决定
#   **52ms**; ADB tap 下发 564ms; cap→tap **616ms** ≈ 窗口的 19%。
# ⇒ 0.35s 门槛留下 3.2-0.35-0.62 ≈ 2.2s 的安全余量, 且够不到只等 ~0.5s。
_BANNER_CYCLE_SEC = 3.2
_BANNER_FRESH_SEC = 0.35
_SWEEP_OPEN_SEC = 12.0     # 掃蕩開始 迟迟不出现才关窗(必须 > _AP_READ_RETRY_SEC,
                           # 否则重试窗跑不完就被关窗旁路抢先判死 —— 840AP 真凶)
_TAIL_QUESTS = 4           # 尾部加成关数量 (有时3有时4, 用户 2026-07-08)
# 进 points 阶段后给这么多秒反复读活動點數; **窗口内一次都不扫荡**。
# ⛔为什么必须是墙钟且必须挡住扫荡(2026-07-28 第一版实锤): 第一版按**扫荡轮次**
# 计数(连续 2 轮读不到才判无点数) —— 而 points 阶段的**第 1 轮就是 MAX 扫荡**,
# 697 AP 一次性花掉 34 次全灌最后一关, 等判出"无点数"转到 Q11 时只剩 17 AP。
# 判定必须发生在**第一次扫荡之前**, 否则修了等于没修。
# 4s: 点数条渲染实测 <1s, 留 4 倍余量; 窗口内是 wait 态, 慢 IO 不挡点击热路径。
_NO_POINTS_SEC = 4.0

# 对话框按钮带 —— 見 `_sweep_flow` 里那段 2026-07-28 的注释。
# 实测(今天全部帧): 对话框 確認 cy = 0.699 / 0.705 / 0.771 / 0.809;
# 被误捡的假阳 cy = 0.961(扫荡面板底部); 顶栏 = 0.03。两头都留足余量。
_DIALOG_BTN_BAND = (0.20, 0.55, 0.82, 0.93)

# 固定位 (live 实测, 帧目检):
_POS_QUEST_TAB = (0.635, 0.151)     # 活动页 Quest tab (cls 活动quest_已选择 仅0.24-0.59)
_POS_TOUCH_CONTINUE = (0.5, 0.90)   # 结算 TOUCH-continue
# digitOCR regions:
_R_POINTS = (0.55, 0.93, 0.82, 0.985)   # quest列表底部 活動點數 "2676/15000"
# 任務資訊 popup 獲得期待獎勵 行的 xN 数字带 (2026-07-11 双帧标定:
# Q09 固定36/Bonus6, Q12 固定36/Bonus35 全中)
_R_SLOT_FIRST_XN = (0.126, 0.780, 0.172, 0.818)   # 首槽固定掉落 xN
# ⛔_R_SLOT_BONUS_XN 是**固定位**, 而 Bonus 道具是**接在固定道具后面**排的 ——
# 槽位随道具数左右漂, 固定位标定天生活不过一次改版/一次 Best Record 刷新。
# 2026-07-25 实锤: 刚给 Q09 打完 Best Record(結算屏白纸黑字 28(+26)→最終 54),
# survey 却读出 `固定x28 vs Bonus x1` 判"烂加成需重打" —— 那个 x1 是第 6 格
# 「稀有」道具的数量, Bonus 已经漂到第 7 格(cx 0.463)去了。再打一次 = 白烧 20AP。
# ⇒ 改**按 cls 锚定**(词表里现成就有 `活动关卡产出额外加成`, 同帧检出 0.97/0.95),
#   固定位只作降级兜底。这就是"先 grep 词表再发明检测"。
_BONUS_TAG_CLS = "活动关卡产出额外加成"
# 数字带相对 Bonus 标签框的外推系数(同帧 6 组参数全读出真值 26, 取最宽容那组)
_BONUS_XN_DY = (3.1, 5.4)      # ×标签高, 自标签 y1 起算
_BONUS_XN_PX = (0.30, 0.60)    # ×标签宽, 分别向左/右扩
_R_SLOT_BONUS_XN = (0.338, 0.780, 0.394, 0.818)   # 兜底: 旧固定位
_R_SQUAD_PCT = (0.005, 0.92, 0.085, 0.99)  # 编队屏左下 活動編輯效果 第一格 "100%"

_POINTS_TARGET_DEFAULT = 15000


def _read_digits(frame, region) -> Optional[str]:
    try:
        from brain.pipeline import run_digit_ocr
        return run_digit_ocr(frame, region)
    except Exception:
        return None


class EventQuestSkill(BaseSkill):
    """活动 quest AP 规划器: Bonus 解锁 → 点数优先 → 货币扫荡 → 领奖。"""

    # no-UI 逃生 (2026-07-09): 405 轮播可能落进模型盲区页(特殊作戰運輸船主页
    # v13 全零检出) — pipeline no-UI 时按 back 回已知页, 本 skill 恢复重试。
    # ⛔2026-07-25 live 实锤必须**限定在入口阶段**: 它是类属性 = 全程武装, 而
    # 战斗一打就是 60-140s **零动作零「加载中」**, pipeline 那三道护栏
    # (亮帧 / 距上次动作<8s / 距加载中<30s) 全部自然过期 → 结算动画那几个
    # ui 域零框帧凑到 3 个就发 back。当场抓到: 「Battle Complete」屏(01:09
    # 通关, 屏上就一个 確認@0.97)上 pipeline 挂起了 `no-UI escape: back`。
    # 那一按会把结算/Best Record 注册流程拆掉 —— 而这一整场真打的唯一目的
    # 就是注册 Best Record。
    # ⇒ 逃生只在**它被发明出来要救的那个阶段**(入口导航 enter/verify)武装;
    #   其余阶段(战斗/编队/扫荡/领奖)一律不逃 —— 那些地方的零框帧是动画,
    #   不是盲区页, 等就对了(no_ba 30-tick abort 兜底仍在)。
    _ESCAPE_PHASES = ("enter", "verify")

    @property
    def no_ui_escape(self):
        return "back" if self.sub_state in self._ESCAPE_PHASES else None

    @staticmethod
    def _parse_farm_stages(spec) -> List[int]:
        """把用户配置的「刷哪几关 + 配比」展开成一个按关号的轮转序列。

        接受三种写法(前端/配置怎么方便怎么来):
            [10, 11]            → 10,11 交替
            {"10": 1, "11": 2}  → 10,11,11 (银币缺口大就多给)
            "10,11,11"          → 同上
        空/None ⇒ 返回 [] = 走自动兜底(尾关轮转)。
        """
        if not spec:
            return []
        out: List[int] = []
        try:
            if isinstance(spec, dict):
                for k, w in spec.items():
                    out += [int(k)] * max(0, int(w))
            elif isinstance(spec, str):
                out = [int(x) for x in spec.replace("，", ",").split(",")
                       if x.strip()]
            else:
                out = [int(x) for x in spec]
        except (TypeError, ValueError):
            return []
        return [n for n in out if n > 0]

    def __init__(self, points_target: int = _POINTS_TARGET_DEFAULT,
                 tail_quests: int = _TAIL_QUESTS,
                 farm_stages=None):
        super().__init__("EventQuest")
        # ⛔2026-07-28 用户实锤:「活动只会刷最后一关」。根因两条, 都在下面修:
        #  ① `_points` 无条件扫 `len(_quests)-1`(最后一关), 只有**点数达标**才
        #     转 `_currency` 轮转 —— 而本期是**无点数活动**(点数永远读不出),
        #     于是 AP 全灌最后一关。今天实测: 22 次全给 Q12=兽爪, 而 memory
        #     event_ops_playbook 记的账是 **兽爪过剩 Q12 该暂停**, 缺口在
        #     Q10 卡带(~6,355) 与 Q11 银币(~10,326)。
        #  ② `_currency_idx` **从头到尾没有 +1 过**, 就算进了轮转也原地打转。
        # ⇒ 显式关号计划优先(用户可在配置/前端直接指定刷哪几关、按什么配比),
        #   没配就自动兜底成尾关轮转(memory 定案:「无点数活动只刷最后三关」)。
        self._farm_stages: List[int] = self._parse_farm_stages(farm_stages)
        # 长跑 skill 的**整体**上限(pipeline.py:2403 `skill.ticks >= max_ticks`
        # → skill.reset() 重来)。⚠与 _BATTLE_MAX 不是一个计数器: 后者是
        # self._battle_ticks(每场战斗归零), 这里是 skill 全程累计。
        # 400 的账: 新活动开荒一轮 = enter/verify(实测峰值 126) + survey(~60)
        # + unlock 4 场解锁战(每场 编队~30 + 战斗~140 + 结算~20 ≈ 190) 760
        # + points 30 轮扫(~240) + currency(~240) + milestone/tasks(~80)
        # ≈ 1500 → 400 必在解锁战中途被砍, reset 后重新 survey(台账保证不会
        # 重复打已解锁的关, 所以不烧 AP, 但白跑一遍)。
        # 7 月实测从没撞上是因为三关早有 Best Record, 全程只扫荡不解锁
        # (skill_ticks 峰值 126, scratchpad/check_maxticks.py)。下个活动开荒必撞。
        self.max_ticks = 1600
        self._points_target = points_target
        self._tail_quests = tail_quests
        self._init_state()

    def _init_state(self) -> None:
        self.sub_state = "enter"
        self._phase_ticks = 0
        self._verify_retries = 0
        # banner 实例记账(pHash): 点错的轮播项拉黑防复点(情人節同为405)
        self._banner_bad_hashes: list = []
        self._pending_hash = None
        # ev_marker(活动商店/任务 cls)命中但一直上不了 quest 列表的计数 —
        # **上期活动(领奖期)主页同样检出这些 cls**(误入午夜派對三连实锤
        # 2026-07-22), 连续 3 次 → 判 wrong event → 拉黑 banner+back。
        self._ev_marker_hits = 0
        # Quest tab 点击后的渲染冷却(tick 数): 冷却中不计 hits 不重点
        self._marker_click_cooldown = 0
        # unlock confirm 阶段的自動保险补点一次性标记(方案C, 每关重置)
        self._auto_insurance = False
        # wrong-event back 后跳过下一次 405 检出。误入根因(20帧@0.45s 实测
        # 钉死 2026-07-22): 轮播 ~2.5s/页, 405@0.97/474@0.93 逐帧交替从不
        # 共存(v14 训练没问题) — 检出405→tap 落屏有 1-2s 延迟, 点下去时轮播
        # 已切到 474 卡。back 后立即再点大概率复现同样时序碰撞 → 跳一次错相。
        self._skip_next_405 = 0
        # 上次落点是否盲区页(no-UI escape 逃生回 hub) — 决定下次 tap 相位修正
        self._blind_landing = False
        # ⭐加成台账(用户 2026-07-11: 本地记录哪些关打过加成, 记过的不再挨个
        # 开 popup 检查; 没打过的活动内一次性补打)
        self._bonus_ledger: dict = {}     # {round(cy,3): True} 只记已解锁
        self._bonus_created = 0.0
        self._load_bonus_ledger()
        # survey 结果: [(cy_of_入场键, bonus_unlocked)] 自上而下
        # L1-②: 每项 = {"num": int|None, "cy": float, "unlocked": bool}
        # num=关号(唯一身份, 跨帧有效) / cy=当帧坐标(只用于点击与帧内去重)
        self._quests: List[Dict[str, Any]] = []
        # 本轮 survey 已看过的关。**两套并存, 用途严格分工**(L1-②):
        #   _surveyed_nums = 关号 → 真身份, 跨帧/跨滚动都成立, 优先用它
        #   _surveyed_cys  = 坐标 → 只在关号读不出时兜底; 仅因为"同一次 survey
        #                    里列表不滚动"这一条前提才勉强成立, **绝不落台账**
        # (2026-07-10 同关反复开合 bug 修: 点过的 cy±0.03 不再点。旧 _survey_idx
        #  索引制病灶: 关 popup 后过渡帧入场键检出不全 → keys[-4:] tail 切片
        #  错位 → 同一关反复开合 + Bonus 状态记错关。)
        self._surveyed_cys: List[float] = []
        self._surveyed_nums: List[Optional[int]] = []
        self._survey_num: Optional[int] = None
        # ⛔2026-07-25 live 崩溃: `_survey_cy` 以前只在"要开 popup 之前"那一行才
        # 首次赋值 —— 而 `survey: popup appeared late` 分支(屏上**已经**有 popup,
        # 比如 pipeline 在 popup 开着时重启)会直接置 _popup_open=True 走去记账,
        # 于是 `self._survey_cy` AttributeError。与 [[money-safety]] 记的
        # `_py_drop_pending` 同一类"从没在 __init__ 初始化"的静默杀手。
        self._survey_cy: Optional[float] = None
        self._popup_wait = 0
        self._survey_swiped = False
        self._survey_t0 = 0.0         # survey 阶段墙钟起点(0=未开始)
        self._unlock_idx = 0
        self._battle_ticks = 0
        self._battle_t0 = 0.0         # 战斗墙钟起点(0=未开打)
        self._ap_fail_t0 = 0.0        # AP 连续读不出的墙钟起点(0=未失败)
        self._popup_stuck_t0 = 0.0    # 掃蕩開始 不出现的墙钟起点(0=未计时)
        self._sweep_rounds = 0
        # 活動點數缓存(见 _points 里的长注释: 读一次要阻塞 770-900ms 抓 ADB 4K 帧,
        # 绝不能每 tick 挡在点击前面)。-1 = 本轮还没读过, 保证首次必读。
        self._pts_cache: Optional[Tuple[int, int]] = None
        self._pts_read_round: int = -1
        self._points_done = False
        self._currency_idx = 0        # 货币关轮转指针 — 按**扫荡轮次**推进
        self._cur_round_mark = -1     # 上次推进指针时的 _sweep_rounds
        self._no_pts_t0 = 0.0         # 「等活動點數读出」墙钟窗口起点(0=未开始)
        # ⛔当前开着的**掃蕩弹窗属于哪一关**。换关时必须先关掉旧弹窗, 否则新
        # quest_idx 只改了日志 label, 实际还在旧关上扫(2026-07-28 用户帧实锤)。
        self._sweep_popup_num: Optional[int] = None
        self._swept = 0
        # confirm sweep 的吞发回滚 flag(与 craft _start_taps 同修法): 上一发
        # confirm 被稳定门吞时, _swept/_sweep_rounds 要回滚再重计。
        self._cs_fired = False
        # 多关轮转的每轮配额计数(加号已点次数, 2026-07-30 均分 AP): 每轮
        # confirm 后归零; 加号被吞时在点击处回滚。
        self._round_plus = 0
        self._last_ap_read: Optional[int] = None   # 最后一次成功读到的 AP(竣工判据)
        self._ap_spend_after_read = False  # 读数之后又扫荡/出击过 → 读数已陈旧
        self._popup_open = False
        self._formation_step = ""     # unlock 子步: '', 'edit', 'auto', 'confirm', 'sortie'
        self._tasks_done = False
        self._milestone_done = False  # 里程碑(獎勵資訊)领奖阶段
        self._milestone_wait_t0 = 0.0  # 等「獎勵資訊」入口渲染的墙钟起点(0=未计时)
        self._ms_claim_fired = False   # milestone 領取 after-ack flag

    def reset(self) -> None:
        super().reset()
        self._init_state()

    # ── helpers ────────────────────────────────────────────────────

    def _set(self, state: str) -> None:
        if state != self.sub_state:
            self.sub_state = state
            self._phase_ticks = 0
            self.mark("phase_wall")    # _phase_ticks 的墙钟版(合取判据用)

    def _read_bonus_xn(self, screen) -> Tuple[Optional[str], str]:
        """读 Bonus 槽的 xN。返回 (数字串|None, 来源标签)。

        主路径 = 按 `活动关卡产出额外加成` cls 框外推数字带(槽位会随道具数左右
        漂, 固定位标定活不长); 检不到标签才退回旧固定位。
        多个 Bonus 槽时取**最左**那个 —— 首个 Bonus 道具与首槽固定道具是同一种
        (实测 Q09: 固定 謎樣的痕跡 x28 / Bonus 謎樣的痕跡 x26, 结算屏 28(+26)=54),
        比例判据必须拿同一种道具比, 否则拿"兽爪 vs 卡带"比是无意义的。
        """
        if screen.frame is None:
            return None, "无帧"
        tags = [b for b in (getattr(screen, "yolo_boxes", None) or [])
                if b.cls_name == _BONUS_TAG_CLS and b.confidence >= 0.5]
        if tags:
            t = min(tags, key=lambda b: b.x1)
            tw, th = (t.x2 - t.x1), (t.y2 - t.y1)
            if tw > 0 and th > 0:
                region = (max(0.0, t.x1 - _BONUS_XN_PX[0] * tw),
                          t.y1 + _BONUS_XN_DY[0] * th,
                          min(1.0, t.x2 + _BONUS_XN_PX[1] * tw),
                          t.y1 + _BONUS_XN_DY[1] * th)
                got = _read_digits(screen.frame, region)
                if got:
                    return got, f"cls锚定 n={len(tags)}"
        return _read_digits(screen.frame, _R_SLOT_BONUS_XN), "固定位兜底"

    def _on_quest_list(self, screen: ScreenState) -> bool:
        """活动 quest 列表页特征: ≥2 入场键 + (关卡得星 或 活动商店/活动任务底栏)."""
        enters = [b for b in (screen.yolo_boxes or [])
                  if b.cls_name == UC.STAGE_ENTER and b.confidence >= _CLS_CONF]
        if len(enters) < 2:
            return False
        # ⛔负特征 (2026-07-09 live 实锤): 特殊作戰運輸船列表也有 入场键+得星_0,
        # 曾骗过 verify → survey 在错活动里跑。「普通关卡选中」= 主线/特殊作戰
        # tab 指示器。⚠2026-07-11 勘误: 活动页选中态 Challenge tab 也会误检成
        # 普通关卡选中@0.97 — 有活动内铁证(奖励资讯/活动quest tab)时不 veto,
        # 交给底栏硬条件判(真·特殊作戰无活动底栏, 本来就过不了)。
        if self.find_cls(screen, "普通关卡选中", conf=0.5) is not None \
                and self.find_cls(screen, [UC.EVENT_REWARD_INFO, UC.EVENT_QUEST,
                                           UC.EVENT_QUEST_SEL],
                                  conf=_WEAK_CONF) is None:
            return False
        # 硬条件: 活动专属底栏 (v13 下 活动商店0.98/活动任务0.96 稳; 弱检出
        # 地板 0.20 兜底)。得星不再单独放行 — 它对特殊作戰无区分度。
        has_bottom = self.find_cls(
            screen, [UC.EVENT_SHOP, UC.EVENT_TASK], conf=_WEAK_CONF) is not None
        # ⛔Challenge tab 假阳性(2026-07-23 实锤): Challenge 列表同样有 ≥2
        # 入场键+活动底栏 — 切 Quest tab 点击被稳定门吞后 verify 假 OK →
        # survey 在 Challenge 3 关上死等 4 keys 超时, 全天 swept=0 两连。
        # 正锚: 活动quest_已选择 在屏(Quest tab 真选中); 兜底: 未选中态
        # 活动quest 不在屏 且 检出 关卡得星_3(Challenge 全 0 星/Story 无星)。
        qsel = self.find_cls(screen, UC.EVENT_QUEST_SEL,
                             conf=_WEAK_CONF) is not None
        qunsel = self.find_cls(screen, UC.EVENT_QUEST,
                               conf=_WEAK_CONF) is not None
        star3 = any(b.cls_name == "关卡得星_3" and b.confidence >= 0.5
                    for b in (screen.yolo_boxes or []))
        return has_bottom and (qsel or (not qunsel and star3))

    def _on_popup(self, screen: ScreenState) -> bool:
        """任務資訊 popup 特征: 扫荡开始 + 任务开始 同屏."""
        return (self.find_cls(screen, UC.SWEEP_START, conf=_CLS_CONF) is not None
                and self.find_cls(screen, UC.TASK_START, conf=_CLS_CONF) is not None)

    def _enter_keys(self, screen: ScreenState) -> List[YoloBox]:
        """列表页所有入场键, 按 cy 升序 (上→下 = 低关→高关)."""
        keys = [b for b in (screen.yolo_boxes or [])
                if b.cls_name == UC.STAGE_ENTER and b.confidence >= 0.5]
        keys.sort(key=lambda b: (b.y1 + b.y2) / 2)
        return keys

    def _read_points(self, screen: ScreenState) -> Optional[Tuple[int, int]]:
        """活動點數 'n/m' from quest-list bottom bar. None = unread."""
        raw = _read_digits(screen.frame, _R_POINTS)
        if raw and "/" in raw:
            a, _, b = raw.partition("/")
            a = "".join(ch for ch in a if ch.isdigit())
            b = "".join(ch for ch in b if ch.isdigit())
            if a.isdigit() and b.isdigit() and int(b) >= 1000:
                return int(a), int(b)
        return None

    # ── banner 实例记账 (⭐感知铁律 2026-07-11: 判断主导=cls, pHash 只做
    # "同一实例防复点"记账辅助) ────────────────────────────────────────
    # hub 轮播实测(v13): 遊戲開發部=405@0.97 / 情人節=405@0.97(复刻, 无关卡,
    # 落成就页) / 鋼鐵大陸=474@0.88(领奖余韵)。405 有两个实例 → 点错落成就页
    # 由 verify 用 cls 证据识别, 该实例 pHash 拉黑, 下次 405 检出先比对。
    _BANNER_BODY_CROP = (0.02, 0.16, 0.16, 0.285)   # hub banner 本体(405条下方)

    def _banner_phash(self, screen: ScreenState, frame=None):
        try:
            import cv2
            fr = frame if frame is not None else screen.frame
            if fr is None:
                return None
            h, w = fr.shape[:2]
            x1, y1, x2, y2 = self._BANNER_BODY_CROP
            crop = fr[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]
            g = cv2.cvtColor(cv2.resize(crop, (16, 16)), cv2.COLOR_BGR2GRAY)
            return (g > g.mean()).flatten()
        except Exception:
            return None

    def _banner_blacklisted(self, screen: ScreenState, frame=None) -> bool:
        cur = self._banner_phash(screen, frame)
        if cur is None or not self._banner_bad_hashes:
            return False
        return any(int((cur != b).sum()) < 40 for b in self._banner_bad_hashes)

    def _blacklist_pending(self, why: str) -> None:
        if self._pending_hash is not None:
            self._banner_bad_hashes.append(self._pending_hash)
            self._pending_hash = None
            self.log(f"banner instance blacklisted ({why}), "
                     f"total={len(self._banner_bad_hashes)}")

    # ── L1-② 列表结构化解析 (2026-07-25) ──────────────────────────────
    # ⛔为什么必须做: 台账主键一直是 **cy**(归一化屏幕纵坐标)。列表一滚动 cy
    # 全变, 台账整个失效 —— data/event_bonus_state.json 落盘的就是
    # {"0.397": true, "0.871": true, ...} 这种东西。"坐标不是身份"。
    #
    # 关号锚定方式(符合感知铁律: YOLO cls 主导定位, OCR 只读数字):
    #   关号数字条 = 已训 cls `关卡得星_3` box **正上方**, x 同宽,
    #   y ∈ [y1 - 2.4h, y1]  (h = 得星 box 高)
    # 实测(scratchpad/qnum_probe2.py, 全 92,855 tick 扫出 **479 帧真活动
    # Quest 列表**, 16 进程并行):
    #   单格读出率 **100.0%** / 整帧全对率 **100.0%** / 错 0
    # 验收判据用**结构自洽**当 ground truth(不靠人工标): 同帧自上而下关号
    # 必须是连续递增整数, 满足才算这一帧全对。参数扫了 5 组, 2.4h 最稳
    # (2.0h 那组会把 08 切成 30 —— 切太紧比切太松危险得多)。
    _QNUM_STRIP_UP = 2.4        # 关号条上边界 = 得星 y1 - 2.4 * 得星高
    _ROW_PAIR_DY = 0.05         # 入场键 ↔ 得星 同行的最大 cy 差(实测 ~0.024)

    def parse_quest_rows(self, screen: ScreenState) -> List[Dict[str, Any]]:
        """把 quest 列表帧解析成**结构化行**, 自上而下。

        返回 [{"num": int|None, "cy": float, "enter": YoloBox,
               "star": YoloBox|None}, ...]
        num=None 表示这一行关号没读出来 —— 调用方必须 fail-closed 处理,
        **绝不允许退回用 cy 当身份**(那正是要消灭的东西)。
        """
        # ⚠行内配对必须收整个得星族(2026-07-25 审计): 未通关行显示的是
        # `关卡得星_0`(已训独立类), 只配 _3 会让开荒期/未通关尾关整行 num=None
        # → 关号锚定静默退回 cy 老路, 恰好败在最需要它的 20AP 解锁场景。
        # (sweep/verify 用 _3 当页面身份锚防 Challenge 假阳性的那些点**不改**,
        # 那是不同用途。⚠_0 的 val 实例=0, 关号条几何只在 _3 行标定过 ——
        # 首个未通关活动期 live 时要盯读出率。)
        stars = [b for b in (screen.yolo_boxes or [])
                 if b.cls_name in (UC.STAGE_STAR_3, UC.STAGE_STAR_0)
                 and b.confidence >= 0.30]
        rows: List[Dict[str, Any]] = []
        for enter in self._enter_keys(screen):
            ecy = (enter.y1 + enter.y2) / 2
            star = None
            for s in stars:
                if abs((s.y1 + s.y2) / 2 - ecy) <= self._ROW_PAIR_DY:
                    if star is None or abs((s.y1 + s.y2) / 2 - ecy) < abs(
                            (star.y1 + star.y2) / 2 - ecy):
                        star = s
            rows.append({"num": self._read_quest_num(screen, star),
                         "cy": ecy, "enter": enter, "star": star})
        rows.sort(key=lambda r: r["cy"])
        # ⭐max_stage 落账(用户 2026-07-30: 币-关映射不 hardcode): 看到的最大
        # 关号=活动 Quest 总关数, planner 用「尾 K 关↔K 币按序」推关。只在
        # 数值增长时写盘(一个活动期最多几次)。
        _nums = [r["num"] for r in rows if r["num"] is not None]
        if _nums:
            self._note_max_stage(max(_nums))
        # ⭐运行时结构自洽校验(2026-07-25 审计): 与离线验收同款判据 — 同帧关号
        # 自上而下必须严格递增。相邻误读(11→12)在 1-30 范围闸内完全合法, 一旦
        # 带错号落账 = 真关整个活动期被跳过加成核对。取最长严格递增子序列,
        # 序外的 num 判误读置 None(fail-closed 回 popup/cy 兜底, 绝不带错号行动)。
        seq = [(i, r["num"]) for i, r in enumerate(rows) if r["num"] is not None]
        if len(seq) >= 2:
            chains: List[List[int]] = []
            for j in range(len(seq)):
                best = None
                for k in range(j):
                    if seq[k][1] < seq[j][1] and (
                            best is None or len(chains[k]) > len(chains[best])):
                        best = k
                chains.append((chains[best] if best is not None else []) + [j])
            keep = {seq[j][0] for j in max(chains, key=len)}
            for i, r in enumerate(rows):
                if r["num"] is not None and i not in keep:
                    self.log(f"⚠关号 {r['num']} 破坏同帧递增序 — 判误读, 置 None")
                    r["num"] = None
        return rows

    def _note_max_stage(self, n: int) -> None:
        """把活动 Quest 最大关号写进 economy 台账(planner 关映射的数据源)。"""
        if n <= getattr(self, "_max_stage_seen", 0):
            return
        self._max_stage_seen = n
        try:
            import json, os
            p = "data/event_economy.json"
            d = (json.load(open(p, encoding="utf-8"))
                 if os.path.exists(p) else {})
            if n > int(d.get("max_stage") or 0):
                d["max_stage"] = n
                json.dump(d, open(p, "w", encoding="utf-8"),
                          ensure_ascii=False, indent=1)
                self.log(f"[economy] max_stage={n} → 台账")
        except Exception:
            pass

    def _read_quest_num(self, screen: ScreenState,
                        star: Optional[YoloBox]) -> Optional[int]:
        """得星 box 正上方的关号(两位数)。读不出 → None(调用方 fail-closed)。"""
        if star is None or screen.frame is None:
            return None
        bh = star.y2 - star.y1
        raw = _read_digits(screen.frame,
                           (star.x1, max(0.0, star.y1 - self._QNUM_STRIP_UP * bh),
                            star.x2, star.y1))
        d = "".join(ch for ch in (raw or "") if ch.isdigit())
        if not d:
            return None
        n = int(d)
        # 活动 Quest 关号实际范围 1-30(尾三关 Q10-12); 越界 = 误读, 宁可 None
        return n if 1 <= n <= 30 else None

    # ── 加成台账 (persistent, data/event_bonus_state.json) ────────────
    _BONUS_STATE_PATH = "data/event_bonus_state.json"

    # ⛔2026-07-25 主键迁移 cy → 关号(L1-②)。旧盘上的 {"0.397": true} 这种
    # **坐标当身份**的记录一律作废: 列表一滚动 cy 就变, 它们既不可信也无法
    # 翻译成关号(当时的滚动位置早没了)。作废的代价 = 最多重扫一次尾关(免费,
    # 因为 survey 本来就会重开 popup 核对 Bonus 状态); 留着的代价 = 台账错位
    # 去解锁错的关(要花 20AP)。所以宁可丢。
    def _load_bonus_ledger(self) -> None:
        try:
            import json, os, time as _t
            if os.path.exists(self._BONUS_STATE_PATH):
                d = json.load(open(self._BONUS_STATE_PATH, encoding="utf-8"))
                created = float(d.get("created", 0))
                if _t.time() - created < 10 * 86400:   # 超10天=新活动, 作废
                    self._bonus_created = created
                    _q = d.get("quests") or {}
                    _kept, _dropped = {}, 0
                    for k, v in _q.items():
                        if not v:
                            continue
                        # 新格式: 纯整数关号字符串("10"); 旧格式: 小数 cy("0.397")
                        if "." in str(k):
                            _dropped += 1
                            continue
                        try:
                            _kept[int(k)] = True
                        except ValueError:
                            _dropped += 1
                    self._bonus_ledger = _kept
                    if _dropped:
                        self.log(f"⚠台账里 {_dropped} 条是旧的 cy 主键(坐标当身份) "
                                 f"— 已作废, 本轮重新核对")
        except Exception:
            self._bonus_ledger = {}

    def _mark_bonus_done(self, num: Optional[int]) -> None:
        """按**关号**落账。num=None(关号没读出) → 不落账, 宁可下次重扫。"""
        if num is None:
            self.log("⚠关号未读出 → 不落台账(绝不退回用 cy 当身份)")
            return
        self._bonus_ledger[int(num)] = True
        try:
            import json, time as _t
            if not self._bonus_created:
                self._bonus_created = _t.time()
            with open(self._BONUS_STATE_PATH, "w", encoding="utf-8") as f:
                json.dump({"created": self._bonus_created,
                           "quests": {str(k): True
                                      for k in self._bonus_ledger}}, f)
        except Exception:
            pass

    def _bonus_recorded(self, num: Optional[int]) -> bool:
        """关号已落账? 关号未知 → False(当没做过, 重扫是免费的; 误判做过要花 AP)。"""
        return num is not None and int(num) in self._bonus_ledger

    def _pyroxene_in_body(self, screen: ScreenState) -> bool:
        """money backstop: 对话框 body(y>0.12) 出现青辉石 = 绝不是纯AP扫荡框."""
        for b in (screen.yolo_boxes or []):
            if b.cls_name == UC.TOPBAR_PYROXENE and b.confidence >= 0.20:
                if (b.y1 + b.y2) / 2 > 0.12:
                    return True
        return False

    # ⛔⛔ 2026-07-11 事故修复(30青辉石买AP, 逐帧取证 run_144712 t30):
    # 「購買AP」框上 v13 对 body 青辉石**完全漏检**(黑名单gate盲) → 需要
    # 结构白名单: 纯AP扫荡确认框(通知) body 只有 取消/确认/叉(t56实锤);
    # 購買AP框 body 必有 数量stepper(MIN_灰色/加号/MAX_可点击@0.96)+
    # 体力图标@0.92(0.820,0.680)。确认框上下文里检到任一 = 购买框, 取消。
    _BUY_DIALOG_MARKERS = ("MAX_可点击", "MIN_灰色", "MIN_可点击", "加号",
                           "减号", "加号灰色", "减号灰色")

    def _dialog_is_purchase(self, screen: ScreenState) -> bool:
        """确认+取消同屏语境下: body 出现数量 stepper / 青辉石 = 购买框。

        ⛔⛔**体力通道已删除(2026-07-26, 全语料实测证伪)**。
        旧码是 `体力 in body and conf>=0.60 → 购买框`, 注释宣称
        "纯AP框体力@0.38 vs 真購買AP框@0.92 干净分离" —— **那个标定是错的**,
        样本太少。全语料重测(分母=798 帧"确认+取消同屏", 即本判据的生效语境):

          只靠体力通道判出购买框的帧 = **6**, 逐条人审 **6/6 全是纯AP扫荡确认框**
          (reason 全是 confirm sweep (pure AP) / confirm special sweep (AP)),
          **一个真购买框都没有**;
          这 6 帧的体力框 x1 全部 = **0.778-0.779**(同一位置), conf 0.67~**0.92**;
          而两帧真購買AP框(run_20260615_182657/t18, run_20260711_144712/t30,
          正是两起 30 青辉石事故现场)的体力 x1 = **0.808 / 0.816**, conf 0.77/0.92。
        ⇒ conf 分布**完全重叠**(纯AP框能到 0.92, 真框有 0.77), 阈值分层不可能成立;
          那个"体力"根本是**弹窗外面**底层扫荡面板的 `AP→AP` 标签
          (目检 run_20260711_144712/t30: 弹窗右界≈0.74, 体力在 0.82 = 框外)。
        ⇒ 体力通道 = **零区分度 + 100% 误伤**, 删掉是纯收益:
          真购买框召回 2/2 不变(两帧都被 stepper 抓住), 误伤 6→0。
        实锤代价: 2026-07-25 晚它误杀了一轮 240AP 扫荡(12 次)。

        保留的两条通道都正交且实测有效:
          ① stepper(MIN/MAX/加减号) 在 body —— 真購買AP框 @0.96/0.97 稳定检出,
             纯AP确认框 **零检出**(dim 盖住底层 popup)。这是最可靠的购买框铁证。
          ② 青辉石 在 body —— 常漏检(小图标压深色价格条), 但**检到即铁证**,
             阈值保持 0.20 最大灵敏。
        """
        if self._pyroxene_in_body(screen):
            return True
        for b in (screen.yolo_boxes or []):
            cy = (b.y1 + b.y2) / 2
            cx = (b.x1 + b.x2) / 2
            if cy <= 0.12 or b.confidence < 0.20:
                continue
            # ⛔stepper 必须在**确认弹窗横向范围内**(2026-07-31 配额首考当场
            # 误杀): 上面"纯AP框 stepper 零检出"是 MAX 时代**灰态**的实测 —
            # 配额模式下面板 stepper 保持亮态, 隔着 dim 层照样 0.98 检出
            # (实锤帧: MAX_可点击 cx0.84/加号 cx0.79 从弹窗右侧露出 → 纯AP
            # 80AP×4 扫荡框被 cancel)。历史两帧真購買AP框的弹窗右界≈0.74,
            # 其 stepper 全在弹窗中央 — cx<0.72 内的 stepper 才是购买框铁证,
            # 面板 stepper(cx≥0.78)排除。青辉石通道不加此限(检到即铁证)。
            if b.cls_name in self._BUY_DIALOG_MARKERS and cx < 0.72:
                return True
        return False

    # ── 竣工判据 ─────────────────────────────────────────────────────────
    def exit_report(self):
        """活动的竣工判据 = AP 灌完了没(单关 20AP, 所以 <20 才叫干净)。

        ⛔2026-07-25 那次 840 AP 原地报废, skill 出口报的是 "complete" ——
        流程确实走完了, 活一点没干。两件事必须分开报。"""
        _ap = self._last_ap_read
        _swept = getattr(self, "_swept", 0)
        if _ap is None:
            return ("UNKNOWN",
                    f"AP 从未成功读出(本轮 swept={_swept}) — **不知道**灌完没")
        if _ap >= 20:
            # ⚠陈旧读数降级(2026-07-25 审计): AP 读点在掃蕩開始/出击**之前**,
            # rounds-cap/phase-timeout 出口不再重读 — 读数后一轮 MAX 扫最多
            # 花 ~240AP 都不反映。读数后有过消耗就不许拿它报 LEFTOVER。
            if getattr(self, "_ap_spend_after_read", False):
                return ("UNKNOWN",
                        f"最后读数 AP={_ap} 但之后还有扫荡/出击消耗未反映 — "
                        f"真实剩余未知, 本轮 swept={_swept}")
            return ("LEFTOVER",
                    f"还剩 {_ap} AP 没灌进活动(≥20 = 至少还能扫一次), "
                    f"本轮 swept={_swept}")
        return ("CLEAN", f"AP={_ap} <20 单次成本, 本轮 swept={_swept}")

    def _purchase_veto(self, screen: ScreenState, where: str):
        """确认键 点击前的**通用**购买框否决闸。返回非 None 就必须原样 return。

        ⛔2026-07-25 全仓金钱审计实锤: `_dialog_is_purchase` 只挂在 _sweep_quest
        的两处(993/1003), 而 unlock 链上三处 確認 点击(编队確認 / battle settle /
        settle)**一个闸都没有** —— 与 schedule.py 那起 30 青辉石事故完全同型:
        上层把「購買AP」框的確認当成结算報告的確認点掉。而 unlock 的 battle 子步
        在长达 _BATTLE_MAX_SEC=300s 的窗口里持续武装 `find_cls(BTN_CONFIRM,0.6)`,
        期间冒出来的任何確認都会被收下 —— 購買AP 框的確認实测 0.961, 稳稳进门。

        误伤实测(scratchpad/fp_scan.py, 全 790 run 语料):
          编队/出击屏 2734 帧(其中 272 帧確認在屏) → 命中 **0**
          unlock 链真实点击帧(confirm auto-formation 2 + 出击 4) → 命中 **0**
        真购买框侧: run_20260711_144712/t30 stepper@0.96 + body体力@0.92 稳定命中
        (那一帧 body **青辉石一个都没检出** —— 所以绝不能只靠青辉石黑名单)。
        代价方向: 误伤 = 少解锁一关加成(策略损失, 明天可补); 漏检 = 花青辉石。
        """
        if not self._dialog_is_purchase(screen):
            return None
        self.log(f"⛔ 购买框结构出现在 {where} — 绝不点確認, 取消/退出")
        cancel = self.find_cls(screen, UC.BTN_CANCEL, conf=0.5)
        self._formation_step = ""     # 放弃本关解锁, 不留半截子步
        self._set("close")
        if cancel is not None:
            return action_click_box(cancel, f"PURCHASE DIALOG @{where} — cancel!")
        return action_back(f"PURCHASE DIALOG @{where} — back!")

    def _read_ap(self, screen: ScreenState) -> Optional[int]:
        """topbar AP 读数(体力icon cls锚定 + 右侧数字strip, span 0.06 只读
        斜杠前; 主tick帧=ADB干净帧)。读不出 → None(调用方 fail-closed)."""
        # ⛔2026-07-25 实锤(840 AP 全废的直接原因): 旧码 find_cls 不带 region,
        # 而 find_cls 取的是**全屏 conf 最高**那个框。掃蕩弹窗一开, 对话框体内
        # 的两个体力图标(cy 0.506/0.681)就跟顶栏那个抢锚点 —— 实测体内 0.94 >
        # 顶栏 0.93, 于是选中体内的, 紧接着撞上下面 cy>0.12 的门直接 return None。
        # 表现: 弹窗没开时读得出(t041-43), 弹窗一开就"AP 连续 7.7s 读不出 →
        # fail-closed 收工", 840 AP 原地报废。
        # 修: 锚点直接约束在 topbar 带内, 不再让体内图标参与竞争。
        icon = self.find_cls(screen, UC.TOPBAR_AP, conf=0.5,
                             region=(0.0, 0.0, 1.0, 0.12))
        if icon is None or screen.frame is None:
            return None
        raw = _read_digits(screen.frame,
                           (icon.x2 + 0.002, max(0.0, icon.y1 - 0.012),
                            icon.x2 + 0.062, icon.y2 + 0.012))
        if not raw:
            return None
        num = ""
        for ch in raw:
            if ch.isdigit():
                num += ch
            elif num:
                break
        if not num:
            return None
        self._last_ap_read = int(num)   # 竣工判据用: 最后一次**成功**读到的 AP
        self._ap_spend_after_read = False   # 新读数 → 陈旧标记清零
        return self._last_ap_read

    # ── tick ───────────────────────────────────────────────────────

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1
        self._phase_ticks += 1
        handler = getattr(self, f"_{self.sub_state}", None)
        if handler is None:
            return action_done(f"event_quest unknown state {self.sub_state}")
        return handler(screen)

    # ── enter / verify ─────────────────────────────────────────────

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        """⭐感知铁律(2026-07-11): cls 主导 + 极端事件驱动 —
        目标 cls(405)出现→立即点; 加载中→wait; 无锁定→wait(轮播在转)。"""
        if self._phase_ticks > _ENTER_MAX and self.since("phase_wall") > _ENTER_MAX_SEC:
            return action_done("event_quest enter timeout")
        # already on the quest list? (re-entry / retry path)
        if self._on_quest_list(screen):
            self._set("survey")
            return action_wait(300, "on event quest list")
        if screen.is_loading():
            return action_wait(400, "加载中 loading — wait")
        # 目标 cls: 405 距離結束還剩 = 进行中活动 banner(hub 轮播实测 0.97)。
        # 474 距離獎勵獲得結束 = 领奖余韵, 不点。情人節同为 405 → 点错由
        # verify 用 cls 证据识别并 pHash 拉黑该实例, 此处先比对黑名单。
        # ⭐帧龄(2026-07-11 工业级链路): 主 tick 帧龄 ~2.2s vs 轮播 2.6s/项
        # → "帧上的项"和"点到的项"错位率 ~85%(五连点全落错实锤)。高频
        # DXcam 线程检出(帧龄≤0.5s)可用且新鲜时优先, 判定/pHash/点击
        # 坐标全用同一新鲜源。
        import time as _time
        fresh = getattr(screen, "fresh_boxes", None)
        fresh_frame = getattr(screen, "fresh_frame", None)
        # ⚠fresh_ts 时基 = server 高频线程的 perf_counter(同进程同时基),
        # 不是 epoch time.time()(2026-07-11 踩坑: 混用 age 恒大 → 永远回退)
        fresh_age = _time.perf_counter() - getattr(screen, "fresh_ts", 0.0)
        tick_banner = self.find_cls(
            screen, [UC.EVENT_END_LEFT, UC.EVENT_REWARD_END], conf=0.5)
        banner = None
        src_frame = None
        src_tag = ""
        # ⛔⛔时敏目标的"新鲜"必须按**目标自己的窗口**定义(2026-07-25 用户点破:
        # "banner轮换有两秒, 怎么可能来不及点… 我要的是逻辑上不打架的看到目标就点,
        # 就和人类一样")。
        # 旧码 `fresh_age < 1.2` —— 通道名叫 fresh, 却接受 **1.2s 前**的数据, 而
        # banner **2s 一换**: 观测本身已用掉 60% 窗口, 再加推理+tap 延迟必然撞换页。
        # 人类是"现在看→现在点"(视觉延迟 ~50ms); bot 却拿 1.2s 前的画面点现在的屏。
        # 实锤: tick_0775 帧上 405@0.96 → 开火 → 0777(1s后)同坐标已是 474@0.70,
        # 最终进了「秘密的午夜派對」。
        # ⇒ 阈值改成窗口的一小部分。够不到就**等下一次新鲜观测**, 绝不拿旧数据开火
        #   —— 等一拍最多损失 ~0.5s, 而误入的代价是整个 verify→back→rescan 循环。
        # ⛔2026-07-25 实测订正: 旧码(含我昨天写的注释)**假设** fresh 通道一定比
        # 主 tick 帧新, 于是只在 fresh 上开火。埋点数据推翻了这个假设 ——
        #   主 tick 帧(scrcpy) 帧龄中位 13.8ms / ADB 兜底 1495ms
        #   fresh 通道         帧龄中位 399ms(最大 3366ms)
        #   两者同时可得的 32 个 tick 里, **17 个(53.1%) fresh 反而更旧**。
        # 原因: overlay 关掉后高频线程降到 2 FPS(app.py 低功耗模式), 而主 tick
        # 每次都现抓一张 scrcpy 帧现推理。
        # ⇒ 不再"钦定"哪条通道快, 改成**按实测帧龄二选一**, 谁新用谁。
        tick_age = getattr(screen, "frame_age", None)
        tick_usable = (tick_banner is not None
                       and isinstance(tick_age, (int, float))
                       and tick_age < _BANNER_FRESH_SEC)
        fresh_usable = (fresh is not None and fresh_age < _BANNER_FRESH_SEC
                        and not getattr(self, "_fresh_distrust", False))
        # 两条都够新时用更新的那条(帧龄小者); tick 更新就直接走 tick。
        if (tick_usable and (not fresh_usable or tick_age <= fresh_age)
                and tick_banner.cls_name == UC.EVENT_END_LEFT):
            banner = tick_banner
            src_frame = screen.frame
            src_tag = f"tick {tick_age*1000:.0f}ms"
            fresh_usable = False          # 本次走 tick 通道, 别再进 fresh 分支
        if fresh_usable:
            for b in fresh:
                if b.cls_name == UC.EVENT_END_LEFT and b.confidence >= 0.5:
                    banner = b
                    src_frame = fresh_frame
                    src_tag = f"fresh {fresh_age*1000:.0f}ms"
                    break
            # 失信裁决(2026-07-11: DXcam 桌面遮挡/窗口错位时 fresh 内容错 —
            # 主 tick 帧看得见 banner 而 fresh 完全没有 = 源不可信, 2 次即弃)
            if banner is None and tick_banner is not None:
                fresh_any = any(b.cls_name in (UC.EVENT_END_LEFT,
                                               UC.EVENT_REWARD_END)
                                and b.confidence >= 0.4 for b in fresh)
                if not fresh_any:
                    self._fresh_mismatch = getattr(self, "_fresh_mismatch", 0) + 1
                    if self._fresh_mismatch >= 2:
                        self._fresh_distrust = True
                        self.log("fresh source DISTRUSTED (tick sees banner, "
                                 "fresh blind) — pure tick mode")
        # ⛔fresh 不够新时**绝不回落到 tick 帧**: tick 帧允许 _SCRCPY_MAX_AGE=1.0s
        # 陈旧(server/app.py:1252), 比 fresh 通道**更旧** —— 旧码的兜底路径反而是
        # 更差的数据源。时敏目标上这等于"看不清就闭眼点"。改成: 等下一拍。
        # (仅当 fresh 源被判定失信时才允许纯 tick 模式 —— 那是 DXcam 桌面遮挡
        #  一类的源故障, 此时 tick 帧是唯一可信源, 与"数据太旧"是两回事。)
        if (banner is None and not fresh_usable
                and fresh is not None
                and not getattr(self, "_fresh_distrust", False)):
            _ta = (f"{tick_age:.2f}s" if isinstance(tick_age, (int, float))
                   else "?")
            _on = (tick_banner.cls_name if tick_banner is not None else "无")
            return action_wait(120,
                               f"banner 观测太旧/非当期 — tick帧龄 {_ta}(帧上={_on}) "
                               f"/ fresh帧龄 {fresh_age:.2f}s, 轮播窗口 "
                               f"{_BANNER_CYCLE_SEC}s, 等新鲜一拍再点")
        if banner is None and not fresh_usable:
            # ⭐轮播时序(2026-07-22 20帧@0.45s 实测钉死, 推翻 0711 5s 模型):
            # 项周期≈**2.5s/页**, 405 与 474 逐帧交替从不共存, conf 稳定分离
            # (405@0.96-0.97 / 474@0.93-0.94 → v14 训练没问题)。可点窗口极窄:
            # 检出405→tap 落屏延迟 >1.5s 就点到切页后的 474 卡。策略=检出即
            # 直点(零人为延迟) + verify wrong-event 自愈闭环兜底。
            # 帧上=474 → wait(直点会落到 474 项自己, 等轮播转到 405)。
            b405 = self.find_cls(screen, UC.EVENT_END_LEFT, conf=0.5)
            if b405 is not None:
                # wrong-event back 后第一次 405 → 跳过一次错开时序相位
                # (轮播 2.5s/页 + tap 延迟 1-2s, 立即再点大概率复现碰撞)。
                if self._skip_next_405 > 0:
                    self._skip_next_405 -= 1
                    return action_wait(2500, "405=刚误入过 — 跳一次错开相位")
                # ⭐检出即直点一次(用户 2026-07-22 定死)。落点=405 气泡下方
                # +0.10 徽章本体(卡整体可点, ev_enter cy+0.10 实证)。⚠轮播
                # 2.5s/页: 检出→tap 延迟 >1.5s 会点到切页后的 474 卡(误入
                # 午夜派對三连根因), 延迟必须压最短 + verify 自愈闭环兜底。
                self._pending_hash = None
                self._blind_landing = False
                self._set("verify")
                bcx = (b405.x1 + b405.x2) / 2
                bcy = (b405.y1 + b405.y2) / 2 + 0.10
                act = action_click(bcx, bcy, "banner tap (405 atomic 零延迟)")
                act["_atomic_no_gate"] = True   # 2.5s 窗口, 同 tick 落屏
                return act
        if banner is not None:
            if src_frame is not None and self._banner_blacklisted(screen, src_frame):
                return action_wait(400, "405=已拉黑实例 — 等轮播下一项")
            # ⭐检出即直点一次(用户 2026-07-22 定死), swipe_tap 相位修正已删。
            self._blind_landing = False
            # pHash 记账只在 fresh 模式有意义(tick 错位下"帧上项"≠"落点项",
            # 拉黑会反噬正确项)
            self._pending_hash = (self._banner_phash(screen, src_frame)
                                  if src_frame is not None else None)
            self._set("verify")
            # ⚠标签必须说清**真实**来源与帧龄 —— 旧码不管走哪条都写 "fresh",
            # 复盘时看到 'fresh atomic' 会以为观测是新的(实测中位 399ms)。
            tag = src_tag or ("tick+1" if src_frame is None else "?")
            act = action_click_box(banner, f"banner tap ({tag} atomic, "
                                           f"帧上={banner.cls_name})")
            act["_atomic_no_gate"] = True       # 2.5s 窗口, 同 tick 落屏
            return act
        hub = self.find_cls(screen, UC.NAV_TASKS, conf=_CLS_CONF)
        if hub is not None:
            return action_click_box(hub, "lobby → task hub")
        # 无锁定: hub 轮播转在 474/卡池项 或 lobby 未识别 — wait 等目标出现;
        # 久等不出(2 个轮播周期+)才 nav_home 重置。
        # 等满一个完整轮播周期(2 页×~5s≈24 tick)再 reset — 8 tick(~3s)等不完
        # 一轮就回大厅重进 = 低效循环(2026-07-22 实锤连续两轮 reset)。
        if self._phase_ticks % 24 == 23:
            return self.nav_home(screen, "no 405 full carousel — reset")
        return action_wait(400, "no target cls — wait (carousel)")

    def _verify(self, screen: ScreenState) -> Dict[str, Any]:
        """落点校验 — ⭐cls 证据制(2026-07-11): 有明确 cls 证据才判,
        转场/无检出/加载中一律 wait(旧版在转场帧 2s 枪毙点对的入口×3)。"""
        if self._on_quest_list(screen):
            self.log("verify OK — on event quest list")
            self._pending_hash = None            # 点对了, 不拉黑
            self._verify_retries = 0
            self._ev_marker_hits = 0
            self._marker_click_cooldown = 0
            self._set("survey")
            return action_wait(300, "verified event")
        if screen.is_loading():
            return action_wait(400, "verify: 加载中 loading — wait")
        # 证据1: 成就页(情人節式复刻) → 顺手领免费 → 拉黑该 banner 实例
        claim_all = self.find_cls(screen, UC.CLAIM_ALL_YELLOW, conf=0.5)
        if claim_all is not None:
            self._blacklist_pending("achievement page (claimable)")
            return action_click_box(claim_all, "claim-all on wrong event (free)")
        grey = self.find_cls(screen, UC.CLAIM_ALL_GREY, conf=0.5)
        if grey is not None:
            self._blacklist_pending("achievement page (all claimed)")
            self._verify_retries += 1
            if self._verify_retries > _VERIFY_RETRY_MAX:
                return action_done("event_quest: carousel retries exhausted")
            self._set("enter")
            self._blind_landing = False   # cls 页秒回 = 快节奏路径
            return action_back("wrong event(领完) → back, rescan carousel")
        got = self.find_cls(screen, UC.GOT_REWARD, conf=0.5)
        if got is not None:
            return action_click(*_POS_TOUCH_CONTINUE, "dismiss reward toast")
        # 证据2前置(2026-07-11 下午 5 连误杀实锤): **选中态 Challenge tab 会被
        # 误检为 普通关卡选中@0.97(0.839,0.151)** — 同视觉模式。活动开出
        # Challenge 关后落地默认停 Challenge tab, 旧判序把点对的入口当特殊作戰
        # 枪毙。页上铁证: 奖励资讯@0.96(0.877,0.898) + 活动quest(未选中Quest
        # tab)@0.73(0.635,0.153) → 已在目标活动内, 点 Quest tab 切页即可。
        ev_marker = self.find_cls(
            screen, [UC.EVENT_REWARD_INFO, UC.EVENT_SHOP, UC.EVENT_TASK,
                     UC.EVENT_QUEST, UC.EVENT_QUEST_SEL], conf=_WEAK_CONF)
        if ev_marker is not None and self._phase_ticks <= _PHASE_MAX * 2:
            # ⛔上期活动陷阱(2026-07-22 三连实锤): 检出405→tap 延迟撞轮播
            # 2.5s 切页 → 点在 474 卡上进了午夜派對, 其主页同样检出 活动商店/
            # 活动任务/Quest tab → 本分支误判"已在目标活动"死循环点 Quest tab
            # (活動期間已結束空页, 永远上不了 quest 列表)。连续 3 次仍没
            # verify OK = wrong event → 拉黑 banner 实例+back 重扫轮播。
            # ⚠hits=Quest tab 切换**尝试**数, 不是 tick 数(2026-07-22 live
            # 假阳性实锤: 点 Quest tab 后列表渲染 ~1s, 3 个 tick(~1s)就把
            # hits 打满 → 把**正确活动**误判 wrong event back 出去)。点击后
            # 冷却 8 tick 等渲染, 冷却中不计数不重点。
            if self._marker_click_cooldown > 0:
                self._marker_click_cooldown -= 1
                return action_wait(300, "Quest tab 已点 — 等列表渲染")
            # ⛔计数只在**上一次动作真落地**后才涨(2026-07-25 live 实锤):
            # 旧码无条件 +1, 而下面返回的「切 Quest tab」点击会被 pipeline 帧稳定门
            # 吞掉(活动页有角色待机动画+对话气泡, 帧几乎永远不"稳定")。
            # 实测日志三连:
            #   suppressed: 'in event, Challenge tab → Quest tab (cls锚定)'   hits=1
            #   suppressed: 'in event → Quest tab (fixed)'                    hits=2
            #   → hits>=3 → back "wrong event(上期领奖页)" → BRINGUP FREEZE 20 ticks
            # **点击一次都没落地, 却记了三次账**, 于是把**当期活动**误判成上期领奖页
            # 退出去 —— 与 [[region-switch-truth]] 的 ARROW_LEFT 同一根因(同一道闸、
            # 同类按钮、第二次犯)。mutate-before-ack 的计数器版本。
            if not self.action_suppressed:
                self._ev_marker_hits += 1
            else:
                self.log("切 Quest tab 被稳定门吞 — 不记账, 重发")
            if self._ev_marker_hits >= 3:
                self._ev_marker_hits = 0
                # ⛔不 pHash 拉黑(2026-07-22 live 实锤): 误入=tap 延迟撞轮播
                # 切页, fresh 分支记的 hash 是**帧上 405 卡=正确卡**, 拉黑会
                # 反噬正确项 → enter fresh 模式永远跳过真活动死锁。只清 hash
                # + skip 一次错开相位, 靠 verify 闭环反复自愈。
                self._pending_hash = None
                self._skip_next_405 = 1     # 跳过下一次 405 检出错开时序相位
                self._verify_retries += 1
                if self._verify_retries > _VERIFY_RETRY_MAX:
                    return action_done("event_quest: carousel retries exhausted")
                self._set("enter")
                self._blind_landing = False
                return action_back("wrong event(上期领奖页) → back, rescan carousel")
            # 冷却同理: 点击被吞时别空等 8 tick, 立刻重发
            if not self.action_suppressed:
                self._marker_click_cooldown = 8
            qtab = self.find_cls(screen, UC.EVENT_QUEST, conf=_WEAK_CONF)
            # ⚠reason 带 "Quest tab" —— 已加进 pipeline `_dedup_click` 的点击豁免
            # 词表(见那里的注释): 切 tab 是**幂等**动作(重复点同一个 tab 无副作用),
            # 与"导航进关卡"不同, 不该受帧稳定门约束; 而活动页背景动画让帧几乎永
            # 不稳定, 不豁免就是死锁。
            if qtab is not None:
                return action_click_box(
                    qtab, "in event, Challenge tab → Quest tab (cls锚定)")
            return action_click(*_POS_QUEST_TAB, "in event → Quest tab (fixed)")
        # 证据2: 特殊作戰列表(普通关卡选中 tab 指示器, 且无任何活动内标记)
        if self.find_cls(screen, "普通关卡选中", conf=0.5) is not None:
            # ⚠不拉黑: 回滑/相位残差下"帧上项"≠"落点项", 拉黑会反噬目标
            self._pending_hash = None
            self._verify_retries += 1
            if self._verify_retries > _VERIFY_RETRY_MAX:
                return action_done("event_quest: carousel retries exhausted")
            self._set("enter")
            self._blind_landing = False   # cls 页秒回 = 快节奏路径
            return action_back("wrong: 特殊作戰 → back, rescan")
        # 证据3: 仍在 hub(hub tile 可见) = tap 落空/轮播翻走 → 直接重扫。
        # ⚠转场等待窗(2026-07-22 实锤): 点击后第一帧转场尚未启动, hub tile
        # 还在 → 秒判"tap missed"回 enter, 而真实转场 ~1s 后才完成 → skill
        # 在活动页里以为自己在 hub 等 405。≥6 tick(~1s) 才允许判。
        if self.find_cls(screen, [UC.HUB_BOUNTY, "学院交流会", "战术大赛"],
                         conf=0.5) is not None:
            if self._phase_ticks < 6:
                return action_wait(300, "hub 仍在屏 — 转场等待窗, 未判 miss")
            self._pending_hash = None            # 没点进任何页, 不拉黑
            self._verify_retries += 1
            if self._verify_retries > _VERIFY_RETRY_MAX:
                return action_done("event_quest: carousel retries exhausted")
            self._set("enter")
            self._blind_landing = False
            return action_wait(300, "still on hub (tap missed) — rescan")
        # 证据4: Challenge tab 落点(活动底栏在, 无入场键) → 切 Quest tab
        if self.find_cls(screen, [UC.EVENT_SHOP, UC.EVENT_TASK],
                         conf=_WEAK_CONF) is not None \
                and not self._enter_keys(screen):
            self.log("landed on Challenge tab — switching to Quest tab")
            return action_click(*_POS_QUEST_TAB, "switch to Quest tab (fixed)")
        # 无证据(转场渲染/模型盲区页) → wait; 超时兜底 back 重扫
        # (墙钟合取: 2.7s 的窗口把「还在加载」和「点错了」混为一谈 → 白跑
        #  一圈还多按一次 back, 把已进对的活动页退掉)
        if self._phase_ticks > _PHASE_MAX and self.since("phase_wall") > _PHASE_MAX_SEC:
            self._verify_retries += 1
            if self._verify_retries > _VERIFY_RETRY_MAX:
                return action_done("event_quest: carousel retries exhausted")
            self._set("enter")
            self._blind_landing = True   # 无证据超时 = 盲区页同型(慢路径)
            return action_back("verify timeout (no evidence) → back, rescan")
        return action_wait(500, "verify: no cls evidence yet — wait")

    # ── survey ─────────────────────────────────────────────────────

    def _survey(self, screen: ScreenState) -> Dict[str, Any]:
        """列表滑到底, 自底向上开尾部 N 关 popup 记录 Bonus 状态."""
        # ⛔墙钟不是 tick(2026-07-25 live 实锤, "tick 当计时器"第三次咬人):
        # 旧码 `_phase_ticks > _PHASE_MAX*2` = 36 tick。survey 要开/关 4 关 popup
        # = 8 次点击, 而**每次点击都被帧稳定门拦到 4s 超时才放行**(活动页背景有
        # 角色待机动画+对话气泡, 帧几乎永不"稳定"; 日志实锤 `action suppressed:
        # survey open quest popup Q10/Q11`)。等待期间 _phase_ticks 照涨 →
        # 36 tick 全耗在等稳定上 → `survey timeout → close` → **skill 根本没进
        # sweep 阶段** → _read_ap 一次都没调用 → 竣工判据报 "AP 从未成功读出",
        # 而 AP 909 一点没灌(840AP 失败模式复发)。
        # ⚠注意 AP 读数本身是好的: 同帧实测 raw='909/24' → 909。别修错地方。
        if not self._survey_t0:
            self._survey_t0 = self.clock()
        _el = self.clock() - self._survey_t0
        if _el > _SURVEY_MAX_SEC:
            self.log(f"survey 墙钟超时 {_el:.0f}s → close (do NOT retry blind)")
            self._set("close")
            return action_wait(300, "survey timeout")
        # ⭐页面丢失自愈(2026-07-11 live: 点入場后 1s 内被弹回 hub, 无我方
        # 动作, 原因未明 — 任何被踢出场景一律重进而不是干等超时吃 swept=0)
        if (self.find_cls(screen, [UC.HUB_BOUNTY, "学院交流会", "战术大赛"],
                          conf=0.5) is not None
                or self.find_cls(screen, UC.NAV_TASKS, conf=0.5) is not None):
            self.log("survey: kicked out to hub/lobby — re-enter")
            self._survey_swiped = False
            self._popup_open = False
            self._set("enter")
            return action_wait(200, "survey: page lost → re-enter")
        if not self._survey_swiped:
            self._survey_swiped = True
            self._survey_settle = 2   # 滚动后等 2 tick 用新鲜坐标 (回弹稳定闸)
            # 向底部滚 = 内容上移 = from 下 to 上 (2026-07-09 live: 方向写反把
            # 列表拖回顶部, tail keys 全点歪 → popup 卡死 stuck20)
            return action_swipe(0.75, 0.72, 0.75, 0.42, duration_ms=400,
                                reason="scroll quest list to bottom")
        if getattr(self, "_survey_settle", 0) > 0:
            self._survey_settle -= 1
            return action_wait(700, "survey: list settling")
        if self._popup_open:
            if self._on_popup(screen):
                # ⭐先停一拍再判(用户 2026-07-11: "点开马上就关, 没好好对比"
                # — 半渲染帧上 EVENT_BONUS 会漏检 → 误判已解锁)。settle 1 tick
                # 用稳定帧判, 判据仍是 cls「活动关卡产出额外加成」(0708 双向
                # 实测: 解锁前 0 检出/解锁后 0.95+; 数字对比=后续升级路径)。
                if not getattr(self, "_popup_judge_settle", 0):
                    self._popup_judge_settle = 1
                    return action_wait(500, "survey: popup settling before judge")
                self._popup_judge_settle = 0
                self._popup_wait = 0
                self._popup_open = False
                # 去重: X 点漏 popup 残留 / late-popup 重入时不重复记账
                if all(abs(self._survey_cy - c) > 0.03
                       for c in self._surveyed_cys):
                    # ⭐数字对比判定(用户 2026-07-11, Q09 实锤: 徽章在场但
                    # Bonus×6 vs 固定×36 = 弱队烂记录, 徽章判定天然不够)。
                    # Bonus xN ≥ 80% 固定 xN 才算加成打满; 读不出→保守按
                    # 徽章判但不记台账(下轮重查)。region 已双帧标定
                    # (Q09 36/6, Q12 36/35 全中)。
                    badge = self.find_cls(screen, UC.EVENT_BONUS, conf=0.5)
                    unlocked = badge is not None
                    numeric_ok = False
                    if unlocked and screen.frame is not None:
                        fn_raw = _read_digits(screen.frame, _R_SLOT_FIRST_XN)
                        bn_raw, bn_src = self._read_bonus_xn(screen)
                        fn = int(fn_raw) if fn_raw and fn_raw.isdigit() else None
                        bn = int(bn_raw) if bn_raw and bn_raw.isdigit() else None
                        if fn and bn is not None:
                            numeric_ok = True
                            unlocked = bn * 5 >= fn * 4
                            self.log(f"survey: 固定x{fn} vs Bonus x{bn}"
                                     f"({bn_src}) → "
                                     f"{'满加成' if unlocked else '烂加成需重打'}")
                    self._quests.append({"num": self._survey_num,
                                         "cy": self._survey_cy,
                                         "unlocked": unlocked})
                    self._surveyed_cys.append(self._survey_cy)
                    self._surveyed_nums.append(self._survey_num)
                    if unlocked and numeric_ok:
                        self._mark_bonus_done(self._survey_num)
                    self.log(f"survey quest#{len(self._quests)} "
                             f"Q{self._survey_num} (cy={self._survey_cy:.3f})"
                             f" bonus_unlocked={unlocked}")
                else:
                    self.log("survey: duplicate popup (already recorded) — close")
                # 关 popup 后 settle 1 tick 再取 keys (过渡帧检出不全是
                # 同关反复开合 bug 的根因之一)
                self._survey_settle = 1
                close = self.find_cls(screen, UC.BTN_CLOSE_X, conf=_CLS_CONF)
                if close is not None:
                    return action_click_box(close, "close survey popup")
                return action_back("close survey popup (back)")
            self._popup_wait += 1
            if self._popup_wait > 4:
                # 点击没开出 popup (过渡帧点歪) → 放弃本次, 回列表重选同关
                self._popup_wait = 0
                self._popup_open = False
                return action_wait(400, "survey: popup never opened — reselect")
            return action_wait(600, "waiting survey popup")
        if self._on_popup(screen):
            # popup 迟到 (开窗动画慢于重选判定) — 续用上次点击的 cy 记账
            # ⛔但"上次点击"必须真的存在: pipeline 在 popup 已开着时启动/重启
            # 时本轮从没点过任何关, _survey_num/_survey_cy 都是 None ——
            # 那样记账等于把一个来路不明的 popup 的加成结论**扣到某个关号头上**,
            # 台账一旦写错, 那关的加成解锁会被永久跳过。fail-closed: 关掉重来。
            if self._survey_num is None or self._survey_cy is None:
                close = self.find_cls(screen, UC.BTN_CLOSE_X, conf=_CLS_CONF)
                self.log("survey: 屏上已有来路不明的 popup(本轮没点过任何关) "
                         "— 关掉重选, 绝不记账")
                if close is not None:
                    return action_click_box(close, "close survey popup")
                return action_back("close survey popup (back)")
            self._popup_open = True
            return action_wait(300, "survey: popup appeared late")
        # ⭐L1-② 结构化行: 每行 = 入场键 + 得星 + **关号**。
        # ⛔坐标与身份的分工写死在这里, 以后别再混:
        #   cy   → 只用来**点击**和**同一帧内去重**(当帧有效, 合法)
        #   num  → 唯一的**身份**(跨帧/跨 run 有效), 台账只认它
        rows = self.parse_quest_rows(screen)
        # 过渡帧检出不全 → tail 错位, 必须看到完整尾部才动手
        if len(rows) < self._tail_quests:
            return action_wait(600, f"survey: partial list ({len(rows)} rows)")
        tail = rows[-self._tail_quests:]
        todo = [r for r in tail if not self._surveyed(r)]
        if not todo:
            # survey 完毕 → unlock 队列 = 未解锁的关 (自底向上: 点数关优先解锁)
            self._quests.sort(key=lambda q: q["cy"])
            self._quests = self._quests[-self._tail_quests:]
            _nums = [q["num"] for q in self._quests]
            self._set("unlock")
            self.mark("unlock_quest")   # 第一关的每关墙钟从进 unlock 起算
            return action_wait(300, f"survey done (tail quests {_nums})")
        r = todo[0]
        _cy, _num = r["cy"], r["num"]
        # ⭐台账命中 = 该关加成已打过, 不再开 popup 复查(用户 2026-07-11)。
        # 关号读不出 → _bonus_recorded 必 False → 老老实实开 popup 复查。
        # 方向保守: 多开一次 popup 是免费的, 误判"做过了"要多花 20AP 去解锁别的关。
        if self._bonus_recorded(_num):
            self._quests.append({"num": _num, "cy": _cy, "unlocked": True})
            self._surveyed_cys.append(_cy)
            self._surveyed_nums.append(_num)
            self.log(f"survey: Q{_num} 台账已记加成解锁 — skip popup")
            return action_wait(200, "survey: ledger hit, next")
        self._survey_cy = _cy
        self._survey_num = _num
        self._popup_open = True
        self._popup_wait = 0
        return action_click_box(
            box=r["enter"],
            reason=f"survey open quest popup Q{_num} (cy={_cy:.3f})")

    def _surveyed(self, row: Dict[str, Any]) -> bool:
        """这一行本轮 survey 过了没。关号已知就按关号(身份), 否则退回 cy
        (**仅**因为同一次 survey 里列表没滚动, cy 在这段时间内还稳)。"""
        if row["num"] is not None and row["num"] in self._surveyed_nums:
            return True
        return any(abs(row["cy"] - c) <= 0.03 for c in self._surveyed_cys)

    # ── unlock (加成解锁: 自动编队 + 真打一次) ────────────────────────

    def _unlock(self, screen: ScreenState) -> Dict[str, Any]:
        # 每关墙钟(2026-07-28): 旧 tick 总和闸在第一场解锁战打到一半就开枪 —
        # 20AP 已花但 settle 没跑到, Best Record 不落账, bot 被丢在战斗屏。
        # since("unlock_quest") 在每关收关处(settle → _unlock_idx+=1)重打点。
        if self.since("unlock_quest") > _UNLOCK_QUEST_SEC:
            return action_done("event_quest unlock timeout (per-quest wall)")
        # ⭐mutate-before-ack 对账(2026-07-22 live 实锤: 任務開始 点击被稳定门
        # 吞(scrcpy 断流 ADB 帧源翻转→帧未稳定), stage 已进 "edit" → 弹窗仍
        # 开却死等编队页, stuck20 差点被叉掉弹窗)。信号只活一个 tick, 入口先
        # 对账; 证据分流: 屏上仍是上一子步锚 cls = 推进那下没落地 → 回滚一级
        # 重打。非推进类点击(切队/AUTO/结算連点)被吞不满足锚条件, 不误伤。
        if self.action_suppressed:
            _s = self._formation_step
            if (_s == "edit"
                    and self.find_cls(screen, UC.TASK_START, conf=0.5) is not None):
                self._formation_step = ""
                self.log("unlock: 任務開始被吞(popup 仍开) → 回滚重点")
            elif (_s == "auto"
                    and self.find_cls(screen, UC.SQUAD_QUICK_EDIT,
                                      conf=_CLS_CONF) is not None
                    and self.find_cls(screen, UC.SQUAD_AUTO_EDIT,
                                      conf=_CLS_CONF) is None):
                self._formation_step = "edit"
                self.log("unlock: 快速編輯被吞 → 回滚")
            elif (_s == "confirm"
                    and self.find_cls(screen, UC.SQUAD_AUTO_EDIT,
                                      conf=_CLS_CONF) is not None
                    and self.find_cls(screen, UC.BTN_CONFIRM, conf=0.5) is None):
                self._formation_step = "auto"
                self.log("unlock: 自動被吞 → 回滚")
            elif (_s == "sortie"
                    and self.find_cls(screen, UC.BTN_CONFIRM, conf=0.5) is not None
                    and self.find_cls(screen, UC.SORTIE, conf=_CLS_CONF) is None):
                self._formation_step = "confirm"
                self.log("unlock: 確認被吞 → 回滚")
            elif (_s == "battle" and self._battle_ticks <= 1
                    and self.find_cls(screen, UC.SORTIE, conf=_CLS_CONF) is not None):
                self._formation_step = "sortie"
                self.log("unlock: 出击被吞 → 回滚")
        # 找下一个未解锁的关
        while (self._unlock_idx < len(self._quests)
               and self._quests[self._unlock_idx]["unlocked"]):
            self._unlock_idx += 1
        if self._unlock_idx >= len(self._quests):
            self._set("points")
            return action_wait(300, "all tail quests bonus-unlocked")
        # ⛔AP 闸(2026-07-11): 解锁=加成队真出击一场(消耗关卡 AP ~20)。
        # AP<20 → 跳过解锁直接 points(台账不记, 明天带 AP 优先补打), 绝不
        # 在 AP 不足时往出击链里走。
        # ⛔2026-07-25 极性对齐(全仓金钱审计 #1, 逐条核实属实): 这个闸原来写的是
        # `if _ap is not None and _ap < 20` = **fail-OPEN** —— AP 读不出(None)
        # 直接放行进出击链。而同一文件 978 行那个孪生闸(掃蕩)对**同一个函数的
        # 同一种失败**写的是 `if ap is None or ap < 20`(fail-CLOSED + 6s 墙钟
        # 重试窗), 两个闸极性相反。_read_ap→None 是已被 live 反复证实的常态
        # (见 _read_ap 上面那段 840 AP 事故注释), 不是理论风险。
        # AP 不足时点 出擊 → BA 弹「購買AP」框 → 下游 battle 子步无条件点走屏上
        # 任何確認(已在 _purchase_veto 补闸, 但防线不能只有一层) = 花青辉石。
        # 修成 fail-closed + 复用 _ap_fail_t0 的 6s 重试窗(读失败≠没 AP, 别一次
        # 抖动就把整个解锁判死 —— 那正是 840 AP 那次的错法)。
        # 实测支撑(scratchpad): quest 列表屏 1787 帧顶栏体力可锚定 96.0%,
        # 6s 窗口足以吃掉 4% 的瞬时漏检。
        if self._formation_step == "":
            _ap = self._read_ap(screen)
            if _ap is None:
                if not self._ap_fail_t0:
                    self._ap_fail_t0 = self.clock()
                _el = self.clock() - self._ap_fail_t0
                if _el < _AP_READ_RETRY_SEC:
                    return action_wait(400, f"unlock: AP 读不出, 重试中 "
                                            f"({_el:.1f}s/{_AP_READ_RETRY_SEC}s)")
                self.log(f"unlock: AP 连续 {_el:.1f}s 读不出 → fail-closed "
                         f"跳过解锁(绝不在 AP 未知时进出击链)")
            else:
                self._ap_fail_t0 = 0.0
            if _ap is None or _ap < 20:
                if _ap is not None:
                    self.log(f"unlock: AP={_ap} <20 无法出击解锁 — 留待明日补打")
                # ⚠清计时器(2026-07-25 审计): 泄漏进 points 阶段会让 sweep 的
                # 第一次读失败拿陈旧 t0 算出 _el>6s, 6s 重试窗被整个跳过 —
                # 单次感知抖动即收工, 复刻 840AP 式整轮报废。
                self._ap_fail_t0 = 0.0
                self._set("points")
                return action_wait(300, "unlock deferred (AP unknown/insufficient)")
        step = self._formation_step
        # 子步机: popup → 任務開始 → 编队 → 快速編輯 → 自動 → 確認 → 出击 → battle
        if step == "":
            if self._on_popup(screen):
                ts = self.find_cls(screen, UC.TASK_START, conf=0.5)
                if ts is not None:
                    self._formation_step = "edit"
                    return action_click_box(ts, "unlock: 任務開始 → formation")
                return action_wait(500, "unlock: waiting 任務開始")
            rows = self.parse_quest_rows(screen)
            if rows:
                _t = self._quests[self._unlock_idx]
                _num, cy = _t["num"], _t["cy"]
                # ⭐L1-② 收益点: **先按关号找**。旧码只有 cy 最近邻 —— 列表
                # 一滚动(或过渡帧复位), 同一个 cy 就落在**另一关**上, 于是拿着
                # 20AP 去解锁了错的关, 而且台账还记成对的那关做过了。
                hit = next((r for r in rows
                            if _num is not None and r["num"] == _num), None)
                if hit is not None:
                    return action_click_box(
                        hit["enter"], f"unlock: open quest Q{_num} (按关号锚定)")
                # 关号通道不可用(本帧没读出/目标关号未知) → 退回 cy 最近邻,
                # 但保留原来的 ±0.04 容差防呆: 目标不在视野绝不点最近邻。
                box = min((r["enter"] for r in rows),
                          key=lambda b: abs((b.y1 + b.y2) / 2 - cy))
                if abs((box.y1 + box.y2) / 2 - cy) <= 0.04:
                    # cy 兜底必须留痕(2026-07-25: 原日志被 _num 门住, 目标关号
                    # 未知时静默退化 — 与 docstring"绝不坐标当身份"自相矛盾却看不见)
                    self.log(f"⚠unlock: 关号通道不可用(target={_num}) — 退回 cy 锚定")
                    return action_click_box(box, f"unlock: open quest cy={cy:.3f}")
                if self._phase_ticks % 3 == 0:
                    return action_swipe(0.75, 0.72, 0.75, 0.42, duration_ms=400,
                                        reason="unlock: re-scroll to bottom")
                return action_wait(600, "unlock: target quest not in view")
            return action_wait(600, "unlock: waiting list")
        if step == "edit":
            sq2_hi = self.find_cls(screen, UC.SQUAD_2_HI, conf=_CLS_CONF)
            if sq2_hi is None:
                sq2 = self.find_cls(screen, UC.SQUAD_2, conf=_CLS_CONF)
                if sq2 is not None:
                    return action_click_box(sq2, "unlock: switch to squad 2")
                return action_wait(600, "unlock: waiting formation screen")
            qe = self.find_cls(screen, UC.SQUAD_QUICK_EDIT, conf=_CLS_CONF)
            if qe is not None:
                self._formation_step = "auto"
                return action_click_box(qe, "unlock: 快速編輯")
            return action_wait(500, "unlock: waiting 快速編輯 btn")
        if step == "auto":
            auto = self.find_cls(screen, UC.SQUAD_AUTO_EDIT, conf=_CLS_CONF)
            if auto is not None:
                self._formation_step = "confirm"
                return action_click_box(auto, "unlock: 自動 (event-optimal)")
            return action_wait(500, "unlock: waiting 自動 btn")
        if step == "confirm":
            # ⭐自動保险(2026-07-24 方案C, workflow实锤[41]): 快速編輯面板上
            # 自動+確認同屏共存 → "自動被吞"回滚锚(BTN_CONFIRM is None)永不
            # 成立, 自動点击被稳定门吞两次 live 实锤跳过自動出击烂队(15%场)。
            # 確認前无条件补点一次自動(自動=重算最优填充, 幂等), 不依赖锚。
            if not self._auto_insurance:
                auto = self.find_cls(screen, UC.SQUAD_AUTO_EDIT, conf=_CLS_CONF)
                if auto is not None:
                    self._auto_insurance = True
                    return action_click_box(
                        auto, "unlock: 自動保险补点 (幂等, 防被吞漏网)")
            conf_btn = self.find_cls(screen, UC.BTN_CONFIRM, conf=0.5)
            if conf_btn is not None:
                _veto = self._purchase_veto(screen, "unlock/编队確認")
                if _veto is not None:
                    return _veto
                self._formation_step = "sortie"
                return action_click_box(conf_btn, "unlock: confirm auto-formation")
            return action_wait(500, "unlock: waiting 確認")
        if step == "sortie":
            sortie = self.find_cls(screen, UC.SORTIE, conf=_CLS_CONF)
            if sortie is not None:
                # 加成% 读数 (>0 才值得打; 读不出保守出击)
                pct_raw = _read_digits(screen.frame, _R_SQUAD_PCT)
                # ⛔2026-07-25 收紧: 这个读数**是个闸**(读到 0 就跳过出击),
                # 而它读得很脏 —— 同帧屏上白纸黑字「90%」, 实测读出 `'06'`
                # (region 0.005-0.085 卡在徽章边上)。差一个字符就会当成 0。
                # 两处 fail-closed:
                #  ① 只认**干干净净的 0**(整串就是 "0"/"0%"), 脏读一律当读不出
                #     → 保守出击(多花 20-30AP), 而不是跳过。
                #  ② **绝不写永久台账** —— 台账按关号永久生效, 一旦把有加成的关
                #     记成"做过了", 那关**再也不会**被解锁(自我修复不了)。
                #     只在本轮内存里标记, 下一轮 survey 用 popup 的真数字重判。
                _clean = (pct_raw or "").strip().rstrip("%")
                if _clean == "0":
                    self.log("unlock: 0% event bonus (raw=%r, 干净读数) — "
                             "本轮跳过出击, **不写台账**(下轮 survey 重判)"
                             % pct_raw)
                    self._quests[self._unlock_idx]["unlocked"] = True
                    self._formation_step = ""
                    return action_back("0% bonus — leave formation")
                if pct_raw and _clean.isdigit() and int(_clean) == 0:
                    self.log(f"unlock: bonus% 读到 {pct_raw!r} 像 0 但不干净 "
                             f"— 当读不出, 保守出击")
                # ⛔点 出擊 前**就地复核 AP**(2026-07-25 审计 #1③): 上面那个闸
                # 只在 _formation_step=="" 评估一次, 之后经过 edit→auto→confirm
                # 四个子步、几十个 tick 才走到这里 —— 真正花 AP 的是这一下点击,
                # 判据就得贴着它。fail-closed: 读不出/不够一律不点, 绝不靠
                # "点了看弹什么框" 来试探(那正是 30 青辉石事故的思路)。
                # 实测: 出击屏 1162 帧顶栏体力可锚定 96.0%(scratchpad/fp_scan 同批)。
                _ap_now = self._read_ap(screen)
                if _ap_now is None or _ap_now < 20:
                    # ⚠计时器只在**读失败**时起表(2026-07-25 审计: 原来读到
                    # <20 的成功读数也起表, 违反字段定义"连续读不出的起点")
                    if _ap_now is None and not self._ap_fail_t0:
                        self._ap_fail_t0 = self.clock()
                    _el = (self.clock() - self._ap_fail_t0
                           if self._ap_fail_t0 else 0.0)
                    if _ap_now is None and _el < _AP_READ_RETRY_SEC:
                        return action_wait(400, f"unlock/出击前: AP 读不出, 重试中 "
                                                f"({_el:.1f}s/{_AP_READ_RETRY_SEC}s)")
                    self.log(f"⛔ unlock: 出击前复核 AP={_ap_now} → 不点出击"
                             f"(fail-closed, 绝不触发購買AP框)")
                    self._ap_fail_t0 = 0.0   # 同上: 别把计时器泄漏给 points 阶段
                    self._formation_step = ""
                    self._set("points")
                    return action_back("unlock: AP 不足/未知 → 离开编队")
                self._ap_fail_t0 = 0.0
                self.log(f"unlock: sortie w/ squad-2 (bonus%={pct_raw!r}, "
                         f"AP={_ap_now})")
                self._formation_step = "battle"
                self._battle_ticks = 0
                self._battle_t0 = self.clock()
                self._ap_spend_after_read = True   # 出击花 AP → 上次读数作废
                return action_click_box(sortie, "unlock: 出击 (bonus run)")
            return action_wait(600, "unlock: waiting 出击")
        if step == "battle":
            self._battle_ticks += 1
            # ⭐墙钟兜底(2026-07-25): tick 数在 ZERO-WAIT 后不再等价于时间, 而
            # 真实 tick 速率**测不出来** —— 7 月 39 个 run 全是 step_mode 逐帧
            # 门控的, per-run 0.65~8.7 s/tick 全是人工停顿的指纹(实测
            # scratchpad/tick_rate.py)。所以别再拿 tick 当计时器: 战斗超时以
            # 墙钟为准, 与 tick 速率无关。_BATTLE_MAX 保留为轮询次数上限。
            _bt0 = getattr(self, "_battle_t0", 0.0)
            if _bt0 and self.clock() - _bt0 > _BATTLE_MAX_SEC:
                self.log(f"battle 墙钟超时 {self.clock() - _bt0:.0f}s")
                return action_done("event_quest battle timeout (wall clock)")
            if self._battle_ticks > _BATTLE_MAX:
                return action_done("event_quest battle timeout")
            # AUTO gate (灰 → 点亮)
            auto_off = self.find_cls(screen, UC.BATTLE_AUTO_OFF, conf=0.5)
            if auto_off is not None:
                return action_click_box(auto_off, "battle: AUTO on (gate)")
            conf_btn = self.find_cls(screen, UC.BTN_CONFIRM, conf=0.6)
            if conf_btn is not None:
                # ⛔这个 find_cls 在长达 _BATTLE_MAX_SEC=300s 的窗口里一直武装,
                # 期间屏上冒出的任何確認都会被当成"战斗结算的確認"点掉。
                _veto = self._purchase_veto(screen, "unlock/battle结算")
                if _veto is not None:
                    return _veto
                self._formation_step = "settle"
                return action_click_box(conf_btn, "battle: settle 確認")
            return action_wait(4000, "battling (AUTO gate armed)")
        if step == "settle":
            conf_btn = self.find_cls(screen, UC.BTN_CONFIRM, conf=0.6)
            if conf_btn is not None:
                _veto = self._purchase_veto(screen, "unlock/settle")
                if _veto is not None:
                    return _veto
                return action_click_box(conf_btn, "settle: 確認 (total reward)")
            if self._on_quest_list(screen):
                _q = self._quests[self._unlock_idx]
                self.log(f"unlock: Q{_q['num']} bonus run DONE")
                _q["unlocked"] = True
                self._mark_bonus_done(_q["num"])
                self._unlock_idx += 1
                self._formation_step = ""
                self._auto_insurance = False   # 下一关重新补点
                self.mark("unlock_quest")      # 下一关的墙钟从这里起算
                return action_wait(500, "back on list after unlock")
            return action_click(*_POS_TOUCH_CONTINUE, "settle: TOUCH continue")
        return action_wait(500, f"unlock: step {step}")

    # ── points / currency sweep ────────────────────────────────────

    def _points(self, screen: ScreenState) -> Dict[str, Any]:
        """尾关(点数关) MAX 扫直到 目标 / AP尽."""
        # fail-closed: survey 没产出任何关就进了 points → quest_idx = -1, 会被
        # _sweep_quest 的 `quest_idx < len()` 守卫放行(负数恒小于长度)然后
        # self._quests[-1] 崩在 tick 里。live 走不到(reset 同时清 _quests 和
        # _surveyed_cys, survey 完成必有条目), 但 replay 台上 302 tick 复现 7 次,
        # 是真实的崩溃路径 — 堵死比留着强。
        if not self._quests:
            self.log("points: _quests 为空(survey 未产出) — 跳过扫荡转 milestone")
            self._set("milestone")
            return action_wait(300, "points: no surveyed quest to sweep")
        # ⛔⛔别把慢 IO 放在点击热路径上(2026-07-26, 帧龄埋点查出来的第二类错):
        # `_read_points` → run_digit_ocr, 而 run_digit_ocr 在**帧宽<3200 时会
        # get_clean_frame() 重抓一张 ADB 4K 帧**(pipeline.py:794-799) —— 主 tick
        # 帧是 scrcpy 1440p(宽 2560), 所以**每次都触发**, 实测阻塞 770-900ms。
        # 实锤 run_20260725_225503/tick_0289: 帧龄 **21.2ms** 一点不旧, 落点上
        # 也确实有 `skip键@0.987` —— 但决策链中途插了这一下 898ms, 等 tap 下发时
        # 掃蕩动画早放完, 那一拍落在结果页的奖励图标上, 弹出道具说明浮窗挡住
        # 確認, skill 卡到 `stuck 20` 才救回来。
        # ⇒ 与感知铁律"目标 cls 出现→立即点, 零人为延迟"直接冲突。
        # 点数只在**扫荡完成后**才变, 每 tick 读纯属浪费 ⇒ 按扫荡轮次 + 墙钟节流。
        # (⚠两个缓存字段都在 _init_state 初始化 — 见 [[money-safety]] 记的
        #  `_py_drop_pending` 从没初始化导致整条守卫静默死掉那种静默杀手。)
        if (self._sweep_rounds != self._pts_read_round
                or self.since("pts_read") > 20.0):
            self._pts_cache = self._read_points(screen)
            self._pts_read_round = self._sweep_rounds
            self.mark("pts_read")
        # ⭐用户显式指定了刷哪几关 ⇒ 点数阶段整个跳过, 直接按计划轮转。
        if self._farm_stages:
            self.log(f"points: 已配置 farm 计划 {self._farm_stages} → 跳过点数阶段")
            self._points_done = True
            self._set("currency")
            return action_wait(200, "farm plan configured → currency")
        pts = self._pts_cache
        if pts is not None:
            self._no_pts_t0 = 0.0     # 读到点数 → 解除"无点数"计时窗
            self.log(f"活動點數 {pts[0]}/{pts[1]}")
            if pts[0] >= min(self._points_target, pts[1]):
                self._points_done = True
                self._set("currency")
                return action_wait(300, "points target reached → currency")
        else:
            # ⛔无点数活动逃生(2026-07-28 用户实锤「只会刷最后一关」的直接根因):
            # 点数读不出时旧码**无条件**继续扫最后一关, 整轮 AP 全灌一关。
            # memory event_ops_playbook 的定案是「无点数活动只刷最后三关」。
            # ⚠⚠判定必须在**第一次扫荡之前**完成 —— 见 `_NO_POINTS_SEC` 那段:
            # 第一版按扫荡轮次判, 结果第 1 轮 MAX 就把 697 AP 花光了。
            if not self._no_pts_t0:
                self._no_pts_t0 = self.clock()
            _el = self.clock() - self._no_pts_t0
            if _el <= _NO_POINTS_SEC:
                # 窗口内**绝不扫荡**, 只反复重读。此处是 wait 态, run_digit_ocr
                # 的 ADB 4K 重抓(~900ms)不挡点击热路径。
                self._pts_cache = self._read_points(screen)
                if self._pts_cache is not None:
                    return action_wait(150, "points: 读到点数了 — 下一 tick 按点数走")
                return action_wait(400, f"points: 等活動點數读出 ({_el:.1f}s)")
            self.log(f"points: {_el:.1f}s 内读不到活動點數 ⇒ 判定**本期无点数活动**"
                     f" → 转尾关轮转(⚠这是'读不出'不是'没有点数'的资源结论; "
                     f"这条反复出现就去查点数条版式/cls 漏检)")
            self._points_done = True
            self._set("currency")
            return action_wait(200, "no-points event → currency rotation")
        return self._sweep_quest(screen, quest_idx=len(self._quests) - 1,
                                 phase_after="milestone", label="points")

    def _currency(self, screen: ScreenState) -> Dict[str, Any]:
        """货币关轮转 MAX 扫。

        优先用**用户显式指定的关号计划**(`event_farm_stages`), 没配就自动兜底成
        尾关轮转。⛔`_currency_idx` 必须**按扫荡轮次**推进 —— 旧码它从初始化到
        永远都是 0, 轮转是死的(2026-07-28 用户实锤「只刷最后一关」的第二个根因)。
        """
        n = len(self._quests)
        if n < 2:
            self._set("tasks")
            return action_wait(200, "no currency quests")
        # 每完成一轮扫荡才转下一关 —— 用 tick 推进会在同一关内乱跳。
        if self._sweep_rounds != self._cur_round_mark:
            self._cur_round_mark = self._sweep_rounds
            self._currency_idx += 1

        if self._farm_stages:
            want = self._farm_stages[self._currency_idx % len(self._farm_stages)]
            idx = next((i for i, q in enumerate(self._quests)
                        if q.get("num") == want), None)
            if idx is not None:
                return self._sweep_quest(screen, quest_idx=idx,
                                         phase_after="milestone",
                                         label=f"farm Q{want:02d}")
            # ⛔关号不在本次 survey 的表里(没解锁/没滚到) —— 报出来别静默降级,
            # 否则"我配了刷 Q10 结果它刷别的"这种事永远查不出。
            self.log(f"⚠farm 计划要 Q{want:02d}, 但本轮 survey 表里没有 "
                     f"{[q.get('num') for q in self._quests]} → 本轮回退尾关轮转")

        # 自动兜底: 尾关轮转(memory 定案「无点数活动只刷最后三关」)。
        # 旧码从 n-2 起算把最后一关排除在外 —— 那是"点数关已经扫饱"的前提下才
        # 成立; 无点数活动没有这个前提, 最后一关同样是货币关, 要一起轮。
        _tail = min(self._tail_quests, n)
        idx = n - 1 - (self._currency_idx % max(1, _tail))
        _num = self._quests[max(0, idx)].get("num")
        return self._sweep_quest(screen, quest_idx=max(0, idx),
                                 phase_after="milestone",
                                 label=f"currency Q{_num:02d}" if _num
                                 else "currency")

    def _sweep_quest(self, screen: ScreenState, quest_idx: int,
                     phase_after: str, label: str) -> Dict[str, Any]:
        """通用 MAX 扫荡子流程 (money-gated)."""
        if self._sweep_rounds > _SWEEP_ROUNDS_MAX:
            self._set(phase_after)
            return action_wait(200, "sweep rounds cap")
        if self._phase_ticks > _PHASE_MAX * 3:
            self._set(phase_after)
            return action_wait(200, f"{label} sweep phase timeout")
        # 结果/确认链
        skip = self.find_cls(screen, UC.BATTLE_SKIP, conf=0.6)
        if skip is not None:
            return action_click_box(skip, f"{label}: skip sweep animation")
        got = self.find_cls(screen, UC.GOT_REWARD, conf=0.5)
        if got is not None:
            return action_click(*_POS_TOUCH_CONTINUE, f"{label}: dismiss reward")
        if self._on_popup(screen):
            # ⛔⛔换关必须先关掉当前弹窗(2026-07-28 用户帧证据实锤):
            # 旧码这里只问"屏上有没有掃蕩弹窗", **从不问这个弹窗是哪一关的**。
            # 于是 points(Q12) → currency(Q11) 切换时, Q12 的弹窗还开着,
            # 本函数拿着 quest_idx=Q11 直接在**Q12 的弹窗**上继续 MAX/掃蕩,
            # **只有日志 label 变成了 Q11**。用户看帧发现是「12 神秘圓圈」,
            # 而我看日志报了"已换到 Q11" —— [[log-is-not-truth]] 第三次。
            # 代价: AP 697→17 全灌 Q12(兽爪, memory 记的账是**过剩**)。
            _want = (self._quests[quest_idx].get("num")
                     if 0 <= quest_idx < len(self._quests) else None)
            _have = self._sweep_popup_num
            if _want is not None and _have is not None and _have != _want:
                self._sweep_popup_num = None
                close = self.find_cls(screen, UC.BTN_CLOSE_X, conf=_CLS_CONF)
                if close is not None:
                    return action_click_box(
                        close, f"{label}: 换关 Q{_have:02d}→Q{_want:02d} — 先关掉当前弹窗")
                return action_back(
                    f"{label}: 换关 Q{_have:02d}→Q{_want:02d} — 关弹窗(无X, ESC)")
            # MAX 可点 → 点; MAX 灰 → AP 不足, 收工
            qmax = self.find_cls(screen, UC.QTY_MAX, conf=_WEAK_CONF)
            qmax_grey = self.find_cls(screen, UC.QTY_MAX_GREY, conf=_CLS_CONF)
            plus_grey = self.find_cls(screen, "加号灰色", conf=_CLS_CONF)
            # ⭐多关轮转不 MAX(2026-07-30 用户: "分配体力把商店搬空"): MAX 一发
            # 把 AP 全灌当前关 → 轮转形同虚设(实录: 909AP 全给 Q11, Q10 卡带
            # 全天零产出, 而商店货架**每日重置**两币每天都要)。多关时每轮固定
            # 配额 = 加号点到 6 次(1+5, ≈90AP), 轮转指针每轮前进 → AP 自然按
            # 配比均分; 单关时保持 MAX(一关吃满没毛病)。加号被稳定门吞时计数
            # 回滚(按物理动作计数, 与 craft _start_taps 同族)。
            _qty_ready = False
            _multi = len(set(self._farm_stages)) > 1
            if _multi and qmax is not None and qmax_grey is None:
                if self.action_suppressed and self._round_plus > 0:
                    self._round_plus -= 1
                if self._round_plus < 5:
                    plus = self.find_cls(screen, "加号", conf=_CLS_CONF,
                                         region=(0.30, 0.20, 1.0, 0.80))
                    if plus is not None:
                        self._round_plus += 1
                        return action_click_box(
                            plus, f"{label}: 轮转配额+1 ({self._round_plus}/5)")
                # 配额满 / 加号不在(已顶格或 AP 上限使加号转灰) → 开扫
                _qty_ready = True
            elif qmax is not None and qmax_grey is None:
                return action_click_box(qmax, f"{label}: MAX")
            if qmax_grey is not None and plus_grey is None and not _qty_ready:
                # MAX 灰但加号亮 = 已顶格待开扫 (MAX 点过)
                pass
            if (qmax_grey is not None and plus_grey is not None) or _qty_ready:
                ss = self.find_cls(screen, UC.SWEEP_START, conf=0.5)
                # ⛔源头闸(2026-07-11 事故修复): 全灰≠可扫 — AP=0 时点掃蕩開始
                # 会弹「購買AP」框(30青辉石实锤)。扫前必读 topbar AP, 读不出
                # 或 <20(单次成本) 一律收工, **绝不试探点掃蕩開始**(fail-closed,
                # 同 special_sweep 0615 教训: 耗尽型扫荡必先读余额)。
                if ss is not None and self._phase_ticks % 3 == 1:
                    ap = self._read_ap(screen)
                    # ⭐区分"读失败"与"真的不够"(2026-07-25 live 实锤: 一次读
                    # 不出就收工, **840 AP 原地作废**)。金钱防线一分不松 ——
                    # 仍然只在读到有效值 ≥20 时才点掃蕩開始; 但读失败属于感知
                    # 抖动, 该在墙钟窗口内重试, 而不是把整轮资源判死。
                    if ap is None:
                        if not self._ap_fail_t0:
                            self._ap_fail_t0 = self.clock()
                        _el = self.clock() - self._ap_fail_t0
                        if _el < _AP_READ_RETRY_SEC:
                            return action_wait(
                                400, f"{label}: AP 读不出, 重试中 "
                                     f"({_el:.1f}s/{_AP_READ_RETRY_SEC}s)")
                        self.log(f"{label}: AP 连续 {_el:.1f}s 读不出 → "
                                 f"fail-closed 收工(绝不碰購買AP框)")
                    else:
                        self._ap_fail_t0 = 0.0
                    if ap is None or ap < 20:
                        if ap is not None:
                            self.log(f"{label}: AP={ap} <单次成本 → 收工")
                        close = self.find_cls(screen, UC.BTN_CLOSE_X,
                                              conf=_CLS_CONF)
                        self._set(phase_after)
                        if close is not None:
                            return action_click_box(
                                close, f"{label}: close popup, AP done")
                        return action_back(f"{label}: AP done → back")
                    # ⛔轮次计数**不在这里加**(2026-07-28 live 守卫拦下):
                    # 掃蕩開始点完只是弹出「要使用NAP掃蕩N次嗎?」确认框, AP
                    # 还没花。旧码在此 +1 → _currency 下一 tick 看到轮次变化
                    # 就把轮转指针推到下一关 → 对着**还没确认的弹窗**走"换关
                    # 先关窗" → 每关只点開始就换走, 一个扫荡都确认不了。
                    # 计数移到 confirm sweep(確認真点出去 = AP 真扣)那里 ——
                    # 与 craft _start_taps/schedule 落账同族: 按物理动作计数。
                    return action_click_box(ss, f"{label}: 掃蕩開始 (round "
                                                f"{self._sweep_rounds + 1}, AP={ap})")
                # ⛔2026-07-25 实锤: 这条旁路**从头到尾没读过 AP**, 却对外宣布
                # "AP exhausted"。8 tick 在 ZERO-WAIT 下 ≈1.6s(server/app.py:1424
                # 非 loading wait 一律压 0.12s), **比同文件 _AP_READ_RETRY_SEC=6.0
                # 还短** —— AP 读不出时的重试返回 action_wait, _phase_ticks 照涨,
                # 重试到第 9 tick 就被这里抢先关窗 = 我的重试修复被直接架空,
                # 840 AP 就是这么没的。
                # 收紧两点: ①掃蕩開始 必须**不在屏**才允许判"开不出来"(在屏说明
                # 还能扫, 只是 AP 没读到) ②改墙钟, 且必须长于 AP 重试窗
                # ③log 不再冒充资源结论(没读过 AP 就不许说 exhausted)。
                close = self.find_cls(screen, UC.BTN_CLOSE_X, conf=_CLS_CONF)
                if ss is None and close is not None:
                    if not self._popup_stuck_t0:
                        self._popup_stuck_t0 = self.clock()
                    _el = self.clock() - self._popup_stuck_t0
                    if _el > _SWEEP_OPEN_SEC:
                        self.log(f"{label}: 掃蕩開始 {_el:.1f}s 未出现 → 关窗收工"
                                 f"(⚠未读到 AP, 不做资源结论)")
                        self._set(phase_after)
                        return action_click_box(close, f"{label}: close popup")
                else:
                    self._popup_stuck_t0 = 0.0
            return action_wait(600, f"{label}: popup settling")
        # 确认框 (要使用NAP掃蕩N次嗎?)
        # ⛔2026-07-28 live 实锤 —— 与 840AP 那次**同一个病**(memory completion_gap:
        # "任何顶栏读数/角落按钮锚定必须带 region"): 这两个 find_cls **没带 region**,
        # 于是全屏 argmax 捡到了一个 `确认键 conf=0.81 @ cy=0.961`(屏幕最底部,
        # 活动扫荡面板上的东西, 不是对话框按钮)。
        # 后果链: 有確認、无取消 → 判成「掃蕩完成框」→ 过结构闸 → 而扫荡面板
        # **自带数量步进器**(MIN/MAX/加减号 @cy0.415, 那时全灰) → `_dialog_is_purchase`
        # 的 "stepper 在 cy>0.12 即体内" 命中 → **判成购买框, 整轮扫荡 back off**。
        # 当时 AP=445, 代价 ≈22 次扫荡。误报方向虽安全, 但"活没干"同样是事故。
        # 定带依据(今天全部帧实测): 对话框 確認 cy = 0.699 / 0.705 / 0.771 / 0.809;
        # 假阳 0.961; 顶栏 0.03。取 0.55~0.93 两头都留足余量。
        # ⚠同型未验: 本文件 unlock/battle 链还有三处无 region 的 BTN_CONFIRM
        #   (`unlock: confirm auto-formation` / `battle: settle 確認` / `settle: 確認`)。
        #   今天没跑到那条链, **没有帧证据就不动**(那条链有 fixture 护着, 盲改风险更大)。
        conf_btn = self.find_cls(screen, UC.BTN_CONFIRM, conf=0.6,
                                 region=_DIALOG_BTN_BAND)
        cancel_btn = self.find_cls(screen, UC.BTN_CANCEL, conf=0.5,
                                   region=_DIALOG_BTN_BAND)
        if conf_btn is not None and cancel_btn is not None:
            # ⛔ money gate(2026-07-11 强化): 青辉石黑名单会漏检(購買AP框上
            # v13 全盲实锤) → 加结构闸: body 有 stepper/体力 = 购买框 → 取消。
            # 纯AP扫荡确认框 body 只有取消/确认/叉(t56 实锤), 误伤=安全重试。
            if self._dialog_is_purchase(screen):
                self.log("⛔ purchase-dialog structure in confirm body — "
                         "CANCEL (fail-closed)")
                self._set("close")
                return action_click_box(cancel_btn, "PURCHASE DIALOG — cancel!")
            if self._cs_fired and self.action_suppressed:
                # 上一发 confirm 被稳定门吞(确认框还在屏) — 回滚计数再重计,
                # 否则一次真确认被记两轮 → 轮转指针多跳一关。
                self._swept = max(0, self._swept - 1)
                self._sweep_rounds = max(0, self._sweep_rounds - 1)
                self.log(f"{label}: confirm 被吞 — 回滚轮次(不算一轮)")
            self._cs_fired = True
            self._swept += 1
            # ⭐轮次在这里推进(确认框確認真发出去 = 这轮扫荡真消费), 见上面
            # 掃蕩開始处的长注释。_currency 据此转下一关。
            self._sweep_rounds += 1
            self._round_plus = 0   # 本轮配额已消费, 下一关面板重新计
            self._ap_spend_after_read = True   # 扫荡花 AP → 上次读数作废
            return action_click_box(conf_btn, f"{label}: confirm sweep (pure AP)")
        # 走到这 = 确认框(取消+確認同屏)不在 → 上一发 confirm(若有)已落地
        self._cs_fired = False
        if conf_btn is not None and cancel_btn is None:
            # 掃蕩完成框 (确认键无取消): tooltip 坑 → 连点两次由 tick 自然完成
            # ⛔同样过结构闸: 取消键偶发漏检时購買AP框会伪装成完成框
            if self._dialog_is_purchase(screen):
                self.log("⛔ purchase-dialog structure (no-cancel frame) — back off")
                self._set("close")
                return action_back("PURCHASE DIALOG suspected — back!")
            return action_click_box(conf_btn, f"{label}: 掃蕩完成 確認")
        # 列表页 → 开目标关 popup
        rows = self.parse_quest_rows(screen)
        # 0 <= 而不是只 < : 负 idx 会从尾部静默取关(空表则直接 IndexError)
        if rows and 0 <= quest_idx < len(self._quests):
            _t = self._quests[quest_idx]
            _num, cy = _t["num"], _t["cy"]
            # ⭐先按关号锚(L1-②); 关号不可用才退回 cy 最近邻 + 容差防呆
            hit = next((r for r in rows
                        if _num is not None and r["num"] == _num), None)
            if hit is not None:
                # 记下这个掃蕩弹窗属于哪一关 —— 上面那道"换关先关窗"闸靠它。
                self._sweep_popup_num = _num
                return action_click_box(hit["enter"],
                                        f"{label}: open quest Q{_num} (按关号锚定)")
            box = min((r["enter"] for r in rows),
                      key=lambda b: abs((b.y1 + b.y2) / 2 - cy))
            if abs((box.y1 + box.y2) / 2 - cy) <= 0.04:
                return action_click_box(box, f"{label}: open quest cy={cy:.3f}")
            if self._phase_ticks % 3 == 0:
                return action_swipe(0.75, 0.72, 0.75, 0.42, duration_ms=400,
                                    reason=f"{label}: re-scroll to bottom")
            return action_wait(600, f"{label}: target quest not in view")
        return action_wait(600, f"{label}: waiting")

    # ── milestone (獎勵資訊里程碑领奖, 2026-07-11 落码; 0710 手动probe验证
    # 过链路: 奖励资讯475@0.96红点→popup→領取獎勵_黄89(一键领所有到达档,
    # +9M信用点实录)→獲得獎勵toast→灰90翻转→叉退回) ────────────────────

    def _milestone(self, screen: ScreenState) -> Dict[str, Any]:
        if self._phase_ticks > _PHASE_MAX and self.since("phase_wall") > _PHASE_MAX_SEC:
            self._set("tasks")
            return action_wait(200, "milestone phase timeout → tasks")
        if self._milestone_done:
            self._set("tasks")
            return action_wait(200, "milestone claimed → tasks")
        got = self.find_cls(screen, UC.GOT_REWARD, conf=0.5)
        if got is not None:
            return action_click(*_POS_TOUCH_CONTINUE, "milestone: dismiss reward")
        claim_y = self.find_cls(screen, UC.CLAIM_REWARD_YELLOW, conf=0.5)
        if claim_y is not None:
            # ⛔after-ack(2026-07-28): 旧 reason 含「領取獎勵」命中关键词豁免 →
            # 稳定门+hold 全跳 → 每 tick 真发一发(0.2s 节奏), 后续发落在
            # 獲得獎勵 toast/弹窗动画中途的坐标上。发一发等帧证据, 且
            # _force_settle 走稳定门(别再拿措辞当控制 API)。
            if self._ms_claim_fired and self.since("ms_claim") < _MS_CLAIM_RETRY_SEC:
                return action_wait(300, "milestone claim 已发 — 等 toast/按钮变灰")
            self._ms_claim_fired = True
            self.mark("ms_claim")
            act = action_click_box(claim_y, "milestone: claim reward (黄)")
            act["_force_settle"] = True
            return act
        claim_g = self.find_cls(screen, UC.CLAIM_REWARD_GREY, conf=0.30)
        if claim_g is not None:
            self._milestone_done = True
            close = self.find_cls(screen, UC.BTN_CLOSE_X, conf=_CLS_CONF)
            if close is not None:
                self._set("tasks")
                return action_click_box(close, "milestone: 领完(灰) → close")
            self._set("tasks")
            return action_back("milestone: 领完(灰) → back")
        info = self.find_cls(screen, UC.EVENT_REWARD_INFO, conf=_WEAK_CONF)
        if info is not None:
            self._milestone_wait_t0 = 0.0
            region = (info.x1 - 0.02, info.y1 - 0.05,
                      info.x2 + 0.04, info.y2 + 0.02)
            if self.dot_in_region(screen, region, dot_classes=("红点",)):
                return action_click_box(info, "milestone: 獎勵資訊 (red dot)")
            self.log("milestone: no red dot on 獎勵資訊 — skip")
            self._milestone_done = True
            self._set("tasks")
            return action_wait(200, "no milestone rewards")
        # ⛔2026-07-27 live 实锤(logs tick 151-167): 旧码在这里是
        # `action_wait(600, "milestone: waiting")` —— 一路等到 `_PHASE_MAX`
        # (=18 **tick**, 又一个 tick 当计时器) 超时, 实测空转 17 tick ≈ **30s**。
        # 问题不只是慢: 它把「这一页压根没有獎勵資訊入口」和「入口还没渲染」
        # **混为一谈**, 于是"到底有没有里程碑奖励可领"这个问题被超时吞掉,
        # 出口(exit_report)也报不出真相 —— 典型的"活干没干净还没人知道"。
        # 改: 短墙钟只等渲染; 窗口过后**明确判定 + 留痕**, 不再空转。
        # ⚠措辞要诚实 —— 判的是"没找到入口", **不是**"没有奖励"。
        if not self._milestone_wait_t0:
            self._milestone_wait_t0 = self.clock()
        _el = self.clock() - self._milestone_wait_t0
        if _el > _MILESTONE_ANCHOR_SEC:
            self.log(f"milestone: {_el:.1f}s 内未见「獎勵資訊」入口 → 判定"
                     f"**本页无该入口**(注意: 这不等于'没有奖励'。若这条反复出现, "
                     f"查是不是活动页版式变了 / cls 漏检)")
            self._milestone_done = True
            self._set("tasks")
            return action_wait(200, "milestone: 无獎勵資訊入口 → tasks")
        return action_wait(300, f"milestone: 等獎勵資訊入口 ({_el:.1f}s)")

    # ── tasks (活动任務领奖) / close ──────────────────────────────────

    def _tasks(self, screen: ScreenState) -> Dict[str, Any]:
        if self._phase_ticks > _PHASE_MAX and self.since("phase_wall") > _PHASE_MAX_SEC:
            self._set("close")
            return action_wait(200, "tasks phase timeout")
        if self._tasks_done:
            self._set("close")
            return action_wait(200, "tasks already claimed")
        # 在任務页: 领取链
        claim_all = self.find_cls(screen, UC.CLAIM_ALL_YELLOW, conf=0.5)
        if claim_all is not None:
            return action_click_box(claim_all, "tasks: 全部領取")
        claim = self.find_cls(screen, UC.CLAIM_YELLOW, conf=0.5)
        if claim is not None:
            return action_click_box(claim, "tasks: 領取")
        got = self.find_cls(screen, UC.GOT_REWARD, conf=0.5)
        if got is not None:
            return action_click(*_POS_TOUCH_CONTINUE, "tasks: dismiss reward")
        grey = self.find_cls(screen, [UC.CLAIM_ALL_GREY], conf=0.5)
        if grey is not None:
            self.log("tasks: all claimed (grey)")
            self._tasks_done = True
            self._set("close")
            return action_back("tasks done → back")
        # 还在 quest 列表: 任務按钮红点门控
        task_btn = self.find_cls(screen, UC.EVENT_TASK, conf=_WEAK_CONF)
        if task_btn is not None:
            region = (task_btn.x1 - 0.02, task_btn.y1 - 0.05,
                      task_btn.x2 + 0.05, task_btn.y2 + 0.02)
            if self.dot_in_region(screen, region, dot_classes=("红点",)):
                return action_click_box(task_btn, "tasks: enter 任務 (red dot)")
            self.log("tasks: no red dot on 任務 — skip")
            self._tasks_done = True
            self._set("close")
            return action_wait(200, "no task rewards")
        return action_wait(600, "tasks: waiting")

    def _close(self, screen: ScreenState) -> Dict[str, Any]:
        page = self.detect_screen_yolo(screen)
        if page == "Mission":
            return action_done(f"event_quest complete on hub (swept={self._swept})")
        if page == "Lobby":
            return action_done(f"event_quest complete (swept={self._swept})")
        if self._phase_ticks > _PHASE_MAX:
            return action_done(f"event_quest exit timeout (swept={self._swept})")
        if self._phase_ticks % 3 != 1:
            return action_wait(600, "closing — settling")
        close = self.find_cls(screen, UC.BTN_CLOSE_X, conf=_CLS_CONF)
        if close is not None:
            return action_click_box(close, "close popup (X)")
        back = self.find_cls(screen, UC.BTN_BACK, conf=_CLS_CONF)
        if back is not None:
            return action_click_box(back, "→ hub (back)")
        return self.nav_home(screen, "event_quest close")
