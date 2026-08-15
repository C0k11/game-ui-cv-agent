# -*- coding: utf-8 -*-
"""cls 词表 —— 全 bot 唯一的类名来源，**带健康度**。

为什么要带训练量（README §A11 + §B）:
  「死判据」在老仓里出现过至少 10 次 —— 代码里堂堂正正写着 `find_cls(战斗失败)`，
  而那个类**一框训练数据都没有**，于是"输了"这件事永远检测不到，而代码看起来
  一切正常。反过来，`_废弃77` 那种废案我又误判成"漏训"，跑去补数据。
   这里给每个用到的类标 (train, val)，并提供 `require()` 在 import 期就把
    "拿死类当唯一信号"打回来。数字由 `scripts/sync_vocab_health.py` 从**实际构建出来的**数据集重算，
    它自带时机闸：数据集比现役权重新时拒绝 emit（那会高估现役模型）。
    当前值 = v16 训练用的那一份（2026-08-12 build / v16 于 08-13 上线）。

  DEAD  = train 0        绝不能当唯一判据，只能当"或"列表里的可选成员
  WEAK  = train < 100    可用但样本薄，别拿它做 fail-closed 的**否定**判据
  SOLID = train >= 100

改名铁律：`_classes.txt` 是按**行号**索引的。废案类只能改名加 `_废弃N_` 前缀，
   **绝不能删行** —— 删了行号索引会让全库标注错位。废案在 detect 层已被丢弃。
"""
from __future__ import annotations

from typing import Dict, Tuple

# ══ 顶栏资源 ═══════════════════════════════════════════════════════════
AP = "体力"
CREDIT = "信用点"
PYROXENE = "青辉石"
PLUS = "加号"
PLUS_GREY = "加号灰色"

# ══ 徽章 ═══════════════════════════════════════════════════════════════
DOT_RED = "红点"
DOT_YELLOW = "黄点"
NEW_MARK = "new"
STORY_NEW = "剧情new"

# ══ 大厅底栏 ═══════════════════════════════════════════════════════════
NAV_MAIL = "邮件箱"
NAV_DAILY_REWARD = "每日领奖"
NAV_MOMOTALK = "MomoTalk"
NAV_CAFE = "咖啡厅入口"
NAV_SCHEDULE = "课程表入口"
NAV_STUDENT = "学生入口"
NAV_EDIT = "编辑入口"
NAV_SOCIAL = "社交入口"
NAV_CRAFT = "制造入口"
NAV_SHOP = "商店入口"
NAV_RECRUIT = "招募入口"
NAV_TASKS = "任务大厅入口"
LOBBY_NAV = [NAV_CAFE, NAV_SCHEDULE, NAV_STUDENT, NAV_EDIT, NAV_SOCIAL,
             NAV_CRAFT, NAV_SHOP, NAV_RECRUIT, NAV_RECRUIT]

# ══ 通用控件 ═══════════════════════════════════════════════════════════
CONFIRM = "确认键"
CONFIRM_GREY = "灰色确认"
CANCEL = "取消键"
CLOSE_X = "弹窗叉叉"
BACK = "返回键"
HOME = "回大厅按钮"
LOADING = "加载中"
ARROW_LEFT = "左切换"
ARROW_RIGHT = "右切换"

# ══ 领取族 ═════════════════════════════════════════════════════════════
CLAIM_ALL_YELLOW = "全部领取_黄"
CLAIM_ALL_GREY = "全部领取_灰色"
CLAIM_ONEKEY_GREY = "一键领取灰色"          # DEAD train=0
CLAIM_ONCE_YELLOW = "一次领取黄色"
CLAIM_ONCE_GREY = "一次领取灰色"
CLAIM_YELLOW = "领取_黄"
CLAIM_GREY = "领取_灰"
CLAIM_BLUE = "领取蓝色"
CLAIM_REWARD_YELLOW = "领取奖励_黄"
CLAIM_REWARD_GREY = "领取奖励_灰"
GOT_REWARD = "获得奖励"
GREEN_CHECK = "绿勾"
CLAIM_ACTIVE = [CLAIM_ALL_YELLOW, CLAIM_ONCE_YELLOW, CLAIM_YELLOW,
                CLAIM_REWARD_YELLOW, CLAIM_BLUE]
CLAIM_DONE = [CLAIM_ALL_GREY, CLAIM_ONCE_GREY, CLAIM_GREY, CLAIM_REWARD_GREY]

# ══ 咖啡厅 ═════════════════════════════════════════════════════════════
CAFE_EARNINGS = "咖啡厅收益"
CAFE_TICKET = "咖啡厅邀请卷"
CAFE_INVITE = "邀请键"
CAFE_MOVE_1F = "移动至一号店"
CAFE_MOVE_2F = "移动至2号点"
EMOTICON = "Emoticon_Action"

# ══ 课程表 ═════════════════════════════════════════════════════════════
SCHED_TICKET = "课程表票"
SCHED_ALL = "全体课程表"
SCHED_START = "课程表开始"
ROOM_LOCKED = "房间区域未解锁"
SCHOOL_OFFICE = "夏莱办公室"
SCHOOL_DORM = "夏莱居住区"
SCHOOL_GEHENNA = "格黑娜学院中央区"
SCHOOL_ABYDOS = "阿拜多斯高中"
SCHOOL_MILLENNIUM = "千年研究所"
SCHOOL_AREAS = [SCHOOL_OFFICE, SCHOOL_DORM, SCHOOL_GEHENNA,
                SCHOOL_ABYDOS, SCHOOL_MILLENNIUM]

# ══ 社团 ═══════════════════════════════════════════════════════════════
CLUB = "社团"

# ══ 任务大厅入口 tile ═══════════════════════════════════════════════════
HUB_CAMPAIGN = "任务关卡推图"
HUB_STORY = "剧情"
HUB_BOUNTY = "悬赏通缉"
HUB_SPECIAL = "特殊任务"
HUB_JFD = "学院交流会"
HUB_ARENA = "战术大赛"
HUB_TILES = [HUB_CAMPAIGN, HUB_STORY, HUB_BOUNTY, HUB_SPECIAL, HUB_JFD, HUB_ARENA]

# ══ 悬赏分支（有 cls，动态识别）═════════════════════════════════════════
BRANCH_HIGHWAY = "高架公路"
BRANCH_DESERT = "沙漠铁道"
BRANCH_CLASSROOM = "教室"
BOUNTY_BRANCHES = [BRANCH_CLASSROOM, BRANCH_HIGHWAY, BRANCH_DESERT]

# ══ 学院交流会学院 有 cls（各 84 train / 5 val，2026-08-08 实测）═══════
# 老 jfd.py 写着"三个学院 tile 没有 YOLO cls（v6 gap）"并用写死坐标选择 ——
#    那是 v6 时代的事实，v15 早就训了。**这就是 §A8 的标准案例：写死坐标一旦
#    落下去，就没人回头看它补上了没。**
ACADEMY_TRINITY = "三一"
ACADEMY_MILLENNIUM = "千年"
ACADEMY_GEHENNA = "格黑娜"
JFD_ACADEMIES = [ACADEMY_MILLENNIUM, ACADEMY_TRINITY, ACADEMY_GEHENNA]

# ══ 票券 ═══════════════════════════════════════════════════════════════
TICKET_BOUNTY = "悬赏通缉票"
TICKET_ARENA = "战术大赛票"
TICKET_JFD = "学院交流会票"

# ══ 关卡 / 出击 ════════════════════════════════════════════════════════
STAGE_ENTER = "入场键"
STAGE_ENTER_LOCKED = "入场键没解锁"
STAGE_NORMAL = "普通关卡"
STAGE_NORMAL_SEL = "普通关卡选中"
STAGE_HARD = "困难关卡"
STAGE_HARD_SEL = "困难关卡选中"
STAR_0 = "关卡得星_0"
STAR_3 = "关卡得星_3"
SWEEP_START = "扫荡开始"
TASK_START = "任务开始"
SORTIE = "出击"
SWEEP_BATCH = "批量扫荡"
SWEEP_BATCH_START = "批量扫荡开始"
SWEEP_BATCH_START_GREY = "批量扫荡开始灰色"

# ══ 编队 ═══════════════════════════════════════════════════════════════
SQUAD_1 = "1部队"
SQUAD_1_HI = "1部队高亮"
SQUAD_2 = "2部队"
SQUAD_2_HI = "2部队高亮"
SQUAD_3 = "3部队"
SQUAD_4 = "4部队"
SQUAD_QUICK_EDIT = "快速编辑"
SQUAD_AUTO_EDIT = "自动编辑按钮"
SQUAD_TABS = {1: (SQUAD_1, SQUAD_1_HI), 2: (SQUAD_2, SQUAD_2_HI),
              3: (SQUAD_3, None), 4: (SQUAD_4, None)}
SKIP_BATTLE = "跳过战斗"
SKIP_BATTLE_OFF = "跳过战斗未选"

# -- 走格子（集中指挥, v16 新增族; 2026-08-13 小号实测 conf 0.85-0.98）--
GRID_CELL = "走格子_格子"
GRID_CELL_FOG = "走格子_格子_迷雾"
GRID_CELL_OPEN = "走格子_格子_可走"       # 叠在格子上的标记, 数格子要去重
GRID_START = "走格子_起点_黄"
GRID_START_GREY = "走格子_起点_灰"
GRID_ARROW = "走格子_队伍箭头"
GRID_BOSS = "走格子_BOSS标记"
GRID_ITEM = "走格子_道具"
GRID_ALLY = "走格子_我方"
GRID_ENEMY = "走格子_敌方"
PHASE_END = "PHASE结束"
PHASE_AUTO_ON = "PHASE自动结束_已勾选"
PHASE_AUTO_OFF = "PHASE自动结束_未勾选"
TASK_START_GREY = "任务开始_灰色"
TASK_ABORT = "中断任务"
TASK_RETRY = "重新挑战"
# 关卡弹窗两页签（两态; 496/421 是各自的选中态）
TAB_COMMAND = "集中指挥"
TAB_COMMAND_SEL = "集中指挥已选中"
TAB_GUIDE = "简易攻略"
TAB_GUIDE_SEL = "简易攻略_已选中"

# ══ 战斗内 ═════════════════════════════════════════════════════════════
BATTLE_PAUSE = "战斗暂停"
BATTLE_1X = "战斗1倍速"
BATTLE_2X = "战斗2倍速"
BATTLE_3X = "战斗三倍速"
BATTLE_AUTO_ON = "自动战斗开启"
BATTLE_AUTO_OFF = "自动战斗关闭"
BATTLE_START = "战斗开始"
BATTLE_RESTART = "重新开始键"
BATTLE_CONTINUE = "继续键"
BATTLE_GIVEUP = "放弃键"
BATTLE_WIN = "战斗胜利"              # WEAK train=33，但 live 实测检得出
BATTLE_COMPLETE = "战斗完成"          # 横幅，train=90
BATTLE_LOSE = "战斗失败"             # DEAD train=0 —— **输了看不见**
BATTLE_SKIP = "skip键"
GOTO_LOBBY_TEXT = "前往大厅文字按钮"
# 战斗内可见 = 这些里出现 ≥2 个（单个会在别处漏出）
BATTLE_CHROME = [BATTLE_PAUSE, BATTLE_AUTO_ON, BATTLE_AUTO_OFF,
                 BATTLE_1X, BATTLE_2X, BATTLE_3X]

# ══ 数量步进 ═══════════════════════════════════════════════════════════
QTY_MAX = "MAX_可点击"
QTY_MAX_GREY = "MAX_灰色"
QTY_MIN = "MIN_可点击"
QTY_MIN_GREY = "MIN_灰色"
QTY_MINUS = "减号"
QTY_MINUS_GREY = "减号灰色"

# ══ 商店 ═══════════════════════════════════════════════════════════════
SHOP_SELECT_ALL = "全部选择"
SHOP_SELECT_ALL_GREY = "全部选择灰"
SHOP_ALL_SELECTED = "已全部选择"
SHOP_BUY = "购买"
SHOP_BUY_GREY = "购买灰色"           # v15 起可用（train=212）
SHOP_BUY_SELECTED = "选择购买"
# 「選擇購買」的**灰态** = 勾上了但**买不起**（或没勾任何东西）。
#    2026-08-13 小号实帧: 信用点 35,544 而货架最贵 500,000 -> `选择购买灰色` **0.78**;
#    大赛币余额 0 -> **0.33**（低于闸的 0.40，那一页目前仍是盲的）。
#    它是**正向证据**: 亮态检不出时，灰态在场同样证明「全部选择」那一下生效了 ——
#      原来契约只写 `expect=(选择购买,)`，在买不起的号上永远不兑现、整段反复重试。
#    同时是**硬停**: 灰的按钮点了也不动，点就是空转。
SHOP_BUY_SELECTED_GREY = "选择购买灰色"
SHOP_BUY_PYROXENE = "购买青辉石"      # 金钱红线锚点
CURRENCY = "货币"
CURRENCY_SEL = "货币_已选择"
CURRENCY_QTY_AREA = "货币数量显示区域"
COMBO_PACK = "组合包未选择"
COMBO_PACK_SEL = "组合包已选择"
FREE = "免费"
SHOP_TAB_CREDIT = "信用点商店"
SHOP_TAB_CREDIT_SEL = "信用点商店_已选中"
SHOP_TAB_PYROXENE = "青辉石商店"
SHOP_TAB_PYROXENE_SEL = "青辉石商店_已选择"     # 进了这个 tab 就是在花青辉石
ARENA_SHOP_TAB = "战术大赛商店"
ARENA_SHOP_TAB_SEL = "战术大赛商店已选择"
ARENA_SHOP_CURRENCY = "战术大赛商店货币"
ENERGY_DRINK_LOW = "下级能量饮料"
ENERGY_DRINK_MID = "一般能量饮料"

# ══ 制造 ═══════════════════════════════════════════════════════════════
CRAFT_QUICK = "快速制造"
CRAFT_START = "开始制造"
CRAFT_START_GREY = "开始制造灰色"      # v15 起模型真会吐（train=289）
CRAFT_NO_MATERIAL = "材料不足"

# ══ 活动 ═══════════════════════════════════════════════════════════════
EVENT_STORY = "活动剧情"
EVENT_STORY_SEL = "活动剧情_已选择"
EVENT_QUEST = "活动quest"
EVENT_QUEST_SEL = "活动quest_已选择"
EVENT_SHOP = "活动商店"
EVENT_TASK = "活动任务"
EVENT_BONUS = "活动关卡产出额外加成"
EVENT_REWARD_INFO = "奖励资讯"
# 这两个才是真正在用的活动入口（train 949 / 130，live 实帧 0.88）。
#    带 `_活动入口` 后缀的 77/78 是**废案**，已在 detect 层丢弃。
EVENT_LIVE = "距离结束还剩"           # 当期，进行中，可打关
EVENT_ENDED = "距离奖励获得结束"       # 上期余韵期，只能领尾奖
EVENT_ENTRIES = [EVENT_LIVE, EVENT_ENDED]
EVENT_AFTERSTORY = "後日談"           # 上期活动的特征物
EVENT_MULTIPLIER = "双倍或三倍活动进行中"
SPECIAL_DEFENSE = "据点防御"
SPECIAL_CREDIT = "信用货币回收"
# 这两个本该用来判"这关打过没"，但 **train=0 / val=0**，一框都没有。
#     关卡完成度只能靠 关卡得星_0 / _3（1263 / 3765 框）间接判。
EVENT_STAGE_STORY_SEEN = "活动剧情关卡_已看"     # DEAD
EVENT_STAGE_BATTLE_DONE = "活动站斗关卡_已打"    # DEAD

# ══ 剧情 / 挖矿 ════════════════════════════════════════════════════════
STORY_MAIN = "主线剧情"
STORY_SHORT = "短篇剧情"
STORY_SIDE = "支线剧情"
STORY_ENTER_CHAPTER = "进入章节"
STORY_SKIP = "跳过故事键"
STORY_SKIP_DISABLED = "跳过故事键不可用"
STORY_TAP_CONTINUE = "点击继续字样"
STORY_MENU = "剧情menu"              # 用这个（train=335）
STORY_MENU_DEAD = "故事菜单键"        # DEAD train=0，别用
STORY_NODE_DONE = "剧情图标已完成"
STORY_NODE_UNDONE = "剧情图标未完成"
STORY_QUIT = "剧情中断退出"
STORY_WATCH = "剧情观看"
STORY_EASY_GUIDE = "简易攻略"
NODE_DONE = "完成"
NODE_DONE_GREY = "完成_灰色"
SCENE_DONE = "战斗图标已完成"
BOND_LEVELUP = "羁绊升级"
REGION_LEVELUP = "地区升级"

# ══ MomoTalk ═══════════════════════════════════════════════════════════
MOMO_TAB = "momotalk学生聊天区域按钮"
MOMO_TAB_SEL = "momotalk学生聊天区域已进入"
MOMO_UNREAD = "学生momotalk信息未读"   # train=6462，非常强
MOMO_SENDING = "学生发送信息中"        # 瞬时：学生打字中  见即等
MOMO_REPLY_OPT = "学生信息回复选项"
MOMO_GOTO_BOND = "前往羁绊剧情"
MOMO_ENTER_BOND = "进入羁绊剧情"

# ══ 大赛 ═══════════════════════════════════════════════════════════════
ARENA_ROW = "战术大赛对战选择区域"
ARENA_ATTACK_FORM = "攻击编制"     # 对手详情面板上的"进攻编队"入口（435）
ARENA_WAIT = "等待时间"            # 左栏出击冷却「等待時間 mm:ss」的标签（526）
#    只框标签四个字不含数字（08-15 帧证）; READY 态显示 --:-- 标签仍在,
#    所以冷却判据 = read_wait_secs 读它右侧的数字, 不是 cls 存在性。

# ══ 剧情过场逃生（interceptor 用）══════════════════════════════════════
# BA 的过场**不吃 KEYCODE_BACK**（2026-08-07 连按 5 次实测全无响应）。
#    唯一有效链：剧情menu(右上)  跳过故事键  确认键。
STORY_ESCAPE_CHAIN = [STORY_MENU, STORY_SKIP]


# ══ 健康度表（ui_v2 数据集实测 2026-08-08）════════════════════════════
# cls  (train, val)。只登记**代码里真的会引用**的类；没登记的默认按 UNKNOWN
# 处理（require() 会放行但打日志），不是"没登记就是死类"。
HEALTH: Dict[str, Tuple[int, int]] = {
    BATTLE_LOSE: (6, 28),
    EVENT_STAGE_STORY_SEEN: (0, 0),
    EVENT_STAGE_BATTLE_DONE: (0, 0),
    STORY_MENU_DEAD: (0, 0),
    CLAIM_ONEKEY_GREY: (0, 0),
    BATTLE_WIN: (32, 20),
    BATTLE_COMPLETE: (121, 1),
    EVENT_AFTERSTORY: (33, 753),
    SKIP_BATTLE_OFF: (52, 93),
    CLUB: (50, 60),
    BRANCH_HIGHWAY: (80, 66),
    BRANCH_DESERT: (78, 66),
    BRANCH_CLASSROOM: (79, 66),
    ACADEMY_TRINITY: (100, 37),
    ACADEMY_MILLENNIUM: (100, 37),
    ACADEMY_GEHENNA: (100, 37),
    CRAFT_NO_MATERIAL: (72, 81),
    CRAFT_START_GREY: (292, 83),
    SHOP_BUY_GREY: (1320, 753),
    NODE_DONE_GREY: (62, 34),
    EVENT_LIVE: (952, 378),
    EVENT_ENDED: (156, 88),
    EVENT_REWARD_INFO: (101, 26),
    STAGE_ENTER: (5150, 4494),
    STAGE_ENTER_LOCKED: (2053, 294),
    SORTIE: (527, 255),
    SQUAD_1_HI: (565, 309),
    SQUAD_2: (131, 80),
    STAR_0: (1365, 180),
    STAR_3: (3461, 3564),
    MOMO_UNREAD: (5015, 1824),
    MOMO_REPLY_OPT: (353, 89),
    MOMO_GOTO_BOND: (33, 14),
    MOMO_ENTER_BOND: (34, 15),
    STORY_MAIN: (69, 34),
    STORY_SHORT: (70, 35),
    STORY_SIDE: (68, 34),
    STORY_NODE_UNDONE: (467, 422),
    STORY_NODE_DONE: (451, 437),
    STORY_SKIP: (259, 197),
    STORY_MENU: (271, 114),
    TICKET_BOUNTY: (402, 222),
    TICKET_JFD: (440, 172),
    TICKET_ARENA: (1298, 529),
    SWEEP_START: (511, 166),
    TASK_START: (637, 186),
    SWEEP_BATCH: (471, 356),
    CAFE_EARNINGS: (1233, 313),
    CAFE_TICKET: (316, 156),
    CAFE_INVITE: (552, 130),
    CAFE_MOVE_1F: (387, 129),
    CAFE_MOVE_2F: (801, 286),
    SCHED_TICKET: (3577, 1090),
    SCHED_ALL: (1021, 282),
    SCHED_START: (187, 52),
    ARENA_ROW: (2202, 843),
    ARENA_WAIT: (1092, 358),
    COMBO_PACK_SEL: (305, 99),
    FREE: (384, 127),
    SHOP_BUY: (3608, 3412),
    CLAIM_ONCE_YELLOW: (148, 35),
    CLAIM_ALL_YELLOW: (219, 39),
}

DEAD = {c for c, (t, _) in HEALTH.items() if t == 0}
WEAK = {c for c, (t, _) in HEALTH.items() if 0 < t < 100}


def require(cls: str, *, sole_signal: bool = True) -> str:
    """在 flow 里声明"我要靠这个类做判断"。

    `sole_signal=True`（默认）= 这是**唯一**信号  死类直接抛，别让死判据上线。
    `sole_signal=False` = 它只是"或"列表里的一员  死类放行但打警告。

    为什么值得有这道闸：老仓里 4 道商店金钱守卫全是死码，全仓 10 条死判据，
       全都是"代码写得好好的，类没数据"。grep 找不出来，只能在这里挡。
    """
    if cls in DEAD:
        if sole_signal:
            raise RuntimeError(
                f"死判据: '{cls}' train=0，不能当唯一信号。"
                f"要么补训练数据，要么换判据（见 routing_v2/README.md §B）")
        print(f"[vocab] '{cls}' train=0，只能当可选成员，别指望它", flush=True)
    elif cls in WEAK:
        print(f"[vocab] '{cls}' 样本薄 train={HEALTH[cls][0]}，别拿它做否定判据",
              flush=True)
    return cls


def health_report() -> str:
    lines = ["cls 健康度（train/val, 见模块 docstring 的来源说明）", "-" * 52]
    for c in sorted(HEALTH, key=lambda x: HEALTH[x][0]):
        t, v = HEALTH[c]
        flag = "DEAD" if t == 0 else ("WEAK" if t < 100 else "  ok ")
        lines.append(f"{flag} {c:<24s} train={t:<6d} val={v}")
    return "\n".join(lines)
