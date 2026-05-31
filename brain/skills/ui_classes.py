"""Authoritative UI YOLO class vocabulary — single source of truth.

All skills resolve click targets through these named constants instead of
hardcoding OCR text. The names match ui_v1/v2 model cls_name exactly (see
data/_ui_v1_class_freq.json — 156 in-use classes out of 448).

DESIGN RULE (user spec 2026-05-28):
  - YOLO cls drives ALL clicking. OCR is for DIGITS ONLY (counts, AP, etc).
  - If YOLO can't see a needed cls, the skill should LOG + WAIT (surface the
    gap) rather than fall back to OCR or blind hardcoded clicks.
  - When a cls is under-trained, the fix is more training data for that cls,
    NOT an OCR fallback that hides the problem.

Frame counts (from ui_v1 train set) annotated so we know which cls are
reliable vs which still need oversampling:
  - >=80f  : solid
  - 20-80f : usable
  - 12-19f : weak (v1 mislabels these — v2 oversample target=60 fixes)
"""
from __future__ import annotations

# ── 顶栏资源 widget (YOLO 定位 icon, OCR 读旁边数字) ──────────────────
TOPBAR_AP = "体力"            # 18  (677f) AP/stamina
TOPBAR_CREDIT = "信用点"       # 3   (637f) credits
TOPBAR_PYROXENE = "青辉石"     # 30  (724f) pyroxene
TOPBAR_PLUS = "加号"           # 26  (518f) "+" buy button next to currency
TOPBAR_PLUS_GREY = "加号灰色"   # 116 (24f)

# ── 红黄点 (dot-driven skill dispatch) ───────────────────────────────
DOT_RED = "红点"               # 5   (559f) unclaimed reward badge
DOT_YELLOW = "黄点"            # 6   (223f) unfinished / new-content badge

# ── lobby 底栏导航入口 (16-19f each — weak in v1, v2 fixes) ───────────
NAV_MAIL = "邮件箱"            # 4   (19f)  top-right envelope
NAV_DAILY_REWARD = "每日领奖"   # 7   (16f)
NAV_MOMOTALK = "MomoTalk"     # 8   (16f)
NAV_CAFE = "咖啡厅入口"         # 9   (16f)
NAV_SCHEDULE = "课程表入口"     # 10  (16f)
NAV_STUDENT = "学生入口"        # 11  (16f)
NAV_EDIT = "编辑入口"          # 12  (16f)
NAV_SOCIAL = "社交入口"         # 13  (16f)
NAV_CRAFT = "制造入口"         # 14  (16f)
NAV_SHOP = "商店入口"          # 15  (16f)
NAV_RECRUIT = "招募入口"        # 16  (16f)
NAV_TASKS = "任务大厅入口"      # 17  (19f)

LOBBY_NAV_ICONS = [
    NAV_CAFE, NAV_SCHEDULE, NAV_STUDENT, NAV_EDIT, NAV_SOCIAL,
    NAV_CRAFT, NAV_SHOP, NAV_RECRUIT,
]  # seeing >=2 of these => we're on the lobby

# ── 通用按钮 / 弹窗 ──────────────────────────────────────────────────
BTN_CONFIRM = "确认键"          # 20  (82f)  blue confirm
BTN_CONFIRM_GREY = "灰色确认"   # 23  (12f)  disabled confirm (insufficient)
BTN_CANCEL = "取消键"           # 118 (47f)
BTN_CLOSE_X = "弹窗叉叉"        # 19  (429f) popup close X
BTN_BACK = "返回键"            # 31  (695f) top-left back arrow
BTN_HOME = "回大厅按钮"         # 28  (689f) home button → lobby
BTN_COLLAPSE = "收起键"         # 29  (0f — NOT trained)
LOADING = "加载中"             # 22  (45f)

# ── 领取按钮 (多状态: 黄=可领, 灰=已领/不可) ─────────────────────────
CLAIM_ALL_YELLOW = "全部领取_黄"     # 107 (12f)  claim-all active
CLAIM_ALL_GREY = "全部领取_灰色"     # 413 (12f)  claim-all done
CLAIM_ONCE_YELLOW = "一次领取黄色"   # 417 (32f)  一次領取 active (mail)
CLAIM_ONCE_GREY = "一次领取灰色"     # 416 (12f)
CLAIM_YELLOW = "领取_黄"            # 106 (28f)  single claim active
CLAIM_BLUE = "领取蓝色"            # 418 (12f)
CLAIM_GREY = "领取_灰"             # 396 (12f)
CLAIM_REWARD_YELLOW = "领取奖励_黄"  # 89  (12f)
CLAIM_REWARD_GREY = "领取奖励_灰"    # 90  (19f)
GOT_REWARD = "获得奖励"            # 397 (53f)  result popup "獲得獎勵" — tap dismiss
GREEN_CHECK = "绿勾"              # 403 (12f)  already-done check mark

# any of these yellow variants = there's something to claim
CLAIM_ACTIVE = [
    CLAIM_ALL_YELLOW, CLAIM_ONCE_YELLOW, CLAIM_YELLOW,
    CLAIM_REWARD_YELLOW, CLAIM_BLUE,
]
CLAIM_DONE = [CLAIM_ALL_GREY, CLAIM_ONCE_GREY, CLAIM_GREY, CLAIM_REWARD_GREY]

# ── cafe ─────────────────────────────────────────────────────────────
CAFE_INVITE_TICKET = "咖啡厅邀请卷"   # 24  (18f)
CAFE_EARNINGS = "咖啡厅收益"          # 25  (20f)
CAFE_INVITE_BTN = "邀请键"            # 32  (12f, 60b)
CAFE_MOVE_1F = "移动至一号店"          # 34  (12f)
CAFE_MOVE_2F = "移动至2号点"          # 27  (16f)

# ── schedule / 课程表 ────────────────────────────────────────────────
SCHED_TICKET = "课程表票"             # 35  (96f)
SCHED_ALL = "全体课程表"              # 41  (31f)  the roster overlay header
SCHED_START = "课程表开始"            # 400 (14f)
ROOM_LOCKED = "房间区域未解锁"        # 50  (27f)
# 学校区域 (schedule + story node select)
SCHOOL_OFFICE = "夏莱办公室"          # 36  (12f)
SCHOOL_DORM = "夏莱居住区"            # 37  (12f)
SCHOOL_GEHENNA = "格黑娜学院中央区"    # 38  (12f)
SCHOOL_ABYDOS = "阿拜多斯高中"        # 39  (12f)
SCHOOL_MILLENNIUM = "千年研究所"      # 40  (12f)

# ── 区域选择 (campaign / story chapter nav) ─────────────────────────
STAGE_HIGHWAY = "高架公路"           # 86  (12f)
STAGE_DESERT_RAIL = "沙漠铁道"        # 87  (12f)
STAGE_CLASSROOM = "教室"             # 88  (12f)

# ── club / social ───────────────────────────────────────────────────
CLUB = "社团"                       # 51  (12f)

# ── 战斗模式入口 (任务 hub tiles) ───────────────────────────────────
HUB_CAMPAIGN = "任务关卡推图"         # 67  (18f)
HUB_STORY = "剧情"                   # 68  (18f)
HUB_BOUNTY = "悬赏通缉"              # 69  (18f)
HUB_SPECIAL = "特殊任务"             # 70  (18f)
HUB_SCHOOL_EXCHANGE = "学院交流会"    # 71  (18f)
HUB_ARENA = "战术大赛"               # 75  (18f)

# ── 关卡 / 战斗准备 ─────────────────────────────────────────────────
STAGE_ENTER = "入场键"               # 79  (94f)  enter stage
STAGE_ENTER_LOCKED = "入场键没解锁"   # 82  (48f)
STAGE_NORMAL_SEL = "普通关卡选中"     # 80  (12f)
STAGE_NORMAL = "普通关卡"            # 420 (12f)
STAGE_HARD = "困难关卡"              # 81  (12f)
STAGE_HARD_SEL = "困难关卡选中"       # 419 (12f)
STAGE_STAR_0 = "关卡得星_0"          # 83  (36f)  uncleared stage (0 stars)
STAGE_STAR_3 = "关卡得星_3"          # 84  (24f)  3-star cleared
SWEEP_START = "扫荡开始"             # 108 (39f)
TASK_START = "任务开始"              # 109 (27f)
SORTIE = "出击"                     # 124 (44f)

# ── 票券 ─────────────────────────────────────────────────────────────
TICKET_BOUNTY = "悬赏通缉票"          # 85  (20f)
TICKET_ARENA = "战术大赛票"           # 91  (41f)
TICKET_SCHOOL_EXCHANGE = "学院交流会票"  # 406 (22f)

# ── 编队 / 部队 ─────────────────────────────────────────────────────
SQUAD_QUICK_EDIT = "快速编辑"         # 121 (51f)
SQUAD_AUTO_EDIT = "自动编辑按钮"      # 127 (12f)
SQUAD_1 = "1部队"                    # 119 (15f)
SQUAD_1_HI = "1部队高亮"             # 125 (38f)
SQUAD_2 = "2部队"                    # 126 (12f)
SQUAD_2_HI = "2部队高亮"             # 120 (15f)
SQUAD_3 = "3部队"                    # 123 (27f)
SQUAD_4 = "4部队"                    # 122 (27f)

# ── 战斗控制 (in-battle) ────────────────────────────────────────────
BATTLE_PAUSE = "战斗暂停"            # 128 (108f)
BATTLE_1X = "战斗1倍速"              # 412 (22f)
BATTLE_2X = "战斗2倍速"              # 135 (20f)
BATTLE_3X = "战斗三倍速"             # 129 (66f)
BATTLE_AUTO_ON = "自动战斗开启"      # 130 (72f)
BATTLE_AUTO_OFF = "自动战斗关闭"      # 134 (12f)
BATTLE_START = "战斗开始"            # 411 (12f)
BATTLE_RESTART = "重新开始键"        # 131 (12f)
BATTLE_CONTINUE = "继续键"           # 132 (12f)
BATTLE_GIVEUP = "放弃键"             # 133 (12f)
BATTLE_WIN = "战斗胜利"              # 136 (12f)
BATTLE_SKIP = "skip键"              # 137 (14f)  in-battle SKIP button
BATTLE_SKIP_TOGGLE = "跳过战斗未选"   # formation 跳過戰鬥 checkbox (arena/sweep)
GOTO_LOBBY_TEXT = "前往大厅文字按钮"  # 138 (12f)
ATTACK_FORMATION = "攻击编制"        # 435 (15f)

# ── 数量增减 (purchase / count steppers) ────────────────────────────
QTY_MAX = "MAX_可点击"              # 111 (39f)
QTY_MAX_GREY = "MAX_灰色"           # 117 (24f)
QTY_MIN = "MIN_可点击"              # 114 (14f)
QTY_MIN_GREY = "MIN_灰色"           # 112 (49f)
QTY_MINUS = "减号"                  # 115 (14f)
QTY_MINUS_GREY = "减号灰色"          # 113 (49f)

# ── 商店 ─────────────────────────────────────────────────────────────
SHOP_SELECT_ALL = "全部选择"         # 55  (18f)
SHOP_SELECT_ALL_GREY = "全部选择灰"   # 404 (12f)
SHOP_BUY = "购买"                   # 103 (30f)
SHOP_BUY_PYROXENE = "购买青辉石"      # 395 (16f)
CURRENCY = "货币"                   # 102 (18f)
CURRENCY_SEL = "货币_已选择"         # 101 (18f)
COMBO_PACK = "组合包未选择"          # 414 (12f)
COMBO_PACK_SEL = "组合包已选择"      # 445 (14f)
FREE = "免费"                       # 446 (14f)

# ── 制造 / craft ────────────────────────────────────────────────────
CRAFT_QUICK = "快速制造"             # 443 (28f)
CRAFT_START = "开始制造"             # 444 (19f)

# ── 活动 (event — 周年庆暂跳过, 但 cls 已训) ────────────────────────
EVENT_STORY = "活动剧情"             # 93  (24f)
EVENT_STORY_SEL = "活动剧情_已选择"   # 97  (12f)
EVENT_QUEST = "活动quest"           # 94  (12f)
EVENT_QUEST_SEL = "活动quest_已选择" # 100 (12f)
EVENT_SHOP = "活动商店"              # 95  (37f)
EVENT_TASK = "活动任务"              # 96  (37f)
EVENT_BONUS = "活动关卡产出额外加成"  # 110 (12f)
EVENT_END_LEFT = "距离结束还剩"       # 405 (23f)

# ── 剧情 / story mining ─────────────────────────────────────────────
STORY_MAIN = "主线剧情"              # 423 (15f)
STORY_SHORT = "短篇剧情"             # 424 (15f)
STORY_SIDE = "支线剧情"              # 425 (12f)
STORY_ENTER_CHAPTER = "进入章节"      # 139 (16f)
STORY_SKIP = "跳过故事键"            # 141 (46f)
STORY_SKIP_DISABLED = "跳过故事键不可用"  # 432 (31f)
STORY_TAP_CONTINUE = "点击继续字样"   # 142 (23f)
STORY_MENU = "剧情menu"             # 431 (24f)
STORY_ICON_DONE = "剧情图标已完成"    # 427 (32f)  cleared chapter node
STORY_ICON_UNDONE = "剧情图标未完成"   # 430 (16f)  uncleared chapter node
STORY_QUIT = "剧情中断退出"          # 433 (12f)
STORY_WATCH = "剧情观看"             # 434 (12f)
STORY_EASY_GUIDE = "简易攻略"        # 422 (12f)
NEW_MARK = "new"                    # 429 (17f)  NEW! badge
NODE_DONE = "完成"                  # 426 (18f)
SCENE_DONE = "战斗图标已完成"         # 447 (12f)

# ── momotalk ─────────────────────────────────────────────────────────
MOMO_UNREAD = "学生momotalk信息未读"  # 439 (121f) unread conversation badge
MOMO_SENDING = "学生发送信息中"       # 438 (28f)
MOMO_REPLY_OPT = "学生信息回复选项"    # 440 (32f)  reply choice
GOTO_BOND_STORY = "前往羁绊剧情"      # 441 (12f)
ENTER_BOND_STORY = "进入羁绊剧情"     # 442 (12f)

# ── 升级 / 全屏过场 (tap to dismiss) ────────────────────────────────
BOND_LEVELUP = "羁绊升级"            # 398 (15f)
REGION_LEVELUP = "地区升级"          # 399 (12f)

# ── 左右翻页 ─────────────────────────────────────────────────────────
ARROW_LEFT = "左切换"               # 0   (117f)
ARROW_RIGHT = "右切换"              # 1   (86f)
FAVORITE_ICON = "收藏图标"           # 33  (14f)


# ── 语义组 helpers ──────────────────────────────────────────────────
# Page-detection cls signatures: seeing these YOLO cls => we're on that page.
# Used by detect_current_screen_yolo() to replace OCR header matching.
PAGE_SIGNATURES = {
    # page_name: (required_any, min_count)
    # NOTE: detect_screen_yolo checks all NON-Lobby entries first (first
    # match wins, dict-insertion order) and Lobby last. Keep specific
    # single-page signatures disjoint so they don't shadow each other.
    "Battle": ([BATTLE_PAUSE, BATTLE_AUTO_ON, BATTLE_AUTO_OFF], 1),
    "Mail": ([CLAIM_ONCE_YELLOW, CLAIM_ONCE_GREY], 1),
    "Schedule": ([SCHED_ALL, SCHED_TICKET, SCHED_START], 1),
    "Cafe": ([CAFE_EARNINGS, CAFE_INVITE_TICKET, CAFE_MOVE_1F, CAFE_MOVE_2F], 1),
    "Craft": ([CRAFT_QUICK, CRAFT_START], 1),
    "MomoTalk": ([MOMO_UNREAD, MOMO_REPLY_OPT, MOMO_SENDING], 1),
    "Story": ([STORY_ENTER_CHAPTER, STORY_MENU, STORY_ICON_DONE, STORY_ICON_UNDONE], 1),
    # ── battle/sweep hub + sub-screens (legacy detect_current_screen names) ──
    "PVP": ([TICKET_ARENA], 1),          # arena main screen (持有票券 X/5)
    "Bounty": ([TICKET_BOUNTY], 1),      # bounty screen
    # campaign hub grid: needs >=2 distinct mode tiles so a single tile
    # leaking into another screen doesn't misfire.
    "Mission": ([HUB_CAMPAIGN, HUB_STORY, HUB_BOUNTY, HUB_SPECIAL,
                 HUB_SCHOOL_EXCHANGE, HUB_ARENA], 2),
    "Lobby": (LOBBY_NAV_ICONS, 2),
}
