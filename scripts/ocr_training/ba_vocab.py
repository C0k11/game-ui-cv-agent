"""Blue Archive vocabulary — every UI text string the pipeline needs to recognize.

This serves as:
1. Ground-truth label source for auto-correcting trajectory OCR crops
2. Corpus for synthetic training data generation
3. Reference for evaluation accuracy metrics

Organized by game screen / UI context.
"""

# ── Lobby & Navigation ──
LOBBY = [
    "任務", "任务", "咖啡廳", "咖啡厅", "咖啡", "商店", "社團", "社团",
    "製作", "制作", "競技場", "竞技场", "課程", "课程", "工作", "工坊",
    "主線", "主线", "支線", "支线", "特殊", "活動", "活动",
    "懸賞通緝", "悬赏通缉", "總力戰", "总力战", "大決戰", "大决战",
    "戰術大賽", "战术大赛", "學園交流會", "学园交流会", "制約解除決戰",
    "MoMoTalk", "郵件", "邮件", "好友", "公告", "通知",
]

# ── Common Buttons ──
BUTTONS = [
    "確認", "确认", "確定", "确定", "取消", "關閉", "关闭",
    "是", "否", "OK", "確", "确",
    "領取", "领取", "前往", "返回", "退出",
    "開始", "开始", "結束", "结束", "下一步", "跳過", "跳过",
    "繼續", "继续", "放棄", "放弃",
    "SKIP", "Skip", "MENU", "AUTO", "NEXT",
]

# ── Campaign / Mission ──
CAMPAIGN = [
    "任務", "任务", "劇情", "剧情", "通常", "困難", "困难",
    "Hard", "Normal", "Area", "入場", "入场",
    "掃蕩", "扫荡", "掃蕩開始", "扫荡开始",
    "任務開始", "任务开始", "任務資訊", "任务资讯",
    "MAX", "MIN", "START",
    "距離結束還剩", "距离结束还剩", "距離结束還剩",
    "活動進行中", "活动进行中",
    "特殊任務", "特殊任务",
]

# ── Event Activity ──
EVENT = [
    "Story", "Quest", "Challenge", "道具獲得方法", "道具获得方法",
    "獲得獎勵", "获得奖励", "獲得獎", "獲得奖",
    "Battle Complete", "BattleComplete",
    "COST", "WAVE",
    "任務開始", "任务开始", "剛剛好！", "刚刚好！",
    "距離結束還剩", "距离结束还剩",
    "距離獎勵獲得結束", "距离奖励获得结束",
    "距離獎勵領取結束", "距离奖励领取结束",
    "獎勵結束", "奖励结束", "獎勵領取", "奖励领取",
]

# ── Cafe ──
CAFE = [
    "咖啡廳", "咖啡厅", "咖啡廳收益", "咖啡厅收益",
    "每小時收益", "收益現況", "收益現况",
    "邀請券", "邀请券", "邀請", "邀请", "邀睛",  # OCR misread
    "額外邀請券", "额外邀请券", "可購買", "可使用",
    "指定訪問", "指定访问", "隨機訪問", "随机访问",
    "冷時間過後即可邀請", "冷却时间过后即可邀请",
    "移動至2號店", "移动至2号店",
    "舒適度", "舒适度", "編輯模式", "编辑模式",
    "禮物", "礼物", "家具資訊", "家具资讯",
    "預設", "预设", "全體收納", "全体收纳",
    "說明", "说明", "訪問學生目錄", "访问学生目录",
    "Rank Up", "好感度", "羈絆升級", "羁绊升级",
    "治愈力", "治癒力", "最大體力", "最大体力",
    "基本情報", "EX技能", "神秘解放",
]

# ── Schedule ──
SCHEDULE = [
    "課程表", "课程表", "選擇時間", "选择时间",
    "票券", "門票", "开始上课", "開始上課",
    "今日不再顯示", "今日不再显示",
    "更新消息", "更新資訊", "更新资讯",
]

# ── Shop ──
SHOP = [
    "商店", "一般", "軍需", "軍需品",
    "購買", "购买", "補充", "补充",
    "每日免費", "每日免费", "已購買", "已购买",
    "剩餘", "剩余",
]

# ── Club ──
CLUB = [
    "社團", "社团", "簽到", "签到",
    "已簽到", "已签到", "加入", "探索",
]

# ── Craft ──
CRAFT = [
    "製作", "制作", "工作", "工坊",
    "製造", "制造", "收取", "完成",
    "節點", "节点", "素材",
]

# ── Bounty / Wanted ──
BOUNTY = [
    "懸賞通緝", "悬赏通缉", "指名手配",
    "掃蕩", "扫荡", "討伐", "讨伐",
]

# ── Total Assault / Grand Assault ──
RAIDS = [
    "總力戰", "总力战", "大決戰", "大决战",
    "舉辦中", "举办中", "辨中",  # OCR misread of 辦中
    "Clear", "Score",
]

# ── Generic / System ──
SYSTEM = [
    "TOUCH TO START", "Touch to Start", "TOUCHTO CONTINUE", "TAP TO CONTINUE",
    "是否結束", "是否结束", "是否退出", "是否离开",
    "UID", "Lv.", "AP",
    "下載", "下载", "登入", "登录",
    "公告", "維護", "维护", "補償", "补偿",
    "題示", "提示",  # 題示 = OCR misread of 提示
]

# ── Numbers / Time Patterns ──
NUMBERS = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "/", ":", "：",
    "AM", "PM",
]

# ── MomoTalk ──
MOMOTALK = [
    "MoMoTalk", "未讀", "未读",
    "回覆", "回复", "選項", "选项",
]

# ── Story ──
STORY = [
    "主線", "主线", "支線", "支线",
    "劇情", "剧情", "Story", "Episode",
    "SKIP", "Skip", "AUTO", "MENU",
    "NEW",
]


def get_all_vocab() -> list[str]:
    """Return deduplicated list of all vocabulary strings."""
    all_lists = [
        LOBBY, BUTTONS, CAMPAIGN, EVENT, CAFE, SCHEDULE, SHOP, CLUB,
        CRAFT, BOUNTY, RAIDS, SYSTEM, NUMBERS, MOMOTALK, STORY,
    ]
    seen = set()
    result = []
    for lst in all_lists:
        for s in lst:
            if s not in seen:
                seen.add(s)
                result.append(s)
    return result


def get_all_characters() -> list[str]:
    """Return deduplicated set of all characters used in vocabulary."""
    chars = set()
    for s in get_all_vocab():
        for c in s:
            chars.add(c)
    return sorted(chars)


# ── Known OCR misread corrections ──
# Maps (misread text) → (correct text) for common Blue Archive OCR errors.
# Used for auto-correcting trajectory labels during training data extraction.
CORRECTIONS = {
    # Traditional/Simplified mixing
    "距離结束還剩": "距離結束還剩",
    "距離结束还剩": "距離結束還剩",
    # 辦→辨 misread
    "辨中": "辦中",
    "辨中！": "辦中！",
    "举辨中": "舉辦中",
    "举辨中！": "舉辦中！",
    # 邀請→邀睛 misread
    "邀睛": "邀請",
    "邀睛券": "邀請券",
    # 提示→題示 misread
    "題示": "提示",
    "题示": "提示",
    # 收益 variants
    "收益現况": "收益現況",
    # 訪問 misreads
    "訪間": "訪問",
    "訪间": "訪問",
    "訪周": "訪問",
    # 法妙妥 (event name misread)
    "法妙妥": "法妙托",
    # 綜合 variants
    "综合術": "綜合術",
    # 鲜升級 (羈絆 misread)
    "鲜升級": "羈絆升級",
    # 冷時間 (full correct form)
    "冷時間過後即可邀請。": "冷時間過後即可邀請。",
    # 指定訪問 misreads
    "指定訪間": "指定訪問",
    "随機訪間": "隨機訪問",
    "隋機訪間": "隨機訪問",
}


if __name__ == "__main__":
    vocab = get_all_vocab()
    chars = get_all_characters()
    print(f"Vocabulary: {len(vocab)} strings")
    print(f"Unique characters: {len(chars)}")
    print(f"Corrections: {len(CORRECTIONS)}")
    print(f"\nSample vocab: {vocab[:20]}")
    print(f"Sample chars: {''.join(chars[:50])}")
