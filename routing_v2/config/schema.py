# -*- coding: utf-8 -*-
"""用户配置。

选项写死在 SCHEMA / 词表（教室、最高难度、千年/三一/格黑娜），
不靠运行时扫屏生成。日常顺序固定，前端不许拖、不许手关。

金钱项 LOCKED：merged() 强制覆盖，改 profile.json 也改不动。
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict

from routing_v2.state import vocab as V

_ROOT = Path(__file__).resolve().parents[2]
PROFILE = _ROOT / "routing_v2" / "config" / "profile.json"


#
DEFAULTS: Dict[str, Any] = {

    #  账号（台账分桶键）
    # 08-15 复盘: event_topped/daybook/课程表房间账/ledger 过去全挤在同一批
    #    文件里, 键只有游戏日/倒数第几关 —— 大小号换着跑会互相把「今天做过/
    #    本期顶过」当成自己的账（ledger_20260813.jsonl 里 05:48 大号 59M,
    #    08:49 小号 35,544, 同一份文件）。空 = 拒绝开跑（fail-closed）。
    "account": {"id": ""},
    # 每个账号只覆盖自己的差异项。当前账号由 account.id 选择。
    "accounts": {},

    #  总开关：每个玩法跑不跑
    "modules": {
        "daily_routine": True,     # 收菜：免费包(第一枪) / 社团 / 制造 / 信用点商店
        "cafe": True,
        "schedule": True,
        "bounty": True,
        "jfd": True,
        "event": True,
        "arena": True,
        "mail": True,
        "daily_mission": True,
        "story_mining": False,     # 剧情挖矿 —— 默认关，前端按钮开
        "momotalk": False,         # MomoTalk 好感度 —— 默认关，前端按钮开
        "campaign": False,         # 任務推图：默认关，要推哪几关见 campaign.stages
        "batch_sweep": False,
        "special_sweep": False,
    },

    #  跑的顺序（固定，前端不许拖）
    # 活动期间 AP 全给活动（用户 2026-07-15 拍板）。
    # batch_sweep / special_sweep 未实现，不进链。
    "order": ["daily_routine", "cafe", "schedule", "bounty", "jfd",
              "event", "arena", "mail", "daily_mission",
              "momotalk", "story_mining"],

    #  AP 怎么分（用户点名）
    # 用户原话：「有活动刷活动，但我认为可以给 user 选择 —— 都刷活动，还是
    #            都刷双三倍的地方，还是分配百分比体力」
    # 三种意图都用**同一个百分比表**表达，不需要额外的模式枚举：
    #     都刷活动      {"event": 100}
    #     都刷双三倍    {"special_sweep": 50, "batch_sweep": 50}
    #     六四分        {"event": 60, "special_sweep": 40}
    # **bot 认不出"哪个是双三倍"**：`452 双倍或三倍活动进行中` 只报"有"、
    #    不报"哪一类"（全仓零使用）。所以是不是双三倍得**用户自己知道**，
    #    这里只负责按用户给的比例把 AP 分下去，不会自动去找双三倍。
    "plan": {
        "ap_mode": "all_in_order",     # all_in_order（按 order 谁先跑谁花）| split
        "ap_split": {},                # {flow名: 百分比}，只在 split 模式生效
        "ap_floor": 0,                 # 全局留底 AP（谁都不许动这部分）
    },

    #  悬赏通缉
    "bounty": {
        "branches": ["教室"],          # 词表: 教室 / 高架公路 / 沙漠铁道
        "difficulty": "highest",       # highest | fixed
        "fixed_stage": None,
        "use_tickets": "all",          # all | keep_n
        "keep_tickets": 0,
        # 用户点名：「票用在哪个地区，还是一个地区用几张」
        #   {分支名: 张数}；空 = 按 branches 顺序打到票光（老行为）。
        #   实现用**票数差**算已用几张（数事实），读取点 flow/sweep.py `_quota`。
        "ticket_plan": {},
    },

    #  学院交流会
    # 三个学院都有 cls（三一/千年/格黑娜，各 84 train）—— 顺序即优先级
    "jfd": {
        "academies": ["千年", "三一", "格黑娜"],
        "difficulty": "highest",
        "use_tickets": "all",
        "keep_tickets": 0,
        "ticket_plan": {},             # 同 bounty：{学院名: 张数}
    },

    #  活动
    "event": {
        "clear_first_with_team": 1,    # 首通用部队1（速推主力，用户规则）
        "bonus_team": 2,               # 加成队 = 部队2
        "order": "clear_then_bonus",   # 先 Q1Qn 通关，再打加成
        "shop_plan_before_bonus": True,  # 先商店推算定缺哪种币，再按币种编队
        # ⚠ farm_stages 是**死配置**: schema 里标着"旧表", 但 `flow/event.py`
        #    从来没读过它(2026-08-26 全仓 grep 确认)。用户以为配了就生效,
        #    实际打哪关一直是商店币档推算出来的。留着只是不想动老前端的表单,
        #    真正生效的是下面两个。
        "farm_stages": {"10": 1, "11": 2},
        # 按**关号**指定打哪几关加成 / 扫哪一关。空 = 沿用商店币档推算(老行为)。
        #   关号是语义(Q11 永远是 Q11), 老的 from_bottom 是位置(列表一滚就变),
        #   所以显式指定优先。关号从屏上读: 它印在得星星星正上方,
        #   `read.stage_numbers()` 从得星框推 ROI 去 OCR, 再用"关号连续递增"
        #   这个列表结构必然成立的事实补洞纠错。
        "bonus_stages": [],            # 例: [10, 11] = 依次给 Q10/Q11 顶纪录
        "sweep_stage": None,           # 例: 11 = 只扫 Q11
        "ap_reserve": 0,
        "min_ap_for_sweep": 20,
        "max_rounds": 1,
        "shop": {
            "auto_buy": True,
            "currencies": [],          # 空 = 全部非红线币种
            "furniture": False,
            "skip_last_tab": True,     # tab3 盒抽币绝不自动买
        },
    },

    #  咖啡厅
    # 用户口述权威走法：弹窗叉  收益≠0 领  邀请卷找角色下滑  摸头给足时间
    #                   2号厅  无黄点才返回
    "cafe": {
        "invite_targets": ["凯伊", "爱丽丝(战斗)", "爱丽丝"],
        "skip_invite": False,
        "headpat": True,
        "headpat_dwell_s": 2.0,        # 摸头给足时间（用户强调）
        "floors": [1, 2],
    },

    #  课程表
    "schedule": {
        # 选人方式。老行为是"取置信度最高的那个" -- 可置信度高低跟该不该给他
        #   上课毫无关系, 结果同几个学生天天被抽中。默认改随机(用户 08-26)。
        #   种子 = 游戏日, 所以同一天重跑抽到同一批(可复现), 换天自然换人。
        "pick_order": "random",        # random / confidence
        "max_students": 0,             # 本次最多上几节课, 0 = 不限(老行为)
        "target_students": [],
        "relationship_first": True,
        "areas": [],                   # 空 = 按屏上可见区域顺序
    },

    #  制造
    "craft": {
        "use_acceleration_ticket": False,
        "phase_priority": ["光辉", "花朵"],
        "quantity": "MAX",
        "claim_finished": True,
    },

    #  商店（信用点）
    "shop": {
        # 这两个原来只有代码里的默认值(facilities.py `_shop_opt`), profile 里
        #   没有 -> 前端看不见、也改不了, 而且哪天默认值一改行为就静默翻转。
        #   显式写进配置, 让前端成为权威(用户 08-26「根据前端的来」)。
        "credit_buy": False,           # 信用点商店: 不点"选择购买"(用户: 不去)
        "arena_shop": True,            # 战术大赛商店: 去(用户: 但是要去)
        "common_priority": [],
        "tactical_priority": [],
        "refresh_times": 0,            # 刷新要花青辉石  锁 0
        "buy_free_pack": True,         # 免費組合包(只领免费, 不点购买)
        "credit_buy": False,           # 大厅信用点「选择购买」。默认关
        "arena_shop": True,            # 战术大赛商店买饮料。要去买
    },

    #  战术大赛
    "arena": {
        "level_diff": 0,
        "stop_at_rank1": True,
        "max_refresh": 0,
        "buy_energy_drink": False,     # 花大赛币买体力（不是青辉石）
    },

    #  剧情挖矿 用户点名
    "story_mining": {
        "sources": ["羁绊剧情", "主线剧情", "支线剧情", "短篇剧情"],
        "target_students": [],         # 空 = 全部；有值 = 只挖这些人的羁绊
        "max_ap": 0,                   # 0 = 不限
        "max_nodes": 0,                # 0 = 不限
        "skip_cutscene": True,
        "stop_on_battle": True,        # 剧情里遇战斗：True=跳过该节点, False=打
    },

    #  MomoTalk 好感度 用户点名
    "momotalk": {
        "reply_policy": "first_option",  # first_option | last_option | random
        "target_students": [],
        "max_students_per_run": 0,       # 0 = 不限
        "follow_bond_story": True,       # 弹「前往羁绊剧情」跟不跟
    },

    #  任務推图（走格子）
    # stages 空 = 退回 stage 单关（兼容正在跑的 profile.campaign.stage）。
    # 两个都空 = 自动打屏上得星_0 那一行。
    "campaign": {
        "stage": "",
        "stages": [],
        "skip_rounds": 0,
        # 部署侧套預設(v20 新族 live 测试入口): {"team": 2, "tab": 1, "row": 2} = 出击前
        #   把部队2 切出来, 开預設面板, 页签1 第2行 組成 -> 變更編輯 確認。None = 不动。
        #   team=1 一律拒绝(用户推图队不许覆盖)。
        "preset_apply": None,
    },

    #  邮件 / 每日任务 / 社团
    "mail": {"claim_all": True},
    "daily_mission": {"claim_all": True},
    "club": {"claim": True},

    #  安全（前端只读，LOCKED 强制覆盖）
    # LOCKED 数字是 bot 花钱上限(青辉石/买票/刷新), 不是「余额一变就停」。
    # 用户自己抽卡/升级造成的余额变化由 ledger 记外部变动, 不 HALT。
    "safety": {
        "forbid_premium_currency": True,
        "ap_purchase_limit": 0,
        "purchase_caps": {"arena": 0, "bounty": 0, "scrimmage": 0, "lesson": 0},
        "money_step_needs_human": True,
    },

    #  运行
    "run": {
        "step_mode": True,             # 逐帧门控（每一发都要人放行）
        "frame_source": "scrcpy",      # 不提供 adb 选项
        "confirm_frames": 3,           # 状态连续 N 帧才认（§A3）
        # 状态不变 N 帧才判"这一发丢了"重发（§A5）。
        #   2026-08-15: 70 -> 38。campaign 实测约 0.041-0.050 s/tick,
        #   70 帧 = 2.9-3.5s 用户嫌不连贯; 38 帧 = 1.6-1.9s。
        #   不要再收到 25: 旧 25 帧在 30fps 假设下约 0.83s, 页面还在开
        #   就重发, 第二发落到新页上把它关掉(2026-08-08 商店/咖啡厅)。
        #   宽松契约超时地板见 gate._LOOSE_RETRY_FLOOR, 不跟本值等比缩小。
        "retry_frames": 38,
        "stuck_frames": 400,           # 同一页面卡这么多帧 = 这个 flow 卡死
        "max_minutes": 90,
        "max_minutes_per_flow": 15,
        "save_frames": True,           # 落干净帧（飞轮素材，常开）
        "unknown_escape": True,        # 长时间 UNKNOWN 时允许温和逃生
        "allow_home_escape": False,    # 回大厅当兜底 —— 默认关，见 §A1
    },

    # 国际服新皮: 任务大厅入口大厅稳帧过不了 0.45。
    # 08-20 live: lobby_enter/归位在 NAV>=3 时走盲点, 不再看这个开关。
    # 开关留给前端展示, v19 训回入口后整段删。
    "nav": {
        "task_hall_blind": False,
    },
}


# 这些路径无论 profile 里写什么，一律强制成这个值。
LOCKED: Dict[str, Any] = {
    "safety.forbid_premium_currency": True,
    "safety.ap_purchase_limit": 0,
    "safety.purchase_caps.arena": 0,
    "safety.purchase_caps.bounty": 0,
    "safety.purchase_caps.scrimmage": 0,
    "safety.purchase_caps.lesson": 0,
    "safety.money_step_needs_human": True,
    "shop.refresh_times": 0,           # 刷新商店要花青辉石
    "run.frame_source": "scrcpy",      # ADB 抓帧是所有时序 bug 的放大器
}


# 日常固定链。前端只展示游戏名，不许拖、不许手关。
# daily_routine 展开为免费包/社团/制造/商店（COMPOSITE, 免费包第一枪）。
DAILY_ORDER = [
    "daily_routine", "cafe", "schedule", "bounty", "jfd",
    "event", "arena", "mail", "daily_mission",
]
DAILY_CHAIN = [
    "免费包", "社团", "制造", "商店", "咖啡厅", "课程表",
    "悬赏通缉", "学院交流会", "活动", "战术大赛", "邮件", "每日任务",
]
EXTRA_MODULES = ["campaign", "story_mining", "momotalk"]
EXTRA_LABELS = {
    "campaign": "推图",
    "story_mining": "剧情挖掘",
    "momotalk": "MomoTalk",
}


def pin_daily(cfg: dict) -> dict:
    """日常模块全开、order 用固定链。推图/剧情/MomoTalk 保持用户值。"""
    mods = cfg.setdefault("modules", {})
    for k in DAILY_ORDER:
        mods[k] = True
    mods["batch_sweep"] = False
    mods["special_sweep"] = False
    extra = [k for k in EXTRA_MODULES if mods.get(k)]
    cfg["order"] = list(DAILY_ORDER) + extra
    return cfg


def write_account_cafe(cfg: dict) -> None:
    """把当前合并后的邀请配置写回 accounts[id]，避免下次被桶覆盖。"""
    aid = str(((cfg or {}).get("account") or {}).get("id") or "").strip()
    accounts = (cfg or {}).get("accounts")
    if not aid or not isinstance(accounts, dict) or aid not in accounts:
        return
    bucket = accounts[aid]
    if not isinstance(bucket, dict):
        return
    cafe = (cfg or {}).get("cafe") or {}
    bc = dict(bucket.get("cafe") or {})
    if "invite_targets" in cafe:
        bc["invite_targets"] = list(cafe.get("invite_targets") or [])
    if "skip_invite" in cafe:
        bc["skip_invite"] = bool(cafe.get("skip_invite"))
    bucket["cafe"] = bc


# 前端用。选项来自词表/已有枚举，不扫屏。
# kind: toggle | multi | select | int | float | text | list | order | readonly | farm_rows
SCHEMA = {
    "modules": {"kind": "toggle_group", "label": "日常模块"},
    "order": {"kind": "order", "label": "执行顺序", "source": "modules"},

    "account.id": {"kind": "select", "label": "当前账号",
                   "options": [],
                   "note": "台账分桶。邀请名单和跳过邀请按账号存"},

    "plan.ap_mode": {"kind": "select", "label": "体力怎么花",
                     "options": ["all_in_order", "split"],
                     "choice_labels": {"all_in_order": "按日常顺序花完",
                                       "split": "按比例分"},
                     "note": "默认按日常顺序谁先跑谁花。按比例分才看下面两行。",
                     "section": "more"},
    "plan.ap_split": {"kind": "map", "label": "体力比例（仅按比例分时）",
                      "options": ["event", "story_mining"],
                      "note": "键是内部玩法名，产品上一般不用改。",
                      "section": "more"},
    "plan.ap_floor": {"kind": "int", "label": "留底体力", "min": 0, "max": 999,
                      "note": "谁都不许花掉的体力。0=不留。",
                      "section": "ap"},

    "bounty.branches": {"kind": "multi", "label": "悬赏刷哪些",
                        "options": list(V.BOUNTY_BRANCHES),
                        "section": "daily"},
    "bounty.difficulty": {"kind": "select", "label": "悬赏难度",
                          "options": ["highest", "fixed"],
                          "choice_labels": {"highest": "最高难度",
                                            "fixed": "固定关卡"},
                          "section": "daily"},
    "bounty.use_tickets": {"kind": "select", "label": "悬赏票",
                           "options": ["all", "keep_n"],
                           "choice_labels": {"all": "用完", "keep_n": "留几张"},
                           "section": "daily"},
    "bounty.keep_tickets": {"kind": "int", "label": "悬赏留票", "min": 0, "max": 20,
                            "section": "daily"},
    "bounty.ticket_plan": {"kind": "map", "label": "悬赏各分支票数",
                           "options": list(V.BOUNTY_BRANCHES),
                           "section": "daily"},

    "jfd.academies": {"kind": "multi", "label": "交流会学院",
                      "options": list(V.JFD_ACADEMIES),
                      "section": "daily"},
    "jfd.difficulty": {"kind": "select", "label": "交流会难度",
                       "options": ["highest", "fixed"],
                       "choice_labels": {"highest": "最高难度",
                                         "fixed": "固定关卡"},
                       "section": "daily"},
    "jfd.use_tickets": {"kind": "select", "label": "交流会票",
                        "options": ["all", "keep_n"],
                        "choice_labels": {"all": "用完", "keep_n": "留几张"},
                        "section": "daily"},
    "jfd.ticket_plan": {"kind": "map", "label": "交流会各学院票数",
                        "options": list(V.JFD_ACADEMIES),
                        "section": "daily"},

    "event.clear_first_with_team": {"kind": "select", "label": "首通部队",
                                    "options": [1, 2, 3, 4],
                                    "section": "daily"},
    "event.bonus_team": {"kind": "select", "label": "加成队",
                         "options": [1, 2, 3, 4],
                         "section": "daily"},
    "event.order": {"kind": "select", "label": "活动打法",
                    "options": ["clear_then_bonus", "bonus_only", "clear_only"],
                    "choice_labels": {"clear_then_bonus": "先通关再打加成",
                                      "bonus_only": "只打加成",
                                      "clear_only": "只通关"},
                    "section": "daily"},
    "event.shop_plan_before_bonus": {"kind": "toggle",
                                     "label": "先看活动商店再编加成队",
                                     "note": "先扫货架看缺哪种币，再决定打哪关、编哪队。",
                                     "section": "shop"},
    "event.farm_stages": {"kind": "farm_rows", "label": "关号配比（旧表）",
                          "note": "关号是活动关卡编号（10=第10关），配比是轮转份数。"
                                  "当前刷关不读这张表，按商店缺货推算倒数关。"
                                  "空=用已存。",
                          "section": "daily"},
    "event.max_rounds": {"kind": "int", "label": "扫荡最多几轮",
                         "min": 0, "max": 99,
                         "note": "0=不限。这是扫荡次数上限，不是关号。",
                         "section": "daily"},
    "event.shop.skip_last_tab": {"kind": "toggle",
                                 "label": "活动商店跳过盒抽",
                                 "note": "最后一档是盒抽币，默认不买。",
                                 "section": "shop"},

    "cafe.invite_targets": {"kind": "chars", "label": "邀请哪些学生",
                            "section": "chars"},
    "cafe.skip_invite": {"kind": "toggle", "label": "跳过邀请",
                         "section": "chars"},
    "cafe.headpat": {"kind": "toggle", "label": "摸头",
                     "section": "daily"},
    "cafe.floors": {"kind": "multi", "label": "去几号厅", "options": [1, 2],
                    "section": "daily"},

    "schedule.target_students": {"kind": "chars", "label": "课程表找人",
                                 "note": "全体课程表里优先点这些人的房间。空=按屏上可见学生排。",
                                 "section": "chars"},

    "craft.phase_priority": {"kind": "multi", "label": "手动制造：优先阶段",
                             "options": ["光辉", "花朵"],
                             "note": "光辉/花朵是制造节点阶段，再拉满。"
                                     "这是手动制造，不是快速制造。"
                                     "进页有「快速制造」键就会点开面板，没有单独开关。",
                             "section": "daily"},
    "craft.quantity": {"kind": "select", "label": "手动制造：数量",
                       "options": ["MAX", "1"],
                       "choice_labels": {"MAX": "拉满", "1": "1 个"},
                       "note": "手动制造开槽时的数量。不是快速制造。",
                       "section": "daily"},

    "story_mining.sources": {"kind": "multi", "label": "挖哪些剧情",
                             "options": ["羁绊剧情", "主线剧情", "支线剧情",
                                         "短篇剧情", "活动剧情", "後日談"],
                             "section": "campaign"},
    "story_mining.target_students": {"kind": "list", "label": "只挖这些学生的羁绊",
                                     "section": "campaign"},
    "story_mining.max_ap": {"kind": "int", "label": "剧情体力上限（0=不限）",
                            "min": 0, "section": "campaign"},
    "story_mining.stop_on_battle": {"kind": "toggle",
                                    "label": "剧情遇战斗就跳过",
                                    "section": "campaign"},

    "momotalk.reply_policy": {"kind": "select", "label": "MomoTalk 回复",
                              "options": ["first_option", "last_option", "random"],
                              "choice_labels": {"first_option": "选第一项",
                                                "last_option": "选最后一项",
                                                "random": "随机"},
                              "section": "campaign"},
    "momotalk.target_students": {"kind": "list", "label": "只回这些学生",
                                 "section": "campaign"},
    "momotalk.follow_bond_story": {"kind": "toggle", "label": "跟进羁绊剧情",
                                   "section": "campaign"},

    "arena.stop_at_rank1": {"kind": "toggle", "label": "大赛到第1名就停",
                            "section": "daily"},

    "campaign.stages": {"kind": "list", "label": "要推的关卡",
                        "placeholder": "3-1, 3-3, H2-1",
                        "note": "按填写顺序连推，可跳号，可普通和困难混。"
                                "例: 3-1, 3-3, H2-1。空则只打单关。",
                        "section": "campaign"},

    "run.step_mode": {"kind": "toggle", "label": "逐帧门控（每发人放行）",
                      "section": "expert"},
    "run.confirm_frames": {"kind": "int", "label": "状态确认帧数",
                           "min": 1, "max": 15, "expert": True,
                           "section": "expert"},
    "run.retry_frames": {"kind": "int", "label": "重发阈值（帧）",
                         "min": 5, "max": 200, "expert": True,
                         "section": "expert"},
    "run.max_minutes": {"kind": "int", "label": "总时长上限（分）",
                        "min": 5, "max": 300, "expert": True,
                        "section": "expert"},
    "run.allow_home_escape": {"kind": "toggle",
                              "label": "允许回大厅当兜底",
                              "expert": True, "section": "expert"},
    "nav.task_hall_blind": {"kind": "toggle",
                            "label": "任务大厅入口盲点(临时)",
                            "note": "新皮入口 cls 大厅稳帧过不了 0.45。默认关。"
                                    "只在已确认大厅时点 (0.944, 0.942)。v19 训回后删。",
                            "expert": True, "section": "expert"},

    "safety.forbid_premium_currency": {"kind": "readonly", "label": "禁止花青辉石",
                                       "section": "safety"},
    "safety.ap_purchase_limit": {"kind": "readonly", "label": "买体力上限",
                                 "section": "safety"},
    "safety.money_step_needs_human": {"kind": "readonly", "label": "金钱步需人审",
                                      "section": "safety"},
    "safety.purchase_caps.arena": {"kind": "readonly", "label": "大赛买票上限",
                                   "section": "safety"},
    "safety.purchase_caps.bounty": {"kind": "readonly", "label": "悬赏买票上限",
                                    "section": "safety"},
    "safety.purchase_caps.scrimmage": {"kind": "readonly", "label": "演习买票上限",
                                       "section": "safety"},
    "safety.purchase_caps.lesson": {"kind": "readonly", "label": "课程买票上限",
                                    "section": "safety"},
    "shop.refresh_times": {"kind": "readonly", "label": "商店刷新次数",
                           "section": "safety"},
    "shop.credit_buy": {"kind": "toggle", "label": "信用点商店购买",
                        "note": "关=进店不点选择购买。默认关。",
                        "section": "shop"},
    "shop.arena_shop": {"kind": "toggle", "label": "战术大赛商店买饮料",
                        "note": "关=不去大赛店。默认开，要买饮料。",
                        "section": "shop"},
}


#  存取
def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _set_path(d: dict, path: str, val) -> None:
    parts = path.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = val


def merged(user: dict | None = None) -> dict:
    """DEFAULTS、用户配置、账号覆盖、LOCKED，按此顺序合并。"""
    cfg = _deep_merge(DEFAULTS, user or {})
    aid = str((cfg.get("account") or {}).get("id") or "").strip()
    accounts = cfg.get("accounts") or {}
    if isinstance(accounts, dict) and accounts and aid not in accounts:
        raise ValueError(
            f"account.id={aid!r} 不在非空 accounts 映射中，拒绝开跑: "
            "账号名拼错会静默套用顶层配置")
    selected = accounts.get(aid) if isinstance(accounts, dict) else None
    if isinstance(selected, dict):
        # 账号覆盖只能改业务项，不能把桶键切到另一个账号。
        selected = {k: v for k, v in selected.items()
                    if k not in ("account", "accounts")}
        cfg = _deep_merge(cfg, selected)
        cfg.setdefault("account", {})["id"] = aid
    for path, val in LOCKED.items():
        _set_path(cfg, path, val)
    return cfg


DATA_ROOT = _ROOT / "data" / "routing_v2"


def data_dir(cfg: dict) -> Path:
    """本账号的落盘桶: data/routing_v2/<account.id>/。

    所有会被"换号"污染的台账（ledger / daybook / event_topped /
    课程表房间账 / event_farm_plan）一律写进桶里, **禁止再往根路径写**。
    account.id 缺失/非法 = 拒绝开跑（fail-closed）: 没有桶键宁可不跑,
    也不能把两个号的「今天做过/本期顶过」记进同一本账。
    """
    aid = str(((cfg or {}).get("account") or {}).get("id") or "").strip()
    if not aid or any(c in aid for c in "\\/:*?\"<>|"):
        raise ValueError(
            "profile.json 缺 account.id（台账分桶键）— 拒绝开跑: "
            "大小号共用台账会互相把「今天做过/本期顶过」当成自己的账")
    p = DATA_ROOT / aid
    p.mkdir(parents=True, exist_ok=True)
    return p


def load(path: Path | str | None = None) -> dict:
    p = Path(path or PROFILE)
    user = {}
    if p.is_file():
        try:
            user = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[config] profile 解析失败({e}) — 用默认值，不静默带病跑",
                  flush=True)
    return merged(user)


# mojibake 自检（2026-08-10 实伤）：UTF-8 的中文被按 Latin-1/GBK 解一遍后，
#    会变成 `æå®¤`（教室）这种串 —— 特征是**大量 U+00C0~U+00FF 的拉丁补充字符**。
#    一旦写进 profile，`bounty.branches` 之类就永远匹配不上屏上的 cls，
#    而且**不报错、不停机，只是那条 flow 静默选不中分支**。属于「静默失败」族，
#    最难查，所以在唯一的写盘口拦住。
_MOJI = set(range(0x00C0, 0x0100)) | {0x00A0, 0x00A5, 0x00A6, 0x00AC, 0x00AD}


def _mojibake(node) -> list:
    """返回疑似乱码的字符串（只查字符串叶子）。"""
    bad = []
    if isinstance(node, str):
        n = sum(1 for ch in node if ord(ch) in _MOJI)
        # 两个以上拉丁补充字符连出现，且没有正常 CJK —— 正常中文配置不会这样
        if n >= 2 and not any("一" <= ch <= "鿿" for ch in node):
            bad.append(node)
    elif isinstance(node, dict):
        for k, v in node.items():
            bad += _mojibake(k) + _mojibake(v)
    elif isinstance(node, (list, tuple)):
        for v in node:
            bad += _mojibake(v)
    return bad


def save(cfg: dict, path: Path | str | None = None) -> Path:
    """存用户配置。LOCKED 项不写盘（写了也没用，徒增误解）。"""
    bad = _mojibake(cfg)
    if bad:
        raise ValueError(
            "配置里有乱码字符串，拒绝写盘（写进去会让 flow 静默选不中分支）：\n  "
            + "\n  ".join(sorted(set(bad))[:8])
            + "\n八成是客户端按错误编码读写了 JSON —— 请用 UTF-8 重发。")
    p = Path(path or PROFILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = copy.deepcopy(cfg)
    for path_ in LOCKED:
        parts = path_.split(".")
        cur = out
        for q in parts[:-1]:
            cur = cur.get(q, {})
            if not isinstance(cur, dict):
                break
        else:
            cur.pop(parts[-1], None)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return p
