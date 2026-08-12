# -*- coding: utf-8 -*-
"""用户配置 —— 用户点名的"能想到的都做成可选"。

前端不用手写表单：`SCHEMA` 描述了每个字段的类型/标签/取值来源，
`/api/schema` 直接吐给前端自动渲染（开关 / 多选 / 数字 / 下拉）。

⛔**金钱项 LOCKED**：前端只读，没有放宽入口。这不是 UI 层的选择，是这里
   `merged()` 强制覆盖 —— 就算有人直接改 profile.json 也改不动。
   （用户铁律：bot 绝不花青辉石/真钱。）

⭐"选项从屏上动态扫"的字段（悬赏分支 / JFD 学院 / 活动关卡）在 SCHEMA 里标
   `dynamic: "..."`，前端进那一页时调 `/api/scan/<name>` 拿当前真实可选项，
   **不写死清单**（§A8/§A9：写死的清单换个版面/换个活动就过期）。
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict

_ROOT = Path(__file__).resolve().parents[2]
PROFILE = _ROOT / "routing_v2" / "config" / "profile.json"


# ══════════════════════════════════════════════════════════════════════
DEFAULTS: Dict[str, Any] = {

    # ── 总开关：每个玩法跑不跑 ────────────────────────────────────────
    "modules": {
        "daily_routine": True,     # 收菜：免费包 / 社团 / 制造 / 信用点商店
        "cafe": True,
        "schedule": True,
        "bounty": True,
        "jfd": True,
        "event": True,
        "arena": True,
        "mail": True,
        "daily_mission": True,
        "story_mining": False,     # ⭐剧情挖矿 —— 默认关，前端按钮开
        "momotalk": False,         # ⭐MomoTalk 好感度 —— 默认关，前端按钮开
        "batch_sweep": False,
        "special_sweep": False,
    },

    # ── 跑的顺序（前端可拖拽）────────────────────────────────────────
    # ⭐活动期间 AP 全给活动（用户 2026-07-15 拍板），所以 batch_sweep /
    #   special_sweep 默认不在链里。没活动时手动打开。
    "order": ["daily_routine", "cafe", "schedule", "bounty", "jfd",
              "event", "arena", "mail", "daily_mission",
              "momotalk", "story_mining"],

    # ── ⭐AP 怎么分（用户点名）────────────────────────────────────────
    # 用户原话：「有活动刷活动，但我认为可以给 user 选择 —— 都刷活动，还是
    #            都刷双三倍的地方，还是分配百分比体力」
    # ⭐三种意图都用**同一个百分比表**表达，不需要额外的模式枚举：
    #     都刷活动     → {"event": 100}
    #     都刷双三倍   → {"special_sweep": 50, "batch_sweep": 50}
    #     六四分       → {"event": 60, "special_sweep": 40}
    # ⛔**bot 认不出"哪个是双三倍"**：`452 双倍或三倍活动进行中` 只报"有"、
    #    不报"哪一类"（全仓零使用）。所以是不是双三倍得**用户自己知道**，
    #    这里只负责按用户给的比例把 AP 分下去，不会自动去找双三倍。
    "plan": {
        "ap_mode": "all_in_order",     # all_in_order（按 order 谁先跑谁花）| split
        "ap_split": {},                # {flow名: 百分比}，只在 split 模式生效
        "ap_floor": 0,                 # 全局留底 AP（谁都不许动这部分）
    },

    # ── 悬赏通缉 ─────────────────────────────────────────────────────
    "bounty": {
        "branches": ["教室"],          # 动态：进页扫 cls 列出实际有哪些分支
        "difficulty": "highest",       # highest | fixed
        "fixed_stage": None,
        "use_tickets": "all",          # all | keep_n
        "keep_tickets": 0,
        # ⭐用户点名：「票用在哪个地区，还是一个地区用几张」
        #   {分支名: 张数}；空 = 按 branches 顺序打到票光（老行为）。
        #   ⭐实现用**票数差**算已用几张（数事实），读取点 flow/sweep.py `_quota`。
        "ticket_plan": {},
    },

    # ── 学院交流会 ───────────────────────────────────────────────────
    # ⭐三个学院都有 cls（三一/千年/格黑娜，各 84 train）—— 顺序即优先级
    "jfd": {
        "academies": ["千年", "三一", "格黑娜"],
        "difficulty": "highest",
        "use_tickets": "all",
        "keep_tickets": 0,
        "ticket_plan": {},             # ⭐同 bounty：{学院名: 张数}
    },

    # ── 活动 ─────────────────────────────────────────────────────────
    "event": {
        "clear_first_with_team": 1,    # ⭐首通用部队1（速推主力，用户规则）
        "bonus_team": 2,               # ⭐加成队 = 部队2
        "order": "clear_then_bonus",   # ⭐先 Q1→Qn 通关，再打加成
        "shop_plan_before_bonus": True,  # ⭐先商店推算定缺哪种币，再按币种编队
        "farm_stages": {"10": 1, "11": 2},
        "ap_reserve": 0,
        "min_ap_for_sweep": 20,
        "max_rounds": 1,
        "shop": {
            "auto_buy": True,
            "currencies": [],          # 空 = 全部非红线币种
            "furniture": False,
            "skip_last_tab": True,     # ⛔tab3 盒抽币绝不自动买
        },
    },

    # ── 咖啡厅 ───────────────────────────────────────────────────────
    # 用户口述权威走法：弹窗叉 → 收益≠0 领 → 邀请卷找角色下滑 → 摸头给足时间
    #                  → 2号厅 → 无黄点才返回
    "cafe": {
        "invite_targets": ["凯伊", "爱丽丝(战斗)", "爱丽丝"],
        "skip_invite": False,
        "headpat": True,
        "headpat_dwell_s": 2.0,        # 摸头给足时间（用户强调）
        "floors": [1, 2],
    },

    # ── 课程表 ───────────────────────────────────────────────────────
    "schedule": {
        "target_students": [],
        "relationship_first": True,
        "areas": [],                   # 空 = 按屏上可见区域顺序；动态扫
    },

    # ── 制造 ─────────────────────────────────────────────────────────
    "craft": {
        "use_acceleration_ticket": False,
        "phase_priority": ["光辉", "花朵"],
        "quantity": "MAX",
        "claim_finished": True,
    },

    # ── 商店（信用点）─────────────────────────────────────────────────
    "shop": {
        "common_priority": [],
        "tactical_priority": [],
        "refresh_times": 0,            # ⛔刷新要花青辉石 → 锁 0
        "buy_free_pack": True,         # 免費組合包（逐帧人审）
    },

    # ── 战术大赛 ─────────────────────────────────────────────────────
    "arena": {
        "level_diff": 0,
        "stop_at_rank1": True,
        "max_refresh": 0,
        "buy_energy_drink": False,     # 花大赛币买体力（不是青辉石）
    },

    # ── 剧情挖矿 ⭐用户点名 ───────────────────────────────────────────
    "story_mining": {
        "sources": ["羁绊剧情", "主线剧情", "支线剧情", "短篇剧情"],
        "target_students": [],         # 空 = 全部；有值 = 只挖这些人的羁绊
        "max_ap": 0,                   # 0 = 不限
        "max_nodes": 0,                # 0 = 不限
        "skip_cutscene": True,
        "stop_on_battle": True,        # 剧情里遇战斗：True=跳过该节点, False=打
    },

    # ── MomoTalk 好感度 ⭐用户点名 ────────────────────────────────────
    "momotalk": {
        "reply_policy": "first_option",  # first_option | last_option | random
        "target_students": [],
        "max_students_per_run": 0,       # 0 = 不限
        "follow_bond_story": True,       # 弹「前往羁绊剧情」跟不跟
    },

    # ── 邮件 / 每日任务 / 社团 ────────────────────────────────────────
    "mail": {"claim_all": True},
    "daily_mission": {"claim_all": True},
    "club": {"claim": True},

    # ── ⛔安全（前端只读，LOCKED 强制覆盖）────────────────────────────
    "safety": {
        "forbid_premium_currency": True,
        "ap_purchase_limit": 0,
        "purchase_caps": {"arena": 0, "bounty": 0, "scrimmage": 0, "lesson": 0},
        "money_step_needs_human": True,
    },

    # ── 运行 ─────────────────────────────────────────────────────────
    "run": {
        "step_mode": True,             # 逐帧门控（每一发都要人放行）
        "frame_source": "scrcpy",      # ⛔不提供 adb 选项
        "confirm_frames": 3,           # 状态连续 N 帧才认（§A3）
        # ⛔状态不变 N 帧才判"这一发丢了"重发（§A5）。
        #   **别调小**：实测「加载中」平均 1.67s ≈ 50 帧@30fps。阈值比页面
        #   打开时间还短的话，页面正在开就重发，第二发落到**新页面**上，
        #   往往正好把它又关掉 —— 2026-08-08 实测商店/咖啡厅连点 5 次进不去
        #   就是这个（旧值 25 帧 ≈ 0.83s）。
        "retry_frames": 70,
        "stuck_frames": 400,           # 同一页面卡这么多帧 = 这个 flow 卡死
        "max_minutes": 90,
        "max_minutes_per_flow": 15,
        "save_frames": True,           # 落干净帧（飞轮素材，常开）
        "unknown_escape": True,        # 长时间 UNKNOWN 时允许温和逃生
        "allow_home_escape": False,    # ⛔⛔回大厅当兜底 —— 默认关，见 §A1
    },
}


# ⛔这些路径无论 profile 里写什么，一律强制成这个值。
LOCKED: Dict[str, Any] = {
    "safety.forbid_premium_currency": True,
    "safety.ap_purchase_limit": 0,
    "safety.purchase_caps.arena": 0,
    "safety.purchase_caps.bounty": 0,
    "safety.purchase_caps.scrimmage": 0,
    "safety.purchase_caps.lesson": 0,
    "shop.refresh_times": 0,           # 刷新商店要花青辉石
    "run.frame_source": "scrcpy",      # ADB 抓帧是所有时序 bug 的放大器
}


# ══ 前端自省表 ═════════════════════════════════════════════════════════
# kind: toggle | multi | select | int | float | text | list | order | readonly
SCHEMA = {
    "modules": {"kind": "toggle_group", "label": "跑哪些玩法",
                "note": "⭐挖矿 / MomoTalk 默认关，按钮开"},
    "order": {"kind": "order", "label": "执行顺序", "source": "modules"},

    "plan.ap_mode": {"kind": "select", "label": "AP 怎么分",
                     "options": ["all_in_order", "split"],
                     "note": "all_in_order = 按执行顺序，谁先跑谁花；"
                             "split = 按下面的百分比分"},
    "plan.ap_split": {"kind": "map", "label": "AP 百分比分配",
                      "options": ["event", "story_mining"],
                      "planned": ["batch_sweep", "special_sweep"],
                      "note": "⭐都刷活动 = event:100。"
                              "⛔「都刷双三倍」要的 special_sweep / batch_sweep "
                              "**两条 flow 还没实现**（见 registry.PLANNED），"
                              "现在填了也不会跑。"
                              "⛔另外 bot 认不出哪个是双三倍，得你自己知道"},
    "plan.ap_floor": {"kind": "int", "label": "全局留底 AP", "min": 0, "max": 999},

    "bounty.branches": {"kind": "multi", "label": "悬赏刷哪些分支",
                        "dynamic": "bounty_branches",
                        "note": "进页时扫 cls 动态列出，不写死"},
    "bounty.difficulty": {"kind": "select", "label": "难度",
                          "options": ["highest", "fixed"]},
    "bounty.use_tickets": {"kind": "select", "label": "票怎么用",
                           "options": ["all", "keep_n"]},
    "bounty.keep_tickets": {"kind": "int", "label": "留几张票", "min": 0, "max": 20},
    "bounty.ticket_plan": {"kind": "map", "label": "每个分支分几张票",
                           "dynamic": "bounty_branches",
                           "note": "⭐留空 = 按上面的顺序打到票光。"
                                   "填了就按张数分配（用票数差算，不靠计数器）"},

    "jfd.academies": {"kind": "multi", "label": "交流会刷哪些学院（顺序=优先级）",
                      "dynamic": "jfd_academies"},
    "jfd.difficulty": {"kind": "select", "label": "难度",
                       "options": ["highest", "fixed"]},
    "jfd.use_tickets": {"kind": "select", "label": "票怎么用",
                        "options": ["all", "keep_n"]},
    "jfd.ticket_plan": {"kind": "map", "label": "每个学院分几张票",
                        "dynamic": "jfd_academies",
                        "note": "⭐留空 = 按上面的顺序打到票光"},

    "event.clear_first_with_team": {"kind": "select", "label": "首通用哪支部队",
                                    "options": [1, 2, 3, 4],
                                    "note": "⭐用户规则：首通用部队1（速推主力）"},
    "event.bonus_team": {"kind": "select", "label": "加成队",
                         "options": [1, 2, 3, 4]},
    "event.order": {"kind": "select", "label": "打法顺序",
                    "options": ["clear_then_bonus", "bonus_only", "clear_only"],
                    "note": "⭐用户规则：先 Q1→Qn 整体打通，再打加成"},
    "event.shop_plan_before_bonus": {"kind": "toggle",
                                     "label": "先商店推算再编加成队"},
    "event.farm_stages": {"kind": "map", "label": "刷哪几关（关→轮数）",
                          "dynamic": "event_stages"},
    "event.shop.currencies": {"kind": "multi", "label": "活动商店买哪些币种",
                              "dynamic": "event_shop_currencies"},
    "event.shop.skip_last_tab": {"kind": "toggle",
                                 "label": "跳过最后一个 tab（盒抽币）"},

    "cafe.invite_targets": {"kind": "list", "label": "邀请哪些学生"},
    "cafe.headpat": {"kind": "toggle", "label": "摸头"},
    "cafe.floors": {"kind": "multi", "label": "去哪几号厅", "options": [1, 2]},

    "schedule.target_students": {"kind": "list", "label": "课程表优先学生"},

    "craft.phase_priority": {"kind": "list", "label": "制造优先阶段"},
    "craft.quantity": {"kind": "select", "label": "数量", "options": ["MAX", "1"]},

    "story_mining.sources": {"kind": "multi", "label": "挖哪些剧情",
                             "options": ["羁绊剧情", "主线剧情", "支线剧情",
                                         "短篇剧情", "活动剧情", "後日談"]},
    "story_mining.target_students": {"kind": "list", "label": "只挖这些学生的羁绊"},
    "story_mining.max_ap": {"kind": "int", "label": "AP 上限（0=不限）", "min": 0},
    "story_mining.stop_on_battle": {"kind": "toggle",
                                    "label": "遇到战斗就跳过（关=打）"},

    "momotalk.reply_policy": {"kind": "select", "label": "回复策略",
                              "options": ["first_option", "last_option", "random"]},
    "momotalk.target_students": {"kind": "list", "label": "只回这些学生"},
    "momotalk.follow_bond_story": {"kind": "toggle", "label": "跟进羁绊剧情"},

    "arena.stop_at_rank1": {"kind": "toggle", "label": "到第1名就停"},

    "run.step_mode": {"kind": "toggle", "label": "逐帧门控（每发人放行）"},
    "run.confirm_frames": {"kind": "int", "label": "状态确认帧数", "min": 1, "max": 15},
    "run.retry_frames": {"kind": "int", "label": "重发阈值（帧）", "min": 5, "max": 200},
    "run.max_minutes": {"kind": "int", "label": "总时长上限（分）", "min": 5, "max": 300},
    "run.allow_home_escape": {"kind": "toggle",
                              "label": "允许「回大厅」当兜底",
                              "note": "⛔默认关。回大厅按钮几乎每页都在，"
                                      "当兜底会把转场帧变成'弹回大厅'（§A1）"},

    "safety.forbid_premium_currency": {"kind": "readonly", "label": "禁止花青辉石"},
    "safety.ap_purchase_limit": {"kind": "readonly", "label": "买 AP 上限"},
    "safety.money_step_needs_human": {"kind": "readonly", "label": "金钱步需人审"},
}


# ══ 存取 ═══════════════════════════════════════════════════════════════
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
    """DEFAULTS ← 用户配置 ← LOCKED（LOCKED 最后落，谁也改不动）。"""
    cfg = _deep_merge(DEFAULTS, user or {})
    for path, val in LOCKED.items():
        _set_path(cfg, path, val)
    return cfg


def load(path: Path | str | None = None) -> dict:
    p = Path(path or PROFILE)
    user = {}
    if p.is_file():
        try:
            user = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[config] ⛔profile 解析失败({e}) — 用默认值，不静默带病跑",
                  flush=True)
    return merged(user)


# ⛔⛔mojibake 自检（2026-08-10 实伤）：UTF-8 的中文被按 Latin-1/GBK 解一遍后，
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
    """存用户配置。⛔LOCKED 项不写盘（写了也没用，徒增误解）。"""
    bad = _mojibake(cfg)
    if bad:
        raise ValueError(
            "⛔配置里有乱码字符串，拒绝写盘（写进去会让 flow 静默选不中分支）：\n  "
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
