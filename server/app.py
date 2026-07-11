import json
import os
import re
import sys
import socket
import subprocess
import time
import hashlib
import threading
import ctypes
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOGS_DIR = REPO_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

_SERVER_STARTED_AT = time.time()

DASHBOARD_PATH = REPO_ROOT / "server" / "dashboard.html"

CAPTURES_DIR = REPO_ROOT / "data" / "captures"
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

RAW_IMAGES_DIR = REPO_ROOT / "data" / "raw_images"
RAW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Trajectory ticks are valid annotation sources too — same JPGs the live
# pipeline captured, perfect for labeling UI states caught in-the-wild.
# Exposed as datasets with `traj/<run_dir>` prefix so the frontend can
# distinguish them from raw_images recordings.
TRAJECTORIES_DIR = REPO_ROOT / "data" / "trajectories"

# Universal master class registry shared across ALL annotation datasets.
# Single source of truth so adding 确认键 in one dataset makes it
# available in every other dataset without re-typing.  Per-dataset
# classes.txt files are kept in sync (copy of master) so YOLO training
# remains compatible — see _ensure_dataset_migrated().
MASTER_CLASSES_FILE = RAW_IMAGES_DIR / "_classes.txt"

CHARACTERS_DIR = CAPTURES_DIR / "角色头像"
CHARACTERS_DIR.mkdir(parents=True, exist_ok=True)

APP_CONFIG_PATH = REPO_ROOT / "data" / "app_config.json"

# Skills organized by 4 daily states.
# Lobby popup/sign-in/title-tap handled by pipeline global interceptor.
# Removed: lobby, ap_overflow, story_cleanup, joint_firing_drill,
# total_assault, hard_farming, campaign_push.
_SKILL_OPTIONS: List[Dict[str, str]] = [
    # Production daily flow (validated 2026-05-13).  All skills below
    # run end-to-end with badge-based skip routing — those with a lobby
    # nav-icon mapping (cafe/schedule/club/craft/shop/pass_reward) auto-
    # skip when their icon shows no red/yellow dot.  Sidebar-routed
    # skills (bounty/arena/daily_tasks/mail/event_activity) always run.
    # 2026-05-28 unification: 12 个收菜 skill 全部并入 daily_routine 单一入口
    # (mail/cafe/schedule/club/daily_tasks/craft/event_activity/pass_reward/
    #  momo_talk/story_mining/shop/ap_planning, 内部按 dot 自动判断 + craft
    #  强制进入). 战斗/扫荡类 skill (campaign_sweep/bounty/arena) 保留独立条目
    # 因为用户要对 AP / 票券消耗有显式控制.
    #
    # 旧 skill id 保留兼容老 profile, 标 [已并入 daily_routine] + 默认不勾选.

    # ── 推荐 (新 profile 默认勾这 2 个) ──
    {"id": "daily_routine", "label": "[★] 日常收菜 全套 (mail/cafe/schedule/club/daily_tasks/craft/event/pass_reward/momo/story/shop/ap, 内部按 dot 判断)"},
    {"id": "campaign_sweep", "label": "[★] 一键扫荡 悬赏/战术/活动副本 (hub 内按 dot 跳过个别)"},

    # 注: 旧 14 个 skill id 已从 dashboard 删除 —
    #   收菜 12 个 (cafe/schedule/club/daily_tasks/craft/pass_reward/
    #     event_activity/mail/shop/ap_planning/momo_talk/story_mining)
    #     全部并入 daily_routine.
    #   战斗扫荡 2 个 (bounty / arena) 全部并入 campaign_sweep.
    # 它们仍在 brain/pipeline.py 的 _skill_registry 里 (因为 daily_routine
    # 和 campaign_sweep 内部需要实例化), 但 dashboard 不再让用户单独勾选.
    # 老 profile 里残留的这些 id 会在 app.py 校验时被过滤掉, 用户重新
    # save profile 即可清理.

    # 战斗扫荡单跑入口 (live 单 skill 测试用 — 2026-06-09: 传 "bounty" 被过滤
    # → fallback 全套 daily_routine 真跑, step gate 拦住才没出事。这三个 id
    # 必须合法, 否则单测只能跑全套):
    {"id": "bounty", "label": "[测试] 悬赏通缉 单跑"},
    {"id": "arena", "label": "[测试] 战术大赛 单跑"},
    {"id": "jfd", "label": "[测试] 学院交流会 单跑"},
    {"id": "batch_sweep", "label": "批量掃蕩 (刷体力, 剩余AP全花)"},
    {"id": "special_sweep", "label": "[测试] 智能AP分配 — 扫2x/3x bonus板块(今天特殊任务)"},
    # 2026-07-08 活动规划器: Bonus解锁→活動點數优先→货币扫荡→领奖 (无活动自动 done)。
    {"id": "event_quest", "label": "活动规划器 (加成解锁+点数优先+货币扫荡)"},
    # 2026-06-11 编排重构: mail / daily_mission 升为顶层(厅后收口), 必须在
    # 白名单否则 _normalize_skill_order 静默过滤(第三次踩这个坑)。
    {"id": "mail", "label": "邮件箱收口"},
    {"id": "daily_mission", "label": "每日领奖 (n/8≥7)"},
    # 2026-06-12: 新顶层 skill 必须进白名单, 否则 _normalize_skill_order 静默过滤
    # (第五次同型陷阱). arena_shop = 战术大赛商店买体力(花战术大赛货币).
    {"id": "arena_shop", "label": "[测试] 战术大赛商店买体力 单跑"},
    # 2026-06-13: schedule/cafe 是 daily_routine 子 skill 但已在 pipeline._skill_registry,
    # 加进白名单即可单跑(step_mode 调阶段辨别+门控). 同型陷阱第六次.
    {"id": "schedule", "label": "[测试] 课程表 单跑"},
    {"id": "cafe", "label": "[测试] 咖啡厅 单跑"},
    # 2026-06-15: shop(信用点商店)/craft(制造)/momo_talk(社交) 也是 daily_routine
    # 子 skill, 加进白名单单跑验证 (同型陷阱第七次).
    {"id": "shop", "label": "[测试] 信用点商店 单跑"},
    {"id": "craft", "label": "[测试] 制造 单跑"},
    {"id": "momo_talk", "label": "[测试] MomoTalk挖矿 单跑(手动)"},
    # club = 社團签到(社交入口红点的真主人, 10AP进邮箱). 单跑验证.
    {"id": "club", "label": "[测试] 社團签到 单跑"},
    # buy_pyroxene = 每日免费组合包(只领免费, 绝不买付费). 单跑验证(陷阱第8次).
    {"id": "buy_pyroxene", "label": "[测试] 每日购买青辉石(免费组合包) 单跑"},
]
# Default order = the 10 production skills in display order.  Mail
# moved to the END so it captures today's club sign-in AP, event
# rewards, etc. that other skills push into the mailbox during this
# run.  Yesterday's accumulated mail (login bonus, etc.) is also
# claimed by the end-of-run Mail since BA's mailbox accumulates
# until claimed — no need for an additional start-of-run Mail.
_DEFAULT_SKILL_ORDER = [
    # ⭐canonical 日常顺序(用户 2026-07-11 定死): 收菜攒AP → 纯票扫荡 →
    # 学园交流会(吃AP) → 活动(剩余AP全灌+加成台账) → 战术大赛 → 邮件 →
    # 每日领奖(必须最后, 且大额扫荡后立跑防 server 3AM 重置吞箱) →
    # 活动再跑一轮(消化 mail/任务回灌的新AP, AP<20 自动秒过)。
    "daily_routine",
    "bounty",
    "jfd",
    "event_quest",
    "arena",
    "mail",
    "daily_mission",
    "event_quest",
]
_VALID_SKILL_IDS = {item["id"] for item in _SKILL_OPTIONS}

# DXcam capture state
_CAPTURE_LOCK = threading.Lock()
_CAPTURE_THREAD = None
_CAPTURE_RUNNING = False
_CAPTURE_STATUS = {"running": False, "frames": 0, "dataset": "", "error": ""}

# Single-step approval mode: when on, the pipeline PAUSES before each
# click/back/swipe, exposes the pending action via status/_STEP_PENDING, and
# blocks until POST /api/v1/step/go. Used for human-in-the-loop walk-throughs.
_STEP_MODE = False
_STEP_PENDING: Optional[Dict[str, Any]] = None
_STEP_GO = threading.Event()


def _box_iou_n(a, b) -> float:
    """IoU of two YoloBox-like objects with normalized x1/y1/x2/y2 coords."""
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
    return inter / ua if ua > 0 else 0.0

OCR_CACHE_VERSION = 2

# ── Pipeline state ─────────────────────────────────────────────────────
_PIPELINE_LOCK = threading.Lock()
_PIPELINE = None          # brain.pipeline.DailyPipeline instance
_PIPELINE_THREAD = None   # background worker thread
_PIPELINE_RUNNING = False
_PIPELINE_STATUS = {"running": False, "error": "", "ticks": 0}
_LAST_PIPELINE_ERROR = ""
_DISPLAY_SYNC_HZ = 240.0
_INPUT_POLL_HZ = 8000.0
_TIMER_RES_ENABLED = False
_PIPELINE_RUN_META: Dict[str, Any] = {}

# ── Daily scheduler ────────────────────────────────────────────────────
# Background thread that, when enabled in the active profile, auto-starts
# the pipeline at `reset_time` (HH:MM, local) and again every
# `interval_hours`. Manual POST /api/v1/start always wins — scheduler
# skips firing if a run is already active.
_SCHEDULER_THREAD: Optional[threading.Thread] = None
_SCHEDULER_STOP: threading.Event = threading.Event()
_SCHEDULER_STATUS: Dict[str, Any] = {
    "enabled": False,
    "next_fire_ts": 0.0,
    "last_fire_ts": 0.0,
    "last_result": "",
}


def _normalize_profile_name(value: Any) -> str:
    name = str(value or "").strip()
    if not name:
        return "default"
    return name[:64]


def _normalize_skill_order(values: Any) -> List[str]:
    # Mail is allowed to appear twice (run at pipeline start AND end so
    # both yesterday's accumulated mail and today's just-generated
    # rewards get claimed within a single run).  Other skills are
    # deduped — running e.g. cafe twice doesn't gain anything.
    # ×2 允许重复: mail(开局+收口) / batch_sweep(厅后+领奖后) / special_sweep
    # (2026-06-16 替 batch 当默认 AP-eater, 回马枪需出现两次否则被去重)。
    # +event_quest(2026-07-11 canonical序尾部再跑一轮消化 mail/任务回灌AP,
    # AP<20 时秒过零成本)
    _ALLOW_DUPLICATES = {"mail", "batch_sweep", "special_sweep", "event_quest"}
    order: List[str] = []
    seen: Set[str] = set()
    if isinstance(values, list):
        for item in values:
            skill_id = str(item or "").strip()
            if skill_id not in _VALID_SKILL_IDS:
                continue
            if skill_id in seen and skill_id not in _ALLOW_DUPLICATES:
                continue
            order.append(skill_id)
            seen.add(skill_id)
    if not order:
        return list(_DEFAULT_SKILL_ORDER)
    return order


def _default_profile_settings() -> Dict[str, Any]:
    return {
        "account_label": "",
        "game_exe_path": "",
        "window_title": "MuMu",
        "goal": "",
        "steps": 0,
        "step_sleep_s": 0.6,
        "dry_run": True,
        "forbid_premium_currency": True,
        "ap_purchase_limit": 0,
        "event_max_rounds": 1,
        "event_ap_reserve": 0,
        # Auto-pick event-bonus characters before quest sortie:
        # 快速編輯 → 自動 → 確認 → 出擊. FSM falls back to direct sortie
        # if any button isn't found within 30 ticks.
        "enable_bonus_team": True,
        # Specific stage to sweep in event_farming. template-based rationale:
        # the LAST stage in an event drops the highest-value shop currency
        # (P-shop items = premium shards / pyroxenes). 0 = old behavior
        # (bottom-most visible stage). 12 = typical 4-stage event finale.
        "event_farming_stage": 12,
        # P1b: hard AP ceiling for event_farming in one run. 0 = disabled
        # (run until max_rounds or ap_reserve). Typical 1-week event
        # budget: 200 sweeps × 40 AP = 8000.
        "event_farming_ap_budget": 0,
        # Minimum AP required before farming enters sweep. When AP < this,
        # just claim reward badges and skip to shop. Stage 12 ≈ 20 AP.
        "event_min_ap_for_sweep": 20,
        # P1c: event_shop behavior.
        #   auto_buy=True       → actually click 購買 on eligible items
        #                         (5:1 exchange traps auto-skipped)
        #   currencies=[]       → allow ALL currency tabs (recommended)
        #   currencies=["tab3"] → only spend from the listed tabs
        "event_shop_auto_buy": True,
        "event_shop_currencies": [],
        # Furniture items (interactive cafe decor — 可愛器皿組合 / 刺繡手帕
        # etc.) priority toggle. False (default) = buy after materials.
        # True = buy furniture first.
        "event_shop_furniture_first": False,
        "exploration_click": False,
        "notify_on_finish": False,
        "notify_webhook_url": "",
        "target_favorites": [],
        "skill_order": list(_DEFAULT_SKILL_ORDER),
        "bounty_branches": ["高架公路", "沙漠鐵道", "教室"],
        # Cafe invite targets — [1F, 2F] 中文角色名 matching the fused_avatar
        # model cls names (e.g. "莉央(战斗)"). cafe.py reads index 0 for 1F,
        # index 1 for 2F. Empty list = invite the first visible rows.
        "cafe_invite_targets": [],
        # Schedule (課程表) target students — 中文角色名 matching the
        # fused_avatar cls (e.g. "莉央(战斗)"). schedule.py Case B (all regions
        # max-level) only dispatches a room when one of these students sits in
        # it. Empty list = spend leftover tickets on any room (fallback).
        "schedule_target_students": [],
        # ── Extended profile config (schema-only; skills consume incrementally) ──
        # Daily scheduler (opt-in). When enabled, server auto-fires pipeline
        # at `reset_time` (HH:MM, local) and re-fires every `interval_hours`.
        "scheduler": {
            "enabled": False,
            "reset_time": "04:00",   # JP server reset; CN uses 04:00 too
            "interval_hours": 24,
        },
        # Paid / ticketed purchase caps (0 = never buy).
        "purchase_caps": {
            "arena_tickets": 0,
            "bounty_tickets": 0,      # 悬赏/rewarded_task
            "scrimmage_tickets": 0,
            "lesson_tickets": 0,
        },
        # Arena tuning.
        "arena": {
            "level_diff": 0,           # opponent level offset (-N .. +N)
            "stop_at_rank1": True,     # stop challenging once we hit rank 1
            "max_refresh": 0,          # refresh opponent pool up to N times
        },
        # Shop purchase strategy.
        "shop": {
            "common_priority_list": [],     # item name keywords in buy order (e.g. "信用點", "強化珠")
            "tactical_priority_list": [],   # tactical-challenge shop items
            "common_refresh_times": 0,      # extra refreshes on common shop
            "tactical_refresh_times": 0,
        },
        # Stage sweep explicit whitelists.
        "hard_task_list": [],             # e.g. ["H14-3", "H20-3"]
        "normal_task_list": [],           # e.g. ["14-1"] for mats/equipment
        # Lesson (course schedule) tuning.
        "lesson": {
            "relationship_first": True,   # prioritize affection > drop mats
        },
        # Craft / creation node priorities.
        "craft": {
            "use_acceleration_ticket": False,
            "phase_priority": ["光辉", "花朵"],  # node types ranked
        },
        # Drill / firing drill formations.
        "drill": {
            "difficulty_list": [],          # ["普通", "困難", "極限", "洞察"]
            "choose_team_method": "preset",
        },
    }


def _normalize_profile_settings(value: Any) -> Dict[str, Any]:
    raw = dict(value or {})
    data = _default_profile_settings()
    data["account_label"] = str(raw.get("account_label") or "").strip()
    data["game_exe_path"] = str(raw.get("game_exe_path") or "").strip()
    data["window_title"] = str(raw.get("window_title") or "MuMu").strip() or "MuMu"
    data["goal"] = str(raw.get("goal") or "").strip()
    try:
        data["steps"] = max(0, int(raw.get("steps") or 0))
    except Exception:
        data["steps"] = 0
    try:
        data["step_sleep_s"] = max(0.0, float(raw.get("step_sleep_s") or 0.6))
    except Exception:
        data["step_sleep_s"] = 0.6
    data["dry_run"] = bool(raw.get("dry_run") if raw.get("dry_run") is not None else True)
    data["forbid_premium_currency"] = bool(raw.get("forbid_premium_currency") if raw.get("forbid_premium_currency") is not None else True)
    try:
        data["ap_purchase_limit"] = max(0, int(raw.get("ap_purchase_limit") or 0))
    except Exception:
        data["ap_purchase_limit"] = 0
    try:
        data["event_max_rounds"] = max(1, min(20, int(raw.get("event_max_rounds") or 1)))
    except Exception:
        data["event_max_rounds"] = 1
    try:
        data["event_ap_reserve"] = max(0, int(raw.get("event_ap_reserve") or 0))
    except Exception:
        data["event_ap_reserve"] = 0
    data["exploration_click"] = bool(raw.get("exploration_click") if raw.get("exploration_click") is not None else False)
    data["notify_on_finish"] = bool(raw.get("notify_on_finish") if raw.get("notify_on_finish") is not None else False)
    data["notify_webhook_url"] = str(raw.get("notify_webhook_url") or "").strip()
    favs: List[str] = []
    seen_favs: Set[str] = set()
    for item in raw.get("target_favorites") or []:
        name = str(item or "").strip()
        if not name or name in seen_favs:
            continue
        favs.append(name)
        seen_favs.add(name)
    data["target_favorites"] = favs
    data["skill_order"] = _normalize_skill_order(raw.get("skill_order"))
    _VALID_BRANCHES = ("高架公路", "沙漠鐵道", "教室")
    raw_branches = raw.get("bounty_branches")
    if isinstance(raw_branches, list):
        norm: List[str] = []
        seen_b: Set[str] = set()
        for b in raw_branches:
            name = str(b or "").strip()
            if name in _VALID_BRANCHES and name not in seen_b:
                norm.append(name)
                seen_b.add(name)
        data["bounty_branches"] = norm if norm else list(_VALID_BRANCHES)

    # Cafe: skip the invite sub-flow entirely (ticket on CD / headpat-only).
    data["cafe_skip_invite"] = bool(raw.get("cafe_skip_invite", False))

    # Cafe invite targets — ordered list of 中文角色名 ([1F, 2F]); trim blanks
    # but PRESERVE order/position (index drives which floor invites whom).
    raw_cafe = raw.get("cafe_invite_targets")
    if isinstance(raw_cafe, list):
        data["cafe_invite_targets"] = [
            str(x or "").strip() for x in raw_cafe if str(x or "").strip()
        ]

    # Schedule target students — list of 中文角色名 (order irrelevant; dedup +
    # trim blanks). schedule.py Case B dispatches only rooms holding one of
    # these; empty = spend leftover tickets on any room.
    raw_sched = raw.get("schedule_target_students")
    if isinstance(raw_sched, list):
        sched_out: List[str] = []
        seen_sched: Set[str] = set()
        for x in raw_sched:
            name = str(x or "").strip()
            if name and name not in seen_sched:
                sched_out.append(name)
                seen_sched.add(name)
        data["schedule_target_students"] = sched_out

    # Extended config blocks: shallow-merge dict-valued keys and
    # pass through list/scalar keys as-is. Skills ignore unknown fields,
    # so type coercion can be added lazily when a skill starts consuming.
    for key in ("scheduler", "purchase_caps", "arena", "shop", "lesson", "craft", "drill"):
        incoming = raw.get(key)
        if isinstance(incoming, dict):
            merged = dict(data[key])
            merged.update(incoming)
            data[key] = merged
    for key in ("hard_task_list", "normal_task_list"):
        incoming = raw.get(key)
        if isinstance(incoming, list):
            data[key] = [str(x).strip() for x in incoming if str(x or "").strip()]
    return data


def _load_app_config() -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    if APP_CONFIG_PATH.exists():
        try:
            data = json.loads(APP_CONFIG_PATH.read_text("utf-8"))
        except Exception:
            data = {}
    profiles_raw = data.get("profiles")
    profiles: Dict[str, Any] = profiles_raw if isinstance(profiles_raw, dict) else {}
    if not profiles:
        profiles = {"default": {}}
    active_profile = _normalize_profile_name(data.get("active_profile") or next(iter(profiles.keys()), "default"))
    normalized_profiles: Dict[str, Dict[str, Any]] = {}
    for profile_name, profile_data in profiles.items():
        normalized_profiles[_normalize_profile_name(profile_name)] = _normalize_profile_settings(profile_data)
    if active_profile not in normalized_profiles:
        normalized_profiles[active_profile] = _default_profile_settings()
    root_favorites = data.get("target_favorites")
    if isinstance(root_favorites, list) and not normalized_profiles[active_profile].get("target_favorites"):
        normalized_profiles[active_profile]["target_favorites"] = _normalize_profile_settings({"target_favorites": root_favorites})["target_favorites"]
    data["profiles"] = normalized_profiles
    data["active_profile"] = active_profile
    data["target_favorites"] = list(normalized_profiles[active_profile].get("target_favorites") or [])
    return data


def _save_app_config(data: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _load_app_config()
    cfg.update(dict(data or {}))
    active_profile = _normalize_profile_name(cfg.get("active_profile") or "default")
    profiles_raw = cfg.get("profiles") or {}
    normalized_profiles: Dict[str, Dict[str, Any]] = {}
    for profile_name, profile_data in dict(profiles_raw).items():
        normalized_profiles[_normalize_profile_name(profile_name)] = _normalize_profile_settings(profile_data)
    if active_profile not in normalized_profiles:
        normalized_profiles[active_profile] = _default_profile_settings()
    cfg["profiles"] = normalized_profiles
    cfg["active_profile"] = active_profile
    cfg["target_favorites"] = list(normalized_profiles[active_profile].get("target_favorites") or [])
    APP_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    APP_CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), "utf-8")
    return cfg


def _get_active_profile_settings() -> tuple[str, Dict[str, Any], Dict[str, Any]]:
    cfg = _load_app_config()
    active_profile = _normalize_profile_name(cfg.get("active_profile") or "default")
    profiles = cfg.get("profiles") or {}
    profile = _normalize_profile_settings(profiles.get(active_profile) or {})
    return active_profile, profile, cfg


def _post_json(url: str, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        str(url),
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        resp.read(1)


def _notify_pipeline_finished(status_name: str, summary: str, progress: Optional[Dict[str, Any]], meta: Dict[str, Any]) -> None:
    webhook_url = str(meta.get("notify_webhook_url") or "").strip()
    if not webhook_url or not bool(meta.get("notify_on_finish")):
        return
    account_label = str(meta.get("account_label") or meta.get("profile_name") or "default").strip() or "default"
    profile_name = str(meta.get("profile_name") or "default").strip() or "default"
    results = list((progress or {}).get("results") or [])
    total_skills = int((progress or {}).get("total_skills") or len(meta.get("skill_names") or []))
    done_count = len([row for row in results if str(row.get("status") or "") == "done"])
    text = "\n".join([
        f"[私人碧蓝档案助手] {account_label}",
        f"状态: {status_name}",
        f"档案: {profile_name}",
        f"完成技能: {done_count}/{total_skills}",
        summary or "(no summary)",
    ])
    low = webhook_url.lower()
    if "discord.com/api/webhooks" in low:
        _post_json(webhook_url, {"content": text})
        return
    if "api.telegram.org" in low and "sendmessage" in low:
        _post_json(webhook_url, {"text": text})
        return
    _post_json(
        webhook_url,
        {
            "title": f"私人碧蓝档案助手 · {account_label}",
            "status": status_name,
            "profile_name": profile_name,
            "account_label": account_label,
            "done_skills": done_count,
            "total_skills": total_skills,
            "summary": summary,
            "results": results,
        },
    )


def _enable_high_resolution_timer() -> None:
    global _TIMER_RES_ENABLED
    if _TIMER_RES_ENABLED:
        return
    try:
        ctypes.WinDLL("winmm", use_last_error=True).timeBeginPeriod(1)
        _TIMER_RES_ENABLED = True
    except Exception:
        pass


def _high_res_sleep(seconds: float) -> None:
    delay = float(seconds)
    if delay <= 0:
        return
    end_t = time.perf_counter() + delay
    if delay > 0.003:
        time.sleep(max(0.0, delay - 0.0015))
    while time.perf_counter() < end_t:
        pass


def _pid_alive(pid: int) -> bool:
    try:
        if pid <= 0:
            return False
    except Exception:
        return False

    if os.name == "nt":
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_bool, ctypes.c_ulong)
            kernel32.OpenProcess.restype = ctypes.c_void_p
            kernel32.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong))
            kernel32.GetExitCodeProcess.restype = ctypes.c_bool
            kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
            kernel32.CloseHandle.restype = ctypes.c_bool
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259

            h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
            if not h:
                return False
            code = ctypes.c_ulong(0)
            ok = kernel32.GetExitCodeProcess(h, ctypes.byref(code))
            try:
                kernel32.CloseHandle(h)
            except Exception:
                pass
            if not ok:
                return False
            return int(code.value) == STILL_ACTIVE
        except Exception:
            return False

    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def _parent_watchdog() -> None:
    try:
        pid_str = os.environ.get("GAMESECRETARY_PARENT_PID") or "0"
        pid = int(pid_str)
    except Exception:
        pid = 0
    
    print(f"DEBUG: Watchdog init. Parent PID: {pid}", flush=True)
    
    if pid <= 0:
        return

    while True:
        try:
            if not _pid_alive(pid):
                print(f"DEBUG: Watchdog triggered. Parent {pid} gone. Exiting.", flush=True)
                try:
                    _stop_pipeline()
                except Exception:
                    pass
                try:
                    # Kill entire process group if possible
                    if hasattr(os, "killpg"):
                        os.killpg(os.getpgrp(), 9)
                    else:
                        os._exit(0)
                except Exception:
                    os._exit(0)
                break
        except Exception as e:
            print(f"DEBUG: Watchdog error: {e}", flush=True)
        time.sleep(2.0)


app = FastAPI()


# ── Daily scheduler worker ────────────────────────────────────────────
def _compute_next_fire(now: float, reset_hhmm: str, interval_hours: int) -> float:
    """Next fire time: aligned to today's `reset_hhmm`, rolled by `interval_hours`."""
    import datetime as _dt
    try:
        hh, mm = reset_hhmm.split(":")
        hh, mm = int(hh), int(mm)
    except Exception:
        hh, mm = 4, 0
    interval_s = max(3600, int(interval_hours) * 3600)  # clamp min 1h
    now_dt = _dt.datetime.fromtimestamp(now)
    base = now_dt.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if base.timestamp() <= now:
        # advance by interval until it's in the future
        advance = _dt.timedelta(seconds=interval_s)
        while base.timestamp() <= now:
            base = base + advance
    return base.timestamp()


def _scheduler_loop() -> None:
    print("[Scheduler] thread started", flush=True)
    while not _SCHEDULER_STOP.is_set():
        try:
            _, prof, _ = _get_active_profile_settings()
            sched = (prof.get("scheduler") or {})
            enabled = bool(sched.get("enabled"))
            _SCHEDULER_STATUS["enabled"] = enabled
            if not enabled:
                _SCHEDULER_STATUS["next_fire_ts"] = 0.0
                _SCHEDULER_STOP.wait(30.0)
                continue
            reset_hhmm = str(sched.get("reset_time") or "04:00")
            interval = int(sched.get("interval_hours") or 24)
            now = time.time()
            next_fire = _SCHEDULER_STATUS.get("next_fire_ts") or 0.0
            if next_fire <= 0 or next_fire <= now:
                next_fire = _compute_next_fire(now, reset_hhmm, interval)
                _SCHEDULER_STATUS["next_fire_ts"] = next_fire
            sleep_s = max(5.0, min(60.0, next_fire - now))
            if _SCHEDULER_STOP.wait(sleep_s):
                break
            now = time.time()
            if now >= _SCHEDULER_STATUS["next_fire_ts"]:
                if _PIPELINE_RUNNING:
                    _SCHEDULER_STATUS["last_result"] = "skipped: already running"
                else:
                    try:
                        _start_pipeline(payload=dict(prof))
                        _SCHEDULER_STATUS["last_result"] = "fired"
                    except Exception as e:
                        _SCHEDULER_STATUS["last_result"] = f"error: {e!r}"
                _SCHEDULER_STATUS["last_fire_ts"] = now
                # schedule next
                _SCHEDULER_STATUS["next_fire_ts"] = _compute_next_fire(
                    now, reset_hhmm, interval,
                )
        except Exception:
            traceback.print_exc()
            _SCHEDULER_STOP.wait(30.0)
    print("[Scheduler] thread stopped", flush=True)


@app.on_event("startup")
def _scheduler_startup() -> None:
    global _SCHEDULER_THREAD
    if _SCHEDULER_THREAD and _SCHEDULER_THREAD.is_alive():
        return
    _SCHEDULER_STOP.clear()
    _SCHEDULER_THREAD = threading.Thread(
        target=_scheduler_loop, name="DailyScheduler", daemon=True,
    )
    _SCHEDULER_THREAD.start()


@app.on_event("shutdown")
def _scheduler_shutdown() -> None:
    _SCHEDULER_STOP.set()


@app.get("/api/v1/scheduler/status")
def api_scheduler_status() -> Dict[str, Any]:
    return dict(_SCHEDULER_STATUS)


# Print env vars for debugging
try:
    print(f"DEBUG: Backend starting. Env GAMESECRETARY_PARENT_PID={os.environ.get('GAMESECRETARY_PARENT_PID')}", flush=True)
    if os.environ.get("GAMESECRETARY_PARENT_PID"):
        t = threading.Thread(target=_parent_watchdog, daemon=True)
        t.start()
        print("DEBUG: Watchdog thread started", flush=True)
except Exception as e:
    print(f"DEBUG: Watchdog start failed: {e}", flush=True)


# ── Pipeline control ──────────────────────────────────────────────────

def _start_pipeline(*, payload: Dict[str, Any]) -> None:
    """Start the DailyPipeline in a background thread."""
    global _PIPELINE, _PIPELINE_THREAD, _PIPELINE_RUNNING, _PIPELINE_STATUS, _PIPELINE_RUN_META
    with _PIPELINE_LOCK:
        if _PIPELINE_RUNNING:
            return
        from brain.pipeline import DailyPipeline
        active_profile, profile_settings, _ = _get_active_profile_settings()
        skill_names = _normalize_skill_order(payload.get("skill_order") or profile_settings.get("skill_order"))
        profile_options = dict(profile_settings)
        for key in [
            "steps",
            "goal",
            "forbid_premium_currency",
            "ap_purchase_limit",
            "event_max_rounds",
            "event_ap_reserve",
            "exploration_click",
            "sub_only",
        ]:
            if payload.get(key) is not None:
                profile_options[key] = payload.get(key)

        _PIPELINE = DailyPipeline(skill_names=skill_names, profile_options=profile_options)
        _PIPELINE.start()
        _PIPELINE_RUNNING = True

        window_title = str(payload.get("window_title") or profile_settings.get("window_title") or "Blue Archive")
        step_sleep = float(payload.get("step_sleep_s") or profile_settings.get("step_sleep_s") or 0.6)
        dry_run = bool(payload.get("dry_run") if payload.get("dry_run") is not None else profile_settings.get("dry_run", True))
        globals()["_STEP_MODE"] = bool(payload.get("step_mode", False))
        globals()["_GAME_OVERLAY"] = str(payload.get("game_overlay")
                                         or profile_settings.get("game_overlay")
                                         or "battle_only").strip().lower()
        globals()["_STEP_PENDING"] = None
        _STEP_GO.clear()
        account_label = str(payload.get("account_label") or profile_settings.get("account_label") or active_profile).strip()
        _PIPELINE_RUN_META = {
            "profile_name": active_profile,
            "account_label": account_label,
            "notify_on_finish": bool(payload.get("notify_on_finish") if payload.get("notify_on_finish") is not None else profile_settings.get("notify_on_finish")),
            "notify_webhook_url": str(payload.get("notify_webhook_url") or profile_settings.get("notify_webhook_url") or "").strip(),
            "skill_names": list(skill_names),
            "window_title": window_title,
            "dry_run": dry_run,
            "forbid_premium_currency": bool(profile_options.get("forbid_premium_currency", True)),
            "ap_purchase_limit": int(profile_options.get("ap_purchase_limit") or 0),
            "steps": int(profile_options.get("steps") or 0),
        }
        _PIPELINE_STATUS = {
            "running": True,
            "error": "",
            "ticks": 0,
            "profile_name": active_profile,
            "account_label": account_label,
            "skill_order": list(skill_names),
        }

        _PIPELINE_THREAD = threading.Thread(
            target=_pipeline_worker,
            args=(window_title, step_sleep, dry_run),
            daemon=True,
        )
        _PIPELINE_THREAD.start()


def _stop_pipeline() -> None:
    """Stop the running pipeline."""
    global _PIPELINE, _PIPELINE_RUNNING
    with _PIPELINE_LOCK:
        _PIPELINE_RUNNING = False
        _STEP_GO.set()  # wake a worker blocked on step-approval so it can exit
        if _PIPELINE is not None:
            try:
                _PIPELINE.stop()
            except Exception:
                pass
            _PIPELINE = None
        _PIPELINE_STATUS["running"] = False


def _pipeline_worker(window_title: str, step_sleep: float, dry_run: bool) -> None:
    """Background thread: screenshot -> OCR -> skill decision -> execute action."""
    global _PIPELINE_RUNNING, _PIPELINE_STATUS
    try:
        from scripts.win_capture import (
            capture_client, find_window_by_title_substring,
            find_largest_visible_child,
        )
        from mumu_runner import AdbInput
        import cv2
        import numpy as np

        _enable_high_resolution_timer()
        # Predefine so the `finally` cleanup never hits UnboundLocalError when we
        # early-return before the high-FPS thread is created (e.g. window not
        # found) — otherwise the real "Window not found" error gets masked.
        _yolo_hfps = None

        hwnd = find_window_by_title_substring(window_title)
        if not hwnd:
            _PIPELINE_STATUS["error"] = f"Window '{window_title}' not found"
            _PIPELINE_RUNNING = False
            _PIPELINE_STATUS["running"] = False
            return

        # For emulators (MuMu, etc.), capture from the render child window
        # so the toolbar/tabs are excluded and coordinates map correctly.
        render_hwnd = hwnd
        try:
            child = find_largest_visible_child(int(hwnd))
            if child:
                render_hwnd = int(child)
                _log_pipeline(f"Using render child hwnd={render_hwnd} (parent={hwnd})")
        except Exception:
            pass

        # Store top-level hwnd for SetForegroundWindow (child windows can't be foregrounded)
        _set_parent_hwnd(int(hwnd))

        # Overlay policy (2026-06-11, user rule): boxes ONLY in battle mode.
        # The daily loop's overlay lags detection by a tick on static UI
        # (drift / flicker / size-jitter) and burns into window-captured
        # frames (training poison — see flywheel overlay-burn incident).
        # Battle tooling (battle_overlay_demo / future combat runner) draws
        # its own Kalman-smoothed overlay where boxes can actually lock on.
        # profile/payload `game_overlay: "always"` re-enables here for debug.
        _overlay = None
        if globals().get("_GAME_OVERLAY") == "always":
            try:
                from scripts.yolo_overlay import YoloOverlay
                # track=False: raw current-frame boxes, no velocity-coast
                # (BoxTracker lead-aim flung boxes around on static screens).
                _overlay = YoloOverlay(render_hwnd, track=False)
                _overlay.start()
                _log_pipeline(f"YOLO overlay started (debug 'always' mode) on render_hwnd={render_hwnd}")
            except Exception as e:
                _log_pipeline(f"YOLO overlay unavailable: {e}")
        else:
            _log_pipeline("YOLO overlay OFF (battle_only policy — daily frames stay clean)")

        adb = None
        android_w, android_h = 1280, 720
        if not dry_run:
            try:
                adb_serial = str(os.environ.get("ADB_SERIAL") or "").strip()
                adb_host = "127.0.0.1"
                adb_port = 7555
                if ":" in adb_serial:
                    host_part, port_part = adb_serial.rsplit(":", 1)
                    adb_host = host_part.strip() or adb_host
                    try:
                        adb_port = max(1, int(port_part.strip()))
                    except Exception:
                        pass
                adb = AdbInput(host=adb_host, port=adb_port)
                if not adb.connect():
                    _log_pipeline(f"ADB connect failed ({adb_host}:{adb_port}); switching pipeline to dry-run")
                    adb = None
                    dry_run = True
                else:
                    try:
                        android_w, android_h = adb.screen_size()
                    except Exception:
                        pass
                    _log_pipeline(f"ADB ready {adb_host}:{adb_port} size={android_w}x{android_h}")
                    # Money-read defense: let skills pull overlay-free frames
                    # (DXcam frames carry burned-in overlay boxes that can kill
                    # small-icon detection — see brain.pipeline.get_clean_frame).
                    try:
                        from brain.pipeline import set_clean_frame_source
                        set_clean_frame_source(adb.capture_frame)
                        _log_pipeline("clean-frame source registered (ADB)")
                    except Exception as e:
                        _log_pipeline(f"clean-frame source unavailable: {e}")
            except Exception as e:
                _log_pipeline(f"ADB unavailable: {e}; switching pipeline to dry-run")
                adb = None
                dry_run = True

        _log_pipeline(f"Pipeline worker started. window='{window_title}' hwnd={hwnd} render={render_hwnd} sleep={step_sleep} dry_run={dry_run}")

        # ── Clean-flywheel recorder (user rule 2026-06-09: 每次启动 bot 实跑都
        # 录干净帧当迭代素材). ADB screencap runs INSIDE Android — the Win32
        # overlay doesn't exist there, so frames are guaranteed overlay-free
        # (DXcam/trajectory frames burn the boxes in). Each capture is its own
        # adb subprocess (stateless, thread-safe vs. the input taps). Low rate
        # (1 frame / 2.5s) keeps the cost invisible to the tick loop.
        if adb is not None:
            _clean_dir = RAW_IMAGES_DIR / ("run_" + time.strftime("%Y%m%d_%H%M%S") + "_clean")
            # Interval env-tunable (2026-07-06): battle-material runs want denser
            # frames (e.g. 1.2s) — VFX/battle scenes change fast; menus dedup out
            # later anyway. All captures serialize through AdbInput._IO_LOCK with
            # the taps, so a faster rate cannot re-introduce the tap-loss bug the
            # way a parallel ADB process would (it would bypass the lock).
            try:
                _fly_iv = max(0.5, float(os.environ.get("GAMESEC_FLYWHEEL_INTERVAL", "2.5")))
            except Exception:
                _fly_iv = 2.5

            def _clean_flywheel_worker():
                import cv2 as _cv2
                _clean_dir.mkdir(parents=True, exist_ok=True)
                idx = 0
                _log_pipeline(f"clean-flywheel recorder → {_clean_dir.name} (ADB, overlay-free, {_fly_iv}s)")
                while _PIPELINE_RUNNING:
                    try:
                        fr = adb.capture_frame()
                        if fr is not None:
                            _cv2.imwrite(str(_clean_dir / f"frame_{idx:06d}.jpg"), fr,
                                         [int(_cv2.IMWRITE_JPEG_QUALITY), 92])
                            idx += 1
                    except Exception:
                        pass
                    time.sleep(_fly_iv)
                _log_pipeline(f"clean-flywheel recorder stopped ({idx} frames)")

            threading.Thread(target=_clean_flywheel_worker, daemon=True).start()

        # OCR + YOLO lazy-load on first use (no pre-warm to avoid deadlocks).
        # Florence pre-warm in background thread (needed for Lobby + Schedule avatar matching).
        def _prewarm_florence():
            try:
                from vision.florence_vision import get_florence_vision
                fv = get_florence_vision()
                print(f"[Pipeline] Florence pre-warm: ready (device={fv.cfg.device})", flush=True)
            except Exception as e:
                print(f"[Pipeline] Florence pre-warm failed: {e}", flush=True)
        threading.Thread(target=_prewarm_florence, daemon=True).start()
        _florence_prewarm_started = True

        # DXcam for fast capture (Desktop Duplication API)
        # Detect which monitor the MuMu window is on for multi-monitor support
        _dxcam_camera = None
        _dxcam_region = None
        _mon_offset_x, _mon_offset_y = 0, 0  # monitor origin for multi-monitor coord conversion
        try:
            import dxcam as _dxcam_mod
            import ctypes.wintypes as _wt

            # Find which monitor the window center is on
            # Use DPI-aware context so GetWindowRect returns physical pixels
            # (critical at 150% scaling: prevents capture region mismatch)
            from scripts.win_capture import _dpi_aware_context
            _dxcam_rc = _wt.RECT()
            with _dpi_aware_context():
                ctypes.windll.user32.GetWindowRect(render_hwnd, ctypes.byref(_dxcam_rc))
            win_cx = (_dxcam_rc.left + _dxcam_rc.right) // 2
            win_cy = (_dxcam_rc.top + _dxcam_rc.bottom) // 2

            _monitor_idx = 0
            try:
                _MONITOR_DEFAULTTONEAREST = 2
                hmon = ctypes.windll.user32.MonitorFromPoint(
                    ctypes.wintypes.POINT(win_cx, win_cy), _MONITOR_DEFAULTTONEAREST
                )
                # Enumerate monitors to find index
                _monitors = []
                _MONITORENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.wintypes.RECT), ctypes.c_long)
                def _enum_cb(hMon, hdc, lprc, lParam):
                    _monitors.append(hMon)
                    return 1
                ctypes.windll.user32.EnumDisplayMonitors(None, None, _MONITORENUMPROC(_enum_cb), 0)
                for i, m in enumerate(_monitors):
                    if m == hmon:
                        _monitor_idx = i
                        break
                _log_pipeline(f"Window on monitor {_monitor_idx} (of {len(_monitors)})")
            except Exception:
                _monitor_idx = 0

            _dxcam_camera = _dxcam_mod.create(output_idx=_monitor_idx, output_color="BGR")
            # Convert virtual screen coords to monitor-relative coords
            # On multi-monitor, GetWindowRect returns virtual coords (e.g. x=3840 on monitor 2)
            # but DXcam expects coords relative to the specific monitor's origin.
            _mon_offset_x, _mon_offset_y = 0, 0
            try:
                class _MONITORINFO(ctypes.Structure):
                    _fields_ = [
                        ("cbSize", ctypes.c_ulong),
                        ("rcMonitor", _wt.RECT),
                        ("rcWork", _wt.RECT),
                        ("dwFlags", ctypes.c_ulong),
                    ]
                _mi = _MONITORINFO()
                _mi.cbSize = ctypes.sizeof(_MONITORINFO)
                with _dpi_aware_context():
                    if ctypes.windll.user32.GetMonitorInfoW(hmon, ctypes.byref(_mi)):
                        _mon_offset_x = _mi.rcMonitor.left
                        _mon_offset_y = _mi.rcMonitor.top
                        _log_pipeline(f"Monitor {_monitor_idx} origin: ({_mon_offset_x}, {_mon_offset_y})")
            except Exception:
                pass
            _dxcam_region = (
                _dxcam_rc.left - _mon_offset_x,
                _dxcam_rc.top - _mon_offset_y,
                _dxcam_rc.right - _mon_offset_x,
                _dxcam_rc.bottom - _mon_offset_y,
            )
            _log_pipeline(f"DXcam region: {_dxcam_region} (window rect: L={_dxcam_rc.left} T={_dxcam_rc.top} R={_dxcam_rc.right} B={_dxcam_rc.bottom})")
            _test = _dxcam_camera.grab(region=_dxcam_region)
            if _test is not None:
                _log_pipeline(f"DXcam capture OK: {_test.shape[1]}x{_test.shape[0]} region_w={_dxcam_region[2]-_dxcam_region[0]} region_h={_dxcam_region[3]-_dxcam_region[1]} (monitor {_monitor_idx})")
            else:
                _dxcam_camera = None
                _log_pipeline("DXcam grab returned None, falling back to ADB/BitBlt")
        except Exception as e:
            _log_pipeline(f"DXcam unavailable ({e}), falling back to ADB/BitBlt")
            _dxcam_camera = None

        # ── High-FPS YOLO detection thread (battle-grade) ──
        # Owns DXcam exclusively (not thread-safe for concurrent grab).
        # Runs YOLO at ~30 FPS, feeds overlay with ByteTrack-tracked boxes.
        # Shares latest YOLO results AND latest frame with pipeline tick.
        _yolo_latest_boxes = []    # shared: latest YOLO results
        _yolo_latest_frame = None  # shared: latest DXcam frame for pipeline tick
        _yolo_latest_ts = 0.0      # shared: timestamp of latest good frame
        _yolo_latest_lock = threading.Lock()
        _yolo_thread_running = True

        def _yolo_highfps_thread():
            """DXcam → YOLO → tracker → overlay, like battle_overlay_demo.py."""
            nonlocal _yolo_latest_boxes, _yolo_latest_frame, _yolo_latest_ts, _yolo_thread_running
            # Wait for pipeline to start and YOLO model to lazy-load on first tick
            time.sleep(5)
            try:
                from brain.pipeline import _run_yolo_on_image
            except Exception as e:
                _log_pipeline(f"YOLO high-FPS thread import failed: {e}")
                return
            # GPU budget (2026-06-11, live-caught: 68% GPU util lagged the
            # user's machine): 30 FPS continuous inference exists for the
            # battle overlay. With the overlay OFF (battle_only policy) the
            # only consumer is the tick loop (1-2s cadence) — 2 FPS keeps
            # frames fresh at ~1/15th the GPU burn.
            if globals().get("_GAME_OVERLAY") == "always":
                _yolo_fps_target = 30
            else:
                _yolo_fps_target = 2
            _log_pipeline(f"YOLO high-FPS thread started ({_yolo_fps_target} FPS"
                          f"{' — overlay off, low-power mode' if _yolo_fps_target == 2 else ''})")
            _interval = 1.0 / _yolo_fps_target
            _frame_count = 0
            _errors = 0
            # Static-mode anti-flicker: a box that drops out for 1-2 detection
            # frames (conf flutter at the floor) is HELD for up to 2 frames
            # (~66ms) so the overlay doesn't blink. Cleared on page switch
            # fast enough to be imperceptible; disabled in tracker mode.
            _hold_boxes: Dict[Any, Any] = {}
            while _yolo_thread_running and _PIPELINE_RUNNING:
                try:
                    t0 = time.perf_counter()
                    frame = None
                    if _dxcam_camera is not None:
                        try:
                            import ctypes.wintypes as _wt3
                            _rc3 = _wt3.RECT()
                            ctypes.windll.user32.GetWindowRect(render_hwnd, ctypes.byref(_rc3))
                            rgn = (
                                _rc3.left - _mon_offset_x,
                                _rc3.top - _mon_offset_y,
                                _rc3.right - _mon_offset_x,
                                _rc3.bottom - _mon_offset_y,
                            )
                            frame = _dxcam_camera.grab(region=rgn)
                        except Exception:
                            frame = None
                    if frame is None:
                        # DXcam returned None — clear shared frame after timeout
                        # so pipeline falls back to ADB (background mode support)
                        with _yolo_latest_lock:
                            if _yolo_latest_ts > 0 and time.perf_counter() - _yolo_latest_ts > 2.0:
                                _yolo_latest_frame = None
                                _yolo_latest_ts = 0.0
                        time.sleep(0.01)
                        continue
                    if frame.mean() < 10:
                        # Black frame (window minimized/off-screen)
                        with _yolo_latest_lock:
                            if _yolo_latest_ts > 0 and time.perf_counter() - _yolo_latest_ts > 2.0:
                                _yolo_latest_frame = None
                                _yolo_latest_ts = 0.0
                        time.sleep(0.01)
                        continue
                    h, w = frame.shape[:2]
                    yolo_boxes = _run_yolo_on_image(frame, w, h)
                    with _yolo_latest_lock:
                        _yolo_latest_boxes = yolo_boxes
                        _yolo_latest_frame = frame
                        _yolo_latest_ts = time.perf_counter()
                    if _overlay and _overlay.is_alive:
                        pipe_ref = None
                        with _PIPELINE_LOCK:
                            pipe_ref = _PIPELINE
                        # current_skill is the meta DailyRoutine — dig into its
                        # active sub-skill for the REAL name + sub_state (the old
                        # `skill_name=="Cafe"` was always False since the meta is
                        # named "DailyRoutine").
                        active_name, active_sub = "", ""
                        cur = pipe_ref.current_skill if pipe_ref and pipe_ref.current_skill else None
                        if cur is not None:
                            active_name = getattr(cur, "name", "")
                            active_sub = getattr(cur, "sub_state", "")
                            if hasattr(cur, "_plan") and hasattr(cur, "_cur_idx"):
                                try:
                                    sub_sk = cur._plan[cur._cur_idx][0]
                                    active_name = getattr(sub_sk, "name", active_name)
                                    active_sub = getattr(sub_sk, "sub_state", "")
                                except Exception:
                                    pass
                        in_cafe = active_name == "Cafe"
                        # Moving targets ⇒ tracker ON only for cafe 摸头 (students
                        # walk). Everything else = static UI → raw boxes, no coast,
                        # no leftover box when the page switches.
                        _track_on = in_cafe and active_sub == "headpat"
                        _overlay.set_track(_track_on)
                        overlay_out = []
                        for yb in yolo_boxes:
                            if hasattr(yb, "cls_name") and "headpat" in yb.cls_name.lower() and not in_cafe:
                                continue
                            overlay_out.append(yb)
                        # (cross-class/model dedup now lives in _run_yolo_on_image)
                        if _track_on:
                            _hold_boxes.clear()
                            _overlay.update(overlay_out)
                        else:
                            # Anti-flicker hold: bridge 1-2 frame dropouts.
                            out = list(overlay_out)
                            for k2 in list(_hold_boxes.keys()):
                                exp, hb = _hold_boxes[k2]
                                if exp < _frame_count:
                                    del _hold_boxes[k2]
                                    continue
                                # re-detected nearby this frame? then no hold copy
                                if any(getattr(c, "cls_name", "") == k2[0]
                                       and abs(c.cx - hb.cx) < 0.03
                                       and abs(c.cy - hb.cy) < 0.03
                                       for c in overlay_out):
                                    continue
                                out.append(hb)
                            for yb in overlay_out:
                                _hold_boxes[(yb.cls_name, int(yb.cx * 50), int(yb.cy * 50))] = (
                                    _frame_count + 2, yb)
                            _overlay.update(out)
                    _frame_count += 1
                    _errors = 0
                    elapsed = time.perf_counter() - t0
                    sleep_t = max(0, _interval - elapsed)
                    time.sleep(sleep_t)
                except Exception as e:
                    _errors += 1
                    if _errors <= 3:
                        _log_pipeline(f"YOLO high-FPS error #{_errors}: {e}")
                    time.sleep(0.5)
                    if _errors > 20:
                        _log_pipeline("YOLO high-FPS too many errors, stopping")
                        break
            _log_pipeline(f"YOLO high-FPS thread stopped ({_frame_count} frames)")

        _yolo_hfps = threading.Thread(target=_yolo_highfps_thread, daemon=True)
        _yolo_hfps.start()

        _OCR_INTERVAL = 3  # run OCR every 3rd tick
        _prev_ocr_boxes = None
        _tick_counter = 0
        _inline_yolo_logged = False

        while _PIPELINE_RUNNING:
            pipe = None
            with _PIPELINE_LOCK:
                pipe = _PIPELINE
            if pipe is None or not pipe.is_running:
                break

            _tick_counter += 1

            # 1. Capture — ADB FIRST (主tick抓帧ADB化, 2026-06-12): the Android
            #    internal screencap is full-res 3840x2160, overlay-free, and
            #    window-size independent. The old DXcam-first path fed the tick
            #    whatever the DESKTOP WINDOW size was (~900px when small) —
            #    digit-OCR physically impossible (топ-bar credit 28,096,458
            #    read as 96,458 → shop refused to buy twice; dialog 持有數量
            #    unreadable at 10px). DXcam shared frame = fallback only;
            #    BitBlt last resort. Bonus: trajectory frames become train-grade
            #    clean flywheel material at full res.
            frame = None
            _frame_src = ""
            if adb is not None:
                try:
                    frame = adb.capture_frame()
                    _frame_src = "adb"
                except Exception:
                    frame = None
            if frame is None:
                with _yolo_latest_lock:
                    if (_yolo_latest_frame is not None
                            and time.perf_counter() - _yolo_latest_ts < 3.0):
                        frame = _yolo_latest_frame.copy()
                        _frame_src = "dxcam"
            if frame is None:
                try:
                    pil_img = capture_client(render_hwnd)
                    if pil_img is not None:
                        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                        _frame_src = "bitblt"
                except Exception:
                    pass
            if frame is None:
                _high_res_sleep(0.1)
                continue

            # Skip black/loading frames (DXcam returns black during transitions)
            if frame.mean() < 10:
                _high_res_sleep(0.1)
                continue

            # Save to temp file for trajectory
            tmp_path = str(REPO_ROOT / "data" / "_pipeline_frame.jpg")
            cv2.imwrite(tmp_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

            # 2. Pipeline tick — OCR at tick rate, YOLO from high-FPS thread
            # ⭐OCR on-demand (2026-07-11): 恒 True, 真正决定权在
            # read_screen_from_frame — YOLO ≥3框(已知屏)零 OCR, <3框(盲区)才跑
            skip_ocr = True
            # YOLO source must match the frame source: ADB frames run INLINE
            # YOLO (boxes correspond to this exact full-res frame). Inject the
            # high-FPS thread's boxes only when we actually fell back to its
            # DXcam frame (they were computed on that feed).
            _injected_yolo = None
            if _frame_src == "dxcam" and _dxcam_camera is not None:
                with _yolo_latest_lock:
                    if (_yolo_latest_frame is not None
                            and time.perf_counter() - _yolo_latest_ts < 3.0):
                        _injected_yolo = list(_yolo_latest_boxes)
            elif not _inline_yolo_logged:
                _log_pipeline(f"main tick frame source = {_frame_src or 'none'}: "
                              f"running YOLO inline at pipeline tick rate")
                _inline_yolo_logged = True
            # 高频线程新鲜检出直通 skill(2026-07-11 工业级链路: 轮播类时敏
            # 点击不能吃主 tick 的 ~2.2s 帧龄)
            with _yolo_latest_lock:
                _fresh_b = list(_yolo_latest_boxes) if _yolo_latest_ts > 0 else None
                _fresh_f = _yolo_latest_frame      # 引用(线程替换式更新, 只读安全)
                _fresh_t = _yolo_latest_ts
            action = pipe.tick_from_frame(frame, screenshot_path=tmp_path,
                                          skip_ocr=skip_ocr,
                                          prev_ocr_boxes=_prev_ocr_boxes,
                                          injected_yolo_boxes=_injected_yolo,
                                          fresh_boxes=_fresh_b,
                                          fresh_frame=_fresh_f, fresh_ts=_fresh_t)
            # Cache OCR boxes for reuse on OCR-skip ticks
            if not skip_ocr and pipe.last_screen:
                _prev_ocr_boxes = pipe.last_screen.ocr_boxes
            action_type = action.get("action", "")
            reason = action.get("reason", "")
            # Loading gate (稳定规则 2026-06-11): 加载中 visible → never act this
            # tick, the game is mid-transition. Skills check is_loading() too,
            # but stale injected boxes can slip an action through — this is the
            # belt-and-braces hold at the execution layer.
            if (action_type in ("click", "back", "swipe", "swipe_tap")
                    and pipe.last_screen is not None
                    and pipe.last_screen.is_loading()):
                _log_pipeline(f"loading gate: 加载中 on screen → holding '{reason}'")
                action = {"action": "wait", "duration_ms": 900,
                          "reason": f"loading-gate hold ({reason})"}
                action_type = "wait"
                reason = action["reason"]
            _PIPELINE_STATUS["ticks"] = pipe._total_ticks
            _PIPELINE_STATUS["progress"] = pipe.progress

            _log_pipeline(f"tick={pipe._total_ticks} action={action_type} reason={reason}")

            # Model pre-warm threads already started at worker init (above main loop)

            # Overlay YOLO boxes are updated by the high-FPS YOLO thread above.
            # Here we only inject Florence + template boxes (these run at tick rate).
            if _overlay and _overlay.is_alive:
                screen = pipe.last_screen
                if screen and (screen.florence_boxes or screen.template_hits):
                    current_skill_name = pipe.current_skill.name if pipe.current_skill else ""
                    in_cafe = current_skill_name == "Cafe"
                    extra_boxes: List[Any] = []
                    extra_boxes.extend(
                        {
                            "cls": f"Florence:{box.text}",
                            "conf": box.confidence,
                            "x1": box.x1,
                            "y1": box.y1,
                            "x2": box.x2,
                            "y2": box.y2,
                        }
                        for box in screen.florence_boxes
                    )
                    if in_cafe:
                        extra_boxes.extend(
                            {
                                "cls": h.label,
                                "conf": h.confidence,
                                "x1": h.x1,
                                "y1": h.y1,
                                "x2": h.x2,
                                "y2": h.y2,
                            }
                            for h in screen.template_hits
                        )
                    if extra_boxes:
                        # Merge with current YOLO boxes in overlay
                        with _yolo_latest_lock:
                            merged = list(_yolo_latest_boxes) + extra_boxes
                        _overlay.update(merged)

            # 3a. Single-step approval gate — pause before each click/back/swipe,
            # expose the pending action, block until POST /api/v1/step/go.
            if _STEP_MODE and not dry_run and action_type in ("click", "back", "swipe", "swipe_tap"):
                _cur = pipe.current_skill
                _aname = getattr(_cur, "name", "") if _cur else ""
                _asub = getattr(_cur, "sub_state", "") if _cur else ""
                if _cur is not None and hasattr(_cur, "_plan") and hasattr(_cur, "_cur_idx"):
                    try:
                        _ss = _cur._plan[_cur._cur_idx][0]
                        _aname = getattr(_ss, "name", _aname)
                        _asub = getattr(_ss, "sub_state", "")
                    except Exception:
                        pass
                globals()["_STEP_PENDING"] = {
                    "action": action_type, "reason": reason,
                    "skill": _aname, "sub_state": _asub,
                    "target": action.get("target") or action.get("from"),
                    "tick": pipe._total_ticks,
                }
                _PIPELINE_STATUS["step_pending"] = _STEP_PENDING
                _log_pipeline(f"STEP PAUSE [{_aname}/{_asub}] {action_type} @ {action.get('target') or action.get('from')} — {reason}")
                _STEP_GO.clear()
                if not _STEP_GO.wait(timeout=900):
                    continue  # approval timeout: stay paused, re-pend next loop
                if not _PIPELINE_RUNNING:
                    break  # stopped while paused → exit without executing
                globals()["_STEP_PENDING"] = None
                _PIPELINE_STATUS["step_pending"] = None

            # 3. Execute action (unless dry_run)
            if not dry_run and action_type != "done":
                _execute_pipeline_action(action, render_hwnd, frame.shape[1], frame.shape[0], adb, android_w, android_h)

            # 4. Sleep + OCR cache management
            # ── ZERO-WAIT policy (user 2026-06-14: 不要有wait time — 出现目标就点 /
            # 没目标pass / 只有"加载中"cls在时才硬wait, wait时长=加载中存在时长). All
            # the per-skill settle/re-scan/render-wait counters are squashed to a
            # fast re-poll; only a 加载中/loading wait keeps its full duration (the
            # loading gate re-checks each cycle until 加载中 is gone). Re-clicking a
            # not-yet-transitioned screen is already prevented by _dedup_click
            # (same-target hold), so we no longer pay the blanket 1.6s settle.
            _wait_reason = str(action.get("reason", "") or "")
            _is_loading_wait = ("加载中" in _wait_reason or "加載中" in _wait_reason
                                or "loading" in _wait_reason.lower())
            if action_type == "wait":
                if _is_loading_wait:
                    wait_ms = action.get("duration_ms", 500)
                    _high_res_sleep(max(1.0 / _DISPLAY_SYNC_HZ, wait_ms / 1000.0))
                else:
                    # non-loading wait → fast re-poll (≈one capture+infer cycle)
                    _high_res_sleep(max(1.0 / _DISPLAY_SYNC_HZ, 0.12))
            elif action_type in ("click", "back", "swipe_tap"):
                # invalidate OCR cache so the next tick OCRs the fresh screen.
                # 0.4→0.15s (user 2026-07-11 二次: "不要给等待时间, 极限测试,
                # 模拟人眼有就点没有就不点")。过点防护三层兜: 加载中 gate +
                # _dedup_click same-target hold + skill 侧 cls 证据制 wait。
                # (历史: 1.6→0.4→0614回调1.0→0711压0.4→0711极限0.15)
                _prev_ocr_boxes = None
                _high_res_sleep(0.15)
            else:
                _high_res_sleep(max(1.0 / _DISPLAY_SYNC_HZ, step_sleep))

    except Exception as e:
        _PIPELINE_STATUS["error"] = f"{type(e).__name__}: {e}"
        _log_pipeline(f"Pipeline worker error: {e}")
        traceback.print_exc()
    finally:
        # Stop high-FPS YOLO thread
        _yolo_thread_running = False
        if _yolo_hfps is not None and _yolo_hfps.is_alive():
            _yolo_hfps.join(timeout=3)
        summary = ""
        progress = None
        pipe = None
        with _PIPELINE_LOCK:
            pipe = _PIPELINE
        if pipe is not None:
            try:
                summary = pipe.get_summary()
            except Exception:
                summary = ""
            try:
                progress = pipe.progress
            except Exception:
                progress = None
        if _overlay:
            try:
                _overlay.stop()
            except Exception:
                pass
        try:
            status_name = "error" if _PIPELINE_STATUS.get("error") else "stopped"
            total_skills = int((progress or {}).get("total_skills") or 0)
            done_skills = len(list((progress or {}).get("results") or []))
            if status_name != "error" and total_skills > 0 and done_skills >= total_skills:
                status_name = "completed"
            _notify_pipeline_finished(status_name, summary, progress, dict(_PIPELINE_RUN_META))
        except Exception as notify_err:
            _log_pipeline(f"Pipeline notify skipped: {notify_err}")
        _PIPELINE_RUNNING = False
        _PIPELINE_STATUS["running"] = False
        _log_pipeline("Pipeline worker stopped.")


_dpi_set = False
_parent_hwnd: Optional[int] = None  # top-level window (for SetForegroundWindow)


def _set_parent_hwnd(hwnd: int) -> None:
    """Store the top-level parent hwnd for SetForegroundWindow."""
    global _parent_hwnd
    _parent_hwnd = hwnd


def _MAKELPARAM(x: int, y: int) -> int:
    """Pack two 16-bit values into a 32-bit LPARAM."""
    return (int(y) << 16) | (int(x) & 0xFFFF)


def _execute_pipeline_action(action: Dict[str, Any], hwnd: int, img_w: int, img_h: int, adb: Any = None, android_w: int = 0, android_h: int = 0) -> None:
    """Convert normalized pipeline action to real input.

    Uses PostMessage to the render child window for reliable emulator input.
    Falls back to SetCursorPos + mouse_event if PostMessage fails.
    """
    global _dpi_set
    import random
    from scripts.win_capture import get_client_rect_on_screen, _dpi_aware_context
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    _enable_high_resolution_timer()

    # Set DPI awareness once so coordinates are in physical pixels
    if not _dpi_set:
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
        except Exception:
            pass
        _dpi_set = True

    action_type = action.get("action", "")

    if adb is not None and android_w > 0 and android_h > 0:
        if action_type == "click":
            nx, ny = action.get("target", [0.5, 0.5])
            adb.tap(int(nx * android_w), int(ny * android_h))
            return
        if action_type == "back":
            adb.back()
            return
        if action_type == "swipe":
            frm = action.get("from", [0.5, 0.5])
            to = action.get("to", [0.5, 0.5])
            dur_ms = int(action.get("duration_ms", 400) or 400)
            adb.swipe(
                int(frm[0] * android_w), int(frm[1] * android_h),
                int(to[0] * android_w), int(to[1] * android_h),
                dur_ms,
            )
            return
        if action_type == "swipe_tap":
            # 原子 swipe→tap: 轮播类 UI 的唯一无竞争落点方式(swipe 拉停轮播,
            # tap 在静止期内落; 分两个 action 发则间隔 >1s 耗尽暂停期)
            frm = action.get("from", [0.5, 0.5])
            to = action.get("to", [0.5, 0.5])
            tgt = action.get("target", [0.5, 0.5])
            dur_ms = int(action.get("duration_ms", 150) or 150)
            adb.swipe_tap(
                int(frm[0] * android_w), int(frm[1] * android_h),
                int(to[0] * android_w), int(to[1] * android_h),
                dur_ms,
                int(tgt[0] * android_w), int(tgt[1] * android_h),
            )
            return
        if action_type == "scroll":
            nx, ny = action.get("target", [0.5, 0.5])
            clicks = int(action.get("clicks", -3) or -3)
            delta = max(80, min(int(android_h * 0.28), 40 * max(1, abs(clicks))))
            x = int(nx * android_w)
            y = int(ny * android_h)
            y1 = max(0, min(android_h - 1, y + delta if clicks < 0 else y - delta))
            y2 = max(0, min(android_h - 1, y - delta if clicks < 0 else y + delta))
            adb.swipe(x, y1, x, y2, max(250, min(900, 120 + abs(clicks) * 40)))
            return

    # Bring top-level parent window to foreground
    fg_hwnd = _parent_hwnd if _parent_hwnd else hwnd
    try:
        user32.SetForegroundWindow(fg_hwnd)
    except Exception:
        pass
    _high_res_sleep(1.0 / _DISPLAY_SYNC_HZ)

    # Get client rect for coordinate conversion
    try:
        rect = get_client_rect_on_screen(hwnd)
        cw = rect.right - rect.left
        ch = rect.bottom - rect.top
    except Exception:
        rect = None
        cw = ch = 0

    # DPI safety: if _dpi_aware_context failed silently, cw/ch may be in
    # logical pixels while frame (screenshot) is physical. Detect and fix.
    _dpi_scale = 1.0
    if cw > 0 and img_w > 0 and abs(img_w - cw) > 20:
        _dpi_scale = img_w / cw
        if not getattr(_execute_pipeline_action, '_dpi_warned', False):
            print(f"[DPI] Scale mismatch detected: frame={img_w}x{img_h} "
                  f"client={cw}x{ch} scale={_dpi_scale:.2f}")
            _execute_pipeline_action._dpi_warned = True

    WM_LBUTTONDOWN = 0x0201
    WM_LBUTTONUP = 0x0202
    WM_MOUSEMOVE = 0x0200
    MK_LBUTTON = 0x0001
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004
    MOUSEEVENTF_WHEEL = 0x0800

    def _screen_xy(nx: float, ny: float):
        if not rect or cw <= 0 or ch <= 0:
            return None
        # Use frame (physical) dimensions for pixel calculation when DPI
        # mismatch is detected; otherwise use client rect directly.
        pw = img_w if _dpi_scale > 1.01 else cw
        ph = img_h if _dpi_scale > 1.01 else ch
        cx = int(nx * pw) + random.randint(-2, 2)
        cy = int(ny * ph) + random.randint(-2, 2)
        # Adjust screen origin: if DPI scale active, rect coords are logical
        # and SetCursorPos (inside _dpi_aware_context) expects physical.
        ox = int(rect.left * _dpi_scale) if _dpi_scale > 1.01 else rect.left
        oy = int(rect.top * _dpi_scale) if _dpi_scale > 1.01 else rect.top
        sx = ox + cx
        sy = oy + cy
        return cx, cy, sx, sy

    if action_type == "click":
        nx, ny = action.get("target", [0.5, 0.5])
        coords = _screen_xy(nx, ny)
        if coords:
            cx, cy, sx, sy = coords
            print(f"[Click] norm=({nx:.3f},{ny:.3f}) client=({cx},{cy}) screen=({sx},{sy}) cw={cw} ch={ch}")
            with _dpi_aware_context():
                user32.SetCursorPos(sx, sy)
                _high_res_sleep(1.0 / _INPUT_POLL_HZ)
                user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                _high_res_sleep(max(1.0 / _INPUT_POLL_HZ, 1.0 / (_DISPLAY_SYNC_HZ * 2.0)))
                user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

    elif action_type == "back":
        # Send Escape key via PostMessage
        WM_KEYDOWN = 0x0100
        WM_KEYUP = 0x0101
        VK_ESCAPE = 0x1B
        user32.PostMessageW(hwnd, WM_KEYDOWN, VK_ESCAPE, 0x00010001)
        _high_res_sleep(max(1.0 / _INPUT_POLL_HZ, 1.0 / _DISPLAY_SYNC_HZ))
        user32.PostMessageW(hwnd, WM_KEYUP, VK_ESCAPE, 0xC0010001)

    elif action_type == "swipe":
        frm = action.get("from", [0.5, 0.5])
        to = action.get("to", [0.5, 0.5])
        dur_ms = action.get("duration_ms", 400)
        coords1 = _screen_xy(frm[0], frm[1])
        coords2 = _screen_xy(to[0], to[1])
        if coords1 and coords2:
            cx1, cy1, sx1, sy1 = coords1
            cx2, cy2, sx2, sy2 = coords2
            steps = max(12, int((dur_ms / 1000.0) * _DISPLAY_SYNC_HZ))
            print(f"[Swipe] from=({frm[0]:.3f},{frm[1]:.3f}) to=({to[0]:.3f},{to[1]:.3f}) steps={steps}")

            with _dpi_aware_context():
                user32.SetCursorPos(sx1, sy1)
                _high_res_sleep(1.0 / _INPUT_POLL_HZ)
                user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                for i in range(1, steps + 1):
                    t = i / steps
                    mx = int(sx1 + (sx2 - sx1) * t)
                    my = int(sy1 + (sy2 - sy1) * t)
                    user32.SetCursorPos(mx, my)
                    _high_res_sleep(dur_ms / 1000.0 / steps)
                user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

    elif action_type == "scroll":
        nx, ny = action.get("target", [0.5, 0.5])
        clicks = action.get("clicks", -3)
        with_ctrl = bool(action.get("with_ctrl", False))
        coords = _screen_xy(nx, ny)
        if coords:
            _, _, sx, sy = coords
            # Ctrl key constants for emulator pinch-zoom (MuMu/LDPlayer)
            VK_CONTROL = 0x11
            KEYEVENTF_KEYUP = 0x0002
            with _dpi_aware_context():
                user32.SetCursorPos(sx, sy)
                _high_res_sleep(1.0 / _DISPLAY_SYNC_HZ)
                if with_ctrl:
                    user32.keybd_event(VK_CONTROL, 0, 0, 0)
                    _high_res_sleep(0.02)
                user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, int(clicks * 120), 0)
                if with_ctrl:
                    _high_res_sleep(0.02)
                    user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)


def _log_pipeline(msg: str) -> None:
    """Append pipeline log message."""
    try:
        log_path = LOGS_DIR / "agent.out.log"
        with log_path.open("a", encoding="utf-8") as f:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass


# ── Utility functions ─────────────────────────────────────────────────

def _tail_text(path: Path, lines: int) -> str:
    if not path.exists():
        return ""
    try:
        data = path.read_bytes()
    except OSError:
        return ""

    data = data.replace(b"\x00", b"")
    parts = data.splitlines()[-max(1, lines) :]
    try:
        return b"\n".join(parts).decode("utf-8", errors="replace")
    except Exception:
        return ""


def _write_state(name: str, state: Dict[str, Any]) -> None:
    try:
        (LOGS_DIR / name).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


# ── Core routes ───────────────────────────────────────────────────────

@app.post("/api/v1/shutdown")
def shutdown_server() -> Dict[str, str]:
    def _do_exit():
        time.sleep(0.2)
        try:
            _stop_pipeline()
        except Exception:
            pass
        os._exit(0)
    threading.Thread(target=_do_exit, daemon=True).start()
    return {"status": "shutting_down"}


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return "<html><head><meta http-equiv='refresh' content='0; url=/dashboard.html'></head><body></body></html>"


@app.get("/dashboard.html")
def dashboard() -> FileResponse:
    if not DASHBOARD_PATH.exists():
        raise HTTPException(status_code=404, detail="dashboard.html not found")
    return FileResponse(
        str(DASHBOARD_PATH),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/api/v1/status")
def status() -> Dict[str, Any]:
    pipeline_running = _PIPELINE_RUNNING
    last_error = _PIPELINE_STATUS.get("error", "")
    if not pipeline_running and not last_error and _LAST_PIPELINE_ERROR:
        last_error = _LAST_PIPELINE_ERROR

    progress = None
    pipe = None
    with _PIPELINE_LOCK:
        pipe = _PIPELINE
    if pipe is not None:
        try:
            progress = pipe.progress
        except Exception:
            pass

    return {
        "server_pid": os.getpid(),
        "server_started_at": round(_SERVER_STARTED_AT, 3),
        "agent_running": pipeline_running,
        "agent_type": "pipeline",
        "pipeline_status": _PIPELINE_STATUS,
        "pipeline_progress": progress,
        "pipeline_run_meta": _PIPELINE_RUN_META,
        "last_error": last_error,
    }


# ── Capture path helpers ──────────────────────────────────────────────

def _safe_capture_path(rel: str) -> Path:
    rel = (rel or "").replace("\\", "/").lstrip("/")
    if not rel:
        raise HTTPException(status_code=400, detail="path is required")

    p = (CAPTURES_DIR / rel).resolve()
    base = CAPTURES_DIR.resolve()
    if not str(p).startswith(str(base)):
        raise HTTPException(status_code=400, detail="invalid path")
    return p


def _ocr_cache_path(rel: str) -> Path:
    key = f"v{OCR_CACHE_VERSION}:{rel}"
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    cache_dir = CAPTURES_DIR / "_cache" / "ocr"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{h}.json"


# ── Config / Favorites ────────────────────────────────────────────────

@app.get("/api/v1/config")
def get_app_config() -> Dict[str, Any]:
    active_profile, profile, cfg = _get_active_profile_settings()
    return {
        "active_profile": active_profile,
        "profile": profile,
        "profiles": cfg.get("profiles") or {},
        "skill_options": list(_SKILL_OPTIONS),
        # The validated production order — frontend "restore default"
        # button uses this instead of dumping every option (including
        # optional extras).
        "default_skill_order": list(_DEFAULT_SKILL_ORDER),
    }


@app.post("/api/v1/config")
def set_app_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _load_app_config()
    profiles = dict(cfg.get("profiles") or {})
    if isinstance(payload.get("profiles"), dict):
        profiles = { _normalize_profile_name(name): _normalize_profile_settings(data) for name, data in dict(payload.get("profiles") or {}).items() }
    target_profile = _normalize_profile_name(payload.get("profile_name") or payload.get("active_profile") or cfg.get("active_profile") or "default")
    active_profile = _normalize_profile_name(payload.get("active_profile") or target_profile)
    if target_profile not in profiles:
        profiles[target_profile] = _default_profile_settings()
    if isinstance(payload.get("profile"), dict):
        merged = dict(profiles.get(target_profile) or {})
        merged.update(dict(payload.get("profile") or {}))
        profiles[target_profile] = _normalize_profile_settings(merged)
    delete_profile = _normalize_profile_name(payload.get("delete_profile") or "") if payload.get("delete_profile") else ""
    if delete_profile and delete_profile in profiles and len(profiles) > 1:
        del profiles[delete_profile]
        if active_profile == delete_profile:
            active_profile = next(iter(profiles.keys()), "default")
    cfg["profiles"] = profiles
    cfg["active_profile"] = active_profile if active_profile in profiles else next(iter(profiles.keys()), "default")
    saved = _save_app_config(cfg)
    active_profile, profile, cfg = _get_active_profile_settings()
    return {
        "ok": True,
        "active_profile": active_profile,
        "profile": profile,
        "profiles": cfg.get("profiles") or {},
        "skill_options": list(_SKILL_OPTIONS),
        "saved": saved.get("active_profile") == active_profile,
    }

@app.get("/api/v1/config/favorites")
def get_favorites() -> Dict[str, Any]:
    active_profile, profile, _ = _get_active_profile_settings()
    return {"active_profile": active_profile, "target_favorites": profile.get("target_favorites", [])}


@app.post("/api/v1/config/favorites")
def set_favorites(payload: Dict[str, Any]) -> Dict[str, Any]:
    active_profile, profile, cfg = _get_active_profile_settings()
    profiles = dict(cfg.get("profiles") or {})
    merged = dict(profile)
    merged["target_favorites"] = payload.get("target_favorites", [])
    profiles[active_profile] = _normalize_profile_settings(merged)
    _save_app_config({"profiles": profiles, "active_profile": active_profile})
    return {"status": "ok", "active_profile": active_profile, "target_favorites": profiles[active_profile].get("target_favorites", [])}


# ── Characters ────────────────────────────────────────────────────────

@app.get("/api/v1/characters/avatars")
def list_character_avatars() -> Dict[str, Any]:
    if not CHARACTERS_DIR.exists():
        return {"avatars": []}
    avatars = sorted([p.name for p in CHARACTERS_DIR.glob("*.png")])
    return {"avatars": avatars}


# ── Captures ──────────────────────────────────────────────────────────

@app.get("/api/v1/captures/sessions")
def list_sessions() -> Dict[str, Any]:
    sessions: List[Dict[str, Any]] = []
    try:
        for d in sorted(CAPTURES_DIR.glob("session_*"), key=lambda x: x.name, reverse=True):
            if not d.is_dir():
                continue
            png_count = 0
            try:
                png_count = len(list(d.glob("*.png")))
            except Exception:
                png_count = 0
            sessions.append({"name": d.name, "png_count": png_count})
    except Exception:
        pass
    return {"sessions": sessions}


@app.get("/api/v1/captures/images")
def list_images(session: str = Query(...)) -> Dict[str, Any]:
    session = os.path.basename((session or "").strip())
    d = _safe_capture_path(session)
    if not d.exists() or not d.is_dir():
        raise HTTPException(status_code=404, detail="session not found")

    images = sorted([p.name for p in d.glob("*.png")])
    return {"images": images}


@app.get("/api/v1/captures/image")
def get_image(path: str = Query(...)) -> FileResponse:
    p = _safe_capture_path(path)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(str(p))


# ── Annotations ───────────────────────────────────────────────────────

def _labels_path_for_image(rel: str) -> Path:
    rel = rel.replace("\\", "/").lstrip("/")
    parts = rel.split("/")
    if parts and parts[0].startswith("session_"):
        return _safe_capture_path(parts[0]) / "labels.jsonl"
    return CAPTURES_DIR / "labels.jsonl"


@app.get("/api/v1/annotations")
def get_annotations(image: str = Query(...)) -> Dict[str, Any]:
    rel = (image or "").replace("\\", "/")
    p = _labels_path_for_image(rel)
    items: List[Dict[str, Any]] = []
    if p.exists():
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                if obj.get("image") == rel:
                    items.append(obj)
        except Exception:
            pass
    return {"image": rel, "items": items}


@app.post("/api/v1/annotations/add")
def add_annotation(payload: Dict[str, Any]) -> Dict[str, Any]:
    image = str(payload.get("image") or "").replace("\\", "/")
    label = str(payload.get("label") or "").strip()
    text = str(payload.get("text") or "").strip()
    bbox = payload.get("bbox")
    typ = str(payload.get("type") or "").strip() or "ocr"

    if not image or not label or bbox is None:
        raise HTTPException(status_code=400, detail="image/label/bbox required")

    _safe_capture_path(image)
    p = _labels_path_for_image(image)
    p.parent.mkdir(parents=True, exist_ok=True)

    rec = {
        "image": image,
        "label": label,
        "text": text,
        "bbox": bbox,
        "type": typ,
        "ts": time.time(),
    }
    try:
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to write label: {e}")

    return {"ok": True}


@app.get("/api/v1/annotations/index")
def annotations_index(session: str = Query(...)) -> Dict[str, Any]:
    session = os.path.basename((session or "").strip())
    if not session:
        raise HTTPException(status_code=400, detail="session is required")

    d = _safe_capture_path(session)
    if not d.exists() or not d.is_dir():
        raise HTTPException(status_code=404, detail="session not found")

    labels_path = d / "labels.jsonl"
    labeled: Set[str] = set()
    if labels_path.exists():
        try:
            for line in labels_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                rel = str(obj.get("image") or "").replace("\\", "/")
                if rel.startswith(session + "/"):
                    labeled.add(rel.split("/", 1)[1])
        except Exception:
            pass

    return {"session": session, "labeled": sorted(labeled)}


@app.get("/api/v1/captures/next_unlabeled")
def next_unlabeled(
    session: str = Query(...),
    after: str = Query(""),
) -> Dict[str, Any]:
    session = os.path.basename((session or "").strip())
    if not session:
        raise HTTPException(status_code=400, detail="session is required")

    d = _safe_capture_path(session)
    if not d.exists() or not d.is_dir():
        raise HTTPException(status_code=404, detail="session not found")

    after = os.path.basename((after or "").strip())
    images = sorted([p.name for p in d.glob("*.png")])

    labeled_res = annotations_index(session=session)
    labeled = set(labeled_res.get("labeled") or [])

    start_idx = 0
    if after and after in images:
        start_idx = images.index(after) + 1

    for name in images[start_idx:]:
        if name not in labeled:
            return {"session": session, "filename": name, "image": f"{session}/{name}"}

    for name in images[:start_idx]:
        if name not in labeled:
            return {"session": session, "filename": name, "image": f"{session}/{name}"}

    return {"session": session, "filename": None, "image": None}


# ── Schedule Roster Region Tuner ────────────────────────────────────────
# Lets the user visually tune the avatar-cell grid used by ScheduleSkill when
# scanning the 全體課程表 roster popup for favorite characters. Source config
# file: data/schedule_avatar_regions.json (loaded by ScheduleSkill).

SCHEDULE_REGIONS_PATH = REPO_ROOT / "data" / "schedule_avatar_regions.json"
TRAJECTORIES_DIR = REPO_ROOT / "data" / "trajectories"

_DEFAULT_STRIPS = [
    {"x1": cx1, "y1": ry1, "x2": cx2, "y2": ry2}
    for (ry1, ry2) in [(0.35, 0.47), (0.54, 0.66), (0.73, 0.84)]
    for (cx1, cx2) in [(0.09, 0.30), (0.34, 0.55), (0.58, 0.79)]
]
_DEFAULT_SCHEDULE_REGIONS = {
    "strips": _DEFAULT_STRIPS,
    "cells_per_room": 4,
}


def _normalize_strips(raw: Any) -> List[Dict[str, float]]:
    """Coerce a raw strips payload into a sanitized list of {x1,y1,x2,y2}."""
    out: List[Dict[str, float]] = []
    if not isinstance(raw, list):
        return out
    for s in raw:
        try:
            x1 = float(s["x1"]); y1 = float(s["y1"])
            x2 = float(s["x2"]); y2 = float(s["y2"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            continue
        out.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return out


def _load_schedule_regions() -> Dict[str, Any]:
    try:
        if SCHEDULE_REGIONS_PATH.exists():
            raw = json.loads(SCHEDULE_REGIONS_PATH.read_text(encoding="utf-8"))
            cpr = int(raw.get("cells_per_room") or _DEFAULT_SCHEDULE_REGIONS["cells_per_room"])
            # New format: flat strip list
            strips = _normalize_strips(raw.get("strips"))
            if strips:
                return {"strips": strips, "cells_per_room": max(1, cpr)}
            # Legacy: rows × cols cross-product
            rows = raw.get("rows_y") or []
            cols = raw.get("cols_x") or []
            if rows and cols:
                legacy = [
                    {"x1": float(cx1), "y1": float(ry1), "x2": float(cx2), "y2": float(ry2)}
                    for (ry1, ry2) in [r for r in rows if len(r) == 2]
                    for (cx1, cx2) in [c for c in cols if len(c) == 2]
                ]
                legacy = _normalize_strips(legacy)
                if legacy:
                    return {"strips": legacy, "cells_per_room": max(1, cpr)}
    except Exception:
        pass
    return json.loads(json.dumps(_DEFAULT_SCHEDULE_REGIONS))


@app.get("/api/v1/schedule/avatar_regions")
def get_schedule_avatar_regions() -> Dict[str, Any]:
    """Return current schedule avatar region config (defaults if no file)."""
    return {
        "config": _load_schedule_regions(),
        "defaults": _DEFAULT_SCHEDULE_REGIONS,
        "path": str(SCHEDULE_REGIONS_PATH),
        "exists": SCHEDULE_REGIONS_PATH.exists(),
    }


@app.post("/api/v1/schedule/avatar_regions")
def save_schedule_avatar_regions(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Save schedule avatar region config to data/schedule_avatar_regions.json."""
    strips = _normalize_strips(payload.get("strips"))
    if not strips:
        raise HTTPException(status_code=400, detail="strips must be non-empty list of {x1,y1,x2,y2}")
    try:
        cpr_int = int(payload.get("cells_per_room"))
        if cpr_int < 1 or cpr_int > 10:
            raise ValueError()
    except Exception:
        raise HTTPException(status_code=400, detail="cells_per_room must be int in 1..10")

    cfg = {"strips": strips, "cells_per_room": cpr_int}
    SCHEDULE_REGIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCHEDULE_REGIONS_PATH.write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {"ok": True, "config": cfg, "path": str(SCHEDULE_REGIONS_PATH)}


@app.get("/api/v1/schedule/roster_samples")
def list_roster_samples(limit: int = Query(60)) -> Dict[str, Any]:
    """List recent trajectory ticks where the 全體課程表 overlay was open.

    Detection: any tick whose OCR contains the overlay header
    "全體課程表" AND at least 2 room labels (視聽室/教室/圖書館/射擊場/
    體育館/實驗室/載具庫).  This is broader than the previous filter
    (Schedule + check_roster + "scanning roster avatars") which missed
    overlay-open frames captured outside that exact sub-state.

    Filter pulls from the most recent 20 trajectory runs to keep the
    sample list manageable; widens or narrows via the `limit` query
    param.
    """
    if not TRAJECTORIES_DIR.exists():
        return {"samples": []}
    samples: List[Dict[str, Any]] = []
    _rooms = ("視聽室", "视听室", "教室", "圖書館", "图书馆",
              "射擊場", "射击场", "體育館", "体育馆",
              "實驗室", "实验室", "載具庫", "载具库")
    try:
        runs = sorted(
            [d for d in TRAJECTORIES_DIR.iterdir() if d.is_dir() and d.name.startswith("run_")],
            key=lambda p: p.name, reverse=True,
        )[:20]
        for run in runs:
            for js in sorted(run.glob("tick_*.json")):
                try:
                    d = json.loads(js.read_text(encoding="utf-8"))
                except Exception:
                    continue
                ocr_boxes = d.get("ocr_boxes") or []
                has_header = any(
                    "全體課程表" in (b.get("text") or "")
                    or "全体课程表" in (b.get("text") or "")
                    for b in ocr_boxes
                )
                if not has_header:
                    continue
                room_hits = sum(
                    1 for b in ocr_boxes
                    if any(rm in (b.get("text") or "") for rm in _rooms)
                )
                if room_hits < 2:
                    continue
                jpg = js.with_suffix(".jpg")
                if not jpg.exists():
                    continue
                samples.append({
                    "run": run.name,
                    "tick": d.get("tick"),
                    "jpg": f"{run.name}/{jpg.name}",
                })
                if len(samples) >= limit:
                    break
            if len(samples) >= limit:
                break
    except Exception as e:
        print(f"[server] roster_samples error: {e}")
    return {"samples": samples}


def _safe_trajectory_jpg(rel: str) -> Path:
    rel = (rel or "").replace("\\", "/").strip()
    if not rel or ".." in rel.split("/"):
        raise HTTPException(status_code=400, detail="invalid path")
    p = (TRAJECTORIES_DIR / rel).resolve()
    base = TRAJECTORIES_DIR.resolve()
    if not str(p).startswith(str(base)) or not p.exists() or p.suffix.lower() != ".jpg":
        raise HTTPException(status_code=404, detail="jpg not found")
    return p


@app.get("/api/v1/schedule/roster_image")
def roster_image(jpg: str = Query(...)) -> FileResponse:
    """Serve a raw trajectory jpg by `run/tick_NNNN.jpg`.

    Client canvas renders the cell overlay; server just streams the image.
    """
    src = _safe_trajectory_jpg(jpg)
    return FileResponse(str(src), media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=3600"})


# ── Dataset & Label Editor APIs ──────────────────────────────────────────

def _safe_dataset_path(name: str) -> Path:
    """Resolve dataset name to filesystem dir.

    Two name styles:
      - 'run_xxx' / arbitrary  → under data/raw_images/
      - 'traj/run_xxx'         → under data/trajectories/<run_xxx>/
    """
    name = (name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="dataset name required")
    if name.startswith("traj/"):
        sub = os.path.basename(name[len("traj/"):].strip())
        if not sub:
            raise HTTPException(status_code=400, detail="dataset name required")
        p = (TRAJECTORIES_DIR / sub).resolve()
        if not str(p).startswith(str(TRAJECTORIES_DIR.resolve())):
            raise HTTPException(status_code=400, detail="invalid trajectory dataset")
        return p
    sub = os.path.basename(name)
    p = (RAW_IMAGES_DIR / sub).resolve()
    if not str(p).startswith(str(RAW_IMAGES_DIR.resolve())):
        raise HTTPException(status_code=400, detail="invalid dataset name")
    return p


# ── Universal class registry helpers ──────────────────────────────────────

def _load_master_classes() -> List[str]:
    """Read the universal class registry shared across all datasets."""
    if not MASTER_CLASSES_FILE.exists():
        return []
    return [
        c.strip()
        for c in MASTER_CLASSES_FILE.read_text(encoding="utf-8").splitlines()
        if c.strip()
    ]


def _save_master_classes(names: List[str]) -> None:
    MASTER_CLASSES_FILE.parent.mkdir(parents=True, exist_ok=True)
    MASTER_CLASSES_FILE.write_text("\n".join(names) + "\n", encoding="utf-8")


def _master_append(name: str) -> int:
    """Append a class to master if not already present.  Returns its index."""
    master = _load_master_classes()
    if name in master:
        return master.index(name)
    master.append(name)
    _save_master_classes(master)
    return len(master) - 1


def _bootstrap_master_from_existing() -> None:
    """One-time: if master is empty, seed it from the union of every
    existing per-dataset classes.txt (preserving first-seen order).

    Idempotent: only runs when master file is absent or empty.
    """
    if _load_master_classes():
        return
    union: List[str] = []
    seen: set = set()
    if RAW_IMAGES_DIR.is_dir():
        for d in sorted(RAW_IMAGES_DIR.iterdir()):
            if not d.is_dir():
                continue
            sub = d / "frames" if (d / "frames").is_dir() else d
            cf = sub / "classes.txt"
            if not cf.exists():
                continue
            for c in cf.read_text(encoding="utf-8").splitlines():
                c = c.strip()
                if c and c not in seen:
                    seen.add(c)
                    union.append(c)
    if union:
        _save_master_classes(union)


def _ensure_dataset_migrated(img_dir: Path) -> None:
    """Lazy-migrate one dataset to master indices.

    Side effects:
      - any local class not in master gets appended to master
      - label .txt files get remapped from local indices → master indices
      - dataset's classes.txt is overwritten with a copy of master
        (YOLO training keeps reading classes.txt as before)

    Idempotent: no-op if dataset's classes.txt already matches master.
    """
    _bootstrap_master_from_existing()
    master = _load_master_classes()
    cf = img_dir / "classes.txt"
    local: List[str] = []
    if cf.exists():
        local = [c.strip() for c in cf.read_text(encoding="utf-8").splitlines() if c.strip()]

    # Fast path: already on master
    if local == master and cf.exists():
        return

    # Append any local-only classes to master
    new_master = list(master)
    changed_master = False
    for c in local:
        if c not in new_master:
            new_master.append(c)
            changed_master = True
    if changed_master:
        _save_master_classes(new_master)
        master = new_master

    # Build remap: local_idx -> master_idx
    remap: Dict[int, int] = {}
    for li, c in enumerate(local):
        if c in master:
            remap[li] = master.index(c)

    # Rewrite label files only if local indices differ from master
    needs_label_remap = any(remap.get(i, i) != i for i in range(len(local)))
    if needs_label_remap:
        for label_file in img_dir.glob("*.txt"):
            if label_file.name == "classes.txt":
                continue
            try:
                content = label_file.read_text(encoding="utf-8")
            except Exception:
                continue
            if not content.strip():
                continue
            new_lines = []
            file_changed = False
            for line in content.splitlines():
                parts = line.strip().split()
                if not parts:
                    new_lines.append(line)
                    continue
                try:
                    old_cls = int(parts[0])
                except (ValueError, IndexError):
                    new_lines.append(line)
                    continue
                new_cls = remap.get(old_cls, old_cls)
                if new_cls != old_cls:
                    file_changed = True
                    parts[0] = str(new_cls)
                new_lines.append(" ".join(parts))
            if file_changed:
                label_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    # Sync local classes.txt to be a copy of master
    img_dir.mkdir(parents=True, exist_ok=True)
    cf.write_text("\n".join(master) + "\n", encoding="utf-8")


def _read_registry() -> Dict[str, Any]:
    """轻量读 data/model_registry.json (不依赖 pipeline import)。"""
    try:
        p = REPO_ROOT / "data" / "model_registry.json"
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _current_generation() -> Dict[str, Any]:
    """当前各域 live 模型代数 + 迭代代号 — dashboard label 区顶部显示。
    各域最强策略: UI 域 unified v6b(待 live 通电)否则 ui v5; 头像 fused v4;
    摸头 emoticon v26n。unified.active 含 PENDING = 未通电, 当前 live 仍 ui.active。"""
    reg = _read_registry()
    ui = reg.get("ui", {})
    uni = reg.get("unified", {})
    fav = reg.get("fused_avatar", {})
    emo = reg.get("emoticon", {})
    uni_active = str(uni.get("active", ""))
    uni_vers = list(uni.get("versions", {}).keys())
    uni_latest = uni_vers[-1] if uni_vers else None  # 最新版本(v6c), 不是第一个(旧 bug 取 v6b)
    uni_pending = ("PENDING" in uni_active) or (uni_active not in uni.get("versions", {}))
    if uni_pending:
        # 分域: pipeline live = 三独立模型 (UI=ui.active=v5). unified 暂不全域上线,
        # 最新版仅借作 prefill 标注 teacher (见 ui.versions.v6c)。
        ui_live = ui.get("active", "?")
        unified_note = f"暂不上(分域); {uni_latest}→标注teacher,等v7" if uni_latest else "—"
    else:
        ui_live = uni_active
        unified_note = f"{uni_active}·已全域上线"
    return {
        "label": uni_latest or ui.get("active", "?"),  # 当前迭代代号 (v6c)
        "ui_live": ui_live,                          # UI 域当前 live (分域=v5 / 全域上线=v6x)
        "unified": unified_note,                     # unified 上线状态
        "avatar": fav.get("active", "?"),            # 头像 fused v4
        "emoticon": emo.get("active", "?"),          # 摸头 v26n
    }


@app.get("/api/v1/datasets")
def list_datasets() -> Dict[str, Any]:
    datasets = []
    # ── raw_images recordings + val pools ──
    # Naming convention:
    #   run_<timestamp>/   → training data (kind="raw")
    #   _val_<purpose>/    → dedicated validation pool (kind="val")
    #                        Build scripts route these 100% to dataset val/.
    # Authoritative grouping from build_ui_v2's lists (don't guess by name):
    # VAL_SOURCES → val, REAL/SYNTH → train(raw), 其余 run_* → unused(未纳入).
    try:
        from scripts.build_ui_v2 import REAL_SOURCES, SYNTH_SOURCES, VAL_SOURCES  # noqa: PLC0415
        _train_set = set(REAL_SOURCES) | set(SYNTH_SOURCES)
        _val_set = set(VAL_SOURCES)
    except Exception:
        _train_set, _val_set = set(), set()
    # battle 模型 sources (build_battle_v3 直接引用原池 = 已并入永久 train 集,
    # 不再是"飞轮待标注" — 2026-07-10 用户: 飞轮区只留没用过/即将要用的)
    try:
        from scripts.build_battle_v3 import SRCS as _battle_srcs  # noqa: PLC0415
        _train_set |= {p.name for p in _battle_srcs}
    except Exception:
        pass
    for d in sorted(RAW_IMAGES_DIR.iterdir()):
        if not d.is_dir():
            continue
        jpg_count = len(list(d.glob("*.jpg")))
        frames_dir = d / "frames"
        if frames_dir.is_dir():
            jpg_count += len(list(frames_dir.glob("*.jpg")))
        if jpg_count == 0:
            continue
        if d.name in _val_set or d.name.startswith("_val_"):
            kind = "val"
        elif d.name in _train_set:
            kind = "raw"
        elif d.name.startswith("run_"):
            # capture/start 录的干净帧 (run_<timestamp>), 未纳入 build_ui_v2
            # sources = 飞轮待标注素材 (标完加进 sources 训下代)。
            kind = "flywheel"
        else:
            kind = "unused"  # 合成/废弃目录 (_synth_* 等), 标了也不进 build
        datasets.append({
            "name": d.name,
            "image_count": jpg_count,
            "kind": kind,
        })
    # ── trajectory 退役 (2026-06-05) ──────────────────────────────────────
    # pipeline 实战 tick 帧 (data/trajectories/) 有 overlay 烧录风险: 跑 pipeline
    # 时 YoloOverlay 透明置顶, DXcam 抓的是合成画面 → 检测框烧进像素 = 训练垃圾
    # (2026-05-28 删过 11 个烧录 run)。故不再进 label 队列。飞轮采集改用
    # capture/start 录的干净 raw_images/run_* (采集时无 pipeline overlay)。
    # TRAJECTORIES_DIR 仍保留 — schedule roster tuner (roster_samples/roster_image)
    # 仍按需读它, 但那是 schedule 专用区, 不混入通用 label dataset 列表。
    return {"datasets": datasets, "generation": _current_generation()}


@app.get("/api/v1/datasets/images")
def list_dataset_images(dataset: str = Query(...)) -> Dict[str, Any]:
    d = _safe_dataset_path(dataset)
    if not d.exists():
        raise HTTPException(status_code=404, detail="dataset not found")
    # Images may be in root or /frames subfolder.  Trajectory ticks live
    # directly in the run dir; raw_images may use a 'frames' subfolder.
    img_dir = d / "frames" if (d / "frames").is_dir() else d
    # Accept both raw_images naming (frame_*.jpg) and trajectory naming
    # (tick_*.jpg / arbitrary).
    images = sorted([p.name for p in img_dir.glob("*.jpg")])
    # Lazy-migrate this dataset to master class indices.  Idempotent —
    # no-op if already on master.  For brand-new trajectory dirs this
    # also seeds classes.txt with the master copy on first access.
    _ensure_dataset_migrated(img_dir)
    classes = _load_master_classes()
    # Load labels per image
    items = []
    for img_name in images:
        label_path = img_dir / Path(img_name).with_suffix(".txt")
        labels = []
        if label_path.exists():
            for line in label_path.read_text().strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) >= 5:
                    lb = {
                        "cls": int(parts[0]),
                        "xc": float(parts[1]), "yc": float(parts[2]),
                        "w": float(parts[3]), "h": float(parts[4]),
                    }
                    if len(parts) >= 6:
                        try:
                            lb["angle"] = float(parts[5])
                        except ValueError:
                            lb["angle"] = 0
                    if len(parts) >= 7:
                        lb["shape"] = parts[6]
                    if lb.get("shape") == "polygon" and len(parts) >= 8:
                        try:
                            lb["points"] = [
                                {"x": float(c.split(",")[0]), "y": float(c.split(",")[1])}
                                for c in parts[7].split(";") if "," in c
                            ]
                        except (ValueError, IndexError):
                            lb["points"] = []
                    labels.append(lb)
        items.append({"img": img_name, "labels": labels})
    return {"dataset": dataset, "classes": classes, "images": items}


@app.get("/api/v1/datasets/image")
def get_dataset_image(dataset: str = Query(...), filename: str = Query(...)) -> FileResponse:
    d = _safe_dataset_path(dataset)
    img_dir = d / "frames" if (d / "frames").is_dir() else d
    filename = os.path.basename(filename)
    p = img_dir / filename
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="image not found")
    return FileResponse(str(p), media_type="image/jpeg")


@app.post("/api/v1/datasets/save_labels")
def save_dataset_labels(payload: Dict[str, Any]) -> Dict[str, Any]:
    dataset = str(payload.get("dataset") or "")
    img_name = str(payload.get("img") or "")
    label_text = str(payload.get("labels") or "")
    d = _safe_dataset_path(dataset)
    img_dir = d / "frames" if (d / "frames").is_dir() else d
    label_path = img_dir / Path(os.path.basename(img_name)).with_suffix(".txt")
    label_path.write_text(label_text, encoding="utf-8")
    return {"ok": True}


@app.post("/api/v1/datasets/add_class")
def add_dataset_class(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Append a class to the MASTER registry.

    The new class is immediately available in every dataset, not just
    the one this request originated from.  We also sync the originating
    dataset's local classes.txt so the YOLO training pipeline keeps
    seeing the full registry on its next read.
    """
    new_name = str(payload.get("name") or "").strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="class name required")
    idx = _master_append(new_name)
    # Sync the originating dataset's classes.txt (and optionally others).
    dataset = str(payload.get("dataset") or "")
    if dataset:
        try:
            d = _safe_dataset_path(dataset)
            img_dir = d / "frames" if (d / "frames").is_dir() else d
            img_dir.mkdir(parents=True, exist_ok=True)
            master = _load_master_classes()
            (img_dir / "classes.txt").write_text(
                "\n".join(master) + "\n", encoding="utf-8"
            )
        except HTTPException:
            pass  # dataset name invalid → still ok, master got the class
    return {"ok": True, "id": idx}


@app.post("/api/v1/datasets/delete_image")
def delete_dataset_image(payload: Dict[str, Any]) -> Dict[str, Any]:
    dataset = str(payload.get("dataset") or "")
    img_name = str(payload.get("img") or "")
    d = _safe_dataset_path(dataset)
    img_dir = d / "frames" if (d / "frames").is_dir() else d
    fname = os.path.basename(img_name)
    img_path = img_dir / fname
    label_path = img_dir / Path(fname).with_suffix(".txt")
    deleted = []
    if img_path.exists():
        img_path.unlink()
        deleted.append(fname)
    if label_path.exists():
        label_path.unlink()
        deleted.append(label_path.name)
    return {"ok": True, "deleted": deleted}


@app.post("/api/v1/datasets/florence_suggest")
def dataset_florence_suggest(payload: Dict[str, Any]) -> Dict[str, Any]:
    dataset = str(payload.get("dataset") or "")
    img_name = str(payload.get("img") or "")
    labels = [str(x).strip() for x in (payload.get("labels") or []) if str(x).strip()]
    limit = max(1, min(int(payload.get("limit") or 24), 64))
    d = _safe_dataset_path(dataset)
    img_dir = d / "frames" if (d / "frames").is_dir() else d
    img_path = img_dir / Path(os.path.basename(img_name))
    if not img_path.exists() or not img_path.is_file():
        raise HTTPException(status_code=404, detail="image not found")
    if not labels:
        classes_file = img_dir / "classes.txt"
        if classes_file.exists():
            labels = [c.strip() for c in classes_file.read_text(encoding="utf-8").splitlines() if c.strip()]
    labels = labels[:24]
    if not labels:
        raise HTTPException(status_code=400, detail="no classes available for Florence suggestions")

    def _iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = float((ix2 - ix1) * (iy2 - iy1))
        area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
        area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
        denom = area_a + area_b - inter
        if denom <= 0:
            return 0.0
        return inter / denom

    try:
        import cv2

        from vision.io_utils import imread_any  # noqa: PLC0415
        img = imread_any(str(img_path))
        if img is None:
            raise HTTPException(status_code=400, detail="cannot read image")
        h, w = img.shape[:2]
        from vision.florence_vision import get_florence_vision

        raw = get_florence_vision().suggest_labels(str(img_path), labels)
        kept: List[Dict[str, Any]] = []
        for item in raw:
            bbox = item.get("bbox") or []
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            box = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            if any(item.get("label") == old.get("label") and _iou(box, old.get("bbox") or [0, 0, 0, 0]) > 0.6 for old in kept):
                continue
            kept.append({
                "label": str(item.get("label") or ""),
                "score": float(item.get("score") or 1.0),
                "bbox": box,
            })
            if len(kept) >= limit:
                break

        suggestions = []
        for item in kept:
            x1, y1, x2, y2 = item["bbox"]
            suggestions.append({
                "label": item["label"],
                "score": item["score"],
                "x1": x1 / w,
                "y1": y1 / h,
                "x2": x2 / w,
                "y2": y2 / h,
                "xc": ((x1 + x2) / 2.0) / w,
                "yc": ((y1 + y2) / 2.0) / h,
                "w": (x2 - x1) / w,
                "h": (y2 - y1) / h,
                "bbox": [x1, y1, x2, y2],
            })
        return {"ok": True, "suggestions": suggestions, "count": len(suggestions), "labels_used": labels}
    except HTTPException:
        raise
    except Exception as e:
        return {"ok": False, "error": str(e), "suggestions": [], "count": 0, "labels_used": labels}


# ── UI YOLO model-assisted suggestions ─────────────────────────────────
# Unlike florence_suggest (open-vocabulary, fuzzy), this runs the ACTIVE ui
# detector (model_registry ui.active) so suggestions come back already tagged
# with the exact trained cls_name. Used by the annotation UI to PRE-FILL boxes
# on trajectory frames (which ARE training material): the model auto-labels the
# reliable cls, the human then corrects mis-labels (select box → change class
# → e.g. "X" → "邮件箱") and adds the boxes the model misses. conf is kept LOW
# (default 0.15) so even weak detections (邮件箱-grade) surface for review.
# 模型加载已并入 scripts.yolo_prefill_run.get_model(model_key) — single-frame
# suggest 与整run prefill 共用一套,支持 ui/fused_avatar/emoticon 多 teacher 选择
# + name→master remap + 权威类段过滤。


def _iou_box(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    bb = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / (aa + bb - inter + 1e-9)


def _parse_target_classes(raw):
    """target_classes payload (master cls name 或 idx 的 list) → set of master
    idx。None/空 = 不过滤 (标该模型全 span)。用于 label "只标目标 cls" — 飞轮补
    单个/几个弱类时只预填它们, 不用标全部。"""
    if not isinstance(raw, list) or not raw:
        return None
    master = _load_master_classes()
    name2idx = {n: i for i, n in enumerate(master)}
    out = set()
    for t in raw:
        ts = str(t or "").strip()
        if not ts:
            continue
        if ts.isdigit():
            out.add(int(ts))
        elif ts in name2idx:
            out.add(name2idx[ts])
    return out or None


@app.post("/api/v1/datasets/yolo_suggest")
def dataset_yolo_suggest(payload: Dict[str, Any]) -> Dict[str, Any]:
    dataset = str(payload.get("dataset") or "")
    img_name = str(payload.get("img") or "")
    conf = float(payload.get("conf") or 0.15)
    model_key = str(payload.get("model") or "ui")
    version = str(payload.get("version") or "").strip() or None  # None=registry active
    d = _safe_dataset_path(dataset)
    img_dir = d / "frames" if (d / "frames").is_dir() else d
    img_path = img_dir / Path(os.path.basename(img_name))
    if not img_path.exists() or not img_path.is_file():
        raise HTTPException(status_code=404, detail="image not found")
    try:
        from vision.io_utils import imread_any  # noqa: PLC0415
        img = imread_any(str(img_path))
        if img is None:
            raise HTTPException(status_code=400, detail="cannot read image")
        h, w = img.shape[:2]
        # Pick the requested teacher (ui|fused_avatar|emoticon|battle_heads),
        # remap its LOCAL class ids → master BY NAME, and keep only boxes inside
        # that model's authoritative span — so an avatar pass won't suggest a
        # spurious UI class on a sprite (and vice-versa). Shared with the 整run
        # prefill so single-frame and batch behave identically.
        from scripts.yolo_prefill_run import get_model, _OWNS, _IMGSZ_BY_TAG  # noqa: PLC0415
        model, remap, tag = get_model(model_key, version)
        owns = _OWNS.get(tag, lambda i: True)
        tgt = _parse_target_classes(payload.get("target_classes"))  # 只标目标 cls
        use_imgsz = int(payload.get("imgsz") or _IMGSZ_BY_TAG.get(tag, 960))
        master = _load_master_classes()
        r = model.predict(img, imgsz=use_imgsz, conf=conf, device=0, verbose=False)[0]
        raw = []
        for b in r.boxes:
            mi = remap.get(int(b.cls[0]))
            if mi is None or not owns(mi):
                continue
            if tgt is not None and mi not in tgt:
                continue
            sc = float(b.conf[0])
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            if x2 <= x1 or y2 <= y1:
                continue
            raw.append((sc, mi, [x1, y1, x2, y2]))
        raw.sort(key=lambda t: -t[0])  # high conf first
        # Dedup within this single model's output (any-cls IoU>0.6, keep higher
        # conf) — one clean box per element. Cross-pass accumulation (a head box
        # sitting next to a UI box) is handled client-side by same-class dedup.
        kept = []
        for sc, mi, box in raw:
            if any(_iou_box(box, k[2]) > 0.6 for k in kept):
                continue
            kept.append((sc, mi, box))
        suggestions = []
        for sc, mi, (x1, y1, x2, y2) in kept:
            nm = master[mi] if 0 <= mi < len(master) else str(mi)
            suggestions.append({
                "label": nm, "cls": mi, "score": round(sc, 3),
                "x1": x1 / w, "y1": y1 / h, "x2": x2 / w, "y2": y2 / h,
                "xc": ((x1 + x2) / 2.0) / w, "yc": ((y1 + y2) / 2.0) / h,
                "w": (x2 - x1) / w, "h": (y2 - y1) / h,
            })
        return {"ok": True, "suggestions": suggestions, "count": len(suggestions)}
    except HTTPException:
        raise
    except Exception as e:
        return {"ok": False, "error": str(e), "suggestions": [], "count": 0}


@app.post("/api/v1/datasets/yolo_prefill_run")
def dataset_yolo_prefill_run(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Batch-prefill every frame in a dataset/run with ONE detector
    (model: ui|fused_avatar|emoticon|battle_heads). mode: 'merge' accumulates
    this model's boxes onto whatever other passes already wrote (shared-frame
    cross-teacher labeling for the unified model — never erases other classes),
    'overwrite' keeps only this model's boxes, 'skip' leaves already-labeled
    frames. Synchronous — ~600 frames ≈ 1 min on GPU. Reuses
    scripts/yolo_prefill_run.prefill_run (name→master remap + span filter +
    dedup). Default mode='skip' preserves the legacy single-pass button."""
    dataset = str(payload.get("dataset") or "")
    conf = float(payload.get("conf") or 0.25)
    model_key = str(payload.get("model") or "ui")
    version = str(payload.get("version") or "").strip() or None  # None=registry active
    mode = str(payload.get("mode") or "skip")
    if payload.get("overwrite"):
        mode = "overwrite"
    target_classes = _parse_target_classes(payload.get("target_classes"))
    d = _safe_dataset_path(dataset)
    img_dir = d / "frames" if (d / "frames").is_dir() else d
    if not img_dir.is_dir():
        raise HTTPException(status_code=404, detail="dataset not found")
    try:
        from scripts.yolo_prefill_run import prefill_run  # noqa: PLC0415
        res = prefill_run(img_dir, model_key=model_key, version=version, conf=conf,
                          mode=mode, target_classes=target_classes)
        return {"ok": True, **res}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/api/v1/datasets/yolo_models")
def dataset_yolo_models() -> Dict[str, Any]:
    """列各 prefill teacher 模型 + 可选版本 (dashboard 模型/版本下拉数据源)。
    每个 key 给 active(pipeline live 版) + teacher(prefill 推荐版, ≠ active) +
    versions[](active 置顶, 其余按 registry 序) + 中文标签。
    各域最强标注策略: UI 默认 teacher=v6c(借 unified, live UI 入口 conf 远胜 v5),
    但 cafe 内部弱类(咖啡厅收益/邀请卷) v6c 退步 → 手动切 version=v5 + 只标目标 cls;
    头像=v4; 摸头=v26n。"""
    from scripts.yolo_prefill_run import _OWNS, _KEY_TO_TAG  # noqa: PLC0415
    reg = _read_registry()
    master = _load_master_classes()
    label = {"ui": "ui", "fused_avatar": "头像", "emoticon": "摸头",
             "battle_heads": "战斗头"}
    teacher_default = {}  # (2026-06-08) ui v7 上线=active 即最强 UI(by-cls 0.921 全面超 v6c) → teacher=active(v7); fused v6 / emoticon v26n 同理 active=最强。旧 v6c teacher 弃用
    out = []
    for key in ("ui", "fused_avatar", "emoticon", "battle_heads"):
        node = reg.get(key)
        if not isinstance(node, dict):
            continue
        active = str(node.get("active", ""))
        vers = list((node.get("versions") or {}).keys())
        ordered = ([active] if active in vers else []) + [v for v in vers if v != active]
        td = teacher_default.get(key, active)
        if td not in vers:
            td = active if active in vers else (vers[0] if vers else "")
        # 该模型 owns-span 内的 master 类名 → 喂前端"只标cls" datalist: 选 ui 只列
        # UI 域 cls(选头像只列头像), 用户输入即自动补全 + 实时校验该 teacher 能否标。
        tag = _KEY_TO_TAG.get(key, "ui")
        owns = _OWNS.get(tag, lambda i: True)
        span_classes = [nm for i, nm in enumerate(master) if owns(i)]
        out.append({"key": key, "label": label.get(key, key),
                    "active": active, "teacher": td, "versions": ordered,
                    "span_classes": span_classes})
    return {"models": out}


# ── OCR API ───────────────────────────────────────────────────────────

_OCR_ENGINE = None
_OCR_LOCK = threading.Lock()

def _get_ocr():
    global _OCR_ENGINE
    with _OCR_LOCK:
        if _OCR_ENGINE is None:
            from rapidocr_onnxruntime import RapidOCR
            custom_rec = Path(__file__).resolve().parent.parent / "data" / "ocr_model" / "ba_rec.onnx"
            if custom_rec.exists():
                _OCR_ENGINE = RapidOCR(rec_model_path=str(custom_rec))
            else:
                _OCR_ENGINE = RapidOCR()
        return _OCR_ENGINE

@app.get("/api/v1/datasets/ocr")
def dataset_ocr(dataset: str = Query(...), filename: str = Query(...)):
    """Run RapidOCR on a single image, return text boxes with pixel + normalized coords."""
    img_path = (RAW_IMAGES_DIR / dataset / filename).resolve()
    if not img_path.exists():
        raise HTTPException(404, "Image not found")
    try:
        import cv2
        from vision.io_utils import imread_any  # noqa: PLC0415
        img = imread_any(str(img_path))
        if img is None:
            raise HTTPException(400, "Cannot read image")
        ocr = _get_ocr()
        result, _ = ocr(img)
        boxes = []
        h, w = img.shape[:2]
        if result:
            for line in result:
                pts, text, conf = line
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                px1, py1 = int(min(xs)), int(min(ys))
                px2, py2 = int(max(xs)), int(max(ys))
                boxes.append({
                    "text": text,
                    "confidence": round(float(conf), 3),
                    "x1": px1 / w, "y1": py1 / h,
                    "x2": px2 / w, "y2": py2 / h,
                    "px1": px1, "py1": py1, "px2": px2, "py2": py2,
                })
        return {"ok": True, "boxes": boxes, "count": len(boxes),
                "image_w": w, "image_h": h}
    except HTTPException:
        raise
    except Exception as e:
        return {"ok": False, "error": str(e), "boxes": [], "count": 0}


@app.post("/api/v1/datasets/open_folder")
def dataset_open_folder(payload: Dict[str, Any]):
    """Open the dataset folder in the system file explorer."""
    dataset = payload.get("dataset", "")
    ds_dir = RAW_IMAGES_DIR / dataset
    if not ds_dir.exists():
        raise HTTPException(404, "Dataset not found")
    subprocess.Popen(["explorer", str(ds_dir.resolve())])
    return {"ok": True, "path": str(ds_dir.resolve())}


@app.post("/api/v1/datasets/ocr/save")
def dataset_ocr_save(payload: Dict[str, Any]):
    """Save OCR results for an image to the dataset's ocr_results.json."""
    dataset = payload.get("dataset", "")
    filename = payload.get("filename", "")
    boxes = payload.get("boxes", [])
    image_w = payload.get("image_w", 0)
    image_h = payload.get("image_h", 0)
    ds_dir = RAW_IMAGES_DIR / dataset
    if not ds_dir.exists():
        raise HTTPException(404, "Dataset not found")
    ocr_path = ds_dir / "ocr_results.json"
    data = {}
    if ocr_path.exists():
        try:
            data = json.loads(ocr_path.read_text("utf-8"))
        except Exception:
            data = {}
    data[filename] = {
        "image_w": image_w, "image_h": image_h,
        "boxes": boxes,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    ocr_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
    total_images = len(data)
    total_texts = sum(len(v.get("boxes", [])) for v in data.values())
    return {"ok": True, "total_images": total_images, "total_texts": total_texts}


@app.post("/api/v1/datasets/ocr/batch")
def dataset_ocr_batch(payload: Dict[str, Any]):
    """Batch scan all images in a dataset with OCR, save results."""
    dataset = payload.get("dataset", "")
    ds_dir = RAW_IMAGES_DIR / dataset
    if not ds_dir.exists():
        raise HTTPException(404, "Dataset not found")
    import cv2
    ocr = _get_ocr()
    ocr_path = ds_dir / "ocr_results.json"
    data = {}
    if ocr_path.exists():
        try:
            data = json.loads(ocr_path.read_text("utf-8"))
        except Exception:
            data = {}
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    imgs = sorted(f for f in ds_dir.iterdir() if f.suffix.lower() in exts)
    scanned = 0
    for img_path in imgs:
        fname = img_path.name
        if fname in data:
            continue  # skip already scanned
        from vision.io_utils import imread_any  # noqa: PLC0415
        img = imread_any(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        result, _ = ocr(img)
        boxes = []
        if result:
            for line in result:
                pts, text, conf = line
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                px1, py1 = int(min(xs)), int(min(ys))
                px2, py2 = int(max(xs)), int(max(ys))
                boxes.append({
                    "text": text, "confidence": round(float(conf), 3),
                    "x1": px1 / w, "y1": py1 / h,
                    "x2": px2 / w, "y2": py2 / h,
                    "px1": px1, "py1": py1, "px2": px2, "py2": py2,
                })
        data[fname] = {
            "image_w": w, "image_h": h,
            "boxes": boxes,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        scanned += 1
    ocr_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
    total_images = len(data)
    total_texts = sum(len(v.get("boxes", [])) for v in data.values())
    return {"ok": True, "scanned": scanned, "total_images": total_images,
            "total_texts": total_texts}


# ── Screen Capture APIs (DXcam) ──────────────────────────────────────

def _capture_worker(dataset_name: str, interval: float, window_title: str):
    """Capture game screen via DXcam (Desktop Duplication API).

    DXcam captures the monitor output directly, which works for GPU-rendered
    MuMu windows as long as they are visible on screen.
    """
    global _CAPTURE_RUNNING, _CAPTURE_STATUS
    try:
        import cv2
        import numpy as np
        import dxcam
        import ctypes
        import ctypes.wintypes as wt
        from scripts.win_capture import (
            find_window_by_title_substring,
            find_largest_visible_child,
        )

        hwnd = find_window_by_title_substring(window_title)
        if not hwnd:
            _CAPTURE_STATUS["error"] = f"Window '{window_title}' not found"
            _CAPTURE_STATUS["running"] = False
            _CAPTURE_RUNNING = False
            return

        child = find_largest_visible_child(hwnd)
        target_hwnd = child if child else hwnd

        # Get screen-space rect for DXcam region
        rc = wt.RECT()
        ctypes.windll.user32.GetWindowRect(target_hwnd, ctypes.byref(rc))
        region = (rc.left, rc.top, rc.right, rc.bottom)

        camera = dxcam.create(output_idx=0, output_color="BGR")
        test_frame = camera.grab(region=region)
        if test_frame is None:
            _CAPTURE_STATUS["error"] = "DXcam grab returned None"
            _CAPTURE_STATUS["running"] = False
            _CAPTURE_RUNNING = False
            return

        # dataset_name may include a path separator for val pools
        # (e.g., "_val_fused/frames"); construct the dir relative to RAW_IMAGES.
        out_dir = RAW_IMAGES_DIR / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)
        _CAPTURE_STATUS["dataset"] = dataset_name
        _CAPTURE_STATUS["error"] = ""

        # Find next available index when appending to existing val pool dir
        existing = list(out_dir.glob("frame_*.jpg")) + list(out_dir.glob("val_*.jpg"))
        start_idx = len(existing)

        count = 0
        while _CAPTURE_RUNNING:
            t0 = time.time()
            try:
                # Re-read window rect in case it moved
                ctypes.windll.user32.GetWindowRect(target_hwnd, ctypes.byref(rc))
                region = (rc.left, rc.top, rc.right, rc.bottom)
                frame = camera.grab(region=region)
                if frame is not None:
                    fp = out_dir / f"frame_{start_idx + count:06d}.jpg"
                    cv2.imwrite(str(fp), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    count += 1
                    _CAPTURE_STATUS["frames"] = count
            except Exception as cap_err:
                _CAPTURE_STATUS["error"] = f"capture error: {cap_err}"
            elapsed = time.time() - t0
            time.sleep(max(0, interval - elapsed))

    except Exception as e:
        _CAPTURE_STATUS["error"] = str(e)
    finally:
        _CAPTURE_STATUS["running"] = False
        _CAPTURE_RUNNING = False


@app.post("/api/v1/capture/start")
def capture_start(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Start screen capture.

    payload:
      window_title: str  (default "MuMu")
      interval:     float (default 0.5)
      split:        "train" | "val"  (default "train")
                    "val" routes frames to data/raw_images/_val_<purpose>/frames/
                    which build scripts route 100% to validation, never train.
      purpose:      "fused" | "static_ui" | str  (default "fused")
                    Tags which model this val set evaluates.  Ignored if
                    split=train.
    """
    global _CAPTURE_THREAD, _CAPTURE_RUNNING, _CAPTURE_STATUS
    with _CAPTURE_LOCK:
        if _CAPTURE_RUNNING:
            return {"ok": False, "error": "already running", "status": _CAPTURE_STATUS}
        interval = float(payload.get("interval", 0.5))
        window_title = str(payload.get("window_title", "MuMu"))
        split = str(payload.get("split", "train")).strip().lower()
        purpose = str(payload.get("purpose", "fused")).strip().lower() or "fused"
        from datetime import datetime
        if split == "val":
            # Append to existing val pool; no timestamp suffix
            ds_name = f"_val_{purpose}/frames"
        else:
            ds_name = "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        _CAPTURE_RUNNING = True
        _CAPTURE_STATUS = {"running": True, "frames": 0, "dataset": ds_name,
                           "split": split, "error": ""}
        _CAPTURE_THREAD = threading.Thread(
            target=_capture_worker,
            args=(ds_name, interval, window_title),
            daemon=True,
        )
        _CAPTURE_THREAD.start()
    return {"ok": True, "status": _CAPTURE_STATUS}


@app.post("/api/v1/capture/stop")
def capture_stop() -> Dict[str, Any]:
    global _CAPTURE_RUNNING
    _CAPTURE_RUNNING = False
    return {"ok": True, "status": _CAPTURE_STATUS}


@app.get("/api/v1/capture/status")
def capture_status_api() -> Dict[str, Any]:
    return {"status": _CAPTURE_STATUS}


# ── Single-step approval ───────────────────────────────────────────────
@app.get("/api/v1/step/pending")
def step_pending_api() -> Dict[str, Any]:
    """Current pending action awaiting approval (None if not paused)."""
    return {"step_mode": _STEP_MODE, "pending": _STEP_PENDING}


@app.post("/api/v1/step/go")
def step_go_api() -> Dict[str, Any]:
    """Approve the pending action → pipeline executes it and advances."""
    approved = _STEP_PENDING
    _STEP_GO.set()
    return {"ok": True, "approved": approved}


# ── Synth Template Editor ────────────────────────────────────────────
# Lets the user define synthetic data composition templates per UI context.
# Each context (schedule_popup, momotalk, student_list, battle_squad,
# tactical_competition, cafe_invite, ...) gets its own JSON template with:
#   - Slot rectangles where refs will be pasted (normalized 0-1 coords)
#   - Ref crop transformation (which fraction of ref to use, shape mask)
#   - Augmentation knobs (UI overlay probs, border ablation, brightness)
# build_fused_avatar_dataset.py reads these templates instead of detecting
# rooms via static_ui — gives the user pixel-perfect control.

SYNTH_TEMPLATES_DIR = REPO_ROOT / "data" / "synth_templates"
SYNTH_SAMPLES_DIR = SYNTH_TEMPLATES_DIR / "samples"


def _synth_default_template(ctx: str) -> Dict[str, Any]:
    """Return a sensible default template for a new context."""
    return {
        "context": ctx,
        "sample_image": "",
        "image_size": [1920, 1080],
        "slot_rects_norm": [],  # list of {"x1","y1","x2","y2"} (normalized)
        "ref_transform": {
            "crop_n": {"x1": 0.0, "y1": 0.0, "x2": 1.0, "y2": 1.0},
            "shape": "square",       # square | rounded_rect | circle
            "radius_px": 4,
            "scale": 1.0,            # 1.0 = fill slot fully
        },
        "augmentation": {
            "ui_overlay_prob": 0.50,
            "ui_components": {
                "lv_text":      0.50,
                "star":         0.30,
                "weapon_icon":  0.40,
                "heart":        0.20,
                "alpha_dim":    0.25,
            },
            "border_ablation_prob": 0.40,
            "brightness_jitter": [0.92, 1.08],
        },
        "synth_count": 200,
        # ── UI-model synth (2026-05-29) ──
        # target: "avatar" (default — slots labelled with student names for the
        # avatar detector) OR "ui" (slots provide BACKGROUND diversity only;
        # the ui_stamps below are the labelled boxes, with ui-cls indices).
        "target": "avatar",
        # ui_stamps: fixed UI elements labelled in EVERY synth frame (normalized
        # rects + master cls index). Used when target=="ui" to teach rare UI cls
        # (e.g. 学生momotalk信息未读=439, 前往羁绊剧情=441, 进入羁绊剧情=442) on
        # the diverse backgrounds the avatar-swap produces. Place these in the
        # dashboard synth editor's "UI 图章" tool.
        "ui_stamps": [],
    }


# Preset contexts that show up by default
SYNTH_CONTEXTS_DEFAULT = [
    "schedule_popup",
    "student_list",
    "momotalk",
    "battle_squad",
    "tactical_competition",
    "cafe_invite",
]


@app.get("/api/v1/synth/templates")
def synth_templates_list() -> Dict[str, Any]:
    """List all template contexts (presets + any user-created files)."""
    SYNTH_TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    found = {}
    for f in SYNTH_TEMPLATES_DIR.glob("*.json"):
        ctx = f.stem
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            found[ctx] = {
                "context": ctx,
                "configured": True,
                "n_slots": len(data.get("slot_rects_norm", [])),
                "sample_image": data.get("sample_image", ""),
            }
        except Exception:
            continue
    # Include unconfigured presets
    out = []
    for ctx in SYNTH_CONTEXTS_DEFAULT:
        if ctx in found:
            out.append(found[ctx])
        else:
            out.append({"context": ctx, "configured": False, "n_slots": 0, "sample_image": ""})
    # Add any extras (user-created beyond presets)
    for ctx, info in found.items():
        if ctx not in SYNTH_CONTEXTS_DEFAULT:
            out.append(info)
    return {"templates": out}


@app.get("/api/v1/synth/template/{ctx}")
def synth_template_get(ctx: str) -> Dict[str, Any]:
    """Load a template — returns defaults if file doesn't exist yet."""
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", ctx)[:64]
    f = SYNTH_TEMPLATES_DIR / f"{safe}.json"
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            pass
    return _synth_default_template(safe)


@app.post("/api/v1/synth/template/{ctx}")
def synth_template_save(ctx: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Save a template JSON to data/synth_templates/<ctx>.json."""
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", ctx)[:64]
    SYNTH_TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    f = SYNTH_TEMPLATES_DIR / f"{safe}.json"
    # Stamp save time
    payload = dict(payload or {})
    payload["context"] = safe
    payload["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    f.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "path": str(f)}


@app.post("/api/v1/synth/build_ui/{ctx}")
def synth_build_ui(ctx: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generate UI-model synth frames for a context: avatar-swap into slots for
    BACKGROUND diversity + label the template's ui_stamps with their ui-cls in
    every frame → data/raw_images/_synth_<ctx>/ (add to build_ui_dataset
    TRAIN_SOURCES). Reuses scripts/build_ui_synth.synth_context."""
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", ctx)[:64]
    count = int((payload or {}).get("count") or 300)
    try:
        from scripts.build_ui_synth import synth_context  # noqa: PLC0415
        synth_context(safe, count=count, seed=0)
        out = REPO_ROOT / "data" / "raw_images" / f"_synth_{safe}"
        n = len(list(out.glob("frame_*.jpg"))) if out.is_dir() else 0
        return {"ok": True, "out": str(out), "frames": n}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/api/v1/synth/ui_classes")
def synth_ui_classes() -> Dict[str, Any]:
    """Master UI class list (index → name) for the synth 'UI 图章' cls picker."""
    return {"classes": _load_master_classes()}


@app.delete("/api/v1/synth/template/{ctx}")
def synth_template_delete(ctx: str) -> Dict[str, Any]:
    """Delete a template JSON and its sample image."""
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", ctx)[:64]
    tpl = SYNTH_TEMPLATES_DIR / f"{safe}.json"
    removed = []
    if tpl.exists():
        tpl.unlink()
        removed.append(str(tpl.name))
    # Also remove the sample image if present
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        s = SYNTH_SAMPLES_DIR / f"{safe}{ext}"
        if s.exists():
            s.unlink()
            removed.append(f"samples/{s.name}")
    return {"ok": True, "removed": removed}


@app.post("/api/v1/synth/upload_sample/{ctx}")
def synth_upload_sample(ctx: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Copy/decode an image into samples/ for use as template background.

    Accepts EITHER:
      * `source_path`: existing file on disk (relative to repo or absolute)
      * `file_b64`: base64-encoded file bytes (with optional `data:image/...;base64,` prefix)
                    + `filename` to determine the extension
    """
    import shutil as _shutil
    import base64 as _b64
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", ctx)[:64]
    payload = payload or {}
    src = (payload.get("source_path") or "").strip()
    file_b64 = payload.get("file_b64") or ""
    filename = payload.get("filename") or ""
    SYNTH_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    if file_b64:
        # Strip optional data URI prefix
        if "," in file_b64 and file_b64.startswith("data:"):
            file_b64 = file_b64.split(",", 1)[1]
        try:
            raw = _b64.b64decode(file_b64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"invalid file_b64: {e}")
        if not raw:
            raise HTTPException(status_code=400, detail="empty file_b64")
        # Allow common image extensions; default to .jpg
        ext = Path(filename).suffix.lower() if filename else ".jpg"
        if ext not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            ext = ".jpg"
        dst = SYNTH_SAMPLES_DIR / f"{safe}{ext}"
        dst.write_bytes(raw)
    elif src:
        p = Path(src)
        if not p.is_absolute():
            p = REPO_ROOT / src
        if not p.exists() or not p.is_file():
            raise HTTPException(status_code=404, detail=f"source not found: {p}")
        dst = SYNTH_SAMPLES_DIR / f"{safe}{p.suffix.lower()}"
        _shutil.copy2(p, dst)
    else:
        raise HTTPException(status_code=400, detail="source_path or file_b64 required")
    # Also record in the template
    template = synth_template_get(safe)
    template["sample_image"] = f"samples/{dst.name}"
    # Read actual image size
    try:
        import cv2 as _cv2
        import numpy as _np
        buf = _np.fromfile(str(dst), dtype=_np.uint8)
        if buf.size:
            im = _cv2.imdecode(buf, _cv2.IMREAD_COLOR)
            if im is not None:
                template["image_size"] = [im.shape[1], im.shape[0]]
    except Exception:
        pass
    synth_template_save(safe, template)
    return {"ok": True, "sample_image": template["sample_image"],
            "image_size": template["image_size"]}


@app.get("/api/v1/synth/ref_image/{cn_name}")
def synth_ref_image(cn_name: str):
    """Stream a character's wiki ref portrait (404×456 from 角色头像/, or 54×59
    from 角色头像_crop/ as fallback) for the dashboard's crop tool.
    """
    from fastapi.responses import FileResponse
    # Resolve CN → EN using all three name maps
    name_maps: Dict[str, str] = {}
    for fname in ("student_name_map.json", "student_name_map_extension.json"):
        p = REPO_ROOT / "data" / fname
        if p.exists():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                for k, v in d.items():
                    if isinstance(v, str):
                        name_maps[k] = v
            except Exception:
                pass
    en = name_maps.get(cn_name)
    if not en:
        raise HTTPException(status_code=404, detail=f"no name map for {cn_name}")
    big = REPO_ROOT / "data" / "captures" / "角色头像" / f"{en}.png"
    crop = REPO_ROOT / "data" / "captures" / "角色头像_crop" / f"{en}.png"
    p = big if big.exists() else (crop if crop.exists() else None)
    if p is None:
        raise HTTPException(status_code=404, detail=f"ref file not found for {en}")
    return FileResponse(str(p), media_type="image/png")


@app.get("/api/v1/synth/portrait/{cn}")
def synth_portrait(cn: str):
    """Small portrait for a fused_avatar cls — schedule-target picker thumbnails.
    Resolves the paren-less 简体 cls → EN via avatar_thumb_map.json (built by
    scripts/build_avatar_thumb_map.py), then serves the 54×59 crop (角色头像_crop/)
    or the large portrait as fallback."""
    from fastapi.responses import FileResponse
    en = None
    tm = REPO_ROOT / "data" / "avatar_thumb_map.json"
    if tm.exists():
        try:
            en = json.loads(tm.read_text(encoding="utf-8")).get(cn)
        except Exception:
            pass
    if not en:
        raise HTTPException(status_code=404, detail=f"no thumb map for {cn}")
    crop = REPO_ROOT / "data" / "captures" / "角色头像_crop" / f"{en}.png"
    big = REPO_ROOT / "data" / "captures" / "角色头像" / f"{en}.png"
    p = crop if crop.exists() else (big if big.exists() else None)
    if p is None:
        raise HTTPException(status_code=404, detail=f"no portrait for {en}")
    return FileResponse(str(p), media_type="image/png")


@app.get("/api/v1/synth/sample_image/{ctx}")
def synth_sample_image(ctx: str):
    """Stream the sample image bytes for a given context."""
    from fastapi.responses import FileResponse
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", ctx)[:64]
    template = synth_template_get(safe)
    rel = (template or {}).get("sample_image", "")
    if not rel:
        raise HTTPException(status_code=404, detail="no sample image set")
    p = SYNTH_TEMPLATES_DIR / rel
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"file missing: {p}")
    return FileResponse(str(p), media_type=f"image/{p.suffix.lstrip('.').lower()}")


# Map context names to filename patterns in val_fused/frames/ for auto-default
_SYNTH_CONTEXT_PATTERNS = {
    "schedule_popup":       ["val_schedule_"],
    "cafe_invite":          ["val_cafe_"],
    "battle_squad":         ["val_arena_fight_", "frame_"],  # 选人 / 编队
    "arena_squad":          ["val_arena_fight_"],
    "tactical_competition": ["val_arena_op_"],               # val_arena_op_* = 战术大战
    "momotalk":             ["frame_"],
    "student_list":         ["frame_"],
}


@app.get("/api/v1/synth/suggested_sample/{ctx}")
def synth_suggested_sample(ctx: str) -> Dict[str, Any]:
    """Suggest sample images for a context (from val_fused/frames/).
    Used by dashboard to auto-populate sample when user hasn't uploaded one yet.
    """
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", ctx)[:64]
    patterns = _SYNTH_CONTEXT_PATTERNS.get(safe, ["val_"])
    val_dir = REPO_ROOT / "data" / "raw_images" / "_val_fused" / "frames"
    if not val_dir.exists():
        return {"suggestions": []}
    suggestions = []
    for jpg in sorted(val_dir.glob("*.jpg")):
        for pat in patterns:
            if jpg.name.startswith(pat):
                suggestions.append(str(jpg.relative_to(REPO_ROOT)))
                break
    return {"suggestions": suggestions[:20], "default": suggestions[0] if suggestions else ""}


@app.get("/api/v1/synth/characters")
def synth_characters() -> Dict[str, Any]:
    """List all available character ref names (from 角色头像 + 角色头像_crop)
    + their resolved CN names from master[143:]. Used by Preview to let user
    pick which character to paste."""
    # CN char names from master
    master_file = REPO_ROOT / "data" / "raw_images" / "_classes.txt"
    if not master_file.exists():
        return {"characters": []}
    lines = [l.strip() for l in master_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    chars_cn = lines[143:] if len(lines) > 143 else []
    return {"characters": chars_cn}


@app.get("/api/v1/avatar_classes")
def avatar_classes() -> Dict[str, Any]:
    """fused_avatar model's ACTUAL cls names — what the detector outputs and
    schedule.py Case B matches on (the trained 252-class subset, NOT master's
    308). Read from data/fused_avatar_classes.json (dumped from the .pt
    model.names; re-dump if the avatar model is retrained). Falls back to
    master _classes.txt[143:] if the dump is missing."""
    f = REPO_ROOT / "data" / "fused_avatar_classes.json"
    if f.exists():
        try:
            names = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(names, list) and names:
                return {"characters": sorted(names)}
        except Exception:
            pass
    master = REPO_ROOT / "data" / "raw_images" / "_classes.txt"
    if master.exists():
        lines = [l.strip() for l in master.read_text(encoding="utf-8").splitlines() if l.strip()]
        return {"characters": sorted(lines[143:])}
    return {"characters": []}


def _synth_apply_ui_overlay(ref_img, ui_components, aug_positions=None):
    """Apply UI overlay aug with optional anchor positions (normalized within
    slot, default to BA-game-like corners).  `aug_positions` is dict like
    `{lv: {x,y}, star: {x,y}, weapon: {x,y}, heart: {x,y}}`.
    """
    import cv2 as _cv2
    import numpy as _np
    import random as _random
    h, w = ref_img.shape[:2]
    out = ref_img.copy()
    AP = aug_positions or {}
    # ±5% jitter so positions aren't pixel-identical across composites
    def _jit(v): return max(0.0, min(1.0, v + _random.uniform(-0.05, 0.05)))

    if _random.random() < float(ui_components.get("lv_text", 0.5)):
        pos = AP.get("lv") or {"x": 0.05, "y": 0.15}
        nx, ny = _jit(float(pos.get("x", 0.05))), _jit(float(pos.get("y", 0.15)))
        lv = _random.randint(1, 90)
        text = f"Lv.{lv}" if _random.random() < 0.75 else "MAX"
        font_scale = max(0.35, min(0.65, w / 100.0))
        # Anchor is text baseline center-ish — offset by char width
        tw = int(35 * font_scale); th = int(20 * font_scale)
        x = max(2, min(w - tw - 2, int(nx * w) - tw // 2))
        y = max(th, min(h - 2, int(ny * h) + th // 2))
        _cv2.putText(out, text, (x, y), _cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0), 2)
        _cv2.putText(out, text, (x, y), _cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), 1)
    if _random.random() < float(ui_components.get("star", 0.3)):
        pos = AP.get("star") or {"x": 0.10, "y": 0.10}
        nx, ny = _jit(float(pos.get("x", 0.10))), _jit(float(pos.get("y", 0.10)))
        r = max(4, w // 12)
        cx = max(r + 1, min(w - r - 1, int(nx * w)))
        cy = max(r + 1, min(h - r - 1, int(ny * h)))
        _cv2.circle(out, (cx, cy), r, (0, 220, 255), -1)
        _cv2.circle(out, (cx, cy), r, (0, 80, 120), 1)
    if _random.random() < float(ui_components.get("weapon_icon", 0.4)):
        pos = AP.get("weapon") or {"x": 0.85, "y": 0.85}
        nx, ny = _jit(float(pos.get("x", 0.85))), _jit(float(pos.get("y", 0.85)))
        size = max(10, w // 6)
        color = _random.choice([(0, 0, 255), (0, 165, 255), (255, 128, 0),
                                (255, 0, 255), (255, 255, 0)])
        cx = max(size // 2, min(w - size // 2, int(nx * w)))
        cy = max(size // 2, min(h - size // 2, int(ny * h)))
        _cv2.rectangle(out, (cx - size // 2, cy - size // 2),
                       (cx + size // 2, cy + size // 2), color, -1)
    if _random.random() < float(ui_components.get("heart", 0.2)):
        pos = AP.get("heart") or {"x": 0.85, "y": 0.85}
        nx, ny = _jit(float(pos.get("x", 0.85))), _jit(float(pos.get("y", 0.85)))
        size = max(6, w // 7)
        cx = max(size + 1, min(w - size - 1, int(nx * w)))
        cy = max(size + 1, min(h - size - 1, int(ny * h)))
        _cv2.circle(out, (cx, cy), size, (147, 20, 255), -1)
        _cv2.circle(out, (cx, cy), size, (255, 255, 255), 2)
        num = str(_random.randint(1, 99))
        font_scale = max(0.35, w / 130.0)
        _cv2.putText(out, num, (max(0, cx - size + 1), min(h - 2, cy + 4)),
                     _cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), 1)
    if _random.random() < float(ui_components.get("alpha_dim", 0.25)):
        alpha = _random.uniform(0.55, 0.85)
        out = _np.clip(out.astype(_np.float32) * alpha, 0, 255).astype(_np.uint8)
    return out


def _synth_apply_border_ablation(ref_img):
    """Mirror of build script's apply_border_ablation — random colored bar on
    one or more sides, forces classifier to use avatar pixels not UI frame."""
    import numpy as _np
    import random as _random
    h, w = ref_img.shape[:2]
    out = ref_img.copy()
    border = max(2, _random.randint(2, max(3, w // 18)))
    sides = _random.choice(["top", "bot", "left", "right", "all-thin", "all-thick"])
    color = (_random.randint(0, 255), _random.randint(0, 255), _random.randint(0, 255))
    b = max(3, border + 2) if sides == "all-thick" else border
    if sides in ("top", "all-thin", "all-thick"):    out[:b, :] = color
    if sides in ("bot", "all-thin", "all-thick"):    out[h - b:, :] = color
    if sides in ("left", "all-thin", "all-thick"):   out[:, :b] = color
    if sides in ("right", "all-thin", "all-thick"):  out[:, w - b:] = color
    return out


@app.post("/api/v1/synth/preview/{ctx}")
def synth_preview(ctx: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Render a single preview composite using the current template config.

    payload: {
      "template": {...} (full template JSON from dashboard, not yet saved),
      "char_cn": "若藻"  (optional — if given, paste this char in all slots;
                          if not given, pick random chars)
    }
    Returns: { "image_b64": "...", "labels": [{class, name, xyxy}...] }
    """
    import base64
    import cv2 as _cv2
    import numpy as _np
    import random as _random
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", ctx)[:64]
    tpl = (payload or {}).get("template")
    if not isinstance(tpl, dict):
        tpl = synth_template_get(safe)
    char_cn = (payload or {}).get("char_cn", "").strip()

    # Load sample bg
    rel = tpl.get("sample_image", "")
    if not rel:
        raise HTTPException(status_code=400, detail="no sample image set")
    bg_path = SYNTH_TEMPLATES_DIR / rel
    if not bg_path.exists():
        raise HTTPException(status_code=404, detail=f"bg missing: {bg_path}")
    buf = _np.fromfile(str(bg_path), dtype=_np.uint8)
    bg = _cv2.imdecode(buf, _cv2.IMREAD_COLOR)
    if bg is None:
        raise HTTPException(status_code=500, detail="bg decode failed")
    H, W = bg.shape[:2]
    composite = bg.copy()

    # Resolve char → ref
    big_dir = REPO_ROOT / "data" / "captures" / "角色头像"
    crop_dir = REPO_ROOT / "data" / "captures" / "角色头像_crop"
    name_maps = {}
    for fname in ("student_name_map.json", "student_name_map_extension.json"):
        p = REPO_ROOT / "data" / fname
        if p.exists():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                for k, v in d.items():
                    if isinstance(v, str):
                        name_maps[k] = v
            except Exception:
                pass

    def load_ref(cn_name: str, prefer_crop: bool = True):
        """Load a ref for the character.

        prefer_crop=True (default for slots): use the 54×59 head crop —
          matches what game shows in schedule popup / momotalk / etc.
        prefer_crop=False: use the 404×456 half-body portrait (e.g. student
          list cards where bigger art is visible).
        """
        en = name_maps.get(cn_name)
        if not en:
            return None
        order = (crop_dir, big_dir) if prefer_crop else (big_dir, crop_dir)
        for d in order:
            p = d / f"{en}.png"
            if p.exists():
                b = _np.fromfile(str(p), dtype=_np.uint8)
                im = _cv2.imdecode(b, _cv2.IMREAD_COLOR)
                if im is not None:
                    return im
        return None

    # Get list of all CN characters with refs
    master_p = REPO_ROOT / "data" / "raw_images" / "_classes.txt"
    chars_all = []
    if master_p.exists():
        lines = [l.strip() for l in master_p.read_text(encoding="utf-8").splitlines() if l.strip()]
        chars_all = [c for c in lines[143:] if c in name_maps]

    rt = tpl.get("ref_transform", {}) or {}
    crop_n = rt.get("crop_n", {})
    cx1 = float(crop_n.get("x1", 0.0))
    cy1 = float(crop_n.get("y1", 0.0))
    cx2 = float(crop_n.get("x2", 1.0))
    cy2 = float(crop_n.get("y2", 1.0))
    shape = rt.get("shape", "square")
    radius_px = int(rt.get("radius_px", 4) or 0)
    scale = float(rt.get("scale", 1.0))
    aug = tpl.get("augmentation", {}) or {}

    labels = []
    aug_stats = {"ui_overlay": 0, "border": 0, "total_slots": 0}
    for slot in tpl.get("slot_rects_norm", []):
        x1 = int(slot["x1"] * W); y1 = int(slot["y1"] * H)
        x2 = int(slot["x2"] * W); y2 = int(slot["y2"] * H)
        sw = x2 - x1; sh = y2 - y1
        if sw < 8 or sh < 8:
            continue
        # Pick char
        if char_cn:
            chosen = char_cn
        elif chars_all:
            chosen = _random.choice(chars_all)
        else:
            chosen = None
        if not chosen:
            continue
        # Slot-size-aware ref pick: big slot uses half-body, small uses head crop
        prefer_crop = max(sw, sh) <= 140
        ref = load_ref(chosen, prefer_crop=prefer_crop)
        if ref is None:
            continue
        # Crop ref
        rh, rw = ref.shape[:2]
        rx1 = int(cx1 * rw); ry1 = int(cy1 * rh)
        rx2 = int(cx2 * rw); ry2 = int(cy2 * rh)
        if rx2 - rx1 < 4 or ry2 - ry1 < 4:
            cropped = ref
        else:
            cropped = ref[ry1:ry2, rx1:rx2]
        # ── Read aug config (apply AFTER resize for visibility at slot pixel res) ──
        ui_overlay_prob = float(aug.get("ui_overlay_prob", 0.5))
        ui_comp = aug.get("ui_components") or {}
        border_prob = float(aug.get("border_ablation_prob", 0.4))
        aug_stats["total_slots"] += 1
        # ── Unified paste logic: ALWAYS preserve ref aspect ratio (no stretch).
        # Compute the slot's pixel AABB (for rect: x1..x2,y1..y2; for quad: poly AABB).
        # Fit ref inside AABB ("contain"), centered. For quad, apply polygon mask after
        # so corners outside the quad show background, not stretched ref pixels.
        quad = slot.get("quad")
        quad_px = None
        if quad and len(quad) == 4:
            try:
                quad_px = _np.array(
                    [[float(q["x"]) * W, float(q["y"]) * H] for q in quad],
                    dtype=_np.float32,
                )
                aabb_x1 = max(0, int(quad_px[:, 0].min()))
                aabb_y1 = max(0, int(quad_px[:, 1].min()))
                aabb_x2 = min(W, int(quad_px[:, 0].max()))
                aabb_y2 = min(H, int(quad_px[:, 1].max()))
            except Exception:
                quad_px = None
        if quad_px is None:
            aabb_x1, aabb_y1 = x1, y1
            aabb_x2, aabb_y2 = x2, y2
        aabb_w = aabb_x2 - aabb_x1; aabb_h = aabb_y2 - aabb_y1
        if aabb_w < 8 or aabb_h < 8:
            continue
        # Preserve ref aspect ratio (COVER: fill AABB, overflow clipped — no bg padding inside)
        rh0, rw0 = cropped.shape[:2]
        if rh0 <= 0 or rw0 <= 0:
            continue
        ref_ar = rw0 / rh0
        max_w = max(4, int(aabb_w * scale))
        max_h = max(4, int(aabb_h * scale))
        slot_ar = max_w / max(max_h, 1)
        # COVER: ref's smaller dim relative to slot fills, larger overflows
        if ref_ar > slot_ar:
            target_h = max_h; target_w = max(4, int(target_h * ref_ar))
        else:
            target_w = max_w; target_h = max(4, int(target_w / ref_ar))
        try:
            resized = _cv2.resize(cropped, (target_w, target_h), interpolation=_cv2.INTER_AREA)
        except Exception:
            continue
        # ── Apply aug AFTER resize — effects are now sized for slot pixels,
        # so Lv text / star / weapon / heart are visible at game resolution ──
        aug_pos = (tpl.get("ref_transform") or {}).get("aug_positions") or {}
        if _random.random() < ui_overlay_prob:
            resized = _synth_apply_ui_overlay(resized.copy(), ui_comp, aug_pos)
            aug_stats["ui_overlay"] += 1
        if _random.random() < border_prob:
            resized = _synth_apply_border_ablation(resized)
            aug_stats["border"] += 1
        # Shape mask (circle/rounded_rect) on the resized ref
        ref_mask = None
        if shape == "circle":
            ref_mask = _np.zeros((target_h, target_w), dtype=_np.uint8)
            _cv2.circle(ref_mask, (target_w//2, target_h//2), min(target_w, target_h)//2, 255, -1)
        elif shape == "rounded_rect" and radius_px > 0:
            ref_mask = _np.zeros((target_h, target_w), dtype=_np.uint8)
            r = min(radius_px, target_h//2, target_w//2)
            _cv2.rectangle(ref_mask, (r, 0), (target_w-r, target_h), 255, -1)
            _cv2.rectangle(ref_mask, (0, r), (target_w, target_h-r), 255, -1)
            _cv2.circle(ref_mask, (r, r), r, 255, -1)
            _cv2.circle(ref_mask, (target_w-r, r), r, 255, -1)
            _cv2.circle(ref_mask, (r, target_h-r), r, 255, -1)
            _cv2.circle(ref_mask, (target_w-r, target_h-r), r, 255, -1)
        # Center ref in AABB (may overflow AABB on one axis)
        ref_left = aabb_x1 + (aabb_w - target_w) // 2
        ref_top  = aabb_y1 + (aabb_h - target_h) // 2
        # Clip ref paste region to AABB ∩ image bounds — overflow outside AABB
        # is invisible (clipped here; further clipped by polygon mask if quad)
        clip_x1 = max(0, aabb_x1, ref_left)
        clip_y1 = max(0, aabb_y1, ref_top)
        clip_x2 = min(W, aabb_x2, ref_left + target_w)
        clip_y2 = min(H, aabb_y2, ref_top + target_h)
        if clip_x2 - clip_x1 < 4 or clip_y2 - clip_y1 < 4:
            continue
        # Source slice within resized
        sx1 = clip_x1 - ref_left; sy1 = clip_y1 - ref_top
        sx2 = sx1 + (clip_x2 - clip_x1); sy2 = sy1 + (clip_y2 - clip_y1)
        source_slice = resized[sy1:sy2, sx1:sx2]
        # Paste (apply ref_mask shape if any)
        if ref_mask is not None:
            mask_slice = ref_mask[sy1:sy2, sx1:sx2]
            bg_patch = composite[clip_y1:clip_y2, clip_x1:clip_x2].copy()
            for cc in range(3):
                bg_patch[..., cc] = _np.where(mask_slice > 0, source_slice[..., cc], bg_patch[..., cc])
            composite[clip_y1:clip_y2, clip_x1:clip_x2] = bg_patch
        else:
            composite[clip_y1:clip_y2, clip_x1:clip_x2] = source_slice
        # Quad: restore original bg outside polygon (within AABB)
        if quad_px is not None:
            poly_full = _np.zeros((H, W), dtype=_np.uint8)
            _cv2.fillPoly(poly_full, [quad_px.astype(_np.int32)], 255)
            bg_patch_aabb = bg[aabb_y1:aabb_y2, aabb_x1:aabb_x2]
            local_poly = poly_full[aabb_y1:aabb_y2, aabb_x1:aabb_x2]
            patch = composite[aabb_y1:aabb_y2, aabb_x1:aabb_x2]
            patch[local_poly == 0] = bg_patch_aabb[local_poly == 0]
            composite[aabb_y1:aabb_y2, aabb_x1:aabb_x2] = patch
        # YOLO label box = visible ref region (clip rect)
        px1, py1, px2, py2 = clip_x1, clip_y1, clip_x2, clip_y2
        # ── end paste, fall through to label & overlay drawing ─────────
        labels.append({"name": chosen, "xyxy": [px1, py1, px2, py2]})
        # ── Overlay: yellow = the slot's TRUE shape (quad polygon or rect).
        # This matches what the user configured in the editor.  We do NOT draw
        # the inner ref AABB rect anymore (was confusing — looked like the slot
        # was axis-aligned even when slot is a parallelogram).
        if quad_px is not None:
            # Thick yellow polygon outline = the slot quad
            _cv2.polylines(composite, [quad_px.astype(_np.int32)], True, (0, 255, 255), 3)
            # Tiny corner dots so 4 corners are obvious
            for pt in quad_px.astype(_np.int32):
                _cv2.circle(composite, tuple(pt.tolist()), 4, (0, 255, 255), -1)
            # Label at top-left of AABB
            _cv2.putText(composite, chosen, (aabb_x1+2, aabb_y1-4),
                         _cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        else:
            # Rect slot: yellow rectangle along the actual slot bounds
            _cv2.rectangle(composite, (aabb_x1, aabb_y1), (aabb_x2, aabb_y2), (0, 255, 255), 3)
            _cv2.putText(composite, chosen, (aabb_x1+2, aabb_y1-4),
                         _cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # Encode
    ok, enc = _cv2.imencode(".jpg", composite, [_cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ok:
        raise HTTPException(status_code=500, detail="encode failed")
    return {
        "image_b64": base64.b64encode(enc.tobytes()).decode("ascii"),
        "labels": labels,
        "image_size": [W, H],
        "aug_stats": aug_stats,
        "aug_config": {
            "ui_overlay_prob": float(aug.get("ui_overlay_prob", 0.0)),
            "border_ablation_prob": float(aug.get("border_ablation_prob", 0.0)),
            "ui_components": aug.get("ui_components") or {},
        },
    }


# ── Game launch ───────────────────────────────────────────────────────

def _maybe_launch_game(exe_path: str, wait_seconds: float = 5.0) -> str:
    """Launch game exe if not already running. Returns a short status message."""
    exe_path = (exe_path or "").strip()
    if not exe_path:
        return "no_exe_path"
    p = Path(exe_path)
    if not p.is_file():
        return f"exe_not_found: {exe_path}"
    exe_name = p.name.lower()
    # Check if the process is already running
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"IMAGENAME eq {p.name}", "/NH"],
            capture_output=True, text=True, timeout=5,
        )
        if exe_name in result.stdout.lower():
            return "already_running"
    except Exception:
        pass  # tasklist failed – try launching anyway
    # Launch the game
    try:
        subprocess.Popen(
            [str(p)],
            cwd=str(p.parent),
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    except Exception as e:
        return f"launch_error: {e}"
    # Give the game a moment to create its window
    if wait_seconds > 0:
        time.sleep(wait_seconds)
    return "launched"


# ── Agent Start / Stop ────────────────────────────────────────────────

@app.post("/api/v1/start")
def api_start(payload: Dict[str, Any]) -> Dict[str, Any]:
    global _LAST_PIPELINE_ERROR
    _LAST_PIPELINE_ERROR = ""
    active_profile, profile_settings, cfg = _get_active_profile_settings()
    effective_payload = dict(profile_settings)
    effective_payload.update(dict(payload or {}))
    if payload.get("active_profile"):
        active_profile = _normalize_profile_name(payload.get("active_profile"))
        cfg["active_profile"] = active_profile
    profiles = dict(cfg.get("profiles") or {})
    profiles[active_profile] = _normalize_profile_settings(effective_payload)
    cfg["profiles"] = profiles
    cfg["active_profile"] = active_profile
    _save_app_config(cfg)
    effective_payload["active_profile"] = active_profile
    effective_payload["profile_name"] = active_profile
    effective_payload["skill_order"] = _normalize_skill_order(effective_payload.get("skill_order"))
    # --- clear logs ---
    try:
        for fn in ("agent.out.log", "agent.err.log"):
            try:
                (LOGS_DIR / fn).write_text("", encoding="utf-8")
            except Exception:
                pass
    except Exception:
        pass
    # --- auto-launch game ---
    _game_launch_status = ""
    try:
        _game_launch_status = _maybe_launch_game(str(effective_payload.get("game_exe_path") or ""))
    except Exception:
        _game_launch_status = f"exception: {traceback.format_exc(limit=5)}"
    try:
        log_path = LOGS_DIR / "agent.out.log"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"[game_launch] {_game_launch_status}\n")
    except Exception:
        pass
    # --- stop any existing pipeline ---
    try:
        if _PIPELINE_RUNNING:
            _stop_pipeline()
    except Exception:
        pass
    # --- start new pipeline ---
    try:
        _start_pipeline(payload=effective_payload)
    except Exception:
        _LAST_PIPELINE_ERROR = traceback.format_exc(limit=20)
    return status()


@app.post("/api/v1/stop")
def api_stop() -> Dict[str, Any]:
    _stop_pipeline()
    return status()


# ── Logs ──────────────────────────────────────────────────────────────

@app.get("/api/v1/logs")
def api_logs(
    name: str = Query(...),
    lines: int = Query(200, ge=1, le=2000),
) -> PlainTextResponse:
    safe = os.path.basename(name)
    path = LOGS_DIR / safe
    return PlainTextResponse(_tail_text(path, lines))
