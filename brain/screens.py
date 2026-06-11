# -*- coding: utf-8 -*-
"""Screen semantizer — classify the current SCREEN/PHASE from the bbox set.

用户 2026-06-11 点题: "理解bbox语义, 通过辨别bbox知道现在是什么阶段, 显然我们
没有这个". This module is that layer: a declarative signature table mapping a
screen id to the cls constellation that proves we are on it, and
`classify_screen(yolo_boxes)` returning the best match.

Design rules
------------
- Signatures use ANCHOR classes only — stable structural UI (tabs, headers,
  fixed buttons). NEVER badges (红点/黄点/绿勾): they come and go with game
  state, not with the screen.
- `need` = classes that MUST all be present (conf >= MIN_CONF).
- `any_of` = at least one must be present (when a screen has alternative
  anchors, e.g. a button's active/grey variants).
- `forbid` = classes that must NOT be present — used to split sibling screens
  (Location Select vs region view both show 課程表 chrome; only the region
  view has 全体课程表).
- Scoring: all `need` present + one `any_of` + no `forbid` → matched.
  Multiple matches → the one with the most satisfied anchors wins.
- UNKNOWN is a first-class answer. Skills must treat UNKNOWN as "do not
  click anything you can't justify" (识别上了再决定, 不抢跑不乱点).

This is signature v1 (hand-written from live-verified probes 2026-06-11).
The trajectory-mining workflow enriches it offline; merge its output here.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from brain.skills import ui_classes as UC

MIN_CONF = 0.45

# ── v8 batch-sweep dialog classes (master 453-468; not yet in ui_classes) ──
KICKBACK_DEFENSE = "据点防御"          # 453  hall tile
CREDIT_RECYCLE = "信用货币回收"         # 454  hall tile
SWEEP_BATCH = "批量扫荡"               # 455  button on the 任務 stage screen
SWEEP_BATCH_START = "批量扫荡开始"      # 456  dialog confirm (active)
SWEEP_BATCH_START_GREY = "批量扫荡开始灰色"  # 457  dialog confirm (disabled)
SWEEP_GEAR_SMALL_SEL = "前置小装备已选中"    # 458
SWEEP_PLAN1 = "困难关卡刷取方案一"       # 459
SWEEP_PLAN2 = "困难关卡刷取方案二"       # 460
SWEEP_GEAR_SMALL = "前置小装备"          # 461
SWEEP_GEAR_BIG_SEL = "大装备已选中"      # 462
SWEEP_GEAR_BIG = "大装备"               # 463
SWEEP_PLAN1_SEL = "困难关卡刷去方案一已选中"  # 464 (typo in master: 刷去)
SWEEP_PLAN2_SEL_B = "困难方案刷取方案二已选中"  # 465 (label variant)
SWEEP_PLAN2_SEL = "困难关卡刷取方案二已选中"   # 466
BATTLE_COMPLETE = "战斗完成"            # 467  sweep/battle result header
GOTO_NOW = "立即前往"                  # 468


# screen_id → signature. Order matters only for documentation; matching is
# score-based across all entries.
SCREEN_SIGNATURES: Dict[str, Dict[str, List[str]]] = {
    # ── hub ────────────────────────────────────────────────────────────
    "lobby": {
        # bottom nav bar — require 3 of the entries via any_of pairs is too
        # loose; lobby is proven by ANY 3+ of these (handled specially below).
        "need": [],
        "any_of": [UC.NAV_CAFE, UC.NAV_SCHEDULE, UC.NAV_STUDENT, UC.NAV_SOCIAL,
                   UC.NAV_CRAFT, UC.NAV_SHOP, UC.NAV_RECRUIT],
        "min_any": 3,
        "forbid": [UC.COMBO_PACK, UC.COMBO_PACK_SEL, UC.SHOP_SELECT_ALL],
    },
    "loading": {"need": [UC.LOADING], "any_of": [], "forbid": []},

    # ── shops ──────────────────────────────────────────────────────────
    "shop_grid": {
        "need": [],
        "any_of": [UC.SHOP_SELECT_ALL, UC.SHOP_SELECT_ALL_GREY, UC.SHOP_ALL_SELECTED],
        "forbid": [UC.COMBO_PACK, UC.COMBO_PACK_SEL],
    },
    "buy_pyroxene_popup": {
        "need": [],
        "any_of": [UC.COMBO_PACK, UC.COMBO_PACK_SEL],
        "forbid": [],
    },

    # ── mail ───────────────────────────────────────────────────────────
    "mailbox": {
        "need": [],
        "any_of": [UC.CLAIM_ONCE_YELLOW, UC.CLAIM_ONCE_GREY],
        "forbid": [UC.NAV_CAFE],   # lobby shows neither claim-once variant
    },

    # ── schedule (課程表) ────────────────────────────────────────────────
    "schedule_location_select": {
        "need": [UC.SCHOOL_OFFICE],
        "any_of": [],
        "forbid": [UC.SCHED_ALL],
    },
    "schedule_region": {
        "need": [UC.SCHED_ALL],
        "any_of": [],
        "forbid": [],
    },

    # ── cafe ───────────────────────────────────────────────────────────
    "cafe_main": {
        "need": [],
        "any_of": [UC.CAFE_EARNINGS, UC.CAFE_INVITE_TICKET,
                   UC.CAFE_MOVE_1F, UC.CAFE_MOVE_2F],
        "forbid": [],
    },

    # ── story ──────────────────────────────────────────────────────────
    "story_hub": {
        "need": [],
        "any_of": [UC.STORY_MAIN, UC.STORY_SHORT, UC.STORY_SIDE],
        "min_any": 2,
        "forbid": [],
    },

    # ── battle ─────────────────────────────────────────────────────────
    "battle": {
        "need": [],
        "any_of": [UC.BATTLE_PAUSE, UC.BATTLE_AUTO_ON, UC.BATTLE_AUTO_OFF,
                   UC.BATTLE_2X, UC.BATTLE_3X, UC.BATTLE_1X],
        "min_any": 2,
        "forbid": [],
    },
    "battle_result": {
        "need": [],
        "any_of": [UC.BATTLE_WIN, BATTLE_COMPLETE],
        "forbid": [UC.BATTLE_PAUSE],
    },

    # ── batch sweep (批量掃蕩, v8 classes; spec via live walk 2026-06-11) ──
    "sweep_batch_dialog": {
        "need": [],
        "any_of": [SWEEP_BATCH_START, SWEEP_BATCH_START_GREY,
                   SWEEP_PLAN1, SWEEP_PLAN2, SWEEP_PLAN1_SEL, SWEEP_PLAN2_SEL,
                   SWEEP_PLAN2_SEL_B, SWEEP_GEAR_SMALL, SWEEP_GEAR_BIG,
                   SWEEP_GEAR_SMALL_SEL, SWEEP_GEAR_BIG_SEL],
        "min_any": 2,
        "forbid": [],
    },
    "campaign_stage_select": {
        "need": [SWEEP_BATCH],
        "any_of": [],
        "forbid": [SWEEP_BATCH_START, SWEEP_BATCH_START_GREY],
    },

    # ── task hall (任務大廳; live walk step 1, 2026-06-11) ──────────────
    # ★ per-activity dots are visible ONLY here — the lobby entry dot is NOT
    #   a work signal (user iron rule: enter and scan, never gate at entry).
    "task_hall": {
        "need": [],
        "any_of": ["任务关卡推图", "悬赏通缉", "学院交流会", "战术大赛",
                   "特殊任务", "剧情", KICKBACK_DEFENSE, CREDIT_RECYCLE],
        "min_any": 3,
        "forbid": [UC.NAV_CAFE],
    },
    "stage_select": {
        # 任務 normal/hard stage list (批量掃蕩 button lives bottom-left)
        "need": [],
        "any_of": ["普通关卡选中", "困难关卡选中", "困难关卡", "普通关卡"],
        "forbid": [SWEEP_BATCH_START, SWEEP_BATCH_START_GREY],
    },

    # ── mined from trajectories (workflow 2026-06-11, presence-rate based) ─
    "craft_main": {
        "need": ["快速制造"],     # unique anchor 0.875, nowhere else
        "any_of": [],
        "forbid": [],
    },
    "daily_mission": {
        "need": [],
        "any_of": [UC.CLAIM_ALL_YELLOW, UC.CLAIM_ALL_GREY, UC.CLAIM_ONEKEY_GREY],
        "forbid": ["快速制造", UC.CLAIM_ONCE_YELLOW],
    },
    "momo_chat": {
        # list + dialog share the chat-area anchor; dialog-only markers
        # (学生发送信息中/学生信息回复选项/前往羁绊剧情) refine when present.
        "need": ["momotalk学生聊天区域已进入"],
        "any_of": [],
        "forbid": [UC.BTN_HOME],
    },
    "bond_story_playback": {
        "need": [UC.STORY_MENU],  # 9/9 unique; NO top bar at all here
        "any_of": [],
        "forbid": [UC.TOPBAR_AP],
    },
    "cafe_earnings_popup": {
        # post-claim earnings popup (mined; reason text was misleading):
        # grey claim + popup X over the cafe scene, 咖啡厅收益 hidden.
        "need": [UC.BTN_CLOSE_X, UC.CLAIM_GREY],
        "any_of": [],
        "forbid": [UC.CAFE_EARNINGS, UC.CLAIM_ONCE_YELLOW],
    },

    # ── sweep progress / result pages (live walk steps 6-8) ────────────
    "sweep_running": {
        "need": [UC.BATTLE_SKIP],          # skip键 fast-forward overlay
        "any_of": [],
        "forbid": [UC.BTN_CONFIRM],
    },
    "result_page": {
        # Generic single-confirm result/notice page: 確認 + popup X but NO
        # 取消 (掃蕩完成 loot page, 通知/背包已滿 notices, etc.). A cancel
        # present means it's a DECISION dialog instead → confirm_dialog.
        "need": [UC.BTN_CONFIRM, UC.BTN_CLOSE_X],
        "any_of": [],
        "forbid": [UC.BTN_CANCEL, SWEEP_PLAN1, SWEEP_PLAN2, UC.SHOP_SELECT_ALL],
    },

    # ── dialogs (generic, lowest priority — see _DIALOG_IDS) ───────────
    "confirm_dialog": {
        "need": [UC.BTN_CONFIRM, UC.BTN_CANCEL],
        "any_of": [],
        "forbid": [],
    },
}

# Facility-screen marker (mined): 回大厅按钮+返回键 ≈1.0 on every facility
# screen, 0.0 on lobby / MomoTalk / story playback. Useful as a cheap
# "inside some facility" predicate when the specific screen is unknown.
FACILITY_MARKERS = [UC.BTN_HOME, UC.BTN_BACK]

# Dialogs OVERLAY a screen — report them alongside, not instead.
_DIALOG_IDS = {"confirm_dialog"}


def classify_screen(yolo_boxes, min_conf: float = MIN_CONF
                    ) -> Tuple[str, float, Optional[str]]:
    """(screen_id, score, overlay_dialog_id|None) from the bbox constellation.

    score = matched anchors / signature anchors (0..1). Returns ("unknown",
    0.0, dialog) when nothing matches — callers must NOT blind-click then.
    """
    present = {b.cls_name for b in (yolo_boxes or [])
               if getattr(b, "confidence", 0.0) >= min_conf}

    dialog = None
    if all(c in present for c in SCREEN_SIGNATURES["confirm_dialog"]["need"]):
        dialog = "confirm_dialog"

    best_id, best_score = "unknown", 0.0
    for sid, sig in SCREEN_SIGNATURES.items():
        if sid in _DIALOG_IDS:
            continue
        need = sig.get("need", [])
        if any(c not in present for c in need):
            continue
        if any(c in present for c in sig.get("forbid", [])):
            continue
        any_of = sig.get("any_of", [])
        min_any = sig.get("min_any", 1 if any_of else 0)
        hits = sum(1 for c in any_of if c in present)
        if hits < min_any:
            continue
        total = len(need) + max(min_any, 1)
        score = (len(need) + min(hits, len(any_of))) / max(1, total)
        if score > best_score:
            best_id, best_score = sid, score
    return best_id, best_score, dialog


def screen_of(screen) -> Tuple[str, float, Optional[str]]:
    """Convenience: classify a ScreenState-like object (has .yolo_boxes)."""
    return classify_screen(getattr(screen, "yolo_boxes", None))
