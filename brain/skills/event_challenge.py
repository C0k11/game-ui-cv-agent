"""Event Challenge phase — EXTRACTED / NOT YET WIRED IN.

This module holds the challenge-stage code that used to live in
``event_activity.py``. It was moved out because Challenge stages
require a dedicated team composition + turn-order plan that the main
event-activity skill cannot produce on its own. Wire this back in as
a separate skill (or via a profile opt-in) once the team-selection /
turn-scripting subsystem exists.

API sketch for re-integration::

    class EventChallengeSkill(BaseSkill):
        def __init__(self, team_preset: dict): ...
        def tick(self, screen): run _challenge(screen)

For now this file is a reference copy — it is NOT imported anywhere
and EventActivity does NOT dispatch to `challenge` phase.
"""
from __future__ import annotations

from typing import Any, Dict

from brain.skills.base import (
    ScreenState,
    action_back,
    action_click,
    action_click_box,
    action_wait,
)


class EventChallengeLogic:
    """Standalone challenge-phase state machine. NOT a BaseSkill yet —
    call ``tick(screen)`` from an owner skill when ready."""

    def __init__(self, owner):
        # ``owner`` gives access to the shared helpers
        # (_has_game_ui, _is_auto_battle, _handle_battle_speed,
        # _is_event_story_screen, _save_state, log, sub_state setter,
        # plus the _challenge_* state vars and _phase_ticks).
        self.o = owner

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        o = self.o
        o._phase_ticks += 1

        if not o._has_game_ui(screen) and len(screen.ocr_boxes) > 5:
            o._no_game_ticks += 1
            if o._no_game_ticks > 15:
                o.log("challenge: no game UI for 15+ ticks, resetting to enter")
                o._no_game_ticks = 0
                o.sub_state = "enter"
                return action_back("no game UI, pressing back to recover")
            return action_wait(500, "no game UI detected, waiting")
        else:
            o._no_game_ticks = 0

        auto = screen.find_any_text(["AUTO", "自動", "自动"],
                                    region=(0.70, 0.0, 1.0, 0.24), min_conf=0.55)
        if auto and not o._grid_auto_enabled:
            o._grid_auto_enabled = True
            return action_click_box(auto, "enable auto on event grid")
        end_turn = screen.find_any_text(["結束回合", "结束回合", "End Turn"],
                                        region=(0.65, 0.0, 1.0, 0.28), min_conf=0.5)
        if end_turn:
            return action_click_box(end_turn, "end turn on event grid")

        skip = screen.find_any_text(["SKIP", "Skip", "跳過", "跳过"], min_conf=0.65)
        if skip:
            o._challenge_idle_ticks = 0
            return action_click_box(skip, "skip event challenge cutscene")

        if o._is_event_story_screen(screen):
            o._challenge_idle_ticks = 0
            menu = screen.find_any_text(["MENU"], region=(0.82, 0.0, 1.0, 0.14), min_conf=0.65)
            if menu:
                return action_click_box(menu, "click MENU during challenge cutscene")
            return action_click(0.94, 0.05, "click MENU (hardcoded) during challenge")

        if o._is_auto_battle(screen):
            o._challenge_idle_ticks = 0
            speed_action = o._handle_battle_speed(screen)
            if speed_action:
                return speed_action
            return action_wait(1500, "challenge battle in progress (auto)")

        if len(screen.ocr_boxes) <= 5:
            o._challenge_idle_ticks = 0
            return action_wait(500, "challenge loading/battle in progress")

        mission_info = screen.find_any_text(
            ["任務資訊", "任务资讯", "任務資讯", "任务資訊", "任務资讯"],
            min_conf=0.55,
        )
        if mission_info:
            o._challenge_idle_ticks = 0
            start_btn = screen.find_any_text(
                ["任務開始", "任务开始", "任務开始", "任务開始"],
                region=(0.55, 0.60, 0.90, 0.85), min_conf=0.55,
            )
            if start_btn:
                return action_click_box(start_btn, "click 任務開始 (challenge battle)")
            no_goals = screen.find_any_text(
                ["該關卡無法", "该关卡无法", "没有任務目標", "沒有任務目標",
                 "没有任务目标"],
                region=(0.50, 0.55, 0.95, 0.90), min_conf=0.60,
            )
            if no_goals:
                o._challenge_completed_count += 1
                o._challenge_stage_index += 1
                if o._challenge_completed_count >= 5:
                    o._challenge_done = True
                    o._save_state()
                    o._phase_ticks = 0
                    o.sub_state = "enter"
                    o._challenge_sweep_stage = 0
                    return action_back("all challenges completed, exiting")
                return action_back("skip already-completed challenge battle")
            return action_wait(400, "mission popup loading (challenge)")

        sortie = screen.find_any_text(
            ["出擊", "出击", "出撃", "出擎", "開始作戰", "开始作战",
             "戰鬥開始", "战斗开始"],
            region=(0.70, 0.75, 1.0, 0.98), min_conf=0.55,
        )
        if sortie:
            o._challenge_idle_ticks = 0
            return action_click_box(sortie, "click sortie (challenge battle)")

        result_confirm = screen.find_any_text(
            ["確認", "确认", "確定", "确定", "確", "确", "OK"],
            region=(0.25, 0.50, 0.95, 0.98), min_conf=0.6,
        )
        if result_confirm and screen.find_any_text(
                ["戰鬥結果", "战斗结果", "VICTORY", "DEFEAT",
                 "勝利", "敗北", "胜利", "败北"], min_conf=0.55):
            o._challenge_idle_ticks = 0
            return action_click_box(result_confirm, "confirm challenge battle result")

        if not o._challenge_tab_clicked:
            tab = screen.find_any_text(
                ["Challenge", "挑戰", "挑战"],
                region=(0.50, 0.0, 1.0, 0.24), min_conf=0.5,
            )
            if tab:
                o._challenge_tab_clicked = True
                o._challenge_idle_ticks = 0
                return action_click_box(tab, "switch to event challenge tab")

        if o._challenge_sweep_stage == 0:
            max_btn = screen.find_any_text(["MAX"], min_conf=0.7)
            if max_btn:
                o._challenge_sweep_stage = 1
                o._challenge_idle_ticks = 0
                return action_click_box(max_btn, "challenge sweep max")

            all_entries = []
            for pat in ["入場", "入场"]:
                all_entries.extend(
                    screen.find_text(pat, region=(0.45, 0.16, 1.0, 0.95), min_conf=0.55)
                )
            all_entries.sort(key=lambda b: b.cy)
            if all_entries:
                idx = min(o._challenge_stage_index, len(all_entries) - 1)
                if o._challenge_stage_index >= len(all_entries):
                    o._challenge_done = True
                    o._save_state()
                    o._phase_ticks = 0
                    o.sub_state = "enter"
                    o._challenge_sweep_stage = 0
                    return action_wait(250, "all challenge entries exhausted")
                target = all_entries[idx]
                o._challenge_idle_ticks = 0
                return action_click_box(target, f"enter event challenge stage {idx + 1}")

            sweep = screen.find_any_text(["掃蕩", "扫荡"],
                                         region=(0.45, 0.35, 1.0, 0.75), min_conf=0.55)
            if sweep:
                o._challenge_idle_ticks = 0
                return action_click_box(sweep, "open event challenge sweep")

        elif o._challenge_sweep_stage == 1:
            sweep_start = screen.find_any_text(["掃蕩開始", "扫荡开始"], min_conf=0.55)
            if sweep_start:
                o._challenge_sweep_stage = 2
                o._challenge_idle_ticks = 0
                return action_click_box(sweep_start, "start challenge sweep")
            start_btn = screen.find_any_text(["開始", "开始"],
                                             region=(0.50, 0.40, 1.0, 0.70), min_conf=0.55)
            if start_btn:
                o._challenge_sweep_stage = 2
                o._challenge_idle_ticks = 0
                return action_click_box(start_btn, "start challenge sweep (fallback)")

        elif o._challenge_sweep_stage == 2:
            confirm = screen.find_any_text(
                ["確認", "确认", "確定", "确定", "確", "确", "OK"],
                region=(0.30, 0.30, 0.80, 0.92), min_conf=0.55,
            )
            if confirm:
                o._challenge_sweep_stage = 3
                o._challenge_idle_ticks = 0
                return action_click_box(confirm, "confirm challenge sweep")

        elif o._challenge_sweep_stage == 3:
            done = screen.find_any_text(
                ["掃蕩完成", "扫荡完成", "獲得獎勵", "获得奖励",
                 "確認", "确认", "OK"], min_conf=0.55,
            )
            if done:
                o._challenge_done = True
                o._save_state()
                o._phase_ticks = 0
                o.sub_state = "enter"
                o._challenge_sweep_stage = 0
                return action_click_box(done, "close challenge sweep result")

        o._challenge_idle_ticks += 1
        if o._phase_ticks > 200 or o._challenge_idle_ticks > 20:
            o._challenge_done = True
            o._save_state()
            o._phase_ticks = 0
            o.sub_state = "enter"
            o._challenge_sweep_stage = 0
            return action_wait(250, "challenge phase done")

        return action_wait(400, "event challenge scanning")
