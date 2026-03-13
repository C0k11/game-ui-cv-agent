"""CampaignPushSkill: fallback campaign progression/AP sink.

Goal:
- Enter campaign mission stage list.
- Prefer Normal stages and attempt up to N pushes.
- Use sweep when available; otherwise start one fight and wait for result.
- Exit to lobby safely on AP exhaustion or completion.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from brain.skills.base import (
    BaseSkill,
    OcrBox,
    ScreenState,
    action_back,
    action_click,
    action_click_box,
    action_done,
    action_swipe,
    action_wait,
)


class CampaignPushSkill(BaseSkill):
    _RESULT_TEXTS = [
        "獲得獎勵", "获得奖励", "獲得道具", "获得道具",
        "戰鬥結果", "战斗结果", "掃蕩完成", "扫荡完成",
        "關卡完成", "关卡完成", "任務完成", "任务完成",
    ]

    def __init__(self, *, max_pushes: int = 3):
        super().__init__("CampaignPush")
        self.max_ticks = 170
        try:
            pushes = int(max_pushes)
        except Exception:
            pushes = 3
        self._max_pushes = min(30, max(1, pushes))

        self._enter_attempts: int = 0
        self._select_attempts: int = 0
        self._scroll_attempts: int = 0
        self._open_ticks: int = 0

        self._pushes_done: int = 0
        self._awaiting_result: bool = False
        self._ap_empty: bool = False

        self._sweep_stage: int = 0
        self._sweep_ticks: int = 0
        self._fight_ticks: int = 0

    def reset(self) -> None:
        super().reset()
        self._enter_attempts = 0
        self._select_attempts = 0
        self._scroll_attempts = 0
        self._open_ticks = 0

        self._pushes_done = 0
        self._awaiting_result = False
        self._ap_empty = False

        self._sweep_stage = 0
        self._sweep_ticks = 0
        self._fight_ticks = 0

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1

        if self.ticks >= self.max_ticks:
            return action_done("campaign push timeout")

        no_ap = screen.find_any_text(
            ["AP不足", "體力不足", "体力不足", "購買AP", "购买AP"],
            min_conf=0.6,
        )
        if no_ap:
            self._ap_empty = True
            self.sub_state = "exit"
            cancel = screen.find_any_text(["取消"], min_conf=0.6)
            if cancel:
                return action_click_box(cancel, "cancel AP purchase")
            return action_back("dismiss AP popup")

        result = screen.find_any_text(self._RESULT_TEXTS, min_conf=0.6)
        if result:
            self._mark_push_complete("result popup")
            confirm = screen.find_any_text(["確認", "确认", "確定", "确定", "確", "确", "OK"], min_conf=0.55)
            if confirm:
                return action_click_box(confirm, "dismiss campaign result popup")
            return action_click(0.5, 0.9, "dismiss campaign result popup fallback")

        if self.sub_state not in ("sweep", "fight_wait"):
            popup = self._handle_common_popups(screen)
            if popup:
                return popup

        if screen.is_loading():
            return action_wait(900, "campaign push loading")

        if self.sub_state == "":
            self.sub_state = "enter"

        if self.sub_state == "enter":
            return self._enter(screen)
        if self.sub_state == "select_normal":
            return self._select_normal(screen)
        if self.sub_state == "select_stage":
            return self._select_stage(screen)
        if self.sub_state == "open_stage":
            return self._open_stage(screen)
        if self.sub_state == "sweep":
            return self._sweep(screen)
        if self.sub_state == "fight_wait":
            return self._fight_wait(screen)
        if self.sub_state == "exit":
            return self._exit(screen)

        return action_wait(300, "campaign push unknown state")

    def _mark_push_complete(self, reason: str) -> None:
        if self._awaiting_result:
            self._pushes_done += 1
            self.log(f"push complete {self._pushes_done}/{self._max_pushes} ({reason})")
        self._awaiting_result = False
        self._sweep_stage = 0
        self._sweep_ticks = 0
        self._fight_ticks = 0
        self._open_ticks = 0
        if self._pushes_done >= self._max_pushes:
            self.sub_state = "exit"
        else:
            self.sub_state = "select_stage"

    def _is_campaign_hub(self, screen: ScreenState) -> bool:
        hub_markers = screen.find_any_text(
            [
                "劇情", "剧情", "懸賞通緝", "悬赏通缉", "總力", "总力",
                "學園交流", "学园交流", "特殊任務", "特殊任务", "制約解除",
            ],
            min_conf=0.5,
        )
        if hub_markers:
            return True
        return bool(screen.find_text_one(r"Area\s*\d+", min_conf=0.5))

    def _is_stage_list(self, screen: ScreenState) -> bool:
        if screen.find_text_one(r"\d{1,2}\s*-\s*\d{1,2}", region=(0.06, 0.16, 0.95, 0.92), min_conf=0.55):
            return True
        return bool(
            screen.find_any_text(
                ["普通", "Normal", "困難", "困难", "Hard", "關卡目錄", "关卡目录"],
                region=(0.0, 0.05, 1.0, 0.22),
                min_conf=0.5,
            )
        )

    def _is_stage_panel(self, screen: ScreenState) -> bool:
        action_btn = screen.find_any_text(
            ["掃蕩", "扫荡", "開始戰鬥", "开始战斗", "出擊", "出击", "挑戰", "挑战"],
            region=(0.35, 0.35, 1.0, 0.98),
            min_conf=0.5,
        )
        if action_btn:
            return True
        return bool(screen.find_text_one(r"\d{1,2}\s*-\s*\d{1,2}", region=(0.30, 0.06, 0.95, 0.25), min_conf=0.5))

    def _parse_stage_tuple(self, text: str) -> Optional[Tuple[int, int]]:
        m = re.search(r"(\d{1,2})\s*-\s*(\d{1,2})", text)
        if not m:
            return None
        try:
            return int(m.group(1)), int(m.group(2))
        except Exception:
            return None

    def _pick_stage_box(self, stage_boxes: List[OcrBox]) -> OcrBox:
        def score(box: OcrBox) -> Tuple[int, int, float, float]:
            parsed = self._parse_stage_tuple(box.text) or (0, 0)
            return parsed[0], parsed[1], box.cy, box.cx

        return max(stage_boxes, key=score)

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        if self._pushes_done >= self._max_pushes:
            self.sub_state = "exit"
            return action_wait(200, "campaign push target reached")

        current = self.detect_current_screen(screen)

        if current == "Mission":
            if self._is_campaign_hub(screen):
                mission_card = screen.find_any_text(["任務", "任务"], region=(0.38, 0.12, 0.72, 0.38), min_conf=0.55)
                if mission_card:
                    return action_click_box(mission_card, "campaign hub -> open mission stage list")
                return action_click(0.56, 0.24, "campaign hub mission card fallback")
            self.sub_state = "select_normal"
            return action_wait(250, "campaign stage list ready")

        if current == "DailyTasks":
            return action_back("back from DailyTasks")

        if current == "Lobby":
            self._enter_attempts += 1
            mission_btn = screen.find_any_text(["任務", "任务"], region=(0.80, 0.70, 1.0, 0.92), min_conf=0.55)
            if mission_btn:
                return action_click_box(mission_btn, "open campaign from lobby right entry")
            return action_click(0.95, 0.83, "open campaign from lobby fallback")

        if current and current != "Mission":
            return action_back(f"campaign push back from {current}")

        self._enter_attempts += 1
        if self._enter_attempts > 10:
            self.sub_state = "exit"
            return action_wait(250, "campaign push unable to enter mission")
        return action_wait(400, "campaign push entering mission")

    def _select_normal(self, screen: ScreenState) -> Dict[str, Any]:
        if self._pushes_done >= self._max_pushes:
            self.sub_state = "exit"
            return action_wait(200, "campaign push target reached")

        if self._is_stage_panel(screen):
            self.sub_state = "open_stage"
            self._open_ticks = 0
            return action_wait(200, "stage panel detected")

        if not self._is_stage_list(screen):
            self.sub_state = "enter"
            return action_wait(300, "stage list lost, re-enter")

        normal = screen.find_any_text(["普通", "Normal", "NORMAL"], region=(0.0, 0.06, 0.45, 0.23), min_conf=0.5)
        if normal:
            self.sub_state = "select_stage"
            return action_click_box(normal, "select Normal tab")

        hard = screen.find_any_text(["困難", "困难", "Hard", "HARD"], region=(0.0, 0.06, 0.65, 0.24), min_conf=0.55)
        if hard:
            self.sub_state = "select_stage"
            return action_click(max(0.05, hard.cx - 0.18), hard.cy, "switch from Hard to Normal tab")

        self._select_attempts += 1
        if self._select_attempts > 6:
            self.sub_state = "select_stage"
            return action_wait(200, "Normal tab assumption")
        return action_wait(300, "finding Normal tab")

    def _select_stage(self, screen: ScreenState) -> Dict[str, Any]:
        if self._pushes_done >= self._max_pushes:
            self.sub_state = "exit"
            return action_wait(200, "campaign push target reached")

        if self._is_stage_panel(screen):
            self.sub_state = "open_stage"
            self._open_ticks = 0
            return action_wait(200, "stage panel already open")

        if not self._is_stage_list(screen):
            self.sub_state = "enter"
            return action_wait(300, "stage list not detected")

        new_hint = screen.find_any_text(
            ["NEW", "未通關", "未通关", "首次", "初次", "可挑戰", "可挑战"],
            region=(0.06, 0.16, 0.95, 0.92),
            min_conf=0.5,
        )
        if new_hint:
            self.sub_state = "open_stage"
            self._open_ticks = 0
            return action_click_box(new_hint, "open preferred campaign stage")

        stage_boxes = screen.find_text(r"\d{1,2}\s*-\s*\d{1,2}", region=(0.06, 0.16, 0.95, 0.92), min_conf=0.5)
        if stage_boxes:
            target = self._pick_stage_box(stage_boxes)
            self.sub_state = "open_stage"
            self._open_ticks = 0
            return action_click_box(target, f"open stage {target.text}")

        if self._scroll_attempts < 5:
            self._scroll_attempts += 1
            return action_swipe(0.78, 0.84, 0.78, 0.30, 450, "scroll campaign stage list")

        self.sub_state = "exit"
        return action_wait(250, "no campaign stages found")

    def _open_stage(self, screen: ScreenState) -> Dict[str, Any]:
        if self._pushes_done >= self._max_pushes:
            self.sub_state = "exit"
            return action_wait(200, "campaign push target reached")

        if not self._is_stage_panel(screen):
            self._open_ticks += 1
            if self._open_ticks > 10:
                self.sub_state = "select_stage"
                return action_wait(250, "stage panel not opened")
            return action_wait(300, "waiting stage panel")

        if not self._awaiting_result:
            sweep = screen.find_any_text(["掃蕩", "扫荡"], region=(0.35, 0.35, 1.0, 0.98), min_conf=0.5)
            if sweep:
                self._awaiting_result = True
                self._sweep_stage = 0
                self._sweep_ticks = 0
                self.sub_state = "sweep"
                return action_wait(120, "prepare campaign sweep")

            start = screen.find_any_text(
                ["開始戰鬥", "开始战斗", "出擊", "出击", "挑戰", "挑战", "Start"],
                region=(0.35, 0.35, 1.0, 0.98),
                min_conf=0.5,
            )
            if start:
                self._awaiting_result = True
                self._fight_ticks = 0
                self.sub_state = "fight_wait"
                return action_click_box(start, "start campaign fight")

        self._open_ticks += 1
        if self._open_ticks > 12:
            self.sub_state = "select_stage"
            return action_back("leave stage panel")
        return action_wait(300, "finding sweep/start button")

    def _sweep(self, screen: ScreenState) -> Dict[str, Any]:
        self._sweep_ticks += 1

        if not self._awaiting_result:
            self.sub_state = "select_stage"
            return action_wait(200, "sweep cancelled")

        if self._sweep_ticks > 24:
            self._awaiting_result = False
            self.sub_state = "select_stage"
            return action_back("sweep timeout, back to stage list")

        cancel = screen.find_any_text(["取消"], region=(0.18, 0.55, 0.60, 0.90), min_conf=0.55)
        confirm = screen.find_any_text(
            ["確認", "确认", "確定", "确定", "掃蕩", "扫荡", "開始", "开始", "確", "确"],
            region=(0.40, 0.55, 0.92, 0.92),
            min_conf=0.52,
        )
        if cancel and confirm:
            self._sweep_stage = 3
            return action_click_box(confirm, "confirm sweep")

        if self._sweep_stage == 0:
            sweep = screen.find_any_text(["掃蕩", "扫荡"], region=(0.35, 0.35, 1.0, 0.98), min_conf=0.5)
            if sweep:
                self._sweep_stage = 1
                return action_click_box(sweep, "campaign sweep")
            return action_wait(250, "waiting sweep button")

        if self._sweep_stage == 1:
            max_btn = screen.find_any_text(["最大", "MAX", "Max"], region=screen.CENTER, min_conf=0.5)
            self._sweep_stage = 2
            if max_btn:
                return action_click_box(max_btn, "set sweep max")
            return action_wait(200, "sweep max not visible")

        if self._sweep_stage == 2:
            confirm2 = screen.find_any_text(
                ["確認", "确认", "確定", "确定", "掃蕩", "扫荡", "開始", "开始", "確", "确"],
                region=(0.35, 0.50, 0.95, 0.95),
                min_conf=0.5,
            )
            if confirm2:
                self._sweep_stage = 3
                return action_click_box(confirm2, "confirm campaign sweep")
            self._sweep_stage = 3
            return action_click(0.78, 0.82, "confirm campaign sweep fallback")

        return action_wait(700, "waiting campaign sweep result")

    def _fight_wait(self, screen: ScreenState) -> Dict[str, Any]:
        self._fight_ticks += 1

        if not self._awaiting_result:
            self.sub_state = "select_stage"
            return action_wait(200, "fight cancelled")

        skip = screen.find_any_text(["SKIP", "跳過", "跳过"], region=(0.72, 0.0, 1.0, 0.24), min_conf=0.5)
        if skip:
            return action_click_box(skip, "skip campaign fight animation")

        if self._fight_ticks > 90:
            self._awaiting_result = False
            self.sub_state = "select_stage"
            return action_back("campaign fight wait timeout")

        return action_wait(900, "waiting campaign fight result")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if screen.is_lobby():
            suffix = " AP exhausted" if self._ap_empty else ""
            return action_done(f"campaign push complete ({self._pushes_done}/{self._max_pushes}){suffix}")

        return action_back("campaign push exit to lobby")
