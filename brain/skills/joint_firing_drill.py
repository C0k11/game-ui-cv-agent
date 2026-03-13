"""JointFiringDrillSkill: clear Joint Firing Drill / Tactical Exam tickets.

reference parity target:
- Enter joint firing drill from campaign hub
- Claim drill rewards
- Consume available drill tickets (sweep when possible, otherwise fight)
- Return to lobby
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from brain.skills.base import (
    BaseSkill,
    ScreenState,
    OcrBox,
    action_back,
    action_click,
    action_click_box,
    action_done,
    action_wait,
)


class JointFiringDrillSkill(BaseSkill):
    _DIFFICULTIES: List[Tuple[str, List[str]]] = [
        ("INSANE", ["INSANE"]),
        ("EXTREME", ["EXTREME"]),
        ("HARDCORE", ["HARDCORE"]),
        ("VERYHARD", ["VERYHARD", "VERY HARD"]),
        ("HARD", ["HARD"]),
        ("NORMAL", ["NORMAL"]),
    ]

    _MODES: List[Tuple[str, List[str], Tuple[float, float]]] = [
        ("assault", ["Assault", "突擊", "突击", "突襲", "突袭"], (0.15, 0.52)),
        ("defense", ["Defense", "防禦", "防御"], (0.56, 0.55)),
        ("shooting", ["Shooting", "射擊", "射击"], (0.87, 0.48)),
    ]

    def __init__(self):
        super().__init__("JointFiringDrill")
        self.max_ticks = 170

        self._enter_ticks: int = 0
        self._reward_ticks: int = 0
        self._ticket_ticks: int = 0
        self._select_ticks: int = 0
        self._mode_pick_idx: int = 0

        self._tickets_remaining: int = -1
        self._fights_done: int = 0

        self._fight_stage: int = 0
        self._fight_ticks: int = 0

        self._sweep_stage: int = 0
        self._sweep_ticks: int = 0

    def reset(self) -> None:
        super().reset()
        self._enter_ticks = 0
        self._reward_ticks = 0
        self._ticket_ticks = 0
        self._select_ticks = 0
        self._mode_pick_idx = 0

        self._tickets_remaining = -1
        self._fights_done = 0

        self._fight_stage = 0
        self._fight_ticks = 0

        self._sweep_stage = 0
        self._sweep_ticks = 0

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1

        if self.ticks >= self.max_ticks:
            self.log("timeout")
            return action_done("joint firing timeout")

        # Generic reward/result popups.
        reward_popup = screen.find_any_text(
            [
                "獲得獎勵", "获得奖励", "獲得道具", "获得道具",
                "掃蕩完成", "扫荡完成", "戰鬥結果", "战斗结果",
            ],
            min_conf=0.6,
        )
        if reward_popup:
            confirm = screen.find_any_text(
                ["確認", "确认", "確定", "确定", "確", "确", "OK"],
                min_conf=0.6,
            )
            if confirm:
                return action_click_box(confirm, "dismiss drill reward/result popup")
            return action_click(0.5, 0.9, "dismiss drill reward/result popup fallback")

        # No ticket popup.
        no_ticket = screen.find_any_text(
            ["票券不足", "票券不夠", "票券不够", "入場券不足", "入场券不足", "Drill Ticket"],
            min_conf=0.6,
        )
        no_ticket_hint = screen.find_any_text(["不足", "不夠", "不够"], min_conf=0.65)
        if no_ticket and no_ticket_hint:
            self.log("drill popup indicates no tickets")
            self.sub_state = "exit"
            confirm = screen.find_any_text(
                ["確認", "确认", "確定", "确定", "確", "确", "OK"],
                min_conf=0.6,
            )
            if confirm:
                return action_click_box(confirm, "confirm no drill ticket popup")
            return action_back("close no drill ticket popup")

        popup = self._handle_common_popups(screen)
        if popup:
            return popup

        if screen.is_loading():
            return action_wait(800, "joint firing loading")

        if self.sub_state == "":
            self.sub_state = "enter"

        if self.sub_state == "enter":
            return self._enter(screen)
        if self.sub_state == "collect_rewards":
            return self._collect_rewards(screen)
        if self.sub_state == "check_tickets":
            return self._check_tickets(screen)
        if self.sub_state == "select_mode":
            return self._select_mode(screen)
        if self.sub_state == "action_mode":
            return self._action_mode(screen)
        if self.sub_state == "sweep":
            return self._sweep(screen)
        if self.sub_state == "fight":
            return self._fight(screen)
        if self.sub_state == "exit":
            return self._exit(screen)

        return action_wait(300, "joint firing unknown state")

    def _is_drill_screen(self, screen: ScreenState) -> bool:
        header = screen.find_any_text(
            [
                "學園交流會", "学园交流会",
                "聯合火力", "联合火力",
                "綜合戰術", "综合战术",
                "制約解除", "制约解除",
                "Joint Firing", "Drill",
            ],
            region=(0.0, 0.0, 0.42, 0.20),
            min_conf=0.5,
        )
        if header:
            return True

        marker = screen.find_any_text(
            ["Assault", "Defense", "Shooting", "Season Record", "Drill Ticket"],
            region=(0.05, 0.06, 1.0, 0.95),
            min_conf=0.55,
        )
        return marker is not None

    def _parse_tickets(self, screen: ScreenState) -> int:
        # Prefer top-right counters like "3/5".
        for box in screen.find_text(r"\d+\s*[/|]\s*\d+", region=(0.68, 0.00, 1.0, 0.22), min_conf=0.35):
            m = re.search(r"(\d+)\s*[/|]\s*(\d+)", box.text)
            if not m:
                continue
            cur = int(m.group(1))
            cap = int(m.group(2))
            if 0 <= cur <= 10 and 1 <= cap <= 10:
                return cur

        # Fallback: any small single number near top-right.
        candidates: List[int] = []
        for box in screen.ocr_boxes:
            if box.confidence < 0.5 or box.cx < 0.78 or box.cy > 0.25:
                continue
            m = re.fullmatch(r"\d", box.text.strip())
            if not m:
                continue
            candidates.append(int(box.text.strip()))
        if candidates:
            return max(candidates)
        return -1

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        self._enter_ticks += 1

        if self._is_drill_screen(screen):
            self.log("inside joint firing drill")
            self.sub_state = "collect_rewards"
            self._reward_ticks = 0
            return action_wait(350, "entered joint firing")

        current = self.detect_current_screen(screen)

        if current == "Lobby":
            campaign_btn = screen.find_any_text(
                ["任務", "任务"],
                region=(0.80, 0.70, 1.0, 0.90),
                min_conf=0.6,
            )
            if campaign_btn:
                return action_click_box(campaign_btn, "open campaign for joint firing")
            return action_click(0.95, 0.83, "open campaign area (hardcoded)")

        if current == "Mission":
            drill_entry = screen.find_any_text(
                [
                    "學園交流會", "学园交流会",
                    "聯合火力", "联合火力",
                    "綜合戰術", "综合战术",
                    "制約解除", "制约解除",
                    "Drill",
                ],
                min_conf=0.5,
            )
            if drill_entry:
                self.log(f"clicking joint firing entry '{drill_entry.text}'")
                return action_click_box(drill_entry, "enter joint firing")

            if self._enter_ticks > 4:
                # reference Global icon near (1002,439) on 1280x720.
                return action_click(0.78, 0.61, "enter joint firing (hardcoded)")
            return action_wait(400, "looking for joint firing entry")

        if current and current not in ("Mission",):
            return action_back(f"back from {current}")

        if self._enter_ticks > 24:
            self.log("failed to reach joint firing, exiting")
            self.sub_state = "exit"
            return action_wait(250, "joint firing enter timeout")

        return action_wait(500, "entering joint firing")

    def _collect_rewards(self, screen: ScreenState) -> Dict[str, Any]:
        self._reward_ticks += 1

        claim_btn = screen.find_any_text(
            ["領取", "领取", "收取", "全部領取", "全部领取", "受取"],
            region=(0.62, 0.56, 1.0, 0.95),
            min_conf=0.55,
        )
        if claim_btn:
            return action_click_box(claim_btn, "claim joint firing reward")

        if self._reward_ticks > 10:
            self.sub_state = "check_tickets"
            self._ticket_ticks = 0
            return action_wait(250, "reward pass complete")

        return action_wait(300, "checking drill rewards")

    def _check_tickets(self, screen: ScreenState) -> Dict[str, Any]:
        self._ticket_ticks += 1

        tickets = self._parse_tickets(screen)
        if tickets >= 0:
            self._tickets_remaining = tickets
            self.log(f"drill tickets remaining: {tickets}")
            if tickets <= 0:
                self.sub_state = "exit"
                return action_wait(200, "no drill tickets left")
            self.sub_state = "select_mode"
            self._select_ticks = 0
            return action_wait(250, f"tickets={tickets}, selecting drill mode")

        if self._ticket_ticks > 8:
            # OCR miss fallback: try once anyway.
            self.log("ticket OCR unreadable, assuming tickets remain")
            self._tickets_remaining = max(self._tickets_remaining, 1)
            self.sub_state = "select_mode"
            self._select_ticks = 0
            return action_wait(250, "ticket OCR unreadable, continue")

        return action_wait(350, "reading drill tickets")

    def _pick_mode_by_text(self, screen: ScreenState) -> Optional[OcrBox]:
        for _, aliases, _ in self._MODES:
            hit = screen.find_any_text(aliases, min_conf=0.55)
            if hit:
                return hit
        return None

    def _select_mode(self, screen: ScreenState) -> Dict[str, Any]:
        self._select_ticks += 1

        # If we already landed in a mode detail page, continue.
        detail_marker = screen.find_any_text(
            ["Season Record", "記錄", "记录", "Drill Ticket", "推薦等級", "推荐等级"],
            min_conf=0.5,
        )
        if detail_marker:
            self.sub_state = "action_mode"
            return action_wait(250, "drill mode detail ready")

        mode_hit = self._pick_mode_by_text(screen)
        if mode_hit:
            self.sub_state = "action_mode"
            return action_click_box(mode_hit, "select drill mode")

        # Hardcoded mode rotation fallback.
        if self._select_ticks > 2:
            _, _, pos = self._MODES[self._mode_pick_idx % len(self._MODES)]
            self._mode_pick_idx += 1
            self.sub_state = "action_mode"
            return action_click(*pos, "select drill mode (hardcoded)")

        return action_wait(300, "looking for drill mode")

    def _action_mode(self, screen: ScreenState) -> Dict[str, Any]:
        # Prefer sweep when available and ticket count > 1.
        sweep_btn = screen.find_any_text(["掃蕩", "扫荡", "Sweep"], min_conf=0.55)
        if sweep_btn and self._tickets_remaining > 1:
            self.sub_state = "sweep"
            self._sweep_stage = 0
            self._sweep_ticks = 0
            return action_click_box(sweep_btn, "open drill sweep")

        # Otherwise fight one ticket.
        start_btn = screen.find_any_text(
            ["開始演習", "开始演习", "開始作戰", "开始作战", "出擊", "出击", "Start"],
            region=(0.45, 0.55, 1.0, 0.95),
            min_conf=0.5,
        )
        if start_btn:
            self.sub_state = "fight"
            self._fight_stage = 0
            self._fight_ticks = 0
            return action_click_box(start_btn, "start drill fight")

        # Difficulty selection if present.
        for _, aliases in self._DIFFICULTIES:
            diff = screen.find_any_text(aliases, min_conf=0.55)
            if diff:
                return action_click_box(diff, "select highest visible drill difficulty")

        # If nothing actionable, return to menu and retry.
        self._select_ticks += 1
        if self._select_ticks > 10:
            self.sub_state = "check_tickets"
            self._ticket_ticks = 0
            return action_back("no drill action found, back to ticket check")

        return action_wait(350, "checking drill mode actions")

    def _sweep(self, screen: ScreenState) -> Dict[str, Any]:
        self._sweep_ticks += 1

        if self._sweep_stage == 0:
            max_btn = screen.find_any_text(["MAX", "最大"], min_conf=0.6)
            if max_btn:
                self._sweep_stage = 1
                return action_click_box(max_btn, "set drill sweep max")
            # If no max button, continue anyway.
            self._sweep_stage = 1
            return action_wait(250, "drill sweep max unavailable")

        if self._sweep_stage == 1:
            start = screen.find_any_text(
                ["掃蕩開始", "扫荡开始", "開始", "开始", "Confirm"],
                region=(0.45, 0.35, 1.0, 0.85),
                min_conf=0.5,
            )
            if start:
                self._sweep_stage = 2
                return action_click_box(start, "start drill sweep")
            return action_wait(250, "looking for drill sweep start")

        if self._sweep_stage == 2:
            confirm = screen.find_any_text(
                ["確認", "确认", "確定", "确定", "確", "确", "OK"],
                region=(0.30, 0.30, 0.80, 0.92),
                min_conf=0.55,
            )
            if confirm:
                self._sweep_stage = 3
                return action_click_box(confirm, "confirm drill sweep")
            sweep_done = screen.find_any_text(["掃蕩完成", "扫荡完成"], min_conf=0.55)
            if sweep_done:
                self._sweep_stage = 3
                return action_wait(200, "drill sweep completed")
            return action_wait(300, "waiting drill sweep confirm")

        # Stage 3: finalize and return to ticket check.
        done = screen.find_any_text(
            ["掃蕩完成", "扫荡完成", "獲得獎勵", "获得奖励", "確認", "确认", "OK"],
            min_conf=0.55,
        )
        if done:
            if self._tickets_remaining > 0:
                self._tickets_remaining -= 1
            self.sub_state = "check_tickets"
            self._ticket_ticks = 0
            return action_click_box(done, "close drill sweep result")

        if self._sweep_ticks > 24:
            if self._tickets_remaining > 0:
                self._tickets_remaining -= 1
            self.sub_state = "check_tickets"
            self._ticket_ticks = 0
            return action_wait(250, "drill sweep timeout fallback")

        return action_wait(350, "drill sweep running")

    def _fight(self, screen: ScreenState) -> Dict[str, Any]:
        self._fight_ticks += 1

        # Stage 0: ensure battle starts.
        if self._fight_stage == 0:
            start = screen.find_any_text(
                ["出擊", "出击", "開始作戰", "开始作战", "戰鬥開始", "战斗开始"],
                region=(0.45, 0.55, 1.0, 0.95),
                min_conf=0.5,
            )
            if start:
                self._fight_stage = 1
                return action_click_box(start, "confirm drill fight start")
            # Could already be in battle.
            if screen.find_any_text(["SKIP", "Skip", "AUTO", "自動", "自动"], min_conf=0.6):
                self._fight_stage = 1
                return action_wait(200, "drill battle in progress")

        # Stage 1: try skip/auto during battle.
        skip = screen.find_any_text(["SKIP", "Skip", "跳過", "跳过"], min_conf=0.7)
        if skip:
            return action_click_box(skip, "skip drill fight")

        auto = screen.find_any_text(["AUTO", "自動", "自动"], region=(0.68, 0.0, 1.0, 0.22), min_conf=0.55)
        if auto and self._fight_ticks < 8:
            return action_click_box(auto, "enable auto in drill fight")

        result = screen.find_any_text(
            ["戰鬥結果", "战斗结果", "VICTORY", "DEFEAT", "勝利", "胜利", "敗北", "败北"],
            min_conf=0.55,
        )
        if result:
            confirm = screen.find_any_text(
                ["確認", "确认", "確定", "确定", "確", "确", "OK"],
                min_conf=0.55,
            )
            if confirm:
                self._fights_done += 1
                if self._tickets_remaining > 0:
                    self._tickets_remaining -= 1
                self.sub_state = "check_tickets"
                self._ticket_ticks = 0
                self._fight_stage = 0
                self._fight_ticks = 0
                self.log(f"drill fight complete (done={self._fights_done})")
                return action_click_box(confirm, "dismiss drill battle result")
            return action_click(0.5, 0.9, "dismiss drill battle result fallback")

        if self._fight_ticks > 45:
            # Defensive fallback: assume battle ended but OCR missed result.
            if self._tickets_remaining > 0:
                self._tickets_remaining -= 1
            self.sub_state = "check_tickets"
            self._ticket_ticks = 0
            self._fight_stage = 0
            self._fight_ticks = 0
            return action_back("drill fight timeout fallback")

        return action_wait(450, "drill fight running")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if screen.is_lobby():
            self.log(f"done (drill fights={self._fights_done})")
            return action_done("joint firing complete")
        return action_back("joint firing exit: back to lobby")
