"""TotalAssaultSkill: Handle 總力戰/总力战 daily tickets and rewards.

Flow:
1. ENTER: From lobby, open campaign hub and enter total assault.
2. COLLECT_REWARDS: Open info panel and claim season/point rewards.
3. CHECK_TICKETS: Parse remaining tickets (X/Y). Exit if none.
4. SELECT_DIFFICULTY: Choose highest visible difficulty.
5. FIGHT: Enter fight, start battle, skip when available.
6. Loop CHECK_TICKETS until no tickets.
7. EXIT: Return to lobby.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from brain.skills.base import (
    BaseSkill,
    ScreenState,
    action_back,
    action_click,
    action_click_box,
    action_done,
    action_wait,
)


class TotalAssaultSkill(BaseSkill):
    _DIFFICULTIES: List[Tuple[str, List[str]]] = [
        ("TORMENT", ["TORMENT"]),
        ("INSANE", ["INSANE"]),
        ("EXTREME", ["EXTREME"]),
        ("HARDCORE", ["HARDCORE"]),
        ("VERYHARD", ["VERYHARD", "VERY HARD"]),
        ("HARD", ["HARD"]),
        ("NORMAL", ["NORMAL"]),
    ]

    def __init__(self):
        super().__init__("TotalAssault")
        self.max_ticks = 180
        self._enter_ticks: int = 0
        self._reward_ticks: int = 0
        self._reward_step: int = 0
        self._claim_clicks: int = 0
        self._ticket_parse_ticks: int = 0
        self._select_ticks: int = 0
        self._tickets_remaining: int = -1
        self._fights_done: int = 0
        self._fight_stage: int = 0
        self._fight_ticks: int = 0

    def reset(self) -> None:
        super().reset()
        self._enter_ticks = 0
        self._reward_ticks = 0
        self._reward_step = 0
        self._claim_clicks = 0
        self._ticket_parse_ticks = 0
        self._select_ticks = 0
        self._tickets_remaining = -1
        self._fights_done = 0
        self._fight_stage = 0
        self._fight_ticks = 0

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1

        if self.ticks >= self.max_ticks:
            self.log("timeout")
            return action_done("total assault timeout")

        # Result/reward popups can appear in multiple phases.
        reward_popup = screen.find_any_text(
            ["獲得獎勵", "获得奖励", "獲得道具", "获得道具"],
            region=screen.CENTER,
            min_conf=0.6,
        )
        if reward_popup:
            return action_click(0.5, 0.9, "dismiss total assault reward popup")

        battle_result = screen.find_any_text(
            ["戰鬥結果", "战斗结果", "勝利", "胜利", "敗北", "败北", "VICTORY", "DEFEAT"],
            min_conf=0.6,
        )
        if battle_result:
            confirm = screen.find_any_text(
                ["確認", "确认", "確定", "确定", "確", "确", "OK"],
                min_conf=0.6,
            )
            if confirm:
                self._fights_done += 1
                self._fight_stage = 0
                self._fight_ticks = 0
                self._ticket_parse_ticks = 0
                self.sub_state = "check_tickets"
                self.log(f"battle result handled (fights done: {self._fights_done})")
                return action_click_box(confirm, "dismiss total assault result")
            return action_click(0.5, 0.9, "tap to dismiss total assault result")

        no_ticket_popup = screen.find_any_text(
            ["票券不足", "票券不夠", "票券不够", "入場券不足", "入场券不足"],
            min_conf=0.6,
        )
        if no_ticket_popup:
            self.log("ticket popup indicates no tickets left")
            self.sub_state = "exit"
            confirm = screen.find_any_text(
                ["確認", "确认", "確定", "确定", "確", "确", "OK"],
                min_conf=0.6,
            )
            if confirm:
                return action_click_box(confirm, "confirm no-ticket popup")
            return action_back("close no-ticket popup")

        popup = self._handle_common_popups(screen)
        if popup:
            return popup

        if screen.is_loading():
            return action_wait(800, "total assault loading")

        if self.sub_state == "":
            self.sub_state = "enter"

        if self.sub_state == "enter":
            return self._enter(screen)
        if self.sub_state == "collect_rewards":
            return self._collect_rewards(screen)
        if self.sub_state == "check_tickets":
            return self._check_tickets(screen)
        if self.sub_state == "select_difficulty":
            return self._select_difficulty(screen)
        if self.sub_state == "fight":
            return self._fight(screen)
        if self.sub_state == "exit":
            return self._exit(screen)

        return action_wait(300, "total assault unknown state")

    def _is_total_assault_screen(self, screen: ScreenState) -> bool:
        header = screen.find_any_text(
            ["總力戰", "总力战", "大決戰", "大决战"],
            region=(0.0, 0.0, 0.35, 0.14),
            min_conf=0.5,
        )
        if header:
            return True

        # Fallback markers for menu/room screens where header OCR can be weak.
        menu_marker = screen.find_any_text(
            ["TORMENT", "INSANE", "EXTREME", "HARDCORE", "VERYHARD", "排名", "积分"],
            region=(0.15, 0.08, 0.95, 0.92),
            min_conf=0.55,
        )
        if menu_marker:
            return True

        room_marker = screen.find_any_text(
            ["攻擊編制", "攻击编制", "出擊", "出击", "入場券", "入场券", "票券"],
            region=(0.30, 0.10, 1.0, 0.95),
            min_conf=0.55,
        )
        return room_marker is not None

    def _is_total_assault_info(self, screen: ScreenState) -> bool:
        return screen.find_any_text(
            ["總力戰資訊", "总力战信息", "排名", "積分", "积分", "賽季", "赛季"],
            region=(0.0, 0.10, 0.60, 0.75),
            min_conf=0.5,
        ) is not None

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        self._enter_ticks += 1
        current = self.detect_current_screen(screen)

        if self._is_total_assault_screen(screen):
            self.log("inside total assault")
            self.sub_state = "collect_rewards"
            self._reward_ticks = 0
            self._reward_step = 0
            return action_wait(400, "entered total assault")

        if current == "Lobby":
            campaign_btn = screen.find_any_text(
                ["任務", "任务"],
                region=(0.80, 0.70, 1.0, 0.90),
                min_conf=0.6,
            )
            if campaign_btn:
                self.log(f"clicking campaign entry '{campaign_btn.text}'")
                return action_click_box(campaign_btn, "open campaign for total assault")
            return action_click(0.95, 0.83, "open campaign area (hardcoded)")

        if current == "Mission":
            total_assault = screen.find_any_text(
                ["總力戰", "总力战", "大決戰", "大决战", "總力", "总力", "決戰", "决战"],
                min_conf=0.5,
            )
            if total_assault:
                self.log(f"clicking total assault '{total_assault.text}'")
                return action_click_box(total_assault, "enter total assault")
            if self._enter_ticks > 4:
                # reference global icon is near (922,447) on 1280x720.
                return action_click(0.72, 0.62, "enter total assault (hardcoded)")
            return action_wait(400, "looking for total assault entry")

        if current and current not in ("Mission",):
            return action_back(f"back from {current}")

        if self._enter_ticks > 24:
            self.log("failed to reach total assault, exiting")
            self.sub_state = "exit"
            return action_wait(300, "total assault enter timeout")

        return action_wait(500, "entering total assault")

    def _collect_rewards(self, screen: ScreenState) -> Dict[str, Any]:
        self._reward_ticks += 1

        claim_btn = screen.find_any_text(
            ["領取", "领取", "收取", "全部領取", "全部领取", "受取"],
            region=(0.65, 0.65, 1.0, 0.95),
            min_conf=0.55,
        )

        info_open = self._is_total_assault_info(screen)

        if self._reward_ticks > 24:
            self.log(f"reward pass done ({self._claim_clicks} claims)")
            self.sub_state = "check_tickets"
            self._ticket_parse_ticks = 0
            return action_wait(300, "finish reward pass")

        if self._reward_step == 0:
            if info_open:
                self._reward_step = 1
                return action_wait(200, "reward info already open")
            info_btn = screen.find_any_text(
                ["資訊", "信息", "詳情", "详情"],
                region=(0.78, 0.76, 1.0, 0.95),
                min_conf=0.5,
            )
            self._reward_step = 1
            if info_btn:
                return action_click_box(info_btn, "open total assault reward info")
            return action_click(0.92, 0.88, "open total assault reward info (hardcoded)")

        if self._reward_step == 1:
            self._reward_step = 2
            return action_click(0.19, 0.43, "open season reward tab")

        if self._reward_step == 2:
            if claim_btn:
                self._claim_clicks += 1
                self._reward_step = 3
                return action_click_box(claim_btn, "claim season reward")
            self._reward_step = 3
            return action_wait(200, "season reward not available")

        if self._reward_step == 3:
            self._reward_step = 4
            return action_click(0.19, 0.33, "open accumulated reward tab")

        if self._reward_step == 4:
            if claim_btn:
                self._claim_clicks += 1
                self._reward_step = 5
                return action_click_box(claim_btn, "claim accumulated reward")
            self._reward_step = 5
            return action_wait(200, "accumulated reward not available")

        if self._reward_step == 5:
            self._reward_step = 6
            if info_open:
                return action_back("close total assault reward info")

        self.log(f"reward pass done ({self._claim_clicks} claims)")
        self.sub_state = "check_tickets"
        self._ticket_parse_ticks = 0
        return action_wait(300, "reward pass complete")

    def _check_tickets(self, screen: ScreenState) -> Dict[str, Any]:
        self._ticket_parse_ticks += 1

        for box in screen.ocr_boxes:
            if box.confidence < 0.5:
                continue
            match = re.search(r"(\d+)\s*/\s*(\d+)", box.text)
            if not match:
                continue
            remaining = int(match.group(1))
            total = int(match.group(2))
            if total <= 0 or total > 5:
                continue

            text = box.text
            is_ticket_row = (
                "票券" in text
                or "入場券" in text
                or "入场券" in text
                or "剩餘" in text
                or "剩余" in text
                or (box.cx > 0.62 and box.cy < 0.24)
            )
            if not is_ticket_row:
                continue

            self._tickets_remaining = remaining
            self.log(f"total assault tickets: {remaining}/{total}")
            if remaining <= 0:
                self.sub_state = "exit"
                return action_wait(300, "no total assault tickets")

            self.sub_state = "select_difficulty"
            self._select_ticks = 0
            return action_wait(250, "tickets available")

        if self._ticket_parse_ticks > 10:
            self.log("ticket parse timeout, trying fight flow anyway")
            self.sub_state = "select_difficulty"
            self._select_ticks = 0
            return action_wait(300, "ticket parse timeout")

        return action_wait(450, "checking total assault tickets")

    def _select_difficulty(self, screen: ScreenState) -> Dict[str, Any]:
        self._select_ticks += 1

        for name, aliases in self._DIFFICULTIES:
            for alias in aliases:
                hit = screen.find_text_one(
                    alias,
                    region=(0.45, 0.12, 0.92, 0.90),
                    min_conf=0.55,
                )
                if hit:
                    self.log(f"selecting difficulty {name}")
                    self.sub_state = "fight"
                    self._fight_stage = 0
                    self._fight_ticks = 0
                    return action_click_box(hit, f"select {name}")

        if self._select_ticks > 2:
            self.log("difficulty OCR miss, using hardcoded row")
            self.sub_state = "fight"
            self._fight_stage = 0
            self._fight_ticks = 0
            return action_click(0.90, 0.30, "select difficulty (hardcoded)")

        return action_wait(400, "looking for total assault difficulty")

    def _fight(self, screen: ScreenState) -> Dict[str, Any]:
        self._fight_ticks += 1

        if self._fight_ticks > 70:
            self.log(f"fight timeout at stage {self._fight_stage}, retrying ticket check")
            self.sub_state = "check_tickets"
            self._ticket_parse_ticks = 0
            self._fight_stage = 0
            self._fight_ticks = 0
            return action_wait(400, "fight timeout")

        if self._fight_stage == 0:
            lineup = screen.find_any_text(
                ["攻擊編制", "攻击编制", "攻擎编制", "攻擎編制", "編制", "编制", "入場", "入场", "挑戰", "挑战"],
                region=(0.35, 0.20, 1.0, 0.95),
                min_conf=0.5,
            )
            if lineup:
                self._fight_stage = 1
                return action_click_box(lineup, "open total assault lineup")

            if self._fight_ticks > 8:
                self.sub_state = "select_difficulty"
                self._select_ticks = 0
                return action_wait(300, "lineup button not found, reselecting")

            return action_wait(450, "waiting for lineup button")

        if self._fight_stage == 1:
            sortie = screen.find_any_text(
                ["出擊", "出击", "開始作戰", "开始作战", "戰鬥開始", "战斗开始", "進入戰鬥", "进入战斗"],
                region=(0.50, 0.60, 1.0, 0.95),
                min_conf=0.5,
            )
            if sortie:
                self._fight_stage = 2
                return action_click_box(sortie, "start total assault battle")

            # Sometimes lineup opens directly into battle transition.
            skip = screen.find_any_text(["SKIP", "Skip"], min_conf=0.7)
            if skip:
                self._fight_stage = 2
                return action_click_box(skip, "skip total assault battle")

            return action_wait(450, "waiting for start battle button")

        # Stage 2+: battle in progress
        skip = screen.find_any_text(["SKIP", "Skip", "跳過", "跳过"], min_conf=0.7)
        if skip:
            return action_click_box(skip, "skip total assault battle")

        # If we returned to menu without an explicit result popup, continue flow.
        if self._is_total_assault_screen(screen) and self._fight_ticks > 12:
            self.log("returned to total assault menu, re-checking tickets")
            self.sub_state = "check_tickets"
            self._ticket_parse_ticks = 0
            self._fight_stage = 0
            self._fight_ticks = 0
            return action_wait(300, "back to total assault menu")

        return action_wait(1000, "total assault battle in progress")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if screen.is_lobby():
            self.log(f"done ({self._fights_done} fights, {self._claim_clicks} reward claims)")
            return action_done("total assault complete")
        return action_back("total assault exit: back to lobby")
