"""EventActivitySkill: clear event story/challenge flow before AP farming.

This skill focuses on non-sweep event content so the daily pipeline can:
1. Enter currently running event.
2. Clear story/dialogue nodes (with skip/confirm handling).
3. Clear challenge entry once (sweep or battle path).
4. Touch mission tab, then return to lobby.

It is intentionally defensive: if no event is available, it exits quickly.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from brain.skills.base import (
    BaseSkill,
    ScreenState,
    action_back,
    action_click,
    action_click_box,
    action_done,
    action_scroll,
    action_wait,
)

try:
    from vision.event_banner_matcher import (
        BannerMatch,
        EventBannerMatcher,
        get_matcher as _get_banner_matcher,
    )
except Exception:  # pragma: no cover — fall back to OCR-only path
    BannerMatch = None  # type: ignore[assignment]
    EventBannerMatcher = None  # type: ignore[assignment]
    _get_banner_matcher = None  # type: ignore[assignment]

# Per-event persistent progress (see brain/skills/event_progress.py).
# Records which story / mission / challenge nodes have been beaten, so a
# bot restart mid-event can resume without re-running completed nodes —
# inspired by reference's per-activity JSON but with a single generic store
# plus per-event metadata keyed by the event id our banner matcher emits.
from brain.skills.event_progress import (  # noqa: E402
    EventProgressStore,
    get_store as _get_progress_store,
)

_STATE_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "event_story_state.json"

# Template-match score threshold for trusting a banner classification.
# Calibrated from scripts/_test_event_banner_classifier.py: training
# frames score 1.000, genuine-but-unseen lobbies score 0.69-0.99.
_BANNER_MATCH_THRESHOLD = 0.55


class EventActivitySkill(BaseSkill):
    def __init__(self):
        super().__init__("EventActivity")
        self.max_ticks = 300  # battles can take 60+ ticks; multiple story nodes

        self._enter_ticks: int = 0
        self._phase_ticks: int = 0
        # Per-event persistent progress store.  Populated lazily — the
        # event id is not known until the banner matcher identifies the
        # current event inside _enter().  Until then we fall back to the
        # conservative "unknown event" default where every phase reports
        # done immediately and the skill exits gracefully.
        self._progress_store: EventProgressStore = _get_progress_store()
        self._current_event_id: str = ""
        # Set each tick _is_schale_signature fires on the bottom-left region;
        # persists for ~N ticks after so a mid-cycle transition (schale just
        # faded out, next banner not yet rendered) still forbids OCR-fallback
        # clicks. Prevents the "click main event banner (bottom-left OCR)"
        # false-positive that lands on 夏莱 during banner rotation.
        self._schale_recent_tick: int = -100  # last tick index schale was seen
        self._SCHALE_GUARD_TICKS: int = 3    # how long the guard persists

        self._story_done: bool = False
        self._story_tab_clicked: bool = False
        self._story_idle_ticks: int = 0
        self._skip_stage: int = 0  # 0=not skipping, 1=clicked MENU, 2=clicked Skip, 3=done
        self._cutscene_taps: int = 0  # how many times we tapped to advance cutscene

        self._challenge_done: bool = False
        self._challenge_tab_clicked: bool = False
        self._challenge_idle_ticks: int = 0
        self._challenge_sweep_stage: int = 0
        self._challenge_stage_index: int = 0  # next challenge entry button to try (0-based)
        self._challenge_completed_count: int = 0  # consecutive already-completed stages
        self._grid_auto_enabled: bool = False
        self._battle_speed_set: bool = False  # clicked speed/auto buttons this battle?
        self._reward_fallback_streak: int = 0  # consecutive fallback clicks on post-battle screens

        # template-based sequential story processing: track by node index (01, 02, ...)
        self._current_story_index: int = 1   # next node to process (1-based)
        self._story_scroll_count: int = 0    # consecutive scrolls without finding target
        self._max_story_index_seen: int = 0  # highest node number observed on screen
        self._story_node_pending: bool = False  # True after clicking 入場, awaiting popup
        self._story_ap_depleted: bool = False  # True when AP purchase popup seen
        self._story_completed_streak: int = 0  # consecutive completed chapters detected

        self._mission_done: bool = False
        self._mission_tab_clicked: bool = False
        self._no_game_ticks: int = 0  # consecutive ticks with no game UI

        # Quest (mission) phase sequential processing
        self._quest_current_index: int = 1
        self._quest_scroll_count: int = 0
        self._quest_node_pending: bool = False
        self._quest_idle_ticks: int = 0
        self._quest_consecutive_locks: int = 0   # stop after 2 in a row
        self._story_consecutive_locks: int = 0   # same, story phase

        # Template-match event classifier (template-based compare_image).
        # Falls back to pure-OCR when templates are missing.
        self._banner_matcher: Optional["EventBannerMatcher"] = None
        if _get_banner_matcher is not None:
            try:
                self._banner_matcher = _get_banner_matcher()
            except Exception:
                self._banner_matcher = None

    def reset(self) -> None:
        super().reset()
        self._enter_ticks = 0
        self._phase_ticks = 0

        self._story_done = False
        self._story_tab_clicked = False
        self._story_idle_ticks = 0
        self._skip_stage = 0
        self._cutscene_taps = 0

        self._challenge_done = False
        self._challenge_tab_clicked = False
        self._challenge_idle_ticks = 0
        self._challenge_sweep_stage = 0
        self._challenge_stage_index = 0
        self._challenge_completed_count = 0
        self._grid_auto_enabled = False
        self._battle_speed_set = False
        self._reward_fallback_streak = 0

        # Event id is set inside _enter() once the banner matcher
        # identifies the current event.  Until that happens the store
        # returns zeroed metadata and all helper calls are no-ops — we
        # keep running on the old in-memory counters as a safety net.
        self._current_event_id = ""

        # Seed the local counters from whatever progress the store has
        # persisted for the last-seen event.  If no event was played
        # before, or this is a fresh event rotation, these default to 1.
        self._current_story_index = 1
        self._quest_current_index = 1
        self._story_done = False
        self._challenge_done = False
        self._mission_done = False

        self._story_scroll_count = 0
        self._max_story_index_seen = 0
        self._story_node_pending = False
        self._story_ap_depleted = False
        self._story_completed_streak = 0
        self._no_game_ticks = 0

        self._mission_tab_clicked = False
        self._quest_scroll_count = 0
        self._quest_node_pending = False
        self._quest_idle_ticks = 0
        self._quest_consecutive_locks = 0
        self._story_consecutive_locks = 0

    # ── Per-event persistence (backed by brain/skills/event_progress) ──
    #
    # The store is keyed by ``event_id`` (e.g. ``"serenade_promenade"``)
    # and tracks, per phase, the SET of node indices that have been
    # beaten.  Derived views:
    #
    #   _phase_done(phase)  →  True iff every node 1..total is beaten
    #   _next_node(phase)   →  smallest unfinished node (>=1), or
    #                          total+1 when the phase is fully done
    #   _mark_node_done()   →  append a node to the completed set and
    #                          fsync the store so a crash between
    #                          nodes still preserves progress
    #
    # If no event id has been detected yet (template matcher hasn't
    # fired), every accessor is a defensive no-op / conservative
    # default so the skill keeps working on the old counters.

    def _event_total(self, phase: str) -> int:
        if not self._current_event_id:
            return 0
        return self._progress_store.metadata(self._current_event_id).total_for(phase)

    def _phase_done(self, phase: str) -> bool:
        if not self._current_event_id:
            return False
        return self._progress_store.phase_done(self._current_event_id, phase)

    def _next_node(self, phase: str) -> int:
        """First unfinished node index for this phase, or ``total + 1``.

        Returns 1 when no event id has been detected yet so the skill's
        existing "start from node 1" behaviour is preserved.
        """
        if not self._current_event_id:
            return 1
        nxt = self._progress_store.next_node(self._current_event_id, phase)
        if nxt is not None:
            return nxt
        return max(1, self._event_total(phase) + 1)

    def _mark_node_done(self, phase: str, node: int) -> None:
        if not self._current_event_id or node < 1:
            return
        self._progress_store.mark_done(self._current_event_id, phase, node)

    def _set_current_event(self, event_id: str) -> None:
        """Called from _enter when banner match identifies the event.

        Resyncs local counters from the store so we resume at the right
        node after a restart.  Idempotent — only does real work the
        first time we see a new event id or when switching events.
        """
        if not event_id or event_id == self._current_event_id:
            return
        self._current_event_id = event_id
        self._current_story_index = self._next_node("story")
        self._quest_current_index = self._next_node("mission")
        # `_story_done` / `_mission_done` / `_challenge_done` remain as
        # in-memory caches but we now derive them from the store so a
        # timeout can't prematurely declare a phase complete.
        self._story_done = self._phase_done("story")
        self._mission_done = self._phase_done("mission")
        self._challenge_done = self._phase_done("challenge")
        meta = self._progress_store.metadata(event_id)
        summary = self._progress_store.summary(event_id)
        self.log(
            f"event identified: {event_id} ({meta.display_name or 'no-metadata'}) "
            f"progress story={summary['story']} mission={summary['mission']} "
            f"challenge={summary['challenge']} — "
            f"story@{self._current_story_index} quest@{self._quest_current_index}"
        )

    # Legacy methods kept as thin shims so the rest of the (large)
    # skill body doesn't need to learn the new API in a single diff.
    # These now write through to the store AND to the legacy JSON file
    # so external tooling (dashboards, unit tests) keeps working.

    @staticmethod
    def _load_state() -> dict:
        try:
            if _STATE_FILE.exists():
                return json.loads(_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {}

    def _save_state(self) -> None:
        try:
            _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            _STATE_FILE.write_text(
                json.dumps({
                    "event_id": self._current_event_id,
                    "current_story_index": self._current_story_index,
                    "current_quest_index": self._quest_current_index,
                    "story_done": self._phase_done("story"),
                    "mission_done": self._phase_done("mission"),
                    "challenge_done": self._phase_done("challenge"),
                }, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _save_story_index(self) -> None:
        self._save_state()

    def _clear_story_state(self) -> None:
        self._save_state()

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1

        if self.ticks >= self.max_ticks:
            self.log("timeout")
            return action_done("event activity timeout")

        # Frequent reward/result popups across story/challenge/battle.
        # "Battle Complete" appears as header on post-battle screens.
        # OCR sometimes misreads "獲得獎勵" as "獲得奖！" — include short prefixes.
        reward_popup = screen.find_any_text(
            [
                "獲得獎勵", "获得奖励", "獲得道具", "获得道具",
                "獲得奖", "獲得獎",
                "戰鬥結果", "战斗结果", "掃蕩完成", "扫荡完成",
                "任務完成", "任务完成", "關卡完成", "关卡完成",
                "Battle Complete", "BattleComplete",
                "VICTORY", "DEFEAT", "勝利", "敗北", "胜利", "败北",
            ],
            min_conf=0.6,
        )
        if reward_popup:
            self._battle_speed_set = False  # battle ended, reset for next
            # Expanded region: BA's 確認 button sits at the far bottom-right
            # (~x=0.95 on Battle Complete screens); the old cutoff at x=0.80
            # missed it entirely. Keep y lower bound loose for reward dialogs
            # whose button is centered higher.
            confirm = screen.find_any_text(
                ["確認", "确认", "確定", "确定", "確", "确", "OK", "NEXT", "Next"],
                region=(0.25, 0.78, 0.995, 0.99),
                min_conf=0.55,
            )
            if confirm:
                self._reward_fallback_streak = 0
                return action_click_box(confirm, "dismiss event popup")
            # Fallback: reference uses (1168, 659) ≈ (0.913, 0.915) on
            # normal_task_fight-confirm. That's consistently where BA's
            # bottom-right confirm button lives across post-battle summaries,
            # quest-complete, and reward-acquired screens.
            self._reward_fallback_streak = getattr(self, "_reward_fallback_streak", 0) + 1
            # If we've clicked the fallback 6+ times and OCR still shows the
            # same reward-complete screen, the click is landing on dead space
            # or a partially-covered modal — press BACK once to unstick, then
            # reset the streak counter.
            if self._reward_fallback_streak >= 6:
                self._reward_fallback_streak = 0
                self.log("reward popup fallback stuck 6x, pressing BACK to unstick")
                return action_back("reward popup fallback stuck, recover via back")
            return action_click(0.913, 0.915, "dismiss event popup fallback")

        popup = self._handle_common_popups(screen)
        if popup:
            return popup

        if screen.is_loading():
            return action_wait(800, "event activity loading")

        if self.sub_state == "":
            self.sub_state = "enter"

        if self.sub_state == "enter":
            return self._enter(screen)
        if self.sub_state == "story":
            return self._story(screen)
        if self.sub_state == "challenge":
            return self._challenge(screen)
        if self.sub_state == "mission":
            return self._mission(screen)
        if self.sub_state == "exit":
            return self._exit(screen)

        return action_wait(300, "event activity unknown state")

    def _is_auto_battle(self, screen: ScreenState) -> bool:
        """Detect active auto-battle screen via HUD elements.

        Battle HUD has AUTO at bottom-right (~0.95, 0.94) and COST at
        bottom-center (~0.63, 0.92).  OCR count is typically 8-10, which
        is above the generic <=5 loading threshold.
        """
        auto = screen.find_any_text(
            ["AUTO"],
            region=(0.85, 0.85, 1.0, 1.0),
            min_conf=0.8,
        )
        if not auto:
            return False
        cost = screen.find_any_text(
            ["COST"],
            region=(0.55, 0.85, 0.70, 0.97),
            min_conf=0.8,
        )
        return cost is not None

    def _handle_battle_speed(self, screen: ScreenState) -> Optional[Dict[str, Any]]:
        """On first auto-battle detection, click speed button to set 3x.

        reference: speed button at (1215/1280, 625/720) = (0.949, 0.868)
              auto  button at (1215/1280, 677/720) = (0.949, 0.940)
        Speed cycles 1x→2x→3x. Click twice to reach 3x from 1x.
        Returns an action if speed needs setting, None if already done.
        """
        if self._battle_speed_set:
            return None
        self._battle_speed_set = True
        # Click speed button area twice (1x→2x→3x), then auto is already on
        return action_click(0.949, 0.868, "set battle speed to 3x")

    def _find_event_timer(self, screen: ScreenState, *, region) -> Optional[Any]:
        # OCR frequently mixes Traditional/Simplified (e.g. "距離结束還剩").
        # Use regex with character classes to match all script combinations.
        return screen.find_any_text(
            [
                "[結结]束[還还]剩",       # core pattern — covers all Trad/Simp mixes
                "距[離离][結结]束",        # prefix variant
                "[結结]束[選遗道]剩",      # OCR misread variants (選/遗/道 for 還)
                "[結结]束剩",             # short fallback
            ],
            region=region,
            min_conf=0.3,
        )

    # Region where the lobby's big main-event banner lives.
    _BOTTOM_LEFT_BANNER_REGION = (0.0, 0.60, 0.32, 0.92)

    def _find_main_event_banner(self, screen: ScreenState) -> Optional[Any]:
        """Detect the big bottom-left event banner (the actual entry to the
        current main event — e.g. Serenade Promenade), as opposed to the
        small cycling widget in the top-right.

        BA paints a BIG banner card on the bottom-left of the lobby for
        the current main event.  Its distinctive markers are:
          * "EVENT!!" / "EVENT!" ribbon on the upper-left corner
          * "復刻" / "复刻" / "Re!" ribbon for rerun events
          * an "活動進行中" / "活动进行中" label at the bottom-right
            of the card
        The banner ALSO cycles between multiple active events, so even
        though we detect the EVENT!! / 活動進行中 markers we must also
        verify the card doesn't currently show 夏莱 (聯邦學生會) — in
        that case return None and let the caller wait for the rotation.
        """
        # Reject the whole region if it currently shows 夏莱.
        if self._is_schale_signature(
            screen, region=self._BOTTOM_LEFT_BANNER_REGION
        ):
            return None

        # Try the loud banner labels first — very distinctive English text.
        label = screen.find_any_text(
            ["EVENT!!", "EVENT!", "EVENT", "復刻", "复刻", "Re!", "Re!!", "NEW!"],
            region=self._BOTTOM_LEFT_BANNER_REGION,
            min_conf=0.55,
        )
        if label:
            return label
        # Fallback: "活動進行中" tag at the card bottom.
        return screen.find_any_text(
            ["活動進行中", "活动进行中", "活動進行", "活动进行"],
            region=(0.0, 0.70, 0.32, 0.95),
            min_conf=0.55,
        )

    def _find_campaign_event_tile(self, screen: ScreenState) -> Optional[Any]:
        """Detect the main-event tile on the Campaign/Mission hub.

        On that screen the layout is:
          * TOP-LEFT small banner (x<0.30, y<0.25)     = 夏莱 grind banner
          * 任務 / 劇情 big tiles                     = main campaign entry
          * Small tile grid at y≈0.50–0.90 with a red "活動進行中" ribbon
            marking the current main event (e.g. 學園交流會 for
            Serenade Promenade).

        We look for the 活動進行中 ribbon in the tile-grid region —
        outside the 夏莱 banner area — so the click lands on the real
        event entry.
        """
        # Primary signal: the red ribbon
        ribbon = screen.find_any_text(
            ["活動進行中", "活动进行中", "活動進行", "活动进行"],
            region=(0.30, 0.65, 0.98, 0.92),
            min_conf=0.55,
        )
        if ribbon:
            return ribbon
        # Secondary: known event-tile titles
        return screen.find_any_text(
            ["學園交流會", "学园交流会", "學園交流", "学园交流"],
            region=(0.30, 0.65, 0.98, 0.92),
            min_conf=0.55,
        )

    # Map legacy OCR-region tuples used by existing call sites to the
    # matcher's position key.  Call sites pass either the bottom-left
    # banner region (x1<0.05) or the top-right widget region (x1>0.5).
    @staticmethod
    def _region_to_banner_position(
        region: tuple,
    ) -> Optional[str]:
        try:
            x1 = float(region[0])
        except Exception:
            return None
        if x1 < 0.30:
            return "bottom_left"
        if x1 > 0.50:
            return "top_right"
        return None

    def _classify_banner(
        self, screen: ScreenState, position: str
    ) -> Optional["BannerMatch"]:
        """Return the matcher's best event classification for a given
        banner position, or None if the matcher / frame is unavailable."""
        if self._banner_matcher is None:
            return None
        frame = screen.load_image()
        if frame is None:
            return None
        try:
            return self._banner_matcher.classify(frame, position)
        except Exception:
            return None

    def _is_schale_signature(
        self,
        screen: ScreenState,
        *,
        region: tuple = (0.55, 0.0, 1.0, 0.35),
    ) -> bool:
        """Return True if the given region shows the 夏莱 總結算
        (Schale Final Settlement) event.

        夏莱 is essentially a daily-task / special-mission grind —
        the user asked us to skip the event-activity skill when 夏莱
        is the only option, so `daily_tasks` / `campaign_push` can
        handle its sub-objectives instead of this skill wasting ticks
        inside its UI.

        Primary signal: template-based template matching
        (``EventBannerMatcher``) against pre-cropped banner PNGs in
        ``data/event_banners/``.  This beats OCR on stylised titles
        where "夏萊" or "總結算" can get mangled.

        Fallback: the legacy OCR text-search, used when templates are
        missing or the classifier is inconclusive.
        """
        position = self._region_to_banner_position(region)
        if position is not None:
            match = self._classify_banner(screen, position)
            if match is not None and match.score >= _BANNER_MATCH_THRESHOLD:
                return match.is_schale

        return bool(screen.find_any_text(
            ["夏萊", "夏莱", "聯邦學生會", "联邦学生会",
             "聯邦學生", "联邦学生",
             "總結算", "总结算", "總結", "总结",
             "聯邦", "联邦"],
            region=region,
            min_conf=0.45,
        ))

    def _has_non_schale_event_banner(
        self, screen: ScreenState, position: str
    ) -> Optional["BannerMatch"]:
        """Return the BannerMatch when the position shows a KNOWN
        non-schale event with high confidence; None otherwise.

        Used by the Lobby branch to decide whether to click a banner
        immediately instead of waiting for the cycle.
        """
        match = self._classify_banner(screen, position)
        if match is None:
            return None
        if match.score < _BANNER_MATCH_THRESHOLD:
            return None
        if not match.is_known_event or match.is_schale:
            return None
        return match

    def _find_reward_claim_timer(self, screen: ScreenState, *, region) -> Optional[Any]:
        """Detect expired event reward-claim banners (not current event)."""
        return screen.find_any_text(
            [
                "距離獎勵", "距离奖励", "獎勵領取", "奖励领取",
                "獎勵獲得結束", "奖励获得结束",
                "距離獎勵獲得結束", "距离奖励获得结束",
                "距離獎勵領取結束", "距离奖励领取结束",
                "獎勵結束", "奖励结束",
            ],
            region=region,
            min_conf=0.3,
        )

    def _is_event_page(self, screen: ScreenState) -> bool:
        event_header = screen.find_any_text(
            ["活動", "活动",
             # Named event headers — the campaign tile opens a page
             # whose header shows the event title directly, not "活動".
             "學園交流會", "学园交流会", "學園交流", "学园交流",
             "交流會", "交流会"],
            region=(0.0, 0.0, 0.22, 0.10),
            min_conf=0.55,
        )
        if event_header:
            return True

        tabs = screen.find_any_text(
            ["Story", "Quest", "Challenge", "劇情", "剧情", "任務", "任务"],
            region=(0.50, 0.0, 1.0, 0.24),
            min_conf=0.55,
        )
        if tabs:
            return True

        item_method = screen.find_any_text(
            ["道具獲得方法", "道具获得方法"],
            min_conf=0.55,
        )
        return item_method is not None

    def _has_game_ui(self, screen: ScreenState) -> bool:
        """Quick check: does the screen contain ANY game-related OCR?

        Detects event page elements, battle HUD, common game buttons.
        Returns False when DXcam is capturing a non-game window (browser, etc).
        """
        game_keywords = [
            "活動", "活动", "Story", "Quest", "Challenge",
            "入場", "入场", "出擊", "出击", "AUTO", "MENU",
            "任務", "任务", "章節", "章节", "確認", "确认",
            "SKIP", "Skip", "COST", "掃蕩", "扫荡",
            "劇情", "商店", "配對", "資訊",
        ]
        for box in screen.ocr_boxes:
            if box.confidence < 0.5:
                continue
            for kw in game_keywords:
                if kw in box.text:
                    return True
        return False

    def _check_chapter_completion(self, screen: ScreenState) -> Optional[bool]:
        """Check if a story chapter is already completed.

        Uses OCR text in the 章節資訊 dialog:
          - "獲得期待" (Expected Rewards) → uncompleted (return False)
          - "已獲得" (Already Obtained)   → completed (return True)
          - Neither found                → unknown (return None)

        Also logs 進入章節 button color for future pixel-based detection.
        """
        # Primary: check for uncompleted indicator
        expected_reward = screen.find_any_text(
            ["獲得期待", "获得期待"],
            region=(0.30, 0.30, 0.65, 0.50),
            min_conf=0.50,
        )
        if expected_reward:
            self.log(f"chapter reward text: '{expected_reward.text}' → uncompleted")
            return False  # NOT completed

        # Check for completed indicator
        already_obtained = screen.find_any_text(
            ["已獲得", "已获得", "已領取", "已领取"],
            region=(0.30, 0.30, 0.65, 0.50),
            min_conf=0.50,
        )
        if already_obtained:
            self.log(f"chapter reward text: '{already_obtained.text}' → completed")
            return True  # completed

        # Log 進入章節 button pixel color for future analysis
        try:
            import cv2 as _cv2
            import numpy as _np
            img = _cv2.imdecode(
                _np.fromfile(screen.screenshot_path, dtype=_np.uint8),
                _cv2.IMREAD_COLOR,
            )
            if img is not None:
                h, w = img.shape[:2]
                # 進入章節 button center
                bx, by = int(0.50 * w), int(0.72 * h)
                patch = img[max(0, by - 2):min(h, by + 3),
                            max(0, bx - 2):min(w, bx + 3)]
                if patch.size > 0:
                    b, g, r = patch.mean(axis=(0, 1))
                    self.log(f"chapter btn pixel: R={r:.0f} G={g:.0f} B={b:.0f}")
        except Exception:
            pass

        return None  # unknown

    def _is_event_story_screen(self, screen: ScreenState) -> bool:
        auto = screen.find_any_text(
            ["AUTO"],
            region=(0.72, 0.0, 0.92, 0.14),
            min_conf=0.65,
        )
        menu = screen.find_any_text(
            ["MENU"],
            region=(0.82, 0.0, 1.0, 0.14),
            min_conf=0.65,
        )
        skip = screen.find_any_text(["SKIP", "Skip", "跳過", "跳过"], min_conf=0.65)
        return bool((auto and menu) or skip)

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        # Safety net: very few OCR boxes (≤3) typically means loading /
        # transition screen.  Don't count toward enter timeout so a long
        # game update doesn't prematurely skip the skill.
        if len(screen.ocr_boxes) <= 3:
            return action_wait(800, "enter: transition screen (few OCR)")
        self._enter_ticks += 1

        if self._is_event_page(screen):
            self._phase_ticks = 0
            # Phase dispatch follows reference order (story → quest → challenge)
            # but uses the persistent store as the source of truth so a
            # timeout inside a phase can't prematurely skip ahead.  Each
            # _phase_done(x) returns True iff every node 1..total for
            # phase x is in the store's completed set for the current
            # event id (unknown-to-metadata events zero-total to True).
            #
            # The legacy in-memory *_done booleans are SESSION-ABORT
            # signals — set when AP depletes, a 4-completion streak
            # fires, or a phase hits its hard timeout.  Rule: if the
            # store says the phase is still incomplete but the session
            # flag says "we gave up", we exit the skill WITHOUT moving
            # on to the next phase.  User invariant: story 1..N must
            # actually finish before any quest node is attempted.
            story_ok = self._phase_done("story")
            if not story_ok:
                if self._story_done:
                    self.log(
                        f"story incomplete ({self._next_node('story') - 1}/"
                        f"{self._event_total('story')}) but session aborted "
                        f"— exiting without touching quest"
                    )
                    self.sub_state = "exit"
                    return action_wait(200, "story incomplete, session aborted")
                self.sub_state = "story"
                self.log(
                    f"event page ready -> story phase "
                    f"(next node {self._next_node('story'):02d}/"
                    f"{self._event_total('story')})"
                )
                return action_wait(250, "start event story phase")
            self.log("story phase fully done, checking mission")

            mission_ok = self._phase_done("mission")
            if not mission_ok:
                if self._mission_done:
                    self.log(
                        f"mission incomplete ({self._next_node('mission') - 1}/"
                        f"{self._event_total('mission')}) but session aborted "
                        f"— exiting without touching challenge"
                    )
                    self.sub_state = "exit"
                    return action_wait(200, "mission incomplete, session aborted")
                self.sub_state = "mission"
                self.log(
                    f"story done -> mission phase "
                    f"(next node {self._next_node('mission'):02d}/"
                    f"{self._event_total('mission')})"
                )
                return action_wait(250, "start event mission phase")
            self.log("mission phase fully done, checking challenge")

            if not self._phase_done("challenge") and not self._challenge_done:
                self.sub_state = "challenge"
                self.log("mission done -> challenge phase")
                return action_wait(250, "start event challenge phase")
            self.log("challenge phase already done")
            self.sub_state = "exit"
            return action_wait(200, "event phases complete")

        if self._is_event_story_screen(screen):
            if self._story_done:
                # Story marked done but still on cutscene — skip via MENU→Skip
                self.log("cutscene after story done, skipping via MENU")
                self._skip_stage = 0
            self._phase_ticks = 0
            self._story_idle_ticks = 0  # reset to prevent immediate exit
            self.sub_state = "story"
            return action_wait(250, "start event story phase from cutscene")

        current = self.detect_current_screen(screen)

        # Dismiss tutorial / help popups (幫助 modal) — BA shows these
        # on first entry to many event pages.  BACK closes them in BA.
        help_popup = screen.find_any_text(
            ["幫助", "帮助"],
            region=(0.40, 0.10, 0.65, 0.20),
            min_conf=0.55,
        )
        if help_popup:
            self.log(f"help popup detected: '{help_popup.text}', closing with BACK")
            return action_back("close help popup")

        # Back out of 劇情 / 主線劇情 (story menus) — these share the
        # '劇情' hub-marker so detect_current_screen classifies them
        # as "Mission", but they have no event entries.  Without this
        # guard the Mission branch's event_entry fallback matches the
        # 主線劇情 / 短篇劇情 / 支線劇情 panel titles and pings back
        # and forth between the two screens.  Detect the page HEADER
        # (region y<0.10) — body panels sit below y≈0.18 so that band
        # is unique to the top-bar.
        story_header = screen.find_any_text(
            ["主線劇情", "主线剧情", "主線劇", "主线剧"],
            region=(0.0, 0.0, 0.28, 0.10),
            min_conf=0.55,
        )
        if story_header is None:
            # Plain "劇情" header means the Story hub (list of
            # main/short/side story panels).  Disambiguate from the
            # Campaign "任務" hub by checking the header doesn't ALSO
            # contain 任務 — the Campaign hub has 任務 header.
            story_header = screen.find_any_text(
                ["^劇情$", "^剧情$"],
                region=(0.0, 0.0, 0.15, 0.10),
                min_conf=0.6,
            )
        if story_header:
            self.log(
                f"on story-type menu ('{story_header.text}') — dead end "
                f"for event_activity, backing out"
            )
            return action_back("back out of story menu")

        # Generic dead-end: headers that indicate lobby-adjacent screens
        # (Recruitment, Student, Social, Craft, Shop, Cafe, etc.) which
        # detect_current_screen either misses (招募 / Recruitment has no
        # entry in base.py's headers dict) or classifies as something
        # unrelated.  None of these host event entries, so BACK out.
        dead_end = screen.find_any_text(
            ["招募", "Recruit",           # gacha screen
             "機率情報", "机率情报"],    # gacha rate-info tab title
            region=(0.0, 0.0, 0.22, 0.10),
            min_conf=0.55,
        )
        if dead_end:
            self.log(
                f"on dead-end screen ('{dead_end.text}'), backing out"
            )
            return action_back("back out of dead-end screen")

        if current == "Lobby":
            # Check for OLD event reward-claim banner first — skip it
            old_reward = self._find_reward_claim_timer(screen, region=(0.55, 0.0, 1.0, 0.30))
            if old_reward:
                if self._enter_ticks <= 8:
                    return action_wait(1200, "old event reward banner, waiting for rotation")
                self.log("only old event reward banner visible, no current event")
                return action_done("reward-claim event only")

            # ── Proactive schale detection ──
            # Record every tick we see schale on the bottom-left banner so
            # the OCR-fallback below knows we're in a cycling state. Must
            # run BEFORE priority 1b so mid-cycle transition frames are
            # protected even if _is_schale_signature reads False on the
            # very frame where 1b would otherwise fire.
            if self._is_schale_signature(
                screen, region=self._BOTTOM_LEFT_BANNER_REGION
            ):
                self._schale_recent_tick = self._enter_ticks

            # ── PRIORITY 1a: template-match bottom-left non-schale event ──
            # template-based compare_image: if the pre-cropped template for
            # a non-夏莱 event (e.g. Serenade Promenade) lights up, click
            # straight to the card center regardless of whether OCR
            # found the EVENT!! / 活動進行中 label.
            bl_match = self._has_non_schale_event_banner(screen, "bottom_left")
            if bl_match is not None:
                # bottom_left region is (0.01, 0.63, 0.33, 0.84) → center
                self.log(
                    f"bottom-left banner template-matched event "
                    f"'{bl_match.event}' score={bl_match.score:+.3f} "
                    f"-> clicking center"
                )
                # Bind the skill to this event id so per-phase progress
                # is loaded/persisted under the right key before we
                # even enter the event page.  Doing it here (lobby)
                # rather than inside the event page lets _enter use
                # the persisted state to decide whether to dispatch to
                # story / mission / challenge when we land on the hub.
                self._set_current_event(bl_match.event)
                return action_click(
                    0.17, 0.735,
                    f"click main event banner ({bl_match.event} template)",
                )

            # ── PRIORITY 1b: OCR fallback for unseen events ──
            # When we don't have a template for the current event yet,
            # fall back to finding the EVENT!! / 活動進行中 label via OCR.
            # _find_main_event_banner already returns None if the card
            # currently shows 夏莱 (daily-task grind handled elsewhere).
            #
            # NARROW GUARD: Only block OCR-fallback on the SAME tick we
            # observed schale. The banner cycle animation is fast enough
            # that a schale frame is followed by a clear non-schale frame
            # within one tick; blocking multiple ticks used to strand us
            # when a 3-event rotation cycled [schale → unknown event →
            # serenade] and the unknown event had no template to match.
            # The mid-cycle transition is also rare: _find_main_event_banner
            # itself rejects the frame via _is_schale_signature first, and
            # the schale priority-2 branch below catches true schale frames.
            schale_guard_active = (
                self._schale_recent_tick == self._enter_ticks
            )
            if schale_guard_active:
                self.log(
                    "schale on bottom-left this tick — skipping OCR fallback, "
                    "letting priority-2 branch handle the wait"
                )
                # Fall through to PRIORITY 2 below instead of returning a
                # generic wait, so the user-visible reason is accurate.
            main_banner = None if schale_guard_active else self._find_main_event_banner(screen)
            if main_banner:
                # Click the banner image.  Label is at the top-left of
                # the card; shift toward the card center before clicking.
                bx = min(max(main_banner.cx + 0.04, 0.06), 0.22)
                by = min(max(main_banner.cy + 0.10, 0.72), 0.86)
                self.log(
                    f"main event banner visible: '{main_banner.text}' at "
                    f"({main_banner.cx:.2f},{main_banner.cy:.2f}) "
                    f"-> clicking ({bx:.2f},{by:.2f})"
                )
                return action_click(bx, by, "click main event banner (bottom-left OCR)")

            # ── PRIORITY 2: if bottom-left CURRENTLY shows 夏莱, wait ──
            # The bottom-left banner is the most reliable detector for
            # a 夏莱 cycle.  When it shows 夏莱 the top-right widget is
            # typically synced to the same event, so clicking either
            # would land on 夏莱 — wait for the cycle to advance
            # instead.
            if self._is_schale_signature(
                screen, region=self._BOTTOM_LEFT_BANNER_REGION
            ):
                # Record schale sighting so the next N ticks of OCR-fallback
                # (1b) are gated — prevents landing on schale when the banner
                # is mid-cycle.
                self._schale_recent_tick = self._enter_ticks
                if self._enter_ticks <= 12:
                    self.log(
                        "bottom-left banner is 夏莱 grind event, "
                        "waiting for banner cycle to advance"
                    )
                    return action_wait(1500, "schale on bottom-left, waiting for cycle")
                self.log("bottom-left only shows schale after rotation wait, skipping")
                return action_done("schale-only bottom banner, deferred to daily tasks")

            # ── PRIORITY 3: top-right cycling widget, skip if 夏莱 ──
            # Only consulted if the bottom-left card is completely
            # absent (no main event currently available at all).
            timer = self._find_event_timer(screen, region=(0.55, 0.0, 1.0, 0.30))
            if timer and self._is_schale_signature(screen):
                if self._enter_ticks <= 6:
                    self.log(
                        f"top-right banner is 夏莱 grind event "
                        f"('{timer.text}'), waiting for rotation"
                    )
                    return action_wait(1500, "schale on top-right, waiting")
                self.log("only schale visible after rotation wait, skipping")
                return action_done("schale-only event, deferred to daily tasks")

            if timer:
                self.log(f"lobby event timer visible: '{timer.text}', entering event")
                return action_click(timer.cx, min(timer.cy + 0.10, 0.32), "click lobby event banner")

            if 3 <= self._enter_ticks <= 5:
                self.log("probing lobby event banner via hardcoded click")
                return action_click(0.79, 0.17, "probe lobby event banner (hardcoded)")

            campaign_btn = screen.find_any_text(
                ["任務", "任务"],
                region=(0.80, 0.70, 1.0, 0.90),
                min_conf=0.6,
            )
            if campaign_btn:
                return action_click_box(campaign_btn, "open campaign for event")
            if self._enter_ticks > 4:
                return action_click(0.95, 0.83, "open campaign area for event (hardcoded)")
            return action_wait(400, "waiting for event entry from lobby")

        if current == "Mission":
            old_reward = self._find_reward_claim_timer(screen, region=(0.0, 0.04, 0.32, 0.26))
            if old_reward:
                if self._enter_ticks <= 10:
                    return action_wait(1200, "campaign old reward banner, waiting")
                self.log("campaign: only old event reward banner")
                return action_done("campaign reward-claim event only")

            # ── PRIORITY 1: main event tile (活動進行中 ribbon / 學園交流會) ──
            # On the Campaign hub the real current event lives in the
            # tile grid with a red 活動進行中 ribbon.  Prefer it over the
            # top-left 夏莱 banner, which is a daily-task grind that
            # daily_tasks / campaign_push already handle.
            event_tile = self._find_campaign_event_tile(screen)
            if event_tile:
                tx = event_tile.cx
                ty = min(max(event_tile.cy + 0.04, 0.74), 0.88)
                self.log(
                    f"campaign main-event tile: '{event_tile.text}' at "
                    f"({event_tile.cx:.2f},{event_tile.cy:.2f}) "
                    f"-> clicking ({tx:.2f},{ty:.2f})"
                )
                return action_click(tx, ty, "click main event tile (campaign)")

            # ── PRIORITY 2: top-left 夏莱 timer — only if no main tile ──
            # If the ONLY event marker on the hub is the 夏莱 banner,
            # give it a few ticks (main tile may appear after a panel
            # load animation) and then skip so daily_tasks handles 夏莱.
            timer = self._find_event_timer(screen, region=(0.0, 0.04, 0.32, 0.26))
            if timer and self._is_schale_signature(screen):
                if self._enter_ticks <= 4:
                    self.log(
                        f"campaign top-left banner is 夏莱 grind "
                        f"('{timer.text}'), waiting for main tile to render"
                    )
                    return action_wait(1000, "schale-only campaign, waiting for main tile")
                self.log("campaign: only 夏莱 banner, skipping")
                return action_done("schale-only campaign, deferred to daily tasks")

            if timer:
                self.log(f"campaign event timer visible: '{timer.text}', opening event")
                return action_click(timer.cx, min(timer.cy + 0.08, 0.35), "click campaign event banner")

            if self._enter_ticks > 6 and self._enter_ticks <= 10:
                self.log("probing campaign event banner via hardcoded click")
                return action_click(0.17, 0.17, "probe campaign event banner (hardcoded)")

            event_entry = screen.find_any_text(
                ["活動", "活动", "Story", "Quest", "Challenge"],
                region=(0.0, 0.04, 0.40, 0.36),
                min_conf=0.5,
            )
            if event_entry:
                return action_click_box(event_entry, "enter event from campaign")

            if self._enter_ticks > 10:
                self.log("no event entry found in campaign")
                return action_done("event not available")
            return action_wait(450, "looking for event entry in campaign")

        if current == "DailyTasks":
            return action_back("back from DailyTasks to enter event")

        if current and current not in ("Lobby", "Mission", "Event"):
            return action_back(f"back from {current} to find event")

        # Detect WebView / external pages (Discord invite, announcement, etc.)
        # These open when the bot accidentally clicks embedded links.
        webview = screen.find_any_text(
            ["discord.com", "Discord", "接受邀请", "Webpage",
             "Official Twitter", "Official Forum", "主要消息",
             "My Office", "Update"],
            min_conf=0.7,
        )
        if webview:
            self.log(f"WebView/external page detected: '{webview.text}', pressing back")
            return action_back("interceptor: close external WebView")

        if self._enter_ticks > 24:
            self.log("event not found, skipping event activity")
            return action_done("event unavailable")

        return action_wait(500, "entering event")

    def _story(self, screen: ScreenState) -> Dict[str, Any]:
        self._phase_ticks += 1

        # Non-game screen guard: if DXcam is capturing a non-game window
        # (browser, etc.), don't proceed blindly through phases.
        if not self._has_game_ui(screen) and len(screen.ocr_boxes) > 5:
            self._no_game_ticks += 1
            if self._no_game_ticks > 15:
                self.log("no game UI detected for 15+ ticks, resetting to enter")
                self._no_game_ticks = 0
                self.sub_state = "enter"
                return action_back("no game UI, pressing back to recover")
            return action_wait(500, "no game UI detected, waiting")
        else:
            self._no_game_ticks = 0

        # Detect expired event — skip all phases immediately
        ended = screen.find_any_text(
            ["已结束", "已結束", "活動期已", "活动期已"],
            min_conf=0.6,
        )
        if ended:
            self.log(f"event ended: '{ended.text}', skipping story")
            self._story_done = True
            self._clear_story_state()
            self._challenge_done = True
            self._mission_done = True
            self._save_state()
            self.sub_state = "exit"
            return action_wait(200, "event ended, skipping all phases")

        # ── Cutscene / dialog screen (AUTO + MENU visible) ──
        # BA hides Skip behind MENU. Flow: MENU → Skip → Confirm.
        # reference coordinates (1280×720): MENU(1205,34) Skip(1213,116) Confirm(766,520)
        on_cutscene = self._is_event_story_screen(screen)
        if on_cutscene:
            self._story_idle_ticks = 0  # cutscene is NOT idle

            # Direct SKIP button visible (some screens show it directly)
            skip = screen.find_any_text(
                ["SKIP", "Skip", "跳過", "跳过"],
                min_conf=0.65,
            )
            if skip:
                self._skip_stage = 2
                return action_click_box(skip, "click visible SKIP")

            # Skip confirmation dialog (center of screen)
            # OCR often truncates "確認" → "確", so match single char too.
            if self._skip_stage >= 2:
                self._cutscene_taps += 1
                confirm = screen.find_any_text(
                    ["確認", "确认", "確定", "确定", "OK", "Yes", "確", "确"],
                    region=(0.45, 0.55, 0.80, 0.85),
                    min_conf=0.55,
                )
                if confirm:
                    self._skip_stage = 3
                    self._cutscene_taps = 0
                    return action_click_box(confirm, "confirm story skip")
                # Detect "是否略過此劇情" dialog → click confirm area
                skip_prompt = screen.find_any_text(
                    ["略過", "略过", "是否略", "跳過此"],
                    region=(0.30, 0.45, 0.70, 0.70),
                    min_conf=0.55,
                )
                if skip_prompt:
                    self._skip_stage = 3
                    self._cutscene_taps = 0
                    # reference confirm pos: (766/1280, 520/720) = (0.60, 0.72)
                    return action_click(0.60, 0.72, "confirm skip (hardcoded)")
                # Fallback: after 10 ticks reset, after 5 blind-click
                if self._cutscene_taps >= 10:
                    self._cutscene_taps = 0
                    self._skip_stage = 0
                    return action_wait(200, "skip confirm timeout, retrying")
                if self._cutscene_taps >= 5:
                    return action_click(0.60, 0.72, "confirm skip (timeout fallback)")
                return action_wait(300, "waiting for skip confirm dialog")

            # Stage 0: click MENU to reveal skip option
            if self._skip_stage == 0:
                menu = screen.find_any_text(
                    ["MENU"],
                    region=(0.82, 0.0, 1.0, 0.14),
                    min_conf=0.65,
                )
                if menu:
                    self._skip_stage = 1
                    return action_click_box(menu, "click MENU to reveal skip")
                # MENU not found by OCR — try hardcoded position
                self._skip_stage = 1
                return action_click(0.94, 0.05, "click MENU (hardcoded)")

            # Stage 1: MENU was clicked, look for Skip button in dropdown
            if self._skip_stage == 1:
                skip_btn = screen.find_any_text(
                    ["SKIP", "Skip", "跳過", "跳过", "スキップ"],
                    min_conf=0.55,
                )
                if skip_btn:
                    self._skip_stage = 2
                    return action_click_box(skip_btn, "click Skip in menu")
                # Try hardcoded position (reference: 1213/1280, 116/720)
                self._skip_stage = 2
                return action_click(0.95, 0.16, "click Skip (hardcoded)")

            # Fallback: tap center to advance dialog text
            self._cutscene_taps += 1
            if self._cutscene_taps > 40:
                # Stuck on cutscene too long — try pressing back
                self._cutscene_taps = 0
                self._skip_stage = 0
                return action_back("cutscene stuck, pressing back")
            return action_click(0.5, 0.5, "advance event story text")

        # ── Not on cutscene — reset skip state ──
        self._skip_stage = 0
        self._cutscene_taps = 0
        self._story_node_pending = False

        # ── Auto-battle in progress (HUD: AUTO + COST at bottom) ──
        # Battle screen has 8-10 OCR boxes (above <=5 threshold).
        # reference: uses fighting_feature RGB check; we use OCR HUD markers.
        if self._is_auto_battle(screen):
            self._story_idle_ticks = 0
            speed_action = self._handle_battle_speed(screen)
            if speed_action:
                return speed_action
            return action_wait(1500, "story battle in progress (auto)")

        # ── Loading / battle transition ──
        # Very low OCR during loading screens or active battles.
        # Don't count as idle; just wait.
        if len(screen.ocr_boxes) <= 5:
            self._story_idle_ticks = 0
            return action_wait(500, "story loading/battle in progress")

        # ── Chapter Info dialog (story chapter node, not battle) ──
        # Clicking 入場 on a story-only node opens a "章節資訊" dialog
        # with a "進入章節" button to actually enter.
        # reference: check pixel at (362,322)/1280x720 for completion color.
        # We use OCR: uncompleted chapters show "獲得期待" (first-clear reward).
        # Completed chapters don't show this text → close and advance.
        chapter_info = screen.find_any_text(
            ["章節資訊", "章节资讯", "章節資", "章节资"],
            min_conf=0.55,
        )
        if chapter_info:
            self._story_idle_ticks = 0
            self._story_node_pending = False
            # AP depleted — can't enter any chapter, close popup and end story
            if self._story_ap_depleted:
                self.log("AP depleted, closing chapter popup, ending story phase")
                self._story_done = True
                self._clear_story_state()
                self._phase_ticks = 0
                self.sub_state = "enter"
                return action_back("close chapter popup (AP depleted)")

            # Check if chapter is already completed via OCR text detection.
            # "獲得期待" = uncompleted (first-clear rewards available).
            # "已獲得" = completed (rewards already claimed).
            chapter_status = self._check_chapter_completion(screen)
            if chapter_status is True:  # explicitly completed
                self._story_completed_streak += 1
                self.log(f"story node {self._current_story_index:02d} already completed "
                         f"(streak={self._story_completed_streak})")
                # In-game UI confirms this node is clear — persist it
                # so a restart skips straight to the next unfinished one.
                self._mark_node_done("story", self._current_story_index)
                self._current_story_index = max(
                    self._current_story_index + 1, self._next_node("story")
                )
                self._save_story_index()
                self._story_scroll_count = 0
                self._story_tab_clicked = False
                if self._story_completed_streak >= 4:
                    self.log("4+ consecutive completed chapters → story phase done")
                    self._story_done = True
                    self._clear_story_state()
                    self._phase_ticks = 0
                    self.sub_state = "enter"
                    return action_back("story done (all chapters completed)")
                return action_back("close completed chapter info")

            # Chapter NOT completed — enter it
            self._story_completed_streak = 0
            enter_chapter = screen.find_any_text(
                ["進入章節", "进入章节"],
                region=(0.30, 0.55, 0.70, 0.80),
                min_conf=0.55,
            )
            if enter_chapter:
                self.log(f"story node {self._current_story_index:02d} chapter entering")
                # Clicking 進入章節 commits us to this node; mark it
                # done proactively so an interruption between click
                # and reward doesn't strand the index (reference also
                # advances their counter right after the start click).
                self._mark_node_done("story", self._current_story_index)
                self._current_story_index = max(
                    self._current_story_index + 1, self._next_node("story")
                )
                self._save_story_index()
                self._story_scroll_count = 0
                self._story_tab_clicked = False
                return action_click_box(enter_chapter, "click 進入章節 (story chapter)")
            # Fallback: click hardcoded 進入章節 position
            self.log(f"story node {self._current_story_index:02d} chapter entering (hardcoded)")
            self._mark_node_done("story", self._current_story_index)
            self._current_story_index = max(
                self._current_story_index + 1, self._next_node("story")
            )
            self._save_story_index()
            self._story_scroll_count = 0
            self._story_tab_clicked = False
            return action_click(0.50, 0.72, "click 進入章節 (hardcoded)")

        # ── Mission Info dialog (battle story node) ──
        # Screen shows "任務資訊" header with "任務開始" yellow button and
        # a disabled "掃蕩開始" (sweep). Must click "任務開始" specifically.
        # reference: "activity_task-info" → click (940/1280, 538/720) = (0.734, 0.747)
        mission_info = screen.find_any_text(
            ["任務資訊", "任务资讯", "任務資讯", "任务資訊", "任務资讯"],
            min_conf=0.55,
        )
        if mission_info:
            self._story_idle_ticks = 0
            self._story_node_pending = False
            # AP depleted — can't start battle, close popup and end story
            if self._story_ap_depleted:
                self.log("AP depleted, closing mission popup, ending story phase")
                self._story_done = True
                self._clear_story_state()
                self._phase_ticks = 0
                self.sub_state = "enter"
                return action_back("close mission popup (AP depleted)")

            # PRIORITY 1: Start button visible → play the node. Do NOT check
            # "已完成" globally first — the mission popup's sub-objective list
            # shows "已完成" badges per reward which trips the skip branch.
            start_btn = screen.find_any_text(
                ["任務開始", "任务开始", "任務开始", "任务開始"],
                region=(0.55, 0.60, 0.90, 0.85),
                min_conf=0.55,
            )
            if start_btn:
                self.log(f"story node {self._current_story_index:02d} battle starting")
                self._mark_node_done("story", self._current_story_index)
                self._current_story_index = max(
                    self._current_story_index + 1, self._next_node("story")
                )
                self._save_story_index()
                self._story_scroll_count = 0
                # Reset tab flag — after battle reward we'll be on hub,
                # need to re-switch to Story tab.
                self._story_tab_clicked = False
                return action_click_box(start_btn, "click 任務開始 (story battle)")

            # PRIORITY 2: Start button absent → either "can't sweep" banner
            # (already cleared) or a system-level "無目標" message. Use tight
            # region (center-right, where start button normally lives).
            no_goals = screen.find_any_text(
                ["該關卡無法", "该关卡无法", "没有任務目標", "沒有任務目標",
                 "没有任务目标"],
                region=(0.50, 0.55, 0.95, 0.90),
                min_conf=0.60,
            )
            if no_goals:
                self.log(
                    f"story node {self._current_story_index:02d} has no objectives: "
                    f"'{no_goals.text}', advancing"
                )
                # No-objectives = already cleared (reward taken); mark it.
                self._mark_node_done("story", self._current_story_index)
                self._current_story_index = max(
                    self._current_story_index + 1, self._next_node("story")
                )
                self._save_story_index()
                self._story_scroll_count = 0
                return action_back("skip node with no objectives")

            # PRIORITY 3: Neither button nor no-goals text — popup partial
            # render. Wait a beat, don't advance.
            return action_wait(400, "mission popup loading (story)")

        # ── Formation screen (team edit before battle) ──
        # After clicking 任務開始, lands on formation screen.
        # reference: click (1156/1280, 659/720) = (0.903, 0.915) for sortie button
        sortie = screen.find_any_text(
            ["出擊", "出击", "出撃", "出擎", "開始作戰", "开始作战",
             "戰鬥開始", "战斗开始"],
            region=(0.70, 0.75, 1.0, 0.98),
            min_conf=0.55,
        )
        if sortie:
            self._story_idle_ticks = 0
            return action_click_box(sortie, "click sortie (story battle)")

        # ── Battle result / reward confirm ──
        # After battle: fight-success-confirm, reward_acquired, etc.
        # reference: "story-fight-success-confirm" at (1117-1219, 639-687)
        next_btn = screen.find_any_text(
            ["下一步", "Next", "確認", "确认", "確定", "确定", "OK", "確", "确"],
            region=(0.25, 0.55, 0.95, 0.98),
            min_conf=0.55,
        )
        if next_btn:
            self._story_idle_ticks = 0
            return action_click_box(next_btn, "event story next/confirm")

        # ── AP purchase popup ──
        # Playing story nodes costs AP; game prompts to buy when depleted.
        # Must dismiss with 取消 to avoid spending premium currency.
        ap_buy = screen.find_any_text(
            ["是否購買", "是否购买", "購買AP", "购买AP"],
            min_conf=0.6,
        )
        if ap_buy:
            self._story_idle_ticks = 0
            self._story_ap_depleted = True
            cancel = screen.find_any_text(
                ["取消"],
                region=(0.25, 0.60, 0.55, 0.80),
                min_conf=0.55,
            )
            if cancel:
                self.log("AP depleted, cancelling purchase")
                return action_click_box(cancel, "cancel AP purchase")
            return action_click(0.40, 0.70, "cancel AP purchase (hardcoded)")

        # Open Story tab once.
        if not self._story_tab_clicked:
            tab = screen.find_any_text(
                ["Story", "劇情", "剧情", "故事"],
                region=(0.50, 0.0, 1.0, 0.24),
                min_conf=0.5,
            )
            if tab:
                self._story_tab_clicked = True
                self._story_idle_ticks = 0
                return action_click_box(tab, "switch to event story tab")

        # ── template-based sequential story node processing ──
        # Find target node by index, scroll if needed, click paired 入場.
        result = self._find_and_click_story_node(screen)
        if result is not None:
            return result

        self._story_idle_ticks += 1
        if self._phase_ticks > 250 or self._story_idle_ticks > 30:
            self.log(f"story phase complete (index={self._current_story_index}, "
                     f"max_seen={self._max_story_index_seen})")
            self._story_done = True
            self._clear_story_state()
            self._phase_ticks = 0
            self.sub_state = "enter"
            return action_wait(250, "story phase done")

        return action_wait(350, "event story scanning")

    def _find_numbered_nodes_on_screen(
        self, screen: ScreenState
    ) -> Tuple[List[Tuple[int, float]], List]:
        """Find numbered nodes and 入場 buttons on Story/Quest tabs.

        Resolution-independent: finds 入場 buttons first, then looks for
        2-digit node numbers to the left of them within the same row.

        Returns:
            nodes: list of (index, cy) for detected node numbers
            entry_buttons: list of OcrBox for 入場 buttons
        """
        # Find entry buttons first (wide region to handle any resolution)
        entry_buttons = []
        for pat in ["入場", "入场"]:
            for box in screen.find_text(pat, region=(0.40, 0.10, 1.0, 0.95),
                                        min_conf=0.45):
                entry_buttons.append(box)

        # Determine node number x-range from entry button positions.
        # Node numbers sit to the LEFT of entry buttons in the same row.
        if entry_buttons:
            entry_min_cx = min(b.cx for b in entry_buttons)
            node_cx_max = entry_min_cx - 0.03
            node_cx_min = max(0.0, entry_min_cx - 0.55)
        else:
            # Fallback: wide range covering both DXcam-full and window captures
            node_cx_min, node_cx_max = 0.15, 0.65

        nodes: List[Tuple[int, float]] = []
        for box in screen.ocr_boxes:
            if (box.confidence < 0.45
                    or box.cx < node_cx_min or box.cx > node_cx_max
                    or box.cy < 0.15 or box.cy > 0.90):
                continue
            text = box.text.strip()
            m = re.match(r"^(\d{1,2})$", text)
            if m:
                idx = int(m.group(1))
                if 1 <= idx <= 30:
                    nodes.append((idx, box.cy))

        return nodes, entry_buttons

    def _find_and_click_story_node(self, screen: ScreenState) -> Optional[Dict[str, Any]]:
        """Sequential story node processing (template-based).

        Finds the current target node by index, pairs it with its 入場 button,
        and clicks it. Scrolls the story list when the target is off-screen.
        Returns an action dict, or None if no actionable state found.
        """
        nodes, entry_buttons = self._find_numbered_nodes_on_screen(screen)

        # Update max seen index for termination detection
        if nodes:
            self._max_story_index_seen = max(
                self._max_story_index_seen,
                max(idx for idx, _ in nodes)
            )
            # On first call, start from the minimum visible node instead of 1.
            # The game auto-scrolls to the frontier; nodes above are completed.
            if self._current_story_index == 1 and min(idx for idx, _ in nodes) > 1:
                first_visible = min(idx for idx, _ in nodes)
                self.log(f"starting from first visible node {first_visible:02d} "
                         f"(skipping 01-{first_visible - 1:02d})")
                self._current_story_index = first_visible
                self._save_story_index()

        target = self._current_story_index
        target_str = f"{target:02d}"

        # Try to find the target node number on screen
        target_cy = None
        for idx, cy in nodes:
            if idx == target:
                target_cy = cy
                break

        if target_cy is not None:
            self._story_scroll_count = 0  # found target, reset scroll counter

            # Find paired 入場 button at similar Y position
            paired_entry = None
            best_dist = 0.08  # max Y distance for pairing
            for btn in entry_buttons:
                dist = abs(btn.cy - target_cy)
                if dist < best_dist:
                    best_dist = dist
                    paired_entry = btn

            if paired_entry:
                self._story_idle_ticks = 0
                self._story_node_pending = True
                self._story_consecutive_locks = 0
                self.log(f"clicking 入場 for story node {target_str} "
                         f"(y={paired_entry.cy:.3f})")
                return action_click_box(
                    paired_entry,
                    f"story node {target_str} 入場"
                )

            # Node visible, no 入場 button → locked. Stop after 2 in a row.
            self._story_consecutive_locks += 1
            if self._story_consecutive_locks >= 2:
                self.log(
                    f"story node {target_str} locked "
                    f"(consecutive={self._story_consecutive_locks}), ending story phase"
                )
                self._story_done = True
                self._clear_story_state()
                self._phase_ticks = 0
                self.sub_state = "enter"
                return action_wait(250, f"story phase stopped at locked node {target_str}")
            self.log(f"story node {target_str} no 入場 button (retry once)")
            self._current_story_index += 1
            self._save_story_index()
            self._story_scroll_count = 0
            return action_wait(300, f"node {target_str} locked, trying next")

        # Target node not found on screen
        if nodes:
            max_visible = max(idx for idx, _ in nodes)
            min_visible = min(idx for idx, _ in nodes)

            if target > max_visible:
                # Target is below visible area → scroll down
                if self._story_scroll_count < 8:
                    self._story_scroll_count += 1
                    self._story_idle_ticks = 0
                    self.log(f"scrolling down to find node {target_str} "
                             f"(visible: {min_visible:02d}-{max_visible:02d}, "
                             f"scroll #{self._story_scroll_count})")
                    return action_scroll(
                        0.75, 0.50, clicks=-5,
                        reason=f"scroll story list for node {target_str}"
                    )
                # Scrolled max times without finding target → no more nodes
                self.log(f"node {target_str} not found after {self._story_scroll_count} "
                         f"scrolls (max_seen={self._max_story_index_seen}), story complete")
                self._story_done = True
                self._clear_story_state()
                self._phase_ticks = 0
                self.sub_state = "enter"
                return action_wait(250, "story phase done (scrolled to end)")

            if target < min_visible:
                # Target is above visible area → scroll up
                if self._story_scroll_count < 8:
                    self._story_scroll_count += 1
                    self._story_idle_ticks = 0
                    return action_scroll(
                        0.75, 0.50, clicks=5,
                        reason=f"scroll up for node {target_str}"
                    )
                # Can't find it above either — skip to first visible
                self.log(f"node {target_str} not found above, jumping to {min_visible:02d}")
                self._current_story_index = min_visible
                self._story_scroll_count = 0
                return action_wait(300, f"jumping to node {min_visible:02d}")

        # If we just clicked an 入場 and are waiting for popup, be patient
        if self._story_node_pending:
            return action_wait(400, f"waiting for node {target_str} popup")

        # No nodes visible at all — might still be loading or transitioning
        return None

    def _challenge(self, screen: ScreenState) -> Dict[str, Any]:
        self._phase_ticks += 1

        # Non-game screen guard
        if not self._has_game_ui(screen) and len(screen.ocr_boxes) > 5:
            self._no_game_ticks += 1
            if self._no_game_ticks > 15:
                self.log("challenge: no game UI for 15+ ticks, resetting to enter")
                self._no_game_ticks = 0
                self.sub_state = "enter"
                return action_back("no game UI, pressing back to recover")
            return action_wait(500, "no game UI detected, waiting")
        else:
            self._no_game_ticks = 0

        # Grid-walk fallback controls (limited generic support).
        auto = screen.find_any_text(["AUTO", "自動", "自动"], region=(0.70, 0.0, 1.0, 0.24), min_conf=0.55)
        if auto and not self._grid_auto_enabled:
            self._grid_auto_enabled = True
            return action_click_box(auto, "enable auto on event grid")
        end_turn = screen.find_any_text(["結束回合", "结束回合", "End Turn"], region=(0.65, 0.0, 1.0, 0.28), min_conf=0.5)
        if end_turn:
            return action_click_box(end_turn, "end turn on event grid")

        # Handle cutscene/dialog during challenge (story interludes)
        skip = screen.find_any_text(["SKIP", "Skip", "跳過", "跳过"], min_conf=0.65)
        if skip:
            self._challenge_idle_ticks = 0
            return action_click_box(skip, "skip event challenge cutscene")

        if self._is_event_story_screen(screen):
            self._challenge_idle_ticks = 0
            # Click MENU to reveal Skip, then tap center as fallback
            menu = screen.find_any_text(["MENU"], region=(0.82, 0.0, 1.0, 0.14), min_conf=0.65)
            if menu:
                return action_click_box(menu, "click MENU during challenge cutscene")
            return action_click(0.94, 0.05, "click MENU (hardcoded) during challenge")

        # ── Auto-battle in progress (HUD: AUTO + COST) ──
        if self._is_auto_battle(screen):
            self._challenge_idle_ticks = 0
            speed_action = self._handle_battle_speed(screen)
            if speed_action:
                return speed_action
            return action_wait(1500, "challenge battle in progress (auto)")

        # ── Loading / battle transition ──
        if len(screen.ocr_boxes) <= 5:
            self._challenge_idle_ticks = 0
            return action_wait(500, "challenge loading/battle in progress")

        # ── Mission Info dialog (battle node) ──
        # Accept Trad/Simp/mixed-OCR variants: RapidOCR sometimes reads
        # "任務資讯" (Trad 資 + Simp 讯) or "任务資訊" etc.
        mission_info = screen.find_any_text(
            ["任務資訊", "任务资讯", "任務資讯", "任务資訊", "任務资讯"],
            min_conf=0.55,
        )
        if mission_info:
            self._challenge_idle_ticks = 0
            # PRIORITY 1: Start button — same anti-false-positive logic as
            # story/quest phases. Sub-objective list uses "★完成" which
            # contains "完成" and trips a global "已完成" scan.
            start_btn = screen.find_any_text(
                ["任務開始", "任务开始", "任務开始", "任务開始"],
                region=(0.55, 0.60, 0.90, 0.85),
                min_conf=0.55,
            )
            if start_btn:
                return action_click_box(start_btn, "click 任務開始 (challenge battle)")
            # PRIORITY 2: No-goals text in tight region where start button
            # normally sits. Drop global "已完成" — too noisy.
            no_goals = screen.find_any_text(
                ["該關卡無法", "该关卡无法", "没有任務目標", "沒有任務目標",
                 "没有任务目标"],
                region=(0.50, 0.55, 0.95, 0.90),
                min_conf=0.60,
            )
            if no_goals:
                self._challenge_completed_count += 1
                self._challenge_stage_index += 1
                self.log(f"challenge stage already completed: '{no_goals.text}' "
                         f"(completed={self._challenge_completed_count}, next_idx={self._challenge_stage_index})")
                if self._challenge_completed_count >= 5:
                    self.log("all challenge stages already completed")
                    self._challenge_done = True
                    self._save_state()
                    self._phase_ticks = 0
                    self.sub_state = "enter"
                    self._challenge_sweep_stage = 0
                    return action_back("all challenges completed, exiting")
                return action_back("skip already-completed challenge battle")
            return action_wait(400, "mission popup loading (challenge)")

        # ── Formation screen ──
        sortie = screen.find_any_text(
            ["出擊", "出击", "出撃", "出擎", "開始作戰", "开始作战",
             "戰鬥開始", "战斗开始"],
            region=(0.70, 0.75, 1.0, 0.98),
            min_conf=0.55,
        )
        if sortie:
            self._challenge_idle_ticks = 0
            return action_click_box(sortie, "click sortie (challenge battle)")

        # ── Battle result confirm ──
        result_confirm = screen.find_any_text(
            ["確認", "确认", "確定", "确定", "確", "确", "OK"],
            region=(0.25, 0.50, 0.95, 0.98),
            min_conf=0.6,
        )
        if result_confirm and screen.find_any_text(
                ["戰鬥結果", "战斗结果", "VICTORY", "DEFEAT",
                 "勝利", "敗北", "胜利", "败北"],
                min_conf=0.55):
            self._challenge_idle_ticks = 0
            return action_click_box(result_confirm, "confirm challenge battle result")

        if not self._challenge_tab_clicked:
            tab = screen.find_any_text(
                ["Challenge", "挑戰", "挑战"],
                region=(0.50, 0.0, 1.0, 0.24),
                min_conf=0.5,
            )
            if tab:
                self._challenge_tab_clicked = True
                self._challenge_idle_ticks = 0
                return action_click_box(tab, "switch to event challenge tab")

        # Sweep mini-FSM for challenge stages.
        if self._challenge_sweep_stage == 0:
            max_btn = screen.find_any_text(["MAX"], min_conf=0.7)
            if max_btn:
                self._challenge_sweep_stage = 1
                self._challenge_idle_ticks = 0
                return action_click_box(max_btn, "challenge sweep max")

            # Find ALL entry buttons, sorted by Y, and click the one at
            # _challenge_stage_index to cycle through stages instead of
            # always re-clicking the same first button.
            all_entries = []
            for pat in ["入場", "入场"]:
                all_entries.extend(
                    screen.find_text(pat, region=(0.45, 0.16, 1.0, 0.95),
                                     min_conf=0.55)
                )
            # Dedupe by Y proximity and sort
            all_entries.sort(key=lambda b: b.cy)
            if all_entries:
                idx = min(self._challenge_stage_index, len(all_entries) - 1)
                if self._challenge_stage_index >= len(all_entries):
                    # Cycled through all visible entry buttons
                    self.log("cycled through all visible challenge entry buttons")
                    self._challenge_done = True
                    self._save_state()
                    self._phase_ticks = 0
                    self.sub_state = "enter"
                    self._challenge_sweep_stage = 0
                    return action_wait(250, "all challenge entries exhausted")
                target = all_entries[idx]
                self._challenge_idle_ticks = 0
                return action_click_box(target, f"enter event challenge stage {idx + 1}")

            sweep = screen.find_any_text(["掃蕩", "扫荡"], region=(0.45, 0.35, 1.0, 0.75), min_conf=0.55)
            if sweep:
                self._challenge_idle_ticks = 0
                return action_click_box(sweep, "open event challenge sweep")

        elif self._challenge_sweep_stage == 1:
            sweep_start = screen.find_any_text(["掃蕩開始", "扫荡开始"], min_conf=0.55)
            if sweep_start:
                self._challenge_sweep_stage = 2
                self._challenge_idle_ticks = 0
                return action_click_box(sweep_start, "start challenge sweep")
            start_btn = screen.find_any_text(["開始", "开始"], region=(0.50, 0.40, 1.0, 0.70), min_conf=0.55)
            if start_btn:
                self._challenge_sweep_stage = 2
                self._challenge_idle_ticks = 0
                return action_click_box(start_btn, "start challenge sweep (fallback)")

        elif self._challenge_sweep_stage == 2:
            confirm = screen.find_any_text(
                ["確認", "确认", "確定", "确定", "確", "确", "OK"],
                region=(0.30, 0.30, 0.80, 0.92),
                min_conf=0.55,
            )
            if confirm:
                self._challenge_sweep_stage = 3
                self._challenge_idle_ticks = 0
                return action_click_box(confirm, "confirm challenge sweep")

        elif self._challenge_sweep_stage == 3:
            done = screen.find_any_text(
                ["掃蕩完成", "扫荡完成", "獲得獎勵", "获得奖励", "確認", "确认", "OK"],
                min_conf=0.55,
            )
            if done:
                self.log("challenge sweep complete")
                self._challenge_done = True
                self._save_state()
                self._phase_ticks = 0
                self.sub_state = "enter"
                self._challenge_sweep_stage = 0
                return action_click_box(done, "close challenge sweep result")

        self._challenge_idle_ticks += 1
        if self._phase_ticks > 200 or self._challenge_idle_ticks > 20:
            self.log("challenge phase complete")
            self._challenge_done = True
            self._save_state()
            self._phase_ticks = 0
            self.sub_state = "enter"
            self._challenge_sweep_stage = 0
            return action_wait(250, "challenge phase done")

        return action_wait(400, "event challenge scanning")

    def _mission(self, screen: ScreenState) -> Dict[str, Any]:
        """Process Quest (mission) stages sequentially, similar to story."""
        self._phase_ticks += 1

        # Non-game screen guard
        if not self._has_game_ui(screen) and len(screen.ocr_boxes) > 5:
            self._no_game_ticks += 1
            if self._no_game_ticks > 15:
                self.log("mission: no game UI for 15+ ticks, resetting to enter")
                self._no_game_ticks = 0
                self.sub_state = "enter"
                return action_back("no game UI, pressing back to recover")
            return action_wait(500, "no game UI detected, waiting")
        else:
            self._no_game_ticks = 0

        # ── Auto-battle in progress ──
        if self._is_auto_battle(screen):
            self._quest_idle_ticks = 0
            speed_action = self._handle_battle_speed(screen)
            if speed_action:
                return speed_action
            return action_wait(1500, "quest battle in progress (auto)")

        # ── Loading / battle transition ──
        if len(screen.ocr_boxes) <= 5:
            self._quest_idle_ticks = 0
            return action_wait(500, "quest loading/battle in progress")

        # ── Mission Info dialog (battle stage) ──
        mission_info = screen.find_any_text(
            ["任務資訊", "任务资讯", "任務資讯", "任务資訊", "任務资讯"],
            min_conf=0.55,
        )
        if mission_info:
            self._quest_idle_ticks = 0
            self._quest_node_pending = False
            # AP depleted — end quest phase
            if self._story_ap_depleted:
                self.log("AP depleted, ending quest phase")
                self._mission_done = True
                self._phase_ticks = 0
                self.sub_state = "enter"
                return action_back("close quest mission popup (AP depleted)")

            # PRIORITY 1: Start button visible → play. (Same reasoning as
            # story phase: mission popup's reward checklist contains "已完成"
            # badges that fire a false skip.)
            start_btn = screen.find_any_text(
                ["任務開始", "任务开始", "任務开始", "任务開始"],
                region=(0.55, 0.60, 0.90, 0.85),
                min_conf=0.55,
            )
            if start_btn:
                self.log(f"quest node {self._quest_current_index:02d} starting battle")
                self._mark_node_done("mission", self._quest_current_index)
                self._quest_current_index = max(
                    self._quest_current_index + 1, self._next_node("mission")
                )
                self._quest_scroll_count = 0
                self._mission_tab_clicked = False
                return action_click_box(start_btn, "click 任務開始 (quest)")

            # PRIORITY 2: tight-region no-goals detection.
            no_goals = screen.find_any_text(
                ["該關卡無法", "该关卡无法", "没有任務目標", "沒有任務目標",
                 "没有任务目标"],
                region=(0.50, 0.55, 0.95, 0.90),
                min_conf=0.60,
            )
            if no_goals:
                self.log(f"quest node {self._quest_current_index:02d} no objectives, advancing")
                self._mark_node_done("mission", self._quest_current_index)
                self._quest_current_index = max(
                    self._quest_current_index + 1, self._next_node("mission")
                )
                self._quest_scroll_count = 0
                return action_back("skip quest node with no objectives")

            return action_wait(400, "mission popup loading (quest)")

        # ── Formation screen ──
        sortie = screen.find_any_text(
            ["出擊", "出击", "出撃", "出擎", "開始作戰", "开始作战",
             "戰鬥開始", "战斗开始"],
            region=(0.70, 0.75, 1.0, 0.98),
            min_conf=0.55,
        )
        if sortie:
            self._quest_idle_ticks = 0
            return action_click_box(sortie, "click sortie (quest battle)")

        # ── AP purchase popup ──
        ap_buy = screen.find_any_text(
            ["是否購買", "是否购买", "購買AP", "购买AP"],
            min_conf=0.6,
        )
        if ap_buy:
            self._quest_idle_ticks = 0
            self._story_ap_depleted = True
            cancel = screen.find_any_text(
                ["取消"], region=(0.25, 0.60, 0.55, 0.80), min_conf=0.55,
            )
            if cancel:
                self.log("AP depleted in quest, cancelling purchase")
                return action_click_box(cancel, "cancel AP purchase (quest)")
            return action_click(0.40, 0.70, "cancel AP purchase (quest hardcoded)")

        # ── Switch to Quest tab ──
        if not self._mission_tab_clicked:
            mission_tab = screen.find_any_text(
                ["Quest", "Mission", "任務", "任务"],
                region=(0.45, 0.0, 1.0, 0.24),
                min_conf=0.5,
            )
            if mission_tab:
                self._mission_tab_clicked = True
                self._quest_idle_ticks = 0
                return action_click_box(mission_tab, "switch to event quest tab")

        # ── Sequential quest node processing (reuse story node detection) ──
        result = self._find_and_click_quest_node(screen)
        if result is not None:
            return result

        self._quest_idle_ticks += 1
        if self._phase_ticks > 200 or self._quest_idle_ticks > 25:
            self.log(f"quest/mission phase complete (index={self._quest_current_index})")
            self._mission_done = True
            self.sub_state = "enter"
            self._phase_ticks = 0
            return action_wait(200, "mission phase done")

        return action_wait(350, "event quest scanning")

    def _find_and_click_quest_node(self, screen: ScreenState) -> Optional[Dict[str, Any]]:
        """Sequential quest node processing, mirroring _find_and_click_story_node."""
        nodes, entry_buttons = self._find_numbered_nodes_on_screen(screen)

        if nodes:
            if self._quest_current_index == 1 and min(idx for idx, _ in nodes) > 1:
                first_visible = min(idx for idx, _ in nodes)
                self.log(f"quest: starting from first visible node {first_visible:02d}")
                self._quest_current_index = first_visible

        target = self._quest_current_index
        target_str = f"{target:02d}"

        target_cy = None
        for idx, cy in nodes:
            if idx == target:
                target_cy = cy
                break

        if target_cy is not None:
            self._quest_scroll_count = 0
            paired_entry = None
            best_dist = 0.08
            for btn in entry_buttons:
                dist = abs(btn.cy - target_cy)
                if dist < best_dist:
                    best_dist = dist
                    paired_entry = btn
            if paired_entry:
                self._quest_idle_ticks = 0
                self._quest_node_pending = True
                self._quest_consecutive_locks = 0  # reset on successful entry
                self.log(f"clicking 入場 for quest node {target_str}")
                return action_click_box(paired_entry, f"quest node {target_str} 入場")
            # No entry button — locked. Quest nodes unlock sequentially, so a
            # lock means all later nodes are also locked. Stop after 1 lock;
            # allow 1 retry in case of OCR miss on a real 入場 button.
            self._quest_consecutive_locks += 1
            if self._quest_consecutive_locks >= 2:
                self.log(
                    f"quest node {target_str} locked (consecutive locks="
                    f"{self._quest_consecutive_locks}), ending quest phase"
                )
                self._mission_done = True
                self._phase_ticks = 0
                self.sub_state = "enter"
                return action_wait(250, f"quest phase stopped at first lock (node {target_str})")
            self.log(f"quest node {target_str} no 入場 (retry once)")
            self._quest_current_index += 1
            self._quest_scroll_count = 0
            return action_wait(300, f"quest node {target_str} locked, retry next")

        if nodes:
            max_visible = max(idx for idx, _ in nodes)
            min_visible = min(idx for idx, _ in nodes)
            if target > max_visible:
                if self._quest_scroll_count < 8:
                    self._quest_scroll_count += 1
                    self._quest_idle_ticks = 0
                    return action_scroll(
                        0.75, 0.50, clicks=-5,
                        reason=f"scroll quest list for node {target_str}"
                    )
                self.log(f"quest node {target_str} not found after scrolling")
                self._mission_done = True
                self._phase_ticks = 0
                self.sub_state = "enter"
                return action_wait(250, "quest phase done (scrolled to end)")
            if target < min_visible:
                if self._quest_scroll_count < 8:
                    self._quest_scroll_count += 1
                    self._quest_idle_ticks = 0
                    return action_scroll(
                        0.75, 0.50, clicks=5,
                        reason=f"scroll up quest list for node {target_str}"
                    )
                self._quest_current_index = min_visible
                self._quest_scroll_count = 0
                return action_wait(300, f"quest: jumping to node {min_visible:02d}")

        if self._quest_node_pending:
            return action_wait(400, f"waiting for quest node {target_str} popup")

        return None

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if screen.is_lobby():
            self.log(
                f"done (story={self._story_done}, challenge={self._challenge_done}, mission={self._mission_done})"
            )
            # Clear state file when all phases done (next event starts fresh)
            if self._story_done and self._challenge_done and self._mission_done:
                try:
                    if _STATE_FILE.exists():
                        _STATE_FILE.unlink()
                except Exception:
                    pass
            return action_done("event activity complete")
        return action_back("event activity exit: back to lobby")
