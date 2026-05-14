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
    action_swipe,
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
        # 12 story nodes (mostly cutscenes, ~10t each) + 12 quest battles
        # (~30-60t each) + farming + shop comfortably exceeds 300t.  Bumped
        # to 800 so the skill can finish a full event in one pass.  Once
        # quest is split out into its own skill (planned), this can drop.
        self.max_ticks = 800

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
        # Top-right carousel swipe counter: cap attempts so we don't loop
        # forever on an all-schale carousel or a carousel with unreadable
        # avatars.
        self._carousel_swipes: int = 0
        self._CAROUSEL_SWIPE_LIMIT: int = 5
        # Delay after a swipe so the carousel animation settles before
        # we re-evaluate the visible slide.
        self._post_swipe_cooldown_tick: int = -100

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
        self._quest_blank_ticks: int = 0  # no 入場 anywhere (popup still covering?)
        # Cooldown after click 任務開始 — suppress node lock-detection while
        # battle is in progress (transient quest-list frames during the
        # popup→formation→battle transition can show next node still
        # locked because just-started node isn't ★3'd yet).
        self._quest_post_task_start_cooldown: int = 0
        self._story_blank_ticks: int = 0

        # ── Event-bonus team auto-select FSM (quick-edit flow) ──
        # See _formation_bonus_team() for the step-by-step flow. Default
        # OFF — enable via profile option `enable_bonus_team=True` once
        # the click targets are verified against your screen resolution.
        self._enable_bonus_team: bool = False
        # Whether the farming phase is allowed to run a one-off setup
        # battle to populate the saved sweep team with rate-up students
        # when the Bonus indicator is partial / missing.  Default ON
        # (corrected 2026-05-13): user wants "加成打满" before sweeping,
        # which requires one quick-edit battle per stage to update the
        # saved team.  The earlier gate-OFF default was a mistake — I
        # conflated this setup battle with the "Challenge" tab the user
        # wanted skipped; they're separate.  Challenge tab dispatch is
        # already disabled at the top of _enter; the bonus-setup battle
        # is a single ~1-minute investment that doubles every sweep
        # afterwards.
        self._enable_bonus_setup_battle: bool = True
        self._form_stage: str = "start"      # see _formation_bonus_team
        self._form_battle_node: int = -1      # which quest node this FSM is for
        self._form_ticks: int = 0             # stage-local tick counter
        # Set to True when 任務資訊 popup shows `Bonus` items in the
        # expected-rewards row — means the game already has a bonus-
        # configured team saved for this stage; skip the quick-edit FSM
        # (saves ~4 clicks + 3s per re-run).
        self._bonus_already_configured: bool = False
        # Mission phase: track whether we've already clicked "1部隊" tab
        # for the current battle node (one click per node, not per tick).
        self._team1_clicked_for_node: bool = False
        # Farming phase: bonus team setup tracking.
        # Per-stage: have we run quick-edit + battle-once to update the
        # saved sweep team?  Without this, sweep uses the team that was
        # last saved (often a no-bonus team from initial clearing) and
        # we lose ~50% drop value.  Keyed by stage idx so re-runs in
        # the same session don't redo it.
        self._farm_bonus_setup_done_stages: Set[int] = set()
        # Sub-FSM stage during bonus-setup battle (separate from
        # _farm_sweep_phase to keep flow readable).
        self._farm_bonus_battle_stage: str = "start"
        self._farm_bonus_battle_ticks: int = 0

        # Drift-recovery: on first tick of each phase, read ★ badges on
        # the visible nodes and advance the persisted index past any
        # node that shows ★★★ (full clear). Prevents replaying work
        # the user did manually between runs / after daily reset.
        self._quest_resync_done: bool = False
        self._story_resync_done: bool = False

        # ── Unified farming+shop phase state (replaces standalone
        # event_farming / event_shop skills) ──
        # Farming: sweep a preferred stage with MAX count, honoring AP
        # budget. Shop: scan each currency tab, skip 5:1 traps.
        self._preferred_stage: int = 0     # 0 = last-visible (reference default)
        self._farming_ap_budget: int = 0   # 0 = disabled
        # Minimum AP required to enter sweep. Below this we don't even
        # try — reward-badges get claimed in phase 0 head, then skill
        # jumps to shop and out. Stage 12 typically ~20 AP/sweep.
        self._min_ap_for_sweep: int = 20
        self._shop_auto_buy: bool = False
        self._shop_spend_currencies: tuple = ()
        # Furniture items (interactive cafe decor): ranked AFTER materials
        # by default. Flip to True via profile to buy furniture first.
        self._shop_furniture_first: bool = False
        # Farming FSM
        self._farm_stage_ticks: int = 0
        self._farm_sweep_phase: int = 0       # 0=pre-click, 1=max, 2=sweep_start, 3=confirm, 4=result
        self._farm_rounds_done: int = 0
        self._farm_max_rounds: int = 0        # 0 = until AP budget / visible cap
        self._farm_ap_spent: int = 0
        self._farm_ap_baseline: int = -1
        self._farm_reward_dialog_open: bool = False
        self._reward_wait_ticks: int = 0       # waiting for dialog to render
        self._reward_attempts: int = 0         # how many open-clicks tried
        self._reward_claim_abandoned: bool = False  # set True after 2 failed attempts
        # Event-nav bottom-bar badge detection (劇情 / 商店 / 任務 / 後日谈)
        self._event_nav_badges_scanned: bool = False
        self._event_nav_badges: dict = {}      # {name: 'red'|'yellow'}
        # 活動點數 progress (5663/15000 style). Cached so downstream
        # logic / next run can see how close to milestone cap we are.
        self._event_points_current: int = -1
        self._event_points_total: int = -1
        self._task_claim_stage: int = 0        # claim_tasks FSM stage
        self._task_claim_done: bool = False    # one-shot guard
        self._task_claim_back_attempts: int = 0
        # Shop FSM
        self._shop_tab_idx: int = 0
        self._shop_visited_tabs: list = []
        self._shop_current_tab: str = ""
        self._shop_state: dict = {}
        self._shop_scroll_attempts: int = 0
        self._shop_last_buy_target: str = ""
        self._shop_last_buy_cost: int = 0
        self._shop_last_buy_remaining: int = -1
        self._shop_buy_retry_count: int = 0
        self._shop_phase_a_wait: int = 0   # OCR settle delay on tab entry
        self._shop_just_scrolled: bool = False   # 1-tick settle after any scroll
        self._shop_total_tabs: int = 0   # learned from first rail scan
        # Last-tab two-phase flag: once all LIMITED items exhausted we
        # unlock purchases of unlimited items (user rule).
        self._shop_last_tab_limited_done: bool = False
        # Shop buy dialog FSM:
        #   0 = idle (scan+click 購買)
        #   1 = dialog open, click MAX (when remaining ≥ 2)
        #   2 = click 確認 (check grayed first → cancel if insufficient)
        self._shop_buy_dialog_stage: int = 0
        self._shop_buy_dialog_ticks: int = 0
        # "insufficient currency for current tab" flag — once set, we
        # stop trying new items in this tab and move on.
        self._shop_tab_insufficient: bool = False

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
        # identifies the current event — BUT we also seed it from the
        # store's persisted "last-seen event" memo so a skill run that
        # starts already on the event page (e.g. chained after daily
        # tasks, or the user manually opened the event page before
        # triggering the bot) still scopes progress to the right event
        # without having to pass through the lobby banner first.
        memo = self._progress_store.get_current_event_id()
        self._current_event_id = memo

        # Seed the local counters from whatever progress the store has
        # persisted for the memoised event.  If no event was played
        # before, or this is a fresh event rotation, these default to 1.
        self._current_story_index = self._next_node("story") if memo else 1
        self._quest_current_index = self._next_node("mission") if memo else 1
        self._story_done = self._phase_done("story") if memo else False
        self._mission_done = self._phase_done("mission") if memo else False
        self._challenge_done = self._phase_done("challenge") if memo else False
        if memo:
            summary = self._progress_store.summary(memo)
            self.log(
                f"restored last-seen event '{memo}' from store memo: "
                f"story={summary['story']} mission={summary['mission']} "
                f"challenge={summary['challenge']} — "
                f"story@{self._current_story_index} quest@{self._quest_current_index}"
            )

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
        self._quest_post_task_start_cooldown = 0
        self._story_consecutive_locks = 0
        self._quest_blank_ticks = 0
        self._story_blank_ticks = 0
        self._form_stage = "start"
        self._form_battle_node = -1
        self._form_ticks = 0
        self._bonus_already_configured = False
        self._farm_bonus_setup_done_stages = set()
        self._farm_bonus_battle_stage = "start"
        self._farm_bonus_battle_ticks = 0
        self._quest_resync_done = False
        self._story_resync_done = False
        self._farm_stage_ticks = 0
        self._farm_sweep_phase = 0
        self._farm_rounds_done = 0
        self._farm_quest_tab_clicked = False
        # Story bottom-of-list detection (resets on every skill start)
        self._story_last_max_visible = -1
        self._story_bottom_repeat = 0
        self._farm_ap_spent = 0
        self._farm_ap_baseline = -1
        self._farm_reward_dialog_open = False
        self._reward_wait_ticks = 0
        self._reward_attempts = 0
        self._reward_claim_abandoned = False
        self._event_nav_badges_scanned = False
        self._event_nav_badges = {}
        self._event_points_current = -1
        self._event_points_total = -1
        self._task_claim_stage = 0
        self._task_claim_done = False
        self._task_claim_back_attempts = 0
        self._shop_tab_idx = 0
        self._shop_visited_tabs = []
        self._shop_current_tab = ""
        self._shop_state = {}
        self._shop_scroll_attempts = 0
        self._shop_last_buy_target = ""
        self._shop_last_buy_cost = 0
        self._shop_last_buy_remaining = -1
        self._shop_buy_retry_count = 0
        self._shop_phase_a_wait = 0
        self._shop_just_scrolled = False
        self._shop_total_tabs = 0
        self._shop_last_tab_limited_done = False
        self._shop_buy_dialog_stage = 0
        self._shop_buy_dialog_ticks = 0
        self._shop_tab_insufficient = False
        self._shop_tab_grey_streak = 0
        self._shop_balance_unknown_ticks = 0

        # Auto stage-down: if persisted shop state says the tab for our
        # current preferred_stage is exhausted, drop stage by 1 (and
        # keep dropping as long as lower stages' tabs are also done).
        # Must run AFTER _current_event_id is set (above) and AFTER
        # _preferred_stage is seeded from profile (by pipeline init).
        self._auto_stage_down_from_shop_state()

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
        # Unknown event with no metadata → total_for(phase)==0 →
        # progress.done(0) returns True (vacuously).  That would cause
        # dispatch to think every phase is complete and exit immediately.
        # Trust the in-memory `_story_done` / `_mission_done` /
        # `_challenge_done` flags instead — they reflect actual scan
        # outcomes (scrolled-to-end, idle-timeout, etc.).
        if self._event_total(phase) <= 0:
            return False
        return self._progress_store.phase_done(self._current_event_id, phase)

    def _detect_event_id_from_page(self, screen) -> str:
        """OCR the activity-period text on an event page and synthesize a
        stable per-rotation event id.

        Looks for a date span like ``2026-04-2810:00～2026-05-1209:59``
        (digits + Chinese tilde / arrow).  The start date is unique per
        rotation, so ``auto_<YYYYMMDD>`` lets each new event get its own
        progress bucket without hand-curated templates.

        Returns "" if no matching period text is visible.
        """
        # Match e.g. "2026-04-28" or "2026-04-2810:00" — game runs the
        # date and time together with no space.
        period_box = screen.find_text_one(
            r"\d{4}-\d{2}-\d{2}",
            region=(0.20, 0.70, 0.90, 0.95),
            min_conf=0.80,
        )
        if not period_box:
            return ""
        import re
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})", period_box.text or "")
        if not m:
            return ""
        return f"auto_{m.group(1)}{m.group(2)}{m.group(3)}"

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
        # Persist the memo so the next skill run can recover the event
        # id even if it starts already on the event page (no lobby
        # banner classification will fire there).
        self._progress_store.set_current_event_id(event_id)
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
                "Battle Complete", "BattleComplete", "Battle-Complete",
                "BATTLE COMPLETE", "Battle Complet",
                "VICTORY", "DEFEAT", "勝利", "敗北", "胜利", "败北",
            ],
            min_conf=0.6,
        )
        if reward_popup:
            self._battle_speed_set = False  # battle ended, reset for next
            # VICTORY / DEFEAT splash screens.  Two distinct behaviors:
            #  - VICTORY: auto-dissolves into the post-battle reward
            #    screen after ~2s. Just wait.
            #  - DEFEAT: has a 確認 button at center-bottom that MUST
            #    be clicked to proceed (no auto-dismiss).  Scripted
            #    "剧情杀" defeats in event story chapter battles use
            #    this — bot was waiting forever otherwise (observed
            #    run_20260513_112359 t667-699 stuck 30+ ticks).
            kw = (reward_popup.text or "").lower()
            is_victory = any(w in kw for w in ("victory", "勝利", "胜利"))
            is_defeat = any(w in kw for w in ("defeat", "敗北", "败北"))
            if is_defeat:
                # DEFEAT screen: click the prominent yellow 確認 button.
                # Position: bottom-center (cx≈0.47, cy≈0.89).
                defeat_confirm = screen.find_any_text(
                    ["確認", "确认", "確", "确"],
                    region=(0.30, 0.80, 0.70, 0.99), min_conf=0.55,
                )
                self._reward_fallback_streak = 0
                if defeat_confirm:
                    return action_click_box(
                        defeat_confirm,
                        f"defeat: click 確認 to acknowledge (scripted loss OK)"
                    )
                # OCR missed the button — hardcoded center-bottom click.
                return action_click(0.47, 0.89,
                    f"defeat: hardcoded 確認 (OCR miss)")
            if is_victory:
                self._reward_fallback_streak = 0
                # Battle ended → clear quest cooldown so next quest scan
                # picks up newly-unlocked nodes (★3 of N unlocks N+1).
                self._quest_post_task_start_cooldown = 0
                return action_wait(500, f"battle splash '{reward_popup.text}', waiting for dismiss")
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

        # Shop-dialog hook: if we're mid-purchase, OUR FSM handles the
        # MAX/確認 dialog — otherwise _handle_common_popups would misfile
        # it as `update` notification and click 確認 repeatedly even when
        # grayed (insufficient currency).
        if (self.sub_state == "shop"
                and self._shop_buy_dialog_stage > 0):
            shop_dialog = self._handle_shop_buy_dialog(screen)
            if shop_dialog is not None:
                return shop_dialog

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
            # Challenge phase moved to brain/skills/event_challenge.py
            # (needs dedicated team composition + turn order). Never
            # dispatched from _enter anymore; if we somehow land here,
            # exit cleanly.
            self.log("challenge sub_state reached but phase is disabled — exiting")
            self.sub_state = "exit"
            return action_wait(200, "challenge phase disabled")
        if self.sub_state == "mission":
            return self._mission(screen)
        if self.sub_state == "farming":
            return self._farming(screen)
        if self.sub_state == "shop":
            return self._shop(screen)
        if self.sub_state == "claim_tasks":
            return self._claim_tasks(screen)
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

    # Battle HUD button states (from user reference screenshots):
    #   Speed button @ (0.949, 0.868):
    #     gray   = 1x (H≈0, S<50, V any)           — needs 2 clicks
    #     blue   = 2x (H≈100-115, S>120)           — needs 1 click
    #     yellow = 3x (H≈20-32,  S>150, V>200)     — already correct, skip
    #   Auto button @ (0.949, 0.940):
    #     gray   = off (S<50)                      — needs 1 click
    #     yellow = on  (H≈20-32, S>150)            — already on, skip
    _SPEED_BTN_XY = (0.949, 0.868)
    _AUTO_BTN_XY = (0.949, 0.940)

    def _classify_battle_btn(self, screen: ScreenState, xy) -> str:
        """Return 'yellow' | 'blue' | 'gray' | 'unknown' for a battle-HUD button."""
        bgr = screen.sample_color(xy[0], xy[1], patch=5)
        if bgr is None:
            return "unknown"
        try:
            import cv2
            import numpy as np
            pix = np.uint8([[list(bgr)]])  # BGR
            hsv = cv2.cvtColor(pix, cv2.COLOR_BGR2HSV)[0][0]
            h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
        except Exception:
            return "unknown"
        if s < 55:
            return "gray"
        if 15 <= h <= 35 and s >= 120 and v >= 160:
            return "yellow"
        if 85 <= h <= 130 and s >= 100:
            return "blue"
        return "unknown"

    def _handle_battle_speed(self, screen: ScreenState) -> Optional[Dict[str, Any]]:
        """Bring battle HUD to {speed=3x yellow, auto=on yellow}.

        Uses HSV color sampling at the speed/auto button positions to
        decide how many clicks are still needed. Returns None when both
        buttons are already in the desired state for the current battle.
        """
        if self._battle_speed_set:
            return None

        speed_color = self._classify_battle_btn(screen, self._SPEED_BTN_XY)
        auto_color = self._classify_battle_btn(screen, self._AUTO_BTN_XY)

        # Step 1: bring speed to yellow (3x). Cycle is 1x→2x→3x→1x.
        if speed_color == "blue":
            # One more click to reach yellow.
            return action_click(*self._SPEED_BTN_XY, "speed blue(2x) → yellow(3x)")
        if speed_color == "gray":
            # Two clicks needed; fire one now, next tick will fire the other.
            return action_click(*self._SPEED_BTN_XY, "speed gray(1x) → blue(2x)")
        # speed_color in {yellow, unknown}: speed is either already 3x or we
        # can't read it — don't click speed blindly.

        # Step 2: ensure auto is on (yellow). If gray, click once.
        if auto_color == "gray":
            return action_click(*self._AUTO_BTN_XY, "auto gray(off) → yellow(on)")

        # Both fine (or unreadable). Mark done for this battle.
        self._battle_speed_set = True
        return None

    # Carousel widget regions. The widget is a timer-badge + avatar +
    # pagination-dots stack that can be horizontally swiped to cycle
    # through active events. It appears in DIFFERENT screen locations
    # depending on which page we're on.
    _CAROUSEL_LOBBY_TIMER    = (0.82, 0.15, 1.0, 0.25)
    _CAROUSEL_LOBBY_SLIDE    = (0.80, 0.05, 1.0, 0.30)
    _CAROUSEL_LOBBY_CENTER   = (0.90, 0.17)
    _CAROUSEL_LOBBY_SWIPE    = ((0.95, 0.17), (0.82, 0.17))
    _CAROUSEL_CAMPAIGN_TIMER = (0.0, 0.08, 0.22, 0.22)
    _CAROUSEL_CAMPAIGN_SLIDE = (0.0, 0.05, 0.25, 0.32)
    # Click and swipe targets on campaign hub.
    # Layout (from user's screenshot): timer badge top y≈0.14, avatar body
    # y≈0.20–0.30. Swiping at y=0.18 overlaps the timer text and can mis-
    # dispatch. Move swipe + click onto the avatar body at y=0.25.
    _CAROUSEL_CAMPAIGN_CENTER = (0.10, 0.25)
    _CAROUSEL_CAMPAIGN_SWIPE = ((0.20, 0.25), (0.03, 0.25))

    def _find_carousel_widget(self, screen: ScreenState):
        """Detect which event carousel is visible on the current page.

        Returns a dict describing the widget's position key + click/swipe
        coordinates, or None if no carousel is found.
        """
        timer = self._find_event_timer(screen, region=self._CAROUSEL_LOBBY_TIMER)
        if timer:
            return {
                "page": "lobby",
                "position": "top_right",
                "slide_region": self._CAROUSEL_LOBBY_SLIDE,
                "click_xy": self._CAROUSEL_LOBBY_CENTER,
                "swipe_from": self._CAROUSEL_LOBBY_SWIPE[0],
                "swipe_to": self._CAROUSEL_LOBBY_SWIPE[1],
                "timer_text": timer.text,
            }
        timer = self._find_event_timer(screen, region=self._CAROUSEL_CAMPAIGN_TIMER)
        if timer:
            return {
                "page": "campaign_hub",
                "position": "top_left",
                "slide_region": self._CAROUSEL_CAMPAIGN_SLIDE,
                "click_xy": self._CAROUSEL_CAMPAIGN_CENTER,
                "swipe_from": self._CAROUSEL_CAMPAIGN_SWIPE[0],
                "swipe_to": self._CAROUSEL_CAMPAIGN_SWIPE[1],
                "timer_text": timer.text,
            }
        return None

    def _handle_carousel_widget(self, screen: ScreenState, carousel: dict):
        """Dispatch action for the detected carousel slide.

        Returns an action dict when we decide to swipe or click, or None
        to let the caller fall through to other priorities (e.g. unknown
        slide on campaign_hub — let the existing campaign-tile detector
        handle it).
        """
        is_schale = self._is_schale_signature(
            screen, region=carousel["slide_region"]
        )
        if is_schale:
            if self._carousel_swipes >= self._CAROUSEL_SWIPE_LIMIT:
                self.log(
                    f"carousel swipe limit {self._CAROUSEL_SWIPE_LIMIT} reached "
                    f"on {carousel['page']} — only 夏莱 found, giving up"
                )
                return action_done("carousel all-schale, deferred to daily tasks")
            self._carousel_swipes += 1
            self._post_swipe_cooldown_tick = self._enter_ticks
            fx, fy = carousel["swipe_from"]
            tx, ty = carousel["swipe_to"]
            self.log(
                f"{carousel['page']} carousel slide #{self._carousel_swipes} "
                f"is 夏莱 — swiping ({fx:.2f},{fy:.2f})→({tx:.2f},{ty:.2f})"
            )
            return action_swipe(
                fx, fy, tx, ty, duration_ms=350,
                reason=f"swipe carousel away from 夏莱 ({carousel['page']})",
            )
        # Non-schale slide — try template match
        tr_match = self._has_non_schale_event_banner(screen, carousel["position"])
        cx, cy = carousel["click_xy"]
        if tr_match is not None:
            self._set_current_event(tr_match.event)
            return action_click(
                cx, cy,
                f"click {carousel['page']} carousel ({tr_match.event} template)",
            )
        # Unknown but non-schale — trust the timer anchor and click.
        # CRITICAL: if we have a stale memoised event_id from a previous
        # rotation, applying its persisted "1-8 story done" state to this
        # NEW unknown event will make us scroll past every uncompleted
        # node looking for node 9 (observed in run_20260428_184236). When
        # we can't confirm the event identity, treat this as a fresh
        # event: clear the memo and reset phase counters to node 1 so we
        # start at the top of the list.
        if self._current_event_id:
            self.log(
                f"unknown event banner — clearing stale memo "
                f"'{self._current_event_id}' and resetting phase counters"
            )
            self._current_event_id = ""
            self._progress_store.set_current_event_id("")
            self._current_story_index = 1
            self._quest_current_index = 1
            self._story_done = False
            self._mission_done = False
            self._challenge_done = False
        self.log(
            f"{carousel['page']} carousel slide shows event "
            f"(timer '{carousel['timer_text']}') — clicking center (no template)"
        )
        return action_click(
            cx, cy,
            f"click {carousel['page']} carousel (unknown non-schale)",
        )

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
        # EVENT page has top-left header `活動` (or named event title).
        # CAMPAIGN HUB has top-left header `任務` + big 任務/劇情 tiles in
        # the center — don't confuse the two. Check the campaign-hub
        # signature first and bail out early so we don't mis-trigger on
        # the `任務` tile label at (0.56, 0.21).
        campaign_header = screen.find_any_text(
            ["任務", "任务"],
            region=(0.0, 0.0, 0.22, 0.10),
            min_conf=0.80,
        )
        if campaign_header:
            return False

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

        # English tabs are unambiguous — only event pages have them.
        eng_tabs = screen.find_any_text(
            ["Story", "Quest", "Challenge"],
            region=(0.50, 0.0, 1.0, 0.24),
            min_conf=0.55,
        )
        if eng_tabs:
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
            # Quick-edit popup signatures (opened via 快速編輯 button)
            "STRIKER", "SPECIAL", "自動", "自动", "快速",
            # 相性 / 敵人 header on the popup left side
            "相性", "敵人", "敌人",
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
        # Battle-active guard: if a quest battle is in progress, do NOT
        # count toward enter timeout — wait until battle finishes.
        # Without this, an orphan battle resume (from prior run abandoned
        # mid-fight) consumes the 24-tick budget while the bot just
        # watches the battle (run_20260513_175504 t085-t104 burned 20 of
        # the 24 ticks here before declaring "event unavailable").
        battle_marker = screen.find_any_text(
            ["AUTO", "Auto", "戰鬥時間", "战斗时间", "戰鬥開始", "战斗开始"],
            min_conf=0.6,
        )
        if battle_marker:
            return action_wait(800, "enter: battle in progress, waiting")
        self._enter_ticks += 1

        # Stranded on 活動任務 sub-page (typically because claim_tasks ran
        # and BACK key didn't fully return us to event page). Click the
        # top-left ← arrow to navigate back. Don't fall through to the
        # "event not available" give-up path.
        task_page = screen.find_any_text(
            ["活動任務", "活动任务", "活動仼務"],
            region=(0.0, 0.0, 0.40, 0.12), min_conf=0.55,
        )
        if task_page:
            self.log("_enter: stranded on 活動任務, clicking ← arrow to go back")
            return action_click(0.04, 0.04, "enter: ← back from 活動任務")

        if self._is_event_page(screen):
            self._phase_ticks = 0
            # Auto-identify the event from the period text on the page if
            # the lobby banner template didn't fire.  Gives every event
            # rotation its own progress bucket keyed by start-date — so a
            # rotation back to a known event recovers prior progress, and
            # an unknown event can't pollute another's state.
            if not self._current_event_id:
                detected = self._detect_event_id_from_page(screen)
                if detected:
                    self.log(f"event auto-identified by period text: {detected}")
                    self._current_event_id = detected
                    self._progress_store.set_current_event_id(detected)
                    # Seed counters from any existing per-event progress.
                    self._current_story_index = self._next_node("story")
                    self._quest_current_index = self._next_node("mission")
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
            # Dispatch rule: strict reference order story → mission → challenge.
            # If story isn't fully done, we STAY in story phase (the node-
            # finder can now auto-rewind the persisted index when it's out
            # of sync with UI — see _find_and_click_story_node). We only
            # exit the whole skill when story session-aborts AND there are
            # no clickable story nodes left — that's a hard block.
            # One-shot scan of nav-bar red badges on event page (劇情 /
            # 商店 / 任務 / 後日谈). These signal unclaimed / new items
            # that other skills may want to act on. Log + cache for the
            # pipeline to consume. If 任務 has a red dot, route to the
            # 全部領取 claim flow BEFORE farming/shop so we grab the
            # pending 活動任務 rewards (often credits + event points).
            if not getattr(self, "_event_nav_badges_scanned", False):
                self._event_nav_badges_scanned = True
                self._scan_event_nav_red_badges(screen)
                # Log current 活動點數 progress so user can track whether
                # farming is pushing the bar toward its 15000 max.
                cur, tot = self._parse_activity_points(screen)
                if cur is not None:
                    pct = int(100 * cur / max(1, tot))
                    self.log(
                        f"event points: {cur}/{tot} ({pct}% of milestone track)"
                    )
                    self._event_points_current = cur
                    self._event_points_total = tot
                if (self._event_nav_badges.get("任務") == "red"
                        and not getattr(self, "_task_claim_done", False)):
                    self.log("event-nav: 任務 has red dot, routing to claim_tasks")
                    self.sub_state = "claim_tasks"
                    return action_wait(200, "routing to claim 活動任務 rewards")

            # AP-low short-circuit (user rule: no AP → don't even bother
            # checking bonus, just claim pending rewards and leave). Skip
            # story + mission entirely — jump to farming which will do
            # the 獎勵資訊 red-badge claim in phase 0 then AP-gate to shop.
            ap_now = self._parse_ap_top_bar(screen)
            if 0 <= ap_now < self._min_ap_for_sweep:
                self.log(
                    f"AP {ap_now} < {self._min_ap_for_sweep} on event page — "
                    f"skipping story/mission, going straight to reward-claim + exit"
                )
                self.sub_state = "farming"
                return action_wait(200, f"AP {ap_now} low, reward-only mode")

            story_ok = self._phase_done("story")
            if not story_ok:
                if self._story_done:
                    # Story phase finished its scan (scrolled to end / idle
                    # timeout / scroll-end-no-more-nodes).  In-memory flag
                    # says "we're done with story".  Fall through to
                    # mission instead of exiting the whole skill — the
                    # remaining phases (mission / challenge / farming)
                    # don't depend on story completion.
                    self.log(
                        f"story phase done at node {self._next_node('story') - 1}"
                        f"/{self._event_total('story') or '?'} "
                        f"-> proceeding to mission phase"
                    )
                    # Don't return here — let dispatch fall through to the
                    # mission-ok check below so we route to mission/farming
                    # without an extra tick.
                else:
                    self.sub_state = "story"
                    self.log(
                        f"event page ready -> story phase "
                        f"(next node {self._next_node('story'):02d}/"
                        f"{self._event_total('story')})"
                    )
                    return action_wait(250, "start event story phase")

            mission_ok = self._phase_done("mission")
            if not mission_ok:
                if self._mission_done:
                    # Mission session-aborted (no entry buttons / consecutive
                    # locks). Remaining nodes need more story / daily unlock.
                    # Proceed to farming (AP sweep) since bonus team was set
                    # on the stages that are playable.
                    self.log(
                        f"mission exhausted at node {self._next_node('mission') - 1}/"
                        f"{self._event_total('mission')}, "
                        f"moving to farming phase"
                    )
                    self.sub_state = "farming"
                    return action_wait(200, "mission done, entering farming phase")
                self.sub_state = "mission"
                self.log(
                    f"story done -> mission phase "
                    f"(next node {self._next_node('mission'):02d}/"
                    f"{self._event_total('mission')})"
                )
                return action_wait(250, "start event mission phase")

            # Mission fully complete → farming (sweep AP) → shop (scan/buy)
            # Challenge phase is still in brain/skills/event_challenge.py
            # (requires team-comp + turn-order scripting).
            self.sub_state = "farming"
            return action_wait(200, "mission complete, entering farming phase")

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

        # ── PRIORITY 0: campaign-hub carousel widget ──
        # User directive (2026-04-24): do NOT use the lobby top-right
        # carousel or bottom-left big banner to enter events. Those are
        # unreliable (rotation-based, competing 夏莱 slides, dead-end
        # cards like 大決戰 OPEN). Always go to the Campaign hub first
        # and use its top-left widget there.
        if (self._enter_ticks - self._post_swipe_cooldown_tick) > 1:
            carousel = self._find_carousel_widget(screen)
            if carousel is not None and carousel["page"] == "campaign_hub":
                action = self._handle_carousel_widget(screen, carousel)
                if action is not None:
                    return action

        if current == "Lobby":
            # Simplified lobby branch: just navigate into Campaign. Skip
            # all lobby-level event-banner detection — the carousel +
            # banner logic used to race with rotation cycles and mis-
            # identified 夏莱 vs non-夏莱. Campaign hub has a stable
            # top-left widget we can handle deterministically.
            campaign_btn = screen.find_any_text(
                ["任務", "任务"],
                region=(0.80, 0.70, 1.0, 0.90),
                min_conf=0.6,
            )
            if campaign_btn:
                return action_click_box(campaign_btn, "open campaign from lobby")
            if self._enter_ticks > 3:
                # BA right-side 任務 button is near (0.95, 0.83) at most
                # aspect ratios; fallback click if OCR didn't find it.
                return action_click(0.95, 0.83, "open campaign (hardcoded)")
            return action_wait(400, "waiting for lobby UI to settle")

        # Unused-but-kept section below: the old lobby event-banner
        # priorities. Kept for reference only — all return None-equivalent
        # paths above. The following block is dead code reached only if
        # current is neither Lobby nor Mission, which we handle further
        # below.
        if False and current == "Lobby":
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

        # Lobby drop-out guard (see _mission comment)
        if screen.is_lobby():
            self.log("story: detected lobby mid-phase — exiting skill")
            self._story_done = True
            self.sub_state = "exit"
            self._clear_story_state()
            return action_wait(200, "dropped to lobby during story phase, exit")

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
                return action_click_box(enter_chapter, "click 進入章節 (story chapter)")
            # Fallback: click hardcoded 進入章節 position
            self.log(f"story node {self._current_story_index:02d} chapter entering (hardcoded)")
            self._mark_node_done("story", self._current_story_index)
            self._current_story_index = max(
                self._current_story_index + 1, self._next_node("story")
            )
            self._save_story_index()
            self._story_scroll_count = 0
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
            if not start_btn:
                # Template fallback: 任務開始 yellow button is visually
                # very distinctive — auto-cropped from real game frames.
                # Catches the case when OCR misses or mis-reads the text.
                start_btn = screen.find_template_one(
                    "task_start_button", region=(0.55, 0.60, 0.95, 0.90),
                )
            if start_btn:
                self.log(f"story node {self._current_story_index:02d} battle starting")
                self._mark_node_done("story", self._current_story_index)
                self._current_story_index = max(
                    self._current_story_index + 1, self._next_node("story")
                )
                self._save_story_index()
                self._story_scroll_count = 0
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
        if not sortie:
            sortie = screen.find_template_one(
                "sortie_button", region=(0.70, 0.75, 1.0, 0.99),
            )
        if sortie:
            self._story_idle_ticks = 0
            # Story phase = BA Story tab (cutscenes + occasional battles).
            # User rule (corrected 2026-04-28): leave story as-is, no
            # team management — the user can't adjust which team plays
            # story battles in any meaningful way and quick-edit isn't
            # needed here.
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

        # Open Story tab on first entry.  We CANNOT smart-skip by "numbered
        # nodes visible" because Quest and Challenge tabs also show
        # numbered nodes (run_20260428_211433 t006: bot was on Challenge
        # tab from previous run's last-active state, smart-skip thought
        # we're on Story tab and played Challenge node 01).  OCR can't
        # tell color (active = yellow text, inactive = dark text), so the
        # only safe option is to always click Story tab once per phase.
        # ~1.5s cost is worth it for correctness — without this we
        # blunder into Challenge battles unprepared.
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

        # Same battle-aware idle counter as _mission (don't increment
        # idle_ticks during active battle/formation/reward popups).
        on_story_hub = bool(self._find_numbered_nodes_on_screen(screen)[0])
        if on_story_hub:
            self._story_idle_ticks += 1
        if self._phase_ticks > 500 or self._story_idle_ticks > 30:
            self.log(f"story phase complete (index={self._current_story_index}, "
                     f"max_seen={self._max_story_index_seen}, "
                     f"phase_ticks={self._phase_ticks}, idle={self._story_idle_ticks})")
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

    def _find_node_star_counts(self, screen: ScreenState, nodes) -> Dict[int, int]:
        """Map node index → max ★ count detected near its row.

        BA shows ★ badges next to node numbers for cleared nodes:
        - ★     = 1-star clear
        - ★★   = 2-star clear
        - ★★★ = full clear (done — skip for farming)

        OCR usually reads the star run as one box; we count '★' chars.
        Returns {node_idx: stars} (stars in 0..3). Nodes not in dict
        have 0 stars visible.
        """
        result: Dict[int, int] = {}
        if not nodes:
            return result
        # Collect all OCR boxes containing ★ at moderate conf
        star_boxes = []
        for box in screen.ocr_boxes:
            if box.confidence < 0.45:
                continue
            txt = box.text
            if "★" not in txt:
                continue
            n_stars = min(3, txt.count("★"))
            if n_stars == 0:
                continue
            star_boxes.append((box.cy, n_stars))
        if not star_boxes:
            return result
        # Assign each star box to the closest node by cy (within ~0.06 band
        # to tolerate badge offset from node-number baseline).
        for cy, n_stars in star_boxes:
            best_node = None
            best_dist = 0.06
            for idx, ncy in nodes:
                d = abs(ncy - cy)
                if d <= best_dist:
                    best_dist = d
                    best_node = idx
            if best_node is not None:
                # Keep max stars seen for this node (if OCR returned multiple)
                result[best_node] = max(result.get(best_node, 0), n_stars)
        return result

    def _resync_index_from_stars(
        self, screen: ScreenState, phase: str, current_idx: int
    ) -> Tuple[int, bool]:
        """Common resync logic for story / mission phases.

        Advances ``current_idx`` past any visible node that shows ★★★.
        Never moves it backward. Returns (new_idx, changed).
        """
        nodes, _ = self._find_numbered_nodes_on_screen(screen)
        if not nodes:
            return current_idx, False
        stars = self._find_node_star_counts(screen, nodes)
        if not stars:
            return current_idx, False
        # Persist all 3-star nodes to the progress store — game's truth-of-
        # record (★★★ = fully cleared).  User rule: stars are the ONLY
        # reliable signal of "this node is done" since runs may end before
        # we finish scanning.  `_mark_node_done` is idempotent so calling
        # it on already-marked nodes is harmless.
        # `phase` here is "story" / "quest"; map "quest" to store's "mission".
        store_phase = "mission" if phase == "quest" else phase
        for idx, n_stars in stars.items():
            if n_stars >= 3:
                self._mark_node_done(store_phase, idx)
        candidates = sorted(set(idx for idx, _ in nodes))
        # Walk from current_idx onward; skip any node that's fully cleared.
        new_idx = current_idx
        while new_idx in candidates and stars.get(new_idx, 0) >= 3:
            new_idx += 1
        # If current_idx is below all visible nodes but visible ones are
        # all fully cleared, jump past the highest cleared.
        if current_idx < min(candidates):
            # Only advance if every visible node (from current_idx onward)
            # is fully cleared; otherwise keep current_idx — game will
            # scroll / we'll play in order.
            pass
        # Edge: all visible nodes >= current_idx are fully cleared → advance
        # past the highest one that IS cleared.
        visible_from_here = [c for c in candidates if c >= current_idx]
        if visible_from_here and all(stars.get(c, 0) >= 3 for c in visible_from_here):
            new_idx = max(new_idx, max(visible_from_here) + 1)
        if new_idx > current_idx:
            self.log(
                f"{phase} resync: persisted={current_idx} → {new_idx} "
                f"(visible stars={dict(sorted(stars.items()))})"
            )
            return new_idx, True
        return current_idx, False

    def _resync_quest_index_from_stars(self, screen: ScreenState) -> bool:
        new_idx, changed = self._resync_index_from_stars(
            screen, "quest", self._quest_current_index
        )
        if changed:
            self._quest_current_index = new_idx
            self._save_state()
        return changed

    def _resync_story_index_from_stars(self, screen: ScreenState) -> bool:
        new_idx, changed = self._resync_index_from_stars(
            screen, "story", self._current_story_index
        )
        if changed:
            self._current_story_index = new_idx
            self._save_story_index()
        return changed

    def _find_and_click_story_node(self, screen: ScreenState) -> Optional[Dict[str, Any]]:
        """Sequential story node processing (template-based).

        Finds the current target node by index, pairs it with its 入場 button,
        and clicks it. Scrolls the story list when the target is off-screen.
        Returns an action dict, or None if no actionable state found.
        """
        # One-shot drift recovery: read ★ badges, skip past fully-cleared
        # nodes the user may have played manually.
        if not self._story_resync_done:
            self._story_resync_done = True
            self._resync_story_index_from_stars(screen)

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
                # Persist nodes 1..first_visible-1 as done so the next run
                # doesn't see "no progress saved" and re-scan from scratch.
                # Game truth-of-record: those nodes ARE done (otherwise game
                # wouldn't have auto-scrolled past them).
                for n in range(1, first_visible):
                    self._mark_node_done("story", n)
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

            # Target has no 入場 → check if any LOWER visible node has one.
            # Persistent state may be out of sync (user did manual play) and
            # point to a locked node while a lower node is still unplayed.
            # Rewind the target to the lowest node that has a paired 入場.
            #
            # BUT: skip nodes that are ALREADY in our completed set — they
            # may show stale 入場 buttons (game UI lag right after a click)
            # and rewinding to them causes infinite replay (observed in
            # run_20260428_201909 t065 where node 03 was replayed three
            # times after each "rewind to lower clickable" hit).
            done_set: set = set()
            if self._current_event_id:
                try:
                    done_set = set(self._progress_store.progress(
                        self._current_event_id
                    ).story.completed)
                except Exception:
                    done_set = set()
            lowest_clickable = None
            lowest_btn = None
            for idx, cy in nodes:
                if idx >= target:
                    continue
                if idx in done_set:
                    continue
                for btn in entry_buttons:
                    if abs(btn.cy - cy) < 0.08:
                        if lowest_clickable is None or idx < lowest_clickable:
                            lowest_clickable = idx
                            lowest_btn = btn
                        break
            if lowest_clickable is not None:
                self.log(
                    f"persisted target {target_str} has no 入場, but lower node "
                    f"{lowest_clickable:02d} does — rewinding index (user may "
                    f"have progressed story out-of-band)"
                )
                self._current_story_index = lowest_clickable
                self._save_story_index()
                self._story_consecutive_locks = 0
                self._story_idle_ticks = 0
                self._story_node_pending = True
                return action_click_box(
                    lowest_btn,
                    f"story node {lowest_clickable:02d} 入場 (rewound)",
                )

            # If the only visible 入場 buttons are on already-done nodes
            # (stale UI lag right after a click), treat as transition and
            # wait — don't lock or scroll.  Avoids the false "all lower
            # nodes have 入場 but they're done" lock-out.
            if entry_buttons and done_set:
                visible_entry_idx = set()
                for idx, cy in nodes:
                    for btn in entry_buttons:
                        if abs(btn.cy - cy) < 0.08:
                            visible_entry_idx.add(idx)
                            break
                if visible_entry_idx and visible_entry_idx <= done_set:
                    self._story_blank_ticks = getattr(self, "_story_blank_ticks", 0) + 1
                    if self._story_blank_ticks <= 4:
                        return action_wait(
                            500,
                            f"story node {target_str}: visible 入場 are on "
                            f"already-done nodes only ({sorted(visible_entry_idx)}), "
                            f"waiting for UI to refresh"
                        )

            # Before incrementing lock: if ZERO 入場 buttons are visible
            # anywhere on screen, the reward/popup may still be covering OR
            # OCR missed everything — wait rather than burn the lock counter.
            if not entry_buttons:
                self._story_blank_ticks = getattr(self, "_story_blank_ticks", 0) + 1
                if self._story_blank_ticks <= 3:
                    return action_wait(500, f"story node {target_str} visible but no 入場 anywhere, waiting")
                self.log(
                    f"story: no 入場 buttons visible after {self._story_blank_ticks} "
                    f"ticks — assuming phase done"
                )
                self._story_done = True
                self._clear_story_state()
                self._phase_ticks = 0
                self.sub_state = "enter"
                return action_wait(250, "story phase done (no entry buttons)")
            self._story_blank_ticks = 0
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
            # Node visible but no 入場 button → genuinely locked.  Common
            # reasons: story-progress gate (need to complete prior nodes
            # in another tab), time-gate (event day N+1 unlock), or
            # daily-reset gate.  Game shows a lock icon at the row's
            # right edge instead of 入場.  We can't unlock it; advance
            # past and let the next-locked-detection bail out the phase.
            self.log(
                f"story node {target_str} LOCKED (no 入場 button — lock icon "
                f"present); advancing past; will stop phase if next is also locked"
            )
            self._current_story_index += 1
            self._save_story_index()
            self._story_scroll_count = 0
            return action_wait(300, f"node {target_str} locked, trying next")

        # Target node not found on screen
        if nodes:
            max_visible = max(idx for idx, _ in nodes)
            min_visible = min(idx for idx, _ in nodes)

            if target > max_visible:
                # Bottom-of-list detection: if visible range hasn't grown
                # since the last scroll, the list is already at its end
                # and the target node doesn't exist (metadata over-counts
                # for this event).  After 2 same-max scrolls, declare
                # story complete instead of burning 8 full scrolls.
                #
                # Without this we burn ~8 scrolls every run on events
                # whose true total is below the auto-detected default
                # (run_20260513_193301: scrolled 8x to find node 12 on
                # an 11-node event before declaring done).
                if max_visible <= getattr(self, "_story_last_max_visible", -1):
                    self._story_bottom_repeat = getattr(self, "_story_bottom_repeat", 0) + 1
                else:
                    self._story_bottom_repeat = 0
                self._story_last_max_visible = max_visible
                if self._story_bottom_repeat >= 2:
                    self.log(
                        f"node {target_str} not visible; list bottom reached at "
                        f"max={max_visible} (2 same-max scrolls), story complete"
                    )
                    self._story_done = True
                    self._clear_story_state()
                    self._phase_ticks = 0
                    self.sub_state = "enter"
                    return action_wait(250, "story phase done (list bottom)")
                # Target is below visible area → swipe up to bring lower
                # nodes into view.  Switched from wheel-scroll to swipe per
                # user 2026-05-13 — they don't want wheel/CTRL-wheel
                # gestures, only finger-style drag.
                if self._story_scroll_count < 8:
                    self._story_scroll_count += 1
                    self._story_idle_ticks = 0
                    self.log(f"scrolling down to find node {target_str} "
                             f"(visible: {min_visible:02d}-{max_visible:02d}, "
                             f"scroll #{self._story_scroll_count})")
                    return action_swipe(
                        0.75, 0.70, 0.75, 0.30, 500,
                        reason=f"swipe story list down for node {target_str}"
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
                # Target is above visible area → swipe down (drag-down
                # gesture brings upper nodes into view).
                if self._story_scroll_count < 8:
                    self._story_scroll_count += 1
                    self._story_idle_ticks = 0
                    return action_swipe(
                        0.75, 0.30, 0.75, 0.70, 500,
                        reason=f"swipe story list up for node {target_str}"
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

    # Challenge phase moved to brain/skills/event_challenge.py.
    # Keep the public method signature so legacy code paths don't
    # AttributeError — just route any accidental call to exit.
    def _challenge(self, screen: ScreenState) -> Dict[str, Any]:
        self.log('_challenge() called but phase is disabled — exiting skill')
        self.sub_state = 'exit'
        return action_wait(200, 'challenge phase disabled')

    # ───────────────── Event-bonus team auto-select (quick-edit flow) ─────

    def _formation_bonus_team(self, screen: ScreenState):
        """FSM: auto-pick event-bonus team via quick-edit → 自動 → 確認.

        User flow (simplified — game does bonus calc for us):
          1. Click 快速編輯 on the formation screen.
          2. Popup opens. Click 自動 button (bottom-center) — game
             auto-selects the highest-bonus team for this quest stage.
          3. Click 確認 (bottom-right of popup).
          4. Popup closes → formation now has bonus team → caller clicks 出擊.

        Returns action or None (None = FSM done; caller should sortie).
        Defensive: any step that can't find its target advances to 'done'
        so we never block the battle.
        """
        self._form_ticks += 1
        if self._form_ticks > 30:
            self.log("bonus-team FSM timeout, falling back to direct sortie")
            self._form_stage = "done"
            self._form_ticks = 0
            return None

        stage = self._form_stage

        # Stage: start — click 快速編輯 on formation screen.
        # Button sits TOP-RIGHT at cx≈0.91, cy≈0.29 (landscape 2243×1262).
        # OCR often drops the middle 編 char — observed variants:
        # 快速編輯 / 快速编辑 / 快速辑 / 快速編 / 快速编 / 速編輯 / 速编辑.
        # CRITICAL: OCR finds the TEXT LABEL but the clickable ICON sits
        # ABOVE the label. Click (cx, y1 - icon_offset) instead of center.
        if stage == "start":
            edit = screen.find_any_text(
                ["快速編輯", "快速编辑", "快速辑", "快速編", "快速编",
                 "速編輯", "速编辑"],
                region=(0.78, 0.15, 1.0, 0.55),
                min_conf=0.50,
            )
            if edit:
                self._form_stage = "click_auto"
                self._form_ticks = 0
                click_y = max(0.02, edit.y1 - 0.05)
                return action_click(edit.cx, click_y,
                                    "bonus-team: open quick edit (icon above label)")
            return action_wait(300, "bonus-team: scanning 快速編輯")

        # Stage: click_auto — in popup, click 自動 button (bottom area).
        if stage == "click_auto":
            auto = screen.find_any_text(
                ["自動", "自动", "AUTO", "Auto"],
                region=(0.35, 0.75, 0.70, 0.98),
                min_conf=0.55,
            )
            if auto:
                self._form_stage = "click_confirm"
                self._form_ticks = 0
                return action_click_box(auto, "bonus-team: click 自動")
            return action_wait(300, "bonus-team: scanning 自動 button")

        # Stage: click_confirm — click 確認 (bottom-right of popup).
        # OCR often clips to single char `確` / `确` at high conf (observed
        # at cx≈0.89, cy≈0.80).
        if stage == "click_confirm":
            confirm = screen.find_any_text(
                ["確認", "确认", "確定", "确定", "確", "确"],
                region=(0.80, 0.75, 1.0, 0.98),
                min_conf=0.55,
            )
            if confirm:
                self._form_stage = "done"
                self._form_ticks = 0
                return action_click_box(confirm, "bonus-team: confirm edit")
            return action_wait(300, "bonus-team: scanning 確認")

        # stage == "done": let caller click sortie.
        return None

    def _mission(self, screen: ScreenState) -> Dict[str, Any]:
        """Process Quest (mission) stages sequentially, similar to story."""
        self._phase_ticks += 1

        # Lobby drop-out guard: accidental ESC / game back-press during
        # popup-heavy scan can push us out of the event page. Don't scroll
        # / click on lobby — exit the skill cleanly.
        if screen.is_lobby():
            self.log("mission: detected lobby mid-phase (likely ESC dropped us out) — exiting skill")
            self._mission_done = True
            self.sub_state = "exit"
            self._save_state()
            return action_wait(200, "dropped to lobby during quest phase, exit")

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

        # ── Bonus-team FSM mid-flight guard ──
        # Once we've opened the quick-edit popup, the formation screen's
        # `出擊` text is hidden, so the sortie gate that bootstraps this
        # FSM can't keep driving it. Advance the FSM FIRST while it's in
        # a non-terminal state so the popup doesn't leak OCR numbers into
        # _find_and_click_quest_node (which would mistake Lv.90 / role
        # counts for quest node indices and scroll away the list).
        # Bonus-team FSM gate: fire when we're on a formation screen
        # (entry into FSM) OR we're already mid-FSM (clicked 快速編輯,
        # now in the auto/confirm popup so 快速編輯 anchor is gone).
        # Without this gate the FSM either never starts (previous
        # trigger rejected 'start' state) or fires on non-formation
        # screens like the quest list.
        fsm_in_progress = self._form_stage in ("click_auto", "click_confirm")
        on_formation = (self._form_stage == "start"
                        and screen.find_any_text(
                            ["快速編輯", "快速编辑", "快速辑", "快速編", "快速编",
                             "速編輯", "速编辑"],
                            region=(0.78, 0.15, 1.0, 0.55), min_conf=0.50,
                        ) is not None)
        if (self._enable_bonus_team and self._form_stage != "done"
                and (on_formation or fsm_in_progress)):
            fsm_action = self._formation_bonus_team(screen)
            if fsm_action is not None:
                self._quest_idle_ticks = 0
                return fsm_action

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
            if not start_btn:
                # Template fallback: 任務開始 yellow button is visually
                # very distinctive — auto-cropped from real game frames.
                # Catches the case when OCR misses or mis-reads the text.
                start_btn = screen.find_template_one(
                    "task_start_button", region=(0.55, 0.60, 0.95, 0.90),
                )
            if start_btn:
                # User's rule: if 獲得期待獎勵 row already shows `Bonus`
                # items on this 任務資訊 popup, the stage has already
                # been cleared with a bonus-equipped team — replaying
                # in mission phase gives nothing new (per-play drops
                # are better collected via sweep in event_farming).
                # Back out of the popup and advance the quest index
                # instead of starting a wasted battle.
                bonus_hits = [
                    b for b in screen.ocr_boxes
                    if "Bonus" in b.text and b.confidence >= 0.55
                ]
                if bonus_hits:
                    self.log(
                        f"quest node {self._quest_current_index:02d}: "
                        f"{len(bonus_hits)} Bonus item(s) in expected rewards — "
                        f"stage already bonus-cleared, skipping replay "
                        f"(mission phase only plays fresh nodes)"
                    )
                    # Mark as done in persisted store (same as if we played it)
                    self._mark_node_done("mission", self._quest_current_index)
                    self._quest_current_index = max(
                        self._quest_current_index + 1, self._next_node("mission")
                    )
                    self._quest_scroll_count = 0
                    self._save_state()
                    return action_back("bonus already obtained, back out + advance")

                self.log(f"quest node {self._quest_current_index:02d} starting battle")
                self._mark_node_done("mission", self._quest_current_index)
                self._quest_current_index = max(
                    self._quest_current_index + 1, self._next_node("mission")
                )
                self._quest_scroll_count = 0
                # Post-task-start cooldown: while > 0, skip lock-detection
                # paths in _find_and_click_quest_node so a brief quest-list
                # transition frame doesn't permanently mark the next node
                # locked when it would actually unlock after this battle's
                # ★3 (run_20260513_181944 t95: 09 task-start, transition
                # frame showed 10 still locked because 09 not ★3 yet, bot
                # marked 10 "retry next" → lost forever).  Ticks down each
                # tick this skill runs; reset to 0 on VICTORY/popup-dismiss.
                self._quest_post_task_start_cooldown = 60
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
        if not sortie:
            sortie = screen.find_template_one(
                "sortie_button", region=(0.70, 0.75, 1.0, 0.99),
            )
        if sortie:
            self._quest_idle_ticks = 0
            # User rule (corrected 2026-04-28): code's `_mission` = BA
            # "Quest" tab = user's "Quest 1-12" (the 12 clearing nodes).
            # Use team 1 + no quick-edit, same as user's saved preset.
            #
            # If a previous run's quick-edit clobbered the saved team to
            # team 2, sortie alone won't restore team 1 — click the
            # "1部隊" tab first if visible AND not already active.  Once
            # per battle node (gated by _team1_clicked_for_node).
            if self._form_battle_node != self._quest_current_index:
                self._form_battle_node = self._quest_current_index
                self._team1_clicked_for_node = False
            if not getattr(self, "_team1_clicked_for_node", False):
                # OCR commonly misreads 隊 as 隧 / 啄 etc. — match the
                # stable prefix `1部` (substring) rather than the full word.
                # Other tabs are `2部` / `3部` / `4部`, so `1部` is unique
                # in the left-edge region.
                team1 = screen.find_text_one(
                    "1部",
                    region=(0.0, 0.05, 0.18, 0.40),
                    min_conf=0.55,
                )
                if team1:
                    self._team1_clicked_for_node = True
                    return action_click_box(team1, "quest: switch to team 1 before sortie")
                self._team1_clicked_for_node = True
            return action_click_box(sortie, "click sortie (quest battle, direct — no quick-edit)")

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

        # ── Switch to Quest tab on first entry (always click — Story and
        # Challenge tabs also show numbered nodes, so "nodes visible"
        # alone can't tell us which tab we're on).
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

        # Idle counter — but ONLY when we're actually on the quest hub list.
        # During battle / formation / reward popups, _find_*_quest_node
        # returns None too, but that's not "idle" — we're working.
        # Without this gate, a 30-60s battle (~50 ticks) easily blows past
        # the 25-tick limit → bot prematurely declares mission done while
        # the battle is still running, then loses event-page state
        # (run_20260513_133808 t539 abandoned mission mid-quest-04 battle).
        on_quest_hub = bool(self._find_numbered_nodes_on_screen(screen)[0])
        if on_quest_hub:
            self._quest_idle_ticks += 1
        if self._phase_ticks > 400 or self._quest_idle_ticks > 25:
            self.log(f"quest/mission phase complete (index={self._quest_current_index}, "
                     f"phase_ticks={self._phase_ticks}, idle={self._quest_idle_ticks})")
            self._mission_done = True
            self.sub_state = "enter"
            self._phase_ticks = 0
            return action_wait(200, "mission phase done")

        return action_wait(350, "event quest scanning")

    def _find_and_click_quest_node(self, screen: ScreenState) -> Optional[Dict[str, Any]]:
        """Sequential quest node processing, mirroring _find_and_click_story_node."""
        # Post-task-start cooldown: a brief quest-list transition frame
        # can show right after click 任務開始 (popup closes → list visible
        # briefly → formation loads).  During this window, scanning for
        # "next node locked" gives a false positive because the just-
        # task-started node hasn't been ★3'd yet and therefore N+1 is
        # still legitimately locked.  Wait for battle + reward to finish
        # before re-scanning (cooldown reset to 0 by VICTORY/popup handlers).
        if getattr(self, "_quest_post_task_start_cooldown", 0) > 0:
            self._quest_post_task_start_cooldown -= 1
            return action_wait(
                400,
                f"post-task-start cooldown "
                f"({self._quest_post_task_start_cooldown} ticks left) — "
                f"waiting for battle+rewards to complete"
            )

        # Popup-residue guard: after bonus-team FSM finishes, the quick-edit
        # popup can stay visible another 1-3 ticks showing a confirmation
        # ("已自動編輯部隊") before collapsing to formation. During this
        # window the Lv.90 / stat numbers inside the popup would be
        # mistaken for quest node indices. Wait instead of scanning.
        if (screen.find_any_text(["STRIKER"], region=(0.40, 0.15, 0.65, 0.30), min_conf=0.70)
                and screen.find_any_text(["SPECIAL"], region=(0.50, 0.15, 0.75, 0.30), min_conf=0.70)):
            return action_wait(350, "quick-edit popup residue, waiting for formation")

        # One-shot drift recovery: read ★ badges, skip past fully-cleared
        # nodes. Handles: user played manually between sessions, daily
        # reset added new unlocks, state file out of sync.
        if not self._quest_resync_done:
            self._quest_resync_done = True
            self._resync_quest_index_from_stars(screen)

        nodes, entry_buttons = self._find_numbered_nodes_on_screen(screen)

        if nodes:
            if self._quest_current_index == 1 and min(idx for idx, _ in nodes) > 1:
                first_visible = min(idx for idx, _ in nodes)
                self.log(f"quest: starting from first visible node {first_visible:02d}")
                self._quest_current_index = first_visible
                # Persist game-truth: 1..first_visible-1 ARE done (game
                # auto-scrolled past them). Otherwise next run starts blank.
                for n in range(1, first_visible):
                    self._mark_node_done("mission", n)

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
                self._quest_stage_lock_ticks = 0    # reset stage-lock debounce
                # Reset bonus-team FSM so it re-runs on this quest's
                # formation screen (each quest may have different bonus
                # students; the FSM ends in 'done' state which would
                # otherwise stay sticky across quests).
                self._form_stage = "start"
                self._form_ticks = 0
                self.log(f"clicking 入場 for quest node {target_str}")
                return action_click_box(paired_entry, f"quest node {target_str} 入場")
            # No paired entry — but before calling it "locked", distinguish:
            #   (a) Target visible + OTHER nodes have 入場 → this specific node
            #       is genuinely locked (sequential unlock, stop phase).
            #   (b) Target visible + ZERO 入場 buttons anywhere → either all
            #       subsequent nodes locked (legit stop) OR popup/reward still
            #       obscuring the list OR cleared-node UI that hides 入場. Wait
            #       a few ticks; if it persists, treat as phase-done (nothing
            #       actionable) rather than lock-abort which over-counts.
            if not entry_buttons:
                self._quest_blank_ticks = getattr(self, "_quest_blank_ticks", 0) + 1
                # Scroll once to see if list was just out of scroll range
                if self._quest_blank_ticks <= 2:
                    return action_wait(500, f"quest node {target_str} visible but no 入場 anywhere, waiting")
                if self._quest_blank_ticks <= 5 and self._quest_scroll_count < 4:
                    self._quest_scroll_count += 1
                    return action_swipe(
                        0.75, 0.70, 0.75, 0.35, 500,
                        reason="no 入場 visible, swipe down to reveal remaining quest nodes"
                    )
                self.log(
                    f"quest: no 入場 buttons visible after {self._quest_blank_ticks} "
                    f"ticks — assuming all remaining nodes cleared or locked"
                )
                self._mission_done = True
                self._phase_ticks = 0
                self.sub_state = "enter"
                self._save_state()
                return action_wait(250, "quest phase done (no entry buttons)")
            # Here: other 入場 buttons exist but target's row has none.
            # BEFORE treating as real lock: the 入場s we see may all be for
            # EARLIER nodes (just-completed). OCR can flicker on a freshly-
            # unlocked row while animations settle — need ≥2 consistent
            # ticks with target-missing-but-other-visible before locking.
            # Also if any 入場 sits at or below target_cy, truly locked.
            later_entry_exists = any(btn.cy >= target_cy - 0.02 for btn in entry_buttons)
            self._quest_blank_ticks = 0
            if not later_entry_exists:
                # All visible 入場 are ABOVE target (earlier nodes). OCR may
                # just be mid-flicker — wait a tick, don't burn lock counter.
                self._quest_flicker_ticks = getattr(self, "_quest_flicker_ticks", 0) + 1
                if self._quest_flicker_ticks <= 3:
                    return action_wait(500, f"quest node {target_str} pending (ocr flicker, earlier 入場 only)")
                # After 3 flicker ticks, accept it as real lock.
            self._quest_flicker_ticks = 0
            # Stage-N-locked debounce: stay on N for K consecutive ticks
            # before giving up on it.  Game often takes 5-10s after a
            # battle's VICTORY for the next-stage unlock animation to
            # complete and refresh the entry button visibility.  Old
            # code advanced past N on first "no 入場 on N" detection
            # which permanently lost the stage if it was a transient
            # rendering issue (run_20260513_184415 t164: bot saw 12
            # locked once → advanced to 13 → past-tail exit → meanwhile
            # 12 actually unlocked at t169 farming phase).
            self._quest_stage_lock_ticks = getattr(
                self, "_quest_stage_lock_ticks", 0
            ) + 1
            STAGE_LOCK_K = 8   # ~4s @ 500ms tick; covers unlock animation
            if self._quest_stage_lock_ticks < STAGE_LOCK_K:
                return action_wait(
                    500,
                    f"quest node {target_str} no 入場 — debounce "
                    f"{self._quest_stage_lock_ticks}/{STAGE_LOCK_K} "
                    f"(waiting for unlock animation)"
                )
            self._quest_stage_lock_ticks = 0
            self._quest_consecutive_locks += 1
            if self._quest_consecutive_locks >= 2:
                self.log(
                    f"quest node {target_str} locked (consecutive locks="
                    f"{self._quest_consecutive_locks}), ending quest phase"
                )
                self._mission_done = True
                self._phase_ticks = 0
                self.sub_state = "enter"
                self._save_state()
                return action_wait(250, f"quest phase stopped at first lock (node {target_str})")
            self.log(f"quest node {target_str} truly locked after {STAGE_LOCK_K}-tick debounce — advancing")
            self._quest_current_index += 1
            self._save_state()
            self._quest_scroll_count = 0
            return action_wait(300, f"quest node {target_str} locked, retry next")

        if nodes:
            max_visible = max(idx for idx, _ in nodes)
            min_visible = min(idx for idx, _ in nodes)
            # Reset past-tail debounce when target is back in range
            if target <= max_visible:
                self._quest_past_tail_ticks = 0
            if target > max_visible:
                # Short-circuit: if we just skipped 4+ consecutive nodes
                # via bonus-already-obtained and target exceeds the
                # highest visible, there are no more nodes. No scroll.
                if target - max_visible > 0 and max_visible >= 9:
                    # Anti-flicker: right after clicking 任務開始 the index
                    # advances eagerly (e.g. 12→13) but the screen still
                    # shows the quest list briefly during the transition
                    # to formation/battle. Without debounce we fire
                    # "phase done" while the actual battle is still
                    # loading (run_20260504_224706 t292 declared done
                    # 1 tick after t291 task-start click → bot left for
                    # farming while quest 12 battle was still in
                    # progress).  Require 3 consecutive ticks of
                    # "past visible tail" before declaring done.
                    self._quest_past_tail_ticks = getattr(
                        self, "_quest_past_tail_ticks", 0
                    ) + 1
                    if self._quest_past_tail_ticks < 3:
                        return action_wait(
                            500,
                            f"quest target {target_str} past tail "
                            f"(transition? debounce {self._quest_past_tail_ticks}/3)"
                        )
                    self._quest_past_tail_ticks = 0
                    self.log(
                        f"quest node {target_str} past visible max {max_visible:02d} "
                        f"— phase done (no more nodes)"
                    )
                    self._mission_done = True
                    self._phase_ticks = 0
                    self.sub_state = "enter"
                    return action_wait(250, "quest phase done (past visible tail)")
                if self._quest_scroll_count < 3:
                    self._quest_scroll_count += 1
                    self._quest_idle_ticks = 0
                    return action_swipe(
                        0.75, 0.70, 0.75, 0.30, 500,
                        reason=f"swipe quest list down for node {target_str}"
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
                    return action_swipe(
                        0.75, 0.30, 0.75, 0.70, 500,
                        reason=f"swipe quest list up for node {target_str}"
                    )
                self._quest_current_index = min_visible
                self._quest_scroll_count = 0
                return action_wait(300, f"quest: jumping to node {min_visible:02d}")

        if self._quest_node_pending:
            return action_wait(400, f"waiting for quest node {target_str} popup")

        return None

    # ════════════════ Unified farming phase ════════════════
    # Runs AFTER mission phase completes. Sweeps the preferred stage
    # (default stage 12) until AP budget exhausted or max rounds reached.
    # Reuses existing popup handlers + stage-list OCR.

    def _parse_ap_top_bar(self, screen: ScreenState) -> int:
        """Read AP from top bar like '612/240'. Returns -1 if not found."""
        box = screen.find_any_text(
            [r"\d+/\d+"], region=(0.35, 0.0, 0.55, 0.08), min_conf=0.85,
        )
        if box is None:
            # Regex pattern doesn't match via find_any_text (it does
            # literal substring). Scan manually.
            for b in screen.ocr_boxes:
                if b.confidence < 0.85:
                    continue
                if not (0.35 <= b.cx <= 0.55 and b.cy <= 0.08):
                    continue
                m = re.match(r"^(\d+)/(\d+)$", b.text.strip())
                if m:
                    return int(m.group(1))
            return -1
        m = re.match(r"^(\d+)/(\d+)$", box.text.strip())
        return int(m.group(1)) if m else -1

    def _farm_bonus_setup_battle(self, screen: ScreenState) -> Dict[str, Any]:
        """One-shot battle to update the saved sweep team for the current
        farming stage so subsequent sweeps include rate-up bonus.

        Sub-stages (`_farm_bonus_battle_stage`):
          - "task_start"   : on 任務資訊 popup, click 任務開始
          - "switch_team2" : on formation, click "2部" tab — user rule:
                             team 1 is the preset clearing team for quest
                             1-12, team 2 is reserved for auto-bonus.
                             Quick-edit overwrites the active team slot,
                             so we MUST switch to team 2 first to avoid
                             clobbering team 1's preset.
          - "quick_edit"   : on formation, run _formation_bonus_team FSM
          - "sortie"       : after FSM done, click 出擊
          - "battle_wait"  : wait for Battle Complete / VICTORY / reward
                             popup (handled by global popup handler at
                             tick top, so this stage just monitors return
                             to event hub by re-detecting numbered nodes)
          - "complete"     : back to phase 0 with bonus_setup_done flag

        Defensive timeouts at every sub-stage to avoid hanging.
        """
        self._farm_bonus_battle_ticks += 1
        if self._farm_bonus_battle_ticks > 80:
            self.log("farming: bonus-setup FSM timeout, falling back to direct sweep")
            stage_idx = self._preferred_stage or 0
            self._farm_bonus_setup_done_stages.add(stage_idx)  # don't retry
            self._farm_sweep_phase = 0
            self._farm_bonus_battle_ticks = 0
            return action_wait(200, "bonus-setup timeout, retrying as direct sweep")
        sub = self._farm_bonus_battle_stage

        # Sub: click 任務開始 on 任務資訊 popup
        if sub == "task_start":
            start_btn = screen.find_any_text(
                ["任務開始", "任务开始", "任務开始", "任务開始"],
                region=(0.55, 0.60, 0.90, 0.85),
                min_conf=0.55,
            )
            if not start_btn:
                # Template fallback: 任務開始 yellow button is visually
                # very distinctive — auto-cropped from real game frames.
                # Catches the case when OCR misses or mis-reads the text.
                start_btn = screen.find_template_one(
                    "task_start_button", region=(0.55, 0.60, 0.95, 0.90),
                )
            if start_btn:
                self._farm_bonus_battle_stage = "switch_team2"
                return action_click_box(start_btn, "bonus-setup: click 任務開始")
            return action_wait(300, "bonus-setup: scanning 任務開始")

        # Sub: switch to team 2 tab on formation BEFORE quick-edit.
        # User rule: team 1 = clearing preset (don't overwrite), team 2 =
        # auto-bonus slot.  Quick-edit overwrites the currently selected
        # team slot, so we must select team 2 first.
        if sub == "switch_team2":
            sortie_visible = screen.find_any_text(
                ["出擊", "出击", "出撃", "出擎"],
                region=(0.70, 0.75, 1.0, 0.98),
                min_conf=0.55,
            )
            if not sortie_visible:
                return action_wait(300, "bonus-setup: waiting for formation screen")
            # `2部` substring covers OCR variants of `2部隊` / `2部隧`
            # (隊→隧 misread).  Other tabs are 1部/3部/4部, so 2部 is
            # unique in the left-edge region.
            team2 = screen.find_text_one(
                "2部",
                region=(0.0, 0.05, 0.18, 0.40),
                min_conf=0.55,
            )
            if team2:
                self._farm_bonus_battle_stage = "quick_edit"
                self._form_stage = "start"  # reset _formation_bonus_team FSM
                self._form_ticks = 0
                return action_click_box(team2, "bonus-setup: switch to team 2 (auto-bonus slot)")
            # Even if we can't find 2部 (already selected, or hidden by
            # animation), advance to quick-edit anyway — risk is minor
            # vs. blocking the whole flow.
            self._farm_bonus_battle_stage = "quick_edit"
            self._form_stage = "start"
            self._form_ticks = 0
            return action_wait(200, "bonus-setup: 2部 tab not detected, proceeding")

        # Sub: run quick-edit FSM (快速編輯 → 自動 → 確認).
        # DON'T gate on sortie visibility — once FSM clicks 快速編輯, the
        # popup overlays the formation and hides sortie. The FSM has its
        # own per-stage scanning so let it drive.
        if sub == "quick_edit":
            if self._form_stage != "done":
                fsm_action = self._formation_bonus_team(screen)
                if fsm_action is not None:
                    return fsm_action
            # FSM complete → next tick scan for sortie (popup closed by 確認)
            self._farm_bonus_battle_stage = "sortie"
            return action_wait(300, "bonus-setup: quick-edit done, finding sortie")

        # Sub: click 出擊 to start the bonus-setup battle
        if sub == "sortie":
            sortie = screen.find_any_text(
                ["出擊", "出击", "出撃", "出擎"],
                region=(0.70, 0.75, 1.0, 0.98),
                min_conf=0.55,
            )
            if sortie:
                self._farm_bonus_battle_stage = "battle_wait"
                self._farm_bonus_battle_ticks = 0  # reset timeout for battle
                return action_click_box(sortie, "bonus-setup: click 出擊")
            return action_wait(300, "bonus-setup: scanning 出擊")

        # Sub: wait for battle to complete and we're back on event hub
        if sub == "battle_wait":
            # Detect return to event hub: numbered quest nodes visible
            # AND "Quest" tab text in the tab-bar region.  Battle screens
            # don't show this combo.
            quest_tab = screen.find_any_text(
                ["Quest", "任務", "任务"],
                region=(0.45, 0.0, 1.0, 0.24), min_conf=0.55,
            )
            nodes, _ = self._find_numbered_nodes_on_screen(screen)
            if quest_tab and nodes:
                stage_idx = self._preferred_stage or 0
                self._farm_bonus_setup_done_stages.add(stage_idx)
                self._farm_sweep_phase = 0  # back to phase 0 (re-enter stage)
                self._farm_bonus_battle_stage = "start"
                self._farm_bonus_battle_ticks = 0
                self.log(
                    f"farming: bonus-setup battle complete for stage {stage_idx}, "
                    f"saved team now has bonus → re-entering for sweep"
                )
                return action_wait(300, "bonus-setup done, sweep next")
            # Still in battle / formation / reward popups — let global
            # popup handler dismiss popups; just wait.
            return action_wait(500, "bonus-setup: waiting for battle to complete")

        # Defensive fallback for unknown sub-state
        self.log(f"bonus-setup unknown stage '{sub}', resetting")
        self._farm_sweep_phase = 0
        self._farm_bonus_battle_stage = "start"
        self._farm_bonus_battle_ticks = 0
        return action_wait(200, "bonus-setup reset")

    def _farming(self, screen: ScreenState) -> Dict[str, Any]:
        """Sweep preferred stage until AP budget done, then → shop."""
        self._phase_ticks += 1

        # Lobby drop-out guard
        if screen.is_lobby():
            self.log("farming: detected lobby mid-phase — skipping farming")
            self.sub_state = "shop"
            return action_wait(200, "dropped to lobby during farming, skip to shop")

        # Prevent runaway — cap at 300 phase ticks regardless of budget.
        if self._phase_ticks > 300:
            self.log("farming: phase_ticks cap hit, moving to shop")
            self.sub_state = "shop"
            self._phase_ticks = 0
            return action_wait(200, "farming phase tick cap")

        # ── 章節資訊 popup guard ──
        # Farming is supposed to click 入場 on Quest-tab stages.  But if
        # the Story tab happens to be the active tab when farming starts,
        # the 入場 column we see belongs to STORY chapters.  Clicking one
        # opens the 章節資訊 popup with 進入章節 button — and the popup
        # masks the 入場 grid behind it.  Without a guard, farming clicks
        # the same y=0.88 入場 for ~700 ticks because OCR still picks up
        # background 入場 text bleeding past the popup edges
        # (run_20260513_195859 t640-1340: stuck at chapter 11 五塵降臨).
        #
        # Fix: detect 章節資訊 + close it via ESC (we explicitly don't
        # want to enter that story chapter from farming).  The next
        # farming tick lands on the bare event page and can re-route
        # through the Quest tab.
        chapter_info = screen.find_any_text(
            ["章節資訊", "章节资讯", "章節資", "章节资"],
            min_conf=0.55,
        )
        if chapter_info:
            self.log(
                "farming: 章節資訊 popup detected (clicked story 入場 by mistake) — "
                "closing and re-routing through Quest tab"
            )
            # Reset farm sub-phase so we re-detect tab + stage from scratch.
            # Also clear quest-tab-clicked flag so we force Quest tab
            # selection again after popup closes.
            self._farm_sweep_phase = 0
            self._farm_stage_ticks = 0
            self._farm_quest_tab_clicked = False
            # Try OCR'd X close button first (top-right of modal); ESC fallback.
            close_btn = screen.find_any_text(
                ["X", "x", "×", "✕"],
                region=(0.62, 0.06, 0.94, 0.30),
                min_conf=0.55,
            )
            if close_btn:
                return action_click_box(close_btn, "close 章節資訊 popup")
            return action_back("ESC out of 章節資訊 popup")

        # Popup-residue guard inherited from _mission — tick top handles
        # reward / confirm popups via the global reward handler before we
        # get here. We handle sweep-specific dialogs below.

        # AP budget gate — read top bar, skip if below reserve
        if self._farm_ap_baseline < 0:
            self._farm_ap_baseline = self._parse_ap_top_bar(screen)
        ap_now = self._parse_ap_top_bar(screen)
        if self._farming_ap_budget > 0 and ap_now >= 0 and self._farm_ap_baseline >= 0:
            spent = max(0, self._farm_ap_baseline - ap_now) + self._farm_ap_spent
            if spent >= self._farming_ap_budget:
                self.log(f"farming: AP budget reached ({spent}/{self._farming_ap_budget}), → shop")
                self.sub_state = "shop"
                return action_wait(200, "AP budget spent, move to shop")

        # Sub-phase dispatch
        phase = self._farm_sweep_phase

        # Phase 0: reward-claim check first (red badge on 獎勵資訊),
        # then find + click preferred stage's 入場 button
        if phase == 0:
            # Activity points cross thresholds → 獎勵資訊 button gets a
            # red badge. Claim before the badge piles up. OCR often
            # reads only `資訊` (partial) — accept that as a fallback,
            # scoped by cx ≥ 0.78 so it's the bottom-right 獎勵資訊
            # button, not unrelated `任務資訊` dialogs.
            reward_btn = screen.find_any_text(
                ["獎勵資訊", "奖励资讯", "獎勵资訊", "奖勵資訊"],
                region=(0.78, 0.80, 1.0, 0.98), min_conf=0.55,
            )
            if reward_btn is None:
                # Weak match: partial `資訊` in the same narrow corner.
                resxin = screen.find_any_text(
                    ["資訊", "资讯"],
                    region=(0.78, 0.85, 1.0, 0.95), min_conf=0.55,
                )
                if resxin is not None:
                    reward_btn = resxin
            # Dialog-open signature: 獎勵資訊 popup. OCR often cuts the
            # full titles in half (`清單` instead of `獎勵清單`, `领取`
            # instead of `領取獎勵`), so accept the partials too.
            # Additional anchors: `目標` / `目标` — the "target reward"
            # label at top-center of dialog (cx≈0.59).
            dialog_sig = screen.find_any_text(
                ["獎勵清單", "奖励清单", "清單", "清单",
                 "目標獎勵", "目标奖励", "目標", "目标",
                 "領取獎勵", "领取奖励", "領取奖励", "领取獎勵",
                 "領取", "领取"],
                region=(0.25, 0.05, 0.80, 0.95), min_conf=0.55,
            )
            if reward_btn and screen.has_red_badge(reward_btn) \
                    and not getattr(self, "_reward_claim_abandoned", False) \
                    and not dialog_sig:
                if not getattr(self, "_farm_reward_dialog_open", False):
                    self._farm_reward_dialog_open = True
                    self._reward_wait_ticks = 0
                    self.log("farming: red badge on 獎勵資訊, opening to claim")
                    # First attempt: box center. Second attempt: slightly
                    # above (button label area, avoiding the red dot
                    # graphic that doesn't respond to taps).
                    if getattr(self, "_reward_attempts", 0) == 0:
                        return action_click_box(reward_btn, "farming: open 獎勵資訊 (try 1: box center)")
                    else:
                        cy = max(0.02, reward_btn.y1 - 0.02)
                        return action_click(reward_btn.cx, cy,
                                            "farming: open 獎勵資訊 (try 2: slightly above)")

            if getattr(self, "_farm_reward_dialog_open", False):
                if not dialog_sig:
                    # Dialog not yet open — wait a few ticks for animation
                    self._reward_wait_ticks = getattr(self, "_reward_wait_ticks", 0) + 1
                    if self._reward_wait_ticks < 6:
                        return action_wait(400, "farming: waiting for 獎勵資訊 dialog")
                    # Gave up waiting. Previous click ineffective.
                    self._farm_reward_dialog_open = False
                    self._reward_wait_ticks = 0
                    self._reward_attempts = getattr(self, "_reward_attempts", 0) + 1
                    if self._reward_attempts >= 2:
                        self._reward_claim_abandoned = True
                        self.log(
                            "farming: reward-claim click ineffective 2x, "
                            "abandoning for this run"
                        )
                    return action_wait(200, "farming: reward dialog didn't open")
                # Dialog IS open — find 領取獎勵 button (yellow button
                # at bottom center, text at cx≈0.50, cy≈0.83). OCR may
                # only read 2 of the 4 chars, so accept partials.
                claim_btn = screen.find_any_text(
                    ["領取獎勵", "领取奖励", "領取奖励", "领取獎勵",
                     "領取", "领取"],
                    region=(0.30, 0.75, 0.80, 0.99), min_conf=0.55,
                )
                if claim_btn:
                    # After a successful claim the 領取獎勵 button turns
                    # GRAY (desaturated, dim). OCR still reads the text
                    # so we'd click forever. Detect "grayed" by sampling
                    # 4 points around the button corners and checking if
                    # NONE of them are yellow (active button = yellow).
                    # Use the text OCR box bounds + small outward offset
                    # to probe both edges of the button.
                    probe_pts = [
                        (claim_btn.x1 - 0.02, claim_btn.cy),  # left edge
                        (claim_btn.x2 + 0.02, claim_btn.cy),  # right edge
                        (claim_btn.cx, claim_btn.y1 - 0.01),  # above text
                        (claim_btn.cx, claim_btn.y2 + 0.01),  # below text
                    ]
                    yellow_hits = sum(
                        1 for (x, y) in probe_pts
                        if 0 < x < 1 and 0 < y < 1
                        and screen.is_button_yellow(x, y)
                    )
                    if yellow_hits == 0:
                        self.log(
                            "farming: 領取獎勵 no-yellow around button (already "
                            "claimed / grayed) → closing dialog"
                        )
                        close_btn = screen.find_any_text(
                            ["X", "×"], region=(0.85, 0.0, 1.0, 0.15), min_conf=0.5,
                        )
                        self._farm_reward_dialog_open = False
                        self._reward_wait_ticks = 0
                        if close_btn:
                            return action_click_box(close_btn, "farming: close 獎勵資訊 (done)")
                        return action_back("farming: close 獎勵資訊 (done, fallback)")
                    self.log(f"farming: clicking 領取獎勵 (yellow_hits={yellow_hits}/4)")
                    return action_click_box(claim_btn, "farming: 領取獎勵")
                # Dialog open but no claim button visible → already claimed
                # / no threshold crossed yet. Close via X or BACK.
                close_btn = screen.find_any_text(
                    ["X", "×"], region=(0.85, 0.0, 1.0, 0.15), min_conf=0.5,
                )
                self._farm_reward_dialog_open = False
                self._reward_wait_ticks = 0
                if close_btn:
                    return action_click_box(close_btn, "farming: close 獎勵資訊")
                return action_back("farming: back out of 獎勵資訊")

            # AP gate: stage 12 sweep costs ~20-40 AP per run. If current
            # AP can't cover a single sweep, there's no point entering the
            # stage list. Skip straight to shop (still scans + persists
            # budget) and out. Reward-claim check above already ran.
            ap_now = self._parse_ap_top_bar(screen)
            if 0 <= ap_now < self._min_ap_for_sweep:
                self.log(
                    f"farming: AP {ap_now} below minimum {self._min_ap_for_sweep} — "
                    f"skipping sweep, reward-badges already checked, → shop"
                )
                self.sub_state = "shop"
                self._phase_ticks = 0
                return action_wait(200, f"AP {ap_now} too low, skip to shop")

            # Make sure Quest tab is selected first.  One-time click on
            # entry to farming — even if Quest text was visible we don't
            # know if the tab is ACTIVE without pixel-color check.  If we
            # were on Story tab, the nodes we'd scan below are story
            # chapters and clicking 入場 would open 章節資訊 popup → loop.
            quest_tab = screen.find_any_text(
                ["Quest", "任務", "任务"],
                region=(0.45, 0.0, 1.0, 0.24), min_conf=0.55,
            )
            if quest_tab and not getattr(self, "_farm_quest_tab_clicked", False):
                self._farm_quest_tab_clicked = True
                self.log("farming: clicking Quest tab on entry to ensure correct list")
                return action_click_box(quest_tab, "farming: force Quest tab active")
            nodes, entry_buttons = self._find_numbered_nodes_on_screen(screen)
            if not entry_buttons:
                # All visible nodes are locked (no 入場 buttons) AND we
                # can see numbered nodes — we scrolled past the playable
                # ones into the locked tail.  Scroll UP to find playable
                # stages (highest unlocked is the best farming target).
                # Without this guard, bot kept clicking Quest tab forever
                # since it's already active (run_20260513_180124 t187+
                # 200+ ticks of "ensure quest tab" throttled).
                if nodes:
                    self._farm_stage_ticks = getattr(self, "_farm_stage_ticks", 0) + 1
                    if self._farm_stage_ticks > 6:
                        self.log("farming: scrolled past unlocked stages too long, → shop")
                        self.sub_state = "shop"
                        self._farm_sweep_phase = 0
                        self._farm_stage_ticks = 0
                        return action_wait(200, "farming: no playable stage, skip to shop")
                    self.log(
                        f"farming: visible nodes {sorted({n for n,_ in nodes})} all locked, "
                        f"scrolling UP to find playable stage"
                    )
                    return action_swipe(
                        0.75, 0.30, 0.75, 0.70, 500,
                        reason="farming: swipe up to find unlocked stage"
                    )
                if quest_tab:
                    return action_click_box(quest_tab, "farming: ensure quest tab")
                return action_wait(400, "farming: waiting for stage list")
            # Prefer stage index; fall back to bottom-most (highest cy)
            target = None
            pref = self._preferred_stage or 0
            if pref > 0:
                target_cy = None
                for idx, cy in nodes:
                    if idx == pref:
                        target_cy = cy
                        break
                if target_cy is not None:
                    for btn in entry_buttons:
                        if abs(btn.cy - target_cy) < 0.08:
                            target = btn
                            break
            if target is None:
                target = max(entry_buttons, key=lambda b: b.cy)
            self._farm_sweep_phase = 1
            self._farm_stage_ticks = 0
            self.log(f"farming: click 入場 for stage {pref or 'bottom-most'} @ y={target.cy:.2f}")
            return action_click_box(target, f"farming: enter stage {pref}")

        # Phase 1: 任務資訊 popup is open; check bonus team status before sweeping
        if phase == 1:
            self._farm_stage_ticks += 1
            if self._farm_stage_ticks > 10:
                # Popup failed to open; reset
                self._farm_sweep_phase = 0
                self._farm_stage_ticks = 0
                return action_back("farming: MAX not found, back out and retry")
            # Detect "stage cannot sweep" — multiple game-side messages:
            #   - "無法掃蕩" (generic)
            #   - "★3 達成時開放" / "★3 達成时开放" (sweep unlocks after 3-star clear)
            # Both indicate this stage isn't sweepable yet.  Skip stage,
            # try a LOWER-numbered stage (which is more likely 3-star cleared).
            # Without lower-stage fallback, run_20260513_181143 t232+ bot
            # cycled forever entering stage 8 (locked sweep) every 12 ticks.
            no_sweep = screen.find_any_text(
                ["無法掃蕩", "无法扫荡",
                 "達成時開放", "达成时开放", "達成時開", "达成时开"],
                min_conf=0.55,
            )
            if no_sweep:
                # Try lower stage on next attempt — track failed stages
                self._farm_unsweepable_stages = getattr(
                    self, "_farm_unsweepable_stages", set()
                )
                # The stage we just entered = max visible idx in entry button row
                # Use preferred_stage as a proxy if not yet known
                guess_idx = self._preferred_stage or 0
                self._farm_unsweepable_stages.add(guess_idx)
                # Step preferred_stage down by 1 (try next lower)
                if self._preferred_stage and self._preferred_stage > 1:
                    self._preferred_stage -= 1
                    self.log(
                        f"farming: stage not sweepable ('{no_sweep.text}') — "
                        f"stepping preferred_stage down to {self._preferred_stage}"
                    )
                    self._farm_sweep_phase = 0
                    self._farm_stage_ticks = 0
                    return action_back("farming: not sweepable, try lower stage")
                self.log("farming: stage not sweepable + no lower stage, skip to shop")
                self.sub_state = "shop"
                self._farm_sweep_phase = 0
                return action_back("farming: stage not sweepable")

            # USER RULE (2026-05-04, refined 2026-05-13): before sweeping,
            # check whether the saved sweep team already has full event
            # bonus.  The game shows `Bonus` items in the 獲得期待獎勵 row
            # only when the saved team contains rate-up students.  If the
            # indicator is absent, the historical behaviour was to run one
            # quick-edit setup battle to populate the saved team.
            #
            # User (run_20260513_185751 feedback): "quest打完不要去打 challenge,
            # 以后单独拿出来做".  After the quest phase completes they do
            # NOT want farming to fire a second battle on the same stage —
            # it doubles AP usage and feels like the bot is running an
            # extra battle no one asked for.  So gate this auto-setup
            # behind `enable_bonus_setup_battle` (default OFF).  When off,
            # we just log the missing Bonus indicator and continue to
            # sweep with whatever team is saved (drops will be smaller
            # but no surprise battle).
            stage_idx = self._preferred_stage or 0
            if stage_idx not in self._farm_bonus_setup_done_stages:
                bonus_hits = [
                    b for b in screen.ocr_boxes
                    if "Bonus" in b.text and b.confidence >= 0.55
                ]
                # User feedback (2026-05-13): "活动还是没有把加成打满,
                # 依旧是直接就去扫荡了" — one Bonus tag means saved team
                # has SOME rate-up students but probably not the full
                # team.  Maxed-out bonus typically shows 2-3 Bonus tags
                # on the expected-rewards row.  Treat ≥2 as "already
                # maxed" and run setup when count is 0 or 1.
                _BONUS_FULL_MIN = 2
                if len(bonus_hits) < _BONUS_FULL_MIN:
                    if self._enable_bonus_setup_battle:
                        self.log(
                            f"farming: stage {stage_idx} has only "
                            f"{len(bonus_hits)} Bonus tag(s) (need ≥{_BONUS_FULL_MIN} "
                            f"for full bonus) → running quick-edit battle to "
                            f"max bonus team"
                        )
                        self._farm_sweep_phase = 100  # bonus-setup battle
                        self._farm_bonus_battle_stage = "task_start"
                        self._farm_bonus_battle_ticks = 0
                        self._farm_stage_ticks = 0
                        return action_wait(200, "farming: starting bonus-setup battle")
                    self.log(
                        f"farming: stage {stage_idx} has {len(bonus_hits)} Bonus tag(s) "
                        f"but enable_bonus_setup_battle=False — sweeping as-is"
                    )
                    self._farm_bonus_setup_done_stages.add(stage_idx)
                else:
                    # Bonus already maxed (≥2 tags) — proceed to sweep
                    self._farm_bonus_setup_done_stages.add(stage_idx)
                    self.log(
                        f"farming: stage {stage_idx} bonus maxed "
                        f"({len(bonus_hits)} tags ≥ {_BONUS_FULL_MIN}), proceeding to sweep"
                    )

            max_btn = screen.find_any_text(["MAX"], min_conf=0.7)
            if max_btn:
                self._farm_sweep_phase = 2
                self._farm_stage_ticks = 0
                return action_click_box(max_btn, "farming: click MAX sweep count")
            return action_wait(400, "farming: waiting for MAX button")

        # Phase 100-104: bonus-setup battle (one-shot per stage to update
        # the saved sweep team to include rate-up students).
        if phase == 100:
            return self._farm_bonus_setup_battle(screen)

        # Phase 2: click 掃蕩開始
        if phase == 2:
            self._farm_stage_ticks += 1
            if self._farm_stage_ticks > 10:
                self._farm_sweep_phase = 0
                return action_back("farming: 掃蕩開始 not found")
            # OCR mis-reads 掃 as 捅 and 蕩 as 荡 systematically on this
            # font (observed conf 0.66 on `捅荡開始`). Accept variants.
            sweep_start = screen.find_any_text(
                ["掃蕩開始", "扫荡开始", "掃荡開始", "扫蕩开始",
                 "捅荡開始", "捅荡开始", "捅蕩開始", "捅蕩开始",
                 "荡開始", "荡开始", "蕩開始", "蕩开始"],
                min_conf=0.50,
            )
            if not sweep_start:
                # Template fallback: cyan 掃蕩開始 button auto-cropped
                # from real frame.  Catches OCR misses (when 掃 character
                # drops entirely or confidence falls below 0.50).
                sweep_start = screen.find_template_one(
                    "sweep_start_button", region=(0.55, 0.40, 0.95, 0.85),
                )
            if sweep_start:
                # If 掃蕩開始 button is greyed out, the stage isn't
                # sweepable (★3 not yet achieved on this stage).  Don't
                # click — fall back to "step preferred_stage down" path
                # to find a sweepable lower stage.
                if screen.is_button_grey(sweep_start.cx, sweep_start.cy):
                    self.log(
                        f"farming: 掃蕩開始 button is GREY (stage not yet "
                        f"sweepable); stepping preferred_stage down"
                    )
                    if self._preferred_stage and self._preferred_stage > 1:
                        self._preferred_stage -= 1
                    self._farm_sweep_phase = 0
                    self._farm_stage_ticks = 0
                    return action_back("farming: sweep button grey, try lower stage")
                self._farm_sweep_phase = 3
                self._farm_stage_ticks = 0
                return action_click_box(sweep_start, "farming: click 掃蕩開始")
            return action_wait(300, "farming: waiting for 掃蕩開始")

        # Phase 3: confirm dialog (optional)
        if phase == 3:
            self._farm_stage_ticks += 1
            if self._farm_stage_ticks > 8:
                # Maybe no confirm dialog; advance
                self._farm_sweep_phase = 4
                return action_wait(200, "farming: no confirm dialog, advancing")
            confirm = screen.find_any_text(
                ["確認", "确认", "確定", "确定", "確", "确"],
                region=(0.30, 0.50, 0.80, 0.92), min_conf=0.55,
            )
            if confirm:
                self._farm_sweep_phase = 4
                self._farm_stage_ticks = 0
                return action_click_box(confirm, "farming: confirm sweep")
            return action_wait(300, "farming: waiting for confirm")

        # Phase 4: dismiss result (reward popup handler at tick top catches
        # the main case; here we just detect result screen + advance round)
        if phase == 4:
            self._farm_stage_ticks += 1
            # SKIP button at top-right lets us bypass the reward animation
            # immediately instead of waiting ~10s for it to finish.
            skip_btn = screen.find_any_text(
                ["SKIP", "Skip", "跳過", "跳过", ">>"],
                region=(0.85, 0.0, 1.0, 0.15), min_conf=0.55,
            )
            if skip_btn:
                return action_click_box(skip_btn, "farming: SKIP reward animation")
            result = screen.find_any_text(
                ["掃蕩完成", "扫荡完成", "獲得獎勵", "获得奖励"],
                min_conf=0.55,
            )
            if result or self._farm_stage_ticks > 12:
                self._farm_rounds_done += 1
                self._farm_sweep_phase = 0
                self._farm_stage_ticks = 0
                self.log(f"farming: round {self._farm_rounds_done} done")
                # Update AP accounting so next diff is accurate
                ap = self._parse_ap_top_bar(screen)
                if ap >= 0 and self._farm_ap_baseline >= 0:
                    self._farm_ap_spent += max(0, self._farm_ap_baseline - ap)
                    self._farm_ap_baseline = ap
                # Stop if max_rounds set and reached (unlimited when 0)
                if self._farm_max_rounds > 0 and self._farm_rounds_done >= self._farm_max_rounds:
                    self.sub_state = "shop"
                    return action_wait(250, f"farming: {self._farm_rounds_done} rounds done, → shop")
                return action_wait(400, f"farming: round {self._farm_rounds_done} complete, loop")
            return action_wait(500, "farming: waiting for sweep result")

        # Unknown phase → reset
        self._farm_sweep_phase = 0
        return action_wait(300, "farming: unknown phase, reset")

    # ════════════════ Unified shop phase ════════════════

    # 5:1+ exchange trap: CHARACTER-SPECIFIC 神名文字 shards (美補(偶像)的神名文字
    # etc.). These cost 200-350 P points for 5 character shards — always
    # lossy vs. just buying the character straight. Match when name ends
    # with 神名文字 AND has 的 (possessive marker).
    #
    # NOTE: keep generic 神名文字碎片 BUYABLE — it's a building material
    # user explicitly wants in P-shop priority #2.
    @staticmethod
    def _looks_like_exchange_trap(name: str, unit_cost: int) -> bool:
        """Detect character-specific 神名文字 shard exchanges."""
        if unit_cost < 5:
            return False
        # `X的神名文字` (X=character) or `X(Y)的神名文字` (skin variant)
        return "的神名文字" in name

    # "Unlimited purchase" heuristic: BA shop items with remaining count
    # way above the per-account-per-week cap (e.g. 500-9999) are usually
    # evergreen currency exchanges (pay event ticket → get 信用貨幣 etc).
    # User rule: non-last tabs MUST NOT buy unlimited items (traps). Last
    # tab MAY buy them ONCE all limited items are exhausted.
    _UNLIMITED_REMAINING_THRESHOLD = 300

    @classmethod
    def _is_unlimited_item(cls, card: dict) -> bool:
        rem = card.get("remaining", 0)
        return rem >= cls._UNLIMITED_REMAINING_THRESHOLD

    def _parse_shop_cards(self, screen: ScreenState) -> List[Dict[str, Any]]:
        """Parse item cards: anchor on `可購買 N 次`, walk up for name,
        down for cost. Shares logic with the old EventShopSkill."""
        import re as _re
        qty_re = _re.compile(r"^[xX×]\s*\d[\d.,]*[KkMm]?$")
        rem_re = _re.compile(r"可購買\s*(\d+)\s*次|可购买\s*(\d+)\s*次")
        banned = ("購買", "购买", "可購買", "可购买", "Bonus", "稀有")
        cards: List[Dict[str, Any]] = []
        anchors: List[Tuple[Any, int]] = []
        for box in screen.ocr_boxes:
            if box.confidence < 0.55:
                continue
            m = rem_re.search(box.text)
            if not m:
                continue
            n = next((g for g in m.groups() if g), None)
            if n:
                anchors.append((box, int(n)))
        for anchor, remaining in anchors:
            col_x = anchor.cx
            # Name: top-most legit text in same column above anchor
            name_cands = []
            for b in screen.ocr_boxes:
                if b.confidence < 0.55:
                    continue
                if abs(b.cx - col_x) >= 0.10:
                    continue
                if b.cy >= anchor.cy or anchor.cy - b.cy >= 0.30:
                    continue
                t = b.text.strip()
                if not t or any(k in t for k in banned):
                    continue
                if qty_re.match(t):
                    continue
                if _re.fullmatch(r"[\dxX×/+\-.,]+", t):
                    continue
                name_cands.append(b)
            name_cands.sort(key=lambda b: b.cy)
            name = name_cands[0].text.strip() if name_cands else ""
            if not name:
                continue
            # Cost: below anchor, numeric (allow comma)
            cost_cands = []
            for b in screen.ocr_boxes:
                if b.confidence < 0.55:
                    continue
                if abs(b.cx - col_x) >= 0.08:
                    continue
                if b.cy <= anchor.cy or b.cy - anchor.cy >= 0.15:
                    continue
                stripped = b.text.strip().replace(",", "")
                if _re.fullmatch(r"\d{1,6}", stripped):
                    cost_cands.append((b, int(stripped)))
            cost_cands.sort(key=lambda t: t[0].cy - anchor.cy)
            unit_cost = cost_cands[0][1] if cost_cands else 0
            cards.append({
                "name": name,
                "remaining": remaining,
                "unit_cost": unit_cost,
                "cy": anchor.cy,   # used by row-grouped ranker
            })
        return cards

    def _persist_shop_state(self) -> None:
        try:
            import time as _time
            path = _STATE_FILE.parent / "event_shop_state.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "timestamp": _time.strftime("%Y-%m-%d %H:%M:%S"),
                "event_id": self._current_event_id,
                "total_tabs": self._shop_total_tabs,
                "points_current": self._event_points_current,
                "points_total": self._event_points_total,
                "by_currency": self._shop_state,
            }
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self.log(f"shop state saved: {path}")
        except Exception as e:
            self.log(f"shop state save failed: {e}")

    def _auto_stage_down_from_shop_state(self) -> None:
        """Inter-run stage-down decision. Reads persisted shop state
        and auto-decrements ``self._preferred_stage`` when the tabs
        corresponding to higher stages are exhausted.

        User rule chain (for 4-tab shop):
          stage 12 (last tab): exhausted when both limited + unlimited done
          stage 11 (2nd-from-last): exhausted when limited done (non-last
                                     tabs never buy unlimited)
          stage 10, 9: same as stage 11

        Tab index mapping for N-tab shop:
          stage 12 → position N-1 (bottom)
          stage 11 → position N-2
          stage 10 → position N-3
          stage 9  → position N-4
        Ignores state from a different event_id (fresh event = clean slate).
        """
        try:
            path = _STATE_FILE.parent / "event_shop_state.json"
            if not path.exists():
                return
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        stored_event = data.get("event_id", "")
        if (stored_event and self._current_event_id
                and stored_event != self._current_event_id):
            self.log(
                f"auto stage-down: shop state from event "
                f"{stored_event!r} ≠ current {self._current_event_id!r}, ignored"
            )
            return
        by_currency = data.get("by_currency", {}) or {}
        total_tabs = int(data.get("total_tabs") or 0)
        if total_tabs == 0:
            # derive from keys like tab0/tab1/...
            idxs = [int(k[3:]) for k in by_currency.keys()
                    if isinstance(k, str) and k.startswith("tab") and k[3:].isdigit()]
            total_tabs = (max(idxs) + 1) if idxs else 0
        if total_tabs == 0:
            return

        orig_stage = self._preferred_stage or 12
        stage = orig_stage
        chain: list = []
        while stage > 9:
            # position: bottom=total-1 corresponds to stage 12
            tab_idx = total_tabs - 1 - (12 - stage)
            if tab_idx < 0:
                break
            tab_key = f"tab{tab_idx}"
            tab = by_currency.get(tab_key, {})
            if stage == 12:
                # last tab: unlimited_done required to stage-down
                if tab.get("unlimited_done"):
                    chain.append(f"stage {stage}→tab {tab_key} unlimited_done")
                    stage -= 1
                    continue
            else:
                # non-last: just limited_exhausted (user rule: no unlimited here)
                if tab.get("limited_exhausted"):
                    chain.append(f"stage {stage}→tab {tab_key} limited_exhausted")
                    stage -= 1
                    continue
            break

        if stage != orig_stage:
            self.log(
                f"auto stage-down: {' → '.join(chain)} → "
                f"preferred_stage {orig_stage} → {stage}"
            )
            self._preferred_stage = stage
        else:
            # Log even when no change so user can see we checked
            pts = f"{self._event_points_current}/{self._event_points_total}" \
                if self._event_points_current > 0 else "?"
            self.log(
                f"auto stage-down: stage {stage} tab still active "
                f"(points {pts})"
            )

    def _shop(self, screen: ScreenState) -> Dict[str, Any]:
        """Scan event shop currency tabs, skip exchange traps, optionally
        auto-buy. Transitions to exit when all tabs scanned."""
        self._phase_ticks += 1
        if self._phase_ticks > 120:
            self._persist_shop_state()
            self.sub_state = "exit"
            return action_wait(200, "shop phase tick cap")

        # Click 商店 tab if we're still on Quest / Story
        shop_tab = screen.find_any_text(
            ["商店"], region=(0.15, 0.85, 0.30, 1.0), min_conf=0.55,
        )
        if shop_tab and not self._shop_visited_tabs:
            return action_click_box(shop_tab, "shop: enter shop tab")

        # Left-rail currency tabs. OCR frequently cuts `貨幣` to single
        # `貨` / `货` char; also include full forms and `P` for the
        # points-currency icon tab.
        rail = []
        for b in screen.ocr_boxes:
            if b.confidence < 0.55:
                continue
            if b.cx > 0.12 or b.cy < 0.14 or b.cy > 0.85:
                continue
            t = b.text.strip()
            if t in ("貨", "货", "貨幣", "货币", "P") or t.startswith(("貨", "货", "P")):
                rail.append(b)
        rail.sort(key=lambda b: b.cy)
        # Dedupe vertically close entries (OCR double-hit on same row)
        dedup = []
        for b in rail:
            if not dedup or abs(dedup[-1].cy - b.cy) > 0.04:
                dedup.append(b)

        if not dedup:
            # Not on shop page yet / rail not rendered
            return action_wait(300, "shop: scanning currency rail")

        # Learn total tab count (for "is last tab?" decisions).
        if self._shop_total_tabs < len(dedup):
            self._shop_total_tabs = len(dedup)

        # Pick-order: user rule — farming stage 12 drops the last-tab
        # currency (P 活動點數 at bottom of rail). So when stage ≥ 12,
        # walk BOTTOM-UP. Otherwise walk TOP-DOWN (default reference order).
        pick_order = dedup[::-1] if self._preferred_stage >= 12 else dedup
        # Map cy → stable label (based on original top→bottom rank) so
        # bottom-up traversal still produces deterministic tab IDs.
        cy_to_label = {id(b): f"tab{i}" for i, b in enumerate(dedup)}

        if not self._shop_current_tab:
            for b in pick_order:
                label = cy_to_label[id(b)]
                if label not in self._shop_visited_tabs:
                    self._shop_current_tab = label
                    self._shop_visited_tabs.append(label)
                    self._shop_scroll_attempts = 0
                    return action_click_box(b, f"shop: switch to {label}")
            # All visited
            self._persist_shop_state()
            self.sub_state = "exit"
            return action_wait(250, f"shop: all {len(dedup)} tabs scanned")

        # Scan current view. Strategy:
        #   Phase A (scroll_attempts < 3): collect cards, scroll down to
        #           reveal more, WITHOUT buying. This ensures we see the
        #           whole tab before committing to any purchase, so the
        #           ranker compares unit prices ACROSS the whole tab
        #           (user: "每行每行比较物品单价").
        #   Phase B (scroll_attempts ≥ 3): all items seen, rank + buy.
        cards = self._parse_shop_cards(screen)
        tab_entry = self._shop_state.setdefault(self._shop_current_tab, {
            "items": [], "total_need": 0, "spent_total": 0,
            "skipped_exchange": [], "failed_buys": [],
        })
        tab_entry.setdefault("failed_buys", [])
        # Dedup by (name, unit_cost) — some event shops list a family of
        # 4 items with IDENTICAL visible name (OCR sees 美補(偶像)的神 for
        # all 4 tiers) but distinct costs (200/250/300/350 P). Keying by
        # name alone loses 3 of them. Using (name, cost) keeps all tiers.
        seen = {(it["name"], it["unit_cost"]) for it in tab_entry["items"]}
        skipped = {(it["name"], it["unit_cost"]) for it in tab_entry["skipped_exchange"]}
        for c in cards:
            key = (c["name"], c["unit_cost"])
            if key in seen or key in skipped:
                # Update remaining count (may have decreased after a buy)
                for it in tab_entry["items"]:
                    if it["name"] == c["name"] and it["unit_cost"] == c["unit_cost"]:
                        it["remaining"] = c["remaining"]
                        break
                continue
            if self._looks_like_exchange_trap(c["name"], c["unit_cost"]):
                tab_entry["skipped_exchange"].append(c)
                skipped.add(key)
                self.log(f"shop: skip trap {c['name']!r} cost {c['unit_cost']}")
                continue
            tab_entry["items"].append(c)
            tab_entry["total_need"] += c["unit_cost"] * max(0, c["remaining"])
            seen.add(c["name"])

        # Phase A: keep scrolling to collect all cards before first buy
        # attempt on this tab. Only proceeds to Phase B after 3 scrolls
        # (enough to cover typical 2-row shop layouts).
        phase_a_done = tab_entry.get("phase_a_done", False)
        if not phase_a_done:
            # User rule: "给我们的ocr一点时间扫描". Two kinds of waits:
            #  (a) First entry to tab — always wait 1 tick before scan.
            #  (b) After any scroll — wait 1 tick for the new view to
            #      OCR-stabilize before scrolling again. "疯狂的滑" fix.
            if getattr(self, "_shop_just_scrolled", False):
                self._shop_just_scrolled = False
                # 600ms gives OCR plenty of time to stabilize on the new
                # view — "给 ocr 时间扫描" per user. Worth the extra 200ms
                # since buying wrong items wastes 10× more time.
                return action_wait(600,
                    f"shop: Phase A OCR settle after scroll")
            if self._shop_scroll_attempts == 0 and not tab_entry["items"]:
                self._shop_phase_a_wait = getattr(self, "_shop_phase_a_wait", 0) + 1
                if self._shop_phase_a_wait < 2:
                    return action_wait(600,
                        f"shop: Phase A initial OCR settle on {self._shop_current_tab}")
            self._shop_phase_a_wait = 0

            # USER RULE (2026-05-04): "如果只有一个商品就不用滑了"
            # Single-item tab signature: after settle, 0 anchors found via
            # `可購買 N 次` BUT a `購買` button is visible — typical of the
            # gacha currency tab (1 paid card slot).  Skip the 4×scroll
            # waste and proceed to direct-buy phase.  This also surfaces
            # the "bottom tab is gacha" exception: zero-item tabs are
            # de-prioritized in the buy ranker but still get a buy attempt
            # if a red-dot signals a fresh unbought item.
            if (self._shop_scroll_attempts == 0
                    and not tab_entry["items"]
                    and not cards):
                buy_btn_visible = screen.find_text_one(
                    "購買", region=(0.45, 0.30, 1.0, 0.95), min_conf=0.55,
                )
                if buy_btn_visible:
                    self.log(
                        f"shop: tab {self._shop_current_tab} has 0 "
                        f"`可購買` anchors but `購買` btn visible — "
                        f"single-item tab, skipping Phase A scrolls"
                    )
                    tab_entry["phase_a_done"] = True
                    tab_entry["single_item_tab"] = True
                    self._shop_scroll_attempts = 0
                    return action_wait(200,
                        "shop: Phase A skipped (single-item tab)")
            # 4 scrolls × 7 clicks each ≈ 28 content-units, enough to
            # cover 3-row shops (e.g. tab2 with 信用貨幣 / 羅洪特抄 /
            # 曼陀羅草 rows). Previous 3×4=12 left the bottom row
            # unscanned when tabs had 3+ rows.
            if self._shop_scroll_attempts < 4:
                self._shop_scroll_attempts += 1
                self._shop_just_scrolled = True
                return action_scroll(
                    # Anchor at cx=0.60 (safely inside item grid, away
                    # from the currency display at ~cx=0.87, and away
                    # from the right-edge buy buttons at cx=0.90+).
                    # cy=0.70 keeps the wheel event on item-card body.
                    0.60, 0.70, clicks=-7,
                    reason=f"shop: Phase A scan scroll "
                    f"({self._shop_scroll_attempts}/4, collected "
                    f"{len(tab_entry['items'])} items)",
                )
            # Scroll back up to top to start buying from the actual highest
            # priority items visible.
            tab_entry["phase_a_done"] = True
            self._shop_scroll_attempts = 0
            self.log(
                f"shop: Phase A done, {len(tab_entry['items'])} items collected, "
                f"ranking + buying"
            )
            return action_scroll(
                # Wheel-up (positive) with anchor at middle-body
                # (cx=0.60, cy=0.55) — avoids currency display at top.
                # 40 clicks easily over-reaches 4×7=28 down-scroll;
                # list clamps at top, over-scroll is harmless.
                0.60, 0.55, clicks=40,
                reason="shop: Phase A done, over-scroll to top for buy phase",
            )

        # ── Priority-based buying ──
        spend_ok = (not self._shop_spend_currencies
                    or self._shop_current_tab in self._shop_spend_currencies)
        # NOTE: we used to have a _shop_tab_insufficient shortcut that
        # skipped a whole tab on the first grayed 確認. Removed per user
        # rule: "单价最贵的没搬完就不会去下一个". A grayed confirm means
        # the *specific tier* is unaffordable, but CHEAPER tiers might
        # still be buyable. Each grayed-confirm now just marks that one
        # (name, cost) entry failed; candidate list shrinks naturally
        # until either balance covers something or it's truly empty.
        # Single-item tab (e.g. gacha P): no `可購買` anchor parsing
        # works, so we never populate `tab_entry["items"]`.  Direct-buy
        # mode: find the lone 購買 button, check red-dot, click if has
        # red badge AND balance covers cost.  This is the user's "扫描
        # 红点清理红点" requirement applied to the small-tab path.
        if tab_entry.get("single_item_tab") and spend_ok:
            buy_btn = screen.find_text_one(
                "購買", region=(0.45, 0.30, 1.0, 0.95), min_conf=0.55,
            )
            if buy_btn is None:
                self.log(
                    f"shop[{self._shop_current_tab}]: single-item tab, "
                    f"no 購買 btn visible — moving on"
                )
                self._persist_shop_state()
                self._shop_current_tab = ""
                self._shop_scroll_attempts = 0
                return action_wait(250, "shop: single-item tab done")
            # Red dot = unbought / new.  Probe checkbox left-of-button
            # area (BA shows red dot at button's top-right corner).
            has_red = screen.has_red_badge(buy_btn)
            balance = self._parse_shop_balance(screen)
            # Cost is usually the numeric box just above the 購買 button
            cost_re_box = None
            cost_val = 0
            for b in screen.ocr_boxes:
                if b.confidence < 0.55:
                    continue
                if abs(b.cx - buy_btn.cx) >= 0.08:
                    continue
                if b.cy >= buy_btn.cy or buy_btn.cy - b.cy >= 0.10:
                    continue
                txt = b.text.strip().replace(",", "")
                if txt.isdigit() and len(txt) <= 6:
                    if cost_re_box is None or b.cy > cost_re_box.cy:
                        cost_re_box = b
                        cost_val = int(txt)
            self.log(
                f"shop[{self._shop_current_tab}]: single-item tab "
                f"red_badge={has_red} cost={cost_val} balance={balance}"
            )
            # Decision: skip gacha by default unless red-dot AND user
            # explicitly added this currency to spend list.
            should_buy = (
                has_red
                and cost_val > 0
                and (balance < 0 or cost_val <= balance)
                and self._shop_current_tab in self._shop_spend_currencies
            )
            if should_buy:
                return action_click_box(
                    buy_btn,
                    f"shop: buy single-item ({self._shop_current_tab}, red-dot)"
                )
            self.log(
                f"shop[{self._shop_current_tab}]: skipping single-item tab "
                f"(red={has_red}, in_spend_list={self._shop_current_tab in self._shop_spend_currencies})"
            )
            self._persist_shop_state()
            self._shop_current_tab = ""
            self._shop_scroll_attempts = 0
            return action_wait(250, "shop: single-item tab skipped")

        if self._shop_auto_buy and spend_ok:
            ranked = self._rank_shop_items(tab_entry["items"], self._shop_current_tab)
            # Step 1: strip sold-out, unknown-cost, traps, known-failed.
            # Don't apply balance filter yet — we need the TRUE top
            # (highest unit cost) before deciding affordability.
            failed = {tuple(x) if isinstance(x, (list, tuple)) else (x, 0)
                      for x in tab_entry.get("failed_buys", [])}
            playable = [
                c for c in ranked
                if c["remaining"] > 0
                and c["unit_cost"] > 0     # OCR-missed cost → skip
                and not self._looks_like_exchange_trap(c["name"], c["unit_cost"])
                and (c["name"], c["unit_cost"]) not in failed
            ]
            # Step 2: user rule — "先找整个tab单价最高, 跟货币比, 够买买,
            # 不够就走". Pick ONLY the top (cost DESC), compare to balance.
            # If top too expensive → abandon tab (not try cheaper tiers).
            balance = self._parse_shop_balance(screen)
            candidates: list = []
            # Always log balance + top for transparency — user wants to
            # see in the trajectory whether we know our currency.
            playable_summary = ", ".join(
                f"{c['name']}@{c['unit_cost']}×{c['remaining']}"
                for c in playable[:3]
            ) or "(none)"
            self.log(
                f"shop[{self._shop_current_tab}]: balance={balance}, "
                f"top3=[{playable_summary}]"
            )
            if playable:
                top = playable[0]
                # User rule: "都知道是0货币还在那一个个商品试" — when
                # balance == 0 the bot was still iterating items because
                # balance<0 (unknown) and balance==0 fell into the same
                # permissive path.  Split them: explicit 0 = abandon tab;
                # unknown (-1) wait one extra tick to let OCR re-fire,
                # then abandon if still unknown.
                if balance == 0:
                    self.log(
                        f"shop[{self._shop_current_tab}]: balance=0 — "
                        f"abandoning tab (nothing to spend)"
                    )
                    self._shop_current_tab = ""
                    return action_wait(150, "shop: zero balance, next tab")
                if balance < 0:
                    self._shop_balance_unknown_ticks = getattr(
                        self, "_shop_balance_unknown_ticks", 0
                    ) + 1
                    if self._shop_balance_unknown_ticks < 3:
                        return action_wait(
                            400, "shop: balance OCR unclear, retrying"
                        )
                    self.log(
                        f"shop[{self._shop_current_tab}]: balance still "
                        f"unknown after 3 ticks — abandoning tab to be safe"
                    )
                    self._shop_current_tab = ""
                    self._shop_balance_unknown_ticks = 0
                    return action_wait(150, "shop: balance unknown, next tab")
                # balance is a real positive number — apply the user's
                # buy-top-or-skip rule.
                self._shop_balance_unknown_ticks = 0
                if top["unit_cost"] <= balance:
                    candidates = [top]
                    self.log(
                        f"shop: BUY top '{top['name']}' @ {top['unit_cost']} "
                        f"(balance {balance} covers it)"
                    )
                else:
                    self.log(
                        f"shop: WALK top '{top['name']}' @ {top['unit_cost']} "
                        f"> balance {balance}"
                    )
                    # User rule: can't afford top → abandon tab, no substitutes
                    candidates = []
                    self._shop_current_tab = ""
                    return action_wait(150, f"shop: top unaffordable, next tab")
            # Last-tab phase A→B: unlimited items unlocked after limited
            # phase is exhausted.
            if (not playable
                    and self._is_last_tab(self._shop_current_tab)
                    and not self._shop_last_tab_limited_done):
                self._shop_last_tab_limited_done = True
                self.log(
                    f"shop: last tab {self._shop_current_tab!r} limited items "
                    f"exhausted, unlocking unlimited-item phase"
                )
                ranked = self._rank_shop_items(tab_entry["items"], self._shop_current_tab)
                playable = [
                    c for c in ranked
                    if c["remaining"] > 0 and c["unit_cost"] > 0
                    and not self._looks_like_exchange_trap(c["name"], c["unit_cost"])
                    and (c["name"], c["unit_cost"]) not in failed
                ]
                if playable:
                    top = playable[0]
                    if balance < 0 or top["unit_cost"] <= balance:
                        candidates = [top]
            if candidates:
                target = candidates[0]
                buy_btn = self._find_buy_button_for_card(screen, target)
                if buy_btn is None:
                    # Top item not visible on current view. User rule
                    # ("不够就走"): top is our only shot. If the top's
                    # button isn't findable after over-scroll-to-top,
                    # treat as "can't buy top" → abandon tab (don't
                    # substitute cheaper items).
                    self.log(
                        f"shop: top '{target['name']}' @ {target['unit_cost']} "
                        f"not visible on screen → walk tab (strict top-only)"
                    )
                    self._shop_current_tab = ""
                    self._shop_buy_retry_count = 0
                    return action_wait(150, f"shop: top not findable, next tab")
                if buy_btn is not None:
                    # Track this buy attempt — next tick we'll verify the
                    # remaining count decreased; if not, mark as failed.
                    self._shop_last_buy_target = target["name"]
                    self._shop_last_buy_cost = target["unit_cost"]
                    self._shop_last_buy_remaining = target["remaining"]
                    self._shop_buy_retry_count = (
                        self._shop_buy_retry_count
                        if self._shop_last_buy_target == target["name"]
                        else 0
                    )
                    if self._shop_buy_retry_count >= 2:
                        # Already retried this card 2+ times with no decrease
                        # → mark failed (insufficient currency / confirm loop)
                        tab_entry["failed_buys"].append((target["name"], target["unit_cost"]))
                        self._shop_buy_retry_count = 0
                        self.log(
                            f"shop: {target['name']!r} buy failed 3x "
                            f"(likely insufficient currency), skipping"
                        )
                        return action_wait(200, f"shop: giving up on {target['name']}")
                    self._shop_buy_retry_count += 1
                    tab_entry["spent_total"] = tab_entry.get("spent_total", 0) + 1
                    self.log(
                        f"shop: buy {target['name']!r} "
                        f"(cost {target['unit_cost']}, {target['remaining']} left)"
                    )
                    # Enter dialog FSM — next ticks handle MAX + confirm
                    # (with grayed-button detection).
                    self._shop_buy_dialog_stage = 1
                    self._shop_buy_dialog_ticks = 0
                    return action_click_box(
                        buy_btn, f"shop: buy {target['name']} ({self._shop_current_tab})"
                    )

        # Phase B fallback: no actionable items. Mark tab as fully
        # exhausted (for next-run stage-down decision).
        #   - "limited_exhausted" = all limited items either sold out or
        #      priced above balance — no progression possible here
        #   - "unlimited_done" = unlimited items (if applicable) also done
        # Persist to shop state file; next run can read and pick a new
        # farming stage when the last tab is flagged done.
        tab_entry["limited_exhausted"] = True
        is_last = self._is_last_tab(self._shop_current_tab)
        if is_last and self._shop_last_tab_limited_done:
            tab_entry["unlimited_done"] = True
        self.log(
            f"shop tab {self._shop_current_tab}: nothing more buyable — "
            f"{len(tab_entry['items'])} items seen, "
            f"{tab_entry.get('spent_total', 0)} buys, "
            f"total need {tab_entry['total_need']} "
            f"(skipped {len(tab_entry['skipped_exchange'])} traps, "
            f"{len(tab_entry.get('failed_buys', []))} failed)"
            f"{' [LAST TAB EXHAUSTED — consider farming stage-1]' if is_last and tab_entry.get('unlimited_done') else ''}"
        )
        self._shop_current_tab = ""
        return action_wait(150, "shop: tab done, pick next currency tab")

    # P-currency (last tab) explicit priority per user spec:
    #   神名文字碎片 (per-unit cost highest among limiteds, most useful)
    #   → 青輝石 (premium gem, capped low)
    #   → 奧秘之書 (upgrade book)
    #   → 信用貨幣 (credits)
    #   → furniture items (toggleable, see event_shop_furniture_first)
    #   → 角色神名文字 / 美補的神名文字 (character-specific, treated as trap)
    _P_SHOP_PRIORITY = (
        "神名文字碎片",
        "青輝石",
        "奧秘之書", "奥秘之书",
        "信用貨幣", "信用货币",
    )

    # Furniture detection: items with an interactive-furniture badge icon
    # (blue square w/ smiling student face in top-right of item art).
    # Name-based heuristic for now — the icon is a graphic, not OCR-able.
    # Template at data/references/ui_markers/furniture_badge.png available
    # for future template-match upgrade. Maintain a keyword list of
    # common furniture nouns seen in event shops.
    _FURNITURE_NAME_HINTS = (
        "組合", "组合",        # 可愛器皿組合, 針線組合
        "手帕", "掛件", "挂件",
        "擺設", "摆设",
        "套組", "套组",
        "應援棒", "应援棒",    # 各色應援棒 → cheer sticks (cafe decor)
        "家具", "装飾", "裝飾",
    )

    @classmethod
    def _is_furniture_item(cls, card: dict) -> bool:
        name = card.get("name", "")
        return any(h in name for h in cls._FURNITURE_NAME_HINTS)

    def _is_last_tab(self, tab_label: str) -> bool:
        """Is this the bottom-most currency tab (P-points shop)?

        Use learned rail size when available; fall back to "tab index ≥ 2"
        heuristic for ≥3-tab shops.
        """
        if not tab_label.startswith("tab"):
            return False
        try:
            idx = int(tab_label[3:])
        except Exception:
            return False
        if self._shop_total_tabs > 0:
            return idx == self._shop_total_tabs - 1
        return idx >= 2   # heuristic fallback

    def _claim_tasks(self, screen: ScreenState) -> Dict[str, Any]:
        """Auto-claim 活動任務 rewards when 任務 nav button shows a red dot.

        Flow:
          stage 0: click 任務 nav (event page bottom-right area, cx≈0.31,
                    cy≈0.93). Wait for 活動任務 page.
          stage 1: click 全部領取 (yellow button bottom-right, ~0.95, 0.95).
          stage 2: reward popup handled by _handle_common_popups (確認).
          stage 3: BACK to event page. Mark task_claim_done, set sub=enter
                    so dispatch resumes (re-scan nav, proceed to farming).
        """
        self._phase_ticks += 1
        # Hard timeout
        if self._phase_ticks > 40:
            self.log("claim_tasks: timeout, back to enter")
            self._task_claim_done = True
            self.sub_state = "enter"
            self._phase_ticks = 0
            return action_back("claim_tasks: timeout")

        # Detect which screen we're on via header OCR
        on_task_page = screen.find_any_text(
            ["活動任務", "活动任务", "活動仼務"],
            region=(0.0, 0.0, 0.40, 0.12), min_conf=0.55,
        )

        stage = getattr(self, "_task_claim_stage", 0)

        if stage == 0:
            if on_task_page:
                # Already on task page — skip click, go to claim stage
                self._task_claim_stage = 1
                return action_wait(200, "claim_tasks: on task page, going to claim")
            # Click 任務 nav button
            nav_task = screen.find_any_text(
                ["任務", "任务"], region=(0.25, 0.88, 0.40, 0.99),
                min_conf=0.55,
            )
            if nav_task:
                self._task_claim_stage = 1
                return action_click_box(nav_task, "claim_tasks: open 活動任務")
            return action_wait(300, "claim_tasks: looking for 任務 nav button")

        if stage == 1:
            # Look for 全部領取 button (yellow, bottom-right of task page)
            claim_all = screen.find_any_text(
                ["全部領取", "全部领取", "全部領收", "全部收取"],
                region=(0.80, 0.85, 1.0, 1.0), min_conf=0.55,
            )
            if claim_all:
                self._task_claim_stage = 2
                self.log("claim_tasks: clicking 全部領取")
                return action_click_box(claim_all, "claim_tasks: 全部領取")
            # Fallback: maybe per-item 領取 on first row
            individual = screen.find_any_text(
                ["領取", "领取"], region=(0.80, 0.20, 1.0, 0.60),
                min_conf=0.55,
            )
            if individual:
                self._task_claim_stage = 2
                return action_click_box(individual, "claim_tasks: 領取 (individual)")
            # Nothing claimable → dialog shows "all claimed" → back out
            self._task_claim_stage = 3
            return action_wait(300, "claim_tasks: no claim button, backing out")

        if stage == 2:
            # Reward popup likely showing — common popup handler dismisses
            # it via 確認. Give a tick for animation, then check if we're
            # back on task page.
            if on_task_page:
                self._task_claim_stage = 3
                return action_wait(200, "claim_tasks: claim done, back to event")
            return action_wait(400, "claim_tasks: waiting for reward popup close")

        # stage 3: back to event page. BA's 活動任務 page doesn't always
        # respond to the Android BACK key, so try the top-left ← arrow
        # first (hardcoded position, always present on BA sub-pages).
        if on_task_page:
            self._task_claim_back_attempts = getattr(self, "_task_claim_back_attempts", 0) + 1
            if self._task_claim_back_attempts <= 3:
                self.log(
                    f"claim_tasks: still on 活動任務 (attempt "
                    f"{self._task_claim_back_attempts}/3), clicking ← arrow"
                )
                return action_click(0.04, 0.04, "claim_tasks: click ← arrow top-left")
            # 3 attempts failed — fall through to Android BACK as last resort
        self._task_claim_done = True
        self.sub_state = "enter"
        self._task_claim_stage = 0
        self._task_claim_back_attempts = 0
        self._phase_ticks = 0
        return action_back("claim_tasks: done, back to event (fallback)")

    def _scan_event_nav_red_badges(self, screen: ScreenState) -> None:
        """Scan the event page's bottom nav (劇情 / 商店 / 任務 / 後日谈)
        for red-dot badges and log findings. Saves to
        ``self._event_nav_badges`` dict for the pipeline / downstream
        skills to consume. Purely diagnostic for now — doesn't navigate.
        """
        nav_names = ("劇情", "剧情", "商店", "任務", "任务", "後日谈", "后日谈")
        found: dict = {}
        for b in screen.ocr_boxes:
            if b.confidence < 0.55:
                continue
            if b.cy < 0.85:           # nav bar is at the bottom
                continue
            txt = b.text.strip()
            matched_name = None
            for nav in nav_names:
                if nav in txt:
                    matched_name = nav
                    break
            if not matched_name:
                continue
            has_red = screen.has_red_badge(b)
            has_yellow = screen.has_yellow_badge(b) if not has_red else False
            if has_red or has_yellow:
                found[matched_name] = "red" if has_red else "yellow"
                self.log(
                    f"event-nav badge: {matched_name!r} has "
                    f"{'RED (unclaimed)' if has_red else 'YELLOW (actionable)'} dot"
                )
        self._event_nav_badges = found
        if found:
            self.log(f"event-nav badges summary: {found}")

    def _parse_activity_points(self, screen: ScreenState):
        """Parse 活動點數 progress bar `X/Y` (e.g. '5663/15000').

        Appears at bottom of event page (cx≈0.68, cy≈0.92). Returns
        (current, total) or (None, None) if not found.
        """
        import re as _re
        for b in screen.ocr_boxes:
            if b.confidence < 0.9:
                continue
            if not (0.55 <= b.cx <= 0.85 and 0.86 <= b.cy <= 0.96):
                continue
            m = _re.match(r"^(\d{1,6})\s*/\s*(\d{1,6})$", b.text.strip())
            if m:
                return int(m.group(1)), int(m.group(2))
        return None, None

    def _parse_shop_balance(self, screen: ScreenState) -> int:
        """Read the current-currency balance shown at the top-right of
        the shop page. Returns -1 if not found.

        OCR sees integer with optional thousand-separator comma (e.g.
        '2,285' / '283' / '1,768') at ~(0.82-0.92, 0.08-0.15).
        """
        for b in screen.ocr_boxes:
            if b.confidence < 0.8:
                continue
            if not (0.80 <= b.cx <= 0.95 and 0.07 <= b.cy <= 0.17):
                continue
            stripped = b.text.strip().replace(",", "").replace("，", "")
            if re.fullmatch(r"\d{1,7}", stripped):
                return int(stripped)
        return -1

    def _rank_shop_items(self, cards: list, tab_label: str) -> list:
        """Sort cards by purchase priority — user rule:
        "find the single HIGHEST-unit-cost item in the whole tab, buy
         that; when sold out move to next most expensive".

        Simple: cost DESC across entire tab.

        Applied filters:
          - non-last tabs: drop unlimited items (remaining ≥ threshold)
          - last tab phase A: drop unlimited items. Phase B allows them.
          - Furniture items ranked AFTER materials unless
            `event_shop_furniture_first` is set.
        """
        def category(c):
            return 1 if self._is_furniture_item(c) else 0
        furn_bias = -1 if self._shop_furniture_first else 1

        if self._is_last_tab(tab_label):
            if not self._shop_last_tab_limited_done:
                cards = [c for c in cards if not self._is_unlimited_item(c)]
        else:
            cards = [c for c in cards if not self._is_unlimited_item(c)]

        # Single ordering key for all tabs: (furniture bias, -cost).
        # Highest cost first; furniture demoted unless toggle says otherwise.
        return sorted(
            cards,
            key=lambda c: (furn_bias * category(c), -c["unit_cost"]),
        )

    def _handle_shop_buy_dialog(self, screen: ScreenState):
        """Shop purchase dialog FSM (replaces relying on _handle_common_popups).

        After clicking 購買 on a card, BA opens a quantity-selector popup
        with MIN / current / MAX / 確認 / 取消 buttons. Flow:

          stage 1: click MAX (if N≥2 purchases desired), else skip to 2
          stage 2: sample 確認 button color:
                     - grayed → can't afford → click 取消, mark tab
                                insufficient, abort current item
                     - normal → click 確認 → purchase → stage 0

        Returns action dict or None (None = dialog not open, fall through).
        Timeout: 15 ticks/stage → force stage 0, mark item failed.
        """
        self._shop_buy_dialog_ticks += 1
        # Tight timeout per stage (8 ticks ≈ 3s at default step_sleep).
        # If dialog never opens (click didn't register / game threw
        # something else on screen), mark item as failed and reset so we
        # move on to the next candidate instead of burning time.
        if self._shop_buy_dialog_ticks > 8:
            self.log(
                f"shop dialog timeout (stage={self._shop_buy_dialog_stage}) — "
                f"marking {self._shop_last_buy_target!r} failed, reset"
            )
            if self._shop_last_buy_target and self._shop_current_tab:
                entry = self._shop_state.setdefault(self._shop_current_tab, {})
                entry.setdefault("failed_buys", []).append(
                    (self._shop_last_buy_target, self._shop_last_buy_cost))
            self._shop_buy_dialog_stage = 0
            self._shop_buy_dialog_ticks = 0
            self._shop_buy_retry_count = 0
            return None

        # Detect dialog presence via 確認 button in the popup band
        confirm_btn = screen.find_any_text(
            ["確認", "确认", "確定", "确定", "確", "确"],
            region=(0.30, 0.55, 0.80, 0.85), min_conf=0.55,
        )
        cancel_btn = screen.find_any_text(
            ["取消"], region=(0.20, 0.55, 0.55, 0.85), min_conf=0.55,
        )

        # Dialog-open signature: MAX/MIN/確認 together. If at stage 1 and
        # neither MAX nor 確認 visible for 3 ticks, dialog didn't open.
        dialog_signal = (confirm_btn is not None
                         or screen.find_any_text(["MAX", "MIN"],
                                                 region=(0.55, 0.30, 1.0, 0.65),
                                                 min_conf=0.55) is not None)

        if self._shop_buy_dialog_stage == 1:
            if not dialog_signal and self._shop_buy_dialog_ticks >= 3:
                # Click didn't open dialog — abort early.
                self.log(
                    f"shop: dialog did not open for "
                    f"{self._shop_last_buy_target!r} after 3 ticks, aborting"
                )
                if self._shop_last_buy_target and self._shop_current_tab:
                    entry = self._shop_state.setdefault(self._shop_current_tab, {})
                    entry.setdefault("failed_buys", []).append(
                    (self._shop_last_buy_target, self._shop_last_buy_cost))
                self._shop_buy_dialog_stage = 0
                self._shop_buy_dialog_ticks = 0
                return None
            # Try to click MAX to bulk-purchase when remaining allows
            max_btn = screen.find_any_text(
                ["MAX", "最大"], region=(0.55, 0.30, 1.0, 0.65), min_conf=0.55,
            )
            if max_btn:
                self._shop_buy_dialog_stage = 2
                self._shop_buy_dialog_ticks = 0
                return action_click_box(max_btn, "shop: click MAX to bulk purchase")
            # No MAX button — this may be a single-purchase item (remaining=1)
            # → jump to confirm stage
            if confirm_btn:
                self._shop_buy_dialog_stage = 2
                self._shop_buy_dialog_ticks = 0
                return action_wait(150, "shop: no MAX, going to confirm stage")
            return action_wait(200, "shop: waiting for quantity dialog")

        if self._shop_buy_dialog_stage == 2:
            if confirm_btn is None:
                # Dialog closed? Reset to idle.
                self._shop_buy_dialog_stage = 0
                self._shop_buy_dialog_ticks = 0
                return None
            # Check if 確認 is grayed — means insufficient currency for
            # this particular (name, cost) tier. Mark just this tier
            # failed; cheaper tiers of the same name may still be
            # affordable and will get their turn next iteration.
            if screen.is_button_grey(confirm_btn.cx, confirm_btn.cy):
                if self._shop_last_buy_target:
                    tab_entry = self._shop_state.setdefault(
                        self._shop_current_tab, {"failed_buys": []})
                    tab_entry.setdefault("failed_buys", []).append(
                        (self._shop_last_buy_target, self._shop_last_buy_cost))
                # Track consecutive grey-confirm failures in this tab.
                # User feedback (2026-05-13): "都知道是 0 货币还在那一个
                #个商品试" — when balance OCR fails (returns -1), the
                # bot iterates EVERY cheaper item on the tab even if
                # the underlying currency is genuinely empty.  3
                # consecutive greyed confirms is a strong signal that
                # the entire tab is unaffordable; abandon it.
                self._shop_tab_grey_streak = getattr(
                    self, "_shop_tab_grey_streak", 0
                ) + 1
                if self._shop_tab_grey_streak >= 3:
                    self.log(
                        f"shop: 3 consecutive 確認 greyed in tab "
                        f"{self._shop_current_tab} — entire tab "
                        f"unaffordable, abandoning"
                    )
                    self._shop_tab_grey_streak = 0
                    self._shop_buy_dialog_stage = 0
                    self._shop_buy_dialog_ticks = 0
                    self._shop_buy_retry_count = 0
                    self._shop_current_tab = ""  # pick next tab
                    if cancel_btn:
                        return action_click_box(cancel_btn, "shop: tab unaffordable, cancel + next tab")
                    return action_back("shop: tab unaffordable, ESC + next tab")
                self.log(
                    f"shop: 確認 grayed for {self._shop_last_buy_target!r}"
                    f"@{self._shop_last_buy_cost} — cancel, try cheaper tier "
                    f"(grey streak {self._shop_tab_grey_streak}/3)"
                )
                self._shop_buy_dialog_stage = 0
                self._shop_buy_dialog_ticks = 0
                self._shop_buy_retry_count = 0
                if cancel_btn:
                    return action_click_box(cancel_btn, "shop: cancel grayed confirm")
                return action_back("shop: back out of grayed confirm dialog")
            # Confirm not grey — reset the grey streak counter.
            self._shop_tab_grey_streak = 0
            # Normal 確認 — click to complete purchase
            self._shop_buy_dialog_stage = 0
            self._shop_buy_dialog_ticks = 0
            self.log(f"shop: confirm purchase of {self._shop_last_buy_target!r}")
            return action_click_box(confirm_btn, "shop: confirm purchase")

        return None

    def _find_buy_button_for_card(self, screen: ScreenState, card: dict):
        """Locate the 購買 button whose column aligns with this card.

        The 購買 button sits directly below the card's cost digit (which
        sits below the `可購買 N 次` anchor). We re-anchor via name OCR
        and then find 購買 at same cx, bounded below.
        """
        # Find the card's name OCR box (top of card)
        name_box = None
        for b in screen.ocr_boxes:
            if b.confidence < 0.55:
                continue
            if card["name"] in b.text or b.text.strip() == card["name"]:
                name_box = b
                break
        if name_box is None:
            return None
        col_x = name_box.cx
        # Find 購買 OCR box in same column, below the name
        buy_cands = []
        for b in screen.ocr_boxes:
            if b.confidence < 0.55:
                continue
            if abs(b.cx - col_x) >= 0.10:
                continue
            if b.cy <= name_box.cy + 0.10:
                continue
            if b.text.strip() not in ("購買", "购买"):
                continue
            buy_cands.append(b)
        if not buy_cands:
            return None
        # Prefer the CLOSEST 購買 below (there may be multiple cards in a
        # column for some layouts; nearest-below is the right card).
        buy_cands.sort(key=lambda b: b.cy - name_box.cy)
        return buy_cands[0]

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
