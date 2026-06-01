"""CafeSkill — Blue Archive cafe daily routine (pure-YOLO rewrite).

Verified flow (interactive probe 2026-06-01), all clicks resolved through
YOLO cls (ui_classes) or the emoticon / fused_avatar detectors — NO OCR, NO
hardcoded button positions (only relative gesture params: the headpat
cx-offset and the swipe start/end points).

State machine
-------------
enter     lobby → click NAV_CAFE → wait for Cafe page.
earnings  click CAFE_EARNINGS → popup → click an active CLAIM cls
          (CLAIM_REWARD_YELLOW / CLAIM_YELLOW). 0% earnings = no claim cls →
          bail after a few ticks (never dead-wait).
invite    open CAFE_INVITE_TICKET → MomoTalk list. Each row = an avatar
          (fused_avatar model, model_tag=="avatar", cls_name=中文角色名,
          cx≈0.35) + a CAFE_INVITE_BTN (cx≈0.60). Pair avatar.cy↔button.cy,
          click the target row's invite button → BTN_CONFIRM in the
          "邀請XXX到咖啡廳" dialog. Up to 2 invites per cafe (1F target, 2F
          target from config). Scroll to find the target; fall back to the
          first rows.
headpat   emoticon model 'Emoticon_Action' (conf≥0.55). Click cx+0.025
          (student body, right of the bubble). Exclude the top-left
          指定/隨機訪問 false-positive zone (cx<0.27 & cy<0.34). Dedup recent
          clicks. loop-until-dry (3 empty frames), then pan left / right to
          sweep the rest of the floor.
switch    CAFE_MOVE_2F → 2F (may pop a 訪問學生目錄 tutorial → BTN_CONFIRM).
          On 2F the button reads CAFE_MOVE_1F. Repeat invite + headpat on 2F.
exit      is_lobby → done; else BTN_HOME / BTN_BACK.

Detectors (set by pipeline.SKILL_YOLO_MAP["Cafe"] = "ui+cafe+avatar"):
  ui      — all the UC.* button classes (find_cls / find_all_cls).
  cafe    — emoticon model, single class 'Emoticon_Action' (headpat marker).
  avatar  — fused_avatar (251 student heads), model_tag=="avatar".
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from brain.skills.base import (
    BaseSkill, ScreenState, YoloBox,
    action_click, action_click_box,
    action_wait, action_back, action_done, action_swipe,
)
from brain.skills import ui_classes as UC

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_CAFE_STATE_FILE = _DATA_DIR / "cafe_state.json"
_APP_CONFIG_FILE = _DATA_DIR / "app_config.json"

# ── tuning knobs ─────────────────────────────────────────────────────────
_CLS_CONF = 0.30            # default UI cls confidence floor
_EMOTICON_CONF = 0.55       # headpat marker (emoticon model) — probe-confirmed
_AVATAR_CONF = 0.25         # fused_avatar row-head confidence (registry default)
_HEADPAT_DX = 0.025         # click this much right of the bubble = student body
_HEADPAT_DEDUP_DIST = 0.06  # skip clicks within this of a recent headpat
_HEADPAT_DEDUP_KEEP = 4     # remember the last N headpat coords for dedup
_HEADPAT_DRY_FRAMES = 3     # consecutive empty frames ⇒ current view cleared
_MAX_HEADPATS_PER_FLOOR = 12  # safety helmet (probe: ~12 students/floor)
_MAX_PANS_PER_FLOOR = 6       # safety helmet on camera sweeps
_INVITES_PER_FLOOR = 1        # invite 1 student per floor (2 total per cafe)
_INVITE_MAX_SCROLLS = 12      # rows to scroll hunting the target
_INVITE_SWIPE_SETTLE = 3      # ticks to wait after a list swipe (anim settle)
_ROW_PAIR_DY = 0.06           # avatar.cy ↔ invite-button.cy max gap = same row

# Top-left false-positive zone: 指定訪問/隨機訪問 buttons get mis-detected as
# Emoticon_Action (fixed assault per probe). Anything here is NOT a student.
_FP_ZONE = (0.27, 0.34)       # cx < 0.27 AND cy < 0.34 ⇒ reject

# Per-sub-state tick budgets — every phase is bounded, never dead-waits.
_ENTER_MAX = 25
_EARNINGS_MAX = 18
_INVITE_MAX = 45
_SWITCH_MAX = 18
_EXIT_MAX = 14


def _game_day() -> str:
    """BA game-day ISO date (resets 04:00 local) — invite-state key."""
    return (datetime.now() - timedelta(hours=4)).date().isoformat()


def _load_cafe_state() -> dict:
    try:
        if _CAFE_STATE_FILE.exists():
            return json.loads(_CAFE_STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_cafe_state(state: dict) -> None:
    try:
        _CAFE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CAFE_STATE_FILE.write_text(
            json.dumps(state, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        pass


def _load_invite_targets() -> List[str]:
    """Read the cafe invite character list from app_config profile.

    Same pattern as BountySkill._load_enabled_branches: app_config.json →
    active_profile → profiles[active].cafe_invite_targets. Returns a list of
    中文角色名 matching fused_avatar cls_name (e.g. "莉央(战斗)"). Index 0 is
    the 1F target, index 1 the 2F target; extra entries are accepted as
    further fallbacks. Empty list ⇒ caller invites the first visible rows.
    """
    try:
        if not _APP_CONFIG_FILE.exists():
            return []
        data = json.loads(_APP_CONFIG_FILE.read_text("utf-8"))
        active = data.get("active_profile", "default")
        profile = (data.get("profiles") or {}).get(active, {})
        raw = profile.get("cafe_invite_targets")
        if not isinstance(raw, list):
            return []
        out: List[str] = []
        for item in raw:
            name = str(item or "").strip()
            if name and name not in out:
                out.append(name)
        return out
    except Exception:
        return []


class CafeSkill(BaseSkill):
    # Lobby cafe entries that carry a red/yellow dot when there's something
    # to do (earnings ready / invite slot open). Drives should_run gating.
    _LOBBY_DOT_ENTRIES = [UC.NAV_CAFE, UC.CAFE_INVITE_TICKET, UC.CAFE_EARNINGS]

    def should_run(self, screen: ScreenState) -> bool:
        return self.dot_on_entry(screen, self._LOBBY_DOT_ENTRIES)

    def __init__(self):
        super().__init__("Cafe")
        # 1F invite(~12)+headpat(~25)+switch(~5)+2F invite(~25)+headpat(~25)
        # ≈ 95 ticks worst-case; 160 leaves slack for retries/loads.
        self.max_ticks = 160
        self._init_state()

    # ── state init / reset ────────────────────────────────────────────────

    def _init_state(self) -> None:
        self._phase_ticks: int = 0          # ticks spent in current sub_state
        self._enter_attempts: int = 0
        # earnings
        self._earnings_done: bool = False
        # invite
        self._invite_targets: List[str] = []
        self._invited: Set[str] = set()      # cls_names invited this run
        self._invite_floor_done: bool = False  # this floor's invite finished
        self._invite_stage: int = 0          # 0=open ticket 1=list 2=confirm
        self._invite_scrolls: int = 0
        self._invite_settle: int = 0         # post-swipe settle countdown
        self._invite_last_sig: str = ""
        self._invite_sig_repeat: int = 0
        self._invite_retry_btn: Optional[Tuple[float, float]] = None
        # headpat
        self._pat_count: int = 0
        self._empty_frames: int = 0
        self._pan_count: int = 0
        self._pan_dir: int = 0               # 0=none yet, 1=swept left, 2=right
        self._pat_settle: int = 0            # post-pat / post-pan settle
        self._recent_pats: List[Tuple[float, float]] = []
        # floor bookkeeping
        self._on_2f: bool = False

    def reset(self) -> None:
        super().reset()
        self._init_state()
        self._invite_targets = _load_invite_targets()
        if self._invite_targets:
            self.log(f"cafe invite targets: {self._invite_targets}")
        else:
            self.log("no cafe_invite_targets configured — will invite first rows")
        # Restore today's invited set so a retry after a timeout doesn't waste
        # a ticket re-inviting the same student. Auto-expires at 04:00.
        try:
            saved = _load_cafe_state()
            if saved.get("game_day") == _game_day():
                restored = [str(n) for n in saved.get("invited_names", []) if n]
                if restored:
                    self._invited = set(restored)
                    self.log(f"restored invited from disk: {sorted(self._invited)}")
        except Exception:
            pass

    # ── shared cls helpers ────────────────────────────────────────────────

    def _close_x(self, screen: ScreenState,
                 region=(0.56, 0.04, 0.94, 0.30)) -> Optional[YoloBox]:
        return self.find_cls(screen, UC.BTN_CLOSE_X, conf=_CLS_CONF, region=region)

    def _confirm_btn(self, screen: ScreenState,
                     region=(0.42, 0.55, 0.78, 0.85)) -> Optional[YoloBox]:
        """The blue 確認 button in a center-bottom dialog (invite/tutorial)."""
        return self.find_cls(screen, UC.BTN_CONFIRM, conf=_CLS_CONF, region=region)

    def _is_cafe(self, screen: ScreenState) -> bool:
        return self.detect_screen_yolo(screen) == "Cafe"

    def _invite_list_open(self, screen: ScreenState) -> bool:
        """MomoTalk invite list = per-row CAFE_INVITE_BTN in the right column."""
        return self.find_cls(
            screen, UC.CAFE_INVITE_BTN, conf=_CLS_CONF, region=(0.50, 0.20, 0.72, 0.92)
        ) is not None

    def _save_invited(self) -> None:
        _save_cafe_state({
            "game_day": _game_day(),
            "invited_names": sorted(self._invited),
        })

    def _goto(self, sub_state: str) -> None:
        """Switch sub_state and reset the per-phase tick counter."""
        self.sub_state = sub_state
        self._phase_ticks = 0

    # ── tick: global popup guards + dispatch ──────────────────────────────

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1
        self._phase_ticks += 1

        if self.ticks >= self.max_ticks:
            self.log("timeout, exiting")
            return action_done("cafe timeout")

        # ── popups that can appear in any sub_state (pure YOLO) ──

        # Reward-result popup ("獲得獎勵") after a claim — tap to dismiss.
        got_reward = self.find_cls(screen, UC.GOT_REWARD, conf=_CLS_CONF)
        if got_reward is not None:
            self.log("reward-result popup, dismissing (YOLO 获得奖励)")
            return action_click_box(got_reward, "dismiss reward result")

        # Full-screen bond / region level-up overlay — tap anywhere.
        levelup = self.find_cls(
            screen, [UC.BOND_LEVELUP, UC.REGION_LEVELUP], conf=_CLS_CONF
        )
        if levelup is not None:
            self.log(f"level-up overlay ({levelup.cls_name}), tapping to dismiss")
            return action_click(0.5, 0.5, "dismiss level-up overlay")

        # Tutorial popup (2F first visit, "訪問學生目錄/說明") — close via
        # BTN_CONFIRM. Only outside invite (the invite list reuses the center).
        if self.sub_state in ("switch", "headpat2"):
            tut_confirm = self._confirm_btn(screen, region=screen.CENTER)
            tut_close = self._close_x(screen)
            # Only treat as tutorial when NOT on a real cafe page and a
            # confirm/close sits center — avoids eating cafe-main buttons.
            if (tut_confirm is not None and not self._is_cafe(screen)
                    and not self._invite_list_open(screen)):
                self.log("dismissing 2F tutorial popup (YOLO 确认键)")
                return action_click_box(tut_confirm, "dismiss 2F tutorial")
            if (tut_close is not None and not self._is_cafe(screen)
                    and self.sub_state == "switch"):
                return action_click_box(tut_close, "close 2F tutorial X")

        # Loading
        if screen.is_loading():
            return action_wait(700, "cafe loading")

        # ── state machine ──
        if self.sub_state == "":
            self._goto("enter")

        handler = {
            "enter": self._enter,
            "earnings": self._earnings,
            "invite": self._invite,
            "headpat": self._headpat,
            "switch": self._switch_floor,
            "headpat2": self._headpat,
            "exit": self._exit,
        }.get(self.sub_state)
        if handler is None:
            return action_wait(300, "cafe unknown state")
        return handler(screen)

    # ── enter ─────────────────────────────────────────────────────────────

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        self._enter_attempts += 1
        page = self.detect_screen_yolo(screen)

        if page == "Cafe":
            self._enter_attempts = 0
            self.log("inside cafe, claiming earnings")
            self._goto("earnings")
            return action_wait(400, "entered cafe")

        if page == "Lobby":
            nav = self.click_cls(screen, UC.NAV_CAFE, "click cafe nav", conf=_CLS_CONF)
            if nav:
                return nav
            self.log("on lobby but no 咖啡厅入口 cls — YOLO gap; waiting")
            return action_wait(400, "waiting for 咖啡厅入口 cls")

        if page is not None:
            self.log(f"wrong screen '{page}', backing out")
            return action_back(f"back from {page}")

        # Unknown / transition screen.
        if self._phase_ticks > _ENTER_MAX:
            self.log("enter budget exhausted, giving up on cafe")
            return action_done("could not reach cafe")
        if screen.is_loading() or len(screen.yolo_boxes or []) < 2:
            return action_wait(700, "no UI detected, likely loading")
        if self._enter_attempts > 8:
            return action_back("recover from unknown screen before cafe")
        return action_wait(400, "entering cafe")

    # ── earnings ──────────────────────────────────────────────────────────

    def _earnings(self, screen: ScreenState) -> Dict[str, Any]:
        """Open CAFE_EARNINGS, claim via the active CLAIM cls.

        Pure YOLO + button-state: there is no popup-title cls, so the popup is
        detected by a CLAIM button in the centered claim band (or the
        CAFE_EARNINGS label leaking near the popup top). An ACTIVE (yellow)
        claim cls ⇒ claim it; otherwise (0% / already claimed) close & move on.
        DIGIT-DEFERRED: the old "skip 0% to save the cap" % read is gone with
        nav-OCR off — we drive purely off the claim button colour/state.
        """
        if self._earnings_done:
            self._begin_invite(floor_2=False)
            return action_wait(300, "earnings done → invite")

        if not self._is_cafe(screen):
            if screen.is_lobby():
                self.log("earnings: on lobby, re-entering")
                self._enter_attempts = 0
                self._goto("enter")
                return action_wait(300, "earnings: back on lobby")
            if self._phase_ticks > _EARNINGS_MAX:
                self.log("earnings: cafe never appeared, skipping to invite")
                self._earnings_done = True
                return action_wait(300, "earnings skip (no cafe)")
            return action_wait(500, "waiting for cafe UI (earnings)")

        _CLAIM_BAND = (0.30, 0.58, 0.70, 0.86)
        claim_active = self.find_cls(
            screen, [UC.CLAIM_REWARD_YELLOW, UC.CLAIM_YELLOW, UC.CLAIM_BLUE],
            conf=_CLS_CONF, region=_CLAIM_BAND,
        )
        claim_grey = self.find_cls(
            screen, [UC.CLAIM_REWARD_GREY, UC.CLAIM_GREY],
            conf=_CLS_CONF, region=_CLAIM_BAND,
        )
        label_leak = self.find_cls(
            screen, UC.CAFE_EARNINGS, conf=_CLS_CONF, region=(0.30, 0.04, 0.70, 0.40)
        )
        popup_open = claim_active is not None or claim_grey is not None or label_leak is not None

        if popup_open:
            if claim_active is not None:
                self.log(f"earnings popup, claiming (YOLO {claim_active.cls_name})")
                self._earnings_done = True
                return action_click_box(claim_active, "claim earnings")
            # Grey / no claim cls ⇒ nothing to collect (0% or done) → close.
            self.log("earnings claim greyed / absent → nothing to claim, closing")
            self._earnings_done = True
            close = self._close_x(screen)
            if close is not None:
                return action_click_box(close, "close earnings popup (nothing to claim)")
            self._begin_invite(floor_2=False)
            return action_wait(300, "earnings nothing to claim → invite")

        # Cafe main screen: open the earnings popup via CAFE_EARNINGS cls.
        earn = self.find_cls(screen, UC.CAFE_EARNINGS, conf=_CLS_CONF)
        if earn is not None:
            self.log("opening earnings popup (YOLO 咖啡厅收益)")
            return action_click_box(earn, "open earnings popup")

        # CAFE_EARNINGS not seen — give the cls a few ticks, then skip.
        if self._phase_ticks > _EARNINGS_MAX:
            self.log("CAFE_EARNINGS cls never found — skipping earnings")
            self._earnings_done = True
            self._begin_invite(floor_2=False)
            return action_wait(300, "no earnings cls → invite")
        return action_wait(350, "waiting for CAFE_EARNINGS cls")

    # ── invite ────────────────────────────────────────────────────────────

    def _begin_invite(self, *, floor_2: bool) -> None:
        """Enter the invite sub_state for the given floor."""
        self._on_2f = floor_2
        self._invite_floor_done = False
        self._invite_stage = 0
        self._invite_scrolls = 0
        self._invite_settle = 0
        self._invite_last_sig = ""
        self._invite_sig_repeat = 0
        self._invite_retry_btn = None
        self._goto("invite")

    def _floor_target(self) -> Optional[str]:
        """Config target for the current floor (1F=idx0, 2F=idx1)."""
        idx = 1 if self._on_2f else 0
        if idx < len(self._invite_targets):
            return self._invite_targets[idx]
        return None

    def _pair_target_row(self, screen: ScreenState, invite_btns: List[YoloBox]
                         ) -> Tuple[Optional[YoloBox], Optional[str], bool]:
        """Pair each invite button with the avatar on its row and pick a target.

        Avatars come from the fused_avatar model (model_tag=="avatar",
        cls_name=中文角色名, cx≈0.35). For each avatar we find the invite button
        whose cy is within _ROW_PAIR_DY (same row) and return:
          (button, cls_name, is_priority)
        is_priority=True when the matched name == this floor's config target.
        Falls back to any not-yet-invited configured target on screen; if no
        config match, returns (None, None, False) so the caller can scroll /
        eventually click the first row.
        """
        avatars = [
            b for b in (screen.yolo_boxes or [])
            if b.model_tag == "avatar" and b.confidence >= _AVATAR_CONF
        ]
        target = self._floor_target()
        target_set = set(self._invite_targets)

        priority_hit: Optional[Tuple[YoloBox, str]] = None
        fallback_hit: Optional[Tuple[YoloBox, str]] = None
        for av in avatars:
            name = av.cls_name
            if name in self._invited:
                continue
            # find the invite button on the same row (closest cy within band)
            btn = None
            best_dy = _ROW_PAIR_DY
            for b in invite_btns:
                dy = abs(b.cy - av.cy)
                if dy < best_dy:
                    best_dy = dy
                    btn = b
            if btn is None:
                continue
            if target is not None and name == target:
                priority_hit = (btn, name)
                break
            if name in target_set and fallback_hit is None:
                fallback_hit = (btn, name)

        if priority_hit is not None:
            return priority_hit[0], priority_hit[1], True
        if fallback_hit is not None:
            return fallback_hit[0], fallback_hit[1], False
        return None, None, False

    def _invite(self, screen: ScreenState) -> Dict[str, Any]:
        floor = 2 if self._on_2f else 1

        # Done with this floor's invite → advance.
        if self._invite_floor_done:
            recover = self._recover_invite_overlay(screen)
            if recover:
                return recover
            if self._on_2f:
                self._goto("headpat2")
            else:
                self._goto("headpat")
            return action_wait(300, f"invite done → headpat (floor {floor})")

        # The invite list / confirm dialog hides the cafe signature, so being
        # off the cafe page mid-invite is normal. Only bail if we're clearly
        # back on the lobby.
        in_invite_ui = self._invite_list_open(screen) or self._confirm_btn(screen) is not None
        if not self._is_cafe(screen) and not in_invite_ui:
            if screen.is_lobby():
                self.log("invite: on lobby, re-entering")
                self._enter_attempts = 0
                self._goto("enter")
                return action_wait(300, "invite: back on lobby")
            if self._phase_ticks > _INVITE_MAX:
                self.log("invite: lost cafe too long, skipping invite")
                self._invite_floor_done = True
                return action_wait(300, "invite skip (lost cafe)")
            return action_wait(450, "waiting for cafe UI (invite)")

        # Hard budget guard.
        if self._phase_ticks > _INVITE_MAX:
            self.log(f"invite budget exhausted (floor {floor}), skipping")
            self._invite_floor_done = True
            return action_wait(300, "invite budget exhausted")

        # Stage 2: confirm the "邀請XXX到咖啡廳" dialog (REQUIRES BTN_CONFIRM —
        # clicking 取消/X cancels the invite and the student never spawns).
        if self._invite_stage == 2:
            confirm = self._confirm_btn(screen)
            if confirm is not None:
                self.log("confirming invite (YOLO 确认键)")
                self._invite_stage = 0
                self._invite_floor_done = True
                return action_click_box(confirm, "confirm invite")
            # No confirm dialog. Probe note: the invite button sometimes needs
            # a second press before the dialog appears — retry the row once.
            if self._invite_retry_btn is not None and self._phase_ticks % 3 == 0:
                bx, by = self._invite_retry_btn
                self.log("invite confirm not shown, re-pressing invite button")
                return action_click(bx, by, "re-press invite button (no dialog)")
            # If the list is back/open with no dialog after a few ticks, the
            # invite likely didn't register → go re-pick a row.
            if self._invite_list_open(screen):
                self.log("invite dialog absent, list still open → re-pick row")
                self._invite_stage = 1
                return action_wait(300, "re-pick invite row")
            if self._is_cafe(screen):
                self.log("invite dialog gone, back on cafe → invite done")
                self._invite_floor_done = True
                return action_wait(300, "invite done (dialog dismissed)")
            return action_wait(350, "waiting for invite confirm dialog")

        # Stage 1: list open — find the target row, scroll, or click first.
        if self._invite_stage == 1:
            # Post-swipe settle: don't scan mid-animation.
            if self._invite_settle > 0:
                self._invite_settle -= 1
                return action_wait(220, f"invite list settle ({self._invite_settle})")

            invite_btns = self.find_all_cls(
                screen, UC.CAFE_INVITE_BTN, conf=_CLS_CONF,
                region=(0.50, 0.20, 0.72, 0.92),
            )
            if not invite_btns:
                # List not rendered yet (or closed). Re-open after a few ticks.
                if self._phase_ticks % 8 == 0:
                    self.log("invite list missing → back to stage 0 (re-open ticket)")
                    self._invite_stage = 0
                return action_wait(350, "waiting for invite list (CAFE_INVITE_BTN)")

            btn, name, is_priority = self._pair_target_row(screen, invite_btns)

            # scroll-stuck detector (list bottom) via button-row y signature
            sig = "|".join(f"{round(b.cy, 2):.2f}" for b in sorted(invite_btns, key=lambda b: b.cy))
            if sig and sig == self._invite_last_sig:
                self._invite_sig_repeat += 1
            else:
                self._invite_sig_repeat = 0
            self._invite_last_sig = sig
            bottom = self._invite_sig_repeat >= 2
            budget_out = self._invite_scrolls >= _INVITE_MAX_SCROLLS

            if btn is not None and is_priority:
                return self._fire_invite(btn, name, floor, "priority")

            if btn is not None and (bottom or budget_out):
                why = "list bottom" if bottom else "scroll budget"
                return self._fire_invite(btn, name, floor, f"fallback/{why}")

            if btn is not None:
                # Found a non-priority configured fav but still hunting the
                # floor target — scroll on unless exhausted.
                self._invite_scrolls += 1
                self._invite_settle = _INVITE_SWIPE_SETTLE
                self.log(f"fav '{name}' found, hunting target — scroll "
                         f"({self._invite_scrolls}/{_INVITE_MAX_SCROLLS})")
                return action_swipe(0.35, 0.68, 0.35, 0.46, 800, "scroll invite list (hunt)")

            # No configured target on screen.
            if bottom or budget_out or not self._invite_targets:
                # Give up hunting → invite the first (top) row as a fallback,
                # unless that student was already invited this run.
                top = sorted(invite_btns, key=lambda b: b.cy)[0]
                why = ("no targets configured" if not self._invite_targets
                       else ("list bottom" if bottom else "scroll budget"))
                # try to name the top row from an avatar on its line
                top_name = self._row_name(screen, top.cy) or "?"
                if top_name in self._invited and len(invite_btns) > 1:
                    top = sorted(invite_btns, key=lambda b: b.cy)[1]
                    top_name = self._row_name(screen, top.cy) or "?"
                return self._fire_invite(top, top_name, floor, f"first-row/{why}")

            self._invite_scrolls += 1
            self._invite_settle = _INVITE_SWIPE_SETTLE
            self.log(f"target not visible, scrolling ({self._invite_scrolls}/{_INVITE_MAX_SCROLLS})")
            return action_swipe(0.35, 0.68, 0.35, 0.46, 800, "scroll invite list")

        # Stage 0: open the invite ticket from the cafe main screen.
        # First: if the list is already open (toggle race), go to stage 1 —
        # clicking the ticket again would CLOSE it.
        if self._invite_list_open(screen):
            self._invite_stage = 1
            return action_wait(200, "invite list already open")

        # Close a leftover earnings popup before opening the ticket.
        leftover = self.find_cls(
            screen, [UC.CLAIM_REWARD_YELLOW, UC.CLAIM_YELLOW, UC.CLAIM_BLUE,
                     UC.CLAIM_REWARD_GREY, UC.CLAIM_GREY],
            conf=_CLS_CONF, region=(0.30, 0.58, 0.70, 0.86),
        )
        if leftover is not None:
            close = self._close_x(screen)
            if close is not None:
                return action_click_box(close, "close leftover earnings before invite")

        ticket = self.find_cls(
            screen, UC.CAFE_INVITE_TICKET, conf=_CLS_CONF, region=(0.55, 0.78, 0.82, 0.99)
        )
        if ticket is not None:
            self.log("opening invite ticket (YOLO 咖啡厅邀请卷)")
            self._invite_stage = 1
            return action_click_box(ticket, "open invite ticket")

        # DIGIT-DEFERRED: the ticket-cooldown HH:MM:SS skip is unreadable with
        # nav-OCR off. We just attempt the ticket; if it's on cooldown the
        # opened list yields no rows and stage 1 self-skips on its budget.
        if self._phase_ticks > 10:
            self.log("CAFE_INVITE_TICKET cls not found (YOLO gap), skipping invite")
            self._invite_floor_done = True
            return action_wait(300, "invite skipped (no ticket cls)")
        return action_wait(350, "waiting for CAFE_INVITE_TICKET cls")

    def _row_name(self, screen: ScreenState, row_cy: float) -> Optional[str]:
        """Best avatar cls_name on the row at row_cy (None if no avatar box)."""
        best = None
        best_dy = _ROW_PAIR_DY
        for b in (screen.yolo_boxes or []):
            if b.model_tag != "avatar" or b.confidence < _AVATAR_CONF:
                continue
            dy = abs(b.cy - row_cy)
            if dy < best_dy:
                best_dy = dy
                best = b
        return best.cls_name if best is not None else None

    def _fire_invite(self, btn: YoloBox, name: str, floor: int, tag: str
                     ) -> Dict[str, Any]:
        if name and name != "?":
            self._invited.add(name)
            self._save_invited()
        self._invite_retry_btn = (btn.cx, btn.cy)
        self._invite_stage = 2
        self.log(f"inviting {tag} '{name}' at ({btn.cx:.2f},{btn.cy:.2f}) floor={floor}")
        return action_click_box(btn, f"invite {tag} {name}")

    def _recover_invite_overlay(self, screen: ScreenState) -> Optional[Dict[str, Any]]:
        """Dismiss a lingering invite confirm / list before leaving invite."""
        confirm = self._confirm_btn(screen)
        if confirm is not None:
            self.log("invite confirm still up, clicking confirm")
            return action_click_box(confirm, "confirm lingering invite")
        if self._invite_list_open(screen):
            close = self._close_x(screen, region=(0.56, 0.04, 0.90, 0.24))
            if close is not None:
                self.log("invite list still open, closing")
                return action_click_box(close, "close lingering invite list")
            return action_back("close lingering invite list (ESC)")
        return None

    # ── headpat ───────────────────────────────────────────────────────────

    def _emoticon_mark(self, screen: ScreenState) -> Optional[YoloBox]:
        """Best Emoticon_Action marker that is a real student (not a UI FP).

        Rejects the top-left 指定/隨機訪問 false-positive zone and any mark too
        close to a recently-patted coordinate (post-pat animation residue).
        """
        best = None
        fp_x, fp_y = _FP_ZONE
        for b in (screen.yolo_boxes or []):
            if "emoticon" not in b.cls_name.lower():
                continue
            if b.confidence < _EMOTICON_CONF:
                continue
            # top-left fixed false positive (指定訪問/隨機訪問 buttons)
            if b.cx < fp_x and b.cy < fp_y:
                continue
            # dedup against recent pats
            if any(abs(b.cx - px) < _HEADPAT_DEDUP_DIST
                   and abs(b.cy - py) < _HEADPAT_DEDUP_DIST
                   for px, py in self._recent_pats):
                continue
            if best is None or b.confidence > best.confidence:
                best = b
        return best

    def _do_headpat(self, mark: YoloBox) -> Dict[str, Any]:
        click_x = min(0.99, mark.cx + _HEADPAT_DX)  # body is right of the bubble
        click_y = mark.cy
        self._pat_count += 1
        self._empty_frames = 0
        self._pat_settle = 1  # one tick for the heart animation
        self._recent_pats.append((mark.cx, mark.cy))
        if len(self._recent_pats) > _HEADPAT_DEDUP_KEEP:
            self._recent_pats.pop(0)
        self.log(f"headpat #{self._pat_count} conf={mark.confidence:.2f} "
                 f"mark=({mark.cx:.2f},{mark.cy:.2f}) click=({click_x:.2f},{click_y:.2f})")
        return action_click(click_x, click_y, f"headpat student #{self._pat_count}")

    def _headpat(self, screen: ScreenState) -> Dict[str, Any]:
        """Tap students until the floor is dry, panning the camera to sweep.

        Algorithm (probe-confirmed): emoticon model only, click cx+0.025.
        loop-until-dry in the current view (3 empty frames), then swipe LEFT to
        reveal the right-side overflow, then swipe RIGHT (×2 distance) to reveal
        the left, then done. Per-floor helmets cap pats and pans.
        """
        is_2f = (self.sub_state == "headpat2")

        if not self._is_cafe(screen):
            if screen.is_lobby():
                if is_2f:
                    self.log("lobby during 2F headpat → done")
                    return action_done("cafe complete (lobby)")
                # kicked to lobby mid-1F (e.g. bond level-up) → re-enter and
                # skip straight to switch (1F headpat counts as done enough).
                self.log("lobby during 1F headpat → re-enter, skip to 2F")
                self._goto("enter")
                self._headpat_to_switch_on_reentry = True
                return action_wait(300, "re-enter cafe from lobby")
            if self._phase_ticks > 12:
                if is_2f:
                    self._goto("exit")
                    return action_wait(300, "lost cafe on 2F → exit")
                self._goto("switch")
                return action_wait(300, "lost cafe on 1F → switch")
            return action_wait(300, "waiting for cafe (headpat)")

        # If we re-entered after a lobby kick on 1F, jump to switch.
        if getattr(self, "_headpat_to_switch_on_reentry", False) and not is_2f:
            self._headpat_to_switch_on_reentry = False
            self._goto("switch")
            return action_wait(300, "resume → switch to 2F")

        # Safety helmet.
        if self._pat_count >= _MAX_HEADPATS_PER_FLOOR:
            return self._finish_headpat(is_2f, "max headpats")

        # Post-pat / post-pan settle — let animation finish before scanning.
        if self._pat_settle > 0:
            self._pat_settle -= 1
            return action_wait(450, f"headpat settle ({self._pat_settle})")

        # PRIORITY: pat any visible marker immediately (markers fade fast).
        mark = self._emoticon_mark(screen)
        if mark is not None:
            return self._do_headpat(mark)

        # No marker this frame.
        self._empty_frames += 1
        if self._empty_frames < _HEADPAT_DRY_FRAMES:
            return action_wait(280, f"scanning headpat (empty={self._empty_frames}, pan={self._pan_dir})")

        # Current view is dry → pan to the next region.
        if self._pan_count >= _MAX_PANS_PER_FLOOR:
            return self._finish_headpat(is_2f, "pan helmet")

        floor_tag = "2F" if is_2f else "1F"
        if self._pan_dir == 0:
            # sweep LEFT (0.75→0.25) to reveal the right-side overflow
            self._pan_dir = 1
            self._pan_count += 1
            self._empty_frames = 0
            self._pat_settle = 2
            self.log(f"{floor_tag} sweep LEFT (reveal right overflow)")
            return action_swipe(0.75, 0.45, 0.25, 0.45, 600, f"pan left ({floor_tag})")
        if self._pan_dir == 1:
            # sweep RIGHT twice the distance (0.25→0.80) to reveal the left
            self._pan_dir = 2
            self._pan_count += 1
            self._empty_frames = 0
            self._pat_settle = 2
            self.log(f"{floor_tag} sweep RIGHT (reveal left overflow)")
            return action_swipe(0.25, 0.45, 0.80, 0.45, 600, f"pan right ({floor_tag})")

        # Both directions swept and dry → floor done.
        return self._finish_headpat(is_2f, "all views dry")

    def _finish_headpat(self, is_2f: bool, reason: str) -> Dict[str, Any]:
        if is_2f:
            self.log(f"2F headpat done ({reason}, {self._pat_count} pats) → exit")
            self._goto("exit")
            return action_wait(300, "2F headpat done → exit")
        self.log(f"1F headpat done ({reason}, {self._pat_count} pats) → switch")
        self._goto("switch")
        return action_wait(300, "1F headpat done → switch")

    # ── switch floor ──────────────────────────────────────────────────────

    def _switch_floor(self, screen: ScreenState) -> Dict[str, Any]:
        """1F → 2F. On 2F the button reads CAFE_MOVE_1F (takes us back to 1F)."""
        # Already on 2F? (CAFE_MOVE_1F button visible top-left)
        on_2f_btn = self.find_cls(
            screen, UC.CAFE_MOVE_1F, conf=_CLS_CONF, region=(0.0, 0.03, 0.30, 0.22)
        )
        if on_2f_btn is not None:
            # If we already ran a 2F cycle, exit (don't loop a 3rd invite).
            if self._on_2f:
                self.log("already on 2F and cycle done → exit")
                self._goto("exit")
                return action_wait(300, "2F cycle complete → exit")
            self.log("already on 2F → start 2F invite")
            self._reset_headpat_for_floor()
            self._begin_invite(floor_2=True)
            return action_wait(300, "already on 2F → invite")

        switch = self.find_cls(screen, UC.CAFE_MOVE_2F, conf=_CLS_CONF)
        if switch is not None:
            self.log("switching to cafe 2F (YOLO 移动至2号点)")
            self._reset_headpat_for_floor()
            self._begin_invite(floor_2=True)
            return action_click_box(switch, "switch to cafe 2F")

        if screen.is_loading():
            return action_wait(700, "cafe switch loading")
        if self._phase_ticks > _SWITCH_MAX:
            self.log("switch budget exhausted, exiting cafe")
            self._goto("exit")
            return action_wait(300, "switch timeout → exit")
        return action_wait(450, "waiting for 移动至2号点 cls")

    def _reset_headpat_for_floor(self) -> None:
        self._pat_count = 0
        self._empty_frames = 0
        self._pan_count = 0
        self._pan_dir = 0
        self._pat_settle = 0
        self._recent_pats = []

    # ── exit ──────────────────────────────────────────────────────────────

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if self.detect_screen_yolo(screen) == "Lobby":
            self.log("back in lobby, cafe done")
            return action_done("cafe complete")
        if self._phase_ticks > _EXIT_MAX:
            self.log("exit budget exhausted, reporting done")
            return action_done("cafe exit timeout")
        home = self.find_cls(screen, UC.BTN_HOME, conf=_CLS_CONF)
        if home is not None:
            return action_click_box(home, "cafe exit: home button (YOLO 回大厅按钮)")
        back = self.find_cls(screen, UC.BTN_BACK, conf=_CLS_CONF)
        if back is not None:
            return action_click_box(back, "cafe exit: back button (YOLO 返回键)")
        return action_back("cafe exit: ESC toward lobby")
