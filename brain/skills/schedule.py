"""ScheduleSkill — Blue Archive 課程表 daily routine (clean pure-YOLO rewrite).

Probe-verified flow (interactive probe 2026-06-01, full spec in
data/_schedule_probe_log.md). Every click resolves through a YOLO ui cls
(ui_classes) or a fused_avatar head box — NO OCR for navigation, NO hardcoded
room coordinates. OCR is used for the ONE thing it's good at: reading the
held-ticket digits 「持有票券 X/7」.

High-level flow
---------------
1. lobby → click NAV_SCHEDULE → region-select screen.
2. Click 夏莱办公室 (list row 0 — region tiles are under-trained, GAP) → enter
   its region-internal 選擇課程表 screen → ARROW_LEFT once = jump to the LAST
   (newest) academy region (list wraps). Traverse backwards (ARROW_LEFT) from
   there.
3. Per region: click SCHED_ALL (全體課程表) → popout. fused_avatar reads the
   heads, clustered into rooms (≤3 heads/room, x-gap ~0.057, room-gap ~0.15;
   rows at y≈0.39 / 0.60 / 0.81). Detect ROOM_LOCKED:
     • Case A (locks present, region not max level): goal = level this region.
       Dispatch EVERY un-clicked room to spend tickets here.
     • Case B (no locks, max level): goal = only the dashboard target students.
       Dispatch a room iff a target head sits in it; else skip.
   When no room is left to click → close popout (BTN_CLOSE_X) → ARROW_LEFT next.
4. Per-room dispatch (one room): click a head → 課程表資訊 (SCHED_START) → click
   SCHED_START → 課程表報告 (BTN_CONFIRM) → click confirm → heads go green, ticket
   −1, back to popout.
5. End: digit-OCR 「持有票券 X/7」 == 0 → done.
6. Fallback: traversed every region (wrapped to start) and targets not found but
   tickets remain → dispatch any un-clicked room to spend the rest (don't waste).

Single-step state machine (strict — one state, one action per tick; see probe
log "自動循環寫糙的教訓"). The two dispatch popups are checked FIRST in tick(),
above the sub-state dispatch, by priority:
   1. report popup   : BTN_CONFIRM in center-bottom band  → click confirm
   2. info popup      : SCHED_START                        → click start
so they're handled identically no matter which sub_state we're in. We never
blind-click head positions to "advance" — that lands on popup backgrounds
(the bug that drained tickets). Ticket dispatch is confirmed by the digit-OCR
count dropping.

Detectors (pipeline.SKILL_YOLO_MAP["Schedule"] = "ui+avatar"):
  ui      — UC.* button classes (find_cls / find_all_cls).
  avatar  — fused_avatar (251 student heads), model_tag=="avatar", 中文角色名.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from brain.skills.base import (
    BaseSkill, ScreenState, YoloBox,
    action_click, action_click_box,
    action_wait, action_back, action_done,
)
from brain.skills import ui_classes as UC

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_APP_CONFIG_FILE = _DATA_DIR / "app_config.json"

# ── tuning knobs ──────────────────────────────────────────────────────────
_CLS_CONF = 0.30            # default UI cls confidence floor
_AVATAR_CONF = 0.30         # fused_avatar head confidence (probe used 0.30)

_MAX_TICKETS = 7            # 持有票券 X/7 — total ticket capacity
_ROOM_X_GAP = 0.085         # heads within this x-gap belong to the same room
_ROOM_MAX_HEADS = 3         # a room holds at most 3 students

# Dispatch popups (info / report) live in a center-bottom band. Both
# SCHED_START and the report's BTN_CONFIRM sit at cx≈0.499, cy≈0.76-0.77.
_DIALOG_BAND = (0.30, 0.66, 0.70, 0.90)

# ── GAP fallbacks (under-trained cls — documented in probe log) ───────────
# Region tiles (SCHOOL_*) are ~3-12 frames each and routinely missed (v6 gap
# #29). We click 夏莱办公室 by its list-row position when the cls isn't seen.
_OFFICE_ROW_POS = (0.64, 0.22)        # region-select list row 1 (夏莱办公室)
# ARROW_LEFT (117f) is usually solid, but fall back to its symmetric left-edge
# coord if the cls isn't detected this frame.
_ARROW_LEFT_POS = (0.023, 0.500)

# Per-sub-state tick budgets — every phase is bounded, never dead-waits.
_ENTER_MAX = 25
_NAVIGATE_MAX = 14
_ROSTER_MAX = 14
_OPEN_ROOM_MAX = 12
_SWITCH_MAX = 12
_EXIT_MAX = 14

# Traverse at most this many regions before giving up the full circle. BA has
# ~10 schedule regions; 14 leaves margin for re-detect retries.
_MAX_REGIONS = 14


def _load_schedule_targets() -> List[str]:
    """Read the schedule target-student list from the active app_config profile.

    Same pattern as CafeSkill._load_invite_targets: app_config.json →
    active_profile → profiles[active].schedule_target_students. Returns a list
    of 中文角色名 matching fused_avatar cls_name (e.g. "莉央(战斗)"). Empty list
    ⇒ Case B has no targets, so the fallback (spend remaining tickets on any
    room) takes over.
    """
    try:
        if not _APP_CONFIG_FILE.exists():
            return []
        data = json.loads(_APP_CONFIG_FILE.read_text("utf-8"))
        active = data.get("active_profile", "default")
        profile = (data.get("profiles") or {}).get(active, {})
        raw = profile.get("schedule_target_students")
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


class ScheduleSkill(BaseSkill):
    # Lobby schedule entries that carry a dot when there's work to do.
    _LOBBY_DOT_ENTRIES = [UC.NAV_SCHEDULE, UC.SCHED_TICKET]

    def should_run(self, screen: ScreenState) -> bool:
        return self.dot_on_entry(screen, self._LOBBY_DOT_ENTRIES)

    def __init__(self):
        super().__init__("Schedule")
        # Worst case: ~10 regions × (open popout + a few rooms + transitions).
        # 300 gives comfortable slack for multi-region traversal + retries.
        self.max_ticks = 320
        self._init_state()

    # ── state init / reset ────────────────────────────────────────────────

    def _init_state(self) -> None:
        self._phase_ticks: int = 0          # ticks spent in current sub_state
        self._enter_attempts: int = 0
        self._targets: List[str] = []
        self._tickets: int = -1             # -1 = unknown; 0 = exhausted
        # traversal bookkeeping
        self._office_clicked: bool = False  # clicked 夏莱办公室 yet?
        self._jumped_to_last: bool = False  # done the initial ARROW_LEFT jump?
        self._regions_seen: int = 0         # regions whose popout we've opened
        self._full_circle: bool = False     # traversed all regions → fallback
        # per-region (reset on each region switch)
        self._region_locked: Optional[bool] = None   # case A (True) / B (False)
        self._clicked_heads: List[Tuple[float, float]] = []  # heads dispatched here
        self._ticket_read_pending: bool = False  # re-read ticket after a dispatch

    def reset(self) -> None:
        super().reset()
        self._init_state()
        self._targets = _load_schedule_targets()
        if self._targets:
            self.log(f"schedule targets: {self._targets}")
        else:
            self.log("no schedule_target_students configured — Case B will "
                     "spend leftover tickets on any room (fallback)")

    def _goto(self, sub_state: str) -> None:
        """Switch sub_state and reset the per-phase tick counter."""
        self.sub_state = sub_state
        self._phase_ticks = 0

    def _reset_region(self) -> None:
        """Clear per-region state on entering a fresh region's popout."""
        self._region_locked = None
        self._clicked_heads = []
        self._ticket_read_pending = False

    # ── ticket digit-OCR ──────────────────────────────────────────────────

    # 「持有票券 X/7」 sits centered just under the popout title. The probe log
    # gives the title band region ~0.42,0.115,0.60,0.17 as the starting guess;
    # we read a slightly wider strip and take the first X/Y match. screen.frame
    # is the raw full-res BGR array (run_digit_ocr crops + upscales).
    _TICKET_REGION = (0.40, 0.105, 0.62, 0.175)

    def _read_tickets(self, screen: ScreenState) -> Optional[int]:
        """Read 持有票券 X/7 via digit-OCR. Returns current count, or None.

        Independent of the global nav-OCR switch (run_digit_ocr always works).
        Never raises / never blocks — None just means "couldn't read this
        frame", and the caller relies on max_ticks + "no room to click" as the
        real safety net rather than dead-waiting on the count.
        """
        if screen.frame is None:
            return None
        try:
            from brain.pipeline import run_digit_ocr, parse_count
        except Exception:
            return None
        raw = run_digit_ocr(screen.frame, self._TICKET_REGION)
        parsed = parse_count(raw)
        if parsed is None:
            return None
        cur, _tot = parsed
        if cur is None or cur < 0 or cur > _MAX_TICKETS:
            return None
        if cur != self._tickets:
            self.log(f"tickets: {cur}/{_MAX_TICKETS} (raw {raw!r})")
        self._tickets = cur
        return cur

    # ── screen-state helpers (pure YOLO) ──────────────────────────────────

    def _is_schedule(self, screen: ScreenState) -> bool:
        """Any schedule surface: detect_screen_yolo()=='Schedule' OR a region
        tile is on screen (the region-select list carries SCHOOL_* tiles but
        may have none of the SCHED_* cls)."""
        if self.detect_screen_yolo(screen) == "Schedule":
            return True
        return self.find_cls(screen, _SCHOOL_TILES, conf=_CLS_CONF) is not None

    def _sched_all_btn(self, screen: ScreenState) -> Optional[YoloBox]:
        """The 全體課程表 button on a region-internal screen (bottom-right,
        @~0.910,0.921)."""
        return self.find_cls(
            screen, UC.SCHED_ALL, conf=_CLS_CONF, region=(0.74, 0.80, 1.0, 1.0)
        )

    def _roster_open(self, screen: ScreenState) -> bool:
        """The 全體課程表 popout is open when its close-X sits top-right
        (@~0.888,0.138) AND fused_avatar heads / SCHED_ALL title are visible.

        We key off the popout close-X in the top-right band — the region-
        internal screen's SCHED_ALL is a bottom-right BUTTON, not a popout, so
        the close-X is the disambiguator that never appears on the plain
        region screen."""
        return self._popout_close(screen) is not None and (
            len(self._roster_heads(screen)) > 0
            or self.find_cls(screen, UC.SCHED_ALL, conf=_CLS_CONF,
                             region=(0.10, 0.0, 0.90, 0.25)) is not None
        )

    def _popout_close(self, screen: ScreenState) -> Optional[YoloBox]:
        """The 弹窗叉叉 at the popout's top-right (@~0.888,0.138)."""
        return self.find_cls(
            screen, UC.BTN_CLOSE_X, conf=_CLS_CONF, region=(0.78, 0.05, 1.0, 0.25)
        )

    def _arrow_left(self, screen: ScreenState) -> Optional[YoloBox]:
        """ARROW_LEFT (左切换) on the left edge (@~0.023,0.500)."""
        return self.find_cls(
            screen, UC.ARROW_LEFT, conf=_CLS_CONF, region=(0.0, 0.30, 0.12, 0.70)
        )

    # ── roster head clustering ────────────────────────────────────────────

    def _roster_heads(self, screen: ScreenState) -> List[YoloBox]:
        """All fused_avatar head boxes in the popout (model_tag=='avatar')."""
        return [
            b for b in (screen.yolo_boxes or [])
            if b.model_tag == "avatar" and b.confidence >= _AVATAR_CONF
        ]

    def _cluster_rooms(self, heads: List[YoloBox]) -> List[List[YoloBox]]:
        """Group heads into rooms. Heads in the same row (similar cy) within
        _ROOM_X_GAP of each other = one room; ≤3 heads/room.

        Probe layout: rows at y≈0.39 / 0.60 / 0.81, each row up to 6 rooms,
        head x-gap within a room ~0.057, room-to-room gap ~0.15.
        """
        if not heads:
            return []
        # bucket by row (cy) — 0.10 tolerance separates the ~0.21-apart rows
        rows: List[List[YoloBox]] = []
        for h in sorted(heads, key=lambda b: b.cy):
            placed = False
            for row in rows:
                if abs(row[0].cy - h.cy) < 0.10:
                    row.append(h)
                    placed = True
                    break
            if not placed:
                rows.append([h])
        rooms: List[List[YoloBox]] = []
        for row in rows:
            row.sort(key=lambda b: b.cx)
            cur: List[YoloBox] = []
            for h in row:
                if not cur:
                    cur = [h]
                    continue
                if (h.cx - cur[-1].cx) <= _ROOM_X_GAP and len(cur) < _ROOM_MAX_HEADS:
                    cur.append(h)
                else:
                    rooms.append(cur)
                    cur = [h]
            if cur:
                rooms.append(cur)
        return rooms

    @staticmethod
    def _room_center(room: List[YoloBox]) -> Tuple[float, float]:
        cx = sum(h.cx for h in room) / len(room)
        cy = sum(h.cy for h in room) / len(room)
        return (cx, cy)

    def _room_already_clicked(self, room: List[YoloBox]) -> bool:
        """True if we've already dispatched this room this region (matched by
        proximity to a stored click point — heads don't move within a region)."""
        cx, cy = self._room_center(room)
        for px, py in self._clicked_heads:
            if abs(px - cx) < _ROOM_X_GAP and abs(py - cy) < 0.10:
                return True
        return False

    def _room_has_target(self, room: List[YoloBox]) -> Optional[str]:
        """Return the matched target name if any head in the room is a
        configured target student, else None."""
        if not self._targets:
            return None
        target_set = set(self._targets)
        for h in room:
            if h.cls_name in target_set:
                return h.cls_name
        return None

    def _pick_room(self, rooms: List[List[YoloBox]]) -> Tuple[
            Optional[List[YoloBox]], str]:
        """Choose the next room to dispatch given the case (A/B) and what's
        already been clicked. Returns (room, reason) or (None, reason)."""
        candidates = [r for r in rooms if not self._room_already_clicked(r)]
        if not candidates:
            return None, "all rooms clicked"

        # Case A (region has locks → level it up): dispatch every room.
        if self._region_locked:
            # top-to-bottom, left-to-right for determinism
            candidates.sort(key=lambda r: (round(self._room_center(r)[1], 2),
                                           self._room_center(r)[0]))
            return candidates[0], "case-A spend-on-this-region"

        # Case B (no locks): only rooms containing a configured target.
        for r in candidates:
            name = self._room_has_target(r)
            if name:
                return r, f"case-B target '{name}'"

        # Fallback: full circle done + tickets remain → spend on any room.
        if self._full_circle and (self._tickets is None or self._tickets > 0):
            candidates.sort(key=lambda r: (round(self._room_center(r)[1], 2),
                                           self._room_center(r)[0]))
            return candidates[0], "fallback spend-leftover"

        return None, "case-B no target in region"

    # ── tick: global popup guards + state dispatch ────────────────────────

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1
        self._phase_ticks += 1

        if self.ticks >= self.max_ticks:
            self.log("timeout, exiting")
            return action_done("schedule timeout")

        # ── popups that can appear in any sub_state (pure YOLO) ──

        # Reward-result popup ("獲得獎勵") — tap to dismiss.
        got_reward = self.find_cls(screen, UC.GOT_REWARD, conf=_CLS_CONF)
        if got_reward is not None:
            self.log("reward-result popup, dismissing (YOLO 获得奖励)")
            return action_click_box(got_reward, "dismiss reward result")

        # Bond / region level-up full-screen splash — tap anywhere advances.
        splash = self.find_cls(
            screen, [UC.BOND_LEVELUP, UC.REGION_LEVELUP], conf=_CLS_CONF
        )
        if splash is not None:
            self.log(f"level-up splash ({splash.cls_name}), tap to dismiss")
            return action_click(0.5, 0.5, f"dismiss splash ({splash.cls_name})")

        # ── DISPATCH POPUP PRIORITY (strict single-step, runs above states) ──
        # These two are the per-room dispatch sequence and must be handled
        # FIRST so they win regardless of sub_state (probe教训: state-machine
        # mis-judgement here = mis-clicks that drained tickets).

        # PRIORITY 1: 課程表報告 result popup — its only actionable is 確認
        # (center-bottom band). Present iff BTN_CONFIRM is in the band AND
        # SCHED_START is NOT (a SCHED_START in-band means the info popup, not
        # the report).
        report_confirm = self.find_cls(
            screen, UC.BTN_CONFIRM, conf=_CLS_CONF, region=_DIALOG_BAND
        )
        if report_confirm is not None and self.find_cls(
                screen, UC.SCHED_START, conf=_CLS_CONF, region=_DIALOG_BAND) is None:
            self.log("schedule report → confirm (YOLO 确认键)")
            self._ticket_read_pending = True  # re-read count back on the popout
            self._goto("roster")
            return action_click_box(report_confirm, "confirm schedule report")

        # PRIORITY 2: 課程表資訊 info popup — click 課程表開始 (SCHED_START).
        start = self.find_cls(
            screen, UC.SCHED_START, conf=_CLS_CONF, region=_DIALOG_BAND
        )
        if start is None:  # SCHED_START sometimes mid-frame outside the band
            start = self.find_cls(screen, UC.SCHED_START, conf=_CLS_CONF)
        if start is not None:
            self.log("schedule info → start (YOLO 课程表开始)")
            self._goto("open_room")
            return action_click_box(start, "start schedule")

        # Generic / ticket-shortage popups — base helper resolves confirm/cancel
        # via OCR+YOLO bottom-up. NOTE: SCHED_ALL is the WORK surface, never a
        # popup — the base helper keys off dialog headers/buttons, not SCHED_ALL,
        # so it won't touch the popout (task #3 dead-loop guard).
        popup = self._handle_common_popups(screen)
        if popup:
            return popup

        if screen.is_loading():
            return action_wait(700, "schedule loading")

        # ── state machine ──
        if self.sub_state == "":
            self._goto("enter")

        handler = {
            "enter": self._enter,
            "navigate": self._navigate,
            "roster": self._roster,
            "open_room": self._open_room,
            "switch": self._switch,
            "exit": self._exit,
        }.get(self.sub_state)
        if handler is None:
            return action_wait(300, "schedule unknown state")
        return handler(screen)

    # ── enter ─────────────────────────────────────────────────────────────

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        self._enter_attempts += 1
        page = self.detect_screen_yolo(screen)

        if page == "Schedule" or self._is_schedule(screen):
            self.log("inside schedule")
            self._goto("navigate")
            return action_wait(400, "entered schedule")

        if page == "Lobby":
            nav = self.click_cls(screen, UC.NAV_SCHEDULE, "click schedule nav",
                                 conf=_CLS_CONF)
            if nav:
                return nav
            self.log("on lobby but no 课程表入口 cls — YOLO gap; waiting")
            return action_wait(400, "waiting for 课程表入口 cls")

        if page is not None:
            self.log(f"wrong screen '{page}', backing out")
            return action_back(f"back from {page}")

        if self._phase_ticks > _ENTER_MAX:
            self.log("enter budget exhausted, giving up on schedule")
            return action_done("could not reach schedule")
        if screen.is_loading() or len(screen.yolo_boxes or []) < 2:
            return action_wait(700, "no UI detected, likely loading")
        return action_wait(400, "entering schedule")

    # ── navigate (region-select / region-internal) ────────────────────────

    def _navigate(self, screen: ScreenState) -> Dict[str, Any]:
        """Drive from the region-select list to a region's popout.

        Sequence (once):  click 夏莱办公室 → ARROW_LEFT (jump to last region).
        Then per region:  click SCHED_ALL → open popout → roster.
        """
        if not self._is_schedule(screen):
            if self.detect_screen_yolo(screen) == "Lobby":
                self.log("on lobby, schedule exited — done")
                return action_done("schedule done (returned to lobby)")
            if self._phase_ticks > _NAVIGATE_MAX:
                self.log("lost schedule UI, backing out")
                return action_back("back (schedule UI lost)")
            return action_wait(500, "waiting for schedule UI")

        # If a popout is already open, go scan it.
        if self._roster_open(screen):
            self._reset_region()
            self._goto("roster")
            return action_wait(300, "popout already open → roster")

        if self._tickets == 0:
            self.log("no tickets remaining, exiting")
            self._goto("exit")
            return action_wait(300, "no tickets")

        # Step 1: first click 夏莱办公室 (region-select list row 0).
        if not self._office_clicked:
            office = self.find_cls(screen, UC.SCHOOL_OFFICE, conf=_CLS_CONF)
            if office is not None:
                self.log("entering 夏莱办公室 (YOLO 夏莱办公室)")
                self._office_clicked = True
                return action_click_box(office, "enter 夏莱办公室")
            # GAP: region tile cls under-trained → click its list-row position.
            self.log("夏莱办公室 cls not seen — clicking list row 0 (GAP)")
            self._office_clicked = True
            return action_click(*_OFFICE_ROW_POS, "enter 夏莱办公室 (list row, GAP)")

        # Step 2: inside a region now → ARROW_LEFT once = jump to last region.
        if not self._jumped_to_last:
            # Only jump once we're actually on a region-internal screen (the
            # SCHED_ALL bottom-right button is present there).
            if self._sched_all_btn(screen) is None and self._phase_ticks < 4:
                return action_wait(400, "waiting for region-internal screen")
            arrow = self._arrow_left(screen)
            self._jumped_to_last = True
            if arrow is not None:
                self.log("ARROW_LEFT → jump to last (newest) region (YOLO 左切换)")
                return action_click_box(arrow, "jump to last region")
            self.log("ARROW_LEFT cls not seen — symmetric left-edge coord (GAP)")
            return action_click(*_ARROW_LEFT_POS, "jump to last region (GAP)")

        # Step 3: on a region-internal screen → open its 全體課程表 popout.
        sched_all = self._sched_all_btn(screen)
        if sched_all is not None:
            self.log("opening 全體課程表 popout (YOLO 全体课程表)")
            return action_click_box(sched_all, "open 全體課程表 popout")

        # SCHED_ALL button not seen — give it a few ticks (region transition),
        # then surface the gap and try the next region rather than stalling.
        if self._phase_ticks > _NAVIGATE_MAX:
            self.log("SCHED_ALL button cls missing — YOLO gap; switching region")
            self._goto("switch")
            return action_wait(300, "no SCHED_ALL, switching region")
        return action_wait(400, "waiting for 全体课程表 button cls")

    # ── roster (popout open) ──────────────────────────────────────────────

    def _roster(self, screen: ScreenState) -> Dict[str, Any]:
        """Popout open: read tickets, detect case A/B, pick + click a room."""
        if not self._roster_open(screen):
            # Popout closed unexpectedly (or game re-rendered). If we're on a
            # region-internal screen, re-open; if lost schedule, bail.
            if not self._is_schedule(screen):
                if self.detect_screen_yolo(screen) == "Lobby":
                    return action_done("schedule done (lobby)")
                if self._phase_ticks > _ROSTER_MAX:
                    return action_back("back (lost schedule in roster)")
                return action_wait(450, "waiting for popout")
            if self._sched_all_btn(screen) is not None:
                self.log("popout closed, re-opening 全體課程表")
                return action_click_box(self._sched_all_btn(screen),
                                        "re-open 全體課程表 popout")
            if self._phase_ticks > _ROSTER_MAX:
                self._goto("switch")
                return action_wait(300, "popout gone, switching region")
            return action_wait(400, "waiting for popout to render")

        # First tick on a fresh popout: count this region + read tickets.
        if self._region_locked is None:
            self._regions_seen += 1
            self._tickets = -1  # force a fresh read on this popout
            self._read_tickets(screen)
            locked = self.find_cls(screen, UC.ROOM_LOCKED, conf=_CLS_CONF) is not None
            self._region_locked = locked
            self.log(f"region #{self._regions_seen}: "
                     f"{'LOCKED (case A — level it)' if locked else 'no locks (case B — targets only)'}"
                     f", tickets={self._tickets}")
            return action_wait(300, "scanned popout")

        # Re-read ticket after a dispatch (confirms it actually decremented).
        if self._ticket_read_pending:
            self._ticket_read_pending = False
            self._read_tickets(screen)
            if self._tickets == 0:
                self.log("tickets exhausted → exit")
                self._goto("exit")
                return action_wait(300, "tickets exhausted")

        heads = self._roster_heads(screen)
        rooms = self._cluster_rooms(heads)
        room, reason = self._pick_room(rooms)

        if room is not None:
            cx, cy = self._room_center(room)
            names = [h.cls_name for h in room]
            self.log(f"dispatch room @({cx:.3f},{cy:.3f}) {names} — {reason}")
            self._clicked_heads.append((cx, cy))
            # Click the first (left-most) head — opens the whole room's info
            # popup (not a single-student select). The dispatch popups in
            # tick() take over from here (info → start → report → confirm).
            head = sorted(room, key=lambda b: b.cx)[0]
            self._goto("open_room")
            return action_click_box(head, "click room head → 課程表資訊")

        # No room to dispatch in this region → close popout, next region.
        self.log(f"no room to dispatch ({reason}) → close popout, next region")
        close = self._popout_close(screen)
        if close is not None:
            self._goto("switch")
            return action_click_box(close, "close popout (YOLO 弹窗叉叉)")
        if self._phase_ticks > _ROSTER_MAX:
            self.log("popout close-X cls missing — YOLO gap; ESC to switch")
            self._goto("switch")
            return action_back("close popout (no X cls)")
        return action_wait(400, "waiting for popout close-X cls")

    # ── open_room (info → start → report → confirm handled in tick) ────────

    def _open_room(self, screen: ScreenState) -> Dict[str, Any]:
        """After clicking a head, wait for the dispatch popups. The info/report
        popups are handled by the PRIORITY guards in tick(); here we only handle
        the gaps: popup didn't appear, drifted off-screen, or we're back on the
        popout already."""
        # Back on the popout (report confirmed, or info dismissed) → re-scan.
        if self._roster_open(screen):
            self.log("back on popout after dispatch → roster")
            self._goto("roster")
            return action_wait(300, "back to roster")

        if not self._is_schedule(screen):
            if self.detect_screen_yolo(screen) == "Lobby":
                self.log("drifted to lobby during dispatch, re-entering")
                self._office_clicked = True   # don't restart the whole nav
                self._jumped_to_last = True
                self._goto("enter")
                return action_wait(300, "re-enter from lobby")
            return action_wait(450, "waiting for dispatch popup")

        # On a schedule surface but neither popup nor popout detected. The
        # tick() guards catch SCHED_START / BTN_CONFIRM the moment they render;
        # if neither appears within budget, treat the room as done and go back
        # to the roster (the popout re-opens after a report).
        if self._phase_ticks > _OPEN_ROOM_MAX:
            self.log("dispatch popup never appeared — back to roster")
            self._goto("roster")
            return action_wait(300, "dispatch timeout → roster")
        return action_wait(400, "waiting for SCHED_START / report cls")

    # ── switch (ARROW_LEFT to next region) ────────────────────────────────

    def _switch(self, screen: ScreenState) -> Dict[str, Any]:
        """Close any lingering popout, then ARROW_LEFT to the next region."""
        if self._tickets == 0:
            self._goto("exit")
            return action_wait(300, "tickets exhausted → exit")

        # Full circle: traversed every region. If targets weren't found and
        # tickets remain, flip the fallback so the next popouts spend leftovers;
        # otherwise we're done.
        if self._regions_seen >= _MAX_REGIONS:
            if (self._tickets is None or self._tickets > 0) and not self._full_circle:
                self.log("full circle done, tickets remain → fallback "
                         "(spend leftover on any room)")
                self._full_circle = True
                self._regions_seen = 0  # one more pass to spend leftovers
            else:
                self.log(f"full circle done ({self._regions_seen} regions) → exit")
                self._goto("exit")
                return action_wait(300, "full circle → exit")

        # Close a lingering popout first.
        if self._roster_open(screen):
            close = self._popout_close(screen)
            if close is not None:
                return action_click_box(close, "close popout before switch")
            return action_back("close popout before switch (no X cls)")

        # ARROW_LEFT to the next region.
        arrow = self._arrow_left(screen)
        if arrow is not None:
            self.log("ARROW_LEFT → next region (YOLO 左切换)")
            self._reset_region()
            self._goto("navigate")
            return action_click_box(arrow, "ARROW_LEFT next region")

        # GAP: arrow cls not seen → symmetric left-edge coord, then navigate.
        if self._phase_ticks > _SWITCH_MAX:
            self.log("ARROW_LEFT cls missing — symmetric coord (GAP)")
            self._reset_region()
            self._goto("navigate")
            return action_click(*_ARROW_LEFT_POS, "ARROW_LEFT next region (GAP)")
        return action_wait(400, "waiting for ARROW_LEFT cls")

    # ── exit ──────────────────────────────────────────────────────────────

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if self.detect_screen_yolo(screen) == "Lobby":
            self.log("back in lobby, schedule done")
            return action_done("schedule complete")
        if self._phase_ticks > _EXIT_MAX:
            self.log("exit budget exhausted, reporting done")
            return action_done("schedule exit timeout")
        # Close a lingering popout before leaving.
        close = self._popout_close(screen)
        if close is not None and self._roster_open(screen):
            return action_click_box(close, "close popout on exit")
        home = self.find_cls(screen, UC.BTN_HOME, conf=_CLS_CONF)
        if home is not None:
            return action_click_box(home, "schedule exit: home (YOLO 回大厅按钮)")
        back = self.find_cls(screen, UC.BTN_BACK, conf=_CLS_CONF)
        if back is not None:
            return action_click_box(back, "schedule exit: back (YOLO 返回键)")
        return action_back("schedule exit: ESC toward lobby")


# Region tiles that HAVE a YOLO ui cls (region-select list). Ordered the way
# the list lays them out. Used only for is-schedule detection + the 夏莱办公室
# anchor; traversal is ARROW_LEFT-driven, NOT tile-clicking (tiles are
# under-trained — see _OFFICE_ROW_POS GAP note).
_SCHOOL_TILES = [
    UC.SCHOOL_OFFICE,      # 夏莱办公室   (list row 0 — full-circle anchor)
    UC.SCHOOL_DORM,        # 夏莱居住区
    UC.SCHOOL_GEHENNA,     # 格黑娜学院中央区
    UC.SCHOOL_ABYDOS,      # 阿拜多斯高中
    UC.SCHOOL_MILLENNIUM,  # 千年研究所
]
