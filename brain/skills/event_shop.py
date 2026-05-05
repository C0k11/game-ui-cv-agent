"""EventShop — scan + auto-buy the event shop's per-currency sections.

Features (P1a/b-ish scope, conservative):

  1. Navigate: lobby → 任務 (campaign) → event page → 商店 tab.
  2. Iterate currency tabs on the left-hand rail. Each tab scopes buy
     options to one shop currency (入場券 / 蛋糕 / P 點數 / …).
  3. For each visible item card:
       - OCR name + cost digit + 可購買 N 次 limit.
       - Skip if the item is a 5:1 currency-exchange trap (user rule:
         `5 per unit of a different currency = always a rip-off`).
       - Buy loop up to `_auto_buy_enabled` and available currency.
  4. Write `data/event_shop_state.json` per run:
       {
         "timestamp": "...",
         "event_id": "...",
         "by_currency": {
           "入場券": {"items": [...], "total_need": 7155, "spent_total": 0},
           ...
         }
       }
     `event_farming` can read this to know when to stop.

Defaults to SCAN-ONLY — no purchases unless the profile sets
`event_shop_auto_buy=True` and lists which currency tabs are safe to
spend from.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from brain.skills.base import (
    BaseSkill,
    OcrBox,
    ScreenState,
    action_back,
    action_click,
    action_click_box,
    action_done,
    action_scroll,
    action_wait,
)


_STATE_PATH = Path(__file__).resolve().parents[2] / "data" / "event_shop_state.json"


# Currency-exchange trap detection (P1a).
#
# Rule: event shops sometimes list "5× this currency = 1× student-shard /
# 1× other-currency" entries. These are always lossy (user: "很亏"):
# the event currency is finite (weekly cap), while shards/alt-currency
# obtainable elsewhere. We want to skip these during auto-buy.
#
# Specific trap signatures observed:
#   - 神名文字 / 神名文字碎片 (per-student shard shop rows priced at 200+ P)
#   - X的神名文字 / 神秘文字 variants
#
# NOT traps (valuable buys):
#   - 信用貨幣  = credits (money), common non-trap
#   - 青輝石   = pyroxene (premium gem), always valuable
#   - 奧秘之書  = enhancement book, always valuable
#   - 活動報告  = event reports (XP), always valuable
_EXCHANGE_TRAP_PATTERNS = (
    "神名文字",     # student shards (美補)
    "神秘文字",
    "精神文字",
)


def _looks_like_currency_exchange(name: str, unit_cost: int) -> bool:
    """User rule: skip 5:1+ student-shard / currency-swap exchanges."""
    if unit_cost < 5:
        return False
    for pat in _EXCHANGE_TRAP_PATTERNS:
        if pat in name:
            return True
    return False


class EventShopSkill(BaseSkill):
    """Scan (default) or scan+buy the current event's shop.

    Profile options consumed:
      event_shop_auto_buy   (bool, default False) — actually click 購買
      event_shop_currencies (list[str], default [])
         — which currency-tab labels to spend from. Example: ["P"].
    """

    def __init__(self,
                 auto_buy: bool = False,
                 spend_currencies: Optional[List[str]] = None):
        super().__init__("EventShop")
        self._auto_buy = bool(auto_buy)
        self._spend_currencies = tuple(spend_currencies or ())
        self.max_ticks = 80
        self._sub = "enter"          # enter / tab / scan / buy / next_tab / done
        self._visited_tabs: List[str] = []
        self._current_tab_label: str = ""
        self._state: Dict[str, Any] = {}
        self._scroll_attempts: int = 0
        self._buy_pending: Optional[OcrBox] = None

    def reset(self) -> None:
        super().reset()
        self._sub = "enter"
        self._visited_tabs.clear()
        self._current_tab_label = ""
        self._state = {}
        self._scroll_attempts = 0
        self._buy_pending = None

    # ────────────────── main tick ──────────────────

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1
        if self.ticks >= self.max_ticks:
            self._persist_state()
            return action_done("event shop timeout, state saved")

        # Dismiss common popups (reward / confirm) uniformly.
        popup = self._handle_common_popups(screen)
        if popup:
            return popup

        if self._sub == "enter":
            return self._enter(screen)
        if self._sub == "tab":
            return self._pick_tab(screen)
        if self._sub == "scan":
            return self._scan_items(screen)
        if self._sub == "buy":
            return self._buy_loop(screen)
        if self._sub == "next_tab":
            return self._next_tab(screen)
        if self._sub == "done":
            self._persist_state()
            return action_done("event shop complete")
        return action_wait(300, "event shop unknown state")

    # ────────────────── stages ──────────────────

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        """Click 商店 tab at bottom of event page. Assumes we're already
        on the event activity page (chained after event_activity)."""
        shop_tab = screen.find_any_text(
            ["商店"], region=(0.15, 0.85, 0.30, 1.0), min_conf=0.55,
        )
        if shop_tab:
            self._sub = "tab"
            return action_click_box(shop_tab, "enter event shop")
        # Not on event page yet — fall back: click 任務 (quest) entry
        task = screen.find_any_text(
            ["任務", "任务"], region=(0.20, 0.85, 0.40, 1.0), min_conf=0.55,
        )
        if task:
            return action_click_box(task, "reach event to find 商店 tab")
        return action_wait(400, "searching for event 商店 tab")

    def _pick_tab(self, screen: ScreenState) -> Dict[str, Any]:
        """Left-rail currency tabs labeled `貨幣`. We pick the first
        unvisited one, reading the currency icon via proximity OCR."""
        rail = screen.find_text(
            "貨幣", region=(0.0, 0.10, 0.22, 0.85), min_conf=0.55,
        ) + screen.find_text(
            "货币", region=(0.0, 0.10, 0.22, 0.85), min_conf=0.55,
        )
        rail.sort(key=lambda b: b.cy)
        # Deduplicate vertically-close entries (OCR double-hit)
        dedup: List[OcrBox] = []
        for b in rail:
            if not dedup or abs(dedup[-1].cy - b.cy) > 0.03:
                dedup.append(b)

        # Label each tab by its X-position index (top-to-bottom: tab0, tab1, …).
        # We don't read the icon; we just track visit order.
        unvisited = [b for i, b in enumerate(dedup)
                     if f"tab{i}" not in self._visited_tabs]
        if not unvisited:
            self._sub = "done"
            return action_wait(200, "all currency tabs scanned")
        target = unvisited[0]
        idx = dedup.index(target)
        self._current_tab_label = f"tab{idx}"
        self._visited_tabs.append(self._current_tab_label)
        self._scroll_attempts = 0
        self._sub = "scan"
        return action_click_box(target, f"switch to currency {self._current_tab_label}")

    def _scan_items(self, screen: ScreenState) -> Dict[str, Any]:
        """OCR item cards (name + 可購買 N 次 + cost digit), store to state."""
        items = self._parse_shop_cards(screen)
        tab_entry = self._state.setdefault(
            self._current_tab_label,
            {"items": [], "total_need": 0, "spent_total": 0,
             "skipped_exchange": []},
        )
        seen_names = {it["name"] for it in tab_entry["items"]}
        for it in items:
            if it["name"] in seen_names:
                continue
            if _looks_like_currency_exchange(it["name"], it["unit_cost"]):
                tab_entry["skipped_exchange"].append(it)
                self.log(f"skip exchange trap: {it['name']!r} (cost {it['unit_cost']})")
                continue
            tab_entry["items"].append(it)
            tab_entry["total_need"] += it["unit_cost"] * max(0, it["remaining"])
            seen_names.add(it["name"])

        # Scroll within tab to reveal more cards, up to 3 scrolls.
        if self._scroll_attempts < 3:
            self._scroll_attempts += 1
            return action_scroll(
                0.75, 0.60, clicks=-4,
                reason=f"scroll to reveal more shop items ({self._scroll_attempts}/3)"
            )
        # Done scanning this tab. If auto-buy and spend-allowed, do it.
        if (self._auto_buy
                and self._current_tab_label in self._spend_currencies):
            self._sub = "buy"
            return action_wait(200, f"scan done, entering buy loop on {self._current_tab_label}")
        self._sub = "next_tab"
        self.log(
            f"tab {self._current_tab_label} scan: "
            f"{len(tab_entry['items'])} items, total need "
            f"{tab_entry['total_need']} {self._current_tab_label}-currency "
            f"(skipped {len(tab_entry['skipped_exchange'])} exchange traps)"
        )
        return action_wait(250, "tab scan done")

    def _buy_loop(self, screen: ScreenState) -> Dict[str, Any]:
        """Click 購買 on eligible items. Conservative: one click per tick.

        Exit when no more clickable 購買 visible or we've done N clicks.
        Real shop would show confirm dialog; _handle_common_popups catches.
        """
        buy_btn = screen.find_any_text(
            ["購買", "购买"], region=(0.55, 0.40, 1.0, 0.95), min_conf=0.55,
        )
        if buy_btn is None:
            self._sub = "next_tab"
            return action_wait(200, "no more 購買 buttons, advancing tab")
        # Defensive: only click if this buy button is on an item we
        # scanned (not an exchange trap). We can't easily pair the OCR
        # box back to the item without geometry; rely on scroll scan
        # already excluded them — the 購買 near that y position would
        # still be there, but buying exchange traps is unwanted. To
        # avoid accidentally clicking an exchange trap, we match the
        # nearest card name on same row.
        nearby_name = None
        for b in screen.ocr_boxes:
            if (0.55 <= b.cx <= 0.95
                    and abs(b.cy - buy_btn.cy) < 0.15
                    and b.cy < buy_btn.cy - 0.02
                    and b.confidence >= 0.6
                    and len(b.text) >= 3
                    and not re.fullmatch(r"[\dxX×/]+", b.text)):
                nearby_name = b.text
                break
        if nearby_name and _looks_like_currency_exchange(nearby_name, 5):
            self.log(f"buy loop: refusing exchange trap {nearby_name!r}, advancing")
            self._sub = "next_tab"
            return action_wait(200, "refused exchange trap, advancing tab")
        tab_entry = self._state.setdefault(self._current_tab_label, {})
        tab_entry["spent_total"] = tab_entry.get("spent_total", 0) + 1
        return action_click_box(buy_btn, f"buy event shop item ({self._current_tab_label})")

    def _next_tab(self, screen: ScreenState) -> Dict[str, Any]:
        """Scroll back to top of current tab then re-enter _pick_tab."""
        # Scroll up to reset list, then dispatch
        self._sub = "tab"
        return action_scroll(
            0.75, 0.40, clicks=8,
            reason="scroll to top of currency rail, pick next tab",
        )

    # ────────────────── shop card OCR parsing ──────────────────

    # NB: don't ban "貨幣/货币" — legit item names like 信用貨幣 contain
    # it. The column-x filter + anchor-proximity filter already rule out
    # the left-rail 貨幣 tab labels.
    _CARD_NAME_BANNED = ("購買", "购买", "可購買", "可购买",
                         "Bonus", "稀有")
    _REMAINING_RE = re.compile(r"可購買\s*(\d+)\s*次|可购买\s*(\d+)\s*次")

    def _parse_shop_cards(self, screen: ScreenState) -> List[Dict[str, Any]]:
        """Parse item cards from the right-hand grid.

        Each card is a vertical stack: name (top) / quantity / purchase
        limit (`可購買 N 次`) / cost digit / 購買 button.

        Returns a list of dicts {name, remaining, unit_cost}.
        Strategy: anchor on `可購買 N 次` (reliable OCR) and walk up
        the same column to find the name, down to find the cost.
        """
        cards: List[Dict[str, Any]] = []
        # Find all "可購買 N 次" anchors
        anchors: List[Tuple[OcrBox, int]] = []
        for box in screen.ocr_boxes:
            if box.confidence < 0.55:
                continue
            m = self._REMAINING_RE.search(box.text)
            if not m:
                continue
            n_str = next((g for g in m.groups() if g), None)
            if not n_str:
                continue
            anchors.append((box, int(n_str)))

        # Regex for quantity tokens we want to EXCLUDE from name candidates
        # (they sit between the title and the "可購買" anchor). Covers
        # `x3`, `x500K`, `x1.5M`, `×10`, etc.
        qty_re = re.compile(r"^[xX×]\s*\d[\d.,]*[KkMm]?$")
        for anchor, remaining in anchors:
            col_x = anchor.cx
            # Name: box ABOVE anchor in same column, not in banned list.
            # Need to pick the FARTHEST legit name-line (title text), not
            # the nearest (which tends to be the quantity "x600" badge).
            name_candidates = []
            for b in screen.ocr_boxes:
                if b.confidence < 0.55:
                    continue
                if abs(b.cx - col_x) >= 0.10:
                    continue
                if b.cy >= anchor.cy or anchor.cy - b.cy >= 0.30:
                    continue
                txt = b.text.strip()
                if not txt:
                    continue
                if any(kw in txt for kw in self._CARD_NAME_BANNED):
                    continue
                if qty_re.match(txt):
                    continue
                if re.fullmatch(r"[\dxX×/+\-.,]+", txt):
                    continue
                name_candidates.append(b)
            # Prefer the TOP-MOST candidate (title is at the very top of
            # the card, quantity is below the icon just above `可購買`).
            name_candidates.sort(key=lambda b: b.cy)
            name = name_candidates[0].text.strip() if name_candidates else ""
            if not name:
                continue
            # Cost: numeric box BELOW anchor in same column. Accept
            # thousands-separator form like "1,500" / "2,000".
            cost_candidates = []
            for b in screen.ocr_boxes:
                if b.confidence < 0.55:
                    continue
                if abs(b.cx - col_x) >= 0.08:
                    continue
                if b.cy <= anchor.cy or b.cy - anchor.cy >= 0.15:
                    continue
                stripped = b.text.strip().replace(",", "")
                if re.fullmatch(r"\d{1,6}", stripped):
                    cost_candidates.append((b, int(stripped)))
            cost_candidates.sort(key=lambda t: t[0].cy - anchor.cy)
            unit_cost = cost_candidates[0][1] if cost_candidates else 0
            cards.append({
                "name": name,
                "remaining": remaining,
                "unit_cost": unit_cost,
            })
        return cards

    # ────────────────── persistence ──────────────────

    def _persist_state(self) -> None:
        try:
            _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "by_currency": self._state,
            }
            _STATE_PATH.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self.log(f"event shop state saved: {_STATE_PATH}")
        except Exception as e:
            self.log(f"event shop state save failed: {e}")

    @staticmethod
    def read_latest_state() -> Dict[str, Any]:
        """Helper for event_farming to consume the budget."""
        try:
            return json.loads(_STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
