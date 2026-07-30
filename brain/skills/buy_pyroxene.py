"""BuyPyroxeneSkill — claim the daily FREE combo pack (pure-YOLO rewrite).

Verified flow (interactive probe 2026-06-01, data/_buy_pyroxene_probe_log.md).
All clicks resolved through YOLO cls (ui_classes) — NO OCR, NO hardcoded pixel
positions. The ONLY purpose is to claim the **每日免費組合包** (gives AP×10 +
credits×10K, NOT pyroxene).

★★ HARD RULE — NEVER spend pyroxene / real money ★★
The shop has CAD$ packs sitting right next to the free one. We ONLY ever click
a 购买 (SHOP_BUY) button that has a 免费 (FREE) price-label directly above it in
the same column, and we ONLY confirm a purchase dialog that shows the 免费 cls.
Any dialog without 免费 (i.e. a CAD price) ⇒ cancel immediately.

State machine
-------------
enter      lobby → click SHOP_BUY_PYROXENE (购买青辉石). Retry on ADB drop
           (re-click while still on lobby). Wait for the shop popup.
combo_tab  shop opens on 特別販售 tab → click COMBO_PACK (组合包未选择) to switch
           to the 組合包 tab. Done when COMBO_PACK_SEL (组合包已选择) shows.
buy        on 組合包 tab: find FREE (免费) label, click the SHOP_BUY directly
           below it (same column). No FREE cls ⇒ today's pack already claimed
           ⇒ exit. NEVER click a 购买 without FREE above it.
confirm    "是否購買該商品？" dialog (BTN_CONFIRM + BTN_CANCEL). Poll for the
           FREE cls in the price area: present ⇒ BTN_CONFIRM; never appears
           ⇒ BTN_CANCEL (treat as unexpected paid item, abort).
reward     GOT_REWARD popup → dismiss via STORY_TAP_CONTINUE / GOT_REWARD header
           (NEVER tap screen center — that hits the item icons). Loop until gone.
exit       close shop via BTN_CLOSE_X (retry on drop) → lobby → done.

Detectors: base "ui" only (no avatar/battle/cafe) — not in SKILL_YOLO_MAP.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from brain.skills.base import (
    BaseSkill, ScreenState, YoloBox,
    action_click_box, action_wait, action_back, action_done,
)
from brain.skills import ui_classes as UC

# ── tuning knobs ─────────────────────────────────────────────────────────
_CLS_CONF = 0.30              # default UI cls confidence floor
# 免费(FREE) is a genuinely weak/flickery cls (14f) — on a settled frame it's
# 0.9 but it dips to 0.13-0.34 on transition frames. Use a TARGETED low floor
# for it only (NOT a global conf drop). v6: oversample 免费 (task #32).
_FREE_CONF = 0.20

# Purchase-confirm dialog button band (确认键 / 取消键 sit center-bottom ~y0.82).
_DIALOG_BAND = (0.28, 0.74, 0.72, 0.92)
# Price area inside the confirm dialog where the 免费 label sits (~0.66, 0.61).
_DIALOG_PRICE_REGION = (0.48, 0.48, 0.88, 0.74)
# Same-column tolerance when pairing a FREE label with its 购买 button.
_FREE_COL_DX = 0.10
# 組合包 TAB red-dot region (top-right of the tab, ~0.80,0.21). Dot present =
# the free pack is NOT yet claimed today (probe: clears after claiming). Used to
# tell "FREE flickering, keep polling" from "genuinely claimed".
_COMBO_DOT_REGION = (0.74, 0.14, 0.90, 0.30)
# CONTENT-area red dot = the badge that sits ON the free (unclaimed) pack
# (probe: 红点 ~0.39,0.32 over the free pack). Excludes the tab dot (cx>0.78),
# top bar (cy<0.10) and the page-arrow dot (cx>0.95). 红点 is a strong cls
# (0.9+) → robust free-pack locator when the weak 免费(14f) cls flickers out.
_CONTENT_DOT_REGION = (0.20, 0.25, 0.76, 0.50)
_BUY_FREE_POLL = 14          # poll this many ticks for the flickery 免费 cls

# Per-sub-state tick budgets — every phase is bounded, never dead-waits.
_ENTER_MAX = 22
_COMBO_MAX = 16
_BUY_MAX = 30                # buy 态现在自己管「点击→等框→重按」全程(2026-07-29)
_CONFIRM_MAX = 10            # poll the dialog for the FREE cls before bailing
# ⛔tick-vs-墙钟家族(2026-07-28): _CONFIRM_MAX 10 tick 被 zero-wait 压到
# 1.5-2.5s, 且「等对话框出现」和「等 免费 cls 渲染」共用同一个 _phase_ticks
# 不重置 — 对话框第 8 tick 才出现时只剩 2 tick 判 免费 → 真免费包被当付费
# 框 cancel 掉。拆成两个独立墙钟(方向都 fail-closed: 等得久只是晚 cancel,
# 绝不会多确认)。⚠FREE 判定 conf 保持 0.30 不降 — 降门槛放大「付费框误判
# 免费」方向, 钱闸只紧不松(agent 建议里降 conf 那半句已驳回)。
_FREE_POLL_SEC = 8.0         # 框出现后 poll 免费 cls(≈40+ 帧, 盖住 flicker)
_BUY_SETTLE_SEC = 2.5        # buy 点击后等确认框渲染的墙钟(过时重按)
_REWARD_WAIT_SEC = 8.0       # 確認已发后等 reward 弹窗证据的墙钟
_EXIT_MAX = 14


class BuyPyroxeneSkill(BaseSkill):
    """Claim the daily free combo pack. Never spends premium currency."""

    def should_run(self, screen: ScreenState) -> bool:
        """Run only when the lobby 购买青辉石 entry carries a red dot.

        The badge sits just above-right of the entry banner (probe: entry
        ~(0.115,0.359), red dot ~(0.149,0.345)), so dot_on_entry's strict
        inside-bbox test can miss it. We find the entry dynamically and scan a
        region expanded up/right of its bbox for a red dot. Entry not visible
        ⇒ defer (return True) — we're probably not on the lobby yet.
        """
        entry = self.find_cls(screen, UC.SHOP_BUY_PYROXENE, conf=0.40)
        if entry is None:
            return True
        region = (entry.x1 - 0.02, entry.y1 - 0.05, entry.x2 + 0.05, entry.y2 + 0.02)
        return self.dot_in_region(screen, region, dot_classes=(UC.DOT_RED,))

    # ── 竣工判据 ─────────────────────────────────────────────────────────
    def exit_report(self):
        """竣工判据 = 今天这个免费包到底领没领。

        ⛔2026-07-28 首次 live 跑通时出口报的是 `UNKNOWN — 未声明竣工判据`,
        正是「活干没干完没人审计」那一类(memory completion_gap)。而这个 skill
        更需要判据: 它**整个存在的意义**就是每天领一次, 领不到就是白跑,
        且它自己 `should_run` 靠红点门控 —— 红点漏检就会静默跳过一整天。
        """
        if self._bought:
            return ("CLEAN", "每日免費組合包已领(內容物 AP x10 + 信用点 x10K)")
        if self._skipped_already_claimed:
            return ("CLEAN", "今日免费包**先前已领**(屏上无 免费 标且无未领红点)")
        return ("LEFTOVER",
                f"免费包**没领到**(sub={self.sub_state}, 確認点击 "
                f"{self._confirm_clicks} 次) — 明天查 免费/红点 两条定位是否都漏检")

    def __init__(self):
        super().__init__("BuyPyroxene")
        # enter(~6)+combo(~4)+buy(~3)+confirm(~6)+reward(~6)+exit(~6) ≈ 31;
        # 80 leaves slack for ADB-drop retries.
        self.max_ticks = 80
        self._init_state()

    # ── state init / reset ────────────────────────────────────────────────

    def _init_state(self) -> None:
        self._phase_ticks: int = 0
        self._bought: bool = False         # set once the free pack is confirmed
        self._buy_retry: Optional[tuple] = None  # (cx,cy) to re-press on drop
        self._buy_fired: bool = False      # buy 点击已发(显式 flag, 裸 since 首调 0.0 会挡第一发)
        self._confirm_clicks: int = 0      # ack-loop: 確認 实际点击次数(防幻影)
        self._dialog_seen: bool = False    # 确认框首见(免费 poll 墙钟起点)
        # 「今天先前已领」的证据(屏上既无 免费 标也无未领红点) —— 与「没领到」
        # 区分开, 否则竣工判据每天都会误报 LEFTOVER。见 exit_report。
        self._skipped_already_claimed: bool = False

    def reset(self) -> None:
        super().reset()
        self._init_state()

    def _goto(self, sub_state: str) -> None:
        self.sub_state = sub_state
        self._phase_ticks = 0

    # ── shared cls helpers ────────────────────────────────────────────────

    def _shop_open(self, screen: ScreenState) -> bool:
        """Shop popup open = either combo-pack tab present (selected or not)."""
        return self.find_cls(
            screen, [UC.COMBO_PACK, UC.COMBO_PACK_SEL], conf=_CLS_CONF
        ) is not None

    def _on_combo_tab(self, screen: ScreenState) -> bool:
        return self.find_cls(screen, UC.COMBO_PACK_SEL, conf=_CLS_CONF) is not None

    def _confirm_dialog(self, screen: ScreenState) -> Optional[YoloBox]:
        """Purchase-confirm dialog = BOTH 确认键 and 取消键 in the button band.
        Returns the 确认键 box (the thing we click on a verified-free pack)."""
        confirm = self.find_cls(screen, UC.BTN_CONFIRM, conf=_CLS_CONF, region=_DIALOG_BAND)
        cancel = self.find_cls(screen, UC.BTN_CANCEL, conf=_CLS_CONF, region=_DIALOG_BAND)
        if confirm is not None and cancel is not None:
            return confirm
        return None

    def _free_buy_button(self, screen: ScreenState) -> Optional[YoloBox]:
        """The 购买 button of the FREE pack — paired geometrically with FREE.

        Find the 免费 (FREE) price label, then among all 购买 (SHOP_BUY)
        buttons pick the one directly BELOW it in the same column. Returns None
        when no FREE label is on screen (pack already claimed today) OR no 购买
        sits under it. This is the SOLE purchase path — a 购买 with no FREE
        above it is a CAD pack and is never returned."""
        buys = self.find_all_cls(screen, UC.SHOP_BUY, conf=_CLS_CONF)
        if not buys:
            return None
        # Primary: pair the 免费 label with the 购买 directly below it (same col).
        free = self.find_cls(screen, UC.FREE, conf=_FREE_CONF)
        if free is not None:
            cands = [b for b in buys if b.cy > free.cy and abs(b.cx - free.cx) < _FREE_COL_DX]
            if cands:
                return min(cands, key=lambda b: abs(b.cx - free.cx))
        # Fallback (免费 14f flickers out): the free/unclaimed pack carries a
        # CONTENT red-dot badge (strong cls). Pick the 购买 nearest-below it.
        # SAFE: the confirm dialog still REQUIRES 免费 → a CAD pack can never be
        # bought even if this mis-selects.
        cdot = self.find_cls(screen, UC.DOT_RED, conf=0.40, region=_CONTENT_DOT_REGION)
        if cdot is not None:
            cands = [b for b in buys if b.cy > cdot.cy and abs(b.cx - cdot.cx) < 0.14]
            if cands:
                self.log("免费 not detected → free-pack via content 红点 badge")
                return min(cands, key=lambda b: abs(b.cx - cdot.cx))
        return None

    def _close_x(self, screen: ScreenState) -> Optional[YoloBox]:
        return self.find_cls(screen, UC.BTN_CLOSE_X, conf=_CLS_CONF,
                             region=(0.55, 0.04, 0.95, 0.30))

    # ── tick: global guards + dispatch ─────────────────────────────────────

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1
        self._phase_ticks += 1

        if self.ticks >= self.max_ticks:
            self.log("timeout, exiting")
            return action_done("buy_pyroxene timeout")

        if screen.is_loading():
            return action_wait(600, "shop loading")

        if self.sub_state == "":
            self._goto("enter")

        handler = {
            "enter": self._enter,
            "combo_tab": self._combo_tab,
            "buy": self._buy,
            "confirm": self._confirm,
            "reward": self._reward,
            "exit": self._exit,
        }.get(self.sub_state)
        if handler is None:
            return action_wait(300, "buy_pyroxene unknown state")
        return handler(screen)

    # ── enter ───────────────────────────────────────────────────────────────

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        if self._shop_open(screen):
            self.log("shop popup open → combo tab")
            self._goto("combo_tab")
            return action_wait(300, "entered shop")

        if screen.is_lobby():
            entry = self.find_cls(screen, UC.SHOP_BUY_PYROXENE, conf=_CLS_CONF)
            if entry is not None:
                # Pace the retry (稳定规则 2026-06-11): the popup takes 1-2s to
                # render and the entry sits OUTSIDE it — an eager re-click
                # dismisses the popup we just opened (live-caught oscillation).
                # Click on tick 1 of every 3-tick window, settle otherwise.
                if self._phase_ticks % 3 != 1:
                    return action_wait(600, "entry clicked — settling for shop popup")
                self.log("clicking 购买青辉石 entry (paced retry)")
                return action_click_box(entry, "open buy-pyroxene shop")
            self.log("on lobby but no 购买青辉石 cls — YOLO gap; waiting")
            return action_wait(400, "waiting for 购买青辉石 cls")

        if self._phase_ticks > _ENTER_MAX:
            self.log("enter budget exhausted, giving up")
            return action_done("could not reach buy-pyroxene shop")
        # Unknown / transition screen — wait, then nudge back toward lobby.
        if len(screen.yolo_boxes or []) < 2:
            return action_wait(600, "no UI detected, likely loading")
        return self.nav_home(screen, "buy_pyroxene recover")

    # ── combo_tab ────────────────────────────────────────────────────────────

    def _combo_tab(self, screen: ScreenState) -> Dict[str, Any]:
        if not self._shop_open(screen):
            if screen.is_lobby():
                self._goto("enter")
                return action_wait(300, "combo_tab: back on lobby, re-enter")
            if self._phase_ticks > _COMBO_MAX:
                self.log("combo_tab: shop lost, exiting")
                self._goto("exit")
                return action_wait(300, "combo_tab lost shop → exit")
            return action_wait(400, "waiting for shop UI (combo_tab)")

        # Already on the 組合包 tab → go buy.
        if self._on_combo_tab(screen):
            self._goto("buy")
            return action_wait(250, "on 組合包 tab → buy")

        # On 特別販售 / 青輝石 tab → click the unselected 組合包 tab.
        tab = self.find_cls(screen, UC.COMBO_PACK, conf=_CLS_CONF)
        if tab is not None:
            self.log("switching to 組合包 tab (YOLO 组合包未选择)")
            return action_click_box(tab, "switch to combo-pack tab")

        if self._phase_ticks > _COMBO_MAX:
            self.log("combo tab cls never found, exiting")
            self._goto("exit")
            return action_wait(300, "no combo tab cls → exit")
        return action_wait(350, "waiting for 组合包 tab cls")

    # ── buy ──────────────────────────────────────────────────────────────────

    def _buy(self, screen: ScreenState) -> Dict[str, Any]:
        # A confirm dialog may already be up (e.g. retry race) → handle it.
        if self._confirm_dialog(screen) is not None:
            self._goto("confirm")
            return action_wait(200, "confirm dialog up → confirm")

        if not self._shop_open(screen):
            if screen.is_lobby():
                self._goto("enter")
                return action_wait(300, "buy: back on lobby, re-enter")
            if self._phase_ticks > _BUY_MAX:
                self._goto("exit")
                return action_wait(300, "buy lost shop → exit")
            return action_wait(400, "waiting for shop UI (buy)")

        free_buy = self._free_buy_button(screen)
        if free_buy is not None:
            # ⛔mutate-before-ack 同病漏网(2026-07-29 live 实锤): 旧码先
            # _goto("confirm") 再返回点击, 点击被稳定门吞("帧未稳定") 时 skill
            # 已在 confirm 等一个从没点出来的框 → 空转 21 tick → pipeline
            # stuck-20 兜底把商店页整个叉掉。07-21 修了 confirm 態的確認,
            # 这里是 _buy 的同形。修: 状态不跳, 停在 buy —— 对话框出现由顶部
            # 分支自然转 confirm; 被吞立即重按; 落地但渲染慢则墙钟节流重按。
            if self._buy_fired and self.action_suppressed:
                self.log("buy tap 被稳定门吞 — 立即重按(帧已稳定)")
                self._buy_fired = False
            if self._buy_fired and self.since("buy_fired") < _BUY_SETTLE_SEC:
                return action_wait(300, "buy tapped — waiting confirm dialog")
            self._buy_fired = True
            self.mark("buy_fired")
            self._buy_retry = (free_buy.cx, free_buy.cy)
            self.log(f"clicking FREE pack 购买 at ({free_buy.cx:.2f},{free_buy.cy:.2f})")
            return action_click_box(free_buy, "buy FREE daily combo pack")

        # 按过购买后的"无 FREE"不再是已领证据 — 确认框弹入过渡帧会把价签/
        # 红点一起遮住, 这里误判 already-claimed 会把真免费包放跑。等对话框
        # 渲染(顶部分支接), 超时由 _BUY_MAX 收口。
        if self._buy_fired:
            if self._phase_ticks > _BUY_MAX:
                self._goto("exit")
                return action_wait(250, "buy fired but no dialog ever → exit")
            return action_wait(350, "buy fired — waiting dialog render")

        # No FREE this frame. Distinguish "已领" from "免费 just flickering" via
        # the 組合包 TAB red dot (probe: dot present ⇒ NOT claimed yet).
        combo_dot = self.find_cls(screen, UC.DOT_RED, conf=0.30, region=_COMBO_DOT_REGION)
        if combo_dot is not None and not self._bought:
            # Unclaimed (dot present) but 免费 not detected → it's flickering
            # (weak 14f cls). Keep polling many ticks before giving up.
            if self._phase_ticks > _BUY_FREE_POLL:
                self.log("⚠️ 组合包红点在但免费cls始终未检出(弱cls欠训) → 退出不盲买, 待v6补样本")
                self._goto("exit")
                return action_wait(250, "FREE undetected despite dot → safe exit")
            return action_wait(350, f"免费 flickering (红点在), polling ({self._phase_ticks})")

        # No FREE AND no combo red dot (or we already bought) → genuinely done.
        if self._phase_ticks > 4:
            why = "claimed just now" if self._bought else "no 红点 → already claimed today"
            # 竣工判据要能区分「今天领到了」和「今天先前已领」——两者都是 CLEAN,
            # 但只有第三种(既没领到又没证据说已领)才是 LEFTOVER。
            if not self._bought:
                self._skipped_already_claimed = True
            self.log(f"no FREE + no combo红点 → done ({why})")
            self._goto("exit")
            return action_wait(250, f"no free pack → exit ({why})")
        return action_wait(300, "settling combo tab before concluding")

    # ── confirm ──────────────────────────────────────────────────────────────

    def _confirm(self, screen: ScreenState) -> Dict[str, Any]:
        # Reward popup already showing (confirm registered) → reward state.
        if (self.find_cls(screen, UC.GOT_REWARD, conf=_CLS_CONF) is not None
                or self.find_cls(screen, UC.STORY_TAP_CONTINUE, conf=_CLS_CONF) is not None
                # interceptor 上一 tick 替我们关掉了奖励弹窗 → 那**就是**落地证据
                # (2026-07-25 live: 不认这条会去重按已消失的免費購買键)
                or self.interceptor_handled == "reward"):
            self._bought = True
            self._goto("reward")
            return action_wait(200, "reward popup up → reward")

        confirm_btn = self._confirm_dialog(screen)
        if confirm_btn is None:
            # 2026-07-29 重构: 进 confirm 的唯一入口是"对话框真在屏"(buy 顶部
            # 分支), 这里框没了只有两种情况 — ①確認已发, 框在关闭动画里 →
            # 停住等 reward 证据(顶部分支接), 超时 fail-closed exit; ②误检
            # 闪没/被吞 → 回 buy 重走(buy 态管重按与节流)。旧版的盲坐标
            # re-press 删除: 它在大厅上就是幽灵点击(今天 step_walk 实拦)。
            if self._confirm_clicks > 0:
                if (self._phase_ticks > _CONFIRM_MAX
                        and self.since("confirm_fired") > _REWARD_WAIT_SEC):
                    self.log("確認已发但始终无 reward 证据 → exit (fail-closed)")
                    self._goto("exit")
                    return action_wait(300, "confirmed, no reward evidence → exit")
                return action_wait(300, "confirm fired — waiting reward popup")
            if self._phase_ticks >= 3:
                self._dialog_seen = False   # 下次进 confirm 重新起 poll 表
                self.clear_timer("dialog_seen")
                self._goto("buy")
                return action_wait(250, "dialog gone → back to buy")
            return action_wait(300, "waiting for purchase-confirm dialog")

        # ★ SAFETY GATE: only confirm when the 免费 cls is in the price area.
        free_in_dialog = self.find_cls(
            screen, UC.FREE, conf=_CLS_CONF, region=_DIALOG_PRICE_REGION
        )
        if free_in_dialog is not None:
            # 2026-07-21 逐帧审实锤 mutate-before-ack: 旧码先 _bought=True+
            # _goto("reward") 再返回 確認 点击 — 点击被稳定门吞(reason 无"確認"
            # 不豁免)/丢 tap 时状态已跳 reward → pipeline 以为领了实际每日免費
            # 包没领(信用点未+10K)。修: 不提前跳状态, 停在 confirm, 顶部
            # reward-popup 检查=落地唯一确认; reason 含"確認键"→ 稳定门豁免立即
            # 点(渲染好的确认键=看到就点); 计数 cap fail-closed。
            self._confirm_clicks += 1
            self.mark("confirm_fired")   # reward 等待墙钟起点(显式 mark, 裸 since 首调 0.0)
            if self._confirm_clicks > 6:
                self.log("⛔ 確認 点了6次仍无 reward 弹窗 → exit (fail-closed)")
                self._goto("exit")
                return action_wait(300, "confirm stuck → exit")
            self.log(f"dialog shows 免费 → 確認 (fire {self._confirm_clicks})")
            return action_click_box(confirm_btn, "confirm FREE purchase (確認键)")

        # Dialog up but no FREE yet. FREE is static here, so poll a few frames
        # (rule #5). If it never shows, this is NOT a free pack → cancel.
        # 墙钟从「第一次看到对话框」起算(独立于等框阶段), 帧数×墙钟合取。
        if not self._dialog_seen:
            self._dialog_seen = True
            self.mark("dialog_seen")
        if (self._phase_ticks > _CONFIRM_MAX
                and self.since("dialog_seen") > _FREE_POLL_SEC):
            self.log("⛔ dialog has NO 免费 cls (paid?) — cancelling, never buy")
            cancel = self.find_cls(screen, UC.BTN_CANCEL, conf=_CLS_CONF, region=_DIALOG_BAND)
            if cancel is not None:
                self._goto("exit")
                return action_click_box(cancel, "cancel non-free purchase")
            close = self._close_x(screen)
            if close is not None:
                self._goto("exit")
                return action_click_box(close, "close non-free dialog")
            self._goto("exit")
            return action_back("abort non-free purchase (ESC)")
        return action_wait(250, "polling dialog for 免费 cls")

    # ── reward ───────────────────────────────────────────────────────────────

    def _reward(self, screen: ScreenState) -> Dict[str, Any]:
        # Prefer the 点击继续字样 strip; else tap the 获得奖励 header box.
        # NEVER tap screen center (probe: that hits the item icons, no advance).
        cont = self.find_cls(screen, UC.STORY_TAP_CONTINUE, conf=_CLS_CONF)
        if cont is not None:
            self.log("dismiss reward (点击继续字样)")
            return action_click_box(cont, "dismiss reward via continue")
        got = self.find_cls(screen, UC.GOT_REWARD, conf=_CLS_CONF)
        if got is not None:
            self.log("dismiss reward (获得奖励 header)")
            return action_click_box(got, "dismiss reward via header")

        # Neither present → reward closed → back on combo tab → exit.
        if self._shop_open(screen) or self._phase_ticks > 2:
            self.log("reward dismissed → exit")
            self._goto("exit")
            return action_wait(250, "reward done → exit")
        return action_wait(300, "waiting for reward popup to settle")

    # ── exit ─────────────────────────────────────────────────────────────────

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if screen.is_lobby():
            # is_lobby() is TRUE even with a residual popup over the nav (the
            # lobby entries peek out behind it) — declaring done here left the
            # free-pack popup open and starved Club of 社交入口 for 50 ticks
            # (live 2026-06-12 t80-128). Popup gone = actually done.
            close = self._close_x(screen)
            if close is not None:
                return action_click_box(close, "close residual popup before done")
            self.log("back in lobby, buy_pyroxene done")
            return action_done("buy_pyroxene complete")
        if self._phase_ticks > _EXIT_MAX:
            self.log("exit budget exhausted, reporting done")
            return action_done("buy_pyroxene exit timeout")
        # Close the shop popup (ADB drop on the X is common — retry).
        close = self._close_x(screen)
        if close is not None:
            return action_click_box(close, "close shop popup (X)")
        home = self.find_cls(screen, UC.BTN_HOME, conf=_CLS_CONF)
        if home is not None:
            return action_click_box(home, "exit: home button")
        return self.nav_home(screen, "buy_pyroxene exit")
