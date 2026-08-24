"""CafeSkill — Blue Archive cafe daily routine (pure-YOLO rewrite).

Verified flow (interactive probe 2026-06-01), all clicks resolved through
YOLO cls (ui_classes) or the emoticon / fused_avatar detectors — NO OCR, NO
hardcoded button positions (only relative gesture params: the headpat
cx-offset and the swipe start/end points).

State machine
----
enter     lobby  click NAV_CAFE  wait for Cafe page.
earnings  click CAFE_EARNINGS  popup  click an active CLAIM cls
          (CLAIM_REWARD_YELLOW / CLAIM_YELLOW). 0% earnings = no claim cls
          bail after a few ticks (never dead-wait).
invite    open CAFE_INVITE_TICKET  MomoTalk list. Each row = an avatar
          (fused_avatar model, model_tag=="avatar", cls_name=中文角色名,
          cx≈0.35) + a CAFE_INVITE_BTN (cx≈0.60). Pair avatar.cy<->button.cy,
          click the target row's invite button  BTN_CONFIRM in the
          "邀請XXX到咖啡廳" dialog. Up to 2 invites per cafe (1F target, 2F
          target from config). Scroll to find the target; fall back to the
          first rows.
headpat   emoticon model 'Emoticon_Action' (conf≥0.55). Click cx+0.025
          (student body, right of the bubble). Exclude the top-left
          指定/隨機訪問 false-positive zone (cx<0.27 & cy<0.34). Dedup recent
          clicks. loop-until-dry (3 empty frames), then pan left / right to
          sweep the rest of the floor.
switch    CAFE_MOVE_2F  2F (may pop a 訪問學生目錄 tutorial  BTN_CONFIRM).
          On 2F the button reads CAFE_MOVE_1F. Repeat invite + headpat on 2F.
exit      is_lobby  done; else BTN_HOME / BTN_BACK.

Detectors (set by pipeline.SKILL_YOLO_MAP["Cafe"] = "ui+cafe+avatar"):
  ui      — all the UC.* button classes (find_cls / find_all_cls). From ui v6
            this ALSO carries 'Emoticon_Action' (cls451) — the headpat bubble,
            folded in from the old standalone emoticon model.
  cafe    — legacy standalone emoticon model (single class 'Emoticon_Action').
            Auto-dropped by pipeline once ui v6 is active; until then it serves
            the headpat marker. _emoticon_mark matches by the "emoticon"
            substring, so it works regardless of which model produced the box.
  avatar  — fused_avatar (251 student heads), model_tag=="avatar".
"""
from __future__ import annotations

import json
import re
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

#  tuning knobs
_CLS_CONF = 0.30            # default UI cls confidence floor
_EMOTICON_CONF = 0.40       # headpat marker. 0.550.40 STOPGAP (user 2026-06-13):
#   v10 folded the retired emoticon v26n (mAP 0.995) into ui cls 451; measured on
#   650 cafe frames its 451 conf is weak/wide (mean 0.65, 18/51 detections <0.55)
#    real 摸头标识 fell under the 0.55 floor  student missed. Lower the floor to
#   catch them; FP risk (指定訪問/隨機訪問 mis-as-451) backstopped by _FP_ZONE +
#   domain dedup. PROPER FIX = v11: retrain 451 with ALL headpat material (old
#   v26n set + new frames) to restore 0.99-grade detection, then raise floor back.
_AVATAR_CONF = 0.25         # fused_avatar row-head confidence (registry default)
_HEADPAT_DX = 0.025         # click this much right of the bubble = student body
_HEADPAT_DEDUP_DIST = 0.035  # skip clicks within this of a recent headpat
                             # (0.06 过肥: 2026-07-21 实锤 0.05 间距两真学生被并)
_HEADPAT_DEDUP_KEEP = 4     # remember the last N headpat coords for dedup
_HEADPAT_DRY_FRAMES = 7     # consecutive empty frames  current view cleared
#   3 (840ms) was too eager — emoticon bubbles render 400-600ms late after a
#   pan/pat settle, so a slow bubble got declared "dry" and a student was
#   missed (deep-dive r2 H3; live 2026-06-09 1F漏摸一个). 57 (user 2026-06-13
#   "要给足摸头处理时间" — 漏摸复发: 7×280≈2s scan before declaring a region dry,
#   extra margin for late bubbles / v10-folded-451 lower-conf renders). 精确根因
#   (时序 vs v10 451检测弱 vs pan盲区) 待明天 live 走一步看漏摸那帧确认。
_MAX_HEADPATS_PER_FLOOR = 12  # safety helmet (probe: ~12 students/floor)
# 墙钟下限(2026-07-28) —— 与上面的帧数**合取**, 见 _init_state 的换算说明。
# 取值忠实于各自注释里写明的设计意图, 不趁机收窄(那些数都是修事故调大的)。
_HEADPAT_DRY_SEC = 2.0    # 注释原话 "7×280≈2s scan before declaring a region dry"
_PAT_SETTLE_SEC = 1.4     # 注释原话 "23 给足时间: late bubbles after a pan"(3×450)
_LOBBY_DWELL_SEC = 1.2    # 大厅徽标渐入(3×350); live 2026-06-09 事故加的驻留窗
_SWITCH_2F_ACK_SEC = 4.0  # 点 移动至2号点  2F 到达证据(CAFE_MOVE_1F 出现)
# 点「領取」 帧证据(按钮变灰 / 弹窗链关掉)。要覆盖自主跑与 step 门控两种节奏
# (门控每步 4-6s), 故取 6.0s —— 放宽近乎零代价: 真领到了下一 tick 就走灰按钮/
# 到达分支, 等不满; 放窄则退回"点被吞却报领完"的老病。
_CLAIM_ACK_SEC = 6.0
_MAX_PANS_PER_FLOOR = 6       # safety helmet on camera sweeps
_INVITES_PER_FLOOR = 1        # invite 1 student per floor (2 total per cafe)
_INVITE_MAX_SCROLLS = 12      # rows to scroll hunting the target
_INVITE_SWIPE_SETTLE = 3      # ticks to wait after a list swipe (anim settle)
_ROW_PAIR_DY = 0.06           # avatar.cy <-> invite-button.cy max gap = same row

# Top-left false-positive zone: 指定訪問/隨機訪問 buttons get mis-detected as
# Emoticon_Action. The buttons hug the LEFT EDGE (cx≈0.04-0.08, cy≈0.25-0.40) —
# the old (0.27, 0.34) zone was 3× too wide and KILLED a real student bubble
# at upper-left (live 2026-06-09: Emoticon_Action 0.97 on a student, rejected,
# never patted — user caught it on screen). Domain-authority dedup in
# _run_yolo_on_image is the main FP layer now; this zone is just a backstop.
_FP_ZONE = (0.10, 0.45)       # cx < 0.10 AND cy < 0.45  reject

# Per-sub-state tick budgets — every phase is bounded, never dead-waits.
_ENTER_MAX = 25
_EARNINGS_MAX = 18
_INVITE_MAX = 45
_SWITCH_MAX = 18
_EXIT_MAX = 14


def _game_day() -> str:
    """BA game-day ISO date (resets 04:00 **server time**) — invite-state key.

    与 schedule.py 的 `_game_day` 同一处修正(2026-07-27 实测): 旧版 "resets
    04:00 **local**" 这句本身就是 bug —— 游戏日界锚在服务器/设备时区, 不是 host。
    实测 host=UTC-4 / device=Asia/Shanghai  差 12h, host 跨过 04:00 时台账 key
    在**同一个游戏日内**翻天, 招待券台账被判过期重置。
    """
    from datetime import timezone
    _SERVER_TZ = timezone(timedelta(hours=8))       # BA 繁中服
    return (datetime.now(_SERVER_TZ) - timedelta(hours=3)).date().isoformat()


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

    Same pattern as BountySkill._load_enabled_branches: app_config.json
    active_profile  profiles[active].cafe_invite_targets. Returns a list of
    中文角色名 matching fused_avatar cls_name (e.g. "莉央(战斗)"). Index 0 is
    the 1F target, index 1 the 2F target; extra entries are accepted as
    further fallbacks. Empty list  caller invites the first visible rows.
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


def _load_skip_invite() -> bool:
    """app_config profile `cafe_skip_invite` — True  headpat-only cafe run
    (invite ticket on CD / user choice). Same profile-read pattern as
    _load_invite_targets."""
    try:
        if not _APP_CONFIG_FILE.exists():
            return False
        data = json.loads(_APP_CONFIG_FILE.read_text("utf-8"))
        active = data.get("active_profile", "default")
        profile = (data.get("profiles") or {}).get(active, {})
        return bool(profile.get("cafe_skip_invite", False))
    except Exception:
        return False


class CafeSkill(BaseSkill):
    # Lobby cafe entries that carry a red/yellow dot when there's something
    # to do (earnings ready / invite slot open). Drives should_run gating.
    _LOBBY_DOT_ENTRIES = [UC.NAV_CAFE, UC.CAFE_INVITE_TICKET, UC.CAFE_EARNINGS]

    def should_run(self, screen: ScreenState) -> bool:
        # 2026-07-28 live 实锤: 这道门控**只在大厅上成立**, 名字里的 LOBBY
        # 就是这个意思, 但旧码不管在哪一页都拿它判。
        # 后果(当场帧证据): bot 停在**咖啡厅 1 号店内**时, 屏上有
        #   `咖啡厅收益 0.98`(= _LOBBY_DOT_ENTRIES 里的 CAFE_EARNINGS)
        # 它被当成"入口"参与判定, 而它身上当然没有黄点(黄点在 移动至2号点 上)
        #  dot_on_entry 返回 False  **整个 cafe 被 dot-gate skip**,
        #   而那一刻屏上挂着 **收益 108,334 信用点 / 123 AP / 9,167 信用点**
        #   和 **3 个 Emoticon_Action(3 个学生等着摸头)**。
        # 根因: CAFE_INVITE_TICKET / CAFE_EARNINGS 是**咖啡厅内部**的 cls,
        # 大厅上根本不会出现 —— 把它们放进"大厅入口"列表, 等于给"人已经在
        # 咖啡厅里"这种情况装了一个必然为假的判据。
        #  不在大厅就**判不了**, 按 dot_on_entry 一贯的语义放行(fail-open 到
        #   "进去看一眼", 而不是 fail-closed 到"今天不干了")。
        if not screen.is_lobby():
            return True
        return self.dot_on_entry(screen, self._LOBBY_DOT_ENTRIES)

    def __init__(self):
        super().__init__("Cafe")
        # 1F invite(~12)+headpat(~25)+switch(~5)+2F invite(~25)+headpat(~25)
        # ≈ 95 ticks worst-case; 160 leaves slack for retries/loads.
        self.max_ticks = 240   # 160 于 2026-07-21 在 2F pan 后精确超时, 帧上
                               # 2 泡泡未拍; invite 空滚修复后仍留 ~80 tick 余量
        self._init_state()

    #  state init / reset

    def _init_state(self) -> None:
        self._phase_ticks: int = 0          # ticks spent in current sub_state
        self._enter_attempts: int = 0
        # "UI 被隐藏"必须**连续多帧**成立才算数(2026-08-07 live 实锤, 见 _enter):
        # 单帧零导航 cls 的成因里混着**转场帧**(进咖啡厅时学生先渲染、UI chrome
        # 还没出来  恰好"有摸头标记 + 零导航 cls"), 而那一步的落点是写死坐标,
        # 误触会点进家具編輯模式。真 UI-隐藏是持续状态, 转场帧是一瞬  用计数分开。
        self._ui_hidden_ticks: int = 0
        # earnings
        self._earnings_done: bool = False
        self._earnings_claimed: bool = False  # set after claim  close popup proactively
        # invite
        self._invite_targets: List[str] = []
        self._invited: Set[str] = set()      # cls_names invited this run
        self._invite_floor_done: bool = False  # this floor's invite finished
        self._invite_stage: int = 0          # 0=open ticket 1=list 2=confirm
        self._invite_pending_name: Optional[str] = None  # 落账延迟到确认发出
        self._invite_scrolls: int = 0
        self._invite_settle: int = 0         # post-swipe settle countdown
        self._invite_last_sig: str = ""
        self._invite_sig_repeat: int = 0
        self._invite_retry_btn: Optional[Tuple[float, float]] = None
        # headpat
        self._pat_count: int = 0
        # 竣工判据用的**粘性**计数(2026-07-28): `_pat_count` 每层清零
        # (_switch 里 =0), `_earnings_claimed` 领完也会被复位成 False ——
        # 拿它们当"今天干了什么"的证据必然低报。另存一份只增不减的。
        self._pats_total: int = 0
        self._earnings_done: bool = False
        self._empty_frames: int = 0
        self._pan_count: int = 0
        self._pan_dir: int = 0               # 0=none yet, 1=swept left, 2=right
        self._pat_settle: int = 0            # post-pat / post-pan settle
        self._recent_pats: List[Tuple[float, float]] = []
        # floor bookkeeping
        self._on_2f: bool = False
        # one-shot badge-verified re-entry (exit sees lobby dot  missed work)
        self._verify_reentered: bool = False
        self._exit_lobby_ticks: int = 0   # dwell counter for the badge check
        # 2026-07-28 墙钟化(tick-vs-wallclock 家族, 全仓审计发现 cafe 是
        # **零墙钟** 的 9 个 skill 之一 —— 所有时序判据都在数 tick)。
        # 真正的机制不是"tick 变快了": server/app.py:1519 的 ZERO-WAIT 只对
        # reason 含 加载中/loading 的 wait 兑现 duration_ms, **其余一律睡 0.12s**
        # (代码注释自己写着 "counters are squashed to a fast re-poll")。
        #  本文件所有 `action_wait(280/350/450, ...)` 的毫秒数**全部被丢弃**,
        #   于是三条为修漏摸事故调大的判据同时缩水:
        #     · _HEADPAT_DRY_FRAMES=7  注释自己算的 "7×280≈2s"  实际 ~1.4s
        #     · _pat_settle=3          设计 3×450=1.35s+  实际 ~0.6s
        #     · _exit_lobby_ticks<3    设计 3×350=1.05s+  实际 ~0.4s
        #   时序铁证: _HEADPAT_DRY_FRAMES 57 的修复日期 user **2026-06-13**,
        #   ZERO-WAIT 政策日期 user **2026-06-14** —— **修复落地第二天就被作废**,
        #   而注释里那句"待明天 live 走一步确认"的"明天"正是那一天。
        # 修法用**合取**而不是单纯把帧数调大: 帧数抗单帧抖动, 墙钟抗渐入/延迟渲染,
        # 两者要的是不同的东西, 缺一不可(verifier 对 :1151 的建议, 我认同并推广)。
        self._pat_settle_sec: float = 0.0  # 本次 settle 还要等多少秒(配合帧数)
        self._switch_issued: bool = False  # 移动至2号点 已点(等到达证据)
        self._floor_back_done: bool = False  # one-shot 2F1F dot-driven re-sweep
        self._pct_retries: int = 0           # earnings % OCR retry counter

    def reset(self) -> None:
        super().reset()
        self._init_state()
        self._invite_targets = _load_invite_targets()
        self._skip_invite = _load_skip_invite()
        if self._skip_invite:
            self.log("cafe_skip_invite=true — invite disabled (券CD), headpat only")
        elif self._invite_targets:
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

    #  shared cls helpers

    def _close_x(self, screen: ScreenState,
                 region=(0.56, 0.04, 0.94, 0.30)) -> Optional[YoloBox]:
        return self.find_cls(screen, UC.BTN_CLOSE_X, conf=_CLS_CONF, region=region)

    def _read_earnings_pct(self, screen: ScreenState, earn_box: YoloBox) -> Optional[float]:
        """DIGIT-ONLY read of the X.X% shown on the 咖啡厅收益 button.

        Signal gate for the earnings popup: 0.0%  nothing to claim  never
        open it. Returns None when unreadable — caller opens the popup as
        before (safe fallback, identical to pre-gate behaviour)."""
        try:
            from brain.pipeline import run_digit_ocr
            # The 咖啡厅收益 cls box covers ONLY the title text (h≈0.025);
            # the X.X% value renders BELOW it (offline-verified 2026-06-09:
            # in-box crops  None, below-box crop  '0' on a 0.0% frame).
            # 2026-07-27 改成**框自身尺寸**为单位(原来是屏幕比例 0.01/0.05)。
            # 咖啡厅收益 框实测 iw 0.0793 / ih 0.0247 (n=3100, 四分位很紧),
            # 故 0.010.126 框宽, 0.052.02 框高 —— 16:9 下等价, 换分辨率/
            # 窗口大小不再漂(理由见 brain.pipeline.icon_strip)。
            _bw = max(1e-6, earn_box.x2 - earn_box.x1)
            _bh = max(1e-6, earn_box.y2 - earn_box.y1)
            x1 = max(0.0, earn_box.x1 - 0.126 * _bw)
            y1 = earn_box.y2
            x2 = min(1.0, earn_box.x2 + 0.126 * _bw)
            y2 = min(1.0, earn_box.y2 + 2.02 * _bh)
            raw = run_digit_ocr(screen.frame, (x1, y1, x2, y2)) or ""
            m = re.search(r"(\d+(?:\.\d+)?)", str(raw))
            if not m:
                return None
            return float(m.group(1))
        except Exception:
            return None

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

    #  竣工判据
    def exit_report(self):
        """咖啡厅的竣工判据 = 收益领了没 / 摸头摸了几个 / 邀请卷用了没。

        2026-07-28 首次纳入编排时出口报 `UNKNOWN — 未声明竣工判据`。
        咖啡厅是**最容易"跑通了但没干活"**的 skill: 收益弹窗动画锚点会打空
        (memory cafe_flow_spec 那个"收益第1次没领"), 摸头靠 `Emoticon_Action`
        这个弱 cls(mean conf 0.65)且学生在走动, 漏一个人从日志上完全看不出来。
        三项都用**粘性**计数 —— `_pat_count` 每层清零、`_earnings_claimed`
        领完就复位, 拿它们报数必然低报。
        """
        _pats, _inv = self._pats_total, len(self._invited)
        _bits = [f"摸头 {_pats} 人", f"邀请 {_inv} 人",
                 "收益已领" if self._earnings_done else "**收益没领到**"]
        if not self._earnings_done:
            return ("LEFTOVER", " / ".join(_bits) +
                    " — 收益是每天必领的, 查「領取」是否锚在弹出动画帧上")
        if _pats == 0:
            return ("LEFTOVER", " / ".join(_bits) +
                    " — 一个都没摸到, 查 Emoticon_Action 检出")
        return ("CLEAN", " / ".join(_bits))

    def _goto(self, sub_state: str) -> None:
        """Switch sub_state and reset the per-phase tick counter."""
        self.sub_state = sub_state
        self._phase_ticks = 0

    #  tick: global popup guards + dispatch

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1
        self._phase_ticks += 1

        if self.ticks >= self.max_ticks:
            self.log("timeout, exiting")
            return action_done("cafe timeout")

        #  popups that can appear in any sub_state (pure YOLO)

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

        #  state machine
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

    #  enter

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        self._enter_attempts += 1
        page = self.detect_screen_yolo(screen)

        # 2026-07-28 live 实锤(差点又丢一次收益): 收益弹窗有**两种版式** ——
        #  · 小弹窗: 咖啡厅 UI 露在外面  `_is_cafe`/page=="Cafe" 成立
        #  · **大弹窗**「咖啡廳收益」(1號店/1號店/2號店 三格明细): **盖住底栏**
        #     page != "Cafe"  旧码走 recover **点叉叉把它关掉**
        # 实测那一刻屏上明明白白挂着 `領取_黄 0.97` + 收益
        # **108,334 信用点 / 123 AP / 9,167 信用点** —— 关掉就等于把这笔钱扔了
        # (下一轮 dot-gate 还可能因为"入口无黄点"直接 skip 整个 cafe)。
        #  收益是**钱**, 遵循「目标 cls 出现立即点」: 只要 claim band 里有可点的
        #   領取(黄/蓝), 不管当前认不认得出这是哪一页, 一律先转 earnings 去领,
        #   绝不先关窗。这条要放在 page 判定**之前**。
        if self.find_cls(screen, [UC.CLAIM_REWARD_YELLOW, UC.CLAIM_YELLOW,
                                  UC.CLAIM_BLUE], conf=_CLS_CONF,
                         region=(0.28, 0.50, 0.74, 0.90)) is not None:
            self._enter_attempts = 0
            self.log("enter 阶段就看到可领的 領取  先去领钱, 绝不先关窗")
            self._goto("earnings")
            return action_wait(300, "看到可领收益  earnings")

        if page == "Cafe":
            self._enter_attempts = 0
            self.log("inside cafe, claiming earnings")
            self._goto("earnings")
            return action_wait(400, "entered cafe")

        if page == "Lobby":
            nav = self.click_cls(screen, UC.NAV_CAFE, "click cafe nav", conf=_CLS_CONF)
            if nav:
                return nav
            # 兜底1: 降 conf 找弱检 (v6b 偶尔出 ~0.25 弱框)。
            nav_weak = self.click_cls(screen, UC.NAV_CAFE, "click cafe nav (弱检兜底)", conf=0.12)
            if nav_weak:
                return nav_weak
            # 兜底2: v6b 咖啡厅入口实时常失效 (even@0.12 漏, live 2026-06-05 实测)
            # 但其他底栏入口检出 OK。底栏 8 入口等距 (LOBBY_NAV_ICONS, 咖啡厅=index0),
            # 用 ≥2 个检出的入口线性回归外推咖啡厅 cx (不硬编码绝对位置, 随检出自适应)。
            # 根治靠飞轮补咖啡厅入口样本  v6c。
            pts = []
            for i, cn in enumerate(UC.LOBBY_NAV_ICONS):
                b = self.find_cls(screen, cn, conf=0.30)
                if b is not None:
                    pts.append((i, b.cx, b.cy))
            if len(pts) >= 2:
                n = len(pts)
                sx = sum(p[0] for p in pts); sy = sum(p[1] for p in pts)
                sxx = sum(p[0] ** 2 for p in pts); sxy = sum(p[0] * p[1] for p in pts)
                denom = n * sxx - sx * sx
                if denom != 0:
                    a = (n * sxy - sx * sy) / denom        # slope
                    cafe_cx = (sy - a * sx) / n             # intercept = index0 = 咖啡厅
                    cafe_cy = sum(p[2] for p in pts) / n
                    if 0.0 < cafe_cx < 0.5:                 # sanity: 咖啡厅在左侧
                        self.log(f"咖啡厅入口外推 cx={cafe_cx:.3f} cy={cafe_cy:.3f} (从 {n} 个底栏入口)")
                        return action_click(cafe_cx, cafe_cy, "cafe nav (底栏外推兜底)")
            self.log("on lobby but no 咖啡厅入口 cls (even @0.12 + 外推失败) — YOLO gap; waiting")
            return action_wait(400, "waiting for 咖啡厅入口 cls")

        if page is not None:
            self.log(f"wrong screen '{page}', backing out")
            return action_back(f"back from {page}")

        # 咖啡厅「UI 隐藏模式」(2026-07-28 live 帧实锤):
        # BA 咖啡厅点**空白地面**会把顶栏/底栏/所有按钮整套隐藏(看房/截图功能)。
        # bot 的 recover 点击(叉叉/home 的坐标在 UI 消失后就是空白)一旦落空,
        # 就会**自己把 UI 点没**, 之后屏上一个导航 cls 都不剩  _enter 只能走
        # `cafe recover: 无 home/X/返回键  等待(绝不瞎按 ESC)` 干等到超时
        # (实录: 连续 12 tick 该 reason  "could not reach cafe")。
        # 当时屏上还有 **4 个 Emoticon_Action(4 个学生等着摸头)** + 未领的收益。
        # 判据(强, 不是瞎点): 有 Emoticon_Action = 人确实站在咖啡厅房间里,
        # 而**零导航 cls**(返回键/回大厅/咖啡厅收益/移动至N号点 全无) = UI 被隐藏。
        #  再点一次空白把 UI toggle 回来。落点取**左下角空地**(0.12,0.88):
        #   避开学生(会摸头/开对话)、避开右侧设施, 是 toggle 最安全的位置。
        _emos = self.find_all_cls(screen, UC.EMOTICON_ACTION, conf=0.30)
        _navs = self.find_cls(screen, [UC.BTN_BACK, UC.BTN_HOME, UC.CAFE_EARNINGS,
                                       UC.CAFE_MOVE_1F, UC.CAFE_MOVE_2F,
                                       UC.CAFE_INVITE_TICKET], conf=0.30)
        # 弹窗在场时**绝不**走「UI 隐藏」分支(2026-08-02 live 帧实锤):
        # 咖啡厅弹了「說明/訪問學生目錄」框(確認键灰、右上带 X), 而这一步的落点
        # (0.12,0.88) 是**写死坐标**, 在 UI 显示时正压在底栏「編輯模式」按钮一带
        # —— 误触会进家具编辑模式。零导航 cls 有两种成因: UI 真被 toggle 隐藏
        # 弹窗盖住/转场帧漏检。在这里必须优先按「关弹窗」处理, 而不是盲点空地。
        _popup_x = self.find_cls(screen, UC.BTN_CLOSE_X, conf=0.60)
        if _popup_x is not None:
            self.log("咖啡厅有弹窗(叉叉在场)  先关弹窗, 不走 UI-隐藏盲点分支")
            return action_click_box(_popup_x, "cafe: close blocking popup (X)")
        # 单帧不算数(2026-08-07 live 实锤): 上面那道弹窗闸是 2026-08-02 加的,
        # 代码在、常量对, 却**没拦住** —— 因为 skill 那一帧上连叉叉都没检出:
        # 那是张**进咖啡厅的转场帧**(学生已渲染  有 1 个 Emoticon_Action,
        # UI chrome 还没出来  零导航 cls), 恰好满足"UI 隐藏"的形状。
        # 我的探针帧(晚一拍)上 回大厅0.98/返回键0.97/移动至2号点0.96/邀请卷0.98
        # /弹窗叉叉0.96 **全在**, 屏上是「說明·訪問學生目錄」弹窗 + 完整底栏,
        # 而写死落点 (0.12,0.88) 正压在「編輯模式/禮物」上。
        #  原注释自己列了两种成因(真隐藏 弹窗/**转场帧漏检**), 但只挡了弹窗。
        # 真 UI-隐藏是**持续状态**, 转场帧是一瞬  要求连续 3 拍成立才动手。
        if _emos and _navs is None:
            self._ui_hidden_ticks += 1
            if self._ui_hidden_ticks < 3:
                self.log(f"疑似咖啡厅 UI 隐藏({self._ui_hidden_ticks}/3) — "
                         f"可能只是转场帧, 再看一拍(绝不盲点写死坐标)")
                return action_wait(350, "cafe: 疑似 UI 隐藏 — 等下一帧确认")
            self.log(f"咖啡厅 UI 被隐藏(连续 {self._ui_hidden_ticks} 拍: "
                     f"{len(_emos)} 个摸头标记 + 零导航 cls)  点空地把 UI toggle 回来")
            self._ui_hidden_ticks = 0
            return action_click(0.12, 0.88, "cafe: UI 被隐藏  点空地唤回")
        self._ui_hidden_ticks = 0

        # Unknown / transition screen.
        if self._phase_ticks > _ENTER_MAX:
            self.log("enter budget exhausted, giving up on cafe")
            return action_done("could not reach cafe")
        if len(screen.yolo_boxes or []) < 2:
            return action_wait(700, "no UI detected, likely loading")
        if self._enter_attempts > 8:
            return self.nav_home(screen, "cafe recover")
        return action_wait(400, "entering cafe")

    #  earnings

    def _earnings(self, screen: ScreenState) -> Dict[str, Any]:
        """Open CAFE_EARNINGS, claim via the active CLAIM cls.

        Pure YOLO + button-state: there is no popup-title cls, so the popup is
        detected by a CLAIM button in the centered claim band (or the
        CAFE_EARNINGS label leaking near the popup top). An ACTIVE (yellow)
        claim cls  claim it; otherwise (0% / already claimed) close & move on.
        DIGIT-DEFERRED: the old "skip 0% to save the cap" % read is gone with
        nav-OCR off — we drive purely off the claim button colour/state.
        """
        if self._earnings_done:
            self._begin_invite(floor_2=False)
            return action_wait(300, "earnings done  invite")

        # Post-claim popup chain fully closed (cafe page back)  done  invite.
        # audit 2026-06-15: claim 后是双层弹窗(领取奖励动画 + 收益明细), 旧码 close
        # 一次就 _earnings_done=True 前进  明细层还盖着  invite 卡 20t 到 stuck-recovery
        # 才关。改成: close 到 _is_cafe(咖啡页签名回来)才算完, 不是 close 一次就走。
        # 2026-07-28 第二次 live 才逮到的真根因: 这条**自称到达证据**的判据
        # `_earnings_claimed and _is_cafe(screen)` **根本不是到达证据** ——
        # 上面那段注释说的「the earnings POPUP covers the cafe page signature
        # (detect_screen_yolo!="Cafe" while it's open)」是 **2026-06-02 的观察**,
        # 模型变强后**已经不成立**: 实测收益弹窗开着时, 屏上照样检出
        # `咖啡厅邀请卷 0.97 / 回大厅按钮 0.97 / 返回键 0.98`  `_is_cafe` 为 True。
        # 于是 `claim earnings` 一被帧稳定门吞掉, 下一 tick 就从这里判"领完了"
        #  关窗走人, 而 `領取_黄 0.97 @cy0.733` 还好端端挂在屏上(帧证据)。
        #  补上真正的到达证据: **屏上已经没有可点的 領取(黄/蓝) 了**。
        #   领到手  按钮变灰或弹窗关  这个合取才成立;
        #   点被吞   領取_黄 仍在  落回下面 claim_active 分支重发。
        # 教训: "判据的名字叫到达证据"不等于"它真的是到达证据" —— 必须问
        #   **它会不会在动作没落地时也成立**。今天这已经是第二遍了。
        _still_claimable = self.find_cls(
            screen, [UC.CLAIM_REWARD_YELLOW, UC.CLAIM_YELLOW, UC.CLAIM_BLUE],
            conf=_CLS_CONF, region=(0.28, 0.50, 0.74, 0.90)) is not None
        if self._earnings_claimed and self._is_cafe(screen) and not _still_claimable:
            self._earnings_claimed = False
            self._earnings_done = True
            self._begin_invite(floor_2=False)
            return action_wait(300, "earnings popup chain closed (無可領)  invite")

        #  FIRST-ENTRY POPUP (user spec: 第一次进咖啡厅有"訪問學生目錄"说明弹窗,
        # 点叉叉关掉再看收益). live 2026-06-14: that popup covered the 咖啡厅收益
        # button (100% FULL!)  _earnings waited 18 ticks for CAFE_EARNINGS, never
        # found it, skipped the income. If a popup-X is up AND there's neither a
        # claim button (not the claim popup) NOR CAFE_EARNINGS visible yet, it's an
        # obstructing popup  dismiss it so the earnings button shows.
        _CB = (0.28, 0.50, 0.74, 0.90)
        _has_claim = self.find_cls(
            screen, [UC.CLAIM_REWARD_YELLOW, UC.CLAIM_YELLOW, UC.CLAIM_BLUE,
                     UC.CLAIM_REWARD_GREY, UC.CLAIM_GREY], conf=_CLS_CONF,
            region=_CB) is not None
        _has_earn = self.find_cls(screen, UC.CAFE_EARNINGS, conf=_CLS_CONF) is not None
        if not _has_claim and not _has_earn and not self._earnings_claimed:
            popup_x = self._close_x(screen, region=(0.55, 0.08, 0.82, 0.34))
            if popup_x is not None:
                self.log("first-entry 訪問學生目錄 popup 挡住收益  先点叉叉关掉 (spec)")
                return action_click_box(popup_x, "dismiss cafe first-entry popup (X)")

        #  next, regardless of _is_cafe: the earnings POPUP covers the cafe
        # page signature (detect_screen_yolo!="Cafe" while it's open), so the
        # claim MUST be checked before any _is_cafe gate — otherwise the bot
        # opens the popup then freezes "waiting for cafe UI" (live 2026-06-02:
        # 领取_黄 @0.5,0.732 sat unclaimed for 17 ticks). The active-claim cls in
        # the centered band = the popup is open.
        _CLAIM_BAND = (0.28, 0.50, 0.74, 0.90)
        claim_active = self.find_cls(
            screen, [UC.CLAIM_REWARD_YELLOW, UC.CLAIM_YELLOW, UC.CLAIM_BLUE],
            conf=_CLS_CONF, region=_CLAIM_BAND,
        )
        if claim_active is not None:
            # Claim it, but DON'T mark earnings done yet — next tick the button
            # greys out and the claim_grey path below CLOSES the popup. Marking
            # done here advances to invite while the popup is STILL OPEN, which
            # covers the cafe signature  _is_cafe False  invite freezes
            # "waiting for cafe UI" (live 2026-06-09: earnings popup stayed open,
            # invite dead-waited 14 ticks until the stuck-20 fallback).
            # 2026-07-28 live 实锤 —— **昨天那个正确的修复暴露了这个老 bug**:
            #   tick=13 wait  帧未稳定(转场/滚动) — 等稳定帧: claim earnings
            #   tick=14 wait  earnings done  invite         状态却已经"领完了"
            #   tick=15 click close leftover earnings before invite   关窗, 钱没领
            # 帧证据: 那一帧屏上 `領取_黄 0.97 @cy0.733` 原封不动。
            # 机制: 下面 `_force_settle=True`(昨天为修"锚在弹出动画帧上打空"加的,
            # 本身正确)会让这一发**被帧稳定门吞掉**; 而旧码在动作发出**之前**就
            # `_earnings_done = True`  下一 tick 本函数第一行 `if self._earnings_done`
            # 直接转 invite, **永远不会重试这一发**。
            # 加 _force_settle 之前这发必然发得出去, 所以 mutate-before-ack 藏着
            #   不显形 —— 一个正确的修复把另一个一直存在的 bug 顶出了水面。
            #  这里**绝不**置 `_earnings_done`, 它只能由下面那条到达证据分支
            #   (`_earnings_claimed and _is_cafe(screen)` = 弹窗链真的关干净了)来置;
            #   补 after-ack: 窗口内不重发, 窗口过了仍见 領取_黄 就判被吞并重发。
            if self._earnings_claimed and self.since("claim_earn") < _CLAIM_ACK_SEC:
                return action_wait(300, f"領取已发 — 等帧证据(按钮变灰/弹窗关) "
                                        f"(after-ack {self.since('claim_earn'):.1f}"
                                        f"/{_CLAIM_ACK_SEC:.1f}s)")
            if self._earnings_claimed:
                self.log("領取发出后 領取_黄 仍在 — 判为被吞(稳定门/丢tap), 重发一次")
            self.log(f"earnings popup, claiming (YOLO {claim_active.cls_name})")
            self._earnings_claimed = True
            self.mark("claim_earn")
            # `_force_settle`(2026-07-27 live 实锤): 收益弹窗**弹出动画**里
            # 「領取」在 cy≈0.825, 稳定后升到 cy≈0.733。reason 含 "claim" 会命中
            # _dedup_click 的关键词豁免  跳过稳定门  在动画帧上锚 0.825 拍下去
            #  **点在按钮下方空白, 收益一分没领**(memory cafe_flow_spec 的
            # "收益第1次没领" 就是它)。这个按钮位置会动, 必须等稳定帧。
            act = action_click_box(claim_active, "claim earnings")
            act["_force_settle"] = True
            return act
        claim_grey = self.find_cls(
            screen, [UC.CLAIM_REWARD_GREY, UC.CLAIM_GREY],
            conf=_CLS_CONF, region=_CLAIM_BAND,
        )
        if claim_grey is not None:
            # Popup open, nothing to collect (0% / already claimed)  close.
            self.log("earnings claim greyed  nothing to claim, closing popup")
            self._earnings_done = True
            close = self._close_x(screen)
            if close is not None:
                return action_click_box(close, "close earnings popup (nothing)")
            # X not detected THIS frame (one-frame miss) — ESC closes the popup
            # just as well. NEVER advance with the popup still covering the
            # students (live 2026-06-09 2nd run: advancing here left it open
            # 1F headpat saw nothing again). _earnings_done is already True so
            # the next tick moves on to invite with a clean screen.
            return action_back("close earnings popup (ESC, X miss)")

        # No claim button visible  we're not in the popup. Need cafe main.
        if not self._is_cafe(screen):
            # Just claimed but the popup still covers the cafe signature (the
            # claim_grey detection flickered out during the post-claim
            # transition): actively CLOSE it instead of dead-waiting  the
            # _EARNINGS_MAX skip would leave the popup OPEN  headpat sees no
            # students (live 2026-06-09: 咖啡1 没摸头).
            if self._earnings_claimed:
                # Keep closing any layered popup (reward anim + 收益 breakdown)
                # until the cafe page is back (the early _is_cafe check above
                # advances). NEVER set _earnings_done here — a single close can
                # leave the 2nd layer up  invite stuck. Timeout  give up (the
                # dry-frame/stuck-recovery catches a stray popup).
                if self._phase_ticks > _EARNINGS_MAX:
                    self.log("earnings close timeout  invite (recovery兜底)")
                    self._earnings_claimed = False
                    self._earnings_done = True
                    return action_wait(300, "earnings close timeout  invite")
                close = self._close_x(screen)
                if close is not None:
                    return action_click_box(close, "close earnings popup (post-claim)")
                return action_back("close earnings popup (ESC, post-claim X miss)")
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

        # Cafe main screen: SIGNAL-DRIVEN earnings (user rule 2026-06-09:
        # "收益明明是0为什么还要点一次" — don't walk the flow blindly). The
        # button itself shows the live percentage; digit-OCR it and only open
        # the popup when there's actually something (>0%). Unreadable  open
        # popup as before (safe fallback).
        earn = self.find_cls(screen, UC.CAFE_EARNINGS, conf=_CLS_CONF)
        if earn is not None:
            pct = self._read_earnings_pct(screen, earn)
            if pct is None:
                # Unreadable THIS frame ≠ "open the popup" — retry a few frames
                # first (live 2026-06-09 round-2: a single None re-opened the
                # popup on a 0.0% cafe). Only after retries fail do we fall
                # open (free-harvest skill must never silently skip real money).
                self._pct_retries = getattr(self, "_pct_retries", 0) + 1
                if self._pct_retries <= 3:
                    return action_wait(300, f"earnings % unreadable, retry ({self._pct_retries}/3)")
                self.log("earnings % unreadable ×3  opening popup to check (fail-open)")
            elif pct <= 0.0:
                self.log("咖啡厅收益 0.0% (digit-OCR)  nothing to claim, skip popup")
                self._earnings_done = True
                self._begin_invite(floor_2=False)
                return action_wait(250, "earnings 0%  skip popup")
            self.log(f"opening earnings popup (YOLO 咖啡厅收益, pct={pct})")
            return action_click_box(earn, "open earnings popup")

        # CAFE_EARNINGS not seen — give the cls a few ticks, then skip.
        if self._phase_ticks > _EARNINGS_MAX:
            self.log("CAFE_EARNINGS cls never found — skipping earnings")
            self._earnings_done = True
            self._begin_invite(floor_2=False)
            return action_wait(300, "no earnings cls  invite")
        return action_wait(350, "waiting for CAFE_EARNINGS cls")

    #  invite

    def _begin_invite(self, *, floor_2: bool) -> None:
        """Enter the invite sub_state for the given floor."""
        self._on_2f = floor_2
        # cafe_skip_invite (app_config profile): invite ticket on CD / user wants
        # headpat-only  mark the floor's invite already done; _invite()'s first
        # tick routes straight to headpat. No ticket click, no list, no dead-wait.
        self._invite_floor_done = bool(getattr(self, "_skip_invite", False))
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

        # Done with this floor's invite  advance.
        if self._invite_floor_done:
            recover = self._recover_invite_overlay(screen)
            if recover:
                return recover
            if self._on_2f:
                self._goto("headpat2")
            else:
                self._goto("headpat")
            return action_wait(300, f"invite done  headpat (floor {floor})")

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
                self._commit_pending_invite()   # 确认动作发出 = 邀请落账
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
            # invite likely didn't register  go re-pick a row.
            if self._invite_list_open(screen):
                self.log("invite dialog absent, list still open  re-pick row")
                self._invite_stage = 1
                return action_wait(300, "re-pick invite row")
            if self._is_cafe(screen):
                self.log("invite dialog gone, back on cafe  invite done")
                self._commit_pending_invite()   # 经此路成功的邀请也落账(防跨楼重复)
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
                    self.log("invite list missing  back to stage 0 (re-open ticket)")
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
                return action_swipe(0.5, 0.70, 0.5, 0.30, 800, "scroll invite list (hunt)")

            # No configured target on screen.
            if bottom or budget_out or not self._invite_targets:
                # Fallback: invite a row, but STRICTLY skip anyone already
                # invited this run (1F/2F 绝不能重复邀请同一角色 — 用户要求
                # "一个角色只能出现在一个 cafe")。遍历所有可见 invite 行 (按 cy),
                # 选第一个 avatar 名可识别且未邀请的。(旧逻辑只在 len>1 时换第二行,
                # 2F invite list 常只渲染 1 个按钮  len>1 False  又邀请季战斗。)
                why = ("no targets configured" if not self._invite_targets
                       else ("list bottom" if bottom else "scroll budget"))
                ranked = sorted(invite_btns, key=lambda b: b.cy)
                for cand in ranked:
                    cand_name = self._row_name(screen, cand.cy) or "?"
                    if cand_name != "?" and cand_name not in self._invited:
                        return self._fire_invite(cand, cand_name, floor, f"first-row/{why}")
                # 所有可见行已邀请/识别不出  scroll 找未邀请的 (2F 防重复关键);
                # 预算内继续翻, 实在耗尽才邀请 top (接受可能重复, 最后兜底)。
                if not (bottom or budget_out) and self._invite_scrolls < _INVITE_MAX_SCROLLS:
                    self._invite_scrolls += 1
                    self._invite_settle = _INVITE_SWIPE_SETTLE
                    self.log("first-row all invited/unknown  scroll for fresh")
                    return action_swipe(0.5, 0.70, 0.5, 0.30, 800, "scroll for uninvited row")
                top = ranked[0]
                top_name = self._row_name(screen, top.cy) or "?"
                return self._fire_invite(top, top_name, floor, "first-row/exhausted")

            self._invite_scrolls += 1
            self._invite_settle = _INVITE_SWIPE_SETTLE
            self.log(f"target not visible, scrolling ({self._invite_scrolls}/{_INVITE_MAX_SCROLLS})")
            return action_swipe(0.5, 0.70, 0.5, 0.30, 800, "scroll invite list")

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
        # fire 只记 pending, 确认框真被点掉才落账(2026-07-21 tick483 实锤:
        # 邀请点击被 pipeline 稳定门吞掉, 凯伊 已进 _invited+落盘  榜单永久
        # 跳过她  空滚 8 次到预算耗尽, 1F 邀请券整张未用+盘上状态毒化到04:00)。
        self._invite_pending_name = name if (name and name != "?") else None
        self._invite_retry_btn = (btn.cx, btn.cy)
        self._invite_stage = 2
        self.log(f"inviting {tag} '{name}' at ({btn.cx:.2f},{btn.cy:.2f}) floor={floor}")
        return action_click_box(btn, f"invite {tag} {name}")

    def _commit_pending_invite(self) -> None:
        """确认动作真发出时才把 pending 邀请落账+落盘(phantom-action 防线)。"""
        _n = getattr(self, "_invite_pending_name", None)
        if _n:
            self._invited.add(_n)
            self._save_invited()
        self._invite_pending_name = None

    def _recover_invite_overlay(self, screen: ScreenState) -> Optional[Dict[str, Any]]:
        """Dismiss a lingering invite confirm / list before leaving invite."""
        confirm = self._confirm_btn(screen)
        if confirm is not None:
            self.log("invite confirm still up, clicking confirm")
            self._commit_pending_invite()   # 2F 实走路径(2026-07-21): 这里落账
            return action_click_box(confirm, "confirm lingering invite")
        if self._invite_list_open(screen):
            close = self._close_x(screen, region=(0.56, 0.04, 0.90, 0.24))
            if close is not None:
                self.log("invite list still open, closing")
                return action_click_box(close, "close lingering invite list")
            return action_back("close lingering invite list (ESC)")
        return None

    #  headpat

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
        self._pats_total += 1
        self._empty_frames = 0
        self._pat_settle = 1  # one tick for the heart animation
        # 爱心动画比 pan 短, 但同样吃 squash —— 给半个 pan 的墙钟。
        self._pat_settle_sec = _PAT_SETTLE_SEC / 2.0
        self.mark("pat_settle")
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
                    self.log("lobby during 2F headpat  done")
                    return action_done("cafe complete (lobby)")
                # kicked to lobby mid-1F (e.g. bond level-up)  re-enter and
                # skip straight to switch (1F headpat counts as done enough).
                self.log("lobby during 1F headpat  re-enter, skip to 2F")
                self._goto("enter")
                self._headpat_to_switch_on_reentry = True
                return action_wait(300, "re-enter cafe from lobby")
            if self._phase_ticks > 12:
                if is_2f:
                    self._goto("exit")
                    return action_wait(300, "lost cafe on 2F  exit")
                self._goto("switch")
                return action_wait(300, "lost cafe on 1F  switch")
            return action_wait(300, "waiting for cafe (headpat)")

        # If we re-entered after a lobby kick on 1F, jump to switch.
        if getattr(self, "_headpat_to_switch_on_reentry", False) and not is_2f:
            self._headpat_to_switch_on_reentry = False
            self._goto("switch")
            return action_wait(300, "resume  switch to 2F")

        # 切回 1F 的到达对账(2026-07-28): _floor_back_done 只认帧证据 —
        # 1F 才有 移動至2號店, 2F 才有 移動至1號店。点击被吞(仍见 2F 按钮)
        # 超过 ACK 窗就有界重发, 绝不一发定生死。
        if getattr(self, "_floor_back_pending", False) and not is_2f:
            _btn_region = (0.0, 0.03, 0.30, 0.22)
            _still_2f = self.find_cls(screen, UC.CAFE_MOVE_1F, conf=_CLS_CONF,
                                      region=_btn_region)
            _proof_1f = self.find_cls(screen, UC.CAFE_MOVE_2F, conf=_CLS_CONF,
                                      region=_btn_region)
            if _proof_1f is not None and _still_2f is None:
                self._floor_back_pending = False
                self._floor_back_done = True
                self.log("帧证据: 移動至2號店 出现 = 已回 1F  重扫落账")
            elif _still_2f is not None and self.since("floor_back") > _SWITCH_2F_ACK_SEC:
                self._floor_back_retries = getattr(self, "_floor_back_retries", 0) + 1
                if self._floor_back_retries > 3:
                    self.log("切回1F 3 次没落地  放弃重扫  exit")
                    self._floor_back_pending = False
                    self._floor_back_done = True   # 防死循环: 当作处理过
                    self._goto("exit")
                    return action_wait(300, "floor-back failed  exit")
                self.mark("floor_back")
                self.log(f"切回1F未落地(仍见 移動至1號店)  重发 #{self._floor_back_retries}")
                return action_click_box(_still_2f, "1F dot on switch btn  back to 1F")
            elif (_proof_1f is None and _still_2f is None
                    and self.since("floor_back") < _SWITCH_2F_ACK_SEC * 2):
                return action_wait(300, "切回1F转场中 — 等楼层按钮出现")

        # Safety helmet.
        if self._pat_count >= _MAX_HEADPATS_PER_FLOOR:
            return self._finish_headpat(screen, is_2f, "max headpats")

        # Post-pat / post-pan settle — let animation finish before scanning.
        # 合取: 帧数没走完 **或** 墙钟没到, 都继续等(见 _init_state 的换算说明)。
        if self._pat_settle > 0 or self.since("pat_settle") < self._pat_settle_sec:
            if self._pat_settle > 0:
                self._pat_settle -= 1
            _psw = self.since("pat_settle")
            return action_wait(450, f"headpat settle (帧{self._pat_settle} / "
                                    f"{_psw:.1f}/{self._pat_settle_sec:.1f}s)")

        # PRIORITY: pat any visible marker immediately (markers fade fast).
        mark = self._emoticon_mark(screen)
        if mark is not None:
            return self._do_headpat(mark)

        # No marker this frame.
        self._empty_frames += 1
        if self._empty_frames == 1:
            self.mark("dry_scan")
        # 合取: 帧数不够 **或** 墙钟不够, 都不许判 dry(见 _init_state 换算说明)。
        # 单看帧数会被 squash 削成 ~1.4s, 而气泡最晚 600ms 才渲染 + v10 折进
        # cls451 后 conf 偏弱 —— 提前判 dry 就 pan 走 = 漏摸(live 2026-06-09)。
        if (self._empty_frames < _HEADPAT_DRY_FRAMES
                or self.since("dry_scan") < _HEADPAT_DRY_SEC):
            return action_wait(280, f"scanning headpat (empty={self._empty_frames}, pan={self._pan_dir})")
        # dry 前最后一扫: 清 dedup 残留(2026-07-21 tick595 实锤: 被吞的 pat
        # 毒化 _recent_pats, 4 泡泡在屏仍判 empty×6)。真拍过的泡泡已消失,
        # 清账无害; 假拍的(点击被吞)泡泡还在  捡回。
        if self._recent_pats:
            self._recent_pats = []
            mark = self._emoticon_mark(screen)
            if mark is not None:
                return self._do_headpat(mark)

        # Current view is dry  pan to the next region.
        if self._pan_count >= _MAX_PANS_PER_FLOOR:
            return self._finish_headpat(screen, is_2f, "pan helmet")

        floor_tag = "2F" if is_2f else "1F"
        if self._pan_dir == 0:
            # sweep LEFT full-span (0.800.10) to reveal the right-side overflow.
            # Deep-dive r2 H4: the old 0.750.25 / 0.250.80 pair was asymmetric
            # and left a mid-floor blind strip — a student standing there was
            # never on screen during either dwell (live 2026-06-09 1F漏摸).
            self._pan_dir = 1
            self._pan_count += 1
            self._empty_frames = 0
            self._pat_settle = 3   # 23 (user 2026-06-13 给足时间): late bubbles after a pan
            self._pat_settle_sec = _PAT_SETTLE_SEC   # 墙钟合取(450ms 被 squash)
            self.mark("pat_settle")
            self.log(f"{floor_tag} sweep LEFT full-span (reveal right overflow)")
            return action_swipe(0.80, 0.45, 0.10, 0.45, 600, f"pan left ({floor_tag})")
        if self._pan_dir == 1:
            # sweep RIGHT full-span (0.100.90) to reveal the left overflow.
            self._pan_dir = 2
            self._pan_count += 1
            self._empty_frames = 0
            self._pat_settle = 3   # 23 (user 2026-06-13 给足时间): late bubbles after a pan
            self._pat_settle_sec = _PAT_SETTLE_SEC   # 墙钟合取(450ms 被 squash)
            self.mark("pat_settle")
            self.log(f"{floor_tag} sweep RIGHT full-span (reveal left overflow)")
            return action_swipe(0.10, 0.45, 0.90, 0.45, 600, f"pan right ({floor_tag})")

        # Both directions swept and dry  floor done.
        return self._finish_headpat(screen, is_2f, "all views dry")

    def _finish_headpat(self, screen: ScreenState, is_2f: bool, reason: str) -> Dict[str, Any]:
        if is_2f:
            # IN-CAFE SIGNAL (user 2026-06-09: "店内黄点没消化"): the floor-
            # switch button carries a 黄点 when the OTHER floor still has work
            # (screenshot: 移動至1號店 with a yellow dot top-right). Consume it:
            # switch back ONCE and re-sweep 1F instead of blindly exiting.
            if not getattr(self, "_floor_back_done", False):
                mv1 = self.find_cls(screen, UC.CAFE_MOVE_1F, conf=_CLS_CONF,
                                    region=(0.0, 0.03, 0.30, 0.22))
                if mv1 is not None and self.dot_in_region(
                        screen, (mv1.x1 - 0.01, mv1.y1 - 0.05, mv1.x2 + 0.04, mv1.y2 + 0.01)):
                    # mutate-before-ack(2026-07-28, 与 _switch_floor :1108 同形):
                    # _floor_back_done 不在这里落账 — 点击被吞时人还在 2F, 却按
                    # 1F 分支把干净的 2F 重扫一遍, 然后 "1F re-sweep done" 把 1F
                    # 真正的活永久丢掉。落账移到 _headpat 的帧证据分支
                    # (移動至2號店 出现 = 真回到 1F)。
                    self._floor_back_pending = True
                    self._floor_back_retries = 0
                    self.mark("floor_back")
                    self._on_2f = False
                    self._reset_headpat_for_floor()
                    self._goto("headpat")
                    self.log(" 移動至1號店 黄点在  1F 还有活, 切回 1F 重扫")
                    return action_click_box(mv1, "1F dot on switch btn  back to 1F")
            self.log(f"2F headpat done ({reason}, {self._pat_count} pats)  exit")
            self._goto("exit")
            return action_wait(300, "2F headpat done  exit")
        if getattr(self, "_floor_back_done", False):
            # This 1F pass WAS the dot-triggered re-sweep  done, don't loop 2F.
            self.log(f"1F re-sweep done ({reason}, {self._pat_count} pats)  exit")
            self._goto("exit")
            return action_wait(300, "1F re-sweep done  exit")
        # IN-CAFE SIGNAL both ways (user 2026-06-09): NO 黄点 on 移動至2號店
        # 2F has nothing to do — skip the whole trip. 2-frame confirmation so a
        # single dot-flicker frame can't wrongly cancel the 2F visit; button
        # not detected  can't judge  go as before.
        mv2 = self.find_cls(screen, UC.CAFE_MOVE_2F, conf=_CLS_CONF)
        if mv2 is not None and not self.dot_in_region(
                screen, (mv2.x1 - 0.01, mv2.y1 - 0.05, mv2.x2 + 0.04, mv2.y2 + 0.01)):
            self._no2f_frames = getattr(self, "_no2f_frames", 0) + 1
            if self._no2f_frames < 2:
                return action_wait(300, "confirm no-2F-dot (1/2)")
            self.log("移動至2號店 无点 ×2帧  2F 没活, 跳过直接退出")
            self._goto("exit")
            return action_wait(300, "no 2F dot  skip 2F, exit")
        self._no2f_frames = 0
        self.log(f"1F headpat done ({reason}, {self._pat_count} pats)  switch")
        self._goto("switch")
        return action_wait(300, "1F headpat done  switch")

    #  switch floor

    def _switch_floor(self, screen: ScreenState) -> Dict[str, Any]:
        """1F  2F. On 2F the button reads CAFE_MOVE_1F (takes us back to 1F)."""
        # Already on 2F? (CAFE_MOVE_1F button visible top-left)
        on_2f_btn = self.find_cls(
            screen, UC.CAFE_MOVE_1F, conf=_CLS_CONF, region=(0.0, 0.03, 0.30, 0.22)
        )
        if on_2f_btn is not None:
            # If we already ran a 2F cycle, exit (don't loop a 3rd invite).
            if self._on_2f:
                self.log("already on 2F and cycle done  exit")
                self._goto("exit")
                return action_wait(300, "2F cycle complete  exit")
            self.log("already on 2F  start 2F invite")
            self._reset_headpat_for_floor()
            self._begin_invite(floor_2=True)
            return action_wait(300, "already on 2F  invite")

        switch = self.find_cls(screen, UC.CAFE_MOVE_2F, conf=_CLS_CONF)
        if switch is not None:
            # mutate-before-ack(2026-07-28 全仓审计, task#23 家族):
            # 旧码在 `action_click_box` **之前**就 `_begin_invite(floor_2=True)`
            # (里面 `_on_2f = True` + `_goto("invite")`), 状态挂在**代码路径**上。
            # 而 reason "switch to cafe 2F" 不在 pipeline 的关键词豁免表里  这一发
            # 完全可能被稳定门吞掉或丢 tap, 而状态机已经离开 switch 态、再也不会
            # 点第二次, 全程没有任何帧证据确认换层成功。后果: 人还在 1F 却按 2F
            # 的名单去邀请/摸头, 2F 整层白丢。
            #  这里**只发动作**; 真正的状态切换交给上面那条
            #   `on_2f_btn is not None`(CAFE_MOVE_1F 出现 = 到达 2F 的帧证据)分支。
            if self._switch_issued and self.since("switch_2f") < _SWITCH_2F_ACK_SEC:
                return action_wait(300, f"移动至2号点 已点 — 等 2F 到达证据 "
                                        f"(after-ack {self.since('switch_2f'):.1f}"
                                        f"/{_SWITCH_2F_ACK_SEC:.1f}s)")
            self.log("switching to cafe 2F (YOLO 移动至2号点) — 状态等到达证据再改")
            self._switch_issued = True
            self.mark("switch_2f")
            return action_click_box(switch, "switch to cafe 2F")

        if self._phase_ticks > _SWITCH_MAX:
            self.log("switch budget exhausted, exiting cafe")
            self._goto("exit")
            return action_wait(300, "switch timeout  exit")
        return action_wait(450, "waiting for 移动至2号点 cls")

    def _reset_headpat_for_floor(self) -> None:
        self._pat_count = 0
        self._empty_frames = 0
        self._pan_count = 0
        self._pan_dir = 0
        self._pat_settle = 0
        self._recent_pats = []

    #  exit

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if self.detect_screen_yolo(screen) == "Lobby":
            # BADGE-VERIFIED completeness (user rule 2026-06-09: 少摸一个角色 —
            # trust the lobby dot, not the dry-scan): if the cafe entry STILL
            # carries a red/yellow dot, work remains (a bubble the pan-sweep
            # missed / earnings that re-accrued)  re-enter ONCE and re-sweep.
            # Region = entry bbox expanded up-right (dots sit top-right of the
            # icon; +0.02 right stays well clear of the 课程表 neighbour).
            if not getattr(self, "_verify_reentered", False):
                entry = self.find_cls(screen, UC.NAV_CAFE, conf=0.40)
                if entry is not None and self.dot_in_region(
                        screen,
                        (entry.x1 - 0.005, entry.y1 - 0.05, entry.x2 + 0.02, entry.y2)):
                    self.log(" lobby cafe entry STILL has dot  missed work, re-entering once")
                    keep_inv, keep_skip = self._invite_targets, self._skip_invite
                    self._init_state()
                    self._invite_targets, self._skip_invite = keep_inv, keep_skip
                    self._verify_reentered = True
                    self._goto("enter")
                    return action_wait(300, "cafe dot persists  re-enter sweep")
                # DWELL: badges fade in AFTER the loading transition — a single
                # clean frame proves nothing (live 2026-06-09: dot was offline-
                # detectable at 0.58 on the very frame the one-shot check ran
                # empty on). Require 3 consecutive dot-free lobby ticks.
                self._exit_lobby_ticks += 1
                if self._exit_lobby_ticks == 1:
                    self.mark("lobby_dwell")
                # 合取(2026-07-28): 帧数抗单帧抖动, 墙钟抗"徽标还没渐入完"。
                # 这道复验是**漏摸/收益重累的唯一兜底** —— 它一空转, cafe 就带着
                # 没干完的活报 done, 而 exit_report 还会给 CLEAN(漏一个人从计数上
                # 完全看不出来)。350ms 被 squash 后原本只剩 ~0.4s。
                if (self._exit_lobby_ticks < 3
                        or self.since("lobby_dwell") < _LOBBY_DWELL_SEC):
                    return action_wait(
                        350, f"lobby badge dwell (帧{self._exit_lobby_ticks}/3, "
                             f"{self.since('lobby_dwell'):.1f}/{_LOBBY_DWELL_SEC:.1f}s)")
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
        return self.nav_home(screen, "cafe exit")
