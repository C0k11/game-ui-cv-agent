# -*- coding: utf-8 -*-
"""EventQuestSkill — 活动 AP 规划器 (2026-07-08, 机制全 live 实锤后落码).

用户 AP 分配铁律 (2026-07-08):
  ①先把「活動點數」关(尾关)刷满里程碑(每200点一档全大件, 10000pt 档有青輝石)
  ②商店货币关(尾部前 3 关)先各用加成队真打一次解锁 Bonus
  ③点数满 15000 后 AP 才灌货币关
  ④活动 > 双倍三倍 > 正常关 (编排上本 skill 排在 special_sweep 之前)

核心机制 (live 实验 + kamigame 印证, 2026-07-08):
  - 扫荡 Bonus = 该关「最高加成通关记录」。编队屏挂加成队≠生效; 必须真出击
    通关一次, 之后扫荡永久按该记录结算 (ドロップ数の最大値は引き継がれる)。
  - 判据 cls = 「活动关卡产出额外加成」(popup 内, 解锁前 0 检出 / 解锁后 0.95+)。
  - 自动编队按「当前关」优化且覆盖共享 2 部队 (Q10 优化后同队对 Q11 只 55%)
    → 每个关解锁前都要重跑 快速編輯→自動→確認。
  - 405 banner 轮播歧义: hub 左上 banner 槽多活动轮播, 点进去必须验证落点;
    错活动 → (顺手领免费成就) → back → 重试, cap 3 次。
  - 落在 Challenge tab → 固定位切 Quest tab (0.635,0.151, 2026-07-07 实测)。

Money (⛔铁律): 全程只走 入场/编队/出击/MAX/扫荡开始/确认/叉/back — 无购买路径。
  - 扫荡确认框 gate: body(y>0.12) 检到 青辉石 → 取消 + abort (fail-closed 黑名单
    backstop; 正向源头闸 = MAX_可点击 才扫, MAX_灰=AP不足 天然不触发買AP框)。
  - 商店购买是独立 skill (event_shop), 本 skill 绝不进商店。

Flow:
  enter    lobby → 任务大厅入口; hub → 405(距离结束还剩); 无405 → done(no event)
  verify   落点校验: quest列表特征(入场键+关卡得星) / Challenge tab→切Quest /
           错活动(如情人節約定成就页: 全部领取_黄)→顺手领→back 重试
  survey   列表滑到底, 自底向上开尾部 N 关 popup, 记录 Bonus 解锁状态
  unlock   未解锁关: 任務開始→编队(2部队→快速編輯→自動→確認→%读数)→出击
           → battle(AUTO gate)→结算→回列表
  points   尾关(点数关) MAX 扫循环, 直到 活動點數≥目标 或 MAX灰(AP不足)
  currency 点数满后: 货币关(倒数第2起轮流) MAX 扫到 AP 尽
  tasks    底栏 活动任务 红点门控 → 全部领取/领取 → back
  close    back → hub, done on hub
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from brain.skills.base import (
    BaseSkill, ScreenState, YoloBox,
    action_click, action_click_box, action_wait, action_back, action_done,
    action_swipe,
)
from brain.skills import ui_classes as UC

_CLS_CONF = 0.30
_WEAK_CONF = 0.20          # 活动皮肤弱类 (活动商店/活动任务/活动quest_已选择) 地板
_ENTER_MAX = 24
_VERIFY_RETRY_MAX = 6      # 405 轮播重试上限 (递增等待破共振)
_PHASE_MAX = 18
_BATTLE_MAX = 46           # battle poll ticks (pipeline tick ~5s → ~230s)
_SWEEP_ROUNDS_MAX = 30     # 点数期一次 MAX 就把 AP 扫光, 这是保险帽
_TAIL_QUESTS = 4           # 尾部加成关数量 (有时3有时4, 用户 2026-07-08)

# 固定位 (live 实测, 帧目检):
_POS_QUEST_TAB = (0.635, 0.151)     # 活动页 Quest tab (cls 活动quest_已选择 仅0.24-0.59)
_POS_TOUCH_CONTINUE = (0.5, 0.90)   # 结算 TOUCH-continue
# digitOCR regions:
_R_POINTS = (0.55, 0.93, 0.82, 0.985)   # quest列表底部 活動點數 "2676/15000"
_R_SQUAD_PCT = (0.005, 0.92, 0.085, 0.99)  # 编队屏左下 活動編輯效果 第一格 "100%"

_POINTS_TARGET_DEFAULT = 15000


def _read_digits(frame, region) -> Optional[str]:
    try:
        from brain.pipeline import run_digit_ocr
        return run_digit_ocr(frame, region)
    except Exception:
        return None


class EventQuestSkill(BaseSkill):
    """活动 quest AP 规划器: Bonus 解锁 → 点数优先 → 货币扫荡 → 领奖。"""

    # no-UI 逃生 (2026-07-09): 405 轮播可能落进模型盲区页(特殊作戰運輸船主页
    # v13 全零检出) — pipeline no-UI 时按 back 回已知页, 本 skill 恢复重试。
    no_ui_escape = "back"

    def __init__(self, points_target: int = _POINTS_TARGET_DEFAULT,
                 tail_quests: int = _TAIL_QUESTS):
        super().__init__("EventQuest")
        self.max_ticks = 400          # 长跑 skill: battle + 多轮扫荡
        self._points_target = points_target
        self._tail_quests = tail_quests
        self._init_state()

    def _init_state(self) -> None:
        self.sub_state = "enter"
        self._phase_ticks = 0
        self._verify_retries = 0
        # survey 结果: [(cy_of_入场键, bonus_unlocked)] 自上而下
        self._quests: List[Tuple[float, bool]] = []
        # cy 记账 (2026-07-10 同关反复开合 bug 修): 点过的 cy±0.03 不再点。
        # 旧 _survey_idx 索引制病灶: 关 popup 后过渡帧入场键检出不全 →
        # keys[-4:] tail 切片错位 → 同一关反复开合 + Bonus 状态记错关。
        self._surveyed_cys: List[float] = []
        self._popup_wait = 0
        self._survey_swiped = False
        self._unlock_idx = 0
        self._battle_ticks = 0
        self._sweep_rounds = 0
        self._points_done = False
        self._currency_idx = 0        # 货币关轮转指针 (倒数第2起)
        self._swept = 0
        self._popup_open = False
        self._formation_step = ""     # unlock 子步: '', 'edit', 'auto', 'confirm', 'sortie'
        self._tasks_done = False

    def reset(self) -> None:
        super().reset()
        self._init_state()

    # ── helpers ────────────────────────────────────────────────────

    def _set(self, state: str) -> None:
        if state != self.sub_state:
            self.sub_state = state
            self._phase_ticks = 0

    def _on_quest_list(self, screen: ScreenState) -> bool:
        """活动 quest 列表页特征: ≥2 入场键 + (关卡得星 或 活动商店/活动任务底栏)."""
        enters = [b for b in (screen.yolo_boxes or [])
                  if b.cls_name == UC.STAGE_ENTER and b.confidence >= _CLS_CONF]
        if len(enters) < 2:
            return False
        # ⛔负特征 (2026-07-09 live 实锤): 特殊作戰運輸船列表也有 入场键+得星_0,
        # 曾骗过 verify → survey 在错活动里跑。「普通关卡选中」= 主线/特殊作戰
        # tab 指示器, 活动页绝不出现。
        if self.find_cls(screen, "普通关卡选中", conf=0.5) is not None:
            return False
        # 硬条件: 活动专属底栏 (v13 下 活动商店0.98/活动任务0.96 稳; 弱检出
        # 地板 0.20 兜底)。得星不再单独放行 — 它对特殊作戰无区分度。
        has_bottom = self.find_cls(
            screen, [UC.EVENT_SHOP, UC.EVENT_TASK], conf=_WEAK_CONF) is not None
        return has_bottom

    def _on_popup(self, screen: ScreenState) -> bool:
        """任務資訊 popup 特征: 扫荡开始 + 任务开始 同屏."""
        return (self.find_cls(screen, UC.SWEEP_START, conf=_CLS_CONF) is not None
                and self.find_cls(screen, UC.TASK_START, conf=_CLS_CONF) is not None)

    def _enter_keys(self, screen: ScreenState) -> List[YoloBox]:
        """列表页所有入场键, 按 cy 升序 (上→下 = 低关→高关)."""
        keys = [b for b in (screen.yolo_boxes or [])
                if b.cls_name == UC.STAGE_ENTER and b.confidence >= 0.5]
        keys.sort(key=lambda b: (b.y1 + b.y2) / 2)
        return keys

    def _read_points(self, screen: ScreenState) -> Optional[Tuple[int, int]]:
        """活動點數 'n/m' from quest-list bottom bar. None = unread."""
        raw = _read_digits(screen.frame, _R_POINTS)
        if raw and "/" in raw:
            a, _, b = raw.partition("/")
            a = "".join(ch for ch in a if ch.isdigit())
            b = "".join(ch for ch in b if ch.isdigit())
            if a.isdigit() and b.isdigit() and int(b) >= 1000:
                return int(a), int(b)
        return None

    # ── k-偏移原子连发 (2026-07-09 终案, 活动无关): swipe×k 翻页后 0.5s 内
    # tap — 一条 adb shell 原子完成(swipe 后轮播暂停, 静止期内必落)。k=0/1/2
    # 覆盖三项轮播, verify 落点错→k+1 重试, 不认 banner 内容(下期活动零改动)。
    _ADB = r"C:\Program Files\Netease\MuMu\nx_device\12.0\shell\adb.exe"
    _DEV = "127.0.0.1:7555"

    def _atomic_banner_tap(self, k: int) -> None:
        import subprocess
        parts = []
        for _ in range(min(k, 4)):
            parts.append("input swipe 420 400 130 400 300")
            parts.append("sleep 0.6")
        parts.append("input tap 264 321")
        try:
            subprocess.run([self._ADB, "-s", self._DEV, "shell",
                            " && ".join(parts)],
                           capture_output=True, timeout=30)
        except Exception as e:
            self.log(f"atomic banner tap failed: {e}")

    # ── banner 指纹 (2026-07-09): crop 直方图记"点过且 verify 失败"的轮播项 ──
    _BANNER_CROP = (0.02, 0.10, 0.16, 0.29)   # hub 左上 banner 本体区

    def _banner_hist(self, screen: ScreenState):
        try:
            import cv2
            fr = screen.frame
            if fr is None:
                return None
            h, w = fr.shape[:2]
            x1, y1 = int(self._BANNER_CROP[0]*w), int(self._BANNER_CROP[1]*h)
            x2, y2 = int(self._BANNER_CROP[2]*w), int(self._BANNER_CROP[3]*h)
            crop = fr[y1:y2, x1:x2]
            hist = cv2.calcHist([crop], [0, 1, 2], None, [8, 8, 8],
                                [0, 256, 0, 256, 0, 256])
            cv2.normalize(hist, hist)
            return hist
        except Exception:
            return None

    def _banner_snapshot(self, screen: ScreenState) -> None:
        """点击前存当前项指纹 (verify 失败时转入黑名单)."""
        self._banner_pending = self._banner_hist(screen)

    def _banner_mark_bad(self) -> None:
        h = getattr(self, "_banner_pending", None)
        if h is not None:
            if not hasattr(self, "_banner_bad"):
                self._banner_bad = []
            self._banner_bad.append(h)
            self._banner_pending = None

    def _banner_seen_bad(self, screen: ScreenState) -> bool:
        import cv2
        bad = getattr(self, "_banner_bad", None)
        if not bad:
            return False
        cur = self._banner_hist(screen)
        if cur is None:
            return False
        return any(cv2.compareHist(cur, b, cv2.HISTCMP_CORREL) > 0.90
                   for b in bad)

    def _pyroxene_in_body(self, screen: ScreenState) -> bool:
        """money backstop: 对话框 body(y>0.12) 出现青辉石 = 绝不是纯AP扫荡框."""
        for b in (screen.yolo_boxes or []):
            if b.cls_name == UC.TOPBAR_PYROXENE and b.confidence >= 0.20:
                if (b.y1 + b.y2) / 2 > 0.12:
                    return True
        return False

    # ── tick ───────────────────────────────────────────────────────

    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1
        self._phase_ticks += 1
        handler = getattr(self, f"_{self.sub_state}", None)
        if handler is None:
            return action_done(f"event_quest unknown state {self.sub_state}")
        return handler(screen)

    # ── enter / verify ─────────────────────────────────────────────

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        if self._phase_ticks > _ENTER_MAX:
            return action_done("event_quest enter timeout")
        # already on the quest list? (re-entry / retry path)
        if self._on_quest_list(screen):
            self._set("survey")
            return action_wait(300, "on event quest list")
        banner = self.find_cls(
            screen, [UC.EVENT_END_LEFT, UC.EVENT_REWARD_END], conf=0.4)
        if banner is not None:
            # k-偏移原子连发(终案, 活动无关): 见 _atomic_banner_tap 注释。
            # verify 失败次数=k, k=0/1/2 覆盖三项轮播, 落错 verify 兜底 k+1。
            k = self._verify_retries
            self.log(f"banner combo: swipe×{k} && tap (atomic)")
            self._atomic_banner_tap(k)
            self._set("verify")
            return action_wait(3500, f"banner combo k={k} → verify landing")
        hub = self.find_cls(screen, UC.NAV_TASKS, conf=_CLS_CONF)
        if hub is not None:
            return action_click_box(hub, "lobby → task hub")
        if self._phase_ticks % 4 == 0:
            return self.nav_home(screen, "event_quest enter")
        return action_wait(500, "enter: settling")

    def _verify(self, screen: ScreenState) -> Dict[str, Any]:
        """405 落点校验 (轮播歧义 + Challenge tab)."""
        if self._phase_ticks > _PHASE_MAX:
            return action_done("event_quest verify timeout")
        if self._on_quest_list(screen):
            self.log("verify OK — on event quest list")
            self._set("survey")
            return action_wait(300, "verified event")
        # 错活动的成就页 (情人節約定式): 顺手领 → back
        claim_all = self.find_cls(screen, UC.CLAIM_ALL_YELLOW, conf=0.5)
        if claim_all is not None:
            self.log("wrong event (achievement page) — claim freebies first")
            return action_click_box(claim_all, "claim-all on wrong event (free)")
        got = self.find_cls(screen, UC.GOT_REWARD, conf=0.5)
        if got is not None:
            return action_click(*_POS_TOUCH_CONTINUE, "dismiss reward toast")
        # Challenge tab 落点: 无入场键但有活动底栏 → 切 Quest tab
        if self.find_cls(screen, [UC.EVENT_SHOP, UC.EVENT_TASK],
                         conf=_WEAK_CONF) is not None \
                and not self._enter_keys(screen):
            self.log("landed on Challenge tab — switching to Quest tab")
            return action_click(*_POS_QUEST_TAB, "switch to Quest tab (fixed)")
        # 其他错落点 → back 重试轮播
        self._verify_retries += 1
        if self._verify_retries > _VERIFY_RETRY_MAX:
            return action_done("event_quest: banner carousel retries exhausted")
        self._set("enter")
        # 回 lobby 重进 hub = 轮播重置到 item0, 下一击零延迟点击 (不 back 回
        # hub — back 不重置轮播, 会继续在错误项上转)。
        return self.nav_home(screen,
                             f"wrong landing → lobby, reset carousel "
                             f"({self._verify_retries}/{_VERIFY_RETRY_MAX})")

    # ── survey ─────────────────────────────────────────────────────

    def _survey(self, screen: ScreenState) -> Dict[str, Any]:
        """列表滑到底, 自底向上开尾部 N 关 popup 记录 Bonus 状态."""
        if self._phase_ticks > _PHASE_MAX * 2:
            self.log("survey timeout → close (do NOT retry blind)")
            self._set("close")
            return action_wait(300, "survey timeout")
        if not self._survey_swiped:
            self._survey_swiped = True
            self._survey_settle = 2   # 滚动后等 2 tick 用新鲜坐标 (回弹稳定闸)
            # 向底部滚 = 内容上移 = from 下 to 上 (2026-07-09 live: 方向写反把
            # 列表拖回顶部, tail keys 全点歪 → popup 卡死 stuck20)
            return action_swipe(0.75, 0.72, 0.75, 0.42, duration_ms=400,
                                reason="scroll quest list to bottom")
        if getattr(self, "_survey_settle", 0) > 0:
            self._survey_settle -= 1
            return action_wait(700, "survey: list settling")
        if self._popup_open:
            if self._on_popup(screen):
                self._popup_wait = 0
                self._popup_open = False
                # 去重: X 点漏 popup 残留 / late-popup 重入时不重复记账
                if all(abs(self._survey_cy - c) > 0.03
                       for c in self._surveyed_cys):
                    unlocked = self.find_cls(
                        screen, UC.EVENT_BONUS, conf=_CLS_CONF) is not None
                    self._quests.append((self._survey_cy, unlocked))
                    self._surveyed_cys.append(self._survey_cy)
                    self.log(f"survey quest#{len(self._quests)} "
                             f"(cy={self._survey_cy:.3f})"
                             f" bonus_unlocked={unlocked}")
                else:
                    self.log("survey: duplicate popup (already recorded) — close")
                # 关 popup 后 settle 1 tick 再取 keys (过渡帧检出不全是
                # 同关反复开合 bug 的根因之一)
                self._survey_settle = 1
                close = self.find_cls(screen, UC.BTN_CLOSE_X, conf=_CLS_CONF)
                if close is not None:
                    return action_click_box(close, "close survey popup")
                return action_back("close survey popup (back)")
            self._popup_wait += 1
            if self._popup_wait > 4:
                # 点击没开出 popup (过渡帧点歪) → 放弃本次, 回列表重选同关
                self._popup_wait = 0
                self._popup_open = False
                return action_wait(400, "survey: popup never opened — reselect")
            return action_wait(600, "waiting survey popup")
        if self._on_popup(screen):
            # popup 迟到 (开窗动画慢于重选判定) — 续用上次点击的 cy 记账
            self._popup_open = True
            return action_wait(300, "survey: popup appeared late")
        keys = self._enter_keys(screen)
        # 过渡帧检出不全 → tail 错位, 必须看到完整尾部才动手
        if len(keys) < self._tail_quests:
            return action_wait(600, f"survey: partial list ({len(keys)} keys)")
        tail = keys[-self._tail_quests:]
        # cy 记账: 点过(±0.03)的关不再点
        todo = [b for b in tail
                if all(abs((b.y1 + b.y2) / 2 - c) > 0.03
                       for c in self._surveyed_cys)]
        if not todo:
            # survey 完毕 → unlock 队列 = 未解锁的关 (自底向上: 点数关优先解锁)
            self._quests.sort(key=lambda q: q[0])
            self._quests = self._quests[-self._tail_quests:]
            self._set("unlock")
            return action_wait(300, f"survey done ({len(self._quests)} tail quests)")
        box = todo[0]
        self._survey_cy = (box.y1 + box.y2) / 2
        self._popup_open = True
        self._popup_wait = 0
        return action_click_box(
            box, f"survey open quest popup cy={self._survey_cy:.3f}")

    # ── unlock (加成解锁: 自动编队 + 真打一次) ────────────────────────

    def _unlock(self, screen: ScreenState) -> Dict[str, Any]:
        if self._phase_ticks > _BATTLE_MAX + _PHASE_MAX:
            return action_done("event_quest unlock timeout")
        # 找下一个未解锁的关
        while self._unlock_idx < len(self._quests) and self._quests[self._unlock_idx][1]:
            self._unlock_idx += 1
        if self._unlock_idx >= len(self._quests):
            self._set("points")
            return action_wait(300, "all tail quests bonus-unlocked")
        step = self._formation_step
        # 子步机: popup → 任務開始 → 编队 → 快速編輯 → 自動 → 確認 → 出击 → battle
        if step == "":
            if self._on_popup(screen):
                ts = self.find_cls(screen, UC.TASK_START, conf=0.5)
                if ts is not None:
                    self._formation_step = "edit"
                    return action_click_box(ts, "unlock: 任務開始 → formation")
                return action_wait(500, "unlock: waiting 任務開始")
            keys = self._enter_keys(screen)
            if keys:
                cy = self._quests[self._unlock_idx][0]
                box = min(keys, key=lambda b: abs((b.y1 + b.y2) / 2 - cy))
                # cy 容差防呆: 目标关不在视野 (过渡帧/列表被复位) 绝不点最近邻
                if abs((box.y1 + box.y2) / 2 - cy) <= 0.04:
                    return action_click_box(box, f"unlock: open quest cy={cy:.3f}")
                if self._phase_ticks % 3 == 0:
                    return action_swipe(0.75, 0.72, 0.75, 0.42, duration_ms=400,
                                        reason="unlock: re-scroll to bottom")
                return action_wait(600, "unlock: target quest not in view")
            return action_wait(600, "unlock: waiting list")
        if step == "edit":
            sq2_hi = self.find_cls(screen, UC.SQUAD_2_HI, conf=_CLS_CONF)
            if sq2_hi is None:
                sq2 = self.find_cls(screen, UC.SQUAD_2, conf=_CLS_CONF)
                if sq2 is not None:
                    return action_click_box(sq2, "unlock: switch to squad 2")
                return action_wait(600, "unlock: waiting formation screen")
            qe = self.find_cls(screen, UC.SQUAD_QUICK_EDIT, conf=_CLS_CONF)
            if qe is not None:
                self._formation_step = "auto"
                return action_click_box(qe, "unlock: 快速編輯")
            return action_wait(500, "unlock: waiting 快速編輯 btn")
        if step == "auto":
            auto = self.find_cls(screen, UC.SQUAD_AUTO_EDIT, conf=_CLS_CONF)
            if auto is not None:
                self._formation_step = "confirm"
                return action_click_box(auto, "unlock: 自動 (event-optimal)")
            return action_wait(500, "unlock: waiting 自動 btn")
        if step == "confirm":
            conf_btn = self.find_cls(screen, UC.BTN_CONFIRM, conf=0.5)
            if conf_btn is not None:
                self._formation_step = "sortie"
                return action_click_box(conf_btn, "unlock: confirm auto-formation")
            return action_wait(500, "unlock: waiting 確認")
        if step == "sortie":
            sortie = self.find_cls(screen, UC.SORTIE, conf=_CLS_CONF)
            if sortie is not None:
                # 加成% 读数 (>0 才值得打; 读不出保守出击)
                pct_raw = _read_digits(screen.frame, _R_SQUAD_PCT)
                if pct_raw:
                    digits = "".join(ch for ch in pct_raw if ch.isdigit())
                    if digits.isdigit() and int(digits) == 0:
                        self.log("unlock: 0% event bonus (no bonus students) — "
                                 "skip sortie, mark unlocked")
                        self._quests[self._unlock_idx] = (
                            self._quests[self._unlock_idx][0], True)
                        self._formation_step = ""
                        return action_back("0% bonus — leave formation")
                self.log(f"unlock: sortie w/ squad-2 (bonus%={pct_raw!r})")
                self._formation_step = "battle"
                self._battle_ticks = 0
                return action_click_box(sortie, "unlock: 出击 (bonus run)")
            return action_wait(600, "unlock: waiting 出击")
        if step == "battle":
            self._battle_ticks += 1
            if self._battle_ticks > _BATTLE_MAX:
                return action_done("event_quest battle timeout")
            # AUTO gate (灰 → 点亮)
            auto_off = self.find_cls(screen, UC.BATTLE_AUTO_OFF, conf=0.5)
            if auto_off is not None:
                return action_click_box(auto_off, "battle: AUTO on (gate)")
            conf_btn = self.find_cls(screen, UC.BTN_CONFIRM, conf=0.6)
            if conf_btn is not None:
                self._formation_step = "settle"
                return action_click_box(conf_btn, "battle: settle 確認")
            return action_wait(4000, "battling (AUTO gate armed)")
        if step == "settle":
            conf_btn = self.find_cls(screen, UC.BTN_CONFIRM, conf=0.6)
            if conf_btn is not None:
                return action_click_box(conf_btn, "settle: 確認 (total reward)")
            if self._on_quest_list(screen):
                self.log(f"unlock: quest#{self._unlock_idx} bonus run DONE")
                self._quests[self._unlock_idx] = (
                    self._quests[self._unlock_idx][0], True)
                self._unlock_idx += 1
                self._formation_step = ""
                return action_wait(500, "back on list after unlock")
            return action_click(*_POS_TOUCH_CONTINUE, "settle: TOUCH continue")
        return action_wait(500, f"unlock: step {step}")

    # ── points / currency sweep ────────────────────────────────────

    def _points(self, screen: ScreenState) -> Dict[str, Any]:
        """尾关(点数关) MAX 扫直到 目标 / AP尽."""
        pts = self._read_points(screen)
        if pts is not None:
            self.log(f"活動點數 {pts[0]}/{pts[1]}")
            if pts[0] >= min(self._points_target, pts[1]):
                self._points_done = True
                self._set("currency")
                return action_wait(300, "points target reached → currency")
        return self._sweep_quest(screen, quest_idx=len(self._quests) - 1,
                                 phase_after="tasks", label="points")

    def _currency(self, screen: ScreenState) -> Dict[str, Any]:
        """货币关轮流 MAX 扫 (点数满后才进入)."""
        n = len(self._quests)
        if n < 2:
            self._set("tasks")
            return action_wait(200, "no currency quests")
        # 轮转: 倒数第2 → 倒数第3 → 倒数第4
        idx = n - 2 - (self._currency_idx % max(1, n - 1))
        return self._sweep_quest(screen, quest_idx=max(0, idx),
                                 phase_after="tasks", label="currency")

    def _sweep_quest(self, screen: ScreenState, quest_idx: int,
                     phase_after: str, label: str) -> Dict[str, Any]:
        """通用 MAX 扫荡子流程 (money-gated)."""
        if self._sweep_rounds > _SWEEP_ROUNDS_MAX:
            self._set(phase_after)
            return action_wait(200, "sweep rounds cap")
        if self._phase_ticks > _PHASE_MAX * 3:
            self._set(phase_after)
            return action_wait(200, f"{label} sweep phase timeout")
        # 结果/确认链
        skip = self.find_cls(screen, UC.BATTLE_SKIP, conf=0.6)
        if skip is not None:
            return action_click_box(skip, f"{label}: skip sweep animation")
        got = self.find_cls(screen, UC.GOT_REWARD, conf=0.5)
        if got is not None:
            return action_click(*_POS_TOUCH_CONTINUE, f"{label}: dismiss reward")
        if self._on_popup(screen):
            # MAX 可点 → 点; MAX 灰 → AP 不足, 收工
            qmax = self.find_cls(screen, UC.QTY_MAX, conf=_WEAK_CONF)
            qmax_grey = self.find_cls(screen, UC.QTY_MAX_GREY, conf=_CLS_CONF)
            plus_grey = self.find_cls(screen, "加号灰色", conf=_CLS_CONF)
            if qmax is not None and qmax_grey is None:
                return action_click_box(qmax, f"{label}: MAX")
            if qmax_grey is not None and plus_grey is None:
                # MAX 灰但加号亮 = 已顶格待开扫 (MAX 点过)
                pass
            if qmax_grey is not None and plus_grey is not None:
                ss = self.find_cls(screen, UC.SWEEP_START, conf=0.5)
                # 全灰: 顶格(可扫) or AP<单次(不可扫)。用 count 区分不了 —
                # 点一次扫荡开始, 确认框出不来(AP不足时按钮无效)则收工。
                if ss is not None and self._phase_ticks % 3 == 1:
                    self._sweep_rounds += 1
                    return action_click_box(ss, f"{label}: 掃蕩開始 (round "
                                                f"{self._sweep_rounds})")
                close = self.find_cls(screen, UC.BTN_CLOSE_X, conf=_CLS_CONF)
                if self._phase_ticks > 8 and close is not None:
                    self.log(f"{label}: sweep not opening (AP exhausted) → done")
                    self._set(phase_after)
                    return action_click_box(close, f"{label}: close popup, AP done")
            return action_wait(600, f"{label}: popup settling")
        # 确认框 (要使用NAP掃蕩N次嗎?)
        conf_btn = self.find_cls(screen, UC.BTN_CONFIRM, conf=0.6)
        cancel_btn = self.find_cls(screen, UC.BTN_CANCEL, conf=0.5)
        if conf_btn is not None and cancel_btn is not None:
            # ⛔ money gate: body 有青辉石 = 買AP框 → 取消 + 全 skill 收工
            if self._pyroxene_in_body(screen):
                self.log("⛔ pyroxene in sweep-confirm body — CANCEL (fail-closed)")
                self._set("close")
                return action_click_box(cancel_btn, "PYROXENE DIALOG — cancel!")
            self._swept += 1
            return action_click_box(conf_btn, f"{label}: confirm sweep (pure AP)")
        if conf_btn is not None and cancel_btn is None:
            # 掃蕩完成框 (确认键无取消): tooltip 坑 → 连点两次由 tick 自然完成
            return action_click_box(conf_btn, f"{label}: 掃蕩完成 確認")
        # 列表页 → 开目标关 popup
        keys = self._enter_keys(screen)
        if keys and quest_idx < len(self._quests):
            cy = self._quests[quest_idx][0]
            box = min(keys, key=lambda b: abs((b.y1 + b.y2) / 2 - cy))
            # cy 容差防呆 (同 survey/unlock): 目标关不在视野绝不点最近邻
            if abs((box.y1 + box.y2) / 2 - cy) <= 0.04:
                return action_click_box(box, f"{label}: open quest cy={cy:.3f}")
            if self._phase_ticks % 3 == 0:
                return action_swipe(0.75, 0.72, 0.75, 0.42, duration_ms=400,
                                    reason=f"{label}: re-scroll to bottom")
            return action_wait(600, f"{label}: target quest not in view")
        return action_wait(600, f"{label}: waiting")

    # ── tasks (活动任務领奖) / close ──────────────────────────────────

    def _tasks(self, screen: ScreenState) -> Dict[str, Any]:
        if self._phase_ticks > _PHASE_MAX:
            self._set("close")
            return action_wait(200, "tasks phase timeout")
        if self._tasks_done:
            self._set("close")
            return action_wait(200, "tasks already claimed")
        # 在任務页: 领取链
        claim_all = self.find_cls(screen, UC.CLAIM_ALL_YELLOW, conf=0.5)
        if claim_all is not None:
            return action_click_box(claim_all, "tasks: 全部領取")
        claim = self.find_cls(screen, UC.CLAIM_YELLOW, conf=0.5)
        if claim is not None:
            return action_click_box(claim, "tasks: 領取")
        got = self.find_cls(screen, UC.GOT_REWARD, conf=0.5)
        if got is not None:
            return action_click(*_POS_TOUCH_CONTINUE, "tasks: dismiss reward")
        grey = self.find_cls(screen, [UC.CLAIM_ALL_GREY], conf=0.5)
        if grey is not None:
            self.log("tasks: all claimed (grey)")
            self._tasks_done = True
            self._set("close")
            return action_back("tasks done → back")
        # 还在 quest 列表: 任務按钮红点门控
        task_btn = self.find_cls(screen, UC.EVENT_TASK, conf=_WEAK_CONF)
        if task_btn is not None:
            region = (task_btn.x1 - 0.02, task_btn.y1 - 0.05,
                      task_btn.x2 + 0.05, task_btn.y2 + 0.02)
            if self.dot_in_region(screen, region, dot_classes=("红点",)):
                return action_click_box(task_btn, "tasks: enter 任務 (red dot)")
            self.log("tasks: no red dot on 任務 — skip")
            self._tasks_done = True
            self._set("close")
            return action_wait(200, "no task rewards")
        return action_wait(600, "tasks: waiting")

    def _close(self, screen: ScreenState) -> Dict[str, Any]:
        page = self.detect_screen_yolo(screen)
        if page == "Mission":
            return action_done(f"event_quest complete on hub (swept={self._swept})")
        if page == "Lobby":
            return action_done(f"event_quest complete (swept={self._swept})")
        if self._phase_ticks > _PHASE_MAX:
            return action_done(f"event_quest exit timeout (swept={self._swept})")
        if self._phase_ticks % 3 != 1:
            return action_wait(600, "closing — settling")
        close = self.find_cls(screen, UC.BTN_CLOSE_X, conf=_CLS_CONF)
        if close is not None:
            return action_click_box(close, "close popup (X)")
        back = self.find_cls(screen, UC.BTN_BACK, conf=_CLS_CONF)
        if back is not None:
            return action_click_box(back, "→ hub (back)")
        return self.nav_home(screen, "event_quest close")
