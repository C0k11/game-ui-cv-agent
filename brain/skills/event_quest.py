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
    action_swipe, action_swipe_tap,
)
from brain.skills import ui_classes as UC

_CLS_CONF = 0.30
_WEAK_CONF = 0.20          # 活动皮肤弱类 (活动商店/活动任务/活动quest_已选择) 地板
_ENTER_MAX = 24
_VERIFY_RETRY_MAX = 10     # 405 轮播重试上限 (双相位修正后期望 ≤3 轮命中)
_PHASE_MAX = 18
_BATTLE_MAX = 46           # battle poll ticks (pipeline tick ~5s → ~230s)
_SWEEP_ROUNDS_MAX = 30     # 点数期一次 MAX 就把 AP 扫光, 这是保险帽
_TAIL_QUESTS = 4           # 尾部加成关数量 (有时3有时4, 用户 2026-07-08)

# 固定位 (live 实测, 帧目检):
_POS_QUEST_TAB = (0.635, 0.151)     # 活动页 Quest tab (cls 活动quest_已选择 仅0.24-0.59)
_POS_TOUCH_CONTINUE = (0.5, 0.90)   # 结算 TOUCH-continue
# digitOCR regions:
_R_POINTS = (0.55, 0.93, 0.82, 0.985)   # quest列表底部 活動點數 "2676/15000"
# 任務資訊 popup 獲得期待獎勵 行的 xN 数字带 (2026-07-11 双帧标定:
# Q09 固定36/Bonus6, Q12 固定36/Bonus35 全中)
_R_SLOT_FIRST_XN = (0.126, 0.780, 0.172, 0.818)   # 首槽固定掉落 xN
_R_SLOT_BONUS_XN = (0.338, 0.780, 0.394, 0.818)   # Bonus 槽 xN(徽章正下)
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
        # banner 实例记账(pHash): 点错的轮播项拉黑防复点(情人節同为405)
        self._banner_bad_hashes: list = []
        self._pending_hash = None
        # 上次落点是否盲区页(no-UI escape 逃生回 hub) — 决定下次 tap 相位修正
        self._blind_landing = False
        # ⭐加成台账(用户 2026-07-11: 本地记录哪些关打过加成, 记过的不再挨个
        # 开 popup 检查; 没打过的活动内一次性补打)
        self._bonus_ledger: dict = {}     # {round(cy,3): True} 只记已解锁
        self._bonus_created = 0.0
        self._load_bonus_ledger()
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
        self._milestone_done = False  # 里程碑(獎勵資訊)领奖阶段

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
        # tab 指示器。⚠2026-07-11 勘误: 活动页选中态 Challenge tab 也会误检成
        # 普通关卡选中@0.97 — 有活动内铁证(奖励资讯/活动quest tab)时不 veto,
        # 交给底栏硬条件判(真·特殊作戰无活动底栏, 本来就过不了)。
        if self.find_cls(screen, "普通关卡选中", conf=0.5) is not None \
                and self.find_cls(screen, [UC.EVENT_REWARD_INFO, UC.EVENT_QUEST,
                                           UC.EVENT_QUEST_SEL],
                                  conf=_WEAK_CONF) is None:
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

    # ── banner 实例记账 (⭐感知铁律 2026-07-11: 判断主导=cls, pHash 只做
    # "同一实例防复点"记账辅助) ────────────────────────────────────────
    # hub 轮播实测(v13): 遊戲開發部=405@0.97 / 情人節=405@0.97(复刻, 无关卡,
    # 落成就页) / 鋼鐵大陸=474@0.88(领奖余韵)。405 有两个实例 → 点错落成就页
    # 由 verify 用 cls 证据识别, 该实例 pHash 拉黑, 下次 405 检出先比对。
    _BANNER_BODY_CROP = (0.02, 0.16, 0.16, 0.285)   # hub banner 本体(405条下方)

    def _banner_phash(self, screen: ScreenState, frame=None):
        try:
            import cv2
            fr = frame if frame is not None else screen.frame
            if fr is None:
                return None
            h, w = fr.shape[:2]
            x1, y1, x2, y2 = self._BANNER_BODY_CROP
            crop = fr[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]
            g = cv2.cvtColor(cv2.resize(crop, (16, 16)), cv2.COLOR_BGR2GRAY)
            return (g > g.mean()).flatten()
        except Exception:
            return None

    def _banner_blacklisted(self, screen: ScreenState, frame=None) -> bool:
        cur = self._banner_phash(screen, frame)
        if cur is None or not self._banner_bad_hashes:
            return False
        return any(int((cur != b).sum()) < 40 for b in self._banner_bad_hashes)

    def _blacklist_pending(self, why: str) -> None:
        if self._pending_hash is not None:
            self._banner_bad_hashes.append(self._pending_hash)
            self._pending_hash = None
            self.log(f"banner instance blacklisted ({why}), "
                     f"total={len(self._banner_bad_hashes)}")

    # ── 加成台账 (persistent, data/event_bonus_state.json) ────────────
    _BONUS_STATE_PATH = "data/event_bonus_state.json"

    def _load_bonus_ledger(self) -> None:
        try:
            import json, os, time as _t
            if os.path.exists(self._BONUS_STATE_PATH):
                d = json.load(open(self._BONUS_STATE_PATH, encoding="utf-8"))
                created = float(d.get("created", 0))
                if _t.time() - created < 10 * 86400:   # 超10天=新活动, 作废
                    self._bonus_created = created
                    self._bonus_ledger = {
                        round(float(k), 3): True
                        for k, v in (d.get("quests") or {}).items() if v}
        except Exception:
            self._bonus_ledger = {}

    def _mark_bonus_done(self, cy: float) -> None:
        self._bonus_ledger[round(cy, 3)] = True
        try:
            import json, time as _t
            if not self._bonus_created:
                self._bonus_created = _t.time()
            with open(self._BONUS_STATE_PATH, "w", encoding="utf-8") as f:
                json.dump({"created": self._bonus_created,
                           "quests": {f"{k:.3f}": True
                                      for k in self._bonus_ledger}}, f)
        except Exception:
            pass

    def _bonus_recorded(self, cy: float) -> bool:
        return any(abs(cy - k) <= 0.03 for k in self._bonus_ledger)

    def _pyroxene_in_body(self, screen: ScreenState) -> bool:
        """money backstop: 对话框 body(y>0.12) 出现青辉石 = 绝不是纯AP扫荡框."""
        for b in (screen.yolo_boxes or []):
            if b.cls_name == UC.TOPBAR_PYROXENE and b.confidence >= 0.20:
                if (b.y1 + b.y2) / 2 > 0.12:
                    return True
        return False

    # ⛔⛔ 2026-07-11 事故修复(30青辉石买AP, 逐帧取证 run_144712 t30):
    # 「購買AP」框上 v13 对 body 青辉石**完全漏检**(黑名单gate盲) → 需要
    # 结构白名单: 纯AP扫荡确认框(通知) body 只有 取消/确认/叉(t56实锤);
    # 購買AP框 body 必有 数量stepper(MIN_灰色/加号/MAX_可点击@0.96)+
    # 体力图标@0.92(0.820,0.680)。确认框上下文里检到任一 = 购买框, 取消。
    _BUY_DIALOG_MARKERS = ("MAX_可点击", "MIN_灰色", "MIN_可点击", "加号",
                           "减号", "加号灰色", "减号灰色")

    def _dialog_is_purchase(self, screen: ScreenState) -> bool:
        """确认+取消同屏语境下: body 出现数量stepper/体力/青辉石 = 购买框.

        阈值分层(2026-07-11 二次校准): 纯AP扫荡确认框正文"要使用NAP掃蕩N次嗎"
        会在(0.78,0.68)冒 体力@0.38 弱检出(t46 误杀实锤, swept=0); 真購買AP框
        的体力图标@0.92 → 体力通道阈值 0.60 干净分离。stepper 通道保持 0.20
        (确认框上 dim 盖住底层 popup, stepper 零检出=t46 实证; 真購買AP框
        stepper@0.96)。青辉石保持 0.20 最大灵敏(body 内青辉石永远不合法)。"""
        if self._pyroxene_in_body(screen):
            return True
        for b in (screen.yolo_boxes or []):
            cy = (b.y1 + b.y2) / 2
            if cy <= 0.12 or b.confidence < 0.20:
                continue
            if b.cls_name in self._BUY_DIALOG_MARKERS:
                return True
            if b.cls_name == UC.TOPBAR_AP and cy > 0.12 and b.confidence >= 0.60:
                return True
        return False

    def _read_ap(self, screen: ScreenState) -> Optional[int]:
        """topbar AP 读数(体力icon cls锚定 + 右侧数字strip, span 0.06 只读
        斜杠前; 主tick帧=ADB干净帧)。读不出 → None(调用方 fail-closed)."""
        icon = self.find_cls(screen, UC.TOPBAR_AP, conf=0.5)
        if icon is None or screen.frame is None:
            return None
        cy = (icon.y1 + icon.y2) / 2
        if cy > 0.12:          # body 里的体力图标(如購買AP框)不是 topbar
            return None
        raw = _read_digits(screen.frame,
                           (icon.x2 + 0.002, max(0.0, icon.y1 - 0.012),
                            icon.x2 + 0.062, icon.y2 + 0.012))
        if not raw:
            return None
        num = ""
        for ch in raw:
            if ch.isdigit():
                num += ch
            elif num:
                break
        return int(num) if num else None

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
        """⭐感知铁律(2026-07-11): cls 主导 + 极端事件驱动 —
        目标 cls(405)出现→立即点; 加载中→wait; 无锁定→wait(轮播在转)。"""
        if self._phase_ticks > _ENTER_MAX:
            return action_done("event_quest enter timeout")
        # already on the quest list? (re-entry / retry path)
        if self._on_quest_list(screen):
            self._set("survey")
            return action_wait(300, "on event quest list")
        if screen.is_loading():
            return action_wait(400, "加载中 loading — wait")
        # 目标 cls: 405 距離結束還剩 = 进行中活动 banner(hub 轮播实测 0.97)。
        # 474 距離獎勵獲得結束 = 领奖余韵, 不点。情人節同为 405 → 点错由
        # verify 用 cls 证据识别并 pHash 拉黑该实例, 此处先比对黑名单。
        # ⭐帧龄(2026-07-11 工业级链路): 主 tick 帧龄 ~2.2s vs 轮播 2.6s/项
        # → "帧上的项"和"点到的项"错位率 ~85%(五连点全落错实锤)。高频
        # DXcam 线程检出(帧龄≤0.5s)可用且新鲜时优先, 判定/pHash/点击
        # 坐标全用同一新鲜源。
        import time as _time
        fresh = getattr(screen, "fresh_boxes", None)
        fresh_frame = getattr(screen, "fresh_frame", None)
        # ⚠fresh_ts 时基 = server 高频线程的 perf_counter(同进程同时基),
        # 不是 epoch time.time()(2026-07-11 踩坑: 混用 age 恒大 → 永远回退)
        fresh_age = _time.perf_counter() - getattr(screen, "fresh_ts", 0.0)
        tick_banner = self.find_cls(
            screen, [UC.EVENT_END_LEFT, UC.EVENT_REWARD_END], conf=0.5)
        banner = None
        src_frame = None
        fresh_usable = (fresh is not None and fresh_age < 1.2
                        and not getattr(self, "_fresh_distrust", False))
        if fresh_usable:
            for b in fresh:
                if b.cls_name == UC.EVENT_END_LEFT and b.confidence >= 0.5:
                    banner = b
                    src_frame = fresh_frame
                    break
            # 失信裁决(2026-07-11: DXcam 桌面遮挡/窗口错位时 fresh 内容错 —
            # 主 tick 帧看得见 banner 而 fresh 完全没有 = 源不可信, 2 次即弃)
            if banner is None and tick_banner is not None:
                fresh_any = any(b.cls_name in (UC.EVENT_END_LEFT,
                                               UC.EVENT_REWARD_END)
                                and b.confidence >= 0.4 for b in fresh)
                if not fresh_any:
                    self._fresh_mismatch = getattr(self, "_fresh_mismatch", 0) + 1
                    if self._fresh_mismatch >= 2:
                        self._fresh_distrust = True
                        self.log("fresh source DISTRUSTED (tick sees banner, "
                                 "fresh blind) — pure tick mode")
        if banner is None and not fresh_usable:
            # ⭐轮播时序全模型(2026-07-11 下午 6+6 落点反推钉死):
            # 项周期≈5s(0709 实测对, 早上 2.6s 估计错), hub 每次进入/back 返回
            # 都重置到 item0=遊戲開發部(目标); 循环序[遊→情→特殊/鋼474→遊]。
            # 两条确定性路径, tap 相位修正按来路选:
            # ① **快节奏**(cls 证据页 back 秒回 / 首次进 hub): enter 下一 tick
            #    就开火, 动作落 item0 的 5s 窗口内 → live==帧上项 → **直点**。
            #    (下午 5/5 落"帧上项的上一项"实锤 = 回滑画蛇添足的反证)
            # ② **慢节奏**(盲区页 no-UI escape ~30s 逃生, verify 多插一个
            #    "still on hub" tick): 动作压过 5s 边界 → live=帧+1 →
            #    **原子回滑一格&&tap** 退回帧上项(swipe 拉停轮播, 静止期内
            #    tap 必落; 下午实锤回滑翻页机制本身 5/5 生效)。
            # 帧上=474 → wait(直点会落到 474 项自己, 等轮播转到 405)。
            b405 = self.find_cls(screen, UC.EVENT_END_LEFT, conf=0.5)
            if b405 is not None:
                # tick 模式不记 pHash("帧上项"≠"落点项"时拉黑反噬)
                self._pending_hash = None
                self._set("verify")
                if self._blind_landing:
                    self._blind_landing = False
                    bcx = (b405.x1 + b405.x2) / 2
                    bcy = (b405.y1 + b405.y2) / 2
                    return action_swipe_tap(
                        0.035, 0.185, 0.11, 0.185, bcx, bcy, duration_ms=150,
                        reason="banner swipe-back tap (盲区退回慢路径, live=帧+1 → 回滑对齐)")
                return action_click_box(
                    b405, "banner tap (快节奏, live==帧上项 → 直点)")
        if banner is not None:
            if src_frame is not None and self._banner_blacklisted(screen, src_frame):
                return action_wait(400, "405=已拉黑实例 — 等轮播下一项")
            if self._blind_landing:
                # fresh 分支同样吃相位修正(2026-07-21 三连同点 miss 实锤:
                # verify 置 _blind_landing 后 fresh 分支从不读 = 死码, 三次
                # 一模一样直点零自适应)。miss 后改 swipe-back 拉停轮播再点。
                self._blind_landing = False
                self._pending_hash = None
                self._set("verify")
                bcx = (banner.x1 + banner.x2) / 2
                bcy = (banner.y1 + banner.y2) / 2
                return action_swipe_tap(
                    0.035, 0.185, 0.11, 0.185, bcx, bcy, duration_ms=150,
                    reason="banner swipe-back tap (fresh, miss后回滑对齐)")
            # pHash 记账只在 fresh 模式有意义(tick 错位下"帧上项"≠"落点项",
            # 拉黑会反噬正确项)
            self._pending_hash = (self._banner_phash(screen, src_frame)
                                  if src_frame is not None else None)
            self._set("verify")
            tag = "fresh" if src_frame is not None else "tick+1"
            return action_click_box(banner, f"banner tap ({tag}, "
                                            f"帧上={banner.cls_name})")
        hub = self.find_cls(screen, UC.NAV_TASKS, conf=_CLS_CONF)
        if hub is not None:
            return action_click_box(hub, "lobby → task hub")
        # 无锁定: hub 轮播转在 474/卡池项 或 lobby 未识别 — wait 等目标出现;
        # 久等不出(2 个轮播周期+)才 nav_home 重置。
        if self._phase_ticks % 8 == 7:
            return self.nav_home(screen, "no 405 for 8 ticks — reset")
        return action_wait(400, "no target cls — wait (carousel)")

    def _verify(self, screen: ScreenState) -> Dict[str, Any]:
        """落点校验 — ⭐cls 证据制(2026-07-11): 有明确 cls 证据才判,
        转场/无检出/加载中一律 wait(旧版在转场帧 2s 枪毙点对的入口×3)。"""
        if self._on_quest_list(screen):
            self.log("verify OK — on event quest list")
            self._pending_hash = None            # 点对了, 不拉黑
            self._verify_retries = 0
            self._set("survey")
            return action_wait(300, "verified event")
        if screen.is_loading():
            return action_wait(400, "verify: 加载中 loading — wait")
        # 证据1: 成就页(情人節式复刻) → 顺手领免费 → 拉黑该 banner 实例
        claim_all = self.find_cls(screen, UC.CLAIM_ALL_YELLOW, conf=0.5)
        if claim_all is not None:
            self._blacklist_pending("achievement page (claimable)")
            return action_click_box(claim_all, "claim-all on wrong event (free)")
        grey = self.find_cls(screen, UC.CLAIM_ALL_GREY, conf=0.5)
        if grey is not None:
            self._blacklist_pending("achievement page (all claimed)")
            self._verify_retries += 1
            if self._verify_retries > _VERIFY_RETRY_MAX:
                return action_done("event_quest: carousel retries exhausted")
            self._set("enter")
            self._blind_landing = False   # cls 页秒回 = 快节奏路径
            return action_back("wrong event(领完) → back, rescan carousel")
        got = self.find_cls(screen, UC.GOT_REWARD, conf=0.5)
        if got is not None:
            return action_click(*_POS_TOUCH_CONTINUE, "dismiss reward toast")
        # 证据2前置(2026-07-11 下午 5 连误杀实锤): **选中态 Challenge tab 会被
        # 误检为 普通关卡选中@0.97(0.839,0.151)** — 同视觉模式。活动开出
        # Challenge 关后落地默认停 Challenge tab, 旧判序把点对的入口当特殊作戰
        # 枪毙。页上铁证: 奖励资讯@0.96(0.877,0.898) + 活动quest(未选中Quest
        # tab)@0.73(0.635,0.153) → 已在目标活动内, 点 Quest tab 切页即可。
        ev_marker = self.find_cls(
            screen, [UC.EVENT_REWARD_INFO, UC.EVENT_SHOP, UC.EVENT_TASK,
                     UC.EVENT_QUEST, UC.EVENT_QUEST_SEL], conf=_WEAK_CONF)
        if ev_marker is not None and self._phase_ticks <= _PHASE_MAX * 2:
            self._pending_hash = None
            qtab = self.find_cls(screen, UC.EVENT_QUEST, conf=_WEAK_CONF)
            if qtab is not None:
                return action_click_box(
                    qtab, "in event, Challenge tab → Quest tab (cls锚定)")
            return action_click(*_POS_QUEST_TAB, "in event → Quest tab (fixed)")
        # 证据2: 特殊作戰列表(普通关卡选中 tab 指示器, 且无任何活动内标记)
        if self.find_cls(screen, "普通关卡选中", conf=0.5) is not None:
            # ⚠不拉黑: 回滑/相位残差下"帧上项"≠"落点项", 拉黑会反噬目标
            self._pending_hash = None
            self._verify_retries += 1
            if self._verify_retries > _VERIFY_RETRY_MAX:
                return action_done("event_quest: carousel retries exhausted")
            self._set("enter")
            self._blind_landing = False   # cls 页秒回 = 快节奏路径
            return action_back("wrong: 特殊作戰 → back, rescan")
        # 证据3: 仍在 hub(hub tile 可见) = tap 落空/轮播翻走 → 直接重扫
        if self.find_cls(screen, [UC.HUB_BOUNTY, "学院交流会", "战术大赛"],
                         conf=0.5) is not None:
            self._pending_hash = None            # 没点进任何页, 不拉黑
            self._verify_retries += 1
            if self._verify_retries > _VERIFY_RETRY_MAX:
                return action_done("event_quest: carousel retries exhausted")
            self._set("enter")
            # 回到 hub 却没经过 cls 证据页 = 盲区页 no-UI escape 逃生(慢路径,
            # verify 已多吃一个 tick, 下次动作会压过项边界) → 相位修正=回滑
            self._blind_landing = True
            return action_wait(300, "still on hub (tap missed) — rescan")
        # 证据4: Challenge tab 落点(活动底栏在, 无入场键) → 切 Quest tab
        if self.find_cls(screen, [UC.EVENT_SHOP, UC.EVENT_TASK],
                         conf=_WEAK_CONF) is not None \
                and not self._enter_keys(screen):
            self.log("landed on Challenge tab — switching to Quest tab")
            return action_click(*_POS_QUEST_TAB, "switch to Quest tab (fixed)")
        # 无证据(转场渲染/模型盲区页) → wait; 超时兜底 back 重扫
        if self._phase_ticks > _PHASE_MAX:
            self._verify_retries += 1
            if self._verify_retries > _VERIFY_RETRY_MAX:
                return action_done("event_quest: carousel retries exhausted")
            self._set("enter")
            self._blind_landing = True   # 无证据超时 = 盲区页同型(慢路径)
            return action_back("verify timeout (no evidence) → back, rescan")
        return action_wait(500, "verify: no cls evidence yet — wait")

    # ── survey ─────────────────────────────────────────────────────

    def _survey(self, screen: ScreenState) -> Dict[str, Any]:
        """列表滑到底, 自底向上开尾部 N 关 popup 记录 Bonus 状态."""
        if self._phase_ticks > _PHASE_MAX * 2:
            self.log("survey timeout → close (do NOT retry blind)")
            self._set("close")
            return action_wait(300, "survey timeout")
        # ⭐页面丢失自愈(2026-07-11 live: 点入場后 1s 内被弹回 hub, 无我方
        # 动作, 原因未明 — 任何被踢出场景一律重进而不是干等超时吃 swept=0)
        if (self.find_cls(screen, [UC.HUB_BOUNTY, "学院交流会", "战术大赛"],
                          conf=0.5) is not None
                or self.find_cls(screen, UC.NAV_TASKS, conf=0.5) is not None):
            self.log("survey: kicked out to hub/lobby — re-enter")
            self._survey_swiped = False
            self._popup_open = False
            self._set("enter")
            return action_wait(200, "survey: page lost → re-enter")
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
                # ⭐先停一拍再判(用户 2026-07-11: "点开马上就关, 没好好对比"
                # — 半渲染帧上 EVENT_BONUS 会漏检 → 误判已解锁)。settle 1 tick
                # 用稳定帧判, 判据仍是 cls「活动关卡产出额外加成」(0708 双向
                # 实测: 解锁前 0 检出/解锁后 0.95+; 数字对比=后续升级路径)。
                if not getattr(self, "_popup_judge_settle", 0):
                    self._popup_judge_settle = 1
                    return action_wait(500, "survey: popup settling before judge")
                self._popup_judge_settle = 0
                self._popup_wait = 0
                self._popup_open = False
                # 去重: X 点漏 popup 残留 / late-popup 重入时不重复记账
                if all(abs(self._survey_cy - c) > 0.03
                       for c in self._surveyed_cys):
                    # ⭐数字对比判定(用户 2026-07-11, Q09 实锤: 徽章在场但
                    # Bonus×6 vs 固定×36 = 弱队烂记录, 徽章判定天然不够)。
                    # Bonus xN ≥ 80% 固定 xN 才算加成打满; 读不出→保守按
                    # 徽章判但不记台账(下轮重查)。region 已双帧标定
                    # (Q09 36/6, Q12 36/35 全中)。
                    badge = self.find_cls(screen, UC.EVENT_BONUS, conf=0.5)
                    unlocked = badge is not None
                    numeric_ok = False
                    if unlocked and screen.frame is not None:
                        fn_raw = _read_digits(screen.frame, _R_SLOT_FIRST_XN)
                        bn_raw = _read_digits(screen.frame, _R_SLOT_BONUS_XN)
                        fn = int(fn_raw) if fn_raw and fn_raw.isdigit() else None
                        bn = int(bn_raw) if bn_raw and bn_raw.isdigit() else None
                        if fn and bn is not None:
                            numeric_ok = True
                            unlocked = bn * 5 >= fn * 4
                            self.log(f"survey: 固定x{fn} vs Bonus x{bn} → "
                                     f"{'满加成' if unlocked else '烂加成需重打'}")
                    self._quests.append((self._survey_cy, unlocked))
                    self._surveyed_cys.append(self._survey_cy)
                    if unlocked and numeric_ok:
                        self._mark_bonus_done(self._survey_cy)
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
        _cy = (box.y1 + box.y2) / 2
        # ⭐台账命中 = 该关加成已打过, 不再开 popup 复查(用户 2026-07-11)
        if self._bonus_recorded(_cy):
            self._quests.append((_cy, True))
            self._surveyed_cys.append(_cy)
            self.log(f"survey: cy={_cy:.3f} 台账已记加成解锁 — skip popup")
            return action_wait(200, "survey: ledger hit, next")
        self._survey_cy = _cy
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
        # ⛔AP 闸(2026-07-11): 解锁=加成队真出击一场(消耗关卡 AP ~20)。
        # AP<20 → 跳过解锁直接 points(台账不记, 明天带 AP 优先补打), 绝不
        # 在 AP 不足时往出击链里走。
        if self._formation_step == "":
            _ap = self._read_ap(screen)
            if _ap is not None and _ap < 20:
                self.log(f"unlock: AP={_ap} <20 无法出击解锁 — 留待明日补打")
                self._set("points")
                return action_wait(300, "unlock deferred (AP insufficient)")
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
                        self._mark_bonus_done(self._quests[self._unlock_idx][0])
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
                self._mark_bonus_done(self._quests[self._unlock_idx][0])
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
                                 phase_after="milestone", label="points")

    def _currency(self, screen: ScreenState) -> Dict[str, Any]:
        """货币关轮流 MAX 扫 (点数满后才进入)."""
        n = len(self._quests)
        if n < 2:
            self._set("tasks")
            return action_wait(200, "no currency quests")
        # 轮转: 倒数第2 → 倒数第3 → 倒数第4
        idx = n - 2 - (self._currency_idx % max(1, n - 1))
        return self._sweep_quest(screen, quest_idx=max(0, idx),
                                 phase_after="milestone", label="currency")

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
                # ⛔源头闸(2026-07-11 事故修复): 全灰≠可扫 — AP=0 时点掃蕩開始
                # 会弹「購買AP」框(30青辉石实锤)。扫前必读 topbar AP, 读不出
                # 或 <20(单次成本) 一律收工, **绝不试探点掃蕩開始**(fail-closed,
                # 同 special_sweep 0615 教训: 耗尽型扫荡必先读余额)。
                if ss is not None and self._phase_ticks % 3 == 1:
                    ap = self._read_ap(screen)
                    if ap is None or ap < 20:
                        self.log(f"{label}: AP={ap} <单次成本/读不出 → "
                                 f"fail-closed 收工(绝不碰購買AP框)")
                        close = self.find_cls(screen, UC.BTN_CLOSE_X,
                                              conf=_CLS_CONF)
                        self._set(phase_after)
                        if close is not None:
                            return action_click_box(
                                close, f"{label}: close popup, AP done")
                        return action_back(f"{label}: AP done → back")
                    self._sweep_rounds += 1
                    return action_click_box(ss, f"{label}: 掃蕩開始 (round "
                                                f"{self._sweep_rounds}, AP={ap})")
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
            # ⛔ money gate(2026-07-11 强化): 青辉石黑名单会漏检(購買AP框上
            # v13 全盲实锤) → 加结构闸: body 有 stepper/体力 = 购买框 → 取消。
            # 纯AP扫荡确认框 body 只有取消/确认/叉(t56 实锤), 误伤=安全重试。
            if self._dialog_is_purchase(screen):
                self.log("⛔ purchase-dialog structure in confirm body — "
                         "CANCEL (fail-closed)")
                self._set("close")
                return action_click_box(cancel_btn, "PURCHASE DIALOG — cancel!")
            self._swept += 1
            return action_click_box(conf_btn, f"{label}: confirm sweep (pure AP)")
        if conf_btn is not None and cancel_btn is None:
            # 掃蕩完成框 (确认键无取消): tooltip 坑 → 连点两次由 tick 自然完成
            # ⛔同样过结构闸: 取消键偶发漏检时購買AP框会伪装成完成框
            if self._dialog_is_purchase(screen):
                self.log("⛔ purchase-dialog structure (no-cancel frame) — back off")
                self._set("close")
                return action_back("PURCHASE DIALOG suspected — back!")
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

    # ── milestone (獎勵資訊里程碑领奖, 2026-07-11 落码; 0710 手动probe验证
    # 过链路: 奖励资讯475@0.96红点→popup→領取獎勵_黄89(一键领所有到达档,
    # +9M信用点实录)→獲得獎勵toast→灰90翻转→叉退回) ────────────────────

    def _milestone(self, screen: ScreenState) -> Dict[str, Any]:
        if self._phase_ticks > _PHASE_MAX:
            self._set("tasks")
            return action_wait(200, "milestone phase timeout → tasks")
        if self._milestone_done:
            self._set("tasks")
            return action_wait(200, "milestone claimed → tasks")
        got = self.find_cls(screen, UC.GOT_REWARD, conf=0.5)
        if got is not None:
            return action_click(*_POS_TOUCH_CONTINUE, "milestone: dismiss reward")
        claim_y = self.find_cls(screen, UC.CLAIM_REWARD_YELLOW, conf=0.5)
        if claim_y is not None:
            return action_click_box(claim_y, "milestone: 領取獎勵 (黄)")
        claim_g = self.find_cls(screen, UC.CLAIM_REWARD_GREY, conf=0.30)
        if claim_g is not None:
            self._milestone_done = True
            close = self.find_cls(screen, UC.BTN_CLOSE_X, conf=_CLS_CONF)
            if close is not None:
                self._set("tasks")
                return action_click_box(close, "milestone: 领完(灰) → close")
            self._set("tasks")
            return action_back("milestone: 领完(灰) → back")
        info = self.find_cls(screen, UC.EVENT_REWARD_INFO, conf=_WEAK_CONF)
        if info is not None:
            region = (info.x1 - 0.02, info.y1 - 0.05,
                      info.x2 + 0.04, info.y2 + 0.02)
            if self.dot_in_region(screen, region, dot_classes=("红点",)):
                return action_click_box(info, "milestone: 獎勵資訊 (red dot)")
            self.log("milestone: no red dot on 獎勵資訊 — skip")
            self._milestone_done = True
            self._set("tasks")
            return action_wait(200, "no milestone rewards")
        return action_wait(600, "milestone: waiting")

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
