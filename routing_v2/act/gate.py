# -*- coding: utf-8 -*-
"""三道闸 —— 全部集中在这里。**flow 里一行闸都不许写。**

老代码的病：同一道闸在 4 个 skill 各写一份、各有各的 bug，修好一处另外三处
照样犯（§A2）。这里三道闸各只有一个实现:

   金钱闸    —— 弹窗体内青辉石 / 未声明的花钱动作 / 青辉石 tab
   落地复验  —— 派发前拿**最新帧**确认目标还在，且落点还在它框里
   连发闸    —— 同一目标短时间内不重复点；重发只认"状态没变"不认计时器

是老代码里十几个 after-ack 的集中替代（§幽灵点击）。决策帧和 tap 落地时刻
   的屏幕不是同一张，我曾在 4 个 skill 各修一个 after-ack **没看出是同一件事**。
   正解是派发前 JIT 复验一次，一处抵十几处。

**绝不能为了复验去 ADB 抓帧**：ADB 4K 抓一张 1588ms，会给正在赛跑的那一发
   再加近 1.6 秒，比不复验还糟。复验只用 scrcpy 最新帧（0ms）+ YOLO（23ms）。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import routing_v2.act.money as money_rules
from routing_v2.act.action import Action
from routing_v2.percept.observe import Observation
from routing_v2.state import vocab as V

# 购买语境里的**导航白名单**：只有这些 cls 允许点，其余一律拦成人审。
#    为什么是白名单不是黑名单：黑名单漏一个花钱按钮 = 真花钱；白名单漏一个
#      导航按钮 = 流程卡住（无害）。**不对称的风险要用不对称的默认值。**
#    起因：组合包页上 flow 只是想「切到組合包 tab」（不花一分钱）也被拦成
#    halt  免費組合包这条链永远走不完（08-09 实锤）。
# **点了会弹确认框的键**：上一发是这些  眼前这个双键框是 bot 自己请求的，
#    不可能是「是否結束？」。这是比 last_solid 可靠得多的正向证据。
#    漏一个 = 那条链的確認被拦（流程卡住，无害）；多一个危险的 = 可能关游戏，
#    所以这里只放**明确会弹确认框的业务键**，绝不放通用导航键。
_EXPECT_DIALOG_AFTER = frozenset({
    V.SWEEP_START, V.TASK_START, V.SCHED_START, V.CRAFT_START,
    V.SHOP_BUY, V.SHOP_BUY_SELECTED, V.SHOP_SELECT_ALL,
    V.STORY_SKIP, V.STORY_MENU,            # 剧情「跳過」的确认
    V.CAFE_INVITE, V.SORTIE, V.QTY_MAX,
    V.CLAIM_ALL_YELLOW, V.CLAIM_ONCE_YELLOW, V.CLAIM_YELLOW,
})

_NAV_SAFE = frozenset({
    V.COMBO_PACK, V.CURRENCY, V.CURRENCY_SEL,          # 切 tab
    V.SHOP_TAB_CREDIT, V.SHOP_TAB_PYROXENE,
    V.ARENA_SHOP_TAB, V.ARENA_SHOP_TAB_SEL,            # 战术大赛商店 tab
    V.CLOSE_X, V.BACK, V.HOME, V.CANCEL,               # 关闭 / 退出 / 取消
})

# **「上一发点的是花钱键」= 眼前这个双键框就是购买确认框**（08-10 实锤）。
#   这一条补的是 `purchase_context` 的一个真盲区：組合包页点「購買」弹出的
#   「是否購買該商品？」框上 —— 顶栏以下**一个青辉石都没有**（价签是 CAD 或
#   「免費」）、tab 被框盖住检不到、也没有数量步进器  money.py 那四条判据
#   **全部返回 None**，闸对这一帧完全是瞎的。今天走免費包时它确实没报。
#   免費包本身无害，但同一个框结构在 **CAD$16.99 的真钱包**上一模一样：
#   只要哪条链在特別販售栏点了購買（点偏 / cls 抢错 / 货架刷新错位），
#   闸就会放行那一下「確認」= 真金白银。
#    反过来用 `_EXPECT_DIALOG_AFTER` 的同款正向证据：上一发是花钱键，
#     这个框必然是购买确认框，**没显式声明 money 就不许点**。
#   豁免只有一个：框上白纸黑字写着 `免费`（每日免費組合包）—— 正好把
#     今天走通的那条链放行，其余全拦。这也是「把灰/免費这类状态 cls
#     真正用起来」的一个落点。
_SPEND_TRIGGERS = frozenset({
    V.SHOP_BUY, V.SHOP_BUY_SELECTED, V.SHOP_BUY_PYROXENE,
    V.COMBO_PACK, V.COMBO_PACK_SEL,
})


# 落点相差在这个范围内 = 同一个目标（归一化坐标）。
# 0.02 ≈ 4K 屏上 77px，比按钮小、比框抖动大。
_SAME_TARGET = 0.02


@dataclass
class Verdict:
    ok: bool
    why: str = ""
    halt: bool = False              # 停**整条链**（只给金钱异常用）
    stop_flow: bool = False         # 只结束**当前 flow**，后面的照跑
    needs_human: bool = False


class Gate:
    def __init__(self, cfg: dict, log=None, on_money_step=None):
        self.cfg = cfg or {}
        self._log = log or (lambda m: print(m, flush=True))
        # 人审回调：返回 True 才放行金钱步。None = 一律拒绝（最安全的默认）。
        self._ask = on_money_step
        self._last: Optional[Tuple[float, float, str]] = None   # (x, y, cls)
        self._last_ts = 0.0
        self._fires = 0
        # 推进闸的待兑现契约（见 arm/advance）。None = 没有在等任何东西。
        self._pending: Optional[dict] = None
        self.stats = {"pass": 0, "money": 0, "jit_drop": 0, "dedup": 0,
                      "expect_hold": 0, "expect_timeout": 0}

    # ══  金钱闸 ═══════════════════════════════════════════════════════
    def money(self, act: Action, obs: Observation,
              last_solid: Optional[str] = None) -> Verdict:
        """判据全部在 `act/money.py`（唯一一份）。这里只负责怎么处置。"""
        safety = self.cfg.get("safety", {})

        # 系统框保护：**永远不许在退出框上点確認**（確認=退出游戏）。
        #    放在闸里是因为八个 flow 都覆写了 on_confirm_dialog，
        #    护栏放基类会被整体绕过。
        #
        # 判据 = **「这个框是不是我自己点出来的」**（08-09 审查实锤后重写）:
        #    历史上试过两版都不成立 ——
        #      「顶栏没货币」：弹窗压暗顶栏时游戏内框也没货币  误拦 340AP 扫荡
        #      「last_solid 不是大厅系」：**Machine 每次新建时 last_solid
        #        恒为 'unknown'**（它只在页面切换时才写），step 每次新进程、
        #        runner 每条 flow 都从这个窗口起步  实测 `ok=True` **放行点
        #        確認 = 关掉游戏**，而离线测试因为没传 last_solid 给了假保证。
        #     真正可靠的是**意图**：游戏内的确认框都是 bot **刚点了某个键**
        #      弹出来的（扫荡开始/任务开始/課程表開始/購買/跳過…）；
        #      而退出框是"没人请求它"就冒出来的（大厅误触返回键）。
        #    所以 `last_solid` 降为**辅助**：它明确是大厅时更可疑，但不再是唯一。
        if act.is_tap and act.target_cls == V.CONFIRM:
            prev = self._last[2] if self._last else None
            requested = prev in _EXPECT_DIALOG_AFTER
            if not requested:
                sysd = money_rules.system_dialog(obs, last_solid, strict=True)
                if sysd is not None:
                    return Verdict(
                        False,
                        f"拦下：{sysd}；且上一发点的是 `{prev or '(无)'}`"
                        f" — **没有任何动作会弹出这个框**  只允许点取消")

        # 购买确认框盲区（见 _SPEND_TRIGGERS 那段）：上一发点的是花钱键
        #     眼前这个双键框就是「是否購買該商品？」，没声明 money 不许点。
        if (act.is_tap and act.target_cls == V.CONFIRM and not act.money
                and (self._last[2] if self._last else None) in _SPEND_TRIGGERS
                and obs.has(V.CANCEL, money_rules.CONF)
                and not obs.has(V.FREE, 0.30)):
            return Verdict(False,
                           f"上一发点的是花钱键 `{self._last[2]}`，眼前是购买"
                           f"确认框且**没写着「免費」** — 这一下確認就是成交，"
                           f"而动作没声明 money  拦下交人审",
                           halt=True, needs_human=True)

        ctxt = money_rules.purchase_context(obs)
        if ctxt is not None and act.is_tap and not act.money:
            # 闸要**贴着花钱按钮**，不是拦死整页（08-09 实测）：
            #    组合包页上 flow 想「切到組合包 tab」（导航，不花一分钱）也被
            #    拦成 halt  免費組合包这条链**永远走不完**。
            #     只拦目标是**成交类按钮**的点击；切 tab / 关闭 / 返回放行。
            #    成交类必须含 `确认键`（购买框的最后一下就是它）。
            if act.target_cls not in _NAV_SAFE:
                return Verdict(False, f"{ctxt} — 停，交人逐帧审",
                               halt=True, needs_human=True)
            self._log(f"    [gate] 在购买语境里（{ctxt}），但这一发点的是导航"
                      f"白名单里的 `{act.target_cls}`（不花钱） 放行")
        # 青辉石商店 tab 选中 = 站在花青辉石的位置上，只许退出
        # 2026-08-12 修 latent bug：注释一直写「只许退出」，代码却拦掉
        #    **所有**非 money 的 tap —— 而退出动作（返回键/叉叉/取消）和
        #    `on_shop_pyroxene_tab` 里那句「切回信用点 tab」**全都是 tap**
        #     真触发时 flow 会被自己的闸锁死在青辉石货架上，直到
        #      `max_minutes_per_flow` 超时才脱身。
        #    之所以一直没暴露：`青辉石商店_已选择` train=0/val=0、990 帧 0 检出，
        #    这条判据恒 False，整段从没跑过（[[cls_ownership_audit]] 那族
        #    「守卫悄悄死掉」）。 复用 `_NAV_SAFE`，让注释和代码一致。
        if (money_rules.on_pyroxene_tab(obs) and act.is_tap and not act.money
                and act.target_cls not in _NAV_SAFE):
            return Verdict(False, "青辉石商店 tab 选中 — 只允许退出/切 tab 动作",
                           halt=False)

        if not act.money:
            return Verdict(True)

        # 显式声明的金钱步
        if act.spend in ("青辉石", "青辉石(premium)"):
            return Verdict(False, "铁律：bot 绝不花青辉石", halt=True)
        if not safety.get("forbid_premium_currency", True):
            return Verdict(False, "安全配置被改动过（forbid_premium_currency=False）"
                                  " — 拒绝一切金钱步", halt=True)
        if safety.get("money_step_needs_human", True):
            if self._ask is None:
                return Verdict(False, f"金钱步「{act.reason}」需人审，但没接人审通道",
                               needs_human=True)
            if not self._ask(act, obs):
                return Verdict(False, f"金钱步「{act.reason}」人审未放行",
                               needs_human=True)
        self.stats["money"] += 1
        return Verdict(True, f"金钱步已放行: {act.reason} (花 {act.spend or '?'})")

    # ══  落地复验（JIT）═══════════════════════════════════════════════
    def jit(self, act: Action, obs: Observation,
            fresh: Callable[[], Optional[Observation]]) -> Verdict:
        """派发前用**最新帧**确认目标还在。

        `fresh()` 由 runner 提供：返回比 obs 更新的 Observation，或 None
        （没有更新的帧 = 决策帧就是最新的 = 无需复验）。
        """
        if not act.is_tap or act.require is None:
            return Verdict(True)
        f = fresh()
        if f is None or f.seq == obs.seq:
            return Verdict(True, "决策帧即最新帧，无需复验")

        # 目标 cls 还在不在？—— 活动入口轮播 3.00s 就靠这一条拦住
        still = f.all(act.require, act.require_conf)
        if not still:
            self.stats["jit_drop"] += 1
            return Verdict(False, f"丢弃: '{act.require}' 在新帧已消失"
                                  f"（轮播已换页 / 页面已跳走）")

        # 锚点框还在原地不在？—— 「附近有同名」≠「还是那一个」（08-01 收紧）。
        # 老版复验只查"这个 cls 还在屏上"，于是列表滚动一格后，落点其实已经
        # 压在隔壁行上，复验照样通过。
        # 判的是**锚点框有没有移动**，不是"落点在不在框内"：有的落点是**故意**
        #    打到框外的（HUB 活动入口 +0.075 打卡片本体，框标的是倒计时气泡）。
        if act.anchor is not None:
            ax, ay = act.anchor
            tol = act.anchor_tol or 0.02
            near = min(still, key=lambda b: (b.cx - ax) ** 2 + (b.cy - ay) ** 2)
            drift = ((near.cx - ax) ** 2 + (near.cy - ay) ** 2) ** 0.5
            if drift > tol:
                self.stats["jit_drop"] += 1
                # 2026-08-12 最隐蔽的一次"选项打架"：**一发被丢弃后，
                #    flow 转头点了相反的按钮**。战术大赛商店实录:
                #      ↗ 等到了 `确认键`（确认框弹出来了）
                #      丢弃 '确认键' 锚点 y 0.9850.842 位移 0.144 > 容差 0.029
                #      丢弃 '取消键' 锚点 y 0.8450.803 位移 0.042 > 容差 0.028
                #       tap **取消键**「模态框唯一有效出口」    购买被取消
                #    —— 框正处在**弹入动画**里，位置每帧都在动，JIT 每帧都丢；
                #    flow 见"点不动"就走了模态框的兜底出口 = 取消。
                #    **目标正在移动 ≠ 目标不可点**，只是"现在还不能点"。
                #     丢弃时按住几帧等它停稳，期间**一发新 tap 都不许出去**，
                #      免得被兜底分支抢走去点相反的东西。
                #      只按 6 帧（≈0.3s）：够动画停稳，又不至于在真的换页时卡住。
                if not self._pending:
                    self._pending = {"expect": (), "expect_gone": (),
                                     "cls": act.require, "reason": "等目标停稳",
                                     "n": 2, "loose": True, "settle": True}
                return Verdict(False,
                               f"丢弃: '{act.require}' 锚点从 ({ax:.3f},{ay:.3f}) "
                               f"漂到 ({near.cx:.3f},{near.cy:.3f})，位移 {drift:.3f} "
                               f"> 容差 {tol:.3f} — 目标在动，先按住等它停稳")
            # 落点**跟着最新帧的 cls 框走**（2026-08-12 用户实测:「制造…
            #    真的就是进去看了一眼马上就回大厅了」「**依旧没点**」「就是卡死
            #    呆在这」）。实帧诊断：`快速制造` 框心 cx=**0.697**（x 0.646~0.747），
            #    而 flow 发出去的落点是 **0.768 —— 已经在框外**，正好落在
            #    「快速製造」和「一次領取」之间的空隙上  点了毫无反应。
            #    原因是**决策帧是动画帧**，那一帧的框位置本身就偏；而 JIT 原来
            #    只验"锚点有没有移动"，**验完并不修正落点**，于是偏掉的坐标
            #    照原样发出去。
            #     复验通过后用**新帧那个框的中心**重算落点，并原样保留决策时
            #      的偏移量（HUB 活动入口 +0.075 那种**故意**打到框外的还得留着）。
            #    这不是硬编码坐标 —— 恰恰相反，是让落点**更严格地 based on cls**：
            #      用的是落地前最后一刻的检出框，而不是几十毫秒前那一帧的。
            dx, dy = act.x - ax, act.y - ay
            nx, ny = near.cx + dx, near.cy + dy
            if abs(nx - act.x) > 1e-6 or abs(ny - act.y) > 1e-6:
                act.x, act.y = nx, ny
                act.anchor = (near.cx, near.cy)
                return Verdict(True, f"JIT 复验通过（落点按最新帧校正到 "
                                     f"{nx:.3f},{ny:.3f}）")
        return Verdict(True, "JIT 复验通过")

    # ══  连发闸 ═══════════════════════════════════════════════════════
    def dedup(self, act: Action, page_changed: bool,
              frames_in_page: int, retry_frames: int) -> Verdict:
        """同一落点不重复点；重发只认"状态连续 N 帧没变"。

        **冷却必须显式记时间戳**：裸 `since()` 首次返回 0.0，会被当成
           "冷却已过"直接放行（连发族三根因之一）。
        **别拿 reason 措辞当控制 API**：老代码按 reason 子串豁免稳定门，
           于是往 reason 里塞个"確認键"就能绕过闸 —— 近失 30 青辉石那次
           就是这么来的。这里只看**落点 + cls**，不看文案。
        """
        # 按键**也要**过连发闸。2026-08-08 引入"没有退出控件就按系统返回键"
        #    之后，如果不管按键，返回键会被每帧重发  一口气连退好几层，
        #    把 bot 弹到谁也不知道的地方。按键的"落点"就是它的 keycode。
        if act.kind == "key":
            act = Action(kind="tap", x=-1.0, y=-1.0,
                         target_cls=f"KEY:{act.keycode}", reason=act.reason)
        if not act.is_tap:
            return Verdict(True)
        # "同一个落点"必须按**距离**判，不能按坐标相等判。
        #    2026-08-08 live 实锤：`邮件箱` 的框每帧抖 ±0.001（0.892↔0.893、
        #    0.045↔0.054），用 `round(x,3)` 当 key  每一帧都被当成**新目标**
        #     闸形同虚设，一个大厅入口连点 12 次。
        #    这是"连发族"的第 4 种形态：前三种是 after-ack 挂错地方 / 判定带宽
        #    不一致 / 没有 after-ack 的重发，这一种是**同一性判据太严**。
        same = (self._last is not None
                and self._last[2] == act.target_cls
                and abs(self._last[0] - act.x) <= _SAME_TARGET
                and abs(self._last[1] - act.y) <= _SAME_TARGET)
        if not same:
            self._last = (act.x, act.y, act.target_cls)
            self._last_ts, self._fires = time.time(), 1
            return Verdict(True)
        # 同一落点：页面变了 = 上一发生效了，这是新的一发
        if page_changed:
            self._last_ts, self._fires = time.time(), 1
            return Verdict(True)
        # 页面没变：只有"待够了 retry_frames 帧"才允许补一发
        if frames_in_page >= retry_frames:
            self._fires += 1
            self._last_ts = time.time()
            if self._fires > 6:
                # 只结束当前 flow，别停整条链 —— 一条支线走不通不该让今天
                #    剩下的活全不干（老代码就是这么"一个 skill 卡死全盘皆输"的）。
                _n = self._fires
                self._fires = 0
                return Verdict(False, f"同一落点已重发 {_n} 次仍无变化"
                                      f" — 这条 flow 走不通，收工换下一个",
                               stop_flow=True)
            return Verdict(True, f"↻ 重发({frames_in_page}帧未变) {act.target_cls}")
        self.stats["dedup"] += 1
        return Verdict(False, "")      # 静默吞掉，不刷屏

    # ══  推进闸（1 step ahead）═════════════════════════════════════════
    def arm(self, act: Action) -> None:
        """一发 tap **真的发出去之后**调 —— 记下推进契约。

        和 counter/once/post 同一条纪律：**数事实，不数意图**。被闸吞掉的
           决策不许 arm，否则闸会为一发从没发出去的点击等下去。

        **默认契约是全局的，不用每处手工声明**（2026-08-12 用户点名:
           「难道不是全局通用吗，要一个个加？」——对，一处处加必然漏）:

             没显式声明  `expect_gone = (刚点的那个 cls,)`

           理由：点了什么，它就该消失或变样。实测覆盖面很广 ——
             購買被确认框盖住 / 邀请卷被面板盖住 / 设施入口换页 /
             摸头气泡消失 / `全部选择`变成`已全部选择`（cls 名变了＝原 cls 没了）/
             `移动至2号点`变成`移动至一号店` / 確認键框关掉
           少数不成立的（加号、翻页箭头这种"点完自己还在"的）走**宽松满足**
           + 超时兜底，不会卡住。

        两级严格度，**由 Action 的性质自动决定，不靠每处手写记得加**:
           · 严格档 = 显式声明了 expect 的 / **或 `money=True` 的**
              不吃"画面动了"这种宽松放行，必须目标真的消失/期待的真的出现，
               否则一直按住到完整 retry_frames 超时。
             金钱步必须严格：组合包页三个「購買」并排，另两个是真钱包，
               误判"没生效"而重发的落点偏一列就是 CAD$16.99。
           · 宽松档 = 其余全部  见 `advance()` 里的 `loose` 分支
        """
        # 滑动也要节流（2026-08-12 用户点名:「还是**猛滑**，也不是滑一下让
        #    头像模型检测一次」「应该是进去让头像模型扫，有就邀请，
        #    **没才滑一次然后再扫**」）。
        #    原来这里第一行就 `if not act.is_tap: return`  swipe **完全不设契约**
        #      （runner 的 swipe 分支也没调 arm） 滑动毫无节流，一帧一发连着滑，
        #      列表还在惯性滚动时模型根本没机会扫干净 —— 目标就这么被滑过去了。
        #     滑完按住固定帧数，逼出「滑一次  扫一次」的节奏。
        #      不能用 cls 契约：滑动没有 target_cls，也没有"该出现/该消失"的东西，
        #        能约束的只有**时间**（等画面停稳）。
        if act.kind == "swipe":
            self._pending = {"expect": (), "expect_gone": (), "cls": "(swipe)",
                             "reason": act.reason, "n": 0, "loose": True,
                             "settle": True}
            return
        if not act.is_tap:
            return
        if act.expect or act.expect_gone:
            self._pending = {"expect": act.expect, "expect_gone": act.expect_gone,
                             "cls": act.target_cls, "reason": act.reason,
                             "n": 0, "loose": False}
            return
        # **点了"会弹确认框"的键  契约就是"确认框出现"**（2026-08-12）。
        #    `_EXPECT_DIALOG_AFTER` 这张表本来就登记着这类键，直接拿来用，
        #    不用在每个 flow 里一处处手写。
        #    为什么默认契约在这里**必然误判**：确认框一弹出来就把按钮盖住了
        #       `expect_gone=(自己,)` 立刻"兑现"放行  flow 下一帧决策时
        #      确认框还没被识别成 overlay，就一路走到退出分支 ——
        #      **确认框没人点，这一发等于白点**。08-12 一晚上同形抓到 4 处：
        #        cafe 开邀请券 / craft 快速製造 / shop 切大赛 tab /
        #        **战术大赛商店「選擇購買」 弹確認框  直接点返回键跑了**
        #        （用户:「体力饮料选了购买的时候…然后就有跑了」，能量饮料没买成）。
        #     走严格档（loose=False）：确认框真出现才算这一发生效。
        if act.target_cls in _EXPECT_DIALOG_AFTER:
            self._pending = {"expect": (V.CONFIRM, V.CANCEL), "expect_gone": (),
                             "cls": act.target_cls, "reason": act.reason,
                             "n": 0, "loose": False}
            return
        # 默认契约：目标 cls 该消失。没有 cls 支撑的 tap_at 不设契约
        #    （连它点的是什么都不知道，无从判断"变没变"）。
        if act.target_cls and act.target_cls != "(no-cls)":
            self._pending = {"expect": (), "expect_gone": (act.target_cls,),
                             "cls": act.target_cls, "reason": act.reason,
                             "n": 0, "loose": not act.money}

    def advance(self, act: Action, obs: Observation, *,
                page_changed: bool, retry_frames: int) -> Verdict:
        """上一发声明了「点完该出现什么」，就等它 —— 期间**一发新 tap 都不许**。

        2026-08-12 用户点名的三条，这一道闸一起治:
          「点了一次就去识别下一个要点的 cls，而不是看到什么点什么」
              契约在 Action 上，不在事后猜。
          「不要有 wait time，完全取决于有没有下一步要点的东西」
              **零 sleep**：每帧检查一次，满足就当帧放行，纯帧驱动。
          「每一步点击的 cls 选项不能打架，一下子把接下来的统统点了一遍」
              拦的是**所有** tap，不只是同落点那一发。屏上同时躺着好几个
               可点 cls 时（组合包页三个「購買」并排就是活例子），
               flow 每帧照样决策，但在契约兑现前一发都出不去。

        为什么 `page_changed` 直接作废契约而不是当"未满足"：页面/overlay 变了
           说明画面确实动了 —— 要么这一发生效了，要么冒出了预期外的局面
           （剧情打断、加载、弹窗）。两种情况下**继续等原来那个 cls 都是错的**，
           尤其后者会把 interrupt 处理一起锁死。
        """
        p = self._pending
        if p is None:
            return Verdict(True)
        if page_changed:
            self._pending = None
            return Verdict(True, f"↗ 页面已变 — 推进契约(`{p['cls']}`)作废")
        hit = next((c for c in p["expect"] if obs.has(c, 0.40)), None)
        if hit is not None:
            self._pending = None
            return Verdict(True, f"↗ 等到了 `{hit}` — `{p['cls']}` 这一发生效，推进")
        if p["expect_gone"] and not any(obs.has(c, 0.40) for c in p["expect_gone"]):
            # **"消失"还不够，必须伴随新东西出现**（2026-08-12 全链审计后加）。
            #    默认契约 `expect_gone=(自己,)` 的误判源就在这里：点"打开面板"
            #    的键，面板一弹出来**把按钮盖住**，目标也"消失"了  契约立刻
            #    兑现放行，可页面身份还要 `confirm_frames` 帧才切过去  那几帧
            #    handler 命中不了任何分支，一路落到收尾  把刚开的面板关掉。
            #    （一晚上 5 处同形 bug 全长这样：cafe 邀请券 / craft 快速製造 /
            #      shop 切大赛 tab / 战术商店選擇購買 / arena_done 自取消。）
            #    **真开了面板  必然带来新的 cls**；只是被盖住则不会。
            #       宽松档要求「目标没了 **且** 屏上多了以前没有的 cls」。
            #      严格档（显式契约/金钱步）不吃这条：它们自己指定了要等什么。
            cur = frozenset(b.cls for b in obs.boxes)
            base = p.get("sig0")
            if not p["loose"] or base is None or (cur - base):
                self._pending = None
                return Verdict(True, f"↗ `{'/'.join(p['expect_gone'])}` 已消失"
                                     f" — `{p['cls']}` 这一发生效，推进")
        if p.get("sig0") is None:
            p["sig0"] = frozenset(b.cls for b in obs.boxes)
        p["n"] += 1
        # 滑动契约：没有 cls 可等，只能等画面停稳。按住 8 帧（≈0.4s）——
        #   够模型把停稳后的列表扫一遍，也就实现了「滑一次  扫一次」。
        if p.get("settle"):
            if p["n"] >= 8:
                self._pending = None
                return Verdict(True)
            self.stats["expect_hold"] += 1
            return Verdict(False, "")
        # 2026-08-12 第二次 live 打脸：这里原来还有一条「屏上 cls 组成一变
        #    就放行」的宽松出口，当天就被抓到漏了 ——「同时点了三一和千年」:
        #    点「三一」之后按钮**还在屏上**（只是变成选中态），但 cls 组成变了
        #     宽松放行  下一帧立刻点「千年」。
        #    **「画面动了」和「这一发生效了」是两回事**，拿前者当后者用，
        #    正好放走了同族按钮之间的连点——而那是这道闸存在的理由。
        #     出口删掉。宽松档和严格档的**满足条件现在完全一致**
        #      （目标消失 / 期待出现 / 页面变），只在**超时长度**上有区别：
        #      宽松 retry//6（约 11 帧、半秒），严格走完整 retry_frames。
        #      「点完自己还在」的控件（加号、翻页箭头）靠那半秒超时兜底，
        #      代价可以接受，换来的是同屏同族按钮不会被连着点掉。
        lim = max(1, retry_frames if not p["loose"] else max(3, retry_frames // 6))
        if p["n"] >= lim:
            self._pending = None
            self.stats["expect_timeout"] += 1
            if p["loose"]:
                return Verdict(True)          # 默认契约超时属常态，别刷屏
            return Verdict(True, f"等 `{'/'.join(p['expect'] + p['expect_gone'])}`"
                                 f" 等了 {p['n']} 帧没等到 — 契约超时作废，"
                                 f"退回连发闸兜底（上一发多半丢了）")
        self.stats["expect_hold"] += 1
        return Verdict(False, "")      # 静默按住，不刷屏

    # ══ 汇总 ═══════════════════════════════════════════════════════════
    def allow(self, act: Action, obs: Observation, *,
              fresh: Callable[[], Optional[Observation]],
              page_changed: bool, frames_in_page: int,
              retry_frames: int = 25,
              last_solid: Optional[str] = None) -> Verdict:
        # **必须短路求值**（2026-08-12 修）：原来写的是 `for v in (a(), b(), c())`
        #    —— 元组在进循环前就把三道闸**全部执行**了，于是 money 闸拦下的那一发
        #    照样让 dedup 记下 `_last`（等于记了一发从没发出去的点击 = 「数意图
        #    不数事实」在闸自己身上复发），也照样让 advance 的等待帧数白涨一格。
        #     包成 lambda 逐个求值，前一道拦下就不跑后面的。
        for _fn in (lambda: self.money(act, obs, last_solid),
                    lambda: self.advance(act, obs, page_changed=page_changed,
                                         retry_frames=retry_frames),
                    lambda: self.dedup(act, page_changed, frames_in_page,
                                       retry_frames),
                    lambda: self.jit(act, obs, fresh)):
            v = _fn()
            if not v.ok:
                if v.why:
                    self._log(f"    [gate] {v.why}")
                return v
            if v.why:
                self._log(f"    [gate] {v.why}")
        self.stats["pass"] += 1
        return Verdict(True)
