# -*- coding: utf-8 -*-
"""票券扫荡型的公共骨架 —— 悬赏通缉 / 学院交流会共用。

流程（全部 live 验过）:
  大厅  任务大厅  tile  **分支选择（动态扫 cls）**  关卡列表（滑到底）
   入场  任務資訊  MAX  扫荡开始  確認  结算  回列表

票数铁律:
  · **屏上票数是唯一权威**，台账只在读不出时兜底
    （老代码用台账当上限，7/7 满票只派 4 次就触顶，浪费 3 张）
  · **读不出 = fail-closed 不出击**；但 fail-closed 只挡得住 None，
    **挡不住"读大"** —— 零票被读成 9 票那次就是这么弃掉 6 张票的。
    所以还要看"扫荡开始"变没变灰 + 結算後票数有没有真的减少。
  · **绝不买票**（票不够就收工）。

滑动铁律（用户 2026-08-07 纠正）:
  别写死"滑 N 次"。滑到**最低那个入场键的 y 不再变化**就是到底了。
"""
from __future__ import annotations

from typing import Optional

from routing_v2.act.action import Action, swipe, tap_box, wait
from routing_v2.flow import nav
from routing_v2.flow.base import ExitMixin, Flow, Outcome
from routing_v2.flow.battle import BattleMixin
from routing_v2.percept import read as R
from routing_v2.percept.observe import Observation
from routing_v2.state import vocab as V
from routing_v2.state.machine import StateView

STAGE_PANEL = (0.50, 0.10, 1.0, 0.95)
CONFIRM_BAND = (0.20, 0.55, 0.95, 0.92)


class TicketSweepFlow(BattleMixin, ExitMixin, Flow):
    entry_page = "task_hall"
    ticket_cls: str = ""
    hub_tile: str = ""
    branch_cls: tuple = ()
    branch_page: str = ""
    stage_page: str = ""
    costs_ap: bool = False
    ap_per_sweep: int = 15

    def setup(self) -> None:
        # 没有 swipes / last_low_y：关卡列表**一次都不滑**（游戏自己会把
        #   最高那一关归位到可见处），那两个字段随滑动逻辑一起删了。
        self.state.update(sweeps=0, tickets0=None, tickets=None,
                          branch_i=0, branch_name="", branch_tix0=None,
                          planned=0)
        self._register_pages()

    # ── 票配额（用户点名：「票用在哪个地区，还是一个地区用几张」）──────
    def _plan(self) -> dict:
        """{分支名: 分配几张}。空 = 老行为（按顺序打到票光）。"""
        p = self.cfg.get("ticket_plan") or {}
        return {str(k): int(v) for k, v in p.items() if str(v).lstrip("-").isdigit()}

    def _quota(self, branch: str):
        """这个分支分配了几张票；没配到就是 None（不限）。"""
        q = self._plan().get(branch)
        return q if (q is not None and q > 0) else None

    # ── 进场 ────────────────────────────────────────────────────────────
    def on_lobby(self, obs, st):
        return nav.enter(obs, V.NAV_TASKS, "任务大厅")

    def on_task_hall(self, obs, st):
        tile = obs.find(self.hub_tile, 0.40)
        if tile is None:
            if self.stalled(st, 120):
                return self.finish(Outcome.UNKNOWN,
                                   f"任务大厅里没检出 `{self.hub_tile}` tile")
            return wait("任务大厅: 等 tile")
        # 各玩法的红点**只在任务大厅可见**（大厅那个入口红点不是工作信号）。
        #   但红点缺失 ≠ 没活干（红点自己也会漏检），所以只当提示，不当闸。
        if not nav.dot_on(obs, tile):
            self.note_lines.append(f"{self.hub_tile} tile 上没红点（仅提示，仍进去看）")
        return tap_box(tile, f"进入 {self.hub_tile}")

    # ── 分支选择（动态：屏上有哪些就是哪些）─────────────────────────
    def _branch_order(self, obs: Observation):
        """配置里点的顺序 ∩ 屏上真的看得到的。**不写死清单。**"""
        # 只配了 ticket_plan 没配 branches 时，plan 的 key 就是想打的分支
        want = list(self.cfg.get("branches") or self.cfg.get("academies")
                    or self._plan().keys())
        seen = {b.cls: b for b in obs.all(list(self.branch_cls), 0.40,
                                          region=STAGE_PANEL)}
        if not seen:
            return []
        ordered = [seen[n] for n in want if n in seen]
        # 配置里没写到的，按屏上从上到下补在后面（用户没排除就是可以打）
        rest = [b for n, b in sorted(seen.items(), key=lambda kv: kv[1].cy)
                if n not in want]
        return ordered + (rest if not want else [])

    def _on_branch(self, obs, st):
        # 真回到分支页了，换分支这件事才算做成（数事实不数意图）
        self.state["switching"] = False
        if obs.has([V.STAGE_ENTER, V.STAGE_ENTER_LOCKED], 0.45, region=STAGE_PANEL):
            return None                      # 已经在关卡列表了
        opts = self._branch_order(obs)
        if not opts:
            if self.stalled(st, 90):
                return self.finish(Outcome.UNKNOWN,
                                   f"分支选择页一个分支 cls 都没检出"
                                   f"（期望 {list(self.branch_cls)}）")
            return wait("等分支 cls")
        # 2026-08-12 用户 live 抓到「同时点了三一和千年」。根因：
        #    `opts` 只收**这一帧检出到的**分支，而 `branch_i` 是**位置索引** ——
        #    某帧「千年」漏检  opts=[三一,格黑娜]  opts[0] 是三一  点三一；
        #    下一帧千年又检出来了  opts[0] 变回千年  再点千年。
        #    **同一个 i 在不同帧指向不同分支**  一口气点掉两个学院。
        #    bounty 用同一个基类，同样中招（用户当场就怀疑到了）。
        #     索引锚在**分支名的稳定序列**上，不锚在会抖动的检出列表上；
        #      轮到的那个这一帧没检出就**等它**，绝不顺位跳到别人身上。
        order = self.state.get("branch_names")
        if not order:
            order = [b.cls for b in opts]
            self.state["branch_names"] = order
        i = self.state["branch_i"]
        if i >= len(order):
            return self.finish(Outcome.CLEAN, f"配置的 {len(order)} 个分支都跑过了")
        name = order[i]
        pick = next((b for b in opts if b.cls == name), None)
        if pick is None:
            return wait(f"等分支 `{name}` 的 cls（这一帧没检出，不跳到别的分支）")
        q = self._quota(pick.cls)
        self.log(f"选分支 [{i+1}/{len(opts)}] {pick.cls}"
                 f"（屏上共 {len(opts)} 个可选"
                 f"{f'，配额 {q} 张' if q else ''}）")
        # 分支身份/票基线只在 tap **真发出后**才落（§数事实不数意图）。
        return tap_box(pick, f"选分支 {pick.cls}",
                       post=lambda n=pick.cls: self.state.update(
                           branch_name=n, branch_tix0=None))

    # ── 票数 ────────────────────────────────────────────────────────────
    def _tickets(self, obs: Observation) -> Optional[int]:
        t = obs.find(self.ticket_cls, 0.30)
        if t is None:
            return None
        return R.read_ticket(obs, t)          # 带越界校验，挡"读大"

    # ── 关卡列表 ────────────────────────────────────────────────────────
    def _on_stage_list(self, obs, st):
        # 换分支**必须真的走回去**，不能只把索引加一（2026-08-12 体外复现）:
        #    原来 `_next_branch_or_done` 只 `branch_i += 1` 然后 return wait，
        #    而 wait 不产生任何动作、页面身份不变，于是**同一个判据下一 tick
        #    原样再成立** —— 实测 tick61/62/63 连着把 branch_i 推到 3 收工，
        #    日志连打三行「换下一个分支」，屏幕从头到尾停在第一个学院的关卡列表，
        #    千年/格黑娜一张票没打就报了结果。
        #    （用户 08-12:「学园交流会也有黄点说明之前也没打好」的另一半原因。）
        #     latch + 真导航：置 `switching` 后先退回分支页，`_on_branch` 里
        #      真到了才清掉 latch，期间判据再成立也不许重复自增。
        if self.state.get("switching"):
            return self.exit_step(obs, prefer_close=False) \
                or wait("换分支: 等退回分支页的控件")
        tix = self._tickets(obs)
        if tix is not None:
            if self.state["tickets0"] is None:
                self.state["tickets0"] = tix
                self.log(f"票数 {tix}（屏上读出，唯一权威）")
            self.state["tickets"] = tix
            # 分支配额：用**票数差**算已用几张（数事实，不用计数器）。
            #   基线在进这个分支的列表页第一次读到票时落 —— 纯观测，可在 decide 期记。
            bn = str(self.state.get("branch_name") or "")
            if self.state.get("branch_tix0") is None:
                self.state["branch_tix0"] = tix
            q = self._quota(bn)
            if q is None and self._plan() and not bn:
                # 配了配额但认不出当前分支  只提示不强限：票不是钱，最坏是
                # 全打在一个分支（= 老行为），而按错的身份限流会浪费票。
                if self.once("plan_no_branch"):
                    self.log("配了票配额，但没经过分支页/分支名未知  本轮按老行为打")
            if q is not None:
                used = self.state["branch_tix0"] - tix
                if used >= q:
                    self.state["planned"] += used
                    return self._next_branch_or_done(
                        f"{bn} 的配额 {q} 张用完了（这个分支用了 {used}）")
            keep = int(self.cfg.get("keep_tickets", 0) or 0) \
                if self.cfg.get("use_tickets") == "keep_n" else 0
            if tix <= keep:
                return self._next_branch_or_done(
                    f"票用完了（剩 {tix}，保留线 {keep}）— 绝不买票",
                    terminal=True)

        enters = obs.all(V.STAGE_ENTER, 0.45, region=STAGE_PANEL)
        if not enters:
            # "没关可打"必须连续 60 tick 成立（§A3 内容层，见 flow/base.hold）
            if not self.hold("no_enter", 60):
                return wait("这一帧没看到入场键 — 连续确认中，别被过渡帧骗了")
            if obs.has(V.STAGE_ENTER_LOCKED, 0.45, region=STAGE_PANEL):
                return self._next_branch_or_done("这个分支只有锁着的关")
            if tix is None:
                # 票数读不出  fail-closed 不出击（金钱铁律 #3）
                return self._next_branch_or_done(
                    "票数读不出  fail-closed，不出击", terminal=True)
            return self._next_branch_or_done("列表里没有入场键")

        # 滑到底：最低那个入场键的 y 不再变化 = 到底了。**不数次数。**
        # 2026-08-12 用户拍板：**根本不用滑**。原话:
        #    「滑到底了的，实际上**都可以不用滑**，因为最下面就是第 10 关，
        #      如果新的一关游戏会自动队列出来，这就和活动一样，活动剧情以及
        #      Q1-12 的时候不用瞎滑动栏目。**这个适用于整个游戏所有，除了商店。**」
        #     游戏自己会把最新/最高那一关归位到列表可见处，屏上最低那个入场键
        #      **就是**最高难度。之前那段「滑到底」的代码才是第 9 关的真凶：
        #      滑动把列表滑过头/滑歪，反而离开了正确位置
        #      （用户实测 bounty 扫到了第 9 关而不是最后一关）。
        #    我第一版把滑动次数上限从 8 提到 25，方向完全错了 —— 那是让它
        #      **滑得更多**，而正解是**一次都不滑**。memory 早记过这条
        #      「用户纠正死滑动路由（新活动游戏会自动归位下一关）」，我又犯了一次。
        #    例外与"什么时候才真的该滑"（用户 2026-08-12 补全的完整规则）:
        #      · **商店 / 活动商店**：进去**永远停在最上面**  要看下面的货架必须滑。
        #        （这两处的 swipe 在 `event_shop.py` / `facilities.py`，不在本 flow。）
        #      · **关卡列表**：只有在「**已经清到最后面了、却要回头打前面的关**」时
        #        才需要往回滑；触发条件是 —— **目标关号没找到 / 屏上所见的关号比
        #        目标大 / 关号被截断**。
        #         也就是说，滑动必须由「**读到的关号和目标对不上**」驱动，
        #          而不是由"我猜列表还没到底"驱动。
        #      · 用户原话:「说实话都是刷难的，难的给的奖励好一些，
        #        **应该不会有人扫荡简单的**」 `difficulty="highest"` 是常态，
        #        而它恰好**永远不需要滑**（屏上最低那个就是最难）。
        #      将来做 `fixed_stage`（指定打第几关）时才需要实现"按关号回滑"，
        #        届时关号读法见 memory：关号 `2-3` **整列读**可行、单裁必 None。
        #    **滑歪了怎么复位 —— 退出去重进，别接着滑**（用户 2026-08-12）:
        #      原话「即使你往回滑动了，**退出去重新进活动又会给你把该账号打到的
        #      最后关卡显示给你**」。 列表位置是游戏自己维护的，重进即归位。
        #      这也是为什么"滑到底"这套逻辑从根上就是多余的：**想要的位置本来
        #      就是默认位置**，任何滑动都只会让它偏离，而偏离后的正解是重进
        #      （一次导航），不是再滑回去（多次盲滑，还会污染 OCR 读数 ——
        #      08-12 实测删掉滑动后，票数从假的 `6` 变成真值 `0`）。
        if self.cfg.get("difficulty", "highest") == "highest":
            target = max(enters, key=lambda b: b.cy)   # 屏上最低 = 最高难度
        else:
            target = min(enters, key=lambda b: b.cy)
        return tap_box(target, f"入场（{self.cfg.get('difficulty','highest')} 难度，"
                               f"屏上{len(enters)}关取"
                               f"{'最低' if self.cfg.get('difficulty','highest')=='highest' else '最高'}"
                               f"那个，不滑列表）")

    # ── 任務資訊  MAX  扫荡 ───────────────────────────────────────────
    def on_stage_popup(self, obs, st):
        if self.state.get("switching"):
            return (self.exit_step(obs, prefer_close=False)
                    or wait("换分支: 先关掉这个面板"))
        # AP 闸（JFD 那类要 AP 的）。游戏自己会把 MAX 钳到付得起的次数，
        # 所以只要够一次就可以放心 MAX —— 老代码那道
        # `ap ≥ 票数×每次AP` 的闸会在票多 AP 少时退化成"只扫 1 次"。
        if self.costs_ap and self.once("apcheck"):
            ap = R.read_topbar(obs, R.AP)
            if ap is not None and ap < self.ap_per_sweep:
                return self._next_branch_or_done(
                    f"AP {ap} < 一次扫荡 {self.ap_per_sweep} — 收工（绝不买 AP）",
                    terminal=True)

        # 没票守卫：**数量步进整行全灰**（减号灰+加号灰、行内无亮态）= 票 0。
        #    08-08 实测：MAX 排干 6 票回到面板，flow 还想再扫 —— 0 票按下去
        #    游戏会弹**用青辉石买票**的框。加号/减号必须按行配对看（顶栏的
        #    加号永远是亮的，全屏 has() 会误判）。
        mg = obs.find(V.QTY_MINUS_GREY, 0.45)
        if mg is not None:
            row = (0.0, mg.cy - 0.04, 1.0, mg.cy + 0.04)
            if (obs.has(V.PLUS_GREY, 0.45, region=row)
                    and not obs.has(V.QTY_MINUS, 0.45, region=row)
                    and not obs.has(V.PLUS, 0.45, region=row)):
                # 步进锁死 = 票 0，这是**屏上事实**，记进台账 —— 不记的话
                # 收尾会拿扫荡前的旧票数报"还剩 N 张没花"（08-09 帧证：
                # 数量 0 / 票预览 0- / AP 已扣，报告却说 票 66 LEFTOVER）
                self.state["tickets"] = 0
                x = obs.find(V.CLOSE_X, 0.45)
                if x is not None and self.pending("noticket_x"):
                    return tap_box(x, "步进器整行全灰 = 没票可扫  关面板",
                                   once="noticket_x")
                return self._next_branch_or_done(
                    "数量步进全灰 — 没票可扫了", terminal=True)

        mx = obs.find(V.QTY_MAX, 0.20)      # 弱类，蓝 MAX 上只到 conf≈0.26
        if mx is not None and self.pending("maxed"):
            return tap_box(mx, "数量拉 MAX（游戏会自己钳到付得起的次数）",
                           once="maxed")

        # 语义优先，不是 conf argmax：08-08 实测 `任务开始` 0.99 > `扫荡开始`
        #    0.98，放一个列表里 argmax 会选中**真打进战斗**的那个键。
        #    扫荡流**永远不许**点 任務開始。
        go = obs.find(V.SWEEP_START, 0.35)
        if go is not None:
            # 数量闸（2026-08-11 特殊任務实帧逼出来的）：数量 **0** 时
            #    「掃蕩開始」**照样是亮的**（conf 0.986，不是灰态），点下去游戏
            #    弹「購買AP 💎30」—— 正是 30 青辉石那次近失的入口。
            #    上面那道「步进器整行全灰」在 AP 流上**分不开 0 和 1**
            #    （数量到上限 1 时 MIN/−/+/MAX 同样全灰） 必须读那个数。
            #    **只做加法**：读出 0 才拦；读不出(None)退回原有行为，
            #      不动 bounty/jfd 已经走通的路径。
            n = R.read_qty(obs)
            if n == 0:
                self.state["tickets"] = 0
                return self._next_branch_or_done(
                    "步进器数量读出来是 0 —— 一次也扫不了（掃蕩鍵亮着也不点，"
                    "点下去就是买 AP/买票的框）", terminal=True)
            self.once_reset("maxed", "apcheck")
            return tap_box(go, f"扫荡开始（数量 {n if n is not None else '?'}）")
        if obs.has(V.TASK_START, 0.35):
            return self._next_branch_or_done(
                "面板上只有『任務開始』没有扫荡键 — 这关不能扫（未三星？），"
                "扫荡流不打真战斗")

        # 灰态的扫荡键 = 这一关现在扫不了
        if obs.has(V.CONFIRM_GREY, 0.45):
            return self._next_branch_or_done("扫荡键是灰的 — 条件不满足")
        if self.stalled(st, 90):
            return self._next_branch_or_done("任務資訊 里没有扫荡/开始键")
        return wait("任務資訊: 等扫荡键")

    def on_confirm_dialog(self, obs, st):
        cf = obs.find(V.CONFIRM, 0.45, region=CONFIRM_BAND)
        if cf is None:
            return wait("确认框: 等確認键")
        return tap_box(cf, "確認扫荡", counter="sweeps")

    def on_sweep_dialog(self, obs, st):
        return self.on_stage_popup(obs, st)

    # ── 收尾 ────────────────────────────────────────────────────────────
    def _next_branch_or_done(self, why: str, terminal: bool = False) -> Action:
        """换下一个分支；`terminal=True` 表示这条理由对**所有分支**都成立。

        票是分支共用的（悬赏 1/6 总数、交流会同理），所以「票用完 / 票读不出 /
           AP 不够」在别的分支上一样成立 —— 挨个走过去只是白跑三趟导航。
           只有「这个分支没关可打 / 只有锁着的关 / 这个分支配额用完」才该换。
        """
        if self.state.get("switching"):
            return wait("已经在换分支了，别重复自增")
        want = (self.cfg.get("branches") or self.cfg.get("academies")
                or list(self._plan().keys()))
        self.state["branch_i"] += 1
        # branch_tix0 必须跟着分支一起清 —— 不清的话下一个分支拿上一个的
        #   票基线算 used，配额会瞬间"用完"直接跳过。
        self.state.update(branch_name="", branch_tix0=None)
        self.once_reset("maxed", "apcheck")
        if terminal:
            return self._wrap(why)
        if self.state["branch_i"] < max(1, len(want)):
            self.log(f"{why}  换下一个分支（先退回分支页）")
            self.note_lines.append(why)
            # 只置 latch，真正的导航动作由 `_on_stage_list` / `on_stage_popup`
            #   下一帧发出（那里能看到当前屏上有哪个退出控件）。
            self.state["switching"] = True
            return wait(f"换分支: {why}")
        return self._wrap(why)

    def _wrap(self, why: str) -> Action:
        t0, t1 = self.state["tickets0"], self.state["tickets"]
        used = (t0 - t1) if (t0 is not None and t1 is not None) else None
        det = f"扫荡 {self.state['sweeps']} 次, {self.battle_stats()}"
        if used is not None:
            det += f", 票 {t0}{t1}(用了 {used})"
        # 竣工判据：票没用完 = LEFTOVER，别谎报 CLEAN
        # 但**配了票配额时剩票是设计不是漏活**：用户明确说"一个地区用几张"，
        #   照旧报 LEFTOVER 会让每一轮都挂假警报，久了就没人看 exit_report 了。
        plan = self._plan()
        by_plan = bool(plan) and self.state.get("planned", 0) >= sum(plan.values())
        if by_plan:
            det += f"（按配额跑完：{plan}）"
        if t1 is not None and t1 > int(self.cfg.get("keep_tickets", 0) or 0) \
                and "票用完" not in why and not by_plan:
            return self.finish(Outcome.LEFTOVER, f"{why}；{det}（还剩 {t1} 张票没花）")
        if t1 is None:
            return self.finish(Outcome.UNKNOWN, f"{why}；{det}（票数读不出，干没干净不确定）")
        # 2026-08-12 用户点名:「学园交流会**也有黄点**说明之前也没打好」。
        #    当轮 jfd 报 `CLEAN: 没票可扫；扫荡 1 次, 票 60(用了 6)` ——
        #    **扫荡 1 次却说用掉 6 张票**，而任务大厅磁贴上黄点还在。
        #     票基线 `tickets0` 是读大来的假数（`0/6` 被读成 `6/6` 之类），
        #      收尾拿假基线算出"用了 6"，于是**没打的活被报成干净**。
        #    **两个独立观测互相打架时，不许挑一个当真相报 CLEAN**：
        #      `sweeps` 是数事实（tap 真发出去才 +1），票差值是 OCR 读的；
        #      差得离谱就说明票读数不可信  报 UNKNOWN，让它在报告里红着。
        #    不用 `used > sweeps` 直接判——一次扫荡可能扣多张票（MAX 批量）。
        #      只在**扫荡次数为 0 却"用掉"了票**这种硬矛盾上出声。
        if used is not None and used > 0 and int(self.state.get("sweeps", 0)) == 0:
            return self.finish(
                Outcome.UNKNOWN,
                f"{why}；{det} — 票差值说用了 {used} 张但**一次扫荡都没发出去**，"
                f"票基线多半是读大的假数，这个分支到底打没打**不确定**")
        return self.finish(Outcome.CLEAN, f"{why}；{det}")

    # ── 页面路由（子类把自己的页面名接到通用处理上）──────────────────
    # 不许覆写 decide()！第一版在这里自己写了按页派发，把基类的
    #    **overlay页面** 顺序整个跳过  扫荡确认框开着时照样跑
    #    `_on_stage_list`，把对话框里的**费用行票图标**当成票数锚，读出 0
    #     伪报「票用完了」收工（2026-08-08 悬赏 live 抓到）。
    #    正解 = 把分支/列表页注册成 `on_<页面名>` 别名，走基类派发。
    def _register_pages(self) -> None:
        if self.branch_page:
            setattr(self, f"on_{self.branch_page}", self._on_branch)
        if self.stage_page:
            setattr(self, f"on_{self.stage_page}", self._on_stage_list)
