# -*- coding: utf-8 -*-
"""Flow 基类 —— 「看到 X 就做 Y」的骨架。

写一个 flow 只需要两件事:
  1. 声明 `module` / `entry`（大厅哪个入口进）
  2. 给每个可能落到的页面写一个 `on_<page>(obs, st)` 方法

`decide()` 按已确认的页面身份派发到对应的 `on_*`；**没有对应方法 = 返回 None
= no-op**（§A1 兜底必须是 no-op）。这个"找不到就不动"的默认值是刻意的：
老代码里那句 `return "STRAY", 回大厅按钮` 就是因为默认值是个动作，才会把
每个转场帧翻译成"把自己弹回大厅"。

竣工判据（§completion_gap）: flow 结束必须给 CLEAN / LEFTOVER / UNKNOWN，
**不许"跑完了"就算完**。UNKNOWN 是合法答案，比谎报 CLEAN 强得多。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from routing_v2.act.action import Action, done, note, tap_box, wait
from routing_v2.percept.observe import Observation
from routing_v2.state import vocab as V
from routing_v2.state.machine import StateView


class Outcome:
    CLEAN = "CLEAN"
    LEFTOVER = "LEFTOVER"
    UNKNOWN = "UNKNOWN"
    SKIPPED = "SKIPPED"
    BLOCKED = "BLOCKED"


@dataclass
class Ctx:
    """跨 flow 共享的运行时。flow 之间要传话（比如商店推算结果给加成队用）
    就放这里，别互相 import。"""
    cfg: Dict[str, Any]
    device: Any = None
    feed: Any = None
    ledger: Any = None
    log: Callable[[str], None] = print
    shots: Any = None                          # 帧落盘器（飞轮素材）
    bag: Dict[str, Any] = field(default_factory=dict)   # flow 间传话
    reports: List[str] = field(default_factory=list)


class Flow:
    name: str = "flow"
    module: str = ""                    # cfg["modules"] 里的键
    entry: Optional[str] = None         # 大厅入口 cls（None = 不从大厅进）
    yolo: tuple = ("ui",)               # 这个 flow 需要哪些模型
    # 这个 flow "在自己地盘上"的页面。用于判断"该退出去了"。
    home_pages: tuple = ()

    def __init__(self, ctx: Ctx):
        self.ctx = ctx
        self.cfg = ctx.cfg.get(self.module or self.name, {}) or {}
        self.log = lambda m: ctx.log(f"  [{self.name}] {m}")
        self.outcome: Optional[str] = None
        self.note_lines: List[str] = []
        self.t0 = time.time()
        self.ticks = 0
        self.state: Dict[str, Any] = {}          # flow 自己的小状态
        self.setup()

    # ── 生命周期 ────────────────────────────────────────────────────────
    def setup(self) -> None:
        """子类覆盖：初始化自己的计数器。"""

    def enabled(self) -> bool:
        return bool(self.ctx.cfg.get("modules", {}).get(self.module, False)) \
            if self.module else True

    def finish(self, outcome: str, why: str = "") -> Action:
        self.outcome = outcome
        if why:
            self.note_lines.append(why)
        return done(outcome, why or outcome)

    def report(self) -> str:
        dur = time.time() - self.t0
        head = f"{self.name:<16s} {self.outcome or Outcome.UNKNOWN:<9s} " \
               f"{dur:5.1f}s / {self.ticks} tick"
        if self.note_lines:
            return head + "\n" + "\n".join(f"      · {l}" for l in self.note_lines)
        return head

    # ── 派发 ────────────────────────────────────────────────────────────
    def decide(self, obs: Observation, st: StateView) -> Optional[Action]:
        """派发顺序：**覆盖层 → 底页 → offsite**。

        ⭐覆盖层先处理：对话框/奖励框盖在底页上时，先把它关掉才谈得上继续。
           但覆盖层处理器返回 None（比如这一帧確認键还没渲染出来）时要**落回
           底页**，否则会卡在"覆盖层认了但点不到"的空转里。
        """
        self.ticks += 1
        self.observe(obs, st)
        if st.overlay:
            fn = getattr(self, f"on_{st.overlay}", None)
            if fn is not None:
                act = fn(obs, st)
                if act is not None:
                    return act
        # ⭐pre_page：阶段机之类"先于页面、但**不许先于 overlay**"的逻辑挂这。
        #    ⛔子类绝不许覆写 decide()（架构不变量测试拦）——2026-08-08 两个
        #    子类各自写了按页派发, overlay 顺序全被跳过（sweep 把确认框里的
        #    费用图标当票数锚; event 同形）。
        act = self.pre_page(obs, st)
        if act is not None:
            return act
        fn = getattr(self, f"on_{st.page}", None)
        if fn is None:
            return self.on_offsite(obs, st)
        return fn(obs, st)

    def pre_page(self, obs: Observation, st: StateView) -> Optional[Action]:
        """overlay 之后、页面派发之前的钩子。默认无动作。"""
        return None

    def observe(self, obs: Observation, st: StateView) -> None:
        """**每一帧**都会调，不管当前是哪一页。

        ⛔为什么需要它：有些信号是**一闪而过**且出现在"哪一页都不是"的帧上。
           2026-08-08 实测：`战斗胜利` conf 0.99 出现在一个 page=unknown 的帧上，
           而胜负判定写在 `on_battle_result` 里 —— 等結算頁确认下来，胜利横幅
           早没了 ⇒ 18 场全记成"胜负未知"。
           ⇒ 这类**瞬时证据**必须在 observe 里粘住，不能等到对应页面才看。
        """

    # ── 全 flow 共享的通用页面处理（§A2：写一次，不是每个 flow 抄一遍）──
    def on_confirm_dialog(self, obs: Observation, st: StateView) -> Optional[Action]:
        """双键决策框的**默认**处理：灰确认 → 取消；否则确认。

        ⛔为什么敢给默认值：金钱闸（`act/money.py`）在这之前就跑了 ——
           真是购买框的话早就 halt 了，轮不到这里。剩下的双键框都是
           "确认扫荡 / 确认使用 / 确认领取"这类，确认是对的。
        ⛔2026-08-08 实测：不给默认值的后果是扫荡确认框落到 unknown，
           bot 在那儿关了 10 次弹窗叉叉，白白空转 200 帧。
        """
        # ⛔⛔系统「是否結束？」框（確認=退出游戏）的第二道防线。
        #    ⛔判据**必须用 act/money.system_dialog() 这唯一一份**（§A2）——
        #    08-09 实锤：这里原来自己抄了一份"顶栏没货币"判据，把 840AP
        #    加成扫荡确认框（盖住顶栏、但 `回大厅按钮` 0.96 在场）当成
        #    系统框点了取消 ⇒ 三层判据两层已修，漏的就是这层散装的。
        from routing_v2.act import money as _money
        if _money.system_dialog(obs, getattr(st, "last_solid", None)) is not None:
            c = obs.find(V.CANCEL, 0.45)
            if c is not None:
                return tap_box(c, "⛔疑似系统框（judged by money.system_dialog）→ 只点取消")
            return None
        if obs.has(V.CONFIRM_GREY, 0.45):
            c = obs.find(V.CANCEL, 0.45)
            return tap_box(c, "确认键是灰的 → 取消") if c is not None else None
        cf = obs.find(V.CONFIRM, 0.45)
        return tap_box(cf, "确认（双键框）") if cf is not None else None

    def on_ack_dialog(self, obs: Observation, st: StateView) -> Optional[Action]:
        """单键确认框 = 通知/结果页，点掉。

        ⛔为什么放在基类：这个框在**每一条链**上都会弹（领邮件、上完课、扫荡完、
           买完东西）。老代码让 13 个 skill 各写一遍，于是有的写了有的没写，
           没写的那条链就卡在这一帧上空转（08-08 mail 实测）。
        ⛔安全性：真是购买框的话 `act/money.py` 的判据会先把它拦成 halt，
           轮不到这里点。
        """
        cf = obs.find(V.CONFIRM, 0.45)
        return tap_box(cf, "确认（单键通知框）") if cf is not None else None

    def on_reward(self, obs: Observation, st: StateView) -> Optional[Action]:
        # ⛔语义优先级，不是 conf argmax（`获得奖励` 是横幅不是按钮，见 battle.py）
        # ⛔⛔**子类绝不许用 `obs.find([A, B])` 覆写这个方法**（08-11 live 实锤）：
        #    `find(列表)` 的语义是**全屏 conf argmax**，不是"先 A 后 B"——
        #    只有这里这种 `or` 链才是优先级。daily_mission/mail 各自抄了一份
        #    `find([GOT_REWARD, CONFIRM], 0.40)`，于是在「獲得獎勵！」全屏
        #    overlay 上（`获得奖励` **0.98** 横幅 @(0.500,0.224) vs
        #    `点击继续字样` **0.93** @(0.501,0.877)，帧上没有確認键）
        #    永远选中横幅 ⇒ 连点 11 次画面纹丝不动（02:00:41→02:02:59），
        #    而 `点击继续字样` 这一档**根本没进过候选**。
        #    ⇒ 覆写没有收益，一律继承本方法。
        b = (obs.find(V.CONFIRM, 0.40)
             or obs.find(V.STORY_TAP_CONTINUE, 0.40)
             or obs.find(V.GOT_REWARD, 0.40))
        return tap_box(b, "关掉奖励明细") if b is not None else None

    def on_sweep_results(self, obs: Observation, st: StateView) -> Optional[Action]:
        """掃蕩完成面板的奖励动画期：点 SKIP 直达总账（08-09 实锤缺口）。"""
        sk = obs.find(V.BATTLE_SKIP, 0.45)
        return tap_box(sk, "结算: SKIP 奖励动画") if sk is not None else None

    def on_claim_panel(self, obs: Observation, st: StateView) -> Optional[Action]:
        """可关闭的领取面板（有弹窗叉叉）：有黄键就领，全灰就叉掉。

        ⛔2026-08-08 live：cafe 收益领完后面板上只剩 `领取_灰`+叉叉，底页
           身份 unknown，flow 无出路卡死。这一层给所有 flow 兜住这个形态。
        """
        b = obs.find(V.CLAIM_ACTIVE, 0.45)
        if b is not None:
            return tap_box(b, f"领取（{b.cls}）", counter="claims")
        x = obs.find(V.CLOSE_X, 0.55)
        return tap_box(x, "领完了 → 叉掉面板") if x is not None else None

    def on_levelup(self, obs: Observation, st: StateView) -> Optional[Action]:
        b = obs.find([V.BOND_LEVELUP, V.REGION_LEVELUP], 0.40)
        return tap_box(b, "点掉升级过场") if b is not None else None

    def on_offsite(self, obs: Observation, st: StateView) -> Optional[Action]:
        """落到了这个 flow 没登记的页面。

        默认行为 = **什么都不做**，等 runner 的导航层把我们带回去。
        ⛔子类别在这里写"点返回键" —— 那正是老代码 4 处复制粘贴的
           `if page is not None: return action_back(...)`（§A2）。
        """
        return None

    # ── 小工具（子类常用；都不含闸，闸在 act/gate.py）────────────────────
    def tap(self, box, why: str, **kw) -> Action:
        return tap_box(box, f"{self.name}: {why}", **kw)

    def first(self, obs: Observation, names, conf: float = 0.5, **kw):
        return obs.find(names, conf, **kw)

    def bump(self, key: str, n: int = 1) -> int:
        self.state[key] = self.state.get(key, 0) + n
        return self.state[key]

    def once(self, key: str) -> bool:
        """同一件事这一轮只做一次（**决策时就标记**）。

        ⚠只给"纯逻辑/纯日志"用。要**点一下**的事情用 `pending()`，见下。
        """
        if self.state.get(f"once:{key}"):
            return False
        self.state[f"once:{key}"] = True
        return True

    def pending(self, key: str) -> bool:
        """这件事**还没真的做成**（不标记；由 runner 在 tap 落地后标记）。

        ⛔⛔2026-08-08 实测：craft 用 `once("quick")` 决定要不要点「快速製造」，
           那一发被 JIT 闸丢掉了（锚点漂移 0.051 > 容差 0.050），但 once 已经
           把标记消耗掉 ⇒ 下一帧不再重试 ⇒ **一次制造都没开成**，还报了 CLEAN。
           这是「数意图不数事实」的第 3 次复发（前两次是计数器和加成阶段）。
        ⇒ 配 `tap_box(..., once="quick")` 用：**tap 真的发出去了才算做过**。
           没发出去就每帧重试，重复由连发闸兜住。
        """
        return not self.state.get(f"once:{key}")

    def once_reset(self, *keys: str) -> None:
        """清掉 once 标记。不给 keys = 全清（换页/换厅时用）。"""
        if not keys:
            for k in [k for k in self.state if k.startswith("once:")]:
                self.state.pop(k, None)
            return
        for k in keys:
            self.state.pop(f"once:{k}", None)

    def hold(self, key: str, n: int) -> bool:
        """某个**内容**判据必须连续 n 个 tick 成立才算数。

        ⛔⛔§A3 的内容层版本（2026-08-08 新架构第一次跑活动就复发）:
           页面身份的连续帧确认在 `state/machine.py` 里，但"屏上还有没有可打的
           关"这类**内容**判据仍然是单帧的 —— 点完入场键的**过渡帧**上，刚点
           的那一行会瞬时消失，于是当场判成"没关可打，收工"，和老代码那个
           bug 一字不差。
           ⇒ **任何"收工/没活了"的判断都必须过这道 hold。**
        """
        k = f"hold:{key}"
        cnt = (self.state.get(k, 0) + 1
               if self.state.get(k + ":t", -99) == self.ticks - 1 else 1)
        self.state[k] = cnt
        self.state[k + ":t"] = self.ticks
        return cnt >= n

    def stalled(self, st: StateView, frames: int = 90) -> bool:
        """在同一页干瞪眼太久了。用来判"这条支线走不通，收工"。"""
        return st.frames_in_page >= frames


class ExitMixin:
    """通用退出：**必须有明确的、当前页面专属的控件**才动手（§A1）。

    ⛔⛔**顺序是 取消 > 结果框確認 > 叉叉 > 返回键**，这个顺序是流血换的
       （2026-07-27 悬赏弃 6 票的第三个根因）:
       停在「要使用6票掃蕩6次嗎?」这种**模态框**上时，屏上那个被检成
       `弹窗叉叉` conf **0.96** 的小 ✕ **根本不吃点击** —— 连点 6 次画面
       纹丝不动。模态框唯一正解是它自己的「取消」。
       ⚠**高 conf 检出 ≠ 可点击**：0.96 的框照样可能是装饰层/被遮挡层。

    ⛔另一条负门禁（2026-06-10 就写过，07-27 又用一次）：
       **结果弹窗永远没有取消键**。全语料 44,669 帧实证：奖励弹窗 494 帧里
       带取消键的 **0 帧**。所以"有取消键"= 这是个**决策框**不是结果框，
       退出它只能点取消，不能点確認（点確認就等于答应了那件事）。
    """

    def exit_step(self, obs: Observation, prefer_close: bool = True):
        # ① 取消键最优先 —— 模态框上它是唯一有效的出口
        c = obs.find(V.CANCEL, 0.45)
        if c is not None:
            return tap_box(c, "退出: 取消（模态框唯一有效出口）")
        # ② 结果框（有確認、**没有**取消）→ 点確認关掉
        cf = obs.find(V.CONFIRM, 0.45)
        if cf is not None:
            return tap_box(cf, "退出: 確認（结果框；结果框永远没有取消键）")
        # ③ 叉叉
        if prefer_close:
            x = obs.find(V.CLOSE_X, 0.55)
            if x is not None:
                return tap_box(x, "退出: 关弹窗")
        # ④ ⛔⛔**要么一路返回，要么一路回大厅，绝不混用**（用户 2026-08-12:
        #    「不要又点返回又点大厅按钮，**要么返回要么大厅**，根据逻辑语义
        #      重新修改」）。
        #    原来是「有返回键就点返回，没有才点回大厅」——**两个键的语义完全不同**:
        #      · `返回键`   = 退**一层**（下一个 flow 可能就在中间层认出入口）
        #      · `回大厅`   = 直接跳回**大厅**（跨越所有中间层）
        #    混着用的表现就是 live 里那串「点一下返回、紧接着点一下回大厅」，
        #    看起来像重复点击，实际是**两套退出策略在同一条链上打架**：
        #    返回退了一层，新页面正好没有返回键，于是又跳回大厅，前一下白点。
        #    ⇒ 选定语义后**坚持到底**：本 flow 的 `exit_step` 是"逐层退"语义
        #      （给下一个 flow 在中间层认出入口的机会），那就**只用返回键**；
        #      屏上没有返回键 = 这一层退不了，交给 ⑤ 的系统返回键，
        #      而不是改用另一套语义。
        #    ⚠真想"直接回大厅"的调用方请显式走 `nav.to_lobby()`，别指望这里。
        b = obs.find(V.BACK, 0.55)
        if b is not None:
            return tap_box(b, "退出: 返回键（逐层退，全程只用这一种）")
        # ⑤ 最后手段：系统返回键。安全性由 `nav.back_key()` **统一**把关
        #    （大厅上按返回会弹退出游戏框 —— 见那里的注释）。
        from routing_v2.flow.nav import back_key
        return back_key(obs, "退出: 屏上无退出控件 → 系统返回键")
