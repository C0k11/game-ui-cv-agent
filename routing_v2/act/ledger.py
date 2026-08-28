# -*- coding: utf-8 -*-
"""余额台账 + 青辉石 kill-switch。

两件事，都是今天的血:

【一】**每 tick 落 balance 台账**。
   2026-08-07 那天青辉石 18,036  19,766（+1,730 全是收入），但**逐笔来源
   追不到** —— trajectory 里根本没有 balance 字段。于是"钱是怎么变的"只能
   靠事后猜。这里把每一次确认读数连同 (页面, 正在跑的 flow) 一起落盘。

【二】**kill-switch 不能靠单次读数开火**。
   同一天 kill-switch 报 `青辉石 1803618003 MONEY BREACH` 急停了整条
   pipeline，而人眼看余额一分没少 —— 是**裁切窗口**读错（§A10）。
    这里的规则：怀疑掉钱 **不立刻停**，而是**重新独立投票一次**；两次
     独立投票都说掉了才停。方向仍然是安全的（真掉钱的话第二次也会掉），
     但把"OCR 抖一下就急停"这个假阳性堵死了。
   反向危害同样存在：**会读低就会读高，读高会掩盖真实掉钱**  读数上升
     也要落盘，异常上升（一次涨超过阈值）同样打标。

【三】**用户自己抽卡/升级不是事故**（2026-08-15）。
   旧 kill-switch 把「青辉石只进不出」做成：两次换页复读都低于基线就
   MONEY BREACH 停整轮。用户上线抽卡、升级角色、手点商店时余额会变，
   bot 没点付费键也被当成自己花了石，日常/推图整轮 HALT。
   现在：无 bot 付费/刷新/确认付费 tap 的下降记 EXTERNAL，更新基线，
   不 HALT。有 bot 付费窗的青辉石下降仍走换页复读 + BREACH。
   OCR 读不出（-1）仍然不点花钱键；读数对不上只拒收/自愈，不杀进程。

【四】**台账是事后账，不是防线**（2026-08-21 用户定 + 全库实测）。
   信用点/青辉石扫描的职责是**对账**：玩家自己要抽卡、升角色，余额本来就会
   动；只有体力（AP）才是拿来做动态规划的输入。
   实测支撑：全库 5,751 行台账里 `MONEY BREACH` 一共只出过 **2 次**（08-08），
     两次都是青辉石 `20,176` 被稳定读成 `20,117` —— 钱一分没动，却把整轮停了。
     换页复读那道闸挡不住它：稳定误读跨页一样稳定。**命中率 0/2**。
   真正拦住花钱的是 `act/gate.py` 里 tap **之前**那两条（`spend=青辉石` 直接
     halt / `purchase_context` 拦成交键）—— 钱还没动就否掉了。台账在钱**已经**
     动了之后才开口，停轮救不回这一笔，只能防"接着再花第二笔"。
    首条掉钱 -> `WARN_MONEY`：大声记账 + 存证帧 + 进收工报告，**不停轮**；
     第二条起 -> `MONEY BREACH` 停轮（防连环失血）。
   注意：**不要**再加"同位数单字符不同就当 OCR 混淆"这类闸（08-21 我加过又
     拆了）：`20176->20117` 差的是两位，它根本抓不到；而 `20176->20146`
     （真花 30 石买 AP，就是 money_safety 那条血泪）恰好是一位，会被静默吃掉。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from routing_v2.percept import read as R
from routing_v2.percept.observe import Observation

_ROOT = Path(__file__).resolve().parents[2]
_OUT = _ROOT / "data" / "routing_v2"


def _one_insert(base: str, big: str) -> bool:
    """big 是否等于 base 插入了恰好一个字符（顺序不变）。"""
    if len(big) != len(base) + 1:
        return False
    i = 0
    skipped = False
    for ch in big:
        if i < len(base) and ch == base[i]:
            i += 1
        elif not skipped:
            skipped = True
        else:
            return False
    return i == len(base)

WATCHED = [R.PYROXENE, R.CREDIT, R.AP]
# 青辉石: bot 绝不花。用户外部消费记 EXTERNAL, 不 HALT。
# 信用点/AP 会正常花掉, 下降从不 BREACH。
GUARDED = R.PYROXENE

# 台账 8s 采一次, 「本 tick 有没有付费 tap」太窄: bot 点完, 余额下一拍才变。
# 20s 盖住采样+换页复读; 只给 note_bot_spend 开过窗的那一种货币。
_SPEND_WINDOW_S = 20.0
_SPEND_TO_WATCH = {
    "青辉石": R.PYROXENE,
    "青辉石(premium)": R.PYROXENE,
    "信用点": R.CREDIT,
}
_SPEND_IGNORE = frozenset({
    "免费", "活動幣", "活动币", "战术大赛货币", "战术大赛币",
})

# **这些页面不采样** —— 顶栏不是标准形态, 别的数字会抢裁切窗口。
#   2026-08-13 实锤: formation 页把青辉石稳定读成 211,286(真值 21,286),
#   稳定错读复读一致, 把「双次投票」共识整个骗过 -> 台账凭空 +190,000
#   又 -190,000。稳定错读跟着**页面**走, 换页就不复现 —— 所以防线是
#   按页面关采样, 不是加投票次数。
# arena 是 08-28 实锤新增: 该页顶栏 8/8 采样全错(2496 读成 2526/2552/5526),
#    页面本身也没有任何青辉石操作, 采了只会下毒。
NO_SAMPLE_PAGES = {"formation", "battle", "battle_result", "grid_quest",
                   "arena"}


@dataclass
class Entry:
    ts: float
    cls: str
    value: int
    page: str
    flow: str
    tag: str = ""            # "" / "suspect" / "breach" / "spike" / "external" / "spend"

    def as_dict(self) -> dict:
        return {"ts": round(self.ts, 3),
                "time": time.strftime("%H:%M:%S", time.localtime(self.ts)),
                "cls": self.cls, "value": self.value,
                "page": self.page, "flow": self.flow, "tag": self.tag}



class Ledger:
    def __init__(self, log=None, sample_every: float = 8.0,
                 out_dir: Optional[Path] = None):
        """out_dir = 账号桶（config.data_dir）。08-15 起生产路径一律分桶:
        ledger 文件名只有日历日, 大小号同一天会写进同一份（ledger_20260813
        实锤 05:48 大号 59M / 08:49 小号 35,544 混在一起）。None 只给
        测试/离线工具用（它们自己改 self.path）。"""
        self._log = log or (lambda m: print(m, flush=True))
        self.sample_every = sample_every
        self._vote = R.Vote()
        self._t_start = 0.0
        self._last_commit = 0.0
        self.confirmed: Dict[str, int] = {}
        self.entries: List[Entry] = []
        self.breach: Optional[str] = None
        self._recheck = False              # 正在做第二次独立投票
        # 无付费窗的下降挂起待确认: {v, page, n}
        self._ext_suspect = None
        # 大幅上涨挂起待确认: {v, page} -- 位数不变的读大(2496->5526)绕过
        #    读大闸, 单发即污染基线, 之后一切真读数全成"下降"
        self._up_suspect = None
        self._suspect: Optional[int] = None
        self._suspect_page: str = ""
        self._maxdigits: Dict[str, int] = {}
        self._rejects: Dict[str, list] = {}   # 位数闸拒收史（基线自愈用）
        self._grow_pending: Dict[str, int] = {}   # 位数变多的待复读值
        self._first_pending: Dict[str, int] = {}  # 初始基线的待复读值
        self._ins_fix: Dict[str, int] = {}    # 插位读大被结构闸咬回的次数
        self._breaches: List[str] = []        # 本轮真掉钱告警（第 2 条才停轮）
        self._bot_spend_ts: Dict[str, float] = {}
        self._bot_spend_why: Dict[str, str] = {}
        out = Path(out_dir) if out_dir else _OUT
        out.mkdir(parents=True, exist_ok=True)
        self.path = out / f"ledger_{time.strftime('%Y%m%d')}.jsonl"

    #  采样
    def feed(self, obs: Observation, page: str, flow: str) -> Optional[str]:
        """每 tick 喂一帧。返回:
           None = 无事件
           'EXTERNAL:...' = 用户外部变动, 已更新基线, runner 不许 HALT
           'MONEY BREACH:...' = bot 付费窗内青辉石下降, runner 停整轮

        只在**顶栏锚点可见**的帧上采样；进不去顶栏的页面（战斗内/全屏过场）
           自然跳过 —— 这就是"非大厅掉钱盲区"的来源，所以**下面还有一条**：
           每次 flow 切换时强制采一次，把盲区窗口压到最小。
        """
        now = time.time()
        if not self._t_start:
            self._t_start = now
        due = (now - self._last_commit) >= self.sample_every
        if not (due or self._recheck):
            return None
        if page in NO_SAMPLE_PAGES:
            return None                     # 顶栏不是标准形态，稳定错读温床
        if obs.find(GUARDED, 0.25, region=R.TOPBAR) is None:
            return None                     # 顶栏看不见，这帧没法读
        self._vote.feed(obs, WATCHED)
        if not self._vote.settled([GUARDED]):
            return None
        return self._commit(page, flow)

    def force_sample(self) -> None:
        """flow 交接点强制采一次（压缩盲区）。下一帧起立刻投票。"""
        self._last_commit = 0.0
        self._vote.reset()

    def note_bot_spend(self, reason: str = "", cls: str = "",
                       spend: str = "") -> None:
        """runner 在付费/刷新/确认付费 tap 之后调用。

        真发出去、tap 返回失败、JIT 丢掉都可能走到这里。漏记窗会把
        bot 花石当成 EXTERNAL。只给对应货币开 20s 窗。信用点商店的
        成功购买不许污染青辉石窗，否则用户接着抽卡会被误判成 bot 花了石。
        """
        cur = _SPEND_TO_WATCH.get(spend)
        if cur is None:
            if spend in _SPEND_IGNORE:
                return
            if cls == "购买青辉石" or "刷新" in (reason or ""):
                cur = GUARDED
            elif spend:
                cur = GUARDED
            elif cls in ("购买", "选择购买"):
                cur = R.CREDIT
            elif cls == "确认键":
                cur = GUARDED
            else:
                return
        self._bot_spend_ts[cur] = time.time()
        why = f"{cls} {reason} spend={spend}".strip()
        self._bot_spend_why[cur] = why
        self._log(f"[ledger] 记下 bot 付费 tap ({cur}): {why}")

    def _bot_spent_recently(self, cls: str) -> bool:
        ts = float(self._bot_spend_ts.get(cls, 0.0) or 0.0)
        return ts > 0.0 and (time.time() - ts) < _SPEND_WINDOW_S

    #  落账 + kill-switch
    def _commit(self, page: str, flow: str) -> Optional[str]:
        vals = self._vote.result()
        self._vote.reset()
        self._last_commit = time.time()
        msg = None
        for cls, v in vals.items():
            if v is None:
                continue
            prev = self.confirmed.get(cls)
            # **初始基线要两次一致才立**（08-15 实锤: 开跑第一读信用点截断成
            #    18,991 直接成了基线, 收尾报表的起点就是假的）。首读没有 prev
            #    可对照, 全部结构闸都使不上劲, 只能靠复读。
            if prev is None:
                if self._first_pending.get(cls) != v:
                    self._first_pending[cls] = v
                    self._write(Entry(time.time(), cls, v, page, flow,
                                      "first_suspect"))
                    continue
            # 读大 = 在真值里多认出一个字形。两种实锤形态:
            #    1)千位逗号读成 0（08-09 event_shop "20,176" -> 200176）;
            #    2)紧排数字读成双（08-15 顶栏 58,723,911 -> 588,723,911,
            #      同页同渲染**稳定复读**, "复读一致"闸拦不住 —— 位数上限被
            #      顶到 9 后, 后续所有真 8 位读数反被当截断整场拒收）。
            #    结构判据一击毙: 新值删掉**任意一个**字符能还原基线字串 ->
            #    就是基线本身。不设自动采信出口: 万一真涨了一位且恰好构成
            #    插位关系, 也只是短暂粘住 —— 下一笔真实变动就破坏该关系,
            #    从增长闸正常入账（信用点只作报表, 方向保守无害）。
            if (prev is not None and len(str(prev)) >= 4
                    and _one_insert(str(prev), str(v))):
                self._ins_fix[cls] = self._ins_fix.get(cls, 0) + 1
                self._write(Entry(time.time(), cls, v, page, flow,
                                  "insert_fix"))
                v = prev
            # 位数收缩闸：新读数**位数变少且掉了一大半** = 典型的 OCR 截断，
            #    不是真的变动。2026-08-08 本轮实测信用点被读成 `54,661`，
            #    真值 `54,618,001` —— 少了三位。信用点的裁切窗口至今没按人眼
            #    真值重标（README §A10 的遗留缺口，9 帧只对 3 次）。
            #    这道闸只挡"读小"，**挡不住读大** —— 读大会掩盖真实掉钱，
            #      所以青辉石那格必须靠标定本身准，不能指望这道闸。
            # 信用点专用：OCR **只会截断，不会凭空多出位数**  位数比历史
            #    最大值少的读数一律当截断丢掉。这一格的裁切窗口至今没按人眼
            #    真值重标（9 帧只对 3 次），是 §A10 的遗留缺口。
            #    只对信用点用：AP 会从 899 真掉到 99（位数真的会少），
            #      青辉石更不能用这条 —— 它靠双次投票，不靠位数。
            if cls == R.CREDIT:
                mx = self._maxdigits.get(cls, 0)
                if len(str(v)) < mx:
                    # **位数上限自愈**（2026-08-13 实锤）: 一次读大(57,776,160
                    #    -> 577,776,160, 稳定复读骗过增长守卫)把上限顶到 9 位,
                    #    之后所有正确的 8 位读数全被当「截断」拒收, 收尾余额
                    #    是假的。判据: 连续 4 次被拒的读数**位数一致** ->
                    #    中毒的是上限不是读数, 降回去并大声记录。
                    hist = self._rejects.setdefault(f"{cls}:digits", [])
                    hist.append(len(str(v)))
                    del hist[:-6]
                    if len(hist) >= 4 and len(set(hist[-4:])) == 1:
                        self._log(f"[ledger] 位数上限自愈: 信用点连续 4 次读到 "
                                  f"{hist[-1]} 位而上限 {mx} 位再没出现过 — "
                                  f"判定上限被读大污染, 降到 {hist[-1]} 位")
                        self._maxdigits[cls] = hist[-1]
                        hist.clear()
                        # 本次读数照常走后面的守卫（不直接入账）
                    else:
                        self._log(f"[ledger] 信用点 {v} 位数({len(str(v))}) < 历史最大"
                                  f"({mx}) — 判为 OCR 截断，不采信")
                        self._write(Entry(time.time(), cls, v, page, flow,
                                          "misread"))
                        continue
                else:
                    self._rejects.get(f"{cls}:digits", []).clear()
                self._maxdigits[cls] = max(self._maxdigits.get(cls, 0),
                                           len(str(v)))
            if prev is not None and len(str(v)) < len(str(prev)) and v < prev * 0.5:
                # 基线自愈（08-09 实锤 A10「会读低就会读高」的反向形态）：
                #    某帧青辉石被读大成 200117 **入了账**，之后所有正确的
                #    20176 全被这道"位数收缩"闸拒收 —— 中毒的是基线不是新读数。
                #    判据：同一个"更短"的值连续 4 次出现、而旧基线再没露过面
                #     基线是读大误读，纠正并大声记录。
                rej = self._rejects.setdefault(cls, [])
                rej.append(v)
                del rej[:-6]
                if len(rej) >= 4 and all(x == v for x in rej[-4:]):
                    self._log(f"[ledger] 基线自愈: {cls} 连续 4 次读到 {v} 而"
                              f"基线 {prev} 再没出现过 — 判定基线是**读大**误读，"
                              f"纠正 {prev}{v}")
                    self.confirmed[cls] = v
                    rej.clear()
                    self._write(Entry(time.time(), cls, v, page, flow,
                                      "baseline_fix"))
                    continue
                self._log(f"[ledger] {cls} 读数 {v} 比上次 {prev} 少了位数 "
                          f"— 判为 OCR 截断，**不采信**（裁切窗口待重标）")
                self._write(Entry(time.time(), cls, v, page, flow, "misread"))
                continue
            self._rejects.get(cls, []).clear()      # 正常读数 = 拒收史清零
            # 对称守卫：**位数变多**的读数 = 读大嫌疑（08-09 实锤：青辉石
            #    20,176 被读成 200117 且跨 3 页稳定复现, 把换页复读都骗过
            #    读大入账后正确值全被位数收缩闸拒收）。要连续两次读到**同一个**
            #    变多的值才收 —— 真实的位数增长(999910000)复读一致, 不受影响。
            # **基线自愈：旧值是新值的前缀 = 旧值被截断了**（2026-08-13 实锤，
            #    而且是我先判反了一次的那条）。
            #    live 抓到 `信用点 59,653 -> 59,653,863`，我第一反应是"读大"，
            #    加了一道"涨 50 倍就拒收"的闸。**用户把那一帧贴出来才发现
            #    屏上真值就是 59,653,863** —— 错的是**第一次读数**（截断少读
            #    3 位），它成了基线，于是真值反而被当成读大嫌疑，
            #    报告里凭空多出一笔 +59,594,210 的假变动。
            #    那道量级闸方向完全反：真值恰好是截断基线的 1000 倍，会被永久拒收。
            #     本文件上面那条原理直接给出正解:「OCR 只会截断，不会凭空多出
            #      位数」。所以 `str(prev)` 是 `str(v)` 的**前缀**时，可疑的是
            #      旧基线不是新读数  **静默采纳新值并修正基线，不记成变动**。
            if (prev is not None and len(str(v)) > len(str(prev))
                    and str(v).startswith(str(prev))):
                self._log(f"[ledger] {cls} 基线 {prev} 是新读数 {v} 的前缀 — "
                          f"**旧基线是截断值**，修正基线（不记成变动）")
                self._write(Entry(time.time(), cls, v, page, flow, "baseline_fix"))
                self.confirmed[cls] = v
                self._maxdigits[cls] = max(self._maxdigits.get(cls, 0), len(str(v)))
                self._grow_pending.pop(cls, None)
                continue
            if prev is not None and len(str(v)) > len(str(prev)):
                if self._grow_pending.get(cls) != v:
                    self._grow_pending[cls] = v
                    self._log(f"[ledger] {cls} 读数 {v} 位数多于基线 {prev} — "
                              f"读大嫌疑，等复读一致")
                    self._write(Entry(time.time(), cls, v, page, flow,
                                      "grow_suspect"))
                    continue
                self._log(f"[ledger] {cls} 位数增长 {prev}{v} 复读一致，收")
            self._grow_pending.pop(cls, None)
            tag = ""
            this = None
            if cls == GUARDED and prev is not None and v < prev:
                # _recheck 只在 bot 付费窗内首读下降时挂起。换页复读
                # 可能超过 20s, 不许半路改 EXTERNAL。
                if self._bot_spent_recently(cls) or self._recheck:
                    if not self._recheck:
                        # bot 刚点过付费键, 第一次发现掉: 不停, 挂起等换页复读
                        self._recheck = True
                        self._suspect = v
                        self._suspect_page = page
                        self._last_commit = 0.0
                        self._log(f"[ledger] 青辉石 {prev} 降到 {v}（-{prev-v}）"
                                  f" 疑似 bot 花石 @page={page} "
                                  f"({self._bot_spend_why.get(cls, '?')}) "
                                  f"-- 挂起等换个页面复读")
                        self._write(Entry(time.time(), cls, v, page, flow,
                                          "suspect"))
                        return None
                    # 复读必须在另一个页面上才算数。
                    #    2026-08-08 实测两次：同一页上的误读是稳定的 --
                    #    cafe 页把 20,176 稳定读成 20,117，独立复读每次都
                    #    复现同一个错值，把误报盖章成两次一致。
                    if page == self._suspect_page:
                        self._last_commit = 0.0
                        return None
                    if v != self._suspect:
                        self._recheck = False
                        self._log(f"[ledger] 换页复读得到 {v}（前一页读的是 "
                                  f"{self._suspect}）-- 两页不一致 = OCR 误读，"
                                  f"警报解除")
                        self._write(Entry(time.time(), cls, v, page, flow,
                                          "misread"))
                        continue
                    self._recheck = False
                    tag = "breach"
                    body = (f"{cls} {prev} 降到 {v}（-{prev-v}）"
                            f" 在 {self._suspect_page} 和 {page} 两个页面上都读到"
                            f" @flow={flow}（本窗有 bot 付费 tap: "
                            f"{self._bot_spend_why.get(cls, '?')}）")
                    self._breaches.append(body)
                    # **台账是事后账，不是防线**（2026-08-21 用户定 + 实测支撑）。
                    #    真正拦住花钱的是 act/gate.py 两条 tap 前的闸
                    #    (spend=青辉石 直接 halt / purchase_context 拦成交键)，
                    #    它们在钱动之前就否掉了。台账在钱**已经**动了之后才开口，
                    #    停轮救不回这一笔，只能防"接着再花第二笔"。
                    #    而它的历史命中率是 **0/2**: 全库 5,751 行台账里
                    #    MONEY BREACH 只出过 2 次(08-08), 两次都是青辉石
                    #    20176 被稳定读成 20117 -- 钱没动却停了整轮。
                    #    假阳性停整轮的代价天天在付，真阳性一次没有。
                    #     第一条: 大声记账 + 进收工报告，**不停轮**；
                    #      第二条起: 才当"在连续掉钱"停轮（防连环失血）。
                    if len(self._breaches) >= 2:
                        this = f"MONEY BREACH: {body}（本轮第 "
                        this += f"{len(self._breaches)} 次，连续掉钱，停轮）"
                        self.breach = this
                    else:
                        this = (f"WARN_MONEY: {body} -- 台账首条掉钱告警，"
                                f"已记账并进收工报告，不停轮；"
                                f"再出一条才停（tap 前的金钱闸不受影响）")
                    self._log(f"[ledger] {this}")
                else:
                    # 用户抽卡/手点: 记外部变动, 不 HALT。但**单次读数不入账**:
                    #    08-26 arena 实锤, 顶栏 2331/2316 双稳态误读半分钟摆
                    #    3 次, 每次向下摆就多记一条假 EXTERNAL + 假基线。
                    #    同页误读是稳定的(08-08 老教训), 确认取两条路之一:
                    #    换页复读到同样低值, 或同页连续 3 次采样(自然节奏
                    #    ~30s)仍是低值 -- 真花钱不会自己涨回去, 反弹即作废
                    #    (作废逻辑在链尾, 那里才看得到 v >= prev 的帧)。
                    #    挂起期间基线不动, 不加速采样(免得同页 3 次变 3 帧)。
                    sus = self._ext_suspect
                    if sus is None or v != sus["v"]:
                        self._ext_suspect = {"v": v, "page": page, "n": 1}
                        self._log(f"[ledger] {cls} {prev} 降到 {v} 且无 bot "
                                  f"付费 tap -- 挂起待换页复读")
                        _e = Entry(time.time(), cls, v, page, flow,
                                   "ext_suspect")
                        self.entries.append(_e)
                        self._write(_e)
                        continue
                    if page == sus["page"]:
                        # 同页复读不管几次都不作数 -- 08-08/08-28 两次实锤:
                        #    同页误读是**稳定的**, N 次一致恰是误读特征。
                        #    真外部消费在下一次换页(flow 交接必 force_sample)
                        #    自然坐实, 不会漏。
                        _e = Entry(time.time(), cls, v, page, flow,
                                   "ext_suspect")
                        self.entries.append(_e)
                        self._write(_e)
                        continue
                    self._ext_suspect = None
                    tag = "external"
                    this = (f"EXTERNAL: {cls} {prev} 降到 {v}（-{prev-v}）"
                            f" 无 bot 付费/刷新/确认付费 tap, 换页复读确认"
                            f" -- 用户抽卡或手点，更新基线，不 HALT "
                            f"@flow={flow} page={page}")
                    self._log(f"[ledger] {this}")
            elif cls == R.CREDIT and prev is not None and v < prev:
                if self._bot_spent_recently(cls):
                    tag = "spend"
                else:
                    tag = "external"
                    this = (f"EXTERNAL: {cls} {prev} 降到 {v}（-{prev-v}）"
                            f" 无 bot 付费 tap -- 用户升级等，更新基线，不 HALT "
                            f"@flow={flow} page={page}")
                    self._log(f"[ledger] {this}")
            elif (cls == GUARDED and prev is not None
                    and v > prev + 300):
                # 位数不变的读大: 2496->5526 一发就把基线顶上去, 之后一切
                #    真读数全成"下降"。>300 的上涨(奖励类单笔通常 <=300)
                #    也要换页复读; 回落 = 误读作废。
                up = self._up_suspect
                if up is not None and v == up["v"] and page != up["page"]:
                    self._up_suspect = None
                    tag = "big_income"
                    self._log(f"[ledger] {cls} {prev} 涨到 {v} 换页复读一致"
                              f" -- 大额入账收下")
                else:
                    if up is None or v != up["v"]:
                        self._up_suspect = {"v": v, "page": page}
                        self._log(f"[ledger] {cls} {prev} 涨到 {v}(+{v-prev})"
                                  f" -- 大额上涨挂起待换页复读(防读大污染基线)")
                    _e = Entry(time.time(), cls, v, page, flow, "up_suspect")
                    self.entries.append(_e)
                    self._write(_e)
                    continue
            elif cls == GUARDED and prev is not None and v > prev + 20000:
                # 读高会掩盖真实掉钱 —— 异常上涨同样打标交人看
                tag = "spike"
                self._log(f"[ledger] 青辉石异常上涨 {prev}  {v} — 标记待人审"
                          f"（读高会掩盖真掉钱）")
            if this:
                if this.startswith("MONEY BREACH") or msg is None:
                    msg = this
            if (cls == GUARDED and self._up_suspect is not None
                    and prev is not None and v <= prev + 300):
                self._up_suspect = None
                self._log(f"[ledger] {cls} 回到 {v} -- 此前的大额上涨是读大, "
                          f"挂起作废")
            if (cls == GUARDED and self._ext_suspect is not None
                    and prev is not None and v >= prev):
                self._ext_suspect = None
                self._log(f"[ledger] {cls} 回到 {v} -- 此前的下降是读数抖动, "
                          f"外部变动挂起作废")
            if cls == GUARDED and self._recheck and tag != "breach":
                self._recheck = False        # 复读结果正常  警报解除
                self._log(f"[ledger] 复读 {v}，与 {prev} 一致/未跌 -- 假警报解除")
            self.confirmed[cls] = v
            e = Entry(time.time(), cls, v, page, flow, tag)
            self.entries.append(e)
            self._write(e)
        return msg

    def _write(self, e: Entry) -> None:
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(e.as_dict(), ensure_ascii=False) + "\n")
        except Exception:
            # 金钱台账的磁盘审计链断了必须出声(内存里的账还在, 但落盘的
            #    对账记录会缺条 -- 静默缺条比报错危险)
            self._wfail = getattr(self, "_wfail", 0) + 1
            if self._wfail == 1:
                self._log(f"[ledger] 台账落盘失败(路径 {self.path}) -- "
                          f"内存账不受影响, 但磁盘审计链在缺条")

    #  报告
    def report(self) -> str:
        if not self.entries:
            return "台账: 一次都没读到（顶栏锚点全程不可见？）"
        lines = ["余额台账"]
        for cls in WATCHED:
            rows = [e for e in self.entries if e.cls == cls]
            if not rows:
                lines.append(f"  {cls:<6s} 无读数")
                continue
            first, last = rows[0].value, rows[-1].value
            d = last - first
            sign = "+" if d >= 0 else ""
            lines.append(f"  {cls:<6s} {first:>10,}  {last:>10,}  ({sign}{d:,})"
                         f"  [{len(rows)} 次确认读数]")
        # 逐笔变动明细（这就是 08-07 追不到的那部分）
        pyx = [e for e in self.entries if e.cls == GUARDED]
        moves = [(a, b) for a, b in zip(pyx, pyx[1:]) if a.value != b.value]
        if moves:
            lines.append(f"  青辉石逐笔变动（{len(moves)} 笔）:")
            for a, b in moves:
                d = b.value - a.value
                lines.append(f"    {b.as_dict()['time']}  {a.value:,}  {b.value:,}"
                             f"  ({'+' if d > 0 else ''}{d:,})  flow={b.flow} page={b.page}")
        wfail = getattr(self, "_wfail", 0)
        if wfail:
            lines.append(f"  WARN: 落盘失败 {wfail} 条 -- 磁盘明细不完整, "
                         f"以上内存账才是全量")
        if self._ins_fix:
            det = " / ".join(f"{k} x{n}" for k, n in self._ins_fix.items())
            lines.append(f"  插位读大被结构闸咬回: {det}"
                         f"（同页稳定复读的多一位误读, 详见 jsonl 的 insert_fix）")
        if self._breaches:
            lines.append(f"  **掉钱告警 {len(self._breaches)} 条**"
                         f"（第 1 条只记账不停轮; 第 2 条起停轮）:")
            for b in self._breaches:
                lines.append(f"    {b}")
        ext_n = sum(1 for e in self.entries if e.tag == "external")
        if ext_n:
            lines.append(f"  外部变动 {ext_n} 笔（用户抽卡/升级等，已更新基线，未停轮）")
        lines.append(f"  明细已落盘: {self.path}")
        return "\n".join(lines)
