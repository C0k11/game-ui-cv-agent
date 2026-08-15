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
# 只有青辉石是"只进不出"的红线货币。信用点/AP 会正常花掉。
GUARDED = R.PYROXENE

# **这些页面不采样** —— 顶栏不是标准形态, 别的数字会抢裁切窗口。
#   2026-08-13 实锤: formation 页把青辉石稳定读成 211,286(真值 21,286),
#   稳定错读复读一致, 把「双次投票」共识整个骗过 -> 台账凭空 +190,000
#   又 -190,000。稳定错读跟着**页面**走, 换页就不复现 —— 所以防线是
#   按页面关采样, 不是加投票次数。
NO_SAMPLE_PAGES = {"formation", "battle", "battle_result", "grid_quest"}


@dataclass
class Entry:
    ts: float
    cls: str
    value: int
    page: str
    flow: str
    tag: str = ""            # "" / "suspect" / "breach" / "spike"

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
        self._suspect: Optional[int] = None
        self._suspect_page: str = ""
        self._maxdigits: Dict[str, int] = {}
        self._rejects: Dict[str, list] = {}   # 位数闸拒收史（基线自愈用）
        self._grow_pending: Dict[str, int] = {}   # 位数变多的待复读值
        self._first_pending: Dict[str, int] = {}  # 初始基线的待复读值
        self._ins_fix: Dict[str, int] = {}    # 插位读大被结构闸咬回的次数
        out = Path(out_dir) if out_dir else _OUT
        out.mkdir(parents=True, exist_ok=True)
        self.path = out / f"ledger_{time.strftime('%Y%m%d')}.jsonl"

    # ── 采样 ────────────────────────────────────────────────────────────
    def feed(self, obs: Observation, page: str, flow: str) -> Optional[str]:
        """每 tick 喂一帧。返回非 None = 出事了（breach 文案）。

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

    # ── 落账 + kill-switch ─────────────────────────────────────────────
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
            #    ①千位逗号读成 0（08-09 event_shop "20,176" -> 200176）;
            #    ②紧排数字读成双（08-15 顶栏 58,723,911 -> 588,723,911,
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
            if cls == GUARDED and prev is not None and v < prev:
                if not self._recheck:
                    # 第一次发现掉  **不停**，挂起等复读
                    self._recheck = True
                    self._suspect = v
                    self._suspect_page = page
                    self._last_commit = 0.0     # 立刻重采
                    self._log(f"[ledger] 青辉石 {prev}  {v}（-{prev-v}）"
                              f" 疑似掉钱 @page={page} — **挂起等换个页面复读**")
                    self._write(Entry(time.time(), cls, v, page, flow, "suspect"))
                    return None
                # 复读**必须在另一个页面上**才算数。
                #    2026-08-08 实测两次：同一页上的误读是**稳定的** ——
                #    cafe 页把 20,176 稳定读成 20,117，于是"独立复读"每次都
                #    复现同一个错值，把误报盖章成"两次一致"，整轮日常被急停。
                #    换个页面，顶栏背景/渲染都不同，稳定误读不会跟着走；
                #    而**真的掉钱在哪一页读都是掉的**。
                if page == self._suspect_page:
                    self._last_commit = 0.0
                    return None                 # 还在同一页，继续等
                if v != self._suspect:
                    self._recheck = False
                    self._log(f"[ledger] 换页复读得到 {v}（前一页读的是 "
                              f"{self._suspect}）— 两页不一致 = OCR 误读，警报解除")
                    self._write(Entry(time.time(), cls, v, page, flow, "misread"))
                    continue
                # 换了页面、值还是一样低  真的出事了
                self._recheck = False
                tag = "breach"
                msg = (f"MONEY BREACH: {cls} {prev}  {v}（-{prev-v}）"
                       f" 在 {self._suspect_page} 和 {page} 两个页面上都读到"
                       f" @flow={flow}")
                self.breach = msg
                self._log(f"[ledger] {msg}")
            elif cls == GUARDED and prev is not None and v > prev + 20000:
                # 读高会掩盖真实掉钱 —— 异常上涨同样打标交人看
                tag = "spike"
                self._log(f"[ledger] 青辉石异常上涨 {prev}  {v} — 标记待人审"
                          f"（读高会掩盖真掉钱）")
            if cls == GUARDED and self._recheck and tag != "breach":
                self._recheck = False        # 复读结果正常  警报解除
                self._log(f"[ledger] 复读 {v}，与 {prev} 一致/未跌 — 假警报解除")
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

    # ── 报告 ────────────────────────────────────────────────────────────
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
        lines.append(f"  明细已落盘: {self.path}")
        return "\n".join(lines)
