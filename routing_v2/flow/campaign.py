# -*- coding: utf-8 -*-
"""任務推图（走格子, 集中指挥模式）-- 按用户点名的关卡列表连推。

三层分工:
   感知层  v16 的走格子族 cls（小号实测 0.85-0.98）
   几何层  flow/grid.py（方向语义 -> 本帧检出的格心, 真实帧 6/6 验证）
   答案层  data/grid_answers/{stage}.json（scripts/convert_baah_grid.py
           从 BAAH 原始转换; normal 150 + Hard 90, 全量含多队/portal/exchange）

**打哪几关是用户的策略**（cfg campaign.stages, 空则退回 campaign.stage 单关）,
   bot 只负责按用户顺序走。可跳号, 可 Normal+Hard 混。
战斗期靠游戏内 AUTO: 检出 `自动战斗关闭` 才点开（状态钮绝不盲 toggle）。
BAAH 是盲走, 我们每步都验: 落点取检出的真格心, 到达靠相位循环。
能力边界（2026-08-13 转换器盘点）: 单队纯 move 33 关（1-3 章 + H1-H3）;
   portal 最早 4-1 / 多队+属性队最早 6-1 / exchange 最早 6-4 --
   这些在**进关花 AP 之前**由 `_capability_gate` 拦下, 不进关走到一半才死。
"""
from __future__ import annotations

import re

from routing_v2.act.action import Action, tap_box, wait
from routing_v2.flow import grid, nav
from routing_v2.flow.base import ExitMixin, Flow, Outcome
from routing_v2.percept import read as R
from routing_v2.percept.observe import Observation
from routing_v2.state import vocab as V
from routing_v2.state.machine import StateView

# 队伍绑格和目标格心的判等距离（相对列步长的比例, 不是绝对值）
REACH = 0.45

# 配置关号: 可选 H + 章-节。不是 N/H（无前缀=Normal）或读不出章-节 = 非法。
_STAGE_ID = re.compile(r"^H?(\d{1,2})-(\d)$", re.I)


def parse_stage_id(raw):
    """配置关号 -> 规范 "3-2" / "H2-1"。非法返回 None。"""
    s = str(raw or "").strip()
    if not s:
        return None
    m = _STAGE_ID.fullmatch(s)
    if m is None:
        return None
    hard = s[0] in "Hh"
    return ("H" if hard else "") + f"{int(m.group(1))}-{m.group(2)}"


def resolve_queue(cfg):
    """(queue, bad)。bad 非空 = 非法号, 不许开跑。

    stages 空则退回 stage 单关; 两个都空 = ([], []) 自动打得星_0。
    去空白、去重, **保持用户顺序**（不按关号排序）。
    """
    raw = (cfg or {}).get("stages", None)
    items = []
    if isinstance(raw, str):
        items = [p.strip() for p in re.split(r"[,，;；]+", raw) if p.strip()]
    elif isinstance(raw, (list, tuple)):
        items = [str(x).strip() for x in raw if str(x).strip()]
    if not items:
        one = str((cfg or {}).get("stage", "") or "").strip()
        if not one:
            return [], []
        items = [one]
    queue = []
    seen = set()
    bad = []
    for s in items:
        canon = parse_stage_id(s)
        if canon is None:
            bad.append(s)
            continue
        if canon in seen:
            continue
        seen.add(canon)
        queue.append(canon)
    if bad:
        return [], bad
    return queue, []


class CampaignFlow(ExitMixin, Flow):
    name = "campaign"
    module = "campaign"
    entry_page = "task_hall"
    phases = ("enter", "stage_list", "popup", "grid", "walk", "result")
    # 战斗一场 2-4 分钟, 默认 600 tick 会把打着一半的仗判成卡死
    phase_cap = 4000

    def setup(self) -> None:
        self.state.update(round_i=0, target=None, need_end=False,
                          battles=0, done=[], skipped=[],
                          queue=[], queue_i=0, queue_bad=[])
        # stages 优先; 空则退回 stage 单关; 两个都空 = 自动打得星_0。
        #    非法号 setup 记下, 第一帧 decide 就 BLOCKED, 不进关。
        queue, bad = resolve_queue(self.cfg)
        self.state["queue_bad"] = bad
        if bad:
            self.state["stage"] = ""
            self.state["answer"] = None
            self.log("关卡号非法: " + "/".join(bad) + " -- 不进关")
            return
        self.state["queue"] = queue
        self.state["queue_i"] = 0
        if queue:
            self._load_current(queue[0])
        else:
            self.state["stage"] = ""
            self.state["answer"] = None
        # 断点续走: 设备上的任务已经走完前 N 步时从第 N+1 步接着走
        #    （中断任务在任务内没有可点的返回键, 拆不掉时用这个接上）。
        #    一次性参数, 用完就该清回 0。只作用于队列第一关。
        skip = int(self.cfg.get("skip_rounds", 0) or 0)
        if skip:
            # 08-30: 不在这就设 round_i -- 这参数是 08-15 留在 profile 里的
            #    调试遗留, 全新部署时也生效, 把 3-1 前两步吞了(队伍停在离
            #    BOSS 两格处, 10 AP 挂半路)。暂存, 走子开局按**部署方式**定:
            #    这一局自己点过 起点_黄 上队 = 全新开局, 配置声明也不听。
            self.state["pending_skip"] = skip
            self.log(f"skip_rounds={skip} 已暂存 — 只在真续走(进来时任务已"
                     f"在图上)时生效")

    @staticmethod
    def _parse_stage(raw, hard: bool):
        """OCR 原始串 -> 关号。"2-5" / 带噪 "12-5." 都收；解析不出返回 None。"""
        import re as _re
        if not raw:
            return None
        m = _re.search(r"(\d{1,2})-(\d)", raw)
        if m is None:
            return None
        return ("H" if hard else "") + f"{m.group(1)}-{m.group(2)}"

    def goto(self, phase, why=""):
        super().goto(phase, why)
        # step CLI 每发新进程, 只落盘 flow.state, 相位本身不存。
        #    不写回的话续走会从 enter 起手, 战斗页就干等任务大厅。
        self.state["phase"] = self.phase

    def pre_page(self, obs, st):
        # 非法号第一帧就收, 任何相位都不许开始找关/进关。
        if self.state.get("queue_bad"):
            bad = self.state["queue_bad"]
            return self.finish(
                Outcome.BLOCKED,
                f"关卡号非法: {'/'.join(bad)} -- 读不出章-节或不是 Normal/Hard, 不进关")
        saved = self.state.get("phase")
        if saved and saved in self.phases and not self.phase:
            self.goto(saved, "续上相位")
        return None

    def finish(self, outcome: str, why: str = ""):
        if (outcome == Outcome.UNKNOWN and self.state.get("queue")
                and "停在" not in (why or "")):
            extra = self._queue_summary("")
            if extra:
                why = f"{why} -- {extra}" if why else extra
        return super().finish(outcome, why)

    def _load_current(self, stage: str) -> None:
        self.state["stage"] = stage
        self.state["answer"] = grid.load_answer(stage) if stage else None

    def _reset_row_cache(self) -> None:
        self.state["row_anchor"] = None
        self.state["row_reads"] = {}
        self.state.pop("stage_vote", None)
        self.state["opened_popup"] = False
        self.state["area_hops"] = 0
        self.state["scrolls"] = 0

    def _reset_for_next_stage(self) -> None:
        self.state.update(
            round_i=0, target=None, need_end=False, battles=0,
            battle_seen=False, issued=False, cycling=False,
            pe_absent=0, bind_last=None, pre_vec=None,
            moved_t=None, moved_frames=0, start_box=None,
            dx_est=None, deploy_round0=0)
        self.state["cell_acc"] = []
        self._reset_row_cache()
        self._wt_clear()
        for k in [k for k in self.state if k.startswith("hold:")]:
            self.state.pop(k, None)

    def _queue_summary(self, head=""):
        q = self.state.get("queue") or []
        done = self.state.get("done") or []
        skipped = self.state.get("skipped") or []
        cur = self.state.get("stage") or "?"
        i = int(self.state.get("queue_i") or 0)
        parts = []
        if head:
            parts.append(head)
        if done:
            parts.append("已完成 " + "/".join(done))
        if skipped:
            parts.append("跳过 " + "/".join(skipped))
        if q:
            parts.append(f"停在 {cur}（队列 {i + 1}/{len(q)}）")
        return ", ".join(parts)

    def _fail_or_skip(self, why: str):
        """当前关没答案/能力不够: 有下一关就跳过继续, 队列全失败才 BLOCKED。"""
        cur = self.state.get("stage") or "?"
        q = self.state.get("queue") or []
        i = int(self.state.get("queue_i") or 0)
        has_next = bool(q and i + 1 < len(q))
        if has_next or self.state.get("done"):
            self.state.setdefault("skipped", []).append(cur)
            self.log(f"跳过 {cur}: {why}")
            self.note_lines.append(f"跳过 {cur}: {why}")
        if has_next:
            nxt = q[i + 1]
            self.state["queue_i"] = i + 1
            self._reset_for_next_stage()
            self._load_current(nxt)
            return None
        if self.state.get("done"):
            return self.finish(Outcome.CLEAN, self._queue_summary("剩余关都跳过"))
        return self.finish(Outcome.BLOCKED, why)

    def _ensure_playable(self):
        """当前关不能打就沿队列跳过。返回收工 Action, 或 None=可以进。"""
        if self.state.get("queue_bad"):
            bad = self.state["queue_bad"]
            return self.finish(
                Outcome.BLOCKED,
                f"关卡号非法: {'/'.join(bad)} -- 读不出章-节或不是 Normal/Hard, 不进关")
        hops = 0
        while hops < 64:
            hops += 1
            stage = self.state.get("stage") or ""
            if not stage:
                return None
            if self.state.get("answer") is None:
                act = self._fail_or_skip(f"没有 {stage} 的答案文件")
                if act is not None:
                    return act
                continue
            lack = self._capability_lack()
            if lack:
                act = self._fail_or_skip(
                    lack + " -- 进关前拦下, AP 一分没花")
                if act is not None:
                    return act
                continue
            ans = self.state.get("answer") or {}
            if ans.get("needs", {}).get("attrs") and self.once("attr_note"):
                self.log(f"答案推荐属性队 {'/'.join(ans['needs']['attrs'])}"
                         f"（blue=神秘 red=爆发 yellow=贯穿 purple=振动）, 打不过按这个配")
            return None
        return self.finish(Outcome.BLOCKED, "队列跳过次数异常")

    def _after_stage_clean(self, why_one: str):
        """一关走完: 有下一关就切关回 stage_list, 不要立刻 finish。"""
        cur = self.state.get("stage") or "?"
        self.state.setdefault("done", []).append(cur)
        self.log(why_one)
        if why_one and (not self.note_lines or self.note_lines[-1] != why_one):
            self.note_lines.append(why_one)
        q = self.state.get("queue") or []
        i = int(self.state.get("queue_i") or 0)
        if not q or i + 1 >= len(q):
            if q and len(q) > 1:
                return self.finish(Outcome.CLEAN, self._queue_summary(why_one))
            return self.finish(Outcome.CLEAN, why_one)
        nxt = q[i + 1]
        self.state["queue_i"] = i + 1
        self._reset_for_next_stage()
        self._load_current(nxt)
        blocked = self._ensure_playable()
        if blocked is not None:
            return blocked
        self.goto("stage_list", f"下一关 {self.state['stage']}")
        return wait(f"切下一关 {self.state['stage']}")

    # 观测: 格心跨帧累积（绿勾累积同款）。同一张地图上格子是**静态**的,
    #    单帧 conf 抖动(同图实测 0.31-0.97 帧间波动)不该抖掉我们对地图的认知。
    #    锁定**不用**战斗那套 ReID -- 格子不动, 位置累积就够; 单位离散跳格,
    #    每步重新按「正下方」绑格, 没有连续跟踪问题。
    def observe(self, obs, st) -> None:
        # 战斗计数按**页面事实的边沿**记（battle -> 非battle 记一场）。
        #    旧版把计数挂在 battle_result 里点確認那一发的 post 上 --
        #    结算被 overlay/确认链吃掉就没路过那分支, 2-1 首通报了"战斗 0 场"
        #    （计数=意图非事实, live_small_account_2026_08_13 记过的小虫）。
        if self.phase in ("walk", "result"):
            if st.page == "battle":
                self.state["battle_seen"] = True
            elif self.state.get("battle_seen"):
                self.state["battle_seen"] = False
                self.state["battles"] = self.state.get("battles", 0) + 1
        # **相位循环是瞬时证据, 必须在这里粘, 不能等页面身份**（08-13 H2-1
        #    live 实锤: 第 1 步走成了、回合已进 2/4, 但敌方相位的 HUD 消失
        #    窗口恰好撞上页面抖动(grid_quest<->unknown), do_walk 只在
        #    page==grid_quest 才看 PHASE 控件 -> 循环发生在它没看的帧里 ->
        #    时钟瞎掉, 拿着回合 1 的方向连发到 UNKNOWN。同 base.observe
        #    docstring 里「战斗胜利出现在 page=unknown 帧」那条的复发。
        #    连续 3 帧不见 PHASE 才算进循环 -- 单帧漏检不许当证据;
        #    overlay 在场不算(弹窗盖住 HUD 不是敌方回合)。
        if self.phase == "walk" and self.state.get("issued"):
            covered = bool(st.overlay) or obs.has(V.CLOSE_X, 0.55)
            if covered:
                pass    # 弹窗盖住 HUD ≠ 相位循环（帮助/资讯面板都带叉叉）
            elif obs.has([V.PHASE_END, V.PHASE_AUTO_ON, V.PHASE_AUTO_OFF],
                         0.40):
                self.state["pe_absent"] = 0
            else:
                n = self.state.get("pe_absent", 0) + 1
                self.state["pe_absent"] = n
                if n >= 3 and not self.state.get("cycling"):
                    self.state["cycling"] = True
                    self.log("PHASE 控件连续 3 帧不在 - 相位循环中（敌方回合/结算）")
            # **位移证据**（第二路回合时钟, 08-13 H2-1 实锤需要）: 迷雾图上
            #    敌方相位可能一个可见动作都没有, HUD 消失窗口只有 1-2 帧,
            #    连续 3 帧判据抓不到 -> 时钟瞎。单位相对**起点地标**的向量是
            #    相机不变量, 它移动了一格量级 = 这回合的落子已被游戏消费。
            #    确认后先给 PHASE 闪没 6s 机会（正常图走原时钟）, 超时仍
            #    没见闪 = 闪太快没抓到, 按位移推进。
            pv = self.state.get("pre_vec")
            if pv is not None and not covered and not self.state.get("cycling"):
                sp = obs.find([V.GRID_START, V.GRID_START_GREY], 0.35)
                un = self._bind_unit(obs)
                if sp is not None and un is not None:
                    dvx, dvy = un.cx - sp.cx, un.cy - sp.cy
                    dx0 = self.state.get("dx_est") or 0.09
                    dd = (dvx - pv[0]) ** 2 + (dvy - pv[1]) ** 2
                    # 位移必须是**一步量级**(0.6~2.0 格距): 敌我混淆让绑定
                    #    在两个立绘间跳变时, 伪位移通常远超一步 -- 不设上限
                    #    会把绑定漂移当成走位(假回合推进)
                    _hi = (float("inf")
                           if self.state.get("issued_do") == "portal"
                           else (2.0 * dx0) ** 2)
                    # portal 回合传送是多格量级, 2.0 格上限是防绑定漂移的,
                    #    对真传送必须放开; 下限保留(仍要求真动了)。
                    if (0.6 * dx0) ** 2 < dd < _hi:
                        m = self.state.get("moved_frames", 0) + 1
                        self.state["moved_frames"] = m
                        if m >= 2 and not self.state.get("moved_t"):
                            import time as _t
                            self.state["moved_t"] = _t.time()
                            self.log("位移证据: 队伍相对起点移动了一格量级")
                    else:
                        self.state["moved_frames"] = 0
            if (self.state.get("moved_t") and not self.state.get("cycling")):
                import time as _t
                if (self.state.get("pe_absent", 0) >= 1
                        or _t.time() - self.state["moved_t"] > 6.0):
                    self.state["cycling"] = True
                    self.log("按位移证据判相位循环（PHASE 闪没窗口太短没抓到）")
        if self.phase not in ("grid", "walk"):
            return
        acc = self.state.setdefault("cell_acc", [])
        for x, y in grid.cells(obs, 0.35):
            if all((x - a) ** 2 + (y - b) ** 2 > 0.04 ** 2 for a, b in acc):
                acc.append((x, y))
        sp = obs.find([V.GRID_START, V.GRID_START_GREY], 0.35)
        if sp is not None and self.state.get("start_box") is None:
            self.state["start_box"] = (sp.cx, sp.y1, sp.y2)

    def _acc_cells(self, obs):
        """本帧检出 + 历史累积的格心合集。"""
        cs = grid.cells(obs, 0.35)
        for a in self.state.get("cell_acc", []):
            if all((a[0] - x) ** 2 + (a[1] - y) ** 2 > 0.04 ** 2 for x, y in cs):
                cs.append(a)
        return cs

    def _plan(self):
        a = self.state.get("answer")
        return a["rounds"] if a else []

    # 墙钟等待闸: 走位里的裸 wait 分支必须有界(墙钟, 不是 tick --
    #    tick 速率实测 0.15-2.294 s/tick 差 15 倍, tick 当计时器一律是 bug)。
    #    「经常卡住不动」的用户观感 = 这些分支在感知缺位时静默空转到 phase_cap。
    def _overdue(self, key: str, secs: float) -> bool:
        import time as _t
        k = f"wt:{key}"
        t0 = self.state.get(k)
        if t0 is None:
            self.state[k] = _t.time()
            return False
        return _t.time() - t0 > secs

    def _wt_clear(self, *keys) -> None:
        if not keys:
            for k in [k for k in self.state if k.startswith("wt:")]:
                self.state.pop(k, None)
            return
        for k in keys:
            self.state.pop(f"wt:{k}", None)

    def _dump_grid_miss(self, obs) -> str:
        """无格框停手时落干净帧, 给 v18 补标, 不开训。"""
        if obs is None or getattr(obs, "frame", None) is None:
            return ""
        try:
            import time as _t
            from pathlib import Path as _P
            import cv2
            d = _P("data") / "raw_images" / f"v18_grid_miss_{_t.strftime('%Y%m%d')}"
            d.mkdir(parents=True, exist_ok=True)
            name = f"{_t.strftime('%H%M%S')}_{int(getattr(obs, 'seq', 0)):06d}_grid_quest.jpg"
            p = d / name
            cv2.imencode(".jpg", obs.frame,
                         [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(str(p))
            return str(p).replace("\\", "/")
        except Exception:
            return ""

    # 进关前能力预检: 答案要什么能力, 现在的 flow 会不会 -- 不会就 BLOCKED
    #    在花 AP 之前。旧行为是进关走到那一回合才 UNKNOWN: AP 扣了、任务挂在
    #    半路、归位链还要去拆「任务进行中」的残局, 三重浪费。
    def _capability_lack(self, ans=None):
        ans = ans if ans is not None else self.state.get("answer")
        if not ans:
            return None
        needs = ans.get("needs") or {}
        lack = []
        if needs.get("teams", 1) > 1:
            lack.append(f"{needs['teams']} 队协同(缺 当前聚焦队伍/切换键 判据)")
        # portal 已实现(08-30): 点击同 move, 事后地标推算作废转单位绑定,
        #    位移证据放开上限。多队/exchange/属性队仍未实现, 继续拦。
        if False and needs.get("portal"):
            lack.append("portal 传送(确认弹窗链未 live 验证)")
        if needs.get("exchange"):
            lack.append("exchange 换位(交換按钮无 cls)")
        if lack:
            return (f"{ans.get('stage', '?')} 的答案需要 " + " + ".join(lack))
        return None

    def _capability_gate(self):
        # 列表已锁定某一行之后的预检: 跳过就停手重找, 不许拿旧行入場去打下一关。
        lack = self._capability_lack()
        if lack:
            act = self._fail_or_skip(
                lack + " -- 进关前拦下, AP 一分没花")
            if act is not None:
                return act
            return wait(f"跳过, 改打 {self.state['stage']}")
        ans = self.state.get("answer") or {}
        if ans.get("needs", {}).get("attrs") and self.once("attr_note"):
            self.log(f"答案推荐属性队 {'/'.join(ans['needs']['attrs'])}"
                     f"（blue=神秘 red=爆发 yellow=贯穿 purple=振动）, 打不过按这个配")
        return None

    # 绑当前队伍（observe 的位移证据和 do_walk 落子共用一份逻辑）。
    #
    # 2026-08-24 改成**只认头顶那个黄色倒三角**（501 走格子_队伍箭头）:
    #    原来是「箭头正下方最近的我方框; 箭头缺席时唯一的我方框」, 依赖
    #    509 走格子_我方。可 509 在大池里**约 40% 框的是敌方/BOSS**
    #    (2,173 框散在 1,640 帧, 带上下文抽 21 张明确敌方 >=8 张, 红底 RANK
    #     菱形 / BOSS 横幅直接压在框上; 两版自动判据都被骗过 —— 红衣服、
    #     START 格黄高亮), 手工修不动, 已整类标废案。
    #    箭头则是**结构上只有我方才有**的标记(同样抽 21 张 21/21 干净),
    #    敌方永远没有, 不存在敌我混淆。
    #    箭头比立绘更高一截, 但 grid.below() 的判据是「同 x 带内正下方最近的
    #    格」, 高一点不影响: 实测同帧 箭头 cy .335 / 脚下格 cy .511, x 差 .001。
    def _bind_unit(self, obs):
        return obs.find(V.GRID_ARROW, 0.25)

    # enter
    def do_enter(self, obs, st):
        g = self._ensure_playable()
        if g is not None:
            return g
        if st.page == "battle":
            self.goto("walk", "已经在战斗中, 接手")
            return wait("进相位 walk")
        if st.page == "battle_result":
            self.goto("walk", "战斗结算, 接手")
            return wait("进相位 walk")
        if st.page == "lobby":
            return nav.lobby_enter(self, obs, V.NAV_TASKS, "任务大厅",
                                   expect=(V.HUB_CAMPAIGN,))
        # 关卡列表签名掉成 facility 时先自愈进 stage_list, 再看大厅磁贴
        #    (两条都可能在 facility 上成立, 列表页签优先, 免得在列表上再点磁贴)
        if (st.page in ("facility", "unknown")
                and obs.has([V.STAGE_NORMAL_SEL, V.STAGE_HARD_SEL,
                             V.STAGE_NORMAL, V.STAGE_HARD, V.SWEEP_BATCH],
                            0.45)):
            self.goto("stage_list", "关卡列表被判成 facility/unknown, 相位自愈")
            return wait("进相位 stage_list")
        # 任务大厅新身份=推图+(返回|回大厅); 旧皮少 tile 仍可能掉成 facility
        #    (08-15 H2-3: 任務/剧情/特殊任务刚好 3, 转场少一个 -> facility
        #    -> 本分支不跑, enter 干等 4001 tick)。磁贴在就当大厅。
        if (st.page in ("task_hall", "facility", "unknown")
                and obs.has(V.HUB_CAMPAIGN, 0.45)):
            t = obs.find(V.HUB_CAMPAIGN, 0.45)
            if t is not None:
                # 游戏会记住 Hard 页签, 进列表不一定出 普通关卡选中
                n = int(self.state.get("hub_taps", 0))
                act = tap_box(t, "进 任務 推图",
                              expect=(V.STAGE_NORMAL_SEL, V.STAGE_NORMAL,
                                      V.STAGE_HARD_SEL, V.STAGE_HARD),
                              post=lambda k=n: self.state.update(hub_taps=k + 1))
                # 任务关卡推图 框是磁贴顶上「Area N」标题条
                #    (08-15 实帧 y 0.215-0.273), 框心偏上, 转场期点标题没进。
                #    往下收到磁贴本体。
                # 08-30 实锤: 双倍活动缎带把这个 cls 打到 0.25-0.45 一带,
                #    框也从标题条胀到盖住「任務」大字(y2~0.35), y2+0.08 落进
                #    卡下死区水面, 7 发全空。两道保险:
                #    ①偏移封顶在卡底边内(第二排小卡 0.44 起, 0.395 是安全带);
                #    ②连空 3 发换打法: 直接点框心 -- 标签印在卡上, 框心必在卡内。
                if n >= 3:
                    act.y = t.cy
                else:
                    act.y = min(0.395, t.y2 + 0.08)
                return act
            return wait("找 任務 磁贴")
        if st.page == "campaign_stage":
            self.goto("stage_list", "到关卡列表了")
            return wait("进相位 stage_list")
        if st.page == "stage_popup":
            # 开局就压着关卡弹窗(上一轮残留/用户手开) -- 交给 stage_list 处理
            self.goto("stage_list", "开局就有关卡弹窗")
            return wait("进相位 stage_list")
        if st.page == "grid_quest":
            # 已经走过步/绑过格 = 续走, 进 walk 不要再经 grid
            #    (do_grid 会重写 deploy_round0, 航位从起点重算)。
            if (self.state.get("issued") or self.state.get("round_i", 0)
                    or self.state.get("bind_last")):
                self.goto("walk", "地图上续走")
                return wait("进相位 walk")
            # 上一轮没退干净, 直接从地图接手
            self.goto("grid", "已经在走格子地图上")
            return wait("进相位 grid")
        # 任務資訊框压在任务上（back 键会弹它, 三键: 中斷任務/重新挑戰/確認;
        #    中断/重挑 两个 cls 欠拟合, 框常被判成 battle_result）——
        #    点確認关掉继续, 关掉后地图露出来走上面的接管分支。
        if (st.page in ("battle_result", "unknown")
                and obs.has(V.CONFIRM, 0.45) and obs.has(V.CLOSE_X, 0.45)):
            cf = obs.find(V.CONFIRM, 0.45)
            return tap_box(cf, "关掉任務資訊框（点確認, 任务继续）")
        return wait("等任务大厅")

    # stage_list: 点**得星_0 那一行的入場键**（下一关就是没有星的那关 --
    #    游戏会自动把它归位到可视区, 老规矩）。
    #    2026-08-13 实帧纠错: `普通关卡选中` 是顶部 Normal **页签**, 不是
    #    选中的关卡行, 点它开不了弹窗。行内按钮锚定和咖啡厅邀请键同款。
    def do_stage_list(self, obs, st):
        # 证据和相位矛盾时退回（base.phases 文档里的自愈路径, 08-13 实录:
        #    归位的返回键其实生效了但死路检测误报, 交班时页面还是旧的
        #    campaign_stage -> 相位进了 stage_list, 而真实画面已经是任务大厅
        #    -> 对着大厅等关卡列表 4001 tick）
        if st.page in ("task_hall", "lobby"):
            self.goto("enter", "画面在任务大厅/大厅, 相位退回 enter 重进")
            return wait("退回 enter")
        if (st.page in ("facility", "unknown")
                and obs.has(V.HUB_CAMPAIGN, 0.45)
                and not obs.has([V.STAGE_NORMAL_SEL, V.STAGE_HARD_SEL,
                                 V.STAGE_NORMAL, V.STAGE_HARD, V.SWEEP_BATCH],
                                0.45)):
            self.goto("enter", "画面其实是任务大厅(判成 facility), 退回 enter")
            return wait("退回 enter")
        # 弹窗开没开也看**内容证据**(页签 0.99 稳), 不只等页面签名 --
        #    复打版式上 任务开始 只有 0.22, 光靠签名会卡死(2026-08-13 实录 4001 tick)
        if (st.page == "stage_popup"
                or obs.has([V.TAB_COMMAND_SEL, V.TAB_GUIDE_SEL], 0.45)):
            # 禁只认**自己点入場开的**弹窗。残留/用户手开的弹窗可能是别的关
            #    (08-13 实录: 设备上留着 2-4 的弹窗, 配置却是 2-2) -- 在别人的
            #    弹窗上点 任務開始 = 进错关烧 AP, 叉掉回列表重选才是对的
            if not self.state.get("opened_popup"):
                x = obs.find(V.CLOSE_X, 0.55)
                if x is not None:
                    return tap_box(x, "残留的关卡弹窗(不是本轮开的) -- 叉掉重选")
                return wait("残留弹窗, 等叉叉检出")
            self.goto("popup", "关卡弹窗开了")
            return wait("进相位 popup")
        # 页签对齐目标: 配置以 H 开头才去 Hard, 否则必须在 Normal
        #    （2026-08-13 实帧: Normal 没打完 Hard 整列 `入场键没解锁`,
        #    上一轮的 Hard 页签还留在屏上, 不切回去会干等）。
        want_hard = self.state["stage"].startswith("H")
        on_hard = obs.has(V.STAGE_HARD_SEL, 0.45)
        if want_hard and not on_hard:
            hd = obs.find(V.STAGE_HARD, 0.45)
            if hd is not None:
                return tap_box(hd, "切到 Hard 页签", expect=(V.STAGE_HARD_SEL,))
            return wait("找 Hard 页签")
        if not want_hard and on_hard:
            nm = obs.find(V.STAGE_NORMAL, 0.45)
            if nm is not None:
                return tap_box(nm, "切回 Normal 页签",
                               expect=(V.STAGE_NORMAL_SEL,))
            return wait("找 Normal 页签")
        # **Hard 永远 3 关; Hard 全锁 = Normal 没打完**（用户 2026-08-13 口述
        #    规则）。全锁列直说, 别在这一页耗。
        if (not obs.has(V.STAGE_ENTER, 0.45)
                and obs.has(V.STAGE_ENTER_LOCKED, 0.45)
                and self.hold("all_locked", 20)):
            return self.finish(Outcome.BLOCKED,
                               "这一列的入場全是锁的 -- Hard 全锁说明 Normal "
                               "没打完, 先推 Normal")
        # 行模型（用户拍板:「每一关都有入场以及对应的关号」）:
        #    行 = 星标(得星_0/得星_3) + 左上关号文本 + 右侧入場键。
        #    没配置 -> 打 `得星_0` 那行(下一关); 配置了 -> 逐行读号找那一行
        #    （复打已通关卡用: 测试走格子闭环拿已 3 星的关当靶, 零进度风险）。
        anchor = self.state.get("row_anchor")
        if anchor is None:
            stars = obs.all([V.STAR_0, V.STAR_3], 0.45)
            cfgd = self.state.get("stage")
            hard = on_hard
            cands = [b for b in stars if b.cls == V.STAR_0] if not cfgd else stars
            reads = []
            hit = None
            cache = self.state.setdefault("row_reads", {})
            for sb in cands:
                key = round(sb.cy, 2)
                if key in cache:
                    got = cache[key]
                else:
                    h = max(sb.y2 - sb.y1, 0.015)
                    rect = (sb.x1 - 1.25 * h, sb.y1 - 4.2 * h,
                            sb.x1 + 5.0 * h, sb.y1 + 0.4 * h)
                    got = self._parse_stage(R.digits(obs.frame, rect), hard)
                    cache[key] = got
                if got is None:
                    continue
                reads.append(got)
                if not cfgd or got == cfgd:
                    hit = (sb, got)
                    break
            if hit is None:
                # **区域对齐先于列表滚动**: 关卡列表按区域(Area)分页, 列表滚动
                #    只在区域内有效 -- 目标章号和读到的章号不同时, 滚 100 次也
                #    到不了, 要点左右切换箭头换区（2026-08-13 live: 配 H2-1
                #    卡在 Area3 的 Hard 列滚了 3 次, 用户当场抓到; `左切换`
                #    在本页实测 0.97, 零新 cls）。方向由读到的章号决定。
                if cfgd and reads:
                    want_ch = int(cfgd.lstrip("H").split("-")[0])
                    seen_ch = sorted({int(r.lstrip("H").split("-")[0])
                                      for r in reads})
                    if want_ch not in seen_ch:
                        hops = self.state.get("area_hops", 0)
                        # 预算按**距离**算, 不写死(08-30 实锤: 30 区回 3 区要
                        #    27 跳, 写死 8 走到 16 区就"交人")。首读时把
                        #    |当前-目标|+4 记下来当预算; 读不出首距才退 8。
                        if "hop_budget" not in self.state and seen_ch:
                            self.state["hop_budget"] = abs(seen_ch[0] - want_ch) + 4
                        if hops >= int(self.state.get("hop_budget", 8)):
                            return self.finish(
                                Outcome.UNKNOWN,
                                f"切了 {hops} 次区域还没到第 {want_ch} 区"
                                f"（现在读到 {reads}）-- 交人看")
                        # 上一跳还没落地(区号读数仍= 出发值)就不许再跳:
                        #    08-30 实锤双发 right 都读"第 3 区", 3 直接过冲 5。
                        #    区号变化就是切区的进度证据; 60 帧没动 = 被吞, 重点。
                        hf = self.state.get("hop_from")
                        if hf is not None and seen_ch and seen_ch[0] == hf:
                            if self.bump("hop_settle") < 60:
                                return wait(f"切区后区号还停在 {hf} -- 等落地")
                        self.state["hop_settle"] = 0
                        left = want_ch < seen_ch[0]
                        arr = obs.find(V.ARROW_LEFT if left else V.ARROW_RIGHT,
                                       0.40)
                        if arr is None:
                            if self.hold("no_area_arrow", 30):
                                return self.finish(
                                    Outcome.UNKNOWN,
                                    f"目标在第 {want_ch} 区(当前第 {seen_ch[0]} 区), "
                                    f"区域切换箭头检不出 -- 交人看")
                            return wait("目标在别的区域, 等区域切换箭头")

                        def _hopped(k=hops + 1, frm=seen_ch[0]):
                            # 换区后行位置不变数字全变 -- 读号缓存/共识/滚动
                            #    计数全部作废, 重来
                            self.state["area_hops"] = k
                            self.state["row_reads"] = {}
                            self.state["scrolls"] = 0
                            self.state["hop_from"] = frm
                            self.state["hop_settle"] = 0
                            self.state.pop("stage_vote", None)
                        _a = tap_box(arr,
                                     f"目标 {cfgd} 在第 {want_ch} 区, 当前第 "
                                     f"{seen_ch[0]} 区 -- 切区域（第 {hops + 1} 次）",
                                     post=_hopped)
                        # 屏上区号就是进度证据 -- 连发闸看它复位(见 gate.py)
                        _a.progress = f"area:{seen_ch[0]}"
                        return _a
                # 目标不在可视区 -> **先扫后滑**: 按读到的号判方向翻列表
                #    （2026-08-13 live: 列表自动归位在最新进度, 配置 2-1 时
                #    可视区只有 2-2..2-5 -- 目标在上面, 要往回翻）。
                #    方向从**读到的数字**推, 几何从星标行距推, 零写死。
                # auto 模式(没配置)同样要翻: 通关回来列表停在已清行附近,
                #    得星_0 可能在视野下方 -> 往后翻找（当天实锤: 2-1 清完
                #    视野里全是 3 星行, 读到 无 直接 UNKNOWN 收工）。
                if (not cfgd) and stars and self.state.get("scrolls", 0) < 6:
                    ys = sorted(b.cy for b in stars)
                    rowh = 0.16
                    if len(ys) >= 2:
                        gaps = [b - a for a, b in zip(ys, ys[1:]) if b - a > 0.05]
                        if gaps:
                            rowh = sorted(gaps)[len(gaps) // 2]
                    x = sorted(b.cx for b in stars)[len(stars) // 2]
                    y0 = ys[len(ys) // 2]
                    y1 = max(0.10, y0 - rowh * 2.5)
                    n = self.state.get("scrolls", 0) + 1

                    def _sc2(k=n):
                        self.state["scrolls"] = k
                        self.state["row_reads"] = {}
                        self.state.pop("stage_vote", None)
                    from routing_v2.act.action import swipe as _swipe
                    return _swipe(x, y0, x, y1,
                                  f"视野里没有得星_0 行 -- 往后翻找下一关"
                                  f"（第 {n} 次）", post=_sc2)
                if cfgd and reads and self.state.get("scrolls", 0) < 6:
                    def _k(t):
                        a, b = t.lstrip("H").split("-")
                        return (int(a), int(b))
                    up = _k(cfgd) < min(_k(r) for r in reads)
                    ys = sorted(b.cy for b in stars)
                    rowh = 0.16
                    if len(ys) >= 2:
                        gaps = [b - a for a, b in zip(ys, ys[1:]) if b - a > 0.05]
                        if gaps:
                            rowh = sorted(gaps)[len(gaps) // 2]
                    x = sorted(b.cx for b in stars)[len(stars) // 2]
                    y0 = ys[len(ys) // 2]
                    y1 = min(0.92, y0 + rowh * 2.5) if up else max(0.10, y0 - rowh * 2.5)
                    n = self.state.get("scrolls", 0) + 1

                    def _sc(k=n):
                        self.state["scrolls"] = k
                        self.state["row_reads"] = {}
                        self.state.pop("stage_vote", None)
                    from routing_v2.act.action import swipe as _swipe
                    return _swipe(x, y0, x, y1,
                                  f"目标 {cfgd} 不在可视区({'/'.join(reads)}) -- "
                                  f"往{'前' if up else '后'}翻列表（第 {n} 次）",
                                  post=_sc)
                if self.hold("stage_ocr", 30):
                    return self.finish(
                        Outcome.UNKNOWN,
                        f"逐行读号没找到目标关（配置 {cfgd or '自动'}, "
                        f"读到 {reads or '无'}）-- 不进错关")
                return wait("逐行读关号中")
            sb, got = hit
            # **两帧共识才算读到**（用户:「进的太快又没读到关卡号」）:
            #    列表刚弹出/惯性滚时单帧 OCR 是孤证, 连续两帧同号才放行。
            if self.state.get("stage_vote") != got:
                self.state["stage_vote"] = got
                return wait(f"读到 {got}, 等下一帧复读确认")
            ans = grid.load_answer(got)
            if ans is None:
                act = self._fail_or_skip(f"没有 {got} 的答案文件")
                if act is not None:
                    return act
                return wait(f"跳过 {got}, 改打 {self.state['stage']}")
            self.state.update(stage=got, answer=ans,
                              row_anchor=(sb.cx, sb.cy))
            g = self._capability_gate()
            if g is not None:
                return g
            if self.state.get("stage") != got:
                return wait(f"跳过 {got}, 改打 {self.state['stage']}")
            self.log(f"锁定目标关 = {got}（{len(ans['rounds'])} 回合）")
            anchor = (sb.cx, sb.cy)
        rows = obs.all(V.STAGE_ENTER, 0.45)
        same = min(rows, key=lambda b: abs(b.cy - anchor[1]), default=None)
        if same is not None and abs(same.cy - anchor[1]) < 0.05:
            # 点之前重核行号: 锁行可能发生在过区动画的瞬帧上(08-30 实锤:
            #    扫过 4 区的半帧锁了 4-1, 屏面落定 3 区) -- 按过期锚点点
            #    入場 = 进错关烧 AP。星标定位锚行, 关号重读, 对不上解锁重找。
            _stars = obs.all([V.STAR_0, V.STAR_3], 0.45)
            _sb = min(_stars, key=lambda b: abs(b.cy - anchor[1]), default=None)
            _got2 = None
            if _sb is not None and abs(_sb.cy - anchor[1]) < 0.05:
                _h = max(_sb.y2 - _sb.y1, 0.015)
                _rect = (_sb.x1 - 1.25 * _h, _sb.y1 - 4.2 * _h,
                         _sb.x1 + 5.0 * _h, _sb.y1 + 0.4 * _h)
                _got2 = self._parse_stage(R.digits(obs.frame, _rect), on_hard)
            if _got2 != self.state.get("stage"):
                if _got2 is not None or self.bump("lock_recheck") >= 30:
                    self.state.update(row_anchor=None, row_reads={},
                                      lock_recheck=0)
                    self.log(f"锚行现在读出 {_got2} != 目标 "
                             f"{self.state.get('stage')} -- 解锁重找")
                    return wait("锁行失效, 重找目标行")
                return wait("锚行关号暂读不出, 复核中")
            self.state["lock_recheck"] = 0
            act = tap_box(same, f"目标关 {self.state['stage']}（同行入場）",
                          expect=(V.TASK_START, V.TASK_START_GREY),
                          post=lambda: self.state.update(opened_popup=True))
            act.anchor_tol = 0.030      # 行内按钮容差要小于半行距
            return act
        # 锁着的行 40 帧等不到同行入場键 = 锚点悬空(瞬帧锁行/滚动把行带走)
        #    -- 解锁重找, 不许无限等(08-30: 这里原来干等了 4001 tick)。
        if self.bump("lock_wait") >= 40:
            self.state.update(row_anchor=None, row_reads={}, lock_wait=0)
            self.log("锁行 40 帧等不到同行入場键 -- 解锁重找")
            return wait("锁行失效, 重找目标行")
        return wait("目标行的入場键没检出（锁行等它）")

    # popup: 保证在「集中指挥」页签, 然后 任務開始
    def do_popup(self, obs, st):
        # 进没进地图不能只等页面签名（2026-08-13 live: 2 章地图格子检出
        #    断崖式掉到 0.77x1, `grid_quest` pred 不成立, page 落在 facility）。
        #    **屏上有任何格子 = 已经在地图上**, 低阈直判。
        if (st.page == "grid_quest"
                or obs.has([V.GRID_CELL, V.GRID_CELL_OPEN, V.GRID_START],
                           0.35)):
            self.goto("grid", "进地图了（格子在屏上）")
            return wait("进相位 grid")
        # 页签两态: 未选中的 `集中指挥` 是绿底(495), 选中是 421。
        #    在「简易攻略」页签上点開始会走简化模式 -- 必须先切过来。
        tab = obs.find(V.TAB_COMMAND, 0.45)
        if tab is not None and not obs.has(V.TAB_COMMAND_SEL, 0.45):
            return tap_box(tab, "切到「集中指揮」页签",
                           expect=(V.TAB_COMMAND_SEL,))
        start = obs.find(V.TASK_START, 0.45)
        if start is None:
            # 复打版式的黄色大按钮欠拟合(实测 0.22) -- 带 region 降阈值:
            #    弹窗右下那一片只有它一个大黄键, 别处不这么降
            start = obs.find(V.TASK_START, 0.18,
                             region=(0.55, 0.62, 1.0, 0.94))
        if start is not None:
            # 进关花 AP（非 premium）; 走格子地图的格子就是到达证据
            return tap_box(start, "任務開始（进关花 AP）",
                           expect=(V.GRID_CELL, V.GRID_CELL_OPEN))
        # 灰的 任務開始 有两个语义: 弹窗上灰 = AP 不够; **部署屏上灰 = 还没
        #    上队**（2026-08-13 live 误报实录: 人已经在地图上, 报了"AP不够"）。
        #    只有弹窗页签还在场时才许下 AP 结论。
        if (obs.has(V.TASK_START_GREY, 0.45)
                and obs.has([V.TAB_COMMAND, V.TAB_COMMAND_SEL, V.TAB_GUIDE,
                             V.TAB_GUIDE_SEL], 0.40)
                and self.hold("start_grey", 20)):
            return self.finish(Outcome.BLOCKED,
                               "弹窗里 任務開始 是灰的（AP 不够？）-- 不硬点")
        extra = "（任务资讯在）" if obs.has(V.TASK_INFO, 0.40) else ""
        return wait("等弹窗控件" + extra)

    # grid: 部署 -- 点起点上队, 然后 任務開始
    def do_grid(self, obs, st):
        # 帮助/教学弹窗**每次**进 Hard 部署屏都弹（08-13 实锤两轮, 不是
        #    一次性的）: 全屏遮罩把部署控件 conf 压到检不出, 帧上只剩
        #    顶栏 + 弹窗叉叉(0.94) -> 下面四个分支全落空, 干等到 phase_cap。
        #    判据 = 有叉叉且部署控件一个都不在（弹窗不盖屏时两者共存,
        #    不会误叉 任務資訊 之外的东西 -- 那个面板叉掉也无害）。
        if (obs.has(V.CLOSE_X, 0.55)
                and not obs.has([V.TASK_START, V.TASK_START_GREY, V.SORTIE],
                                0.35)):
            x = obs.find(V.CLOSE_X, 0.55)
            return tap_box(x, "部署屏被弹窗盖住(有叉叉无部署控件) -- 叉掉")
        # 点了起点会弹编队页(出击键) -- 相位机下页面 handler 不跑, 在这处理
        if st.page == "formation" or obs.has(V.SORTIE, 0.45):
            s = obs.find(V.SORTIE, 0.45)
            if s is not None:
                return tap_box(s, "编队确认: 出击（队伍上到起点格）",
                               expect=(V.TASK_START, V.TASK_START_GREY))
            return wait("编队页, 等出击键")
        # **真开局的事实 = PHASE結束 出现**（任務開始被 HUD 替掉）。
        #    2026-08-13 live 实锤: 任務開始那一发 tap 发出去了但游戏没收
        #    (面板收起动画期), 而 post 挂在"发出"上 -> flow 已经在 walk 相位
        #    "走"了, 屏上任務開始还亮着 -- 数意图不数事实在新代码里复发。
        if obs.has(V.PHASE_END, 0.40) or obs.has(V.PHASE_AUTO_ON, 0.40):
            # 航位推算的锚点: 本次部署时已走到第几回合（多区域地图第二次
            #    部署后, 累加方向只从这里开始算）
            self.state["deploy_round0"] = self.state["round_i"]
            sk = int(self.state.pop("pending_skip", 0) or 0)
            if sk and self.state.get("deployed_fresh"):
                self.log(f"skip_rounds={sk} 配置了, 但本局是全新部署"
                         f"(自己点的起点上队) — 忽略, 从第 1 步走")
            elif sk:
                self.state["round_i"] = sk
                self.log(f"断点续走: 跳过前 {sk} 步, 从第 {sk + 1} 步开始")
            self.goto("walk", "PHASE 控件出现 = 真开局了")
            return wait("进回合")
        start_btn = obs.find(V.TASK_START, 0.45)
        if start_btn is not None:
            # 严格契约: 等 PHASE 控件出现才算开了; 没收下会超时自动重发
            return tap_box(start_btn, "任務開始（部署完成）",
                           expect=(V.PHASE_END, V.PHASE_AUTO_ON,
                                   V.PHASE_AUTO_OFF))
        if obs.has(V.TASK_START_GREY, 0.35):
            # 还没上队: 点起点格（黄 = 可部署）。起点降到 0.35 -- 2 章地图
            #    整族 conf 断崖（见下面那条 UNKNOWN 的理由）。
            sp = obs.find(V.GRID_START, 0.35)
            box = self.state.get("start_box") or (
                sp and (sp.cx, sp.y1, sp.y2))
            if box is not None:
                # 落点取**框的上三分之一** -- 2-5 实帧: 起点框重心偏下
                #    (cy=0.483 而 START 六边形贴图在 y 0.30-0.42), 按框心点
                #    会打在贴图下沿外, 选队面板不弹（两轮 fail-closed 的根因）。
                cx, y1, y2 = box
                anchor = sp if sp is not None else obs.find(
                    V.TASK_START_GREY, 0.35)
                self.state["deployed_fresh"] = True
                act = tap_box(anchor, "点起点格上队(框上1/3处)",
                              expect=(V.SORTIE,))
                act.x, act.y = cx, y1 + 0.30 * (y2 - y1)
                return act
            if self.hold("no_start_cell", 40):
                # fail-closed 的**诚实版本**: 不是 AP、不是 bug, 是感知在这章
                #    地图上不够用（2-5 实测: 起点 0 检出/格子 0.77x1/敌方 3 出 1
                #    -- 训练素材全来自 1 章同一张图, [[grid_quest_baah]] 的债）。
                #    这一轮采到的帧就是补这个洞的料。
                return self.finish(
                    Outcome.UNKNOWN,
                    "部署屏上起点格检不出 -- 这章地图的走格子族泛化不足, "
                    "已采帧待标注, 不瞎点")
            return wait("找起点格（这章地图检出弱, 多看几帧）")
        return wait("等部署界面")

    # walk: 按答案 rounds 逐回合走
    def do_walk(self, obs, st):
        # 战斗期: 时钟重置(一场 2-4 分钟), 只管把 AUTO 打开
        if st.page == "battle":
            self._phase_t0 = self.ticks - 1
            self._wt_clear()
            off = obs.find(V.BATTLE_AUTO_OFF, 0.50)
            if off is not None and self.pending(f"auto{self.state['battles']}"):
                return tap_box(off, "AUTO 是关的 - 打开（状态钮只在检出关态时才点）",
                               once=f"auto{self.state['battles']}",
                               expect=(V.BATTLE_AUTO_ON,))
            return wait("战斗中（AUTO）")
        if st.page == "battle_result":
            # 战斗计数在 observe() 按页面边沿记, 不挂在这一发的 post 上
            cf = obs.find(V.CONFIRM, 0.45)
            if cf is not None:
                return tap_box(cf, "战斗结算: 確認")
            return wait("等结算確認")
        if st.page == "campaign_stage":
            # 走完之前就回到列表 = 关卡打完被送出来了
            self.goto("result", "回到关卡列表")
            return wait("进相位 result")

        plan = self._plan()
        if self.state["round_i"] >= len(plan):
            self.goto("result", "答案回合走完了")
            return wait("等结算")
        # 真多区域地图（H1-2 实测有）: 计划没走完游戏就弹回部署屏 --
        #    重新部署后**继续同一份 rounds**。禁数字键不是"区域"（BAAH 官方
        #    grid_solution_format.json: 数字键=按目标区分的**备选解法**,
        #    一份 fight_plan 覆盖整关）; 中途部署是游戏自己的节奏, 别换计划。
        # 只有走过至少一步才可能是真重部署 -- 开局 任務開始 刚点完、PHASE
        #    还没渲染出来的几帧会让 TASK_START 残留触发假重部署(H2-2 实录
        #    walk->grid->walk 弹跳, 无害但脏)
        if ((self.state["round_i"] > 0 or self.state.get("issued"))
                and (st.page == "formation" or obs.has(V.SORTIE, 0.45)
                     or (not obs.has(V.PHASE_END, 0.40)
                         and obs.has([V.TASK_START, V.TASK_START_GREY], 0.40)
                         and self.hold("redeploy", 6)))):
            self.state.update(issued=False, cycling=False, start_box=None,
                              pe_absent=0, bind_last=None,
                              pre_vec=None, moved_t=None, moved_frames=0)
            self.state["cell_acc"] = []
            self._wt_clear()
            self.goto("grid", f"回合 {self.state['round_i'] + 1} 前被要求重新部署"
                              f"（多区域地图, 部署后继续同一份答案）")
            return wait("重新部署")
        # 走位中弹窗盖屏（教学/帮助族）: 有叉叉且 PHASE 控件全不在 = 被盖住,
        #    叉掉再走。放在 page 判定之前 -- 弹窗一盖, 页面身份多半也掉了
        if (obs.has(V.CLOSE_X, 0.55)
                and not obs.has([V.PHASE_END, V.PHASE_AUTO_ON,
                                 V.PHASE_AUTO_OFF, V.CONFIRM], 0.40)):
            x = obs.find(V.CLOSE_X, 0.55)
            return tap_box(x, "走位中被弹窗盖住(有叉叉无PHASE控件) -- 叉掉")
        if st.page != "grid_quest":
            # 过场/动画通常几秒; 页面身份长时间不回来 = 感知或编排出事了,
            #    别静默空转到 phase_cap（4000 tick 在慢速率下是十几分钟）
            if self._overdue("offpage", 120):
                return self.finish(
                    Outcome.UNKNOWN,
                    f"走位中在非地图页面滞留超 120s（page={st.page}）-- 交人看")
            return wait(f"过场/动画（page={st.page}）")
        self._wt_clear("offpage")

        # **回合时钟 = 相位循环**（镜头无关的事实）: PHASE結束 消失(敌方回合/
        #    战斗把 HUD 藏起来) 再出现 = 新回合开始。坐标相等那套到位判据
        #    已废 -- 镜头一动(开局居中/回合间/战斗前后)它就假到位（05 轮实锤:
        #    两回合"到位"在同一坐标, 队伍没真动）。
        #    前提: PHASE 自动结束已勾（walk 入口保证, 下面）。
        pe = obs.has(V.PHASE_END, 0.40)
        if pe and obs.has(V.PHASE_AUTO_OFF, 0.45):
            # 状态钮只在检出"关"态时才点（和战斗 AUTO 同款纪律）
            ao = obs.find(V.PHASE_AUTO_OFF, 0.45)
            return tap_box(ao, "勾上 PHASE 自动结束（回合时钟靠相位循环）",
                           expect=(V.PHASE_AUTO_ON,))
        if self.state.get("issued"):
            if not pe:
                # cycling 由 observe() 按连续 3 帧缺席判定（页面无关）,
                #    这里不再单帧置位 -- 单帧漏检当循环会造成假回合推进
                # 敌方回合几秒~几十秒, 敌方打我方会走上面的 battle 分支;
                #    5 分钟不回来 = 相位时钟前提破了, 交人看
                if self._overdue("cycle_back", 300):
                    return self.finish(Outcome.UNKNOWN,
                                       "落子后 PHASE 控件消失超 300s 没回来 -- 交人看")
                return wait("敌方回合/战斗中（PHASE 控件不在）")
            if self.state.get("cycling"):
                self.state.update(issued=False, cycling=False,
                                  pe_absent=0, bind_last=None,
                                  pre_vec=None, moved_t=None, moved_frames=0)
                self.state["round_i"] += 1
                self.state.pop("hold:move_wait", None)
                self._wt_clear()
                self.log(f"相位循环完成 - 进回合 {self.state['round_i'] + 1}")
                return wait("新回合")
            # 落子发了但相位一直没走 = 那一发没被游戏收下 -> 有界重发
            #    （150 tick 是 live 验过的值: 开局动画期实测第 4 发才被收下）
            # 08-15 3-2 replay: observe 已记下 moved_t（人走到了目标格）,
            #    但 PHASE 不闪、6s 超时又输给 150 tick; 重发走 _issued()
            #    清掉 moved_t, 再点自己站的格没有新位移, 3 次后谎报没走动。
            #    位移已确认 = 游戏收下了这一步, 禁止重发, 等 observe 置 cycling。
            if self.state.get("moved_t"):
                return wait("等相位循环（位移已确认）")
            if self.hold("move_wait", 150):
                n = self.bump(f"reissue:{self.state['round_i']}")
                if n > 3:
                    return self.finish(Outcome.UNKNOWN,
                                       f"回合 {self.state['round_i'] + 1} 落子 3 次"
                                       f"都没走动 -- 交人看")
                self.state["issued"] = False
                self.log(f"落子后相位 150 tick 没循环 - 重发（第 {n} 次）")
            return wait("等相位循环")
        if not pe:
            if self._overdue("no_phase", 150):
                return self.finish(Outcome.UNKNOWN,
                                   "150s 没等到我方回合（PHASE 控件） -- 交人看")
            return wait("等我方回合（PHASE 控件出现）")
        self._wt_clear("no_phase", "cycle_back")

        # 我方回合, 发本回合的落子。绑格**只用 我方 身体框** -- 队伍箭头浮在
        #    头顶约 1.6 行高, 拿它绑格会把身后的格子认成所在格（探针实锤:
        #    箭头绑到灰起点, 而队伍明明站在下一格）。
        # 绑格锚定链: **箭头 -> 正下方最近的我方框 -> 该框正下方的格子**。
        #    只用 我方 会中敌我单向混淆的枪（memory battle_side_confusion:
        #    敌->我 22.5%; 本关实锤: 绑回灰起点因为某个敌方立绘被检成我方）;
        #    只用箭头会把身后的格子认成所在格（箭头悬高不定, 0.5-1.6 行波动）。
        #    箭头全场唯一属于我队 -> 拿它筛掉不在其正下方的假我方框。
        cs = grid.cells(obs, 0.35)
        stp = grid.steps(cs)
        if len(cs) < 2 or stp is None:
            if self._overdue("no_cells", 75):
                return self.finish(Outcome.UNKNOWN,
                                   "75s 内格子检出不足 2 个 -- 这章地图感知不足, "
                                   "已存帧待标注, 不瞎点")
            return wait("格子检出不足, 等一帧")
        self._wt_clear("no_cells")
        dx, dy = stp
        # **航位推算优先**: 当前逻辑格 = 起点地标 + 本区已确认执行的答案方向
        #    累加（相机不变, 不依赖我方立绘检出）。H2-2 r2 实锤的病: 单位自己
        #    的格子被立绘挡住没检出, below() 就近绑到隔壁起点格 -> "right-down"
        #    解析出来的目标 = 自己站的格 -> 8 发全点在自己脚下(点自己=取消选中,
        #    游戏无响应)。立绘绑定只做起点检不出时的兜底。
        cur = None
        unit = self._bind_unit(obs)
        sp_lm = obs.find([V.GRID_START, V.GRID_START_GREY], 0.35)
        # 已执行的回合里有 portal = 队伍被传送过, 起点地标 + 方向累加的
        #    航位推算从那一步起就不再描述真实位置 -- 整条链作废, 落到
        #    单位绑定兜底(箭头->我方框->格, 自带两帧共识)。
        _teleported = any(
            m.get("do") == "portal"
            for i in range(self.state.get("deploy_round0", 0),
                           min(self.state["round_i"], len(plan)))
            for m in plan[i])
        if sp_lm is not None and not _teleported:
            ex, ey = sp_lm.cx, sp_lm.cy
            for i in range(self.state.get("deploy_round0", 0),
                           self.state["round_i"]):
                mv = [m for m in plan[i] if m.get("do") == "move"]
                if mv and mv[0]["dir"] in grid.DIRS:
                    mul = grid.DIRS[mv[0]["dir"]]
                    ex, ey = ex + mul[0] * dx, ey + mul[1] * dy
            near = min(cs, key=lambda c: (c[0] - ex) ** 2 + (c[1] - ey) ** 2,
                       default=None)
            if (near is not None
                    and (near[0] - ex) ** 2 + (near[1] - ey) ** 2
                    < (0.6 * dx) ** 2):
                cur = near
        if cur is None:
            if unit is None:
                if self._overdue("no_unit", 75):
                    return self.finish(Outcome.UNKNOWN,
                                       "75s 内起点航位和队伍箭头都拿不到 -- "
                                       "感知不足, 已存帧待标注, 不瞎点")
                return wait("等 起点地标或队伍箭头（绑当前格）")
            self._wt_clear("no_unit")
            cur = grid.below(unit, cs, dx)
            if cur is None:
                if self._overdue("no_bind", 75):
                    return self.finish(Outcome.UNKNOWN,
                                       "75s 内队伍绑不到格子 -- 交人看")
                return wait("队伍绑不到格子")
            self._wt_clear("no_bind")
        else:
            self._wt_clear("no_unit", "no_bind")
        # **绑格要连续两帧同格共识才许落子**（08-13 H2-1 live 实锤: 迷雾光照下
        #    道具/敌方会闪检成 我方, 真我方恰好漏检的帧里「单我方框无歧义」
        #    兜底信了独苗假框 -> cur 绑进雾区 -> right 解析到迷雾格连点三发）。
        #    单帧是孤证 -- 假框闪一帧就消失, 连续两帧绑同一格基本只有真身。
        last = self.state.get("bind_last")
        self.state["bind_last"] = cur
        if (last is None
                or (cur[0] - last[0]) ** 2 + (cur[1] - last[1]) ** 2
                > 0.03 ** 2):
            return wait("绑格首帧, 等下一帧共识")
        moves = [m for m in plan[self.state["round_i"]]
                 if m.get("do") in ("move", "portal")]
        if len(moves) != len(plan[self.state["round_i"]]) or len(moves) != 1:
            # 预检在进关前就该拦掉 exchange/多队 -- 走到这说明预检
            #    漏了或答案被人改过, 照旧 fail-closed
            return self.finish(Outcome.UNKNOWN,
                               f"回合 {self.state['round_i'] + 1} 不是单队单步"
                               f"（应被进关预检拦下）-- 不瞎走")
        d = moves[0]["dir"]
        d_do = moves[0].get("do", "move")
        # **目的格候选排除自己站的格**（05 轮实锤: 绑格偏一格时 resolve 会解析
        #    回自己 -> 点自己=no-op -> 假到位）
        # 有格用格; 无格 wait/UNKNOWN, 不拿 BOSS/道具当落点（用户否掉硬搞）。
        goal = grid.resolve(cur, d,
                            [c for c in cs
                             if (c[0] - cur[0]) ** 2 + (c[1] - cur[1]) ** 2
                             > (0.4 * dx) ** 2],
                            dx, dy)
        if goal is None:
            if self.hold("no_goal", 20):
                note = grid.dir_miss_report(obs, cur, d, dx, dy)
                path = self._dump_grid_miss(obs)
                return self.finish(
                    Outcome.UNKNOWN,
                    f"方向 {d} 落不到任何检出的格子 -- 不瞎点"
                    f"（cur={cur} dx={dx:.3f}）{note}"
                    + (f" 干净帧 {path}" if path else ""))
            return wait(f"方向 {d} 暂时解析不到格子, 再看几帧")
        cell_box = min(obs.all(grid.CELL_CLS, 0.35),
                       key=lambda b: (b.cx - goal[0]) ** 2 + (b.cy - goal[1]) ** 2)
        # 黄描边(可走)只做**标注不做门**: 缺标记多半是漏检（2 章断崖）, 硬门
        #    会把能走的关拦死; 但"目标格没有黄描边=真够不着=点了没反馈"是已知
        #    最难判的失败形态, 把这个事实写进 reason, 重发耗尽时人一眼能定位
        opens = obs.all(V.GRID_CELL_OPEN, 0.30)
        marked = any((b.cx - goal[0]) ** 2 + (b.cy - goal[1]) ** 2
                     < (0.4 * dx) ** 2 for b in opens)
        # 位移证据的基线: 落子时刻 单位相对起点地标 的向量（相机不变量）。
        #    航位推算路线下立绘可能没检出 -- 那就这轮不启用位移证据(None)
        self.state["dx_est"] = dx
        pv = ((unit.cx - sp_lm.cx, unit.cy - sp_lm.cy)
              if (sp_lm is not None and unit is not None) else None)

        def _issued(dd_=d_do):
            self.state.update(issued=True, cycling=False, pe_absent=0,
                              pre_vec=pv, moved_frames=0, moved_t=None,
                              issued_do=dd_)
            self._wt_clear()
        act = tap_box(cell_box,
                      f"回合 {self.state['round_i'] + 1}: "
                      f"{'踩传送门' if d_do == 'portal' else '走'} {d} -> "
                      f"({goal[0]:.3f},{goal[1]:.3f})"
                      + ("" if marked else "（无可走标记, 若不走动多半是够不着）"),
                      post=_issued)
        # 落点校正: 可走/起点**标记**贴图重心偏下, 拿它的框心点会压到两行
        #    格子的缝上(H2-2 r2 实锤: 目标格本体被敌人立绘挡住没检出, 4 发
        #    全点在格界上游戏不收)。格子本体框在场就吸附它的框心;
        #    只有标记撑着时往上收 1/4 框高(起点框上 1/3 修正的同族病)。
        body = [b for b in obs.all([V.GRID_CELL, V.GRID_CELL_FOG], 0.35)
                if (b.cx - goal[0]) ** 2 + (b.cy - goal[1]) ** 2
                < (0.4 * dx) ** 2]
        if body:
            b0 = max(body, key=lambda b: b.conf)
            act.x, act.y = b0.cx, b0.cy
        elif cell_box.cls in (V.GRID_CELL_OPEN, V.GRID_START,
                              V.GRID_START_GREY):
            act.y = cell_box.cy - 0.25 * (cell_box.y2 - cell_box.y1)
        return act

    # result: 等结算链把我们送回关卡列表
    def do_result(self, obs, st):
        if st.page == "battle":
            self._phase_t0 = self.ticks - 1
            return wait("收尾战斗中")
        if st.page == "battle_result":
            cf = obs.find(V.CONFIRM, 0.45)
            return tap_box(cf, "结算確認") if cf is not None else wait("等結算")
        if st.page == "campaign_stage":
            n = self.state["round_i"]
            return self._after_stage_clean(
                f"{self.state['stage']} 按答案走完 {n} 回合, "
                f"战斗 {self.state['battles']} 场, 已回到关卡列表")
        cf = obs.find(V.CONFIRM, 0.40)
        if cf is not None:
            return tap_box(cf, "结算页: 確認")
        if self.phase_ticks > 900:
            return self.finish(Outcome.UNKNOWN,
                               "结算后 900 tick 没回到关卡列表 -- 交人看")
        return wait("等结算/回列表")

    def on_confirm_dialog(self, obs, st):
        cf = obs.find(V.CONFIRM, 0.45)
        return tap_box(cf, "确认") if cf is not None else wait("等確認键")
