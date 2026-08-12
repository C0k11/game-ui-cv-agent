# -*- coding: utf-8 -*-
"""主循环 —— 零 wait，对齐 fps。

    帧(scrcpy 0ms)  YOLO(23ms)  状态(连续N帧确认)  flow  三道闸  tap(32ms)

**没有一个 sleep**。推进靠"下一步的 cls 出现"，重发靠"状态连续 N 帧没变"。
   老代码里的 `sleep(2)/sleep(4)/sleep(6)` 全部删掉了 —— 用户原话：
   「点了入场 popout 识别到了再点下一步，通过识别到 cls 而不是硬给 wait time」
   「点击也要对齐 fps」。

**flow 返回 None ≠ 可以乱点**。None 的含义是"这一页不归我管"，此时由
   `_recover()` 处理，而 `_recover()` 只会在**已确认的、具名的**页面上发起
   退出动作。UNKNOWN 页面的默认行为是**什么都不做**（§A1）。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from routing_v2.act.action import Action
from routing_v2.act.gate import Gate
from routing_v2.act.ledger import Ledger
from routing_v2.config import load as load_cfg
from routing_v2.flow import nav
from routing_v2.flow.base import Ctx, Flow, Outcome
from routing_v2.flow.interrupt import Interrupts
from routing_v2.flow.registry import ALL, build
from routing_v2.percept import detect
from routing_v2.percept.device import Device
from routing_v2.percept.feed import Feed
from routing_v2.percept.observe import Observation, annotate
from routing_v2.state.machine import Machine
from routing_v2.state.pages import classify

_ROOT = Path(__file__).resolve().parents[2]


def ap_reserve_for(ap: int, split: dict, me: str, on_order: List[str],
                   floor: int = 0):
    """「我分到 p%」「我该给后面的人留多少 AP」。纯算术，好测。

    翻译成 `ap_reserve` 是整个设计的关键：各 AP flow 本来就会刷到
       `AP - reserve < 一轮的量` 收手，所以**不用改任何 flow 的逻辑**。
    返回 `(reserve, 参与分配且还没跑的 flow 列表)`；不该注入时返回 None。
    """
    rest, hit = [], False
    for nm in on_order:
        if nm == me:
            hit = True
        if hit and nm in split:
            rest.append(nm)
    if me not in rest:                 # order 里没写它  只认它自己
        rest = [me] if me in split else []
    total = sum(split[n] for n in rest)
    if total <= 0:
        return None
    later = sum(split[n] for n in rest if n != me)
    return int(round(max(0, ap - floor) * later / total)) + floor, rest


@dataclass
class RunStats:
    t0: float = field(default_factory=time.time)
    ticks: int = 0
    taps: int = 0
    frames_saved: int = 0
    halted: str = ""
    reports: List[str] = field(default_factory=list)


class Runner:
    def __init__(self, cfg: Optional[dict] = None, log: Callable = print,
                 approver: Optional[Callable[[Action, Observation], bool]] = None):
        self.cfg = cfg or load_cfg()
        self.run = self.cfg.get("run", {})
        self.log = log
        self.approver = approver
        self.device: Optional[Device] = None
        self.feed: Optional[Feed] = None
        self.machine = Machine(int(self.run.get("confirm_frames", 3)), log=log)
        self.interrupts = Interrupts(log=log)
        self.ledger = Ledger(log=log)
        self.gate = Gate(self.cfg, log=log, on_money_step=self._money_review)
        self.stats = RunStats()
        self.ctx: Optional[Ctx] = None
        self._last_seq = -1
        self._ap_replanned = False       # AP 复盘只做一次（见 run_all）
        self._shot_dir: Optional[Path] = None
        self._flow: Optional[Flow] = None
        self._recover_n = 0
        self._blank_taps = 0

    def state_blank_taps(self) -> int:
        self._blank_taps += 1
        return self._blank_taps - 1

    # ══ 启动 ═══════════════════════════════════════════════════════════
    def boot(self) -> bool:
        self.log("── routing_v2 启动 ──────────────────────────────────")
        self.device = Device(log=self.log)
        if not self.device.alive():
            self.log("[boot] adb 不通  走恢复阶梯")
            if not self.device.recover():
                self.log("[boot] adb 救不回来")
                return False
        # **总是**把 BA 拉到前台，不只是"没在跑才拉"。
        #    2026-08-08 实测：模拟器前台是**另一个游戏**（mCurrentFocus=
        #    com.bilibili.fatego），BA 进程活着但被压到后台  scrcpy attach 到的
        #    display 全黑，YOLO 零检出，bot 对着黑屏空转。
        #    monkey 对已在跑的 app 只是把它拉到前台，不会重启，代价可忽略。
        was_running = self.device.game_running()
        self.log(f"[boot] BA 进程 {'在' if was_running else '不在'}  拉到前台")
        self.device.launch_game()
        if not was_running:
            time.sleep(8.0)          # 仅此一处 sleep：等 app 冷启动，不在决策链上

        # 预热闸：冷启动的加载帧会把第一个 flow 的预算烧光（08-01 事故）
        self.log("[boot] 预热 YOLO…")
        detect.warm(("ui",))

        self.feed = Feed(serial=self.device.serial, log=self.log)
        if not self.feed.start():
            self.log("[boot] scrcpy 起不来")
            return False
        # 必须用首帧朝向标定 tap 坐标空间（`wm size` 报的是竖屏面板，直接信
        #   会让 y 飞出屏幕 —— 08-08 新架构第一次真机单步就撞上）
        fr, _, _ = self.feed.wait_new(-1, timeout=6.0)
        if fr is None:
            self.log("[boot] scrcpy 起来了但没出帧")
            return False
        self.device.calibrate(fr.shape[1], fr.shape[0])
        self.log(f"[boot] scrcpy up；设备 {self.device.serial}")

        if self.run.get("save_frames", True):
            self._shot_dir = (_ROOT / "data" / "raw_images"
                              / f"v2_{time.strftime('%Y%m%d_%H%M%S')}")
            self._shot_dir.mkdir(parents=True, exist_ok=True)

        self.ctx = Ctx(cfg=self.cfg, device=self.device, feed=self.feed,
                       ledger=self.ledger, log=self.log)
        return True

    # ══ 感知 ═══════════════════════════════════════════════════════════
    def _observe(self, frame, age, seq) -> Observation:
        h, w = frame.shape[:2]
        # 2026-08-12 live 实锤：这里原来**硬写 `("ui",)`**，把 flow 声明的
        #    `yolo = (...)` 整个无视掉。后果：`CafeFlow` 明明声明了
        #    `yolo = ("ui", "avatar")`（注释还专门写了「只跑 ui 的话
        #    `obs.find("凯伊")` 永远 None」），跑起来还是只推 ui ——
        #    而**学生名 143-394 属于 fused_avatar，build_ui_v2 明确 drop 掉了**
        #     邀请目标一个都找不到，收尾报「配置的角色没找到，未消耗邀请卷」。
        #    今晚 live 帧上 `Kei` 就躺在列表第一行，而 flow 说找不到。
        #     按当前 flow 的声明推模型；没声明就退回 ui。
        models = getattr(self._flow, "yolo", None) or ("ui",)
        boxes = detect.infer(frame, tuple(models))
        return Observation(boxes=boxes, frame=frame, seq=seq, age=age,
                           ts=time.time(), w=w, h=h)

    def _fresh(self, older: Observation) -> Optional[Observation]:
        """给 Gate 用的"最新帧"。scrcpy 0ms + YOLO 23ms，**绝不 ADB 抓帧**。"""
        fr, age, seq = self.feed.latest()
        if fr is None or seq == older.seq:
            return None
        return self._observe(fr, age, seq)

    # ══ 人审 ═══════════════════════════════════════════════════════════
    def _money_review(self, act: Action, obs: Observation) -> bool:
        if self.approver is None:
            self.log(f"    金钱步「{act.reason}」没有人审通道  拒绝")
            return False
        shot = self._save(obs, tag="MONEY", target=act)
        self.log(f"    金钱步待人审: {act.reason}（花 {act.spend or '?'}）")
        if shot:
            self.log(f"       帧证据: {shot}")
        return bool(self.approver(act, obs))

    # ══ 落盘 ═══════════════════════════════════════════════════════════
    def _save(self, obs: Observation, tag: str = "", target=None) -> Optional[str]:
        if self._shot_dir is None or obs.frame is None:
            return None
        try:
            import cv2
            import numpy as _np
            name = f"{obs.seq:07d}_{tag or 'f'}"
            # 2026-08-12 用户判断准确:「数据飞轮本来就会录进去，我怀疑
            #    **就是录制的 jpg**，你之前门控一帧一看那会可能是没设置好导致成了 png」
            #    —— 查下来比这更糟：**两种格式是反的**
            #      干净帧  `.png`       `build_ui_v2.frames_in()` 历史上只扫 jpg，
            #                             前端 datasets 也只按 jpg 计数  **整批看不见**
            #      标注渲染图  `_ann.jpg`  **反而会被扫进训练集**
            #    也就是说，把这些池接进 sources，进集的是**烧了框的渲染图**，
            #    干净帧全漏 —— 等于教模型认自己画的框
            #    （[[flywheel_label_import]] 那条 overlay 烧录陷阱的复刻版）。
            #     干净帧统一写 **jpg**（q95，肉眼无损，整条工具链才看得见）
            #      渲染图挪进 **`_ann/` 子目录**（下划线开头的目录被所有
            #        build/前端扫描排除），**绝不可能再混进训练集**。
            #    用 imencode+tofile 而不是 imwrite：路径含中文时 imwrite 会静默失败。
            fp = self._shot_dir / f"{name}.jpg"
            cv2.imencode(".jpg", obs.frame,
                         [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(str(fp))
            self.stats.frames_saved += 1
            if tag:      # 只给关键帧画锁定框，省磁盘
                ann = annotate(obs, getattr(target, "require_box", None),
                               note=f"{tag}")
                if ann is not None:
                    ad = self._shot_dir / "_ann"
                    ad.mkdir(exist_ok=True)
                    cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 85])[1] \
                        .tofile(str(ad / f"{name}_ann.jpg"))
            return str(fp)
        except Exception:
            return None

    # ══ 兜底恢复（唯一允许发起"退出类"动作的地方）════════════════════
    def _recover(self, obs: Observation, st) -> Optional[Action]:
        from routing_v2.act.action import tap_at, tap_box
        from routing_v2.state import vocab as V

        if st.flapping:
            return None                     # 判据在打架，别动手

        # ── 空屏：一个框都没有 ─────────────────────────────────────────
        # 大厅点背景会把 UI 收起来（08-08 我一发算错坐标的点击就把它收了，
        # 之后 bot 永远看不见任何 cls = 死循环）。此时点屏幕中央就能唤回；
        # 若是过场则等于推进对话；若是加载则无害。
        # 前提是 **len(boxes)==0** —— 屏上什么都没有，也就没有按钮会被误点。
        #    这跟老代码那个盲点「編輯模式」的 bug 有本质区别（那次屏上有框）。
        if st.page == "blank":
            a = nav.blank_escape(st)
            return a if (a is not None and self.state_blank_taps() < 6) else None

        if st.page == "unknown":
            # UNKNOWN 的默认行为就是**什么都不做**。
            #   只有在 UNKNOWN **持续**很久（不是转场帧）时才允许最温和的一步：
            #   关弹窗。`回大厅` 默认不给 —— 它几乎每页都在，当兜底就等于
            #   "任何看不懂的帧都把自己弹回大厅"（§A1，用户现场目击过）。
            if not self.run.get("unknown_escape", True):
                return None
            if st.frames_in_page < 200:
                return None
            # 逐级升级，**从最不破坏性的开始**。
            #   §A1 反对的是"任何一帧认不出就点回大厅"（转场帧会中招）；
            #   这里已经连续 200 帧（≈7s）都认不出，**不可能是转场帧**。
            #   顺序：取消 > 叉叉 > 返回键(上一层) > 系统返回键 > 回大厅(要配置允许)
            self._recover_n += 1
            n = self._recover_n
            c = obs.find(V.CANCEL, 0.50)
            if c is not None:
                return tap_box(c, f"recover: UNKNOWN {st.frames_in_page} 帧  取消({n})")
            x = obs.find(V.CLOSE_X, 0.60)
            if x is not None:
                return tap_box(x, f"recover: UNKNOWN {st.frames_in_page} 帧  关弹窗({n})")
            b = obs.find(V.BACK, 0.60)
            if b is not None:
                return tap_box(b, f"recover: UNKNOWN {st.frames_in_page} 帧  返回上一层({n})")
            if n <= 8:
                # 安全性统一走 `nav.back_key()`（大厅上按返回 = 退出游戏框）。
                #    第一版这里自己造了个 Action，绕过了那道闸 —— 于是它在
                #    大厅上连按了 6 次返回，把游戏关掉过一次。
                return nav.back_key(
                    obs, f"recover: UNKNOWN {st.frames_in_page} 帧，"
                         f"屏上无退出控件  系统返回键({n})")
            if self.run.get("allow_home_escape", False):
                h = obs.find(V.HOME, 0.60)
                if h is not None:
                    return tap_box(h, "recover: 配置允许回大厅兜底")
            return None
        # 具名页面但当前 flow 不管  往大厅走
        return nav.to_lobby(obs, st)

    # ══ 派发 ═══════════════════════════════════════════════════════════
    def _dispatch(self, act: Action, obs: Observation, st) -> bool:
        """返回 False = 要停整条链。"""
        # **上一发还没兑现，就不许收工/退出**（2026-08-12 通用闸）。
        #    用户要求「从头到尾每个环节都要排查按键逻辑以及打架，还有就是
        #    **看没看到下一步 cls 的逻辑**」——与其一处处补，不如在这里堵死:
        #    一晚上抓到 4 处同形 bug，形状完全一样 ——
        #      点了"开面板/弹确认框"的键  面板要几帧才出来  那几帧里 handler
        #      照常从头跑、命中不了任何分支  一路落到 `finish(CLEAN…)`
        #       **把自己刚打开的东西关掉，还报告"干净"**。
        #      （cafe 开邀请券 / craft 快速製造 / shop 切大赛 tab /
        #        战术大赛商店「選擇購買」 确认框没人点就跑了）
        #     推进闸里还挂着未兑现的契约时，`done` 一律降级成 `wait`。
        #      契约超时会自己作废（宽松 retry//6、严格 retry_frames），
        #      所以不会永远卡住 —— 只是**不许在动作悬空时下"干完了"的结论**。
        if act.kind == "done" and getattr(self.gate, "_pending", None):
            p = self.gate._pending
            self.log(f"    ⏸ 想收工（{act.reason}）但上一发 `{p.get('cls')}` "
                     f"的契约还没兑现 — 先等它，别在动作悬空时下结论")
            return True
        if act.kind in ("wait", "note"):
            if act.kind == "note":
                self.log(f"    · {act.reason}")
            return True
        if act.kind == "halt":
            self.stats.halted = act.reason
            self.log(f"HALT: {act.reason}")
            self._save(obs, tag="HALT", target=act)
            return False
        if act.kind == "swipe":
            self.log(f"    ↕ {act.reason}")
            self.device.swipe(act.x, act.y, act.x2, act.y2, act.ms)
            # 滑动也 arm 推进闸：不然一帧一发连着滑（猛滑），列表还在惯性
            #   滚动时模型没机会扫干净，眼前的目标就被滑过去了。
            self.gate.arm(act)
            # 08-10：这里原本**不执行 `act.post`** —— 于是滑动类 flow 只能把
            #    「滑了第几次 / 上次最低 y」写在 decide 里（mutate-before-ack），
            #    动作被闸吞掉时计数照样涨、防空转的基准也被污染。
            #    swipe 和 tap 一样：**真发出去了才记账**。
            if act.post is not None:
                act.post()
            return True
        if act.kind == "key":
            # 按键也要过连发闸（见 gate.dedup 里的说明）
            v = self.gate.dedup(act, st.changed, st.frames_in_page,
                                int(self.run.get("retry_frames", 25)))
            if not v.ok:
                if v.why:
                    self.log(f"    [gate] {v.why}")
                return not v.stop_flow
            self.log(f"    ⌨ {act.keycode}  {act.reason}")
            self.device.key(act.keycode)
            self.stats.taps += 1
            return True
        if not act.is_tap:
            return True

        v = self.gate.allow(act, obs, fresh=lambda: self._fresh(obs),
                            page_changed=st.changed,
                            frames_in_page=st.frames_in_page,
                            retry_frames=int(self.run.get("retry_frames", 25)),
                            last_solid=getattr(st, "last_solid", None))
        if not v.ok:
            if v.halt:
                self.stats.halted = v.why
                self._save(obs, tag="MONEYSTOP", target=act)
                return False
            if v.stop_flow and self._flow is not None:
                self._flow.finish(Outcome.UNKNOWN, v.why)
                self._save(obs, tag="SPIN", target=act)
                self._stop_flow = True
            return True

        if self.run.get("step_mode", True) and self.approver is not None:
            shot = self._save(obs, tag="STEP", target=act)
            if not self.approver(act, obs):
                self.log("    ⏸ 人工未放行 —— 停")
                self.stats.halted = "人工中止"
                return False

        ok = self.device.tap(act.x, act.y)
        self.stats.taps += 1
        # 推进契约也只在**真发出去之后**才 arm（同一条「数事实」纪律）——
        #   被闸吞掉的决策若 arm 了，闸会为一发从没发出去的点击一直等下去。
        if ok:
            self.gate.arm(act)
        # 计数只在**真的点出去之后**才 +1（数事实，不数意图）。
        #    flow 在 decide() 里自己加过的那些，全是被闸吞掉也照样涨的假数字。
        if ok and self._flow is not None:
            if act.counter:
                self._flow.bump(act.counter)
            if act.once_key:                      # 真发出去了才算"做过"
                self._flow.state[f"once:{act.once_key}"] = True
            if act.post is not None:              # 任意状态突变同理：落地才记
                act.post()
        n = (self._flow.state.get(act.counter) if act.counter and self._flow
             else None)
        self.log(f"     tap {act.target_cls or '(no-cls)'} "
                 f"@({act.x:.3f},{act.y:.3f})  {act.reason}"
                 + (f"  [{act.counter}={n}]" if n is not None else "")
                 + ("" if ok else "   tap 返回失败"))
        return True

    # ══ 主循环 ═════════════════════════════════════════════════════════
    def run_all(self) -> RunStats:
        if not self.boot():
            self.stats.halted = "boot 失败"
            return self.stats
        flows = build(self.cfg, self.ctx, log=self.log)
        self.log(f"[run] 本轮 {len(flows)} 个 flow: "
                 f"{[f.name for f in flows]}")
        queue: List[Flow] = list(flows)
        budget_s = float(self.run.get("max_minutes", 90)) * 60

        while queue:
            if time.time() - self.stats.t0 > budget_s:
                self.log(f"[run] ⏱总时长上限 {budget_s/60:.0f} 分到了，剩下的不跑了")
                for f in queue:
                    f.outcome = f.outcome or Outcome.SKIPPED
                    self.stats.reports.append(f.report())
                break
            f = queue.pop(0)
            self._run_flow(f)
            self.stats.reports.append(f.report())
            self.ledger.force_sample()          # flow 交接点强制采一次余额
            if self.stats.halted:
                for rest in queue:
                    rest.outcome = Outcome.SKIPPED
                    self.stats.reports.append(rest.report())
                break
            # flow 请求插队（event  event_shop 推算）
            req = self.ctx.bag.pop("request_flow", None)
            if req and req in ALL:
                self.log(f"[run] {f.name} 请求插入 {req}")
                queue.insert(0, ALL[req](self.ctx))
                queue.insert(1, f)              # 回来接着跑原 flow

            # **AP 复盘**（用户 2026-08-10 提议，当天就有实证）:
            #    order 是 …eventarenamaildaily_mission，而 **mail / 每日领奖
            #    才是当天 AP 的大头**（社团签到、咖啡厅溢出、任务奖励全进信箱）。
            #    08-10 实测：event 把 AP 花到 **9**  mail 领回 **219** 
            #    daily_mission  收工，**那 219 AP 原封不动留到第二天**。
            #     队列跑空后读一次 AP，够一轮就把 event 再挂回去把它花掉。
            #    只做**一次**（`_ap_replanned`）：event 自己有 AP 闸和扫荡上限，
            #      但复盘无限次就成了死循环（领奖励刷再领…）。
            #    读不出 AP 就**不复盘**（fail-closed）——宁可少刷一轮，
            #      也不要凭空多跑一趟把时间预算烧光。
            if not queue and not self._ap_replanned:
                self._ap_replanned = True
                extra = self._ap_replan()
                if extra is not None:
                    queue.append(extra)

        self._teardown()
        return self.stats

    def _ap_replan(self) -> Optional[Flow]:
        """队列跑空  看看 mail/每日领奖之后 AP 是不是又攒起来了。

        **为什么必须花掉**（用户 2026-08-10 口述的 AP 资源模型）:
           AP **240 是自然回复上限，到 240 就停止回复** —— 停在高位不是"存着"，
           是**每分钟都在亏**。240~999 只能靠领取堆进去，>999 溢出进邮箱（不丢）。
            唯一会**真损失**的形态就是"AP 卡在 240 附近不动"，所以要尽量花到低位。
        """
        import routing_v2.percept.read as R
        mods = self.cfg.get("modules", {})
        # 别写死 `event`：活动结束 / event 关掉时，AP 就**无处可花  一路涨到 240
        #    卡死  回复完全停摆**（这正是 memory `mainline_ap_routing` 那条
        #    「全仓没有『无活动自动切回推图/扫荡』分支」的代价，在这个资源模型下
        #    比原先估计的严重得多）。 按优先级找**第一个开着的** AP 消耗 flow。
        order = [str(x) for x in (self.cfg.get("ap_replan_flows")
                                  or ["event", "batch_sweep", "special_sweep"])]
        target = None
        for nm in order:
            if nm not in ALL:
                continue
            on = mods.get(nm)
            if on is None:
                on = mods.get(ALL[nm].module, False)
            if on:
                target = nm
                break
        if target is None:
            self.log("[run] AP 复盘: 没有开着的 AP 消耗 flow（活动结束了？）"
                     "  AP 会停在高位，回复停摆，**需要人来定刷什么**")
            return None
        need = int(self.cfg.get("event", {}).get("min_ap_for_sweep", 20))
        fr, age, seq = self.feed.latest()
        if fr is None:
            return None
        obs = self._observe(fr, age, seq)
        ap = R.read_topbar(obs, R.AP)
        if ap is None:
            self.log("[run] AP 复盘: 顶栏 AP 读不出  不复盘（fail-closed）")
            return None
        if ap < need:
            self.log(f"[run] AP 复盘: 现在 {ap} AP，不够一轮（需 {need}） 收工")
            return None
        self.log(f"[run] AP 复盘: 领奖后又有 {ap} AP（≥{need}）"
                 f" 再挂一轮 {target} 花掉（AP 停在 240 就不回复了）")
        return ALL[target](self.ctx)

    def _apply_ap_split(self, flow: Flow) -> None:
        """按用户配的百分比，给这个 flow 算出它该留多少 AP 给后面的人。

        **不改任何 flow 的逻辑** —— 复用现成的 `ap_reserve` 闸：
           各 flow 本来就会刷到 `AP - reserve < 一轮的量` 就收手。
           把"我分到 60%"翻译成"给后面留 40%"，就是 reserve。

        **每个 flow 开跑前实时读 AP** 而不是开跑时一次算死：中途 mail /
           每日领奖会把 AP 领回来（08-10 实测领回 219），实时读才能把它也分掉。
        AP 读不出就**不注入**，落回配置里原本的 reserve（= 老行为）。
           这里不需要 fail-closed：各 AP flow 自己都有"读不出就停手"的闸，
           在这儿再拦一道只会让人搞不清是谁停的。
        """
        import routing_v2.percept.read as R
        plan = self.cfg.get("plan") or {}
        if str(plan.get("ap_mode") or "all_in_order") != "split":
            return
        split = {str(k): float(v) for k, v in (plan.get("ap_split") or {}).items()
                 if float(v or 0) > 0}
        if flow.name not in split:
            return
        mods = self.cfg.get("modules", {})
        on_order = [nm for nm in self.cfg.get("order", [])
                    if mods.get(nm) or
                    mods.get(getattr(ALL.get(nm), "module", ""), False)]
        fr, age, seq = self.feed.latest()
        if fr is None:
            return
        ap = R.read_topbar(self._observe(fr, age, seq), R.AP)
        if ap is None:
            self.log(f"[run] AP 分配: 顶栏读不出  {flow.name} 沿用配置里的 reserve")
            return
        got = ap_reserve_for(ap, split, flow.name, on_order,
                             int(plan.get("ap_floor", 0) or 0))
        if got is None:
            return
        reserve, rest = got
        sub = self.ctx.cfg.get(flow.module or flow.name)
        if not isinstance(sub, dict):
            return
        sub["ap_reserve"] = reserve
        self.log(f"[run] AP 分配: 现在 {ap} AP，{flow.name} 占 "
                 f"{split[flow.name]:.0f}/{sum(split[n] for n in rest):.0f}  给后面 "
                 f"({'+'.join(n for n in rest if n != flow.name) or '无'}) "
                 f"留 {reserve}，自己可花 ~{max(0, ap - reserve)}")

    def _run_flow(self, flow: Flow) -> None:
        self._flow = flow
        # 预热闸同理（08-01 那条）：avatar 是 cafe/schedule 才用的第二个模型，
        #    不预热的话第一帧要现场加载权重，正好卡在热路径上。
        _m = tuple(getattr(flow, "yolo", None) or ("ui",))
        if _m != ("ui",):
            self.log(f"[run] 预热 {flow.name} 需要的模型 {_m}…")
            detect.warm(_m)
        self._apply_ap_split(flow)
        self.machine = Machine(int(self.run.get("confirm_frames", 3)), log=self.log)
        self._recover_n = 0
        self._blank_taps = 0
        self._stop_flow = False
        self._feed_healed = False        # 冻结自愈：每条 flow 给一次机会
        self.log(f"\n── flow: {flow.name} ─────────────────────────────")
        t0 = time.time()
        cap = float(self.run.get("max_minutes_per_flow", 15)) * 60
        stuck = int(self.run.get("stuck_frames", 400))
        idle = 0

        while True:
            if time.time() - t0 > cap:
                flow.finish(Outcome.UNKNOWN, f"⏱超过单 flow 上限 {cap/60:.0f} 分")
                return
            fr, age, seq = self.feed.wait_new(self._last_seq, timeout=3.0)
            if fr is None:
                self.log("[run] 取不到帧（scrcpy 断了？）")
                continue
            if seq == self._last_seq:
                continue                        # 同一帧不重复决策 = 对齐 fps
            self._last_seq = seq
            self.stats.ticks += 1

            obs = self._observe(fr, age, seq)
            st = self.machine.update(obs)

            breach = self.ledger.feed(obs, st.page, flow.name)
            if breach:
                self.stats.halted = breach
                self._save(obs, tag="BREACH")
                return

            # 死机探针（08-08 实锤）：内容 2 分钟纹丝不动 = 流死了或游戏死了。
            #    冻结帧 YOLO 照样 0.97，不拦的话 bot 会对着尸体帧"正常"决策。
            #    08-09 更正：两次 live 冻结都是 **scrcpy 流单独死**（游戏和
            #    模拟器都活着，新开流立刻正常） 先**重建 feed 自愈一次**，
            #    自愈后还冻才是真死机，才 halt 交人。
            fz = self.feed.frozen_s()
            if fz > 120.0:
                if not self._feed_healed:
                    self._feed_healed = True
                    self.log(f"屏幕内容 {fz:.0f}s 没变 — 疑似流死，重建 feed 自愈…")
                    try:
                        self.feed.stop()
                    except Exception:
                        pass
                    self.feed = Feed(serial=self.device.serial, log=self.log)
                    if self.feed.start():
                        self.ctx.feed = self.feed
                        self._last_seq = -1
                        self.log("feed 已重建，继续")
                        continue
                self.stats.halted = (f"屏幕内容 {fz:.0f}s 没变化且 feed 自愈无效"
                                     f" — 疑似游戏死机（需重启 MuMu/游戏）")
                self._save(obs, tag="FROZEN")
                return

            if st.changed:
                self._save(obs, tag=st.page)
            # 战斗中**定期录帧**（2026-08-12 用户点名:「打的这几次加成录制
            #    下来没，可以喂给最新的战斗模型」）。
            #    上面那句 `if st.changed` 是**页面变化**才存 —— 而战斗全程
            #      页面身份纹丝不动  打完 4 场加成，`battle` 页只存下 4 帧
            #      （battle_result 16 / formation 8），整场战斗等于没录。
            #     battle 页按固定 tick 间隔补录。战斗帧是 battle 模型唯一的
            #      真实素材来源（[[training_data_pipeline]] 飞轮），且**每一帧
            #      都是自己打出来的、知道上下文**，比事后翻录屏值钱得多。
            #    只录 battle，别把 lobby/加载帧也按间隔灌进来（那些 `st.changed`
            #      已经够用，多录只会让去重后的唯一帧比例更差）。
            elif st.page == "battle":
                self._battle_tick = getattr(self, "_battle_tick", 0) + 1
                if self._battle_tick % 12 == 0:
                    self._save(obs, tag="battle")

            #  全局打断优先
            act = None
            if st.interrupt:
                # 「下一章節」框点 中斷 还是 觀看 —— 由当前 flow 的意图决定
                #   （剧情挖矿要连着推，别的 flow 撞上剧情就是要逃出来）。
                self.interrupts.watch_next_chapter = bool(
                    self.ctx.bag.get("watch_next_chapter"))
                act = self.interrupts.handle(st.interrupt, obs)
            if st.interrupt != "loading":
                _d = self.interrupts.note_no_loading()
                if _d is not None:
                    self.log(f"    ⏳ 加载中持续了 {_d:.2f}s（等待时长由 cls 决定）")
            #  flow 决策
            if act is None:
                act = flow.decide(obs, st)
            #  flow 不管这一页  温和恢复（只在具名页面上）
            if act is None:
                act = self._recover(obs, st)
                idle += 1
            else:
                idle = 0

            if act is None:
                if idle % 120 == 1:
                    self.log(f"    … 没有可执行动作（page={st.page} "
                             f"raw={st.raw} 帧龄{obs.age*1000:.0f}ms）"
                             f"  屏上: {obs.summary(6)}")
                if idle > stuck:
                    flow.finish(Outcome.UNKNOWN,
                                f"在 page={st.page} 上连续 {idle} 帧没有可执行动作"
                                f" — 判据不认识这个局面，交人看")
                    self._save(obs, tag="STUCK")
                    return
                continue

            if act.kind == "done":
                flow.outcome = act.outcome
                self.log(f"   {flow.name}  {act.outcome}: {act.reason}")
                return
            if not self._dispatch(act, obs, st):
                return
            if self._stop_flow:
                self.log(f"   {flow.name}  {flow.outcome}: {flow.note_lines[-1]}"
                         if flow.note_lines else f"   {flow.name} 提前收工")
                return

    # ══ 收尾 ═══════════════════════════════════════════════════════════
    def _teardown(self) -> None:
        try:
            if self.feed:
                self.feed.stop()
        except Exception:
            pass
        dur = (time.time() - self.stats.t0) / 60
        self.log("\n══ 本轮报告 ══════════════════════════════════════")
        for r in self.stats.reports:
            self.log("  " + r)
        self.log(f"\n  用时 {dur:.1f} 分 / {self.stats.ticks} tick / "
                 f"{self.stats.taps} tap / 存帧 {self.stats.frames_saved}")
        self.log(f"  闸: {self.gate.stats}")
        # 分工体检：YOLO 主导、OCR 只读数字。ocr/tick 比值应该很低（<5%）；
        #   接近 1 就说明有人把 OCR 塞进决策热路径了。
        from routing_v2.percept.read import STATS as _RS
        _n = max(1, self.stats.ticks)
        self.log(f"  分工: YOLO 决策 {self.stats.ticks} 帧 / "
                 f"OCR 只读数字 {_RS['ocr_calls']} 次"
                 f"（{_RS['ocr_calls']*100.0/_n:.1f}% of tick, "
                 f"共 {_RS['ocr_ms']/1000:.1f}s）")
        if self.interrupts.load_count:
            self.log(f"  ⏳「加载中」{self.interrupts.load_count} 次，"
                     f"共等 {self.interrupts.load_total:.1f}s，"
                     f"均 {self.interrupts.load_total/self.interrupts.load_count:.2f}s"
                     f" —— 这是本轮**全部**的等待（其余全靠 cls 推进）")
        if self.interrupts.counts:
            self.log(f"  打断: {self.interrupts.counts}")
        self.log("\n" + self.ledger.report())
        if self.stats.halted:
            self.log(f"\n 本轮被中止: {self.stats.halted}")
        if self._shot_dir:
            self.log(f"  干净帧（飞轮素材）: {self._shot_dir}")
