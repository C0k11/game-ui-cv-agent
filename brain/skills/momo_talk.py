"""MomoTalkSkill — mine all unread MomoTalk conversations (pure-YOLO rewrite).

Verified flow (interactive probe 2026-06-01, data/_mining_probe_log.md). Mining
MomoTalk = reading unread conversations to unlock 羁绊剧情 (each story ≈ 80
pyroxene — a top free-pyroxene source).

Probe refinements over the old skill:
- After entering MomoTalk, click the 对话区域 tab (MOMO_CHAT_TAB) to reach the
  未讀訊息 list (the default name-sorted view doesn't show unread directly).
- Open a student by tapping the row's LEFT (avatar, x≈0.22), NOT the unread
  badge (x≈0.505) — tapping the badge is unreliable (probe: 莉 didn't open).
- 学生发送信息中 (MOMO_SENDING) is a TRANSIENT "student is typing" cls — it only
  flickers. WGC polls ~55fps so we catch it; we only declare a student done
  after a **wall-clock** window with NO sending / reply / goto-bond
  (2026-07-28: 原来是 tick 计数, 在 zero-wait 下缩水成 3.5s  连丢学生)。
- 前往羁绊剧情 does NOT retrigger consecutively — post-bond chatter is just
  more reply options to clear.

State machine
----
enter     lobby  NAV_MOMOTALK  MomoTalk  click MOMO_CHAT_TAB  unread list.
scan      tap the top 学生momotalk信息未读 row (via avatar-x). None  done.
dialogue  metronome: 发送信息中wait; 回复选项tap; 前往羁绊剧情story. STABLE_N
          empty frames  student done  scan next.
story     进入羁绊剧情  剧情menu跳过故事键确认键获得奖励点击继续字样  list.
exit      BTN_HOME / BTN_BACK  lobby  done.

Detectors: ui + **avatar** (2026-07-28)。未读列表点的是行头像, 而头像属于
fused_avatar 域 —— 只挂 ui 时那一下在感知层看是"半径 0.06 内零 cls = 盲拍"。
挂上后同帧 5 个头像 conf 0.99-1.00 且带中文角色名, 顺带知道正在挖谁。
"""
from __future__ import annotations

from typing import Any, Dict, List

from brain.skills.base import (
    BaseSkill, ScreenState,
    action_click, action_click_box, action_wait, action_back, action_done,
    action_swipe,
)
from brain.skills import ui_classes as UC

_CLS_CONF = 0.30
_UNREAD_LIST_REGION = (0.0, 0.15, 0.55, 0.95)   # left conversation-list panel
_AVATAR_DX = 0.28          # 头像漏检时的兜底外推量(badge.cx 左移这么多)
_AVATAR_CONF = 0.30        # fused_avatar 头像框置信度下限(实测该页 0.99-1.00)
_ROW_DY = 0.035            # badge <-> 同行头像的 cy 容差(实测行距 0.107, 取 1/3)
# Switch students ONLY when the screen has NO 学生信息回复选项 (reply) AND NO 学生发
# 送信息中 (sending) AND NO bond CTA — driven purely by cls detection, NOT a timer.
# This tiny window only bridges a 1-2 frame render gap (sending vanishes before
# the reply renders) and the weak cls flickering (sending 28f / reply 32f  an
# occasional missed frame). It is NOT a per-student cooldown — any reply/sending
# frame instantly resets it, so a still-talking student is never abandoned.
# 2026-07-28 墙钟化(tick-vs-wallclock 家族第四例, 与 craft `_COLLECT_SETTLE` /
# `_MAX_SENDING` / `_FIGHT_HOLD` 同病): 旧值 `_STABLE_EMPTY = 14` 是 **tick**。
# 注释自己写着「8 was too impatient (live 2026-06-10)」—— 那时 ~1.6s/tick,
# 8 tick=12.8s 嫌短, 14 tick=22.4s 才够。**zero-wait 上线后 0.25s/tick  14 tick
# 只剩 3.5s**, 那个修复被悄悄作废了。
# live 复现(2026-07-28, 帧证据): 连开 贵音  爱丽丝(战斗)  凯伊, **前两个各等
# 5.0-5.3s 就判 "student fully done"**, 而实测**会话面板渲染要 ~5s**(凯伊正是在
# 第 5.0s 才出 回复选项) —— 卡在边界上。事后帧: 未讀訊息仍是 (16), 贵音/爱丽丝
# 红标「2」原封不动且行未高亮 = **根本没打开过**, 不是"没东西可回"。
#
# 拆成两段, 因为两件事的时间尺度差一个量级, 合成一个数必然一头错:
#    刚点开学生, 还没见过任何会话 cls  等**渲染**(实测 5s, 给 12s)
#    已经聊过至少一轮  等**学生的消息间隙**(2026-06-10 实测可超 12.8s,
#      22.4s 验证通过  沿用 22s, 绝不为省时间收窄一个事故换来的窗口)
# 任一 reply/sending/bond 帧都会刷新计时(仍是 cls 驱动, 墙钟只定"多久算聊完")。
_OPEN_RENDER_SEC = 12.0    # 打开学生  会话面板出第一个 reply/sending 的窗口
_TAIL_EMPTY_SEC = 22.0     # 聊过之后, 静默多久算这个学生聊完了
_ROW_OPEN_CAP = 2          # re-open a still-badged row at most this many times
                           # (badge = ground truth; the cap only guards the 一花
                           # class of badges that never clear).
# 2026-07-25 墙钟化: 旧值 `_MAX_SENDING = 30` **ticks**。自主跑实测
# 0.15-0.25 s/tick(口径见 BaseSkill.mark)  真实只有 **4.5-7.5s**, 而注释自己
# 写着"真实打字 <13s" —— 判据比它要判的现象还短, 学生打字打到一半就被当成
# 误检当空聊天放弃。(我第一版用被 step_mode 停顿污染的均值算成 17.5s, 据此
# 错误地驳回了 workflow 这一条; 见 BaseSkill.mark 的口径说明。)
_MAX_SENDING_SEC = 18.0    # sending stuck this long = mis-detect (a real msg types <13s)
# Lowered detection floors for the two weak chat cls so a faint reply/sending
# frame still registers (was missing  false "done"). v6 should add samples.
_SENDING_CONF = 0.15
_REPLY_CONF = 0.18
_MAX_SCROLLS = 6           # list swipes before giving up (mine visible, then scroll down)

# 同一族全部墙钟化(2026-07-28)。旧值全是 **tick**, 在 0.25s/tick 下缩水 6.4 倍:
#   _ENTER_MAX=22    **5.5s** 就报 `momotalk unreachable`
#       这极可能正是 2026-07-27 夜那条 unreachable 的真根因: 当晚 scrcpy 断流
#         17.7s, 帧都拿不稳, 5.5s 根本不够走完"点入口页面渲染"。今天同一落点
#         bot 一发就进(它的 tap 走 AdbInput._IO_LOCK) —— 落点从来没错。
#   _DIALOGUE_MAX=60  **15s**, 比上面 22s 的聊完窗口还短  窗口永远等不满,
#         学生必被中途丢下(横跳 bug 的第二条腿)
#   _STORY_MAX=70     **17.5s** 走完 menu跳过確認奖励点击继续 整条链, 不够
#   _EXIT_MAX=14      **3.5s**, 关不掉 MomoTalk 面板就报 exit timeout
# 换算口径: 旧注释都写于 ~1.6s/tick 年代, 故按 ×1.6 还原成秒。
_ENTER_MAX_SEC = 35.0
_TAB_MAX_SEC = 20.0
_DIALOGUE_MAX_SEC = 100.0
_STORY_MAX_SEC = 112.0
_EXIT_MAX_SEC = 22.0
# 整个 skill 的墙钟预算。16 条未读 × (渲染5s + 对话 + 静默22s + 羁绊剧情跳过~30s)
# ≈ 15-20 分钟。这是**手动挖矿**skill(用户前端选择去不去), 不在每日主链上抢时间,
# 而每条羁绊剧情 ≈ 80 青辉石 —— 宁可跑久也不要丢矿。
_SKILL_BUDGET_SEC = 1500.0
# 跳过故事键/剧情menu  「是否略過此劇情?」确认框渲染的 after-ack 窗口。
# 2026-07-28 live: 没有这道闸时连点 5 次, 第二发把刚弹出的确认框又关掉 = 自锁。
# 窗口必须**同时覆盖自主跑和 step 模式两种节奏**(2026-07-28 live 教训):
# 第一版写 2.0s, 自主跑(0.25 s/tick)够用, 但 step_walk 每步要抓干净帧+跑 YOLO,
# 步间隔 **4-6s**  窗口必然已过期  在门控下**照样连发 5 次**, 看起来"修了没用"。
# 手动实测证明按钮本身没问题: 点一次 ≫ 就弹出「是否略過此劇情?」(取消0.98/确认0.98)
# —— 所以那 5 连发确实是**弹出来又被下一发关掉**的自锁。
# 放宽到 6.0s 近乎零代价: 框正常 1-2s 就渲染出来, 上面的 confirm 分支立刻接走,
# 根本等不满; 放窄却直接把一整段剧情卡死。(同 _FIGHT_HOLD 的取舍口径)
_SKIP_ACK_SEC = 6.0
# 点开学生  右侧会话面板真的换成他的窗口(实测切换 ~5s, 给 8s)。
# 必须 ≤ _OPEN_RENDER_SEC, 否则"面板没换"会被"还没渲染"那条先判掉。
_PANEL_SWITCH_SEC = 8.0
# 未知確認框点了取消  框真的关掉的 after-ack 窗口(防连发)。
_UNKNOWN_ACK_SEC = 1.5


class MomoTalkSkill(BaseSkill):
    def exit_report(self):
        """竣工判据 —— 「跑完了」和「未读清空了」是两件事。

        2026-07-27 之前只报 `momotalk complete (N)`, N 是内部计数, 谁也不知道
        屏上还剩几条未读。现在 avatar 域挂上后能报**具体名单**, 对得上账。
        """
        if not self._mined:
            return ("LEFTOVER" if self._students_done == 0 else "UNKNOWN",
                    f"一个学生都没挖到(sub={self.sub_state}, "
                    f"scroll={self._scrolls}) — 查未读badge/进入点击")
        return ("CLEAN", f"挖了 {len(self._mined)} 人: {'/'.join(self._mined)}"
                         f" (scroll {self._scrolls}/{_MAX_SCROLLS})")

    def should_run(self, screen: ScreenState) -> bool:
        # MomoTalk + bond-story mining is NOT a daily-routine auto-task — it is a
        # MANUAL, player-chosen action (user 2026-06-15: "momotalk和剧情不在每日
        # 里面, 是玩家在前端选择去不去挖矿的"). So when this skill is triggered it
        # ALWAYS runs (the player already decided to mine); never dot-gate it.
        # (The 社交入口 navbar dot is CLUB's — 社團 sign-in, owned by ClubSkill —
        # NOT MomoTalk's, so it must not gate this skill either.) The scan phase
        # exits cleanly if there happen to be no unread conversations.
        return True

    def __init__(self):
        super().__init__("MomoTalk")
        # 2026-07-28 同族最后一处: `max_ticks = 400` 也是 **tick** 预算。注释按
        # ~22 tick/学生 算的, 那是 1.6s/tick 年代(≈35s/人)。墙钟化窗口后每个学生
        # ≈ 渲染5s + 对话 + 静默22s  120-240 tick, 400 跑到第 3-4 人就超时,
        # 剩下十几个未读**静默丢掉**(且出口只报 "momotalk timeout (N)")。
        # 改成: max_ticks 只当**跑飞兜底**(不再是有效上限), 真上限走墙钟预算。
        self.max_ticks = 6000
        self._init_state()

    def _init_state(self) -> None:
        self._phase_ticks: int = 0
        self._enter_ticks: int = 0
        self._enter_t0 = None                # 进入阶段的墙钟起点(见 _ENTER_MAX_SEC)
        self._skill_t0 = None                # 整个 skill 的墙钟起点(见 _SKILL_BUDGET_SEC)
        self._seen_convo: bool = False       # 本学生的会话 cls 见到过没(选窗口用)
        self._unknown_cancel_issued: bool = False   # 未知確認框的取消已发(after-ack)
        self._skip_confirm_issued: bool = False     # 略過確認已发(等框消失才复位)
        self._tab_opened: bool = False
        self._students_done: int = 0
        self._cur_student: str = ""          # 当前正在挖的角色名(avatar cls)
        self._mined: List[str] = []          # 挖完的名单 — 竣工判据/日志用
        self._empty_streak: int = 0
        self._sending_streak: int = 0        # consecutive sending frames (mis-detect cap)
        self._scan_misses: int = 0
        self._scrolls: int = 0               # list swipes done
        self._row_opens: List[List[float]] = []  # [row-cy, opens] — re-open cap per view
        self._reply_positions: set = set()     # tapped reply spots this student (skip mis-detect repeats)
        self._reply_gone: int = 0              # consecutive frames with NO reply option
        self._story_taps: int = 0
        self._story_cut: int = 0

    def reset(self) -> None:
        super().reset()
        self._init_state()

    def _goto(self, sub_state: str) -> None:
        self.sub_state = sub_state
        self._phase_ticks = 0
        # 阶段墙钟起点 —— 上面所有 *_MAX_SEC 都读这个(裸 since() 首次返回 0.0,
        # 所以必须每次进阶段显式 mark, 不能靠"没 mark 就当 0")。
        self.mark("phase")

    def _open_count(self, cy: float) -> int:
        """How many times we've opened a student at ~this row-cy this view."""
        for entry in self._row_opens:
            if abs(entry[0] - cy) < 0.05:
                return int(entry[1])
        return 0

    def _bump_open(self, cy: float) -> None:
        for entry in self._row_opens:
            if abs(entry[0] - cy) < 0.05:
                entry[0] = cy
                entry[1] += 1
                return
        self._row_opens.append([cy, 1])

    #  page predicates
    def _on_momotalk(self, screen: ScreenState) -> bool:
        return self.find_cls(
            screen, [UC.MOMO_CHAT_TAB, UC.MOMO_CHAT_TAB_SEL, UC.MOMO_UNREAD,
                     UC.MOMO_REPLY_OPT, UC.MOMO_SENDING, UC.FAVORITE_ICON],
            conf=0.25,
        ) is not None

    def _on_unread_list(self, screen: ScreenState) -> bool:
        return (self.find_cls(screen, UC.MOMO_CHAT_TAB_SEL, conf=_CLS_CONF) is not None
                or self.find_cls(screen, UC.MOMO_UNREAD, conf=_CLS_CONF, region=_UNREAD_LIST_REGION) is not None)

    def _convo_alive(self) -> None:
        """看到任一会话 cls(reply / sending / bond CTA / 剧情后续) 就调 ——
        把"聊完了"的墙钟清零, 并 latch"这个学生的会话确实开起来过"。"""
        self._empty_streak = 0
        self._seen_convo = True
        self.mark("convo")

    def _panel_student(self, screen: ScreenState):
        """右侧会话面板**当前显示的是谁** —— 用消息气泡旁的头像(avatar 域)。

        左侧未读列表的头像固定在 cx≈0.205, 右侧会话面板的在 cx≈0.55-0.80,
        两者不会混。取该带内出现次数最多的角色名(一屏会有多条同人的气泡)。
        """
        names: Dict[str, int] = {}
        for b in (screen.yolo_boxes or []):
            if getattr(b, "model_tag", "") != "avatar":
                continue
            if b.confidence < _AVATAR_CONF or not (0.52 <= b.cx <= 0.82):
                continue
            names[b.cls_name] = names.get(b.cls_name, 0) + 1
        if not names:
            return None
        return max(names.items(), key=lambda kv: kv[1])[0]

    def _row_avatar(self, screen: ScreenState, row_cy: float):
        """未读列表**同一行**的学生头像框(fused_avatar 域, 带中文角色名)。

        判据: model_tag=="avatar" 且落在列表左侧栏(cx<0.35) 且与 badge 同行
        (|Δcy| < 半行高)。不做全屏 argmax —— 那是 840AP 事故的病根
        (find_cls 全屏最高分抢锚点)。
        """
        best = None
        for b in (screen.yolo_boxes or []):
            if getattr(b, "model_tag", "") != "avatar":
                continue
            if b.confidence < _AVATAR_CONF or b.cx > 0.35 or b.cx < 0.05:
                continue
            if abs(b.cy - row_cy) > _ROW_DY:
                continue
            if best is None or abs(b.cy - row_cy) < abs(best.cy - row_cy):
                best = b
        return best

    def _in_conversation(self, screen: ScreenState) -> bool:
        return self.find_cls(
            screen, [UC.MOMO_REPLY_OPT, UC.MOMO_SENDING,
                     UC.GOTO_BOND_STORY, UC.ENTER_BOND_STORY], conf=0.25,
        ) is not None

    def _in_story(self, screen: ScreenState) -> bool:
        return self.find_cls(
            screen, [UC.STORY_MENU, UC.STORY_SKIP, UC.STORY_SKIP_DISABLED,
                     UC.STORY_TAP_CONTINUE, UC.ENTER_BOND_STORY], conf=0.30,
        ) is not None

    #  tick
    def tick(self, screen: ScreenState) -> Dict[str, Any]:
        self.ticks += 1
        self._phase_ticks += 1

        if self._skill_t0 is None:
            self._skill_t0 = self.clock()
        _run = self.clock() - self._skill_t0
        if _run >= _SKILL_BUDGET_SEC or self.ticks >= self.max_ticks:
            self.log(f"timeout ({_run:.0f}s / {self.ticks} tick, "
                     f"{self._students_done} students: "
                     f"{'/'.join(self._mined) or '-'})")
            return action_done(f"momotalk timeout ({self._students_done})")

        # 全局: **未知的 確認+取消 框  一律取消, 绝不確認**(2026-07-28 帧实锤)
        # 实拦到一个代码完全不认识的页面: 点「進入羈絆劇情」后弹出**劇透警告框**
        #   「警告 / 本內容包含 Ex.十字神名篇結局的強烈劇透 / …
        #    **點擊確認時將前往活動頁面。** / 取消 · 確認」
        # 两个要害:
        #   按钮在 **cy≈0.913**, 而 _story 的略過框判定带是 y 0.55-0.85
        #     **没有任何代码路径处理它**, 它会一直糊在屏上; 当时 bot 在 scan 里
        #     照常要去 swipe 滑列表(列表在框后面仍被检出 0.86) = 对着遮罩瞎滑。
        #   点確認 = **跳去活动页面**, 直接把 bot 带出 MomoTalk, 而这个状态机
        #     根本不覆盖活动页  必然走失。这正是 story_mining P0.9 防的那类
        #     「導航離開」框, momo_talk 一直缺这道闸。
        # 判据(干净且可证): MomoTalk 全流程里**合法的** confirm+cancel 框只有
        # 「是否略過此劇情?」, 而它出现时屏幕右上角**一定同时有 story chrome**
        # (实测那帧: 剧情menu 0.98 + 跳过故事键 0.96)。没有 chrome 的双钮框 =
        # 未知  fail-closed 取消。
        _cancel = self.find_cls(screen, UC.BTN_CANCEL, conf=_CLS_CONF)
        _confirm = self.find_cls(screen, UC.BTN_CONFIRM, conf=_CLS_CONF)
        if _cancel is not None and _confirm is not None:
            _chrome = self.find_cls(screen, [UC.STORY_MENU, UC.STORY_SKIP],
                                    conf=0.30, region=(0.80, 0.0, 1.0, 0.30))
            if _chrome is None:
                # after-ack —— **这道闸自己第一版就漏了它**, 上线第一次 live 就被
                # step_walk 连发守卫逮到「取消连发 3 次」(第 3 发时框已经关掉,
                # 落点半径 0.06 内零 cls, 拍在 MomoTalk 面板下方的空白上)。
                # 记这一笔是因为它证明了一件事: **新写的每一个 click 分支都要先
                # 回答"这一下发出去之后凭什么知道它落地了"** —— 我今天一整天都在
                # 修这个族, 自己新写的闸照样踩。
                if self._unknown_cancel_issued:
                    _uw = self.since("unknown_cancel")
                    if _uw < _UNKNOWN_ACK_SEC:
                        return action_wait(300, f"未知框取消已发 — 等它关掉 "
                                                f"(after-ack {_uw:.1f}/"
                                                f"{_UNKNOWN_ACK_SEC:.1f}s)")
                    self.log("取消发出后未知框仍在 — 判为被吞, 重发一次")
                self.log(f"未知確認框(确认@cy{_confirm.cy:.3f} 取消@cy"
                         f"{_cancel.cy:.3f}, 无剧情chrome)  取消, 绝不確認 "
                         f"(可能是劇透警告/導航離開框)")
                self._unknown_cancel_issued = True
                self.mark("unknown_cancel")
                return action_click_box(_cancel, "未知確認框  取消(绝不確認)")
        else:
            # 框没了 = 取消真的落地了  复位, 下一个未知框才能重新触发
            self._unknown_cancel_issued = False

        # Global: bond level-up splash + reward popup.
        levelup = self.find_cls(screen, [UC.BOND_LEVELUP, UC.REGION_LEVELUP], conf=_CLS_CONF)
        if levelup is not None:
            return action_click(0.5, 0.5, f"dismiss level-up ({levelup.cls_name})")

        if screen.is_loading():
            return action_wait(700, "momotalk loading")

        if self.sub_state == "":
            self._goto("enter")

        handler = {
            "enter": self._enter,
            "open_tab": self._open_tab,
            "scan": self._scan,
            "dialogue": self._dialogue,
            "story": self._story,
            "exit": self._exit,
        }.get(self.sub_state)
        if handler is None:
            return action_wait(300, "momotalk unknown state")
        return handler(screen)

    def _enter(self, screen: ScreenState) -> Dict[str, Any]:
        self._enter_ticks += 1
        if self._enter_t0 is None:
            self._enter_t0 = self.clock()
        # skill 从**剧情屏**启动(上一次跑到一半被打断 / 用户手动停在剧情里):
        # 剧情屏既没有 momotalk cls, 也没有 home/返回键  旧码走 nav_home 的
        # "无 home/X/返回键  等待(绝不瞎按 ESC)" 一直等到 _ENTER_MAX 报
        # unreachable(2026-07-28 live 复现)。剧情屏是**我们自己认识的页面**,
        # 直接交给 story 状态用跳过链走出去, 不是"不认识的屏"。
        if self._in_story(screen):
            self.log("启动时就在剧情屏  交给 story 跳过链走出去(不报 unreachable)")
            self._story_taps = 0
            self._story_cut = 0
            self._goto("story")
            return action_wait(300, "启动即在剧情中  story")
        if self._on_momotalk(screen):
            self.log("inside MomoTalk  open 对话区域 tab")
            self._goto("open_tab")
            return action_wait(350, "entered MomoTalk")

        if screen.is_lobby():
            act = self.click_cls(screen, UC.NAV_MOMOTALK, "open MomoTalk", conf=_CLS_CONF)
            if act is not None:
                return act
            return action_wait(400, "waiting for MomoTalk entry cls")

        _el = self.clock() - (self._enter_t0 or self.clock())
        if _el > _ENTER_MAX_SEC:
            self.log(f" MomoTalk 进不去: {_el:.1f}s > {_ENTER_MAX_SEC:.0f}s "
                     f"(点击 {self._enter_ticks} tick) — 抓这一刻的帧看是入口没检出"
                     f"还是点了不响应")
            return action_done(f"momotalk unreachable ({_el:.0f}s)")
        if self.detect_screen_yolo(screen) not in (None, "Lobby"):
            return self.nav_home(screen, "momotalk recover")
        return action_wait(400, "entering MomoTalk")

    def _open_tab(self, screen: ScreenState) -> Dict[str, Any]:
        """Switch to the 对话区域 (未讀) tab so the unread list shows."""
        if self._on_unread_list(screen) or self._in_conversation(screen):
            self._goto("scan")
            return action_wait(250, "unread list ready  scan")

        # or action_suppressed(2026-07-21 mutate-before-ack 缓解): tab 点击被吞时
        # _tab_opened 已 True 会跳过重点  干等到 _TAB_MAX。被吞则重检重点。
        if not self._tab_opened or self.action_suppressed:
            tab = self.find_cls(screen, UC.MOMO_CHAT_TAB, conf=_CLS_CONF)
            if tab is not None:
                self._tab_opened = True
                self.log("clicking 对话区域 tab (MOMO_CHAT_TAB)")
                return action_click_box(tab, "open 对话区域 tab")

        if self.since("phase") > _TAB_MAX_SEC:
            # Tab cls missed but maybe already on the list — proceed.
            self._goto("scan")
            return action_wait(300, "tab timeout  scan")
        return action_wait(350, "waiting for 对话区域 tab")

    def _scan(self, screen: ScreenState) -> Dict[str, Any]:
        # Story splash surfaced mid-scan?
        if self._in_story(screen):
            self._goto("story")
            return action_wait(250, "story detected  story")

        #  Do NOT treat a lingering right-pane conversation as work. After a
        # student finishes, the right pane keeps showing its last messages
        # (reply/bond cls stay detected) — the old `_in_conversation  dialogue`
        # branch re-entered dialogue forever (total ran 1447, instant-done loop,
        # only 3 students actually mined before max_ticks). The open student's own
        # multi-turn dialogue is fully handled INSIDE _dialogue; scan only ever
        # opens a FRESH unread badge (below). No residual-conversation shortcut.

        if not self._on_momotalk(screen):
            if self._phase_ticks > 6:
                self._goto("enter")
                self._enter_ticks = 0
                return action_wait(400, "lost MomoTalk  re-enter")
            return action_wait(400, "waiting for MomoTalk UI")

        # Badge = GROUND TRUTH (user 2026-06-10: 从上到下挨个打完, 全部清空才
        # scroll). Open the TOP-MOST visible badge, top-to-bottom, re-opening a
        # still-badged row up to _ROW_OPEN_CAP times (an unfinished student's
        # badge stays until truly done; the cap only guards 一花-class badges
        # that never clear). NEVER scroll while a workable badge is visible.
        all_unread = [u for u in self.find_all_cls(screen, UC.MOMO_UNREAD, conf=_CLS_CONF)
                      if 0.0 <= u.cx <= 0.62 and 0.15 <= u.cy <= 0.95]
        fresh = [u for u in all_unread if self._open_count(u.cy) < _ROW_OPEN_CAP]
        if fresh:
            unread = min(fresh, key=lambda b: b.cy)  # top-most, strict top-down
            self._bump_open(unread.cy)
            self._scan_misses = 0
            self._empty_streak = 0
            self._sending_streak = 0
            self._reply_positions = set()
            self._reply_gone = 0
            # 新学生: 墙钟归零, 且 seen_convo=False -> 先走"等渲染"那一段窗口
            self._seen_convo = False
            self.mark("convo")
            self._goto("dialogue")
            # 真锚定优先(2026-07-28): 同一行的 fused_avatar 头像框(model_tag
            # =="avatar", 中文角色名)。旧码只按 `badge.cx - 0.28` **外推**, 在
            # ui 域看落点半径 0.06 内零 cls = 盲拍(step_walk 守卫实拦)。实测挂上
            # avatar 后同帧 5 个头像 0.99-1.00 全中, cx=0.205 而外推值 0.226 已
            # 压到框右边缘(x2=0.229) —— 外推能用但没有余量, 一旦版式微调就滑出去。
            av = self._row_avatar(screen, unread.cy)
            if av is not None:
                self._cur_student = av.cls_name
                self.log(f"open unread student 「{av.cls_name}」 "
                         f"(avatar cls {av.confidence:.2f} @{av.cx:.3f},{av.cy:.3f}, "
                         f"open #{self._open_count(unread.cy)})")
                return action_click_box(
                    av, f"open unread student 「{av.cls_name}」 (avatar cls)")
            # 兜底: 头像漏检(未训练的新角色/换装)  仍按行几何外推, 保持可用。
            self._cur_student = ""
            row_x = min(0.30, max(0.12, unread.cx - _AVATAR_DX))
            self.log(f"open unread student y={unread.cy:.3f} (头像漏检  外推 "
                     f"x={row_x:.2f}, open #{self._open_count(unread.cy)})")
            return action_click(row_x, unread.cy, "open unread student (avatar, not badge)")

        # No workable badge (none visible, or the visible ones are 一花-class
        # stale after CAP re-opens). Settle (badges may still render), then
        # SCROLL DOWN for more. Rows shift after a swipe  reset open counts.
        self._scan_misses += 1
        if self._scan_misses < 3:
            return action_wait(400, f"no workable unread (settle {self._scan_misses})")
        if self._scrolls < _MAX_SCROLLS:
            self._scrolls += 1
            self._scan_misses = 0
            self._row_opens = []   # rows shifted by the swipe
            self.log(f"visible list cleared  scroll down ({self._scrolls}/{_MAX_SCROLLS})")
            return action_swipe(0.25, 0.72, 0.25, 0.42, 700,
                                f"scroll unread list down ({self._scrolls})")
        self.log(f"no more unread after {self._scrolls} scrolls ({self._students_done} mined)")
        self._goto("exit")
        return action_wait(300, "scan complete  exit")

    def _dialogue(self, screen: ScreenState) -> Dict[str, Any]:
        # open-student 点击被吞回弹(2026-07-21 mutate-before-ack 缓解): scan 提前
        # goto dialogue, 若开启点击被吞且未进会话  回 scan 重开(_bump_open 多计
        # 1 无害, fresh 过滤仍会重开)。
        if (self.action_suppressed and self._phase_ticks <= 1
                and not self._in_conversation(screen)
                and not self._in_story(screen)):
            self.log("open-student 点击被吞  回 scan 重开")
            self._goto("scan")
            return action_wait(300, "open swallowed  rescan")
        if self.since("phase") > _DIALOGUE_MAX_SEC:
            self.log(f"dialogue timeout ({self.since('phase'):.0f}s)  scan")
            self._goto("scan")
            self._scan_misses = 0
            return action_back("dialogue timeout")

        # 面板身份闸(2026-07-28 帧实锤): 点开学生 A 之后, 右侧会话面板可能
        # **还停在上一个学生 B** 上 —— 实测点了「贵音」, 右侧面板的气泡头像全是
        # `凯伊 cx=0.717`, 连羁绊剧情概要正文写的都是 Kei。旧码不问"现在这个面板
        # 是谁的"就直接点 CTA/回复  在 B 的面板上替 A 干活:
        #   · A 的未读**永远清不掉**(点两次到 _ROW_OPEN_CAP 后被跳过)
        #   · B 被重复挖(这次侥幸 B 也确实有矿, 所以看不出来)
        # 与 event_quest「换关不关旧弹窗、只有日志 label 变了」**完全同构**:
        # **参数变了 ≠ 屏幕变了**(见 memory log_is_not_truth 的元教训)。
        # 只在**两边都认出来**时才判不符 —— 面板还没渲染出头像(names 空)属于
        #   "判不了", 交给上面的 _OPEN_RENDER_SEC 窗口, 绝不在这里瞎猜。
        if self._cur_student:
            _panel = self._panel_student(screen)
            if _panel is not None and _panel != self._cur_student:
                _w = self.since("convo")
                if _w < _PANEL_SWITCH_SEC:
                    return action_wait(300, f"面板还是「{_panel}」不是"
                                            f"「{self._cur_student}」— 等切换 "
                                            f"({_w:.1f}/{_PANEL_SWITCH_SEC:.0f}s)")
                self.log(f"面板 {_w:.1f}s 仍停在「{_panel}」而我点的是"
                         f"「{self._cur_student}」— 判开学生那一下没落地  回 scan 重开")
                self._goto("scan")
                self._scan_misses = 0
                return action_wait(300, "面板未切换  rescan")

        # Bond-story CTA (priority — the 80-pyroxene payoff).
        goto_bond = self.find_cls(screen, UC.GOTO_BOND_STORY, conf=_CLS_CONF)
        if goto_bond is not None:
            self._convo_alive()
            self.log("前往羁绊剧情")
            return action_click_box(goto_bond, "goto bond story")
        enter_bond = self.find_cls(screen, UC.ENTER_BOND_STORY, conf=_CLS_CONF)
        if enter_bond is not None:
            self._convo_alive()
            self.log("进入羁绊剧情  story")
            self._story_taps = 0
            self._story_cut = 0
            self._goto("story")
            return action_click_box(enter_bond, "enter bond story")

        #  Metronome: student typing  wait. Cap consecutive sending: a real
        # "typing" lasts a few frames; if it sticks (mis-detect) treat it as empty
        # so the student can finish instead of waiting forever.
        if self.find_cls(screen, UC.MOMO_SENDING, conf=_SENDING_CONF) is not None:
            if self._sending_streak == 0:
                self.mark("sending")       # 本波打字的起点(连续段第一帧)
            self._sending_streak += 1
            _w = self.since("sending")
            if _w < _MAX_SENDING_SEC:
                self._convo_alive()
                # A new message wave makes previously-tapped reply spots STALE —
                # the next option legitimately renders at the SAME fixed spot.
                self._reply_positions.clear()
                return action_wait(450, f"学生发送信息中 — waiting "
                                        f"({_w:.1f}/{_MAX_SENDING_SEC:.0f}s)")
            # sending stuck  mis-detect; fall through to empty handling.
        else:
            self._sending_streak = 0
            self.clear_timer("sending")

        # Reply option  tap, with DYNAMIC position-dedup. ️ The 回覆 box
        # renders at a FIXED spot, so consecutive turns reuse the same position
        # — a permanent dedup ate the 2nd+ option and ABANDONED the student
        # mid-chat (live 2026-06-10 横跳 root cause: 40 opens for ~24 students).
        # Dedup now invalidates when the option DISAPPEARS ≥2 frames (consumed)
        # or a sending wave arrives. A mis-detected static chat bubble (一花
        # class) never disappears  stays deduped  empty-streak still ends the
        # student.
        reply = self.find_cls(screen, UC.MOMO_REPLY_OPT, conf=_REPLY_CONF)
        if reply is not None:
            self._reply_gone = 0
            rpos = (round(reply.cx, 2), round(reply.cy, 2))
            # or action_suppressed(2026-07-21 mutate-before-ack 缓解): reply 点击
            # 被吞时 rpos 已进 dedup  同位被跳过  学生半途弃聊(横跳 bug 类)。
            # 被吞则重点同位。
            if rpos not in self._reply_positions or self.action_suppressed:
                self._reply_positions.add(rpos)
                self._convo_alive()
                self._sending_streak = 0
                return action_click_box(reply, "pick reply option")
            # same spot already tapped  mis-detect; fall through to empty.
        elif self._reply_positions:
            self._reply_gone += 1
            if self._reply_gone >= 2:
                self._reply_positions.clear()
                self._reply_gone = 0

        # Nothing fresh to do  empty. 判"这个学生聊完了"改用**墙钟**(见文件头
        # _OPEN_RENDER_SEC/_TAIL_EMPTY_SEC 的实测与推导), 并按"有没有见过会话"
        # 分两段 —— 合成一个数必然一头错。仍是 cls 驱动: 任一 reply/sending/bond
        # 帧都会 mark("convo") 把计时清零。
        self._empty_streak += 1
        _limit = _TAIL_EMPTY_SEC if self._seen_convo else _OPEN_RENDER_SEC
        _w = self.since("convo")
        if _w >= _limit:
            self._students_done += 1
            self._mined.append(self._cur_student or f"#{self._students_done}")
            if self._seen_convo:
                self.log(f"student 「{self._cur_student or '?'}」 聊完了 "
                         f"(静默 {_w:.1f}s ≥ {_limit:.0f}s, #{self._students_done})"
                         f"  scan next")
            else:
                # 这条要显眼: 点开了但**一个会话 cls 都没见过** = 要么点击没落地,
                # 要么这个学生真没内容。_ROW_OPEN_CAP=2 会让它再被开一次;
                # 若两次都这样, badge 还在就说明是感知/点击问题, 去翻这一刻的帧。
                self.log(f" student 「{self._cur_student or '?'}」 打开后 {_w:.1f}s "
                         f"内**没出现任何会话 cls**(reply/sending/bond) —— "
                         f"点击没落地? 还是真没内容? (#{self._students_done})")
            self._goto("scan")
            self._scan_misses = 0
            return action_wait(300, "student done  scan")
        return action_wait(350, f"dialogue settle ({_w:.1f}/{_limit:.0f}s, "
                                f"seen_convo={self._seen_convo})")

    def _story(self, screen: ScreenState) -> Dict[str, Any]:
        """Skip the bond-story cutscene  claim the ~80-pyroxene reward."""
        if self.since("phase") > _STORY_MAX_SEC:
            self.log("story timeout  scan")
            self._goto("scan")
            self._scan_misses = 0
            return action_back("story timeout")

        # Reward splash  claim, then RETURN TO DIALOGUE (not scan): the bond
        # story is mined but the student still has post-bond chatter to clear.
        # Jumping to scan here (+ row-dedup) skipped it and switched students —
        # the "剧情打完没打后续就换人" bug. dialogue clears the post-bond replies,
        # THEN scans the next student.
        cont = self.find_cls(screen, UC.STORY_TAP_CONTINUE, conf=_CLS_CONF)
        got = self.find_cls(screen, UC.GOT_REWARD, conf=_CLS_CONF)
        if got is not None or cont is not None:
            self._goto("dialogue")
            self._convo_alive()             # 剧情打完回到会话 = 确实开过
            self._sending_streak = 0
            self._reply_positions = set()   # post-bond replies are fresh options
            if cont is not None:
                return action_click_box(cont, "dismiss bond reward  post-bond chatter")
            return action_click_box(got, "dismiss bond reward (header)  post-bond")

        # Skip-confirm dialog (是否略過)  确认键.
        # 2026-07-28 live 实锤的死循环(mutate-before-ack, task#23 家族):
        #   tick=8  wait  帧未稳定(转场/滚动) — 等稳定帧: confirm story skip
        #   tick=9  click 跳过故事键                     又退回去点 skip
        # 两个毛病叠在一起:
        #   `self._story_cut = 0` **写在动作发出之前** —— 动作被【帧稳定门】吞掉
        #     后, `_story_cut` 已经是 0, 下一 tick 这条分支根本不再成立  掉回
        #     去点 跳过故事键  把刚弹出的确认框又关掉  **无限循环**
        #     (帧证据: 那一帧 确认键 0.98@cy0.725 明明就在屏上、也在判定带内)。
        #   这一步**该豁免稳定门**: 落点是对话框上的固定按钮, 不是滚动列表里的
        #     目标, 不需要"等画面停"。用**显式 `_settle_exempt`**, 绝不靠在 reason
        #     里塞"確認"骗过去(memory reason_string_as_api: 那种做法同时把
        #     same-target hold 也一起豁免掉, 是 30 青辉石近失的根因)。
        #  latch 延后到**确认框真的消失**(下方到达证据), 这里只发动作。
        if self._story_cut > 0:
            confirm = self.find_cls(screen, UC.BTN_CONFIRM, conf=_CLS_CONF, region=(0.30, 0.55, 0.85, 0.85))
            if confirm is not None:
                self._skip_confirm_issued = True
                self.mark("skip_confirm")
                _act = action_click_box(confirm, "confirm story skip")
                _act["_settle_exempt"] = True     # 只豁免稳定门, hold 仍然生效
                return _act
        # 到达证据: 点过確認 且确认框现已不在  这一段真的跳掉了  复位计数
        if self._skip_confirm_issued and self.find_cls(
                screen, UC.BTN_CONFIRM, conf=_CLS_CONF,
                region=(0.30, 0.55, 0.85, 0.85)) is None:
            self._skip_confirm_issued = False
            self._story_cut = 0
            self.clear_timer("story_cut")

        # MENU  跳过故事键 (skip ASAP — story auto-plays).
        # after-ack(2026-07-28 live 实拦, double_fire_family 第六例):
        # 旧码**没有任何 after-ack** —— 只要 跳过故事键 还在屏上, 每 tick 就再点
        # 一次。而「是否略過此劇情?」确认框要时间渲染, 第二发正好把刚弹出来的框
        # 又关掉  **自锁**。step_walk 连发守卫当场抓到 **连点 5 次**, 屏上始终
        # 只有 剧情menu + 跳过故事键 两框, 确认框一次都没留住。
        # (同一 tick 速率下 5 次 ≈ 1.5s —— 这不是"点了没反应", 是**点太快**。)
        # 状态挂在物理动作上: 发过就等帧证据(确认框出现, 上面那段接走), 窗口内不重发。
        menu = self.find_cls(screen, UC.STORY_MENU, conf=_CLS_CONF)
        skip = self.find_cls(screen, UC.STORY_SKIP, conf=_CLS_CONF)
        if skip is not None or menu is not None:
            if self._story_cut > 0 and self.since("story_cut") < _SKIP_ACK_SEC:
                return action_wait(300, f"跳过/menu 已发 — 等确认框渲染 "
                                        f"(after-ack {self.since('story_cut'):.1f}"
                                        f"/{_SKIP_ACK_SEC:.1f}s)")
            self._story_cut += 1
            self.mark("story_cut")
            if skip is not None:
                return action_click_box(skip, "跳过故事键")
            return action_click_box(menu, "open 剧情menu")

        # Skip greyed  advance narration via tap-continue.
        if self.find_cls(screen, UC.STORY_SKIP_DISABLED, conf=_CLS_CONF) is not None:
            self._story_taps += 1
            return action_click(0.5, 0.5, "advance story (skip disabled)")

        # Bond story done — the game returns to the student's chat. The left list
        # tab is still visible so _on_unread_list trips, but the RIGHT pane is the
        # student's POST-BOND chatter. Go to DIALOGUE to clear it, NOT scan. The
        # reward popup is usually eaten by the global interceptor before our
        # reward branch above fires, so THIS is the real "story done" exit — and
        # it must return to the SAME student, else we jump to the next (横跳 bug).
        if self._on_unread_list(screen):
            self._goto("dialogue")
            self._convo_alive()             # 剧情打完回到会话 = 确实开过
            self._sending_streak = 0
            self._reply_positions = set()   # post-bond replies are fresh options
            return action_wait(300, "story done  post-bond chatter (same student)")
        # 2026-07-28 live: 有些羁绊剧情放完**直接把 MomoTalk 一起关掉回大厅**,
        # 而这里的出口只认 `_on_unread_list`  在大厅上干等 `story: waiting for
        # menu/skip cls` 到 _STORY_MAX_SEC(112s) 才超时。虽然超时后能自愈(scan
        # 发现不在 momotalk  回 enter), 但白白空转近两分钟, 而 skill 总预算才
        # 1500s。回到大厅是**明确的到达证据**, 直接回 enter 重进继续挖。
        if self.detect_screen_yolo(screen) == "Lobby":
            self.log("剧情放完直接回了大厅  重进 MomoTalk 继续挖(不干等超时)")
            self._enter_ticks = 0
            self._enter_t0 = None
            self._tab_opened = False
            self._goto("enter")
            return action_wait(300, "story done  回大厅了, 重进 MomoTalk")
        return action_wait(400, "story: waiting for menu/skip cls")

    def _exit(self, screen: ScreenState) -> Dict[str, Any]:
        if self.detect_screen_yolo(screen) == "Lobby":
            self.log(f"done ({self._students_done} students: "
                     f"{'/'.join(self._mined) or '-'})")
            return action_done(f"momotalk complete ({self._students_done})")
        if self.since("phase") > _EXIT_MAX_SEC:
            return action_done("momotalk exit timeout")
        # Standard exit kit (2026-06-10): cancel-first (quit-prompt / cost
        # dialogs), then the MomoTalk close-X, then home/back cls, then PACED
        # blind ESC.
        cancel = self.find_cls(screen, UC.BTN_CANCEL, conf=0.20)
        if cancel is not None:
            return action_click_box(cancel, "momotalk exit: cancel pending dialog")
        # MomoTalk closes via its top-right X (弹窗叉叉) — ESC alone left it open
        # (live 2026-06-15: mined 未讀0 但 exit timeout 停在 MomoTalk 屏, ESC 没关).
        close_x = self.find_cls(screen, UC.BTN_CLOSE_X, conf=_CLS_CONF, region=(0.55, 0.05, 0.99, 0.25))
        if close_x is not None:
            return action_click_box(close_x, "momotalk exit: close X (弹窗叉叉)")
        home = self.find_cls(screen, UC.BTN_HOME, conf=_CLS_CONF)
        if home is not None:
            return action_click_box(home, "momotalk exit: home")
        back = self.find_cls(screen, UC.BTN_BACK, conf=_CLS_CONF)
        if back is not None:
            return action_click_box(back, "momotalk exit: back")
        if self._phase_ticks % 3 != 0:
            return action_wait(600, "exit: settle before next ESC")
        return self.nav_home(screen, "momotalk exit")
