# -*- coding: utf-8 -*-
"""离线回归 —— 不碰设备、不碰模型，纯逻辑。

为什么必须有这个（memory replay_harness）:
   老仓的回放测试**测不到跨 tick 状态**（每 tick 新建 skill），于是 after-ack
   这类改动回放出来"0 diff"是**假验证**。这里用真的 Flow 实例 + 手搓的
   Observation 序列驱动，跨 tick 状态是真的。

跑法:  py -m routing_v2.tests.test_offline
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import routing_v2.act.money as money_rules                    # noqa: E402
from routing_v2.act.action import Action                      # noqa: E402
from routing_v2.act.gate import Gate                          # noqa: E402
from routing_v2.flow.base import Ctx                          # noqa: E402
from routing_v2.flow.registry import ALL                      # noqa: E402
from routing_v2.percept.observe import Box, Observation       # noqa: E402
from routing_v2.state import vocab as V                       # noqa: E402
from routing_v2.state.machine import Machine                  # noqa: E402
from routing_v2.state.pages import classify                   # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"{'' if cond else ''} {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def B(cls, conf=0.9, cx=0.5, cy=0.5, w=0.06, h=0.04):
    return Box(cls=cls, conf=conf, x1=cx - w / 2, y1=cy - h / 2,
               x2=cx + w / 2, y2=cy + h / 2)


def O(*boxes, seq=1):
    return Observation(boxes=list(boxes), seq=seq, w=3840, h=2160)


def cfg():
    """测试**不许**读 profile.json —— 那是用户的实时配置，改一下测试就飘。
    2026-08-08 实测：profile 里把 event.order 改成 bonus_only 之后，通关阶段
    那条用例当场变红。回归测试的输入必须是自己声明的。"""
    from routing_v2.config import merged
    return merged({"event": {"order": "clear_then_bonus",
                             "clear_first_with_team": 1, "bonus_team": 2,
                             "shop_plan_before_bonus": False},
                   "bounty": {"branches": ["教室"]},
                   "jfd": {"academies": ["千年", "三一", "格黑娜"]}})


# ══ 1. 页面身份 ════════════════════════════════════════════════════════
def t_pages():
    print("\n── 页面身份 ─────────────────────────────────")
    check("大厅", classify(O(B(V.NAV_CAFE), B(V.NAV_SHOP), B(V.NAV_CRAFT))).page
          == "lobby")
    check("任务大厅", classify(O(B(V.HUB_BOUNTY), B(V.HUB_ARENA),
                                B(V.HUB_JFD))).page == "task_hall")
    check("战斗内(2个chrome)", classify(O(B(V.BATTLE_PAUSE), B(V.BATTLE_3X))).page
          == "battle")
    check("单个chrome不算战斗", classify(O(B(V.BATTLE_PAUSE))).page != "battle")

    # §A9 新活动首日：1 关开 + 4 关锁，必须仍然认作关卡列表
    day1 = O(B(V.EVENT_SHOP, cx=0.1), B(V.EVENT_TASK, cx=0.2),
             B(V.STAGE_ENTER, cy=0.30),
             *[B(V.STAGE_ENTER_LOCKED, cy=0.40 + i * 0.1) for i in range(4)])
    check("§A9 新活动首日(1开4锁)仍是关卡列表",
          classify(day1).page == "event_quest_list",
          f"实际 {classify(day1).page}")

    # §A1 转场帧：只有 回大厅/返回键的帧**不许**当场触发"点回大厅"。
    #    （它现在会被认成泛化的 `facility`，但泛化页面要待够 90 帧才准走人 ——
    #      任何转场都活不过 90 帧，所以用户现场看到的"刚要进战斗就被弹回大厅"
    #      不会复现。行为断言比页面名断言更有意义。）
    stray = O(B(V.HOME, cx=0.05, cy=0.05), B(V.BACK, cx=0.02, cy=0.05))
    from routing_v2.flow import nav as _nav
    from routing_v2.state.machine import Machine as _MM
    _m = _MM(3)
    for _ in range(3):
        _s = _m.update(stray)
    check("§A1 只有回大厅/返回键的转场帧  **不动作**",
          _nav.to_lobby(stray, _s) is None, f"page={_s.page} f={_s.frames_in_page}")
    for _ in range(95):
        _s = _m.update(stray)
    check("同一页面待够 90 帧（真卡住了） 才允许走人",
          _nav.to_lobby(stray, _s) is not None, f"f={_s.frames_in_page}")

    # 编队页必须被认出（老状态机不认它  落 STRAY  差点点返回）
    form = O(B(V.SORTIE, cx=0.92, cy=0.92), B(V.SQUAD_1_HI, cx=0.06, cy=0.26),
             B(V.HOME, cx=0.05, cy=0.05))
    check("编队页认得出", classify(form).page == "formation",
          f"实际 {classify(form).page}")

    # 结算页：底部確認 + 无入场键
    res = O(B(V.CONFIRM, cx=0.5, cy=0.90), B(V.HOME, cx=0.05, cy=0.05))
    check("战斗结算认得出", classify(res).page == "battle_result",
          f"实际 {classify(res).page}")

    # 金钱：弹窗体内青辉石 = 打断；顶栏青辉石不是
    body = O(B(V.PYROXENE, cx=0.5, cy=0.55), B(V.CONFIRM, cx=0.5, cy=0.7),
             B(V.CANCEL, cx=0.3, cy=0.7))
    top = O(B(V.PYROXENE, cx=0.80, cy=0.045), B(V.NAV_CAFE), B(V.NAV_SHOP),
            B(V.NAV_CRAFT))
    check("弹窗体内青辉石  money_popup 打断",
          classify(body).interrupt == "money_popup")
    check("顶栏青辉石不误报", classify(top).interrupt is None,
          f"实际 {classify(top).interrupt}")

    # 2026-08-08 live 实锤：大厅左侧常驻「购买青辉石」广告位 conf 0.95。
    #    按词判会让每一轮的第一帧就 HALT。必须配上对话框控件才算购买流程。
    banner = O(B(V.SHOP_BUY_PYROXENE, conf=0.95, cx=0.117, cy=0.360),
               B(V.PYROXENE, cx=0.670, cy=0.053),
               B(V.NAV_CAFE, cx=0.073, cy=0.953), B(V.NAV_SHOP, cx=0.531, cy=0.953),
               B(V.NAV_CRAFT, cx=0.437, cy=0.953))
    check("大厅买石广告位不误报（live 实帧复现）",
          classify(banner).interrupt is None and classify(banner).page == "lobby",
          f"实际 页面={classify(banner).page} 打断={classify(banner).interrupt}")
    real_buy = O(B(V.SHOP_BUY_PYROXENE, conf=0.95, cx=0.5, cy=0.4),
                 B(V.CONFIRM, cx=0.6, cy=0.8), B(V.CANCEL, cx=0.4, cy=0.8))
    check("购买青辉石 + 对话框控件  真的停",
          classify(real_buy).interrupt == "money_popup")

    # 2026-08-08 live 实锤（反向误报）：活动首通的「獲得獎勵」奖励卡里
    #    就有 **青輝石 x30**（cy=0.511 conf 0.94）—— 那是**收入**，不是价签。
    #    打赢一场就被自己的金钱闸 HALT 掉。同一个图标，语境相反。
    reward = O(B(V.GOT_REWARD, conf=0.98, cx=0.5, cy=0.22),
               B(V.PYROXENE, conf=0.94, cx=0.669, cy=0.511))
    check("奖励卡里的青辉石=收入，不误报",
          classify(reward).interrupt is None,
          f"实际 {classify(reward).interrupt}")
    real_dialog = O(B(V.PYROXENE, conf=0.94, cx=0.5, cy=0.45),
                    B(V.CONFIRM, cx=0.6, cy=0.75), B(V.CANCEL, cx=0.4, cy=0.75))
    check("真购买框（体内青辉石+确认/取消，无收入标记） 停",
          classify(real_dialog).interrupt == "money_popup")

    # 2026-08-08 连锁事故：大厅按返回键  弹「離開遊戲」确认框，
    #    它的 `确认键` + 大厅常驻的 `購買青輝石` 广告位 = 误报购买流程停整轮。
    quit_dlg = O(B(V.SHOP_BUY_PYROXENE, conf=0.96, cx=0.117, cy=0.360),
                 B(V.CONFIRM, cx=0.60, cy=0.70), B(V.CANCEL, cx=0.40, cy=0.70),
                 B(V.NAV_CAFE, cx=0.073, cy=0.953), B(V.NAV_SHOP, cx=0.621, cy=0.953),
                 B(V.NAV_CRAFT, cx=0.531, cy=0.953))
    check("大厅「離開遊戲」确认框不误报成购买流程（该判 quit_dialog）",
          classify(quit_dlg).interrupt == "quit_dialog",
          f"实际 {classify(quit_dlg).interrupt}")
    # 且**大厅上绝不发返回键**（那个確認是退出游戏）
    from routing_v2.flow import nav as _nv2
    from routing_v2.state.machine import Machine as _M2
    _m2 = _M2(1)
    lobby_like = O(B(V.NAV_CAFE, cx=0.073, cy=0.953), B(V.NAV_SHOP, cx=0.621, cy=0.953),
                   B(V.NAV_CRAFT, cx=0.531, cy=0.953), B(V.CLUB, cx=0.22, cy=0.46))
    for _ in range(120):
        _s2 = _m2.update(lobby_like)
    _a2 = _nv2.to_lobby(lobby_like, _s2)
    check("大厅/大厅浮层上不发 KEYCODE_BACK（会弹退出游戏）",
          _a2 is None or _a2.kind != "key", str(_a2))

    # 系统「是否結束？」框：確認 = 退出游戏，只许点取消
    sysq = O(B(V.CONFIRM, conf=0.98, cx=0.598, cy=0.699),
             B(V.CANCEL, conf=0.99, cx=0.401, cy=0.701),
             B(V.CLOSE_X, conf=0.96, cx=0.692, cy=0.230))
    check("「是否結束？」（无顶栏货币） quit_dialog 打断",
          classify(sysq).interrupt == "quit_dialog",
          f"实际 {classify(sysq).interrupt}")
    from routing_v2.flow.interrupt import Interrupts as _I
    _act = _I(log=lambda m: None).handle("quit_dialog", sysq)
    check("处置只能是**取消**，永不点確認",
          _act is not None and _act.target_cls == V.CANCEL, str(_act))
    # 第二道防线：顶栏不在时，**任何 flow 的 on_confirm_dialog 都不许点確認**
    #   （08-08 实测：取消后的关闭动画帧中 quit_dialog 掉了确认，就是这里替它
    #     点了確認，差一点把游戏关掉）
    _mf = ALL["mail"](Ctx(cfg=cfg(), log=lambda m: None))
    # fixture 要还原**真实来路**：退出框只从大厅弹（08-09 起判据用 last_solid）。
    #   先在大厅待一帧，再弹框，这样 last_solid=lobby —— 和真机一致。
    _ml = Machine(1)
    _pure_lobby = O(B(V.NAV_CAFE, cx=0.073, cy=0.953),
                    B(V.NAV_SHOP, cx=0.621, cy=0.953),
                    B(V.NAV_CRAFT, cx=0.531, cy=0.953),
                    B(V.NAV_SCHEDULE, cx=0.165, cy=0.953))
    _ml.update(_pure_lobby)
    _sv = _ml.update(sysq)
    _a3 = _mf.on_confirm_dialog(sysq, _sv)
    check("退出框（从大厅弹） on_confirm_dialog 绝不点確認",
          _a3 is None or _a3.target_cls == V.CANCEL, str(_a3))
    # 反向：**游戏内**的双键框（底页是关卡弹窗）不该被当退出框拦
    #   ——「要使用340AP掃蕩17次嗎」被误拦过，扫荡直接被取消（08-09 实锤）。
    _mg = Machine(1)
    _stage = O(B(V.SWEEP_START, cx=0.73, cy=0.56), B(V.TASK_START, cx=0.73, cy=0.75))
    _mg.update(_stage)
    _sv2 = _mg.update(sysq)
    _a4 = _mf.on_confirm_dialog(sysq, _sv2)
    check("游戏内双键框（底页=关卡弹窗）不当退出框  点確認",
          _a4 is not None and _a4.target_cls == V.CONFIRM, str(_a4))
    # 「退出框」判据的四向锁（08-09 多 agent 审查 + 实测复现后重写）:
    #    历史上错过两版 ——「顶栏没货币」误拦 340AP 扫荡框；
    #    「last_solid 不在大厅系」在 **Machine 新建时 last_solid 恒为
    #      'unknown'** 的窗口里 fail-OPEN，实测 gate.ok=True  **会点確認
    #      关掉游戏**；而当时的测试**没传 last_solid**，给了假保证。
    #     现在的判据是「这个框是不是我自己点出来的」，四个方向都要锁死。
    from routing_v2.act.action import tap_box as _tbq
    _cfm = _tbq(B(V.CONFIRM, cx=0.598, cy=0.699), "点確認")
    _gq = Gate(cfg(), log=lambda m: None)
    check("A 退出框 + 没点过任何会弹框的键 + last_solid=unknown  必须拦",
          not _gq.money(_cfm, sysq, "unknown").ok,
          _gq.money(_cfm, sysq, "unknown").why)
    _gq2 = Gate(cfg(), log=lambda m: None)
    _gq2._last = (0.731, 0.563, V.SWEEP_START)      # 上一发点了「扫荡开始」
    check("B 同一个框但上一发是「扫荡开始」 放行（不许误杀游戏内框）",
          _gq2.money(_cfm, sysq, "unknown").ok)
    _gq3 = Gate(cfg(), log=lambda m: None)
    _gq3._last = (0.5, 0.7, V.SCHED_START)          # facility 底页 + 上课
    check("D facility 底页 + 上一发「課程表開始」 放行（08-09 前会无限取消上课）",
          _gq3.money(_cfm, sysq, "facility").ok)
    # C：剧情「跳過」确认框不许被 quit_dialog 抢（117 帧真实帧曾 117/117 全中）
    _story_dlg = O(B(V.CONFIRM, conf=0.97, cx=0.599, cy=0.725),
                   B(V.CANCEL, conf=0.98, cx=0.401, cy=0.727),
                   B(V.STORY_MENU, conf=0.98, cx=0.941, cy=0.055),
                   B(V.CLOSE_X, conf=0.96, cx=0.704, cy=0.200))
    check("C 剧情跳過确认框  story_cutscene，不是 quit_dialog",
          classify(_story_dlg).interrupt == "story_cutscene",
          f"实际 {classify(_story_dlg).interrupt}")

    # 組合包页：**不停机**（bot 主动进来拿免費包），但花钱的那一下仍要被拦。
    #   08-09 前是无条件 halt  默认配置每次自主跑都在 shop 整轮停机。
    from routing_v2.flow.interrupt import Interrupts as _Iq
    _combo = O(B(V.COMBO_PACK, conf=0.98, cx=0.701, cy=0.251),
               B(V.SHOP_BUY, conf=0.98, cx=0.299, cy=0.695),
               B(V.FREE, conf=0.97, cx=0.301, cy=0.651))
    check("組合包页  money_popup 不 halt（否则免費包永远拿不到）",
          _Iq(log=lambda m: None).handle("money_popup", _combo) is None)
    _gc = Gate(cfg(), log=lambda m: None)
    check("但組合包页上点 `购买` 仍被拦成人审",
          not _gc.money(_tbq(B(V.SHOP_BUY, cx=0.299, cy=0.695), "买"), _combo).ok)

    # 真正不可绕过的那道：**闸**。八个 flow 都覆写了 on_confirm_dialog，
    #    护栏放基类会被整体绕过（08-08 craft 就这么在退出框上点了確認）。
    from routing_v2.act.action import tap_box as _tb2
    _g = Gate(cfg(), log=lambda m: None)
    _bad = _tb2(B(V.CONFIRM, cx=0.598, cy=0.699), "flow 自己要点確認")
    check("闸拦下「顶栏全无的双键框上点確認」（谁都绕不过）",
          not _g.money(_bad, sysq).ok, _g.money(_bad, sysq).why)
    _ingame = O(B(V.CONFIRM, cx=0.6, cy=0.7), B(V.CANCEL, cx=0.4, cy=0.7),
                B(V.AP, cx=0.40, cy=0.033), B(V.CREDIT, cx=0.51, cy=0.033),
                B(V.PYROXENE, cx=0.67, cy=0.053))
    _ok = _tb2(B(V.CONFIRM, cx=0.6, cy=0.7), "游戏内确认")
    check("游戏内确认框（顶栏在）照常放行", _g.money(_ok, _ingame).ok)

    # 游戏内的确认框（顶栏货币在）不该被当成系统退出框
    ingame = O(B(V.CONFIRM, cx=0.6, cy=0.7), B(V.CANCEL, cx=0.4, cy=0.7),
               B(V.AP, cx=0.40, cy=0.033), B(V.CREDIT, cx=0.51, cy=0.033),
               B(V.PYROXENE, cx=0.67, cy=0.053))
    check("游戏内确认框（顶栏在）不误判成系统退出框",
          classify(ingame).interrupt != "quit_dialog",
          f"实际 {classify(ingame).interrupt}")

    # 上期活动
    check("後日談  event_ended",
          classify(O(B(V.EVENT_AFTERSTORY), B(V.EVENT_SHOP))).page == "event_ended")


# ══ 2. 连续 N 帧确认 ═══════════════════════════════════════
def t_machine():
    print("\n── §A3 连续帧确认 ───────────────────────────")
    m = Machine(confirm_frames=3)
    lobby = O(B(V.NAV_CAFE), B(V.NAV_SHOP), B(V.NAV_CRAFT))
    for _ in range(3):
        st = m.update(lobby)
    check("3 帧后确认大厅", st.page == "lobby")

    # 一帧转场噪声不该翻页
    noise = O(B(V.SORTIE))
    st = m.update(noise)
    check("单帧转场噪声不翻页", st.page == "lobby", f"raw={st.raw}")
    st = m.update(lobby)
    st = m.update(lobby)
    check("噪声后仍是大厅", st.page == "lobby")

    # 真的换页要 3 帧
    form = O(B(V.SORTIE), B(V.SQUAD_1_HI))
    st = m.update(form)
    check("换页第 1 帧不认", st.page == "lobby")
    st = m.update(form)
    st = m.update(form)
    check("换页第 3 帧才认", st.page == "formation", f"实际 {st.page}")

    # blank 兜底只允许"上一个认得出的页面是大厅"时开火（战斗中很多帧零检出，
    #    在那里点屏幕中央可能触发学生技能 —— 08-08 live 实锤）
    from routing_v2.flow import nav
    m2 = Machine(1)
    blank = O()
    for _ in range(3):
        stb = m2.update(blank)
    check("blank 且刚起进程(last_solid=unknown)  允许唤回，别卡死",
          nav.blank_escape(stb, 0) is not None, f"last_solid={stb.last_solid}")
    m2b = Machine(1)
    battle = O(B(V.BATTLE_PAUSE), B(V.BATTLE_3X))
    m2b.update(battle)
    for _ in range(60):
        stb2 = m2b.update(blank)
    check("blank 但上一页是战斗  **不点**（可能触发学生技能）",
          nav.blank_escape(stb2, 0) is None, f"last_solid={stb2.last_solid}")
    m3 = Machine(1)
    m3.update(lobby)
    for _ in range(60):
        stb = m3.update(blank)
    check("blank 且上一页是大厅  点中央唤回",
          nav.blank_escape(stb, 45) is not None, f"last_solid={stb.last_solid}")


# ══ 3. 三道闸 ══════════════════════════════════════════════════════
def t_gate():
    print("\n── 三道闸 ───────────────────────────────────")
    g = Gate(cfg(), log=lambda m: None)

    #  金钱
    body = O(B(V.PYROXENE, cx=0.5, cy=0.55))
    act = Action(kind="tap", x=0.5, y=0.7, reason="随便点", target_cls="x")
    v = g.money(act, body)
    check("弹窗体内青辉石  拦下且 halt", (not v.ok) and v.halt)
    # 但**导航白名单**里的 cls 要放行 —— 否则购买页会被拦死，
    #   免費組合包永远走不完（08-09 实锤：连"切到組合包 tab"都被 halt）。
    #   是白名单不是黑名单：漏一个花钱按钮=真花钱，漏一个导航按钮=只是卡住。
    _navs = [V.COMBO_PACK, V.CLOSE_X, V.BACK, V.CANCEL]
    for _n in _navs:
        _va = g.money(Action(kind="tap", x=0.7, y=0.25, reason="切tab/退出",
                             target_cls=_n), body)
        check(f"购买页上点导航 `{_n}`  放行（不花钱）", _va.ok, _va.why)
    _vb = g.money(Action(kind="tap", x=0.3, y=0.7, reason="点購買",
                         target_cls=V.SHOP_BUY), body)
    check("购买页上点 `购买`  仍拦成人审", not _vb.ok, _vb.why)

    v = g.money(Action(kind="tap", x=0.5, y=0.5, reason="买", money=True,
                       spend="青辉石"), O(B(V.NAV_CAFE)))
    check("声明花青辉石  直接拒", not v.ok)

    # 购买确认框盲区（08-10 live 抓到，四向锁）
    #   「是否購買該商品？」这一帧, 顶栏以下没有青辉石（价签是 CAD/免費）。
    #   tab 被框盖住、也没有步进器  money.purchase_context() **四条判据全 None**，
    #   闸对这一帧原本完全是瞎的。真钱包的确认框和免費包**长得一模一样**。
    _dlg = O(B(V.CONFIRM, cx=0.60, cy=0.83), B(V.CANCEL, cx=0.40, cy=0.83),
             B(V.CREDIT, cx=0.51, cy=0.05))          # 顶栏货币在，弹窗体内没青辉石
    check(" 这一帧 purchase_context 确实是 None（盲区成立）",
          money_rules.purchase_context(_dlg) is None)
    # `_last` 由 `note_fired` 落 —— 它只在 tap **真的发出去之后**才被调用
    #   （原来写在 dedup 里，等于把被 JIT 丢弃的那一发也记成「上一发」，
    #    而 `_last` 是金钱闸唯一的正向证据，见 Gate.note_fired）
    _prev = lambda g, c: g.note_fired(Action(kind="tap", x=0.3, y=0.7,
                                             reason="上一发", target_cls=c), 0)
    g4 = Gate(cfg(), log=lambda m: None)
    _prev(g4, V.SHOP_BUY)
    _vc = g4.money(Action(kind="tap", x=0.60, y=0.83, reason="确认",
                          target_cls=V.CONFIRM), _dlg)
    check(" 上一发是 `购买` + 双键框 + 没写「免費」 拦成人审",
          (not _vc.ok) and _vc.needs_human, _vc.why)
    g5 = Gate(cfg(), log=lambda m: None)
    _prev(g5, V.SHOP_BUY)
    _free = O(B(V.CONFIRM, cx=0.60, cy=0.83), B(V.CANCEL, cx=0.40, cy=0.83),
              B(V.CREDIT, cx=0.51, cy=0.05), B(V.FREE, cx=0.66, cy=0.62))
    _vd = g5.money(Action(kind="tap", x=0.60, y=0.83, reason="确认",
                          target_cls=V.CONFIRM), _free)
    check(" 同一个框写着「免費」 放行（免費組合包那条链不能被拦死）",
          _vd.ok, _vd.why)
    g6 = Gate(cfg(), log=lambda m: None)
    _prev(g6, V.SWEEP_START)
    _ve = g6.money(Action(kind="tap", x=0.60, y=0.83, reason="确认",
                          target_cls=V.CONFIRM), _dlg)
    check(" 上一发是「掃蕩開始」(不花钱的键)  不误拦",
          _ve.ok, _ve.why)

    #  JIT 落地复验
    from routing_v2.act.action import tap_box
    g2 = Gate(cfg(), log=lambda m: None)
    tgt = B(V.EVENT_LIVE, cx=0.5, cy=0.20)
    a = tap_box(tgt, "进活动", dy=0.075)          # 落点故意在框外
    check("HUB 落点确实在锚点框外", not tgt.contains(a.x, a.y),
          f"tap y={a.y:.3f} 框 y2={tgt.y2:.3f}")
    old = O(tgt, seq=1)
    gone = O(B(V.EVENT_ENDED, cx=0.5, cy=0.20), seq=2)      # 轮播翻页了
    v = g2.jit(a, old, lambda: gone)
    check("§A7 轮播翻页  JIT 丢弃这一发", not v.ok, v.why)

    moved = O(B(V.EVENT_LIVE, cx=0.5, cy=0.60), seq=3)      # 同名但位置变了
    v = g2.jit(a, old, lambda: moved)
    check("「附近有同名」≠「还是那一个」 也丢弃", not v.ok, v.why)

    same = O(B(V.EVENT_LIVE, cx=0.5, cy=0.20), seq=4)
    v = g2.jit(a, old, lambda: same)
    check("锚点没动  放行（哪怕落点在框外）", v.ok, v.why)

    #  连发
    g3 = Gate(cfg(), log=lambda m: None)
    a2 = Action(kind="tap", x=0.3, y=0.3, reason="r", target_cls="btn")
    check("第一发放行", g3.dedup(a2, False, 0, 25).ok)
    g3.note_fired(a2, 0)                 # 真发出去了才记账
    check("同落点页面没变  吞掉", not g3.dedup(a2, False, 1, 25).ok)
    check("待够 retry_frames  补一发", g3.dedup(a2, False, 25, 25).ok)
    # 别拿 reason 措辞当控制 API：换个文案不该绕过闸
    a3 = Action(kind="tap", x=0.3, y=0.3, reason="確認键!!", target_cls="btn")
    check("换 reason 文案绕不过连发闸", not g3.dedup(a3, False, 2, 25).ok)
    # 重发必须按**边沿**算间隔: 原来判的是 `frames_in_page >= retry`,
    #   那是电平 —— 跨过一次就恒真，实测连续 6 帧连发 6 下（设计意图是
    #   6 x retry 帧的重试预算，实际缩水到 1/70）。
    g7 = Gate(cfg(), log=lambda m: None)
    a4 = Action(kind="tap", x=0.4, y=0.4, reason="r", target_cls="btn2")
    _fired = []
    for _f in range(0, 400):
        _v = g7.dedup(a4, False, _f, 70)
        if _v.stop_flow:
            break
        if _v.ok:
            _fired.append(_f)
            g7.note_fired(a4, _f)
    check("重发按边沿算: 间隔恒为 retry_frames，不是每帧连发",
          all(b - a == 70 for a, b in zip(_fired, _fired[1:])) and len(_fired) >= 5,
          f"放行帧 {_fired}")


# ══ 4. flow 跨 tick 状态（真实例，不是 stateless 回放）════════════════
def t_flows():
    print("\n── flow 跨 tick 行为 ────────────────────────")
    c = cfg()
    ctx = Ctx(cfg=c, log=lambda m: None)

    # craft: 灰态 cls 优先（§A6 —— 模型修好不能把兜底打死）
    craft = ALL["craft"](ctx)
    m = Machine(1)
    grey = O(B(V.CRAFT_START_GREY, conf=0.99, cx=0.8, cy=0.85))
    st = m.update(grey)
    act = craft.decide(grey, st)
    check("§A6 检出「开始制造灰色」 直接收工不点",
          act is not None and act.kind == "done", str(act))

    # craft: 亮态才点
    craft2 = ALL["craft"](ctx)
    m2 = Machine(1)
    bright = O(B(V.CRAFT_START, conf=0.95, cx=0.8, cy=0.85))
    st2 = m2.update(bright)
    act2 = craft2.decide(bright, st2)
    check("亮态「开始制造」 点它", act2 is not None and act2.is_tap, str(act2))

    # event: 轮播闸 —— 405 一直在场时**不点**，只在跃迁那帧点
    ev = ALL["event"](ctx)
    m3 = Machine(1)
    hall_cur = O(B(V.HUB_BOUNTY, cx=0.3, cy=0.4), B(V.HUB_ARENA, cx=0.5, cy=0.4),
                 B(V.HUB_JFD, cx=0.7, cy=0.4),
                 B(V.EVENT_LIVE, conf=0.88, cx=0.5, cy=0.15))
    hall_other = O(B(V.HUB_BOUNTY, cx=0.3, cy=0.4), B(V.HUB_ARENA, cx=0.5, cy=0.4),
                   B(V.HUB_JFD, cx=0.7, cy=0.4),
                   B(V.EVENT_ENDED, conf=0.88, cx=0.5, cy=0.15))
    a = ev.decide(hall_cur, m3.update(hall_cur))
    check("§轮播 405 已在场（可能在窗口尾巴） 不点",
          a is not None and a.kind == "wait", str(a))
    ev.decide(hall_other, m3.update(hall_other))          # 看见别的了
    a = ev.decide(hall_cur, m3.update(hall_cur))
    check("§轮播 捕到 (非405405) 跃迁  点", a is not None and a.is_tap, str(a))
    if a is not None and a.is_tap:
        check("§HUB 落点 +0.075 打到卡片本体", abs(a.y - (0.15 + 0.075)) < 1e-6,
              f"y={a.y:.3f}")
        check("§落地复验锚点 = 距离结束还剩", a.require == V.EVENT_LIVE)

    # event: 通关阶段从上往下打**未通关**的关（得星_0）
    ev2 = ALL["event"](ctx)
    m4 = Machine(1)
    lst = O(B(V.EVENT_SHOP, cx=0.1, cy=0.5), B(V.EVENT_QUEST_SEL, cx=0.6, cy=0.15),
            B(V.STAGE_ENTER, cx=0.9, cy=0.30), B(V.STAR_3, cx=0.6, cy=0.30),
            B(V.STAGE_ENTER, cx=0.9, cy=0.50), B(V.STAR_0, cx=0.6, cy=0.50),
            B(V.STAGE_ENTER, cx=0.9, cy=0.70), B(V.STAR_0, cx=0.6, cy=0.70))
    st4 = m4.update(lst)
    a = ev2.decide(lst, st4)
    check("活动: 跳过已 3 星的，打第一个 得星_0 的关",
          a is not None and a.is_tap and abs(a.y - 0.50) < 0.01, str(a))

    # 编队：确认不了是哪支部队  不出击
    ev3 = ALL["event"](ctx)
    m5 = Machine(1)
    form_unknown = O(B(V.SORTIE, cx=0.92, cy=0.92))
    for i in range(3):
        a = ev3.decide(form_unknown, m5.update(form_unknown))
    check("编队页看不到部队 cls  不出击（Best Record 会被永久锁死）",
          a is not None and a.kind == "wait", str(a))
    form_t1 = O(B(V.SORTIE, cx=0.92, cy=0.92), B(V.SQUAD_1_HI, cx=0.06, cy=0.26))
    a = ev3.decide(form_t1, m5.update(form_t1))
    check("部队1 已高亮  出击", a is not None and a.is_tap
          and abs(a.x - 0.92) < 0.01, str(a))

    # 加成阶段，**没赢过就不许扫荡**（08-08 live 事故：把"点了入场键"
    #    当成"打完一场顶好纪录"，加成队一次没上场，780 AP 按旧纪录刷掉）
    # 离线 fixture 没有真帧，OCR 读不出 AP；而 08-09 起 AP 闸是 **fail-closed**
    #    （读不出就停手，防 AP耗尽还去点扫荡游戏弹購買AP框）。
    #     测试必须把读数打桩，否则测的是 读不出 而不是加成逻辑。
    import routing_v2.percept.read as _RD
    _orig_topbar = _RD.read_topbar
    _RD.read_topbar = lambda o, c: (500 if c == _RD.AP else _orig_topbar(o, c))

    evb = ALL["event"](ctx)
    evb.state["phase"] = "bonus_clear"
    # fixture：不许读 data/routing_v2/event_topped.json 那份真实台账
    #    （线上顶过关之后这条测试会红 —— 08-09 已经中过一次）
    evb.ctx.bag["event_topped"] = {}
    m8 = Machine(1)
    lst2 = O(B(V.EVENT_SHOP, cx=0.1, cy=0.5), B(V.EVENT_QUEST_SEL, cx=0.6, cy=0.15),
             B(V.STAGE_ENTER, cx=0.9, cy=0.70), B(V.STAR_3, cx=0.6, cy=0.70),
             B(V.AP, conf=0.9, cx=0.40, cy=0.033))
    a = evb.decide(lst2, m8.update(lst2))
    check("加成: 先进关顶纪录", a is not None and a.is_tap
          and "顶纪录" in a.reason, str(a))
    a = evb.decide(lst2, m8.update(lst2))
    check("点了入场键**不算**打完  仍停在，不许跳去扫荡",
          evb.state["phase"] == "bonus_clear", f"phase={evb.state['phase']}")
    # 测试自带 plan fixture：flow 会回退读**真实的** data/routing_v2/
    #    event_farm_plan.json（08-09 加的跨进程副本），线上有 2 个目标时
    #    need=2，这条测试就红了。测试绝不能依赖外部可变状态。
    evb.ctx.bag["event_farm_plan"] = [{"from_bottom": 0, "why": "fixture"}]
    evb._bt()["win"] = 1                       # 模拟真的赢了一场
    evb.decide(lst2, m8.update(lst2))
    check("赢了一场后才转入扫荡", evb.state["phase"] == "bonus_sweep",
          f"phase={evb.state['phase']}")
    evb2 = ALL["event"](ctx)
    evb2.state["phase"] = "bonus_sweep"        # 直接跳到但没赢过
    a = evb2.decide(lst2, Machine(1).update(lst2))
    check("没赢过直接进ate  BLOCKED，拒绝低倍率扫荡",
          a is not None and a.kind == "done" and a.outcome == "BLOCKED", str(a))
    # 进关 once 保护必须**活着**（08-09 审查抓到我把它写成死码）:
    #    解锁 once 的时机写成了"人在关卡列表页"，而列表页会连续几十 tick
    #     每 tick 都清  等于没有 once  入场键重发时列表滚了就**点到隔壁关**。
    #    正解：只在 `st.changed`（刚从别的页面切回来）那一 tick 解锁。
    _evo = ALL["event"](Ctx(cfg=cfg(), log=lambda m: None))
    _evo.state["phase"] = "bonus_clear"
    _evo.ctx.bag["event_topped"] = {}
    _evo.ctx.bag["event_farm_plan"] = [{"from_bottom": 0, "why": "fixture"}]
    _mo = Machine(1)
    _taps = 0
    for _ in range(4):                      # 连续 4 tick 停在列表页
        _sto = _mo.update(lst2)
        _ao = _evo.decide(lst2, _sto)
        if _ao is not None and _ao.is_tap:
            _taps += 1
            if _ao.once_key:                # 模拟 runner：tap 落地才落 once
                _evo.state["once:" + _ao.once_key] = True
    check("连续停在关卡列表页时，入场键**只点一次**（once 保护不是死码）",
          _taps == 1, f"实际点了 {_taps} 次")

    _RD.read_topbar = _orig_topbar          # 还原读数打桩

    # 金钱，**双键框内有数量步进器 = 购买/兑换框**（08-09 差点花 30 青辉石：
    #    AP 耗尽后又点扫荡，游戏弹「購買AP 單價💎30」，而弹窗体内的青辉石图标
    #    一个都没检出  非「看见青辉石」的判据全瞎）
    _buyap = O(B(V.CONFIRM, conf=0.98, cx=0.598, cy=0.699),
               B(V.CANCEL, conf=0.98, cx=0.402, cy=0.699),
               B(V.QTY_MAX, conf=0.98, cx=0.687, cy=0.480),
               B(V.PLUS, conf=0.97, cx=0.630, cy=0.480))
    from routing_v2.act import money as _mny
    check("購買AP 框（双键+步进器，青辉石图标检不出） 判为购买框",
          _mny.purchase_context(_buyap) is not None,
          str(_mny.purchase_context(_buyap)))
    from routing_v2.act.action import tap_box as _tbx
    _g2 = Gate(cfg(), log=lambda m: None)
    _tapc = _tbx(B(V.CONFIRM, cx=0.598, cy=0.699), "flow 要点確認")
    check("闸拦下「購買AP 框上点確認」",
          not _g2.money(_tapc, _buyap, "stage_popup").ok,
          _g2.money(_tapc, _buyap, "stage_popup").why)
    # 反向：扫荡确认框是**纯文字**（无步进器） 不许误拦
    _sweepc = O(B(V.CONFIRM, conf=0.98, cx=0.598, cy=0.699),
                B(V.CANCEL, conf=0.98, cx=0.402, cy=0.701),
                B(V.CLOSE_X, conf=0.96, cx=0.692, cy=0.230))
    check("扫荡确认框（无步进器）不误判成购买框",
          _mny.purchase_context(_sweepc) is None,
          str(_mny.purchase_context(_sweepc)))

    # 没登记的设施页  facility（有退出控件，不会卡死）
    fac = O(B(V.HOME, cx=0.965, cy=0.033), B(V.BACK, cx=0.045, cy=0.052),
            B(V.PYROXENE, cx=0.67, cy=0.053), B(V.CREDIT, cx=0.51, cy=0.052))
    check("回大厅+返回键 = 「在某个设施里」，不是 unknown",
          classify(fac).page == "facility", f"实际 {classify(fac).page}")
    from routing_v2.flow import nav as _nv
    from routing_v2.state.machine import Machine as _M
    _mm = _M(1)
    for _ in range(95):
        _fs = _mm.update(fac)
    check("facility 待够后 nav 能走人（不会永久卡死）",
          _nv.to_lobby(fac, _fs) is not None, f"f={_fs.frames_in_page}")
    # 具体页面必须盖过 facility
    fac2 = O(*fac.boxes, B(V.CRAFT_QUICK, cx=0.7, cy=0.86))
    check("有具体页面签名时 facility 让位", classify(fac2).page == "craft",
          f"实际 {classify(fac2).page}")

    # 退出顺序：取消 > 結果框確認 > 叉叉 > 返回（2026-07-27 悬赏弃 6 票根因）
    from routing_v2.act.action import tap_box as _tb
    ml = ALL["mail"](ctx)
    modal = O(B(V.CANCEL, cx=0.40, cy=0.75), B(V.CONFIRM, cx=0.60, cy=0.75),
              B(V.CLOSE_X, conf=0.96, cx=0.85, cy=0.20), B(V.BACK, cx=0.04, cy=0.05))
    a = ml.exit_step(modal)
    check("模态框上退出  点**取消**（叉叉 conf0.96 也不吃点击）",
          a is not None and a.target_cls == V.CANCEL, str(a))
    result = O(B(V.CONFIRM, cx=0.5, cy=0.8), B(V.CLOSE_X, cx=0.85, cy=0.20))
    a = ml.exit_step(result)
    check("结果框（无取消）退出  点確認",
          a is not None and a.target_cls == V.CONFIRM, str(a))

    # 票数越界校验：'0/6' 被读成 '9/0' 那种必须毙掉
    from routing_v2.percept.read import read_ticket
    import routing_v2.percept.read as _R
    _orig = _R.read_beside
    try:
        _R.read_beside = lambda o, a, s=None: (9, 0)
        check("票数 9/0（分子>分母） 判无效返回 None",
              read_ticket(O(), B(V.TICKET_BOUNTY)) is None)
        _R.read_beside = lambda o, a, s=None: (6, 6)
        check("票数 6/6 正常读出", read_ticket(O(), B(V.TICKET_BOUNTY)) == 6)
        _R.read_beside = lambda o, a, s=None: (999, None)
        check("票数 999（超上限） None",
              read_ticket(O(), B(V.TICKET_BOUNTY)) is None)
    finally:
        _R.read_beside = _orig

    # 商店推算  "刷倒数第几关"的映射（用户 2026-08-08 口述的规则）
    from routing_v2.flow.event_shop import farm_targets
    tabs = [{"from_bottom": 0, "buyable": 12}, {"from_bottom": 1, "buyable": 5}]
    withpts = farm_targets(tabs, True)
    check("有活动点数: 点数倒数第1关, 最下面的币倒数第2关, 上面一个倒数第3关",
          [t["from_bottom"] for t in withpts] == [0, 1, 2],
          str([t["from_bottom"] for t in withpts]))
    nopts = farm_targets(tabs, False)
    check("无活动点数: **商店最下面的币就是最后一关**，再往上推",
          [t["from_bottom"] for t in nopts] == [0, 1],
          str([t["from_bottom"] for t in nopts]))

    # 悬赏：分支**动态从屏上判**
    bo = ALL["bounty"](ctx)
    m6 = Machine(1)
    br = O(B(V.TICKET_BOUNTY, cx=0.2, cy=0.05),
           B(V.BRANCH_CLASSROOM, cx=0.8, cy=0.30),
           B(V.BRANCH_HIGHWAY, cx=0.8, cy=0.50))
    st6 = m6.update(br)
    a = bo.decide(br, st6)
    check("悬赏: 按配置顺序选到「教室」", a is not None and a.is_tap
          and a.target_cls == V.BRANCH_CLASSROOM, str(a))

    # JFD：用 cls 不用写死坐标
    jf = ALL["jfd"](ctx)
    m7 = Machine(1)
    ac = O(B(V.TICKET_JFD, cx=0.2, cy=0.05),
           B(V.ACADEMY_TRINITY, cx=0.92, cy=0.253),
           B(V.ACADEMY_MILLENNIUM, cx=0.93, cy=0.549),
           B(V.ACADEMY_GEHENNA, cx=0.915, cy=0.401))
    a = jf.decide(ac, m7.update(ac))
    check("§A8 JFD 用 cls 选学院（配置首选 千年）", a is not None and a.is_tap
          and a.target_cls == V.ACADEMY_MILLENNIUM, str(a))

    # ── 票配额（用户点名「票用在哪个地区，还是一个地区用几张」）────────
    # 关键实现约束：用**票数差**算已用几张，不用计数器（数事实不数意图）。
    def _bounty_with(plan, tix, branch, tix0):
        f = ALL["bounty"](ctx)
        f.cfg = dict(f.cfg or {})
        f.cfg["ticket_plan"] = plan
        f.setup()
        f.state.update(branch_name=branch, branch_tix0=tix0, tickets0=6)
        m = Machine(1)
        ob = O(B(V.TICKET_BOUNTY, cx=0.2, cy=0.05),
               B(V.STAGE_ENTER, cx=0.75, cy=0.40),
               B(V.STAGE_ENTER, cx=0.75, cy=0.60))
        return f, ob, m.update(ob)

    f, ob, _ = _bounty_with({"教室": 3}, 6, "教室", 6)
    check("票配额: 分支名/基线都在，配额未用完  _quota 认得出",
          f._quota("教室") == 3 and f._quota("沙漠铁道") is None)
    check("票配额: 没配的分支不限流", f._quota("高架公路") is None)

    f2, _, _ = _bounty_with({}, 6, "教室", 6)
    check("留空 ticket_plan = 老行为（不限流）",
          f2._plan() == {} and f2._quota("教室") is None)

    # 换分支必须清 branch_tix0 —— 不清的话下一个分支拿上一个的基线，
    #   配额会瞬间"用完"，第二个分支一张票都不打。
    f3, _, _ = _bounty_with({"教室": 3, "高架公路": 3}, 6, "教室", 6)
    f3.state["tickets"] = 3
    f3._next_branch_or_done("配额用完")
    check("换分支清掉 branch_tix0（否则下个分支配额瞬间'用完'）",
          f3.state["branch_tix0"] is None and f3.state["branch_name"] == "")

    # 只配 ticket_plan 不配 branches 时，分支数要按 plan 的 key 算，
    #   否则 max(1, len([])) = 1  打完第一个就收工。
    f4, _, _ = _bounty_with({"教室": 3, "高架公路": 3}, 6, "教室", 6)
    f4.cfg["branches"] = []
    f4.state["branch_i"] = 0
    f4.state["tickets"] = 3
    r = f4._next_branch_or_done("配额用完")
    check("只配 plan 不配 branches  仍会去打第二个分支",
          f4.state["branch_i"] == 1 and not getattr(r, "is_finish", False),
          f"branch_i={f4.state['branch_i']} r={r}")

    # 核心路径：票 63 用掉 3 张 = 配额满  换分支。
    #   票数走 OCR，离线测不了  打桩 `_tickets`。打桩的是**读数**，
    #   被测的是**配额判定逻辑**，不是自己测自己。
    def _quota_hit(plan, tix_now, tix0, branch="教室", bi=0):
        f = ALL["bounty"](ctx)
        f.cfg = dict(f.cfg or {})
        f.cfg.update(ticket_plan=plan, branches=["教室", "高架公路"])
        f.setup()
        f.state.update(branch_name=branch, branch_tix0=tix0,
                       tickets0=tix0, branch_i=bi)
        f._tickets = lambda _obs: tix_now
        ob = O(B(V.TICKET_BOUNTY, cx=0.2, cy=0.05),
               B(V.STAGE_ENTER, cx=0.75, cy=0.40))
        return f, f._on_stage_list(ob, Machine(1).update(ob))

    f5, r5 = _quota_hit({"教室": 3}, 3, 6)
    check("票 63 = 用满配额 3  换分支（不是继续打）",
          f5.state["branch_i"] == 1, f"branch_i={f5.state['branch_i']} r={r5}")

    f6, r6 = _quota_hit({"教室": 3}, 5, 6)
    check("票 65 = 才用 1 张，配额没满  不换分支",
          f6.state["branch_i"] == 0, f"branch_i={f6.state['branch_i']} r={r6}")

    # 票数读不出必须 fail-closed（金钱铁律 #3：读不出不出击）
    f7 = ALL["bounty"](ctx)
    f7.cfg = dict(f7.cfg or {})
    f7.cfg.update(ticket_plan={"教室": 3}, branches=["教室"])
    f7.setup()
    f7.state.update(branch_name="教室", branch_tix0=6, branch_i=0)
    f7._tickets = lambda _obs: None
    _ob7 = O(B(V.TICKET_BOUNTY, cx=0.2, cy=0.05))
    # hold() 靠 `self.ticks` 判"连续" —— 不推 ticks 的话 cnt 永远是 1，
    #   循环 70 次也只会一直返回"连续确认中"。第一版就是这么**假通过**的：
    #   断言（不是 tap）碰巧成立，却根本没走到被测的那条分支。
    for _ in range(70):
        f7.ticks += 1
        r7 = f7._on_stage_list(_ob7, Machine(1).update(_ob7))
    check("票数读不出  fail-closed 不出击（且确实走到了这条分支）",
          r7 is not None and not getattr(r7, "is_tap", False)
          and "读不出" in str(r7), str(r7))


# ══ 5. 配置锁死 ════════════════════════════════════════════════════════
def t_config():
    print("\n── 配置 ─────────────────────────────────────")
    from routing_v2.config import merged
    evil = {"safety": {"forbid_premium_currency": False, "ap_purchase_limit": 99},
            "shop": {"refresh_times": 5},
            "run": {"frame_source": "adb"}}
    c = merged(evil)
    check("改不动 forbid_premium_currency",
          c["safety"]["forbid_premium_currency"] is True)
    check("改不动 ap_purchase_limit", c["safety"]["ap_purchase_limit"] == 0)
    check("改不动 shop.refresh_times", c["shop"]["refresh_times"] == 0)
    check("frame_source 强制 scrcpy", c["run"]["frame_source"] == "scrcpy")
    check("挖矿默认关", c["modules"]["story_mining"] is False)

    # ── AP 百分比分配（用户点名）────────────────────────────────────
    # 只测**算术**：给定 AP / 百分比 / 谁还没跑  reserve 该是多少。
    # 抓帧和 OCR 不在这一层测（那是 read.py 的活）。
    # 调**真函数**，不在测试里复制一份算法 —— 复制的话真代码改了测试照样绿。
    from routing_v2.app.runner import ap_reserve_for

    def _reserve(ap, split, me, order, floor=0):
        got = ap_reserve_for(ap, split, me, order, floor)
        return got[0] if got else None

    od = ["event", "special_sweep", "batch_sweep"]
    check("AP 分配: event 占 60%  给后面留 40%",
          _reserve(1000, {"event": 60, "special_sweep": 40}, "event", od) == 400)
    check("AP 分配: 最后一个 flow 留 0（全花光，AP 停 240 就不回复了）",
          _reserve(400, {"event": 60, "special_sweep": 40}, "special_sweep", od) == 0)
    check("AP 分配: 「都刷活动」= event:100  留 0",
          _reserve(800, {"event": 100}, "event", od) == 0)
    check("AP 分配: 「都刷双三倍」= special+batch 各 50  第一个留一半",
          _reserve(600, {"special_sweep": 50, "batch_sweep": 50},
                   "special_sweep", od) == 300)
    check("AP 分配: 留底 100 时，60% 只分剩下的 900",
          _reserve(1000, {"event": 60, "special_sweep": 40}, "event", od,
                   floor=100) == 460)   # 900*0.4 + 100
    check("AP 分配: 没配百分比的 flow 不参与（返回 None 表示不注入）",
          _reserve(1000, {}, "event", od) is None)

    # ── 死开关不许再静默（08-10：modules 有 batch_sweep/special_sweep，
    #    ALL 里从来没有  前端打开后 build() 一声不吭地跳过）────────────
    from routing_v2.flow.registry import ALL as _ALL, PLANNED, build as _build
    said = []
    _build({"modules": {"batch_sweep": True, "special_sweep": True,
                        "nonexistent_flow": True},
            "order": ["batch_sweep", "nonexistent_flow"]},
           None, log=said.append)
    check("开着一条没实现的 flow  build() 必须出声",
          any("batch_sweep" in s and "还没实现" in s for s in said), str(said))
    check("order 里有注册表没有的名字  也必须出声",
          any("nonexistent_flow" in s for s in said), str(said))
    check("PLANNED 里的名字确实都还不在 ALL 里（实现了就该从 PLANNED 删掉）",
          all(n not in _ALL for n in PLANNED), str(sorted(set(PLANNED) & set(_ALL))))
    # schema 里凡是 PLANNED 的 flow，必须标在 `planned` 而不是 `options` 里
    from routing_v2.config.schema import SCHEMA as _S
    _sp = _S.get("plan.ap_split", {})
    check("没实现的 flow 不许出现在可选项里（会变成死配置）",
          not (set(_sp.get("options", [])) & set(PLANNED)),
          str(set(_sp.get("options", [])) & set(PLANNED)))

    # ── 红点归属半径必须随框尺寸放大（08-11：奖励躺着没领的根因）────
    #    红点画在按钮**右上角**，框越宽红点离框心越远。实测同一页：
    #      活动任务 w=0.029  dx≈+0.02   ；奖励资讯 w=0.076  **dx=+0.056**
    #    固定 0.05 对宽按钮永远判不到  红点驱动的领取整条失效。
    from routing_v2.flow.event import EventFlow as _EF
    # 数值全是 08-11 真机实测（奖励资讯 0.807@(0.877,0.898) w=0.076 h=0.035；
    # 红点 0.931@(0.933,0.860)）
    _wide = B(V.EVENT_REWARD_INFO, conf=0.81, cx=0.877, cy=0.898, w=0.076, h=0.035)
    _obs_dot = O(_wide, B(V.DOT_RED, conf=0.93, cx=0.933, cy=0.860))
    check("宽按钮(奖励资讯 w=0.076)身上的红点必须判得到",
          _EF._dot_on(_obs_dot, _wide), "实测 dx=+0.056，超出固定半径 0.05")
    _narrow = B(V.EVENT_TASK, conf=0.89, cx=0.393, cy=0.933, w=0.029, h=0.027)
    check("窄按钮(活动任务)不受影响：远处的红点不算它的",
          not _EF._dot_on(O(_narrow, B(V.DOT_RED, conf=0.93, cx=0.60, cy=0.90)),
                          _narrow))
    check("窄按钮自己的红点照样判得到（别把下限也放宽了）",
          _EF._dot_on(O(_narrow, B(V.DOT_RED, conf=0.93,
                                   cx=0.413, cy=0.913)), _narrow))

    # ── 买入分支必须有页面身份前提（08-11 审计抓到的 fail-OPEN）─────
    #    原代码依赖 `信用点商店`(train=0, live 990 帧 0 检出) 做前置检查，
    #    而那个 if 既没 else 也没 return  买入分支零前提。
    #    危险面：战术大赛货架上 `全部选择` conf 0.98(30/31 帧)，
    #    且 arena_shop 身份有 3/34 帧掉成 shop  会把 6 个神名文字×5 一起买走。
    _sctx = Ctx(cfg=cfg(), log=lambda m: None)   # t_config 里没有 t_flows 的 ctx
    _ash = ALL["shop"](_sctx)
    _ash.setup()
    # 战术大赛货架的样子：全部选择亮着、但**没有**信用点商店_已选中
    _arena_shelf = O(B(V.SHOP_SELECT_ALL, conf=0.98, cx=0.931, cy=0.121),
                     B(V.SHOP_BUY_SELECTED, conf=0.98, cx=0.909, cy=0.918),
                     B(V.ARENA_SHOP_CURRENCY, conf=0.97, cx=0.749, cy=0.794),
                     B(V.ARENA_SHOP_CURRENCY, conf=0.97, cx=0.866, cy=0.793))
    _aa = _ash.on_shop(_arena_shelf, Machine(1).update(_arena_shelf))
    check("认不出站在信用点货架上  绝不点「全部选择/选择购买」",
          _aa is None or getattr(_aa, "target_cls", "") not in
          (V.SHOP_SELECT_ALL, V.SHOP_BUY_SELECTED), str(_aa))
    # 正例：信用点货架（有 信用点商店_已选中）照常买
    _cs = ALL["shop"](_sctx)
    _cs.setup()
    _credit_shelf = O(B(V.SHOP_TAB_CREDIT_SEL, conf=0.98, cx=0.050, cy=0.195),
                      B(V.SHOP_SELECT_ALL, conf=0.97, cx=0.931, cy=0.122))
    _cc = _cs.on_shop(_credit_shelf, Machine(1).update(_credit_shelf))
    check("在信用点货架上照常「全部选择」（别误伤正常流程）",
          _cc is not None and getattr(_cc, "target_cls", "") == V.SHOP_SELECT_ALL,
          str(_cc))

    # ── shop 三段做完必须收工（08-11 live：全 True 了还再进一次商店，
    #    自主跑会死循环到 max_minutes_per_flow 超时）────────────────────
    _sf = ALL["shop"](_sctx)
    _sf.setup()
    _lob = O(B(V.NAV_SHOP, conf=0.97, cx=0.621, cy=0.953),
             B(V.SHOP_BUY_PYROXENE, conf=0.97, cx=0.116, cy=0.360))
    _sf.state.update(pack_done=True, bought=True, arena_done=True)
    _a = _sf.on_lobby(_lob, Machine(1).update(_lob))
    check("三段都做完  收工，不许再进商店（否则自主跑死循环）",
          _a is not None and not getattr(_a, "is_tap", False), str(_a))
    _sf2 = ALL["shop"](_sctx)
    _sf2.setup()
    _sf2.state.update(pack_done=True, bought=True)     # arena 还没做
    _a2 = _sf2.on_lobby(_lob, Machine(1).update(_lob))
    check("还有一段没做  照常进商店（别提前收工）",
          _a2 is not None and getattr(_a2, "is_tap", False), str(_a2))
    _sf3 = ALL["shop"](_sctx)
    _sf3.cfg = dict(_sf3.cfg or {})
    _sf3.cfg["buy_free_pack"] = False                  # 这段用户关了
    _sf3.setup()
    _sf3.state.update(bought=True, arena_done=True)
    _a3 = _sf3.on_lobby(_lob, Machine(1).update(_lob))
    check("关掉的那段不算没做完（否则永远收不了工）",
          _a3 is not None and not getattr(_a3, "is_tap", False), str(_a3))

    # ── 战术大赛商店不许只靠弱 cls 认页（08-11 live：tab 已高亮但
    #    `战术大赛商店已选择` conf 仅 0.116  页面判不成 arena_shop，整条支线静默失效）
    _shelf = O(B(V.ARENA_SHOP_CURRENCY, conf=0.97, cx=0.749, cy=0.794),
               B(V.ARENA_SHOP_CURRENCY, conf=0.97, cx=0.866, cy=0.793),
               B(V.ENERGY_DRINK_LOW, conf=0.98, cx=0.779, cy=0.682),
               B(V.ENERGY_DRINK_MID, conf=0.98, cx=0.896, cy=0.685))
    check("tab cls 弱到检不出时，靠货架大赛币价签仍认得出 arena_shop",
          classify(_shelf).page == "arena_shop", classify(_shelf).page)
    # 2026-08-13 live 推翻了这条原本的「或」语义: `战术大赛商店已选择` 认的是
    #   **左栏里被选中的那一行**，不是「战术大赛」这一行 —— 帧证是 bot 点歪到
    #   大決戰之后，那一行照样给 0.90~0.95，页面于是判成 arena_shop、handler
    #   照常执行，站在大決戰商店里找能量饮料。**tab 选中态单独不能成立**。
    _tabonly = O(B(V.ARENA_SHOP_TAB_SEL, conf=0.95, cx=0.067, cy=0.396))
    check("光有 tab 选中态、货架没大赛币价签  **不认**（它会在大決戰上误报）",
          classify(_tabonly).page != "arena_shop", classify(_tabonly).page)
    # 别误伤：信用点商店的价签是 `信用点`，不该被认成大赛商店
    _credit = O(B(V.SHOP_TAB_CREDIT_SEL, conf=0.98, cx=0.050, cy=0.195),
                B(V.SHOP_SELECT_ALL, conf=0.97, cx=0.931, cy=0.122),
                B(V.CREDIT, conf=0.97, cx=0.749, cy=0.445),
                B(V.CREDIT, conf=0.97, cx=0.866, cy=0.446))
    check("信用点商店不许被误判成战术大赛商店",
          classify(_credit).page == "shop", classify(_credit).page)

    # ── 配置乱码必须拒绝写盘（08-10 实伤：我把 profile.json 写乱了）──
    #    乱码的 branches 永远匹配不上屏上 cls  那条 flow **静默**选不中分支。
    import os
    import tempfile
    from routing_v2.config.schema import save as _save
    _tmp = os.path.join(tempfile.gettempdir(), "_v2_moji_test.json")
    try:
        # 这串是**故意的乱码样本**（"教室"被当 latin-1 解出来的样子），
        #   写成转义形式免得源码里再混进一个不可见控制字符。
        _save({"bounty": {"branches": ["æ\x95▙å®¤"]}}, _tmp)
        check("乱码配置必须拒绝写盘", False, "居然写进去了")
    except ValueError as e:
        check("乱码配置必须拒绝写盘", "乱码" in str(e), str(e)[:60])
    try:
        _save({"bounty": {"branches": ["教室", "高架公路"]},
               "cafe": {"invite_targets": ["爱丽丝(战斗)"]}}, _tmp)
        check("正常中文配置照常写盘（别误伤）", True)
    except Exception as e:
        check("正常中文配置照常写盘（别误伤）", False, str(e)[:80])
    finally:
        if os.path.exists(_tmp):
            os.remove(_tmp)
    check("MomoTalk 默认关", c["modules"]["momotalk"] is False)
    check("回大厅兜底默认关", c["run"]["allow_home_escape"] is False)


# ══ 6. 死判据 ══════════════════════════════════════════════════════
def t_invariants():
    """架构不变量 —— 扫源码，防"修一处没 grep 全仓同形"。"""
    print("\n── 架构不变量 ───────────────────────────────")
    import re
    root = _ROOT / "routing_v2"
    # 返回键**只能有一个发起处**（nav.back_key）。它是唯一挡住"大厅按返回
    #    弹退出游戏框"的地方；2026-08-08 我把闸写了两份、漏了第三处，
    #    结果 runner 绕过闸在大厅连按 6 次，把游戏关掉了。
    origins = []
    for p in root.rglob("*.py"):
        if p.name in ("device.py", "action.py") or "tests" in p.parts:
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            s = line.strip()
            if s.startswith("#") or '"""' in s:
                continue
            if re.search(r'keycode\s*=\s*["\']KEYCODE_BACK', s):
                origins.append(f"{p.relative_to(root)}:{i}")
    check("返回键只有一个发起处（nav.back_key）", len(origins) == 1,
          str(origins))

    # OCR 只能从 read.py 发起（YOLO 主导、OCR 只读数字）
    ocr = []
    for p in root.rglob("*.py"):
        if p.name == "read.py" or "tests" in p.parts:
            continue
        if "RapidOCR" in p.read_text(encoding="utf-8"):
            ocr.append(str(p.relative_to(root)))
    check("OCR 只在 percept/read.py 里出现（不许进决策链）", not ocr, str(ocr))

    # 热路径不许 ADB 抓帧
    grabs = []
    for p in root.rglob("*.py"):
        if p.name in ("device.py", "feed.py") or "tests" in p.parts:
            continue
        if "screencap" in p.read_text(encoding="utf-8"):
            grabs.append(str(p.relative_to(root)))
    check("除 device/feed 外没有任何地方 ADB 抓帧", not grabs, str(grabs))

    # **滑动前不许改状态**（08-10 全仓抓到 5 处，含我当天自己写的一处）。
    #    `swipes += 1` / `sig = 新指纹` / `done_rows = []` 写在 `return swipe(...)`
    #    之前 = mutate-before-ack + 数意图不数事实：动作被闸吞掉、或 step 只是
    #    看了一眼决策，计数照涨、防空转的基准也被污染  提前判「滑到底了」，
    #    momotalk 那处更会把已聊过的行台账清空  重复点。
    #     一律挂 `post=`（runner/cli 的 swipe 分支都会在真滑出去后执行它）。
    bad = []
    for p in (root / "flow").glob("*.py"):
        lines = p.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if "return swipe(" not in line:
                continue
            # 只看**同一条执行路径**上的语句：向上遇到别的 `return` 就停
            #   （早退分支里的赋值跟这一发 swipe 无关，算它就是误报）。
            for j in range(i - 1, max(-1, i - 6), -1):
                s = lines[j].strip()
                if s.startswith("#"):
                    continue
                if s.startswith("return") or s.startswith("raise"):
                    break
                if re.search(r"self\.state\[[^\]]+\]\s*(\+=|=)(?!=)", s):
                    bad.append(f"{p.name}:{j+1} {s[:48]}")
    # **滑动的几何量必须来自检出**（用户 2026-08-13 定的全局规矩:「先扫，确定
    #   滑的位置，也确定有没有目标然后再滑，不然怎么适配其他分辨率以及
    #   aspect ratio」）。写死 `swipe(0.5, 0.72, 0.5, 0.40)` 是拿某一个分辨率下
    #   量出来的比例当普适值 —— 实测设备就有 19 种分辨率。
    #   例外: 扫荡/活动 quest 那几条另有规则（用户点名），目前它们不调 swipe。
    import glob as _glob2
    import re as _re2
    _hard = []
    for _p in _glob2.glob("routing_v2/flow/*.py"):
        _src = open(_p, encoding="utf-8").read()
        for _m in _re2.finditer(r"[^.\w]swipe\(([^)]*)\)", _src, _re2.S):
            _args = _m.group(1).split(",")[:4]
            if _src[:_m.start()].rfind("SWIPE_HARDCODED_OK") > _src[:_m.start()].rfind(chr(10) + "    def "):
                continue                 # 具名豁免（必须在同一个方法里写明理由）
            if sum(1 for _a in _args if _re2.match(r"\s*0\.\d+\s*$", _a)) >= 3:
                _hard.append(f"{_p.split(chr(92))[-1]}: {_m.group(1)[:40]}")
    check("flow 里没有写死几何的 swipe（几何必须从检出推）", not _hard, str(_hard))

    check("滑动前不许改状态（计数/指纹/台账一律挂 post）", not bad,
          " | ".join(bad))

    # **`任务开始` 只许出现在真的要打关卡的 flow 里**（08-10 差一步花 30 青辉石）。
    #    实锤：`ArenaFlow.on_stage_popup` 写着
    #      `find([TASK_START, SWEEP_START])  tap("开始")`，
    #    而战术大赛压根没有关卡弹窗  event 收工停在活动关卡弹窗上、arena 接手，
    #    一上来就点了「任务开始」，当时 AP=9  游戏弹「購買AP 單價💎30」，
    #    全靠 money 闸 halt 才没成交。momotalk / mining 里也混着同一个键（已清）。
    #     白名单：只有 sweep（悬赏/JFD）和 event（活动）会真的开关卡。
    _STAGE_OK = {"sweep.py", "event.py"}
    bad2 = []
    for p in (root / "flow").glob("*.py"):
        if p.name in _STAGE_OK:
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            s = line.strip()
            if s.startswith("#") or s.startswith('"'):
                continue
            if "V.TASK_START" in s:
                bad2.append(f"{p.name}:{i}")
    check("只有 sweep/event 可以引用 `任务开始`（别的 flow 点它 = 误打战斗吃 AP）",
          not bad2, " | ".join(bad2))

    # flow 子类不许覆写 decide() —— 2026-08-08 悬赏 live 实锤:
    #    TicketSweepFlow 自己写了按页派发, 把基类 **overlay页面** 顺序整个
    #    跳过  扫荡确认框开着时照样跑列表逻辑, 把对话框里的费用行票图标
    #    当票数锚读出 0  伪报"票用完了"。
    overrides = []
    for p in (root / "flow").glob("*.py"):
        if p.name == "base.py":
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if re.match(r"\s*def decide\s*\(", line):
                overrides.append(f"flow/{p.name}:{i}")
    check("flow 子类不许覆写 decide()（overlay 顺序会被跳过）",
          not overrides, str(overrides))


def t_vocab():
    print("\n── cls 健康度 ───────────────────────────────")
    from routing_v2.state.vocab import DEAD, HEALTH, WEAK, require
    # 断言**机制**，别断言某个具体类的等级 —— 等级每次重建数据集都会变。
    #    原来这里写死「战斗失败 是死类」，v16 重建后它有 6 框了，测试当场红。
    #    测试写死会过期的事实，和注释写死会过期的行为是同一种漂移。
    check("DEAD/WEAK 分级和 HEALTH 表自洽",
          DEAD == {c for c, (t, _) in HEALTH.items() if t == 0}
          and WEAK == {c for c, (t, _) in HEALTH.items() if 0 < t < 100})
    _dead = next(iter(DEAD), None)
    check("表里确实还有死类（没有的话这道闸就没意义了）", _dead is not None,
          f"死类 {len(DEAD)} 个")
    if _dead is not None:
        try:
            require(_dead)
            ok = False
        except RuntimeError:
            ok = True
        check("拿死类当唯一信号  import 期就抛", ok, _dead)
        require(_dead, sole_signal=False)           # 当'或'成员：放行+警告
        check("死类当'或'成员  放行", True)


def t_route():
    """交班归位 -- 「该返回还是该回大厅」由**下一条 flow 的入口层**决定。

    用户 2026-08-12:「返回和大厅按钮要结合位置语义来判断是返回还是回大厅，
       要根据板块来的，比方说我们打完学园交流会，这个时候就是返回任务大厅就行。」
    """
    print("\n-- 交班归位 --------------------------------")
    from routing_v2.flow import nav as _nv
    from routing_v2.state.machine import StateView as _SV

    ctrl = [B(V.BACK, cx=0.05, cy=0.06), B(V.HOME, cx=0.12, cy=0.06)]

    def go(page, target, extra=(), plan=None):
        st = _SV(page=page, frames_in_page=10)
        return _nv.route(O(*ctrl, *extra), st, target, plan)

    # 层级图本身
    check("jfd_stage 深度 3", _nv.depth("jfd_stage") == 3)
    check("jfd_stage 祖先链经过任务大厅",
          _nv.ancestors("jfd_stage") == ["jfd_academy", "task_hall", "lobby"])
    check("未登记页面 depth=-1", _nv.depth("facility") == -1)

    # 用户点名的那条：打完学园交流会只需退回任务大厅
    a, p = go("jfd_stage", "task_hall")
    check("jfd 打完去任务大厅, 用返回键", a is not None
          and a.target_cls == V.BACK and p.get("mode") == "back",
          f"{getattr(a, 'target_cls', None)} mode={p.get('mode')}")
    # 隔两层且目标是大厅, 一步回大厅, 别按两次返回
    a, p = go("arena", "lobby")
    check("arena 打完去大厅, 用回大厅键(隔 2 层)", a is not None
          and a.target_cls == V.HOME and p.get("mode") == "home",
          f"{getattr(a, 'target_cls', None)} mode={p.get('mode')}")
    # 只隔一层, 返回键就够
    a, p = go("mail", "lobby")
    check("mail 打完去大厅, 用返回键(只隔 1 层)",
          a is not None and a.target_cls == V.BACK)
    # 目标不在祖先链上, 先回大厅再从大厅进
    a, _ = go("cafe_invite_list", "task_hall")
    check("咖啡厅里要去任务大厅, 先回大厅",
          a is not None and a.target_cls == V.HOME)
    a, _ = go("lobby", "task_hall", extra=[B(V.NAV_TASKS, cx=0.5, cy=0.9)])
    check("大厅去任务大厅, 点入口",
          a is not None and a.target_cls == V.NAV_TASKS)
    a, _ = go("task_hall", "task_hall")
    check("已经在目标层, 不动", a is None)

    # 覆盖层不参与页面身份竞争: 底页可以早就是目标层而屏上还压着确认框。
    #    不关就交班 = 下一条 flow 第一帧在别人的框上点确认。
    st_ov = _SV(page="task_hall", overlay="confirm_dialog", frames_in_page=10)
    a, _ = _nv.route(O(*ctrl, B(V.CONFIRM, cx=0.6, cy=0.7),
                       B(V.CANCEL, cx=0.4, cy=0.7)), st_ov, "task_hall", None)
    check("已到目标层但压着决策框, 先点取消", a is not None
          and a.target_cls == V.CANCEL, f"{getattr(a, 'target_cls', None)}")
    st_ov = _SV(page="task_hall", overlay="ack_dialog", frames_in_page=10)
    a, _ = _nv.route(O(*ctrl, B(V.CONFIRM, cx=0.5, cy=0.7)),
                     st_ov, "task_hall", None)
    check("已到目标层但压着单键通知框, 点掉它",
          a is not None and a.target_cls == V.CONFIRM)
    st_ov = _SV(page="task_hall", overlay="sweep_results", frames_in_page=10)
    a, _ = _nv.route(O(*ctrl, B(V.BATTLE_SKIP, cx=0.5, cy=0.9)),
                     st_ov, "task_hall", None)
    check("结算奖励动画期, 点 SKIP 而不是干等",
          a is not None and a.target_cls == V.BATTLE_SKIP)
    st_ov = _SV(page="task_hall", overlay="claim_panel", frames_in_page=10)
    a, _ = _nv.route(O(*ctrl), st_ov, "task_hall", None)
    check("覆盖层这一帧关不掉, 等它而不是掉头往别处走",
          a is not None and a.kind == "wait", f"{a.kind if a else None}")

    # 模态框必须先关干净, 否则下一条 flow 会在别人的弹窗上动手
    for page, extra, want in [
            ("stage_popup", [B(V.TASK_START, cy=0.85)], V.BACK),
            ("formation", [B(V.SORTIE, cx=0.85, cy=0.9)], V.BACK),
            ("sweep_dialog", [B(V.SWEEP_BATCH_START, cy=0.85),
                              B(V.CLOSE_X, cx=0.9, cy=0.15)], V.CLOSE_X)]:
        a, _ = go(page, "task_hall", extra=extra)
        check(f"{page} 归位时先关掉, 绝不留给下一条 flow",
              a is not None and a.target_cls == want,
              f"{getattr(a, 'target_cls', None)}")

    # 同一次归位不许换策略(用户点名: 不要又点返回又点大厅)
    a1, p1 = _nv.route(O(B(V.BACK, cx=0.05, cy=0.06)),
                       _SV(page="arena", frames_in_page=10), "lobby", None)
    check("选了回大厅但屏上暂时没有, 等它而不是改用返回键",
          a1 is not None and a1.kind == "wait" and p1.get("mode") == "home",
          f"{a1.kind if a1 else None}")
    a2, p2 = go("arena", "lobby", plan=p1)
    check("回大厅键出现后按原策略点它",
          a2 is not None and a2.target_cls == V.HOME and p2.get("mode") == "home")

    # 每条 flow 都得声明入口层, 且必须是层级图里认得的页面
    from routing_v2.flow.registry import ALL as _ALL
    bad = [n for n, c in _ALL.items()
           if getattr(c, "entry_page", "") != "lobby"
           and _nv.depth(getattr(c, "entry_page", "")) < 0]
    check("每条 flow 的 entry_page 都在层级图里", not bad, str(bad))

    # 死字段闸（2026-08-13 用户点名「注释有误导性噪音也要修，代码和注释统一」）:
    #    这轮审计抓到的漏洞几乎全是「注释说 A 代码做 B」，而最常见的形态就是
    #    **字段还在、注释还在解释它，代码早就不读它了**（`Gate._last_ts` 说是
    #    连发冷却的时间戳，实际只写不读；`sweep` 的 `swipes`/`last_low_y` 是
    #    "滑到底"那套删掉的逻辑留下的）。
    #    这是**语法事实**不是统计猜测 —— 我先试过"扫注释里引用的标识符存不存在"，
    #      108 处命中全是误报（文件名/memory 名/页面名字符串），那条路是死的。
    #    允许清单里的都是有意保留的诊断字段；**新增一个死字段就会红**。
    import ast as _ast
    import glob as _glob
    _files = [f for f in sorted(_glob.glob("routing_v2/**/*.py", recursive=True))
              if "tests" not in f and "__pycache__" not in f]
    _reads, _writes = set(), {}
    for _p in _files:
        try:
            _t = _ast.parse(open(_p, encoding="utf-8").read())
        except (SyntaxError, OSError):
            continue
        for _n in _ast.walk(_t):
            if isinstance(_n, _ast.Attribute):
                if isinstance(_n.ctx, _ast.Load):
                    _reads.add(_n.attr)
                elif isinstance(_n.ctx, _ast.Store):
                    _writes.setdefault(_n.attr, _p)
            # getattr("x") / state["x"] / .get("x") 里的字符串一律算作读
            if isinstance(_n, _ast.Constant) and isinstance(_n.value, str):
                _reads.add(_n.value)
    _ALLOW = {"cap", "last_frame", "resolution", "restarts", "rotations",
              "last_tap_pt", "last_tap_ts"}      # 诊断/统计用，故意留着
    _dead = sorted(a for a in _writes
                   if a not in _reads and not a.startswith("__")
                   and a not in _ALLOW)
    check("没有'写了但从来没人读'的实例字段（死字段=注释漂移的温床）",
          not _dead, str([f"{a} @{_writes[a]}" for a in _dead]))

    # 組合包页的"不停机"豁免必须认**选中态**，不能认页签按钮（2026-08-13 审计）:
    #   `组合包未选择` 是另一个页签的标签，站在「特別販售」页（整页 CAD 真钱货架）
    #   上它照样在场 —— 拿它当豁免 = 在一整页真钱商品上关掉金钱 HALT。
    check("只有页签按钮(未选中) 不算組合包页  金钱打断照常生效",
          not money_rules.is_combo_pack_page(O(B(V.COMBO_PACK, conf=0.97),
                                               B(V.SHOP_BUY, conf=0.98))))
    check("选中态 = 真的在組合包页  豁免",
          money_rules.is_combo_pack_page(O(B(V.COMBO_PACK_SEL, conf=0.97))))
    check("屏上有「免費」也算（免費包那一列的正向证据）",
          money_rules.is_combo_pack_page(O(B(V.FREE, conf=0.96))))

    # 台账基线自愈（2026-08-13 live，**我先判反了一次**）: 抓到
    #   `信用点 59,653 -> 59,653,863`，我按"读大"处理加了量级闸；用户把那一帧
    #   贴出来才发现**屏上真值就是 59,653,863**，错的是第一次读数（截断）。
    #   ledger 自己的原理是「OCR 只会截断，不会凭空多出位数」⇒ 旧基线是新读数的
    #   **前缀**时，可疑的是旧基线，应该修正基线而不是拒收新值。
    def _is_trunc(a, b):
        return len(str(b)) > len(str(a)) and str(b).startswith(str(a))
    for _a, _b, _want in ((59653, 59653863, True),      # live 实际那一幕
                          (2125, 21256, True),          # 同形（截断一位）
                          (59653, 1059653, False),      # 领邮件的合法涨幅
                          (9999, 10000, False),         # 真实位数增长
                          (21256, 211256, False),       # 中间插字，交给复读闸
                          (240, 238, False)):           # 正常减少
        check(f"台账 {_a}->{_b} {'判为旧基线截断' if _want else '不判截断'}",
              _is_trunc(_a, _b) == _want)

    # 左栏滑动的前提: 屏上不能压着对话框/奖励框（2026-08-13 用户点名
    #   「没检测到就滑动一次**再检测**」）。三次滑动全发生在确认框盖住左栏时,
    #   滑了没动、扫也扫不到, 预算烧光后判"没有这个 tab", 整个大赛商店被跳过。
    _sf = ALL["shop"](Ctx(cfg=cfg(), log=lambda m: None))
    _sf.state["bought"] = True
    _covered = O(B(V.SHOP_TAB_CREDIT_SEL, cx=0.05, cy=0.19),
                 B(V.CONFIRM, cx=0.6, cy=0.79), B(V.CANCEL, cx=0.4, cy=0.79))
    _a = _sf._goto_arena_tab(_covered)
    check("确认框盖着左栏时不滑（也不消耗滑动预算）",
          _a is not None and _a.kind == "wait", f"{_a and _a.kind}")
    check("被盖住时 tabscroll 没涨", _sf.state.get("tabscroll", 0) == 0)
    _clean = O(B(V.SHOP_TAB_CREDIT_SEL, cx=0.05, cy=0.19))
    _a2 = _sf._goto_arena_tab(_clean)
    check("屏面干净时才滑左栏", _a2 is not None and _a2.kind == "swipe",
          f"{_a2 and _a2.kind}")

    # 相位机（2026-08-13）: 分派器是 flow 自己的相位, **不是抖动的页面身份**。
    #   原来按 st.page 分派, 于是点开邀请卷后页面身份要 3 帧才切, 那几帧
    #   on_cafe 从头重跑、第 2 条分支把自己刚开的面板叉掉 —— live 实测
    #   开-叉循环 11 次。断言**机制**: 相位在 invite 时, 那条叉叉分支
    #   (它住在 do_earnings 里) 根本不在调用链上。
    _cf = ALL["cafe"](Ctx(cfg=cfg(), log=lambda m: None))
    _cf.goto("invite")
    _panel = O(B(V.CLOSE_X, cx=0.79, cy=0.16), B(V.CAFE_INVITE, cx=0.62, cy=0.40))
    _a5 = _cf.decide(_panel, _SV(page="cafe", frames_in_page=5))
    check("邀请相位: 页面身份还是 cafe 也不许叉掉邀请面板",
          _a5 is None or _a5.target_cls != V.CLOSE_X,
          f"{_a5 and _a5.target_cls}")
    # 同一帧、同一个页面身份, 换个相位就该叉 —— 证明决定权在相位不在页面
    _cf2 = ALL["cafe"](Ctx(cfg=cfg(), log=lambda m: None))
    _cf2.goto("earnings")
    _stray = O(B(V.CLOSE_X, cx=0.79, cy=0.16))
    _a6 = _cf2.decide(_stray, _SV(page="cafe", frames_in_page=5))
    check("收益相位: 真的挡路弹窗照常叉掉（别误伤）",
          _a6 is not None and _a6.target_cls == V.CLOSE_X,
          f"{_a6 and _a6.target_cls}")
    # 相位机的两条结构不变量
    _bad_ph = []
    for _n, _cls in ALL.items():
        for _p in getattr(_cls, "phases", ()):
            if not hasattr(_cls, f"do_{_p}"):
                _bad_ph.append(f"{_n}.{_p}")
    check("声明的每个相位都有 do_ 处理器", not _bad_ph, str(_bad_ph))
    _cf3 = ALL["cafe"](Ctx(cfg=cfg(), log=lambda m: None))
    try:
        _cf3.goto("不存在的相位")
        _raised = False
    except ValueError:
        _raised = True
    check("goto 到没声明的相位要当场报错（别默默走错路）", _raised)

    # 真钱货架页: 关掉走人, **不停整轮**（2026-08-13 小号: 購買青輝石默认开在
    #   「特別販售」, 整页 CAD 25.99/16.99, 而 414 只是没选中的页签标签）。
    #   但双键确认框仍然要 halt —— 那才是成交前一刻。
    from routing_v2.flow.interrupt import Interrupts as _IC
    _ic = _IC(log=lambda m: None)
    _cad = O(B(V.COMBO_PACK, cx=0.70, cy=0.36), B(V.CLOSE_X, cx=0.79, cy=0.28))
    _a_c = _ic._on_money_popup(_cad)
    check("停在真钱页签上要切到組合包，不是关掉走人也不是停整轮",
          _a_c is not None and _a_c.kind == "tap"
          and _a_c.target_cls == V.COMBO_PACK,
          f"{_a_c and (_a_c.kind, _a_c.target_cls)}")
    _dlg = O(B(V.COMBO_PACK, cx=0.70, cy=0.36), B(V.CLOSE_X, cx=0.79, cy=0.28),
             B(V.CONFIRM, cx=0.58, cy=0.72), B(V.CANCEL, cx=0.42, cy=0.72))
    _a_d = _ic._on_money_popup(_dlg)
    check("双键确认框照常 halt（成交前一刻不许放行）",
          _a_d is not None and _a_d.kind == "halt", f"{_a_d and _a_d.kind}")
    _free = O(B(V.COMBO_PACK_SEL, cx=0.70, cy=0.36), B(V.FREE, cx=0.40, cy=0.60))
    check("真組合包页仍然不停机（免費包这条链要走得通）",
          _ic._on_money_popup(_free) is None)

    # 大厅广告位 + 一个八竿子打不着的单键框 != 购买流程（2026-08-13 小号实测:
    #   制造锁着弹框, 框把底栏盖住 -> LOBBY_NAV<3 挡不住 -> 误报 HALT 停整轮）
    import routing_v2.act.money as _mr
    _far = O(B(V.SHOP_BUY_PYROXENE, cx=0.116, cy=0.360),
             B(V.CONFIRM, cx=0.499, cy=0.699))
    check("广告位和框隔半屏 — 不算购买流程",
          _mr.purchase_context(_far) is None, str(_mr.purchase_context(_far)))
    _near = O(B(V.SHOP_BUY_PYROXENE, cx=0.470, cy=0.520),
              B(V.CONFIRM, cx=0.499, cy=0.699))
    check("同一个框里的价签+確認 — 照常判成购买流程",
          _mr.purchase_context(_near) is not None)

    # 买不起 != 买过了（2026-08-13 小号实帧: 信用点 35,544 / 货最贵 500,000,
    #   屏上只出 `选择购买灰色` 0.78, 亮态零检出）。灰按钮点了也不动。
    _sh = ALL["shop"](Ctx(cfg=cfg(), log=lambda m: None))
    _sh.state["pack_done"] = True
    _short = O(B(V.SHOP_TAB_CREDIT_SEL, cx=0.05, cy=0.12),
               B(V.SHOP_BUY_SELECTED_GREY, cx=0.91, cy=0.92))
    for _ in range(10):
        # 走 decide 而不是直调 on_shop —— `hold()` 数的是 self.ticks，
        #   只有 decide 会推进它（直调等于每帧都是第 1 帧，hold 永远不满）
        _a_s = _sh.decide(_short, _SV(page="shop", frames_in_page=20))
    check("「選擇購買」是灰的就不许点它",
          _a_s is None or _a_s.target_cls != V.SHOP_BUY_SELECTED_GREY,
          f"{_a_s and _a_s.target_cls}")
    check("买不起要落 credit_short 收敛", bool(_sh.state.get("credit_short")))
    # 信用点这栏完了 **不等于** 整条 flow 完了 —— 大赛商店那段必须有交代
    #   （用户 2026-08-13:「也没去战术大战商店那边接着选择饮料然后检测」）
    _sh2 = ALL["shop"](Ctx(cfg=cfg(), log=lambda m: None))
    _sh2.state.update(pack_done=True, bought=True)
    _a_ar = None
    for _ in range(30):
        _a_ar = _sh2.decide(O(B(V.SHOP_TAB_CREDIT_SEL, cx=0.05, cy=0.12)),
                            _SV(page="shop", frames_in_page=60))
    check("信用点买完但还没去大赛商店 — 不许收工（还在找 tab）",
          _sh2.outcome is None, f"outcome={_sh2.outcome}")
    _sh2.state["arena_skip"] = True
    check("大赛 tab 找不到要记成「没找到入口」，不谎报已处理",
          "没找到入口" in _sh2._segments(), _sh2._segments())
    # 咖啡厅: 放弃邀请后**不许再开卷**（用户: 找两下没找到就关了又打开继续抽风）
    _cf8 = ALL["cafe"](Ctx(cfg=cfg(), log=lambda m: None))
    _cf8.goto("invite")
    _cf8.state["gaveup_f1"] = True
    _a_g = _cf8.decide(O(B(V.CAFE_TICKET, cx=0.90, cy=0.80),
                         B(V.CAFE_MOVE_2F, cx=0.10, cy=0.14)),
                       _SV(page="cafe", frames_in_page=10))
    check("放弃过邀请就不再点邀请卷",
          _a_g is None or _a_g.target_cls != V.CAFE_TICKET,
          f"{_a_g and _a_g.target_cls}")
    check("报告要说「买不起」不是「已处理」", "买不起" in _sh._segments(),
          _sh._segments())

    # 领不到 != 领过了（用户 2026-08-13:「体力999了领不了咖啡厅收益了,
    #   也没法辨别」）。原来"面板里没有可点的领取键"被写成 claimed=True,
    #   收尾堂而皇之报「收益已领」—— 谎报。
    _cf4 = ALL["cafe"](Ctx(cfg=cfg(), log=lambda m: None))
    _cf4.goto("earnings")
    _grey = O(B(V.CLAIM_GREY, cx=0.62, cy=0.62))
    _cf4.decide(_grey, _SV(page="cafe", frames_in_page=5))
    check("领取键全灰时不许标 claimed", not _cf4.state.get("claimed"),
          str(_cf4.state.get("claimed")))
    # 反过来: 领取链整条走 overlay 处理器, claimed 必须在**那里**落账
    #   （2026-08-13 live: claims=3 明明领到了, 收尾却报「收益没领到」）
    _cf7 = ALL["cafe"](Ctx(cfg=cfg(), log=lambda m: None))
    _cf7.goto("earnings")
    _act7 = _cf7.decide(O(B(V.CLAIM_YELLOW, cx=0.50, cy=0.73)),
                        _SV(page="cafe", overlay="claim_panel", frames_in_page=5))
    check("覆盖层领到了就要标 claimed", bool(_cf7.state.get("claimed")),
          f"act={_act7 and _act7.target_cls}")
    check("覆盖层在动时相位计时要清零（别在领的时候判超时）",
          _cf7.phase_ticks == 0, str(_cf7.phase_ticks))
    check("领取键全灰时要标 earn_done 收敛（别开-关死循环）",
          bool(_cf4.state.get("earn_done")))
    # 竣工判据 = **黄点**（用户口述:「判断有没有活没干的可以靠黄点来判断」）
    _cf4.goto("exit")
    _cf4.state["pats"] = 3
    _clean_scr = O(B(V.CAFE_MOVE_2F, cx=0.10, cy=0.14))
    for _ in range(25):
        _fin = _cf4.decide(_clean_scr, _SV(page="cafe", frames_in_page=30))
    check("没领到收益就不许报 CLEAN", _cf4.outcome == "LEFTOVER",
          f"{_cf4.outcome}")
    _cf5 = ALL["cafe"](Ctx(cfg=cfg(), log=lambda m: None))
    _cf5.goto("exit")
    _cf5.state.update(claimed=True, pats=3)
    for _ in range(25):
        _fin5 = _cf5.decide(_clean_scr, _SV(page="cafe", frames_in_page=30))
    check("真领到了且无黄点才 CLEAN", _cf5.outcome == "CLEAN", f"{_cf5.outcome}")
    _cf6 = ALL["cafe"](Ctx(cfg=cfg(), log=lambda m: None))
    _cf6.goto("exit")
    _cf6.state.update(claimed=True, pats=3, backs=9)
    _dotty = O(B(V.CAFE_MOVE_2F, cx=0.10, cy=0.14), B(V.DOT_YELLOW, cx=0.55, cy=0.70))
    for _ in range(25):
        _fin6 = _cf6.decide(_dotty, _SV(page="cafe", frames_in_page=30))
    check("屏上还有黄点就是还有活 — 不许 CLEAN", _cf6.outcome == "LEFTOVER",
          f"{_cf6.outcome}")

    # 绿勾归属必须 1:1 最近邻（2026-08-13 真帧量出来的）: 绿勾长在自己学生
    #   右上偏移约 (+0.028,-0.030), 而学生横向间距只有 0.057 —— 旧判据
    #   `|dx|<0.06` 比间距还大, 每个学生都能蹭到邻居的勾。实帧: 2 个绿勾
    #   判出 3 个"已选", `美咲泳装` 蹭了 `花子` 的勾被跳过。
    _sc = ALL["schedule"](Ctx(cfg=cfg(), log=lambda m: None))
    _stu = [Box(cls=n, conf=0.95, model="avatar",
                x1=cx - 0.02, y1=0.38, x2=cx + 0.02, y2=0.42)
            for n, cx in (("斯大萝", 0.403), ("花子", 0.460), ("美咲泳装", 0.524))]
    _chk = [Box(cls=V.GREEN_CHECK, conf=0.96,
                x1=cx - 0.008, y1=0.358, x2=cx + 0.008, y2=0.376)
            for cx in (0.431, 0.488)]
    # 面板身份: 体内(cy>0.15)要有 课程表票, 否则不认这个面板
    _tick = Box(cls=V.SCHED_TICKET, conf=0.95, x1=0.60, y1=0.20, x2=0.66, y2=0.24)
    _ob = Observation(boxes=_stu + _chk + [_tick], seq=1, w=3840, h=2160)
    _sc.goto("roster")
    _act = _sc.decide(_ob, _SV(page="schedule_region", frames_in_page=10))
    check("2 个绿勾只认领 2 个学生 — 没勾的那位仍会被选中",
          _act is not None and _act.target_cls == "美咲泳装",
          f"{_act and _act.target_cls}")
    # 绿勾**会闪**（老代码 _accumulate_green_marks）: 下一帧勾没检出来,
    #   也不许把已经上过课的两位重新当成可选 —— 累积表说了算。
    _ob2 = Observation(boxes=_stu + [_tick], seq=2, w=3840, h=2160)
    _act2 = _sc.decide(_ob2, _SV(page="schedule_region", frames_in_page=11))
    check("绿勾这一帧没检出, 累积表仍然记得那两位上过课",
          _act2 is not None and _act2.target_cls == "美咲泳装",
          f"{_act2 and _act2.target_cls}")
    # 房间面板一开（屏上有「課程表開始」）就必须换相位, **不许再挑学生** ——
    #   这是「抢拍」的结构性断点（原来 t1138/1144/1147 连点 3 个学生）。
    _sc2 = ALL["schedule"](Ctx(cfg=cfg(), log=lambda m: None))
    _sc2.goto("roster")
    _start = Box(cls=V.SCHED_START, conf=0.95, x1=0.55, y1=0.70, x2=0.70, y2=0.76)
    _ob3 = Observation(boxes=_stu + [_tick, _start], seq=3, w=3840, h=2160)
    _a_r = _sc2.decide(_ob3, _SV(page="schedule_region", frames_in_page=10))
    check("房间面板开着时 roster 相位不许点学生",
          _a_r is not None and _a_r.kind == "wait" and _sc2.phase == "open_room",
          f"{_a_r and _a_r.kind} phase={_sc2.phase}")
    _a_o = _sc2.decide(_ob3, _SV(page="schedule_region", frames_in_page=11))
    check("换到 open_room 相位后只按「課程表開始」",
          _a_o is not None and _a_o.target_cls == V.SCHED_START,
          f"{_a_o and _a_o.target_cls}")

    # **抢拍**（用户 2026-08-13:「咖啡厅依旧抢拍乱点，课程表也是抢拍」）:
    #   点一个学生 -> 面板盖上来 -> 那个学生名消失 -> 默认契约当场"兑现" ->
    #   下一帧点下一个学生。每一发目标 cls 都不同, 连发闸不管。
    #   拦它的补丁「消失还不够, 必须伴随新 cls」原来因为 sig0 懒设而恒被短路。
    from routing_v2.act.action import Action as _A2
    from routing_v2.act.gate import Gate as _G2
    _g3 = _G2(cfg())
    _before = O(B("留美"), B("晴露营"), B("沙织"))          # 全体课程表面板
    _tap = _A2(kind="tap", x=0.3, y=0.4, target_cls="留美", reason="进 留美 的房间")
    _g3.arm(_tap, _before)
    check("arm 时就记下点之前的 cls 基线",
          _g3._pending.get("sig0") is not None)
    # 面板盖住 -> 目标消失, 但**没有任何新 cls** = 那一下没生效
    _covered2 = O(B("晴露营"), B("沙织"))
    _v3 = _g3.advance(_tap, _covered2, page_changed=False, retry_frames=25)
    check("目标消失但没冒出新 cls  按住, 不许点下一个学生",
          not _v3.ok, f"ok={_v3.ok}")
    # 真开了面板 -> 必然带来新 cls（課程表開始）。但**节拍闸先按住固定帧数**:
    #   页面身份在同一个面板上会跳(facility <-> schedule_region), 而
    #   `page_changed` 会作废契约 —— 所有兑现条件的修复都被那条绕过去。
    #   所以在一切判定之前先无条件按住 _MIN_HOLD 帧。
    from routing_v2.act.gate import _MIN_HOLD as _MH
    _opened = O(B("晴露营"), B("沙织"), B(V.SCHED_START))
    _rel = None
    for _i in range(1, 40):
        _v4 = _g3.advance(_tap, _opened, page_changed=False, retry_frames=25)
        if _v4.ok:
            _rel = _i
            break
    check(f"面板开了也要先按住 {_MH} 帧的节拍再放行",
          _rel is not None and _rel > 1, f"第 {_rel} 次尝试才放行")
    # 连页面变化都绕不过节拍闸（这是抢拍能穿透所有闸的那条路）
    _g4 = _G2(cfg())
    _g4.arm(_tap, _before)
    _v5 = _g4.advance(_tap, _covered2, page_changed=True, retry_frames=25)
    check("page_changed 也不能立刻放行（页面在同一面板上跳 != 上一发生效）",
          not _v5.ok, f"ok={_v5.ok}")

    # 严格契约不该被"页面跳了一下"作废（2026-08-13 抢拍的最后一条通路）:
    #   点了咖啡厅邀请卷(expect=邀请键) -> 面板还没渲染 -> 页面跳一下 -> 契约作废
    #   -> 下一帧「关掉挡路的弹窗」把自己刚开的面板叉掉。
    _g5 = _G2(cfg())
    _inv = _A2(kind="tap", x=0.5, y=0.5, target_cls=V.CAFE_TICKET,
               reason="开邀请卷", expect=(V.CAFE_INVITE,))
    _g5.arm(_inv, O(B(V.CAFE_TICKET)))
    _rel2 = None
    for _i in range(1, 30):                    # 页面每帧都在跳
        _vv = _g5.advance(_inv, O(B(V.CLOSE_X)), page_changed=True,
                          retry_frames=70)
        if _vv.ok:
            _rel2 = _i
            break
    check("严格契约: 页面反复跳也不放行（等的是邀请键，不是页面动没动）",
          _rel2 is None, f"第 {_rel2} 次就放行了")
    # 宽松契约（没写 expect）照旧被页面变化作废 —— 那条语义是对的
    _g6 = _G2(cfg())
    _loose = _A2(kind="tap", x=0.5, y=0.5, target_cls="某键", reason="r")
    _g6.arm(_loose, O(B("某键")))
    for _i in range(_MH + 2):
        _vl = _g6.advance(_loose, O(B("别的")), page_changed=True, retry_frames=70)
        if _vl.ok:
            break
    check("宽松契约: 页面变了就作废（保留原语义）", _vl.ok)

    # 契约超时要退回 once 标记（2026-08-13 用户现场诊断）: 按钮还在归位时
    #   点下去, tap 确实发出去了但游戏没收到 —— `once` 的旧契约是"发出去就算
    #   做过", 于是标记被消耗、`pending()` 永为 False、**再也不重试**。
    #   发出去 != 游戏有反应, 所以严格契约超时必须把标记退回去。
    from routing_v2.act.action import Action as _A
    from routing_v2.act.gate import Gate as _G
    _g = _G(cfg())
    _act = _A(kind="tap", x=0.5, y=0.5, target_cls="全部选择", reason="全部选择",
              once_key="selectall", expect=("选择购买",))
    _g.arm(_act)
    check("严格契约记下了 once 标记", _g._pending.get("once") == "selectall")
    _v = None
    for _i in range(200):                       # 一直等不到「选择购买」
        _v = _g.advance(_act, O(B("信用点")), page_changed=False, retry_frames=25)
        if _g._pending is None:
            break
    check("契约超时后把 once 标记退回来", _v is not None
          and _v.rollback_once == "selectall", str(_v and _v.rollback_once))
    # 宽松契约（没显式 expect）超时属常态, 不该退标记
    _g2 = _G(cfg())
    _act2 = _A(kind="tap", x=0.5, y=0.5, target_cls="某个键", reason="r",
               once_key="k2")
    _g2.arm(_act2)
    for _i in range(200):
        _v2 = _g2.advance(_act2, O(B("某个键")), page_changed=False, retry_frames=25)
        if _g2._pending is None:
            break
    check("宽松契约超时不退标记（那是常态，不是没生效）",
          not _v2.rollback_once)

    # Gate 的逐次上下文必须能清(跨 flow 泄漏会误拦/误放)
    from routing_v2.act.gate import Gate as _G
    g = _G(cfg())
    g._pending = {"cls": "x"}
    g._last = (0.1, 0.2, "旧flow点的键")
    g._fires = 5
    g.stats["pass"] = 7
    g.reset()
    check("Gate.reset 清掉 _pending/_last/_fires 但保留统计",
          g._pending is None and g._last is None and g._fires == 0
          and g.stats["pass"] == 7)


if __name__ == "__main__":
    t_pages()
    t_machine()
    t_gate()
    t_flows()
    t_route()
    t_config()
    t_invariants()
    t_vocab()
    print("\n" + "═" * 52)
    if FAILS:
        print(f" {len(FAILS)} 项没过:")
        for f in FAILS:
            print(f"   · {f}")
        sys.exit(1)
    print(" 全过")
