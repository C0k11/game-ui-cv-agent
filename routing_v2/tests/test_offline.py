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
    return merged({"account": {"id": "_test"},   # 台账进 _test 桶, 不碰生产账
                   "event": {"order": "clear_then_bonus",
                             "clear_first_with_team": 1, "bonus_team": 2,
                             "shop_plan_before_bonus": False},
                   "bounty": {"branches": ["教室"]},
                   "jfd": {"academies": ["千年", "三一", "格黑娜"]}})


#  1. 页面身份
def t_pages():
    print("\n 页面身份 ")
    check("大厅", classify(O(B(V.NAV_CAFE), B(V.NAV_SHOP), B(V.NAV_CRAFT))).page
          == "lobby")
    check("任务大厅", classify(O(B(V.HUB_CAMPAIGN), B(V.BACK))).page
          == "task_hall")
    check("三旧tile无推图不当task_hall",
          classify(O(B(V.HUB_BOUNTY), B(V.HUB_ARENA),
                     B(V.HUB_JFD))).page != "task_hall")
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
    check("组合包页  money_popup 不 halt（否则免费包永远拿不到）",
          _Iq(log=lambda m: None).handle("money_popup", _combo) is None)
    check("组合包货架不是 purchase_context（08-17 页不是成交框）",
          money_rules.purchase_context(_combo) is None,
          str(money_rules.purchase_context(_combo)))
    check("组合包页不打 money_popup 打断（交给 free_pack）",
          classify(_combo).interrupt is None,
          f"实际 {classify(_combo).interrupt}")
    _gc = Gate(cfg(), log=lambda m: None)
    _vbuy = _gc.money(_tbq(B(V.SHOP_BUY, cx=0.299, cy=0.695), "买"), _combo)
    check("但组合包页上点 购买 仍被拦（拒这一发, 不停轮）",
          (not _vbuy.ok) and (not _vbuy.halt) and _vbuy.needs_human, _vbuy.why)

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


#  2. 连续 N 帧确认
def t_machine():
    print("\n §A3 连续帧确认 ")
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


#  3. 三道闸
def t_gate():
    print("\n 三道闸 ")
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
    _vdk = g.money(Action(kind="tap", x=0.91, y=0.92, reason="买饮料",
                          target_cls=V.SHOP_BUY_SELECTED,
                          spend="战术大赛货币"),
                   O(B(V.SHOP_BUY_SELECTED, cx=0.91, cy=0.92)))
    check("大赛币选择购买不声明 money 仍放行", _vdk.ok, _vdk.why)
    _vcr = g.money(Action(kind="tap", x=0.91, y=0.92, reason="信用点批量买",
                          target_cls=V.SHOP_BUY_SELECTED, spend="信用点"),
                   O(B(V.SHOP_BUY_SELECTED, cx=0.91, cy=0.92)))
    check("信用点选择购买不声明 money 仍放行", _vcr.ok, _vcr.why)
    _vns = g.money(Action(kind="tap", x=0.91, y=0.92, reason="没写 spend",
                          target_cls=V.SHOP_BUY_SELECTED),
                   O(B(V.SHOP_BUY_SELECTED, cx=0.91, cy=0.92)))
    check("选择购买未写 spend 仍拦", not _vns.ok, _vns.why)
    g_cf = Gate(cfg(), log=lambda m: None)
    g_cf.note_fired(Action(kind="tap", x=0.91, y=0.92, reason="上一发",
                           target_cls=V.SHOP_BUY_SELECTED), 0)
    _vcf = g_cf.money(Action(kind="tap", x=0.60, y=0.83, reason="确认",
                             target_cls=V.CONFIRM, spend="战术大赛货币"),
                      O(B(V.CONFIRM, cx=0.60, cy=0.83),
                        B(V.CANCEL, cx=0.40, cy=0.83),
                        B(V.CREDIT, cx=0.51, cy=0.05)))
    check("大赛币确认不声明 money 仍放行", _vcf.ok, _vcf.why)

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
    # 2026-08-15: retry 70->38 后仍必须是边沿, 不是电平连发
    g8 = Gate(cfg(), log=lambda m: None)
    a5 = Action(kind="tap", x=0.4, y=0.4, reason="r", target_cls="btn3")
    _fired38 = []
    for _f in range(0, 250):
        _v = g8.dedup(a5, False, _f, 38)
        if _v.stop_flow:
            break
        if _v.ok:
            _fired38.append(_f)
            g8.note_fired(a5, _f)
    check("retry=38 仍按边沿: 间隔恒为 38, 不是每帧连发",
          all(b - a == 38 for a, b in zip(_fired38, _fired38[1:]))
          and len(_fired38) >= 5,
          f"放行帧 {_fired38}")


#  3b. 购买青辉石页只领免费包
def t_free_pack():
    print("\n 免费包只领不买 ")
    from routing_v2.act.action import tap_box as _tbf
    _g = Gate(cfg(), log=lambda m: None)
    _combo = O(B(V.COMBO_PACK_SEL, cx=0.70, cy=0.25),
               B(V.FREE, cx=0.301, cy=0.651),
               B(V.SHOP_BUY, cx=0.299, cy=0.695),
               B(V.SHOP_BUY, cx=0.500, cy=0.695),
               B(V.SHOP_BUY, cx=0.701, cy=0.695))
    _vf = _g.money(_tbf(B(V.FREE, cx=0.301, cy=0.651), "领免费包",
                        money=False, spend=""), _combo)
    check("闸: 组合包页点免费 spend空 放行", _vf.ok, _vf.why)
    _vb = _g.money(_tbf(B(V.SHOP_BUY, cx=0.299, cy=0.695), "买"), _combo)
    check("闸: 组合包页点购买仍拦但不停轮",
          (not _vb.ok) and (not _vb.halt) and _vb.needs_human, _vb.why)
    _g2 = Gate(cfg(), log=lambda m: None)
    _g2.note_fired(Action(kind="tap", x=0.301, y=0.651, reason="领免费",
                          target_cls=V.FREE), 0)
    _dlg_free = O(B(V.CONFIRM, cx=0.60, cy=0.83), B(V.CANCEL, cx=0.40, cy=0.83),
                  B(V.CREDIT, cx=0.51, cy=0.05), B(V.FREE, cx=0.66, cy=0.62))
    _vd = _g2.money(_tbf(B(V.CONFIRM, cx=0.60, cy=0.83), "确认领取免费包",
                         money=False, spend=""), _dlg_free)
    check("闸: 确认框有免费且 spend空 放行", _vd.ok, _vd.why)

    from routing_v2.config.schema import DAILY_CHAIN as _DC
    from routing_v2.flow.registry import COMPOSITE as _COMP, build as _bld
    check("COMPOSITE 第一项是 free_pack",
          _COMP["daily_routine"][0] == "free_pack",
          str(_COMP["daily_routine"]))
    check("DAILY_CHAIN 第一项是免费包", _DC[0] == "免费包", str(_DC))
    _built = _bld(cfg(), Ctx(cfg=cfg(), log=lambda m: None))
    _names = [f.name for f in _built]
    check("build 日常第一枪是 free_pack",
          _names[:4] == ["free_pack", "club", "craft", "shop"],
          str(_names[:6]))

    _sctx = Ctx(cfg=cfg(), log=lambda m: None)
    _sf = ALL["free_pack"](_sctx)
    _sf.state["pack_done"] = False
    _lob = O(B(V.NAV_SHOP, conf=0.97, cx=0.621, cy=0.953),
             B(V.SHOP_BUY_PYROXENE, conf=0.97, cx=0.116, cy=0.360))
    _al = _sf.on_lobby(_lob, Machine(1).update(_lob))
    check("大厅看见购买青辉石(无红点)也进页",
          _al is not None and getattr(_al, "target_cls", "") == V.SHOP_BUY_PYROXENE,
          str(_al))

    _sf3 = ALL["free_pack"](_sctx)
    _sf3.state["pack_done"] = False
    _ac = _sf3.on_combo_pack(_combo, Machine(1).update(_combo))
    check("免费+购买双亮只点免费, 不点购买",
          _ac is not None and _ac.target_cls == V.FREE
          and (not _ac.money) and _ac.spend == "",
          str(_ac))

    _sf4 = ALL["free_pack"](_sctx)
    _sf4.state["pack_done"] = False
    _paid = O(B(V.COMBO_PACK_SEL, cx=0.70, cy=0.25),
              B(V.SHOP_BUY, cx=0.299, cy=0.695),
              B(V.SHOP_BUY, cx=0.500, cy=0.695),
              B(V.SHOP_BUY, cx=0.701, cy=0.695))
    _ap = _sf4.on_combo_pack(_paid, Machine(1).update(_paid))
    check("没有免费标绝不点购买",
          _ap is None or getattr(_ap, "target_cls", "") != V.SHOP_BUY,
          str(_ap))

    _sf5 = ALL["free_pack"](_sctx)
    _sf5.state["pack_done"] = False
    _ad = _sf5.on_confirm_dialog(_dlg_free, Machine(1).update(_dlg_free))
    check("确认框有免费: 点确认且 spend 空 money 否",
          _ad is not None and _ad.target_cls == V.CONFIRM
          and (not _ad.money) and _ad.spend == "",
          str(_ad))

    _sf6 = ALL["free_pack"](_sctx)
    _sf6.state["pack_done"] = False
    _dlg_paid = O(B(V.CONFIRM, cx=0.60, cy=0.83), B(V.CANCEL, cx=0.40, cy=0.83),
                  B(V.COMBO_PACK_SEL, cx=0.70, cy=0.25))
    _ax = _sf6.on_confirm_dialog(_dlg_paid, Machine(1).update(_dlg_paid))
    check("组合包确认框没免费: 不点确认",
          _ax is None or getattr(_ax, "target_cls", "") != V.CONFIRM,
          str(_ax))


#  4. flow 跨 tick 状态（真实例，不是 stateless 回放）
def t_flows():
    print("\n flow 跨 tick 行为 ")
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

    # 08-16 live after_craft: once:quick 没清, 200 帧谎报 UNKNOWN
    from routing_v2.state.machine import StateView as _CSt
    craft3 = ALL["craft"](ctx)
    craft3.state.update({"once:quick": True, "started": 1, "claimed": 2})
    after_c = O(B(V.CLAIM_ONCE_GREY, conf=0.99, cx=0.878, cy=0.858),
                B(V.CRAFT_QUICK, conf=0.99, cx=0.696, cy=0.857))
    a3 = craft3.on_craft(after_c, _CSt(page="craft", frames_in_page=40))
    check("after_craft 已开已领  CLEAN 且清 once:quick, 不 UNKNOWN",
          a3 is not None and a3.kind == "done"
          and craft3.outcome == "CLEAN"
          and not craft3.state.get("once:quick"), str(a3))
    craft4 = ALL["craft"](ctx)
    craft4.state["once:quick"] = True
    a4 = craft4.on_craft(after_c, _CSt(page="craft", frames_in_page=10))
    check("once:quick 未开未领  等面板, 不收工",
          a4 is not None and a4.kind == "wait", str(a4))

    # event: 轮播闸 —— 405 一直在场时**不点**，只在跃迁那帧点
    ev = ALL["event"](ctx)
    m3 = Machine(1)
    hall_cur = O(B(V.HUB_CAMPAIGN, cx=0.4, cy=0.4),
                 B(V.BACK, cx=0.05, cy=0.05),
                 B(V.HUB_BOUNTY, cx=0.3, cy=0.4), B(V.HUB_ARENA, cx=0.5, cy=0.4),
                 B(V.HUB_JFD, cx=0.7, cy=0.4),
                 # h 用**实测值**: 26 个真大厅帧里 405 的框高是 0.020-0.023,
                 #   B() 的默认 0.04 是工厂缺省值, 不是观测到的东西。
                 #   落点偏移现在按框高的倍数推(见 event.HUB_TILE_RATIO),
                 #   夹具高度不真实就会把兜底带判红。
                 B(V.EVENT_LIVE, conf=0.88, cx=0.5, cy=0.15, h=0.022))
    hall_other = O(B(V.HUB_CAMPAIGN, cx=0.4, cy=0.4),
                   B(V.BACK, cx=0.05, cy=0.05),
                   B(V.HUB_BOUNTY, cx=0.3, cy=0.4), B(V.HUB_ARENA, cx=0.5, cy=0.4),
                   B(V.HUB_JFD, cx=0.7, cy=0.4),
                   B(V.EVENT_ENDED, conf=0.88, cx=0.5, cy=0.15))
    a = ev.decide(hall_cur, m3.update(hall_cur))
    check("§轮播 405 已在场（可能在窗口尾巴） 不点",
          a is not None and a.kind == "wait", str(a))
    ev.decide(hall_other, m3.update(hall_other))          # 看见别的了
    a = ev.decide(hall_cur, m3.update(hall_cur))
    check("§轮播 捕到 (非405405) 跃迁  点", a is not None and a.is_tap, str(a))
    if a is not None and a.is_tap:
        # 不再断言写死的 0.075: 那是 08-07 手点一次反推的绝对常量。
        #   现在是"框高 x HUB_TILE_RATIO", 尺度无关 -- 断言按同一个式子算。
        from routing_v2.flow.event import HUB_TILE_RATIO as _HR
        check("§HUB 落点按框高推, 打到卡片本体",
              abs(a.y - (0.15 + 0.022 * _HR)) < 1e-6, f"y={a.y:.3f}")
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
    check("部队1 已高亮但没有槽位帧  仍不出击",
          a is not None and a.kind == "wait"
          and not (a.is_tap and a.target_cls == V.SORTIE), str(a))

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
    # 08-15 起 _topped_mark 对 bag fixture 只写内存（不再污染真实台账文件）,
    #    上面 evb 那场胜利会把 "0" 记进**共享的** ctx.bag —— 这条用例的前提
    #    是"台账里什么都没有", 必须自带干净台账。
    ctx.bag["event_topped"] = {}
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

    #  08-15 日常 live 复发：once 解锁不能挂在 st.changed 上
    #    活动页身份抖（event_quest_list/facility/unknown 来回翻），每次翻回
    #    列表 changed 边沿都全清 once  16 tick 内重发两处不同入场键，
    #    第二发离「掃蕩開始」只差 2% 屏高。现在解锁只按事实：
    #    「点了入场键、又攒了 40 个列表帧还没等到弹窗」才重武装。
    _flap = O(B(V.HOME, cx=0.05, cy=0.05), B(V.BACK, cx=0.02, cy=0.05))
    _mo.update(_flap)                       # 身份翻走
    _sto = _mo.update(lst2)                 # 翻回来 = changed 边沿
    _ao = _evo.decide(lst2, _sto)
    check("身份抖动翻回列表页, 进关 once **不被清**(不再连发第二处)",
          not (_ao is not None and _ao.is_tap), str(_ao))
    _taps2 = 0
    for _ in range(45):                     # 列表页干等, 弹窗一直不来
        _sto = _mo.update(lst2)
        _ao = _evo.decide(lst2, _sto)
        if _ao is not None and _ao.is_tap:
            _taps2 += 1
            if _ao.once_key:
                _evo.state["once:" + _ao.once_key] = True
    check("40 个列表帧等不到弹窗  重武装恰好补一发(不死等也不机关枪)",
          _taps2 == 1, f"补了 {_taps2} 发")

    #  08-15 日常 live：快速编辑面板收起窗口不许交 None
    #    確認提交后面板自收有动画、页面身份还滞后, None 会落到 nav 归位
    #    点返回键, 和自收赛跑输了就把编队页整层退掉（当天连环三圈白工）。
    _qe = ALL["event"](Ctx(cfg=cfg(), log=lambda m: None))
    for _k in ("af_edit", "af_auto", "af_ins", "af_confirm"):
        _qe.state["once:" + _k] = True      # 链已走完
    _stq = Machine(1).update(O(B(V.HOME, cx=0.05, cy=0.05)))
    _aq = _qe.on_squad_quick_edit(O(), _stq)   # 面板上已无確認键
    check("確認已提交、面板收起中: 等它自收, **不交 None**(不落归位点返回)",
          _aq is not None and _aq.kind == "wait", str(_aq))
    _qe.state["qe_close_wait"] = 59
    check("收起窗口有界(60 tick 后放回兜底, 防真卡死)",
          _qe.on_squad_quick_edit(O(), _stq) is None)

    #  08-15 大赛冷却闸: 读「等待時間」, 不再编队页盲等
    #    老 brain/skills/arena.py 本来就 OCR 这个倒计时(06-13 用户拍板
    #    "倒计时早没了别傻等"), 新架构重写时丢了  盲等 20s < 真冷却 ~25s,
    #    每场第一发出击都打早撞提示框, 确认吃掉后墙钟重置再蹲 ~36s。
    #    cls 526 只框标签四个字, READY 态「--:--」标签仍在屏上
    #    判据 = 读标签右侧数字; 无读数连续 2 帧才放行。
    from routing_v2.percept.read import _mmss_secs as _mm
    check("mm:ss 解析", _mm("01:25") == 85 and _mm("0025") == 25
          and _mm("25") == 25)
    check("--:--/垃圾数不当真", _mm("--:--") is None and _mm(None) is None
          and _mm("999999") is None)
    _arf = ALL["arena"](Ctx(cfg=cfg(), log=lambda m: None))
    _arf.setup()
    _arf.state.update(entry_claims=3, claims=8)
    _apg = O(B(V.ARENA_WAIT, cx=0.074, cy=0.732, w=0.052, h=0.024),
             B(V.ARENA_ROW, cx=0.66, cy=0.35, w=0.40, h=0.10),
             B(V.ARENA_ROW, cx=0.66, cy=0.79, w=0.40, h=0.10))
    _sar = Machine(1).update(_apg)
    _orig_ws = _RD.read_wait_secs
    _RD.read_wait_secs = lambda o, a: 21
    _arf.ticks += 1
    _aa = _arf.on_arena(_apg, _sar)
    check("等待時間 21s 可见  在对手页等冷却, 不进编队",
          _aa is not None and _aa.kind == "wait" and "等待時間" in _aa.reason,
          str(_aa))
    _RD.read_wait_secs = lambda o, a: None
    _arf.ticks += 1
    _a1 = _arf.on_arena(_apg, _sar)
    check("--:-- 第 1 帧只确认不放行(单帧误读不放跑)",
          _a1 is not None and _a1.kind == "wait", str(_a1))
    _arf.ticks += 1
    _a2 = _arf.on_arena(_apg, _sar)
    check("--:-- 连续 2 帧  放行点对手", _a2 is not None and _a2.is_tap,
          str(_a2))
    _RD.read_wait_secs = _orig_ws

    _RD.read_topbar = _orig_topbar          # 还原读数打桩

    #  08-15 活动奖励同层抢拍（trace t13901t13953 三连换目标）
    #    规矩: 横幅「获得奖励」永不点; 「前往大厅」在奖励层禁点(点了回 lobby
    #    拆断活动链); 同层锁一个出口; 没真按钮就 wait 不落底页。
    _rw = ALL["event"](Ctx(cfg=cfg(), log=lambda m: None))
    _st_rw = Machine(1).update(O(B(V.GOT_REWARD, cx=0.50, cy=0.22)))
    _a = _rw.on_reward(O(B(V.GOT_REWARD, conf=0.98, cx=0.50, cy=0.22)), _st_rw)
    check("奖励层只有横幅  wait，不许点横幅",
          _a is not None and _a.kind == "wait", str(_a))
    _o2 = O(B(V.GOT_REWARD, conf=0.98, cx=0.50, cy=0.22),
            B(V.STORY_TAP_CONTINUE, conf=0.85, cx=0.51, cy=0.88))
    _a = _rw.on_reward(_o2, _st_rw)
    check("横幅+点击继续  只点继续（conf 高的横幅不进候选）",
          _a is not None and _a.is_tap
          and _a.target_cls == V.STORY_TAP_CONTINUE, str(_a))
    if _a.post:
        _a.post()                       # 模拟 runner: tap 真发出去才落锁
    _o3 = O(B(V.GOT_REWARD, cx=0.50, cy=0.22),
            B(V.STORY_TAP_CONTINUE, cx=0.51, cy=0.88),
            B(V.CONFIRM, cx=0.60, cy=0.91))
    _a = _rw.on_reward(_o3, _st_rw)
    check("同层已锁出口  确认键冒出来也不换目标",
          _a is not None and ((_a.is_tap
                               and _a.target_cls == V.STORY_TAP_CONTINUE)
                              or _a.kind == "wait"), str(_a))
    _rw2 = ALL["event"](Ctx(cfg=cfg(), log=lambda m: None))
    _o4 = O(B(V.GOT_REWARD, conf=0.99, cx=0.50, cy=0.22),
            B(V.GOTO_LOBBY_TEXT, conf=0.99, cx=0.40, cy=0.95),
            B(V.CONFIRM, conf=0.97, cx=0.60, cy=0.91))
    _a = _rw2.on_reward(_o4, _st_rw)
    check("横幅+前往大厅+确认  只点确认，永不点前往大厅",
          _a is not None and _a.is_tap and _a.target_cls == V.CONFIRM, str(_a))
    # 层掉了锁必须复位（decide 里做, 因为 on_reward 在层掉之后不再被调）
    _rw2.state["reward_exit"] = V.CONFIRM
    _mlob = Machine(1)
    _lob_o = O(B(V.NAV_CAFE, cx=0.07, cy=0.95), B(V.NAV_SHOP, cx=0.53, cy=0.95),
               B(V.NAV_CRAFT, cx=0.44, cy=0.95))
    _rw2.decide(_lob_o, _mlob.update(_lob_o))
    check("奖励层掉了  出口锁复位（下一层重新选）",
          "reward_exit" not in _rw2.state)
    # 08-16 live freepack4: 点击继续常 <0.40, 叉叉 0.94 却不点
    _fp_rw = ALL["free_pack"](Ctx(cfg=cfg(), log=lambda m: None))
    _ox = O(B(V.GOT_REWARD, conf=0.99, cx=0.501, cy=0.224),
            B(V.CLOSE_X, conf=0.94, cx=0.788, cy=0.157))
    _ax = _fp_rw.on_reward(_ox, _st_rw)
    check("奖励层无继续有叉叉0.94  点叉叉",
          _ax is not None and _ax.is_tap and _ax.target_cls == V.CLOSE_X, str(_ax))
    _olow = O(B(V.GOT_REWARD, conf=0.99, cx=0.501, cy=0.224),
              B(V.STORY_TAP_CONTINUE, conf=0.25, cx=0.502, cy=0.878),
              B(V.CLOSE_X, conf=0.94, cx=0.788, cy=0.157))
    _al = _fp_rw.on_reward(_olow, _st_rw)
    check("点击继续 0.25 低于 0.40 仍点继续(认字样)",
          _al is not None and _al.is_tap
          and _al.target_cls == V.STORY_TAP_CONTINUE, str(_al))
    _obuy = O(B(V.GOT_REWARD, conf=0.99, cx=0.501, cy=0.224),
              B(V.SHOP_BUY, conf=0.96, cx=0.299, cy=0.695),
              B(V.CLOSE_X, conf=0.94, cx=0.788, cy=0.157))
    _ab = _fp_rw.on_reward(_obuy, _st_rw)
    check("奖励层有购买键  绝不点购买, 点叉叉",
          _ab is not None and _ab.is_tap and _ab.target_cls == V.CLOSE_X
          and _ab.target_cls != V.SHOP_BUY, str(_ab))

    # 金钱，**双键框内有数量步进器 = 购买/兑换框**（08-09 差点花 30 青辉石：
    #    AP 耗尽后又点扫荡，游戏弹「購買AP 單價30」，而弹窗体内的青辉石图标
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

    # 购买框有条件交回 flow（event_shop 高价优先购买, 2026-08-13）:
    #   三重合取 —— flow 声明 + 12s 授权金钱步宽限窗 + 框体无青辉石。
    #   缺任何一项都必须照旧 halt（08-09 那道防线不许被这次放行弄坏）。
    import time as _tm
    from routing_v2.flow.interrupt import Interrupts as _Itc
    _itc = _Itc(log=lambda m: None)
    check("购买框: 无声明无宽限  照旧 halt",
          _itc._on_money_popup(_buyap).kind == "halt")
    _itc.flow_handles_purchase = True
    check("购买框: 只有声明没有宽限窗  照旧 halt",
          _itc._on_money_popup(_buyap).kind == "halt")
    _itc.money_grace_until = _tm.time() + 5
    check("购买框: 声明+宽限+框内无青辉石  交回 flow（返回 None）",
          _itc._on_money_popup(_buyap) is None)
    _buyap_pyx = O(B(V.CONFIRM, conf=0.98, cx=0.598, cy=0.699),
                   B(V.CANCEL, conf=0.98, cx=0.402, cy=0.699),
                   B(V.QTY_MAX, conf=0.98, cx=0.687, cy=0.480),
                   B(V.PYROXENE, conf=0.90, cx=0.560, cy=0.470))
    check("购买框: 框体内有青辉石  就算有声明+宽限也 halt",
          _itc._on_money_popup(_buyap_pyx).kind == "halt")
    _itc.money_grace_until = _tm.time() - 1
    check("购买框: 宽限窗过期  照旧 halt",
          _itc._on_money_popup(_buyap).kind == "halt")

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

    #  票配额（用户点名「票用在哪个地区，还是一个地区用几张」）
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


#  5. 配置锁死
def t_config():
    print("\n 配置 ")
    from routing_v2.config import merged
    evil = {"safety": {"forbid_premium_currency": False,
                       "ap_purchase_limit": 99,
                       "money_step_needs_human": False},
            "shop": {"refresh_times": 5},
            "run": {"frame_source": "adb"}}
    c = merged(evil)
    check("改不动 forbid_premium_currency",
          c["safety"]["forbid_premium_currency"] is True)
    check("改不动 ap_purchase_limit", c["safety"]["ap_purchase_limit"] == 0)
    check("改不动 money_step_needs_human",
          c["safety"]["money_step_needs_human"] is True)
    check("改不动 shop.refresh_times", c["shop"]["refresh_times"] == 0)
    check("shop.credit_buy 默认关", c["shop"]["credit_buy"] is False)
    check("shop.arena_shop 默认开", c["shop"]["arena_shop"] is True)
    check("frame_source 强制 scrcpy", c["run"]["frame_source"] == "scrcpy")
    check("挖矿默认关", c["modules"]["story_mining"] is False)
    from routing_v2.config.schema import DEFAULTS as _DEF
    check("schema 默认 retry_frames=38", _DEF["run"]["retry_frames"] == 38)

    try:
        merged({"account": {"id": "typo"},
                "accounts": {"main": {"cafe": {"skip_invite": False}}}})
        _typo_refused = False
    except ValueError:
        _typo_refused = True
    check("非空 accounts 不含 account.id 时拒绝开跑", _typo_refused)
    _single = merged({"account": {"id": "_single"}, "accounts": {}})
    check("accounts 为空时保留单账号兼容",
          _single["account"]["id"] == "_single")
    _covered = merged({
        "account": {"id": "main"},
        "accounts": {
            "main": {
                "account": {"id": "other"},
                "accounts": {},
                "safety": {"money_step_needs_human": False},
            }
        },
    })
    check("账号覆盖不能改 account/accounts",
          _covered["account"]["id"] == "main"
          and "main" in _covered["accounts"])
    check("账号覆盖也掀不开 money_step_needs_human",
          _covered["safety"]["money_step_needs_human"] is True)

    #  AP 百分比分配（用户点名）
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

    #  死开关不许再静默（08-10：modules 有 batch_sweep/special_sweep，
    #    ALL 里从来没有  前端打开后 build() 一声不吭地跳过）
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

    #  红点归属半径必须随框尺寸放大（08-11：奖励躺着没领的根因）
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
    # 08-16 live taskhall1: 黄点在 tile 右上角, 名字框中心对不上
    from routing_v2.flow import nav as _navdot
    _jfd = B(V.HUB_JFD, conf=0.98, cx=0.5456, cy=0.8039, w=0.0967, h=0.0328)
    _yel_jfd = B(V.DOT_YELLOW, conf=0.93, cx=0.6087, cy=0.7669)
    _bounty = B(V.HUB_BOUNTY, conf=0.98, cx=0.5607, cy=0.5494, w=0.0770, h=0.0362)
    _yel_bounty = B(V.DOT_YELLOW, conf=0.92, cx=0.6339, cy=0.5117)
    _arena = B(V.HUB_ARENA, conf=0.99, cx=0.6669, cy=0.8041, w=0.0815, h=0.0377)
    _red_arena = B(V.DOT_RED, conf=0.91, cx=0.7648, cy=0.7679)
    _story = B(V.HUB_STORY, conf=0.99, cx=0.8137, cy=0.2439, w=0.0700, h=0.0625)
    _far = B(V.DOT_YELLOW, conf=0.91, cx=0.9713, cy=0.1912)
    _th = O(_jfd, _bounty, _arena, _yel_jfd, _yel_bounty, _red_arena, _far)
    check("taskhall1 学院交流会黄点在名框右上  判得到",
          _navdot.dot_on(_th, _jfd))
    check("taskhall1 悬赏黄点在名框右上  判得到",
          _navdot.dot_on(_th, _bounty))
    check("taskhall1 战术大赛红点在名框右上  判得到",
          _navdot.dot_on(_th, _arena))
    check("taskhall1 右侧无关黄点不算剧情的",
          not _navdot.dot_on(O(_story, _far), _story))
    _mail = B(V.NAV_MAIL, conf=0.96, cx=0.892, cy=0.054, w=0.04, h=0.04)
    _rd = B(V.DOT_RED, conf=0.93, cx=0.922, cy=0.034)
    check("大厅入口红点离框心 0.05 内仍算",
          _navdot.dot_on(O(_mail, _rd), _mail))
    import inspect as _ins
    from routing_v2.app.runner import Runner as _Rn
    check("末流后归位大厅",
          '_handoff("lobby")' in _ins.getsource(_Rn.run_all))

    #  买入分支必须有页面身份前提（08-11 审计抓到的 fail-OPEN）
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
    # 正例：信用点货架走全部选择, 不滑货架单卡买
    _cs = ALL["shop"](_sctx)
    _cs.setup()
    _cs.cfg = dict(_cs.cfg or {})
    _cs.cfg["credit_buy"] = True
    _credit_shelf = O(B(V.SHOP_TAB_CREDIT_SEL, conf=0.98, cx=0.050, cy=0.195),
                      B(V.SHOP_SELECT_ALL, conf=0.97, cx=0.931, cy=0.122),
                      B(V.SHOP_BUY, cx=0.55, cy=0.48),
                      B(V.SHOP_BUY, cx=0.78, cy=0.48),
                      B(V.SHOP_BUY, cx=0.55, cy=0.83),
                      B(V.SHOP_BUY, cx=0.78, cy=0.83),
                      B(V.CREDIT, cx=0.55, cy=0.40))
    _cc = _cs.on_shop(_credit_shelf, Machine(1).update(_credit_shelf))
    check("信用点货架点全部选择, 不探底单卡",
          _cc is not None and getattr(_cc, "target_cls", "") == V.SHOP_SELECT_ALL,
          str(_cc))

    #  shop 三段做完必须收工（08-11 live：全 True 了还再进一次商店，
    #    自主跑会死循环到 max_minutes_per_flow 超时）
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

    #  战术大赛商店不许只靠弱 cls 认页（08-11 live：tab 已高亮但
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

    #  配置乱码必须拒绝写盘（08-10 实伤：我把 profile.json 写乱了）
    #    乱码的 branches 永远匹配不上屏上 cls  那条 flow **静默**选不中分支。
    import os
    import tempfile
    from routing_v2.config.schema import save as _save
    _tmp = os.path.join(tempfile.gettempdir(), "_v2_moji_test.json")
    try:
        # 这串是**故意的乱码样本**（"教室"被当 latin-1 解出来的样子），
        #   写成转义形式免得源码里再混进一个不可见控制字符。
        _save({"bounty": {"branches": ["æ\x95å®¤"]}}, _tmp)
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


#  6. 死判据
def t_invariants():
    """架构不变量 —— 扫源码，防"修一处没 grep 全仓同形"。"""
    print("\n 架构不变量 ")
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
    #    一上来就点了「任务开始」，当时 AP=9  游戏弹「購買AP 單價30」，
    #    全靠 money 闸 halt 才没成交。momotalk / mining 里也混着同一个键（已清）。
    #     白名单：只有 sweep（悬赏/JFD）和 event（活动）会真的开关卡。
    # campaign 是合法例外: 它的本职就是花 AP 打关, 且 stage 由用户配置、
    #   没配就 BLOCKED（策略是用户的, bot 只负责走）
    _STAGE_OK = {"sweep.py", "event.py", "campaign.py"}
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
    # 数据集卫生(v17 修复战役的常设闸): val 无泄漏/无同坐标异类/单实例无双标。
    #    --fast = 用哈希缓存 + train 抽样; 数据集不在本机就跳过(别绑死环境)。
    import subprocess as _sp
    _ds = Path("D:/Project/ml_cache/models/yolo/dataset/ui_v2")
    if _ds.is_dir():
        _r = _sp.run([sys.executable, "-X", "utf8",
                      str(_ROOT / "scripts" / "audit_dataset_hygiene.py"),
                      "--fast"], capture_output=True, text=True,
                     encoding="utf-8", errors="replace")
        check("ui_v2 数据集卫生（泄漏/同坐标异类/单实例双标）",
              _r.returncode == 0,
              (_r.stdout or "").strip().splitlines()[-1] if _r.stdout else "")
    check("flow 子类不许覆写 decide()（overlay 顺序会被跳过）",
          not overrides, str(overrides))


def t_vocab():
    print("\n cls 健康度 ")
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
    print("\n-- 交班归位 ----")
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

    # 组合包页豁免必须认选中态或免费标, 不能认未选中页签。
    # 特别贩售上组合包未选择也在, 那一页交给 FreePackFlow 切 tab, 不是成交框。
    check("只有页签按钮(未选中) 不算组合包已选中",
          not money_rules.is_combo_pack_page(O(B(V.COMBO_PACK, conf=0.97),
                                               B(V.SHOP_BUY, conf=0.98))))
    check("选中态 = 真的在組合包页  豁免",
          money_rules.is_combo_pack_page(O(B(V.COMBO_PACK_SEL, conf=0.97))))
    check("屏上有「免費」也算（免費包那一列的正向证据）",
          money_rules.is_combo_pack_page(O(B(V.FREE, conf=0.96))))

    # 台账基线自愈（2026-08-13 live，**我先判反了一次**）: 抓到
    #   `信用点 59,653 -> 59,653,863`，我按"读大"处理加了量级闸；用户把那一帧
    #   贴出来才发现**屏上真值就是 59,653,863**，错的是第一次读数（截断）。
    #   ledger 自己的原理是「OCR 只会截断，不会凭空多出位数」=> 旧基线是新读数的
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

    # 特别贩售: 切到组合包, 不停整轮。货架页不是成交框。
    from routing_v2.flow.interrupt import Interrupts as _IC
    _ic = _IC(log=lambda m: None)
    _cad = O(B(V.COMBO_PACK, cx=0.70, cy=0.36), B(V.CLOSE_X, cx=0.79, cy=0.28))
    check("特别贩售不是 purchase_context",
          money_rules.purchase_context(_cad) is None,
          str(money_rules.purchase_context(_cad)))
    _a_c = _ic._on_money_popup(_cad)
    check("停在特别贩售要切到组合包，不是关掉走人也不是停整轮",
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

    # 契约时钟跟帧走, 不跟动作走（2026-08-13 商店死按钮死锁）:
    #   点了加载期的死按钮「全部選擇」, 绿勾契约永不兑现; flow 之后每帧 wait,
    #   没有动作过闸 -> advance 的计时冻结 -> once 永不退回 -> 卡到 stalled。
    #   修 = gate.heartbeat 每帧走表, 超时把 once key 交还 runner 退回。
    from routing_v2.act.gate import Gate as _G
    from routing_v2.act.action import tap_box as _tb
    _g = _G(cfg(), log=lambda m: None)
    _dead = _tb(B(V.SHOP_SELECT_ALL, cx=0.93, cy=0.12), "全部选择",
                once="selectall", expect=(V.GREEN_CHECK,))
    _g.arm(_dead, O(B(V.SHOP_SELECT_ALL, cx=0.93, cy=0.12)))
    _g._pending["t0"] -= 9.0
    _empty = O(B(V.SHOP_SELECT_ALL, cx=0.93, cy=0.12))
    _rb = ""
    for _ in range(80):          # 只喂帧, 一发动作都不派
        _rb = _g.heartbeat(_empty, page_changed=False, retry_frames=70) or _rb
    check("flow 光等待时契约也会超时（时钟跟帧走）", _rb == "selectall",
          f"rb={_rb!r} pending={_g._pending is not None}")
    # 兑现路径: 绿勾出现 -> 契约当帧释放
    _g.arm(_dead, _empty)
    _g._pending["t0"] -= 9.0
    _g.heartbeat(_empty, page_changed=False, retry_frames=70)
    for _ in range(5):
        _g.heartbeat(O(B(V.SHOP_SELECT_ALL, cx=0.93, cy=0.12),
                       B(V.GREEN_CHECK, cx=0.52, cy=0.16)),
                     page_changed=False, retry_frames=70)
    check("绿勾一出现契约就兑现释放", _g._pending is None)
    # 节拍闸的墙钟半边（用户: 每步间隔 0.5s 保稳定）: 帧数够了墙钟没到也不放
    _g_w = _G(cfg(), log=lambda m: None)
    _g_w.arm(_dead, _empty)
    for _ in range(20):                    # 20 帧瞬间跑完, 墙钟 < 0.5s
        _g_w.heartbeat(_empty, page_changed=False, retry_frames=70)
        _v_w = _g_w.advance(_dead, _empty, page_changed=False, retry_frames=70)
    check("帧数够了但 0.5s 墙钟没到 — 仍然按住（帧数与墙钟合取）",
          not _v_w.ok, f"ok={_v_w.ok}")

    # 走格子几何层（fixture = walk_20260813_083604 帧119 的真实检出, 人工核对过）
    from routing_v2.flow import grid as _grid
    _gb = [B(V.GRID_START, conf=0.96, cx=0.385, cy=0.482),
           B(V.GRID_CELL_OPEN, conf=0.96, cx=0.385, cy=0.488),   # 叠在起点上
           B(V.GRID_CELL, conf=0.95, cx=0.429, cy=0.603),
           B(V.GRID_CELL, conf=0.96, cx=0.525, cy=0.602),
           B(V.GRID_CELL, conf=0.94, cx=0.616, cy=0.603),
           B(V.GRID_CELL, conf=0.96, cx=0.477, cy=0.722),
           B(V.GRID_ENEMY, conf=0.95, cx=0.424, cy=0.512),
           B(V.GRID_ENEMY, conf=0.92, cx=0.469, cy=0.627),
           B(V.GRID_BOSS, conf=0.85, cx=0.599, cy=0.522)]
    _go = Observation(boxes=_gb, seq=1, w=3840, h=2160)
    _cs = _grid.cells(_go)
    check("可走/起点叠在同一格上要去重（5 个格心不是 6 个）", len(_cs) == 5,
          str(len(_cs)))
    _st = _grid.steps(_cs)
    check("步长从检出现量（dx~0.096 dy~0.12）",
          _st is not None and 0.08 < _st[0] < 0.11 and 0.10 < _st[1] < 0.14,
          str(_st))
    _dx, _dy = _st
    # 单位不站格心: 立绘框心比格心高 ~0.09, 朴素最近邻会把敌方绑到起点(0.048)
    #   而不是真格子(0.091) -- 归属必须「正下方最近」
    _e1 = next(b for b in _gb if b.cls == V.GRID_ENEMY and b.cy < 0.55)
    _c1 = _grid.below(_e1, _cs, _dx)
    check("敌方绑到正下方的格子, 不是欧氏最近的起点",
          _c1 is not None and abs(_c1[0] - 0.429) < 0.01
          and abs(_c1[1] - 0.603) < 0.01, str(_c1))
    # 3-2 开局原分辨率帧 (v2_20260815_105347/0000567): 我方框心和右上
    #    邻格几乎同高, 欧氏最近会绑到邻格; 正下方必须是脚下那格。
    # below() 是**不看 cls 的纯几何**, 这里换成 队伍箭头(509 已废案) 坐标不动
    _n32_ally = B(V.GRID_ARROW, conf=0.95, cx=0.4036, cy=0.5020,
                  w=0.084, h=0.232)
    _n32_cs = [(0.4591, 0.4972), (0.6900, 0.6131),
               (0.5018, 0.6159), (0.4085, 0.6121)]
    _n32_dx, _n32_dy = 0.0933, 0.1188
    _n32_cur = _grid.below(_n32_ally, _n32_cs, _n32_dx)
    check("3-2 我方绑脚下格, 不绑同排右上邻格",
          _n32_cur is not None and abs(_n32_cur[0] - 0.4085) < 0.01
          and abs(_n32_cur[1] - 0.6121) < 0.01, str(_n32_cur))
    _n32_ru = None
    if _n32_cur is not None:
        _n32_ru = _grid.resolve(
            _n32_cur, "right-up",
            [c for c in _n32_cs
             if (c[0] - _n32_cur[0]) ** 2 + (c[1] - _n32_cur[1]) ** 2
             > (0.4 * _n32_dx) ** 2],
            _n32_dx, _n32_dy)
    check("3-2 脚下格 right-up 解析到邻格",
          _n32_ru is not None and abs(_n32_ru[0] - 0.4591) < 0.01
          and abs(_n32_ru[1] - 0.4972) < 0.01, str(_n32_ru))
    _rd = _grid.resolve((0.385, 0.482), "right-down", _cs, _dx, _dy)
    check("right-down 解析到真格心（不是推算点）",
          _rd is not None and abs(_rd[0] - 0.429) < 0.01, str(_rd))
    check("没有格子的方向要 fail-closed 返回 None",
          _grid.resolve((0.385, 0.482), "left", _cs, _dx, _dy) is None)
    # 3-2 r3: 右邻无格框。用户否掉拿 BOSS 当落点, resolve 必须 None。
    _n32r3_at = (0.593, 0.549)
    _n32r3_cs = [(0.454, 0.662), (0.547, 0.665), (0.735, 0.668),
                 (0.501, 0.548), (0.593, 0.548)]
    _n32r3_dx, _n32r3_dy = 0.0932, 0.1166
    _n32r3_oth = [c for c in _n32r3_cs
                  if (c[0] - _n32r3_at[0]) ** 2 + (c[1] - _n32r3_at[1]) ** 2
                  > (0.4 * _n32r3_dx) ** 2]
    check("3-2 r3 右邻无格框 resolve(right) 是 None（有BOSS也不点）",
          _grid.resolve(_n32r3_at, "right", _n32r3_oth,
                        _n32r3_dx, _n32r3_dy) is None)
    _n32r3_cell = _grid.resolve(
        _n32r3_at, "right", _n32r3_oth + [(0.686, 0.549)],
        _n32r3_dx, _n32r3_dy)
    check("3-2 r3 有真格子框才出点",
          _n32r3_cell is not None and abs(_n32r3_cell[0] - 0.686) < 0.01,
          str(_n32r3_cell))
    _a12 = _grid.load_answer("1-2")
    check("答案 1-2（我们的格式）: 单队 2 回合",
          _a12 is not None and len(_a12["rounds"]) == 2
          and _a12["needs"]["teams"] == 1, str(_a12 and _a12["rounds"]))
    check("答案 1-1 没有 rounds（不用走位, 不是文件缺失）",
          _grid.load_answer("1-1") is not None
          and _grid.load_answer("1-1")["rounds"] == [])
    # 禁BAAH 数字键是**备选解法**不是多区域（官方 grid_solution_format.json）:
    #    多解法文件必须只取一个主解法, 其余进 alts -- 旧版当"区域"顺序打,
    #    打完第一个解法会干等第二次部署
    _a21h = _grid.load_answer("H2-1")
    check("多解法文件取主解法, 备选进 alts 不混进 rounds",
          _a21h is not None and len(_a21h["rounds"]) == 4
          and len(_a21h.get("alts", [])) == 1, str(_a21h and len(_a21h["rounds"])))

    # CampaignFlow 骨架行为
    # 关号解析（用户拍板: 找关卡用 digitOCR 读数字, 点击 cls 主导）
    _P = ALL["campaign"]._parse_stage
    check("关号解析: '2-5' 直读 / 带噪 '.12-3.' 也收 / Hard 加前缀 / 垃圾拒收",
          _P("2-5", False) == "2-5" and _P(".12-3.", False) == "12-3"
          and _P("2-5", True) == "H2-5" and _P("25", False) is None
          and _P(None, False) is None,
          f"{_P('2-5',False)},{_P('.12-3.',False)},{_P('2-5',True)},{_P('25',False)}")
    # 配置了 stage 但屏上读到的不一致 -> BLOCKED（不在错的关上用错的答案）
    _cfgm = cfg()
    _cfgm["campaign"] = {"stage": "3-1"}
    _cpm = ALL["campaign"](Ctx(cfg=_cfgm, log=lambda m: None))
    _cpm.goto("stage_list")
    _cpm.state["stage_seen"] = False
    import numpy as _np
    _fr = _np.zeros((216, 384, 3), dtype=_np.uint8)   # OCR 会读空 -> 走 hold
    _st0 = Box(cls=V.STAR_0, conf=0.95, x1=0.545, y1=0.795, x2=0.585, y2=0.825)
    _ent = Box(cls=V.STAGE_ENTER, conf=0.96, x1=0.84, y1=0.78, x2=0.91, y2=0.82)
    _obs_sl = Observation(boxes=[_st0, _ent], frame=_fr, seq=1, w=3840, h=2160)
    _a_sl = _cpm.decide(_obs_sl, _SV(page="campaign_stage", frames_in_page=10))
    check("关号读不出时不点入場（等 OCR, 不进错关）",
          _a_sl is not None and _a_sl.kind == "wait", f"{_a_sl and _a_sl.kind}")
    _cfg2 = cfg()
    _cfg2["campaign"] = {"stage": "1-2"}
    _cp2 = ALL["campaign"](_Ctx2 := Ctx(cfg=_cfg2, log=lambda m: None))
    _cp2.goto("walk")
    # 走格子帧(fixture 同前): 我方箭头在起点上方, 回合0 = right-up... 1-2 的
    #   plan 是 right-up/right, 但 fixture 地图没有 right-up 格 -> fail-closed
    # 2026-08-24: 绑格锚**只认队伍箭头**(509 我方框池子里 40% 是敌方, 已废案)。
    #    箭头比立绘更高, 但 below() 的判据是同 x 带内正下方最近的格, 不受影响。
    _wb = list(_gb) + [B(V.GRID_ARROW, conf=0.90, cx=0.385, cy=0.377),
                       B(V.PHASE_END, conf=0.90, cx=0.92, cy=0.88)]
    _wo = Observation(boxes=_wb, seq=2, w=3840, h=2160)
    _act_w = None
    for _ in range(25):
        _act_w = _cp2.decide(_wo, _SV(page="grid_quest", frames_in_page=10))
        if _cp2.outcome:
            break
    check("答案方向落不到检出格子 — fail-closed 收 UNKNOWN 不瞎点",
          _cp2.outcome == "UNKNOWN", f"{_cp2.outcome} act={_act_w}")
    # 换一个方向能解析的答案: 手工喂 plan right-down -> 应该点到 (0.429,0.603)
    _cp3 = ALL["campaign"](Ctx(cfg=_cfg2, log=lambda m: None))
    _cp3.goto("walk")
    _cp3.state["answer"] = {
        "stage": "x", "type": "normal",
        "teams": [{"name": "A", "attr": "any", "pos": "center"}],
        "rounds": [[{"team": "A", "do": "move", "dir": "right-down"}]],
        "needs": {"teams": 1, "portal": False, "exchange": False, "attrs": []}}
    _act3 = None
    for _ in range(3):      # 绑格两帧共识: 首帧 wait, 第二帧才许落子
        _act3 = _cp3.decide(_wo, _SV(page="grid_quest", frames_in_page=10))
        if _act3 is not None and _act3.kind == "tap":
            break
    check("我方回合按答案点目标格心（绑格两帧共识后 tap）",
          _act3 is not None and _act3.kind == "tap"
          and abs(_act3.x - 0.429) < 0.02 and abs(_act3.y - 0.603) < 0.02,
          f"{_act3 and (_act3.kind, round(_act3.x,3), round(_act3.y,3))}")
    # 战斗期: 检出 AUTO 关着才点它（状态钮绝不盲 toggle）
    _bt_on = O(B(V.BATTLE_AUTO_ON, cx=0.9, cy=0.05), B(V.BATTLE_PAUSE, cx=0.95, cy=0.05))
    _a_on = _cp3.decide(_bt_on, _SV(page="battle", frames_in_page=10))
    check("AUTO 已开就不碰（战斗期只等待）",
          _a_on is not None and _a_on.kind == "wait", f"{_a_on and _a_on.kind}")
    _bt_off = O(B(V.BATTLE_AUTO_OFF, cx=0.9, cy=0.05), B(V.BATTLE_PAUSE, cx=0.95, cy=0.05))
    _a_off = _cp3.decide(_bt_off, _SV(page="battle", frames_in_page=10))
    check("检出 AUTO 关着才点开",
          _a_off is not None and _a_off.target_cls == V.BATTLE_AUTO_OFF,
          f"{_a_off and _a_off.target_cls}")
    # 进关前能力预检: 11-1 要 2 队 + portal + 属性队, 现有 flow 走不了 --
    #    必须 BLOCKED 在 enter 相位（花 AP 之前）, 不是进关走到那一回合才死
    _cfgb = cfg()
    _cfgb["campaign"] = {"stage": "11-1"}
    _cpb = ALL["campaign"](Ctx(cfg=_cfgb, log=lambda m: None))
    _ab = _cpb.decide(O(), _SV(page="task_hall", frames_in_page=5))
    check("答案要多队/portal -- 进关前预检 BLOCKED, AP 一分不花",
          _cpb.outcome == "BLOCKED"
          and any("进关前拦下" in l for l in _cpb.note_lines),
          f"{_cpb.outcome} {_cpb.note_lines}")
    # 真多区域: 计划没走完游戏弹回部署屏 -> 回 grid 相位重新部署,
    #    round_i 不动（继续同一份答案, 不是换"下一区域的解法"）
    _cpr = ALL["campaign"](Ctx(cfg=_cfg2, log=lambda m: None))
    _cpr.goto("walk")
    _cpr.state["answer"] = {
        "stage": "x", "type": "hard",
        "teams": [{"name": "A", "attr": "any", "pos": "center"}],
        "rounds": [[{"team": "A", "do": "move", "dir": "right"}],
                   [{"team": "A", "do": "move", "dir": "right"}]],
        "needs": {"teams": 1, "portal": False, "exchange": False, "attrs": []}}
    _cpr.state["round_i"] = 1
    _cpr.state["cell_acc"] = [(0.5, 0.5)]
    _o_rd = O(B(V.SORTIE, cx=0.88, cy=0.92))
    _a_rd = _cpr.decide(_o_rd, _SV(page="formation", frames_in_page=5))
    check("走位中被要求重新部署 -> 回 grid 相位且 round_i 不重置",
          _cpr.phase == "grid" and _cpr.state["round_i"] == 1
          and _cpr.state["cell_acc"] == [],
          f"phase={_cpr.phase} round_i={_cpr.state['round_i']}")
    # 墙钟卡死闸: 感知缺位的裸等待有界（75s 等不到 箭头+我方 就诚实收工,
    #    不再静默空转到 phase_cap -- 用户观感的「卡住不动」）
    import time as _tm
    _cpt = ALL["campaign"](Ctx(cfg=_cfg2, log=lambda m: None))
    _cpt.goto("walk")
    _cpt.state["answer"] = _cp3.state["answer"]
    # 起点也拿掉: 起点在场时航位推算本来就该走下去(见下一条正向用例)
    _wb_nou = [b for b in _wb
               if b.cls not in (V.GRID_ARROW, V.GRID_START, V.GRID_START_GREY)]
    _cpt.state["wt:no_unit"] = _tm.time() - 80
    _a_t = _cpt.decide(Observation(boxes=_wb_nou, seq=3, w=3840, h=2160),
                       _SV(page="grid_quest", frames_in_page=10))
    check("起点航位和队伍箭头都拿不到 75s -> UNKNOWN 收工不空转",
          _cpt.outcome == "UNKNOWN"
          and any("感知不足" in l for l in _cpt.note_lines),
          f"{_cpt.outcome} {_cpt.note_lines}")
    # 航位推算: 我方立绘全没检出(被自己格子挡/敌我混淆), 起点地标 + 已执行
    #    方向累加照样能落子（H2-2 r2 实锤: 立绘绑格绑到隔壁起点格 ->
    #    目标解析成自己站的格 -> 8 发全点在自己脚下）
    _cpe = ALL["campaign"](Ctx(cfg=_cfg2, log=lambda m: None))
    _cpe.goto("walk")
    _cpe.state["answer"] = _cp3.state["answer"]
    _wb_noally = [b for b in _wb if b.cls != V.GRID_ARROW]
    _a_dr = None
    for _ in range(3):
        _a_dr = _cpe.decide(
            Observation(boxes=_wb_noally, seq=6, w=3840, h=2160),
            _SV(page="grid_quest", frames_in_page=10))
        if _a_dr is not None and _a_dr.kind == "tap":
            break
    check("我方立绘全没检出 -> 起点航位推算照样落子",
          _a_dr is not None and _a_dr.kind == "tap"
          and abs(_a_dr.x - 0.429) < 0.03,
          f"{_a_dr and (_a_dr.kind, round(_a_dr.x, 3), round(_a_dr.y, 3))}")
    # step CLI 不存相位: 战斗页从 enter 起手必须接手 walk, 不能干等大厅
    _cpbt = ALL["campaign"](Ctx(cfg=_cfg2, log=lambda m: None))
    _cpbt.state.update(issued=True, bind_last=(0.46, 0.50),
                       answer=_cp3.state["answer"])
    _o_bt = O(B(V.BATTLE_AUTO_ON, cx=0.95, cy=0.94),
              B(V.BATTLE_PAUSE, cx=0.96, cy=0.06))
    _a_bt = _cpbt.decide(_o_bt, _SV(page="battle", frames_in_page=5))
    check("enter 遇战斗页接手 walk, 不干等大厅",
          _cpbt.phase == "walk"
          and _a_bt is not None and _a_bt.kind == "wait"
          and "任务大厅" not in (_a_bt.reason or ""),
          f"phase={_cpbt.phase} act={_a_bt}")
    _a_bt2 = _cpbt.decide(_o_bt, _SV(page="battle", frames_in_page=6))
    check("接手后战斗页走 AUTO 等待",
          _a_bt2 is not None and _a_bt2.kind == "wait"
          and "战斗" in (_a_bt2.reason or ""),
          f"{_a_bt2}")
    # 相位写进 state 后, 新实例续上 walk 不再经 enter
    _cprp = ALL["campaign"](Ctx(cfg=_cfg2, log=lambda m: None))
    _cprp.state.update(phase="walk", issued=True,
                       answer=_cp3.state["answer"])
    _a_rp = _cprp.decide(_o_bt, _SV(page="battle", frames_in_page=3))
    check("state.phase=walk 续上相位, 直接战斗等待",
          _cprp.phase == "walk"
          and _a_rp is not None and "战斗" in (_a_rp.reason or ""),
          f"phase={_cprp.phase} act={_a_rp}")
    # 区域对齐先于列表滚动: 配 H2-1 而列表在 Area3(读到 H3-x) -> 点 左切换
    #    换区, 不是在区域内滚 6 次然后 UNKNOWN（2026-08-13 live 用户抓到）
    import routing_v2.flow.campaign as _cpm_mod
    _cfgh = cfg()
    _cfgh["campaign"] = {"stage": "H2-1"}
    _cph = ALL["campaign"](Ctx(cfg=_cfgh, log=lambda m: None))
    _cph.goto("stage_list")
    _o_h3 = Observation(
        boxes=[B(V.STAGE_HARD_SEL, cx=0.832, cy=0.219),
               B(V.STAR_0, cx=0.562, cy=0.369, w=0.04, h=0.03),
               B(V.STAGE_ENTER_LOCKED, cx=0.875, cy=0.344),
               B(V.STAGE_ENTER, cx=0.875, cy=0.503),
               B(V.ARROW_LEFT, cx=0.028, cy=0.498)],
        frame=_fr, seq=4, w=3840, h=2160)
    _orig_digits = _cpm_mod.R.digits
    _cpm_mod.R.digits = lambda frame, rect: "3-1"
    try:
        _a_h = _cph.decide(_o_h3, _SV(page="campaign_stage", frames_in_page=10))
    finally:
        _cpm_mod.R.digits = _orig_digits
    check("目标关在别的区域 -> 点 左切换 换区, 不在区域内瞎滚",
          _a_h is not None and _a_h.kind == "tap"
          and _a_h.target_cls == V.ARROW_LEFT,
          f"{_a_h and (_a_h.kind, _a_h.target_cls)}")
    # 相位循环是瞬时证据, observe() 粘住: PHASE 缺席 3 帧就算循环,
    #    **页面身份抖成 unknown 也照样感知**（08-13 H2-1 live: 循环发生在
    #    page 抖动的帧里, 旧版 do_walk 只在 grid_quest 页看 -> 时钟瞎掉）
    _cpc2 = ALL["campaign"](Ctx(cfg=_cfg2, log=lambda m: None))
    _cpc2.goto("walk")
    _cpc2.state["answer"] = {
        "stage": "x", "type": "hard",
        "teams": [{"name": "A", "attr": "any", "pos": "center"}],
        "rounds": [[{"team": "A", "do": "move", "dir": "right-down"}],
                   [{"team": "A", "do": "move", "dir": "right-down"}]],
        "needs": {"teams": 1, "portal": False, "exchange": False, "attrs": []}}
    _cpc2.state.update(issued=True, cycling=False)
    _o_nope = O(B(V.GRID_CELL, cx=0.5, cy=0.5))      # 没有 PHASE 控件
    for _ in range(3):
        _cpc2.decide(_o_nope, _SV(page="unknown", frames_in_page=2))
    check("PHASE 缺席 3 帧（页面抖成 unknown）也感知到循环",
          _cpc2.state.get("cycling") is True,
          f"cycling={_cpc2.state.get('cycling')} absent={_cpc2.state.get('pe_absent')}")
    _cpc2.decide(_wo, _SV(page="grid_quest", frames_in_page=5))
    check("PHASE 回来 -> 回合推进到 2",
          _cpc2.state["round_i"] == 1, f"round_i={_cpc2.state['round_i']}")
    # 反向: 单帧缺席不算循环（旧版单帧置位会造成假回合推进）
    _cpc3 = ALL["campaign"](Ctx(cfg=_cfg2, log=lambda m: None))
    _cpc3.goto("walk")
    _cpc3.state["answer"] = _cpc2.state["answer"]
    _cpc3.state.update(issued=True, cycling=False)
    _cpc3.decide(_o_nope, _SV(page="unknown", frames_in_page=2))
    _cpc3.decide(_wo, _SV(page="grid_quest", frames_in_page=5))
    check("单帧 PHASE 缺席不算循环, 不假推进回合",
          _cpc3.state.get("cycling") is not True
          and _cpc3.state["round_i"] == 0,
          f"cycling={_cpc3.state.get('cycling')} round_i={_cpc3.state['round_i']}")
    # 位移证据时钟: 迷雾图敌方相位 HUD 闪没只有 1-2 帧, 3 帧判据抓不到
    #    （08-13 H2-1 两轮实锤: 步走成了回合时钟没走）。单位相对起点地标的
    #    向量是相机不变量, 位移一格量级 + 6s 内没等到 PHASE 闪 -> 判循环。
    _cpd = ALL["campaign"](Ctx(cfg=_cfg2, log=lambda m: None))
    _cpd.goto("walk")
    _cpd.state["answer"] = _cpc2.state["answer"]
    _cpd.state.update(issued=True, cycling=False, pre_vec=(0.0, -0.15),
                      dx_est=0.093, moved_frames=0)
    _o_mv = O(B(V.PHASE_END, cx=0.92, cy=0.88),
              B(V.GRID_START_GREY, cx=0.40, cy=0.55),
              B(V.GRID_ARROW, cx=0.49, cy=0.35),
              B(V.GRID_CELL, cx=0.49, cy=0.52))
    for _ in range(2):
        _cpd.decide(_o_mv, _SV(page="grid_quest", frames_in_page=8))
    check("位移证据被观测到（相对起点移动一格量级, 连续两帧）",
          _cpd.state.get("moved_t") is not None,
          f"moved_t={_cpd.state.get('moved_t')} frames={_cpd.state.get('moved_frames')}")
    _cpd.state["moved_t"] -= 7.0
    _cpd.decide(_o_mv, _SV(page="grid_quest", frames_in_page=9))
    check("PHASE 没闪但位移超时 -> 按位移判循环并推进回合",
          _cpd.state["round_i"] == 1,
          f"round_i={_cpd.state['round_i']} cycling={_cpd.state.get('cycling')}")
    # 08-15 3-2 replay: 位移已记下但 PHASE 不闪, 150 tick 重发清掉 moved_t,
    #    人已在目标格再点 = 谎报没走动。位移确认后禁止重发。
    _cpmv = ALL["campaign"](Ctx(cfg=_cfg2, log=lambda m: None))
    _cpmv.goto("walk")
    _cpmv.state["answer"] = _cpc2.state["answer"]
    _cpmv.state.update(issued=True, cycling=False, moved_t=_tm.time(),
                       round_i=1, dx_est=0.093, pre_vec=(0.04, -0.22))
    _a_mv = None
    for _ in range(155):
        _a_mv = _cpmv.decide(_wo, _SV(page="grid_quest", frames_in_page=8))
        if _cpmv.outcome:
            break
    check("位移已确认后 150 tick 不重发、不报没走动",
          _cpmv.outcome is None and _cpmv.state["round_i"] == 1
          and _cpmv.state.get("issued") is True
          and _cpmv.state.get("moved_t") is not None,
          f"out={_cpmv.outcome} ri={_cpmv.state['round_i']} "
          f"issued={_cpmv.state.get('issued')} act={_a_mv}")

    # 连续推关: stages 列表可跳号、可 Normal+Hard 混; 空则退回 stage 单关
    from routing_v2.flow.campaign import parse_stage_id as _psid
    from routing_v2.flow.campaign import resolve_queue as _rq
    _qm, _badm = _rq({"stages": ["3-2", "H2-1", "3-4"]})
    check("解析混排保序（不按关号排序）",
          _qm == ["3-2", "H2-1", "3-4"] and not _badm, str(_qm))
    _qd, _ = _rq({"stages": ["3-2", "3-2", "H2-1"]})
    check("去重保持用户先写的顺序", _qd == ["3-2", "H2-1"], str(_qd))
    _qs, _bads = _rq({"stage": "3-2", "stages": []})
    check("单关 stage 兼容（stages 空退回 stage）",
          _qs == ["3-2"] and not _bads, str(_qs))
    _qstr, _ = _rq({"stages": "3-2, H2-1, 3-4"})
    check("逗号串也按用户顺序拆",
          _qstr == ["3-2", "H2-1", "3-4"], str(_qstr))
    check("关号规范: h2-1 -> H2-1 / 非法拒收",
          _psid("h2-1") == "H2-1" and _psid("TR-5") is None
          and _psid("foo") is None and _psid("3-2") == "3-2",
          f"{_psid('h2-1')},{_psid('TR-5')},{_psid('foo')}")
    from routing_v2.config.schema import SCHEMA as _SC_ST
    _sc_st = _SC_ST.get("campaign.stages") or {}
    check("FORM 对外文案是要推的关卡, 不暴露 JSON 术语",
          _sc_st.get("label") == "要推的关卡"
          and "stages" not in (_sc_st.get("note") or "")
          and "3-1, 3-3, H2-1" in ((_sc_st.get("placeholder") or "")
                                   + (_sc_st.get("note") or "")),
          str(_sc_st))

    _cfgq = cfg()
    _cfgq["campaign"] = {"stages": ["1-2", "1-3"]}
    _cpq = ALL["campaign"](Ctx(cfg=_cfgq, log=lambda m: None))
    check("setup 队列第一关是当前关并已 load_answer",
          _cpq.state["queue"] == ["1-2", "1-3"]
          and _cpq.state["stage"] == "1-2"
          and _cpq.state.get("answer") is not None
          and _cpq.state["queue_i"] == 0,
          f"q={_cpq.state.get('queue')} st={_cpq.state.get('stage')}")
    _cpq.goto("result")
    _cpq.state["round_i"] = 2
    _cpq.state["battles"] = 1
    _a_qn = _cpq.decide(O(), _SV(page="campaign_stage", frames_in_page=5))
    check("一关 CLEAN 后相位回 stage_list 且 stage 变成下一关",
          _cpq.outcome is None
          and _cpq.phase == "stage_list"
          and _cpq.state["stage"] == "1-3"
          and _cpq.state.get("done") == ["1-2"]
          and _cpq.state.get("row_anchor") is None
          and _a_qn is not None and _a_qn.kind == "wait",
          f"out={_cpq.outcome} ph={_cpq.phase} st={_cpq.state.get('stage')}"
          f" done={_cpq.state.get('done')} act={_a_qn and _a_qn.kind}")

    _cfgu = cfg()
    _cfgu["campaign"] = {"stages": ["1-2", "1-3"]}
    _cpuq = ALL["campaign"](Ctx(cfg=_cfgu, log=lambda m: None))
    _cpuq.goto("walk")
    _act_uq = None
    for _ in range(25):
        _act_uq = _cpuq.decide(_wo, _SV(page="grid_quest", frames_in_page=10))
        if _cpuq.outcome:
            break
    check("UNKNOWN 不推进队列（停在这一关, 不偷打下一关）",
          _cpuq.outcome == "UNKNOWN"
          and _cpuq.state["queue_i"] == 0
          and _cpuq.state["stage"] == "1-2"
          and not _cpuq.state.get("done")
          and any("停在 1-2" in l for l in _cpuq.note_lines),
          f"{_cpuq.outcome} i={_cpuq.state.get('queue_i')} "
          f"st={_cpuq.state.get('stage')} notes={_cpuq.note_lines}")

    _cfgs = cfg()
    _cfgs["campaign"] = {"stages": ["99-1", "1-2"]}
    _cpsk = ALL["campaign"](Ctx(cfg=_cfgs, log=lambda m: None))
    _a_sk = _cpsk.decide(O(B(V.HUB_CAMPAIGN, cx=0.30, cy=0.40)),
                         _SV(page="task_hall", frames_in_page=5))
    check("无答案的关跳过, 改打下一关, 不卡死整条",
          _cpsk.outcome is None
          and _cpsk.state["stage"] == "1-2"
          and "99-1" in (_cpsk.state.get("skipped") or [])
          and _a_sk is not None and _a_sk.kind != "done",
          f"out={_cpsk.outcome} st={_cpsk.state.get('stage')} "
          f"skip={_cpsk.state.get('skipped')} act={_a_sk and _a_sk.kind}")

    _cfgi = cfg()
    _cfgi["campaign"] = {"stages": ["TR-5", "foo"]}
    _cpi = ALL["campaign"](Ctx(cfg=_cfgi, log=lambda m: None))
    _a_i = _cpi.decide(O(B(V.HUB_CAMPAIGN, cx=0.30, cy=0.40)),
                       _SV(page="task_hall", frames_in_page=5))
    check("非法号 setup/decide 不进关",
          _cpi.outcome == "BLOCKED"
          and _a_i is not None and _a_i.kind == "done"
          and any("关卡号非法" in l for l in _cpi.note_lines),
          f"{_cpi.outcome} act={_a_i and _a_i.kind} notes={_cpi.note_lines}")

    # 大赛商店: 余额读数不许一票否决, 必须**勾选探针**（用户 2026-08-13:
    #   「也没选饮料然后辨别是否买得起啊？」）。余额 0 也要点饮料, 然后
    #   由「選擇購買」的亮/灰给结论。
    _as = ALL["shop"](Ctx(cfg=cfg(), log=lambda m: None))
    _shelf = O(B(V.ARENA_SHOP_TAB_SEL, cx=0.06, cy=0.51),
               B(V.ARENA_SHOP_CURRENCY, cx=0.56, cy=0.08),   # 顶栏余额锚(读不出数)
               B(V.ARENA_SHOP_CURRENCY, cx=0.74, cy=0.55),
               B(V.ARENA_SHOP_CURRENCY, cx=0.86, cy=0.55),
               B(V.ENERGY_DRINK_LOW, cx=0.76, cy=0.49),
               B(V.ENERGY_DRINK_MID, cx=0.90, cy=0.49))
    _a_p = _as.on_arena_shop(_shelf, _SV(page="arena_shop", frames_in_page=10))
    check("余额读不出/为0 也要勾选饮料当探针（不许看一眼就走）",
          _a_p is not None and _a_p.kind == "tap"
          and _a_p.target_cls in (V.ENERGY_DRINK_LOW, V.ENERGY_DRINK_MID),
          f"{_a_p and (_a_p.kind, _a_p.target_cls)}")
    _asx = ALL["shop"](Ctx(cfg=cfg(), log=lambda m: None))
    _cross = O(B(V.ARENA_SHOP_TAB_SEL, cx=0.06, cy=0.51),
               B(V.ARENA_SHOP_CURRENCY, cx=0.56, cy=0.08),
               B(V.ARENA_SHOP_CURRENCY, cx=0.74, cy=0.55),
               B(V.ENERGY_DRINK_LOW, cx=0.90, cy=0.49),
               B(V.ENERGY_DRINK_MID, cx=0.76, cy=0.49))
    for _ in range(4):
        _asx.on_arena_shop(_cross, _SV(page="arena_shop", frames_in_page=10))
    check("472/473 左右反了拒勾",
          _asx.state.get("drink_side_dirty") is True
          and _asx.state.get(f"picked:{V.ENERGY_DRINK_LOW}")
          and _asx.state.get(f"picked:{V.ENERGY_DRINK_MID}"))
    # 勾上了、右下角出灰（本页实测 0.33, 判据带 region 降到 0.25）-> 买不起收工
    _as.state.update({f"picked:{V.ENERGY_DRINK_LOW}": True,
                      f"picked:{V.ENERGY_DRINK_MID}": True})
    _grey_shelf = O(B(V.ARENA_SHOP_TAB_SEL, cx=0.06, cy=0.51),
                    B(V.ARENA_SHOP_CURRENCY, cx=0.74, cy=0.55),
                    B(V.ARENA_SHOP_CURRENCY, cx=0.86, cy=0.55),
                    B(V.GREEN_CHECK, cx=0.76, cy=0.49),
                    B(V.SHOP_BUY_SELECTED_GREY, conf=0.30, cx=0.91, cy=0.92))
    for _ in range(10):
        _a_g2 = _as.decide(_grey_shelf, _SV(page="arena_shop", frames_in_page=20))
    check("灰的選擇購買(0.30) = 大赛币不够, 探针给结论收工",
          _as.state.get("arena_done") and _as.state.get("arena_short"),
          f"done={_as.state.get('arena_done')}")

    # 买不起 != 买过了（2026-08-13 小号实帧: 信用点 35,544 / 货最贵 500,000,
    #   屏上只出 `选择购买灰色` 0.78, 亮态零检出）。灰按钮点了也不动。
    _sh = ALL["shop"](Ctx(cfg=cfg(), log=lambda m: None))
    # 08-20 起 shop.credit_buy 默认关(schema.py:148), 信用点买路整段被开关罩住。
    #    本用例考的就是"开着时买不起怎么收敛", 所以按 cfg() 的规矩自己声明输入,
    #    不去改产品默认(同 t_shelf_walk 里 sh3.cfg["credit_buy"]=True 的写法)。
    _sh.cfg = dict(_sh.cfg or {})
    _sh.cfg["credit_buy"] = True
    _sh.state["pack_done"] = True
    _short = O(B(V.SHOP_TAB_CREDIT_SEL, cx=0.05, cy=0.12),
               B(V.GREEN_CHECK, cx=0.88, cy=0.20),
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
    # 放弃时面板还开着 -> 必须先关面板, 不许直接换相位（live: goto 抢在叉叉
    #   前面, 面板盖着屏, headpat/switch 整条尾巴全断）
    _cf10 = ALL["cafe"](Ctx(cfg=cfg(), log=lambda m: None))
    _cf10.goto("invite")
    _cf10.state["gaveup_f1"] = True
    _a_x = _cf10.decide(O(B(V.CAFE_INVITE, cx=0.62, cy=0.40),
                          B(V.CLOSE_X, cx=0.79, cy=0.16)),
                        _SV(page="cafe_invite_list", frames_in_page=10))
    check("放弃时面板开着要先叉掉, 不许带着面板换相位",
          _a_x is not None and _a_x.target_cls == V.CLOSE_X
          and _cf10.phase == "invite",
          f"{_a_x and _a_x.target_cls} phase={_cf10.phase}")
    # 邀请列表滑动几何（用户 2026-08-13 口述）: 横坐标在头像列和按钮列**中间**
    #   的空白区（不许压在按钮列上误触）, 起点在列表垂直中心（不许从底 bound 起手）
    _cf9 = ALL["cafe"](Ctx(cfg=cfg(), log=lambda m: None))
    _rows = [B(V.CAFE_INVITE, cx=0.780, cy=c) for c in (0.31, 0.42, 0.53, 0.63)]
    _avs = [Box(cls=n, conf=0.95, model="avatar",
                x1=0.415, y1=c - 0.03, x2=0.455, y2=c + 0.03)
            for n, c in (("凯伊", 0.31), ("季(战斗)", 0.42), ("冬", 0.53))]
    _obl = Observation(boxes=_rows + _avs, seq=9, w=3840, h=2160)
    _sw = _cf9._invite_swipe(_obl, "test", None)
    check("滑动横坐标在头像列(0.435)和按钮列(0.780)之间",
          _sw is not None and 0.50 < _sw.x < 0.72, f"x={_sw and _sw.x:.3f}")
    check("滑动起点在列表垂直中心, 不在底 bound",
          _sw is not None and 0.35 < _sw.y < 0.55, f"y={_sw and _sw.y:.3f}")
    # 到底判据 = 指纹重复 2 次（老代码 _invite_sig_repeat >= 2）, 次数只是安全帽
    _cf9.state.update(sig_repeat=2)
    _cf9.goto("invite")
    _cf9.cfg["invite_targets"] = ["优香"]
    _a_b = _cf9.decide(_obl, _SV(page="cafe_invite_list", frames_in_page=20))
    check("指纹重复 2 次 = 到底, 走放弃分支不再滑",
          _a_b is not None and _a_b.kind != "swipe", f"{_a_b and _a_b.kind}")
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
    _g3._pending["t0"] -= 9.0   # 测试跑得比墙钟快, 拨回去只测帧数半边
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
    # 时钟归 heartbeat（每帧走表, 模拟 runner 主循环）, advance 只判状态
    _rel = None
    for _i in range(1, 40):
        _g3.heartbeat(_opened, page_changed=False, retry_frames=25)
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
        _g5.heartbeat(O(B(V.CLOSE_X)), page_changed=True, retry_frames=70)
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
    _g6._pending["t0"] -= 9.0
    for _i in range(_MH + 2):
        _g6.heartbeat(O(B("别的")), page_changed=True, retry_frames=70)
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
    _g._pending["t0"] -= 9.0
    check("严格契约记下了 once 标记", _g._pending.get("once") == "selectall")
    # 超时和退标记归 heartbeat（时钟跟帧走 —— flow 光等待时也必须能超时）
    _rb0 = ""
    for _i in range(200):                       # 一直等不到「选择购买」
        _rb0 = _g.heartbeat(O(B("信用点")), page_changed=False,
                            retry_frames=25) or _rb0
        if _g._pending is None:
            break
    check("契约超时后把 once 标记退回来", _rb0 == "selectall", repr(_rb0))
    # 宽松契约（没显式 expect）超时属常态, 不该退标记
    _g2 = _G(cfg())
    _act2 = _A(kind="tap", x=0.5, y=0.5, target_cls="某个键", reason="r",
               once_key="k2")
    _g2.arm(_act2)
    _g2._pending["t0"] -= 9.0
    _rb2 = ""
    for _i in range(200):
        _rb2 = _g2.heartbeat(O(B("某个键")), page_changed=False,
                             retry_frames=25) or _rb2
        if _g2._pending is None:
            break
    check("宽松契约超时不退标记（那是常态，不是没生效）", not _rb2)

    # 2026-08-15: 补发/严格 与 宽松地板拆开, 宽松不随 retry 等比塌缩
    from routing_v2.act.gate import (
        _LOOSE_RETRY_FLOOR as _LRF, _contract_lim as _clim)
    check("retry=38 时宽松不低于地板(不会塌成 38//6=6)",
          _clim(True, 38) >= _LRF and _clim(True, 38) == max(_LRF, 38 // 6))
    check("retry=25 时宽松也不塌到 3-4 帧",
          _clim(True, 25) >= _LRF and _clim(True, 25) >= 8)
    check("严格档仍等于 retry_frames",
          _clim(False, 38) == 38 and _clim(False, 25) == 25)
    _g_loose = _G(cfg())
    _act_l = _A(kind="tap", x=0.5, y=0.5, target_cls="某键", reason="r")
    _g_loose.arm(_act_l, O(B("某键")))
    _g_loose._pending["t0"] -= 9.0
    _n_to = 0
    for _i in range(1, 80):
        _g_loose.heartbeat(O(B("某键")), page_changed=False, retry_frames=25)
        if _g_loose._pending is None:
            _n_to = _i
            break
    check("retry=25 宽松超时仍 >= 8 帧(地板兜住, 不是 4 帧连发)",
          _n_to >= 8, f"n={_n_to}")
    # 未完成契约时不许对第二个目标开火(防同时点多个东西)
    _g_two = _G(cfg())
    _first = _A(kind="tap", x=0.3, y=0.4, target_cls="键A", reason="点A")
    _second = _A(kind="tap", x=0.6, y=0.4, target_cls="键B", reason="点B")
    _obs_ab = O(B("键A"), B("键B"))
    _g_two.arm(_first, _obs_ab)
    _g_two._pending["t0"] -= 9.0
    for _ in range(6):
        _g_two.heartbeat(_obs_ab, page_changed=False, retry_frames=38)
    _v_two = _g_two.allow(_second, _obs_ab, fresh=lambda: None,
                          page_changed=False, frames_in_page=1,
                          retry_frames=38)
    check("契约未兑现时第二目标不许开火",
          (not _v_two.ok) and _v_two.by == "advance",
          f"ok={_v_two.ok} by={_v_two.by}")

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


def t_ledger():
    """余额台账的 OCR 结构闸（08-15 日常 live: 信用点收尾报 +5.99 亿假账）。"""
    print("\n 余额台账读数闸 ")
    import routing_v2.percept.read as _RD
    from routing_v2.act.ledger import Ledger, _one_insert

    led = Ledger(log=lambda m: None)
    led.path = _ROOT / "data" / "routing_v2" / "_test_ledger_tmp.jsonl"

    class _VS:                       # 打桩投票器: 直接喂确定读数
        vals: dict = {}
        def result(self):
            return dict(self.vals)
        def reset(self):
            pass

    led._vote = _VS()

    def push(v):
        _VS.vals = {_RD.CREDIT: v}
        led._commit("lobby", "test")

    try:
        # 首读双确认: 开跑第一读截断成 18,991, 不许直接立成基线
        push(18991)
        check("首读不立基线(要两次一致)", _RD.CREDIT not in led.confirmed)
        push(58723911)
        push(58723911)
        check("两次一致才立基线", led.confirmed.get(_RD.CREDIT) == 58723911,
              str(led.confirmed.get(_RD.CREDIT)))
        # 插位读大: 同页同渲染稳定复读, "复读一致"闸拦不住 —— 结构闸咬回
        push(588723911)
        push(588723911)
        check("插位读大稳定复读也咬得回(58,723,911->588,723,911)",
              led.confirmed.get(_RD.CREDIT) == 58723911,
              str(led.confirmed.get(_RD.CREDIT)))
        push(59227991)
        check("咬回后位数上限没被顶坏, 真读数照常入账",
              led.confirmed.get(_RD.CREDIT) == 59227991,
              str(led.confirmed.get(_RD.CREDIT)))
        check("千位逗号0形态(20,176->200176)被同一道闸覆盖",
              _one_insert("20176", "200176"))
        check("真实等长变动不误伤", not _one_insert("59227991", "59252667"))

        import time as _t
        from routing_v2.app.runner import (
            is_ledger_spend_tap, money_watch_should_halt)

        def push_pyx(ledx, v, page="lobby"):
            _VS.vals = {_RD.PYROXENE: v}
            return ledx._commit(page, "test")

        led_ext = Ledger(log=lambda m: None)
        led_ext.path = _ROOT / "data" / "routing_v2" / "_test_ledger_ext.jsonl"
        led_ext._vote = _VS()
        try:
            push_pyx(led_ext, 220)
            push_pyx(led_ext, 220)
            check("青辉石两次一致立基线",
                  led_ext.confirmed.get(_RD.PYROXENE) == 220)
            # 08-26 起单次读数不入账: 顶栏双稳态误读(2331/2316 半分钟摆
            #    3 次)每下摆一次就记一条假 EXTERNAL。第一读挂起, 换页复读
            #    同低值(或同页 3 次)才入账, 反弹作废。
            ext_msg = push_pyx(led_ext, 190)
            check("外部下降第一读挂起不入账",
                  ext_msg is None
                  and led_ext.confirmed.get(_RD.PYROXENE) == 220
                  and any(e.tag == "ext_suspect" for e in led_ext.entries))
            ext_msg = push_pyx(led_ext, 190, page="cafe")
            check("外部青辉石下降且无付费 tap 不 HALT",
                  ext_msg is not None and ext_msg.startswith("EXTERNAL")
                  and not money_watch_should_halt(ext_msg)
                  and led_ext.confirmed.get(_RD.PYROXENE) == 190
                  and led_ext.breach is None, str(ext_msg))
            check("外部下降写入 external 标签",
                  any(e.tag == "external" and e.cls == _RD.PYROXENE
                      for e in led_ext.entries))
            n_ext = sum(1 for e in led_ext.entries if e.tag == "external")
            push_pyx(led_ext, 175)
            push_pyx(led_ext, 190)
            check("读数抖动(降后反弹)不入外部账",
                  sum(1 for e in led_ext.entries if e.tag == "external") == n_ext
                  and led_ext._ext_suspect is None)
        finally:
            led_ext.path.unlink(missing_ok=True)

        led_bot = Ledger(log=lambda m: None)
        led_bot.path = _ROOT / "data" / "routing_v2" / "_test_ledger_bot.jsonl"
        led_bot._vote = _VS()
        try:
            push_pyx(led_bot, 250)
            push_pyx(led_bot, 250)
            led_bot.note_bot_spend("确认购买", "确认键", "青辉石")
            sus = push_pyx(led_bot, 220, "lobby")
            check("bot 付费后首读下降只挂起不 HALT",
                  sus is None and led_bot.confirmed.get(_RD.PYROXENE) == 250)
            br = push_pyx(led_bot, 220, "cafe")
            # 08-21 契约改: 台账是**事后账**, 首条掉钱只记账告警不停轮
            #    (真防线是 gate.py tap 前那两条)。第二条才升级停轮。
            check("bot 付费 tap 后青辉石下降落 WARN_MONEY 且不停轮",
                  br is not None and br.startswith("WARN_MONEY")
                  and not money_watch_should_halt(br)
                  and led_bot.breach is None, str(br))
            check("首条掉钱仍写 breach 标签 + 进 _breaches 明细",
                  len(led_bot._breaches) == 1
                  and any(e.tag == "breach" and e.cls == _RD.PYROXENE
                          for e in led_bot.entries))
            check("首条掉钱进收工报告", "掉钱告警 1 条" in led_bot.report())
            # 第二次: 连续掉钱 -> 停轮
            led_bot.note_bot_spend("确认购买", "确认键", "青辉石")
            push_pyx(led_bot, 190, "lobby")
            br2 = push_pyx(led_bot, 190, "cafe")
            check("第二条掉钱升级成 MONEY BREACH 停轮",
                  br2 is not None and br2.startswith("MONEY BREACH")
                  and money_watch_should_halt(br2)
                  and led_bot.breach is not None, str(br2))
        finally:
            led_bot.path.unlink(missing_ok=True)

        check("购买键要记窗",
              is_ledger_spend_tap(Action(kind="tap", target_cls=V.SHOP_BUY,
                                         reason="买")))
        check("确认付费要记窗",
              is_ledger_spend_tap(Action(kind="tap", target_cls=V.CONFIRM,
                                         reason="确认"), V.SHOP_BUY))
        check("无上一发的确认键不记窗",
              not is_ledger_spend_tap(Action(kind="tap", target_cls=V.CONFIRM,
                                             reason="确认")))
        check("出击不记窗",
              not is_ledger_spend_tap(Action(kind="tap", target_cls=V.SORTIE,
                                             reason="出击")))

        led_jit = Ledger(log=lambda m: None)
        led_jit.path = _ROOT / "data" / "routing_v2" / "_test_ledger_jit.jsonl"
        led_jit._vote = _VS()
        try:
            push_pyx(led_jit, 250)
            push_pyx(led_jit, 250)
            # JIT 丢掉: runner 以 uncertain 开青辉石窗
            led_jit.note_bot_spend("购买", V.SHOP_BUY, "青辉石")
            sus_j = push_pyx(led_jit, 220, "lobby")
            br_j = push_pyx(led_jit, 220, "cafe")
            check("JIT 丢掉仍记窗: 石头下降仍被抓成掉钱告警",
                  sus_j is None
                  and br_j is not None and br_j.startswith("WARN_MONEY")
                  and len(led_jit._breaches) == 1, str(br_j))
        finally:
            led_jit.path.unlink(missing_ok=True)

        led_exp = Ledger(log=lambda m: None)
        led_exp.path = _ROOT / "data" / "routing_v2" / "_test_ledger_exp.jsonl"
        led_exp._vote = _VS()
        try:
            push_pyx(led_exp, 250)
            push_pyx(led_exp, 250)
            led_exp.note_bot_spend("确认购买", "确认键", "青辉石")
            sus_e = push_pyx(led_exp, 220, "lobby")
            check("窗内首读下降挂起", sus_e is None)
            led_exp._bot_spend_ts[_RD.PYROXENE] = _t.time() - 30
            br_e = push_pyx(led_exp, 220, "cafe")
            check("换页复读时窗已过仍抓成掉钱告警",
                  br_e is not None and br_e.startswith("WARN_MONEY")
                  and len(led_exp._breaches) == 1, str(br_e))
        finally:
            led_exp.path.unlink(missing_ok=True)

        led_ocr = Ledger(log=lambda m: None)
        led_ocr.path = _ROOT / "data" / "routing_v2" / "_test_ledger_ocr.jsonl"
        led_ocr._vote = _VS()
        try:
            _VS.vals = {_RD.CREDIT: 1097169}
            led_ocr._commit("lobby", "test")
            led_ocr._commit("lobby", "test")
            check("信用点 1097169 立基线",
                  led_ocr.confirmed.get(_RD.CREDIT) == 1097169)
            _VS.vals = {_RD.CREDIT: 97169}
            ocr_msg = led_ocr._commit("lobby", "test")
            check("信用点 1097169 截成 97169 不改基线也不 HALT",
                  led_ocr.confirmed.get(_RD.CREDIT) == 1097169
                  and not money_watch_should_halt(ocr_msg),
                  str(led_ocr.confirmed.get(_RD.CREDIT)))
        finally:
            led_ocr.path.unlink(missing_ok=True)

        led_up = Ledger(log=lambda m: None)
        led_up.path = _ROOT / "data" / "routing_v2" / "_test_ledger_up.jsonl"
        led_up._vote = _VS()
        try:
            _VS.vals = {_RD.CREDIT: 1097169}
            led_up._commit("lobby", "test")
            led_up._commit("lobby", "test")
            _VS.vals = {_RD.CREDIT: 1050000}
            up_msg = led_up._commit("lobby", "test")
            check("用户升级花信用点不 HALT, 记外部变动",
                  up_msg is not None and up_msg.startswith("EXTERNAL")
                  and not money_watch_should_halt(up_msg)
                  and led_up.confirmed.get(_RD.CREDIT) == 1050000,
                  str(up_msg))
        finally:
            led_up.path.unlink(missing_ok=True)

        check("runner: EXTERNAL 不停轮",
              not money_watch_should_halt("EXTERNAL: 青辉石 220 降到 190"))
        check("runner: BREACH 停轮",
              money_watch_should_halt("MONEY BREACH: 青辉石 220 降到 190"))
        check("runner: 空 watch 不停轮",
              not money_watch_should_halt(None))
    finally:
        led.path.unlink(missing_ok=True)


def t_event_bonus_shop():
    """08-15 复盘三修: 兜底过台账 / 台账写入口防测试污染 / 商店拒买不拒推算。"""
    print("\n 活动加成台账与商店扫买 ")
    import routing_v2.percept.read as _RD
    _orig_topbar = _RD.read_topbar
    _RD.read_topbar = lambda o, c: (500 if c == _RD.AP else _orig_topbar(o, c))
    lst2 = O(B(V.EVENT_SHOP, cx=0.1, cy=0.5), B(V.EVENT_QUEST_SEL, cx=0.6, cy=0.15),
             B(V.STAGE_ENTER, cx=0.9, cy=0.70), B(V.STAR_3, cx=0.6, cy=0.70),
             B(V.AP, conf=0.9, cx=0.40, cy=0.033))
    try:
        #  修1: plan 空(推算被金钱闸拦掉/文件过期)时, 兜底关也要过台账
        #    08-15 实锤: 台账里 02:24 刚顶过倒数第 1 关, 第三通道 plan 空,
        #    跳过循环被 `plan and ...` 短路 -> 又打了一场(20AP 白花)。
        _ev = ALL["event"](Ctx(cfg=cfg(), log=lambda m: None))
        _ev.state["phase"] = "bonus_clear"
        # 08-26 起老字符串条目一律无效(上期 10 条跨期串关烧了 820 AP),
        #    fixture 必须是 v2 条目(关号可 None, 兜底路走 fb 键)且当天有效。
        from routing_v2.flow.daybook import game_day as _gd
        _ev.ctx.bag["event_topped"] = {
            "0": {"stage": None, "fb": 0, "day": _gd(), "note": "fixture 已顶"}}
        _ev.ctx.bag["event_farm_plan"] = []
        _ev._plan_from_file = lambda: None       # 隔离真实计划文件
        _m = Machine(1)
        _ev.decide(lst2, _m.update(lst2))
        check("plan 空 + 台账已顶: 兜底关不再重打, 直接转扫荡",
              _ev.state["phase"] == "bonus_sweep",
              f"phase={_ev.state['phase']}")
        a = _ev.decide(lst2, _m.update(lst2))
        check("兜底通道对上台账: 扫荡放行而不是 BLOCKED",
              a is not None and a.kind != "done" and getattr(a, "is_tap", False)
              and "扫荡" in (a.reason or ""), str(a))

        #  修2: _topped_mark 对 bag fixture 只写内存, 真实台账文件不许碰
        #    08-12/08-15(x2) 三次实锤: 离线套件驱动"赢一场"路径把
        #    data/routing_v2/event_topped.json 真写了。
        _ev2 = ALL["event"](Ctx(cfg=cfg(), log=lambda m: None))
        _ev2.ctx.bag["event_topped"] = {}
        _tp = _ev2._topped_path()
        _before = _tp.read_bytes() if _tp.exists() else None
        _ev2._topped_mark(7, "fixture 打赢")
        check("bag fixture 台账: 落账写进内存",
              "fb:7" in _ev2.ctx.bag["event_topped"]
              and _ev2.ctx.bag["event_topped"]["fb:7"]["fb"] == 7)
        _ev2.state["cur_stage_no"] = 11
        _ev2._topped_mark(3, "fixture 打赢2")
        check("进关时记了关号的落账用语义键",
              "11" in _ev2.ctx.bag["event_topped"]
              and _ev2.ctx.bag["event_topped"]["11"]["stage"] == 11)
        check("老字符串条目一律判无效",
              not _ev2._topped_ok("08-12 03:38 部队2 打赢")
              and _ev2._topped_ok(_ev2.ctx.bag["event_topped"]["11"]))
        _after = _tp.read_bytes() if _tp.exists() else None
        check("真实台账文件未被测试写过", _before == _after)

        #  修3: event_shop 金钱被拒后不再试买, 但扫描推算照做
        #    附带回归: 滑动后停稳闸(假到底) + 自底向上单遍 + 双向滑幅同尺。
        _orig_coin = _RD.read_event_coin
        _RD.read_event_coin = lambda o: 189
        from routing_v2.config import data_dir as _dd
        _pf = _dd(cfg()) / "event_farm_plan.json"   # 08-15 分桶后在 _test 桶
        _pf_bytes = _pf.read_bytes() if _pf.exists() else None
        try:
            es = ALL["event_shop"](Ctx(cfg=cfg(), log=lambda m: None))
            es.state["moneyno:购买"] = 1          # 模拟第一发已被人审拒绝
            v1 = O(B(V.SHOP_BUY, cx=0.5, cy=0.40, w=0.05, h=0.03),
                   B(V.SHOP_BUY, cx=0.5, cy=0.62, w=0.05, h=0.03),
                   B(V.CURRENCY_SEL, cx=0.08, cy=0.30, w=0.06, h=0.04))
            v2 = O(B(V.SHOP_BUY, cx=0.5, cy=0.30, w=0.05, h=0.03),
                   B(V.SHOP_BUY, cx=0.5, cy=0.52, w=0.05, h=0.03),
                   B(V.CURRENCY_SEL, cx=0.08, cy=0.30, w=0.06, h=0.04))
            _stq = Machine(1).update(v1)
            acts = []

            def step(o):
                es.ticks += 1
                a = es.on_event_shop(o, _stq)
                acts.append(a)
                if a is not None and getattr(a, "post", None):
                    a.post()
                return a

            a1 = step(v1)
            check("下行第一滑是向下(起点在下终点在上)",
                  a1 is not None and a1.kind == "swipe" and a1.y > a1.y2,
                  str(a1))
            a2 = step(v2)
            check("滑动后画面未停稳不下结论(不固定 sleep)",
                  a2 is not None and a2.kind == "wait" and "停稳" in (a2.reason or ""),
                  str(a2))
            a_down = None
            for _ in range(6):
                a_down = step(v2)
                if a_down is not None and a_down.kind == "swipe" and a_down.y > a_down.y2:
                    break
            check("新货出现后继续下滑, 不提前向上",
                  a_down is not None and a_down.kind == "swipe" and a_down.y > a_down.y2,
                  str(a_down))
            # 08-26 起拒买 = 纯扫描: 下行触底即收 tab, 不再上翻回顶
            #    (用户实看"瞎滑"实锤: 扫描模式跑着买路的探底+回滑两趟)。
            saw_up = False
            a5 = None
            for _ in range(12):
                a5 = step(v2)
                if a5 is not None and a5.kind == "swipe" and a5.y < a5.y2:
                    saw_up = True
                    break
            check("拒买扫描触底即收 tab, 全程零上翻", not saw_up, str(a5))
            for _ in range(6):
                step(v1)
            fin = None
            for _ in range(8):
                fin = step(v1)
                if fin is not None and fin.kind == "done":
                    break
            check("金钱被拒也要扫完出推算(不再整条 BLOCKED 死掉)",
                  fin is not None and fin.kind == "done"
                  and es.outcome == "CLEAN"
                  and bool(es.ctx.bag.get("event_farm_plan")), str(fin))
            check("拒买状态下全程零 money 动作",
                  not any(getattr(x, "money", False) for x in acts if x))
        finally:
            _RD.read_event_coin = _orig_coin
            if _pf_bytes is not None:
                _pf.write_bytes(_pf_bytes)        # 计划文件不许被测试搞新鲜
            else:
                _pf.unlink(missing_ok=True)
    finally:
        _RD.read_topbar = _orig_topbar


def t_deadbtn():
    """死按钮复盘（08-15）: 假超时 = 契约把不弹框的键锁进死等确认/取消;
    归位假死 = 把被闸按住的提案数成连发。两刀分别在 gate.arm 和 _dead_tap。"""
    print("\n 死按钮: 契约拆档 + 归位只数真发 ")
    from routing_v2.act.action import tap_box as _tb
    from routing_v2.act.gate import _EXPECT_DIALOG_AFTER as _DLG_EVIDENCE

    ga = Gate(cfg(), log=lambda m: None)
    ga.arm(_tb(B(V.CLAIM_YELLOW, cx=0.50, cy=0.73), "领取"),
           O(B(V.CLAIM_YELLOW, cx=0.50, cy=0.73)))
    check("领取_黄 点完不再死等确认框(契约=自己消失)",
          ga._pending is not None and ga._pending["expect"] == ()
          and V.CLAIM_YELLOW in ga._pending["expect_gone"],
          str(ga._pending))
    _nxt = Action(kind="tap", x=0.50, y=0.88, reason="继续",
                  target_cls=V.STORY_TAP_CONTINUE)
    ga._pending["n"] = 5              # 拨过节拍闸(_MIN_HOLD): 只测兑现语义
    ga._pending["t0"] -= 5.0
    _v = ga.advance(_nxt, O(B(V.GOT_REWARD, cx=0.50, cy=0.22),
                            B(V.STORY_TAP_CONTINUE, cx=0.51, cy=0.88)),
                    page_changed=False, retry_frames=70)
    check("奖励层盖住领取键 = 契约几帧内兑现, 不再空等 70 帧", _v.ok, _v.why)

    gb = Gate(cfg(), log=lambda m: None)
    gb.arm(_tb(B(V.QTY_MAX, cx=0.85, cy=0.42), "MAX"), O())
    check("MAX_可点击=步进器不设契约(自己不消失, expect_gone 永不兑现)",
          gb._pending is None, str(gb._pending))

    gc2 = Gate(cfg(), log=lambda m: None)
    gc2.arm(_tb(B(V.SWEEP_START, cx=0.73, cy=0.56), "扫荡"), O())
    check("扫荡开始仍是严格档(真会弹确认框, 回归护栏)",
          gc2._pending is not None and V.CONFIRM in gc2._pending["expect"],
          str(gc2._pending))

    gd = Gate(cfg(), log=lambda m: None)
    gd.arm(_tb(B(V.TASK_START, cx=0.73, cy=0.75), "任务开始"), O())
    check("任务开始契约=弹窗自己关(直进编队, 不等确认框)",
          gd._pending is not None and gd._pending["expect"] == (),
          str(gd._pending))
    check("确认合法性证据表一个成员不动(领取/任务开始仍算 bot 请求过框)",
          V.CLAIM_YELLOW in _DLG_EVIDENCE and V.TASK_START in _DLG_EVIDENCE
          and V.QTY_MAX in _DLG_EVIDENCE)

    from routing_v2.app.runner import _dead_tap
    stp = {"key": None, "n": 0}
    kk = ("确认键", 0.5, 0.68)
    _seq = [_dead_tap(stp, kk, True)] + [_dead_tap(stp, kk, False)
                                         for _ in range(5)]
    check("1 发真 tap + 5 发被闸按住 != 死路(闸的正常工作不算按钮死)",
          not any(_seq), str(stp))
    check("真发满 4 次仍无变化才判死",
          not _dead_tap(stp, kk, True) and not _dead_tap(stp, kk, True)
          and _dead_tap(stp, kk, True), str(stp))
    stp2 = {"key": None, "n": 0}
    _dead_tap(stp2, kk, True)
    _dead_tap(stp2, kk, True)
    check("换了目标计数清零", not _dead_tap(stp2, ("返回键", 0.04, 0.05), True),
          str(stp2))


def t_ocr_geom():
    """icon_strip 天花板单位修复（08-15）: 天花板是布局量(别吃邻居行), 该随
    UI 等比缩放; 写成绝对像素后 4K 顶栏被夹瘦切首位(47,143,185->433185 /
    7,555->555), 而 1440p 全对 —— 单位错, 不是标定错。地板仍是真绝对像素
    (DB 检测器物性)。"""
    print("\n OCR 裁片几何 ")
    from routing_v2.percept.read import CREDIT, PYROXENE, STRIP, icon_strip
    xf, xt, yp, pmin, pmax = STRIP[CREDIT]
    b = Box(cls=CREDIT, conf=0.9, x1=0.500, y1=(720 - 24.7) / 1440,
            x2=0.500 + 54 / 2560, y2=(720 + 24.7) / 1440)
    r = icon_strip(b, xf, xt, yp, 1440, pmin, pmax)
    pad = (b.y1 - r[1]) * 1440
    check("1440p 信用点留白=图标倍数(~32px), 标定分辨率行为不动",
          31.0 < pad < 33.0, f"{pad:.1f}px")
    b4 = Box(cls=CREDIT, conf=0.9, x1=0.500, y1=(540 - 36.5) / 2160,
             x2=0.500 + 82 / 3840, y2=(540 + 36.5) / 2160)
    r = icon_strip(b4, xf, xt, yp, 2160, pmin, pmax)
    pad = (b4.y1 - r[1]) * 2160
    check("4K 信用点留白 ~47px 不再被 42 绝对像素天花板切首位",
          46.0 < pad < 49.0, f"{pad:.1f}px")
    xf, xt, yp, pmin, pmax = STRIP[PYROXENE]
    b4g = Box(cls=PYROXENE, conf=0.9, x1=0.700, y1=(115 - 38.5) / 2160,
              x2=0.700 + 77 / 3840, y2=(115 + 38.5) / 2160)
    r = icon_strip(b4g, xf, xt, yp, 2160, pmin, pmax)
    pad = (b4g.y1 - r[1]) * 2160
    check("4K 青辉石留白 ~62px, 天花板按 1440 参考等比放开(实测 555->7555)",
          58.0 < pad < 64.0, f"{pad:.1f}px")


def t_bucket():
    """台账账号分桶（08-15）: 落盘键过去只有游戏日/倒数第几关, 大小号换着跑
    互相把「今天做过/本期顶过」当成自己的账（ledger_20260813 同一份文件里
    05:48 大号 59M / 08:49 小号 35,544）。现在一切按 account.id 分桶。"""
    print("\n 台账账号分桶 ")
    from routing_v2.act.ledger import Ledger
    from routing_v2.config import data_dir, merged
    d = data_dir(cfg())
    check("台账桶 = data/routing_v2/<account.id>",
          d.name == "_test" and d.parent.name == "routing_v2", str(d))
    try:
        data_dir(merged({}))          # DEFAULTS 里 id 为空
        _refused = False
    except ValueError:
        _refused = True
    check("缺 account.id 拒绝开跑(fail-closed, 不往共享路径写账)", _refused)
    led = Ledger(log=lambda m: None, out_dir=d)
    check("ledger 落盘进账号桶", str(led.path).startswith(str(d)),
          str(led.path))
    e1 = ALL["event"](Ctx(cfg=cfg(), log=lambda m: None))
    check("event_topped 进账号桶", e1._topped_path().parent == d,
          str(e1._topped_path()))
    from routing_v2.flow import daybook as _db
    check("daybook 进账号桶", _db._file(cfg()).parent == d,
          str(_db._file(cfg())))
    s1 = ALL["schedule"](Ctx(cfg=cfg(), log=lambda m: None))
    check("课程表房间账进账号桶", s1._rooms_file().parent == d,
          str(s1._rooms_file()))
    d2 = data_dir(merged({"account": {"id": "_test2"}}))
    check("换号 = 换桶(互不可见)", d2 != d and d2.name == "_test2", str(d2))
    import shutil
    shutil.rmtree(d2, ignore_errors=True)


def t_alt_gates():
    """08-15 小号离线闸：锁入口、编队、归位、咖啡厅和账号分层。"""
    print("\n-- 小号五项离线闸 ----")
    import copy
    import json
    import cv2 as _cv
    import numpy as _np
    from routing_v2.flow import nav as _nv
    from routing_v2.flow.battle import formation_ready, formation_slot_saturation
    from routing_v2.state.machine import StateView as _SV

    tiles = {"bounty": V.HUB_BOUNTY, "jfd": V.HUB_JFD,
             "arena": V.HUB_ARENA}

    def _hall_obs(*extra, special=False):
        boxes = [
            B(V.HUB_STORY, cx=0.20, cy=0.45),
            B(V.HUB_CAMPAIGN, cx=0.40, cy=0.45),
            B(V.BACK, cx=0.05, cy=0.05),
        ]
        if special:
            boxes.append(B(V.HUB_SPECIAL, cx=0.60, cy=0.45))
        boxes.extend(extra)
        return O(*boxes)

    def _hall_state(page, overlay=None):
        return _SV(page=page, raw=page, overlay=overlay,
                   frames_in_page=10, last_solid="task_hall")

    one_anchor = O(B(V.HUB_STORY, cx=0.20, cy=0.45))
    check("任务大厅单锚只计数，不满足普通大厅证据",
          _nv.task_hall_anchor_count(one_anchor) == 1
          and not _nv.task_hall_evidence(one_anchor))

    for name, tile in tiles.items():
        flow = ALL[name](Ctx(cfg=cfg(), log=lambda m: None))
        actions = []
        for i in range(80):
            page = "task_hall" if i % 2 == 0 else "facility"
            actions.append(flow.decide(
                _hall_obs(special=i % 3 == 0),
                _hall_state(page)))
        tile_taps = [a for a in actions
                     if a is not None and a.is_tap and a.target_cls == tile]
        confirm_taps = [a for a in actions
                        if a is not None and a.is_tap
                        and a.target_cls == V.CONFIRM]
        check(f"{name} 真实零 tile 形态跨 task_hall/facility 80 帧后 SKIPPED",
              flow.outcome == "SKIPPED"
              and not tile_taps and not confirm_taps,
              f"outcome={flow.outcome} tile={len(tile_taps)} "
              f"confirm={len(confirm_taps)}")

    for name in tiles:
        flow = ALL[name](Ctx(cfg=cfg(), log=lambda m: None))
        no_evidence_actions = []
        for i in range(24):
            page = "facility" if i % 2 == 0 else "unknown"
            no_evidence_actions.append(flow.decide(
                O(B(V.BACK, cx=0.05, cy=0.05)),
                _hall_state(page)))
        check(f"{name} facility/unknown 无大厅证据不计 miss 也不按返回",
              flow.state.get("hub_tile_misses") == 0
              and flow.outcome is None
              and not any(a is not None and a.is_tap
                          for a in no_evidence_actions),
              f"miss={flow.state.get('hub_tile_misses')}")

    for name, tile in tiles.items():
        flow = ALL[name](Ctx(cfg=cfg(), log=lambda m: None))
        for i in range(40):
            page = "task_hall" if i % 2 == 0 else "facility"
            flow.decide(_hall_obs(special=i % 2 == 0), _hall_state(page))
        appeared = flow.decide(
            _hall_obs(B(tile, cx=0.72, cy=0.62), special=True),
            _hall_state("facility"))
        check(f"{name} 漏 40 帧后 tile 出现立即清 miss 并正常 tap",
              appeared is not None and appeared.is_tap
              and appeared.target_cls == tile
              and flow.state.get("hub_tile_misses") == 0,
              f"{appeared} miss={flow.state.get('hub_tile_misses')}")

    st_lock = _hall_state("facility", overlay="ack_dialog")
    for name, tile in tiles.items():
        flow = ALL[name](Ctx(cfg=cfg(), log=lambda m: None))
        first = flow.on_task_hall(
            _hall_obs(B(tile, cx=0.72, cy=0.62), special=True),
            _hall_state("task_hall"))
        check(f"{name} 第一次探测入口",
              first is not None and first.is_tap and first.target_cls == tile,
              str(first))
        if first is not None and first.post:
            first.post()
        rejected = flow.decide(
            _hall_obs(B(V.CONFIRM, cx=0.50, cy=0.70), special=True),
            st_lock)
        check(f"{name} 当前大厅证据下单键锁通知快速 SKIPPED",
              rejected is not None and rejected.kind == "done"
              and flow.outcome == "SKIPPED"
              and not (rejected.is_tap
                       and rejected.target_cls == V.CONFIRM),
              f"{rejected} outcome={flow.outcome}")

    for name in tiles:
        flow = ALL[name](Ctx(cfg=cfg(), log=lambda m: None))
        flow.state["entry_probe"] = True
        rejected = flow.decide(
            O(B(V.HUB_STORY, cx=0.20, cy=0.45),
              B(V.CONFIRM, cx=0.50, cy=0.70)),
            st_lock)
        check(f"{name} 单剧情锚锁通知快速 SKIPPED 且不点确认",
              rejected is not None and rejected.kind == "done"
              and flow.outcome == "SKIPPED"
              and not (rejected.is_tap
                       and rejected.target_cls == V.CONFIRM),
              f"{rejected} outcome={flow.outcome}")

    for name in tiles:
        flow = ALL[name](Ctx(cfg=cfg(), log=lambda m: None))
        flow.state["entry_probe"] = True
        notified = flow.decide(
            O(B(V.CONFIRM, cx=0.50, cy=0.70)),
            st_lock)
        check(f"{name} 无大厅锚内页通知绝不误 SKIP",
              flow.outcome != "SKIPPED"
              and notified is not None and notified.is_tap
              and notified.target_cls == V.CONFIRM,
              f"{notified} outcome={flow.outcome}")

    hard_evidence = (
        ("sweep ticket", "bounty", B(V.TICKET_BOUNTY, cx=0.08, cy=0.20)),
        ("sweep branch", "jfd", B(V.JFD_ACADEMIES[0], cx=0.70, cy=0.40)),
        ("sweep stage", "bounty", B(V.STAGE_ENTER, cx=0.82, cy=0.55)),
        ("arena ticket", "arena", B(V.TICKET_ARENA, cx=0.08, cy=0.20)),
        ("arena row", "arena", B(V.ARENA_ROW, cx=0.70, cy=0.40)),
        ("arena attack form", "arena",
         B(V.ARENA_ATTACK_FORM, cx=0.72, cy=0.70)),
    )
    for case, name, inside_box in hard_evidence:
        tile = tiles[name]
        flow = ALL[name](Ctx(cfg=cfg(), log=lambda m: None))
        first = flow.on_task_hall(
            _hall_obs(B(tile, cx=0.72, cy=0.62), special=True),
            _hall_state("task_hall"))
        if first is not None and first.post:
            first.post()
        raced = flow.decide(
            _hall_obs(inside_box,
                      B(V.CONFIRM, cx=0.50, cy=0.70),
                      special=True),
            st_lock)
        check(f"{case} 进内页通知竞态绝不误 SKIP",
              flow.outcome != "SKIPPED"
              and raced is not None and raced.is_tap
              and raced.target_cls == V.CONFIRM,
              f"{raced} outcome={flow.outcome}")

    for case, name, inside_box in hard_evidence:
        flow = ALL[name](Ctx(cfg=cfg(), log=lambda m: None))
        flow.state["entry_probe"] = True
        raced = flow.decide(
            O(B(V.HUB_STORY, cx=0.20, cy=0.45),
              inside_box,
              B(V.CONFIRM, cx=0.50, cy=0.70)),
            st_lock)
        check(f"{case} 单锚内页硬证据绝不误 SKIP",
              flow.outcome != "SKIPPED"
              and raced is not None and raced.is_tap
              and raced.target_cls == V.CONFIRM,
              f"{raced} outcome={flow.outcome}")

    for case, name, inside_box in hard_evidence:
        flow = ALL[name](Ctx(cfg=cfg(), log=lambda m: None))
        flow.state["entry_probe"] = True
        raced = flow.decide(
            O(inside_box, B(V.CONFIRM, cx=0.50, cy=0.70)),
            st_lock)
        check(f"{case} 无大厅锚内页硬证据绝不误 SKIP",
              flow.outcome != "SKIPPED"
              and raced is not None and raced.is_tap
              and raced.target_cls == V.CONFIRM,
              f"{raced} outcome={flow.outcome}")

    event = ALL["event"](Ctx(cfg=cfg(), log=lambda m: None))
    event.state["saw_other"] = True
    event_enter = event.decide(
        _hall_obs(B(V.EVENT_LIVE, cx=0.30, cy=0.18, h=0.022), special=True),
        _hall_state("facility"))
    check("event facility 有大厅证据时转发 on_task_hall 进场",
          event_enter is not None and event_enter.is_tap
          and event_enter.target_cls == V.EVENT_LIVE,
          str(event_enter))
    event_no_hall = ALL["event"](Ctx(cfg=cfg(), log=lambda m: None))
    no_hall_action = event_no_hall.decide(
        O(B(V.EVENT_LIVE, cx=0.30, cy=0.18, h=0.022),
          B(V.BACK, cx=0.05, cy=0.05)),
        _hall_state("facility"))
    check("event facility 无大厅证据保持 no-op，不按返回制造乒乓",
          no_hall_action is None, str(no_hall_action))
    event_reward = ALL["event"](Ctx(cfg=cfg(), log=lambda m: None))
    event_reward.state["in_reward"] = True
    reward_action = event_reward.decide(
        _hall_obs(B(V.CLAIM_REWARD_YELLOW, cx=0.50, cy=0.70),
                  special=True),
        _hall_state("facility"))
    check("event facility 保留 in_reward 奖励分支",
          reward_action is not None and reward_action.is_tap
          and reward_action.target_cls == V.CLAIM_REWARD_YELLOW,
          str(reward_action))

    schedule = ALL["schedule"](Ctx(cfg=cfg(), log=lambda m: None))
    schedule_banner = schedule.on_reward(
        O(B(V.GOT_REWARD, conf=0.99, cx=0.50, cy=0.22)),
        _SV(page="schedule_region", overlay="reward"))
    check("课程表奖励层只有横幅时继承基类并 wait",
          schedule_banner is not None and schedule_banner.kind == "wait",
          str(schedule_banner))
    schedule_continue = schedule.on_reward(
        O(B(V.GOT_REWARD, conf=0.99, cx=0.50, cy=0.22),
          B(V.STORY_TAP_CONTINUE, conf=0.90, cx=0.50, cy=0.88)),
        _SV(page="schedule_region", overlay="reward"))
    check("课程表横幅和点击继续同屏时只点继续",
          schedule_continue is not None and schedule_continue.is_tap
          and schedule_continue.target_cls == V.STORY_TAP_CONTINUE,
          str(schedule_continue))

    a_rw, _ = _nv.route(
        O(B(V.GOT_REWARD, cx=0.50, cy=0.22), B(V.BACK, cx=0.05, cy=0.06)),
        _SV(page="mail", overlay="reward", frames_in_page=10), "lobby", None)
    check("归位不点获得奖励横幅(等)",
          a_rw is not None and a_rw.kind == "wait"
          and a_rw.target_cls != V.GOT_REWARD, str(a_rw))
    a_ct, _ = _nv.route(
        O(B(V.GOT_REWARD, cx=0.50, cy=0.22),
          B(V.STORY_TAP_CONTINUE, cx=0.50, cy=0.88)),
        _SV(page="mail", overlay="reward", frames_in_page=10), "lobby", None)
    check("归位奖励层点点击继续",
          a_ct is not None and a_ct.target_cls == V.STORY_TAP_CONTINUE, str(a_ct))

    def _cafe():
        return ALL["cafe"](Ctx(cfg=cfg(), log=lambda m: None))

    cf = _cafe()
    cf.state["floor"] = 1
    cf.observe(
        O(B(V.CAFE_MOVE_1F, cx=0.126, cy=0.143, w=0.08, h=0.05),
          B(V.ROOM_LOCKED, cx=0.103, cy=0.148, w=0.02, h=0.02)),
        _SV(page="cafe"))
    check("咖啡锁框与换厅钮重叠时不改 floor",
          cf.state.get("floor") == 1, str(cf.state.get("floor")))
    cf2 = _cafe()
    cf2.state["floor"] = 1
    cf2.observe(
        O(B(V.CAFE_MOVE_1F, cx=0.126, cy=0.143, w=0.08, h=0.05),
          B(V.ROOM_LOCKED, cx=0.190, cy=0.143, w=0.02, h=0.02)),
        _SV(page="cafe"))
    check("不重叠的锁框不屏蔽单一 to1",
          cf2.state.get("floor") == 2, str(cf2.state.get("floor")))
    cf3 = _cafe()
    cf3.state["floor"] = 2
    cf3.observe(O(B(V.CAFE_MOVE_2F, cx=0.126, cy=0.143, w=0.08, h=0.05)),
                _SV(page="cafe"))
    check("无锁的单一 to2 仍校正到 1F",
          cf3.state.get("floor") == 1, str(cf3.state.get("floor")))

    sample_paths = {
        "full_event_7994": (
            _ROOT / "data" / "raw_images" / "v2_20260815_021657"
            / "0007994_new_formation.jpg"),
        "full_event_8736": (
            _ROOT / "data" / "raw_images" / "v2_20260815_021657"
            / "0008736_new_formation.jpg"),
        "full_arena_11024": (
            _ROOT / "data" / "raw_images" / "v2_20260815_021657"
            / "0011024_new_formation.jpg"),
        "full_alt_0641": (
            _ROOT / "data" / "raw_images" / "v2_20260815_084829"
            / "0000641_formation.jpg"),
        "full_alt_0835": (
            _ROOT / "data" / "raw_images" / "v2_20260815_084829"
            / "0000835_formation.jpg"),
        "empty_40323": (
            _ROOT / "data" / "raw_images" / "v2_20260815_050854"
            / "0040323_formation.jpg"),
        "empty_7781": (
            _ROOT / "data" / "raw_images" / "v2_20260815_021657"
            / "0007781_new_formation.jpg"),
        "partial_0582": (
            _ROOT / "data" / "raw_images" / "v2_20260815_084829"
            / "0000582_formation.jpg"),
        "fade_8523": (
            _ROOT / "data" / "raw_images" / "v2_20260815_021657"
            / "0008523_new_formation.jpg"),
    }
    check("九张真实编队帧都在盘上",
          len(sample_paths) > 0 and all(p.is_file() for p in sample_paths.values()),
          ", ".join(str(p) for p in sample_paths.values() if not p.is_file()))
    images = {
        name: (_cv.imdecode(_np.fromfile(str(path), dtype=_np.uint8),
                            _cv.IMREAD_COLOR)
               if path.is_file() else None)
        for name, path in sample_paths.items()
    }
    check("九张真实编队帧都能解码",
          len(images) > 0 and all(frame is not None for frame in images.values()))

    must_pass = (
        "full_event_7994", "full_event_8736", "full_arena_11024",
        "full_alt_0641", "full_alt_0835",
    )
    must_block = ("empty_40323", "empty_7781", "partial_0582", "fade_8523")

    def _form_obs(frame, *, arena=False):
        boxes = [
            B(V.SORTIE, cx=0.92, cy=0.92),
            B(V.SQUAD_1_HI, cx=0.06, cy=0.26),
        ]
        if arena:
            boxes.append(B(V.SKIP_BATTLE, cx=0.91, cy=0.84))
        return Observation(
            boxes=boxes, frame=frame, seq=1,
            w=frame.shape[1], h=frame.shape[0])

    if all(frame is not None for frame in images.values()):
        for name in must_pass:
            metric = formation_slot_saturation(Observation(frame=images[name]))
            check(f"{name} 六人满编必须放行",
                  formation_ready(Observation(frame=images[name])),
                  str(metric))
            ready_flow = ALL["event"](Ctx(cfg=cfg(), log=lambda m: None))
            ready_act = ready_flow.decide(
                _form_obs(images[name]),
                _SV(page="formation", raw="formation", frames_in_page=10))
            check(f"{name} event 路径可生成出击 Action",
                  ready_act is not None and ready_act.is_tap
                  and ready_act.target_cls == V.SORTIE,
                  str(ready_act))
        for name in must_block:
            metric = formation_slot_saturation(Observation(frame=images[name]))
            check(f"{name} 有空槽或未就绪必须拦",
                  not formation_ready(Observation(frame=images[name])),
                  str(metric))

        import time as _time

        def _run_unready(frame, elapsed, ticks=40):
            flow = ALL["event"](Ctx(cfg=cfg(), log=lambda m: None))
            flow.state["formation_unready_since"] = _time.monotonic() - elapsed
            acts = []
            for _ in range(ticks):
                acts.append(flow.decide(
                    _form_obs(frame),
                    _SV(page="formation", raw="formation", frames_in_page=10)))
            sorties = [a for a in acts if a is not None
                       and a.is_tap and a.target_cls == V.SORTIE]
            return flow, acts[-1], sorties

        for name in ("empty_40323", "empty_7781", "partial_0582"):
            blocked_flow, blocked_last, blocked_sorties = _run_unready(
                images[name], 6.0)
            check(f"{name} 满 40 tick 且 5s 后 BLOCKED，出击 Action 为 0",
                  not blocked_sorties and blocked_flow.outcome == "BLOCKED"
                  and blocked_last.kind == "done",
                  f"{blocked_last} sorties={len(blocked_sorties)}")

        fade_flow, fade_last, fade_sorties = _run_unready(
            images["fade_8523"], 0.5)
        check("淡入帧即使满 40 tick，墙钟仅 0.5s 也只 wait 不终结",
              not fade_sorties and fade_flow.outcome is None
              and fade_last.kind == "wait",
              f"{fade_last} outcome={fade_flow.outcome}")

        reset_flow = ALL["event"](Ctx(cfg=cfg(), log=lambda m: None))
        reset_flow.state["formation_unready_since"] = _time.monotonic() - 6.0
        for _ in range(12):
            reset_flow.decide(
                _form_obs(images["empty_40323"]),
                _SV(page="formation", raw="formation", frames_in_page=10))
        reset_flow.decide(
            O(), _SV(page="lobby", raw="lobby",
                     frames_in_page=1, changed=True))
        check("编队页面变化会清空未就绪 hold 和墙钟",
              "formation_unready_since" not in reset_flow.state
              and "hold:formation_unready" not in reset_flow.state)

        recovered_flow = ALL["event"](Ctx(cfg=cfg(), log=lambda m: None))
        recovered_flow.state["formation_unready_since"] = _time.monotonic() - 6.0
        for _ in range(12):
            recovered_flow.decide(
                _form_obs(images["empty_40323"]),
                _SV(page="formation", raw="formation", frames_in_page=10))
        recovered_act = recovered_flow.decide(
            _form_obs(images["full_alt_0641"]),
            _SV(page="formation", raw="formation", frames_in_page=10))
        check("编队准备成功会清状态并立即允许出击",
              recovered_act is not None and recovered_act.is_tap
              and recovered_act.target_cls == V.SORTIE
              and "formation_unready_since" not in recovered_flow.state,
              str(recovered_act))

        arena_flow = ALL["arena"](Ctx(cfg=cfg(), log=lambda m: None))
        arena_actions = [
            arena_flow.decide(
                _form_obs(images["full_arena_11024"], arena=True),
                _SV(page="formation", raw="formation", frames_in_page=10))
            for _ in range(8)
        ]
        check("arena 0011024 满编通过像素闸并可出击",
              any(a is not None and a.is_tap and a.target_cls == V.SORTIE
                  for a in arena_actions)
              and arena_flow.outcome != "BLOCKED",
              str(arena_actions[-1]))

    unknown_flow = ALL["event"](Ctx(cfg=cfg(), log=lambda m: None))
    unknown_obs = O(B(V.SORTIE, cx=0.92, cy=0.92),
                    B(V.SQUAD_1_HI, cx=0.06, cy=0.26))
    import time as _time
    unknown_flow.state["formation_unready_since"] = _time.monotonic() - 6.0
    unknown_acts = [
        unknown_flow.decide(
            unknown_obs,
            _SV(page="formation", raw="formation", frames_in_page=10))
        for _ in range(40)
    ]
    check("无帧无法证明编队，满 40 tick 且 5s 后 BLOCKED 且不出击",
          not any(a is not None and a.is_tap and a.target_cls == V.SORTIE
                  for a in unknown_acts)
          and unknown_flow.outcome == "BLOCKED")

    from routing_v2.act.ledger import Ledger
    from routing_v2.config import data_dir, merged
    from routing_v2.flow import daybook as _daybook
    # 这块原来 json.loads(PROFILE.read_text()) 再断言 account.id == "alt" --
    #    违反本文件 L53 "测试不许读 profile.json"。08-26 用户把运行账号切回
    #    main, 四条用例红了一路, 红的是用户配置不是代码。要验的是**双账号
    #    机制**(覆盖合并/桶隔离/邀请禁用), 合成 profile 全量自证。
    main_targets = ["凯伊", "爱丽丝(战斗)", "爱丽丝"]
    user = {
        "account": {"id": "alt"},
        "accounts": {"main": {},
                     "alt": {"cafe": {"invite_targets": [],
                                        "skip_invite": True}}},
        "cafe": {"invite_targets": list(main_targets), "skip_invite": False},
    }
    alt_cfg = merged(user)
    profile_unlocked = copy.deepcopy(user)
    profile_unlocked.setdefault("safety", {})["money_step_needs_human"] = False
    check("profile 直接写 false 也掀不开 money_step_needs_human",
          merged(profile_unlocked)["safety"]["money_step_needs_human"] is True)
    main_user = copy.deepcopy(user)
    main_user.setdefault("account", {})["id"] = "main"
    main_cfg = merged(main_user)
    check("当前运行账号明确为 alt",
          alt_cfg["account"]["id"] == "alt")
    check("alt 邀请目标为空且明确禁用",
          alt_cfg["cafe"]["invite_targets"] == []
          and alt_cfg["cafe"]["skip_invite"] is True)
    check("main 邀请目标未被 alt 覆盖",
          main_cfg["cafe"]["invite_targets"] == main_targets
          and main_cfg["cafe"]["skip_invite"] is False)
    alt_dir = data_dir(alt_cfg)
    alt_ledger = Ledger(log=lambda m: None, out_dir=alt_dir)
    alt_sched = ALL["schedule"](Ctx(cfg=alt_cfg, log=lambda m: None))
    check("alt 的 ledger/daybook/schedule_rooms 都落 alt 桶",
          alt_dir.name == "alt"
          and alt_ledger.path.parent == alt_dir
          and _daybook._file(alt_cfg).parent == alt_dir
          and alt_sched._rooms_file().parent == alt_dir)
    alt_cafe = ALL["cafe"](Ctx(cfg=alt_cfg, log=lambda m: None))
    alt_cafe.goto("invite", "测试账号邀请配置")
    no_invite = alt_cafe.decide(O(), _SV(page="cafe", frames_in_page=10))
    check("alt 不开邀请名单也不盲滑",
          no_invite is not None and no_invite.kind == "wait"
          and alt_cafe.state.get("invite_disabled")
          and alt_cafe.state.get("invite_scrolls", 0) == 0)

    def _alt_exit(with_dot):
        flow = ALL["cafe"](Ctx(cfg=alt_cfg, log=lambda m: None))
        flow.state.update(claimed=True, invite_disabled=True)
        flow.goto("exit", "测试收尾语义")
        obs = O(B(V.DOT_YELLOW, cx=0.50, cy=0.50)) if with_dot else O()
        act = None
        for _ in range(20):
            act = flow.decide(obs, _SV(page="cafe", frames_in_page=10))
        return flow, act

    left_flow, left_act = _alt_exit(True)
    clean_flow, clean_act = _alt_exit(False)
    check("alt 禁邀但仍有黄点时诚实报 LEFTOVER",
          left_flow.outcome == "LEFTOVER" and left_act.kind == "done")
    check("alt 禁邀且无黄点时可报 CLEAN",
          clean_flow.outcome == "CLEAN" and clean_act.kind == "done")
    caps = alt_cfg["safety"]["purchase_caps"]
    check("钱相关默认仍为 0",
          alt_cfg["safety"]["ap_purchase_limit"] == 0
          and alt_cfg["shop"]["refresh_times"] == 0
          and alt_cfg["safety"]["money_step_needs_human"] is True
          and all(caps[k] == 0 for k in ("arena", "bounty",
                                         "scrimmage", "lesson")))


def t_shelf_walk():
    """大厅店走全部选择/选择购买; 活动店可见有 103 时不向上."""
    print("\n 大厅选择流 / 活动店有103不向上 ")
    import inspect as _insp
    from routing_v2.flow import facilities as _fac
    from routing_v2.percept.read import CREDIT as _CR

    check("大厅店源码不 import shelf_walk",
          "shelf_walk" not in _insp.getsource(_fac))

    def drive(flow, obs, fn):
        flow.ticks += 1
        a = fn(obs, Machine(1).update(obs))
        if a is not None and getattr(a, "post", None):
            a.post()
        return a

    # 大厅: 全部选择 -> 选择购买, 不滑货架, 不点 103 / 购买青辉石
    sh = ALL["shop"](Ctx(cfg=cfg(), log=lambda m: None))
    sh.cfg = dict(sh.cfg or {})
    sh.cfg["credit_buy"] = True
    grid = O(B(V.SHOP_TAB_CREDIT_SEL, cx=0.05, cy=0.19),
             B(V.SHOP_SELECT_ALL, cx=0.93, cy=0.12),
             B(V.SHOP_BUY_SELECTED, cx=0.91, cy=0.92),
             B(V.SHOP_BUY, cx=0.55, cy=0.48),
             B(V.SHOP_BUY, cx=0.90, cy=0.48),
             B(V.SHOP_BUY, cx=0.55, cy=0.83),
             B(V.SHOP_BUY, cx=0.90, cy=0.83),
             B(V.CREDIT, cx=0.55, cy=0.40))
    grid.balances[_CR] = 50000000
    a_s = drive(sh, grid, sh.on_shop)
    check("大厅店点全部选择",
          a_s is not None and getattr(a_s, "target_cls", "") == V.SHOP_SELECT_ALL,
          str(a_s))
    check("大厅店不走 shelf_walk/不点购买青辉石",
          a_s.kind != "swipe"
          and getattr(a_s, "target_cls", "") not in
          (V.SHOP_BUY, V.SHOP_BUY_PYROXENE),
          str(a_s))
    sh.state["once:selectall"] = True
    grid_checked = O(B(V.SHOP_TAB_CREDIT_SEL, cx=0.05, cy=0.19),
                     B(V.SHOP_SELECT_ALL, cx=0.93, cy=0.12),
                     B(V.GREEN_CHECK, cx=0.88, cy=0.20),
                     B(V.SHOP_BUY_SELECTED, cx=0.91, cy=0.92),
                     B(V.CREDIT, cx=0.55, cy=0.40))
    grid_checked.balances[_CR] = 50000000
    a_buy = drive(sh, grid_checked, sh.on_shop)
    check("绿勾在场不再点全部选择",
          a_buy is None or getattr(a_buy, "target_cls", "") != V.SHOP_SELECT_ALL,
          str(a_buy))
    check("大厅店全部选择后点选择购买",
          a_buy is not None and getattr(a_buy, "target_cls", "") == V.SHOP_BUY_SELECTED,
          str(a_buy))
    check("信用点选择购买不走人审",
          a_buy is not None and (not getattr(a_buy, "money", False)),
          str(a_buy))

    from routing_v2.state.machine import StateView as _SV
    sh_off = ALL["shop"](Ctx(cfg=cfg(), log=lambda m: None))
    a_off = drive(sh_off, grid, sh_off.on_shop)
    check("credit_buy 默认关不点全部选择",
          a_off is None or getattr(a_off, "target_cls", "") != V.SHOP_SELECT_ALL,
          str(a_off))
    check("credit_buy 默认关不点选择购买",
          a_off is None or getattr(a_off, "target_cls", "") != V.SHOP_BUY_SELECTED,
          str(a_off))
    check("credit_buy 关记 credit_off",
          sh_off.state.get("credit_off") is True
          and sh_off.state.get("bought") is True)
    sh_left = ALL["shop"](Ctx(cfg=cfg(), log=lambda m: None))
    sh_left.state.update(bought=True, credit_off=True, arena_done=True)
    a_left = sh_left.on_shop(
        grid_checked, _SV(page="shop", frames_in_page=50))
    check("credit_buy 关时货架选择购买不报 LEFTOVER",
          a_left is not None and a_left.kind == "done"
          and sh_left.outcome == "CLEAN"
          and "没买成" not in (a_left.reason or "")
          and "已关" in (a_left.reason or ""),
          str(a_left))

    idle = O(B(V.SHOP_TAB_CREDIT_SEL, cx=0.05, cy=0.19),
             B(V.SHOP_SELECT_ALL, cx=0.93, cy=0.12),
             B(V.SHOP_BUY_SELECTED, cx=0.91, cy=0.92),
             B(V.CREDIT, cx=0.55, cy=0.40))
    idle.balances[_CR] = 50000000
    sh_idle = ALL["shop"](Ctx(cfg=cfg(), log=lambda m: None))
    sh_idle.cfg = dict(sh_idle.cfg or {})
    sh_idle.cfg["credit_buy"] = True
    sh_idle.state["once:selectall"] = True
    a_idle = drive(sh_idle, idle, sh_idle.on_shop)
    check("空闲底栏 450 无勾选证据不当选择购买",
          a_idle is None or getattr(a_idle, "target_cls", "") != V.SHOP_BUY_SELECTED,
          str(a_idle))
    from routing_v2.flow.base import qty_max_ok as _qmax
    check("灰 MAX 在场不点 111",
          _qmax(O(B(V.QTY_MAX, conf=0.40, cx=0.80, cy=0.50),
                  B(V.QTY_MAX_GREY, conf=0.50, cx=0.80, cy=0.50))) is None)
    check("只有亮 MAX 放行",
          _qmax(O(B(V.QTY_MAX, conf=0.40, cx=0.80, cy=0.50))) is not None)
    check("TASK_INFO 在词表", V.TASK_INFO == "任务资讯")
    check("529/530 不建常量",
          not hasattr(V, "STAR_1") and not hasattr(V, "STAR_2"))

    lob = O(B(V.NAV_SHOP, cx=0.621, cy=0.953),
            B(V.SHOP_BUY_PYROXENE, cx=0.116, cy=0.360))
    a_ent = ALL["shop"](Ctx(cfg=cfg(), log=lambda m: None)).on_lobby(
        lob, Machine(1).update(lob))
    check("大厅入口点底栏商店不是购买青辉石",
          a_ent is not None and a_ent.target_cls == V.NAV_SHOP
          and a_ent.y > 0.88, str(a_ent))
    # 2026-08-24 起 56/61 已是废案, detect 出口不吐 -> 这条路线现实中永不触发。
    #    用例留着当**安全网的回归**: 哪天有人把类翻回来, 逻辑还得是"切回信用点"
    #    而不是点购买或退出。
    pyx = O(B(V.SHOP_TAB_PYROXENE_SEL, cx=0.05, cy=0.30, w=0.06, h=0.04),
            B(V.SHOP_BUY, cx=0.70, cy=0.50))
    a_tab = ALL["shop"](Ctx(cfg=cfg(), log=lambda m: None)).on_shop_pyroxene_tab(
        pyx, Machine(1).update(pyx))
    check("青辉石 tab 切回信用点, 不点购买/不退出",
          a_tab is not None and a_tab.kind == "tap"
          and a_tab.target_cls != V.SHOP_BUY
          and a_tab.y < 0.30, str(a_tab))
    grey_buy = O(B(V.SHOP_TAB_CREDIT_SEL, cx=0.05, cy=0.19),
                 B(V.GREEN_CHECK, cx=0.88, cy=0.20),
                 B(V.SHOP_BUY_SELECTED_GREY, cx=0.91, cy=0.92),
                 B(V.CREDIT, cx=0.70, cy=0.55))
    sh3 = ALL["shop"](Ctx(cfg=cfg(), log=lambda m: None))
    sh3.cfg = dict(sh3.cfg or {})
    sh3.cfg["credit_buy"] = True
    sh3.state["once:selectall"] = True
    a_nb = None
    for _ in range(10):
        a_nb = drive(sh3, grey_buy, sh3.on_shop)
    check("大厅店选择购买灰色不当成交",
          sh3.state.get("credit_short") is True
          and getattr(a_nb, "target_cls", "") != V.SHOP_BUY_SELECTED,
          str((sh3.state.get("credit_short"), a_nb)))

    # 活动店: 可见有 103 时不向上. 即使刚滑完/同画面也不回滑.
    ev = ALL["event_shop"](Ctx(cfg=cfg(), log=lambda m: None))
    ev.state["plan"] = {}
    top = O(B(V.CURRENCY_SEL, cx=0.08, cy=0.30, w=0.06, h=0.04),
            B(V.SHOP_BUY, cx=0.40, cy=0.36),
            B(V.SHOP_BUY, cx=0.80, cy=0.36),
            B(V.SHOP_BUY, cx=0.40, cy=0.72),
            B(V.SHOP_BUY, cx=0.80, cy=0.72))
    ups = []
    last = None
    for _ in range(8):
        last = drive(ev, top, ev.on_event_shop)
        if last is not None and last.kind == "swipe" and last.y < last.y2:
            ups.append(last)
    check("活动店可见有 103 时不向上",
          not ups, str(ups[0] if ups else last))
    check("活动店有 103 不因同画面回滑",
          last is None or last.kind != "swipe" or last.y > last.y2,
          str(last))
    ev_fake = ALL["event_shop"](Ctx(cfg=cfg(), log=lambda m: None))
    ev_fake.state.update(plan={}, shelf_went_down=True, shelf_moved=False)
    a_fake = drive(ev_fake, top, ev_fake.on_event_shop)
    check("活动店有 103 即使标记已探底也不向上",
          a_fake is None or a_fake.kind != "swipe" or a_fake.y > a_fake.y2,
          str(a_fake))

    sold = O(B(V.CURRENCY_SEL, cx=0.08, cy=0.30, w=0.06, h=0.04),
             B(V.SHOP_BUY_GREY, cx=0.40, cy=0.36),
             B(V.SHOP_BUY_GREY, cx=0.80, cy=0.36),
             B(V.SHOP_BUY_GREY, cx=0.40, cy=0.72),
             B(V.SHOP_BUY_GREY, cx=0.80, cy=0.72))
    ev2 = ALL["event_shop"](Ctx(cfg=cfg(), log=lambda m: None))
    ev2.state["plan"] = {}
    a_down = drive(ev2, sold, ev2.on_event_shop)
    check("活动店本屏无 103 才下滑",
          a_down is not None and a_down.kind == "swipe" and a_down.y > a_down.y2,
          str(a_down))

    _yolo_shop_fixture()


def _yolo_shop_fixture():
    """YOLO 标注帧: 大厅走选择流, 活动店有 103 不向上. 分母先印且不为 0."""
    root = Path(__file__).resolve().parents[2]
    cls_path = root / "data" / "raw_images" / "_classes.txt"
    names = cls_path.read_text(encoding="utf-8").splitlines()
    frames = [
        ("credit", root / "data" / "raw_images" / "walk_20260813_083604" / "007_shop.txt"),
        ("event", root / "data" / "raw_images" / "v2_20260813_134818" / "0006580_MONEY.txt"),
    ]
    exist = [(tag, p) for tag, p in frames if p.exists()]
    print(f"YOLO货架夹具分母 {len(exist)}/{len(frames)}")
    check("YOLO货架夹具分母不为 0", len(exist) > 0, str(len(exist)))
    for tag, p in exist:
        boxes = []
        for line in p.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cid = int(float(parts[0]))
            if cid < 0 or cid >= len(names):
                continue
            name = names[cid].strip()
            cx, cy, w, h = map(float, parts[1:5])
            boxes.append(B(name, conf=0.9, cx=cx, cy=cy, w=w, h=h))
        obs = O(*boxes)
        n103 = sum(1 for b in boxes if b.cls == V.SHOP_BUY)
        n489 = sum(1 for b in boxes if b.cls == V.SHOP_BUY_GREY)
        print(f"  {tag} {p.name}: n103={n103} n489={n489}")
        if tag == "credit":
            sh = ALL["shop"](Ctx(cfg=cfg(), log=lambda m: None))
            sh.cfg = dict(sh.cfg or {})
            sh.cfg["credit_buy"] = True
            sh.cfg["arena_shop"] = False
            shelf_sw = []
            taps103 = []
            last = None
            for i in range(12):
                sh.ticks += 1
                last = sh.on_shop(obs, Machine(1).update(obs))
                if last is not None and getattr(last, "post", None):
                    last.post()
                if last is not None and last.kind == "swipe":
                    why = last.reason or ""
                    if "货架" in why or "探底" in why:
                        shelf_sw.append(last)
                if last is not None and getattr(last, "target_cls", "") == V.SHOP_BUY:
                    taps103.append(last)
            print(f"    大厅007 last={last}")
            check("大厅007不走 shelf_walk(不滑货架)",
                  not shelf_sw, str(shelf_sw[0] if shelf_sw else last))
            check("大厅007不点单卡购买103",
                  not taps103, str(taps103[0] if taps103 else last))
            sel_ok = (sh.state.get("credit_short")
                      or sh.state.get("bought")
                      or (last is not None and last.kind == "wait")
                      or getattr(last, "target_cls", "") in (
                          V.SHOP_SELECT_ALL, V.SHOP_BUY_SELECTED,
                          V.SHOP_BUY_SELECTED_GREY, V.SHOP_SELECT_ALL_GREY))
            check("大厅007走全部选择/选择购买系列",
                  bool(sel_ok) and last is not None and last.kind != "swipe",
                  str((sh.state.get("credit_short"), last)))
        if tag == "event":
            es = ALL["event_shop"](Ctx(cfg=cfg(), log=lambda m: None))
            es.setup()
            ups = []
            last = None
            for _ in range(8):
                es.ticks += 1
                last = es._scan_shelf(obs, Machine(1).update(obs),
                                      "tab1/1", 0, 0, 1)
                if last is not None and getattr(last, "post", None):
                    last.post()
                if last is not None and last.kind == "swipe" and last.y < last.y2:
                    ups.append(last)
            print(f"    活动6580 n103={n103} last={last}")
            check("活动店6580可见有103时不向上",
                  n103 == 0 or not ups, str(ups[0] if ups else last))


def t_remain_gates_0816():
    """08-16 remain: 社团浮层不秒收 / AP 909->9 / 大赛续打 / 活动店确认不滑走."""
    print("\n-- remain 五条门控 ----")
    from routing_v2.flow import nav as _nv
    from routing_v2.state.machine import StateView as _SV

    club = ALL["club"](Ctx(cfg=cfg(), log=lambda m: None))
    overlay = O(B(V.CLUB, cx=0.22, cy=0.46))
    st_c = Machine(1).update(overlay)
    a = club.on_club(overlay, st_c)
    check("社团浮层点卡不置 entered",
          a is not None and a.target_cls == V.CLUB
          and not club.state.get("inside"))
    club.state["once:enter_card"] = True
    for _ in range(15):
        club.ticks += 1
        a3 = club.on_club(overlay, st_c)
    check("浮层 15 帧不秒收 CLEAN",
          getattr(club, "outcome", None) in (None, ""),
          str(getattr(club, "outcome", None)))
    check("点卡后浮层仍在是 wait",
          a3 is not None and a3.kind == "wait")

    ev = ALL["event"](Ctx(cfg=cfg(), log=lambda m: None))
    ev.state["ap_seen"] = 909
    import routing_v2.percept.read as _RD
    _orig = _RD.read_topbar
    try:
        _RD.read_topbar = lambda o, c: 9
        check("AP 909 截成 9 不采信", ev._ap_read(O(B(V.AP))) is None)
        ev.state["ap_seen"] = 29
        check("AP 29 花到 9 采信", ev._ap_read(O(B(V.AP))) == 9)
    finally:
        _RD.read_topbar = _orig

    ar = ALL["arena"](Ctx(cfg=cfg(), log=lambda m: None))
    ar.state.update(tickets0=5, tickets=2, fights=4, inside=True,
                    entry_claims=3)
    row = O(B(V.ARENA_ROW, cx=0.70, cy=0.30, w=0.30, h=0.08))
    st_a = Machine(1).update(row)
    st_a.page = "arena"
    st_a.frames_in_page = 60
    aa = ar.on_arena(row, st_a)
    check("大赛进门 5 打了 4 票读不出仍出战",
          aa is not None and aa.kind != "done", str(aa))

    a, _ = _nv.route(
        O(B(V.HOME, cx=0.12, cy=0.06), B(V.BACK, cx=0.05, cy=0.06),
          B(V.CANCEL, cx=0.40, cy=0.70)),
        _SV(page="formation", frames_in_page=10), "lobby", None)
    check("编队归位大厅优先回大厅键",
          a is not None and a.target_cls == V.HOME,
          str(getattr(a, "target_cls", None)))

    es = ALL["event_shop"](Ctx(cfg=cfg(), log=lambda m: None))
    es.setup()
    o = O(B(V.SHOP_BUY, cx=0.50, cy=0.50),
          B(V.SHOP_BUY_SELECTED, cx=0.80, cy=0.90),
          B(V.CURRENCY_SEL, cx=0.08, cy=0.30))
    asel = es._scan_shelf(o, Machine(1).update(o), "tab1/2", 0, 0, 2)
    check("活动店选择购买在场先点不滑走",
          asel is not None and asel.target_cls == V.SHOP_BUY_SELECTED,
          str(asel))

    lobby = O(B(V.NAV_DAILY_REWARD, cx=0.90, cy=0.06, w=0.04, h=0.04),
              B(V.NAV_CAFE, cx=0.20, cy=0.95), B(V.NAV_SHOP, cx=0.40, cy=0.95))
    dm = ALL["daily_mission"](Ctx(cfg=cfg(), log=lambda m: None))
    ad = dm.on_lobby(lobby, Machine(1).update(lobby))
    check("大厅每日领奖还在就进任务页",
          ad is not None and ad.target_cls == V.NAV_DAILY_REWARD, str(ad))


def t_v18_gates_0820():
    """08-20 国际服新皮门控: 任务厅身份 / 信用点0.70 / 双扫荡开始 / 藏UI。"""
    print("\n-- v18 0820 门控 ----")
    from routing_v2.flow import nav as _nv
    from routing_v2.state.machine import StateView as _SV

    camp_back = O(B(V.HUB_CAMPAIGN, cx=0.40, cy=0.45),
                  B(V.BACK, cx=0.05, cy=0.05))
    check("evidence 推图+返回", _nv.task_hall_evidence(camp_back))
    check("evidence 只有推图不够",
          not _nv.task_hall_evidence(O(B(V.HUB_CAMPAIGN, cx=0.40, cy=0.45))))
    check("evidence 剧情+特殊无推图不够",
          not _nv.task_hall_evidence(
              O(B(V.HUB_STORY, cx=0.20, cy=0.45),
                B(V.HUB_SPECIAL, cx=0.60, cy=0.45),
                B(V.BACK, cx=0.05, cy=0.05))))
    lobby_plus = O(B(V.NAV_CAFE, cx=0.07, cy=0.95),
                   B(V.NAV_SHOP, cx=0.62, cy=0.95),
                   B(V.NAV_CRAFT, cx=0.53, cy=0.95),
                   B(V.HUB_CAMPAIGN, cx=0.40, cy=0.45),
                   B(V.BACK, cx=0.05, cy=0.05))
    check("底栏NAV>=3 不当 task_hall", classify(lobby_plus).page == "lobby")

    sh = ALL["shop"](Ctx(cfg=cfg(), log=lambda m: None))
    gem = O(B(V.ARENA_SHOP_TAB_SEL, conf=0.98, cx=0.06, cy=0.30),
            B(V.ARENA_SHOP_CURRENCY, conf=0.97, cx=0.03, cy=0.30),
            B(V.SHOP_TAB_CREDIT_SEL, conf=0.47, cx=0.05, cy=0.19))
    check("青辉石左栏大赛已选 + 信用点0.47 不当信用点货架",
          not sh._on_credit_shelf(gem))
    cred = O(B(V.SHOP_TAB_CREDIT_SEL, conf=0.975, cx=0.05, cy=0.19),
             B(V.SHOP_SELECT_ALL, cx=0.93, cy=0.12),
             B(V.CREDIT, cx=0.55, cy=0.40))
    check("信用点已选 0.975 认货架", sh._on_credit_shelf(cred))
    cred_noise = O(B(V.ARENA_SHOP_TAB_SEL, conf=0.98, cx=0.06, cy=0.30),
                   B(V.SHOP_TAB_CREDIT_SEL, conf=0.975, cx=0.05, cy=0.19),
                   B(V.SHOP_SELECT_ALL, cx=0.93, cy=0.12),
                   B(V.CREDIT, cx=0.55, cy=0.40))
    check("左栏大赛已选不挡真信用点货架", sh._on_credit_shelf(cred_noise))
    sh.state["bought"] = True
    a_tab = sh._goto_arena_tab(
        O(B(V.ARENA_SHOP_CURRENCY, conf=0.97, cx=0.03, cy=0.30),
          B(V.SHOP_TAB_CREDIT_SEL, cx=0.05, cy=0.19)))
    check("左栏大赛币 cy0.30 不当大赛tab",
          a_tab is None or not (getattr(a_tab, "is_tap", False)
                                and abs(getattr(a_tab, "y", 0) - 0.30) < 0.05),
          str(a_tab))

    dual = O(B(V.SWEEP_START, conf=0.99, cx=0.73, cy=0.60),
             B(V.SWEEP_START, conf=0.96, cx=0.73, cy=0.79))
    go = _nv.real_sweep_start(dual, 0.35)
    check("双扫荡开始只取 cy 小的",
          go is not None and abs(go.cy - 0.60) < 0.02)
    check("cy>0.72 扫荡开始当任务开始禁点",
          _nv.real_sweep_start(
              O(B(V.SWEEP_START, conf=0.96, cx=0.73, cy=0.79)), 0.35) is None)

    flow = ALL["mail"](Ctx(cfg=cfg(), log=lambda m: None))
    st_blank = _SV(page="blank", raw="blank", overlay=None,
                   frames_in_page=2, last_solid="lobby")
    w = _nv.wake_hidden_lobby(O(), st_blank, flow)
    check("大厅藏UI 唤醒",
          w is not None and w.kind == "tap"
          and abs(w.x - 0.40) < 0.01 and abs(w.y - 0.55) < 0.01)
    st_bat = _SV(page="blank", raw="blank", overlay=None,
                 frames_in_page=2, last_solid="battle")
    check("战斗后 blank 不唤醒",
          _nv.wake_hidden_lobby(O(), st_bat, flow) is None)
    # `nav.task_hall_blind` 08-21 起是**死开关**(盲点已删, 无人读),
    #   留着只为不动 profile.json。默认仍是关。
    check("盲点默认关", cfg().get("nav", {}).get("task_hall_blind") is False)

    sm_door = ALL["story_mining"](Ctx(cfg=cfg(), log=lambda m: None))
    sm_door.setup()
    o_door = O(B(V.NAV_CAFE, cx=0.07, cy=0.95),
               B(V.NAV_SHOP, cx=0.62, cy=0.95),
               B(V.NAV_CRAFT, cx=0.53, cy=0.95))
    a_door = _nv.lobby_enter(sm_door, o_door, V.NAV_TASKS, "任务大厅",
                             expect=(V.HUB_CAMPAIGN,))
    # 08-21 契约改: 猜控件位置的盲坐标已删。没帧(离线 fixture)= 低阈通道
    #   开不了 -> **如实返回 None**(fail-closed), 而不是往右下角盲点一发。
    check("大厅 NAV>=3 但入口检不出 -> 不动(不再盲点)",
          a_door is None, str(a_door))
    sm_few = ALL["story_mining"](Ctx(cfg=cfg(), log=lambda m: None))
    sm_few.setup()
    a_few = _nv.lobby_enter(
        sm_few, O(B(V.NAV_CAFE, cx=0.07, cy=0.95)), V.NAV_TASKS, "任务大厅")
    check("大厅 NAV<3 更不动", a_few is None, str(a_few))
    a_rt, _pl = _nv.route(o_door, _SV(page="lobby", frames_in_page=3),
                          "task_hall")
    check("归位大厅进厅同样不盲点",
          a_rt is None or a_rt.kind != "tap", str(a_rt))
    # 入口过 0.45 时照常点**检出框**(落点来自 cls, 不是坐标)
    sm_ok = ALL["story_mining"](Ctx(cfg=cfg(), log=lambda m: None))
    sm_ok.setup()
    o_ok = O(B(V.NAV_CAFE, cx=0.07, cy=0.95),
             B(V.NAV_SHOP, cx=0.62, cy=0.95),
             B(V.NAV_CRAFT, cx=0.53, cy=0.95),
             B(V.NAV_TASKS, conf=0.88, cx=0.9393, cy=0.9428, w=0.1017, h=0.0553))
    a_ok = _nv.lobby_enter(sm_ok, o_ok, V.NAV_TASKS, "任务大厅",
                           expect=(V.HUB_CAMPAIGN,))
    check("入口检得出 -> 点检出框",
          a_ok is not None and a_ok.kind == "tap"
          and a_ok.target_cls == V.NAV_TASKS
          and abs(a_ok.x - 0.9393) < 0.01, str(a_ok))
    # 低阈通道的门: 没确认在大厅就不许开
    check("低阈通道要先确认在大厅",
          _nv.weak_task_entry(O(B(V.NAV_CAFE, cx=0.07, cy=0.95))) is None)

    from routing_v2.flow.interrupt import Interrupts
    o_nx = O(B(V.STORY_QUIT, cx=0.39, cy=0.72),
             B(V.STORY_WATCH, cx=0.61, cy=0.72))
    it = Interrupts(log=lambda m: None)
    it.watch_next_chapter = True
    a_w = it.handle("story_cutscene", o_nx)
    check("watch_next 点观看",
          a_w is not None and a_w.target_cls == V.STORY_WATCH, str(a_w))
    it2 = Interrupts(log=lambda m: None)
    a_q = it2.handle("story_cutscene", o_nx)
    check("默认过场点中断",
          a_q is not None and a_q.target_cls == V.STORY_QUIT, str(a_q))
    check("story_mining 声明 watch_next",
          bool(getattr(ALL["story_mining"], "watch_next_chapter", False)))


def t_story_mining_0820():
    """08-20 剧情：章节图有 new 就点；全完成滑一次再扫；节点无入场键回图不收工。"""
    print("\n-- story_mining 0820 滑一次扫一次 ----")
    from routing_v2.state.machine import StateView as _SV

    ctx = Ctx(cfg=cfg(), log=lambda m: None)
    st_map = _SV(page="story_chapter_map", frames_in_page=3)
    st_nodes = _SV(page="story_nodes", frames_in_page=3)

    sm = ALL["story_mining"](ctx)
    sm.setup()
    o_new = O(B(V.NEW_MARK, conf=0.96, cx=0.588, cy=0.547),
              B(V.NODE_DONE, conf=0.97, cx=0.139, cy=0.546),
              B(V.NODE_DONE, conf=0.96, cx=0.418, cy=0.250),
              B(V.BACK, conf=0.98, cx=0.045, cy=0.051))
    a = sm.on_story_chapter_map(o_new, st_map)
    check("159 有 new 就点 new",
          a is not None and a.kind == "tap" and a.target_cls == V.NEW_MARK,
          str(a))

    sm2 = ALL["story_mining"](ctx)
    sm2.setup()
    o_done = O(B(V.NODE_DONE, conf=0.97, cx=0.139, cy=0.546),
               B(V.NODE_DONE, conf=0.96, cx=0.418, cy=0.250),
               B(V.BACK, conf=0.98, cx=0.045, cy=0.051))
    a2 = sm2.on_story_chapter_map(o_done, st_map)
    check("章节图全完成滑一次找 new",
          a2 is not None and a2.kind == "swipe" and a2.x > a2.x2, str(a2))
    check("滑一次不换源", sm2.state.get("src_i") == 0)

    sm3 = ALL["story_mining"](ctx)
    sm3.setup()
    o156 = O(B(V.STORY_NODE_UNDONE, conf=0.97, cx=0.57, cy=0.35),
             B(V.STORY_NODE_DONE, conf=0.98, cx=0.52, cy=0.50),
             B(V.BACK, conf=0.98, cx=0.045, cy=0.051))
    a3 = sm3.on_story_nodes(o156, st_nodes)
    check("156 无入场键回章节图不整条收工",
          a3 is not None and a3.kind != "done" and a3.target_cls == V.BACK,
          str(a3))
    check("156 不 src_i++", sm3.state.get("src_i") == 0)

    smt = ALL["story_mining"](ctx)
    smt.setup()
    o_yd = O(B(V.HUB_CAMPAIGN, cx=0.40, cy=0.45),
             B(V.BACK, cx=0.05, cy=0.05),
             B(V.DOT_YELLOW, cx=0.831, cy=0.303))
    a_yd = smt.on_task_hall(o_yd, _SV(page="task_hall", frames_in_page=3))
    check("剧情 tile 0 点卡上黄点",
          a_yd is not None and a_yd.target_cls == V.DOT_YELLOW
          and abs(a_yd.x - 0.871) < 0.02, str(a_yd))

    # 08-21 契约改: 旧的"主线四字点立绘 / 优先点卡上黄点往左下"是在不知道
    #   卡片是**堆叠**的前提下拟合出来的落点, 已被用户口述 + live 实测推翻
    #   (点后卡只翻面不进入)。完整机制断言见 t_story_stack_0821, 这里只留
    #   单卡形态最小检查: 落点必须挂在剧情卡框上, 且推到卡身而不是标签行。
    smm = ALL["story_mining"](ctx)
    smm.setup()
    a_m = smm.on_story_hub(
        O(B(V.STORY_MAIN, cx=0.441, cy=0.721, w=0.124, h=0.050)),
        _SV(page="story_hub", frames_in_page=3))
    check("主线单卡: 落点挂在剧情卡框上并推到卡身",
          a_m is not None and a_m.target_cls == V.STORY_MAIN
          and abs(a_m.x - 0.441) < 0.02 and a_m.y < 0.60,
          str(a_m))
    # 卡顶有黄点但只有一张卡 -> 没有后卡可翻, 照样直接进
    smy = ALL["story_mining"](ctx)
    smy.setup()
    a_ydot = smy.on_story_hub(
        O(B(V.STORY_MAIN, cx=0.441, cy=0.721, w=0.124, h=0.050),
          B(V.DOT_YELLOW, cx=0.410, cy=0.165, w=0.010, h=0.018)),
        _SV(page="story_hub", frames_in_page=3))
    check("主线单卡带黄点: 没后卡可翻, 直接进(不再往左下推)",
          a_ydot is not None and a_ydot.target_cls == V.STORY_MAIN
          and abs(a_ydot.x - 0.441) < 0.02 and a_ydot.y < 0.60,
          str(a_ydot))
    smf = ALL["story_mining"](ctx)
    smf.setup()
    a_fac = smf.on_facility(
        O(B(V.ARROW_RIGHT, cx=0.92, cy=0.50),
          B(V.BACK, cx=0.045, cy=0.05)),
        _SV(page="facility", frames_in_page=3))
    check("支线墙无黄点先翻页",
          a_fac is not None and a_fac.target_cls == V.ARROW_RIGHT, str(a_fac))



def t_schedule_locked_card_0821():
    """08-21: 全體課程表面板上锁着的房间卡, 卡上的灰头像不许挑。

    fixture 全部来自小号 Lv30 真采帧 `v19_117_alt_roster2`:
      面板是 3 列 x 3 行房间卡; 锁 `房间区域未解锁` conf 0.977-0.981, 落在
      cx .2339/.5028/.7718 三列, cy .3310/.5411/.7512 三行。
      視聽室(列1行1)/體育館(列2行1) 开着; 其余五张锁着(需要RANK 3/5/6/8/9)。
    """
    print("\n-- schedule locked card 0821 ----")
    from routing_v2.state.machine import StateView as _SV

    ctx = Ctx(cfg=cfg(), log=lambda m: None)
    st = _SV(page="schedule_region", frames_in_page=5)

    def lock(cx, cy):
        return B(V.ROOM_LOCKED, conf=0.98, cx=cx, cy=cy, w=0.0217, h=0.0327)

    def stud(name, cx, cy):
        # Box 是 frozen dataclass, model 只能在构造时给
        return Box(cls=name, conf=0.90, x1=cx - 0.0225, y1=cy - 0.040,
                   x2=cx + 0.0225, y2=cy + 0.040, model="avatar")

    panel = [B(V.SCHED_TICKET, conf=0.97, cx=0.4521, cy=0.2086),
             B(V.CLOSE_X, conf=0.97, cx=0.8886, cy=0.1401)]
    locks = [lock(0.7718, 0.3310),            # 圖書館 需要RANK 3
             lock(0.2339, 0.5411), lock(0.5028, 0.5412), lock(0.7717, 0.5411),
             lock(0.2339, 0.7512)]
    free = stud("白子", 0.1400, 0.3980)        # 視聽室(没锁)
    locked = stud("小春", 0.6800, 0.3960)      # 圖書館(锁着), 在锁下方同一列

    sc = ALL["schedule"](ctx)
    sc.setup()
    sc.goto("roster", "test")
    check("锁在头像上方同一张卡 -> 判成锁卡",
          sc._on_locked_card(O(*panel, *locks), locked))
    check("没锁的那张卡不判成锁卡",
          not sc._on_locked_card(O(*panel, *locks), free))
    # 锁**下方**才算同卡: 头像跑到锁上面去就不算(防跨行误伤)
    above = stud("小春", 0.6800, 0.3000)
    check("头像在锁上方不算同卡",
          not sc._on_locked_card(O(*panel, *locks), above))
    # 横向差超过卡宽也不算(防蹭隔壁列)
    far = stud("小春", 0.4000, 0.3960)
    check("横向超出卡宽不算同卡",
          not sc._on_locked_card(O(*panel, *locks), far))

    a = sc.do_roster(O(*panel, *locks, free, locked), st)
    check("roster 只挑没锁那张卡上的人",
          a is not None and a.kind == "tap" and abs(a.x - 0.1400) < 0.02,
          detail=str(a))
    # 只剩锁卡上的人 -> 不许点, 该换区域
    sc2 = ALL["schedule"](ctx)
    sc2.setup()
    sc2.goto("roster", "test")
    a2 = sc2.do_roster(O(*panel, *locks, locked), st)
    check("面板上只剩锁卡的人 -> 不点, 去关面板/换区域",
          a2 is None or a2.target_cls != "小春", detail=str(a2))


def t_story_stack_0821():
    """08-21: 剧情卡堆叠 —— 点后卡只翻面, 点前卡才进。

    fixture 数字全部来自当天真采的四件套 `flywheel_v19_ui_20260821`:
      v19_048 (第2部在前): 前卡 cx .3274 w .1241 / 后卡 cx .4400 w .0620 / 黄点 cx .4102
      v19_047 (第1部在前): 前卡 cx .3211 w .1354 / 后卡 cx .4398 w .0625 / 黄点 cx .4899
    """
    print("\n-- story stack 0821 ----")
    from routing_v2.state.machine import StateView as _SV
    from routing_v2.flow import nav as _nav

    ctx = Ctx(cfg=cfg(), log=lambda m: None)
    st_hub = _SV(page="story_hub", frames_in_page=3)

    def card(cx, w, conf):
        return B(V.STORY_MAIN, conf=conf, cx=cx, cy=0.7240, w=w, h=0.0500)

    # 隔壁短篇/支线卡上的黄点(cx .7103) 每帧都在, 不许被算进主线那一摞
    other = [B(V.STORY_SHORT, conf=0.996, cx=0.6566, cy=0.5352, w=0.0786, h=0.0367),
             B(V.DOT_YELLOW, conf=0.941, cx=0.7103, cy=0.1666, w=0.0100, h=0.0175)]

    # A 带黄点那部在**前** (v19_048) -> 直接进, 不许翻面
    o_front = O(*other, card(0.3274, 0.1241, 0.407), card(0.4400, 0.0620, 0.924),
                B(V.DOT_YELLOW, conf=0.941, cx=0.4102, cy=0.1654, w=0.0100, h=0.0175))
    f, backs = _nav.story_cards(o_front, V.STORY_MAIN)
    check("前卡=最宽那个(不是 conf 最高那个)",
          f is not None and abs(f.cx - 0.3274) < 0.01 and len(backs) == 1,
          detail=str((f and round(f.cx, 4), len(backs))))
    d = _nav.story_stack_dot(o_front, f)
    check("卡顶黄点只取主线那一摞的(隔壁 .7103 不算)",
          d is not None and abs(d.cx - 0.4102) < 0.01, detail=str(d and round(d.cx, 4)))
    check("黄点在前卡上 -> story_dot_on_front True",
          _nav.story_dot_on_front(f, d))

    m = ALL["story_mining"](ctx)
    m.setup()
    a = m.on_story_hub(o_front, st_hub)
    check("带黄点那部在前 -> 点前卡进入",
          a is not None and a.kind == "tap" and abs(a.x - 0.3274) < 0.04
          and a.y < 0.60, detail=str(a))

    # B 带黄点那部在**后** (v19_047) -> 先点后卡翻面, 不许直接进
    o_back = O(*other, card(0.3211, 0.1354, 0.344), card(0.4398, 0.0625, 0.964),
               B(V.DOT_YELLOW, conf=0.942, cx=0.4899, cy=0.1911, w=0.0093, h=0.0175))
    f2, backs2 = _nav.story_cards(o_back, V.STORY_MAIN)
    d2 = _nav.story_stack_dot(o_back, f2)
    check("黄点在后卡 -> story_dot_on_front False",
          not _nav.story_dot_on_front(f2, d2),
          detail=str((f2 and round(f2.x2, 4), d2 and round(d2.cx, 4))))

    m2 = ALL["story_mining"](ctx)
    m2.setup()
    b1 = m2.on_story_hub(o_back, st_hub)
    check("带黄点那部在后 -> 点后卡翻面(不是点前卡进入)",
          b1 is not None and b1.kind == "tap" and abs(b1.x - 0.4398) < 0.04,
          detail=str(b1))
    m2.state["once:rot0_0"] = True          # 模拟 runner 在 tap 落地后写 once
    b2 = m2.on_story_hub(o_back, st_hub)
    check("翻面那一发不重复点", b2 is None or b2.kind != "tap", detail=str(b2))
    # 翻完之后画面变成 A 形态 -> 该进了
    m2.state["rot"] = 1
    b3 = m2.on_story_hub(o_front, st_hub)
    check("翻面后黄点到前卡 -> 改成点前卡进入",
          b3 is not None and b3.kind == "tap" and abs(b3.x - 0.3274) < 0.04,
          detail=str(b3))

    # C 单卡(短篇/支线): 没有后卡, 直接进, 不许被隔壁黄点带跑
    o_single = O(B(V.STORY_SHORT, conf=0.996, cx=0.6566, cy=0.5352, w=0.0786, h=0.0367),
                 B(V.DOT_YELLOW, conf=0.941, cx=0.7103, cy=0.1666, w=0.0100, h=0.0175))
    f3, backs3 = _nav.story_cards(o_single, V.STORY_SHORT)
    check("单卡没有后卡", f3 is not None and backs3 == [], detail=str(backs3))
    # 单独一个 ctx: cfg 是 ctx 上的共享 dict, 在这儿改 sources 会漏给下面的用例
    ctx_single = Ctx(cfg=cfg(), log=lambda m: None)
    m3 = ALL["story_mining"](ctx_single)
    m3.setup()
    m3.cfg["sources"] = ["短篇剧情"]
    c1 = m3.on_story_hub(o_single, st_hub)
    check("单卡直接进(不翻面)",
          c1 is not None and c1.kind == "tap" and abs(c1.x - 0.6566) < 0.04,
          detail=str(c1))

    # D 死循环闸: 一直翻不到 -> 换下一类, 不许无限点
    m4 = ALL["story_mining"](ctx)
    m4.setup()
    m4.state["rot"] = 9
    d1 = m4.on_story_hub(o_back, st_hub)
    check("翻够次数仍不对 -> 换下一类而不是接着点",
          (d1 is None or d1.kind != "tap") and m4.state["src_i"] == 1,
          detail=str((d1, m4.state["src_i"])))


def t_daily_fix_0820():
    """08-20 档B: tile 0 点卡上点 / HALT 连坐打标 / event AP OCR 文案."""
    print("\n-- daily fix 0820 ----")
    from routing_v2.state.machine import StateView as _SV
    from routing_v2.flow import nav as _nav

    ctx = Ctx(cfg=cfg(), log=lambda m: None)
    st_hall = _SV(page="task_hall", frames_in_page=3)

    # 任务大厅证据(推图 + 返回, 底栏 NAV 簇不在) —— 与 nav.task_hall_evidence 一致
    hall = [B(V.HUB_CAMPAIGN, conf=0.92, cx=0.62, cy=0.31),
            B(V.BACK, cx=0.045, cy=0.052)]

    # 1) 悬赏: tile cls 0 + 卡上黄点 -> 点黄点(检出框), 不是盲坐标
    bo = ALL["bounty"](ctx)
    bo.setup()
    o_dot = O(*hall, B(V.DOT_YELLOW, conf=0.88, cx=0.61, cy=0.545))
    a = bo.on_task_hall(o_dot, st_hall)
    check("bounty tile0 + 卡上黄点 -> tap 黄点",
          a is not None and a.kind == "tap"
          and abs(a.x - 0.61) < 0.05 and abs(a.y - 0.545) < 0.05,
          detail=str(a))
    check("bounty 点黄点后不再计 tile_dead miss",
          bo.state.get("hub_tile_misses", 0) == 0)
    # once 标记由 runner 在 tap 真落地后才写(base.pending 的契约, 08-08 craft
    #    JIT 丢发那次定的规矩), 离线这里手动模拟 ack 再验不重复点。
    bo.state["once:hub_tile_dot"] = True
    a2 = bo.on_task_hall(o_dot, st_hall)
    check("bounty 卡上点落地后不再重复点",
          a2 is None or a2.kind != "tap", detail=str(a2))

    # 2) 悬赏: 卡外的点不许当退路(防点到隔壁卡)
    bo2 = ALL["bounty"](ctx)
    bo2.setup()
    o_far = O(*hall, B(V.DOT_YELLOW, conf=0.88, cx=0.20, cy=0.90))
    a3 = bo2.on_task_hall(o_far, st_hall)
    check("bounty 卡区外的黄点不当入口",
          a3 is None or a3.kind != "tap", detail=str(a3))

    # 3) 大赛: 卡上红点 -> 点红点
    ar = ALL["arena"](ctx)
    ar.setup()
    o_red = O(*hall, B(V.DOT_RED, conf=0.91, cx=0.71, cy=0.82))
    a4 = ar.on_task_hall(o_red, st_hall)
    check("arena tile0 + 卡上红点 -> tap 红点",
          a4 is not None and a4.kind == "tap"
          and abs(a4.x - 0.71) < 0.05 and abs(a4.y - 0.82) < 0.05,
          detail=str(a4))

    # 4) jfd: 明确没有 hub_dot_region, 不许硬套(同帧那张卡上没点)
    jf = ALL["jfd"](ctx)
    jf.setup()
    a5 = jf.on_task_hall(O(*hall, B(V.DOT_YELLOW, conf=0.88, cx=0.61, cy=0.545)),
                         st_hall)
    check("jfd 无 hub_dot_region 不点隔壁卡的点",
          a5 is None or a5.kind != "tap", detail=str(a5))

    # 5) nav.hub_tile_dot 区域过滤本身
    inside = _nav.hub_tile_dot(O(B(V.DOT_RED, conf=0.9, cx=0.71, cy=0.82)),
                               (0.62, 0.74, 0.79, 0.92))
    outside = _nav.hub_tile_dot(O(B(V.DOT_RED, conf=0.9, cx=0.20, cy=0.20)),
                                (0.62, 0.74, 0.79, 0.92))
    check("hub_tile_dot 区内命中", inside is not None)
    check("hub_tile_dot 区外不命中", outside is None)

    # 6) HALT 连坐的 SKIPPED 必须带 skipped_because
    src_runner = (Path(__file__).resolve().parents[1] / "app" / "runner.py").read_text(
        encoding="utf-8")
    check("runner HALT 连坐打 upstream_halt 标",
          "skipped_because=upstream_halt" in src_runner)
    check("runner 超时未跑打 budget_exhausted 标",
          "skipped_because=budget_exhausted" in src_runner)

    # 7) event AP 读不出的收工文案点明是 OCR 失败, 且不写成没 AP
    src_event = (Path(__file__).resolve().parents[1] / "flow" / "event.py").read_text(
        encoding="utf-8")
    check("event AP 文案写明 OCR 失败", src_event.count("OCR 失败") >= 2)
    check("event AP 文案写明不等于没 AP", "这不等于没 AP" in src_event)

    src_shop = (Path(__file__).resolve().parents[1] / "flow" / "facilities.py").read_text(
        encoding="utf-8")
    buy_credit = src_shop.split("批量购买（花信用点）", 1)[1][:180]
    check("信用点批量买源码不带 money=True", "money=True" not in buy_credit)
    src_gate = (Path(__file__).resolve().parents[1] / "act" / "gate.py").read_text(
        encoding="utf-8")
    check("闸承认信用点/大赛币为非付费成交",
          '_SOFT_SPEND' in src_gate
          and "战术大赛货币" in src_gate)


def t_prod_three_0820():
    """08-20 生产三洞: 节点图误判 facility / 签到簿误 HALT / event tab_tries."""
    print("\n-- prod three 0820 ----")
    from routing_v2.state.machine import StateView as _SV

    ctx = Ctx(cfg=cfg(), log=lambda m: None)
    st_fac = _SV(page="facility", frames_in_page=3)

    sm = ALL["story_mining"](ctx)
    sm.setup()
    o_nodes = O(B(V.HOME, cx=0.965, cy=0.033),
                B(V.BACK, cx=0.045, cy=0.052),
                B(V.STORY_NODE_UNDONE, conf=0.97, cx=0.70, cy=0.25),
                B(V.STAGE_ENTER_LOCKED, conf=0.98, cx=0.85, cy=0.25),
                B(V.ARROW_RIGHT, cx=0.92, cy=0.50))
    a_n = sm.on_facility(o_nodes, st_fac)
    check("节点图误判 facility 不 src_i++", sm.state.get("src_i") == 0)
    check("节点图误判 facility 改走节点图(回图不换源)",
          a_n is not None and a_n.kind != "done"
          and a_n.target_cls == V.BACK, str(a_n))

    smw = ALL["story_mining"](ctx)
    smw.setup()
    a_w = smw.on_facility(
        O(B(V.ARROW_RIGHT, cx=0.92, cy=0.50),
          B(V.BACK, cx=0.045, cy=0.05)),
        st_fac)
    check("真支线墙(右切换无节点图标)仍翻页",
          a_w is not None and a_w.target_cls == V.ARROW_RIGHT, str(a_w))
    check("真支线墙翻页不 src_i++", smw.state.get("src_i") == 0)

    smo = ALL["story_mining"](ctx)
    smo.setup()
    smo.state["fac_swipes"] = 4
    a_o = smo.on_facility(
        O(B(V.BACK, cx=0.045, cy=0.05), B(V.HOME, cx=0.965, cy=0.033)),
        st_fac)
    check("真支线墙翻尽才换源", smo.state.get("src_i") == 1, str(a_o))

    book = O(B(V.PYROXENE, conf=0.87, cx=0.971, cy=0.385),
             B(V.CREDIT, conf=0.40, cx=0.72, cy=0.38),
             B(V.AP, conf=0.91, cx=0.80, cy=0.62))
    check("签到簿 BODY 青辉石不是 purchase_context",
          money_rules.purchase_context(book) is None,
          str(money_rules.purchase_context(book)))
    check("签到簿不打 money_popup",
          classify(book).interrupt is None,
          f"实际 {classify(book).interrupt}")

    booklet = O(B(V.NAV_DAILY_REWARD, conf=0.96, cx=0.22, cy=0.28),
                B(V.PYROXENE, conf=0.87, cx=0.80, cy=0.40),
                B(V.CLOSE_X, conf=0.95, cx=0.90, cy=0.10))
    check("每日领奖 booklet + BODY 石不当购买框",
          money_rules.purchase_context(booklet) is None,
          str(money_rules.purchase_context(booklet)))

    real = O(B(V.PYROXENE, conf=0.94, cx=0.50, cy=0.45),
             B(V.CONFIRM, cx=0.60, cy=0.75), B(V.CANCEL, cx=0.40, cy=0.75))
    check("真购买框仍是 purchase_context",
          money_rules.purchase_context(real) is not None,
          str(money_rules.purchase_context(real)))
    buyap = O(B(V.CONFIRM, conf=0.98, cx=0.598, cy=0.699),
              B(V.CANCEL, conf=0.98, cx=0.402, cy=0.699),
              B(V.QTY_MAX, conf=0.98, cx=0.687, cy=0.480),
              B(V.PLUS, conf=0.97, cx=0.630, cy=0.480))
    check("購買AP 双键+步进器仍是购买框",
          money_rules.purchase_context(buyap) is not None,
          str(money_rules.purchase_context(buyap)))
    near = O(B(V.SHOP_BUY_PYROXENE, cx=0.470, cy=0.520),
             B(V.CONFIRM, cx=0.60, cy=0.80), B(V.CANCEL, cx=0.40, cy=0.80))
    check("购买青辉石+对话框仍是购买流程",
          money_rules.purchase_context(near) is not None,
          str(money_rules.purchase_context(near)))

    ev = ALL["event"](ctx)
    ev.setup()
    ev.state.pop("tab_tries", None)
    st_ev = _SV(page="event_page", frames_in_page=3)
    try:
        a_tab = ev.on_event_page(
            O(B(V.EVENT_QUEST, cx=0.635, cy=0.151)), st_ev)
        tab_err = None
    except KeyError as e:
        a_tab = None
        tab_err = e
    check("event tab_tries 缺键不崩", tab_err is None, str(tab_err))
    check("event 缺键仍切 Quest",
          a_tab is not None and a_tab.target_cls == V.EVENT_QUEST, str(a_tab))
    check("event setup 有 tab_tries",
          "tab_tries" in ALL["event"](ctx).state)


if __name__ == "__main__":
    t_pages()
    t_machine()
    t_gate()
    t_free_pack()
    t_flows()
    t_route()
    t_config()
    t_invariants()
    t_vocab()
    t_ledger()
    t_deadbtn()
    t_ocr_geom()
    t_bucket()
    t_event_bonus_shop()
    t_alt_gates()
    t_shelf_walk()
    t_remain_gates_0816()
    t_v18_gates_0820()
    t_story_mining_0820()
    t_prod_three_0820()
    t_daily_fix_0820()
    t_story_stack_0821()
    t_schedule_locked_card_0821()
    print("\n" + "" * 52)
    if FAILS:
        print(f" {len(FAILS)} 项没过:")
        for f in FAILS:
            print(f"   · {f}")
        sys.exit(1)
    print(" 全过")
