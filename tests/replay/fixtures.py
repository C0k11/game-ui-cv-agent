# -*- coding: utf-8 -*-
"""事故回归 fixture(逻辑层) — 把已实锤的 live 事故钉死成秒级离线用例。

与 scripts/regression_suite.py 的分工:
  regression_suite = **模型层**  —— 跑 YOLO 看"还认不认得出"(要 GPU, 输入是 jpg)
  本文件           = **逻辑层**  —— 给定检出框看"skill 判得对不对"(不要 GPU,
                                    输入是 tick json 里的 yolo_boxes)
两层都过才叫没回归。统一入口: py scripts/regression_suite.py --domain logic

⛔定位: 这是**把已修 bug 钉死不复发**的工具, 不是发现新 bug 的手段, 更不是
跳过 live 逐帧审的理由(execution_doctrine #13)。

跑: py tests/replay/fixtures.py
"""
from __future__ import annotations

import json
import os
import sys

# 直跑入口在 GBK 控制台打印 '⛔' 会 UnicodeEncodeError 中断(后面用例全没跑,
# exit=1 走的是 traceback 不是 veto 契约) — 与 regression_suite.py:20 对齐。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

from tests.replay.harness import screen_from_tick  # noqa: E402

FX_DIR = os.path.join(_HERE, "frames")


def _load(name: str) -> dict:
    with open(os.path.join(FX_DIR, name), encoding="utf-8") as f:
        return json.load(f)


# ── ① Challenge tab 假阳性 ──────────────────────────────────────────────
def fx_challenge_false_positive():
    """run_20260723_204309 t799: Challenge tab 列表 — 恰 3 个入场键(cy 0.266/
    0.403/0.539) + 活动商店/活动任务底栏 + 未选中态「活动quest」在屏 + 0 个三星。

    旧 `_on_quest_list` 只看「≥2 入场键 + 活动底栏」→ 判 True → survey 在
    Challenge 的 3 关上死等 4 个 key 直到超时, **全天 swept=0, 两天连续**。
    正锚修复见 70b2540。此用例判 True = 同款事故重演。
    """
    from brain.skills.event_quest import EventQuestSkill
    raw = _load("challenge_tab_3keys.json")
    sk = EventQuestSkill()
    sk.reset()
    screen = screen_from_tick(raw)
    got = sk._on_quest_list(screen)
    return (got is False,
            f"_on_quest_list={got} (期望 False — Challenge 页不是 Quest 列表)")


def fx_quest_list_true_positive():
    """反向锚: 真正的 Quest 列表必须判 True, 否则上面那条只是把判据焊死成
    永远 False。run_20260724_033034 t48: 4 入场键 + 活动quest_已选择 + 三星。"""
    from brain.skills.event_quest import EventQuestSkill
    raw = _load("quest_list_real.json")
    sk = EventQuestSkill()
    sk.reset()
    got = sk._on_quest_list(screen_from_tick(raw))
    return (got is True,
            f"_on_quest_list={got} (期望 True — 这是真 Quest 列表)")


# ── ② 上期活动(领奖期)banner 绝不点 ─────────────────────────────────────
def fx_prev_event_banner_no_tap():
    """run_20260722_163044 t1: hub 轮播停在**上期活动**卡 —— 只有 474
    「距离奖励获得结束」@0.93, 没有 405「距离结束还剩」。

    405/474 逐帧交替从不共存(20 帧@0.45s 实测)。误入午夜派對三连(2026-07-22)
    的根因就是把 474 当成可进入的活动去点。此 tick 必须 **不点击**。
    """
    from brain.skills.event_quest import EventQuestSkill
    raw = _load("prev_event_banner_474.json")
    sk = EventQuestSkill()
    sk.reset()
    sk.sub_state = "enter"
    act = sk.tick(screen_from_tick(raw))
    a = act.get("action")
    return (a != "click",
            f"action={a!r} reason={act.get('reason')!r} (期望非 click — "
            f"474 在屏 ≠ 有活动可进)")


# ── ③ schedule 金钱防线 ⛔veto ─────────────────────────────────────────
# ⚠原计划是拿"5 区 tickets=-1"的真实录像帧建 fixture, **建不出来**: Schedule
# 是 DailyRoutine 的子 skill, 7 月 48 个 run 里 skill=="Schedule" 的 tick 数
# = **0**(全被记成 DailyRoutine, 子 skill 自己的 sub_state 从没落盘)。已补录
# `inner_sub_state` 字段, 但要等新 run 攒出来。
#
# 而且逐帧读码后发现: **"票数读不出就不派遣"根本不是这个 skill 的契约**。
# 派遣不靠票数读数把关(schedule.py:572-593 注释写死), 真防线是两条:
#   ② 弹窗 body 出现青辉石 → 取消, 绝不点確認
#   ③ 单日派遣硬上限 _MAX_TICKETS=7, 到顶什么都不点(不是"到顶还点确认")
# 所以钉这两条真契约, 而不是钉我以为的那条。
def _sched_screen(*boxes):
    from brain.skills.base import ScreenState
    return ScreenState(ocr_boxes=[], image_w=2560, image_h=1440, frame=None,
                       yolo_boxes=list(boxes))


def _yb(name, cx, cy, conf=0.95):
    from brain.skills.base import YoloBox
    return YoloBox(cls_id=-1, cls_name=name, confidence=conf,
                   x1=cx - 0.05, y1=cy - 0.02, x2=cx + 0.05, y2=cy + 0.02,
                   model_tag="ui")


def fx_schedule_buy_dialog_never_confirm():
    """買票對話框(body 出现青辉石) → 绝不点確認, 必须取消/退出。
    2026-06-02 那次青辉石买票事故就是这一步失守(money_safety)。"""
    from brain.skills.schedule import ScheduleSkill
    # 确认+取消同屏(_DIALOG_BAND y 0.66-0.90) + body(_PYROXENE_BODY_REGION
    # y 0.12-0.64) 出现青辉石价格 = 买票框
    screen = _sched_screen(
        _yb("确认键", 0.60, 0.72), _yb("取消键", 0.40, 0.72),
        _yb("青辉石", 0.50, 0.45),          # ★ body 里的价格 = 要花钱
        _yb("青辉石", 0.36, 0.034),         # 顶栏余额(不该被当成价格)
    )
    sk = ScheduleSkill()
    sk.reset()
    bad = []
    for _ in range(6):
        act = sk.tick(screen)
        r = str(act.get("reason", ""))
        if act.get("action") == "click" and "确认键" in r.replace("確認", "确认"):
            bad.append(r[:70])
    return (not bad, f"违规確認点击={bad or '无'} (期望: 买票框只能取消)")


def fx_schedule_ticket_cap_stops():
    """已达单日上限 → 一个键都不点(不是"到顶还点确认")。
    深挖 C1 实锤: 旧码先 +1 再比 `>` 放走了第 8 次派遣, 而第 8 次那个"确认"
    很可能正是买票框的确认键 = 亲手买票。"""
    from brain.skills.schedule import ScheduleSkill, _MAX_TICKETS
    screen = _sched_screen(
        _yb("确认键", 0.50, 0.72),          # 结算报告確認(=消耗一张票)
        _yb("课程表", 0.50, 0.10),
    )
    sk = ScheduleSkill()
    sk.reset()
    sk._dispatch_count = _MAX_TICKETS       # 已到顶
    bad = []
    for _ in range(4):
        act = sk.tick(screen)
        if act.get("action") == "click":
            bad.append(str(act.get("reason", ""))[:70])
    return (not bad,
            f"到顶({_MAX_TICKETS})后仍点击={bad or '无'} (期望: 零点击)")


def fx_schedule_buy_dialog_no_pyroxene_cls():
    """⛔2026-07-25 真实事故帧回归 —— 30 青辉石就是这么花掉的。

    run_20260724_201229 tick_0101: 屏上明明是「購買課程表票券 單價💎30
    總購買價格💎30」, 但 YOLO **对话框体内一个青辉石都没检出**(小图标压在深色
    价格条上), 只检出顶栏余额那个 → 旧版 _buy_dialog 单点依赖 body 青辉石 →
    返回 False → 防线整条哑火 → PRIORITY 1 把购买框的「確認」当成報告框的
    「確認」点掉 = 亲手买票。

    本 fixture 用的正是那一帧的真实检出清单(逐条抄自 tick_0101.json), 一个
    body 青辉石都没有。判据必须靠结构特征(数量步进器 / 取消+确认同屏)兜住。
    """
    from brain.skills.schedule import ScheduleSkill
    screen = _sched_screen(
        _yb("取消键", 0.402, 0.699, 0.98),
        _yb("确认键", 0.598, 0.699, 0.97),
        _yb("MAX_可点击", 0.687, 0.480, 0.97),   # ★ 数量步进器 = 购买框铁证
        _yb("减号灰色", 0.370, 0.480, 0.97),
        _yb("加号", 0.630, 0.480, 0.96),
        _yb("MIN_灰色", 0.315, 0.479, 0.94),
        _yb("弹窗叉叉", 0.889, 0.140, 0.96),
        _yb("课程表票", 0.054, 0.142, 0.96),
        _yb("返回键", 0.045, 0.051, 0.95),
        _yb("体力", 0.403, 0.032, 0.93),
        _yb("信用点", 0.564, 0.033, 0.88),
        _yb("青辉石", 0.740, 0.034, 0.51),        # 仅顶栏余额 — body 没有!
    )
    sk = ScheduleSkill()
    sk.reset()
    if not sk._buy_dialog(screen):
        return (False, "_buy_dialog 没认出真实购买框(事故重演)")
    bad = []
    for _ in range(6):
        act = sk.tick(screen)
        r = str(act.get("reason", "")).replace("確認", "确认")
        if act.get("action") == "click" and "确认键" in r:
            bad.append(r[:70])
    return (not bad, f"违规確認点击={bad or '无'} (期望: 只取消/退出)")


# ── ④ event_quest unlock 链: 購買AP 框绝不点確認 ────────────────────────
def _eq_buy_ap_screen():
    """run_20260711_144712/tick_0030 的**逐条实检出**(那一帧就是「購買AP」框)。
    ⚠body 里 **一个青辉石都没有** —— 单靠青辉石黑名单的闸在这一帧全盲。"""
    return _sched_screen(
        _yb("取消键", 0.402, 0.699, 0.976),
        _yb("返回键", 0.045, 0.051, 0.970),
        _yb("加号", 0.630, 0.480, 0.963),
        _yb("MAX_可点击", 0.687, 0.480, 0.962),
        _yb("确认键", 0.598, 0.699, 0.961),
        _yb("弹窗叉叉", 0.727, 0.242, 0.959),
        _yb("MIN_灰色", 0.313, 0.479, 0.957),
        _yb("体力", 0.820, 0.680, 0.916),
        _yb("信用点", 0.564, 0.033, 0.467),
    )


def fx_event_unlock_never_confirm_buy_ap():
    """⛔2026-07-25 全仓金钱审计 #1/#2: unlock 链上三处 確認 点击原本**零金钱闸**
    (编队確認 / battle settle / settle), 而同文件 _sweep_quest 的两处早就过
    `_dialog_is_purchase`。battle 子步的 `find_cls(BTN_CONFIRM, 0.6)` 在 300s
    窗口里持续武装 → 購買AP 框的確認@0.961 会被稳稳收下 = schedule 30 青辉石
    事故逐字复刻。本用例把三个子步逐个摆到这一帧上, 任何一个点確認 = 事故重演。
    """
    from brain.skills.event_quest import EventQuestSkill
    screen = _eq_buy_ap_screen()
    bad = []
    for step in ("confirm", "battle", "settle"):
        sk = EventQuestSkill()
        sk.reset()
        if sk._purchase_veto(screen, "test") is None:
            bad.append(f"{step}: _purchase_veto 没认出購買AP框")
            continue
        sk.sub_state = "unlock"
        sk._formation_step = step
        sk._quests = [{"num": 10, "cy": 0.5, "unlocked": False}]
        sk._unlock_idx = 0
        act = sk._unlock(screen)
        r = str(act.get("reason", "")).replace("確認", "确认")
        if act.get("action") == "click" and "PURCHASE" not in r and "cancel" not in r:
            bad.append(f"{step}: {r[:60]}")
    return (not bad, f"违规点击={bad or '无'} (期望: 三个子步都 veto)")


def fx_event_unlock_ap_gate_fail_closed():
    """AP 读不出(None)时 unlock **绝不进出击链**。旧码 `_ap is not None and
    _ap < 20` 是 fail-OPEN — 读不出直接放行, 而 _read_ap→None 是 live 常态。"""
    from brain.skills.event_quest import EventQuestSkill
    import brain.skills.event_quest as eq
    sk = EventQuestSkill()
    sk.reset()
    sk.sub_state = "unlock"
    sk._quests = [{"num": 10, "cy": 0.5, "unlocked": False}]
    sk._unlock_idx = 0
    sk._formation_step = ""
    screen = _sched_screen(_yb("入场键", 0.75, 0.50), _yb("入场键", 0.75, 0.62))
    # frame=None → _read_ap 必 None(数字 strip 读不了)
    acts = []
    for _ in range(int(eq._AP_READ_RETRY_SEC) + 30):
        acts.append(sk._unlock(screen))
        if sk.sub_state != "unlock":
            break
    # ⚠原版把迭代次数当秒数用: 紧凑调用只耗毫秒, 6s 重试窗内全是 wait,
    # "最终 defer" 分支从没被执行过(断言永真)。回拨墙钟, 真走到过期分支。
    import time as _t
    if sk.sub_state == "unlock":
        sk._ap_fail_t0 = _t.time() - (eq._AP_READ_RETRY_SEC + 1.0)
        acts.append(sk._unlock(screen))
    clicked = [str(a.get("reason", "")) for a in acts if a.get("action") == "click"]
    deferred = sk.sub_state == "points"
    timer_clean = getattr(sk, "_ap_fail_t0", -1.0) == 0.0
    ok = (not clicked) and deferred and timer_clean
    return (ok,
            f"AP 未知却点击={clicked[:2] or '无'}, defer到points={deferred}, "
            f"计时器清零={timer_clean} (期望: 零点击+defer+计时器不泄漏)")


# ── ⑤ schedule 关 popout 不许连发(after-ack) ───────────────────────────
def fx_schedule_popout_close_no_double_fire():
    """⛔2026-07-25: `close popout before switch` 连发两次, 第二发落在后面的
    区域屏上误开了一个设施。旧节流是 `_phase_ticks % 2` —— **tick 奇偶既不是
    时间也不是证据**, 而帧滞后让 `_roster_open` 在点击后仍为 True, `_dedup_click`
    又因为结构指纹变了而放行重复点击。

    ⚠这条**必须写成 fixture**: tests/replay 默认 `--mode stateless` 每 tick 新建
    skill, 结构上测不到任何跨 tick 状态; 而 sequential 模式下录制帧不会响应我们的
    点击, 状态机立刻发散 —— 两种回放都逮不到这一族。
    """
    from brain.skills.schedule import ScheduleSkill
    # popout 开着: 右上关闭叉(_popout_close 的 region) + 顶部居中课程表票
    screen = _sched_screen(
        _yb("弹窗叉叉", 0.888, 0.138, 0.96),
        _yb("课程表票", 0.450, 0.210, 0.95),
    )
    sk = ScheduleSkill()
    sk.reset()
    if not sk._roster_open(screen):
        return (False, "构造的 popout-open 帧没被 _roster_open 认出(用例失效)")
    sk.sub_state = "switch"
    acts = [sk._switch(screen) for _ in range(6)]
    clicks = [i for i, a in enumerate(acts)
              if a.get("action") in ("click", "back")]
    return (len(clicks) == 1,
            f"关 popout 发了 {len(clicks)} 次(期望 1, 其余等帧证据); "
            f"动作={[a.get('action') for a in acts]}")


# ── ⑥ L1-② 列表结构化解析: 关号是身份, 坐标不是 ────────────────────────
def fx_quest_rows_numbers():
    """run_20260724_204934/t0056 真帧: 5 行 Q08-Q12。

    ⛔这条钉的是 L1-② 的地基 —— 台账主键从 cy 换成关号。旧盘上落的是
    {"0.397": true, "0.871": true, ...} 这种**坐标当身份**的记录, 列表一滚动
    全部失效。关号条 = 已训 cls `关卡得星_3` box 正上方 [y1-2.4h, y1]
    (全语料 479 帧活动列表实测: 单格读出率 100%, 整帧连续递增 100%)。

    需要真 jpg 才能跑 OCR — 没有就 skip(不算失败, 但会说出来)。
    """
    import os
    import cv2
    from brain.skills.event_quest import EventQuestSkill
    jpg = os.path.join(FX_DIR, "quest_rows_5.jpg")
    if not os.path.exists(jpg):
        return (True, "SKIP: 缺 quest_rows_5.jpg")
    raw = _load("quest_rows_5.json")
    # ⚠load_frame 默认 False — 漏传过一次导致本用例永远 SKIP 还计成 PASS
    screen = screen_from_tick(raw, jpg_path=jpg, load_frame=True)
    if screen.frame is None:
        return (False, "帧读不出(jpg 在但解码失败) — 不许再当 SKIP 混过去")
    sk = EventQuestSkill()
    sk.reset()
    rows = sk.parse_quest_rows(screen)
    nums = [r["num"] for r in rows]
    if nums != [8, 9, 10, 11, 12]:
        return (False, f"关号解析={nums} (期望 [8,9,10,11,12])")
    # 行内配对: 每行都要有 star, 且 cy 自上而下递增
    if any(r["star"] is None for r in rows):
        return (False, f"有行没配到得星: {[r['star'] is None for r in rows]}")
    cys = [r["cy"] for r in rows]
    if cys != sorted(cys):
        return (False, f"行未按 cy 排序: {cys}")
    return (True, f"5 行解析正确 Q{nums} (关号=身份, cy 仅用于点击)")


def fx_bonus_ledger_rejects_cy_keys():
    """旧的 cy 主键台账必须被**作废**而不是当成关号读进来。
    `{"0.397": true}` 若被 int() 或误当关号, 会让 bot 以为某关做过了 →
    跳过真正该解锁的关(而解锁一关要 20AP)。"""
    import json as _j
    import os
    import tempfile
    from brain.skills.event_quest import EventQuestSkill
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    import time as _t
    _j.dump({"created": _t.time(),
             "quests": {"0.397": True, "0.871": True, "10": True}},
            open(path, "w", encoding="utf-8"))
    sk = EventQuestSkill()
    sk._BONUS_STATE_PATH = path
    sk.reset()
    led = dict(sk._bonus_ledger)
    os.unlink(path)
    ok = (led == {10: True})
    return (ok, f"台账={led} (期望 {{10: True}} — 两条 cy 主键必须作废)")




# ── ⑦ L2 页面图 ────────────────────────────────────────────────────────
def fx_page_graph_vocab_and_routes():
    """页面图的 cls 名必须全在模型词表里 —— 拼错一个字就是一条**永不命中**
    的死判据(实测逮到过 格黑娜学园中央区/千年研究区域 两个错名)。
    另钉几条关键路径, 防止改图时把边改断。"""
    import pathlib as _pl
    from brain.nav import page_graph as PG
    # ⚠路径必须锚仓库根(2026-07-25 审计): cwd 相对路径 + "缺文件→SKIP 计 PASS"
    # = 不从仓库根启动时整条词表校验静默消失 — 与 quest_rows load_frame 同族假验证。
    mf = _pl.Path(_HERE).parents[1] / "data" / "raw_images" / "_classes.txt"
    if not mf.exists():
        return (False, "master 词表缺失 — 词表校验没跑, 不许计 PASS")
    master = [l.strip() for l in mf.read_text(encoding="utf-8").splitlines()
              if l.strip()]
    bad = PG.validate_vocab(master)
    if bad:
        return (False, f"页面图引用了词表里没有的 cls: {bad}")
    want = {
        ("Lobby", "Bounty_SweepPanel"): 4,
        ("Cafe_Hall2", "Arena_Opponents"): 3,
        ("Formation_Attack", "Lobby"): 1,
        ("Lobby", "EventQuestList"): 2,      # 钉死 2026-07-25 补的活动入边
        ("Lobby", "EventSweepPanel"): 3,
    }
    for (a, b), n in want.items():
        r = PG.route(a, b)
        if r is None or len(r) != n:
            return (False, f"route {a}→{b} = {r} (期望 {n} 跳)")
    return (True, f"{len(PG.PAGES)} 页 / {len(PG.EDGES)} 边, 词表全对, 路径通")


def fx_page_graph_sweep_panels_not_confused():
    """⛔悬赏/交流会/活动 三家扫荡面板 cls 集合几乎一样, 只有票据不同。
    实测 219 帧活动扫荡面板被判成悬赏的(min_core 让票种没参与判定) →
    票据必须进 require。这条钉死那次修复。"""
    from brain.nav.page_graph import identify
    from brain.skills.base import ScreenState, YoloBox

    def scr(*names):
        return ScreenState(
            ocr_boxes=[], image_w=2560, image_h=1440, frame=None,
            yolo_boxes=[YoloBox(cls_id=-1, cls_name=n, confidence=0.95,
                                x1=0.4, y1=0.4, x2=0.5, y2=0.5,
                                model_tag="ui") for n in names])

    cases = [
        (("扫荡开始", "任务开始", "弹窗叉叉", "悬赏通缉票"), "Bounty_SweepPanel"),
        (("扫荡开始", "任务开始", "弹窗叉叉", "学院交流会票"), "Exchange_SweepPanel"),
        (("扫荡开始", "任务开始", "弹窗叉叉"), "EventSweepPanel"),
    ]
    bad = []
    for names, want in cases:
        got, why = identify(scr(*names))
        if got != want:
            bad.append(f"{names} → {got} (期望 {want})")
    return (not bad, f"误判={bad or '无'}")


def fx_arena_never_reached_not_complete():
    """⛔2026-07-25 假成功回归(run_20260725_231337 帧实锤):

    Arena 被丢在信用点商店页, enter 干等 24 tick 超时, 从商店退回大厅, 然后
    `_exit` 的 Lobby 分支报了 `done (0 fights, 0 rewards)` + "arena complete"。
    而当天真有活: 票 5/5 满 + 2 个未领奖励(重跑后 5 场全打完, 排名 37→32)。
    钉死两件事: ①没到达过就绝不许 reason 里出现 complete ②exit_report 三态正确。
    """
    from brain.skills.arena import ArenaSkill
    bad = []
    a = ArenaSkill(); a.reset()

    # ① 没到达 → done 的 reason 必须是 timeout 而不是 complete
    class _Scr:
        yolo_boxes: list = []
        frame = None
    a._goto("exit")
    a._enter_ticks = 25
    orig = a.detect_screen_yolo
    a.detect_screen_yolo = lambda s: "Lobby"        # 装成"已退回大厅"
    try:
        act = a._exit(_Scr())
    finally:
        a.detect_screen_yolo = orig
    r = str(act.get("reason", ""))
    if act.get("action") != "done":
        bad.append(f"未到达时 action={act.get('action')} (期望 done)")
    if "complete" in r.lower():
        bad.append(f"⛔未到达却报完成: {r!r}")
    if "timeout" not in r.lower():
        bad.append(f"reason 不含 timeout(拿不到 pipeline 重试): {r!r}")

    # ② 三态
    v0, _ = a.exit_report()
    a._reached = True; a._tickets = None
    v1, _ = a.exit_report()
    a._tickets = 3
    v2, _ = a.exit_report()
    a._tickets = 0
    v3, _ = a.exit_report()
    for got, want, lab in ((v0, "UNKNOWN", "未到达"), (v1, "UNKNOWN", "票数未读出"),
                           (v2, "LEFTOVER", "剩3张"), (v3, "CLEAN", "票0")):
        if got != want:
            bad.append(f"exit_report[{lab}]={got} (期望 {want})")
    return (not bad, f"违规={bad or '无'}; 未到达时 reason={r!r}")


def fx_shop_chain_flag_follows_plan():
    """⛔上游触发点回归: sub_only 路径下 shop 不该"留在店里等 arena_shop 接力"。

    2026-07-25 事故链的第一环 —— pipeline.py:1228 只给 top-level 那个 ShopSkill
    置了 chain_in_shop=False, daily_routine 内部 new 的那个没人管, 于是
    sub_only=["shop"] 跑完收在商店网格上, 把下一个 skill 丢在了未知页面。
    """
    from brain.skills.daily_routine import DailyRoutineSkill
    from brain.skills.shop import ShopSkill
    bad = []
    for so, want in ((["shop"], False), (["shop", "arena_shop"], True), (None, True)):
        d = DailyRoutineSkill(sub_only=so)
        for sk, _ in d._plan:
            if isinstance(sk, ShopSkill) and sk.chain_in_shop != want:
                bad.append(f"sub_only={so}: chain_in_shop={sk.chain_in_shop} "
                           f"(期望 {want})")
    return (not bad, f"违规={bad or '无'}")


# ── ⑦ 悬赏票数 strip: 关卡列表页读得出 + 零票绝不读成非零 ─────────────────
def _bounty_read_tickets(json_name: str, jpg_name: str):
    """跑真 Bounty skill 的 `_read_tickets` —— 不是复刻算法, 是调它本人。"""
    import os
    from brain.skills.bounty import BountySkill
    jpg = os.path.join(FX_DIR, jpg_name)
    if not os.path.exists(jpg):
        return None, f"SKIP: 缺 {jpg_name}"
    screen = screen_from_tick(_load(json_name), jpg_path=jpg, load_frame=True)
    if screen.frame is None:
        return None, "帧读不出(jpg 在但解码失败)"
    # ⚠这两张 fixture 都是 >=3200 宽的真 4K/准 4K 帧, 所以 run_digit_ocr 里
    # "帧太窄就换 ADB 干净帧重抓"那条路**不会触发** —— 离线跑得到的就是 live 值。
    if screen.frame.shape[1] < 3200:
        return None, f"fixture 帧宽 {screen.frame.shape[1]} <3200, 会走 ADB 升级路径"
    sk = BountySkill()
    sk.reset()
    return sk._read_tickets(screen), ""


def fx_bounty_tickets_stagelist_readable():
    """run_20260727_042153/t0339 真帧: 关卡列表页, 屏上「懸賞通緝票券 6/6」。

    ⛔旧 strip 在**这一整个页面**上从来读不出 —— 全语料 4K 实测 **0/211 帧**。
    根因不是切太紧也不是对比度: 两个页面的标签字数不同(持有票券 4 字 /
    懸賞通緝票券 6 字), 数字被推到 icon.x2+6.1~7.3 个 icon 宽, 而旧右界只到
    +0.112 绝对宽; 更要命的是 y 留白 0.4*bh 让 DB 检测器整条返回空
    (同帧放到 1.2*bh 立刻检出 '6/6' score 0.75)。
    读不出 ⇒ fail-closed ⇒ **6 张票 + 当天 2 倍奖励整轮弃掉**。
    """
    got, err = _bounty_read_tickets("bounty_stagelist_6.json",
                                    "bounty_stagelist_6.jpg")
    if err:
        return (err.startswith("SKIP"), err)
    return (got == 6, f"读出 {got} (期望 6 — 屏上「懸賞通緝票券 6/6」)")


def fx_bounty_zero_tickets_never_overread():
    """run_20260612_211617/t0199 真帧: 屏上「持有票券 **0/6**」。

    ⛔旧 strip 把它读成 `'9/0'` → `_read_tickets` 返回 **9** → `_ticket_check`
    以为有票 → 出击 → 0 票出击弹出的正是**青辉石買票框**。
    `tickets == 0 → exit` 是 money_safety 的**源头闸**, 被读数直接架空。
    全语料 4K 实测: 114 张零票屏里旧逻辑有 **18 张**读成 >0。
    ⚠fail-closed 只挡 None, **挡不住读大** —— 所以这条用例判的是"绝不 >0",
    读成 None 也算过(安全), 读成正数就是事故重演。
    """
    got, err = _bounty_read_tickets("bounty_zero_tickets.json",
                                    "bounty_zero_tickets.jpg")
    if err:
        return (err.startswith("SKIP"), err)
    return (got is None or got == 0,
            f"读出 {got} (屏上 0/6; 期望 0 或 None, 绝不能是正数)")


def fx_sweep_confirm_not_mistaken_for_done():
    """run_20260602_182826/t0010 真帧: 掃蕩确认框 — 確認@cy0.792 + 取消@cy0.796,
    **无 获得奖励**。

    ⛔2026-07-27 live 实锤(bounty 6 票 + 2 倍奖励): `_confirm` 的"掃蕩完成了"
    分支只看「確認键落在 _DONE_CONFIRM_BAND(y 0.74-0.90)」, 而**确认框弹入
    动画**里確認键正好从下往上扫过这条带(终值 cy 0.699 在带外) ⇒ 假计一次
    cycle → goto result → **真正的確認从没点过** → 票数被弹窗挡住读不出 →
    exit → 点叉叉 = 取消 = 6 张票一张没花。

    负门禁(全语料 44,669 有检出帧实测): 结果弹窗在屏 494 帧, **带取消键 0 帧**;
    確認键落 DONE 带 588 帧, **其中 252 帧同屏有取消键**(全是确认框), 且这 588
    帧一帧都没有 获得奖励。

    本用例判"进了 result / 计了 cycle" = 同款事故重演。
    """
    import os
    from brain.skills.bounty import BountySkill
    jpg = os.path.join(FX_DIR, "sweep_confirm_dialog.jpg")
    if not os.path.exists(jpg):
        return (True, "SKIP: 缺 sweep_confirm_dialog.jpg")
    screen = screen_from_tick(_load("sweep_confirm_dialog.json"),
                              jpg_path=jpg, load_frame=True)
    sk = BountySkill()
    sk.reset()
    sk._goto("confirm")
    sk._phase_ticks = 30          # 越过 pre-transition 等待窗, 直接考判定
    act = sk._confirm(screen)
    bad = []
    if sk.sub_state == "result":
        bad.append("进了 result")
    if sk._sweep_cycles != 0:
        bad.append(f"假计 cycle={sk._sweep_cycles}")
    if act.get("action") != "click":
        bad.append(f"没去点確認(action={act.get('action')})")
    return (not bad,
            f"{'/'.join(bad) if bad else '未误判'} — "
            f"sub={sk.sub_state} cycles={sk._sweep_cycles} "
            f"action={act.get('action')} reason={act.get('reason')!r}")


def fx_event_panel_fake_confirm_not_a_dialog():
    """run_20260727_220013/t0276 真帧: **裸的活动扫荡面板**(步进器全灰), 屏上
    多出一个 `确认键 conf=0.81 @ cy=0.961` —— 面板底部的东西, 不是对话框按钮。

    ⛔旧码 `find_cls(BTN_CONFIRM, conf=0.6)` **不带 region**(与 840AP 那次同病),
    全屏 argmax 捡到它 → 有確認无取消 ⇒ 判成「掃蕩完成框」→ 过结构闸 →
    而扫荡面板**自带数量步进器**(MIN/MAX/加减号 @cy0.415) ⇒ `_dialog_is_purchase`
    命中 ⇒ **判成购买框, 整轮扫荡 back off**。当时 AP=445 ≈ 22 次扫荡。

    本用例同时钉两件事:
      ① 带 region 后这一帧**取不到** 対话框確認(假阳被排除) —— 这是修复本身;
      ② `_dialog_is_purchase` 在这一帧**确实为 True** —— 证明"band 才是那道闸",
         不是结构闸自己变好了。②失败说明这条用例已经测不到原 bug 了。
    """
    import os
    from brain.skills.event_quest import EventQuestSkill, _DIALOG_BTN_BAND
    from brain.skills import ui_classes as UC
    jpg = os.path.join(FX_DIR, "event_sweep_panel_fake_confirm.jpg")
    if not os.path.exists(jpg):
        return (True, "SKIP: 缺 event_sweep_panel_fake_confirm.jpg")
    screen = screen_from_tick(_load("event_sweep_panel_fake_confirm.json"),
                              jpg_path=jpg, load_frame=True)
    sk = EventQuestSkill()
    sk.reset()
    banded = sk.find_cls(screen, UC.BTN_CONFIRM, conf=0.6,
                         region=_DIALOG_BTN_BAND)
    unbanded = sk.find_cls(screen, UC.BTN_CONFIRM, conf=0.6)
    is_buy = sk._dialog_is_purchase(screen)
    bad = []
    if banded is not None:
        bad.append(f"带 region 仍取到確認 @cy={(banded.y1 + banded.y2) / 2:.3f}")
    if unbanded is None:
        bad.append("不带 region 也取不到 — 这帧已复现不出原 bug, 用例失效")
    if not is_buy:
        bad.append("_dialog_is_purchase=False — 结构闸变了, 用例不再测原路径")
    return (not bad, "/".join(bad) if bad else
            f"假確認被 band 排除(裸帧 argmax 会捡到 cy="
            f"{(unbanded.y1 + unbanded.y2) / 2:.3f}), 结构闸仍会误判={is_buy}")


def _sched_read_tickets(json_name: str, jpg_name: str):
    """跑真 ScheduleSkill 的 `_read_tickets`(不是复刻算法, 是调它本人)。"""
    import os
    from brain.skills.schedule import ScheduleSkill
    jpg = os.path.join(FX_DIR, jpg_name)
    if not os.path.exists(jpg):
        return None, f"SKIP: 缺 {jpg_name}"
    screen = screen_from_tick(_load(json_name), jpg_path=jpg, load_frame=True)
    if screen.frame is None:
        return None, "帧读不出(jpg 在但解码失败)"
    if screen.frame.shape[1] < 3200:
        return None, f"fixture 帧宽 {screen.frame.shape[1]} <3200, 会走 ADB 升级路径"
    sk = ScheduleSkill()
    sk.reset()
    return sk._read_tickets(screen), ""


def fx_sched_tickets_cls_anchored():
    """walk_20260728_a 步 024 真帧: 全體課程表 popout, 屏上「持有票券 3/7」。

    ⛔旧法(写死矩形 `_TICKET_REGION`)在这一帧读 **None**。今天 46 帧对拍:
    旧法读出 11 帧 / 新法(cls 锚定)读出 30 帧, **19 处分歧全是「旧 None 新有值」**
    —— 新法从不与旧法矛盾, 只在旧法瞎的地方补上。新法读出的序列
    7,7,7→6,6,6→5,5,5→4,4→3×6→2,2 单调递减零跳变; 人眼在两点交叉验证过
    (步18 缩略图「4/7」/ 步21「3/7」)。
    """
    got, err = _sched_read_tickets("sched_tickets_3.json", "sched_tickets_3.jpg")
    if err:
        return (err.startswith("SKIP"), err)
    return (got == 3, f"读出 {got} (期望 3 — 屏上「持有票券 3/7」)")


def fx_sched_ticket_decoy_never_read():
    """walk_20260728_a 步 034 真帧: 同一个 `课程表票` cls 在屏上有**两处** ——
    popout 表头 cy≈0.142(被弹窗压住, 读不出) 与 **課程表資訊 弹窗里 cy≈0.698**。

    ⛔后者是「3 → 2」**派遣前后指示器**, 不是余额。往右取数字条读出来是
    '21'/'1' —— 而 `'1'` **格式完全合法且在 0..7 内**, 不带 y 门的"找到票图标
    就往右读"会把它当成"还剩 1 张"(真值是 2)。票数是金钱闸输入, 读小丢票、
    读大则可能在 0 票时去点開始 → 弹青辉石購買框。
    本用例判"读出 1" = 诱饵没挡住。
    """
    got, err = _sched_read_tickets("sched_ticket_decoy.json",
                                   "sched_ticket_decoy.jpg")
    if err:
        return (err.startswith("SKIP"), err)
    return (got in (None, 2),
            f"读出 {got} (期望 None 或 2; 读成 1 = 吃了 cy0.698 那个「3→2」诱饵)")


def fx_craft_empty_slots_still_enter():
    """⛔制造红点门控的**第三态**: 三个槽位全空 → 同样没有红点 → 旧码永远 skip。

    2026-07-28 帧实锤(data 帧: 材料清單三行全是「＋開始製造」, 一次領取灰):
    旧 should_run 只认两态 ①有红点=可领 ②无红点=还在造→skip。第三态「根本没在
    造」被归进 ② ⇒ **一次都不会开新制造**, 而且一旦空了就再也不会有红点 =
    **自锁**。修法 = 红点 OR 今日未进过(日台账)。

    这里同时钉住反向: **今天已经进过且无红点就必须 skip** —— 否则台账形同虚设,
    每一轮 daily 都要白跑一趟制造往返。
    """
    import brain.skills.craft as cm
    from brain.skills.craft import CraftSkill
    day = cm._game_day()
    saved = cm._load_craft_state()
    # 大厅帧: 制造入口在, 身上没有红点(红点在别处 = 不属于制造)
    screen = _sched_screen(
        _yb("制造入口", 0.531, 0.953, 0.96),
        _yb("商店入口", 0.621, 0.953, 0.97),
        _yb("红点", 0.856, 0.714, 0.92),      # 任务大厅的红点, 不在制造入口上
        _yb("黄点", 0.098, 0.940, 0.89),      # 咖啡厅的黄点
    )
    try:
        cm._save_craft_state({"game_day": day, "visited": False, "started": 0})
        first = CraftSkill().should_run(screen)
        cm._save_craft_state({"game_day": day, "visited": True, "started": 1})
        second = CraftSkill().should_run(screen)
        cm._save_craft_state({"game_day": "1970-01-01", "visited": True,
                              "started": 9})
        stale = CraftSkill().should_run(screen)
    finally:
        cm._save_craft_state(saved)
    ok = (first is True and second is False and stale is True)
    return (ok, f"今日未进过={first}(期望True, 第三态) / 已进过={second}"
                f"(期望False) / 昨日台账={stale}(期望True)")


def fx_momo_panel_identity_gate():
    """⛔点开学生 A 之后, 右侧会话面板还停在 B —— 绝不在 B 的面板上替 A 干活。

    2026-07-28 帧实锤: 点了「贵音」, 右侧气泡头像全是 `凯伊 cx=0.717`, 连羁绊
    剧情概要正文写的都是 Kei。与 event_quest「换关不关旧弹窗、只有 label 变了」
    同构 —— **参数变了 ≠ 屏幕变了**。
    钉两件事: ①面板是别人时**一个 click 都不许发** ②面板换对了要立刻放行。
    """
    from brain.skills.momo_talk import MomoTalkSkill

    def _av(name, cx, cy, conf=0.99):
        from brain.skills.base import YoloBox
        return YoloBox(cls_id=-1, cls_name=name, confidence=conf,
                       x1=cx - 0.02, y1=cy - 0.02, x2=cx + 0.02, y2=cy + 0.02,
                       model_tag="avatar")

    # 左栏列表头像(cx 0.205) + 右侧会话面板头像(cx 0.57-0.72) + 一个可点的回复选项
    left = [_av("贵音", 0.205, 0.349), _av("爱丽丝(战斗)", 0.205, 0.455)]
    reply = _yb("学生信息回复选项", 0.747, 0.678, 0.98)

    def _screen(panel_name):
        return _sched_screen(*left, _av(panel_name, 0.568, 0.337),
                             _av(panel_name, 0.717, 0.381), reply)

    sk = MomoTalkSkill()
    sk.reset()
    sk._cur_student = "贵音"
    sk._goto("dialogue")
    sk.mark("convo")

    # ① 面板是「凯伊」→ 不许点
    bad = []
    for _ in range(4):
        act = sk._dialogue(_screen("凯伊"))
        if act.get("action") == "click":
            bad.append(str(act.get("reason", ""))[:60])

    # ② 面板换成「贵音」→ 必须立刻能干活(点回复选项)
    act2 = sk._dialogue(_screen("贵音"))
    ok2 = act2.get("action") == "click" and "reply" in str(act2.get("reason", ""))

    return (not bad and ok2,
            f"面板是别人时的违规点击={bad or '无'}; 面板换对后能干活={ok2} "
            f"(实际 {act2.get('action')}/{str(act2.get('reason'))[:40]})")


def fx_momo_unknown_dialog_never_confirm():
    """⛔MomoTalk 里的未知 確認+取消 框 → 只能取消。

    2026-07-28 帧实锤: 点「進入羈絆劇情」弹出**劇透警告框**(按钮 cy≈0.913,
    落在 _story 判定带 y0.55-0.85 之外 ⇒ 旧码根本不处理它), 而框上明写
    **「點擊確認時將前往活動頁面」** —— 点確認就把 bot 带出 MomoTalk。
    反向也要钉: 合法的「是否略過此劇情?」框(同屏有 剧情menu/跳过故事键 chrome)
    必须仍然能確認, 否则羁绊剧情永远跳不过去。
    """
    from brain.skills.momo_talk import MomoTalkSkill

    # ① 劇透警告框: 双钮 @cy0.913, 屏上无 story chrome
    warn = _sched_screen(
        _yb("确认键", 0.603, 0.913), _yb("取消键", 0.397, 0.913),
        _yb("学生momotalk信息未读", 0.506, 0.778, 0.86),
    )
    sk = MomoTalkSkill()
    sk.reset()
    bad = []
    cancels = 0
    for _ in range(4):
        act = sk.tick(warn)
        r = str(act.get("reason", ""))
        if act.get("action") == "click":
            if "取消" not in r:
                bad.append(r[:60])
            else:
                cancels += 1
    # ⛔连发也要钉: 这道闸第一版就漏了 after-ack, 上线第一次 live 就被 step_walk
    # 逮到「取消连发 3 次」(第 3 发时框已关掉, 拍在面板下方空白上)。
    if cancels != 1:
        bad.append(f"取消发了 {cancels} 次(期望 1, 其余该是 after-ack wait)")

    # ② 略過剧情框: 双钮 @cy0.724 + 右上角 story chrome → 必须还能確認
    skipdlg = _sched_screen(
        _yb("确认键", 0.599, 0.724), _yb("取消键", 0.401, 0.727),
        _yb("剧情menu", 0.941, 0.055), _yb("跳过故事键", 0.946, 0.166),
    )
    sk2 = MomoTalkSkill()
    sk2.reset()
    sk2.sub_state = "story"
    sk2._story_cut = 1
    ok2 = False
    for _ in range(3):
        act = sk2.tick(skipdlg)
        if act.get("action") == "click" and "confirm story skip" in str(act.get("reason", "")):
            ok2 = True
            break

    return (not bad and ok2,
            f"劇透警告框的违规非取消点击={bad or '无'}; 略過框仍能確認={ok2}")


def fx_cafe_switch_2f_no_mutate_before_ack():
    """⛔点「移动至2号点」时**绝不能**先把状态改成"我在2F了"。

    2026-07-28 全仓审计确认(我逐行核过代码): 旧码在 action_click_box 之前就
    `_begin_invite(floor_2=True)`(内含 `_on_2f=True` + `_goto("invite")`)。
    reason "switch to cafe 2F" 不在 pipeline 关键词豁免表里 ⇒ 这一发可能被稳定门
    吞掉/丢 tap, 而状态机已离开 switch 态、再不会点第二次, 全程零帧证据。
    后果: 人还在 1F 却按 2F 名单邀请/摸头, **2F 整层白丢**。
    到达证据是现成的: 2F 上按钮变成 CAFE_MOVE_1F。
    同时钉住不许连发(after-ack)。
    """
    from brain.skills.cafe import CafeSkill
    screen = _sched_screen(_yb("移动至2号点", 0.12, 0.10, 0.97))
    sk = CafeSkill()
    sk.reset()
    sk._goto("switch")
    act1 = sk._switch_floor(screen)
    mutated = bool(sk._on_2f) or sk.sub_state != "switch"
    # 第二次: 还没到 2F(屏上仍是 移动至2号点) → 必须是 after-ack wait, 不许再点
    act2 = sk._switch_floor(screen)
    double = act2.get("action") == "click"
    ok = (act1.get("action") == "click" and not mutated and not double)
    return (ok, f"首点={act1.get('action')} / 点前就改状态={mutated}"
                f"(必须 False) / 连发={double}(必须 False)")


def fx_cafe_dry_gate_needs_wallclock():
    """⛔摸头「这片视野扫干净了」不能只数帧 —— 帧数够但墙钟没到时不许判 dry。

    `_HEADPAT_DRY_FRAMES=7` 的注释自己算过账「7×280≈2s」, 但那 280ms 被
    server/app.py:1519 的 ZERO-WAIT 丢弃(reason 不含 loading → 一律 0.12s),
    实际只剩 ~1.4s, 而气泡最晚 600ms 才渲染 ⇒ 提前判 dry 就 pan 走 = 漏摸
    (live 2026-06-09 1F漏摸一个)。
    本 fixture 在**紧循环**里跑, 真实墙钟 ≈0 ⇒ 即使喂满 7+ 个空帧, 也必须
    继续 settle, 绝不能判 dry 去 pan/收工。
    """
    from brain.skills.cafe import CafeSkill, _HEADPAT_DRY_FRAMES
    empty = _sched_screen(_yb("移动至2号点", 0.12, 0.10, 0.97))
    sk = CafeSkill()
    sk.reset()
    sk._goto("headpat")
    sk._pat_settle = 0
    sk._pat_settle_sec = 0.0
    bad = []
    for i in range(_HEADPAT_DRY_FRAMES + 4):
        act = sk._headpat(empty)
        r = str(act.get("reason", ""))
        if act.get("action") in ("swipe", "click") or "done" in str(act.get("action")):
            bad.append(f"第{i+1}帧就 {act.get('action')}: {r[:50]}")
            break
    return (not bad,
            f"喂 {_HEADPAT_DRY_FRAMES + 4} 个空帧(墙钟≈0) 的越界动作={bad or '无'} "
            f"(期望全程 settle/scan, 不许 pan 或收工)")


def fx_cafe_claim_earnings_no_premature_done():
    """⛔点「領取」时绝不能先把 _earnings_done 置 True —— 那一发会被稳定门吞掉。

    2026-07-28 live 实锤:
        tick=13 wait  帧未稳定(转场/滚动) — 等稳定帧: claim earnings
        tick=14 wait  earnings done → invite      ← 状态却已"领完了"
        tick=15 click close leftover earnings before invite   ← 关窗, 钱没领
    帧证据: 那一帧 `領取_黄 0.97 @cy0.733` 原封不动。
    机制: `_force_settle=True`(昨天为修"锚在弹出动画帧上打空"加的, 本身正确)
    让这一发可能被吞, 而旧码在动作发出前就 `_earnings_done=True` ⇒ 下一 tick
    本函数第一行直接转 invite, **永远不重试**。
    ⇒ 一个正确的修复把另一个一直存在的 bug 顶出了水面。
    本 fixture 钉两件事: ①点击那一刻 _earnings_done 必须仍为 False
                        ②同一帧再来一次必须是 after-ack wait, 不许连发
    """
    from brain.skills.cafe import CafeSkill
    # 收益弹窗开着, 領取_黄 在 claim band (0.28,0.50,0.74,0.90) 内
    screen = _sched_screen(
        _yb("领取_黄", 0.500, 0.733, 0.97),
        _yb("咖啡厅收益", 0.919, 0.893, 0.98),
    )
    sk = CafeSkill()
    sk.reset()
    sk._goto("earnings")
    act1 = sk._earnings(screen)
    premature = bool(sk._earnings_done)
    act2 = sk._earnings(screen)
    double = act2.get("action") == "click"
    # ⛔第二次 live 才逮到的那条: 点被吞后, 只要 領取_黄 还在屏上, **绝不许**
    # 从"到达证据"分支判领完 —— 旧判据 `_earnings_claimed and _is_cafe` 会成立,
    # 因为咖啡页签名(咖啡厅邀请卷/回大厅按钮/返回键)在弹窗开着时照样检出。
    screen_with_cafe_sig = _sched_screen(
        _yb("领取_黄", 0.500, 0.733, 0.97),
        _yb("咖啡厅邀请卷", 0.692, 0.904, 0.97),
        _yb("回大厅按钮", 0.965, 0.032, 0.97),
        _yb("返回键", 0.045, 0.052, 0.98),
    )
    act3 = sk._earnings(screen_with_cafe_sig)
    false_done = bool(sk._earnings_done) or "invite" in str(act3.get("reason", ""))
    ok = (act1.get("action") == "click"
          and "claim earnings" in str(act1.get("reason", ""))
          and act1.get("_force_settle") is True
          and not premature and not double and not false_done)
    return (ok, f"首点={str(act1.get('reason'))[:28]} / force_settle="
                f"{act1.get('_force_settle')} / 点时就报领完={premature}(必须False)"
                f" / 连发={double}(必须False) / 領取_黄还在却判领完={false_done}"
                f"(必须False)")


CASES = [
    ("⛔cafe领收益_点时绝不报领完", fx_cafe_claim_earnings_no_premature_done, True),
    ("⛔cafe换2F_点前绝不改状态", fx_cafe_switch_2f_no_mutate_before_ack, True),
    ("⛔cafe判dry_帧数够也要等墙钟", fx_cafe_dry_gate_needs_wallclock, True),
    ("⛔momo未知確認框_只能取消", fx_momo_unknown_dialog_never_confirm, True),
    ("⛔momo面板是别人时_一个click都不许发", fx_momo_panel_identity_gate, True),
    ("⛔制造槽位全空_无红点也要进", fx_craft_empty_slots_still_enter, True),
    ("⛔arena未到达_绝不报完成", fx_arena_never_reached_not_complete, True),
    ("⛔shop留店标志跟随plan", fx_shop_chain_flag_follows_plan, True),
    ("challenge_tab_假阳性", fx_challenge_false_positive, True),
    ("quest_list_真阳性(反向锚)", fx_quest_list_true_positive, True),
    ("上期活动474_不点", fx_prev_event_banner_no_tap, True),
    ("⛔买票框_绝不確認", fx_schedule_buy_dialog_never_confirm, True),
    ("⛔票到顶_零点击", fx_schedule_ticket_cap_stops, True),
    ("⛔真实事故帧_购买框无青辉石cls", fx_schedule_buy_dialog_no_pyroxene_cls, True),
    ("⛔unlock链_購買AP框绝不確認", fx_event_unlock_never_confirm_buy_ap, True),
    ("⛔unlock_AP读不出_fail-closed", fx_event_unlock_ap_gate_fail_closed, True),
    ("关popout不连发(after-ack)", fx_schedule_popout_close_no_double_fire, False),
    ("L1②_关号解析Q08-12", fx_quest_rows_numbers, True),
    ("L1②_旧cy主键台账必作废", fx_bonus_ledger_rejects_cy_keys, True),
    ("L2_页面图词表+路径", fx_page_graph_vocab_and_routes, True),
    ("L2_三家扫荡面板不混淆", fx_page_graph_sweep_panels_not_confused, True),
    ("悬赏票数_关卡列表页读得出", fx_bounty_tickets_stagelist_readable, False),
    ("⛔悬赏零票_绝不读成非零", fx_bounty_zero_tickets_never_overread, True),
    ("⛔掃蕩确认框_绝不当成完成弹窗", fx_sweep_confirm_not_mistaken_for_done, True),
    ("课程表票数_cls锚定读得出", fx_sched_tickets_cls_anchored, False),
    ("⛔课程表票_绝不吃「3→2」诱饵", fx_sched_ticket_decoy_never_read, True),
    ("⛔面板底部假確認_不当对话框", fx_event_panel_fake_confirm_not_a_dialog, True),
]


def run() -> int:
    print("=== 逻辑层事故回归 fixture ===")
    fails = 0
    veto_fail = 0
    for name, fn, veto in CASES:
        try:
            ok, detail = fn()
        except Exception as e:                                   # noqa: BLE001
            ok, detail = False, f"异常 {type(e).__name__}: {e}"
        mark = "PASS" if ok else ("FAIL⛔" if veto else "FAIL")
        print(f"  [{mark}] {name}\n         {detail}")
        if not ok:
            fails += 1
            if veto:
                veto_fail += 1
    print(f"\n{len(CASES) - fails}/{len(CASES)} 通过")
    return 2 if veto_fail else (1 if fails else 0)


if __name__ == "__main__":
    sys.exit(run())
