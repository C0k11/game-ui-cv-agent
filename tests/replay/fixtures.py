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
    clicked = [str(a.get("reason", "")) for a in acts if a.get("action") == "click"]
    return (not clicked,
            f"AP 未知却点击={clicked[:2] or '无'} (期望零点击, 最终 defer)")


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
    screen = screen_from_tick(raw, jpg_path=jpg)
    if screen.frame is None:
        return (True, "SKIP: 帧读不出")
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
    mf = _pl.Path("data/raw_images/_classes.txt")
    if not mf.exists():
        return (True, "SKIP: 缺 master 词表")
    master = [l.strip() for l in mf.read_text(encoding="utf-8").splitlines()
              if l.strip()]
    bad = PG.validate_vocab(master)
    if bad:
        return (False, f"页面图引用了词表里没有的 cls: {bad}")
    want = {
        ("Lobby", "Bounty_SweepPanel"): 4,
        ("Cafe_Hall2", "Arena_Opponents"): 3,
        ("Formation_Attack", "Lobby"): 1,
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


CASES = [
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
