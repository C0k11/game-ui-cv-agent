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


CASES = [
    ("challenge_tab_假阳性", fx_challenge_false_positive, True),
    ("quest_list_真阳性(反向锚)", fx_quest_list_true_positive, True),
    ("上期活动474_不点", fx_prev_event_banner_no_tap, True),
    ("⛔买票框_绝不確認", fx_schedule_buy_dialog_never_confirm, True),
    ("⛔票到顶_零点击", fx_schedule_ticket_cap_stops, True),
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
