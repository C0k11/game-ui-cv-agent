# -*- coding: utf-8 -*-
"""活动 farm 选关推断器 v1 — 建议模式(只打印账单, 不接管配置)。

用户 2026-07-28: "怎么才能让程序自己推断(每天扫哪关)?"
核心信号 = 「商店买完后该币剩多少」(buy_event_shop 落的 event_economy.json):
  买完基本清零  → 商店还饿 → 这关继续刷
  买完剩一大笔  → 商店饱和(买无可买) → 这关停刷
这天然就是人工推断的逻辑(爪 21,035 花不掉 → Q12 停刷), 零新增 OCR 依赖。

⛔接管纪律(与 L2 页面图同款「先建议后接管」):
  v1 只输出建议与账单; event_farm_stages 的**用户显式配置永远优先**。
  建议连续与人工判断一致数日后, 才考虑让 daily 开工时自动覆写。

用法: py scripts/event_planner.py            # 打印账单+建议
      py scripts/event_planner.py --json     # 机器可读输出
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"D:\Project\ai game secretary")

# 本期常量(奇普托斯, 2026-07-21~08-04; 换活动要重标)
# ⛔只管前两币(用户 2026-07-30 纠偏: 第三币=盒抽/内置小游戏产出, 不好量化,
# Q12=盒抽门票关 — 这条链**不归程序推断**, 刷不刷 Q12 由用户手动配置决定。
# 我第一版把爪台账/盒抽需求塞进推断器 = 过度设计, 已删)。
TAB_STAGE = {1: 10, 2: 11}             # 币 tab ↔ 掉落关
HUNGRY_MAX = 500       # 购后余额 < 此值 = 商店还饿, 继续刷
SATURATED_MIN = 3000   # 购后余额 ≥ 此值 = 饱和, 停刷


def load(p, default):
    try:
        return json.loads((ROOT / p).read_text(encoding="utf-8"))
    except Exception:
        return default


def plan():
    eco = load("data/event_economy.json", {})
    tabs = eco.get("tabs", {})
    lines = []
    stages = []
    for ti in (1, 2):
        rec = tabs.get(str(ti))
        stage = TAB_STAGE[ti]
        if rec is None or rec.get("balance_after_buy") is None:
            # 无台账/读数失败 → fail 到"继续刷"(宁多刷不漏刷, 币不会烂手里)
            lines.append(f"tab{ti}(Q{stage}): 无购后余额台账 → 默认继续刷")
            stages.append(stage)
            continue
        bal = rec["balance_after_buy"]
        if bal < HUNGRY_MAX:
            lines.append(f"tab{ti}(Q{stage}): 购后余额 {bal} <{HUNGRY_MAX} "
                         f"= 商店还饿 → 刷")
            stages.append(stage)
        elif bal >= SATURATED_MIN:
            lines.append(f"tab{ti}(Q{stage}): 购后余额 {bal} ≥{SATURATED_MIN} "
                         f"= 饱和 → 停刷")
        else:
            lines.append(f"tab{ti}(Q{stage}): 购后余额 {bal} 中间带 → 半量(轮转尾)")
            stages.append(stage)   # v1 不做半量, 记账单即可

    cfg = load("data/app_config.json", {})
    prof = (cfg.get("profiles") or {}).get(cfg.get("active_profile", "default"), {})
    manual = prof.get("event_farm_stages") or []
    # 一致性只比 Q10/Q11: Q12(第三币门票)不归推断器管, 用户配了就照配置刷。
    manual_cmp = [s for s in manual if s in TAB_STAGE.values()]
    return {"suggested_stages": stages, "manual_stages": manual,
            "agree": (sorted(set(stages)) == sorted(set(manual_cmp))
                      if manual else None),
            "ledger_lines": lines}


if __name__ == "__main__":
    r = plan()
    if "--json" in sys.argv:
        print(json.dumps(r, ensure_ascii=False))
    else:
        print("== 活动选关账单 (建议模式, 不接管) ==")
        for ln in r["ledger_lines"]:
            print(" ", ln)
        print(f"建议 farm_stages = {r['suggested_stages']}")
        print(f"人工 farm_stages = {r['manual_stages']}"
              + (f"  → 一致性: {'一致 ✓' if r['agree'] else '不一致 ⛔'}"
                 if r["agree"] is not None else "  (未配置)"))
