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
TAB_STAGE = {1: 10, 2: 11, 3: 12}      # 币 tab ↔ 掉落关
HUNGRY_MAX = 500       # 购后余额 < 此值 = 商店还饿, 继续刷
SATURATED_MIN = 3000   # 购后余额 ≥ 此值 = 饱和, 停刷
# 爪(tab3 掉落币)特判: 商店 tab3 用的是符咒不是爪, 爪的需求=盒抽。
# 盒抽台账没建立前, 用保守常量: 剩余盒数未知按 5 盒 × 300抽 × 6爪 = 9000。
CLAW_DEMAND_FALLBACK = 9000


def load(p, default):
    try:
        return json.loads((ROOT / p).read_text(encoding="utf-8"))
    except Exception:
        return default


def plan():
    eco = load("data/event_economy.json", {})
    tabs = eco.get("tabs", {})
    claw = eco.get("claw", {})     # 盒抽台账(gift 脚本落): balance / boxes_left
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
    # 爪/Q12: 余额 vs 盒抽需求
    claw_bal = claw.get("balance")
    claw_demand = claw.get("demand", CLAW_DEMAND_FALLBACK)
    if claw_bal is None:
        lines.append(f"爪(Q12): 无余额台账 → 保守停刷(2026-07-28 帧证据 21,035 过剩)")
    elif claw_bal < claw_demand:
        lines.append(f"爪(Q12): 余额 {claw_bal} < 盒抽需求 {claw_demand} → 刷")
        stages.append(12)
    else:
        lines.append(f"爪(Q12): 余额 {claw_bal} ≥ 盒抽需求 {claw_demand} → 停刷")

    cfg = load("data/app_config.json", {})
    prof = (cfg.get("profiles") or {}).get(cfg.get("active_profile", "default"), {})
    manual = prof.get("event_farm_stages") or []
    return {"suggested_stages": stages, "manual_stages": manual,
            "agree": sorted(set(stages)) == sorted(set(manual)) if manual else None,
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
