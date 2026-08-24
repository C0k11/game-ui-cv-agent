# -*- coding: utf-8 -*-
"""活动 farm 选关推断器 v2 — 建议模式(只打印账单, 不接管配置)。

架构(用户 2026-07-30 定): 一切从**商店状态**推断, 零 hardcode。
  币种: buy_event_shop 从 YOLO 货币 tab cls 数出 K 个(4 币活动自动成立),
        最后一个 = 盒抽/小活动产物币, 不归程序管(用户手配是否刷门票关)。
  关映射: 活动 Quest 尾 K 关 = K 币专刷关按序对应(playbook 实证规律),
        max_stage 由 event_quest 从 YOLO 关号 cls 落账, 不写死。
  刷/停判据: **货架, 不是余额**(2026-07-30 纠偏: 余额 0 + 货架有亮位 =
        还饿要刷; 余额 0 + 全暗 = 搬空停刷 — 纯余额把这两种判反)。
        亮位 = YOLO 购买按钮 det + gray 双峰(买得到的非家具位)。
  AP 分配目标 = 把商店搬空。

接管纪律(与 L2 同款「先建议后接管」): v2 只输出建议与账单;
  event_farm_stages 的**用户显式配置永远优先**。

用法: py scripts/event_planner.py            # 打印账单+建议
      py scripts/event_planner.py --json     # 机器可读输出
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"D:\Project\ai game secretary")

HUNGRY_MAX = 500       # 余额兜底判据(shelf 缺失时): < 此值 = 还饿
SATURATED_MIN = 3000   # 余额兜底判据: ≥ 此值 = 饱和


def load(p, default):
    try:
        return json.loads((ROOT / p).read_text(encoding="utf-8"))
    except Exception:
        return default


def plan():
    eco = load("data/event_economy.json", {})
    tabs = eco.get("tabs", {})
    n_tabs = eco.get("n_tabs")          # 含盒抽币(buy_event_shop 落)
    max_stage = eco.get("max_stage")    # 活动 Quest 最大关号(event_quest 落)
    lines = []
    stages = []

    def stage_of(ti):
        """尾 K 关规律: tab i <-> 关(max_stage - (n_tabs - i))。缺账时 None。"""
        if max_stage and n_tabs:
            return max_stage - (n_tabs - ti)
        return None

    managed = sorted(int(k) for k in tabs.keys())
    if n_tabs:
        managed = [t for t in managed if t < n_tabs]   # 最后一币不管
    for ti in managed:
        rec = tabs.get(str(ti)) or {}
        stage = stage_of(ti)
        tag = f"Q{stage}" if stage else "关=?(待event_quest落关号)"
        bright = rec.get("shelf_bright")
        bal = rec.get("balance_after_buy")
        if bright is not None:
            # 主判据 = 货架(YOLO 亮位)
            if bright > 0:
                lines.append(f"tab{ti}({tag}): 货架亮位 {bright} 个 = 还有可买"
                             f"  刷")
                verdict = True
            else:
                lines.append(f"tab{ti}({tag}): 货架全暗/空 = 搬空  停刷")
                verdict = False
        elif bal is not None:
            # 兜底 = 余额(粗信号, 只在无货架数据时用)
            verdict = bal < SATURATED_MIN
            lines.append(f"tab{ti}({tag}): 无货架数据, 余额 {bal} 兜底  "
                         f"{'刷' if verdict else '停刷'}")
        else:
            verdict = True
            lines.append(f"tab{ti}({tag}): 无台账  默认继续刷(宁多刷不漏刷)")
        if verdict and stage:
            stages.append(stage)

    cfg = load("data/app_config.json", {})
    prof = (cfg.get("profiles") or {}).get(cfg.get("active_profile", "default"), {})
    manual = prof.get("event_farm_stages") or []
    # 一致性只比推断器管辖的关(盒抽门票关用户手配, 不置评)
    ruled = {stage_of(t) for t in managed if stage_of(t)}
    manual_cmp = [s for s in manual if s in ruled] if ruled else None
    agree = (sorted(set(stages)) == sorted(set(manual_cmp))
             if (manual and manual_cmp is not None and ruled) else None)
    return {"suggested_stages": stages, "manual_stages": manual,
            "agree": agree, "ledger_lines": lines,
            "n_tabs": n_tabs, "max_stage": max_stage}


if __name__ == "__main__":
    r = plan()
    if "--json" in sys.argv:
        print(json.dumps(r, ensure_ascii=False))
    else:
        print("== 活动选关账单 (建议模式, 不接管) ==")
        print(f"  币种 {r['n_tabs'] or '?'} 个(末位=盒抽币不管) / "
              f"最大关号 {r['max_stage'] or '?(待event_quest落账)'}")
        for ln in r["ledger_lines"]:
            print(" ", ln)
        print(f"建议 farm_stages = {r['suggested_stages'] or '(关号未知, 见上)'}")
        print(f"人工 farm_stages = {r['manual_stages']}"
              + (f"   一致性: {'一致 ' if r['agree'] else '不一致 '}"
                 if r["agree"] is not None else "  (未配置或关号未知, 不置评)"))
