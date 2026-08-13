# -*- coding: utf-8 -*-
"""把 BAAH 的 grid_solution 原始 JSON 转换成我们走格子 flow 的答案格式。

用法: py -X utf8 scripts/convert_baah_grid.py
输入: data/baah_grid_solution/*.json  (BAAH 原始, MIT, 只读不改)
输出: data/grid_answers/{stage}.json  (我们的 schema, grid.load_answer 只认这个)

BAAH 原始格式 (官方 DATA/grid_solution/grid_solution_format.json 为准):
  数字键(规范里叫 solN) = **按目标(3star/challenge/gift)区分的备选解法**,
  不是多区域顺序打 -- 2026-08-13 之前 grid.load_answer 把它当"区域"逐个打,
  29 个多解法文件会在打完第一个解法后干等第二次部署, 这是个真 bug。
  requires[解法键] = 该解法能达成的目标; 我们这份拉取里 269 个解法全是 ["any"]。
  一份解法 = initial_teams(部署) + fight_plan(逐回合逐队动作)。

转换规则:
  1. 主解法 = 队伍属性全为 any 的解法里键最小的; 没有全 any 的取键最小的。
     源文件键序即作者优先序; requires 全 any 没有更多信号, 不自造启发式。
  2. 丢 click 像素字段 -- 方向语义由本帧检出的格子 + 相对角度落地。
  3. 改名消歧: initial_teams.type -> attr (属性要求, blue=神秘/red=爆发/
     yellow=贯穿/purple=振动/any=任意, 和 task_type 的 normal/hard 是两回事);
     position -> pos (部署方位 8 向, 和移动方向 6 向不是一套词汇);
     fight_plan -> rounds, action -> do, target -> dir。
  4. needs 汇总能力需求(teams/portal/exchange/attrs), flow 进关前预检,
     不够格就 BLOCKED 在花 AP 之前, 不进关走到一半才死。
  5. 其余备选解法原样收进 alts, 换路线时人工切换。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "baah_grid_solution"
DST = ROOT / "data" / "grid_answers"

META_KEYS = {"task_location", "task_level", "task_type", "requires"}


def convert_solution(sol: dict) -> dict:
    teams = [{"name": t.get("name", "A"),
              "attr": t.get("type", "any"),
              "pos": t.get("position", "center")}
             for t in sol.get("initial_teams", [])]
    rounds = [[{"team": m.get("team", "A"),
                "do": m.get("action", "move"),
                "dir": m.get("target", "")}
               for m in rnd]
              for rnd in sol.get("fight_plan", [])]
    acts = [m["do"] for rnd in rounds for m in rnd]
    return {
        "teams": teams,
        "rounds": rounds,
        "needs": {
            "teams": len(teams),
            "portal": "portal" in acts,
            "exchange": "exchange" in acts,
            "attrs": sorted({t["attr"] for t in teams} - {"any"}),
        },
    }


def pick_primary(sol_keys: list) -> str:
    """全 any 的解法里键最小的; 没有就键最小的。键按数值排序。"""
    def _num(k):
        return int(k) if k.isdigit() else 10 ** 9
    return sorted(sol_keys, key=_num)[0]


def convert_file(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    stage = path.stem
    sol_keys = [k for k in d if k not in META_KEYS]
    converted = {k: convert_solution(d[k]) for k in sol_keys}
    all_any = [k for k in sol_keys
               if not converted[k]["needs"]["attrs"]]
    primary = pick_primary(all_any or sol_keys) if sol_keys else None
    out = {
        "stage": stage,
        "type": d.get("task_type", "normal"),
        "source": "BAAH (MIT), sol=" + (primary if primary else "none"),
    }
    if primary is None:
        # 1-1 这类: 元数据在但没有解法 = 那关不用走位
        out.update({"teams": [], "rounds": [],
                    "needs": {"teams": 0, "portal": False,
                              "exchange": False, "attrs": []}})
        return out
    out.update(converted[primary])
    alts = [dict(converted[k], sol=k) for k in sol_keys if k != primary]
    if alts:
        out["alts"] = alts
    return out


def stage_order(stage: str):
    """游戏推进序: normal 全部在前, hard 在后; 章内按 章-关。"""
    hard = stage.startswith("H")
    a, b = stage.lstrip("H").split("-")
    return (1 if hard else 0, int(a), int(b))


def main() -> int:
    files = sorted(SRC.glob("*.json"), key=lambda p: stage_order(p.stem))
    if not files:
        print(f"没有输入文件: {SRC}")
        return 1
    DST.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in files:
        out = convert_file(p)
        (DST / p.name).write_text(
            json.dumps(out, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8")
        rows.append(out)

    total = len(rows)
    multi = [r["stage"] for r in rows if r["needs"]["teams"] > 1]
    portal = [r["stage"] for r in rows if r["needs"]["portal"]]
    exchg = [r["stage"] for r in rows if r["needs"]["exchange"]]
    attrs = [r["stage"] for r in rows if r["needs"]["attrs"]]
    single_move_only = [r["stage"] for r in rows
                        if r["needs"]["teams"] <= 1
                        and not r["needs"]["portal"]
                        and not r["needs"]["exchange"]]
    print(f"converted {total} -> {DST}")
    print(f"单队纯move(现有 flow 能力内): {len(single_move_only)} 关")
    for name, lst in (("多队", multi), ("portal", portal),
                      ("exchange", exchg), ("要求属性队", attrs)):
        head = " ".join(lst[:8]) + (" ..." if len(lst) > 8 else "")
        first = lst[0] if lst else "-"
        print(f"{name}: {len(lst)} 关, 最早出现 {first} | {head}")
    # 能力内的推进边界: 按游戏序第一个走不了的关
    for kind in ("normal", "hard"):
        seq = [r for r in rows if r["type"] == kind]
        stop = next((r["stage"] for r in seq
                     if r["stage"] not in single_move_only), None)
        print(f"{kind} 推进边界: 第一个能力外的关 = {stop or '无(全部能力内)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
