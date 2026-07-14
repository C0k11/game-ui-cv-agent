# -*- coding: utf-8 -*-
"""轴表 GT 自动评测 (combat 2.0 tracker Phase 1 打样, 2026-07-14).

轴表(dashboard 凹轴 tab 打点, data/axis_sheets/<stem>.json)= 决策时刻清单:
每行 t 秒该由谁(char)干什么(action)。对"按轴放技能"来说, 每个 t 时刻的
前提是 tracker 还锁着人 — 本脚本逐时刻回放检查:

  该帧我方 GT 框数(人审过的池才有) / tracker 我方轨迹数 /
  开场站位绑定(左→右=编成序)的各 slot 主 tid 是否仍存活且没换人

输出逐行 ✓/✗ + 总锁定率 = 系统级 99% 的验收标准打样。

用法:
  py scripts/axis_sheet_eval.py <pool_substr>                 # 用 data/axis_sheets/ 里同源轴表
  py scripts/axis_sheet_eval.py <pool_substr> --sheet x.json  # 指定轴表(demo)
  py scripts/axis_sheet_eval.py <pool_substr> --fps 3
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))
from track_eval import (build_gt_chains, find_pool, iou_mat,  # noqa: E402
                        load_cache, run_tracker, MATCH_IOU)

SHEET_DIR = Path(r"D:\Project\ai game secretary\data\axis_sheets")
CFG = {"track_buffer": 9, "match_thresh": 0.9}   # = bytetrack_axis3fps.yaml


def main():
    pat = sys.argv[1]
    fps = 3.0
    if "--fps" in sys.argv:
        fps = float(sys.argv[sys.argv.index("--fps") + 1])
    pool = find_pool(pat)
    print(f"pool = {pool.name[:70]}")

    if "--sheet" in sys.argv:
        sheet_f = Path(sys.argv[sys.argv.index("--sheet") + 1])
    else:
        cands = [f for f in SHEET_DIR.glob("*.json")
                 if pat in f.stem]
        if not cands:
            sys.exit(f"data/axis_sheets/ 无匹配轴表(先在 dashboard 凹轴 tab 打点), "
                     f"或用 --sheet 指定")
        sheet_f = cands[0]
    rows = json.loads(sheet_f.read_text(encoding="utf-8"))["rows"]
    print(f"sheet = {sheet_f.name} ({len(rows)} 行)")

    per_frame, n_frames, W, H = load_cache(pool)
    chains = build_gt_chains(pool, n_frames, W, H)
    tf = run_tracker(per_frame, n_frames, CFG, pool=pool)

    # ── 开场站位绑定: slot i(左→右) → 主 tid(与 track_eval --slots 同源) ──
    ally = [c for c in chains
            if Counter(cls for cls, _ in c.values()).most_common(1)[0][0] == 476]
    ally.sort(key=lambda c: min(c))
    first_fi = min(min(c) for c in ally) if ally else 0
    opening = sorted([c for c in ally if min(c) <= first_fi + 6],
                     key=lambda c: c[min(c)][1][0])
    slot_tid = {}
    for si, ch in enumerate(opening, 1):
        tids = []
        for fi, (cls, gbox) in sorted(ch.items()):
            cand = tf.get(fi, [])
            if not cand:
                continue
            m = iou_mat(gbox[None, :], np.array([b for _, _, b in cand]))[0]
            j = int(np.argmax(m))
            if m[j] >= MATCH_IOU:
                tids.append(cand[j][0])
        if tids:
            slot_tid[si] = Counter(tids).most_common(1)[0][0]
    print(f"开场绑定: " + " ".join(f"slot{s}=tid{t}" for s, t in slot_tid.items()))

    # ── 逐决策时刻检查 ──
    ok = 0
    print(f"\n{'t':>7} {'cost':>4} {'char':<10} {'action':<16} "
          f"{'GT我方':>5} {'trk我方':>6} {'slot存活':<14} 判定")
    for r in rows:
        fi = min(max(int(round(r["t"] * fps)), 0), n_frames - 1)
        gt_n = sum(1 for c in chains if fi in c and c[fi][0] == 476)
        cand = tf.get(fi, [])
        trk_ally = [(tid, b) for tid, cls, b in cand if cls == 476]
        alive = {s: t for s, t in slot_tid.items()
                 if any(tid == t for tid, _ in trk_ally)}
        # 打样判定: 该时刻 tracker 我方数 ≥ GT 我方数的 80%, 且开场 slot
        # 至少一半仍锁着(轴表尚无 char↔slot 映射, 有了编成序后收紧成
        # "目标 char 的 slot 必须存活")
        good = (gt_n == 0 or len(trk_ally) >= 0.8 * gt_n) and \
               (not slot_tid or len(alive) * 2 >= len(slot_tid))
        ok += good
        alive_s = ",".join(f"s{s}" for s in alive) or "-"
        print(f"{r['t']:>7.2f} {r['cost']:>4} {r['char']:<10.10} "
              f"{r['action']:<16.16} {gt_n:>5} {len(trk_ally):>6} "
              f"{alive_s:<14} {'✓' if good else '✗'}")
    if rows:
        print(f"\n锁定可用率: {ok}/{len(rows)} = {ok / len(rows):.1%}")


if __name__ == "__main__":
    main()
