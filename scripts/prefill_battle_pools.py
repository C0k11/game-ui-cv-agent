# -*- coding: utf-8 -*-
"""双模型预标战斗素材池 (2026-07-10, 用户手录总力战对局).

路由(用户 spec):
  - HUD/UI 类(一倍速/二倍速/自动战斗关闭/暂停/暂停菜单 继续键132 重新开始键131
    放弃键133/结算链等) → 通用 ui 模型 v13
  - 我方(476)/敌方(477) 小人 → battle_yolo26n_v2 (只贡献身份类, HUD 检出丢弃)

闸(战斗帧特化):
  - 红点/黄点全丢 — 战斗画面无 UI 入口点, 检出全是位置先验假阳
  - 397获得奖励 尺寸闸 h>0.105 (WIN! 横幅误标, 2026-07-08 定式)
写 5 列 master-idx 标签(绝不 6 列) + master classes.txt(478行 含我方/敌方)。

Usage: py scripts/prefill_battle_pools.py run_20260710_104427 run_20260710_104718 ...
       --force 重标已有 txt 的帧
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\Project\ai game secretary")

RAW = Path(r"D:\Project\ai game secretary\data\raw_images")
UI_W = r"D:\Project\ml_cache\models\yolo\runs\ui_yolo26m_v13\weights\last.pt"
BT_W = r"D:\Project\ml_cache\models\yolo\runs\battle_yolo26n_v2\weights\best.pt"
MASTER = [l.strip() for l in open(RAW / "_classes.txt", encoding="utf-8") if l.strip()]
NAME2IDX = {n: i for i, n in enumerate(MASTER)}

_DROP_UI = {"红点", "黄点"}          # 战斗帧上纯假阳
_BATTLE_KEEP = {"我方", "敌方"}      # battle v2 只贡献身份类
_GOT_REWARD = "获得奖励"             # 397 尺寸闸


def main():
    force = "--force" in sys.argv
    pool_names = [a for a in sys.argv[1:] if not a.startswith("--")]
    pools = [RAW / p for p in pool_names]
    for p in pools:
        if not p.is_dir():
            print(f"pool not found: {p}")
            return

    from ultralytics import YOLO
    ui = YOLO(UI_W)
    bt = YOLO(BT_W)

    for pool in pools:
        jpgs = sorted(pool.glob("*.jpg")) + sorted(pool.glob("*.png"))
        if not jpgs:
            print(f"skip {pool.name}: no images")
            continue
        (pool / "classes.txt").write_text("\n".join(MASTER) + "\n", encoding="utf-8")
        todo = [j for j in jpgs
                if force or not j.with_suffix(".txt").exists()]
        print(f"\n== {pool.name}: {len(jpgs)} imgs, prefill {len(todo)}")
        stats = {}
        empty = 0
        B = 24
        for s in range(0, len(todo), B):
            chunk = [str(j) for j in todo[s:s + B]]
            r_ui = ui.predict(chunk, conf=0.20, imgsz=960, half=True,
                              verbose=False, device=0)
            r_bt = bt.predict(chunk, conf=0.35, imgsz=640, half=True,
                              verbose=False, device=0)
            for path, ru, rb in zip(chunk, r_ui, r_bt):
                lines = []
                for b in ru.boxes:
                    name = ui.names[int(b.cls)]
                    if name in _DROP_UI or name not in NAME2IDX:
                        continue
                    x, y, w, h = b.xywhn[0].tolist()
                    if name == _GOT_REWARD and h > 0.105:
                        continue   # WIN! 横幅假阳
                    lines.append(f"{NAME2IDX[name]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                    stats[name] = stats.get(name, 0) + 1
                for b in rb.boxes:
                    name = bt.names[int(b.cls)]
                    if name not in _BATTLE_KEEP or name not in NAME2IDX:
                        continue
                    x, y, w, h = b.xywhn[0].tolist()
                    lines.append(f"{NAME2IDX[name]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                    stats[name] = stats.get(name, 0) + 1
                if not lines:
                    empty += 1
                Path(path).with_suffix(".txt").write_text(
                    "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        print(f"   empty frames: {empty}")
        for name, n in sorted(stats.items(), key=lambda kv: -kv[1]):
            print(f"   {NAME2IDX[name]:4d} {name}: {n}")


if __name__ == "__main__":
    main()
