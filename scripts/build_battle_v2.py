# -*- coding: utf-8 -*-
"""Build battle_v2 dataset from run_battle_material_20260708 (user-labelled).

战斗模型 v2 (yolo26n, 高频战斗循环专用) — 7 类:
  0 我方 / 1 敌方 (新身份类, 用户 2026-07-08 标注 509/176 框)
  2 战斗暂停 / 3 战斗三倍速 / 4 自动战斗开启 / 5 自动战斗关闭 / 6 战斗胜利
  (HUD 类与 ui 模型重复是有意的: 战斗高频循环单模型拿到 AUTO gate/胜利,
   不用等 5s 主 tick 的 ui 26m。)
cls 18(体力, 1 框) 丢弃 — 顶栏读数走 digitOCR 不走 battle 模型。

Split: 85/15 by frame (seed 42)。133 帧小数据, aug 靠训练配方。
"""
import random
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

SRC = Path(r"D:\Project\ai game secretary\data\raw_images\run_battle_material_20260708")
OUT = Path(r"D:\Project\ml_cache\models\yolo\dataset\battle_v2")

REMAP = {476: 0, 477: 1, 128: 2, 129: 3, 130: 4, 134: 5, 136: 6}
NAMES = ["我方", "敌方", "战斗暂停", "战斗三倍速",
         "自动战斗开启", "自动战斗关闭", "战斗胜利"]
VAL_FRAC = 0.15
SEED = 42


def main() -> None:
    pairs = []
    for txt in sorted(SRC.glob("*.txt")):
        if txt.name == "classes.txt":
            continue
        img = txt.with_suffix(".jpg")
        if not img.exists():
            img = txt.with_suffix(".png")
        if not img.exists():
            print(f"skip (no image): {txt.name}")
            continue
        lines = []
        for raw in txt.read_text(encoding="utf-8").splitlines():
            p = raw.split()
            if len(p) < 5:
                continue
            c = int(p[0])
            if c not in REMAP:
                continue
            lines.append(" ".join([str(REMAP[c])] + p[1:5]))
        if lines:
            pairs.append((img, lines))

    random.Random(SEED).shuffle(pairs)
    n_val = max(1, int(len(pairs) * VAL_FRAC))
    splits = {"val": pairs[:n_val], "train": pairs[n_val:]}

    if OUT.exists():
        shutil.rmtree(OUT)
    box_cnt = {i: 0 for i in range(len(NAMES))}
    for split, items in splits.items():
        (OUT / "images" / split).mkdir(parents=True)
        (OUT / "labels" / split).mkdir(parents=True)
        for img, lines in items:
            shutil.copyfile(img, OUT / "images" / split / img.name)
            (OUT / "labels" / split / (img.stem + ".txt")).write_text(
                "\n".join(lines) + "\n", encoding="utf-8")
            for ln in lines:
                box_cnt[int(ln.split()[0])] += 1

    yaml = [f"path: {OUT.as_posix()}", "train: images/train", "val: images/val",
            f"nc: {len(NAMES)}", "names:"]
    yaml += [f"  {i}: {n}" for i, n in enumerate(NAMES)]
    (OUT / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")

    print(f"train={len(splits['train'])} val={len(splits['val'])} frames")
    for i, n in enumerate(NAMES):
        print(f"  {i} {n}: {box_cnt[i]} boxes")
    print(f"→ {OUT / 'data.yaml'}")


if __name__ == "__main__":
    main()
