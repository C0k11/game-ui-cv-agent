# -*- coding: utf-8 -*-
"""Build battle_v4 dataset (2026-07-11, v3三池 + 用户审定两新池).

Sources (用户指定, 110759/104427 不纳入 — 留给 v3 预标做下代素材):
  run_battle_material_20260708  133帧 活动关(遊戲開發部) — v2 老底 + 用户重审(+5 Boss)
  run_20260710_110430           233帧 总力战賽特(用户审+固定物传播)
  run_20260710_104718           132帧 綜合戰術考試(用户删53废帧+标89 Boss)

nc=14: v2 的 0-6 不动, 尾部追加(warm-start/部署 idx 兼容):
  7 塞特的愤怒(总力战当期boss专属) 8 Boss(BOSS图标通用boss)
  9/10 一倍速/二倍速(v2 时无素材) 11/12/13 暂停菜单三键(用户: 战斗模型
  要会点暂停→继续/重开/放弃)。
不带: 弹窗叉叉/确认/取消/加载(样本≤30 且 ui 模型稳, 战斗内低频) 体力/左切换(孤框)。
"""
import random
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

RAW = Path(r"D:\Project\ai game secretary\data\raw_images")
SRCS = [RAW / n for n in ["run_battle_material_20260708",
                          "run_20260710_110430", "run_20260710_104718",
                          # v4 增量(2026-07-11): v3预标→用户人审身份类 +
                          # ui v13重建HUD(fix_battle_ui_labels.py, 双框/错标根治)
                          "run_20260710_110759", "run_20260710_104427"]]
OUT = Path(r"D:\Project\ml_cache\models\yolo\dataset\battle_v4")
REMAP = {476: 0, 477: 1, 128: 2, 129: 3, 130: 4, 134: 5, 136: 6,
         478: 7, 479: 8, 412: 9, 135: 10, 131: 11, 132: 12, 133: 13}
NAMES = ["我方", "敌方", "战斗暂停", "战斗三倍速", "自动战斗开启", "自动战斗关闭",
         "战斗胜利", "塞特的愤怒", "Boss", "战斗1倍速", "战斗2倍速",
         "重新开始键", "继续键", "放弃键"]
VAL_FRAC = 0.15
SEED = 42


def main() -> None:
    pairs = []
    for src in SRCS:
        for txt in sorted(src.glob("*.txt")):
            if txt.name == "classes.txt":
                continue
            img = txt.with_suffix(".jpg")
            if not img.exists():
                continue
            lines = []
            for raw in txt.read_text(encoding="utf-8").splitlines():
                p = raw.split()
                if len(p) >= 5 and int(p[0]) in REMAP:
                    lines.append(" ".join([str(REMAP[int(p[0])])] + p[1:5]))
            if lines:
                pairs.append((img, lines))

    random.Random(SEED).shuffle(pairs)
    n_val = max(1, int(len(pairs) * VAL_FRAC))
    splits = {"val": pairs[:n_val], "train": pairs[n_val:]}
    if OUT.exists():
        shutil.rmtree(OUT)
    cnt = {i: 0 for i in range(len(NAMES))}
    for split, items in splits.items():
        (OUT / "images" / split).mkdir(parents=True)
        (OUT / "labels" / split).mkdir(parents=True)
        for img, lines in items:
            stem = f"{img.parent.name}__{img.stem}"   # 池间同名防冲突
            shutil.copyfile(img, OUT / "images" / split / (stem + ".jpg"))
            (OUT / "labels" / split / (stem + ".txt")).write_text(
                "\n".join(lines) + "\n", encoding="utf-8")
            for ln in lines:
                cnt[int(ln.split()[0])] += 1
    yaml = [f"path: {OUT.as_posix()}", "train: images/train",
            "val: images/val", f"nc: {len(NAMES)}", "names:"]
    yaml += [f"  {i}: {n}" for i, n in enumerate(NAMES)]
    (OUT / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    print(f"train={len(splits['train'])} val={len(splits['val'])}")
    for i, n in enumerate(NAMES):
        print(f"  {i:2d} {n}: {cnt[i]}")


if __name__ == "__main__":
    main()
