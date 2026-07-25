# -*- coding: utf-8 -*-
"""battle_v10s —— **标注助手**数据集(不是 v11 正式模型)。

## 为什么单独建这个
用户 2026-07-25: "可以先拿我人工审核的升级一波模型然后再去给那几个标注啊,
都是黄色机器人"。—— 对的, 而且比我原来那套 cx 位置规则干净:
  · v10 的敌方 87% 来自赫赛德池的黑白机器人, botplay 的黄色量産型机甲**没见过**
    → 先验"战场小人=我方" → 敌方→我方 22.5% / 反向 0.0%
  · 用户人审的 025638(54 帧含 battle 框)里 我方164 : 敌方**226** —— 比例是**反的**,
    正是纠偏信号
  · 其余 6 池是**同一 session、同一张图、同一种黄机甲** → 只要模型在这个域上学会,
    泛化距离几乎为零

## ⚠三个必须说清的设计取舍
1. **过采样是必须的, 不是调参**: 54 帧只占 v10 train 2001 帧的 2.7%, 直接混进去
   敌方比例仅 4.72→4.33:1, 动不了先验。这里 ×SEED_REPEAT 复制。
   代价 = 对这 54 帧过拟合。**对标注助手这是可接受甚至想要的**(目标域完全相同),
   但**绝不能拿它当 v11 上线** —— 正式模型必须等 6 池人审完一起重训。
2. **val 用时间切分, 不随机切**: 素材是连续 tick 帧, 随机切 = 相邻帧泄漏。
   实测 battle_v10 的 val **96.0% 有 ±1 帧邻居在 train**(±2 内 99.4%),
   它的 val mAP 测的是"记没记住相邻帧" —— 这正是"指标好看上 botplay 就崩"的
   度量学解释。这里种子按帧号 < SEED_SPLIT 进 train, >= 进 val, 零重叠。
3. **val 只放种子 holdout**: 我要测的就是"黄机甲还认不认成我方", 不是综合 mAP。
   其余 18 类的回归另跑(拿 v10 的 val 单独评), 不混进这个量尺。
"""
import random
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

RAW = Path(r"D:\Project\ai game secretary\data\raw_images")
SEED_POOL = RAW / "run_20260715_025638_botplay_clean"   # ⭐用户人审, 唯一 GT
BASE_SRCS = [RAW / n for n in [
    "run_battle_material_20260708",
    "run_20260710_110430", "run_20260710_104718",
    "run_20260710_110759", "run_20260710_104427",
    "axis_碧蓝档案_大决战_33_耶罗尼姆斯_作业考古合集_p02_2_重甲_水局4010w_BV1KNNc64EEf_p2",
    "axis_碧蓝档案_大决战_28_赫赛德_作业考古合集_p08_8_弹甲_4003w_BV19XFNzHEup_p8",
    "axis_碧蓝档案_大决战_32_白_黑_作业考古合集_p02_2_特甲_妹爱黑子3984w_BV1PtLn6zEF4_p2",
    "axis_碧蓝档案_大决战_27_薇娜_作业考古合集_p05_5_弹甲_国家队3949w_BV1giiYBeELr_p5",
    "defeat_candidates_v10",
]]
OUT = Path(r"D:\Project\ml_cache\models\yolo\dataset\battle_v10s")
REMAP = {476: 0, 477: 1, 128: 2, 129: 3, 130: 4, 134: 5, 136: 6,
         478: 7, 479: 8, 412: 9, 135: 10, 131: 11, 132: 12, 133: 13,
         480: 14, 481: 15, 482: 16, 483: 17, 484: 18}
NAMES = ["我方", "敌方", "战斗暂停", "战斗三倍速", "自动战斗开启", "自动战斗关闭",
         "战斗胜利", "塞特的愤怒", "Boss", "战斗1倍速", "战斗2倍速",
         "重新开始键", "继续键", "放弃键", "主教", "球", "黑白", "大蛇", "战斗失败"]

SEED_SPLIT = 48      # 种子帧号 <48 进 train, >=48 进 val(时间切, 防相邻帧泄漏)
SEED_REPEAT = 8      # 种子在 train 里复制几份(把 2.7% 抬到 ~18%)
SEED = 42


def load(txt: Path):
    lines = []
    for raw in txt.read_text(encoding="utf-8").splitlines():
        p = raw.split()
        if len(p) >= 5 and int(p[0]) in REMAP:
            lines.append(" ".join([str(REMAP[int(p[0])])] + p[1:5]))
    return lines


def frame_idx(stem: str) -> int:
    import re
    m = re.search(r"(\d+)$", stem)
    return int(m.group(1)) if m else -1


def main() -> None:
    base = []
    for src in BASE_SRCS:
        for txt in sorted(src.glob("*.txt")):
            if txt.name == "classes.txt" or not txt.with_suffix(".jpg").exists():
                continue
            ln = load(txt)
            if ln:
                base.append((txt.with_suffix(".jpg"), ln))

    seed_tr, seed_va = [], []
    for txt in sorted(SEED_POOL.glob("*.txt")):
        if txt.name == "classes.txt" or not txt.with_suffix(".jpg").exists():
            continue
        ln = load(txt)
        if not ln:
            continue
        (seed_tr if frame_idx(txt.stem) < SEED_SPLIT else seed_va).append(
            (txt.with_suffix(".jpg"), ln))

    rng = random.Random(SEED)
    rng.shuffle(base)
    train = base + seed_tr * SEED_REPEAT
    rng.shuffle(train)

    if OUT.exists():
        shutil.rmtree(OUT)
    cnt = {"train": {i: 0 for i in range(len(NAMES))},
           "val": {i: 0 for i in range(len(NAMES))}}
    for split, items in (("train", train), ("val", seed_va)):
        (OUT / "images" / split).mkdir(parents=True)
        (OUT / "labels" / split).mkdir(parents=True)
        seen = {}
        for img, lines in items:
            stem = f"{img.parent.name[:40]}__{img.stem}"
            n = seen.get(stem, 0)
            seen[stem] = n + 1
            if n:                      # 过采样副本要唯一文件名
                stem = f"{stem}__r{n}"
            shutil.copyfile(img, OUT / "images" / split / (stem + ".jpg"))
            (OUT / "labels" / split / (stem + ".txt")).write_text(
                "\n".join(lines) + "\n", encoding="utf-8")
            for ln in lines:
                cnt[split][int(ln.split()[0])] += 1

    yaml = [f"path: {OUT.as_posix()}", "train: images/train",
            "val: images/val", f"nc: {len(NAMES)}", "names:"]
    yaml += [f"  {i}: {n}" for i, n in enumerate(NAMES)]
    (OUT / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")

    print(f"train {len(train)} 帧 (base {len(base)} + 种子 {len(seed_tr)}×{SEED_REPEAT})")
    print(f"val   {len(seed_va)} 帧 (种子 holdout, 帧号 >= {SEED_SPLIT}, 与 train 零重叠)")
    a, e = cnt["train"][0], cnt["train"][1]
    print(f"\ntrain 我方 {a} : 敌方 {e} = {a/max(e,1):.2f}:1  "
          f"(v10 原为 9826:2082 = 4.72:1)")
    for i, n in enumerate(NAMES):
        if cnt["train"][i] or cnt["val"][i]:
            print(f"  {i:2d} {n}: train {cnt['train'][i]} / val {cnt['val'][i]}")


if __name__ == "__main__":
    main()
