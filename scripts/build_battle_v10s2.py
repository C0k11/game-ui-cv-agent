# -*- coding: utf-8 -*-
"""battle_v10s2 —— **标注助手 v2**(仍然不上线, 只用来给那 6 池打预标)。

## 和 v10s 的区别(用户 2026-08-11 拍板"这组可以先拿去迭代一波战斗模型")
1. **种子用全 68 帧**, 不再留 holdout。
   v10s 当初切 48 是为了**验证方法成不成立**, 那个问题已经答完了:
   敌方召回 32.4%→85.3% / 敌→友 22.7%→0.0%(0/58)。方法既已证实,
   再扣掉 14 帧 GT 不训就是纯浪费 —— 这 68 帧是**全仓唯一的人审战斗 GT**。
2. **val 换成 battle_v10 的 353 帧**(回归量尺)。
   v10s 的 val 只有 16 帧种子 holdout, 训完即废; 换成 v10 val 才能回答
   "其余 18 类有没有被过采样带崩"这个真问题。
   ⛔无泄漏: botplay×7 在 2026-07-20 就被用户叫停撤出 v10 了(见
     build_battle_v10.py:30-39 那段注释掉的源), 所以 v10 的 val 里
     **没有一帧** 025638。

## ⛔纪律照旧(和 v10s 完全一致, 别因为"升级了"就松)
- **过拟合是设计内的**: 目标域 = 同 session 同一张图的黄色量産型机甲。
  ⇒ **绝不能拿它当 v11 上线**, active 保持 v10。正式 v11 等 6 池人审完重训。
- **真验证不在 val 上**: 这个模型好不好, 唯一口径是"给 6 池打的预标, 人眼抽查
  准不准"。val 只回答"别的类没忘"。
"""
import random
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

RAW = Path(r"D:\Project\ai game secretary\data\raw_images")
SEED_POOL = RAW / "run_20260715_025638_botplay_clean"   # ⭐用户人审, 全仓唯一 GT
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
OUT = Path(r"D:\Project\ml_cache\models\yolo\dataset\battle_v10s2")
VAL_FROM = Path(r"D:\Project\ml_cache\models\yolo\dataset\battle_v10")
REMAP = {476: 0, 477: 1, 128: 2, 129: 3, 130: 4, 134: 5, 136: 6,
         478: 7, 479: 8, 412: 9, 135: 10, 131: 11, 132: 12, 133: 13,
         480: 14, 481: 15, 482: 16, 483: 17, 484: 18}
NAMES = ["我方", "敌方", "战斗暂停", "战斗三倍速", "自动战斗开启", "自动战斗关闭",
         "战斗胜利", "塞特的愤怒", "Boss", "战斗1倍速", "战斗2倍速",
         "重新开始键", "继续键", "放弃键", "主教", "球", "黑白", "大蛇", "战斗失败"]

SEED_REPEAT = 8      # 种子在 train 里复制几份(v10s 实证这个量级能翻转先验)
SEED = 42


def load(txt: Path):
    lines = []
    for raw in txt.read_text(encoding="utf-8").splitlines():
        p = raw.split()
        if len(p) >= 5 and int(p[0]) in REMAP:
            lines.append(" ".join([str(REMAP[int(p[0])])] + p[1:5]))
    return lines


def main() -> None:
    base = []
    for src in BASE_SRCS:
        for txt in sorted(src.glob("*.txt")):
            if txt.name == "classes.txt" or not txt.with_suffix(".jpg").exists():
                continue
            ln = load(txt)
            if ln:
                base.append((txt.with_suffix(".jpg"), ln))

    seed = []
    for txt in sorted(SEED_POOL.glob("*.txt")):
        if txt.name == "classes.txt" or not txt.with_suffix(".jpg").exists():
            continue
        ln = load(txt)
        if ln:
            seed.append((txt.with_suffix(".jpg"), ln))

    rng = random.Random(SEED)
    rng.shuffle(base)
    train = base + seed * SEED_REPEAT
    rng.shuffle(train)

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "images" / "train").mkdir(parents=True)
    (OUT / "labels" / "train").mkdir(parents=True)
    cnt = {i: 0 for i in range(len(NAMES))}
    seen = {}
    for img, lines in train:
        stem = f"{img.parent.name[:40]}__{img.stem}"
        n = seen.get(stem, 0)
        seen[stem] = n + 1
        if n:                      # 过采样副本要唯一文件名
            stem = f"{stem}__r{n}"
        shutil.copyfile(img, OUT / "images" / "train" / (stem + ".jpg"))
        (OUT / "labels" / "train" / (stem + ".txt")).write_text(
            "\n".join(lines) + "\n", encoding="utf-8")
        for ln in lines:
            cnt[int(ln.split()[0])] += 1

    # ⭐val 直接指向 battle_v10 的 val（绝对路径，不复制 353 帧）
    nval = len(list((VAL_FROM / "images" / "val").glob("*.jpg")))
    yaml = [f"train: {(OUT / 'images' / 'train').as_posix()}",
            f"val: {(VAL_FROM / 'images' / 'val').as_posix()}",
            f"nc: {len(NAMES)}", "names:"]
    yaml += [f"  {i}: {n}" for i, n in enumerate(NAMES)]
    (OUT / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")

    a, e = cnt[0], cnt[1]
    print(f"train {len(train)} 帧 = base {len(base)} + 种子 {len(seed)}×{SEED_REPEAT}")
    print(f"val   {nval} 帧 ← battle_v10 的 val（回归量尺，无 025638 泄漏）")
    print(f"\ntrain 我方 {a} : 敌方 {e} = {a/max(e,1):.2f}:1"
          f"   (v10 原 9826:2082 = 4.72:1 / v10s 3.40:1)")
    for i, n in enumerate(NAMES):
        if cnt[i]:
            print(f"  {i:2d} {n}: {cnt[i]}")


if __name__ == "__main__":
    main()
