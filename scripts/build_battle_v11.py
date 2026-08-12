# -*- coding: utf-8 -*-
"""battle_v11 —— **正式模型**（不是标注助手），第一次有可信 val。

## 和 v10s / v10s2 的根本区别
v10s/v10s2 是**标注助手**：只拿 `025638` 一个种子池 ×8 过采样，
故意过拟合到 botplay 域，用来给其余 6 池打预标。**active 一直是 v10。**
v11 是正式的：那 6 池已经**全部人审完**，连同今天新采的走格子关卡战一起进 train。

## val 的选择（这次终于不是幻觉）
`025638`（59帧/纯人工标）当 **主量尺**，理由三条：
   **标注独立**：它是用户最早**纯人工**标的，没有任何模型预标参与。
     其余 6 池是 v10s2 预标+人审 —— 人审能改错框，但**漏检很难被人发现**
     （不会凭空补出模型没检出的目标） 拿它们当 val 会**高估召回**（标注同源偏差）。
   **模型没见过**：v11 从 **v10** warm（不是 v10s2）。`025638` 在 2026-07-20 被
     用户叫停撤出 v10，所以对 v10 是干净的 holdout。
   零相邻帧泄漏：整池 holdout，和 train 不同 session。
**老的 battle_v10 val(353帧) 不当主量尺** —— 实测 96% 有 ±1 帧邻居在 train，
   它测的是"记没记住邻居"。但**训完单独跑一次**当回归量尺（18 个老类没退步）。

## 已剔除的脏数据（2026-08-11 用户从前端截图里逐个抓出来的）
- **空标帧 48 张**：加载中/编队页，没有战斗对象（`_dropped_empty/`）
- **编队页 2 张**：`024743/frame_00000` 被错标了一个 `放弃键`(133) ——
  而 133 在 REMAP 里**会进训练集**，等于教模型"编队页的学生卡=放弃键"；
  `025638/frame_00000` 标注干净但画面是编队页，同样不属于战斗域（`_dropped_formation/`）
"""
import random
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

RAW = Path(r"D:\Project\ai game secretary\data\raw_images")
VAL_POOL = RAW / "run_20260715_025638_botplay_clean"     # 纯人工标, 主量尺
NEW_SRCS = [RAW / n for n in [
    "run_20260715_024743_botplay_clean",
    "run_20260715_030821_botplay_clean",
    "run_20260715_031051_botplay_clean",
    "run_20260715_031909_botplay_clean",
    "run_20260715_042834_botplay_clean",
    "v2battle_20260811",            # 今天新采: 走格子关卡战(全新域)
]]
BASE_SRCS = [RAW / n for n in [
    "run_battle_material_20260708",
    "run_20260710_110430", "run_20260710_104718",
    "run_20260710_110759", "run_20260710_104427",
    #  大决战录屏四池（总力战域）
    "axis_碧蓝档案_大决战_33_耶罗尼姆斯_作业考古合集_p02_2_重甲_水局4010w_BV1KNNc64EEf_p2",
    "axis_碧蓝档案_大决战_28_赫赛德_作业考古合集_p08_8_弹甲_4003w_BV19XFNzHEup_p8",
    "axis_碧蓝档案_大决战_32_白_黑_作业考古合集_p02_2_特甲_妹爱黑子3984w_BV1PtLn6zEF4_p2",
    "axis_碧蓝档案_大决战_27_薇娜_作业考古合集_p05_5_弹甲_国家队3949w_BV1giiYBeELr_p5",
    "defeat_candidates_v10",
]]
OUT = Path(r"D:\Project\ml_cache\models\yolo\dataset\battle_v11")
REMAP = {476: 0, 477: 1, 128: 2, 129: 3, 130: 4, 134: 5, 136: 6,
         478: 7, 479: 8, 412: 9, 135: 10, 131: 11, 132: 12, 133: 13,
         480: 14, 481: 15, 482: 16, 483: 17, 484: 18}
NAMES = ["我方", "敌方", "战斗暂停", "战斗三倍速", "自动战斗开启", "自动战斗关闭",
         "战斗胜利", "塞特的愤怒", "Boss", "战斗1倍速", "战斗2倍速",
         "重新开始键", "继续键", "放弃键", "主教", "球", "黑白", "大蛇", "战斗失败"]
SEED = 42


def load(txt: Path):
    out = []
    for raw in txt.read_text(encoding="utf-8").splitlines():
        p = raw.split()
        if len(p) >= 5 and int(p[0]) in REMAP:
            out.append(" ".join([str(REMAP[int(p[0])])] + p[1:5]))
    return out


def collect(dirs):
    got = []
    for src in dirs:
        if not src.is_dir():
            print(f"  源缺失: {src.name}")
            continue
        n = 0
        for txt in sorted(src.glob("*.txt")):
            if txt.name == "classes.txt":
                continue
            img = txt.with_suffix(".jpg")
            if not img.exists():
                continue
            ln = load(txt)
            if ln:
                got.append((img, ln))
                n += 1
        print(f"  {src.name[:52]:<52} {n:5d} 帧")
    return got


def main() -> None:
    print("── TRAIN 新增（全部人审）──")
    new = collect(NEW_SRCS)
    print("── TRAIN 基座（v10 原九池, 含大决战四池）──")
    base = collect(BASE_SRCS)
    print("── VAL（纯人工标, 主量尺）──")
    val = collect([VAL_POOL])

    rng = random.Random(SEED)
    train = base + new
    rng.shuffle(train)

    if OUT.exists():
        shutil.rmtree(OUT)
    cnt = {"train": {i: 0 for i in range(len(NAMES))},
           "val": {i: 0 for i in range(len(NAMES))}}
    for split, items in (("train", train), ("val", val)):
        (OUT / "images" / split).mkdir(parents=True)
        (OUT / "labels" / split).mkdir(parents=True)
        seen = {}
        for img, lines in items:
            stem = f"{img.parent.name[:40]}__{img.stem}"
            k = seen.get(stem, 0)
            seen[stem] = k + 1
            if k:
                stem = f"{stem}__r{k}"
            shutil.copyfile(img, OUT / "images" / split / (stem + ".jpg"))
            (OUT / "labels" / split / (stem + ".txt")).write_text(
                "\n".join(lines) + "\n", encoding="utf-8")
            for ln in lines:
                cnt[split][int(ln.split()[0])] += 1

    yaml = [f"path: {OUT.as_posix()}", "train: images/train", "val: images/val",
            f"nc: {len(NAMES)}", "names:"]
    yaml += [f"  {i}: {n}" for i, n in enumerate(NAMES)]
    (OUT / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")

    a, e = cnt["train"][0], cnt["train"][1]
    print(f"\ntrain {len(train)} 帧 = 基座 {len(base)} + 新增 {len(new)}")
    print(f"val   {len(val)} 帧（{VAL_POOL.name}, 纯人工标）")
    print(f"\ntrain 我方 {a} : 敌方 {e} = {a/max(e,1):.2f}:1"
          f"   (v10 4.72 / v10s2 3.05)")
    print(f"{'类':<12}{'train':>8}{'val':>7}")
    for i, n in enumerate(NAMES):
        if cnt["train"][i] or cnt["val"][i]:
            flag = "  val=0" if cnt["train"][i] and not cnt["val"][i] else ""
            print(f"{n:<12}{cnt['train'][i]:>8}{cnt['val'][i]:>7}{flag}")


if __name__ == "__main__":
    main()
