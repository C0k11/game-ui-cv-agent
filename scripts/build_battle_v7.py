# -*- coding: utf-8 -*-
"""Build battle_v7 dataset (2026-07-12, v5六池 + 赫赛德池(球boss+2k敌方小怪手标)).

新增 vs v4:
  axis_碧蓝档案_大决战_33_耶罗尼姆斯_* 551帧(451有框) — **第一份真实视频域**
  (B站1080p60压缩录屏/暗色夜战/UP攻略条遮挡), 用户人审身份类+HUD半自动补标
  (fix_axis_bishop_pool.py)。新类 主教(master 480) → local 14, nc=15。
  主教仅标可见帧(268), 被大招CG/全屏VFX遮死的帧不标=教模型全遮挡不输出。
训练配套: 26s @ imgsz 960 (v4=26n@640; 密集血条/暗帧弱检出主因是容量+分辨率)。
"""
import random
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

RAW = Path(r"D:\Project\ai game secretary\data\raw_images")
SRCS = [RAW / n for n in [
    "run_battle_material_20260708",
    "run_20260710_110430", "run_20260710_104718",
    "run_20260710_110759", "run_20260710_104427",
    # v5 增量: 凹轴真实视频域 (下载→抽帧→v4预标→人审→半自动补标)
    "axis_碧蓝档案_大决战_33_耶罗尼姆斯_作业考古合集_p02_2_重甲_水局4010w_BV1KNNc64EEf_p2",
    # v6 增量: 赫赛德池(用户人审: 球boss新类167框 + 敌方浮游炮2083框手标 +
    #  空帧84已删; 并框拆分器已过)
    "axis_碧蓝档案_大决战_28_赫赛德_作业考古合集_p08_8_弹甲_4003w_BV19XFNzHEup_p8",
    # v7 增量: 白&黑池(黑白单类155框(用户104+模板高置信51) + HUD v13重写 +
    #  日文暂停键全域零标注 + 星野泡泡改回我方)
    "axis_碧蓝档案_大决战_32_白_黑_作业考古合集_p02_2_特甲_妹爱黑子3984w_BV1PtLn6zEF4_p2",
]]
OUT = Path(r"D:\Project\ml_cache\models\yolo\dataset\battle_v7")
REMAP = {476: 0, 477: 1, 128: 2, 129: 3, 130: 4, 134: 5, 136: 6,
         478: 7, 479: 8, 412: 9, 135: 10, 131: 11, 132: 12, 133: 13,
         480: 14, 481: 15, 482: 16}
NAMES = ["我方", "敌方", "战斗暂停", "战斗三倍速", "自动战斗开启", "自动战斗关闭",
         "战斗胜利", "塞特的愤怒", "Boss", "战斗1倍速", "战斗2倍速",
         "重新开始键", "继续键", "放弃键", "主教", "球", "黑白"]
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
    cnt_tr = {i: 0 for i in range(len(NAMES))}
    cnt_va = {i: 0 for i in range(len(NAMES))}
    for split, items in splits.items():
        (OUT / "images" / split).mkdir(parents=True)
        (OUT / "labels" / split).mkdir(parents=True)
        for img, lines in items:
            stem = f"{img.parent.name[:40]}__{img.stem}"   # 池间同名防冲突
            shutil.copyfile(img, OUT / "images" / split / (stem + ".jpg"))
            (OUT / "labels" / split / (stem + ".txt")).write_text(
                "\n".join(lines) + "\n", encoding="utf-8")
            cnt = cnt_tr if split == "train" else cnt_va
            for ln in lines:
                cnt[int(ln.split()[0])] += 1
    yaml = [f"path: {OUT.as_posix()}", "train: images/train",
            "val: images/val", f"nc: {len(NAMES)}", "names:"]
    yaml += [f"  {i}: {n}" for i, n in enumerate(NAMES)]
    (OUT / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    print(f"train={len(splits['train'])} val={len(splits['val'])}")
    for i, n in enumerate(NAMES):
        print(f"  {i:2d} {n}: train {cnt_tr[i]} / val {cnt_va[i]}")


if __name__ == "__main__":
    main()
