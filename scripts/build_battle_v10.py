# -*- coding: utf-8 -*-
"""Build battle_v10 dataset (2026-07-17, v9九池 + DEFEAT池 + botplay实战7池).

新增 vs v9:
  defeat_candidates_v10 28帧 — 新类 战斗失败(master 484) → local 18, nc=19。
  用户人审 28 框(DEFEAT 红横幅口径); botplay 侧同帧副本已移
  _defeat_dedup_backup_20260717(矛盾标签双重毒防线)。
  run_20260715_*_botplay_clean ×7 (498帧) — combat 2.0 实战飞轮(scrcpy 源):
  battle 域框由 REMAP 过滤自取, ui/头像框自动丢弃(box 级路由)。
战斗胜利+战斗失败均稀缺 → 分层切分保 val ≥2(v9 起的惯例)。
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
    "axis_碧蓝档案_大决战_33_耶罗尼姆斯_作业考古合集_p02_2_重甲_水局4010w_BV1KNNc64EEf_p2",
    "axis_碧蓝档案_大决战_28_赫赛德_作业考古合集_p08_8_弹甲_4003w_BV19XFNzHEup_p8",
    "axis_碧蓝档案_大决战_32_白_黑_作业考古合集_p02_2_特甲_妹爱黑子3984w_BV1PtLn6zEF4_p2",
    "axis_碧蓝档案_大决战_27_薇娜_作业考古合集_p05_5_弹甲_国家队3949w_BV1giiYBeELr_p5",
    # v10 增量: DEFEAT 池(cls484 战斗失败×28, 用户人审✓)
    "defeat_candidates_v10",
    # ⛔botplay×7(498帧)撤出 v10(2026-07-20 用户抓): battle 域框=battle v9 自己预标,
    # 未过人审 — 自蒸馏会固化 v9 系统性误检。待用户审完(flywheel_review 清单②:
    # DEFEAT 口径/瞄准态/技能卡抽查)再进 v11。
    # "run_20260715_024743_botplay_clean",
    # "run_20260715_025638_botplay_clean",
    # "run_20260715_030821_botplay_clean",
    # "run_20260715_031051_botplay_clean",
    # "run_20260715_031909_botplay_clean",
    # "run_20260715_042334_botplay_clean",
    # "run_20260715_042834_botplay_clean",
]]
OUT = Path(r"D:\Project\ml_cache\models\yolo\dataset\battle_v10")
REMAP = {476: 0, 477: 1, 128: 2, 129: 3, 130: 4, 134: 5, 136: 6,
         478: 7, 479: 8, 412: 9, 135: 10, 131: 11, 132: 12, 133: 13,
         480: 14, 481: 15, 482: 16, 483: 17, 484: 18}
NAMES = ["我方", "敌方", "战斗暂停", "战斗三倍速", "自动战斗开启", "自动战斗关闭",
         "战斗胜利", "塞特的愤怒", "Boss", "战斗1倍速", "战斗2倍速",
         "重新开始键", "继续键", "放弃键", "主教", "球", "黑白", "大蛇", "战斗失败"]
VICTORY = 6              # 稀缺类: 分层保 val
DEFEAT = 18              # 同稀缺(28帧), 与胜利同策略
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

    rng = random.Random(SEED)
    vic = [pr for pr in pairs
           if any(int(l.split()[0]) in (VICTORY, DEFEAT) for l in pr[1])]
    rest = [pr for pr in pairs if pr not in vic]
    rng.shuffle(vic)
    rng.shuffle(rest)
    n_val_vic = max(2, int(len(vic) * VAL_FRAC)) if vic else 0
    n_val_rest = max(1, int(len(rest) * VAL_FRAC))
    splits = {"val": vic[:n_val_vic] + rest[:n_val_rest],
              "train": vic[n_val_vic:] + rest[n_val_rest:]}
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
