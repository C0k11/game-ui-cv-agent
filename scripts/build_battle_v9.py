# -*- coding: utf-8 -*-
"""Build battle_v9 dataset (2026-07-14, v8八池 + 大蛇池).

新增 vs v8:
  axis_碧蓝档案_大决战_27_薇娜_* 235帧(空帧101已删) — 新类 大蛇(master 483)
  → local 17, nc=18。用户人审身份类+大蛇204框; 倍速投票毒100框已修正
  (audit_speed_labels.py 排查, track_prefill HUD 已改单帧直写); 并框拆分
  141 过刀; 孤儿补检 163 全驳回(UP 主绿色日文字幕伪装成血条, 目检实锤)。
  ⭐白亮底单▶(战斗1倍速)新形态样本 = 回归用例 speed_1x_bright_bg 的解药。
战斗胜利分层切分: 历史 seed42 随机切分连续 4 代切出 0 val(15train), 本代
含「战斗胜利」帧单独切, 保证 val ≥2 帧。
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
    # v9 增量: 大蛇池(薇娜 大决战#27 弹甲 国家队3949w, 1080p60)
    "axis_碧蓝档案_大决战_27_薇娜_作业考古合集_p05_5_弹甲_国家队3949w_BV1giiYBeELr_p5",
]]
OUT = Path(r"D:\Project\ml_cache\models\yolo\dataset\battle_v9")
REMAP = {476: 0, 477: 1, 128: 2, 129: 3, 130: 4, 134: 5, 136: 6,
         478: 7, 479: 8, 412: 9, 135: 10, 131: 11, 132: 12, 133: 13,
         480: 14, 481: 15, 482: 16, 483: 17}
NAMES = ["我方", "敌方", "战斗暂停", "战斗三倍速", "自动战斗开启", "自动战斗关闭",
         "战斗胜利", "塞特的愤怒", "Boss", "战斗1倍速", "战斗2倍速",
         "重新开始键", "继续键", "放弃键", "主教", "球", "黑白", "大蛇"]
VICTORY = 6              # 稀缺类: 分层保 val
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
           if any(int(l.split()[0]) == VICTORY for l in pr[1])]
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
