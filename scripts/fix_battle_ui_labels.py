# -*- coding: utf-8 -*-
"""ui v13 重建两个战斗池的 HUD/UI 类标注 (2026-07-11 用户指令).

背景: battle v3 预标在 HUD 类上输出重叠框(104427: 自动战斗关闭 72 对
IoU>0.6 重复 / 86 帧)+ 漏标错标(v3 HUD 训练样本少)。用户已人审
我方/敌方/Boss/塞特身份类 — 一律不碰。

做法: 每帧跑 ui v13(imgsz960, HUD 类训练样本多), 删掉旧 HUD 类行,
写入 v13 检出(ultralytics NMS 已去重)。其余类(含身份类)原样保留。
执行前全量备份到 data/_backups/。
"""
import glob
import os
import shutil
import sys
import time
from collections import Counter

sys.path.insert(0, r"D:\Project\ai game secretary")

POOLS = [
    r"data/raw_images/run_20260710_110759",
    r"data/raw_images/run_20260710_104427",
]
V13 = r"D:/Project/ml_cache/models/yolo/runs/ui_yolo26m_v13/weights/last.pt"
REBUILD_NAMES = {
    "战斗暂停", "战斗三倍速", "自动战斗开启", "重新开始键", "继续键",
    "放弃键", "自动战斗关闭", "战斗2倍速", "战斗胜利", "战斗1倍速",
}
CONF = 0.35   # HUD 类 v13 检出通常 0.9+; 0.35 防弱假阳污染人审队列


def main() -> None:
    from ultralytics import YOLO
    model = YOLO(V13)
    name_by_idx = model.names  # v13 内部 idx → name

    ts = time.strftime("%H%M%S")
    for pool in POOLS:
        pname = os.path.basename(pool)
        classes = open(os.path.join(pool, "classes.txt"),
                       encoding="utf-8").read().splitlines()
        idx_by_name = {n: i for i, n in enumerate(classes)}
        rebuild_idx = {idx_by_name[n] for n in REBUILD_NAMES if n in idx_by_name}

        # 备份
        bdir = os.path.join("data", "_backups", f"{pname}_uifix_{ts}")
        os.makedirs(bdir, exist_ok=True)
        for f in glob.glob(os.path.join(pool, "frame_*.txt")):
            shutil.copy2(f, bdir)

        imgs = sorted(glob.glob(os.path.join(pool, "frame_*.jpg")))
        removed = Counter()
        added = Counter()
        # 批量推理 (4090, imgsz960)
        results = model.predict(imgs, imgsz=960, conf=CONF, device=0,
                                verbose=False, batch=16, stream=True)
        for img_path, res in zip(imgs, results):
            lbl_path = img_path[:-4] + ".txt"
            old_lines = []
            if os.path.exists(lbl_path):
                old_lines = [l for l in open(lbl_path).read().splitlines()
                             if l.strip()]
            keep = []
            for l in old_lines:
                cls = int(l.split()[0])
                if cls in rebuild_idx:
                    removed[classes[cls]] += 1
                else:
                    keep.append(l)
            new = []
            for b in res.boxes:
                nm = name_by_idx[int(b.cls)]
                if nm not in REBUILD_NAMES or nm not in idx_by_name:
                    continue
                x, y, w, h = b.xywhn[0].tolist()
                new.append(f"{idx_by_name[nm]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                added[nm] += 1
            with open(lbl_path, "w") as f:
                f.write("\n".join(keep + new) + ("\n" if keep or new else ""))
        print(f"== {pname}: {len(imgs)}帧  backup={bdir}")
        print(f"   删旧HUD: {dict(removed)}")
        print(f"   v13重建: {dict(added)}")


if __name__ == "__main__":
    main()
