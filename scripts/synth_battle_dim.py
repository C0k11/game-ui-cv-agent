# -*- coding: utf-8 -*-
"""battle_v3 dim/选中微高亮合成 (2026-07-10, 用户 spec).

模拟卡牌选中目标时的瞄准态: 全图暗化 45-55%, 随机 1-3 个身份目标框
(我方/敌方/塞特/Boss)保持原亮度(边缘羽化防矩形 artifact)。GT 标签不变
(暗态下的 HUD/小人也是有效训练信号 — 真实瞄准态就长这样)。

合成量: train split 的 ~15% (synth 占比过高会让 best.pt 偏拟合合成分布,
fused_avatar v6 教训: synth 63% → 真实场景退化)。只对含身份类框的帧合成。
"""
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

DS = Path(r"D:\Project\ml_cache\models\yolo\dataset\battle_v3")
IDENTITY_CLS = {0, 1, 7, 8}     # 我方/敌方/塞特的愤怒/Boss
FRAC = 0.15
SEED = 1042


def main() -> None:
    img_dir, lbl_dir = DS / "images" / "train", DS / "labels" / "train"
    # 幂等: 清掉旧合成
    for f in list(img_dir.glob("*__dim.jpg")) + list(lbl_dir.glob("*__dim.txt")):
        f.unlink()

    candidates = []
    for lbl in sorted(lbl_dir.glob("*.txt")):
        boxes = []
        for l in lbl.read_text(encoding="utf-8").splitlines():
            p = l.split()
            if len(p) >= 5 and int(p[0]) in IDENTITY_CLS:
                boxes.append(tuple(map(float, p[1:5])))
        if boxes:
            candidates.append((lbl, boxes))

    rng = random.Random(SEED)
    picks = rng.sample(candidates, int(len(candidates) * FRAC))
    for lbl, boxes in picks:
        img_p = img_dir / (lbl.stem + ".jpg")
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        H, W = img.shape[:2]
        dim = rng.uniform(0.45, 0.58)
        # 高亮保持 mask: 随机 1-3 个身份框原亮, 边缘羽化 ~12px
        mask = np.zeros((H, W), np.float32)
        for (x, y, w, h) in rng.sample(boxes, min(len(boxes), rng.randint(1, 3))):
            x1, y1 = int((x - w / 2) * W), int((y - h / 2) * H)
            x2, y2 = int((x + w / 2) * W), int((y + h / 2) * H)
            mask[max(0, y1):y2, max(0, x1):x2] = 1.0
        mask = cv2.GaussianBlur(mask, (25, 25), 0)
        gain = dim + (1.0 - dim) * mask[..., None]
        out = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        cv2.imwrite(str(img_dir / (lbl.stem + "__dim.jpg")), out,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        (lbl_dir / (lbl.stem + "__dim.txt")).write_text(
            lbl.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"synthesized {len(picks)} dim variants "
          f"(train {len(list(img_dir.glob('*.jpg')))} imgs total)")


if __name__ == "__main__":
    main()
