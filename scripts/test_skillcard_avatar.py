# -*- coding: utf-8 -*-
"""战斗中 EX 技能卡头像识别稳定性实测 (2026-07-14, combat 2.0 前置).

战斗进行时跑: 连续抓 N 帧(ADB 干净帧) → fused_avatar(registry active)@960
→ 底部技能卡区域(y>0.72)检出 → 按 cx 聚类成卡位 → 统计每卡位的类别稳定性
(翻转率/conf 分布/漏检率)。历史基线: EX 技能卡 recall 0.854 (2026-06-11)。

用法: py scripts/test_skillcard_avatar.py [n_frames] [interval_s]
产物: data/_skillcard_test/report.txt + 标注可视化帧
"""
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\Project\ai game secretary")
from mumu_runner import AdbInput  # noqa: E402

ROOT = Path(r"D:\Project\ai game secretary")
OUT = ROOT / "data" / "_skillcard_test"
CARD_Y = 0.72          # 技能卡区域: 画面底部
CONF = 0.25


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    iv = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
    reg = json.loads((ROOT / "data" / "model_registry.json")
                     .read_text(encoding="utf-8"))
    fav = reg["fused_avatar"]
    weights = fav["versions"][fav["active"]]["path"]
    print(f"fused_avatar {fav['active']} = {weights}")
    from ultralytics import YOLO
    model = YOLO(weights)

    adb = AdbInput()
    adb.connect()
    OUT.mkdir(parents=True, exist_ok=True)

    frames_dets = []            # [ [(name, conf, cx, cy)] ]
    for i in range(n):
        t0 = time.time()
        fr = adb.capture_frame()
        if fr is None:
            print(f"  frame {i}: capture failed")
            frames_dets.append([])
            continue
        H, W = fr.shape[:2]
        r = model.predict(fr, conf=CONF, imgsz=960, verbose=False)[0]
        dets = []
        vis = None
        for b in (r.boxes or []):
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            cy = (y1 + y2) / 2 / H
            if cy < CARD_Y:
                continue
            name = model.names[int(b.cls[0])]
            dets.append((name, float(b.conf[0]), (x1 + x2) / 2 / W, cy))
        frames_dets.append(dets)
        if i < 3:                # 头几帧存可视化
            vis = fr.copy()
            for name, conf, cx, cy in dets:
                px, py = int(cx * W), int(cy * H)
                cv2.circle(vis, (px, py), 40, (0, 255, 0), 4)
                cv2.putText(vis, f"{conf:.2f}", (px - 40, py - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.imencode(".jpg", cv2.resize(vis, (1920, 1080)))[1].tofile(
                str(OUT / f"vis_{i}.jpg"))
        print(f"  frame {i}: {len(dets)} card dets "
              f"{[(nm, round(c, 2)) for nm, c, *_ in dets]}")
        dt = time.time() - t0
        if dt < iv:
            time.sleep(iv - dt)

    # ── 卡位聚类(cx 量化到 0.04 桶)+ 稳定性 ──
    slots = defaultdict(list)   # cx_bucket -> [name]
    for dets in frames_dets:
        for name, conf, cx, cy in dets:
            slots[round(cx / 0.04)].append((name, conf))
    n_nonempty = sum(1 for d in frames_dets if d)
    print(f"\n== 稳定性 ({n} 帧, {n_nonempty} 帧有检出) ==")
    lines = [f"frames={n} nonempty={n_nonempty}"]
    for bucket in sorted(slots):
        obs = slots[bucket]
        c = Counter(nm for nm, _ in obs)
        main_name, n_main = c.most_common(1)[0]
        confs = [cf for _, cf in obs]
        line = (f"卡位 cx≈{bucket * 0.04:.2f}: 主={main_name} "
                f"purity {n_main}/{len(obs)} "
                f"conf p50={sorted(confs)[len(confs) // 2]:.2f} "
                f"检出率 {len(obs)}/{n}")
        print(line)
        lines.append(line)
    (OUT / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n产物 → {OUT}")


if __name__ == "__main__":
    main()
