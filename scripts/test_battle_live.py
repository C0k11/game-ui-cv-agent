# -*- coding: utf-8 -*-
"""实战战斗感知综合实测 (2026-07-15, combat 2.0 首次 live 验证).

战斗进行时跑: 连抓 ADB 干净帧 → battle v9(registry 最新, 全 18 类) +
fused_avatar(技能卡区 y>0.72) 双模型逐帧检出 → jsonl 记录 + 抽样可视化。
产物: data/_battle_live_test/<stamp>/  (frames jsonl + vis*.jpg + 干净帧)

用法: py scripts/test_battle_live.py [秒数, 默认120]
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\Project\ai game secretary")
from mumu_runner import AdbInput  # noqa: E402

ROOT = Path(r"D:\Project\ai game secretary")


def load_models():
    reg = json.loads((ROOT / "data" / "model_registry.json")
                     .read_text(encoding="utf-8"))
    import re
    from ultralytics import YOLO
    vers = reg["battle_heads"]["versions"]
    vn = max((v for v in vers if re.fullmatch(r"v\d+", v)),
             key=lambda x: int(x[1:]))
    battle = YOLO(vers[vn]["path"])
    fav = reg["fused_avatar"]
    avatar = YOLO(fav["versions"][fav["active"]]["path"])
    print(f"battle={vn} avatar={fav['active']}")
    return battle, avatar


def main():
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 120
    iv = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0   # 0=无节流测极限
    battle, avatar = load_models()
    adb = AdbInput()
    adb.connect()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = ROOT / "data" / "_battle_live_test" / stamp
    out.mkdir(parents=True)
    log = (out / "dets.jsonl").open("w", encoding="utf-8")

    t_end = time.time() + dur
    fi = 0
    bat_cnt, ava_cnt = Counter(), Counter()
    while time.time() < t_end:
        t0 = time.time()
        fr = adb.capture_frame()
        if fr is None:
            continue
        H, W = fr.shape[:2]
        rec = {"i": fi, "t": round(t0, 2), "battle": [], "cards": []}
        rb = battle.predict(fr, conf=0.30, imgsz=960, verbose=False)[0]
        for b in (rb.boxes or []):
            name = battle.names[int(b.cls[0])]
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            rec["battle"].append([name, round(float(b.conf[0]), 3),
                                  round((x1 + x2) / 2 / W, 4),
                                  round((y1 + y2) / 2 / H, 4)])
            bat_cnt[name] += 1
        ra = avatar.predict(fr, conf=0.25, imgsz=960, verbose=False)[0]
        for b in (ra.boxes or []):
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            cy = (y1 + y2) / 2 / H
            if cy < 0.72:
                continue
            name = avatar.names[int(b.cls[0])]
            rec["cards"].append([name, round(float(b.conf[0]), 3),
                                 round((x1 + x2) / 2 / W, 4), round(cy, 4)])
            ava_cnt[name] += 1
        log.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if fi % 15 == 0:
            log.flush()
            cv2.imencode(".jpg", cv2.resize(fr, (1920, 1080)))[1].tofile(
                str(out / f"raw_{fi:04d}.jpg"))
            print(f"[{fi}] battle={rec['battle'][:4]} cards={rec['cards']}",
                  flush=True)
        fi += 1
        dt = time.time() - t0
        if dt < iv:
            time.sleep(iv - dt)
    log.close()
    print(f"\n== {fi} 帧汇总 ==")
    print("battle:", dict(bat_cnt.most_common(12)))
    print("cards:", dict(ava_cnt.most_common(10)))
    print("out →", out)


if __name__ == "__main__":
    main()
