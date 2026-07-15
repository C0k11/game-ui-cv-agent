# -*- coding: utf-8 -*-
"""事件驱动战斗结算处理器 (2026-07-15, 用户点破"结算页干等浪费时间").

替代盲 sleep: 轮询 ADB 帧跑 battle v9, 「战斗胜利」cls 出现 → 立即点結算
確認链(感知铁律: 目标 cls 出现零延迟点击)。战斗中零干预。

用法: py scripts/wait_battle_end.py [max_wait_s=180]
退出码: 0=结算链走完回到关卡页  1=超时
"""
import json
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, r"D:\Project\ai game secretary")
from mumu_runner import AdbInput  # noqa: E402

ROOT = Path(r"D:\Project\ai game secretary")
# 结算链(4K 实测坐标): 結算確認 → 活动道具中间页確認 → 獲得獎勵確認
SETTLE_TAPS = [(3437, 1998), (1920, 2005), (2318, 1998)]


def main():
    max_wait = float(sys.argv[1]) if len(sys.argv) > 1 else 180
    reg = json.loads((ROOT / "data" / "model_registry.json")
                     .read_text(encoding="utf-8"))
    vers = reg["battle_heads"]["versions"]
    vn = max((v for v in vers if re.fullmatch(r"v\d+", v)),
             key=lambda x: int(x[1:]))
    from ultralytics import YOLO
    model = YOLO(vers[vn]["path"])
    adb = AdbInput()
    adb.connect()

    t0 = time.time()
    while time.time() - t0 < max_wait:
        fr = adb.capture_frame()
        if fr is None:
            continue
        r = model.predict(fr, conf=0.5, imgsz=960, verbose=False)[0]
        names = {model.names[int(b.cls[0])] for b in (r.boxes or [])}
        if "战斗胜利" in names:
            dt = time.time() - t0
            print(f"战斗胜利 detected at {dt:.0f}s → settle chain")
            for x, y in SETTLE_TAPS:
                adb.tap_px(x, y) if hasattr(adb, "tap_px") else \
                    adb._shell(f"input tap {x} {y}")
                time.sleep(6)
            print("settled")
            return 0
    print("timeout waiting for battle end")
    return 1


if __name__ == "__main__":
    sys.exit(main())
