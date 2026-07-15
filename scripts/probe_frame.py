# -*- coding: utf-8 -*-
"""抓一帧当前模拟器画面 → data/_probe_now.jpg (960x540 缩略) + 可选 4K 原图。
用法: py scripts/probe_frame.py [--full out.png]"""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, r"D:\Project\ai game secretary")
from mumu_runner import AdbInput  # noqa: E402

adb = AdbInput()
adb.connect()
fr = adb.capture_frame()
if fr is None:
    sys.exit("capture failed")
root = Path(r"D:\Project\ai game secretary")
cv2.imencode(".jpg", cv2.resize(fr, (960, 540)))[1].tofile(
    str(root / "data" / "_probe_now.jpg"))
if "--full" in sys.argv:
    out = sys.argv[sys.argv.index("--full") + 1]
    cv2.imencode(".png", fr)[1].tofile(out)
    print("full →", out)
print("probe saved", fr.shape)
