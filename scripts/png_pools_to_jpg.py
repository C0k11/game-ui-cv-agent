# -*- coding: utf-8 -*-
"""把**已标注但存成 png**的池转成 jpg —— 否则整条工具链看不见它们。

⛔背景（2026-08-12 结案）：`runner._save` / step `_fly()` 历史上把**干净帧写成
   png**、把标注渲染图写成 jpg（格式正好反了）。而 `build_ui_v2.frames_in()`
   和前端 datasets 都按 jpg 认素材 ⇒ 这些池**标注全在、却一帧都进不了训练集**，
   前端下拉框里也不出现（用户 2026-08-12 就是这么发现的：
   「其余几个都没选中」「之前给了很多 sample 出图难道没有保存」）。
⭐ txt 与图**同名不同扩展名**，所以只转图片、标注原样不动即可。
⛔ 只转**有对应 .txt** 的 png（没标注的转过去只会污染池子的"已标注率"）。
"""
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import cv2
import numpy as np

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "raw_images")
POOLS = ["v2step_20260810", "v2step_20260811", "v2walk_20260811",
         "v2alt_20260811", "v2alt_tabs_20260811", "v2grid_20260811",
         "v2main_20260811", "v2alt_story_20260811", "v2gridmap_20260811",
         "v2tabs_20260811"]

tot_c = tot_s = 0
for p in POOLS:
    d = os.path.join(RAW, p)
    if not os.path.isdir(d):
        continue
    conv = skip = noy = 0
    for f in sorted(os.listdir(d)):
        if not f.lower().endswith(".png") or f.startswith("_"):
            continue
        stem = os.path.splitext(f)[0]
        if not os.path.exists(os.path.join(d, stem + ".txt")):
            noy += 1                      # 没标注的不转
            continue
        dst = os.path.join(d, stem + ".jpg")
        if os.path.exists(dst):
            skip += 1
            continue
        im = cv2.imdecode(np.fromfile(os.path.join(d, f), np.uint8),
                          cv2.IMREAD_COLOR)
        if im is None:
            continue
        cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(dst)
        conv += 1
    jpg = len([f for f in os.listdir(d) if f.endswith(".jpg")])
    txt = len([f for f in os.listdir(d)
               if f.endswith(".txt") and f != "classes.txt"])
    tot_c += conv
    tot_s += skip
    print(f"{p:<24} 转 {conv:4d}（已有 {skip}，无标注跳过 {noy}）"
          f"  →  jpg {jpg:4d} / 标注 {txt:4d}")
print(f"\n✅ 共转出 {tot_c} 张（{tot_s} 张已存在）")
