# -*- coding: utf-8 -*-
"""把 v15 之后攒下的飞轮帧收拢成**一个可标注的池**。

⭐用户 2026-08-12:「收集好从上次训练完到现在的所有数据飞轮，然后预标，
   然后把小号那些，还有很多就 build up 起来训练」。

⛔三个必须踩准的坑（全是今天/历史实锤过的）:
 1. **`_ann.jpg` 不是干净帧** —— runner 每次存帧写两份:
    `NNNN_page.png`（干净）+ `NNNN_page_ann.jpg`（**烧了标注框的渲染图**）。
    把 ann 当素材训进去 = 教模型认自己画的框。⇒ 一律排除。
 2. **png 在这套工具链里等于不存在** —— `build_ui_v2.frames_in()` 历史上只读
    `*.jpg`，前端 datasets 也只按 jpg 计数（[[v16_dataset_integration]] 实锤过
    三批素材"接了源却一帧没进集"）。⇒ 干净的 png 必须转成 jpg。
 3. **按内容 md5 去重** —— runner 一轮存几十帧，相邻帧大量重复；
    [[v16_dataset_integration]]「数唯一帧不数框」。

用法:
    python scripts/collect_flywheel.py            # 只统计，不动文件
    python scripts/collect_flywheel.py --go       # 真的收拢
"""
import hashlib
import io
import os
import shutil
import sys
import time
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import cv2
import numpy as np

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "raw_images")
CUT = time.mktime(time.strptime("2026-08-07", "%Y-%m-%d"))   # v15 上线日
OUT = os.path.join(RAW, "flywheel_v16_20260812")
# 已经人审过 / 已单独成池的，不要再卷进来
KEEP_OUT = {"flywheel_v16_20260812"}


def src_pools():
    for d in sorted(os.listdir(RAW)):
        dp = os.path.join(RAW, d)
        if not os.path.isdir(dp) or d.startswith("_") or d in KEEP_OUT:
            continue
        # 只收 runner 自动存帧池（v2_YYYYMMDD_HHMMSS）和 run_*_clean
        if not (d.startswith("v2_2026") or d.startswith("run_2026")):
            continue
        files = [f for f in os.listdir(dp)
                 if f.lower().endswith((".png", ".jpg"))
                 and not f.startswith("_")
                 and "_ann." not in f]          # ⛔坑①：排除标注渲染图
        if not files:
            continue
        newest = max(os.path.getmtime(os.path.join(dp, f)) for f in files)
        if newest < CUT:
            continue
        yield d, dp, files


def main():
    go = "--go" in sys.argv
    stat = Counter()
    seen = {}
    pools = list(src_pools())
    print(f"候选池 {len(pools)} 个（已排除 _ann 渲染图、只收 v15 之后的）\n")
    if go:
        os.makedirs(OUT, exist_ok=True)
    for d, dp, files in pools:
        for f in files:
            p = os.path.join(dp, f)
            # 已经有同名 .txt 标注的，说明这池在人审流程里，别动
            if os.path.exists(os.path.splitext(p)[0] + ".txt"):
                stat["跳过(已标注)"] += 1
                continue
            try:
                buf = np.fromfile(p, np.uint8)
                h = hashlib.md5(buf.tobytes()).hexdigest()
            except Exception:
                stat["读不了"] += 1
                continue
            if h in seen:                       # ⛔坑③：内容去重
                stat["重复帧"] += 1
                continue
            seen[h] = p
            stat["唯一帧"] += 1
            if not go:
                continue
            im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if im is None:
                stat["解码失败"] += 1
                continue
            # ⛔坑②：统一落成 jpg，否则整条工具链看不见它
            dst = os.path.join(OUT, f"{d[-13:]}__{os.path.splitext(f)[0]}.jpg")
            cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(dst)
            stat["已写出"] += 1
    for k, v in stat.most_common():
        print(f"  {k:<14} {v}")
    if go:
        shutil.copyfile(os.path.join(RAW, "_classes.txt"),
                        os.path.join(OUT, "classes.txt"))
        n = len([f for f in os.listdir(OUT) if f.endswith(".jpg")])
        print(f"\n✅ 收拢到 {OUT}\n   共 {n} 张唯一干净帧（jpg），等预标")
    else:
        print("\n（只统计，没动文件；加 --go 真的收拢）")


if __name__ == "__main__":
    main()
