# -*- coding: utf-8 -*-
"""对指定池就地预标 + 统计弱/新 cls 的覆盖情况。

用户 2026-08-12:「你要去操刀下**购买_灰色**以及**新加的 cls**」「还有那些**弱 cls**」。
   4,987 张飞轮帧里本来就混着今天 live 反复跑出来的买完态/新场景，
   先用现役模型预标一遍，再看哪些弱 cls 真的被覆盖到、哪些还得单独去采。

纪律:
 · **预标 ≠ 标注** —— 产出必须过前端人审才准进训练集（项目铁律）。
 · 权重跟 **registry 的 active**，别硬编码（老脚本写死 v14，现役已是 v15；
   memory `flywheel: 预标权重v9改v14跟线上模型` 就是为这个改过一次）。
 · **绝不覆盖已有 txt**（可能是人审过的）。
 · 写 **5 列** YOLO txt —— 第 6 列会被当成 OBB angle（[[flywheel_label_import]]）。

用法:
  py scripts/prelabel_pool.py <池名> [--conf 0.35] [--dry]
"""
import io
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import cv2
import numpy as np

from routing_v2.percept import detect                     # noqa: E402

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "raw_images")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    pool = sys.argv[1]
    dry = "--dry" in sys.argv
    conf = 0.35
    if "--conf" in sys.argv:
        conf = float(sys.argv[sys.argv.index("--conf") + 1])
    d = os.path.join(RAW, pool)
    if not os.path.isdir(d):
        sys.exit(f" 没有这个池: {d}")
    M = [l.strip() for l in open(os.path.join(RAW, "_classes.txt"),
                                 encoding="utf-8") if l.strip()]
    IDX = {n: i for i, n in enumerate(M)}
    imgs = sorted(f for f in os.listdir(d)
                  if f.lower().endswith(".jpg") and not f.startswith("_"))
    detect.warm(("ui",))
    hit, per_cls, frames_with = Counter(), Counter(), Counter()
    t0, n_new, n_skip = time.time(), 0, 0
    for i, f in enumerate(imgs):
        t = os.path.join(d, os.path.splitext(f)[0] + ".txt")
        if os.path.exists(t):                 # 绝不覆盖人审过的
            n_skip += 1
            continue
        im = cv2.imdecode(np.fromfile(os.path.join(d, f), np.uint8),
                          cv2.IMREAD_COLOR)
        if im is None:
            continue
        bs = detect.infer(im, ("ui",), conf_override=conf)
        rows, seen = [], set()
        for b in bs:
            j = IDX.get(b.cls)
            if j is None:
                continue
            rows.append("%d %.6f %.6f %.6f %.6f" % (j, b.cx, b.cy, b.w, b.h))
            per_cls[b.cls] += 1
            seen.add(b.cls)
        for c in seen:
            frames_with[c] += 1
        hit[len(rows) > 0] += 1
        if rows and not dry:
            open(t, "w", encoding="utf-8").write("\n".join(rows) + "\n")
            n_new += 1
        if (i + 1) % 500 == 0:
            print(f"  …{i+1}/{len(imgs)}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"\n共 {len(imgs)} 帧：预标 {n_new}，跳过(已有txt) {n_skip}，"
          f"空标 {hit[False]}   用时 {time.time()-t0:.0f}s")
    if not dry:
        import shutil
        shutil.copyfile(os.path.join(RAW, "_classes.txt"),
                        os.path.join(d, "classes.txt"))

    # ── 弱/新 cls 覆盖 ──────────────────────────────────────────────
    try:
        from routing_v2.state import vocab as V
        weak = {c for c, (t_, _) in V.HEALTH.items() if t_ < 100}
    except Exception:
        weak = set()
    new_cls = set(M[495:]) if len(M) > 495 else set()
    print("\n=== 这批帧对**弱 cls**的覆盖（帧数）===")
    got = [(c, frames_with[c]) for c in sorted(weak) if frames_with[c]]
    for c, n in sorted(got, key=lambda x: -x[1]):
        print(f"   {c:<24} {n} 帧")
    print(f"   —— 覆盖到 {len(got)}/{len(weak)} 个弱 cls")
    print("\n=== 对**新 cls(495+)**的覆盖 ===")
    got2 = [(c, frames_with[c]) for c in sorted(new_cls) if frames_with[c]]
    for c, n in sorted(got2, key=lambda x: -x[1]):
        print(f"   {c:<24} {n} 帧")
    print(f"   —— 覆盖到 {len(got2)}/{len(new_cls)} 个新 cls"
          "（新 cls 现役模型没学过，检不出是正常的，要靠人标）")
    print("\n=== 命中最多的 20 个 cls ===")
    for c, n in per_cls.most_common(20):
        print(f"   {c:<24} {n}")


if __name__ == "__main__":
    main()
