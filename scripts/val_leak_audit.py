# -*- coding: utf-8 -*-
"""建完集之后对账 val 有没有泄漏, 并且能切出一把**干净尺子**。

为什么要有这个工具(2026-08-25):
  v19 训到 ep30 报 mAP50 0.9706, v18 是 0.9201。涨太多, 一查:
  **val 里 40.6% 的帧和 train 共享 dHash, 抽样拆出约 5% 是逐像素同一张。**
  `build_ui_v2` 只按 **md5** 去重, 而 `_arrow_boost` 这类源装的是**重编码副本**,
  md5 完全不同 -> 挡不住(v17 就栽过一次: "md5 挡不住重编码副本")。
  build 那头已经补了按**文件名源前缀**的硬闸(砍掉 289 帧), 但那只覆盖
  "副本文件名带源 run 前缀"这一种。跨天重录、重新合并出来的同图它抓不到,
  所以还需要这把按**图像内容**对账的尺子。

判据分两级(不要只看 dHash):
  dHash 64bit 基于 8x8 梯度, **同一页面只差一个时钟数字的两帧会完全相同**,
  所以 dHash 撞上只说明"同一个视图", 不等于"同一个文件"。再用缩到 1/8 分辨率的
  逐像素 MAE 拆:
     MAE < HARD(默认 1.0)  -> **硬泄漏**, 基本就是同一张, 必须从 val 剔除
     HARD..SOFT(默认 12)   -> 同页面不同时刻, 静态 UI 天生如此, 留着
     > SOFT                -> dHash 撞车但不是一个东西, 留着

用法:
  py -X utf8 scripts/val_leak_audit.py                 # 只对账
  py -X utf8 scripts/val_leak_audit.py --emit-clean    # 另外切一份干净 val 数据集
  --workers N  默认 4(训练在跑的时候别调高, 会跟 dataloader 抢 CPU)

注意: `--emit-clean` **只往新目录写硬链接, 绝不动 ui_v2 本体** -- 训练可能正在读它。
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

DS = Path(r"D:/Project/ml_cache/models/yolo/dataset/ui_v2")
CACHE = Path(r"D:/Project/ml_cache/models/yolo/dataset/_dhash_cache.json")


def _dhash(path: str):
    import cv2
    import numpy as np
    g = cv2.imdecode(np.fromfile(path, dtype=np.uint8),
                     cv2.IMREAD_REDUCED_GRAYSCALE_8)
    if g is None:
        return None
    s = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    v = 0
    for b in (s[:, 1:] > s[:, :-1]).flatten():
        v = (v << 1) | int(b)
    return v


def _job(a):
    p, tag = a
    return tag, Path(p).name, _dhash(p)


def _mae(a: str, b: str):
    import cv2
    import numpy as np
    ga = cv2.imdecode(np.fromfile(a, dtype=np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_8)
    gb = cv2.imdecode(np.fromfile(b, dtype=np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_8)
    if ga is None or gb is None:
        return None
    if ga.shape != gb.shape:
        gb = cv2.resize(gb, (ga.shape[1], ga.shape[0]))
    return float(np.abs(ga.astype(np.int16) - gb.astype(np.int16)).mean())


def _maejob(a):
    v, t, vp, tp = a
    return v, t, _mae(vp, tp)


def hashes(workers: int, refresh: bool):
    tr = sorted((DS / "images" / "train").glob("*.jpg"))
    va = sorted((DS / "images" / "val").glob("*.jpg"))
    print("DENOM train %d / val %d" % (len(tr), len(va)), flush=True)
    assert tr and va
    if CACHE.is_file() and not refresh:
        d = json.loads(CACHE.read_text(encoding="utf-8"))
        if d.get("ntrain") == len(tr) and d.get("nval") == len(va):
            print("[cache] 用缓存的 dHash", CACHE.name, flush=True)
            return ({k: int(v) for k, v in d["train"].items()},
                    {k: int(v) for k, v in d["val"].items()})
        print("[cache] 数量对不上, 重算", flush=True)
    th, vh = {}, {}
    jobs = [(str(p), "t") for p in tr] + [(str(p), "v") for p in va]
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for tag, name, h in ex.map(_job, jobs, chunksize=64):
            done += 1
            if h is not None:
                (th if tag == "t" else vh)[name] = h
            if done % 10000 == 0:
                print("  hashed %d/%d" % (done, len(jobs)), flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps({"ntrain": len(tr), "nval": len(va),
                                 "train": {k: str(v) for k, v in th.items()},
                                 "val": {k: str(v) for k, v in vh.items()}}),
                     encoding="utf-8")
    print("[cache] 写了", CACHE, flush=True)
    return th, vh


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--hard", type=float, default=1.0)
    ap.add_argument("--soft", type=float, default=12.0)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--emit-clean", action="store_true",
                    help="另切一份剔掉硬泄漏的 val 数据集(只写新目录)")
    args = ap.parse_args()

    th, vh = hashes(args.workers, args.refresh)
    by = defaultdict(list)
    for n, h in th.items():
        by[h].append(n)
    shared = [(n, by[h][0]) for n, h in vh.items() if h in by]
    print("\n== dHash 撞上 train 的 val 帧: %d / %d = %.1f%%"
          % (len(shared), len(vh), 100 * len(shared) / len(vh)))

    print("== 逐像素 MAE 拆(全量, 每帧只比一个 train 对手) ...", flush=True)
    jobs = [(v, t, str(DS / "images" / "val" / v), str(DS / "images" / "train" / t))
            for v, t in shared]
    hard, soft, far = [], 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for v, t, m in ex.map(_maejob, jobs, chunksize=32):
            if m is None:
                continue
            if m < args.hard:
                hard.append((v, t, m))
            elif m <= args.soft:
                soft += 1
            else:
                far += 1
    print("   MAE<%.1f  **硬泄漏(同一张)**   %5d  占 val %.2f%%"
          % (args.hard, len(hard), 100 * len(hard) / len(vh)))
    print("   %.1f-%.1f  同页不同时刻        %5d  占 val %.2f%%"
          % (args.hard, args.soft, soft, 100 * soft / len(vh)))
    print("   >%.1f     dHash 撞车非同物     %5d" % (args.soft, far))

    if hard:
        print("\n硬泄漏的 val 帧来自哪些源(前 12):")
        for k, v in Counter(x[0].split("__")[0] for x in hard).most_common(12):
            print("   %-46s x%d" % (k, v))
        print("对应的 train 侧来自哪些源(前 12):")
        for k, v in Counter(x[1].split("__")[0] for x in hard).most_common(12):
            print("   %-46s x%d" % (k, v))

    if not args.emit_clean:
        print("\n(只对账; 要切干净 val 加 --emit-clean)")
        return 0

    drop = {v for v, _, _ in hard}
    out = DS.parent / "ui_v2_valclean"
    iv, lv = out / "images" / "val", out / "labels" / "val"
    for d in (iv, lv):
        d.mkdir(parents=True, exist_ok=True)
    kept = 0
    for p in sorted((DS / "images" / "val").glob("*.jpg")):
        if p.name in drop:
            continue
        t = DS / "labels" / "val" / (p.stem + ".txt")
        if not t.is_file():
            continue
        for src, dst in ((p, iv / p.name), (t, lv / t.name)):
            if not dst.exists():
                try:
                    os.link(src, dst)
                except OSError:
                    dst.write_bytes(src.read_bytes())
        kept += 1
    y = (DS / "data.yaml").read_text(encoding="utf-8")
    y = y.replace("path: %s" % DS.as_posix(), "path: %s" % out.as_posix())
    y = y.replace("train: images/train", "train: images/val")
    (out / "data.yaml").write_text(y, encoding="utf-8")
    print("\n干净 val 已切: %s  (%d 帧, 剔掉 %d 个硬泄漏)" % (out, kept, len(drop)))
    print("用法: yolo val model=<权重> data=%s" % (out / "data.yaml").as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
