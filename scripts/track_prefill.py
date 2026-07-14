# -*- coding: utf-8 -*-
"""tracker 化预标 (2026-07-13, 用户点破: 预标该先吃 tracker 红利, 不用等实战).

单帧 predict 的三种病, ByteTrack + 轨迹后处理逐个治:
  闪烁漏检   → ByteTrack 天生保留"低分但与轨迹匹配"的框(它名字的由来)
  短暂遮挡   → 轨迹 gap ≤3 帧线性插值补框
  类别闪变   → 轨迹级类别投票(一条轨迹 80% 帧是我方 → 全轨迹统一)
  假阳性     → 短命轨迹(<3帧)丢弃

用法:
  py scripts/track_prefill.py <pool_substr>            # 分析: 对比单帧版, 不写盘
  py scripts/track_prefill.py <pool_substr> --apply    # 写盘(备份先行)
只处理 battle 域(身份类+HUD), 与 yolo_prefill_run 同格式(master idx 5列)。
"""
import glob
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\Project\ai game secretary")
from vision.io_utils import imread_any  # noqa: E402

RAW = Path(r"D:\Project\ai game secretary\data\raw_images")
WEIGHTS = r"D:\Project\ml_cache\models\yolo\runs\battle_yolo26s_v6\weights\best.pt"
MASTER = [l.strip() for l in
          open(RAW / "_classes.txt", encoding="utf-8") if l.strip()]
NAME2IDX = {n: i for i, n in enumerate(MASTER)}
MIN_TRACK_LEN = 3
MAX_GAP = 3


def main():
    pat = sys.argv[1]
    apply = "--apply" in sys.argv
    pool = Path([p for p in glob.glob(str(RAW / "axis_*")) if pat in p][0])
    jpgs = sorted(pool.glob("*.jpg"),
                  key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
    print(f"pool={pool.name[:60]} {len(jpgs)}帧")

    from ultralytics import YOLO
    model = YOLO(WEIGHTS)
    n2m = {i: NAME2IDX[n] for i, n in model.names.items() if n in NAME2IDX}

    # ── pass 1: 按帧序 track (conf 放低到 0.10, 让 ByteTrack 捞低分框) ──
    tracks = defaultdict(list)   # tid -> [(fi, mi, conf, xc,yc,w,h)]
    per_frame_plain = []         # 单帧基线: conf 0.35 会保留的框数
    for fi, p in enumerate(jpgs):
        img = imread_any(str(p))
        r = model.track(img, persist=True, conf=0.10, iou=0.5, imgsz=960,
                        tracker="bytetrack.yaml", verbose=False)[0]
        n_plain = 0
        if r.boxes is not None and len(r.boxes):
            H, W = r.orig_shape
            for b in r.boxes:
                mi = n2m.get(int(b.cls[0]))
                if mi is None:
                    continue
                conf = float(b.conf[0])
                if conf >= 0.35:
                    n_plain += 1
                if b.id is None:
                    continue
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                tracks[int(b.id[0])].append(
                    (fi, mi, conf,
                     (x1 + x2) / 2 / W, (y1 + y2) / 2 / H,
                     (x2 - x1) / W, (y2 - y1) / H))
        per_frame_plain.append(n_plain)
        if fi % 100 == 0:
            print(f"  track {fi}/{len(jpgs)}", flush=True)

    # ── pass 2: 轨迹后处理 ──
    out_frames = defaultdict(list)   # fi -> [(mi, xc,yc,w,h)]
    n_short = n_vote = n_interp = n_lowconf_saved = 0
    for tid, obs in tracks.items():
        if len(obs) < MIN_TRACK_LEN:
            n_short += 1
            continue
        vote = Counter(mi for _, mi, *_ in obs).most_common(1)[0][0]
        n_vote += sum(1 for _, mi, *_ in obs if mi != vote)
        n_lowconf_saved += sum(1 for _, _, c, *_ in obs if c < 0.35)
        obs = sorted(obs)
        for i, (fi, _mi, _c, xc, yc, w, h) in enumerate(obs):
            out_frames[fi].append((vote, xc, yc, w, h))
            if i + 1 < len(obs):
                nfi = obs[i + 1][0]
                gap = nfi - fi
                if 1 < gap <= MAX_GAP:      # 空档线性插值
                    nxt = obs[i + 1]
                    for g in range(1, gap):
                        t = g / gap
                        out_frames[fi + g].append((
                            vote,
                            xc + (nxt[3] - xc) * t, yc + (nxt[4] - yc) * t,
                            w + (nxt[5] - w) * t, h + (nxt[6] - h) * t))
                        n_interp += 1

    total_track = sum(len(v) for v in out_frames.values())
    print(f"\n== 对比 ==")
    print(f"单帧版(conf0.35) 总框: {sum(per_frame_plain)}")
    print(f"track版 总框: {total_track}  "
          f"(轨迹捞回低分框 {n_lowconf_saved} + 插值补 {n_interp})")
    print(f"短命轨迹丢弃 {n_short} 条 | 类别投票纠正 {n_vote} 框")

    if apply:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        bak = RAW / "_backups" / f"{stamp}_trackprefill"
        bak.mkdir(parents=True, exist_ok=True)
        for fi, p in enumerate(jpgs):
            lbl = p.with_suffix(".txt")
            if lbl.exists():
                shutil.copy2(lbl, bak / lbl.name)
            lines = [f"{mi} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
                     for mi, xc, yc, w, h in out_frames.get(fi, [])]
            lbl.write_text("\n".join(lines) + ("\n" if lines else ""),
                           encoding="utf-8")
        print(f"[apply] {len(jpgs)} 帧已写, 备份 → {bak}")


if __name__ == "__main__":
    main()
