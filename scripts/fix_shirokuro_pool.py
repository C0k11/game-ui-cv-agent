# -*- coding: utf-8 -*-
"""白&黑池收尾手术 (2026-07-14, 用户三指令).

  A. HUD 重写: 用户 shift+V 批量贴可能弄乱倍速/暂停类 → 删池内全部 HUD 框,
     ui v13 (imgsz960 conf0.35) 重写。**暂停菜单键(131重开/132继续/133放弃)
     一律不写** — 用户拍板: 视频域日文按钮不进训练(实战=繁中服)。
  B. 全 axis 池清日文暂停键残留 (赫赛德 131×6)。
  C. 「黑白」传播: 104/473 战斗帧覆盖=部分标注毒 → 全帧模板搜索(按已标段
     抽模板, 同 fix_axis_bishop_pool 方案), 分析模式看分布定阈值再 --apply。

用法: py scripts/fix_shirokuro_pool.py [--apply] [--bw-th=0.68]
"""
import glob
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"D:\Project\ai game secretary")
from vision.io_utils import imread_any  # noqa: E402
from scripts.yolo_prefill_run import is_dup_box  # noqa: E402

RAW = Path(r"D:\Project\ai game secretary\data\raw_images")
POOL = Path(glob.glob(str(RAW / "axis_*白_黑*"))[0])
UI_WEIGHTS = r"D:\Project\ml_cache\models\yolo\runs\ui_yolo26m_v13\weights\last.pt"
MASTER = [l.strip() for l in open(RAW / "_classes.txt", encoding="utf-8") if l.strip()]
N2I = {n: i for i, n in enumerate(MASTER)}
BW = N2I["黑白"]
HUD_REBUILD = {N2I[n] for n in ("战斗暂停", "战斗三倍速", "自动战斗开启",
                                "自动战斗关闭", "战斗2倍速", "战斗1倍速",
                                "战斗胜利")}
PAUSE_KEYS = {N2I[n] for n in ("重新开始键", "继续键", "放弃键")}
APPLY = "--apply" in sys.argv
BW_TH = float(next((a.split("=")[1] for a in sys.argv
                    if a.startswith("--bw-th=")), 0.68))
SCRATCH = Path(r"C:\Users\shien\AppData\Local\Temp\claude"
               r"\D--Project-ai-game-secretary--claude-worktrees-magical-tharp-fa5d91"
               r"\a4e15e41-e17a-4cd8-8e96-4b51142a5c5a\scratchpad")


def load(t):
    out = []
    for ln in t.read_text(encoding="utf-8").splitlines():
        p = ln.split()
        if len(p) >= 5:
            out.append([int(p[0])] + [float(v) for v in p[1:5]])
    return out


def save(t, boxes):
    t.write_text("\n".join(
        f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}"
        for b in boxes) + ("\n" if boxes else ""), encoding="utf-8")


def crop(img, b):
    h, w = img.shape[:2]
    x1, y1 = max(0, int((b[1] - b[3] / 2) * w)), max(0, int((b[2] - b[4] / 2) * h))
    x2, y2 = min(w, int((b[1] + b[3] / 2) * w)), min(h, int((b[2] + b[4] / 2) * h))
    return img[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None


def main():
    frames = {int(re.search(r"(\d+)", t.stem).group(1)): t
              for t in POOL.glob("*.txt") if t.name != "classes.txt"}
    bak = RAW / "_backups" / (time.strftime("%Y%m%d_%H%M%S") + "_shirokuro")
    if APPLY:
        bak.mkdir(parents=True, exist_ok=True)
        for t in frames.values():
            shutil.copy2(t, bak / t.name)

    # ── A. HUD 重写 ──
    from ultralytics import YOLO
    ui = YOLO(UI_WEIGHTS)
    n_del = n_add = 0
    data = {}
    for n, t in sorted(frames.items()):
        boxes = load(t)
        kept = [b for b in boxes if b[0] not in HUD_REBUILD | PAUSE_KEYS]
        n_del += len(boxes) - len(kept)
        data[n] = kept
    # v13 批量重写(只在有身份框的战斗帧上跑, 空帧/字幕帧不写 HUD)
    jpgs = {n: (POOL / f"frame_{n:05d}.jpg") for n in frames}
    for n in sorted(frames):
        if not any(b[0] in (476, 477, BW) for b in data[n]):
            continue
        img = imread_any(str(jpgs[n]))
        if img is None:
            continue
        H, W = img.shape[:2]
        r = ui.predict(img, imgsz=960, conf=0.35, verbose=False)[0]
        for b in r.boxes:
            nm = ui.names[int(b.cls[0])]
            mi = N2I.get(nm)
            if mi is None or mi not in HUD_REBUILD:
                continue
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            if is_dup_box([x1, y1, x2, y2], mi,
                          (((k[0]), [(k[1] - k[3] / 2) * W, (k[2] - k[4] / 2) * H,
                                     (k[1] + k[3] / 2) * W, (k[2] + k[4] / 2) * H])
                           for k in data[n])):
                continue
            data[n].append([mi, (x1 + x2) / 2 / W, (y1 + y2) / 2 / H,
                            (x2 - x1) / W, (y2 - y1) / H])
            n_add += 1
    print(f"A. HUD: 删 {n_del} → v13 重写 {n_add} (暂停菜单键零写入)")

    # ── B. 其他 axis 池清日文暂停键 ──
    n_purge = 0
    for other in glob.glob(str(RAW / "axis_*")):
        op = Path(other)
        if op == POOL:
            continue
        for t in op.glob("*.txt"):
            if t.name == "classes.txt":
                continue
            boxes = load(t)
            kept = [b for b in boxes if b[0] not in PAUSE_KEYS]
            if len(kept) != len(boxes):
                n_purge += len(boxes) - len(kept)
                if APPLY:
                    shutil.copy2(t, bak / f"other__{t.name}")
                    save(t, kept)
    print(f"B. 其他池日文暂停键清除: {n_purge}")

    # ── C. 黑白全帧模板传播 ──
    labeled = sorted(n for n in frames if any(b[0] == BW for b in data[n]))
    combat = sorted(n for n in frames
                    if any(b[0] in (476, 477, BW) for b in data[n]))
    tpl_src = [(n, next(b for b in data[n] if b[0] == BW)) for n in labeled]
    segs = []
    for n, b in tpl_src:
        if segs and n - segs[-1][-1][0] <= 3:
            segs[-1].append((n, b))
        else:
            segs.append([(n, b)])
    picks = []
    for seg in segs:
        for i in sorted({0, len(seg) // 2, len(seg) - 1}):
            picks.append(seg[i])
    SCALE = 0.5
    tpls = []
    for n, b in picks:
        c = crop(imread_any(str(jpgs[n])), b)
        if c is not None:
            tpls.append((cv2.cvtColor(cv2.resize(c, None, fx=SCALE, fy=SCALE),
                                      cv2.COLOR_BGR2GRAY), b[3], b[4]))
    scores = []
    for n in combat:
        if n in labeled:
            continue
        img = imread_any(str(jpgs[n]))
        gs = cv2.cvtColor(cv2.resize(img, None, fx=SCALE, fy=SCALE),
                          cv2.COLOR_BGR2GRAY)
        best = (-1.0, 0, 0, 0, 0)
        for g, bw_, bh_ in tpls:
            if g.shape[0] >= gs.shape[0] or g.shape[1] >= gs.shape[1]:
                continue
            r = cv2.matchTemplate(gs, g, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(r)
            if mx > best[0]:
                best = (mx, (loc[0] + g.shape[1] / 2) / gs.shape[1],
                        (loc[1] + g.shape[0] / 2) / gs.shape[0], bw_, bh_)
        scores.append((n, round(best[0], 3)) + best[1:])
    arr = np.array([s[1] for s in scores]) if scores else np.array([0.0])
    print(f"C. 黑白传播候选 {len(scores)}帧: p10={np.percentile(arr,10):.2f} "
          f"p50={np.percentile(arr,50):.2f} p90={np.percentile(arr,90):.2f}")
    scores.sort(key=lambda x: x[1])
    print("  最低8:", [(s[0], s[1]) for s in scores[:8]])
    filled = skipped = 0
    if APPLY:
        for n, s, cx, cy, w, h in scores:
            if s >= BW_TH:
                data[n].append([BW, cx, cy, w, h])
                filled += 1
            else:
                skipped += 1
        print(f"  写入 {filled} 跳过 {skipped} (阈值 {BW_TH})")

    if APPLY:
        for n, t in frames.items():
            save(t, data[n])
        print(f"[apply] {len(frames)} 帧写盘, 备份 → {bak}")


if __name__ == "__main__":
    main()
