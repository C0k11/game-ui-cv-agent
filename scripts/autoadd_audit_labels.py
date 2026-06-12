# -*- coding: utf-8 -*-
"""Auto-add high-confidence MISSING_LABEL audit findings back into the
SOURCE raw_images label files (add-only, assist_label_weakcls style).

Input: data/_audit_train_labels.csv + data/_audit_val_labels.csv
(produced by audit_train_labels.py — v8 disagreement mining over ui_v2).

Rules:
  - kind == MISSING_LABEL and conf >= 0.85 only
  - class must be in WHITELIST (badge/chrome/currency/strong entries).
    Excluded on purpose: 任务开始/选择购买/购买 (yellow-button confusion),
    Emoticon_Action (FP-on-credit-icon disease), location rows 38/39.
  - 红点/黄点 additionally pass HSV arbitration on the crop — pred color
    must match dominant hue, else the add is skipped (never relabeled).
  - ADD ONLY: re-predict the source frame, append preds that still have no
    same-class GT overlap (IoU>=0.3). Existing lines are never touched.
  - every touched .txt is backed up once to data/_autoadd_backup/<src>/.

Frame name mapping: ui_v2 train stems are '{src}__{stem}[__dN]' — dups
(__dN) collapse onto the same source file and are processed once.
"""
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

RAW = Path(r"D:\Project\ai game secretary\data\raw_images")
BACKUP = Path(r"D:\Project\ai game secretary\data\_autoadd_backup")
CSVS = [Path(r"D:\Project\ai game secretary\data\_audit_train_labels.csv"),
        Path(r"D:\Project\ai game secretary\data\_audit_val_labels.csv")]
WEIGHTS = r"D:\Project\ml_cache\models\yolo\runs\ui_yolo26m_v8b\weights\best_real.pt"

MASTER = [l.strip() for l in open(RAW / "_classes.txt", encoding="utf-8") if l.strip()]
NAME2M = {n: i for i, n in enumerate(MASTER)}

WHITELIST = {
    "红点", "黄点", "绿勾", "体力", "信用点", "青辉石", "加号", "减号",
    "左切换", "右切换", "返回键", "回大厅按钮", "加载中",
    "双倍或三倍活动进行中", "咖啡厅收益", "收藏图标", "邮件箱", "MomoTalk",
    "战术大赛", "咖啡厅入口", "制造入口", "课程表票", "学院交流会票",
    "咖啡厅邀请卷", "弹窗叉叉", "MAX_灰色", "MIN_可点击",
    "学生momotalk信息未读", "学生发送信息中", "房间区域未解锁",
    "任务大厅入口", "获得奖励", "跳过战斗", "战斗暂停", "战斗三倍速",
    "简易攻略", "每日领奖", "距离结束还剩",
}
ADD_CONF = 0.85
POOLS = sorted((p.name for p in RAW.iterdir() if p.is_dir()), key=len,
               reverse=True)  # longest-first so 'a_b' wins over prefix 'a'


def src_of(frame: str):
    stem = frame[:-4] if frame.endswith(".jpg") else frame
    for pool in POOLS:
        if stem.startswith(pool + "__"):
            rest = stem[len(pool) + 2:]
            # strip oversample suffix __dN
            if "__d" in rest:
                head, _, tail = rest.rpartition("__d")
                if tail.isdigit():
                    rest = head
            return pool, rest
    return None, None


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter)


def dot_color_ok(img, box, want: str) -> bool:
    """HSV arbitration for 红点/黄点 — dominant hue must match pred class."""
    import cv2
    import numpy as np
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (int(box[0]*w), int(box[1]*h), int(box[2]*w), int(box[3]*h))
    # center 60% of the box — dot core, skip anti-aliased rim
    mx, my = (x2-x1)//5, (y2-y1)//5
    crop = img[y1+my:y2-my, x1+mx:x2-mx]
    if crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = (hsv[..., 1] > 90) & (hsv[..., 2] > 120)
    if mask.sum() < 30:
        return False
    hue = float(np.median(hsv[..., 0][mask]))
    if want == "红点":
        return hue <= 14 or hue >= 165
    return 16 <= hue <= 40  # 黄点


def main():
    import cv2
    from ultralytics import YOLO

    # findings → unique source frames + the classes flagged there
    targets = {}
    for c in CSVS:
        for r in csv.DictReader(open(c, encoding="utf-8")):
            if r["kind"] != "MISSING_LABEL" or float(r["conf"]) < ADD_CONF:
                continue
            cls = r["gt_or_pred_cls"]
            if cls not in WHITELIST:
                continue
            pool, stem = src_of(r["frame"])
            if pool is None:
                continue
            targets.setdefault((pool, stem), set()).add(NAME2M[cls])
    print(f"{len(targets)} unique source frames to process")

    model = YOLO(WEIGHTS)
    n2m = {i: NAME2M[n] for i, n in model.names.items() if n in NAME2M}

    added = Counter()
    skipped_hsv = Counter()
    missing_file = 0
    keys = sorted(targets)
    B = 24
    for s in range(0, len(keys), B):
        chunk = keys[s:s + B]
        paths = []
        for pool, stem in chunk:
            jp = RAW / pool / (stem + ".jpg")
            paths.append(jp if jp.exists() else None)
        live = [(k, p) for k, p in zip(chunk, paths) if p is not None]
        missing_file += len(chunk) - len(live)
        if not live:
            continue
        results = model.predict([str(p) for _, p in live], conf=0.10,
                                imgsz=960, verbose=False)
        for ((pool, stem), jp), res in zip(live, results):
            txt = RAW / pool / (stem + ".txt")
            gt = []
            lines = []
            if txt.exists():
                lines = [l for l in txt.read_text(encoding="utf-8").splitlines()
                         if l.strip()]
                for ln in lines:
                    p = ln.split()
                    if len(p) != 5:
                        continue
                    c = int(p[0])
                    xc, yc, w, h = map(float, p[1:])
                    gt.append((c, (xc - w/2, yc - h/2, xc + w/2, yc + h/2)))
            H, W = res.orig_shape
            img = None
            new_lines = []
            for b in res.boxes:
                mi = n2m.get(int(b.cls[0]))
                if mi is None or mi not in targets[(pool, stem)]:
                    continue
                if float(b.conf[0]) < ADD_CONF:
                    continue
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                pb = (x1/W, y1/H, x2/W, y2/H)
                if any(c == mi and iou(pb, gb) >= 0.3 for c, gb in gt):
                    continue
                name = MASTER[mi]
                if name in ("红点", "黄点"):
                    if img is None:
                        img = cv2.imread(str(jp))
                    if img is None or not dot_color_ok(img, pb, name):
                        skipped_hsv[name] += 1
                        continue
                xc, yc = (pb[0]+pb[2])/2, (pb[1]+pb[3])/2
                bw, bh = pb[2]-pb[0], pb[3]-pb[1]
                new_lines.append(f"{mi} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                gt.append((mi, pb))  # avoid double-adding overlapping preds
                added[name] += 1
            if new_lines:
                bdir = BACKUP / pool
                bdir.mkdir(parents=True, exist_ok=True)
                bak = bdir / (stem + ".txt")
                if txt.exists() and not bak.exists():
                    shutil.copy2(txt, bak)
                txt.write_text("\n".join(lines + new_lines) + "\n",
                               encoding="utf-8")
        if (s // B) % 10 == 0:
            print(f"  {s + len(chunk)}/{len(keys)} frames, "
                  f"{sum(added.values())} added", flush=True)

    print(f"\n[done] added {sum(added.values())} boxes "
          f"(hsv-rejected: {sum(skipped_hsv.values())}, "
          f"missing files: {missing_file})")
    for n, c in added.most_common():
        print(f"  +{c:4d} {n}")
    if skipped_hsv:
        print("hsv rejects:", dict(skipped_hsv))


if __name__ == "__main__":
    main()
