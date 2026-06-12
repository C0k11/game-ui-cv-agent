# -*- coding: utf-8 -*-
"""Whole-history label repair pass (2026-06-11, user: "老的数据可能会有轻微的
标记问题或者红点黄点问题，可以的话你一并修复"):

1. 红点(5)/黄点(6) HSV arbitration over EVERY source pool that feeds ui_v2:
   crop the dot core, dominant hue decides red vs yellow. Label disagrees
   with hue → FLIP the class id in place. Ambiguous (low saturation / hue
   outside both bands) → leave untouched (never delete).
2. Schedule Location-Select row order: the row list is canonical
   (夏莱办公室 → 夏莱居住区 → 格黑娜学院中央区 → 阿拜多斯高中 → 千年研究所).
   Within a frame the labeled subset must follow that order top-to-bottom;
   if inverted, reassign the present ids by y-order (fixes the
   夏莱居住区↔夏莱办公室 audit swaps systematically).

Every modified .txt is backed up once to data/_dotfix_backup/<pool>/.
Multiprocess: one worker per pool (7950X3D — fan out wide).
"""
import shutil
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

RAW = Path(r"D:\Project\ai game secretary\data\raw_images")
BACKUP = Path(r"D:\Project\ai game secretary\data\_dotfix_backup")
MASTER = [l.strip() for l in open(RAW / "_classes.txt", encoding="utf-8") if l.strip()]
NAME2M = {n: i for i, n in enumerate(MASTER)}

RED, YELLOW = NAME2M["红点"], NAME2M["黄点"]
ROW_ORDER = [NAME2M[n] for n in
             ("夏莱办公室", "夏莱居住区", "格黑娜学院中央区",
              "阿拜多斯高中", "千年研究所") if n in NAME2M]

POOLS = [
    # current REAL_SOURCES + VAL_SOURCES + tonight's recovered pools
    "run_20260521_103956_distinct", "run_20260527_094158", "run_20260527_101545",
    "run_20260518_002646", "run_20260529_000756", "run_20260529_123209",
    "run_20260531_110516", "run_20260531_143201", "run_20260518_163513",
    "run_20260531_173326", "run_20260531_174456", "run_20260531_175038",
    "run_20260603_134626", "run_20260603_134649", "run_20260603_140153",
    "run_20260603_170116", "run_20260603_175151", "run_20260603_175217",
    "run_20260603_175238", "run_20260603_175257", "run_v6weak_20260603",
    "_emoticon_v2", "run_20260607_193003", "run_20260607_140123",
    "run_20260610_v8queue", "run_20260610_024533", "_arrow_boost",
    "run_20260606_flywheel", "_val_v8flywheel",
    "run_20260611_044844_clean", "run_20260611_050507_clean",
    "run_20260611_051200_clean", "run_20260611_052955_clean",
    "run_20260611_053359_clean", "run_20260611_053709_clean",
    "run_20260611_055934_clean", "run_20260611_061637_clean",
    "run_20260611_061938_clean", "run_20260611_064804_clean",
    "run_20260611_071526_clean", "run_20260611_072242_clean",
    "run_20260611_073139_clean", "run_20260611_073341_clean",
    "run_20260611_074607_clean", "run_20260611_205439",
    "run_20260611_205540", "run_20260611_212919",
    "run_20260603_171121", "_ui_val_pool",
]


def dot_hue(img, box):
    """→ 'red' | 'yellow' | None (ambiguous)."""
    import cv2
    import numpy as np
    h, w = img.shape[:2]
    xc, yc, bw, bh = box
    x1, y1 = int((xc - bw/2) * w), int((yc - bh/2) * h)
    x2, y2 = int((xc + bw/2) * w), int((yc + bh/2) * h)
    mx, my = max(1, (x2-x1)//5), max(1, (y2-y1)//5)
    crop = img[max(0, y1+my):y2-my, max(0, x1+mx):x2-mx]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = (hsv[..., 1] > 90) & (hsv[..., 2] > 120)
    if mask.sum() < 30:
        return None
    hue = float(np.median(hsv[..., 0][mask]))
    if hue <= 14 or hue >= 165:
        return "red"
    if 16 <= hue <= 40:
        return "yellow"
    return None


def process_pool(pool: str):
    import cv2
    pdir = RAW / pool
    flips = Counter()
    row_fixes = 0
    touched = 0
    for txt in sorted(pdir.glob("*.txt")):
        if txt.name == "classes.txt":
            continue
        lines = [l for l in txt.read_text(encoding="utf-8").splitlines() if l.strip()]
        parsed = []
        ok = True
        for ln in lines:
            p = ln.split()
            if len(p) != 5:
                parsed.append((None, ln))
                continue
            try:
                parsed.append(((int(p[0]), tuple(map(float, p[1:]))), ln))
            except ValueError:
                ok = False
                break
        if not ok:
            continue
        img = None
        changed = False

        # 1) dot color arbitration
        for i, (item, ln) in enumerate(parsed):
            if item is None or item[0] not in (RED, YELLOW):
                continue
            if img is None:
                jp = txt.with_suffix(".jpg")
                img = cv2.imread(str(jp)) if jp.exists() else False
            if img is False or img is None:
                continue
            verdict = dot_hue(img, item[1])
            want = RED if verdict == "red" else YELLOW if verdict == "yellow" else None
            if want is not None and want != item[0]:
                parsed[i] = ((want, item[1]),
                             f"{want} " + " ".join(f"{v:.6f}" for v in item[1]))
                flips[f"{MASTER[item[0]]}→{MASTER[want]}"] += 1
                changed = True

        # 2) location-row order repair
        rows = [(i, item) for i, (item, _ln) in enumerate(parsed)
                if item is not None and item[0] in ROW_ORDER]
        if len(rows) >= 2:
            present = sorted({it[0] for _i, it in rows}, key=ROW_ORDER.index)
            by_y = sorted(rows, key=lambda r: r[1][1][1])
            want_seq = [c for c in present for _ in range(1)]
            got_seq = [it[0] for _i, it in by_y]
            if len(by_y) == len(present) and got_seq != want_seq:
                for (i, item), want_cls in zip(by_y, want_seq):
                    if item[0] != want_cls:
                        parsed[i] = ((want_cls, item[1]),
                                     f"{want_cls} " + " ".join(
                                         f"{v:.6f}" for v in item[1]))
                        row_fixes += 1
                        changed = True

        if changed:
            bdir = BACKUP / pool
            bdir.mkdir(parents=True, exist_ok=True)
            bak = bdir / txt.name
            if not bak.exists():
                shutil.copy2(txt, bak)
            txt.write_text("\n".join(ln for _it, ln in parsed) + "\n",
                           encoding="utf-8")
            touched += 1
    return pool, dict(flips), row_fixes, touched


def main():
    total_flips = Counter()
    total_rows = 0
    total_touched = 0
    with ProcessPoolExecutor(max_workers=16) as ex:
        for pool, flips, row_fixes, touched in ex.map(process_pool, POOLS):
            if flips or row_fixes:
                print(f"{pool}: flips={flips} row_fixes={row_fixes} "
                      f"({touched} files)", flush=True)
            total_flips.update(flips)
            total_rows += row_fixes
            total_touched += touched
    print(f"\n[done] dot flips: {dict(total_flips)} | row fixes: {total_rows} "
          f"| files touched: {total_touched}")


if __name__ == "__main__":
    main()
