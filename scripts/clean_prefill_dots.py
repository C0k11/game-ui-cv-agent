# -*- coding: utf-8 -*-
"""HSV-clean 红点/黄点 prefill labels in a pool (break the position prior).

The ui model fires 红点/黄点 at LEARNED ENTRY-BADGE POSITIONS even when the
badge is a GREY empty placeholder (user 2026-06-13: "完全是记位置去了根本没在
识别颜色和形状"). Validate every dot label by the colour actually under it:
  red%   = red-hue, high-sat, bright pixel fraction in the box core
  yellow%= yellow-hue, high-sat, bright fraction
Verdict per 红点/黄点 label:
  matches its colour (>FRAC)      → KEEP
  matches the OTHER colour        → FLIP (红点↔黄点)
  neither (grey/blue/empty badge) → DELETE   ← the position-prior false dots
Deleting the false dots ALSO turns those grey squares into NEGATIVES, which is
exactly what v10 needs to learn "no coloured dot here → no label".

Backup: data/_dotclean_backup/<pool>/ (first touch). Multiprocess over frames.

Usage: py scripts/clean_prefill_dots.py <pool> [pool2 ...]
"""
import shutil
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

RAW = Path(r"D:\Project\ai game secretary\data\raw_images")
BACKUP = Path(r"D:\Project\ai game secretary\data\_dotclean_backup")
MASTER = [l.strip() for l in open(RAW / "_classes.txt", encoding="utf-8") if l.strip()]
RED, YEL = MASTER.index("红点"), MASTER.index("黄点")
FRAC = 0.08   # ≥8% coloured pixels in the core ⇒ that colour is really there


def _is_empty(img, xc, yc, bw, bh):
    """True ONLY for a genuinely EMPTY/grey placeholder square — <2% of the box
    core has saturated pixels. A real coloured dot (even dark-red, V~60) has its
    saturated disc, so it can NEVER read empty. This is the only SAFE auto-delete:
    eyeball (2026-06-13) showed dark/pink-red real dots get mis-flagged by any
    hue/brightness FRACTION threshold, so we delete on ABSENCE-of-colour only.
    Blue / ambiguous false dots are LEFT for the human dashboard review."""
    import cv2
    h, w = img.shape[:2]
    x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
    x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
    mx, my = max(1, (x2 - x1) // 6), max(1, (y2 - y1) // 6)
    crop = img[y1 + my:y2 - my, x1 + mx:x2 - mx]
    if crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat_frac = float(((hsv[..., 1] > 70) & (hsv[..., 2] > 40)).mean())
    return sat_frac < 0.02


def process_pool(pool):
    import cv2  # noqa
    pdir = RAW / pool
    deletes = keeps = touched = flips = 0
    for txt in sorted(pdir.glob("*.txt")):
        if txt.name == "classes.txt":
            continue
        lines = [l for l in txt.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not any(int(l.split()[0]) in (RED, YEL)
                   for l in lines if len(l.split()) == 5):
            continue
        img = cv2.imread(str(txt.with_suffix(".jpg")))
        if img is None:
            continue
        out = []
        changed = False
        for l in lines:
            p = l.split()
            if len(p) != 5 or int(p[0]) not in (RED, YEL):
                out.append(l)
                continue
            xc, yc, bw, bh = map(float, p[1:])
            if _is_empty(img, xc, yc, bw, bh):
                deletes += 1
                changed = True
                continue   # empty grey square → drop (becomes a negative)
            keeps += 1
            out.append(l)
        if changed:
            bdir = BACKUP / pool
            bdir.mkdir(parents=True, exist_ok=True)
            bak = bdir / txt.name
            if not bak.exists():
                shutil.copy2(txt, bak)
            txt.write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")
            touched += 1
    return pool, keeps, flips, deletes, touched


def main():
    pools = sys.argv[1:]
    if not pools:
        print("usage: clean_prefill_dots.py <pool> [pool2 ...]")
        return
    tot = Counter()
    with ProcessPoolExecutor(max_workers=16) as ex:
        for pool, k, f, d, t in ex.map(process_pool, pools):
            print(f"{pool}: keep={k} flip={f} delete={d} ({t} files)")
            tot["keep"] += k
            tot["flip"] += f
            tot["delete"] += d
    print(f"\n[done] keep={tot['keep']} flip={tot['flip']} delete={tot['delete']}")


if __name__ == "__main__":
    main()
