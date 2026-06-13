# -*- coding: utf-8 -*-
"""关卡得星_0/_3 label repair (2026-06-12, user-spotted teacher disease).

Disease (live evidence frame_000007 @ run_20260612_191319 + user's Area-17
dashboard shot): star boxes drift onto / above the stage NUMBER ("29-2")
instead of the ☆☆☆ row, and _0/_3 class can be flipped vs the actual
star colors. Plain OCR-in-box finds nothing (the drifted box covers blank
card), geometry is normal-sized — so detection needs CONTEXT + CONTENT:

  Validation is CONTENT-ONLY, v4 — thresholds measured on 16 eyeballed
  samples + 14 cls84 samples (run_20260612_191319, separations 10-20x):
    yellow fill (18<=H<=35, S>120, V>150) > 0.04 → real ★★★ → class 84
      (real ★★★ ≈ 0.28)
    grey star fill (S<60, 150<V<228)      > 0.25 → real ☆☆☆ → class 83
      (real ☆☆☆ ≈ 0.61; polluted number-top boxes ≈ 0.03)
    else → DELETE (drifted box on number-top sliver / blank card)
  Failed detectors for the record: in-box OCR (slivers read as nothing),
  above-strip OCR (cropped digit glyphs flaky), dark-ratio/Canny (number-top
  slivers are mostly white + thin strokes — fooled both ways).

Fix policy: DELETE drifted boxes, REWRITE class on mismatch, everything
else untouched. Backup: data/_starfix_backup/<pool>/.

Usage: py scripts/fix_star_labels.py [pool ...]   (default: all active pools)
"""
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, r"D:\Project\ai game secretary")
sys.stdout.reconfigure(encoding="utf-8")

RAW = Path(r"D:\Project\ai game secretary\data\raw_images")
BACKUP = Path(r"D:\Project\ai game secretary\data\_starfix_backup")
S0, S3 = 83, 84  # 关卡得星_0 / 关卡得星_3
SKIP = {"_synth_ui_swap", "_synth_bond", "_synth_bond_enter", "_synth_bond_goto",
        "_fused_synth_remap", "run_20260606_flywheel_labels_bak", "_v8queue_meta"}


def yellow_ratio(img, x1, y1, x2, y2):
    import cv2
    crop = img[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 0] >= 18) & (hsv[..., 0] <= 35)
         & (hsv[..., 1] > 120) & (hsv[..., 2] > 150))
    return float(m.mean())


def verdict(img, x1, y1, x2, y2):
    """→ ('keep', cls) | ('del', reason)."""
    import cv2
    crop = img[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return ("del", "empty")
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    yellow = float(((hsv[..., 0] >= 18) & (hsv[..., 0] <= 35)
                    & (hsv[..., 1] > 120) & (hsv[..., 2] > 150)).mean())
    if yellow > 0.04:
        return ("keep", S3)
    grey = float(((hsv[..., 1] < 60) & (hsv[..., 2] > 150)
                  & (hsv[..., 2] < 228)).mean())
    if grey > 0.25:
        return ("keep", S0)
    return ("del", "no-stars")


def main():
    import cv2

    pools = sys.argv[1:] or [p.name for p in RAW.iterdir()
                             if p.is_dir() and p.name not in SKIP]
    deleted = Counter()
    reclassed = Counter()
    files_touched = 0
    for pool in pools:
        pdir = RAW / pool
        if not pdir.is_dir():
            continue
        for txt in sorted(pdir.glob("*.txt")):
            if txt.name == "classes.txt":
                continue
            raw_lines = [l for l in txt.read_text(encoding="utf-8").splitlines()
                         if l.strip()]
            star_idx = [i for i, l in enumerate(raw_lines)
                        if l.split()[0] in (str(S0), str(S3))]
            if not star_idx:
                continue
            img = cv2.imread(str(txt.with_suffix(".jpg")))
            if img is None:
                continue
            h, w = img.shape[:2]
            out = list(raw_lines)
            changed = False
            for i in star_idx:
                p = raw_lines[i].split()
                cls = int(p[0])
                xc, yc, bw, bh = map(float, p[1:])
                x1, y1 = int((xc - bw/2) * w), int((yc - bh/2) * h)
                x2, y2 = int((xc + bw/2) * w), int((yc + bh/2) * h)
                kind, val = verdict(img, x1, y1, x2, y2)
                if kind == "del":
                    out[i] = None
                    deleted[pool] += 1
                    changed = True
                elif val != cls:
                    out[i] = f"{val} " + " ".join(p[1:])
                    reclassed[pool] += 1
                    changed = True
            if changed:
                bdir = BACKUP / pool
                bdir.mkdir(parents=True, exist_ok=True)
                bak = bdir / txt.name
                if not bak.exists():
                    shutil.copy2(txt, bak)
                txt.write_text("\n".join(l for l in out if l is not None) + "\n",
                               encoding="utf-8")
                files_touched += 1
        if deleted[pool] or reclassed[pool]:
            print(f"{pool}: deleted={deleted[pool]} reclassed={reclassed[pool]}",
                  flush=True)
    print(f"\n[done] deleted {sum(deleted.values())} drifted boxes, "
          f"reclassed {sum(reclassed.values())}, files touched {files_touched}")


if __name__ == "__main__":
    main()
