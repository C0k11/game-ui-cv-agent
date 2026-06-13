# -*- coding: utf-8 -*-
"""Add-only labeler for 批量扫荡(455) on stage_select frames.

v9 live-misses the blue 批量掃蕩 button (only ~127 hand frames + thin
prefill). The button sits at a FIXED spot on stage_select (all frames
3840x2160, same UI scale) → fixed-scale template matching is exact.

Per frame in each pool:
  - skip if a 455 line already exists (add-only, never touch existing)
  - matchTemplate(blue-button template) in the bottom-left window
    (x 0.30-0.55, y 0.72-0.92); score >= 0.85 → append a 455 line at the
    matched center using the hand-label convention box (text-only,
    w=0.0635 h=0.0320 — measured from run_20260610_024533 hand labels)
Backup: data/_sweepbtn_backup/<pool>/ (first touch only).

Usage: py scripts/add_sweep_btn_labels.py [pool ...]  (default: active pools)
"""
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

RAW = Path(r"D:\Project\ai game secretary\data\raw_images")
BACKUP = Path(r"D:\Project\ai game secretary\data\_sweepbtn_backup")
TPL = Path(r"D:\Project\ai game secretary\logs\_tpl_455_blue.png")
CLS = 455
BOX_W, BOX_H = 0.0635, 0.0320
THR = 0.85
SKIP = {"_synth_ui_swap", "_synth_bond", "_synth_bond_enter", "_synth_bond_goto",
        "_fused_synth_remap", "run_20260606_flywheel_labels_bak", "_v8queue_meta",
        "_emoticon_v2", "_battle_val"}


def process_pool(pool: str):
    import cv2
    tpl0 = cv2.imread(str(TPL))   # cut from a 3840x2160 frame
    tpl_cache = {3840: tpl0}
    pdir = RAW / pool
    added = 0
    scanned = 0
    for txt in sorted(pdir.glob("*.txt")):
        if txt.name == "classes.txt":
            continue
        jp = txt.with_suffix(".jpg")
        if not jp.exists():
            continue
        lines = [l for l in txt.read_text(encoding="utf-8").splitlines()
                 if l.strip()]
        if any(l.split()[0] == str(CLS) for l in lines):
            continue
        img = cv2.imread(str(jp))
        if img is None:
            continue
        h, w = img.shape[:2]
        if w not in tpl_cache:   # UI scales with width (16:9 layouts)
            s = w / 3840.0
            tpl_cache[w] = cv2.resize(tpl0, (max(8, int(tpl0.shape[1]*s)),
                                             max(8, int(tpl0.shape[0]*s))))
        tpl = tpl_cache[w]
        th, tw = tpl.shape[:2]
        scanned += 1
        win = img[int(0.72*h):int(0.92*h), int(0.30*w):int(0.55*w)]
        if win.shape[0] <= th or win.shape[1] <= tw:
            continue
        res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
        _mn, mx, _l, loc = cv2.minMaxLoc(res)
        if mx < THR:
            continue
        cx = (int(0.30*w) + loc[0] + tw/2) / w
        cy = (int(0.72*h) + loc[1] + th/2) / h
        bdir = BACKUP / pool
        bdir.mkdir(parents=True, exist_ok=True)
        bak = bdir / txt.name
        if not bak.exists():
            shutil.copy2(txt, bak)
        lines.append(f"{CLS} {cx:.6f} {cy:.6f} {BOX_W:.6f} {BOX_H:.6f}")
        txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
        added += 1
    return pool, scanned, added


def main():
    from concurrent.futures import ProcessPoolExecutor
    pools = sys.argv[1:] or [p.name for p in RAW.iterdir()
                             if p.is_dir() and p.name not in SKIP]
    pools = [p for p in pools if (RAW / p).is_dir()]
    total_added = total_scanned = 0
    with ProcessPoolExecutor(max_workers=16) as ex:
        for pool, scanned, added in ex.map(process_pool, pools):
            total_scanned += scanned
            total_added += added
            if added:
                print(f"{pool}: +{added}", flush=True)
    print(f"\n[done] scanned {total_scanned} no-455 frames, added {total_added}")


if __name__ == "__main__":
    main()
