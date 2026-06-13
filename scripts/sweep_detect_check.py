# -*- coding: utf-8 -*-
"""Did v10 fix '普通关卡选中(80) screen → 批量扫荡(455) not detected'?
Run v9 & v10 on flywheel _clean frames, report per-class detect rate for the
farming-screen cls, and specifically: on frames showing 普通关卡选中(80),
how often is 批量扫荡(455) found? Run: py scripts/sweep_detect_check.py
"""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

RAW = Path(r"D:\Project\ai game secretary\data\raw_images")
V9 = r"D:\Project\ml_cache\models\yolo\runs\ui_yolo26m_v9\weights\best.pt"
V10 = r"D:\Project\ml_cache\models\yolo\runs\ui_yolo26m_v10\weights\best.pt"
WATCH = {80: "普通关卡选中", 420: "普通关卡", 455: "批量扫荡",
         456: "批量扫荡开始", 457: "批量扫荡开始灰", 108: "扫荡开始",
         111: "MAX_可点击", 117: "MAX_灰色"}


def scan(model, frames, names):
    name2idx = {v: k for k, v in names.items()}
    # master-name -> local idx
    want = {}
    for mi, nm in WATCH.items():
        master = [l.strip() for l in open(RAW / "_classes.txt", encoding="utf-8")
                  if l.strip()]
        if nm in name2idx:
            want[mi] = name2idx[nm]
    per = {mi: [] for mi in WATCH}       # mi -> list of (frame, conf)
    has80 = []                            # frames where 80 detected
    sweep_on_80 = []                      # frames with 80 AND 455
    B = 24
    for s in range(0, len(frames), B):
        chunk = frames[s:s + B]
        res = model.predict(chunk, conf=0.20, imgsz=960, verbose=False)
        for p, r in zip(chunk, res):
            local2conf = {}
            for bx in r.boxes:
                c = int(bx.cls[0])
                cf = float(bx.conf[0])
                local2conf[c] = max(local2conf.get(c, 0), cf)
            for mi, loc in want.items():
                if loc in local2conf:
                    per[mi].append((p, local2conf[loc]))
            l80 = want.get(80)
            l455 = want.get(455)
            if l80 is not None and l80 in local2conf:
                has80.append(p)
                if l455 is not None and l455 in local2conf:
                    sweep_on_80.append((p, local2conf[l455]))
    return per, has80, sweep_on_80


def main():
    pools = sorted(p for p in RAW.iterdir()
                   if p.is_dir() and p.name.endswith("_clean")
                   and ("20260612" in p.name or "20260613" in p.name))
    frames = []
    for pool in pools:
        frames += [str(j) for j in sorted(pool.glob("*.jpg"))]
    print(f"{len(frames)} flywheel frames", flush=True)

    from ultralytics import YOLO
    out = {}
    for tag, path in [("v9", V9), ("v10", V10)]:
        print(f"scanning {tag} …", flush=True)
        m = YOLO(path)
        out[tag] = scan(m, frames, m.names)

    print("\n========== farming-screen cls detect counts ==========")
    print(f"{'cls':<22}{'v9':>8}{'v10':>8}")
    for mi, nm in WATCH.items():
        a = len(out["v9"][0][mi])
        b = len(out["v10"][0][mi])
        print(f"{mi} {nm:<16}{a:>8}{b:>8}")

    for tag in ("v9", "v10"):
        per, has80, sweep80 = out[tag]
        n80 = len(has80)
        ns = len(sweep80)
        rate = ns / n80 if n80 else 0
        print(f"\n[{tag}] 普通关卡选中(80) frames: {n80}  |  of those, "
              f"批量扫荡(455) also found: {ns} ({rate:.0%})")
        if sweep80[:5]:
            print(f"      sample confs: "
                  f"{', '.join(f'{c:.2f}' for _, c in sweep80[:8])}")


if __name__ == "__main__":
    main()
