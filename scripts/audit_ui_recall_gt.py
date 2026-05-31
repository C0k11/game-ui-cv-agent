"""Ground-truth per-cls recall audit on the ui_v1 LABELED dataset.

Complements audit_ui_reliability.py (which scans unlabeled trajectories and
can only measure detection FREQUENCY — a cls whose screen never appears in
the trajectories shows det=0 but may be perfectly learned, e.g.
学生momotalk信息未读 trained 4549× but momotalk screen was never captured).

This instead runs the active ui model against the dataset's OWN ground-truth
labels (which DO cover every screen type) and reports per-cls recall /
precision / mAP50. That separates:
  • recall ~0 with GT>0  → TRULY BROKEN / mislabeled (model can't find it even
                            on the frames it was trained on)
  • absent from this set  → just not labeled here (different gap)

Note: split='train' is the model's OWN training data, so recall here is the
UPPER BOUND ("did it even learn the cls"). split='val' = held-out generalization
(but the 51-frame val pool doesn't cover every cls). We run train by default
for full cls coverage; pass --split val for the held-out view.

Usage:
  py scripts/audit_ui_recall_gt.py                 # split=train, active ui model
  py scripts/audit_ui_recall_gt.py --split val
  py scripts/audit_ui_recall_gt.py --weights <pt> --conf 0.25 --iou 0.5
Output: D:\\Project\\ai game secretary\\data\\_ui_recall_gt.xlsx
"""
from __future__ import annotations
import sys, json
from pathlib import Path


def main():
    argv = sys.argv[1:]
    weights = None; split = "train"; conf = 0.25; iou = 0.5
    out = r"D:\Project\ai game secretary\data\_ui_recall_gt.xlsx"
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--weights": weights = argv[i+1]; i += 2; continue
        if a == "--split": split = argv[i+1]; i += 2; continue
        if a == "--conf": conf = float(argv[i+1]); i += 2; continue
        if a == "--iou": iou = float(argv[i+1]); i += 2; continue
        if a == "--out": out = argv[i+1]; i += 2; continue
        i += 1
    if not weights:
        reg = json.loads((Path(r"D:\Project\ai game secretary\data\model_registry.json")).read_text(encoding="utf-8"))
        ui = reg["ui"]; weights = ui["versions"][ui["active"]]["path"]
    data_yaml = r"D:\Project\ml_cache\models\yolo\dataset\ui_v1\data.yaml"
    from ultralytics import YOLO
    m = YOLO(weights)
    print(f"weights={weights}\nsplit={split} conf={conf} iou={iou}")
    metrics = m.val(data=data_yaml, split=split, imgsz=960, device=0,
                    conf=conf, iou=iou, plots=False, verbose=False)
    names = m.names
    def nm(c): return names.get(c, str(c)) if isinstance(names, dict) else names[c]
    box = metrics.box
    idxs = list(box.ap_class_index)   # cls ids that had GT in this split
    rows = []
    for j, cid in enumerate(idxs):
        cid = int(cid)
        r = float(box.r[j]); p = float(box.p[j]); ap = float(box.ap50[j])
        if r < 0.30:      flag = "BROKEN(recall<0.30)"
        elif r < 0.60:    flag = "WEAK(recall<0.60)"
        elif r < 0.85:    flag = "MEH"
        else:             flag = "ok"
        rows.append((cid, nm(cid), round(r, 3), round(p, 3), round(ap, 3), flag))
    rows.sort(key=lambda x: x[2])  # worst recall first
    print(f"\n=== per-cls recall on {split} ({len(idxs)} cls with GT) ===")
    print(f"{'cls':>4} {'name':<18} {'recall':>6} {'prec':>6} {'ap50':>6}  flag")
    for cid, name, r, p, ap, flag in rows:
        if flag.startswith(("BROKEN", "WEAK")):
            print(f"{cid:>4} {str(name)[:18]:<18} {r:>6} {p:>6} {ap:>6}  {flag}")
    from collections import Counter
    fc = Counter(x[5] for x in rows)
    print("\n--- flag counts ---")
    for k, v in sorted(fc.items()):
        print(f"  {v:>4}  {k}")
    # which trained cls have NO GT in this split (so we know coverage gaps)
    print(f"\n(cls with GT in {split}: {len(idxs)} / {len(names)} total)")
    header = ["cls", "name", "recall", "precision", "ap50", "flag"]
    try:
        from openpyxl import Workbook
        wb = Workbook(); ws = wb.active; ws.title = f"recall_{split}"
        ws.append(header)
        for row in rows: ws.append(list(row))
        wb.save(out); print(f"\n[xlsx] {out}")
    except ImportError:
        import csv
        co = out.replace(".xlsx", ".csv")
        with open(co, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f); w.writerow(header); w.writerows(rows)
        print(f"\n[csv] {co}")


if __name__ == "__main__":
    main()
