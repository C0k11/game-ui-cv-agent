# -*- coding: utf-8 -*-
"""Per-class val comparison v9 vs v10 on the current ui_v2 val set.
Focus: dot classes (红点/黄点) precision (position-prior FP proxy) + new
arena_shop classes 469-473 recall. Run: py scripts/val_compare_v9_v10.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

from ultralytics import YOLO

DATA = r"D:\Project\ml_cache\models\yolo\dataset\ui_v2\data.yaml"
MODELS = {
    "v9":  r"D:\Project\ml_cache\models\yolo\runs\ui_yolo26m_v9\weights\best.pt",
    "v10": r"D:\Project\ml_cache\models\yolo\runs\ui_yolo26m_v10\weights\best.pt",
}
FOCUS = ["红点", "黄点", "绿勾", "战术大赛商店", "战术大赛商店已选择",
         "战术大赛商店货币", "下级能量饮料", "一般能量饮料"]


def run(tag, path):
    m = YOLO(path)
    r = m.val(data=DATA, imgsz=960, conf=0.001, iou=0.6, plots=False,
              verbose=False, split="val")
    b = r.box
    # per-class arrays are indexed by position in ap_class_index
    idx2pos = {int(c): i for i, c in enumerate(b.ap_class_index)}
    names = m.names
    per = {}
    for cname in FOCUS:
        ci = [k for k, v in names.items() if v == cname]
        if not ci:
            per[cname] = None
            continue
        ci = ci[0]
        if ci in idx2pos:
            i = idx2pos[ci]
            per[cname] = (float(b.p[i]), float(b.r[i]),
                          float(b.ap50[i]), float(b.ap[i]))
        else:
            per[cname] = "absent"   # no GT in val OR no preds
    return {
        "mAP50": float(b.map50), "mAP50-95": float(b.map),
        "mP": float(b.mp), "mR": float(b.mr), "per": per,
    }


def main():
    res = {}
    for tag, p in MODELS.items():
        print(f"\n===== val {tag} =====", flush=True)
        res[tag] = run(tag, p)

    print("\n\n================ SUMMARY ================")
    print(f"{'metric':<22}{'v9':>12}{'v10':>12}{'Δ':>10}")
    for k in ["mAP50", "mAP50-95", "mP", "mR"]:
        a, bb = res["v9"][k], res["v10"][k]
        print(f"{k:<22}{a:>12.4f}{bb:>12.4f}{bb-a:>+10.4f}")

    def fmt(x):
        if x is None:
            return "no-class"
        if x == "absent":
            return "no-GT/pred"
        return f"{x[0]:.3f}/{x[1]:.3f}/{x[2]:.3f}"

    print(f"\n{'class':<16}{'v9 P/R/AP50':>26}{'v10 P/R/AP50':>26}")
    for c in FOCUS:
        a, bb = res["v9"]["per"][c], res["v10"]["per"][c]
        print(f"{c:<16}{fmt(a):>26}{fmt(bb):>26}")


if __name__ == "__main__":
    main()
