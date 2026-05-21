"""Help user clean noisy val labels by comparing GT vs model prediction.

Runs current best.pt on every val frame and outputs cases where:
  1. wrong-class: GT box matches a model prediction at IoU>=0.5 but different cls
  2. likely-miss: GT box has NO model prediction near it (real miss)
  3. extra-pred: model proposes a high-conf box where GT has nothing (potential
     GT under-labeling — user forgot to label this character)

For each case, generate a small report image (crop of the disputed box) so
the user can quickly review and fix in the dashboard Annotate page.

Usage:
    py scripts/val_label_cleanup.py
    py scripts/val_label_cleanup.py --conf 0.30  (only consider model preds >=conf)
"""
from __future__ import annotations
import argparse
import base64
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(r"D:\Project\ml_cache\models\yolo\runs\fused_avatar_yolo26x\weights\best.pt")
DATA_YAML = Path(r"D:\Project\ml_cache\models\yolo\dataset\fused_avatar_v1\data.yaml")
OUT_HTML = REPO / "_val_cleanup.html"


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1: return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / max(1e-6, union)


def crop_to_b64(img, xyxy, pad=20) -> str:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
    crop = img[y1:y2, x1:x2]
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.30)
    args = ap.parse_args()

    with open(DATA_YAML, encoding="utf-8") as f:
        ycfg = yaml.safe_load(f)
    names = ycfg["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names.keys())]

    data_root = Path(ycfg["path"])
    val_dir = data_root / "images" / "val"
    val_lbl_dir = data_root / "labels" / "val"

    print(f"Loading {MODEL_PATH.name}")
    model = YOLO(str(MODEL_PATH))

    wrong_cls_cases = []     # (img_path, gt_xyxy, gt_name, pred_xyxy, pred_name, pred_conf)
    extra_pred_cases = []    # (img_path, pred_xyxy, pred_name, pred_conf)
    miss_cases = []          # (img_path, gt_xyxy, gt_name)

    for img_path in sorted(val_dir.glob("*.jpg")):
        lbl_path = val_lbl_dir / (img_path.stem + ".txt")
        img = cv2.imread(str(img_path))
        if img is None: continue
        H, W = img.shape[:2]

        # GT
        truths = []
        if lbl_path.exists():
            for line in lbl_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) < 5: continue
                cls = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])
                x1 = (xc - w/2) * W; y1 = (yc - h/2) * H
                x2 = (xc + w/2) * W; y2 = (yc + h/2) * H
                truths.append((cls, (x1, y1, x2, y2)))

        # Predictions
        r = model.predict(source=str(img_path), conf=args.conf, imgsz=960, verbose=False)[0]
        preds = []
        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            for box, c, cf in zip(boxes, clss, confs):
                preds.append((int(c), tuple(box.tolist()), float(cf)))

        # Match: each GT to its best-IoU pred
        matched_pred_idx = set()
        for gt_cls, gt_box in truths:
            best_iou = 0; best_pi = -1
            for pi, (p_cls, p_box, p_cf) in enumerate(preds):
                if pi in matched_pred_idx: continue
                io = iou(gt_box, p_box)
                if io > best_iou:
                    best_iou = io; best_pi = pi
            if best_iou >= 0.5 and best_pi >= 0:
                matched_pred_idx.add(best_pi)
                p_cls, p_box, p_cf = preds[best_pi]
                if p_cls != gt_cls:
                    wrong_cls_cases.append((img_path, gt_box, names[gt_cls], p_box, names[p_cls], p_cf))
            else:
                # Real miss (no nearby pred)
                miss_cases.append((img_path, gt_box, names[gt_cls]))

        # Extra preds (high-conf preds with NO matching GT)
        for pi, (p_cls, p_box, p_cf) in enumerate(preds):
            if pi in matched_pred_idx: continue
            if p_cf < 0.40: continue  # only flag confident extras
            extra_pred_cases.append((img_path, p_box, names[p_cls], p_cf))

    # Build HTML
    html = ["<!DOCTYPE html><html><head><meta charset='utf-8'><title>Val Cleanup</title>"]
    html.append("<style>")
    html.append("body{background:#111;color:#eee;font-family:monospace;padding:20px}")
    html.append("h1{color:#facc15}")
    html.append("h2{color:#22c55e;border-bottom:1px solid #333;padding-bottom:5px}")
    html.append(".case{display:inline-block;margin:8px;padding:10px;background:#1a1a1f;border:1px solid #333;border-radius:5px;vertical-align:top;width:280px}")
    html.append(".case img{max-width:260px;display:block;margin:5px auto}")
    html.append(".case .lbl{font-size:11px;color:#aaa;margin:3px 0}")
    html.append(".case .gt{color:#ef4444}")
    html.append(".case .pred{color:#22c55e}")
    html.append(".case .conf{color:#facc15}")
    html.append("</style></head><body>")
    html.append(f"<h1>Val Label Cleanup — {MODEL_PATH.name}</h1>")
    html.append(f"<p>conf threshold: {args.conf} · model: {MODEL_PATH}</p>")

    # Section 1: wrong-class (most likely human error)
    html.append(f"<h2>🔴 Wrong-class disputes ({len(wrong_cls_cases)}) — model 跟 你 标的不一样, 多半 你 错</h2>")
    html.append("<p>对每张图: 看 <span class='pred'>绿框</span> (model 说) 跟 <span class='gt'>红框</span> (你标的) 哪个对. 通常 model 1.6% wrong-cls 里至少一半是 你 标错.</p>")
    for img_path, gt_box, gt_name, pred_box, pred_name, pred_conf in wrong_cls_cases:
        img = cv2.imread(str(img_path))
        b64 = crop_to_b64(img, gt_box)
        rel = img_path.name
        html.append(f"<div class='case'>")
        html.append(f"<img src='data:image/jpeg;base64,{b64}'/>")
        html.append(f"<div class='lbl'>📁 {rel}</div>")
        html.append(f"<div class='gt'>GT: {gt_name}</div>")
        html.append(f"<div class='pred'>Model: {pred_name} <span class='conf'>conf={pred_conf:.2f}</span></div>")
        html.append(f"</div>")

    # Section 2: extra-pred (might be user under-labeling)
    extra_pred_cases.sort(key=lambda x: -x[3])
    html.append(f"<h2>🟡 Extra predictions ({len(extra_pred_cases)}) — model 检出 你 没标, 多半 你 漏标了</h2>")
    html.append("<p>conf 高的优先看 — model 自信说 这有 X 角色, 但你 val 这位置没框. 可能是你忘标了.</p>")
    for img_path, pred_box, pred_name, pred_conf in extra_pred_cases[:50]:
        img = cv2.imread(str(img_path))
        b64 = crop_to_b64(img, pred_box)
        rel = img_path.name
        html.append(f"<div class='case'>")
        html.append(f"<img src='data:image/jpeg;base64,{b64}'/>")
        html.append(f"<div class='lbl'>📁 {rel}</div>")
        html.append(f"<div class='pred'>Model: {pred_name} <span class='conf'>conf={pred_conf:.2f}</span></div>")
        html.append(f"<div class='gt'>GT: (没标)</div>")
        html.append(f"</div>")

    # Section 3: misses (real model failures)
    html.append(f"<h2>🔵 True misses ({len(miss_cases)}) — 你 标了, model 完全看不到</h2>")
    html.append("<p>这种确实是 model 没检出 (model 的真实弱点). 但抽样看几个 — 也有可能 你 标错位置/标错角色, 实际框里没那角色.</p>")
    for img_path, gt_box, gt_name in miss_cases[:50]:
        img = cv2.imread(str(img_path))
        b64 = crop_to_b64(img, gt_box)
        rel = img_path.name
        html.append(f"<div class='case'>")
        html.append(f"<img src='data:image/jpeg;base64,{b64}'/>")
        html.append(f"<div class='lbl'>📁 {rel}</div>")
        html.append(f"<div class='gt'>GT: {gt_name}</div>")
        html.append(f"<div class='pred'>Model: (没检出)</div>")
        html.append(f"</div>")

    html.append("</body></html>")
    OUT_HTML.write_text("\n".join(html), encoding="utf-8")

    print(f"\n=== Summary ===")
    print(f"  Wrong-class (人 vs 模型 分类不一致): {len(wrong_cls_cases)}")
    print(f"  Extra preds  (model 多检, 可能 人 漏标): {len(extra_pred_cases)}")
    print(f"  True misses  (model 没检出 — 真弱点 / 也可能 人 标错): {len(miss_cases)}")
    print(f"\nReport: file:///{str(OUT_HTML).replace(chr(92), '/')}")
    print(f"Open in browser to review case-by-case.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
