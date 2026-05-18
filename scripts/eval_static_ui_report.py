"""Human eval for static_ui detection: render truth vs prediction overlay HTML.

For each frame in the val split (or any source dir of labeled .jpg):
  - Load truth bboxes from .txt
  - Run model inference
  - Overlay BOTH on the image (truth = green, pred = red w/ class name)
  - Compute IoU matches → categorize:
      ✓ MATCHED  (truth bbox has overlapping pred with same class, IoU>0.5)
      ✗ MISS     (truth bbox has no matching pred → false negative)
      ⚠ FP       (pred bbox with no matching truth → false positive)
      ⚠ WRONG    (matched bbox but wrong class)
  - Render summary stats per frame

Output: HTML with full frames + per-frame stats. Human scrolls visually.

Usage:
    py scripts/eval_static_ui_report.py                          # uses v3 model, val split
    py scripts/eval_static_ui_report.py --model v2 --frames 20
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
MODELS = {
    "v2": Path(r"D:\Project\ml_cache\models\yolo\runs\static_ui_yolo26n\weights\best.pt"),
    "v3": Path(r"D:\Project\ml_cache\models\yolo\runs\static_ui_v3_yolo26n\weights\best.pt"),
    "v4": Path(r"D:\Project\ml_cache\models\yolo\runs\static_ui_v4_yolo26n\weights\best.pt"),
}
DATA_YAML = Path(r"D:\Project\ml_cache\models\yolo\dataset\static_ui_v1\data.yaml")


def img_to_b64(img: np.ndarray, max_w: int = 1000, quality: int = 75) -> str:
    h, w = img.shape[:2]
    if w > max_w:
        scale = max_w / w
        img = cv2.resize(img, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


def iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    """IoU of two boxes (x1,y1,x2,y2)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / max(1e-6, area_a + area_b - inter)


def load_truth_boxes(label_path: Path, img_w: int, img_h: int, names: List[str]) -> List[Dict]:
    """Read YOLO label file, return list of {cls_idx, name, xyxy}."""
    if not label_path.exists():
        return []
    out = []
    try:
        text = label_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = label_path.read_text(encoding="utf-16")
        except Exception:
            return []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5 or not parts[0].lstrip("-").isdigit():
            continue
        try:
            ci = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
        except ValueError:
            continue
        if not (0 <= ci < len(names)):
            continue
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        out.append({"cls_idx": ci, "name": names[ci], "xyxy": (x1, y1, x2, y2)})
    return out


def match_truth_to_pred(truths: List[Dict], preds: List[Dict], iou_thr: float = 0.5) -> Tuple[List[Tuple[int,int]], List[int], List[int], List[Tuple[int,int]]]:
    """Greedy IoU match.  Returns:
      matches:    [(truth_idx, pred_idx)] same class, IoU >= iou_thr
      misses:     [truth_idx] no pred match
      fps:        [pred_idx] no truth match
      wrong_cls:  [(truth_idx, pred_idx)] IoU OK but class mismatch
    """
    used_pred = set()
    matches, misses, wrong_cls = [], [], []
    for ti, t in enumerate(truths):
        # Find best IoU pred
        best_p = -1
        best_iou = 0.0
        for pi, p in enumerate(preds):
            if pi in used_pred:
                continue
            v = iou(t["xyxy"], p["xyxy"])
            if v > best_iou:
                best_iou = v
                best_p = pi
        if best_iou >= iou_thr and best_p >= 0:
            used_pred.add(best_p)
            if preds[best_p]["name"] == t["name"]:
                matches.append((ti, best_p))
            else:
                wrong_cls.append((ti, best_p))
        else:
            misses.append(ti)
    fps = [pi for pi in range(len(preds)) if pi not in used_pred]
    return matches, misses, fps, wrong_cls


def draw_overlay(img: np.ndarray, truths: List[Dict], preds: List[Dict],
                 matches, misses, fps, wrong_cls) -> np.ndarray:
    """Draw truth (green) + pred (red) bboxes with status colors."""
    out = img.copy()
    # Mark truths
    matched_truth_idx = {ti for ti, _ in matches}
    wrong_truth_idx = {ti for ti, _ in wrong_cls}
    for ti, t in enumerate(truths):
        x1, y1, x2, y2 = map(int, t["xyxy"])
        if ti in matched_truth_idx:
            color = (0, 255, 0)  # green = correct
            lw = 1
        elif ti in wrong_truth_idx:
            color = (0, 165, 255)  # orange = wrong class
            lw = 2
        else:
            color = (0, 0, 255)  # red = miss
            lw = 3
        cv2.rectangle(out, (x1, y1), (x2, y2), color, lw)

    # Mark FP preds (no matching truth)
    for pi in fps:
        p = preds[pi]
        x1, y1, x2, y2 = map(int, p["xyxy"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 255), 2)  # magenta = FP
        label = f"FP {p['name']} {p['conf']:.2f}"
        cv2.putText(out, label, (x1, max(y1-3, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    # Mark wrong-class preds
    for _, pi in wrong_cls:
        p = preds[pi]
        x1, y1, x2, y2 = map(int, p["xyxy"])
        cv2.putText(out, f"got: {p['name']}", (x1, max(y1-3, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v2", choices=list(MODELS.keys()))
    ap.add_argument("--data", default=str(DATA_YAML))
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--frames", type=int, default=None, help="Cap frames to render (default: all)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model_path = MODELS[args.model]
    if not model_path.is_file():
        print(f"[err] model not found: {model_path}")
        return 1

    print(f"Loading {args.model} model: {model_path.name}")
    model = YOLO(str(model_path))
    cfg = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    names = [cfg["names"][i] for i in sorted(cfg["names"].keys())]
    print(f"classes: {len(names)}")

    img_dir = Path(cfg["path"]) / "images" / args.split
    lbl_dir = Path(cfg["path"]) / "labels" / args.split
    images = sorted(img_dir.glob("*.jpg"))
    if args.frames:
        images = images[:args.frames]
    print(f"frames: {len(images)} from {args.split} split")

    out_path = Path(args.out) if args.out else (REPO / "data" / "yolo_datasets" / f"static_ui_eval_{args.model}.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Aggregate counters across all frames
    tot_truth = tot_match = tot_miss = tot_fp = tot_wrong = 0
    per_class_truth = defaultdict(int)
    per_class_match = defaultdict(int)
    frame_blocks = []

    for fi, img_path in enumerate(images):
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]

        # Truth
        truths = load_truth_boxes(lbl_dir / (img_path.stem + ".txt"), w, h, names)

        # Predictions
        det = model(img, conf=args.conf, verbose=False)[0]
        preds = []
        for b in det.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            preds.append({
                "cls_idx": int(b.cls[0]),
                "name": names[int(b.cls[0])],
                "xyxy": (x1, y1, x2, y2),
                "conf": float(b.conf[0]),
            })

        # Match
        matches, misses, fps, wrong = match_truth_to_pred(truths, preds, args.iou_thr)
        tot_truth += len(truths)
        tot_match += len(matches)
        tot_miss += len(misses)
        tot_fp += len(fps)
        tot_wrong += len(wrong)
        for t in truths:
            per_class_truth[t["name"]] += 1
        for ti, _ in matches:
            per_class_match[truths[ti]["name"]] += 1

        # Overlay
        annotated = draw_overlay(img, truths, preds, matches, misses, fps, wrong)
        ann_b64 = img_to_b64(annotated, max_w=1100)
        miss_names = sorted({truths[ti]["name"] for ti in misses})
        fp_list = [f'{preds[pi]["name"]}({preds[pi]["conf"]:.2f})' for pi in fps]
        wrong_list = [f'{truths[ti]["name"]}→{preds[pi]["name"]}({preds[pi]["conf"]:.2f})' for ti, pi in wrong]
        frame_blocks.append({
            "name": img_path.name,
            "img_b64": ann_b64,
            "n_truth": len(truths),
            "n_match": len(matches),
            "n_miss": len(misses),
            "n_fp": len(fps),
            "n_wrong": len(wrong),
            "miss_names": miss_names,
            "fp_list": fp_list,
            "wrong_list": wrong_list,
        })

    # Recall per frame, sort worst first
    frame_blocks.sort(key=lambda b: (b["n_match"] / max(1, b["n_truth"])))

    # Build HTML
    overall_recall = tot_match / max(1, tot_truth)
    fp_rate = tot_fp / max(1, tot_match + tot_fp)
    html = [f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>static_ui {args.model} eval</title>
<style>
body {{ font-family: system-ui, sans-serif; background:#0f1115; color:#e0e0e0; padding:20px; max-width:1400px; margin:0 auto }}
h1, h2 {{ color:#fff }}
h2 {{ margin-top:30px; padding:6px 10px; background:#1a1d24; border-radius:6px; font-size:15px }}
.summary {{ display:flex; gap:16px; flex-wrap:wrap; margin:15px 0 }}
.summary div {{ padding:10px 16px; background:#1a1d24; border-radius:6px; font-family:monospace }}
.legend {{ background:#1a1d24; padding:12px 16px; border-radius:6px; margin:10px 0; line-height:1.8 }}
.legend span {{ font-family:monospace; padding:1px 6px; border-radius:3px; margin-right:8px }}
.frame-img {{ display:block; max-width:100%; border-radius:6px; margin:6px 0 }}
.stats {{ font-family:monospace; font-size:13px; color:#bbb; line-height:1.6 }}
.miss {{ color:#fca5a5 }}
.fp {{ color:#ee82ee }}
.wrong {{ color:#fb923c }}
.good {{ color:#86efac }}
</style></head><body>
<h1>static_ui {args.model} 模型检测 - 人工审核</h1>
<div class="legend">
<span style="background:#22c55e;color:#000">绿 = 正确</span>
<span style="background:#fb923c;color:#000">橙 = 类错（IoU 对但类别错）</span>
<span style="background:#ef4444;color:#fff">红 = 漏检（truth 有但 pred 没有）</span>
<span style="background:#d946ef;color:#fff">紫 = 误检 (pred 但无 truth)</span>
</div>
<div class="summary">
<div><b>整体</b> truth={tot_truth} | match={tot_match} | miss={tot_miss} | fp={tot_fp} | wrong-cls={tot_wrong}</div>
<div><b>Recall</b> {100*overall_recall:.2f}% (match / truth)</div>
<div><b>FP rate</b> {100*fp_rate:.2f}% (fp / (match+fp))</div>
<div><b>Frames</b> {len(images)}</div>
</div>
<p style="color:#888">按"召回率从低到高"排序（最糟糕的帧放最上面）</p>
"""]
    for fb in frame_blocks:
        recall = fb['n_match'] / max(1, fb['n_truth'])
        html.append(f'<h2>{fb["name"]} — recall {100*recall:.0f}% (match {fb["n_match"]}/{fb["n_truth"]}, miss {fb["n_miss"]}, fp {fb["n_fp"]}, wrong-cls {fb["n_wrong"]})</h2>')
        html.append(f'<img class="frame-img" src="data:image/jpeg;base64,{fb["img_b64"]}">')
        stats = []
        if fb["miss_names"]:
            stats.append(f'<div class="stats miss">漏检 ({fb["n_miss"]}): {", ".join(fb["miss_names"])}</div>')
        if fb["wrong_list"]:
            stats.append(f'<div class="stats wrong">类错: {", ".join(fb["wrong_list"][:15])}</div>')
        if fb["fp_list"]:
            stats.append(f'<div class="stats fp">误检 (top 10): {", ".join(fb["fp_list"][:10])}</div>')
        if not stats:
            stats.append('<div class="stats good">完美 ✓</div>')
        html.extend(stats)

    # Per-class table
    html.append('<h2>Per-class recall</h2>')
    html.append('<table style="border-collapse:collapse"><tr><th style="text-align:left;padding:4px 12px">class</th><th style="padding:4px 12px">truth</th><th style="padding:4px 12px">matched</th><th style="padding:4px 12px">recall</th></tr>')
    for cls in sorted(per_class_truth.keys(), key=lambda c: per_class_match[c] / max(1, per_class_truth[c])):
        t = per_class_truth[cls]
        m = per_class_match[cls]
        rec = m / max(1, t)
        bgcolor = '#22c55e' if rec >= 0.7 else '#fbbf24' if rec >= 0.4 else '#ef4444'
        html.append(f'<tr><td style="padding:4px 12px">{cls}</td><td style="padding:4px 12px;text-align:right">{t}</td><td style="padding:4px 12px;text-align:right">{m}</td><td style="padding:4px 12px;background:{bgcolor};color:#000;text-align:right">{100*rec:.0f}%</td></tr>')
    html.append('</table></body></html>')

    out_path.write_text("".join(html), encoding="utf-8")
    print(f"\nREPORT: {out_path}")
    print(f"Open: file:///{out_path.as_posix()}")
    print()
    print(f"Overall recall: {100*overall_recall:.2f}%  ({tot_match}/{tot_truth})")
    print(f"FP rate: {100*fp_rate:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
